"""THE COMPOSITION TEST — the label path end to end, both halves, on a battle it plays itself.

    pytest -m sim src/agents/training/cf_producer_integration_test.py

Producer and consumer are separate processes that share only a file format, and this tree has now
shipped **two** contract bugs into exactly that gap — the buffer keyed its byte offsets on filename
alone (a recreated file silently dropped rows), and the buffer's `obs_npz` path ignored
`decision_idx`, which made `cf_audit`'s DEFAULT output 100% unconsumable while both halves' own
unit tests were green. The method lesson from the second one is the reason this file exists:
*both halves were tested, and neither test ever ran the other half's real output.*

So this runs the REAL composition, in order:

1. play a REAL gen3ou battle in-process through the bridge (no server);
2. ring its reconstruction record through the REAL `CfRecordRing`, in the shape the TRAINING tap
   writes — i.e. with **no `trainee_username`**, which is the fact that makes a training record
   different from an eval sibling and the one a producer must survive;
3. run ONE REAL `cf_producer` cycle over that ring — real anchor, real replay, real materialized
   obs, real bridge rollouts;
4. feed the label files it wrote to the REAL `CfLabelBuffer` and assert every row is INGESTED with
   ZERO skips, digests verifying, sane values and the right `policy_step`.

What is real and what is substituted, stated plainly: the battle, the record, the ring, the
scan, the anchor, the divergence, the rollouts, the label files and the buffer are all real. The
POLICY is not — a current-architecture checkpoint is not something a test can conjure — so the
`snapshot_loader` seam returns a stub that plays `RandomPlayer` on both sides and scores with fixed
numbers. That substitutes exactly one thing (which net picks the move, and which net's win-prob
head ranks the decisions) and leaves the whole label path under test. It is the same single
substitution `cf_audit_integration_test` makes.

The driver is **node** here rather than the production `rust` default, deliberately: a fresh
worktree's first rust-backed test pays for a `cargo build --release` that saturates every core and
turns contention-scaled timeouts into a wall of TIMEOUTs (recorded twice, both times misread as a
rust defect). The two impls' equivalence is `bridge_impl_parity_test`'s job, not this file's.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time

import numpy as np
import pytest

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.gen3_battle import Gen3Battle
from agents.inference.player import Gen3Player
from agents.training import cf_producer as P
from agents.training.cf_label_buffer import CfLabelBuffer
from agents.training.cf_records import CfRecordRing
from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

pytestmark = pytest.mark.sim

IMPL = "node"
STEP = 1_234_000
#: How many times `_play_and_ring` will redraw a battle that hit the 250-turn draw cap.
_MAX_DRAWS = 4


# ---------------------------------------------------------------------------
# The one substitution: a Snapshot whose policy is a RandomPlayer
# ---------------------------------------------------------------------------

class _Gen3RandomPlayer(Gen3Player):
    """A seeded-random policy behind the REAL Gen3 decision path.

    Deliberately a `Gen3Player` rather than poke-env's `RandomPlayer` (`gen3_cf_twin_heads_v1`).
    The producer's `mc_return` stream is measured through the player's own obs/order seams
    (`agents/training/cf_mc_return.attach_return_recording`), which a poke-env baseline does not
    have — so with a `RandomPlayer` the shadow critic's half of the label path would be structurally
    unreachable from this test and the composition would go uncovered. This double keeps the ONE
    substitution the file's header declares (which policy picks the move) while restoring
    everything the label path actually reads.
    """

    def __init__(self, *, rng_seed: int, **kwargs):
        super().__init__(battle_class=Gen3Battle, **kwargs)
        self._rng = np.random.RandomState(rng_seed)

    def _predict_best_action(self, battle, **_kw):
        """`RLPlayer`'s seam, with dice where the net would be — returns ``(idx, probs, mask)``.

        Present because `cf_mc_return.attach_return_recording` records through it and
        `choose_move`, NOT through `action_to_order` (which `counterfactual._invert_choice` calls
        once per legal action on every scripted decision). A double that lacked it would make the
        shadow critic's whole label path unreachable from this test."""
        mask = np.asarray(self.embed_battle(battle)["action_mask"])
        if int(mask.sum()) == 0:
            return None, None, mask
        return int(self._rng.choice(np.flatnonzero(mask))), None, mask

    def choose_move(self, battle):
        forfeit = self._handle_stall(battle, "CFP_STALL")
        if forfeit:
            return forfeit
        idx, _probs, _mask = self._predict_best_action(battle)
        if idx is None:
            return self.choose_default_move()
        return self.action_to_order(idx, battle)


class _RandomPolicySnapshot:
    """A `cf_producer.Snapshot` stand-in — same two jobs, a random policy where the net would be.

    `score` returns a deterministic SPREAD of win-probabilities so the priority sampler has
    something to rank (a constant would make the top-N an arbitrary tie-break and prove nothing
    about the ordering)."""

    def __init__(self, path: str, step: int) -> None:
        self.path = path
        self.step = int(step)
        self.mappings = None
        self.scored: "list[int]" = []
        self.players_built = 0

    def score(self, obs, masks):
        n = len(obs)
        self.scored.append(n)
        rng = np.random.default_rng(7)
        return rng.uniform(0.05, 0.95, size=n), rng.uniform(0.1, 0.9, size=n)

    def make_player(self, record, side, *, role):
        self.players_built += 1
        return _Gen3RandomPlayer(
            rng_seed=1000 + self.players_built,
            observation_encoder=None, mappings=None,
            battle_format=record.format_id, team=record.packed_team(side),
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration(
                f"CfP{role}{self.players_built % 99999}", "pw"),
            max_concurrent_battles=1, start_listening=False)


# ---------------------------------------------------------------------------
# Fixture: a real battle → a real ring record in the TRAINING tap's shape
# ---------------------------------------------------------------------------

def _play_and_ring(run_dir: str) -> "tuple[str, str]":
    """Play one real DECIDED battle; ring its record the way `--cf-records` does. Returns
    ``(ring_path, trace_prefix)`` — the trace prefix is kept only so the test can cross-check the
    producer's materialized obs against the ones the LIVE player actually saw.

    REDRAWS on a draw (the 250-turn cap). A drawn game is perfectly labelable — the producer scores
    a tie 0.5 rather than a loss — but it is 250 turns long, and every rollout replays that prefix.
    Redrawing keeps this test's cost bounded. (The same wall-clock-seed draw is a genuine flake in
    `cf_audit_integration_test`, where a draw empties the frame outright; see its `_build_run_dir`.)
    """
    traces = os.path.join(run_dir, "_trace")
    os.makedirs(traces, exist_ok=True)
    pool = TeamLoader().get_all_teams()
    trainee = prefix = None
    for draw in range(_MAX_DRAWS):
        ts = (int(time.time() * 1000) + draw) % 100000
        trainee = RecordingFuzzPlayer(
            out_dir=traces, rng_seed=ts, battle_format="gen3ou", team=Gen3Teambuilder(pool),
            account_configuration=AccountConfiguration(f"CPt{ts}", "pw"),
            server_configuration=LocalhostServerConfiguration,
            start_listening=False, max_concurrent_battles=1)
        opp = RandomPlayer(
            battle_format="gen3ou", team=Gen3Teambuilder(pool),
            account_configuration=AccountConfiguration(f"CPo{ts}", "pw"),
            server_configuration=LocalhostServerConfiguration,
            start_listening=False, max_concurrent_battles=1)
        asyncio.run(run_local_battles(trainee, opp, 1))
        cand = trainee.trace_prefixes[0]
        with open(cand + "_summary.json") as f:
            result = ((json.load(f).get("meta") or {}).get("result") or "").lower()
        if result in ("win", "loss"):
            prefix = cand
            break
    assert prefix is not None, f"{_MAX_DRAWS} battles in a row ended in a draw"

    raw = json.loads(open(prefix + "_reconstruction.json").read())
    tag = raw.get("battle_tag")
    # THE PRODUCTION SHAPE. The eval path merges `trainee_username` into its artifact; the
    # TRAINING tap writes the bare `__RECON__` payload, which names no trainee at all. Stripping it
    # is what makes this a training record rather than an eval one — and it is the fact
    # `_trainee_side` (BridgeSession seats agent1 on p1) exists to answer.
    raw.pop("trainee_username", None)
    ring = CfRecordRing(os.path.join(run_dir, P.RECORDS_DIRNAME), keep=8)
    path = ring.write_record(tag, raw)
    assert path is not None, "the ring refused a record it was handed"
    assert "trainee_username" not in json.loads(path.read_text())
    # The producer will read this side; assert the transport invariant it relies on holds here.
    rec = ReconstructionRecord.load(str(path))
    assert rec.side_of(trainee.username) == "p1", (
        "run_local_battles seats its first player on p1 — if that changed, `_trainee_side`'s "
        "default is wrong and every training label would be the OPPONENT's view")
    return str(path), prefix


def _mk_run(tmp_path) -> str:
    run = os.path.join(str(tmp_path), "run")
    os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
    # A checkpoint file that only has to EXIST: resolution is production code, loading is stubbed.
    open(os.path.join(run, "checkpoints", f"checkpoint_{STEP}_steps.zip"), "w").write("x")
    with open(os.path.join(run, "latest.txt"), "w") as f:
        f.write(f"checkpoints/checkpoint_{STEP}_steps.zip")
    return run


def _obs_dim() -> int:
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    return int(Gen3ObservationEncoder(load_mappings()).dimension)


# ---------------------------------------------------------------------------
# THE test
# ---------------------------------------------------------------------------

def test_the_whole_label_path_composes_ring_to_buffer(tmp_path):
    run = _mk_run(tmp_path)
    ring_path, prefix = _play_and_ring(run)
    snaps: "list[_RandomPolicySnapshot]" = []

    def _loader(path, step):
        s = _RandomPolicySnapshot(path, step or STEP)
        snaps.append(s)
        return s

    rc = P.main([run, "--rollouts", "2", "--top-n", "2", "--records-per-cycle", "1",
                 "--cycles", "1", "--cycle-seconds", "0", "--impl", IMPL],
                snapshot_loader=_loader)
    assert rc == 0, "one cycle over one real record must complete"
    assert snaps and snaps[0].scored, "no decision was ever scored — the ranking leg is untested"
    assert snaps[0].players_built > 0, "no player was built — the rollout leg is untested"

    # -- (1) the producer wrote label files, and its state file says so -------------
    labels_dir = os.path.join(run, P.LABELS_DIRNAME)
    files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".jsonl"))
    assert files, "the cycle produced no label file"
    assert files[0].startswith(f"labels_cf_producer_{STEP}_")
    rows = [json.loads(ln) for f in files for ln in open(os.path.join(labels_dir, f)) if ln.strip()]
    assert 1 <= len(rows) <= 2, f"top-n 2 must cap the batch, got {len(rows)}"

    state = json.loads(open(os.path.join(run, P.STATE_FILENAME)).read())
    assert state["labels_total"] == len(rows)
    assert state["rollouts_total"] == 2 * len(rows)
    assert state["anchors_run"] == 1 and state["anchors_reproduced"] == 1, (
        "the startup anchor must have RUN and reproduced — a scripted full replay has no dice to "
        "blame")
    assert state["records_processed"] == 1
    assert state["sampler_version"] == P.SAMPLER_VERSION
    assert os.path.basename(ring_path) in state["processed"]

    # -- (2) the rows are the shared v1 schema, and they name their ecology ---------
    for r in rows:
        assert r["schema"] == 1 and r["kind"] == "mc_winprob"
        assert r["policy_step"] == STEP, "the label must be stamped with the snapshot's step"
        assert r["opponent"] == "self_current", (
            "a training record names no opponent; a row that claims one is a value claim with a "
            "population it cannot support")
        assert r["n_rollouts"] == 2
        assert 0.0 <= r["label"] <= 1.0
        assert r["wilson_lo"] <= r["label"] <= r["wilson_hi"]
        assert r["obs_inline"] and r["obs_npz"] is None, (
            "a ring record has no states.npz, so the obs MUST travel inline")
        assert r["battle"] == ring_path
        assert r["sampler_version"] == P.SAMPLER_VERSION
        assert r["turn"] >= P.MIN_LABELABLE_TURN

    # -- (3) THE CROSS-CHECK: the obs the producer materialized is the obs the LIVE
    #        player actually saw. `scan_record` recovers the action history by inverting
    #        recorded choices rather than reading them, so this is the one assertion that
    #        proves the recovered history did not desync the encoder's trackers.
    with np.load(prefix + "_states.npz") as z:
        live_obs = np.asarray(z["obs"], dtype=np.float32)
    for r in rows:
        got = np.frombuffer(base64.b64decode(r["obs_inline"]), dtype=np.float32)
        want = live_obs[r["decision_idx"]]
        assert np.array_equal(got, want), (
            f"decision {r['decision_idx']}: the producer's replayed obs differs from the one the "
            f"live player encoded — the inverted action history desynced the trackers")

    # -- (4) THE CONSUMER. The real buffer, at the real obs width, at the real step. -
    buf = CfLabelBuffer(labels_dir, obs_dim=_obs_dim(), lag_bound=150_000)
    accepted = buf.poll(STEP)
    assert accepted == len(rows), f"buffer accepted {accepted} of {len(rows)} rows"
    assert buf.skipped_total == 0, (
        f"the consumer REJECTED rows the producer wrote: {buf.skip_reasons}")
    assert buf.expired_total == 0 and buf.future_total == 0
    assert len(buf) == len(rows)
    stats = buf.stats(STEP)
    assert stats["cf/labels_ingested_total"] == len(rows)
    assert stats["cf/label_age_steps_p50"] == 0.0, "age must be 0 at the stamping step"

    # The rows the buffer resolved are the rows the producer measured — digest-verified end to end
    # (the buffer recomputes `obs_sha1` from the bytes it loaded, never copies the declared one).
    by_digest = {row.obs_sha1: row for row in buf.sample(len(rows))}
    for r in rows:
        assert r["obs_sha1"] in by_digest
        assert by_digest[r["obs_sha1"]].label == pytest.approx(r["label"])
        assert by_digest[r["obs_sha1"]].policy_step == STEP
        assert by_digest[r["obs_sha1"]].obs.shape == (_obs_dim(),)

    # -- (4b) THE TWIN STREAMS, through the SAME real buffer (gen3_cf_twin_heads_v1) ----
    # The both-halves composition rule: a producer-side unit test alone is exactly how the last
    # two contract bugs shipped, and these two fields are consumed by heads that produce no shape
    # error when starved. `outcome_label` must be present and in range on EVERY row (it is free —
    # the producer already computed it for the critic-surprise term), and `mc_return` must be
    # present with its reward digest, since the driver defaults `--mc-return` ON.
    for r in rows:
        assert r["outcome_label"] in (0.0, 0.5, 1.0), (
            f"decision {r['decision_idx']}: outcome_label {r.get('outcome_label')!r} is not a "
            f"win/loss/tie scalar — head B would be trained on it")
        assert r.get("reward_sha1"), "an mc_return row must name the reward it was measured under"
    assert any(r.get("mc_return") is not None for r in rows), (
        "the producer shipped no mc_return at all — the shadow critic would train on nothing while "
        "every other counter read healthy")
    for row in buf.sample(len(rows)):
        assert row.outcome_label is not None, (
            "the buffer dropped the outcome_label the producer wrote — head B would silently "
            "become a copy of head A and C-B would silently become C-A")
    cov = buf.stats(STEP)
    assert cov["cf/outcome_label_coverage"] == pytest.approx(1.0)
    assert cov["cf/mc_return_coverage"] > 0.0
    assert cov["cf/labels_mc_return_rejected_total"] == 0.0

    # And the GIGO guard from the other side: a buffer configured with a DIFFERENT reward digest
    # must refuse the same rows' mc_return while keeping their win-prob labels.
    foreign = CfLabelBuffer(labels_dir, obs_dim=_obs_dim(), lag_bound=150_000,
                            reward_sha1="a-different-reward")
    assert foreign.poll(STEP) == len(rows)
    assert foreign.mc_return_rejected_total > 0
    assert all(row.mc_return is None for row in foreign.sample(len(rows)))
    assert foreign.skipped_total == 0, "a reward mismatch must not cost the row its win-prob label"

    # -- (5) a SECOND poll must be a no-op (the incremental reader's offsets) -------
    assert buf.poll(STEP) == 0
    assert buf.replaced_total == 0, "the producer double-shipped a state within one cycle"

    # -- (6) a second producer CYCLE over the unchanged ring produces nothing new ---
    rc2 = P.main([run, "--rollouts", "2", "--top-n", "2", "--records-per-cycle", "1",
                  "--cycles", "1", "--cycle-seconds", "0", "--impl", IMPL],
                 snapshot_loader=_loader)
    assert rc2 == 0
    again = sorted(f for f in os.listdir(labels_dir) if f.endswith(".jsonl"))
    assert again == files, "a restarted producer re-labelled a record it had already done"


def test_a_record_the_replay_cannot_reproduce_refuses_to_produce(tmp_path):
    """LABEL TRUST BEFORE LABELS, the `cf_audit` rule inherited. If the scripted full replay does
    not reproduce the recorded outcome, the replay is not exact and every label after it measures
    that bug — so the producer must exit 3 having written NO labels."""
    run = _mk_run(tmp_path)
    _play_and_ring(run)

    class _Liar(_RandomPolicySnapshot):
        pass

    def _loader(path, step):
        return _Liar(path, step or STEP)

    import agents.training.cf_producer as mod
    real = mod._run_one

    def _flip(record, **kw):
        out = real(record, **kw)
        if kw.get("divergence_turn") is None:      # the ANCHOR arm — report the wrong winner
            out = dict(out)
            out["outcome"] = "tie" if out["outcome"] != "tie" else "win"
        return out

    mod._run_one = _flip
    try:
        rc = mod.main([run, "--rollouts", "2", "--top-n", "1", "--cycles", "1",
                       "--cycle-seconds", "0", "--impl", IMPL], snapshot_loader=_loader)
    finally:
        mod._run_one = real
    assert rc == 3, "a failed anchor must be a non-zero exit, not a warning"
    assert not os.path.isdir(os.path.join(run, P.LABELS_DIRNAME)), "labels were emitted anyway"
    state = json.loads(open(os.path.join(run, P.STATE_FILENAME)).read())
    assert state["anchors_run"] == 1 and state["anchors_reproduced"] == 0
    assert state["records_processed"] == 0
