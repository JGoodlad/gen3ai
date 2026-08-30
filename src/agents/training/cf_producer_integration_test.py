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


def test_a_new_checkpoint_mid_run_restamps_the_labels_and_the_buffer_takes_both(tmp_path):
    """THE MULTI-CYCLE seam: a checkpoint lands between cycles, and the labels move with it.

    The single-cycle test above holds the snapshot fixed, so the whole refresh leg — resolve the
    freshest checkpoint, reload it, stamp ITS step on the rows, and have the consumer hold rows of
    two different ages at once — was uncovered by anything that runs a real producer. A live R1 arm
    does that on every checkpoint for the length of the run.

    The second checkpoint is deliberately a **FORCED** one (`checkpoint_forced_<step>_<HHMMSS>.zip`,
    what SIGUSR1 / the launcher's `c` key writes). That name used not to parse as a step, which made
    every periodic checkpoint outrank it and walked the producer BACKWARDS onto an older snapshot —
    found by the R1 composition smoke, 2026-08-23. `cf_producer_test` pins the resolver directly;
    this pins that the label path as a whole moves forward.
    """
    run = _mk_run(tmp_path)
    ring_a, _ = _play_and_ring(run)
    ring_b, _ = _play_and_ring(run)
    assert ring_a != ring_b
    step2 = STEP + 40_000
    loaded: "list[str]" = []

    def _loader(path, step):
        loaded.append(os.path.basename(path))
        return _RandomPolicySnapshot(path, step or STEP)

    argv = [run, "--rollouts", "2", "--top-n", "1", "--records-per-cycle", "1",
            "--cycles", "1", "--cycle-seconds", "0", "--impl", IMPL]
    assert P.main(list(argv), snapshot_loader=_loader) == 0

    # The checkpoint the trainer would write on a forced save, plus the pointer it updates.
    ck2 = os.path.join(run, "checkpoints", f"checkpoint_forced_{step2:010d}_120000.zip")
    open(ck2, "w").write("x")
    with open(os.path.join(run, "latest.txt"), "w") as f:
        f.write(os.path.relpath(ck2, run))
    assert P.resolve_latest_checkpoint(run)[1] == step2, (
        "the resolver passed over a NEWER forced checkpoint — the producer would keep labelling "
        "against a snapshot it has already moved past")

    assert P.main(list(argv), snapshot_loader=_loader) == 0
    assert len(loaded) == 2 and loaded[1] == os.path.basename(ck2), (
        f"the second cycle did not reload the new checkpoint: {loaded}")

    labels_dir = os.path.join(run, P.LABELS_DIRNAME)
    files = sorted(f for f in os.listdir(labels_dir) if f.endswith(".jsonl"))
    rows = [json.loads(ln) for f in files for ln in open(os.path.join(labels_dir, f)) if ln.strip()]
    stamps = sorted({r["policy_step"] for r in rows})
    assert stamps == [STEP, step2], f"labels were not re-stamped across the refresh: {stamps}"
    assert {r["battle"] for r in rows} == {ring_a, ring_b}, "the two cycles took the same record"
    assert any(f.startswith(f"labels_cf_producer_{step2}_") for f in files)

    # -- the CONSUMER holds both vintages, and their ages differ by the refresh ------
    buf = CfLabelBuffer(labels_dir, obs_dim=_obs_dim(), lag_bound=150_000)
    assert buf.poll(step2) == len(rows)
    assert buf.skipped_total == 0, f"the consumer refused the producer's rows: {buf.skip_reasons}"
    assert buf.expired_total == 0 and buf.future_total == 0 and buf.replaced_total == 0
    ages = sorted({step2 - row.policy_step for row in buf.sample(len(rows))})
    assert ages == [0, step2 - STEP], (
        f"the buffer collapsed two vintages into one age: {ages} — the older rows are what the "
        f"staleness bound exists to expire")

    # -- the GIGO refusal, beside GOOD rows, through the same real buffer ------------
    # `labels_skipped_total` must PARTITION the input with `labels_ingested_total`: one poisoned
    # row costs exactly itself, never the file and never the run.
    poisoned = dict(rows[0])
    poisoned["obs_sha1"] = "0" * 40                     # disagrees with its own bytes
    poisoned["decision_idx"] = int(poisoned["decision_idx"]) + 10_000   # a fresh dedup identity
    with open(os.path.join(labels_dir, "labels_zz_poisoned_0_0.jsonl"), "w") as f:
        f.write(json.dumps(poisoned) + "\n")
    before = len(buf)
    assert buf.poll(step2) == 0, "a row whose digest disagrees with its bytes was ACCEPTED"
    assert buf.skipped_total == 1, f"the poisoned row was not counted: {buf.skip_reasons}"
    assert len(buf) == before, "a poisoned row evicted a good one"
    assert buf.stats(step2)["cf/labels_skipped_total"] == 1.0


def test_a_rollout_that_reaches_the_TURN_CAP_is_a_draw_on_either_seat(tmp_path):
    """`gen3_cf_draw_at_cap_v1`, on REAL battles — the half a stubbed `_run_one` cannot prove.

    THE DEFECT. Both sides of a producer rollout stall-forfeit at `MAX_TURNS`, so a continuation
    that reaches the cap has BOTH sides forfeit and the recorded winner is decided by which
    ``FORCELOSE`` the sim processes first. Measured here — and, when this was root-caused, over 16
    lines on `node` and `rust` alike — the ordering is not even random: **p1 always loses**. A
    training record always seats the trainee on p1 (`_trainee_side`), so before the fix EVERY
    capped rollout scored a hard 0 and the tight-MC P(win) labels of stall-shaped positions were
    biased DOWNWARD, silently.

    The two seats are the assertion. The same board, the same dice, played from either side, must
    produce the same label — and only a draw-at-cap score does that.

    REPRODUCIBLE, not wall-clock-seeded (root `CLAUDE.md`'s fuzz-script-vs-collected-test rule):
    the record comes from `record_fixture_battle(key=…)` and the cap is FORCED down to a handful
    of turns rather than waiting 250, so the battle reaches it by construction on every run.

    REVERT-VERIFIED: with `rollout_outcome_score`'s ``capped`` branch removed, the two seats score
    0.0 and 1.0 and the equality assertion fails.
    """
    import dataclasses

    from agents.training.stall import StallConfig
    from agents.training.obs_roundtrip_fuzz_test import record_fixture_battle
    from utils.bridge.counterfactual import replay_counterfactual

    cap = 6
    raw, _summary, _npz = record_fixture_battle(str(tmp_path), key=3, tag="Cap", impl=IMPL)

    scores, flags = {}, {}
    for side in ("p1", "p2"):
        other = "p2" if side == "p1" else "p1"
        rec = dataclasses.replace(raw, trainee_username=raw.username(side))

        def _mk(role, seat):
            return _Gen3RandomPlayer(
                rng_seed=77, observation_encoder=None, mappings=None,
                battle_format=rec.format_id, team=rec.packed_team(seat),
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(f"Cap{role}{side}", "pw"),
                stall_config=StallConfig(threshold=cap),
                max_concurrent_battles=1, start_listening=False)

        res = replay_counterfactual(
            rec, trainee=_mk("T", side), opponent=_mk("O", other),
            divergence_turn=3, seed=rec.start_options().get("seed"), impl=IMPL)
        assert res["turns"] >= cap and res["ended"], (
            f"the {cap}-turn cap was not reached ({res['turns']} turns) — this test proves "
            f"nothing unless the battle actually caps")
        flags[side] = res["capped"]
        scores[side] = P.rollout_outcome_score(res)

    assert flags == {"p1": True, "p2": True}, (
        f"the runner did not flag a capped battle as capped: {flags}")
    assert scores["p1"] == scores["p2"] == 0.5, (
        f"a capped rollout must be a DRAW on either seat, got {scores} — the label is being "
        f"decided by which side's forfeit the sim processed first, not by the position")


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


# ---------------------------------------------------------------------------
# The PER-ACTION stream, composed the same way (gen3_cf_q_labels_v1)
# ---------------------------------------------------------------------------

def test_the_PER_ACTION_stream_composes_ring_to_buffer(tmp_path):
    """`--q-labels` end to end: real record → real choice map → real paired rollouts → the REAL
    `CfLabelBuffer`'s per-action columns.

    The v107 Q head landed as a trained consumer of a stream nothing wrote, so the failure this
    guards against is the one that already happened once in this gap: two halves each green on
    their own, neither ever run against the other's real output. Everything here is real except the
    policy — the choice strings come from the live mapper at a replayed decision, the sibling arms
    are real bridge battles, and the pairing assertion is live inside the sweep.
    """
    run = _mk_run(tmp_path)
    _ring_path, _prefix = _play_and_ring(run)

    rc = P.main([run, "--rollouts", "2", "--top-n", "1", "--records-per-cycle", "1",
                 "--cycles", "1", "--cycle-seconds", "0", "--impl", IMPL,
                 "--q-labels", "--q-top-n", "1", "--q-max-actions", "3"],
                snapshot_loader=lambda path, step: _RandomPolicySnapshot(path, step or STEP))
    assert rc == 0

    rows = []
    for name in sorted(os.listdir(os.path.join(run, P.LABELS_DIRNAME))):
        with open(os.path.join(run, P.LABELS_DIRNAME, name)) as f:
            rows += [json.loads(ln) for ln in f if ln.strip()]
    assert rows, "the producer wrote no labels at all"
    swept = [r for r in rows if r.get("q_labels")]
    assert swept, "no row carried a per-action block — the supply side did not run"

    # -- the wire shape, as written by the REAL producer ---------------------------
    for r in swept:
        assert r["schema"] == 1, "the sweep must never bump the schema — it is a REFUSAL gate"
        assert r["taken_action"] == r["recorded_action"]
        assert r["q_sweep"]["version"] == "cf_q_sweep_v1"
        assert 1 <= len(r["q_labels"]) <= 3, "--q-max-actions 3 must bound the arms"
        seen = set()
        for e in r["q_labels"]:
            assert set(e) == {"action", "label", "n_rollouts"}
            assert 0 <= e["action"] < 11 and 0.0 <= e["label"] <= 1.0 and e["n_rollouts"] > 0
            assert e["action"] not in seen, "one entry per action"
            seen.add(e["action"])
        assert r["taken_action"] in seen, "the recorded action anchors every sweep"
        # THE FREE-ARM IDENTITY: at --q-rollouts == --rollouts the recorded action's q-label is
        # the row's own tight-MC label, on the same dice. Not an approximation.
        rec = next(e for e in r["q_labels"] if e["action"] == r["taken_action"])
        assert rec["label"] == pytest.approx(r["label"]), (
            "the recorded arm was reused, so its q-label must EQUAL the row's own label")
        assert rec["n_rollouts"] == r["n_rollouts"]

    # -- and the REAL consumer reads it -------------------------------------------
    buf = CfLabelBuffer(os.path.join(run, P.LABELS_DIRNAME), obs_dim=_obs_dim(), lag_bound=0)
    assert buf.poll(STEP) == len(rows)
    assert buf.skipped_total == 0 and buf.field_skipped_total == 0
    stats = buf.stats(STEP)
    assert stats["cf/q_label_coverage"] > 0.0, (
        "the launch-window counter that separates a live factory from a dead one")
    assert stats["cf/q_labels_per_row"] > 1.0, (
        "at most one label per row is the ON-POLICY TRICKLE the Q head exists to escape")

    from agents.training.cf_label_buffer import batch_tensors
    resident = [r for r in buf.sample(len(buf)) if r.q_labels]
    b = batch_tensors(resident, "cpu")
    for i, row in enumerate(resident):
        for action, value, n in row.q_labels:
            assert b.q_mask[i, action].item() == 1.0
            assert b.q_label[i, action].item() == pytest.approx(value, abs=1e-6)
            assert b.q_n[i, action].item() == n
        assert int(b.q_mask[i].sum().item()) == len(row.q_labels), \
            "an action nobody swept must stay masked OFF, not inherit a neighbour's column"


def test_scan_record_recovers_the_FULL_choice_map_at_every_decision(tmp_path):
    """`capture_choices=True` must agree with the recorded choice it already inverts.

    The sweep substitutes a sibling action's choice STRING into the sim, so a wrong map is a
    silently mislabelled action rather than an error. The check that cannot be fooled is the
    overlap with the one entry already known independently: the map's value at the RECOVERED action
    index must be the string the side actually committed.
    """
    from agents.training.obs_materializer import scan_record

    run = _mk_run(tmp_path)
    ring_path, _prefix = _play_and_ring(run)
    record = ReconstructionRecord.load(ring_path)

    plain = scan_record(record, "p1", impl=IMPL, encode=False)
    mapped = scan_record(record, "p1", impl=IMPL, encode=False, capture_choices=True)

    assert [d.action for d in plain] == [d.action for d in mapped], \
        "capturing the map must not change the replay it rides on"
    assert all(d.choices is None for d in plain), "OFF must leave the field absent"
    assert mapped and all(d.choices for d in mapped)
    for d in mapped:
        assert d.choices[d.action] == d.choice, (
            f"decision {d.index}: the map's entry for the recovered action "
            f"{d.choices.get(d.action)!r} disagrees with the committed choice {d.choice!r}")
        assert set(d.choices) <= set(int(i) for i in np.flatnonzero(np.asarray(d.mask))), \
            "the map must offer nothing the mask does not"
