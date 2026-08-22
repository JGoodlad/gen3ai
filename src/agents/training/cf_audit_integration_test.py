"""cf_audit end to end on a REAL bridge battle it plays itself — the fuzz-test pattern.

Records one real gen3ou battle in-process (no server), assembles it into the ``eval_traces``
layout a run writes, then runs ``cf_audit.main`` over it at R=2: frame → stratified sample →
anchor arm → tight-MC labels (REAL rollouts through the REAL bridge) → bias map → label rows.

What is real and what is substituted, stated plainly: the battle, the reconstruction record,
the scripted-prefix divergence, the rollouts and every outcome are real. The POLICY is not —
a current-architecture checkpoint is not something a test can conjure, so the session stub
plays ``RandomPlayer`` on both sides through the same ``utils.bridge.counterfactual`` driver
``prober.replay`` uses. That substitutes exactly one thing (which net picks the move) and
leaves the whole audit pipeline under test.

The recorded ``win_probs`` are synthesised too, for the same reason: the fuzz recorder has no
model, so it writes no win-prob head output. They are seeded and spread across the deciles so
the stratifier has something to stratify.

    pytest -m sim src/agents/training/cf_audit_integration_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import numpy as np
import pytest

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.training import cf_audit
from agents.training.obs_materializer import materialize_from_record
from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
from utils.bridge.counterfactual import replay_counterfactual as _run_one
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

pytestmark = pytest.mark.sim

STEP = 1000
OPPONENT = "random"


class _RandomPolicySession:
    """A ProbeSession stand-in: same ``replay_counterfactual`` contract, real bridge rollouts,
    a RandomPlayer where the trained policy would be."""

    def __init__(self, traces: str, **_kw):
        self.traces = traces
        self._pool = TeamLoader().get_all_teams()
        self.calls = 0

    def _player(self, record, side, tag):
        return RandomPlayer(
            battle_format=record.format_id, team=record.packed_team(side),
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration(tag, "pw"),
            max_concurrent_battles=1, start_listening=False)

    def replay_counterfactual(self, summary_path, inv, action, *, n_rollouts=1,
                              opponent_ckpt=None, **_kw):
        base = summary_path[: -len("_summary.json")]
        record = ReconstructionRecord.load(base + "_reconstruction.json")
        with np.load(base + "_states.npz") as z:
            actions = np.asarray(z["actions"], dtype=int)
        with open(summary_path) as f:
            summary = json.load(f)
        turn = int(summary["invocations"][inv]["turn"])
        trace = materialize_from_record(record, actions=actions, map_actions_at=inv,
                                        stop_after_decision=inv)
        choice = (trace.action_choices or {})[int(action)]
        side = record.side_of(record.trainee_username)
        seed = record.start_options().get("seed")
        from main.prober.falsifier import fresh_seeds
        post = ([None] if n_rollouts <= 1
                else list(fresh_seeds(n_rollouts, salt=f"{record.battle_tag}:{inv}")))
        outcomes = []
        for k, ps in enumerate(post):
            self.calls += 1
            res = _run_one(record, trainee=self._player(record, side, f"CfT{self.calls % 9999}"),
                           opponent=self._player(record, "p2" if side == "p1" else "p1",
                                                 f"CfO{self.calls % 9999}"),
                           divergence_turn=turn, substitute_choice=choice, seed=seed,
                           post_t_seed=ps)
            outcomes.append(res["outcome"])
        n = len(outcomes)
        wins = outcomes.count("win")
        return {"wins": wins, "losses": outcomes.count("loss"), "n_rollouts": n,
                "win_rate": wins / n if n else None,
                "win_rate_ci": list(cf_audit.wilson_ci(wins, n)),
                "outcomes": {o: outcomes.count(o) for o in sorted(set(outcomes))},
                "opponent_source": "random_policy_stub",
                "recorded_result": (summary.get("meta") or {}).get("result"),
                "substitute": {"choice": choice}, "chosen": {"choice": choice}}


def _build_run_dir(root: str) -> str:
    """Play one real battle and lay it out the way a run's eval_traces tree is."""
    trace_dir = os.path.join(root, "eval_traces", f"step_{STEP}", OPPONENT)
    os.makedirs(trace_dir, exist_ok=True)
    ts = int(time.time() * 1000) % 100000
    pool = TeamLoader().get_all_teams()
    trainee = RecordingFuzzPlayer(
        out_dir=trace_dir, rng_seed=ts, battle_format="gen3ou", team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"CAt{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    opp = RandomPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"CAo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(trainee, opp, 1))
    prefix = trainee.trace_prefixes[0]

    # The fuzz recorder has no model, so it records no win-prob head output. Synthesise a
    # SPREAD of confidences (seeded) so the stratifier has deciles to stratify over.
    npz_path = f"{prefix}_states.npz"
    with np.load(npz_path) as z:
        arrays = {k: z[k] for k in z.files}
    n = len(arrays["actions"])
    rng = np.random.default_rng(11)
    arrays["win_probs"] = rng.uniform(0.05, 0.98, size=n).astype(np.float32)
    np.savez_compressed(npz_path, **arrays)
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump({"latest_eval": {}}, f)
    return root


def test_cf_audit_runs_end_to_end_on_a_real_battle(tmp_path):
    run_dir = _build_run_dir(str(tmp_path / "run"))
    out = str(tmp_path / "out")
    session = {}

    def factory(traces, **kw):
        session["s"] = _RandomPolicySession(traces, **kw)
        return session["s"]

    rc = cf_audit.main(
        [run_dir, "--rollouts", "2", "--states", "4", "--anchors", "2",
         "--anchor-tolerance", "0.0", "--out", out, "--seed", "3"],
        session_factory=factory)
    assert rc == 0, "the audit must complete on a battle it played itself"
    assert session["s"].calls > 0, "no rollout was actually run — the bridge leg is untested"

    # (1) the bias map exists, is JSON, and carries its sampler design + accounting
    with open(os.path.join(out, "bias_map.json")) as f:
        bm = json.load(f)
    assert bm["n_rollouts"] == 2
    assert bm["sampler"]["sampler_version"] and bm["sampler"]["seed"] == 3
    acc = bm["accounting"]
    assert acc["frame_decisions"] > 0 and acc["labelled"] > 0
    assert acc["rollouts"] == 2 * acc["labelled"]
    assert acc["label_trust_passed"] is True
    # turn 1 is a KNOWN coverage gap and must be COUNTED, never silently dropped
    assert acc["skipped"].get("turn_1_unopenable", 0) >= 1

    # (2) the human-readable summary names the meter
    with open(os.path.join(out, "bias_map.md")) as f:
        md = f.read()
    assert "sd_true_excess" in md and "Accounting" in md and "Caveats" in md

    # (3) label rows are the shared v1 schema, and their obs_sha1 identifies the ROW they
    #     claim — a consumer must be able to verify what it loaded.
    (lp,) = [os.path.join(out, "cf_labels", f) for f in os.listdir(os.path.join(out, "cf_labels"))]
    rows = [json.loads(ln) for ln in open(lp) if ln.strip()]
    assert len(rows) == acc["labelled"] >= 1
    for r in rows:
        assert r["schema"] == 1 and r["kind"] == "mc_winprob"
        assert r["policy_step"] == STEP and r["opponent"] == OPPONENT
        assert 0.0 <= r["label"] <= 1.0
        assert r["wilson_lo"] <= r["label"] <= r["wilson_hi"]
        assert r["n_rollouts"] == 2
        assert r["battle"].endswith("_reconstruction.json") and os.path.exists(r["battle"])
        path, key = r["obs_npz"].split("::")
        with np.load(path) as z:
            obs = np.asarray(z[key][r["decision_idx"]], dtype=np.float32)
        assert cf_audit.obs_digest(obs) == r["obs_sha1"]


def test_cf_audit_refuses_to_emit_labels_when_the_anchor_arm_fails(tmp_path):
    """LABEL TRUST BEFORE MAP TRUST. If the recorded action on the recorded dice does not
    reproduce the recorded battle, the replay is not exact and everything downstream is GIGO
    — so the run must exit non-zero and write NO labels."""
    run_dir = _build_run_dir(str(tmp_path / "run"))
    out = str(tmp_path / "out")

    class _Liar(_RandomPolicySession):
        def replay_counterfactual(self, summary_path, inv, action, *, n_rollouts=1, **kw):
            res = super().replay_counterfactual(summary_path, inv, action,
                                                n_rollouts=n_rollouts, **kw)
            if n_rollouts == 1:      # the anchor arm — report an outcome that cannot match
                res["wins"] = res["losses"] = 0
                res["win_rate"] = 0.0
            return res

    rc = cf_audit.main(
        [run_dir, "--rollouts", "2", "--states", "2", "--anchors", "2",
         "--anchor-tolerance", "0.9", "--out", out, "--seed", "3"],
        session_factory=lambda traces, **kw: _Liar(traces, **kw))
    assert rc == 3, "a failed anchor arm must be a non-zero exit, not a warning"
    assert not os.path.isdir(os.path.join(out, "cf_labels")), "labels were emitted anyway"
    with open(os.path.join(out, "bias_map.json")) as f:
        assert json.load(f)["accounting"]["label_trust_passed"] is False
