"""End-to-end falsifier integration: record a REAL bridge battle, then attribute
its decisions (fix-both luck percentile + paired alternative sweep) through the
actual reconstruction → re-roll → materializer pipeline. Needs the Node bridge;
no server."""

import asyncio
import os
import tempfile
import time

import numpy as np
import pytest

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
from main.prober.falsifier import falsify_battle, falsify_decision, select_anchors
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

_SEEDS = 8   # small but enough for distribution structure; keep the test fast


def _record_one_battle(out_dir: str):
    """Play one real battle with the round-trip recording harness; return
    (record, summary, npz)."""
    ts = int(time.time()) % 100000
    pool = TeamLoader().get_all_teams()
    trainee = RecordingFuzzPlayer(
        out_dir=out_dir, rng_seed=ts,
        battle_format="gen3ou", team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"FZz{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1,
    )
    opp = RandomPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"FZo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1,
    )
    asyncio.run(run_local_battles(trainee, opp, 1))
    prefix = trainee.trace_prefixes[0]
    record = ReconstructionRecord.load(f"{prefix}_reconstruction.json")
    import json
    with open(f"{prefix}_summary.json") as f:
        summary = json.load(f)
    with np.load(f"{prefix}_states.npz") as z:
        npz = {k: z[k] for k in z.files}
    return record, summary, npz


@pytest.mark.integration
def test_falsify_battle_end_to_end():
    with tempfile.TemporaryDirectory(prefix="falsifier_it_") as out_dir:
        record, summary, npz = _record_one_battle(out_dir)
        # RecordingFuzzPlayer captures real values? No model — values are 0.0
        # placeholders, so δ ranking degenerates; anchor explicitly instead on
        # the first few move decisions (the machinery under test is the re-roll
        # + attribution pipeline, not the ranking).
        invs = summary["invocations"]
        anchors = [i for i, inv in enumerate(invs)
                   if inv.get("phase") == "move_selection"][1:3]  # turns 2..3-ish
        assert anchors, "battle too short to anchor"

        out = falsify_battle(record, summary, npz, invs=anchors,
                             n_seeds=_SEEDS, n_alts=2)

        assert out["battle"] == record.battle_tag
        assert out["trainee_side"] == "p1"
        assert not out["errors"], f"falsification errors: {out['errors']}"
        assert len(out["decisions"]) == len(anchors)

        for d in out["decisions"]:
            # The realized line exists and is a finite margin.
            assert isinstance(d["realized_margin"], float)
            # The dice distribution was actually sampled.
            assert d["dice_distribution"]["n"] > 0
            assert d["luck_percentile"] is None or 0.0 <= d["luck_percentile"] <= 1.0
            # The chosen action mapped through the real mapper to a sim choice.
            assert d["chosen"]["choice"]
            assert d["verdict"] in ("LUCK", "MISTAKE", "MIXED", "NEUTRAL")
            # The crater-magnitude weight the run-level scan keys on is surfaced
            # (a float when this anchor had a scorable δ, else None).
            assert "anchor_delta" in d
            assert d["anchor_delta"] is None or isinstance(d["anchor_delta"], float)
            for alt in d["alternatives"]:
                assert alt["choice"] and alt["action"] != d["chosen"]["action"]
                assert 0.0 <= alt["refused_frac"] <= 1.0

        # Determinism: the same falsification reproduces exactly (same seeds,
        # same replay) — byte-stable JSON modulo nothing.
        again = falsify_decision(record, summary, npz, anchors[0],
                                 n_seeds=_SEEDS, n_alts=2)
        first = next(d for d in out["decisions"] if d["inv"] == anchors[0])
        # falsify_battle annotates each decision with the battle-level `anchor_delta`
        # (the δ that ranks + weights it); falsify_decision returns the un-annotated
        # analysis. Modulo that annotation the two are byte-identical (determinism).
        assert {k: v for k, v in first.items() if k != "anchor_delta"} == again


@pytest.mark.integration
def test_select_anchors_on_real_trace_shape():
    """select_anchors runs on a real summary/npz shape (placeholder values ⇒
    δ = r only — still a valid ranking exercise over real reward totals)."""
    with tempfile.TemporaryDirectory(prefix="falsifier_it2_") as out_dir:
        _, summary, npz = _record_one_battle(out_dir)
        anchors = select_anchors(summary, npz, gamma=0.99, worst=3)
        assert anchors
        for a in anchors:
            assert summary["invocations"][a]["phase"] == "move_selection"


def _build_recon_tree(run_dir: str, n: int) -> str:
    """Record ``n`` real bridge battles and lay each triple into a discoverable
    ``eval_traces/step_/<opp>/loss_NNN_*`` tree (the filename drives discovery's
    outcome label). Returns the run dir for ProbeSession."""
    import json

    traces = os.path.join(run_dir, "eval_traces", "step_1000", "RandomPlayer")
    os.makedirs(traces, exist_ok=True)
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump({"gamma": 0.99}, f)
    for i in range(n):
        with tempfile.TemporaryDirectory(prefix="rec_") as d:
            record, summary, npz = _record_one_battle(d)
        base = os.path.join(traces, f"loss_{i:03d}")
        record.save(base + "_reconstruction.json")
        with open(base + "_summary.json", "w") as f:
            json.dump(summary, f)
        np.savez(base + "_states.npz", **npz)
    return run_dir


@pytest.mark.integration
def test_falsify_scan_end_to_end(tmp_path):
    """The RUN-LEVEL scan over a real discoverable tree: every matching loss is
    falsified through the actual re-roll pipeline and aggregated into the
    reducible-vs-aleatoric crater partition. (Placeholder values ⇒ δ from reward
    only, so the verdict mix is degenerate — we assert the PLUMBING + accounting,
    not specific shares; the meaningful number comes from real eval traces.)"""
    from main.prober.session import ProbeSession

    root = _build_recon_tree(str(tmp_path / "run"), 2)
    out = ProbeSession(root).falsify_scan(outcome="loss", worst=2, n_seeds=6, n_alts=1)

    assert out["coverage"]["n_matched"] == 2
    assert out["coverage"]["n_with_record"] == 2
    assert out["coverage"]["n_falsified"] == 2
    assert out["coverage"]["n_skipped_no_record"] == 0
    assert out["coverage"]["n_decisions"] >= 1
    assert set(out["verdict_counts"]) == {"LUCK", "MISTAKE", "MIXED", "NEUTRAL"}
    assert 0.0 <= out["gate"]["critic_headroom_upper_bound"] <= 1.0
    if out["coverage"]["n_decisions"] > 0:           # count_shares always sum to 1 with decisions
        assert abs(sum(out["count_shares"].values()) - 1.0) < 1e-3
    # Concurrency must not change the aggregate (each battle is independent).
    par = ProbeSession(root).falsify_scan(outcome="loss", worst=2, n_seeds=6,
                                          n_alts=1, concurrency=2)
    assert par["verdict_counts"] == out["verdict_counts"]
