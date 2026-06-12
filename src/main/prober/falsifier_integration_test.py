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
            for alt in d["alternatives"]:
                assert alt["choice"] and alt["action"] != d["chosen"]["action"]
                assert 0.0 <= alt["refused_frac"] <= 1.0

        # Determinism: the same falsification reproduces exactly (same seeds,
        # same replay) — byte-stable JSON modulo nothing.
        again = falsify_decision(record, summary, npz, anchors[0],
                                 n_seeds=_SEEDS, n_alts=2)
        first = next(d for d in out["decisions"] if d["inv"] == anchors[0])
        assert again == first


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
