"""End-to-end better-line SEARCH: record a REAL bridge battle, then search for a better line from an
anchored move decision with a deterministic FAKE model, exercising the full
SearchSession-clone → materialize → V pipeline (depth 1 AND depth 2, the interior opponent).

The decisive gate is MODEL-FREE: the fake model's V = obs.sum(), so the depth-1 CHOSEN action's value
MUST equal the sum of the recorded ``states.npz`` next obs (its ``recorded_exact`` clone reproduces the
real next state bit-for-bit — the value_crn anchor). Depth 2 additionally checks the beam produces a
principal variation of the right length through the (self-model) interior opponent.

Needs the Node bridge; no server. Run directly or via ``pytest -m integration``."""

import asyncio
import json
import tempfile
import time

import numpy as np
import pytest

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
from main.prober.better_line import better_line_decision
from main.prober.engine import _has_state
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


class _SumModel:
    """V(s) = obs.sum() (a deterministic function of the obs, so a faithful materialization is
    provable); a deterministic non-uniform policy so interior top-k differs per state."""

    def value(self, obs, mask):
        return float(np.asarray(obs, dtype=np.float64).sum())

    def values_batch(self, obs, masks):
        return np.asarray(obs, dtype=np.float64).reshape(len(obs), -1).sum(1)

    def _probs(self, obs, mask):
        x = np.asarray(obs, dtype=np.float64)
        m = np.asarray(mask, dtype=float)
        sc = np.array([x[(i * 97) % len(x)] for i in range(len(m))])
        sc = np.where(m > 0, sc, -1e9)
        p = np.exp(sc - sc.max())
        return p / p.sum()

    def action_dist(self, obs, mask):
        p = self._probs(obs, mask)
        return p, np.log(p + 1e-12)

    def action_probs_batch(self, obs, masks):
        return np.stack([self._probs(o, m) for o, m in zip(np.asarray(obs), np.asarray(masks))])

    def win_prob_at(self, obs, mask):
        return None

    def value_dist_at(self, obs, mask):
        return None


def _record_one_battle(out_dir: str):
    ts = int(time.time() * 1000) % 100000
    pool = TeamLoader().get_all_teams()
    trainee = RecordingFuzzPlayer(
        out_dir=out_dir, rng_seed=ts, battle_format="gen3ou", team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"BLt{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False, max_concurrent_battles=1)
    opp = RandomPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"BLo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(trainee, opp, 1))
    prefix = trainee.trace_prefixes[0]
    record = ReconstructionRecord.load(f"{prefix}_reconstruction.json")
    with open(f"{prefix}_summary.json") as f:
        summary = json.load(f)
    with np.load(f"{prefix}_states.npz") as z:
        npz = {k: z[k] for k in z.files}
    return record, summary, npz


def _mid_anchor(summary, npz):
    invs = summary["invocations"]
    cand = [i for i, inv in enumerate(invs)
            if inv.get("phase") == "move_selection" and i + 1 < len(invs)
            and _has_state(npz, i) and _has_state(npz, i + 1)]
    return cand[len(cand) // 2] if cand else None


@pytest.mark.integration
def test_better_line_depth1_chosen_value_matches_recorded_next_obs():
    with tempfile.TemporaryDirectory(prefix="better_line_d1_") as out_dir:
        record, summary, npz = _record_one_battle(out_dir)
        anchor = _mid_anchor(summary, npz)
        assert anchor is not None
        out = better_line_decision(_SumModel(), record, summary, npz, anchor, depth=1)

        assert out["inv"] == anchor and out["depth"] == 1
        assert out["opp_model_used"] == "recorded@divergence"
        chosen = next(c for c in out["candidates"] if c["is_chosen"])
        if chosen["value"] is not None:        # None only if the chosen move ENDED the battle
            expected = float(np.asarray(npz["obs"][anchor + 1], dtype=np.float64).sum())
            assert abs(chosen["value"] - expected) < 1e-2, (
                "depth-1 chosen value != sum(recorded next obs) — the value_crn anchor broke "
                "(clone recorded_exact ≠ recorded next state)")
        # Every candidate maps to a choice and is scored or terminal; ΔV is relative to the chosen line.
        for c in out["candidates"]:
            assert c["choice"]
            assert (c["value"] is not None) or (c["terminal"] is not None)


@pytest.mark.integration
def test_better_line_depth2_beam_produces_principal_variation():
    with tempfile.TemporaryDirectory(prefix="better_line_d2_") as out_dir:
        record, summary, npz = _record_one_battle(out_dir)
        anchor = _mid_anchor(summary, npz)
        assert anchor is not None
        model = _SumModel()
        out = better_line_decision(model, record, summary, npz, anchor,
                                   depth=2, beam=3, top_k=4, opp_model=model)
        assert out["depth"] == 2 and out["opp_model_used"] == "reloaded"
        best = out["best_alternative"]
        if best is not None and best["terminal"] is None:
            pv = best["principal_variation"]
            assert 1 <= len(pv) <= 2          # a depth-2 line: the divergence ply + (≤1) interior ply
            assert pv[0]["action"] == best["action"]


@pytest.mark.integration
def test_better_line_is_deterministic():
    with tempfile.TemporaryDirectory(prefix="better_line_det_") as out_dir:
        record, summary, npz = _record_one_battle(out_dir)
        anchor = _mid_anchor(summary, npz)
        assert anchor is not None
        a = better_line_decision(_SumModel(), record, summary, npz, anchor, depth=2, opp_model=_SumModel())
        b = better_line_decision(_SumModel(), record, summary, npz, anchor, depth=2, opp_model=_SumModel())
        assert [c["action"] for c in a["candidates"]] == [c["action"] for c in b["candidates"]]
        assert [c["value"] for c in a["candidates"]] == [c["value"] for c in b["candidates"]]


if __name__ == "__main__":
    for fn in (test_better_line_depth1_chosen_value_matches_recorded_next_obs,
               test_better_line_depth2_beam_produces_principal_variation,
               test_better_line_is_deterministic):
        fn()
        print(f"{fn.__name__}: PASSED")
