"""End-to-end better-line SEARCH: play a REAL Rust-core battle (written as a core trace), then search for a better line from an
anchored move decision with a deterministic FAKE model, exercising the full
SearchSession-clone → materialize → V pipeline (depth 1 AND depth 2, the interior opponent).

The decisive gate is MODEL-FREE: the fake model's V = obs.sum(), so the depth-1 CHOSEN action's value
MUST equal the sum of the recorded ``states.npz`` next obs (its ``recorded_exact`` clone reproduces the
real next state bit-for-bit — the value_crn anchor). Depth 2 additionally checks the beam produces a
principal variation of the right length through the (self-model) interior opponent.

The search runs on the Rust ``search_driver`` (``src/rust_sim/src/bin/search_driver.rs``), the only
offline driver. The node-vs-rust cross-impl test that lived here was REMOVED with the Node driver
(2026-10-10, owner: "delete node"): it was a LIVE comparison with no golden, so with ``search_driver.js``
deleted there is nothing left to compare against. The driver's node-free properties are pinned by the
cargo tests (``src/rust_sim/tests/search_driver_test.rs`` / ``replay_driver_test.rs``).

Needs the env core's cdylib (and the Node team validator, for the fixture's pool teams); no server."""

import tempfile

import numpy as np
import pytest

from main.prober.core_trace_integration_test import build_core_lib, record_core_battle
from main.prober.better_line import better_line_decision
from main.prober.engine import _has_state

# gen3 test tiers (MEASURED 2026-08-14): 38.8 s / 8 tests
pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def lib():
    return build_core_lib()


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


def _record_one_battle(lib, out_dir: str):
    """The SAME real battle every run (``record_core_battle``).

    This fixture is the antipattern's FIRST named instance: it seeded from the clock, so it
    played a different battle every run and an unlucky one outran the recorded command list
    the depth-2 beam replays (ledger 2026-08-22). A fixed battle makes that either always
    true or always false, which is the only state a failure can be diagnosed from."""
    return record_core_battle(lib, out_dir, key=2, tag="BL")


def _mid_anchor(summary, npz):
    invs = summary["invocations"]
    cand = [i for i, inv in enumerate(invs)
            if inv.get("phase") == "move_selection" and i + 1 < len(invs)
            and _has_state(npz, i) and _has_state(npz, i + 1)]
    return cand[len(cand) // 2] if cand else None


@pytest.mark.slow  # ship gate (2026-10-19): 10 s beside a live run
@pytest.mark.integration
def test_better_line_depth1_chosen_value_matches_recorded_next_obs(lib):
    with tempfile.TemporaryDirectory(prefix="better_line_d1_") as out_dir:
        record, summary, npz = _record_one_battle(lib, out_dir)
        anchor = _mid_anchor(summary, npz)
        assert anchor is not None
        out = better_line_decision(_SumModel(), record, summary, npz, anchor, depth=1)

        assert out["inv"] == anchor and out["depth"] == 1
        assert out["opp_model_used"] == "recorded@divergence"
        # The declared interior-ply regime rides on the live payload too (task #29).
        assert out["interior_opponent_regime"] == "greedy"
        chosen = next(c for c in out["candidates"] if c["is_chosen"])
        # ASSERTED, not branched on: `_mid_anchor` only returns a decision that HAS a recorded
        # successor, so the chosen move provably did not end the battle and a None value is a
        # clone defect. Branching on it let an unlucky fixture skip this file's decisive gate.
        assert chosen["value"] is not None, (
            "the anchor has a recorded successor, so the chosen line cannot have ended the "
            "battle — a None value is a clone/materialize defect, not an unlucky battle")
        expected = float(np.asarray(npz["obs"][anchor + 1], dtype=np.float64).sum())
        assert abs(chosen["value"] - expected) < 1e-2, (
            "depth-1 chosen value != sum(recorded next obs) — the value_crn anchor broke "
            "(clone recorded_exact ≠ recorded next state)")
        # Every candidate maps to a choice and is scored or terminal; ΔV is relative to the chosen line.
        for c in out["candidates"]:
            assert c["choice"]
            assert (c["value"] is not None) or (c["terminal"] is not None)


@pytest.mark.slow  # ship gate (2026-10-19): 9 s beside a live run
@pytest.mark.integration
def test_better_line_depth2_beam_produces_principal_variation(lib):
    with tempfile.TemporaryDirectory(prefix="better_line_d2_") as out_dir:
        record, summary, npz = _record_one_battle(lib, out_dir)
        anchor = _mid_anchor(summary, npz)
        assert anchor is not None
        model = _SumModel()
        out = better_line_decision(model, record, summary, npz, anchor,
                                   depth=2, beam=3, top_k=4, opp_model=model)
        assert out["depth"] == 2 and out["opp_model_used"] == "reloaded"
        best = out["best_alternative"]
        # The battle is FIXED, so "did the beam produce a scorable alternative at all?" is a
        # property of the code, not of the draw — assert it rather than skipping past it.
        assert best is not None, "the depth-2 beam returned no alternative to the chosen line"
        assert best["terminal"] is None, (
            "the mid-battle anchor's best alternative ended the battle — the fixture battle "
            "changed shape; re-check the anchor rather than weakening the gate")
        pv = best["principal_variation"]
        assert 1 <= len(pv) <= 2          # a depth-2 line: the divergence ply + (≤1) interior ply
        assert pv[0]["action"] == best["action"]


@pytest.mark.slow  # ship gate (2026-10-19): 12 s beside a live run
@pytest.mark.integration
def test_better_line_is_deterministic(lib):
    with tempfile.TemporaryDirectory(prefix="better_line_det_") as out_dir:
        record, summary, npz = _record_one_battle(lib, out_dir)
        anchor = _mid_anchor(summary, npz)
        assert anchor is not None
        a = better_line_decision(_SumModel(), record, summary, npz, anchor, depth=2,
                                 opp_model=_SumModel())
        b = better_line_decision(_SumModel(), record, summary, npz, anchor, depth=2,
                                 opp_model=_SumModel())
        assert [c["action"] for c in a["candidates"]] == [c["action"] for c in b["candidates"]]
        assert [c["value"] for c in a["candidates"]] == [c["value"] for c in b["candidates"]]
