"""End-to-end one-ply lookahead: play a REAL Rust-core battle (written as a core trace), then for an
anchored move decision re-roll each legal action, read the resulting successor obs off the core's parse
chain, and read a FAKE model's V(s′).

The decisive gate is MODEL-FREE: the CHOSEN action under the ``"original"`` (CRN) seed reproduces the
REAL next state, so its materialized successor obs must equal the recorded ``states.npz`` obs at the
next decision bit-for-bit — proving the re-roll → core-walk → value pipeline is faithful on a re-rolled
line.

Needs the Node bridge (the default re-roll driver) and the env core's cdylib; no server."""

import tempfile

import numpy as np
import pytest

from main.prober.core_trace_integration_test import build_core_lib, record_core_battle
from main.prober.engine import _has_state
from main.prober.lookahead import lookahead_decision

# gen3 test tiers (MEASURED 2026-08-14): 7.3 s / 2 tests
pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def lib():
    return build_core_lib()


class _SumValueModel:
    """A fake ProbeModel exposing only the readouts the lookahead calls: ``value`` = obs.sum() (a
    deterministic function of the obs, so a faithful successor row is provable), and no aux heads. Every
    obs it is shown is KEPT (``seen``), so a test can hold a successor row bit-for-bit."""

    def __init__(self):
        self.seen = []

    def value(self, obs, mask):
        self.seen.append(np.asarray(obs, dtype=np.float32).copy())
        return float(np.asarray(obs, dtype=np.float64).sum())

    def win_prob_at(self, obs, mask):
        return None


def _record_one_battle(lib, out_dir: str):
    """The SAME real battle every run (``record_core_battle``). This fixture used a clock seed until
    2026-08-22, so the gate below was one unlucky draw away from being skipped (or from a battle with no
    anchor at all)."""
    return record_core_battle(lib, out_dir, key=0, tag="LA")


def _first_anchor_with_successor(summary, npz):
    invs = summary["invocations"]
    for i, inv in enumerate(invs):
        if (inv.get("phase") == "move_selection" and i + 1 < len(invs)
                and _has_state(npz, i) and _has_state(npz, i + 1)):
            return i
    return None


@pytest.mark.integration
def test_lookahead_chosen_crn_reproduces_recorded_next_obs(lib):
    with tempfile.TemporaryDirectory(prefix="lookahead_it_") as out_dir:
        record, summary, npz = _record_one_battle(lib, out_dir)
        anchor = _first_anchor_with_successor(summary, npz)
        assert anchor is not None, "battle had no move decision with a captured successor"

        model = _SumValueModel()
        out = lookahead_decision(model, record, summary, npz, anchor, n_seeds=0)

        assert out["inv"] == anchor
        assert out["side"] == record.side_of(record.trainee_username)
        assert out["candidates"], "no legal candidates were swept"
        chosen = next(c for c in out["candidates"] if c["is_chosen"])
        assert chosen["choice"], "chosen action did not map to a sim choice string"

        # THE GATE: the chosen action's CRN successor reproduces the realized next state, so its
        # materialized obs (→ obs.sum() via the fake model) equals the recorded next obs's sum.
        #
        # ASSERTED, not branched on. `value_crn` is None only if the chosen move ENDED the battle,
        # and the anchor was picked to HAVE a recorded successor — so on the CRN seed it cannot
        # have ended. An `if` here would have let an unlucky fixture skip the only gate this test
        # exists for, silently and greenly.
        assert chosen["value_crn"] is not None, (
            "the anchor has a recorded successor, so its CRN re-roll cannot have ended the "
            "battle — a None here is a re-roll defect, not an unlucky battle")
        expected = float(np.asarray(npz["obs"][anchor + 1], dtype=np.float64).sum())
        assert abs(chosen["value_crn"] - expected) < 1e-2, (
            f"chosen CRN successor obs.sum()={chosen['value_crn']} != "
            f"recorded next obs.sum()={expected} — re-roll/materialize desync")
        # STRENGTHENED (the sum gate above is loose on a ~19,500-magnitude sum): the recorded next obs
        # must be one of the successor rows the lookahead actually valued, BIT-FOR-BIT — the real proof
        # the V(s′) input is faithful on the RE-ROLLED line. (It formerly re-derived the row through the
        # poke-env materializer, deleted in T27 P6 slice 6d-2; the row the lookahead itself read off the
        # core's parse chain is the stronger subject.)
        nxt = np.asarray(npz["obs"][anchor + 1], dtype=np.float32)
        assert any(np.array_equal(o, nxt) for o in model.seen), (
            "no successor row the lookahead valued equals the recorded next obs BIT-FOR-BIT — "
            "re-roll / core-walk desync")

        # Every candidate maps to a sim choice and gets a numeric V(s′) or a terminal outcome.
        # COUNTED (gen3_vacuity_hunt_v1): the `delta_v` check hangs off `value_crn is not None`,
        # so a decision whose candidates all came back TERMINAL would satisfy the line above
        # while never once evaluating it — the SCORED path, which is the entire point of
        # `lookahead`, would go untested and the test would still pass green.
        n_scored = 0
        for c in out["candidates"]:
            assert c["choice"]
            assert (c["value_crn"] is not None) or (c["terminal"] is not None)
            if c["value_crn"] is not None:
                n_scored += 1
                assert c["delta_v"] is not None
        assert n_scored >= 1, (
            f"no candidate carried a numeric V(s′) — {len(out['candidates'])} candidates, all "
            f"terminal, so the ΔV assertion above never ran")

        # The headline ΔV is relative to the chosen line (chosen ΔV == 0 when it has a successor).
        assert abs(chosen["delta_v"]) < 1e-6


@pytest.mark.integration
def test_lookahead_is_deterministic(lib):
    with tempfile.TemporaryDirectory(prefix="lookahead_det_") as out_dir:
        record, summary, npz = _record_one_battle(lib, out_dir)
        anchor = _first_anchor_with_successor(summary, npz)
        assert anchor is not None
        a = lookahead_decision(_SumValueModel(), record, summary, npz, anchor, n_seeds=4)
        b = lookahead_decision(_SumValueModel(), record, summary, npz, anchor, n_seeds=4)
        # Same fresh-seed salt + same replay ⇒ byte-identical value sweep.
        assert [c["value_crn"] for c in a["candidates"]] == [c["value_crn"] for c in b["candidates"]]
        assert [c["value_mean"] for c in a["candidates"]] == [c["value_mean"] for c in b["candidates"]]
