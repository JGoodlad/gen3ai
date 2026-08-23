"""End-to-end one-ply lookahead: record a REAL bridge battle, then for an anchored move decision
re-roll each legal action, materialize the resulting successor obs, and read a FAKE model's V(s′).

The decisive gate is MODEL-FREE: the CHOSEN action under the ``"original"`` (CRN) seed reproduces the
REAL next state, so its materialized successor obs must equal the recorded ``states.npz`` obs at the
next decision bit-for-bit — proving the re-roll → materialize → value pipeline is faithful (the same
guarantee ``obs_roundtrip_fuzz_test`` gives the recorded line, here exercised on a re-rolled one).

Needs the Node bridge; no server. Run directly or via ``pytest -m integration``."""

import tempfile

import numpy as np
import pytest

from agents.training.obs_roundtrip_fuzz_test import record_fixture_battle
from main.prober.engine import _has_state
from main.prober.lookahead import lookahead_decision

# gen3 test tiers (MEASURED 2026-08-14): 7.3 s / 2 tests
pytestmark = pytest.mark.sim


class _SumValueModel:
    """A fake ProbeModel exposing only the readouts the lookahead calls: ``value`` = obs.sum() (a
    deterministic function of the obs, so a faithful materialization is provable), and no aux heads."""

    def value(self, obs, mask):
        return float(np.asarray(obs, dtype=np.float64).sum())

    def value_dist_at(self, obs, mask):
        return None

    def win_prob_at(self, obs, mask):
        return None


def _record_one_battle(out_dir: str):
    """The SAME real battle every run — see `record_fixture_battle`'s wall-clock-seed note.
    This fixture used a clock seed until 2026-08-22, so the gate below was one unlucky draw
    away from being skipped (or from a battle with no anchor at all)."""
    return record_fixture_battle(out_dir, key=0, tag="LA")


def _first_anchor_with_successor(summary, npz):
    invs = summary["invocations"]
    for i, inv in enumerate(invs):
        if (inv.get("phase") == "move_selection" and i + 1 < len(invs)
                and _has_state(npz, i) and _has_state(npz, i + 1)):
            return i
    return None


@pytest.mark.integration
def test_lookahead_chosen_crn_reproduces_recorded_next_obs():
    with tempfile.TemporaryDirectory(prefix="lookahead_it_") as out_dir:
        record, summary, npz = _record_one_battle(out_dir)
        anchor = _first_anchor_with_successor(summary, npz)
        assert anchor is not None, "battle had no move decision with a captured successor"

        out = lookahead_decision(_SumValueModel(), record, summary, npz, anchor, n_seeds=0)

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
        # STRENGTHENED (the sum gate above is loose on a ~19,500-magnitude sum): independently
        # reproduce the chosen action's CRN successor obs (recorded action + "original" seed, exactly
        # what lookahead does internally) and assert it equals the recorded next obs BIT-FOR-BIT — the
        # obs_roundtrip guarantee, here on the RE-ROLLED line (the real proof the V(s′) input is faithful).
        from agents.training.obs_materializer import materialize_decisions
        from utils.bridge.reconstruction import reroll_turn
        actions = np.asarray(npz["actions"], dtype=int)
        side = record.side_of(record.trainee_username)
        rr = reroll_turn(record, int(summary["invocations"][anchor]["turn"]),
                         seeds=["original"], p1_action="recorded", p2_action="recorded")
        r0 = rr.rerolls[0]
        assert not r0.outcome.get("ended"), "anchor's turn ended — pick an earlier move decision"
        prefix = rr.prefix_p1_chunks if side == "p1" else rr.prefix_p2_chunks
        suffix = r0.p1_chunks if side == "p1" else r0.p2_chunks
        mt = materialize_decisions(
            list(prefix) + list(suffix), username=record.username(side),
            packed_team=record.packed_team(side), side=side,
            actions=list(actions[:anchor]) + [int(actions[anchor])],
            battle_format=record.format_id, battle_tag=record.battle_tag,
            stop_after_decision=anchor + 1)
        assert len(mt.decisions) > anchor + 1, "no successor row materialized"
        assert np.array_equal(mt.decisions[anchor + 1].obs,
                              np.asarray(npz["obs"][anchor + 1], dtype=np.float32)), (
            "re-rolled CRN successor obs != recorded next obs BIT-FOR-BIT — materialize desync")

        # Every candidate maps to a sim choice and gets a numeric V(s′) or a terminal outcome.
        for c in out["candidates"]:
            assert c["choice"]
            assert (c["value_crn"] is not None) or (c["terminal"] is not None)
            if c["value_crn"] is not None:
                assert c["delta_v"] is not None

        # The headline ΔV is relative to the chosen line (chosen ΔV == 0 when it has a successor).
        assert abs(chosen["delta_v"]) < 1e-6


@pytest.mark.integration
def test_lookahead_is_deterministic():
    with tempfile.TemporaryDirectory(prefix="lookahead_det_") as out_dir:
        record, summary, npz = _record_one_battle(out_dir)
        anchor = _first_anchor_with_successor(summary, npz)
        assert anchor is not None
        a = lookahead_decision(_SumValueModel(), record, summary, npz, anchor, n_seeds=4)
        b = lookahead_decision(_SumValueModel(), record, summary, npz, anchor, n_seeds=4)
        # Same fresh-seed salt + same replay ⇒ byte-identical value sweep.
        assert [c["value_crn"] for c in a["candidates"]] == [c["value_crn"] for c in b["candidates"]]
        assert [c["value_mean"] for c in a["candidates"]] == [c["value_mean"] for c in b["candidates"]]


if __name__ == "__main__":
    # Run directly (no server needed — the bridge is in-process).
    for fn in (test_lookahead_chosen_crn_reproduces_recorded_next_obs, test_lookahead_is_deterministic):
        fn()
        print(f"{fn.__name__}: PASSED")
