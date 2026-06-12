"""Tests for the decision-table forensic export (forensics.py).

The pure move taxonomy is unit-tested directly; the decision-table builder is tested against a
hand-written tmp trace through a fake session (no torch, no bridge)."""
import json
import unittest

import numpy as np

from main.prober import forensics


class TestMoveCategory(unittest.TestCase):
    def test_each_category(self):
        self.assertEqual(forensics.move_category("explosion"), "selfko")
        self.assertEqual(forensics.move_category("selfdestruct"), "selfko")
        self.assertEqual(forensics.move_category("recover"), "recovery")
        self.assertEqual(forensics.move_category("softboiled"), "recovery")
        self.assertEqual(forensics.move_category("dragondance"), "setup")
        self.assertEqual(forensics.move_category("protect"), "stall")
        self.assertEqual(forensics.move_category("substitute"), "stall")
        self.assertEqual(forensics.move_category("toxic"), "status")
        self.assertEqual(forensics.move_category("leechseed"), "status")
        self.assertEqual(forensics.move_category("switch:gengar"), "switch")
        self.assertEqual(forensics.move_category("rockslide"), "attack_or_other")
        self.assertEqual(forensics.move_category(""), "unknown")
        self.assertEqual(forensics.move_category(None), "unknown")

    def test_case_insensitive(self):
        self.assertEqual(forensics.move_category("Explosion"), "selfko")


# --------------------------------------------------------------------------- #
class _FakeSession:
    """A stand-in for ProbeSession exposing only .battles() over hand-written traces."""
    def __init__(self, battles):
        self._battles = battles

    def battles(self):
        return self._battles


def _write_trace(tmp_path, invs, npz):
    smf = tmp_path / "loss_s0_001_summary.json"
    smf.write_text(json.dumps({"meta": {"result": "LOSS"}, "invocations": invs}))
    np.savez(str(smf).replace("_summary.json", "_states.npz"), **npz)
    return str(smf)


class TestBuildDecisionTable(unittest.TestCase):
    def _make(self, tmp_path):
        invs = [
            {"turn": 1, "phase": "move_selection", "chosen": "explosion",
             "our": {"species": "tyranitar", "hp": "100%"},
             "opp": {"species": "swampert", "hp": "80%"},
             "outcome": {"reward": {"total": -2.0}, "events": ["our:tyranitar:fainted"]}},
            {"turn": 2, "phase": "move_selection", "chosen": "rockslide",
             "our": {"species": "aerodactyl", "hp": "90%"},
             "opp": {"species": "swampert", "hp": "55%"},
             "outcome": {"reward": {"total": 1.0}, "events": ["opp:swampert:fainted"]}},
        ]
        logits = np.zeros((2, 11), dtype=np.float32)
        logits[0, 3] = 5.0   # peak at the chosen explosion action index
        logits[1, 0] = 5.0
        npz = dict(
            obs=np.zeros((2, 3409), dtype=np.float32),
            logits=logits,
            values=np.array([-5.0, -3.0], dtype=np.float32),  # dV[0] = -3-(-5) = +2.0
            has_state=np.array([1, 1], dtype=np.int8),
            actions=np.array([3, 0], dtype=np.int16),
        )
        smf = _write_trace(tmp_path, invs, npz)
        sess = _FakeSession([{"id": smf, "short_id": "step_100/heuristic/loss_0",
                              "step": 100, "opponent": "heuristic", "outcome": "loss"}])
        return sess

    def test_columns(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            sess = self._make(pathlib.Path(td))
            rows = forensics.build_decision_table(sess)
        self.assertEqual(len(rows), 2)
        r0, r1 = rows
        self.assertEqual(r0["cat"], "selfko")
        self.assertEqual(r0["our"], "tyranitar")
        self.assertEqual(r0["our_hp"], 100)
        self.assertEqual(r0["opp_hp"], 80)
        self.assertAlmostEqual(r0["reward"], -2.0)
        self.assertAlmostEqual(r0["dV"], 2.0)            # the self-KO over-valuation signal
        self.assertTrue(r0["faint_us"])
        self.assertFalse(r0["faint_opp"])
        self.assertGreater(r0["conf"], 0.9)              # logits peaked at the chosen action
        self.assertEqual(r1["cat"], "attack_or_other")
        self.assertIsNone(r1["dV"])                      # last decision has no next value
        self.assertTrue(r1["faint_opp"])

    def test_category_filter(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            sess = self._make(pathlib.Path(td))
            rows = forensics.build_decision_table(sess, categories=["selfko"])
        self.assertEqual([r["cat"] for r in rows], ["selfko"])

    def test_digest(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            sess = self._make(pathlib.Path(td))
            rows = forensics.build_decision_table(sess)
        dig = forensics.decision_table_digest(rows)
        self.assertEqual(dig["n_decisions"], 2)
        self.assertEqual(dig["outcomes"], {"loss": 2})
        self.assertIn("selfko", dig["by_category"])
        self.assertEqual(dig["by_category"]["selfko"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
