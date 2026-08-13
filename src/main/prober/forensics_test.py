"""Tests for the decision-table forensic export (forensics.py).

The pure move taxonomy is unit-tested directly; the decision-table builder is tested against a
hand-written tmp trace through a fake session (no torch, no bridge)."""
import json
import unittest

import numpy as np

from main.prober import forensics
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

# Fake-obs buffer width — the LIVE encoder dim, resolved dynamically so it tracks obs-layout changes
# (was a hardcoded 3457 that silently drifted; `build_decision_table` decodes the incoming-belief block
# at live offsets, so the buffer must be ≥ total_dim — pin it to the real value).
_OBS_W = Gen3ObservationEncoder(load_mappings()).get_layout()["total_dim"]


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
        # CURE ≠ recovery ≠ status: Refresh/Heal Bell CLEAR status, Recover heals HP, Toxic inflicts it.
        self.assertEqual(forensics.move_category("refresh"), "cure")
        self.assertEqual(forensics.move_category("healbell"), "cure")
        self.assertEqual(forensics.move_category("aromatherapy"), "cure")
        self.assertEqual(forensics.move_category("rest"), "recovery")   # cures by INFLICTING sleep
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
            obs=np.zeros((2, _OBS_W), dtype=np.float32),
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
        self.assertIsNone(dig["cure_uptake"])            # no statused-with-a-legal-cure decision here


class TestCureUptake(unittest.TestCase):
    """The cure columns + digest block: over decisions where a status cure was genuinely available
    (statused AND legal), how often the policy took it — the 'heals HP but never clears status' read."""
    def _rows(self, td):
        import pathlib
        invs = [
            # statused + a legal Refresh, took Recover instead → offered, not taken
            {"turn": 1, "phase": "move_selection", "chosen": "recover",
             "our": {"species": "milotic", "hp": "83%", "status": "TOX(1)"},
             "opp": {"species": "swampert", "hp": "94%"},
             "actions": {"recover": {"prob": "48.6%", "valid": True},
                         "refresh": {"prob": "2.1%", "valid": True}},
             "outcome": {"reward": {"total": 0.3}, "events": []}},
            # statused + took the cure → offered AND taken
            {"turn": 2, "phase": "move_selection", "chosen": "refresh",
             "our": {"species": "milotic", "hp": "88%", "status": "TOX(2)"},
             "opp": {"species": "swampert", "hp": "88%"},
             "actions": {"recover": {"prob": "20.0%", "valid": True},
                         "refresh": {"prob": "70.0%", "valid": True}},
             "outcome": {"reward": {"total": 0.1}, "events": []}},
            # TAUNTED: the cure is illegal, so it was never on the table → NOT offered
            {"turn": 3, "phase": "move_selection", "chosen": "surf",
             "our": {"species": "milotic", "hp": "88%", "status": "TOX(2)|TAUNT"},
             "opp": {"species": "skarmory", "hp": "100%"},
             "actions": {"surf": {"prob": "90.0%", "valid": True},
                         "refresh": {"prob": "0.0%", "valid": False}},
             "outcome": {"reward": {"total": 0.0}, "events": []}},
        ]
        npz = dict(
            obs=np.zeros((3, _OBS_W), dtype=np.float32),
            logits=np.zeros((3, 11), dtype=np.float32),
            values=np.zeros(3, dtype=np.float32),
            has_state=np.array([1, 1, 1], dtype=np.int8),
            actions=np.array([0, 0, 0], dtype=np.int16),
        )
        smf = _write_trace(pathlib.Path(td), invs, npz)
        sess = _FakeSession([{"id": smf, "short_id": "step_100/random/loss_0",
                              "step": 100, "opponent": "random", "outcome": "loss"}])
        return forensics.build_decision_table(sess)

    def test_columns_and_digest(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            rows = self._rows(td)
        self.assertEqual([r["cat"] for r in rows], ["recovery", "cure", "attack_or_other"])
        self.assertEqual(rows[0]["cure_avail"], "refresh")
        self.assertAlmostEqual(rows[0]["cure_prob"], 0.021)
        self.assertFalse(rows[0]["chose_cure"])
        self.assertEqual(rows[0]["our_status"], "TOX(1)")
        self.assertTrue(rows[1]["chose_cure"])
        self.assertIsNone(rows[2]["cure_avail"])         # Taunted → the cure was never legal
        self.assertIsNone(rows[2]["chose_cure"])
        cure = forensics.decision_table_digest(rows)["cure_uptake"]
        self.assertEqual((cure["n_offered"], cure["n_taken"]), (2, 1))
        self.assertAlmostEqual(cure["uptake"], 0.5)
        self.assertEqual(cure["instead"], {"recover": 1})


if __name__ == "__main__":
    unittest.main()
