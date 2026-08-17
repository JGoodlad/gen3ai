"""Unit tests for the Phase-0 selection funnel (scan → falsify-gate → MISTAKE-only → window → rank)
with a fake ProbeSession + monkeypatched falsify_battle — no model, no bridge."""

from types import SimpleNamespace


import agents.training.teacher.selection as SEL
from agents.training.teacher.selection import select_candidates


def _summary(n_turns=10):
    # one move_selection invocation per turn (inv index == turn-1 here for simplicity).
    return {"invocations": [{"phase": "move_selection", "turn": t} for t in range(1, n_turns + 1)]}


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.tree = SimpleNamespace(steps=[SimpleNamespace(step=100)])
        self._gamma = 0.99

    def scan(self, *, outcome, step, limit, metric):
        return self._rows

    def _battle(self, _id):
        return object()

    def _summary(self, _b):
        return _summary()

    def _npz(self, _b):
        return {}


def _install(monkeypatch, rows, falsify_decisions, recon_exists=True):
    # selection.py imports these INSIDE the function, so patch them at their SOURCE modules.
    monkeypatch.setattr("main.prober.session.ProbeSession", lambda run_dir: _FakeSession(rows))
    monkeypatch.setattr("main.prober.falsifier.falsify_battle",
                        lambda *a, **k: {"decisions": falsify_decisions})
    monkeypatch.setattr("utils.bridge.reconstruction.ReconstructionRecord.load",
                        staticmethod(lambda p: object()))
    monkeypatch.setattr(SEL.os.path, "exists", lambda p: recon_exists)


def _row(path, inv, turn, delta, opp="sentinel_1"):
    return {"id": path, "opponent": opp, "step": 100,
            "worst": {"inv": inv, "turn": turn, "delta_v": delta}}


def test_falsify_gate_keeps_only_mistakes(monkeypatch):
    rows = [_row("/x/loss_a_summary.json", inv=5, turn=6, delta=-40.0)]
    # falsify says inv 5 is a MISTAKE, inv 2 is LUCK → only 5 (and its window) survive.
    decisions = [{"inv": 5, "turn": 6, "verdict": "MISTAKE", "anchor_delta": -40.0},
                 {"inv": 2, "turn": 3, "verdict": "LUCK", "anchor_delta": -30.0}]
    _install(monkeypatch, rows, decisions)
    cands = select_candidates("/run", budget=20, falsify_gate=True, window=1)
    invs = {c.inv_index for c in cands}
    assert 5 in invs and 2 not in invs                  # the LUCK turn is NOT a candidate
    assert all(c.verdict == "MISTAKE" for c in cands)


def test_window_expansion_queues_preceding_turns(monkeypatch):
    rows = [_row("/x/loss_a_summary.json", inv=5, turn=6, delta=-40.0)]
    decisions = [{"inv": 5, "turn": 6, "verdict": "MISTAKE", "anchor_delta": -40.0}]
    _install(monkeypatch, rows, decisions)
    cands = select_candidates("/run", budget=20, falsify_gate=True, window=2)
    invs = sorted(c.inv_index for c in cands)
    # crater at turn 6 (inv 5) + window turns 5 (inv 4) and 4 (inv 3).
    assert invs == [3, 4, 5]
    # window priority decays below the crater's.
    by = {c.inv_index: c.anchor_delta for c in cands}
    assert by[5] > by[4] > by[3]


def test_no_gate_uses_crater_directly(monkeypatch):
    rows = [_row("/x/loss_a_summary.json", inv=7, turn=8, delta=-50.0)]
    _install(monkeypatch, rows, [])      # falsify not called when gate off
    cands = select_candidates("/run", budget=20, falsify_gate=False, window=0)
    assert len(cands) == 1 and cands[0].inv_index == 7 and cands[0].verdict == "CRATER"


def test_skips_battles_without_reconstruction(monkeypatch):
    rows = [_row("/x/loss_a_summary.json", inv=5, turn=6, delta=-40.0)]
    _install(monkeypatch, rows, [], recon_exists=False)
    assert select_candidates("/run", budget=20, falsify_gate=False) == []


def test_ranked_by_anchor_delta_and_capped(monkeypatch):
    rows = [_row("/x/a_summary.json", 3, 4, -10.0), _row("/x/b_summary.json", 4, 5, -50.0),
            _row("/x/c_summary.json", 2, 3, -30.0)]
    _install(monkeypatch, rows, [])
    cands = select_candidates("/run", budget=2, falsify_gate=False, window=0)
    assert len(cands) == 2                                # capped at budget
    assert cands[0].anchor_delta >= cands[1].anchor_delta and cands[0].anchor_delta == 50.0
