"""Pure unit tests for the one-ply lookahead ORCHESTRATION (candidate sweep, ΔV, terminal handling,
ranking) — the bridge (reroll_many), the core's walk / stream reader and the model are monkeypatched, so
no binary, no torch. The real re-roll → core row → value pipeline is exercised end-to-end (against a real battle)
in ``lookahead_integration_test.py``; this file pins the logic on top of it."""

from types import SimpleNamespace

import numpy as np
import pytest

import main.prober.lookahead as LA
from main.prober.core_walk import StreamDecision
from main.prober.lookahead import _stats, lookahead_decision

_DIM = 4
# choice_map for the decision: action idx → sim choice string. Two "value" moves + a terminal switch.
_CHOICES = {0: "move alpha", 1: "move beta", 6: "switch gamma"}
_TERMINAL_CHOICE = "switch gamma"
_TURN = 5
_INV = 2
_CHOSEN = 1


def _fake_record():
    return SimpleNamespace(
        battle_tag="battle-gen3ou-1", trainee_username="trainee", format_id="gen3ou",
        side_of=lambda name: "p1", username=lambda side: "trainee", packed_team=lambda side: "TEAM",
    )


def _summary():
    # 11-key actions dict so _label_of can index 0..6; only `phase`/`turn` matter to the engine.
    acts = {f"a{i}": {"valid": True} for i in range(11)}
    # The core walk (fake) reaches the anchor decision at turn _TURN, so the anchor inv's turn must
    # match (the engine asserts the walked turn == inv["turn"]).
    invs = [{"phase": "move_selection", "turn": _TURN, "actions": acts} for _ in range(_INV + 2)]
    return {"invocations": invs, "meta": {"result": "loss"}}


def _npz():
    actions = np.zeros(_INV + 2, dtype=int)
    actions[_INV] = _CHOSEN
    values = np.arange(_INV + 2, dtype=np.float32)   # recorded V(s): 0,1,2,3,...
    return {"actions": actions, "values": values}


class _Model:
    """value(obs) = obs.sum(); the fake stream reader encodes the candidate idx into the successor obs."""

    def value(self, obs, mask):
        return float(np.asarray(obs, dtype=np.float64).sum())

    def win_prob_at(self, obs, mask):
        return None


def _install_fakes(monkeypatch):
    def fake_decision_choices(record, side, index):
        # the core walk's legal choice map at the anchor, and the turn it walks it at
        return dict(_CHOICES), _TURN

    def fake_reroll_many(record, turn, arms, *, followup="random", **kw):
        # The batched driver resolves every (candidate × seed) arm in one process; each arm carries its
        # own p1/p2 action source + seed + label (the action index). A terminal arm ends the battle.
        out_arms = []
        for arm in arms:
            our = arm.get("p1_action")
            ended = (our == _TERMINAL_CHOICE)
            out_arms.append(SimpleNamespace(
                label=arm["label"], seed=arm["seed"],
                outcome={"ended": ended, "winner": "trainee" if ended else None},
                p1_chunks=["S"], p2_chunks=["S"]))
        return SimpleNamespace(prefix_p1_chunks=["P"], prefix_p2_chunks=["P"], arms=out_arms)

    def fake_read_streams(requests, **kw):
        # One stream per arm (the recorded prefix + the arm's ply) — the shape the core's parse-chain
        # reader returns: every decision up to the successor, which is the one encoded.
        out = []
        for r in requests:
            assert list(r.chunks) == ["P", "S"] and r.encode_at == [len(r.actions)]
            cand = int(list(r.actions)[-1])           # the candidate played this re-roll
            decs = [StreamDecision(k=k, turn=_TURN, mask=np.ones(11, np.int8), tokens={}, obs=None)
                    for k in range(len(r.actions))]
            # The successor row (one past the last action) carries the candidate signal → distinct V.
            decs.append(StreamDecision(k=len(r.actions), turn=_TURN + 1, mask=np.ones(11, np.int8),
                                       tokens={}, obs=np.full(_DIM, float(cand), np.float32)))
            out.append(decs)
        return out

    monkeypatch.setattr(LA, "decision_choices", fake_decision_choices)
    monkeypatch.setattr(LA, "reroll_many", fake_reroll_many)
    monkeypatch.setattr(LA, "read_streams", fake_read_streams)


def test_stats_helper():
    assert _stats([]) == (None, None)
    m, s = _stats([1.0, 3.0])
    assert m == 2.0 and s == 1.0


def test_lookahead_sweeps_all_candidates_and_computes_delta(monkeypatch):
    _install_fakes(monkeypatch)
    out = lookahead_decision(_Model(), _fake_record(), _summary(), _npz(), _INV, n_seeds=0)

    assert out["inv"] == _INV and out["turn"] == _TURN and out["side"] == "p1"
    assert {c["action"] for c in out["candidates"]} == set(_CHOICES)   # every legal action swept
    by_action = {c["action"]: c for c in out["candidates"]}

    # value_crn = successor obs.sum() = _DIM * candidate idx, for the non-terminal candidates.
    assert by_action[0]["value_crn"] == 0.0
    assert by_action[1]["value_crn"] == float(_DIM * 1)
    # The terminal switch ended the battle → no successor V, flagged terminal=win.
    assert by_action[6]["value_crn"] is None
    assert by_action[6]["terminal"] == "win"
    assert by_action[6]["terminal_frac"] == 1.0

    # ΔV is relative to the CHOSEN action's value (idx 1 → baseline _DIM); chosen ΔV == 0.
    assert by_action[_CHOSEN]["is_chosen"] is True
    assert by_action[_CHOSEN]["delta_v"] == 0.0
    assert by_action[0]["delta_v"] == float(-_DIM)        # 0 - _DIM


def test_lookahead_ranks_terminal_win_first_and_picks_best_alt(monkeypatch):
    _install_fakes(monkeypatch)
    out = lookahead_decision(_Model(), _fake_record(), _summary(), _npz(), _INV, n_seeds=0)
    # Ranking: terminal WIN floats to the top, then by value_crn desc.
    order = [c["action"] for c in out["candidates"]]
    assert order[0] == 6                                  # terminal win first
    assert order.index(1) < order.index(0)                # value 4 ranks above value 0
    # best_alternative = the top-ranked NON-chosen candidate (the terminal win).
    assert out["best_alternative"] == out["candidates"][0]["label"]
    assert out["recorded_value"] == 2.0                   # values[_INV]
    assert out["recorded_next_value"] == 3.0              # values[_INV+1]


def test_lookahead_dice_average_with_seeds(monkeypatch):
    _install_fakes(monkeypatch)
    out = lookahead_decision(_Model(), _fake_record(), _summary(), _npz(), _INV, n_seeds=3)
    nonterminal = [c for c in out["candidates"] if c["value_crn"] is not None]
    for c in nonterminal:
        # All seeds give the same successor (the fake is seed-independent) → std 0, mean == crn.
        assert c["n_evaluated"] == 4                       # 3 fresh + "original"
        assert c["value_mean"] == c["value_crn"]
        assert c["value_std"] == 0.0


def test_lookahead_rejects_non_move_selection(monkeypatch):
    _install_fakes(monkeypatch)
    summary = _summary()
    summary["invocations"][_INV]["phase"] = "force_switch"
    with pytest.raises(ValueError, match="move_selection"):
        lookahead_decision(_Model(), _fake_record(), summary, _npz(), _INV, n_seeds=0)


def test_lookahead_requires_actions_array(monkeypatch):
    _install_fakes(monkeypatch)
    npz = _npz()
    del npz["actions"]
    with pytest.raises(ValueError, match="actions"):
        lookahead_decision(_Model(), _fake_record(), _summary(), npz, _INV, n_seeds=0)
