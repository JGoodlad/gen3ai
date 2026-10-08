"""Pure unit tests for the counterfactual replay ORCHESTRATION on the Rust core (``replay.py``, P6):
win-rate aggregation, opponent resolution (who plays, in which regime, with which seeds and stall
rule), the Monte-Carlo seed plumbing, the narrated trajectories and every refusal. The core walk and the
play-out are monkeypatched, so no core, no torch. The play-out's own semantics are pinned by
``src/rust_env/tests/search_playout_cf_test.rs`` and ``utils/rust_env/cf_playout_test.py``; that the
whole command RUNS with poke-env blocked by ``src/poke_env_free_entry_points_test.py``."""

from types import SimpleNamespace

import numpy as np
import pytest

import main.prober.replay as RP
from main.prober import core_walk
from main.prober.replay import replay_counterfactual_battle, wilson_ci
from utils.rust_env import counterfactual as CF

_CHOICES = {0: "move alpha", 1: "move beta", 6: "switch gamma"}
_TURN = 8
_INV = 4
_CHOSEN = 1
_TAG = "battle-gen3ou-1"


def _record(side="p1"):
    return SimpleNamespace(battle_tag=_TAG, trainee_username="trainee", side_of=lambda name: side)


def _summary():
    acts = {f"a{i}": {"valid": True} for i in range(11)}
    invs = [{"phase": "move_selection", "turn": _TURN, "actions": acts} for _ in range(_INV + 2)]
    return {"invocations": invs, "meta": {"result": "LOSS"}}


def _npz():
    actions = np.zeros(_INV + 2, dtype=int)
    actions[_INV] = _CHOSEN
    return {"actions": actions}


class _Core:
    obs_dim = 2761

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _model(dim=2761):
    return SimpleNamespace(observation_space={"observation": SimpleNamespace(shape=(dim,))})


def _install(monkeypatch, outcomes, *, walk_turn=_TURN, root_tokens=None, text=False):
    """Fake the walk + the play-out; ``outcomes`` is one per rollout. Returns the play-out calls."""
    calls = []
    monkeypatch.setattr(core_walk, "decision_choices", lambda record, side, index: (dict(_CHOICES), walk_turn))

    def fake_cf(record, **kw):
        calls.append(kw)
        n = len(kw["post_t_seeds"])
        ro = [CF.Rollout(outcome=outcomes[k % len(outcomes)], ended=True, turns=20 + k, forfeit=False, capped=False,
                         reseed=kw["post_t_seeds"][k], decisions=(3, 3),
                         text=[f"|turn|{9 + k}", f"|move|p1a: X|Move{k}|p2a: Y"] if kw["text"] else None, cmds=None)
              for k in range(n)]
        return {"schema": CF.CF_SCHEMA, "our_side": "p1", "rollouts": ro, "answered": 0, "wall_s": 0.0,
                "root": {"tokens": {str(a): t for a, t in (root_tokens or _CHOICES).items()},
                         "other_recorded": "move opp"}}

    monkeypatch.setattr(CF, "replay_counterfactual", fake_cf)
    return calls


def _run(action=0, **kw):
    kw.setdefault("play_model", _model())
    kw.setdefault("opp_name", "heuristic")
    kw.setdefault("core", _Core())
    return replay_counterfactual_battle(_record(), _summary(), _npz(), _INV, action, **kw)


def test_wilson_ci_bounds():
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = wilson_ci(5, 10)
    assert 0.0 < lo < 0.5 < hi < 1.0
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and lo < 1.0


def test_single_line_is_not_a_probability(monkeypatch):
    calls = _install(monkeypatch, ["win"])
    out = _run(0, n_rollouts=1)
    assert out["n_rollouts"] == 1 and out["deterministic_line"] is True
    assert out["win_rate"] == 1.0 and out["wins"] == 1
    assert out["substitute"] == {"action": 0, "label": "a0", "choice": "move alpha"}
    assert out["chosen"]["action"] == _CHOSEN and out["chosen"]["choice"] == "move beta"
    assert out["recorded_result"] == "loss" and out["turn"] == _TURN and out["trainee_side"] == "p1"
    assert any("Single realized-dice line" in c for c in out["caveats"])
    assert calls[0]["post_t_seeds"] == [None]               # the battle's own dice
    assert calls[0]["divergence_turn"] == _TURN and calls[0]["substitute_action"] == 0
    assert out["engine"] == "rust_core" and out["play_out"]["opponent_recorded_choice"] == "move opp"


def test_monte_carlo_aggregates_win_rate_and_seeds(monkeypatch):
    calls = _install(monkeypatch, ["win", "win", "loss", "win"])
    out = _run(6, n_rollouts=4)
    assert out["n_rollouts"] == 4
    assert out["wins"] == 3 and out["losses"] == 1
    assert out["win_rate"] == 0.75
    assert out["outcomes"] == {"loss": 1, "win": 3}
    lo, hi = out["win_rate_ci"]
    assert 0.0 < lo < 0.75 < hi <= 1.0
    # Each rollout got a DISTINCT post-divergence reseed (Monte-Carlo over dice), from the battle + decision.
    seeds = calls[0]["post_t_seeds"]
    assert len(seeds) == 4 and len(set(seeds)) == 4 and None not in seeds
    assert seeds == RP.fresh_seeds(4, salt=f"{_TAG}:{_INV}:cf")


def test_a_roster_bot_is_the_in_core_port_seeded_from_the_decision(monkeypatch):
    calls = _install(monkeypatch, ["loss"])
    out = _run(0, opp_name="staller", n_rollouts=2)
    kw = calls[0]
    assert out["opponent_source"] == "bot:staller"
    assert kw["opp_bot"] == {"name": "staller", "seed": RP.bot_seed(_TAG, _INV)} and kw["opp_policy"] is None
    assert kw["stall_sides"] == ["p1"]                       # a bot never stall-forfeits (F-LF-5)
    assert any("in-core Rust port" in c for c in out["caveats"])
    assert 0 <= RP.bot_seed(_TAG, _INV) < 2 ** 53
    assert RP.bot_seed(_TAG, _INV) != RP.bot_seed(_TAG, _INV + 1)


def test_a_checkpoint_opponent_plays_the_recorded_regime_with_a_seeded_sampler(monkeypatch):
    calls = _install(monkeypatch, ["win"])
    opp = _model()
    out = _run(0, opp_model=opp, opponent_ckpt="snap.zip", n_rollouts=3)
    kw = calls[0]
    assert out["opponent_source"] == "ckpt_stochastic:snap.zip"
    pol = kw["opp_policy"]
    assert pol.model is opp and pol.stochastic and pol.seeds == RP.sampler_seeds(_TAG, _INV, 3)
    assert len(set(pol.seeds)) == 3
    assert kw["stall_sides"] == ["p1", "p2"]                 # an RLPlayer opponent stall-forfeits too
    assert kw["our_policy"].stochastic is False and kw["opp_bot"] is None
    # greedy override: flagged as biased LOW
    calls.clear()
    out = _run(0, opp_model=opp, opponent_ckpt="snap.zip", opponent_stochastic=False)
    assert out["opponent_source"] == "ckpt_greedy:snap.zip" and calls[0]["opp_policy"].stochastic is False
    assert "biased LOW" in out["caveats"][0]


def test_self_model_fallback_and_the_source_refusals(monkeypatch):
    calls = _install(monkeypatch, ["win"])
    me = _model()
    out = _run(0, play_model=me, opp_name="sentinel_3")
    assert out["opponent_source"] == "self_model_approx" and "OWN model" in out["caveats"][0]
    assert calls[0]["opp_policy"].model is me and calls[0]["opp_policy"].stochastic
    with pytest.raises(ValueError, match="not a known bot"):
        _run(0, opp_name="sentinel_3", opponent_source="bot")
    with pytest.raises(ValueError, match="requires --opponent-ckpt"):
        _run(0, opponent_source="ckpt")
    with pytest.raises(ValueError, match="unknown opponent_source"):
        _run(0, opponent_source="nope")


def test_rejects_non_move_selection(monkeypatch):
    _install(monkeypatch, ["win"])
    summary = _summary()
    summary["invocations"][_INV]["phase"] = "force_switch"
    with pytest.raises(ValueError, match="move round"):
        replay_counterfactual_battle(_record(), summary, _npz(), _INV, 0, play_model=_model(),
                                     opp_name="heuristic", core=_Core())


def test_rejects_illegal_substitute_a_turn_mismatch_and_a_foreign_root(monkeypatch):
    _install(monkeypatch, ["win"])
    with pytest.raises(ValueError, match="not legal"):
        _run(9)                                               # 9 not in _CHOICES
    _install(monkeypatch, ["win"], walk_turn=_TURN + 1)
    with pytest.raises(ValueError, match="disagree"):
        _run(0)
    _install(monkeypatch, ["win"], root_tokens={0: "move other"})
    with pytest.raises(RuntimeError, match="not the decision"):
        _run(0)


def test_obs_version_mismatch_is_refused_up_front_and_an_owned_core_is_closed(monkeypatch):
    calls = _install(monkeypatch, ["win"])
    core = _Core()
    with pytest.raises(ValueError, match="obs-version mismatch"):
        _run(0, play_model=_model(2669), core=core)
    assert not calls
    # an injected core is the caller's: not closed here
    _run(0, core=core)
    assert not core.closed


def test_missing_actions_and_out_of_range(monkeypatch):
    _install(monkeypatch, ["win"])
    with pytest.raises(ValueError, match="no 'actions' array"):
        replay_counterfactual_battle(_record(), _summary(), {}, _INV, 0, play_model=_model(), opp_name="heuristic",
                                     core=_Core())
    with pytest.raises(IndexError):
        replay_counterfactual_battle(_record(), _summary(), _npz(), 99, 0, play_model=_model(), opp_name="heuristic",
                                     core=_Core())


def test_narrate_captures_the_first_win_and_the_first_loss(monkeypatch):
    calls = _install(monkeypatch, ["loss", "win", "win", "loss"])
    out = _run(0, n_rollouts=4, narrate=True)
    assert calls[0]["text"] is True
    assert out["losing_trajectory"] == [{"turn": 9, "events": ["we used Move0"]}]     # rollout 0
    assert out["winning_trajectory"] == [{"turn": 10, "events": ["we used Move1"]}]   # rollout 1
    calls.clear()
    out = _run(0, n_rollouts=2, narrate=False)
    assert calls[0]["text"] is False and out["winning_trajectory"] is None and out["losing_trajectory"] is None


def test_impl_node_is_said_not_obeyed(monkeypatch):
    _install(monkeypatch, ["win"])
    assert any("--impl node does not apply" in c for c in _run(0, impl="node")["caveats"])
    assert not any("--impl" in c for c in _run(0, impl="rust")["caveats"])
