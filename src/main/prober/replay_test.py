"""Pure unit tests for the counterfactual replay ORCHESTRATION (win-rate aggregation, opponent
resolution, Monte-Carlo seed plumbing) — the bridge runner, the models and the players are
monkeypatched, so no Node, no torch. The real scripted-prefix faithfulness is proven against real
battles in ``utils/bridge/counterfactual_fuzz_test.py``."""

from types import SimpleNamespace

import numpy as np
import pytest

import main.prober.replay as RP
from agents.training.obs_materializer import MaterializedDecision, MaterializedTrace
from main.prober.replay import build_opponent, replay_counterfactual_battle, wilson_ci

_CHOICES = {0: "move alpha", 1: "move beta", 6: "switch gamma"}
_TURN = 8
_INV = 4
_CHOSEN = 1


def _record():
    return SimpleNamespace(
        battle_tag="battle-gen3ou-1", trainee_username="trainee", format_id="gen3ou",
        side_of=lambda name: "p1", username=lambda side: "trainee",
        packed_team=lambda side: "TEAM", start_options=lambda: {"seed": [1, 2, 3, 4]})


def _summary():
    acts = {f"a{i}": {"valid": True} for i in range(11)}
    invs = [{"phase": "move_selection", "turn": _TURN, "actions": acts} for _ in range(_INV + 2)]
    return {"invocations": invs, "meta": {"result": "LOSS"}}


def _npz():
    actions = np.zeros(_INV + 2, dtype=int)
    actions[_INV] = _CHOSEN
    return {"actions": actions}


def _install(monkeypatch, outcomes):
    """Outcomes is a list cycled across rollouts; record the post_t_seed each rollout saw."""
    seeds_seen = []

    def fake_mat(record, *, actions, mappings=None, map_actions_at=None, stop_after_decision=None,
                 impl="node"):
        decisions = [MaterializedDecision(np.zeros(2, np.float32), np.ones(11, np.int8), _TURN)
                     for _ in range(_INV + 1)]
        return MaterializedTrace(decisions=decisions, actions_complete=True, action_choices=dict(_CHOICES))

    counter = {"i": 0}

    def fake_run_one(record, *, trainee, opponent, divergence_turn, substitute_choice, seed=None,
                     post_t_seed=None, capture_trajectory=False, impl="node"):
        seeds_seen.append(post_t_seed)
        o = outcomes[counter["i"] % len(outcomes)]
        counter["i"] += 1
        res = {"outcome": o, "ended": True}
        if capture_trajectory:
            res["trajectory"] = [{"turn": 1, "events": [f"we used move ({o})"]}]
        return res

    monkeypatch.setattr(RP, "materialize_from_record", fake_mat)
    monkeypatch.setattr(RP, "build_trainee", lambda *a, **k: SimpleNamespace(kind="trainee"))
    monkeypatch.setattr(RP, "build_opponent",
                        lambda *a, **k: (SimpleNamespace(kind="opp"), "bot:heuristic"))
    monkeypatch.setattr(RP, "_run_one", fake_run_one)
    return seeds_seen


def test_wilson_ci_bounds():
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = wilson_ci(5, 10)
    assert 0.0 < lo < 0.5 < hi < 1.0
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and lo < 1.0


def test_single_line_is_not_a_probability(monkeypatch):
    _install(monkeypatch, ["win"])
    out = replay_counterfactual_battle(_record(), _summary(), _npz(), _INV, 0,
                                       play_model=object(), opp_name="heuristic",
                                       mappings=None, server_config=None, n_rollouts=1)
    assert out["n_rollouts"] == 1 and out["deterministic_line"] is True
    assert out["win_rate"] == 1.0 and out["wins"] == 1
    assert out["substitute"]["action"] == 0 and out["substitute"]["choice"] == "move alpha"
    assert out["chosen"]["action"] == _CHOSEN
    assert out["recorded_result"] == "loss"
    assert any("Single realized-dice line" in c for c in out["caveats"])


def test_monte_carlo_aggregates_win_rate_and_seeds(monkeypatch):
    seeds_seen = _install(monkeypatch, ["win", "win", "loss", "win"])
    out = replay_counterfactual_battle(_record(), _summary(), _npz(), _INV, 6,
                                       play_model=object(), opp_name="heuristic",
                                       mappings=None, server_config=None, n_rollouts=4)
    assert out["n_rollouts"] == 4
    assert out["wins"] == 3 and out["losses"] == 1
    assert out["win_rate"] == 0.75
    assert out["outcomes"] == {"loss": 1, "win": 3}
    lo, hi = out["win_rate_ci"]
    assert 0.0 < lo < 0.75 < hi <= 1.0
    # Each rollout got a DISTINCT post-divergence reseed (Monte-Carlo over dice).
    assert len(seeds_seen) == 4 and len(set(seeds_seen)) == 4 and None not in seeds_seen


def test_rejects_non_move_selection(monkeypatch):
    _install(monkeypatch, ["win"])
    summary = _summary()
    summary["invocations"][_INV]["phase"] = "force_switch"
    with pytest.raises(ValueError, match="move round"):
        replay_counterfactual_battle(_record(), summary, _npz(), _INV, 0,
                                     play_model=object(), opp_name="heuristic",
                                     mappings=None, server_config=None, n_rollouts=1)


def test_rejects_illegal_substitute(monkeypatch):
    _install(monkeypatch, ["win"])
    with pytest.raises(ValueError, match="not legal"):
        replay_counterfactual_battle(_record(), _summary(), _npz(), _INV, 9,   # 9 not in _CHOICES
                                     play_model=object(), opp_name="heuristic",
                                     mappings=None, server_config=None, n_rollouts=1)


# --- opponent resolution (build_opponent) ---

class _FakeRL:
    def __init__(self, *, model, stochastic, **kw):
        self.model, self.stochastic = model, stochastic


class _FakeBot:
    def __init__(self, **kw):
        self.kw = kw


@pytest.fixture
def _patch_players(monkeypatch):
    import agents.inference.player as player_mod
    monkeypatch.setattr(player_mod, "RLPlayer", _FakeRL)
    monkeypatch.setattr(RP, "_bot_classes", lambda: {"heuristic": _FakeBot})


def test_build_opponent_prefers_explicit_ckpt(_patch_players):
    opp_model = object()
    player, label = build_opponent("heuristic", _record(), play_model=object(), mappings=None,
                                   server_config=None, opponent_ckpt="/c.zip", opp_model=opp_model,
                                   opponent_source="auto", tag="t")
    assert label == "ckpt:/c.zip"
    assert isinstance(player, _FakeRL) and player.model is opp_model and player.stochastic is False


def test_build_opponent_rebuilds_known_bot(_patch_players):
    player, label = build_opponent("heuristic", _record(), play_model=object(), mappings=None,
                                   server_config=None, opponent_source="auto", tag="t")
    assert label == "bot:heuristic" and isinstance(player, _FakeBot)


def test_build_opponent_self_model_fallback(_patch_players):
    play = object()
    player, label = build_opponent("sentinel_2", _record(), play_model=play, mappings=None,
                                   server_config=None, opponent_source="auto", tag="t")
    assert label == "self_model_approx"
    assert isinstance(player, _FakeRL) and player.model is play and player.stochastic is True


def test_build_opponent_bot_source_rejects_unknown(_patch_players):
    with pytest.raises(ValueError, match="not a known bot"):
        build_opponent("sentinel_2", _record(), play_model=object(), mappings=None,
                       server_config=None, opponent_source="bot", tag="t")


def test_build_opponent_ckpt_source_requires_ckpt(_patch_players):
    # An explicit --opponent-source ckpt without --opponent-ckpt must RAISE (not silently self-model).
    with pytest.raises(ValueError, match="requires --opponent-ckpt"):
        build_opponent("heuristic", _record(), play_model=object(), mappings=None,
                       server_config=None, opponent_source="ckpt", opp_model=None, tag="t")


def test_obs_version_mismatch_raises(monkeypatch):
    # A play model trained on a different obs dim than the (current-encoder) materialized obs → loud raise.
    _install(monkeypatch, ["win"])

    class _OS:
        def __getitem__(self, k):
            return SimpleNamespace(shape=(9999,))            # model expects 9999, the fake obs is dim 2

    play = SimpleNamespace(observation_space=_OS())
    with pytest.raises(ValueError, match="obs-version mismatch"):
        replay_counterfactual_battle(_record(), _summary(), _npz(), _INV, 0,
                                     play_model=play, opp_name="heuristic",
                                     mappings=None, server_config=None, n_rollouts=1)


def test_self_model_fallback_caveat_in_aggregate(monkeypatch):
    _install(monkeypatch, ["loss"])
    monkeypatch.setattr(RP, "build_opponent",
                        lambda *a, **k: (SimpleNamespace(kind="opp"), "self_model_approx"))
    out = replay_counterfactual_battle(_record(), _summary(), _npz(), _INV, 0,
                                       play_model=object(), opp_name="sentinel_2",
                                       mappings=None, server_config=None, n_rollouts=1)
    assert out["opponent_source"] == "self_model_approx"
    assert any("self-play APPROXIMATION" in c for c in out["caveats"])


def test_narrate_captures_first_win_and_loss_trajectory(monkeypatch):
    _install(monkeypatch, ["loss", "win", "win"])
    out = replay_counterfactual_battle(_record(), _summary(), _npz(), _INV, 6,
                                       play_model=object(), opp_name="heuristic",
                                       mappings=None, server_config=None, n_rollouts=3, narrate=True)
    assert out["winning_trajectory"] is not None and out["losing_trajectory"] is not None
    assert "win" in out["winning_trajectory"][0]["events"][0]
    assert "loss" in out["losing_trajectory"][0]["events"][0]


def test_no_narrate_no_trajectory(monkeypatch):
    _install(monkeypatch, ["win"])
    out = replay_counterfactual_battle(_record(), _summary(), _npz(), _INV, 6,
                                       play_model=object(), opp_name="heuristic",
                                       mappings=None, server_config=None, n_rollouts=1)
    assert out["winning_trajectory"] is None and out["losing_trajectory"] is None
