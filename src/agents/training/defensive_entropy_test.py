"""Unit tests for gen3_defensive_entropy_v1 — the state-conditioned entropy boost on defensive-opportunity
decisions: the env's `_defensive_opportunity` flag logic + the PPO boost-anneal schedule.

The flag drives a per-decision entropy multiplier in the PPO loss (explore Recover/Soft-Boiled/Wish/Refresh/
Heal Bell more, without touching the reward). These exercise the *novel* logic in isolation (the loss-side
weighting is integration-tested by the serverless smoke)."""
import types

import torch as th

from agents.training.gen3_env import Gen3Env, _DEFENSIVE_HEAL_HP
from agents.training.instrumented_ppo import InstrumentedMaskablePPO as PPO


# ----------------------------------------------------------------- the env flag
def _mon(hp, status=None):
    return types.SimpleNamespace(current_hp_fraction=hp, status=status)


def _move(mid):
    return types.SimpleNamespace(id=mid)


def _flag(active, moves, team_statuses=()):
    team = {f"t{i}": types.SimpleNamespace(status=s) for i, s in enumerate(team_statuses)}
    b = types.SimpleNamespace(active_pokemon=active, available_moves=list(moves), team=team)
    return Gen3Env._defensive_opportunity(types.SimpleNamespace(battle1=b))


def test_heal_move_fires_only_when_damaged():
    # Soft-Boiled legal + HP below the threshold → opportunity.
    assert _flag(_mon(0.5), [_move("softboiled"), _move("icebeam")]) == 1.0
    assert _flag(_mon(0.4), [_move("recover")]) == 1.0
    # Near-full HP → too little to restore → no opportunity (the accepted Wish-at-full-HP miss).
    assert _flag(_mon(0.95), [_move("softboiled")]) == 0.0
    assert _flag(_mon(_DEFENSIVE_HEAL_HP + 0.01), [_move("softboiled")]) == 0.0


def test_self_cure_fires_only_when_statused():
    assert _flag(_mon(1.0, status="brn"), [_move("refresh"), _move("return")]) == 1.0   # Refresh + a status
    assert _flag(_mon(1.0, status=None), [_move("refresh")]) == 0.0                      # no status → nothing to cure


def test_team_cure_fires_only_when_a_teammate_statused():
    # Heal Bell legal + a bench mon statused → opportunity (party-scoped).
    assert _flag(_mon(1.0), [_move("healbell")], team_statuses=(None, "tox", None)) == 1.0
    assert _flag(_mon(1.0), [_move("aromatherapy")], team_statuses=("par",)) == 1.0
    assert _flag(_mon(1.0), [_move("healbell")], team_statuses=(None, None)) == 0.0


def test_no_defensive_move_or_no_moves_is_zero():
    assert _flag(_mon(0.3), [_move("earthquake"), _move("rockslide")]) == 0.0    # damaged but only attacks
    assert _flag(_mon(0.3), []) == 0.0                                           # forced switch (no moves)
    assert _flag(None, [_move("softboiled")]) == 0.0                             # no active mon


def test_flag_never_raises_on_garbage():
    # Hot path: a malformed battle must yield 0.0, never crash.
    assert Gen3Env._defensive_opportunity(types.SimpleNamespace(battle1=None)) == 0.0
    bad = types.SimpleNamespace(active_pokemon=object(), available_moves=[_move("softboiled")], team={})
    assert Gen3Env._defensive_opportunity(types.SimpleNamespace(battle1=bad)) == 0.0


# ----------------------------------------------------------------- the boost-anneal schedule
def _beff(B, af, progress_remaining):
    fs = types.SimpleNamespace(defensive_entropy_boost=B, defensive_entropy_anneal_frac=af,
                               _current_progress_remaining=progress_remaining)
    return PPO._defensive_entropy_boost_eff(fs)


def test_boost_constant_when_off_or_no_anneal():
    assert _beff(1.0, 0.0, 1.0) == 1.0          # OFF
    assert _beff(1.0, 0.5, 0.5) == 1.0          # OFF stays off regardless of anneal
    assert _beff(3.0, 0.0, 0.5) == 3.0          # constant boost (no anneal)


def test_boost_anneals_linearly_to_one():
    # B=3, anneal over the first 50% of training. progress_remaining 1.0→0.0 ⇒ done 0.0→1.0.
    assert abs(_beff(3.0, 0.5, 1.00) - 3.0) < 1e-6     # start (done 0.0)
    assert abs(_beff(3.0, 0.5, 0.75) - 2.0) < 1e-6     # done 0.25 (halfway through the anneal)
    assert abs(_beff(3.0, 0.5, 0.50) - 1.0) < 1e-6     # done 0.50 (anneal complete)
    assert abs(_beff(3.0, 0.5, 0.20) - 1.0) < 1e-6     # past the anneal → stays 1.0


# ----------------------------------------------------------------- the weighting invariant
def test_weighting_is_identity_off_flag_and_off_boost():
    """The exact loss-side expression: a weight of 1 off-flag (and everywhere at boost 1) ⇒ unweighted."""
    ent = th.rand(16)
    flag = (th.arange(16) % 2).float()
    unweighted = -th.mean(ent)
    # boost 1 (off) → weight ≡ 1 regardless of flag
    assert th.allclose(-th.mean((1.0 + (1.0 - 1.0) * flag) * ent), unweighted)
    # boost 3 but no flagged decisions → weight ≡ 1
    assert th.allclose(-th.mean((1.0 + (3.0 - 1.0) * th.zeros(16)) * ent), unweighted)
    # boost 3 with flags → strictly more entropy bonus (larger magnitude)
    assert float(-(-th.mean((1.0 + (3.0 - 1.0) * flag) * ent))) > float(-unweighted)
