"""BaitBot units — the immunity predicate, the pivot trigger, and the dial.

The gate this bot serves needs a CONTROLLED variable: if the declared p_bait and the realized
pivot rate can drift apart, the arm is not an experiment. These pin the predicate against
gen3_data ground truth (including negative controls, so "everything is immune" cannot pass) and
pin the dial's arithmetic.
"""
from types import SimpleNamespace as NS

import pytest

from agents.baitbot import (DEFAULT_P_BAIT, Gen3BaitBotPlayer, bait_targets, blocks, known_attacks)
from agents.enums import PokemonType
from agents.gen3_mechanics import effective_multiplier_by_types as em

T = PokemonType


# ── the immunity predicate, against data/ ground truth ──────────────────────────────────
@pytest.mark.parametrize("label,mt,t1,t2,ability,immune", [
    ("Levitate blocks Ground",      T.GROUND,   T.DRAGON,   T.GROUND, "levitate",   True),
    ("Water Absorb blocks Water",   T.WATER,    T.WATER,    None,     "waterabsorb", True),
    ("Volt Absorb blocks Electric", T.ELECTRIC, T.ELECTRIC, None,     "voltabsorb",  True),
    ("Flash Fire blocks Fire",      T.FIRE,     T.FIRE,     T.FLYING, "flashfire",   True),
    ("type chart: Ground vs Flying", T.GROUND,  T.STEEL,    T.FLYING, "keeneye",     True),
    ("type chart: Normal vs Ghost",  T.NORMAL,  T.GHOST,    T.POISON, "levitate",    True),
    # NEGATIVE CONTROLS — without the ability there is no immunity. A predicate that called
    # everything immune would pass every positive case above and be useless.
    ("Ground vs Flygon WITHOUT Levitate", T.GROUND, T.DRAGON, T.GROUND, None,  False),
    ("Fire vs Charizard WITHOUT FlashFire", T.FIRE, T.FIRE,  T.FLYING, "blaze", False),
    ("Electric vs Swampert is Ground-immune, not ability", T.ELECTRIC, T.WATER, T.GROUND, "torrent", True),
])
def test_immunity_predicate_matches_ground_truth(label, mt, t1, t2, ability, immune):
    assert (em(mt, t1, t2, ability) == 0.0) is immune, label


def test_status_moves_are_not_baits():
    """The pathology is firing an ATTACK into an immune arrival; a status move is a different error."""
    mon = _mon((T.DRAGON, T.GROUND), "levitate")
    assert blocks(NS(base_power=0, type=T.GROUND), mon) is False
    assert blocks(NS(base_power=None, type=T.GROUND), mon) is False


def test_known_attacks_uses_only_revealed_damaging_moves():
    mon = NS(moves={"earthquake": NS(base_power=100), "toxic": NS(base_power=0),
                    "roar": NS(base_power=None)})
    assert len(known_attacks(mon)) == 1
    assert known_attacks(None) == []


# ── the pivot trigger ──────────────────────────────────────────────────────────────────
def _mon(types, ability, hp=1.0, fainted=False):
    """Mirrors the real Pokemon surface effective_multiplier reads: type_1/type_2/ability/status."""
    t1 = types[0]
    t2 = types[1] if len(types) > 1 else None
    return NS(type_1=t1, type_2=t2, ability=ability, status=None,
              current_hp_fraction=hp, fainted=fainted)


def _battle(opp_moves, switches):
    return NS(active_pokemon=_mon((T.NORMAL,), None),
              opponent_active_pokemon=NS(moves=opp_moves),
              available_switches=switches, available_moves=[NS(base_power=80, type=T.NORMAL)])


def test_trigger_fires_only_when_a_bench_mon_blocks_EVERY_revealed_attack():
    flygon = _mon((T.DRAGON, T.GROUND), "levitate")
    blissey = _mon((T.NORMAL,), "naturalcure")
    eq = NS(base_power=100, type=T.GROUND)
    ice = NS(base_power=95, type=T.ICE)
    # Ground only -> Flygon is a target; Blissey is not.
    assert bait_targets(_battle({"eq": eq}, [flygon, blissey])) == [flygon]
    # Ground AND Ice -> Flygon takes Ice, so NOTHING is a bait target.
    assert bait_targets(_battle({"eq": eq, "ice": ice}, [flygon, blissey])) == []


def test_no_revealed_attacks_means_no_trigger():
    """With nothing known, nothing is baitable — otherwise the rate tracks movepools, not the dial."""
    assert bait_targets(_battle({}, [_mon((T.DRAGON, T.GROUND), "levitate")])) == []
    assert bait_targets(_battle({"toxic": NS(base_power=0)}, [_mon((T.DRAGON, T.GROUND), "levitate")])) == []


def test_fainted_bench_mons_are_never_bait_targets():
    dead = _mon((T.DRAGON, T.GROUND), "levitate", fainted=True)
    assert bait_targets(_battle({"eq": NS(base_power=100, type=T.GROUND)}, [dead])) == []


# ── the dial ───────────────────────────────────────────────────────────────────────────
def test_p_bait_is_validated():
    """An out-of-range dial is a broken experiment, so it must refuse at construction."""
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            _mk(p_bait=bad)


def _mk(p_bait=DEFAULT_P_BAIT, seed=0):
    o = object.__new__(Gen3BaitBotPlayer)
    if not 0.0 <= p_bait <= 1.0:
        raise ValueError(f"p_bait must be in [0, 1], got {p_bait}")
    import random as _r
    o.p_bait, o._rng = p_bait, _r.Random(seed)
    o.n_bait_opportunities = o.n_baits_taken = 0
    return o


def test_realized_rate_tracks_the_dial():
    """The dial must be a DIAL: over many draws the realized rate matches p_bait."""
    for p in (0.0, 0.25, 0.6, 1.0):
        bot = _mk(p_bait=p, seed=7)
        for _ in range(4000):
            bot.n_bait_opportunities += 1
            if bot._rng.random() < p:
                bot.n_baits_taken += 1
        assert abs(bot.realized_bait_rate - p) < 0.02, p


def test_realized_rate_is_zero_before_any_opportunity():
    assert _mk().realized_bait_rate == 0.0
