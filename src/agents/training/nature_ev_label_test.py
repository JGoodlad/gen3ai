"""The NATURE / EV belief labels are the opponent's TRUE declared spread (`gen3_true_spread_labels_v1`).

Two defects this pins (M5 Lane C F-LC-5 / F-LC-6, 2026-09-29):

* **F-LC-5 — coverage.** The label used to be INVERTED from the derived stats assuming IV 31. The
  training pool's Hidden Power sets carry IV-30 stats (``fix_gen3_hp_ivs``), so an IV-30 stat with
  0 EVs has no IV-31 decomposition and the slot went unlabelled: 2,406 of the pool's 4,314 mons
  (55.8 %; 54.6 % of revealed-slot decisions). Where it DID invert an IV-30 mon, the EVs (and for
  17 mons the nature) were wrong. The label is now read from the set, so every pool mon is labelled
  and the label IS the set.
* **F-LC-6 — staleness.** ``Gen3Env._nature_ev_map`` cached per battle keyed by the opponent's
  species SET, so a next opponent with the same six species and a different spread got the previous
  team's labels. The cache is gone.

Every test here fails on a revert of ``Gen3Env._nature_ev_map`` to the inversion / the cache.
"""
from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

from agents import gen3_data
from agents.model.belief_tables import SpreadLabelError, true_nature_ev_label

_COLS = ("atk", "def", "spa", "spd", "spe")
_IDX = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}


def _server_stats(species: str, evs, ivs, level: int, nature: str) -> dict:
    """Showdown's `spreadModify` + `natureModify` (sim/dex-species / sim/battle), written out here so
    the test does not share the label's own formula: floor((2b + iv + floor(ev/4)) * L / 100) + 5,
    then floor(x * 110 / 100) / floor(x * 90 / 100)."""
    sd = gen3_data.species.get(species)
    nd = gen3_data.natures.raw()[nature]
    out = {}
    for stat in _COLS:
        k = _IDX[stat]
        x = (2 * int(sd.base_stats[stat]) + int(ivs[k]) + int(evs[k]) // 4) * level // 100 + 5
        m = float(nd.get(stat, 1.0))
        out[stat] = x * 110 // 100 if m > 1 else x * 90 // 100 if m < 1 else x
    return out


def _mon(tb) -> SimpleNamespace:
    """A truth mon as poke-env holds it after ``backfill_spread_from_teambuilder``: species, the
    request's stats, and the declared spread (nature lower-cased, ``serious`` when absent)."""
    from poke_env.data.normalize import to_id_str

    species = to_id_str(tb.species or tb.nickname)
    nature = tb.nature.lower() if tb.nature is not None else "serious"
    return SimpleNamespace(species=species, nature=nature, evs=list(tb.evs), ivs=list(tb.ivs),
                           stats=_server_stats(species, tb.evs, tb.ivs, tb.level or 100, nature))


def _battle(packed: str) -> SimpleNamespace:
    from poke_env.teambuilder.teambuilder import Teambuilder

    return SimpleNamespace(team={f"p2: {i}": _mon(tb) for i, tb in enumerate(Teambuilder.parse_packed_team(packed))})


@functools.lru_cache(maxsize=1)
def _env():
    from poke_env import AccountConfiguration

    from agents.observation.state_encoder import load_mappings
    from agents.training.gen3_env import Gen3Env

    return Gen3Env(load_mappings(), battle_format="gen3ou", account_configuration1=AccountConfiguration("NatEvLbl", None),
                   start_listening=False, emit_spread_labels=True)


def _expected(m) -> tuple:
    return (int(gen3_data.natures.raw()[m.nature]["num"]), [int(m.evs[_IDX[s]]) // 4 * 4 for s in _COLS])


# --------------------------------------------------------------------------- F-LC-5: coverage

@pytest.mark.integration
def test_every_pool_mon_gets_its_true_nature_ev_label():
    """All 719 pool teams (through the TRAINING teambuilder's packing, the HP IV fix included): every
    mon is labelled — coverage 100 %, at least the ladder's 98.8 % — and the label is the set.
    Under the IV-31 inversion 2,406 of 4,314 mons went unlabelled."""
    from main.rust_core_cutover.envs import packed_teams

    teams = packed_teams("pool")
    assert len(teams) == 719, len(teams)
    env = _env()
    n_mons = n_labelled = n_iv30 = 0
    wrong = []
    for packed in teams:
        b2 = _battle(packed)
        got = env._nature_ev_map(b2)
        for m in b2.team.values():
            n_mons += 1
            n_iv30 += any(m.ivs[_IDX[s]] != 31 for s in _COLS)
            if m.species in got:
                n_labelled += 1
                if (got[m.species][0], list(got[m.species][1])) != _expected(m):
                    wrong.append((m.species, got[m.species], _expected(m)))
    assert n_mons == 4314, n_mons
    assert n_iv30 > 2000, n_iv30          # the population the old inversion dropped is still present
    assert n_labelled == n_mons, f"{n_mons - n_labelled} of {n_mons} pool mons carry no nature/EV label"
    assert not wrong, wrong[:5]


# --------------------------------------------------------------------------- F-LC-6: no stale cache

_PAIR_A = "Tyranitar||leftovers|sandstream|rockslide,earthquake,hiddenpowerbug,dragondance|Adamant|252,252,,,4,|||||]" \
          "Skarmory||leftovers|keeneye|drillpeck,spikes,roar,toxic|Impish|252,,252,,4,|||||"
_PAIR_B = "Tyranitar||leftovers|sandstream|crunch,fireblast,icebeam,pursuit|Modest|252,,,252,4,|||||]" \
          "Skarmory||leftovers|keeneye|drillpeck,spikes,roar,toxic|Careful|252,,4,,252,|||||"


def test_a_same_species_opponent_with_a_different_spread_gets_its_own_labels():
    """Two opponents with the SAME species set and DIFFERENT spreads, back to back on ONE env: the second
    must be labelled from its own team. The species-set-keyed cache served it the first team's labels."""
    env = _env()
    a, b = _battle(_PAIR_A), _battle(_PAIR_B)
    assert {m.species for m in a.team.values()} == {m.species for m in b.team.values()}
    got_a = env._nature_ev_map(a)
    got_b = env._nature_ev_map(b)
    for battle, got in ((a, got_a), (b, got_b)):
        for m in battle.team.values():
            assert (got[m.species][0], list(got[m.species][1])) == _expected(m), (m.species, got[m.species])
    assert got_a["tyranitar"] != got_b["tyranitar"] and got_a["skarmory"] != got_b["skarmory"]


# --------------------------------------------------------------------------- the throwing guard

def _one(species="tyranitar", nature="adamant", evs=(4, 252, 0, 0, 0, 252), ivs=(31,) * 6):
    stats = _server_stats(species, evs, ivs, 100, nature)
    base = [float(gen3_data.species.get(species).base_stats[s]) for s in _COLS]
    return species, [float(stats[s]) for s in _COLS], base


def test_the_label_is_the_declared_set_including_hp_ivs():
    # Hidden Power Bug's IVs (atk 30, def 30, spd 30): the IV-31 inversion has no answer for def / spd at 0 EVs
    ivs = (31, 30, 30, 31, 30, 31)
    sid, derived, base = _one(ivs=ivs)
    assert true_nature_ev_label(sid, derived, base, "adamant", [4, 252, 0, 0, 0, 252], list(ivs)) == \
        (int(gen3_data.natures.raw()["adamant"]["num"]), [252, 0, 0, 0, 252])
    # an EV that is not a multiple of 4 is labelled at its stat-effective value; a missing nature is neutral
    sid, derived, base = _one(nature="serious", evs=(0, 85, 0, 0, 170, 255))
    assert true_nature_ev_label(sid, derived, base, None, [0, 85, 0, 0, 170, 255], [31] * 6)[1] == [84, 0, 0, 168, 252]


def test_a_spread_that_does_not_reproduce_the_stats_raises():
    sid, derived, base = _one()
    with pytest.raises(SpreadLabelError, match="server's stats"):
        true_nature_ev_label(sid, derived, base, "jolly", [4, 252, 0, 0, 0, 252], [31] * 6)
    with pytest.raises(SpreadLabelError, match="server's stats"):          # the old IV-31 assumption, caught
        true_nature_ev_label(sid, derived, base, "adamant", [4, 252, 0, 0, 0, 252], [31, 30, 30, 31, 30, 31])
    with pytest.raises(SpreadLabelError, match="no declared spread"):
        true_nature_ev_label(sid, derived, base, "adamant", None, None)
    with pytest.raises(SpreadLabelError, match="unknown nature"):
        true_nature_ev_label(sid, derived, base, "grumpy", [4, 252, 0, 0, 0, 252], [31] * 6)
    with pytest.raises(SpreadLabelError, match="six long"):
        true_nature_ev_label(sid, derived, base, "adamant", [252, 0, 0, 0, 252], [31] * 6)


def test_the_env_raises_rather_than_label_a_mismatched_spread():
    env = _env()
    b2 = _battle(_PAIR_A)
    next(iter(b2.team.values())).nature = "jolly"            # the declared spread no longer matches the stats
    with pytest.raises(SpreadLabelError):
        env._nature_ev_map(b2)
