import numpy as np
import pytest
from unittest.mock import MagicMock

from poke_env.battle.effect import Effect
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

from agents.gen3_mechanics import (
    ABILITY_TYPE_MULTIPLIER,
    STATUS_MOVE_IMMUNITY,
    NOTABLE_EFFECTS,
    PHAZING_MOVES,
    INVULNERABLE_MOVES,
    STATUS_MOVES,
    RECOVERY_MOVES,
    HAZARD_CLEAR_MOVES,
    SETUP_MOVES,
    BOOST_STATS,
    BOOST_DIM,
    effective_multiplier,
    is_status_move_immune,
    status_move_lands,
    status_land_probability,
    status_land_estimate,
    ABILITY_STATUS_IMMUNITY,
    mon_status_str,
    boosts_array,
    boosts_str,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mon(type_1=PokemonType.NORMAL, type_2=None, ability=None, status=None, effects=None):
    m = MagicMock()
    m.type_1 = type_1
    m.type_2 = type_2
    m.ability = ability
    m.status = status
    m.effects = effects or {}
    m.boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}
    return m


# ---------------------------------------------------------------------------
# effective_multiplier
# ---------------------------------------------------------------------------

class TestEffectiveMultiplier:
    def test_levitate_blocks_ground(self):
        mon = _mon(type_1=PokemonType.GHOST, type_2=PokemonType.POISON, ability="levitate")
        assert effective_multiplier(PokemonType.GROUND, mon) == 0.0

    def test_levitate_does_not_block_other_types(self):
        mon = _mon(type_1=PokemonType.GHOST, type_2=PokemonType.POISON, ability="levitate")
        # Psychic is super-effective vs Poison (2×)
        assert effective_multiplier(PokemonType.PSYCHIC, mon) == 2.0

    def test_voltabsorb_blocks_electric(self):
        mon = _mon(type_1=PokemonType.WATER, ability="voltabsorb")
        assert effective_multiplier(PokemonType.ELECTRIC, mon) == 0.0

    def test_waterabsorb_blocks_water(self):
        mon = _mon(type_1=PokemonType.FIRE, ability="waterabsorb")
        assert effective_multiplier(PokemonType.WATER, mon) == 0.0

    def test_flashfire_blocks_fire(self):
        mon = _mon(type_1=PokemonType.STEEL, ability="flashfire")
        assert effective_multiplier(PokemonType.FIRE, mon) == 0.0

    def test_no_ability_uses_type_chart(self):
        # Ground is super-effective vs Poison (2×)
        mon = _mon(type_1=PokemonType.POISON, ability=None)
        assert effective_multiplier(PokemonType.GROUND, mon) == 2.0

    def test_none_ability_uses_type_chart(self):
        mon = _mon(type_1=PokemonType.NORMAL, ability=None)
        assert effective_multiplier(PokemonType.NORMAL, mon) == 1.0

    def test_case_insensitive_ability(self):
        mon = _mon(type_1=PokemonType.GHOST, ability="Levitate")
        assert effective_multiplier(PokemonType.GROUND, mon) == 0.0

    def test_immune_via_type_not_ability(self):
        # Ghost is immune to Normal (type chart, not ability)
        mon = _mon(type_1=PokemonType.GHOST, ability=None)
        assert effective_multiplier(PokemonType.NORMAL, mon) == 0.0


# ---------------------------------------------------------------------------
# is_status_move_immune
# ---------------------------------------------------------------------------

class TestIsStatusMoveImmune:
    def test_ground_immune_to_thunderwave(self):
        mon = _mon(type_1=PokemonType.GROUND)
        assert is_status_move_immune("thunderwave", mon) is True

    def test_steel_immune_to_toxic(self):
        mon = _mon(type_1=PokemonType.STEEL)
        assert is_status_move_immune("toxic", mon) is True

    def test_poison_immune_to_toxic(self):
        mon = _mon(type_1=PokemonType.POISON)
        assert is_status_move_immune("toxic", mon) is True

    def test_fire_immune_to_willowisp(self):
        mon = _mon(type_1=PokemonType.FIRE)
        assert is_status_move_immune("willowisp", mon) is True

    def test_already_statused_is_immune(self):
        mon = _mon(type_1=PokemonType.NORMAL, status=Status.PAR)
        assert is_status_move_immune("thunderwave", mon) is True

    def test_already_statused_blocks_any_status_move(self):
        mon = _mon(type_1=PokemonType.NORMAL, status=Status.BRN)
        assert is_status_move_immune("toxic", mon) is True

    def test_normal_type_not_immune_to_toxic(self):
        mon = _mon(type_1=PokemonType.NORMAL)
        assert is_status_move_immune("toxic", mon) is False

    def test_unknown_move_not_immune(self):
        mon = _mon(type_1=PokemonType.NORMAL)
        assert is_status_move_immune("tackle", mon) is False

    def test_sleep_move_no_type_immunity(self):
        # No type is immune to sleep moves by type alone
        mon = _mon(type_1=PokemonType.STEEL)
        assert is_status_move_immune("spore", mon) is False


# ---------------------------------------------------------------------------
# mon_status_str
# ---------------------------------------------------------------------------

class TestMonStatusStr:
    def test_none_mon_returns_none(self):
        assert mon_status_str(None) is None

    def test_clean_mon_returns_none(self):
        mon = _mon()
        assert mon_status_str(mon) is None

    def test_burn_status(self):
        mon = _mon(status=Status.BRN)
        assert mon_status_str(mon) == "BRN"

    def test_paralysis_status(self):
        mon = _mon(status=Status.PAR)
        assert mon_status_str(mon) == "PAR"

    def test_taunt_effect(self):
        mon = _mon(effects={Effect.TAUNT: 3})
        assert mon_status_str(mon) == "taunt"

    def test_confusion_effect(self):
        mon = _mon(effects={Effect.CONFUSION: 2})
        assert mon_status_str(mon) == "confusion"

    def test_combined_status_and_effect(self):
        mon = _mon(status=Status.BRN, effects={Effect.TAUNT: 2})
        result = mon_status_str(mon)
        assert result == "BRN, taunt"

    def test_only_notable_effects_shown(self):
        # An effect not in NOTABLE_EFFECTS should not appear
        from poke_env.battle.effect import Effect as E
        non_notable = next(e for e in Effect if e not in NOTABLE_EFFECTS)
        mon = _mon(effects={non_notable: 1})
        assert mon_status_str(mon) is None


# ---------------------------------------------------------------------------
# boosts_array
# ---------------------------------------------------------------------------

class TestBoostsArray:
    def test_none_mon_returns_zeros(self):
        arr = boosts_array(None)
        assert arr.shape == (BOOST_DIM,)
        assert arr.dtype == np.int8
        np.testing.assert_array_equal(arr, np.zeros(BOOST_DIM, dtype=np.int8))

    def test_all_zero_boosts(self):
        mon = _mon()
        arr = boosts_array(mon)
        np.testing.assert_array_equal(arr, np.zeros(BOOST_DIM, dtype=np.int8))

    def test_positive_boosts(self):
        mon = _mon()
        mon.boosts = {"atk": 2, "def": 0, "spa": 1, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}
        arr = boosts_array(mon)
        assert arr[BOOST_STATS.index("atk")] == 2
        assert arr[BOOST_STATS.index("spa")] == 1
        assert arr[BOOST_STATS.index("def")] == 0

    def test_negative_boosts(self):
        mon = _mon()
        mon.boosts = {"atk": 0, "def": -1, "spa": 0, "spd": -2, "spe": 0, "accuracy": 0, "evasion": 0}
        arr = boosts_array(mon)
        assert arr[BOOST_STATS.index("def")] == -1
        assert arr[BOOST_STATS.index("spd")] == -2

    def test_missing_boosts_default_to_zero(self):
        mon = MagicMock()
        mon.boosts = {"atk": 3}  # only atk set
        arr = boosts_array(mon)
        assert arr[BOOST_STATS.index("atk")] == 3
        assert arr[BOOST_STATS.index("def")] == 0

    def test_no_boosts_attribute(self):
        mon = MagicMock(spec=[])  # no boosts attr
        arr = boosts_array(mon)
        np.testing.assert_array_equal(arr, np.zeros(BOOST_DIM, dtype=np.int8))

    def test_dtype_is_int8(self):
        arr = boosts_array(None)
        assert arr.dtype == np.int8


# ---------------------------------------------------------------------------
# boosts_str
# ---------------------------------------------------------------------------

class TestBoostsStr:
    def test_none_mon_returns_none(self):
        assert boosts_str(None) is None

    def test_all_zero_returns_none(self):
        mon = _mon()
        assert boosts_str(mon) is None

    def test_positive_boost(self):
        mon = _mon()
        mon.boosts = {"atk": 2, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}
        assert boosts_str(mon) == "atk:+2"

    def test_mixed_boosts(self):
        mon = _mon()
        mon.boosts = {"atk": 2, "def": -1, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}
        result = boosts_str(mon)
        assert "atk:+2" in result
        assert "def:-1" in result


# ---------------------------------------------------------------------------
# Move category set sanity checks
# ---------------------------------------------------------------------------

class TestMoveSets:
    def test_phazing_moves(self):
        assert "roar" in PHAZING_MOVES
        assert "whirlwind" in PHAZING_MOVES

    def test_invulnerable_moves(self):
        assert "protect" in INVULNERABLE_MOVES
        assert "detect" in INVULNERABLE_MOVES
        assert "endure" in INVULNERABLE_MOVES

    def test_setup_moves(self):
        assert "swordsdance" in SETUP_MOVES
        assert "calmmind" in SETUP_MOVES
        assert "dragondance" in SETUP_MOVES

    def test_recovery_moves(self):
        assert "recover" in RECOVERY_MOVES
        assert "rest" in RECOVERY_MOVES
        assert "softboiled" in RECOVERY_MOVES

    def test_status_moves(self):
        assert "toxic" in STATUS_MOVES
        assert "willowisp" in STATUS_MOVES
        assert "thunderwave" in STATUS_MOVES

    def test_hazard_clear_moves(self):
        assert "rapidspin" in HAZARD_CLEAR_MOVES

    def test_boost_stats_length(self):
        assert BOOST_DIM == 7
        assert len(BOOST_STATS) == 7


# ---------------------------------------------------------------------------
# Effectiveness fast-path parity (precomputed chart + lru_cache)
#
# Proves the matchup-encoder optimization (precomputed _CHART + memoized _eff_cached +
# value-based effective_multiplier_by_types) is OBS-BYTE-IDENTICAL to the original
# damage_multiplier-based path — i.e. no ARCH_SIGNATURE bump / retrain is needed.
# ---------------------------------------------------------------------------

def _reference_eff(move_type, type_1, type_2, ability, status):
    """Replicates the pre-optimization effective_multiplier EXACTLY — type product taken
    straight from PokemonType.damage_multiplier — as the oracle the fast path must match."""
    from agents.gen3_mechanics import _type_chart
    ab = (ability or "").lower()
    base = move_type.damage_multiplier(type_1, type_2, type_chart=_type_chart)
    if ab == "wonderguard":
        return base if base > 1.0 else 0.0
    if ab == "flashfire" and status == Status.FRZ:
        return base
    return base * ABILITY_TYPE_MULTIPLIER.get(ab, {}).get(move_type, 1.0)


def test_effective_multiplier_by_types_matches_reference_exhaustively():
    """Every (attacking type × defender type1 × defender type2 × ability × status) combo
    must equal the damage_multiplier-based reference."""
    from agents.gen3_mechanics import _REAL_TYPES, effective_multiplier_by_types
    abilities = [None, "", "Levitate", "voltabsorb", "waterabsorb", "flashfire",
                 "thickfat", "wonderguard", "intimidate"]
    statuses = [None, Status.FRZ, Status.BRN]
    checked = 0
    for att in _REAL_TYPES:
        for t1 in _REAL_TYPES:
            for t2 in (None, *_REAL_TYPES):
                for ability in abilities:
                    for status in statuses:
                        got = effective_multiplier_by_types(att, t1, t2, ability, status)
                        exp = _reference_eff(att, t1, t2, ability, status)
                        assert got == exp, (att, t1, t2, ability, status, got, exp)
                        checked += 1
    assert checked > 100_000  # exhaustive, not a token sample


def test_effective_multiplier_object_wrapper_delegates():
    """The object-based wrapper reads (type_1, type_2, ability, status) off the mon and
    returns the same value as the value-based primitive (including the lowercase + the
    Gen 3 Flash-Fire-while-frozen quirk)."""
    from agents.gen3_mechanics import effective_multiplier_by_types
    mon = _mon(type_1=PokemonType.WATER, type_2=PokemonType.FLYING, ability="intimidate")
    assert effective_multiplier(PokemonType.ELECTRIC, mon) == 4.0  # 2× (Water) × 2× (Flying)
    assert effective_multiplier(PokemonType.ELECTRIC, mon) == effective_multiplier_by_types(
        PokemonType.ELECTRIC, PokemonType.WATER, PokemonType.FLYING, "intimidate", None)
    # Flash Fire does NOT absorb while frozen → Fire-vs-Fire resists to 0.5×, not 0×.
    frz = _mon(type_1=PokemonType.FIRE, ability="flashfire", status=Status.FRZ)
    assert effective_multiplier(PokemonType.FIRE, frz) == 0.5


# ---------------------------------------------------------------------------
# status_move_lands (gen3_move_effects_v1) — does a dedicated status move apply?
# ---------------------------------------------------------------------------

class TestStatusMoveLands:
    def test_none_status_returns_false(self):
        # A non-status move (status_id None) never "lands a status".
        assert status_move_lands("tackle", None, _mon()) is False

    def test_lands_on_clean_target(self):
        assert status_move_lands("toxic", "tox", _mon(PokemonType.NORMAL)) is True
        assert status_move_lands("thunderwave", "par", _mon(PokemonType.NORMAL)) is True

    def test_type_immune_blocks(self):
        # Poison/Steel immune to Toxic; Ground immune to Thunder Wave; Fire immune to WoW.
        assert status_move_lands("toxic", "tox", _mon(PokemonType.POISON)) is False
        assert status_move_lands("toxic", "tox", _mon(PokemonType.STEEL)) is False
        assert status_move_lands("thunderwave", "par", _mon(PokemonType.GROUND)) is False
        assert status_move_lands("willowisp", "brn", _mon(PokemonType.FIRE)) is False

    def test_already_statused_blocks(self):
        assert status_move_lands("toxic", "tox", _mon(PokemonType.WATER, status=Status.BRN)) is False

    def test_known_ability_immunity_blocks(self):
        # Immunity (Snorlax) blocks poison/toxic; Limber blocks paralysis. poke-env reveals
        # the ability after the first immune proc, so this engages from attempt 2 onward.
        assert status_move_lands("toxic", "tox", _mon(PokemonType.NORMAL, ability="immunity")) is False
        assert status_move_lands("thunderwave", "par", _mon(PokemonType.NORMAL, ability="limber")) is False

    def test_unknown_ability_does_not_over_claim(self):
        # Unrevealed ability ⇒ assume it lands (best guess) — never a false "whiff".
        assert status_move_lands("toxic", "tox", _mon(PokemonType.NORMAL, ability=None)) is True
        assert status_move_lands("toxic", "tox", _mon(PokemonType.NORMAL, ability="unknownability")) is True

    def test_irrelevant_ability_does_not_block(self):
        # An ability that doesn't grant status immunity leaves the status landing.
        assert status_move_lands("toxic", "tox", _mon(PokemonType.NORMAL, ability="thickfat")) is True

    def test_substitute_blocks(self):
        assert status_move_lands(
            "toxic", "tox", _mon(PokemonType.NORMAL, effects={Effect.SUBSTITUTE: 1})
        ) is False

    def test_ability_immunity_map_is_correct(self):
        assert "tox" in ABILITY_STATUS_IMMUNITY["immunity"]
        assert "psn" in ABILITY_STATUS_IMMUNITY["immunity"]
        assert ABILITY_STATUS_IMMUNITY["limber"] == frozenset({"par"})


# ---------------------------------------------------------------------------
# status_land_probability (gen3_move_effects_v1) — priors first, then confirmation
# ---------------------------------------------------------------------------

class TestStatusLandProbability:
    def test_unrevealed_uses_prior_mass(self):
        # Unrevealed Snorlax: Immunity 0.86 (blocks tox) / Thick Fat 0.14 (doesn't) →
        # P(Toxic lands) = 1 - 0.86 = 0.14.
        mon = _mon(PokemonType.NORMAL, ability=None)
        dist = [("immunity", 0.86), ("thickfat", 0.14)]
        assert status_land_probability("toxic", "tox", mon, dist) == pytest.approx(0.14)

    def test_revealed_blocking_ability_collapses_to_zero(self):
        mon = _mon(PokemonType.NORMAL, ability="immunity")
        assert status_land_probability("toxic", "tox", mon, [("immunity", 1.0)]) == 0.0

    def test_revealed_nonblocking_ability_collapses_to_one(self):
        mon = _mon(PokemonType.NORMAL, ability="thickfat")
        assert status_land_probability("toxic", "tox", mon, [("thickfat", 1.0)]) == pytest.approx(1.0)

    def test_no_info_distribution_lands(self):
        # [(None, 1.0)] sentinel (no priors, ability unknown) → ability contributes nothing.
        mon = _mon(PokemonType.NORMAL, ability=None)
        assert status_land_probability("toxic", "tox", mon, [(None, 1.0)]) == pytest.approx(1.0)

    def test_type_immunity_is_zero_regardless_of_priors(self):
        # A Poison-type can't be Toxic'd no matter what its ability prior says.
        mon = _mon(PokemonType.POISON, ability=None)
        dist = [("levitate", 0.5), ("clearbody", 0.5)]  # neither blocks status
        assert status_land_probability("toxic", "tox", mon, dist) == 0.0

    def test_already_statused_is_zero(self):
        mon = _mon(PokemonType.NORMAL, status=Status.BRN, ability=None)
        assert status_land_probability("toxic", "tox", mon, [("thickfat", 1.0)]) == 0.0

    def test_status_specific_immunity(self):
        # Immunity blocks poison but NOT paralysis → Thunder Wave still lands on a
        # (revealed-Immunity) Snorlax.
        mon = _mon(PokemonType.NORMAL, ability="immunity")
        assert status_land_probability("thunderwave", "par", mon, [("immunity", 1.0)]) == pytest.approx(1.0)

    def test_bool_wrapper_matches_revealed_probability(self):
        # status_move_lands is the singleton-from-live-ability special case.
        immune = _mon(PokemonType.NORMAL, ability="immunity")
        clean = _mon(PokemonType.NORMAL, ability=None)
        assert status_move_lands("toxic", "tox", immune) is False
        assert status_move_lands("toxic", "tox", clean) is True


# ---------------------------------------------------------------------------
# status_land_estimate — the (probability, known) pair; known mirrors abilities
# ---------------------------------------------------------------------------

class TestStatusLandEstimate:
    def test_revealed_ability_is_known(self):
        mon = _mon(PokemonType.NORMAL, ability="immunity")
        prob, known = status_land_estimate("toxic", "tox", mon, [("immunity", 1.0)])
        assert prob == 0.0 and known is True

    def test_unrevealed_prior_is_not_known(self):
        # The headline case: fractional value from a Smogon prior → known=False.
        mon = _mon(PokemonType.NORMAL, ability=None)
        prob, known = status_land_estimate("toxic", "tox", mon, [("immunity", 0.86), ("thickfat", 0.14)])
        assert prob == pytest.approx(0.14) and known is False

    def test_type_immunity_is_known_even_if_ability_unrevealed(self):
        # A Steel type can't be Toxic'd regardless of ability — certain, so known=True.
        mon = _mon(PokemonType.STEEL, ability=None)
        prob, known = status_land_estimate("toxic", "tox", mon, [("immunity", 0.5), ("thickfat", 0.5)])
        assert prob == 0.0 and known is True

    def test_already_statused_and_substitute_are_known(self):
        statused = _mon(PokemonType.NORMAL, status=Status.BRN, ability=None)
        assert status_land_estimate("toxic", "tox", statused, [("immunity", 1.0)]) == (0.0, True)
        subbed = _mon(PokemonType.NORMAL, ability=None, effects={Effect.SUBSTITUTE: 1})
        assert status_land_estimate("toxic", "tox", subbed, [("thickfat", 1.0)]) == (0.0, True)

    def test_unknownability_sentinel_is_not_known(self):
        # Mirrors the ability block: "unknownability" counts as unrevealed → known=False.
        mon = _mon(PokemonType.NORMAL, ability="unknownability")
        prob, known = status_land_estimate("toxic", "tox", mon, [("immunity", 0.86), ("thickfat", 0.14)])
        assert prob == pytest.approx(0.14) and known is False

    def test_known_predicate_matches_ability_block(self):
        # status_will_land_known uses the SAME reveal predicate as AbilitiesEncoder's `known`:
        # set ability whose normalized id != "unknownability".
        from agents.gen3_mechanics import _ability_revealed
        assert _ability_revealed(_mon(ability="immunity")) is True
        assert _ability_revealed(_mon(ability="Thick Fat")) is True   # normalized
        assert _ability_revealed(_mon(ability=None)) is False
        assert _ability_revealed(_mon(ability="unknownability")) is False


# ---------------------------------------------------------------------------
# (B) Mechanics-side completeness — ABILITY_STATUS_IMMUNITY covers every gen3
#     ability that grants full immunity to a MAJOR status, DERIVED FROM SOURCE.
#     Pairs with the obs-side lockstep guard in gen3_effects_test.py
#     (test_status_immunity_abilities_all_have_volatile_slot): together they keep this
#     mechanics map and the volatile allowlist in lockstep, so a status-immunity ability
#     can never live in one without the other — the waterveil crash, made impossible.
# ---------------------------------------------------------------------------
_MAJOR_STATUSES = frozenset({"slp", "brn", "frz", "par", "psn", "tox"})


def _gen3_status_immunity_from_source():
    """Derive ``{ability_id -> set(major status ids it fully blocks)}`` from Showdown
    ``abilities.ts``, restricted to gen3 abilities (``data/pokemon/gen3_abilities.json``).

    ``gen3_abilities.json`` carries only ``{num, name}`` — the status-blocking *semantics*
    live in the Showdown source — so this is the genuine source of truth for the class. A
    gen3 ability grants full immunity to a major status iff its body uses one of the two
    prevention idioms keyed on that status:
      * ``onSetStatus``: ``if (status.id !== 'X') return;`` — no-ops for other statuses,
        blocks X (Immunity lists two: ``!== 'psn' && !== 'tox'``).
      * ``onImmunity``: ``if (type === 'X') return false;`` — Magma Armor's freeze block.
    Restricting to :data:`_MAJOR_STATUSES` cleanly drops the non-status immunities that share
    the idiom — Oblivious (``attract``) and Sand Veil (``sandstorm``) — and the faster-cure
    abilities never match (Shed Skin / Early Bird / Natural Cure *cure*, they don't *block*).
    If poke-env/Showdown later makes a status-immunity ability gen3-legal, this fails until
    :data:`ABILITY_STATUS_IMMUNITY` is re-derived to match."""
    import json
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2]
    txt = (root / "deps/pokemon-showdown/data/abilities.ts").read_text(
        encoding="utf-8", errors="replace"
    )
    gen3 = set(json.load(open(root / "data/pokemon/gen3_abilities.json")).keys())

    def _callback_body(body: str, name: str) -> str:
        """Brace-matched body of the ``name(...) { ... }`` callback within ``body`` (so a
        status literal elsewhere in the ability can't leak into the scan), or ''."""
        m = re.search(name + r"\s*\([^)]*\)\s*\{", body)
        if not m:
            return ""
        start, depth = m.end() - 1, 0
        for j in range(start, len(body)):
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
                if depth == 0:
                    return body[start:j + 1]
        return body[start:]

    keys = list(re.finditer(r"\n\t([a-z0-9]+): \{", txt))
    derived: dict[str, set[str]] = {}
    for i, k in enumerate(keys):
        aid = k.group(1)
        if aid not in gen3:
            continue
        body = txt[k.end():(keys[i + 1].start() if i + 1 < len(keys) else len(txt))]
        blocked = set()
        for m in re.finditer(
            r"status\.id\s*!==\s*'([a-z]+)'", _callback_body(body, "onSetStatus")
        ):
            if m.group(1) in _MAJOR_STATUSES:
                blocked.add(m.group(1))
        for m in re.finditer(
            r"type\s*===\s*'([a-z]+)'\s*\)\s*return false", _callback_body(body, "onImmunity")
        ):
            if m.group(1) in _MAJOR_STATUSES:
                blocked.add(m.group(1))
        if blocked:
            derived[aid] = blocked
    return derived


@pytest.mark.integration  # needs deps/pokemon-showdown checked out
def test_ability_status_immunity_covers_every_gen3_source_ability():
    """ABILITY_STATUS_IMMUNITY must equal the source-derived gen3 status-immunity set —
    keys AND blocked-status sets. EQUALITY, not just superset: a MISSING ability
    under-claims a whiff (``status_will_land`` reads high when the status can't land) and,
    for any ability that also activates, risks the volatile crash-don't-drop; an EXTRA
    ability over-claims immunity. A newly-relevant gen3 status-immunity ability fails HERE,
    in CI, instead of silently degrading the obs signal or crashing a run hours in."""
    derived = _gen3_status_immunity_from_source()
    assert derived, "source scan found no gen3 status-immunity abilities — the scan broke"
    actual = {aid: set(s) for aid, s in ABILITY_STATUS_IMMUNITY.items()}
    missing = {a: sorted(derived[a]) for a in derived if actual.get(a) != derived[a]}
    extra = {a: sorted(actual[a]) for a in actual if a not in derived}
    assert actual == derived, (
        f"ABILITY_STATUS_IMMUNITY drifted from abilities.ts — re-derive it.\n"
        f"  missing / wrong statuses : {missing}\n"
        f"  extra (not in source)    : {extra}\n"
        f"  source of truth          : "
        f"{dict(sorted((a, sorted(s)) for a, s in derived.items()))}"
    )
