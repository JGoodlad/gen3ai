import pytest
from unittest.mock import MagicMock

from agents.enums import PokemonType, Status
from agents.gen3_mechanics import (
    ABILITY_TYPE_MULTIPLIER,
    PHAZING_MOVES,
    INVULNERABLE_MOVES,
    STATUS_MOVES,
    RECOVERY_MOVES,
    HAZARD_CLEAR_MOVES,
    SETUP_MOVES,
    BOOST_STATS,
    BOOST_DIM,
    ABILITY_STATUS_IMMUNITY,
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
    import re

    from utils.paths import repo_root

    root = repo_root()
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
