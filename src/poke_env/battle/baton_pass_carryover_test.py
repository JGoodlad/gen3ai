"""Pin the Baton Pass carry-over in `Battle.switch` — the boosts/volatiles the PROTOCOL never repeats.

The bug this defends against (found 2026-08-23 from a live probe trace, present since the fork was
vendored 2026-05-12): `Battle.switch` unconditionally called `switch_out()` on the outgoing mon,
which `clear_boosts()`es and `_clear_effects()`s it, and `_parse_message` sliced the switch event
as `event[2:5]` — throwing the `[from] Baton Pass` tag away before anyone could look at it. So a
Celebi that Calm-Minded to +2 and passed to Charizard produced a Charizard that this client believed
had NO boosts, while the SIM had `spa +2 / spd +2`.

That is a GIGO defect, not a display one: `LiveView` reads `mon.boosts` / `mon.effects`, the
observation's active-context block reads `LiveView`, and the reward's boost PBRS term reads the same
— so a successful pass was observed as a total loss of setup and *penalised*.

The sim's contract (`sim/pokemon.ts::copyVolatileFrom`) is `this.boosts = pokemon.boosts` plus a
shallow copy of every non-`noCopy` volatile, emitting NOTHING. Verified against the omniscient
BattleStream: the passer's `|switch|…|[from] Baton Pass` line is the entire protocol trace.
"""
import pytest

from poke_env.battle.battle import Battle
from poke_env.battle.effect import BATON_PASS_COPIED_EFFECTS, Effect


def _battle() -> Battle:
    b = Battle("tag", "player1", None, gen=3)  # type: ignore[arg-type]
    b.player_role = "p1"
    return b


def _feed(battle: Battle, *lines: str) -> None:
    for line in lines:
        battle.parse_message(line.split("|"))


def _set_up_boosted_passer(battle: Battle) -> None:
    _feed(
        battle,
        "|switch|p1a: Celebi|Celebi|100/100",
        "|switch|p2a: Blissey|Blissey, F|100/100",
        "|-boost|p1a: Celebi|spa|2",
        "|-boost|p1a: Celebi|spd|2",
        "|-start|p1a: Celebi|Substitute",
    )


def test_baton_pass_carries_boosts_to_the_entrant():
    battle = _battle()
    _set_up_boosted_passer(battle)
    assert battle.active_pokemon.boosts["spa"] == 2

    _feed(battle, "|switch|p1a: Charizard|Charizard, M|100/100|[from] Baton Pass")

    entrant = battle.active_pokemon
    assert entrant.species == "charizard"
    assert entrant.boosts["spa"] == 2, (
        "the entrant lost the passed Special Attack stages — the protocol never repeats "
        "them, so `Battle.switch` is the only place they can survive"
    )
    assert entrant.boosts["spd"] == 2


def test_baton_pass_carries_copyable_volatiles_to_the_entrant():
    battle = _battle()
    _set_up_boosted_passer(battle)

    _feed(battle, "|switch|p1a: Charizard|Charizard, M|100/100|[from] Baton Pass")

    assert Effect.SUBSTITUTE in battle.active_pokemon.effects, (
        "Substitute is `noCopy`-falsy — the sim hands it to the entrant and says nothing"
    )


def test_a_plain_switch_still_clears_boosts():
    """The other direction: the fix must not make every switch preserve state."""
    battle = _battle()
    _set_up_boosted_passer(battle)

    _feed(battle, "|switch|p1a: Charizard|Charizard, M|100/100")

    assert battle.active_pokemon.boosts["spa"] == 0
    assert Effect.SUBSTITUTE not in battle.active_pokemon.effects


def test_a_phaze_drag_never_carries_boosts():
    """Roar/Whirlwind is not a self-switch; it can't carry a pass and never tags one."""
    battle = _battle()
    _set_up_boosted_passer(battle)

    _feed(battle, "|drag|p1a: Charizard|Charizard, M|100/100")

    assert battle.active_pokemon.boosts["spa"] == 0


def test_the_opponents_baton_pass_is_tracked_too():
    """We model the opponent's board from the same messages — the mirror must work."""
    battle = _battle()
    _feed(
        battle,
        "|switch|p1a: Celebi|Celebi|100/100",
        "|switch|p2a: Jolteon|Jolteon, M|100/100",
        "|-boost|p2a: Jolteon|spe|2",
        "|switch|p2a: Snorlax|Snorlax, M|100/100|[from] Baton Pass",
    )
    assert battle.opponent_active_pokemon.boosts["spe"] == 2


def test_negative_stages_ride_the_pass_as_well():
    """`copyVolatileFrom` assigns the whole boosts table — drops included."""
    battle = _battle()
    _feed(
        battle,
        "|switch|p1a: Celebi|Celebi|100/100",
        "|switch|p2a: Blissey|Blissey, F|100/100",
        "|-boost|p1a: Celebi|spa|2",
        "|-unboost|p1a: Celebi|def|2",
        "|switch|p1a: Charizard|Charizard, M|100/100|[from] Baton Pass",
    )
    assert battle.active_pokemon.boosts["spa"] == 2
    assert battle.active_pokemon.boosts["def"] == -2


@pytest.mark.integration  # reads the vendored Showdown source tree
def test_baton_pass_copied_effects_match_the_dex():
    """Every member of the allow-list must be `noCopy`-falsy in the sim we actually run.

    The list is an allow-list (see `effect.py` for why), so this can only catch a member
    that should NOT be there — which is the direction that would invent state.
    """
    import re
    from pathlib import Path

    # …/<repo>/src/poke_env/battle/<this file>
    root = Path(__file__).resolve().parents[3] / "deps" / "pokemon-showdown"
    sources = [
        root / "data" / "moves.ts",
        root / "data" / "conditions.ts",
        root / "data" / "mods" / "gen3" / "moves.ts",
    ]
    assert all(p.is_file() for p in sources), (
        f"vendored Showdown source missing under {root} — run `git submodule update --init`. "
        "A skip here would read exactly like a clean pass."
    )

    no_copy_ids: set[str] = set()
    for path in sources:
        lines = path.read_text().split("\n")
        for i, line in enumerate(lines):
            if "noCopy" not in line or "true" not in line:
                continue
            for j in range(i, -1, -1):
                m = re.match(r"^\t([a-z0-9]+): \{", lines[j])
                if m:
                    no_copy_ids.add(m.group(1))
                    break

    assert "encore" in no_copy_ids and "destinybond" in no_copy_ids, (
        f"the noCopy scrape found {len(no_copy_ids)} ids but not the known ones — "
        "the dex layout changed and this check has gone vacuous"
    )

    # The wrap family is tracked per-move by poke-env but is ONE `partiallytrapped`
    # volatile in the sim; likewise the four perish counters.
    sim_id = {
        Effect.BIND: "partiallytrapped",
        Effect.CLAMP: "partiallytrapped",
        Effect.FIRE_SPIN: "partiallytrapped",
        Effect.SAND_TOMB: "partiallytrapped",
        Effect.WHIRLPOOL: "partiallytrapped",
        Effect.WRAP: "partiallytrapped",
        Effect.PERISH0: "perishsong",
        Effect.PERISH1: "perishsong",
        Effect.PERISH2: "perishsong",
        Effect.PERISH3: "perishsong",
        Effect.LEECH_SEED: "leechseed",
        Effect.FOCUS_ENERGY: "focusenergy",
    }
    offenders = sorted(
        e.name
        for e in BATON_PASS_COPIED_EFFECTS
        if sim_id.get(e, e.name.lower().replace("_", "")) in no_copy_ids
    )
    assert not offenders, (
        f"these are marked `noCopy: true` in the dex and must not ride a Baton Pass: {offenders}"
    )
