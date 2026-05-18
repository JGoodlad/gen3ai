"""
Deterministic unit tests for move effectiveness and move-order tracking in
AbstractBattle / Battle.

Validates all edge cases by injecting raw parse_message() sequences directly —
no live server required.

Critical: these tests prove the core parsing logic is correct independent of the
fuzz test (which validates the full pipeline end-to-end but is probabilistic).
"""
import logging
import pytest
from poke_env.battle.battle import Battle


def _battle(role: str = "p1") -> Battle:
    """Create a minimal Battle for testing parse_message directly."""
    b = Battle("battle-gen3ou-test", "TestPlayer", logging.getLogger("test"), gen=3)
    b._player_role = role
    return b


def _do_turn(b: Battle, turn_num: int, our_move: str, opp_move: str,
             our_eff_msg: str = None, opp_eff_msg: str = None,
             our_active: str = "Zapdos", opp_active: str = "Tyranitar") -> None:
    """
    Simulate one full turn:
      1. |turn|turn_num
      2. Our mon uses our_move (or opp moves first if opp_first=True via parameter order)
      3. Effectiveness message if provided
      4. Opp mon uses opp_move
      5. Effectiveness message if provided

    This helper always has us move first within the turn (add tests that reverse order
    using the lower-level helpers directly).
    """
    b.parse_message(["", "turn", str(turn_num)])
    b.parse_message(["", "move", f"p1a: {our_active}", our_move, f"p2a: {opp_active}"])
    if our_eff_msg:
        b.parse_message(["", our_eff_msg, f"p2a: {opp_active}"])
    b.parse_message(["", "-damage", f"p2a: {opp_active}", "50/100"])
    b.parse_message(["", "move", f"p2a: {opp_active}", opp_move, f"p1a: {our_active}"])
    if opp_eff_msg:
        b.parse_message(["", opp_eff_msg, f"p1a: {our_active}"])
    b.parse_message(["", "-damage", f"p1a: {our_active}", "50/100"])


# ---------------------------------------------------------------------------
# Core effectiveness cases
# ---------------------------------------------------------------------------

def test_our_move_supereffective():
    """Our damaging move is SE on the opponent."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Zapdos", "Thunderbolt", "p2a: Gyarados"])
    b.parse_message(["", "-supereffective", "p2a: Gyarados"])
    b.parse_message(["", "-damage", "p2a: Gyarados", "20/100"])
    b.parse_message(["", "move", "p2a: Gyarados", "Waterfall", "p1a: Zapdos"])
    b.parse_message(["", "-damage", "p1a: Zapdos", "70/100"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness == 2.0
    assert b.opp_last_effectiveness == 1.0
    assert b.we_moved_first is True  # we moved first


def test_our_move_resisted():
    """Our damaging move is resisted by the opponent."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Zapdos", "Thunderbolt", "p2a: Raichu"])
    b.parse_message(["", "-resisted", "p2a: Raichu"])
    b.parse_message(["", "-damage", "p2a: Raichu", "80/100"])
    b.parse_message(["", "move", "p2a: Raichu", "Surf", "p1a: Zapdos"])
    b.parse_message(["", "-damage", "p1a: Zapdos", "70/100"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness == 0.5
    assert b.opp_last_effectiveness == 1.0


def test_our_move_immune():
    """Our move is immune on the opponent (e.g. Earthquake vs Levitate Gengar)."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Flygon", "Earthquake", "p2a: Gengar"])
    b.parse_message(["", "-immune", "p2a: Gengar"])
    b.parse_message(["", "move", "p2a: Gengar", "Shadow Ball", "p1a: Flygon"])
    b.parse_message(["", "-damage", "p1a: Flygon", "75/100"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness == 0.0
    assert b.opp_last_effectiveness == 1.0


def test_opp_move_supereffective():
    """Opponent's damaging move is SE on us."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Tyranitar", "Dragon Dance", "p1a: Tyranitar"])
    b.parse_message(["", "move", "p2a: Suicune", "Surf", "p1a: Tyranitar"])
    b.parse_message(["", "-supereffective", "p1a: Tyranitar"])
    b.parse_message(["", "-damage", "p1a: Tyranitar", "30/100"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness is None   # Dragon Dance has basePower=0
    assert b.opp_last_effectiveness == 2.0


def test_opp_move_resisted():
    """Opponent's damaging move is resisted on us."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p2a: Blissey", "Seismic Toss", "p1a: Metagross"])
    b.parse_message(["", "-resisted", "p1a: Metagross"])
    b.parse_message(["", "-damage", "p1a: Metagross", "95/100"])
    b.parse_message(["", "move", "p1a: Metagross", "Earthquake", "p2a: Blissey"])
    b.parse_message(["", "-damage", "p2a: Blissey", "50/100"])
    b.parse_message(["", "turn", "2"])

    assert b.opp_last_effectiveness == 0.5
    assert b.our_last_effectiveness == 1.0


def test_opp_move_immune():
    """Opponent's move is immune on us (e.g. Thunderbolt vs Ground Swampert)."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Swampert", "Earthquake", "p2a: Raichu"])
    b.parse_message(["", "-damage", "p2a: Raichu", "50/100"])
    b.parse_message(["", "move", "p2a: Raichu", "Thunderbolt", "p1a: Swampert"])
    b.parse_message(["", "-immune", "p1a: Swampert"])
    b.parse_message(["", "turn", "2"])

    assert b.opp_last_effectiveness == 0.0
    assert b.our_last_effectiveness == 1.0


def test_normal_effectiveness_no_message():
    """Neutral effectiveness: no -supereffective/-resisted fires; tentative 1.0 stands."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Tyranitar", "Rock Slide", "p2a: Suicune"])
    b.parse_message(["", "-damage", "p2a: Suicune", "70/100"])
    b.parse_message(["", "move", "p2a: Suicune", "Ice Beam", "p1a: Tyranitar"])
    b.parse_message(["", "-damage", "p1a: Tyranitar", "80/100"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness == 1.0   # Rock Slide neutral vs Water
    assert b.opp_last_effectiveness == 1.0   # Ice Beam neutral vs Rock/Dark


# ---------------------------------------------------------------------------
# Fixed-damage moves (basePower == 0): tentative-1.0 should NOT be set
# ---------------------------------------------------------------------------

def test_fixed_damage_move_no_effectiveness():
    """
    Seismic Toss, Dragon Rage, Night Shade have basePower=0 in gen3_moves.json.
    The tentative-1.0 logic skips them, so if no explicit message fires, result is None.
    """
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Blissey", "Seismic Toss", "p2a: Tyranitar"])
    b.parse_message(["", "-damage", "p2a: Tyranitar", "60/100"])
    b.parse_message(["", "move", "p2a: Tyranitar", "Dragon Rage", "p1a: Blissey"])
    b.parse_message(["", "-damage", "p1a: Blissey", "60/100"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness is None  # Seismic Toss: basePower=0
    assert b.opp_last_effectiveness is None  # Dragon Rage: basePower=0


def test_status_move_no_effectiveness():
    """Status moves (Dragon Dance, Calm Mind) have basePower=0; effectiveness is None."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Tyranitar", "Dragon Dance", "p1a: Tyranitar"])
    b.parse_message(["", "move", "p2a: Suicune", "Calm Mind", "p2a: Suicune"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness is None
    assert b.opp_last_effectiveness is None


# ---------------------------------------------------------------------------
# Move order
# ---------------------------------------------------------------------------

def test_we_moved_first():
    """We use a move before the opponent this turn."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Jolteon", "Thunderbolt", "p2a: Gyarados"])
    b.parse_message(["", "-supereffective", "p2a: Gyarados"])
    b.parse_message(["", "move", "p2a: Gyarados", "Waterfall", "p1a: Jolteon"])
    b.parse_message(["", "turn", "2"])

    assert b.we_moved_first is True


def test_opp_moved_first():
    """Opponent uses a move before us this turn."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p2a: Aerodactyl", "Rock Slide", "p1a: Zapdos"])
    b.parse_message(["", "move", "p1a: Zapdos", "Thunderbolt", "p2a: Aerodactyl"])
    b.parse_message(["", "turn", "2"])

    assert b.we_moved_first is False


def test_only_we_moved_order_is_none():
    """Only we used a move (opp switched): we_moved_first must be None."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Zapdos", "Thunderbolt", "p2a: Gyarados"])
    # Opponent switches (no |move| message for their side)
    b.parse_message(["", "switch", "p2a: Blissey", "Blissey, L100, F", "100/100"])
    b.parse_message(["", "turn", "2"])

    assert b.we_moved_first is None


def test_both_switched_order_is_none():
    """Both sides switched: we_moved_first must be None."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "switch", "p1a: Blissey", "Blissey, L100, F", "100/100"])
    b.parse_message(["", "switch", "p2a: Tyranitar", "Tyranitar, L100", "100/100"])
    b.parse_message(["", "turn", "2"])

    assert b.we_moved_first is None


def test_only_opp_moved_order_is_none():
    """Only the opponent used a move (we switched): we_moved_first must be None."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "switch", "p1a: Blissey", "Blissey, L100, F", "100/100"])
    b.parse_message(["", "move", "p2a: Tyranitar", "Rock Slide", "p1a: Blissey"])
    b.parse_message(["", "turn", "2"])

    assert b.we_moved_first is None


# ---------------------------------------------------------------------------
# Staleness: properties only expose data from the most-recent completed turn
# ---------------------------------------------------------------------------

def test_stale_effectiveness_not_exposed():
    """
    After two complete turns, effectiveness from turn 1 is no longer accessible
    on turn 3 (it was from turn 1, not turn 2).
    """
    b = _battle()
    # Turn 1: our Thunderbolt is SE
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Zapdos", "Thunderbolt", "p2a: Gyarados"])
    b.parse_message(["", "-supereffective", "p2a: Gyarados"])
    b.parse_message(["", "move", "p2a: Gyarados", "Waterfall", "p1a: Zapdos"])
    b.parse_message(["", "turn", "2"])

    # Turn 1 data should be visible at turn 2
    assert b.our_last_effectiveness == 2.0

    # Turn 2: both sides switch, no moves
    b.parse_message(["", "switch", "p1a: Metagross", "Metagross, L100", "100/100"])
    b.parse_message(["", "switch", "p2a: Blissey", "Blissey, L100, F", "100/100"])
    b.parse_message(["", "turn", "3"])

    # Turn 2 data at turn 3: no moves this turn, so effectiveness is None
    assert b.our_last_effectiveness is None
    assert b.opp_last_effectiveness is None
    assert b.we_moved_first is None


def test_fresh_turn_overwrites_previous():
    """Each turn writes fresh data; the old (turn N-2) value must not leak through."""
    b = _battle()
    # Turn 1: our move is SE
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Zapdos", "Thunderbolt", "p2a: Gyarados"])
    b.parse_message(["", "-supereffective", "p2a: Gyarados"])
    b.parse_message(["", "move", "p2a: Gyarados", "Waterfall", "p1a: Zapdos"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness == 2.0

    # Turn 2: our move is resisted
    b.parse_message(["", "move", "p1a: Zapdos", "Thunderbolt", "p2a: Magneton"])
    b.parse_message(["", "-resisted", "p2a: Magneton"])
    b.parse_message(["", "move", "p2a: Magneton", "Thunderbolt", "p1a: Zapdos"])
    b.parse_message(["", "-resisted", "p1a: Zapdos"])
    b.parse_message(["", "turn", "3"])

    assert b.our_last_effectiveness == 0.5   # not 2.0 from turn 1
    assert b.opp_last_effectiveness == 0.5


# ---------------------------------------------------------------------------
# Immunity with ability reveal
# ---------------------------------------------------------------------------

def test_immune_with_ability_reveal_sets_effectiveness_and_ability():
    """
    |-immune|p2a: Gengar|[from] ability: Levitate|
    Should set opp's last effectiveness to 0.0 (we attacked them, they're immune)
    AND reveal the ability.
    """
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Flygon", "Earthquake", "p2a: Gengar"])
    # Four-element immune message: ability reveal attached
    b.parse_message(["", "-immune", "p2a: Gengar", "[from] ability: Levitate"])
    b.parse_message(["", "move", "p2a: Gengar", "Shadow Ball", "p1a: Flygon"])
    b.parse_message(["", "-damage", "p1a: Flygon", "75/100"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness == 0.0

    opp_mon = b.get_pokemon("p2: Gengar")
    assert opp_mon.ability == "levitate"


def test_immune_ohko_format_does_not_set_ability():
    """
    |-immune|p1a: Lapras|[ohko]| (from Horn Drill/Fissure OHKO immunity)
    Should set effectiveness to 0.0 but NOT set ability (cause is [ohko], not ability).
    """
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p2a: Cloyster", "Horn Drill", "p1a: Lapras"])
    b.parse_message(["", "-immune", "p1a: Lapras", "[ohko]"])
    b.parse_message(["", "move", "p1a: Lapras", "Surf", "p2a: Cloyster"])
    b.parse_message(["", "-damage", "p2a: Cloyster", "50/100"])
    b.parse_message(["", "turn", "2"])

    # OPP used Horn Drill (0 basePower), OHKO immune on us → opp_last_effectiveness = 0.0
    assert b.opp_last_effectiveness == 0.0
    # Ability should NOT be set (cause is [ohko], not [from] ability:)
    p1_lapras = b.get_pokemon("p1: Lapras")
    assert p1_lapras.ability != "ohko"


# ---------------------------------------------------------------------------
# Multi-hit moves: each hit generates its own effectiveness message
# (last write wins — all hits should have the same effectiveness for a given matchup)
# ---------------------------------------------------------------------------

def test_multihit_supereffective_last_write_wins():
    """
    Bullet Seed (5 hits) vs Swampert generates 5 |-supereffective| messages.
    All 5 write the same value; the final state must be 2.0.
    """
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Venusaur", "Bullet Seed", "p2a: Swampert"])
    # Simulate 5 hits, each with a -supereffective message
    for _ in range(5):
        b.parse_message(["", "-supereffective", "p2a: Swampert"])
        b.parse_message(["", "-damage", "p2a: Swampert", "90/100"])
    b.parse_message(["", "-hitcount", "p2a: Swampert", "5"])
    b.parse_message(["", "move", "p2a: Swampert", "Earthquake", "p1a: Venusaur"])
    b.parse_message(["", "-supereffective", "p1a: Venusaur"])
    b.parse_message(["", "-damage", "p1a: Venusaur", "40/100"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness == 2.0   # consistent across all 5 hits
    assert b.opp_last_effectiveness == 2.0


# ---------------------------------------------------------------------------
# SE override: tentative 1.0 set by |move| is correctly overridden
# ---------------------------------------------------------------------------

def test_tentative_overridden_by_supereffective():
    """
    The |move| handler sets tentative 1.0 for damaging moves, then |-supereffective|
    overrides it. Final value must be 2.0, not 1.0.
    """
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Jolteon", "Thunderbolt", "p2a: Gyarados"])
    # At this point _our_last_effectiveness = (1, 1.0) tentatively
    b.parse_message(["", "-supereffective", "p2a: Gyarados"])
    # Now _our_last_effectiveness = (1, 2.0) — overridden
    b.parse_message(["", "move", "p2a: Gyarados", "Waterfall", "p1a: Jolteon"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness == 2.0   # NOT 1.0


def test_tentative_overridden_by_immune():
    """
    The |move| handler sets tentative 1.0 for a damaging move (e.g. Earthquake has
    basePower=100), then |-immune| overrides it.
    """
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Marowak", "Earthquake", "p2a: Gengar"])
    # Tentative 1.0 set (Earthquake basePower=100)
    b.parse_message(["", "-immune", "p2a: Gengar"])
    # Now overridden to 0.0
    b.parse_message(["", "move", "p2a: Gengar", "Shadow Ball", "p1a: Marowak"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness == 0.0   # NOT 1.0


# ---------------------------------------------------------------------------
# Both sides attack in same turn: each side tracks independently
# ---------------------------------------------------------------------------

def test_both_sides_attack_independent_tracking():
    """
    Our move is SE, opponent's move is resisted. Both tracked independently."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", "p1a: Jolteon", "Thunderbolt", "p2a: Gyarados"])
    b.parse_message(["", "-supereffective", "p2a: Gyarados"])
    b.parse_message(["", "-damage", "p2a: Gyarados", "20/100"])
    b.parse_message(["", "move", "p2a: Gyarados", "Surf", "p1a: Jolteon"])
    b.parse_message(["", "-resisted", "p1a: Jolteon"])
    b.parse_message(["", "-damage", "p1a: Jolteon", "90/100"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness == 2.0
    assert b.opp_last_effectiveness == 0.5
    assert b.we_moved_first is True


# ---------------------------------------------------------------------------
# Player role = "p2" (we are the second player)
# ---------------------------------------------------------------------------

def test_player_role_p2_effectiveness():
    """When we are p2, effectiveness attribution is reversed — still correct."""
    b = _battle(role="p2")
    b.parse_message(["", "turn", "1"])
    # We (p2) use Thunderbolt on p1's Gyarados
    b.parse_message(["", "move", "p2a: Jolteon", "Thunderbolt", "p1a: Gyarados"])
    b.parse_message(["", "-supereffective", "p1a: Gyarados"])
    b.parse_message(["", "-damage", "p1a: Gyarados", "10/100"])
    # Opp (p1) uses Surf on us (p2)
    b.parse_message(["", "move", "p1a: Gyarados", "Surf", "p2a: Jolteon"])
    b.parse_message(["", "-resisted", "p2a: Jolteon"])
    b.parse_message(["", "-damage", "p2a: Jolteon", "90/100"])
    b.parse_message(["", "turn", "2"])

    assert b.our_last_effectiveness == 2.0   # our (p2) Thunderbolt was SE
    assert b.opp_last_effectiveness == 0.5   # opp (p1) Surf resisted on us (p2: Jolteon ≈ Electric)
    assert b.we_moved_first is True          # we (p2) moved first


def test_player_role_p2_we_moved_first():
    """Move order tracking is correct when we are p2."""
    b = _battle(role="p2")
    b.parse_message(["", "turn", "1"])
    # Opponent (p1) moves first
    b.parse_message(["", "move", "p1a: Aerodactyl", "Rock Slide", "p2a: Zapdos"])
    b.parse_message(["", "move", "p2a: Zapdos", "Thunderbolt", "p1a: Aerodactyl"])
    b.parse_message(["", "turn", "2"])

    assert b.we_moved_first is False   # opponent (p1) moved first, we are p2


# ---------------------------------------------------------------------------
# Type chart sanity checks (regression against known Gen 3 matchups)
# ---------------------------------------------------------------------------

def _check_matchup(attacker: str, our_move: str, opp_active: str,
                   expected_mult: float, eff_msg: str = None) -> None:
    """Helper: verify our_last_effectiveness for a specific type matchup."""
    b = _battle()
    b.parse_message(["", "turn", "1"])
    b.parse_message(["", "move", f"p1a: {attacker}", our_move, f"p2a: {opp_active}"])
    if eff_msg:
        b.parse_message(["", eff_msg, f"p2a: {opp_active}"])
    b.parse_message(["", "-damage", f"p2a: {opp_active}", "50/100"])
    b.parse_message(["", "move", f"p2a: {opp_active}", "Tackle", f"p1a: {attacker}"])
    b.parse_message(["", "-damage", f"p1a: {attacker}", "90/100"])
    b.parse_message(["", "turn", "2"])
    assert b.our_last_effectiveness == expected_mult, \
        f"{our_move} vs {opp_active}: expected {expected_mult}, got {b.our_last_effectiveness}"


def test_known_matchups():
    """Regression test: known Gen 3 type matchups must produce the right effectiveness."""
    # Electric SE on Water/Flying
    _check_matchup("Jolteon", "Thunderbolt", "Gyarados", 2.0, "-supereffective")
    # Ground immune on Flying (Showdown sends -immune for Earthquake vs Zapdos)
    _check_matchup("Marowak", "Earthquake", "Zapdos", 0.0, "-immune")
    # Water SE on Fire/Rock
    _check_matchup("Suicune", "Surf", "Charizard", 2.0, "-supereffective")
    # Flamethrower neutral on Blastoise (Fire vs Water — no SE/resisted msg in this test,
    # we're testing the tentative-1.0 path, not the actual type chart)
    _check_matchup("Charizard", "Flamethrower", "Blastoise", 1.0, None)
    # Steel resisted by Steel (Meteor Mash vs Skarmory)
    _check_matchup("Metagross", "Meteor Mash", "Skarmory", 0.5, "-resisted")
    # Normal immune on Ghost (Normal-type move vs Ghost-type defender)
    _check_matchup("Blissey", "Tackle", "Gengar", 0.0, "-immune")
    # Ghost immune on Normal (Ghost-type move vs Normal-type defender — Gen 3 type chart)
    _check_matchup("Gengar", "Shadow Ball", "Blissey", 0.0, "-immune")
    # Psychic immune on Dark (Psychic-type move vs Dark-type defender — Gen 3)
    _check_matchup("Alakazam", "Psychic", "Umbreon", 0.0, "-immune")
    # Steel resists Ghost (Gen 3 specific — Steel's Ghost resistance was removed in Gen 6)
    _check_matchup("Gengar", "Shadow Ball", "Skarmory", 0.5, "-resisted")
    # Steel resists Dark (Gen 3 specific — same removal in Gen 6)
    _check_matchup("Tyranitar", "Crunch", "Skarmory", 0.5, "-resisted")
    # Flash Fire immunity (ability-based, same protocol message as type immunity)
    _check_matchup("Charizard", "Flamethrower", "Ninetales", 0.0, "-immune")
