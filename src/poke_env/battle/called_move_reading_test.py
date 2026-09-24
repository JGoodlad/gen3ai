"""Pins for the CALLED-MOVE reading class (`gen3_called_move_reading_v1`).

gen3's `useMoveInner` announces a move that another move CALLED as
``|move|<user>|<called>|<target>|[from] <Caller>`` — the BARE condition name, no ``move:``
prefix — and ``attrLastMove`` can append ``[still]`` (which also BLANKS the target), ``[miss]``
(sometimes twice) or ``[notarget]`` after the ``[from]``. Upstream poke-env knew only the modern
``[from]move: Metronome`` form, so a Metronome / Assist / Nature Power call RAISED
``ValueError: Unhandled move message format`` (a live ladder game ends on the timer: 2 of 710
Metamon battles and 4 of 20,000 human logs, 2026-09-24), and a ``[still]``-blanked call was
silently added to the actor's OWN moveset.

Every tail shape below is one the real gen3 sim emitted in
``node src/rust_sim/harness/probe_called_move_shapes.js 1500 1`` (1,500 battles, 25,419 sourced
``|move|`` lines, 31 distinct shapes); each ``*_upstream`` test FAILS on the pre-fix parser. The
reading they pin is the SIM's, not poke-env's: the called move is not revealed (Metronome draws
from the dex, Assist from a teammate, Nature Power is always Swift in gen3), and no PP is charged
for it anywhere — the caller's own ``|move|`` line already took one, and gen3 skips the Pressure
deduction for a sourced move (measured: a Metronome user's PP after three uses is 14/16 against a
Pressure Zapdos, the same as against any foe). The end-to-end pin is
``agents/battle/called_move_bridge_integration_test.py`` (a real node-bridge battle), and the Rust
core's ``present()`` mirror is pinned by ``src/rust_sim/src/present/called_move_tests.rs`` and
``agents/battle/rust_core_present_test.py``.
"""
import pytest

from poke_env.battle.abstract_battle import GEN3_BARE_MOVE_CALLERS, _canonical_from_tail
from poke_env.battle.battle import Battle


def _battle() -> Battle:
    b = Battle("tag", "A", None, gen=3)  # type: ignore[arg-type]
    b.player_role = "p1"
    return b


def _feed(battle: Battle, *lines: str) -> None:
    for line in lines:
        battle.parse_message(line.split("|"))


def _opp(b: Battle, *lines: str):
    _feed(b, "|switch|p1a: Zapdos|Zapdos|100/100", "|switch|p2a: Clefable|Clefable, F|100/100",
          "|turn|1", *lines)
    return b.opponent_team["p2: Clefable"]


def _pp(mon, move_id):
    return mon.moves[move_id].current_pp


# ---------------------------------------------------------------------------------------------
# the random callers: every tail shape the gen3 sim emits
# ---------------------------------------------------------------------------------------------

SHAPES = [
    # (caller line, called line) — verbatim shapes from probe_called_move_shapes.js
    ("|move|p2a: Clefable|Metronome|p2a: Clefable",
     "|move|p2a: Clefable|Thunderbolt|p1a: Zapdos|[from] Metronome"),                # <foe>
    ("|move|p2a: Clefable|Metronome|p2a: Clefable",
     "|move|p2a: Clefable|Swords Dance|p2a: Clefable|[from] Metronome"),             # <self>
    ("|move|p2a: Clefable|Metronome|p2a: Clefable",
     "|move|p2a: Clefable|Slack Off||[from] Metronome|[still]"),                     # <empty>, [still]
    ("|move|p2a: Clefable|Metronome|p2a: Clefable",
     "|move|p2a: Clefable|Rock Slide|p1a: Zapdos|[from] Metronome|[miss]"),          # [miss]
    ("|move|p2a: Clefable|Metronome|p2a: Clefable",
     "|move|p2a: Clefable|Super Fang|p1a: Zapdos|[from] Metronome|[miss]|[miss]"),   # [miss] x2
    ("|move|p2a: Clefable|Metronome|p2a: Clefable",
     "|move|p2a: Clefable|Nature Power|p2a: Clefable|[from] Metronome|[miss]"),      # <self>, [miss]
    ("|move|p2a: Clefable|Assist|p2a: Clefable",
     "|move|p2a: Clefable|Explosion|p1a: Zapdos|[from] Assist"),
    ("|move|p2a: Clefable|Assist|p2a: Clefable",
     "|move|p2a: Clefable|Roar||[from] Assist|[still]"),
    ("|move|p2a: Clefable|Assist|p2a: Clefable",
     "|move|p2a: Clefable|Rollout|p1a: Zapdos|[from] Assist|[miss]"),
    ("|move|p2a: Clefable|Nature Power|p2a: Clefable",
     "|move|p2a: Clefable|Swift|p1a: Zapdos|[from] Nature Power"),
    ("|move|p2a: Clefable|Nature Power|p2a: Clefable",
     "|move|p2a: Clefable|Swift|p1a: Zapdos|[from] Nature Power|[miss]"),
]


@pytest.mark.parametrize("caller,called", SHAPES, ids=[s[1].split("|", 5)[-1] for s in SHAPES])
def test_a_called_move_is_read_as_the_callers_and_not_the_actors(caller, called):
    b = _battle()
    mon = _opp(b, caller, called)
    caller_id = caller.split("|")[3].lower().replace(" ", "")
    called_id = called.split("|")[3].lower().replace(" ", "")
    assert set(mon.moves) == {caller_id}, f"the called {called_id!r} leaked into the moveset"
    assert _pp(mon, caller_id) == mon.moves[caller_id].max_pp - 1, "one PP, on the caller's line"


def test_a_nested_call_reveals_only_the_outer_caller():
    """Metronome -> Nature Power -> Swift (a real sim sequence): only Metronome is the actor's."""
    b = _battle()
    mon = _opp(b, "|move|p2a: Clefable|Metronome|p2a: Clefable",
               "|move|p2a: Clefable|Nature Power|p2a: Clefable|[from] Metronome",
               "|move|p2a: Clefable|Swift|p1a: Zapdos|[from] Nature Power")
    assert set(mon.moves) == {"metronome"}
    assert _pp(mon, "metronome") == 15


def test_no_pressure_pp_for_a_called_move():
    """Our Zapdos has Pressure (inferred: its only ability). gen3 charges the called move's
    Pressure to NOBODY — the sim's Metronome PP after one use is 15/16 either way."""
    b = _battle()
    mon = _opp(b, "|move|p2a: Clefable|Metronome|p2a: Clefable",
               "|move|p2a: Clefable|Thunderbolt|p1a: Zapdos|[from] Metronome")
    assert b.active_pokemon.ability == "pressure", "fixture: the target is a Pressure mon"
    assert _pp(mon, "metronome") == 15


def test_the_called_move_still_owns_its_outcome_lines():
    """The called move is what HIT: the damaging-move capture and the tentative effectiveness
    follow it (Thunderbolt), not the caller (Metronome, bp 0)."""
    b = _battle()
    _opp(b, "|move|p2a: Clefable|Metronome|p2a: Clefable",
         "|move|p2a: Clefable|Thunderbolt|p1a: Zapdos|[from] Metronome")
    pending = b._pending_opp_damaging_move
    assert pending is not None and pending[1].move_id == "thunderbolt"


@pytest.mark.parametrize("caller", ["Metronome", "Assist", "Nature Power"])
def test_the_bare_caller_crashed_upstream(caller):
    """The bare form is the whole class, with and without the space the sim may omit."""
    assert f"[from] {caller}" in GEN3_BARE_MOVE_CALLERS
    assert f"[from]{caller}" in GEN3_BARE_MOVE_CALLERS
    b = _battle()
    _opp(b, f"|move|p2a: Clefable|{caller}|p2a: Clefable",
         f"|move|p2a: Clefable|Tackle|p1a: Zapdos|[from]{caller}")
    assert set(b.opponent_team["p2: Clefable"].moves) == {caller.lower().replace(" ", "")}


# ---------------------------------------------------------------------------------------------
# the multi-flag tail, for EVERY [from] source (it stranded a flag whatever the source was)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tail,revealed", [
    ("[from] Metronome|[miss]|[miss]", False),     # seen in the sim (Super Fang)
    ("[from] Sleep Talk|[miss]|[miss]", True),     # constructed: the same tail, another source
    ("[from] Mirror Move|[miss]|[still]", False),  # constructed: flags out of strip order
    ("[from] lockedmove|[miss]|[miss]", False),    # constructed
    ("[from] Assist|[miss]|[notarget]", False),    # constructed: flags out of strip order
])
def test_a_multi_flag_tail_parses_for_every_source(tail, revealed):
    b = _battle()
    caller = ["|move|p2a: Snorlax|Sleep Talk|p2a: Snorlax"] if "Sleep Talk" in tail else []
    _feed(b, "|switch|p1a: Zapdos|Zapdos|100/100", "|switch|p2a: Snorlax|Snorlax, M|100/100",
          "|turn|1", *caller, "|move|p2a: Snorlax|Body Slam|p1a: Zapdos|" + tail)
    assert ("bodyslam" in b.opponent_active_pokemon.moves) is revealed


def test_canonical_tail_touches_only_a_multi_flag_tail():
    same = ["", "move", "p2a: X", "Tackle", "p1a: Y", "[from] Metronome", "[miss]"]
    assert _canonical_from_tail(same) is same
    plain = ["", "move", "p2a: X", "Tackle", "p1a: Y", "[miss]", "[still]"]
    assert _canonical_from_tail(plain) is plain, "no [from]: the existing strip handles it"
    assert _canonical_from_tail(["", "move", "p2a: X", "Tackle", "p1a: Y", "[from] Metronome",
                                 "[miss]", "[miss]"])[5:] == ["[from] Metronome", "[miss]"]
    assert _canonical_from_tail(["", "move", "p2a: X", "Tackle", "", "[from] Assist", "[miss]",
                                 "[still]", "[notarget]"])[5:] == [
        "[from] Assist", "[notarget]", "[still]", "[miss]"]


# ---------------------------------------------------------------------------------------------
# the already-handled members of the class keep their reading (training input, byte-identical)
# ---------------------------------------------------------------------------------------------

def test_sleep_talk_still_reveals_the_called_own_move():
    """Sleep Talk calls the actor's OWN moves, so the called move IS revealed; unchanged."""
    b = _battle()
    _feed(b, "|switch|p1a: Zapdos|Zapdos|100/100", "|switch|p2a: Snorlax|Snorlax, M|100/100",
          "|turn|1", "|-status|p2a: Snorlax|slp", "|move|p2a: Snorlax|Sleep Talk|p2a: Snorlax",
          "|move|p2a: Snorlax|Body Slam|p1a: Zapdos|[from] Sleep Talk")
    assert set(b.opponent_active_pokemon.moves) == {"sleeptalk", "bodyslam"}


@pytest.mark.parametrize("src", ["Mirror Move", "Magic Coat", "Snatch"])
def test_the_redirect_family_is_unchanged(src):
    b = _battle()
    _feed(b, "|switch|p1a: Zapdos|Zapdos|100/100", "|switch|p2a: Pidgeot|Pidgeot, M|100/100",
          "|turn|1", f"|move|p2a: Pidgeot|Quick Attack|p1a: Zapdos|[from] {src}")
    assert not b.opponent_active_pokemon.moves
