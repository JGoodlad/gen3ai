"""Does a TRAILING BLANK LINE in a Showdown paste still become an empty 7th Pokemon?

Sits beside `poke_env_fork_gate_test.py` at the `src/` root for the same reason: its subject is
the VENDORED FORK at `src/poke_env/`, which is not ours to litter with test files, and the defect
it pins is one the fork and upstream poke-env DISAGREE about.

## The hazard, stated plainly

`Teambuilder.parse_showdown_team` splits a paste on `"\\n\\n"`. Before 2026-09-14 it skipped a
chunk only when that chunk was **exactly** `""`. Every Metamon `competitive` gen3ou team file ends
`"...\\n\\n\\n"`, so its final chunk is `"\\n"` — not `""` — and became a seventh, empty
`TeambuilderPokemon`. `Gen3Teambuilder` then packed seven entries and Showdown answered:

    |popup|Your team was rejected for the following reasons:||||- You may only bring up to
    6 Pokemon (your team has 7).

**The symptom is a HANG, not an error.** A rejected challenge never becomes a battle, so a
head-to-head series simply sits there; it killed a 2026-09-14 pilot at game 2 after quietly
biasing the team draw toward the few files that happened to end without a blank line. Three
further properties make it the expensive class:

* `utils.bridge.team_validator.validate_teams_locally` PASSES the file — it validates the paste,
  and the seventh entry is created by the PACK.
* **Upstream poke-env 0.8.3.3 parses the same file correctly** (6 mons, measured), so a team file
  crossing between the two packages parses differently depending on whose parser reads it. That is
  the exact mirror of the de-risk's H1, where upstream was the wrong one — two disagreements on
  team FILES, in opposite directions.
* Nothing in the suite exercised a paste with a trailing blank line, so the class was invisible.

Measurement and both hazards:
`designs/research_state/measurements/metamon_matched_regime_2026-09-14/` (H-A) and
`designs/research_state/measurements/metamon_derisk_2026-09-14/` (H1).

## What each test pins

1. `test_a_trailing_blank_line_is_not_a_seventh_pokemon` — the fix itself. FAILS on revert to
   `if ps_mon == "":`, because the paste ends `"\\n\\n\\n"` and the count goes 6 -> 7.
2. `test_interior_blank_runs_do_not_create_empty_pokemon` — the same class from the other side: a
   paste with a doubled separator between two mons.
3. `test_a_six_mon_paste_still_packs_and_is_unchanged` — the control. A gate that rejected
   everything would pass 1-2 and be useless.
4. `test_the_pack_path_refuses_an_oversize_team` — the SECOND line of defence. `Gen3Teambuilder`
   raises on a packed team of more than six, so any future member of this class is an ERROR
   naming the offender rather than a series that hangs. It reinstates the pre-fix parser on the
   class under test, because with the fix in place nothing else can produce a 7-mon pack.
5. `test_local_validation_sees_a_spelled_out_seventh_but_not_a_blank_line` — why 4 is not
   redundant, measured: the existing paste validator rejects a spelled-out seventh mon and ACCEPTS
   the trailing-blank paste that hung a series.

Cost: milliseconds for 1-2; test 3 pays one Node validation.
"""
from __future__ import annotations

import pytest

from poke_env.teambuilder import Teambuilder

# A minimal but REAL gen3ou paste. Six mons, no nicknames (a nickname on an item-less line is the
# sibling hazard H1 — kept out of this file so the two are never confused), explicit Hidden Power
# IVs where they matter, so it packs and validates without `fix_gen3_hp_ivs` having to intervene.
SIX_MONS = """Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 8 Def / 248 SpD
Careful Nature
IVs: 0 Atk
- Spikes
- Protect
- Roar
- Toxic

Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 Def / 252 SpA / 4 Spe
Modest Nature
IVs: 0 Atk
- Soft-Boiled
- Ice Beam
- Toxic
- Fire Blast

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 248 HP / 196 Atk / 12 Def / 52 SpD
Adamant Nature
- Focus Punch
- Rock Slide
- Earthquake
- Crunch

Swampert @ Leftovers
Ability: Torrent
EVs: 240 HP / 136 Def / 40 SpA / 48 SpD / 44 Spe
Relaxed Nature
- Earthquake
- Ice Beam
- Hydro Pump
- Protect

Gengar @ Leftovers
Ability: Levitate
EVs: 168 HP / 164 SpD / 176 Spe
Timid Nature
IVs: 0 Atk
- Will-O-Wisp
- Thunderbolt
- Ice Punch
- Explosion

Starmie @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Hydro Pump
- Ice Beam
- Thunderbolt
- Rapid Spin"""

#: The shape every Metamon `competitive` gen3ou file has on disk, and the one that broke.
TRAILING_BLANK = SIX_MONS + "\n\n\n"


def test_a_trailing_blank_line_is_not_a_seventh_pokemon() -> None:
    """THE FIX. A paste ending "\\n\\n\\n" must parse to exactly six Pokemon.

    Reverting `if not ps_mon.strip(): continue` to `if ps_mon == "": continue` makes this read 7.
    """
    mons = Teambuilder.parse_showdown_team(TRAILING_BLANK)
    assert len(mons) == 6, (
        f"a trailing blank line produced {len(mons)} Pokemon, not 6 — "
        "`parse_showdown_team` is back to skipping only an EXACTLY-empty chunk, and a packed "
        "7-mon team makes Showdown reject the challenge, which HANGS the match rather than "
        "erroring (metamon_matched_regime_2026-09-14 hazard H-A)"
    )
    # ... and the six it found are the six that were written, not five plus a ghost.
    assert [m.species or m.nickname for m in mons][:2] == ["Skarmory", "Blissey"]


@pytest.mark.parametrize(
    "paste, label",
    [
        (SIX_MONS, "no trailing newline at all"),
        (SIX_MONS + "\n", "one trailing newline"),
        (SIX_MONS + "\n\n", "one trailing blank line"),
        (TRAILING_BLANK, "two trailing blank lines (the Metamon competitive shape)"),
        (SIX_MONS + "\n\n\n\n\n", "four trailing blank lines"),
        (SIX_MONS + "\n\n   \n\t\n", "trailing whitespace-only lines"),
        ("\n\n" + SIX_MONS, "leading blank lines"),
    ],
)
def test_blank_padding_anywhere_still_parses_to_six(paste: str, label: str) -> None:
    """The class, not the instance: no amount of blank padding may change the mon COUNT."""
    assert len(Teambuilder.parse_showdown_team(paste)) == 6, (
        f"{label}: parsed to {len(Teambuilder.parse_showdown_team(paste))} Pokemon, not 6"
    )


def test_interior_blank_runs_do_not_create_empty_pokemon() -> None:
    """A doubled separator INSIDE the paste is the same defect one line further in."""
    doubled = SIX_MONS.replace("\n\nTyranitar", "\n\n\nTyranitar", 1)
    assert len(Teambuilder.parse_showdown_team(doubled)) == 6


def test_a_six_mon_paste_still_packs_and_is_unchanged() -> None:
    """THE CONTROL. A parser that dropped everything would pass every test above."""
    packed = Teambuilder.join_team(Teambuilder.parse_showdown_team(SIX_MONS))
    assert len(packed.split("]")) == 6
    # The trailing-blank variant must pack to the SAME bytes — the fix removes a ghost, it does
    # not alter the six real entries.
    assert Teambuilder.join_team(Teambuilder.parse_showdown_team(TRAILING_BLANK)) == packed


@pytest.mark.integration
def test_the_pack_path_refuses_an_oversize_team(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE SECOND LINE OF DEFENCE, and the one that makes the hang unrepresentable.

    The guard has to live at the PACK, because that is where the seventh entry is created. Note
    what `validate_teams_locally` does and does not cover, measured by the sibling test below: it
    REJECTS a paste that genuinely spells out seven Pokemon, and it ACCEPTS the paste that broke
    H-A — six real mons and a trailing blank line — because the paste is fine and the pack is not.
    So the only way to reach the guard is to reinstate the defect, which is what this does: the
    pre-2026-09-14 parser is restored on the class under test and the same file shape is fed
    through the real `Gen3Teambuilder` constructor.

    Patching the CLASS attribute (not a module global) is deliberate — `Gen3Teambuilder` resolves
    `self.parse_showdown_team` at call time, so the stub genuinely reaches the code under test.
    """
    from utils.teambuilder import Gen3Teambuilder

    def reverted_parse(team: str):
        """`Teambuilder.parse_showdown_team` exactly as it was before the fix."""
        from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon
        return [TeambuilderPokemon.from_showdown(m) for m in team.split("\n\n") if m != ""]

    monkeypatch.setattr(Gen3Teambuilder, "parse_showdown_team", staticmethod(reverted_parse))

    with pytest.raises(ValueError) as excinfo:
        Gen3Teambuilder(TRAILING_BLANK)
    message = str(excinfo.value)
    assert "more than 6" in message, message
    # The message must name the HANG, because the reader's next question is always "so what?"
    assert "HANG" in message, message
    assert "->7" in message.replace(" ", ""), message


@pytest.mark.integration
def test_local_validation_sees_a_spelled_out_seventh_but_not_a_blank_line() -> None:
    """WHY THE GUARD IS NOT REDUNDANT, measured rather than asserted.

    `validate_teams_locally` is the existing defence. It catches a paste that really does carry
    seven Pokemon — and it passes the one that hung a series, because there the paste holds six and
    the seventh is manufactured by the pack. That asymmetry is the entire argument for a
    pack-site guard, so it is pinned here rather than left in a comment.
    """
    from utils.bridge.team_validator import validate_teams_locally

    seventh = """

Jirachi @ Leftovers
Ability: Serene Grace
EVs: 252 HP / 224 SpD / 32 Spe
Careful Nature
- Wish
- Protect
- Body Slam
- Fire Punch"""

    spelled_out, blank_line = validate_teams_locally("gen3ou", [SIX_MONS + seventh, TRAILING_BLANK])
    assert spelled_out.get("valid") is False, spelled_out
    assert any("6" in e for e in spelled_out.get("errors", [])), spelled_out
    assert blank_line.get("valid") is True, (
        "the paste that hung a 2026-09-14 series validates CLEAN — which is the point: the defect "
        f"is created by the pack, not the paste. {blank_line}"
    )


@pytest.mark.integration
def test_a_trailing_blank_line_no_longer_reaches_the_pack_guard() -> None:
    """End to end: the exact file shape that hung a series now builds cleanly.

    This is the test that would have caught H-A in the first place — it goes through the real
    `Gen3Teambuilder` construction path, not just the parser.
    """
    from utils.teambuilder import Gen3Teambuilder

    tb = Gen3Teambuilder(TRAILING_BLANK)
    assert len(tb.packed_teams) == 1
    assert len(tb.packed_teams[0].split("]")) == 6
