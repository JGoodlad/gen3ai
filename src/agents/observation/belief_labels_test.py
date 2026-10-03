"""Unit tests for the pure hidden-opponent belief label builder."""

import pytest

from agents.observation.belief_labels import (
    build_belief_labels, build_known_move_labels, zero_belief_labels, zero_known_moves,
    build_known_spread_labels, zero_spread_labels, N_SPREAD_STATS,
    PAD, BELIEF_MOVE_SLOTS, BeliefLabelError,
)
from agents.observation.constants import TEAM_SIZE


def _norm(s):
    return "".join(c for c in s.lower() if c.isalnum())


SP = {"skarmory": 227, "blissey": 242, "claydol": 344, "tyranitar": 248, "salamence": 373, "swampert": 260}
MV = {"spikes": 191, "roar": 46, "toxic": 92, "earthquake": 89, "icebeam": 58, "surf": 57, "rest": 156}


def test_assigns_hidden_mons_to_trailing_believed_slots_in_num_order():
    team_species = ["Skarmory", "Blissey", "Claydol", "Tyranitar", "Salamence", "Swampert"]
    team_moves = [["Spikes", "Roar"], ["Toxic"], ["Earthquake"], ["Earthquake"], ["Earthquake"], ["Surf"]]
    # Skarmory + Blissey revealed -> slots 0,1 known; 2..5 believed.
    revealed = ["Skarmory", "Blissey"]
    species_known = [1, 1, 0, 0, 0, 0]
    bs, bm = build_belief_labels(team_species, team_moves, revealed, species_known, SP, MV, _norm)
    # revealed slots not scored
    assert bs[0] == PAD and bs[1] == PAD
    # believed slots (2..5) get the 4 hidden mons sorted by species num:
    # claydol 344, tyranitar 248, salamence 373, swampert 260 -> sorted: 248,260,344,373
    assert list(bs[2:]) == [248, 260, 344, 373]


def test_move_labels_padded_and_truncated():
    team_species = ["Tyranitar"]
    team_moves = [["Earthquake", "Rest", "Toxic", "Surf", "Rock Slide?notinmap"]]  # 5 moves; the 5th is past the cap
    revealed = []  # tyranitar hidden
    species_known = [0] + [1] * (TEAM_SIZE - 1)  # slot 0 believed
    bs, bm = build_belief_labels(team_species, team_moves, revealed, species_known, SP, MV, _norm)
    assert bs[0] == 248
    # capped at BELIEF_MOVE_SLOTS=4 -> earthquake,rest,toxic,surf
    assert list(bm[0]) == [MV["earthquake"], MV["rest"], MV["toxic"], MV["surf"]]
    assert len(bm[0]) == BELIEF_MOVE_SLOTS


def test_an_unmapped_move_RAISES():
    """F-X5-3: an unmapped move inside the cap was skipped (the next move slid into its place)."""
    with pytest.raises(BeliefLabelError, match="notinmap"):
        build_belief_labels(["Tyranitar"], [["Earthquake", "Rock Slide?notinmap", "Rest"]], [],
                            [0] + [1] * (TEAM_SIZE - 1), SP, MV, _norm)


def test_all_revealed_yields_all_pad():
    team_species = ["Skarmory", "Blissey"]
    team_moves = [["Spikes"], ["Toxic"]]
    revealed = ["Skarmory", "Blissey"]
    species_known = [1, 1, 1, 1, 1, 1]
    bs, bm = build_belief_labels(team_species, team_moves, revealed, species_known, SP, MV, _norm)
    assert (bs == PAD).all() and (bm == PAD).all()


def test_unknown_species_RAISES_never_skipped():
    """F-X5-3: a hidden mon with no species num was SKIPPED, undercounting the label."""
    team_species = ["Skarmory", "Mystmon"]  # Mystmon not in map
    team_moves = [["Spikes"], ["Toxic"]]
    with pytest.raises(BeliefLabelError, match="Mystmon"):
        build_belief_labels(team_species, team_moves, ["Skarmory"], [1, 0, 1, 1, 1, 1], SP, MV, _norm)


def test_a_revealed_species_off_the_team_RAISES():
    with pytest.raises(BeliefLabelError, match="not on the privileged team"):
        build_belief_labels(["Skarmory", "Blissey"], [["Spikes"], ["Toxic"]], ["Tyranitar"],
                            [1, 0, 0, 0, 0, 0], SP, MV, _norm)


def test_more_hidden_mons_than_believed_slots_RAISES():
    with pytest.raises(BeliefLabelError, match="believed slots"):
        build_belief_labels(["Skarmory", "Blissey", "Claydol"], [["Spikes"], ["Toxic"], ["Earthquake"]], [],
                            [1, 1, 1, 1, 1, 0], SP, MV, _norm)


def test_fewer_hidden_than_slots_leaves_pad():
    team_species = ["Skarmory", "Blissey", "Claydol"]
    team_moves = [["Spikes"], ["Toxic"], ["Earthquake"]]
    revealed = ["Skarmory", "Blissey"]  # 1 hidden (claydol), but 4 believed slots
    species_known = [1, 1, 0, 0, 0, 0]
    bs, bm = build_belief_labels(team_species, team_moves, revealed, species_known, SP, MV, _norm)
    assert bs[2] == SP["claydol"]
    assert list(bs[3:]) == [PAD, PAD, PAD]


def test_zero_belief_labels_shape():
    bs, bm = zero_belief_labels()
    assert bs.shape == (TEAM_SIZE,) and bm.shape == (TEAM_SIZE, BELIEF_MOVE_SLOTS)
    assert (bs == PAD).all() and (bm == PAD).all()


# --------------------------------------------------------------------------- known-move labels


def test_known_moves_full_moveset_at_revealed_slots():
    """Each REVEALED slot gets the FULL privileged moveset of the species revealed there (so the head
    learns the as-yet-unrevealed moves). Believed slots stay PAD."""
    team_species = ["Skarmory", "Blissey", "Claydol"]
    team_moves = [["Spikes", "Roar", "Toxic", "Rest"], ["Toxic", "Surf"], ["Earthquake"]]
    # Skarmory + Blissey revealed (slots 0,1); Claydol hidden (slot 2 believed).
    species_known = [1, 1, 0, 1, 1, 1]
    revealed_in_slot_order = ["Skarmory", "Blissey"]
    km = build_known_move_labels(revealed_in_slot_order, team_species, team_moves, species_known, MV, _norm)
    assert list(km[0]) == [MV["spikes"], MV["roar"], MV["toxic"], MV["rest"]]   # FULL set, incl. unrevealed
    assert [x for x in km[1] if x != PAD] == [MV["toxic"], MV["surf"]]
    assert (km[2] == PAD).all()   # believed slot not scored by the known path


def test_known_moves_truncates_and_raises_on_unmapped():
    team_species = ["Tyranitar"]
    species_known = [1] + [0] * (TEAM_SIZE - 1)
    km = build_known_move_labels(["Tyranitar"], team_species, [["Earthquake", "Rest", "Toxic", "Surf", "Bogusmove"]],
                                 species_known, MV, _norm)
    assert list(km[0]) == [MV["earthquake"], MV["rest"], MV["toxic"], MV["surf"]]   # capped at 4
    with pytest.raises(BeliefLabelError, match="Bogusmove"):
        build_known_move_labels(["Tyranitar"], team_species, [["Earthquake", "Bogusmove"]], species_known, MV, _norm)


def test_known_moves_revealed_species_absent_from_team_RAISES():
    """F-X5-3: a revealed slot whose species isn't on the privileged team was a silent PAD."""
    with pytest.raises(BeliefLabelError, match="Mystmon"):
        build_known_move_labels(["Mystmon"], ["Skarmory"], [["Spikes"]], [1, 0, 0, 0, 0, 0], MV, _norm)


def test_zero_known_moves_shape():
    km = zero_known_moves()
    assert km.shape == (TEAM_SIZE, BELIEF_MOVE_SLOTS) and (km == PAD).all()


# ---- SPREAD belief labels (gen3_unified_spread_belief_v1) ----
_SPREAD = {  # normalised species -> TRUE derived stats {atk,def,spa,spd,spe}
    "tyranitar": [367, 256, 203, 237, 159],
    "skarmory": [259, 317, 104, 177, 179],
    "blissey": [49, 75, 207, 437, 95],
}


def test_spread_labels_revealed_slots_only_matched_by_species():
    # 2 revealed (Tyranitar, Skarmory), 4 believed → only the 2 revealed slots scored, by species.
    species_known = [1, 1, 0, 0, 0, 0]
    sp, mask = build_known_spread_labels(["Tyranitar", "Skarmory"], _SPREAD, species_known, _norm)
    assert sp.shape == (TEAM_SIZE, N_SPREAD_STATS) and mask.shape == (TEAM_SIZE,)
    assert mask.tolist() == [1, 1, 0, 0, 0, 0]
    assert sp[0].tolist() == [367, 256, 203, 237, 159]   # slot 0 = Tyranitar (atk,def,spa,spd,spe order)
    assert sp[1].tolist() == [259, 317, 104, 177, 179]   # slot 1 = Skarmory
    assert (sp[2:] == 0).all()                           # believed slots untouched


def test_spread_labels_unmappable_or_incomplete_species_RAISE():
    """F-X5-3: an absent or incomplete true spread was a silent mask 0."""
    bad = dict(_SPREAD); bad["claydol"] = [120, None, 140, 130, 100]   # incomplete
    with pytest.raises(BeliefLabelError, match="Ditto"):
        build_known_spread_labels(["Tyranitar", "Ditto"], bad, [1, 1, 0, 0, 0, 0], _norm)
    with pytest.raises(BeliefLabelError, match="Claydol"):
        build_known_spread_labels(["Tyranitar", "Claydol"], bad, [1, 1, 0, 0, 0, 0], _norm)


def test_spread_labels_all_believed_yields_empty_mask():
    sp, mask = build_known_spread_labels([], _SPREAD, [0, 0, 0, 0, 0, 0], _norm)
    assert (mask == 0).all() and (sp == 0).all()


def test_zero_spread_labels_shape():
    sp, mask = zero_spread_labels()
    assert sp.shape == (TEAM_SIZE, N_SPREAD_STATS) and (sp == 0).all()
    assert mask.shape == (TEAM_SIZE,) and (mask == 0).all()
