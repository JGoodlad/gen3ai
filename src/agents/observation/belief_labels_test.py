"""Unit tests for the pure hidden-opponent belief label builder."""
import numpy as np

from agents.observation.belief_labels import build_belief_labels, zero_belief_labels, PAD, BELIEF_MOVE_SLOTS
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
    team_moves = [["Earthquake", "Rock Slide?notinmap", "Rest", "Toxic", "Surf"]]  # 5 moves, one unknown
    revealed = []  # tyranitar hidden
    species_known = [0] + [1] * (TEAM_SIZE - 1)  # slot 0 believed
    bs, bm = build_belief_labels(team_species, team_moves, revealed, species_known, SP, MV, _norm)
    assert bs[0] == 248
    # unknown move skipped; capped at BELIEF_MOVE_SLOTS=4 -> earthquake,rest,toxic,surf
    row = [x for x in bm[0] if x != PAD]
    assert row == [MV["earthquake"], MV["rest"], MV["toxic"], MV["surf"]]
    assert len(bm[0]) == BELIEF_MOVE_SLOTS


def test_all_revealed_yields_all_pad():
    team_species = ["Skarmory", "Blissey"]
    team_moves = [["Spikes"], ["Toxic"]]
    revealed = ["Skarmory", "Blissey"]
    species_known = [1, 1, 1, 1, 1, 1]
    bs, bm = build_belief_labels(team_species, team_moves, revealed, species_known, SP, MV, _norm)
    assert (bs == PAD).all() and (bm == PAD).all()


def test_unknown_species_skipped_not_raised():
    team_species = ["Skarmory", "Mystmon"]  # Mystmon not in map
    team_moves = [["Spikes"], ["Toxic"]]
    revealed = ["Skarmory"]
    species_known = [1, 0, 1, 1, 1, 1]  # slot 1 believed
    bs, bm = build_belief_labels(team_species, team_moves, revealed, species_known, SP, MV, _norm)
    # only believed slot 1, but the lone hidden mon is unmappable -> stays PAD, no crash
    assert bs[1] == PAD


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
