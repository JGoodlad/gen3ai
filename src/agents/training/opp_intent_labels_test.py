"""Gates for the `α`/`β` privileged labels — DECISION vs CONSEQUENCE.

The whole point of this module is refusing to supervise on things that were not the opponent's
choice. A phaze is OUR Roar; a post-faint replacement is the rules. Labelling either as a
voluntary SWITCH teaches `β` to predict our own move, and — worse — makes `α`'s reported accuracy
un-interpretable, because the denominator would silently include rows where no decision happened.
"""
import pytest

from agents.training.opp_intent_labels import (KIND_MOVE, KIND_SWITCH, KIND_UNKNOWN,
                                               SWITCH_SLOT_NONE, build_opp_intent_label)

_NUMS = {"thunderbolt": 85, "icebeam": 58, "roar": 46}
_SLOTS = {"blissey": 3, "skarmory": 5}
_SPNUM = {"blissey": 242, "skarmory": 227, "ghostmon": 999}


def _num(mid):
    return _NUMS.get(mid)


def _slot(sp):
    return _SLOTS.get(sp)


def _spnum(sp):
    return _SPNUM.get(sp)


class _D:
    """Minimal TurnDelta stand-in — only the fields the label builder reads."""
    def __init__(self, move=None, switch=None, fainted=False, forced=False):
        self._move, self.opp_switch_to = move, switch
        self.opp_fainted, self.phase_is_forced_switch = fainted, forced

    @property
    def opp_resolved_move_id(self):
        return self._move


def test_a_plain_move_is_a_move():
    assert build_opp_intent_label(_D(move="thunderbolt"), _num, _slot, _spnum) == (KIND_MOVE, 85, SWITCH_SLOT_NONE, 0)


def test_a_voluntary_pivot_is_the_only_switch_beta_learns_from():
    assert build_opp_intent_label(_D(switch="blissey"), _num, _slot, _spnum) == (KIND_SWITCH, 0, 3, 242)


def test_a_phaze_is_NOT_their_decision():
    """Roar moved them. Supervising this would train beta to predict OUR move."""
    d = _D(move="roar", switch="skarmory")
    assert build_opp_intent_label(d, _num, _slot, _spnum)[0] == KIND_UNKNOWN


def test_a_post_faint_replacement_is_NOT_their_decision():
    d = _D(switch="skarmory", fainted=True)
    assert build_opp_intent_label(d, _num, _slot, _spnum)[0] == KIND_UNKNOWN


def test_a_forced_switch_window_carries_no_decision():
    assert build_opp_intent_label(_D(move="thunderbolt", forced=True), _num, _slot, _spnum)[0] == KIND_UNKNOWN


def test_an_unnameable_move_is_masked_not_guessed():
    assert build_opp_intent_label(_D(move="somethingnew"), _num, _slot, _spnum)[0] == KIND_UNKNOWN


def test_a_switch_to_an_unaddressable_slot_is_masked():
    """Beta is a POINTER over their six tokens — a target it cannot address must be masked."""
    kind, _, slot, sp = build_opp_intent_label(_D(switch="ghostmon"), _num, _slot, _spnum)
    assert kind == KIND_SWITCH and slot == SWITCH_SLOT_NONE
    assert sp == 999, "the SPECIES key must survive so content-addressing can still resolve it"


def test_no_delta_and_empty_delta_are_fully_masked():
    assert build_opp_intent_label(None, _num, _slot, _spnum) == (KIND_UNKNOWN, 0, SWITCH_SLOT_NONE, 0)
    assert build_opp_intent_label(_D(), _num, _slot, _spnum) == (KIND_UNKNOWN, 0, SWITCH_SLOT_NONE, 0)


@pytest.mark.parametrize("d", [_D(move="roar", switch="blissey"), _D(switch="blissey", fainted=True)])
def test_no_consequence_case_ever_yields_a_usable_switch_slot(d):
    kind, _, slot, _sp = build_opp_intent_label(d, _num, _slot, _spnum)
    assert kind == KIND_UNKNOWN and slot == SWITCH_SLOT_NONE


# ---------------------------------------------------------------- the alignment (GIGO risk)

import numpy as np
from agents.training.opp_intent_labels import align_labels_to_predictions

_FILL = -100


def test_alignment_shifts_the_label_back_one_row():
    """Row i's label is stored at row i+1; after aligning it must sit at row i."""
    labels = np.array([[0], [11], [22], [33]])          # [n_steps, n_envs]
    starts = np.zeros((4, 1))
    out = align_labels_to_predictions(labels, starts, _FILL)
    assert out[0, 0] == 11 and out[1, 0] == 22 and out[2, 0] == 33
    assert out[3, 0] == _FILL, "the last row has no successor in this rollout"


def test_a_label_is_NEVER_spliced_across_an_episode_boundary():
    """THE bug this function exists to prevent: row 2 starts a new battle, so row 1 has no pair."""
    labels = np.array([[0], [11], [99], [33]])
    starts = np.array([[0.0], [0.0], [1.0], [0.0]])     # row 2 opens a new episode
    out = align_labels_to_predictions(labels, starts, _FILL)
    assert out[1, 0] == _FILL, "row 1 was paired with the NEXT battle's first decision"
    assert out[0, 0] == 11 and out[2, 0] == 33, "unaffected rows must still align"


def test_alignment_is_per_env_column_and_does_not_bleed_sideways():
    labels = np.array([[0, 0], [11, 77], [22, 88]])
    starts = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    out = align_labels_to_predictions(labels, starts, _FILL)
    assert out[0, 0] == 11, "env 0 is unaffected by env 1's episode boundary"
    assert out[0, 1] == _FILL, "env 1's row 0 pairs with a new episode ⇒ masked"
    assert out[1, 0] == 22 and out[1, 1] == 88


def test_alignment_does_not_mutate_its_input():
    labels = np.array([[0], [11], [22]])
    before = labels.copy()
    align_labels_to_predictions(labels, np.zeros((3, 1)), _FILL)
    assert (labels == before).all(), "the buffer must not be corrupted in place"


def test_alignment_works_on_torch_tensors_too():
    import torch
    labels = torch.tensor([[0], [11], [22]])
    out = align_labels_to_predictions(labels, torch.zeros((3, 1)), _FILL)
    assert int(out[0, 0]) == 11 and int(out[2, 0]) == _FILL
