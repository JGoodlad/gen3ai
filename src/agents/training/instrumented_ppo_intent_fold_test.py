"""K8 — `intent_fold`, the opponent-intent fold's ENTRY POINT inside region R1 (`micro_step`).

It dispatches to the X5 flat pointer's fold (`flat_intent_fold`) whenever the extractor published
`flat_intent_logits`, and returns `None` — the block SKIPPED, no term, no metric — otherwise. The
flat fold's own arithmetic, metrics and fullgraph trace are pinned in
`agents/model/flat_intent_test.py`; the label weight in `agents/model/intent_label_bot_weight_test.py`.
"""
from __future__ import annotations

import torch

from agents.model.flat_intent import FlatIntentInputs, flat_width, slot_col
from agents.training.instrumented_ppo.flat_intent_fold import flat_intent_fold_tensors
from agents.training.instrumented_ppo.intent_fold import intent_fold

K, M, S, TEAM = 6, 400, 400, 6


def _fi(B: int) -> FlatIntentInputs:
    live = torch.ones(B, flat_width(K), dtype=torch.bool)
    live[:, 3:K] = False
    live[:, slot_col(K, 0)] = False
    return FlatIntentInputs(
        k=K, live=live, cand_ids=torch.zeros(B, flat_width(K), dtype=torch.long),
        log_pi=torch.zeros(B, flat_width(K)),
        seat_nums=torch.tensor([[89, 57, 237, 0, 0, 0]]).expand(B, -1).clone(),
        seat_on=torch.tensor([[True, True, True, False, False, False]]).expand(B, -1).clone(),
        hp_seat=torch.zeros(B, K, dtype=torch.bool), beyond=torch.zeros(B, M, dtype=torch.bool),
        slot_species=torch.zeros(B, TEAM, dtype=torch.long),
        slot_is_hypothesis=torch.zeros(B, TEAM, dtype=torch.bool), in_tail=torch.zeros(B, S, dtype=torch.bool))


class _FE:
    """The two reads `intent_fold` / `flat_intent_fold` make of the extractor."""

    def __init__(self, flat, fi):
        self._flat = flat
        self.last_flat_intent = fi

    def belief_supervision(self, key):
        assert key == "flat_intent_logits", key
        return self._flat


def _obs(B: int):
    return {"opp_action_kind": torch.tensor([0, 0, 1, 2])[:B], "opp_action_num": torch.tensor([89, 57, 0, 0])[:B],
            "opp_switch_slot": torch.tensor([-100, -100, 2, -100])[:B],
            "opp_switch_species": torch.zeros(B, dtype=torch.long),
            "opp_class": torch.tensor([[0], [1], [1], [0]])[:B]}


def test_no_flat_pointer_is_a_skipped_block():
    assert intent_fold(_FE(None, _fi(4)), _obs(4), intent_coef=0.05, bot_label_weight=0.25) is None


def test_the_flat_pointer_is_folded_by_the_flat_fold():
    flat = torch.randn(4, flat_width(K), generator=torch.Generator().manual_seed(0))
    out = intent_fold(_FE(flat, _fi(4)), _obs(4), intent_coef=0.05, bot_label_weight=0.25)
    obs = _obs(4)
    want = flat_intent_fold_tensors(flat, _fi(4), kind=obs["opp_action_kind"], num=obs["opp_action_num"],
                                    switch_slot=obs["opp_switch_slot"], switch_species=obs["opp_switch_species"],
                                    opp_class=obs["opp_class"], intent_coef=0.05, bot_label_weight=0.25)
    assert out is not None and bool(out.intent_present)
    assert torch.equal(out.intent_term, want.intent_term)
    assert list(out.metrics) == list(want.metrics)
    assert "opp_intent/flat_loss" in out.metrics


def test_a_flat_pointer_without_a_label_in_the_obs_is_a_skipped_block():
    flat = torch.zeros(4, flat_width(K))
    assert intent_fold(_FE(flat, _fi(4)), {}, intent_coef=0.05, bot_label_weight=1.0) is None
