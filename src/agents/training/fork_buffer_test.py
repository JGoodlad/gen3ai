"""Gates for the FORK ARM's FILL TABLE (`gen3_fork_v1`) — `agents.training.fork_buffer.FILL`.

What survives of the Python arm's buffer gates after deletion pass L5: the table the Rust fork pass
(`rust_rollout/fork.py`) fills every injected row from. The branch-row builder, the GAE and the buffer
subclass those gates covered are deleted with the arm; THE MASK RULE and the prefix-counted-once rule are
pinned by `rust_rollout/fork_test.py` (the FIFO / hand-GAE test and the `fork_pg_m` test).
"""
from __future__ import annotations

import numpy as np

from agents.training.fork_arm import PG_MASK_KEY
from agents.training.fork_buffer import FILL, refusal_text, unfillable_keys

CTX = {"outcome": 1.0, "pg_mask": np.asarray([0.0, 1.0], dtype=np.float32)}


def _fill(key, n=2):
    builder, _why = FILL[key]
    return builder(n, CTX)


# ── the fill table ───────────────────────────────────────────────────────────────────────────
def test_every_key_the_production_obs_dict_can_carry_is_filled():
    """A key that is neither is the failure mode this table exists to make impossible: an injected
    row would silently carry whatever a heuristic guessed."""
    for key in ("win_target", "win_mask", "win_margin", "opp_class", PG_MASK_KEY,
                "belief_species", "belief_moves", "known_moves", "belief_spread",
                "belief_spread_mask", "belief_nature", "belief_nature_mask", "belief_ev",
                "belief_ev_mask", "hp_type_label", "hp_type_mask", "item_label", "item_mask",
                "opp_action_kind", "opp_action_num", "opp_switch_slot", "opp_switch_species"):
        assert key in FILL, key


def test_an_unknown_key_refuses_with_its_own_name_and_a_generic_reason():
    bad = unfillable_keys(["observation", "action_mask", "win_target", "some_new_key"])
    assert bad == ["some_new_key"]
    text = refusal_text(bad)
    assert "some_new_key" in text and "REFUSED" in text


def test_the_refusal_names_the_flag_rather_than_only_the_key():
    assert "--fork-fraction" in refusal_text(["some_new_key"])


def test_every_label_key_a_branch_cannot_supply_is_filled_NOT_SCORED():
    for masked in ("item_mask", "hp_type_mask", "belief_spread_mask"):
        assert (_fill(masked) == 0.0).all(), masked
    assert (_fill("item_label") == -1).all()
    from agents.training.opp_intent_labels import KIND_UNKNOWN
    assert (_fill("opp_action_kind") == KIND_UNKNOWN).all()


def test_the_opponent_class_of_an_injected_row_says_POOL_not_the_parents_class():
    """The ecology approximation, LABELLED rather than hidden: the branch really was played
    against a self-like opponent, so the class tag says so and `fork/bot_share` prices it."""
    assert (_fill("opp_class") == 1).all()


def test_the_pg_mask_and_the_win_label_come_from_the_ctx_not_a_constant():
    """The two keys the arm MEANS: the fork step is out of the policy term and the branch's own outcome
    is the win label (`rust_rollout/fork.py` hands the ctx per row)."""
    assert _fill(PG_MASK_KEY).reshape(-1).tolist() == [0.0, 1.0]
    assert (_fill("win_target") == 1.0).all() and (_fill("win_mask") == 1.0).all()
