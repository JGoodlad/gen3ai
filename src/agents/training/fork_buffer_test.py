"""Gates for the FORK ARM's buffer surgery (`gen3_fork_v1`) — `agents.training.fork_buffer`.

Four properties, and each one is a way the arm could put a WRONG number into the PPO objective
rather than merely fail:

1. **the branch's return is the ordinary GAE/lambda-return** — cross-checked against SB3's own
   `RolloutBuffer.compute_returns_and_advantage` on the same inputs, so the specialisation to a
   whole-episode trajectory is proven equal to the recursion the collected rows use rather than
   asserted to be.
2. **THE MASK RULE is uniform** — the fork step is out of the policy term for EVERY branch.
3. **the prefix is counted ONCE** — a branch's rows begin AT the fork step.
4. **OFF is bit-identical** — with nothing injected, `get()` is upstream's generator, called.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer

from agents.training.fork_arm import PG_MASK_KEY
from agents.training.fork_buffer import (
    FILL, REFUSE_KEYS, ForkRolloutBuffer, branch_rewards, build_branch_rows, concat_blocks,
    fork_buffer_class, gae, refusal_text, unfillable_keys,
)

D, A = 6, 11


# ── 1. the branch's value target ─────────────────────────────────────────────────────────────
def test_gae_equals_sb3s_own_recursion_on_a_whole_episode():
    rng = np.random.default_rng(3)
    n = 12
    rewards = np.zeros(n, dtype=np.float32)
    rewards[-1] = 1.0
    values = rng.random(n).astype(np.float32)
    gamma, lam = 1.0, 0.95

    buf = RolloutBuffer(n, spaces.Box(-1, 1, (1,), np.float32), spaces.Discrete(2),
                        gae_lambda=lam, gamma=gamma, n_envs=1)
    for t in range(n):
        buf.add(np.zeros((1, 1), np.float32), np.zeros((1, 1)), np.array([rewards[t]]),
                np.array([1.0 if t == 0 else 0.0]), th.tensor([[values[t]]]), th.tensor([0.0]))
    buf.compute_returns_and_advantage(last_values=th.tensor([[0.0]]), dones=np.array([True]))

    adv, ret = gae(rewards, values, gamma, lam)
    np.testing.assert_allclose(adv, buf.advantages.reshape(-1), rtol=0, atol=1e-6)
    np.testing.assert_allclose(ret, buf.returns.reshape(-1), rtol=0, atol=1e-6)


def test_a_branch_reward_stream_is_zeros_then_the_win_indicator():
    """🚨 The whole reason the arm REFUSES any critic but `winprob`: this is reconstructible from
    the outcome bit alone only because the reward stream IS the terminal indicator."""
    np.testing.assert_array_equal(branch_rewards(4, 1.0), [0, 0, 0, 1])
    np.testing.assert_array_equal(branch_rewards(4, 0.0), [0, 0, 0, 0])
    assert branch_rewards(0, 1.0).shape == (0,)


def test_a_losing_branch_at_gamma_one_has_a_zero_return_everywhere():
    adv, ret = gae(branch_rewards(5, 0.0), np.zeros(5, np.float32), 1.0, 1.0)
    np.testing.assert_allclose(ret, np.zeros(5), atol=1e-7)


# ── 2/3. the mask rule and the prefix ────────────────────────────────────────────────────────
def _block(n=4, outcome=1.0, keys=("observation", "action_mask", "win_target", "win_mask",
                                   PG_MASK_KEY)):
    obs = np.arange(n * D, dtype=np.float32).reshape(n, D)
    masks = np.ones((n, A), dtype=np.int8)
    return build_branch_rows(obs=obs, masks=masks, actions=np.arange(n),
                             values=np.zeros(n, np.float32), log_probs=np.zeros(n, np.float32),
                             outcome=outcome, gamma=1.0, gae_lambda=1.0, obs_keys=list(keys),
                             mask_dims=A)


def test_the_fork_step_is_masked_out_of_the_policy_term_and_nothing_else_is():
    b = _block(5)
    m = b["observations"][PG_MASK_KEY].reshape(-1)
    assert m[0] == 0.0
    assert (m[1:] == 1.0).all()
    assert b["n_masked"] == 1


def test_the_mask_rule_is_uniform_across_branches():
    """Masking only the RANDOM branch would leave top-1/top-2 as the only fork-step rows in the
    policy term — an on-policy re-weighting of exactly the contested states the arm selects for,
    shipped under a flag whose declared job is to buy VALUE data."""
    for _name in ("top1", "top2", "rand"):
        assert _block(3)["observations"][PG_MASK_KEY].reshape(-1)[0] == 0.0


def test_the_fork_step_stays_fully_in_the_value_terms():
    b = _block(3, outcome=1.0)
    assert b["observations"]["win_mask"].reshape(-1)[0] == 1.0
    assert b["observations"]["win_target"].reshape(-1)[0] == 1.0
    assert b["returns"].reshape(-1)[0] == pytest.approx(1.0)


def test_the_prefix_is_counted_once_rows_begin_at_the_fork_step():
    """A branch contributes exactly its own captured decisions — row 0 the fork STATE, row 1 the
    successor — and never the turns before it, which are already in the buffer as the parent's."""
    obs = np.arange(3 * D, dtype=np.float32).reshape(3, D)
    b = build_branch_rows(obs=obs, masks=np.ones((3, A), np.int8), actions=np.zeros(3, int),
                          values=np.zeros(3, np.float32), log_probs=np.zeros(3, np.float32),
                          outcome=1.0, gamma=1.0, gae_lambda=1.0,
                          obs_keys=["observation", "action_mask", PG_MASK_KEY], mask_dims=A)
    assert b["n_rows"] == 3
    np.testing.assert_array_equal(b["observations"]["observation"][0], obs[0])


def test_three_branches_of_one_fork_inject_the_fork_state_three_times_with_three_actions():
    """That is the EXPLORING START, not a duplicate: the rows differ in the action, the return and
    the successor — and none of them contributes a policy gradient."""
    blocks = [_block(3, outcome=y) for y in (1.0, 0.0, 1.0)]
    for k, b in enumerate(blocks):
        b["actions"][0, 0] = k
    cat = concat_blocks(blocks)
    assert cat["n_rows"] == 9 and cat["n_masked"] == 3
    fork_rows = cat["observations"][PG_MASK_KEY].reshape(-1) == 0.0
    assert int(fork_rows.sum()) == 3
    assert sorted(cat["actions"].reshape(-1)[fork_rows].tolist()) == [0, 1, 2]


def test_concat_of_nothing_is_none():
    assert concat_blocks([]) is None
    assert concat_blocks([{"n_rows": 0}]) is None


# ── the fill table ───────────────────────────────────────────────────────────────────────────
def test_every_key_the_production_obs_dict_can_carry_is_either_filled_or_refused_by_name():
    """A key that is neither is the failure mode this table exists to make impossible: an injected
    row would silently carry whatever a heuristic guessed."""
    from agents.observation.true_team import TRUE_TEAM_KEY
    for key in ("win_target", "win_mask", "win_margin", "win_row_w", "opp_class", PG_MASK_KEY,
                "belief_species", "belief_moves", "known_moves", "belief_spread",
                "belief_spread_mask", "belief_nature", "belief_nature_mask", "belief_ev",
                "belief_ev_mask", "hp_type_label", "hp_type_mask", "item_label", "item_mask",
                "opp_action_kind", "opp_action_num", "opp_switch_slot", "opp_switch_species"):
        assert key in FILL, key
    for key in (TRUE_TEAM_KEY, "defensive_opportunity", "bait_opportunity", "distill_mask",
                "aux_target"):
        assert key not in FILL and key in REFUSE_KEYS, key


def test_an_unknown_key_refuses_with_its_own_name_and_a_generic_reason():
    bad = unfillable_keys(["observation", "action_mask", "win_target", "some_new_key"])
    assert bad == ["some_new_key"]
    text = refusal_text(bad)
    assert "some_new_key" in text and "REFUSED" in text


def test_a_declared_refusal_names_the_flag_rather_than_only_the_key():
    assert "value-true-team" in refusal_text(["opp_true_team"])
    assert "dense-aux" in refusal_text(["aux_target"])


def test_every_label_key_a_branch_cannot_supply_is_filled_NOT_SCORED():
    keys = ["observation", "action_mask", "win_target", "win_mask", PG_MASK_KEY,
            "item_mask", "hp_type_mask", "belief_spread_mask", "opp_action_kind", "item_label"]
    b = _block(2, keys=keys)
    for masked in ("item_mask", "hp_type_mask", "belief_spread_mask"):
        assert (b["observations"][masked] == 0.0).all(), masked
    assert (b["observations"]["item_label"] == -1).all()
    from agents.training.opp_intent_labels import KIND_UNKNOWN
    assert (b["observations"]["opp_action_kind"] == KIND_UNKNOWN).all()


def test_the_opponent_class_of_an_injected_row_says_POOL_not_the_parents_class():
    """The ecology approximation, LABELLED rather than hidden: the branch really was played
    against a self-like opponent, so the class tag says so and `fork/bot_share` prices it."""
    b = _block(2, keys=["observation", "action_mask", "win_target", "win_mask", PG_MASK_KEY,
                        "opp_class"])
    assert (b["observations"]["opp_class"] == 1).all()


# ── 4. the buffer ────────────────────────────────────────────────────────────────────────────
def _space():
    return spaces.Dict({"observation": spaces.Box(-1, 1, (D,), np.float32),
                        "action_mask": spaces.Box(0, 1, (A,), np.float32),
                        PG_MASK_KEY: spaces.Box(0, 1, (1,), np.float32)})


def _fill(buf, n_steps, n_envs, rng, maskable):
    for t in range(n_steps):
        obs = {"observation": rng.random((n_envs, D)).astype(np.float32),
               "action_mask": np.ones((n_envs, A), np.float32),
               PG_MASK_KEY: np.ones((n_envs, 1), np.float32)}
        kw = {"action_masks": np.ones((n_envs, A), np.float32)} if maskable else {}
        buf.add(obs, np.zeros((n_envs, 1)), np.zeros(n_envs, np.float32),
                np.zeros(n_envs, np.float32), th.zeros(n_envs, 1), th.zeros(n_envs), **kw)
    buf.compute_returns_and_advantage(th.zeros(n_envs, 1), np.ones(n_envs, bool))


def _mk(cls, n_steps=4, n_envs=2, seed=0):
    b = cls(n_steps, _space(), spaces.Discrete(A), gae_lambda=1.0, gamma=1.0, n_envs=n_envs)
    _fill(b, n_steps, n_envs, np.random.default_rng(seed),
          issubclass(cls, MaskableDictRolloutBuffer))
    return b


def test_OFF_is_the_upstream_generator_called():
    """The bit-identity claim that matters for a neutrality read: with nothing injected the fork
    buffer's `get()` yields upstream's own batches, drawn from upstream's own permutation."""
    Fork = fork_buffer_class(DictRolloutBuffer)
    plain, forked = _mk(DictRolloutBuffer), _mk(Fork)
    np.random.seed(11)
    a = [s.observations["observation"].numpy().copy() for s in plain.get(3)]
    np.random.seed(11)
    b = [s.observations["observation"].numpy().copy() for s in forked.get(3)]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.tobytes() == y.tobytes()


@pytest.mark.parametrize("base", [DictRolloutBuffer, MaskableDictRolloutBuffer])
def test_injected_rows_reach_get_and_are_counted(base):
    """Parametrised over the PRODUCTION buffer (`MaskableDictRolloutBuffer`, which carries an
    `action_masks` array) and the plain one, because the mixin claims to compose with whichever
    buffer the algorithm selected."""
    Fork = fork_buffer_class(base)
    buf = _mk(Fork)
    block = concat_blocks([_block(3, keys=["observation", "action_mask", PG_MASK_KEY])])
    # The collected rows carry the env's placeholder, so the only zeros are the fork steps.
    assert buf.add_fork_rows(block) == 3
    assert buf.n_fork_rows == 3
    rows = sum(int(s.observations["observation"].shape[0]) for s in buf.get(5))
    assert rows == 4 * 2 + 3
    masks = np.concatenate([s.observations[PG_MASK_KEY].numpy().reshape(-1) for s in buf.get(5)])
    assert int((masks == 0.0).sum()) == 1, "exactly the one fork step"


def test_reset_drops_the_fork_rows():
    """They are as per-rollout as the buffer: a row left over was measured under weights this
    rollout has already left behind."""
    Fork = fork_buffer_class(DictRolloutBuffer)
    buf = _mk(Fork)
    buf.add_fork_rows(concat_blocks([_block(2, keys=["observation", "action_mask", PG_MASK_KEY])]))
    buf.reset()
    assert buf.n_fork_rows == 0


def test_a_key_set_or_width_mismatch_REFUSES_at_injection_not_inside_the_generator():
    Fork = fork_buffer_class(DictRolloutBuffer)
    buf = _mk(Fork)
    with pytest.raises(ValueError, match="obs keys"):
        buf.add_fork_rows(concat_blocks([_block(2, keys=["observation", "action_mask"])]))
    bad = concat_blocks([_block(2, keys=["observation", "action_mask", PG_MASK_KEY])])
    bad["observations"]["observation"] = np.zeros((2, D + 1), np.float32)
    with pytest.raises(ValueError, match="shape"):
        buf.add_fork_rows(bad)


def test_add_fork_rows_of_none_clears_and_returns_zero():
    Fork = fork_buffer_class(DictRolloutBuffer)
    buf = _mk(Fork)
    assert buf.add_fork_rows(None) == 0 and buf.n_fork_rows == 0


def test_the_fork_buffer_class_is_memoised_and_keeps_the_bases_behaviour():
    a = fork_buffer_class(DictRolloutBuffer)
    assert a is fork_buffer_class(DictRolloutBuffer)
    assert issubclass(a, (ForkRolloutBuffer, DictRolloutBuffer))
