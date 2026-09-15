"""Gates for the FORK ARM's selector and meters (`gen3_fork_v1`).

The arithmetic half — what gets forked and what the `fork/*` family says about it. The buffer
surgery is `fork_buffer_test`, the CRN pairing is `fork_crn_test` / `fork_crn_sim_test`, and the
flag surface is `fork_flags_test`.

**What these tests are FOR.** The selector is the arm's distribution: the registered endpoint is a
held-out pairwise accuracy read against an offline baseline that was measured on one particular
contested population, so a silent change to WHICH decisions are forked makes the two numbers
incomparable. Every rule the paired-refit dataset declared is pinned here against its own source.
"""
from __future__ import annotations

import numpy as np
import pytest

from agents.action.constants import MOVE_START
from agents.training.cf_producer_sampler import MIN_LABELABLE_TURN
from agents.training.fork_arm import (
    BRANCH_NAMES, MAX_FORKABLE_TURN, MIN_LEGAL_ACTIONS, branch_actions, candidate_pool,
    contested_select, contested_threshold, eligible_mask, fork_metrics, forks_per_battle,
    is_move_round_mask, n_forks_for, pairwise_accuracy, pairwise_rows, pool_size_for,
    random_wins_rate, sim_steps_share, slice_ids, tie_rate, top2_gaps,
)

N_ACT = 11


def _mask(*idx):
    m = np.zeros(N_ACT, dtype=np.int8)
    for i in idx:
        m[i] = 1
    return m


def _move_mask(n=3):
    return _mask(*range(MOVE_START, MOVE_START + n))


# ── eligibility ──────────────────────────────────────────────────────────────────────────────
def test_eligible_needs_all_four_conditions():
    wm = np.ones((1, 5), dtype=np.float32)
    turns = np.full((1, 5), 10, dtype=np.int64)
    am = np.stack([[_move_mask(3)] * 5], axis=0).astype(np.float32)
    assert eligible_mask(wm, turns, am).all()

    # win_mask 0 — the episode did NOT terminate in this buffer, so no __RECON__ record exists.
    wm2 = wm.copy(); wm2[0, 0] = 0.0
    assert not eligible_mask(wm2, turns, am)[0, 0]
    # no handle
    t2 = turns.copy(); t2[0, 1] = -1
    assert not eligible_mask(wm, t2, am)[0, 1]
    # below cf_producer's floor / above forks.py's ceiling
    t3 = turns.copy(); t3[0, 2] = MIN_LABELABLE_TURN - 1; t3[0, 3] = MAX_FORKABLE_TURN + 1
    el = eligible_mask(wm, t3, am)
    assert not el[0, 2] and not el[0, 3]
    # too few legal actions: a random branch needs a third option to exist at all
    am2 = am.copy(); am2[0, 4] = _move_mask(MIN_LEGAL_ACTIONS - 1)
    assert not eligible_mask(wm, turns, am2)[0, 4]


def test_a_mid_turn_forced_switch_is_never_forkable():
    """A counterfactual replay cuts at a TURN boundary, so a forced-switch round has no divergence
    point — `cf_producer_sampler.is_move_round`'s rule, inherited."""
    switch_only = _mask(0, 1, 2)                       # three legal SWITCHES, no move
    assert not is_move_round_mask(switch_only)
    wm = np.ones((1, 1), dtype=np.float32)
    turns = np.full((1, 1), 10, dtype=np.int64)
    am = switch_only.reshape(1, 1, N_ACT).astype(np.float32)
    assert not eligible_mask(wm, turns, am).any()


def test_slice_ids_separate_two_games_of_one_env():
    es = np.array([[1.0], [0.0], [0.0], [1.0], [0.0]])
    assert slice_ids(es).reshape(-1).tolist() == [1, 1, 1, 2, 2]


# ── the pool ─────────────────────────────────────────────────────────────────────────────────
def _grid(n_steps, n_envs):
    wm = np.ones((n_steps, n_envs), dtype=np.float32)
    turns = np.full((n_steps, n_envs), 10, dtype=np.int64)
    am = np.tile(_move_mask(4), (n_steps, n_envs, 1)).astype(np.float32)
    return wm, turns, am


def test_pool_respects_max_per_battle():
    wm, turns, am = _grid(8, 2)
    el = eligible_mask(wm, turns, am)
    es = np.zeros((8, 2)); es[0] = 1.0                  # one episode per env
    for per in (1, 3):
        picks = candidate_pool(el, es, 64, per, np.random.default_rng(0))
        by_env: dict = {}
        for _t, e in picks:
            by_env[e] = by_env.get(e, 0) + 1
        assert max(by_env.values()) <= per


def test_pool_is_deterministic_in_its_rng_and_bounded_by_n_pool():
    wm, turns, am = _grid(8, 4)
    el = eligible_mask(wm, turns, am)
    es = np.zeros((8, 4)); es[0] = 1.0
    a = candidate_pool(el, es, 3, 1, np.random.default_rng(7))
    b = candidate_pool(el, es, 3, 1, np.random.default_rng(7))
    assert a == b and len(a) == 3
    # ...and every pick is an ELIGIBLE row of a distinct env (one per slice, one episode per env).
    assert all(bool(el[t, e]) for t, e in a)
    assert len({e for _t, e in a}) == len(a)


def test_empty_eligibility_pools_nothing():
    assert candidate_pool(np.zeros((4, 2), dtype=bool), np.zeros((4, 2)), 10, 1,
                          np.random.default_rng(0)) == []


def test_pool_size_covers_the_quantile_and_has_a_floor():
    assert pool_size_for(100, 0.40) == 250              # 100 / 0.40
    assert pool_size_for(1, 0.40) >= 32                 # the floor, so a threshold has support


# ── the contested test ───────────────────────────────────────────────────────────────────────
def test_top2_gaps_ignores_illegal_actions():
    logits = np.array([[5.0, 0.1, 0.2, 0.15]])
    masks = np.array([[0, 1, 1, 1]])                    # the 5.0 is ILLEGAL
    gap, a1, a2 = top2_gaps(logits, masks)
    assert a1[0] == 2 and a2[0] == 3
    assert gap[0] == pytest.approx(0.05)


def test_a_row_with_one_legal_action_is_never_contested_and_never_moves_the_quantile():
    logits = np.array([[1.0, 0.0], [3.0, 2.0]])
    masks = np.array([[1, 0], [1, 1]])
    gap, _a1, _a2 = top2_gaps(logits, masks)
    assert np.isinf(gap[0])
    # the quantile is taken over the FINITE gaps alone, so the inf cannot drag it
    assert contested_threshold(gap, 1.0) == pytest.approx(1.0)


def test_threshold_is_the_shards_own_quantile():
    gaps = np.arange(10, dtype=float)                   # 0..9
    assert contested_threshold(gaps, 0.40) == pytest.approx(np.quantile(gaps, 0.40))
    assert contested_threshold(np.array([np.inf]), 0.4) == float("-inf")


def test_select_takes_the_tightest_gaps_first_and_caps():
    pool = [(0, 0), (1, 0), (2, 0), (3, 0)]
    gaps = np.array([3.0, 0.5, 1.0, 0.2])
    thr = contested_threshold(gaps, 1.0)
    idx = contested_select(pool, gaps, None, thr, 0.0, 2)
    assert idx == [3, 1], "tightest gap first"


def test_the_absv_arm_is_off_by_default_and_admits_when_on():
    """🚨 The paired-refit selector NEVER reads V. Off, a wide-gap row stays out however close V is
    to 0.5; on, it is admitted — which is what forfeits the comparison to the registered
    baseline."""
    pool = [(0, 0)]
    gaps = np.array([9.0])
    absv = np.array([0.0])                              # V == 0.5 exactly
    assert contested_select(pool, gaps, absv, 1.0, 0.0, 5) == []
    assert contested_select(pool, gaps, absv, 1.0, 0.05, 5) == [0]


# ── the branches ─────────────────────────────────────────────────────────────────────────────
def test_three_branches_are_top2_plus_a_random_LEGAL_alternative():
    legal = [0, 1, 6, 7, 8]
    acts = branch_actions(legal, 6, 7, 3, salt="s", seed=1)
    assert set(acts) == set(BRANCH_NAMES)
    assert acts["top1"] == 6 and acts["top2"] == 7
    assert acts["rand"] in (0, 1, 8), "the random branch must avoid what the policy already covers"


def test_two_branches_is_the_control_and_draws_nothing():
    acts = branch_actions([0, 1, 6, 7], 6, 7, 2, salt="s", seed=1)
    assert set(acts) == {"top1", "top2"}


def test_no_third_legal_action_yields_no_random_branch():
    assert set(branch_actions([6, 7], 6, 7, 3, salt="s", seed=1)) == {"top1", "top2"}


def test_the_random_draw_is_decision_keyed_and_reproducible():
    legal = list(range(11))
    a = branch_actions(legal, 6, 7, 3, salt="battle:12", seed=1001)
    b = branch_actions(legal, 6, 7, 3, salt="battle:12", seed=1001)
    c = branch_actions(legal, 6, 7, 3, salt="battle:13", seed=1001)
    assert a == b
    # Not an assertion that they DIFFER (a 1-in-9 collision is legal); the pin is reproducibility.
    assert c["rand"] in legal


# ── the cost / count arithmetic ──────────────────────────────────────────────────────────────
def test_n_forks_is_a_fraction_of_the_whole_buffer_and_is_capped():
    assert n_forks_for(0.0, 2048, 48) == 0
    assert n_forks_for(0.02, 2048, 48, cap=10**9) == int(0.02 * 2048 * 48)
    assert n_forks_for(0.5, 2048, 48, cap=128) == 128


def test_sim_steps_share_is_measured_against_the_trainees_own_decisions():
    forks = [{"branches": {"top1": {"decisions": 40.0}, "top2": {"decisions": 60.0}}}]
    assert sim_steps_share(forks, 10, 10) == pytest.approx(1.0)
    assert np.isnan(sim_steps_share(forks, 0, 0))


def test_rate_is_forks_per_battle():
    es = np.zeros((10, 2)); es[0] = 1.0; es[5, 0] = 1.0     # 3 episodes started
    assert forks_per_battle(6, es) == pytest.approx(2.0)


# ── the meters ───────────────────────────────────────────────────────────────────────────────
def _fork(*branches):
    return {"branches": {n: b for n, b in zip(BRANCH_NAMES, branches)}}


def _b(v, y, capped=False, decisions=1.0):
    return {"succ_value": v, "outcome": y, "capped": capped, "decisions": decisions}


def test_pairwise_accuracy_is_the_sign_agreement_on_non_tied_pairs():
    right = _fork(_b(0.9, 1.0), _b(0.1, 0.0))
    wrong = _fork(_b(0.1, 1.0), _b(0.9, 0.0))
    acc, n = pairwise_accuracy([right, wrong])
    assert n == 2 and acc == pytest.approx(0.5)
    acc, n = pairwise_accuracy([right, right])
    assert n == 2 and acc == pytest.approx(1.0)


def test_a_tied_pair_is_dropped_not_scored():
    """79.7 % of measured branch pairs share an outcome — the structural tax. A tie carries no
    ordering, so scoring it 0.5 would drag every reading toward 0.5 by the tie rate."""
    tied = _fork(_b(0.9, 1.0), _b(0.1, 1.0))
    acc, n = pairwise_accuracy([tied])
    assert n == 0 and np.isnan(acc)


def test_a_capped_branch_never_enters_a_pair():
    """A 250-turn cap is decided by SEAT, not by the position, so it is excluded — never 0.5."""
    f = _fork(_b(0.9, 1.0, capped=True), _b(0.1, 0.0))
    assert pairwise_rows([f])[0].size == 0


def test_a_branch_with_no_successor_never_enters_a_pair():
    f = _fork(_b(None, 1.0), _b(0.1, 0.0))
    assert pairwise_rows([f])[0].size == 0


def test_an_exactly_tied_value_scores_half_rather_than_disappearing():
    """A head that collapsed to a constant must read 0.5, not 'no data' — the two are different
    facts about the run and only one of them is a reason to kill an arm."""
    f = _fork(_b(0.5, 1.0), _b(0.5, 0.0))
    acc, n = pairwise_accuracy([f])
    assert n == 1 and acc == pytest.approx(0.5)


def test_tie_rate_counts_forks_whose_branches_all_agree():
    tied = _fork(_b(0.9, 1.0), _b(0.1, 1.0))
    split = _fork(_b(0.9, 1.0), _b(0.1, 0.0))
    assert tie_rate([tied, split, tied]) == pytest.approx(2 / 3)
    assert np.isnan(tie_rate([]))


def test_random_wins_is_the_blind_spot_rate():
    """The offline dataset read 4.5 % [3.99, 5.16] — the share of forks where a uniformly random
    legal alternative beat BOTH policy candidates. It is the meter that says whether the third
    branch is earning its simulation."""
    blind = _fork(_b(0.9, 0.0), _b(0.8, 0.0), _b(0.1, 1.0))    # rand wins, both candidates lose
    ordinary = _fork(_b(0.9, 1.0), _b(0.8, 1.0), _b(0.1, 0.0))
    assert random_wins_rate([blind, ordinary]) == pytest.approx(0.5)
    assert np.isnan(random_wins_rate([_fork(_b(0.9, 1.0), _b(0.1, 0.0))]))   # no rand branch


def test_fork_metrics_publishes_the_six_registered_meters():
    m = fork_metrics(forks=[_fork(_b(0.9, 1.0, decisions=10.0), _b(0.1, 0.0, decisions=10.0))],
                     requested=1, eligible=100, pool=10, threshold=0.5, injected_rows=40,
                     buffer_rows=160, masked_rows=2, fraction=0.02, branches=3,
                     crn="dice_and_draws", seconds=1.0, records_missing=0, n_steps=10, n_envs=10,
                     opp_class=[0])
    for key in ("tie_rate", "random_wins", "pairwise_acc", "sim_steps_share", "branch_share"):
        assert key in m
    assert m["branch_share"] == pytest.approx(40 / 200)
    assert m["sim_steps_share"] == pytest.approx(0.2)
    assert m["crn_draws"] == 1.0
    assert m["bot_share"] == 1.0


def test_fork_metrics_of_an_empty_pass_reports_zero_forks_not_a_crash():
    m = fork_metrics(forks=[], requested=3, eligible=0, pool=0, threshold=float("-inf"),
                     injected_rows=0, buffer_rows=100, masked_rows=0, fraction=0.02, branches=3,
                     crn="dice", seconds=0.0, records_missing=3, n_steps=10, n_envs=10)
    assert m["forks"] == 0.0 and m["failed"] == 3.0 and m["records_missing"] == 3.0
    assert m["crn_draws"] == 0.0
    assert np.isnan(m["pairwise_acc"])
