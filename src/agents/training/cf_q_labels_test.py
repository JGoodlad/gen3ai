"""Unit tests for the PER-ACTION sweep's arithmetic (`cf_q_labels.py`, `gen3_cf_q_labels_v1`).

Pure — no model, no bridge, no record. Everything here is a property of the decisions that make a
sweep of rollouts COMPARABLE to itself: the common-random-number pairing, the declared selection
rule, and the wire shape. The producer-side wiring (that the pairing survives a real `_rollout`,
that the recorded arm's identity holds, that the budget knobs bite) lives in `cf_producer_test.py`,
and the end-to-end mating with the consumer is in `cf_producer_integration_test.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.training import cf_q_labels as Q


# ---------------------------------------------------------------------------
# The dice — the pairing is the point
# ---------------------------------------------------------------------------

class TestPairedDice:
    def test_the_salt_carries_no_action_term(self):
        """The whole pairing property in one assertion: the salt is a function of the DECISION.

        If an action index ever enters this string, sibling arms draw independent dice and the
        sweep's product — a RANKING — becomes noise at the producer's R.
        """
        salt = Q.q_arm_salt(tag="battle-x", decision_index=4, producer_seed=7)
        assert salt == "battle-x:4:cfp7"
        # There is no parameter to vary; the type signature is the guarantee.
        assert "action" not in Q.q_arm_salt.__code__.co_varnames

    def test_the_salt_is_byte_identical_to_the_per_state_paths_historical_one(self):
        """`cf_producer._rollout` derived its dice from this exact string before the sweep existed.

        That identity is what keeps `--no-q-labels` byte-identical AND what makes the recorded
        action's already-paid-for arm reusable as a q-label rather than merely similar to one.
        """
        tag, idx, seed = "0000000000000000001_1_tag_reconstruction.json", 4, 20260822
        assert Q.q_arm_salt(tag=tag, decision_index=idx, producer_seed=seed) == \
            f"{tag}:{idx}:cfp{seed}"

    def test_every_sibling_action_draws_the_same_list(self):
        kw = dict(tag="t", decision_index=3, producer_seed=1, n=6)
        assert Q.q_arm_seeds(**kw) == Q.q_arm_seeds(**kw)

    def test_a_smaller_R_is_a_PREFIX_of_a_larger_one(self):
        """So `--q-rollouts 4` against `--rollouts 8` is a SUB-SAMPLE of the same dice, not a
        different experiment — which is what lets the two be compared at all."""
        big = Q.q_arm_seeds(tag="t", decision_index=3, producer_seed=1, n=8)
        small = Q.q_arm_seeds(tag="t", decision_index=3, producer_seed=1, n=4)
        assert big[:4] == small

    def test_a_different_decision_draws_different_dice(self):
        a = Q.q_arm_seeds(tag="t", decision_index=3, producer_seed=1, n=4)
        b = Q.q_arm_seeds(tag="t", decision_index=4, producer_seed=1, n=4)
        assert a != b, "pairing is WITHIN a decision; across decisions the dice must vary"

    def test_the_producer_seed_moves_the_dice(self):
        a = Q.q_arm_seeds(tag="t", decision_index=3, producer_seed=1, n=4)
        b = Q.q_arm_seeds(tag="t", decision_index=3, producer_seed=2, n=4)
        assert a != b, "--seed must reach the sweep, or a re-run cannot be varied"


class TestAssertPairedDice:
    def test_identical_lists_pass(self):
        seeds = ("a", "b")
        Q.assert_paired_dice({6: seeds, 7: seeds, 0: ("a", "b")})

    def test_an_empty_or_single_action_sweep_is_trivially_paired(self):
        Q.assert_paired_dice({})
        Q.assert_paired_dice({6: ("a",)})

    def test_DIFFERENT_dice_across_siblings_RAISE(self):
        """THE regression this function exists for. A sweep whose arms drew independent dice
        reports a ranking made of noise, and nothing about the labels looks wrong."""
        with pytest.raises(RuntimeError, match="DICE ARE NOT PAIRED"):
            Q.assert_paired_dice({6: ("a", "b"), 7: ("c", "d")})

    def test_the_message_names_the_offending_actions(self):
        with pytest.raises(RuntimeError) as e:
            Q.assert_paired_dice({6: ("a",), 7: ("z",)})
        assert "6:" in str(e.value) and "7:" in str(e.value)

    def test_a_SHORTER_list_is_also_unpaired(self):
        """Same dice for the first k arms is not the same experiment — the means differ in n."""
        with pytest.raises(RuntimeError):
            Q.assert_paired_dice({6: ("a", "b"), 7: ("a",)})


# ---------------------------------------------------------------------------
# The selection rule (cf_q_sweep_v1)
# ---------------------------------------------------------------------------

def _mask(*idx):
    m = np.zeros(11, dtype=np.int8)
    m[list(idx)] = 1
    return m


class TestSelectQActions:
    def test_all_legal_actions_by_default(self):
        got = Q.select_q_actions(_mask(0, 1, 6, 7, 8), 7)
        assert sorted(got) == [0, 1, 6, 7, 8]

    def test_the_recorded_action_leads(self):
        """It is the FREE arm and the sweep's anchor, so a truncation must never drop it."""
        for rec in (0, 6, 8):
            assert Q.select_q_actions(_mask(0, 1, 6, 7, 8), rec)[0] == rec

    def test_illegal_indices_never_enter(self):
        got = Q.select_q_actions(_mask(1, 7), 7)
        assert sorted(got) == [1, 7]

    def test_the_cap_takes_a_prefix_and_keeps_the_recorded_action(self):
        got = Q.select_q_actions(_mask(0, 1, 2, 6, 7, 8, 9), 8, max_actions=3)
        assert len(got) == 3 and got[0] == 8

    def test_a_cap_at_or_above_the_legal_count_changes_nothing(self):
        full = Q.select_q_actions(_mask(1, 6, 7), 6)
        assert Q.select_q_actions(_mask(1, 6, 7), 6, max_actions=9) == full

    def test_it_is_reproducible_across_calls_and_processes(self):
        kw = dict(max_actions=3, tag="t", decision_index=5, producer_seed=11)
        a = Q.select_q_actions(_mask(0, 1, 2, 6, 7, 8, 9), 7, **kw)
        b = Q.select_q_actions(_mask(0, 1, 2, 6, 7, 8, 9), 7, **kw)
        assert a == b, "the order is keyed to the decision, so it must not move between calls"

    def test_different_decisions_get_different_orders(self):
        m, rec = _mask(0, 1, 2, 3, 6, 7, 8, 9), 7
        orders = {tuple(Q.select_q_actions(m, rec, tag="t", decision_index=i, producer_seed=1))
                  for i in range(12)}
        assert len(orders) > 1, "a constant order would make every capped sweep the same subset"

    def test_a_CAPPED_sweep_does_not_systematically_prefer_SWITCHES(self):
        """The anti-bias property, and the reason the order is a shuffle rather than an index sort.

        The action space is ``[switch x6, move x4, struggle]``, so a cap that took the lowest
        indices would teach the Q head about switching and almost nothing about attacking — on
        exactly the decisions (move rounds) where the attacking options are the question. Measured
        over many decisions, the moves must appear about as often as their share of the legal set.
        """
        m, rec = _mask(0, 1, 2, 3, 4, 6, 7, 8), 0      # 5 switches, 3 moves; recorded is a switch
        moves = 0
        trials = 400
        for i in range(trials):
            # 3 arms: the recorded switch plus two drawn from the remaining 4 switches + 3 moves.
            got = Q.select_q_actions(m, rec, max_actions=3, tag="t", decision_index=i,
                                     producer_seed=3)
            moves += sum(1 for a in got if a >= 6)
        share = moves / float(2 * trials)
        # The non-recorded pool is 4 switches + 3 moves, so an unbiased draw puts moves at 3/7.
        assert 0.30 < share < 0.56, f"moves took {share:.2f} of the capped arms — index bias?"

    def test_an_index_ordered_rule_would_FAIL_that_test(self):
        """Pins the counterfactual: the naive rule this shuffle replaced really is biased."""
        m = _mask(0, 1, 2, 3, 4, 6, 7, 8)
        naive = ([0] + sorted(a for a in np.flatnonzero(m) if a != 0))[:3]
        assert all(a < 6 for a in naive), "sorted-by-index is all switches — the bias being avoided"


class TestRecordedArmReuse:
    def test_reusable_exactly_when_the_counts_match(self):
        assert Q.recorded_arm_is_reusable(q_rollouts=8, rollouts=8)
        assert not Q.recorded_arm_is_reusable(q_rollouts=4, rollouts=8)
        assert not Q.recorded_arm_is_reusable(q_rollouts=8, rollouts=4)


# ---------------------------------------------------------------------------
# The wire shape
# ---------------------------------------------------------------------------

class TestWireShape:
    def test_an_entry_is_an_OBJECT_naming_its_own_action(self):
        """Never parallel arrays: three same-length lists can be written in the wrong order and
        read as valid, and this tree treats an order mismatch as drop-everything."""
        e = Q.q_label_entry(7, wins=3.0, n=4)
        assert e == {"action": 7, "label": 0.75, "n_rollouts": 4}
        assert set(e) == {"action", "label", "n_rollouts"}

    def test_a_fractional_win_total_survives(self):
        """A draw — including the 250-turn cap, a forfeit-ordering artifact — scores 0.5."""
        assert Q.q_label_entry(6, wins=1.5, n=3)["label"] == pytest.approx(0.5)

    def test_a_zero_evidence_arm_is_DROPPED_not_shipped_at_n_zero(self):
        """An `n_rollouts: 0` entry would mask ON a cell whose target is the 0.0 fallback — a
        confident LOSS for an action nobody measured. Omission leaves it unsupervised, which is
        what it is."""
        block = Q.q_labels_block([(6, 2.0, 4), (7, 0.0, 0)])
        assert [e["action"] for e in block] == [6]

    def test_the_block_preserves_the_selection_order(self):
        block = Q.q_labels_block([(7, 1.0, 2), (0, 1.0, 2), (6, 1.0, 2)])
        assert [e["action"] for e in block] == [7, 0, 6]

    def test_labels_are_inside_the_unit_interval(self):
        for wins, n in ((0.0, 4), (4.0, 4), (2.0, 4), (1.5, 3)):
            assert 0.0 <= Q.q_label_entry(6, wins=wins, n=n)["label"] <= 1.0


class TestProvenance:
    def test_it_reports_the_measured_arm_count(self):
        """`arms` is the ~n_legal MULTIPLIER for this row — the number an operator sizes with, and
        the one a reader cannot re-derive from `q_labels` once zero-evidence arms are dropped."""
        p = Q.q_provenance(actions=[7, 0, 6], rollouts=4, capped=1, reused_recorded=True,
                           max_actions=0, wall_seconds=12.25)
        assert p["arms"] == 3 and p["rollouts_per_arm"] == 4
        assert p["recorded_arm_reused"] is True and p["n_capped"] == 1
        assert p["wall_seconds"] == 12.25
        assert p["version"] == Q.Q_SWEEP_VERSION == "cf_q_sweep_v1"

    def test_the_wall_is_omitted_when_not_measured(self):
        assert "wall_seconds" not in Q.q_provenance(
            actions=[], rollouts=0, capped=0, reused_recorded=False, max_actions=0)
