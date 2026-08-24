"""The iterative-deepening TREE: the backup arithmetic, the beam, and what a ply costs.

Every tree here is built by hand. That is the point of splitting :mod:`deepen` out of
:mod:`search` — the backup is the one part of a deepening search whose correctness is a claim
about arithmetic rather than about a sim, and an arithmetic claim should be checkable without
spawning a driver, materializing an observation, or loading a checkpoint.
"""

from __future__ import annotations

import pytest

from main.search_dividend.deepen import (MIN_BEAM, TreeNode, backup, beam_actions, leaves_under,
                                         per_action_values, plan_beam)


def _leaf(value, path=(), **kw):
    return TreeNode(node_id=kw.pop("node_id", "n"), ended=kw.pop("ended", False), value=value,
                    path=tuple(path), **kw)


def _root(children):
    """``children`` = ``{our_action: [(weight, node), ...]}``."""
    r = TreeNode(node_id="root", ended=False, our_tokens={0: "move a", 1: "move b"})
    for a, kids in children.items():
        for w, n in kids:
            r.add_child(a, w, n)
    return r


# -- the backup ---------------------------------------------------------------


def test_the_opponent_axis_is_an_ALPHA_WEIGHTED_MEAN_not_a_min_or_a_mean():
    """We marginalize the opponent, we do not assume its best reply and we do not treat its
    options as equally likely. The weights ARE the belief — a plain mean would throw away the α
    head, and a min would turn the probe into a different (paranoid) algorithm."""
    root = _root({0: [(0.9, _leaf(1.0)), (0.1, _leaf(0.0))]})
    assert per_action_values(root)[0] == pytest.approx(0.9)


def test_our_axis_is_a_MAX_and_the_two_alternate_down_the_tree():
    """Depth-2 in one assertion: the grandchild layer is maxed over OUR options and averaged over
    THEIRS, and that value replaces the child's own leaf value in the layer above."""
    # Child of action 0: leaf value 0.10, but deepening finds a line worth 0.80.
    child = _leaf(0.10, path=(0,))
    child.add_child(7, 0.5, _leaf(0.80, path=(0, 7)))     # our best deeper action ...
    child.add_child(7, 0.5, _leaf(0.80, path=(0, 7)))     # ... under both of their replies
    child.add_child(8, 0.5, _leaf(0.20, path=(0, 8)))
    child.add_child(8, 0.5, _leaf(0.20, path=(0, 8)))
    assert backup(child) == pytest.approx(0.80), "MAX over ours, mean over theirs"

    root = _root({0: [(1.0, child)], 1: [(1.0, _leaf(0.50, path=(1,)))]})
    vals = per_action_values(root)
    assert vals[0] == pytest.approx(0.80) and vals[1] == pytest.approx(0.50)
    assert max(vals, key=lambda a: vals[a]) == 0, "the deepened line wins on its BACKED-UP value"


def test_an_unequal_opponent_weighting_survives_a_ply():
    """The deeper ply must use the SAME α rules, not a uniform stand-in — otherwise depth 2 would
    quietly answer a different question than depth 1."""
    child = _leaf(0.0, path=(0,))
    child.add_child(7, 0.75, _leaf(1.0, path=(0, 7)))
    child.add_child(7, 0.25, _leaf(0.0, path=(0, 7)))
    assert backup(child) == pytest.approx(0.75)


def test_a_leaf_keeps_its_own_value_when_it_was_never_deepened():
    """Iterative deepening stops wherever the clock stops it, and a leaf that was scored is worth
    its score. A leaf collapsing to None here would silently drop whole actions from the mean."""
    assert backup(_leaf(0.42)) == pytest.approx(0.42)


def test_a_node_whose_children_all_failed_to_score_falls_back_to_its_own_value():
    """A ply that expanded but produced nothing scorable must not destroy the value the node
    already had — the depth-1 estimate is still the best available one."""
    n = _leaf(0.30, path=(0,))
    n.add_child(7, 1.0, _leaf(None, path=(0, 7)))
    assert backup(n) == pytest.approx(0.30)


def test_an_action_with_no_scorable_child_scores_zero_rather_than_vanishing():
    root = _root({0: [(1.0, _leaf(0.9))], 1: [(1.0, _leaf(None))]})
    vals = per_action_values(root)
    assert set(vals) == {0, 1} and vals[1] == 0.0


def test_depth_one_is_the_registered_expression_verbatim():
    """The whole safety property of the amendment: deepening must not change the estimator it
    deepens. With no grandchildren the backup IS `argmax_a Sum_c alpha(c) V(s')`."""
    root = _root({0: [(0.6, _leaf(1.0)), (0.4, _leaf(0.0))],
                  1: [(0.6, _leaf(0.5)), (0.4, _leaf(0.5))]})
    vals = per_action_values(root)
    assert vals[0] == pytest.approx(0.6) and vals[1] == pytest.approx(0.5)


def test_the_dice_average_and_the_opponent_average_are_the_SAME_weighted_mean():
    """Two CRN draws of one candidate enter as two children sharing that candidate's weight, so
    the dice average falls out of the same expression rather than needing its own."""
    root = _root({0: [(0.5, _leaf(1.0)), (0.5, _leaf(0.0)),      # candidate A, two dice
                      (0.5, _leaf(0.4)), (0.5, _leaf(0.4))]})    # candidate B, two dice
    assert per_action_values(root)[0] == pytest.approx((1.0 + 0.0 + 0.4 + 0.4) / 4)


# -- the beam -----------------------------------------------------------------


def test_the_beam_is_the_top_m_and_ties_break_reproducibly():
    vals = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.9}
    assert beam_actions(vals, 2) == [1, 3], "equal values order by index, so a beam is stable"
    assert beam_actions(vals, 1) == [1]
    assert beam_actions(vals, 99) == [1, 3, 2, 0]


def test_leaves_are_found_only_under_the_beam_and_only_at_the_asked_depth():
    a0 = _leaf(0.5, path=(0,), node_id="c0")
    a0.our_tokens = {7: "move x"}
    a1 = _leaf(0.4, path=(1,), node_id="c1")
    a1.our_tokens = {7: "move y"}
    root = _root({0: [(1.0, a0)], 1: [(1.0, a1)]})
    assert leaves_under(root, [0], depth=1) == [a0]
    assert leaves_under(root, [0, 1], depth=1) == [a0, a1]
    assert leaves_under(root, [0], depth=2) == [], "nothing has been deepened yet"


def test_a_terminal_or_tokenless_leaf_is_NOT_expandable():
    """Three independent reasons a node is a leaf by construction, kept together so none of them
    can be forgotten at a call site: no live node, the battle ended, or no legal surface of ours."""
    assert not _leaf(1.0, node_id=None).expandable()
    assert not TreeNode(node_id="n", ended=True, our_tokens={1: "x"}).expandable()
    assert not TreeNode(node_id="n", ended=False, our_tokens={}).expandable()
    assert TreeNode(node_id="n", ended=False, our_tokens={1: "x"}).expandable()


# -- what a ply costs ---------------------------------------------------------


def _beamable_root():
    kids = {}
    for a in range(4):
        leaf = _leaf(1.0 - 0.1 * a, path=(a,), node_id=f"c{a}")
        leaf.our_tokens = {0: "x", 1: "y"}           # 2 of our actions at the deeper ply
        kids[a] = [(1.0, leaf)]
    return _root(kids)


def test_the_beam_is_the_WIDEST_that_fits_the_remaining_budget():
    """A ply that could have compared four actions and only compared two has thrown away the
    comparison it was spent to make."""
    root = _beamable_root()
    vals = per_action_values(root)
    beam, leaves, n_arms = plan_beam(root, vals, depth=1, m_opp=2, n_opp_at=lambda _n: 2,
                                     arm_cost_s=0.01, ply_overhead_s=0.0, remaining_s=10.0)
    assert beam == [0, 1, 2, 3] and len(leaves) == 4
    assert n_arms == 4 * 2 * 2, "our actions x their candidates, per leaf"


def test_a_tight_budget_NARROWS_the_beam_rather_than_half_expanding_a_ply():
    """A ply is whole or not at all. Expanding part of one leaves a MAX node whose children sit at
    two different depths — the same inconsistency the root beam exists to prevent, one level down
    where it is harder to see."""
    root = _beamable_root()
    vals = per_action_values(root)
    beam, _leaves, n_arms = plan_beam(root, vals, depth=1, m_opp=2, n_opp_at=lambda _n: 2,
                                      arm_cost_s=0.01, ply_overhead_s=0.0, remaining_s=0.09)
    assert beam == [0, 1] and n_arms == 8, "8 arms x 0.01 s fits 0.09 s; 12 would not"


def test_a_budget_too_small_for_even_two_actions_declines_the_ply():
    """Below `MIN_BEAM` there is nothing left to compare, so a ply there can only spend budget and
    inflate the reported depth. Declining keeps `depth_realized` honest."""
    root = _beamable_root()
    vals = per_action_values(root)
    beam, leaves, n_arms = plan_beam(root, vals, depth=1, m_opp=2, n_opp_at=lambda _n: 2,
                                     arm_cost_s=0.01, ply_overhead_s=0.0, remaining_s=0.05)
    assert beam == [] and leaves == [] and n_arms == 0
    assert MIN_BEAM == 2


def test_the_ply_OVERHEAD_counts_against_the_budget_too():
    """The prefix replay is paid once per ply and it is the measured majority of an arm's cost —
    a planner blind to it would promise a depth the clock cannot deliver."""
    root = _beamable_root()
    vals = per_action_values(root)
    fits = dict(root=root, values=vals, depth=1, m_opp=2, n_opp_at=lambda _n: 2,
                arm_cost_s=0.001)
    wide, _l, _n = plan_beam(**fits, ply_overhead_s=0.0, remaining_s=0.02)
    narrow, _l2, _n2 = plan_beam(**fits, ply_overhead_s=0.018, remaining_s=0.02)
    assert wide == [0, 1, 2, 3]
    assert len(narrow) < len(wide) or narrow == []


def test_the_opponent_cap_bounds_the_ply_even_when_the_request_is_wide():
    """`m_opp` is the registered first width axis and it does not stop applying at depth 2."""
    root = _beamable_root()
    vals = per_action_values(root)
    _b, _l, n_arms = plan_beam(root, vals, depth=1, m_opp=2, n_opp_at=lambda _n: 9,
                               arm_cost_s=0.0001, ply_overhead_s=0.0, remaining_s=10.0)
    assert n_arms == 4 * 2 * 2, "9 legal opponent choices, capped to the planned 2"
