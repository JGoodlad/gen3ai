"""The engine's arm contract and its FALLBACK accounting.

A search that fell back silently would report a null dividend indistinguishable from a real one,
so every decline has a named reason and every reason is exercised here. The sim is faked: these
are tests of the bookkeeping, and the bookkeeping is what a live run can only ever show you
indirectly.
"""

from __future__ import annotations

import pytest

from main.search_dividend.budget import FALLBACK_REASONS, WidthCaps
from main.search_dividend.search import (ARMS, SearchConfig, SearchEngine,
                                         _selectable_across_worlds, _terminal_label, batch_scores,
                                         branchable)
from utils.bridge.reconstruction import ReconstructionRecord

RECORD = ReconstructionRecord(
    format_id="gen3ou", prng_seed="sodium,ab",
    input_log=('>start {"formatid":"gen3ou","seed":"sodium,ab"}',
               '>player p1 {"name": "Me", "team": "T1"}',
               '>player p2 {"name": "You", "team": "T2"}'),
    commands=(("p1", "move surf"), ("p2", "move earthquake")), battle_tag="battle-x",
    trainee_username="Me")

OBSERVED = ["|switch|p2a: A|Skarmory, M|100/100", "|turn|1"]


class _Root:
    def __init__(self, prefix, requests=None):
        self.node_id = "n0"
        self.prefix_p1_chunks = ["\n".join(prefix)]
        self.prefix_p2_chunks = ["\n".join(prefix)]
        self.requests = requests or {}
        self.recorded_choices = {}
        self.pre_state = {}


class _FakeSession:
    """Only the two methods the engine calls."""

    def __init__(self, root=None, raise_on_open=None):
        self.root = root
        self.raise_on_open = raise_on_open
        self.opened = []

    def open_root(self, turn, record=None):
        self.opened.append((turn, record))
        if self.raise_on_open:
            raise self.raise_on_open
        return self.root

    def expand_many(self, arms):
        return []

    def close(self):
        pass


def _engine(arm, **kw):
    cfg = SearchConfig(arm=arm, budget_s=1.0, caps=WidthCaps(m_opp=3, k_worlds=2, r_dice=2), **kw)
    return SearchEngine(model=None, mappings=None, cfg=cfg, pool_packed=[])


def _choose(engine, **kw):
    base = dict(record=RECORD, side="p1", turn=1, our_history=[], our_tokens={0: "move surf",
                                                                              1: "move ice"},
                observed_our_lines=OBSERVED, pub=None, policy_action=1)
    base.update(kw)
    return engine.choose(**base)


# -- the arm contract ---------------------------------------------------------


def test_the_base_arm_plays_the_POLICYS_action_and_says_so():
    """The control must be the literal policy, not a re-implementation of it — otherwise the
    'dividend' includes whatever the re-implementation does differently."""
    res = _choose(_engine("base"))
    assert res.fallback == "no_search"
    assert res.action == 1 and res.policy_action == 1
    assert res.changed is False
    assert res.widths.planned == {"m_opp": 0, "k_worlds": 0, "r_dice": 0}


def test_the_oracle_arm_is_ONE_world_by_construction():
    assert _engine("oracle").cfg.resolved_caps().k_worlds == 1
    assert _engine("honest").cfg.resolved_caps().k_worlds == 2
    assert _engine("base").cfg.resolved_caps().k_worlds == 0


def test_the_oracle_arm_REFUSES_to_run_without_the_true_team():
    """An oracle arm silently searching a determinized world would be an honest arm wearing the
    oracle's label — and the oracle-minus-honest gap is the whole point of having both."""
    res = _choose(_engine("oracle"), opp_true_packed=None)
    assert res.fallback == "search_error"
    assert "true packed team" in res.diagnostics["error"]


def test_the_honest_arm_REFUSES_to_be_handed_the_truth():
    """The mirror guard, and the more dangerous direction: an honest arm given the true team
    would report an ORACLE result under an honest label."""
    res = _choose(_engine("honest"), opp_true_packed="TRUE_TEAM")
    assert res.fallback == "search_error"
    assert "must not be given" in res.diagnostics["error"]


def test_the_honest_arm_DECLINES_rather_than_searching_the_true_team():
    """The subtler half of the same leak, and the one that shipped before this test.

    The live record's `>player` team IS the truth (the builder reads it off the opponent object),
    so a 'no world could be built, use the base team' fallback is an ORACLE search under an
    honest label. With hidden slots still open the engine must decline — and it declines with its
    OWN reason, because 'the pool has no completion for this information set' and 'something
    threw' send a reader to different places."""
    eng = _engine("honest")
    eng._pool_packed = []                       # an empty pool cannot complete anything
    eng._session = _FakeSession(root=_Root(OBSERVED))
    res = _choose(eng)
    assert res.fallback == "no_world"
    assert "no pool-consistent completion" in res.diagnostics["error"]
    assert eng._session.opened == [], "it must not have opened a root on the true team"


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_is_a_known_arm(arm):
    assert _engine(arm).cfg.arm in ARMS


# -- the fallback ladder ------------------------------------------------------


def test_a_decision_with_no_legal_tokens_is_not_a_search_failure():
    res = _choose(_engine("oracle"), our_tokens={}, opp_true_packed="T")
    assert res.fallback == "not_move_selection"
    assert res.action == res.policy_action


def test_a_dead_driver_becomes_root_failed_not_a_crash():
    """A search failure must never cost the battle: a crashed decision would silently drop a
    game, and a dropped game biases the win rate by whatever made it crash."""
    eng = _engine("oracle")
    eng._session = _FakeSession(raise_on_open=RuntimeError("driver died"))
    res = _choose(eng, opp_true_packed="T")
    assert res.fallback == "root_failed"
    assert res.widths.worlds_open_failed == 1
    assert res.widths.worlds_gate_failed == 0, (
        "a dead DRIVER must not be reported as a bad WORLD — the two send a reader to "
        "different places")
    assert res.action == res.policy_action


def test_a_world_that_fails_the_PREFIX_GATE_is_dropped_with_a_counter():
    """The gate is what makes a synthesized record a measurement. A world whose replay does not
    reproduce what we observed is a different battle, and searching it would answer about a
    position we were never in."""
    eng = _engine("oracle")
    eng._session = _FakeSession(root=_Root(["|switch|p2a: A|Blissey, F|100/100", "|turn|1"]))
    res = _choose(eng, opp_true_packed="T")
    assert res.fallback == "prefix_gate_failed"
    assert res.widths.worlds_gate_failed == 1
    assert res.widths.worlds_gated_ok == 0


def test_a_gated_world_with_no_opponent_request_yields_no_candidates():
    eng = _engine("oracle")
    eng._session = _FakeSession(root=_Root(OBSERVED, requests={"p2": {"forceSwitch": [True]}}))
    res = _choose(eng, opp_true_packed="T")
    assert res.widths.worlds_gated_ok == 1
    assert res.fallback == "no_candidates"


def test_a_gated_world_whose_arms_never_score_says_no_scored_arm():
    eng = _engine("oracle")
    req = {"p2": {"active": [{"moves": [{"id": "surf", "move": "Surf", "pp": 10}]}],
                  "side": {"pokemon": []}}}
    eng._session = _FakeSession(root=_Root(OBSERVED, requests=req))   # expand_many returns []
    res = _choose(eng, opp_true_packed="T")
    assert res.widths.worlds_gated_ok == 1
    assert res.fallback == "no_scored_arm"


def test_every_fallback_the_engine_can_emit_is_a_DECLARED_reason():
    """The reason list is the report's vocabulary; an undeclared string would show up in a
    histogram nobody knows how to read."""
    eng_o, eng_h = _engine("oracle"), _engine("honest")
    seen = set()
    eng_o._session = _FakeSession(raise_on_open=RuntimeError("x"))
    seen.add(_choose(eng_o, opp_true_packed="T").fallback)
    seen.add(_choose(_engine("base")).fallback)
    seen.add(_choose(eng_h, opp_true_packed="T").fallback)
    seen.add(_choose(eng_o, our_tokens={}, opp_true_packed="T").fallback)
    assert seen <= set(FALLBACK_REASONS), seen - set(FALLBACK_REASONS)


def test_a_fallback_never_reports_a_CHANGED_decision():
    """`change_rate` is a headline number. A fallback that counted as a change would inflate it
    with decisions the search never made."""
    res = _choose(_engine("base"))
    assert res.fallback and res.changed is False


def test_elapsed_is_recorded_even_on_a_failed_search():
    eng = _engine("oracle")
    eng._session = _FakeSession(raise_on_open=RuntimeError("x"))
    res = _choose(eng, opp_true_packed="T")
    assert res.widths.elapsed_s >= 0.0


# -- iterative deepening ------------------------------------------------------


def test_the_depth_CAP_defaults_to_the_amendments_three_and_is_a_cap_not_a_target():
    """The budget governs the realized depth; `--max-depth` only stops it climbing further. A
    default of 3 therefore costs a 0.5 s cell nothing — it simply never affords a second ply."""
    assert SearchConfig().max_depth == 3
    assert _engine("oracle", max_depth=1).cfg.max_depth == 1


def test_a_decision_reports_the_depth_it_PLANNED_and_the_depth_it_REACHED():
    """The whole content of the amendment is that a budget cell says what it bought. A row that
    printed only the cap would report an intention."""
    eng = _engine("oracle", max_depth=3)
    eng._session = _FakeSession(raise_on_open=RuntimeError("x"))
    res = _choose(eng, opp_true_packed="T")
    assert res.widths.depth_planned == 3
    assert res.widths.depth_realized == 1, "a failed search reached nothing deeper than ply 1"
    assert res.widths.beam_m == 0, "never deepened, so there was no beam"


def test_the_base_arm_never_claims_a_depth_it_did_not_search():
    res = _choose(_engine("base", max_depth=3))
    assert res.widths.depth_realized == 1 and res.widths.beam_m == 0


def test_only_a_clean_MOVE_SELECTION_may_be_branched_at_any_ply():
    """The root already declines a forced switch (`not_move_selection`); every deeper ply obeys the
    same rule. It is not symmetry for its own sake: a deeper node's legal surface comes from the
    action MAPPER, which enumerates switch targets at a forced switch quite happily — branching
    there sends the sim a move for a side that was never asked for one."""
    move_req = {"active": [{"moves": [{"id": "surf", "move": "Surf", "pp": 10}]}],
                "side": {"pokemon": []}}
    assert branchable({"p1": move_req}, "p1") is True
    assert branchable({"p1": {"forceSwitch": [True]}}, "p1") is False
    assert branchable({"p1": {"wait": True}}, "p1") is False
    assert branchable({"p1": {"teamPreview": True}}, "p1") is False
    assert branchable(None, "p1") is False
    assert branchable({"p2": move_req}, "p1") is False, "the rule is per SIDE"


def test_the_cross_world_choice_set_is_the_INTERSECTION_of_the_per_world_beams():
    """An action must be deepened in EVERY world for its cross-world mean to be at one depth.
    Anything else compares a depth-2 value against a depth-1 one — and the comparison has a
    DIRECTION, because a deeper value integrates more opponent replies and is systematically the
    more pessimistic of the two."""
    choices, rule = _selectable_across_worlds([0, 1, 2, 3], [[0, 1, 2], [1, 2, 3]])
    assert choices == [1, 2] and rule == "intersection"


def test_no_deepening_leaves_EVERY_action_selectable_exactly_as_before():
    """Depth 1 must stay byte-for-byte the registered experiment — the amendment adds a ply, it
    does not change what happens when there is no budget for one."""
    choices, rule = _selectable_across_worlds([0, 1, 2], [])
    assert choices == [0, 1, 2] and rule == "depth1"


def test_disjoint_beams_degrade_to_the_UNION_and_SAY_SO():
    """Mixing depths is a defect of the reading, not of the search. Naming it keeps it out of the
    'it just worked' bucket instead of leaving a silent inconsistency in the mean."""
    choices, rule = _selectable_across_worlds([0, 1, 2, 3], [[0, 1], [2, 3]])
    assert choices == [0, 1, 2, 3] and rule == "union"


# -- terminal scoring ---------------------------------------------------------


def test_a_terminal_arm_is_labelled_from_OUR_username():
    assert _terminal_label({"winner": "Me"}, "Me") == "win"
    assert _terminal_label({"winner": "You"}, "Me") == "loss"
    assert _terminal_label({"winner": None}, "Me") == "tie"
    assert _terminal_label({}, "Me") == "tie"


# -- scoring mode -------------------------------------------------------------


class _FakePolicy:
    def __init__(self, values, wp=None):
        import torch

        self._v = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
        self.features_extractor = type("E", (), {"last_win_prob_logits": wp})()

    def predict_values(self, inp):
        return self._v


class _FakeModel:
    def __init__(self, policy):
        self.policy = policy
        self.device = "cpu"


def test_auto_scoring_prefers_the_WIN_PROB_head_when_the_run_trained_one():
    """A shaped-return V and a +/-1 terminal are not the same units, so the calibrated [0,1] head
    is the default whenever it exists — and the mode used is reported per row."""
    import numpy as np
    import torch

    obs = np.zeros((2, 3), dtype=np.float32)
    mask = np.ones((2, 4), dtype=np.float32)
    m = _FakeModel(_FakePolicy([0.7, -0.2], wp=torch.tensor([[0.0], [10.0]])))
    scores, mode = batch_scores(m, obs, mask, "auto")
    assert mode == "win_prob"
    assert scores[0] == pytest.approx(0.5, abs=1e-4)
    assert scores[1] > 0.99


def test_auto_scoring_falls_back_to_V_on_a_run_with_no_win_prob_head():
    import numpy as np

    obs = np.zeros((2, 3), dtype=np.float32)
    mask = np.ones((2, 4), dtype=np.float32)
    m = _FakeModel(_FakePolicy([0.7, -0.2], wp=None))
    scores, mode = batch_scores(m, obs, mask, "auto")
    assert mode == "value"
    assert scores == pytest.approx([0.7, -0.2], abs=1e-6)


def test_an_explicit_value_mode_ignores_the_win_prob_head():
    import numpy as np
    import torch

    obs = np.zeros((1, 3), dtype=np.float32)
    mask = np.ones((1, 4), dtype=np.float32)
    m = _FakeModel(_FakePolicy([0.7], wp=torch.tensor([[10.0]])))
    scores, mode = batch_scores(m, obs, mask, "value")
    assert mode == "value" and scores == pytest.approx([0.7])


def test_a_win_prob_stash_from_a_DIFFERENT_forward_raises_instead_of_scoring_one_arm():
    """The stash lives on ONE shared extractor and the mirror runs two players through it — the
    searched side in a worker thread, the unsearched side on POKE_LOOP. A B=1 forward landing
    between ``predict_values`` and the stash read yields a length-1 score vector for an N-arm
    batch, which ``zip`` in ``_expand_ply`` truncates in silence: N-1 arms vanish and the decision
    goes to whichever action sorted first. Width-check it like α's clause 3 does."""
    import numpy as np
    import torch

    obs = np.zeros((4, 3), dtype=np.float32)
    mask = np.ones((4, 4), dtype=np.float32)
    # 4 values (the batch really was 4 rows) but a stash left behind by someone else's B=1 forward.
    m = _FakeModel(_FakePolicy([0.7, 0.1, 0.2, 0.3], wp=torch.tensor([[10.0]])))
    with pytest.raises(RuntimeError, match="win-prob stash width 1 != scored batch 4"):
        batch_scores(m, obs, mask, "auto")
