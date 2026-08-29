"""The racing allocator's decision rule, its error rate, and its wiring into the engine.

Two halves, deliberately separate. The first drives :class:`~main.search_dividend.racing.Racer`
on SYNTHETIC value streams with a known best arm and a known noise level — the only setting in
which "did it eliminate the right thing, and how often did it eliminate the wrong thing" is
answerable at all, because a live decision has no ground truth. The second drives the ENGINE's
round loop with a scripted world scorer, so the bookkeeping (round accounting, the incomplete-round
discard, the untouched grid path) is tested without a sim.
"""

from __future__ import annotations

import random

import pytest

from main.search_dividend.budget import FALLBACK_REASONS, RealizedWidths, WidthCaps, WidthPlan
from main.search_dividend.racing import (RULES, SEQ_MIN_SAMPLES, Racer, RacingConfig,
                                         grid_over_bank,
                                         race_over_bank, separation_radius, _Pair)
from main.search_dividend.search import ROOT_STRATEGIES, SearchConfig, SearchEngine
from main.search_dividend.search_test import OBSERVED, RECORD, _Root


def _bank(means, sd, n, seed=0, common_sd=0.0):
    """``n`` CRN-paired rounds over ``means``.

    ``common_sd`` is the per-round term SHARED by every arm — the position noise the pairing is
    supposed to cancel. It is a free parameter here precisely because it must NOT affect any
    elimination: a rule whose behaviour moves when it is raised is not differencing.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        shared = rng.gauss(0.0, common_sd) if common_sd else 0.0
        out.append({a: m + shared + rng.gauss(0.0, sd) for a, m in enumerate(means)})
    return out


# -- the decision rule --------------------------------------------------------


def test_a_clearly_dominated_action_is_eliminated_and_the_best_one_survives():
    out = race_over_bank(_bank([1.0, 0.0, 0.0], sd=0.05, n=40), [0, 1, 2])
    assert out.action == 0
    assert set(out.eliminated) == {1, 2}
    assert out.stop_reason == "resolved"


def test_actions_are_eliminated_FURTHEST_FIRST():
    """A graded field must go out in order of how far it is from the leader — that ordering is the
    whole reason an adaptive allocator beats a uniform one, and a rule that dropped the near-tie
    before the hopeless action would be spending its samples backwards."""
    out = race_over_bank(_bank([1.0, 0.85, 0.5, 0.0], sd=0.1, n=60), [0, 1, 2, 3])
    assert out.action == 0
    assert out.eliminated[3] <= out.eliminated[2] <= out.eliminated[1]


def test_a_field_of_TIED_actions_never_separates_and_says_so():
    """The regime where racing saves nothing. It must end with several arms still live and a
    stop_reason that is not `resolved` — an allocator that reported a confident winner over pure
    noise would be worse than the grid, not cheaper than it."""
    out = race_over_bank(_bank([0.0, 0.0, 0.0, 0.0], sd=0.1, n=40, seed=7), [0, 1, 2, 3],
                         RacingConfig(rule="seq"))
    assert len(out.live) > 1
    assert out.stop_reason != "resolved"
    assert out.separated_at is None


def test_the_COMMON_per_round_term_does_not_change_a_single_elimination():
    """CRN pairing's entire claim, as an equality. Adding a shared per-round offset ten times the
    size of the arms' own spread must leave the race bit-identical, because every statistic here is
    computed on differences. A rule reading un-differenced values would be destroyed by this."""
    plain = race_over_bank(_bank([0.3, 0.0, -0.2], sd=0.05, n=30, seed=3), [0, 1, 2])
    shared = race_over_bank(_bank([0.3, 0.0, -0.2], sd=0.05, n=30, seed=3, common_sd=0.5),
                            [0, 1, 2])
    # Everything the DECISION rests on is identical; only the absolute means moved, and by exactly
    # the shared term the rule is blind to.
    assert (plain.action, plain.eliminated, plain.live, plain.rounds, plain.arms_spent) == \
           (shared.action, shared.eliminated, shared.live, shared.rounds, shared.arms_spent)


def test_no_elimination_happens_before_the_min_samples_FLOOR():
    """Even against an infinitely obvious loser. Two paired differences have a sample sd that is
    not an estimate of anything, and an elimination made on one is a coin flip the race then treats
    as settled forever."""
    r = Racer([0, 1], RacingConfig(min_samples=5))
    for i in range(4):
        assert r.observe({0: 10.0 + i * 0.01, 1: 0.0}) == []
        assert len(r.live) == 2
    assert r.observe({0: 10.04, 1: 0.0}) == [1]


def test_a_DETERMINISTIC_gap_separates_rather_than_racing_forever():
    """Many gen-3 turns have no dice in them, so two actions can differ by exactly the same amount
    in every world. A zero measured spread is a real measurement, and refusing to act on it would
    leave a dominated action live until the clock ran out."""
    r = Racer([0, 1], RacingConfig(rule="z", min_samples=3))
    for _ in range(3):
        r.observe({0: 1.0, 1: 0.5})
    assert r.live == [0]


def test_an_exactly_constant_TIE_never_separates():
    """The mirror of the case above, and the one that must not be folded in with it: a zero spread
    around a zero gap is evidence of nothing."""
    r = Racer([0, 1], RacingConfig(min_samples=3))
    for _ in range(10):
        r.observe({0: 1.0, 1: 1.0})
    assert sorted(r.live) == [0, 1]


def test_the_empirical_LEADER_is_never_eliminated():
    """With arms leaving at different rounds the pair statistics rest on different sample sets, so
    the dominance relation they induce is not guaranteed transitive. Without this guard a cycle
    would empty the live set and the decision would have nothing to return."""
    for seed in range(60):
        out = race_over_bank(_bank([0.2, 0.1, 0.05, 0.0, -0.1], sd=0.15, n=40, seed=seed),
                             [0, 1, 2, 3, 4])
        assert out.live, f"seed {seed} emptied the live set"
        assert out.action in out.live


@pytest.mark.parametrize("rule", list(RULES))
def test_the_true_best_arm_is_eliminated_at_a_BOUNDED_rate(rule):
    """The error the whole design is exposed to, MEASURED rather than argued.

    A near-tied field (gap 0.02 against a per-round sd of 0.10) sampled 30 times, 400 independent
    draws, counting how often the arm with the highest TRUE mean is eliminated. The two bounds are
    not the same kind of number and that is the finding: ``seq``'s radius carries a union bound
    over every look and every comparison, so its family-wise rate is <= delta by construction;
    ``z`` tests at each look independently and pays for the sequence of looks, which is exactly why
    it eliminates sooner and is the aggressive option rather than the safe one.
    """
    cfg = RacingConfig(rule=rule, z=2.0, delta=0.05)
    bad = sum(0 in race_over_bank(_bank([0.02, 0.0, 0.0, 0.0], sd=0.10, n=30, seed=s),
                                  [0, 1, 2, 3], cfg).eliminated
              for s in range(400))
    rate = bad / 400
    # seq must deliver its own delta (measured 0.008 at its enforced floor of 5); z is the
    # aggressive rule and is held only to a documented ceiling (measured 0.25 — a quarter of
    # near-tied races drop the true best, which is the price of eliminating early and is why the
    # A/B reports agreement with a gold argmax rather than trusting the rule).
    bound = cfg.delta if rule == "seq" else 0.35
    assert rate <= bound, f"{rule}: eliminated the true best in {rate:.3f} of races"


def test_seq_is_strictly_more_conservative_than_z():
    def rate(rule):
        return sum(0 in race_over_bank(_bank([0.02, 0.0, 0.0, 0.0], sd=0.10, n=30, seed=s),
                                       [0, 1, 2, 3], RacingConfig(rule=rule)).eliminated
                   for s in range(200))
    assert rate("seq") < rate("z")


def test_the_radius_is_infinite_below_two_observations():
    """No spread is estimable from one point, so nothing may separate on one."""
    p = _Pair()
    p.add(5.0)
    assert separation_radius(p, RacingConfig(), 1) == float("inf")


# -- the pairing seam ---------------------------------------------------------


def test_a_round_that_scores_only_SOME_live_actions_RAISES():
    """The pairing seam. A partial round contributes to some pairs and not others, and the pair
    means would then rest on different sets of rounds while being compared as though they did
    not — a corruption that produces a plausible answer rather than an error."""
    r = Racer([0, 1, 2])
    with pytest.raises(ValueError, match="exactly the live set"):
        r.observe({0: 1.0, 1: 0.5})


def test_a_round_that_scores_an_ELIMINATED_action_RAISES():
    r = Racer([0, 1, 2], RacingConfig(rule="z", min_samples=2))
    r.observe({0: 1.0, 1: 0.0, 2: 0.0})
    r.observe({0: 1.0, 1: 0.0, 2: 0.0})
    assert r.live == [0]
    with pytest.raises(ValueError, match="exactly the live set"):
        r.observe({0: 1.0, 1: 0.0, 2: 0.0})


def test_the_live_set_a_caller_reads_is_a_COPY():
    r = Racer([0, 1, 2])
    live = r.live
    live.append(99)
    assert r.live == [0, 1, 2]


# -- accounting ---------------------------------------------------------------


def test_arm_accounting_charges_exactly_the_LIVE_set_each_round():
    r = Racer([0, 1, 2], RacingConfig(rule="z", min_samples=2))
    r.observe({0: 1.0, 1: 0.0, 2: 0.0})       # 3 arms
    r.observe({0: 1.0, 1: 0.0, 2: 0.0})       # 3 arms, then 1 and 2 go out
    out = r.outcome()
    assert out.arms_spent == 6
    assert out.arms_grid == 6
    assert out.separated_at == 2


def test_racing_spends_STRICTLY_FEWER_arms_than_the_grid_over_the_same_rounds():
    # A GRADED field, deliberately: with everything eliminated in one round the race stops on the
    # spot and spends exactly what the grid would have. The saving is realized by CONTINUING to
    # sample a narrowed field, so a field that narrows is what demonstrates it.
    bank = _bank([1.0, 1.0, 0.0, 0.0], sd=0.05, n=40, seed=2)
    raced = race_over_bank(bank, [0, 1, 2, 3])
    grid = grid_over_bank(bank, [0, 1, 2, 3], max_rounds=raced.rounds)
    assert raced.rounds > raced.eliminated[3], "the field narrowed but the race stopped anyway"
    assert raced.arms_spent < grid.arms_spent
    assert raced.action == grid.action
    assert raced.saving > 0.0


def test_the_grid_replay_never_eliminates_anything():
    """The property under test, asserted on the control so it cannot drift into the treatment."""
    out = grid_over_bank(_bank([1.0, 0.0], sd=0.01, n=10), [0, 1])
    assert out.eliminated == {} and out.separated_at is None and out.live == [0, 1]


def test_an_arm_budget_stops_the_race_before_it_overruns():
    bank = _bank([0.0, 0.0, 0.0, 0.0], sd=0.2, n=100, seed=5)
    out = race_over_bank(bank, [0, 1, 2, 3], RacingConfig(rule="seq"), arm_budget=20)
    assert out.arms_spent <= 20
    assert out.stop_reason == "budget"


def test_a_replayed_race_is_REPRODUCIBLE():
    bank = _bank([0.4, 0.2, 0.0], sd=0.2, n=25, seed=11)
    assert (race_over_bank(bank, [0, 1, 2]).as_dict()
            == race_over_bank(bank, [0, 1, 2]).as_dict())


def test_an_exact_tie_breaks_TOWARD_the_preferred_action():
    """The engine passes the policy's own action, so a race that separated nothing reports no
    change rather than a change it has no evidence for."""
    bank = [{0: 1.0, 1: 1.0}] * 5
    assert race_over_bank(bank, [0, 1], prefer=1).action == 1
    assert race_over_bank(bank, [0, 1], prefer=0).action == 0


# -- configuration ------------------------------------------------------------


@pytest.mark.parametrize("kw", [{"rule": "bogus"}, {"min_samples": 1}, {"z": 0.0},
                                {"delta": 0.0}, {"delta": 1.0}])
def test_an_unusable_racing_config_RAISES_at_construction(kw):
    with pytest.raises(ValueError):
        RacingConfig(**kw)


def test_a_race_needs_at_least_one_action():
    with pytest.raises(ValueError):
        Racer([])


# -- the engine wiring --------------------------------------------------------


TOKENS = {0: "move surf", 1: "move ice", 2: "switch 2", 3: "switch 3"}
REQ = {"p2": {"active": [{"moves": [{"id": "eq", "move": "Earthquake"}]}]}}


def _engine(strategy="grid", *, racing=RacingConfig(), caps=WidthCaps(m_opp=2, k_worlds=4,
                                                                     r_dice=2)):
    cfg = SearchConfig(arm="oracle", budget_s=100.0, caps=caps, root_strategy=strategy,
                       racing=racing)
    return SearchEngine(model=None, mappings=None, cfg=cfg, pool_packed=[])


def _script(engine, rows):
    """Replace the per-world scorer with a scripted one: ``rows[i]`` is round *i*'s value per
    action over the FULL set, and the engine sees only the live slice — the same substitution the
    offline A/B makes, so the round loop is tested against known values."""
    seen = []

    def fake(w, ctx, our_tokens, actions, widths, deadline, *, max_depth=None):
        i = len(seen)
        seen.append((sorted(actions), max_depth))
        if i >= len(rows):
            return None
        row = rows[i]
        vals = {int(a): float(row[int(a)]) for a in actions if int(a) in row}
        return vals, [], {"score_mode": "value", "n_scored": len(vals), "n_terminal": 0,
                          "tier": 0, "depth": 1, "beam": [], "valued": sorted(vals)}

    engine._score_world = fake                      # type: ignore[assignment]
    return seen


def _choose(engine, tokens=TOKENS, policy_action=1):
    return engine.choose(record=RECORD, side="p1", turn=1, our_history=[], our_tokens=tokens,
                         observed_our_lines=OBSERVED, pub=None, policy_action=policy_action,
                         opp_true_packed="T2")


class _Session:
    def __init__(self, root):
        self.root = root
        self.opened = 0

    def open_root(self, turn, record=None):
        self.opened += 1
        return self.root

    def expand_many(self, arms):
        return []

    def close(self):
        pass


def test_the_DEFAULT_root_strategy_is_the_registered_grid():
    assert SearchConfig().root_strategy == "grid"
    assert ROOT_STRATEGIES[0] == "grid"


def test_an_unknown_root_strategy_RAISES():
    with pytest.raises(ValueError, match="root_strategy"):
        SearchConfig(root_strategy="mcts")


def test_the_GRID_path_never_reaches_the_racing_allocator():
    """OFF means untouched, asserted rather than reviewed: the grid run must not so much as call
    the racing entry point."""
    eng = _engine("grid")
    eng._session = _Session(_Root(OBSERVED, requests=REQ))
    eng._run_racing = lambda *a, **k: pytest.fail("grid must not reach _run_racing")
    res = _choose(eng)
    assert res.widths.racing_rounds == 0 and res.widths.racing_arms_saved == 0
    assert res.diagnostics.get("racing") is None


def test_the_grid_path_leaves_every_racing_counter_at_ZERO():
    """What makes a mixed results file readable — a row that did not race must say so in the same
    field a row that did races in."""
    w = RealizedWidths(planned=WidthPlan().as_dict()).as_dict()
    assert w["racing_rounds"] == 0 and w["racing_resolved"] is False
    assert w["racing_eliminated"] == 0 and w["racing_rounds_incomplete"] == 0


def test_racing_stops_as_soon_as_the_field_collapses_and_records_the_round():
    eng = _engine("racing", racing=RacingConfig(rule="z", min_samples=2))
    eng._session = _Session(_Root(OBSERVED, requests=REQ))
    seen = _script(eng, [{0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}] * 8)
    res = _choose(eng)
    assert res.fallback is None and res.action == 0
    assert res.widths.racing_resolved is True
    assert res.widths.racing_rounds == 2
    assert res.diagnostics["racing"]["separated_at"] == 2
    assert len(seen) == 2                       # it STOPPED; it did not keep scoring one arm


def test_a_racing_round_is_scored_at_DEPTH_ONE():
    """Racing and iterative deepening are two ways to spend the same clock and are not composed
    here — a first round allowed to deepen would swallow the budget the race needs."""
    eng = _engine("racing")
    eng._session = _Session(_Root(OBSERVED, requests=REQ))
    seen = _script(eng, [{0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}] * 8)
    _choose(eng)
    assert seen and all(md == 1 for _acts, md in seen)


def test_elimination_SHRINKS_the_action_set_the_sim_is_asked_to_expand():
    """The mechanism, observed at the seam it has to act on: after a candidate goes out, the very
    next round must not ask the sim for it. Everything else here is bookkeeping about this."""
    eng = _engine("racing", racing=RacingConfig(rule="z", min_samples=3),
                  caps=WidthCaps(m_opp=2, k_worlds=6, r_dice=8))
    eng._session = _Session(_Root(OBSERVED, requests=REQ))
    # NOISE, not constants: a noiseless script separates every pair in the same round, which would
    # test the elimination of a whole field rather than the narrowing of one.
    seen = _script(eng, _bank([0.50, 0.48, 0.0, 0.0], sd=0.02, n=8, seed=1))
    _choose(eng)
    assert seen[0][0] == [0, 1, 2, 3]
    assert seen[-1][0] == [0, 1]                # 2 and 3 stopped being expanded


def test_an_INCOMPLETE_round_is_discarded_rather_than_scored_as_zero():
    """On the grid a missing arm dilutes a mean; here it would eliminate that action permanently on
    a value that was never measured."""
    eng = _engine("racing", racing=RacingConfig(rule="z", min_samples=2),
                  caps=WidthCaps(m_opp=2, k_worlds=4, r_dice=6))
    eng._session = _Session(_Root(OBSERVED, requests=REQ))
    # Action 3 never backs up to a value. If those rounds counted, 3 would be eliminated at 0.0.
    _script(eng, [{0: 0.1, 1: 0.2, 2: 0.15}] * 3 + [{0: 0.1, 1: 0.2, 2: 0.15, 3: 0.9}] * 3)
    res = _choose(eng)
    assert res.widths.racing_rounds_incomplete == 3
    assert res.widths.racing_rounds >= 2
    assert 3 not in (res.diagnostics["racing"]["eliminated"] or {})
    assert res.action == 3                      # the arm the discarded rounds would have buried


def test_racing_reports_the_arms_its_elimination_SAVED():
    eng = _engine("racing", racing=RacingConfig(rule="z", min_samples=3),
                  caps=WidthCaps(m_opp=2, k_worlds=6, r_dice=8))
    eng._session = _Session(_Root(OBSERVED, requests=REQ))
    # The two hopeless arms separate; the near-tied leader pair does not, so the race keeps
    # sampling a NARROWED field — which is the only configuration in which a saving exists at all.
    _script(eng, _bank([0.50, 0.49, 0.0, 0.0], sd=0.03, n=8, seed=1))
    res = _choose(eng)
    r = res.diagnostics["racing"]
    assert res.widths.racing_arms_saved == r["arms_grid"] - r["arms_spent"] > 0
    assert res.widths.racing_eliminated == 2
    assert set(r["eliminated"]) == {2, 3}


@pytest.mark.parametrize("r_dice,rounds", [(2, 2), (5, 5)])
def test_the_ROUND_SUPPLY_is_the_caps_full_grid_not_the_budget_plan(r_dice, rounds):
    """`plan.k_worlds`/`plan.r_dice` are what a UNIFORM sweep could afford over the full action
    set — precisely the numbers racing exists to beat — so planning to them would cap the
    experiment at its own control. The supply is `k_worlds * r_dice` of the RESOLVED caps, which on
    the oracle arm (one true world by construction) is the dice axis alone: raising --max-dice
    raises the number of samples the race may draw, which is the registered width order arriving
    at an adaptive allocator unchanged."""
    eng = _engine("racing", caps=WidthCaps(m_opp=1, k_worlds=3, r_dice=r_dice))
    eng._session = _Session(_Root(OBSERVED, requests=REQ))
    seen = _script(eng, [{0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}] * 20)
    _choose(eng)
    assert len(seen) == rounds


def test_a_race_that_never_separates_still_returns_a_decision_and_says_it_did_not():
    eng = _engine("racing", caps=WidthCaps(m_opp=1, k_worlds=2, r_dice=4))
    eng._session = _Session(_Root(OBSERVED, requests=REQ))
    _script(eng, [{0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}] * 6)
    res = _choose(eng, policy_action=2)
    assert res.fallback is None
    assert res.widths.racing_resolved is False
    assert res.action == 2                      # the exact tie broke toward the policy
    assert res.changed is False


def test_a_racing_decision_that_never_scored_a_round_falls_back_with_a_DECLARED_reason():
    eng = _engine("racing")
    eng._session = _Session(_Root(["|switch|p2a: A|Blissey, F|100/100", "|turn|1"], requests=REQ))
    res = _choose(eng)
    assert res.fallback in FALLBACK_REASONS
    assert res.fallback == "prefix_gate_failed"
    assert res.widths.worlds_gate_failed > 0


def test_a_dead_driver_on_the_racing_path_is_counted_not_crashed():
    eng = _engine("racing")

    class _Dead(_Session):
        def open_root(self, turn, record=None):
            raise RuntimeError("driver died")

    eng._session = _Dead(None)
    res = _choose(eng)
    assert res.fallback == "root_failed" and res.action == 1


def test_two_engines_on_the_same_seed_draw_the_same_CRN_dice():
    """The reproducibility the A/B rests on: the race's only randomness is the seed sequence, and
    two runs of one config must therefore replay the same battle."""
    a, b = _engine("racing"), _engine("racing")
    assert [a._crn_seed(1, i) for i in range(6)] == [b._crn_seed(1, i) for i in range(6)]


def test_a_racing_decision_is_REPRODUCIBLE_end_to_end():
    def run():
        eng = _engine("racing", racing=RacingConfig(min_samples=3))
        eng._session = _Session(_Root(OBSERVED, requests=REQ))
        _script(eng, _bank([0.3, 0.25, 0.0, -0.2], sd=0.05, n=12, seed=4))
        return _choose(eng).diagnostics["racing"]

    assert run() == run()


# -- the battery fold ---------------------------------------------------------


def test_a_GRID_row_folds_the_racing_counters_to_zero():
    """ADDITIVE: a results file mixing both strategies must have one schema, and a row that did not
    race must read exactly as it always did rather than as a race that achieved nothing."""
    from main.search_dividend.racing import fold_racing
    out = fold_racing([{"widths": RealizedWidths(planned={}).as_dict()}, {}, {"fallback": "no_search"}])
    assert set(out.values()) == {0}


def test_the_battery_fold_SUMS_rather_than_averaging_per_game_rates():
    """A cell pools unequal decision counts, so a mean of per-game means would weight a 5-decision
    game like a 40-decision one — the same rule eval_sharding follows."""
    from main.search_dividend.racing import fold_racing
    rows = [{"widths": {"racing_rounds": 4, "racing_resolved": True, "racing_eliminated": 3,
                        "racing_arms_saved": 7, "racing_rounds_incomplete": 1}},
            {"widths": {"racing_rounds": 9, "racing_resolved": False, "racing_eliminated": 1,
                        "racing_arms_saved": 12, "racing_rounds_incomplete": 0}}]
    out = fold_racing(rows)
    assert out["n_racing"] == 2 and out["n_racing_resolved"] == 1
    assert out["racing_rounds_total"] == 13 and out["racing_arms_saved_total"] == 19
    assert out["racing_rounds_incomplete_total"] == 1


def test_the_DEFAULT_rule_is_the_one_that_MEASURED_better():
    """`z` was the registered rule and lost. On 180 real decisions its agreement with a large-budget
    gold argmax ceilings at 0.933 — it cannot reach 95% at ANY budget, because the decisions it got
    wrong were settled and abandoned in three rounds — while `seq` reaches 1.000 and beats it at
    every quality level from 90% up. A default that structurally cannot reach the quality bar is
    the wrong default however cheap it is."""
    assert RacingConfig().rule == "seq"
    assert RacingConfig().effective_min_samples() == SEQ_MIN_SAMPLES
