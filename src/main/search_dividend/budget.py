"""The per-decision WALL-CLOCK budget and the width allocator.

The registered experiment fixes DEPTH at 1 and spends the budget on WIDTH, in this order:

    1. alpha-pruned OPPONENT actions   (``m_opp``)
    2. determinized WORLDS             (``k_worlds``)
    3. CRN dice resamples              (``r_dice``)

The order is a claim about where the value-target variance lives, and it is not a guess: the
three-axis measurement (``project_three_axis_value_variance``) put the BEHAVIOR-weighted split
at **OPP 59.7% >> DICE 26.5% > OUR 10.0% > OUR x OPP 3.7%**, with the policy concentrating
(median top-action 0.75) while alpha stays FLAT (0.97 ratio). Marginalize the OPPONENT; never
spend the first marginal second on the dice. Worlds sit between them because a determinized
world changes WHICH opponent actions exist at all, so it is the axis that makes the alpha
marginalization mean something.

Two separate mechanisms, deliberately kept apart:

* :func:`allocate` picks widths AHEAD of the search from a measured cost model. It is pure and
  unit-tested.
* :class:`Deadline` is the CLOCK the search consults DURING expansion. A plan that turns out to
  be too expensive is truncated by the clock, and the truncation is COUNTED
  (``deadline_truncated``) — never silently absorbed, because a search that quietly does less
  work than its budget says would make a null dividend unfalsifiable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

# The order the budget is spent in. Exported so the CLI, the tests and the report all read the
# SAME list rather than three hand-copied ones (the failure mode that made a whole edge family
# unlaunchable in `train`'s CLI validator).
WIDTH_ORDER = ("m_opp", "k_worlds", "r_dice")


@dataclass(frozen=True)
class WidthCaps:
    """Upper bounds a plan may not exceed, whatever the budget allows.

    ``k_worlds`` is 1 for the ORACLE arm by construction (the true state is one world), and 0
    for the policy-alone arm (no search at all)."""

    m_opp: int = 6
    k_worlds: int = 8
    r_dice: int = 8


@dataclass(frozen=True)
class WidthPlan:
    """The widths a decision will actually try to run at."""

    m_opp: int = 1
    k_worlds: int = 1
    r_dice: int = 1

    def n_arms(self, n_our_actions: int) -> int:
        return int(n_our_actions) * self.m_opp * self.k_worlds * self.r_dice

    def bumped(self, axis: str) -> "WidthPlan":
        return WidthPlan(**{**asdict(self), axis: getattr(self, axis) + 1})

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CostModel:
    """Measured per-decision cost, in seconds.

    ``world_open_s`` is what one :meth:`SearchSession.open_root` costs (it replays the battle
    prefix, so it grows with the turn number); ``arm_s`` is one expanded arm end to end —
    the sim branch plus the obs materialization plus its share of the batched critic forward.
    The materializer dominates, which is why ``arm_s`` is also turn-dependent and why the
    allocator is re-fed a fresh measurement at every decision rather than a constant.
    """

    world_open_s: float = 0.05
    arm_s: float = 0.01

    def estimate(self, plan: WidthPlan, n_our_actions: int) -> float:
        return (plan.k_worlds * self.world_open_s
                + plan.n_arms(n_our_actions) * self.arm_s)


def allocate(budget_s: float, n_our_actions: int, cost: CostModel,
             caps: WidthCaps = WidthCaps(), *, floor: WidthPlan = WidthPlan()) -> WidthPlan:
    """The largest plan that fits ``budget_s``, spending in :data:`WIDTH_ORDER`.

    Greedy and one-pass on purpose: each axis is raised as far as it will go before the next
    axis is touched, which is precisely what "budget buys width in THIS order" means. A
    smarter joint optimizer would silently re-order the axes and answer a different question.

    Returns at least ``floor`` even when the floor does not fit — a decision must always have a
    plan, and the clock (:class:`Deadline`) is what stops an over-budget one. Caps below the
    floor still win (``k_worlds=0`` on the policy-alone arm must stay 0).
    """
    plan = WidthPlan(
        m_opp=min(floor.m_opp, caps.m_opp),
        k_worlds=min(floor.k_worlds, caps.k_worlds),
        r_dice=min(floor.r_dice, caps.r_dice),
    )
    if plan.k_worlds <= 0 or plan.m_opp <= 0 or n_our_actions <= 0:
        return plan
    for axis in WIDTH_ORDER:
        cap = getattr(caps, axis)
        while getattr(plan, axis) < cap:
            nxt = plan.bumped(axis)
            if cost.estimate(nxt, n_our_actions) > budget_s:
                break
            plan = nxt
    return plan


class Deadline:
    """A wall-clock deadline the expansion loop consults between batches.

    ``clock`` is injectable so the tests do not sleep."""

    def __init__(self, budget_s: float, clock=time.monotonic):
        self._clock = clock
        self._budget = float(budget_s)
        self._start = clock()

    @property
    def budget_s(self) -> float:
        return self._budget

    def elapsed(self) -> float:
        return self._clock() - self._start

    def remaining(self) -> float:
        return self._budget - self.elapsed()

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def fits(self, cost_s: float) -> bool:
        """Would a piece of work costing ``cost_s`` finish inside the budget?

        Asked BEFORE starting a batch, so an over-run is declined rather than committed to."""
        return cost_s <= self.remaining()


@dataclass
class RealizedWidths:
    """What a decision's search ACTUALLY did, as opposed to what it planned.

    Recorded per decision into the results file so the report can show what each budget
    bought — the difference between ``planned`` and the realized counts is the whole content of
    a budget sweep, and a driver that only logged the plan would be reporting its intentions.
    """

    planned: dict
    n_our_actions: int = 0
    opp_candidates: int = 0
    worlds_requested: int = 0
    worlds_gated_ok: int = 0
    # TWO counters, not one. `open_root` raising means the DRIVER is broken; a prefix mismatch
    # means the DETERMINIZATION is wrong. Folding them made a dead driver report
    # `prefix_gate_failed`, i.e. blame the world sampler for a subprocess crash.
    worlds_open_failed: int = 0
    worlds_gate_failed: int = 0
    dice: int = 1
    # MEASURED seconds spent inside `open_root`, summed over every world this decision opened
    # (including the ones that then failed the prefix gate — they cost the same). Recorded because
    # it was the one term of the cost model that was never measured: `world_open_s` sat at its
    # 0.05 default while a real open costs 0.055-0.064 s and grows with the turn, and the shortfall
    # was silently charged to `arm_s`, which is the number the allocator divides the budget by.
    open_s: float = 0.0
    arms_expanded: int = 0
    arms_scored: int = 0
    arms_terminal: int = 0
    # ITERATIVE DEEPENING (the registered depth amendment). `depth_planned` is the cap the CLI
    # asked for; `depth_realized` is what the wall-clock actually bought, which is the reportable
    # one — the whole content of the amendment is that a budget cell should say what depth it
    # reached rather than assert one. A ply is expanded WHOLE or not at all, so this is an integer
    # and not an average over half-explored plies.
    depth_planned: int = 1
    depth_realized: int = 1
    # The widest beam that fit at the last deepened ply (0 = never deepened). Reported beside the
    # depth because "depth 2 over 2 candidates" and "depth 2 over 6" are different searches.
    beam_m: int = 0
    deep_arms_expanded: int = 0
    deep_arms_scored: int = 0
    # RACING (`--root-strategy racing`; see `racing.py`). All zero on the default GRID allocator,
    # which is what makes a mixed results file readable: a row with `racing_rounds == 0` did not
    # race. `racing_arms_saved` is the headline — arm evaluations the elimination avoided against
    # what a uniform sweep over the SAME rounds would have spent — and `racing_resolved` says
    # whether the field actually collapsed to one action or the clock simply ran out with several
    # still live, which are opposite findings and must not read alike.
    racing_rounds: int = 0
    racing_eliminated: int = 0
    racing_resolved: bool = False
    racing_arms_saved: int = 0
    #: Rounds DISCARDED because some live action produced no value. Counted rather than folded in:
    #: on the grid a missing arm dilutes a mean, but a racer would eliminate on it permanently.
    racing_rounds_incomplete: int = 0
    # DEFENSIVE (`--root-strategy defensive`; see `defensive.py`). All empty/zero on every other
    # allocator, which is what keeps a mixed results file readable. The three that carry the
    # finding are kept APART on purpose:
    #   `defensive_gate_reason`  — why the search never ran (a decided position vs no choice)
    #   `defensive_verdict`      — forced | futility | kept | overruled; the strategy's own word
    #   `defensive_banked_s`     — the clock NOT spent, which is what a time manager redistributes
    # A forced decision and a futility stop both play the policy's action and are OPPOSITE
    # findings ("the position was decided" vs "the actions were indistinguishable"); folding them
    # into one counter would hide exactly the thing the strategy is a bet about.
    defensive_verdict: str = ""
    defensive_gate_reason: str = ""
    #: The root P(win) the gate read, or ``-1.0`` when the head published none. A sentinel rather
    #: than ``None`` so the row's dtype is stable, and NEGATIVE rather than 0.5 so an absent
    #: measurement can never be mistaken for a maximally-contested one.
    defensive_root_win_prob: float = -1.0
    defensive_no_win_prob: bool = False
    #: The confirm stage's outcome when ``--defensive-confirm`` is on (``""`` when it is off).
    defensive_confirm_stage: str = ""
    defensive_banked_s: float = 0.0
    deadline_truncated: bool = False
    elapsed_s: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


# The reasons a decision can fall back to the policy's own action. A search that fell back
# SILENTLY would fake a null dividend — every one of these is counted and reported.
FALLBACK_REASONS = (
    "no_search",             # the policy-alone arm — not a failure, the control
    "not_move_selection",    # a forced switch / team preview: no branchable root
    "record_unavailable",    # the live record could not be synthesized yet (turn 0)
    "history_desync",        # our own action history is unreconstructable (a `/choose default`)
    "order_failed",          # the chosen index would not map to a legal order
    "root_failed",           # open_root raised (driver dead, prefix would not build)
    "prefix_gate_failed",    # EVERY world failed the byte-identity gate
    "no_world",              # the honest arm found NO pool-consistent completion (never the truth)
    "no_candidates",         # no opponent candidate survived alpha extraction
    "no_scored_arm",         # every arm failed to materialize an obs
    "deadline",              # the clock expired before a single arm was scored
    "search_error",          # any other exception, captured rather than crashing the battle
    # --- the `playoff` arm's SECOND stage (see `playoff.py`). All three hand the decision back to
    # the POLICY, and that is the arm's design rule rather than a convenience: an unresolved
    # playoff must never fall through to the screen's own argmax, because the screen is the biased
    # estimator the rollouts were called in to replace.
    "playoff_inconclusive",  # the paired difference did not clear 2·SE — the honest refusal
    "playoff_no_budget",     # the deadline bought no pair (or a candidate had no sim token)
    "playoff_error",         # a rollout family raised; counted, never a lost game
    # --- `--root-strategy defensive` (see `defensive.py`). NOTE only these two are fallbacks: a
    # FUTILITY stop is a search VERDICT (the race ran and could not tell the actions apart), so it
    # carries `fallback=None` like the playoff's `screen_decisive` does, keeps the decision inside
    # `n_searched`, and is counted by `defensive_verdict`. That makes `change_rate` over `searched`
    # read as exactly the overrule rate among RACED decisions, which is the registered quantity.
    "defensive_forced",      # the triage gate declined to search (decided position / no choice)
    "defensive_no_win_prob",  # the checkpoint published no P(win); the gate refuses to impute one
)
