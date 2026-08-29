"""RACING root-action selection — best-arm identification by successive elimination.

The registered search spends its budget on a FIXED GRID: every one of our legal actions is scored
on every one of the K x R samples the allocator bought, and the argmax is taken at the end. That
allocation is uniform over actions, and uniform is the one allocation that is never right — a
decision where one action is obviously best pays full price for the six that are obviously not,
and a decision where two actions are genuinely tied spends five sixths of its clock on the other
four.

This module is the adaptive alternative, and it is deliberately the SMALLEST thing that could
work: no bandit index, no posterior, no tuning surface beyond an elimination threshold and a
floor. One *round* draws one sample (a determinized world, with its CRN dice) and scores EVERY
LIVE action on it. Because the sample is shared, the comparison between two actions is PAIRED by
construction — the estimator is the mean of the paired difference ``d_i = v_a(i) - v_b(i)``, whose
variance is the variance of the *disagreement* between two actions rather than the (much larger)
variance of the value itself. An action whose paired-difference confidence interval sits entirely
below the leader's is ELIMINATED and never sampled again, so every later round is cheaper. The
race ends when one action is left, the clock expires, or the sample supply does.

**Why paired, and why that is the whole reason this can win.** The three-axis variance measurement
(``project_three_axis_value_variance``) put the behavior-weighted spread at OPP 59.7% >> DICE
26.5%: most of the variance in ``V(s')`` is variance *about the position*, and it is COMMON to
every action we could take from it. A world where the opponent has a surprise Choice Band lowers
every arm together. Differencing removes exactly that common term, and what is left — how much the
actions actually disagree — is what a decision rule needs. An unpaired allocator has to pay for
the common term N times over; this one never sees it.

**Three properties, each of which a test pins:**

* **CRN pairing is CHECKED, never assumed.** :meth:`Racer.observe` requires a sample covering
  exactly the live set and raises otherwise. A round that quietly scored a subset would make the
  pair statistics compare different rounds against each other, which is the one failure that
  looks like a result.
* **A FLOOR before any elimination.** ``min_samples`` rounds are spent on everything before the
  first candidate may be dropped, because the sample standard deviation of one or two paired
  differences is not an estimate of anything, and an elimination made on it is a coin flip that
  the race then treats as settled forever.
* **The empirical LEADER is never eliminated.** With arms leaving at different rounds the pair
  statistics rest on different sample sets, so the dominance relation they induce is not
  guaranteed to be a total order; without this guard a three-cycle could empty the live set.

**What this module is NOT.** It has no sim, no session, no model and no clock — it consumes
samples and emits a live set, so its arithmetic is unit-testable against synthetic value streams
with a known best arm. The engine's side of the contract (draw a world, score the live actions,
call :meth:`observe`) lives in :mod:`search`; the offline A/B that replays both allocators over a
banked sample matrix lives in :mod:`ab_racing`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: The elimination rules. ``z`` is a plain one-sided normal test on the paired difference at each
#: look — the cheapest, the most aggressive, and the right choice when a wrong elimination is cheap.
#: ``seq`` inflates the radius by a union bound over every round so far and every comparison, so its
#: error rate is controlled ANYTIME rather than per-look; it eliminates later and is the DEFAULT
#: (see :data:`DEFAULT_RULE` — this was measured, not preferred).
RULES = ("z", "seq")

#: MEASURED, not chosen (racing_root_selection_2026-08-28, 180 real decisions, 32 paired samples
#: each). ``z`` at its floor of 3 stops so early that its agreement with a large-budget gold argmax
#: CEILINGS at 0.933 — it structurally cannot reach 95%, however much budget it is given, because
#: the decisions it got wrong were settled and abandoned in three rounds. ``seq`` reaches **1.000**
#: on the same bank, beats ``z`` at every quality level from 90% up (1.41x vs 1.38x at 90%; 1.47x
#: vs never at 95%), and is the only rule that ever pays off. ``z`` stays selectable because it is
#: the cheapest and is right when a wrong elimination is cheap.
DEFAULT_RULE = "seq"

DEFAULT_Z = 2.0
DEFAULT_DELTA = 0.05
DEFAULT_MIN_SAMPLES = 3

#: The floor ``rule="seq"`` enforces on top of :attr:`RacingConfig.min_samples`, because its stated
#: error target does not hold below it. The union-bound radius is exact in the true standard
#: deviation and plugs in the SAMPLE one, which is biased low on a handful of points — so the
#: nominal δ is not delivered at a small floor. MEASURED (600 races, four arms, true gap 0.02
#: against a per-round sd of 0.10, δ=0.05), the rate at which the TRUE best arm is eliminated:
#:
#:     floor      3       4       5       6       8
#:     rate    0.080   0.030   0.0083  0.0050  0.0033
#:
#: and the power to separate a real gap (0.30 against the same noise) is **1.000 at every one of
#: them**, so the floor is free. 5 is where the measurement first clears δ with margin. This is
#: enforced in the RULE rather than documented for the caller, because an error target that
#: silently depends on another parameter is not an error target.
SEQ_MIN_SAMPLES = 5


@dataclass(frozen=True)
class RacingConfig:
    """How aggressively to eliminate.

    ``z`` is the per-comparison one-sided threshold under ``rule="z"``: an action goes out when
    ``mean(d) - z*SE(d) > 0`` against some live rival. At ``z=2`` a single look has a ~2.3%
    per-comparison false-drop rate under normality, and the race takes many looks — which is what
    ``rule="seq"`` and ``delta`` exist for. ``min_samples`` is the floor of rounds a pair must
    share before it may separate. ``max_rounds`` of 0 means the caller's clock governs.
    """

    rule: str = DEFAULT_RULE
    z: float = DEFAULT_Z
    delta: float = DEFAULT_DELTA
    min_samples: int = DEFAULT_MIN_SAMPLES
    max_rounds: int = 0

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"unknown racing rule {self.rule!r} (want one of {RULES})")
        if self.min_samples < 2:
            # A sample standard deviation needs two points. A floor of 1 does not merely eliminate
            # early, it eliminates on an undefined spread.
            raise ValueError("min_samples must be >= 2 — a spread needs two observations")
        if self.z <= 0 or not (0.0 < self.delta < 1.0):
            raise ValueError("z must be > 0 and delta in (0, 1)")

    def effective_min_samples(self) -> int:
        """The floor actually applied. See :data:`SEQ_MIN_SAMPLES` — ``seq`` raises its own."""
        return max(self.min_samples, SEQ_MIN_SAMPLES) if self.rule == "seq" else self.min_samples

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Pair:
    """Running paired-difference statistics for one ordered pair, ``d = v_a - v_b``."""

    n: int = 0
    s: float = 0.0
    s2: float = 0.0

    def add(self, d: float) -> None:
        self.n += 1
        self.s += d
        self.s2 += d * d

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n else 0.0

    @property
    def sd(self) -> float:
        if self.n < 2:
            return float("inf")
        var = (self.s2 - self.s * self.s / self.n) / (self.n - 1)
        return math.sqrt(max(0.0, var))


def separation_radius(pair: _Pair, cfg: RacingConfig, n_comparisons: int) -> float:
    """The half-width the paired mean must clear before an elimination is allowed.

    ``inf`` below two observations (no spread is estimable), and ``0.0`` on an EXACTLY constant
    difference — a pair that produced the same gap on every shared round has a measured spread of
    zero, and refusing to act on that would leave a deterministically dominated action in the race
    forever. That case is real rather than pathological: many gen-3 turns have no dice in them at
    all, so two actions can differ by the same amount in every world.
    """
    if pair.n < 2:
        return float("inf")
    sd = pair.sd
    if sd <= 0.0:
        return 0.0
    if cfg.rule == "z":
        return cfg.z * sd / math.sqrt(pair.n)
    # Union bound over the rounds so far (n and n+1 give a convergent 1/n(n+1) series) and over the
    # comparisons being made, so the FAMILY-WISE error across the whole race is <= delta rather
    # than the per-look error being <= delta.
    k = max(1, int(n_comparisons))
    return sd * math.sqrt(2.0 * math.log(k * pair.n * (pair.n + 1) / cfg.delta) / pair.n)


@dataclass
class RaceOutcome:
    """What a race did — the diagnostics row a decision records."""

    action: int
    rounds: int
    n_actions: int
    live: List[int]
    eliminated: Dict[int, int] = field(default_factory=dict)   # action -> round it went out
    means: Dict[int, float] = field(default_factory=dict)
    stop_reason: str = "running"
    #: Arm evaluations actually spent (Σ over rounds of the live count) against what the fixed
    #: grid would have spent for the SAME number of rounds. The ratio is the saving, per decision.
    arms_spent: int = 0
    arms_grid: int = 0
    #: The round at which the field first collapsed to a single action, or ``None`` if it never
    #: did. This is the "samples to separation" the report distributes.
    separated_at: Optional[int] = None

    @property
    def saving(self) -> float:
        return 1.0 - (self.arms_spent / self.arms_grid) if self.arms_grid else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["saving"] = round(self.saving, 4)
        d["means"] = {int(k): round(float(v), 6) for k, v in self.means.items()}
        return d


class Racer:
    """Successive elimination over CRN-paired samples.

    Drive it one round at a time::

        r = Racer([0, 1, 6, 7], RacingConfig())
        while not r.resolved():
            sample = score(r.live)          # every live action, ONE shared world+dice
            r.observe(sample)
        action = r.leader()
    """

    def __init__(self, actions: Iterable[int], cfg: RacingConfig = RacingConfig()):
        acts = sorted({int(a) for a in actions})
        if not acts:
            raise ValueError("a race needs at least one action")
        self.cfg = cfg
        self._actions: Tuple[int, ...] = tuple(acts)
        self._live: List[int] = list(acts)
        self._eliminated: Dict[int, int] = {}
        self._pairs: Dict[Tuple[int, int], _Pair] = {}
        self._n: Dict[int, int] = {a: 0 for a in acts}
        self._sum: Dict[int, float] = {a: 0.0 for a in acts}
        self.rounds = 0
        self.arms_spent = 0
        self._separated_at: Optional[int] = None
        if len(acts) == 1:
            self._separated_at = 0

    # -- state --------------------------------------------------------------

    @property
    def actions(self) -> Tuple[int, ...]:
        return self._actions

    @property
    def live(self) -> List[int]:
        """The actions a round must score. A COPY — the caller iterates it while ``observe``
        mutates the real one."""
        return list(self._live)

    def resolved(self) -> bool:
        """One action left, or the configured round cap reached."""
        if len(self._live) <= 1:
            return True
        return bool(self.cfg.max_rounds) and self.rounds >= self.cfg.max_rounds

    def means(self) -> Dict[int, float]:
        return {a: (self._sum[a] / self._n[a] if self._n[a] else 0.0) for a in self._actions}

    def leader(self, prefer: Optional[int] = None) -> int:
        """The best live action by its own mean.

        ``prefer`` is broken TOWARD on an exact tie — the engine passes the policy's own action, so
        a race that never separated anything reports no change rather than a change it has no
        evidence for. Same rule the grid path's argmax uses, for the same reason.
        """
        m = self.means()
        return max(self._live, key=lambda a: (m[a], a == prefer, -a))

    # -- the round ----------------------------------------------------------

    def observe(self, sample: Mapping[int, float]) -> List[int]:
        """Fold in ONE CRN-paired round and return the actions it eliminated.

        ``sample`` must cover exactly :attr:`live`. That is the pairing seam and it RAISES rather
        than tolerating a subset: a round that scored only some live actions would contribute to
        some pairs and not others, and the pair means would then rest on different sets of rounds
        while being compared as though they did not.
        """
        keys = {int(k) for k in sample}
        want = set(self._live)
        if keys != want:
            missing = sorted(want - keys)
            extra = sorted(keys - want)
            raise ValueError(
                f"a racing round must score exactly the live set — missing {missing}, "
                f"unexpected {extra}. CRN pairing is what makes the difference CI valid; a "
                "partial round silently invalidates it.")
        self.rounds += 1
        self.arms_spent += len(self._live)
        vals = {int(k): float(v) for k, v in sample.items()}
        for a in self._live:
            self._n[a] += 1
            self._sum[a] += vals[a]
        for i, a in enumerate(self._live):
            for b in self._live[i + 1:]:
                self._pairs.setdefault((a, b), _Pair()).add(vals[a] - vals[b])
        return self._eliminate()

    def _eliminate(self) -> List[int]:
        floor = self.cfg.effective_min_samples()
        if self.rounds < floor or len(self._live) <= 1:
            return []
        n_comparisons = max(1, len(self._actions) - 1)
        # Never drop the empirical leader. With arms leaving at different rounds the pair means
        # rest on different sample sets, so the dominance relation is not guaranteed transitive and
        # an unguarded sweep could empty the live set on a cycle.
        keep = self.leader()
        m = self.means()
        out: List[int] = []
        for b in self._live:
            if b == keep:
                continue
            for a in self._live:
                if a == b or m[a] <= m[b]:
                    continue
                pair = self._pairs.get((a, b)) or self._pairs.get((b, a))
                if pair is None or pair.n < floor:
                    continue
                mean = pair.mean if (a, b) in self._pairs else -pair.mean
                if mean - separation_radius(pair, self.cfg, n_comparisons) > 0.0:
                    out.append(b)
                    break
        for b in out:
            self._live.remove(b)
            self._eliminated[b] = self.rounds
        if len(self._live) <= 1 and self._separated_at is None:
            self._separated_at = self.rounds
        return sorted(out)

    # -- the verdict --------------------------------------------------------

    def outcome(self, *, prefer: Optional[int] = None, stop_reason: str = "") -> RaceOutcome:
        reason = stop_reason or ("resolved" if len(self._live) <= 1 else
                                 ("rounds" if self.resolved() else "running"))
        return RaceOutcome(
            action=self.leader(prefer), rounds=self.rounds, n_actions=len(self._actions),
            live=list(self._live), eliminated=dict(self._eliminated), means=self.means(),
            stop_reason=reason, arms_spent=self.arms_spent,
            arms_grid=self.rounds * len(self._actions), separated_at=self._separated_at)


def race_over_bank(bank: Sequence[Mapping[int, float]], actions: Sequence[int],
                   cfg: RacingConfig = RacingConfig(), *,
                   max_rounds: Optional[int] = None,
                   arm_budget: Optional[int] = None,
                   prefer: Optional[int] = None) -> RaceOutcome:
    """Replay a race over a PRE-COMPUTED bank of paired samples.

    ``bank[i]`` is round *i*'s ``{action: value}`` over the FULL action set — one determinized
    world with its CRN dice, scored for everything. The racer consumes only the live entries, so
    the replay charges exactly what a live race would have spent while every arm is drawn from the
    identical sample that the fixed grid sees. That is what makes the offline A/B paired rather
    than merely matched: the two allocators differ in allocation and in nothing else.

    ``arm_budget`` stops the race when the next round would take it past that many arm
    evaluations — the matched-compute axis. ``max_rounds`` stops it by rounds instead.
    """
    r = Racer(actions, cfg)
    reason = "bank_exhausted"
    n_max = len(bank) if max_rounds is None else min(len(bank), int(max_rounds))
    for i in range(n_max):
        if r.resolved():
            break
        if arm_budget is not None and r.arms_spent + len(r.live) > int(arm_budget):
            reason = "budget"
            break
        row = bank[i]
        r.observe({a: row[a] for a in r.live})
    if len(r.live) <= 1:
        reason = "resolved"
    elif reason == "bank_exhausted" and cfg.max_rounds and r.rounds >= cfg.max_rounds:
        reason = "rounds"
    return r.outcome(prefer=prefer, stop_reason=reason)


def grid_over_bank(bank: Sequence[Mapping[int, float]], actions: Sequence[int], *,
                   max_rounds: Optional[int] = None,
                   arm_budget: Optional[int] = None,
                   prefer: Optional[int] = None) -> RaceOutcome:
    """The FIXED-GRID allocator over the same bank — every action on every round it can afford.

    Returned in the same shape as :func:`race_over_bank` so the A/B compares two rows of one type
    rather than two types. ``eliminated`` is always empty and ``separated_at`` always ``None``:
    the grid never separates anything, which is precisely the property under test.
    """
    acts = sorted({int(a) for a in actions})
    n_max = len(bank) if max_rounds is None else min(len(bank), int(max_rounds))
    if arm_budget is not None and acts:
        n_max = min(n_max, int(arm_budget) // len(acts))
    sums = {a: 0.0 for a in acts}
    for i in range(n_max):
        for a in acts:
            sums[a] += float(bank[i][a])
    means = {a: (sums[a] / n_max if n_max else 0.0) for a in acts}
    best = max(acts, key=lambda a: (means[a], a == prefer, -a)) if acts else -1
    return RaceOutcome(action=best, rounds=n_max, n_actions=len(acts), live=list(acts),
                       means=means, stop_reason="budget" if arm_budget is not None else "rounds",
                       arms_spent=n_max * len(acts), arms_grid=n_max * len(acts))


def fold_racing(decisions: Sequence[Mapping]) -> dict:
    """Fold per-decision racing counters into the ones a battery results row carries.

    ADDITIVE by construction, the same rule its ``playoff.fold_playoff`` sibling follows (ladder
    requirement 3, 87a3f91): a GRID row folds to zeros and reads exactly as it always did, so one
    schema still covers the whole battery.

    SUMS, never means of per-game means — a cell pools unequal decision counts, and
    ``eval_sharding``'s Σwon/Σfinished rule applies here for the same reason. The two counters that
    matter are kept apart on purpose: ``n_racing_resolved`` is the field collapsing to one action,
    ``n_racing`` is merely a race having happened, and a cell where the second is large while the
    first is zero is the "nothing ever separates" regime rather than a working search.
    """
    out = {"n_racing": 0, "n_racing_resolved": 0, "racing_rounds_total": 0,
           "racing_eliminated_total": 0, "racing_arms_saved_total": 0,
           "racing_rounds_incomplete_total": 0}
    for d in decisions:
        w = d.get("widths") or {}
        if not int(w.get("racing_rounds", 0) or 0):
            continue
        out["n_racing"] += 1
        out["n_racing_resolved"] += 1 if w.get("racing_resolved") else 0
        out["racing_rounds_total"] += int(w.get("racing_rounds", 0) or 0)
        out["racing_eliminated_total"] += int(w.get("racing_eliminated", 0) or 0)
        out["racing_arms_saved_total"] += int(w.get("racing_arms_saved", 0) or 0)
        out["racing_rounds_incomplete_total"] += int(w.get("racing_rounds_incomplete", 0) or 0)
    return out
