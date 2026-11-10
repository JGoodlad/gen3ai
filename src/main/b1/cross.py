"""BOUNDARY-1's from-scratch 2 x 2 (width x breadth): METER 1, the head-to-head cross, as code.

The registration is ``designs/research_state/measurements/boundary1_scratch_2026-11-09/PREREG.md`` (§5); this module
is its rule and nothing else: pure functions over cell scores, no I/O, no ledger write. The measurement folder's
``looks/`` scripts plan the cells, play them and call :func:`read_look`.

**The design.** Four arms (:data:`ARMS`): ``C`` control, ``W`` wide, ``Br`` breadth, ``WB`` both; ``n`` runs an arm
(2 as registered; any equal ``n >= 2`` is accepted, so a registered extension reads by the same code). A **cell** is
one run against one run of ANOTHER arm (:meth:`Design.cells`, 24 at ``n = 2``), stored once, from the side of the arm
that comes later in :data:`ARMS`; the other side's score is ``100 - score`` (a mirrored pair is symmetric, a draw is
half a point to each).

**A cell's score** is a weighted mean of its rates on the registered panels (:func:`blend`); a **view** is one set of
weights (:data:`VIEWS`). Each factor is read on ITS view (:data:`DECIDING`): **width on** ``N`` = 1/2 home72 + 1/2
mean(heldout72_hpfix, forum72, p72); **breadth on** ``forum72`` **alone**, the one unfamiliar panel its training teams
do not overlap (hazard M4: 62 % of heldout72_hpfix's and 56.5 % of p72's full sets are in ``n1_proc_v1``, so a
breadth-trained run meets its own training teams there; width needs no such care, because its ON and OFF sides each
hold one breadth-trained arm and the overlap cancels). Every other view is REPORTED.

**A factor's statistic** (:func:`factor_stat`). Rows = the factor's ``2n`` ON runs, columns = its ``2n`` OFF runs
(:data:`FACTORS`), ``h`` in pp from the row's side.

* ``D = mean(h) - 50``.
* A run's row mean is its mean score against ALL the factor's OFF runs, so the other factor's effect is a constant
  within an arm and cancels in the arm's WITHIN-arm sample variance of the row means. ``s2_rows`` is the mean, over
  the two ON arms, of that within-arm variance, **each floored at** :data:`FLOOR_PP` ``** 2`` (the measured seed SD:
  a lucky small variance at two seeds an arm must not pass). ``s2_cols`` the same over the OFF arms.
* ``V = s2_rows / 2n + s2_cols / 2n``, ``df = 4 (n - 1)`` (one per within-arm variance at ``n = 2``), a two-sided
  :data:`LEVEL` interval by Student's t. The floor makes ``V > 0`` always.

**Classes** (:func:`classify`): the 100M look is WIN / LOSS / NO-USEFUL-GAIN / UNDECIDED on the factor's deciding
view; the 50M look is STOP-HARM / CONTINUE on its harm view (:data:`HARM_VIEW`); the 75M and the equal-compute (EQ) looks are REPORTED. A bound
within :data:`EPS_PP` of the threshold it is compared with fails that comparison and is named in ``at_boundary`` (root
rule 8: the conservative class, deterministically).

**Breadth's net rule** (:func:`net_rule`, the owner's exchange rate r = 1): breadth's WIN on forum72 is adoptable only
if ``D_home72 + D_forum72 > 0`` (point estimates of the same statistic; exactly zero fails); otherwise it is
WIN-NET-NEGATIVE. That sum
is the ONLY home-team cost this registration recognises: **the home72 statistic alone never vetoes a generalisation
lever** (it is reported as the cost, and no class is read from its interval). Breadth's 50M harm stop reads the same
sum's view, ``HF`` = 1/2 home72 + 1/2 forum72 (``net = 2 D_HF``).

**The confirming meter** (:func:`confirm`): Kakuna clustered by team (``main.reads``' ``b1_kakuna`` rule), one
contrast a factor (:data:`CONFIRMING`). It never decides; a contradiction (IN-HOUSE-ONLY, KAKUNA-ONLY) goes to the
owner, and an INCONCLUSIVE Kakuna read is UNREAD (finish or replay it; no outcome is acted on meanwhile).

**After a STOP-HARM** (:class:`PairDesign`, :func:`read_survivor`): two arms are left, the stopped factor's OFF arms.
Their one simple effect is REPORTED on every view with its 2-df interval and NO class: the registration decides
nothing further by rule, and the choice goes to the owner.

**The interaction** (:func:`interaction_stat`, REPORTED): ``I = (width's effect with breadth on) - (width's effect
with breadth off)``, estimated from both decompositions of the same four arm means; ``V = sum over arms of the
floored within-arm variance / n``, ``df = 4 (n - 1)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from main.h2h.cross import CrossInputError, t_ppf

#: The four arms, in the order that fixes a cell's stored orientation (the LATER arm is the player).
ARMS: Tuple[str, ...] = ("C", "W", "Br", "WB")
#: factor -> (its ON arms, its OFF arms). Each side lists the other factor's OFF arm first.
FACTORS: Dict[str, Tuple[Tuple[str, str], Tuple[str, str]]] = {
    "width": (("W", "WB"), ("C", "Br")),
    "breadth": (("Br", "WB"), ("C", "W")),
}
#: The registered panels (``designs/research_state/heldout_panels.json``) and the mirrored pairs a cell plays on each.
PAIRS: Dict[str, int] = {"home72": 1000, "heldout72_hpfix": 500, "forum72": 500, "p72": 500}
PANELS: Tuple[str, ...] = tuple(PAIRS)
#: view -> panel weights.
VIEWS: Dict[str, Dict[str, float]] = {
    "N": {"home72": 0.5, "heldout72_hpfix": 1.0 / 6.0, "forum72": 1.0 / 6.0, "p72": 1.0 / 6.0},
    "home72": {"home72": 1.0},
    "U": {"heldout72_hpfix": 1.0 / 3.0, "forum72": 1.0 / 3.0, "p72": 1.0 / 3.0},
    "heldout72_hpfix": {"heldout72_hpfix": 1.0},
    "forum72": {"forum72": 1.0},
    "p72": {"p72": 1.0},
    "HF": {"home72": 0.5, "forum72": 0.5},
}
N_VIEW = "N"
#: The one unfamiliar panel no arm's training teams overlap (hazard M4).
CLEAN_VIEW = "forum72"
#: factor -> the view its 100M class is read on.
DECIDING: Dict[str, str] = {"width": N_VIEW, "breadth": CLEAN_VIEW}
#: factor -> the view its 50M harm stop is read on.
HARM_VIEW: Dict[str, str] = {"width": N_VIEW, "breadth": "HF"}
#: factor -> (cost view, gain view): its WIN is adoptable only if the two estimates sum to MORE than 0 (r = 1).
NET: Dict[str, Tuple[str, str]] = {"breadth": ("home72", CLEAN_VIEW)}
#: factor -> the Kakuna contrast (``main.b1.kakuna_rule``) that confirms it.
CONFIRMING: Dict[str, str] = {"width": "width_U", "breadth": "breadth_F"}

FLOOR_PP = 1.76  # the pooled noise-removed seed SD at 15M (seed_spread.log, 90 df)
BAR_PP = 4.0  # the smallest gain worth the arm's cost: NO-USEFUL-GAIN's bar
HARM_PP = -2.0  # the 50M harm stop
LEVEL = 0.90  # two-sided
EPS_PP = 0.005  # a bound this close to its threshold fails the comparison (AT-BOUNDARY)

LOOKS: Tuple[str, ...] = ("50M", "75M", "100M", "EQ")
#: The looks that classify nothing: 75M, and EQ, the equal-compute read (wide arms at an earlier step).
REPORTED_LOOKS: Tuple[str, ...] = ("75M", "EQ")
WIN, LOSS, NO_USEFUL_GAIN, UNDECIDED = "WIN", "LOSS", "NO-USEFUL-GAIN", "UNDECIDED"
STOP_HARM, CONTINUE, REPORTED = "STOP-HARM", "CONTINUE", "REPORTED"
WIN_NET_NEGATIVE = "WIN-NET-NEGATIVE"
#: The confirming meter's classes (``main.b1.kakuna_rule``) and what they make of an in-house class.
KAKUNA_POSITIVE, KAKUNA_NEGATIVE, KAKUNA_SPANS_ZERO, KAKUNA_INCONCLUSIVE = ("POSITIVE", "NEGATIVE", "SPANS-ZERO",
                                                                            "INCONCLUSIVE")
CONFIRMED, UNCONFIRMED, IN_HOUSE_ONLY, KAKUNA_ONLY, NO_CONTRADICTION = ("CONFIRMED", "UNCONFIRMED", "IN-HOUSE-ONLY",
                                                                       "KAKUNA-ONLY", "NO-CONTRADICTION")
#: The confirming read was INCONCLUSIVE (unplayed units, a short panel, timeouts): nothing was confirmed or contradicted.
UNREAD = "UNREAD"
#: surviving factor -> (its ON arm, the control): the two arms left once the OTHER factor was stopped at 50M.
SURVIVOR_ARMS: Dict[str, Tuple[str, str]] = {"width": ("W", "C"), "breadth": ("Br", "C")}
NO_CLASS = "NO-CLASS"
#: A cell is INCONCLUSIVE when more than this share of its games were DRAWN: ``main.h2h`` ends a game at the turn
#: limit as a draw (it records no aborted game), so a draw is the head-to-head's timeout (root: a timeout is never a
#: semantic outcome). Compared in integers: ``DRAW_LIMIT[1] * draws > DRAW_LIMIT[0] * games``.
DRAW_LIMIT: Tuple[int, int] = (1, 4)
SYNERGY, ANTAGONISM, NOT_SHOWN = "SYNERGY", "ANTAGONISM", "NOT-SHOWN"

Cell = Tuple[str, str]
Scores = Mapping[Cell, float]


# ------------------------------------------------------------------------------------------------- the design
@dataclass(frozen=True)
class Design:
    """arm -> its runs (any hashable-as-string identity: a run name, a sha256), in seed order."""

    runs: Mapping[str, Tuple[str, ...]]

    def __post_init__(self) -> None:
        if set(self.runs) != set(ARMS):
            raise CrossInputError(f"the design's arms {sorted(self.runs)} are not {sorted(ARMS)}")
        sizes = {a: len(self.runs[a]) for a in ARMS}
        if len(set(sizes.values())) != 1 or min(sizes.values()) < 2:
            raise CrossInputError(f"every arm needs the same number (>= 2) of runs, got {sizes}")
        flat = [r for a in ARMS for r in self.runs[a]]
        if len(set(flat)) != len(flat):
            raise CrossInputError("a run appears twice in the design")

    @property
    def n(self) -> int:
        return len(self.runs[ARMS[0]])

    def arm_of(self, run: str) -> str:
        for a in ARMS:
            if run in self.runs[a]:
                return a
        raise CrossInputError(f"{run!r} is not a run of the design")

    def cells(self) -> List[Cell]:
        """Every pair of runs from different arms, once: (player, opponent), the player from the LATER arm."""
        out: List[Cell] = []
        for i, lo in enumerate(ARMS):
            for hi in ARMS[i + 1:]:
                out.extend((p, o) for p in self.runs[hi] for o in self.runs[lo])
        return out


def design_of(runs: Mapping[str, Sequence[str]]) -> Design:
    return Design({a: tuple(rs) for a, rs in runs.items()})


@dataclass(frozen=True)
class PairDesign:
    """What is left of the 2 x 2 after a STOP-HARM: the SURVIVING factor's ON arm and the control
    (:data:`SURVIVOR_ARMS`), ``n`` runs each. It has the cells of those two arms alone and reads no factor."""

    runs: Mapping[str, Tuple[str, ...]]

    def __post_init__(self) -> None:
        if tuple(sorted(self.runs)) not in {tuple(sorted(v)) for v in SURVIVOR_ARMS.values()}:
            raise CrossInputError(f"a survivor design holds one of {sorted(SURVIVOR_ARMS.values())}, "
                                  f"got {sorted(self.runs)}")
        sizes = {a: len(rs) for a, rs in self.runs.items()}
        if len(set(sizes.values())) != 1 or min(sizes.values()) < 2:
            raise CrossInputError(f"every arm needs the same number (>= 2) of runs, got {sizes}")
        flat = [r for rs in self.runs.values() for r in rs]
        if len(set(flat)) != len(flat):
            raise CrossInputError("a run appears twice in the design")

    @property
    def arms(self) -> Tuple[str, str]:
        """(the surviving factor's ON arm, the control)."""
        on = next(a for a in self.runs if a != "C")
        return on, "C"

    @property
    def survivor(self) -> str:
        return next(f for f, v in SURVIVOR_ARMS.items() if v == self.arms)

    @property
    def n(self) -> int:
        return len(self.runs["C"])

    def cells(self) -> List[Cell]:
        on, off = self.arms
        return [(p, o) for p in self.runs[on] for o in self.runs[off]]


def pair_design_of(runs: Mapping[str, Sequence[str]]) -> PairDesign:
    return PairDesign({a: tuple(rs) for a, rs in runs.items()})


AnyDesign = Any  # a Design or a PairDesign: ``.runs``, ``.n`` and ``.cells()``


def oriented(scores: Scores, a: str, b: str) -> float:
    """``a``'s score against ``b`` in pp, whichever side the cell is stored from."""
    fwd, rev = scores.get((a, b)), scores.get((b, a))
    if fwd is not None and rev is not None:
        raise CrossInputError(f"the cell {a} vs {b} is stored from both sides")
    if fwd is not None:
        return float(fwd)
    if rev is not None:
        return 100.0 - float(rev)
    raise CrossInputError(f"missing cell {a} vs {b}")


def blend(design: AnyDesign, by_panel: Mapping[str, Scores], view: str = N_VIEW) -> Dict[Cell, float]:
    """A cell's score on ``view``: the weighted mean of its panel rates. Every panel the view weights must hold
    EXACTLY the design's cells (a missing or an extra cell is refused, never averaged around)."""
    if view not in VIEWS:
        raise CrossInputError(f"view {view!r} is not one of {sorted(VIEWS)}")
    weights = VIEWS[view]
    want = design.cells()
    for panel in weights:
        got = by_panel.get(panel)
        if got is None:
            raise CrossInputError(f"view {view}: no scores for panel {panel}")
        missing = [c for c in want if c not in got]
        extra = sorted(set(got) - set(want))
        if missing or extra:
            raise CrossInputError(f"view {view}, panel {panel}: {len(missing)} cell(s) missing, {len(extra)} outside "
                                  f"the design (first: {(missing or extra)[0]})")
        bad = [c for c in want if not (math.isfinite(float(got[c])) and 0.0 <= float(got[c]) <= 100.0)]
        if bad:
            raise CrossInputError(f"view {view}, panel {panel}: the score of {bad[0]} is not a rate in [0, 100] pp")
    return {c: math.fsum(w * float(by_panel[p][c]) for p, w in weights.items()) for c in want}


# ---------------------------------------------------------------------------------------------- the statistic
def _mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def _var(xs: Sequence[float]) -> float:
    m = _mean(xs)
    return math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1)


@dataclass(frozen=True)
class ArmVar:
    """One arm's within-arm sample variance of its runs' means, before and after the floor."""

    arm: str
    runs: Tuple[str, ...]
    means: Tuple[float, ...]
    s2_raw: float
    s2: float
    floored: bool

    def as_dict(self) -> Dict[str, Any]:
        return {"arm": self.arm, "runs": list(self.runs), "means_pp": list(self.means), "s2_raw": self.s2_raw,
                "s2": self.s2, "floored": self.floored}


def _arm_var(arm: str, runs: Sequence[str], means: Sequence[float], floor_pp: float) -> ArmVar:
    raw = _var(means)
    floor = floor_pp * floor_pp
    return ArmVar(arm=arm, runs=tuple(runs), means=tuple(means), s2_raw=raw, s2=max(raw, floor), floored=raw < floor)


@dataclass(frozen=True)
class Stat:
    """An estimate with its floored variance and its two-sided interval. ``what`` names it (a factor, the
    interaction, a simple effect); ``view`` is the panel view its cells were blended on."""

    what: str
    view: str
    estimate: float
    groups: Tuple[ArmVar, ...]
    v_hat: float
    se: float
    df: int
    level: float
    lo: float
    hi: float
    floor_pp: float

    def as_dict(self) -> Dict[str, Any]:
        return {"what": self.what, "view": self.view, "estimate_pp": self.estimate, "v_hat": self.v_hat,
                "se_pp": self.se, "df": self.df, "level": self.level, "interval_pp": [self.lo, self.hi],
                "floor_pp": self.floor_pp, "groups": [g.as_dict() for g in self.groups]}


def _stat(what: str, view: str, estimate: float, groups: Sequence[ArmVar], v_hat: float, df: int, level: float,
          floor_pp: float) -> Stat:
    if not (0.0 < level < 1.0):
        raise CrossInputError(f"level {level} is not in (0, 1)")
    if not v_hat > 0.0:
        raise CrossInputError("zero variance: the interval is undefined (a zero floor on identical run means?)")
    se = math.sqrt(v_hat)
    q = t_ppf(0.5 + level / 2.0, df)
    return Stat(what=what, view=view, estimate=estimate, groups=tuple(groups), v_hat=v_hat, se=se, df=df, level=level,
                lo=estimate - q * se, hi=estimate + q * se, floor_pp=floor_pp)


def factor_matrix(design: Design, scores: Scores, factor: str) -> Tuple[Tuple[str, ...], Tuple[str, ...],
                                                                         Tuple[Tuple[float, ...], ...]]:
    """``(row runs, column runs, h)``: the factor's ON runs against its OFF runs, pp from the row's side."""
    if factor not in FACTORS:
        raise CrossInputError(f"factor {factor!r} is not one of {sorted(FACTORS)}")
    on, off = FACTORS[factor]
    rows = tuple(r for a in on for r in design.runs[a])
    cols = tuple(r for a in off for r in design.runs[a])
    return rows, cols, tuple(tuple(oriented(scores, r, c) for c in cols) for r in rows)


def factor_stat(design: Design, scores: Scores, factor: str, *, view: str = N_VIEW,
                floor_pp: float = FLOOR_PP, level: float = LEVEL) -> Stat:
    """The factor's main effect on already-blended ``scores`` (module docstring). ``view`` only labels the result."""
    rows, cols, h = factor_matrix(design, scores, factor)
    on, off = FACTORS[factor]
    n = design.n
    row_mean = {r: _mean(h[i]) for i, r in enumerate(rows)}
    col_mean = {c: _mean([h[i][j] for i in range(len(rows))]) for j, c in enumerate(cols)}
    g_rows = [_arm_var(a, design.runs[a], [row_mean[r] for r in design.runs[a]], floor_pp) for a in on]
    g_cols = [_arm_var(a, design.runs[a], [col_mean[c] for c in design.runs[a]], floor_pp) for a in off]
    s2_rows = _mean([g.s2 for g in g_rows])
    s2_cols = _mean([g.s2 for g in g_cols])
    v = s2_rows / (2 * n) + s2_cols / (2 * n)
    d = _mean([x for line in h for x in line]) - 50.0
    return _stat(factor, view, d, g_rows + g_cols, v, 4 * (n - 1), level, floor_pp)


def _cell_mean(design: Design, scores: Scores, a: str, b: str) -> float:
    return _mean([oriented(scores, r, c) for r in design.runs[a] for c in design.runs[b]])


def interaction_stat(design: Design, scores: Scores, *, view: str = N_VIEW, floor_pp: float = FLOOR_PP,
                     level: float = LEVEL) -> Stat:
    """``I`` = width's effect with breadth on minus width's effect with breadth off (= breadth's effect on the wide
    arm minus on the narrow one). With arm-mean scores ``m(a, b)``:

        I = 1/2 [(m(WB, Br) - m(W, C)) + (m(WB, W) - m(Br, C))]

    Each run enters through ONE per-run quantity ``u`` (below) whose arm means sum to ``I`` taken over {WB, C} and,
    equally, over {W, Br}; a run's own strength moves ``u`` one for one, so ``V = sum over the four arms of the
    floored within-arm variance of u, / n``."""
    n = design.n

    def vs(run: str, arm: str) -> float:
        return _mean([oriented(scores, run, o) for o in design.runs[arm]])

    u: Dict[str, List[float]] = {
        "WB": [0.5 * (vs(r, "Br") + vs(r, "W")) for r in design.runs["WB"]],
        "C": [0.5 * (vs(r, "W") + vs(r, "Br")) for r in design.runs["C"]],
        "W": [0.5 * (vs(r, "C") + vs(r, "WB")) for r in design.runs["W"]],
        "Br": [0.5 * (vs(r, "C") + vs(r, "WB")) for r in design.runs["Br"]],
    }
    est = 0.5 * ((_cell_mean(design, scores, "WB", "Br") - _cell_mean(design, scores, "W", "C"))
                 + (_cell_mean(design, scores, "WB", "W") - _cell_mean(design, scores, "Br", "C")))
    groups = [_arm_var(a, design.runs[a], u[a], floor_pp) for a in ARMS]
    v = math.fsum(g.s2 for g in groups) / n
    return _stat("interaction", view, est, groups, v, 4 * (n - 1), level, floor_pp)


def simple_effect(design: AnyDesign, scores: Scores, a: str, b: str, *, view: str = N_VIEW,
                  floor_pp: float = FLOOR_PP, level: float = LEVEL) -> Stat:
    """Arm ``a`` against arm ``b`` alone (REPORTED): ``mean - 50`` over their ``n x n`` cells, ``V = s2_a / n +
    s2_b / n`` with each within-arm variance floored, ``df = 2 (n - 1)``."""
    if a not in design.runs or b not in design.runs or a == b:
        raise CrossInputError(f"a simple effect needs two different arms of {sorted(design.runs)}, got {a!r}, {b!r}")
    n = design.n
    ra = [_mean([oriented(scores, r, c) for c in design.runs[b]]) for r in design.runs[a]]
    rb = [_mean([oriented(scores, r, c) for r in design.runs[a]]) for c in design.runs[b]]
    ga, gb = _arm_var(a, design.runs[a], ra, floor_pp), _arm_var(b, design.runs[b], rb, floor_pp)
    return _stat(f"{a}_vs_{b}", view, _mean(ra) - 50.0, [ga, gb], ga.s2 / n + gb.s2 / n, 2 * (n - 1), level, floor_pp)


# ------------------------------------------------------------------------------------------------- the classes
def _clears(value: float, threshold: float, *, above: bool) -> Tuple[bool, bool]:
    """``(clears, near)``: within :data:`EPS_PP` of the threshold is NEAR and never clears."""
    if abs(value - threshold) <= EPS_PP:
        return False, True
    return (value > threshold) if above else (value < threshold), False


@dataclass(frozen=True)
class Verdict:
    look: str
    outcome: str
    at_boundary: Tuple[str, ...]
    stat: Stat

    def as_dict(self) -> Dict[str, Any]:
        return {"look": self.look, "outcome": self.outcome, "at_boundary": list(self.at_boundary),
                "stat": self.stat.as_dict()}


def classify(look: str, stat: Stat, *, bar_pp: float = BAR_PP, harm_pp: float = HARM_PP) -> Verdict:
    """A factor's class at ``look`` from its interval (the 75M and EQ looks are REPORTED and classify nothing)."""
    if look not in LOOKS:
        raise CrossInputError(f"look {look!r} is not one of {LOOKS}")
    near: List[str] = []

    def test(name: str, value: float, threshold: float, above: bool) -> bool:
        ok, nb = _clears(value, threshold, above=above)
        if nb:
            near.append(f"{name}: {value!r} within {EPS_PP} pp of {threshold}")
        return ok

    if look in REPORTED_LOOKS:
        return Verdict(look, REPORTED, (), stat)
    if look == "50M":
        harm = test("STOP-HARM (upper end)", stat.hi, harm_pp, False)
        return Verdict(look, STOP_HARM if harm else CONTINUE, tuple(near), stat)
    if test("WIN (lower end)", stat.lo, 0.0, True):
        return Verdict(look, WIN, tuple(near), stat)
    if test("LOSS (upper end)", stat.hi, 0.0, False):
        return Verdict(look, LOSS, tuple(near), stat)
    if test("NO-USEFUL-GAIN (upper end)", stat.hi, bar_pp, False):
        return Verdict(look, NO_USEFUL_GAIN, tuple(near), stat)
    return Verdict(look, UNDECIDED, tuple(near), stat)


def net_rule(factor: str, verdict: Verdict, cost: Optional[Stat], gain: Optional[Stat]
             ) -> Tuple[str, Optional[Dict[str, Any]], Tuple[str, ...]]:
    """``(outcome, net, at_boundary)``: the net rule applied to a factor's 100M class. A factor of :data:`NET` whose
    class is WIN keeps it only if its cost-view and gain-view estimates sum to MORE than zero (a sum within
    :data:`EPS_PP` of zero fails, AT-BOUNDARY); otherwise it is WIN-NET-NEGATIVE. Nothing else is touched: the cost
    view's own interval is never read, so the home panel alone vetoes nothing."""
    if factor not in NET:
        return verdict.outcome, None, ()
    cost_view, gain_view = NET[factor]
    if cost is None or gain is None or cost.view != cost_view or gain.view != gain_view or cost.what != factor \
            or gain.what != factor:
        raise CrossInputError(f"{factor}'s net rule needs its {cost_view} and {gain_view} statistics")
    net = cost.estimate + gain.estimate
    near = abs(net) <= EPS_PP
    passes = net > EPS_PP
    info = {"cost_view": cost_view, "gain_view": gain_view, "cost_pp": cost.estimate, "gain_pp": gain.estimate,
            "net_pp": net, "passes": passes}
    if verdict.outcome != WIN:
        return verdict.outcome, info, ()
    notes = (f"net rule: {net!r} within {EPS_PP} pp of 0",) if near else ()
    return (WIN if passes else WIN_NET_NEGATIVE), info, notes


def confirm(outcome: str, kakuna_class: str) -> Tuple[str, bool]:
    """``(label, to_owner)``: what the confirming meter makes of a factor's in-house 100M outcome. The Kakuna read
    never changes the outcome; it CONFIRMS a WIN (its interval above zero), leaves it UNCONFIRMED (spans zero), or
    CONTRADICTS: a WIN whose Kakuna interval is below zero is IN-HOUSE-ONLY, and any other outcome whose Kakuna
    interval is above zero is KAKUNA-ONLY. Both contradictions go to the owner. An INCONCLUSIVE Kakuna read confirms
    and contradicts nothing: UNREAD, whatever the outcome. The read is finished or replayed and read again; ``to_owner``
    is True because no outcome is acted on while it stands (so an unfinished read is never a way past the meter)."""
    if outcome not in (WIN, WIN_NET_NEGATIVE, LOSS, NO_USEFUL_GAIN, UNDECIDED):
        raise CrossInputError(f"{outcome!r} is not a 100M outcome")
    if kakuna_class not in (KAKUNA_POSITIVE, KAKUNA_NEGATIVE, KAKUNA_SPANS_ZERO, KAKUNA_INCONCLUSIVE):
        raise CrossInputError(f"{kakuna_class!r} is not a class of the confirming meter")
    if kakuna_class == KAKUNA_INCONCLUSIVE:
        return UNREAD, True
    if outcome == WIN:
        if kakuna_class == KAKUNA_POSITIVE:
            return CONFIRMED, False
        if kakuna_class == KAKUNA_NEGATIVE:
            return IN_HOUSE_ONLY, True
        return UNCONFIRMED, False
    if kakuna_class == KAKUNA_POSITIVE:
        return KAKUNA_ONLY, True
    return NO_CONTRADICTION, False


def interaction_class(stat: Stat) -> Tuple[str, Tuple[str, ...]]:
    """REPORTED: SYNERGY (the interval above zero), ANTAGONISM (below) or NOT-SHOWN."""
    near: List[str] = []
    up, n1 = _clears(stat.lo, 0.0, above=True)
    down, n2 = _clears(stat.hi, 0.0, above=False)
    if n1:
        near.append(f"SYNERGY (lower end): {stat.lo!r} within {EPS_PP} pp of 0")
    if n2:
        near.append(f"ANTAGONISM (upper end): {stat.hi!r} within {EPS_PP} pp of 0")
    return (SYNERGY if up else ANTAGONISM if down else NOT_SHOWN), tuple(near)


# ---------------------------------------------------------------------------------------------------- a look
#: The simple effects a look reports: (a, b) = arm a against arm b.
SIMPLE_EFFECTS: Tuple[Tuple[str, str], ...] = (("W", "C"), ("Br", "C"), ("WB", "C"), ("WB", "Br"), ("WB", "W"),
                                               ("Br", "W"))


def read_look(design: Design, by_panel: Mapping[str, Scores], look: str, *, floor_pp: float = FLOOR_PP,
              level: float = LEVEL) -> Dict[str, Any]:
    """Everything a look reports, JSON-ready. ``factors.<f>.outcome`` is the factor's class at this look: at 100M its
    class on :data:`DECIDING` after the net rule, at 50M STOP-HARM / CONTINUE on :data:`HARM_VIEW`, at 75M and EQ REPORTED.
    ``factors.<f>.views`` holds the same statistic on every view (REPORTED); the interaction and the simple effects
    are REPORTED on ``N`` and on :data:`CLEAN_VIEW`."""
    if look not in LOOKS:
        raise CrossInputError(f"look {look!r} is not one of {LOOKS}")
    blended = {v: blend(design, by_panel, v) for v in VIEWS}
    out: Dict[str, Any] = {"look": look, "n_per_arm": design.n, "runs": {a: list(design.runs[a]) for a in ARMS},
                           "floor_pp": floor_pp, "level": level, "factors": {}}
    for f in FACTORS:
        stats = {v: factor_stat(design, blended[v], f, view=v, floor_pp=floor_pp, level=level) for v in VIEWS}
        read_on = HARM_VIEW[f] if look == "50M" else DECIDING[f]
        verdict = classify(look, stats[read_on])
        outcome, net, near = verdict.outcome, None, ()
        if f in NET:
            cost_view, gain_view = NET[f]
            if look == "100M":
                outcome, net, near = net_rule(f, verdict, stats[cost_view], stats[gain_view])
            else:  # REPORTED at the earlier looks: the same sum, deciding nothing
                net = {"cost_view": cost_view, "gain_view": gain_view, "cost_pp": stats[cost_view].estimate,
                       "gain_pp": stats[gain_view].estimate,
                       "net_pp": stats[cost_view].estimate + stats[gain_view].estimate, "passes": None}
        rows, cols, h = factor_matrix(design, blended[read_on], f)
        out["factors"][f] = {"outcome": outcome, "class_before_net_rule": verdict.outcome, "read_on": read_on,
                             "net": net, "confirming_contrast": CONFIRMING[f],
                             "at_boundary": list(verdict.at_boundary) + list(near), "rows": list(rows),
                             "cols": list(cols), "h_pp": [list(line) for line in h],
                             "views": {v: s.as_dict() for v, s in stats.items()}}
    inter = {v: interaction_stat(design, blended[v], view=v, floor_pp=floor_pp, level=level)
             for v in (N_VIEW, CLEAN_VIEW)}
    cls, near_i = interaction_class(inter[N_VIEW])
    out["interaction"] = {"class": cls, "class_is": REPORTED, "at_boundary": list(near_i),
                          "views": {v: s.as_dict() for v, s in inter.items()}}
    out["simple_effects"] = {f"{a}_vs_{b}": {v: simple_effect(design, blended[v], a, b, view=v, floor_pp=floor_pp,
                                                                level=level).as_dict()
                                              for v in (N_VIEW, CLEAN_VIEW)} for a, b in SIMPLE_EFFECTS}
    return out


def read_survivor(design: PairDesign, by_panel: Mapping[str, Scores], look: str, *, floor_pp: float = FLOOR_PP,
                  level: float = LEVEL) -> Dict[str, Any]:
    """A look AFTER a STOP-HARM, JSON-ready: the surviving factor's ONE simple effect (its ON arm against the control)
    on every view, each with its ``2 (n - 1)``-df interval. REPORTED: ``outcome`` is always :data:`NO_CLASS`, whatever
    the interval, and nothing here is compared with a threshold. A factor of :data:`NET` also reports its sum."""
    if look not in LOOKS or look == "50M":
        raise CrossInputError(f"a survivor look is one of {[x for x in LOOKS if x != '50M']}, got {look!r}")
    on, off = design.arms
    factor = design.survivor
    stats = {v: simple_effect(design, blend(design, by_panel, v), on, off, view=v, floor_pp=floor_pp, level=level)
             for v in VIEWS}
    net = None
    if factor in NET:
        cost_view, gain_view = NET[factor]
        net = {"cost_view": cost_view, "gain_view": gain_view, "cost_pp": stats[cost_view].estimate,
               "gain_pp": stats[gain_view].estimate,
               "net_pp": stats[cost_view].estimate + stats[gain_view].estimate, "passes": None}
    return {"look": look, "survivor": factor, "stopped": next(f for f in FACTORS if f != factor),
            "n_per_arm": design.n, "runs": {a: list(design.runs[a]) for a in (off, on)}, "floor_pp": floor_pp,
            "level": level, "outcome": NO_CLASS, "class_is": REPORTED, "to_owner": True,
            "effect": f"{on}_vs_{off}", "net": net, "views": {v: s.as_dict() for v, s in stats.items()}}


# ------------------------------------------------------------------------------------- scores from the ledger
def scores_from_cells(cells: Sequence[Any], run_of: Mapping[str, str], design: AnyDesign, pairs: int
                      ) -> Tuple[Optional[Dict[Cell, float]], Tuple[str, ...]]:
    """One panel's ``{(player run, opponent run): pp}`` from ``eval_ledger.cells`` Cells (``.player`` / ``.opponent``
    are checkpoint sha256s, mapped to runs by ``run_of``). ``(None, reasons)`` when any cell of the design is missing,
    listed twice, played from the wrong side, not OK by the ledger's own rule (more than 25 % of its games aborted),
    holds more than 25 % DRAWN games (:data:`DRAW_LIMIT`: the head-to-head's timeouts), does not hold EXACTLY the
    registered ``pairs`` completed pairs (fewer is short; more is games added after the plan), or when a cell outside
    the design is present: INCONCLUSIVE, never a class."""
    want = design.cells()
    by: Dict[Cell, Any] = {}
    reasons: List[str] = []
    for c in cells:
        p, o = run_of.get(c.player), run_of.get(c.opponent)
        if p is None or o is None:
            reasons.append(f"a cell names a checkpoint outside the look ({str(c.player)[:12]} vs "
                           f"{str(c.opponent)[:12]})")
            continue
        if (p, o) in by:
            reasons.append(f"cell {p} vs {o} appears twice")
        by[(p, o)] = c
    for k in sorted(set(by) - set(want)):
        reasons.append(f"cell {k[0]} vs {k[1]} is outside the design (or stored from the wrong side)")
    scores: Dict[Cell, float] = {}
    for k in want:
        c = by.get(k)
        if c is None:
            reasons.append(f"missing cell {k[0]} vs {k[1]}")
            continue
        if c.verdict != "OK":
            reasons.extend(f"cell {k[0]} vs {k[1]}: {x}" for x in c.reasons)
        if c.n_pairs != pairs:
            reasons.append(f"cell {k[0]} vs {k[1]}: {c.n_pairs} completed pairs "
                           f"{'<' if c.n_pairs < pairs else '>'} the registered {pairs}")
        if DRAW_LIMIT[1] * int(c.d) > DRAW_LIMIT[0] * int(c.games):
            reasons.append(f"cell {k[0]} vs {k[1]}: {c.d} of {c.games} games drawn (the turn limit) > 25 %")
        if c.score is None or not math.isfinite(float(c.score)):
            reasons.append(f"cell {k[0]} vs {k[1]}: no score")
            continue
        scores[k] = 100.0 * float(c.score)
    return (None, tuple(reasons)) if reasons else (scores, ())
