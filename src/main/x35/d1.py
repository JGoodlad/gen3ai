"""D1's REGISTERED readout (X38: the fork of P* that widens the training pool, ``--team-mix <forum corpus>:0.5``),
``designs/endstate/design_diversity_teams_opponents.md`` §4.2 and §5.6. Pure functions; no game is played here.
``python -m main.x35.readout decide-d1 <read.json>`` / ``self-check`` (the D1 suite of :mod:`main.x35.selfcheck`).

D1 is an X35 arm read against X35's shared control C0 (same P*, pin and seeds 3501-3503, then 3504-3505 at look 2).
Its primary is NOT X35's R1 (R1 scores pool teams only).

PRIMARY H -- the gain on the UNFAMILIAR cells. Each seed final plays 7 role-balanced 20-game units per panel against
``metamon:Kakuna`` greedy (``main.x35.kakuna --teamsets unfamiliar``; unit seeds 0-6 shared across arms) on the
symmetric cells ``heldout72_hpfix``, ``forum72`` and ``p72`` (:data:`PANELS`). The success is a WIN (ties are
non-wins, the anchor read's ``win_rate`` and G-K's convention). EXACTLY HOW IT IS POOLED, equal weight per panel:

* per arm a and panel k, the counts are pooled over the arm's seed finals: p_ak = x_ak / n_ak;
* the arm's score is the EQUAL-WEIGHT MEAN over the K = 3 panels, p̄_a = (1/K) Σ_k p_ak (NOT the pooled proportion
  Σx / Σn, which would weight a panel by its counted games);
* its variance is the stratified binomial variance V_a = (1/K²) Σ_k p_ak (1 − p_ak) / n_ak (the panels are fixed
  strata: every final plays a fixed number of games on each), and its EFFECTIVE size n_eff,a = p̄_a (1 − p̄_a) / V_a
  (Kish's);
* each arm's interval is Wilson's at (p̄_a, n_eff,a), z = :data:`Z` (``main.anchors.results.Z95``), and Δ_H's 95 %
  interval is Newcombe's method-10 hybrid of the two (``main.anchors.results.newcombe``'s formula, which this IS
  exactly when the panels' n and p are equal: then n_eff = Σn and p̄ = Σx / Σn -- checked by the self-check, as is
  the K = 1 case against Newcombe's published 1998 table, and the coverage by simulation);
* **WIDENED FOR SEED-TO-SEED SPREAD (AM-5, orchestrator 2026-10-19, before any D1 data):** the pooled counts treat
  the seeds as one binomial sample, so the interval hides the between-seed variance (review F-RV-5: 8-21 % false
  GAIN at a seed SD of 2-4 pp). Each Newcombe half-width is multiplied by √max(1, φ̂), where φ̂ is the
  OVERDISPERSION (:func:`overdispersion`): per arm, the between-seed variance of the per-seed equal-weight panel
  means (ddof 1) over its binomial expectation, the mean over the arm's seeds of (1/K²) Σ_k p_ik (1 − p_ik) / n_ik
  with each seed's own cell; φ̂ is the mean of the two arms' ratios (df 2(n − 1), recorded). Every class below
  reads the WIDENED interval; the unwidened one is recorded beside it.

Classes, in this order, on Δ_H in pp (the per-seed sign rule needs ≥ :data:`SIGN_MIN` [look] of the look's seeds
with Δ_i > 0, where Δ_i = ON seed i's equal-weight panel mean − OFF seed i's, seeds paired by number):

* **GAIN** iff the lower bound > 0 AND the sign rule holds;
* **LOSS** iff the upper bound < 0;
* **NULL** iff the interval lies inside [−3.0, +3.0];
* else **UNDECIDED**: at look 1 with Δ̂ > 0 it CONTINUEs to look 2 (5 seeds), which is final.
* RULE 8: a bound within :data:`RULE8_EPS` (1e-9) of ANY of the boundaries 0, −3.0, +3.0 makes the class UNDECIDED;
  a Δ_i or Δ̂ within 1e-9 of 0 is not "> 0".

CO-PRIMARY P -- no loss worse than −2.0 pp on SAMPLE-TEAM ground, IN-HOUSE. S(final) is X35's population score
(:func:`main.x35.readout.population_score`) of each seed final against X35's FROZEN reference population
(``reference_population_production.json``), mirrored pairs on the REGISTERED ``home72`` panel
(``main.h2h play-many --panel home72``, B-D7). Δ_P = mean ON S − mean OFF S over seed finals, Welch's two-sided
97.5 % interval (:func:`main.x35.readout.welch_interval`). **NON-INFERIOR** iff its lower bound > −2.0 pp (a lower
bound within 1e-9 of −2.0 is NOT non-inferior, recorded as ``near``); otherwise inferiority is NOT EXCLUDED
(the verdict table's "P inferior").

GUARDS: X35's G-F, G-E and G-C (:mod:`main.x35.readout`; D1's table names these three; G-K's Kakuna home72 / away
read is REPORTED for D1, with no rule).

VERDICT. ADOPT-CANDIDATE iff H is GAIN, P is NON-INFERIOR and every guard PASSes. TRADE iff H is GAIN and P's
inferiority is not excluded (the EXCHANGE RATE is reported: held-out pp gained per in-pool pp lost, Δ̂_H / −Δ̂_P,
or none when Δ̂_P ≥ 0 and the bar fails on the interval's width), or H is GAIN and a guard FAILs (X35's convention).
INCONCLUSIVE: an input is missing or invalid where the verdict needs it. CONTINUE: look 1 sent H to look 2.
Otherwise H's class (NULL / LOSS / UNDECIDED). Nothing is adopted by this rule alone.

DECLARED HERE, BEFORE ANY REAL READ (choices §4.2 leaves open; each a FINDING of this build):
* the equal-weight pooling, its variance and the effective-size Wilson / Newcombe above;
* the sign rule at look 2: §4.2 registers "≥ 2 of 3 seeds"; look 2's 5 seeds use a strict MAJORITY, 3 of 5;
* H's confidence is 95 % at BOTH looks, as §4.2 registers it (no Bonferroni split, unlike R1's 97.5 % per look);
* the floor: each arm must count ≥ 140·seeds − 20 games per panel (at most ONE uncounted 20-game unit per arm per
  panel, G-K's 400-of-420 at 3 seeds), and every (seed final, panel) cell > 0 games; else INCONCLUSIVE;
* zero variance on an arm (every p_ak 0 or 1) leaves the effective size undefined: INCONCLUSIVE;
* P's seed sets must equal H's, and both sides' must be equal and of the look's size (as R1's);
* each panel's ON − OFF row is REPORTED with its role for this arm (:data:`PANEL_ROLE`: the forum v2 corpus makes
  ``forum72`` the arm's own population on unseen teams); no rule reads the per-panel rows;
* AM-5's φ̂ never NARROWS the interval (φ̂ < 1 reads as 1); an arm whose binomial expectation is 0 (every per-seed
  cell at 0 or 1) leaves φ̂ undefined: INCONCLUSIVE; φ̂'s df is reported, no t / F quantile is applied (the
  amendment scales the half-widths, as the review's simulation ``review_2026-10-19/sim_d1_od.py`` does);
* with the widening, a look-1 GAIN at 3 seeds realistically needs the seeds to AGREE: the sign rule rarely binds
  any more (a lower bound > 0 with 1 of 3 seeds up is all but unreachable once the spread widens the interval), and
  D1's ADOPT-CANDIDATE realistically needs look 2 because of P (review F-RV-5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.training import eval_ledger as L
from main.x35 import readout as R

#: The PRIMARY H panels, in the registered order (``heldout_panels.json`` names).
PANELS = ("heldout72_hpfix", "forum72", "p72")
#: What each panel IS for X38's arm (REPORTED beside the per-panel rows; no rule reads it). Amendment of 2026-10-19:
#: the widening corpus is drawn from the Smogon FORUM v2 collection (not ``hl_05_26``), so ``forum72`` is the arm's
#: own population on unseen teams, and the other two are populations it does not train on. The pooled primary is
#: unchanged.
PANEL_ROLE: Dict[str, str] = {
    "heldout72_hpfix": "a population X38 does not train on (ladder replays, HP repaired)",
    "forum72": "X38's own population (Smogon forum), unseen teams: a gain here alone = 'learned the population'",
    "p72": "a population X38 does not train on (procedural, MATERIALLY WEAKER, F-DV-20)",
}
#: The same rule reads X41 / D3 (set mutation, ``x41_d3_setmut_2026-10-25/PREREG.md``): it trains on mutated POOL
#: teams, so no panel is its own population. Each arm's roles, by its read label.
PANEL_ROLES: Dict[str, Dict[str, str]] = {
    "X38": PANEL_ROLE,
    "X41": {"heldout72_hpfix": "a population X41 does not train on (ladder replays, HP repaired)",
            "forum72": "a population X41 does not train on (Smogon forum, human-built)",
            "p72": "a population X41 does not train on (procedural Smogon usage, MATERIALLY WEAKER, F-DV-20)"},
}
#: The NULL band of H, pp.
H_BAND_PP = 3.0
#: P's non-inferiority bar, pp.
P_BAR_PP = -2.0
#: Newcombe / Wilson z (the anchors' own, ``main.anchors.results.Z95``).
Z = 1.959963985
RULE8_EPS = R.RULE8_EPS
LOOK_SEEDS = R.LOOK_SEEDS
#: Seeds with Δ_i > 0 that H's GAIN needs at each look (look 1 registered; look 2 DECLARED, module docs).
SIGN_MIN: Dict[int, int] = {1: 2, 2: 3}
#: The plan: 7 units of 20 games per (seed final, panel).
UNIT_GAMES = 20
UNITS_PER_SEED = 7
P_CONFIDENCE = R.R1_CONFIDENCE
D1_GUARDS = ("G-F", "G-E", "G-C")

GAIN, NULL, LOSS, UNDECIDED = R.GAIN, R.NULL, R.LOSS, R.UNDECIDED
CONTINUE, INCONCLUSIVE, ADOPT, TRADE = R.CONTINUE, R.INCONCLUSIVE, R.ADOPT, R.TRADE
NON_INFERIOR, NOT_EXCLUDED = "NON-INFERIOR", "INFERIORITY-NOT-EXCLUDED"


def min_games(seeds: int) -> int:
    """The per-arm, per-panel counted-games floor at ``seeds`` seed finals (module docs)."""
    return UNIT_GAMES * UNITS_PER_SEED * int(seeds) - UNIT_GAMES


def wilson_pn(p: float, n: float, z: float = Z) -> Tuple[float, float]:
    """Wilson's score interval at a (possibly effective, non-integer) size ``n`` -- ``main.anchors.results.wilson``'s
    formula with ``k / n`` replaced by ``p``."""
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class StratScore:
    """One arm's equal-weight panel mean, its stratified variance and effective size."""

    per_panel: Tuple[Tuple[str, int, int], ...]     # (panel, wins, n)
    mean: float
    var: float
    n_eff: float
    ci: Tuple[float, float]

    def as_dict(self) -> Dict[str, Any]:
        return {"per_panel": {k: [x, n] for k, x, n in self.per_panel}, "mean": self.mean, "var": self.var,
                "n_eff": self.n_eff, "wilson95": list(self.ci)}


def strat_score(counts: Mapping[str, Sequence[int]], panels: Sequence[str] = PANELS, z: float = Z) -> StratScore:
    """``counts`` = ``{panel: (wins, n)}`` pooled over one arm's seed finals (module docs)."""
    rows: List[Tuple[str, int, int]] = []
    for k in panels:
        if k not in counts:
            raise R.ReadoutInputError(f"panel {k}: no counts")
        x, n = int(counts[k][0]), int(counts[k][1])
        if n <= 0 or not 0 <= x <= n:
            raise R.ReadoutInputError(f"panel {k}: invalid counts {x}/{n}")
        rows.append((k, x, n))
    K = len(rows)
    ps = [x / n for _, x, n in rows]
    mean = math.fsum(ps) / K
    var = math.fsum(p * (1.0 - p) / n for p, (_, _, n) in zip(ps, rows)) / (K * K)
    if not var > 0.0 or not 0.0 < mean < 1.0:
        raise R.ReadoutInputError("zero variance (every panel's score is 0 or 1): the effective size is undefined")
    n_eff = mean * (1.0 - mean) / var
    return StratScore(tuple(rows), mean, var, n_eff, wilson_pn(mean, n_eff, z))


def panel_mean(counts: Mapping[str, Sequence[int]], panels: Sequence[str] = PANELS) -> float:
    """The equal-weight panel mean of one seed final's ``{panel: (wins, n)}`` (the sign rule's per-seed score)."""
    return math.fsum(int(counts[k][0]) / int(counts[k][1]) for k in panels) / len(panels)


def newcombe_strat(on: StratScore, off: StratScore) -> Tuple[float, float, float]:
    """``(Δ, lo, hi)`` -- Newcombe's method-10 hybrid of the two arms' Wilson intervals (proportions, not pp)."""
    d = on.mean - off.mean
    lo = d - math.sqrt((on.mean - on.ci[0]) ** 2 + (off.ci[1] - off.mean) ** 2)
    hi = d + math.sqrt((on.ci[1] - on.mean) ** 2 + (off.mean - off.ci[0]) ** 2)
    return d, max(-1.0, lo), min(1.0, hi)


@dataclass(frozen=True)
class Overdispersion:
    """AM-5's φ̂ and its parts (module docs)."""

    phi: float
    ratios: Tuple[float, float]          # (ON, OFF): between-seed variance / binomial expectation
    df: int
    factor: float                        # √max(1, φ̂), the half-width multiplier

    def as_dict(self) -> Dict[str, Any]:
        return {"phi_hat": self.phi, "ratio_on": self.ratios[0], "ratio_off": self.ratios[1], "df": self.df,
                "half_width_factor": self.factor}


def _arm_ratio(cells: Mapping[str, Mapping[str, Sequence[int]]], panels: Sequence[str]) -> float:
    K = len(panels)
    means: List[float] = []
    exps: List[float] = []
    for seed in sorted(cells):
        ps = [(int(cells[seed][k][0]) / int(cells[seed][k][1]), int(cells[seed][k][1])) for k in panels]
        means.append(math.fsum(p for p, _ in ps) / K)
        exps.append(math.fsum(p * (1.0 - p) / n for p, n in ps) / (K * K))
    e = math.fsum(exps) / len(exps)
    if not e > 0.0:
        raise R.ReadoutInputError("AM-5: an arm's binomial expectation is 0 (every per-seed cell at 0 or 1): "
                                  "the overdispersion is undefined")
    m = math.fsum(means) / len(means)
    return math.fsum((x - m) ** 2 for x in means) / (len(means) - 1) / e


def overdispersion(on: Mapping[str, Mapping[str, Sequence[int]]], off: Mapping[str, Mapping[str, Sequence[int]]],
                   panels: Sequence[str] = PANELS) -> Overdispersion:
    """AM-5's φ̂ from each side's ``{seed: {panel: (wins, n)}}`` (module docs; >= 2 seeds per side)."""
    if len(on) < 2 or len(off) < 2:
        raise R.ReadoutInputError("AM-5: the overdispersion needs >= 2 seeds per side")
    r_on, r_off = _arm_ratio(on, panels), _arm_ratio(off, panels)
    phi = (r_on + r_off) / 2.0
    return Overdispersion(phi, (r_on, r_off), (len(on) - 1) + (len(off) - 1), math.sqrt(max(1.0, phi)))


def widen(d: float, lo: float, hi: float, factor: float) -> Tuple[float, float]:
    """Each half-width of ``(lo, d, hi)`` times ``factor``, clipped to [−1, 1] as :func:`newcombe_strat` clips."""
    return max(-1.0, d - (d - lo) * factor), min(1.0, d + (hi - d) * factor)


def classify_h(ci_pp: Tuple[float, float], sign_ok: bool) -> Tuple[str, Tuple[str, ...]]:
    """H's class of an interval (pp) and the sign rule's outcome; every boundary a bound touched (rule 8 ⇒
    UNDECIDED)."""
    lo, hi = ci_pp
    near = tuple(f"{name} {x!r} within {RULE8_EPS} of {b:+}" for name, x in (("lower", lo), ("upper", hi))
                 for b in (-H_BAND_PP, 0.0, H_BAND_PP) if abs(x - b) <= RULE8_EPS)
    if near:
        return UNDECIDED, near
    if lo > 0.0 and sign_ok:
        return GAIN, ()
    if hi < 0.0:
        return LOSS, ()
    if lo > -H_BAND_PP and hi < H_BAND_PP:
        return NULL, ()
    return UNDECIDED, ()


def positive(x: float) -> bool:
    """``x > 0`` under rule 8 (within 1e-9 of 0 is not positive)."""
    return x > 0.0 and abs(x) > RULE8_EPS


@dataclass(frozen=True)
class HDecision:
    look: int
    cls: Optional[str]
    outcome: str
    final: bool
    delta_pp: Optional[float]
    ci_pp: Optional[Tuple[float, float]]
    on: Optional[StratScore]
    off: Optional[StratScore]
    seed_deltas_pp: Tuple[Tuple[str, float], ...]
    positive_seeds: int
    sign_ok: bool
    per_panel: Dict[str, Any]
    near: Tuple[str, ...]
    reasons: Tuple[str, ...]
    od: Optional[Overdispersion] = None
    ci_unwidened_pp: Optional[Tuple[float, float]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"look": self.look, "class": self.cls, "outcome": self.outcome, "final": self.final,
                "delta_pp": self.delta_pp, "ci95_pp": list(self.ci_pp) if self.ci_pp else None,
                "ci95_unwidened_pp": list(self.ci_unwidened_pp) if self.ci_unwidened_pp else None,
                "overdispersion": self.od.as_dict() if self.od else None,
                "on": self.on.as_dict() if self.on else None, "off": self.off.as_dict() if self.off else None,
                "seed_deltas_pp": dict(self.seed_deltas_pp), "positive_seeds": self.positive_seeds,
                "sign_rule": f">= {SIGN_MIN.get(self.look)} of {LOOK_SEEDS.get(self.look)}", "sign_ok": self.sign_ok,
                "per_panel_reported": self.per_panel, "near_boundary": list(self.near), "reasons": list(self.reasons)}


Counts = Mapping[str, Mapping[str, Sequence[int]]]       # {seed: {panel: (wins, n)}}


def _check_seeds(look: int, on: Mapping[str, Any], off: Mapping[str, Any], what: str) -> List[str]:
    reasons: List[str] = []
    if set(on) != set(off):
        reasons.append(f"{what}: ON seeds {sorted(on)} != OFF seeds {sorted(off)} (they share seed i)")
    want = LOOK_SEEDS[look]
    if len(on) != want or len(off) != want:
        reasons.append(f"{what}: {len(on)} ON / {len(off)} OFF seed finals; look {look} registers {want} per side")
    return reasons


def d1_primary(look: int, on: Counts, off: Counts, *, panels: Sequence[str] = PANELS,
               roles: Optional[Mapping[str, str]] = None) -> HDecision:
    """PRIMARY H at ``look`` from each side's ``{seed: {panel: (wins, n)}}`` (module docs); ``roles`` (reported
    only) default to X38's :data:`PANEL_ROLE`."""
    from main.anchors.results import newcombe

    if look not in LOOK_SEEDS:
        raise ValueError(f"look {look} is not one of the registered looks {sorted(LOOK_SEEDS)}")
    reasons = _check_seeds(look, on, off, "H")
    pooled: Dict[str, Dict[str, List[int]]] = {"on": {k: [0, 0] for k in panels}, "off": {k: [0, 0] for k in panels}}
    for side, d in (("on", on), ("off", off)):
        for seed, cells in sorted(d.items()):
            for k in panels:
                c = cells.get(k)
                if c is None or len(c) != 2:
                    reasons.append(f"{side} seed {seed}: no {k} counts")
                    continue
                x, n = int(c[0]), int(c[1])
                if n <= 0 or not 0 <= x <= n:
                    reasons.append(f"{side} seed {seed} {k}: invalid counts {x}/{n}")
                    continue
                pooled[side][k][0] += x
                pooled[side][k][1] += n
            extra = sorted(set(cells) - set(panels))
            if extra:
                reasons.append(f"{side} seed {seed}: panels {extra} are not D1's {list(panels)}")
        floor = min_games(len(d))
        for k in panels:
            if pooled[side][k][1] < floor:
                reasons.append(f"{side} {k}: {pooled[side][k][1]} counted games < {floor} (at most one uncounted unit)")
    if reasons:
        return HDecision(look, None, INCONCLUSIVE, True, None, None, None, None, (), 0, False, {}, (), tuple(reasons))
    try:
        s_on, s_off = strat_score(pooled["on"], panels), strat_score(pooled["off"], panels)
        seed_d = [(seed, 100.0 * (panel_mean(on[seed], panels) - panel_mean(off[seed], panels)))
                  for seed in sorted(on)]
        od = overdispersion(on, off, panels)
    except R.ReadoutInputError as exc:
        return HDecision(look, None, INCONCLUSIVE, True, None, None, None, None, (), 0, False, {}, (), (str(exc),))
    d, lo0, hi0 = newcombe_strat(s_on, s_off)
    lo, hi = widen(d, lo0, hi0, od.factor)                      # AM-5
    ci = (100.0 * lo, 100.0 * hi)
    n_pos = sum(positive(x) for _, x in seed_d)
    sign_ok = n_pos >= SIGN_MIN[look]
    per_panel = {}
    for k in panels:
        (x1, n1), (x2, n2) = pooled["on"][k], pooled["off"][k]
        pd, plo, phi = newcombe(x1, n1, x2, n2)
        per_panel[k] = {"on": [x1, n1], "off": [x2, n2], "delta_pp": 100.0 * pd, "ci95_pp": [100.0 * plo, 100.0 * phi],
                        "role": (PANEL_ROLE if roles is None else roles).get(k, "")}
    cls, near = classify_h(ci, sign_ok)
    common = dict(delta_pp=100.0 * d, ci_pp=ci, on=s_on, off=s_off, seed_deltas_pp=tuple(seed_d),
                  positive_seeds=n_pos, sign_ok=sign_ok, per_panel=per_panel, reasons=(), od=od,
                  ci_unwidened_pp=(100.0 * lo0, 100.0 * hi0))
    if cls == UNDECIDED and look == 1:
        pos = positive(100.0 * d)
        nn = near + (() if pos or abs(100.0 * d) > RULE8_EPS else
                     (f"Δ̂ {100.0 * d!r} within {RULE8_EPS} of 0 (the look-2 rule)",))
        return HDecision(look, cls, CONTINUE if pos else cls, not pos, near=nn, **common)  # type: ignore[arg-type]
    return HDecision(look, cls, cls, True, near=near, **common)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PDecision:
    look: int
    cls: Optional[str]
    stat: Optional[R.WelchStat]
    near: Tuple[str, ...]
    reasons: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {"look": self.look, "class": self.cls, "stat": self.stat.as_dict() if self.stat else None,
                "bar_pp": P_BAR_PP, "near_boundary": list(self.near), "reasons": list(self.reasons)}


def d1_coprimary(look: int, on: Mapping[str, Optional[float]], off: Mapping[str, Optional[float]], *,
                 inconclusive: Sequence[str] = ()) -> PDecision:
    """CO-PRIMARY P at ``look`` from ``{seed: S_pp}`` per side (``None`` = that final's S is INCONCLUSIVE)."""
    if look not in LOOK_SEEDS:
        raise ValueError(f"look {look} is not one of the registered looks {sorted(LOOK_SEEDS)}")
    reasons = list(inconclusive) + _check_seeds(look, on, off, "P")
    reasons.extend(f"P {side} seed {s}: no population score" for side, d in (("ON", on), ("OFF", off))
                   for s, v in sorted(d.items()) if v is None)
    stat: Optional[R.WelchStat] = None
    if not reasons:
        seeds = sorted(on)
        try:
            stat = R.welch_interval([float(on[s]) for s in seeds], [float(off[s]) for s in seeds],  # type: ignore[arg-type]
                                    confidence=P_CONFIDENCE)
        except R.ReadoutInputError as exc:
            reasons.append(str(exc))
    if reasons or stat is None:
        return PDecision(look, None, None, (), tuple(reasons))
    ok, near = R._clears(stat.ci[0], P_BAR_PP, above=True)
    return PDecision(look, NON_INFERIOR if ok else NOT_EXCLUDED, stat,
                     (f"lower {stat.ci[0]!r} within {RULE8_EPS} of {P_BAR_PP}",) if near else (), ())


def exchange_rate(h: HDecision, p: PDecision) -> Dict[str, Any]:
    """Held-out pp gained per in-pool pp lost (TRADE's report): Δ̂_H / −Δ̂_P when Δ̂_P < 0, else none."""
    if h.delta_pp is None or p.stat is None:
        return {"rate": None, "why": "H or P not read"}
    dp = p.stat.delta
    if dp < 0.0 and abs(dp) > RULE8_EPS:
        return {"rate": h.delta_pp / -dp, "delta_h_pp": h.delta_pp, "delta_p_pp": dp, "p_lower_pp": p.stat.ci[0]}
    return {"rate": None, "delta_h_pp": h.delta_pp, "delta_p_pp": dp, "p_lower_pp": p.stat.ci[0],
            "why": "no point loss on P (Δ̂_P >= 0): the −2.0 pp bar fails on the interval's width"}


@dataclass(frozen=True)
class D1Verdict:
    arm: str
    look: int
    verdict: str
    h: HDecision
    p: PDecision
    guards: Tuple[R.GuardResult, ...]
    reasons: Tuple[str, ...]
    reported: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"arm": self.arm, "look": self.look, "verdict": self.verdict, "h": self.h.as_dict(),
                "p": self.p.as_dict(), "guards": {g.name: g.as_dict() for g in self.guards},
                "reasons": list(self.reasons), "reported": self.reported}


def d1_verdict(arm: str, h: HDecision, p: PDecision, guards: Sequence[R.GuardResult], *,
               reported: Optional[Dict[str, Any]] = None) -> D1Verdict:
    """The pre-registered D1 verdict (module docs)."""
    if h.look != p.look:
        raise ValueError(f"H is look {h.look}, P is look {p.look}")
    by = {g.name: g for g in guards}
    gs = tuple(by.get(n) or R.GuardResult(n, INCONCLUSIVE, reasons=("not read",)) for n in D1_GUARDS)
    rep = dict(reported or {})

    def out(v: str, *why: str) -> D1Verdict:
        return D1Verdict(arm, h.look, v, h, p, gs, tuple(why), rep)

    if h.outcome == INCONCLUSIVE:
        return out(INCONCLUSIVE, "H is INCONCLUSIVE", *h.reasons)
    if h.outcome == CONTINUE:
        return out(CONTINUE, f"H UNDECIDED with Δ̂ > 0 at look {h.look}: go to look 2")
    if h.cls != GAIN:
        return out(str(h.cls), "H is not a GAIN")
    if p.cls is None:
        return out(INCONCLUSIVE, "H GAIN, but P is INCONCLUSIVE", *p.reasons)
    open_ = [g.name for g in gs if g.status == INCONCLUSIVE]
    if open_:
        return out(INCONCLUSIVE, f"H GAIN, but guard(s) {open_} INCONCLUSIVE")
    failed = [g.name for g in gs if g.status == R.FAIL]
    why = []
    if p.cls != NON_INFERIOR:
        why.append(f"P: inferiority not excluded (lower {p.stat.ci[0] if p.stat else None!r} <= {P_BAR_PP})")
    if failed:
        why.append(f"guard(s) {failed} FAIL")
    if why:
        rep["exchange_rate"] = exchange_rate(h, p)
        return out(TRADE, "H GAIN, but " + "; ".join(why))
    return out(ADOPT, "H GAIN, P NON-INFERIOR; every guard PASSES")


def read_d1(doc: Mapping[str, Any]) -> D1Verdict:
    """The whole D1 decision from one JSON-shaped read (``decide-d1``'s input):

    ``{"arm": "X38", "look": 1, "h": {"on": {seed: {panel: [wins, n]}}, "off": {...}},
    "p": {"on": {seed: S_pp}, "off": {...}, "inconclusive": [...]}, "r2": {"on": [w, n], "off": [w, n]},
    "g_f": {"h_pp": [[...]]}, "g_c": <the main.x35.gc record over H's seeds>,
    "kakuna_in_pool": {"home72": {"on": [w, n], "off": [w, n]}, "away": {...}}, "dose": {"on": [...], "off": [...]}}``

    ``h`` is ``python -m main.x35.kakuna d1h``'s output (its ``on`` / ``off``); P's seed sets must equal H's."""
    arm, look = str(doc.get("arm", "X38")), int(doc["look"])
    hd = doc.get("h") or {}
    h = d1_primary(look, dict(hd.get("on") or {}), dict(hd.get("off") or {}), roles=PANEL_ROLES.get(arm, PANEL_ROLE))
    pd = doc.get("p") or {}
    p_on, p_off = dict(pd.get("on") or {}), dict(pd.get("off") or {})
    extra = []
    if set(p_on) != set(hd.get("on") or {}) or set(p_off) != set(hd.get("off") or {}):
        extra.append(f"P's seeds {sorted(p_on)} / {sorted(p_off)} are not H's")
    p = d1_coprimary(look, p_on, p_off, inconclusive=tuple(pd.get("inconclusive") or ()) + tuple(extra))
    gap: Optional[R.GapDiff] = None
    r2d = doc.get("r2")
    gap_why = None
    if r2d:
        try:
            gap = R.gap_diff(r2d["on"], r2d["off"])
        except (R.ReadoutInputError, KeyError, TypeError, IndexError) as exc:
            gap_why = f"G-E input: {exc}"
    g_e = R.guard_e(gap) if gap_why is None else R.GuardResult("G-E", INCONCLUSIVE, reasons=(gap_why,))
    guards = [R.guard_f((doc.get("g_f") or {}).get("h_pp"), look), g_e,
              R.guard_c_record(doc.get("g_c"), list(hd.get("on") or {}), list(hd.get("off") or {}))]
    dose = doc.get("dose") or {}
    rep: Dict[str, Any] = {"dose": R.dose_confound(dose.get("on"), dose.get("off"))}
    if doc.get("kakuna_in_pool") is not None:
        rep["kakuna_in_pool"] = R.guard_k(doc["kakuna_in_pool"]).detail
    return d1_verdict(arm, h, p, guards, reported=rep)


# ------------------------------------------------------------------------------------------------ P's input
def refpop_members(path: str, *, verify: bool = True) -> Dict[str, str]:
    """``{label: sha256}`` of a FROZEN reference population (``main.x35.refpop``); with ``verify`` every named file
    is re-hashed first (a moved member is a refusal, never a different read)."""
    import json
    from pathlib import Path

    from main.x35 import refpop as RP

    if verify:
        probs = RP.verify(path)
        if probs:
            raise R.ReadoutInputError(f"{path}: the frozen R does not verify: {probs}")
    doc = json.loads(Path(path).read_text())
    return {str(m["label"]): str(m["sha256"]) for m in doc["members"]}


def p_input(root: Any, family: str, request: str, finals: Sequence[Mapping[str, Any]], on: str, off: str,
            members: Mapping[str, str], *, panel: str = "home72", min_pairs: int = R.MIN_PAIRS) -> Dict[str, Any]:
    """P's input for :func:`read_d1` from an eval ledger: each seed final's S against R (``members``) from the cells
    of ONE look (``request``) of a registered family, played ``main.h2h play-many --panel <panel> --purpose ab
    --family <family> --request <request>``. ``finals`` = ``[{label, arm, seed, sha256}]`` (the Kakuna runner's
    finals file, hashed). A row of the request NOT on the registered panel's team set makes the whole P
    INCONCLUSIVE (the panel is checked by content, through the regime's ``team_set``)."""
    from main.h2h import cross as CX
    from main.h2h import panel as PN

    want_ts = PN.panel_team_set_id(panel)
    got = _read_family(root, family)
    rows = [r for r in got.rows if (r.get("request") or {}).get("id") == request]
    off_panel = sorted({r["regime"]["team_set"] for r in rows if r["regime"].get("team_set") != want_ts})
    reasons: List[str] = []
    if not rows:
        reasons.append(f"request {request} of family {family}: no row")
    if off_panel:
        reasons.append(f"request {request}: rows on team set(s) {off_panel}, not panel {panel}'s {want_ts}")
    cells = CX.look_cells(root, family, request, "gen3_eval_protocol_v1_h2h")
    out: Dict[str, Any] = {"on": {}, "off": {}, "inconclusive": reasons, "per_final": {}, "panel": panel,
                           "team_set": want_ts}
    for f in finals:
        side = "on" if f["arm"] == on else "off" if f["arm"] == off else None
        if side is None:
            continue
        seed = str(f["seed"])
        if seed in out[side]:
            reasons.append(f"two {f['arm']} finals share seed {seed}")
            continue
        ps = R.population_score(cells, str(f["sha256"]), members, min_pairs=min_pairs)
        out[side][seed] = ps.s_pp
        out["per_final"][str(f["label"])] = ps.as_dict()
    return out


#: P's look read: an A/B family's rows at the h2h protocol (``main.h2h.cross.FAMILY_READ_OFF``'s terms), read only
#: to check every row of the look is on the registered panel's team set.
P_FAMILY_READ = L.ReaderDecl(
    name="main.x35.d1.p_family", purposes=frozenset({"ab"}),
    regime=L.RegimeFilter(protocol="gen3_eval_protocol_v1_h2h", play="greedy", opponent_play="greedy", mirrored=True),
    requests="family", selection="include", flags_ok=frozenset(), inference="across_runs",
    decision_kind="ab_verdict")


def _read_family(root: Any, family: str) -> Any:
    """The family's rows at the h2h protocol (:data:`P_FAMILY_READ`)."""
    return L.read(P_FAMILY_READ, root=root, family=family)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(prog="python -m main.x35.d1", description="D1's (X38's) readout inputs; the "
                                 "verdict is `python -m main.x35.readout decide-d1`")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("p-input", help="P's {on, off} population scores on the registered panel, from a ledger")
    p.add_argument("--ledger", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--refpop", required=True, help="the FROZEN reference_population_<side>.json")
    p.add_argument("--finals", required=True, help="JSON [{label, arm, seed, path}] (the Kakuna runner's)")
    p.add_argument("--on", required=True)
    p.add_argument("--off", required=True)
    p.add_argument("--panel", default="home72")
    a = ap.parse_args(argv)
    from main.h2h.play import file_sha256

    finals = [dict(f, sha256=file_sha256(f["path"])) for f in json.loads(Path(a.finals).read_text())]
    try:
        out = p_input(a.ledger, a.family, a.request, finals, a.on, a.off, refpop_members(a.refpop),
                      panel=a.panel)
    except R.ReadoutInputError as exc:
        print(f"[x35 d1] REFUSED: {exc}", file=sys.stderr)
        return 2
    json.dump(out, sys.stdout, indent=1, sort_keys=True, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
