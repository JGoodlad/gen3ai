"""The ACROSS-SEED inference for a purpose metric (§7.4, review M8): one value per run, a two-sample
t over seeds, judged against a boundary the CALLER passes (the stopping look's group-sequential
t-boundary: 5.761 / 2.683 / 1.874 at looks 1 / 2 / 3 — never full α at the crossing look; Hung, Wang &
O'Neill 2007; Tamhane, Mehta & Liu 2010).

    d̂ = improvement of the treatment arm (X5) over the control arm (blob), in the metric's BETTER
         direction: mean(control) − mean(treat) for a lower-is-better metric, the reverse otherwise;
    s²_p = ((n_t − 1)·s²_t + (n_c − 1)·s²_c) / (n_t + n_c − 2)       (pooled; s² with ddof = 1)
    t = (d̂ − margin) / √(s²_p · (1/n_t + 1/n_c))   on df = n_t + n_c − 2  (= 2(n − 1) at equal n)

CROSSED iff ``t − boundary ≥ 1e-9`` (rule 8: a t within 1e-9 of the boundary is NOT a crossing).
``margin`` defaults to 0 (superiority: "(1) improves past that boundary"). The one-sided p-value is
reported beside it, never the decision.

The inputs are the readers' JSON files (``python -m main.belief_roles read``) — FINISHED artifacts, so a
control group of BANKED blob reads (written before the X5 version break, when the reader still read a
blob checkpoint) compares with X5 reads made today; a comparison is REFUSED
when its runs were read on different banks, role sets, reader schemas or strata, when a checkpoint
appears twice, when a group mixes arms (or does not hold the arm its side declares), when either arm has
fewer than 2 runs, when a metric is missing in a run, or when both arms have zero variance.

**The adoption-gate metric** (``intent_logloss_conditional``, Amendment 3(b)) is scored on E_row = blob's
named set, so it carries one more refusal: every control (blob) read must be on its OWN set, and EVERY
treat (fixed_mass) read must reference EXACTLY the control group's blob checkpoints — all of them, no
other, no repeat (its value is the mean over those sets; §7.7(b)'s decision, mirroring the cross). So
every fixed_mass run is scored over the same collection of supports the blob arm is. The t itself is
unchanged: two-sample, unpaired.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from main.belief_roles.metrics import ADOPTION_GATE_METRIC, PER_RUN_DIRECTION

#: Rule 8 at the boundary.
BOUNDARY_EPS = 1e-9


class InferenceRefused(ValueError):
    """A typed refusal: the comparison is not well-defined (module docstring)."""


@dataclass(frozen=True)
class TResult:
    metric: str
    direction: str
    n_treat: int
    n_control: int
    mean_treat: float
    mean_control: float
    improvement: float
    margin: float
    se: float
    t: float
    df: int
    boundary: float
    crossed: bool
    p_one_sided: Optional[float]

    def to_json(self) -> dict:
        return dict(self.__dict__)


def _mean_var(x: Sequence[float]):
    n = len(x)
    m = sum(x) / n
    return m, sum((v - m) ** 2 for v in x) / (n - 1)


def two_sample_t(treat: Sequence[float], control: Sequence[float], *, direction: str,
                 boundary: float, margin: float = 0.0, metric: str = "metric") -> TResult:
    """The statistic of the module docstring on plain per-run values."""
    if direction not in ("lower", "higher"):
        raise InferenceRefused(f"direction must be 'lower' or 'higher', not {direction!r}")
    treat, control = [float(v) for v in treat], [float(v) for v in control]
    if len(treat) < 2 or len(control) < 2:
        raise InferenceRefused(f"{metric}: each arm needs >= 2 runs (got {len(treat)} / {len(control)})")
    if not all(math.isfinite(v) for v in treat + control):
        raise InferenceRefused(f"{metric}: a per-run value is not finite")
    mt, vt = _mean_var(treat)
    mc, vc = _mean_var(control)
    nt, nc = len(treat), len(control)
    df = nt + nc - 2
    sp2 = ((nt - 1) * vt + (nc - 1) * vc) / df
    if sp2 <= 0.0:
        raise InferenceRefused(f"{metric}: zero variance in both arms — t is undefined")
    se = math.sqrt(sp2 * (1.0 / nt + 1.0 / nc))
    d = (mc - mt) if direction == "lower" else (mt - mc)
    t = (d - float(margin)) / se
    try:
        from scipy.stats import t as student_t
        p: Optional[float] = float(student_t.sf(t, df))
    except ImportError:                                  # the decision never needs it
        p = None
    return TResult(metric=metric, direction=direction, n_treat=nt, n_control=nc, mean_treat=mt,
                   mean_control=mc, improvement=d, margin=float(margin), se=se, t=t, df=df,
                   boundary=float(boundary), crossed=(t - float(boundary)) >= BOUNDARY_EPS,
                   p_one_sided=p)


def _load(paths: Sequence[Path]) -> List[dict]:
    return [json.loads(Path(p).read_text()) for p in paths]


def compare_reads(treat_paths: Sequence[Path], control_paths: Sequence[Path], metric: str,
                  boundary: float, margin: float = 0.0, stratum: str = "on_pool",
                  treat_arm: str = "fixed_mass", control_arm: str = "blob") -> TResult:
    """:func:`two_sample_t` on the ``per_run`` values of the readers' JSON files, with every refusal of
    the module docstring. ``stratum`` ``on_pool`` reads ``per_run`` (the primary); ``off_pool`` reads
    ``per_run_off_pool``."""
    if metric not in PER_RUN_DIRECTION:
        raise InferenceRefused(f"unknown metric {metric!r} (known: {sorted(PER_RUN_DIRECTION)})")
    t_reads, c_reads = _load(treat_paths), _load(control_paths)
    allr = t_reads + c_reads
    for key in ("schema", "bank_sha256", "role_set_sha256"):
        vals = {r.get(key) for r in allr}
        if len(vals) != 1:
            raise InferenceRefused(f"the reads disagree on {key}: {sorted(map(str, vals))}")
    shas = [r["checkpoint"]["sha256"] for r in allr]
    if len(set(shas)) != len(shas):
        raise InferenceRefused("a checkpoint appears more than once across the two arms")
    for side, reads, arm in (("treat", t_reads, treat_arm), ("control", c_reads, control_arm)):
        arms = {r["arm"] for r in reads}
        if arms != {arm}:
            raise InferenceRefused(f"the {side} group holds arms {sorted(arms)}, not only {arm!r}")
    key = "per_run" if stratum == "on_pool" else "per_run_off_pool"
    if metric == ADOPTION_GATE_METRIC:
        check_set_pairing(t_reads, c_reads)

    def vals(reads: List[dict]) -> List[float]:
        out = []
        for r in reads:
            v = (r.get(key) or {}).get(metric)
            if v is None:
                raise InferenceRefused(f"{r['label']}: no {stratum} value for {metric}")
            out.append(float(v))
        return out

    return two_sample_t(vals(t_reads), vals(c_reads), direction=PER_RUN_DIRECTION[metric],
                        boundary=boundary, margin=margin, metric=metric)


def check_set_pairing(t_reads: List[dict], c_reads: List[dict]) -> None:
    """The conditional metric's E_row refusals (module docstring)."""
    c_shas = []
    for r in c_reads:
        ref = r.get("eset_reference") or {}
        if ref.get("mode") != "own" or ref.get("checkpoint_sha256") != r["checkpoint"]["sha256"]:
            raise InferenceRefused(f"{r['label']}: a control (blob) read must be scored on its OWN named "
                                   f"set (eset_reference {ref})")
        c_shas.append(r["checkpoint"]["sha256"])
    for r in t_reads:
        ref = r.get("eset_reference") or {}
        if ref.get("mode") != "all_blob_mean":
            raise InferenceRefused(f"{r['label']}: a treat read must be scored on EVERY blob run of the "
                                   f"look (read --reference; eset_reference mode {ref.get('mode')!r})")
        shas = [x.get("checkpoint_sha256") for x in ref.get("references") or []]
        if sorted(map(str, shas)) != sorted(c_shas):
            raise InferenceRefused(f"{r['label']}: its reference blob checkpoints are not EXACTLY the "
                                   "control group's (a missing, extra or repeated reference) — the arms "
                                   "would be scored over different collections of supports")


def summarize(results: Dict[str, TResult]) -> str:
    lines = []
    for m, r in results.items():
        tag = " [ADOPTION GATE, Amendment 3(b)]" if m == ADOPTION_GATE_METRIC else " [descriptive]"
        lines.append(f"{m}{tag}: improvement {r.improvement:+.5g} (treat {r.mean_treat:.5g} n={r.n_treat}, "
                     f"control {r.mean_control:.5g} n={r.n_control}), t = {r.t:.4f} on {r.df} df "
                     f"vs boundary {r.boundary} → {'CROSSED' if r.crossed else 'not crossed'}")
    return "\n".join(lines)
