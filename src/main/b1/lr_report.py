"""BOUNDARY-1's LEARNING-RATE REPORT beside a look: what step each run trained at, and where it landed. REPORTED.

    python -m main.b1.lr_report --look 50M                          # the twelve registered runs; exit 3 = REFUSED
    python -m main.b1.lr_report --look 50M --displacement --out <FILE.json>   # once, at 50M: the weights too
    python -m main.b1.lr_report --look 100M --runs models/rb_b1c_s5101 models/rb_b1w_s5103 ...

The registration is ``designs/research_state/measurements/boundary1_scratch_2026-11-09/PREREG.md`` §18 Amendment 8:
this record is REPORTED beside the 50M and the 100M outcome and decides nothing (no bar, no class). It reads ONLY
each run's TensorBoard event files (``<run>/tb/``) and, with ``--displacement``, checkpoint zips; it writes nothing
in a run directory, and ``--out`` is refused inside a models directory (:func:`check_out`).

**The runs** default to the twelve registered names (:data:`main.b1.arms.LAUNCH_ORDER`) under the models directory;
``--runs`` names the directories instead. A directory's NAME gives its arm and seed (:func:`arm_of`), so it must be a
registered name. **No run is ever dropped:** a run that is missing, or that has not reached the look's step, REFUSES
the whole report with that run's name (:class:`ReportError`).

**The series** (:func:`read_series`) are ``train/learning_rate`` and ``train/approx_kl``, one row an update. The
event files are read in the order of the timestamp in their names; a later file SUPERSEDES every earlier row at or
after its own first step (the updates a stop discarded before a resume from an earlier checkpoint), and each run's
``superseded_rows`` says how many. Only steps that carry BOTH tags are used (``rows_lr_only`` / ``rows_kl_only``).

**The windows** (:func:`run_row`), on a look step ``L``. The COVERING row is the first row at or after ``L``; it must
lie within one checkpoint interval of ``L`` (the rule :func:`main.b1.look.find_checkpoint` applies to a look's
checkpoint), else the run lacks the look step. "Up to the look" is every row up to and including the covering row;
the SECOND HALF is those of them with ``step >= L / 2``. Steps are integers, so both edges are exact.

**Per run:** the geometric-mean learning rate over the second half; the learning rate of the covering row and its
RUNG, the integer ``k`` in ``3e-4 * 1.2 ** k`` (:func:`rung`; the base is production's ``--lr`` and 1.2 the
controller's ``lr_factor``, which every BOUNDARY-1 run inherits); the summed ``train/approx_kl`` over the second
half, and over every row up to the look; and the median over the second half's updates of ``approx_kl / LR ** 2``
(the policy's sensitivity to a step).

**A rung is decided deterministically** (:func:`rung`). A rate within :data:`RUNG_ON_REL` of a rung (float error: the
events store a float32) is ON it; one more than :data:`RUNG_OFF_REL` from the nearest rung is OFF the ladder (a
bound, a frozen rate) and reports ``None``, never a rounded index; anything between is REFUSED as neither.

**Per factor** (:func:`factor_ratios`; width, and breadth for symmetry), the ON / OFF ratio of each quantity with the
OTHER factor matched: ordinary least squares on the LOG of the per-run value with an intercept and an additive 0/1
term for each factor; the ratio is ``exp`` of the factor's coefficient and its two-sided 90 % interval is
``exp(coefficient -/+ t * se)``, ``t`` the 95th percentile of Student's t on ``runs - 3`` degrees of freedom and
``se`` from the residual variance. This is the handoff's method (``b1_lr_confound_plan_2026-11-17``: nine runs, 6
degrees of freedom); the interval describes run-to-run spread and nothing else.

**``--displacement``** (once, at the 50M look; :func:`displacement`): per run, the root-mean-square change per weight
between its checkpoints at :data:`DISPLACEMENT_FROM` and at the look (each the first at or after its step, as a
look's checkpoint is), by MODULE GROUP, and each group's ratio per factor as above. It is the NET displacement, not a
path length. The groups come from the checkpoints' own parameter names (:func:`group_of`): the policy tower, the
critic head, the embeddings and the projection's weight by name, the pointer head and the rest of the trunk split
into the tensors the width lever RESIZED and the ones it left the same shape. "Resized" is read off the runs
themselves (a tensor whose shape differs between two of them); every floating-point tensor lands in exactly one
group, the policy's two alias copies of the trunk are skipped only after they are verified identical to it, and a
named group that matches no tensor REFUSES.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import statistics
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from main.b1 import arms as A
from main.b1 import cross as X
from main.b1 import look as K

SCHEMA = "b1_lr_report_v1"
LR_TAG = "train/learning_rate"
KL_TAG = "train/approx_kl"
#: The looks this report is registered beside (Amendment 8), and their steps.
LOOKS: Dict[str, int] = {k: K.LOOK_STEPS[k] for k in ("50M", "100M")}
#: The controller's ladder: every BOUNDARY-1 run starts at production's ``--lr`` and moves by its ``lr_factor``.
LADDER_BASE = 3e-4
LADDER_FACTOR = 1.2
#: A rate this close to a rung, relatively, is ON it (a float32 holds a rate to about 6e-8) ...
RUNG_ON_REL = 1e-6
#: ... and one farther than this from the nearest rung is OFF the ladder. Between the two: refused.
RUNG_OFF_REL = 1e-4
#: The two-sided level of every interval (the registration's).
LEVEL = X.LEVEL
#: ``--displacement``: the one look it is reported at, and the step of the earlier checkpoint.
DISPLACEMENT_LOOK = "50M"
DISPLACEMENT_FROM = 30_000_000
#: The policy's alias copies of the trunk (``features_extractor.``): skipped once verified identical to it.
TRUNK = "features_extractor."
ALIASES: Tuple[str, ...] = ("pi_features_extractor.", "vf_features_extractor.")
#: group -> the parameter-name prefix that names it (checked first, in this order).
NAMED_GROUPS: Dict[str, str] = {
    "policy_tower": "mlp_extractor.policy_net.",
    "critic_head": TRUNK + "win_head.",
    "embeddings": TRUNK + "embeddings.",
    "projection": TRUNK + "projection.weight",
}
POINTER = "pointer_head."
#: Every group, in print order. ``*_resized``: tensors whose shape the width lever changed; ``*_same_shape``: not.
GROUPS: Tuple[str, ...] = ("trunk_resized", "projection", "policy_tower", "critic_head", "embeddings",
                           "trunk_same_shape", "pointer_head_resized", "pointer_head_same_shape")
#: The per-run quantities a factor's ratio is taken of.
QUANTITIES: Tuple[str, ...] = ("lr_geomean_second_half", "lr_at_look", "kl_sum_second_half", "kl_sum_to_look",
                               "sensitivity_median_second_half")
_EVENT_FILE = re.compile(r"^events\.out\.tfevents\.(\d+)\.")


class ReportError(ValueError):
    """A run that is missing, unnamed or short of the look step; an undecidable rung; an ``--out`` inside models."""


# ------------------------------------------------------------------------------------------------ small statistics
def geomean(xs: Sequence[float]) -> float:
    """The geometric mean of positive values."""
    if not xs:
        raise ReportError("a geometric mean of no values")
    if any(not (x > 0.0) or math.isinf(x) for x in xs):
        raise ReportError("a geometric mean needs finite positive values")
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))


def rung(lr: float) -> Tuple[Optional[int], float]:
    """``(k, exact)``: ``exact`` is ``log(lr / base) / log(factor)``; ``k`` its integer when ``lr`` is ON that rung,
    ``None`` when it is OFF the ladder. A rate between the two tolerances is refused (module docstring)."""
    if not (lr > 0.0) or math.isinf(lr):
        raise ReportError(f"a learning rate must be finite and positive, not {lr!r}")
    exact = math.log(lr / LADDER_BASE) / math.log(LADDER_FACTOR)
    k = round(exact)
    rel = abs(lr / (LADDER_BASE * LADDER_FACTOR ** k) - 1.0)
    if rel <= RUNG_ON_REL:
        return k, exact
    if rel >= RUNG_OFF_REL:
        return None, exact
    raise ReportError(f"learning rate {lr!r} is {rel:.3g} (relative) from rung {k}: farther than float error "
                      f"({RUNG_ON_REL:g}) and nearer than an off-ladder rate ({RUNG_OFF_REL:g}); not decided")


def factor_levels(arm: str) -> Tuple[int, ...]:
    """The 0/1 level of each factor (:data:`main.b1.cross.FACTORS`' order) for a kit arm (``c``, ``w``, ``br``, ``wb``)."""
    return tuple(int(arm in {K.KIT_ARM[a] for a in on}) for on, _off in X.FACTORS.values())


def factor_ratios(arms: Sequence[str], values: Sequence[float]) -> Dict[str, Dict[str, float]]:
    """factor -> ``{ratio, lo, hi, df}``: the ON / OFF ratio of ``values`` (one per run, ``arms`` its kit arm), the
    other factor matched, by least squares on the log with additive factor terms (module docstring)."""
    import numpy as np

    if len(arms) != len(values):
        raise ReportError("one value a run")
    if any(not (v > 0.0) or math.isinf(v) for v in values):
        raise ReportError("a ratio needs finite positive values")
    design = np.array([[1.0, *factor_levels(a)] for a in arms], dtype=float)
    width = design.shape[1]
    df = len(arms) - width
    if df < 1 or int(np.linalg.matrix_rank(design)) < width:
        raise ReportError(f"{len(arms)} runs over arms {sorted(set(arms))} cannot separate the factors with an "
                          f"interval: both levels of each factor and more than {width} runs are needed")
    y = np.log(np.array(values, dtype=float))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    cov = float(resid @ resid) / df * np.linalg.inv(design.T @ design)
    t = X.t_ppf(0.5 + LEVEL / 2.0, df)
    out: Dict[str, Dict[str, float]] = {}
    for j, factor in enumerate(X.FACTORS, start=1):
        b, se = float(beta[j]), math.sqrt(max(float(cov[j, j]), 0.0))
        out[factor] = {"ratio": math.exp(b), "lo": math.exp(b - t * se), "hi": math.exp(b + t * se), "df": df}
    return out


# ------------------------------------------------------------------------------------------------ the runs
def arm_of(run_dir: Path) -> Tuple[str, int]:
    """``(kit arm, seed)`` from the directory's name, which must be a registered run name (or a spare-seed one)."""
    by_name = {A.run_name(a, s): (a, s) for a in A.ARMS for s in (*A.SEEDS[a], A.SPARE_SEED)}
    if run_dir.name not in by_name:
        raise ReportError(f"{run_dir.name!r} is not a BOUNDARY-1 run name; its arm cannot be read from it")
    return by_name[run_dir.name]


def default_runs(models_dir: Optional[Path] = None) -> List[Path]:
    """The twelve registered run directories, in launch order, under ``models_dir`` (default: the main checkout's)."""
    if models_dir is None:
        from utils.paths import main_models_dir

        models_dir = main_models_dir()
    if models_dir is None:
        raise ReportError("there is no models directory here; pass --runs or --models-dir")
    return [Path(models_dir) / A.run_name(a, s) for a, s in A.LAUNCH_ORDER]


def event_files(run_dir: Path) -> List[Path]:
    """The run's TensorBoard event files, oldest first (the timestamp in the name, then the name)."""
    tb = run_dir / "tb"
    found = [(int(m.group(1)), p.name, p) for p in (tb.rglob("events.out.tfevents.*") if tb.is_dir() else [])
             if (m := _EVENT_FILE.match(p.name))]
    return [p for _ts, _name, p in sorted(found)]


def read_series(run_dir: Path) -> Dict[str, Any]:
    """``{steps, lr, kl, files, superseded_rows, rows_lr_only, rows_kl_only}``: the two series on their common steps,
    ascending (module docstring: a later file supersedes the earlier rows at or after its first step)."""
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util import tensor_util

    files = event_files(run_dir)
    if not files:
        raise ReportError(f"{run_dir.name}: no TensorBoard event file under {run_dir / 'tb'}")
    series: Dict[str, Dict[int, float]] = {LR_TAG: {}, KL_TAG: {}}
    superseded = 0
    for f in files:
        own: Dict[str, Dict[int, float]] = {LR_TAG: {}, KL_TAG: {}}
        for event in EventFileLoader(str(f)).Load():
            for v in event.summary.value:
                if v.tag in own:
                    own[v.tag][int(event.step)] = (float(v.simple_value) if v.HasField("simple_value")
                                                   else float(tensor_util.make_ndarray(v.tensor)))
        for tag, rows in own.items():
            if rows:
                first = min(rows)
                stale = [s for s in series[tag] if s >= first]
                superseded += len(stale)
                for s in stale:
                    del series[tag][s]
                series[tag].update(rows)
    steps = sorted(set(series[LR_TAG]) & set(series[KL_TAG]))
    return {"steps": steps, "lr": [series[LR_TAG][s] for s in steps], "kl": [series[KL_TAG][s] for s in steps],
            "files": [f.name for f in files], "superseded_rows": superseded,
            "rows_lr_only": len(set(series[LR_TAG]) - set(series[KL_TAG])),
            "rows_kl_only": len(set(series[KL_TAG]) - set(series[LR_TAG]))}


def run_row(run_dir: Path, look_step: int, series: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """One run's record at a look step (module docstring: the windows, the per-run quantities)."""
    arm, seed = arm_of(run_dir)
    s = read_series(run_dir) if series is None else series
    steps, lr, kl = s["steps"], s["lr"], s["kl"]
    cover = next((i for i, step in enumerate(steps) if step >= look_step), None)
    if cover is None:
        last = f"its last row is at {steps[-1]:,}" if steps else "it has no row"
        raise ReportError(f"{run_dir.name} lacks the look step {look_step:,}: {last}")
    if steps[cover] >= look_step + A.CHECKPOINT_EVERY:
        raise ReportError(f"{run_dir.name} lacks the look step {look_step:,}: its first row at or after it is "
                          f"{steps[cover]:,}, more than one interval ({A.CHECKPOINT_EVERY:,}) late")
    half = next(i for i, step in enumerate(steps) if 2 * step >= look_step)
    lr2, kl2 = lr[half:cover + 1], kl[half:cover + 1]
    k, exact = rung(lr[cover])
    return {"run": run_dir.name, "arm": arm, "seed": seed, "tb_files": list(s["files"]),
            "superseded_rows": s["superseded_rows"], "rows_lr_only": s["rows_lr_only"],
            "rows_kl_only": s["rows_kl_only"], "rows_to_look": cover + 1, "rows_second_half": len(lr2),
            "second_half_first_step": steps[half], "look_row_step": steps[cover],
            "lr_geomean_second_half": geomean(lr2), "lr_at_look": lr[cover], "rung": k, "rung_exact": exact,
            "kl_sum_second_half": math.fsum(kl2), "kl_sum_to_look": math.fsum(kl[:cover + 1]),
            "sensitivity_median_second_half": statistics.median(q / r ** 2 for q, r in zip(kl2, lr2))}


# ------------------------------------------------------------------------------------------------ the weights
def load_policy(zip_path: Path) -> Dict[str, Any]:
    """``policy.pth``'s floating-point tensors (float64), minus the alias copies of the trunk, which must each be
    identical to the trunk tensor they copy."""
    import torch

    with zipfile.ZipFile(zip_path) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    out: Dict[str, Any] = {}
    for key, v in sd.items():
        alias = next((a for a in ALIASES if key.startswith(a)), None)
        if alias is not None:
            twin = TRUNK + key[len(alias):]
            if twin not in sd or not torch.equal(sd[twin], v):
                raise ReportError(f"{zip_path}: {key} is not a copy of {twin}; it cannot be skipped as an alias")
        elif torch.is_floating_point(v):
            out[key] = v.double()
    return out


def group_of(key: str, resized: bool) -> str:
    """The module group of one parameter name; ``resized``: the width lever changed its shape."""
    for group, prefix in NAMED_GROUPS.items():
        if key.startswith(prefix):
            return group
    if key.startswith(POINTER):
        return "pointer_head_resized" if resized else "pointer_head_same_shape"
    if key.startswith(TRUNK):
        return "trunk_resized" if resized else "trunk_same_shape"
    raise ReportError(f"parameter {key!r} belongs to no module group this report knows")


def displacement(run_dirs: Sequence[Path], from_step: int, to_step: int) -> Dict[str, Any]:
    """Per run and group, the root-mean-square change per weight between the checkpoints at ``from_step`` and
    ``to_step``; per factor, each group's ratio (module docstring)."""
    pairs: Dict[str, Tuple[Path, int, Path, int]] = {}
    missing: List[str] = []
    for run in run_dirs:
        got = [K.find_checkpoint(run, step) for step in (from_step, to_step)]
        missing += [f"{run.name}: {why}" for _z, _n, why in got if why]
        if not any(why for _z, _n, why in got):
            (za, na, _), (zb, nb, _) = got
            assert za is not None and na is not None and zb is not None and nb is not None
            pairs[run.name] = (za, na, zb, nb)
    if missing:
        raise ReportError("no displacement; " + "; ".join(missing))
    sums: Dict[str, Dict[str, Tuple[float, int]]] = {}
    shapes: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    for run in run_dirs:
        za, _na, zb, _nb = pairs[run.name]
        a, b = load_policy(za), load_policy(zb)
        if {k: tuple(v.shape) for k, v in a.items()} != {k: tuple(v.shape) for k, v in b.items()}:
            raise ReportError(f"{run.name}: its two checkpoints do not hold the same tensors")
        shapes[run.name] = {k: tuple(v.shape) for k, v in a.items()}
        sums[run.name] = {k: (float(((b[k] - a[k]) ** 2).sum()), int(a[k].numel())) for k in a}
    names = [r.name for r in run_dirs]
    keys = list(shapes[names[0]])
    if any(list(shapes[n]) != keys for n in names):
        raise ReportError("the runs' checkpoints do not hold the same parameter names")
    arms = [arm_of(r)[0] for r in run_dirs]
    wide = {n: factor_levels(a)[list(X.FACTORS).index("width")] for n, a in zip(names, arms)}
    for level in (0, 1):
        same = [n for n in names if wide[n] == level]
        if any(shapes[n] != shapes[same[0]] for n in same):
            raise ReportError(f"runs of one width hold tensors of different shapes: {same}")
    resized = {k: len({shapes[n][k] for n in names}) > 1 for k in keys}
    per_run: Dict[str, Any] = {}
    for n in names:
        acc: Dict[str, List[float]] = {g: [0.0, 0.0] for g in GROUPS}
        for k in keys:
            sq, count = sums[n][k]
            g = acc[group_of(k, resized[k])]
            g[0] += sq
            g[1] += count
        empty = [g for g, (_sq, count) in acc.items() if g in NAMED_GROUPS and not count]
        if empty:
            raise ReportError(f"{n}: no tensor matches the named group(s) {empty}")
        _za, na, _zb, nb = pairs[n]
        per_run[n] = {"from_checkpoint_step": na, "to_checkpoint_step": nb,
                      "groups": {g: {"rms": math.sqrt(sq / count) if count else None, "weights": int(count)}
                                 for g, (sq, count) in acc.items()}}
    factors: Dict[str, Dict[str, Any]] = {f: {} for f in X.FACTORS}
    for g in GROUPS:
        vals = [per_run[n]["groups"][g]["rms"] for n in names]
        if all(v is not None for v in vals):
            for f, r in factor_ratios(arms, vals).items():
                factors[f][g] = r
    return {"from_step": from_step, "to_step": to_step, "runs": per_run, "factors": factors,
            "resized_tensors": sum(resized.values()), "tensors": len(keys)}


# ------------------------------------------------------------------------------------------------ the report
def check_out(out: Path, run_dirs: Sequence[Path]) -> Path:
    """The resolved ``--out`` path, refused inside the main checkout's models directory, a run directory read, or
    the directory that holds one (this report writes nothing where runs live)."""
    from utils.paths import main_models_dir

    target = out.resolve()
    roots = {r.resolve() for r in run_dirs} | {r.resolve().parent for r in run_dirs}
    main = main_models_dir()
    if main is not None:
        roots.add(Path(main).resolve())
    for root in sorted(roots):
        if target == root or root in target.parents:
            raise ReportError(f"--out {target} lies inside {root}: the report is written OUTSIDE models/")
    return target


def report(run_dirs: Sequence[Path], look: str, with_displacement: bool = False) -> Dict[str, Any]:
    """The whole record for one look (schema :data:`SCHEMA`); raises :class:`ReportError` rather than drop a run."""
    if look not in LOOKS:
        raise ReportError(f"look {look!r} is not one of {sorted(LOOKS)}")
    if with_displacement and look != DISPLACEMENT_LOOK:
        raise ReportError(f"--displacement is reported once, at the {DISPLACEMENT_LOOK} look, not at {look}")
    names = [r.name for r in run_dirs]
    if len(set(names)) != len(names):
        raise ReportError(f"a run is named twice: {sorted(n for n in set(names) if names.count(n) > 1)}")
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for run in run_dirs:
        try:
            if not run.is_dir():
                raise ReportError(f"{run.name}: no run directory at {run}")
            rows.append(run_row(run, LOOKS[look]))
        except ReportError as e:
            problems.append(str(e))
    if problems:
        raise ReportError(f"{len(problems)} of {len(run_dirs)} runs cannot be read at {look}, and no run is "
                          "dropped: " + "; ".join(problems))
    arms = [r["arm"] for r in rows]
    factors: Dict[str, Dict[str, Any]] = {f: {} for f in X.FACTORS}
    for q in QUANTITIES:
        for f, r in factor_ratios(arms, [r[q] for r in rows]).items():
            factors[f][q] = r
    doc: Dict[str, Any] = {
        "schema": SCHEMA, "look": look, "look_step": LOOKS[look], "second_half_from": LOOKS[look] / 2,
        "ladder": {"base": LADDER_BASE, "factor": LADDER_FACTOR}, "level": LEVEL, "tags": [LR_TAG, KL_TAG],
        "method": "per factor: least squares on log(value) with an intercept and additive 0/1 width and breadth "
                  "terms; ratio = exp(coefficient), interval = exp(coefficient -/+ t(0.95, runs - 3) * se)",
        "n_runs": len(rows), "runs": rows, "factors": factors}
    if with_displacement:
        doc["displacement"] = displacement(run_dirs, DISPLACEMENT_FROM, LOOKS[look])
    return doc


def _ratio(r: Mapping[str, float]) -> str:
    return f"x{r['ratio']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def render(doc: Mapping[str, Any]) -> str:
    """The record as text: one row a run, then each factor's ratios."""
    lines = [f"BOUNDARY-1 learning-rate report at the {doc['look']} look (REPORTED; {doc['n_runs']} runs; second half "
             f"= steps >= {int(doc['second_half_from']):,} up to each run's first row at or after "
             f"{doc['look_step']:,})",
             f"{'run':<16}{'arm':<4}{'rows':>5} {'geomean LR':>11} {'LR at look':>11} {'rung':>5} "
             f"{'sum KL 2nd half':>16} {'sum KL to look':>15} {'median KL/LR^2':>15}"]
    for r in doc["runs"]:
        k = "off" if r["rung"] is None else f"{r['rung']:+d}"
        lines.append(f"{r['run']:<16}{r['arm']:<4}{r['rows_second_half']:>5} {r['lr_geomean_second_half']:>11.4e} "
                     f"{r['lr_at_look']:>11.4e} {k:>5} {r['kl_sum_second_half']:>16.4f} "
                     f"{r['kl_sum_to_look']:>15.4f} {r['sensitivity_median_second_half']:>15.4e}")
        if r["superseded_rows"] or r["rows_lr_only"] or r["rows_kl_only"]:
            lines.append(f"    {r['run']}: {r['superseded_rows']} superseded rows, {r['rows_lr_only']} with the "
                         f"rate alone, {r['rows_kl_only']} with the KL alone")
    for f, qs in doc["factors"].items():
        df = next(iter(qs.values()))["df"]
        lines.append(f"{f} ON / OFF, the other factor matched ({int(doc['level'] * 100)} % interval, {df} df):")
        lines += [f"    {q:<32} {_ratio(r)}" for q, r in qs.items()]
    d = doc.get("displacement")
    if d:
        lines.append(f"displacement: RMS change per weight, checkpoints at {d['from_step']:,} -> {d['to_step']:,} "
                     f"({d['resized_tensors']} of {d['tensors']} tensors resized by the width lever)")
        first = next(iter(d["runs"].values()))["groups"]
        for f, gs in d["factors"].items():
            lines.append(f"  {f} ON / OFF:")
            lines += [f"    {g:<26} {_ratio(r)}" for g, r in gs.items()]
        lines.append("  weights a group (first run): " + ", ".join(f"{g} {first[g]['weights']:,}" for g in GROUPS))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.b1.lr_report", description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--look", required=True, choices=tuple(LOOKS))
    ap.add_argument("--runs", nargs="+", help="run directories (default: the twelve registered runs)")
    ap.add_argument("--models-dir", help="where the registered runs live (default: the main checkout's models/)")
    ap.add_argument("--displacement", action="store_true",
                    help=f"also the weights' displacement by module group (the {DISPLACEMENT_LOOK} look only)")
    ap.add_argument("--out", help="also write the record as JSON here (refused inside a models directory)")
    a = ap.parse_args(argv)
    try:
        if a.runs and a.models_dir:
            raise ReportError("--runs and --models-dir do not go together")
        runs = ([Path(r) for r in a.runs] if a.runs
                else default_runs(Path(a.models_dir) if a.models_dir else None))
        out = check_out(Path(a.out), runs) if a.out else None
        doc = report(runs, a.look, a.displacement)
    except ReportError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    print(render(doc))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
