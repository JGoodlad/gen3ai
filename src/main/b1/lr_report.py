"""BOUNDARY-1's LEARNING-RATE REPORT beside a look: what step each run trained at, and where it landed. REPORTED.

    python -m main.b1.lr_report --look 50M                          # the twelve registered runs; exit 3 = REFUSED
    python -m main.b1.lr_report --look 50M --displacement --out <FILE.json>   # once, at 50M: the weights too
    python -m main.b1.lr_report --look 100M --runs models/rb_b1c_s5101 models/rb_b1w_s5103 ...
    python -m main.b1.lr_report --look 50M --fresh-arm <fresh_arm.json>          # a FRESH CELL beside its banked cell
                                                                   # (+ its REPORTED meters: clip, entropy, ...)

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

**``--fresh-arm``** (:func:`fresh_report`, schema :data:`FRESH_SCHEMA`; REPORTED, it voids nothing and changes no
class): the same reader on a fresh cell's runs, named by the card's arm file (:mod:`main.b1.fresh`; ``--replace`` is
B1's replaced seed, as in that kit), beside its banked cell's. Everything above this paragraph is untouched by it.
The registration is the Q5-N card's ``PREREG.md`` §3 (``q5n_epochs_fresh_2026-11-20``). Per run: the quantities
above, on the run's OWN ladder (:func:`cell_bases`: a fresh run's base is the ``--lr`` its lever types, else
production's; the banked cell's is production's), the RUNG HISTOGRAM of every row up to the look
(:func:`rung_histogram`: rows a rung, and the rows OFF the ladder with their rates, which is where a floor or a
ceiling shows), and the median of ``train/approx_kl`` over the rows with :data:`KL_WINDOW` (10M <= step <= 50M, the
same window at either look; superseded rows are already gone). Per cell: the summed histogram and the MEDIAN of its
runs' KL medians. **The label** :data:`LABEL` is printed unless the FRESH cell's figure lies strictly inside the
controller's band shrunk by the registered edge, ``0.005 + 1e-6 < figure < 0.02 - 1e-6`` (:func:`movement_label`).
The window of the histogram is this module's choice (the registration names none); for a card other than Q5-N the
label is the same arithmetic with no registered meaning.

**The fresh cell's REPORTED meters** (:func:`run_meters`, :func:`cell_meters`; Q5-N PREREG §3's list, read in the same
one pass over the event files, :func:`load_tags`): per run, each of :data:`METERS` (the clip fraction
``train/clip_fraction``, which is the mean over ALL epochs of an update; the LAST epoch's, the highest
``train/clip_fraction_epoch_<k>`` at a row, whose ``epochs`` say which k that was; the entropy, the negated
``train/entropy_loss``; the value loss) as the median over each :data:`METER_WINDOW` window ``[k W, (k + 1) W)``
below the look and over :data:`KL_WINDOW`; the rows (up to the look) AT the registered floor (:data:`LR_FLOOR`) or
ceiling (:data:`CEILING_FACTOR` x the side's base) within float error, and BEYOND either; the realized dose to the
look (``main.dose`` with ``upto`` = the look checkpoint's step, the file the look plays); and the STEP TIME
(:func:`step_time`: the card check's one formula, ``pilot_rho.py``'s, over :data:`STEP_TIME_WINDOW`, in exact rational
arithmetic). Per cell, the MEAN of its runs' figures; the record's ``dose_rate_ratio`` and ``step_time_ratio`` (rho,
also ``_exact`` as ``"p/q"``) are fresh over banked. **Absence is never a zero and never refuses the label:** a meter
with no row reads ``None``, a step time or dose that cannot be read carries its ``error``, and a ratio is ``None``
unless every run of both cells is read. The same rho, by :func:`step_time_ratio`, prices the matched-GPU-time cross
(``python -m main.b1.fresh plan-gpu-time``); **its use as rho_run waits on Q5-N's dated amendment registering this
meter** (PREREG §3).
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
from fractions import Fraction
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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
#: ``--fresh-arm``: the record's schema; the label's window, band and edge (Q5-N PREREG §3), and the label.
FRESH_SCHEMA = "b1_fresh_lr_report_v1"
KL_WINDOW: Tuple[int, int] = (10_000_000, 50_000_000)
KL_BAND: Tuple[float, float] = (0.005, 0.02)
KL_BAND_EDGE = 1e-6
LABEL = "MOVEMENT-UNMATCHED"
#: ``--fresh-arm``'s REPORTED training meters (Q5-N PREREG §3), read from the same event files in the same pass.
CLIP_TAG = "train/clip_fraction"
CLIP_EPOCH_PREFIX = "train/clip_fraction_epoch_"
ENTROPY_LOSS_TAG = "train/entropy_loss"
VALUE_LOSS_TAG = "train/value_loss"
#: meter -> (what it is read from, its sign): the entropy is the NEGATED entropy loss (``loss = pg + ent_coef *
#: entropy_loss``); ``clip_fraction_last_epoch`` is the highest ``train/clip_fraction_epoch_<k>`` at a row.
METERS: Dict[str, Tuple[str, float]] = {"clip_fraction": (CLIP_TAG, 1.0),
                                        "clip_fraction_last_epoch": (CLIP_EPOCH_PREFIX + "<last k>", 1.0),
                                        "entropy": (ENTROPY_LOSS_TAG, -1.0), "value_loss": (VALUE_LOSS_TAG, 1.0)}
METER_WINDOW = 5_000_000
#: The controller's registered bounds (Q5-N PREREG §2): the floor ``--min-lr``'s 1e-5, the ceiling 2 x the side's base.
LR_FLOOR = 1e-5
CEILING_FACTOR = 2.0
#: The step-time meter: Q6-N PREREG §7 item 5's card-check formula, on the real runs' rows in STEP_TIME_WINDOW.
UPDATE_TAG = "lifecycle/update_wall_s"
COLLECT_TAG = "rollout/collect_ms"
TRAIN_MS_TAG = "train/train_ms"
STEP_TIME_WINDOW: Tuple[int, int] = KL_WINDOW
STEP_TIME_MIN_ROWS = 10
FRESH_TAGS = frozenset({LR_TAG, KL_TAG, CLIP_TAG, ENTROPY_LOSS_TAG, VALUE_LOSS_TAG, UPDATE_TAG, COLLECT_TAG,
                        TRAIN_MS_TAG})
_CLIP_EPOCH = re.compile(r"^" + re.escape(CLIP_EPOCH_PREFIX) + r"(\d+)$")
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


def rung(lr: float, base: float = LADDER_BASE) -> Tuple[Optional[int], float]:
    """``(k, exact)``: ``exact`` is ``log(lr / base) / log(factor)``; ``k`` its integer when ``lr`` is ON that rung,
    ``None`` when it is OFF the ladder. A rate between the two tolerances is refused (module docstring). ``base`` is
    BOUNDARY-1's unless a fresh cell's lever types another."""
    if not (lr > 0.0) or math.isinf(lr):
        raise ReportError(f"a learning rate must be finite and positive, not {lr!r}")
    exact = math.log(lr / base) / math.log(LADDER_FACTOR)
    k = round(exact)
    rel = abs(lr / (base * LADDER_FACTOR ** k) - 1.0)
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


TagRows = Dict[str, Dict[int, float]]


def load_tags(run_dir: Path, wanted: Callable[[str], bool]) -> Tuple[TagRows, Dict[str, int], List[str]]:
    """``(tag -> {step: value}, tag -> superseded rows, event file names)`` for every tag ``wanted`` accepts, read in
    ONE pass of the run's event files (module docstring: a later file supersedes, tag by tag, the earlier rows at or
    after its own first step of that tag)."""
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util import tensor_util

    files = event_files(run_dir)
    if not files:
        raise ReportError(f"{run_dir.name}: no TensorBoard event file under {run_dir / 'tb'}")
    series: TagRows = {}
    superseded: Dict[str, int] = {}
    for f in files:
        own: TagRows = {}
        for event in EventFileLoader(str(f)).Load():
            for v in event.summary.value:
                if wanted(v.tag):
                    own.setdefault(v.tag, {})[int(event.step)] = (
                        float(v.simple_value) if v.HasField("simple_value")
                        else float(tensor_util.make_ndarray(v.tensor)))
        for tag, rows in own.items():
            have = series.setdefault(tag, {})
            first = min(rows)
            stale = [s for s in have if s >= first]
            superseded[tag] = superseded.get(tag, 0) + len(stale)
            for s in stale:
                del have[s]
            have.update(rows)
    return series, superseded, [f.name for f in files]


def series_of(loaded: Tuple[TagRows, Mapping[str, int], Sequence[str]]) -> Dict[str, Any]:
    """:func:`read_series`'s record from a :func:`load_tags` result that holds (at least) the two tags."""
    series, superseded, files = loaded
    lr, kl = series.get(LR_TAG, {}), series.get(KL_TAG, {})
    steps = sorted(set(lr) & set(kl))
    return {"steps": steps, "lr": [lr[s] for s in steps], "kl": [kl[s] for s in steps], "files": list(files),
            "superseded_rows": superseded.get(LR_TAG, 0) + superseded.get(KL_TAG, 0),
            "rows_lr_only": len(set(lr) - set(kl)), "rows_kl_only": len(set(kl) - set(lr))}


def read_series(run_dir: Path) -> Dict[str, Any]:
    """``{steps, lr, kl, files, superseded_rows, rows_lr_only, rows_kl_only}``: the two series on their common steps,
    ascending (module docstring: a later file supersedes the earlier rows at or after its first step)."""
    return series_of(load_tags(run_dir, {LR_TAG, KL_TAG}.__contains__))


def run_row(run_dir: Path, look_step: int, series: Optional[Mapping[str, Any]] = None, *,
            named: Optional[Tuple[str, int]] = None, base: float = LADDER_BASE) -> Dict[str, Any]:
    """One run's record at a look step (module docstring: the windows, the per-run quantities). ``named`` and
    ``base`` are a fresh cell's: the run's ``(side, seed)`` from its arm file, and its ladder's base."""
    arm, seed = arm_of(run_dir) if named is None else named
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
    k, exact = rung(lr[cover], base)
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


# ------------------------------------------------------------------------------------------------ a fresh cell
def cell_bases(lever: Sequence[str], declared_keys: Sequence[str]) -> Dict[str, float]:
    """side -> its ladder's base: the ``--lr`` the lever types for the fresh side, production's otherwise. An arm
    that declares ``lr`` without typing ``--lr`` (or the reverse) is refused: its base cannot be read."""
    typed = [lever[i + 1] if t == "--lr" and i + 1 < len(lever) else t.split("=", 1)[1]
             for i, t in enumerate(lever) if t == "--lr" or t.startswith("--lr=")]
    if len(typed) != int("lr" in declared_keys):
        raise ReportError(f"the arm declares {sorted(declared_keys)} and its lever types --lr {len(typed)} time(s): "
                          "the fresh ladder's base cannot be read from it")
    try:
        fresh = float(typed[0]) if typed else LADDER_BASE
    except ValueError:
        raise ReportError(f"the lever's --lr {typed[0]!r} is not a number") from None
    if not (fresh > 0.0) or math.isinf(fresh):
        raise ReportError(f"the lever's --lr {fresh!r} is not a finite positive rate")
    return {"fresh": fresh, "banked": LADDER_BASE}


def rung_histogram(lrs: Sequence[float], base: float) -> Dict[str, Any]:
    """``{rungs: {k: rows}, off: rows, off_rates: [[rate, rows]]}`` of one run's rates on the ladder from ``base``
    (:func:`rung`: an undecidable rate refuses)."""
    on: Dict[int, int] = {}
    off: Dict[float, int] = {}
    for lr in lrs:
        k, _exact = rung(lr, base)
        if k is None:
            off[lr] = off.get(lr, 0) + 1
        else:
            on[k] = on.get(k, 0) + 1
    return {"rungs": {str(k): on[k] for k in sorted(on)}, "off": sum(off.values()),
            "off_rates": [[r, n] for r, n in sorted(off.items())]}


def kl_window_median(run: str, steps: Sequence[int], kl: Sequence[float]) -> Tuple[float, int]:
    """``(median, rows)`` of the KL over the rows with :data:`KL_WINDOW`; refused when the run's rows do not span it."""
    lo, hi = KL_WINDOW
    if not steps or steps[0] > lo or steps[-1] < hi:
        span = f"{steps[0]:,} to {steps[-1]:,}" if steps else "none"
        raise ReportError(f"{run}: its rows ({span}) do not span the KL window {lo:,} to {hi:,}")
    rows = [q for s, q in zip(steps, kl) if lo <= s <= hi]
    if not rows:
        raise ReportError(f"{run}: no row in the KL window {lo:,} to {hi:,}")
    return statistics.median(rows), len(rows)


def movement_label(figure: float) -> Optional[str]:
    """:data:`LABEL` unless ``0.005 + 1e-6 < figure < 0.02 - 1e-6`` (the registered inequality, as written)."""
    lo, hi = KL_BAND
    return None if lo + KL_BAND_EDGE < figure < hi - KL_BAND_EDGE else LABEL


def last_epoch_clip(series: TagRows) -> Tuple[Dict[int, float], List[int]]:
    """``(step -> the clip fraction of the highest epoch logged at that step, the sorted set of those epochs)``."""
    best: Dict[int, Tuple[int, float]] = {}
    for tag, rows in series.items():
        m = _CLIP_EPOCH.match(tag)
        if m:
            k = int(m.group(1))
            for step, v in rows.items():
                if step not in best or k > best[step][0]:
                    best[step] = (k, v)
    return {step: v for step, (_k, v) in best.items()}, sorted({k for k, _v in best.values()})


def meter(points: Mapping[int, float], look_step: int, sign: float = 1.0) -> Dict[str, Any]:
    """One REPORTED meter of one run: the median of ``sign * value`` over the rows of each :data:`METER_WINDOW`
    window ``[k W, (k + 1) W)`` below the look step (``None``: a window with no row), and over :data:`KL_WINDOW`."""
    lo, hi = KL_WINDOW
    medians: List[Optional[float]] = []
    counts: List[int] = []
    for k in range(look_step // METER_WINDOW):
        xs = [sign * v + 0.0 for step, v in points.items() if k * METER_WINDOW <= step < (k + 1) * METER_WINDOW]
        medians.append(statistics.median(xs) if xs else None)
        counts.append(len(xs))
    span = [sign * v + 0.0 for step, v in points.items() if lo <= step <= hi]
    return {"rows": len(points), "kl_window_median": statistics.median(span) if span else None,
            "kl_window_rows": len(span), "by_window": medians, "by_window_rows": counts}


def bound_touches(steps: Sequence[int], lrs: Sequence[float], base: float) -> Dict[str, Any]:
    """The rows (up to the look) AT the registered floor or ceiling (within :data:`RUNG_ON_REL`, as a rung is), the
    first step of each, and the rows BEYOND either (a moved bound shows there)."""
    floor, ceiling = LR_FLOOR, CEILING_FACTOR * base
    at_f = [s for s, r in zip(steps, lrs) if abs(r / floor - 1.0) <= RUNG_ON_REL]
    at_c = [s for s, r in zip(steps, lrs) if abs(r / ceiling - 1.0) <= RUNG_ON_REL]
    return {"floor": floor, "ceiling": ceiling, "at_floor": len(at_f), "first_at_floor": at_f[0] if at_f else None,
            "at_ceiling": len(at_c), "first_at_ceiling": at_c[0] if at_c else None,
            "below_floor": sum(1 for r in lrs if r < floor * (1.0 - RUNG_ON_REL)),
            "above_ceiling": sum(1 for r in lrs if r > ceiling * (1.0 + RUNG_ON_REL))}


def exact(q: Fraction) -> str:
    """A rational as ``"p/q"`` (``Fraction(...)`` reads it back exactly)."""
    return f"{q.numerator}/{q.denominator}"


def step_time(run: str, series: TagRows) -> Dict[str, Any]:
    """One run's seconds per 1M steps over :data:`STEP_TIME_WINDOW`, by the card check's ONE formula
    (``pilot_rho.py``, Q6-N PREREG §7 item 5): ``(median update_wall_s + median collect_ms / 1000) / median
    d(steps) * 1e6`` over the rollout rows (steps carrying both tags; a resume's superseded rows already gone) with
    ``lo <= step <= hi``, ``d(steps)`` between consecutive rollout rows whose later row is in the window. Computed
    in EXACT rational arithmetic (``sec_per_1M_steps_exact``), so a ratio and a step chosen from it carry no float
    rounding. Also REPORTED, never the meter: ``train/train_ms`` on the same rows and steps. Refused when the rows do
    not span the window or are fewer than :data:`STEP_TIME_MIN_ROWS`."""
    upd, col, tms = (series.get(t, {}) for t in (UPDATE_TAG, COLLECT_TAG, TRAIN_MS_TAG))
    lo, hi = STEP_TIME_WINDOW
    every = sorted(set(upd) & set(col))
    if not every or every[0] > lo or every[-1] < hi:
        span = f"{every[0]:,} to {every[-1]:,}" if every else "none"
        raise ReportError(f"{run}: its rollout rows ({span}) do not span the step-time window {lo:,} to {hi:,}")
    rows = [step for step in every if lo <= step <= hi]
    if len(rows) < STEP_TIME_MIN_ROWS:
        raise ReportError(f"{run}: {len(rows)} rollout rows in the step-time window; the meter needs "
                          f"{STEP_TIME_MIN_ROWS}")
    d = statistics.median([Fraction(b - a) for a, b in zip(every, every[1:]) if lo <= b <= hi])
    u = statistics.median([Fraction(upd[step]) for step in rows])
    c = statistics.median([Fraction(col[step]) for step in rows]) / 1000
    if not (d > 0 and u >= 0 and c >= 0 and u + c > 0):
        raise ReportError(f"{run}: the step-time medians are not positive (update {float(u)} s, collect "
                          f"{float(c)} s, {float(d)} steps a rollout)")
    sec = (u + c) / d * 10 ** 6
    t = [Fraction(tms[step]) for step in rows if step in tms]
    train = statistics.median(t) / 1000 / d * 10 ** 6 if t else None
    return {"rows": len(rows), "first_step": rows[0], "last_step": rows[-1], "median_update_s": float(u),
            "median_collect_s": float(c), "median_dsteps": float(d), "update_share": float(u / (u + c)),
            "sec_per_1M_steps": float(sec), "sec_per_1M_steps_exact": exact(sec),
            "train_ms_sec_per_1M_steps": None if train is None else float(train), "train_ms_rows": len(t)}


def cell_mean_exact(figures: Sequence[str]) -> Fraction:
    """The exact mean of runs' ``"p/q"`` figures."""
    if not figures:
        raise ReportError("a cell mean of no runs")
    return sum((Fraction(f) for f in figures), Fraction(0)) / len(figures)


def step_time_ratio(fresh: Sequence[Mapping[str, Any]], banked: Sequence[Mapping[str, Any]]) -> Fraction:
    """rho: the fresh cell's mean seconds per 1M steps over the banked cell's (each run a :func:`step_time`)."""
    return (cell_mean_exact([r["sec_per_1M_steps_exact"] for r in fresh])
            / cell_mean_exact([r["sec_per_1M_steps_exact"] for r in banked]))


def run_step_time(run_dir: Path) -> Dict[str, Any]:
    """:func:`step_time` of one run directory (one pass over its event files, the three timing tags alone)."""
    series, _sup, _files = load_tags(run_dir, {UPDATE_TAG, COLLECT_TAG, TRAIN_MS_TAG}.__contains__)
    return step_time(run_dir.name, series)


def _wanted(tag: str) -> bool:
    return tag in FRESH_TAGS or bool(_CLIP_EPOCH.match(tag))


def run_meters(run_dir: Path, series: TagRows, row: Mapping[str, Any], base: float, look_step: int,
               lr_steps: Sequence[int], lrs: Sequence[float]) -> Dict[str, Any]:
    """The REPORTED meters of one fresh-arm run (module docstring): each :data:`METERS` entry, the bounds' touches,
    the dose to the look (``main.dose`` cut at the look's checkpoint, the file the look plays) and the step time. A meter with no row reads
    ``None`` and a step time that cannot be read carries its ``error``: absence is never a zero, and never refuses
    the registered label."""
    from main import dose as D

    last, epochs = last_epoch_clip(series)
    points = {"clip_fraction": series.get(CLIP_TAG, {}), "clip_fraction_last_epoch": last,
              "entropy": series.get(ENTROPY_LOSS_TAG, {}), "value_loss": series.get(VALUE_LOSS_TAG, {})}
    out: Dict[str, Any] = {m: meter(points[m], look_step, METERS[m][1]) for m in METERS}
    out["clip_fraction_last_epoch"]["epochs"] = epochs
    try:
        st: Dict[str, Any] = step_time(run_dir.name, series)
    except ReportError as e:
        st = {"error": str(e)}
    _zip, look_ckpt, why = K.find_checkpoint(run_dir, look_step)
    dose = (D.read_run(str(run_dir), upto=look_ckpt) if look_ckpt is not None
            else {"error": f"no look checkpoint to cut the dose at ({why})", "upto": None})
    return {"meters": out, "lr_bounds": bound_touches(lr_steps, lrs, base), "step_time": st,
            "dose": {k: dose.get(k) for k in ("dose_rate", "lr_median", "n_lr", "source", "n_epochs",
                                              "updates_per_env_step", "num_timesteps", "upto", "error")}}


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    return None if not xs or any(x is None for x in xs) else math.fsum(xs) / len(xs)  # type: ignore[arg-type]


def cell_meters(mine: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """A cell's REPORTED meters: the MEAN of its runs' figures, window by window (``None`` if any run lacks one), the
    bounds' touches summed, the step time's exact mean (``None`` if any run's cannot be read), the dose's mean."""
    meters: Dict[str, Any] = {}
    for m in METERS:
        rs = [r["meters"][m] for r in mine]
        meters[m] = {"kl_window_median_mean": _mean([r["kl_window_median"] for r in rs]),
                     "by_window_mean": [_mean(list(w)) for w in zip(*(r["by_window"] for r in rs))]}
    sts = [r["step_time"] for r in mine]
    st = None if any("error" in x for x in sts) else cell_mean_exact([x["sec_per_1M_steps_exact"] for x in sts])
    return {"meters": meters,
            "lr_bounds": {k: sum(r["lr_bounds"][k] for r in mine)
                          for k in ("at_floor", "at_ceiling", "below_floor", "above_ceiling")},
            "sec_per_1M_steps_mean": None if st is None else float(st),
            "sec_per_1M_steps_mean_exact": None if st is None else exact(st),
            "dose_rate_mean": _mean([r["dose"]["dose_rate"] for r in mine])}


def fresh_runs(arm_file: Path, models_dir: Optional[Path], replace: Optional[int]) -> Tuple[Any, Dict[str, List[Any]]]:
    """``(the arm, {side: [its runs]})`` from a card's arm file; the fresh side first."""
    from main.b1 import fresh as F

    try:
        spec = F.load_arm(arm_file)
        sides = {"fresh": list(spec.fresh), "banked": list(F.banked_runs(spec, replace))}
    except (F.FreshError, K.LookError) as e:
        raise ReportError(f"{arm_file}: {e}") from e
    if models_dir is None:
        from utils.paths import main_models_dir

        models_dir = main_models_dir()
    if models_dir is None:
        raise ReportError("there is no models directory here; pass --models-dir")
    return spec, {side: [(r, Path(models_dir) / r.run) for r in runs] for side, runs in sides.items()}


def fresh_report(spec: Any, sides: Mapping[str, Sequence[Tuple[Any, Path]]], look: str,
                 replace: Optional[int] = None) -> Dict[str, Any]:
    """The fresh cell's record for one look (schema :data:`FRESH_SCHEMA`; module docstring). No run is dropped."""
    if look not in LOOKS:
        raise ReportError(f"look {look!r} is not one of {sorted(LOOKS)}")
    bases = cell_bases(spec.lever, spec.declared_keys)
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for side, runs in sides.items():
        for run, run_dir in runs:
            try:
                if not run_dir.is_dir():
                    raise ReportError(f"{run.run}: no run directory at {run_dir}")
                loaded = load_tags(run_dir, _wanted)
                s = series_of(loaded)
                row = run_row(run_dir, LOOKS[look], s, named=(side, run.seed), base=bases[side])
                median, n = kl_window_median(run.run, s["steps"], s["kl"])
                to_look = row["rows_to_look"]
                rows.append({**row, "side": side, "ladder_base": bases[side], "kl_window_median": median,
                             "kl_window_rows": n, "rung_histogram": rung_histogram(s["lr"][:to_look], bases[side]),
                             **run_meters(run_dir, loaded[0], row, bases[side], LOOKS[look], s["steps"][:to_look],
                                          s["lr"][:to_look])})
            except ReportError as e:
                problems.append(str(e))
    n_runs = sum(len(runs) for runs in sides.values())
    if problems:
        raise ReportError(f"{len(problems)} of {n_runs} runs cannot be read at {look}, and no run is dropped: "
                          + "; ".join(problems))
    cells: Dict[str, Dict[str, Any]] = {}
    for side in sides:
        mine = [r for r in rows if r["side"] == side]
        rungs: Dict[int, int] = {}
        for r in mine:
            for k, n in r["rung_histogram"]["rungs"].items():
                rungs[int(k)] = rungs.get(int(k), 0) + n
        cells[side] = {"runs": [r["run"] for r in mine], "ladder_base": bases[side],
                       "kl_window_median": statistics.median(r["kl_window_median"] for r in mine),
                       "rung_histogram": {"rungs": {str(k): rungs[k] for k in sorted(rungs)},
                                          "off": sum(r["rung_histogram"]["off"] for r in mine)},
                       **cell_meters(mine)}
    figure = cells["fresh"]["kl_window_median"]
    f_st, b_st = (cells[k]["sec_per_1M_steps_mean_exact"] for k in ("fresh", "banked"))
    rho = None if f_st is None or b_st is None else Fraction(f_st) / Fraction(b_st)
    f_d, b_d = (cells[k]["dose_rate_mean"] for k in ("fresh", "banked"))
    return {"schema": FRESH_SCHEMA, "card": spec.card, "arm_sha256": spec.sha256, "prereg": spec.prereg,
            "look": look, "look_step": LOOKS[look], "second_half_from": LOOKS[look] / 2,
            "replaced_seed": replace, "replaced_fresh_seed": spec.replaced_fresh,
            "ladder": {"factor": LADDER_FACTOR, "base": bases}, "tags": [LR_TAG, KL_TAG],
            "kl_window": list(KL_WINDOW), "kl_band": list(KL_BAND), "kl_band_edge": KL_BAND_EDGE,
            "n_runs": len(rows), "runs": rows, "cells": cells, "figure": figure, "label": movement_label(figure),
            "meter_window": METER_WINDOW, "step_time_window": list(STEP_TIME_WINDOW),
            "step_time_ratio": None if rho is None else float(rho),
            "step_time_ratio_exact": None if rho is None else exact(rho),
            "dose_rate_ratio": None if not (f_d is not None and b_d) else f_d / b_d}


def _histogram(h: Mapping[str, Any]) -> str:
    return "  ".join([*(f"{int(k):+d}:{n}" for k, n in h["rungs"].items()), f"off:{h['off']}"])


def render_fresh(doc: Mapping[str, Any]) -> str:
    """The fresh cell's record as text: one row a run, each cell's figure and histogram, then the label's line."""
    lo, hi = doc["kl_window"]
    lines = [f"Fresh-cell learning-rate report, card {doc['card']} at the {doc['look']} look (REPORTED; "
             f"{doc['n_runs']} runs; rungs over every row up to each run's first row at or after "
             f"{doc['look_step']:,}; KL median over rows with {lo:,} <= step <= {hi:,})",
             f"{'run':<16}{'side':<7}{'base':>8} {'rows':>5} {'LR at look':>11} {'rung':>5} {'geomean LR 2nd':>15} "
             f"{'KL median':>10} {'rows':>5}  rung histogram to the look"]
    for r in doc["runs"]:
        k = "off" if r["rung"] is None else f"{r['rung']:+d}"
        lines.append(f"{r['run']:<16}{r['side']:<7}{r['ladder_base']:>8.1e} {r['rows_to_look']:>5} "
                     f"{r['lr_at_look']:>11.4e} {k:>5} {r['lr_geomean_second_half']:>15.4e} "
                     f"{r['kl_window_median']:>10.6f} {r['kl_window_rows']:>5}  {_histogram(r['rung_histogram'])}")
        if r["rung_histogram"]["off_rates"]:
            lines.append(f"    {r['run']} off its ladder: "
                         + ", ".join(f"{n} rows at {rate:.6e}" for rate, n in r["rung_histogram"]["off_rates"]))
        if r["superseded_rows"] or r["rows_lr_only"] or r["rows_kl_only"]:
            lines.append(f"    {r['run']}: {r['superseded_rows']} superseded rows, {r['rows_lr_only']} with the "
                         f"rate alone, {r['rows_kl_only']} with the KL alone")
    for side, c in doc["cells"].items():
        lines.append(f"cell {side:<7} base {c['ladder_base']:.1e}  median of its runs' KL medians "
                     f"{c['kl_window_median']:.6f}  rungs {_histogram(c['rung_histogram'])}")
    band = f"{doc['kl_band'][0]} + {doc['kl_band_edge']:g} < figure < {doc['kl_band'][1]} - {doc['kl_band_edge']:g}"
    lines.append(f"fresh cell's figure {doc['figure']:.6f}; inside the band ({band}): "
                 + ("yes, no label" if doc["label"] is None else f"NO -> {doc['label']}"))
    return "\n".join(lines + render_meters(doc))


def _g(x: Optional[float], width: int = 9) -> str:
    return f"{'-':>{width}}" if x is None else f"{x:>{width}.4g}"


def render_meters(doc: Mapping[str, Any]) -> List[str]:
    """The REPORTED meters as text: each meter by window (a run a row, then each cell's mean), the bounds, the dose
    and the step time with their ratios."""
    w = int(doc["meter_window"])
    n = int(doc["look_step"]) // w
    lo, hi = doc["kl_window"]
    head = "".join(f"{f'{k * w // 10**6}-{(k + 1) * w // 10**6}M':>9}" for k in range(n))
    lines = [f"REPORTED meters, never deciding: the median of a run's rows in each {w:,}-step window below the look "
             f"('-': no row), and over {lo:,} <= step <= {hi:,}; a cell's line is the MEAN of its runs"]
    for m, (tag, sign) in METERS.items():
        what = f"-{tag}" if sign < 0 else tag
        epochs = sorted({k for r in doc["runs"] for k in r["meters"][m].get("epochs", [])})
        lines.append(f"{m} ({what}{f'; last epochs seen {epochs}' if m == 'clip_fraction_last_epoch' else ''})")
        lines.append(f"  {'':<16}{head}{'10M-50M':>10}")
        for r in doc["runs"]:
            mm = r["meters"][m]
            lines.append(f"  {r['run']:<16}{''.join(_g(x) for x in mm['by_window'])}{_g(mm['kl_window_median'], 10)}")
        for side, c in doc["cells"].items():
            cm = c["meters"][m]
            lines.append(f"  {'cell ' + side:<16}{''.join(_g(x) for x in cm['by_window_mean'])}"
                         f"{_g(cm['kl_window_median_mean'], 10)}")
    lines.append("LR floor / ceiling, rows up to the look (at = within float error; beyond = past the bound):")
    for r in doc["runs"]:
        b = r["lr_bounds"]
        lines.append(f"  {r['run']:<16} floor {b['floor']:.1e}: {b['at_floor']} at (first {b['first_at_floor']}), "
                     f"{b['below_floor']} below; ceiling {b['ceiling']:.1e}: {b['at_ceiling']} at (first "
                     f"{b['first_at_ceiling']}), {b['above_ceiling']} above")
    lines.append("dose_rate to the look (main.dose, the rows at or before each run's look checkpoint):")
    for r in doc["runs"]:
        d = r["dose"]
        lines.append(f"  {r['run']:<16} " + (f"ERROR {d['error']}" if d.get("error") else
                                             f"{d['dose_rate']:.4g} (lr median {d['lr_median']:.4g} over {d['n_lr']} "
                                             f"{d['source']} rows, {d['n_epochs']} epochs, to step {d['num_timesteps']})"))
    lines.append("  cell means: " + ", ".join(f"{k} {_g(c['dose_rate_mean'], 0)}" for k, c in doc["cells"].items())
                 + f"; fresh / banked {_g(doc['dose_rate_ratio'], 0)}")
    slo, shi = doc["step_time_window"]
    lines.append(f"step time over {slo:,} <= step <= {shi:,} (the card check's formula; s per 1M steps):")
    for r in doc["runs"]:
        t = r["step_time"]
        lines.append(f"  {r['run']:<16} " + (f"ERROR {t['error']}" if "error" in t else
                                             f"{t['sec_per_1M_steps']:8.2f}  (update {t['median_update_s']:.3f} s + "
                                             f"collect {t['median_collect_s']:.3f} s per {t['median_dsteps']:,.0f} "
                                             f"steps, {t['rows']} rows, update share {t['update_share']:.3f}; "
                                             f"train_ms alone {_g(t['train_ms_sec_per_1M_steps'], 0)})"))
    lines.append("  cell means: " + ", ".join(f"{k} {_g(c['sec_per_1M_steps_mean'], 0)}"
                                              for k, c in doc["cells"].items())
                 + f"; fresh / banked (rho) {_g(doc['step_time_ratio'], 0)}"
                 + (f" = {doc['step_time_ratio_exact']}" if doc["step_time_ratio_exact"] and
                    len(doc["step_time_ratio_exact"]) <= 60 else ""))
    return lines


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
    ap.add_argument("--fresh-arm", help="a card's fresh-arm file: report its fresh cell beside its banked cell")
    ap.add_argument("--replace", type=int, help="with --fresh-arm: the REGISTERED B1 seed B1's spare replaced")
    a = ap.parse_args(argv)
    try:
        if a.fresh_arm and (a.runs or a.displacement):
            raise ReportError("--fresh-arm names its own runs and reports no displacement: drop --runs and "
                              "--displacement")
        if a.runs and a.models_dir:
            raise ReportError("--runs and --models-dir do not go together")
        if a.fresh_arm:
            spec, sides = fresh_runs(Path(a.fresh_arm), Path(a.models_dir) if a.models_dir else None, a.replace)
            runs = [d for pairs in sides.values() for _r, d in pairs]
            out = check_out(Path(a.out), runs) if a.out else None
            doc = fresh_report(spec, sides, a.look, a.replace)
            text = render_fresh(doc)
        else:
            if a.replace is not None:
                raise ReportError("--replace goes with --fresh-arm; BOUNDARY-1's own report names a spare run by --runs")
            runs = ([Path(r) for r in a.runs] if a.runs
                    else default_runs(Path(a.models_dir) if a.models_dir else None))
            out = check_out(Path(a.out), runs) if a.out else None
            doc = report(runs, a.look, a.displacement)
            text = render(doc)
    except ReportError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    print(text)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
