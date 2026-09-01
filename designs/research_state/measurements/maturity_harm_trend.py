#!/usr/bin/env python3
"""M7 — HARM vs PARENT MATURITY: pure re-analysis of the distillability-index battery.

The owner's question: *does distillation hurt LESS when the parent is more mature?*

This script does not train, does not battle, does not load a model. It re-reads the
committed cell + curve artifact of the 2026-08-28 distillability-index probe
(`distillability_index_gen_2026-08-28.json`) and extracts the HARM axis explicitly, at
both step sizes, with the zero-content control beside the with-content cells.

Three matching regimes, because "harm" is only defined relative to what was bought:

  * matched STEPS      — identical optimizer work. The ONLY regime in which the
                         zero-content control is comparable (it buys no absorption by
                         construction, so it has no matched-absorption point).
  * matched ABSORPTION  — harm at the point held-out teacher agreement first reaches an
                         absolute level A*.
  * matched GAIN        — harm at the point absorption has risen d above its own start.

The last two are NORMALISED meters and are therefore scored against the correlation they
MANUFACTURE under an age-invariant-harm null, never against zero — the binding method rule
from `substrate_hypothesis_2026-08-31.md` §2.3.

Run:
    nice -n 15 python maturity_harm_trend.py --out maturity_harm_trend_2026-08-31
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import math
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent / "distillability_index_gen_2026-08-28.json"

# The six ages of the rev-1 lineage, in order, with their nominal step counts.
AGES = [
    ("02M", 2_000_016),
    ("06M", 6_000_000),
    ("12M", 12_000_000),
    ("18M", 18_000_000),
    ("24M", 24_000_000),
    ("25M_final", 24_988_992),
]
# The gen-17 lineage continues into `ai_v9_25_E4_baitbot_0822` (a gate experiment forked off the
# same base, CURRENT architecture). A SPLICE, not one run — see the .md §6.2.
AGES_EXT = AGES + [("30M", 30_000_000), ("36M", 36_000_000), ("42M", 42_000_000)]
CONTROL_AGES = ["02M", "12M", "25M_final"]  # the committed control's coverage

ARMS = {
    # arm key -> (cell prefix, lr, lineage, seeds present)
    "lr3e4_ancestor": ("age_", 3e-4, "rev-1 (ancestor of the teacher)", [1, 2]),
    "lr1e4_ancestor": ("lr1e4_", 1e-4, "rev-1 (ancestor of the teacher)", [1, 2]),
    "lr3e4_ancestryfree": ("ctrl17_", 3e-4, "gen-17 (shares no weights)", [1, 2]),
    "lr3e4_zerocontent": ("ctrlself_", 3e-4, "rev-1, targets = own argmax", [1]),
    # NEW (this probe) — the committed battery's two "not run (budget)" gaps.
    "lr1e4_zerocontent": ("ctrlself1e4_", 1e-4, "rev-1, targets = own argmax", [1]),
}
# Cells whose prefix differs but which EXTEND an existing arm's age axis.
ARM_EXTRA_PREFIX = {"lr3e4_ancestryfree": "ctrl17x_"}


def arm_ages(arm: str) -> list[tuple[str, int]]:
    return AGES_EXT if arm in ARM_EXTRA_PREFIX else AGES

# Harm meters. sign = +1 when a LARGER value means MORE harm.
HARM_METERS = {
    "off_kl": (+1, "off-slice KL(now || original) — drift in nats"),
    "off_disagree": (+1, "1 - off-slice top-1 agreement with original"),
    "off_value_mad": (+1, "mean |dV| off-slice, on a +/-12 value scale"),
    "off_value_decorr": (+1, "1 - corr(V_now, V_original) off-slice"),
}

MATCH_STEPS = [1, 4, 12, 32, 84, 135, 400]
MATCH_ABS = ["0.60", "0.70", "0.78", "0.80"]
MATCH_GAIN = ["0.03", "0.05", "0.10"]


# ---------------------------------------------------------------- statistics


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rho. None when fewer than 3 usable pairs or either side is constant."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    rx = _rank([p[0] for p in pairs])
    ry = _rank([p[1] for p in pairs])
    n = len(pairs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


_MC_DRAWS = 200_000
_MC_SEED = 20260831


def spearman_exact_p(x: list[float], y: list[float]) -> tuple[float | None, int]:
    """Two-sided permutation p for Spearman: EXACT for n <= 8, seeded Monte-Carlo above.

    Spearman(x, y) is Pearson on RANKS, and permuting y permutes its rank vector without
    changing the multiset — so the whole null is a dot product against a fixed centred rank
    vector. That makes the exact enumeration cheap at n <= 8 (<= 40,320) and lets n = 9+ use a
    seeded 200k-draw Monte-Carlo instead of returning nothing.

    The p FLOOR is the honest small-n caveat and is why it is returned alongside the count:
    2/720 = 0.0028 at n = 6, and **2/6 = 0.333 at n = 3**, i.e. an n = 3 ordering can never be
    significant no matter how clean it looks.
    """
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None, 0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    obs = spearman(xs, ys)
    if obs is None:
        return None, 0
    n = len(xs)
    rx = _rank(xs)
    ry = _rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cx = [v - mx for v in rx]
    cy = [v - my for v in ry]
    dx = math.sqrt(sum(v * v for v in cx))
    dy = math.sqrt(sum(v * v for v in cy))
    if dx == 0 or dy == 0:
        return None, 0
    denom = dx * dy
    target = abs(obs) - 1e-12

    if n <= 8:
        hits = tot = 0
        for perm in itertools.permutations(cy):
            tot += 1
            if abs(sum(a * b for a, b in zip(cx, perm))) / denom >= target:
                hits += 1
        return hits / tot, tot

    rng = random.Random(_MC_SEED)
    buf = list(cy)
    hits = 0
    for _ in range(_MC_DRAWS):
        rng.shuffle(buf)
        if abs(sum(a * b for a, b in zip(cx, buf))) / denom >= target:
            hits += 1
    # +1/+1 (Davison-Hinkley): an MC p is never reported as exactly 0.
    return (hits + 1) / (_MC_DRAWS + 1), _MC_DRAWS


def interp_curve(curve: list[dict], step: float, field: str) -> float | None:
    """Read `field` off a curve at `step`, linearly in log(1+step) between grid points."""
    pts = [(p["step"], p.get(field)) for p in curve if p.get(field) is not None]
    if not pts:
        return None
    pts.sort()
    if step <= pts[0][0]:
        return pts[0][1]
    if step >= pts[-1][0]:
        return pts[-1][1]
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        if s0 <= step <= s1:
            if s1 == s0:
                return v0
            w = (math.log1p(step) - math.log1p(s0)) / (math.log1p(s1) - math.log1p(s0))
            return v0 + w * (v1 - v0)
    return pts[-1][1]


def harm_from_point(pt: dict[str, Any]) -> dict[str, float | None]:
    """Map a raw curve/index point onto the four harm meters."""
    out: dict[str, float | None] = {}
    out["off_kl"] = pt.get("off_kl")
    ag = pt.get("off_agree")
    out["off_disagree"] = None if ag is None else 1.0 - ag
    out["off_value_mad"] = pt.get("off_value_mad")
    co = pt.get("off_value_corr")
    out["off_value_decorr"] = None if co is None else 1.0 - co
    return out


# ---------------------------------------------------------------- extraction


def load() -> dict:
    with SRC.open() as fh:
        data = json.load(fh)
    merge_new_cells(data)
    return data


def _derive_cell(cell: str, rec: dict) -> dict:
    """Rebuild the committed artifact's cell-level fields from a raw producer `curve`.

    Re-derived here rather than by running the producer's `aggregate`, which would overwrite
    the committed 2026-08-28 artifact. Crossing steps use the same linear-in-log(1+step)
    interpolation, and a level the cell never reaches stays **None** — never interpolated.
    """
    curve = rec["curve"]
    a = [(p["step"], p["on_held_agree"]) for p in curve]
    a0 = a[0][1]
    a_max = max(v for _s, v in a)
    out: dict[str, Any] = {
        "cell": cell,
        "student": rec.get("student"),
        "teacher_set": rec.get("teacher_set"),
        "seed": rec.get("probe_seed"),
        "lr": rec.get("lr"),
        "n_steps": rec.get("n_steps"),
        "a0": a0,
        "a_max": a_max,
        "gain_max": a_max - a0,
        "step1_shock_kl": next((p["off_kl"] for p in curve if p["step"] == 1), None),
        "final_off_kl": curve[-1]["off_kl"],
        "final_off_agree": curve[-1]["off_agree"],
        "final_off_value_mad": curve[-1]["off_value_mad"],
    }

    def crossing(target: float) -> dict | None:
        prev = None
        for p in curve:
            if p["on_held_agree"] >= target:
                if prev is None:
                    s = float(p["step"])
                else:
                    lo, hi = prev, p
                    span = hi["on_held_agree"] - lo["on_held_agree"]
                    w = 0.0 if span == 0 else (target - lo["on_held_agree"]) / span
                    s = lo["step"] + w * (hi["step"] - lo["step"])
                return {
                    "step": s,
                    "exact": False,
                    "off_kl": interp_curve(curve, s, "off_kl"),
                    "off_agree": interp_curve(curve, s, "off_agree"),
                    "off_value_mad": interp_curve(curve, s, "off_value_mad"),
                    "off_value_corr": interp_curve(curve, s, "off_value_corr"),
                }
            prev = p
        return None

    for lvl in MATCH_ABS:
        out[f"idx_abs_{lvl}"] = crossing(float(lvl))
    for lvl in MATCH_GAIN:
        out[f"idx_gain_{lvl}"] = crossing(a0 + float(lvl))
    return out


def merge_new_cells(data: dict) -> None:
    """Fold this probe's NEW `results/<cell>.json` files into the committed structures.

    `ctrl17x_*` cells (the E4 age-axis extension) are ALIASED onto the `ctrl17_` prefix, so the
    ancestry-free arm simply gains three more ages. The splice is declared in the .md, not hidden
    here — this alias is a key rename, and `arm_ages()` is what says the arm now spans 42M.
    """
    rdir = Path(__file__).resolve().parent / "results"
    if not rdir.is_dir():
        data.setdefault("new_cells", [])
        return
    added = []
    for f in sorted(rdir.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        if "curve" not in rec:
            continue
        cell = rec.get("cell", f.stem)
        key = cell.replace("ctrl17x_", "ctrl17_", 1) if cell.startswith("ctrl17x_") else cell
        if key in data["curves"]:
            continue  # never shadow a committed cell
        data["curves"][key] = rec["curve"]
        data["cells"][key] = _derive_cell(key, rec)
        data.setdefault("meta", {})[key] = {
            "student": rec.get("student"),
            "teacher": rec.get("teacher"),
            "teacher_set": rec.get("teacher_set"),
            "probe_seed": rec.get("probe_seed"),
            "lr": rec.get("lr"),
            "wall_s": rec.get("wall_s"),
            "dropped_kwargs": rec.get("dropped_kwargs"),
            "source": "M7 (2026-08-31), same instrument + bit-identical state set",
        }
        added.append(key)
    data["new_cells"] = added


def cell_key(prefix: str, age: str, seed: int) -> str:
    return f"{prefix}{age}__s{seed}"


def harm_at_steps(data: dict, prefix: str, age: str, seed: int) -> dict[int, dict]:
    key = cell_key(prefix, age, seed)
    curve = data["curves"].get(key)
    if curve is None:
        return {}
    out = {}
    for s in MATCH_STEPS:
        pt = {
            f: interp_curve(curve, s, f)
            for f in ("off_kl", "off_agree", "off_value_mad", "off_value_corr")
        }
        pt["on_held_agree"] = interp_curve(curve, s, "on_held_agree")
        h = harm_from_point(pt)
        h["on_held_agree"] = pt["on_held_agree"]
        out[s] = h
    return out


def harm_at_index(data: dict, prefix: str, age: str, seed: int, kind: str, lvl: str):
    key = cell_key(prefix, age, seed)
    cell = data["cells"].get(key)
    if cell is None:
        return None
    idx = cell.get(f"idx_{kind}_{lvl}")
    if idx is None:
        return None
    h = harm_from_point(idx)
    h["step"] = idx.get("step")
    return h


# ---------------------------------------------------------------- the nulls


def manufactured_null(
    data: dict, arm: str, prefix: str, seed: int, kind: str, lvl: str, meter: str
) -> dict[str, Any]:
    """The correlation a matched-ABSORPTION / matched-GAIN meter manufactures under an
    AGE-INVARIANT-HARM null.

    Construction: pool the arm's harm-vs-step curves into ONE age-independent curve
    Hbar(s) (mean over ages at each grid step). Then evaluate Hbar at the step each age
    ACTUALLY crossed the target. Any resulting rho-vs-age is pure arithmetic: it is what
    "harm depends only on optimizer steps, never on age" produces on this data.

    A matched-absorption reading must be scored against THIS number, not against zero
    (substrate_hypothesis_2026-08-31 §2.3).
    """
    field = {
        "off_kl": "off_kl",
        "off_disagree": "off_agree",
        "off_value_mad": "off_value_mad",
        "off_value_decorr": "off_value_corr",
    }[meter]
    curves = []
    for age, _ in arm_ages(arm):
        c = data["curves"].get(cell_key(prefix, age, seed))
        if c:
            curves.append(c)
    if len(curves) < 3:
        return {"null_rho": None, "reason": "fewer than 3 curves in arm"}
    grid = sorted({p["step"] for c in curves for p in c})

    def hbar(s: float) -> float | None:
        vals = [interp_curve(c, s, field) for c in curves]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        m = sum(vals) / len(vals)
        return (1.0 - m) if meter in ("off_disagree", "off_value_decorr") else m

    steps, null_h, ages_used = [], [], []
    for age, nsteps in arm_ages(arm):
        cell = data["cells"].get(cell_key(prefix, age, seed))
        if not cell:
            continue
        idx = cell.get(f"idx_{kind}_{lvl}")
        if idx is None or idx.get("step") is None:
            continue
        ages_used.append(nsteps)
        steps.append(idx["step"])
        null_h.append(hbar(idx["step"]))
    rho = spearman(ages_used, null_h) if len(ages_used) >= 3 else None
    return {
        "null_rho": rho,
        "n": len(ages_used),
        "crossing_steps": dict(zip([a for a, _ in arm_ages(arm)], steps)) if steps else {},
        "grid": grid[:1] and None,
        "note": "age-invariant-harm null: Hbar(s) pooled over ages, read at each age's own crossing step",
    }


# ---------------------------------------------------------------- assembly


def build(data: dict) -> dict:
    out: dict[str, Any] = {
        "source": SRC.name,
        "question": "does the HARM a distillation step does fall as the parent matures?",
        "harm_meters": {k: v[1] for k, v in HARM_METERS.items()},
        "arms": {},
        "matched_steps": {},
        "matched_absorption": {},
        "matched_gain": {},
        "net_of_control": {},
        "trend": {},
        "nulls": {},
        "new_cells": data.get("new_cells", []),
        "new_cell_meta": {
            k: data["meta"][k] for k in data.get("new_cells", []) if k in data.get("meta", {})
        },
    }
    for arm, (prefix, lr, lineage, seeds) in ARMS.items():
        out["arms"][arm] = {"prefix": prefix, "lr": lr, "lineage": lineage, "seeds": seeds}

    # --- matched STEPS -----------------------------------------------------
    for arm, (prefix, lr, _lin, seeds) in ARMS.items():
        out["matched_steps"][arm] = {}
        for seed in seeds:
            per_age = {}
            for age, _n in arm_ages(arm):
                h = harm_at_steps(data, prefix, age, seed)
                if h:
                    per_age[age] = {str(s): h[s] for s in MATCH_STEPS}
            if per_age:
                out["matched_steps"][arm][f"s{seed}"] = per_age

    # --- matched ABSORPTION / GAIN ----------------------------------------
    for tag, kind, levels in (
        ("matched_absorption", "abs", MATCH_ABS),
        ("matched_gain", "gain", MATCH_GAIN),
    ):
        for arm, (prefix, lr, _lin, seeds) in ARMS.items():
            out[tag][arm] = {}
            for seed in seeds:
                per_lvl: dict[str, Any] = {}
                for lvl in levels:
                    per_age = {}
                    for age, _n in arm_ages(arm):
                        h = harm_at_index(data, prefix, age, seed, kind, lvl)
                        per_age[age] = h  # None is meaningful: MISS, never interpolated
                    per_lvl[lvl] = per_age
                out[tag][arm][f"s{seed}"] = per_lvl

    # --- NET of the zero-content control (matched steps only) --------------
    PAIRS = {"lr3e4_ancestor": "lr3e4_zerocontent", "lr1e4_ancestor": "lr1e4_zerocontent"}
    for arm, ctrl_arm in PAIRS.items():
        czp = ARMS[ctrl_arm][0]
        prefix = ARMS[arm][0]
        out["net_of_control"][arm] = {}
        for seed in (1,):
            per_age = {}
            for age, _n in arm_ages(arm):
                hw = harm_at_steps(data, prefix, age, seed)
                hc = harm_at_steps(data, czp, age, seed)
                if not hw or not hc:
                    continue
                per_age[age] = {}
                for s in MATCH_STEPS:
                    row = {}
                    for m in HARM_METERS:
                        a, b = hw[s].get(m), hc[s].get(m)
                        row[m] = None if (a is None or b is None) else a - b
                        row[m + "__with"] = a
                        row[m + "__zero"] = b
                        row[m + "__content_share"] = (
                            None if (a in (None, 0) or b is None) else 1.0 - b / a
                        )
                    row["absorption_gain_with"] = (
                        None
                        if hw[s].get("on_held_agree") is None
                        else hw[s]["on_held_agree"] - hw[1].get("on_held_agree", 0.0)
                    )
                    per_age[age][str(s)] = row
            out["net_of_control"][arm][f"s{seed}"] = per_age

    # --- TRENDS ------------------------------------------------------------
    def ages_vec(arm: str, seed: int, getter) -> tuple[list[float], list]:
        xs, ys = [], []
        for age, nsteps in arm_ages(arm):
            v = getter(age, seed)
            if v is not None:
                xs.append(float(nsteps))
                ys.append(v)
        return xs, ys

    for arm, (prefix, lr, _lin, seeds) in ARMS.items():
        out["trend"][arm] = {}
        out["nulls"][arm] = {}
        for seed in seeds:
            t: dict[str, Any] = {}
            # matched steps
            for s in MATCH_STEPS:
                for m in HARM_METERS:
                    xs, ys = ages_vec(
                        arm,
                        seed,
                        lambda a, sd, s=s, m=m: (harm_at_steps(data, prefix, a, sd) or {})
                        .get(s, {})
                        .get(m),
                    )
                    if len(xs) >= 3:
                        rho = spearman(xs, ys)
                        p, nperm = spearman_exact_p(xs, ys)
                        t[f"steps{s}__{m}"] = {
                            "rho": rho,
                            "p_exact": p,
                            "n": len(xs),
                            "perms": nperm,
                            "null_rho": 0.0,
                        }
            # matched absorption / gain, each with its manufactured null
            for kind, levels in (("abs", MATCH_ABS), ("gain", MATCH_GAIN)):
                for lvl in levels:
                    for m in HARM_METERS:
                        xs, ys = ages_vec(
                            arm,
                            seed,
                            lambda a, sd, k=kind, l=lvl, m=m: (
                                harm_at_index(data, prefix, a, sd, k, l) or {}
                            ).get(m),
                        )
                        if len(xs) >= 3:
                            rho = spearman(xs, ys)
                            p, nperm = spearman_exact_p(xs, ys)
                            nl = manufactured_null(data, arm, prefix, seed, kind, lvl, m)
                            t[f"{kind}{lvl}__{m}"] = {
                                "rho": rho,
                                "p_exact": p,
                                "n": len(xs),
                                "perms": nperm,
                                "null_rho": nl.get("null_rho"),
                                "null_n": nl.get("n"),
                            }
            out["trend"][arm][f"s{seed}"] = t

    # --- the ratio meter's manufactured null (efficiency = gain / harm) ----
    # Reported explicitly because `eff` appears in the source artifact and a reader will
    # reach for it: under "harm is age-invariant", eff inherits gain's own rho exactly.
    eff_null = {}
    for arm, (prefix, lr, _lin, seeds) in ARMS.items():
        for seed in seeds:
            xs, ys = [], []
            for age, nsteps in arm_ages(arm):
                cell = data["cells"].get(cell_key(prefix, age, seed))
                if cell and cell.get("gain_max") is not None:
                    xs.append(float(nsteps))
                    ys.append(cell["gain_max"])
            if len(xs) >= 3:
                eff_null[f"{arm}__s{seed}"] = {
                    "rho_gain_max_vs_age": spearman(xs, ys),
                    "note": "eff = gain/harm; under age-invariant harm, rho(eff) == this number, NOT 0",
                }
    out["nulls"]["efficiency_ratio"] = eff_null

    # --- seed reproduction (the only replication this artifact carries) ----
    rep = {}
    for arm, (prefix, lr, _lin, seeds) in ARMS.items():
        if len(seeds) < 2:
            continue
        diffs = []
        for age, _n in arm_ages(arm):
            h1 = harm_at_steps(data, prefix, age, 1)
            h2 = harm_at_steps(data, prefix, age, 2)
            if not h1 or not h2:
                continue
            for s in MATCH_STEPS:
                for m in HARM_METERS:
                    a, b = h1[s].get(m), h2[s].get(m)
                    if a is not None and b is not None:
                        diffs.append((m, s, abs(a - b)))
        if diffs:
            per_m = {}
            for m in HARM_METERS:
                vals = sorted(d for mm, _s, d in diffs if mm == m)
                if vals:
                    per_m[m] = {
                        "median_abs_seed_diff": vals[len(vals) // 2],
                        "max_abs_seed_diff": vals[-1],
                        "n": len(vals),
                    }
            rep[arm] = per_m
    out["seed_reproduction"] = rep
    return out


# ---------------------------------------------------------------- rendering


def fmt(v, nd=3):
    if v is None:
        return "MISS"
    return f"{v:.{nd}f}"


def _rho_cell(pts: list[tuple[float, Any]]) -> str:
    """`rho (p=…)` for an (age, value) series, or MISS below 3 points — never interpolated."""
    pts = [(x, y) for x, y in pts if y is not None]
    if len(pts) < 3:
        return "MISS"
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    rho = spearman(xs, ys)
    p, _ = spearman_exact_p(xs, ys)
    return "MISS" if rho is None else f"{rho:+.2f} (p={fmt(p)})"


def render(res: dict) -> str:
    L = []
    A = L.append
    A("## Harm at MATCHED OPTIMIZER STEPS (the only regime the zero-content control fits)\n")
    for arm in ("lr3e4_ancestor", "lr1e4_ancestor", "lr3e4_ancestryfree", "lr3e4_zerocontent"):
        info = ARMS[arm]
        A(f"\n### {arm} — lr {info[1]:g}, {info[2]}\n")
        A("| age | KL@1 | KL@32 | KL@135 | KL@400 | disagree@32 | disagree@400 | \\|dV\\|@400 |")
        A("|---|---|---|---|---|---|---|---|")
        for age, _n in arm_ages(arm):
            row = res["matched_steps"][arm].get("s1", {}).get(age)
            if not row:
                continue
            A(
                f"| {age} | {fmt(row['1']['off_kl'])} | {fmt(row['32']['off_kl'])} | "
                f"{fmt(row['135']['off_kl'])} | {fmt(row['400']['off_kl'])} | "
                f"{fmt(row['32']['off_disagree'])} | {fmt(row['400']['off_disagree'])} | "
                f"{fmt(row['400']['off_value_mad'], 2)} |"
            )
    # --- the doc's zero-content-control table, both step sizes -------------
    A("\n## Zero-content control, both step sizes (doc table 2.4)\n")
    A("| age | lr | KL@1 | KL@32 | KL@135 | KL@400 | disagree@400 |")
    A("|---|---|---|---|---|---|---|")
    for arm, lr in (("lr3e4_zerocontent", "3e-4"), ("lr1e4_zerocontent", "1e-4")):
        for age, _n in arm_ages(arm):
            row = res["matched_steps"].get(arm, {}).get("s1", {}).get(age)
            if not row:
                continue
            A(
                f"| {age} | {lr} | {fmt(row['1']['off_kl'])} | {fmt(row['32']['off_kl'])} | "
                f"{fmt(row['135']['off_kl'])} | {fmt(row['400']['off_kl'])} | "
                f"{fmt(row['400']['off_disagree'])} |"
            )

    # --- the doc's NET-of-control table, both step sizes -------------------
    for arm, lr in (("lr3e4_ancestor", "3e-4"), ("lr1e4_ancestor", "1e-4")):
        net = res["net_of_control"].get(arm, {}).get("s1", {})
        if not net:
            continue
        A(f"\n## CONTENT-attributable harm = total - zero-content, lr {lr} (doc table 3)\n")
        A("| matched step | " + " | ".join(sorted(net)) + " | direction |")
        A("|---" * (len(net) + 2) + "|")
        for s in ("1", "32", "135", "400"):
            vals = [net[a].get(s, {}).get("off_kl") for a in sorted(net)]
            clean = [v for v in vals if v is not None]
            direction = "—"
            if len(clean) >= 3:
                direction = "RISES" if clean[-1] > clean[0] else "falls"
            A(f"| {s} | " + " | ".join(fmt(v) for v in vals) + f" | {direction} |")

    # --- THE decisive table: does the age trend live in the optimizer or the content? ---
    A("\n## TOTAL vs CONTROL vs NET — Spearman rho vs age, exact p (doc table 3.2)\n")
    A("| lr | matched step | TOTAL (with content) | CONTROL (zero content) | NET (content only) |")
    A("|---|---|---|---|---|")
    for lr, arm, ctrl in (
        ("3e-4", "lr3e4_ancestor", "lr3e4_zerocontent"),
        ("1e-4", "lr1e4_ancestor", "lr1e4_zerocontent"),
    ):
        for s in ("1", "32", "135", "400"):
            cells = []
            for which in (arm, ctrl):
                ms = res["matched_steps"].get(which, {}).get("s1", {})
                pts = [(n, ms[a][s]["off_kl"]) for a, n in arm_ages(which) if a in ms]
                cells.append(_rho_cell(pts))
            net = res["net_of_control"].get(arm, {}).get("s1", {})
            pts = [
                (n, net[a][s]["off_kl"])
                for a, n in arm_ages(arm)
                if a in net and net[a].get(s, {}).get("off_kl") is not None
            ]
            cells.append(_rho_cell(pts))
            A(f"| {lr} | {s} | " + " | ".join(cells) + " |")

    A("\n## Trend vs age (Spearman rho, seed 1 / seed 2), harm meters\n")
    A("| arm | regime | rho s1 | p s1 | rho s2 | null |")
    A("|---|---|---|---|---|---|")
    for arm in ARMS:
        t1 = res["trend"][arm].get("s1", {})
        t2 = res["trend"][arm].get("s2", {})
        for regime in ("steps32__off_kl", "steps400__off_kl", "abs0.70__off_kl", "gain0.05__off_kl"):
            if regime not in t1:
                continue
            a = t1[regime]
            b = t2.get(regime, {})
            A(
                f"| {arm} | {regime} | {fmt(a['rho'],2)} | {fmt(a.get('p_exact'),3)} | "
                f"{fmt(b.get('rho'),2)} | {fmt(a.get('null_rho'),2)} |"
            )
    return "\n".join(L)


def selftest() -> int:
    """Prove `_derive_cell` reproduces the COMMITTED producer's cell fields exactly.

    The NEW cells' index fields are re-derived here rather than by running the producer's own
    `aggregate`, which would overwrite the committed 2026-08-28 artifact. That shortcut is only
    legitimate if the re-derivation is equivalent wherever both exist — so it is CHECKED, not
    assumed, including agreement on which absorption levels are MISSES (a level a cell never
    reaches must be None on both sides, never interpolated).
    """
    with SRC.open() as fh:
        raw = json.load(fh)
    bad = checked = 0
    for cell, curve in raw["curves"].items():
        ref = raw["cells"].get(cell)
        if not ref:
            continue
        got = _derive_cell(
            cell, {"curve": curve, "probe_seed": ref.get("seed"), "lr": ref.get("lr")}
        )
        for k in ("a0", "a_max", "gain_max", "step1_shock_kl", "final_off_kl", "final_off_agree"):
            a, b = got.get(k), ref.get(k)
            if a is None and b is None:
                continue
            checked += 1
            if a is None or b is None or abs(a - b) > 1e-9:
                print(f"MISMATCH {cell}.{k}: derived={a} committed={b}")
                bad += 1
        for lvl in MATCH_ABS:
            a, b = got.get(f"idx_abs_{lvl}"), ref.get(f"idx_abs_{lvl}")
            checked += 1
            if (a is None) != (b is None):
                print(f"MISS-DISAGREE {cell}.idx_abs_{lvl}")
                bad += 1
    print(f"selftest: {checked} fields over {len(raw['curves'])} committed cells -> {bad} mismatches")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="maturity_harm_trend_2026-08-31")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="run the re-derivation gate, then exit")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if selftest():
        raise SystemExit("re-derivation gate FAILED — refusing to emit an artifact")
    data = load()
    res = build(data)
    outp = Path(args.out).with_suffix(".json")
    outp.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"wrote {outp}")
    if args.print:
        print(render(res))


if __name__ == "__main__":
    main()
