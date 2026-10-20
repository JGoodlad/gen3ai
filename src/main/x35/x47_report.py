"""X47's REPORTED quantities (registered 2026-10-20 by the AM-X47 amendment block, BEFORE any X47 read). Nothing here
decides a verdict; ``main.x35.x47.read`` carries each beside the verdict.

* :func:`per_step_gap` (AM-X47-5): D = mean over seeds of (S_final − S_s_eq), its two-sided 95 % t interval on n − 1 df;
  labelled "throughput-dependent" only when the lower bound is > 0 (a bound within 1e-9 of 0 is not, rule 8; zero
  variance has no interval), otherwise "BOUNDARY".
* :func:`per_member` / :func:`lineage_label` (AM-X47-7): each R member's Δ (X47 at s_eq − C0 finals) with Welch's
  interval at the look's 97.5 % (``readout.R1_CONFIDENCE``), the out-of-lineage member :data:`OUT_OF_LINEAGE` on its
  own, and the "in-lineage" label on a primary GAIN whose out-of-lineage Δ̂ <= 0.
* :func:`round_norms` / :func:`run_norms` (AM-X47-10, the bot screen §2.1 DESCRIPTOR): the Frobenius norm of each NEW
  round's ``out_proj`` and ``linear2`` (``team_transformer.extra_rounds.<i>``), and its ratio to the mean norm of the
  same matrix over the base rounds (``team_transformer.transformer_layers.<j>``) at the same checkpoint. The reading line
  at the last node: a round whose two ratios are both >= :data:`ENGAGED_RATIO` "engaged", else "stayed near identity".
"""
from __future__ import annotations

import glob
import math
import os
import re
import zipfile
from typing import Any, Dict, List, Mapping, Optional, Tuple

from main.x35 import readout as R

#: the toy R's one member outside T*'s lineage (production P*), ``reference_population_toy.json``.
OUT_OF_LINEAGE = "pstar_x26"
IN_LINEAGE = "in-lineage"
#: AM-X47-5's interval.
D_CONFIDENCE = 0.95
THROUGHPUT = "throughput-dependent"
BOUNDARY = "BOUNDARY: the secondary crossed the bar within measurement noise"
#: AM-X47-10: the reading line's threshold (a ratio to the base rounds' norm of the same matrix). DESCRIPTIVE.
ENGAGED_RATIO = 0.01
ENGAGED, NEAR_IDENTITY = "engaged", "stayed near identity"
_MATS = ("out_proj", "linear2")
_KEY = re.compile(r"^(?P<prefix>.*team_transformer\.)(?P<kind>extra_rounds|transformer_layers)\.(?P<i>\d+)\."
                  r"(?P<mat>out_proj|linear2)\.weight$")
_CKPT = re.compile(r"checkpoint_(\d+)_steps\.zip$")


# ------------------------------------------------------------------------------------------ AM-X47-5
def per_step_gap(final: Mapping[str, Optional[float]], s_eq: Mapping[str, Optional[float]]) -> Dict[str, Any]:
    """D per seed = S(final) − S(s_eq) (pp), its mean and the two-sided 95 % t interval on n − 1 df, and the label."""
    out: Dict[str, Any] = {"confidence": D_CONFIDENCE}
    if set(final) != set(s_eq):
        return dict(out, label=None, reasons=[f"final seeds {sorted(final)} != s_eq seeds {sorted(s_eq)}"])
    seeds = sorted(final)
    if any(final[s] is None or s_eq[s] is None for s in seeds):
        return dict(out, label=None, reasons=["a seed's S is missing"])
    d = [float(final[s]) - float(s_eq[s]) for s in seeds]  # type: ignore[arg-type]
    out["per_seed_pp"] = dict(zip(seeds, d))
    if len(d) < 2 or not all(math.isfinite(x) for x in d):
        return dict(out, label=None, reasons=["the interval needs >= 2 finite seed gaps"])
    m = math.fsum(d) / len(d)
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in d) / (len(d) - 1))
    out.update(n=len(d), d_pp=m, sd_pp=sd, df=len(d) - 1)
    if not sd > 0.0:
        return dict(out, ci95_pp=None, label=BOUNDARY, reasons=["zero variance: the interval is undefined"])
    q = R.t_quantile(1.0 - (1.0 - D_CONFIDENCE) / 2.0, len(d) - 1)
    lo, hi = m - q * sd / math.sqrt(len(d)), m + q * sd / math.sqrt(len(d))
    above, near = R._clears(lo, 0.0, above=True)
    return dict(out, q=q, ci95_pp=[lo, hi], label=THROUGHPUT if above else BOUNDARY,
                near_boundary=[f"lower {lo!r} within {R.RULE8_EPS} of 0"] if near else [], reasons=[])


# ------------------------------------------------------------------------------------------ AM-X47-7
def per_member(on: Mapping[str, Mapping[str, float]], off: Mapping[str, Mapping[str, float]], *,
               s_on: Optional[Mapping[str, Any]] = None, s_off: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Each member's Δ (pp) with Welch's interval at ``readout.R1_CONFIDENCE`` (``on`` / ``off`` = {seed: {member:
    pp}}). ``s_on`` / ``s_off`` (each seed's S) are cross-checked: a seed's member mean that differs from its S by
    more than 1e-6 pp is listed under ``mismatch`` (reported; the S the verdict read is unchanged)."""
    members = sorted({m for d in (*on.values(), *off.values()) for m in d})
    out: Dict[str, Any] = {"confidence": R.R1_CONFIDENCE, "members": {}, "mismatch": [], "reasons": []}
    for side, rows in (("X47", on), ("C0", off)):
        for seed, d in sorted(rows.items()):
            if sorted(d) != members:
                out["reasons"].append(f"{side} seed {seed}: members {sorted(d)} != {members}")
    for side, rows, s in (("X47", on, s_on), ("C0", off, s_off)):
        for seed, d in sorted(rows.items()):
            want = (s or {}).get(seed)
            if want is not None and d and abs(math.fsum(d.values()) / len(d) - float(want)) > 1e-6:
                out["mismatch"].append(f"{side} seed {seed}: member mean {math.fsum(d.values()) / len(d)!r} != S {want!r}")
    for m in members:
        xs = [float(d[m]) for d in on.values() if m in d]
        ys = [float(d[m]) for d in off.values() if m in d]
        try:
            w = R.welch_interval(xs, ys)
        except R.ReadoutInputError as exc:
            out["members"][m] = {"delta_pp": None, "reason": str(exc)}
            continue
        out["members"][m] = {"delta_pp": w.delta, "ci_pp": list(w.ci), "n_on": w.n_on, "n_off": w.n_off}
    out["out_of_lineage"] = {OUT_OF_LINEAGE: out["members"].get(OUT_OF_LINEAGE)}
    return out


def lineage_label(pm: Optional[Mapping[str, Any]]) -> Optional[str]:
    """AM-X47-7 on a primary GAIN: "in-lineage" iff the out-of-lineage member's Δ̂ <= 0; ``None`` when it is not read
    (the report then says so)."""
    row = ((pm or {}).get("members") or {}).get(OUT_OF_LINEAGE) or {}
    d = row.get("delta_pp")
    if d is None:
        return None
    return IN_LINEAGE if d <= 0.0 else None


# ------------------------------------------------------------------------------------------ AM-X47-10
def round_norms(state: Mapping[str, Any]) -> Dict[str, Any]:
    """The descriptor on one state dict: per ``team_transformer`` prefix, each new round's two norms and ratios."""
    by: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    for k, t in state.items():
        m = _KEY.match(k)
        if m:
            by.setdefault(m["prefix"], {}).setdefault(m["kind"], {}).setdefault(int(m["i"]), {})[m["mat"]] = \
                float(t.detach().float().norm())
    if not by:
        raise ValueError("no team_transformer rounds in the state dict")
    out: Dict[str, Any] = {}
    for prefix, kinds in sorted(by.items()):
        base = kinds.get("transformer_layers") or {}
        ref = {mat: math.fsum(r[mat] for r in base.values()) / len(base) for mat in _MATS} if base else {}
        rounds = {}
        for i, r in sorted((kinds.get("extra_rounds") or {}).items()):
            row = {mat: r.get(mat) for mat in _MATS}
            row.update({f"{mat}_ratio": (r[mat] / ref[mat]) if ref.get(mat) else None for mat in _MATS if mat in r})
            rounds[str(i)] = row
        out[prefix] = {"base_mean": ref, "extra_rounds": rounds}
    return out


def reading(norms: Mapping[str, Any]) -> Dict[str, str]:
    """The one-line reading per new round (every prefix must agree for "engaged")."""
    verdict: Dict[str, str] = {}
    for p in norms.values():
        for i, row in p["extra_rounds"].items():
            ok = all((row.get(f"{mat}_ratio") or 0.0) >= ENGAGED_RATIO for mat in _MATS)
            verdict[i] = ENGAGED if ok and verdict.get(i, ENGAGED) == ENGAGED else NEAR_IDENTITY
    return verdict


def _policy_state(zip_path: str) -> Mapping[str, Any]:
    import torch

    with zipfile.ZipFile(zip_path) as z, z.open("policy.pth") as f:
        return torch.load(f, map_location="cpu", weights_only=True)


def run_norms(run_dir: str, *, loader: Any = None) -> Dict[str, Any]:
    """Every checkpoint of a run (and its final, when present), read-only: the norm series and the last node's line.
    A file that vanishes mid-read (a live run's retention) is listed under ``skipped``."""
    load = loader or _policy_state
    files: List[Tuple[int, str]] = []
    for f in glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint_*_steps.zip")):
        m = _CKPT.search(f)
        if m:
            files.append((int(m.group(1)), f))
    files.sort()
    final = os.path.join(run_dir, "final_model.zip")
    if os.path.isfile(final):
        files.append((-1, final))
    series: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for step, f in files:
        try:
            series.append({"step": step if step >= 0 else "final", "file": os.path.relpath(f, run_dir),
                           "norms": round_norms(load(f))})
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            skipped.append(f"{f}: {exc}")
    last = series[-1] if series else None
    return {"run": os.path.basename(os.path.normpath(run_dir)), "engaged_ratio": ENGAGED_RATIO, "series": series,
            "skipped": skipped, "last_node": last["step"] if last else None,
            "reading": reading(last["norms"]) if last else None}

