"""X47's REPORTED quantities (registered 2026-10-20 by the AM-X47 amendment block, BEFORE any X47 read). Nothing here
decides a verdict; ``main.x35.x47.read`` carries each beside the verdict.

* :func:`per_step_gap` (AM-X47-5): D = mean over seeds of (S_final − S_s_eq), its two-sided 95 % t interval on n − 1 df;
  labelled "throughput-dependent" only when the lower bound is > 0 (a bound within 1e-9 of 0 is not, rule 8; zero
  variance has no interval), otherwise "BOUNDARY".
* :func:`per_member` / :func:`lineage_label` (AM-X47-7): each R member's Δ (X47 at s_eq − C0 finals) with Welch's
  interval at the look's 97.5 % (``readout.R1_CONFIDENCE``), the out-of-lineage member :data:`OUT_OF_LINEAGE` on its
  own, and the "in-lineage" label on a primary GAIN whose out-of-lineage Δ̂ <= 0.
* :func:`round_norms` / :func:`run_norms` (AM-X47-10, the bot screen §2.1 DESCRIPTOR): the Frobenius norm of each NEW
  round's ``out_proj`` and ``linear2`` (``team_transformer.extra_rounds.<i>``; on a widened trunk the width branch's
  ``ffn_extra.linear2`` is part of the same matrix, their norms taken together), and its ratio to the mean norm of the
  same matrix over the base rounds (``team_transformer.transformer_layers.<j>``) at the same checkpoint. The reading line
  at the last node: a round whose two ratios are both >= :data:`ENGAGED_RATIO` "engaged", else "stayed near identity".
* :func:`policy_diff` / :func:`run_policy_diff` (AM-X47b-3, registered 2026-10-20 before any read; REPORTED ONLY, in no
  decision): per node, KL(π_node ‖ π_zeroed) on a fixed bank (the registered one is the collapse pilot's ``bank_x26``),
  π_zeroed = the same weights with every new round's ``out_proj`` and ``linear2`` (weight AND bias) set to zero, which
  makes each ``trunk_depth.IdentityInitRound`` exactly the identity (the network is then the base rounds' function).
  The mean over every decision and over the free ones, the max, and the share of decisions whose argmax moves. Unlike
  the norm, it separates "the rounds changed the policy" from "the rounds' weights drifted".
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
#: ``linear2`` is the round's WHOLE FFN output map: the round's own, plus a width branch's (``ffn_extra.linear2``,
#: `--trunk-ffn-dim` > 256) or a per-kind expert's (``kind_ffn.<kind>.linear2``), whichever the round carries. A key
#: this misses would read a wide trunk's round as the narrow one.
_KEY = re.compile(r"^(?P<prefix>.*team_transformer\.)(?P<kind>extra_rounds|transformer_layers)\.(?P<i>\d+)\."
                  r"(?:ffn_extra\.|kind_ffn\.\w+\.)?(?P<mat>out_proj|linear2)\.weight$")
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
    pieces: Dict[str, Dict[str, Dict[int, Dict[str, List[float]]]]] = {}
    for k, t in state.items():
        m = _KEY.match(k)
        if m:
            pieces.setdefault(m["prefix"], {}).setdefault(m["kind"], {}).setdefault(int(m["i"]), {}) \
                .setdefault(m["mat"], []).append(float(t.detach().float().norm()))
    if not pieces:
        raise ValueError("no team_transformer rounds in the state dict")
    # one matrix's norm, or the Frobenius norm of the round's linear2 pieces taken together (the base map + its branch)
    by = {pre: {kind: {i: {mat: (v[0] if len(v) == 1 else math.sqrt(math.fsum(x * x for x in v)))
                           for mat, v in mats.items()} for i, mats in rounds.items()}
                for kind, rounds in kinds.items()} for pre, kinds in pieces.items()}
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



# ------------------------------------------------------------------------------------------ AM-X47b-3
def zero_extra_rounds(module: Any) -> int:
    """Zero ``out_proj`` and EVERY ``linear2`` (weight and bias) of every round inside a module named ``extra_rounds``,
    IN PLACE: the round's own, a width branch's (``ffn_extra``) and each per-kind expert's (``kind_ffn.*``), because
    a branch left trained makes the round no longer the identity. Returns the number of rounds zeroed (0 = no new
    round: the descriptor is undefined)."""
    import torch

    n = 0
    with torch.no_grad():
        for name, mod in module.named_modules():
            if name.rsplit(".", 1)[-1] != "extra_rounds":
                continue
            for rnd in mod:
                lins = [rnd.out_proj] + [sub for sname, sub in rnd.named_modules()
                                         if sname.rsplit(".", 1)[-1] == "linear2"]
                for lin in lins:
                    lin.weight.zero_()
                    if lin.bias is not None:
                        lin.bias.zero_()
                n += 1
    return n


def policy_diff(p_node: Any, p_zeroed: Any, free: Optional[Any] = None) -> Dict[str, Any]:
    """KL(p_node ‖ p_zeroed) per row of masked probabilities (illegal = 0), summarised (module docs)."""
    import numpy as np

    p = np.asarray(p_node, dtype=np.float64)
    q = np.asarray(p_zeroed, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2 or len(p) == 0:
        raise ValueError(f"policy_diff: shapes {p.shape} / {q.shape}")
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - np.log(np.where(q > 0, q, 1.0))), 0.0)
    if np.any((p > 0) & (q <= 0)):
        raise ValueError("policy_diff: the zeroed policy puts no mass where the node does (KL infinite)")
    kl = np.maximum(terms.sum(axis=1), 0.0)
    moved = np.argmax(p, axis=1) != np.argmax(q, axis=1)
    out: Dict[str, Any] = {"n": int(len(kl)), "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                           "argmax_moved_share": float(moved.mean())}
    if free is not None:
        f = np.asarray(free, dtype=bool)
        out.update({"n_free": int(f.sum()), "kl_mean_free": float(kl[f].mean()) if f.any() else None})
    return out


def run_policy_diff(run_dir: str, bank_dir: str, *, stride: int = 10, nodes: Optional[List[int]] = None,
                    threads: int = 1) -> Dict[str, Any]:
    """The descriptor on a run's checkpoints (every ``stride``-th after sorting, or exactly ``nodes``) and its final,
    read-only, on CPU. Each node is loaded twice (as saved, then with the new rounds zeroed)."""
    from pathlib import Path

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals, load_checkpoint, policy_logits, reencode
    from main.policy_spectrum.spectrum import masked_probs

    if stride < 1:
        raise ValueError("--stride must be >= 1")
    files: List[Tuple[int, str]] = []
    for f in glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint_*_steps.zip")):
        m = _CKPT.search(f)
        if m:
            files.append((int(m.group(1)), f))
    files.sort()
    if nodes:
        want = set(int(n) for n in nodes)
        files = [x for x in files if x[0] in want]
        missing = sorted(want - {x[0] for x in files})
        if missing:
            raise ValueError(f"no checkpoint at {missing}")
    else:
        files = files[stride - 1::stride]
        final = os.path.join(run_dir, "final_model.zip")
        if os.path.isfile(final):
            files.append((-1, final))
    if not files:
        raise ValueError(f"{run_dir}: no checkpoint to read")
    bank = load_bank(Path(bank_dir))
    rows, masks, gate = reencode(bank)
    free = [d.get("kind") == "free" for d in bank.decisions]
    series: List[Dict[str, Any]] = []
    with inference_globals(threads):
        for step, f in files:
            model = load_checkpoint(Path(f))
            p_node = masked_probs(policy_logits(model, rows, masks), masks)
            n = zero_extra_rounds(model.policy)
            if n == 0:
                raise ValueError(f"{f}: no extra_rounds (not a grown network)")
            p_zero = masked_probs(policy_logits(model, rows, masks), masks)
            series.append({"step": step if step >= 0 else "final", "file": os.path.relpath(f, run_dir),
                           "rounds_zeroed": n, **policy_diff(p_node, p_zero, free)})
            del model
    return {"run": os.path.basename(os.path.normpath(run_dir)), "bank": bank.manifest.get("content_sha256"),
            "obs_as_recorded": gate.get("obs_as_recorded"), "reported_only": True, "series": series}
