"""X47's (and N4's) REPORTED quantities (registered 2026-10-20 by the AM-X47 amendment block, BEFORE any X47 read). Nothing here
decides a verdict; ``main.x35.x47.read`` carries each beside the verdict.

* :func:`per_step_gap` (AM-X47-5): D = mean over seeds of (S_final − S_s_eq), its two-sided 95 % t interval on n − 1 df;
  labelled "throughput-dependent" only when the lower bound is > 0 (a bound within 1e-9 of 0 is not, rule 8; zero
  variance has no interval), otherwise "BOUNDARY".
* :func:`per_member` / :func:`lineage_label` (AM-X47-7): each R member's Δ (X47 at s_eq − C0 finals) with Welch's
  interval at the look's 97.5 % (``readout.R1_CONFIDENCE``), the out-of-lineage member :data:`OUT_OF_LINEAGE` on its
  own, and the "in-lineage" label on a primary GAIN whose out-of-lineage Δ̂ <= 0.
* THE GROWTH DESCRIPTORS read a run's DECLARED growth (``model_config.json`` ``growth.fields``, :func:`declared_growth`)
  and dispatch on its field; a run that declares no growth, or a field no reader handles, is REFUSED (`ValueError`; the
  CLI exits 2), never an empty reading. The norm reader per field is :data:`NORM_READERS`; the zeroed twin is the field's
  ``zero_new`` hook on its `agents.model.growth.GROWABLE` row (:func:`growth_zeroer`).
* :func:`round_norms` / :func:`run_norms` on ``trunk_layers`` (AM-X47-10, the bot screen §2.1 DESCRIPTOR): the
  Frobenius norm of each NEW round's ``out_proj`` and ``linear2`` (``team_transformer.extra_rounds.<i>``; on a widened
  trunk the width branch's ``ffn_extra.linear2`` is part of the same matrix, their norms taken together), and its ratio
  to the mean norm of the same matrix over the base rounds (``team_transformer.transformer_layers.<j>``) at the same
  checkpoint. The reading line at the last node: a round whose two ratios are both >= :data:`ENGAGED_RATIO` "engaged",
  else "stayed near identity".
* :func:`width_norms` on ``trunk_ffn_dim`` (N4's E1, ``design_n4_capacity_screen.md`` §5): per round,
  ‖``ffn_extra.linear2``‖_F / the same round's ‖``linear2``‖_F, "engaged" iff >= :data:`ENGAGED_RATIO` at the last node.
* :func:`policy_diff` / :func:`run_policy_diff` (AM-X47b-3, registered 2026-10-20 before any read; REPORTED ONLY, in no
  decision; N4's E2): per node, KL(π_node ‖ π_zeroed) on a fixed bank (the registered one is the collapse pilot's
  ``bank_x26``), π_zeroed = the same weights with the growth's zero-init output maps set to zero again (depth: every new
  round's ``out_proj`` and ``linear2``, weight AND bias, so each ``trunk_depth.IdentityInitRound`` is the identity;
  width: every ``ffn_extra.linear2``), so the network is its parent's architecture's function. The mean over every
  decision and over the free ones, the max, and the share of decisions whose argmax moves. Unlike the norm, it
  separates "the new units changed the policy" from "their weights drifted".
* :func:`run_width_null` (N4 REVIEW M1, because E1 cannot fail on its own): on a NARROW control's (C0′) final against
  its parent T*, E1's null = the base ``linear2`` drift ‖W_final − W_T*‖ / ‖W_T*‖ × √((N − 256) / 256)
  (:func:`width_drift_null`), and E2's null = the KL between the final and the same final with every base ``linear2``
  (weight and bias) reverted to T*'s (:func:`revert_base_linear2`).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import zipfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

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
#: N4's E1 (a ``trunk_ffn_dim`` growth): a round's own ``linear2`` and its width branch's, read apart.
_KINDS = ("transformer_layers", "extra_rounds")
_WKEY = re.compile(r"^(?P<prefix>.*team_transformer\.)(?P<kind>extra_rounds|transformer_layers)\.(?P<i>\d+)\."
                   r"(?P<branch>ffn_extra\.)?linear2\.weight$")
#: a trunk round's own FFN output map, as a module name (REVIEW M1's E2 null reverts it)
_BASE_L2 = re.compile(r"^.*team_transformer\.(?:extra_rounds|transformer_layers)\.\d+\.linear2$")


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


def width_norms(state: Mapping[str, Any]) -> Dict[str, Any]:
    """N4's E1 on one state dict (a ``trunk_ffn_dim`` growth): per ``team_transformer`` prefix and per round (named
    ``transformer_layers.<j>`` / ``extra_rounds.<i>``), ‖``ffn_extra.linear2``‖_F, the round's own ‖``linear2``‖_F and
    their ratio. `ValueError` when no round carries a branch, or when one round lacks it (a width is one value per
    network) or lacks its base map: never an empty reading."""
    pieces: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = {}
    for k, t in state.items():
        m = _WKEY.match(k)
        if m:
            key = (_KINDS.index(m["kind"]), int(m["i"]))
            pieces.setdefault(m["prefix"], {}).setdefault(key, {})["branch" if m["branch"] else "base"] = \
                float(t.detach().float().norm())
    if not any("branch" in r for rounds in pieces.values() for r in rounds.values()):
        raise ValueError("no width branch (team_transformer.*.ffn_extra.linear2) in the state dict: not a width-grown "
                         "network")
    out: Dict[str, Any] = {}
    for prefix, rounds in sorted(pieces.items()):
        rows: Dict[str, Any] = {}
        for (kind, i), r in sorted(rounds.items()):
            name = f"{_KINDS[kind]}.{i}"
            if "branch" not in r or "base" not in r:
                raise ValueError(f"{prefix}{name}: has {sorted(r)} of the base linear2 and its width branch, not both")
            rows[name] = {"ffn_extra_linear2": r["branch"], "linear2": r["base"],
                          "ratio": (r["branch"] / r["base"]) if r["base"] > 0.0 else None}
        out[prefix] = {"rounds": rows}
    return out


def width_reading(norms: Mapping[str, Any]) -> Dict[str, str]:
    """E1's line per round: "engaged" iff the ratio is >= :data:`ENGAGED_RATIO` at every prefix (AM-X47-10's
    convention), else "stayed near identity"."""
    verdict: Dict[str, str] = {}
    for p in norms.values():
        for name, row in p["rounds"].items():
            ok = (row.get("ratio") or 0.0) >= ENGAGED_RATIO
            verdict[name] = ENGAGED if ok and verdict.get(name, ENGAGED) == ENGAGED else NEAR_IDENTITY
    return verdict


#: The norm descriptor each declared growth field reads (``state dict -> norms``, ``norms -> reading``). A declared
#: field that is not here is REFUSED (B-KFFN's experts are copies, not a zero-init block: no norm from zero).
NORM_READERS: Dict[str, Tuple[Callable[[Mapping[str, Any]], Dict[str, Any]], Callable[[Mapping[str, Any]],
                                                                                        Dict[str, str]]]] = {
    "trunk_layers": (round_norms, reading), "trunk_ffn_dim": (width_norms, width_reading)}


def declared_growth(run_dir: str) -> Dict[str, Any]:
    """The run's declared growth (``model_config.json``'s ``growth.fields``: ``{field: {parent, child}}``), or
    `ValueError`: a run that records no growth has no new parameters for a growth descriptor to read."""
    path = os.path.join(run_dir, "model_config.json")
    if not os.path.isfile(path):
        raise ValueError(f"{run_dir}: no model_config.json, so its declared growth is unknown")
    with open(path) as f:
        cfg = json.load(f)
    fields = (cfg.get("growth") or {}).get("fields") or {}
    if not fields:
        raise ValueError(f"{run_dir}: declares no growth (model_config.json growth.fields is empty): there are no new "
                         "parameters to read")
    return dict(fields)


def _run_files(run_dir: str) -> List[Tuple[int, str]]:
    files: List[Tuple[int, str]] = []
    for f in glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint_*_steps.zip")):
        m = _CKPT.search(f)
        if m:
            files.append((int(m.group(1)), f))
    return sorted(files)


def run_norms(run_dir: str, *, loader: Any = None, growth: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Every checkpoint of a run (and its final, when present), read-only: the norm series and the last node's line,
    by the reader of the run's DECLARED growth (:func:`declared_growth`, or ``growth``; :data:`NORM_READERS`).
    `ValueError` (the CLI's refusal, exit 2) on a growth no reader handles, on more than one declared field, on no node
    read and on an empty reading. A file that vanishes mid-read (a live run's retention) is listed under ``skipped``."""
    fields = dict(growth) if growth is not None else declared_growth(run_dir)
    if len(fields) != 1 or next(iter(fields)) not in NORM_READERS:
        raise ValueError(f"{run_dir}: declared growth {sorted(fields)}: the norm descriptor reads exactly one declared "
                         f"field among {sorted(NORM_READERS)}")
    field = next(iter(fields))
    norms_of, reading_of = NORM_READERS[field]
    load = loader or _policy_state
    files = _run_files(run_dir)
    final = os.path.join(run_dir, "final_model.zip")
    if os.path.isfile(final):
        files.append((-1, final))
    series: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for step, f in files:
        try:
            series.append({"step": step if step >= 0 else "final", "file": os.path.relpath(f, run_dir),
                           "norms": norms_of(load(f))})
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            skipped.append(f"{f}: {exc}")
    if not series:
        raise ValueError(f"{run_dir}: no checkpoint read ({len(skipped)} skipped)")
    last = series[-1]
    line = reading_of(last["norms"])
    if not line:
        raise ValueError(f"{run_dir}: the {field} reader found no new block at {last['file']}: an empty reading is "
                         "refused")
    return {"run": os.path.basename(os.path.normpath(run_dir)), "growth": fields, "reader": field,
            "engaged_ratio": ENGAGED_RATIO, "series": series, "skipped": skipped, "last_node": last["step"],
            "reading": line}


# ------------------------------------------------------------------------------------------ REVIEW M1's nulls
def width_drift_null(final: Mapping[str, Any], parent: Mapping[str, Any], *, ffn_dim: int) -> Dict[str, Any]:
    """REVIEW M1's E1 null on a NARROW control (C0′): per prefix and round, the base drift ‖linear2(final) −
    linear2(parent)‖_F / ‖linear2(parent)‖_F and that drift × √((``ffn_dim`` − 256) / 256), the ratio a width branch
    of ``ffn_dim − 256`` hidden units reaches under the same per-element drift with no signal. `ValueError` on a state
    carrying a width branch (the null reads the control), a missing or reshaped round, or no round."""
    from agents.model.trunk_ffn_width import FFN_DIM_BASE, check_ffn_dim

    extra = check_ffn_dim(ffn_dim) - FFN_DIM_BASE
    if extra <= 0:
        raise ValueError(f"--ffn-dim {ffn_dim}: the null scales to a branch, so it must exceed {FFN_DIM_BASE}")
    scale = math.sqrt(extra / FFN_DIM_BASE)
    for side, sd in (("final", final), ("parent", parent)):
        if any(".ffn_extra." in k for k in sd):
            raise ValueError(f"the {side} carries a width branch: the drift null reads the NARROW control (C0') and "
                             "its parent")
    rounds: Dict[str, Dict[str, Any]] = {}
    for k, t in final.items():
        m = _WKEY.match(k)
        if not m:
            continue
        if k not in parent or tuple(parent[k].shape) != tuple(t.shape):
            raise ValueError(f"{k}: absent from the parent or a different shape")
        w0 = parent[k].detach().double()
        den = float(w0.norm())
        drift = float((t.detach().double() - w0).norm()) / den if den > 0.0 else None
        rounds.setdefault(m["prefix"], {})[f"{m['kind']}.{m['i']}"] = {
            "drift": drift, "e1_null": drift * scale if drift is not None else None}
    if not rounds:
        raise ValueError("no team_transformer linear2 in the state dict")
    return {"ffn_dim": int(ffn_dim), "scale": scale,
            "rounds": {p: dict(sorted(r.items())) for p, r in sorted(rounds.items())}}


def revert_base_linear2(parent: Mapping[str, Any]) -> Callable[[Any], int]:
    """REVIEW M1's E2 null twin: a ``zero``-hook for :func:`run_policy_diff` that copies the PARENT's weight and bias
    into every trunk round's own ``linear2`` (the base FFN output map), IN PLACE. Refuses a module carrying a width
    branch (the null reads the narrow control) and a round the parent lacks."""
    import torch

    def revert(module: Any) -> int:
        n = 0
        with torch.no_grad():
            for name, mod in module.named_modules():
                if name.endswith(".ffn_extra"):
                    raise ValueError(f"{name}: a width branch; the E2 null reads the NARROW control (C0')")
                if not _BASE_L2.match(name):
                    continue
                for attr in ("weight", "bias"):
                    t = getattr(mod, attr, None)
                    if t is None:
                        continue
                    src = parent.get(f"{name}.{attr}")
                    if src is None or tuple(src.shape) != tuple(t.shape):
                        raise ValueError(f"{name}.{attr}: absent from the parent or a different shape")
                    t.copy_(src.to(device=t.device, dtype=t.dtype))
                n += 1
        return n
    return revert


def run_width_null(run_dir: str, parent_zip: str, bank_dir: str, *, ffn_dim: int, threads: int = 1,
                   loader: Any = None) -> Dict[str, Any]:
    """REVIEW M1's two nulls on ONE narrow-control run's final (``<run>/final_model.zip``) against its parent
    checkpoint (T*): E1's drift null (:func:`width_drift_null`) and E2's KL (:func:`run_policy_diff` with every base
    ``linear2`` reverted to the parent's, :func:`revert_base_linear2`). Read-only, CPU."""
    final = os.path.join(run_dir, "final_model.zip")
    if not os.path.isfile(final):
        raise ValueError(f"{run_dir}: no final_model.zip (the null reads a control's FINAL)")
    if os.path.isfile(os.path.join(run_dir, "model_config.json")):
        try:
            fields = declared_growth(run_dir)
        except ValueError:
            fields = {}
        if fields:
            raise ValueError(f"{run_dir}: declares growth {sorted(fields)}; the null reads the NARROW control (C0')")
    load = loader or _policy_state
    parent = load(parent_zip)
    e1 = width_drift_null(load(final), parent, ffn_dim=ffn_dim)
    e2 = run_policy_diff(run_dir, bank_dir, threads=threads, zero=revert_base_linear2(parent), final_only=True)
    return {"run": os.path.basename(os.path.normpath(run_dir)), "parent": parent_zip, "reported_only": True,
            "e1_null": e1, "e2_null": dict(e2["series"][0], bank=e2["bank"])}


# ------------------------------------------------------------------------------------------ AM-X47b-3
def zero_extra_rounds(module: Any) -> int:
    """The ``trunk_layers`` growth's ``zero_new`` hook (`agents.model.trunk_depth.zero_extra_rounds`): kept here by
    name for X47's readers."""
    from agents.model.trunk_depth import zero_extra_rounds as zero

    return zero(module)


def growth_zeroer(fields: Mapping[str, Any]) -> Callable[[Any], int]:
    """The π_zeroed twin of a declared growth: every declared field's ``zero_new`` hook (`agents.model.growth.GROWABLE`,
    in its declaration order), each of which must find a block. `ValueError` on a field with no hook (its new
    parameters are not a zero-init block: the zeroed twin is undefined)."""
    from agents.model.growth import GROWABLE

    bad = [f for f in fields if GROWABLE.get(f) is None or GROWABLE[f].zero_new is None]
    if bad or not fields:
        raise ValueError(f"declared growth {sorted(fields)}: no zero_new hook for {bad or 'an empty growth'} (zeroable: "
                         f"{sorted(f for f, r in GROWABLE.items() if r.zero_new is not None)})")
    order = [f for f in GROWABLE if f in fields]

    def zero(module: Any) -> int:
        total = 0
        for f in order:
            hook = GROWABLE[f].zero_new
            assert hook is not None
            n = hook(module)
            if n == 0:
                raise ValueError(f"declared growth {f}: its zero_new hook found no new block (not a grown network)")
            total += n
        return total
    return zero


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
                    threads: int = 1, zero: Optional[Callable[[Any], int]] = None,
                    final_only: bool = False) -> Dict[str, Any]:
    """The descriptor on a run's checkpoints (every ``stride``-th after sorting, or exactly ``nodes``, or with
    ``final_only`` the final alone) and its final, read-only, on CPU. Each node is loaded twice: as saved, then with
    ``zero`` applied (default: the run's DECLARED growth's ``zero_new`` hooks, :func:`growth_zeroer`; REVIEW M1's E2
    null passes :func:`revert_base_linear2`). ``rounds_zeroed`` is the number of blocks the hook changed."""
    from pathlib import Path

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals, load_checkpoint, policy_logits, reencode
    from main.policy_spectrum.spectrum import masked_probs

    if stride < 1:
        raise ValueError("--stride must be >= 1")
    growth: Optional[Dict[str, Any]] = None
    if zero is None:                                   # refuse an undeclared / unhandled growth before any heavy load
        growth = declared_growth(run_dir)
        zero = growth_zeroer(growth)
    files = _run_files(run_dir)
    final = os.path.join(run_dir, "final_model.zip")
    if final_only:
        if nodes:
            raise ValueError("final_only reads the final alone; drop the nodes")
        files = [(-1, final)] if os.path.isfile(final) else []
    elif nodes:
        want = set(int(n) for n in nodes)
        files = [x for x in files if x[0] in want]
        missing = sorted(want - {x[0] for x in files})
        if missing:
            raise ValueError(f"no checkpoint at {missing}")
    else:
        files = files[stride - 1::stride]
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
            n = zero(model.policy)
            if n == 0:
                raise ValueError(f"{f}: the zero hook changed no block (not a grown network)")
            p_zero = masked_probs(policy_logits(model, rows, masks), masks)
            series.append({"step": step if step >= 0 else "final", "file": os.path.relpath(f, run_dir),
                           "rounds_zeroed": n, **policy_diff(p_node, p_zero, free)})
            del model
    return {"run": os.path.basename(os.path.normpath(run_dir)), "growth": growth,
            "bank": bank.manifest.get("content_sha256"), "obs_as_recorded": gate.get("obs_as_recorded"),
            "reported_only": True, "series": series}
