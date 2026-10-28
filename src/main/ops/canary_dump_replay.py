"""REPLAY AN ARCHIVED CANARY WEIGHT DUMP — compiled vs eager bf16 backward, on the canary's own rows, N times.

THE EVENT IT WAS WRITTEN FOR. `rb_x35_x48kffn_s3502` (X48: production + `--grow trunk_ffn_by_kind=on`, a fork of P* at
pin f4b25238, `--compile-trainer`) logged at update 200 (299,668,245 steps) an UNCONFIRMED compile canary: region R1
under the bf16 rule refused on `features_extractor.edge_bias.s1_map.weight` (compiled 1.29 vs eager 0.0587) and
`edge_bias.c2_map.weight` (1.27 vs 0.0547), worst ratio 23.2 > K = 12; "reproduced on slice(s) [2] of [2, 1]". The
canary dumped the policy `state_dict` (`compile_canary._dump_weights`); the dump, the run's verdict records and a
SHA256SUMS are archived together. This module replays that dump; it works for any canary dump with that layout.

WHAT THE DUMP HOLDS, AND WHAT IT DOES NOT. Only `model.policy.state_dict()` (every key the run's checkpoints carry).
NOT the rows: the canary judges the K9 learner golden's committed rows (`compile_regions.r1_batch`, 64 rows tiled to
the run's B; slice 0 = every row, the scheduled check; slice 2 = the first half, slice 1 = the second half, the two
confirmations, `compile_trainer.FIXTURE_HALVES`), which this checkout re-reads from
`learner_golden.BUFFER_PATH`. The replay is the canary's STATE only if those rows and the hparams that shape the
micro-step are the run's — so every repeat is checked against the dump's own records (VALIDITY below).

THE ARMS, per repeat and slice (the learner built at THIS checkout from the run's resolved config, the dump loaded
STRICT, the trunk at bf16, the run's init record attached so `unmoved_parameters` reads as in the run):

  * ``canary`` (CUDA): `CompileCanary._check` VERBATIM — the canary's own rungs, relocation, yardstick re-run and
    bf16 rule — with a `GradWatch` over EVERY parameter. Its refusal / pass is THE verdict this replay counts.
  * ``direct``: E32 (eager, trunk fp32: the reference), Ebf (eager bf16), and on CUDA Cbf (the production compile,
    `compile_regions.install`, the canary's own compiled route) on the rows relocated off the clip boundary exactly as
    the canary relocates them; per-parameter distances from E32 through the rule's own `per_param_grad_errors` (filled
    by `_r1_verdict_bf16` into a `GradWatch`), the sha256 of each arm's flat gradient (bitwise stability across
    repeats). On CPU only E32 / Ebf run; Cbf is recorded as SKIPPED (there is no CUDA kernel to judge).

THE SUMMARY RULE (PRE-DECLARED, 2026-10-28, before any GPU read; deterministic, standing rule 8). For each repeat and
slice the canary arm is classified: ``boundary`` if any ``edge_bias.*`` ratio lies within `BOUNDARY_REL` (0.1 %) of
K; else ``refused_edge_bias`` if the canary refused AND some ``edge_bias.*`` parameter's ratio is > K; else
``refused_other`` if it refused; else ``pass``. With N = the repeats run:

  * REPRODUCES         iff ``refused_edge_bias`` in >= ceil(N / 2) repeats on the slice that fired (`FIRED_SLICE` = 2);
  * DOES NOT REPRODUCE iff no repeat on any slice is ``refused_edge_bias``, ``refused_other`` or ``boundary``;
  * MIXED              otherwise;
  * INVALID-REPLAY     overrides all three when VALIDITY fails (the replay is not the canary's state);
  * EAGER-ONLY         on CPU (no compiled arm: nothing to count), unless INVALID-REPLAY.

VALIDITY (pre-declared): for every replayed slice the dump's record carries (`RECORD_KEY`), the replay's numbers are
compared with the record's (`canary_verdicts.jsonl`, the record whose ``policy_pt`` names the dump):

  * STATE (fp32 facts, both devices): ``clip_margin.relocated_rows`` and ``rows`` EXACT; ``nearest_unmoved`` within
    `NEAREST_ABS` absolute (the record keeps 4 significant digits);
  * EAGER bf16 (Ebf vs E32), CUDA only: the global ``drift_grad``, the record's worst parameter and its detail line's
    ``edge_bias.*`` eager distances, and every recorded watch-list (``pointer_head.*``) eager distance, each within
    `EAGER_REL_CUDA` relative (CUDA eager bf16 is run-to-run bit-identical, window D). On CPU they are REPORTED with
    their ratio to the record, never judged: CPU bf16 kernels are not CUDA's. (A CPU band [0.5, 2] was declared first
    and dropped after the first CPU read of the X48 dump: the STATE checks matched exactly, the eager c2_map distance
    read 0.170 vs CUDA's 0.0572 — CPU bf16 rounds this family ~3x noisier, so an eager band is no state test there.)

Also reported: Cbf's bitwise stability across repeats in one process (the number of distinct gradient sha256s per
slice) and the spread (max - min) of each ``edge_bias.*`` compiled distance across repeats.

    python -m main.ops.canary_dump_replay --dump <dir>/canary_u200_unconfirmed_policy.pt \\
        --run-dir models/rb_x35_x48kffn_s3502 --out /tmp/replay --device cuda --repeats 5

EXIT: 0 the replay ran (whatever it found), 2 a refusal (a sha mismatch, a missing file, an out dir inside models/ or
a checkout, CUDA asked but absent), 1 an error. No wall budget: a timeout is never a semantic outcome.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: the slice that FIRED in the event (reproduced on slice(s) [2] of [2, 1]) — the summary rule's slice
FIRED_SLICE = 2
#: the canary's slices: the scheduled check (all rows), then its two confirmations (`FIXTURE_HALVES`)
DEFAULT_SLICES: Tuple[int, ...] = (0, 2, 1)
#: the record key each slice's numbers are stored under in `canary_verdicts.jsonl`
RECORD_KEY = {0: "scheduled", 2: "confirm_slice2", 1: "confirm_slice1"}
EDGE_RE = re.compile(r"(^|\.)edge_bias\.")
BOUNDARY_REL = 1e-3
NEAREST_ABS = 2e-3
EAGER_REL_CUDA = 0.05
CLASSES = ("pass", "refused_edge_bias", "refused_other", "boundary")
RULE_TEXT = (f"per repeat x slice: boundary if any edge_bias.* ratio within {BOUNDARY_REL:g} relative of K; else "
             "refused_edge_bias if the canary refused and some edge_bias.* ratio > K; else refused_other if it refused; "
             f"else pass. REPRODUCES iff refused_edge_bias in >= ceil(N/2) repeats on slice {FIRED_SLICE}; DOES NOT "
             "REPRODUCE iff no repeat on any slice is refused_* or boundary; MIXED otherwise; INVALID-REPLAY overrides "
             "when validity fails; EAGER-ONLY on CPU (no compiled arm).")


class Refusal(RuntimeError):
    """A precondition does not hold: the replay would not mean what it says (exit 2)."""


# ------------------------------------------------------------------------------------------------ preconditions
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_dump_sha(dump: Path) -> str:
    """The dump's sha256, REFUSED unless it equals the entry for its name in the SHA256SUMS beside it."""
    if not dump.is_file():
        raise Refusal(f"no dump file {dump}")
    sums = dump.parent / "SHA256SUMS"
    if not sums.is_file():
        raise Refusal(f"no SHA256SUMS beside the dump ({sums})")
    want = None
    for line in sums.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == dump.name:
            want = parts[0].lower()
    if want is None:
        raise Refusal(f"{sums} has no entry for {dump.name}")
    got = sha256_file(dump)
    if got != want:
        raise Refusal(f"sha256 MISMATCH for {dump}: {got} != SHA256SUMS {want}")
    return got


def _inside(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def check_out_dir(out: Path, models_dir: Optional[Path] = None) -> Path:
    """The out dir, resolved: REFUSED inside the main checkout's `models/` (or ``models_dir``), inside any git
    checkout (a parent with `.git`), or when it already holds files (a replay never mixes with an earlier one)."""
    o = out.expanduser().resolve()
    if models_dir is None:
        from utils.paths import main_models_dir
        models_dir = main_models_dir()
    if models_dir is not None and _inside(o, Path(models_dir).resolve()):
        raise Refusal(f"--out {o} is inside models/ ({models_dir}): models/ is read-only")
    for parent in (o, *o.parents):
        if (parent / ".git").exists():
            raise Refusal(f"--out {o} is inside the git checkout {parent}: write outside any checkout")
    if o.exists() and any(o.iterdir()):
        raise Refusal(f"--out {o} is not empty: a replay writes into a fresh directory")
    return o


def recorded(verdicts: Path, dump_name: str) -> Dict[str, Any]:
    """The canary record whose ``policy_pt`` names the dump (the numbers the replay must reproduce)."""
    if not verdicts.is_file():
        raise Refusal(f"no canary verdict records at {verdicts}")
    hit = None
    for line in verdicts.read_text().splitlines():
        with contextlib.suppress(ValueError):
            r = json.loads(line)
            if r.get("policy_pt") == dump_name:
                hit = r
    if hit is None:
        raise Refusal(f"{verdicts} has no record naming {dump_name} (policy_pt)")
    return hit


def run_anchor(run_dir: Path) -> Path:
    """A zip of the run whose sidecars resolve the run's config (`--model <zip>`) and whose ``data`` carries the
    run's init record: final_model.zip, else the newest checkpoint."""
    for f in ("model_config.json", "metadata.json"):
        if not (run_dir / f).is_file():
            raise Refusal(f"the run dir {run_dir} has no {f}")
    if (run_dir / "final_model.zip").is_file():
        return run_dir / "final_model.zip"
    ck = sorted((run_dir / "checkpoints").glob("checkpoint_*_steps.zip"),
                key=lambda p: int(re.search(r"_(\d+)_steps", p.name).group(1)))  # type: ignore[union-attr]
    if not ck:
        raise Refusal(f"the run dir {run_dir} has neither final_model.zip nor a checkpoint")
    return ck[-1]


def init_record_of(zip_path: Path) -> Optional[Dict[str, str]]:
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read("data"))
    rec = data.get("param_init_sha256")
    return rec if isinstance(rec, dict) and rec else None


# ------------------------------------------------------------------------------------------------ the learner
def build_model(run_dir: Path, dump: Path, device: str, batch: int) -> Tuple[Any, Dict[str, Any]]:
    """The run's learner at THIS checkout (the trainer's resolution of ``--model <anchor>``: every flag from the run's
    config), the dump loaded STRICT, the trunk at bf16, the run's init record attached, on ``device``."""
    import torch

    from agents.model import compile_regions as cr
    from agents.model.trunk_precision import set_trunk_precision, trunk_precision_of
    from agents.training import learner_golden as LG
    from main.ops.cuda_build_validate import resolved_args
    anchor = run_anchor(run_dir)
    args = resolved_args(["--model", str(anchor)], "cpu")
    if getattr(args, "trunk_precision", None) != "bf16":
        raise Refusal(f"the run's resolved --trunk-precision is {getattr(args, 'trunk_precision', None)!r}, not bf16")
    model = LG.build_learner(args=args)
    set_trunk_precision(model.policy, "bf16")
    if trunk_precision_of(model.policy) != "bf16":
        raise Refusal(f"the learner's trunk is {trunk_precision_of(model.policy)}, wanted bf16")
    sd = torch.load(dump, map_location="cpu", weights_only=True)
    model.policy.load_state_dict(sd, strict=True)
    rec = init_record_of(anchor)
    if rec is not None:
        setattr(model, cr.INIT_RECORD_ATTR, rec)
    dev = torch.device(device)
    model.policy.to(dev)
    model.device = dev
    model.rollout_buffer.device = dev
    model.batch_size = int(batch)
    model.behaviour_dump_dir = None                 # nothing the canary code calls may write a run dir
    model._learner_cudagraphs = False
    model.policy.set_training_mode(True)
    return model, {"anchor": str(anchor), "init_record": rec is not None, "keys": len(sd),
                   "shift_max": cr.shift_max_of(model), "margin": cr.clip_margin_of(model)}


def install_compiled(model: Any, backend: Optional[str] = None) -> None:
    """The production compile of R1 (`compile_regions.install`, as the trainer's `arm_compile_sentinel` does it);
    ``backend`` is a TEST seam (dynamo's ``eager`` on CPU)."""
    from agents.model import compile_control as cc
    from agents.model import compile_regions as cr
    cc.control().install()
    cr.install(model, backend=backend)


# ------------------------------------------------------------------------------------------------ the reads
def _flat_sha(t: Any) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _edge(params: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The JUDGED edge_bias.* rows of a GradWatch (an unjudged parameter carries no numbers)."""
    return {n: d for n, d in params.items() if EDGE_RE.search(n) and "eager" in d}


def _watch_out(w: Any) -> Dict[str, Any]:
    """A GradWatch's UNROUNDED numbers: the edge_bias rows, the worst, the globals, the margin."""
    return {"rule": w.rule, "edge_bias": _edge(w.params), "worst": w.worst, "global": dict(w.globals),
            "clip_margin": w.margin, "yardstick_out_of_margin": list(w.yardstick)}


def canary_check(model: Any, slice_: int, batch: int) -> Dict[str, Any]:
    """`CompileCanary._check` VERBATIM on ``slice_`` with a GradWatch over every parameter; never raises a refusal."""
    from agents.model import compile_regions as cr
    from agents.model import compile_trainer as ct
    from agents.model.compile_canary import CompileCanary
    w = cr.GradWatch(("",))
    can = CompileCanary(model, batch_size=batch, row_null=False)
    t0 = time.perf_counter()
    out: Dict[str, Any] = {}
    try:
        rules = can._check(slice_, {}, w)
        out.update(refused=False, text=" | ".join(rules))
    except ct.CompileTrainerError as exc:
        out.update(refused=True, error=type(exc).__name__, text=str(exc)[:4000])
    out.update(seconds=time.perf_counter() - t0, memory=dict(can.memory), watch=_watch_out(w),
               all_params={n: d for n, d in w.params.items() if "eager" in d})
    return out


def direct_arms(model: Any, slice_: int, batch: int, compiled: bool) -> Dict[str, Any]:
    """E32 / Ebf (and Cbf when ``compiled``) on the rows relocated exactly as the canary relocates them; the rule's
    own per-parameter distances from E32 (`_r1_verdict_bf16` filling a GradWatch), each arm's gradient sha256."""
    from agents.model import compile_regions as cr
    from agents.model import compile_trainer as ct
    from agents.training.instrumented_ppo.micro_step import micro_step
    t0 = time.perf_counter()
    policy = model.policy
    policy.set_training_mode(True)
    b = cr.r1_batch(model, batch, slice_)
    args, fix = cr.relocate_clip_boundary(model, cr._r1_args(model, b))
    names = [n for n, _ in ct.grad_parameters(model, policy.features_extractor)]
    regime = cr.weights_regime(model)
    unmoved = cr.unmoved_parameters(model) if regime != "fresh" else []
    idx = {n: i for i, n in enumerate(names)}
    over = {idx[n]: cr.R1_PARAM_BAR["fresh"] for n in unmoved}
    ref = cr._r1_reference(model, args)
    eager = cr._r1_arm(model, micro_step, args)
    arms = {"fp32 reference": ref, "eager": eager}
    comp = None
    if compiled:
        comp = cr._r1_arm(model, model._compiled_micro_step, args)
        arms["compiled"] = comp
    w = cr.GradWatch(("",))
    w.fixture_rows = [int(x) for x in b.fixture_rows] if b.fixture_rows is not None else None
    w.fill_margin(fix, cr._margin_reads(fix, arms), cr._same_precision(arms))
    gl = cr.golden_row_losses({"ref": ref, "eager": eager, "comp": comp if comp is not None else eager}, fix, [args])
    rule: Dict[str, Any] = {}
    try:
        rule["text"] = cr._r1_verdict_bf16(eager, comp if comp is not None else eager, ref, names, regime,
                                           bar_by_index=over, watch=w, losses=(gl["ref"], gl["eager"], gl["comp"]))
        rule["refused"] = False
    except ct.CompileTrainerError as exc:
        rule.update(refused=True, error=type(exc).__name__, text=str(exc)[:4000])
    if comp is None:                                # compiled := eager above: only the EAGER numbers mean anything
        rule["note"] = "no compiled arm: the rule ran with compiled := eager, so only its EAGER numbers mean anything"
        for d in w.params.values():
            d.pop("compiled", None)
            d.pop("ratio", None)
        w.worst = None
        w.globals = {q: v for q, v in w.globals.items() if not q.startswith("envelope")}
    eager_err = {n: d["eager"] for n, d in w.params.items() if "eager" in d}
    worst_e = max(eager_err.items(), key=lambda kv: kv[1]) if eager_err else None
    out = {"regime": regime, "unmoved": unmoved, "relocated_rows": int(fix.moved.sum()),
           "rows": int(fix.moved.numel()), "nearest_unmoved": float(fix.nearest_unmoved),
           "losses": {"E32": float(ref["loss"].sum()), "Ebf": float(eager["loss"].sum()),
                      "Cbf": float(comp["loss"].sum()) if comp is not None else None},
           "grad_sha256": {"E32": _flat_sha(ref["grad"]), "Ebf": _flat_sha(eager["grad"]),
                           "Cbf": _flat_sha(comp["grad"]) if comp is not None else "SKIPPED (cpu)"},
           "rule": rule, "watch": _watch_out(w), "eager_worst_param": worst_e,
           "per_param": {n: {k: d[k] for k in ("eager", "compiled") if k in d} for n, d in w.params.items()
                         if "eager" in d},
           "seconds": time.perf_counter() - t0}
    del ref, eager, comp, arms
    cr.release_gate_memory(model)
    for p in policy.parameters():
        p.grad = None
    return out


# ------------------------------------------------------------------------------------------------ the rule
def classify(refused: bool, edge: Dict[str, Dict[str, Any]], k: float) -> str:
    """One repeat x slice of the CANARY arm (module docstring)."""
    ratios = [float(d["ratio"]) for d in edge.values() if d.get("ratio") is not None]
    if any(abs(r / k - 1.0) <= BOUNDARY_REL for r in ratios if math.isfinite(r)):
        return "boundary"
    if refused and any(not r <= k for r in ratios):
        return "refused_edge_bias"
    return "refused_other" if refused else "pass"


def summarize(classes: List[Dict[int, str]], valid: bool, compiled: bool) -> Dict[str, Any]:
    """The pre-declared rule over per-repeat ``{slice: class}`` (module docstring)."""
    n = len(classes)
    slices = sorted({s for c in classes for s in c})
    counts = {s: {c: sum(1 for r in classes if r.get(s) == c) for c in CLASSES} for s in slices}
    need = math.ceil(n / 2) if n else 1
    if not valid:
        verdict = "INVALID-REPLAY"
    elif not compiled:
        verdict = "EAGER-ONLY"
    elif n and counts.get(FIRED_SLICE, {}).get("refused_edge_bias", 0) >= need:
        verdict = "REPRODUCES"
    elif n and all(counts[s]["refused_edge_bias"] == counts[s]["refused_other"] == counts[s]["boundary"] == 0
                   for s in slices):
        verdict = "DOES NOT REPRODUCE"
    else:
        verdict = "MIXED"
    return {"verdict": verdict, "rule": RULE_TEXT, "repeats": n, "need": need, "fired_slice": FIRED_SLICE,
            "counts": {str(s): c for s, c in counts.items()}}


_DETAIL = re.compile(r"(\S+) compiled ([0-9.eE+-]+) vs eager ([0-9.eE+-]+)")


def _cmp(name: str, replay: Optional[float], rec: float, device: str, judged: bool) -> Dict[str, Any]:
    row: Dict[str, Any] = {"quantity": name, "replay": replay, "recorded": rec, "judged": judged}
    if replay is None or not math.isfinite(replay) or rec == 0:
        row["ok"] = False if judged else None
        return row
    ratio = replay / rec
    row["ratio"] = ratio
    ok = abs(ratio - 1.0) <= EAGER_REL_CUDA
    row["ok"] = ok if judged else None
    row["within"] = ok
    return row


def validity(direct: Dict[int, Dict[str, Any]], canary: Dict[int, Dict[str, Any]], record: Dict[str, Any],
             device: str) -> Dict[str, Any]:
    """The replay vs the dump's own record (module docstring, VALIDITY). ``direct`` / ``canary``: per slice, one
    repeat's reads (the first). The eager numbers are read from the canary arm when it ran (exactly the record's
    computation), else from the direct arms."""
    rows: List[Dict[str, Any]] = []
    watches = record.get("watch") or {}
    for s, d in direct.items():
        rk = watches.get(RECORD_KEY.get(s, ""))
        if not rk:
            rows.append({"slice": s, "quantity": "record", "ok": None, "note": "the record holds no numbers for it"})
            continue
        cm = rk.get("clip_margin") or {}
        src = canary.get(s, {}).get("watch") if s in canary else None
        params = canary[s]["all_params"] if s in canary else d["per_param"]
        glob = (src or d["watch"])["global"]
        for q, want in (("relocated_rows", cm.get("relocated_rows")), ("rows", cm.get("rows"))):
            rows.append({"slice": s, "quantity": q, "replay": d[q], "recorded": want, "judged": True,
                         "ok": want is not None and int(d[q]) == int(want)})
        if cm.get("nearest_unmoved") is not None:
            rows.append({"slice": s, "quantity": "nearest_unmoved", "replay": d["nearest_unmoved"],
                         "recorded": cm["nearest_unmoved"], "judged": True,
                         "ok": abs(d["nearest_unmoved"] - float(cm["nearest_unmoved"])) <= NEAREST_ABS})
        g = rk.get("global") or {}
        if "drift_grad" in g:
            rows.append(dict(_cmp("drift_grad", glob.get("drift_grad"), float(g["drift_grad"]), device,
                                  device == "cuda"), slice=s))
        recs: Dict[str, float] = {}
        w = rk.get("worst") or {}
        if w.get("param") and "eager" in w:
            recs[w["param"]] = float(w["eager"])
        if s == 0:
            for n, _c, e in _DETAIL.findall(record.get("detail", "")):
                recs.setdefault(n, float(e))
        for n, e in recs.items():
            rows.append(dict(_cmp(f"eager[{n}]", (params.get(n) or {}).get("eager"), e, device, device == "cuda"),
                             slice=s))
        for n, pd in (rk.get("params") or {}).items():
            if "eager" in pd:
                rows.append(dict(_cmp(f"eager[{n}]", (params.get(n) or {}).get("eager"), float(pd["eager"]), device,
                                      device == "cuda"), slice=s))
    bad = [r for r in rows if r.get("judged") and r.get("ok") is False]
    return {"valid": not bad, "failures": bad, "rows": rows,
            "tolerances": {"state": "relocated_rows / rows exact; nearest_unmoved abs <= %g" % NEAREST_ABS,
                           "eager": (f"relative <= {EAGER_REL_CUDA:g} (cuda)" if device == "cuda" else
                                     "cpu: reported with the ratio replay / recorded, never judged")}}


def stability(reps: List[Dict[str, Any]], slices: Sequence[int]) -> Dict[str, Any]:
    """Cbf (and E32 / Ebf) across repeats: distinct gradient sha256s per slice; each edge_bias.* compiled distance's
    spread (max - min) across repeats, from the canary arm and from the direct Cbf arm."""
    out: Dict[str, Any] = {}
    for s in slices:
        key = str(s)
        shas = {a: sorted({r["slices"][key]["direct"]["grad_sha256"][a] for r in reps if key in r["slices"]})
                for a in ("E32", "Ebf", "Cbf")}
        spread: Dict[str, Any] = {}
        for arm in ("canary", "direct"):
            vals: Dict[str, List[float]] = {}
            for r in reps:
                sr = r["slices"].get(key, {}).get(arm)
                if not sr:
                    continue
                for n, d in sr["watch"]["edge_bias"].items():
                    if d.get("compiled") is not None:
                        vals.setdefault(n, []).append(float(d["compiled"]))
            spread[arm] = {n: {"min": min(v), "max": max(v), "spread": max(v) - min(v), "n": len(v)}
                           for n, v in vals.items()}
        shas = {a: v for a, v in shas.items() if v and not v[0].startswith("SKIPPED")}   # Cbf on CPU: no arm ran
        out[key] = {"distinct_grad_sha256": {a: len(v) for a, v in shas.items()},
                    "bitwise_stable": {a: len(v) == 1 for a, v in shas.items()},
                    "edge_bias_compiled_spread": spread}
    return out


# ------------------------------------------------------------------------------------------------ the driver
def _write(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def _say(line: str) -> None:
    print(line, flush=True)


def run(a: argparse.Namespace) -> int:
    import torch
    dump = Path(a.dump).expanduser().resolve()
    sha = check_dump_sha(dump)
    run_dir = Path(a.run_dir).expanduser().resolve()
    record = recorded(Path(a.verdicts).expanduser() if a.verdicts else dump.parent / "canary_verdicts.jsonl",
                      dump.name)
    out = check_out_dir(Path(a.out))
    slices = tuple(int(x) for x in str(a.slices).split(",") if x.strip())
    if FIRED_SLICE not in slices or any(s not in (0, 1, 2) for s in slices):
        raise Refusal(f"--slices {slices}: each of 0 / 1 / 2, and the rule's slice {FIRED_SLICE} must be among them")
    if a.repeats < 1:
        raise Refusal("--repeats must be >= 1")
    rows = ((record.get("watch") or {}).get("scheduled") or {}).get("clip_margin", {}).get("rows")
    meta_b = json.loads((run_dir / "metadata.json").read_text()).get("batch_size") if (
        run_dir / "metadata.json").is_file() else None
    batch = int(a.batch or rows or meta_b or 0)
    if not batch or (rows and int(rows) != batch) or (meta_b and int(meta_b) != batch):
        raise Refusal(f"the canary's B: --batch {a.batch}, the record's rows {rows}, the run's batch_size {meta_b}")
    device = a.device
    if device == "cuda":
        if not torch.cuda.is_available():
            raise Refusal("--device cuda but CUDA is not available")
        torch.cuda.set_device(0)
    elif a.torch_threads:
        torch.set_num_threads(int(a.torch_threads))
    torch.set_float32_matmul_precision("highest")      # the canary's one precision
    out.mkdir(parents=True, exist_ok=True)
    compiled = device == "cuda" or bool(a.compile_backend)
    t0 = time.perf_counter()
    model, info = build_model(run_dir, dump, device, batch)
    if compiled:
        install_compiled(model, a.compile_backend or None)
    meta = {"dump": str(dump), "sha256": sha, "run_dir": str(run_dir), "device": device, "batch": batch,
            "slices": list(slices), "repeats": a.repeats, "torch": torch.__version__, "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "threads": torch.get_num_threads(), "compile_backend": a.compile_backend or ("inductor" if compiled else None),
            "record_update": record.get("update"), "record_verdict": record.get("verdict"), "build": info,
            "build_seconds": time.perf_counter() - t0, "argv": sys.argv[1:], "rule": RULE_TEXT}
    try:
        from utils.git import head_sha
        meta["commit"] = str(head_sha())
    except Exception:                                         # a report field only
        meta["commit"] = "unknown"
    _write(out / "meta.json", meta)
    _say(f"[replay] learner built in {meta['build_seconds']:.0f}s on {device} (B {batch}, slices {list(slices)}, "
         f"init record {info['init_record']}, margin {info['margin']:.3f})")
    from agents.model.compile_regions import BF16_ENVELOPE_K
    k = float(BF16_ENVELOPE_K["param"]) if BF16_ENVELOPE_K else float("nan")
    reps: List[Dict[str, Any]] = []
    classes: List[Dict[int, str]] = []
    first_direct: Dict[int, Dict[str, Any]] = {}
    first_canary: Dict[int, Dict[str, Any]] = {}
    for i in range(a.repeats):
        rep: Dict[str, Any] = {"repeat": i, "slices": {}}
        cls: Dict[int, str] = {}
        for s in slices:
            sr: Dict[str, Any] = {}
            if compiled:
                c = canary_check(model, s, batch)
                c["class"] = classify(c["refused"], c["watch"]["edge_bias"], k)
                cls[s] = c["class"]
                sr["canary"] = c
            else:
                sr["canary"] = None
                sr["canary_skipped"] = "no compiled arm on this device (Cbf is CUDA-only)"
            sr["direct"] = direct_arms(model, s, batch, compiled)
            rep["slices"][str(s)] = sr
            if i == 0:
                first_direct[s] = sr["direct"]
                if sr["canary"]:
                    first_canary[s] = sr["canary"]
            eb = sr["canary"]["watch"]["edge_bias"] if sr["canary"] else sr["direct"]["watch"]["edge_bias"]
            top = sorted(eb.items(), key=lambda kv: -float(kv[1].get("ratio") or kv[1].get("eager") or 0))[:2]
            _say(f"[replay] repeat {i} slice {s}: "
                 + (f"canary {cls[s]} ({sr['canary']['seconds']:.1f}s); " if compiled else "eager only; ")
                 + "; ".join(f"{n.split('edge_bias.')[-1]} " + ", ".join(f"{q} {float(v):.4g}" for q, v in d.items()
                                                                       if q in ("ratio", "compiled", "eager")
                                                                       and v is not None)
                             for n, d in top)
                 + f"; relocated {sr['direct']['relocated_rows']}/{sr['direct']['rows']}")
        rep["classes"] = {str(s): c for s, c in cls.items()}
        reps.append(rep)
        classes.append(cls)
        _write(out / f"repeat_{i:02d}.json", rep)
        val = validity(first_direct, first_canary, record, device)
        summ = summarize(classes, val["valid"], compiled)
        _write(out / "summary.json", {"complete": i + 1 == a.repeats, **summ, "validity": val,
                                      "stability": stability(reps, slices), "meta": meta,
                                      "elapsed_s": time.perf_counter() - t0})
    _say(f"[replay] VERDICT {summ['verdict']} (counts {summ['counts']}); validity {'OK' if val['valid'] else 'FAILED'}"
         f" ({len(val['failures'])} failure(s)); {time.perf_counter() - t0:.0f}s -> {out}/summary.json")
    for r in val["rows"]:
        if r.get("judged") or "edge_bias" in str(r.get("quantity")) or r.get("quantity") == "drift_grad":
            _say(f"[replay]   validity slice {r.get('slice')}: {r.get('quantity')} replay {r.get('replay')} "
                 f"recorded {r.get('recorded')} (ratio {r.get('ratio')}) -> "
                 + ("judged OK" if r.get("ok") else "judged FAIL" if r.get("judged") else "reported"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main.ops.canary_dump_replay", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True, help="the canary's *_policy.pt (SHA256SUMS beside it)")
    p.add_argument("--run-dir", required=True, help="the run directory (model_config.json / metadata.json; read-only)")
    p.add_argument("--out", required=True, help="a NEW directory outside models/ and outside any checkout")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--slices", default=",".join(str(s) for s in DEFAULT_SLICES))
    p.add_argument("--verdicts", default="", help="the canary records (default: canary_verdicts.jsonl beside the dump)")
    p.add_argument("--batch", type=int, default=0, help="rows per read (default: the record's, = the run's B)")
    p.add_argument("--torch-threads", type=int, default=1, help="CPU only: torch threads (default 1: an unpinned pool "
                   "on the shared agent cores runs 34-125x slower)")
    p.add_argument("--compile-backend", default="", help=argparse.SUPPRESS)   # TEST seam: dynamo backend on CPU
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return run(a)
    except Refusal as exc:
        print(f"[replay] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
