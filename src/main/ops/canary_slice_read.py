"""THE SLICE-2 DISCRIMINATION READ (2026-10-29): is a slice-2-only canary refusal the compiler, or the function?

THE EVENTS. Three canary states refuse R1's bf16 per-parameter rule on the confirmation slice 2 (fixture rows 0-31) and
pass slice 1 (rows 32-63), each on a different parameter family: X48 s3502 u200 (`edge_bias.s1_map / c2_map`), N1-P
s3505 u100 (`flat_intent_head.hidden.*`), X26c @482M at the stage-2 margin 1.2 (`spread_belief.nature_head.weight`);
N1-P s3501 u200 (`hp_type_belief_head.norm.bias`) refused slice 0 and neither half. The CPU read
(`designs/research_state/measurements/canary_slice2_2026-10-29/`) found, on every one of them: (1) the refusing
parameter's gradient on slices 0 / 2 is carried by one to three FIRST-HALF rows (9; 18 / 22; 24), which slice 1 lacks
(X48's two parameters have no gradient at all on slice 1, so slice 1 can never confirm them); (2) the fp32 FUNCTION'S
OWN gradient of that parameter moves, under the canary's half-ulp weight dither, by as much as the compiled arm's
distance (median 0.1-1.0, max 0.18-2.4), while eager bf16 sits 5-50x BELOW that null. So the ratio's denominator is
the outlier, not the numerator. What the CPU cannot read is the compiled arm under the same dithers; this module does.

WHAT IT READS, per state (a dump, or a checkpoint as the anchor), in ONE process (one compile; on CUDA the compile is
~4 of the ~8 minutes):

  1. THE CANARY, `CompileCanary._check` verbatim (`canary_dump_replay.canary_check`) on slices 0, 2, 1 x
     ``--repeats`` (2: CUDA is run-to-run bit-identical, so the second read is the determinism check), and once on each
     ``--row-set LABEL=ROWS`` (the fixture tiled from those rows only, e.g. slice 2 without its carrier row, slice 1
     with it). The dump's own VALIDITY / the anchor's self-validity are `canary_dump_replay`'s, unchanged.
  2. THE DITHER READ (``--dither N``), on slices 0, 2, 1: for seeds 0..N-1 the policy's weights x (1 + 2^-9 N(0, 1))
     (`diag_bf16_canary.dithered`, name-keyed, restored bit-exactly); per seed and per judged parameter p:
       * ``null``     d(E32_s, E32_0) — the fp32 function's own move, both on the base weights' relocated rows;
       * ``eager``    d(Ebf_s, E32_s) and ``compiled`` d(Cbf_s, E32_s) — the canary's two distances at the seed's
         weights, on the rows relocated on THOSE weights (as the canary relocates);
     d(x, y) = ||x_p - y_p|| / ||y_p||, the rule's per-parameter distance.

THE RULE (PRE-DECLARED 2026-10-29, before any GPU read; deterministic, standing rule 8). For each target (a ``--watch``
parameter on slice 0 and on slice 2, where it is judged), with c = the median over seeds of ``compiled`` and n = the
maximum over seeds of ``null``:

  * ``REAL``          iff c > n x (1 + `BOUNDARY_REL`): the compiled gradient sits outside every rounding-sized move of the
                      fp32 function itself — compiled bf16 computes this parameter's gradient somewhere the function
                      does not go (hypothesis (i): a real numerics difference that training, which runs compiled, has);
  * ``CONDITIONING``  iff c < n x (1 - `BOUNDARY_REL`): the compiled distance is inside the function's own rounding
                      sensitivity, so the per-parameter ratio judges the function's conditioning on these rows;
  * ``BOUNDARY``      otherwise (within 1 % of n: never a pass by chance).

The state reads ``REAL`` iff any target is REAL; ``CONDITIONING`` iff every read target is; else ``INCONCLUSIVE``.
``INVALID-REPLAY`` overrides when the replay's validity fails. Reported beside it, never judged: per target the
canary statistic's per-seed share above K (c_s / max(e_s, floor) > 12) and eager's median against the null's.

    python -m main.ops.canary_slice_read --dump <x>_policy.pt --run-dir models/<run> --out D --device cuda \\
        --watch features_extractor.edge_bias.s1_map.weight --row-set s2_minus_18_22=0-17,19-21,23-31 --dither 16
    python -m main.ops.canary_slice_read --anchor models/<run>/checkpoints/<zip> --run-dir models/<run> \\
        --canary-shift-max 0.3 --out D --device cuda --watch <param> --dither 16
    python -m main.ops.canary_slice_read verdict --root D1,D2,...    # the window's CPU verdict over the states

EXIT: 0 the read ran (whatever it found), 2 a refusal, 1 an error. No wall budget inside (the window's step has one).
Nothing here trains, writes under ``models/`` or a checkout, or changes a canary rule.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from main.ops import canary_dump_replay as R

#: the dither read's slices: the scheduled check and the two halves
DITHER_SLICES: Tuple[int, ...] = (0, 2, 1)
#: the targets' slices: the scheduled check and the slice that fired
TARGET_SLICES: Tuple[int, ...] = (0, 2)
#: a target within this RELATIVE distance of the null's maximum is BOUNDARY (rule 8)
BOUNDARY_REL = 0.01
#: the dither's relative sigma (half a bf16 ulp): `diag_bf16_canary.DITHER_REL`
DITHER_REL = 2.0 ** -9
#: below this share of the largest per-parameter reference norm a parameter is not judged (`_PARAM_GRAD_FLOOR`)
PARAM_GRAD_FLOOR = 1e-3
STATE_VERDICTS = ("REAL", "CONDITIONING", "INCONCLUSIVE", "INVALID-REPLAY", "EAGER-ONLY")
RULE_TEXT = ("per target (a --watch parameter on slices 0 and 2, where judged): c = median over the dither seeds of "
             "d(Cbf_s, E32_s), n = max over seeds of d(E32_s, E32_0); REAL iff c > n(1+0.01), CONDITIONING iff "
             "c < n(1-0.01), else BOUNDARY. State: REAL iff any target REAL; CONDITIONING iff every read target is; else "
             "INCONCLUSIVE; INVALID-REPLAY overrides when the replay's validity fails; EAGER-ONLY without a compiled arm")


# ------------------------------------------------------------------------------------------------ row sets
def parse_rows(spec: str, n_rows: int = 64) -> List[int]:
    """``"0-8,10-31"`` -> the fixture rows, in order, each once; every row in [0, n_rows). Refuses an empty set, a
    duplicate and fewer than two rows (one row tiled has a zero advantage spread: normalised to all zeros)."""
    rows: List[int] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        lo, _, hi = part.partition("-")
        try:
            a, b = int(lo), int(hi) if hi else int(lo)
        except ValueError:
            raise R.Refusal(f"row set {spec!r}: {part!r} is not a row or a range a-b") from None
        if a > b:
            raise R.Refusal(f"row range {part!r} runs backwards")
        rows.extend(range(a, b + 1))
    if len(rows) < 2:
        raise R.Refusal(f"row set {spec!r}: at least two rows (one row's normalised advantages are all zero)")
    if len(set(rows)) != len(rows):
        raise R.Refusal(f"row set {spec!r} names a row twice")
    bad = [r for r in rows if not 0 <= r < n_rows]
    if bad:
        raise R.Refusal(f"row set {spec!r}: rows {bad} are outside the fixture's {n_rows}")
    return rows


def parse_row_sets(specs: Sequence[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for s in specs:
        label, eq, rows = s.partition("=")
        if not eq or not label.strip():
            raise R.Refusal(f"--row-set {s!r}: LABEL=ROWS")
        if label in out or label.strip() in ("0", "1", "2"):
            raise R.Refusal(f"--row-set label {label!r} repeats or shadows a slice")
        out[label.strip()] = parse_rows(rows)
    return out


@contextlib.contextmanager
def custom_rows(rows: Optional[Sequence[int]]) -> Iterator[None]:
    """Inside, `compile_trainer.fixture_index` tiles ``rows`` (whatever slice is asked); restored on exit. None = no
    change. This process only: the canary's own code reads the fixture through that one function."""
    if rows is None:
        yield
        return
    import numpy as np

    from agents.model import compile_trainer as ct
    real = ct.fixture_index
    arr = np.asarray(list(rows), dtype=np.int64)

    def tiled(batch: int, n_rows: int, slice_: int = 0) -> "np.ndarray":
        if int(arr.max()) >= n_rows:
            raise R.Refusal(f"a row set names row {int(arr.max())} of a {n_rows}-row fixture")
        return arr[np.arange(int(batch)) % len(arr)]
    ct.fixture_index = tiled
    try:
        yield
    finally:
        ct.fixture_index = real


# ------------------------------------------------------------------------------------------------ the dither read
def _split_norms(flat: Any, sizes: Sequence[int]) -> List[Any]:
    import torch
    return list(torch.split(flat.detach().double(), list(sizes)))


def _rel(a: Any, b: Any) -> float:
    nb = float(b.norm())
    v = float((a - b).norm()) / nb if nb > 0 else float("nan")
    return v if math.isfinite(v) else float("nan")


def dither_read(model: Any, slice_: int, batch: int, compiled: bool, seeds: int) -> Dict[str, Any]:
    """Per judged parameter, per seed: ``null`` d(E32_s, E32_0), ``eager`` d(Ebf_s, E32_s), ``compiled`` d(Cbf_s,
    E32_s) (module docstring). The base weights' parameters judged: reference norm > `PARAM_GRAD_FLOOR` x the largest."""
    from agents.model import compile_regions as cr
    from agents.model.compile_gate_probe import grad_parameters
    from agents.training.instrumented_ppo.micro_step import micro_step
    from main.ops.diag_bf16_canary import dithered
    t0 = time.perf_counter()
    model.policy.set_training_mode(True)
    names = [n for n, _ in grad_parameters(model, model.policy.features_extractor)]
    b = cr.r1_batch(model, batch, slice_)
    args0, fix0 = cr.relocate_clip_boundary(model, cr._r1_args(model, b))
    ref0 = cr._r1_reference(model, args0)
    sizes = [int(x) for x in ref0["grad_sizes"].tolist()]
    r0 = _split_norms(ref0["grad"], sizes)
    norms = [float(x.norm()) for x in r0]
    top = max(norms) if norms else 0.0
    judged = [i for i, nm in enumerate(norms) if top > 0 and nm > PARAM_GRAD_FLOOR * top]
    floor = float(cr.R1_PARAM_BAR[cr.weights_regime(model)])
    per: Dict[str, Dict[str, List[float]]] = {names[i]: {"null": [], "eager": [], "compiled": []} for i in judged}
    relocated: List[int] = []
    for s in range(int(seeds)):
        with dithered([model], s, DITHER_REL):
            null = _split_norms(cr._r1_reference(model, args0)["grad"], sizes)
            args, fix = cr.relocate_clip_boundary(model, cr._r1_args(model, b))
            relocated.append(int(fix.moved.sum()))
            ref = _split_norms(cr._r1_reference(model, args)["grad"], sizes)
            eag = _split_norms(cr._r1_arm(model, micro_step, args)["grad"], sizes)
            comp = (_split_norms(cr._r1_arm(model, model._compiled_micro_step, args)["grad"], sizes)
                    if compiled else None)
        for i in judged:
            d = per[names[i]]
            d["null"].append(_rel(null[i], r0[i]))
            d["eager"].append(_rel(eag[i], ref[i]))
            if comp is not None:
                d["compiled"].append(_rel(comp[i], ref[i]))
        cr.release_gate_memory(model)
    for p in model.policy.parameters():
        p.grad = None
    return {"slice": slice_, "seeds": int(seeds), "rel": DITHER_REL, "floor": floor, "batch": batch,
            "relocated_base": int(fix0.moved.sum()), "relocated_per_seed": relocated, "judged": len(judged),
            "params": per, "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------------------------------------------ the rule
def target_verdict(d: Dict[str, List[float]], floor: float, k: float) -> Dict[str, Any]:
    """One target's class (module docstring, THE RULE) from its per-seed ``null`` / ``eager`` / ``compiled``."""
    null = [x for x in d.get("null", []) if math.isfinite(x)]
    comp = [x for x in d.get("compiled", []) if math.isfinite(x)]
    eag = [x for x in d.get("eager", []) if math.isfinite(x)]
    out: Dict[str, Any] = {"seeds": len(null)}
    if not null or not comp or len(comp) != len(d.get("compiled", [])) or len(null) != len(d.get("null", [])):
        out["class"] = "UNREAD" if not comp else "NONFINITE"
        return out
    c, n = statistics.median(comp), max(null)
    out.update(compiled_median=c, null_max=n, null_median=statistics.median(null),
               eager_median=statistics.median(eag) if eag else None)
    if n <= 0:
        out["class"] = "REAL" if c > 0 else "BOUNDARY"
    elif c > n * (1 + BOUNDARY_REL):
        out["class"] = "REAL"
    elif c < n * (1 - BOUNDARY_REL):
        out["class"] = "CONDITIONING"
    else:
        out["class"] = "BOUNDARY"
    if eag and len(eag) == len(comp):
        ratios = [ci / max(ei, floor) for ci, ei in zip(comp, eag)]
        out["canary_ratio_per_seed"] = ratios
        out["share_over_k"] = sum(1 for r in ratios if not r <= k) / len(ratios)
    if eag and out["null_median"] > 0:
        out["eager_median_over_null_median"] = out["eager_median"] / out["null_median"]
    return out


def state_verdict(dither: Dict[str, Dict[str, Any]], watch: Sequence[str], k: float, valid: bool,
                  compiled: bool) -> Dict[str, Any]:
    """The state's verdict over its targets (``dither``: per slice key, `dither_read`'s output)."""
    targets: List[Dict[str, Any]] = []
    for s in TARGET_SLICES:
        dr = dither.get(str(s))
        for p in watch:
            if dr is None or p not in dr["params"]:
                targets.append({"param": p, "slice": s, "class": "NOT-JUDGED",
                                "why": "no dither read for the slice" if dr is None else "below the judged floor"})
                continue
            targets.append({"param": p, "slice": s, **target_verdict(dr["params"][p], float(dr["floor"]), k)})
    read = [t for t in targets if t["class"] in ("REAL", "CONDITIONING", "BOUNDARY")]
    if not valid:
        v = "INVALID-REPLAY"
    elif not compiled:
        v = "EAGER-ONLY"
    elif any(t["class"] == "REAL" for t in read):
        v = "REAL"
    elif read and all(t["class"] == "CONDITIONING" for t in read):
        v = "CONDITIONING"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "rule": RULE_TEXT, "targets": targets}


def window_verdict(states: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Over every state: hypothesis (i) is LIVE iff some state reads REAL; RULED OUT at these states iff every state
    reads CONDITIONING; else INCONCLUSIVE (a missing, invalid or inconclusive state)."""
    got = {k: (v or {}).get("verdict", "MISSING") for k, v in states.items()}
    if any(v == "REAL" for v in got.values()):
        h = "LIVE"
    elif got and all(v == "CONDITIONING" for v in got.values()):
        h = "RULED OUT AT THESE STATES"
    else:
        h = "INCONCLUSIVE"
    return {"hypothesis_i_real_numerics": h, "states": got,
            "rule": "(i) LIVE iff some state reads REAL; RULED OUT AT THESE STATES iff every state reads CONDITIONING; "
                    "else INCONCLUSIVE"}


# ------------------------------------------------------------------------------------------------ the driver
def _canary_summary(c: Dict[str, Any], watch: Sequence[str]) -> Dict[str, Any]:
    ap = c.get("all_params") or {}
    return {"refused": c["refused"], "text": c.get("text", "")[:600], "worst": c["watch"].get("worst"),
            "clip_margin": c["watch"].get("clip_margin"),
            "watch": {p: ap.get(p, "not judged") for p in watch}}


def run(a: argparse.Namespace) -> int:
    import torch
    if bool(a.dump) == bool(a.anchor):
        raise R.Refusal("exactly one of --dump or --anchor")
    run_dir = Path(a.run_dir).expanduser().resolve()
    record: Dict[str, Any] = {}
    dump = anchor = None
    if a.dump:
        dump = Path(a.dump).expanduser().resolve()
        sha: Optional[str] = R.check_dump_sha(dump)
        record = R.recorded(dump.parent / "canary_verdicts.jsonl", dump.name)
        if a.canary_shift_max is not None:
            raise R.Refusal("--canary-shift-max is ANCHOR mode only: a dump replays the margin its own record judged")
    else:
        anchor = R.check_anchor(Path(a.anchor), run_dir)
        sha = None
    watch = [p for p in a.watch.split(",") if p.strip()]
    if not watch:
        raise R.Refusal("--watch names no parameter: the rule's targets are the refusing parameters")
    row_sets = parse_row_sets(a.row_set)
    if a.repeats < 1 or a.dither < 0:
        raise R.Refusal("--repeats >= 1 and --dither >= 0")
    shift_max = R.resolve_shift_arg(a.canary_shift_max)
    row_bars = R.replay_row_bars(record)
    out = R.check_out_dir(Path(a.out))
    rows = ((record.get("watch") or {}).get("scheduled") or {}).get("clip_margin", {}).get("rows")
    meta_b = json.loads((run_dir / "metadata.json").read_text()).get("batch_size") if (
        run_dir / "metadata.json").is_file() else None
    batch = int(a.batch or rows or meta_b or 0)
    if not batch or (rows and int(rows) != batch) or (meta_b and int(meta_b) != batch):
        raise R.Refusal(f"the canary's B: --batch {a.batch}, the record's rows {rows}, the run's batch_size {meta_b}")
    if a.device == "cuda":
        if not torch.cuda.is_available():
            raise R.Refusal("--device cuda but CUDA is not available")
        torch.cuda.set_device(0)
    else:
        torch.set_num_threads(int(a.torch_threads))
    torch.set_float32_matmul_precision("highest")
    out.mkdir(parents=True, exist_ok=True)
    compiled = a.device == "cuda" or bool(a.compile_backend)
    t0 = time.perf_counter()
    model, info = R.build_model(run_dir, dump, a.device, batch, anchor=anchor, shift_max=shift_max)
    if compiled:
        R.install_compiled(model, a.compile_backend or None)
    from agents.model.compile_regions import BF16_ENVELOPE_K
    k = float(BF16_ENVELOPE_K["param"]) if BF16_ENVELOPE_K else float("nan")
    meta = {"dump": str(dump) if dump else None, "anchor": str(anchor) if anchor else None, "sha256": sha,
            "run_dir": str(run_dir), "golden_rows_sha256": R.golden_rows_sha(), "device": a.device, "batch": batch,
            "watch": watch, "row_sets": row_sets, "repeats": a.repeats, "dither": a.dither, "k": k,
            "torch": torch.__version__, "build": info, "rule": RULE_TEXT, "argv": sys.argv[1:],
            "record_update": record.get("update"), "record_verdict": record.get("verdict")}
    R._write(out / "meta.json", meta)
    R._say(f"[slice-read] built in {time.perf_counter() - t0:.0f}s on {a.device} (B {batch}, margin "
           f"{info['margin']:.3f}, watch {watch})")
    reps: List[Dict[str, Any]] = []
    for i in range(a.repeats):
        rep: Dict[str, Any] = {"repeat": i, "slices": {}}
        for s in DITHER_SLICES:
            sr: Dict[str, Any] = {"canary": R.canary_check(model, s, batch, row_bars) if compiled else None,
                                  "direct": R.direct_arms(model, s, batch, compiled)}
            rep["slices"][str(s)] = sr
            c = sr["canary"]
            R._say(f"[slice-read] repeat {i} slice {s}: "
                   + (f"canary {'REFUSED' if c['refused'] else 'pass'}; worst {c['watch'].get('worst')}" if c
                      else "eager only"))
        reps.append(rep)
    val = (R.validity({s: reps[0]["slices"][str(s)]["direct"] for s in DITHER_SLICES},
                      {s: reps[0]["slices"][str(s)]["canary"] for s in DITHER_SLICES if reps[0]["slices"][str(s)]["canary"]},
                      record, a.device) if dump is not None else R.self_validity(reps, DITHER_SLICES))
    canary = {"repeats": [{str(s): _canary_summary(r["slices"][str(s)]["canary"], watch) for s in DITHER_SLICES}
                          for r in reps] if compiled else None,
              "stability": R.stability(reps, DITHER_SLICES), "validity": val}
    for label, rws in row_sets.items():
        with custom_rows(rws):
            c = R.canary_check(model, 0, batch, row_bars) if compiled else None
        canary.setdefault("row_sets", {})[label] = {"rows": rws, **(_canary_summary(c, watch) if c else {})}
        if c:
            R._say(f"[slice-read] row set {label}: canary {'REFUSED' if c['refused'] else 'pass'}; "
                   f"watch {[(p.split('.')[-2], (c['all_params'].get(p) or {}).get('ratio')) for p in watch]}")
    R._write(out / "canary.json", canary)
    dither: Dict[str, Any] = {}
    for s in DITHER_SLICES if a.dither else ():
        dither[str(s)] = dither_read(model, s, batch, compiled, a.dither)
        R._write(out / "dither.json", dither)
        R._say(f"[slice-read] dither slice {s}: {a.dither} seeds in {dither[str(s)]['seconds']:.0f}s")
    v = state_verdict(dither, watch, k, bool(val.get("valid")), compiled) if a.dither else {"verdict": "NO-DITHER"}
    R._write(out / "verdict.json", {**v, "complete": True, "elapsed_s": time.perf_counter() - t0, "meta": meta})
    R._say(f"[slice-read] STATE VERDICT {v['verdict']}: "
           + "; ".join(f"{t['param'].split('.')[-2]}/{t['slice']} {t['class']}"
                       + (f" (c {t['compiled_median']:.3g} vs null max {t['null_max']:.3g})" if "null_max" in t else "")
                       for t in v.get("targets", []))
           + f" -> {out}/verdict.json")
    return 0


def run_verdict(roots: Sequence[str], out: Optional[str]) -> int:
    states: Dict[str, Optional[Dict[str, Any]]] = {}
    for r in roots:
        f = Path(r) / "verdict.json"
        states[Path(r).name] = json.loads(f.read_text()) if f.is_file() else None
    v = window_verdict(states)
    if out:
        R._write(Path(out), {**v, "roots": list(roots)})
    R._say(f"[slice-read] WINDOW: hypothesis (i) {v['hypothesis_i_real_numerics']} ({v['states']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main.ops.canary_slice_read", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", default="")
    p.add_argument("--anchor", default="")
    p.add_argument("--run-dir", default="")
    p.add_argument("--out", default="")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--watch", default="", help="comma list: the refusing parameter(s), the rule's targets")
    p.add_argument("--row-set", action="append", default=[], help="LABEL=ROWS (e.g. s2_minus_9=0-8,10-31); repeatable")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--dither", type=int, default=16)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--canary-shift-max", type=float, default=None)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--compile-backend", default="", help=argparse.SUPPRESS)   # TEST seam: dynamo backend on CPU
    p.add_argument("--root", default="", help="verdict mode: comma list of per-state out dirs")
    p.add_argument("--verdict-out", default="")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    verdict_mode = bool(argv) and argv[0] == "verdict"
    a = build_parser().parse_args(argv[1:] if verdict_mode else argv)
    try:
        if verdict_mode:
            if not a.root:
                raise R.Refusal("verdict needs --root D1,D2,...")
            return run_verdict([r for r in a.root.split(",") if r], a.verdict_out or None)
        if not a.run_dir or not a.out:
            raise R.Refusal("--run-dir and --out are required")
        return run(a)
    except R.Refusal as exc:
        print(f"[slice-read] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
