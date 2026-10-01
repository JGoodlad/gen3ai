"""COMPILED vs EAGER, from a `torch.profiler` chrome trace — per phase of `train()`, per component.

The question (d) asks: of the update's time, how much is spent inside compiled graphs and how much
in eager ATen / Python? A trace answers it without touching the code under test, because every
piece of compiled work is ENCLOSED by a marker range on the thread that ran it:

  * ``CompiledFunction`` / ``CompiledFunctionBackward`` — AOTAutograd's forward / backward of a
    compiled training graph (the production `--compile-trainer` path, any backend);
  * ``k8.graph#<id>:fw|bw`` — this tool's own marker around every graph a capture compiled
    (inference graphs too, which have no `CompiledFunction`);
  * a ``triton_*`` cpu op — an Inductor kernel launch.

A HOST op (top-level `aten::*`, i.e. not nested in another aten op) is compiled iff one of its
ancestors is such a marker. A DEVICE event (kernel, memcpy, memset) inherits the verdict of the
runtime call that launched it (matched by `correlation`). The PHASE is the `k8.seg#<i>` range (see
`worker.SegmentMarker`) enclosing the launch in time — on any thread, because the backward runs on
autograd's device thread while the main thread waits inside the `backward` phase.

Eager host ops are attributed to a COMPONENT from the enclosing `python_function` frames (a trace
recorded ``with_stack=True``) through `attribution.classify`; an op under an autograd
`evaluate_function` is ``backward (eager autograd)``, keyed by its node name.
"""
from __future__ import annotations

import bisect
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from main.compile_inventory.attribution import classify, parse_profiler_frame

SEG_PREFIX = "k8.seg#"
COMPILED_NAMES = ("CompiledFunction", "CompiledFunctionBackward")
GRAPH_PREFIX = "k8.graph#"
DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
RUNTIME_CATS = ("cuda_runtime", "cuda_driver")
BACKWARD_EAGER = "backward (eager autograd)"


def _is_compiled_marker(name: str) -> bool:
    return (name in COMPILED_NAMES or name.startswith(GRAPH_PREFIX) or name.startswith("triton_")
            or name == "autograd::engine::evaluate_function: CompiledFunctionBackward")


_KEEP = ("ph", "cat", "name", "ts", "dur", "pid", "tid")


def _slim(e: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: e[k] for k in _KEEP if k in e}
    corr = (e.get("args") or {}).get("correlation")
    if corr is not None:
        out["args"] = {"correlation": corr}
    return out


def _stream_events(f: Any) -> Iterator[Dict[str, Any]]:
    """`torch.profiler`'s chrome export writes each trace event as its own block, opened by a
    line that is exactly ``  {`` and closed by ``  },`` / ``  }`` (2.5.1 and 2.8 alike). Parse one
    block at a time and keep only the fields the classifier reads — so a multi-GB trace never
    exists as one Python object (the 2026-09-30 host OOMs; `memcap.py`)."""
    in_events = False
    block: List[str] = []
    for line in f:
        if not in_events:
            if line.strip().startswith('"traceEvents"'):
                in_events = True
            continue
        if not block:
            if line.rstrip("\n") == "  {":
                block.append("{")
            elif line.strip().startswith("]"):
                return
            continue
        s = line.rstrip("\n")
        if s in ("  },", "  }"):
            block.append("}")
            yield _slim(json.loads("".join(block)))
            block = []
        else:
            block.append(s)


def load_trace(src: Union[str, Path, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The trace's events, SLIMMED (only the classifier's fields), read by streaming."""
    if isinstance(src, dict):
        return [_slim(e) for e in src.get("traceEvents", [])]
    p = Path(src)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as f:  # type: ignore[operator]
        evs = list(_stream_events(f))
    if not evs:                    # not the expected layout: the plain reader, refusing silence
        with opener(p, "rt") as f:  # type: ignore[operator]
            evs = [_slim(e) for e in json.load(f).get("traceEvents", [])]
    return evs


def _union(iv: Iterable[Tuple[float, float]]) -> float:
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s  # type: ignore[operator]
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s  # type: ignore[operator]
    return tot


def classify_events(events: Sequence[Dict[str, Any]],
                    segment_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Pure core of `classify_trace` (unit-tested on synthetic events). Times are microseconds in,
    milliseconds out."""
    xs = [e for e in events if e.get("ph") == "X" and "ts" in e]
    # phases: the k8.seg ranges, by time
    # HOST ranges only: a CUDA trace also carries a `gpu_user_annotation` copy of every range on
    # the device timeline, whose timestamps would corrupt the by-time phase lookup.
    segs = sorted(((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0.0)),
                    int(e["name"][len(SEG_PREFIX):])) for e in xs
                   if str(e.get("name", "")).startswith(SEG_PREFIX)
                   and e.get("cat") == "user_annotation"))
    seg_starts = [s[0] for s in segs]

    def phase_of(ts: float) -> str:
        i = bisect.bisect_right(seg_starts, ts) - 1
        if i < 0 or ts > segs[i][1]:
            return "(outside train)"
        idx = segs[i][2]
        if segment_names is not None and idx < len(segment_names):
            return segment_names[idx]
        return f"seg{idx}"

    host_cats = ("cpu_op", "user_annotation", "python_function") + RUNTIME_CATS
    by_tid: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for e in xs:
        if e.get("cat") in host_cats:
            by_tid[(e.get("pid"), e.get("tid"))].append(e)

    launch: Dict[Any, Tuple[bool, str]] = {}     # correlation -> (compiled, phase)
    ph_host: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    comp_ops: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    bw_nodes: Dict[str, float] = defaultdict(float)
    for _tid, evs in by_tid.items():
        evs.sort(key=lambda e: (float(e["ts"]), -float(e.get("dur", 0.0))))
        stack: List[Tuple[float, Dict[str, Any], bool, bool, Optional[str]]] = []
        # entries: (end, event, compiled_here_or_above, aten_above, bw_node_above)
        for e in evs:
            ts = float(e["ts"])
            end = ts + float(e.get("dur", 0.0))
            while stack and stack[-1][0] <= ts:
                stack.pop()
            parent_comp = stack[-1][2] if stack else False
            parent_aten = stack[-1][3] if stack else False
            parent_bw = stack[-1][4] if stack else None
            name = str(e.get("name", ""))
            cat = e.get("cat")
            comp = parent_comp or _is_compiled_marker(name)
            bw = parent_bw
            if name.startswith("autograd::engine::evaluate_function: "):
                bw = name.split(": ", 1)[1]
            is_aten = cat == "cpu_op" and name.startswith("aten::")
            if cat in RUNTIME_CATS:
                corr = (e.get("args") or {}).get("correlation")
                if corr is not None:
                    launch[corr] = (comp, phase_of(ts))
            if is_aten and not parent_aten:
                phase = phase_of(ts)
                key = "compiled" if comp else "eager"
                dur_ms = float(e.get("dur", 0.0)) / 1e3
                ph_host[phase][f"{key}_ops"] += 1
                ph_host[phase][f"{key}_host_ms"] += dur_ms
                if not comp:
                    if bw is not None:
                        c = BACKWARD_EAGER
                        bw_nodes[bw] += 1
                    else:
                        frames = [f for f in (parse_profiler_frame(s[1]["name"]) for s in stack
                                              if s[1].get("cat") == "python_function") if f]
                        c = classify(frames) if frames else "(no python stack)"
                    comp_ops[c]["eager_ops"] += 1
                    comp_ops[c]["eager_host_ms"] += dur_ms
            stack.append((end, e, comp, parent_aten or is_aten, bw))

    dev = [e for e in xs if e.get("cat") in DEVICE_CATS]
    ph_dev: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    unmatched = 0
    for e in dev:
        corr = (e.get("args") or {}).get("correlation")
        comp, phase = launch.get(corr, (None, "(unmatched)"))
        if comp is None:
            unmatched += 1
        kind = "kernel" if e.get("cat") == "kernel" else "memop"
        key = f"{'compiled' if comp else 'eager'}_{kind}_ms"
        ph_dev[phase][key] += float(e.get("dur", 0.0)) / 1e3
        ph_dev[phase][f"{'compiled' if comp else 'eager'}_{kind}s"] += 1
    seg_wall: Dict[str, float] = defaultdict(float)
    for s, en, idx in segs:
        nm = (segment_names[idx] if segment_names is not None and idx < len(segment_names)
              else f"seg{idx}")
        seg_wall[nm] += (en - s) / 1e3
    span = ((max(float(e["ts"]) + float(e.get("dur", 0.0)) for e in xs)
             - min(float(e["ts"]) for e in xs)) / 1e3) if xs else 0.0
    train_span = ((segs[-1][1] - segs[0][0]) / 1e3) if segs else span
    gpu_busy = _union((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0.0))) for e in dev
                      if (not segs) or segs[0][0] <= float(e["ts"]) <= segs[-1][1] + 5e6) / 1e3
    phases = sorted(set(ph_host) | set(ph_dev) | set(seg_wall))
    table = {}
    for p in phases:
        row: Dict[str, float] = {"host_wall_ms": seg_wall.get(p, 0.0)}
        row.update(ph_host.get(p, {}))
        row.update(ph_dev.get(p, {}))
        table[p] = row

    def _tot(k: str) -> float:
        return sum(r.get(k, 0.0) for r in table.values())
    ck, ek = _tot("compiled_kernel_ms"), _tot("eager_kernel_ms")
    cm, em = _tot("compiled_memop_ms"), _tot("eager_memop_ms")
    co, eo = _tot("compiled_ops"), _tot("eager_ops")
    totals = {
        "trace_span_ms": span, "train_span_ms": train_span, "gpu_busy_ms": gpu_busy,
        "gpu_idle_ms": max(0.0, train_span - gpu_busy) if dev else None,
        "compiled_kernel_ms": ck, "eager_kernel_ms": ek, "compiled_memop_ms": cm,
        "eager_memop_ms": em,
        "compiled_kernel_share": (ck / (ck + ek)) if (ck + ek) else None,
        "compiled_share_of_train_wall": (ck / train_span) if (dev and train_span) else None,
        "compiled_host_ops": co, "eager_host_ops": eo,
        "compiled_host_op_share": (co / (co + eo)) if (co + eo) else None,
        "unmatched_device_events": unmatched, "device_events": len(dev),
    }
    return {"totals": totals, "phases": table,
            "eager_by_component": {k: dict(v) for k, v in sorted(comp_ops.items())},
            "backward_eager_nodes_top": sorted(bw_nodes.items(), key=lambda kv: -kv[1])[:25]}


def classify_trace(src: Union[str, Path, Dict[str, Any]],
                   segment_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    return classify_events(load_trace(src), segment_names)
