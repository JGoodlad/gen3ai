"""Render the inventory's JSON results as the markdown TABLES of the K8 readout.

`render([dir, ...])` reads each directory's ``trace_result.json`` (the trace stage) or
``time_analysis.json`` (the time stage, after `analyze`) and emits one section per result. The
prose — the fix plan, the region table, the acceptance criteria — is written by a person beside
these tables; this module only guarantees that every NUMBER in the readout came from a result file,
and names the file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

VIEWS = ("today", "whole_step")
LABELS = ("update1:first(diagnostics)", "update2:diag_skipped", "rollout:eval_no_grad",
          "update3:diag_skipped(steady)")


def _f(x: Any, nd: int = 1) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{100.0 * x:.1f}%"


def _md_escape(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _table(head: Sequence[str], rows: List[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_md_escape(c) for c in r) + " |")
    return "\n".join(out)


def _provenance(res: Dict[str, Any], src: Path) -> str:
    g = res.get("geometry", {})
    return (f"- source: `{src}`\n"
            f"- torch `{g.get('torch', res.get('torch'))}`, device `{g.get('device', '?')}`, "
            f"matmul `{g.get('matmul_precision', res.get('matmul_precision'))}`, git "
            f"`{str(res.get('git_head'))[:10]}`\n"
            f"- checkpoint `{res.get('checkpoint')}` (sha256 `{str(res.get('checkpoint_sha256'))[:12]}`), "
            f"buffer `{res.get('buffer')}`\n"
            f"- geometry: {g.get('rows')} rows, micro {g.get('micro_batch')} × "
            f"{g.get('micro_per_epoch')}/epoch, accum {g.get('grad_accum_steps')}, "
            f"epochs {g.get('n_epochs')}, diagnostics_every {g.get('diagnostics_every')}")


#: The readout's AREAS — by SOURCE FILE of the graph root / break site, which is the same on both
#: torches (a stack-based component is not: 2.8 logs frames above the compiled frame, 2.5.1 not).
AREAS = (
    ("extractor", ("agents/model/features_extractor.py", "agents/model/encoders.py",
                   "agents/model/extractor_forward.py",
                   "agents/model/team_transformer.py", "agents/model/damage_op")),
    ("heads", ("agents/model/policy.py", "agents/model/pointer_head.py",
               "sb3_contrib/common/maskable/policies.py", "stable_baselines3/common/policies.py")),
    ("masking+distribution", ("sb3_contrib/common/maskable/distributions.py",
                              "stable_baselines3/common/distributions.py")),
    ("train() fold (inline)", ("agents/training/instrumented_ppo/ppo.py",)),
    ("loss terms", ("agents/training/belief_bank.py", "agents/training/instrumented_ppo/value_terms.py",
                    "agents/model/opp_intent.py", "agents/training/instrumented_ppo/aux_terms.py",
                    "agents/training/opp_intent_labels.py")),
    ("diagnostics", ("agents/training/rank_metrics.py", "agents/training/grad_balance.py",
                     "agents/training/instrumented_ppo/calibration.py", "agents/training/scaffolding.py",
                     "agents/training/instrumented_ppo/signal_metrics.py",
                     "agents/training/instrumented_ppo/metrics_export.py",
                     "agents/training/instrumented_ppo/noise_scale",
                     "agents/training/instrumented_ppo/rollout_probes.py",
                     "stable_baselines3/common/utils.py",
                     "agents/training/instrumented_ppo/train_setup.py")),
    ("optimizer", ("torch/optim/",)),
    ("rollout buffer", ("buffers.py",)),
)


def area_of(path: Optional[str]) -> str:
    for name, keys in AREAS:
        if any(k in (path or "") for k in keys):
            return name
    return "other"


def area_table(view: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}

    def row(a: str) -> Dict[str, int]:
        return out.setdefault(a, {"graphs": 0, "ops": 0, "break_sites": 0, "break_events": 0,
                                  "skipped_frames": 0, "fallbacks": 0, "recompiles": 0})
    for g in view.get("graphs", []):
        r = row(area_of(str(g.get("root") or "").split(":")[0]))
        r["graphs"] += 1
        r["ops"] += int(g.get("n_ops", 0))
    for x in view.get("sites", []):
        r = row(area_of(str(x.get("site") or "").split(":")[0]))
        if x["kind"] == "graph_break":
            r["break_sites"] += 1
            r["break_events"] += int(x["events"])
        elif x["kind"] == "skip_frame":
            r["skipped_frames"] += int(x["events"])
        else:
            r["fallbacks"] += int(x["events"])
    for rc in view.get("recompiles", []):
        row(area_of(rc.get("site")))["recompiles"] += 1
    return out


def render_areas(results: Sequence[Tuple[str, Dict[str, Any]]]) -> str:
    """One table: per AREA, per (torch, view) — graphs / break sites / recompiles."""
    cols = []
    for label, res in results:
        for v in VIEWS:
            if res.get(v):
                cols.append((f"{label} {v}", area_table(res[v])))
    names = [a for a, _ in AREAS] + ["other"]
    head = ["area"] + [c for c, _ in cols]
    rows = []
    for a in names:
        cells = []
        for _, t in cols:
            d = t.get(a)
            cells.append("—" if not d else
                         f"{d['graphs']} g / {d['ops']} ops / {d['break_sites']} brk "
                         f"({d['break_events']} ev) / {d['skipped_frames']} skip / "
                         f"{d['fallbacks']} fb / {d['recompiles']} rc")
        if any(c != "—" for c in cells):
            rows.append([a] + cells)
    return _table(head, rows)


def render_trace(res: Dict[str, Any], src: Path) -> str:
    parts = [f"### Trace — torch {res.get('geometry', {}).get('torch')} "
             f"({res.get('geometry', {}).get('device')})", _provenance(res, src)]
    rss = res.get("rss_peak_mb_by_stage") or {}
    if rss:
        parts.append("- peak RSS by stage (MB; the 12 GB-capped scope): " + ", ".join(
            f"{k} {v}" for k, v in rss.items()) + f"; classify (separate process) "
            f"{res.get('analyze_peak_rss_mb')}")
    dbg = res.get("debugger", {})
    parts.append(f"- ObservationDebugger: api present {dbg.get('debugger_api_present')}, "
                 f"attached by the run {dbg.get('was_attached')} (dropped for both views, as the "
                 f"compile path drops it); micro-check: {json.dumps(res.get('debugger_microcheck'))}")
    rows = []
    for v in VIEWS:
        t = (res.get(v) or {}).get("totals")
        if not t:
            continue
        rows.append([v, t["graphs"], t["graphs_ending_in_break"], t["break_sites"],
                     t["break_events"], t["break_events_in_skipped_frames"], t["skipped_frames"],
                     t["fallback_frames"], t["recompiles"], t["frames_traced"], t["graph_ops"]])
    parts.append("\n**Totals per view** (all four calls):\n\n" + _table(
        ["view", "graphs", "graphs ending in a break", "break sites", "break events",
         "…in skipped frames", "skipped frames", "eager fallbacks", "recompiles", "frames traced",
         "ops in graphs"], rows))
    for v in VIEWS:
        vr = res.get(v)
        if not vr:
            continue
        per = vr.get("per_label", {})
        prow = [[lab, per.get(lab, {}).get("graphs", 0), per.get(lab, {}).get("graph_break", 0),
                 per.get(lab, {}).get("skip_frame", 0), per.get(lab, {}).get("recompiles", 0),
                 per.get(lab, {}).get("convert_failed", 0) + per.get(lab, {}).get(
                     "recompile_limit", 0)] for lab in LABELS]
        parts.append(f"\n**{v}: per call** (a NEW graph/break is counted at the call that first "
                     f"traced it; a cached call adds nothing):\n\n" + _table(
                         ["call", "new graphs", "break events", "skipped frames", "recompiles",
                          "eager fallbacks"], prow))
        comp = vr.get("by_component", {})
        crow = [[c, d.get("graph_ops", 0), d.get("graph_break_sites", 0),
                 d.get("skip_frame_sites", 0),
                 d.get("convert_failed_sites", 0) + d.get("recompile_limit_sites", 0)]
                for c, d in comp.items()]
        parts.append(f"\n**{v}: by component**:\n\n" + _table(
            ["component", "ops in graphs", "break sites", "skipped-frame sites",
             "eager-fallback sites"], crow))
        sites = sorted(vr.get("sites", []), key=lambda s: (-s["events"], str(s["site"])))
        files: Dict[str, Dict[str, Any]] = {}
        for x in sites:
            f = str(x["site"]).split(":")[0]
            d = files.setdefault(f, {"sites": 0, "events": 0, "kinds": set(), "top": x["reason"]})
            d["sites"] += 1
            d["events"] += x["events"]
            d["kinds"].add(x["kind"])
        frow = [[f, d["sites"], d["events"], ",".join(sorted(d["kinds"])), d["top"][:110]]
                for f, d in sorted(files.items(), key=lambda kv: -kv[1]["events"])]
        parts.append(f"\n**{v}: by source file** (every site; top reason by events):\n\n" + _table(
            ["file", "sites", "events", "kinds", "top reason"], frow))
        cap_n = 40
        srow = [[x["component"], x["kind"], x["site"], x["reason"][:150], x["events"],
                 ",".join(y.split(":")[0] for y in x["labels"]), x.get("detail") or ""]
                for x in sites[:cap_n]]
        more = f" (top {cap_n} of {len(sites)} by events; the JSON has all)" if len(sites) > cap_n \
            else ""
        parts.append(f"\n**{v}: sites**{more} (dynamo's reason, verbatim first line):\n\n" + _table(
            ["component", "kind", "site", "reason", "events", "calls", "detail"], srow))
        rc: Dict[Any, int] = {}
        for r in vr.get("recompiles", []):
            k = (r["label"], r["function"], r["site"], r["guard"][:140])
            rc[k] = rc.get(k, 0) + 1
        if rc:
            rc = dict(sorted(rc.items(), key=lambda kv: -kv[1])[:30])
            parts.append(f"\n**{v}: recompiles** (first failing guard):\n\n" + _table(
                ["call", "function", "site", "guard", "n"], [list(k) + [n] for k, n in rc.items()]))
        u3c = (vr.get("calls") or {}).get("update3", {})
        u3 = u3c.get("classified")
        if u3:
            t = u3["totals"]
            how = ("with Python stacks" if u3c.get("with_stack", True) else
                   "no stacks — eager ops by phase only")
            parts.append(f"\n**{v}: steady-state update, host ATen ops compiled vs eager** "
                         f"(top-level ops; profiled {how}): compiled "
                         f"{int(t['compiled_host_ops'])}, eager {int(t['eager_host_ops'])} — "
                         f"compiled share {_pct(t['compiled_host_op_share'])}\n")
            prow2 = [[p, int(d.get("compiled_ops", 0)), int(d.get("eager_ops", 0)),
                      _f(d.get("compiled_host_ms", 0.0)), _f(d.get("eager_host_ms", 0.0))]
                     for p, d in u3["phases"].items()]
            parts.append(_table(["phase", "compiled ops", "eager ops", "compiled host ms",
                                 "eager host ms"], prow2))
            erow = [[c, int(d.get("eager_ops", 0)), _f(d.get("eager_host_ms", 0.0))]
                    for c, d in sorted(u3["eager_by_component"].items(),
                                       key=lambda kv: -kv[1].get("eager_ops", 0))]
            parts.append("\nEager ops by component:\n\n" + _table(
                ["component", "eager ops", "eager host ms"], erow))
    scan = res.get("fold_static_scan") or {}
    if scan:
        parts.append(f"\n**Static scan of `train()`'s minibatch loop** (AST, not dynamo; "
                     f"`{scan.get('source')}` from line {scan.get('loop_line')}): "
                     + ", ".join(f"{k} ×{n}" for k, n in sorted(scan.get("counts", {}).items())))
    return "\n".join(parts)


def render_time(res: Dict[str, Any], src: Path) -> str:
    g = res.get("geometry") or {}
    parts = [f"### Time — torch {res.get('torch')} (CUDA), matmul `{res.get('matmul_precision')}`",
             f"- source: `{src}`",
             f"- geometry: {g.get('rows')} rows, micro {g.get('micro_batch')} × "
             f"{g.get('micro_per_epoch')}/epoch, accum {g.get('grad_accum_steps')}, epochs "
             f"{g.get('n_epochs')}"]
    b = res.get("bracketed") or {}
    ub = res.get("unbracketed") or {}
    parts.append(f"- full update, `diag_skipped`: unbracketed train_ms {_f(ub.get('train_ms'), 0)}, "
                 f"bracketed train_ms {_f(b.get('train_ms'), 0)}")
    ph = b.get("phases_s") or {}
    if ph:
        tot = sum(ph.values()) or 1.0
        parts.append("\nBracketed full update, per phase (synchronised segment wall):\n\n" + _table(
            ["phase", "s", "share", "marks"],
            [[k, _f(v, 2), _pct(v / tot), (b.get("phase_counts") or {}).get(k)]
             for k, v in sorted(ph.items(), key=lambda kv: -kv[1])]))
    for p in res.get("profiles", []):
        t = p["totals"]
        parts.append(f"\n**Profiled `{p.get('config')}` update, {p.get('epochs')} epoch(s)** — "
                     f"train span {_f(t['train_span_ms'], 0)} ms, GPU busy "
                     f"{_f(t['gpu_busy_ms'], 0)} ms, idle {_f(t['gpu_idle_ms'], 0)} ms; kernel time "
                     f"compiled {_f(t['compiled_kernel_ms'], 0)} ms vs eager "
                     f"{_f(t['eager_kernel_ms'], 0)} ms → **compiled share of kernel time "
                     f"{_pct(t['compiled_kernel_share'])}**, of train wall "
                     f"{_pct(t['compiled_share_of_train_wall'])}; unmatched device events "
                     f"{t['unmatched_device_events']}/{t['device_events']}\n")
        rows = []
        for name, d in p["phases"].items():
            ck, ek = d.get("compiled_kernel_ms", 0.0), d.get("eager_kernel_ms", 0.0)
            rows.append([name, _f(d.get("host_wall_ms", 0.0), 0), _f(ck, 0), _f(ek, 0),
                         _f(d.get("compiled_memop_ms", 0.0) + d.get("eager_memop_ms", 0.0), 0),
                         _pct(ck / (ck + ek)) if (ck + ek) else "—",
                         int(d.get("compiled_kernels", 0)), int(d.get("eager_kernels", 0))])
        rows.sort(key=lambda r: -float(r[1]) if r[1] != "—" else 0)
        parts.append(_table(["phase", "host wall ms", "compiled kernel ms", "eager kernel ms",
                             "memcpy/set ms", "compiled share", "compiled kernels",
                             "eager kernels"], rows))
    return "\n".join(parts)


def render(dirs: Sequence[Path]) -> str:
    out: List[str] = []
    traces = []
    for d in dirs:
        tr = d / "trace_result.json"
        if tr.exists():
            r = json.loads(tr.read_text())
            traces.append((str((r.get("geometry") or {}).get("torch", "?")).split("+")[0], r))
    if traces:
        out.append("### By area, every trace given (graphs / ops in graphs / break sites (events) / "
                   "skipped frames / eager fallbacks / recompiles)\n\n" + render_areas(traces))
    for d in dirs:
        tr, ta = d / "trace_result.json", d / "time_analysis.json"
        if tr.exists():
            out.append(render_trace(json.loads(tr.read_text()), tr))
        elif ta.exists():
            out.append(render_time(json.loads(ta.read_text()), ta))
        else:
            out.append(f"### {d}\n\nno result file (trace_result.json / time_analysis.json)")
    return "\n\n".join(out) + "\n"
