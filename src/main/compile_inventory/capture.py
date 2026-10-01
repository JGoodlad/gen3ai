"""THE CAPTURE ENGINE — every graph, every break, every skipped frame and every recompile dynamo
produces while a block runs, with the stack that caused it (M5 Lane K8, measurement first).

    with InventoryCapture(backend="aot_eager") as cap:
        step = torch.compile(fn, backend=cap.backend)
        with cap.label("update1"):
            step(...)
    cap.summary()   # -> a JSON-ready dict

WHY NOT `torch._dynamo.explain`. `explain` sees a break only through the graph it ENDS, so it
misses (a) a frame dynamo abandons whole — "Skipping frame because there is a graph break in a
for/while loop", which is exactly how `train()` is lost — (b) a break at a frame's first
instruction, (c) recompiles and their guard failures, and (d) the cache-limit fallback. This
engine reads the same facts from dynamo's own log artifacts (`graph_breaks`, `recompiles`, the
INFO frame log) AND from every graph's `compile_subgraph_reason`, and keeps both: a break reported
by either source is a break.

THE FOUR SOURCES, and what each one is authoritative for:

  * the BACKEND hook  — one `GraphRecord` per FX graph: op count, ops per component (from every
    node's `stack_trace`), the reason the graph ended, and the AOT forward/backward node counts;
    every graph's execution is wrapped in `record_function("k8.graph#<id>:fw|bw")`, so a profiler
    trace can tell compiled work from eager work EXACTLY (`trace_classify`).
  * `graph_breaks`    — one `BreakEvent` per break, with dynamo's reason and the user stack. The
    duplicate-suppression checker is disabled for the block, so a break re-hit by a recompile or
    reached from a second caller is counted again (sites are also reported de-duplicated).
  * the INFO log      — frames started, frames SKIPPED (a break inside a loop), and (WARNING) the
    frames that failed to convert or hit the recompile limit and now run eager.
  * `recompiles`      — one `RecompileEvent` per recompile with the first failing guard.

Scope: this module only OBSERVES. It never changes a production module; it sets dynamo config only
inside `torch._dynamo.config.patch` and restores the logging configuration on exit.
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import torch

from main.compile_inventory.attribution import (Frame, classify, innermost_user, loss_detail,
                                                parse_stack, short_path, user_frames)

#: The `record_function` prefix around every captured graph's execution.
GRAPH_MARK = "k8.graph#"

_CALL_OPS = ("call_function", "call_method", "call_module")


@dataclass
class GraphRecord:
    gid: int
    label: str
    compile_id: str
    root: Optional[str]
    n_ops: int
    ops_by_component: Dict[str, int]
    end_reason: str
    ends_in_break: bool
    break_site: Optional[str]
    aot_fw_nodes: Optional[int] = None
    aot_bw_nodes: Optional[int] = None
    calls_fw: int = 0
    calls_bw: int = 0


@dataclass
class BreakEvent:
    kind: str                  # graph_break | skip_frame | convert_failed | recompile_limit
    reason: str
    site: Optional[str]        # innermost user frame, "path:line fn"
    component: str
    label: str
    detail: Optional[str] = None
    stack: List[str] = field(default_factory=list)
    in_skipped_frame: bool = False
    source: str = "log"        # log | graph_reason


@dataclass
class RecompileEvent:
    label: str
    function: str
    site: Optional[str]
    guard: str
    text: str


class _AlwaysNew:
    """Stands in for dynamo's duplicate-break checker: every break is logged, every time."""

    def add(self, _key: Any) -> bool:
        return True

    def reset(self) -> None:
        pass


_DUP_ATTR = "graph_break_dup_warning_checker"


def _reason_first_line(text: str) -> str:
    m = re.search(r"Graph Break Reason: (.*)", text)
    if m:
        return m.group(1).strip()
    return (text or "").strip().splitlines()[0] if (text or "").strip() else ""


class _Handler(logging.Handler):
    def __init__(self, cap: "InventoryCapture") -> None:
        super().__init__(level=logging.DEBUG)
        self.cap = cap

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 - logging API
        try:
            self.cap._on_record(record)
        except Exception as e:  # noqa: BLE001 - a parse failure must not break the traced run
            self.cap.parse_errors.append(f"{record.name}: {e!r}")


class _Mute(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return False


class InventoryCapture:
    """See the module docstring. ``backend``: ``"aot_eager"`` (default — the forward AND the
    backward are traced by AOTAutograd, so a graph that cannot be differentiated fails HERE, and
    both are marked), ``"eager"`` (dynamo only; the backward runs as eager autograd), or
    ``"inductor"`` (the real compiler; forward marked, backward via `CompiledFunctionBackward`)."""

    def __init__(self, backend: str = "aot_eager", *, quiet: bool = True,
                 suppress_errors: bool = True) -> None:
        if backend not in ("aot_eager", "eager", "inductor"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend_name = backend
        self.quiet = quiet
        self.suppress_errors = suppress_errors
        self.graphs: List[GraphRecord] = []
        self.breaks: List[BreakEvent] = []
        self.recompiles: List[RecompileEvent] = []
        self.frames_started: Counter = Counter()
        self.parse_errors: List[str] = []
        self._label = "-"
        self._stack: Optional[contextlib.ExitStack] = None
        self._handler = _Handler(self)
        self._mutes: List[logging.Handler] = []
        self._lock = threading.Lock()
        self.counters: Dict[str, Dict[str, int]] = {}

    # -------------------------------------------------------------------------------- lifecycle
    def __enter__(self) -> "InventoryCapture":
        import torch._dynamo
        import torch._logging
        from torch._dynamo.utils import counters
        st = contextlib.ExitStack()
        self._stack = st
        counters.clear()
        st.enter_context(torch._dynamo.config.patch(suppress_errors=self.suppress_errors))
        # dup checker: patch every module that holds a binding to it
        import torch._dynamo.symbolic_convert as sc
        import torch._dynamo.utils as du
        for mod in (sc, du):
            if hasattr(mod, _DUP_ATTR):
                prev = getattr(mod, _DUP_ATTR)
                setattr(mod, _DUP_ATTR, _AlwaysNew())
                st.callback(setattr, mod, _DUP_ATTR, prev)
        torch._logging.set_logs(dynamo=logging.INFO, graph_breaks=True, recompiles=True)
        st.callback(self._restore_logging)
        lg = logging.getLogger("torch._dynamo")
        lg.addHandler(self._handler)
        st.callback(lg.removeHandler, self._handler)
        if self.quiet:
            mute = _Mute()
            for name, obj in list(logging.root.manager.loggerDict.items()):
                if not name.startswith("torch") or not isinstance(obj, logging.Logger):
                    continue
                for h in obj.handlers:
                    if h is not self._handler and isinstance(h, logging.StreamHandler):
                        h.addFilter(mute)
                        st.callback(h.removeFilter, mute)
        return self

    def __exit__(self, *exc: Any) -> None:
        from torch._dynamo.utils import counters
        self.counters = {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in counters.items()}
        assert self._stack is not None
        self._stack.close()
        self._stack = None
        self._reconcile_graph_breaks()

    @staticmethod
    def _restore_logging() -> None:
        import torch._logging
        torch._logging.set_logs()
        if os.environ.get("TORCH_LOGS"):
            from torch._logging import _internal
            _internal._init_logs()

    @contextlib.contextmanager
    def label(self, name: str) -> Iterator[None]:
        prev, self._label = self._label, name
        try:
            yield
        finally:
            self._label = prev

    # -------------------------------------------------------------------------------- backend
    def backend(self, gm: torch.fx.GraphModule, example_inputs: Sequence[Any]) -> Callable[..., Any]:
        from torch._guards import CompileContext
        gid = len(self.graphs)
        comp: Counter = Counter()
        root: Optional[str] = None
        n_ops = 0
        for node in gm.graph.nodes:
            if node.op not in _CALL_OPS:
                continue
            n_ops += 1
            frames = parse_stack(node.meta.get("stack_trace") or "")
            comp[classify(frames)] += 1
            if root is None and frames:
                root = frames[0].short()
        reason_obj = getattr(gm, "compile_subgraph_reason", None)
        reason = str(getattr(reason_obj, "reason", "") or "")
        ends_in_break = bool(getattr(reason_obj, "graph_break", False))
        bsite = None
        ustack = list(getattr(reason_obj, "user_stack", None) or [])
        if ends_in_break and ustack:
            fr = [Frame(s.filename, int(s.lineno or 0), s.name) for s in ustack]
            iu = innermost_user(fr)
            bsite = iu.short() if iu else None
        try:
            cid = str(CompileContext.current_compile_id())
        except Exception:  # noqa: BLE001
            cid = "?"
        rec = GraphRecord(gid=gid, label=self._label, compile_id=cid, root=root, n_ops=n_ops,
                          ops_by_component=dict(comp), end_reason=reason.splitlines()[0][:300]
                          if reason else "", ends_in_break=ends_in_break, break_site=bsite)
        self.graphs.append(rec)
        if ends_in_break and ustack:
            fr = [Frame(s.filename, int(s.lineno or 0), s.name) for s in ustack]
            self.breaks.append(BreakEvent(
                kind="graph_break", reason=rec.end_reason, site=bsite, component=classify(fr),
                label=self._label, detail=loss_detail(fr), stack=[f.short() for f in fr],
                source="graph_reason"))
        return self._compile(gm, example_inputs, rec)

    def _marked(self, fn: Callable[..., Any], rec: GraphRecord, side: str, boxed: bool
                ) -> Callable[..., Any]:
        name = f"{GRAPH_MARK}{rec.gid}:{side}"

        def run(*args: Any) -> Any:
            if side == "fw":
                rec.calls_fw += 1
            else:
                rec.calls_bw += 1
            with torch.profiler.record_function(name):
                return fn(*args)
        if boxed:
            run._boxed_call = True  # type: ignore[attr-defined]
        return run

    def _compile(self, gm: torch.fx.GraphModule, example_inputs: Sequence[Any],
                 rec: GraphRecord) -> Callable[..., Any]:
        if self.backend_name == "eager":
            return self._marked(gm.forward, rec, "fw", boxed=False)
        if self.backend_name == "inductor":
            from torch._inductor.compile_fx import compile_fx
            return self._marked(compile_fx(gm, example_inputs), rec, "fw", boxed=False)
        from torch._dynamo.backends.common import aot_autograd

        def _fw(fx_g: torch.fx.GraphModule, _ex: Sequence[Any]) -> Callable[..., Any]:
            rec.aot_fw_nodes = sum(1 for n in fx_g.graph.nodes if n.op in _CALL_OPS)
            return self._marked(torch.fx.Interpreter(fx_g).boxed_run, rec, "fw", boxed=True)

        def _bw(fx_g: torch.fx.GraphModule, _ex: Sequence[Any]) -> Callable[..., Any]:
            rec.aot_bw_nodes = sum(1 for n in fx_g.graph.nodes if n.op in _CALL_OPS)
            return self._marked(torch.fx.Interpreter(fx_g).boxed_run, rec, "bw", boxed=True)
        return aot_autograd(fw_compiler=_fw, bw_compiler=_bw)(gm, example_inputs)

    # -------------------------------------------------------------------------------- logs
    def _on_record(self, record: logging.LogRecord) -> None:
        name = record.name
        msg = record.getMessage()
        with self._lock:
            if name.endswith("__graph_breaks"):
                self._on_graph_break(record, msg)
            elif name.endswith("__recompiles"):
                self._on_recompile(msg)
            elif "Skipping frame because there is a graph break in a for/while loop" in msg:
                self._on_skip(msg)
            elif msg.startswith("Step 1: torchdynamo start tracing"):
                m = re.match(r"Step 1: torchdynamo start tracing (\S+) (.+):(\d+)", msg)
                if m:
                    self.frames_started[f"{short_path(m.group(2))}:{m.group(3)} {m.group(1)}"] += 1
            elif record.levelno >= logging.WARNING and ("WON'T CONVERT" in msg
                                                         or "hit config." in msg):
                self._on_fallback(msg)

    def _on_graph_break(self, record: logging.LogRecord, msg: str) -> None:
        if "Graph break" not in msg:
            return
        exc = record.exc_info[1] if record.exc_info else None
        reason = ""
        if exc is not None and getattr(exc, "msg", None):
            reason = str(exc.msg).strip().splitlines()[0]
        if not reason:
            reason = _reason_first_line(msg)
        text = msg.split("User code traceback:", 1)[-1]
        frames = parse_stack(text)
        if exc is not None and not frames and getattr(exc, "real_stack", None):
            frames = [Frame(s.filename, int(s.lineno or 0), s.name) for s in exc.real_stack]
        site_f = innermost_user(frames)
        site = site_f.short() if site_f else None
        if site is None:
            m = re.search(r"in user code at (.+?):(\d+)", msg)
            if m:
                site = f"{short_path(m.group(1))}:{m.group(2)} ?"
        self.breaks.append(BreakEvent(
            kind="graph_break", reason=reason[:300], site=site,
            component=classify(frames) if frames else "other", label=self._label,
            detail=loss_detail(frames), stack=[f.short() for f in user_frames(frames)][-12:]))

    def _on_skip(self, msg: str) -> None:
        frames = parse_stack(msg)
        f0 = frames[-1] if frames else None
        site = f0.short() if f0 else None
        comp = classify(frames) if frames else "other"
        self.breaks.append(BreakEvent(kind="skip_frame", reason="graph break in a for/while loop",
                                      site=site, component=comp, label=self._label))
        # the break that caused it: the latest graph_break whose stack enters this frame
        if f0 is not None:
            key = f"{short_path(f0.file)}"
            for b in reversed(self.breaks[:-1]):
                if b.kind == "graph_break" and b.source == "log" and any(
                        s.startswith(key) and s.endswith(" " + f0.fn) for s in b.stack):
                    b.in_skipped_frame = True
                    break

    def _on_fallback(self, msg: str) -> None:
        kind = "recompile_limit" if "hit config." in msg else "convert_failed"
        frames = parse_stack(msg)
        site = frames[-1].short() if frames else None
        m = re.search(r"WON'T CONVERT (\S+) (\S+) line (\d+)", msg)
        if site is None and m:
            site = f"{short_path(m.group(2))}:{m.group(3)} {m.group(1)}"
        due = msg.split("due to:", 1)[-1].strip() if "due to:" in msg else ""
        self.breaks.append(BreakEvent(kind=kind, reason=msg.strip().splitlines()[0][:300],
                                      site=site, component=classify(frames) if frames else "other",
                                      label=self._label,
                                      detail=(due.splitlines()[0][:300] if due else None),
                                      stack=[msg[:1500]]))

    def _on_recompile(self, msg: str) -> None:
        m = re.search(r"Recompiling function (\S+) in (.+?):(\d+)", msg)
        fn = m.group(1) if m else "?"
        site = f"{short_path(m.group(2))}:{m.group(3)}" if m else None
        g = re.search(r"-\s*\d+/\d+: (.+)", msg)
        guard = g.group(1).strip() if g else msg.strip().splitlines()[-1].strip()
        self.recompiles.append(RecompileEvent(label=self._label, function=fn, site=site,
                                              guard=guard[:300], text=msg[:1500]))

    def _reconcile_graph_breaks(self) -> None:
        """A break seen by BOTH sources is one break: drop the graph_reason copy of a site the log
        already reported under the same label."""
        logged = {(b.label, b.site) for b in self.breaks if b.kind == "graph_break"
                  and b.source == "log"}
        self.breaks = [b for b in self.breaks if not (b.source == "graph_reason"
                                                     and (b.label, b.site) in logged)]

    # -------------------------------------------------------------------------------- summary
    def summary(self) -> Dict[str, Any]:
        breaks = [b for b in self.breaks if b.kind == "graph_break"]
        resumed = [b for b in breaks if not b.in_skipped_frame]
        skipped = [b for b in self.breaks if b.kind == "skip_frame"]
        fallbacks = [b for b in self.breaks if b.kind in ("convert_failed", "recompile_limit")]
        sites: Dict[str, Dict[str, Any]] = {}
        for b in self.breaks:
            key = f"{b.kind}|{b.site}"
            s = sites.setdefault(key, {"kind": b.kind, "site": b.site, "component": b.component,
                                       "reason": b.reason, "detail": b.detail, "events": 0,
                                       "labels": [], "in_skipped_frame": b.in_skipped_frame,
                                       "stack": b.stack})
            s["events"] += 1
            if b.label not in s["labels"]:
                s["labels"].append(b.label)
        by_comp: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for g in self.graphs:
            for c, n in g.ops_by_component.items():
                by_comp[c]["graph_ops"] += n
        for s in sites.values():
            by_comp[s["component"]][f"{s['kind']}_sites"] += 1
        per_label: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for g in self.graphs:
            per_label[g.label]["graphs"] += 1
        for b in self.breaks:
            per_label[b.label][b.kind] += 1
        for r in self.recompiles:
            per_label[r.label]["recompiles"] += 1
        return {
            "backend": self.backend_name,
            "torch": torch.__version__,
            "totals": {
                "graphs": len(self.graphs),
                "graphs_ending_in_break": sum(1 for g in self.graphs if g.ends_in_break),
                "break_events": len(breaks),
                "break_events_resumed": len(resumed),
                "break_events_in_skipped_frames": len(breaks) - len(resumed),
                "break_sites": len({b.site for b in breaks}),
                "skipped_frames": len(skipped),
                "skipped_frame_sites": len({b.site for b in skipped}),
                "fallback_frames": len(fallbacks),
                "recompiles": len(self.recompiles),
                "frames_traced": len(self.frames_started),
                "graph_ops": sum(g.n_ops for g in self.graphs),
            },
            "per_label": {k: dict(v) for k, v in per_label.items()},
            "by_component": {k: dict(v) for k, v in sorted(by_comp.items())},
            "sites": sorted(sites.values(), key=lambda s: (s["kind"], s["component"],
                                                           str(s["site"]))),
            "graphs": [asdict(g) for g in self.graphs],
            "recompiles": [asdict(r) for r in self.recompiles],
            "frames_started": dict(self.frames_started),
            "dynamo_counters": self.counters,
            "parse_errors": self.parse_errors,
        }
