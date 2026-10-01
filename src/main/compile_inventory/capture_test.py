"""The capture engine must report a KNOWN model's graphs, breaks, skipped frames and recompiles
EXACTLY (M5 Lane K8). Each count is pinned to a construct in `_Tiny`, so a regression in any one
source the engine reads (the backend hook, the `graph_breaks` artifact, the INFO frame log, the
`recompiles` artifact, the de-duplication override) changes a number here.

Runs on both torches (2.5.1 and 2.8 report identically, measured 2026-09-30). ~2 s.
"""
from __future__ import annotations

import json
import logging

import pytest
import torch

from main.compile_inventory.capture import GRAPH_MARK, InventoryCapture


def _side(y: torch.Tensor) -> torch.Tensor:
    print("", end="")               # break 1: an untraceable builtin, inside an inlined helper
    return y


def _loopy(z: torch.Tensor) -> torch.Tensor:
    for _ in range(2):              # a break INSIDE a loop: dynamo abandons the whole frame
        if z.sum().item() > 1e9:
            z = z + 1
        z = z * 2
    return z


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(4, 4)
        self.b = torch.nn.Linear(4, 4)
        self.c = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _side(self.a(x))
        z = self.b(y)
        torch._dynamo.graph_break()  # break 2: a declared split
        return self.c(z)


@pytest.fixture
def cap_run():
    torch._dynamo.reset()
    torch.manual_seed(0)
    m = _Tiny()
    with InventoryCapture() as cap:
        opt = torch.compile(m, backend=cap.backend)
        with cap.label("train"):
            opt(torch.randn(2, 4)).sum().backward()
            opt(torch.randn(2, 4)).sum().backward()      # cached: no new graph, no new break
        m.eval()
        with cap.label("eval"), torch.no_grad():
            opt(torch.randn(2, 4))
        with cap.label("loopy"):
            torch.compile(_loopy, backend=cap.backend)(torch.randn(3))
    torch._dynamo.reset()
    return cap.summary()


def test_graphs_and_breaks_of_the_known_model_are_exact(cap_run):
    per = cap_run["per_label"]
    # train: [a] | print | [b] | graph_break() | [c]  => 3 graphs, 2 of them end in a break
    assert per["train"]["graphs"] == 3
    train_graphs = [g for g in cap_run["graphs"] if g["label"] == "train"]
    assert sum(g["ends_in_break"] for g in train_graphs) == 2
    # the second train call is served from the cache
    assert all(g["calls_fw"] == 2 and g["calls_bw"] == 2 for g in train_graphs)
    # AOTAutograd traced a forward AND a backward for every training graph
    assert all(g["aot_fw_nodes"] and g["aot_bw_nodes"] for g in train_graphs)


def test_break_sites_name_the_construct(cap_run):
    breaks = [s for s in cap_run["sites"] if s["kind"] == "graph_break"]
    fns = sorted(s["site"].rsplit(" ", 1)[-1] for s in breaks)
    assert len(breaks) == 3, breaks                       # print, graph_break(), .item()
    assert "_side" in fns and "_loopy" in fns
    side = next(s for s in breaks if s["site"].endswith(" _side"))
    assert side["site"].split(":")[0].endswith("compile_inventory/capture_test.py")
    assert side["reason"]                                  # dynamo's own words, never empty


def test_a_break_in_a_loop_is_a_skipped_frame(cap_run):
    t = cap_run["totals"]
    assert t["skipped_frames"] == 1
    assert t["break_events_in_skipped_frames"] == 1
    item = next(s for s in cap_run["sites"] if s["kind"] == "graph_break"
                and s["site"].endswith(" _loopy"))
    assert item["in_skipped_frame"] is True
    assert cap_run["per_label"]["loopy"].get("graphs", 0) == 0


def test_every_rehit_break_is_counted_not_deduplicated(cap_run):
    # eval re-traces all three frames (grad_mode guard) and re-hits both breaks; dynamo's own
    # duplicate checker would log each site ONCE per process — the engine disables it.
    side = next(s for s in cap_run["sites"] if s["kind"] == "graph_break"
                and s["site"].endswith(" _side"))
    assert side["labels"] == ["train", "eval"]
    assert cap_run["per_label"]["eval"]["graph_break"] == cap_run["per_label"]["train"][
        "graph_break"]


def test_recompiles_name_the_failing_guard(cap_run):
    rec = cap_run["recompiles"]
    assert rec and all(r["label"] == "eval" for r in rec)
    assert len(rec) == 4                                   # forward, _side, two resume frames
    assert all("grad_mode" in r["guard"] for r in rec)


def test_logging_is_restored():
    before = {n: (lg.level, list(lg.handlers)) for n, lg in logging.root.manager.loggerDict.items()
              if isinstance(lg, logging.Logger) and n.startswith("torch._dynamo")
              and not n.endswith("__graph_breaks")}
    with InventoryCapture():
        pass
    lg = logging.getLogger("torch._dynamo")
    assert all(type(h).__name__ != "_Handler" for h in lg.handlers)
    assert lg.level == before.get("torch._dynamo", (lg.level,))[0]


def test_graph_markers_enclose_every_compiled_execution(tmp_path):
    torch._dynamo.reset()
    m = _Tiny()
    with InventoryCapture() as cap:
        opt = torch.compile(m, backend=cap.backend)
        opt(torch.randn(2, 4)).sum().backward()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            opt(torch.randn(2, 4)).sum().backward()
    torch._dynamo.reset()
    raw = tmp_path / "t.json"
    prof.export_chrome_trace(str(raw))
    ev = json.loads(raw.read_text())["traceEvents"]
    marks = [e["name"] for e in ev if str(e.get("name", "")).startswith(GRAPH_MARK)]
    assert sorted(marks) == sorted([f"{GRAPH_MARK}{i}:{s}" for i in range(3) for s in ("fw", "bw")])


def _loopy_caller(z: torch.Tensor) -> torch.Tensor:
    for _ in range(2):              # inlining `_loopy` hits its break inside THIS loop too
        z = _loopy(z)
    return z


def test_a_rehit_break_in_skipped_frames_is_counted_twice():
    # No graph ends at either hit, so only the `graph_breaks` log can see them — and dynamo's
    # duplicate checker would log the second hit of the same site at a suppressed level.
    torch._dynamo.reset()
    with InventoryCapture() as cap:
        with cap.label("nested"):
            torch.compile(_loopy_caller, backend=cap.backend)(torch.randn(3))
    torch._dynamo.reset()
    s = cap.summary()
    site = next(x for x in s["sites"] if x["kind"] == "graph_break"
                and x["site"].endswith(" _loopy"))
    assert site["events"] == 2, site
    assert s["totals"]["skipped_frames"] == 2 and s["totals"]["graphs"] == 0
