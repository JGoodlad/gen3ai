"""`trace_classify` on SYNTHETIC chrome-trace events whose answer is known by construction: which
host ops and which device kernels are compiled, which phase each belongs to, and which component
an eager op is charged to. A regression in the enclosure sweep, the correlation join or the phase
lookup changes a number here."""
from __future__ import annotations

from main.compile_inventory.trace_classify import BACKWARD_EAGER, classify_events

MAIN, AUTOGRAD = 1, 2


def _x(name, ts, dur, cat="cpu_op", tid=MAIN, **args):
    e = {"ph": "X", "name": name, "ts": ts, "dur": dur, "cat": cat, "pid": 0, "tid": tid}
    if args:
        e["args"] = args
    return e


def _events():
    ev = [
        # phase segments on the main thread: seg#0 = "forward" (0-100), seg#1 = "backward" (100-200)
        _x("k8.seg#0", 0, 100, cat="user_annotation"),
        _x("k8.seg#1", 100, 100, cat="user_annotation"),
        # the device-timeline COPY of a range (CUDA traces carry one): never a phase boundary
        _x("k8.seg#0", 150, 5000, cat="gpu_user_annotation", tid=7),
        # forward: a compiled graph with one aten op that launches kernel c1 ...
        _x("CompiledFunction", 10, 30),
        _x("aten::mm", 12, 10),
        _x("cudaLaunchKernel", 13, 1, cat="cuda_runtime", correlation=1),
        _x("aten::t", 14, 2),                                     # nested: not top-level
        # ... and an eager op from the policy's heads, under Python frames
        _x("agents/model/policy.py(303): evaluate_actions", 50, 40, cat="python_function"),
        _x("aten::masked_fill", 55, 10),
        _x("cudaLaunchKernel", 56, 1, cat="cuda_runtime", correlation=2),
        # backward on the autograd thread: one compiled node, one eager node
        _x("autograd::engine::evaluate_function: CompiledFunctionBackward", 110, 30, tid=AUTOGRAD),
        _x("CompiledFunctionBackward", 111, 28, tid=AUTOGRAD),
        _x("triton_poi_fused_0", 112, 5, tid=AUTOGRAD),
        _x("cuLaunchKernel", 113, 1, cat="cuda_driver", tid=AUTOGRAD, correlation=3),
        _x("autograd::engine::evaluate_function: MaskedFillBackward0", 150, 20, tid=AUTOGRAD),
        _x("aten::where", 152, 8, tid=AUTOGRAD),
        _x("cudaLaunchKernel", 153, 1, cat="cuda_runtime", tid=AUTOGRAD, correlation=4),
        # device
        _x("ampere_sgemm", 20, 4000, cat="kernel", tid=7, correlation=1),
        _x("masked_fill_kernel", 60, 1000, cat="kernel", tid=7, correlation=2),
        _x("triton_poi_fused_0", 120, 2000, cat="kernel", tid=7, correlation=3),
        _x("where_kernel", 160, 500, cat="kernel", tid=7, correlation=4),
    ]
    return ev


def test_kernels_are_split_by_their_launch_site_and_phase():
    out = classify_events(_events(), segment_names=["forward", "backward"])
    ph = out["phases"]
    assert ph["forward"]["compiled_kernel_ms"] == 4.0
    assert ph["forward"]["eager_kernel_ms"] == 1.0
    assert ph["backward"]["compiled_kernel_ms"] == 2.0      # autograd thread, by launch time
    assert ph["backward"]["eager_kernel_ms"] == 0.5
    t = out["totals"]
    assert abs(t["compiled_kernel_share"] - 6.0 / 7.5) < 1e-12
    assert t["unmatched_device_events"] == 0


def test_host_ops_top_level_only_and_eager_ones_attributed():
    out = classify_events(_events(), segment_names=["forward", "backward"])
    ph = out["phases"]
    assert ph["forward"]["compiled_ops"] == 1               # aten::mm (aten::t is nested)
    assert ph["forward"]["eager_ops"] == 1                  # aten::masked_fill
    comp = out["eager_by_component"]
    assert comp["heads"]["eager_ops"] == 1
    assert comp[BACKWARD_EAGER]["eager_ops"] == 1
    assert dict(out["backward_eager_nodes_top"]) == {"MaskedFillBackward0": 1}


def test_segment_wall_is_booked_to_its_closing_mark():
    out = classify_events(_events(), segment_names=["forward", "backward"])
    assert out["phases"]["forward"]["host_wall_ms"] == 0.1
    assert out["phases"]["backward"]["host_wall_ms"] == 0.1


def test_the_streaming_loader_reads_a_real_export_like_json_load(tmp_path):
    import gzip
    import json

    import torch

    from main.compile_inventory.trace_classify import _slim, load_trace
    x = torch.randn(8, 8)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with torch.profiler.record_function("k8.seg#0"):
            (x @ x).relu().sum()
    raw = tmp_path / "t.json"
    prof.export_chrome_trace(str(raw))
    gz = tmp_path / "t.json.gz"
    gz.write_bytes(gzip.compress(raw.read_bytes()))
    want = [_slim(e) for e in json.loads(raw.read_text())["traceEvents"]]
    got = load_trace(gz)
    assert got == want and any(e.get("name") == "aten::mm" for e in got)
