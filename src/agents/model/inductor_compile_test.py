"""`gen3_cpu_compile_sites_v1` (2026-10-19): THE one compile seam (`agents.model.inductor_compile`) and its guard.

The torch 2.8.0 Inductor tile2d codegen bug (`cpu_inductor_tile2d_test.py`; `designs/training/compile_flags.md` "The
CPU Inductor tile2d codegen bug") kills ANY CPU Inductor compile of a per-row gather beside a strided [B] reduction
at B >= 16 in g++. The workaround used to be installed on R1 alone; every compile site in `src/` now goes through
`compile_on` / `inductor_options`, and this file:

  * pins the seam's decision (CPU + Inductor gets `CPU_INDUCTOR_OPTIONS`; CUDA and a stand-in backend get their
    kwargs back unchanged);
  * SCANS `src/` and fails a compile call that bypasses the seam (a new CPU compile site is caught at the gate);
  * proves the fix on the failing shapes B = 16 / 32 / 64 through each route a CPU compile can take — `compile_on`
    (every `torch.compile` site) and `compile_inventory.capture`'s own Inductor call (`compile_fx`).

The real-graph proofs (the production extractor's CPU compile at B = 16 / 32 / 64, the opponents' site) are in
`cpu_compile_sites_slow_test.py`.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import List

import pytest
import torch

from agents.model import inductor_compile as ic
from utils.paths import src_path

K = 18                                   # the failing K (>= 16 fp32 lanes); the bug's own shape table
FAILING_B = (16, 32, 64)                 # measured FAIL without the option (shapes.py, 2026-10-15)

_skip_compile = pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_COMPILE_TESTS") == "1",
                                   reason="GEN3AI_SKIP_COMPILE_TESTS=1")


# ------------------------------------------------------------------------------ the decision, as data
@pytest.mark.parametrize("device,backend,expect", [
    ("cpu", None, ic.CPU_INDUCTOR_OPTIONS),
    ("cpu", "inductor", ic.CPU_INDUCTOR_OPTIONS),
    (torch.device("cpu"), None, ic.CPU_INDUCTOR_OPTIONS),
    ("cuda", None, {}),                  # Triton never reaches the CPP codegen
    ("cuda:0", "inductor", {}),
    ("cpu", "eager", {}),                # a stand-in backend takes no Inductor options
    ("cpu", "aot_eager", {}),
    ("cpu", lambda gm, ex: gm, {}),      # a callable backend decides for itself (`capture._compile`)
])
def test_only_a_cpu_inductor_compile_gets_the_option(device, backend, expect):
    assert ic.inductor_options(device, backend) == expect


def test_the_workaround_is_applied_last_so_a_caller_option_cannot_undo_it():
    got = ic.inductor_options("cpu", None, {"cpp.simdlen": None, "coordinate_descent_tuning": True})
    assert got == {**ic.CPU_INDUCTOR_OPTIONS, "coordinate_descent_tuning": True}
    assert ic.inductor_options("cuda", None, {"coordinate_descent_tuning": True}) == {
        "coordinate_descent_tuning": True}


def test_device_of_inputs_reads_the_first_non_cpu_tensor():
    assert ic.device_of_inputs([torch.ones(1), 3, None]) == "cpu"
    assert ic.device_of_inputs([]) == "cpu"
    meta = torch.empty(1, device="meta")
    assert ic.device_of_inputs([torch.ones(1), meta]) == "meta"


@pytest.mark.parametrize("device,kw,expect", [
    ("cpu", {"dynamic": False}, {"dynamic": False, "options": ic.CPU_INDUCTOR_OPTIONS}),
    # CUDA: exactly the caller's kwargs — no `options` key at all, no `backend` key when none was named
    ("cuda", {"dynamic": False, "fullgraph": True}, {"dynamic": False, "fullgraph": True}),
    ("cuda", {"options": {"triton.cudagraphs": True}}, {"options": {"triton.cudagraphs": True}}),
    ("cpu", {"backend": "eager"}, {"backend": "eager"}),
    ("cpu", {"cpu_workaround": False}, {}),     # the bug's own repro, and nothing else
])
def test_compile_on_hands_torch_compile_exactly_these_kwargs(monkeypatch, device, kw, expect):
    seen = []
    monkeypatch.setattr(torch, "compile", lambda fn, **k: seen.append(k) or fn)   # read at CALL time
    ic.compile_on(len, device=device, **kw)
    assert seen == [expect]


# ------------------------------------------------------------------------------ the guard: no site bypasses the seam
_SEAM = "agents/model/inductor_compile.py"
#: The only files that may turn the workaround OFF (`cpu_workaround=False`): the bug's repro and this seam's own pins.
_REPRO_FILES = ("agents/model/cpu_inductor_tile2d_test.py", "agents/model/inductor_compile_test.py")
_INDUCTOR_ENTRY = ("compile_fx", "aoti_compile_and_package")   # backends / AOT calls that run Inductor themselves


def _is_torch_compile(call: ast.Call) -> bool:
    f = call.func
    return (isinstance(f, ast.Attribute) and f.attr == "compile" and isinstance(f.value, ast.Name)
            and f.value.id in ("torch", "th"))


def _is_inductor_entry(call: ast.Call) -> bool:
    f = call.func
    name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
    return name in _INDUCTOR_ENTRY


def _names_a_stand_in_backend(call: ast.Call) -> bool:
    """A test's `torch.compile(..., backend=X)` with X written out and not Inductor (`"eager"`, `"aot_eager"`, a
    capture or `aot_autograd(...)` backend): it cannot reach the CPP codegen. No `backend=`, `"inductor"`, or only
    a `**kw` splat is an Inductor compile."""
    for k in call.keywords:
        if k.arg == "backend":
            return not (isinstance(k.value, ast.Constant) and k.value.value in (None, "inductor"))
    return False


def _scan() -> tuple:
    """(offenders, production modules routed through the seam)."""
    offenders: List[str] = []
    routed: List[str] = []
    root = Path(src_path())
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel == _SEAM:
            continue
        text = p.read_text(errors="replace")
        if "compile" not in text:
            continue
        tree = ast.parse(text)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        called = {(n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None)) for n in calls}
        is_test = p.name.endswith("_test.py") or p.name == "conftest.py"
        for c in calls:
            if _is_torch_compile(c):
                if is_test and _names_a_stand_in_backend(c):
                    continue
                offenders.append(f"{rel}:{c.lineno} torch.compile (use inductor_compile.compile_on)")
            elif _is_inductor_entry(c) and "inductor_options" not in called:
                offenders.append(f"{rel}:{c.lineno} {ast.unparse(c.func)} without inductor_options")
        for c in calls:
            if any(k.arg == "cpu_workaround" for k in c.keywords) and rel not in _REPRO_FILES:
                offenders.append(f"{rel}:{c.lineno} cpu_workaround= (the bug's own repro only)")
        if not is_test and called & {"compile_on", "inductor_options"}:
            routed.append(rel)
    return offenders, routed


def test_no_compile_site_in_src_bypasses_the_seam():
    """A `torch.compile` call anywhere in `src/` outside the seam FAILS (a test may name a non-Inductor stand-in
    backend: it never reaches the CPP codegen), and a direct Inductor entry (`compile_fx`,
    `aoti_compile_and_package`) must sit in a module that reads `inductor_options`. Fails on a revert of any
    site's routing."""
    offenders, _ = _scan()
    assert not offenders, ("compile sites that bypass `agents.model.inductor_compile` — on CPU they would die "
                           "in g++ on the torch 2.8 tile2d bug at B >= 16 (designs/training/compile_flags.md):\n  "
                           + "\n  ".join(offenders))


def test_the_scan_sees_every_known_production_site():
    """The guard is not vacuous: the inventory of 2026-10-19 (`compile_flags.md` "Every CPU compile site") is
    what the scan finds routed."""
    _, routed = _scan()
    expect = {"agents/model/compile_regions.py", "agents/model/compile_opponents.py",
              "agents/inference/service/engine.py", "agents/inference/service/aot.py",
              "main/ops/diag_bf16_canary.py", "main/compile_inventory/fullgraph_check.py",
              "main/compile_inventory/worker.py", "main/compile_inventory/capture.py"}
    assert expect <= set(routed), sorted(expect - set(routed))


def test_the_scan_catches_a_bypass(tmp_path, monkeypatch):
    """Planted offenders: a bare CPU `torch.compile`, an explicit Inductor backend and a `compile_fx` without
    the seam are found; a stand-in backend in a test is not."""
    pkg = tmp_path / "src"
    pkg.mkdir()
    (pkg / "site.py").write_text("import torch\nf = torch.compile(len)\n"
                                 "from torch._inductor.compile_fx import compile_fx\ng = compile_fx(1, 2)\n")
    (pkg / "x_test.py").write_text("import torch\na = torch.compile(len, backend='eager')\n"
                                   "b = torch.compile(len, backend='inductor')\nc = torch.compile(len, **{})\n")
    monkeypatch.setattr(sys.modules[__name__], "src_path", lambda *a: str(pkg))
    offenders, _ = _scan()
    assert sorted(o.split(" ")[0] for o in offenders) == ["site.py:2", "site.py:4", "x_test.py:3", "x_test.py:4"]


# ------------------------------------------------------------------------------ the proof on the failing shapes
def _repro_at(b: int):
    """The bug's smallest repro (`cpu_inductor_tile2d_test._repro`) at batch ``b``: a row gathered at each batch
    row's own slot (`arange` built inside the graph), a [b] reduction over K of a strided column, the gather stored."""
    def repro(x: torch.Tensor, idx: torch.Tensor, q: torch.Tensor):
        g = x[torch.arange(b), idx]
        return (g * q[..., 5]).sum(-1), g
    gen = torch.Generator().manual_seed(b)
    args = (torch.rand(b, 6, K, generator=gen), torch.randint(0, 6, (b,), generator=gen),
            torch.rand(b, K, 6, generator=gen))
    return repro, args


@pytest.fixture
def one_thread(restore_torch_globals):
    torch.set_num_threads(1)
    torch._dynamo.reset()
    try:
        yield
    finally:
        torch._dynamo.reset()


def _assert_eager(fn, args, out):
    ref = fn(*args)
    assert torch.equal(out[1], ref[1]), "the gathered rows are an exact copy"
    torch.testing.assert_close(out[0], ref[0], rtol=1e-5, atol=1e-6)


@pytest.mark.slow  # three cold CPU Inductor compiles (~10-20 s each beside a live run)
@_skip_compile
@pytest.mark.parametrize("b", FAILING_B)
def test_compile_on_compiles_the_repro_at_the_failing_batch(one_thread, b):
    """Every `torch.compile` site's route: the repro at B = 16 / 32 / 64 compiles on CPU through `compile_on` with
    no vector / tile code left, and matches eager. Fails if the seam stops adding the option (g++ `tmp<N>` was
    not declared — the tripwire in `cpu_inductor_tile2d_test` pins that the bare compile still dies at B = 16)."""
    from torch._inductor.utils import run_and_get_cpp_code
    fn, args = _repro_at(b)
    out, code = run_and_get_cpp_code(ic.compile_on(fn, device="cpu", fullgraph=True, dynamic=False), *args)
    assert "transpose_mxn<" not in code and "Vectorized<float>" not in code
    _assert_eager(fn, args, out)


@pytest.mark.slow
@_skip_compile
@pytest.mark.parametrize("b", FAILING_B)
def test_the_inventory_capture_inductor_backend_compiles_the_repro(one_thread, b):
    """`compile_inventory.capture`'s ``inductor`` mode calls `compile_fx` itself (a callable backend, so
    `compile_on` adds nothing): it reads `inductor_options` from its example inputs. Fails on a revert of that."""
    from main.compile_inventory.capture import InventoryCapture
    fn, args = _repro_at(b)
    with InventoryCapture(backend="inductor", suppress_errors=False) as cap:   # a failure RAISES, never eager
        out = ic.compile_on(fn, device="cpu", backend=cap.backend, fullgraph=True, dynamic=False)(*args)
    _assert_eager(fn, args, out)
