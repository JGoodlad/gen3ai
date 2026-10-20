"""THE ONE `torch.compile` SEAM — every compile site in `src/` goes through `compile_on` (`gen3_cpu_compile_sites_v1`).

WHY. torch 2.8.0's Inductor CPU (C++) backend emits INVALID C++ for a graph with a per-row gather
(`x[arange(B), idx]`) beside a [B] reduction over a strided column once B and K are both >= the 16 fp32
lanes of AVX-512: the 2-D tiled kernel's transposed load names the gather's index `tmp<N>` before it is
declared and g++ stops (`error: 'tmp8' was not declared in this scope`). The end-state extractor carries
that idiom (the move-resolution family), so ANY CPU Inductor compile of it at B >= 16 can die in g++ —
and the failure surfaces as a "graph break" / a "compile disabled" warning, not as the codegen bug it is.
`CPU_INDUCTOR_OPTIONS` (`cpp.simdlen = 1`: scalar C++ kernels, no tiling; the same graph, only an fp32
summation order inside a kernel may move) steps around it. Kernel-level account, the measured shape
threshold and the upstream issues: `designs/training/compile_flags.md` "The CPU Inductor tile2d codegen
bug"; the smallest repro and its tripwire: `cpu_inductor_tile2d_test.py`.

THE RULE. `compile_on(fn, device=..., **torch_compile_kwargs)` is `torch.compile` with the workaround
added when — and only when — the compile reaches Inductor's CPP codegen: the target device is CPU and the
backend is Inductor (``None`` / ``"inductor"``). A CUDA compile (Triton) and a stand-in backend
(``"eager"`` / ``"aot_eager"`` / a callable) get their kwargs back UNCHANGED, byte for byte. A backend
that calls Inductor ITSELF (`compile_inventory.capture`'s ``inductor`` mode, `compile_fx`) takes
`inductor_options(device)` as its ``config_patches``. `inductor_compile_test` scans `src/` and fails a
`torch.compile` / `compile_fx` / `aoti_compile_and_package` call that bypasses this module.

`torch.compile` is read at CALL time (`torch.compile`, not a captured reference), so a test's
`monkeypatch.setattr(torch, "compile", spy)` still sees every call.

RETIREMENT. When torch stops failing the repro (`cpu_inductor_tile2d_test`'s tripwire fails), empty
`CPU_INDUCTOR_OPTIONS`; this seam stays as the one place a compile's device-specific options are decided.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

import torch

#: The CPU-only Inductor options (`gen3_cpu_inductor_scalar_v1`, 2026-10-15). See the module docstring.
CPU_INDUCTOR_OPTIONS: Dict[str, Any] = {"cpp.simdlen": 1}

DeviceLike = Union[str, torch.device]


def device_type(device: DeviceLike) -> str:
    """``"cpu"`` / ``"cuda"`` / ... of ``device`` (``torch.device("cuda")`` never touches the card)."""
    return torch.device(device).type


def reaches_cpp_codegen(device: DeviceLike, backend: Any = None) -> bool:
    """True when a compile for ``device`` through ``backend`` lowers to Inductor's C++ kernels."""
    return backend in (None, "inductor") and device_type(device) == "cpu"


def inductor_options(device: DeviceLike, backend: Any = None,
                     options: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The Inductor options a compile for ``device`` through ``backend`` runs with: ``options`` plus
    `CPU_INDUCTOR_OPTIONS` when it reaches the CPP codegen. The workaround is applied LAST — a caller's
    option cannot silently re-enable the tiled kernel (to reproduce the bug on purpose, pass
    ``cpu_workaround=False`` to `compile_on`)."""
    out = dict(options or {})
    if reaches_cpp_codegen(device, backend):
        out.update(CPU_INDUCTOR_OPTIONS)
    return out


def device_of_inputs(inputs: Iterable[Any]) -> str:
    """The device type a backend's example inputs live on: ``"cuda"`` (or another accelerator) if any
    tensor input is there, else ``"cpu"``."""
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.device.type != "cpu":
            return x.device.type
    return "cpu"


def compile_on(fn: Callable[..., Any], *, device: DeviceLike, backend: Any = None,
               options: Optional[Mapping[str, Any]] = None, cpu_workaround: bool = True,
               **kw: Any) -> Any:
    """`torch.compile(fn, backend=backend, options=..., **kw)` for a graph that runs on ``device``.

    ``backend`` / ``options`` are passed on only when set (an empty option set is NOT passed, so a CUDA
    compile's kwargs are exactly the caller's). ``cpu_workaround=False`` is the bug's own repro
    (`cpu_inductor_tile2d_test`) and nothing else."""
    opts = inductor_options(device, backend, options) if cpu_workaround else dict(options or {})
    if backend is not None:
        kw["backend"] = backend
    if opts:
        kw["options"] = opts
    return torch.compile(fn, **kw)
