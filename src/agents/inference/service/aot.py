"""The ``aot`` backend: one AOTInductor package per slot group x bucket, WEIGHTS AS INPUTS.

WHY WEIGHTS AS INPUTS. A package built from the module embeds its weights as constants, and
Inductor also FOLDS weight-only subgraphs into further baked constants that ``load_constants``
does not reach: swapping a perturbed policy's weights into a real checkpoint's package served
max|dlogp| 0.076 / |dV| 0.026 against that policy (torch 2.8.0+cu126, measured 2026-09-29), while
swapping the real weights back restored it exactly; ``use_runtime_constant_folding`` failed to load
(CUDA "invalid argument" at container creation). Lifting every parameter and persistent buffer to
a graph INPUT removes both problems: nothing weight-derived can be baked, and ONE package per
bucket serves every slot of the group (the slot's stacked-storage views are passed per call). Only
the non-persistent construction tables stay constants (56 of them, ~15 MB) — identical for every
slot of an architecture by construction.

WHEN. torch >= 2.8 only. On 2.5.1 the package is one graph, so the ``6521f420`` trunk split cannot
exist in it, and the unsplit graph MISCOMPILES there (max|dlogp| 0.68 on the real checkpoint). The
C++ build needs the CUDA 12.6 headers (CCCL's ``<nv/target>``, ``crt/host_defines.h``), which live
in the ``gen3ai_torch28`` env only (``environment_torch28.yml``); ``cuda_home()`` finds them there.

WHAT IT BUYS. Not speed: an AOT call is not capturable as a CUDA graph (the model container
synchronises on its own run-finished event — "operation not permitted when stream is capturing")
and runs launch-bound. It buys a PYTHON-FREE artifact: the package loads and runs from C++
through libtorch's ``AOTIModelPackageLoader`` with no libpython (measured: equal to eager at
6e-6 / 1.8e-7), which is what a Rust caller of T2 would load.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.func import functional_call

from agents.inference.service.spec import ServiceError

#: torch versions the ``aot`` backend is allowed on (the unsplit graph measured correct).
AOT_MIN_TORCH = (2, 8)


def torch_version() -> Tuple[int, int]:
    major, minor = torch.__version__.split("+")[0].split(".")[:2]
    return int(major), int(minor)


def cuda_home() -> str:
    """``$CUDA_HOME`` if set, else the running env's conda CUDA target dir; refuses (typed) when
    the headers AOTInductor's C++ build includes are absent."""
    home = os.environ.get("CUDA_HOME") or str(Path(sys.prefix) / "targets" / "x86_64-linux")
    missing = [h for h in ("cuda_runtime_api.h", "crt/host_defines.h", "nv/target")
               if not (Path(home) / "include" / h).exists()]
    if missing:
        raise ServiceError(
            f"backend 'aot' needs the CUDA 12.6 headers under {home}/include (missing {missing}). "
            "They are installed in the gen3ai_torch28 env only (environment_torch28.yml: "
            "cuda-cccl, cuda-cudart-dev, cuda-crt-dev_linux-64); run under that interpreter.")
    return home


def check_available() -> None:
    if torch_version() < AOT_MIN_TORCH:
        raise ServiceError(
            f"backend 'aot' needs torch >= {'.'.join(map(str, AOT_MIN_TORCH))} (running "
            f"{torch.__version__}): an AOT package is ONE graph, so the 2.5.1 trunk split cannot "
            "exist in it, and the unsplit graph miscompiles on 2.5.1 (max|dlogp| 0.68, measured "
            "2026-09-29). Use backend 'graph'.")
    cuda_home()


def weight_names(module: torch.nn.Module) -> List[str]:
    """Every parameter and PERSISTENT buffer of the decision module — the package's weight inputs."""
    persistent = set(module.state_dict().keys())
    return ([n for n, _ in module.named_parameters()]
            + [n for n, _ in module.named_buffers() if n in persistent])


def weight_tensors(module: torch.nn.Module, names: List[str]) -> List[torch.Tensor]:
    table: Dict[str, torch.Tensor] = dict(module.named_parameters())
    table.update(dict(module.named_buffers()))
    return [table[n] for n in names]


class _WeightsAsInputs(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, names: List[str]):
        super().__init__()
        self.inner = module
        self.names = names

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, *w: torch.Tensor) -> Any:
        logp, value = functional_call(self.inner, dict(zip(self.names, w)), (obs, mask))
        return logp, value, logp.argmax(-1)


def build_package(module: torch.nn.Module, static_obs: torch.Tensor, static_mask: torch.Tensor,
                  path: Path) -> Tuple[Any, List[str]]:
    """Export + compile + load one bucket's package. Returns ``(runner, weight names)``;
    ``runner(obs, mask, *weights) -> (logp, value, greedy)``."""
    os.environ.setdefault("CUDA_HOME", cuda_home())
    names = weight_names(module)
    wrapped = _WeightsAsInputs(module, names).eval()
    with torch.no_grad():
        ep = torch.export.export(wrapped, (static_obs, static_mask,
                                           *weight_tensors(module, names)), strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    inductor: Any = torch._inductor               # 2.8 API; 2.5.1 is refused before this
    # torch 2.8's AOTI codegen writes `metadata["AOTI_DEVICE_KEY"] = device_type` INTO the
    # process-global `torch._inductor.config.aot_inductor.metadata` dict (codecache.py, in place) and
    # never restores it — a leaked global the torch-state guard (rightly) fails. Give this compile its
    # OWN dict via `config.patch`: the mutation lands on the patched object, and the process's dict
    # is handed back untouched. (Slow-tier row RED since the T2 AOT backend landed; K3, 2026-09-30.)
    import torch._inductor.config as inductor_config
    with inductor_config.patch({"aot_inductor.metadata": {}}):
        pkg = inductor.aoti_compile_and_package(ep, package_path=str(path))
    runner = inductor.aoti_load_package(pkg)
    return runner, names


def artifact_dir(explicit: Optional[str]) -> Path:
    """Where packages are written: the declared dir, else a fresh dir INSIDE the compile-cache root in
    force (the run's `compile_cache/t2_aot/`, or the process's private one — K3), never shared and
    never a stray `/tmp` dir that outlives its process."""
    if explicit:
        return Path(explicit)
    from agents.model.compile_cache import t2_aot_dir

    return Path(t2_aot_dir())
