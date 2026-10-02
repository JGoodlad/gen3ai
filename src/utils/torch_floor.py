"""THE TORCH FLOOR — HEAD's code runs on torch >= 2.8 only (deletion pass K1, 2026-10-02).

WHY A REFUSAL, not a fallback. HEAD deleted every torch-2.5.1 code path: the CUDA trunk split that
worked around 2.5.1's Inductor miscompile of the single fused extractor graph, the extractor-only
compile and its parity gate (the regions need 2.8's `fullgraph=True`), `compile_control`'s 2.5.1
rows and the 2.5.1 learner-golden entry. HEAD code on 2.5.1 would therefore compile a graph known to
miscompile there, or die deep in startup with a confusing torch error.

WHO STILL RUNS 2.5.1. A run that RECORDED torch 2.5.1 (or none — `metadata.json` predating the
record) resumes through the launcher PINNED to its own commit, whose `src/` still carries every
2.5.1 path, under the `gen3ai_stable` env the launcher selects (`main.launcher.torch_runtime`). What
this module refuses is HEAD code on that torch: `--no-pin`, `--sync-to-main`, or a bare
`train_rl_agent.py --model` on a 2.5.1 interpreter.

The version is read from the package METADATA (no `import torch`), so the check can run first.
"""
from __future__ import annotations

import importlib.metadata
import re
from typing import Optional, Tuple

#: The lowest torch RELEASE HEAD runs on.
MIN_TORCH: Tuple[int, int] = (2, 8)


def installed_torch() -> Optional[str]:
    """The installed torch version string (`2.8.0+cu126`), or None when torch is not installed."""
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def release(version: str) -> Tuple[int, int]:
    """`2.5.1+cu121` -> (2, 5). Raises ValueError on an unparseable version."""
    m = re.match(r"\s*(\d+)\.(\d+)", str(version))
    if m is None:
        raise ValueError(f"unparseable torch version {version!r}")
    return int(m.group(1)), int(m.group(2))


def refusal(version: Optional[str] = None) -> Optional[str]:
    """None when ``version`` (default: the installed torch) is at or above the floor, else the
    refusal text. An unreadable or missing torch is refused too — never assumed new enough."""
    ver = installed_torch() if version is None else version
    floor = ".".join(map(str, MIN_TORCH))
    try:
        ok = ver is not None and release(ver) >= MIN_TORCH
    except ValueError:
        ok = False
    if ok:
        return None
    return (f"this code (HEAD) runs on torch >= {floor} only, but the interpreter has torch {ver}. "
            f"Every torch-2.5.1 path (the CUDA trunk split, the extractor-only compile, the 2.5.1 "
            f"compile-control rows) was deleted on 2026-10-02. A run that trained on 2.5.1 resumes "
            f"PINNED to its own commit through the launcher (drop --no-pin / --sync-to-main; the "
            f"launcher selects the gen3ai_stable env), or switches torch deliberately with "
            f"--allow-torch-switch under gen3ai_torch28.")
