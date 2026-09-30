"""THE SCRATCH ROOT — where every temp directory this project makes lives: `$GEN3AI_SCRATCH`, else
`~/.cache/gen3ai/tmp` — the REAL DISK, never tmpfs `/tmp`.

Why: `/tmp` on this box is tmpfs — RAM, and a FIXED 1,048,576-inode table. On 2026-09-30 it ran out
of INODES (1,047,791 / 1,048,576, ~395k under the box-wide Inductor cache) with 20 GB of bytes still
free, and a routine gate died with 40 `could not create numbered dir` errors.

ONE implementation, two kinds of caller: `agents.model.compile_cache` (K3's temp compile caches)
imports it; the root `conftest.py` (pytest's temp root) LOADS THIS FILE BY PATH (an `importlib` spec,
no `sys.path` change — it runs before anything guarantees `src/` is importable, and must not disturb
the import-precedence gates). So: stdlib only, no project imports.
"""
from __future__ import annotations

import os
from typing import Optional

SCRATCH_ENV = "GEN3AI_SCRATCH"
DEFAULT_SUBPATH = (".cache", "gen3ai", "tmp")
REFUSED_FS = ("tmpfs", "ramfs")


class ScratchOnRamError(RuntimeError):
    """The declared scratch root is on a RAM filesystem (tmpfs / ramfs)."""


def fs_type(path: str) -> Optional[str]:
    """The filesystem type of the mount holding `path` (longest `/proc/mounts` prefix), or None."""
    real = os.path.realpath(path)
    best, kind = "", None
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1].replace("\\040", " ")
                if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(mnt) >= len(best):
                    best, kind = mnt, parts[2]
    except OSError:
        return None
    return kind


def scratch_root() -> str:
    """`$GEN3AI_SCRATCH`, else `~/.cache/gen3ai/tmp` (created). Raises `ScratchOnRamError` on tmpfs."""
    root = os.environ.get(SCRATCH_ENV) or os.path.join(os.path.expanduser("~"), *DEFAULT_SUBPATH)
    os.makedirs(root, exist_ok=True)
    fs = fs_type(root)
    if fs in REFUSED_FS:
        raise ScratchOnRamError(
            f"the scratch root {root!r} is on {fs} (RAM + a fixed inode table); point "
            f"${SCRATCH_ENV} at a directory on a real disk")
    return root
