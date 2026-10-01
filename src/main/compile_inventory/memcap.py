"""Is THIS process under a cgroup-v2 memory cap? — the precondition of a stack-profiled trace.

WHY (2026-09-30): three host OOM kills (74–82 GB anonymous RSS) each tore down the whole tmux scope
and every Claude session in it. The cause was this tool's CPU `torch.profiler(with_stack=True)`
over a dynamo-compiled update: under a 12 GB scope RSS went 4.6 → 11.05 GB in under 9 s inside
the profiler block, and the scope's OOM killer took only the worker
(`designs/research_state/measurements/k8_inventory/README.md`, "Memory"). A stack profile is
therefore REFUSED unless it is asked for explicitly AND this process's cgroup — or an ancestor
below the root — carries a finite `memory.max`, so an overrun kills that scope and nothing else.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

CGROUP_ROOT = Path("/sys/fs/cgroup")


def own_cgroup(proc_cgroup: Path = Path("/proc/self/cgroup")) -> Optional[str]:
    """The cgroup-v2 path of this process (the ``0::<path>`` line), or None."""
    try:
        for line in proc_cgroup.read_text().splitlines():
            if line.startswith("0::"):
                return line[3:].strip() or "/"
    except OSError:
        return None
    return None


def memory_cap_bytes(cgroup: Optional[str] = None, root: Path = CGROUP_ROOT) -> Optional[int]:
    """The TIGHTEST finite `memory.max` on ``cgroup`` or any ancestor below the root; None when
    every level reads ``max`` (uncapped) or nothing can be read."""
    cg = own_cgroup() if cgroup is None else cgroup
    if not cg:
        return None
    caps = []
    parts = [p for p in cg.strip("/").split("/") if p]
    for i in range(len(parts), 0, -1):
        f = root.joinpath(*parts[:i]) / "memory.max"
        try:
            v = f.read_text().strip()
        except OSError:
            continue
        if v and v != "max":
            try:
                caps.append(int(v))
            except ValueError:
                continue
    return min(caps) if caps else None


class StackProfileRefused(RuntimeError):
    pass


def require_cap_for_stack_profile(allowed: bool, cap: Optional[int]) -> None:
    """Raise unless a stack profile was explicitly allowed AND a finite cap is in force."""
    if not allowed:
        raise StackProfileRefused(
            "a with_stack CPU profile is OFF by default: it drove three host OOM kills on "
            "2026-09-30 (74–82 GB). Pass --allow-stack-profile AND run under a memory cap.")
    if cap is None:
        raise StackProfileRefused(
            "--allow-stack-profile given, but this process's cgroup memory.max reads 'max' at "
            "every level — run it under a cap (scripts/ops/mem_cap.sh, or `systemd-run --user "
            "--scope -p MemoryMax=12G -p MemorySwapMax=0 ...`).")
