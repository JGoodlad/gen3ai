"""A GPU window inserted BETWEEN two arms: the previous arm has ended, the next has not launched.

(A window INSIDE a live run is the launcher's ``--gpu-window``, ``designs/launcher/gpu_window.md``; this is only for
the gap between runs, which the launcher cannot see.) The window scripts' rules, in code:

* **Preconditions first, and a failed one is the window's failure** (SOP §3 rules 3-4): the preempt marker absent;
  the GPU idle by a SUCCESSFUL ``nvidia-smi`` (a failing or hanging ``nvidia-smi`` is NOT an idle GPU — the class
  the launcher's window fixed in T46 and the job scripts still carried); the lease held by the chain's owner; the
  window's worktree clean and at or after its required commit.
* **One process per step**, each under ``scripts/ops/gpu_lock.sh`` + ``mem_cap.sh`` with its own wall timeout (one
  pytest for 33 GPU ids let one device-side assert poison 23 verdicts, 10-24).
* **The preempt marker is read before every step**; a step not STARTED inside ``budget_min`` is NOT RUN.
* **The GPU must be idle again after the last step** (a guest left behind would OOM the next arm's first update).
* What a failure MEANS is the chain file's, not this module's: ``on_fail`` ``hold`` stops the chain; ``continue``
  hands off to the next arm (the X59-S window: a failed pre-flight trains no student, and the N1-P chain resumes).
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional, Tuple

from main.ops.arm_chain.runner import Runner, cpu_env, tail
from main.ops.arm_chain.spec import ChainSpec, WindowItem

Probe = Callable[[], Tuple[bool, str]]


def nvidia_idle(timeout_s: float = 60.0) -> Tuple[bool, str]:
    try:
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"nvidia-smi did not answer ({type(e).__name__}) — NOT an idle GPU"
    if p.returncode != 0:
        return False, f"nvidia-smi rc {p.returncode} — NOT an idle GPU"
    pids = " ".join(p.stdout.split())
    return (not pids), ("idle" if not pids else f"compute process(es) {pids}")


def lease_status(repo: str) -> str:
    p = subprocess.run([os.path.join(repo, "scripts", "ops", "gpu_lease.sh"), "status"], capture_output=True, text=True,
                       timeout=60)
    return (p.stdout or p.stderr).splitlines()[0] if (p.stdout or p.stderr) else f"(gpu_lease.sh status rc {p.returncode})"


@dataclass
class WindowResult:
    ok: bool
    lines: List[str] = field(default_factory=list)


def _git(cwd: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True)


def preconditions(spec: ChainSpec, w: WindowItem, gpu_idle: Probe, lease: Callable[[], str],
                  idle_wait_s: float = 300.0, sleep: Callable[[float], None] = time.sleep,
                  monotonic: Callable[[], float] = time.monotonic) -> Optional[str]:
    """None if the window may start, else why not."""
    if os.path.exists(_marker(spec, w)):
        return f"the preempt marker {_marker(spec, w)} is present — the window is NOT run"
    t0 = monotonic()
    while True:
        ok, why = gpu_idle()
        if ok:
            break
        if monotonic() - t0 > idle_wait_s:
            return f"the GPU is not idle after {idle_wait_s:.0f} s: {why}"
        sleep(10)
    st = lease()
    if f"lease owner '{spec.backend.lease_owner}'" not in st:
        return f"the lease is not {spec.backend.lease_owner}'s: {st}"
    if not os.path.isdir(w.worktree):
        return f"no worktree {w.worktree}"
    if _git(w.worktree, "merge-base", "--is-ancestor", w.require_commit, "HEAD").returncode != 0:
        return f"{w.worktree} lacks {w.require_commit}"
    dirty = _git(w.worktree, "status", "--porcelain").stdout
    if dirty.strip():
        return f"{w.worktree} is not clean: {dirty.strip()[:120]}"
    return None


def _marker(spec: ChainSpec, w: WindowItem) -> str:
    return w.preempt_marker if os.path.isabs(w.preempt_marker) else os.path.join(spec.job_dir, w.preempt_marker)


def run_window(spec: ChainSpec, w: WindowItem, runner: Runner, base_env: Mapping[str, str], gpu_idle: Probe,
               lease: Callable[[], str], say: Callable[[str], None],
               monotonic: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
               idle_wait_s: float = 300.0) -> WindowResult:
    why = preconditions(spec, w, gpu_idle, lease, idle_wait_s=idle_wait_s, sleep=sleep, monotonic=monotonic)
    if why:
        say(f"window {w.name}: precondition failed: {why}")
        return WindowResult(False, [why])
    head = _git(w.worktree, "rev-parse", "--short=9", "HEAD").stdout.strip()
    say(f"window {w.name}: preconditions PASS (worktree {head}); {len(w.steps)} step(s), budget {w.budget_min:g} min")
    t0 = monotonic()
    res = WindowResult(True)
    ops = os.path.join(spec.repo, "scripts", "ops")
    for i, st in enumerate(w.steps):
        if os.path.exists(_marker(spec, w)):
            msg = f"step {st.name}: the preempt marker appeared — it and every later step NOT RUN"
            say(f"window {w.name}: {msg}")
            res.ok, res.lines = False, res.lines + [msg]
            break
        if monotonic() - t0 > w.budget_min * 60:
            msg = f"step {st.name}: NOT RUN (past the {w.budget_min:g}-min budget)"
            say(f"window {w.name}: {msg}")
            res.ok, res.lines = False, res.lines + [msg]
            continue
        cwd = w.worktree if st.cwd == "worktree" else spec.repo
        env = cpu_env(base_env, cwd)
        if st.gpu:
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env["GEN3AI_GPU_LEASE_TOKEN_FILE"] = spec.backend.lease_token_file
            env["GEN3AI_TEST_ALLOW_GPU"] = "1"
            argv = [os.path.join(ops, "gpu_lock.sh"), os.path.join(ops, "mem_cap.sh"), f"{st.mem_gb:g}",
                    "timeout", str(st.timeout_s), *st.argv]
        else:
            argv = ["timeout", str(st.timeout_s), *st.argv]
        out = os.path.join(spec.work_dir, f"window_{w.name}_{i}_{st.name}.txt")
        s0 = monotonic()
        rc = runner.run(argv, cwd=cwd, env=env, timeout_s=st.timeout_s + 600, out_path=out)
        msg = f"step {st.name} rc={rc} in {(monotonic() - s0) / 60:.0f} min: {tail(out, 1, 140)}"
        say(f"window {w.name}: {msg}")
        res.lines.append(msg)
        if rc != 0:
            res.ok = False
    ok, why2 = gpu_idle()
    if not ok:
        say(f"window {w.name}: the GPU is NOT idle after the window: {why2}")
        res.ok, res.lines = False, res.lines + [f"GPU not idle after the window: {why2}"]
    say(f"window {w.name}: {'PASS' if res.ok else 'FAILED'} in {(monotonic() - t0) / 60:.0f} min")
    return res
