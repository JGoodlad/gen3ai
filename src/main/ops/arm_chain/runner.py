"""The one seam through which the chain runs anything, and the environments it runs it in.

Two environments, built from scratch per call so nothing leaks from the supervisor's own environment:

* :func:`cpu_env` — a CPU step: ``CUDA_VISIBLE_DEVICES=""`` (root rule 5), ``PYTHONPATH=<repo>/src``, and
  ``GEN3AI_BLOCKING`` when the step blocks the GPU's next work (an arm end does);
* :func:`go_env` — a launch: an ALLOW-list of the base environment (so ``CUDA_VISIBLE_DEVICES`` and
  ``GEN3AI_TEST_ALLOW_GPU`` can never leak in), ``PYTHONPATH=<repo>/src`` and the lease token file.

THE CLASS BUG :func:`go_env` closes: the arm-end's CPU steps exported ``CUDA_VISIBLE_DEVICES=""``, the next GO
inherited it, and the launch saw no GPU (toy_x47 3602's first GO, 2026-10-20); every later script re-learned it.
:func:`assert_card_visible` is the belt to that brace, checked again right before every GO.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import Dict, List, Mapping, Optional, Protocol, Sequence

from main.ops.arm_chain.procwatch import ProcRef, ref_of


class CardNotVisible(RuntimeError):
    pass


def cpu_env(base: Mapping[str, str], repo: str, blocking: Optional[str] = None,
            models_dir: Optional[str] = None) -> Dict[str, str]:
    env = dict(base)
    if models_dir:   # the run archive every CPU tool resolves is the chain's, never the supervisor's ambient one
        env["GEN3AI_MODELS_DIR"] = models_dir
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.path.join(repo, "src")
    env.pop("GEN3AI_TEST_ALLOW_GPU", None)
    if blocking:
        env["GEN3AI_BLOCKING"] = blocking
    else:
        env.pop("GEN3AI_BLOCKING", None)
    return env


#: The GO's environment is an ALLOW-list: nothing a window or a CPU step exported (an empty CUDA_VISIBLE_DEVICES,
#: GEN3AI_TEST_ALLOW_GPU, a worktree's PYTHONPATH) can reach a launch.
GO_ENV_KEEP = ("HOME", "PATH", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "SHELL", "XDG_RUNTIME_DIR")


def go_env(base: Mapping[str, str], repo: str, lease_token_file: str, python: str) -> Dict[str, str]:
    """``GEN3AI_PYTHON`` is the chain file's interpreter, never the supervisor's; ``GEN3AI_BLOCKING`` is NOT passed
    (the trainer would inherit it, and anything it starts would skip the heavy slots)."""
    env = {k: v for k, v in base.items() if k in GO_ENV_KEEP}
    env["PYTHONPATH"] = os.path.join(repo, "src")
    env["GEN3AI_GPU_LEASE_TOKEN_FILE"] = lease_token_file
    env["GEN3AI_PYTHON"] = python
    return env


def assert_card_visible(env: Mapping[str, str]) -> None:
    if "CUDA_VISIBLE_DEVICES" in env and env["CUDA_VISIBLE_DEVICES"] == "":
        raise CardNotVisible("CUDA_VISIBLE_DEVICES is set and EMPTY: the GO would see no GPU")


def agent_prefix(cores: str, nice: int) -> List[str]:
    """Root rule 5b: agent CPU work runs on the agent cores (``taskset``), at ``nice`` (19, or 0 when BLOCKING)."""
    return ["nice", "-n", str(nice), "taskset", "-c", cores]


class Runner(Protocol):
    def run(self, argv: Sequence[str], *, cwd: str, env: Mapping[str, str], timeout_s: float, out_path: str) -> int:
        """Run to completion; stdout + stderr to ``out_path``; the exit code (124 on a timeout)."""

    def spawn(self, argv: Sequence[str], *, cwd: str, env: Mapping[str, str], out_path: str) -> ProcRef:
        """Start detached in its own session (``setsid``, stdin /dev/null); the child's identity."""


class SubprocessRunner:
    #: after a timeout's SIGTERM to the step's process group, how long before the SIGKILL
    term_grace_s: float = 10.0

    def run(self, argv: Sequence[str], *, cwd: str, env: Mapping[str, str], timeout_s: float, out_path: str) -> int:
        """The step runs in its OWN process group (``start_new_session``), so a timeout signals the whole group: a
        ``subprocess.run`` timeout SIGKILLed only the direct child (bash) and left the kit's python children — a
        launcher ``--dry-run`` holding a CUDA context — to fail the next GO's GPU-idle check (review 10-26, finding 5).
        SIGTERM first (a CUDA context is released cleanly), SIGKILL after ``term_grace_s``."""
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as out:
            p = subprocess.Popen(list(argv), cwd=cwd, env=dict(env), stdout=out, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                return p.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                left = kill_group(p, self.term_grace_s)
                out.write(f"\n[arm_chain] TIMEOUT after {timeout_s:.0f} s: process group {p.pid} signalled"
                          f"{'' if not left else f'; STILL ALIVE after SIGKILL: {left}'}\n".encode())
                return 124
            except BaseException:   # the supervisor itself is dying: never leave the step's group behind
                kill_group(p, self.term_grace_s)
                raise

    def spawn(self, argv: Sequence[str], *, cwd: str, env: Mapping[str, str], out_path: str) -> ProcRef:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "ab") as out:
            p = subprocess.Popen(list(argv), cwd=cwd, env=dict(env), stdout=out, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        ref = ref_of(p.pid)
        # reap it when it exits (a long-lived supervisor must not collect zombies); identity is (pid, start ticks),
        # so a reaped and re-used pid can never read as this child
        threading.Thread(target=p.wait, daemon=True).start()
        if ref is None:   # it already exited: an identity that can never read as alive
            return ProcRef(p.pid, -1)
        return ref


def _group_pids(pgid: int, proc: str = "/proc") -> List[int]:
    from utils import procfs
    return sorted(pid for pid, pr in procfs.snapshot(proc).items() if pr.pgrp == pgid and pr.state not in ("Z", "X"))


def kill_group(p: "subprocess.Popen[bytes]", grace_s: float) -> List[int]:
    """SIGTERM the process group ``p`` leads (its own session: nothing outside it shares the pgid), wait up to
    ``grace_s`` for it to empty, SIGKILL what is left, reap ``p``. The pids still alive after that (normally none)."""
    pgid = p.pid
    with_group = True
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        with_group = False
    t0 = time.monotonic()
    while with_group and _group_pids(pgid) and time.monotonic() - t0 < grace_s:
        if p.poll() is None:
            try:
                p.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(0.2)
    if with_group and _group_pids(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    t1 = time.monotonic()
    left = _group_pids(pgid)
    while left and time.monotonic() - t1 < 5:
        time.sleep(0.1)
        left = _group_pids(pgid)
    return left


def tail(path: str, n: int = 2, width: int = 220) -> str:
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return f"(no {path})"
    return " ".join(lines[-n:])[:width]
