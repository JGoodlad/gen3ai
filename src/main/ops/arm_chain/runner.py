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
import subprocess
import threading
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
    def run(self, argv: Sequence[str], *, cwd: str, env: Mapping[str, str], timeout_s: float, out_path: str) -> int:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as out:
            try:
                return subprocess.run(list(argv), cwd=cwd, env=dict(env), stdout=out, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, timeout=timeout_s).returncode
            except subprocess.TimeoutExpired:
                out.write(f"\n[arm_chain] TIMEOUT after {timeout_s:.0f} s\n".encode())
                return 124

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


def tail(path: str, n: int = 2, width: int = 220) -> str:
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return f"(no {path})"
    return " ".join(lines[-n:])[:width]
