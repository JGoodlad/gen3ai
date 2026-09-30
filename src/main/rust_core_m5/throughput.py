"""THE THROUGHPUT A/B (M5 Lane J): today's Python rollout path vs the Rust env core at matched N.

A DESCRIPTOR, not a gate: the program doc (``designs/endstate/program_rust_core.md`` §2 M5, order
constraint 4) names "the throughput A/B at ``--n-envs 48``" as part of M5's gate but registers no
bar, so this reports and judges nothing. A benchmark's output IS the measurement: it WARNS on a busy
box (``utils.contention.warn_if_contended``) and never scales anything by contention.

* **arm ``python`` (today's path)** — N SB3 ``SubprocVecEnv`` workers (start method ``forkserver``,
  as training), each one production-surface ``Gen3Env`` (``trainee_env_kwargs(production_args())``,
  the rust ``sim_bridge`` transport with the core obs), wrapped as training wraps it
  (``MaskableAgentWrapper`` — which absorbs every step the trainee is not asked on — then
  ``Monitor``), the opponent from the :class:`hooks.OpponentMix`.
* **arm ``rust``** — the Rust env core: N envs on T threads, front end ``proc`` (training's default
  per the program doc) or ``ffi``, the production spec (every label family, the production terminal,
  ``turn_limit = episode.stall_threshold()``), the opponent route from the same hook; ``ep_team`` /
  ``ep_seed`` / ``ep_opp`` re-staged for every env whose ``episode`` column moved (F-LE-4).

Both arms take the trainee's action from ONE :class:`hooks.TraineeInference` instance over the
batch of decision rows (what SB3 does). Each arm is BUILT ONCE (48 Python workers take tens of
seconds to start; that is reported as ``startup_s``, not timed), warmed, then timed in interleaved
blocks: pair k runs ``python, rust`` for even k and ``rust, python`` for odd k. A block runs until
``--block-seconds`` have passed (and at least ``--min-steps`` vec steps).

Per block: wall ms per vec step, vec steps/s, TRAINEE DECISIONS/s (python: rows with a legal action,
one per env per step since the wrapper absorbs wait steps; rust: rows with ``need[:, 0] = 1``), the
time inside the inference hook, and CPU-seconds per decision over the WHOLE process tree (this
process + every descendant: workers, their ``sim_bridge`` children, the ``proc`` child), read from
``/proc/<pid>/stat`` before and after the window. The ratio rust / python of decisions/s and of
CPU-s per decision carries a 95 % bootstrap CI over pairs (geometric mean of per-pair ratios).

    export PYTHONPATH=$PYTHONPATH:src
    python -m main.rust_core_m5 throughput --n-envs 48 --threads 8 --pairs 4 --out <dir>/throughput.json

A CUDA inference hook re-executes this module under ``flock`` on the box's GPU lock (``gates.GPU_LOCK``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from main.rust_core_m5 import hooks as H

SCHEMA = "m5_laneJ_throughput_v1"
from main.rust_core_m5.gates import GPU_LOCK  # noqa: E402 — one definition of the lock path
GPU_LOCK_MARKER = "GEN3AI_M5J_GPU_LOCK"
ARMS = ("python", "rust")
#: The PRODUCTION-shape arms (M5 Lane G, ``production.py``): today's path, and the Rust collector SERIAL
#: or OVERLAPPED (env-step / inference overlap on two core halves), each with p2's sampling KEYED or by
#: per-row GENERATORS (F-LE-8's before / after).
PRODUCTION_ARMS = ("python", "rust_serial_keyed", "rust_serial_generator", "rust_overlap_keyed",
                   "rust_overlap_generator")
_TICK = float(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else 100.0


# ---------------------------------------------------------------------------------------------
# the argv
# ---------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m main.rust_core_m5 throughput",
                                 description="The M5 throughput A/B: Python rollout path vs the Rust env core.")
    ap.add_argument("--out", required=True, help="the JSON result (refused under models/)")
    ap.add_argument("--n-envs", type=int, default=48)
    ap.add_argument("--threads", type=int, default=8, help="rust arm: core worker threads (T)")
    ap.add_argument("--front", choices=("proc", "ffi"), default="proc", help="rust arm front end")
    ap.add_argument("--profile", choices=("release", "selfcheck"), default="release",
                    help="build of BOTH arms' Rust binaries (rust_env core; the python arm's sim_bridge)")
    ap.add_argument("--arms", default="python,rust", help="comma list; one arm = no ratio")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--block-seconds", type=float, default=20.0)
    ap.add_argument("--min-steps", type=int, default=1, help="vec steps a block runs at least")
    ap.add_argument("--warmup-steps", type=int, default=20, help="per arm, after startup, untimed")
    ap.add_argument("--inference", choices=tuple(H.INFERENCE), default="random")
    ap.add_argument("--ckpt", default=None, help="t2: a .zip, a baselines.json name, or none (perturbed fresh)")
    ap.add_argument("--t2-backend", default="eager", choices=("eager", "graph", "aot"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--buckets", default="8,48,128")
    ap.add_argument("--opponent", choices=tuple(H.OPPONENTS), default="uniform_random")
    ap.add_argument("--collector", choices=tuple(H.COLLECTORS), default="step")
    ap.add_argument("--n-steps", type=int, default=2048, help="the step collector's window")
    ap.add_argument("--seed", type=int, default=20260929)
    # M5 Lane G — the PRODUCTION shape (``--opponent production --collector complete_game --inference learner``)
    ap.add_argument("--pool", default=None, help="production: a run's snapshots/ dir (read-only; the last "
                                                 "--pool-size are symlinked into a temp dir)")
    ap.add_argument("--pool-size", type=int, default=20)
    ap.add_argument("--self-play-fraction", type=float, default=0.9)
    ap.add_argument("--target", type=int, default=98_304, help="complete_game: rows per update (the trigger)")
    ap.add_argument("--opponent-sampling", choices=("keyed", "generator"), default="keyed",
                    help="production rust arm: p2's sampling (Lane G, F-LE-8)")
    ap.add_argument("--no-compile-opponents", dest="compile_opponents", action="store_false",
                    help="production python arm: eager CPU opponents (production compiles them)")
    ap.add_argument("--t2-buckets", default=None, help="production rust arms: T2 buckets (default the collector's (8, N))")
    ap.add_argument("--compile-trainee", action="store_true",
                    help="learner inference: --compile-trainer's extractor compile (CUDA)")
    ap.add_argument("--wait-while-exists", default=None,
                    help="before the timed blocks, wait while this file exists (a peer's busy flag); recorded")
    return ap


def wants_cuda(a: argparse.Namespace) -> bool:
    return a.inference in ("t2", "learner") and str(a.device).startswith("cuda")


def flock_reexec_argv(argv: Sequence[str], env: Mapping[str, str], python: str) -> Optional[List[str]]:
    """The argv to re-exec under the GPU lock, or ``None`` to run here: re-exec iff the run asks for
    a CUDA device and ``$GEN3AI_M5J_GPU_LOCK`` is not already set (i.e. we are not already under it)."""
    a, _ = build_parser().parse_known_args(list(argv))
    if not wants_cuda(a) or env.get(GPU_LOCK_MARKER) == "1":
        return None
    return ["flock", GPU_LOCK, python, "-m", "main.rust_core_m5.throughput", *argv]


def refuse_models_out(out: str, models_dirs: Optional[Iterable[Optional[Path]]] = None) -> Path:
    """``out`` resolved, or ``ValueError`` when it lies under a run archive (``models/``)."""
    p = Path(out).expanduser().resolve()
    if models_dirs is None:
        from utils.paths import main_models_dir, repo_root

        env_dir = os.environ.get("GEN3AI_MODELS_DIR")
        models_dirs = [main_models_dir(), repo_root() / "models", Path(env_dir) if env_dir else None]
    for d in models_dirs:
        if d is None:
            continue
        d = Path(d).expanduser().resolve()
        if p == d or d in p.parents:
            raise ValueError(f"--out {out} is under the run archive {d}: a benchmark never writes into models/")
    return p


# ---------------------------------------------------------------------------------------------
# /proc: CPU over a process tree
# ---------------------------------------------------------------------------------------------

def read_stat(pid: int) -> Optional[Tuple[int, int, int]]:
    """``(ppid, starttime, cpu_ticks)`` of ``pid`` from ``/proc/<pid>/stat`` — ``cpu_ticks`` =
    utime + stime + cutime + cstime (the reaped children's share too, so a child that exits inside
    the window is not lost). ``None`` if the process is gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            text = f.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    rest = text[text.rindex(")") + 2:].split()
    # rest[0] is field 3 (state): ppid = 4, utime..cstime = 14..17, starttime = 22
    return int(rest[1]), int(rest[19]), int(rest[11]) + int(rest[12]) + int(rest[13]) + int(rest[14])


def tree_snapshot(root: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Every process in ``root``'s tree (itself included): ``{(pid, starttime): (ppid, cpu_ticks)}``."""
    stats: Dict[int, Tuple[int, int, int]] = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            s = read_stat(int(name))
            if s is not None:
                stats[int(name)] = s
    kids: Dict[int, List[int]] = {}
    for pid, (ppid, _, _) in stats.items():
        kids.setdefault(ppid, []).append(pid)
    out: Dict[Tuple[int, int], Tuple[int, int]] = {}
    todo = [root] if root in stats else []
    while todo:
        pid = todo.pop()
        ppid, start, ticks = stats[pid]
        out[(pid, start)] = (ppid, ticks)
        todo.extend(kids.get(pid, ()))
    return out


def _subtree(snap: Mapping[Tuple[int, int], Tuple[int, int]], roots: Iterable[int]) -> set:
    parent = {k[0]: v[0] for k, v in snap.items()}
    rs = set(roots)
    keep = set()
    for (pid, start) in snap:
        p: Optional[int] = pid
        while p is not None:
            if p in rs:
                keep.add((pid, start))
                break
            p = parent.get(p)
    return keep


def cpu_delta(before: Mapping, after: Mapping, only: Optional[set] = None) -> float:
    """CPU-seconds the tree spent between two snapshots (a process new in ``after`` counts whole);
    ``only`` restricts to those keys."""
    ticks = 0
    for k, (_pp, t) in after.items():
        if only is not None and k not in only:
            continue
        ticks += t - (before[k][1] if k in before else 0)
    return ticks / _TICK


# ---------------------------------------------------------------------------------------------
# the ratio + CI
# ---------------------------------------------------------------------------------------------

def ratio_ci(num: Sequence[float], den: Sequence[float], *, n_boot: int = 10_000, seed: int = 0,
             alpha: float = 0.05) -> Dict[str, Any]:
    """num / den over matched PAIRS: the point is the geometric mean of the per-pair ratios; the CI
    a percentile bootstrap of that mean over pairs (resampled with replacement). One pair: no CI."""
    import numpy as np

    if len(num) != len(den) or not num:
        raise ValueError("ratio_ci needs equal, non-empty paired sequences")
    x, y = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    if (x <= 0).any() or (y <= 0).any():
        raise ValueError("ratio_ci: every value must be > 0")
    r = np.log(x / y)
    k = r.size
    out: Dict[str, Any] = {"point": float(np.exp(r.mean())), "per_pair": [float(v) for v in np.exp(r)],
                           "n_pairs": int(k), "method": "geometric mean of per-pair ratios; percentile bootstrap"}
    if k < 2:
        out.update(lo=None, hi=None, note="one pair: no CI")
        return out
    rng = np.random.default_rng(seed)
    means = r[rng.integers(0, k, size=(n_boot, k))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    out.update(lo=float(np.exp(lo)), hi=float(np.exp(hi)), level=1 - alpha, n_boot=n_boot)
    if k < 5:
        out["note"] = f"{k} pairs: a bootstrap over so few pairs is coarse (its CI can only take a few values)"
    return out


# ---------------------------------------------------------------------------------------------
# arm: python (today's path)
# ---------------------------------------------------------------------------------------------

def _python_worker(idx: int, args: Any, env_kwargs: Dict[str, Any], opponent: Any, sim_bridge_bin: str,
                   tag: str) -> Any:
    """ONE SubprocVecEnv worker's env, built INSIDE the worker (``create_training_env_random``'s
    shape: the production-surface Gen3Env on the rust bridge, MaskableAgentWrapper, Monitor)."""
    os.environ["POKESIM_SIM_BRIDGE_BIN"] = sim_bridge_bin   # the declared binary, never a cargo build
    import torch

    torch.set_num_threads(1)
    from poke_env import AccountConfiguration
    from stable_baselines3.common.monitor import Monitor

    from agents.observation.state_encoder import load_mappings
    from agents.training.gen3_env import Gen3Env
    from agents.training.reward_config import RewardConfig
    from agents.training.reward_manager import Gen3RewardManager
    from agents.training.stall import StallConfig
    from agents.training.wrappers import MaskableAgentWrapper
    from utils.bridge.bridge_session import attach_bridge_transport
    from utils.logging.levels import LogLevel
    from utils.team_sources import team_list
    from utils.teambuilder import Gen3Teambuilder

    teams = team_list("pool")
    tb = Gen3Teambuilder(teams)
    env = Gen3Env(load_mappings(), battle_format="gen3ou", team=tb, log_level=LogLevel.QUIET,
                  stall_config=StallConfig(), reward_fn=Gen3RewardManager(config=RewardConfig.from_args(args)),
                  account_configuration1=AccountConfiguration(f"{tag}e{idx}"[:18], None),
                  start_listening=False, opponent_team=tb, **env_kwargs)
    attach_bridge_transport(env, battle_format="gen3ou", impl="rust",
                            core_obs=(env_kwargs.get("obs_source") == "core"))
    wrapped = MaskableAgentWrapper(env, opponent=opponent.python_opponent(idx, tag, tb), self_play_fraction=0.0,
                                   rng_seed=idx, opponent_pool_team=tb,
                                   critic=getattr(args, "critic", None) or "shaped")
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space

    class _Monitor(Monitor):
        """Training's ``Monitor``; only ``close`` differs: poke-env refuses to reset a player whose
        battle is mid-game, which is every env at the A/B's teardown. Shutdown only — the stepping
        path is the parent class's, untouched."""

        def close(self) -> None:
            try:
                super().close()
            except OSError as exc:
                if "still running" not in str(exc):
                    raise

    return _Monitor(wrapped)


def sim_bridge_path(profile: str) -> Path:
    """The python arm's ``sim_bridge``: THIS checkout's build of ``profile`` (never auto-built)."""
    from utils.bridge.sim_bridge_bin import build_command, expected_bin_path

    p = expected_bin_path("sim_bridge", selfcheck=(profile == "selfcheck"))
    if not p.is_file():
        raise FileNotFoundError(f"{p} is missing — build it: CARGO_TARGET_DIR=<this checkout>/src/rust_sim/target "
                                f"{build_command('sim_bridge', selfcheck=(profile == 'selfcheck'))}")
    return p.resolve()


class PythonArm:
    name = "python"

    def __init__(self, n: int, inference: Any, opponent: Any, collector: Any, profile: str):
        self.n, self.inference, self.opponent, self.collector, self.profile = n, inference, opponent, collector, profile
        self.vec: Any = None
        self.info: Dict[str, Any] = {}

    def build(self) -> None:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        from main.rust_core_cutover.envs import production_args
        from main.train.env_factory import resolved_obs_source, trainee_env_kwargs

        args = production_args()
        kw = trainee_env_kwargs(args)
        if resolved_obs_source(args) != kw["obs_source"]:
            raise AssertionError("trainee_env_kwargs and resolved_obs_source disagree")
        binary = str(sim_bridge_path(self.profile))
        tag = f"M5J{os.getpid() % 10000}"
        fns = [partial(_python_worker, i, args, kw, self.opponent, binary, tag) for i in range(self.n)]
        self.vec = SubprocVecEnv(fns, start_method="forkserver")
        obs = self.vec.reset()
        self._obs, self._mask = obs["observation"], obs["action_mask"]
        self.info = {"vec_env": "SubprocVecEnv", "start_method": "forkserver", "sim_bridge": binary,
                     "obs_source": kw["obs_source"], "wrapper": "MaskableAgentWrapper + Monitor",
                     "worker_pids": self.roots()}

    def roots(self) -> List[int]:
        return [int(p.pid) for p in self.vec.processes] if self.vec is not None else []

    def step(self) -> Tuple[int, float, float, int]:
        import numpy as np

        m = np.asarray(self._mask).astype(bool)
        rows = np.flatnonzero(m.any(axis=1))
        actions = np.zeros(self.n, dtype=np.int64)
        t0 = time.perf_counter()
        if rows.size:
            actions[rows] = self.inference(np.asarray(self._obs[rows], dtype=np.float32), m[rows])
        t1 = time.perf_counter()
        obs, _rew, dones, _infos = self.vec.step(actions)
        t2 = time.perf_counter()
        self._obs, self._mask = obs["observation"], obs["action_mask"]
        self.collector.observe(dones)
        return int(rows.size), t1 - t0, t2 - t1, int(np.count_nonzero(dones))

    def close(self) -> None:
        if self.vec is not None:
            self.vec.close()
            self.vec = None


# ---------------------------------------------------------------------------------------------
# arm: rust (the env core)
# ---------------------------------------------------------------------------------------------

class RustArm:
    name = "rust"

    def __init__(self, n: int, threads: int, front: str, profile: str, inference: Any, opponent: Any,
                 collector: Any, seed: int):
        import numpy as np

        self.n, self.threads, self.front, self.profile = n, threads, front, profile
        self.inference, self.opponent, self.collector = inference, opponent, collector
        self._rng = np.random.default_rng(seed)
        self.core: Any = None
        self.info: Dict[str, Any] = {}
        self.refused = 0

    def spec(self, teams: Sequence[str]) -> str:
        from main.rust_core_cutover.envs import production_args
        from utils.rust_env import columns as C
        from utils.rust_env import episode as EP
        from utils.rust_env import protocol as P

        args = production_args()
        return P.spec_json(n=self.n, threads=self.threads, teams=list(teams), names=("m5jone", "m5jtwo"),
                           decision_tense=bool(getattr(args, "progress_decision_tense", False)),
                           switch_freeze=bool(getattr(args, "progress_switch_freeze", False)),
                           turn_limit=EP.stall_threshold(), refusal_budget=max(64, self.n), bank_dir=None,
                           labels=tuple(C.LABEL_FAMILIES), terminal=None, opponents=self.opponent.rust_routes())

    def build(self) -> None:
        import numpy as np

        from main.rust_core_cutover.envs import packed_teams
        from utils.rust_env import ffi, proc

        self.teams = packed_teams("pool")
        spec = self.spec(self.teams)
        poison = self.profile == "selfcheck"
        if self.front == "proc":
            binary = proc.default_path(self.profile)
            self.core = proc.ProcCore(spec, binary=binary, nan_poison=poison, auto_respawn=False)
        else:
            binary = ffi.default_path(self.profile)
            self.core = ffi.FfiCore(spec, lib=ffi.load(binary, nan_poison=poison))
        c = self.core.cols
        self._stage(range(self.n))
        self.core.reset()
        self._stage(range(self.n))         # RESET consumed the staged inputs: stage every env's NEXT episode
        self._ep = np.array(c["episode"], copy=True)
        self.info = {"front": self.front, "binary": str(binary), "threads": self.threads,
                     "labels": sorted(json.loads(spec)["labels"]), "turn_limit": json.loads(spec)["turn_limit"],
                     "terminal": json.loads(spec)["terminal"], "opponents": json.loads(spec)["opponents"],
                     "teams": len(self.teams), "proc_pid": self.roots()}

    def _stage(self, envs: Iterable[int]) -> None:
        c = self.core.cols
        k = len(self.teams)
        for i in envs:
            x = int(self._rng.integers(k))
            y = int(self._rng.integers(k - 1))
            c["ep_team"][i] = [x, y + (y >= x)]
            c["ep_seed"][i] = self._rng.integers(0, 65536, 4)
            c["ep_opp"][i] = self.opponent.ep_opp(int(i))

    def roots(self) -> List[int]:
        pid = getattr(self.core, "pid", None)
        return [int(pid)] if pid else []

    def step(self) -> Tuple[int, float, float, int]:
        import numpy as np

        c = self.core.cols
        if c["need"][:, 1].any():
            raise RuntimeError("the core asks the CALLER for p2 on a bot route — the opponent hook is not in the core")
        idx = np.flatnonzero(c["need"][:, 0] == 1)
        c["action"][:] = -1                  # no stale action on an env without a decision this op
        t0 = time.perf_counter()
        if idx.size:
            c["action"][idx, 0] = self.inference(c["obs"][idx, 0], c["mask"][idx, 0].astype(bool))
        t1 = time.perf_counter()
        self.core.step()
        t2 = time.perf_counter()
        moved = np.flatnonzero(c["episode"] != self._ep)
        if moved.size:                       # F-LE-4: re-stage every env whose episode moved
            self._stage(moved)
            self._ep[moved] = c["episode"][moved]
        self.refused += int(np.count_nonzero(c["refused"]))
        self.collector.observe(c["done"])
        return int(idx.size), t1 - t0, t2 - t1, int(np.count_nonzero(c["done"]))

    def close(self) -> None:
        if self.core is not None:
            self.info["after_freeze"] = self.core.after_freeze()
            self.info["counters"] = self.core.counters()
            self.core.close()
            self.core = None


# ---------------------------------------------------------------------------------------------
# the timed block
# ---------------------------------------------------------------------------------------------

def run_block(arm: Any, *, seconds: float, min_steps: int, root: Optional[int] = None) -> Dict[str, Any]:
    """Step ``arm`` for at least ``seconds`` and ``min_steps``; the metrics of that window."""
    root = os.getpid() if root is None else root
    before = tree_snapshot(root)
    steps = decisions = dones = 0
    inf_s = env_s = 0.0
    comp0 = arm.components() if hasattr(arm, "components") else None
    gpu = _GpuUtil() if getattr(arm, "watch_gpu", False) else None
    t0 = time.perf_counter()
    while True:
        d, i_s, e_s, k = arm.step()
        steps += 1
        decisions += d
        inf_s += i_s
        env_s += e_s
        dones += k
        wall = time.perf_counter() - t0
        if wall >= seconds and steps >= min_steps:
            break
    gpu_mean = gpu.stop() if gpu is not None else None
    comp = None
    if comp0 is not None:
        c1 = arm.components()
        comp = {k: 1e3 * (c1[k] - comp0.get(k, 0.0)) / steps for k in c1}
    after = tree_snapshot(root)
    own = _subtree(after, arm.roots())
    cpu = cpu_delta(before, after)
    host = cpu_delta(before, after, only={k for k in after if k[0] == root})
    own_cpu = cpu_delta(before, after, only=own)
    return {"arm": arm.name, "wall_s": wall, "vec_steps": steps, "decisions": decisions, "episodes_ended": dones,
            "ms_per_vec_step": 1e3 * wall / steps, "vec_steps_per_s": steps / wall,
            "decisions_per_s": decisions / wall if wall > 0 else 0.0,
            "inference_s": inf_s, "env_s": env_s, "inference_share": inf_s / wall if wall > 0 else 0.0,
            "cpu_s_tree": cpu, "cpu_s_host": host, "cpu_s_arm_subtree": own_cpu,
            "cpu_s_bystander": cpu - host - own_cpu,
            "cpu_us_per_decision": 1e6 * cpu / decisions if decisions else None,
            "tree_processes": len(after), "components_ms_per_step": comp, "gpu_util_mean": gpu_mean,
            "stamp": getattr(arm, "stamp", None)}


class _GpuUtil:
    """GPU utilisation over a block: ``nvidia-smi --query-gpu=utilization.gpu -lms 250`` streamed from a
    child for the block's length; ``stop()`` returns the mean % (None when nvidia-smi is unavailable)."""

    def __init__(self, interval_ms: int = 250):
        self.samples: List[float] = []
        try:
            self.p: Any = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", f"-lms={interval_ms}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            self.p = None

    def stop(self) -> Optional[float]:
        if self.p is None:
            return None
        self.p.terminate()
        try:
            out, _ = self.p.communicate(timeout=5)
        except Exception:
            self.p.kill()
            return None
        vals = []
        for line in (out or "").splitlines():
            try:
                vals.append(float(line.strip().split(",")[0]))
            except ValueError:
                continue
        return float(sum(vals) / len(vals)) if vals else None


def schedule(pairs: int, arms: Sequence[str]) -> List[Tuple[int, str]]:
    """Interleaved ``(pair, arm)``: A B, B A, A B, … for two arms (one arm: once per pair); more than two
    arms ROTATE (A B C, B C A, C A B, …), so every arm takes every position equally often."""
    out = []
    arms = list(arms)
    for k in range(pairs):
        if len(arms) <= 2:
            order = arms if k % 2 == 0 else list(reversed(arms))
        else:
            r = k % len(arms)
            order = arms[r:] + arms[:r]
        out.extend((k, a) for a in order)
    return out


def summarize(blocks: Sequence[Mapping[str, Any]], arms: Sequence[str], seed: int) -> Tuple[Dict, Dict]:
    import numpy as np

    per: Dict[str, Dict[str, Any]] = {}
    for a in arms:
        bs = [b for b in blocks if b["arm"] == a]
        cpd = [b["cpu_us_per_decision"] for b in bs if b["cpu_us_per_decision"] is not None]
        per[a] = {"blocks": len(bs), "decisions": int(sum(b["decisions"] for b in bs)),
                  "decisions_per_s_mean": float(np.mean([b["decisions_per_s"] for b in bs])),
                  "ms_per_vec_step_mean": float(np.mean([b["ms_per_vec_step"] for b in bs])),
                  "vec_steps_per_s_mean": float(np.mean([b["vec_steps_per_s"] for b in bs])),
                  "cpu_us_per_decision_mean": float(np.mean(cpd)) if cpd else None,
                  "inference_share_mean": float(np.mean([b["inference_share"] for b in bs]))}
    for a in arms:                        # a 95 % bootstrap CI over BLOCKS for the two headline rates
        bs = [b for b in blocks if b["arm"] == a]
        for key in ("decisions_per_s", "ms_per_vec_step"):
            per[a][f"{key}_ci95"] = _mean_ci([b[key] for b in bs], seed=seed)
        comps = [b["components_ms_per_step"] for b in bs if b.get("components_ms_per_step")]
        if comps:
            mean_c = {k: float(np.mean([c[k] for c in comps])) for k in comps[0]}
            per[a]["components_ms_per_step_mean"] = mean_c
            per[a]["owner_breakdown_ms_per_step"] = owner_breakdown(mean_c, per[a]["ms_per_vec_step_mean"])
        gpu = [b["gpu_util_mean"] for b in bs if b.get("gpu_util_mean") is not None]
        per[a]["gpu_util_mean"] = float(np.mean(gpu)) if gpu else None
        stamps = [b.get("stamp") for b in bs if b.get("stamp")]
        if stamps:
            per[a]["stamp"] = stamps[0]
    ratios: Dict[str, Any] = {}
    pairs = sorted({b["pair"] for b in blocks})
    by = {(b["pair"], b["arm"]): b for b in blocks}
    for other in [x for x in arms if x != "python"] if "python" in arms else []:
        ok = [k for k in pairs if (k, "python") in by and (k, other) in by
              and by[(k, "python")]["decisions"] and by[(k, other)]["decisions"]]
        if not ok:
            continue
        tag = "rust" if other == "rust" else other
        ratios[f"decisions_per_s_{tag}_over_python"] = ratio_ci(
            [by[(k, other)]["decisions_per_s"] for k in ok], [by[(k, "python")]["decisions_per_s"] for k in ok],
            seed=seed)
        ratios[f"cpu_s_per_decision_{tag}_over_python"] = ratio_ci(
            [by[(k, other)]["cpu_us_per_decision"] for k in ok],
            [by[(k, "python")]["cpu_us_per_decision"] for k in ok], seed=seed + 1)
    rust_arms = [x for x in arms if x.startswith("rust_")]
    for x in rust_arms:                   # the rust arms against the SERIAL keyed one (the overlap / sampling reads)
        base = "rust_serial_keyed"
        if x == base or base not in arms:
            continue
        ok = [k for k in pairs if (k, base) in by and (k, x) in by]
        if ok:
            ratios[f"decisions_per_s_{x}_over_{base}"] = ratio_ci(
                [by[(k, x)]["decisions_per_s"] for k in ok], [by[(k, base)]["decisions_per_s"] for k in ok], seed=seed)
    return per, ratios


#: The owner's five components (2026-09-30) over the collector's timers. ``forward`` is ONE T2 flush
#: serving the trainee AND every opponent slot (the split between them is the fan-out read's, not
#: this timer's); ``gpu_wait`` is the host blocking on that flush's completion — F-LE-8's 5.1 ms was
#: this, billed to sampling; ``sampling`` is the draws alone; ``host_glue`` is the named host work plus
#: the RESIDUAL (the step's wall minus every timer), so the five always sum to ms/step.
_OWNER_GROUPS = {"core_step": ("core",), "forward_launch_and_flush": ("submit", "flush"),
                 "gpu_wait": ("gpu_wait",), "sampling": ("draw", "opp_draw"),
                 "host_glue": ("write", "post", "fill")}


def owner_breakdown(comp: Mapping[str, float], ms_per_step: float) -> Dict[str, float]:
    """``comp`` (ms per vec step, one arm's timers) folded into the owner's five components."""
    if "trainee_forward" in comp:                    # today's path: the forward vs everything the workers do
        return {"trainee_forward": float(comp["trainee_forward"]),
                "vec_step_env_and_opponents": float(comp.get("vec_step_env_and_opponents", 0.0)),
                "host_glue": float(ms_per_step - comp["trainee_forward"]
                                   - comp.get("vec_step_env_and_opponents", 0.0))}
    out = {g: float(sum(comp.get(k, 0.0) for k in ks)) for g, ks in _OWNER_GROUPS.items()}
    if "core_wait_unhidden" in comp:
        out["core_step"] = float(comp["core_wait_unhidden"])     # overlapped: only the part NOT hidden
        out["core_step_total_both_halves"] = float(comp.get("core", 0.0))
    named = sum(v for k, v in out.items() if k != "core_step_total_both_halves")
    out["host_glue"] += float(ms_per_step - named)
    return out


def _mean_ci(xs: Sequence[float], *, seed: int = 0, n_boot: int = 10_000) -> Dict[str, Any]:
    import numpy as np

    x = np.asarray(xs, dtype=np.float64)
    if x.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    if x.size < 2:
        return {"mean": float(x.mean()), "lo": None, "hi": None, "n": int(x.size)}
    boot = x[np.random.default_rng(seed).integers(0, x.size, (n_boot, x.size))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(x.size),
            "method": "percentile bootstrap of the block mean"}


# ---------------------------------------------------------------------------------------------
# the regime
# ---------------------------------------------------------------------------------------------

def _git() -> Dict[str, Any]:
    from utils.paths import repo_root

    def run(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=str(repo_root()), capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception as exc:
            return f"unknown ({type(exc).__name__})"

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len([x for x in run("status", "--porcelain").splitlines() if x.strip()])}


def _gpu_processes() -> Any:
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        return [line.strip() for line in q.stdout.splitlines() if line.strip()]
    except Exception as exc:
        return f"unknown ({type(exc).__name__})"


def _load() -> Dict[str, Any]:
    from utils.contention import cpu_contention_factor

    one, five, fifteen = os.getloadavg()
    return {"load1": one, "load5": five, "load15": fifteen, "contention_factor": cpu_contention_factor(refresh=True)}


def build_hooks(a: argparse.Namespace) -> Tuple[Any, Any, Dict[str, Any]]:
    """The inference and opponent hooks (a NOT-BUILT name raises ``HookNotBuilt`` here) and one
    collector per arm."""
    if a.inference == "t2":
        inference: Any = H.T2Inference(a.ckpt, backend=a.t2_backend, device=a.device,
                                       buckets=tuple(int(b) for b in a.buckets.split(",")),
                                       max_rows_per_flush=max(1024, a.n_envs))
    elif a.inference == "learner":
        inference = H.LearnerSampling(a.ckpt, device=a.device, compile=a.compile_trainee)
    else:
        inference = H.RandomLegal(a.seed)
    if a.opponent == "production":
        opponent: Any = H.ProductionMix(a.pool, a.pool_size, a.self_play_fraction, a.compile_opponents)
    else:
        opponent = H.OPPONENTS[a.opponent](seed=a.seed)
    if a.collector == "complete_game":
        # today's path has no complete-game collector: its arm keeps today's window
        collectors = {"python": H.StepCollector(a.n_steps)}
        for name in PRODUCTION_ARMS[1:] + ("rust",):
            collectors[name] = H.CompleteGameCollector(a.target)
    else:
        collectors = {arm: H.COLLECTORS[a.collector](a.n_steps) for arm in ARMS}
    return inference, opponent, collectors


def parse_rust_arm(name: str) -> Optional[Tuple[str, str, Optional[int]]]:
    """``rust_<serial|overlap>_<keyed|generator>[_p<K>]`` -> ``(shape, sampling, K or None)``, else None.
    ``_p<K>`` is the FEWER-SNAPSHOTS candidate (F-LE-8 follow-up): the same pool, only K snapshots
    routed to (so at most K distinct opponent slots in any flush)."""
    import re

    m = re.fullmatch(r"rust_(serial|overlap)_(keyed|generator)(?:_p([1-9][0-9]*))?", name)
    return (m.group(1), m.group(2), int(m.group(3)) if m.group(3) else None) if m else None


def make_arm(name: str, a: argparse.Namespace, inference: Any, opponent: Any, collector: Any) -> Any:
    """The arm for ``name``: the PRODUCTION arms (Lane G, ``production.py``) under ``--opponent production``
    (which needs ``--collector complete_game`` and ``--inference learner``), else Lane J's env-only arms."""
    if a.opponent == "production":
        if a.collector != "complete_game" or a.inference != "learner":
            raise SystemExit("--opponent production needs --collector complete_game and --inference learner "
                             "(both arms as training runs them)")
        from main.rust_core_m5.production import CollectorRustArm, OverlappedRustArm, ProductionPythonArm

        if name == "python":
            arm: Any = ProductionPythonArm(a.n_envs, inference, opponent, collector, a.profile)
        else:
            parsed = parse_rust_arm(name)
            if parsed is None:
                raise SystemExit(f"unknown production arm {name!r}")
            shape, sampling, active = parsed
            cls = OverlappedRustArm if shape == "overlap" else CollectorRustArm
            arm = cls(a.n_envs, a.threads, a.front, a.profile, inference, opponent, collector, a.seed,
                      sampling=sampling, device=a.device, name=name, active_snapshots=active,
                      backend=(a.t2_backend if a.t2_backend != "eager" or not str(a.device).startswith("cuda")
                               else "graph"),
                      buckets=tuple(int(b) for b in a.t2_buckets.split(",")) if a.t2_buckets else ())
        arm.watch_gpu = str(a.device).startswith("cuda")
        return arm
    if a.collector == "complete_game":
        raise SystemExit("--collector complete_game runs on the production arms: add --opponent production "
                         "--inference learner")
    return (PythonArm(a.n_envs, inference, opponent, collector, a.profile) if name == "python"
            else RustArm(a.n_envs, a.threads, a.front, a.profile, inference, opponent, collector, a.seed))


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    out = refuse_models_out(a.out)
    reexec = flock_reexec_argv(argv, os.environ, sys.executable)
    if reexec is not None:
        env = dict(os.environ, **{GPU_LOCK_MARKER: "1"})
        print(f"[throughput] CUDA requested — re-exec under {GPU_LOCK}", file=sys.stderr, flush=True)
        os.execvpe(reexec[0], reexec, env)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")            # training's worker setting (train_rl_agent.py)
    arms = tuple(x.strip() for x in a.arms.split(",") if x.strip())
    allowed = PRODUCTION_ARMS if a.opponent == "production" else ARMS
    if a.opponent == "production" and a.arms == "python,rust":
        arms = ("python", "rust_serial_keyed")
    bad = [x for x in arms if x not in allowed and not (a.opponent == "production" and parse_rust_arm(x))]
    if not arms or bad or len(set(arms)) != len(arms):
        raise SystemExit(f"--arms must be a subset of {allowed} (production: plus rust_<serial|overlap>_"
                         f"<keyed|generator>_p<K>, K active snapshots), got {a.arms!r}")
    from utils.contention import warn_if_contended

    warnings: List[str] = []
    load_start = _load()
    if warn_if_contended("m5 throughput A/B"):
        warnings.append(f"BUSY BOX at start: contention factor {load_start['contention_factor']:.2f} "
                        "(warn, never stretch — a same-load A/B only)")
    prod = (a.opponent == "production", a.collector == "complete_game", a.inference == "learner")
    if any(prod[:2]) and not all(prod):
        print("[throughput] the PRODUCTION arms need all three: --opponent production --collector complete_game "
              "--inference learner (both arms as training runs them)", file=sys.stderr)
        return 2
    try:
        inference, opponent, collectors = build_hooks(a)
        for name in arms:                    # a _p<K> arm's collector (the fewer-snapshots candidate)
            if a.collector == "complete_game" and name not in collectors:
                collectors[name] = H.CompleteGameCollector(a.target)
    except (H.HookNotBuilt, ValueError, OSError) as exc:
        print(f"[throughput] {exc}", file=sys.stderr)
        return 2
    regime: Dict[str, Any] = {
        "n_envs": a.n_envs, "threads": a.threads, "front": a.front, "profile": a.profile, "arms": list(arms),
        "pairs": a.pairs, "block_seconds": a.block_seconds, "min_steps": a.min_steps, "warmup_steps": a.warmup_steps,
        "inference": inference.describe(), "opponent": opponent.describe(),
        "collector": {"name": collectors[arms[0]].name, "n_steps": a.n_steps,
                      "per_arm": {arm: collectors[arm].name for arm in arms}},
        "target": a.target if a.collector == "complete_game" else None, "seed": a.seed,
        "cpu_count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)), "nice": os.nice(0),
        "git": _git(), "python": sys.version.split()[0], "device": a.device,
        "gpu_processes": _gpu_processes() if str(a.device).startswith("cuda") else None,
        "gpu_lock": GPU_LOCK if os.environ.get(GPU_LOCK_MARKER) == "1" else None,
        "load_start": load_start, "schedule": "each arm built ONCE, then interleaved timed blocks (A B, B A, ...)",
    }
    built: Dict[str, Any] = {}
    startup: Dict[str, float] = {}
    blocks: List[Dict[str, Any]] = []
    try:
        for name in arms:
            t0 = time.perf_counter()
            arm: Any = make_arm(name, a, inference, opponent, collectors[name])
            built[name] = arm
            arm.build()
            startup[name] = time.perf_counter() - t0
            for _ in range(a.warmup_steps):
                arm.step()
            print(f"[throughput] {name} arm up in {startup[name]:.1f} s (+{a.warmup_steps} warm-up steps)",
                  file=sys.stderr, flush=True)
        waited = 0.0
        if a.wait_while_exists:
            t0 = time.perf_counter()
            while os.path.exists(a.wait_while_exists):
                time.sleep(5)
            waited = time.perf_counter() - t0
        regime["waited_for_busy_flag"] = {"path": a.wait_while_exists, "seconds": round(waited, 1)}
        regime["load_before_blocks"] = _load()
        for k, name in schedule(a.pairs, arms):
            b = run_block(built[name], seconds=a.block_seconds, min_steps=a.min_steps)
            b.update(pair=k, load1_after=os.getloadavg()[0])
            blocks.append(b)
            print(f"[throughput] pair {k} {name}: {b['decisions_per_s']:.0f} decisions/s, "
                  f"{b['ms_per_vec_step']:.2f} ms/step, {b['cpu_us_per_decision'] or 0:.0f} CPU-us/decision",
                  file=sys.stderr, flush=True)
    finally:
        for arm in built.values():
            try:
                arm.close()
            except Exception as exc:
                warnings.append(f"closing the {arm.name} arm: {type(exc).__name__}: {exc}")
    per, ratios = summarize(blocks, arms, a.seed)
    regime["load_end"] = _load()
    if regime["load_end"]["contention_factor"] >= 1.25:
        warnings.append(f"BUSY BOX at end: contention factor {regime['load_end']['contention_factor']:.2f}")
    refused = sum(int(getattr(x, "refused", 0) or 0) for n_, x in built.items() if n_ != "python")
    if refused:
        warnings.append(f"the rust arm QUARANTINED {refused} battle(s) during the run")
    res = {"schema": SCHEMA, "verdict": "DESCRIPTOR — no throughput bar is registered for M5 (program doc §2 M5)",
           "regime": regime, "startup_s": startup,
           "arm_info": {n: getattr(x, "info", {}) for n, x in built.items()},
           "collectors": {n: collectors[n].describe() for n in arms}, "blocks": blocks, "arms": per,
           "ratios": ratios, "warnings": warnings}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    for name, s in per.items():
        print(f"[throughput] {name}: {s['decisions_per_s_mean']:.0f} decisions/s, "
              f"{s['cpu_us_per_decision_mean'] or math.nan:.0f} CPU-us/decision, startup {startup.get(name, 0):.1f} s")
    for key, r in ratios.items():
        print(f"[throughput] {key}: {r['point']:.3f} [{r.get('lo')}, {r.get('hi')}]")
    print(f"[throughput] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
