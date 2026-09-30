"""The throughput A/B at the PRODUCTION SHAPE — both arms as TRAINING runs them (M5 Lane G's hooks).

``hooks.ProductionMix`` + ``hooks.CompleteGameCollector`` + ``hooks.LearnerSampling`` select these arms
(``throughput.build_arms``). Both are built from ONE resolved args namespace (``production_args()``)
and ONE snapshot pool (the last K snapshots of a real run, symlinked into a temp dir — nothing is
written under the source):

* **``ProductionPythonArm`` — today's path.** N ``SubprocVecEnv`` workers, each built by
  ``env_factory.create_training_env_random`` itself (the production-surface ``Gen3Env`` on the rust
  bridge with the core obs, ``MaskableAgentWrapper`` with the 8-bot floor roster and the self-play pool
  at the declared fraction, the pool opponent an ``RLPlayer`` on CPU — compiled when
  ``--compile-opponents`` is on, as production), and the trainee's action from
  :class:`hooks.LearnerSampling` — the learner policy's own sampling forward on the device, the call
  ``collect_rollouts`` makes. Its collector is today's fixed window (``StepCollector``).
* **``CollectorRustArm`` — the new path.** ``build_collector`` (Lane G) over the same pool and args: one
  env core (process front end), the trainee (the SAME checkpoint) and every policy opponent through T2
  in one flush, the bots in the core, the keyed draw, and the COMPLETE-GAME collector — every host
  step is ``RustCollector.host_step`` and, when the target is reached, the fill into a learner-shaped
  buffer is paid inside the timed block (a real rollout's cost; nothing trains).

What each arm's ``step()`` reports: trainee decisions, the seconds inside inference (python: the
learner forward; rust: T2 submit + flush + the keyed draw), the seconds in the env (python: the vec
step; rust: the core STEP), and episodes ended.
"""
from __future__ import annotations

import os
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def pool_dir_from(source: str, k: int) -> Tuple[Path, List[str]]:
    """A temp pool dir holding symlinks to the last ``k`` snapshots of ``source`` + its model_config."""
    src = Path(source)
    zips = sorted(src.glob("snapshot_*.zip"))[-int(k):]
    if not zips:
        raise FileNotFoundError(f"{src} holds no snapshot_*.zip")
    wd = Path(tempfile.mkdtemp(prefix="m5_throughput_pool_"))
    for z in zips:
        (wd / z.name).symlink_to(z)
    (wd / "model_config.json").symlink_to(src / "model_config.json")
    return wd, [z.name for z in zips]


def load_trainee(ckpt: str, device: str) -> Any:
    """The checkpoint's MODEL on ``device`` (its policy is the trainee on both arms)."""
    from agents.model.model_version import ModelVersion
    from agents.model.snapshot import load_foreign_opponent

    p = Path(ckpt)
    cfg = p.parent / "model_config.json"
    if not cfg.is_file():
        cfg = p.parent.parent / "model_config.json"
    ver = ModelVersion.from_json_file(str(cfg))
    model, _ = load_foreign_opponent(str(p), current_version=ver, device=device, config_path=str(cfg))
    return model


def training_bot_classes() -> List[Any]:
    import importlib

    from utils.rust_env import bot_inventory as BI

    out = []
    for row in BI.ROWS:
        if "train" in row.used_by:
            mod, _, name = row.cls.rpartition(".")
            out.append(getattr(importlib.import_module(mod), name))
    return out


class ProductionPythonArm:
    """Today's path (module docs)."""

    name = "python"

    def __init__(self, n: int, inference: Any, mix: Any, collector: Any, profile: str):
        self.n, self.inference, self.mix, self.collector, self.profile = n, inference, mix, collector, profile
        self.vec: Any = None
        self.info: Dict[str, Any] = {}
        self._inf_s = self._env_s = 0.0

    def build(self) -> None:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        from agents.model.extractor_arch import build_extractor_arch_kwargs
        from agents.model.model_version import ModelVersion
        from agents.observation.state_encoder import load_mappings
        from agents.training.reward_config import RewardConfig
        from agents.training.reward_manager import Gen3RewardManager
        from agents.training.stall import StallConfig
        from main.rust_core_m5.throughput import sim_bridge_path
        from main.train.env_factory import create_training_env_random
        from utils.logging.levels import LogLevel
        from utils.team_sources import team_list
        from utils.teambuilder import Gen3Teambuilder

        args = self.mix.args
        os.environ["POKESIM_SIM_BRIDGE_BIN"] = str(sim_bridge_path(self.profile))
        mappings = load_mappings()
        if args.compile_opponents:
            from agents.model.compile_opponents import arm_compile_quorum
            from agents.model.compile_prewarm import prewarm_extractor_compile

            arm_compile_quorum(None)
            prewarm_extractor_compile(build_extractor_arch_kwargs(args), mappings, quiet=True)
        tb = Gen3Teambuilder(team_list("pool"))
        ver = ModelVersion.from_json_file(str(self.mix.pool_dir / "model_config.json"))
        fns = [create_training_env_random(
            i, stall_config=StallConfig(), opponent_device="cpu", opponent_version=ver,
            snapshot_dir=str(self.mix.pool_dir), self_play_fraction=float(self.mix.self_play_fraction),
            self_play=True, heuristic_weights=None, stable_opponents=None, exploiter_entry=None,
            cf_records_dir=None, args=args, mappings=mappings, log_level=LogLevel.QUIET,
            trainee_teambuilder=tb, opponent_teambuilder=tb, server_config=None,
            OPPONENT_CLASSES=training_bot_classes(),
            reward_factory=partial(Gen3RewardManager, config=RewardConfig.from_args(args)))
            for i in range(self.n)]
        self.vec = SubprocVecEnv(fns, start_method="forkserver")
        obs = self.vec.reset()
        self._obs, self._mask = obs["observation"], obs["action_mask"]
        self.stamp = {"mix": {"self_play_fraction": self.mix.self_play_fraction,
                              "bot_share": round(1.0 - self.mix.self_play_fraction, 4),
                              "pool_size": len(self.mix.snapshots)},
                      "opponents": "per-worker RLPlayer on CPU (" + ("compiled" if args.compile_opponents else "eager")
                                   + "), sampled by torch.multinomial", "trainee": "the learner's sampling forward"}
        self.info = {"vec_env": "SubprocVecEnv(forkserver)", "worker": "env_factory.create_training_env_random",
                     "pool": self.mix.describe(), "compile_opponents": bool(args.compile_opponents),
                     "collector": "fixed window (today's)", **self.stamp, "worker_pids": self.roots()}

    def components(self) -> Dict[str, float]:
        return {"trainee_forward": self._inf_s, "vec_step_env_and_opponents": self._env_s}

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
        self._inf_s += t1 - t0
        self._env_s += t2 - t1
        return int(rows.size), t1 - t0, t2 - t1, int(np.count_nonzero(dones))

    def close(self) -> None:
        if self.vec is not None:
            try:
                self.vec.close()
            except OSError as exc:        # a battle still running at teardown (the J harness's rule)
                if "still running" not in str(exc):
                    raise
            self.vec = None


def _build_rust_collector(arm: Any, n: int) -> Any:
    """One ``RustCollector`` of ``n`` envs over the mix's pool and args, sharing the mix's T2 service when
    an earlier arm started it (every rust arm of one A/B reads the SAME slots / buckets / lanes)."""
    from agents.model.model_version import ModelVersion
    from agents.training import rust_env_opponents as E
    from agents.training.rust_rollout.build import OpponentSources, RustEnvDecl, build_collector, trainee_spaces
    from agents.training.snapshot_pool import SnapshotPool
    from main.train.rust_env_setup import _bot_names
    from utils.rust_env import episode as EP
    from utils.team_sources import team_list
    from utils.teambuilder import Gen3Teambuilder

    mix = arm.mix
    args = mix.args
    obs_space, act_space = trainee_spaces(args)
    target = int(arm.collector.target)
    per = max(n, target * n // arm.n // n * n)                    # the halves' share of the target
    mix._n_collectors = getattr(mix, "_n_collectors", 0) + 1        # a distinct keyed seed per collector
    decl = RustEnvDecl(
        n_envs=n, threads=arm.threads, front=arm.front, profile=arm.profile, trigger="complete_game",
        n_steps=per // n, micro_batch=per // n, target=per, gamma=1.0, gae_lambda=0.8,
        run_seed=arm.seed + 1000 * mix._n_collectors,
        turn_limit=EP.stall_threshold(), device=arm.device, backend=arm.backend, buckets=arm.buckets,
        lanes=int(getattr(arm, "lanes", 0) or 0), opponent_sampling=arm.sampling, policy_seed=arm.seed)
    ver = ModelVersion.from_json_file(str(mix.pool_dir / "model_config.json"))
    active = int(getattr(arm, "active_snapshots", None) or len(mix.snapshots))
    if active > len(mix.snapshots):
        raise ValueError(f"{arm.name}: {active} active snapshots from a pool of {len(mix.snapshots)}")
    # FEWER ACTIVE SNAPSHOTS (``_p<K>``): the plan (so the slot layout and the shared service) is the full
    # pool's; the arm's POOL holds only the mix's K newest snapshots (its own temp dir of symlinks —
    # ``max_snapshots`` alone would not do it: SnapshotPool applies its window only when ADDING), so every
    # env routes to one of <= K slots and a flush meets at most K distinct pool slots
    pool_dir = mix.pool_dir if active == len(mix.snapshots) else pool_dir_from(str(mix.pool_dir), active)[0]
    pool = SnapshotPool(pool_dir, current_version=ver, device=arm.device, max_snapshots=len(mix.snapshots),
                        lru_cache_size=active + 4)
    if len(pool.steps()) != active:
        raise RuntimeError(f"{arm.name}: the pool holds {len(pool.steps())} snapshots, declared {active} active")
    plan = E.OpponentPlan.from_args(args, bot_names=_bot_names(training_bot_classes()),
                                    max_snapshots=len(mix.snapshots))
    tb = Gen3Teambuilder(team_list("pool"))
    # the trainee template is the arm's OWN load of the checkpoint: the python arm's policy may carry
    # --compile-trainer's patched forward, which T2 must never deep-copy (build_collector refuses it)
    if getattr(mix, "rust_trainee", None) is None:
        mix.rust_trainee = load_trainee(arm.inference.ckpt, arm.device).policy.eval()
    col = build_collector(decl, obs_space=obs_space, trainee_policy=mix.rust_trainee, plan=plan,
                          sources=OpponentSources(pool=pool, self_play_fraction=mix.self_play_fraction),
                          trainee_builder=tb, opponent_builder=tb, emit=lambda _m: None,
                          svc=getattr(mix, "rust_svc", None))
    mix.rust_svc = col.svc
    col.start()
    return col, obs_space, act_space, decl, plan


def _stamp(arm: Any, decl: Any, plan: Any, svc: Any) -> Dict[str, Any]:
    return {"mix": {"self_play_fraction": arm.mix.self_play_fraction,
                    "bot_share": round(1.0 - arm.mix.self_play_fraction, 4),
                    "pool_size": len(arm.mix.snapshots), "pool_slots": plan.pool_slots,
                    "active_snapshots": int(getattr(arm, "active_snapshots", None) or len(arm.mix.snapshots))},
            "t2": {"backend": decl.backend, "device": decl.device,
                   # the SERVICE's buckets: a shared service keeps the first arm's (an overlapped half's
                   # declaration would say (8, N / 2) while it is served by (8, N))
                   "buckets": [int(b) for b in (getattr(getattr(svc, "engine", None), "buckets", None)
                                                or decl.resolved_buckets)],
                   "slots": len(getattr(svc, "_slots", ())), "lanes": getattr(getattr(svc, "engine", None), "n_lanes", None)},
            "opponent_sampling": arm.sampling, "front": arm.front, "threads": arm.threads}


class CollectorRustArm:
    """The new path, SERIAL (module docs): one collector, one host step = prepare → core STEP → finish.
    ``sampling`` is p2's draw (``keyed`` / ``generator``, F-LE-8's before / after)."""

    def __init__(self, n: int, threads: int, front: str, profile: str, inference: Any, mix: Any, collector: Any,
                 seed: int, *, sampling: str = "keyed", device: str = "cuda", backend: str = "graph",
                 buckets: Sequence[int] = (), name: str = "rust", active_snapshots: Optional[int] = None):
        self.n, self.threads, self.front, self.profile = n, threads, front, profile
        self.active_snapshots = active_snapshots
        self.inference, self.mix, self.collector, self.seed = inference, mix, collector, seed
        self.sampling, self.device, self.backend, self.buckets = sampling, device, backend, tuple(buckets)
        self.name = name
        self.col: Any = None
        self.info: Dict[str, Any] = {}
        self.refused = 0
        self.fills = 0

    def build(self) -> None:
        from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer

        self.col, obs_space, act_space, decl, plan = _build_rust_collector(self, self.n)
        self._buf = MaskableDictRolloutBuffer(decl.target // self.n, obs_space, act_space, device="cpu",
                                              gamma=1.0, gae_lambda=0.8, n_envs=self.n)
        self._target = decl.target
        self.stamp = _stamp(self, decl, plan, self.col.svc)
        self.info = {"collector": self.col.cfg.trigger.describe(), **self.stamp, "routes": len(plan.routes()),
                     "overlap": False, "proc_pid": self.roots()}

    def roots(self) -> List[int]:
        pid = getattr(getattr(self.col, "core", None), "pid", None)
        return [int(pid)] if pid else []

    def components(self) -> Dict[str, float]:
        return _components([self.col])

    def _fill(self, col: Any, buf: Any, target: int) -> None:
        from agents.training.rust_rollout import store as S

        t = time.perf_counter()
        S.fill_complete(buf, col.log, int(target), current_version=col.version)
        col.stats.seconds["fill"] += time.perf_counter() - t
        self.fills += 1
        self.collector.observe_fill()

    def step(self) -> Tuple[int, float, float, int]:
        sec = self.col.stats.seconds
        before = (_inf(sec), sec["core"], self.col.stats.games_ended)
        k = self.col.host_step()
        dones = self.col.stats.games_ended - before[2]
        if self.col.ready():
            self._fill(self.col, self._buf, self._target)
        self.collector.observe_step(dones)
        return k, _inf(sec) - before[0], sec["core"] - before[1], dones

    def close(self) -> None:
        if self.col is not None:
            self.info["after_freeze"] = self.col.check_lifecycle()
            self.info["stats"] = {k: v for k, v in vars(self.col.stats).items() if k != "seconds"}
            self.info["seconds"] = dict(self.col.stats.seconds)
            self.info["fills"] = self.fills
            ss = getattr(self.col.server, "stats", None)
            if ss is not None and getattr(ss, "submits", 0):
                self.info["opponent_slots_per_flush_mean"] = ss.slot_submits / ss.submits
            self.refused = int(self.col.stats.quarantines)
            self.col.close()
            self.col = None


class OverlappedRustArm(CollectorRustArm):
    """The new path, OVERLAPPED (the lever J measured: inference and the env step in series): the envs
    split into two HALVES, each its own core (N / 2 envs) over the SAME T2 service; while half A's core
    STEPs on a worker thread (the GIL is released — the child runs the battles), the main thread runs
    half B's T2 flush, the wait for it and its draws; then the roles swap. Each half is a full collector
    (its own arena and complete-game FIFO, half the target) — a throughput arm, not yet a training path."""

    def build(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer

        if self.n % 2:
            raise ValueError("the overlapped arm needs an even --n-envs")
        h = self.n // 2
        self.cols_: List[Any] = []
        self._bufs: List[Any] = []
        for _ in range(2):
            col, obs_space, act_space, decl, plan = _build_rust_collector(self, h)
            self.cols_.append(col)
            self._bufs.append(MaskableDictRolloutBuffer(decl.target // h, obs_space, act_space, device="cpu",
                                                        gamma=1.0, gae_lambda=0.8, n_envs=h))
        self._target = decl.target
        self.col = self.cols_[0]
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="core_step")
        self._primed = False
        self.stamp = _stamp(self, decl, plan, self.col.svc)
        self.info = {"collector": f"2 halves x {self.cols_[0].cfg.trigger.describe()}", **self.stamp,
                     "overlap": True, "halves": 2, "proc_pid": self.roots()}
        self._core_wait = 0.0

    def roots(self) -> List[int]:
        out = []
        for c in getattr(self, "cols_", []):
            pid = getattr(getattr(c, "core", None), "pid", None)
            if pid:
                out.append(int(pid))
        return out

    def components(self) -> Dict[str, float]:
        out = _components(self.cols_)
        out["core_wait_unhidden"] = self._core_wait
        return out

    def step(self) -> Tuple[int, float, float, int]:
        a, b = self.cols_
        before = (sum(_inf(c.stats.seconds) for c in self.cols_), self._core_wait,
                  sum(c.stats.games_ended for c in self.cols_))
        k = 0
        if not self._primed:
            a.prepare()
            self._primed = True
        for x, y in ((a, b), (b, a)):           # x's core steps while y's inference runs; then swap
            k += int(getattr(x, "_prepared", 0))
            fut = self._ex.submit(x.step_core)
            y_k = y.prepare()
            tw = time.perf_counter()
            fut.result()
            self._core_wait += time.perf_counter() - tw
            x.finish()
            del y_k
        for i, c in enumerate(self.cols_):
            if c.ready():
                self._fill(c, self._bufs[i], self._target)
        dones = sum(c.stats.games_ended for c in self.cols_) - before[2]
        self.collector.observe_step(dones)
        return (k, sum(_inf(c.stats.seconds) for c in self.cols_) - before[0], self._core_wait - before[1], dones)

    def close(self) -> None:
        if getattr(self, "_ex", None) is not None:
            self._ex.shutdown(wait=True)
        for i, c in enumerate(getattr(self, "cols_", [])):
            self.info[f"half{i}_after_freeze"] = c.check_lifecycle()
            self.info[f"half{i}_seconds"] = dict(c.stats.seconds)
            ss = getattr(c.server, "stats", None)
            if ss is not None and getattr(ss, "submits", 0):
                self.info[f"half{i}_opponent_slots_per_flush_mean"] = ss.slot_submits / ss.submits
            self.refused += int(c.stats.quarantines)
            c.close()
        self.cols_ = []
        self.col = None


def _inf(sec: Dict[str, float]) -> float:
    """The inference share of a collector's host step: submits, the flush call, the GPU wait, both draws."""
    return sec["submit"] + sec["flush"] + sec["gpu_wait"] + sec["opp_draw"] + sec["draw"]


def _components(cols: Sequence[Any]) -> Dict[str, float]:
    """Cumulative seconds per component over ``cols`` (the A/B takes each block's delta)."""
    keys = ("submit", "flush", "gpu_wait", "opp_draw", "draw", "write", "core", "post", "fill")
    return {k: float(sum(c.stats.seconds[k] for c in cols)) for k in keys}
