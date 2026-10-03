"""`OwnedLoop` — THE PPO LOOP IS OURS (`gen3_owned_ppo_loop_v1`; `designs/endstate/design_own_ppo_loop.md` stages 1-3).

Before stage 1, `InstrumentedMaskablePPO.train()` was ours (vendored, hash-pinned) but the loop AROUND
it was sb3's: `MaskablePPO.learn`, `BaseAlgorithm._setup_learn`, `OnPolicyAlgorithm.dump_logs`,
`_update_current_progress_remaining` and `_update_info_buffer`. They are vendored here, each one
operation for operation (stage 1 was IDENTITY, proved against upstream by a lockstep differential and a
real-run A/B). Stage 3 (deletion pass U4) took the rest of sb3's RUNTIME off the loop: the rollout
buffer (`agents/training/rollout_buffer.py`), the logger (`agents/training/train_logger.py`), the
callback protocol (`agents/training/loop_callbacks.py`) and the env base (`agents/training/trainer_env.py`)
are ours, and the `GEN3AI_PPO_LOOP=sb3_reference` test seam that ran upstream's `learn` is deleted with
nothing left to compare against. What is still sb3 (stage 4, a separate owner decision): the constructor
and `_setup_model` (which this mixin extends), the policy base classes and their init, and the `.zip`.

**The loop, as a declared table (`LOOP_PHASES`).** One `learn()` call is

    setup -> training_start -> [collect -> progress -> dump -> update]* -> training_end -> final_dump

and nothing else. `collect` includes the post-collect window (the fork arm and the value sidecar in
their `on_rollout_end`), unchanged.

Four properties are CONTRACTS, each load-bearing for something outside this file:

* **`dump` comes BEFORE `update`.** `train()`'s `train/*` scalars sit in the logger until the NEXT
  iteration's dump, so update k's statistics are stamped at the step after rollout k+1. Every
  TensorBoard series in the archive has that convention. Moving the dump after the update would shift
  every `train/*` series by one rollout. (The eval callbacks' mid-rollout dump holds the update's
  scalars aside — `logger_scope.isolated_dump`, stage 2.)
* **`final_dump` writes the LAST update's scalars** (P3, `gen3_final_update_dump_v1`): with the dump
  before each update, the last `train()`'s `train/*` were still pending when `learn()` returned and were
  never written — one update per process, so one per launcher restart. After `training_end` whatever is
  pending is dumped at the current `num_timesteps`. On a normal end that is the step the last iteration's
  dump already used, so the `train/*` tags carry TWO points at the final step (update k-1's, then update
  k's, in that order in the event file); on a stop from inside a collection (the graceful restart) it is
  a later step.
* **`collect` and `update` are called THROUGH ATTRIBUTE LOOKUP** (`self.collect_rollouts`,
  `self.train`), inside their DECLARED HOOK POINTS (`loop_hooks.HOOK_POINTS`: learn > collect | update).
  K6's freeze guard and the compile sentinel register there; a tool that wraps the methods as instance
  attributes (`learner_benchmark`) still intercepts.
* **The rollout is the Rust collector's** (`RolloutProbes.collect_rollouts` → `_collect_rust`), the
  only env core; the loop calls it through the attribute and never steps an env itself. The env the
  learner holds is a `TrainerVecEnv` (`_wrap_env` refuses anything else).

**Drift.** Each vendored upstream source is hash-pinned (`UPSTREAM_SOURCE_SHA256`) and checked at
import, like `train()`'s pin in the hub: the sb3 constructor that still builds the learner sets the
attributes these methods read, so an sb3 upgrade that changes one of them is a loud error naming the
method, not a silent fork.
"""
from __future__ import annotations

import hashlib
import inspect
import random
import sys
import time
from collections import deque
from typing import Any, Dict, Tuple

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common import utils as _sb3_utils
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from agents.training.loop_callbacks import init_callback
from agents.training.loop_hooks import LoopHooks
from agents.training.rollout_buffer import RolloutBuffer
from agents.training.train_logger import configure
from agents.training.trainer_env import TrainerVecEnv

#: The loop, in order. `learn()` below is this table written out; `owned_loop_test` pins both.
LOOP_PHASES: Tuple[str, ...] = (
    "setup",            # _setup_learn: counters, ep-info buffers, env reset if needed, logger, callback init
    "training_start",   # callback.on_training_start()
    "collect",          # self.collect_rollouts(...) — on_rollout_start, on_step x n, on_rollout_end
    "progress",         # iteration += 1; _current_progress_remaining
    "dump",             # dump_logs(iteration) — BEFORE the update (the TB step contract)
    "update",           # self.train()
    "training_end",     # callback.on_training_end()
    "final_dump",       # the last update's pending scalars, at num_timesteps (P3; module docstring)
)
#: The phases that repeat, once per iteration, in this order.
LOOP_ITERATION: Tuple[str, ...] = ("collect", "progress", "dump", "update")

#: sha256 of `inspect.getsource(...)` of every upstream method vendored here (sb3 / sb3-contrib 2.8.0).
UPSTREAM_SOURCE_SHA256: Dict[str, str] = {
    "MaskablePPO.learn": "ad8b3b8f3935053cca795d1f77b7faa1d26430e231050bf5d60454ed7106c18e",
    "BaseAlgorithm._setup_learn": "1c5a903f4b48a0fb90050520408fc9affc400195ee862745b2169d014cc553ce",
    "OnPolicyAlgorithm.dump_logs": "22e7a7c5dd2a01858e567d878765ed6e43da9da0570a800652e322e2e0912fb8",
    "BaseAlgorithm._update_current_progress_remaining":
        "3ffbde55601336c514e501cbfe93702adbd629109faccb5dc62e68f277bf17b2",
    "BaseAlgorithm._update_info_buffer": "f3aeefb5ce5b17e139f8e907af6a60d26f68035a790c7b126e3df0177beb80f9",
    "BaseAlgorithm.set_random_seed": "0e7d5cb60fca15ab4919fab79381a7455f783bfd2977890dcf063d4bf57d63c7",
    "utils.set_random_seed": "72309e17cff74c17ec9184d90a6661364724c169b612904b36e63614b4f3f23b",
}


def _upstream_sources() -> Dict[str, Any]:
    return {
        "MaskablePPO.learn": MaskablePPO.learn,
        "BaseAlgorithm._setup_learn": BaseAlgorithm._setup_learn,
        "OnPolicyAlgorithm.dump_logs": OnPolicyAlgorithm.dump_logs,
        "BaseAlgorithm._update_current_progress_remaining": BaseAlgorithm._update_current_progress_remaining,
        "BaseAlgorithm._update_info_buffer": BaseAlgorithm._update_info_buffer,
        "BaseAlgorithm.set_random_seed": BaseAlgorithm.set_random_seed,
        "utils.set_random_seed": _sb3_utils.set_random_seed,
    }


def verify_upstream_loop_unchanged() -> None:
    """Fail loud if any upstream method this module vendors has changed since it was vendored."""
    drift = []
    for name, fn in _upstream_sources().items():
        actual = hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()
        if actual != UPSTREAM_SOURCE_SHA256[name]:
            drift.append(f"  {name}: expected {UPSTREAM_SOURCE_SHA256[name]}, actual {actual}")
    if drift:
        raise RuntimeError(
            "[OwnedLoop] DRIFT DETECTED: upstream sb3 / sb3-contrib source that "
            f"{__file__} vendors has changed:\n" + "\n".join(drift) + "\n"
            "ACTION REQUIRED: diff each upstream method against its vendored copy here, port any "
            "behaviour change (or decide not to, and say why in design_own_ppo_loop.md), then update "
            "UPSTREAM_SOURCE_SHA256.")


verify_upstream_loop_unchanged()


class OwnedLoop:
    """Mixin: the PPO loop. Mixed in AFTER `RolloutProbes` and BEFORE `MaskablePPO`, so
    `RolloutProbes.collect_rollouts` (the Rust collector's entry) still wins."""

    # ------------------------------------------------------------------ learn
    def learn(  # type: ignore[override]
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        tb_log_name: str = "MaskablePPO",
        reset_num_timesteps: bool = True,
        use_masking: bool = True,
        progress_bar: bool = False,
    ) -> Any:
        """`LOOP_PHASES`, written out (vendored from `MaskablePPO.learn`, plus `final_dump`). The three
        DECLARED HOOK POINTS (`loop_hooks.HOOK_POINTS`: learn > collect | update) are opened here, in that
        nesting, with the owners in `loop_hooks.HOOK_OWNERS` order (gen3_declared_loop_hooks_v1)."""
        # the hook table is complete at training start: a later registration is a typed FATAL
        self._loop_hooks.freeze("learn() — training start")
        with self._loop_hooks.around("learn"):
            iteration = 0

            # setup
            total_timesteps, callback = self._setup_learn(
                total_timesteps,
                callback,
                reset_num_timesteps,
                tb_log_name,
                progress_bar,
            )

            # training_start
            callback.on_training_start()

            assert self.env is not None

            while self.num_timesteps < total_timesteps:
                # collect (the hook point, then the method through the attribute)
                with self._loop_hooks.around("collect"):
                    continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                              self.n_steps, use_masking)

                if not continue_training:
                    break

                # progress
                iteration += 1
                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

                # dump — BEFORE the update (module docstring)
                if log_interval is not None and iteration % log_interval == 0:
                    self.dump_logs(iteration)

                # update (the hook point, then the method through the attribute)
                with self._loop_hooks.around("update"):
                    self.train()

            # training_end
            callback.on_training_end()

            # final_dump — the last update's scalars (P3; module docstring)
            self._final_dump()

            return self

    def _final_dump(self) -> None:
        """Dump whatever is still pending — the last `train()`'s scalars and anything recorded after the
        last dump — at the current step. Nothing pending (no update since the last dump) writes nothing."""
        logger = getattr(self, "_logger", None)
        if logger is not None and len(logger.name_to_value) > 0:
            logger.dump(step=self.num_timesteps)

    # ------------------------------------------------------------------ the hook table
    def _setup_model(self) -> None:
        """sb3's model setup with OUR rollout buffer (`rollout_buffer.RolloutBuffer` — forced, so a `.zip`
        that recorded sb3-contrib's buffer class still loads onto ours), then the loop's HOOK TABLE
        (startup; `loop_hooks`). Runs on a fresh build and on `load`, so every learner object has exactly
        one, empty until its owners register."""
        self.rollout_buffer_class = RolloutBuffer
        super()._setup_model()   # type: ignore[misc]
        self._loop_hooks = LoopHooks()

    # ------------------------------------------------------------------ the env
    @staticmethod
    def _wrap_env(env: Any, verbose: int = 0, monitor_wrapper: bool = True) -> Any:
        """sb3's constructor / ``load`` / ``set_env`` hook for the env: a `TrainerVecEnv` (the Rust env, a
        test's toy env) is the learner's env AS IS; anything else REFUSES — sb3 would wrap it in a
        ``DummyVecEnv`` + ``Monitor`` that nothing here steps."""
        if not isinstance(env, TrainerVecEnv):
            raise TypeError(f"the learner's env must be a `agents.training.trainer_env.TrainerVecEnv`, not "
                            f"{type(env).__module__}.{type(env).__name__} (the Rust env is `RustVecEnv`; a test's "
                            "toy env is `rust_rollout.testkit.ToyVecEnv`)")
        return env

    # ------------------------------------------------------------------ setup
    def _setup_learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, Any]:
        """Vendored from `BaseAlgorithm._setup_learn`. 🚨 With ``reset_num_timesteps=False`` the
        target is RELATIVE (`total_timesteps += num_timesteps`): the resume site passes the REMAINING
        budget (`model_build.py`). The logger and the callbacks are OURS (`train_logger`, `loop_callbacks`)."""
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            assert self.env is not None
            self._last_obs = self.env.reset()  # type: ignore[assignment]
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # The logger, if none was set (`set_logger`; a run's is `run_io._attach_run_tb_logger`): the stdout
        # table at verbose >= 1, else the null logger. sb3's `tensorboard_log` run-id directories are not served.
        if not self._custom_logger:
            if self.tensorboard_log is not None:
                raise ValueError("tensorboard_log is not served: set the run's logger with set_logger "
                                 "(main.train.run_io._attach_run_tb_logger writes <run>/tb)")
            self._logger = configure(None, ["stdout"] if self.verbose >= 1 else [])

        # Callback init (the `init` event of `loop_callbacks`)
        if progress_bar:
            raise ValueError("progress_bar is not served by the owned loop")
        callback = init_callback(self, callback)

        return total_timesteps, callback

    # ------------------------------------------------------------------ progress / info
    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """Vendored from `BaseAlgorithm`: 1.0 at the start -> 0.0 at the end (the schedules' input)."""
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_info_buffer(self, infos: Any, dones: Any = None) -> None:
        """Vendored from `BaseAlgorithm`: the Rust collector's ``info["episode"]`` into the 100-episode
        window `dump_logs` reads."""
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    # ------------------------------------------------------------------ seeding
    def set_random_seed(self, seed: Any = None) -> None:
        """OWNED SEEDING (`gen3_owned_seeding_v1`; `design_own_ppo_loop.md` stage 2). sb3's
        `BaseAlgorithm.set_random_seed` + `utils.set_random_seed`, the SAME draws in the same order
        (python, numpy, torch, the action space, the env), MINUS one side effect: on a CUDA device
        sb3 also set the PROCESS-WIDE `torch.backends.cudnn.deterministic = True` /
        `benchmark = False` — from `_setup_model`, i.e. on every construction AND every `load`, and
        nothing restored it, so every CUDA trainer, the inference service in its process and every
        tool that loaded a checkpoint on CUDA ran under it. A seed is not a numerics policy: this
        seeds and touches no backend flag. A REGIME BOUNDARY on CUDA (CPU is unchanged — the K9
        golden's init hash with it)."""
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        self.action_space.seed(seed)
        # self.env is always a TrainerVecEnv (`_wrap_env`)
        if self.env is not None:
            self.env.seed(seed)

    # ------------------------------------------------------------------ dump
    def dump_logs(self, iteration: int = 0) -> None:
        """Vendored from `OnPolicyAlgorithm.dump_logs`. The tag names are read BY NAME by
        `main.ops.killbar` / `tb_read` / `restart_startup` / `stall_exhibit`, the launcher's
        `format.py` and `utils/plot_tb.py` — never rename one."""
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", _safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", _safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", _safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)


def _safe_mean(arr: Any) -> Any:
    """sb3's `safe_mean`, operation for operation: NaN for an empty window, else `float(np.mean)`."""
    return np.nan if len(arr) == 0 else float(np.mean(arr))
