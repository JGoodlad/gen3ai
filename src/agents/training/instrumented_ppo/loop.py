"""`OwnedLoop` — THE PPO LOOP IS OURS (`gen3_owned_ppo_loop_v1`; `designs/endstate/design_own_ppo_loop.md` stage 1).

Before this module, `InstrumentedMaskablePPO.train()` was ours (vendored, hash-pinned) but the loop
AROUND it was still sb3's: `MaskablePPO.learn`, `BaseAlgorithm._setup_learn`,
`OnPolicyAlgorithm.dump_logs`, `_update_current_progress_remaining`, `_update_info_buffer` and, on
the Python env core, `MaskablePPO.collect_rollouts`. They are vendored here, each one OPERATION FOR
OPERATION, so stage 1 is IDENTITY: the same calls in the same order, on the same objects, with the
same RNG draws. `own_ppo_loop_test` holds the owned loop against upstream `MaskablePPO.learn` (still
callable unbound while sb3 is installed) — hook trace, buffers, logger dumps and parameters, exactly.

**The loop, as a declared table (`LOOP_PHASES`).** One `learn()` call is

    setup -> training_start -> [collect -> progress -> dump -> update]* -> training_end

and nothing else. `collect` includes the post-collect window (the win-prob labels and the fork arm in
their `on_rollout_end`, PBRS / frozen-phi in `RolloutProbes.collect_rollouts`), unchanged.

Three properties are CONTRACTS, each load-bearing for something outside this file:

* **`dump` comes BEFORE `update`.** `train()`'s `train/*` scalars sit in the logger until the NEXT
  iteration's dump, so update k's statistics are stamped at the step after rollout k+1. Every
  TensorBoard series in the archive has that convention, and the KL->LR controller reads
  `train/approx_kl` from `logger.name_to_value` at that next `on_rollout_end`. Moving the dump after
  the update would shift every `train/*` series by one rollout. (The eval callbacks' mid-rollout
  `dump(step)` clears that value on eval cycles — design §2.1 — and that defect is PRESERVED here
  bit-for-bit; stage 2 fixes it as a labelled behaviour change.)
* **`collect` and `update` are called THROUGH ATTRIBUTE LOOKUP** (`self.collect_rollouts`,
  `self.train`). K6's freeze guard (`learner_lifecycle`) and the compile sentinel (`compile_control`)
  wrap them as instance attributes; a direct call would bypass both, silently.
* **The Python collect keeps sb3's LOCAL NAMES** (`infos`, `dones`, `rollout_buffer`, …), because
  callbacks read them through `self.locals` after `callback.update_locals(locals())`, and it fires
  `on_step` BEFORE `rollout_buffer.add` (`WinProbLabelCallback` reads
  `buf.pos` as the row about to be written).

**What is still sb3 after stage 1** (design §3.1): the constructor / `_setup_model`, the policy base
classes and their init, the buffer class (its `reset` / `add` / `get` / GAE), `BaseCallback` /
`CallbackList` (driven from here), the logger, the VecEnv classes and the `.zip`.

**The reference seam.** `GEN3AI_PPO_LOOP=sb3_reference` (read ONCE per `learn()` call) runs
upstream's `learn` / `collect_rollouts` instead — the A/B arm of the real-run equivalence check
(design §4 E3). It is a test seam, deleted in stage 3 with the Python core; any other value refuses.

**Drift.** Each vendored upstream source is hash-pinned (`UPSTREAM_SOURCE_SHA256`) and checked at
import, like `train()`'s pin in the hub: an sb3 upgrade that changes one of them is a loud error
naming the method, not a silent fork.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import random
import sys
import time
from collections import deque
from typing import Any, Dict, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableRolloutBuffer
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common import utils as _sb3_utils
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor

from agents.training.loop_hooks import LoopHooks

#: The loop, in order. `learn()` below is this table written out; `own_ppo_loop_test` pins both.
LOOP_PHASES: Tuple[str, ...] = (
    "setup",            # _setup_learn: counters, ep-info buffers, env reset if needed, callback init
    "training_start",   # callback.on_training_start(locals(), globals())
    "collect",          # self.collect_rollouts(...) — on_rollout_start, on_step x n, on_rollout_end
    "progress",         # iteration += 1; _current_progress_remaining
    "dump",             # dump_logs(iteration) — BEFORE the update (the TB step contract)
    "update",           # self.train()
    "training_end",     # callback.on_training_end()
)
#: The phases that repeat, once per iteration, in this order.
LOOP_ITERATION: Tuple[str, ...] = ("collect", "progress", "dump", "update")

#: `GEN3AI_PPO_LOOP` values: unset / "owned" = this module; "sb3_reference" = upstream (test seam).
LOOP_ENV = "GEN3AI_PPO_LOOP"
LOOP_OWNED, LOOP_REFERENCE = "owned", "sb3_reference"

#: sha256 of `inspect.getsource(...)` of every upstream method vendored here (sb3 / sb3-contrib 2.8.0).
UPSTREAM_SOURCE_SHA256: Dict[str, str] = {
    "MaskablePPO.learn": "ad8b3b8f3935053cca795d1f77b7faa1d26430e231050bf5d60454ed7106c18e",
    "MaskablePPO.collect_rollouts": "6b894b173404c6f6762c4e930335641215aa26c1e5710e2da74533a9b6e9f314",
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
        "MaskablePPO.collect_rollouts": MaskablePPO.collect_rollouts,
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


def loop_mode() -> str:
    """Which loop `learn()` runs — `GEN3AI_PPO_LOOP`, read once per call. An unknown value REFUSES."""
    v = os.environ.get(LOOP_ENV, "").strip() or LOOP_OWNED
    if v not in (LOOP_OWNED, LOOP_REFERENCE):
        raise ValueError(f"${LOOP_ENV}={v!r}: want {LOOP_OWNED!r} (the default) or {LOOP_REFERENCE!r} "
                         "(the upstream sb3 loop — a test seam for the equivalence A/B)")
    return v


class OwnedLoop:
    """Mixin: the PPO loop. Mixed in AFTER `RolloutProbes` and BEFORE `MaskablePPO`, so
    `RolloutProbes.collect_rollouts` (the env-core dispatch) still wins, and its Python-core branch
    calls `_collect_python` below instead of `super()`."""

    def _reference_loop(self) -> bool:
        """True only inside a `learn()` the `GEN3AI_PPO_LOOP=sb3_reference` seam selected: then EVERY
        vendored method below defers to its upstream original, so the reference arm is upstream
        end to end (its `learn` calls `self._setup_learn` / `self.dump_logs` / … through the MRO)."""
        return getattr(self, "_ppo_loop_mode", LOOP_OWNED) == LOOP_REFERENCE

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
        """`LOOP_PHASES`, written out. Vendored from `MaskablePPO.learn` — the local names are
        unchanged (`on_training_start` hands `locals()` to every callback, so this body binds NO new
        local: the hook table is reached through `self`). The three DECLARED HOOK POINTS
        (`loop_hooks.HOOK_POINTS`: learn > collect | update) are opened here, in that nesting, with
        the owners in `loop_hooks.HOOK_OWNERS` order (gen3_declared_loop_hooks_v1)."""
        self._ppo_loop_mode = loop_mode()
        # the hook table is complete at training start: a later registration is a typed FATAL
        self._loop_hooks.freeze("learn() — training start")
        with self._loop_hooks.around("learn"):
            if self._ppo_loop_mode == LOOP_REFERENCE:
                return self._reference_learn(total_timesteps, callback, log_interval, tb_log_name,
                                             reset_num_timesteps, use_masking, progress_bar)
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
            callback.on_training_start(locals(), globals())

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

            return self

    def _reference_learn(self, total_timesteps: int, callback: Any, log_interval: int, tb_log_name: str,
                         reset_num_timesteps: bool, use_masking: bool, progress_bar: bool) -> Any:
        """The `sb3_reference` seam: upstream's `learn`, with the loop's collect / update hooks applied
        the only way upstream can see them — as instance wrappers, for this call only."""
        print(f"[PPO LOOP] ${LOOP_ENV}={LOOP_REFERENCE}: running the UPSTREAM sb3 loop (test seam)", flush=True)
        hooks = self._loop_hooks
        saved = {n: vars(self).get(n) for n in ("collect_rollouts", "train")}
        orig_collect, orig_train = self.collect_rollouts, self.train

        def collect_rollouts(*a: Any, **k: Any) -> Any:
            with hooks.around("collect"):
                return orig_collect(*a, **k)

        def train(*a: Any, **k: Any) -> Any:
            with hooks.around("update"):
                return orig_train(*a, **k)

        self.collect_rollouts, self.train = collect_rollouts, train  # type: ignore[method-assign]
        try:
            return MaskablePPO.learn(self, total_timesteps, callback=callback,  # type: ignore[arg-type]
                                     log_interval=log_interval, tb_log_name=tb_log_name,
                                     reset_num_timesteps=reset_num_timesteps, use_masking=use_masking,
                                     progress_bar=progress_bar)
        finally:
            for n, v in saved.items():
                if v is None:
                    vars(self).pop(n, None)
                else:
                    setattr(self, n, v)

    # ------------------------------------------------------------------ the hook table
    def _setup_model(self) -> None:
        """sb3's model setup, then the loop's HOOK TABLE (startup; `loop_hooks`). Runs on a fresh build
        and on `load`, so every learner object has exactly one, empty until its owners register."""
        super()._setup_model()   # type: ignore[misc]
        self._loop_hooks = LoopHooks()

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
        budget (`model_build.py`)."""
        if self._reference_loop():
            return BaseAlgorithm._setup_learn(self, total_timesteps, callback,  # type: ignore[arg-type]
                                              reset_num_timesteps, tb_log_name, progress_bar)
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

        # Configure logger's outputs if no logger was passed (the logger is still sb3's — stage 3)
        if not self._custom_logger:
            self._logger = _sb3_utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name,
                                                       reset_num_timesteps)

        # Callback init (BaseCallback / CallbackList are still sb3's — stage 3)
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    # ------------------------------------------------------------------ progress / info
    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """Vendored from `BaseAlgorithm`: 1.0 at the start -> 0.0 at the end (the schedules' input)."""
        if self._reference_loop():
            return BaseAlgorithm._update_current_progress_remaining(self, num_timesteps,  # type: ignore[arg-type]
                                                                    total_timesteps)
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_info_buffer(self, infos: Any, dones: Any = None) -> None:
        """Vendored from `BaseAlgorithm`: Monitor's ``info["episode"]`` (and the Rust collector's) into
        the 100-episode window `dump_logs` reads."""
        if self._reference_loop():
            return BaseAlgorithm._update_info_buffer(self, infos, dones)  # type: ignore[arg-type]
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
        # self.env is always a VecEnv
        if self.env is not None:
            self.env.seed(seed)

    # ------------------------------------------------------------------ dump
    def dump_logs(self, iteration: int = 0) -> None:
        """Vendored from `OnPolicyAlgorithm.dump_logs`. The tag names are read BY NAME by
        `main.ops.killbar` / `tb_read` / `restart_startup` / `stall_exhibit`, the launcher's
        `format.py` and `utils/plot_tb.py` — never rename one."""
        if self._reference_loop():
            return OnPolicyAlgorithm.dump_logs(self, iteration)  # type: ignore[arg-type]
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

    # ------------------------------------------------------------------ collect (Python env core)
    def _collect_python(
        self,
        env: Any,
        callback: Any,
        rollout_buffer: Any,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """Vendored from `MaskablePPO.collect_rollouts` — the Python env core's collection. 🚨 THE
        LOCAL NAMES ARE A CONTRACT (`callback.update_locals(locals())`; `signal_callback` /
        `win_prob_callback` read them), and `on_step` fires BEFORE
        `rollout_buffer.add`. Under the reference seam it is upstream's own method."""
        if self._reference_loop():
            return MaskablePPO.collect_rollouts(self, env, callback, rollout_buffer,  # type: ignore[arg-type]
                                                n_rollout_steps, use_masking)
        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]

                # This is the only change related to invalid action masking
                if use_masking:
                    action_masks = get_action_masks(env)

                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            # Masking is not needed here, the choice of action doesn't matter.
            # We only want the value of the current observation.
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True


def _safe_mean(arr: Any) -> Any:
    """sb3's `safe_mean`, operation for operation: NaN for an empty window, else `float(np.mean)`."""
    return np.nan if len(arr) == 0 else float(np.mean(arr))
