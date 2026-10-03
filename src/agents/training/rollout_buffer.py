"""The learner's ROLLOUT BUFFER — ours (`gen3_owned_rollout_buffer_v1`; `designs/endstate/design_own_ppo_loop.md` §3.3).

Until deletion pass U4 the learner's buffer was sb3-contrib's ``MaskableDictRolloutBuffer``. This is that
buffer, written out OPERATION FOR OPERATION for the one layout the learner uses — a ``Dict`` observation
of ``Box`` (or ``MultiBinary``) spaces, each stored at its own shape, and a ``Discrete`` action — so the K9 learner golden (which fills it from the pinned
rollout and runs one ``train()``) is bit-identical. Any other layout is REFUSED at construction, never
approximated.

THE CONTRACT ``train()`` and the K9 golden rest on (each property is load-bearing):

* **Storage**: host ``numpy`` arrays, ``[buffer_size, n_envs, ...]`` until the first ``get()``: one array per
  observation key (the key's own dtype), ``actions`` (``[.., 1]``, the action space's dtype), ``rewards`` /
  ``returns`` / ``episode_starts`` / ``values`` / ``log_probs`` / ``advantages`` (float32), ``action_masks``
  (``[.., n_actions]``, float32, ones until written). ``reset()`` re-allocates every one of them.
  The writers are the Rust collector's two fills (``rust_rollout/store.py``: ``_copy_rows`` writes the
  arrays in place, then ``pos`` / ``full``) and, in tests, the toy collector's ``add``.
* **``get(batch_size)``**: ONE ``np.random.permutation(buffer_size * n_envs)`` from the GLOBAL numpy RNG
  per call (one call per epoch), drawn BEFORE the flatten; the first call flattens every array in place
  (axes 0/1 swapped, so a flat row index is ``env * buffer_size + step``); then consecutive
  ``batch_size`` slices of the permutation, each served by ``_get_samples``. `device_batches` replaces
  ``_get_samples`` / ``get`` on the INSTANCE for one update and depends on exactly this shape.
* **``_get_samples``**: a ``th.tensor`` COPY of each field's rows on the buffer's device — values,
  log-probs, advantages and returns flattened to 1-D, the masks reshaped to ``[-1, n_actions]`` — as a
  `RolloutSamples`.
* **Window GAE** (``compute_returns_and_advantage``, the fixed-window trigger's fill and the toy
  collector): sb3's arithmetic, step for step, with ``returns = advantages + values``. The complete-game
  fill computes its own per game (``store.game_gae``, the same operations) and writes the arrays.

It stays on the HOST, deliberately: a device-resident buffer (the design's first sketch) would undo the
staged micro-batch path `device_batches` chose for memory — ~2 micro-batches on the card instead of the
whole flattened buffer (`gen3_device_batch_mode_v1`).
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, NamedTuple, Optional

import numpy as np
import torch as th
from gymnasium import spaces


class RolloutSamples(NamedTuple):
    """One micro-batch, as ``train()`` reads it (sb3-contrib's ``MaskableDictRolloutBufferSamples`` fields)."""

    observations: Dict[str, th.Tensor]
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor


class RolloutBufferLayoutError(TypeError):
    """A buffer was asked for a layout the learner never uses (named)."""


def get_device(device: Any = "auto") -> th.device:
    """sb3's ``get_device``: ``"auto"`` is cuda; a cuda request on a box without one is the cpu."""
    if device == "auto":
        device = "cuda"
    device = th.device(device)
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")
    return device


class RolloutBuffer:
    """See the module docstring. The constructor's signature is the one sb3's ``_setup_model`` calls
    (``rollout_buffer_class(n_steps, obs_space, act_space, device, gamma=, gae_lambda=, n_envs=)``)."""

    #: The flat per-row arrays beside the observation dict, in the order ``get()`` flattens them.
    FLAT = ("actions", "values", "log_probs", "advantages", "returns", "action_masks")

    observations: Dict[str, np.ndarray]

    def __init__(self, buffer_size: int, observation_space: Any, action_space: Any, device: Any = "auto",
                 gae_lambda: float = 1, gamma: float = 0.99, n_envs: int = 1):
        if not isinstance(observation_space, spaces.Dict):
            raise RolloutBufferLayoutError(f"RolloutBuffer stores a Dict observation, not {type(observation_space).__name__}")
        bad = {k: type(sp).__name__ for k, sp in observation_space.spaces.items()
               if not isinstance(sp, (spaces.Box, spaces.MultiBinary))}
        if bad:
            raise RolloutBufferLayoutError(f"RolloutBuffer stores Box / MultiBinary observation keys only; not {bad}")
        if not isinstance(action_space, spaces.Discrete):
            raise RolloutBufferLayoutError(f"RolloutBuffer stores a Discrete action, not {type(action_space).__name__}")
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = {k: tuple(sp.shape) for k, sp in observation_space.spaces.items()}
        self.action_dim = 1
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    # ------------------------------------------------------------------ storage
    def reset(self) -> None:
        """Re-allocate every array at ``[buffer_size, n_envs, ...]``; ``pos`` 0, not full, not flattened."""
        self.mask_dims = int(self.action_space.n)
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)
        self.observations = {}
        for key, shape in self.obs_shape.items():
            self.observations[key] = np.zeros((self.buffer_size, self.n_envs, *shape),
                                              dtype=self.observation_space[key].dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    def add(self, obs: Dict[str, np.ndarray], action: np.ndarray, reward: np.ndarray, episode_start: np.ndarray,
            value: th.Tensor, log_prob: th.Tensor, action_masks: Optional[np.ndarray] = None) -> None:
        """Write one vector step at ``pos`` (the toy collector's writer; the Rust fills write the arrays)."""
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)
        for key in self.observations.keys():
            self.observations[key][self.pos] = np.array(obs[key])
        action = action.reshape((self.n_envs, self.action_dim))
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    # ------------------------------------------------------------------ window GAE
    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """GAE(λ) over the window, bootstrapping each env's last row with ``last_values`` unless ``dones``;
        ``returns = advantages + values`` (the TD(λ) target)."""
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]
        last_gae_lam: Any = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------ reading
    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """``[n_steps, n_envs, ...]`` -> ``[n_envs * n_steps, ...]`` (env-major), a trailing 1 for a 2-D array."""
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def get(self, batch_size: Optional[int] = None) -> Iterator[RolloutSamples]:
        """One epoch's micro-batches (module docstring: the permutation is drawn FIRST, then the flatten)."""
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)
            for tensor in ("actions", "values", "log_probs", "advantages", "returns", "action_masks"):
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    def _get_samples(self, batch_inds: np.ndarray, env: Any = None) -> RolloutSamples:
        return RolloutSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            action_masks=self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
        )
