"""The learner's ENV — the declared surface the learner holds (`gen3_owned_trainer_env_v1`; deletion pass U4).

Until deletion pass U4 the trainer's env (``RustVecEnv``) was an sb3 ``VecEnv`` subclass, so sb3's
constructor would accept it. The learner never STEPS it — the rollout is the Rust collector's — so what
the learner and its callbacks actually use is small, and this base class is it:

* ``num_envs``, ``observation_space``, ``action_space`` — what the model is built / loaded against (the
  buffer's columns, the spaces a load is checked against);
* ``reset()`` — ``_setup_learn``'s first observation (unused by the Rust collector, which owns every row);
* ``seed(seed)`` — owned seeding's last draw (``OwnedLoop.set_random_seed``);
* ``env_method(name, *args, indices=None, **kwargs)`` — the callbacks' push / pull surface;
* ``close()``.

`OwnedLoop._wrap_env` accepts an instance of this class AS IS and refuses anything else (an sb3 ``VecEnv``
or a raw gym env would be wrapped by sb3 into a ``DummyVecEnv`` + ``Monitor`` that nothing here steps).
The production subclass is `rust_vec_env.RustVecEnv`; the toy learners in tests use
``rust_rollout.testkit.ToyVecEnv``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class TrainerVecEnv(ABC):
    """See the module docstring."""

    def __init__(self, num_envs: int, observation_space: Any, action_space: Any):
        self.num_envs = int(num_envs)
        self.observation_space = observation_space
        self.action_space = action_space
        self._seeds: List[Optional[int]] = [None] * self.num_envs

    def seed(self, seed: Optional[int] = None) -> Sequence[Optional[int]]:
        """Record per-env seeds ``seed + i`` for the next ``reset`` (sb3's ``VecEnv.seed``, which draws no RNG)."""
        if seed is None:
            return self._seeds
        self._seeds = [int(seed) + i for i in range(self.num_envs)]
        return self._seeds

    @abstractmethod
    def reset(self) -> Dict[str, np.ndarray]:
        """The first observation, ``{key: [num_envs, ...]}``."""

    @abstractmethod
    def env_method(self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any) -> List[Any]:
        """Call ``method_name`` per env; one result per env in ``indices`` (all when None)."""

    def close(self) -> None:
        return None

    def _indices(self, indices: Any) -> Sequence[int]:
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return list(indices)
