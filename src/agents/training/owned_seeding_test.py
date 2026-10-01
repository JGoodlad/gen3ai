"""`gen3_owned_seeding_v1` — the learner seeds like sb3 and touches no cuDNN flag
(`designs/endstate/design_own_ppo_loop.md` stage 2).

sb3's `BaseAlgorithm.set_random_seed` ran from `_setup_model` on every construction and every `load`,
and on a CUDA device it set the PROCESS-WIDE `cudnn.deterministic = True` / `benchmark = False`, which
nothing restored. The owned method makes the same draws in the same order and leaves those flags
alone. `test_cuda_seeding_leaves_the_cudnn_flags_alone` FAILS on revert (sb3's method flips
`deterministic`).
"""
from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, Iterator, Tuple

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm

from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.instrumented_ppo import loop as L


@pytest.fixture
def cudnn_flags() -> Iterator[Tuple[bool, bool]]:
    prev = (th.backends.cudnn.deterministic, th.backends.cudnn.benchmark)
    th.backends.cudnn.deterministic, th.backends.cudnn.benchmark = False, True   # a non-sb3 setting
    try:
        yield (False, True)
    finally:
        th.backends.cudnn.deterministic, th.backends.cudnn.benchmark = prev


def _stub(device: str) -> Any:
    return SimpleNamespace(device=th.device(device), action_space=spaces.Discrete(11), env=None)


def _draws(stub: Any) -> Tuple[float, float, float, int]:
    return (random.random(), float(np.random.rand()), float(th.rand(1)), int(stub.action_space.sample()))


def test_the_learner_class_seeds_through_the_owned_method() -> None:
    assert InstrumentedMaskablePPO.set_random_seed is L.OwnedLoop.set_random_seed


def test_cuda_seeding_leaves_the_cudnn_flags_alone(cudnn_flags: Tuple[bool, bool]) -> None:
    L.OwnedLoop.set_random_seed(_stub("cuda"), 5)          # type: ignore[arg-type]
    assert (th.backends.cudnn.deterministic, th.backends.cudnn.benchmark) == cudnn_flags


def test_sb3_seeding_is_the_leak_this_replaces(cudnn_flags: Tuple[bool, bool]) -> None:
    """The documented upstream behaviour (non-vacuity: the flags CAN move under this fixture)."""
    BaseAlgorithm.set_random_seed(_stub("cuda"), 5)        # type: ignore[arg-type]
    assert th.backends.cudnn.deterministic is True and th.backends.cudnn.benchmark is False


def test_the_draws_equal_sb3s_in_order() -> None:
    a = _stub("cpu")
    BaseAlgorithm.set_random_seed(a, 7)                    # type: ignore[arg-type]
    want = _draws(a)
    b = _stub("cpu")
    L.OwnedLoop.set_random_seed(b, 7)                       # type: ignore[arg-type]
    assert _draws(b) == want


def test_none_seeds_nothing() -> None:
    th.manual_seed(3)
    before = th.random.get_rng_state().clone()
    L.OwnedLoop.set_random_seed(_stub("cpu"), None)        # type: ignore[arg-type]
    assert th.equal(th.random.get_rng_state(), before)
