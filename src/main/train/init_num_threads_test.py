"""REGRESSION PIN (`metadata.json` `init_num_threads`, X5 A/B §7.4 precondition): a run RECORDS the torch thread
count its FRESH learner build ran at, taken from inside `single_thread_build` (the value that helper set), and
the first recorded value is IMMUTABLE.

Before this the key was named by the design note and written by nothing (training-agent FINDING, 2026-10-04),
so "equal across every run" could not be checked, only accepted by construction (F-X5-5, `50fdfdc2`).

FAILS on revert:
  * `construct_fresh_learner` not setting the attribute            -> test_fresh_build_records_one_thread...
  * recording the process default instead of the build's own count -> the `3 threads in the caller` case
  * hard-coding 1 instead of what the helper reported              -> test_the_value_is_what_single_thread_build_set
  * `save_model_snapshot` letting a later save overwrite it         -> test_a_resume_keeps_the_original...
  * a fork not recording the value its checkpoint carries           -> test_a_fork_records_its_own...
"""
from __future__ import annotations

import contextlib
import json
import zipfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.model_version import ModelVersion
from agents.model.snapshot import save_model_snapshot
from utils.torch_state_guard import INIT_NUM_THREADS_ATTR, single_thread_build, torch_globals


def _build() -> Any:
    """A FRESH learner through the trainer's own construction site, on the production surface."""
    from agents.training.rust_rollout.testkit import ToyVecEnv
    from main.fresh_checkpoint import _production_policy_kwargs
    from main.train.model_build import construct_fresh_learner

    args, layout, pk = _production_policy_kwargs()
    args.device = "cpu"
    args.seed = 0
    args.batch_size = min(args.batch_size, args.n_steps)
    d = layout["total_dim"]
    space = gym.spaces.Dict({"observation": gym.spaces.Box(-np.inf, np.inf, (d,), np.float32),
                             "action_mask": gym.spaces.MultiBinary(11)})

    class E(gym.Env):
        observation_space = space
        action_space = gym.spaces.Discrete(11)

        def reset(self, **k: Any) -> Any:
            return {"observation": np.zeros(d, np.float32), "action_mask": np.ones(11, np.int8)}, {}

        def step(self, a: Any) -> Any:
            return self.reset()[0], 0.0, False, False, {}

    model = construct_fresh_learner(args, ToyVecEnv([E]), pk)
    return model, args, pk, layout


@pytest.fixture(scope="module")
def fresh_at_three_threads() -> Any:
    """ONE fresh build made while the CALLER runs 3 threads (so "the process default" is 3, not 1)."""
    with torch_globals(num_threads=3):
        model, args, pk, layout = _build()
        caller_after = torch.get_num_threads()
    return model, args, pk, layout, caller_after


def _version(args: Any, pk: dict, layout: dict) -> ModelVersion:
    return ModelVersion.from_layout_and_policy_kwargs(
        layout, pk, vf_coef=args.vf_coef, move_belief_coef=args.move_belief_coef,
        spread_belief_coef=args.spread_belief_coef, opp_belief_aux_coef=args.opp_belief_aux_coef,
        hp_type_belief_coef=args.hp_type_belief_coef, item_belief_coef=args.item_belief_coef)


def _meta(d: Path) -> dict:
    return json.loads((d / "metadata.json").read_text())


def test_fresh_build_records_one_thread_not_the_callers_default(fresh_at_three_threads: Any, tmp_path: Path) -> None:
    from main.train.run_io import _model_hparams

    model, args, pk, layout, caller_after = fresh_at_three_threads
    assert caller_after == 3, "the build must restore the caller's count"
    assert getattr(model, INIT_NUM_THREADS_ATTR) == 1, (
        "the fresh build ran at ONE thread (`single_thread_build`); the record must say so, not the "
        "caller's 3")
    hp = _model_hparams(model)
    assert hp[INIT_NUM_THREADS_ATTR] == 1
    save_model_snapshot(str(tmp_path), _version(args, pk, layout), git_hash="test", hparams=hp)
    assert _meta(tmp_path)["init_num_threads"] == 1


def test_a_resume_keeps_the_original(fresh_at_three_threads: Any, tmp_path: Path) -> None:
    from main.train.run_io import _model_hparams

    model, args, pk, layout, _ = fresh_at_three_threads
    v = _version(args, pk, layout)
    save_model_snapshot(str(tmp_path), v, git_hash="test", hparams=_model_hparams(model))
    assert _meta(tmp_path)["init_num_threads"] == 1
    # a LATER save in the same run dir whose model claims something else (a resumed process, a hand edit)
    # must not rewrite it ...
    hp = dict(_model_hparams(model))
    hp["init_num_threads"] = 7
    save_model_snapshot(str(tmp_path), v, git_hash="test", hparams=hp)
    assert _meta(tmp_path)["init_num_threads"] == 1
    # ... and a save that knows nothing (a periodic checkpoint, no hparams) carries it forward
    save_model_snapshot(str(tmp_path), v, git_hash="test")
    assert _meta(tmp_path)["init_num_threads"] == 1


def test_a_resume_of_an_unrecorded_run_stays_unrecorded(fresh_at_three_threads: Any, tmp_path: Path) -> None:
    """A checkpoint that predates the record carries no attribute: the key is ABSENT (unknown), never a guess."""
    from main.train.run_io import _model_hparams

    model, args, pk, layout, _ = fresh_at_three_threads
    v = _version(args, pk, layout)
    legacy = {k: val for k, val in _model_hparams(model).items() if k != INIT_NUM_THREADS_ATTR}
    save_model_snapshot(str(tmp_path), v, git_hash="test", hparams=legacy)
    assert "init_num_threads" not in _meta(tmp_path)


def test_a_fork_records_its_own(fresh_at_three_threads: Any, tmp_path: Path) -> None:
    """A fork is a NEW run dir: its metadata records the build the loaded weights came from, in its own file,
    and does not depend on the parent's metadata."""
    from main.train.run_io import _model_hparams

    model, args, pk, layout, _ = fresh_at_three_threads
    v = _version(args, pk, layout)
    parent, fork = tmp_path / "parent", tmp_path / "fork"
    save_model_snapshot(str(parent), v, git_hash="test", hparams=_model_hparams(model))
    (parent / "metadata.json").unlink()                       # the fork must not read the parent's file
    save_model_snapshot(str(fork), v, git_hash="test", hparams=_model_hparams(model))
    assert _meta(fork)["init_num_threads"] == 1


def test_the_value_is_what_single_thread_build_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """`construct_fresh_learner` records the count `single_thread_build` REPORTED from inside the block —
    not a hard-coded 1 and not the process default read afterwards."""
    @contextlib.contextmanager
    def two_thread_build() -> Any:
        with torch_globals(num_threads=2):
            yield torch.get_num_threads()

    monkeypatch.setattr("main.train.model_build.single_thread_build", two_thread_build)
    with torch_globals(num_threads=3):
        model, *_ = _build()
    assert getattr(model, INIT_NUM_THREADS_ATTR) == 2


def test_single_thread_build_yields_the_count_in_effect() -> None:
    with torch_globals(num_threads=3):
        with single_thread_build() as n:
            assert n == 1 == torch.get_num_threads()
        assert torch.get_num_threads() == 3


def test_the_attribute_rides_the_checkpoint(fresh_at_three_threads: Any, tmp_path: Path) -> None:
    """SB3 `save` writes the plain int into the zip's `data`, so a resume / fork loads it with the weights."""
    model = fresh_at_three_threads[0]
    p = tmp_path / "m"
    model.save(str(p))
    with zipfile.ZipFile(str(p) + ".zip") as z:
        data = z.read("data").decode()
    assert '"init_num_threads": 1' in data
