"""`gen3_learner_segment_seed_v1` — a resumed / forked learner trains under ITS run's `--seed`, not the
checkpoint's saved one (`main.train.learner_seed`; `gigo_audit_datapath_2026-10-26`)."""
from __future__ import annotations

import inspect
import random

import numpy as np
import pytest
import torch as th

from main.train import learner_seed as LS


class _Learner:
    """The two things the module touches: ``seed`` and the owned ``set_random_seed``."""

    def __init__(self, saved_seed, num_timesteps):
        self.seed = saved_seed
        self.num_timesteps = num_timesteps

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)


@pytest.fixture
def restore_global_rng():
    state = (random.getstate(), np.random.get_state(), th.get_rng_state())
    yield
    random.setstate(state[0])
    np.random.set_state(state[1])
    th.set_rng_state(state[2])


def test_a_fork_records_its_own_seed_and_the_guard_refuses_the_parents(restore_global_rng):
    """The X35 shape: a fork of `rb_x26_s2601` declared `--seed 3501` loads a checkpoint that saved 2601. The
    guard refuses the loaded learner (FAILS on revert of the reseed), and passes once it is reseeded."""
    m = _Learner(saved_seed=2601, num_timesteps=280_000_170)
    with pytest.raises(LS.UndeclaredLearnerSeed, match="2601"):
        LS.check_declared_seed(m, 3501)
    s = LS.seed_loaded_learner(m, 3501)
    assert m.seed == 3501
    LS.check_declared_seed(m, 3501)
    assert s == LS.learner_segment_seed(3501, 280_000_170)
    got = np.random.permutation(64)
    np.random.seed(s)
    assert np.array_equal(got, np.random.permutation(64))


def test_replicate_seeds_and_restart_segments_get_distinct_streams():
    """Seeds 3501-3505 forked at one step, and one run's restart segments, never share a stream."""
    at_fork = {LS.learner_segment_seed(s, 280_000_170) for s in range(3501, 3506)}
    assert len(at_fork) == 5
    segments = {LS.learner_segment_seed(3501, t) for t in (280_000_170, 286_396_721, 292_000_000)}
    assert len(segments) == 3
    assert LS.learner_segment_seed(3501, 7) == LS.learner_segment_seed(3501, 7)
    assert all(0 <= s < 2 ** 31 for s in at_fork | segments)


def test_a_fresh_learner_passes_the_guard_and_a_none_seed_is_zero():
    LS.check_declared_seed(_Learner(42, 0), 42)
    LS.check_declared_seed(_Learner(None, 0), None)
    with pytest.raises(LS.UndeclaredLearnerSeed):
        LS.check_declared_seed(_Learner(None, 0), 42)


def test_the_trainer_reseeds_after_the_load_and_guards_both_learn_calls():
    """Source pin of the wiring: the reseed runs on the resume path after the hparams, and the guard sits
    before BOTH `model.learn` calls (fresh and resume)."""
    from main.train import model_build

    src = inspect.getsource(model_build.build_and_train)
    assert src.count("seed_loaded_learner(model, args.seed)") == 1
    assert src.count("check_declared_seed(model, args.seed)") == 2
    assert src.count("model.learn(") == 2
    assert src.index("apply_training_hparams(model, args, mappings=mappings)") < src.index("seed_loaded_learner(")
    for part in src.split("model.learn(")[:-1]:
        assert part.rstrip().endswith("FATAL_CONFIG on a mismatch")
