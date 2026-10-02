"""K6's RNG half (`gen3_no_global_reseed_v1`): no process-global RNG is SEEDED after the learner
freezes, and a model load never touches the global streams.

THE BUG (2026-10-02, deletion-pass manifest P1): `InferenceMaskablePPO._setup_model` called sb3's
`set_random_seed(self.seed)`, so every opponent / pool / sentinel load mid-run re-seeded Python
`random`, NumPy and torch to the snapshot's saved seed (the run's own `--seed`) — and the PPO minibatch
permutation (`np.random.permutation` in `RolloutBuffer.get`) replayed the run's first updates' stream
after every load.

Each test fails on revert:
  * restore `self.set_random_seed(self.seed)` in the inference load → the global states move across a
    load, the post-load permutation EQUALS the startup one, and a load under a frozen learner raises
    `GlobalReseedError`;
  * drop the `isolated_global_rng()` around the construction → torch's CPU state moves across a load,
    and with the ride-along heads ON (their constructors `torch.manual_seed`) a load under the freeze
    raises;
  * drop `global_rng_guard.arm` from `LearnerFreeze.freeze` → a seed after the freeze passes silently.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch as th

from agents.training import global_rng_guard as G
from agents.training.opponent_inference_load_test import RIDEALONG_OFF, RIDEALONG_ON, _build, _save


def _states() -> Tuple[Any, bytes, bytes]:
    st = np.random.get_state()
    return (random.getstate(), bytes(st[1].tobytes()) + bytes(str(st[2:]), "ascii"),
            bytes(th.get_rng_state().numpy().tobytes()))


@pytest.fixture
def restore_rng():
    """These tests deliberately seed the global streams; give them back."""
    saved = (random.getstate(), np.random.get_state(), th.get_rng_state())
    try:
        yield
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        th.set_rng_state(saved[2])


@pytest.fixture(scope="module")
def world(tmp_path_factory: Any) -> Dict[str, Any]:
    root = tmp_path_factory.mktemp("rngload")
    on, v_on, _ = _build(RIDEALONG_ON, 1001)
    off, v_off, _ = _build(RIDEALONG_OFF, 1001)
    assert on.seed == 1001 and off.seed == 1001          # precondition: the snapshot's saved seed
    return {"on": on, "v_on": v_on, "off": off, "v_off": v_off,
            "zip_on": _save(on, v_on, root / "heads"), "zip_off": _save(off, v_off, root / "bare"),
            "root": root}


# ------------------------------------------------------------------------------- the guard itself
_SEEDERS = [
    ("python", lambda: random.seed(7)),
    ("numpy", lambda: np.random.seed(7)),
    ("torch", lambda: th.manual_seed(7)),
    ("torch", lambda: th.random.manual_seed(7)),
    ("torch", lambda: th.seed()),
    ("torch", lambda: th.cuda.manual_seed_all(7)),
    ("torch", lambda: th.cuda.manual_seed(7)),
]


@pytest.mark.parametrize("stream,call", _SEEDERS)
def test_an_ARMED_guard_refuses_every_global_seed_and_names_the_site(stream: str, call: Any,
                                                                      restore_rng: Any) -> None:
    sink: list = []
    before = _states()
    G.arm(sink)
    try:
        with pytest.raises(G.GlobalReseedError) as ei:
            call()
    finally:
        G.disarm(sink)
    assert f"the global {stream} RNG was SEEDED" in str(ei.value)
    assert "global_rng_guard_test.py" in str(ei.value)                 # the call SITE
    assert len(sink) == 1 and "global_rng_guard_test.py" in sink[0]    # sticky
    assert _states() == before                                          # nothing was seeded


def test_disarm_restores_the_original_functions(restore_rng: Any) -> None:
    import torch.cuda
    import torch.random
    origs = (random.seed, np.random.seed, th.manual_seed, torch.random.manual_seed, th.seed,
             torch.cuda.manual_seed, torch.cuda.manual_seed_all)
    sink: list = []
    G.arm(sink)
    assert random.seed is not origs[0] and G.armed()
    G.disarm(sink)
    assert (random.seed, np.random.seed, th.manual_seed, torch.random.manual_seed, th.seed,
            torch.cuda.manual_seed, torch.cuda.manual_seed_all) == origs
    assert not G.armed()
    random.seed(3)                                       # unarmed: an ordinary seed again


def test_a_seed_INSIDE_an_isolated_scope_is_local_and_the_global_streams_come_back(restore_rng: Any) -> None:
    random.seed(11)
    np.random.seed(11)
    th.manual_seed(11)
    before = _states()
    sink: list = []
    G.arm(sink)
    try:
        with G.isolated_global_rng():
            th.manual_seed(5)
            th.randn(64)                                  # a construction's init draws
            random.seed(5)
            random.random()
            np.random.seed(5)
            np.random.rand(3)
            with G.isolated_global_rng():                 # nested scopes compose
                th.manual_seed(6)
    finally:
        G.disarm(sink)
    assert sink == []
    assert _states() == before


def test_an_isolated_scope_leaves_an_UNSEEDED_draw_advanced(restore_rng: Any) -> None:
    """Restoring a stream that was only DRAWN from would rewind it — the very class this closes."""
    np.random.seed(2)
    ref = np.random.RandomState(2)
    ref.rand(4)
    with G.isolated_global_rng():
        np.random.rand(4)
    assert np.random.rand() == ref.rand()


def test_a_FORKED_child_is_not_armed(restore_rng: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    sink: list = []
    G.arm(sink)
    try:
        monkeypatch.setattr(G, "_armed_pid", os.getpid() + 1)   # as a forked child sees it
        random.seed(1)
        assert sink == []
    finally:
        monkeypatch.setattr(G, "_armed_pid", os.getpid())
        G.disarm(sink)


# ----------------------------------------------------------------------- wired into the K6 freeze
def test_the_FREEZE_arms_the_guard_and_a_swallowed_seed_still_fails_the_check(world: Dict[str, Any],
                                                                              restore_rng: Any) -> None:
    from agents.training.learner_lifecycle import LazyAcquisitionError, LearnerFreeze

    freeze = LearnerFreeze(world["off"])
    freeze.freeze("test: end of startup")
    try:
        try:
            np.random.seed(42)
        except Exception:                                 # a caller swallows it ...
            pass
        with pytest.raises(LazyAcquisitionError, match="SEEDED after the learner froze"):
            freeze.check("test: update end")              # ... the next check does not
    finally:
        freeze.release()
    assert not G.armed()
    np.random.seed(1)                                     # released: allowed again


# -------------------------------------------------------------------- the load touches no stream
@pytest.mark.parametrize("which", ["zip_off", "zip_on"])
def test_an_OPPONENT_load_leaves_every_global_stream_bit_identical(world: Dict[str, Any], which: str,
                                                                   restore_rng: Any) -> None:
    from agents.model.snapshot import load_foreign_opponent, load_opponent_snapshot

    random.seed(123)
    np.random.seed(123)
    th.manual_seed(123)
    random.random()
    np.random.rand(5)
    th.randn(5)
    before = _states()
    load_opponent_snapshot(str(world[which]), current_version=world["v_on"])
    load_foreign_opponent(str(world[which]), current_version=world["v_on"])
    assert _states() == before


def test_the_minibatch_permutation_after_a_POOL_load_does_not_replay_the_startup_one(
        world: Dict[str, Any], restore_rng: Any) -> None:
    """The measured symptom: the trainee seeds at startup with its `--seed`; every snapshot it saves
    carries that seed; before the fix a pool load re-seeded NumPy to it and the next update drew the
    run's FIRST permutation again."""
    from agents.training.snapshot_pool import SnapshotPool

    trainee = world["off"]
    n = 4096
    np.random.seed(trainee.seed)                          # the trainee's startup seeding
    first = np.random.permutation(n)                      # update 1's first epoch
    np.random.permutation(n)                              # ... the run goes on
    pool = SnapshotPool(world["root"] / "pool_perm", current_version=world["v_off"], device="cpu")
    pool.add(trainee, step=4000)
    pool.load_model(pool._entries[0])                     # a promotion / pool refresh
    after = np.random.permutation(n)
    assert not np.array_equal(after, first), "the post-load update replayed the run's first permutation"


def test_a_POOL_load_with_ride_along_heads_under_a_FROZEN_learner_raises_nothing(
        world: Dict[str, Any], restore_rng: Any) -> None:
    """The heads' constructors `torch.manual_seed` — inside the load's isolated scope, so local."""
    from agents.training.learner_lifecycle import LearnerFreeze
    from agents.training.snapshot_pool import SnapshotPool

    trainee = world["on"]
    pool = SnapshotPool(world["root"] / "pool_frozen", current_version=world["v_on"], device="cpu")
    pool.add(trainee, step=1000)
    freeze = LearnerFreeze(trainee)
    freeze.freeze("test: end of startup")
    try:
        before = _states()
        m = pool.load_model(pool._entries[0])
        assert m.policy.ridealong is not None             # precondition: the heads were BUILT
        assert _states() == before
        assert freeze.violations == []
        freeze.check("test: after the pool load")
    finally:
        freeze.release()


def test_the_isolated_load_plays_the_identical_function(world: Dict[str, Any], restore_rng: Any) -> None:
    """The construction's discarded init draws are the only thing the scope changes: the loaded
    weights are the saved ones."""
    from agents.model.snapshot import load_opponent_snapshot

    m = load_opponent_snapshot(str(world["zip_off"]), current_version=world["v_off"])
    saved = world["off"].policy.state_dict()
    got = m.policy.state_dict()
    assert set(saved) == set(got)
    for k, v in saved.items():
        assert th.equal(v.cpu(), got[k].cpu()), k


def test_the_static_twin_exists() -> None:
    from utils.paths import src_path
    assert src_path("global_rng_seed_gate_test.py").is_file()
