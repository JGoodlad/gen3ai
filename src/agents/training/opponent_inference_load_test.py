"""An OPPONENT never trains, so an opponent load never acquires training state
(`gen3_opponent_inference_load_v1`), and it may differ from the trainee in the DECLARED ride-along
key set only (F-MEM).

THE BUG (2026-10-01, the X26 ride-along baseline at N = 256, `--self-play-start-wr 0`): the first pool
load ran `InstrumentedMaskablePPO._setup_model` → `_ridealong_acquire`, which built and PRE-STEPPED an
Adam over the snapshot's ride-along tensors AFTER the learner froze; K6's global step hook FATALed the
run ("an optimizer outside the frozen set STEPPED after the freeze"). And a pool / sentinel snapshot
that differed from the trainee in ride-along heads alone was refused (`ridealong_ensemble mismatch`)
although the heads are detached and no opponent forward reads them.

Each test fails on revert:
  * the pool / sentinel / foreign-opponent load under a FROZEN trainee builds no optimizer and records
    no violation (revert the pool to `load_model_snapshot` → the pre-step raises `LazyAcquisitionError`);
  * a snapshot WITH heads into a run WITHOUT, and the reverse, both load as opponents (revert
    `check_opponent_snapshot_compatible` to `check_compatible` → `ModelVersionError`);
  * a NON-ride-along mismatch is still refused on the opponent path;
  * the TRAINEE's resume stays strict on a head mismatch, and its own ride-along optimizers are still
    acquired in `_setup_model` (startup) — and an opponent load after the freeze leaves them untouched;
  * the T2 slot identity ignores the heads in both directions, and a served replica carries none.
"""
from __future__ import annotations

import dataclasses
import gc
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch as th

RIDEALONG_ON = {"ridealong_ensemble": 2, "ridealong_rnd": True, "ridealong_adv": 2,
                "ridealong_opp": 2, "ridealong_rnd_variants": "all"}
RIDEALONG_OFF = {"ridealong_ensemble": 0, "ridealong_rnd": False, "ridealong_adv": 0,
                 "ridealong_opp": 0, "ridealong_rnd_variants": "off"}


def _build(ridealong: Dict[str, Any], seed: int) -> Tuple[Any, Any, Dict[str, Any]]:
    """A CPU `InstrumentedMaskablePPO` (the trainee's class) at the production surface, plus its
    `ModelVersion`."""
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.model.model_version import ModelVersion
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from main.fresh_checkpoint import _production_policy_kwargs

    args, layout, pk = _production_policy_kwargs()
    pk = {**pk, "features_extractor_kwargs": {**pk["features_extractor_kwargs"], **ridealong}}
    total_dim = layout["total_dim"]
    obs_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (total_dim,), np.float32),
        "action_mask": gym.spaces.MultiBinary(11)})

    class _E(gym.Env):
        observation_space = obs_space
        action_space = gym.spaces.Discrete(11)

        def reset(self, **kwargs: Any) -> Any:
            return {"observation": np.zeros(total_dim, np.float32),
                    "action_mask": np.ones(11, np.int8)}, {}

        def step(self, action: Any) -> Any:
            return self.reset()[0], 0.0, False, False, {}

    th.manual_seed(seed)
    model = InstrumentedMaskablePPO(Gen3DualHeadMaskablePolicy, DummyVecEnv([_E]), n_steps=8,
                                    batch_size=4, device="cpu", seed=seed, policy_kwargs=pk,
                                    vf_coef=args.vf_coef)
    version = ModelVersion.from_layout_and_policy_kwargs(layout, pk, vf_coef=args.vf_coef)
    return model, version, pk


def _save(model: Any, version: Any, run_dir: Path) -> Path:
    from agents.model.snapshot import save_model_snapshot

    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / "final_model"))
    save_model_snapshot(str(run_dir), version, git_hash="test")
    return run_dir / "final_model.zip"


@pytest.fixture(scope="module")
def world(tmp_path_factory: Any) -> Dict[str, Any]:
    root = tmp_path_factory.mktemp("oppload")
    on, v_on, _ = _build(RIDEALONG_ON, 1)
    off, v_off, _ = _build(RIDEALONG_OFF, 2)
    assert on.policy.ridealong is not None and off.policy.ridealong is None   # precondition
    return {"on": on, "v_on": v_on, "off": off, "v_off": v_off,
            "zip_on": _save(on, v_on, root / "heads"), "zip_off": _save(off, v_off, root / "bare"),
            "root": root}


def _optimizers() -> set:
    import warnings

    with warnings.catch_warnings():        # isinstance on torch.distributed's deprecated proxies warns
        warnings.simplefilter("ignore", FutureWarning)
        return {id(o) for o in gc.get_objects() if isinstance(o, th.optim.Optimizer)}


def _assert_inference_only(m: Any) -> None:
    from agents.training.instrumented_ppo.inference import InferenceMaskablePPO

    assert isinstance(m, InferenceMaskablePPO)
    assert m.policy.optimizer is None
    assert m._ridealong_opt is None and m._ridealong_vopts == {}
    assert getattr(m, "rollout_buffer", None) is None


# ---------------------------------------------------------------- after the freeze: acquires nothing
def test_a_POOL_load_after_the_freeze_builds_no_optimizer_and_trips_nothing(world: Dict[str, Any]) -> None:
    """The X26 chain, in miniature: the trainee (heads ON) seeds its pool, FREEZES, and the pool loads
    the snapshot. Before the fix the load pre-stepped a ride-along Adam and the step hook raised."""
    from agents.training.learner_lifecycle import LearnerFreeze
    from agents.training.snapshot_pool import SnapshotPool

    trainee = world["on"]
    pool = SnapshotPool(world["root"] / "pool", current_version=world["v_on"], device="cpu")
    for step in (0, 1000, 2000):
        pool.add(trainee, step=step)
    own = (trainee._ridealong_opt, dict(trainee._ridealong_vopts), trainee.policy.optimizer)
    freeze = LearnerFreeze(trainee)
    freeze.freeze("test: end of startup")
    try:
        before = _optimizers()
        for e in list(pool._entries):          # >= 3 pool loads, like the real proof
            m = pool.load_model(e)
            _assert_inference_only(m)
            assert m.policy.ridealong is not None   # the heads' WEIGHTS load (a reader reads them)
        made = _optimizers() - before
        assert not made, f"an opponent load built {len(made)} optimizer(s)"
        assert freeze.violations == []
        freeze.check("test: after the pool loads")
    finally:
        freeze.release()
    # the trainee's own startup-acquired optimizers are untouched by the opponent loads
    assert (trainee._ridealong_opt, trainee._ridealong_vopts, trainee.policy.optimizer) == \
        (own[0], own[1], own[2])


def test_SENTINEL_and_FOREIGN_opponent_loads_after_the_freeze_acquire_nothing(world: Dict[str, Any]) -> None:
    from agents.model.snapshot import load_foreign_opponent, load_opponent_snapshot
    from agents.training.learner_lifecycle import LearnerFreeze

    freeze = LearnerFreeze(world["off"])
    freeze.freeze("test")
    try:
        before = _optimizers()
        a = load_opponent_snapshot(str(world["zip_on"]), current_version=world["v_off"])
        b, _ = load_foreign_opponent(str(world["zip_on"]), current_version=world["v_off"])
        for m in (a, b):
            _assert_inference_only(m)
        assert not (_optimizers() - before)
        freeze.check("test")
    finally:
        freeze.release()


# ---------------------------------------------------------------- F-MEM: the ride-along key set only
def test_an_opponent_may_differ_in_ride_along_heads_in_EITHER_direction(world: Dict[str, Any]) -> None:
    from agents.model.snapshot import load_opponent_snapshot

    with_heads = load_opponent_snapshot(str(world["zip_on"]), current_version=world["v_off"])
    bare = load_opponent_snapshot(str(world["zip_off"]), current_version=world["v_on"])
    assert with_heads.policy.ridealong is not None
    assert bare.policy.ridealong is None
    # same-arch control: the pool's own case
    load_opponent_snapshot(str(world["zip_on"]), current_version=world["v_on"])


def test_a_NON_ride_along_mismatch_is_still_refused_on_the_opponent_path(world: Dict[str, Any]) -> None:
    from agents.model.model_version import ModelVersionError
    from agents.model.snapshot import load_opponent_snapshot

    bad = dataclasses.replace(world["v_on"], use_popart=not world["v_on"].use_popart)
    with pytest.raises(ModelVersionError, match="PopArt"):
        load_opponent_snapshot(str(world["zip_off"]), current_version=bad)
    bad2 = dataclasses.replace(world["v_off"], value_dist_bins=world["v_off"].value_dist_bins + 7)
    with pytest.raises(ModelVersionError, match="value_dist_bins"):
        world["v_on"].check_opponent_snapshot_compatible(bad2)


def test_the_TRAINEE_resume_stays_strict_on_a_head_mismatch(world: Dict[str, Any]) -> None:
    from agents.model.model_version import ModelVersionError
    from agents.model.snapshot import load_model_snapshot

    with pytest.raises(ModelVersionError, match="ridealong_ensemble mismatch"):
        load_model_snapshot(str(world["zip_off"]), env=None, current_version=world["v_on"])
    with pytest.raises(ModelVersionError, match="ridealong_ensemble mismatch"):
        world["v_off"].check_compatible(world["v_on"])


def test_the_TRAINEE_acquires_its_ride_along_optimizers_in_setup(world: Dict[str, Any]) -> None:
    """Startup only: a fresh build and a resume both come out of `_setup_model` holding every
    optimizer (nothing is left to a later lazy build)."""
    from agents.model.ridealong_heads import RND_VARIANTS
    from agents.model.snapshot import load_model_snapshot
    from agents.training.instrumented_ppo.inference import InferenceMaskablePPO

    for m in (world["on"], load_model_snapshot(str(world["zip_on"]), env=None,
                                               current_version=world["v_on"])):
        assert not isinstance(m, InferenceMaskablePPO)
        assert m._ridealong_opt is not None and m._ridealong_opt_owner is m.policy.ridealong
        assert set(m._ridealong_vopts) == set(RND_VARIANTS)
        assert m.policy.optimizer is not None


def test_the_ride_along_key_set_is_every_ride_along_field() -> None:
    """One source of truth: `RIDEALONG_FLAGS` names exactly the `ridealong_*` fields of `ModelVersion`
    — a new head's field cannot escape the opponent allowlist, nor can a non-head field enter it."""
    from agents.model.model_version import ModelVersion
    from agents.model.ridealong_heads import RIDEALONG_FLAGS

    fields = {f.name for f in dataclasses.fields(ModelVersion) if f.name.startswith("ridealong")}
    assert fields == set(RIDEALONG_FLAGS)


# ---------------------------------------------------------------- the inference class refuses training
def test_an_inference_load_refuses_to_train_or_save(world: Dict[str, Any], tmp_path: Path) -> None:
    from agents.model.snapshot import load_foreign_opponent
    from agents.training.instrumented_ppo.inference import InferenceOnlyModelError
    from agents.training.instrumented_ppo.ridealong_terms import RideAlongLifecycleViolation

    m, _ = load_foreign_opponent(str(world["zip_on"]), current_version=world["v_on"])
    with pytest.raises(InferenceOnlyModelError):
        m.save(str(tmp_path / "x"))
    with pytest.raises(InferenceOnlyModelError):
        m.learn(total_timesteps=8)
    with pytest.raises(InferenceOnlyModelError):
        m.train()
    with pytest.raises(RideAlongLifecycleViolation):
        m._ridealong_optimizer(m.policy.ridealong)
    # the opt-out an offline FITTING tool takes (the warm-start's student) is the full learner
    full, _ = load_foreign_opponent(str(world["zip_on"]), current_version=world["v_on"],
                                    inference_only=False)
    assert full.policy.optimizer is not None and full._ridealong_opt is not None


def test_the_inference_load_plays_the_identical_function(world: Dict[str, Any]) -> None:
    """Same weights, same forward: the inference load's policy state dict equals the full load's."""
    from agents.model.snapshot import load_foreign_opponent

    a, _ = load_foreign_opponent(str(world["zip_on"]), current_version=world["v_on"])
    b, _ = load_foreign_opponent(str(world["zip_on"]), current_version=world["v_on"], inference_only=False)
    sa, sb = a.policy.state_dict(), b.policy.state_dict()
    assert sa.keys() == sb.keys()
    assert all(th.equal(sa[k], sb[k]) for k in sa)


# ---------------------------------------------------------------- T2: the slot identity
def test_the_T2_slot_identity_ignores_the_heads_in_either_direction(world: Dict[str, Any]) -> None:
    from agents.inference.service.slots import (SlotGroup, forward_fingerprint, served_state_dict,
                                                state_signature)
    from agents.inference.service.spec import SlotGroupSpec
    from agents.model.ridealong_heads import RIDEALONG_STATE_PREFIX

    on, off = world["on"].policy, world["off"].policy
    assert state_signature(served_state_dict(on)) == state_signature(served_state_dict(off))
    assert forward_fingerprint(on) == forward_fingerprint(off)
    group = SlotGroup(SlotGroupSpec(name="t", n_slots=2, template=on), th.device("cpu"))
    assert all(r.ridealong is None for r in group.policies), "a served replica carries the heads"
    assert not any(k.startswith(RIDEALONG_STATE_PREFIX) for k in group.stacked)
    group.copy_in(1, group.check_loadable(off))
    assert on.ridealong is not None, "the template was mutated"
    back = SlotGroup(SlotGroupSpec(name="u", n_slots=1, template=off), th.device("cpu"))
    back.check_loadable(on)
