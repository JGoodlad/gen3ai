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
    from agents.training.rust_rollout.testkit import ToyVecEnv

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
    model = InstrumentedMaskablePPO(Gen3DualHeadMaskablePolicy, ToyVecEnv([_E]), n_steps=8,
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

    bad = dataclasses.replace(world["v_on"], opp_belief_cls_k=world["v_on"].opp_belief_cls_k + 2)
    with pytest.raises(ModelVersionError, match="opp_belief_cls_k"):
        load_opponent_snapshot(str(world["zip_off"]), current_version=bad)
    bad2 = dataclasses.replace(world["v_off"], damage_topk_k=world["v_off"].damage_topk_k + 7)
    with pytest.raises(ModelVersionError, match="damage_topk_k"):
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


# ---------------------------------------------------------------- gen3_strict_checkpoint_load_v1 (P10 F4)
def _rewrite_policy(src: Path, dst_dir: Path, edit: Any) -> Path:
    """A copy of the run at ``src`` (zip + its ``model_config.json``) whose ``policy.pth`` is ``edit(sd)``."""
    import io
    import shutil
    import zipfile

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src.parent / "model_config.json", dst_dir / "model_config.json")
    dst = dst_dir / "final_model.zip"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if name == "policy.pth":
                sd = th.load(io.BytesIO(data), map_location="cpu")
                edit(sd)
                buf = io.BytesIO()
                th.save(sd, buf)
                data = buf.getvalue()
            zout.writestr(name, data)
    return dst


def _drop(submodule: str) -> Any:
    def edit(sd: Dict[str, Any]) -> None:
        gone = [k for k in sd if f".{submodule}." in k]
        # precondition: the submodule is an EXTRACTOR submodule, so the strict error names
        # `pi_features_extractor` — exactly sb3's "SB3 < 1.7.0" non-strict retry trigger
        assert gone and any(k.startswith("pi_features_extractor.") for k in gone), submodule
        for k in gone:
            del sd[k]
    return edit


# `flat_intent_head`: X5's intent pointer (the blob-era `alpha_head` is RETIRED by the policy since the X5 version
# break and holds no key to drop).
@pytest.mark.parametrize("submodule", ["flat_intent_head", "win_head"])
def test_a_checkpoint_MISSING_an_extractor_submodule_is_refused_by_every_load(
        world: Dict[str, Any], tmp_path: Path, submodule: str) -> None:
    """P10 F4: sb3's `load` retried with `exact_match=False` whenever the strict error mentioned
    `pi_features_extractor` (any missing extractor key does, via the alias), so a checkpoint with a
    submodule's keys deleted loaded "normally" with that submodule at FRESH INIT. The opponent path
    (pool / sentinel / foreign) and the trainee's resume must all REFUSE it."""
    from agents.model.snapshot import load_foreign_opponent, load_model_snapshot, load_opponent_snapshot
    from agents.training.instrumented_ppo.strict_load import StrictLoadError

    bad = _rewrite_policy(world["zip_off"], tmp_path / "dropped", _drop(submodule))
    with pytest.raises(StrictLoadError, match=submodule):
        load_opponent_snapshot(str(bad), current_version=world["v_off"])
    with pytest.raises(StrictLoadError, match=submodule):
        load_foreign_opponent(str(bad), current_version=world["v_off"])
    with pytest.raises(StrictLoadError, match=submodule):
        load_model_snapshot(str(bad), env=None, current_version=world["v_off"])


def test_a_checkpoint_with_an_UNEXPECTED_key_is_refused(world: Dict[str, Any], tmp_path: Path) -> None:
    from agents.model.snapshot import load_model_snapshot, load_opponent_snapshot

    def extra(sd: Dict[str, Any]) -> None:
        # under a LIVE submodule: a key under a RETIRED one (`alpha_head`, registered as None) is not reported by
        # torch's strict load at all — the parent skips a None child's whole prefix
        sd["features_extractor.flat_intent_head.not_a_parameter"] = th.zeros(3)

    bad = _rewrite_policy(world["zip_off"], tmp_path / "extra", extra)
    with pytest.raises(RuntimeError, match="not_a_parameter"):
        load_opponent_snapshot(str(bad), current_version=world["v_off"])
    with pytest.raises(RuntimeError, match="not_a_parameter"):
        load_model_snapshot(str(bad), env=None, current_version=world["v_off"])


def test_set_parameters_REFUSES_exact_match_False_on_a_mismatch(world: Dict[str, Any]) -> None:
    from agents.training.instrumented_ppo.strict_load import StrictLoadError

    m = world["off"]
    sd = {k: v.clone() for k, v in m.policy.state_dict().items()}
    _drop("flat_intent_head")(sd)
    with pytest.raises(StrictLoadError, match="flat_intent_head"):
        m.set_parameters({"policy": sd}, exact_match=False)
    # a MATCHING set loads either way (the refusal is about the mismatch, not the flag)
    full = {k: v.clone() for k, v in m.policy.state_dict().items()}
    m.set_parameters({"policy": full, "policy.optimizer": m.policy.optimizer.state_dict()},
                     exact_match=False)


# ---------------------------------------------------------------- the READER loads (P10 follow-up F1)
def test_a_READER_load_REFUSES_a_checkpoint_missing_an_extractor_submodule(
        world: Dict[str, Any], tmp_path: Path) -> None:
    """P10 follow-up F1: the plain `MaskablePPO.load` callers (the ladder session, the prober, the eval
    worker, the offline meters) kept sb3's non-strict retry after P10-B closed it on the learner and the
    opponents. They all load through `load_checkpoint_strict` now. The precondition is the bug itself:
    sb3's own load takes the SAME zip, warns once, and leaves the deleted submodule at fresh init."""
    import warnings

    from sb3_contrib import MaskablePPO

    from agents.model.snapshot import load_checkpoint_strict
    from agents.training.instrumented_ppo.strict_load import StrictLoadError

    bad = _rewrite_policy(world["zip_off"], tmp_path / "reader", _drop("flat_intent_head"))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        bare = MaskablePPO.load(str(bad), env=None, device="cpu")           # the BUG: loads, silently
    assert bare is not None and any("SB3 < 1.7.0" in str(w.message) for w in seen)
    with pytest.raises(StrictLoadError, match="flat_intent_head"):
        load_checkpoint_strict(str(bad), device="cpu")


def test_the_strict_reader_load_is_otherwise_a_plain_MaskablePPO_load(world: Dict[str, Any]) -> None:
    """The strict class changes the load's STRICTNESS and nothing else: same weights as a bare load, an
    optimizer, the seed applied, a working `save` (`winprob_finetune`'s graft writes through one)."""
    import torch as torch_
    from sb3_contrib import MaskablePPO

    from agents.model.snapshot import load_checkpoint_strict
    from agents.training.instrumented_ppo.inference import InferenceMaskablePPO
    from agents.training.instrumented_ppo.strict_load import StrictMaskablePPO

    zip_off = str(world["zip_off"])
    strict = load_checkpoint_strict(zip_off, device="cpu")
    bare = MaskablePPO.load(zip_off, env=None, device="cpu")
    assert type(strict) is StrictMaskablePPO and isinstance(strict, MaskablePPO)
    assert not isinstance(strict, InferenceMaskablePPO)
    assert strict.policy.optimizer is not None
    sd_s, sd_b = strict.policy.state_dict(), bare.policy.state_dict()
    assert sd_s.keys() == sd_b.keys() and all(torch_.equal(sd_s[k], sd_b[k]) for k in sd_s)
    # the learner's own weights are what was saved, bit for bit
    sd_l = world["off"].policy.state_dict()
    assert all(torch_.equal(sd_s[k], sd_l[k]) for k in sd_l)


def test_the_reader_surfaces_refuse_a_dropped_submodule_and_the_prober_still_DIAGNOSES(
        world: Dict[str, Any], tmp_path: Path) -> None:
    """The ladder session's `play.load_policy` refuses it; the prober's `ProbeModel.load` still turns
    the refusal into its `ArchDriftError` diagnosis (the cause chained), never a bare crash."""
    from main.play import load_policy
    from main.prober.model import ArchDriftError, ProbeModel
    from agents.training.instrumented_ppo.strict_load import StrictLoadError

    bad = _rewrite_policy(world["zip_off"], tmp_path / "surfaces", _drop("win_head"))
    with pytest.raises(StrictLoadError, match="win_head"):
        load_policy(str(bad), "cpu")
    with pytest.raises(ArchDriftError) as ei:
        ProbeModel.load(str(bad))
    assert isinstance(ei.value.__cause__, StrictLoadError) and "win_head" in str(ei.value.__cause__)
    assert "StrictLoadError" in str(ei.value) and "state dict does not match" in str(ei.value)
