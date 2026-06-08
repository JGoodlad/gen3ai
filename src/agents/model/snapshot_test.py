import dataclasses
import json
import os
import tempfile

import numpy as np
import pytest
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.model.model_version import (
    MODEL_CONFIG_VERSION,
    ARCH_SIGNATURE,
    ModelVersion,
    ModelVersionError,
    _migrate_config,
)
from agents.model.snapshot import save_model_snapshot, load_model_snapshot, write_checkpoint_metadata, read_checkpoint_metadata, record_snapshot_in_history, record_checkpoint, record_eval_results, _checkpoint_metadata_path, _latest_checkpoint
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mappings():
    return load_mappings()


@pytest.fixture(scope="module")
def layout(mappings):
    return Gen3ObservationEncoder(mappings).get_layout()


@pytest.fixture(scope="module")
def policy_kwargs():
    return {"net_arch": [512, 512]}


@pytest.fixture(scope="module")
def version(layout, policy_kwargs):
    return ModelVersion.from_layout_and_policy_kwargs(layout, policy_kwargs)


# ---------------------------------------------------------------------------
# ModelVersion JSON round-trip
# ---------------------------------------------------------------------------

def test_model_version_json_roundtrip(version):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model_config.json")
        with open(path, "w") as f:
            f.write(version.to_json())
        loaded = ModelVersion.from_json_file(path)
    assert loaded == version


def test_model_version_all_fields_present(version):
    data = json.loads(version.to_json())
    expected_fields = [
        "config_version", "arch_signature",
        "species_embedding_dim", "max_species",
        "move_embedding_dim", "max_moves",
        "item_embedding_dim", "max_items",
        "ability_embedding_dim", "max_abilities",
        "type_embedding_dim", "max_types",
        "total_dim", "active_context_dim",
        "role_token_size", "projection_dim",
        "move_net_hidden", "role_encoder_hidden", "active_ctx_hidden",
        "n_history_turns",
        "net_arch",
        "vf_coef",
    ]
    for field in expected_fields:
        assert field in data, f"Missing field in serialized ModelVersion: {field}"


def test_n_history_turns_in_version(version):
    from agents.model.features_extractor import N_HISTORY_TURNS
    assert version.n_history_turns == N_HISTORY_TURNS


# ---------------------------------------------------------------------------
# check_compatible
# ---------------------------------------------------------------------------

def test_check_compatible_identical(version):
    version.check_compatible(dataclasses.replace(version))  # must not raise


def test_check_compatible_arch_signature_mismatch(version):
    bad = dataclasses.replace(version, arch_signature="gen3_lstm_v1")
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(bad)
    assert "gen3_lstm_v1" in str(exc_info.value)
    assert "Architecture family" in str(exc_info.value)


def test_check_compatible_total_dim_mismatch(version):
    bad = dataclasses.replace(version, total_dim=version.total_dim + 1)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(bad)
    assert "total_dim" in str(exc_info.value)
    assert str(version.total_dim) in str(exc_info.value)


def test_check_compatible_embedding_mismatch(version):
    bad = dataclasses.replace(version, max_species=999)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(bad)
    assert "max_species" in str(exc_info.value)


def test_check_compatible_net_arch_mismatch(version):
    bad = dataclasses.replace(version, net_arch=[256, 256])
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(bad)
    assert "net_arch" in str(exc_info.value)


def test_check_compatible_reports_all_mismatches(version):
    bad = dataclasses.replace(version, total_dim=9999, max_moves=9999)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(bad)
    msg = str(exc_info.value)
    assert "total_dim" in msg
    assert "max_moves" in msg


def test_check_compatible_n_history_turns_mismatch(version):
    """n_history_turns is a weight-relevant field — mismatch must raise ModelVersionError.

    A saved model trained with N=1 (single TurnDelta) is incompatible with current
    code running N=5, because the positional embedding and attention weights are
    indexed by N.
    """
    saved_v1 = dataclasses.replace(version, n_history_turns=1)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(saved_v1)
    assert "n_history_turns" in str(exc_info.value)


# ---------------------------------------------------------------------------
# vf_coef — resume-only value-meaning check (NOT part of check_compatible)
# ---------------------------------------------------------------------------

def test_check_compatible_ignores_vf_coef(version):
    """vf_coef is a training-loss coefficient, not a weight-shape field — check_compatible
    (which gates EVERY load, incl. frozen eval/pool/distill opponents) must ignore it."""
    differing = dataclasses.replace(version, vf_coef=version.vf_coef + 0.25)
    version.check_compatible(differing)  # must NOT raise


def test_check_vf_coef_match_does_not_raise(version):
    saved = dataclasses.replace(version, vf_coef=0.3)
    saved.check_vf_coef(0.3)  # must not raise


def test_check_vf_coef_mismatch_raises(version):
    saved = dataclasses.replace(version, vf_coef=0.3)
    with pytest.raises(ModelVersionError) as exc_info:
        saved.check_vf_coef(0.5)
    msg = str(exc_info.value)
    assert "vf_coef" in msg
    assert "0.3" in msg and "0.5" in msg


def test_check_vf_coef_tolerates_float_repr(version):
    """Equality is within tolerance, so a JSON round-trip of the same value passes."""
    saved = dataclasses.replace(version, vf_coef=0.1 + 0.2)  # 0.30000000000000004
    saved.check_vf_coef(0.3)  # must not raise


# ---------------------------------------------------------------------------
# check_reward_config — resume-only value-meaning check (NOT part of check_compatible)
# ---------------------------------------------------------------------------
def _reward_cfg(**kw):
    from agents.training.reward_manager import RewardConfig
    return RewardConfig(**kw)


def test_check_reward_config_match_does_not_raise(version):
    saved = dataclasses.replace(version, bias_additivity=0.5, mat_alive_weight=1.25,
                                bias_redesign=False, switch_bias_weight=1.5)
    saved.check_reward_config(_reward_cfg(bias_additivity=0.5, mat_alive_weight=1.25,
                                          bias_redesign=False, switch_bias_weight=1.5))  # no raise


def test_check_reward_config_switch_bias_weight_mismatch_raises(version):
    """switch_bias_weight is resume-immutable (it changes the objective) — a drift must FATAL."""
    saved = dataclasses.replace(version, switch_bias_weight=1.5)
    with pytest.raises(ModelVersionError) as exc_info:
        saved.check_reward_config(_reward_cfg(switch_bias_weight=0.0))
    assert "switch_bias_weight" in str(exc_info.value)


def test_check_reward_config_default_off_matches(version):
    """A fresh default run (lever OFF) matches a default-OFF saved config."""
    dataclasses.replace(version, switch_bias_weight=0.0).check_reward_config(_reward_cfg())  # no raise


def test_check_compatible_ignores_switch_bias_weight(version):
    """Like vf_coef, switch_bias_weight is value-meaning, NOT weight-shape — frozen eval / pool /
    distill loads (which go through check_compatible) must accept any value."""
    differing = dataclasses.replace(version, switch_bias_weight=version.switch_bias_weight + 1.0)
    version.check_compatible(differing)  # must not raise


# ---------------------------------------------------------------------------
# save_model_snapshot
# ---------------------------------------------------------------------------

def test_save_snapshot_creates_files(version):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc123")
        assert os.path.exists(os.path.join(tmpdir, "model_config.json"))
        assert os.path.exists(os.path.join(tmpdir, "metadata.json"))


def test_save_snapshot_metadata_fields(version):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="deadbeef")
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert meta["git_hash"] == "deadbeef"
    assert "saved_at" in meta
    assert "python_version" in meta
    assert "sb3_version" in meta


# ---------------------------------------------------------------------------
# Full save → load round-trip
# ---------------------------------------------------------------------------

def _make_vec_env(total_dim: int) -> DummyVecEnv:
    _obs_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        ),
        "action_mask": gym.spaces.MultiBinary(11),
    })
    _act_space = gym.spaces.Discrete(11)

    class _TrivialEnv(gym.Env):
        observation_space = _obs_space
        action_space = _act_space

        def reset(self, **kwargs):
            return _obs_space.sample(), {}

        def step(self, action):
            return _obs_space.sample(), 0.0, False, False, {}

    return DummyVecEnv([_TrivialEnv])


def test_snapshot_save_load_roundtrip(layout, version, mappings):
    from agents.model.features_extractor import Gen3FeaturesExtractor, PROJECTION_DIM
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from sb3_contrib import MaskablePPO

    total_dim = layout["total_dim"]
    vec_env = _make_vec_env(total_dim)

    full_policy_kwargs = {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": {"layout": layout, "mappings": mappings},
        "net_arch": [512, 512],
    }
    model = MaskablePPO(Gen3DualHeadMaskablePolicy, vec_env, policy_kwargs=full_policy_kwargs, verbose=0)

    def _dummy_obs(m):
        dev = next(m.parameters()).device
        return {
            "observation": torch.zeros(1, total_dim, device=dev),
            "action_mask": torch.ones(1, 11, dtype=torch.int8, device=dev),
        }

    with torch.no_grad():
        pi_before, vf_before = model.policy.features_extractor(_dummy_obs(model.policy))

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "test_model")
        model.save(zip_path)
        save_model_snapshot(tmpdir, version, git_hash="test")

        loaded = load_model_snapshot(
            zip_path + ".zip",
            env=vec_env,
            current_version=version,
        )

    with torch.no_grad():
        pi_after, vf_after = loaded.policy.features_extractor(_dummy_obs(loaded.policy))

    # Both heads of the dual-head extractor must reproduce exactly across save/load.
    assert pi_before.shape == (1, PROJECTION_DIM)
    assert vf_before.shape == (1, PROJECTION_DIM)
    assert torch.allclose(pi_before, pi_after, atol=1e-6), (
        "Policy feature output changed after save/load round-trip"
    )
    assert torch.allclose(vf_before, vf_after, atol=1e-6), (
        "Value feature output changed after save/load round-trip"
    )


def test_load_model_snapshot_enforce_vf_coef(layout, mappings):
    """load_model_snapshot(enforce_vf_coef=...) is the resume guard: it must FATAL when the
    saved config's vf_coef differs, and load cleanly when it matches. Frozen-snapshot loads
    (enforce_vf_coef=None) must ignore vf_coef entirely."""
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from sb3_contrib import MaskablePPO

    total_dim = layout["total_dim"]
    vec_env = _make_vec_env(total_dim)
    full_policy_kwargs = {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": {"layout": layout, "mappings": mappings},
        "net_arch": [512, 512],
    }
    model = MaskablePPO(Gen3DualHeadMaskablePolicy, vec_env, policy_kwargs=full_policy_kwargs, verbose=0)

    # The run was started with vf_coef=0.3 → that is what model_config.json records.
    saved_version = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]}, vf_coef=0.3)

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "test_model")
        model.save(zip_path)
        save_model_snapshot(tmpdir, saved_version, git_hash="test")

        # Resuming with a different vf_coef is a hard error (before the model even loads).
        with pytest.raises(ModelVersionError) as exc_info:
            load_model_snapshot(zip_path + ".zip", env=vec_env, current_version=saved_version,
                                enforce_vf_coef=0.5)
        assert "vf_coef" in str(exc_info.value)

        # Matching value loads fine.
        load_model_snapshot(zip_path + ".zip", env=vec_env, current_version=saved_version,
                            enforce_vf_coef=0.3)

        # Frozen-snapshot load (no enforcement) ignores vf_coef even when it would differ.
        load_model_snapshot(zip_path + ".zip", env=vec_env, current_version=saved_version)


# ---------------------------------------------------------------------------
# Checkpoint metadata
# ---------------------------------------------------------------------------

def test_checkpoint_metadata_path_strips_zip():
    assert _checkpoint_metadata_path("/models/run/checkpoint_1000_steps.zip") == "/models/run/checkpoint_1000_steps.json"


def test_checkpoint_metadata_path_no_zip_extension():
    assert _checkpoint_metadata_path("/models/run/checkpoint_1000_steps") == "/models/run/checkpoint_1000_steps.json"


def test_write_and_read_checkpoint_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "checkpoint_500000_steps.zip")
        write_checkpoint_metadata(ckpt, lr=2.5e-5, n_epochs=7)
        result = read_checkpoint_metadata(ckpt)
    # Union of both naming conventions — neither name is dropped.
    assert result["lr"] == pytest.approx(2.5e-5)
    assert result["current_lr"] == pytest.approx(2.5e-5)
    assert result["n_epochs"] == 7
    assert result["current_epochs"] == 7


def test_read_checkpoint_metadata_returns_empty_when_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "checkpoint_500000_steps.zip")
        result = read_checkpoint_metadata(ckpt)
    assert result == {}


def test_checkpoint_metadata_overwrites_on_second_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "checkpoint_500000_steps.zip")
        write_checkpoint_metadata(ckpt, lr=3e-4, n_epochs=10)
        write_checkpoint_metadata(ckpt, lr=1e-5, n_epochs=4)
        result = read_checkpoint_metadata(ckpt)
    assert result["lr"] == pytest.approx(1e-5)
    assert result["current_lr"] == pytest.approx(1e-5)
    assert result["n_epochs"] == 4
    assert result["current_epochs"] == 4


def test_checkpoint_metadata_independent_per_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_50m = os.path.join(tmpdir, "checkpoint_50000000_steps.zip")
        ckpt_100m = os.path.join(tmpdir, "checkpoint_100000000_steps.zip")
        write_checkpoint_metadata(ckpt_50m, lr=3e-4, n_epochs=10)
        write_checkpoint_metadata(ckpt_100m, lr=8e-5, n_epochs=6)
        r50 = read_checkpoint_metadata(ckpt_50m)
        r100 = read_checkpoint_metadata(ckpt_100m)
    assert r50["n_epochs"] == r50["current_epochs"] == 10
    assert r100["n_epochs"] == r100["current_epochs"] == 6


# ---------------------------------------------------------------------------
# snapshot_history
# ---------------------------------------------------------------------------

def test_record_snapshot_creates_history(version):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc")
        record_snapshot_in_history(tmpdir, "checkpoint_50000000_steps.zip", lr=2.5e-4, n_epochs=10)
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert "snapshot_history" in meta
    entry = meta["snapshot_history"]["checkpoint_50000000_steps.zip"]
    # History entries carry the same union of names as the per-checkpoint sidecar.
    assert entry["lr"] == pytest.approx(2.5e-4)
    assert entry["current_lr"] == pytest.approx(2.5e-4)
    assert entry["n_epochs"] == 10
    assert entry["current_epochs"] == 10


def test_record_snapshot_accumulates_multiple_entries(version):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc")
        record_snapshot_in_history(tmpdir, "checkpoint_50000000_steps.zip", lr=3e-4, n_epochs=10)
        record_snapshot_in_history(tmpdir, "checkpoint_100000000_steps.zip", lr=8e-5, n_epochs=7)
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            history = json.load(f)["snapshot_history"]
    assert len(history) == 2
    assert history["checkpoint_50000000_steps.zip"]["n_epochs"] == 10
    assert history["checkpoint_100000000_steps.zip"]["n_epochs"] == 7


def test_save_model_snapshot_preserves_history(version):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc")
        record_snapshot_in_history(tmpdir, "checkpoint_50000000_steps.zip", lr=3e-4, n_epochs=10)
        # Simulate a subsequent save_model_snapshot call (e.g. signal handler on restart)
        save_model_snapshot(tmpdir, version, git_hash="def", current_lr=8e-5, current_epochs=7)
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert "snapshot_history" in meta
    assert "checkpoint_50000000_steps.zip" in meta["snapshot_history"]
    # New fields also present
    assert meta["current_lr"] == pytest.approx(8e-5)
    assert meta["current_epochs"] == 7


def test_record_snapshot_creates_metadata_if_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        # No prior save_model_snapshot call
        record_snapshot_in_history(tmpdir, "checkpoint_50000000_steps.zip", lr=3e-4, n_epochs=10)
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert meta["snapshot_history"]["checkpoint_50000000_steps.zip"]["lr"] == pytest.approx(3e-4)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

_SAMPLE_HPARAMS = {
    "gamma": 0.9999,
    "gae_lambda": 0.8,
    "ent_coef": 0.02,
    "batch_size": 16384,
    "n_steps": 2048,
    "clip_range": 0.2,
}


# ---------------------------------------------------------------------------
# hparams propagation
# ---------------------------------------------------------------------------

def test_write_checkpoint_metadata_includes_hparams():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "checkpoint_500000_steps.zip")
        write_checkpoint_metadata(ckpt, lr=2.5e-5, n_epochs=7, hparams=_SAMPLE_HPARAMS, git_hash="deadbeef")
        result = read_checkpoint_metadata(ckpt)
    for key, val in _SAMPLE_HPARAMS.items():
        assert result[key] == pytest.approx(val), f"hparams[{key!r}] not written correctly"
    assert result["git_hash"] == "deadbeef"


def test_write_checkpoint_metadata_lr_epochs_win_over_hparams():
    """Both lr/n_epochs naming conventions must beat conflicting hparams keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "checkpoint_500000_steps.zip")
        write_checkpoint_metadata(
            ckpt,
            lr=1.23e-5,
            n_epochs=9,
            hparams={"lr": 999.0, "n_epochs": 999, "current_lr": 999.0, "current_epochs": 999},
        )
        result = read_checkpoint_metadata(ckpt)
    assert result["lr"] == pytest.approx(1.23e-5)
    assert result["current_lr"] == pytest.approx(1.23e-5)
    assert result["n_epochs"] == 9
    assert result["current_epochs"] == 9


def test_record_snapshot_in_history_includes_hparams(version):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc")
        record_snapshot_in_history(
            tmpdir, "checkpoint_50000000_steps.zip",
            lr=2.5e-4, n_epochs=10, hparams=_SAMPLE_HPARAMS, git_hash="cafebabe",
        )
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            entry = json.load(f)["snapshot_history"]["checkpoint_50000000_steps.zip"]
    for key, val in _SAMPLE_HPARAMS.items():
        assert entry[key] == pytest.approx(val), f"history entry missing hparams[{key!r}]"
    assert entry["git_hash"] == "cafebabe"


def test_record_checkpoint_propagates_hparams(version):
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "checkpoint_50000000_steps.zip")
        record_checkpoint(tmpdir, ckpt, lr=3e-4, n_epochs=10, hparams=_SAMPLE_HPARAMS, git_hash="aabbccdd")

        per_ckpt = read_checkpoint_metadata(ckpt)
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            history_entry = json.load(f)["snapshot_history"]["checkpoint_50000000_steps.zip"]

    for key, val in _SAMPLE_HPARAMS.items():
        assert per_ckpt[key] == pytest.approx(val), f"per-checkpoint JSON missing hparams[{key!r}]"
        assert history_entry[key] == pytest.approx(val), f"history entry missing hparams[{key!r}]"
    assert per_ckpt["git_hash"] == "aabbccdd"
    assert history_entry["git_hash"] == "aabbccdd"


def test_save_model_snapshot_includes_hparams(version):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc", hparams=_SAMPLE_HPARAMS)
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    for key, val in _SAMPLE_HPARAMS.items():
        assert meta[key] == pytest.approx(val), f"metadata.json missing hparams[{key!r}]"


def test_save_model_snapshot_lr_epochs_win_over_hparams(version):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(
            tmpdir, version, git_hash="abc",
            current_lr=1.23e-5, current_epochs=9,
            hparams={"current_lr": 999.0, "current_epochs": 999},
        )
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert meta["current_lr"] == pytest.approx(1.23e-5)
    assert meta["current_epochs"] == 9


def test_hparams_none_does_not_affect_output(version):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc", hparams=None)
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert "gamma" not in meta
    assert "gae_lambda" not in meta


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_migrate_config_noop_on_current(version):
    data = json.loads(version.to_json())
    migrated = _migrate_config(data)
    assert migrated == data


def test_migrate_v1_adds_n_history_turns_with_default_1():
    """v1 configs lack n_history_turns (single-TurnDelta era). Migration must inject 1."""
    data = {
        "config_version": 1,
        "arch_signature": ARCH_SIGNATURE,
        "species_embedding_dim": 32, "max_species": 400,
        "move_embedding_dim": 16, "max_moves": 400,
        "item_embedding_dim": 16, "max_items": 600,
        "ability_embedding_dim": 16, "max_abilities": 100,
        "type_embedding_dim": 16, "max_types": 20,
        "total_dim": 1000, "active_context_dim": 22,
        "role_token_size": 128, "projection_dim": 512,
        "move_net_hidden": [64, 32],
        "role_encoder_hidden": [256, 128],
        "active_ctx_hidden": [64, 32],
        "net_arch": [512, 512],
    }
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["n_history_turns"] == 1


def test_migrate_v2_adds_vf_coef_default():
    """v2 configs predate the vf_coef flag — migration must inject the SB3 default 0.5
    (the value every pre-flag run was trained with) and bump to the current version."""
    data = {
        "config_version": 2,
        "arch_signature": ARCH_SIGNATURE,
        "species_embedding_dim": 32, "max_species": 400,
        "move_embedding_dim": 16, "max_moves": 400,
        "item_embedding_dim": 16, "max_items": 600,
        "ability_embedding_dim": 16, "max_abilities": 100,
        "type_embedding_dim": 16, "max_types": 20,
        "total_dim": 1000, "active_context_dim": 22,
        "role_token_size": 128, "projection_dim": 512,
        "move_net_hidden": [64, 32],
        "role_encoder_hidden": [256, 128],
        "active_ctx_hidden": [64, 32],
        "n_history_turns": 10,
        "net_arch": [512, 512],
    }
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["vf_coef"] == pytest.approx(0.5)
    # The migrated dict must construct a valid ModelVersion (no unexpected keys).
    ModelVersion(**result)


def test_migrate_does_not_overwrite_existing_vf_coef():
    """A config that already carries vf_coef must keep its value through migration."""
    data = {
        "config_version": 2,
        "arch_signature": ARCH_SIGNATURE,
        "species_embedding_dim": 32, "max_species": 400,
        "move_embedding_dim": 16, "max_moves": 400,
        "item_embedding_dim": 16, "max_items": 600,
        "ability_embedding_dim": 16, "max_abilities": 100,
        "type_embedding_dim": 16, "max_types": 20,
        "total_dim": 1000, "active_context_dim": 22,
        "role_token_size": 128, "projection_dim": 512,
        "move_net_hidden": [64, 32],
        "role_encoder_hidden": [256, 128],
        "active_ctx_hidden": [64, 32],
        "n_history_turns": 10,
        "net_arch": [512, 512],
        "vf_coef": 0.25,
    }
    result = _migrate_config(data)
    assert result["vf_coef"] == pytest.approx(0.25)


def test_migrate_v1_does_not_overwrite_existing_n_history_turns():
    """If a v1 config somehow already has n_history_turns, migration must not clobber it."""
    data = {
        "config_version": 1,
        "arch_signature": ARCH_SIGNATURE,
        "species_embedding_dim": 32, "max_species": 400,
        "move_embedding_dim": 16, "max_moves": 400,
        "item_embedding_dim": 16, "max_items": 600,
        "ability_embedding_dim": 16, "max_abilities": 100,
        "type_embedding_dim": 16, "max_types": 20,
        "total_dim": 1000, "active_context_dim": 22,
        "role_token_size": 128, "projection_dim": 512,
        "move_net_hidden": [64, 32],
        "role_encoder_hidden": [256, 128],
        "active_ctx_hidden": [64, 32],
        "net_arch": [512, 512],
        "n_history_turns": 3,   # already present — must survive migration
    }
    result = _migrate_config(data)
    assert result["n_history_turns"] == 3


# ---------------------------------------------------------------------------
# _latest_checkpoint
# ---------------------------------------------------------------------------

def test_latest_checkpoint_returns_highest_step():
    history = {
        "checkpoint_50000000_steps.zip": {},
        "checkpoint_200000000_steps.zip": {},
        "checkpoint_100000000_steps.zip": {},
    }
    assert _latest_checkpoint(history) == "checkpoint_200000000_steps.zip"


def test_latest_checkpoint_single_entry():
    history = {"checkpoint_75000000_steps.zip": {}}
    assert _latest_checkpoint(history) == "checkpoint_75000000_steps.zip"


def test_latest_checkpoint_empty_returns_none():
    assert _latest_checkpoint({}) is None


def test_latest_checkpoint_skips_unrecognized_names():
    history = {
        "final_model.zip": {},
        "checkpoint_50000000_steps.zip": {},
    }
    assert _latest_checkpoint(history) == "checkpoint_50000000_steps.zip"


# ---------------------------------------------------------------------------
# record_eval_results
# ---------------------------------------------------------------------------

_SAMPLE_EVALS = {
    "win_rate_mean": 0.78,
    "win_rate_vs_bots": 0.74,
    "mean_reward_vs_bots": 0.24,
    "mean_ep_len_vs_bots": 22.1,
    "opponents": {
        "random": {"win_rate": 0.95, "mean_reward": 0.82, "mean_ep_len": 14.3},
        "heuristic": {"win_rate": 0.72, "mean_reward": 0.23, "mean_ep_len": 22.1},
    },
}


def test_record_eval_results_writes_top_level_latest_eval():
    with tempfile.TemporaryDirectory() as tmpdir:
        record_snapshot_in_history(tmpdir, "checkpoint_50000000_steps.zip", lr=3e-4, n_epochs=10)
        record_eval_results(tmpdir, step=51_000_000, metrics=_SAMPLE_EVALS)
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    ev = meta["latest_eval"]  # top-level, NOT nested under a checkpoint
    assert ev["step"] == 51_000_000
    assert "evaluated_at" in ev
    assert ev["win_rate_vs_bots"] == pytest.approx(0.74)
    assert ev["opponents"]["heuristic"]["win_rate"] == pytest.approx(0.72)
    # Not bound to any checkpoint entry.
    assert "evals" not in meta["snapshot_history"]["checkpoint_50000000_steps.zip"]


def test_record_eval_results_independent_of_checkpoints():
    # The eval is for a frozen snapshot; even with a newer checkpoint present it is
    # NOT attached to it — it lives top-level, labeled by its own snapshot step.
    with tempfile.TemporaryDirectory() as tmpdir:
        record_snapshot_in_history(tmpdir, "checkpoint_50000000_steps.zip", lr=3e-4, n_epochs=10)
        record_snapshot_in_history(tmpdir, "checkpoint_100000000_steps.zip", lr=1e-4, n_epochs=8)
        record_eval_results(tmpdir, step=60_000_000, metrics={"win_rate_vs_bots": 0.8})
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert meta["latest_eval"]["step"] == 60_000_000
    assert meta["latest_eval"]["win_rate_vs_bots"] == pytest.approx(0.8)
    for ck in meta["snapshot_history"].values():
        assert "evals" not in ck


def test_record_eval_results_overwrites_previous():
    with tempfile.TemporaryDirectory() as tmpdir:
        record_eval_results(tmpdir, step=51_000_000, metrics={"win_rate_vs_bots": 0.5})
        record_eval_results(tmpdir, step=52_000_000, metrics={"win_rate_vs_bots": 0.75})
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            ev = json.load(f)["latest_eval"]
    assert ev["step"] == 52_000_000
    assert ev["win_rate_vs_bots"] == pytest.approx(0.75)


def test_record_eval_results_writes_without_any_checkpoint():
    # The whole point of top-level storage: an early eval (before any checkpoint)
    # must still be persisted, not silently dropped.
    with tempfile.TemporaryDirectory() as tmpdir:
        record_eval_results(tmpdir, step=500_000, metrics={"win_rate_vs_bots": 0.6})
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert meta["latest_eval"]["step"] == 500_000
    assert meta["latest_eval"]["win_rate_vs_bots"] == pytest.approx(0.6)


def test_save_model_snapshot_preserves_latest_eval(version):
    """A checkpoint saved after an eval must NOT erase the top-level latest_eval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        record_snapshot_in_history(tmpdir, "checkpoint_50000000_steps.zip", lr=3e-4, n_epochs=10)
        record_eval_results(tmpdir, step=51_000_000, metrics=_SAMPLE_EVALS)
        save_model_snapshot(tmpdir, version, git_hash="def", current_lr=1e-5)
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert meta["latest_eval"]["step"] == 51_000_000
    assert meta["latest_eval"]["win_rate_vs_bots"] == pytest.approx(0.74)
    assert meta["git_hash"] == "def"
    assert meta["current_lr"] == pytest.approx(1e-5)
    assert "evals" not in meta  # must not appear at top level


def test_save_model_snapshot_no_top_level_evals_key(version):
    """save_model_snapshot must never write a top-level 'evals' key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc")
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert "evals" not in meta


# ---------------------------------------------------------------------------
# Per-checkpoint latest_eval stamp (record_checkpoint snapshots the current
# top-level latest_eval into the sidecar + snapshot_history at save time)
# ---------------------------------------------------------------------------

_SAMPLE_EVALS_WITH_POOL = {
    **_SAMPLE_EVALS,
    "pool": {
        "win_rate": 0.61,
        "mean_reward": 0.11,
        "snapshot_count": 4,
        "monotonicity": 0.83,
        "sentinels": [
            {"step": 40_000_000, "win_rate": 0.55, "snapshot": "snapshot_000040000000.zip"},
            {"step": 50_000_000, "win_rate": 0.67, "snapshot": "snapshot_000050000000.zip"},
        ],
    },
}


def test_record_checkpoint_stamps_latest_eval_in_sidecar_and_history(version):
    """After an eval, record_checkpoint embeds the eval+pool block in BOTH the
    per-checkpoint sidecar and the snapshot_history entry under 'latest_eval'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc")
        record_eval_results(tmpdir, step=49_000_000, metrics=_SAMPLE_EVALS_WITH_POOL)

        ckpt = os.path.join(tmpdir, "checkpoint_50000000_steps.zip")
        record_checkpoint(tmpdir, ckpt, lr=3e-4, n_epochs=10, git_hash="cafe")

        per_ckpt = read_checkpoint_metadata(ckpt)
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            history_entry = json.load(f)["snapshot_history"]["checkpoint_50000000_steps.zip"]

    for entry, where in ((per_ckpt, "sidecar"), (history_entry, "history")):
        ev = entry["latest_eval"]
        assert ev["step"] == 49_000_000, where
        assert ev["win_rate_vs_bots"] == pytest.approx(0.74), where
        assert ev["opponents"]["heuristic"]["win_rate"] == pytest.approx(0.72), where
        # The pool-play results ride along verbatim.
        assert ev["pool"]["win_rate"] == pytest.approx(0.61), where
        assert ev["pool"]["sentinels"][1]["step"] == 50_000_000, where


def test_record_checkpoint_no_eval_stamp_before_first_eval(version):
    """A checkpoint saved before any eval has run carries no latest_eval key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc")
        ckpt = os.path.join(tmpdir, "checkpoint_50000000_steps.zip")
        record_checkpoint(tmpdir, ckpt, lr=3e-4, n_epochs=10)

        per_ckpt = read_checkpoint_metadata(ckpt)
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            history_entry = json.load(f)["snapshot_history"]["checkpoint_50000000_steps.zip"]

    assert "latest_eval" not in per_ckpt
    assert "latest_eval" not in history_entry


def test_record_checkpoint_eval_stamp_is_point_in_time(version):
    """The stamp is the eval as-of save time: a later checkpoint catches a newer
    eval, while the earlier checkpoint keeps the one it was saved with."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc")

        record_eval_results(tmpdir, step=48_000_000, metrics={"win_rate_vs_bots": 0.50})
        ckpt_a = os.path.join(tmpdir, "checkpoint_50000000_steps.zip")
        record_checkpoint(tmpdir, ckpt_a, lr=3e-4, n_epochs=10)

        record_eval_results(tmpdir, step=98_000_000, metrics={"win_rate_vs_bots": 0.80})
        ckpt_b = os.path.join(tmpdir, "checkpoint_100000000_steps.zip")
        record_checkpoint(tmpdir, ckpt_b, lr=2e-4, n_epochs=8)

        # The earlier checkpoint's sidecar is frozen at its save-time eval.
        sidecar_a = read_checkpoint_metadata(ckpt_a)
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)

    hist = meta["snapshot_history"]
    assert hist["checkpoint_50000000_steps.zip"]["latest_eval"]["step"] == 48_000_000
    assert hist["checkpoint_100000000_steps.zip"]["latest_eval"]["step"] == 98_000_000
    assert sidecar_a["latest_eval"]["win_rate_vs_bots"] == pytest.approx(0.50)
    # The canonical top-level block still tracks the newest eval.
    assert meta["latest_eval"]["step"] == 98_000_000
