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
from agents.model.snapshot import save_model_snapshot, load_model_snapshot, write_checkpoint_sidecar, read_checkpoint_sidecar, _sidecar_path
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
    from sb3_contrib import MaskablePPO

    total_dim = layout["total_dim"]
    vec_env = _make_vec_env(total_dim)

    full_policy_kwargs = {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": {"layout": layout, "mappings": mappings},
        "net_arch": [512, 512],
    }
    model = MaskablePPO("MultiInputPolicy", vec_env, policy_kwargs=full_policy_kwargs, verbose=0)

    def _dummy_obs(m):
        dev = next(m.parameters()).device
        return {
            "observation": torch.zeros(1, total_dim, device=dev),
            "action_mask": torch.ones(1, 11, dtype=torch.int8, device=dev),
        }

    with torch.no_grad():
        features_before = model.policy.features_extractor(_dummy_obs(model.policy))

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
        features_after = loaded.policy.features_extractor(_dummy_obs(loaded.policy))

    assert features_before.shape == (1, PROJECTION_DIM)
    assert torch.allclose(features_before, features_after, atol=1e-6), (
        "Feature extractor output changed after save/load round-trip"
    )


# ---------------------------------------------------------------------------
# Checkpoint sidecar
# ---------------------------------------------------------------------------

def test_sidecar_path_strips_zip():
    assert _sidecar_path("/models/run/checkpoint_1000_steps.zip") == "/models/run/checkpoint_1000_steps.json"


def test_sidecar_path_no_zip_extension():
    assert _sidecar_path("/models/run/checkpoint_1000_steps") == "/models/run/checkpoint_1000_steps.json"


def test_write_and_read_sidecar():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "checkpoint_500000_steps.zip")
        write_checkpoint_sidecar(ckpt, current_lr=2.5e-5, current_epochs=7)
        result = read_checkpoint_sidecar(ckpt)
    assert result["current_lr"] == pytest.approx(2.5e-5)
    assert result["current_epochs"] == 7


def test_read_sidecar_returns_empty_when_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "checkpoint_500000_steps.zip")
        result = read_checkpoint_sidecar(ckpt)
    assert result == {}


def test_sidecar_overwrites_on_second_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "checkpoint_500000_steps.zip")
        write_checkpoint_sidecar(ckpt, current_lr=3e-4, current_epochs=10)
        write_checkpoint_sidecar(ckpt, current_lr=1e-5, current_epochs=4)
        result = read_checkpoint_sidecar(ckpt)
    assert result["current_lr"] == pytest.approx(1e-5)
    assert result["current_epochs"] == 4


def test_sidecar_independent_per_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_50m = os.path.join(tmpdir, "checkpoint_50000000_steps.zip")
        ckpt_100m = os.path.join(tmpdir, "checkpoint_100000000_steps.zip")
        write_checkpoint_sidecar(ckpt_50m, current_lr=3e-4, current_epochs=10)
        write_checkpoint_sidecar(ckpt_100m, current_lr=8e-5, current_epochs=6)
        r50 = read_checkpoint_sidecar(ckpt_50m)
        r100 = read_checkpoint_sidecar(ckpt_100m)
    assert r50["current_epochs"] == 10
    assert r100["current_epochs"] == 6


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
