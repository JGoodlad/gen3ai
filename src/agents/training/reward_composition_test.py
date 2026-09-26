"""`agents/training/reward_composition.py` — the recorded composition, and the INERT-flag list.

Since the shaped-reward deletion (2026-09-26) every config's census is `1 TERMINAL + 0 PBRS + 0 BIAS`;
what this file pins is that the announcer, the `metadata.json` block and the `model_config.json`
annotation still say so, in the schema their readers expect, and that the one remaining INERT case
(`draw_penalty` under `--terminal-indicator`) is named.

Run:
    pytest src/agents/training/reward_composition_test.py -q
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import json

import pytest

from agents.training.reward_composition import (
    format_reward_composition,
    inert_reward_flags,
    reward_class_composition,
    reward_composition_block,
)
from agents.training.reward_manager import RewardConfig

#: The production reward composition.
_WINPROB = dict(victory_value=1.0, draw_penalty=0.0, terminal_indicator=True)


def test_every_config_is_ONE_TERMINAL():
    for cfg in (RewardConfig(), RewardConfig(**_WINPROB)):
        comp = reward_class_composition(cfg)
        assert (comp["terminal"], comp["pbrs"], comp["bias"]) == (1, 0, 0)
        assert comp["terminal_terms"] == ["win_loss"]
        assert comp["pbrs_terms"] == comp["bias_terms"] == []


def test_draw_penalty_is_inert_under_the_indicator_terminal_and_LIVE_without_it():
    """Under `--terminal-indicator` the terminal pays `+victory_value` on a win and `0.0` on a
    loss, a tie AND a timeout alike, so the flag's NUMBER is never read."""
    assert inert_reward_flags(RewardConfig(**_WINPROB)) == ["draw_penalty"]
    assert inert_reward_flags(RewardConfig(victory_value=1.0, draw_penalty=-1.0)) == []


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE metadata.json BLOCK — additive, and it carries the ANNOUNCED line
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_block_is_ADDITIVE_over_the_census_every_existing_reader_uses():
    """`callbacks` and the `reward/` export read the three term lists. None of them may move."""
    for cfg in (RewardConfig(), RewardConfig(**_WINPROB)):
        census, block = reward_class_composition(cfg), reward_composition_block(cfg)
        for key, value in census.items():
            assert block[key] == value, f"{key} moved"
        assert set(block) - set(census) == {
            "composition_line", "class_shares", "inert_reward_flags"}


def test_the_block_carries_the_announcers_string_VERBATIM():
    """The launch PRINTS this and nothing kept it — a launcher rotates the child log away. It is
    the announcer's own output rather than a re-render, so the recorded line and the printed line
    cannot drift."""
    cfg = RewardConfig(**_WINPROB)
    line = reward_composition_block(cfg)["composition_line"]
    assert line == format_reward_composition(cfg)
    assert "1 TERMINAL + 0 PBRS + 0 BIAS" in line and "fully policy-invariant" in line


def test_the_class_shares_partition_the_active_terms():
    for cfg in (RewardConfig(), RewardConfig(**_WINPROB)):
        block = reward_composition_block(cfg)
        total = block["terminal"] + block["pbrs"] + block["bias"]
        assert sum(block["class_shares"].values()) == pytest.approx(1.0)
        for k in ("terminal", "pbrs", "bias"):
            assert block["class_shares"][k] == pytest.approx(block[k] / total)
    assert reward_composition_block(RewardConfig(**_WINPROB))["class_shares"]["terminal"] == 1.0


def test_the_block_is_JSON_SERIALIZABLE_because_metadata_json_carries_it():
    json.dumps(reward_composition_block(RewardConfig(**_WINPROB)))


def test_the_launch_records_the_BLOCK_and_not_the_bare_census():
    """The seam: `train_rl_agent` must build the richer block, or the announced line reaches
    nothing. Read off the source, because the alternative is a whole training run."""
    from main.train import entry_source

    src = entry_source()
    assert "reward_composition = reward_composition_block(reward_config)" in src
    assert "reward_class_composition(reward_config)" not in src, (
        "the bare census would drop the line, the shares and the inert list")


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE model_config.json ANNOTATION — beside the values, and it must not break a resume
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _real_layout():
    """The REAL encoder layout — `from_layout_and_policy_kwargs` reads the embedding dims off it,
    so a stub dict is not a shortcut, it is a KeyError. Cached, because building it loads the data
    singletons."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    return Gen3ObservationEncoder(load_mappings()).get_layout()


def _version_for(cfg):
    from agents.model.model_version import ModelVersion
    return ModelVersion.from_layout_and_policy_kwargs(
        _real_layout(), {"net_arch": [512, 512]}, reward_config=cfg)


def _write_config(tmp_path, cfg):
    from agents.model.snapshot import save_model_snapshot

    version = _version_for(cfg)
    save_model_snapshot(str(tmp_path), version)
    with open(tmp_path / "model_config.json") as f:
        return version, json.load(f)


def test_the_config_file_carries_the_annotation_BESIDE_the_values(tmp_path):
    _version, data = _write_config(tmp_path, RewardConfig(**_WINPROB))
    assert data["inert_reward_flags"] == ["draw_penalty"]
    assert data["draw_penalty"] == 0.0 and data["terminal_indicator"] is True
    assert "hand_shaping" not in data, "the deleted shaped fields must not be written"


def test_a_config_carrying_the_annotation_still_LOADS(tmp_path):
    """`from_json_file` does `cls(**data)`, which TypeErrors on a stale key — so the annotation is
    POPped by the version-INDEPENDENT sanitizer rather than becoming a field."""
    from agents.model.model_version import ModelVersion

    version, _data = _write_config(tmp_path, RewardConfig(**_WINPROB))
    loaded = ModelVersion.from_json_file(str(tmp_path / "model_config.json"))
    assert not hasattr(loaded, "inert_reward_flags")
    assert loaded.arch_signature == version.arch_signature


def test_the_annotated_config_RESUMES_under_its_own_reward_and_refuses_another(tmp_path):
    from agents.model.model_version import ModelVersion, ModelVersionError

    cfg = RewardConfig(**_WINPROB)
    _version, _data = _write_config(tmp_path, cfg)
    saved = ModelVersion.from_json_file(str(tmp_path / "model_config.json"))
    saved.check_reward_config(cfg)          # the resuming process rebuilds the same config
    with pytest.raises(ModelVersionError):
        saved.check_reward_config(RewardConfig(victory_value=30.0, draw_penalty=-35.0))


def test_a_SIGNED_config_records_the_key_too_so_ABSENT_never_reads_as_UNKNOWN(tmp_path):
    """An absent key and an empty list would otherwise be the same on disk."""
    _version, data = _write_config(tmp_path, RewardConfig())
    assert data["inert_reward_flags"] == []


def test_to_json_itself_stays_exactly_asdict(tmp_path):
    """Several tests round-trip `ModelVersion(**json.loads(v.to_json()))` with no migration, so the
    annotation belongs to the WRITER, not to the serializer."""
    import dataclasses

    from agents.model.model_version import ModelVersion

    v = _version_for(RewardConfig(**_WINPROB))
    assert json.loads(v.to_json()) == dataclasses.asdict(v)
    ModelVersion(**json.loads(v.to_json()))     # must not TypeError
