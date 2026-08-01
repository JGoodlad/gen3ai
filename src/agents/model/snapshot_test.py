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
import agents.model.snapshot as snapshot
from agents.model.snapshot import save_model_snapshot, load_model_snapshot, load_foreign_opponent, write_checkpoint_metadata, read_checkpoint_metadata, record_snapshot_in_history, record_checkpoint, record_eval_results, _checkpoint_metadata_path, _latest_checkpoint
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
# belief_grad_mode — resume-only training-signal check (NOT part of check_compatible)
# ---------------------------------------------------------------------------

def test_check_compatible_ignores_belief_grad_mode(version):
    """belief_grad_mode is a training-gradient knob, not a weight-shape field — detach() is value-
    preserving so a frozen eval/pool/distill forward is bit-identical either way. check_compatible
    (which gates EVERY load) must ignore it, else self-play would FATAL on its own snapshots."""
    differing = dataclasses.replace(version, belief_grad_mode="detached")
    version.check_compatible(differing)  # version is shaping (default) — must NOT raise


def test_check_belief_grad_mode_match_does_not_raise(version):
    saved = dataclasses.replace(version, belief_grad_mode="detached")
    saved.check_belief_grad_mode("detached")  # must not raise


def test_check_belief_grad_mode_mismatch_raises(version):
    """Flipping shaping↔detached mid-run silently changes whether the belief reshapes the trunk —
    a drift on resume must FATAL, not change the training signal quietly."""
    saved = dataclasses.replace(version, belief_grad_mode="detached")
    with pytest.raises(ModelVersionError) as exc_info:
        saved.check_belief_grad_mode("shaping")
    msg = str(exc_info.value)
    assert "belief_grad_mode" in msg
    assert "detached" in msg and "shaping" in msg


def test_belief_grad_mode_read_from_features_extractor_kwargs(layout):
    pk = {"net_arch": [512, 512], "features_extractor_kwargs": {"belief_grad_mode": "detached"}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.belief_grad_mode == "detached" and v.config_version == MODEL_CONFIG_VERSION
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.belief_grad_mode == "shaping"   # default = the legacy trunk-shaping behaviour


def test_migrate_pre_v41_adds_belief_grad_mode_default(version):
    """Pre-v41 configs lack belief_grad_mode — migration injects 'shaping' (byte-identical legacy
    behaviour) and bumps to current."""
    data = json.loads(version.to_json())
    data.pop("belief_grad_mode", None)
    data["config_version"] = 40
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["belief_grad_mode"] == "shaping"
    ModelVersion(**result)


def test_check_value_tail_weight_match_and_mismatch(version):
    """② value_tail_weight is resume-immutable (like vf_coef): a matching resume passes, a drift FATALs."""
    saved = dataclasses.replace(version, value_tail_weight=0.3)
    saved.check_value_tail_weight(0.3)               # must not raise
    with pytest.raises(ModelVersionError) as exc_info:
        saved.check_value_tail_weight(0.0)
    assert "value_tail_weight" in str(exc_info.value)


def test_check_compatible_ignores_value_tail_weight(version):
    """value_tail_weight is a value-loss hparam, not weight-shape → check_compatible (which gates frozen
    eval/pool/distill opponents) must IGNORE it, exactly like vf_coef."""
    differing = dataclasses.replace(version, value_tail_weight=version.value_tail_weight + 0.5)
    version.check_compatible(differing)  # must NOT raise


def test_value_tail_weight_recorded_and_config_version(layout):
    pk = {"net_arch": [512, 512]}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk, value_tail_weight=0.4)
    assert v.value_tail_weight == 0.4 and v.config_version == MODEL_CONFIG_VERSION
    v0 = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v0.value_tail_weight == 0.0   # default = plain MSE


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


def test_check_reward_config_draw_penalty_mismatch_raises(version):
    """draw_penalty is resume-immutable (it changes the terminal reward) — a drift must FATAL."""
    saved = dataclasses.replace(version, draw_penalty=-35.0)
    with pytest.raises(ModelVersionError) as exc_info:
        saved.check_reward_config(_reward_cfg(draw_penalty=-30.0))
    assert "draw_penalty" in str(exc_info.value)


def test_check_reward_config_draw_penalty_default_matches(version):
    """A fresh default run (draw_penalty -30) matches a default saved config (no raise)."""
    dataclasses.replace(version, draw_penalty=-30.0).check_reward_config(_reward_cfg())


def test_check_compatible_ignores_draw_penalty(version):
    """draw_penalty is value-meaning, NOT weight-shape — frozen eval / pool / distill loads
    (which go through check_compatible) must accept any value."""
    differing = dataclasses.replace(version, draw_penalty=version.draw_penalty - 5.0)
    version.check_compatible(differing)  # must not raise


@pytest.mark.parametrize("field", ["drop_redundant_bias", "drop_switch_bias"])
def test_check_reward_config_debias_flag_mismatch_raises(version, field):
    """The de-bias cleanup flags are resume-immutable (they change the reward) — a drift must FATAL."""
    saved = dataclasses.replace(version, **{field: True})
    with pytest.raises(ModelVersionError) as exc_info:
        saved.check_reward_config(_reward_cfg(**{field: False}))
    assert field in str(exc_info.value)


def test_check_reward_config_debias_default_off_matches(version):
    """A fresh default run (both flags OFF) matches a default-OFF saved config (no raise)."""
    dataclasses.replace(version, drop_redundant_bias=False,
                        drop_switch_bias=False).check_reward_config(_reward_cfg())


@pytest.mark.parametrize("field", ["drop_redundant_bias", "drop_switch_bias"])
def test_check_compatible_ignores_debias_flags(version, field):
    """The de-bias flags are value-meaning, NOT weight-shape — frozen eval / pool / distill loads
    (which go through check_compatible) must accept any value."""
    differing = dataclasses.replace(version, **{field: not getattr(version, field)})
    version.check_compatible(differing)  # must not raise


def test_migrate_v11_adds_debias_flags_default(version):
    """Pre-v12 configs lack the de-bias flags — migration injects False (prior behavior) and bumps."""
    data = json.loads(version.to_json())
    data.pop("drop_redundant_bias", None)
    data.pop("drop_switch_bias", None)
    data["config_version"] = 11
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["drop_redundant_bias"] is False and result["drop_switch_bias"] is False
    ModelVersion(**result)


# ---------------------------------------------------------------------------
# v13 — all_shaping_pbrs (end-state PBRS switch) + no_progress_penalty (now recorded)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["all_shaping_pbrs", "no_progress_penalty"])
def test_check_reward_config_v13_field_mismatch_raises(version, field):
    """all_shaping_pbrs + no_progress_penalty are resume-immutable (they change the reward) — drift FATAL."""
    diff = {"all_shaping_pbrs": True, "no_progress_penalty": 0.30}[field]
    saved = dataclasses.replace(version, **{field: diff})
    with pytest.raises(ModelVersionError) as exc_info:
        saved.check_reward_config(_reward_cfg(**{field: {"all_shaping_pbrs": False,
                                                         "no_progress_penalty": 0.15}[field]}))
    assert field in str(exc_info.value)


def test_check_reward_config_v13_default_matches(version):
    """A fresh default run (flag OFF, penalty 0.15) matches a default-OFF saved config (no raise)."""
    dataclasses.replace(version, all_shaping_pbrs=False,
                        no_progress_penalty=0.15).check_reward_config(_reward_cfg())


@pytest.mark.parametrize("field", ["all_shaping_pbrs", "no_progress_penalty"])
def test_check_compatible_ignores_v13_fields(version, field):
    """all_shaping_pbrs + no_progress_penalty are value-meaning, NOT weight-shape — frozen eval / pool /
    distill loads (which go through check_compatible) must accept any value."""
    diff = {"all_shaping_pbrs": not version.all_shaping_pbrs,
            "no_progress_penalty": version.no_progress_penalty + 0.1}[field]
    version.check_compatible(dataclasses.replace(version, **{field: diff}))  # must not raise


def test_all_shaping_pbrs_recorded_and_config_version(layout):
    """from_layout_and_policy_kwargs records all_shaping_pbrs + no_progress_penalty from reward_config."""
    rc = _reward_cfg(all_shaping_pbrs=True, no_progress_penalty=0.25)
    v = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]}, reward_config=rc)
    assert v.all_shaping_pbrs is True and v.no_progress_penalty == 0.25
    assert v.config_version == MODEL_CONFIG_VERSION
    v0 = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v0.all_shaping_pbrs is False and v0.no_progress_penalty == 0.15


def test_migrate_v12_adds_v13_fields_default(version):
    """Pre-v13 configs lack all_shaping_pbrs + no_progress_penalty — migration injects the prior
    defaults (False / 0.15) and bumps the version."""
    data = json.loads(version.to_json())
    data.pop("all_shaping_pbrs", None)
    data.pop("no_progress_penalty", None)
    data["config_version"] = 12
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["all_shaping_pbrs"] is False
    assert result["no_progress_penalty"] == 0.15
    ModelVersion(**result)


# ---------------------------------------------------------------------------
# v14 — stall_pbrs (the "stall" end-state switch, split out of all_shaping_pbrs)
# ---------------------------------------------------------------------------
def test_check_reward_config_v14_stall_pbrs_mismatch_raises(version):
    """stall_pbrs is resume-immutable (it changes the anti-stall reward) — a drift must FATAL."""
    saved = dataclasses.replace(version, stall_pbrs=True)
    with pytest.raises(ModelVersionError) as exc_info:
        saved.check_reward_config(_reward_cfg(stall_pbrs=False))
    assert "stall_pbrs" in str(exc_info.value)


def test_check_reward_config_v14_default_matches(version):
    """A fresh default run (stall_pbrs OFF) matches a default-OFF saved config (no raise)."""
    dataclasses.replace(version, stall_pbrs=False).check_reward_config(_reward_cfg())


def test_check_compatible_ignores_stall_pbrs(version):
    """stall_pbrs is value-meaning, NOT weight-shape — frozen eval / pool / distill loads must accept it."""
    version.check_compatible(dataclasses.replace(version, stall_pbrs=not version.stall_pbrs))


def test_stall_pbrs_recorded_and_config_version(layout):
    """from_layout_and_policy_kwargs records stall_pbrs from reward_config; absent → False."""
    rc = _reward_cfg(all_shaping_pbrs=True, stall_pbrs=True)
    v = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]}, reward_config=rc)
    assert v.stall_pbrs is True and v.config_version == MODEL_CONFIG_VERSION
    v0 = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v0.stall_pbrs is False


def test_migrate_v13_adds_stall_pbrs_default(version):
    """Pre-v14 configs lack stall_pbrs — migration injects False (prior behavior) and bumps the version."""
    data = json.loads(version.to_json())
    data.pop("stall_pbrs", None)
    data["config_version"] = 13
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["stall_pbrs"] is False
    ModelVersion(**result)


# ---------------------------------------------------------------------------
# check_opponent_compatible — the stable-opponent gate (obs-family only)
# ---------------------------------------------------------------------------

def test_check_opponent_compatible_same_arch(version):
    """A foreign opponent sharing the current arch_signature loads — the common case."""
    version.check_opponent_compatible(dataclasses.replace(version))  # must not raise


def test_check_opponent_compatible_arch_mismatch_raises(version):
    """A different arch_signature = different observation family → loud refusal (startup FATAL)."""
    foreign = dataclasses.replace(version, arch_signature="gen3_some_other_arch_v1")
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_opponent_compatible(foreign)
    msg = str(exc_info.value)
    assert "gen3_some_other_arch_v1" in msg
    assert "observation layout" in msg


def test_check_opponent_compatible_ignores_popart(version):
    """use_popart only affects the value head, which an opponent never reads — must NOT gate.
    (This is the key difference from check_compatible, which DOES reject a use_popart mismatch.)"""
    popart_on = dataclasses.replace(version, use_popart=not version.use_popart)
    version.check_opponent_compatible(popart_on)  # must not raise
    # Sanity: check_compatible (the trainee/pool gate) WOULD reject the same mismatch.
    with pytest.raises(ModelVersionError):
        version.check_compatible(popart_on)


def test_check_compatible_rejects_attend_unrevealed_mismatch(version):
    """attend_unrevealed_opponents changes the forward mask the policy trained under, so
    check_compatible (resume + pool/sentinel/distill gate) must reject a mismatch — like use_popart."""
    flipped = dataclasses.replace(
        version, attend_unrevealed_opponents=not version.attend_unrevealed_opponents)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(flipped)
    assert "attend_unrevealed_opponents" in str(exc_info.value)


def test_attend_unrevealed_read_from_features_extractor_kwargs(layout):
    """from_layout_and_policy_kwargs sources the flag from policy_kwargs.features_extractor_kwargs
    (where SB3 forwards it to the extractor), not a top-level key."""
    pk = {"net_arch": [512, 512],
          "features_extractor_kwargs": {"attend_unrevealed_opponents": True}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.attend_unrevealed_opponents is True
    # Absent → False (baseline).
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.attend_unrevealed_opponents is False


def test_check_compatible_rejects_opp_belief_cls_k_on_off_mismatch(version):
    """k=0↔k>0 adds/removes the belief module + widens both projections — a weight-shape change, so
    check_compatible must reject it (one unconditional compare, like use_popart)."""
    on = dataclasses.replace(version, opp_belief_cls_k=2)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(on)   # version has k=0 (default)
    assert "opp_belief_cls_k" in str(exc_info.value)


def test_check_compatible_rejects_opp_belief_cls_k_value_mismatch(version):
    """k=3 vs k=5 is a different projection width → FATAL (both on, different K)."""
    on3 = dataclasses.replace(version, opp_belief_cls_k=3)
    on5 = dataclasses.replace(version, opp_belief_cls_k=5)
    with pytest.raises(ModelVersionError) as exc_info:
        on3.check_compatible(on5)
    assert "opp_belief_cls_k" in str(exc_info.value)


def test_check_compatible_accepts_matching_opp_belief_cls_k(version):
    """Same k (incl. the k=0 baseline) must load — no false rejection."""
    version.check_compatible(dataclasses.replace(version))            # k=0 vs k=0
    on4 = dataclasses.replace(version, opp_belief_cls_k=4)
    on4.check_compatible(dataclasses.replace(on4))                    # k=4 vs k=4


def test_opp_belief_cls_k_read_from_features_extractor_kwargs(layout):
    """opp_belief_cls_k sources from features_extractor_kwargs (SB3 forwards it to the extractor);
    absent → 0 (off, the baseline)."""
    pk = {"net_arch": [512, 512],
          "features_extractor_kwargs": {"opp_belief_cls_k": 4}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.opp_belief_cls_k == 4
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.opp_belief_cls_k == 0


# --- move_belief_mode: a structural string toggle (the MoveBelief reinjection head, v17) -----------


def test_check_compatible_rejects_move_belief_mode_on_off_mismatch(version):
    """off↔non-off adds/removes the MoveBelief head (move_head + reinject + norm) — a state_dict change,
    so check_compatible must reject it (string compare, like opp_belief_slots)."""
    on = dataclasses.replace(version, move_belief_mode="revealed")
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(on)   # version has "off" (default)
    assert "move_belief_mode" in str(exc_info.value)


def test_check_compatible_rejects_move_belief_mode_value_mismatch(version):
    """revealed vs both is a different trained forward (which slots are enriched) → FATAL (both on)."""
    revealed = dataclasses.replace(version, move_belief_mode="revealed")
    both = dataclasses.replace(version, move_belief_mode="both")
    with pytest.raises(ModelVersionError) as exc_info:
        revealed.check_compatible(both)
    assert "move_belief_mode" in str(exc_info.value)


def test_check_compatible_accepts_matching_move_belief_mode(version):
    """Same mode (incl. the 'off' baseline) must load — no false rejection."""
    version.check_compatible(dataclasses.replace(version))                 # off vs off
    both = dataclasses.replace(version, move_belief_mode="both")
    both.check_compatible(dataclasses.replace(both))                       # both vs both


def test_move_belief_mode_read_from_features_extractor_kwargs(layout):
    """move_belief_mode sources from features_extractor_kwargs; absent → 'off' (baseline)."""
    pk = {"net_arch": [512, 512], "features_extractor_kwargs": {"move_belief_mode": "unrevealed"}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.move_belief_mode == "unrevealed"
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.move_belief_mode == "off"


def test_check_compatible_ignores_move_belief_coef(version):
    """move_belief_coef is a training-only loss weight (not weight-shape) → check_compatible must not
    gate it (a frozen eval/pool/distill opponent never runs the loss)."""
    differing = dataclasses.replace(version, move_belief_coef=0.3)
    version.check_compatible(differing)   # must NOT raise


def test_migrate_pre_v17_adds_move_belief_defaults(version):
    """Pre-v17 configs lack the move-belief fields — migration injects mode='off' / coef=0.0 and bumps
    to the current version. The migrated dict must build a valid ModelVersion."""
    data = json.loads(version.to_json())
    data.pop("move_belief_mode", None)
    data.pop("move_belief_coef", None)
    data["config_version"] = 16
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["move_belief_mode"] == "off" and result["move_belief_coef"] == 0.0
    ModelVersion(**result)


# --- damage_op: a structural bool toggle (the differentiable damage operator, v18) -----------------


def test_check_compatible_rejects_damage_op_mismatch(version):
    """damage_op widens BOTH projection heads → a weight-shape change check_compatible must reject
    (like value_active_readout / opp_belief_slots)."""
    on = dataclasses.replace(version, damage_op=True)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(on)        # version has damage_op=False (default)
    assert "damage_op" in str(exc_info.value)


def test_check_compatible_accepts_matching_damage_op(version):
    """Same value (incl. the off baseline) must load — no false rejection."""
    version.check_compatible(dataclasses.replace(version))            # off vs off
    on = dataclasses.replace(version, damage_op=True)
    on.check_compatible(dataclasses.replace(on))                      # on vs on


def test_damage_op_read_from_features_extractor_kwargs(layout):
    """damage_op sources from features_extractor_kwargs; absent → False (baseline)."""
    pk = {"net_arch": [512, 512], "features_extractor_kwargs": {"damage_op": True}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.damage_op is True and v.config_version == MODEL_CONFIG_VERSION
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.damage_op is False


def test_migrate_pre_v18_adds_damage_op_default(version):
    """Pre-v18 configs lack damage_op — migration injects False and bumps to the current version."""
    data = json.loads(version.to_json())
    data.pop("damage_op", None)
    data["config_version"] = 17
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["damage_op"] is False
    ModelVersion(**result)


# --- damage_reattend: a structural bool toggle (re-attend the team tokens to the damage, v31) --------


def test_check_compatible_rejects_damage_reattend_mismatch(version):
    """damage_reattend adds a damage→token projection + encoder layer → a state_dict change
    check_compatible must reject (like opp_belief_slots), even though projection WIDTHS are unchanged."""
    on = dataclasses.replace(version, damage_reattend=True)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(on)        # version has damage_reattend=False (default)
    assert "damage_reattend" in str(exc_info.value)


def test_check_compatible_accepts_matching_damage_reattend(version):
    """Same value (incl. the off baseline) must load — no false rejection."""
    version.check_compatible(dataclasses.replace(version))            # off vs off
    on = dataclasses.replace(version, damage_reattend=True)
    on.check_compatible(dataclasses.replace(on))                      # on vs on


def test_damage_reattend_read_from_features_extractor_kwargs(layout):
    """damage_reattend sources from features_extractor_kwargs; absent → False (baseline)."""
    pk = {"net_arch": [512, 512], "features_extractor_kwargs": {"damage_reattend": True}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.damage_reattend is True and v.config_version == MODEL_CONFIG_VERSION
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.damage_reattend is False


def test_migrate_pre_v31_adds_damage_reattend_default(version):
    """Pre-v31 configs lack damage_reattend — migration injects False and bumps to the current version."""
    data = json.loads(version.to_json())
    data.pop("damage_reattend", None)
    data["config_version"] = 30
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["damage_reattend"] is False
    ModelVersion(**result)


# --- damage_topk_k: a structural INT toggle (the discrete top-K incoming block, v30) ----------------


def test_check_compatible_rejects_damage_topk_k_mismatch(version):
    """damage_topk_k scales the DamageOperator out_dim → both projection in_features. EVERY distinct K
    (incl. 0↔N) is a weight-shape change check_compatible must reject (like opp_belief_cls_k)."""
    on = dataclasses.replace(version, damage_topk_k=5)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(on)        # version has damage_topk_k=0 (default off)
    assert "damage_topk_k" in str(exc_info.value)
    # a different nonzero K is also a mismatch
    with pytest.raises(ModelVersionError):
        dataclasses.replace(version, damage_topk_k=4).check_compatible(on)


def test_check_compatible_accepts_matching_damage_topk_k(version):
    """Same K (incl. the off baseline) must load — no false rejection."""
    version.check_compatible(dataclasses.replace(version))            # 0 vs 0
    on = dataclasses.replace(version, damage_topk_k=5)
    on.check_compatible(dataclasses.replace(on))                      # 5 vs 5


def test_damage_topk_k_read_from_features_extractor_kwargs(layout):
    """damage_topk_k sources from features_extractor_kwargs; absent → 0 (baseline off)."""
    pk = {"net_arch": [512, 512], "features_extractor_kwargs": {"damage_topk_k": 5}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.damage_topk_k == 5 and v.config_version == MODEL_CONFIG_VERSION
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.damage_topk_k == 0


def test_migrate_pre_v30_adds_damage_topk_k_default(version):
    """Pre-v30 configs lack damage_topk_k — migration injects 0 (off) and bumps to the current version."""
    data = json.loads(version.to_json())
    data.pop("damage_topk_k", None)
    data["config_version"] = 29
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["damage_topk_k"] == 0
    ModelVersion(**result)


# --- damage_refine_rounds: a structural INT toggle (iterative damage refinement, v31) ----------------


def test_check_compatible_rejects_damage_refine_rounds_mismatch(version):
    """damage_refine_rounds adds/removes refine_proj (0↔N a state_dict change) or changes the forward
    (N↔M), so EVERY distinct value must be rejected by check_compatible (like opp_belief_cls_k)."""
    on = dataclasses.replace(version, damage_refine_rounds=2)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(on)        # version has damage_refine_rounds=0 (default off)
    assert "damage_refine_rounds" in str(exc_info.value)
    # a different nonzero N is also a mismatch (forward-behavior change, same params)
    with pytest.raises(ModelVersionError):
        dataclasses.replace(version, damage_refine_rounds=1).check_compatible(on)


def test_check_compatible_accepts_matching_damage_refine_rounds(version):
    """Same N (incl. the off baseline) must load — no false rejection."""
    version.check_compatible(dataclasses.replace(version))            # 0 vs 0
    on = dataclasses.replace(version, damage_refine_rounds=2)
    on.check_compatible(dataclasses.replace(on))                      # 2 vs 2


def test_damage_refine_rounds_read_from_features_extractor_kwargs(layout):
    """damage_refine_rounds sources from features_extractor_kwargs; absent → 0 (baseline off)."""
    pk = {"net_arch": [512, 512], "features_extractor_kwargs": {"damage_refine_rounds": 2}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.damage_refine_rounds == 2 and v.config_version == MODEL_CONFIG_VERSION
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.damage_refine_rounds == 0


def test_migrate_pre_v31_adds_damage_refine_rounds_default(version):
    """Pre-v31 configs lack damage_refine_rounds — migration injects 0 (off) and bumps to current."""
    data = json.loads(version.to_json())
    data.pop("damage_refine_rounds", None)
    data["config_version"] = 30
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["damage_refine_rounds"] == 0
    ModelVersion(**result)


# --- damage_matrices_outgoing: a structural BOOL toggle (the outgoing per-move damage matrix, v32) --------


def test_check_compatible_rejects_damage_matrices_outgoing_mismatch(version):
    """The outgoing matrix widens the op out_dim → both projection in_features; toggling it is a weight-shape
    change check_compatible must reject (like damage_op)."""
    on = dataclasses.replace(version, damage_matrices_outgoing=True)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(on)        # version has it off (default)
    assert "damage_matrices_outgoing" in str(exc_info.value)


def test_check_compatible_accepts_matching_damage_matrices_outgoing(version):
    version.check_compatible(dataclasses.replace(version))            # off vs off
    on = dataclasses.replace(version, damage_matrices_outgoing=True)
    on.check_compatible(dataclasses.replace(on))                      # on vs on


def test_damage_matrices_outgoing_read_from_features_extractor_kwargs(layout):
    pk = {"net_arch": [512, 512], "features_extractor_kwargs": {"damage_matrices_outgoing": True}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.damage_matrices_outgoing is True and v.config_version == MODEL_CONFIG_VERSION
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.damage_matrices_outgoing is False


def test_migrate_pre_v32_adds_damage_matrices_outgoing_default(version):
    """Pre-v32 configs lack damage_matrices_outgoing — migration injects False and bumps to current."""
    data = json.loads(version.to_json())
    data.pop("damage_matrices_outgoing", None)
    data["config_version"] = 31
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["damage_matrices_outgoing"] is False
    ModelVersion(**result)


# --- damage_matrices_outgoing_all: the TRANSPOSED outgoing matrix, a structural BOOL toggle (v39) ----------


def test_check_compatible_rejects_damage_matrices_outgoing_all_mismatch(version):
    """The transposed outgoing matrix widens the op out_dim → both projection in_features; toggling it is a
    weight-shape change check_compatible must reject (like damage_op)."""
    on = dataclasses.replace(version, damage_matrices_outgoing_all=True)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(on)        # version has it off (default)
    assert "damage_matrices_outgoing_all" in str(exc_info.value)


def test_check_compatible_accepts_matching_damage_matrices_outgoing_all(version):
    version.check_compatible(dataclasses.replace(version))            # off vs off
    on = dataclasses.replace(version, damage_matrices_outgoing_all=True)
    on.check_compatible(dataclasses.replace(on))                      # on vs on


def test_damage_matrices_outgoing_all_read_from_features_extractor_kwargs(layout):
    pk = {"net_arch": [512, 512], "features_extractor_kwargs": {"damage_matrices_outgoing_all": True}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.damage_matrices_outgoing_all is True and v.config_version == MODEL_CONFIG_VERSION
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.damage_matrices_outgoing_all is False


def test_migrate_pre_v39_adds_damage_matrices_outgoing_all_default(version):
    """Pre-v39 configs lack damage_matrices_outgoing_all — migration injects False and bumps to current."""
    data = json.loads(version.to_json())
    data.pop("damage_matrices_outgoing_all", None)
    data["config_version"] = 38
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["damage_matrices_outgoing_all"] is False
    ModelVersion(**result)


# --- damage_matrices_incoming: a structural BOOL toggle (the incoming per-move damage matrix, v33) --------


def test_check_compatible_rejects_damage_matrices_incoming_mismatch(version):
    on = dataclasses.replace(version, damage_matrices_incoming=True)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(on)
    assert "damage_matrices_incoming" in str(exc_info.value)


def test_check_compatible_accepts_matching_damage_matrices_incoming(version):
    version.check_compatible(dataclasses.replace(version))
    on = dataclasses.replace(version, damage_matrices_incoming=True)
    on.check_compatible(dataclasses.replace(on))


def test_damage_matrices_incoming_read_from_features_extractor_kwargs(layout):
    pk = {"net_arch": [512, 512], "features_extractor_kwargs": {"damage_matrices_incoming": True}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.damage_matrices_incoming is True and v.config_version == MODEL_CONFIG_VERSION
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.damage_matrices_incoming is False


def test_migrate_pre_v33_adds_damage_matrices_incoming_default(version):
    """Pre-v33 configs lack damage_matrices_incoming — migration injects False and bumps to current."""
    data = json.loads(version.to_json())
    data.pop("damage_matrices_incoming", None)
    data["config_version"] = 32
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["damage_matrices_incoming"] is False
    ModelVersion(**result)


# --- move_prior_fusion: a forward-behavior bool toggle (the unified two-part move belief, v20) -------


def test_check_compatible_rejects_move_prior_fusion_mismatch(version):
    """Fusing the move prior changes the forward the policy trained under (no weight-shape change), so
    check_compatible must reject a flip — like attend_unrevealed_opponents."""
    on = dataclasses.replace(version, move_prior_fusion=True)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(on)        # version has move_prior_fusion=False (default)
    assert "move_prior_fusion" in str(exc_info.value)


def test_check_compatible_accepts_matching_move_prior_fusion(version):
    version.check_compatible(dataclasses.replace(version))               # off vs off
    on = dataclasses.replace(version, move_prior_fusion=True)
    on.check_compatible(dataclasses.replace(on))                         # on vs on


def test_move_prior_fusion_read_from_features_extractor_kwargs(layout):
    pk = {"net_arch": [512, 512], "features_extractor_kwargs": {"move_prior_fusion": True}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.move_prior_fusion is True and v.config_version == MODEL_CONFIG_VERSION
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.move_prior_fusion is False


def test_migrate_pre_v20_adds_move_prior_fusion_default(version):
    """Pre-v20 configs lack move_prior_fusion — migration injects False and bumps to the current version."""
    data = json.loads(version.to_json())
    data.pop("move_prior_fusion", None)
    data["config_version"] = 19
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["move_prior_fusion"] is False
    ModelVersion(**result)


# --- move_belief_prefuse: a forward-behavior bool toggle (PRE-transformer reinjection, v32) ----------


def test_check_compatible_rejects_move_belief_prefuse_mismatch(version):
    """Moving the move-belief reinjection before the transformer changes the forward the policy trained
    under (no weight-shape change), so check_compatible must reject a flip — like move_prior_fusion."""
    on = dataclasses.replace(version, move_belief_prefuse=True)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(on)        # version has move_belief_prefuse=False (default)
    assert "move_belief_prefuse" in str(exc_info.value)


def test_check_compatible_accepts_matching_move_belief_prefuse(version):
    version.check_compatible(dataclasses.replace(version))               # off vs off
    on = dataclasses.replace(version, move_belief_prefuse=True)
    on.check_compatible(dataclasses.replace(on))                         # on vs on


def test_move_belief_prefuse_read_from_features_extractor_kwargs(layout):
    pk = {"net_arch": [512, 512], "features_extractor_kwargs": {"move_belief_prefuse": True}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.move_belief_prefuse is True and v.config_version == MODEL_CONFIG_VERSION
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.move_belief_prefuse is False


def test_migrate_pre_v32_adds_move_belief_prefuse_default(version):
    """Pre-v32 configs lack move_belief_prefuse — migration injects False and bumps to the current version."""
    data = json.loads(version.to_json())
    data.pop("move_belief_prefuse", None)
    data["config_version"] = 31
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["move_belief_prefuse"] is False
    ModelVersion(**result)

def test_check_compatible_rejects_value_active_readout_mismatch(version):
    """① value_active_readout widens the value projection → a weight-shape change check_compatible
    must reject (like use_popart)."""
    flipped = dataclasses.replace(version, value_active_readout=not version.value_active_readout)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_compatible(flipped)
    assert "value_active_readout" in str(exc_info.value)


def test_value_active_readout_read_from_features_extractor_kwargs(layout):
    """① value_active_readout sources from features_extractor_kwargs; absent → False."""
    pk = {"net_arch": [512, 512], "features_extractor_kwargs": {"value_active_readout": True}}
    v = ModelVersion.from_layout_and_policy_kwargs(layout, pk)
    assert v.value_active_readout is True and v.config_version == MODEL_CONFIG_VERSION
    v_default = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
    assert v_default.value_active_readout is False


def test_migrate_v9_adds_value_active_readout_default(version):
    """Pre-v10 configs lack value_active_readout — migration injects False and bumps the version."""
    data = json.loads(version.to_json())
    data.pop("value_active_readout", None)
    data.pop("value_tail_weight", None)
    data["config_version"] = 9
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["value_active_readout"] is False
    ModelVersion(**result)


def test_migrate_v10_adds_value_tail_weight_default(version):
    """Pre-v11 configs lack value_tail_weight — migration injects 0.0 (plain MSE) and bumps the version."""
    data = json.loads(version.to_json())
    data.pop("value_tail_weight", None)
    data["config_version"] = 10
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["value_tail_weight"] == 0.0
    ModelVersion(**result)


def test_opp_belief_cls_survives_save_load_and_rebuilds_module(layout, mappings):
    """e2e: --opp-belief-cls-k must be recorded in model_config.json AND survive SB3 save/load,
    rebuilding HiddenOppBeliefPool at the same k (the projection widths must match the saved weights).
    Guards the config↔weights coupling: a refactor that dropped the flag from the serialized kwargs
    would rebuild a baseline-width extractor and FAIL the state_dict load — this pins it loudly."""
    from agents.model.features_extractor import Gen3FeaturesExtractor, D_MODEL
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from sb3_contrib import MaskablePPO

    ek = {"layout": layout, "mappings": mappings,
          "attend_unrevealed_opponents": True, "opp_belief_cls_k": 3}
    pk = {"features_extractor_class": Gen3FeaturesExtractor,
          "features_extractor_kwargs": ek, "net_arch": [512, 512]}
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MaskablePPO(Gen3DualHeadMaskablePolicy, _make_vec_env(layout["total_dim"]),
                            policy_kwargs=pk, verbose=0, device="cpu")
        model.save(os.path.join(tmpdir, "belief"))
        save_model_snapshot(tmpdir, ModelVersion.from_layout_and_policy_kwargs(layout, pk), git_hash="b")
        saved_cfg = ModelVersion.from_json_file(os.path.join(tmpdir, "model_config.json"))
        assert saved_cfg.opp_belief_cls_k == 3
        del model
        loaded = MaskablePPO.load(os.path.join(tmpdir, "belief.zip"), device="cpu")

    ext = loaded.policy.features_extractor
    assert ext.opp_belief_cls_k == 3 and ext.hidden_opp_belief is not None and ext.hidden_opp_belief.k == 3
    # The reloaded projection Linears must have rebuilt at the belief-augmented width (k*D_MODEL wider
    # than baseline) to match the saved weights — proven by the state_dict load succeeding + a finite
    # forward below; a dropped flag would rebuild a narrower extractor and raise on load.
    assert ext.projection.in_features == ext.projection_input_dim
    obs = {"observation": torch.zeros(2, layout["total_dim"]),
           "action_mask": torch.ones(2, 11, dtype=torch.int8)}
    with torch.no_grad():
        dist = loaded.policy.get_distribution(obs)
        val = loaded.policy.predict_values(obs)
    assert dist.distribution.logits.shape == (2, 11) and val.shape == (2, 1)
    assert torch.isfinite(val).all()


def test_check_opponent_compatible_ignores_vf_coef_and_reward(version):
    """vf_coef / reward-config are value-meaning training hparams, irrelevant to an opponent forward."""
    differing = dataclasses.replace(
        version, vf_coef=version.vf_coef + 0.25,
        bias_additivity=0.0, switch_bias_weight=version.switch_bias_weight + 1.0,
    )
    version.check_opponent_compatible(differing)  # must not raise


def test_check_opponent_compatible_total_dim_mismatch_raises(version):
    """Defensive: a hand-edited config with the same arch but a different total_dim is rejected
    (feeding the opponent a wrong-width obs would be a silent-garbage bug)."""
    foreign = dataclasses.replace(version, total_dim=version.total_dim + 1)
    with pytest.raises(ModelVersionError) as exc_info:
        version.check_opponent_compatible(foreign)
    assert "total_dim" in str(exc_info.value)


# ---------------------------------------------------------------------------
# load_foreign_opponent — full save → load of a cross-run opponent
# ---------------------------------------------------------------------------

def _build_and_save_model(tmpdir, layout, mappings, version, *, name="opp_model"):
    """Build a real MaskablePPO, save its .zip + the given version's model_config.json."""
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from sb3_contrib import MaskablePPO

    vec_env = _make_vec_env(layout["total_dim"])
    full_policy_kwargs = {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": {"layout": layout, "mappings": mappings},
        "net_arch": [512, 512],
    }
    model = MaskablePPO(Gen3DualHeadMaskablePolicy, vec_env, policy_kwargs=full_policy_kwargs, verbose=0, device="cpu")
    zip_path = os.path.join(tmpdir, name)
    model.save(zip_path)
    save_model_snapshot(tmpdir, version, git_hash="opp")
    return zip_path + ".zip"


def test_load_foreign_opponent_same_arch_loads_and_predicts(layout, version, mappings):
    """A same-arch foreign model loads inference-only and produces a usable action distribution
    WITHOUT ever calling check_compatible against the live trainee."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = _build_and_save_model(tmpdir, layout, mappings, version)
        model, foreign = load_foreign_opponent(zip_path, current_version=version, device="cpu")

    assert foreign.arch_signature == version.arch_signature
    total_dim = layout["total_dim"]
    obs = {
        "observation": torch.zeros(1, total_dim),
        "action_mask": torch.ones(1, 11, dtype=torch.int8),
    }
    with torch.no_grad():
        dist = model.policy.get_distribution(obs)
    assert dist.distribution.logits.shape == (1, 11)


def test_attend_unrevealed_flag_survives_save_load_and_drives_mask(layout, mappings):
    """e2e: --attend-unrevealed-opponents must (a) be recorded in model_config.json AND (b) survive
    SB3 save/load back into the rebuilt ObsUnpack and actually change the forward mask. Guards the
    config↔forward coupling the version check exists to protect: a refactor that dropped the flag
    from the serialized features_extractor_kwargs would silently revert the forward to the masked
    baseline while model_config.json still read True — and every in-process unit test (which builds
    Gen3FeaturesExtractor directly) would still pass. The two sides are wired independently in
    train_rl_agent.py (config from from_layout_and_policy_kwargs; forward from SB3 serializing the
    same kwarg into the zip), so they must be pinned together here."""
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.model.phase_modules_test import _opp_three_slot_obs
    from sb3_contrib import MaskablePPO

    extractor_kwargs = {"layout": layout, "mappings": mappings,
                        "attend_unrevealed_opponents": True}
    policy_kwargs = {"features_extractor_class": Gen3FeaturesExtractor,
                     "features_extractor_kwargs": extractor_kwargs, "net_arch": [512, 512]}
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MaskablePPO(Gen3DualHeadMaskablePolicy, _make_vec_env(layout["total_dim"]),
                            policy_kwargs=policy_kwargs, verbose=0, device="cpu")
        model.save(os.path.join(tmpdir, "flagged"))
        # (a) config side: model_config.json records the SAME flag the policy was built with.
        save_model_snapshot(tmpdir, ModelVersion.from_layout_and_policy_kwargs(layout, policy_kwargs),
                            git_hash="flagtest")
        saved_cfg = ModelVersion.from_json_file(os.path.join(tmpdir, "model_config.json"))
        assert saved_cfg.attend_unrevealed_opponents is True
        assert saved_cfg.config_version == MODEL_CONFIG_VERSION
        del model
        loaded = MaskablePPO.load(os.path.join(tmpdir, "flagged.zip"), device="cpu")

    # (b) forward side: the flag round-tripped into the rebuilt extractor...
    ext = loaded.policy.features_extractor
    assert ext.attend_unrevealed_opponents is True
    assert ext.unpack.attend_unrevealed_opponents is True
    # ...and it actually drives the mask — unrevealed opp slot attendable, revealed-fainted masked.
    ctx = ext.unpack({"observation": _opp_three_slot_obs(layout)})
    assert bool(ctx.fainted_mask_opp[0, 2]) is False   # unrevealed — attendable after reload
    assert bool(ctx.fainted_mask_opp[0, 1]) is True    # revealed-fainted — still masked


def test_load_foreign_opponent_arch_mismatch_raises(layout, version, mappings):
    """A foreign model whose model_config.json carries a different arch_signature is refused."""
    with tempfile.TemporaryDirectory() as tmpdir:
        foreign_version = dataclasses.replace(version, arch_signature="gen3_old_arch_v1")
        zip_path = _build_and_save_model(tmpdir, layout, mappings, foreign_version)
        with pytest.raises(ModelVersionError) as exc_info:
            load_foreign_opponent(zip_path, current_version=version, device="cpu")
    assert "gen3_old_arch_v1" in str(exc_info.value)


def test_load_foreign_opponent_missing_config_raises(layout, version, mappings):
    """No sibling model_config.json → refuse to load blind (provenance is required)."""
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from sb3_contrib import MaskablePPO

    with tempfile.TemporaryDirectory() as tmpdir:
        vec_env = _make_vec_env(layout["total_dim"])
        model = MaskablePPO(
            Gen3DualHeadMaskablePolicy, vec_env, verbose=0, device="cpu",
            policy_kwargs={
                "features_extractor_class": Gen3FeaturesExtractor,
                "features_extractor_kwargs": {"layout": layout, "mappings": mappings},
                "net_arch": [512, 512],
            },
        )
        zip_path = os.path.join(tmpdir, "no_config_model")
        model.save(zip_path)  # NOTE: no save_model_snapshot → no model_config.json
        with pytest.raises(FileNotFoundError) as exc_info:
            load_foreign_opponent(zip_path + ".zip", current_version=version, device="cpu")
    assert "model_config.json" in str(exc_info.value)


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
# original_command — the immutable original invocation
# ---------------------------------------------------------------------------

def test_original_command_explicit_recorded(version):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc", original_command="train.py --steps 5")
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert meta["original_command"] == "train.py --steps 5"


def test_original_command_derived_from_argv(version, monkeypatch):
    monkeypatch.delenv("LAUNCHER_COMMAND", raising=False)
    monkeypatch.setattr(snapshot.sys, "argv", ["train.py", "--steps", "10"])
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc")
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert meta["original_command"] == "train.py --steps 10"


def test_original_command_prefers_launcher_env(version, monkeypatch):
    monkeypatch.setenv("LAUNCHER_COMMAND", "python -m main.launcher --steps 99")
    monkeypatch.setattr(snapshot.sys, "argv", ["train.py", "--run-dir", "models/x"])
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="abc")
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert meta["original_command"] == "python -m main.launcher --steps 99"


def test_original_command_immutable_across_restarts(version, monkeypatch):
    """The original invocation is set ONCE at creation and never overwritten on a resume —
    even when the resuming process passes (or would derive) a different command."""
    monkeypatch.delenv("LAUNCHER_COMMAND", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Creation
        save_model_snapshot(tmpdir, version, git_hash="abc", original_command="train.py --fresh")
        # Restart: a DIFFERENT command — must be ignored, the original wins.
        save_model_snapshot(tmpdir, version, git_hash="def", original_command="train.py --resumed")
        # Restart again, this time deriving (no explicit arg) — still preserved.
        monkeypatch.setattr(snapshot.sys, "argv", ["train.py", "--also-resumed"])
        save_model_snapshot(tmpdir, version, git_hash="ghi")
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
    assert meta["original_command"] == "train.py --fresh"


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


def _make_vec_env_extra_key(total_dim: int) -> DummyVecEnv:
    """Like `_make_vec_env` but with an EXTRA training-only obs key (`win_margin`) the saved policy
    predates — mimics resuming a win-prob run after that key was added to the env."""
    _obs = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (total_dim,), np.float32),
        "action_mask": gym.spaces.MultiBinary(11),
        "win_margin": gym.spaces.Box(-1.0, 1.0, (1,), np.float32),
    })

    class _E(gym.Env):
        observation_space = _obs
        action_space = gym.spaces.Discrete(11)

        def reset(self, **kwargs):
            return {"observation": np.zeros(total_dim, np.float32),
                    "action_mask": np.ones(11, np.int8), "win_margin": np.zeros(1, np.float32)}, {}

        def step(self, action):
            return self.reset()[0], 0.0, False, False, {}

    return DummyVecEnv([_E])


def test_load_tolerates_extra_training_only_obs_key(layout, version, mappings):
    """Resume regression: when the live env declares a TRAINING-ONLY obs key the saved policy predates
    (e.g. `win_margin`, added mid-run), the load must NOT FATAL on SB3's strict `check_for_correct_spaces`
    — the model forward reads only obs['observation'] (which `check_compatible` already pinned via
    total_dim). Reproduces the `ai_v6_03_win_pred` crash where adding win_margin broke resume."""
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from sb3_contrib import MaskablePPO

    total_dim = layout["total_dim"]
    pk = {"features_extractor_class": Gen3FeaturesExtractor,
          "features_extractor_kwargs": {"layout": layout, "mappings": mappings}, "net_arch": [512, 512]}
    model = MaskablePPO(Gen3DualHeadMaskablePolicy, _make_vec_env(total_dim),  # saved WITHOUT win_margin
                        policy_kwargs=pk, verbose=0, device="cpu")
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "m")
        model.save(zip_path)
        save_model_snapshot(tmpdir, version, git_hash="test")
        # The resume env now HAS win_margin — would raise ValueError("Observation spaces do not match")
        # without the tolerance in load_model_snapshot.
        loaded = load_model_snapshot(zip_path + ".zip", env=_make_vec_env_extra_key(total_dim),
                                     current_version=version, device="cpu")
    assert "win_margin" in loaded.observation_space.spaces  # tolerated + loaded


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
    model = MaskablePPO(Gen3DualHeadMaskablePolicy, vec_env, policy_kwargs=full_policy_kwargs, verbose=0, device="cpu")

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
            device="cpu",
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
    model = MaskablePPO(Gen3DualHeadMaskablePolicy, vec_env, policy_kwargs=full_policy_kwargs, verbose=0, device="cpu")

    # The run was started with vf_coef=0.3 → that is what model_config.json records.
    saved_version = ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]}, vf_coef=0.3)

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "test_model")
        model.save(zip_path)
        save_model_snapshot(tmpdir, saved_version, git_hash="test")

        # Resuming with a different vf_coef is a hard error (before the model even loads).
        with pytest.raises(ModelVersionError) as exc_info:
            load_model_snapshot(zip_path + ".zip", env=vec_env, current_version=saved_version,
                                enforce_vf_coef=0.5, device="cpu")
        assert "vf_coef" in str(exc_info.value)

        # Matching value loads fine.
        load_model_snapshot(zip_path + ".zip", env=vec_env, current_version=saved_version,
                            enforce_vf_coef=0.3, device="cpu")

        # Frozen-snapshot load (no enforcement) ignores vf_coef even when it would differ.
        load_model_snapshot(zip_path + ".zip", env=vec_env, current_version=saved_version, device="cpu")


def test_load_model_snapshot_finds_config_in_parent_when_zip_in_checkpoints(layout, mappings, version):
    """A checkpoint .zip now lives in <run>/checkpoints/ while model_config.json stays at the
    run ROOT. load_model_snapshot must still find + enforce the arch check via the parent-dir
    fallback — proven by a mismatched current_version FATALing instead of silently loading."""
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
    model = MaskablePPO(Gen3DualHeadMaskablePolicy, vec_env, policy_kwargs=full_policy_kwargs, verbose=0, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="test")  # model_config.json at run root
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(ckpt_dir)
        zip_path = os.path.join(ckpt_dir, "checkpoint_1000_steps")
        model.save(zip_path)
        assert not os.path.exists(os.path.join(ckpt_dir, "model_config.json"))  # config only at root

        # A mismatched arch must FATAL → proves the run-root config was found + checked
        # (if it weren't found, load would print the "legacy model" warning and succeed).
        bad = dataclasses.replace(version, arch_signature="some_other_arch")
        with pytest.raises(ModelVersionError):
            load_model_snapshot(zip_path + ".zip", env=vec_env, current_version=bad, device="cpu")

        # The matching version loads cleanly through that same parent-fallback path.
        assert load_model_snapshot(zip_path + ".zip", env=vec_env, current_version=version, device="cpu") is not None


def test_load_saved_version_finds_config_in_parent_when_zip_in_checkpoints(version):
    """A flagless resume reads the saved ModelVersion via train_rl_agent._load_saved_version. With
    the checkpoint now in <run>/checkpoints/ and model_config.json at the run ROOT, it must search
    the parent dir — else a toggle-ON resume reads no saved version, falls back to OFF defaults, and
    FATALs at the arch check. Regression guard for that resume path."""
    from main.train_rl_agent import _load_saved_version

    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_snapshot(tmpdir, version, git_hash="test")  # model_config.json at run root
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(ckpt_dir)
        zip_path = os.path.join(ckpt_dir, "checkpoint_1000_steps.zip")
        open(zip_path, "w").close()  # _resolve_paths only needs the zip to EXIST, not be valid

        loaded = _load_saved_version(zip_path)
        assert loaded is not None, "flagless resume could not read the run-root model_config.json"
        assert loaded.arch_signature == version.arch_signature


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


def test_migrate_v6_adds_draw_penalty_default(version):
    """Pre-v7 configs lack draw_penalty — migration injects -30.0 (== the prior tie==loss behavior)
    and bumps to the current version. The migrated dict must construct a valid ModelVersion."""
    data = json.loads(version.to_json())
    data.pop("draw_penalty", None)
    data["config_version"] = 6
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["draw_penalty"] == pytest.approx(-30.0)
    ModelVersion(**result)


def test_migrate_v7_adds_attend_unrevealed_opponents_default(version):
    """Pre-v8 configs lack attend_unrevealed_opponents — migration injects False (baseline masking)
    and bumps to the current version. The migrated dict must construct a valid ModelVersion."""
    data = json.loads(version.to_json())
    data.pop("attend_unrevealed_opponents", None)
    data.pop("opp_belief_cls_k", None)
    data["config_version"] = 7
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["attend_unrevealed_opponents"] is False
    ModelVersion(**result)


def test_migrate_v8_adds_opp_belief_cls_k_default(version):
    """Pre-v9 configs lack the hidden-opponent belief toggle — migration injects k=0 (no belief
    module) and bumps to the current version. The migrated dict must build a valid ModelVersion."""
    data = json.loads(version.to_json())
    data.pop("opp_belief_cls_k", None)
    data["config_version"] = 8
    result = _migrate_config(data)
    assert result["config_version"] == MODEL_CONFIG_VERSION
    assert result["opp_belief_cls_k"] == 0
    ModelVersion(**result)


def test_migrate_v8_drops_interim_opp_belief_cls_bool(version):
    """A dev config from the interim two-field design (opp_belief_cls bool) must have the bool
    DROPPED by migration so it doesn't break ModelVersion(**result) (no such field any more)."""
    data = json.loads(version.to_json())
    data["opp_belief_cls"] = True          # interim field that never shipped
    data.pop("opp_belief_cls_k", None)
    data["config_version"] = 8
    result = _migrate_config(data)
    assert "opp_belief_cls" not in result and result["opp_belief_cls_k"] == 0
    ModelVersion(**result)                 # must not raise on an unexpected kwarg


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


def test_current_model_version_threads_belief_toggle(mappings):
    """The eval/self-play worker gate must reflect the RUN's belief toggle, else a belief-ON run
    FATALs on its own sentinels. current_model_version(opp_belief_slots=True) must say True."""
    from agents.model.snapshot import current_model_version
    assert current_model_version(mappings).opp_belief_slots is False          # default off
    on = current_model_version(mappings, opp_belief_slots=True, attend_unrevealed_opponents=True,
                               opp_belief_aux_coef=0.2)
    assert on.opp_belief_slots is True
    assert on.attend_unrevealed_opponents is True
    # a belief-ON "current" version is compatible with a belief-ON saved version (no FATAL)…
    on.check_compatible(on)
    # …and a toggle-OFF current version is NOT (the bug this fix prevents)
    off = current_model_version(mappings)
    import pytest as _pytest
    with _pytest.raises(Exception):
        off.check_compatible(on)


def test_arch_toggles_from_model_extracts_flags():
    """arch_toggles_from_model reads the live model's toggles for the worker cfg. EVERY toggle must be
    pinned here: a missing key → the worker rebuilds current_model_version toggle-OFF → a feature-ON
    self-play run FATALs on its own sentinels (exactly the bug the v23 damage_outgoing omission caused)."""
    from agents.model.snapshot import arch_toggles_from_model, current_model_version
    import inspect
    import types
    # Note: the fe attribute for the damage op is `damage_op_enabled`, the emitted key is `damage_op`.
    fe = types.SimpleNamespace(attend_unrevealed_opponents=True, opp_belief_cls_k=0,
                               opp_belief_slots=True, value_active_readout=False,
                               move_belief_mode="revealed", opp_belief_latent=True,
                               damage_op_enabled=True, damage_outgoing=True, move_candidate_floor=0.3,
                               move_latent=True, move_prior_fusion=True,
                               move_belief_prefuse=True, move_belief_single_compute=True,
                               win_prob_mode="read_only",
                               damage_topk_k=5, damage_refine_rounds=2, damage_matrices_outgoing=True,
                               damage_matrices_incoming=True, damage_matrices_outgoing_all=True)
    model = types.SimpleNamespace(policy=types.SimpleNamespace(features_extractor=fe, popart=object()))
    t = arch_toggles_from_model(model)
    assert t["opp_belief_slots"] is True and t["attend_unrevealed_opponents"] is True
    assert t["use_popart"] is True and t["value_active_readout"] is False
    assert t["move_belief_mode"] == "revealed"
    # v23/v24 keys (these were the threading gaps): every one must round-trip.
    assert t["damage_op"] is True and t["damage_outgoing"] is True
    assert t["move_candidate_floor"] == 0.3 and t["move_latent"] is True
    assert t["move_prior_fusion"] is True
    # v32/v47: the PRE-transformer move-belief reinjection + the frozen single-compute belief (a run
    # with either ON must gate its sentinels). gen3_cpu_damage_deleted_v1 removed mask_incoming_damage_obs.
    assert t["move_belief_prefuse"] is True and t["move_belief_single_compute"] is True
    assert t["win_prob_mode"] == "read_only"
    # v30: the discrete top-K incoming block's K (a topk-ON self-play run must gate its sentinels with it).
    assert t["damage_topk_k"] == 5
    # v31: the iterative-refinement round count (a refine-ON self-play run must gate its sentinels with it).
    assert t["damage_refine_rounds"] == 2
    # v32: the outgoing per-move damage matrix (a matrix-ON self-play run must gate its sentinels with it).
    assert t["damage_matrices_outgoing"] is True
    # v33: the incoming per-move damage matrix.
    assert t["damage_matrices_incoming"] is True
    # v39: the TRANSPOSED outgoing matrix (our 6 mons → opp active; a switch-in-offense-ON run must gate it).
    assert t["damage_matrices_outgoing_all"] is True
    # Every emitted toggle MUST be an accepted current_model_version kwarg — else a future toggle that
    # isn't threaded fails here in a unit test, not only at a self-play load (TypeError).
    assert set(t) <= set(inspect.signature(current_model_version).parameters)


def test_current_model_version_threads_move_belief_mode(mappings):
    """A move-belief-ON self-play run must build its 'current' version with the SAME mode, or it FATALs
    on its own (move-belief-ON) sentinels. The mode round-trips and a mode-OFF current version is NOT
    compatible with a mode-ON saved one."""
    from agents.model.snapshot import current_model_version
    assert current_model_version(mappings).move_belief_mode == "off"          # default
    on = current_model_version(mappings, move_belief_mode="revealed", attend_unrevealed_opponents=True)
    assert on.move_belief_mode == "revealed"
    on.check_compatible(on)                                                   # ON vs ON: fine
    off = current_model_version(mappings)
    with pytest.raises(ModelVersionError):
        off.check_compatible(on)                                             # OFF current vs ON saved → FATAL


def test_belief_works_for_selfplay_and_stable_play(mappings):
    """The belief arch must interoperate in BOTH opponent modes:
    - SELF-PLAY: a belief-ON snapshot (the trainee's own sentinel) loads against a belief-ON current
      version (check_compatible — the full gate, which DOES compare opp_belief_slots).
    - STABLE PLAY: a belief-OFF FOREIGN opponent (from another run) loads against a belief-ON run —
      the opponent gate (check_opponent_compatible) keys ONLY on arch_signature (the obs family), and
      belief does NOT bump arch_signature, so mixed belief settings interoperate either direction."""
    from agents.model.snapshot import current_model_version
    belief_on = current_model_version(mappings, opp_belief_slots=True, attend_unrevealed_opponents=True)
    belief_off = current_model_version(mappings)
    belief_on.check_compatible(belief_on)                 # self-play: ON trainee vs its ON sentinels
    assert belief_on.arch_signature == belief_off.arch_signature   # belief does NOT change the obs family
    assert belief_on.total_dim == belief_off.total_dim             # obs["observation"] interface unchanged
    belief_on.check_opponent_compatible(belief_off)      # stable: a belief-OFF opponent in a belief-ON run
    belief_off.check_opponent_compatible(belief_on)      # …and a belief-ON opponent in a belief-OFF run


# ---------------------------------------------------------------------------
# Matchup provenance (the four diligence fixes): era history, per-row/manifest/
# sidecar regime tags, and the recorded-matchup readers.
# ---------------------------------------------------------------------------

def _cli(h, spec=None):
    return {"_matchup_spec_hash": h, "_matchup_spec": spec or {"mix_kind": "bots"}}


def test_matchup_history_appends_once_per_era(version):
    with tempfile.TemporaryDirectory() as tmp:
        save_model_snapshot(tmp, version, cli_args=_cli("aaa1111111"))
        save_model_snapshot(tmp, version, cli_args=_cli("aaa1111111"))   # same era → no dup
        meta = json.load(open(os.path.join(tmp, "metadata.json")))
        assert [e["hash"] for e in meta["matchup_history"]] == ["aaa1111111"]
        # a resume with a DIFFERENT declared matchup → a new era entry, old one preserved
        save_model_snapshot(tmp, version, cli_args=_cli("bbb2222222", {"mix_kind": "exploiter"}))
        meta = json.load(open(os.path.join(tmp, "metadata.json")))
        assert [e["hash"] for e in meta["matchup_history"]] == ["aaa1111111", "bbb2222222"]
        assert meta["matchup_history"][1]["spec"] == {"mix_kind": "exploiter"}
        assert all("recorded_at" in e for e in meta["matchup_history"])


def test_matchup_history_preserved_by_cli_less_saves(version):
    # the periodic checkpoint save path passes no cli_args — it must not drop the history
    with tempfile.TemporaryDirectory() as tmp:
        save_model_snapshot(tmp, version, cli_args=_cli("aaa1111111"))
        save_model_snapshot(tmp, version, current_lr=1e-4, current_epochs=7)
        meta = json.load(open(os.path.join(tmp, "metadata.json")))
        assert [e["hash"] for e in meta["matchup_history"]] == ["aaa1111111"]


def test_read_matchup_hash_and_recorded_matchup(version):
    with tempfile.TemporaryDirectory() as tmp:
        assert snapshot._read_matchup_hash(tmp) is None                  # no metadata yet
        save_model_snapshot(tmp, version, cli_args=_cli("cafe000042", {"mix_kind": "bots"}))
        assert snapshot._read_matchup_hash(tmp) == "cafe000042"
        # read_recorded_matchup resolves from a checkpoint path (checkpoints/ subdir → parent)
        ck = os.path.join(tmp, "checkpoints", "checkpoint_5_steps.zip")
        os.makedirs(os.path.dirname(ck))
        open(ck, "w").close()
        h, spec = snapshot.read_recorded_matchup(ck)
        assert h == "cafe000042" and spec == {"mix_kind": "bots"}
        assert snapshot.read_recorded_matchup(os.path.join(tmp, "nope.zip"))[0] == "cafe000042"


def test_eval_row_carries_externals_and_matchup_hash(version):
    from agents.model.snapshot import append_eval_result_row
    with tempfile.TemporaryDirectory() as tmp:
        save_model_snapshot(tmp, version, cli_args=_cli("cafe000042"))
        append_eval_result_row(
            tmp, 1000, 100, {"heuristic": 0.9}, [],
            externals={"ext_target": {"win_rate": 0.84, "counts": (84, 100)}})
        row = json.loads(open(os.path.join(tmp, "eval_results.jsonl")).read())
        assert row["matchup_hash"] == "cafe000042"
        assert row["externals"] == {"ext_target": {"win_rate": 0.84, "counts": [84, 100]}}
        assert "ext_target" not in row["bots"]          # never inside the ELO-fit ladder
        # omitted args → the old row shape (additive change)
        append_eval_result_row(tmp, 2000, 100, {"heuristic": 0.9}, [])
        row2 = json.loads(open(os.path.join(tmp, "eval_results.jsonl")).readlines()[1])
        assert "externals" not in row2 and row2["matchup_hash"] == "cafe000042"


def test_checkpoint_sidecar_carries_matchup_hash(version):
    with tempfile.TemporaryDirectory() as tmp:
        save_model_snapshot(tmp, version, cli_args=_cli("cafe000042"))
        ck = os.path.join(tmp, "checkpoints", "checkpoint_9_steps.zip")
        os.makedirs(os.path.dirname(ck))
        open(ck, "w").close()
        record_checkpoint(tmp, ck, 1e-4, 7)
        sidecar = read_checkpoint_metadata(ck)
        assert sidecar["matchup_hash"] == "cafe000042"
        meta = json.load(open(os.path.join(tmp, "metadata.json")))
        assert meta["snapshot_history"]["checkpoint_9_steps.zip"]["matchup_hash"] == "cafe000042"


def test_check_belief_grad_mode_allow_change_permits_migration(version, capsys):
    """--allow-belief-grad-mode-change: a mismatch prints a loud migration notice instead of raising
    (the intentional shaping<->detached flip; the gate exists against ACCIDENTAL drift only)."""
    saved = dataclasses.replace(version, belief_grad_mode="detached")
    saved.check_belief_grad_mode("shaping", allow_change=True)  # must not raise
    out = capsys.readouterr().out
    assert "MIGRATION" in out and "'detached' -> 'shaping'" in out
    with pytest.raises(ModelVersionError):
        saved.check_belief_grad_mode("shaping")  # default stays FATAL
