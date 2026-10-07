"""THE X5 VERSION BREAK, part 1 (config v144, `gen3_x5_version_break_v1`; `model_version/version_break.py`).

X5's hypothesis tokens became the ONLY belief representation and the blob path was DELETED. Each test here
FAILS ON REVERT of one piece of that:

* the production surface builds the hypothesis set and the flat pointer with no switch to turn them off
  (`belief_tokens` is no constructor kwarg any more) — and BeliefSlots / α / β are not in the state_dict
  (part 2 deleted the classes too: `x5_version_break_part2_test.py`);
* an OFF value of each flag X5 requires is REFUSED at build, naming the flag (no blob fallback);
* a typed `--belief-tokens` is refused at PARSE time WITH its reason (`designs/deleted_flags.md`);
* a pre-break config is refused at `MIGRATION_FLOOR` with its OWN reason — blob deleted (recorded, or absent
  below v136) vs a pre-break fixed_mass checkpoint the break reshaped — and both name the pinned commit;
* the trainer's resume / fork path maps that refusal to `FATAL_CONFIG`, and `main.checkargs` reports it;
* a PICKLED `belief_tokens='blob'` is refused (`_DEAD_FEK_JUDGED`), `'fixed_mass'` pops, and
  `load_checkpoint_strict` refuses a blob zip before SB3 sees its kwargs.
"""
from __future__ import annotations

import json
import zipfile
from typing import Any, Dict

import pytest

from agents.model.model_version import MIGRATION_FLOOR, MODEL_CONFIG_VERSION, ModelVersionError, _migrate_config
from agents.model.model_version.version_break import (LAST_BLOB_COMMIT, VERSION_BREAK_CONFIG,
                                                      VERSION_BREAK_SIGNATURE)


@pytest.fixture(scope="module")
def production():
    """The production extractor built from the mirror (the delivery graph's build seam) + its kwargs."""
    from agents.model.delivery_graph import build_extractor
    fe, cfg, layout = build_extractor()
    return fe, cfg, layout


def _build(production, **override: Any):
    import inspect

    import gymnasium as gym
    import numpy as np

    from agents.model.damage_tables import sanitize_historical_move_floor
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.observation.state_encoder import load_mappings
    _fe, cfg, layout = production
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kwargs = {k: v for k, v in cfg.items() if k in sig}
    sanitize_historical_move_floor(kwargs)
    kwargs.update(override)
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(space, layout=layout, mappings=load_mappings(), **kwargs)


# ----------------------------------------------------------------------------- the flip, unconditional
def test_the_version_stamps_are_the_breaks():
    assert MODEL_CONFIG_VERSION == VERSION_BREAK_CONFIG == MIGRATION_FLOOR == 144
    from agents.model.model_version import ARCH_SIGNATURE, SIGNATURE_FIRST_VERSION
    assert ARCH_SIGNATURE == VERSION_BREAK_SIGNATURE == "gen3_x5_version_break_v1"
    assert SIGNATURE_FIRST_VERSION[ARCH_SIGNATURE] == 144
    assert LAST_BLOB_COMMIT == "26131c0ce62b2de7c9913d9c84e8c318a0200b19"


def test_production_builds_the_hypothesis_set_and_the_flat_pointer_with_no_switch(production):
    fe, cfg, _layout = production
    assert "belief_tokens" not in cfg, "the production mirror records a deleted field"
    assert fe.hypothesis_builder is not None and fe.flat_intent_head is not None
    assert not hasattr(fe, "belief_slots"), "BeliefSlots (the blob path's hidden-slot token) is DELETED (part 2)"
    import inspect

    from agents.model.features_extractor import Gen3FeaturesExtractor
    assert "belief_tokens" not in inspect.signature(Gen3FeaturesExtractor.__init__).parameters
    with pytest.raises(TypeError):
        _build(production, belief_tokens="blob")


def test_a_built_policy_holds_no_blob_parameter():
    """On a REAL policy: α / β and BeliefSlots are DELETED (part 2) — no blob state_dict key."""
    from agents.training import learner_golden as LG
    keys = list(LG.build_learner().policy.state_dict())
    for blob_key in ("features_extractor.alpha_head.", "features_extractor.beta_head.",
                     "features_extractor.belief_slots."):
        assert not [k for k in keys if k.startswith(blob_key)], blob_key
    assert [k for k in keys if k.startswith("features_extractor.hypothesis_builder.")]
    assert [k for k in keys if k.startswith("features_extractor.flat_intent_head.")]


_REQUIRED_OFF = {"t0_species_prior": False, "move_belief_mode": "off", "opp_intent": False,
                 "opp_belief_slots": False, "entity_tail_seats": False, "move_prior_fusion": False}


@pytest.mark.parametrize("flag", sorted(_REQUIRED_OFF))
def test_an_off_value_of_each_flag_x5_requires_is_refused_naming_it(production, flag):
    """With the belief family on there is no blob path to fall back to: each OFF value is a ValueError
    naming the flag (the one that is OFF, or — for the two belief toggles — the one that needs it)."""
    with pytest.raises(ValueError) as exc:
        _build(production, **{flag: _REQUIRED_OFF[flag]})
    named = {"opp_belief_slots": "opp_belief_slots", "opp_intent": "opp_intent"}.get(flag, flag)
    assert named in str(exc.value), str(exc.value)


def test_the_belief_off_ablation_still_builds_and_builds_no_x5(production):
    """Both belief toggles OFF is not blob: no hidden-slot token and no intent at all — still constructible."""
    fe = _build(production, opp_belief_slots=False, opp_intent=False, t0_species_prior=False,
                intent_move_cell=False, intent_threshold=False, intent_conditional=False,
                switch_branch_cell=False, species_prior_fusion=False, ridealong_opp=0)
    assert fe.hypothesis_builder is None and fe.flat_intent_head is None and not hasattr(fe, "belief_slots")


# ----------------------------------------------------------------------------- the CLI
def test_a_typed_belief_tokens_is_refused_at_parse_time_with_its_reason(capsys):
    from main.train.parser import build_parser
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--belief-tokens", "fixed_mass"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "--belief-tokens was DELETED" in err and "X5 VERSION BREAK" in err, err[-600:]


# ----------------------------------------------------------------------------- the config refusal
def _cfg(version: int, **kw: Any) -> Dict[str, Any]:
    return {"config_version": version, "arch_signature": "gen3_event_record_v2", **kw}


def test_a_v143_blob_config_is_refused_naming_the_deletion_and_the_pin():
    with pytest.raises(ModelVersionError) as exc:
        _migrate_config(_cfg(143, belief_tokens="blob"))
    msg = str(exc.value)
    assert "PRE-GENERATION" in msg and "belief_tokens='blob'" in msg and "DELETED" in msg
    assert LAST_BLOB_COMMIT in msg and "PINNED" in msg
    assert "PRE-BREAK X5" not in msg


def test_a_pre_v136_config_without_the_key_is_a_blob_checkpoint():
    with pytest.raises(ModelVersionError) as exc:
        _migrate_config(_cfg(130))
    msg = str(exc.value)
    assert "predates belief_tokens" in msg and "DELETED" in msg and LAST_BLOB_COMMIT in msg


def test_a_v143_fixed_mass_config_is_refused_with_its_own_reason():
    with pytest.raises(ModelVersionError) as exc:
        _migrate_config(_cfg(143, belief_tokens="fixed_mass"))
    msg = str(exc.value)
    assert "PRE-BREAK X5 checkpoint" in msg and "reshaped" in msg and LAST_BLOB_COMMIT[:12] in msg
    assert "was DELETED" not in msg


def test_an_older_generation_gets_the_plain_pre_generation_diagnosis():
    with pytest.raises(ModelVersionError) as exc:
        _migrate_config(_cfg(100))
    assert "PRE-GENERATION" in str(exc.value) and "belief_tokens" not in str(exc.value)


def _run_dir(tmp_path, data: Dict[str, Any]) -> str:
    run = tmp_path / "rb_run_old"
    (run / "checkpoints").mkdir(parents=True)
    (run / "model_config.json").write_text(json.dumps(data))
    zip_path = run / "checkpoints" / "checkpoint_1000_steps.zip"
    zip_path.write_bytes(b"")
    return str(zip_path)


@pytest.mark.parametrize("recorded", ["blob", "fixed_mass"])
def test_the_trainer_resume_path_exits_FATAL_CONFIG(tmp_path, capsys, recorded):
    from main.exit_codes import TrainExitCode
    from main.train.config import enforce_not_shaped_parent
    model = _run_dir(tmp_path, _cfg(143, belief_tokens=recorded))
    with pytest.raises(SystemExit) as e:
        enforce_not_shaped_parent(model)
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    out = capsys.readouterr().out
    assert "[ModelVersion] FATAL" in out and LAST_BLOB_COMMIT[:12] in out
    assert ("was DELETED" in out) == (recorded == "blob")


def test_a_current_config_passes_the_trainer_check(tmp_path):
    from main.train.config import enforce_not_shaped_parent
    enforce_not_shaped_parent(_run_dir(tmp_path, _cfg(MIGRATION_FLOOR, arch_signature="gen3_x5_version_break_v1")))


def test_checkargs_reports_the_pre_break_parent(tmp_path):
    from main.checkargs import version_break_finding
    model = _run_dir(tmp_path, _cfg(143, belief_tokens="blob"))
    f = version_break_finding(["--model", model, "--steps", "1"])
    assert f is not None and f["last_commit"] == LAST_BLOB_COMMIT and "DELETED" in f["message"]
    assert version_break_finding(["--steps", "1"]) is None


# ----------------------------------------------------------------------------- the pickled kwarg
def test_a_pickled_blob_kwarg_is_refused_and_fixed_mass_pops():
    from agents.model.snapshot import _DEAD_FEK_JUDGED, sanitize_dead_extractor_kwargs
    assert ("belief_tokens", "fixed_mass") in _DEAD_FEK_JUDGED
    fek = {"belief_tokens": "fixed_mass", "damage_op": True}
    assert sanitize_dead_extractor_kwargs(fek) is True and fek == {"damage_op": True}
    with pytest.raises(ModelVersionError) as exc:
        sanitize_dead_extractor_kwargs({"belief_tokens": "blob"})
    assert "DELETED" in str(exc.value) and LAST_BLOB_COMMIT in str(exc.value)


def _blob_zip(tmp_path, value: str) -> str:
    """A minimal sb3 zip whose `data` member pickles policy_kwargs with ``belief_tokens=value`` (sb3's own
    serializer, so `load_from_zip_file` reads it exactly as a real checkpoint's)."""
    from stable_baselines3.common.save_util import data_to_json
    data = {"policy_kwargs": {"features_extractor_kwargs": {"belief_tokens": value, "damage_op": True}}}
    path = tmp_path / f"{value}.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data", data_to_json(data))
        z.writestr("_stable_baselines3_version", "2.3.2")
    return str(path)


def test_load_checkpoint_strict_refuses_a_blob_zip_before_sb3_sees_it(tmp_path):
    from agents.model.snapshot import load_checkpoint_strict, refuse_deleted_pickled_kwargs
    path = _blob_zip(tmp_path, "blob")
    with pytest.raises(ModelVersionError) as exc:
        load_checkpoint_strict(path)
    assert "belief_tokens='blob'" in str(exc.value) and LAST_BLOB_COMMIT in str(exc.value)
    refuse_deleted_pickled_kwargs(_blob_zip(tmp_path, "fixed_mass"))      # the reproducible value passes
