"""The RETIRED LEVERS (deletion pass L1, config v131; L2, config v132; K2, TF32 — a metadata.json
runtime knob, no config bump) — what a recorded config / a pickled zip that still names one does on
every path that reads it.

`model_version.retired_levers` is the ONE table; this file drives each consumer of it:

* `_migrate_config` — pops every retired field from ANY config (a stale key TypeErrors the
  dataclass), refuses on EVERY load a structural lever recorded ON (PopArt, the distributional value
  head, `value_from_dist`, the dense auxiliary head, the privileged true-team route: parameters / a
  critic route the surviving code cannot rebuild), and pops a
  training-only lever silently (a frozen forward never reads it);
* `check_no_retired_levers` / `config.enforce_not_shaped_parent` / `checkargs.retired_levers_finding`
  — a RESUME or FORK of a run that recorded ANY lever ON is refused, naming the flag and the pin;
* `snapshot.sanitize_dead_policy_kwargs` + the extractor dead lists — a checkpoint written BEFORE the
  deletion (every v121+ zip pickles `use_popart=False`, `value_from_dist=False`, `value_dist_*`,
  and (L2) the extractor kwargs `value_true_team=False`, `dense_aux=False`)
  still LOADS, end to end, through the loaders a run uses; and a bare load of the same zip is the
  failure they exist to prevent (the non-vacuity control).
"""
from __future__ import annotations

import json

import pytest

from agents.model.model_version import (
    MODEL_CONFIG_VERSION,
    ModelVersion,
    ModelVersionError,
    RETIRED_FIELDS,
    RetiredLeverCheckpointError,
    _migrate_config,
)
from agents.model.model_version.retired_levers import (
    LAST_COMMIT_K2,
    LAST_COMMIT_L1,
    LAST_COMMIT_L2,
    RETIRED,
    RETIRED_FIELD_FLAGS,
    check_no_retired_levers,
    retired_lever_evidence,
)

#: Every retired field at its OFF value — what every v121+ run on record recorded.
_OFF = {
    "use_popart": False, "value_dist_mode": "none", "value_dist_bins": 0, "value_dist_vmin": 0.0,
    "value_dist_vmax": 0.0, "value_dist_coef": 1.0, "value_from_dist": False,
    "value_tail_weight": 0.0, "win_prob_coef": 1.0, "win_prob_pbrs_coef": 0.0,
    "win_prob_pbrs_source": None, "win_prob_pbrs_frozen": None,
    # L2 (config v132)
    "value_true_team": False, "dense_aux": False, "win_prob_dense_aux": 0.0,
    "win_prob_lambda": 1.0, "win_prob_lambda_truncated": "bootstrap",
    "win_prob_rollout_target": 0.0, "win_prob_rollout_r": 8, "win_prob_rollout_mode": "replace",
    "win_prob_rollout_weight": 1.0,
    # K2 (a metadata.json runtime knob; popped from a config if one ever carried it)
    "matmul_precision": "highest",
}


def _current_config() -> dict:
    """A CURRENT config as a v130 writer recorded it: today's fields + every retired field OFF."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    cfg = json.loads(ModelVersion.from_layout_and_policy_kwargs(layout, {"net_arch": [512, 512]})
                     .to_json())
    cfg.update(_OFF)
    cfg["config_version"] = 130
    return cfg


def test_the_table_covers_exactly_the_recorded_off_set():
    """`_OFF` here and `RETIRED_FIELDS` must name the same fields, or a deleted field has no test."""
    assert set(_OFF) == set(RETIRED_FIELDS)
    assert set(RETIRED_FIELD_FLAGS) == {r.field for r in RETIRED}
    assert all(f.startswith("--") for f in RETIRED_FIELD_FLAGS.values())


def test_a_v130_config_migrates_to_the_current_schema_and_stays_constructible():
    cfg = _current_config()
    out = _migrate_config(dict(cfg))
    assert not (set(out) & set(RETIRED_FIELDS)), "a retired field survived the migration"
    assert out["config_version"] == MODEL_CONFIG_VERSION
    ModelVersion(**out)      # `cls(**data)` TypeErrors on any stale key — the reason for the POP


@pytest.mark.parametrize("field,value", [("use_popart", True), ("value_dist_mode", "shaping"),
                                         ("value_dist_mode", "read_only"),
                                         ("value_from_dist", True),
                                         ("value_true_team", True), ("dense_aux", True)])
def test_a_STRUCTURAL_lever_recorded_ON_is_refused_on_every_load(field, value):
    cfg = {**_current_config(), field: value}
    with pytest.raises(ModelVersionError, match=RETIRED_FIELD_FLAGS[field].lstrip("-")):
        _migrate_config(cfg)


@pytest.mark.parametrize("field,value", [("value_tail_weight", 0.3), ("win_prob_pbrs_coef", 0.1),
                                         ("win_prob_pbrs_source", "models/x/final_model.zip"),
                                         ("win_prob_pbrs_frozen", "models/x"),
                                         ("win_prob_coef", 0.05),
                                         ("win_prob_lambda", 0.9), ("win_prob_rollout_target", 0.01),
                                         ("win_prob_rollout_weight", 64.0),
                                         ("win_prob_dense_aux", 0.1)])
def test_a_TRAINING_ONLY_lever_recorded_ON_still_LOADS_it_just_cannot_resume(field, value):
    """A frozen forward (an eval opponent, a pool snapshot, the prober) never reads these, so the
    migration pops them silently — the refusal belongs to the resume / fork path below."""
    cfg = {**_current_config(), field: value,
           "win_prob_mode": "shaping", "critic": "shaped"}     # `win_prob_coef` needs a head + shaped
    out = _migrate_config(cfg)
    assert field not in out


def _write(tmp_path, **over):
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "model_config.json").write_text(json.dumps({**_current_config(), **over}))
    (run / "final_model.zip").write_bytes(b"")     # a resume names a checkpoint; the config sits beside it
    return run


def test_an_all_OFF_parent_is_not_a_retired_lever_parent(tmp_path):
    run = _write(tmp_path)
    check_no_retired_levers(str(run / "model_config.json"))           # must not raise
    assert retired_lever_evidence(json.loads((run / "model_config.json").read_text())) == []


def test_a_missing_or_unreadable_config_is_not_a_verdict(tmp_path):
    check_no_retired_levers(None)
    check_no_retired_levers(str(tmp_path / "nope.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    check_no_retired_levers(str(bad))


@pytest.mark.parametrize("field,value", [
    ("use_popart", True), ("value_dist_mode", "shaping"), ("value_from_dist", True),
    ("value_tail_weight", 0.3), ("win_prob_pbrs_coef", 0.1),
    ("win_prob_pbrs_source", "models/x/final_model.zip"), ("win_prob_pbrs_frozen", "models/x"),
])
def test_a_resume_of_a_run_that_recorded_a_lever_ON_is_refused_naming_flag_and_pin(
        tmp_path, field, value):
    run = _write(tmp_path, **{field: value})
    with pytest.raises(RetiredLeverCheckpointError) as ei:
        check_no_retired_levers(str(run / "model_config.json"))
    msg = str(ei.value)
    assert RETIRED_FIELD_FLAGS[field] in msg, "the refusal must name the flag"
    assert LAST_COMMIT_L1[:8] in msg, "the refusal must name the pin that still has the lever"
    assert ei.value.last_commit == LAST_COMMIT_L1
    assert ei.value.evidence and "--pin-commit" in msg


@pytest.mark.parametrize("field,value", [
    ("value_true_team", True), ("dense_aux", True), ("win_prob_dense_aux", 0.1),
    ("win_prob_lambda", 0.95), ("win_prob_rollout_target", 0.002),
    ("win_prob_rollout_weight", 64.0),
])
def test_an_L2_lever_recorded_ON_refuses_a_resume_and_names_the_L2_pin(tmp_path, field, value):
    run = _write(tmp_path, **{field: value})
    with pytest.raises(RetiredLeverCheckpointError) as ei:
        check_no_retired_levers(str(run / "model_config.json"))
    msg = str(ei.value)
    assert RETIRED_FIELD_FLAGS[field] in msg and LAST_COMMIT_L2[:8] in msg
    assert ei.value.last_commit == LAST_COMMIT_L2


def test_an_L2_STRUCTURAL_refusal_names_its_own_unit_and_version():
    with pytest.raises(ModelVersionError, match=r"deletion pass L2, config v132"):
        _migrate_config({**_current_config(), "dense_aux": True})


def test_the_OFF_boundary_of_each_L2_lever_is_not_a_finding():
    """The defaults (and a tolerated absent record) are NOT evidence — the lambda below one and the
    weight above one are the ON positions, so exactly 1.0 must read OFF."""
    for field in ("win_prob_lambda", "win_prob_rollout_weight"):
        assert retired_lever_evidence({field: 1.0}) == []
        assert retired_lever_evidence({field: None}) == []
        assert retired_lever_evidence({}) == []
    assert [r.field for r in retired_lever_evidence({"win_prob_lambda": 0.999})] == ["win_prob_lambda"]
    assert [r.field for r in retired_lever_evidence({"win_prob_rollout_weight": 1.5})] == [
        "win_prob_rollout_weight"]


def test_the_aux_BCE_coefficient_only_matters_where_there_was_a_head_a_shaped_critic_and_a_non_default():
    base = {"win_prob_mode": "shaping", "critic": "shaped", "win_prob_coef": 0.05}
    assert [r.field for r in retired_lever_evidence(base)] == ["win_prob_coef"]
    assert retired_lever_evidence({**base, "critic": "winprob"}) == []       # the BCE IS the value loss
    assert retired_lever_evidence({**base, "win_prob_mode": "none"}) == []   # no head, nothing weighted
    assert retired_lever_evidence({**base, "win_prob_coef": 1.0}) == []      # the default stays default
    assert retired_lever_evidence({"win_prob_coef": None}) == []             # an absent record


def test_the_launch_path_exits_FATAL_CONFIG_on_such_a_parent(tmp_path):
    from main.exit_codes import TrainExitCode
    from main.train.config import enforce_not_shaped_parent

    run = _write(tmp_path, win_prob_pbrs_coef=0.1)
    with pytest.raises(SystemExit) as ei:
        enforce_not_shaped_parent(str(run))
    assert ei.value.code == int(TrainExitCode.FATAL_CONFIG)
    enforce_not_shaped_parent(str(_write(tmp_path / "clean")))     # an all-OFF parent passes


def test_checkargs_reports_the_same_finding_offline(tmp_path):
    from main.checkargs import retired_levers_finding

    run = _write(tmp_path, use_popart=True, value_tail_weight=0.2)
    f = retired_levers_finding(["--model", str(run / "final_model.zip")])
    assert f is not None and f["last_commit"] == LAST_COMMIT_L1
    assert any("--use-popart" in e for e in f["evidence"])
    assert any("--value-tail-weight" in e for e in f["evidence"])
    assert retired_levers_finding(["--model", str(_write(tmp_path / "ok") / "final_model.zip")]) is None
    assert retired_levers_finding(["--steps", "1"]) is None          # no --model: nothing to read


# ----------------------------------------------------------------------------- K2: TF32 (metadata.json)


def _record_precision(run, value):
    """The run's `metadata.json`, beside its model_config.json — where `matmul_precision` is recorded."""
    (run / "metadata.json").write_text(json.dumps({"saved_at": "x", "matmul_precision": value}))


def test_a_run_that_recorded_TF32_refuses_a_resume_naming_flag_and_pin(tmp_path):
    """The record lives in `metadata.json`, NOT `model_config.json`: `check_no_retired_levers` overlays it,
    so the one refusal path covers it. Revert the overlay ⇒ the clean model_config reads as an all-OFF
    parent and a TF32 run resumes silently at fp32."""
    run = _write(tmp_path)
    _record_precision(run, "high")
    assert retired_lever_evidence(json.loads((run / "model_config.json").read_text())) == [], \
        "precondition: the model_config alone carries no evidence"
    with pytest.raises(RetiredLeverCheckpointError) as ei:
        check_no_retired_levers(str(run / "model_config.json"))
    msg = str(ei.value)
    assert "--matmul-precision" in msg and LAST_COMMIT_K2[:8] in msg and "--pin-commit" in msg
    assert ei.value.last_commit == LAST_COMMIT_K2
    assert [e for e in ei.value.evidence if "matmul_precision" in e]


@pytest.mark.parametrize("record", ["highest", None])
def test_a_run_that_recorded_fp32_or_no_precision_is_not_a_finding(tmp_path, record):
    run = _write(tmp_path)
    if record is not None:
        _record_precision(run, record)
    check_no_retired_levers(str(run / "model_config.json"))           # must not raise
    (run / "metadata.json").write_text("{not json")                    # unreadable: not a verdict either
    check_no_retired_levers(str(run / "model_config.json"))


def test_a_TF32_run_is_refused_by_the_launch_path_and_by_checkargs(tmp_path):
    from main.checkargs import retired_levers_finding
    from main.exit_codes import TrainExitCode
    from main.train.config import enforce_not_shaped_parent

    run = _write(tmp_path)
    _record_precision(run, "high")
    with pytest.raises(SystemExit) as ei:
        enforce_not_shaped_parent(str(run))
    assert ei.value.code == int(TrainExitCode.FATAL_CONFIG)
    f = retired_levers_finding(["--model", str(run / "final_model.zip")])
    assert f is not None and f["last_commit"] == LAST_COMMIT_K2
    assert any("--matmul-precision" in e for e in f["evidence"])


def test_the_TF32_row_is_training_only_so_a_frozen_load_of_such_a_run_still_works(tmp_path):
    """A frozen forward (an eval opponent, a pool snapshot, the prober) never reads the precision, and
    `matmul_precision` is not a ModelVersion field: only a RESUME / FORK refuses."""
    row = next(r for r in RETIRED if r.field == "matmul_precision")
    assert row.structural is False and row.flag == "matmul-precision" and row.unit == "K2"
    out = _migrate_config({**_current_config(), "matmul_precision": "high"})
    assert "matmul_precision" not in out
    ModelVersion(**out)


# ----------------------------------------------------------------------------- the pickled zip


def test_sanitize_dead_policy_kwargs_pops_OFF_and_refuses_ON():
    from agents.model.snapshot import _DEAD_POLICY_KWARGS_JUDGED, sanitize_dead_policy_kwargs

    pk = {"net_arch": [512, 512], **{k: s for k, s in _DEAD_POLICY_KWARGS_JUDGED}}
    assert sanitize_dead_policy_kwargs(pk) is True
    assert pk == {"net_arch": [512, 512]}
    assert sanitize_dead_policy_kwargs(pk) is False
    for dead, supported in _DEAD_POLICY_KWARGS_JUDGED:
        with pytest.raises(ModelVersionError, match=dead):
            sanitize_dead_policy_kwargs({dead: not supported})


def _pre_deletion_zip(tmp_path):
    """A REAL checkpoint zip carrying what every v121-v130 writer pickled: the policy kwargs
    `use_popart` / `value_from_dist` and the extractor kwargs `value_dist_*`, all OFF."""
    from main.fresh_checkpoint import build_fresh_model

    model, _args, _pk = build_fresh_model(seed=3)
    fek = dict(model.policy_kwargs["features_extractor_kwargs"])
    fek.update(value_dist_mode="none", value_dist_bins=0, value_dist_vmin=0.0, value_dist_vmax=0.0,
               value_true_team=False, dense_aux=False)
    model.policy_kwargs = {**model.policy_kwargs, "features_extractor_kwargs": fek,
                           "use_popart": False, "value_from_dist": False}
    zip_path = tmp_path / "pre_deletion"
    model.save(str(zip_path))
    return str(zip_path) + ".zip"


def test_a_checkpoint_written_before_the_deletion_still_loads_through_the_loaders(tmp_path):
    from sb3_contrib import MaskablePPO

    from agents.model.snapshot import historical_load_kwargs

    zip_path = _pre_deletion_zip(tmp_path)
    # NON-VACUITY: the bare load is exactly the failure the sanitizers exist to prevent.
    with pytest.raises(TypeError):
        MaskablePPO.load(zip_path, env=None, device="cpu")
    # `play.py`'s ladder load, and the prober's, strip the deleted kwargs and load.
    kw = historical_load_kwargs(zip_path)
    assert "custom_objects" in kw
    model = MaskablePPO.load(zip_path, env=None, device="cpu", **kw)
    assert not hasattr(model.policy, "popart")
    from main.prober.model import sanitized_load_custom_objects
    custom, dropped = sanitized_load_custom_objects(zip_path)
    assert {"use_popart", "value_from_dist", "value_dist_mode", "value_true_team",
            "dense_aux"} <= set(dropped)
    assert MaskablePPO.load(zip_path, env=None, device="cpu", custom_objects=custom) is not None


def test_a_pre_deletion_checkpoint_loads_as_an_OPPONENT(tmp_path):
    """The path a self-play pool / an eval sentinel / a stable opponent takes."""
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings

    zip_path = _pre_deletion_zip(tmp_path)
    from main.fresh_checkpoint import build_fresh_model
    _model, args, _pk = build_fresh_model(seed=3)
    cfg_dir = tmp_path
    from agents.model.snapshot import save_model_snapshot
    version = ModelVersion.from_layout_and_policy_kwargs(
        _pk["features_extractor_kwargs"]["layout"], _pk, vf_coef=args.vf_coef)
    save_model_snapshot(str(cfg_dir), version, git_hash="test")
    opp, _ = load_foreign_opponent(zip_path, current_model_version(load_mappings()), device="cpu",
                                   config_path=str(cfg_dir / "model_config.json"))
    assert opp is not None


def test_a_pre_deletion_checkpoint_that_had_a_lever_ON_is_REFUSED_not_loaded(tmp_path):
    from agents.model.snapshot import historical_load_kwargs
    from main.fresh_checkpoint import build_fresh_model

    model, _args, _pk = build_fresh_model(seed=3)
    model.policy_kwargs = {**model.policy_kwargs, "use_popart": True}
    zip_path = tmp_path / "popart_on"
    model.save(str(zip_path))
    with pytest.raises(ModelVersionError, match="use_popart"):
        historical_load_kwargs(str(zip_path) + ".zip")


@pytest.mark.parametrize("dead", ["value_true_team", "dense_aux"])
def test_a_pre_deletion_checkpoint_that_had_an_L2_structural_lever_ON_is_REFUSED(tmp_path, dead):
    """The pickled extractor kwarg recorded ON names a module in the state_dict the surviving
    extractor has no home for — refused with the re-read diagnosis, never a bare TypeError."""
    from agents.model.snapshot import historical_load_kwargs
    from main.fresh_checkpoint import build_fresh_model

    model, _args, _pk = build_fresh_model(seed=3)
    fek = dict(model.policy_kwargs["features_extractor_kwargs"])
    fek[dead] = True
    model.policy_kwargs = {**model.policy_kwargs, "features_extractor_kwargs": fek}
    zip_path = tmp_path / f"{dead}_on"
    model.save(str(zip_path))
    with pytest.raises(ModelVersionError, match=dead):
        historical_load_kwargs(str(zip_path) + ".zip")
