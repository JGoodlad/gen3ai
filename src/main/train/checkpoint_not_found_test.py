"""A `--model` that names NO checkpoint is refused as ITSELF — "file not found" — never as a recipe value or a precision
the missing record could not supply (`checkpoint_state.require_checkpoint`, 2026-10-25).

THE INCIDENT (the x26c canary-margin read, F7; 0dcf0ea4 FINDING 6). Re-reading `rb_x26c_s2601`'s 479M checkpoint at
HEAD refused "could not find --vf-coef (the checkpoint's model_config.json)" although the run dir's
`model_config.json` records ``vf_coef: 0.5``. Root cause: the zip had been REMOVED by the run's checkpoint retention
(its step sidecar ``checkpoint_479000039_steps.json`` left behind). `_load_saved_version` resolved the config through
`snapshot._resolve_paths`, which raises for a missing zip, swallowed that as "unreadable config" and returned None —
while the launch path's own refusals (`shaped_reward.saved_config_path`) fall back to the directory and read the
run's config: two resolvers for one file. The recipe resolver then reported the `model_config`-route row as missing,
and with ``--vf-coef`` typed through, the read went on to "the run's resolved --trunk-precision is 'fp32'" (the
untyped default). Here, on a fixture shaped like the real run dir (run-level `model_config.json` + `metadata.json`
cli_args, `checkpoints/` holding only the step sidecar):

* the launch resolution exits FATAL_CONFIG naming the missing checkpoint and its sidecar, and NOT --vf-coef — typed
  or not (fails on revert: the refusal named --vf-coef);
* an EXISTING checkpoint beside the same record resolves ``vf_coef`` 0.5 from the run's config;
* a model_config-route miss names the FILE and the KEY (and the recorded value when the file has one);
* `_load_saved_version` raises `CheckpointNotFoundError` for a missing zip and reads THE one resolver's file."""
from __future__ import annotations

import contextlib
import io
import json

import pytest

from main.checkargs_test import _minimal_model_config
from main.exit_codes import TrainExitCode
from main.train import recipe_surface as rs
from main.train.checkpoint_state import CheckpointNotFoundError, _load_saved_version, require_checkpoint
from main.train.recipe_testkit import write_recorded_recipe

GONE = "checkpoint_479000039_steps"          # the real rotated-away step (rb_x26c_s2601)
KEPT = "checkpoint_485000240_steps"


@pytest.fixture
def x26c_run(tmp_path):
    """The `rb_x26c_s2601` shape: the run-level config recording vf_coef 0.5, the recorded recipe, one checkpoint
    zip still present and one removed by retention (its step sidecar left)."""
    run = tmp_path / "rb_x26c_s2601"
    (run / "checkpoints").mkdir(parents=True)
    (run / "model_config.json").write_text(json.dumps(_minimal_model_config(vf_coef=0.5, trunk_precision="bf16")))
    (run / "metadata.json").write_text(json.dumps({"original_command": "launcher --steps 1"}))
    write_recorded_recipe(run)
    (run / "checkpoints" / f"{GONE}.json").write_text(json.dumps({"num_timesteps": 479000039}))
    (run / "checkpoints" / f"{KEPT}.json").write_text(json.dumps({"num_timesteps": 485000240}))
    (run / "checkpoints" / f"{KEPT}.zip").write_bytes(b"")
    return run


def _resolve(argv):
    from main.train.config import resolve_config
    from main.train.parser import build_parser
    p = build_parser()
    a = p.parse_args(["--steps", "100"] + list(argv))
    err, out = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        try:
            resolve_config(a, p)
            code = None
        except SystemExit as e:
            code = e.code
    return a, code, err.getvalue() + out.getvalue()


@pytest.mark.parametrize("typed", [[], ["--vf-coef", "0.5"]])
def test_a_REMOVED_checkpoint_is_refused_as_file_not_found_never_as_vf_coef(x26c_run, typed):
    gone = str(x26c_run / "checkpoints" / f"{GONE}.zip")
    _a, code, text = _resolve(["--model", gone] + typed)
    assert code == int(TrainExitCode.FATAL_CONFIG), text
    assert "checkpoint file not found" in text and f"{GONE}.json exists" in text, text
    assert "--vf-coef" not in text and "trunk-precision" not in text, text


def test_an_EXISTING_checkpoint_beside_the_same_record_reads_vf_coef_from_the_runs_config(x26c_run):
    kept = str(x26c_run / "checkpoints" / f"{KEPT}.zip")
    saved = _load_saved_version(kept)
    assert saved is not None and saved.vf_coef == 0.5
    from main.train.parser import build_parser
    ns = build_parser().parse_args(["--steps", "1", "--model", kept])
    got = {d: (v, s) for d, v, s in rs.inherit_recorded_recipe(ns, None, saved)}
    assert got["vf_coef"] == (0.5, "model_config.json")


def test_a_model_config_miss_names_the_FILE_and_the_KEY(x26c_run, tmp_path):
    gone = str(x26c_run / "checkpoints" / f"{GONE}.zip")
    kept = str(x26c_run / "checkpoints" / f"{KEPT}.zip")
    cfg = str(x26c_run / "model_config.json")
    assert "'vf_coef'" in rs.model_config_miss(gone, "vf_coef") and "file not found" in rs.model_config_miss(
        gone, "vf_coef")
    # the config exists and records the key, but was not read under this tree: the file, the key AND the value
    msg = rs.model_config_miss(kept, "vf_coef", None)
    assert cfg in msg and "'vf_coef'" in msg and "vf_coef=0.5" in msg, msg
    from main.train.parser import build_parser
    ns = build_parser().parse_args(["--steps", "1", "--model", kept])
    with pytest.raises(rs.RecipeRecordError) as exc:
        rs.inherit_recorded_recipe(ns, None, None)
    assert "--vf-coef" in str(exc.value) and cfg in str(exc.value) and "vf_coef=0.5" in str(exc.value)
    # no config anywhere beside the checkpoint
    lone = tmp_path / "lone" / "checkpoints"
    lone.mkdir(parents=True)
    (lone / "c.zip").write_bytes(b"")
    assert "no model_config.json in" in rs.model_config_miss(str(lone / "c.zip"), "vf_coef")


def test_load_saved_version_raises_for_a_missing_zip_and_reads_the_one_resolvers_file(x26c_run):
    gone = str(x26c_run / "checkpoints" / f"{GONE}.zip")
    with pytest.raises(CheckpointNotFoundError, match="file not found"):
        _load_saved_version(gone)
    with pytest.raises(CheckpointNotFoundError):
        require_checkpoint(str(x26c_run / "checkpoints" / "never_written.zip"))
    from agents.model.model_version.shaped_reward import saved_config_path
    kept = str(x26c_run / "checkpoints" / f"{KEPT}.zip")
    assert saved_config_path(kept) == str(x26c_run / "model_config.json")
    assert require_checkpoint(kept) == kept


def test_the_diagnostic_build_names_a_missing_checkpoint_first(x26c_run):
    """0dcf0ea4 FINDING 6: `diag_bf16_canary.build_learner` said "--trunk-precision is 'fp32'" for a missing zip."""
    from pathlib import Path

    from main.ops import diag_bf16_canary as D
    with pytest.raises(D.DiagError, match="file not found"):
        D.build_learner(Path(x26c_run / "checkpoints" / f"{GONE}.zip"), "cpu")
