"""`gen3_shaped_reward_deletion_v1` — a resume or fork of a SHAPED-reward checkpoint REFUSES.

Owner decision, 2026-09-26: once the shaped reward path is deleted, a checkpoint trained WITH it
must never continue silently on the terminal alone (that would change the experiment under the
same run name). Every test here FAILS if the refusal is reverted — the launch path
(`resolve_config`) exits 0-and-continues, or `checkargs` says the command launches.

And the other direction, which matters as much: a production (terminal-only) v121 checkpoint —
the live burn-in and the new lineage are exactly that — must still load and resume on this code.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from agents.model.model_version import ModelVersionError, _migrate_config
from agents.model.model_version.shaped_reward import (
    DELETED_SHAPED_REWARD_FIELDS, LAST_SHAPED_COMMIT, SHAPED_DELETION_VERSION,
    ShapedRewardCheckpointError, check_not_shaped, saved_config_path, shaped_reward_evidence,
)
from main.exit_codes import TrainExitCode


def _production_v121() -> dict:
    """The production mirror as a v121 run recorded it — `hand_shaping: False` and the other 13
    deleted fields present (the mirror dropped them at the deletion; a v121 file still has them)."""
    from agents.training.baselines import production_config
    raw = dict(production_config())
    raw.update(config_version=121, hand_shaping=False, pbrs_material=True, pbrs_belief=True,
               all_shaping_pbrs=True, stall_pbrs=False, bias_redesign=False,
               bias_additivity=1.0, mat_alive_weight=1.25, no_progress_penalty=0.15,
               switch_bias_weight=0.0, self_ko_hp_penalty=0.0, drop_redundant_bias=False,
               drop_switch_bias=False, no_progress_tax_armed=False)
    return raw


def _shaped_v121() -> dict:
    raw = _production_v121()
    raw["hand_shaping"] = True
    return raw


def _write_run(tmp_path, raw: dict) -> str:
    """A run dir with a config + an (empty) checkpoint zip beside it; returns the zip path."""
    run = tmp_path / "run_shaped"
    ckpt = run / "checkpoints"
    ckpt.mkdir(parents=True)
    (run / "model_config.json").write_text(json.dumps(raw))
    z = ckpt / "ckpt_1000_steps.zip"
    z.write_bytes(b"")
    return str(z)


# ─── the predicate ──────────────────────────────────────────────────────────────────────────────

def test_the_production_record_is_NOT_shaped():
    assert shaped_reward_evidence(_production_v121()) == []


def test_hand_shaping_True_IS_shaped_and_says_so():
    ev = shaped_reward_evidence(_shaped_v121())
    assert len(ev) == 1 and "hand_shaping=True" in ev[0] and "--hand-shaping" in ev[0]


def test_an_ABSENT_hand_shaping_means_shaped_before_the_deletion_and_not_after():
    """Pre-v105 configs had no key and every hand term was unconditional; post-deletion configs
    have no key because there is nothing left to switch."""
    assert shaped_reward_evidence({"config_version": 104})
    assert shaped_reward_evidence({"config_version": 121})
    assert shaped_reward_evidence({"config_version": SHAPED_DELETION_VERSION}) == []


def test_a_re_armed_no_progress_tax_is_shaped_unless_stall_pbrs_had_zeroed_it():
    raw = _production_v121()
    raw["no_progress_tax_armed"] = True
    assert "no_progress_tax_armed=True" in shaped_reward_evidence(raw)[0]
    raw["stall_pbrs"] = True            # the tilt's own gate: (redesign or asp) and not stall
    assert shaped_reward_evidence(raw) == []


# ─── the typed error ────────────────────────────────────────────────────────────────────────────

def test_check_not_shaped_raises_the_TYPED_error_naming_the_flag_and_the_fix(tmp_path):
    p = tmp_path / "model_config.json"
    p.write_text(json.dumps(_shaped_v121()))
    with pytest.raises(ShapedRewardCheckpointError) as e:
        check_not_shaped(str(p))
    assert isinstance(e.value, ModelVersionError)          # every FATAL_CONFIG handler catches it
    msg = str(e.value)
    assert "hand_shaping=True" in msg
    assert f"≤ {LAST_SHAPED_COMMIT[:8]}" in msg and "--pin-commit" in msg
    assert "silently" in msg
    assert e.value.config_path == str(p)


def test_a_missing_config_is_not_a_verdict(tmp_path):
    check_not_shaped(None)
    check_not_shaped(str(tmp_path / "nope.json"))


# ─── the LAUNCH path: resolve_config (resume AND fork) ──────────────────────────────────────────

def _resolve(argv):
    from main.train.config import resolve_config
    from main.train_rl_agent import build_parser
    parser = build_parser()
    return resolve_config(parser.parse_args(["--steps", "1", "--debug", *argv]), parser)


def test_resolve_config_REFUSES_a_shaped_parent_with_FATAL_CONFIG(tmp_path, capsys):
    """THE REVERT-CATCHER for the launch path: without `enforce_not_shaped_parent` this resolves
    (the migration pops the shaped fields silently) and the run continues on the terminal alone."""
    zip_path = _write_run(tmp_path, _shaped_v121())
    assert saved_config_path(zip_path) == str(tmp_path / "run_shaped" / "model_config.json")
    with pytest.raises(SystemExit) as e:
        _resolve(["--model", zip_path])
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    out = capsys.readouterr().out
    assert "[ModelVersion] FATAL" in out and "TRAINED WITH A SHAPED REWARD" in out
    assert LAST_SHAPED_COMMIT[:8] in out


def test_resolve_config_ACCEPTS_a_production_v121_parent(tmp_path, capsys):
    zip_path = _write_run(tmp_path, _production_v121())
    # (the parent is a `--critic winprob` run, so the three terminal flags it REQUIRES are re-passed)
    _resolve(["--model", zip_path, "--terminal-indicator", "--victory-value", "1.0",
              "--draw-penalty", "0"])                      # no SystemExit
    assert "SHAPED REWARD" not in capsys.readouterr().out


def test_a_production_v121_config_still_LOADS_with_the_shaped_fields_popped(tmp_path):
    """Frozen loads (opponents, teachers, the prober) and resumes of the live v121 runs."""
    from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersion
    out = _migrate_config(_production_v121())
    assert not (set(DELETED_SHAPED_REWARD_FIELDS) & set(out))
    # migrated all the way to HEAD, which is at or past the deletion (v123 added policy_gae_lambda)
    assert out["config_version"] == MODEL_CONFIG_VERSION >= SHAPED_DELETION_VERSION
    p = tmp_path / "model_config.json"
    p.write_text(json.dumps(_production_v121()))
    v = ModelVersion.from_json_file(str(p))
    assert v.terminal_indicator is True and v.victory_value == 1.0


# ─── the OFFLINE path: checkargs ────────────────────────────────────────────────────────────────

def test_checkargs_says_a_shaped_parent_WOULD_FAIL(tmp_path, capsys):
    """THE REVERT-CATCHER for the offline answer: 'does this launch' must say no."""
    from main.checkargs import main as checkargs_main, shaped_reward_finding
    zip_path = _write_run(tmp_path, _shaped_v121())
    assert shaped_reward_finding(["--model", zip_path])["evidence"]
    rc = checkargs_main(["--argv", f"--model {zip_path} --steps 10"])
    out = capsys.readouterr().out
    assert "shaped-reward parent             : YES" in out
    assert rc == int(TrainExitCode.FATAL_CONFIG)
    assert "this command still launches" not in out


def test_checkargs_passes_the_shaped_check_for_a_production_parent(tmp_path):
    from main.checkargs import shaped_reward_finding
    assert shaped_reward_finding(["--model", _write_run(tmp_path, _production_v121())]) is None
    assert shaped_reward_finding(["--steps", "10"]) is None


# ─── the named pin is real ──────────────────────────────────────────────────────────────────────

def _git(*args):
    from utils.paths import repo_root
    return subprocess.run(["git", *args], cwd=str(repo_root()), capture_output=True, text=True)


def test_the_named_pre_deletion_commit_STILL_HAS_the_shaped_path():
    """The fix the refusal prints must work: that commit's tree carries the shaped modules and the
    flag the refusal names. A wrong sha here would send an operator to a tree that cannot help."""
    if _git("cat-file", "-e", f"{LAST_SHAPED_COMMIT}^{{commit}}").returncode != 0:
        pytest.skip("the pre-deletion commit is not in this clone (shallow?)")
    for path in ("src/agents/training/reward_potentials.py",
                 "src/agents/training/reward_bias_terms.py"):
        assert _git("cat-file", "-e", f"{LAST_SHAPED_COMMIT}:{path}").returncode == 0, path
    parser_src = _git("show", f"{LAST_SHAPED_COMMIT}:src/main/train/parser/clean_world.py").stdout
    assert '"--hand-shaping"' in parser_src
    from main.checkargs import pin_predates_shaped_deletion
    assert pin_predates_shaped_deletion(LAST_SHAPED_COMMIT) is True
    assert pin_predates_shaped_deletion("0" * 40) is None


def test_the_deleted_field_list_matches_what_left_the_version_record():
    from dataclasses import fields as dc_fields
    from agents.model.model_version import _REWARD_IMMUTABLE_FIELDS, ModelVersionFields
    live = {f.name for f in dc_fields(ModelVersionFields)}
    assert not (set(DELETED_SHAPED_REWARD_FIELDS) & live)
    assert not (set(DELETED_SHAPED_REWARD_FIELDS) & set(_REWARD_IMMUTABLE_FIELDS))
    assert len(DELETED_SHAPED_REWARD_FIELDS) == 14
    assert os.path.basename(__file__) == "shaped_reward_test.py"
