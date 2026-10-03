"""THE M5 SWITCH (`gen3_env_core_switch_v1`), as it stands after `--env-core` was deleted (deletion pass P11b):
the Rust env core is the ONLY core, so what is left to pin is keyed on the RECORD, never on a flag.

Pinned here, each failing on a revert of the piece it names:

* ``recipe.sizing`` is the ONE place the sizing verdict fills — N, the n_steps maximum, the collector's
  update size and T2's slots / buckets / lanes (the env core is no row of it) — and `--arch production`
  applies it like every recipe knob (a planted N in the block is the N a fresh launch resolves);
* a `--model` launch (a same-run restart or a fork) of a PYTHON-ERA checkpoint (recorded python, or
  predating the record) moves onto the Rust core, ANNOUNCED as a core switch (deletion pass D4,
  `env_core_switch_line`); a checkpoint that trained the SHAPED critic is REFUSED (D4: run it pinned,
  `refuse_python_era_checkpoint`), read off its recorded critic;
* a typed `--env-core` is REFUSED with the reason (it is a deleted flag: `census_deleted_flags_test`), and
  the namespace has no `env_core` attribute at all;
* a same-run restart of an `--arch production` run restores no env core from `cli_args` (nothing to restore).
"""
from __future__ import annotations

import contextlib
import copy
import io
import json

import pytest

from main.train import recipe_surface as rs
from main.train.parser import build_parser
from main.train.rust_env_setup import (PythonEraShapedCheckpoint, env_core_switch_line,
                                       refuse_python_era_checkpoint)


def _desugared(argv, mirror=None):
    from main.train.config import desugar_umbrella_flags
    ns = build_parser().parse_args(["--steps", "1", *argv])
    if mirror is None:
        with contextlib.redirect_stdout(io.StringIO()):
            desugar_umbrella_flags(ns)
    else:
        rs.apply_production_recipe(ns, mirror)
    return ns


def _run(tmp_path, *, env_core=None, original="launcher --steps 9 --run-name x", cli=None):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "c.zip"
    ckpt.write_bytes(b"")
    meta = {"original_command": original}
    if cli is not None:
        meta["cli_args"] = cli
    if env_core is not None:
        meta["env_core"] = {"env_core": env_core}
    (run / "metadata.json").write_text(json.dumps(meta))
    return run, ckpt


# ------------------------------------------------------------------------- the single place
def test_the_sizes_live_in_recipe_sizing_and_the_env_core_is_not_a_row_of_it():
    sz = rs.sizing_block()
    assert "env_core" not in sz and not hasattr(rs, "production_env_core")
    assert {r.dest for r in rs.SIZING_ROWS} | {rs.VERDICT_KEY} == set(sz)
    fresh = rs._raw_mirror()["recipe"]["fresh"]
    assert not {r.dest for r in rs.SIZING_ROWS} & set(fresh), "a size must live in ONE place"


def test_the_sizing_block_is_what_a_fresh_production_launch_resolves(tmp_path):
    planted = copy.deepcopy(rs._raw_mirror())
    planted["recipe"]["sizing"].update(n_envs=1024, n_steps=96, t2_buckets="64,256", t2_lanes=4)
    ns = _desugared(["--arch", "production"], mirror=planted)
    assert (ns.n_envs, ns.n_steps) == (1024, 96)
    assert (ns.t2_buckets, ns.t2_lanes) == ("64,256", 4)
    assert ns.trainee_slots is None                      # null = the collector derives it
    # the live block (the SIZING verdict's N* = 256 shape, 2026-10-02) through the real umbrella
    live = _desugared(["--arch", "production"])
    assert (live.n_envs, live.n_steps) == (256, 384)
    assert live.t2_buckets is None and live.rollout_target_samples == 98304


def test_a_typed_env_core_is_refused_with_the_reason_and_leaves_no_attribute(capsys):
    """The Python env core is gone (U3) and the one-valued flag with it (P11b): the refusal names why."""
    for value in ("python", "rust"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--steps", "1", "--arch", "production", "--env-core", value])
        err = " ".join(capsys.readouterr().err.split())
        assert "--env-core was DELETED" in err and "ONLY env core" in err, err
    assert not hasattr(build_parser().parse_args(["--steps", "1"]), "env_core")


def test_a_sizing_block_defect_is_refused_by_name():
    raw = rs._raw_mirror()
    dup = copy.deepcopy(raw)
    dup["recipe"]["fresh"]["n_envs"] = 48
    with pytest.raises(rs.RecipeError, match="one place"):
        rs.production_recipe(dup)
    stray = copy.deepcopy(raw)
    stray["recipe"]["sizing"]["n_slots"] = 4
    with pytest.raises(rs.RecipeError, match="sizing"):
        rs.production_recipe(stray)
    core = copy.deepcopy(raw)
    core["recipe"]["sizing"]["env_core"] = "rust"           # the deleted row: a stray key, refused by name
    with pytest.raises(rs.RecipeError, match="env_core"):
        rs.production_recipe(core)


# ------------------------------------------------------- the record-keyed core switch (no flag to type)
def test_fresh_arch_production_and_a_bare_argv_have_no_core_to_resolve_or_switch():
    for argv in (["--arch", "production"], []):
        ns = _desugared(argv)
        assert not hasattr(ns, "env_core")
        assert env_core_switch_line(ns) is None             # a fresh run has nothing to switch from
    rep = rs.report(_desugared(["--arch", "production"]), fresh=True)
    assert not rep.diffs and not rep.refuses                # production applies the whole recipe: nothing differs


@pytest.mark.parametrize("recorded", ["rust", "python", None])
def test_a_launch_MOVES_a_python_era_checkpoint_announced_and_keeps_a_rust_one(tmp_path, recorded):
    """A fork (no --run-dir) and a same-run restart (--run-dir) read the same record."""
    run, ckpt = _run(tmp_path, env_core=recorded)
    for extra in ([], ["--run-dir", str(run)]):
        ns = build_parser().parse_args(["--steps", "1", "--model", str(ckpt), *extra])
        refuse_python_era_checkpoint(str(ckpt))             # no critic record readable here: no refusal
        assert ("CORE SWITCH" in (env_core_switch_line(ns) or "")) is (recorded != "rust")   # D4: never silent


def test_a_restart_of_an_arch_production_run_restores_no_env_core(tmp_path):
    """The restart route restored `env_core` from `cli_args` while it was a recipe row; it is not one now."""
    full = dict(_production_cli())
    full["env_core"] = "python"                              # a pre-P11b run's cli_args still carries it
    run, ckpt = _run(tmp_path, original="launcher --steps 9 --arch production --run-name x", cli=full)
    ns = build_parser().parse_args(["--steps", "1", "--model", str(ckpt), "--run-dir", str(run)])
    got = {d for d, _v, _s in rs.inherit_on_restart(ns, str(run), _saved())}
    assert "env_core" not in got and not hasattr(ns, "env_core")


# ----------------------------------------------------- deletion pass D4: the python-era resume rule
@pytest.mark.parametrize("cfg", [{"critic": "shaped"}, {}])        # {} = recorded before --critic
@pytest.mark.parametrize("env_core", ["python", "rust"])
def test_a_SHAPED_critic_checkpoint_is_REFUSED_whatever_the_core(tmp_path, cfg, env_core):
    run, ckpt = _run(tmp_path, env_core=env_core)
    (run / "model_config.json").write_text(json.dumps(cfg))
    with pytest.raises(PythonEraShapedCheckpoint, match="PINNED"):
        refuse_python_era_checkpoint(str(ckpt))


def test_a_WINPROB_python_era_checkpoint_moves_to_rust_announced(tmp_path):
    run, ckpt = _run(tmp_path, env_core="python")
    (run / "model_config.json").write_text(json.dumps({"critic": "winprob"}))
    ns = build_parser().parse_args(["--steps", "1", "--model", str(ckpt)])
    refuse_python_era_checkpoint(str(ckpt))                  # winprob: not refused
    assert env_core_switch_line(ns).startswith("🔀 [ENV CORE] CORE SWITCH")


def test_the_launch_path_exits_FATAL_CONFIG_on_a_shaped_parent(monkeypatch, capsys):
    from types import SimpleNamespace

    from main.exit_codes import TrainExitCode
    from main.train import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "_load_saved_version", lambda path: SimpleNamespace(critic="shaped"))
    p = build_parser()
    args = p.parse_args(["--steps", "1", "--model", "models/parent/checkpoints/c_10_steps.zip"])
    with pytest.raises(SystemExit) as exc:
        cfg_mod.resolve_config(args, p)
    assert exc.value.code == int(TrainExitCode.FATAL_CONFIG)
    assert "deletion pass D4" in capsys.readouterr().err


def test_checkargs_reports_the_shaped_refusal(tmp_path):
    from main.checkargs import resolve_against_parent

    from agents.model.model_version import ModelVersion
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    run, ckpt = _run(tmp_path, env_core="python")
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    (run / "model_config.json").write_text(ModelVersion.from_layout_and_policy_kwargs(
        layout, {"net_arch": [512, 512], "critic": "shaped"}).to_json())
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        res = resolve_against_parent(["--steps", "1", "--model", str(ckpt)])
    assert "deletion pass D4" in (res.get("env_core_refusal") or ""), res


def _production_cli():
    ns = _desugared(["--arch", "production"])
    return {r.dest: getattr(ns, r.dest) for r in rs.ROWS}


def _saved():
    class _V:
        terminal_indicator, victory_value, draw_penalty, vf_coef = True, 1.0, 0.0, 0.5
    return _V()
