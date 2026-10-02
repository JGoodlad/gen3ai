"""THE M5 SWITCH (`gen3_env_core_switch_v1`): `--env-core rust` is the PRODUCTION env core.

Pinned here, each failing on a revert of the piece it names:

* ``recipe.sizing`` is the ONE place the sizing verdict fills — the env core, N, the n_steps maximum,
  the collector's update size and T2's slots / buckets / lanes — and `--arch production` applies it
  like every recipe knob (a planted N in the block is the N a fresh launch resolves);
* an UNTYPED `--env-core` resolves: fresh `--arch production` → rust; `--model` (a same-run restart or
  a fork) → INHERITED, the core the checkpoint recorded (python when it predates the record); a bare
  non-production fresh argv → python (it defaults to `--critic shaped`, which the Rust core refuses);
* a TYPED `--env-core python` keeps the Python core reachable, reported as a TYPED deviation;
* the collector-only sizing rows are applied only on the Rust core, so `--arch production
  --env-core python` is not refused for flags that would be inert there;
* a same-run restart of an `--arch production` run restores the core from `cli_args`, and one
  launched before `--env-core` existed restores ``python`` — a run never changes core because the
  default moved.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json

import pytest

from main.train import recipe_surface as rs
from main.train.parser import build_parser
from main.train.rust_env_setup import resolve_env_core_default


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
def test_the_production_env_core_is_rust_and_lives_in_recipe_sizing():
    sz = rs.sizing_block()
    assert sz["env_core"] == "rust" and rs.production_env_core() == "rust"
    assert {r.dest for r in rs.SIZING_ROWS} | {rs.VERDICT_KEY} == set(sz)
    fresh = rs._raw_mirror()["recipe"]["fresh"]
    assert not {r.dest for r in rs.SIZING_ROWS} & set(fresh), "a size must live in ONE place"


def test_the_sizing_block_is_what_a_fresh_production_launch_resolves(tmp_path):
    planted = copy.deepcopy(rs._raw_mirror())
    planted["recipe"]["sizing"].update(n_envs=1024, n_steps=96, t2_buckets="64,256", t2_lanes=4)
    ns = _desugared(["--arch", "production"], mirror=planted)
    assert (ns.env_core, ns.n_envs, ns.n_steps) == ("rust", 1024, 96)
    assert (ns.t2_buckets, ns.t2_lanes) == ("64,256", 4)
    assert ns.trainee_slots is None                      # null = the collector derives it
    # the live block (the SIZING verdict's N* = 256 shape, 2026-10-02) through the real umbrella
    live = _desugared(["--arch", "production"])
    assert (live.env_core, live.n_envs, live.n_steps) == ("rust", 256, 384)
    assert live.t2_buckets is None and live.rollout_target_samples == 98304


def test_collector_rows_are_never_applied_on_the_python_core():
    planted = copy.deepcopy(rs._raw_mirror())
    planted["recipe"]["sizing"].update(t2_buckets="64,256", trainee_slots=2)
    ns = _desugared(["--arch", "production", "--env-core", "python"], mirror=planted)
    assert ns.env_core == "python" and ns.t2_buckets is None and ns.trainee_slots is None
    diffs = {d.dest: d for d in rs.diff_against_production(ns, planted)}
    assert set(diffs) == {"env_core"} and diffs["env_core"].source == "argv"


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
    core["recipe"]["sizing"]["env_core"] = "jax"
    with pytest.raises(rs.RecipeError, match="env_core"):
        rs.production_recipe(core)


# --------------------------------------------------------------- an UNTYPED --env-core resolves
def test_fresh_arch_production_resolves_rust_and_a_bare_argv_keeps_python():
    assert _desugared(["--arch", "production"]).env_core == "rust"
    bare = _desugared([])
    assert bare.env_core == "python"
    assert resolve_env_core_default(bare) is None           # nothing to resolve without --model


def test_a_typed_python_core_is_reachable_and_reported_TYPED():
    ns = _desugared(["--arch", "production", "--env-core", "python"])
    assert ns.env_core == "python"
    rep = rs.report(ns, fresh=True)
    assert [d.dest for d in rep.diffs] == ["env_core"] and not rep.refuses


@pytest.mark.parametrize("recorded,want", [("rust", "rust"), ("python", "python"), (None, "python")])
def test_a_fork_INHERITS_its_parents_core_and_a_typed_core_wins(tmp_path, recorded, want):
    _run_dir, ckpt = _run(tmp_path, env_core=recorded)
    ns = build_parser().parse_args(["--steps", "1", "--model", str(ckpt)])   # no --run-dir: a new run
    core = resolve_env_core_default(ns)
    assert core is not None and core[0] == want and ns.env_core == want
    for typed_core in ("python", "rust"):
        typed = build_parser().parse_args(["--steps", "1", "--model", str(ckpt), "--env-core", typed_core])
        assert resolve_env_core_default(typed) is None and typed.env_core == typed_core


@pytest.mark.parametrize("recorded,want", [("rust", "rust"), ("python", "python"), (None, "python")])
def test_a_same_run_restart_keeps_the_core_the_run_recorded(tmp_path, recorded, want):
    run, ckpt = _run(tmp_path, env_core=recorded)
    ns = build_parser().parse_args(["--steps", "1", "--model", str(ckpt), "--run-dir", str(run)])
    core = resolve_env_core_default(ns)
    assert core is not None and core[0] == want and ns.env_core == want


@pytest.mark.parametrize("cli,want", [({"env_core": "rust"}, "rust"), ({"env_core": "python"}, "python"),
                                      ({}, "python")])
def test_a_restart_of_an_arch_production_run_restores_its_core_from_cli_args(tmp_path, cli, want):
    full = dict(_production_cli())
    full.pop("env_core", None)
    full.update(cli)
    run, ckpt = _run(tmp_path, original="launcher --steps 9 --arch production --run-name x", cli=full)
    ns = build_parser().parse_args(["--steps", "1", "--model", str(ckpt), "--run-dir", str(run)])
    got = {d: (v, s) for d, v, s in rs.inherit_on_restart(ns, str(run), _saved())}
    assert ns.env_core == want and got["env_core"][0] == want
    assert resolve_env_core_default(ns) is None          # resolved by the restart route already


def _production_cli():
    ns = _desugared(["--arch", "production"])
    return {r.dest: getattr(ns, r.dest) for r in rs.ROWS}


def _saved():
    class _V:
        terminal_indicator, victory_value, draw_penalty, vf_coef = True, 1.0, 0.0, 0.5
    return _V()
