"""K10(a) THE RECIPE SURFACE — `main.train.recipe_surface`, and the surfaces that read it.

Every test here FAILS on a revert of the piece it names: the umbrella's recipe half, the typed
record, the fresh-argv refusal, the fresh / fork split, the restart routes and their refusal, the
mirror strip + exemption + agreement, the KL-controller pin, and the `--sync-config` carry-over.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
import inspect
import io
import json
import pathlib
import types

import pytest

from agents.training import baselines
from main.train import recipe_surface as rs
from main.train.parser import build_parser


def _desugared(argv):
    from main.train.config import desugar_umbrella_flags
    ns = build_parser().parse_args(["--steps", "1", *argv])
    with contextlib.redirect_stdout(io.StringIO()):
        desugar_umbrella_flags(ns)
    return ns


# ------------------------------------------------------------------------------ the umbrella
def test_a_fresh_arch_production_argv_omitting_ent_coef_resolves_to_N0s_recipe():
    ns = _desugared(["--arch", "production"])
    assert ns.ent_coef == 0.05                      # parser default 0.02
    want = rs.production_recipe()
    for r in rs.ROWS:
        assert getattr(ns, r.dest) == want[r.dest], r.dest
    assert (ns.n_envs, ns.batch_size, ns.grad_accum_steps, ns.n_epochs, ns.lr) == (256, 2048, 32, 10, 3e-4)
    assert ns.clip_range_vf is None and ns.self_play is True and ns.critic == "winprob"
    assert ns.opp_intent_coef == 0.05 and ns.max_lr is None
    assert ns.recipe_source.startswith("production_config@")


def test_an_explicit_token_wins_even_when_it_equals_the_parser_default():
    ns = _desugared(["--arch", "production", "--ent-coef", "0.02", "--no-self-play"])
    assert ns.ent_coef == 0.02 and ns.self_play is False
    assert {"ent_coef", "self_play"} <= rs.typed_dests(ns)
    assert ns.n_envs == 256                         # the untyped rest still applied


def test_without_the_umbrella_nothing_is_applied():
    ns = _desugared([])
    assert ns.ent_coef == 0.02 and ns.n_envs == 32 and rs.typed_dests(ns) == frozenset()


# ------------------------------------------------------------------ the fresh / fork split
def test_E5s_five_epochs_live_only_in_the_fork_block_beside_its_frozen_lr():
    """E5 (5 epochs) was measured at a FROZEN 5.6e-5 fork LR. It is never paired with the fresh
    3e-4 — that pairing was never measured (Decision record)."""
    fresh, fork = rs.production_recipe(), rs.production_recipe(kind="fork")
    assert fresh["n_epochs"] == 10 and fresh["lr"] == 3e-4
    assert fork["n_epochs"] == 5 and fork["fork_lr"] == 5.6e-05 and fork["fork_lr_freeze"] is True
    raw = rs._raw_mirror()
    assert raw["recipe"]["fork"]["n_epochs"] == 5 and "fork_lr" not in raw["recipe"]["fresh"]


def test_the_kl_controller_record_is_what_the_trainer_constructs():
    """N0 ran `AdaptivePPOCallback` with its constructor defaults (no flag sets them); a change to
    those defaults must move the recipe record with it."""
    from agents.training.adaptive_lr_callback import AdaptivePPOCallback, TwoPhaseLRCallback
    kl = rs.kl_controller()
    for cls in (AdaptivePPOCallback, TwoPhaseLRCallback):
        sig = inspect.signature(cls.__init__).parameters
        for k in rs.KL_CONSTANTS:
            assert sig[k].default == kl[k], (cls.__name__, k)


# ------------------------------------------------------------------------------- the refusal
def _check(argv):
    from main.checkargs import check
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return check(["--steps", "1", *argv])


def test_checkargs_refuses_a_fresh_argv_whose_untyped_recipe_differs():
    rep = _check(["--allow-nonproduction-arch"])["recipe"]
    assert rep.fresh and rep.refuses and rep.kind == "fresh"
    silent = {d.dest for d in rep.silent}
    assert {"n_envs", "batch_size", "grad_accum_steps", "n_epochs", "ent_coef", "clip_range_vf",
            "self_play", "opp_intent_coef"} <= silent
    # the bare-argv flip (deletion pass D2): its three reward values
    # now AGREE with the recipe untyped, so none of them is a silent difference any more
    assert not {"terminal_indicator", "victory_value", "draw_penalty", "gamma"} & silent
    assert any("--ent-coef" in line for line in rs.report_lines(rep))


def test_checkargs_main_exits_nonzero_on_recipe_drift_alone(capsys):
    from main.checkargs import main
    assert main(["--argv", "--steps 1 --arch production --clip-range-vf 0.5 --ent-coef 0.05"]) == 0
    assert main(["--argv", "--steps 1 --allow-nonproduction-arch"]) == 1
    assert "trains a recipe nobody chose" in capsys.readouterr().out


def test_a_typed_deviation_or_the_consent_flag_does_not_refuse():
    typed = _check(["--arch", "production", "--n-epochs", "2", "--ent-coef", "0.01"])["recipe"]
    assert {d.dest for d in typed.diffs} == {"n_epochs", "ent_coef"} and not typed.refuses
    allowed = _check(["--allow-nonproduction-arch", "--allow-nonproduction-recipe"])["recipe"]
    assert allowed.silent and not allowed.refuses


def test_the_production_umbrella_passes_clean():
    res = _check(["--arch", "production"])
    assert not res["recipe"].diffs and not res["recipe"].refuses


def test_a_fork_is_info_only_and_is_judged_against_the_fork_recipe(tmp_path):
    ckpt = tmp_path / "parent" / "checkpoints" / "c.zip"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    rep = _check(["--model", str(ckpt), "--fork-lr", "2.8e-05", "--fork-lr-freeze",
                  "--n-epochs", "10"])["recipe"]
    assert not rep.fresh and not rep.refuses and rep.kind == "fork"
    assert {"fork_lr", "n_epochs"} <= {d.dest for d in rep.diffs}


# ---------------------------------------------------------------------------- the restart path
_LAUNCH_CLI = {"ent_coef": 0.05, "n_envs": 48, "grad_accum_steps": 32, "n_epochs": 10,
               "min_lr": 1e-05, "max_lr": None, "anneal_lr_start_steps": None,
               "weight_decay": 1e-05, "clip_range": 0.15, "clip_range_vf": None, "self_play": True,
               "beta_setvalued_coef": 0.05, "opp_intent_coef": 0.05,
               # INERT on resume: deliberately DIFFERENT from the parser's, to prove they are
               # never written back
               "lr": 9e-9, "batch_size": 7, "n_steps": 3, "gamma": 0.5}
_SAVED = types.SimpleNamespace(terminal_indicator=True, victory_value=1.0, draw_penalty=0.0,
                               vf_coef=0.5)


def _run(tmp_path, original_command, cli_args):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "c.zip").write_bytes(b"")
    meta = {"original_command": original_command}
    if cli_args is not None:
        meta["cli_args"] = cli_args
    (run / "metadata.json").write_text(json.dumps(meta))
    return run


def _restart_ns(run, *extra):
    return build_parser().parse_args(["--steps", "1", "--model", str(run / "checkpoints" / "c.zip"),
                                      "--run-dir", str(run), *extra])


def test_each_row_has_exactly_one_restart_route():
    fields = rs._model_version_fields()
    p = build_parser()
    route = {r.dest: rs.restart_route(r.dest, p.get_default(r.dest), fields) for r in rs.ROWS}
    assert {d for d, v in route.items() if v == "inert"} == {"lr", "batch_size", "n_steps", "gamma"}
    assert "critic" not in route and route["move_belief_coef"] == "resume"     # the critic is no recipe row (P11b)
    assert route["policy_gae_lambda"] == "resume"
    assert {route[d] for d in ("terminal_indicator", "victory_value", "draw_penalty", "vf_coef")} \
        == {"model_config"}
    assert {route[d] for d in ("n_envs", "n_epochs", "ent_coef", "self_play", "beta_setvalued_coef")} \
        == {"cli_args"}


def test_a_same_run_restart_resolves_every_untyped_row_by_its_route(tmp_path):
    run = _run(tmp_path, "launcher --steps 9 --arch production --run-name x", _LAUNCH_CLI)
    ns = _restart_ns(run, "--ent-coef", "0.03")
    got = {d: (v, s) for d, v, s in rs.inherit_on_restart(ns, str(run), _SAVED)}
    assert ns.ent_coef == 0.03 and "ent_coef" not in got                 # typed wins
    assert got["n_envs"] == (48, "metadata.json:cli_args") and ns.n_epochs == 10
    assert ns.clip_range_vf is None and ns.beta_setvalued_coef == 0.05
    # value-CHECKED recorded fields come from the CHECKPOINT, not cli_args:
    assert got["terminal_indicator"] == (True, "model_config.json") and ns.draw_penalty == 0.0
    # INERT on resume — never written back, whatever cli_args says:
    assert (ns.lr, ns.batch_size, ns.n_steps, ns.gamma) == (3e-4, 4096, 2048, None)
    assert not {"lr", "batch_size", "n_steps", "gamma"} & set(got)
    # recorded tri-state fields are the resume's own inheritance:
    assert ns.critic == "winprob" and ns.move_belief_coef is None      # a constant; the tri-state ones inherit
    rep = rs.report(ns, fresh=False, restart=True)
    assert rep.restart_inherited and not rep.refuses and rep.kind == "fresh"


def test_a_missing_value_REFUSES_by_name_never_a_default(tmp_path):
    cli = dict(_LAUNCH_CLI)
    del cli["ent_coef"]
    run = _run(tmp_path / "a", "launcher --arch production", cli)
    with pytest.raises(rs.RecipeRestartError, match="--ent-coef"):
        rs.inherit_on_restart(_restart_ns(run), str(run), _SAVED)
    no_cli = _run(tmp_path / "b", "launcher --arch production", None)
    with pytest.raises(rs.RecipeRestartError, match="--n-envs"):
        rs.inherit_on_restart(_restart_ns(no_cli), str(no_cli), _SAVED)
    run_c = _run(tmp_path / "c", "launcher --arch production", _LAUNCH_CLI)
    with pytest.raises(rs.RecipeRestartError, match="--terminal-indicator"):
        rs.inherit_on_restart(_restart_ns(run_c), str(run_c), None)
    # …and a typed value closes the gap it names
    ns = _restart_ns(run, "--ent-coef", "0.05")
    assert rs.inherit_on_restart(ns, str(run), _SAVED)


def test_checkargs_reports_the_restart_refusal(tmp_path, capsys):
    from main.checkargs import main
    cli = dict(_LAUNCH_CLI)
    del cli["n_envs"]
    run = _run(tmp_path, "launcher --arch production", cli)
    (run / "model_config.json").write_text(json.dumps(baselines.production_config()))
    rc = main(["--argv", f"--steps 2 --model {run / 'checkpoints' / 'c.zip'} --run-dir {run}"])
    assert rc == 3 and "--n-envs" in capsys.readouterr().out


def test_no_restoration_on_a_fork_or_a_run_not_launched_with_the_umbrella(tmp_path):
    run = _run(tmp_path, "launcher --steps 9 --run-name x", _LAUNCH_CLI)
    ns = _restart_ns(run)
    assert rs.inherit_on_restart(ns, str(run), _SAVED) == [] and ns.ent_coef == 0.02
    run2 = _run(tmp_path / "b", "launcher --arch production", _LAUNCH_CLI)
    fork = build_parser().parse_args(["--steps", "1", "--model", str(run / "checkpoints" / "c.zip")])
    assert rs.inherit_on_restart(fork, str(run2), _SAVED) == []


def test_arch_production_is_still_refused_beside_model():
    from main.train.combination_checks import fresh_only_flags
    assert "--arch" in fresh_only_flags()


# ------------------------------------------------------------------------ the mirror and rows
def test_the_mirror_accessor_strips_the_block_and_the_validator_exempts_it():
    assert baselines.RECIPE_BLOCK_KEY not in baselines.production_config()
    assert set(baselines.production_recipe_block()) == {"fresh", "fork", "sizing"}
    b = baselines.get("production")
    raw = rs._raw_mirror()
    run_cfg = {k: v for k, v in raw.items() if k != baselines.RECIPE_BLOCK_KEY}
    for k, v in b.config_overrides.items():          # make the declared overrides DIFFER
        run_cfg[k] = "run-value" if not isinstance(v, bool) else (not v)
    assert baselines.compare_production(run_cfg, raw, b) == []
    # …and WITHOUT a declared migration, i.e. not by the migration's grace for key deltas:
    no_migration = dataclasses.replace(b, config_mirror_version=None)
    assert baselines.compare_production(run_cfg, raw, no_migration) == []


def test_the_live_mirror_validator_still_passes():
    findings = [f for f in baselines.validate(verify_sha=False)
                if f.name == "production" and f.level == "error"]
    assert not findings, findings


def test_every_row_names_a_real_parser_dest_and_flag():
    p = build_parser()
    by_dest: dict = {}
    for a in p._actions:
        by_dest.setdefault(a.dest, set()).update(a.option_strings)
    for r in rs.ALL_ROWS:
        assert r.flag in by_dest.get(r.dest, set()), r


def test_every_tri_state_unset_equals_what_resolve_config_fills():
    from agents.model.critic_mode import CRITIC_DEFAULT
    from agents.training.reward_weights import PBRS_GAMMA
    src = pathlib.Path(__file__).with_name("config.py").read_text()
    literals = {}
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_resolve"
                and len(node.args) == 2 and isinstance(node.args[0], ast.Constant)):
            try:
                literals[node.args[0].value] = ast.literal_eval(node.args[1])
            except ValueError:
                pass
    p = build_parser()
    for r in rs.ALL_ROWS:
        if r.unset == rs.PARSER_DEFAULT:
            assert p.get_default(r.dest) is not None, f"{r.dest}: tri-state needs a declared unset"
        elif r.dest == "gamma":
            # paired with the critic (critic_mode.critic_gamma): the default critic's is WINPROB_GAMMA,
            # a typed shaped one's PBRS_GAMMA
            from agents.model.critic_mode import WINPROB_GAMMA, critic_gamma
            assert r.unset == rs.CRITIC_PAIRED and critic_gamma(CRITIC_DEFAULT) == WINPROB_GAMMA
            assert critic_gamma("shaped") == PBRS_GAMMA
        elif r.dest in literals:
            assert r.unset == literals[r.dest], r.dest


def test_block_defects_are_refused_by_name():
    raw = rs._raw_mirror()
    stray = copy.deepcopy(raw)
    stray["recipe"]["fresh"]["mystery_knob"] = 1
    with pytest.raises(rs.RecipeError, match="mystery_knob"):
        rs.production_recipe(stray)
    dropped = copy.deepcopy(raw)
    del dropped["recipe"]["fresh"]["ent_coef"]
    with pytest.raises(rs.RecipeError, match="ent_coef"):
        rs.production_recipe(dropped)
    clash = copy.deepcopy(raw)
    clash["recipe"]["fresh"]["vf_coef"] = 1.5
    with pytest.raises(rs.RecipeError, match="vf_coef"):
        rs.production_recipe(clash)
    fork = copy.deepcopy(raw)
    fork["recipe"]["fork"]["ent_coef"] = 0.01
    with pytest.raises(rs.RecipeError, match="recipe.fork"):
        rs.production_recipe(fork)


def test_the_arch_block_no_longer_lists_recipe_knobs_as_unapplied():
    from main.train.arch_surface import unapplied_production_keys
    flags = {f for f, _ in unapplied_production_keys()}
    assert not flags & {r.flag for r in rs.ROWS}


def test_sync_config_carries_the_recipe_block_over(tmp_path, monkeypatch):
    from agents.model import delivery_graph
    mirror = tmp_path / "production_config.json"
    mirror.write_text(json.dumps({"recipe": {"fresh": {"n_envs": 48}}, "total_dim": 1}))
    run_cfg = tmp_path / "model_config.json"
    run_cfg.write_text(json.dumps({"arch_signature": "s", "config_version": 1, "total_dim": 2}))
    monkeypatch.setattr(delivery_graph, "_DEFAULT_CONFIG", str(mirror))

    class _Stop(Exception):
        pass

    def _stop(*_a, **_k):
        raise _Stop

    monkeypatch.setattr(delivery_graph, "build_graph", _stop)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(_Stop):
        delivery_graph.main(["--sync-config", str(run_cfg)])
    out = json.loads(mirror.read_text())
    assert out["recipe"] == {"fresh": {"n_envs": 48}} and out["total_dim"] == 2


# ------------------------------------------------ end to end, on cutover-prep's restart mechanism
def test_a_launcher_restart_of_a_fresh_arch_production_run_keeps_the_whole_recipe(tmp_path, capsys):
    """fresh `--arch production` (NO recipe token typed) → save → the launcher's OWN restart argv
    (`resume_child_args`, which strips `--arch`) → `resolve_config`. Every non-INERT recipe row comes
    back as launched: the recorded ones through the resume's inheritance (`_resolve`, cutover-prep's
    v125 `opp_intent_coef` included), the rest through `inherit_on_restart`, each announced."""
    from main.launcher.checkpoint import resume_child_args
    from main.train.derived_toggle_resume_test import _resolve, _run_dir, _saved_version

    argv = ["--arch", "production", "--steps", "1000"]
    fresh = _resolve(argv)
    # `_saved_version` records only the same-named constructor kwargs; a real save records the
    # reward fields through `reward_config`, so the double gets the launch's values here.
    saved = dataclasses.replace(_saved_version(fresh), terminal_indicator=fresh.terminal_indicator,
                                victory_value=fresh.victory_value, draw_penalty=fresh.draw_penalty)
    run, ckpt = _run_dir(tmp_path, saved, cli_args=json.loads(json.dumps(vars(fresh), default=str)))
    meta = pathlib.Path(run) / "metadata.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()),
                                "original_command": "launcher " + " ".join(argv)}))
    restart_argv, stripped = resume_child_args(list(argv), ckpt, run)
    assert stripped and "--arch" not in restart_argv
    capsys.readouterr()
    restarted = _resolve(restart_argv)
    drift = {r.dest: (getattr(fresh, r.dest), getattr(restarted, r.dest)) for r in rs.ROWS
             if r.dest not in rs.INERT_ON_RESUME
             and not rs._agree(getattr(fresh, r.dest), getattr(restarted, r.dest))}
    assert not drift, f"a launcher restart changed what the recipe set: {drift}"
    out = capsys.readouterr().out
    assert "ent_coef=0.05 from metadata.json:cli_args" in out
    assert "terminal_indicator=True from model_config.json" in out
    # a recorded field is the resume's own inheritance (cutover-prep's v125 route), never ours:
    assert "opp_intent_coef=" not in out and "critic=" not in out
    assert restarted.opp_intent_coef == 0.05
