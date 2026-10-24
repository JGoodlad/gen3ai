"""EVERY CLI FLAG IS CLASSED FOR A `--model` LAUNCH — `main.train.launch_flags`, and the resolver behind it.

THE DEFECT (Training session, 2026-10-24, production X35). A `--model <checkpoint>` dry run of `rb_x35_d1wide_s3505`
printed "K9(b) check : fatal (the default: --behaviour-check is neither typed nor inherited)" although the arm's
`metadata.json:cli_args` holds `behaviour_check=warn`, and planned saves every 2.4M instead of the run's 1M:
`inherit_recorded_recipe` restored the recipe rows only, so a manual resume or a fork silently changed the run's
safety mode and its save cadence. Root CLAUDE.md promises the opposite ("with `--model`, every flag you do not type
is inherited from the run").

(a) is the CLASS GATE: every trainer and launcher flag is in exactly one of the declared classes, so a new flag
cannot fall through to a silent parser default. (b)-(e) pin the behaviour and each FAILS on a revert of the piece it
names: the restart and fork routes, the typed override, the launcher's typed copies, and `resolve_config` end to end.
"""
from __future__ import annotations

import ast
import dataclasses
import json
import pathlib

import pytest

from main.train import launch_flags as LF
from main.train import recipe_surface as rs
from main.train.parser import build_parser
from main.train.recipe_surface_test import (_LAUNCH_CLI, _PARENT_CLI, _SAVED, _fork_ns, _restart_ns, _run)

#: A run recorded the way the X35 arms were launched: K9(b) at `warn`, a 1M save cadence, its own seed.
_RUN_FLAGS = {"behaviour_check": "warn", "checkpoint_every_steps": 1_000_000, "seed": 3504, "keep_stalls": 7,
              "eval_games": 60, "self_play_temp": 0.8, "trainee_team": "teams/pin.txt", "eval_freq": None,
              "anneal_min_lr": 5e-6, "supply_starve_cycles": "self_play_pool=5"}


def _restored(ns, run_dir, saved=_SAVED):
    return {d: (v, s) for d, v, s in rs.inherit_recorded_recipe(ns, str(run_dir), saved)}


# ------------------------------------------------------------------------------ (a) the class gate
def _resolved_field_names() -> set:
    """Every flag `config.resolve_config` resolves by a plain `_resolve("<name>", default)` /
    `inherit_saved_flag(args, _saved_ver, "<name>", …)` line (an AST scan of the source, like
    `combination_checks_test`'s)."""
    import main.train.config as cfg
    tree = ast.parse(pathlib.Path(cfg.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func.id if isinstance(node.func, ast.Name) else None
        if fn == "_resolve" and node.args and isinstance(node.args[0], ast.Constant):
            names.add(node.args[0].value)
        elif fn == "inherit_saved_flag" and len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
            names.add(node.args[2].value)
    return names


def _parsers():
    from main.launcher.run import build_launcher_parser
    trainer = {a.dest: a for a in build_parser()._actions}
    launcher = {a.dest: a for a in build_launcher_parser()._actions}
    return trainer, launcher


def _classes(dest: str, action, fields, resolved) -> list:
    out = []
    if dest in {r.dest for r in rs.ROWS}:                  # (FORK_ROWS are a comparison, not a restored row)
        out.append("recipe row")
    elif dest in fields and action.default is None and (dest in resolved or dest in LF.FIELD_SPECIAL):
        out.append("recorded field")
    for name, members in (("RECORDED_ROWS", LF.RECORDED_DESTS), ("RESTART_ONLY_ROWS", LF.RESTART_ONLY_DESTS),
                          ("OWN_ROUTE", LF.OWN_ROUTE), ("PER_LAUNCH", LF.PER_LAUNCH)):
        if dest in members:
            out.append(name)
    return out


def test_every_flag_is_in_exactly_one_inheritance_class():
    """THE GATE. A flag in none of the classes would resume at its parser default with nothing saying so (the
    2026-10-24 class); a flag in two would have two answers. The fix for a failure is to class the new flag:
    `launch_flags.RECORDED_ROWS` (it defines the run: safety, cadence, retention, schedule), `RESTART_ONLY_ROWS`
    (its identity or matchup), a ModelVersion field with a `_resolve` line in `resolve_config`, or — with a reason —
    `PER_LAUNCH`."""
    from agents.model.model_version import ModelVersion
    fields = {f.name for f in dataclasses.fields(ModelVersion)}
    resolved = _resolved_field_names()
    assert len(resolved) > 60, "the AST scan of resolve_config found too few `_resolve` lines to trust"
    trainer, launcher = _parsers()
    assert set(trainer) & set(launcher) == {"help"}, "a trainer and a launcher flag share a dest: the table is keyed by dest"
    unclassed, doubled = [], []
    for dest, action in {**launcher, **trainer}.items():
        cls = _classes(dest, action, fields, resolved)
        if not cls:
            unclassed.append(f"{dest} ({'/'.join(action.option_strings)}, default {action.default!r})")
        elif len(cls) > 1:
            doubled.append(f"{dest}: {cls}")
    assert not unclassed, f"flags in NO inheritance class (see this test's docstring): {unclassed}"
    assert not doubled, f"flags in MORE THAN ONE class: {doubled}"


def test_the_tables_name_only_real_flags_with_real_reasons():
    trainer, launcher = _parsers()
    every = {**launcher, **trainer}
    from agents.model.model_version import ModelVersion
    fields = {f.name for f in dataclasses.fields(ModelVersion)}
    p = build_parser()
    for table in (LF.PER_LAUNCH, LF.OWN_ROUTE, LF.FIELD_SPECIAL):
        assert not set(table) - set(every), f"stale entries (no such flag): {sorted(set(table) - set(every))}"
    for r in LF.INHERITED_ROWS:
        assert r.dest in trainer, f"{r.dest}: no such trainer flag"
        assert r.flag in trainer[r.dest].option_strings, (r.dest, r.flag, trainer[r.dest].option_strings)
        # the route is `cli_args`: a recorded ModelVersion field belongs to `_resolve`, never to a row here
        assert rs.restart_route(r.dest, p.get_default(r.dest), frozenset(fields)) == "cli_args", r.dest
        assert r.note.strip()
    assert not LF.RECORDED_DESTS & LF.RESTART_ONLY_DESTS
    groups = {"identity", "process", "fresh-only", "consent", "fork-act", "host", "launcher-session"}
    for dest, pl in LF.PER_LAUNCH.items():
        assert pl.group in groups and len(pl.reason) >= 20, (dest, pl)
    # the two flags of the 2026-10-24 defect, and the seed, are carried
    assert {"behaviour_check", "checkpoint_every_steps"} <= LF.RECORDED_DESTS and "seed" in LF.RESTART_ONLY_DESTS


def test_the_per_launch_table_is_what_the_launcher_really_does_not_carry():
    """A flag in `PER_LAUNCH` that a row restores would be restored AND declared not carried."""
    assert not set(LF.PER_LAUNCH) & LF.INHERITED_DESTS
    assert not set(LF.PER_LAUNCH) & {r.dest for r in rs.ROWS}
    from main.train.combination_checks import fresh_only_flags
    assert "--arch" in fresh_only_flags() and "arch" in LF.PER_LAUNCH     # the fresh-only claim is the trainer's own list


# ------------------------------------------------------ (b) a RESTART carries the run's own flags
def test_a_restart_carries_the_runs_recorded_mode_and_cadence_when_neither_is_typed(tmp_path):
    """THE DEFECT. A manual resume of a run recorded `warn` / 1M read the parser's None for both (K9(b) FATAL, saves
    every 2.4M). Fails on revert: both read None."""
    run = _run(tmp_path, "launcher --steps 9 --arch production --run-name x", {**_LAUNCH_CLI, **_RUN_FLAGS})
    ns = _restart_ns(run)
    assert (ns.behaviour_check, ns.checkpoint_every_steps) == (None, None)       # what the parser alone gives
    got = _restored(ns, run)
    assert (ns.behaviour_check, ns.checkpoint_every_steps) == ("warn", 1_000_000)
    assert got["behaviour_check"] == ("warn", "metadata.json:cli_args")
    assert got["checkpoint_every_steps"] == (1_000_000, "metadata.json:cli_args")
    # the identity and the rest of the safety / cadence / schedule set ride along on a restart
    assert (ns.seed, ns.keep_stalls, ns.eval_games, ns.self_play_temp) == (3504, 7, 60, 0.8)
    assert ns.trainee_team == "teams/pin.txt" and ns.anneal_min_lr == 5e-6
    assert ns.supply_starve_cycles == "self_play_pool=5"
    # a recorded None is "untyped": no row, no announcement, the trainer's own default stands
    assert ns.eval_freq is None and "eval_freq" not in got
    assert rs.value_source(ns, "behaviour_check") == ("inherited", "metadata.json:cli_args")


def test_a_typed_flag_still_wins_even_when_it_equals_the_parser_default(tmp_path):
    run = _run(tmp_path, "launcher --steps 9", {**_LAUNCH_CLI, **_RUN_FLAGS})
    ns = _restart_ns(run, "--behaviour-check", "fatal", "--checkpoint-every-steps", "2400000", "--seed", "42")
    got = _restored(ns, run)
    assert (ns.behaviour_check, ns.checkpoint_every_steps, ns.seed) == ("fatal", 2_400_000, 42)
    assert not {"behaviour_check", "checkpoint_every_steps", "seed"} & set(got)
    assert rs.value_source(ns, "behaviour_check") == ("typed", "")      # `--seed 42` equals the parser's, and is still typed
    assert rs.value_source(ns, "seed") == ("typed", "")
    assert rs.value_source(ns, "keep_stalls")[0] == "inherited"          # the untyped neighbours are still restored


def test_a_run_that_predates_a_flag_leaves_it_untyped_and_is_not_refused(tmp_path):
    """A recipe row a record lacks REFUSES; a run flag it lacks means the run predates the flag (K9(b), the value
    sidecar, …): the flag stays untyped and resolves to its own default. Never an error, never a guess."""
    run = _run(tmp_path, "launcher --steps 9", dict(_LAUNCH_CLI))                  # no run flag recorded at all
    ns = _restart_ns(run)
    got = _restored(ns, run)
    assert not set(got) & LF.INHERITED_DESTS
    assert (ns.behaviour_check, ns.checkpoint_every_steps, ns.seed) == (None, None, 42)
    assert rs.value_source(ns, "behaviour_check") == ("default", "")


# ----------------------------------------------- (c) a FORK carries its parent's flags, not its identity
def test_a_fork_carries_the_recorded_flags_but_not_the_seed_or_the_matchup(tmp_path):
    parent = _run(tmp_path, "launcher --steps 9 --run-name parent", {**_PARENT_CLI, **_RUN_FLAGS})
    ns = _fork_ns(parent)
    got = _restored(ns, tmp_path / "the_fork")
    assert (ns.behaviour_check, ns.checkpoint_every_steps, ns.keep_stalls) == ("warn", 1_000_000, 7)
    assert got["behaviour_check"] == ("warn", "parent metadata.json:cli_args")
    # a fork is a new experiment: its seed and matchup are its own — the parser's, until it types them
    assert (ns.seed, ns.trainee_team) == (42, None)
    assert not {"seed", "trainee_team"} & set(got)
    assert dict(ns._recipe_record_not_carried) == {"seed": 3504, "trainee_team": "teams/pin.txt"}
    lines = "\n".join(rs.report_lines(rs.report(ns, fresh=False)))
    assert "NOT carried" in lines and "seed=3504" in lines and "run flags" in lines
    # …and typing them is the fork's choice
    typed = _fork_ns(parent, "--seed", "3505", "--behaviour-check", "fatal")
    _restored(typed, tmp_path / "the_fork")
    assert (typed.seed, typed.behaviour_check) == (3505, "fatal")


# ------------------------------------------- (d) the launcher passes them to the child TYPED
def test_the_launcher_passes_the_restored_run_flags_typed_so_a_PRE_FIX_child_runs_them(tmp_path):
    run = _run(tmp_path, "launcher --steps 9", {**_LAUNCH_CLI, **_RUN_FLAGS})
    argv = ["--steps", "1", "--model", str(run / "checkpoints" / "c.zip"), "--run-dir", str(run)]
    out, added = rs.materialize_recorded_recipe(argv, str(run))
    child = build_parser().parse_args(out)                        # the pre-fix child: argv only, no inheritance
    assert (child.behaviour_check, child.checkpoint_every_steps, child.seed) == ("warn", 1_000_000, 3504)
    assert child.eval_freq is None and "eval_freq" not in {d for d, _v, _s in added}
    assert {"behaviour_check", "checkpoint_every_steps"} <= rs.typed_dests(child)
    # a TYPED flag is never overwritten, and a FORK's argv gets no restart-only token
    out2, _ = rs.materialize_recorded_recipe(argv + ["--behaviour-check", "off"], str(run))
    assert build_parser().parse_args(out2).behaviour_check == "off"
    parent_cli = {**_PARENT_CLI, **_RUN_FLAGS}
    parent = _run(tmp_path / "p", "launcher --steps 9", parent_cli)
    fork_out, _ = rs.materialize_recorded_recipe(
        ["--steps", "1", "--model", str(parent / "checkpoints" / "c.zip"), "--fork-lr", "3.6e-4"], str(tmp_path / "f"))
    fork_child = build_parser().parse_args(fork_out)
    assert (fork_child.behaviour_check, fork_child.seed, fork_child.trainee_team) == ("warn", 42, None)


def test_a_restored_bool_is_written_bare_so_a_pin_with_another_arity_still_parses_it(tmp_path):
    """THE REAL DRY RUN (rb_x35_d1wide_s3505, pin 440b61c3): `--team-wr-tracking true` / `--stable-opponent-pfsp false`
    were REFUSED by the pinned parser (`BooleanOptionalAction` / `store_true` there: "unconsumed value 'true'"). A bool
    is written `--flag` / `--no-flag`, and not at all at the parser's own value. Fails on revert (the `true` / `false`
    token reaches the stand-in pin parser as a stray value)."""
    import argparse
    cli = {**_LAUNCH_CLI, "pool_spread": True, "team_wr_tracking": False, "stable_opponent_pfsp": False,
           "overlap_collect": False}
    run = _run(tmp_path, "launcher --steps 9", cli)
    argv = ["--steps", "1", "--model", str(run / "checkpoints" / "c.zip"), "--run-dir", str(run)]
    out, added = rs.materialize_recorded_recipe(argv, str(run))
    tail = out[len(argv):]
    assert "true" not in tail and "false" not in tail, tail
    assert "--pool-spread" in tail and "--no-team-wr-tracking" in tail and "--self-play" in tail
    assert "--no-overlap-collect" in tail                                    # a tri-state row recorded False
    assert "--stable-opponent-pfsp" not in tail and "--no-stable-opponent-pfsp" not in tail      # the parser's own False
    assert {"pool_spread", "team_wr_tracking", "overlap_collect"} <= {d for d, _v, _s in added}
    assert "stable_opponent_pfsp" not in {d for d, _v, _s in added}
    # …and a True is typed even when it is the parser's own value (a pin's default may differ): bare, no value
    run2 = _run(tmp_path / "t", "launcher --steps 9", {**_LAUNCH_CLI, "team_wr_tracking": True})
    argv2 = ["--steps", "1", "--model", str(run2 / "checkpoints" / "c.zip"), "--run-dir", str(run2)]
    again, _ = rs.materialize_recorded_recipe(argv2, str(run2))
    assert "--team-wr-tracking" in again[len(argv2):] and "true" not in again[len(argv2):]
    child = build_parser().parse_args(out)
    assert (child.pool_spread, child.team_wr_tracking, child.self_play, child.stable_opponent_pfsp) == (
        True, False, True, False)
    # the stand-in for the pin: the spellings its parser had (store_true / BooleanOptionalAction), no value taken
    pin = argparse.ArgumentParser(allow_abbrev=False)
    pin.add_argument("--pool-spread", action="store_true")
    pin.add_argument("--team-wr-tracking", action=argparse.BooleanOptionalAction, default=True)
    pin.add_argument("--stable-opponent-pfsp", action="store_true")
    pin.add_argument("--self-play", action="store_true")
    known, rest = pin.parse_known_args([t for t in tail if t.startswith("--") and t in (
        "--pool-spread", "--no-team-wr-tracking", "--self-play", "--stable-opponent-pfsp")])
    assert rest == [] and (known.pool_spread, known.team_wr_tracking) == (True, False)


def test_a_launcher_restart_keeps_the_typed_flags_of_a_kit_launch():
    """The crash-restart path. A launcher RESTART re-launches the session's own child argv (`resume_child_args`
    strips only the trainer's fresh-only flags), so an arm launched with `--behaviour-check warn
    --checkpoint-every-steps 1000000` TYPED keeps both across every interval, crash and forced restart: the live X35
    arms are not subject to the 2026-10-24 defect, which is a MANUAL `--model` launch that omits them."""
    from main.launcher.checkpoint import resume_child_args
    kit = ["--model", "models/p/checkpoints/c.zip", "--run-name", "rb_x35_d1wide_s3505", "--fork-lr", "5.6e-05",
           "--behaviour-check", "warn", "--checkpoint-every-steps", "1000000", "--seed", "3505", "--steps", "9"]
    argv, stripped = resume_child_args(kit, "models/rb_x35_d1wide_s3505/checkpoints/c2.zip", "models/rb_x35_d1wide_s3505")
    assert stripped == []
    for flag, value in (("--behaviour-check", "warn"), ("--checkpoint-every-steps", "1000000"), ("--seed", "3505")):
        assert argv[argv.index(flag) + 1] == value


# ----------------------------------------- (e) the trainer's own resolution, end to end
def _fresh_and_run(tmp_path):
    from main.train.derived_toggle_resume_test import _resolve, _run_dir, _saved_version
    fresh = _resolve(["--arch", "production", "--steps", "1000", "--behaviour-check", "warn",
                      "--checkpoint-every-steps", "1000000", "--seed", "3504"])
    assert (fresh.behaviour_check, fresh.checkpoint_every_steps, fresh.seed) == ("warn", 1_000_000, 3504)
    run, ckpt = _run_dir(tmp_path, _saved_version(fresh), cli_args=json.loads(json.dumps(vars(fresh), default=str)))
    return _resolve, run, ckpt


def test_resolve_config_resumes_and_forks_with_the_recorded_mode_and_cadence_end_to_end(tmp_path):
    """`resolve_config` on a REAL recorded run: untyped, a restart reads warn / 1M / its seed and a fork warn / 1M;
    typed, the argv wins. Fails on revert (a restart read FATAL / None / 42 — `resolve_env_core_args` fills the
    default once nothing restored the record)."""
    _resolve, run, ckpt = _fresh_and_run(tmp_path)
    restart = _resolve(["--model", ckpt, "--run-dir", run, "--steps", "2000"])
    assert (restart.behaviour_check, restart.checkpoint_every_steps, restart.seed) == ("warn", 1_000_000, 3504)
    fork = _resolve(["--model", ckpt, "--fork-lr", "3.6e-4", "--steps", "2000"])
    assert (fork.behaviour_check, fork.checkpoint_every_steps, fork.seed) == ("warn", 1_000_000, 42)
    typed = _resolve(["--model", ckpt, "--run-dir", run, "--steps", "2000", "--behaviour-check", "fatal",
                      "--checkpoint-every-steps", "2400000"])
    assert (typed.behaviour_check, typed.checkpoint_every_steps) == ("fatal", 2_400_000)


def test_checkargs_reports_the_restored_mode_and_its_source(tmp_path, capsys):
    from main.checkargs import main as checkargs_main
    _resolve, run, ckpt = _fresh_and_run(tmp_path)
    checkargs_main(["--argv", f"--model {ckpt} --run-dir {run} --steps 2000"])
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.lstrip().startswith("K9(b) behaviour check"))
    assert line.split(":", 1)[1].split()[0] == "warn" and "inherited from metadata.json:cli_args" in line, line
    assert "NOT fatal" in line
    checkargs_main(["--argv", f"--model {ckpt} --run-dir {run} --steps 2000 --behaviour-check fatal"])
    typed = next(ln for ln in capsys.readouterr().out.splitlines() if ln.lstrip().startswith("K9(b) behaviour check"))
    assert typed.split(":", 1)[1].split()[0] == "fatal" and "(typed)" in typed, typed


@pytest.mark.parametrize("role", ["restart", "fork"])
def test_inherits_on_agrees_with_the_resolver(tmp_path, role):
    """`launch_flags.inherits_on` is what the docs render; it must say what `inherit_recorded_recipe` does."""
    cli = {**_PARENT_CLI, **_RUN_FLAGS}
    run = _run(tmp_path, "launcher --steps 9", cli)
    ns = _restart_ns(run) if role == "restart" else _fork_ns(run)
    got = set(_restored(ns, run if role == "restart" else tmp_path / "f"))
    for dest in _RUN_FLAGS:
        if _RUN_FLAGS[dest] is None:
            continue
        assert (dest in got) == LF.inherits_on(dest, role), (role, dest)
