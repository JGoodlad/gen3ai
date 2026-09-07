"""Gates for `main.checkargs` — and for the parser being INSPECTABLE at all.

The load-bearing test here is `test_every_help_string_renders`. `--help` was broken on main by a
single unescaped `%` ("~0.6% of the static table" → argparse reads `% o` as a space-flag `%o`
conversion and raises `TypeError: %o format: an integer is required, not dict`). Nothing rendered
the help text, so nothing caught it — and with `--help` down there was no offline way to ask what
the parser accepts, which is exactly when you need one.
"""
from __future__ import annotations

import json

import pytest

from main.checkargs import (argv_from_run, check, known_option_strings,
                            split_argv, unsatisfiable_pairs)


# ------------------------------------------------------------------ the parser is inspectable

def test_every_help_string_renders():
    """THE regression. argparse formats help lazily, so a bad `%` only explodes when something
    actually renders it — which nothing did. Rendering the whole parser is the guard."""
    import argparse as _ap
    from main.train_rl_agent import build_parser
    parser = build_parser()
    fmt = _ap.HelpFormatter(prog="train_rl_agent.py")
    for action in parser._actions:
        if action.help:
            fmt._expand_help(action)          # raises TypeError/ValueError on a bad conversion
    assert parser.format_help()               # and the whole document composes


def test_build_parser_is_importable_without_running_main():
    """`build_parser()` must not need an event loop, a GPU, or argv — the whole point of pulling
    it out of `main()` is that a tool can ask the parser questions cheaply."""
    from main.train_rl_agent import build_parser
    assert len(build_parser()._actions) > 100


def test_known_option_strings_covers_a_live_flag():
    known = known_option_strings()
    assert "--steps" in known and known["--steps"] == "steps"
    assert "--pair-value-route" in known, "a v95 flag should be present on a v96 tree"


# ------------------------------------------------------------------ the checker itself

def test_clean_argv_reports_nothing():
    res = check(["--steps", "100", "--device", "cuda"])
    assert res["unknown"] == []
    assert set(res["accepted"]) == {"--steps", "--device"}


def test_deleted_flag_is_reported_with_its_value():
    """The motivating case: `--pubval-*` survived in gen-12's argv after v88 deleted it."""
    res = check(["--steps", "100", "--pubval-mode", "none", "--pubval-coef", "0.1"])
    assert [f for f, _ in res["unknown"]] == ["--pubval-mode", "--pubval-coef"]
    assert dict(res["unknown"])["--pubval-mode"] == ["none"]
def test_split_argv_attaches_values_to_their_flag():
    assert split_argv(["--a", "1", "2", "--b"]) == [("--a", ["1", "2"]), ("--b", [])]


def test_split_argv_tolerates_a_leading_positional():
    assert split_argv(["run_dir", "--a"])[0] == ("run_dir", [])


# ------------------------------------------------------------------ the run-dir front door

def test_argv_from_run_reads_the_recorded_command(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"launcher_command": "/path/to/__main__.py --steps 25000000 --device cuda"}))
    assert argv_from_run(str(tmp_path)) == ["--steps", "25000000", "--device", "cuda"]


def test_argv_from_run_falls_back_to_original_command(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"original_command": "/p/__main__.py --steps 1"}))
    assert argv_from_run(str(tmp_path)) == ["--steps", "1"]


def test_argv_from_run_fails_loud_when_nothing_is_recorded(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"git_hash": "abc"}))
    with pytest.raises(SystemExit):
        argv_from_run(str(tmp_path))


def test_missing_metadata_is_a_clean_message_not_a_traceback(tmp_path):
    """A run writes metadata.json at its first save, so a just-launched run legitimately has none.
    Asking about one is reasonable; a raw FileNotFoundError at the reader is not."""
    with pytest.raises(SystemExit) as e:
        argv_from_run(str(tmp_path))
    assert "no metadata.json yet" in str(e.value)


def test_missing_run_dir_says_so(tmp_path):
    with pytest.raises(SystemExit) as e:
        argv_from_run(str(tmp_path / "nope"))
    assert "no such run dir" in str(e.value)


# ------------------------------------------------- the flag_registry dependency graph, threaded in

def test_an_explicitly_negated_dependency_is_reported():
    """The whole point: this argv passes argparse and dies inside the extractor constructor."""
    pairs = unsatisfiable_pairs(
        ["--steps", "100", "--intent-conditional", "--damage-matrices", "off"])
    assert ("intent_conditional", "damage_matrices_outgoing", "--damage-matrices off") in pairs


def test_a_merely_ABSENT_dependency_is_not_reported():
    """A resume inherits every unspecified flag from the checkpoint's config, so absence is not
    evidence. Reporting it would make the tool cry wolf on the commands people actually rerun."""
    assert unsatisfiable_pairs(["--steps", "100", "--intent-conditional"]) == []


def test_a_satisfied_dependency_is_silent():
    argv = ["--intent-conditional", "--damage-op", "--damage-outgoing",
            "--damage-matrices", "both", "--opp-intent-coef", "0.5"]
    assert unsatisfiable_pairs(argv) == []


def test_a_zero_COEFFICIENT_disables_a_derived_toggle():
    """`opp_intent` is set by `--opp-intent-coef`, where 0 is OFF — the one place a numeric value,
    not a mode string, decides whether a dependency is satisfied.

    (This used `--value-intent` until the critic-route deletion wave deleted that flag;
    `--intent-threshold` carries the same single `opp_intent` dependency plus `damage_op`, so the
    assertion names both rather than assuming a one-element list.)"""
    pairs = unsatisfiable_pairs(["--intent-threshold", "--damage-op", "--opp-intent-coef", "0"])
    assert [(f, d) for f, d, _ in pairs] == [("intent_threshold", "opp_intent")]


def test_check_reports_both_failure_kinds_in_one_pass():
    res = check(["--pubval-mode", "none", "--intent-conditional", "--damage-matrices", "off"])
    assert [f for f, _ in res["unknown"]] == ["--pubval-mode"]
    assert res["unsatisfiable"], "the dependency half must not be masked by the unknown-flag half"


# ---------------------------------------- an ARGV IS NOT A CONFIG: resolving against the parent
#
# The third instance of one class: `checkargs` passes, the launch fails. C1 (2026-09-01) forked a
# parent whose `model_config.json` recorded `distill_target="action"`, passed `--distill-coef 0`,
# and never named a target — so `config._resolve` INHERITED `action` and `resolve_config` refused
# the pair, after this tool had printed "✓ this command still launches". The fixtures below are the
# minimum shape that reproduces it: a run dir with a recorded config and a checkpoint path under it.


def _minimal_model_config(**overrides) -> dict:
    """A `model_config.json` dict `ModelVersion.from_json_file` accepts.

    Required fields are filled by TYPE rather than by a hand-written list, so a new required field
    on `ModelVersion` does not silently rot this fixture into a skip.
    """
    import dataclasses
    from agents.model.model_version import ARCH_SIGNATURE, MODEL_CONFIG_VERSION, ModelVersion
    out: dict = {}
    for f in dataclasses.fields(ModelVersion):
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
            continue                                   # has a default — leave it to the dataclass
        ann = str(f.type)
        if "List" in ann or "list" in ann:
            out[f.name] = [64]
        elif "str" in ann:
            out[f.name] = ""
        elif "float" in ann:
            out[f.name] = 0.0
        else:
            out[f.name] = 8
    out["config_version"] = MODEL_CONFIG_VERSION
    out["arch_signature"] = ARCH_SIGNATURE
    out.update(overrides)
    return out


def _parent_run(tmp_path, name="parent", *, write_config=True, **recorded):
    """`(checkpoint_path, run_dir)` for a run whose recorded config carries `recorded`."""
    run = tmp_path / name
    (run / "checkpoints").mkdir(parents=True)
    if write_config:
        (run / "model_config.json").write_text(json.dumps(_minimal_model_config(**recorded)))
    (run / "metadata.json").write_text(json.dumps({"original_command": "x.py --steps 1"}))
    return str(run / "checkpoints" / "checkpoint_10_steps.zip"), str(run)


def _c1_argv(ckpt, *extra):
    """C1's shape: fork a parent, turn the distill term OFF, never name the target form."""
    return ["--model", ckpt, "--run-name", "child_run", "--steps", "1000",
            "--distill-coef", "0", *extra]


def test_an_inherited_distill_target_is_REPORTED(tmp_path):
    """(a) THE C1 DEFECT. Nothing in the argv is wrong; the effective config is."""
    ckpt, _ = _parent_run(tmp_path, distill_target="action")
    res = check(_c1_argv(ckpt))
    names = [c.name for c, _ in res["combinations"]]
    assert names == ["distill_target_needs_coef"], res["combinations"]
    combo, provenance = res["combinations"][0]
    assert "--distill-target action requires --distill-coef > 0" in combo.message
    assert any("--distill-target 'action'" in p and "INHERITED" in p for p in provenance), provenance
    assert res["resolution"]["inherited"]["distill_target"] == "action"


def test_the_same_argv_with_an_explicit_target_passes(tmp_path):
    """(b) The fix that actually launched C1: name the target the argv means."""
    ckpt, _ = _parent_run(tmp_path, distill_target="action")
    res = check(_c1_argv(ckpt, "--distill-target", "kl"))
    assert res["combinations"] == []
    assert "distill_target" not in res["resolution"]["inherited"]


def test_main_exits_1_on_the_inherited_refusal_and_0_once_named(tmp_path, capsys):
    """The exit code is the whole interface — a wrapper script reads that, not the prose."""
    from main.checkargs import main as checkargs_main
    ckpt, _ = _parent_run(tmp_path, distill_target="action")
    assert checkargs_main(["--argv", " ".join(_c1_argv(ckpt))]) == 1
    out = capsys.readouterr().out
    assert "WOULD FAIL IN resolve_config" in out and "FORK PARENT" in out
    assert checkargs_main(["--argv", " ".join(_c1_argv(ckpt, "--distill-target", "kl"))]) == 0


def test_a_same_run_restart_is_classified_as_a_restart_not_a_fork(tmp_path, capsys):
    """(c) `--model` INSIDE the dir the argv writes into is a RESTART, and is labelled one.

    It is still resolved against that checkpoint's recorded config, because `config._resolve` reads
    `_load_saved_version(args.model)` on EVERY `--model` — fork or restart. The fork/restart split
    (`fork_lr.is_same_run_checkpoint`, imported, never re-derived) decides what the report CALLS the
    source, not whether there is one; a launcher restart that inherits an incoherent value fails at
    launch exactly like a fork does, so suppressing the check there would re-open this hole.
    """
    from main.checkargs import main as checkargs_main, resolve_against_parent
    ckpt, run = _parent_run(tmp_path, distill_target="action")
    argv = ["--model", ckpt, "--run-dir", run, "--steps", "1000", "--distill-coef", "0"]
    assert resolve_against_parent(argv)["same_run"] is True
    checkargs_main(["--argv", " ".join(argv)])
    out = capsys.readouterr().out
    assert "same-run RESTART checkpoint" in out and "FORK PARENT" not in out


def test_a_fork_is_classified_as_a_fork(tmp_path):
    from main.checkargs import resolve_against_parent
    ckpt, _ = _parent_run(tmp_path, distill_target="action")
    assert resolve_against_parent(_c1_argv(ckpt))["same_run"] is False


def test_a_parent_with_no_recorded_config_WARNS_and_still_checks_the_argv(tmp_path, capsys):
    """(d) A missing parent config is never a silent pass: it names every path it tried, and the
    argv-only checks still run."""
    from main.checkargs import main as checkargs_main
    ckpt, _ = _parent_run(tmp_path, write_config=False)
    rc = checkargs_main(["--argv", " ".join(_c1_argv(ckpt))])
    out = capsys.readouterr().out
    assert "could not read the FORK PARENT's recorded config" in out
    assert "ARGV-ONLY" in out
    assert out.count("tried: ") == 2, out          # the checkpoint's dir AND the run root
    assert rc == 0                                 # nothing in the ARGV itself is wrong


def test_no_model_still_builds_a_namespace_and_runs_the_combination_checks():
    """A fresh run has nothing to INHERIT from — but the argv is still a config.

    Until 2026-09-06 the whole combination half was skipped whenever `--model` was absent, so a
    FRESH control arm carrying `--distill-coef 0` beside the fold instruments printed "✓ this
    command still launches" and then died three times (G5). The resolution is now reported with
    `no_parent`, and the DEPENDENCY half stays conservative exactly as before.
    """
    res = check(["--steps", "100", "--device", "cuda"])
    assert res["resolution"] is not None and res["resolution"]["no_parent"] is True
    assert res["resolution"]["inherited"] == {}
    assert res["combinations"] == []                       # a plain argv is still clean

    g5 = check(["--steps", "100", "--distill-coef", "0", "--distill-anchor-monitor"])
    assert "anchor_needs_live_distill" in [c.name for c, _ in g5["combinations"]], g5["combinations"]


def test_a_resolved_namespace_reports_an_ABSENT_dependency_the_argv_alone_cannot(tmp_path):
    """The conservative rule is a consequence of not knowing, not a preference. Once the parent is
    read, an unnamed dependency has a KNOWN value, so `--intent-conditional` over a parent that
    recorded `damage_op` OFF is reported — where the same argv with no `--model` is not.

    (`damage_matrices_outgoing` is deliberately NOT among the pairs: it has no argparse dest of its
    own — `--damage-matrices` desugars into it inside `resolve_config` — so on this namespace its
    value is genuinely undetermined, and an undetermined value is skipped rather than guessed.)"""
    ckpt, _ = _parent_run(tmp_path, damage_op=False)
    argv = ["--model", ckpt, "--run-name", "child_run", "--steps", "1000", "--intent-conditional"]
    res = check(argv)
    assert ("intent_conditional", "damage_op") in [(f, d) for f, d, _ in res["unsatisfiable"]]
    assert unsatisfiable_pairs(["--steps", "1000", "--intent-conditional"]) == []


# ------------------------------------------------------ the ONE declaration both surfaces read

def test_config_and_checkargs_share_the_combination_rules():
    """`resolve_config` prints these messages and `checkargs` reports them. If the launch path ever
    stops importing the shared list, the two can drift apart again — which is the whole defect."""
    from main.train import config as _config
    from main.train.combination_checks import COMBINATION_CHECKS, refuse_first
    assert _config.refuse_first is refuse_first
    assert "distill_target_needs_coef" in {c.name for c in COMBINATION_CHECKS}
    # The list is EXHAUSTIVE over its class now, not a sample of four — see
    # `main.train.combination_checks_test`, which AST-scans `config.py` to keep it that way.
    assert len(COMBINATION_CHECKS) > 40, len(COMBINATION_CHECKS)


def test_the_launch_path_still_refuses_the_C1_combination():
    """The check has to fire where it always did. A `SimpleNamespace` is enough — `failing_checks`
    is pure over an args-shaped object, which is why both surfaces can call it."""
    from types import SimpleNamespace
    from main.train.combination_checks import failing_checks
    broken = SimpleNamespace(distill_target="action", distill_coef=0.0, distill_topk=1,
                             distill_gate="none", distill_gate_tau=0.0)
    assert [c.name for c in failing_checks(broken)] == ["distill_target_needs_coef"]
    # `distill_teacher` is now load-bearing on the OK side: a live coefficient with no teacher is
    # itself one of the migrated refusals, so leaving it out would trip a different rule.
    ok = SimpleNamespace(distill_target="action", distill_coef=0.1, distill_topk=1,
                         distill_gate="none", distill_gate_tau=0.0,
                         distill_teacher="models/t:data/teams/sample/a.txt")
    assert failing_checks(ok) == []


def test_an_unresolved_value_is_never_a_verdict():
    """A `None` is "nothing determined this", not "off" — reporting it would make the tool cry wolf
    on every argv that has no parent to resolve against."""
    from types import SimpleNamespace
    from main.train.combination_checks import failing_checks
    unresolved = SimpleNamespace(distill_target=None, distill_coef=None, distill_topk=None,
                                 distill_gate=None, distill_gate_tau=None)
    assert failing_checks(unresolved) == []


# --- gen3_distill_instruments_default_v1 ------------------------------------------------------

def test_the_tri_state_monitor_flag_and_its_negation_both_still_validate():
    """`--distill-anchor-monitor` went from `store_true` to a tri-state `BoolFlag` so it can carry
    a "not typed" state and a `--no-` opt-out. Both spellings have to survive the offline check, or
    every recorded fold command that names one becomes un-validatable."""
    from main.checkargs import check
    base = ["--distill-teacher", "models/t:data/teams/sample/a.txt", "--distill-coef", "0.1"]
    for flag in ("--distill-anchor-monitor", "--no-distill-anchor-monitor"):
        got = check([*base, flag])
        assert got["unknown"] == [], got["unknown"]
        assert got["combinations"] == [] and got["unsatisfiable"] == []


def test_a_fold_argv_and_a_teacherless_argv_both_still_pass(tmp_path):
    """(f) The two ends of the new default: a fold command and an ordinary one. Neither the
    defaulted monitor nor the defaulted stop rule may make an argv that LAUNCHES read as one that
    would fail.

    ⚠️ The fold argv now carries `--model`, and that is a correction, not a convenience. The
    version of this test written for `gen3_distill_instruments_default_v1` used a `--distill-stop
    warn` fold with NO parent and asserted checkargs said nothing — but `resolve_config` REFUSES
    that command (`--distill-stop requires the anchor MONITOR`: with no `--model` and no
    `--distill-anchor-parent` the monitor cannot default on, so the rule's rise half could never
    fire). checkargs agreed only because it never ran the check. Verified against the pre-migration
    tree, 2026-09-06: exit 2, that message.
    """
    from main.checkargs import check
    ckpt, _ = _parent_run(tmp_path)
    fold = check(["--steps", "10", "--model", ckpt, "--run-name", "child_run",
                  "--distill-teacher", "models/t:data/teams/sample/a.txt",
                  "--distill-coef", "0.3", "--distill-stop", "warn"])
    assert fold["unknown"] == [] and fold["combinations"] == [], fold["combinations"]
    plain = check(["--steps", "10", "--device", "cuda"])
    assert plain["unknown"] == [] and plain["combinations"] == []


# --------------------------------------------------------- (g) models/ lives in the MAIN checkout
#
# THE WORKTREE DEFECT (2026-09-06). A recorded command names the archive RELATIVELY, `models/`
# exists only in the main checkout, and most agents run in a git worktree — so `--model
# models/<run>/checkpoints/x.zip` resolved to nothing, the parent's config could not be read, and
# the tool degraded to ARGV-ONLY. The INHERITED half — the half C1 exists to catch — was inert for
# exactly the readers most likely to need it. Every test below runs from a temp cwd with NO
# `models/`, which is what a worktree is.


def _archive_run(archive, name="parent", **recorded):
    """A synthetic run inside an ARCHIVE dir, addressed the way a recorded command addresses it."""
    run = archive / name
    (run / "checkpoints").mkdir(parents=True)
    (run / "model_config.json").write_text(json.dumps(_minimal_model_config(**recorded)))
    (run / "metadata.json").write_text(json.dumps(
        {"original_command": "x.py --steps 1", "launcher_command": "x.py --steps 7 --device cuda"}))
    return f"models/{name}/checkpoints/checkpoint_10_steps.zip"


def test_a_relative_model_path_resolves_into_the_archive_from_a_worktree(tmp_path, monkeypatch):
    """(g1) THE FIX. From a cwd with no models/, the parent config is READ and its inherited value
    reported — the C1 finding, on the relative path a real recorded command carries."""
    from main.checkargs import check
    archive, cwd = tmp_path / "archive", tmp_path / "worktree"
    archive.mkdir()
    cwd.mkdir()
    rel = _archive_run(archive, distill_target="action")
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(archive))
    monkeypatch.chdir(cwd)
    res = check(_c1_argv(rel))
    assert res["resolution"]["config_path"] == str(archive / "parent" / "model_config.json")
    assert res["resolution"]["inherited"]["distill_target"] == "action"
    assert [c.name for c, _ in res["combinations"]] == ["distill_target_needs_coef"]


def test_no_archive_still_WARNS_and_still_runs_the_argv_only_checks(tmp_path, monkeypatch, capsys):
    """(g2) `$GEN3AI_MODELS_DIR` set-and-missing ⇒ `main_models_dir()` is None ⇒ the path is left
    exactly as typed, the warning names every path tried, and the flag checks still run. A silent
    pass here would be worse than the defect."""
    from main.checkargs import main as checkargs_main, resolve_models_path
    cwd = tmp_path / "worktree"
    cwd.mkdir()
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(tmp_path / "nope"))
    monkeypatch.chdir(cwd)
    rel = "models/parent/checkpoints/checkpoint_10_steps.zip"
    assert resolve_models_path(rel) == rel
    argv = _c1_argv(rel, "--intent-conditional", "--damage-matrices", "off")
    rc = checkargs_main(["--argv", " ".join(argv)])
    out = capsys.readouterr().out
    assert "could not read the FORK PARENT's recorded config" in out and "ARGV-ONLY" in out
    assert out.count("tried: ") == 2, out
    # ...and the ARGV-ONLY half still ran on top of the warning, rather than passing in silence.
    assert "WOULD FAIL IN THE EXTRACTOR" in out and rc == 1


def test_an_absolute_path_is_never_rerouted(tmp_path, monkeypatch):
    """(g3) The archive is a FALLBACK, never an override — an absolute path (and a cwd-existing
    one) is returned untouched even when an archive holds a same-named run."""
    from main.checkargs import resolve_models_path
    archive = tmp_path / "archive"
    (archive / "parent").mkdir(parents=True)
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(archive))
    absolute = str(tmp_path / "elsewhere" / "parent" / "final_model.zip")
    assert resolve_models_path(absolute) == absolute
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "parent").mkdir(parents=True)
    assert resolve_models_path("models/parent") == "models/parent"
    assert resolve_models_path(None) is None


def test_the_run_dir_POSITIONAL_resolves_into_the_archive(tmp_path, monkeypatch):
    """(g4) `checkargs models/<run>` — the form the docs give — from a worktree cwd."""
    from main.checkargs import argv_from_run
    archive, cwd = tmp_path / "archive", tmp_path / "worktree"
    archive.mkdir()
    cwd.mkdir()
    _archive_run(archive, name="a_run")
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(archive))
    monkeypatch.chdir(cwd)
    assert argv_from_run("models/a_run") == ["--steps", "7", "--device", "cuda"]


def test_a_relative_restart_is_still_classified_as_a_RESTART(tmp_path, monkeypatch):
    """(g5) BOTH sides go through the resolver: a resolved checkpoint beside an UNresolved run dir
    would sit "outside" it, and every launcher restart would be mislabelled a FORK."""
    from main.checkargs import resolve_against_parent
    archive, cwd = tmp_path / "archive", tmp_path / "worktree"
    archive.mkdir()
    cwd.mkdir()
    rel = _archive_run(archive, name="live_run")
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(archive))
    monkeypatch.chdir(cwd)
    argv = ["--model", rel, "--run-dir", "models/live_run", "--steps", "10"]
    assert resolve_against_parent(argv)["same_run"] is True
    fork = ["--model", rel, "--run-name", "child_run", "--steps", "10"]
    assert resolve_against_parent(fork)["same_run"] is False


def test_the_pin_auto_derivation_reads_the_archives_checkpoint(tmp_path, monkeypatch):
    """(g6) The `--pin` default is the checkpoint's recorded `git_hash`; unresolved, the read fails
    and the whole argv silently falls back to the CURRENT tree's parser — the wrong authority."""
    from main.checkargs import resolve_pin_for
    archive, cwd = tmp_path / "archive", tmp_path / "worktree"
    archive.mkdir()
    cwd.mkdir()
    rel = _archive_run(archive, name="pinned_run")
    (archive / "pinned_run" / "checkpoints" / "checkpoint_10_steps.json").write_text(
        json.dumps({"git_hash": "b13b30b2" + "0" * 32}))
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(archive))
    monkeypatch.chdir(cwd)
    sha, why = resolve_pin_for(["--model", rel], None)
    assert sha == "b13b30b2" + "0" * 32, (sha, why)


def test_a_repeated_flag_reads_the_LAST_value_the_way_argparse_does():
    """(g7) A recorded `launcher_command` carries `--model` TWICE — the operator's fork parent and
    the run's own checkpoint, appended by the launcher on every restart. argparse takes the last,
    so anything deriving a pin from the FIRST pins to the parent's commit and reports HEAD-only
    flags as absent on a command that trained to completion (a false POSITIVE)."""
    import argparse

    from main.checkargs import argv_value, model_arg
    argv = ["--model", "models/parent/final_model.zip", "--steps", "10",
            "--model", "models/child/final_model.zip"]
    assert model_arg(argv) == "models/child/final_model.zip"

    # ...and "the way argparse does" is asserted against argparse, not against a belief about it.
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--steps")
    assert model_arg(argv) == p.parse_args(argv).model

    assert argv_value(["--run-name", "a", "--run-name=b"], "--run-name") == "b"
    assert argv_value(["--pin-commit", "deadbeef"], "--pin-commit") == "deadbeef"
    assert argv_value(["--steps", "1"], "--pin-commit") is None


# --------------------------------------------------- (g8) a FRESH argv carries a pin too
#
# 2026-09-06. `resolve_pin_for` derived the pin only when a `--model` was present, so an argv
# with `--pin-commit <sha>` and NO checkpoint — every arm of a batch that starts a NEW run on one
# commit — printed "no --pin — this argv would run on HEAD" and was judged by the parser of a tree
# the child will never run. Observed on the first win-prob arm's launch (`--pin-commit e798c13a`,
# no `--model`): harmless only because HEAD happened to BE the pin that afternoon.

#: A REAL commit in this repository, one before `--critic` existed. Real on purpose: the whole
#: mechanism is `git archive` + that commit's own `build_parser()`, so a fabricated sha would
#: exercise nothing. `--critic` is likewise a real current-only flag — `flags_only_in_current_tree`
#: asks the LIVE parser what it knows, and a made-up spelling would land in the ordinary
#: unrecognized bucket instead.
_COMMIT_BEFORE_CRITIC = "08dac300"
_CURRENT_ONLY_FLAG = "--critic"

#: A `--critic winprob` argv the CURRENT tree accepts outright — every companion the mode's own
#: combination checks require, so the only thing separating the two arms below is the pin.
# `--allow-nonproduction-arch` is carried deliberately: these two tests are about WHICH PARSER
# judges an argv, and without it the ARCH-SURFACE guard (gen3_arch_surface_guard_v1) refuses the
# command for a different, correct reason — a bare fresh argv is not the production architecture —
# which would make the pinned/unpinned arms differ in two things instead of one. That the guard
# fires on this exact shape is `main/train/arch_surface_test.py`'s subject.
_WINPROB_ARGV = ("--steps 1000 --critic winprob --no-hand-shaping --terminal-indicator "
                 "--victory-value 1.0 --draw-penalty 0 --device cuda "
                 "--allow-nonproduction-arch")


def _require_commit(sha: str) -> None:
    import subprocess

    from utils.paths import repo_root
    ok = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                        cwd=str(repo_root()), capture_output=True)
    if ok.returncode != 0:
        pytest.skip(f"{sha} is not in this checkout (shallow clone?)")


def test_a_fresh_argvs_pin_commit_is_honoured_and_the_rule_is_named():
    """(g8) No `--model` at all — the pin comes from the argv's own `--pin-commit`, resolved by
    the LAUNCHER's `resolve_pin` (called, not re-derived) exactly as its fresh-run path does."""
    from main.checkargs import resolve_pin_for
    _require_commit(_COMMIT_BEFORE_CRITIC)
    sha, why = resolve_pin_for(["--pin-commit", _COMMIT_BEFORE_CRITIC, "--steps", "1"], None)
    assert sha is not None and sha.startswith(_COMMIT_BEFORE_CRITIC), (sha, why)
    assert "--pin-commit" in why, f"the report must say WHICH rule chose the commit: {why}"


def test_a_fresh_argv_with_no_pin_still_reads_as_HEAD():
    """…and the fix must not widen: with neither a `--model` nor a `--pin-commit` the argv really
    does run on HEAD, and the note says so."""
    from main.checkargs import resolve_pin_for
    sha, why = resolve_pin_for(["--steps", "1", "--device", "cuda"], None)
    assert sha is None and "HEAD" in why, (sha, why)


def test_the_legacy_pin_to_hash_spelling_is_read_too_and_the_LAST_one_wins():
    """One `dest` behind two spellings, so the value that RUNS is the last occurrence of EITHER —
    the same rule `argv_value` documents, which is why it is applied across the alias pair rather
    than to one spelling with the other silently invisible."""
    from main.checkargs import pin_commit_arg
    assert pin_commit_arg(["--pin-to-hash", "deadbeef"]) == "deadbeef"
    assert pin_commit_arg(["--pin-commit", "aaa", "--pin-to-hash", "bbb"]) == "bbb"
    assert pin_commit_arg(["--pin-to-hash", "aaa", "--pin-commit=bbb"]) == "bbb"
    assert pin_commit_arg(["--steps", "1"]) is None


@pytest.mark.integration
def test_a_fresh_pinned_argv_is_judged_by_THAT_commits_parser(capsys):
    """(g8) END TO END, on the real repository: a flag that exists ONLY in the current tree is
    refused at a commit that predates it, and the finding names the pin. This is the failure the
    defect hid — the child runs the pinned tree, and would die at startup with a run dir already
    on disk."""
    from main.checkargs import main as checkargs_main
    from main.exit_codes import TrainExitCode
    _require_commit(_COMMIT_BEFORE_CRITIC)
    rc = checkargs_main(
        ["--argv", f"--pin-commit {_COMMIT_BEFORE_CRITIC} {_WINPROB_ARGV}"])
    out = capsys.readouterr().out
    assert rc == int(TrainExitCode.FATAL_CONFIG), out
    assert f"PINNED commit {_COMMIT_BEFORE_CRITIC}" in out, out
    assert "--pin-commit" in out.split("\n")[1], "the parser line must name the RULE"
    assert "NOT IN PINNED TREE" in out and _CURRENT_ONLY_FLAG in out, out


@pytest.mark.integration
def test_the_same_argv_WITHOUT_the_pin_passes(capsys):
    """The paired control. Drop the `--pin-commit` and the identical flags are judged by THIS
    tree, where they all exist — so the arms differ in the pin and nothing else."""
    from main.checkargs import main as checkargs_main
    rc = checkargs_main(["--argv", _WINPROB_ARGV])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "this argv would run on HEAD" in out and "✓ this command still launches" in out


def test_pin_commit_beside_sync_to_main_is_REPORTED_as_the_launcher_would_refuse_it(capsys):
    """The launcher's OWN parser kills this pair at parse time (a mutually-exclusive group), and
    the trainer parser this module otherwise reads cannot see either flag — so without this the
    argv passed every check and still never launched. checkargs REPORTS; it never raises."""
    from main.checkargs import launcher_refusals, main as checkargs_main
    assert launcher_refusals(["--pin-commit", "abc1234", "--sync-to-main"])
    assert launcher_refusals(["--pin-commit", "abc1234", "--no-pin"])
    assert launcher_refusals(["--sync-to-main", "--steps", "1"]) == [], \
        "--sync-to-main ALONE is the ordinary fork spelling, not a refusal"

    rc = checkargs_main(["--argv", "--pin-commit abc1234 --sync-to-main --steps 1000"])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "refused by the LAUNCHER parser" in out and "--sync-to-main" in out
    assert "✓ this command still launches" not in out
