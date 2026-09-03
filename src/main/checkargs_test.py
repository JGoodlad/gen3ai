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


def test_no_model_means_no_resolution_at_all():
    """A fresh run has nothing to inherit from — the argv IS the config, and the old conservative
    argv-only behaviour is what runs."""
    res = check(["--steps", "100", "--device", "cuda"])
    assert res["resolution"] is None and res["combinations"] == []


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
    from main.train.combination_checks import COMBINATION_CHECKS, failing_checks
    assert _config.failing_checks is failing_checks
    assert "distill_target_needs_coef" in {c.name for c in COMBINATION_CHECKS}


def test_the_launch_path_still_refuses_the_C1_combination():
    """The check has to fire where it always did. A `SimpleNamespace` is enough — `failing_checks`
    is pure over an args-shaped object, which is why both surfaces can call it."""
    from types import SimpleNamespace
    from main.train.combination_checks import failing_checks
    broken = SimpleNamespace(distill_target="action", distill_coef=0.0, distill_topk=1,
                             distill_gate="none", distill_gate_tau=0.0)
    assert [c.name for c in failing_checks(broken)] == ["distill_target_needs_coef"]
    ok = SimpleNamespace(distill_target="action", distill_coef=0.1, distill_topk=1,
                         distill_gate="none", distill_gate_tau=0.0)
    assert failing_checks(ok) == []


def test_an_unresolved_value_is_never_a_verdict():
    """A `None` is "nothing determined this", not "off" — reporting it would make the tool cry wolf
    on every argv that has no parent to resolve against."""
    from types import SimpleNamespace
    from main.train.combination_checks import failing_checks
    unresolved = SimpleNamespace(distill_target=None, distill_coef=None, distill_topk=None,
                                 distill_gate=None, distill_gate_tau=None)
    assert failing_checks(unresolved) == []
