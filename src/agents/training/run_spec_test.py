"""THE `path[@step]` run-spec splitter, its two throwing guards, and the call-site CENSUS.

`gen3_run_spec_split_v1` (2026-09-05). The defect this file pins: a `--distill-teacher` of the
form `<run_dir>@<step>:*` parsed, and the wildcard resolver was handed the path WITH the suffix
still attached — so `read_recorded_trainee_teams` looked for a `metadata.json` beside a directory
that does not exist, found none, and answered `[]`, which is the SAME answer a real generalist run
gives. A fold written the obvious way therefore reported teachers that teach nothing.

Four layers are tested here, because a one-place fix is only a class fix if nothing routes around
it: the splitter, the reader's raise, the launch-time refusal, and a CENSUS that fails (naming the
file) when a module re-derives the `@` split locally.
"""
from __future__ import annotations

import ast
import json
import os
import tempfile

import pytest

from agents.training.distill_spec import check_teacher_spec, parse_distill_teacher_spec
from agents.training.matchup_spec import read_recorded_trainee_teams
from agents.training.run_spec import run_dir_of, split_run_spec
from utils.paths import main_models_dir, src_path


# ---------------------------------------------------------------------------
# a synthetic run dir — the fixture that ALWAYS runs (models/ is not committed)
# ---------------------------------------------------------------------------

def _write_specialist_run(tmp: str, *, name: str = "spec_run",
                          n_teams: int = 2) -> "tuple[str, list[str]]":
    """A run dir whose metadata records `--trainee-teams` — i.e. what a teacher looks like."""
    run = os.path.join(tmp, name)
    os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
    files = []
    for i in range(n_teams):
        f = os.path.join(tmp, f"{name}_team{i}.txt")
        with open(f, "w") as fh:
            fh.write(f"Skarmory @ Leftovers\nability: Keen Eye\n- Spikes  # {i}\n")
        files.append(f)
    with open(os.path.join(run, "metadata.json"), "w") as fh:
        json.dump({"cli_args": {"trainee_teams": ",".join(files)}}, fh)
    return run, files


# ---------------------------------------------------------------------------
# the splitter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec,expect", [
    ("models/r", ("models/r", None)),
    ("models/r@123", ("models/r", 123)),
    ("models/r@0", ("models/r", 0)),
    ("/abs/models/r@26267760", ("/abs/models/r", 26267760)),
    ("models/r/checkpoints/c_5_steps.zip", ("models/r/checkpoints/c_5_steps.zip", None)),
    ("models/r@", ("models/r", None)),          # a bare trailing @ carries no step
    ("  models/r@7  ", ("models/r", 7)),
])
def test_split_run_spec(spec, expect):
    assert split_run_spec(spec) == expect
    assert run_dir_of(spec) == expect[0]


@pytest.mark.parametrize("bad", ["models/r@best", "models/r@1.5", "models/r@last@2"])
def test_a_non_integer_step_is_refused_loudly(bad):
    with pytest.raises(ValueError, match="is not an integer"):
        split_run_spec(bad)


def test_an_empty_path_is_refused():
    with pytest.raises(ValueError, match="has no path"):
        split_run_spec("@123")


# ---------------------------------------------------------------------------
# THE REPRODUCTION — the reader used to answer [] for an `@step` spec
# ---------------------------------------------------------------------------

def test_the_reader_raises_on_an_at_step_spec_instead_of_answering_empty():
    """THE DEFECT, on a synthetic run: `<run>@<step>` must never read as 'no recorded teams'."""
    with tempfile.TemporaryDirectory() as tmp:
        run, files = _write_specialist_run(tmp)
        assert read_recorded_trainee_teams(run) == files          # the run dir: 2 teams
        with pytest.raises(FileNotFoundError, match="does not exist"):
            read_recorded_trainee_teams(run + "@26267760")        # was: []


def test_the_reader_raises_on_any_missing_path():
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_recorded_trainee_teams("/nonexistent/run/dir")


def test_a_generalist_run_still_reads_as_no_teams():
    """The fold-back contract's legitimate empty: a run that exists and pinned nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        run = os.path.join(tmp, "generalist")
        os.makedirs(run)
        with open(os.path.join(run, "metadata.json"), "w") as fh:
            json.dump({"cli_args": {}}, fh)
        assert read_recorded_trainee_teams(run) == []
        with pytest.raises(ValueError, match="recorded NO trainee teams"):
            read_recorded_trainee_teams(run, require_teams=True)


def test_a_run_dir_with_no_metadata_is_empty_but_require_teams_says_so():
    with tempfile.TemporaryDirectory() as tmp:
        run = os.path.join(tmp, "bare")
        os.makedirs(run)
        assert read_recorded_trainee_teams(run) == []
        with pytest.raises(ValueError, match="no metadata.json"):
            read_recorded_trainee_teams(run, require_teams=True)


# ---------------------------------------------------------------------------
# the FIXED behaviour, end to end through the teacher spec
# ---------------------------------------------------------------------------

def test_the_wildcard_resolves_through_an_at_step_suffix():
    with tempfile.TemporaryDirectory() as tmp:
        run, files = _write_specialist_run(tmp)
        got = parse_distill_teacher_spec(f"{run}@26267760:*",
                                         resolve_wildcard=read_recorded_trainee_teams)
        assert got == [(f"{run}@26267760", files)]
        # …and it agrees with the suffix-free spelling on the TEAMS, which is the whole point.
        assert got[0][1] == parse_distill_teacher_spec(
            f"{run}:*", resolve_wildcard=read_recorded_trainee_teams)[0][1]


def test_a_malformed_step_is_a_parse_error_not_a_lookup():
    with tempfile.TemporaryDirectory() as tmp:
        run, _ = _write_specialist_run(tmp)
        with pytest.raises(ValueError, match="is not an integer"):
            parse_distill_teacher_spec(f"{run}@best:*",
                                       resolve_wildcard=read_recorded_trainee_teams)


def test_resolve_zip_and_config_splits_the_step_for_a_stepless_caller():
    """The choke point: every `--distill-teacher` / `--win-prob-pbrs-source` /
    `--distill-anchor-parent` / `--warmstart-consensus` caller passes `step=None`."""
    from agents.training.fixed_opponent_pool import _resolve_zip_and_config
    with tempfile.TemporaryDirectory() as tmp:
        run = os.path.join(tmp, "r")
        os.makedirs(os.path.join(run, "checkpoints"))
        ckpt = os.path.join(run, "checkpoints", "checkpoint_4242_steps.zip")
        with open(ckpt, "wb") as fh:
            fh.write(b"z")
        with open(os.path.join(run, "model_config.json"), "w") as fh:
            json.dump({"arch_signature": "x"}, fh)
        zip_path, cfg, base = _resolve_zip_and_config(f"{run}@4242", None)
        assert zip_path == ckpt and base == "r"
        assert cfg == os.path.join(run, "model_config.json")
        # a disagreeing explicit step is refused, never silently overridden
        with pytest.raises(ValueError, match="give the step once"):
            _resolve_zip_and_config(f"{run}@4242", 999)


# ---------------------------------------------------------------------------
# THE GUARDS — the launch-time refusal and its offline twin
# ---------------------------------------------------------------------------

def test_check_teacher_spec_is_quiet_on_a_good_spec():
    with tempfile.TemporaryDirectory() as tmp:
        run, files = _write_specialist_run(tmp)
        assert check_teacher_spec(f"{run}@26267760:*",
                                  resolve_wildcard=read_recorded_trainee_teams) == []
        assert check_teacher_spec(f"{run}:{files[0]}") == []
        assert check_teacher_spec(None) == []


def test_check_teacher_spec_names_a_teacher_that_is_not_there():
    with tempfile.TemporaryDirectory() as tmp:
        run, files = _write_specialist_run(tmp)
        found = check_teacher_spec(f"{run}_typo:{files[0]}")
        assert found and "does not exist" in found[0] and f"{run}_typo" in found[0]
        # …and the STRUCTURAL half stays quiet: on a real launch that path question is answered
        # downstream (model_build's FATAL_CONFIG), so `resolve_config` does not re-ask it.
        assert check_teacher_spec(f"{run}_typo:{files[0]}", check_paths=False) == []


def test_check_teacher_spec_names_a_missing_team_file():
    with tempfile.TemporaryDirectory() as tmp:
        run, _ = _write_specialist_run(tmp)
        found = check_teacher_spec(f"{run}:{tmp}/nope.txt")
        assert found and "nope.txt" in found[0]
        assert check_teacher_spec(f"{run}:{tmp}/nope.txt", check_paths=False) == []


def test_check_teacher_spec_reports_a_wildcard_that_resolves_to_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        run = os.path.join(tmp, "generalist")
        os.makedirs(run)
        with open(os.path.join(run, "metadata.json"), "w") as fh:
            json.dump({"cli_args": {}}, fh)

        def _require(p):
            return read_recorded_trainee_teams(p, require_teams=True)

        found = check_teacher_spec(f"{run}:*", resolve_wildcard=_require)
        assert found and "recorded NO trainee teams" in found[0]


def _resolve_argv(*extra):
    """Parse + resolve a real argv through the real parser (raises SystemExit on a refusal).

    `--use-bridge node` keeps `resolve_config` off the rust `sim_bridge` binary (the
    `distill_team_bias_test` convention); irrelevant to everything under test here.
    """
    from main.train.config import resolve_config
    from main.train_rl_agent import build_parser
    parser = build_parser()
    args = parser.parse_args(["--steps", "10", "--use-bridge", "node", *extra])
    resolve_config(args, parser)
    return args


def test_the_launch_path_refuses_a_teacher_that_resolves_to_zero_teams():
    """Guard (b): `resolve_config` turns the finding into a parser.error (FATAL_CONFIG class)."""
    with tempfile.TemporaryDirectory() as tmp:
        generalist = os.path.join(tmp, "generalist")
        os.makedirs(generalist)
        with open(os.path.join(generalist, "metadata.json"), "w") as fh:
            json.dump({"cli_args": {}}, fh)
        with pytest.raises(SystemExit):
            _resolve_argv("--distill-coef", "0.1", "--distill-teacher", f"{generalist}:*")


def test_the_launch_path_refuses_a_wildcard_teacher_whose_run_is_not_there():
    """THE REPORTED SHAPE, end to end: `<run>@<step>:*` used to read as a generalist and pass."""
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(SystemExit):
            _resolve_argv("--distill-coef", "0.1",
                          "--distill-teacher", f"{tmp}/not_a_run@26267760:*")


def test_the_launch_path_accepts_an_at_step_teacher_over_a_real_run():
    with tempfile.TemporaryDirectory() as tmp:
        run, files = _write_specialist_run(tmp)
        args = _resolve_argv("--distill-coef", "0.1",
                             "--distill-teacher", f"{run}@26267760:*")
        assert args._distill_pairs == [(f"{run}@26267760", files)]


def test_a_coef_zero_control_arm_with_an_archived_teacher_run_still_resolves():
    """DELIBERATELY NOT REFUSED. At coef 0 no teacher is loaded, so a teacher dir that has since
    been archived is not an error; only the TEAM files have to be real, and they are read by the
    team bias. `main.checkargs` still REPORTS the absent dir — see `check_paths`."""
    with tempfile.TemporaryDirectory() as tmp:
        _run, files = _write_specialist_run(tmp)
        args = _resolve_argv("--distill-coef", "0.0",
                             "--distill-teacher", f"{tmp}/archived_away:{files[0]}")
        assert args._distill_pairs == [(f"{tmp}/archived_away", [files[0]])]


def test_checkargs_reports_the_same_finding_offline():
    """Guard (c): one declaration, both surfaces."""
    from main.checkargs import check, teacher_spec_findings
    with tempfile.TemporaryDirectory() as tmp:
        run, files = _write_specialist_run(tmp)
        bad = ["--steps", "10", "--distill-coef", "0.1",
               "--distill-teacher", f"{run}_typo:{files[0]}"]
        assert any("does not exist" in f for f in teacher_spec_findings(bad))
        assert check(bad)["teacher_spec"]
        good = ["--steps", "10", "--distill-coef", "0.1",
                "--distill-teacher", f"{run}@26267760:*"]
        assert teacher_spec_findings(good) == []
        assert check(good)["teacher_spec"] == []


# ---------------------------------------------------------------------------
# THE CENSUS — nobody re-derives the split
# ---------------------------------------------------------------------------

# Every module that consumes a RUN SPEC. A local `@` split here is the regression this file exists
# to catch; `main/train/matchup_setup.py` is deliberately absent because its `split("@")` parses a
# Showdown team EXPORT line ("Skarmory @ Leftovers"), which is not a run spec.
_RUN_SPEC_MODULES = (
    "agents/training/fixed_opponent_pool.py",
    "agents/training/distill_spec.py",
    "agents/training/exploiter_ladder.py",
    "agents/training/matchup_spec.py",
    "agents/training/warmstart.py",
    "main/train/model_build.py",
    "main/train/callbacks.py",
    "main/train/config.py",
    "main/checkargs.py",
)


def test_no_run_spec_consumer_re_derives_the_at_step_split():
    """An AST census: a `.partition("@")` / `.split("@")` in a run-spec module NAMES that file.

    The splitter is only a class fix while it is the only implementation. This is the test that
    fails when the next consumer writes its own — with the offending file and line in the message.
    """
    offenders = []
    for rel in _RUN_SPEC_MODULES:
        path = str(src_path(rel))
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("partition", "split", "rsplit", "rpartition")
                    and node.args
                    and isinstance(node.args[0], ast.Constant) and node.args[0].value == "@"):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "these run-spec consumers split '@' themselves instead of calling "
        "agents.training.run_spec.split_run_spec: " + ", ".join(offenders))


def test_the_choke_point_calls_the_splitter():
    """`resolve_model_ref` is what makes every `step=None` caller correct at once.

    It is the choke point since `gen3_last_snapshot_resolution_v1`; `_resolve_zip_and_config` is a
    3-tuple wrapper over it (a signature the offline probe scripts import by name), so BOTH are
    checked — the wrapper must reach the splitter, and it must do so through the resolver rather
    than by re-deriving the split for itself.
    """
    path = str(src_path("agents/training/fixed_opponent_pool.py"))
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    called_by_resolver = {n.func.id for n in ast.walk(fns["resolve_model_ref"])
                          if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "split_run_spec" in called_by_resolver, (
        "fixed_opponent_pool.resolve_model_ref no longer splits the run spec — every step=None "
        "caller (--distill-teacher, --win-prob-pbrs-source, --distill-anchor-parent, "
        "--warmstart-consensus) silently loses @step support again.")
    called_by_wrapper = {n.func.id for n in ast.walk(fns["_resolve_zip_and_config"])
                         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "resolve_model_ref" in called_by_wrapper, (
        "_resolve_zip_and_config no longer delegates to resolve_model_ref — the 3-tuple wrapper "
        "and the provenance-carrying resolver would then be two implementations of one rule.")


# ---------------------------------------------------------------------------
# THE CONSUMER CENSUS — every run-spec consumer routes through the ONE resolver
# ---------------------------------------------------------------------------

#: Every module that turns a run spec into a model FILE, and the flags it serves. The rung order
#: (`gen3_last_snapshot_resolution_v1`) is only one rule while these all reach it; a module that
#: opens `best_model/best_model.zip` (or globs `checkpoints/`) for itself is a second rule that
#: will drift from the documented one. Names here are the ONLY sanctioned entry points.
_RESOLVER_ENTRY_POINTS = ("resolve_model_ref", "_resolve_zip_and_config", "resolve_stable_opponents")

#: module -> the flags whose file it resolves (for the failure message)
_RESOLVER_CONSUMERS = {
    "main/train/model_build.py": "--distill-teacher, --win-prob-pbrs-source, --warmstart-consensus",
    "main/train/callbacks.py": "--distill-anchor-parent",
    "main/train/matchup_setup.py": "--stable-opponents, --exploiter",
    "agents/training/warmstart.py": "--warmstart-consensus (the standalone CLI)",
    "agents/training/exploiter_ladder.py": "--exploiter-ladder",
}

#: Filenames a consumer must not construct for itself — the rungs the resolver owns.
#:
#: ⚠️ `final_model*.zip` is deliberately ABSENT. The trainer WRITES those names
#: (`_write_latest_txt(model_dir, "final_model.zip")`), and an AST scan cannot tell a producer's
#: literal from a consumer's; listing it would fail on `main/train/model_build.py`'s own save path.
#: The two below are named only in order to RESOLVE, which is the regression this catches — a
#: consumer reaching for `best_model/best_model.zip` keeps the OLD bot-win-rate selection after the
#: rest of the tree moved to the run's last snapshot.
_RUNG_FILENAMES = ("best_model.zip", "latest.txt")


def test_every_run_spec_consumer_routes_through_the_one_resolver():
    """Each consumer imports a sanctioned entry point and hand-builds no rung filename.

    The rung order is a single rule only while every flag reaches it. This is the test that fails
    — naming the file and the flags — when the next consumer opens `best_model/best_model.zip`
    itself and quietly keeps the OLD (bot-win-rate) selection after the rest of the tree moved to
    the run's last snapshot.
    """
    missing, offenders = [], []
    for rel, flags in _RESOLVER_CONSUMERS.items():
        path = str(src_path(rel))
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        tree = ast.parse(text, filename=path)
        imported = {alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    and (node.module or "").endswith("fixed_opponent_pool")
                    for alias in node.names}
        if not (imported & set(_RESOLVER_ENTRY_POINTS)):
            missing.append(f"{rel} ({flags})")
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for name in _RUNG_FILENAMES:
                    if node.value.endswith(name):
                        offenders.append(f"{rel}:{node.lineno} builds {node.value!r}")
    assert not missing, (
        "these run-spec consumers no longer import agents.training.fixed_opponent_pool's resolver, "
        "so their flag resolves a model file by some other rule: " + ", ".join(missing))
    assert not offenders, (
        "these consumers name a resolution RUNG's filename themselves instead of calling "
        "resolve_model_ref, which is how the rung order stops being one rule: "
        + ", ".join(offenders))


# ---------------------------------------------------------------------------
# the REAL run archive — reproduces the reported case; skips cleanly without models/
# ---------------------------------------------------------------------------

def test_a_real_specialist_run_resolves_with_and_without_the_step():
    models = main_models_dir()
    if models is None:
        pytest.skip("no models/ archive on this box (see utils.paths.main_models_dir)")
    run = None
    for name in sorted(os.listdir(str(models))):
        cand = os.path.join(str(models), name)
        meta = os.path.join(cand, "metadata.json")
        if not os.path.isfile(meta):
            continue
        try:
            with open(meta) as fh:
                cli = (json.load(fh).get("cli_args") or {})
        except (OSError, ValueError):
            continue
        if not (cli.get("trainee_teams") or cli.get("trainee_team")):
            continue
        try:
            if read_recorded_trainee_teams(cand):
                run = cand
                break
        except (FileNotFoundError, ValueError):
            continue        # a recorded team file that moved: not this test's subject
    if run is None:
        pytest.skip("no run in models/ records a --trainee-team(s) pin")
    teams = read_recorded_trainee_teams(run)
    assert teams
    assert parse_distill_teacher_spec(
        f"{run}@26267760:*", resolve_wildcard=read_recorded_trainee_teams) == [
            (f"{run}@26267760", teams)]
