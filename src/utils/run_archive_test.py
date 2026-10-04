"""Runs ALWAYS land in the MAIN checkout's run archive — even when launched from a git worktree.

THE DEFECT (2026-09-23: eight runs; 2026-10-01/02: three agents' run data nearly). The trainer and
the launcher named a run dir ``models/<name>`` RELATIVE to the cwd. In a linked worktree that is the
worktree's own ``models/`` — gitignored, so a clean ``git status`` hides it — and removing the
worktree deleted the runs SILENTLY.

THE FIX under test (``utils.paths``): ``run_archive_dir()`` is the ONE answer to "where does a run
directory land" (``$GEN3AI_MODELS_DIR`` if set, else the MAIN checkout's ``models/``, never the
worktree's), ``checked_run_dir()`` is the ONE refusal for an explicit run dir inside a linked
worktree's own ``models/``, and ``RunArchiveError`` is a typed FATAL_CONFIG.

Every test here BUILDS a throwaway main checkout + linked worktree and points ``utils.paths._archive_anchor``
(what ``run_archive_dir`` / ``checked_run_dir`` read at call time) at the worktree, so "launched from a worktree" is
exercised on any box — including the owner's, where this suite itself runs from a worktree and must
never touch the real ``models/``. Each assertion fails on revert of the line it pins.
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import utils.paths as paths
from main.exit_codes import TrainExitCode, exit_code_for
from utils.paths import (
    MODELS_DIR_ENV_VAR,
    RUN_ARCHIVE_SEAL_ENV_VAR,
    RunArchiveError,
    checked_run_dir,
    is_linked_worktree,
    new_run_dir,
    resolve_archive_ref,
    run_archive_dir,
)

launcher_run = importlib.import_module("main.launcher.run")
TS = "20260608_120000"


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    """A throwaway MAIN checkout (with a ``models/`` archive) and a linked WORKTREE of it."""
    main = (tmp_path / "main").resolve()
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    _git(main, "config", "user.email", "t@t")
    _git(main, "config", "user.name", "t")
    (main / "marker.txt").write_text("x")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "c1")
    (main / "models").mkdir()
    wt = (tmp_path / "wt").resolve()
    _git(main, "worktree", "add", "-q", "-b", "feat", str(wt))
    return main, wt


@pytest.fixture
def from_worktree(repos, monkeypatch):
    """This process 'is' the worktree: the archive is asked from the worktree, cwd is the worktree,
    and neither ``$GEN3AI_MODELS_DIR`` nor the test seal is set — exactly an agent's launch."""
    main, wt = repos
    monkeypatch.setattr(paths, "_archive_anchor", lambda: wt)
    monkeypatch.delenv(MODELS_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(RUN_ARCHIVE_SEAL_ENV_VAR, raising=False)
    monkeypatch.chdir(wt)
    return main, wt


# ------------------------------------------------------------------------ run_archive_dir
def test_a_worktree_launch_resolves_to_the_MAIN_checkouts_models(repos):
    main, wt = repos
    assert is_linked_worktree(wt) and not is_linked_worktree(main)
    assert run_archive_dir(anchor=wt) == main / "models"      # NOT wt / "models"
    assert new_run_dir("my_run", anchor=wt) == str(main / "models" / "my_run")


def test_the_main_checkout_is_unchanged_its_own_models(repos):
    main, _wt = repos
    assert run_archive_dir(anchor=main) == main / "models"


def test_a_fresh_clone_with_no_models_yet_still_creates_it_in_the_main_checkout(tmp_path):
    clone = (tmp_path / "clone").resolve()
    clone.mkdir()
    _git(clone, "init", "-q", "-b", "main")
    assert not (clone / "models").exists()
    assert run_archive_dir(anchor=clone) == clone / "models"      # creatable, as before this change


def test_a_worktree_whose_main_has_no_archive_REFUSES_naming_the_env_var(repos):
    main, wt = repos
    (main / "models").rmdir()
    with pytest.raises(RunArchiveError) as e:
        run_archive_dir(anchor=wt)
    assert MODELS_DIR_ENV_VAR in str(e.value), "the refusal must name the remedy"
    assert not (wt / "models").exists(), "and it must not have quietly made a worktree-local one"


def test_a_worktree_git_cannot_resolve_REFUSES_instead_of_falling_back(repos, monkeypatch):
    _main, wt = repos
    monkeypatch.setattr(paths, "_main_checkout_of", lambda here: None)
    with pytest.raises(RunArchiveError, match=MODELS_DIR_ENV_VAR):
        run_archive_dir(anchor=wt)


def test_the_env_var_is_AUTHORITATIVE_and_a_missing_dir_refuses(repos, tmp_path, monkeypatch):
    _main, wt = repos
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv(MODELS_DIR_ENV_VAR, str(elsewhere))
    assert run_archive_dir(anchor=wt) == elsewhere
    monkeypatch.setenv(MODELS_DIR_ENV_VAR, str(tmp_path / "typo_dir"))
    with pytest.raises(RunArchiveError, match=MODELS_DIR_ENV_VAR):
        run_archive_dir(anchor=wt)
    assert not (tmp_path / "typo_dir").exists(), "a typo must not mint an archive"


def test_the_run_archive_is_SEALED_for_every_test_unless_isolated(monkeypatch):
    """THE TEST-ISOLATION GUARANTEE. conftest seals the archive for the whole pytest tree, so a
    test that creates a run dir and forgot to isolate FAILS here instead of writing into the real
    `models/`. Reverting the conftest line (or the seal branch in `run_archive_dir`) fails this."""
    assert os.environ.get(RUN_ARCHIVE_SEAL_ENV_VAR), "the root conftest must set the seal"
    monkeypatch.delenv(MODELS_DIR_ENV_VAR, raising=False)
    with pytest.raises(RunArchiveError, match="SEALED"):
        run_archive_dir()
    with pytest.raises(RunArchiveError, match="SEALED"):
        new_run_dir("anything")


def test_the_run_archive_fixture_isolates(run_archive):
    assert run_archive_dir() == run_archive
    assert str(run_archive).startswith(str(Path(os.environ[MODELS_DIR_ENV_VAR]).parent))


def test_run_archive_error_is_a_typed_FATAL_CONFIG():
    assert RunArchiveError.exit_code == int(TrainExitCode.FATAL_CONFIG)
    assert exit_code_for(RunArchiveError("x")) == int(TrainExitCode.FATAL_CONFIG)


# ------------------------------------------------------------------------ checked_run_dir
def test_an_explicit_dir_in_the_worktrees_own_models_is_REFUSED(from_worktree):
    main, wt = from_worktree
    with pytest.raises(RunArchiveError) as e:
        checked_run_dir(wt / "models" / "my_run")
    msg = str(e.value)
    assert "OWN models/" in msg and str(wt) in msg and str(main / "models") in msg
    assert "2026-09-23" in msg, "the refusal names the hazard it prevents"
    # a cwd-relative spelling is the same directory
    with pytest.raises(RunArchiveError):
        checked_run_dir("models/my_run")


def test_an_explicit_dir_in_mains_models_or_anywhere_else_is_allowed(from_worktree, tmp_path):
    main, wt = from_worktree
    assert checked_run_dir(main / "models" / "my_run") == str(main / "models" / "my_run")
    assert checked_run_dir(tmp_path / "scratch" / "run") == str(tmp_path / "scratch" / "run")
    assert checked_run_dir(wt / "elsewhere" / "run") == str(wt / "elsewhere" / "run")  # not models/


def test_ANOTHER_linked_worktrees_models_is_refused_too(repos, tmp_path):
    main, wt = repos
    other = (tmp_path / "other").resolve()
    _git(main, "worktree", "add", "-q", "-b", "feat2", str(other))
    with pytest.raises(RunArchiveError, match=str(other)):
        checked_run_dir(other / "models" / "r", anchor=wt)


def test_a_worktree_models_symlinked_OUT_to_mains_archive_is_allowed(repos):
    main, wt = repos
    (wt / "models").symlink_to(main / "models", target_is_directory=True)
    got = checked_run_dir(wt / "models" / "r", anchor=wt)      # lands in main, so it survives
    assert os.path.realpath(got) == str(main / "models" / "r")


def test_an_archive_entry_that_symlinks_INTO_a_worktree_is_refused(repos):
    """The 2026-09-23 shape: main's models/ held a SYMLINK into a worktree's gitignored tree."""
    main, wt = repos
    (wt / "models").mkdir()
    (wt / "models" / "run_x").mkdir()
    (main / "models" / "run_x").symlink_to(wt / "models" / "run_x", target_is_directory=True)
    with pytest.raises(RunArchiveError):
        checked_run_dir(main / "models" / "run_x", anchor=wt)


# ------------------------------------------------------------------------ resolve_archive_ref
def test_a_models_ref_the_cwd_lacks_is_anchored_at_the_archive(from_worktree):
    main, wt = from_worktree
    (main / "models" / "parent" / "checkpoints").mkdir(parents=True)
    ck = main / "models" / "parent" / "checkpoints" / "c.zip"
    ck.write_text("x")
    assert resolve_archive_ref("models/parent/checkpoints/c.zip") == str(ck)
    assert resolve_archive_ref("models/not_there/c.zip") == "models/not_there/c.zip"   # unchanged
    assert resolve_archive_ref(str(ck)) == str(ck)                                    # absolute
    assert resolve_archive_ref("other/dir/c.zip") == "other/dir/c.zip"                # not models/
    (wt / "models" / "parent" / "checkpoints").mkdir(parents=True)                    # cwd HAS it
    assert resolve_archive_ref("models/parent/checkpoints") == "models/parent/checkpoints"


# ------------------------------------------------------------------------ the trainer
def test_the_trainer_from_a_worktree_lands_in_mains_models(from_worktree):
    from main.train.run_io import _resolve_fresh_model_dir, _resolve_model_dir
    main, wt = from_worktree
    assert _resolve_fresh_model_dir("exp1", None, None) == str(main / "models" / "exp1")
    assert _resolve_fresh_model_dir(None, "ext_tgt", None) == str(main / "models" / "rb_exploiter_vs_tgt")
    minted = _resolve_fresh_model_dir(None, None, None)
    assert Path(minted).parent == main / "models" and Path(minted).name.startswith("rb_run_")
    assert _resolve_model_dir(None, "exp2", None, None) == str(main / "models" / "exp2")
    assert not (wt / "models").exists(), "nothing may have been made under the worktree"


def test_the_trainers_main_decides_its_run_dir_ONLY_through_resolve_model_dir():
    """The wiring: `main()` must not take `args.run_dir` verbatim or name a `models/` itself (the
    `--debug` smoke cannot cover it from a worktree — it would write the real archive)."""
    import inspect

    import main.train_rl_agent as hub
    src = inspect.getsource(hub.main)
    assert "_resolve_model_dir(" in src
    assert "model_dir = args.run_dir" not in src and "_resolve_fresh_model_dir(" not in src
    assert '"models"' not in src and "'models'" not in src


def test_the_trainer_REFUSES_an_explicit_run_dir_in_a_worktrees_own_models(from_worktree, capsys):
    from main.train.run_io import _resolve_model_dir
    main, wt = from_worktree
    with pytest.raises(SystemExit) as e:
        _resolve_model_dir(str(wt / "models" / "r"), None, None, None)
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    assert "OWN models/" in capsys.readouterr().err
    assert _resolve_model_dir(str(main / "models" / "r"), None, None, None) == str(main / "models" / "r")


def test_the_trainer_with_no_archive_exits_FATAL_CONFIG_naming_the_env_var(from_worktree, capsys):
    from main.train.run_io import _resolve_fresh_model_dir
    main, _wt = from_worktree
    (main / "models").rmdir()
    with pytest.raises(SystemExit) as e:
        _resolve_fresh_model_dir("exp", None, None)
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    assert MODELS_DIR_ENV_VAR in capsys.readouterr().err


def test_planned_run_dir_agrees_with_the_directory_the_trainer_will_pick(from_worktree):
    """`matchup_setup._planned_run_dir` decides RESTART vs FORK for a `--run-name` launch; a
    cwd-relative `models/<name>` would disagree with the trainer's dir from a worktree."""
    from types import SimpleNamespace

    from main.train.matchup_setup import _planned_run_dir
    from main.train.run_io import _resolve_fresh_model_dir
    ns = SimpleNamespace(run_dir=None, run_name="named")
    assert _planned_run_dir(ns) == _resolve_fresh_model_dir("named", None, None)


# ------------------------------------------------------------------------ the launcher
def test_the_launcher_from_a_worktree_lands_in_mains_models(from_worktree):
    from main.launcher.checkpoint import _resolve_fresh_run_dir, resolve_launch_run_dir
    main, wt = from_worktree
    assert resolve_launch_run_dir(["--debug"], TS) == str(main / "models" / f"rb_run_{TS}")
    assert resolve_launch_run_dir(["--run-name", "named"], TS) == str(main / "models" / "named")
    assert _resolve_fresh_run_dir(["--run-name=named"], TS) == str(main / "models" / "named")
    assert not (wt / "models").exists()


def test_the_launcher_REFUSES_an_explicit_run_dir_in_the_worktrees_models(from_worktree):
    from main.launcher.checkpoint import resolve_launch_run_dir
    _main, wt = from_worktree
    for argv in (["--run-dir", "models/typed"], ["--run-dir", str(wt / "models" / "typed")]):
        with pytest.raises(RunArchiveError) as e:
            resolve_launch_run_dir(argv, TS)
        assert e.value.exit_code == int(TrainExitCode.FATAL_CONFIG)


def test_a_RESUME_of_a_checkpoint_inside_a_worktrees_models_is_refused(from_worktree):
    from main.launcher.checkpoint import resolve_launch_run_dir
    _main, wt = from_worktree
    ck = wt / "models" / "old_run" / "checkpoints" / "checkpoint_1_steps.zip"
    ck.parent.mkdir(parents=True)
    ck.write_text("x")
    with pytest.raises(RunArchiveError):
        resolve_launch_run_dir(["--model", str(ck)], TS)


def test_a_relative_models_checkpoint_typed_in_a_worktree_resumes_the_ARCHIVES_run(from_worktree):
    """`--model models/<run>/checkpoints/x.zip` — the CLAUDE.md spelling — typed where the cwd has no
    `models/`: the run it continues is main's, not a phantom `<worktree>/models/<run>`."""
    from main.launcher.checkpoint import (_find_model_arg, archive_anchored_args,
                                          resolve_launch_run_dir)
    main, _wt = from_worktree
    ck = main / "models" / "live_run" / "checkpoints" / "checkpoint_5_steps.zip"
    ck.parent.mkdir(parents=True)
    ck.write_text("x")
    argv = archive_anchored_args(["--model", "models/live_run/checkpoints/checkpoint_5_steps.zip"])
    assert _find_model_arg(argv) == str(ck)
    assert resolve_launch_run_dir(argv, TS) == str(main / "models" / "live_run")


def test_a_main_checkout_launch_is_byte_identical_for_a_model_the_cwd_has(repos, monkeypatch):
    from main.launcher.checkpoint import archive_anchored_args
    main, _wt = repos
    monkeypatch.setattr(paths, "_archive_anchor", lambda: main)
    monkeypatch.delenv(MODELS_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(RUN_ARCHIVE_SEAL_ENV_VAR, raising=False)
    monkeypatch.chdir(main)
    (main / "models" / "r").mkdir()
    (main / "models" / "r" / "c.zip").write_text("x")
    argv = ["--model", "models/r/c.zip", "--steps", "9"]
    assert archive_anchored_args(argv) == argv


def test_the_restart_lookup_is_run_scoped_and_never_cwd_relative(from_worktree):
    """`find_latest_checkpoint(run_dir=…)` takes no `models/` root at all, and an UNSCOPED lookup
    searches the archive — main's, not a cwd-relative one."""
    from main.launcher.checkpoint import find_latest_checkpoint
    main, wt = from_worktree
    run = main / "models" / "r"
    (run / "checkpoints").mkdir(parents=True)
    ck = run / "checkpoints" / "checkpoint_9_steps.zip"
    ck.write_text("x")
    assert find_latest_checkpoint(run_dir=str(run)) == str(ck)
    assert find_latest_checkpoint() == str(ck)                  # unscoped → the ARCHIVE
    assert not (wt / "models").exists()


def _pinned_session(repos_main, argv, monkeypatch, *, pin=True):
    """`_prepare_session` with the PIN on (the production path), against the throwaway repo."""
    from main.launcher.state import LauncherState
    first = _git(repos_main, "rev-parse", "HEAD")
    monkeypatch.setattr(launcher_run, "get_repo_root", lambda *a, **k: str(repos_main))
    monkeypatch.setattr(launcher_run, "_prune_stale_launcher_worktrees", lambda *a, **k: None)
    monkeypatch.setattr(launcher_run, "_create_run_worktree",
                        lambda h, run_dir=None: ("/t/train.py", "/t/src", lambda: None))
    monkeypatch.setattr(launcher_run.atexit, "register", lambda *a, **k: None)
    return launcher_run._prepare_session(
        LauncherState(0.0), [*argv, "--allow-nonproduction-arch", "--allow-nonproduction-recipe"],
        interval_hours=0.0, pin=pin, sync_to_main=False, pin_commit=first[:8] if pin else None,
        grace_minutes=1.0, max_crash_restarts=0)


def test_a_PINNED_child_gets_an_ABSOLUTE_run_dir_in_mains_models(from_worktree, monkeypatch):
    """Item 3 of the brief. A pinned child (`/tmp/launcher-<sha>/…`) is spawned with NO `cwd=` (it
    inherits the launcher's), and may run an OLD commit that knows nothing of the archive — so what
    keeps it writing in main's `models/` is that the launcher hands it an ABSOLUTE `--run-dir`.
    From a worktree that used to be a cwd-relative `models/<run>` = the worktree's own."""
    from main.launcher.checkpoint import _peek_arg
    main, wt = from_worktree
    ctx = _pinned_session(main, ["--run-name", "pinned_run", "--steps", "1"], monkeypatch)
    child_run_dir = _peek_arg(ctx.child_args, "--run-dir")
    assert child_run_dir == str(main / "models" / "pinned_run")
    assert os.path.isabs(child_run_dir)
    assert (main / "models" / "pinned_run").is_dir(), "created in main's archive"
    assert not (wt / "models").exists(), "and NOT under the worktree the launcher stands in"


def _archived_run(main, name="live_run", steps=5):
    """A resumable run in MAIN's archive (and only there), recorded under THIS interpreter's torch."""
    import json
    from importlib.metadata import version
    run = main / "models" / name
    (run / "checkpoints").mkdir(parents=True)
    ck = run / "checkpoints" / f"checkpoint_{steps}_steps.zip"
    ck.write_text("x")
    (run / "metadata.json").write_text(json.dumps({"torch_version": version("torch")}))
    (run / "model_config.json").write_text("{}")
    (run / "latest.txt").write_text(f"checkpoints/checkpoint_{steps}_steps.zip")
    return run, ck


def test_a_launch_resuming_a_relative_models_checkpoint_in_a_worktree_continues_MAINS_run(
        from_worktree, monkeypatch):
    """The CLAUDE.md resume spelling (`--model models/<run>/checkpoints/<ckpt>.zip`) typed in a
    worktree: the session's run dir and the child's `--model` are the ARCHIVE's, absolute — not a
    phantom `<worktree>/models/<run>` (which is now refused, and before was silently a dead path)."""
    from main.launcher.checkpoint import _find_model_arg, _peek_arg
    main, wt = from_worktree
    run, ck = _archived_run(main)
    ctx = _pinned_session(main, ["--model", "models/live_run/checkpoints/checkpoint_5_steps.zip",
                                 "--steps", "9"], monkeypatch, pin=False)
    assert _find_model_arg(ctx.child_args) == str(ck)
    assert _peek_arg(ctx.child_args, "--run-dir") == str(run) == ctx.run_dir
    assert not (wt / "models").exists()


def test_dry_run_anchors_the_same_relative_checkpoint_at_the_archive(from_worktree, capsys):
    from main.launcher.dry_run import dry_run
    main, wt = from_worktree
    run, _ck = _archived_run(main)
    rc = dry_run(["--model", "models/live_run/checkpoints/checkpoint_5_steps.zip", "--steps", "9"],
                 interval_hours=0.0, pin=False, sync_to_main=False, pin_commit=None,
                 grace_minutes=1.0, max_crash_restarts=0, nice=10)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"run dir     : {run}" in out and f"RESTART of {run}" in out
    assert not (wt / "models").exists()


def test_the_child_is_spawned_with_no_cwd_so_only_the_absolute_run_dir_pins_its_location(monkeypatch):
    """The other half of item 3, pinned so it cannot be 'fixed' by accident: `_launch_child` passes no
    `cwd=`, which is exactly why the run dir handed to it must not be cwd-relative."""
    from unittest.mock import MagicMock, patch

    from main.launcher.child import _launch_child
    from main.launcher.state import LauncherState
    proc = MagicMock(pid=1, stdout=None)
    with patch("main.launcher.child.subprocess.Popen", return_value=proc) as popen, \
            patch("main.launcher.child.threading.Thread"), \
            patch("main.launcher.child.os.pipe", return_value=(3, 4)), \
            patch("main.launcher.child.os.close"), \
            patch("main.launcher.child._open_child_log", return_value=None):
        _launch_child(["--run-dir", "/abs/models/r"], {}, LauncherState(0.0), "train.py", "/src")
    assert "cwd" not in popen.call_args.kwargs


def test_dry_run_from_a_worktree_prints_mains_models_and_creates_nothing(from_worktree, capsys):
    from main.launcher.dry_run import dry_run
    main, wt = from_worktree
    rc = dry_run(["--run-name", "dry", "--steps", "1", "--allow-nonproduction-arch",
                  "--allow-nonproduction-recipe"],
                 interval_hours=0.0, pin=False, sync_to_main=False, pin_commit=None,
                 grace_minutes=1.0, max_crash_restarts=0, nice=10)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"run dir     : {main / 'models' / 'dry'}" in out
    assert "would be created" in out
    assert not (main / "models" / "dry").exists() and not (wt / "models").exists()


def test_dry_run_REFUSES_a_run_dir_in_the_worktrees_own_models(from_worktree, capsys):
    from main.launcher.dry_run import dry_run
    _main, _wt = from_worktree
    rc = dry_run(["--run-dir", "models/typed", "--steps", "1"],
                 interval_hours=0.0, pin=False, sync_to_main=False, pin_commit=None,
                 grace_minutes=1.0, max_crash_restarts=0, nice=10)
    assert rc == int(TrainExitCode.FATAL_CONFIG)
    assert "OWN models/" in capsys.readouterr().out


def test_prepare_session_REFUSES_with_FATAL_CONFIG_and_creates_nothing(from_worktree, monkeypatch,
                                                                    capsys):
    main, wt = from_worktree
    with pytest.raises(SystemExit) as e:
        _pinned_session(main, ["--run-dir", "models/typed", "--steps", "1"], monkeypatch)
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    assert "OWN models/" in capsys.readouterr().err
    assert not (wt / "models").exists()


def test_checkargs_names_the_run_dir_the_launch_will_use_and_never_raises(from_worktree):
    """`main.checkargs` is the OFFLINE twin of the launch: its `--run-name` run dir must be the
    launch's (main's archive), or a worktree's RESTART-vs-FORK read and pin decision disagree with
    the real launch. Report-only: a box with no archive answers the old relative path, not a raise
    (the real launch is what refuses)."""
    from types import SimpleNamespace

    from main.checkargs import argv_run_dir, effective_run_dir
    main, _wt = from_worktree
    want = str(main / "models" / "n")
    assert argv_run_dir(["--run-name", "n"]) == want
    assert effective_run_dir(SimpleNamespace(run_dir=None, run_name="n")) == want
    assert argv_run_dir(["--run-dir", "given", "--run-name", "n"]) == "given"
    (main / "models").rmdir()
    assert argv_run_dir(["--run-name", "n"]) == os.path.join("models", "n")


# ------------------------------------------------------------------------ the other writers
def test_the_team_completion_trainer_and_visualize_arch_default_into_mains_archive(from_worktree):
    """`train_team_completion` built `<git toplevel>/models/team_prediction` (the WORKTREE) and
    `visualize_arch` defaulted `--out` to a cwd-relative `models/_arch/…`. Both are source-pinned to
    the archive helper now — no `repo_root`/cwd-relative `models` join may come back."""
    import inspect

    import main.train_team_completion as ttc
    import main.visualize_arch as va
    assert 'os.path.join(repo_root, "models"' not in inspect.getsource(ttc)
    assert 'default="models/_arch' not in inspect.getsource(va)
    assert "run_archive_dir" in inspect.getsource(ttc) and "run_archive_dir" in inspect.getsource(va)


def test_elo_REFUSES_to_write_its_default_output_inside_a_worktrees_models(from_worktree,
                                                                           monkeypatch, capsys):
    import main.elo as elo
    _main, wt = from_worktree
    (wt / "models" / "old_run").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["elo", str(wt / "models" / "old_run")])
    assert elo.main() == int(TrainExitCode.FATAL_CONFIG)
    assert "OWN models/" in capsys.readouterr().err
    assert not (wt / "models" / "old_run" / "elo").exists()


def test_elo_refit_apply_REFUSES_inside_a_worktrees_models_but_the_read_is_unaffected(
        from_worktree, capsys):
    import main.elo as elo
    from main.elo_refit_test import _build_run          # a three-snapshot run with a stale ladder
    from agents.training import snapshot_ladder as sl
    main, wt = from_worktree
    (wt / "models").mkdir()
    run = _build_run(wt / "models")                      # <wt>/models/run_fixture
    assert elo.refit_main([run]) == 0, "the read-only refit writes nothing and is unaffected"
    before = open(sl.ladder_json_path(run)).read()
    assert elo.refit_main([run, "--apply"]) == int(TrainExitCode.FATAL_CONFIG)
    assert "OWN models/" in capsys.readouterr().err
    assert open(sl.ladder_json_path(run)).read() == before, "the committed ladder is untouched"
    assert not os.path.exists(os.path.join(run, "snapshot_ladder", sl.PRE_RECIPE_BACKUP_NAME))
    # the same run in MAIN's archive is applied as always (the guard is about WHERE, not about refit)
    (main / "models").mkdir(exist_ok=True)
    ok_run = _build_run(main / "models")
    assert elo.refit_main([ok_run, "--apply"]) == 0
    assert os.path.exists(os.path.join(ok_run, "snapshot_ladder", sl.PRE_RECIPE_BACKUP_NAME))


def test_capacity_and_scaffolding_gauge_REFUSE_a_default_output_in_a_worktrees_models(
        from_worktree, capsys):
    import main.capacity as cap
    import main.scaffolding_gauge as sg
    _main, wt = from_worktree
    run = wt / "models" / "old_run"
    run.mkdir(parents=True)
    (run / "final_model.zip").write_text("x")
    assert cap.main([str(run), "--quiet"]) == int(TrainExitCode.FATAL_CONFIG)
    assert sg.main([str(run), "--quiet"]) == int(TrainExitCode.FATAL_CONFIG)
    err = capsys.readouterr().err
    assert err.count("OWN models/") == 2
    assert not (run / "capacity_battery.json").exists()
    assert not (run / "scaffolding_gauge.json").exists()


def test_lineage_backfill_apply_REFUSES_inside_a_worktrees_models(from_worktree, capsys):
    import json

    import main.lineage as lin
    _main, wt = from_worktree
    run = wt / "models" / "old_run"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps(
        {"original_command": "python x --model models/parent/checkpoints/c.zip --run-name old_run"}))
    before = (run / "metadata.json").read_text()
    # dry-run (no --apply) is unaffected; --apply is refused and writes nothing
    assert lin.main([str(run), "--backfill"]) == 0
    assert lin.main([str(run), "--backfill", "--apply"]) == int(TrainExitCode.FATAL_CONFIG)
    assert "OWN models/" in capsys.readouterr().err
    assert (run / "metadata.json").read_text() == before
