"""`--pin-commit` — pin the isolated worktree to a NAMED commit.

WHY IT EXISTS. A batch of arms launched sequentially under `--sync-to-main` each pins to
whatever HEAD is at *its own* launch, so a commit landing mid-batch silently splits the batch
across two commits (2026-09-04: arm 1 on 0c76e2ee, arms 2-4 on 52ab5914 — and nothing in any
run's output said so). `--pin-commit <sha>` takes HEAD out of the decision entirely.

The four refusals this file pins, each of which would otherwise be a silently-wrong-code run:
  * an unresolvable sha (never a quiet fall-back to HEAD)          -> FATAL_CONFIG (3)
  * `--pin-commit` beside `--sync-to-main` (two sources of truth)  -> argparse, at parse time
  * a same-run RESTART whose checkpoint records a DIFFERENT hash   -> FATAL_CONFIG (3)
  * `--pin-commit` beside `--no-pin` (pin what, exactly?)          -> argparse

Run: python -m pytest src/main/launcher/pin_commit_test.py -q
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import json
import os
import subprocess

import pytest

from main.exit_codes import TrainExitCode
import importlib

import main.launcher.worktree as wt

# `main.launcher.__init__` re-exports the `run` FUNCTION under that name, so
# `import main.launcher.run` would hand back the function, not the module.
launcher_run = importlib.import_module("main.launcher.run")
from main.launcher.state import LauncherState


# ---------------------------------------------------------------------------------------
# A real (tiny) git repo — the resolution IS `git rev-parse`, so a fake would test nothing.
# ---------------------------------------------------------------------------------------

def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A 2-commit repo; yields (root, [sha_first, sha_second])."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    # `_create_run_worktree` symlinks deps/pokemon-showdown into the checkout, so the tree
    # must actually have a deps/ directory — as the real repo does.
    (root / "deps").mkdir()
    (root / "deps" / ".keep").write_text("")
    shas = []
    for n, text in ((1, "first"), (2, "second")):
        (root / "marker.txt").write_text(text)
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", f"commit {n}: {text}")
        shas.append(_git(root, "rev-parse", "HEAD"))
    return str(root), shas


# ---------------------------------------------------------------------------------------
# (a) a prefix resolves to the full sha, and the worktree really lands there
# ---------------------------------------------------------------------------------------

def test_a_prefix_resolves_to_the_full_sha_and_its_subject(repo):
    root, (first, _second) = repo
    sha, subject = wt.resolve_commit(first[:7], repo_root=root)
    assert sha == first, "a 7-char prefix must resolve to the FULL sha"
    assert subject == "commit 1: first"


def test_a_the_worktree_is_checked_out_at_the_named_commit(repo, monkeypatch):
    """The pin is only real if the files on disk are that commit's."""
    root, (first, second) = repo
    monkeypatch.setattr(wt, "get_repo_root", lambda *a, **k: root)
    sha, _ = wt.resolve_commit(first[:7], repo_root=root)
    _train_script, src_dir, cleanup = wt._create_run_worktree(sha)
    try:
        marker = os.path.join(os.path.dirname(src_dir), "marker.txt")
        with open(marker) as f:
            assert f.read() == "first", "worktree is checked out at the wrong commit"
        assert _git(root, "rev-parse", "HEAD") == second, "HEAD must be untouched"
    finally:
        cleanup()


# ---------------------------------------------------------------------------------------
# (b) an unresolvable sha fails LOUDLY, with the FATAL_CONFIG exit code
# ---------------------------------------------------------------------------------------

def test_b_unresolvable_sha_is_refused_never_silently_head(repo):
    root, _ = repo
    with pytest.raises(wt.PinRefused) as e:
        wt.resolve_commit("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", repo_root=root)
    assert e.value.exit_code == int(TrainExitCode.FATAL_CONFIG)
    assert "deadbeef" in str(e.value) and "--pin-commit" in str(e.value)


def test_b_a_name_that_does_not_exist_never_falls_back_to_head(repo):
    root, (first, _) = repo
    for spec in ("no-such-branch", "zzzz"):
        with pytest.raises(wt.PinRefused):
            wt.resolve_commit(spec, repo_root=root)
    assert wt.resolve_commit(first, repo_root=root)[0] == first


def test_b_prepare_session_exits_fatal_config(repo, tmp_path, monkeypatch, capsys):
    """End to end through the launcher's own session setup: exit code 3, reason on stderr."""
    root, _ = repo
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher_run, "get_repo_root", lambda *a, **k: root)
    monkeypatch.setattr(launcher_run, "_prune_stale_launcher_worktrees", lambda *a, **k: None)
    monkeypatch.setattr(
        launcher_run, "_create_run_worktree",
        lambda h: pytest.fail("must never reach worktree creation with an unresolvable sha"))
    with pytest.raises(SystemExit) as e:
        launcher_run._prepare_session(
            LauncherState(0.0), ["--steps", "1"],
            interval_hours=0.0, pin=True, sync_to_main=False,
            pin_commit="deadbeefdeadbeef", grace_minutes=1.0, max_crash_restarts=0,
        )
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    assert "--pin-commit" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------
# (c) --pin-commit + --sync-to-main is refused AT PARSE TIME (two sources of truth)
# ---------------------------------------------------------------------------------------

def test_c_pin_commit_with_sync_to_main_is_refused(monkeypatch):
    parser = launcher_run.build_launcher_parser()
    monkeypatch.setattr("sys.argv", ["launcher", "--pin-commit", "abc1234", "--sync-to-main"])
    with pytest.raises(SystemExit):
        parser.parse_known_args()


def test_c_either_one_alone_parses(monkeypatch):
    parser = launcher_run.build_launcher_parser()
    monkeypatch.setattr("sys.argv", ["launcher", "--pin-commit", "abc1234"])
    assert parser.parse_known_args()[0].pin_commit == "abc1234"
    monkeypatch.setattr("sys.argv", ["launcher", "--sync-to-main"])
    known = parser.parse_known_args()[0]
    assert known.sync_to_main and known.pin_commit is None


def test_c_legacy_pin_to_hash_spelling_still_parses(monkeypatch):
    parser = launcher_run.build_launcher_parser()
    monkeypatch.setattr("sys.argv", ["launcher", "--pin-to-hash", "abc1234"])
    assert parser.parse_known_args()[0].pin_commit == "abc1234"


def test_c_pin_commit_with_no_pin_is_refused(monkeypatch, capsys):
    """--no-pin says "no worktree at all"; naming a commit for it is incoherent."""
    monkeypatch.setattr(
        "sys.argv", ["launcher", "--pin-commit", "abc1234", "--no-pin", "--steps", "1"])
    with pytest.raises(SystemExit) as e:
        launcher_run.main()
    assert e.value.code == 2                      # argparse's usage error
    assert "--pin-commit" in capsys.readouterr().err


def test_c_the_flag_is_launcher_owned_and_never_forwarded():
    from main.launcher.checkpoint import _strip_launcher_args
    for argv in (["--pin-commit", "abc1234", "--steps", "5"],
                 ["--pin-commit=abc1234", "--steps", "5"],
                 ["--pin-to-hash", "abc1234", "--steps", "5"]):
        assert _strip_launcher_args(argv) == ["--steps", "5"]


# ---------------------------------------------------------------------------------------
# (d) a same-run RESTART may never MOVE the pin
# ---------------------------------------------------------------------------------------

def _run_with_checkpoint(tmp_path, git_hash, name="run_x"):
    """A run dir holding one checkpoint whose metadata records `git_hash`."""
    run_dir = tmp_path / "models" / name
    (run_dir / "checkpoints").mkdir(parents=True)
    ckpt = run_dir / "checkpoints" / "checkpoint_100_steps.zip"
    ckpt.write_text("")
    (run_dir / "metadata.json").write_text(json.dumps({"git_hash": git_hash}))
    return str(run_dir), str(ckpt)


def test_d_restart_with_a_mismatching_pin_commit_is_refused(repo, tmp_path):
    root, (first, second) = repo
    run_dir, ckpt = _run_with_checkpoint(tmp_path, first)
    with pytest.raises(wt.PinRefused) as e:
        wt.resolve_pin(model_path=ckpt, run_dir=run_dir, pin_commit=second,
                       sync_to_main=False, repo_root=root)
    msg = str(e.value)
    assert e.value.exit_code == int(TrainExitCode.FATAL_CONFIG)
    assert first in msg and second in msg, "the refusal must name BOTH commits"


def test_d_restart_with_a_matching_pin_commit_is_accepted(repo, tmp_path):
    root, (first, _second) = repo
    run_dir, ckpt = _run_with_checkpoint(tmp_path, first)
    got = wt.resolve_pin(model_path=ckpt, run_dir=run_dir, pin_commit=first,
                         sync_to_main=False, repo_root=root)
    assert (got.sha, got.source) == (first, "pin_commit")


def test_d_a_matching_short_prefix_is_the_same_commit_not_a_mismatch(repo, tmp_path):
    """The recorded hash may be short; a prefix naming the same commit must not refuse."""
    root, (first, _second) = repo
    run_dir, ckpt = _run_with_checkpoint(tmp_path, first[:8])
    got = wt.resolve_pin(model_path=ckpt, run_dir=run_dir, pin_commit=first[:12],
                         sync_to_main=False, repo_root=root)
    assert got.sha == first


def test_d_restart_without_a_pin_commit_uses_the_checkpoint_hash(repo, tmp_path):
    root, (first, _) = repo
    run_dir, ckpt = _run_with_checkpoint(tmp_path, first)
    got = wt.resolve_pin(model_path=ckpt, run_dir=run_dir, pin_commit=None,
                         sync_to_main=False, repo_root=root)
    assert (got.sha, got.source) == (first, "checkpoint")


# ---------------------------------------------------------------------------------------
# (e) a FORK / a FRESH run: --pin-commit WINS over the checkpoint hash and over HEAD
# ---------------------------------------------------------------------------------------

def test_e_fork_pin_commit_beats_the_parents_recorded_hash(repo, tmp_path):
    root, (first, second) = repo
    _parent_dir, parent_ckpt = _run_with_checkpoint(tmp_path, first, name="parent")
    fork_dir = str(tmp_path / "models" / "fork")     # a DIFFERENT run dir => a genuine fork
    got = wt.resolve_pin(model_path=parent_ckpt, run_dir=fork_dir, pin_commit=second,
                         sync_to_main=False, repo_root=root)
    assert (got.sha, got.source) == (second, "pin_commit")
    # …and with no --pin-commit that same fork would have taken the parent's hash.
    assert wt.resolve_pin(model_path=parent_ckpt, run_dir=fork_dir, pin_commit=None,
                          sync_to_main=False, repo_root=root).sha == first


def test_e_fresh_run_pin_commit_beats_head(repo, tmp_path, monkeypatch):
    root, (first, second) = repo
    monkeypatch.setattr(wt, "get_git_hash", lambda *a, **k: second)  # "HEAD"
    fresh = str(tmp_path / "models" / "fresh")
    got = wt.resolve_pin(model_path=None, run_dir=fresh, pin_commit=first,
                         sync_to_main=False, repo_root=root)
    assert (got.sha, got.source) == (first, "pin_commit")
    head = wt.resolve_pin(model_path=None, run_dir=fresh, pin_commit=None,
                          sync_to_main=False, repo_root=root)
    assert (head.sha, head.source) == (second, "head")


def test_e_sync_to_main_still_takes_head_and_says_so(repo, tmp_path, monkeypatch):
    root, (first, second) = repo
    monkeypatch.setattr(wt, "get_git_hash", lambda *a, **k: second)
    run_dir, ckpt = _run_with_checkpoint(tmp_path, first)
    got = wt.resolve_pin(model_path=ckpt, run_dir=run_dir, pin_commit=None,
                         sync_to_main=True, repo_root=root)
    assert (got.sha, got.source) == (second, "sync_to_main")


def test_e_a_resume_with_no_recorded_git_hash_still_exits_1_not_3(tmp_path):
    """Unchanged legacy behaviour: that refusal is its own thing and keeps its exit code."""
    run_dir = tmp_path / "models" / "run_y"
    (run_dir / "checkpoints").mkdir(parents=True)
    ckpt = run_dir / "checkpoints" / "checkpoint_1_steps.zip"
    ckpt.write_text("")
    with pytest.raises(wt.PinRefused) as e:
        wt.resolve_pin(model_path=str(ckpt), run_dir=str(run_dir), pin_commit=None,
                       sync_to_main=False)
    assert e.value.exit_code == 1


# ---------------------------------------------------------------------------------------
# (f) checkargs must recognise it as launcher-owned, not report it stale
# ---------------------------------------------------------------------------------------

def test_f_checkargs_does_not_flag_pin_commit_as_stale():
    from main.checkargs import check
    res = check(["--pin-commit", "0c76e2ee", "--steps", "1000"])
    assert "--pin-commit" in res["launcher_only"]
    assert res["unknown"] == []


def test_f_checkargs_also_knows_the_legacy_spelling():
    from main.checkargs import check
    assert check(["--pin-to-hash", "0c76e2ee", "--steps", "1000"])["unknown"] == []


# ---------------------------------------------------------------------------------------
# Provenance: the run must be able to say WHERE its pin came from.
# ---------------------------------------------------------------------------------------

def test_pin_source_is_handed_to_the_child_and_announced(repo, tmp_path, monkeypatch):
    root, (first, _second) = repo
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher_run, "get_repo_root", lambda *a, **k: root)
    monkeypatch.setattr(launcher_run, "_prune_stale_launcher_worktrees", lambda *a, **k: None)
    monkeypatch.setattr(
        launcher_run, "_create_run_worktree",
        lambda h, run_dir=None: ("/t/train.py", "/t/src", lambda: None))
    # _prepare_session registers at-exit summary printers for the REAL process; a unit test
    # must not leave those behind (they would fire, and print, at the end of the pytest run).
    monkeypatch.setattr(launcher_run.atexit, "register", lambda *a, **k: None)
    state = LauncherState(0.0)
    ctx = launcher_run._prepare_session(
        state, ["--steps", "1"], interval_hours=0.0, pin=True, sync_to_main=False,
        pin_commit=first[:8], grace_minutes=1.0, max_crash_restarts=0,
    )
    assert ctx.child_env["LAUNCHER_PIN_SOURCE"] == "pin_commit"
    assert ctx.child_env["LAUNCHER_GIT_HASH"] == first, "the FULL sha is what gets recorded"
    events = " ".join(state.snapshot().events)
    assert first in events and "commit 1: first" in events, \
        "startup must print the resolved full sha AND its subject line"


def test_metadata_records_pin_source_when_the_launcher_set_it(tmp_path, monkeypatch):
    """The env the launcher exports is what lands in metadata.json — and only then."""
    from agents.model.model_version import ModelVersion
    from agents.model.snapshot import save_model_snapshot
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    version = ModelVersion.from_layout_and_policy_kwargs(
        Gen3ObservationEncoder(load_mappings()).get_layout(), {"net_arch": [512, 512]})

    monkeypatch.setenv("LAUNCHER_PIN_SOURCE", "pin_commit")
    on = tmp_path / "on"
    on.mkdir()
    save_model_snapshot(str(on), version, git_hash="0" * 40)
    meta = json.loads((on / "metadata.json").read_text())
    assert meta["pin_source"] == "pin_commit"
    assert meta["git_hash"] == "0" * 40

    monkeypatch.delenv("LAUNCHER_PIN_SOURCE")
    off = tmp_path / "off"
    off.mkdir()
    save_model_snapshot(str(off), version, git_hash="0" * 40)
    assert "pin_source" not in json.loads((off / "metadata.json").read_text())
