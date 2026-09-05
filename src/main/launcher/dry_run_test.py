"""`--dry-run` — the launcher resolves a launch and touches NOTHING.

WHY THIS FILE EXISTS (2026-09-05). To validate a same-run RESTART with a larger `--steps`, a
session "dry launched" the real command and killed it after the startup lines. A FORK dry-launched
that way is harmless (it writes a NEW directory); a RESTART is not — those seconds wrote
`final_model_interrupted.zip`/`.json` into the LIVE run dir, repointed `latest.txt` at that phantom
artifact, overwrote `metadata.json` (its `steps` became a target that never ran) and
`model_config.json`, and left `.compile_quorum` files behind. "Dry" was a property of forks, never
of the launcher.

The central test here is therefore not "does it print the right thing" but **(a)**: sha256 + mtime
of every file in a fake run dir, before and after a same-run-restart dry run, must be IDENTICAL,
and `git worktree list` unchanged. A dry run that is merely *usually* harmless is the thing that
caused the incident.

Run: python -m pytest src/main/launcher/dry_run_test.py -q
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import hashlib
import importlib
import json
import os
import subprocess

import pytest

from main.exit_codes import TrainExitCode
import main.launcher.child as child_mod
import main.launcher.dry_run as dry_run_mod
import main.launcher.worktree as wt

launcher_run = importlib.import_module("main.launcher.run")


# ---------------------------------------------------------------------------------------
# A real (tiny) git repo — the pin resolution IS `git rev-parse`, so a fake would test
# nothing. Same shape as pin_commit_test.py's fixture, for the same reason.
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
    (root / "deps").mkdir()
    (root / "deps" / ".keep").write_text("")
    shas = []
    for n, text in ((1, "first"), (2, "second")):
        (root / "marker.txt").write_text(text)
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", f"commit {n}: {text}")
        shas.append(_git(root, "rev-parse", "HEAD"))
    return str(root), shas


@pytest.fixture
def isolated(repo, tmp_path, monkeypatch):
    """cwd in a scratch dir, every repo-root lookup pointed at the temp repo, and every
    effectful launcher entry point booby-trapped so a future edit that reaches one FAILS."""
    root, shas = repo
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(dry_run_mod, "get_repo_root", lambda *a, **k: root)
    monkeypatch.setattr(wt, "get_repo_root", lambda *a, **k: root)
    for name in ("_create_run_worktree", "_prune_stale_launcher_worktrees"):
        monkeypatch.setattr(
            wt, name, lambda *a, **k: pytest.fail(f"--dry-run must never call {name}"))
        if hasattr(launcher_run, name):
            monkeypatch.setattr(
                launcher_run, name,
                lambda *a, **k: pytest.fail(f"--dry-run must never call {name}"))
    monkeypatch.setattr(
        child_mod, "_launch_child",
        lambda *a, **k: pytest.fail("--dry-run must never spawn a child"))
    monkeypatch.setattr(
        launcher_run, "_apply_nice",
        lambda *a, **k: pytest.fail("--dry-run must not change this process's niceness"))
    return root, shas, work


def _make_run(root_dir, git_hash, name="ai_v9_171", steps=28_115_184, pool=0):
    """A fake run dir with the files a real one carries at the moment of a restart."""
    run_dir = root_dir / "models" / name
    (run_dir / "checkpoints").mkdir(parents=True)
    ckpt = run_dir / "checkpoints" / f"checkpoint_{steps}_steps.zip"
    ckpt.write_text("not-a-real-zip")
    (run_dir / "checkpoints" / f"checkpoint_{steps}_steps.json").write_text(
        json.dumps({"git_hash": git_hash, "num_timesteps": steps, "lr": 1e-4}))
    (run_dir / "metadata.json").write_text(json.dumps({"git_hash": git_hash}))
    (run_dir / "model_config.json").write_text(json.dumps({"obs_dim": 2501}))
    (run_dir / "latest.txt").write_text(f"checkpoints/checkpoint_{steps}_steps.zip")
    if pool:
        (run_dir / "snapshots").mkdir()
        for i in range(pool):
            (run_dir / "snapshots" / f"snapshot_{i}.zip").write_text("x")
        (run_dir / "snapshots" / "win_rate_vs_bots.txt").write_text("0.83")
    return str(run_dir), str(ckpt)


def _snapshot_tree(path):
    """{relpath: (sha256, mtime_ns)} for every file under `path` — the byte-identity probe."""
    out = {}
    for dirpath, _dirnames, filenames in os.walk(path):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            st = os.stat(full)
            with open(full, "rb") as f:
                out[os.path.relpath(full, path)] = (hashlib.sha256(f.read()).hexdigest(),
                                                    st.st_mtime_ns)
    return out


def _dry_run(argv, monkeypatch, expect=0):
    """Drive the REAL entry point (`main()`), so the strip / port-default / branch plumbing is
    covered too. Returns the captured stdout block."""
    monkeypatch.setattr("sys.argv", ["launcher", "--dry-run", *argv])
    with pytest.raises(SystemExit) as e:
        launcher_run.main()
    assert e.value.code == expect, f"expected exit {expect}, got {e.value.code}"
    return e


# ---------------------------------------------------------------------------------------
# (a) THE ONE THAT MATTERS: a same-run RESTART dry run leaves the run dir byte-identical
# ---------------------------------------------------------------------------------------

def test_a_same_run_restart_dry_run_leaves_the_run_dir_byte_identical(
        isolated, monkeypatch, capsys):
    root, (first, _second), work = isolated
    run_dir, ckpt = _make_run(work, first)
    before = _snapshot_tree(run_dir)
    assert before, "the fixture must actually have written files"

    _dry_run(["--model", ckpt, "--steps", "30000000"], monkeypatch)
    out = capsys.readouterr().out

    assert _snapshot_tree(run_dir) == before, (
        "a dry run of a same-run RESTART must not write, touch or delete ANYTHING in the "
        "live run dir — that is the whole incident this flag exists for")
    assert "RESTART of" in out and run_dir in out
    assert "would launch" in out


def test_a_same_run_restart_dry_run_creates_no_worktree(isolated, monkeypatch, capsys):
    root, (first, _second), work = isolated
    _run_dir, ckpt = _make_run(work, first)
    before = _git(root, "worktree", "list")
    _dry_run(["--model", ckpt, "--steps", "30000000"], monkeypatch)
    capsys.readouterr()
    assert _git(root, "worktree", "list") == before, "--dry-run must create no worktree"


def test_a_the_step_delta_is_printed_from_the_sidecar(isolated, monkeypatch, capsys):
    """The exact question the 2026-09-05 dry launch was asked: "+how many steps?"."""
    _root, (first, _second), work = isolated
    _run_dir, ckpt = _make_run(work, first, steps=28_115_184)
    _dry_run(["--model", ckpt, "--steps", "30000000"], monkeypatch)
    out = capsys.readouterr().out
    assert "28,115,184" in out and "sidecar" in out
    assert "+1,884,816 steps" in out


# ---------------------------------------------------------------------------------------
# (b) a FORK dry run creates no directory either
# ---------------------------------------------------------------------------------------

def test_b_fork_dry_run_creates_no_directory(isolated, monkeypatch, capsys):
    _root, (first, _second), work = isolated
    parent_dir, ckpt = _make_run(work, first, name="parent")
    parent_before = _snapshot_tree(parent_dir)

    _dry_run(["--model", ckpt, "--run-name", "fork_x", "--steps", "1000"], monkeypatch)
    out = capsys.readouterr().out

    assert not os.path.exists(work / "models" / "fork_x"), \
        "a dry run must not create the fork's run dir"
    assert _snapshot_tree(parent_dir) == parent_before, "and must not touch the fork PARENT"
    assert "FORK of" in out and parent_dir in out
    assert "would be created" in out


# ---------------------------------------------------------------------------------------
# (c) a FRESH run: role FRESH, pinned to HEAD, and nothing under models/
# ---------------------------------------------------------------------------------------

def test_c_fresh_dry_run_prints_role_fresh_and_a_head_pin(isolated, monkeypatch, capsys):
    _root, (_first, second), work = isolated
    monkeypatch.setattr(wt, "get_git_hash", lambda *a, **k: second)
    _dry_run(["--steps", "1000"], monkeypatch)
    out = capsys.readouterr().out
    assert "role        : FRESH" in out
    assert second in out and "(source: head)" in out
    assert "commit 2: second" in out, "the pin's SUBJECT is what makes the sha readable"
    assert not os.path.exists(work / "models"), "a dry run must not mint the fresh run dir"


def test_c_no_pin_says_so_and_still_resolves_everything_else(isolated, monkeypatch, capsys):
    _root, (_first, _second), _work = isolated
    _dry_run(["--no-pin", "--steps", "1000"], monkeypatch)
    out = capsys.readouterr().out
    assert "--no-pin" in out and "role        : FRESH" in out


# ---------------------------------------------------------------------------------------
# (d) the refusals — same ones the real path makes, same FATAL_CONFIG exit code
# ---------------------------------------------------------------------------------------

def test_d_unresolvable_pin_commit_exits_fatal_config(isolated, monkeypatch, capsys):
    _root, _shas, _work = isolated
    _dry_run(["--pin-commit", "deadbeefdeadbeef", "--steps", "1000"],
             monkeypatch, expect=int(TrainExitCode.FATAL_CONFIG))
    out = capsys.readouterr().out
    assert "REFUSED (pin)" in out and "deadbeef" in out


def test_d_a_restart_that_would_move_the_pin_exits_fatal_config(isolated, monkeypatch, capsys):
    """The refusal that protects a live run from being walked onto other code."""
    _root, (first, second), work = isolated
    _run_dir, ckpt = _make_run(work, first)
    _dry_run(["--model", ckpt, "--pin-commit", second, "--steps", "1000"],
             monkeypatch, expect=int(TrainExitCode.FATAL_CONFIG))
    out = capsys.readouterr().out
    assert first in out and second in out, "the refusal must name BOTH commits"


def test_d_a_pin_commit_that_matches_the_restart_is_accepted_and_printed(
        isolated, monkeypatch, capsys):
    _root, (first, _second), work = isolated
    _run_dir, ckpt = _make_run(work, first)
    _dry_run(["--model", ckpt, "--pin-commit", first, "--steps", "1000"], monkeypatch)
    out = capsys.readouterr().out
    assert "(source: pin_commit)" in out and "commit 1: first" in out


def test_d_an_unknown_flag_exits_fatal_config_and_names_it(isolated, monkeypatch, capsys):
    _root, _shas, _work = isolated
    _dry_run(["--steps", "1000", "--pubval-mode", "none"],
             monkeypatch, expect=int(TrainExitCode.FATAL_CONFIG))
    out = capsys.readouterr().out
    assert "--pubval-mode" in out and "would NOT launch" in out


def test_d_pin_commit_with_sync_to_main_is_still_refused_at_parse_time(isolated, monkeypatch):
    """--dry-run must not become a way around a refusal the parser owns."""
    _root, (first, _second), _work = isolated
    monkeypatch.setattr("sys.argv",
                        ["launcher", "--dry-run", "--pin-commit", first, "--sync-to-main"])
    with pytest.raises(SystemExit) as e:
        launcher_run.main()
    assert e.value.code == 2                        # argparse's usage error


# ---------------------------------------------------------------------------------------
# (e) the pool line — drift is invisible until the run is already training on it
# ---------------------------------------------------------------------------------------

def test_e_the_pool_line_appears_when_snapshots_exist(isolated, monkeypatch, capsys):
    _root, (first, _second), work = isolated
    _run_dir, ckpt = _make_run(work, first, pool=3)
    _dry_run(["--model", ckpt, "--steps", "1000"], monkeypatch)
    out = capsys.readouterr().out
    assert "3 snapshot(s)" in out and "win_rate_vs_bots 0.830" in out


def test_e_an_empty_pool_directory_says_what_that_means(isolated, monkeypatch, capsys):
    _root, (first, _second), work = isolated
    run_dir, ckpt = _make_run(work, first)
    os.makedirs(os.path.join(run_dir, "snapshots"))
    _dry_run(["--model", ckpt, "--steps", "1000"], monkeypatch)
    out = capsys.readouterr().out
    assert "0 snapshot(s)" in out and "self_play_fraction=0%" in out


# ---------------------------------------------------------------------------------------
# (f) checkargs must recognise it as launcher-owned, and it must never be forwarded
# ---------------------------------------------------------------------------------------

def test_f_checkargs_does_not_flag_dry_run_as_stale():
    from main.checkargs import check
    res = check(["--dry-run", "--steps", "1000"])
    assert "--dry-run" in res["launcher_only"]
    assert res["unknown"] == []


def test_f_the_flag_is_stripped_and_never_reaches_the_child():
    from main.launcher.checkpoint import _strip_launcher_args
    assert _strip_launcher_args(["--dry-run", "--steps", "5"]) == ["--steps", "5"]


def test_f_the_parser_knows_it_and_it_defaults_off():
    parser = launcher_run.build_launcher_parser()
    assert parser.parse_known_args(["--steps", "1"])[0].dry_run is False
    assert parser.parse_known_args(["--dry-run", "--steps", "1"])[0].dry_run is True


# ---------------------------------------------------------------------------------------
# The gaps are DECLARED, not guessed at.
# ---------------------------------------------------------------------------------------

def test_the_child_only_gaps_are_printed(isolated, monkeypatch, capsys):
    _root, _shas, _work = isolated
    _dry_run(["--steps", "1000"], monkeypatch)
    out = capsys.readouterr().out
    assert out.count("(child-only:") == len(dry_run_mod.CHILD_ONLY)
    assert "check_compatible" in out, "the arch check is the gap a reader most needs named"
