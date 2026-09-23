"""The worktree-removal guard, at its BLOCKED state — and ``scripts/land.sh`` driven end to end.

🚨 THE INCIDENT (ledger 2026-09-23). Early launcher worktrees held their run directories INSIDE
the worktree behind a symlink in the main checkout's `models/`; `models/` is gitignored, so a clean
`git status` hid them and `git worktree remove --force` destroyed eight runs. A guard that does not
fire is worse than none, so every refusal below is exercised on a real repo with a real worktree,
and the `land.sh` cases run the actual script (push to a local bare origin included) and assert the
worktree and its data SURVIVE. Each land.sh refusal test FAILS on the pre-guard land.sh: that
script removed the worktree unconditionally.

Cheap (a few seconds): the only "large" file is SPARSE — the guard counts apparent size, which is
what a removal would lose.

Run: python -m pytest src/utils/worktree_guard_test.py -q
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from utils import worktree_guard as guard
from utils.paths import repo_path

MIB = 1024 * 1024

#: git isolated from the box's global/system config (hooks, signing, templates).
_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
}


def _env(**extra):
    env = {**os.environ, **_GIT_ENV, **extra}
    env.pop("GEN3AI_MODELS_DIR", None)
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(k, None)
    return env


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, env=_env(), capture_output=True, text=True,
                          check=True).stdout.strip()


def _sparse(path, size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.truncate(size)


@pytest.fixture
def repos(tmp_path):
    """A bare ``origin``, its MAIN clone carrying land.sh + the guard, and a linked worktree on
    branch ``feat`` one commit ahead. Returns ``(main, wt)`` as absolute strings."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    main = tmp_path / "main"
    _git(tmp_path, "clone", "-q", str(origin), str(main))
    _git(main, "checkout", "-q", "-b", "main")
    (main / "scripts").mkdir()
    shutil.copy(repo_path("scripts", "land.sh"), main / "scripts" / "land.sh")
    (main / "src" / "utils").mkdir(parents=True)
    (main / "src" / "utils" / "__init__.py").write_text("")
    shutil.copy(repo_path("src", "utils", "worktree_guard.py"),
                main / "src" / "utils" / "worktree_guard.py")
    (main / ".gitignore").write_text("models/\n*.bin\n__pycache__/\nsrc/rust_sim/target/\n")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "-m", "base")
    _git(main, "push", "-q", "-u", "origin", "main")
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", "-b", "feat", str(wt))
    (wt / "feature.txt").write_text("the work")
    _git(wt, "add", "feature.txt")
    _git(wt, "commit", "-q", "-m", "feature")
    return str(main), str(wt)


def _plant_incident(main, wt):
    """The 2026-09-23 shape: a run dir INSIDE the worktree, a symlink to it in main's models/."""
    run = os.path.join(wt, "models", "ai_v9_01_gen1")
    os.makedirs(os.path.join(run, "checkpoints"))
    with open(os.path.join(run, "checkpoints", "checkpoint_100_steps.zip"), "wb") as fh:
        fh.write(b"weights")                      # SMALL: rule (a) must fire with no size help
    os.makedirs(os.path.join(main, "models"), exist_ok=True)
    link = os.path.join(main, "models", "ai_v9_01_gen1")
    os.symlink(run, link)
    return run, link


# ------------------------------------------------------------------ the guard itself

def test_a_models_symlink_into_the_worktree_is_a_hazard_at_any_size(repos):
    main, wt = repos
    _run, link = _plant_incident(main, wt)
    hz = guard.find_hazards(main, wt)
    assert [h.kind for h in hz] == ["models_symlink"]
    assert hz[0].path == link


def test_a_symlink_one_level_into_a_real_run_dir_is_found(repos):
    main, wt = repos
    os.makedirs(os.path.join(wt, "stash"))
    os.makedirs(os.path.join(main, "models", "real_run"))
    os.symlink(os.path.join(wt, "stash"), os.path.join(main, "models", "real_run", "checkpoints"))
    assert [h.kind for h in guard.find_hazards(main, wt)] == ["models_symlink"]


def test_the_models_dir_override_is_scanned_too(repos, tmp_path, monkeypatch):
    main, wt = repos
    archive = tmp_path / "archive"
    archive.mkdir()
    os.makedirs(os.path.join(wt, "models", "r"))
    os.symlink(os.path.join(wt, "models", "r"), archive / "r")
    assert guard.find_hazards(main, wt) == []
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(archive))
    assert [h.kind for h in guard.find_hazards(main, wt)] == ["models_symlink"]


def test_a_symlink_elsewhere_is_not_a_hazard(repos, tmp_path):
    main, wt = repos
    os.makedirs(os.path.join(main, "models"))
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", os.path.join(main, "models", "fine"))
    assert guard.find_hazards(main, wt) == []


def test_untracked_data_above_the_threshold_is_refused_and_named(repos):
    main, wt = repos
    _sparse(os.path.join(wt, "scratch", "dump.bin"), 60 * MIB)
    hz = guard.find_hazards(main, wt)
    assert [(h.kind, h.path) for h in hz] == [("untracked_data", "scratch/")]
    assert hz[0].size == 60 * MIB
    text = guard.report(hz, wt)
    assert "REFUSING" in text and "scratch/" in text and "60.0 MiB" in text


def test_the_threshold_is_a_TOTAL_not_a_per_file_bound(repos):
    main, wt = repos
    for i in range(3):
        _sparse(os.path.join(wt, f"part{i}.bin"), 20 * MIB)     # 60 MiB across three entries
    hz = guard.find_hazards(main, wt)
    assert sorted(h.path for h in hz) == ["part0.bin", "part1.bin", "part2.bin"]


def test_untracked_data_at_or_below_the_threshold_passes(repos):
    main, wt = repos
    _sparse(os.path.join(wt, "scratch", "dump.bin"), 50 * MIB)
    assert guard.find_hazards(main, wt) == []


def test_allowlisted_build_artifacts_never_count(repos):
    main, wt = repos
    _sparse(os.path.join(wt, "src", "rust_sim", "target", "release", "sim_bridge"), 400 * MIB)
    _sparse(os.path.join(wt, "src", "agents", "__pycache__", "x.cpython-311.pyc"), 80 * MIB)
    _sparse(os.path.join(wt, ".mypy_cache", "blob"), 80 * MIB)
    assert guard.find_hazards(main, wt) == []


@pytest.mark.parametrize("rel,benign", [
    ("src/rust_sim/target/", True), ("src/rust_sim/Cargo.lock", True),
    ("deps/pokemon-showdown/dist", True), ("src/a/__pycache__/", True), (".ruff_cache/", True),
    ("x.egg-info/", True), ("models/", False), ("tmp/", False), ("harvest/", False),
    ("src/rust_sim/targetx", False), ("scratch/big.bin", False),
])
def test_allowlist_classification(rel, benign):
    assert guard.is_benign(rel) is benign


def test_the_main_checkout_itself_is_refused(repos):
    main, _wt = repos
    assert [h.kind for h in guard.find_hazards(main, main)] == ["is_main_checkout"]


def test_a_tree_git_cannot_read_fails_CLOSED(tmp_path):
    (tmp_path / "main").mkdir()
    (tmp_path / "notgit").mkdir()
    with pytest.raises(guard.GuardError):
        guard.find_hazards(str(tmp_path / "main"), str(tmp_path / "notgit"))
    rc = guard.main(["--main", str(tmp_path / "main"), str(tmp_path / "notgit")])
    assert rc == guard.EXIT_ERROR


def test_cli_exit_codes(repos):
    main, wt = repos
    assert guard.main(["--main", main, wt]) == guard.EXIT_SAFE
    _plant_incident(main, wt)
    assert guard.main(["--main", main, wt]) == guard.EXIT_REFUSED


# ------------------------------------------------------------------ scripts/land.sh, end to end

def _fake_python(tmp_path):
    """An interpreter for land.sh whose GATES (ruff / mypy / pytest over a tree that has no src
    worth gating) are no-ops, and which runs everything else — the guard — for real."""
    path = tmp_path / "fakepy"
    path.write_text("#!/bin/sh\ncase \"$2\" in ruff|mypy|pytest) exit 0 ;; esac\n"
                    f"exec {sys.executable} \"$@\"\n")
    path.chmod(0o755)
    return str(path)


def _land(tmp_path, main, wt):
    """Run the WORKTREE's own copy of land.sh — the production path, re-exec included."""
    return subprocess.run(
        ["bash", os.path.join(wt, "scripts", "land.sh"), "feat", wt],
        cwd=wt, env=_env(GEN3AI_PYTHON=_fake_python(tmp_path), PYTHONPATH=""),
        capture_output=True, text=True, timeout=120)


def _origin_main(main):
    return _git(main, "ls-remote", "origin", "refs/heads/main").split()[0]


def test_land_REFUSES_to_remove_a_worktree_a_models_symlink_points_into(repos, tmp_path):
    main, wt = repos
    run, link = _plant_incident(main, wt)
    feat = _git(wt, "rev-parse", "HEAD")
    proc = _land(tmp_path, main, wt)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "REFUSING" in proc.stderr and link in proc.stderr
    assert "worktree was NOT removed" in proc.stderr
    # the run data, the link to it, the worktree and its branch all SURVIVE
    assert os.path.isfile(os.path.join(run, "checkpoints", "checkpoint_100_steps.zip"))
    assert os.path.exists(link)
    assert wt in _git(main, "worktree", "list")
    assert _git(main, "rev-parse", "--verify", "refs/heads/feat") == feat
    # ... and the code DID land: the refusal is about the data, not the commit
    assert _origin_main(main) == feat


def test_land_REFUSES_a_worktree_holding_large_non_build_ignored_data(repos, tmp_path):
    main, wt = repos
    blob = os.path.join(wt, "runs_scratch", "replay_buffer.bin")
    _sparse(blob, 60 * MIB)
    proc = _land(tmp_path, main, wt)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "runs_scratch/" in proc.stderr and "untracked_data" in proc.stderr
    assert os.path.getsize(blob) == 60 * MIB
    assert wt in _git(main, "worktree", "list")


def test_land_proceeds_on_a_clean_worktree_whose_bulk_is_build_artifacts(repos, tmp_path):
    main, wt = repos
    _sparse(os.path.join(wt, "src", "rust_sim", "target", "release", "sim_bridge"), 300 * MIB)
    _sparse(os.path.join(wt, "src", "utils", "__pycache__", "x.cpython-311.pyc"), 1 * MIB)
    feat = _git(wt, "rev-parse", "HEAD")
    proc = _land(tmp_path, main, wt)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not os.path.exists(wt)
    assert wt not in _git(main, "worktree", "list")
    assert _origin_main(main) == feat
    assert subprocess.run(["git", "rev-parse", "--verify", "refs/heads/feat"], cwd=main,
                          env=_env(), capture_output=True).returncode != 0


def test_land_fails_CLOSED_when_the_guard_cannot_run(repos, tmp_path):
    """A guard that cannot run is a refusal — never a removal."""
    main, wt = repos
    os.remove(os.path.join(main, "src", "utils", "worktree_guard.py"))
    proc = _land(tmp_path, main, wt)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert os.path.isdir(wt)
