"""A PINNED run reads ``data/`` AS OF ITS PIN, not the main checkout's (``gen3_pinned_data_root_v1``).

The launcher spawns its child with ``PYTHONPATH=<pinned worktree>/src`` and NO ``cwd=`` (`child._launch_child`), so the
child imports its pin's code while STANDING IN the launching (main) checkout. A ``data/`` reader resolved against the
CWD therefore read MAIN's files in a pinned run: the 2026-09-23 team-split revert (`e74c0610`) and root rule 6's
``data/`` freeze. Every runtime ``data/`` reader now resolves through ``utils.paths.repo_path`` (the checkout the code
was IMPORTED from) — for a pinned child, the pin's own worktree, which carries ``data/`` because it is all tracked.

These tests run the REAL launcher path end to end over a throwaway git repo: ``worktree._create_run_worktree`` pins a
commit, MAIN's ``data/`` then moves on, and ``child._launch_child`` spawns a probe (never the trainer script — its name
in a live argv trips the run watchers) with cwd = main. The fake tree is built from THIS checkout's own modules, so a
revert of any reader to a CWD-relative path fails here.

Run: python -m pytest src/main/launcher/pinned_data_test.py -q
(in a linked worktree, first: export PYTHONPATH=$PWD/src)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import main.launcher.worktree as wt
from main.launcher.child import _build_child_env, _launch_child
from main.launcher.state import LauncherState
from utils import showdown_deps
from utils.paths import repo_root, src_path

#: The modules the probe imports, copied from THIS checkout into the fake repo (so the test judges today's code).
_COPIED = ("utils/__init__.py", "utils/paths.py", "utils/team_loader/__init__.py", "utils/team_loader/loader.py",
           "utils/team_corpora/__init__.py", "agents/__init__.py", "agents/training/elo.py")

_PROBE = """
import json, os, sys
from utils.team_loader import TeamLoader
from agents.training import elo
import utils.paths as P
loader = TeamLoader()
json.dump({"teams": loader.get_all_teams(), "sample": loader.get_sample_teams(),
           "anchors": elo.load_bot_anchors(), "anchors_path": elo.BOT_ANCHORS_PATH,
           "repo_root": str(P.repo_root()), "cwd": os.getcwd()}, open(sys.argv[1], "w"))
"""


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _write_data(root: Path, tag: str) -> None:
    sample = root / "data" / "teams" / "sample"
    sample.mkdir(parents=True, exist_ok=True)
    (sample / "t1.txt").write_text(f"{tag} TEAM\n")
    (sample / "teams.json").write_text(json.dumps([{"file": "teams/sample/t1.txt"}]))
    (root / "data" / "gen3_bot_elo_anchors.json").write_text(json.dumps({"ratings": {f"bot:{tag}": 1000.0}}))


@pytest.fixture
def repo(tmp_path):
    """A 2-commit repo: commit 1 (the PIN) carries ``PIN`` data, commit 2 (main's HEAD) ``MAIN`` data, and main's
    working tree additionally holds an UNCOMMITTED edit — the two ways main's ``data/`` moves under a live pin."""
    root = tmp_path / "main_checkout"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    for rel in _COPIED:
        dst = root / "src" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path(*rel.split("/")), dst)
    (root / "deps").mkdir()
    (root / "deps" / ".keep").write_text("")
    _write_data(root, "PIN")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "the pin")
    pin = _git(root, "rev-parse", "HEAD")
    _write_data(root, "MAIN")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "main moves on")
    (root / "data" / "teams" / "sample" / "t1.txt").write_text("UNCOMMITTED MAIN TEAM\n")
    showdown_deps.write_stub_checkout(str(root))      # untracked; the worktree's submodule link points here
    return root, pin


def _run_pinned_probe(root: Path, pin: str, tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.setattr(wt, "get_repo_root", lambda *a, **k: str(root))
    monkeypatch.chdir(root)                            # the launcher's cwd = the MAIN checkout; the spawn passes no cwd=
    _train_script, src_dir, cleanup = wt._create_run_worktree(pin)
    try:
        probe, out = tmp_path / "probe.py", tmp_path / "probe_out.json"
        probe.write_text(_PROBE)
        env = {k: v for k, v in _build_child_env().items() if k != "PYTHONPATH"}   # only the pin's src, as in a run
        state = LauncherState(interval_hours=0.0)
        state.run_dir = str(tmp_path)
        proc = _launch_child([str(out)], env, state, str(probe), src_dir)
        assert proc.wait(timeout=120) == 0, "\n".join(state.snapshot().log_lines)
        for _ in range(100):                           # the stdout reader thread drains after the exit
            if out.exists():
                break
            time.sleep(0.05)
        got = json.loads(out.read_text())
        got["worktree"] = os.path.dirname(src_dir)
        return got
    finally:
        cleanup()


def test_a_pinned_child_reads_its_PINS_data_while_mains_data_differs(repo, tmp_path, monkeypatch):
    root, pin = repo
    got = _run_pinned_probe(root, pin, tmp_path, monkeypatch)
    assert os.path.realpath(got["cwd"]) == os.path.realpath(root), "precondition: the child stands in MAIN"
    assert os.path.realpath(got["repo_root"]) == os.path.realpath(got["worktree"]), "precondition: it imports the pin"
    assert got["teams"] == ["PIN TEAM"] and got["sample"] == ["PIN TEAM"], (
        f"the pinned child's team pool is not its pin's: {got['teams']} — a CWD-relative TeamLoader default reads "
        "the MAIN checkout's data/teams (e74c0610)")
    assert got["anchors"] == {"ratings": {"bot:PIN": 1000.0}}, (
        f"the pinned child's ELO anchors are not its pin's: {got['anchors']} ({got['anchors_path']})")


def test_the_three_teams_and_anchor_artifacts_are_this_checkouts_files():
    """The module-level paths a trainer reads (the eval ELO anchors, the archetype and species-prior artifacts) are
    ABSOLUTE and inside the imported checkout — never a CWD-relative ``data/…``."""
    from agents.training import elo, species_priors, team_archetypes

    for path in (elo.BOT_ANCHORS_PATH, species_priors.ARTIFACT_PATH, team_archetypes.ARTIFACT_PATH):
        assert os.path.isabs(path) and Path(path).is_relative_to(repo_root() / "data"), path
        assert os.path.exists(path), f"{path} is missing from this checkout"
