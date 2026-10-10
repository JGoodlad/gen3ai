"""The launcher's SUBMODULE PREFLIGHT (`main/launcher/submodule_gate.py`, `utils/showdown_deps.py`).

THE INCIDENT (2026-10-09, `designs/research_state/measurements/gpu_checks_endstate_2026-10-09/` F-GE-4). The pinned
worktree's `deps/pokemon-showdown` is a LINK to the LAUNCHING checkout's submodule. A launch from an agent worktree
whose submodule was never initialised linked an empty placeholder, and the run died minutes later - after the model and
the inference service were up - in team validation ("Cannot find module .../deps/pokemon-showdown"). The first P attempt
of the end-state GPU check died this way.

What is pinned here: the launcher REFUSES (`FATAL_CONFIG`, the fix named) BEFORE the pin, the worktree and the run dir
exist; `--dry-run` prints the same verdict and fails; `--no-pin` asks about its own tree; and the directory the preflight
checks is the directory the worktree is actually LINKED to (one definition: `worktree.showdown_link_source`).
`utils/showdown_deps_test.py` pins the file list itself.
"""
import importlib
import os
import subprocess

import pytest

import main.launcher.dry_run as dry_run_mod
import main.launcher.submodule_gate as gate
import main.launcher.worktree as wt
from main.exit_codes import TrainExitCode
from main.launcher.state import LauncherState
from utils import showdown_deps as sd

launcher_run = importlib.import_module("main.launcher.run")

#: Fresh-run consent flags, so the arch / recipe surface guards (not this file's subject) stay out of the way.
_FRESH = ["--steps", "1", "--allow-nonproduction-arch", "--allow-nonproduction-recipe"]


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _make_repo(tmp_path, name, *, state):
    """A 1-commit throwaway repo whose `deps/pokemon-showdown` is in `state`:
    `placeholder` (an EMPTY directory - what `git worktree add` / a never-initialised submodule leaves),
    `unbuilt` (checked out, no dist/ or node_modules/), or `ok`."""
    root = tmp_path / name
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "deps").mkdir()
    (root / "deps" / ".keep").write_text("")
    (root / "marker.txt").write_text("x")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "c1")
    sha = _git(root, "rev-parse", "HEAD")
    ps = root / "deps" / "pokemon-showdown"
    if state == "placeholder":
        ps.mkdir()
    elif state == "unbuilt":
        sd.write_stub_checkout(str(root))
        for n in sd.NEEDS:
            if n.kind != sd.KIND_CHECKOUT:
                os.remove(ps.joinpath(*n.rel.split("/")))
    elif state == "ok":
        sd.write_stub_checkout(str(root))
    else:
        raise AssertionError(state)
    return str(root), sha


def _session(root, sha, monkeypatch, tmp_path, *, pin=True, created=None, argv=None, sink=None):
    """`_prepare_session` against the throwaway repo, with worktree creation observable."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher_run, "get_repo_root", lambda *a, **k: root)
    monkeypatch.setattr(launcher_run, "_prune_stale_launcher_worktrees", lambda *a, **k: None)

    def _fake_create(h, run_dir=None):
        if created is not None:
            created.append(h)
        return ("/t/train.py", "/t/src", lambda: None)

    monkeypatch.setattr(launcher_run, "_create_run_worktree", _fake_create)
    monkeypatch.setattr(launcher_run.atexit, "register", lambda *a, **k: None)
    state = LauncherState(0.0)
    state.event_sink = sink
    ctx = launcher_run._prepare_session(
        state, list(argv or _FRESH), interval_hours=0.0, pin=pin, sync_to_main=False,
        pin_commit=sha[:8] if pin else None, grace_minutes=1.0, max_crash_restarts=0)
    return ctx, state


# ---------------------------------------------------------------------------------------
# the launch is REFUSED, before anything exists
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("state", ["placeholder", "unbuilt"])
def test_a_pinned_launch_from_an_unusable_checkout_is_refused_FATAL_CONFIG_and_creates_nothing(
        state, tmp_path, run_archive, monkeypatch, capsys):
    root, sha = _make_repo(tmp_path, "repo", state=state)
    worktrees_before = _git(root, "worktree", "list")
    created = []
    with pytest.raises(SystemExit) as e:
        _session(root, sha, monkeypatch, tmp_path, created=created)
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    err = capsys.readouterr().err
    assert "showdown deps : ✗ REFUSED" in err
    assert ("package.json" if state == "placeholder" else "dist/sim/index.js") in err, "name what is missing"
    assert root in err, "the refusal must name the checkout the fix applies to"
    assert "./scripts/bootstrap.sh --skip-env" in err
    if state == "placeholder":
        assert "git submodule update --init" in err
    else:
        assert "git submodule update --init" not in err, "an initialised-but-unbuilt checkout needs a build, not a re-init"
        assert "node build" in err
    assert created == [], "a refusal must come BEFORE the worktree"
    assert list(run_archive.iterdir()) == [], "...and before the run dir"
    assert _git(root, "worktree", "list") == worktrees_before


def test_a_healthy_checkout_launches_and_says_so(tmp_path, run_archive, monkeypatch):
    root, sha = _make_repo(tmp_path, "repo", state="ok")
    created = []
    heard = []                                  # the state keeps only the last 30 events; the sink hears them all
    _ctx, state = _session(root, sha, monkeypatch, tmp_path, created=created, sink=heard.append)
    assert created == [sha], "the pinned worktree is created for a usable checkout"
    assert any("showdown deps : ✓" in ev for ev in heard), "the verdict line must reach the event sink"


def test_a_no_pin_launch_asks_about_THIS_tree_not_the_repo_root(tmp_path, run_archive, monkeypatch, capsys):
    """Under --no-pin the child imports and reads the launching tree itself (no worktree, no link), so the preflight
    asks `dirname(_SRC_DIR)/deps/pokemon-showdown`."""
    this_tree = tmp_path / "tree"
    (this_tree / "src").mkdir(parents=True)
    (this_tree / "deps" / "pokemon-showdown").mkdir(parents=True)          # the empty placeholder
    root, sha = _make_repo(tmp_path, "repo", state="ok")                  # the repo root is healthy and must NOT save it
    monkeypatch.setattr(gate, "_SRC_DIR", str(this_tree / "src"))
    with pytest.raises(SystemExit) as e:
        _session(root, sha, monkeypatch, tmp_path, pin=False)
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    assert str(this_tree) in capsys.readouterr().err
    assert gate.verdict_for_launch(pin=False).refused
    sd.write_stub_checkout(str(this_tree))
    assert not gate.verdict_for_launch(pin=False).refused


# ---------------------------------------------------------------------------------------
# --dry-run prints the same verdict and fails the dry run
# ---------------------------------------------------------------------------------------

def _dry(root, monkeypatch, tmp_path, lines):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dry_run_mod, "get_repo_root", lambda *a, **k: root)
    return dry_run_mod.dry_run(
        list(_FRESH), interval_hours=0.0, pin=True, sync_to_main=False, pin_commit=None, grace_minutes=1.0,
        max_crash_restarts=0, nice=0, out=lines.append)


def test_dry_run_refuses_an_unusable_checkout_and_prints_the_fix(tmp_path, run_archive, monkeypatch):
    root, _sha = _make_repo(tmp_path, "repo", state="placeholder")
    lines = []
    code = _dry(root, monkeypatch, tmp_path, lines)
    text = "\n".join(lines)
    assert code == int(TrainExitCode.FATAL_CONFIG)
    assert "showdown deps : ✗ REFUSED" in text and "git submodule update --init" in text
    assert "would NOT launch" in text


def test_dry_run_passes_a_usable_checkout_and_prints_the_ok_line(tmp_path, run_archive, monkeypatch):
    root, _sha = _make_repo(tmp_path, "repo", state="ok")
    lines = []
    code = _dry(root, monkeypatch, tmp_path, lines)
    text = "\n".join(lines)
    assert code == 0, text
    assert "showdown deps : ✓" in text and "would launch" in text


# ---------------------------------------------------------------------------------------
# the preflight checks the directory the worktree is actually LINKED to
# ---------------------------------------------------------------------------------------

def test_the_checked_directory_is_the_one_the_pinned_worktree_is_linked_to(tmp_path, monkeypatch):
    """ONE definition (`worktree.showdown_link_source`): create a REAL pinned worktree from a checkout whose submodule is
    usable and require that what the child would see - `<worktree>/deps/pokemon-showdown` - passes the same check."""
    root, sha = _make_repo(tmp_path, "repo", state="ok")
    monkeypatch.setattr(wt, "get_repo_root", lambda *a, **k: root)
    _train, src_dir, cleanup = wt._create_run_worktree(sha)
    try:
        linked = os.path.join(os.path.dirname(src_dir), "deps", "pokemon-showdown")
        assert os.path.islink(linked)
        assert os.path.realpath(linked) == os.path.realpath(wt.showdown_link_source(root))
        assert os.path.realpath(linked) == os.path.realpath(gate.source_for_launch(True, root))
        assert not sd.check(linked, which=lambda _n: "/usr/bin/node").refused, \
            "the preflight said OK for a root whose link the child cannot use"
    finally:
        cleanup()


def test_the_real_checkout_this_test_runs_in_passes_the_launch_preflight(monkeypatch):
    """The conftest guard already refuses a session without the submodule; this pins that the PREFLIGHT agrees, so a
    healthy dev checkout is never refused by it (a phantom path in NEEDS would show here first)."""
    from utils.git import get_repo_root
    v = gate.verdict_for_launch(pin=True, repo_root=get_repo_root())
    assert not v.refused, "\n".join(v.lines())
