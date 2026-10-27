"""The records-only ship against REAL git: a bare origin, a main clone, a submodule, the real push and worktree guards.

Only the static tier is a stand-in (it asserts the submodule is checked out, which the real freshness gate needs).

``integration``: git subprocesses in temp dirs. ~3 s in all.
"""
from __future__ import annotations

import fcntl
import os
import subprocess
from typing import Tuple

import pytest

from main.ops.arm_chain.launch import Hold
from main.ops.arm_chain.ship import Shipper

pytestmark = pytest.mark.integration

STAGE = "designs/stage"


def git(cwd: str, *a: str) -> str:
    return subprocess.run(["git", "-C", cwd, *a], check=True, capture_output=True, text=True).stdout


def _ident(path: str) -> None:
    git(path, "config", "user.email", "t@example.com")
    git(path, "config", "user.name", "t")


@pytest.fixture
def world(tmp_path, monkeypatch):
    # a local submodule needs the file protocol (the real one is https)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "protocol.file.allow")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "always")
    sub = tmp_path / "subsrc"
    sub.mkdir()
    git(str(sub), "init", "-q", "-b", "main")
    _ident(str(sub))
    (sub / "validator.js").write_text("ok\n")
    git(str(sub), "add", "-A")
    git(str(sub), "commit", "-q", "-m", "s")
    seed = tmp_path / "seed"
    seed.mkdir()
    git(str(seed), "init", "-q", "-b", "main")
    _ident(str(seed))
    (seed / STAGE / "k9b").mkdir(parents=True)
    (seed / STAGE / "k9b" / "old.json").write_text("{}\n")
    git(str(seed), "submodule", "add", "-q", str(sub), "deps/sub")
    git(str(seed), "add", "-A")
    git(str(seed), "commit", "-q", "-m", "seed")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(seed), str(origin)], check=True)
    main = tmp_path / "main"
    subprocess.run(["git", "clone", "-q", str(origin), str(main)], check=True)
    _ident(str(main))
    src = tmp_path / "scratch"
    (src / "k9b").mkdir(parents=True)
    (src / "k9b" / "rb_a_s1.json").write_text('{"verdict": "PASS"}\n')
    (src / "dose.json").write_text('{"run": "rb_a_s1"}\n')
    files = {f"{STAGE}/k9b/rb_a_s1.json": str(src / "k9b" / "rb_a_s1.json"),
             f"{STAGE}/dose/rb_a_s1.json": str(src / "dose.json")}
    return {"main": str(main), "origin": str(origin), "files": files, "tmp": tmp_path}


def shipper(world, static=None, timeout=5.0) -> Shipper:
    seen = {}

    def tier(wt: str) -> Tuple[int, str]:
        seen["submodule"] = os.path.isfile(os.path.join(wt, "deps", "sub", "validator.js"))
        return (static(wt) if static else 0), "fake static tier"
    s = Shipper(world["main"], STAGE, ["deps/sub"], str(world["tmp"] / "work"), str(world["tmp"] / "ship.lock"),
                static_tier=tier, lock_timeout_s=timeout)
    s.seen = seen  # type: ignore[attr-defined]
    return s


def _worktrees(main: str) -> int:
    return git(main, "worktree", "list").count("\n")


def test_a_ship_lands_byte_identical_records_and_leaves_main_clean(world):
    """CB-11 / CB-14 / CB-15: from a copy, with the submodule a gate reads, and a guarded --force remove."""
    s = shipper(world)
    res = s.ship(world["files"], "measure(t): records", tag="t1")
    assert s.seen["submodule"] is True   # type: ignore[attr-defined]
    m = world["main"]
    assert git(m, "status", "--porcelain") == ""
    for rel, src in world["files"].items():
        assert open(os.path.join(m, rel), "rb").read() == open(src, "rb").read()
    assert git(world["origin"], "log", "-1", "--format=%s") == "measure(t): records\n"
    assert res.commit and _worktrees(m) == 1 and "armchain-ship" not in git(m, "branch")


@pytest.mark.parametrize("rel", ["src/main/evil.py", f"{STAGE}/k9b/x.txt", f"{STAGE}/../../x.json", "/abs/x.json"])
def test_a_record_ship_never_commits_anything_but_stage_json(world, rel):
    with pytest.raises(Hold, match="writes only .json files"):
        shipper(world).ship({rel: next(iter(world["files"].values()))}, "m", tag="t")


def test_a_source_inside_main_is_refused(world):
    """arm_end v3-v12 shipped the record from main's own tree."""
    p = os.path.join(world["main"], "inside.json")
    open(p, "w").write("{}")
    with pytest.raises(Hold, match="inside the main checkout"):
        shipper(world).ship({f"{STAGE}/k9b/rb_a_s1.json": p}, "m", tag="t")


def test_main_already_holding_the_record_is_refused(world):
    os.makedirs(os.path.join(world["main"], STAGE, "dose"), exist_ok=True)
    # a stray copy makes main dirty; either refusal names the problem before anything is created
    open(os.path.join(world["main"], STAGE, "dose", "rb_a_s1.json"), "w").write("{}")
    with pytest.raises(Hold, match="dirty|already holds"):
        shipper(world).ship(world["files"], "m", tag="t")
    assert _worktrees(world["main"]) == 1


def test_a_failed_static_tier_pushes_nothing_and_removes_its_worktree(world):
    """H-15: a HOLD left the worktree and branch behind, so the retry died at ``worktree add``."""
    before = git(world["origin"], "rev-parse", "HEAD")
    with pytest.raises(Hold, match="static tier failed.*ship worktree was removed"):
        shipper(world, static=lambda wt: 1).ship(world["files"], "m", tag="t")
    assert git(world["origin"], "rev-parse", "HEAD") == before
    assert _worktrees(world["main"]) == 1 and "armchain-ship" not in git(world["main"], "branch")
    shipper(world).ship(world["files"], "m", tag="t")   # the same tag retries cleanly


def test_main_moving_during_the_ship_is_rebased_over(world, tmp_path):
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "-q", world["origin"], str(other)], check=True)
    _ident(str(other))

    def land_meanwhile(wt: str) -> int:
        (other / "NOTES.md").write_text("someone else landed\n")
        git(str(other), "add", "-A")
        git(str(other), "commit", "-q", "-m", "other")
        git(str(other), "push", "-q", "origin", "HEAD:main")
        return 0
    shipper(world, static=land_meanwhile).ship(world["files"], "measure(t): records", tag="t")
    log = git(world["origin"], "log", "--format=%s", "-2")
    assert log.splitlines() == ["measure(t): records", "other"]
    assert git(world["main"], "status", "--porcelain") == ""


def test_two_ships_never_push_at_once(world):
    s = shipper(world, timeout=0.3)
    fd = os.open(s.lock_path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with pytest.raises(Hold, match="another ship held"):
            s.ship(world["files"], "m", tag="t")
    finally:
        os.close(fd)


def test_a_reship_after_a_death_between_push_and_state_is_idempotent(world):
    """A supervisor that died after the push re-ships on restart: the same bytes already on main are a success."""
    first = shipper(world).ship(world["files"], "measure(t): records", tag="t1")
    again = shipper(world).ship(world["files"], "measure(t): records", tag="t2")
    assert again.already and again.commit == first.commit
    assert git(world["origin"], "log", "--format=%s").count("measure(t): records") == 1


def test_a_reship_with_different_bytes_holds(world):
    shipper(world).ship(world["files"], "m", tag="t1")
    src = world["files"][f"{STAGE}/dose/rb_a_s1.json"]
    open(src, "w").write('{"run": "changed"}\n')
    with pytest.raises(Hold, match="DIFFERENT bytes"):
        shipper(world).ship(world["files"], "m", tag="t2")


def test_a_push_beaten_by_another_landing_is_rebased_and_retried(world, tmp_path):
    """Review 10-26: a non-fast-forward push HELD and left its worktree; now it rebases over once and retries."""
    other = tmp_path / "other2"
    subprocess.run(["git", "clone", "-q", world["origin"], str(other)], check=True)
    _ident(str(other))
    state = {"n": 0}

    def guard_then_someone_lands(wt: str) -> int:
        state["n"] += 1
        if state["n"] == 1:
            (other / "LATE.md").write_text("landed between the rebase and the push\n")
            git(str(other), "add", "-A")
            git(str(other), "commit", "-q", "-m", "late")
            git(str(other), "push", "-q", "origin", "HEAD:main")
        return 0
    s = shipper(world)
    s.push_guard = guard_then_someone_lands
    s.ship(world["files"], "measure(t): records", tag="t")
    assert git(world["origin"], "log", "--format=%s", "-2").splitlines() == ["measure(t): records", "late"]
    assert state["n"] == 2 and _worktrees(world["main"]) == 1


def test_a_failure_after_the_push_cleans_up_and_the_retry_lands(world):
    """Review 10-26, finding 12: main's fast-forward failed after the push (a stray ``index.lock``) and left the ship's
    worktree and branch behind, so the by-hand retry died at ``worktree add``."""
    m = world["main"]
    lock = os.path.join(m, ".git", "index.lock")

    def guard_then_lock(wt: str) -> int:      # runs just before the push: main is locked from here on
        open(lock, "w").close()
        return 0
    s = Shipper(m, STAGE, ["deps/sub"], str(world["tmp"] / "work"), str(world["tmp"] / "ship.lock"),
                static_tier=lambda wt: (0, "fake"), push_guard=guard_then_lock)
    s._fast_forward_main.__func__.__defaults__ = (3, 0.1)   # type: ignore[attr-defined]
    try:
        with pytest.raises(Hold, match="(?s)pushed as .*fast-forward FAILED.*the ship worktree was removed — the records ARE on the remote"):
            s.ship(world["files"], "measure(t): records", tag="t1")
    finally:
        s._fast_forward_main.__func__.__defaults__ = (3, 2.0)   # type: ignore[attr-defined]
    assert _worktrees(m) == 1 and "armchain-ship" not in git(m, "branch")
    assert git(world["origin"], "log", "-1", "--format=%s") == "measure(t): records\n"
    os.remove(lock)
    again = shipper(world).ship(world["files"], "measure(t): records", tag="t2")
    assert again.already and git(m, "status", "--porcelain") == ""
