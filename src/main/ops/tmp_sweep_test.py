"""``scripts/ops/tmp_sweep.py`` against REAL directories and REAL live processes.

Every test sweeps its own ``tmp_path`` as ``--root`` — never the box's ``/tmp`` — with mtimes pushed
into the past by ``os.utime``. The live references are genuine: a ``sleep`` whose CWD is a stale
dir, a python child holding a FILE open (path passed on stdin, so only its fd can name it), and a
python child that names a dir on its COMMAND LINE. Each is killed by explicit PID.

``integration``: the tests spawn and kill their own processes. ~2 s in all.
"""
from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, List

import pytest

from utils.paths import repo_path

pytestmark = pytest.mark.integration

_SCRIPT = repo_path("scripts", "ops", "tmp_sweep.py")
_spec = importlib.util.spec_from_file_location("tmp_sweep", _SCRIPT)
assert _spec is not None and _spec.loader is not None
S = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = S          # @dataclass resolves its module through sys.modules
_spec.loader.exec_module(S)

HOUR = 3600.0


def age(path: Path, hours: float) -> None:
    """Push the mtime of ``path`` and everything under it ``hours`` into the past (no link follow)."""
    t = time.time() - hours * HOUR
    for dirpath, dirnames, filenames in os.walk(path, topdown=False, followlinks=False):
        for nm in dirnames + filenames:
            os.utime(os.path.join(dirpath, nm), (t, t), follow_symlinks=False)
    os.utime(path, (t, t), follow_symlinks=False)


def mkstale(root: Path, name: str, hours: float = 10.0, files: int = 2) -> Path:
    d = root / name
    (d / "sub").mkdir(parents=True)
    for i in range(files):
        (d / "sub" / f"f{i}.bin").write_bytes(b"x" * 5000)
    age(d, hours)
    return d


@pytest.fixture
def procs() -> Iterator[List[subprocess.Popen]]:
    live: List[subprocess.Popen] = []
    yield live
    for p in live:
        if p.poll() is None:
            os.kill(p.pid, 9)          # explicit PID, never a pattern
            p.wait(timeout=10)


def hold_cwd(procs: List[subprocess.Popen], d: Path) -> None:
    procs.append(subprocess.Popen(["sleep", "120"], cwd=d))


def hold_fd(procs: List[subprocess.Popen], f: Path) -> None:
    p = subprocess.Popen([sys.executable, "-c",
                          "import sys,time; fh=open(sys.stdin.readline().strip()); "
                          "print('ok', flush=True); time.sleep(120)"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd="/")
    assert p.stdin is not None and p.stdout is not None
    p.stdin.write(f"{f}\n")
    p.stdin.flush()
    assert p.stdout.readline().strip() == "ok"
    procs.append(p)


def hold_cmdline(procs: List[subprocess.Popen], d: Path) -> None:
    procs.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", str(d)],
                                  cwd="/"))
    time.sleep(0.2)                    # the argv is in /proc once exec has happened


def run(root: Path, *extra: str, capsys=None) -> str:
    rc = S.main(["--root", str(root), *extra])
    assert rc == 0
    return capsys.readouterr().out if capsys is not None else ""


def test_dry_run_plans_but_deletes_nothing(tmp_path, capsys):
    stale = mkstale(tmp_path, "stale_dir")
    loose = tmp_path / "stale.log"
    loose.write_text("x")
    age(loose, 10)
    out = run(tmp_path, capsys=capsys)
    assert stale.exists() and loose.exists()
    assert "stale_dir" in out and "stale.log" in out and "DRY RUN" in out
    assert "NOTHING WAS DELETED" in out
    assert "inodes" in out                 # inode use is reported beside bytes


def test_age_is_the_NEWEST_mtime_inside(tmp_path, capsys):
    """A directory whose top is old but with ONE fresh file deep inside is live work — kept."""
    d = mkstale(tmp_path, "old_top_fresh_leaf")
    (d / "sub" / "deep").mkdir()
    (d / "sub" / "deep" / "new.txt").write_text("fresh")
    os.utime(d, (time.time() - 10 * HOUR,) * 2)
    os.utime(d / "sub", (time.time() - 10 * HOUR,) * 2)
    run(tmp_path, "--apply", capsys=capsys)
    assert (d / "sub" / "deep" / "new.txt").exists()


def test_apply_spares_every_live_reference_and_the_keep_list(tmp_path, procs, capsys):
    gone = mkstale(tmp_path, "gone")
    by_cwd = mkstale(tmp_path, "held_by_cwd")
    by_fd = mkstale(tmp_path, "held_by_fd")
    by_cmd = mkstale(tmp_path, "held_by_cmdline")
    kept = mkstale(tmp_path, "keep_me")
    sock_dir = tmp_path / "has_socket"
    sock_dir.mkdir()
    s = socket.socket(socket.AF_UNIX)
    here = os.getcwd()
    os.chdir(sock_dir)                 # bind RELATIVE: a deep tmp_path can exceed sun_path's 108 bytes
    try:
        s.bind("agent.sock")
    finally:
        os.chdir(here)
        s.close()                      # the socket FILE stays on disk, unbound
    age(sock_dir, 10)
    hold_cwd(procs, by_cwd)
    hold_fd(procs, by_fd / "sub" / "f0.bin")
    hold_cmdline(procs, by_cmd)

    out = run(tmp_path, "--apply", "--keep", "keep_me", capsys=capsys)

    assert not gone.exists(), out
    for d in (by_cwd, by_fd, by_cmd, kept, sock_dir):
        assert d.exists(), f"{d.name} was deleted:\n{out}"
    assert "referenced by a live process" in out
    assert "contains a socket" in out
    assert "declared KEEP" in out


def test_symlinks_are_never_followed_out_of_the_root(tmp_path_factory, capsys):
    root = tmp_path_factory.mktemp("sweep_root")
    outside = tmp_path_factory.mktemp("outside")
    target = outside / "precious"
    target.mkdir()
    (target / "data.bin").write_bytes(b"y" * 100)
    inner = mkstale(root, "dir_with_link")
    (inner / "link_out").symlink_to(target, target_is_directory=True)
    (root / "top_link").symlink_to(target, target_is_directory=True)
    age(inner, 10)
    os.utime(root / "top_link", (time.time() - 10 * HOUR,) * 2, follow_symlinks=False)
    age(outside, 10)

    run(root, "--apply", capsys=capsys)

    assert not inner.exists() and not os.path.lexists(root / "top_link")
    assert (target / "data.bin").read_bytes() == b"y" * 100


def test_cache_dirs_are_FILE_pruned_not_deleted(tmp_path, procs, capsys):
    cache = tmp_path / "my_inductor"
    (cache / "old_sub").mkdir(parents=True)
    (cache / "mixed").mkdir()
    old = cache / "old_sub" / "kernel.so"
    old.write_bytes(b"k" * 4096)
    held = cache / "mixed" / "held.so"
    held.write_bytes(b"h")
    new = cache / "mixed" / "fresh.py"
    age(cache, 72)
    new.write_text("fresh")            # after the aging: this one is new
    hold_fd(procs, held)

    out = run(tmp_path, "--apply", "--cache", "my_inductor", capsys=capsys)

    assert cache.is_dir(), "the cache ROOT must survive"
    assert not old.exists() and not (cache / "old_sub").exists(), out   # file + its emptied dir
    assert new.exists() and held.exists(), out


def test_pytest_basetemp_children_are_judged_whole(tmp_path, procs, capsys):
    pyt = tmp_path / f"pytest-of-{S._user()}"
    stale = mkstale(pyt, "pytest-1", hours=72)
    live = mkstale(pyt, "pytest-2", hours=72)   # copied fixtures keep old mtimes
    fresh = mkstale(pyt, "pytest-3", hours=1)
    os.utime(pyt, (time.time() - 72 * HOUR,) * 2)
    hold_cwd(procs, live / "sub")

    run(tmp_path, "--apply", capsys=capsys)

    assert not stale.exists()
    assert live.exists() and fresh.exists() and pyt.exists()


def test_another_uid_is_never_touched(tmp_path, capsys):
    d = mkstale(tmp_path, "mine_but_filtered")
    out = run(tmp_path, "--apply", "--uid", str(os.getuid() + 1), capsys=capsys)
    assert d.exists() and "not owned by the uid" in out


def test_an_entry_that_comes_alive_after_the_plan_survives(tmp_path):
    d = mkstale(tmp_path, "revived")
    root = os.path.realpath(tmp_path)
    refs = S.scan_live_refs(root)
    plan = S.build_plan(root, os.getuid(), set(), set(), set(), 6.0, 48.0, refs)
    assert [i.name for i in plan.delete] == ["revived"]
    (d / "sub" / "written_now.txt").write_text("alive")
    S.apply_plan(plan, root, 6.0, 48.0)
    assert d.exists()


def test_warns_loudly_above_the_bar_and_still_exits_zero(tmp_path, capsys):
    rc = S.main(["--root", str(tmp_path), "--warn-pct", "0"])
    cap = capsys.readouterr()
    assert rc == 0
    assert "WARNING" in cap.out and "WARNING" in cap.err and "INODES" in cap.out
