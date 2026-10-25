"""``rebuild_main_rust``: the refusals and the stamp verification, with FAKES only (a fake /proc, a fake runner, fake
stamps). No real cargo, no real main target, no GPU. The decision logic is what a ship relies on, so it is what is pinned:
a live user of the target REFUSES before anything is built; a stale or foreign artifact FAILS verification; a build
that dirties main FAILS."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest

from utils import rebuild_main_rust as R
from utils.rust_env import columns
from utils.rust_env import stamp as S

SRC = {"port": ("aaaa000000000001", 90), "env": ("bbbb000000000002", 125)}


@pytest.fixture(autouse=True)
def _fake_source_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tree's hash is a constant per scope, so a fake stamp is right or wrong by construction."""
    monkeypatch.setattr(S, "source_hash", lambda src=None, scope="env": SRC[scope])


@pytest.fixture
def main_dir(tmp_path: Path) -> Path:
    m = (tmp_path / "main").resolve()
    (m / "data" / "pokemon").mkdir(parents=True)
    for crate in ("rust_sim", "rust_env"):
        (m / "src" / crate / "target").mkdir(parents=True)
    return m


def stamp_for(a: R.Artifact, main: Path, **over: str) -> str:
    h, n = SRC[a.kind]
    d = {"stamp": "v2", "kind": a.kind, "commit": "c" * 40, "src": h, "nfiles": str(n)}
    if a.kind == "port":
        d["selfcheck"] = str(int(a.selfcheck))
    else:
        d.update(nan_poison=str(int(a.selfcheck)), schema=columns.schema_id(), data=str(main / "data" / "pokemon"))
    d.update(over)
    return ";".join(f"{k}={v}" for k, v in d.items())


def good_reader(main: Path, **over: str):
    return lambda a: stamp_for(a, main, **over)


def make_proc(root: Path, pid: int, *, comm: str = "x", exe: str = "", cmdline: List[str] = (), maps: List[str] = ()) -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "comm").write_text(comm + "\n")
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in cmdline) + (b"\0" if cmdline else b""))
    (d / "maps").write_text("".join(f"7f00-7f01 r-xp 0 0:0 1 {m}\n" for m in maps))
    if exe:
        os.symlink(exe, d / "exe")


# --------------------------------------------------------------------------------------------- the plan


def test_plan_is_the_four_builds_with_the_documented_cargo_flags(main_dir: Path) -> None:
    builds = R.plan(main_dir)
    assert [b.label for b in builds] == ["rust_sim/release", "rust_sim/selfcheck", "rust_env/release", "rust_env/selfcheck"]
    by = {b.label: b for b in builds}
    assert by["rust_sim/release"].cargo_args == ("build", "--release")
    assert by["rust_sim/selfcheck"].cargo_args == ("build", "--profile", "selfcheck", "--features", "emission-selfcheck")
    env = by["rust_env/selfcheck"].cargo_args
    assert "--lib" in env and "rust_env_proc" in env and "bot_reader" in env, "bot_reader is its own bin"
    assert all(b.cwd == main_dir / "src" / b.crate for b in builds)
    names = {a.path.name for a in by["rust_env/release"].artifacts}
    assert names == {"rust_env_proc", "bot_reader", "libpokesim_env.so"}
    assert {a.path.name for a in by["rust_sim/release"].artifacts} == set(R.PORT_STAMPED)
    # the three offline tools are VERIFIED after a rebuild too (they once "carried no stamp")
    assert {"ab_replay", "scan_move_probe", "bridge_replay"} <= {a.path.name for a in by["rust_sim/selfcheck"].artifacts}
    assert all(a.path.parent == main_dir / "src" / b.crate / "target" / b.profile for b in builds for a in b.artifacts)


def test_build_argv_is_niced_pinned_and_capped_with_the_env_scrubbed(main_dir: Path) -> None:
    b = R.plan(main_dir)[0]
    argv = R.build_argv(b, "/x/mem_cap.sh", "/c/cargo")
    assert argv[:7] == ["nice", "-n", "19", "taskset", "-c", "5-7,13-15", "/x/mem_cap.sh"]
    assert argv[7:10] == ["8", "/c/cargo", "build"]
    env = R.build_env({"CARGO_TARGET_DIR": "/elsewhere", "PATH": "/p"})
    assert "CARGO_TARGET_DIR" not in env and env["GEN3AI_BLOCKING"] and env["PATH"] == "/p"


def test_run_builds_runs_in_order_in_each_crate_and_stops_at_the_first_failure(main_dir: Path) -> None:
    calls = []

    def run(argv, cwd=None, env=None, timeout=None, **kw):
        calls.append((argv[-3:], cwd, env.get("CARGO_TARGET_DIR"), env["GEN3AI_BLOCKING"]))
        return SimpleNamespace(returncode=101 if len(calls) == 2 else 0)

    why = R.run_builds(R.plan(main_dir), "mc", "cargo", run=run, emit=lambda s: None,
                       environ={"CARGO_TARGET_DIR": "/elsewhere"})
    assert len(calls) == 2 and "rust_sim/selfcheck" in why and "101" in why
    assert calls[0][1] == str(main_dir / "src" / "rust_sim") and calls[0][2] is None and calls[0][3]


def test_a_build_timeout_is_a_failure_not_a_pass(main_dir: Path) -> None:
    def run(*a, **kw):
        raise subprocess.TimeoutExpired("cargo", 1)

    assert "did not finish" in R.run_builds(R.plan(main_dir), "mc", "cargo", run=run, emit=lambda s: None)


# --------------------------------------------------------------------------------------------- refusals


def test_live_users_are_found_by_maps_exe_and_argv_and_named(main_dir: Path, tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    lib = str(main_dir / "src/rust_env/target/release/libpokesim_env.so")
    ab = str(main_dir / "src/rust_sim/target/release/ab_replay")
    make_proc(proc, 11, comm="python3", maps=["/usr/lib/libc.so", lib + " (deleted)"])
    make_proc(proc, 12, comm="ab_replay", exe=ab)
    make_proc(proc, 13, comm="sh", cmdline=["sh", "-c", f"{ab} repro"])
    make_proc(proc, 14, comm="vim", cmdline=["vim", "/elsewhere/src/rust_sim/target/x"], maps=["/usr/lib/libc.so"])
    make_proc(proc, 15, comm="me")
    (proc / "meminfo").write_text("")                                   # a non-pid entry is ignored
    lines, unreadable = R.find_live_users(main_dir, str(proc), self_pids=(15,))
    assert unreadable == 0
    text = "\n".join(lines)
    assert f"11 python3: maps {lib}" in text and f"12 ab_replay: runs {ab}" in text and f"13 sh: names {ab} repro" in text
    assert "14 " not in text and "15 " not in text


def test_an_ancestor_that_only_mentions_the_path_is_not_a_user(main_dir: Path, tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    ab = str(main_dir / "src/rust_sim/target/release/ab_replay")
    make_proc(proc, 21, comm="bash", cmdline=["bash", "-c", f"echo {ab}"])
    make_proc(proc, 22, comm="bash", maps=[ab])
    lines, _ = R.find_live_users(main_dir, str(proc), argv_exempt=(21, 22))
    assert [ln.split()[0] for ln in lines] == ["22"], "argv of an ancestor is no evidence; its maps still are"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any maps file")
def test_unreadable_maps_are_counted_not_silently_clear(main_dir: Path, tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    make_proc(proc, 31)
    (proc / "31" / "maps").chmod(0)
    assert R.find_live_users(main_dir, str(proc)) == ([], 1)


def test_a_target_symlinked_outside_main_is_foreign(main_dir: Path, tmp_path: Path) -> None:
    assert R.foreign_targets(main_dir) == []
    other = tmp_path / "other_checkout" / "target"
    other.mkdir(parents=True)
    (main_dir / "src/rust_sim/target").rmdir()
    (main_dir / "src/rust_sim/target").symlink_to(other)
    bad = R.foreign_targets(main_dir)
    assert len(bad) == 1 and "rust_sim/target" in bad[0]


# --------------------------------------------------------------------------------------------- verification


def all_artifacts(main: Path) -> List[R.Artifact]:
    return [a for b in R.plan(main) for a in b.artifacts]


def test_verify_accepts_main_current_stamps(main_dir: Path) -> None:
    assert R.verify(R.plan(main_dir), main_dir, reader=good_reader(main_dir)) == []


@pytest.mark.parametrize("over, needle", [
    ({"src": "dead000000000000"}, "STALE RUST BUILD"),
    ({"nfiles": "7"}, "STALE RUST BUILD"),
    ({"stamp": "v1"}, "version"),
])
def test_verify_names_a_stale_artifact(main_dir: Path, over: Dict[str, str], needle: str) -> None:
    problems = R.verify(R.plan(main_dir), main_dir, reader=good_reader(main_dir, **over))
    assert len(problems) == len(all_artifacts(main_dir)) and all(needle in p for p in problems), problems


def test_verify_checks_the_profile_flags_and_the_data_dir(main_dir: Path) -> None:
    def reader(a: R.Artifact) -> str:
        if a.kind == "port":
            return stamp_for(a, main_dir, selfcheck=str(int(not a.selfcheck)))          # a release build in selfcheck/
        if a.path.name == "bot_reader":
            return stamp_for(a, main_dir, data="/some/other/checkout/data/pokemon")
        return stamp_for(a, main_dir, nan_poison=str(int(not a.selfcheck)))
    problems = "\n".join(R.verify(R.plan(main_dir), main_dir, reader=reader))
    assert "selfcheck=" in problems and "reads data /some/other/checkout" in problems and "nan_poison" in problems


def test_verify_reports_a_missing_or_silent_artifact(main_dir: Path) -> None:
    def reader(a: R.Artifact) -> str:
        if a.path.name == "sim_bridge":
            raise FileNotFoundError(f"{a.path} is missing")
        return stamp_for(a, main_dir)
    problems = R.verify(R.plan(main_dir), main_dir, reader=reader)
    assert len(problems) == 2 and all("sim_bridge is missing" in p for p in problems)


def test_read_stamp_refuses_a_missing_binary_and_a_silent_one(main_dir: Path) -> None:
    a = R.plan(main_dir)[0].artifacts[0]
    with pytest.raises(FileNotFoundError):
        R.read_stamp(a)
    a.path.parent.mkdir(parents=True)
    a.path.write_text("")
    with pytest.raises(RuntimeError, match="gave no stamp"):
        R.read_stamp(a, run=lambda *x, **k: SimpleNamespace(returncode=0, stdout="usage: x\n"))
    ok = R.read_stamp(a, run=lambda *x, **k: SimpleNamespace(returncode=0, stdout="noise\nstamp=v2;kind=port\n"))
    assert ok == "stamp=v2;kind=port"


# --------------------------------------------------------------------------------------------- main()


class Runner:
    """A fake `run` for git status AND the cargo builds."""

    def __init__(self, dirty_after_build: bool = False, fail_build: bool = False) -> None:
        self.builds = 0
        self.dirty_after_build = dirty_after_build
        self.fail_build = fail_build
        self.dirty_now = ""

    def __call__(self, argv, **kw):
        if argv[0] == "git":
            return SimpleNamespace(returncode=0, stdout=self.dirty_now, stderr="")
        self.builds += 1
        if self.dirty_after_build:
            self.dirty_now = " M src/rust_sim/Cargo.lock\n"
        return SimpleNamespace(returncode=101 if self.fail_build else 0)


@pytest.fixture
def empty_proc(tmp_path: Path) -> str:
    p = tmp_path / "emptyproc"
    p.mkdir()
    return str(p)


@pytest.fixture(autouse=True)
def _cargo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "cargo_path", lambda: "cargo")


def cli(main_dir: Path, *extra: str) -> List[str]:
    return [*extra, "--main", str(main_dir), "--mem-cap", "mem_cap.sh"]


def test_the_happy_path_builds_four_times_and_verifies(main_dir: Path, empty_proc: str, capsys) -> None:
    run = Runner()
    rc = R.main(cli(main_dir, "run"), proc=empty_proc, run=run, reader=good_reader(main_dir))
    assert rc == R.EXIT_OK and run.builds == 4
    assert "OK: 4 builds" in capsys.readouterr().out


def test_a_live_user_refuses_before_any_build_and_names_it(main_dir: Path, tmp_path: Path, capsys) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    make_proc(proc, 99, comm="rust_env_proc", exe=str(main_dir / "src/rust_env/target/release/rust_env_proc"))
    run = Runner()
    rc = R.main(cli(main_dir, "run"), proc=str(proc), run=run, reader=good_reader(main_dir))
    err = capsys.readouterr().err
    assert rc == R.EXIT_LIVE and run.builds == 0
    assert "REFUSING" in err and "99 rust_env_proc" in err


def test_a_dirty_main_refuses_before_any_build(main_dir: Path, empty_proc: str, capsys) -> None:
    run = Runner()
    run.dirty_now = " M data/x\n"
    assert R.main(cli(main_dir, "run"), proc=empty_proc, run=run, reader=good_reader(main_dir)) == R.EXIT_DIRTY
    assert run.builds == 0 and "not git-clean" in capsys.readouterr().err


def test_a_build_that_dirties_main_fails(main_dir: Path, empty_proc: str, capsys) -> None:
    run = Runner(dirty_after_build=True)
    assert R.main(cli(main_dir, "run"), proc=empty_proc, run=run, reader=good_reader(main_dir)) == R.EXIT_DIRTY
    assert "after the build" in capsys.readouterr().err


def test_a_failed_build_stops_and_a_stale_result_fails_verification(main_dir: Path, empty_proc: str, capsys) -> None:
    run = Runner(fail_build=True)
    assert R.main(cli(main_dir, "run"), proc=empty_proc, run=run, reader=good_reader(main_dir)) == R.EXIT_BUILD
    assert run.builds == 1
    stale = good_reader(main_dir, src="dead000000000000")
    assert R.main(cli(main_dir, "run"), proc=empty_proc, run=Runner(), reader=stale) == R.EXIT_VERIFY
    assert "VERIFICATION FAILED" in capsys.readouterr().err


def test_check_mode_verifies_without_building(main_dir: Path, empty_proc: str) -> None:
    run = Runner()
    assert R.main(cli(main_dir, "--check"), proc=empty_proc, run=run, reader=good_reader(main_dir)) == R.EXIT_OK
    assert run.builds == 0
    stale = good_reader(main_dir, src="dead000000000000")
    assert R.main(cli(main_dir, "--check"), proc=empty_proc, run=run, reader=stale) == R.EXIT_VERIFY


def test_no_mode_refuses_and_dry_run_runs_nothing(main_dir: Path, empty_proc: str, capsys) -> None:
    run = Runner()
    assert R.main(cli(main_dir), proc=empty_proc, run=run) == R.EXIT_USAGE
    assert R.main(cli(main_dir, "--dry-run"), proc=empty_proc, run=run) == R.EXIT_OK and run.builds == 0
    out = capsys.readouterr().out
    assert out.count("nice -n 19 taskset -c 5-7,13-15 mem_cap.sh 8") == 4 and "CARGO_TARGET_DIR unset" in out
