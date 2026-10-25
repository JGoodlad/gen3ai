"""The build-stamp check for the code that is NOT Python: ``python -m utils.rust_env.stamp --check-binary`` and the Node
harnesses' ``stamp_guard.js`` that calls it (``ab_fuzz.js``, ``bridge_ab_fuzz.js``, ``probe_batch7_isolated.js`` exec a
replayer by path; a stale one fuzzed another tree's engine silently). Fake binaries answer ``--build-stamp``; the check
itself is live. Each assertion fails on a revert of the check."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from utils.paths import src_path
from utils.rust_env import stamp as S
from utils.rust_env.stamp_fakes import make_stamp, write_fake_binary

STALE = dict(src="0123456789abcdef", nfiles="3", commit="19d5bad")


def _cli(path, *extra, env_extra=None):
    env = {**os.environ, "PYTHONPATH": str(src_path())}
    env.pop(S.ALLOW_STALE_ENV, None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "utils.rust_env.stamp", "--check-binary", str(path), *extra],
                          capture_output=True, text=True, env=env, timeout=120)


def test_the_cli_passes_a_current_binary_and_refuses_a_stale_one_in_one_line(tmp_path):
    good = write_fake_binary(tmp_path / "ab_replay", make_stamp("port"))
    assert _cli(good).returncode == 0
    stale = write_fake_binary(tmp_path / "stale_ab_replay", make_stamp("port", **STALE))
    r = _cli(stale, "--what", "ab_replay")
    assert r.returncode == 1 and r.stdout == ""
    assert len(r.stderr.strip().splitlines()) == 1 and "STALE RUST BUILD" in r.stderr and "src=0123456789abcdef" in r.stderr


def test_the_cli_refuses_a_binary_that_predates_the_flag_and_a_missing_one(tmp_path):
    assert _cli(write_fake_binary(tmp_path / "old", None)).returncode == 1
    assert _cli(tmp_path / "does_not_exist").returncode == 1


def test_the_cli_opt_out_runs_a_stale_binary_with_the_loud_warning(tmp_path):
    stale = write_fake_binary(tmp_path / "ab_replay", make_stamp("port", **STALE))
    r = _cli(stale, env_extra={S.ALLOW_STALE_ENV: "1"})
    assert r.returncode == 0 and "STALE RUST BUILD ALLOWED" in r.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is the harnesses' runtime")
def test_the_node_guard_exits_on_a_stale_replayer_and_lets_a_current_one_through(tmp_path):
    guard = src_path("rust_sim", "harness", "stamp_guard.js")
    good = write_fake_binary(tmp_path / "ab_replay", make_stamp("port"))
    stale = write_fake_binary(tmp_path / "stale_ab_replay", make_stamp("port", **STALE))

    def run(binary):
        code = f"require({str(guard)!r}).requireFreshBinary({str(binary)!r}, 'ab_replay'); console.log('PASSED')"
        env = {"PATH": os.environ.get("PATH", ""), "GEN3AI_PYTHON": sys.executable}
        return subprocess.run(["node", "-e", code], capture_output=True, text=True, env=env, timeout=120)

    ok = run(good)
    assert ok.returncode == 0 and "PASSED" in ok.stdout, ok.stderr
    bad = run(stale)
    assert bad.returncode == 1 and "PASSED" not in bad.stdout
    assert "REFUSED ab_replay" in bad.stderr and "STALE RUST BUILD" in bad.stderr, bad.stderr
