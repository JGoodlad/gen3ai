"""Pins for ``utils.rust_env.build`` (F-LG-6): the trainer builds THIS checkout's env core, for the declared
profile, into the crate's OWN ``target/`` — exactly where ``proc.default_path`` / ``ffi.default_path`` look —
and a failure is a typed error naming the command, never a fallback."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from utils.rust_env import build as B
from utils.rust_env import ffi, proc


def test_the_build_lands_where_the_loaders_look():
    for profile in ("release", "selfcheck"):
        child, lib = B.artifacts(profile)
        assert child == proc.default_path(profile) and lib == ffi.default_path(profile)
        assert child.parent == B.crate_dir().resolve() / "target" / profile


def test_build_argv_per_profile():
    rel = B.build_argv("release")
    assert rel[:2] == ["build", "--release"] and "--lib" in rel and rel[rel.index("--bin") + 1] == proc.BIN_NAME
    sc = B.build_argv("selfcheck")
    assert sc[1:5] == ["--profile", "selfcheck", "--features", "emission-selfcheck"]
    with pytest.raises(B.EnvCoreBuildError, match="unknown rust env profile"):
        B.build_argv("debug")


def _completed(rc=0, err=""):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout="", stderr=err)


def test_ensure_built_forces_the_crates_own_target_dir(monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        return _completed()

    monkeypatch.setenv("CARGO_TARGET_DIR", "/somewhere/else")     # never another checkout's target
    monkeypatch.setattr(B, "artifacts", lambda profile: (Path(__file__), Path(__file__)))
    lines = []
    B.ensure_built("release", emit=lines.append, run=run)
    (argv, kw), = calls
    assert argv[1:] == B.build_argv("release")
    assert kw["env"]["CARGO_TARGET_DIR"] == str(B.crate_dir() / "target")
    assert kw["cwd"] == str(B.crate_dir())
    assert lines and "[ENV CORE BUILD] release" in lines[0]


def test_a_failed_build_or_a_missing_artifact_is_refused_by_name(monkeypatch):
    with pytest.raises(B.EnvCoreBuildError, match="failed \\(exit 101\\)"):
        B.ensure_built("release", run=lambda *a, **k: _completed(101, "error[E0425]"))
    monkeypatch.setattr(B, "artifacts", lambda profile: (Path("/nonexistent/rust_env_proc"), Path(__file__)))
    with pytest.raises(B.EnvCoreBuildError, match="is missing"):
        B.ensure_built("release", run=lambda *a, **k: _completed())


def test_the_missing_build_hint_names_the_requested_profile(tmp_path):
    """The loaders' hint used to name the SELF-CHECK build for a missing RELEASE artifact."""
    missing = tmp_path / "target" / "release" / proc.BIN_NAME
    with pytest.raises(proc.ProcLoadError) as e:
        proc.ProcCore("{}", binary=missing)
    assert "--release" in str(e.value) and "selfcheck" not in str(e.value)
    with pytest.raises(ffi.FfiLoadError) as e2:
        ffi.load(tmp_path / "target" / "release" / "libpokesim_env.so")
    assert "--release" in str(e2.value)
