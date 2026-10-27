"""``fetch_filter_repo``: a cached script is reused; a wheel whose sha256 differs from the pinned one is REFUSED (no network)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from utils.history_rewrite import rewrite as W


def _fake_pip(wheel_bytes: bytes):
    def run(argv, check=False, **kw):
        dest = Path(argv[argv.index("-d") + 1])
        (dest / W.FILTER_REPO_WHEEL).write_bytes(wheel_bytes)
        return None
    return run


def _wheel(script: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("git_filter_repo.py", script)
    return buf.getvalue()


def test_a_cached_script_is_returned_without_touching_pip(tmp_path, monkeypatch):
    (tmp_path / "git_filter_repo.py").write_text("# cached\n")
    monkeypatch.setattr(W.subprocess, "run", lambda *a, **k: pytest.fail("pip must not run when the script is cached"))
    assert W.fetch_filter_repo(tmp_path) == tmp_path / "git_filter_repo.py"


def test_a_wheel_with_the_wrong_sha256_is_refused_and_nothing_is_unpacked(tmp_path, monkeypatch):
    monkeypatch.setattr(W.subprocess, "run", _fake_pip(_wheel(b"print('tampered')\n")))
    with pytest.raises(RuntimeError, match="pinned"):
        W.fetch_filter_repo(tmp_path / "dest")
    assert not (tmp_path / "dest" / "git_filter_repo.py").exists()


def test_the_pin_is_a_full_sha256_of_the_named_release():
    assert len(W.FILTER_REPO_WHEEL_SHA256) == 64 and set(W.FILTER_REPO_WHEEL_SHA256) <= set("0123456789abcdef")
    assert W.FILTER_REPO_VERSION in W.FILTER_REPO_WHEEL
