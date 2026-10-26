"""``main.reads`` test fixtures. Every test takes its supervisors' path locks in its own tmp dir, never the box's
``~/.cache/gen3ai/reads_locks`` (``main.reads.supervisors``)."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reads_lock_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GEN3AI_READS_LOCK_DIR", str(tmp_path / "reads_locks"))


@pytest.fixture
def run_archive_env(tmp_path, monkeypatch):
    """A scratch run archive (``models/``) for the test's ``final`` refs."""
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()
    return tmp_path / "models"


@pytest.fixture
def no_git(monkeypatch):
    """The playing checkout is ``pipeline_test.CODE`` (no git call)."""
    from main.reads import runner as RN
    from main.reads.pipeline_test import CODE

    monkeypatch.setattr(RN, "checkout_code", lambda root=None: CODE)
