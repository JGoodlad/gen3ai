"""The team_strength CLI finds Metamon's team cache through the anchors config, never a literal path."""
from pathlib import Path

import pytest

from main.team_strength import cli


def test_default_metamon_teams_follows_the_anchors_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEN3AI_METAMON_CACHE_DIR", str(tmp_path))
    assert cli.default_metamon_teams() == tmp_path / "teams"


def test_default_metamon_teams_resolves_from_the_committed_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEN3AI_METAMON_CACHE_DIR", raising=False)
    monkeypatch.delenv("GEN3AI_ANCHORS_CONFIG", raising=False)
    got = cli.default_metamon_teams()
    assert got.name == "teams" and got.parent.name == "cache"
