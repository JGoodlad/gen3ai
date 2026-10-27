"""Test support for a site that resolves a RECORDED commit sha: a fake history-rewrite map entry.

``use_map(monkeypatch, tmp_path, {OLD: NEW})`` points :data:`utils.git_sha.MAP_PATH` at a one-off map, so
``git_sha.resolve`` (and every site routed through it) answers ``OLD`` with ``NEW`` while ``OLD`` names no object in any
repository. A site test builds a tiny repo, picks ``OLD = fake_old()`` and maps it to one of that repo's commits: the
test passes only if the site asked the map (it fails on revert, because git alone does not know ``OLD``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from utils import git_sha


def fake_old(tag: str = "a") -> str:
    """A 40-hex sha that names no commit anywhere (``tag`` must be one hex digit)."""
    return "0123456" + tag * 33


def use_map(monkeypatch, tmp_path: Path, rows: Mapping[str, str]) -> Path:
    p = Path(tmp_path) / "sha_map.tsv"
    p.write_text("# fake map\n" + "".join(f"{o}\t{n}\n" for o, n in rows.items()))
    monkeypatch.setattr(git_sha, "MAP_PATH", p)
    return p
