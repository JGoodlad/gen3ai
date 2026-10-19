"""Test helpers for ``--team-mix`` (``gen3_team_mix_v1``): a tiny ``gen3_team_corpus_v1`` corpus built in a temp dir
and registered under a test name, so no test depends on the real (large) corpus.

``install_corpus(monkeypatch, tmp_path, texts, tiers={"A": 1, "AB": 2})`` writes ``manifest.json`` +
``teams.jsonl.gz`` (through the loader's own ``encode_rows`` / ``tier_sha256``), registers the directory in
``utils.team_corpora.CORPORA`` under ``name`` and clears ``load_corpus``'s cache (a test module clears it again on
teardown, :func:`clear_corpus_cache`, so a temp corpus never outlives its test).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

TEST_CORPUS = "t_mix_v1"


def write_corpus(directory: Path, texts: Sequence[str], tiers: Dict[str, int], default_tier: Optional[str] = None,
                 name: str = TEST_CORPUS) -> Path:
    from utils.team_corpora import DATA_NAME, MANIFEST_NAME, SCHEMA, encode_rows, tier_sha256

    directory.mkdir(parents=True, exist_ok=True)
    texts = [t.strip() for t in texts]
    rows = [{"i": i, "id": f"{name}_{i:04d}", "tier": "A", "curator": "test",
             "sha256": hashlib.sha256(t.encode()).hexdigest(), "text": t} for i, t in enumerate(texts)]
    blob = encode_rows(rows)
    (directory / DATA_NAME).write_bytes(blob)
    manifest = {"schema": SCHEMA, "name": name, "default_tier": default_tier or list(tiers)[-1],
                "tiers": {k: {"n": n, "sha256": tier_sha256(texts[:n])} for k, n in tiers.items()},
                "data_sha256": hashlib.sha256(blob).hexdigest()}
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1))
    return directory


def install_corpus(monkeypatch, tmp_path: Path, texts: Sequence[str], tiers: Optional[Dict[str, int]] = None,
                   name: str = TEST_CORPUS) -> Path:
    import utils.team_corpora as TC

    d = write_corpus(tmp_path / name, texts, tiers or {"A": len(texts)}, name=name)
    monkeypatch.setitem(TC.CORPORA, name, d)
    TC.load_corpus.cache_clear()
    return d


def clear_corpus_cache() -> None:
    import utils.team_corpora as TC

    TC.load_corpus.cache_clear()
