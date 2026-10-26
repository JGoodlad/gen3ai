"""PIN-SAFE TRAINING TEAM CORPORA (``gen3_team_corpus_v1``): the pool-widening corpora ``--team-mix`` draws from.

Design: ``designs/endstate/design_diversity_teams_opponents.md`` §4.1 ("Where it lives", F-DV-2) and §4.2 (D1 / X38).

🚨 **Why under ``src/`` and not ``data/teams/``.** ``TeamLoader`` walks ``data/teams/`` RECURSIVELY, and a pinned run
reads ``data/`` from MAIN: a corpus placed there would silently widen every live run's pool at its next restart. A
pinned run reads ``src/`` from its own pin, so a corpus here reaches only a run that NAMES it (``--team-mix``) at a
commit that holds it. Owner / orchestrator decision Q2 (2026-10-18): a ``src/`` corpus now; ``data/`` only once pinned
children resolve ``data/`` through the pin tree (TECH_DEBT (f) P1).

🚨 **Training DATA, never a prior.** Nothing the network reads as a prior may be fitted on a corpus here (owner rule:
priors trace to Smogon). No ``tools/compute_priors.py`` input, no ``agents.gen3_data`` table, reads this package.

**Layout.** One directory per corpus (``CORPORA``): ``manifest.json`` (schema, provenance, the build's rules and counts,
the tiers, ``data_sha256``) and ``teams.jsonl.gz`` (one row per team, in TIER ORDER: ``{"i", "id", "tier", "curator",
"sha256", "text"}``; ``text`` is the Showdown paste, ``.strip()``ed, ``sha256`` is over it). A TIER is a PREFIX of the
rows (``A`` ⊂ ``AB`` ⊂ ``ABC``), each stamped in the manifest with its count and :func:`tier_sha256`; every reader
verifies the data file and the tier against the manifest and REFUSES on a mismatch (:class:`CorpusError`).
"""
from __future__ import annotations

import functools
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
SCHEMA = "gen3_team_corpus_v1"
DATA_NAME = "teams.jsonl.gz"
MANIFEST_NAME = "manifest.json"
#: The registered corpora, by name. A name not here is refused (a typo must not train on nothing).
CORPORA: Dict[str, Path] = {"w_forum_v1": HERE / "w_forum_v1", "n1_proc_v1": HERE / "n1_proc_v1"}


class CorpusError(RuntimeError):
    """A corpus is unknown, or its committed files do not match its manifest."""


@dataclass(frozen=True)
class Corpus:
    """One verified tier of a corpus: the teams ``--team-mix`` draws from, with their identity."""

    name: str
    tier: str
    ids: Tuple[str, ...]
    texts: Tuple[str, ...]
    tier_sha256: str
    data_sha256: str
    manifest_sha256: str

    def __len__(self) -> int:
        return len(self.texts)

    def ident(self) -> Dict[str, object]:
        """The provenance block every consumer records (metadata, the startup echo)."""
        return {"corpus": self.name, "tier": self.tier, "n": len(self.texts), "tier_sha256": self.tier_sha256,
                "data_sha256": self.data_sha256, "manifest_sha256": self.manifest_sha256, "schema": SCHEMA}


def tier_sha256(texts: Sequence[str]) -> str:
    """A tier's stamp: sha256 over its team texts, in order, NUL-joined."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()


def corpus_dir(name: str) -> Path:
    if name not in CORPORA:
        raise CorpusError(f"unknown team corpus {name!r}; registered: {', '.join(sorted(CORPORA))}")
    return CORPORA[name]


def encode_rows(rows: Sequence[Dict[str, object]]) -> bytes:
    """The data file's bytes: one sorted-key JSON row per line, gzip at level 9 with mtime 0 (reproducible)."""
    blob = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode()
    return gzip.compress(blob, compresslevel=9, mtime=0)


def read_dir(directory: Path) -> Tuple[Dict[str, object], Tuple[Dict[str, object], ...], str, str]:
    """``(manifest, rows, data_sha256, manifest_sha256)`` of a corpus directory, the data file checked."""
    man_bytes = (directory / MANIFEST_NAME).read_bytes()
    man = json.loads(man_bytes)
    if man.get("schema") != SCHEMA:
        raise CorpusError(f"{directory}: schema {man.get('schema')!r} != {SCHEMA!r}")
    raw = (directory / DATA_NAME).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != man["data_sha256"]:
        raise CorpusError(f"{directory / DATA_NAME}: sha256 {digest} != manifest {man['data_sha256']}")
    rows = tuple(json.loads(ln) for ln in gzip.decompress(raw).decode().splitlines() if ln)
    for r in rows:
        if hashlib.sha256(str(r["text"]).encode()).hexdigest() != r["sha256"]:
            raise CorpusError(f"{directory}: row {r.get('i')} ({r.get('id')}) text does not match its sha256")
    return man, rows, digest, hashlib.sha256(man_bytes).hexdigest()


def load_dir(directory: Path, tier: str, name: str = "") -> Corpus:
    """One verified tier of the corpus in ``directory`` (tests build corpora in a temp dir through this)."""
    man, rows, data_sha, man_sha = read_dir(directory)
    tiers = man["tiers"]
    if tier not in tiers:
        raise CorpusError(f"corpus {name or directory}: tier {tier!r} not in {list(tiers)}")
    n = int(tiers[tier]["n"])
    if n > len(rows):
        raise CorpusError(f"corpus {name or directory}: tier {tier} needs {n} rows, the data holds {len(rows)}")
    sel = rows[:n]
    texts = tuple(str(r["text"]) for r in sel)
    got = tier_sha256(texts)
    if got != tiers[tier]["sha256"]:
        raise CorpusError(f"corpus {name or directory}: tier {tier} sha256 {got} != manifest {tiers[tier]['sha256']}")
    return Corpus(name=name or str(man.get("name", directory.name)), tier=tier,
                  ids=tuple(str(r["id"]) for r in sel), texts=texts, tier_sha256=got, data_sha256=data_sha,
                  manifest_sha256=man_sha)


@functools.lru_cache(maxsize=None)
def load_corpus(name: str, tier: str = "") -> Corpus:
    """The verified ``tier`` of the registered corpus ``name`` (``""`` = the manifest's ``default_tier``)."""
    d = corpus_dir(name)
    if not tier:
        tier = str(json.loads((d / MANIFEST_NAME).read_text())["default_tier"])
    return load_dir(d, tier, name)


def tiers(name: str) -> List[str]:
    return list(json.loads((corpus_dir(name) / MANIFEST_NAME).read_text())["tiers"])
