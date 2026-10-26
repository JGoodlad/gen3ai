"""A registered human-replay SNAPSHOT, read through its MANIFEST (REVIEW A M5; REVIEW D M1).

The registry (``designs/research_state/human_replay_snapshots.json``) holds each snapshot's manifest digest and file
count; the manifest (``<store>/<id>.tsv.gz``: relative path, sha256, bytes per log) IS the snapshot's definition. A
consumer reads the manifest after checking its digest against the registry, and re-hashes every log against its row
before parsing it. Any mismatch raises :class:`SnapshotError` and the consumer REFUSES.

:class:`SnapshotError` is an ordinary ``Exception`` (REVIEW D M1). Its first version subclassed ``SystemExit``: raised
inside a ``multiprocessing.Pool`` worker, a ``SystemExit`` kills the worker, the pool replaces it, the task's result
never arrives and the parent waits FOREVER. An ``Exception`` raised in a worker is pickled back and re-raised in the
parent by ``imap`` / ``map``, so a tampered log ends the job with a non-zero exit and the log's path in the message.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

#: the population's header signature: every spectator log starts with it, no archive log does (REVIEW A, G_*)
HEADER = b"|init|battle"
REGISTRY_REL = "designs/research_state/human_replay_snapshots.json"


class SnapshotError(Exception):
    """A snapshot whose manifest, file hash, size or header disagrees with the registry: every consumer refuses."""


def default_registry() -> Path:
    from utils.paths import repo_path

    return Path(repo_path(*REGISTRY_REL.split("/")))


def default_store() -> Path:
    return Path.home() / "gen3ai_archive" / "human_replay_snapshots"


def default_corpus() -> Path:
    """The main checkout's gitignored corpus (the originals; only ever read)."""
    from utils.git import get_main_repo_root

    return Path(get_main_repo_root()) / "replays" / "showdown" / "gen3ou"


def registry_row(snap: str, registry: Path) -> Dict[str, Any]:
    reg = json.loads(Path(registry).read_text())
    if snap not in reg.get("snapshots", {}):
        raise SnapshotError(f"snapshot {snap!r} is not in the registry {registry}")
    return reg["snapshots"][snap]


def read_manifest(snap: str, store: Path, registry: Path) -> List[Tuple[str, str, int]]:
    """The snapshot's (relative path, sha256, bytes) rows, after checking the manifest's digest and row count against
    the registry (extraction reads THE MANIFEST, never a date folder)."""
    row = registry_row(snap, registry)
    with gzip.open(Path(store) / f"{snap}.tsv.gz", "rb") as fh:
        body = fh.read()
    digest = hashlib.sha256(body).hexdigest()
    if digest != row["manifest_sha256"]:
        raise SnapshotError(f"{snap}: manifest sha256 {digest} != registry {row['manifest_sha256']}")
    out = []
    for ln in body.decode().splitlines():
        rel, sha, n = ln.split("\t")
        out.append((rel, sha, int(n)))
    if len(out) != int(row["files"]):
        raise SnapshotError(f"{snap}: manifest rows {len(out)} != registry files {row['files']}")
    return out


def verify_bytes(snap: str, population: str, rel: str, sha: str, n: int, b: bytes) -> None:
    """Refuse a log whose bytes, size or population header differ from its manifest row."""
    if len(b) != n or hashlib.sha256(b).hexdigest() != sha:
        raise SnapshotError(f"{snap}: {rel} differs from its manifest row (size {len(b)} vs {n}, or sha256)")
    if (population == "spectator") != b.startswith(HEADER):
        raise SnapshotError(f"{snap}: {rel} has the wrong header signature for population {population!r}")


def read_verified(snap: str, population: str, corpus: Path, item: Tuple[str, str, int]) -> bytes:
    """One manifest row's bytes, verified (:func:`verify_bytes`); never a silent substitute."""
    rel, sha, n = item
    b = (Path(corpus) / rel).read_bytes()
    verify_bytes(snap, population, rel, sha, n, b)
    return b
