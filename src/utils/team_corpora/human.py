"""HUMAN-BUILT TEAM CORPORA (``gen3_human_team_stub_v1``): the corpora ``--human-data teams:<rowset>@<share>`` draws
from, kept OUTSIDE git.

Design: ``designs/endstate/human_data_architecture.md`` §6 (the switch) and ``design_human_replay_lever.md`` §6.4
(X60-T's ``hr_spec_v1.r<n>``). **Human replay data never leaves this machine** (the repo pushes to GitHub), so a
human corpus is split in two:

* the DATA, in ``utils.paths.human_teams_dir() / <id>/`` (``~/gen3ai_archive/human_teams/<id>/``): the usual
  ``gen3_team_corpus_v1`` layout (``manifest.json`` with ``"human_data": true`` and a tier ``ALL`` covering every
  row, ``teams.jsonl.gz``, ``soak_selfcheck.json``), written by the corpus builder;
* the STUB, committed in ``human_corpora.json`` beside this file: the id (= the rowset id the clause names), the
  sha256 of the manifest, the data file and the soak record, the counts and ``"human_data": true``. It holds no
  team text. ``python -m utils.team_corpora.human register <id>`` writes it from the built directory.

:func:`load` verifies the directory against its stub (each sha256, the count, the human-data flag, the tier ``ALL``
= the whole corpus) and raises :class:`utils.team_corpora.CorpusError` on a miss or a mismatch, which the trainer
turns into ``FATAL_CONFIG``. It is the ONLY reader that passes ``human_data_ok`` to ``load_dir``; ``--team-mix``
refuses a human corpus (``utils.team_corpora.refuse_human_data``, REVIEW F F-M7).

CLI::

    python -m utils.team_corpora.human register <id> [--dir <path>]   # write / replace the stub
    python -m utils.team_corpora.human verify <id>                    # load it as training would
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.team_corpora import (DATA_NAME, HUMAN_DATA_KEY, MANIFEST_NAME, SCHEMA, Corpus, CorpusError, load_dir,
                                read_dir)

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "human_corpora.json"
STUB_SCHEMA = "gen3_human_team_stub_v1"
SOAK_NAME = "soak_selfcheck.json"
#: The tier a ``teams`` clause draws: the whole corpus (``--team-mix <corpus>@ALL``).
WHOLE_TIER = "ALL"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """The committed stubs, by id (``{}`` when none is registered)."""
    doc = json.loads((path or REGISTRY).read_text())
    if doc.get("schema") != STUB_SCHEMA:
        raise CorpusError(f"{path or REGISTRY}: schema {doc.get('schema')!r} != {STUB_SCHEMA!r}")
    return dict(doc.get("corpora") or {})


def is_registered(corpus_id: str) -> bool:
    return corpus_id in registry()


def stub(corpus_id: str) -> Dict[str, Any]:
    stubs = registry()
    if corpus_id not in stubs:
        raise CorpusError(f"human team corpus {corpus_id!r} is not registered ({REGISTRY.name} holds "
                          f"{sorted(stubs) or 'none'}); build it, then `python -m utils.team_corpora.human register "
                          f"{corpus_id}`")
    return dict(stubs[corpus_id])


def corpus_dir(corpus_id: str) -> Path:
    from utils.paths import human_teams_dir

    return human_teams_dir() / corpus_id


def _measure(directory: Path, corpus_id: str) -> Dict[str, Any]:
    """The stub fields of the corpus in ``directory``, every shape rule checked (shared by register and load)."""
    for name in (MANIFEST_NAME, DATA_NAME, SOAK_NAME):
        if not (directory / name).is_file():
            raise CorpusError(f"human team corpus {corpus_id!r}: {directory / name} is missing (the corpus lives "
                              "outside git; build it on this machine, or copy it from the durable archive)")
    man, rows, data_sha, man_sha = read_dir(directory)
    if man.get(HUMAN_DATA_KEY) is not True:
        raise CorpusError(f"human team corpus {corpus_id!r}: its manifest lacks \"{HUMAN_DATA_KEY}\": true")
    tiers = man.get("tiers") or {}
    if WHOLE_TIER not in tiers or int(tiers[WHOLE_TIER]["n"]) != len(rows):
        raise CorpusError(f"human team corpus {corpus_id!r}: tier {WHOLE_TIER!r} must cover all {len(rows)} rows "
                          f"(the clause draws the whole corpus); the manifest's tiers are "
                          f"{ {k: v.get('n') for k, v in tiers.items()} }")
    return {"id": corpus_id, "rowset": corpus_id, "schema": SCHEMA, HUMAN_DATA_KEY: True, "tier": WHOLE_TIER,
            "n": len(rows), "tier_sha256": str(tiers[WHOLE_TIER]["sha256"]), "data_sha256": data_sha,
            "manifest_sha256": man_sha, "soak_sha256": _sha256(directory / SOAK_NAME),
            "snapshot": man.get("snapshot"), "built_by": man.get("built_by")}


#: The stub fields a load must reproduce exactly (``registered_at`` and the free-text ones are not identity).
IDENTITY_FIELDS = ("n", "tier", "tier_sha256", "data_sha256", "manifest_sha256", "soak_sha256", HUMAN_DATA_KEY)


@functools.lru_cache(maxsize=None)
def load(corpus_id: str) -> Corpus:
    """The verified whole corpus ``corpus_id``: every :data:`IDENTITY_FIELDS` value equals the committed stub's, or
    :class:`CorpusError` (a missing directory, a changed file, an unregistered id)."""
    want = stub(corpus_id)
    d = corpus_dir(corpus_id)
    got = _measure(d, corpus_id)
    bad = [f"{k}: {got.get(k)!r} != stub {want.get(k)!r}" for k in IDENTITY_FIELDS if got.get(k) != want.get(k)]
    if bad:
        raise CorpusError(f"human team corpus {corpus_id!r} at {d} does not match its committed stub "
                          f"({REGISTRY.name}): " + "; ".join(bad))
    return load_dir(d, WHOLE_TIER, corpus_id, human_data_ok=True)


def identity(corpus_id: str, corpus: Corpus) -> Dict[str, Any]:
    """The corpus block a ``teams`` clause's record carries."""
    s = stub(corpus_id)
    return {**corpus.ident(), "soak_sha256": s["soak_sha256"], HUMAN_DATA_KEY: True, "snapshot": s.get("snapshot")}


def register(corpus_id: str, directory: Optional[Path] = None, path: Optional[Path] = None) -> Dict[str, Any]:
    """Measure the built corpus and write (or replace) its stub in the committed registry."""
    from utils.atomic_io import atomic_write_json

    entry = _measure(Path(directory) if directory else corpus_dir(corpus_id), corpus_id)
    entry["registered_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    reg_path = path or REGISTRY
    doc = json.loads(reg_path.read_text())
    doc.setdefault("corpora", {})[corpus_id] = entry
    atomic_write_json(reg_path, doc, indent=1, sort_keys=True)
    load.cache_clear()
    return entry


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register", help="write the committed stub of a built human corpus")
    r.add_argument("id")
    r.add_argument("--dir", default=None, help="the built corpus (default: utils.paths.human_teams_dir()/<id>)")
    v = sub.add_parser("verify", help="load a registered corpus exactly as training does")
    v.add_argument("id")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "register":
            print(json.dumps(register(a.id, Path(a.dir) if a.dir else None), indent=1, sort_keys=True))
        else:
            c = load(a.id)
            print(f"OK {a.id}: {len(c)} teams, data sha256 {c.data_sha256}, manifest sha256 {c.manifest_sha256}")
    except CorpusError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
