"""The LADDER-USAGE corpus — a third TEAM SOURCE for the fuzz and parity gates
(`gen3_ladder_usage_corpus_v1`).

Our correctness gates saw two team sources: the 719-team training pool (``data/teams/``, one narrow
human meta) and the procedural generator (``src/rust_sim/harness/ou_random_teams.js``). Real
ladder teams found a live crash on first contact (Heal Bell in the obs encode, ``40330daa``), so
the gates now also play the teams people actually bring: Metamon's ``hl_05_26`` gen3ou set
(22,862 teams from the public ladder, ``teams`` revision v5), filtered to what the ENGINE can play.

🚨 **A TEST corpus, never a prior.** Nothing the network reads may come from here (owner rule:
priors trace to Smogon). It lives under ``src/`` beside its loader, NOT under ``data/`` — pinned
runs read ``data/`` from main, so a corpus there would reach a live run.

**The filter** (``python -m utils.ladder_corpus.build``; the counts are in ``manifest.json``):
``Teams.import`` reads it, six Pokemon, ``TeamValidator('gen3ou')``-legal, and every species /
move / item / ability is one the ENGINE runs (``scan_move_probe``, the coverage oracle). Nothing
else is dropped: the already-fixed Heal Bell teams stay IN, and so do teams whose species set
matches a pool team. The procedural generator's stricter predicate (``isModeledMove`` & co.) is
recorded for information only — it is a PICKER predicate with false negatives (it rejects Sleep
Talk, which the engine plays).

**Tiers are PREFIXES of one seeded permutation** (seed ``SEED``), each hash-stamped in the
manifest, so a tier is the same teams on every box and a larger tier contains the smaller:

=========  =====================  =================================================================
tier       size                   used by
=========  =====================  =================================================================
commit     ``COMMIT_N`` teams     the recorded COMMIT-tier battles of the Rust Core parity gate
milestone  ``MILESTONE_N`` teams  slice E/V's MILESTONE ladder battles, the fuzzers' default draw
full       every team             the CUTOVER tier, a ``--ladder-tier full`` fuzz soak
=========  =====================  =================================================================

Every reader verifies the data file's sha256 against the manifest and REFUSES on a mismatch.
The JS twin for the fuzzers is ``src/rust_sim/harness/ladder_corpus.js``; the team-source hook
every consumer shares is :mod:`utils.team_sources`.
"""
from __future__ import annotations

import functools
import gzip
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
DATA = HERE / "teams.jsonl.gz"
MANIFEST = HERE / "manifest.json"

SCHEMA = "gen3_ladder_usage_corpus_v1"
SEED = 20260924
COMMIT_N = 16
MILESTONE_N = 800
TIERS = ("commit", "milestone", "full")


class LadderCorpusError(RuntimeError):
    """The committed corpus does not match its manifest."""


def tier_sha256(packed: List[str]) -> str:
    """The digest a tier is stamped with: sha256 over its packed teams, in order, NUL-joined."""
    h = hashlib.sha256()
    for p in packed:
        h.update(p.encode())
        h.update(b"\x00")
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


@functools.lru_cache(maxsize=1)
def _all_rows() -> Tuple[dict, ...]:
    m = manifest()
    raw = DATA.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != m["data_sha256"]:
        raise LadderCorpusError(f"{DATA.name}: sha256 {digest} != manifest {m['data_sha256']} — "
                                "rebuild with `python -m utils.ladder_corpus.build`")
    rows = tuple(json.loads(line) for line in gzip.decompress(raw).decode().splitlines() if line)
    if len(rows) != m["tiers"]["full"]["n"]:
        raise LadderCorpusError(f"{len(rows)} rows for a manifest of {m['tiers']['full']['n']}")
    return rows


def rows(tier: str = "milestone") -> List[dict]:
    """``tier``'s rows (``{"i", "sha", "file", "packed"}``), in corpus order."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    n = manifest()["tiers"][tier]["n"]
    return list(_all_rows()[:n])


def teams(tier: str = "milestone") -> List[str]:
    """``tier``'s packed teams, in corpus order."""
    return [r["packed"] for r in rows(tier)]


def pair(key: int, tier: str = "milestone") -> Tuple[str, str]:
    """The two teams battle ``key`` plays: ``2·key`` and ``2·key + 1`` (mod the tier), so the keys
    ``0 .. n/2 - 1`` play every team of the tier exactly once."""
    t = teams(tier)
    return t[(2 * key) % len(t)], t[(2 * key + 1) % len(t)]


def verify() -> Dict[str, int]:
    """Every tier's stamp against the data; returns ``{tier: n}`` or raises."""
    m = manifest()
    out = {}
    for tier in TIERS:
        got = tier_sha256(teams(tier))
        want = m["tiers"][tier]["sha256"]
        if got != want:
            raise LadderCorpusError(f"tier {tier}: sha256 {got} != manifest {want}")
        out[tier] = m["tiers"][tier]["n"]
    return out
