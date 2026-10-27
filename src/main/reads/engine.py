"""THE ENGINE A READ PLAYED ON: the Rust build-stamp identity of the sim and core, part of every regime
(``designs/tools/reads.md`` §4; REVIEW F F-M2, 2026-10-26).

Kakuna units and S cells play on OUR in-process Rust engine. The anchor regime id used to omit it: two reads with "the
same regime id" could play on different engines (the 10-26 full-parity batches changed OU-visible mechanics), and a
banked cell imported across them silently mixed engines. An ENGINE here is the build-stamp system's refusal key
(``utils.rust_env.stamp``: FNV-1a-64 over the non-test sources of a scope, with the file count):

* an anchor unit's: ``port=<src>/<n>;env=<src>/<n>``: the ``sim_bridge`` binary (scope ``port``) and the env core our
  side's ``rust_core_slot`` loads (scope ``env``). A summary of this code stamps it at play (``cell.engine_stamp``,
  ``main.anchors.server.engine_stamp``: the tree's hashes, which each artifact's stamp check proved it was built from);
* an S cell's: ``env=<src>/<n>``, read from its ledger row's ``compute.core_stamp`` (the row stamps the core it ran).

A summary written before the stamp existed has no ``engine_stamp``: its engine is :data:`LEGACY_ENGINE` of the playing
commit its ``server_version`` names, reconstructed once from git (:func:`engine_at`) and FROZEN here. A commit not in
the table has an UNKNOWN engine (``None``): it may pool within one state as before, and an import never accepts it.

Cargo.lock is not tracked (both crates' ``.gitignore``), yet each build hashes its root's lock. :func:`engine_at` takes
the lock blob :data:`LEGACY_LOCK_BLOBS` names; with it, the reconstruction reproduces the core stamps the eval ledger
recorded at 6c428453 (``7c53280566d666ae``), 1ed71c9dd (``9ecb4249afb76504``) and 46b9e572 (``393631b886965b0c``).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

SCHEMA = "gen3_read_engine_v1"

#: The ``Cargo.lock`` blob ids every banked build hashed (``port``: ``src/rust_sim``'s, ``env``: ``src/rust_env``'s).
LEGACY_LOCK_BLOBS: Dict[str, str] = {"port": "676e633ed0f069788af9c039a35eb5ee194a632d",
                                     "env": "3d00802e5b8fb5f7466b8b90dc681025e845e51d"}

E0 = "port=05b106e2f0fa7e16/84;env=dfd24aec97851576/119"
E0B = "port=e64b127b4be134a8/87;env=b92ea7ad9217b606/122"
E1 = "port=7d932165eced1c79/90;env=7c53280566d666ae/125"
E2 = "port=806f97bd59b9c5bb/90;env=9ecb4249afb76504/125"
E3 = "port=0d3d7629f16a0eaf/91;env=393631b886965b0c/126"

#: FROZEN: every banked anchor read's playing commit (as ``server_version`` stamps it) -> its engine (:func:`engine_at`
#: at that commit). ``designs/tools/reads.md`` §4 names the read behind each row. Never edited: a new read stamps its
#: own engine.
LEGACY_ENGINE: Dict[str, str] = {
    "63f97a57575e6af1db8278b2ae60f6d37466954d": E0,   # X51, x37a (10-17 .. 10-20)
    "cb0f1769ca118aaf2114711b68129a50d789dc8d": E0,
    "acb99527635247257e87bcd9f5a00c7f3ac02fec": E0,
    "b9c75884136cb4f239458714f0b2bb82eb3c3676": E0B,  # X51's home72 rows
    "f4b25238def57900c0305bb2a3a5a8f9edaef285": E1,   # X52; the X35 stage's production pin
    "6c428453a59b4b64b44df64b879b56c105c49e5a": E1,   # D1 h_a .. h_c
    "5e389ddac2c7ece26d043737f81dd3e94a7d06e8": E1,   # D1 h_d, h_e
    "e05c45eeba06cc13a798de356725a425a9bd4232": E1,   # x26c
    "d32ffcee4fb19d18df9d4f8083097f368df338fe": E1,   # human-only vs P*, metamon strongest
    "4b0ab220e68a7c2e4fd51d6a71f1bad3eb420e96": E1,
    "62bb07c002f6a978a42bde8d3ee0e5f208e2e32f": E1,   # PastaMittens
    "4ee655648b1c287cd27894e56359c036b2140293": E1,   # belief prior mix
    "033fcbcd25a98013d7d65e68e360470fec0063c8": E1,   # belief prior mix smoke
    "2cc864f2fb6b69a7dac42204d01ae88079f26c6d": E2,   # D3 look 1
    "46b9e572cc0fa237f62ddd948fb5fa4d115e44a4": E3,   # N1-P look 1
    "de15dfc7b6d19d6f0e4a3724f269e26e7aa22957": E3,   # N1-P look 1 (repin)
}

#: FROZEN: a banked control file that names no engine per row -> ``{lane: engine}`` (its sha256 keys it). D1's look-2
#: input (D3's ``import_off``), the C0 s3501 games D3's guard compared, and G1's S results, all played on E1.
LEGACY_BANK_ENGINE: Dict[str, Dict[str, str]] = {
    "8c4884aafcb5fa77a15b2e8be517460cf64f411a43a6ffa67bf02d1f4f2170b7": {"anchor": E1, "s": "env=7c53280566d666ae/125"},
    "be989806098e1b06f994ce9175c81238f6926ef94f367ab74b450db2e30ba336": {"anchor": E1},
    # G1's S results (SPRT look 1's imported C0), played at 3f3a2ed41 (E1's core, its rows' core_stamp)
    "393b7d572bb03c25afc089e3c7f9ca121c79aab9e1d436380fbbac96d9cf6786": {"s": "env=7c53280566d666ae/125"},
}

_COMMIT = re.compile(r"@([0-9a-f]{7,40})\+")
_CORE = re.compile(r"(?:^|;)src=([0-9a-f]{16});nfiles=(\d+)")


class EngineError(RuntimeError):
    """An engine that cannot be established or that differs where one engine is required (named)."""


def legacy_engine(commit: Optional[str]) -> Optional[str]:
    """The frozen engine of a banked playing commit (a full sha or the prefix ``server_version`` stamps), else None."""
    if not commit or len(commit) < 7:
        return None
    hits = {e for c, e in LEGACY_ENGINE.items() if c.startswith(commit)}
    return hits.pop() if len(hits) == 1 else None


def server_commit(server_version: Optional[str]) -> Optional[str]:
    """The playing commit a ``server_version`` (``ws_frontend[in-process]@<sha>+rust:...``) names, else None."""
    m = _COMMIT.search(str(server_version or ""))
    return m.group(1) if m else None


def unit_engine(cell: Mapping[str, Any]) -> Optional[str]:
    """One anchor unit's engine: the summary's own ``engine_stamp``, else the legacy table's for its commit."""
    stamped = cell.get("engine_stamp")
    if stamped:
        return str(stamped)
    return legacy_engine(server_commit(cell.get("server_version")))


def row_engine(row: Mapping[str, Any]) -> Optional[str]:
    """One S ledger row's engine (``env=<src>/<n>``) from its ``compute.core_stamp``, else None."""
    m = _CORE.search(str((row.get("compute") or {}).get("core_stamp") or ""))
    return f"env={m.group(1)}/{m.group(2)}" if m else None


def is_legacy(engine: Optional[str]) -> bool:
    """Whether ``engine`` is one a banked read played on (its regime ids were computed without it)."""
    return engine in set(LEGACY_ENGINE.values())


def one(engines: Iterable[Optional[str]]) -> Tuple[Optional[str], Optional[str]]:
    """``(engine, why)``: the single engine of a set of units / rows, or ``(None, reason)`` when they span several or
    any is unknown beside a known one. An all-unknown set is ``(None, None)`` (a legacy state, comparable to nothing)."""
    got = set(engines)
    known = sorted(e for e in got if e is not None)
    if len(known) > 1:
        return None, f"span {len(known)} engines {known}"
    if known and None in got:
        return None, f"mix engine {known[0]} with units of an unknown engine"
    return (known[0] if known else None), None


def bank_engine(entry: Mapping[str, Any], lane: str, rows: Iterable[Mapping[str, Any]] = ()) -> Optional[str]:
    """A banked control's engine for ``lane``: what its rows stamp (``engine_stamp`` or a ``server_version`` commit
    in the table; every row must agree), else the entry's registered ``engines[lane]``, else
    :data:`LEGACY_BANK_ENGINE` by its sha256, else None."""
    per = [unit_engine(r) for r in rows if r.get("engine_stamp") or r.get("server_version")]
    if per:
        e, why = one(per)
        if why:
            raise EngineError(f"banked rows {entry.get('path')}: {why}")
        return e
    reg = (entry.get("engines") or {}).get(lane)
    if reg:
        return str(reg)
    sha = str(entry.get("sha256") or "")
    return (LEGACY_BANK_ENGINE.get(sha) or {}).get(lane)


def require_same(what: str, here: Optional[str], there: Optional[str]) -> None:
    """Refuse (:class:`EngineError`) unless both engines are known and equal: an import across engines, or from an
    engine nobody can name, pools two different games."""
    if here is None or there is None:
        raise EngineError(f"{what}: engine unknown ({'this read' if here is None else 'the banked side'}); an import "
                          "needs both engines named (re-play the control, or register the bank's engine)")
    if here != there:
        raise EngineError(f"{what}: played on engine {there}, this read on {here}: an import never crosses engines "
                          "(re-play the control on this read's engine, or play this read on the bank's)")


# ------------------------------------------------------------------------------------------------ reconstruction
def engine_at(commit: str, repo: Optional[Path] = None, lock_blobs: Optional[Mapping[str, str]] = None) -> str:
    """The anchor engine ``port=..;env=..`` of ``commit``'s tree, from git objects (never the working tree): the build
    stamp's listing (``utils.rust_env.stamp.source_listing``) with each file's blob id, the untracked root
    ``Cargo.lock`` taken as ``lock_blobs`` (default :data:`LEGACY_LOCK_BLOBS`)."""
    from utils.rust_env import stamp as ST

    locks = dict(LEGACY_LOCK_BLOBS if lock_blobs is None else lock_blobs)
    out = []
    for scope in ("port", "env"):
        tags = ST.SCOPES[scope]
        lines = []
        for tag in tags:
            crate, extras = ST._CRATES[tag]
            listing = _git(repo, "ls-tree", "-r", commit, "--", f"src/{crate}")
            for ln in listing.splitlines():
                meta, path = ln.split("\t", 1)
                blob = meta.split()[2]
                rel = path[len(f"src/{crate}/"):]
                if ((rel.startswith("src/") and rel.endswith(".rs")) or rel in ("Cargo.toml", *extras)) \
                        and not ST.is_test_only(rel):
                    lines.append(f"{tag}/{rel}\t{blob}\n")
            if tag == tags[-1]:
                lines.append(f"{tag}/Cargo.lock\t{locks[tag]}\n")
        lines.sort()
        out.append(f"{scope}={ST._fnv64(''.join(lines).encode())}/{len(lines)}")
    return ";".join(out)


def lock_blob(path: Path) -> str:
    """The git blob id of a file (a ``Cargo.lock`` on disk)."""
    import hashlib

    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _git(repo: Optional[Path], *args: str) -> str:
    from utils.paths import repo_root

    r = subprocess.run(["git", "-C", str(repo or repo_root()), *args], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise EngineError(f"git {' '.join(args[:3])}: {r.stderr.strip()[:300]}")
    return r.stdout
