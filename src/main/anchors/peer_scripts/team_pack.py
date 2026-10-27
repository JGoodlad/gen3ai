"""The Metamon peer's TEAM PACK: every team file packed as a real Showdown server plays it, verified, or no play.

STDLIB ONLY, and shared by both interpreters (like ``weight_pins.py``): ``metamon_side.py`` runs under the Metamon env
(``PYTHONPATH=""``) and loads this file by path; ``main.anchors`` and the tests import it as an ordinary module.

THE DEFECT THIS CLOSES (2026-10-27, the H-CTX Hidden Power FINDING and its sweep). The peer packed its team files with
upstream poke-env's ``Teambuilder``, which is not Showdown's reading of a paste:

* a named Hidden Power with all-31 IVs got poke-env's ``HP_TO_IVS`` row, written in the formula's bit order (hp, atk,
  def, spe, spa, spd) but packed as (hp, atk, def, spa, spd, spe): 12 of the 16 types read as another type and the
  server refused the team (rc72); Flying kept its type with other IVs (away ``team_23``'s Gyarados);
* ``Happiness: 0`` packed as the default 255 (``happiness or ""``), so Frustration hit at base power 1, not 102
  (forum72 ``sft431``'s Gyarados, pool ``a66608c5e5``);
* a ``Hidden Power: <Type>`` line on a mon without the move left its IVs at 31 (pool ``16d681e970``'s Salamence),
  where the validator writes the type's.

Now the peer packs through ``utils/bridge/showdown_pack.js`` (import, the validator, ``Teams.pack`` of the validated
sets, as ``server/team-validator-async.ts`` does), and :func:`build` REFUSES (:class:`TeamPackError`, naming the file
and the set) unless ``utils/bridge/pack_oracle.js`` agrees that every pack is Showdown's reading of its file. Every
file is packed and checked at startup, before the first battle, so a bad file is a startup failure, never a game.

THE REGIME STAMP. :func:`build` also packs each file the LEGACY way (the caller's poke-env packer) and records every
file whose played set differs (``showdown_pack.js``'s ``legacy_diffs``). ``peer_pack_changed`` is True iff any file of
the set the peer may draw differs: the cell stamps it and ``main.reads.assemble.anchor_regime_id`` adds it to the id
only then, so a team set the old packer played identically (home72, p72, heldout72_hpfix, rc72b, ...) keeps its id and
pools with banked reads, and one it played differently (forum72, away, the pool) gets a new id.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: The regime's name for a cell whose peer packs through Showdown AND whose team set the legacy packer played
#: differently (``main.reads.assemble.anchor_regime_id``).
PEER_PACK_SCHEMA = "gen3_peer_showdown_pack_v1"
_NODE_TIMEOUT_S = 600


class TeamPackError(RuntimeError):
    """A team file Showdown refuses, or whose pack is not Showdown's reading of it: the peer must not play."""


def _node(script: str, payload: Dict[str, Any], n: int) -> List[Any]:
    try:
        p = subprocess.run(["node", script], input=json.dumps(payload), capture_output=True, text=True,
                           timeout=_NODE_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TeamPackError(f"{os.path.basename(script)} could not run: {exc}") from None
    try:
        out = json.loads(p.stdout)
    except ValueError:
        raise TeamPackError(f"{os.path.basename(script)} gave no verdict (exit {p.returncode}): "
                            f"{p.stderr.strip()[-800:]}") from None
    if not isinstance(out, list) or len(out) != n:
        raise TeamPackError(f"{os.path.basename(script)} returned {type(out).__name__} for {n} rows")
    return out


def showdown_packs(pastes: Sequence[str], legacy: Optional[Sequence[Optional[str]]], fmt: str,
                   bridge_dir: str) -> List[Dict[str, Any]]:
    """``showdown_pack.js`` over ``pastes`` (with each file's legacy pack when ``legacy`` is given)."""
    rows: List[Dict[str, Any]] = []
    for i, paste in enumerate(pastes):
        row: Dict[str, Any] = {"paste": paste}
        if legacy is not None:
            row["legacy"] = legacy[i]
        rows.append(row)
    return _node(os.path.join(bridge_dir, "showdown_pack.js"), {"format": fmt, "rows": rows}, len(rows))


def oracle_defects(pastes: Sequence[str], packs: Sequence[str], fmt: str, bridge_dir: str) -> List[List[str]]:
    """``pack_oracle.js``: per file, how its pack differs from Showdown's reading of the paste (``[]`` = the same)."""
    return _node(os.path.join(bridge_dir, "pack_oracle.js"),
                 {"format": fmt, "rows": [{"paste": a, "packed": b} for a, b in zip(pastes, packs)]}, len(pastes))


def _named(failures: Sequence[Tuple[str, Sequence[str]]], what: str) -> str:
    head = "; ".join(f"{os.path.basename(f)}: {', '.join(list(d)[:3])}" for f, d in failures[:5])
    more = f" (+{len(failures) - 5} more)" if len(failures) > 5 else ""
    return f"{len(failures)} team file(s) {what} — {head}{more}"


def build(files: Sequence[str], fmt: str, bridge_dir: str,
          legacy_pack: Optional[Callable[[str], str]] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """``({abspath: packed}, record)`` for every team file the peer may draw, or :class:`TeamPackError`.

    ``legacy_pack``: the packer the peer used before (poke-env's ``join_team(parse_showdown_team(text))``); a file it
    raises on records ``None`` (a changed file). The record: ``schema``, ``n_files``, ``files_sha256`` (over the sorted
    file digests), ``changed`` (``[{file, diffs}]``) and ``peer_pack_changed``."""
    paths = sorted({os.path.abspath(f) for f in files})
    if not paths:
        raise TeamPackError("no team files to pack")
    texts = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            texts.append(fh.read())
    legacy: Optional[List[Optional[str]]] = None
    if legacy_pack is not None:
        legacy = []
        for t in texts:
            try:
                legacy.append(legacy_pack(t))
            except Exception:                         # noqa: BLE001 - a file the old packer crashed on is a CHANGE
                legacy.append(None)
    got = showdown_packs(texts, legacy, fmt, bridge_dir)
    refused = [(p, g.get("errors") or ["no pack"]) for p, g in zip(paths, got) if not g.get("packed")]
    if refused:
        raise TeamPackError(_named(refused, "refused by Showdown's validator"))
    packs = [str(g["packed"]) for g in got]
    bad = [(p, d) for p, d in zip(paths, oracle_defects(texts, packs, fmt, bridge_dir)) if d]
    if bad:
        raise TeamPackError(_named(bad, "packed other than Showdown reads them (gen3_peer_showdown_pack_v1)"))
    changed = [{"file": os.path.basename(p), "diffs": list(g.get("legacy_diffs") or [])}
               for p, g in zip(paths, got) if g.get("legacy_diffs")]
    digest = hashlib.sha256("".join(sorted(hashlib.sha256(t.encode()).hexdigest() for t in texts)).encode()).hexdigest()
    record = {"schema": PEER_PACK_SCHEMA, "n_files": len(paths), "files_sha256": digest,
              "legacy_compared": legacy is not None, "changed": changed,
              "peer_pack_changed": bool(changed) if legacy is not None else None}
    return dict(zip(paths, packs)), record


def pack_of(packs: Dict[str, str], path: str) -> str:
    """The verified pack of ``path``, or :class:`TeamPackError`: a file not checked at startup never plays."""
    key = os.path.abspath(path)
    if key not in packs:
        raise TeamPackError(f"team file {path} was not packed and verified at startup")
    return packs[key]
