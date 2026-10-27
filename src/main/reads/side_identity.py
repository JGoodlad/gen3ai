"""WHAT EACH SIDE PLAYED WITH: the Metamon peer's code and our policy's load mode, part of every anchor regime
(``designs/tools/reads.md`` §4; TECH_DEBT P1 (b), 2026-10-27).

The regime id named the weights, the team pack and the engine, but two more things change play without any of them:

* **the Metamon peer's Python** (``cell.peer_code`` / ``our_peer_code``): the digest of what the peer EXECUTED
  (``main.anchors.peer_scripts.peer_code``: our peer scripts, the Metamon checkout it imports, its interpreter and
  packages). SOP H22 is the proof that the peer's Python moves play;
* **our policy's load mode** (``cell.policy_mode``): "train" or "eval", READ BACK from the loaded module
  (``main.anchors.core_side.policy_mode``). Every anchors loader returned a TRAINING-mode policy before ``141f36f7f``
  and an eval-mode one since; ``nn.MultiheadAttention`` runs another kernel in each (logits within 4.8e-7). The loader's
  NAME (``model_loader``: bare / foreign) was already in the id.

THE LEGACY RULE, as the weights and the engine have it. A stamp enters the id only when it is not what the banked reads
played: a peer code not in :data:`LEGACY_PEER_CODE`, a mode other than :data:`LEGACY_POLICY_MODE`. A summary written
before the stamps carries neither, so every banked id is unchanged. The ids cannot tell the legacy values apart (the N1-P
look-1 units played in eval mode with ids computed without it), so every import compares the values themselves
(:func:`require_same`), each unit's either stamped or reconstructed from its playing commit (:data:`LEGACY_COMMIT_MODE`;
a banked commit's peer is legacy). A commit outside the tables is UNKNOWN (``None``), and an import never accepts it.

THE LEGACY PEER CODE is reconstructed (:func:`peer_code_at`): the peer scripts at each banked commit (three versions; the
diffs between them add the ensemble router's no-RNG check, the preset pin and the weight pins: instruments and
refusals, none a choice a single policy makes, so the three are ONE legacy regime), the Metamon checkout every archived
Metamon summary stamps (``opponent_commit`` 0a00a759: 1,281 of 1,281 archived units, 2026-10-27; clean then, its dirty state at
earlier reads UNVERIFIED), and the ``metamon`` env as it stands (:data:`LEGACY_PEER_ENV`: conda's last change and every
site-packages file date from 2026-09-14, before the first banked read, X51 on 10-17).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from main.reads import engine as EN

SCHEMA = "gen3_read_side_identity_v1"

#: The Metamon checkout every banked Metamon unit played (its ``opponent_commit``), clean.
LEGACY_METAMON: Dict[str, str] = {"commit": "0a00a759c9a4382a2877088d828302ec294a05a5", "dirty": ""}
#: The ``metamon`` env every banked read ran (``python peer_scripts/peer_code.py`` under it, 2026-10-27).
LEGACY_PEER_ENV: Dict[str, Any] = {"python": "3.10.21 (main, Aug 27 2026, 14:42:07) [GCC 14.3.0]",
                                   "packages": "02a75679f0bdaeba9ddc875c3864dee7f3fe335a63ffeaccadfaff590c725c1e",
                                   "n_packages": 104}
#: FROZEN: the peer-code digests the banked reads played (:func:`peer_code_at` of every :data:`EN.LEGACY_ENGINE` commit),
#: each with the ``metamon_side.py`` blob it hashes. One legacy regime (the module docstring).
LEGACY_PEER_CODE: Dict[str, str] = {
    "f230353c40e43488": "metamon_side e17d4869: 63f97a575 cb0f1769c acb99527 b9c758841 f4b25238d 6c428453a 5e389ddac "
                        "e05c45eeb 4b0ab220e 62bb07c00",
    "64c220c05cca7382": "metamon_side 9f8520a1: d32ffcee4 4ee655648 033fcbcd2 2cc864f2f",
    "b817d3216c418f75": "metamon_side 9a550a80 + weight_pins 3fc9d471: 46b9e572c de15dfc7b",
}
#: The mode every banked read played in but N1-P look 1's (the id's legacy value).
LEGACY_POLICY_MODE = "train"
#: The commit since which every anchors loader returns an EVAL-mode policy.
EVAL_MODE_COMMIT = "141f36f7fba07a76b3736a8d3f30bf314f35ba37"
#: FROZEN: every banked playing commit (:data:`EN.LEGACY_ENGINE`'s) -> our policy's mode there (:func:`policy_mode_at`).
LEGACY_COMMIT_MODE: Dict[str, str] = {c: "train" for c in EN.LEGACY_ENGINE}
LEGACY_COMMIT_MODE.update({"46b9e572cc0fa237f62ddd948fb5fa4d115e44a4": "eval",       # N1-P look 1
                           "de15dfc7b6d19d6f0e4a3724f269e26e7aa22957": "eval"})      # N1-P look 1 (repin)
#: FROZEN: a banked anchor control file that names no side per row -> its sides (by sha256), as
#: :data:`EN.LEGACY_BANK_ENGINE`: D1's look-2 input and the C0 s3501 games, played at 6c428453 / 5e389ddac.
LEGACY_BANK_SIDE: Dict[str, Dict[str, str]] = {
    "8c4884aafcb5fa77a15b2e8be517460cf64f411a43a6ffa67bf02d1f4f2170b7": {"peer_code": "their=legacy",
                                                                         "policy_mode": "train"},
    "be989806098e1b06f994ce9175c81238f6926ef94f367ab74b450db2e30ba336": {"peer_code": "their=legacy",
                                                                         "policy_mode": "train"},
}
SIDE_KEYS = ("peer_code", "policy_mode")


class SideError(RuntimeError):
    """A side identity that differs, or is unknown, where one is required (named)."""


# ------------------------------------------------------------------------------------------------ the id
def regime_identity(cell: Mapping[str, Any]) -> Dict[str, Any]:
    """The side inputs of a regime id, each only when stamped and NOT legacy: ``peer_code`` / ``our_peer_code`` (a digest
    not in :data:`LEGACY_PEER_CODE`), ``policy_mode`` (a mode other than :data:`LEGACY_POLICY_MODE`). ``{}`` for a
    summary that stamps neither (every banked one) or stamps the legacy values."""
    out: Dict[str, Any] = {}
    for key in ("peer_code", "our_peer_code"):
        d = cell.get(key)
        if d and str(d) not in LEGACY_PEER_CODE:
            out[key] = str(d)
    mode = cell.get("policy_mode")
    if mode and mode != LEGACY_POLICY_MODE:
        out["policy_mode"] = str(mode)
    return out


# ------------------------------------------------------------------------------------------------ per unit
def _banked(cell: Mapping[str, Any]) -> Optional[str]:
    """The full banked commit this unit played at (``server_version``), else None."""
    c = EN.server_commit(cell.get("server_version"))
    if not c or len(c) < 7:
        return None
    hits = [k for k in EN.LEGACY_ENGINE if k.startswith(c)]
    return hits[0] if len(hits) == 1 else None


def unit_peer_code(cell: Mapping[str, Any]) -> Optional[str]:
    """One unit's Metamon peer code, ``their=<v>`` / ``our=<v>`` joined by ``;`` (``<v>``: ``legacy`` or the digest):
    ``""`` when no side is a Metamon peer, None when a side's is unknown (no stamp, and not a banked commit)."""
    sides = []
    if cell.get("opponent_kind") == "metamon":
        sides.append(("their", cell.get("peer_code")))
    if str(cell.get("our_side") or "").startswith("metamon:"):
        sides.append(("our", cell.get("our_peer_code")))
    out = []
    for name, stamped in sides:
        if stamped:
            v = "legacy" if str(stamped) in LEGACY_PEER_CODE else str(stamped)
        elif _banked(cell):
            v = "legacy"
        else:
            return None
        out.append(f"{name}={v}")
    return ";".join(out)


def unit_policy_mode(cell: Mapping[str, Any]) -> Optional[str]:
    """One unit's policy mode: the stamp, else :data:`LEGACY_COMMIT_MODE` of its banked commit; ``""`` when our side is
    not a checkpoint (a bot or a peer), None when unknown."""
    if str(cell.get("our_side") or "model") != "model":
        return ""
    if cell.get("policy_mode"):
        return str(cell["policy_mode"])
    c = _banked(cell)
    return LEGACY_COMMIT_MODE.get(c) if c else None


def unit_sides(cell: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    return {"peer_code": unit_peer_code(cell), "policy_mode": unit_policy_mode(cell)}


def one(values: Iterable[Optional[str]], what: str) -> Tuple[Optional[str], Optional[str]]:
    """``(value, why)``: the single value of a set of units, or ``(None, reason)`` when they span several or any is
    unknown beside a known one. An all-unknown set is ``(None, None)``."""
    got = set(values)
    known = sorted(v for v in got if v is not None)
    if len(known) > 1:
        return None, f"span {len(known)} {what}s {known}"
    if known and None in got:
        return None, f"mix {what} {known[0]!r} with units of an unknown {what}"
    return (known[0] if known else None), None


def group_sides(rows: Iterable[Mapping[str, Any]]) -> Tuple[Dict[str, Optional[str]], list]:
    """``({key: value}, [why])`` over the COUNTED unit rows' :data:`SIDE_KEYS`."""
    counted = [r for r in rows if r.get("counted")]
    out: Dict[str, Optional[str]] = {}
    why = []
    for key in SIDE_KEYS:
        out[key], w = one((r.get(key) for r in counted), key.replace("_", " "))
        if w:
            why.append(f"counted units {w}")
    return out, why


def require_same(what: str, here: Mapping[str, Optional[str]], there: Mapping[str, Optional[str]]) -> None:
    """Refuse (:class:`SideError`) unless both sides' peer code and policy mode are known and equal: an import across
    either pools two different players."""
    for key in SIDE_KEYS:
        a, b = here.get(key), there.get(key)
        if a is None or b is None:
            raise SideError(f"{what}: {key.replace('_', ' ')} unknown ({'this read' if a is None else 'the banked side'}); "
                            "an import needs both named (re-play the control, or register the bank's sides)")
        if a != b:
            raise SideError(f"{what}: the banked side played {key.replace('_', ' ')} {b!r}, this read {a!r}: an import "
                            "never crosses the peer's code or our policy's load mode (re-play the control here)")


def bank_sides(entry: Mapping[str, Any], rows: Iterable[Mapping[str, Any]] = ()) -> Dict[str, Optional[str]]:
    """A banked anchor control's sides: its registered ``sides``; else what its rows say (each row a Kakuna unit's game
    row: its stamps, or its banked ``server_version`` commit; every row must agree); else :data:`LEGACY_BANK_SIDE` by
    its sha256; else unknown."""
    reg = entry.get("sides")
    if isinstance(reg, Mapping):
        return {k: (str(reg[k]) if reg.get(k) is not None else None) for k in SIDE_KEYS}
    per = [unit_sides({"opponent_kind": "metamon", **r}) for r in rows
           if r.get("server_version") or r.get("policy_mode") or r.get("peer_code")]
    if per:
        out: Dict[str, Optional[str]] = {}
        for key in SIDE_KEYS:
            out[key], why = one((p[key] for p in per), key.replace("_", " "))
            if why:
                raise SideError(f"banked rows {entry.get('path')}: {why}")
        return out
    return dict(LEGACY_BANK_SIDE.get(str(entry.get("sha256") or ""), {k: None for k in SIDE_KEYS}))


# ------------------------------------------------------------------------------------------------ reconstruction
def peer_code_at(commit: str, repo: Optional[Path] = None, env: Optional[Mapping[str, Any]] = None,
                 metamon: Optional[Mapping[str, str]] = None) -> str:
    """The peer-code digest a Metamon peer launched from ``commit``'s tree would stamp, from git objects (never the
    working tree), with the Metamon checkout ``metamon`` (default :data:`LEGACY_METAMON`) and the env ``env`` (default
    :data:`LEGACY_PEER_ENV`)."""
    from main.anchors.peer_scripts import peer_code as PC

    roots = {"peer_scripts": "src/main/anchors/peer_scripts", "bridge": "src/utils/bridge"}

    def read(where: str, name: str) -> Optional[bytes]:
        r = subprocess.run(["git", "-C", str(repo or _root()), "show", f"{commit}:{roots[where]}/{name}"],
                           capture_output=True, check=False)
        return r.stdout if r.returncode == 0 else None

    e = dict(LEGACY_PEER_ENV if env is None else env)
    return PC.digest({"schema": PC.SCHEMA, "scripts": PC.hashes_from(read),
                      "metamon": dict(LEGACY_METAMON if metamon is None else metamon),
                      "python": e["python"], "packages": e["packages"]})


def policy_mode_at(commit: str, repo: Optional[Path] = None) -> str:
    """Our policy's mode at ``commit``: "eval" iff :data:`EVAL_MODE_COMMIT` is an ancestor (both anchors loaders changed
    there), else "train". A hybrid pin (D3 look 2's ``c4a5c37b6`` restores the older ``snapshot.py``) is not decided
    by ancestry: it is in no table and its units are unknown."""
    r = subprocess.run(["git", "-C", str(repo or _root()), "merge-base", "--is-ancestor", EVAL_MODE_COMMIT, commit],
                       capture_output=True, check=False)
    if r.returncode not in (0, 1):
        raise SideError(f"git merge-base {commit[:12]}: {r.stderr.decode(errors='replace').strip()[:200]}")
    return "eval" if r.returncode == 0 else "train"


def _root() -> Path:
    from utils.paths import repo_root

    return Path(repo_root())
