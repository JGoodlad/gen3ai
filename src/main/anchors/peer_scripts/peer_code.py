"""The Metamon peer's CODE identity (``gen3_peer_code_v1``): the files it executes, the Metamon checkout, its interpreter.

STDLIB ONLY, and shared by both interpreters (like ``weight_pins.py``): ``metamon_side.py`` runs under the Metamon env
(``PYTHONPATH=""``) and loads this file by path; ``main.anchors`` and ``main.reads`` import it as an ordinary module. One
implementation, so the digest the peer reports and the digest a reconstruction computes cannot drift.

THE GAP THIS CLOSES (TECH_DEBT P1 (b), 2026-10-27). The regime id named the weights, the team pack and the engine, but
not the peer's own Python: our peer scripts, the Metamon checkout (an EDITABLE install) and the env's packages (upstream
poke-env 0.8.3.3, amago, torch, ...). SOP H22 is the proof that the peer's Python moves play (poke-env's packer played
forum72's Frustration at base power 1); a pull or a package upgrade would move play again with no id change. Now the
peer reports :func:`identity` at startup, the cell stamps its :func:`digest` (``peer_code``), and
``main.reads.side_identity`` enters it in the id unless it is a LEGACY digest (what every banked read played).

What it covers:

* ``scripts``: sha256 of each file of :data:`PEER_FILES` the peer executes (this directory's, and the two bridge scripts
  it runs through Node to pack its teams, :func:`hashes_from`);
* ``metamon``: the checkout's full HEAD and, when it is not clean, a digest of its tracked diff plus every untracked
  file inside the importable ``metamon/`` package (``""`` when clean; the weights cache is pinned on its own);
* ``python``: the interpreter's ``sys.version``;
* ``packages``: sha256 over every installed distribution, ``name==version`` plus its VCS commit or ``editable``.

NOT covered: the Metamon weights (``weight_pins``), the team files (the team digests and ``peer_pack_changed``), Node
and the Showdown submodule (``showdown_pin``).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

SCHEMA = "gen3_peer_code_v1"
#: The files the peer executes, as ``<dir>/<name>``: ``peer_scripts`` is this directory, ``bridge`` is the
#: ``--bridge-dir`` the peer packs through. A file absent (an older commit) is left out.
PEER_FILES: Tuple[Tuple[str, str], ...] = (
    ("peer_scripts", "metamon_side.py"), ("peer_scripts", "peer_code.py"), ("peer_scripts", "team_pack.py"),
    ("peer_scripts", "weight_pins.py"), ("bridge", "showdown_pack.js"), ("bridge", "pack_oracle.js"))
#: The fields :func:`digest` reads (``dirty_paths`` and ``n_packages`` are recorded, not identity).
DIGEST_KEYS = ("schema", "scripts", "metamon", "python", "packages")
_GIT_TIMEOUT_S = 60


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def script_hashes(peer_scripts_dir: str, bridge_dir: str) -> Dict[str, str]:
    """``{"<dir>/<name>": sha256}`` of every :data:`PEER_FILES` file the peer executes, read from disk."""
    roots = {"peer_scripts": peer_scripts_dir, "bridge": bridge_dir}

    def read(where: str, name: str) -> Optional[bytes]:
        path = os.path.join(roots[where], name)
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as fh:
            return fh.read()

    return hashes_from(read)


def hashes_from(read: Callable[[str, str], Optional[bytes]]) -> Dict[str, str]:
    """``{"<dir>/<name>": sha256}`` over :data:`PEER_FILES`, each file's bytes from ``read(dir, name)`` (None = absent:
    a tree that predates it). The bridge scripts count only beside ``team_pack.py``: the peer runs them only through it
    (gen3_peer_showdown_pack_v1); before it, ``pack_oracle.js`` existed but the peer never ran it."""
    out: Dict[str, str] = {}
    for where, name in PEER_FILES:
        data = read(where, name)
        if data is not None:
            out[f"{where}/{name}"] = sha256_bytes(data)
    if "peer_scripts/team_pack.py" not in out:
        out = {k: v for k, v in out.items() if not k.startswith("bridge/")}
    return out


def _git(root: str, *args: str) -> bytes:
    r = subprocess.run(["git", "-C", root, *args], capture_output=True, timeout=_GIT_TIMEOUT_S, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} in {root}: {r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout


def checkout_state(root: str, package: str = "metamon") -> Dict[str, Any]:
    """The Metamon checkout: ``{"commit": <full HEAD>, "dirty": "" | sha256, "dirty_paths": n}``. ``dirty`` digests the
    tracked diff against HEAD (``git diff HEAD --binary``) and every untracked file under ``<root>/<package>/`` (what an
    import can load; ``__pycache__`` aside), each with its sha256. Raises when ``root`` is not a git checkout."""
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    diff = _git(root, "diff", "HEAD", "--binary")
    untracked = [p for p in _git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", package)
                 .decode().split("\0") if p and "__pycache__" not in p.split("/")]
    tracked = [p for p in _git(root, "diff", "HEAD", "--name-only", "-z").decode().split("\0") if p]
    if not diff and not untracked:
        return {"commit": commit, "dirty": "", "dirty_paths": 0}
    h = hashlib.sha256(diff)
    for rel in sorted(untracked):
        with open(os.path.join(root, rel), "rb") as fh:
            h.update(f"\0untracked\0{rel}\0{sha256_bytes(fh.read())}".encode())
    return {"commit": commit, "dirty": h.hexdigest(), "dirty_paths": len(tracked) + len(untracked)}


def package_lines() -> list:
    """``name==version[@<vcs commit>|@editable]`` of every installed distribution, sorted (duplicates kept)."""
    import importlib.metadata as md

    lines = []
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").strip().lower().replace("_", "-")
        line = f"{name}=={dist.version}"
        try:
            direct = json.loads(dist.read_text("direct_url.json") or "null")
        except (OSError, ValueError):
            direct = None
        if isinstance(direct, dict):
            commit = (direct.get("vcs_info") or {}).get("commit_id")
            if commit:
                line += f"@{commit}"
            elif (direct.get("dir_info") or {}).get("editable"):
                line += "@editable"
        lines.append(line)
    return sorted(lines)


def packages_digest(lines: Optional[list] = None) -> Tuple[str, int]:
    """``(sha256, n)`` over :func:`package_lines` (the running interpreter's when ``lines`` is None)."""
    lines = package_lines() if lines is None else lines
    return sha256_bytes("\n".join(lines).encode()), len(lines)


def identity(peer_scripts_dir: str, bridge_dir: str, metamon_root: str) -> Dict[str, Any]:
    """The full record the peer reports (``report["peer_code"]``). Raises if any part cannot be established: the
    caller records the error and the harness voids the unit's regime verification."""
    packages, n = packages_digest()
    return {"schema": SCHEMA, "scripts": script_hashes(peer_scripts_dir, bridge_dir),
            "metamon": checkout_state(metamon_root), "python": sys.version, "packages": packages, "n_packages": n}


def digest(record: Mapping[str, Any]) -> str:
    """The ONE id of a peer's code (16 hex): sha256 over the canonical JSON of :data:`DIGEST_KEYS`, the checkout as
    ``{commit, dirty}``."""
    m = record.get("metamon") or {}
    doc = {k: record.get(k) for k in DIGEST_KEYS}
    doc["metamon"] = {"commit": m.get("commit"), "dirty": m.get("dirty")}
    return sha256_bytes(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode())[:16]


if __name__ == "__main__":                                   # the env half, for a reconstruction: python peer_code.py
    _p, _n = packages_digest()
    print(json.dumps({"python": sys.version, "packages": _p, "n_packages": _n}))
