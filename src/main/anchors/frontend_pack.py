"""Does the front end's PLAYED-TEAM rule change what Foul Play plays? (``gen3_frontend_validated_pack_v1``, SOP H22.)

Until 2026-10-27 ``utils.bridge.ws_frontend`` validated a ``/utm`` pack and then played the RAW one; a real Showdown
server plays the validator's re-pack (``server/ladders.ts``: ``valResult.slice(1)``). Foul Play packs each team file with
its own ``export_to_packed``, which writes no Hidden Power IVs, so a ``Hidden Power: <Type>`` mon was played at the
all-31 IV type (Dark) where the validator writes the named type's IVs. Measured on Foul Play's actual team sources
(2026-10-27): 11 of 72 files of its ``gen3ai_pool`` and the one validator-accepted ``away`` file with a Hidden Power
(``team_23``'s Gyarados, Dark 70 -> Flying 70) play another set under the validator's re-pack. So a Foul Play cell's play
CHANGED, and the cell stamps it: ``frontend_pack_changed`` is True iff any file Foul Play may draw (its ``load_team`` walk
of the team directory) that the validator ACCEPTS plays a different set re-packed than as Foul Play packed it
(``showdown_pack.js``'s field-for-field ``legacy_diffs``). It enters the regime id only when True
(``main.reads.assemble.peer_pack_identity``), so a Foul Play cell whose files play identically keeps its banked id, and an
import never crosses the change (``main.reads.imports.check_peer_pack``).

A file the validator REFUSES never plays, before or after (a real server refuses it too), so it is recorded in
``refused`` and is not a change. Fail closed: a record that cannot be computed is ``frontend_pack_changed = None`` and
voids the regime verification (``FoulPlayPeer.read_report``), never a silent "unchanged".
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.paths import src_path

FRONTEND_PACK_SCHEMA = "gen3_frontend_validated_pack_v1"
_TIMEOUT_S = 600
_CACHE: Dict[str, Dict[str, Any]] = {}


class FrontendPackError(RuntimeError):
    """The record could not be computed (Foul Play's interpreter, its packer, or Showdown's scripts failed)."""


def fp_team_files(team_dir: Path) -> List[str]:
    """Every file Foul Play's ``load_team`` may draw from ``team_dir``: the walk it does (hidden dirs and files skipped);
    a plain file is its own single team."""
    if team_dir.is_file():
        return [str(team_dir.resolve())]
    files: List[str] = []
    for root, dirs, names in os.walk(team_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        files.extend(os.path.join(root, f) for f in sorted(names)
                     if not f.startswith(".") and os.path.isfile(os.path.join(root, f)))
    return files


def foul_play_packs(python: str, checkout: Path, files: List[str]) -> Dict[str, Optional[str]]:
    """``{file: Foul Play's own pack | None}`` from Foul Play's interpreter (``peer_scripts/fp_pack_dump.py``)."""
    script = str(Path(__file__).parent / "peer_scripts" / "fp_pack_dump.py")
    try:
        p = subprocess.run([python, script], input=json.dumps({"checkout": str(checkout), "files": files}),
                           capture_output=True, text=True, timeout=_TIMEOUT_S, check=False,
                           env={**os.environ, "PYTHONPATH": ""})
        out = json.loads(p.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise FrontendPackError(f"Foul Play's packer could not run ({python}): {exc}") from None
    if not isinstance(out, dict) or set(out) != set(files):
        raise FrontendPackError(f"Foul Play's packer returned {type(out).__name__} for {len(files)} files: "
                                f"{p.stderr.strip()[-400:]}")
    return out


def build(files: List[str], raw: Dict[str, Optional[str]], fmt: str = "gen3ou",
          bridge_dir: Optional[str] = None) -> Dict[str, Any]:
    """The record for ``files`` whose Foul Play packs are ``raw``. Validator verdicts come from
    ``utils.bridge.team_validator`` (the front end's own); the played-set comparison from ``showdown_pack.js``."""
    from main.anchors.peer_scripts.team_pack import showdown_packs
    from utils.bridge.team_validator import validate_teams_locally

    bridge = bridge_dir or str(src_path("utils", "bridge"))
    packable = [f for f in files if raw.get(f)]
    verdicts = validate_teams_locally(fmt, [str(raw[f]) for f in packable]) if packable else []
    accepted = [f for f, v in zip(packable, verdicts) if v.get("valid") and v.get("packed")]
    refused = sorted(set(files) - set(accepted))
    rows = showdown_packs([str(raw[f]) for f in accepted], [str(raw[f]) for f in accepted], fmt, bridge) if accepted else []
    changed = [{"file": os.path.basename(f), "diffs": list(r.get("legacy_diffs") or [])}
               for f, r in zip(accepted, rows) if r.get("legacy_diffs")]
    digest = hashlib.sha256("".join(sorted(hashlib.sha256(str(raw.get(f)).encode()).hexdigest()
                                           for f in files)).encode()).hexdigest()
    return {"schema": FRONTEND_PACK_SCHEMA, "n_files": len(files), "n_accepted": len(accepted),
            "refused": [os.path.basename(f) for f in refused], "packs_sha256": digest,
            "changed": changed, "frontend_pack_changed": bool(changed)}


def cell_record(python: str, checkout: Path, team_dir: Path, fmt: str = "gen3ou") -> Dict[str, Any]:
    """The record for the files Foul Play may draw from ``team_dir`` (memoised per process by the files' content)."""
    files = fp_team_files(team_dir)
    if not files:
        raise FrontendPackError(f"no team files under {team_dir}")
    key = hashlib.sha256((str(python) + str(checkout) + "".join(
        f + Path(f).read_text(errors="replace") for f in files)).encode()).hexdigest()
    if key not in _CACHE:
        _CACHE[key] = build(files, foul_play_packs(python, checkout, files), fmt)
    return _CACHE[key]
