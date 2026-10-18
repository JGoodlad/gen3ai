"""The pinned Showdown's verdict on every Hidden Power set of a list of pastes (``hp_set_check.js``).

``check_teams(teams)`` returns, per team, the validator's verdict and per set the Hidden Power type and power
``dex.getHiddenPower`` gives the set's IVs under gen 3, the IVs after the validator ran (it rewrites them for a
typed HP with all-31 IVs, ``sim/team-validator.ts`` ~1154-1164), the type it settled on, and the battle's stored
stats. A Node failure RAISES (a dev / test oracle, never a silent "invalid").
"""
from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List

from utils.paths import src_path

SCRIPT = src_path("utils", "bridge", "hp_set_check.js")


def check_teams(teams: List[str], format_id: str = "gen3ou") -> List[Dict[str, Any]]:
    p = subprocess.run(["node", str(SCRIPT)], input=json.dumps({"format": format_id, "teams": teams}),
                       capture_output=True, text=True, check=False)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(f"hp_set_check.js failed (exit {p.returncode}): {p.stderr.strip()[-2000:]}")
    out = json.loads(p.stdout)
    if len(out) != len(teams):
        raise RuntimeError(f"hp_set_check.js returned {len(out)} verdicts for {len(teams)} teams")
    return out
