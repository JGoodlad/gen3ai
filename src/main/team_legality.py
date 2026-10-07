"""Validate every team pool we play against the gen3ou FORMAT SPEC — a REPORT, never an edit.

The pools: the TRAINING pool (every ``data/teams/**/teams.json`` entry not marked ``valid: false``, what
``utils.team_loader.TeamLoader`` serves — the eval bots draw from it too), split into the 32 Smogon SAMPLE
teams (the ladder campaign's), the 40 PROMOTED fleet teams and the ``others/`` dumps; and the SPECIALIST bot
teams (``data/teams/specialist/*.txt``). Each team goes through ``agents.gen3_data.team_legality``
(bans, per-set combos, Accuracy Trap, Species Clause, One Boost Passer, Speed Pass — the LADDER's rules,
``ladder_only`` marking what only master rejects). Showdown's own validator owns the rest of
``Obtainable`` (learnsets, EVs).

    python -m main.team_legality                 # summary + every illegal team
    python -m main.team_legality --json out.json # the full per-team record

Exit 0 always when the scan ran (illegal teams are a REPORT, not a failure); 2 = no teams found.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from agents.gen3_data import team_legality as tl
from utils.paths import repo_path

TEAMS_ROOT = repo_path("data", "teams")


@dataclass
class TeamRecord:
    path: str
    pool: str                 # smogon_sample | promoted | others/<source> | specialist
    in_training: bool         # TeamLoader serves it
    problems: List[Dict[str, object]] = field(default_factory=list)

    @property
    def illegal(self) -> bool:
        return bool(self.problems)

    @property
    def ladder_only(self) -> bool:
        return self.illegal and all(bool(p["ladder_only"]) for p in self.problems)


def _records(root: Path) -> List[TeamRecord]:
    out: List[TeamRecord] = []
    data_dir = root.parent
    for manifest in sorted(root.rglob("teams.json")):
        rel_dir = manifest.parent.relative_to(root)
        for entry in json.loads(manifest.read_text()):
            rel = entry.get("file")
            if not rel:
                continue
            if rel_dir.parts[:1] == ("sample",):
                pool = "promoted" if entry.get("promoted") else "smogon_sample"
            else:
                pool = "/".join(rel_dir.parts)
            out.append(TeamRecord(str(data_dir / rel), pool, entry.get("valid") is not False))
    for p in sorted((root / "specialist").glob("*.txt")):
        out.append(TeamRecord(str(p), "specialist", False))
    return out


def scan(root: Path = TEAMS_ROOT) -> List[TeamRecord]:
    recs = _records(root)
    for r in recs:
        p = Path(r.path)
        if not p.exists():
            r.problems.append({"rule": "missing file", "slot": -1, "message": f"{p} not found",
                               "ladder_only": False})
            continue
        r.problems = [asdict(x) for x in tl.validate_team(tl.parse_paste(p.read_text(encoding="utf-8")))]
    return recs


def report(recs: Sequence[TeamRecord]) -> str:
    lines: List[str] = []
    pools = Counter(r.pool for r in recs)
    bad = Counter(r.pool for r in recs if r.illegal)
    lad = Counter(r.pool for r in recs if r.ladder_only)
    lines.append(f"{'pool':<22}{'teams':>7}{'illegal':>9}{'ladder-only':>13}")
    for pool in sorted(pools):
        lines.append(f"{pool:<22}{pools[pool]:>7}{bad[pool]:>9}{lad[pool]:>13}")
    tr = [r for r in recs if r.in_training]
    lines.append(f"{'TRAINING POOL':<22}{len(tr):>7}{sum(r.illegal for r in tr):>9}"
                 f"{sum(r.ladder_only for r in tr):>13}")
    rules = Counter(str(p["rule"]) for r in recs for p in r.problems)
    lines.append("problems by rule: " + ", ".join(f"{k} {v}" for k, v in rules.most_common()))
    lines.append("")
    for r in recs:
        if r.illegal:
            msgs = "; ".join(f"[{p['rule']}{' LADDER-ONLY' if p['ladder_only'] else ''}] {p['message']}"
                             for p in r.problems)
            lines.append(f"  {r.pool:<20} {Path(r.path).relative_to(TEAMS_ROOT.parent.parent)}: {msgs}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", type=Path, default=None, help="write the per-team record here")
    args = ap.parse_args(argv)
    recs = scan()
    if not recs:
        print("[teams] no teams found under data/teams", file=sys.stderr)
        return 2
    print(report(recs))
    if args.json is not None:
        args.json.write_text(json.dumps([{**asdict(r), "illegal": r.illegal, "ladder_only": r.ladder_only}
                                         for r in recs], indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
