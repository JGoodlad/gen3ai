"""TSR — export the read's game rows (committed beside it): rows_x26.jsonl (6,000 games, two per mirrored pair, from
the RATED team's side) and rows_kakuna.jsonl (pilot B's counting games, from side A = the subset team).

  PYTHONPATH=<checkout>/src python export_rows.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402

from main.team_strength import cli  # noqa: E402
from main.team_strength import register as RG  # noqa: E402
from utils.paths import repo_path  # noqa: E402


def main() -> int:
    reg = RG.load(analyze.HERE / "tsr_registration.json")
    pts = {"WIN": 1.0, "DRAW": 0.5, "LOSS": 0.0}
    with open(analyze.HERE / "rows_x26.jsonl", "w") as f:
        for p in sorted((analyze.STATE / "x26" / "units").glob("x26_*.json")):
            rec = json.loads(p.read_text())
            for row in rec["pairs"]:
                for gi, g in enumerate(row["games"]):
                    s = pts[g["result"]] if gi == 0 else 1.0 - pts[g["result"]]
                    f.write(json.dumps({"unit": rec["unit"], "k": row["k"], "game": gi, "subject": row["subject"],
                                        "ref": row["ref"], "subject_seat": "p1" if gi == 0 else "p2", "s": s,
                                        "end_turn": g["end_turn"], "near_ties_wide": g.get("near_ties_wide", 0)})
                            + "\n")
    kk = cli.load_kakuna(reg, analyze.STATE, repo_path("deps", "pokemon-showdown", "dist", "sim"))
    with open(analyze.HERE / "rows_kakuna.jsonl", "w") as f:
        for g in kk["games"]:
            f.write(json.dumps({"a": g.a, "b": g.b, "s": g.s, "cluster": g.cluster}) + "\n")
    print(f"x26 rows written; kakuna {len(kk['games'])} games, {kk['self_games']} self games dropped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
