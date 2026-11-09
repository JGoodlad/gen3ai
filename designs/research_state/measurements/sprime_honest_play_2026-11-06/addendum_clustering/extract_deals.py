"""S'-HONEST clustering addendum, step 1: one row per registered deal, with BOTH sides' teams.

The record's game tuples carry our team but not the opponent's. Kakuna's draw is in each one-game unit's
`<half>/peer_{acceptor,challenger}_report.json` (`team_draws`), so this script joins the record's deals to those reports. Frozen
record script, never a tool.

    python3 extract_deals.py ~/gen3ai_archive/sprime_honest_play_2026-11-06 > deals.csv

Columns: panel, unit_seed, game, our_team, their_team, raw, searched. An arm's cell is `win` / `loss` / `tie`, or
`missing` for a unit that exhausted its attempts (3 of 10,200). Refuses (assert) when a deal's two arms disagree on
either side's team, when a unit does not hold exactly one game, or when a counted unit has no peer report.
"""
import glob
import json
import os
import sys

PANELS = ("heldout72_hpfix", "forum72", "p72")


def their_team(root, label, unit):
    reports = sorted(glob.glob(os.path.join(root, "anchor", "u", label, "cells", unit, "*", "peer_*_report.json")))
    draws = []
    for path in reports:
        draws += json.load(open(path)).get("team_draws") or []
    return draws


def main(root):
    rec = json.load(open(os.path.join(root, "record.json")))
    units = {u["unit"]: u for u in rec["inputs"]["units"]}
    rows = []
    for d in rec["inputs"]["deals"]:
        cells, ours, theirs = {}, set(), set()
        for arm, label in (("raw", "pstar"), ("searched", "pstar_s")):
            name, status = d[arm]["unit"], d[arm]["status"]
            assert status in ("counted", "exhausted"), d
            if status != "counted":
                cells[arm] = "missing"
                continue
            u = units[name]
            assert u["counted"] and u["n"] == 1 and len(u["games"]) == 1, name
            _half, _index, our, result, _decisions = u["games"][0]
            assert result in ("win", "loss", "tie"), (name, result)
            draws = their_team(root, label, name)
            assert len(draws) == 1, (name, draws)
            cells[arm] = result
            ours.add(our)
            theirs.add(draws[0])
        assert len(ours) == 1 and len(theirs) == 1, (d["deal"], ours, theirs)   # one deal, one pair of teams
        rows.append((PANELS.index(d["panel"]), d["unit_seed"], d["game"], d["panel"], ours.pop(), theirs.pop(),
                     cells["raw"], cells["searched"]))
    rows.sort()
    assert len(rows) == 5100 and len({r[:3] for r in rows}) == 5100
    print("panel,unit_seed,game,our_team,their_team,raw,searched")
    for _, seed, game, panel, our, their, raw, searched in rows:
        print(f"{panel},{seed},{game},{our},{their},{raw},{searched}")


if __name__ == "__main__":
    main(sys.argv[1])
