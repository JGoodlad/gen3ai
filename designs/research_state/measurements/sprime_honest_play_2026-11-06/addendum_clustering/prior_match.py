"""How close the search's sampled worlds are to the opponent's TRUE team, by panel. A REPORTED covariate, no verdict.

    python3 prior_match.py <state dir> prior_match.json

Reads every COUNTED searched unit of ``<state dir>/record.json``: its ``search_decisions.jsonl`` and its capture's START
line (the true packed teams) under ``<state dir>/anchor/u/pstar_s/cells/``. For each searched decision and each of its worlds:

* ``species_hit``: the share of the opponent's six true species the world's team holds (the logged covariate);
* ``move_hit`` / ``item_hit`` (only where the row carries ``honest.world_teams``): over the true mons whose species
  the world holds, the share of the true set's moves the world's set has, and whether the item is equal.

A revealed fact is the truth's in every world, so these mix what play revealed with what the prior drew: read them
ACROSS panels (the reveal process is the same), and at the FIRST searched decision (turn 10) for the closest match.
The unit is the game (the mean over its decisions and worlds); the interval is a percentile bootstrap over the
opponent's TRUE team within the panel (B = 2,000, ``default_rng(20261109)``).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PANELS = ("heldout72_hpfix", "forum72", "p72")
SEED, B = 20261109, 2000


def toid(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse(team):
    """packed team -> {species id: (set of move ids, item id)}"""
    out = {}
    for e in team.split("]"):
        if not e:
            continue
        f = e.split("|")
        out[toid(f[1] or f[0])] = (frozenset(toid(m) for m in f[4].split(",") if m), toid(f[2]))
    return out


def unit_rows(cell):
    cap = json.loads((cell / "capture" / "battle-gen3ou-1.json").read_text())
    start = json.loads(cap["commands"][0][len("START "):])
    first = None
    acc = defaultdict(list)
    with open(cell / "search_decisions.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("searched"):
                continue
            theirs = start["p2" if r["side"] == "p1" else "p1"]["team"]
            truth = parse(theirs)
            assert len(truth) == 6, cell
            row = {"species_hit": float(np.mean(r["species_hit"]))}
            worlds = (r.get("honest") or {}).get("world_teams")
            if worlds:
                mh, ih = [], []
                for w in worlds:
                    got = parse(w)
                    both = [s for s in truth if s in got]
                    assert abs(len(both) / 6 - r["species_hit"][len(mh)]) < 1e-3, cell
                    mh.append(np.mean([len(truth[s][0] & got[s][0]) / len(truth[s][0]) for s in both]))
                    ih.append(np.mean([truth[s][1] == got[s][1] for s in both]))
                row["move_hit"], row["item_hit"] = float(np.mean(mh)), float(np.mean(ih))
            for k, v in row.items():
                acc[k].append(v)
            if first is None:
                first = dict(row, their=theirs)
    if first is None:
        return None
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out.update({f"first_{k}": v for k, v in first.items() if k != "their"})
    out["their"], out["n"] = first["their"], len(acc["species_hit"])
    return out


def main(state, out_path):
    cells = Path(state) / "anchor" / "u" / "pstar_s" / "cells"
    units = json.loads((Path(state) / "record.json").read_text())["inputs"]["units"]
    per, no_search = defaultdict(list), defaultdict(int)
    for u in sorted((u for u in units if u["label"] == "pstar_s" and u["counted"]), key=lambda u: u["unit"]):
        row = unit_rows(cells / u["unit"])                # the record's COUNTED searched units, and only those
        if row is None:
            no_search[u["panel"]] += 1                    # the game ended before its first searched decision
        else:
            per[u["panel"]].append(row)
    out = {"state": str(state), "seed": SEED, "B": B, "panels": {}}
    for pi, p in enumerate(PANELS):
        rows = per[p]
        ids = np.unique([r["their"] for r in rows], return_inverse=True)[1]
        res = {"games": len(rows), "games_without_a_searched_decision": no_search[p], "their_teams": int(ids.max() + 1),
               "searched_decisions": int(sum(r["n"] for r in rows))}
        for k in sorted({k for r in rows for k in r} - {"their", "n"}):
            x = np.array([r[k] for r in rows], dtype=float)
            g = int(ids.max() + 1)
            s, c = np.bincount(ids, weights=x, minlength=g), np.bincount(ids, minlength=g)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, pi, sorted(rows[0]).index(k)]))
            idx = rng.integers(0, g, size=(B, g))
            reps = s[idx].sum(axis=1) / c[idx].sum(axis=1)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            res[k] = {"mean": float(x.mean()), "ci95": [float(lo), float(hi)]}
        out["panels"][p] = res
    Path(out_path).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for p in PANELS:
        r = out["panels"][p]
        print(p, r["games"], r["their_teams"], r["searched_decisions"],
              {k: (round(v["mean"], 4), [round(z, 4) for z in v["ci95"]]) for k, v in r.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
