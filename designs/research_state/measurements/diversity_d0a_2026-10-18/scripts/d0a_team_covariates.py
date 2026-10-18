"""D0a — a ZERO-GAME, POST-HOC, DESCRIPTIVE read of the three Kakuna team sets (2026-10-18).

For `design_diversity_teams_opponents.md` §2. It plays nothing. It reads:
  * the 719-team training pool (this checkout's `data/teams`, read-only, via `TeamLoader` with an absolute base);
  * the three read sets: home72 (`n0_endofrun_2026-09-27/teams_home72`), away (Metamon's `competitive/gen3ou`,
    read-only from the Metamon cache), heldout72 (`heldout_teams_kakuna_2026-10-18/teams_heldout72`);
  * the banked game rows of the two 2026-10-17/18 reads (X26 @161,000,169 and K2, greedy vs Kakuna greedy).

Per read set it reports ENTITY COVERAGE against the pool (species / moves / items / abilities that no pool team
carries), mon-set overlap, Hidden Power Dark counts, the rule-based archetype mix and game length. Per our PILOTED
team it reports covariates and splits each model's win rate by them (Wilson 95 %, games not independent within a
team: the intervals are optimistic, and nothing here is a verdict).

Only OUR side's team is used. The peer's per-game team is NOT recoverable reliably from the banked artifacts
(FINDING in the design doc: the peer CSV's `Team File` column does not align one-to-one with its `team_draws`).

Usage (rule 5b, CPU only):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src nice -n 19 taskset -c 5-7,13-15 \
    python designs/research_state/measurements/diversity_d0a_2026-10-18/scripts/d0a_team_covariates.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from main.anchors import team_overlap as TO
from agents.training.team_archetypes import classify_team
from utils.paths import repo_path
from utils.team_loader import TeamLoader

MEAS = repo_path("designs", "research_state", "measurements")
OUT = MEAS / "diversity_d0a_2026-10-18" / "d0a.json"
SETS = {
    "home72": MEAS / "n0_endofrun_2026-09-27" / "teams_home72",
    "away": Path(os.path.expanduser("~/dev/metamon/cache/teams/competitive/gen3ou")),
    "heldout72": MEAS / "heldout_teams_kakuna_2026-10-18" / "teams_heldout72",
}
ROWS = [MEAS / "k2_x26_kakuna_both_teamsets_2026-10-17" / "rows.jsonl",
        MEAS / "heldout_teams_kakuna_2026-10-18" / "rows.jsonl"]
CELL_SET = {"X26_greedy_h72": ("X26", "home72"), "K2_greedy_h72": ("K2", "home72"),
            "X26_greedy_away": ("X26", "away"), "K2_greedy_away": ("K2", "away"),
            "X26_greedy_hl72": ("X26", "heldout72"), "K2_greedy_hl72": ("K2", "heldout72")}


def wilson(k: float, n: int, z: float = 1.959964) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def read_dir(d: Path) -> dict:
    out = {}
    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix in (".txt", ".gen3ou_team"):
            out[f.name] = f.read_text()
    if not out:
        raise SystemExit(f"no teams in {d}")
    return out


def main() -> int:
    pool_texts = TeamLoader(base_dir=str(repo_path("data", "teams"))).get_all_teams()
    if len(pool_texts) != 719:
        raise SystemExit(f"expected the 719-team pool, got {len(pool_texts)}")
    pool = [TO.parse_team(t, f"pool_{i}") for i, t in enumerate(pool_texts)]
    ref = TO.Reference(pool)
    pool_species = {m.species for t in pool for m in t.mons}
    pool_moves = {mv for t in pool for m in t.mons for mv in m.moves}
    pool_items = {m.item for t in pool for m in t.mons}
    pool_abilities = {m.ability for t in pool for m in t.mons}
    pool_species_moves = {(m.species, mv) for t in pool for m in t.mons for mv in m.moves}

    team_cov = {}      # (set, file name stem) -> covariates
    set_summary = {}
    for sname, d in SETS.items():
        texts = read_dir(d)
        rows = []
        for fname, text in texts.items():
            t = TO.parse_team(text, fname)
            shared, _ = ref.max_shared(t)
            arch = classify_team(text)
            cov = {
                "n_mons": len(t.mons),
                "unseen_species": sum(1 for m in t.mons if m.species not in pool_species),
                "unseen_moves": sum(1 for m in t.mons for mv in m.moves if mv not in pool_moves),
                "unseen_species_move_pairs": sum(1 for m in t.mons for mv in m.moves
                                                 if (m.species, mv) not in pool_species_moves),
                "unseen_species_move_pairs_non_hp": sum(1 for m in t.mons for mv in m.moves
                                                        if (m.species, mv) not in pool_species_moves
                                                        and not mv.startswith("hiddenpower")),
                "unseen_items": sum(1 for m in t.mons if m.item and m.item not in pool_items),
                "unseen_abilities": sum(1 for m in t.mons if m.ability and m.ability not in pool_abilities),
                "mon_sets_seen": sum(1 for m in t.mons if m.key in ref.mon_index),
                "max_shared_with_one_pool_team": shared,
                "same_species6_as_a_pool_team": bool(ref.species6[t.species6]),
                "hp_dark_mons": sum(1 for m in t.mons for mv in m.moves if mv == "hiddenpowerdark"),
                "archetype": arch.get("archetype"),
            }
            team_cov[(sname, Path(fname).stem)] = cov
            rows.append(cov)
        n = len(rows)
        mons = sum(r["n_mons"] for r in rows)
        set_summary[sname] = {
            "n_teams": n, "n_mons": mons,
            "mons_with_unseen_species": sum(r["unseen_species"] for r in rows),
            "move_slots_with_a_move_no_pool_mon_carries": sum(r["unseen_moves"] for r in rows),
            "species_move_pairs_absent_from_pool": sum(r["unseen_species_move_pairs"] for r in rows),
            "of_which_not_hidden_power": sum(r["unseen_species_move_pairs_non_hp"] for r in rows),
            "mons_with_unseen_item": sum(r["unseen_items"] for r in rows),
            "mons_with_unseen_ability": sum(r["unseen_abilities"] for r in rows),
            "teams_with_any_unseen_entity": sum(1 for r in rows if r["unseen_species"] or r["unseen_moves"]
                                                or r["unseen_items"] or r["unseen_abilities"]),
            "mon_sets_seen": sum(r["mon_sets_seen"] for r in rows),
            "same_species6_teams": sum(1 for r in rows if r["same_species6_as_a_pool_team"]),
            "max_shared_hist": dict(sorted(Counter(r["max_shared_with_one_pool_team"] for r in rows).items())),
            "hp_dark_mons": sum(r["hp_dark_mons"] for r in rows),
            "archetype_mix": dict(sorted(Counter(r["archetype"] for r in rows).items())),
        }

    # pool archetype mix, for reference
    pool_arch = Counter(classify_team(t).get("archetype") for t in pool_texts)
    set_summary["pool719"] = {"n_teams": len(pool_texts), "archetype_mix": dict(sorted(pool_arch.items()))}

    # game rows: split each (model, set) win rate by our team's covariates
    games = defaultdict(list)
    for f in ROWS:
        for line in open(f):
            r = json.loads(line)
            cell = r.get("read_cell")
            if cell not in CELL_SET:
                continue
            model, sname = CELL_SET[cell]
            stem = Path(r["our_team"]).stem
            cov = team_cov.get((sname, stem))
            if cov is None:
                raise SystemExit(f"{cell}: our team {r['our_team']} not in the {sname} set")
            score = 1.0 if r["result"] == "win" else (0.5 if r["result"] == "tie" else 0.0)
            games[(model, sname)].append((score, int(r["turns"]), cov))

    def split(rows, key, bins):
        out = {}
        for label, pred in bins:
            sel = [s for s, _, c in rows if pred(c[key])]
            k = sum(sel)
            out[label] = {"games": len(sel), "score": round(k / len(sel), 4) if sel else None,
                          "wilson95": wilson(k, len(sel))}
        return out

    per_cell = {}
    for (model, sname), rows in sorted(games.items()):
        turns = sorted(t for _, t, _ in rows)
        cell = {
            "games": len(rows), "score": round(sum(s for s, _, _ in rows) / len(rows), 4),
            "median_turns": turns[len(turns) // 2],
            "by_max_shared_with_one_pool_team": split(rows, "max_shared_with_one_pool_team",
                                                      [("0", lambda v: v == 0), ("1", lambda v: v == 1),
                                                       ("2-3", lambda v: 2 <= v <= 3), ("4-6", lambda v: v >= 4)]),
            "by_same_species6": split(rows, "same_species6_as_a_pool_team",
                                      [("yes", lambda v: v), ("no", lambda v: not v)]),
            "by_unseen_species": split(rows, "unseen_species",
                                       [("0", lambda v: v == 0), (">=1", lambda v: v >= 1)]),
            "by_unseen_species_move_pairs": split(rows, "unseen_species_move_pairs",
                                                  [("0-2", lambda v: v <= 2), ("3-5", lambda v: 3 <= v <= 5),
                                                   (">=6", lambda v: v >= 6)]),
            "by_unseen_species_move_pairs_non_hp": split(rows, "unseen_species_move_pairs_non_hp",
                                                         [("0-1", lambda v: v <= 1), ("2-3", lambda v: 2 <= v <= 3),
                                                          (">=4", lambda v: v >= 4)]),
            "by_hp_dark_mons": split(rows, "hp_dark_mons",
                                     [("0-1", lambda v: v <= 1), ("2", lambda v: v == 2), (">=3", lambda v: v >= 3)]),
            "by_archetype": split(rows, "archetype", [(a, (lambda a: lambda v: v == a)(a))
                                                     for a in ("hyper_offense", "offense", "balance",
                                                               "semi_stall", "stall")]),
        }
        per_cell[f"{model}|{sname}"] = cell

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": "diversity_d0a_v1", "post_hoc": True, "descriptive_only": True,
           "pool_teams": len(pool_texts), "matcher": TO.OVERLAP_VERSION,
           "note": "our piloted team only; Wilson intervals treat games as independent (optimistic: teams repeat)",
           "sets": set_summary, "cells": per_cell}
    OUT.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    print(json.dumps(doc["sets"], indent=1))
    for k, v in per_cell.items():
        print(k, v["games"], v["score"], "median turns", v["median_turns"])
        for dim in ("by_max_shared_with_one_pool_team", "by_same_species6", "by_unseen_species",
                    "by_unseen_species_move_pairs", "by_unseen_species_move_pairs_non_hp", "by_hp_dark_mons",
                    "by_archetype"):
            print("   ", dim, {b: (x["games"], x["score"]) for b, x in v[dim].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
