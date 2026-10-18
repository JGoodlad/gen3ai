"""Pool hq_hpfix.py's done units into rows.jsonl + cells.json, and set heldout72_hpfix beside heldout72 (HP Dark),
home72 and away. `heldout_teams_kakuna_2026-10-18/scripts/aggregate.py` with the cells renamed (hl72 -> hl72hp), the
heldout72 cells added to the comparisons, and the overlap computed on heldout72_hpfix. Its docstring:

The 2026-10-17 `aggregate.py` with three changes: the units are hq.py's (X26 and K2 on heldout72, paired seeds 0-19);
the difference is named `heldout72`; and cells.json adds (a) the 10-17 read's home72 / away cells for comparison, with
Newcombe intervals of heldout72 - home72 and heldout72 - away per model, and (b) the team-set overlap of heldout72
(`main.anchors.team_overlap`, the CURRENT module: the pinned trees predate it, so their summaries carry no block).

A unit COUNTS iff status OK, n == 20, regime_verified_decisions, no team_source_asymmetry AND peer_clean.

    PYTHONPATH=<checkout>/src python aggregate.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "src"))
from main.anchors.results import newcombe, wilson  # noqa: E402

sys.path.insert(0, str(HERE))
import hq_hpfix as hq  # noqa: E402

OUT = HERE.parent
PREV = OUT.parent / "k2_x26_kakuna_both_teamsets_2026-10-17" / "cells.json"
PREV72 = OUT.parent / "heldout_teams_kakuna_2026-10-18" / "cells.json"
GAMES = 20
STAMP = ("opponent", "opponent_version", "opponent_commit", "our_regime", "their_regime",
         "regime_matched", "teamset", "our_team_source", "our_team_count", "their_team_count",
         "battle_format", "model_zip", "model_step", "model_loader", "forfeit_turn_limit",
         "showdown_pin", "server_impl", "server_version", "our_side", "our_transport",
         "mirrored_pairs")

SOURCES = [(cell, unit, seed, hq.STATE / "cells" / unit, "heldout72_hpfix_kakuna_2026-10-18")
           for unit, cell, seed in hq.units()]


def verdict(lo, hi):
    return "BETTER" if lo > 0.5 else ("WORSE" if hi < 0.5 else "NOT DETECTED")


def main():
    rows, cells, skipped = [], {}, []
    for cell, unit, seed, d, read in SOURCES:
        sp = d / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        ok = (s.get("status") == "OK" and s.get("n") == GAMES
              and s.get("regime_verified_decisions") is True
              and s.get("team_source_asymmetry") is False
              and s.get("peer_clean") is True)
        if not ok:
            skipped.append({"cell": cell, "unit": unit, "read": read, "status": s.get("status"),
                            "n": s.get("n"), "regime_verified_decisions": s.get("regime_verified_decisions"),
                            "team_source_asymmetry": s.get("team_source_asymmetry"),
                            "peer_clean": s.get("peer_clean"), "peer_exit_notes": s.get("peer_exit_notes")})
            continue
        c = cells.setdefault(cell, {"units": [], "seeds": [], "wins": 0, "losses": 0, "ties": 0, "n": 0,
                                    "hit_forfeit_limit": 0, "their_argmax": [], "turns": [], "teams": set(),
                                    "stamps": {}, "reads": set(), "by_half": {}})
        c["units"].append(unit)
        c["seeds"].append(seed)
        c["reads"].add(read)
        for k in ("wins", "losses", "ties", "n", "hit_forfeit_limit"):
            c[k] += s[k]
        c["their_argmax"] += s.get("their_argmax_match_rates") or []
        for line in (d / "games.jsonl").read_text().splitlines():
            r = json.loads(line)
            r.update({"read_cell": cell, "unit": unit, "unit_seed": seed, "source_read": read})
            rows.append(r)
            c["turns"].append(r["turns"])
            c["teams"].add(r["our_team"])
            h = c["by_half"].setdefault(r["half"], [0, 0])
            h[0] += bool(r["won"])
            h[1] += 1
            st = json.dumps({k: r.get(k) for k in STAMP}, sort_keys=True)
            c["stamps"][st] = c["stamps"].get(st, 0) + 1
    out = {}
    for k, c in cells.items():
        p, lo, hi = wilson(c["wins"], c["n"])
        t = sorted(c["turns"])
        out[k] = {
            "units": len(c["units"]), "seeds": sorted(c["seeds"]), "reads": sorted(c["reads"]),
            "n": c["n"], "wins": c["wins"], "losses": c["losses"], "ties": c["ties"],
            "win_rate": p, "wilson95": [lo, hi], "vs_0.50": verdict(lo, hi),
            "by_half": {h: {"wins": w, "n": n, "rate": w / n} for h, (w, n) in sorted(c["by_half"].items())},
            "hit_forfeit_limit": c["hit_forfeit_limit"],
            "their_argmax_min": min(c["their_argmax"]) if c["their_argmax"] else None,
            "their_argmax_max": max(c["their_argmax"]) if c["their_argmax"] else None,
            "median_turns": t[len(t) // 2] if t else None, "max_turns": t[-1] if t else None,
            "distinct_our_teams": len(c["teams"]),
            "regime_stamps": [{"rows": n, **json.loads(st)} for st, n in c["stamps"].items()],
            "stamp_identical": len(c["stamps"]) == 1,
        }
    diffs = {}
    for ts, (x, k) in {"heldout72_hpfix": ("X26_greedy_hl72hp", "K2_greedy_hl72hp")}.items():
        if x in out and k in out:
            d, dlo, dhi = newcombe(out[x]["wins"], out[x]["n"], out[k]["wins"], out[k]["n"])
            diffs[ts] = {"x26": [out[x]["wins"], out[x]["n"]], "k2": [out[k]["wins"], out[k]["n"]],
                         "x26_minus_k2": [d, dlo, dhi],
                         "verdict": "X26 BETTER" if dlo > 0 else ("X26 WORSE" if dhi < 0 else "NOT DETECTED")}
    prev = json.loads(PREV.read_text())["cells"]
    prev72 = json.loads(PREV72.read_text())
    prev.update(prev72["cells"])
    vs_prev = {}
    for model, cell in (("X26", "X26_greedy_hl72hp"), ("K2", "K2_greedy_hl72hp")):
        if cell not in out:
            continue
        for ts, pc in (("heldout72", f"{model}_greedy_hl72"), ("home72", f"{model}_greedy_h72"),
                       ("away", f"{model}_greedy_away")):
            p = prev[pc]
            d, dlo, dhi = newcombe(out[cell]["wins"], out[cell]["n"], p["wins"], p["n"])
            vs_prev[f"{model}: heldout72_hpfix - {ts}"] = {
                "heldout72_hpfix": [out[cell]["wins"], out[cell]["n"]], ts: [p["wins"], p["n"], p["win_rate"], p["wilson95"]],
                "diff": [d, dlo, dhi], "verdict": "HIGHER" if dlo > 0 else ("LOWER" if dhi < 0 else "NOT DETECTED")}
    from main.anchors import team_overlap as ov
    from main.anchors.session import load_team_texts, pool_team_texts

    read = ov.parse_items(load_team_texts(hq.OUR_TEAMS))
    refs = {"training_pool": [(f"pool{i:03d}", t) for i, t in enumerate(pool_team_texts())],
            "metamon:competitive": load_team_texts(Path("/home/goodlad/dev/metamon/cache/teams/competitive/gen3ou"))}
    overlap = {k: ov.overlap(read, ov.Reference(ov.parse_items(v))) for k, v in refs.items()}
    (OUT / "cells.json").write_text(json.dumps({"cells": out, "x26_minus_k2": diffs, "vs_prior_cells": vs_prev,
                                                "team_overlap": {"version": ov.OVERLAP_VERSION, "ours": overlap},
                                                "skipped_units": skipped}, indent=1) + "\n")
    for k, v in vs_prev.items():
        d, lo, hi = v["diff"]
        print(f"{k:24s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {v['verdict']}")
    with open(OUT / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for k, v in out.items():
        print(f"{k:16s} units {v['units']:2d} n {v['n']:3d} W-L-T {v['wins']}-{v['losses']}-{v['ties']} "
              f"{v['win_rate']:.3f} [{v['wilson95'][0]:.3f}, {v['wilson95'][1]:.3f}] {v['vs_0.50']} "
              f"stamp_identical={v['stamp_identical']}")
    for ts, v in diffs.items():
        d, lo, hi = v["x26_minus_k2"]
        print(f"X26 - K2 {ts:7s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {v['verdict']}")
    print(f"skipped units: {len(skipped)}")
    for s in skipped:
        print("  ", s)


if __name__ == "__main__":
    main()
