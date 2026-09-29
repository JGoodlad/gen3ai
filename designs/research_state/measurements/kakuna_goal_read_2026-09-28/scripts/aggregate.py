#!/usr/bin/env python3
"""Pool the DONE units of the Kakuna goal read into rows.jsonl + cells.json (this directory's parent).

A unit counts iff its summary.json is status OK with n == its games, regime_verified_decisions is
true and team_source_asymmetry is false (the N0 read's rule). Every row carries its unit, seed,
lane and the unit's box load (1-min loadavg mean/max, sampled every 5 s by the supervisor), and —
for the Foul Play cell — FP's think time.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import kq  # noqa: E402

OUT = Path(__file__).resolve().parents[1]


def f_deltas():
    """(f): greedy - sampled for N0 against the same Kakuna setting, on the SAME seeds. Newcombe's
    hybrid score interval from two independent Wilson intervals (ignores the pairing: conservative)."""
    from main.anchors.results import newcombe

    def pool(cell, seeds=None):
        w = n = 0
        for (name, c, games, seed, *_r) in kq.all_units():
            if c == cell and (seeds is None or seed in seeds) and kq.is_done(name, games):
                s = kq.summary(name); w += s["wins"]; n += s["n"]
        return w, n
    out = {}
    for key, g, s in (("vs_kakuna_t1: c(b_t10) - f1, seeds 0-9", pool("b_t10"),
                       pool("f1_n0t1_vs_kakuna_t1")),
                      ("vs_kakuna_greedy: b_greedy seeds 0-4 - f2", pool("b_greedy", range(5)),
                       pool("f2_n0t1_vs_kakuna_greedy")),
                      ("vs_kakuna_greedy: b_greedy all 200 - f2", pool("b_greedy"),
                       pool("f2_n0t1_vs_kakuna_greedy"))):
        if g[1] and s[1]:
            d, lo, hi = newcombe(g[0], g[1], s[0], s[1])
            out[key] = {"greedy": g, "sampled": s, "delta": d, "ci95": [lo, hi],
                        "method": "Newcombe hybrid score (independent Wilson)"}
    return out


def main():
    ends = {}
    for e in kq.jl(kq.STATE / "units.jsonl"):
        if e.get("event") == "end" and e.get("done"):
            ends[e["unit"]] = e
    rows, cells = [], {}
    for (name, cell, games, seed, *_r) in kq.all_units():
        s = kq.summary(name)
        if not (s and s.get("status") == "OK" and s.get("n") == games):
            continue
        ok = bool(s.get("regime_verified_decisions")) and not s.get("team_source_asymmetry")
        e = ends.get(name, {})
        c = cells.setdefault(cell, {"units": [], "excluded_units": [], "wins": 0, "losses": 0,
                                    "ties": 0, "n": 0, "argmax_match_rates": [],
                                    "peer_clean_false_units": [], "load1_means": [],
                                    "turns": [], "fp_visits": [],
                                    "our_argmax_matches": 0, "our_argmax_decisions": 0})
        if not ok:
            c["excluded_units"].append(name)
            continue
        c["units"].append(name)
        for k in ("wins", "losses", "ties", "n"):
            c[k] += s[k]
        c["argmax_match_rates"].append(s.get("their_argmax_match_rates"))
        c["our_argmax_matches"] += s.get("our_argmax_matches") or 0
        c["our_argmax_decisions"] += s.get("our_argmax_decisions") or 0
        if not s.get("peer_clean"):
            c["peer_clean_false_units"].append({"unit": name, "notes": s.get("peer_exit_notes")})
        if e.get("load1_mean") is not None:
            c["load1_means"].append(e["load1_mean"])
        for line in (kq.STATE / "cells" / name / "games.jsonl").read_text().splitlines():
            r = json.loads(line)
            r.update({"x22_cell": cell, "unit": name, "unit_seed": seed, "lane": e.get("lane"),
                      "unit_load1_mean": e.get("load1_mean"), "unit_load1_max": e.get("load1_max"),
                      "fp_search_time_ms": kq.FP_MS if cell.startswith("e_") else None})
            rows.append(r)
            c["turns"].append(r.get("turns"))
            if r.get("their_visits_mean"):
                c["fp_visits"].append(r["their_visits_mean"])
    for cell, c in cells.items():
        lo, hi = kq.wilson(c["wins"], c["n"])
        c["win_rate"] = c["wins"] / c["n"] if c["n"] else None
        # (f3) OUR own-argmax rate, pooled over decisions (None where no decision was observed)
        c["our_argmax_match_rate"] = (c["our_argmax_matches"] / c["our_argmax_decisions"]
                                      if c["our_argmax_decisions"] else None)
        c["wilson95"] = [lo, hi]
        c["load1_mean_over_units"] = (sum(c["load1_means"]) / len(c["load1_means"])
                                      if c["load1_means"] else None)
        t = sorted(x for x in c.pop("turns") if x is not None)
        c["median_turns"] = t[len(t) // 2] if t else None
        v = c.pop("fp_visits")
        c["fp_visits_per_decision_mean"] = sum(v) / len(v) if v else None
        c["fp_visits_range"] = [min(v), max(v)] if v else None
    topup = kq.STATE / "topup.json"
    meta = {"topup": json.loads(topup.read_text()) if topup.exists() else None,
            "f_deltas": f_deltas()}
    (OUT / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT / "cells.json").write_text(json.dumps({"cells": cells, **meta}, indent=1))
    for cell, c in sorted(cells.items()):
        print(f"{cell:22s} n={c['n']:4d} W{c['wins']} L{c['losses']} T{c['ties']} "
              f"wr={c['win_rate']:.3f} [{c['wilson95'][0]:.3f}, {c['wilson95'][1]:.3f}] "
              f"load={c['load1_mean_over_units']} excl={c['excluded_units']}")


if __name__ == "__main__":
    main()
