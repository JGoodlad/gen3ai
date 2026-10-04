"""Eval U2's real-launch gate, check 3: every ledger CYCLE row's [w, w+l+d] equals that cycle's
`eval_results.jsonl` `counts[<opp>]` (and the per-team counters / cycle wall are read beside it).

    python check_ledger_vs_results.py --ledger <root>/_ledger --run-dir <models>/<run> [--out check.json]

Joins on (run, step, opponent name). A row for a bot has `opponent.id == "bot:<name>"`; a sentinel row
`<run>:pool@<step>` joins `counts[<label>]` by its label (none in a bot-only cycle). Prints one line per
row and exits non-zero on ANY mismatch, an unmatched row, an unmatched counts entry or a cycle with a
different number of rows than opponents.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(f"{a.ledger}/rows/*/*.jsonl")):
        rows += [json.loads(line) for line in open(p) if line.strip()]
    run = a.run_dir.rstrip("/").split("/")[-1]
    rows = [r for r in rows if r.get("run") == run and r.get("purpose") == "cycle"]
    results = {}
    for line in open(f"{a.run_dir}/eval_results.jsonl"):
        d = json.loads(line)
        results[int(d["step"])] = d
    by_step = defaultdict(list)
    for r in rows:
        by_step[int(r["compute"]["step"])].append(r)
    bad = 0
    out = {"run": run, "cycles": {}}
    for step, rs in sorted(by_step.items()):
        res = results.get(step)
        if res is None:
            print(f"step {step}: NO eval_results.jsonl row"); bad += 1; continue
        counts = res["counts"]
        seen = set()
        cyc = {"rows": len(rs), "opponents_in_results": len(counts) + len(res.get("sentinels", [])), "cycle_wall_s": set(), "per_row": []}
        for r in sorted(rs, key=lambda r: r["opponent"]["id"]):
            oid = r["opponent"]["id"]
            c = r["counts"]
            w, n = c["w"], c["w"] + c["l"] + c["d"]
            if oid.startswith("bot:"):
                name = oid[4:]
                want = counts.get(name)
            else:                                   # a pool sentinel `<run>:pool@<step>` -> results["sentinels"][step]
                name = oid.split(":", 1)[-1]
                sstep = int(oid.rsplit("@", 1)[-1])
                hit = [s for s in res.get("sentinels", []) if int(s["step"]) == sstep]
                want = hit[0]["counts"] if len(hit) == 1 else None
            ok = want is not None and [w, n] == list(want) and c.get("aborted", 0) == 0
            # per-team counters are [games, wins] for the player's team `p` and the opponent's team `o`
            tp = sum(v["p"][0] for v in r["teams"].values()); tw = sum(v["p"][1] for v in r["teams"].values())
            to = sum(v["o"][1] for v in r["teams"].values())
            cyc["per_row"].append({"opp": name, "ledger": [w, n], "results": want, "ok": ok, "d": c["d"], "aborted": c.get("aborted", 0),
                                   "team_p_games": tp, "team_p_wins": tw, "team_o_wins": to,
                                   "team_counters_consistent": bool(tp == n and tw == w and to == c["l"]),
                                   "digest_all": r["compute"]["outcome_digest_all"][:12]})
            cyc["cycle_wall_s"].add(r["compute"]["cycle_wall_s"])
            print(f"step {step:>9,d} {name:<12s} ledger [w={w:3d}, n={n:3d}] results {want} -> {'OK' if ok else 'MISMATCH'}"
                  f"  (d={c['d']} aborted={c.get('aborted', 0)}; team counters p games/wins {tp}/{tw}, o wins {to})")
            seen.add(name)
            bad += 0 if ok else 1
            if not cyc["per_row"][-1]["team_counters_consistent"]:
                print(f"   team counters inconsistent with counts at {name}"); bad += 1
        extra = set(counts) - seen
        n_opp = len(counts) + len(res.get("sentinels", []))
        if extra or len(rs) != n_opp:
            print(f"step {step}: rows {len(rs)} vs results opponents {n_opp}; counts not in ledger: {sorted(extra)}"); bad += 1
        cyc["cycle_wall_s"] = sorted(cyc["cycle_wall_s"])
        out["cycles"][str(step)] = cyc
    out["mismatches"] = bad
    out["cycles_checked"] = len(by_step)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"cycles checked: {len(by_step)}; rows: {len(rows)}; mismatches: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
