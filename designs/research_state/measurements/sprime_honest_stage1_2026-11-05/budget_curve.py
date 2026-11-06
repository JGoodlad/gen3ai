"""S'-HONEST stage 1, the BUDGET CURVE (2026-11-05, OUTCOME-BLIND): r0s4_honest's exclusion rate and cost per panel
at several builder budgets (chain steps, chain feeds, world attempts), on the timing check's 300 states.

The builder is deterministic and a larger budget replays a smaller one's trajectory exactly, going on only where the
smaller one stopped. So a state is RE-RUN at a larger budget only if the base run hit a limit there: it was excluded,
its chain stopped on its step or feed budget (``phase`` budget / exhausted), or its worlds spent their attempts
(``world_replay`` failures >= the base attempts). Every other state's row is the base row (the spot check re-runs
some of them at the largest budget and asserts byte-identical records).

    python3 budget_curve.py select --base DIR --attempts 3 --out keys.txt
    python3 budget_curve.py merge --base DIR --rolls DIR --setting NAME=DIR [--setting …] --out curve.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

HON, ROL = "r0s4_honest@10", "r0s4_rolls@10"
SLIVE_GAME_CPU_MIN, SLIVE_DECISIONS_PER_GAME = 6.2, 103616 / 2698


def rows(d):
    out = {}
    for x in (Path(d) / "rows.jsonl").read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            out.setdefault(r["key"], r)
    return out


def limited(r, attempts):
    h = r["res"][HON]
    if not h["searched"]:
        return True
    hb = h.get("honest") or {}
    if (hb.get("chain") or {}).get("phase") in ("budget", "exhausted"):
        return True
    return int((hb.get("fails") or {}).get("world_replay", 0)) >= attempts


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--base", required=True)
    s.add_argument("--attempts", type=int, default=3)
    s.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--base", required=True)
    m.add_argument("--rolls", required=True)
    m.add_argument("--setting", action="append", default=[])
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    base = rows(a.base)
    if a.cmd == "select":
        keys = sorted(k for k, r in base.items() if limited(r, a.attempts))
        Path(a.out).write_text("\n".join(keys) + "\n")
        print(json.dumps({"states": len(base), "limited": len(keys)}))
        return
    rol = rows(a.rolls)
    settings = [("base", None)] + [tuple(x.split("=", 1)) for x in a.setting]
    out = {}
    for name, d in settings:
        over = rows(d) if d else {}
        merged = {k: over.get(k, r) for k, r in base.items()}
        res = {}
        for panel in ("H", "F", "P72", "U"):
            rs = [r for r in merged.values() if panel == "U" or r["panel"] == panel]
            h = [r["res"][HON] for r in rs]
            cpu = np.array([v["cpu_s"] + v.get("validator_cpu_s", 0.0) for v in h])
            rc = np.array([rol[r["key"]]["res"][ROL]["cpu_s"] for r in rs])
            ex = [not v["searched"] for v in h]
            res[panel] = {"n": len(rs), "excluded": int(sum(ex)), "exclusion_rate": round(float(np.mean(ex)), 4),
                          "honest_cpu_s_mean": round(float(cpu.mean()), 3),
                          "honest_cpu_s_p90": round(float(np.quantile(cpu, 0.9)), 3),
                          "rolls_cpu_s_mean": round(float(rc.mean()), 3),
                          "projected_cpu_min_per_game": {
                              "ratio_to_slive": round(SLIVE_GAME_CPU_MIN * float(cpu.mean() / rc.mean()), 2),
                              "absolute": round(SLIVE_DECISIONS_PER_GAME * float(cpu.mean()) / 60.0, 2)},
                          "rerun": sum(1 for r in rs if r["key"] in over)}
        out[name] = res
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
