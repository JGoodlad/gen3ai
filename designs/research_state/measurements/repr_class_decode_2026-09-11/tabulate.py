"""Apply PREDICTION.md's branch test IN CODE to `frame_check.py`'s output.

Reads `frame_check.json` (keys `<run>@<draw>`), emits the per-run/per-draw table, the deltas
against `ai_v12_11_ladder_ctrl10M`, the CONTROL-PAIR FLOOR on each row (the campaign's rule-19
two-draw floor: the wider |control - control| on that draw, no CI), the BETWEEN-DRAW spread as the
eval-draw component, and the MOVES-WITH verdict per arm per draw exactly as §3 fixes it:

    MOVES WITH  <=>  sign(d_pooled) == sign(d_V)  AND  |d_pooled| > FLOOR_pooled(draw)

A mixed outcome (one arm moves with, one does not) is BRANCH B — pre-committed, not decided here.

Nothing is fitted in this file; it only arithmetics and prints. CPU, seconds.
"""
from __future__ import annotations

import argparse
import json
import os

CTRL = "ai_v12_11_ladder_ctrl10M"
CONTROLS = ("ai_v12_15_ladder_ctrl10M_b", "ai_v12_16_ladder_ctrl10M_c")
ARMS = ("ai_v12_17_ladder_strata", "ai_v12_10_ladder_vf15")
SHORT = {"ai_v12_17_ladder_strata": "strata", "ai_v12_10_ladder_vf15": "vf15",
         "ai_v12_11_ladder_ctrl10M": "ctrl10M", "ai_v12_15_ladder_ctrl10M_b": "ctrl10M_b",
         "ai_v12_16_ladder_ctrl10M_c": "ctrl10M_c"}
ROWS = (("pooled_to_opp_class_AUC", "pooled"), ("V_to_opp_class_AUC", "V"))
BUCKETS = ("t1", "t1_3", "t4_10", "t11_24")
DECISION = "t4_10"


def get(res, run, draw, bucket, field):
    k = f"{run}@{draw}"
    if k not in res:
        return None
    o = res[k]["observability"].get(bucket)
    return None if o is None else o[field]


def qc_of(res, run, draw):
    k = f"{run}@{draw}"
    if k not in res:
        return None
    p = os.path.join(res[k]["dir"], "extract_meta.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        m = json.load(f)
    return {"qc_max_abs_Vfwd_minus_Vrec": m.get("qc_max_abs_Vfwd_minus_Vrec"),
            "n_states": m.get("n_states"), "n_battles": m.get("n_battles"),
            "n_teams": m.get("n_teams"), "overall_true_wr":
            round(list(m["trees"].values())[0]["overall_true_wr"], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    res = json.load(open(a.inp))
    draws = sorted({k.split("@", 1)[1] for k in res})
    runs = [CTRL, *CONTROLS, *ARMS]

    out = {"draws": draws, "points": {}, "qc": {}, "floors": {}, "deltas": {},
           "between_draw_spread": {}, "branch_test": {}}

    for run in runs:
        for draw in draws:
            if f"{run}@{draw}" not in res:
                continue
            key = f"{run}@{draw}"
            out["points"][key] = {
                b: {n: get(res, run, draw, b, f) for f, n in ROWS} for b in BUCKETS}
            out["points"][key]["own_team_leak"] = \
                res[key]["own_team_leak"]["own_team_LOO_to_opp_class_AUC"]
            out["points"][key]["teams"] = res[key]["own_team_leak"]["teams"]
            out["points"][key]["battles"] = res[key]["own_team_leak"]["battles"]
            out["qc"][key] = qc_of(res, run, draw)

    # ── floors: the wider |control - ctrl10M| on that draw, per row, per bucket
    for draw in draws:
        for f, n in ROWS:
            for b in BUCKETS:
                base = get(res, CTRL, draw, b, f)
                if base is None:
                    continue
                pairs = {}
                for c in CONTROLS:
                    v = get(res, c, draw, b, f)
                    if v is not None:
                        pairs[SHORT[c]] = round(v - base, 4)
                if pairs:
                    out["floors"][f"{draw}|{n}|{b}"] = {
                        "pairs": pairs, "floor": round(max(abs(v) for v in pairs.values()), 4),
                        "n_pairs": len(pairs)}

    # ── deltas arm - ctrl10M
    for run in (*ARMS, *CONTROLS):
        for draw in draws:
            for f, n in ROWS:
                for b in BUCKETS:
                    v, base = get(res, run, draw, b, f), get(res, CTRL, draw, b, f)
                    if v is None or base is None:
                        continue
                    out["deltas"][f"{SHORT[run]}|{draw}|{n}|{b}"] = round(v - base, 4)

    # ── between-draw spread (the eval-draw component), per run and row
    for run in runs:
        for f, n in ROWS:
            for b in BUCKETS:
                vs = [get(res, run, d, b, f) for d in draws]
                vs = [v for v in vs if v is not None]
                if len(vs) == 2:
                    out["between_draw_spread"][f"{SHORT[run]}|{n}|{b}"] = round(vs[1] - vs[0], 4)

    # ── THE BRANCH TEST, §3, at the decision bucket
    verdicts = {}
    for run in ARMS:
        per_draw = {}
        for draw in draws:
            dp = out["deltas"].get(f"{SHORT[run]}|{draw}|pooled|{DECISION}")
            dv = out["deltas"].get(f"{SHORT[run]}|{draw}|V|{DECISION}")
            fp = out["floors"].get(f"{draw}|pooled|{DECISION}")
            fv = out["floors"].get(f"{draw}|V|{DECISION}")
            if dp is None or dv is None or fp is None:
                continue
            same_sign = (dp > 0) == (dv > 0)
            clears = abs(dp) > fp["floor"]
            per_draw[draw] = {
                "d_pooled": dp, "d_V": dv, "floor_pooled": fp["floor"],
                "floor_V": (fv or {}).get("floor"),
                "rel_pooled": round(abs(dp) / fp["floor"], 2) if fp["floor"] else None,
                "rel_V": round(abs(dv) / fv["floor"], 2) if fv and fv["floor"] else None,
                "same_sign": bool(same_sign), "clears_floor": bool(clears),
                "moves_with": bool(same_sign and clears)}
        mw = {d: v["moves_with"] for d, v in per_draw.items()}
        if not mw:
            agg = "NO DATA"
        elif all(mw.values()):
            agg = "MOVES WITH"
        elif not any(mw.values()):
            agg = "DOES NOT MOVE WITH"
        else:
            agg = "NOT ESTABLISHED (draws disagree)"
        verdicts[SHORT[run]] = {"per_draw": per_draw, "verdict": agg}
    allmw = [v["verdict"] for v in verdicts.values()]
    branch = ("A — REDUNDANT" if allmw and all(x == "MOVES WITH" for x in allmw)
              else "B — RE-SPECIFY")
    out["branch_test"] = {"decision_bucket": DECISION, "arms": verdicts, "BRANCH": branch,
                          "rule": "mixed or any non-MOVES-WITH => branch B (pre-committed §3)"}

    json.dump(out, open(a.out_json, "w"), indent=1)

    # ── markdown
    L = ["# Table — representation vs output opponent-class decode, `@step_10000032`", ""]
    L.append("## Points (AUC, out-of-fold, battle-grouped)")
    L.append("")
    hdr = "| run | draw | pooled t1–3 | pooled t4–10 | V t1–3 | V t4–10 | pooled t11–24 |"
    L += [hdr, "|" + "---|" * 7]
    for run in runs:
        for draw in draws:
            k = f"{run}@{draw}"
            if k not in out["points"]:
                continue
            p = out["points"][k]
            L.append(f"| `{SHORT[run]}` | {draw} | {p['t1_3']['pooled']} | "
                     f"**{p['t4_10']['pooled']}** | {p['t1_3']['V']} | **{p['t4_10']['V']}** | "
                     f"{p['t11_24']['pooled']} |")
    L += ["", "## Frame QC", "",
          "| run | draw | battles | teams | own-team → class leak | max \\|V_fwd − V_rec\\| | true WR |",
          "|" + "---|" * 7]
    for run in runs:
        for draw in draws:
            k = f"{run}@{draw}"
            if k not in out["points"]:
                continue
            p, q = out["points"][k], out["qc"][k] or {}
            L.append(f"| `{SHORT[run]}` | {draw} | {p['battles']} | {p['teams']} | "
                     f"{p['own_team_leak']} | {q.get('qc_max_abs_Vfwd_minus_Vrec')} | "
                     f"{q.get('overall_true_wr')} |")
    L += ["", f"## The branch test at {DECISION} (PREDICTION §3)", ""]
    for nm, v in verdicts.items():
        L.append(f"### `{nm}` — **{v['verdict']}**")
        L.append("")
        L.append("| draw | Δ pooled | floor(pooled) | ×floor | Δ V | floor(V) | ×floor | "
                 "same sign | clears | MOVES WITH |")
        L.append("|" + "---|" * 10)
        for d, r in v["per_draw"].items():
            L.append(f"| {d} | **{r['d_pooled']:+.4f}** | {r['floor_pooled']} | {r['rel_pooled']} "
                     f"| **{r['d_V']:+.4f}** | {r['floor_V']} | {r['rel_V']} | "
                     f"{'yes' if r['same_sign'] else 'NO'} | "
                     f"{'yes' if r['clears_floor'] else 'NO'} | "
                     f"{'YES' if r['moves_with'] else '**NO**'} |")
        L.append("")
    L += [f"**BRANCH: {branch}**", ""]
    L += ["## Between-draw spread (the eval-draw component)", "",
          "| run | pooled t4–10 | V t4–10 | pooled t1–3 | V t1–3 |", "|" + "---|" * 5]
    for run in runs:
        s = out["between_draw_spread"]
        n = SHORT[run]
        if f"{n}|pooled|{DECISION}" not in s:
            continue
        L.append(f"| `{n}` | {s.get(f'{n}|pooled|t4_10')} | {s.get(f'{n}|V|t4_10')} | "
                 f"{s.get(f'{n}|pooled|t1_3')} | {s.get(f'{n}|V|t1_3')} |")
    open(a.out_md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-40:]))
    print("\nwrote", a.out_json, a.out_md)


if __name__ == "__main__":
    main()
