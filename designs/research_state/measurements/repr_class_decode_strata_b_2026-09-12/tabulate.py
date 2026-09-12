"""Apply PREDICTION.md's TRUNK/HEAD branch test IN CODE to the merged `frame_check.json`.

Reads a `frame_check.json` whose keys are `<run>@<draw>` and which holds BOTH this measurement's
two new `ai_v12_24_ladder_strata_b` frames and the NINE frames the parent measurement
(`../repr_class_decode_2026-09-11/`) already computed, reused verbatim.

The branch test, exactly as PREDICTION §3 fixes it, at bucket `t4_10`:

  AXIS B (THE BRANCH TEST — the within-lever contrast, the one with the power):
      dB_pooled = q_pooled(strata_b) - q_pooled(strata)
      dB_V      = q_V(strata_b)      - q_V(strata)
    TRUNK          <=> dB_pooled < -FLOOR_pooled  AND  dA_pooled <= +FLOOR_pooled
    HEAD           <=> |dB_pooled| <= FLOOR_pooled  AND  |dB_V| > FLOOR_V
    PARTIAL-TRUNK  <=> dB_pooled < -FLOOR_pooled  AND  dA_pooled >  +FLOOR_pooled
    VOID           <=> |dB_V| <= FLOOR_V   (the HEAD branch's precondition is unmet; §3 rule 3)
    UNANTICIPATED  <=> dB_pooled > +FLOOR_pooled

  AXIS A (REPORTED, NOT the branch test — §5's declared power limit):
      dA_pooled = q_pooled(strata_b) - q_pooled(ctrl10M),  likewise dA_V.

  FLOOR_q(draw) = max over available control pairs of |q(ctrl_j) - q(ctrl10M)| on that draw
  (campaign rule 19: a two- or three-point spread BOUNDS the floor and carries no CI).

  Draws disagree => NOT ESTABLISHED (§3 rule 2).

Nothing is fitted here; it only arithmetics and prints. CPU, seconds.
"""
from __future__ import annotations

import argparse
import json
import os

CTRL = "ai_v12_11_ladder_ctrl10M"
CONTROLS = ("ai_v12_15_ladder_ctrl10M_b", "ai_v12_16_ladder_ctrl10M_c")
REF = "ai_v12_17_ladder_strata"          # the lever's FIRST seed — the within-lever reference
ARM = "ai_v12_24_ladder_strata_b"        # the replicate this measurement reads
VF15 = "ai_v12_10_ladder_vf15"
SHORT = {REF: "strata", ARM: "strata_b", VF15: "vf15", CTRL: "ctrl10M",
         CONTROLS[0]: "ctrl10M_b", CONTROLS[1]: "ctrl10M_c"}
ORDER = (CTRL, CONTROLS[0], CONTROLS[1], REF, ARM, VF15)
ROWS = (("pooled_to_opp_class_AUC", "pooled"), ("V_to_opp_class_AUC", "V"))
BUCKETS = ("t1", "t1_3", "t4_10", "t11_24")
DECISION = "t4_10"


def fmt_qc(v):
    return "—" if v is None else f"{v:.2e}"


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
            "n_teams": m.get("n_teams"),
            "overall_true_wr": round(list(m["trees"].values())[0]["overall_true_wr"], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    res = json.load(open(a.inp))
    draws = sorted({k.split("@", 1)[1] for k in res})

    out = {"draws": draws, "points": {}, "qc": {}, "floors": {},
           "axis_A_vs_ctrl10M": {}, "axis_B_vs_strata": {}, "between_draw_spread": {},
           "branch_test": {}}

    for run in ORDER:
        for draw in draws:
            k = f"{run}@{draw}"
            if k not in res:
                continue
            out["points"][k] = {b: {n: get(res, run, draw, b, f) for f, n in ROWS}
                                for b in BUCKETS}
            out["points"][k]["own_team_leak"] = \
                res[k]["own_team_leak"]["own_team_LOO_to_opp_class_AUC"]
            out["points"][k]["teams"] = res[k]["own_team_leak"]["teams"]
            out["points"][k]["battles"] = res[k]["own_team_leak"]["battles"]
            out["qc"][k] = qc_of(res, run, draw)

    # ── floors: the widest |control - ctrl10M| on that draw, per row, per bucket
    for draw in draws:
        for f, n in ROWS:
            for b in BUCKETS:
                base = get(res, CTRL, draw, b, f)
                if base is None:
                    continue
                pairs = {SHORT[c]: round(v - base, 4) for c in CONTROLS
                         if (v := get(res, c, draw, b, f)) is not None}
                if pairs:
                    out["floors"][f"{draw}|{n}|{b}"] = {
                        "pairs": pairs, "floor": round(max(abs(v) for v in pairs.values()), 4),
                        "n_pairs": len(pairs)}

    # ── axis A (vs ctrl10M) and axis B (vs strata), every run, every row, every bucket
    for run in (REF, ARM, VF15, *CONTROLS):
        for draw in draws:
            for f, n in ROWS:
                for b in BUCKETS:
                    v = get(res, run, draw, b, f)
                    if v is None:
                        continue
                    base_a = get(res, CTRL, draw, b, f)
                    base_b = get(res, REF, draw, b, f)
                    if base_a is not None:
                        out["axis_A_vs_ctrl10M"][f"{SHORT[run]}|{draw}|{n}|{b}"] = \
                            round(v - base_a, 4)
                    if base_b is not None:
                        out["axis_B_vs_strata"][f"{SHORT[run]}|{draw}|{n}|{b}"] = \
                            round(v - base_b, 4)

    # ── between-draw spread (the eval-draw component)
    for run in ORDER:
        for f, n in ROWS:
            for b in BUCKETS:
                vs = [v for d in draws if (v := get(res, run, d, b, f)) is not None]
                if len(vs) == 2:
                    out["between_draw_spread"][f"{SHORT[run]}|{n}|{b}"] = round(vs[1] - vs[0], 4)

    # ── THE BRANCH TEST, §3, at the decision bucket
    per_draw = {}
    for draw in draws:
        dBp = out["axis_B_vs_strata"].get(f"strata_b|{draw}|pooled|{DECISION}")
        dBv = out["axis_B_vs_strata"].get(f"strata_b|{draw}|V|{DECISION}")
        dAp = out["axis_A_vs_ctrl10M"].get(f"strata_b|{draw}|pooled|{DECISION}")
        dAv = out["axis_A_vs_ctrl10M"].get(f"strata_b|{draw}|V|{DECISION}")
        fp = out["floors"].get(f"{draw}|pooled|{DECISION}")
        fv = out["floors"].get(f"{draw}|V|{DECISION}")
        if None in (dBp, dBv, dAp, dAv) or fp is None or fv is None:
            continue
        FP, FV = fp["floor"], fv["floor"]
        strata_gain = out["axis_A_vs_ctrl10M"].get(f"strata|{draw}|pooled|{DECISION}")
        if abs(dBv) <= FV:
            br = "VOID (|dB_V| within FLOOR_V — §3 rule 3, the HEAD precondition is unmet)"
        elif dBp > FP:
            br = "UNANTICIPATED (strata_b pooled ABOVE strata past the floor)"
        elif dBp < -FP:
            br = "TRUNK" if dAp <= FP else "PARTIAL-TRUNK"
        else:
            br = "HEAD"
        per_draw[draw] = {
            "q_pooled_strata_b": get(res, ARM, draw, DECISION, ROWS[0][0]),
            "q_pooled_strata": get(res, REF, draw, DECISION, ROWS[0][0]),
            "q_pooled_ctrl10M": get(res, CTRL, draw, DECISION, ROWS[0][0]),
            "q_V_strata_b": get(res, ARM, draw, DECISION, ROWS[1][0]),
            "q_V_strata": get(res, REF, draw, DECISION, ROWS[1][0]),
            "dB_pooled": dBp, "dB_V": dBv, "dA_pooled": dAp, "dA_V": dAv,
            "FLOOR_pooled": FP, "FLOOR_V": FV, "n_pairs_pooled": fp["n_pairs"],
            "n_pairs_V": fv["n_pairs"],
            "relB_pooled": round(abs(dBp) / FP, 2) if FP else None,
            "relB_V": round(abs(dBv) / FV, 2) if FV else None,
            "relA_pooled": round(abs(dAp) / FP, 2) if FP else None,
            "relA_V": round(abs(dAv) / FV, 2) if FV else None,
            "strata_gain_pooled_vs_ctrl": strata_gain,
            "closed_fraction_of_strata_gain":
                (round(-dBp / strata_gain, 3) if strata_gain else None),
            "dB_pooled_over_dB_V": round(dBp / dBv, 2) if dBv else None,
            "BRANCH": br}
    brs = {d: v["BRANCH"] for d, v in per_draw.items()}
    uniq = set(brs.values())
    agg = ("NO DATA" if not brs else
           brs[draws[0]] if len(uniq) == 1 else
           f"NOT ESTABLISHED (draws disagree: {brs})")
    out["branch_test"] = {
        "decision_bucket": DECISION, "per_draw": per_draw, "BRANCH": agg,
        "rule": ("TRUNK: dB_pooled < -FLOOR and dA_pooled <= +FLOOR; "
                 "HEAD: |dB_pooled| <= FLOOR and |dB_V| > FLOOR_V; "
                 "PARTIAL-TRUNK: dB_pooled < -FLOOR but dA_pooled > +FLOOR; "
                 "draws disagree => NOT ESTABLISHED (PREDICTION §3)")}

    json.dump(out, open(a.out_json, "w"), indent=1)

    # ── markdown
    L = ["# Table — `strata_b` at the representation, `@step_10000032`", "",
         "Generated by [`tabulate.py`](tabulate.py); every number is computed, none is typed.", "",
         "## 1. Points — opponent-class decode AUC, out of fold, grouped by battle", "",
         "| run | draw | pooled t1–3 | **pooled t4–10** | pooled t11–24 | V t1–3 | **V t4–10** "
         "| V t11–24 |", "|" + "---|" * 8]
    for run in ORDER:
        for draw in draws:
            k = f"{run}@{draw}"
            if k not in out["points"]:
                continue
            p = out["points"][k]
            b = "**" if run == ARM else ""
            L.append(f"| {b}`{SHORT[run]}`{b} | {draw} | {p['t1_3']['pooled']} | "
                     f"**{p['t4_10']['pooled']}** | {p['t11_24']['pooled']} | {p['t1_3']['V']} "
                     f"| **{p['t4_10']['V']}** | {p['t11_24']['V']} |")
    L += ["", "## 2. THE BRANCH TEST at " + DECISION + " (PREDICTION §3)", ""]
    for d, r in per_draw.items():
        L += [f"### {d} — **{r['BRANCH']}**", "",
              "| quantity | value | floor | ×floor |", "|---|---|---|---|",
              f"| **axis B** `strata_b` − `strata`, pooled | **{r['dB_pooled']:+.4f}** | "
              f"{r['FLOOR_pooled']} ({r['n_pairs_pooled']} pair"
              f"{'s' if r['n_pairs_pooled'] > 1 else ''}) | **{r['relB_pooled']}** |",
              f"| **axis B** `strata_b` − `strata`, V | **{r['dB_V']:+.4f}** | "
              f"{r['FLOOR_V']} ({r['n_pairs_V']} pair"
              f"{'s' if r['n_pairs_V'] > 1 else ''}) | **{r['relB_V']}** |",
              f"| axis A `strata_b` − `ctrl10M`, pooled (REPORTED, §5) | {r['dA_pooled']:+.4f} | "
              f"{r['FLOOR_pooled']} | {r['relA_pooled']} |",
              f"| axis A `strata_b` − `ctrl10M`, V (REPORTED, §5) | {r['dA_V']:+.4f} | "
              f"{r['FLOOR_V']} | {r['relA_V']} |",
              f"| dB_pooled / dB_V | {r['dB_pooled_over_dB_V']} | — | — |",
              f"| fraction of `strata`'s pooled gain that `strata_b` gives back | "
              f"{r['closed_fraction_of_strata_gain']} | — | — |", ""]
    L += [f"**BRANCH: {agg}**", "",
          "## 3. Axis A — every run against `ctrl10M`, both rows, the decision bucket", "",
          "| run | draw | Δ pooled | ×floor | Δ V | ×floor |", "|" + "---|" * 6]
    for run in (CONTROLS[0], CONTROLS[1], REF, ARM, VF15):
        for draw in draws:
            kp = f"{SHORT[run]}|{draw}|pooled|{DECISION}"
            kv = f"{SHORT[run]}|{draw}|V|{DECISION}"
            if kp not in out["axis_A_vs_ctrl10M"]:
                continue
            dp = out["axis_A_vs_ctrl10M"][kp]
            dv = out["axis_A_vs_ctrl10M"].get(kv)
            FP = out["floors"][f"{draw}|pooled|{DECISION}"]["floor"]
            FV = out["floors"][f"{draw}|V|{DECISION}"]["floor"]
            L.append(f"| `{SHORT[run]}` | {draw} | {dp:+.4f} | {abs(dp) / FP:.2f} | "
                     f"{dv:+.4f} | {abs(dv) / FV:.2f} |")
    L += ["", "## 4. Frame QC", "",
          "| run | draw | battles | teams | **own-team → class leak** | "
          "**max \\|V_fwd − V_rec\\|** | true WR |", "|" + "---|" * 7]
    for run in ORDER:
        for draw in draws:
            k = f"{run}@{draw}"
            if k not in out["points"]:
                continue
            p, q = out["points"][k], out["qc"][k] or {}
            L.append(f"| `{SHORT[run]}` | {draw} | {p['battles']} | {p['teams']} | "
                     f"**{p['own_team_leak']}** | {fmt_qc(q.get('qc_max_abs_Vfwd_minus_Vrec'))} | "
                     f"{q.get('overall_true_wr')} |")
    L += ["", "## 5. Between-draw spread (the eval-draw component)", "",
          "| run | pooled t4–10 | V t4–10 | pooled t1–3 | V t1–3 |", "|" + "---|" * 5]
    for run in ORDER:
        n = SHORT[run]
        s = out["between_draw_spread"]
        if f"{n}|pooled|{DECISION}" not in s:
            continue
        L.append(f"| `{n}` | {s.get(f'{n}|pooled|t4_10')} | {s.get(f'{n}|V|t4_10')} | "
                 f"{s.get(f'{n}|pooled|t1_3')} | {s.get(f'{n}|V|t1_3')} |")
    open(a.out_md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nwrote", a.out_json, a.out_md)


if __name__ == "__main__":
    main()
