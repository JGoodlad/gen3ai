"""APPLY THE READING RULE IN CODE, and render the curves as tables.

The three readings were registered in `PREDICTION.md` before any number existed. This file turns
them into a function of the numbers rather than of the author:

  guard 0   `cond_oracle`'s turn-1-3 ratio CI lower bound must clear 0.80, or the (b) column is
            INCONCLUSIVE and never folded into the reading (the head refit's rule, kept verbatim).
  (i)   RISE       the terminal-label TREND delta (N_max - N_min) is POSITIVE with its own
                   battle-clustered CI clear of zero, on the turn-1-3 spread ratio OR the turn-1
                   opponent-class AUC, and the early segment (N_min -> 8k) already carries it.
  (iii) LATE RISE  the same trend is detected, but the EARLY segment straddles zero and the LATE
                   one (8k -> N_max) does not.
  (ii)  FLAT       neither trend is detected AND `mlp_cond` is detected above `online` on the
                   turn-1-3 ratio at EVERY N. (The second clause matters: a curve that is flat
                   because the whole frame is dead says nothing about non-stationarity.)
  else  MIXED      reported as such, with the rows that disagree named.

The per-doubling slope is an OLS of the meter on log2(N) computed INSIDE the shared bootstrap, so
it carries a battle-clustered CI rather than a fitted line through six point estimates.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

FITS = ("lin_term", "mlp_term", "lin_cond", "mlp_cond", "cond_oracle")
DECLAB = {"t1|opp_class": "turn-1 opponent class (AUC) — ⚠️ UNOBSERVABLE window",
          "t1|own_team_wr": "turn-1 own-team LOO win rate (R²)",
          "t4_10|opp_class": "turns 4-10 opponent class (AUC)",
          "t4_10|own_team_wr": "turns 4-10 own-team LOO win rate (R²)"}
LABEL = {"lin_term": "linear . terminal", "mlp_term": "MLP . terminal (a)",
         "lin_cond": "linear . conditional", "mlp_cond": "MLP . conditional (b)",
         "cond_oracle": "the conditional TARGET (ceiling)",
         "mlp_term_long": "MLP . terminal, 5x steps, NO early stop",
         "mlp_cond_long": "MLP . conditional, 5x steps, NO early stop",
         "online": "online (frozen forward)", "online_rec": "online (recorded)",
         "pooled": "`value_pooled` — the head's INPUT"}


def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def ci_s(c, nd=3):
    if not c or c[0] is None:
        return "—"
    return f"[{c[0]:+.{nd}f}, {c[1]:+.{nd}f}]"


def detected(c):
    return bool(c and c[0] is not None and (c[0] > 0) == (c[1] > 0))


def main(a):
    R = json.load(open(a.read))
    ns = R["ns"]
    key = a.key
    sp = R[f"spread_{key}"]
    dec = R.get("decode", {})
    out = [f"# N-curve tables — `{a.name}` (spread key: {key})", ""]
    out.append(f"Held-out: **{R['n_holdout_battles']} battles / {R['n_holdout_states']} states**, "
               f"fixed across every N. Training pool {R['fit_frame']['pool_battles']} battles "
               f"(of {R['fit_frame']['n_battles']} total, {R['fit_frame']['n_teams']} teams). "
               f"Cells: {sp['info']['cells_with_data']}/{sp['info']['n_cells']}, "
               f"{sp['info']['games_per_cell']} games each on the outcome side. "
               f"Every HT weight is 1.0 (full capture).")
    out.append("")

    # ── the identity ─────────────────────────────────────────────────────────
    out.append("## 1. The mixture identity — between-cell spread of the prediction / of the outcome")
    out.append("")
    out.append("Bold = the delta vs `online` has its own battle-clustered CI clear of zero.")
    out.append("")
    for bk, title in (("t1", "turn 1 — ⚠️ the opponent is UNOBSERVABLE here (README hazard 1); "
                       "ratio 0 is Bayes-optimal, not a defect"),
                      ("t1_3", "turns 1-3 — the head refit's headline window"),
                      ("mid", "turns 11-24 — the opponent IS observable here"),
                      ("all", "all states")):
        out.append(f"### {title}")
        out.append("")
        out.append("| condition | " + " | ".join(f"N={n:,}" for n in ns) + " |")
        out.append("|---|" + "---|" * len(ns))
        o_pt = sp["point"].get(f"online|{bk}|ratio")
        o_ci = sp["ci"].get(f"online|{bk}|ratio")
        out.append(f"| {LABEL['online']} | " + " | ".join(
            [f"{fmt(o_pt)} {ci_s(o_ci)}"] + ["·"] * (len(ns) - 1)) + " |")
        for c in FITS:
            row = []
            for n in ns:
                k = f"{n}|{c}"
                p = sp["point"].get(f"{k}|{bk}|ratio")
                d = sp["delta"].get(f"{k}-online|{bk}|ratio")
                cell = fmt(p)
                if d and detected(d["ci"]):
                    cell = f"**{cell}**"
                if d:
                    cell += f" {ci_s(d['ci'])}"
                row.append(cell)
            out.append(f"| {LABEL[c]} | " + " | ".join(row) + " |")
        for c in ("mlp_term_long", "mlp_cond_long"):
            k = f"{ns[-1]}|{c}"
            if f"{k}|{bk}|ratio" in sp["point"]:
                d = sp["delta"].get(f"{k}-online|{bk}|ratio")
                cell = fmt(sp["point"][f"{k}|{bk}|ratio"])
                if d and detected(d["ci"]):
                    cell = f"**{cell}**"
                out.append(f"| {LABEL[c]} | " + " | ".join(["·"] * (len(ns) - 1)
                                                           + [cell + " " + ci_s(d["ci"] if d else None)]) + " |")
        out.append("")

    # ── the N-trend ──────────────────────────────────────────────────────────
    out.append("## 2. The N-TREND — the whole question, in four numbers per substrate")
    out.append("")
    out.append(f"Delta between N={ns[-1]:,} and N={ns[0]:,}, with the delta's OWN battle-clustered CI.")
    out.append("")
    n_doub = float(np.log2(ns[-1] / ns[0]))
    out.append(f"The span is **{n_doub:.2f} doublings** ({ns[0]:,} -> {ns[-1]:,} battles), so the "
               f"per-doubling column is the delta and its CI divided by {n_doub:.2f} — a "
               f"linear-in-log2(N) summary, which is what an extrapolation to an online run's "
               f"data volume would use.")
    out.append("")
    out.append("| meter | condition | N_min | N_max | Δ (N_max − N_min) | per doubling | detected |")
    out.append("|---|---|---|---|---|---|---|")
    trend = {}
    for bk in ("t1", "t1_3", "mid", "all"):
        for c in ("mlp_term", "mlp_cond", "lin_term", "cond_oracle"):
            d = sp["delta"].get(f"{ns[-1]}|{c}-{ns[0]}|{c}|{bk}|ratio")
            if not d:
                continue
            lo = sp["point"].get(f"{ns[0]}|{c}|{bk}|ratio")
            hi = sp["point"].get(f"{ns[-1]}|{c}|{bk}|ratio")
            trend[f"ratio|{bk}|{c}"] = d
            out.append(f"| spread ratio @{bk} | {LABEL[c]} | {fmt(lo)} | {fmt(hi)} | "
                       f"{fmt(d['point'])} {ci_s(d['ci'])} | "
                       f"{fmt((d['point'] or 0) / n_doub)} "
                       f"{ci_s([x / n_doub for x in d['ci']] if d['ci'][0] is not None else None)} | "
                       f"{'**YES**' if detected(d['ci']) else 'no'} |")
    for tname, lab in sorted((k, DECLAB.get(k, k)) for k in dec):
        dd = dec[tname]["delta_ci"]
        for c in ("mlp_term", "mlp_cond", "lin_term", "cond_oracle"):
            k = f"{ns[-1]}|{c}-{ns[0]}|{c}"
            if k not in dd:
                continue
            trend[f"{tname}|{c}"] = {"point": round(dec[tname]["score"][f"{ns[-1]}|{c}"]
                                                    - dec[tname]["score"][f"{ns[0]}|{c}"], 4),
                                     "ci": dd[k]}
            pv = trend[f"{tname}|{c}"]["point"]
            out.append(f"| {lab} | {LABEL[c]} | "
                       f"{fmt(dec[tname]['score'][f'{ns[0]}|{c}'])} | "
                       f"{fmt(dec[tname]['score'][f'{ns[-1]}|{c}'])} | "
                       f"{fmt(pv)} {ci_s(dd[k])} | "
                       f"{fmt((pv or 0) / n_doub)} "
                       f"{ci_s([x / n_doub for x in dd[k]] if dd[k][0] is not None else None)} | "
                       f"{'**YES**' if detected(dd[k]) else 'no'} |")
    out.append("")

    # ── the decodes ──────────────────────────────────────────────────────────
    if dec:
        out.append("## 3. What the PREDICTION decodes at turn 1")
        out.append("")
        for tname, lab in sorted((k, DECLAB.get(k, k)) for k in dec):
            D = dec[tname]
            out.append(f"### {lab} — n = {D['n_states']} states / {D['n_battles']} battles, "
                       f"null p95 (pooled) {fmt(D['null_p95']['pooled'])}")
            out.append("")
            out.append("| condition | " + " | ".join(f"N={n:,}" for n in ns) + " | null p95 |")
            out.append("|---|" + "---|" * (len(ns) + 1))
            for ref in ("pooled", "online"):
                out.append(f"| {LABEL[ref]} | " + " | ".join(
                    [f"{fmt(D['score'][ref])} {ci_s(D['ci'].get(ref))}"]
                    + ["·"] * (len(ns) - 1)) + f" | {fmt(D['null_p95'][ref])} |")
            for c in FITS:
                row, np95 = [], None
                for n in ns:
                    k = f"{n}|{c}"
                    s = D["score"].get(k)
                    d = D["delta_ci"].get(f"{k}-online")
                    np95 = D["null_p95"].get(k, np95)
                    cell = fmt(s)
                    if d and detected(d):
                        cell = f"**{cell}**"
                    row.append(cell + (f" {ci_s(d)}" if d else ""))
                out.append(f"| {LABEL[c]} | " + " | ".join(row) + f" | {fmt(np95)} |")
            for c in ("mlp_term_long", "mlp_cond_long"):
                k = f"{ns[-1]}|{c}"
                if k in D["score"]:
                    d = D["delta_ci"].get(f"{k}-online")
                    out.append(f"| {LABEL[c]} | " + " | ".join(
                        ["·"] * (len(ns) - 1) + [fmt(D['score'][k]) + " " + ci_s(d)])
                        + f" | {fmt(D['null_p95'].get(k))} |")
            out.append("")

    # ── Brier / calibration ──────────────────────────────────────────────────
    out.append("## 4. Brier, resolution and the calibration slope on the held-out battles")
    out.append("")
    for bk in ("t1_3", "all"):
        S = R["scalar"][bk]
        out.append(f"### {bk}")
        out.append("")
        out.append("| condition | " + " | ".join(f"N={n:,}" for n in ns)
                   + " | (Brier / resolution / calib slope) |")
        out.append("|---|" + "---|" * (len(ns) + 1))
        for ref in ("online", "online_rec"):
            p = S["point"][ref]
            out.append(f"| {LABEL[ref]} | " + " | ".join(
                [f"{fmt(p['brier'],4)} / {fmt(p['resolution'],4)} / {fmt(p['calib_slope'],2)}"]
                + ["·"] * (len(ns) - 1)) + " | |")
        for c in FITS:
            row = []
            for n in ns:
                k = f"{n}|{c}"
                p = S["point"].get(k)
                row.append("—" if not p else
                           f"{fmt(p['brier'],4)} / {fmt(p['resolution'],4)} / "
                           f"{fmt(p['calib_slope'],2)}")
            out.append(f"| {LABEL[c]} | " + " | ".join(row) + " | |")
        for c in ("mlp_term_long", "mlp_cond_long"):
            k = f"{ns[-1]}|{c}"
            if k in S["point"]:
                p = S["point"][k]
                out.append(f"| {LABEL[c]} | " + " | ".join(["·"] * (len(ns) - 1) + [
                    f"{fmt(p['brier'],4)} / {fmt(p['resolution'],4)} / "
                    f"{fmt(p['calib_slope'],2)}"]) + " | |")
        out.append("")

    # ── the MORE-OPTIMISATION control ────────────────────────────────────────
    out.append("## 5. The MORE-OPTIMISATION control at the largest N")
    out.append("")
    out.append("| contrast | Δ ratio @t1-3 | Δ opp-class AUC @t4-10 | Δ own-team R² @t1 | "
               "Δ Brier @all |")
    out.append("|---|---|---|---|---|")
    longres = {}
    for c in ("mlp_term", "mlp_cond"):
        k = f"{ns[-1]}|{c}_long-{ns[-1]}|{c}"
        d = sp["delta"].get(f"{k}|t1_3|ratio")
        if not d:
            continue
        da = dec.get("t4_10|opp_class", {}).get("delta_ci", {}).get(k)
        dt = dec.get("t1|own_team_wr", {}).get("delta_ci", {}).get(k)
        db = R["scalar"]["all"]["delta"].get(f"{k}|brier")
        longres[c] = {"ratio_t1_3": d, "opp_class": da, "own_team_wr": dt, "brier": db}
        out.append(f"| {LABEL[c + '_long']} − {LABEL[c]} | {fmt(d['point'])} {ci_s(d['ci'])} | "
                   f"{ci_s(da)} | {ci_s(dt)} | {ci_s(db['ci'] if db else None, 4)} |")
    out.append("")

    # ── THE READING RULE, in code ────────────────────────────────────────────
    oracle_ci = sp["ci"].get(f"{ns[-1]}|cond_oracle|t1_3|ratio") or [None, None]
    guard_ok = bool(oracle_ci[0] is not None and oracle_ci[0] >= 0.80)
    mid_n = [n for n in ns if n <= 8000][-1]

    def pos(d):
        """DETECTED and in the RIGHT DIRECTION. The registered rule said only `detected`, which
        is satisfied by a CI clear of zero on the WRONG side — and on this frame the conditional
        arm is detected BELOW the online head on most buckets, so the sign has to be in the rule."""
        return bool(d and detected(d["ci"]) and (d["point"] or 0) > 0)

    tr = {bk: trend.get(f"ratio|{bk}|mlp_term") for bk in ("t1", "t1_3", "mid", "all")}
    tr_class = trend.get("t4_10|opp_class|mlp_term")
    tr_team = trend.get("t1|own_team_wr|mlp_term")
    rise_buckets = [bk for bk in ("t1_3", "mid", "all") if pos(tr[bk])]
    rise = bool(rise_buckets) or pos(tr_class)
    seg_early = sp["delta"].get(f"{mid_n}|mlp_term-{ns[0]}|mlp_term|mid|ratio")
    seg_late = sp["delta"].get(f"{ns[-1]}|mlp_term-{mid_n}|mlp_term|mid|ratio")
    # the row that separates "the curve climbs toward the run's OWN head" from "stationary data
    # buys conditioning the online head does not have"
    vs_online = {bk: sp["delta"].get(f"{ns[-1]}|mlp_term-online|{bk}|ratio")
                 for bk in ("t1", "t1_3", "mid", "all")}
    exceeds = [bk for bk, d in vs_online.items() if pos(d)]

    if rise and pos(seg_early) and not pos(seg_late):
        verdict = "(i) RISE, SATURATED by N=%d" % mid_n
    elif rise and not pos(seg_early) and pos(seg_late):
        verdict = "(iii) LATE RISE"
    elif rise:
        verdict = "(i) RISE, still climbing at N_max"
    elif guard_ok:
        verdict = "(ii) FLAT"
    else:
        verdict = "MIXED — no trend AND the meter is not shown to be movable on this frame"

    # what the REGISTERED rule (PREDICTION.md) would have emitted, reported beside the amended
    # one rather than quietly replaced. Its (ii) clause required `mlp_cond` to be DETECTED above
    # `online` at every N as a liveness check; on this frame the conditional target is detected
    # BELOW the online head, which the registered wording could not express.
    cond_every_n = all(detected((sp["delta"].get(f"{n}|mlp_cond-online|t1_3|ratio") or {})
                                .get("ci")) for n in ns)
    reg_rise = bool((tr["t1_3"] and detected(tr["t1_3"]["ci"]) and tr["t1_3"]["point"] > 0)
                    or (tr_class and detected(tr_class["ci"]) and tr_class["point"] > 0))
    verdict_registered = ("(i) RISE" if reg_rise else
                          ("(ii) FLAT" if cond_every_n else "MIXED"))

    reading = {"verdict": verdict, "verdict_as_registered": verdict_registered,
               "n_doublings": round(n_doub, 3),
               "guard_cond_oracle_ge_0.80": guard_ok,
               "cond_oracle_t1_3_ratio_ci_at_Nmax": oracle_ci,
               "trend_ratio_mlp_term": tr,
               "trend_opp_class_t4_10_mlp_term": tr_class,
               "trend_own_team_wr_t1_mlp_term": tr_team,
               "rise_buckets": rise_buckets, "rise": rise,
               "segment_1k_to_%d_mid" % mid_n: seg_early,
               "segment_%d_to_Nmax_mid" % mid_n: seg_late,
               "mlp_term_at_Nmax_vs_online": vs_online,
               "mlp_term_EXCEEDS_online_at_Nmax_in": exceeds,
               "online_ratio": {bk: sp["point"].get(f"online|{bk}|ratio")
                                for bk in ("t1", "t1_3", "mid", "all")},
               "mlp_cond_detected_at_every_N_either_sign": cond_every_n,
               "more_optimisation": longres}
    out.append("## 6. THE READING — emitted by the rule, not by the author")
    out.append("")
    out.append(f"**{verdict}**  ·  as the PRE-REGISTERED wording would have emitted it: "
               f"**{verdict_registered}**" + ("" if guard_ok else
                                   "  ⚠️ the `cond_oracle` guard is NOT met — the (b) column is "
                                   "INCONCLUSIVE and is not folded into this reading"))
    out.append("")
    out.append("```json")
    out.append(json.dumps(reading, indent=1))
    out.append("```")
    open(a.out_md, "w").write("\n".join(out) + "\n")
    json.dump(reading, open(a.out_json, "w"), indent=1)
    print(f"{a.name} [{key}]: {verdict}  (guard {guard_ok}) -> {a.out_md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--key", default="cell", choices=("cell", "opponent"))
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    main(ap.parse_args())
