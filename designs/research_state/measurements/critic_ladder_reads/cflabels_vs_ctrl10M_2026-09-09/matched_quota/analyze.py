"""Read `matched_quota.json` into the three tables the matched-frame question needs.

TABLE A — every conditioning row on the MATCHED frame: the arm's point (median over subsample
seeds, with the 2.5/97.5 across-seed quantiles), the control's point, and the ARM - CONTROL delta
with its own interval. Two intervals are reported and they answer different questions:

* **per-seed** — the median seed's own battle-clustered CI, i.e. what the registered read would
  have printed had the arm been traced at the control's quota on that draw;
* **pooled** — every seed's bootstrap draws concatenated before the difference, so the interval
  prices the SUBSAMPLE choice on top of the battle resampling. It is the wider and the honest one
  for the question "would a matched read have detected this".

TABLE B — the FRAME-SIZE CURVE. A decode that climbs monotonically with the frame on the arm
alone is the signature of a decoder-power artefact; a flat curve is the signature of a real
effect.

TABLE C — the arm against ITSELF: full frame vs matched frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import numpy as np

from main.ops import conditioning_meters as CM
from main.ops import critic_readouts as R

QUANT = (2.5, 97.5)


def q(xs: List[float]) -> str:
    a = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return "n/a"
    lo, hi = np.percentile(a, QUANT)
    return f"{np.median(a):+.4f} [{lo:+.4f}, {hi:+.4f}]"


def med(xs: List[float]) -> float:
    a = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
    return float(np.median(a)) if a.size else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--seed", type=int, default=0, help="the delta bootstrap's pairing seed")
    args = ap.parse_args(argv)
    d = json.load(open(args.json))
    ctl, full = d["control_block"], d["arm_full"]
    rungs = {name: d["rung_reads"][name] for name, *_ in [tuple(r) for r in d["rungs"]]}
    matched = rungs["matched"]
    n_seeds = len(matched)

    out: Dict[str, Any] = {"n_seeds": n_seeds, "boot": d["boot"], "block_seed": d["block_seed"]}
    print(f"# matched-quota re-read — {n_seeds} subsample seeds, "
          f"{d['boot']} bootstrap draws per block, block seed {d['block_seed']}\n")

    # ---------------------------------------------------------------- frames
    print("## Frames\n")
    print("| view | battles | teams | states | own-team t1 decoder battles | opp-class battles |")
    print("|---|---|---|---|---|---|")

    def frame_row(name: str, rows: List[dict]) -> None:
        f = [r["frame"] for r in rows]
        sf = [r["frame"]["score_frames"] for r in rows]
        ot = [s.get("cond.own_team_r2.t1", {}).get("n_battles", float("nan")) for s in sf]
        oc = [s.get("cond.opp_class_auc.t1", {}).get("n_battles", float("nan")) for s in sf]
        print(f"| {name} | {med([x['n_battles'] for x in f]):.0f} | "
              f"{med([x['n_teams'] for x in f]):.0f} | {med([x['n_states'] for x in f]):.0f} | "
              f"{med(ot):.0f} | {med(oc):.0f} |")

    frame_row("control (as traced, 5/10/5)", [ctl])
    frame_row(f"arm MATCHED (8/12/5, median of {n_seeds})", matched)
    for name in ("decoder_matched", "x2", "x4"):
        frame_row(f"arm {name}", rungs[name])
    frame_row("arm FULL (as traced, 40/40/10)", [full])
    out["frames"] = {"control": ctl["frame"], "arm_full": full["frame"],
                     "arm_matched_median_battles": med([r["frame"]["n_battles"]
                                                        for r in matched])}

    # ---------------------------------------------------------------- table A
    def table_a(rows_rung, title: str):
        ns = len(rows_rung)
        print(f"\n## {title}\n")
        print("| row | arm SUBSAMPLED (median [2.5,97.5] over seeds) | arm FULL | control | "
              "Δ, per-seed median CI | Δ, POOLED CI | pooled label | "
              "seeds whose own CI clears 0 |")
        print("|---|---|---|---|---|---|---|---|")
        acc = []
        for key, _quantity, _stratum in CM.METERS:
            pts = [r["points"].get(key, float("nan")) for r in rows_rung]
            if not np.isfinite(med(pts)) or key not in ctl["points"]:
                continue
            c_pt = ctl["points"][key]
            c_dr = np.asarray(ctl["draws"].get(key, []), dtype=float)
            per_seed, clears = [], 0
            for r in rows_rung:
                dd = R.independent_delta(r["points"].get(key, float("nan")),
                                         np.asarray(r["draws"].get(key, []), dtype=float),
                                         c_pt, c_dr, seed=args.seed)
                per_seed.append(dd)
                lo, hi = dd["ci"]
                if np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0):
                    clears += 1
            deltas = [p["delta"] for p in per_seed]
            mid = per_seed[int(np.argsort(deltas)[len(deltas) // 2])]
            pooled_draws = np.concatenate([np.asarray(r["draws"].get(key, []), dtype=float)
                                           for r in rows_rung])
            pooled = R.independent_delta(med(pts), pooled_draws, c_pt, c_dr, seed=args.seed)
            lab = R.label_delta(pooled["delta"], pooled["ci"], None)
            print(f"| `{key}` | {q(pts)} | {full['points'].get(key, float('nan')):+.4f} | "
                  f"{c_pt:+.4f} | {mid['delta']:+.4f} [{mid['ci'][0]:+.4f}, "
                  f"{mid['ci'][1]:+.4f}] | {pooled['delta']:+.4f} "
                  f"[{pooled['ci'][0]:+.4f}, {pooled['ci'][1]:+.4f}] | {lab['label']} | "
                  f"{clears}/{ns} |")
            acc.append({"key": key, "arm_median": med(pts),
                        "arm_q": list(np.percentile([x for x in pts if np.isfinite(x)], QUANT)),
                        "arm_full": full["points"].get(key), "control": c_pt,
                        "delta_median_seed": mid, "delta_pooled": {**pooled, **lab},
                        "seeds_clearing_zero": clears, "n_seeds": ns})
        return acc

    out["table_a"] = table_a(
        matched, "Table A — every conditioning row on the BATTLE-MATCHED frame "
                 "(caps 8/12/5 — the control's own realized profile)")
    out["table_a_decoder_matched"] = table_a(
        rungs["decoder_matched"],
        "Table A2 — the same rows on the DECODER-MATCHED frame (caps 11/16/5). The arm carries "
        "more distinct teams per battle, so battle-matching UNDER-fills its own-team decoder; "
        "this rung matches the decoder's own battle count instead")

    # ---------------------------------------------------------------- table B
    print("\n## Table B — frame-size curve (the ARM alone)\n")
    print("| rung | caps (win/loss/draw) | battles | teams | own-team t1 decoder battles | "
          "`cond.own_team_r2.t1` median [2.5,97.5] | `cond.own_team_r2.all` |")
    print("|---|---|---|---|---|---|---|")
    curve = []
    cap_of = {name: (cw, cl, cd) for name, cw, cl, cd in [tuple(r) for r in d["rungs"]]}
    for name in ("matched", "decoder_matched", "x2", "x4", "full"):
        rows = rungs[name]
        cw, cl, cd = cap_of[name]
        t1 = [r["points"].get("cond.own_team_r2.t1", float("nan")) for r in rows]
        al = [r["points"].get("cond.own_team_r2.all", float("nan")) for r in rows]
        nb = med([r["frame"]["score_frames"].get("cond.own_team_r2.t1", {}).get("n_battles",
                                                                                float("nan"))
                  for r in rows])
        caps = "as traced" if cw is None else f"{cw}/{cl}/{cd}"
        print(f"| {name} | {caps} | {med([r['frame']['n_battles'] for r in rows]):.0f} | "
              f"{med([r['frame']['n_teams'] for r in rows]):.0f} | {nb:.0f} | {q(t1)} | {q(al)} |")
        curve.append({"rung": name, "caps": [cw, cl, cd],
                      "battles": med([r["frame"]["n_battles"] for r in rows]),
                      "teams": med([r["frame"]["n_teams"] for r in rows]),
                      "decoder_battles_t1": nb, "own_team_t1_median": med(t1),
                      "own_team_all_median": med(al), "n_seeds": len(rows)})
    out["table_b"] = curve
    print(f"\nControl for reference: 198 battles, 76 teams, "
          f"{ctl['frame']['score_frames']['cond.own_team_r2.t1']['n_battles']} decoder battles, "
          f"`cond.own_team_r2.t1` = {ctl['points']['cond.own_team_r2.t1']:+.4f} "
          f"{R.ci_of(ctl['points']['cond.own_team_r2.t1'], np.asarray(ctl['draws']['cond.own_team_r2.t1'])) }")

    # ---------------------------------------------------------------- table C
    print("\n## Table C — the arm against ITSELF (matched frame minus full frame)\n")
    print("| row | arm FULL | arm MATCHED (median) | Δ (matched - full), POOLED CI | label |")
    print("|---|---|---|---|---|")
    rowsC = []
    for key, _qn, _st in CM.METERS:
        if key not in full["points"]:
            continue
        pts = [r["points"].get(key, float("nan")) for r in matched]
        if not np.isfinite(med(pts)):
            continue
        pooled_draws = np.concatenate([np.asarray(r["draws"].get(key, []), dtype=float)
                                       for r in matched])
        f_dr = np.asarray(full["draws"].get(key, []), dtype=float)
        dd = R.independent_delta(med(pts), pooled_draws, full["points"][key], f_dr, seed=args.seed)
        lab = R.label_delta(dd["delta"], dd["ci"], None)
        print(f"| `{key}` | {full['points'][key]:+.4f} | {med(pts):+.4f} | "
              f"{dd['delta']:+.4f} [{dd['ci'][0]:+.4f}, {dd['ci'][1]:+.4f}] | {lab['label']} |")
        rowsC.append({"key": key, "arm_full": full["points"][key], "arm_matched": med(pts),
                      **dd, **lab})
    out["table_c"] = rowsC

    print("\n<!-- machine-readable -->\n```json")
    print(json.dumps(out, indent=1, default=float))
    print("```")
    return 0


if __name__ == "__main__":
    sys.exit(main())
