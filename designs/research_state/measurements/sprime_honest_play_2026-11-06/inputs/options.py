"""S'-HONEST play: the coordinator's options table (2026-11-06, OUTCOME-BLIND). Local as-is, fewer deals, a rented box,
and an INCREMENTAL (particle-filter carry-forward) builder, each with CPU-h, wall days and power.

It reuses `cost_model.py`'s inputs (S'-LIVE's per-panel turn-bucket decision counts, kappa and overhead, stage 1's
early buckets, the late probe) and splits every bucket's honest cost into its parts: the playouts and validator (paid
by any builder), the world building of a SEARCHED decision (what carry-forward would cut), and an EXCLUSION (the
chain's whole budget, burnt).

The incremental builder is NOT built. Its world-building cost on a searched decision is modelled as
    world_inc = world_full * (f_b * w + eps)
with f_b the share of decisions at which the carried hypothesis fails the new segment(s) (a GUESS by bucket), w the
warm-started chain's cost against a cold one when it does (a GUESS), and eps the extension itself (re-playing each
carried world on its reseed schedule plus matching the new segment: ~10-60 ms a world, `world_cost.json`'s 11.8 ms a
replay). An exclusion keeps its full cost; the optimistic case also cuts the exclusion rate (a warm chain starts from a
hypothesis that fit everything but the new segment).

    python3 options.py --slive STATE --stage1 FINAL_SUMMARY --probe ROWS --out OUT.json
"""
import argparse
import collections
import importlib.util
import json
import math
import statistics as S
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cost_model", HERE / "cost_model.py")
CM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CM)

N = NormalDist()
SE0, N0, D0 = (6.48 - 2.66) / (2 * 1.96), 2698, 0.341       # S'-LIVE's realised paired SE (pp) at 2,698 pairs
K6_RATE = None                                                # CPU-h of whole-unit CPU a wall hour at K = 6 (S'-LIVE life 4)
#: the incremental builder's GUESSES: (f by bucket, w, exclusion-rate multiplier)
INC = {"optimistic": ({"10-19": 0.30, "20-29": 0.20, "30-63": 0.10, "64+": 0.05}, 0.2, 0.3),
       "pessimistic": ({"10-19": 0.50, "20-29": 0.40, "30-63": 0.25, "64+": 0.15}, 0.5, 1.0)}
EXT_S = 8 * 0.05                                              # eps: 8 worlds x ~50 ms (replay on the schedule + one match)


def power(n, delta=2.30):
    r, c = SE0 * math.sqrt(N0 / n), 100 * math.sqrt(D0 / n)
    return 1 - N.cdf(1.96 - delta / r), 1 - N.cdf(1.96 - delta / c)


def parts(rows):
    """Per (panel, bucket): the honest search's exclusion rate and the mean CPU-s of its parts."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["panel"], r["bucket"])].append(r["res"]["r0s4_honest@10"])
    out = {}
    for k, hs in by.items():
        srch = [h for h in hs if h["searched"]]
        excl = [h for h in hs if h["why"] == "excluded"]
        out[k] = {"n": len(hs), "excl_rate": len(excl) / len(hs),
                  "world_searched": S.mean([h["world_cpu_s"] for h in srch]) if srch else 0.0,
                  "other_searched": S.mean([h["cpu_s"] - h["world_cpu_s"] + h["validator_cpu_s"] for h in srch])
                  if srch else 0.0,
                  "excl_cost": S.mean([h["cpu_s"] + h["validator_cpu_s"] for h in excl]) if excl else None}
    return out


def main(a):
    sl = CM.slive_inputs(a.slive)
    ko = CM.kappa_and_overhead(a.slive)
    st1 = json.load(open(a.stage1))["panels"]
    rows = [json.loads(x) for x in Path(a.probe).read_text().splitlines() if x.strip()]
    pa = parts(rows)
    rate = max(sl["k6_cpu_h_per_wall_h"].values())
    excl_all = [p["excl_cost"] for p in pa.values() if p["excl_cost"] is not None]
    excl_cost_default = S.mean(excl_all) if excl_all else 200.0

    def bucket_cost(panel, b, mode):
        """(CPU-s a search-path decision at K = 2, exclusion rate) under ``mode``: as_is / optimistic / pessimistic."""
        if b in CM.EARLY:
            # stage 1's means are over every decision, exclusions included: split them out at the probe's mean
            # exclusion cost (an exclusion burns the chain's whole budget at any turn)
            h = st1[panel]["honest"]
            e = h["exclusion_rate"]
            ex_cost = excl_cost_default
            if mode == "as_is":
                return h["cpu_s_mean"] + h["validator_cpu_s_mean"], e
            world = max(0.0, (h["world_cpu_s_mean"] - e * ex_cost) / (1 - e))
            other = h["playout_cpu_s_mean"] + h["validator_cpu_s_mean"]
        else:
            p = pa.get((panel, b)) or pa[(a.fallback, b)]          # --fallback: a provisional run only
            e, world, other = p["excl_rate"], p["world_searched"], p["other_searched"]
            ex_cost = p["excl_cost"] if p["excl_cost"] is not None else excl_cost_default
        if mode != "as_is":
            f, w, em = INC[mode]
            world = world * (f[b] * w) + EXT_S
            e = e * em
        return (1 - e) * (world + other) + e * ex_cost, e

    out = {"inputs": {"probe_states": len(rows), "k6_cpu_h_per_wall_h": rate, "kappa": ko["kappa"],
                      "overhead_s": ko["overhead_s"], "incremental_guesses": INC, "ext_s": EXT_S,
                      "parts": {f"{p}/{b}": v for (p, b), v in sorted(pa.items())}}, "modes": {}}
    raw_min_k6 = sl["raw_unit_cpu_min_k6"]
    raw_min_k2 = sl["raw_unit_cpu_min"]
    for mode in ("as_is", "optimistic", "pessimistic"):
        per = {}
        for panel in ("H", "F", "P72"):
            n = sl["per_game"][panel]
            k2 = sum(n[b] * bucket_cost(panel, b, mode)[0] for b, _l, _h in CM.BUCKETS) + ko["overhead_s"]["k2"]
            k6 = sum(n[b] * bucket_cost(panel, b, mode)[0] * ko["kappa"][b] for b, _l, _h in CM.BUCKETS) \
                + ko["overhead_s"]["k6"]
            excl = sum(n[b] * bucket_cost(panel, b, mode)[1] for b, _l, _h in CM.BUCKETS) / sum(n.values())
            hours = sorted((sum(g.get(b, 0) * bucket_cost(panel, b, mode)[0] * ko["kappa"][b] for b, _l, _h in CM.BUCKETS)
                            + ko["overhead_s"]["k6"]) / 3600 for g in sl["each_game"][panel])
            per[panel] = {"game_cpu_min_k2": k2 / 60, "game_cpu_min_k6": k6 / 60, "excl_share": excl,
                          "watchdog": {"max_h": hours[-1], "over_10h": sum(x > 10 for x in hours),
                                       "over_10h_at_x1.5": sum(1.5 * x > 10 for x in hours),
                                       "over_8h": sum(x > 8 for x in hours), "games": len(hours)}}
        out["modes"][mode] = {"per_panel": per,
                              "game_cpu_min_k6_U": S.mean(per[p]["game_cpu_min_k6"] for p in per),
                              "game_cpu_min_k2_U": S.mean(per[p]["game_cpu_min_k2"] for p in per),
                              "excl_share_U": S.mean(per[p]["excl_share"] for p in per)}
    table = []
    for label, mode, deals in (("local as-is", "as_is", 3840), ("local as-is, 3,240 deals", "as_is", 3240),
                               ("local as-is, 2,700 deals", "as_is", 2700),
                               ("incremental builder, local (optimistic)", "optimistic", 3840),
                               ("incremental builder, local (pessimistic)", "pessimistic", 3840)):
        m = out["modes"][mode]
        cpu = deals * (m["game_cpu_min_k6_U"] + raw_min_k6) / 60
        pw = power(deals)
        table.append({"option": label, "deals": deals, "cpu_h_k6": round(cpu, 0),
                      "wall_days_free": round(cpu / rate / 24, 1),
                      "wall_days_pinned_trainer": round(cpu * CM.TRAINER_CPU_RATIO / rate / 24, 1),
                      "power_at_2.30": [round(pw[0], 2), round(pw[1], 2)]})
    m = out["modes"]["as_is"]
    cpu_k2 = 3840 * (m["game_cpu_min_k2_U"] + raw_min_k2) / 60
    table.append({"option": "rented 64-vCPU box, as-is", "deals": 3840, "cpu_h_k2": round(cpu_k2, 0),
                  "box_hours_at_45_30_games": [round(cpu_k2 / 45, 1), round(cpu_k2 / 30, 1)],
                  "usd_at_1.5_3.5": [round(cpu_k2 / 45 * 1.5), round(cpu_k2 / 30 * 3.5)],
                  "power_at_2.30": [round(x, 2) for x in power(3840)]})
    out["table"] = table
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"table": table, "modes": {k: {kk: (round(vv, 2) if isinstance(vv, float) else vv)
                                                     for kk, vv in v.items() if kk != "per_panel"}
                                                 for k, v in out["modes"].items()},
                      "per_panel_as_is": {p: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}
                                          for p, d in out["modes"]["as_is"]["per_panel"].items()}},
                     indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slive", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fallback", default=None, help="PROVISIONAL: a panel whose late bucket stands in for a missing one")
    main(ap.parse_args())
