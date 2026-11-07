"""S'-HONEST play PREREG input (2026-11-06, OUTCOME-BLIND): the cost, wall-time, exclusion-bound, watchdog and
identity re-score model for the INCREMENTAL builder ``r0s4_honest_inc@10`` (design_sprime_honest.md §9).

It replaces ``cost_model.py``'s per-decision inputs (``r0s4_honest``: stage 1 and the late probe) with stage 1b's
(``measurements/sprime_honest_inc_stage1b_2026-11-06/``: the incremental builder over 407 battles' whole decision
sequences, 9,758 decisions) and keeps ``cost_model.json``'s S'-LIVE inputs (the per-game decision counts by turn
bucket, kappa by bucket, the per-game overhead, the K = 6 whole-unit rate, the raw unit's CPU). Inputs, none of them a
game result:

- ``cost_model.json`` (``cost_model.py``'s output): its ``inputs.slive`` and ``inputs.kappa_overhead``;
- stage 1b's ``summary.json`` (per panel and bucket: the builder's mean world CPU-s over every decision, the forward
  and playouts of the searched targets, the exclusion rate) and ``inc_rows.jsonl.gz`` (each decision's world CPU-s,
  for the p90 by bucket);
- S'-LIVE's searched cells (``search_decisions.jsonl``: ``turn``, ``searched``, ``why``, ``deviation`` only) for every
  game's ordered search-path decisions: the watchdog over every game, the identity re-score's REBUILD cost (a sampled
  decision rebuilds every earlier checkpoint of its battle), and the share of games with a departure from pi.

    python3 cost_model_inc.py --cost-model cost_model.json --summary SUMMARY --rows INC_ROWS --slive STATE --out OUT
        [--deals 5100]
"""
import argparse
import collections
import gzip
import json
import math
import os
import statistics as S
from pathlib import Path

PANELS = {"heldout72_hpfix": "H", "forum72": "F", "p72": "P72"}
BUCKETS = (("10-19", 10, 19), ("20-29", 20, 29), ("30-63", 30, 63), ("64+", 64, 10 ** 6))
N_DEALS = 5100                     # the registered deals (85 unit seeds x 20 x 3 panels); --deals overrides
TRAINER_CPU_RATIO = 1.153          # X65-F1 (5fba84342): a pinned production trainer, per-decision CPU
SMT_GAIN = 1.63                    # stage 1b's SMT probe: a core with both siblings busy, x one worker alone (builder only)
RESCORE_N, RESCORE_MOD = 600, 16
WATCHDOG_H = 10.0


def bucket(t):
    return next(b for b, lo, hi in BUCKETS if lo <= t <= hi)


def slive_games(state):
    """Every complete S'-LIVE searched game: (panel, [bucket of each search-path decision in order], any departure)."""
    cells = Path(state) / "anchor/u/pstar_s/cells"
    out = []
    for d in sorted(os.listdir(cells)):
        if ".partial." in d:
            continue
        p = cells / d / "search_decisions.jsonl"
        if not p.is_file():
            continue
        seq, dev = [], False
        for x in open(p):
            r = json.loads(x)
            if "key" in r and (r.get("searched") or r.get("why") == "excluded"):
                seq.append(bucket(int(r["turn"])))
                dev = dev or bool(r.get("deviation"))
        out.append((PANELS[d.split("__")[1]], seq, dev))
    return out


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(f * len(xs))) - 1)] if xs else None


def main(a):
    n_deals = int(a.deals)
    n_per_panel = n_deals // 3
    cm = json.load(open(a.cost_model))
    sl, ko = cm["inputs"]["slive"], cm["inputs"]["kappa_overhead"]
    summ = json.load(open(a.summary))
    rows = [r for r in (json.loads(x) for x in gzip.open(a.rows, "rt") if x.strip())
            if "group_done" not in r and not r.get("group_error")]        # stage 1b's decision rows only
    world_p90 = {}
    for panel in ("H", "F", "P72"):
        for b, _lo, _hi in BUCKETS:
            w = [r["res"]["world_cpu_s"] + r["validator_cpu_s"] for r in rows
                 if r["panel"] == panel and r["bucket"] == b and not r["target"]]
            world_p90[(panel, b)] = q(w, 0.9)
    cost = {}                                          # (panel, bucket) -> (mean decision CPU-s, p90-world variant)
    excl = {}
    for panel in ("H", "F", "P72"):
        for b, _lo, _hi in BUCKETS:
            e = summ["by"][f"{panel}/{b}"]
            oth = e["other_cpu_s_used"]
            cost[(panel, b)] = (e["world_cpu_s"]["mean"] + oth, world_p90[(panel, b)] + oth)
            excl[(panel, b)] = e["excl_rate"]
    model = {"inputs": {"cost_model": a.cost_model, "summary": a.summary, "rows": a.rows, "slive": a.slive,
                        "per_decision_cpu_s": {f"{p}/{b}": {"mean": round(c[0], 3), "p90_world": round(c[1], 3)}
                                               for (p, b), c in cost.items()},
                        "excl_rate": {f"{p}/{b}": e for (p, b), e in excl.items()},
                        "kappa": ko["kappa"], "overhead_s": ko["overhead_s"]},
             "by_panel": {}}
    games = slive_games(a.slive)
    tot_k6 = tot_k2 = 0.0
    for panel in ("H", "F", "P72"):
        n = sl["per_game"][panel]
        dec = sum(n.values())
        proj = sum(n[b] * excl[(panel, b)] for b in n) / dec
        k2 = (sum(n[b] * cost[(panel, b)][0] for b in n) + ko["overhead_s"]["k2"]) / 60
        k6 = (sum(n[b] * cost[(panel, b)][0] * ko["kappa"][b] for b in n) + ko["overhead_s"]["k6"]) / 60
        bound = max(2 * proj, 0.05)
        hours = sorted((sum(cost[(panel, b)][0] * ko["kappa"][b] for b in seq) + ko["overhead_s"]["k6"]) / 3600
                       for p, seq, _d in games if p == panel)
        hours90 = sorted((sum(cost[(panel, b)][1] * ko["kappa"][b] for b in seq) + ko["overhead_s"]["k6"]) / 3600
                         for p, seq, _d in games if p == panel)
        model["by_panel"][panel] = {
            "search_path_decisions_a_game": round(dec, 2), "projected_excl": round(proj, 4),
            "bound": math.ceil(round(bound, 6) * 100) / 100,
            "searched_game_cpu_min_k2": round(k2, 2), "searched_game_cpu_min_k6": round(k6, 2),
            "watchdog": {"games": len(hours), "max_h_at_mean": round(hours[-1], 2),
                         "p99_h_at_mean": round(hours[int(0.99 * len(hours)) - 1], 2),
                         "max_h_at_p90_world": round(hours90[-1], 2),
                         "over_10h_at_mean_x3": sum(1 for x in hours if 3 * x > WATCHDOG_H)}}
        tot_k6 += n_per_panel * k6 / 60
        tot_k2 += n_per_panel * k2 / 60
    # THE IDENTITY RE-SCORE in REBUILD mode (live_identity --carry rebuild): a sampled decision rebuilds every earlier
    # checkpoint of its battle (the builder step alone: world CPU) and then runs in full; a 1-in-16 hash sample is a
    # uniform draw over search-path decisions, so its mean cost is the mean over every decision of every game. One
    # process on a quiet box (K = 2 CPU, kappa 1); --jobs J on the whole box costs ~kappa more CPU, J at once.
    world_mean = {(p, b): summ["by"][f"{p}/{b}"]["world_cpu_s"]["mean"] for p in ("H", "F", "P72") for b, _l, _h in BUCKETS}
    reb, seeded = collections.defaultdict(list), collections.defaultdict(list)
    for p, seq, _d in games:
        cum = 0.0
        for b in seq:
            reb[p].append(cum + cost[(p, b)][0])
            seeded[p].append(cost[(p, b)][0])
            cum += world_mean[(p, b)]
    per_dec = S.mean(S.mean(reb[p]) for p in reb)
    model["rescore"] = {
        "rebuild_cpu_s_a_decision": {p: {"mean": round(S.mean(v), 1), "median": round(q(v, 0.5), 1),
                                         "p90": round(q(v, 0.9), 1), "max": round(max(v), 1)} for p, v in reb.items()},
        "rebuild_cpu_h_600": round(RESCORE_N * per_dec / 3600, 1),
        "seeded_cpu_h_600": round(RESCORE_N * S.mean(S.mean(seeded[p]) for p in seeded) / 3600, 2),
        "sampled_expected": round(n_deals * S.mean(sl["per_game"][p][b] for p in sl["per_game"]
                                                   for b in sl["per_game"][p]) * 4 / RESCORE_MOD),
    }
    rate6 = cm["read"]["k6_cpu_h_per_wall_h"]
    raw6 = n_deals * sl["raw_unit_cpu_min_k6"] / 60
    total = tot_k6 + raw6
    per_worker = rate6 / 6
    model["read"] = {
        "searched_cpu_h_k6": round(tot_k6, 1), "searched_cpu_h_k2": round(tot_k2, 1), "raw_cpu_h_k6": round(raw6, 1), "total_cpu_h_k6": round(total, 1),
        "k6_cpu_h_per_wall_h": rate6,
        "wall_h_box_k14": round(total / (14 * per_worker), 1),
        # the SMT probe's figure instead of kappa: each worker's CPU = its K = 2 CPU x 2 / SMT_GAIN (both siblings busy)
        "wall_h_box_k14_at_smt_probe": round((tot_k2 + n_deals * sl["raw_unit_cpu_min"] / 60) * 2 / SMT_GAIN
                                             / (14 * per_worker), 1),
        "wall_h_agent_k6": round(total / rate6, 1),
        "wall_h_agent_k6_pinned_trainer": round(total * TRAINER_CPU_RATIO / rate6, 1),
        "deals": n_deals,
        "third_arm_cpu_h_k6": {k: round(n_deals * f * sl["rolls_searched_unit_cpu_min_k6"] / 60, 1)
                               for k, f in (("full", 1.0), ("half", 0.5), ("third", 1 / 3))},
        "games_with_a_departure_slive": {p: round(sum(1 for pp, _s, d in games if pp == p and d)
                                                  / sum(1 for pp, _s, _d in games if pp == p), 4)
                                         for p in ("H", "F", "P72")},
    }
    model["U"] = {k: round(S.mean(model["by_panel"][p][k] for p in ("H", "F", "P72")), 4)
                  for k in ("projected_excl", "searched_game_cpu_min_k2", "searched_game_cpu_min_k6")}
    Path(a.out).write_text(json.dumps(model, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: model[k] for k in ("by_panel", "U", "read", "rescore")}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-model", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--slive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deals", default=N_DEALS, type=int)
    main(ap.parse_args())
