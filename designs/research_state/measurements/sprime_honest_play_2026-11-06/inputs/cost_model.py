"""S'-HONEST play PREREG input (2026-11-06, OUTCOME-BLIND): the cost, wall-time, exclusion-bound and watchdog model.

Inputs, none of them a game result:
- S'-LIVE's searched cells (`search_decisions.jsonl` of every complete attempt dir: `turn`, `searched`, `why`, `cpu_s`
  only) for its per-panel count of search-path decisions a game in each turn bucket, and each game's bucket counts
  (the watchdog's worst game);
- S'-LIVE's `units.jsonl` and supervisor log: the whole-unit CPU and the K = 2 / K = 6 lives (kappa, the overhead,
  the CPU-h a wall hour at K = 6);
- stage 1's `final_summary.json` (r0s4_honest at budget B3 on dlive's states, turns 10-30): the early buckets;
- the late probe's `rows.jsonl` (r0s4_honest and r0s4_rolls on S'-LIVE's own sampled states): the late buckets.

    python3 cost_model.py --slive STATE --stage1 FINAL_SUMMARY --probe ROWS --out OUT.json
"""
import argparse
import collections
import json
import math
import os
import statistics as S
from pathlib import Path

PANELS = {"heldout72_hpfix": "H", "forum72": "F", "p72": "P72"}
BUCKETS = (("10-19", 10, 19), ("20-29", 20, 29), ("30-63", 30, 63), ("64+", 64, 10 ** 6))
EARLY, LATE = ("10-19", "20-29"), ("30-63", "64+")
N_DEALS, N_PER_PANEL = 3840, 1280
TRAINER_CPU_RATIO = 1.153          # X65-F1 (5fba84342): a pinned production trainer, per-decision CPU


def bucket(t):
    return next(b for b, lo, hi in BUCKETS if lo <= t <= hi)


def slive_inputs(state):
    cells = Path(state) / "anchor/u/pstar_s/cells"
    per = collections.defaultdict(lambda: collections.Counter())      # panel -> bucket -> search-path decisions
    games = collections.Counter()
    each = collections.defaultdict(list)                                # panel -> every game's bucket counts
    worst = {}                                                          # panel -> the game with most search-path decisions
    for d in sorted(os.listdir(cells)):
        if ".partial." in d:
            continue
        p = cells / d / "search_decisions.jsonl"
        if not p.is_file():
            continue
        panel = PANELS[d.split("__")[1]]
        games[panel] += 1
        c = collections.Counter()
        for x in open(p):
            r = json.loads(x)
            if "key" in r and (r.get("searched") or r.get("why") == "excluded"):
                c[bucket(int(r["turn"]))] += 1
        per[panel].update(c)
        each[panel].append(dict(c))
        if sum(c.values()) > sum(worst.get(panel, collections.Counter()).values()):
            worst[panel] = c
    sup = [json.loads(x) for x in open(Path(state) / "anchor_supervisor.jsonl")]
    starts = [(e["t"], e.get("concurrency")) for e in sup if e["event"] == "start"]

    def life(t):
        k = None
        for i, (t0, _c) in enumerate(starts):
            if t >= t0:
                k = i
        return k

    unit_cpu = collections.defaultdict(list)
    t_end = collections.defaultdict(float)
    cpu_life = collections.defaultdict(float)
    n_life = collections.Counter()
    for lab in ("pstar_s", "pstar"):
        for x in open(Path(state) / f"anchor/u/{lab}/units.jsonl"):
            e = json.loads(x)
            if e["event"] != "end":
                continue
            L = life(e["t"])
            unit_cpu[(lab, L)].append(float(e.get("cpu_s") or 0.0))
            cpu_life[L] += float(e.get("cpu_s") or 0.0)
            t_end[L] = max(t_end[L], e["t"])
            n_life[(lab, L)] += 1
    k6 = [i for i, (_t, c) in enumerate(starts) if c == 6]
    rate = {i: cpu_life[i] / 3600 / ((t_end[i] - starts[i][0]) / 3600) for i in k6}
    deals_h = {i: n_life[("pstar_s", i)] / ((t_end[i] - starts[i][0]) / 3600) for i in k6}
    raw_cpu_min = S.mean([c for (lab, _L), v in unit_cpu.items() if lab == "pstar" for c in v]) / 60
    raw_k6_min = S.mean([c for i in k6 for c in unit_cpu[("pstar", i)]]) / 60
    rolls_k6_min = S.mean([c for i in k6 for c in unit_cpu[("pstar_s", i)]]) / 60
    return {"per_game": {p: {b: per[p][b] / games[p] for b, _lo, _hi in BUCKETS} for p in per},
            "games": dict(games), "each_game": dict(each), "worst_game": {p: dict(c) for p, c in worst.items()},
            "k6_cpu_h_per_wall_h": rate, "k6_searched_deals_per_h": deals_h, "raw_unit_cpu_min": raw_cpu_min,
            "raw_unit_cpu_min_k6": raw_k6_min, "rolls_searched_unit_cpu_min_k6": rolls_k6_min}


def kappa_and_overhead(state):
    """K = 6 (life 4) over K = 2 (life 1) per-decision searched CPU by bucket, and the whole-unit overhead (whole-unit
    CPU minus the decision records' CPU) at each."""
    st = Path(state)
    sup = [json.loads(x) for x in open(st / "anchor_supervisor.jsonl")]
    starts = [(e["t"], e.get("concurrency")) for e in sup if e["event"] == "start"]

    def life(t):
        k = None
        for i, (t0, _c) in enumerate(starts):
            if t >= t0:
                k = i
        return k

    ends = {}
    for x in open(st / "anchor/u/pstar_s/units.jsonl"):
        e = json.loads(x)
        if e["event"] == "end" and e.get("counted"):
            ends[e["unit"]] = e
    acc = collections.defaultdict(list)
    over = collections.defaultdict(list)
    for u, e in ends.items():
        p = st / "anchor/u/pstar_s/cells" / u / "search_decisions.jsonl"
        if not p.is_file():
            continue
        L = life(e["t"])
        s = 0.0
        for x in open(p):
            r = json.loads(x)
            if "key" not in r:
                continue
            s += float(r.get("cpu_s") or 0.0)
            if r.get("searched"):
                acc[(L, bucket(int(r["turn"])))].append(float(r["cpu_s"]))
        over[L].append(float(e["cpu_s"]) - s)
    # the K = 2 life with the most searched units (life 1: 1,009 of them), never the 60-unit checkpoint life
    k2 = sorted((i for i, (_t, c) in enumerate(starts) if c == 2 and acc.get((i, "10-19"))),
                key=lambda i: -len(over[i]))
    k6 = [i for i, (_t, c) in enumerate(starts) if c == 6]
    out = {"kappa": {}, "life_k2": k2[0], "life_k6": k6[0]}
    for b, _lo, _hi in BUCKETS:
        out["kappa"][b] = S.mean(acc[(k6[0], b)]) / S.mean(acc[(k2[0], b)])
        out.setdefault("rolls_k2_cpu_s", {})[b] = S.mean(acc[(k2[0], b)])
    out["overhead_s"] = {"k2": S.mean(over[k2[0]]), "k6": S.mean(over[k6[0]])}
    return out


def probe_inputs(rows_path):
    rows = [json.loads(x) for x in Path(rows_path).read_text().splitlines() if x.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["panel"], r["bucket"])].append(r)
    out = {}
    for (panel, b), rs in by.items():
        h = [r["res"]["r0s4_honest@10"] for r in rs]
        o = [r["res"]["r0s4_rolls@10"] for r in rs]
        hc = sorted(x["cpu_s"] + x["validator_cpu_s"] for x in h)
        oc = [x["cpu_s"] + x["validator_cpu_s"] for x in o]
        out[(panel, b)] = {"n": len(rs), "excl": sum(1 for x in h if x["why"] == "excluded"),
                           "rolls_excl": sum(1 for x in o if x["why"] == "excluded"),
                           "cpu_mean": S.mean(hc), "cpu_p90": hc[min(len(hc) - 1, int(math.ceil(0.9 * len(hc))) - 1)],
                           "cpu_max": hc[-1], "rolls_cpu_mean": S.mean(oc),
                           "ratio": sum(hc) / sum(oc),
                           "same_pick": [sum(1 for a, c in zip(h, o) if a["searched"] and c["searched"]
                                             and a["played"] == c["played"]),
                                         sum(1 for a, c in zip(h, o) if a["searched"] and c["searched"])],
                           "rolls_reproduces_live": sum(1 for r in rs if r["res"]["r0s4_rolls@10"]["played"]
                                                        == r["live_rolls"]["played"])}
    return out


def main(a):
    sl = slive_inputs(a.slive)
    ko = kappa_and_overhead(a.slive)
    st1 = json.load(open(a.stage1))["panels"]
    pr = probe_inputs(a.probe)
    model = {"inputs": {"slive": {k: v for k, v in sl.items() if k != "each_game"}, "kappa_overhead": ko,
                        "probe": {f"{p}/{b}": v for (p, b), v in sorted(pr.items())}}, "by_panel": {}}
    tot_k6 = 0.0
    tot_k2 = 0.0
    for panel in ("H", "F", "P72"):
        h1 = st1[panel]["honest"]
        early = {"excl": h1["exclusion_rate"], "cpu": h1["cpu_s_mean"] + h1["validator_cpu_s_mean"]}
        n = sl["per_game"][panel]
        ex_num, dec, cpu_k2, cpu_k6, cpu_k6_p90 = 0.0, 0.0, 0.0, 0.0, 0.0
        parts = {}
        for b, _lo, _hi in BUCKETS:
            if b in EARLY:
                e, c, c90 = early["excl"], early["cpu"], early["cpu"]
            else:
                q = pr.get((panel, b))
                if q is None:
                    continue
                e, c, c90 = q["excl"] / q["n"], q["cpu_mean"], q["cpu_p90"]
            ex_num += n[b] * e
            dec += n[b]
            cpu_k2 += n[b] * c
            cpu_k6 += n[b] * c * ko["kappa"][b]
            cpu_k6_p90 += n[b] * c90 * ko["kappa"][b]
            parts[b] = {"decisions_a_game": round(n[b], 2), "excl": round(e, 4), "cpu_s": round(c, 2)}
        g_k2 = (cpu_k2 + ko["overhead_s"]["k2"]) / 60
        g_k6 = (cpu_k6 + ko["overhead_s"]["k6"]) / 60
        proj = ex_num / dec
        bound = max(2 * proj, 0.05)
        w = sl["worst_game"][panel]
        def worst_h(stat):
            s = 0.0
            for b, _lo, _hi in BUCKETS:
                if b in EARLY:
                    c = early["cpu"]
                else:
                    q = pr.get((panel, b))
                    c = q[stat] if q else 0.0
                s += w.get(b, 0) * c * ko["kappa"][b]
            return s / 3600
        model["by_panel"][panel] = {
            "parts": parts, "search_path_decisions_a_game": round(dec, 2), "projected_excl": round(proj, 4),
            "bound": math.ceil(bound * 100) / 100, "searched_game_cpu_min_k2": round(g_k2, 2),
            "searched_game_cpu_min_k6": round(g_k6, 2),
            "searched_game_cpu_min_k6_at_p90": round((cpu_k6_p90 + ko["overhead_s"]["k6"]) / 60, 2),
            "worst_game": {"buckets": w, "wall_h_k6_at_mean": round(worst_h("cpu_mean"), 2),
                           "wall_h_k6_at_p90": round(worst_h("cpu_p90"), 2),
                           "wall_h_k6_at_max": round(worst_h("cpu_max"), 2)}}
        tot_k6 += N_PER_PANEL * g_k6 / 60
        tot_k2 += N_PER_PANEL * g_k2 / 60
    # THE WATCHDOG: every S'-LIVE searched game's bucket counts at this read's per-decision cost (the probe's means;
    # x1.5 as a sensitivity for a game whose decisions are all harder than the bucket's mean), at K = 6
    wd = {}
    for panel in ("H", "F", "P72"):
        h1 = st1[panel]["honest"]
        early = h1["cpu_s_mean"] + h1["validator_cpu_s_mean"]
        cost = {b: (early if b in EARLY else (pr[(panel, b)]["cpu_mean"] if (panel, b) in pr else 0.0)) for b, _l, _h
                in BUCKETS}
        hours = sorted((sum(g.get(b, 0) * cost[b] * ko["kappa"][b] for b, _l, _h in BUCKETS)
                        + ko["overhead_s"]["k6"]) / 3600 for g in sl["each_game"][panel])
        n = len(hours)
        wd[panel] = {"games": n, "max_h": round(hours[-1], 2), "p99_h": round(hours[int(0.99 * n) - 1], 2),
                     "over_10h": sum(1 for x in hours if x > 10), "over_10h_at_x1.5": sum(1 for x in hours if 1.5 * x > 10),
                     "over_8h": sum(1 for x in hours if x > 8)}
    model["watchdog"] = wd
    raw_k6 = N_DEALS * sl["raw_unit_cpu_min_k6"] / 60
    rate = max(sl["k6_cpu_h_per_wall_h"].values())
    model["read"] = {"searched_cpu_h_k6": round(tot_k6, 1), "searched_cpu_h_k2": round(tot_k2, 1),
                     "raw_cpu_h_k6": round(raw_k6, 1),
                     "total_cpu_h_k6": round(tot_k6 + raw_k6, 1),
                     "k6_cpu_h_per_wall_h": round(rate, 2),
                     "wall_days_free": round((tot_k6 + raw_k6) / rate / 24, 1),
                     "wall_days_pinned_trainer": round((tot_k6 + raw_k6) * TRAINER_CPU_RATIO / rate / 24, 1),
                     "rent_box_h_at_30_45": [round((tot_k2 + N_DEALS * sl["raw_unit_cpu_min"] / 60) / k, 1)
                                             for k in (45, 30)],
                     "third_arm_cpu_h_k6": {"full": round(N_DEALS * sl["rolls_searched_unit_cpu_min_k6"] / 60, 1),
                                            "half": round(N_DEALS / 2 * sl["rolls_searched_unit_cpu_min_k6"] / 60, 1),
                                            "third": round(N_DEALS / 3 * sl["rolls_searched_unit_cpu_min_k6"] / 60, 1)}}
    Path(a.out).write_text(json.dumps(model, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"by_panel": {p: {k: v for k, v in d.items() if k != "parts"} for p, d in model["by_panel"].items()},
                      "read": model["read"], "watchdog": model["watchdog"], "kappa": ko}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slive", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
