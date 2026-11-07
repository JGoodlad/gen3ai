"""S'-HONEST stage 1b: the summary's tables in markdown (`seq_probe.py summarize`'s summary.json in, the README's and
the design doc §9.5's tables out).

    python3 tables.py SUMMARY.json
"""
import json
import sys

BUCKETS = ("10-19", "20-29", "30-63", "64+")


def pct(x):
    return f"{100 * x:.1f} %"


def frac(p):
    a, n = p
    return f"{a} / {n} ({100 * a / n:.0f} %)" if n else "—"


def main(path):
    s = json.load(open(path))
    out = []
    out.append("**Per panel and turn bucket** (world CPU-s over EVERY decision of the sequences, exclusions included; "
               "\"other\" = the forward and playouts of the searched targets; exclusions over every decision; at the "
               "targets, both builders' exclusions and full-decision CPU-s):\n")
    out.append("| panel / bucket | decisions | world CPU-s mean (median, p90, max) | other | excl. (all decisions) | "
               "targets | excl. at targets inc / honest | CPU-s at targets inc / honest | same pick inc vs honest |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(s["by"], key=lambda k: (k.split("/")[0], BUCKETS.index(k.split("/")[1]))):
        v = s["by"][k]
        w = v["world_cpu_s"]
        out.append(f"| {k} | {v['decisions']} | {w['mean']} ({w['median']}, {w['p90']}, {w['max']}) | "
                   f"{v['other_cpu_s_used']} | {pct(v['excl_rate'])} | {v['targets']} | {v['targets_excluded']} / "
                   f"{v['honest_targets_excluded']} | {v['target_cpu_s_mean']} / {v['honest_target_cpu_s_mean']} | "
                   f"{frac(v['same_pick_vs_honest'])} |")
    out.append("\n**Projected per searched game** (S′-LIVE's search-path decisions a game by turn bucket × the bucket's "
               "mean decision CPU, plus S′-LIVE's per-game overhead; K = 6 with its κ by bucket: the replication "
               "designer's cost model, `sprime_honest_play_2026-11-06/inputs/cost_model.json`):\n")
    out.append("| panel | CPU-min K = 2 inc / honest | CPU-min K = 6 inc / honest | projected exclusions inc / honest |")
    out.append("|---|---|---|---|")
    for p, v in list(s["panels"].items()) + [("**U**", s["U"])]:
        out.append(f"| {p} | {v['game_cpu_min_k2']:.1f} / {v['honest_game_cpu_min_k2']:.1f} | "
                   f"{v['game_cpu_min_k6']:.1f} / {v['honest_game_cpu_min_k6']:.1f} | {pct(v['projected_excl'])} / "
                   f"{pct(v['honest_projected_excl'])} |")
    r = s["read"]
    out.append("\n**The read** (3,840 searched games and their raw twins):\n")
    out.append("| builder | CPU-h (K = 6) | wall days, agent cores K = 6 | wall days, whole box K = 14 |")
    out.append("|---|---|---|---|")
    for b in ("inc", "honest"):
        out.append(f"| {b} | {r[b]['cpu_h_k6']:.0f} | {r[b]['wall_days_agent_k6']:.1f} | {r[b]['wall_days_box_k14']:.1f} |")
    out.append("\n**Agreement at the targets** (both searched; the floors are each builder against its own replicate "
               "under another key tag, on the replicate subset):\n")
    out.append("| panel | inc vs honest | honest vs honest′ | inc vs honest on that subset | inc vs inc′ | departs from π: "
               "inc / honest |")
    out.append("|---|---|---|---|---|---|")
    for p in ("H", "F", "P72", "U"):
        a = s["agreement"][p]
        out.append(f"| {p} | {frac(a['inc_vs_honest'])} | {frac(a.get('honest_vs_honest_rep', [0, 0]))} | "
                   f"{frac(a.get('inc_vs_honest_on_floor', [0, 0]))} | {frac(a.get('inc_vs_inc_rep', [0, 0]))} | "
                   f"{pct(a['deviation_inc'])} / {pct(a['deviation_honest'])} |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
