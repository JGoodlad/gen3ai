"""Print the tables of ``analyze.py``'s JSON (procs, hourly trajectories, seams).

    python .../report.py <out.json> [procs|hourly RUN...|seams]
"""
from __future__ import annotations

import json
import sys


def main(argv):
    res = json.load(open(argv[0]))["runs"]
    mode = argv[1] if len(argv) > 1 else "procs"
    if mode == "procs":
        print(f"{'run':28} {'#':>2} {'h':>5} {'steps0':>10} {'steps1':>10} {'fps':>5} {'fpsS':>5} "
              f"{'c/100k':>6} {'train':>5} {'eval%':>5} {'eplen':>5} {'sp':>4} {'cot':>3} end prewarm")
        for run, r in res.items():
            for i, p in enumerate(r["procs"]):
                w = p["whole"]
                print(f"{run:28} {i:2} {p['hours']:5} {p['start_step']:10} {p['end_step']:10} "
                      f"{w.get('fps_all')!s:>5} {w.get('fps_steady_median')!s:>5} "
                      f"{w.get('collect_s_per_100k_median')!s:>6} {w.get('train_s_median')!s:>5} "
                      f"{round(100 * (w.get('eval_share_of_wall') or 0), 1):5} {w.get('ep_len_mean')!s:>5} "
                      f"{w.get('selfplay_fraction')!s:>4} {len(p['cotenants_whole']):3} "
                      f"{p['log'].get('end')} {p['log'].get('prewarm_s')}")
    elif mode == "hourly":
        for run in argv[2:]:
            for i, p in enumerate(res[run]["procs"]):
                print(f"== {run} proc {i} ({p['hours']} h)")
                for h in p["hourly"]:
                    print(f"  h{h['hour']:<2} fps {h.get('fps_all')!s:>5} steady {h.get('fps_steady_median')!s:>5} "
                          f"c/100k {h.get('collect_s_per_100k_median')!s:>6} train {h.get('train_s_median')!s:>5} "
                          f"other {h.get('other_s_median')!s:>5} eval% {round(100 * (h.get('eval_share_of_wall') or 0), 1):4} "
                          f"eplen {h.get('ep_len_mean')!s:>5} core {h.get('core_ms_per_host_step')!s:>6} "
                          f"gpuw {h.get('gpu_wait_ms_per_host_step')!s:>6} rows {h.get('trainee_rows_per_host_step')!s:>6} "
                          f"sp {h.get('selfplay_fraction')} rc {h.get('recompiles_after_lock')}")
    elif mode == "seams":
        for run, r in res.items():
            for s in r["seams"]:
                b, a = s["before_1h"], s["after_1h"]
                print(f"{run:26} prev_end={s['prev_end']:14} gap={s['gap_s']:>8} human={s['human_gap']!s:5} "
                      f"steps {s['steps_prev_end']}->{s['steps_resume_first']} prewarm={s['prewarm_s']} "
                      f"cache={s['cache_on_disk_gib']} excess={s['first_cycles_excess_s']} cyc={s['steady_cycle_s']}")
                print(f"{'':26}   before: fpsS {b.get('fps_steady_median')} c/100k {b.get('collect_s_per_100k_median')} "
                      f"train {b.get('train_s_median')} eplen {b.get('ep_len_mean')} sp {b.get('selfplay_fraction')} n={b.get('n_steady')} cot={s['cotenants_before']}")
                print(f"{'':26}   after : fpsS {a.get('fps_steady_median')} c/100k {a.get('collect_s_per_100k_median')} "
                      f"train {a.get('train_s_median')} eplen {a.get('ep_len_mean')} sp {a.get('selfplay_fraction')} n={a.get('n_steady')} cot={s['cotenants_after']}")
                for tag, j in s["learning"].items():
                    print(f"{'':26}   {tag:28} {j['before_mean']} -> {j['after_mean']}  jump/sd {j['jump_over_sd']}  (step-diff med {j['median_abs_step_diff']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
