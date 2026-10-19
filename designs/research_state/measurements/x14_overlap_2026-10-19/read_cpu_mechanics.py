"""Summarise `result/cpu_mechanics.json` (per arm, over the iterations after the first two updates)."""
import json
import statistics as st
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "result/cpu_mechanics.json"))
print("config", d["config"], "loadavg at end", [round(x, 1) for x in d["loadavg_at_end"]])
for r in d["runs"]:
    rows = [x for x in r["dumps"][2:] if "train/train_ms" in x]       # steady state: from the 3rd dump with an update
    def med(k):
        v = [x[k] for x in rows if k in x]
        return float(f"{st.median(v):.4g}") if v else None
    print(f"{'ON ' if r['overlap'] else 'OFF'} rep {r['rep']}: wall {r['wall_s']:.1f} s, {r['decisions']} decisions "
          f"({r['decisions'] / r['wall_s']:.1f}/s), updates {r['updates']}, overlapped {r['overlapped_rollouts']}, "
          f"peak RSS {r['peak_rss_mib']:.0f} MiB, snapshots {r['snapshot_bytes'] / 2**20:.1f} MiB | medians: "
          f"train {med('train/train_ms')} ms, exposed collect {med('rollout/collect_ms')} ms, play {med('overlap/play_ms')}, "
          f"join wait {med('overlap/join_wait_ms')}, prox {med('overlap/prox_forward_ms')}, ESS {med('overlap/is_ess')}, "
          f"w {med('overlap/is_weight_mean')}, KL(mu->prox) {med('overlap/behaviour_kl')}, approx_kl {med('train/approx_kl')}, "
          f"snapshot |dlogp| {med('behaviour_snapshot/max_abs_dlogp_judged')}")
