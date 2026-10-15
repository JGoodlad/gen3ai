"""Mean per-update phase timings from a run's launcher_child.log metrics tables (read-only)."""
import re
import statistics as st
import sys
from pathlib import Path

KEYS = ["collect_ms", "train_ms", "update_wall_s", "fps", "host_steps", "core_ms_per_host_step",
        "gpu_wait_ms_per_host_step", "flush_ms_per_host_step", "submit_ms_per_host_step",
        "post_ms_per_host_step", "write_ms_per_host_step", "draw_ms_per_host_step",
        "opp_draw_ms_per_host_step", "fill_ms", "collect_decisions", "total_timesteps", "ep_len_mean",
        "probe_forward_ms", "flip_forward_ms"]
pat = re.compile(r"^\|\s+(\S+)\s+\|\s+([-+0-9.e]+)\s+\|")
for run in sys.argv[1:]:
    p = Path(run) / "launcher_child.log"
    tables, cur = [], {}
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("----") and cur:
            if "total_timesteps" in cur:
                tables.append(cur)
            cur = {}
            continue
        m = pat.match(line)
        if m and m.group(1) in KEYS and m.group(1) not in cur:
            try:
                cur[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    tail = [t for t in tables if "collect_ms" in t and "train_ms" in t]
    if not tail:
        print(run, "no tables")
        continue
    use = tail[len(tail) // 4:]  # drop warm-up quarter
    print(f"== {Path(run).name}: {len(tail)} tables, using last {len(use)}; steps {use[0].get('total_timesteps'):.3g}"
          f"..{use[-1].get('total_timesteps'):.3g}")
    for k in KEYS:
        v = [t[k] for t in use if k in t]
        if v:
            print(f"  {k:28s} median {st.median(v):10.4g}  mean {st.mean(v):10.4g}  n {len(v)}")
