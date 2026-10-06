"""Watch one capped job: print a line when its row count changes, with the scope's CPU seconds
(cgroup ``cpu.stat`` usage_usec), and append the last reading to ``<log>.cpu`` so the CPU cost
survives the scope. Exits when the job's log says ``[mem_cap] done``.

    python cpu_watch.py <log> <rows.jsonl> [--every 5] [--quiet-until N]
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path


def scope_of(log: Path):
    m = re.findall(r"unit (gen3ai-capped-[\w.-]+\.scope)", log.read_text() if log.exists() else "")
    return m[-1] if m else None


def cpu_s(scope: str):
    for p in Path("/sys/fs/cgroup").rglob(scope):
        st = p / "cpu.stat"
        if st.exists():
            for line in st.read_text().splitlines():
                if line.startswith("usage_usec"):
                    return int(line.split()[1]) / 1e6
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("rows")
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument("--report-every-rows", type=int, default=1)
    a = ap.parse_args()
    log, rows = Path(a.log), Path(a.rows)
    prev, last_cpu, t0 = -1, None, time.time()
    rows_at_last = -10**9
    while True:
        txt = log.read_text() if log.exists() else ""
        txt = txt[txt.rfind("[mem_cap] cap"):] if "[mem_cap] cap" in txt else txt   # this launch only
        n = sum(1 for x in rows.read_text().splitlines() if x.strip()) if rows.exists() else 0
        sc = scope_of(log)
        c = cpu_s(sc) if sc else None
        if c is not None:
            last_cpu = c
            Path(str(log) + ".cpu").write_text(f"{sc} cpu_s={c:.1f} rows={n} t={time.time():.0f}\n")
        if n != prev and n - rows_at_last >= a.report_every_rows:
            print(f"rows={n} cpu_s={last_cpu} wall_s={time.time() - t0:.0f}", flush=True)
            rows_at_last = n
        prev = n
        if "[mem_cap] done" in txt or "Traceback" in txt:
            print(f"END rows={n} cpu_s={last_cpu} :: {txt.strip().splitlines()[-1][:200]}", flush=True)
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
