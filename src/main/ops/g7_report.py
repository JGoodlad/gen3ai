"""THE G7 REPORT — the incident fix (ledger 391df12c) + READ AMENDMENT 6, in one command.

    python -m main.ops.g7_report <run-name|run-dir> --parent NAME --famine-comparator NAME
                                 [--ep-bots-ref F --ep-bots-ref-n N] [--ep-pool-anchor F]
                                 [--bar F] [--timeout S]

🚨 A G7 LINE IN ANY REPORT IS A QUOTE OF critic_gate's ROW. NEVER AN INFERENCE.
G7 fired KILL at 32M and 34M and went unnoticed for four hours because seven report lines
said "G7 untouched" from inference while the tool sat one command away. This script RUNS the
tool and prints its rows verbatim; if the tool fails it prints NOT MEASURED and nothing else.
"untouched" and "clean" are not available as outputs.

READ AMENDMENT 6 (+6b, ledger 48747444) — what the printed rows mean:
  stall-rate half   <= 0.05 on eval traces (250-turn cap). UNCHANGED, and the PRIMARY stall
                    signal.
  ep_bots half      re-referenced to the arm's OWN first 2M inside the self-play regime,
                    bar 1.25x, SUSTAINED over two consecutive snapshots. A breach is a
                    REPORT, never a kill.
  ep_pool half      DESCRIPTIVE, NO BAR. Its population is NON-STATIONARY by construction: the
                    sentinels are drawn from a pool that gains a stronger self every 2M, so
                    eval-vs-pool episode length rises as the sentinels strengthen. A reference
                    frozen at any early window is a reference from a DIFFERENT opponent
                    population — the same non-commensurability that dismissed the fold-parent
                    era's 18.28. Choosing the window that passes would be fitting the bar to
                    the data, so there is no bar. The value and its ratio to the anchor are
                    printed.

🚨 THE TOOL'S OWN ERA REFERENCE IS NOT AMENDMENT 6's. critic_gate still prints KILL/OK against
the parent's era; that verdict column is SUPERSEDED for an arm carrying its own frozen
reference, and is reproduced only because it is part of the quoted row.

THE REFERENCES ARE ARGUMENTS, and every one of them is PRINTED at the read. They are per-arm
registered facts (the arm's own 4M-6M window), not properties of the tool: an inherited default
from another arm is exactly the substitution AMENDMENT 6 exists to prevent. The defaults are the
values registered for the arm this was written for, and the report names them.

Promoted 2026-09-07 from the Training Run session's ``g7_report.py`` (the run directory, the
interpreter, the repo path and both baseline registry names were literals).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Sequence

from main.ops.run_ref import interpreter, refuse, resolve_run_dir
from utils.paths import repo_root, src_root

#: The arm this was written for: its OWN 4M-6M eval/mean_ep_len_vs_bots, frozen, n=2 cycles.
DEFAULT_EP_BOTS_REF, DEFAULT_EP_BOTS_N = 21.911, 2
#: Descriptive anchor only, n=1 cycle. There is no bar on the pool stratum.
DEFAULT_EP_POOL_ANCHOR = 25.890
DEFAULT_BAR = 1.25


def _opt(args, name, cast, default):
    return cast(args[args.index(name) + 1]) if name in args else default


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    run = resolve_run_dir(args[0])
    if "--parent" not in args or "--famine-comparator" not in args:
        refuse("REFUSING: --parent and --famine-comparator are both required.",
               "  They are BASELINE REGISTRY names (designs/baselines.json). The read states what",
               "  the arm is measured against by name; it is never assumed. Nothing is concluded.")
    parent = args[args.index("--parent") + 1]
    comparator = args[args.index("--famine-comparator") + 1]
    ep_bots_ref = _opt(args, "--ep-bots-ref", float, DEFAULT_EP_BOTS_REF)
    ep_bots_n = _opt(args, "--ep-bots-ref-n", int, DEFAULT_EP_BOTS_N)
    ep_pool_anchor = _opt(args, "--ep-pool-anchor", float, DEFAULT_EP_POOL_ANCHOR)
    bar = _opt(args, "--bar", float, DEFAULT_BAR)
    timeout = _opt(args, "--timeout", int, 560)

    env = dict(os.environ,
               PYTHONPATH=os.environ.get("PYTHONPATH", "") + os.pathsep + str(src_root()))
    cmd = [interpreter(), "-m", "main.critic_gate", str(run), "--parent", parent,
           "--famine-comparator", comparator, "--skip-meter"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                             cwd=str(repo_root())).stdout
    except Exception as e:  # noqa: BLE001 - any failure is the same outcome: NOT MEASURED
        print(f"G7: NOT MEASURED — critic_gate failed to run ({type(e).__name__}: {e})")
        print("  This is NOT 'untouched' and NOT 'clean'. The condition's state is UNKNOWN.")
        return 2

    m = re.search(r"\(3\) G7 KILL CONDITION.*?stall rate source:.*?\n", out, re.S)
    if not m:
        print("G7: NOT MEASURED — critic_gate ran but emitted no G7 section.")
        print("  This is NOT 'untouched' and NOT 'clean'. The condition's state is UNKNOWN.")
        return 2

    print("=== G7 — QUOTED FROM critic_gate (incident fix 391df12c) ===")
    print(f"    run {run}   parent {parent}   famine comparator {comparator}")
    print(m.group(0).rstrip())

    rows = []
    for line in m.group(0).splitlines():
        f = line.split()
        if len(f) >= 5 and re.match(r"^[\d,]+$", f[0]):
            def num(x):
                try:
                    return float(x)
                except ValueError:
                    return None
            rows.append((int(f[0].replace(",", "")), num(f[1]), num(f[3]), num(f[4])))

    print("\n=== READ AMENDMENT 6 / 6b (ledger 48747444) — the references THIS ARM is judged "
          "against ===")
    print(f"  ep_bots  reference {ep_bots_ref:.3f} (the arm's OWN early self-play window, "
          f"n={ep_bots_n} cycles)  "
          f"bar {bar}x = {ep_bots_ref*bar:.2f}, SUSTAINED over two consecutive snapshots")
    print("  ep_pool  DESCRIPTIVE, NO BAR — non-stationary population (the sentinels strengthen")
    print(f"           every 2M); ratio shown against the anchor {ep_pool_anchor:.3f} for orientation")
    print("  stall    <= 0.05, unchanged, and the PRIMARY stall signal")
    print("  ⚠️  the quoted rows' KILL/OK column uses the PARENT's era reference, which may sit")
    print("      below every value this arm has produced. That column is SUPERSEDED here.\n")

    print(f"  {'step':>12}{'stall':>8}{'ep_bots':>9}{'ratio':>7}  bar?   {'ep_pool':>9}"
          f"{'ratio':>7}  (desc)")
    for st_, stall, eb, ep in rows[-6:]:
        rb = eb / ep_bots_ref if eb else None
        rp = ep / ep_pool_anchor if ep else None
        flag = "OVER" if (rb and rb > bar) else ""
        print(f"  {st_:>12,}{stall if stall is not None else float('nan'):>8.4f}"
              f"{eb if eb else float('nan'):>9.2f}{rb if rb else float('nan'):>7.3f}  {flag:<5}"
              f"  {ep if ep else float('nan'):>9.2f}{rp if rp else float('nan'):>7.3f}")

    allover = [st_ for st_, _, eb, _ in rows if eb and eb / ep_bots_ref > bar]
    sustained = (len(allover) >= 2
                 and any(rows[i][0] in allover and rows[i - 1][0] in allover
                         for i in range(1, len(rows))))
    print(f"\n  ep_bots over the {bar}x bar at: "
          f"{[f'{s/1e6:.0f}M' for s in allover] if allover else 'NO SNAPSHOT'}")
    print(f"  SUSTAINED over two consecutive snapshots: {'YES — REPORT' if sustained else 'no'}")
    stalls = [s for s, st_, _, _ in rows if st_ is not None and st_ > 0.05]
    print(f"  stall-rate half (primary): "
          f"{'BREACH at ' + str(stalls) if stalls else 'no snapshot above 0.05'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
