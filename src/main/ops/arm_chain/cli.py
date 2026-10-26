"""``python -m main.ops.arm_chain <verb>`` — the arm chain's command line (design: ``designs/ops/arm_chain.md``).

    plan        CHAIN [--check-pins] [--describe]   validate the chain file; --describe asks the kit for each arm's run
    run         CHAIN                               the SUPERVISOR: walk the items from the saved state (nohup setsid)
    status      CHAIN                               each item's phase and hold
    clear-hold  CHAIN --item KEY --by WHO --why WHY the orchestrator's decision to retry a held step
    abandon     CHAIN --item KEY --by WHO --why WHY the orchestrator's decision that a held item will not finish
    ship-records CHAIN --item KEY                   ship one arm's records (the async worker; also the by-hand retry)
    live        [--models-dir D]                    live runs from launcher argv (never a glob)
    watch-files --status-file F [--models-dir D]    the alert watch's file list: F + each live launcher's stdout log
                                                    + its run's launcher_child.full.log
    monitor-filter --chain NAME                     the ONE ERE the alert Monitor greps over the watch-files
    canary      --out-dir D [--models-dir D] …      the canary logger loop (bounded, with an EXPIRING line)
    cpu-gate                                        one CPU-gate sample (exit 0 PASS, 1 FAIL, 2 cannot measure)
    wait-pid    PID --timeout-s N                   wait for PID (identity = pid + start ticks) to go; 0 gone, 1 timeout
    signal-if   PID ERE [--signal TERM]             signal PID only if its cmdline matches ERE (exit 0 sent, 2 not)

Exit codes of ``run``: 0 the chain advanced cleanly (or is complete), 3 HOLD (a status line says why), 4 another
supervisor holds this chain. Nothing here touches the GPU except a GO and a window's steps, both inside ``run``.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from typing import List, Optional, Sequence

from main.ops.arm_chain import canary, cpugate, live, spec as spec_mod
from main.ops.arm_chain.chain import Chain, ChainError, ship_worker
from main.ops.arm_chain.launch import Hold, X35Launcher
from main.ops.arm_chain.procwatch import ProcRef, ref_of, signal_if, wait_gone
from main.ops.arm_chain.runner import SubprocessRunner
from main.ops.arm_chain.statuslog import StatusLog


def _models_default() -> str:
    from utils.paths import main_models_dir
    return str(main_models_dir() or "")


def _plan(a: argparse.Namespace) -> int:
    s = spec_mod.load(a.chain)
    for i, it in enumerate(s.items):
        print(f"{i:2d}. {spec_mod.describe_item(it)}")
    bad: List[str] = []
    if a.check_pins:
        bad += spec_mod.check_pins(s)
    if a.describe:
        lau = X35Launcher(s, SubprocessRunner(), dict(os.environ))
        for it in s.arms():
            try:
                d = lau.describe_offline(it)
                print(f"    {it.key}: the kit agrees — FORK of {d.parent_run_dir}, {d.delta_text}, --steps {d.steps:,}")
            except Hold as h:
                bad.append(str(h))
    for b in bad:
        print(f"REFUSED: {b}")
    print("PLAN OK" if not bad else f"PLAN REFUSED ({len(bad)} problem(s))")
    return 0 if not bad else 1


def _chain(a: argparse.Namespace) -> Chain:
    return Chain(spec_mod.load(a.chain))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.ops.arm_chain", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("chain")
    p.add_argument("--check-pins", action="store_true")
    p.add_argument("--describe", action="store_true")
    for name in ("run", "status"):
        sub.add_parser(name).add_argument("chain")
    for name in ("clear-hold", "abandon"):
        c = sub.add_parser(name)
        c.add_argument("chain")
        c.add_argument("--item", required=True)
        c.add_argument("--by", required=True)
        c.add_argument("--why", required=True)
    sr = sub.add_parser("ship-records")
    sr.add_argument("chain")
    sr.add_argument("--item", required=True)
    mf = sub.add_parser("monitor-filter")
    mf.add_argument("--chain", required=True, help="the chain file's name (the ARMCHAIN[<name>] tag)")
    lv = sub.add_parser("live")
    lv.add_argument("--models-dir", default=None)
    wf = sub.add_parser("watch-files")
    wf.add_argument("--status-file", required=True)
    wf.add_argument("--models-dir", default=None)
    cw = sub.add_parser("canary")
    cw.add_argument("--out-dir", required=True)
    cw.add_argument("--status-file", required=True, help="where an alert line goes ('CANARYLOG: HOLD: …')")
    cw.add_argument("--models-dir", default=None)
    cw.add_argument("--interval-s", type=float, default=300.0)
    cw.add_argument("--max-hours", type=float, default=168.0)
    cw.add_argument("--archive-root", default=os.path.expanduser("~/gen3ai_archive"),
                    help="where canary weight dumps are copied (<root>/<run>_canary_u<u>/, with a .sha256)")
    cw.add_argument("--first-record-min", type=float, default=60.0)
    g = sub.add_parser("cpu-gate")
    g.add_argument("--trainer-cores", default="0-4,8-12")
    g.add_argument("--agent-cores", default="5-7,13-15")
    g.add_argument("--sample-s", type=float, default=60.0)
    w = sub.add_parser("wait-pid")
    w.add_argument("pid", type=int)
    w.add_argument("--timeout-s", type=float, required=True)
    w.add_argument("--poll-s", type=float, default=30.0)
    si = sub.add_parser("signal-if")
    si.add_argument("pid", type=int)
    si.add_argument("ere")
    si.add_argument("--signal", default="TERM")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "plan":
            return _plan(a)
        if a.cmd == "run":
            ch = _chain(a)
            rc = ch.run()
            if rc == 4:
                print(f"another supervisor holds {ch.spec.state_path}.lock — not started", file=sys.stderr)
            print("\n".join(ch.status_lines()))
            return rc
        if a.cmd == "status":
            print("\n".join(_chain(a).status_lines()))
            return 0
        if a.cmd == "clear-hold":
            print(_chain(a).clear_hold(a.item, a.by, a.why))
            return 0
        if a.cmd == "abandon":
            print(_chain(a).abandon(a.item, a.by, a.why))
            return 0
        if a.cmd == "ship-records":
            s = spec_mod.load(a.chain)
            return ship_worker(s, a.item, Chain(s)._default_shipper(s))
    except (spec_mod.SpecError, ChainError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if a.cmd == "monitor-filter":
        print(canary.monitor_filter(a.chain))
        return 0
    md = getattr(a, "models_dir", None) or _models_default()
    if a.cmd == "live":
        for r in live.discover(md):
            print(r.line())
        return 0
    if a.cmd == "watch-files":
        log = StatusLog(a.status_file, "AWFILES")
        print(a.status_file)
        alert = canary.Alerter(a.status_file + ".awfiles_alerted", lambda m: log.append(f"HOLD: {m}"))
        for f in canary.watch_files(live.discover(md), alert):
            print(f)
        return 0
    if a.cmd == "canary":
        log = StatusLog(a.status_file, "CANARYLOG")
        return canary.watch(md, a.out_dir, lambda m: log.append(f"HOLD: {m}" if "EXPIRING" not in m else m),
                            archive_root=a.archive_root, interval_s=a.interval_s, max_hours=a.max_hours,
                            first_record_min=a.first_record_min)
    if a.cmd == "cpu-gate":
        r = cpugate.sample(a.trainer_cores, a.agent_cores, sample_s=a.sample_s, me=os.getpid())
        print(r.line)
        return 0 if r.ok else (1 if r.measured else 2)
    if a.cmd == "wait-pid":
        ref = ref_of(a.pid)
        if ref is None:
            print(f"pid {a.pid} is not running")
            return 0
        gone = wait_gone(ref, a.timeout_s, poll_s=a.poll_s)
        print(f"pid {a.pid} {'gone' if gone else 'STILL ALIVE at the timeout'}")
        return 0 if gone else 1
    if a.cmd == "signal-if":
        ref = ref_of(a.pid) or ProcRef(a.pid, -1)
        ok, msg = signal_if(ref, a.ere, getattr(signal, f"SIG{a.signal.upper()}"))
        print(msg)
        return 0 if ok else 2
    raise AssertionError(a.cmd)

