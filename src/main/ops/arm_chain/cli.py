"""``python -m main.ops.arm_chain <verb>`` — the arm chain's command line (design: ``designs/ops/arm_chain.md``).

    plan        CHAIN [--check-pins] [--describe]   validate the chain file; --describe asks the kit for each arm's run
    run         CHAIN                               the SUPERVISOR: walk the items from the saved state (nohup setsid)
    status      CHAIN                               supervisor ALIVE / STALE / DEAD, then each item's phase and hold
    clear-hold  CHAIN --item KEY --by WHO --why WHY [--resolution retry|relaunch|accept]
                                                    the orchestrator's decision on a held item (recorded)
    abandon     CHAIN --item KEY --by WHO --why WHY the orchestrator's decision that a held item will not finish
    ship-records CHAIN --item KEY                   ship one arm's records (the async worker; also the by-hand retry)
    live        [--models-dir D]                    live runs from launcher argv (never a glob)
    watch-files --status-file F | --chain CHAIN     the alert watch's file list: the status file + each live launcher's
                                                    stdout log + its run's launcher_child.full.log (--chain: + every
                                                    unfinished arm's fixed log paths, existing or not; none
                                                    inside a run dir from ENDED on). Never under tail -F (CB-47)
    monitor-filter --chain NAME                     the ONE ERE the alert Monitor greps over the watch-files
    monitor     --chain CHAIN [--max-hours H]       the alert follower: re-reads the file set every pass, prints each
                                                    filtered line, pages a dead / stale supervisor (no re-arm at a GO);
                                                    holds no file open, so it never blocks a thin
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

from main.ops.arm_chain import canary, cpugate, live, monitor, spec as spec_mod
from main.ops.arm_chain.chain import RESOLUTIONS, Chain, ChainError, default_shipper, own_tag, ship_worker
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


class SupervisorSignalled(BaseException):
    """SIGTERM / SIGHUP to the supervisor: unwound like an exception, so its last words are a HOLD line."""


def _on_signal(signum: int, _frame: object) -> None:
    raise SupervisorSignalled(signal.Signals(signum).name)


def _run(a: argparse.Namespace) -> int:
    """The supervisor. Whatever kills it from inside (an exception outside the per-item HOLD, a full disk under
    ``_hold`` itself, a SIGTERM) leaves ONE single-line ``HOLD: supervisor died: …`` status line (review 10-26,
    findings 1 and 2); a SIGKILL / OOM leaves none, which ``status`` and ``monitor`` read from the lock and the
    heartbeat instead."""
    s = spec_mod.load(a.chain)
    log = StatusLog(s.status_file, own_tag(s))
    try:
        ch = Chain(s)
    except ChainError as e:
        log.append(f"HOLD: the supervisor REFUSED to start: {e}")
        raise
    # a signal the launch already ignores (``nohup`` ignores SIGHUP) stays ignored
    old = {sig: signal.signal(sig, _on_signal) for sig in (signal.SIGTERM, signal.SIGHUP)
           if signal.getsignal(sig) is not signal.SIG_IGN}
    try:
        rc = ch.run()
    except BaseException as e:
        try:
            log.append(f"HOLD: supervisor died: {type(e).__name__}: {str(e)[:300]}")
        except Exception:   # noqa: BLE001 -- the page is best-effort; the traceback still reaches <name>.out
            pass
        raise
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)
    if rc == 4:
        print(f"another supervisor holds {ch.spec.state_path}.lock — not started", file=sys.stderr)
    print("\n".join(ch.status_lines()))
    return rc


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
        if name == "clear-hold":
            c.add_argument("--resolution", choices=RESOLUTIONS, default="retry",
                           help="retry the held step (default); relaunch an arm held at LAUNCHED whose kit never "
                                "launched; accept an arm's post-spawn gate on the orchestrator's evidence")
    sr = sub.add_parser("ship-records")
    sr.add_argument("chain")
    sr.add_argument("--item", required=True)
    mf = sub.add_parser("monitor-filter")
    mf.add_argument("--chain", required=True, help="the chain file's name (the ARMCHAIN[<name>] tag)")
    lv = sub.add_parser("live")
    lv.add_argument("--models-dir", default=None)
    wf = sub.add_parser("watch-files")
    wfg = wf.add_mutually_exclusive_group(required=True)
    wfg.add_argument("--status-file")
    wfg.add_argument("--chain", dest="chain_file", help="the chain file: adds every unfinished arm's fixed log paths")
    wf.add_argument("--models-dir", default=None)
    mo = sub.add_parser("monitor")
    mo.add_argument("--chain", required=True, dest="chain_file", help="the chain FILE")
    mo.add_argument("--interval-s", type=float, default=10.0)
    mo.add_argument("--max-hours", type=float, default=24.0)
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
            return _run(a)
        if a.cmd == "status":
            print("\n".join(_chain(a).status_lines()))
            return 0
        if a.cmd == "clear-hold":
            print(_chain(a).clear_hold(a.item, a.by, a.why, resolution=a.resolution))
            return 0
        if a.cmd == "abandon":
            print(_chain(a).abandon(a.item, a.by, a.why))
            return 0
        if a.cmd == "ship-records":
            s = spec_mod.load(a.chain)   # the chain FILE only: never the state (finding 11)
            return ship_worker(s, a.item, default_shipper(s, SubprocessRunner(), dict(os.environ)))
        if a.cmd == "monitor":
            return monitor.run_monitor(a.chain_file, interval_s=a.interval_s, max_hours=a.max_hours)
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
        cs = spec_mod.load(a.chain_file) if a.chain_file else None
        status_file = cs.status_file if cs else a.status_file
        log = StatusLog(status_file, "AWFILES")
        alert = canary.Alerter(status_file + ".awfiles_alerted", lambda m: log.append(f"HOLD: {m}"))
        runs = live.discover(cs.models_dir if cs and not a.models_dir else md)
        if cs is None:
            files = [status_file, *canary.watch_files(runs, alert)]
        else:
            files = monitor.chain_files(cs, monitor.read_state(cs), runs, alert)
        print("\n".join(files))
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

