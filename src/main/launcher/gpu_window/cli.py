"""``python -m main.launcher.gpu_window`` — arm, serve, watch and end a GPU WINDOW (T46).

USAGE
    # 1. ARM (the lease holder, while the run is live; checks every precondition FIRST, then writes request.json)
    python -m main.launcher.gpu_window arm models/<run> --precheck "<cpu dry-check of the guest>" \\
        [--at-epoch E | --in-minutes M] [--guest-start-timeout-s 180] [--max-window-min 60] [--note "…"]
    # 2. SERVE (the guest side, started any time after the arm, BOUNDED): waits for GO, claims, runs, writes DONE
    GEN3AI_GPU_LEASE_TOKEN_FILE=<holder's token file> python -m main.launcher.gpu_window guest models/<run> \\
        --wait-max-min 120 -- scripts/ops/gpu_lock.sh <the GPU job …>
    # watch / end
    python -m main.launcher.gpu_window status models/<run>
    python -m main.launcher.gpu_window resume models/<run> --reason "…"     # ends a HOLD (the run re-launches once the GPU is empty)
    python -m main.launcher.gpu_window preempt models/<run> [--kill]        # stops the guest's job (its session / group, nothing else)
    python -m main.launcher.gpu_window cancel models/<run>                  # withdraws a request not yet triggered

The run's launcher must have been started with ``--gpu-window``; the mechanism and its rules are
``designs/launcher/gpu_window.md``.

EXIT CODES  0 ok (``guest``: the guest command's own rc otherwise) · 2 usage · 3 REFUSED (a precondition
failed; nothing was written) · 4 ``guest`` did not run (the window was MISSED, abandoned or held before
GO) · 5 ``guest`` waited its whole ``--wait-max-min`` for GO.
"""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from main.launcher.gpu_window import controller
from main.launcher.gpu_window import protocol as P
from utils import procgroup
from utils.procgroup import run_in_own_group

EXIT_REFUSED, EXIT_NOT_RUN, EXIT_NO_GO = 3, 4, 5
PRECHECK_TIMEOUT_S = 900.0
GUEST_POLL_S = 2.0
GUEST_WAIT_CEILING_MIN = 24 * 60.0


class Refused(RuntimeError):
    """A precondition failed; the command wrote nothing."""


def _run_dir(arg: str) -> Path:
    p = Path(arg)
    if not p.exists() and os.sep not in arg:
        from utils.paths import main_models_dir
        archive = main_models_dir()
        if archive is not None and (archive / arg).exists():
            p = archive / arg
    return p.resolve()


def _launcher_record(run_dir: Path) -> dict:
    rec = P.read_json(P.window_root(run_dir) / P.LAUNCHER_FILE)
    if rec is None or not P.identity_alive(rec.get("launcher")):
        raise Refused(f"no LIVE launcher honours GPU windows for {run_dir} (no {P.WINDOW_DIRNAME}/{P.LAUNCHER_FILE}, "
                      "or its launcher is gone): launch or re-launch the run with `--gpu-window`")
    return rec


def _open_window(run_dir: Path) -> Optional[str]:
    """A description of the window already in flight for this run, or None."""
    if (P.window_root(run_dir) / P.REQUEST_FILE).exists():
        req = P.read_json(P.window_root(run_dir) / P.REQUEST_FILE) or {}
        return f"request {req.get('window_id')} is armed and not yet triggered"
    cur = P.read_current(run_dir)
    if cur is not None and cur.get("phase") in P.OPEN_PHASES:
        return f"window {cur.get('window_id')} is in phase {cur.get('phase').upper()} ({cur.get('reason', '')})"
    return None


def _lease_line(no_lease_check: Optional[str], launcher_rec: Optional[dict] = None) -> str:
    """The lease the guest will run on. The caller must HOLD it (its token is in this environment):
    the guest's GPU commands run on the holder's token, so the lease never moves during a window."""
    from utils.gpu_lease import holds_lease, read_lease
    from utils.gpu_lock import lock_path
    if no_lease_check is not None:
        # The opt-out is for a box with NO lease (a CPU-only run, a test). While anyone holds the lease,
        # it would let a non-holder pause the holder's run (review 2026-10-20): refuse it.
        held = read_lease(lock_path())
        if held is not None:
            raise Refused(f"--no-lease-check is refused while a GPU lease is held ({held.describe()}): "
                          "put the holder's GEN3AI_GPU_LEASE_TOKEN(_FILE) in front of the call instead")
        # ...and a run LAUNCHED under a lease (launcher.json `leaseless` false or absent) keeps the lease rule for
        # life: once its holder releases, "no lease is held" must not open it to any caller (T46 residual, 2026-10-25).
        if (launcher_rec or {}).get("leaseless") is not True:
            raise Refused("--no-lease-check is refused: this run's launcher was not recorded as started with NO GPU "
                          "lease (launcher.json `leaseless`), so only a caller holding the lease may arm or "
                          "guest a window; acquire one (scripts/ops/gpu_lease.sh), or re-launch the run leaseless")
        return f"lease NOT checked: {no_lease_check}"
    mine = holds_lease(lock_path())
    if mine is None:
        other = read_lease(lock_path())
        who = other.describe() if other is not None else "no lease is held"
        raise Refused(f"this caller does not hold the GPU lease ({who}); put the holder's "
                      "GEN3AI_GPU_LEASE_TOKEN(_FILE) in front of the call, or pass --no-lease-check REASON")
    return f"lease owner {mine.owner!r} (holder pid {mine.pid})"


def _run_precheck(cmd: str, timeout_s: float) -> dict:
    """The guest's CPU dry-check, run NOW with the GPU hidden. A non-zero exit or a timeout refuses."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    t0 = time.monotonic()
    try:
        out = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True,
                             timeout=timeout_s, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise Refused(f"the precheck did not finish within {timeout_s:.0f} s: {cmd!r}") from None
    tail = (out.stdout + out.stderr).strip().splitlines()[-12:]
    rec = {"cmd": cmd, "rc": out.returncode, "seconds": round(time.monotonic() - t0, 1), "tail": tail}
    if out.returncode != 0:
        raise Refused(f"the precheck exited rc {out.returncode}: {cmd!r}\n    " + "\n    ".join(tail))
    return rec


def cmd_arm(a: argparse.Namespace) -> int:
    run_dir = _run_dir(a.run_dir)
    now = time.time()
    rec = _launcher_record(run_dir)
    busy = _open_window(run_dir)
    if busy:
        raise Refused(f"a window is already open for this run: {busy}")
    if a.at_epoch is not None:
        not_before = float(a.at_epoch)
    else:
        not_before = now + 60.0 * float(a.in_minutes or 0.0)
    req = P.Request(window_id=P.new_window_id(now), launcher=dict(rec["launcher"]), armed_epoch=now,
                    not_before_epoch=not_before, guest_start_timeout_s=float(a.guest_start_timeout_s),
                    max_window_s=60.0 * float(a.max_window_min), note=a.note or "",
                    horizon_h=float(a.horizon_hours))
    problems = req.problems()
    if problems:
        raise Refused("; ".join(problems))
    lease = _lease_line(a.no_lease_check, rec)
    if a.precheck is not None:
        pre = _run_precheck(a.precheck, float(a.precheck_timeout_s))
    else:
        pre = {"skipped": a.no_precheck}
    req = P.Request(**{**req.__dict__, "precheck": pre, "lease": lease})
    # The launcher may have exited while the precheck ran: re-check, then create the request EXCLUSIVELY.
    _launcher_record(run_dir)
    if not P.create_json_exclusive(P.window_root(run_dir) / P.REQUEST_FILE, req.to_json()):
        raise Refused("another request was armed while this one's precheck ran")
    print(f"ARMED window {req.window_id} for {run_dir}")
    print(f"  trigger : {P.describe_epoch(req.not_before_epoch)} (epoch {req.not_before_epoch:.0f})")
    print(f"  launcher: pid {req.launcher['pid']}  ·  {lease}")
    print(f"  bounds  : guest must claim ≤ {req.guest_start_timeout_s:.0f} s after GO (else MISSED → resume); "
          f"window ≤ {req.max_window_s / 60:.0f} min after GO (else HOLD)")
    print(f"  precheck: {'rc 0 in %.1f s' % pre['seconds'] if 'rc' in pre else 'SKIPPED: ' + str(pre['skipped'])}")
    print(f"  guest   : python -m main.launcher.gpu_window guest {run_dir} --wait-max-min <N> -- <cmd …>")
    return 0


def _guest_window_id(run_dir: Path) -> str:
    req = P.read_json(P.window_root(run_dir) / P.REQUEST_FILE)
    if req is not None and req.get("window_id"):
        return str(req["window_id"])
    cur = P.read_current(run_dir)
    if cur is not None and cur.get("phase") in (P.PAUSING, P.GO):
        return str(cur["window_id"])
    raise Refused(f"no window is armed or pausing for {run_dir} (status: "
                  f"{(cur or {}).get('phase', 'none')}) — `arm` first")


def cmd_guest(a: argparse.Namespace) -> int:
    run_dir = _run_dir(a.run_dir)
    cmd: List[str] = list(a.cmd)
    if not cmd:
        raise Refused("no guest command after `--`")
    if not (0 < float(a.wait_max_min) <= GUEST_WAIT_CEILING_MIN):
        raise Refused(f"--wait-max-min must be in (0, {GUEST_WAIT_CEILING_MIN:.0f}]: a guest's wait is always bounded")
    _lease_line(a.no_lease_check, P.read_json(P.window_root(run_dir) / P.LAUNCHER_FILE))
    wid = _guest_window_id(run_dir)
    wdir = P.window_dir(run_dir, wid)
    deadline = time.time() + 60.0 * float(a.wait_max_min)
    print(f"[gpu_window guest] waiting ≤ {a.wait_max_min:g} min for GO of window {wid}", flush=True)
    while not (wdir / P.GO_FILE).exists():
        cur = P.read_current(run_dir) or {}
        if cur.get("window_id") == wid and cur.get("phase") not in (P.PAUSING, P.GO, None):
            print(f"[gpu_window guest] window {wid} ended before GO ({cur.get('phase')}: {cur.get('reason')}) "
                  "— NOT running", flush=True)
            return EXIT_NOT_RUN
        if time.time() > deadline:
            print(f"[gpu_window guest] no GO for window {wid} within {a.wait_max_min:g} min — NOT running", flush=True)
            return EXIT_NO_GO
        time.sleep(GUEST_POLL_S)
    me = P.process_identity(os.getpid())
    claim = {"by": "guest", "window_id": wid, "guest": me, "cmd": shlex.join(cmd), "claimed_epoch": time.time(),
             "job": None}
    if not P.create_json_exclusive(wdir / P.CLAIM_FILE, claim):
        print(f"[gpu_window guest] window {wid} was claimed first ({P.read_json(wdir / P.CLAIM_FILE)}) — NOT running",
              flush=True)
        return EXIT_NOT_RUN
    started = time.time()
    print(f"[gpu_window guest] CLAIMED window {wid}; running: {shlex.join(cmd)}", flush=True)
    job: dict = {}

    def record_job(pgid: int) -> None:
        """The command is the leader of its OWN session: record that group in the claim (atomically REPLACED, the
        exclusive create above already decided who runs), so the launcher judges the job by the group and an
        operator's `preempt` signals exactly that group."""
        job.update({"pgid": pgid, "sid": pgid, "leader": P.process_identity(pgid), "started_epoch": time.time()})
        P.write_json_atomic(wdir / P.CLAIM_FILE, {**claim, "job": job})
        print(f"[gpu_window guest] the job is session / process group {pgid}: preempt = `python -m "
              f"main.launcher.gpu_window preempt {run_dir}` or `kill -TERM -- -{pgid}` (never this guest's pid)",
              flush=True)

    try:
        rc = run_in_own_group(cmd, session=True, on_start=record_job)
    except OSError as e:
        print(f"[gpu_window guest] could not start the guest command: {e}", flush=True)
        rc = 127
    P.write_json_atomic(wdir / P.DONE_FILE, {"window_id": wid, "rc": rc, "started_epoch": started,
                                             "ended_epoch": time.time(), "cmd": shlex.join(cmd),
                                             "pgid": job.get("pgid")})
    print(f"[gpu_window guest] DONE window {wid}: rc {rc}", flush=True)
    return rc


def cmd_status(a: argparse.Namespace) -> int:
    run_dir = _run_dir(a.run_dir)
    root = P.window_root(run_dir)
    rec = P.read_json(root / P.LAUNCHER_FILE)
    if rec is None:
        print(f"{run_dir}: GPU windows NOT enabled (no launcher with --gpu-window has run here)")
        return 0
    alive = P.identity_alive(rec.get("launcher"))
    print(f"{run_dir}")
    print(f"  launcher : pid {rec['launcher'].get('pid')} — {'LIVE' if alive else 'GONE'}")
    req = P.read_json(root / P.REQUEST_FILE)
    if req is not None:
        print(f"  armed    : {req.get('window_id')} — triggers at {P.describe_epoch(float(req.get('not_before_epoch', 0)))}")
    cur = P.read_current(run_dir)
    if cur is not None:
        print(f"  window   : {cur.get('window_id')} — {str(cur.get('phase')).upper()} since "
              f"{P.describe_epoch(float(cur.get('since_epoch', 0)))}: {cur.get('reason')}")
        if cur.get("phase") in (P.CLAIMED, P.HOLD):
            claim = P.read_json(P.window_dir(run_dir, str(cur.get("window_id"))) / P.CLAIM_FILE) or {}
            if claim.get("by") == "guest":
                print(f"  guest job: {P.job_note(claim.get('job'))}")
    elif req is None:
        print("  window   : none yet")
    return 0


def cmd_resume(a: argparse.Namespace) -> int:
    run_dir = _run_dir(a.run_dir)
    cur = P.read_current(run_dir)
    if cur is None or cur.get("phase") != P.HOLD:
        raise Refused(f"no window is in HOLD for {run_dir} (phase: {(cur or {}).get('phase', 'none')})")
    wid = str(cur["window_id"])
    P.write_json_atomic(P.window_dir(run_dir, wid) / P.RESUME_FILE, {
        "window_id": wid, "reason": a.reason, "epoch": time.time(), "by": P.process_identity(os.getpid())})
    print(f"RESUME written for window {wid} (HOLD: {cur.get('reason')}). The launcher re-checks the GPU first: it "
          "re-launches the run once `nvidia-smi` lists no compute process (waiting up to "
          f"{controller.GPU_RELEASE_TIMEOUT_S:.0f} s, naming the pids it waits on), else the HOLD stays with the "
          "reason and a new `resume` is needed. `status` shows which.")
    return 0


def cmd_preempt(a: argparse.Namespace) -> int:
    """Stop the guest's JOB: SIGTERM (``--kill``: SIGKILL) to the process group(s) of the SESSION its claim recorded.
    Never the guest's own pid or group, never a pid: the groups are re-derived from the live session (a command that
    nests a group of its own, e.g. a ``gpu_lock.sh`` inside it, is stopped whole, and a SIGKILL needs no forwarding)."""
    run_dir = _run_dir(a.run_dir)
    cur = P.read_current(run_dir)
    if cur is None or cur.get("phase") not in (P.CLAIMED, P.HOLD):
        raise Refused(f"no window with a running or held guest for {run_dir} (phase: {(cur or {}).get('phase', 'none')})")
    wid = str(cur["window_id"])
    claim = P.read_json(P.window_dir(run_dir, wid) / P.CLAIM_FILE) or {}
    job = claim.get("job")
    if claim.get("by") != "guest" or claim.get("window_id") != wid or not job:
        raise Refused(f"window {wid} has no guest job group on record (claim: {claim or 'none'}): nothing to preempt")
    members = P.job_members(job)
    if not members:
        print(f"window {wid}: the guest's job (pgid {job.get('pgid')}) has no live process; nothing to signal")
        return 0
    groups = sorted({p.pgrp for p in members}, key=lambda g: (g != job.get("pgid"), g))
    guest_pid = (claim.get("guest") or {}).get("pid")
    unsafe = [g for g in groups if g <= 1 or g == os.getpgrp() or g == guest_pid]
    if job.get("sid") in (os.getsid(0), 0, 1) or (isinstance(guest_pid, int) and job.get("sid") == guest_pid):
        unsafe.append(job.get("sid"))
    if unsafe:
        raise Refused(f"refusing to signal {unsafe}: this caller's own group / session, or the guest's: the claim's job "
                      f"record {job} is not trustworthy")
    sig = signal.SIGKILL if a.kill else signal.SIGTERM
    for g in groups:
        procgroup.killpg_quiet(g, sig)
    print(f"PREEMPT window {wid}: {signal.Signals(sig).name} to process group(s) {groups} "
          f"({len(members)} live process(es): {', '.join(str(p.pid) for p in members[:8])}) - the guest's job only; "
          "the guest then writes DONE with a non-zero rc and the window HOLDS until `resume`")
    return 0


def cmd_cancel(a: argparse.Namespace) -> int:
    run_dir = _run_dir(a.run_dir)
    path = P.window_root(run_dir) / P.REQUEST_FILE
    req = P.read_json(path)
    try:
        os.remove(path)                    # atomic against the launcher's consuming rename: one of us wins
    except FileNotFoundError:
        cur = P.read_current(run_dir) or {}
        raise Refused(f"nothing to cancel: no armed request (window {cur.get('window_id')} is "
                      f"{cur.get('phase', 'none')}; a triggered window ends by GO/DONE, MISSED or a HOLD's `resume`)") from None
    print(f"CANCELLED armed window {(req or {}).get('window_id')} (never triggered)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m main.launcher.gpu_window", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    sub = ap.add_subparsers(dest="verb", required=True)

    arm = sub.add_parser("arm", help="check every precondition, then arm a window", allow_abbrev=False)
    arm.add_argument("run_dir")
    pre = arm.add_mutually_exclusive_group(required=True)
    pre.add_argument("--precheck", metavar="CMD",
                     help="the guest's CPU dry-check (bash -c, CUDA hidden); a non-zero exit REFUSES the arm")
    pre.add_argument("--no-precheck", metavar="REASON", help="arm without a precheck; the reason is recorded")
    when = arm.add_mutually_exclusive_group()
    when.add_argument("--at-epoch", type=float, help="trigger at this ABSOLUTE epoch (date +%%s)")
    when.add_argument("--in-minutes", type=float, help="trigger this many minutes from now (default: now)")
    arm.add_argument("--horizon-hours", type=float, default=P.DEFAULT_HORIZON_H,
                     help="refuse a trigger further ahead than this (default %(default)s)")
    arm.add_argument("--guest-start-timeout-s", type=float, default=P.DEFAULT_GUEST_START_TIMEOUT_S,
                     help="GO → the guest's claim; past it the window is MISSED and the run resumes (default %(default)s)")
    arm.add_argument("--max-window-min", type=float, default=P.DEFAULT_MAX_WINDOW_MIN,
                     help="GO → DONE; past it the window HOLDS (default %(default)s)")
    arm.add_argument("--precheck-timeout-s", type=float, default=PRECHECK_TIMEOUT_S)
    arm.add_argument("--no-lease-check", metavar="REASON",
                     help="do not require this caller to hold the GPU lease (a CPU-only run / a test)")
    arm.add_argument("--note", default="")
    arm.set_defaults(fn=cmd_arm)

    g = sub.add_parser("guest", help="wait (bounded) for GO, claim, run the guest command, write DONE",
                       allow_abbrev=False)
    g.add_argument("run_dir")
    g.add_argument("--wait-max-min", type=float, required=True, help="the longest this waits for GO")
    g.add_argument("--no-lease-check", metavar="REASON")
    # The guest command follows `--` and is split off in `main()` (an argparse REMAINDER positional after
    # `run_dir` would swallow the options too).
    g.set_defaults(fn=cmd_guest)

    st = sub.add_parser("status", help="the window state of a run", allow_abbrev=False)
    st.add_argument("run_dir")
    st.set_defaults(fn=cmd_status)

    rs = sub.add_parser("resume", help="end a HOLD: the launcher re-launches the run", allow_abbrev=False)
    rs.add_argument("run_dir")
    rs.add_argument("--reason", required=True)
    rs.set_defaults(fn=cmd_resume)

    pr = sub.add_parser("preempt", help="stop the guest's job: SIGTERM (--kill: SIGKILL) to its session's group(s) only",
                        allow_abbrev=False)
    pr.add_argument("run_dir")
    pr.add_argument("--kill", action="store_true", help="SIGKILL instead of SIGTERM")
    pr.set_defaults(fn=cmd_preempt)

    c = sub.add_parser("cancel", help="withdraw an armed request that has not triggered", allow_abbrev=False)
    c.add_argument("run_dir")
    c.set_defaults(fn=cmd_cancel)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd: List[str] = []
    if "--" in args:
        i = args.index("--")
        args, cmd = args[:i], args[i + 1:]
    a = build_parser().parse_args(args)
    a.cmd = cmd
    try:
        return int(a.fn(a))
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
