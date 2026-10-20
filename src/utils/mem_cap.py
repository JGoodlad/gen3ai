"""Run ONE command under a hard memory cap, in its OWN transient systemd scope, inside ONE shared
heavy-job slice — so a runaway job is OOM-killed ALONE and never takes the Claude session with it.

USAGE
    scripts/ops/mem_cap.sh <GB> <cmd> [args...]                 # the shell form
    scripts/ops/mem_cap.sh --name k8trace 24 timeout 2h python -m main.compile_inventory run ...
    scripts/ops/mem_cap.sh --status                              # the slice and every capped job now
    python -m utils.mem_cap [--name N] [--slice-gb G] [--term-grace S] <GB> -- <cmd> [args...]

    # composes with the GPU lock in BOTH orders; a timeout INSIDE bounds the command alone (the
    # heavy-slot wait is not counted), one OUTSIDE bounds the whole wrapper and ALSO stops the scope
    # (STOPPING A JOB, below):
    scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 24 timeout 1h python ...
    scripts/ops/mem_cap.sh 24 scripts/ops/gpu_lock.sh timeout 1h python ...
    timeout 3000 scripts/ops/mem_cap.sh 24 python ...             # outside: SIGTERM, then the grace, then SIGKILL
    scripts/ops/mem_cap.sh --term-grace 300 24 python ...         # a longer grace for a job that saves on SIGTERM

WHY. On 2026-09-30 three global OOM kills (15:33, 16:44, 16:54; ``journalctl -k``) each killed one
python3 that had grown to 74-82 GB anonymous RSS on this 89 GB box. Every agent process and the Claude
Code session itself lived in ONE ``tmux-spawn-*.scope`` whose ``OOMPolicy`` was ``stop`` (the user
manager's ``DefaultOOMPolicy``), so after the kernel killed the offender systemd tore down the whole
scope — claude, every agent, every detached ``setsid`` job — three times. The owner has since set
``DefaultOOMPolicy=continue`` (``~/.config/systemd/user.conf.d/oom.conf``), so an OOM kill no longer
stops the session's scope; this module is the other half: a heavy job never gets to exhaust the box.

THE THREE LAYERS, all put in force and VERIFIED before the command starts:

1. **A per-job cap.** The command runs in a transient scope ``gen3ai-capped-<name>-<pid>.scope`` with
   ``MemoryMax=<GB>G``, ``MemorySwapMax=0`` (swap would turn an overrun into an hour of thrash) and
   ``OOMPolicy=stop`` (an OOM kill inside the job stops the WHOLE job, not one worker of it).
2. **An aggregate cap.** Every capped job lives in ONE slice, ``gen3ai-heavy.slice``
   (``$GEN3AI_HEAVY_SLICE``), whose ``MemoryMax`` is DECLARED at every launch — default
   ``DEFAULT_SLICE_GB`` (64 GB of the box's 89 GB, leaving ~25 GB for Claude, the desktop and the page
   cache; ``--slice-gb`` / ``$GEN3AI_HEAVY_SLICE_GB``). Set with ``systemctl --user set-property
   --runtime`` — a RUNTIME drop-in under ``/run/user/<uid>/systemd/user.control/``, gone at reboot and
   re-declared by the next launch. Two jobs that fit their own caps but not the slice's: the kernel
   kills the LARGER first. ⚠️ That is NOT a guarantee that a sibling survives: a sibling that
   ALLOCATES in the few milliseconds while the victim's memory is still being released finds the
   slice still at its limit and no eligible victim left but itself, and the kernel OOM-kills it too
   (kernel 7.0 log, 2026-10-01: 2 of 85 slice-level OOMs on this box; both times the SECOND OOM was
   invoked by the sibling's own allocation, 1-5 ms after the first kill). Its report is truthful
   (its own scope's ``Result=oom-kill``, exit 86). A per-job cap has no such cross-job effect —
   treat the slice cap as the box's backstop, not as a scheduler.
3. **A victim preference.** Inside the scope the job's ``oom_score_adj`` is raised to ``+500``
   (``OOM_SCORE_ADJ``; raising needs no privilege), so if the KERNEL ever has to choose box-wide it
   picks a heavy job over the Claude processes (which sit at 0, and could only be lowered with
   ``CAP_SYS_RESOURCE``).

The command runs as a child of a tiny stdlib SUPERVISOR inside the scope, which checks
``memory.max`` / ``memory.swap.max`` / the slice's ``memory.max`` in its own cgroup BEFORE starting the
command — a cap that is not in force is a REFUSAL (``EXIT_NOCAP``), never an uncapped run — and reads
``memory.peak`` after it. A scope that ended cleanly is garbage-collected at once, so this is the only
place the peak of a well-behaved job can be read; an OOM-killed scope stays loaded as ``failed`` with
``Result=oom-kill`` and ``MemoryPeak``, which this wrapper reads and then ``reset-failed``s.

HEAVY-JOB SLOTS (2026-10-19, ``utils.heavy_slots``): while a training run is live (or with ``--slot``)
the job first takes one of the box's 2 heavy slots — rule 5b's "<= 2 heavy jobs" — waiting with a
visible line (``--no-wait`` refuses at once, exit 87). A job the orchestrator marks BLOCKING
(``$GEN3AI_BLOCKING=<reason>`` or ``--blocking <reason>``) bypasses them; a job inside a holder (the
routine gate holds one) takes none. ``--status`` lists holders, the queue and the bypassers.

The slot is tied to the JOB, not the wrapper (D-2, 2026-10-20): the locked fd is handed to the supervisor and the
job (``pass_fds``), so the slot stays held while any process of the job's tree lives - including the grace a
SIGKILLed wrapper leaves a job that defers SIGTERM (``--status`` marks a slot whose locker is gone "owner gone");
a normal end still releases it at once.

STOPPING A JOB (2026-10-19, ``gen3_mem_cap_stop_v1``; backlog row "a ``timeout`` OUTSIDE mem_cap.sh does not
stop a trainer", U4 finding (a) 2026-10-02: ``timeout 590 mem_cap.sh … <trainer>`` ran 12 min past its bound
and needed a SIGKILL by PID). SIGINT / SIGTERM / SIGHUP — from ``timeout``, a terminal, a ``kill`` of the
wrapper — are forwarded wrapper -> supervisor -> command (the command may save and exit: the trainer's
deferred abort does). Then ``--term-grace`` seconds (default ``DEFAULT_TERM_GRACE_S``; ``$GEN3AI_MEM_CAP_
TERM_GRACE_S``) after the FIRST stop signal the supervisor SIGKILLs **every process left in the scope** — a
worker that ``setsid``-ed out of the process group included, which ``timeout``'s own group kill never
reaches — and a command that exits on the signal gets its leftover descendants swept the same way (SIGTERM,
then SIGKILL at the deadline), so a stopped job leaves an EMPTY scope. The wrapper's own death by SIGKILL
(``timeout -k``, ``kill -9`` of its PID) is a stop signal too: the supervisor asks the kernel for
``PR_SET_PDEATHSIG`` SIGTERM, so the job never outlives its wrapper (and its heavy slot) uncapped and
unslotted. The exit status is the command's (143 for SIGTERM, 137 once the grace killed it). What it does
NOT cover: a command that exits NORMALLY leaving a detached background process behind (nothing was asked to
stop; that process stays in the scope, listed by ``--status``).

EXIT STATUS: the command's own (128+N if signal N killed it); ``EXIT_OOM`` (86) when the job was
OOM-killed at its cap or the slice's; ``EXIT_NOCAP`` (85) when the cap could not be put in force (no
user manager, no memory controller, systemd-run refused); 2 = usage. The ``[mem_cap]`` lines on
stderr are the record — a command may itself exit 85/86.

VERIFIED on this box (systemd 259, cgroup v2, ``memory`` delegated to ``user@1000.service``): a
``systemd-run --user --scope`` from INSIDE the tmux scope creates a sibling scope under the user
manager with the cap in force; it does NOT inherit the caller's cgroup. Linux + systemd only.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from utils import heavy_slots as HS

SLICE_ENV = "GEN3AI_HEAVY_SLICE"
SLICE_GB_ENV = "GEN3AI_HEAVY_SLICE_GB"
DEFAULT_SLICE = "gen3ai-heavy.slice"
DEFAULT_SLICE_GB = 64.0
OOM_SCORE_ADJ = 500
TERM_GRACE_ENV = "GEN3AI_MEM_CAP_TERM_GRACE_S"
#: Seconds from the FIRST stop signal to SIGKILL of everything left in the scope. A graceful stop
#: (the trainer's checkpoint-and-exit, a pytest teardown, a cargo shutdown) takes seconds; a job that has
#: not ended in a minute is not going to, and "a timeout that does not stop the job" was the hazard.
DEFAULT_TERM_GRACE_S = 60.0
UNIT_PREFIX = "gen3ai-capped-"
EXIT_NOCAP = 85
EXIT_OOM = 86
GiB = 1 << 30
PAGE = 4096

# Runs INSIDE the scope, as ``python -I -S -c SUPERVISOR <report> <job max> <slice max> <adj> <grace> <wrapper pid> -- cmd...``.
# Stdlib only and import-light: it must not depend on PYTHONPATH (and must not perturb the command's).
# STOPPING (module docs): the handlers are installed BEFORE the command starts (a stop signal in the window
# between its start and its handlers used to kill the supervisor and orphan the command); the first stop
# signal is forwarded to the command and arms SIGALRM for ``grace`` seconds, which SIGKILLs EVERY process in
# the scope; a command that ends on the signal has its leftover descendants swept; and the supervisor asks the
# kernel for SIGTERM when its own parent (the wrapper) dies.
SUPERVISOR = r"""
import os, signal, subprocess, sys, time
report, want, want_slice, adj, grace, wrapper, slot_fd = sys.argv[1:8]
cmd = sys.argv[9:]
grace = float(grace)
def rd(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return ""
def out(s):
    with open(report, "a") as f:
        f.write(s + "\n")
def say(s):
    os.write(2, ("[mem_cap] " + s + "\n").encode())
rel = ""
for line in rd("/proc/self/cgroup").splitlines():
    if line.startswith("0::"):
        rel = line[3:]
cg = "/sys/fs/cgroup" + rel
got, swap, got_slice = rd(cg + "/memory.max"), rd(cg + "/memory.swap.max"), rd(os.path.dirname(cg) + "/memory.max")
if (got, swap, got_slice) != (want, "0", want_slice):
    out("nocap cgroup=%s memory.max=%r (want %s) memory.swap.max=%r (want 0) slice memory.max=%r (want %s)"
        % (rel, got, want, swap, got_slice, want_slice))
    sys.exit(%(nocap)d)
with open("/proc/self/oom_score_adj", "w") as f:
    f.write(adj)
def scope_pids():
    # every LIVE process of the scope but this one (a zombie is dead; its parent or the subreaper reaps it)
    me, live = os.getpid(), []
    for tok in rd(cg + "/cgroup.procs").split():
        pid = int(tok)
        st = rd("/proc/%d/stat" % pid)
        if pid != me and st and st[st.rfind(")") + 2:][:1] != "Z":
            live.append(pid)
    return live
def sweep(sig):
    n = 0
    for pid in scope_pids():
        try:
            os.kill(pid, sig)
            n += 1
        except OSError:
            pass
    return n
child, stop = None, []        # stop = [first stop signal's time, its number]
def on_stop(n, _f):
    if not stop:
        stop.extend([time.monotonic(), n])
        signal.setitimer(signal.ITIMER_REAL, grace)
        say("signal %d received: forwarded to the job; every process left in the scope is SIGKILLed in %g s" % (n, grace))
    if child is not None:
        child.send_signal(n)
def on_alarm(_n, _f):
    say("the %g s grace after the stop signal is over: SIGKILL to the %d process(es) still in the scope" % (grace, sweep(signal.SIGKILL)))
for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(s, on_stop)
signal.signal(signal.SIGALRM, on_alarm)
try:    # the wrapper's death (kill -9, timeout -k) is a stop signal too: the job must not outlive it
    import ctypes
    ctypes.CDLL(None).prctl(1, int(signal.SIGTERM), 0, 0, 0)        # PR_SET_PDEATHSIG
    if os.getppid() != int(wrapper):                                # already orphaned: the kernel will not say so
        on_stop(signal.SIGTERM, None)
except Exception:
    pass
out("started cgroup=%s" % rel)
try:
    # the heavy slot's locked fd rides into the job (D-2): the slot lives as long as the job's process tree does,
    # so the grace a SIGKILLed wrapper leaves a SIGTERM-deferring job is not a window with a free slot
    child = subprocess.Popen(cmd, pass_fds=(int(slot_fd),) if int(slot_fd) >= 0 else ())
except OSError as e:
    out("done 127 %s 0" % (rd(cg + "/memory.peak") or "0"))
    print("[mem_cap] cannot run %r: %s" % (cmd[0], e), file=sys.stderr)
    sys.exit(127)
if stop:
    child.send_signal(stop[1])
rc = child.wait()
if stop:    # asked to stop: the scope must end up EMPTY (the command's descendants, a setsid-ed worker too)
    left = scope_pids()
    if left:
        say("the job ended on the stop signal with %d process(es) left in the scope: SIGTERM to them" % sweep(signal.SIGTERM))
        while scope_pids() and time.monotonic() < stop[0] + grace:
            time.sleep(0.05)
    for _ in range(100):
        if not sweep(signal.SIGKILL):
            break
        time.sleep(0.02)
    signal.setitimer(signal.ITIMER_REAL, 0)
kills = 0
for line in rd(cg + "/memory.events").splitlines():
    k, _, v = line.partition(" ")
    if k == "oom_kill":
        kills = int(v)
out("done %d %s %d" % (rc, rd(cg + "/memory.peak") or "0", kills))
sys.exit(128 - rc if rc < 0 else rc)
""".replace("%(nocap)d", str(EXIT_NOCAP))


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def gb_to_bytes(gb: float) -> int:
    """GB (GiB) -> bytes, page-aligned DOWN exactly as the kernel stores ``memory.max``."""
    if not gb > 0:
        raise ValueError(f"a memory cap must be > 0 GB, got {gb!r}")
    return max(PAGE, int(gb * GiB) // PAGE * PAGE)


def fmt_gb(n: Optional[int]) -> str:
    return "?" if n is None else f"{n / GiB:.2f} GB"


def slice_name() -> str:
    return os.environ.get(SLICE_ENV) or DEFAULT_SLICE


def slice_gb() -> float:
    raw = os.environ.get(SLICE_GB_ENV)
    return float(raw) if raw else DEFAULT_SLICE_GB


def term_grace_s() -> float:
    """The grace between the first stop signal and the SIGKILL of the scope (``$GEN3AI_MEM_CAP_TERM_GRACE_S``
    or `DEFAULT_TERM_GRACE_S`). A malformed value raises ValueError — never a silently different bound."""
    raw = os.environ.get(TERM_GRACE_ENV)
    return parse_grace(raw) if raw else DEFAULT_TERM_GRACE_S


def parse_grace(raw: object) -> float:
    try:
        g = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"a stop grace must be a number of seconds > 0, got {raw!r}") from None
    if not (0.0 < g < float("inf")):        # NaN fails the comparison too
        raise ValueError(f"a stop grace must be a number of seconds > 0 and finite, got {raw!r}")
    return g


def default_name(cmd: Sequence[str]) -> str:
    """A unit-name-safe label for ``cmd``: the module of a ``python -m X``, else argv[0]'s basename,
    looking through the wrappers that usually lead (``timeout``, ``env``, ``nice``, the GPU lock)."""
    args = list(cmd)
    while args:
        base = os.path.basename(args[0])
        if base in ("timeout", "env", "nice", "ionice", "nohup", "setsid", "gpu_lock.sh", "mem_cap.sh"):
            args = args[1:]
            while args and (args[0].startswith("-") or "=" in args[0] or re.fullmatch(r"[\d.]+[smhd]?", args[0])):
                args = args[1:]
            continue
        if base.startswith("python") and "-m" in args[1:]:
            i = args.index("-m")
            if i + 1 < len(args):
                return sanitize(args[i + 1])
        return sanitize(base)
    return "job"


def sanitize(s: str) -> str:
    return (re.sub(r"[^A-Za-z0-9_.]+", "_", s).strip("_.") or "job")[:48]


def _env() -> dict:
    """The environment the systemd clients need — ``XDG_RUNTIME_DIR`` locates the user manager, and a
    cron or a ``setsid`` job may not carry it."""
    env = dict(os.environ)
    if not env.get("XDG_RUNTIME_DIR"):
        cand = f"/run/user/{os.getuid()}"
        if os.path.isdir(cand):
            env["XDG_RUNTIME_DIR"] = cand
    return env


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], env=_env(), capture_output=True, text=True)


def unit_props(unit: str, *props: str) -> dict:
    r = _systemctl("show", unit, *(f"-p{p}" for p in props))
    out = {}
    for line in r.stdout.splitlines():
        k, _, v = line.partition("=")
        out[k] = v
    return out


def declare_slice(name: str, cap_bytes: int) -> Optional[str]:
    """Put the aggregate cap on the slice (runtime only). Returns an error string, or ``None``."""
    r = _systemctl("set-property", "--runtime", name, f"MemoryMax={cap_bytes}", "MemorySwapMax=0")
    return None if r.returncode == 0 else (r.stderr.strip() or f"systemctl exited {r.returncode}")


@dataclass
class CapResult:
    rc: int                   # this wrapper's exit status (the command's own, or EXIT_OOM / EXIT_NOCAP)
    unit: str
    oom: bool
    peak_bytes: Optional[int]
    cmd_rc: Optional[int]     # the command's own exit status, when it ran to an exit


def capped_argv(cmd: Sequence[str], cap_bytes: int, *, unit: str, slice_: str, slice_bytes: int,
                report: str, adj: int = OOM_SCORE_ADJ, grace: float = DEFAULT_TERM_GRACE_S,
                slot_fd: Optional[int] = None) -> List[str]:
    """The ``systemd-run`` argv that runs ``cmd`` under the supervisor in a capped scope."""
    return ["systemd-run", "--user", "--scope", "--quiet", f"--slice={slice_}",
            "-p", f"MemoryMax={cap_bytes}", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=stop",
            "--unit", unit, "--",
            sys.executable, "-I", "-S", "-c", SUPERVISOR, report, str(cap_bytes), str(slice_bytes), str(adj),
            repr(grace), str(os.getpid()), str(-1 if slot_fd is None else slot_fd), "--", *cmd]


def run_capped(cmd: Sequence[str], gb: float, *, name: Optional[str] = None,
               slice_: Optional[str] = None, slice_gb_: Optional[float] = None,
               term_grace: Optional[float] = None, slot_fd: Optional[int] = None,
               log: Callable[[str], None] = _stderr) -> CapResult:
    """Run ``cmd`` capped at ``gb`` GB in its own scope inside the heavy slice; report and return.
    ``slot_fd``: the heavy slot's locked fd (``heavy_slots.slot_fd``), handed to the supervisor and the job so
    the slot lives as long as any process of the job's tree (a SIGKILLed wrapper stops the job after the grace).
    A stop signal (SIGINT / SIGTERM / SIGHUP) reaching this process is forwarded to the supervisor, which
    SIGKILLs the whole scope ``term_grace`` seconds after the first one (module docs, STOPPING A JOB)."""
    cap = gb_to_bytes(gb)
    grace = parse_grace(term_grace) if term_grace is not None else term_grace_s()
    sl = slice_ or slice_name()
    sl_bytes = gb_to_bytes(slice_gb() if slice_gb_ is None else slice_gb_)
    unit = f"{UNIT_PREFIX}{sanitize(name) if name else default_name(cmd)}-{os.getpid()}.scope"
    if shutil.which("systemd-run") is None or shutil.which("systemctl") is None:
        log("[mem_cap] REFUSING: systemd-run/systemctl not found — the cap cannot be put in force")
        return CapResult(EXIT_NOCAP, unit, False, None, None)
    err = declare_slice(sl, sl_bytes)
    if err is not None:
        log(f"[mem_cap] REFUSING: could not declare {sl} MemoryMax={fmt_gb(sl_bytes)}: {err}")
        return CapResult(EXIT_NOCAP, unit, False, None, None)
    if cap > sl_bytes:
        log(f"[mem_cap] WARNING: job cap {fmt_gb(cap)} exceeds the slice cap {fmt_gb(sl_bytes)} — the slice binds")
    log(f"[mem_cap] cap {fmt_gb(cap)} (swap 0) · unit {unit} · slice {sl} (aggregate cap {fmt_gb(sl_bytes)}) "
        f"· oom_score_adj +{OOM_SCORE_ADJ}")
    fd, report = tempfile.mkstemp(prefix="gen3ai_mem_cap_", suffix=".txt", dir=_env().get("XDG_RUNTIME_DIR"))
    os.close(fd)
    holder: List[subprocess.Popen] = []
    pending: List[int] = []                 # a stop signal that arrives before the supervisor exists

    def _forward(n: int, _frame: object) -> None:
        if holder:
            holder[0].send_signal(n)
        else:
            pending.append(n)

    # the handlers are in place BEFORE the supervisor starts: a SIGTERM in that window used to kill this
    # wrapper outright, leaving a scope nobody reports on or stops
    fwd = {s: signal.signal(s, _forward) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        child = subprocess.Popen(capped_argv(cmd, cap, unit=unit, slice_=sl, slice_bytes=sl_bytes, report=report,
                                             grace=grace, slot_fd=slot_fd), env=_env(),
                                 pass_fds=() if slot_fd is None else (slot_fd,))
        holder.append(child)
        for n in pending:
            child.send_signal(n)
        raw_rc = child.wait()
        with open(report) as f:
            lines = f.read().splitlines()
    finally:
        for s, h in fwd.items():
            signal.signal(s, h)
        os.unlink(report)
    return _verdict(unit, cap, sl, sl_bytes, raw_rc, lines, log)


def _verdict(unit: str, cap: int, sl: str, sl_bytes: int, raw_rc: int, lines: List[str],
             log: Callable[[str], None]) -> CapResult:
    props = unit_props(unit, "Result", "MemoryPeak", "LoadState")
    rc = 128 - raw_rc if raw_rc < 0 else raw_rc
    done = [ln.split() for ln in lines if ln.startswith("done ")]
    nocap = [ln for ln in lines if ln.startswith("nocap ")]
    peak: Optional[int] = None
    if props.get("Result") == "oom-kill" or (done and int(done[-1][3]) > 0):
        try:
            peak = int(props.get("MemoryPeak", ""))
        except ValueError:
            peak = int(done[-1][2]) if done else None
        which = "its OWN cap" if peak is not None and peak >= cap - (8 << 20) else f"the SLICE's aggregate cap ({sl})"
        log(f"[mem_cap] OOM-KILLED: {unit} hit {which} — peak {fmt_gb(peak)} of job cap {fmt_gb(cap)} "
            f"(slice cap {fmt_gb(sl_bytes)}); only this scope was stopped. exit {EXIT_OOM}")
        _systemctl("reset-failed", unit)
        return CapResult(EXIT_OOM, unit, True, peak, None)
    if nocap:
        log(f"[mem_cap] REFUSING: the cap was not in force inside {unit}: {nocap[-1][6:]}")
        return CapResult(EXIT_NOCAP, unit, False, None, None)
    if not done:
        if not any(ln.startswith("started") for ln in lines):
            log(f"[mem_cap] REFUSING: {unit} never started (systemd-run exit {rc}) — the command did NOT run")
            _systemctl("reset-failed", unit)
            return CapResult(EXIT_NOCAP, unit, False, None, None)
        log(f"[mem_cap] {unit} ended without reporting (supervisor exit {rc}: killed by a signal?) — peak unknown")
        _systemctl("reset-failed", unit)
        return CapResult(rc, unit, False, None, None)
    cmd_rc, peak = int(done[-1][1]), int(done[-1][2])
    cmd_rc = 128 - cmd_rc if cmd_rc < 0 else cmd_rc
    log(f"[mem_cap] done: {unit} exit {cmd_rc} · peak {fmt_gb(peak)} of cap {fmt_gb(cap)} "
        f"({100.0 * peak / cap:.0f}%) · not OOM-killed")
    return CapResult(cmd_rc, unit, False, peak, cmd_rc)


def status(log: Callable[[str], None] = print) -> int:
    sl = slice_name()
    p = unit_props(sl, "MemoryMax", "MemoryCurrent", "MemoryPeak", "ActiveState")
    log(f"{sl}: {p.get('ActiveState', '?')} · current {p.get('MemoryCurrent', '?')} · cap {p.get('MemoryMax', '?')}")
    r = _systemctl("list-units", "--all", "--no-legend", "--plain", f"{UNIT_PREFIX}*")
    for line in r.stdout.splitlines():
        unit = line.split()[0]
        q = unit_props(unit, "ActiveState", "Result", "MemoryCurrent", "MemoryPeak", "MemoryMax")
        log(f"  {unit}: {q.get('ActiveState')}/{q.get('Result')} current {q.get('MemoryCurrent')} "
            f"peak {q.get('MemoryPeak')} cap {q.get('MemoryMax')}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    if args[0] == "--status":
        status()
        print(HS.format_human(HS.collect()))
        return 0
    name: Optional[str] = None
    sgb: Optional[float] = None
    grace: Optional[float] = None
    force_slot, wait = False, True
    while args and args[0].startswith("--") and args[0] != "--":
        flag = args.pop(0)
        if flag == "--name" and args:
            name = args.pop(0)
        elif flag == "--slice-gb" and args:
            sgb = float(args.pop(0))
        elif flag == "--term-grace" and args:
            try:
                grace = parse_grace(args.pop(0))
            except ValueError as e:
                print(f"[mem_cap] REFUSING: --term-grace: {e}", file=sys.stderr)
                return 2
        elif flag == "--slot":
            force_slot = True
        elif flag == "--no-wait":
            wait = False
        elif flag == "--blocking" and args:
            os.environ[HS.BLOCKING_ENV] = args.pop(0)      # the child inherits the mark
        else:
            print(f"[mem_cap] unknown option {flag!r}", file=sys.stderr)
            return 2
    if not args:
        print("[mem_cap] REFUSING: no cap given (mem_cap <GB> <cmd...>)", file=sys.stderr)
        return 2
    try:
        gb = float(args.pop(0))
        gb_to_bytes(gb)
    except ValueError as e:
        print(f"[mem_cap] REFUSING: the first argument must be the cap in GB (> 0): {e}", file=sys.stderr)
        return 2
    if args and args[0] == "--":
        args.pop(0)
    if not args:
        print("[mem_cap] REFUSING: no command to run", file=sys.stderr)
        return 2
    try:
        grace = term_grace_s() if grace is None else grace        # a bad $GEN3AI_MEM_CAP_TERM_GRACE_S is a refusal
    except ValueError as e:
        print(f"[mem_cap] REFUSING: ${TERM_GRACE_ENV}: {e}", file=sys.stderr)
        return 2
    try:
        with HS.heavy_slot(args, force=force_slot, wait=wait) as slot:
            return run_capped(args, gb, name=name, slice_gb_=sgb, term_grace=grace, slot_fd=HS.slot_fd(slot)).rc
    except (HS.SlotsFull, TimeoutError) as e:
        print(str(e), file=sys.stderr)
        return HS.EXIT_SLOTS_FULL


if __name__ == "__main__":
    sys.exit(main())
