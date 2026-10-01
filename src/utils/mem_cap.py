"""Run ONE command under a hard memory cap, in its OWN transient systemd scope, inside ONE shared
heavy-job slice — so a runaway job is OOM-killed ALONE and never takes the Claude session with it.

USAGE
    scripts/ops/mem_cap.sh <GB> <cmd> [args...]                 # the shell form
    scripts/ops/mem_cap.sh --name k8trace 24 timeout 2h python -m main.compile_inventory run ...
    scripts/ops/mem_cap.sh --status                              # the slice and every capped job now
    python -m utils.mem_cap [--name N] [--slice-gb G] <GB> -- <cmd> [args...]

    # composes with the GPU lock in BOTH orders; a timeout goes INSIDE:
    scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 24 timeout 1h python ...
    scripts/ops/mem_cap.sh 24 scripts/ops/gpu_lock.sh timeout 1h python ...

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
   kills the LARGER, and only its scope stops.
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

SLICE_ENV = "GEN3AI_HEAVY_SLICE"
SLICE_GB_ENV = "GEN3AI_HEAVY_SLICE_GB"
DEFAULT_SLICE = "gen3ai-heavy.slice"
DEFAULT_SLICE_GB = 64.0
OOM_SCORE_ADJ = 500
UNIT_PREFIX = "gen3ai-capped-"
EXIT_NOCAP = 85
EXIT_OOM = 86
GiB = 1 << 30
PAGE = 4096

# Runs INSIDE the scope, as ``python -I -S -c SUPERVISOR <report> <job max> <slice max> <adj> -- cmd...``.
# Stdlib only and import-light: it must not depend on PYTHONPATH (and must not perturb the command's).
SUPERVISOR = r"""
import os, signal, subprocess, sys
report, want, want_slice, adj = sys.argv[1:5]
cmd = sys.argv[6:]
def rd(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return ""
def out(s):
    with open(report, "a") as f:
        f.write(s + "\n")
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
out("started cgroup=%s" % rel)
try:
    child = subprocess.Popen(cmd)
except OSError as e:
    out("done 127 %s 0" % (rd(cg + "/memory.peak") or "0"))
    print("[mem_cap] cannot run %r: %s" % (cmd[0], e), file=sys.stderr)
    sys.exit(127)
for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(s, lambda n, _f: child.send_signal(n))
rc = child.wait()
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
                report: str, adj: int = OOM_SCORE_ADJ) -> List[str]:
    """The ``systemd-run`` argv that runs ``cmd`` under the supervisor in a capped scope."""
    return ["systemd-run", "--user", "--scope", "--quiet", f"--slice={slice_}",
            "-p", f"MemoryMax={cap_bytes}", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=stop",
            "--unit", unit, "--",
            sys.executable, "-I", "-S", "-c", SUPERVISOR, report, str(cap_bytes), str(slice_bytes), str(adj),
            "--", *cmd]


def run_capped(cmd: Sequence[str], gb: float, *, name: Optional[str] = None,
               slice_: Optional[str] = None, slice_gb_: Optional[float] = None,
               log: Callable[[str], None] = _stderr) -> CapResult:
    """Run ``cmd`` capped at ``gb`` GB in its own scope inside the heavy slice; report and return."""
    cap = gb_to_bytes(gb)
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
    try:
        child = subprocess.Popen(capped_argv(cmd, cap, unit=unit, slice_=sl, slice_bytes=sl_bytes, report=report),
                                 env=_env())
        fwd = {s: signal.signal(s, lambda n, _f: child.send_signal(n)) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            raw_rc = child.wait()
        finally:
            for s, h in fwd.items():
                signal.signal(s, h)
        with open(report) as f:
            lines = f.read().splitlines()
    finally:
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
        return status()
    name: Optional[str] = None
    sgb: Optional[float] = None
    while args and args[0].startswith("--") and args[0] != "--":
        flag = args.pop(0)
        if flag == "--name" and args:
            name = args.pop(0)
        elif flag == "--slice-gb" and args:
            sgb = float(args.pop(0))
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
    return run_capped(args, gb, name=name, slice_gb_=sgb).rc


if __name__ == "__main__":
    sys.exit(main())
