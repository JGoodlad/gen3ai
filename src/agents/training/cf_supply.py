"""The cf label SUPPLY as a DECLARED run resource — started (or verified) at startup, guarded in flight.

`gen3_supply_guard_v1` (2026-09-30). The class this closes: `ai_v12_12_ladder_cflabels` trained 10M
steps at `--cf-winprob-coef 0.5` and received ZERO counterfactual labels. The label producer
(`cf_producer.py`) is a separate out-of-process program; nobody started it, and every signal that
could have said so — an empty `cf_labels/`, `cf/labels_ingested_total` flat at 0, `train/cf_loss`
never written — was a scalar or a directory nobody thresholded. The duty-cycle refusal checks
checkpoint CADENCE, not whether a producer EXISTS. So a lever whose flag was set did nothing for a
whole run, and the run read as a result about that lever.

**Two layers, both fail CLOSED.**

1. **STARTUP — the supply is a declared resource.** Any live cf-buffer consumer
   (:data:`CF_CONSUMER_COEFS`) makes the supply mandatory, and `--cf-label-supply` says who provides
   it:

   * ``producer`` (default): THE TRAINER STARTS `cf_producer` itself (:class:`CfProducerSupply`),
     as its child, right after the run dir exists — log `<run>/cf_producer.log`, bound to the trainer
     by `--parent-pid` (PR_SET_PDEATHSIG + a ppid check), so a trainer that dies by ANY path takes
     its producer with it and the next launcher restart starts a fresh one that resumes from the
     crash-safe `cf_producer_state.json`. It REQUIRES `--cf-records` (the producer labels the ring's
     records; without the tap there is nothing to label) — a `combination_checks` FATAL_CONFIG.
   * ``external``: an operator-run producer serves the run. Verified at startup by the producer's
     single-instance LOCK (`<run>/cf_producer.lock`, held for the producer's lifetime): not held →
     `FATAL_CONFIG`, naming the exact command to start one.

   Why the trainer and not the launcher owns the child: the launcher is optional (a bare trainer run,
   `--debug`, a test harness), and a guard that only exists under one entry point is the hole this
   module closes. The launcher already restarts the trainer; the producer rides that lifecycle.

2. **IN FLIGHT — :class:`CfSupplyGuard`.** Once a checkpoint exists (the producer cannot label
   before one — it stamps every row with the newest checkpoint's step), a STREAM a live coefficient
   reads must accept at least one row within ``--cf-supply-starve-cycles`` consecutive train()
   cycles AND ``--cf-supply-starve-minutes`` of wall. Otherwise it raises
   :class:`CfLabelSupplyError` (a `main.exit_codes.SupplyStarvedError`) → exit `FATAL_SUPPLY` (5),
   which the launcher does NOT restart. A spawned producer that EXITS is the same FATAL at once, with
   the tail of its log. The end-of-run summary prints the ingest totals and is LOUD at zero.

Streams: the base row stream (`ingested_total`) serves `cf_winprob` / `cf_evidential` / `cf_twin` /
`cf_shadow`; the per-action stream (`q_ingested_total`) serves `q_winprob_coef`; the on-policy
fallback stream (`onpolicy_ingested_total`) serves `q_winprob_onpolicy_coef`. A producer that ships
rows but no `q_labels` starves the Q head exactly as a dead producer starves the win-prob head, so
each live coefficient is guarded on the stream it actually reads.

Nothing here imports torch; the trainer's startup imports it cheaply.
"""
from __future__ import annotations

import fcntl
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from main.exit_codes import SupplyStarvedError

#: Every coefficient whose term reads rows from the cf label buffer. ONE list, read by the trainer's
#: buffer gate, the startup supply check and the in-flight guard alike.
CF_CONSUMER_COEFS: Tuple[str, ...] = (
    "cf_winprob_coef", "cf_evidential_coef", "cf_twin_coef", "cf_shadow_coef",
    "q_winprob_coef", "q_winprob_onpolicy_coef")

#: coefficient → the buffer counter (attribute name) that measures the stream it reads.
STREAM_OF: Dict[str, str] = {
    "cf_winprob_coef": "ingested_total",
    "cf_evidential_coef": "ingested_total",
    "cf_twin_coef": "ingested_total",
    "cf_shadow_coef": "ingested_total",
    "q_winprob_coef": "q_ingested_total",
    "q_winprob_onpolicy_coef": "onpolicy_ingested_total",
}

SUPPLY_CHOICES = ("producer", "external")
DEFAULT_SUPPLY = "producer"
DEFAULT_STARVE_CYCLES = 5
DEFAULT_STARVE_MINUTES = 30.0

LOCK_NAME = "cf_producer.lock"
PRODUCER_LOG_NAME = "cf_producer.log"
#: `cf_producer`'s exit code when another producer already holds the run's lock.
PRODUCER_EXIT_LOCK_HELD = 4
#: How long the trainer waits for a previous segment's producer to release the lock (it is SIGTERM'd
#: by PDEATHSIG the moment its trainer dies, so this is normally sub-second).
LOCK_RELEASE_WAIT_S = 30.0


class CfLabelSupplyError(SupplyStarvedError):
    """The cf label supply died or starved while a coefficient that reads it was live."""


# -- which coefficients are live ------------------------------------------------------------------

def _coef(args, name: str) -> float:
    try:
        return float(getattr(args, name, None) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def live_cf_consumers(args) -> List[str]:
    """The live cf-buffer consumers in `args` (a coefficient > 0), in declaration order."""
    return [c for c in CF_CONSUMER_COEFS if _coef(args, c) > 0.0]


def supply_mode(args) -> str:
    return str(getattr(args, "cf_label_supply", None) or DEFAULT_SUPPLY)


# -- the producer's single-instance lock ------------------------------------------------------------

def lock_path(run_dir: str) -> str:
    return os.path.join(run_dir, LOCK_NAME)


def acquire_producer_lock(run_dir: str) -> Optional[int]:
    """Take the run's producer lock for this process's lifetime. Returns the fd, or None if held.

    `flock` is released by the kernel when the holder dies by ANY path (SIGKILL included), so a
    stale lock cannot outlive its producer. The holder's pid is written into the file for the
    message a second claimant prints."""
    fd = os.open(lock_path(run_dir), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    os.fsync(fd)
    return fd


def live_producer_pid(run_dir: str) -> Optional[int]:
    """The pid of a producer HOLDING the run's lock right now, else None (probe, never keeps it)."""
    path = lock_path(run_dir)
    if not os.path.exists(path):
        return None
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            try:
                return int((os.pread(fd, 32, 0).decode().strip() or "0")) or -1
            except ValueError:
                return -1
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def bind_to_parent(parent_pid: int) -> bool:
    """Die with the parent: PR_SET_PDEATHSIG(SIGTERM), then close the fork/prctl race by checking
    the parent is still the one we were told. Returns False when it already is not."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)          # PR_SET_PDEATHSIG == 1
    except (OSError, AttributeError):                          # pragma: no cover - non-Linux
        pass
    return os.getppid() == int(parent_pid)


# -- the producer command ---------------------------------------------------------------------------

def producer_args(args) -> List[str]:
    """The producer flags THIS run's live coefficients require, then the operator's own.

    The trainer knows which streams it reads, so it asks for them: a live Q coefficient needs
    `--q-labels` (a producer without it ships rows with no `q_labels` and no `taken_action`, and
    both Q terms would fold nothing — `cf/q_label_coverage` 0.0 for a whole run)."""
    out: List[str] = []
    if _coef(args, "q_winprob_coef") > 0 or _coef(args, "q_winprob_onpolicy_coef") > 0:
        out.append("--q-labels")
    extra = getattr(args, "cf_producer_args", None)
    if extra:
        out += shlex.split(str(extra))
    return out


def producer_command(run_dir: str, args, *, python: Optional[str] = None) -> List[str]:
    """The argv of the producer this run needs (without `--parent-pid`)."""
    return [python or sys.executable, "-m", "agents.training.cf_producer", run_dir,
            *producer_args(args)]


def producer_command_for_humans(run_dir: str, args) -> str:
    """The exact line an operator types to start this run's producer (the `external` remedy)."""
    cmd = " ".join(shlex.quote(c) for c in producer_command(run_dir, args, python="python"))
    return (f"export PYTHONPATH=$PYTHONPATH:src && nohup nice -n 10 {cmd} "
            f">> {shlex.quote(os.path.join(run_dir, PRODUCER_LOG_NAME))} 2>&1 < /dev/null &")


def _tail(path: str, n: int = 12) -> List[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 16384))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


# -- the supply handle --------------------------------------------------------------------------------

@dataclass
class CfProducerSupply:
    """The run's cf label supply: a producer the trainer SPAWNED, or an EXTERNAL one it verified."""

    run_dir: str
    mode: str
    consumers: Tuple[str, ...]
    proc: Optional[subprocess.Popen] = None
    external_pid: Optional[int] = None
    log_path: str = ""
    argv: List[str] = field(default_factory=list)

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc is not None else self.external_pid

    def describe(self) -> str:
        who = (f"spawned cf_producer pid {self.pid}" if self.mode == "producer"
               else f"external cf_producer pid {self.pid}")
        return f"{who} → {self.log_path or '(its own log)'}"

    def check_alive(self) -> None:
        """A SPAWNED producer that exited is a dead supply: raise now, with its log's tail."""
        if self.proc is None:
            return
        rc = self.proc.poll()
        if rc is None:
            return
        tail = _tail(self.log_path)
        raise CfLabelSupplyError(
            f"[SUPPLY] FATAL: the cf label producer (pid {self.proc.pid}) EXITED with code {rc} "
            f"while {', '.join(self.consumers)} is live — the run would keep training on a dead "
            f"supply. Its log ({self.log_path}) ends:\n    " + "\n    ".join(tail or ["(empty)"]))

    def stop(self, timeout: float = 10.0) -> Optional[int]:
        """SIGTERM the spawned producer by its PID, SIGKILL after `timeout`. External: untouched."""
        if self.proc is None or self.proc.poll() is not None:
            return None if self.proc is None else self.proc.returncode
        try:
            self.proc.send_signal(signal.SIGTERM)
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return self.proc.wait(timeout=timeout)


def _wait_for_lock_release(run_dir: str, wait_s: float) -> Optional[int]:
    deadline = time.monotonic() + max(0.0, wait_s)
    holder = live_producer_pid(run_dir)
    while holder is not None and time.monotonic() < deadline:
        time.sleep(0.25)
        holder = live_producer_pid(run_dir)
    return holder


class CfSupplyConfigError(RuntimeError):
    """A startup refusal — the caller exits `FATAL_CONFIG` with this message."""


def preflight_cf_label_supply(args, run_dir: str) -> None:
    """The half of the startup check that needs no run dir — called BEFORE the trainer creates it,
    so a refusal leaves nothing behind. `external` with no live producer is refused here."""
    consumers = live_cf_consumers(args)
    if not consumers:
        return
    mode = supply_mode(args)
    if mode not in SUPPLY_CHOICES:
        raise CfSupplyConfigError(f"[SUPPLY] FATAL: --cf-label-supply {mode!r} is not one of "
                                  f"{SUPPLY_CHOICES}.")
    if mode == "external" and live_producer_pid(os.path.abspath(run_dir)) is None:
        run_dir = os.path.abspath(run_dir)
        raise CfSupplyConfigError(
            f"\n[SUPPLY] FATAL: {', '.join(consumers)} is live and --cf-label-supply external "
            f"declares an operator-run producer, but NO cf_producer holds {lock_path(run_dir)}.\n"
            f"  A live coefficient with no supply folds NOTHING for the whole run "
            f"(ai_v12_12_ladder_cflabels, 10M steps, 0 labels).\n"
            f"  Start it (then relaunch):\n    mkdir -p {shlex.quote(os.path.join(run_dir, 'cf_records'))} "
            f"&& {producer_command_for_humans(run_dir, args)}\n"
            f"  or drop --cf-label-supply external and the trainer starts it itself.")


def start_cf_label_supply(args, run_dir: str, *, python: Optional[str] = None,
                          env: Optional[Dict[str, str]] = None,
                          lock_wait_s: float = LOCK_RELEASE_WAIT_S,
                          emit: Callable[[str], None] = print) -> Optional[CfProducerSupply]:
    """Acquire the cf label supply for this run at STARTUP, or refuse (:class:`CfSupplyConfigError`).

    Returns None when no cf-buffer consumer is live (nothing to supply; a default run is untouched).
    """
    consumers = tuple(live_cf_consumers(args))
    if not consumers:
        return None
    mode = supply_mode(args)
    run_dir = os.path.abspath(run_dir)
    if mode == "external":
        preflight_cf_label_supply(args, run_dir)
        pid = live_producer_pid(run_dir)
        emit(f"🏭 [SUPPLY] cf labels ← external cf_producer pid {pid} (holds {LOCK_NAME}) for "
             f"{', '.join(consumers)}")
        return CfProducerSupply(run_dir=run_dir, mode=mode, consumers=consumers,
                                external_pid=pid, argv=[])
    preflight_cf_label_supply(args, run_dir)      # an unknown mode refuses here
    # mode == "producer": the trainer owns it.
    if not os.path.isdir(os.path.join(run_dir, "cf_records")):
        raise CfSupplyConfigError(
            f"\n[SUPPLY] FATAL: {', '.join(consumers)} is live but {run_dir}/cf_records/ does not "
            f"exist — the producer labels the reconstruction records --cf-records rings there, so "
            f"it would have nothing to label. Pass --cf-records, or set the coefficient(s) to 0.")
    holder = _wait_for_lock_release(run_dir, lock_wait_s)
    if holder is not None:
        raise CfSupplyConfigError(
            f"\n[SUPPLY] FATAL: a cf_producer (pid {holder}) already holds {lock_path(run_dir)} and "
            f"did not release it within {lock_wait_s:.0f} s. Two producers on one run share one "
            f"state file. Either stop it (kill {holder}) and relaunch, or declare it: "
            f"--cf-label-supply external.")
    argv = producer_command(run_dir, args, python=python) + ["--parent-pid", str(os.getpid())]
    log_path = os.path.join(run_dir, PRODUCER_LOG_NAME)
    child_env = dict(os.environ if env is None else env)
    # The producer must import THIS tree's code (a pinned run's worktree), so the trainer's own
    # import root goes first on its path, whatever the ambient PYTHONPATH says.
    from utils.paths import src_root
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(src_root())] + [p for p in child_env.get("PYTHONPATH", "").split(os.pathsep) if p])
    with open(log_path, "ab") as log:
        log.write(f"\n=== spawned by trainer pid {os.getpid()} at "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(argv)}\n".encode())
        log.flush()
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=child_env, close_fds=True)
    supply = CfProducerSupply(run_dir=run_dir, mode=mode, consumers=consumers, proc=proc,
                              log_path=log_path, argv=argv)
    emit(f"🏭 [SUPPLY] cf labels ← {supply.describe()} for {', '.join(consumers)} "
         f"(a DECLARED startup resource: it dies with this trainer; a starved or dead supply is "
         f"FATAL_SUPPLY, never a quiet run)")
    return supply


# -- the in-flight guard ------------------------------------------------------------------------------

@dataclass
class _StreamState:
    last_total: int = 0
    starved_cycles: int = 0
    starved_since: float = 0.0


class CfSupplyGuard:
    """Raise when a live coefficient's stream accepts nothing for N cycles AND T minutes.

    Pure state machine — :meth:`observe` is called once per train() cycle with the buffer and a
    flag saying whether any checkpoint exists yet. Disarmed until the first checkpoint (the
    producer cannot stamp a label before one); `starve_cycles <= 0` disables it (announced)."""

    def __init__(self, consumers: Sequence[str], *, starve_cycles: int = DEFAULT_STARVE_CYCLES,
                 starve_minutes: float = DEFAULT_STARVE_MINUTES,
                 supply: Optional[CfProducerSupply] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.consumers = tuple(consumers)
        self.streams: Dict[str, Tuple[str, ...]] = {}
        for c in self.consumers:
            s = STREAM_OF[c]
            self.streams[s] = self.streams.get(s, ()) + (c,)
        self.starve_cycles = int(starve_cycles)
        self.starve_s = float(starve_minutes) * 60.0
        self.supply = supply
        self._clock = clock
        self.armed = False
        self.cycles = 0
        self._state: Dict[str, _StreamState] = {s: _StreamState() for s in self.streams}

    @property
    def enabled(self) -> bool:
        return self.starve_cycles > 0

    def starved_cycles(self) -> int:
        return max((st.starved_cycles for st in self._state.values()), default=0)

    def observe(self, buffer, *, checkpoint_present: bool) -> None:
        self.cycles += 1
        if self.supply is not None:
            self.supply.check_alive()
        now = self._clock()
        if not self.armed:
            if not checkpoint_present:
                return
            self.armed = True
            for s, st in self._state.items():
                st.last_total = int(getattr(buffer, s, 0))
                st.starved_since = now
            # The arming cycle counts as observed: rows that already arrived are not "new".
            return
        for s, st in self._state.items():
            total = int(getattr(buffer, s, 0))
            if total > st.last_total:
                st.last_total, st.starved_cycles, st.starved_since = total, 0, now
                continue
            st.starved_cycles += 1
            if (self.enabled and st.starved_cycles >= self.starve_cycles
                    and now - st.starved_since >= self.starve_s):
                raise CfLabelSupplyError(self._message(s, st, buffer, now))

    def _message(self, stream: str, st: _StreamState, buffer, now: float) -> str:
        coefs = ", ".join(self.streams[stream])
        expired = int(getattr(buffer, "expired_total", 0))
        skipped = int(getattr(buffer, "skipped_total", 0))
        rows = int(getattr(buffer, "ingested_total", 0))
        if stream != "ingested_total" and rows > 0:
            why = (f"base rows DO arrive ({rows}) but none carries the `{stream}` field — the "
                   f"producer is not shipping the stream this coefficient reads (--q-labels?)")
        elif expired:
            why = (f"rows ARE arriving but every one EXPIRED at ingest (cf/labels_expired_total "
                   f"{expired}) — the producer lags the policy past --cf-label-lag-steps; read the "
                   f"duty cycle the launch printed")
        else:
            why = (f"NO usable row arrived (skipped {skipped}) — the producer is dead, absent or "
                   f"writing somewhere else")
        who = self.supply.describe() if self.supply is not None else "(no supply handle)"
        return (f"\n[SUPPLY] FATAL: {coefs} is live but its label stream `{stream}` accepted "
                f"{st.last_total} row(s) in total and NONE for {st.starved_cycles} consecutive "
                f"train() cycles / {(now - st.starved_since) / 60.0:.1f} min since the first "
                f"checkpoint or the last arrival (floor: --cf-supply-starve-cycles "
                f"{self.starve_cycles} and --cf-supply-starve-minutes {self.starve_s / 60.0:g}).\n"
                f"  Cause: {why}.\n  Supply: {who}.\n"
                f"  Training on would make this run read as a result about a lever that never "
                f"engaged (ai_v12_12_ladder_cflabels: 10M steps, 0 labels). Not restarting.")

    def summary(self, buffer) -> List[str]:
        """The end-of-run lines — LOUD when a live coefficient's stream never delivered a row."""
        lines: List[str] = []
        for s, coefs in self.streams.items():
            total = int(getattr(buffer, s, 0)) if buffer is not None else 0
            if total == 0:
                lines.append(
                    f"🚨🚨 [SUPPLY] ZERO cf labels ARRIVED on `{s}` this segment — "
                    f"{', '.join(coefs)} folded NOTHING. This segment is NOT evidence about that "
                    f"lever. (guard armed={self.armed}, enabled={self.enabled}, "
                    f"cycles={self.cycles})")
            else:
                lines.append(f"🏭 [SUPPLY] `{s}`: {total} rows accepted this segment "
                             f"({', '.join(coefs)}); expired "
                             f"{int(getattr(buffer, 'expired_total', 0))}, skipped "
                             f"{int(getattr(buffer, 'skipped_total', 0))}")
        return lines


def any_checkpoint(run_dir: str) -> bool:
    """Does the run have a checkpoint the producer could load? (Its own resolver, lazily.)"""
    from agents.training.cf_producer_snapshot import resolve_latest_checkpoint
    return resolve_latest_checkpoint(run_dir) is not None
