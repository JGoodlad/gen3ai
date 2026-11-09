"""``python -m main.ops.rss_watch``: the per-arm HOST-RSS sampler, with its BASELINE RULE written down.

Every 5 minutes it sums the resident memory of the launcher and every descendant (the trainer child and its env
workers) and appends one row to ``<rows-dir>/<run>.tsv`` and, for the events below, one ``RSSWATCH[<run>]: ...``
line to ``--status-file``. It replaces the session script ``rss_watch_v2.sh`` (a job-directory copy no commit held,
so the rule below could not be tested or fixed in review). The line texts and thresholds are v2's, so a reader of the
status file sees no change.

**THE BASELINE RULE (the fix; 2026-11-09).** The baseline is the first sample taken AFTER A NEW ``total_timesteps``
ROW has appeared in ``<run>/launcher_child.full.log``, that is, after the first update of THIS watcher's life: the
row count must exceed the count seen when the watcher armed (``rows_at_arm``). Never a startup sample: the pool load
and the compile ramp are not growth (v1's 2026-10-29 false page).

Why a count and not "the log has a row": ``launcher_child.full.log`` is APPENDED across lives (each resume writes a
``===== child attached <time> (pid N) =====`` line and carries on), so on a RESUME the log already holds hundreds of
rows from the earlier lives. v2 asked "does any row exist" (``grep -q total_timesteps``), answered yes at once, and
took the baseline at the resumed child's start-up sample: 4 procs, 1.70 GiB on 2026-11-06 10:32:50 (``rb_x60t_s3501``,
resume #2), after which the normal ramp to 14.2 GiB paged ("total RSS 14.20 GiB > 125 % of the baseline 1.70 GiB").
A fresh launch has ``rows_at_arm == 0``, so the count rule is v2's rule there. A watcher armed AFTER the resumed child's
first update has that update in ``rows_at_arm`` and waits for the next one: a later baseline, never a wrong one.
A log that SHRINKS (replaced by a new file) resets ``rows_at_arm`` to 0: every row in the new file is this life's.

Other events (v2's): PAGE when MemAvailable < 8 GiB, or when the total exceeds 125 % of the baseline (then it
re-baselines at the new total); a SUMMARY when the newest checkpoint reaches ``--fork-step`` + 10M, when the launcher
is gone, or after ``--max-samples`` samples (36 at 5 min is the 3-hour bound).

CPU only. Start it detached beside the launch, like the other watchers::

    CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 nohup setsid \\
        python -m main.ops.rss_watch --launcher-pid <pid> --run <name> --fork-step <N> \\
        --status-file <status.txt> --rows-dir <dir> &

It reads ``/proc`` only, signals nothing, and never touches the run. Tests: ``src/main/ops/rss_watch_test.py``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from utils import procfs
from utils.paths import MODELS_DIR_ENV_VAR, main_models_dir

#: The log line the trainer writes once per update (SB3's logger table row).
UPDATE_ROW = "total_timesteps"
#: v2's thresholds, unchanged.
PAGE_PCT = 125
MIN_AVAIL_BYTES = 8 * 2**30
SUMMARY_AFTER_FORK_STEPS = 10_000_000
DEFAULT_INTERVAL_S = 300.0
DEFAULT_MAX_SAMPLES = 36
_CKPT = re.compile(r"^checkpoint_(\d+)")
_GIB = float(2**30)


def gib(n_bytes: float) -> str:
    return f"{n_bytes / _GIB:.2f}"


# ------------------------------------------------------------------------------------------------ the baseline rule


def phase(log_rows: int, rows_at_arm: int) -> str:
    """``train`` once the child log holds an update row this watcher has not seen at arm time, else ``startup``.

    THE BASELINE RULE (module docstring): a baseline is only ever taken in the ``train`` phase."""
    return "train" if log_rows > rows_at_arm else "startup"


def count_update_rows(log: Path) -> int:
    """How many ``total_timesteps`` rows the child log holds (0 when it does not exist yet)."""
    n = 0
    try:
        with open(log, "r", errors="replace") as fh:
            for line in fh:
                if UPDATE_ROW in line:
                    n += 1
    except OSError:
        return 0
    return n


@dataclass(frozen=True)
class Sample:
    """One reading of the launcher's process tree."""

    step: int
    nproc: int
    total_bytes: int
    max_bytes: int
    max_pid: int
    avail_bytes: int
    log_rows: int


class RssWatch:
    """The decision core: feed it samples, it returns the status lines each one earns. No clock, no files, no /proc."""

    def __init__(self, fork_step: int, rows_at_arm: int, *, page_pct: int = PAGE_PCT,
                 min_avail_bytes: int = MIN_AVAIL_BYTES) -> None:
        self.end_step = fork_step + SUMMARY_AFTER_FORK_STEPS
        self.rows_at_arm = rows_at_arm
        self.page_pct = page_pct
        self.min_avail_bytes = min_avail_bytes
        self.baseline: Optional[int] = None
        self.max_total = 0
        self.last_rows = rows_at_arm
        self.last_phase = "startup"
        self.done = False

    def observe(self, s: Sample) -> List[str]:
        """The status lines ``s`` earns; ``last_phase`` is its phase column."""
        out: List[str] = []
        if s.log_rows < self.last_rows:           # the child log was replaced by a shorter file: all its rows are new
            self.rows_at_arm = 0
        self.last_rows = s.log_rows
        ph = self.last_phase = phase(s.log_rows, self.rows_at_arm)
        self.max_total = max(self.max_total, s.total_bytes)
        if self.baseline is None and ph == "train":
            self.baseline = s.total_bytes
            out.append(f"baseline (first sample after the first update): {s.nproc} procs, total {gib(s.total_bytes)} GiB, "
                       f"largest {gib(s.max_bytes)} GiB (pid {s.max_pid}), MemAvailable {gib(s.avail_bytes)} GiB")
        if s.avail_bytes < self.min_avail_bytes:
            out.append(f"PAGE: MemAvailable {gib(s.avail_bytes)} GiB < {self.min_avail_bytes // 2**30} GiB "
                       f"(total {gib(s.total_bytes)} GiB, {s.nproc} procs)")
        if self.baseline is not None and s.total_bytes * 100 > self.baseline * self.page_pct:
            out.append(f"PAGE: total RSS {gib(s.total_bytes)} GiB > {self.page_pct} % of the baseline {gib(self.baseline)} GiB "
                       "— re-baselining")
            self.baseline = s.total_bytes
        if s.step >= self.end_step:
            out.append(f"SUMMARY at 10M (step {s.step}): baseline {gib(self.baseline or 0)} GiB, last {gib(s.total_bytes)} GiB "
                       f"({s.nproc} procs, largest {gib(s.max_bytes)} GiB), max {gib(self.max_total)} GiB, "
                       f"MemAvailable {gib(s.avail_bytes)} GiB")
            self.done = True
        return out

    def launcher_gone(self, launcher: int) -> str:
        self.done = True
        return f"SUMMARY (launcher {launcher} gone): baseline {gib(self.baseline or 0)} GiB, max {gib(self.max_total)} GiB"

    def bound(self, rows_path: Path) -> str:
        self.done = True
        return f"SUMMARY (3-h bound): baseline {gib(self.baseline or 0)} GiB, max {gib(self.max_total)} GiB — see {rows_path}"


# ------------------------------------------------------------------------------------------------ the live readers


def newest_step(ckpt_dir: Path) -> int:
    best = 0
    try:
        names = os.listdir(ckpt_dir)
    except OSError:
        return 0
    for name in names:
        m = _CKPT.match(name)
        if m:
            best = max(best, int(m.group(1)))
    return best


def mem_available_bytes(meminfo: str = "/proc/meminfo") -> int:
    with open(meminfo) as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def tree_rss(launcher: int) -> Tuple[int, int, int, int]:
    """``(nproc, total_bytes, max_bytes, max_pid)`` over the launcher and every live descendant."""
    procs = procfs.snapshot()
    pids = [launcher] + procfs.descendants(launcher, procfs.children_map(procs))
    n = total = mx = 0
    mxp = 0
    for p in pids:
        r = procfs.rss_bytes(p)
        if r <= 0:
            continue
        n += 1
        total += r
        if r > mx:
            mx, mxp = r, p
    return n, total, mx, mxp


def take_sample(launcher: int, run_dir: Path) -> Sample:
    n, total, mx, mxp = tree_rss(launcher)
    return Sample(step=newest_step(run_dir / "checkpoints"), nproc=n, total_bytes=total, max_bytes=mx, max_pid=mxp,
                  avail_bytes=mem_available_bytes(), log_rows=count_update_rows(run_dir / "launcher_child.full.log"))


# ------------------------------------------------------------------------------------------------ the loop


@dataclass(frozen=True)
class Config:
    launcher: int
    run: str
    fork_step: int
    run_dir: Path
    status_file: Path
    rows_dir: Path
    interval_s: float = DEFAULT_INTERVAL_S
    max_samples: int = DEFAULT_MAX_SAMPLES


def _stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _rows_path(cfg: Config) -> Path:
    p = cfg.rows_dir / f"{cfg.run}.tsv"
    if p.exists():
        p = cfg.rows_dir / f"{cfg.run}.{datetime.now().strftime('%Y%m%d-%H%M%S')}.tsv"
    return p


def run(cfg: Config, *, sample: Optional[Callable[[], Sample]] = None, alive: Optional[Callable[[int], bool]] = None,
        sleep: Callable[[float], None] = time.sleep) -> int:
    """The watch loop; ``sample`` / ``alive`` / ``sleep`` are the seams the tests drive. Returns the exit code (0)."""
    take = sample if sample is not None else (lambda: take_sample(cfg.launcher, cfg.run_dir))
    is_alive = alive if alive is not None else procfs.pid_alive
    cfg.rows_dir.mkdir(parents=True, exist_ok=True)
    cfg.status_file.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows_path(cfg)
    (cfg.rows_dir / f"{cfg.run}.pid").write_text(f"{os.getpid()}\n")

    def log(msg: str) -> None:
        with open(cfg.status_file, "a") as fh:
            fh.write(f"[{_stamp()}] RSSWATCH[{cfg.run}]: {msg}\n")

    watch = RssWatch(cfg.fork_step, count_update_rows(cfg.run_dir / "launcher_child.full.log"))
    with open(rows, "w") as fh:
        fh.write("time\tstep\tnproc\ttotal_rss_gib\tmax_rss_gib\tmax_pid\tmem_avail_gib\tphase\n")
    log(f"armed (pid {os.getpid()}): launcher {cfg.launcher}, fork {cfg.fork_step} → summary at {watch.end_step}; rows {rows}")
    for _ in range(cfg.max_samples):
        if not is_alive(cfg.launcher):
            log(watch.launcher_gone(cfg.launcher))
            return 0
        s = take()
        lines = watch.observe(s)
        with open(rows, "a") as fh:
            fh.write(f"{datetime.now().strftime('%H:%M:%S')}\t{s.step}\t{s.nproc}\t{gib(s.total_bytes)}\t{gib(s.max_bytes)}\t"
                     f"{s.max_pid}\t{gib(s.avail_bytes)}\t{watch.last_phase}\n")
        for line in lines:
            log(line)
        if watch.done:
            return 0
        sleep(cfg.interval_s)
    log(watch.bound(rows))
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m main.ops.rss_watch", description=__doc__.split("\n\n")[0],
                                 epilog="Exit 0 after any summary; 2 on a refusal. Nothing is signalled or written outside "
                                        "--status-file and --rows-dir.")
    ap.add_argument("--launcher-pid", type=int, required=True, help="the launcher's pid (the tree root)")
    ap.add_argument("--run", required=True, help="the run NAME (its dir is <models>/<run> unless --run-dir)")
    ap.add_argument("--fork-step", type=int, required=True,
                    help="the step the run forks from; the summary fires at this + 10M (use 0 for a fresh run)")
    ap.add_argument("--status-file", type=Path, required=True, help="the file the RSSWATCH[<run>]: lines are appended to")
    ap.add_argument("--rows-dir", type=Path, required=True, help="where <run>.tsv (and <run>.pid) are written")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help=f"the run directory (default: the archive's <run>; ${MODELS_DIR_ENV_VAR} overrides the archive)")
    ap.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S, help="seconds between samples (default 300)")
    ap.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES, help="the bound (default 36 = 3 h at 300 s)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.interval_s <= 0 or args.max_samples < 1:
        print("REFUSING: --interval-s must be > 0 and --max-samples >= 1.")
        return 2
    if not procfs.pid_alive(args.launcher_pid):
        print(f"REFUSING: launcher pid {args.launcher_pid} is not alive; a watch of nothing reads as a clean run.")
        return 2
    run_dir: Optional[Path] = args.run_dir
    if run_dir is None:
        models = main_models_dir()
        if models is None:
            print(f"REFUSING: no run archive here (${MODELS_DIR_ENV_VAR} unset or not a directory); pass --run-dir.")
            return 2
        run_dir = models / args.run
    cfg = Config(launcher=args.launcher_pid, run=args.run, fork_step=args.fork_step, run_dir=run_dir,
                 status_file=args.status_file, rows_dir=args.rows_dir, interval_s=args.interval_s,
                 max_samples=args.max_samples)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
