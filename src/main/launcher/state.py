"""Thread-safe mutable state and immutable snapshot for the training launcher."""

import collections
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class LauncherSnapshot:
    pid: Optional[int]
    run_start: float           # time.monotonic()
    deadline: float            # time.monotonic(), float("inf") if no restart
    restart_count: int
    interval_hours: float
    view_mode: str             # "dashboard" | "logs" | "events" | "confirm_quit"
    metrics: dict              # {tag: float}
    metrics_step: int
    metrics_ts: Optional[float]      # monotonic time of last rollout/train metrics update
    eval_metrics_ts: Optional[float] # monotonic time of last eval/* metrics update
    log_lines: list            # recent child stdout lines
    events: list               # launcher event strings (timestamped)
    initial_git_hash: Optional[str]
    run_dir: Optional[str]     # current checkpoint output directory
    ent_coef: Optional[float]  # entropy coefficient passed via --ent-coef


class LauncherState:
    def __init__(self, interval_hours: float) -> None:
        self._lock = threading.Lock()
        self.interval_hours = interval_hours
        # Keep a deep scrollback so a multi-line child traceback (~25+ lines) is
        # never pushed out of the live logs view or the on-exit dump by routine
        # progress chatter. The full stream is also persisted to disk by child.py.
        self._log_lines: collections.deque = collections.deque(maxlen=5000)
        self._events: list = []
        self._metrics: dict = {}
        self._metrics_step: int = 0
        self._metrics_ts: Optional[float] = None
        self._eval_metrics_ts: Optional[float] = None
        self.pid: Optional[int] = None
        self.run_start: float = time.monotonic()
        self.deadline: float = float("inf")
        self.restart_count: int = 0
        self.view_mode: str = "dashboard"
        self.initial_git_hash: Optional[str] = None
        self.run_dir: Optional[str] = None
        self.ent_coef: Optional[float] = None

    def add_log(self, line: str) -> None:
        with self._lock:
            self._log_lines.append(line)

    def add_event(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with self._lock:
            self._events.append(f"[{ts}] {msg}")
            if len(self._events) > 30:
                self._events = self._events[-30:]

    def update_metrics(self, payload: dict) -> None:
        step = int(payload.get("_step", 0))
        cleaned = {k: v for k, v in payload.items() if k != "_step"}
        now = time.monotonic()
        with self._lock:
            self._metrics.update(cleaned)
            self._metrics_step = step
            self._metrics_ts = now
            if any(k.startswith("eval/") for k in cleaned):
                self._eval_metrics_ts = now

    def snapshot(self) -> LauncherSnapshot:
        with self._lock:
            return LauncherSnapshot(
                pid=self.pid,
                run_start=self.run_start,
                deadline=self.deadline,
                restart_count=self.restart_count,
                interval_hours=self.interval_hours,
                view_mode=self.view_mode,
                metrics=dict(self._metrics),
                metrics_step=self._metrics_step,
                metrics_ts=self._metrics_ts,
                eval_metrics_ts=self._eval_metrics_ts,
                log_lines=list(self._log_lines),
                events=list(self._events),
                initial_git_hash=self.initial_git_hash,
                run_dir=self.run_dir,
                ent_coef=self.ent_coef,
            )
