"""Isolated UI state and rendering for the training launcher.

LauncherState    — thread-safe mutable state (written from multiple threads)
LauncherSnapshot — immutable copy taken for rendering (no lock held during render)
LauncherUI       — pure rendering logic (no threads, no subprocess, fully testable)
"""

import collections
import threading
import time
from dataclasses import dataclass
from typing import Optional

import rich.box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ── Snapshot (immutable, passed to LauncherUI.render) ───────────────────────

@dataclass
class LauncherSnapshot:
    pid: Optional[int]
    run_start: float           # time.monotonic()
    deadline: float            # time.monotonic(), float("inf") if no restart
    restart_count: int
    interval_hours: float
    view_mode: str             # "dashboard" | "logs" | "confirm_restart" | "confirm_quit"
    metrics: dict              # {tag: float}
    metrics_step: int
    metrics_ts: Optional[float]    # monotonic time of last metrics update
    log_lines: list            # recent child stdout lines
    events: list               # launcher event strings (timestamped)
    initial_git_hash: Optional[str]
    pending_restart_git_hash: Optional[str]


# ── Thread-safe mutable state ────────────────────────────────────────────────

class LauncherState:
    def __init__(self, interval_hours: float) -> None:
        self._lock = threading.Lock()
        self.interval_hours = interval_hours
        self._log_lines: collections.deque = collections.deque(maxlen=500)
        self._events: list = []
        self._metrics: dict = {}
        self._metrics_step: int = 0
        self._metrics_ts: Optional[float] = None
        self.pid: Optional[int] = None
        self.run_start: float = time.monotonic()
        self.deadline: float = float("inf")
        self.restart_count: int = 0
        self.view_mode: str = "dashboard"
        self.initial_git_hash: Optional[str] = None
        self.pending_restart_git_hash: Optional[str] = None

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
        with self._lock:
            self._metrics = cleaned
            self._metrics_step = step
            self._metrics_ts = time.monotonic()

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
                log_lines=list(self._log_lines),
                events=list(self._events),
                initial_git_hash=self.initial_git_hash,
                pending_restart_git_hash=self.pending_restart_git_hash,
            )


# ── Formatting helpers (pure functions, tested separately) ───────────────────

def _elapsed_str(seconds: float) -> str:
    h, rem = divmod(int(max(0, seconds)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _fmt_val(v: float) -> str:
    """4 significant figures; comma-separated integers for large whole numbers."""
    if isinstance(v, float) and v == int(v) and abs(v) >= 1000:
        return f"{int(v):,}"
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 1000 else str(v)
    return f"{v:.4g}"


# Preferred display order; any unlisted keys are appended alphabetically.
_METRIC_ORDER = [
    "eval/mean_ep_length",
    "eval/mean_reward",
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    "time/fps",
    "time/total_timesteps",
    "train/approx_kl",
    "train/clip_fraction",
    "train/clip_range",
    "train/entropy_loss",
    "train/explained_variance",
    "train/learning_rate",
    "train/loss",
    "train/n_updates",
    "train/policy_gradient_loss",
    "train/value_loss",
]


# ── Pure rendering ───────────────────────────────────────────────────────────

class LauncherUI:
    """No threads, no subprocess, no I/O — only Rich renderables out."""

    def render(self, snap: LauncherSnapshot, console_height: int = 40):
        if snap.view_mode == "logs":
            return self._render_logs(snap, console_height)
        if snap.view_mode == "confirm_restart":
            return self._render_confirm(snap)
        if snap.view_mode == "confirm_quit":
            return self._render_confirm_quit(snap)
        return self._render_dashboard(snap)

    # ── Dashboard ────────────────────────────────────────────────────────────

    def _render_dashboard(self, snap: LauncherSnapshot):
        now = time.monotonic()
        elapsed = now - snap.run_start

        # ── Row 1: PID / run# / elapsed / restart-in ──────────────────────
        row1 = Table.grid(padding=(0, 2), expand=True)
        for _ in range(4):
            row1.add_column(ratio=1)

        pid_str = f"[bold]PID {snap.pid}[/bold]" if snap.pid else "[dim]starting…[/dim]"
        run_str = f"[cyan]run #{snap.restart_count + 1}[/cyan]"
        elapsed_str = f"[dim]{_elapsed_str(elapsed)} elapsed[/dim]"
        if snap.interval_hours > 0 and snap.deadline < float("inf"):
            remaining = max(0.0, snap.deadline - now)
            restart_str = f"[yellow]restart in {_elapsed_str(remaining)}[/yellow]"
        else:
            restart_str = "[dim]no restart[/dim]"
        row1.add_row(pid_str, run_str, elapsed_str, restart_str)

        # ── Row 2: git badge + key metrics ────────────────────────────────
        git = self._git_badge(snap)
        highlights = []
        if (steps := snap.metrics.get("time/total_timesteps")) is not None:
            highlights.append(f"steps [bold]{_fmt_val(steps)}[/bold]")
        if (fps := snap.metrics.get("time/fps")) is not None:
            highlights.append(f"fps [bold]{_fmt_val(fps)}[/bold]")
        if (rew := snap.metrics.get("rollout/ep_rew_mean")) is not None:
            col = "green" if rew >= 0 else "red"
            highlights.append(f"reward [{col}]{_fmt_val(rew)}[/{col}]")
        hl = "  │  ".join(highlights) if highlights else "[dim]waiting for first rollout…[/dim]"

        row2 = Table.grid(padding=(0, 1), expand=True)
        row2.add_column()
        row2.add_row(f"  {git}  │  {hl}")

        # ── Metrics table ─────────────────────────────────────────────────
        metrics_panel = self._render_metrics_table(snap.metrics, snap.metrics_ts, now)

        # ── Log tail ──────────────────────────────────────────────────────
        log_text = self._render_log_lines(snap.log_lines, n=6)

        # ── Events ────────────────────────────────────────────────────────
        evt_text = Text()
        for ev in snap.events[-5:]:
            evt_text.append(ev + "\n", style="dim")
        if not snap.events:
            evt_text.append("[dim]No events yet.[/dim]")

        footer = Text(
            "  [r] restart  [c] checkpoint  [q] quit  [l] logs",
            style="dim",
        )

        return Panel(
            Group(
                row1,
                row2,
                Text(""),
                Text("  Metrics", style="bold"),
                metrics_panel,
                Text(""),
                Text("  Recent Output", style="bold"),
                log_text,
                Text(""),
                Text("  Events", style="bold"),
                evt_text,
                Text(""),
                footer,
            ),
            title="[bold blue]🎮 Gen3AI Training[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )

    def _git_badge(self, snap: LauncherSnapshot) -> str:
        h = snap.initial_git_hash
        if not h:
            return "[dim]git: unknown[/dim]"
        return f"[green]✅ {h}[/green]"

    def _render_metrics_table(self, metrics: dict, metrics_ts, now: float):
        tbl = Table(
            box=rich.box.SIMPLE_HEAD,
            header_style="dim",
            show_edge=False,
            expand=True,
            padding=(0, 1),
        )
        tbl.add_column("Metric", no_wrap=True)
        tbl.add_column("Value", justify="right")

        if not metrics:
            tbl.add_row("[dim]—[/dim]", "[dim]waiting…[/dim]")
            return tbl

        stale_badge = ""
        if metrics_ts is not None:
            age = now - metrics_ts
            if age > 60:
                stale_badge = f" [dim]({int(age)}s ago)[/dim]"

        # Known metrics first (preserving the preferred order), then extras.
        seen: set = set()
        ordered = []
        for k in _METRIC_ORDER:
            if k in metrics:
                ordered.append(k)
                seen.add(k)
        for k in sorted(metrics):
            if k not in seen:
                ordered.append(k)

        current_section = None
        first_section = True
        for key in ordered:
            section, _, name = key.partition("/")
            if section != current_section:
                current_section = section
                badge = stale_badge if first_section else ""
                first_section = False
                tbl.add_row(f"[bold dim]{section}/[/bold dim]{badge}", "")
            tbl.add_row(f"  [dim]{name}[/dim]", f"[bold]{_fmt_val(metrics[key])}[/bold]")

        return tbl

    def _render_log_lines(self, log_lines: list, n: int = 6) -> Text:
        text = Text()
        for line in log_lines[-n:]:
            if "🛑" in line or "ERROR" in line.upper():
                text.append(line + "\n", style="bold red")
            elif "⚠" in line or "WARNING" in line.upper():
                text.append(line + "\n", style="yellow")
            elif "✅" in line or "🎉" in line:
                text.append(line + "\n", style="green")
            else:
                text.append(line + "\n", style="dim")
        if not log_lines:
            text.append("[dim]No output yet.[/dim]")
        return text

    # ── Log view ─────────────────────────────────────────────────────────────

    def _render_logs(self, snap: LauncherSnapshot, console_height: int = 40):
        # 2 border lines + 1 empty spacer + 1 footer = 4 lines of chrome
        n = max(1, console_height - 4)
        log_text = self._render_log_lines(snap.log_lines, n=n)
        footer = Text("  [d] dashboard", style="dim")
        return Panel(
            Group(log_text, Text(""), footer),
            title="[bold blue]🎮 Gen3AI Training[/bold blue] — [yellow]LOGS[/yellow]",
            border_style="yellow",
            padding=(0, 1),
        )

    # ── Confirm-restart view ──────────────────────────────────────────────────

    def _render_confirm(self, snap: LauncherSnapshot):
        old_h = snap.initial_git_hash or "unknown"
        new_h = snap.pending_restart_git_hash or "unknown"

        body = Text()
        body.append("\n  Was:  ", style="dim")
        body.append(old_h, style="bold green")
        body.append("      Now:  ", style="dim")
        body.append(new_h, style="bold yellow")
        body.append(
            "\n\n  The code changed since this run started.\n"
            "  The model version checker will catch architecture mismatches,\n"
            "  but proceed with caution.\n\n",
            style="dim",
        )
        body.append("  [y] proceed with restart    [n] quit launcher", style="bold")

        return Panel(
            body,
            title="[bold yellow]⚠️  Git Working Tree Changed[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )

    # ── Confirm-quit view ─────────────────────────────────────────────────────

    def _render_confirm_quit(self, snap: LauncherSnapshot):
        body = Text()
        body.append(
            "\n  This will SIGTERM the training child and wait for it to save a checkpoint.\n\n",
            style="dim",
        )
        body.append("  [y] confirm quit    [n] cancel", style="bold")

        return Panel(
            body,
            title="[bold red]⚠️  Confirm Quit[/bold red]",
            border_style="red",
            padding=(0, 1),
        )
