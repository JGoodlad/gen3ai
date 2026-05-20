"""Pure rendering logic for the training launcher TUI."""

import os
import time

import rich.box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from main.launcher.state import LauncherSnapshot


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


class LauncherUI:
    """No threads, no subprocess, no I/O — only Rich renderables out."""

    def render(self, snap: LauncherSnapshot, console_height: int = 40):
        if snap.view_mode == "logs":
            return self._render_logs(snap, console_height)
        if snap.view_mode == "confirm_quit":
            return self._render_confirm_quit(snap)
        return self._render_dashboard(snap)

    def _render_dashboard(self, snap: LauncherSnapshot):
        now = time.monotonic()
        elapsed = now - snap.run_start

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

        git = self._git_badge(snap)
        model_badge = self._model_badge(snap)
        ec_badge = (
            f"[cyan]ent_coef [bold]{snap.ent_coef}[/bold][/cyan]"
            if snap.ent_coef is not None
            else ""
        )
        highlights = []
        if (steps := snap.metrics.get("time/total_timesteps")) is not None:
            highlights.append(f"steps [bold]{_fmt_val(steps)}[/bold]")
        if (fps := snap.metrics.get("time/fps")) is not None:
            highlights.append(f"fps [bold]{_fmt_val(fps)}[/bold]")
        if (rew := snap.metrics.get("rollout/ep_rew_mean")) is not None:
            col = "green" if rew >= 0 else "red"
            highlights.append(f"reward [{col}]{_fmt_val(rew)}[/{col}]")
        hl = "  │  ".join(highlights) if highlights else "[dim]waiting for first rollout…[/dim]"

        ec_sep = f"  │  {ec_badge}" if ec_badge else ""
        row2 = Table.grid(padding=(0, 1), expand=True)
        row2.add_column()
        row2.add_row(f"  {git}  │  {model_badge}{ec_sep}  │  {hl}")

        metrics_panel = self._render_metrics_table(snap.metrics, snap.metrics_ts, now)
        log_text = self._render_log_lines(snap.log_lines, n=6)

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
        return f"[green]📌 {h[:8]}[/green]"

    def _model_badge(self, snap: LauncherSnapshot) -> str:
        if not snap.run_dir:
            return "[dim]model: —[/dim]"
        return f"[magenta]🗂  {os.path.basename(snap.run_dir)}[/magenta]"

    def _render_metrics_table(self, metrics: dict, metrics_ts, now: float):
        if not metrics:
            tbl = Table(box=rich.box.SIMPLE_HEAD, header_style="dim", show_edge=False, expand=True, padding=(0, 1))
            tbl.add_column("Metric", no_wrap=True)
            tbl.add_column("Value", justify="right")
            tbl.add_row("[dim]—[/dim]", "[dim]waiting…[/dim]")
            return tbl

        stale_badge = ""
        if metrics_ts is not None:
            age = now - metrics_ts
            if age > 60:
                stale_badge = f" [dim]({int(age)}s ago)[/dim]"

        # Order keys and group by section prefix.
        seen: set = set()
        ordered = []
        for k in _METRIC_ORDER:
            if k in metrics:
                ordered.append(k)
                seen.add(k)
        for k in sorted(metrics):
            if k not in seen:
                ordered.append(k)

        by_section: dict = {}
        for key in ordered:
            section = key.partition("/")[0]
            by_section.setdefault(section, []).append(key)

        def _make_col(sections: list) -> Table:
            t = Table(box=None, show_header=False, show_edge=False, padding=(0, 1))
            t.add_column(no_wrap=True)
            t.add_column(justify="right")
            first = True
            for sec in sections:
                keys = by_section.get(sec, [])
                if not keys:
                    continue
                for key in keys:
                    badge = stale_badge if first else ""
                    first = False
                    name = key.partition("/")[2]
                    t.add_row(f"[dim]{name}[/dim]{badge}", f"[bold]{_fmt_val(metrics[key])}[/bold]")
            return t

        left = _make_col(["rollout", "eval"])
        right = _make_col(["train", "time"])

        # Any leftover sections not in the two fixed panels.
        fixed = {"rollout", "eval", "train", "time"}
        extra_sections = [s for s in by_section if s not in fixed]

        grid = Table.grid(expand=True, padding=(0, 2))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(left, right)

        if extra_sections:
            extra = _make_col(extra_sections)
            wrapper = Table.grid(expand=True)
            wrapper.add_column()
            wrapper.add_row(grid)
            wrapper.add_row(extra)
            return wrapper

        return grid

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
