"""Pure rendering logic for the training launcher TUI."""

import os
import time

_TMUX_OFFSET = 1 if os.environ.get("TMUX") else 0

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


def _fmt_metric(key: str, v: float) -> str:
    """Format a metric value; win rates render as percentages."""
    if "win_rate" in key:
        return f"{v * 100:.1f}%"
    return _fmt_val(v)


# Preferred display order; any unlisted keys are appended alphabetically.
_METRIC_ORDER = [
    # Eval — aggregate first, then per-opponent win rates, then per-opponent rewards.
    # Episode lengths fall alphabetically after (less actionable).
    "eval/win_rate_mean",
    "eval/mean_reward_mean",
    "eval/win_rate_vs_Random",
    "eval/win_rate_vs_Heuristic",
    "eval/win_rate_vs_Staller",
    "eval/win_rate_vs_Aggressive",
    "eval/win_rate_vs_SetupSweep",
    "eval/mean_reward_vs_Random",
    "eval/mean_reward_vs_Heuristic",
    "eval/mean_reward_vs_Staller",
    "eval/mean_reward_vs_Aggressive",
    "eval/mean_reward_vs_SetupSweep",
    # Rollout
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    # Time
    "time/fps",
    "time/total_timesteps",
    # Train
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
        if snap.view_mode == "events":
            return self._render_events(snap, console_height)
        if snap.view_mode == "confirm_quit":
            return self._render_confirm_quit(snap)
        return self._render_dashboard(snap, console_height)

    def _render_dashboard(self, snap: LauncherSnapshot, console_height: int = 40):
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
        row2.add_row(f"  {git}  │  {model_badge}  │  {hl}")

        metrics_panel = self._render_metrics_table(snap.metrics, snap.metrics_ts, now)

        evt_text = Text()
        for ev in snap.events[-10:]:
            evt_text.append(ev + "\n", style="dim")
        if not snap.events:
            evt_text.append("[dim]No events yet.[/dim]")

        footer = Text(
            "  [r] restart  [c] checkpoint  [q] quit  [l] logs  [e] events",
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
                Text("  Events", style="bold"),
                evt_text,
                Text(""),
                footer,
            ),
            title="[bold blue]🎮 Gen3AI Training[/bold blue]",
            border_style="blue",
            padding=(0, 1),
            height=console_height - _TMUX_OFFSET,
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

        # Drop episode-length-per-opponent keys — not actionable in TUI.
        display = {k: v for k, v in metrics.items() if "mean_ep_len_vs_" not in k}

        # Order keys.
        seen: set = set()
        ordered = []
        for k in _METRIC_ORDER:
            if k in display:
                ordered.append(k)
                seen.add(k)
        for k in sorted(display):
            if k not in seen:
                ordered.append(k)

        # Split eval: summary rows stay in the main columns; per-opponent detail
        # goes into a compact table below so train/time aren't crowded out.
        _EVAL_SUMMARY = frozenset({"eval/win_rate_mean", "eval/mean_reward_mean"})
        per_opponent: list = []
        by_section: dict = {}
        for key in ordered:
            section = key.partition("/")[0]
            if section == "eval" and key not in _EVAL_SUMMARY:
                per_opponent.append(key)
            else:
                by_section.setdefault(section, []).append(key)

        def _make_col(sections: list) -> Table:
            t = Table(box=None, show_header=False, show_edge=False, padding=(0, 1))
            t.add_column(no_wrap=True)
            t.add_column(justify="right")
            first = True
            active = [(s, by_section.get(s, [])) for s in sections if by_section.get(s)]
            show_headers = len(active) > 1
            for sec, keys in active:
                if show_headers:
                    badge = stale_badge if first else ""
                    first = False
                    t.add_row(f"[dim italic]{sec}{badge}[/dim italic]", "")
                for key in keys:
                    badge = stale_badge if first else ""
                    first = False
                    indent = "  " if show_headers else ""
                    name = key.partition("/")[2]
                    t.add_row(f"[dim]{indent}{name}[/dim]{badge}", f"[bold]{_fmt_metric(key, display[key])}[/bold]")
            return t

        left = _make_col(["rollout", "eval"])
        right = _make_col(["train", "time"])

        fixed = {"rollout", "eval", "train", "time"}
        extra_sections = [s for s in by_section if s not in fixed]

        grid = Table.grid(expand=True, padding=(0, 2))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(left, right)

        components: list = [grid]
        if per_opponent:
            components.append(self._make_per_opponent_table(per_opponent, display))
        if extra_sections:
            components.append(_make_col(extra_sections))

        if len(components) == 1:
            return components[0]
        wrapper = Table.grid(expand=True)
        wrapper.add_column()
        for c in components:
            wrapper.add_row(c)
        return wrapper

    def _make_per_opponent_table(self, keys: list, metrics: dict) -> Table:
        """Compact 3-column table: opponent | win rate | reward."""
        opponents: dict = {}
        order: list = []
        for key in keys:
            name_part = key.partition("/")[2]
            if "win_rate_vs_" in name_part:
                opp = name_part[len("win_rate_vs_"):]
                if opp not in opponents:
                    order.append(opp)
                    opponents[opp] = {}
                opponents[opp]["win_rate"] = metrics[key]
            elif "mean_reward_vs_" in name_part:
                opp = name_part[len("mean_reward_vs_"):]
                if opp not in opponents:
                    order.append(opp)
                    opponents[opp] = {}
                opponents[opp]["reward"] = metrics[key]

        t = Table(box=None, show_header=True, show_edge=False,
                  padding=(0, 2), header_style="dim")
        t.add_column("", style="dim", no_wrap=True)
        t.add_column("win rate", justify="right")
        t.add_column("reward", justify="right")

        for opp in order:
            data = opponents[opp]
            wr = data.get("win_rate")
            rw = data.get("reward")
            if wr is not None:
                col = "green" if wr >= 0.5 else "red"
                wr_str = f"[{col}][bold]{wr * 100:.1f}%[/bold][/{col}]"
            else:
                wr_str = "[dim]—[/dim]"
            if rw is not None:
                col = "green" if rw >= 0 else "red"
                rw_str = f"[{col}][bold]{_fmt_val(rw)}[/bold][/{col}]"
            else:
                rw_str = "[dim]—[/dim]"
            t.add_row(f"  vs {opp}", wr_str, rw_str)

        return t

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
        n = max(1, console_height - 4 - _TMUX_OFFSET)
        log_text = self._render_log_lines(snap.log_lines, n=n)
        footer = Text("  [d] dashboard", style="dim")
        return Panel(
            Group(log_text, Text(""), footer),
            title="[bold blue]🎮 Gen3AI Training[/bold blue] — [yellow]LOGS[/yellow]",
            border_style="yellow",
            padding=(0, 1),
            height=console_height - _TMUX_OFFSET,
        )

    def _render_events(self, snap: LauncherSnapshot, console_height: int = 40):
        # 2 border lines + 1 empty spacer + 1 footer = 4 lines of chrome
        n = max(1, console_height - 4 - _TMUX_OFFSET)
        evt_text = Text()
        for ev in snap.events[-n:]:
            evt_text.append(ev + "\n", style="dim")
        if not snap.events:
            evt_text.append("[dim]No events yet.[/dim]")
        footer = Text("  [d] dashboard", style="dim")
        return Panel(
            Group(evt_text, Text(""), footer),
            title="[bold blue]🎮 Gen3AI Training[/bold blue] — [yellow]EVENTS[/yellow]",
            border_style="yellow",
            padding=(0, 1),
            height=console_height - _TMUX_OFFSET,
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
