"""Textual UI for the training launcher.

``LauncherApp`` is a thin renderer: a ``set_interval`` timer reads ``state.snapshot()`` and
repaints widgets. Input is split by latency sensitivity:

  * **View navigation** (``l``/``e``/``d``, the ``q`` confirm overlay, ``n``/``y``) is handled
    **app-locally** on the event loop via the ``view_mode`` reactive — switching is instant,
    not round-tripped through the worker thread.
  * **Child-control** commands (``r`` restart, ``c`` checkpoint, ``p`` plots, ``s`` status) and
    the confirmed-quit sentinel (``__quit__``) go to the supervisor's ``cmd_q``; latency there
    is irrelevant (they aren't view changes), and ``_supervise`` handles them via
    ``input._dispatch_command``.

The blocking child-supervision loop runs in a ``@work(thread=True)`` worker (``run.run()``
wires it in); it drives ``LauncherState`` and never touches widgets.

Pure formatting helpers/constants come from ``format.py`` so the numbers stay consistent,
and win-rate/reward cells use the shared ``gradient_color`` from ``main.tui``.
"""

from __future__ import annotations

import os
import time

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import ContentSwitcher, DataTable, Static

from main.tui import Gen3App, THEME_PATH, gradient_color
from main.launcher.state import LauncherState
from main.launcher.format import _METRIC_ORDER, _elapsed_str, _fmt_metric

# Eval keys shown in the aggregate summary block (everything else under eval/ is treated
# as per-opponent detail). Mirrors the local set in ui.py's _make_eval_table.
_EVAL_SUMMARY = frozenset({
    "eval/win_rate_mean",
    "eval/win_rate_vs_bots",
    "eval/win_rate_vs_pool",
    "eval/mean_reward_mean",
    "eval/mean_reward_vs_bots",
    "eval/mean_reward_vs_pool",
    "eval/duration_sec",
})

# View ids — the app owns its own view state (the `view_mode` reactive), independent of the
# supervisor; these match the ContentSwitcher child ids.
_VIEWS = ("dashboard", "logs", "events", "confirm_quit")


class LauncherApp(Gen3App):
    """Live training-launcher dashboard, rendered with Textual.

    Pair with the supervisor in ``run.run()`` for a real session; for tests, construct
    with ``supervisor=None`` (no child is spawned) and drive ``LauncherState`` directly."""

    CSS_PATH = [str(THEME_PATH), "launcher.tcss"]
    TITLE = "🎮 Gen3AI Training"
    # No command palette — keep the footer to the launcher's own keys (parity with Rich).
    ENABLE_COMMAND_PALETTE = False

    # Header sub-title per view (Rich puts the view name in its panel title).
    _SUBTITLES = {"dashboard": "", "logs": "LOGS", "events": "EVENTS", "confirm_quit": "CONFIRM QUIT"}

    # App-owned view state. init=False so the watcher doesn't fire before #views mounts
    # (ContentSwitcher's initial="dashboard" already shows the right pane on startup).
    view_mode = reactive("dashboard", init=False)

    # Fresh bindings (NOT Gen3App.BINDINGS + …) so our own `q` overrides the base `q → quit`.
    # View nav is app-local (instant); control keys forward to the supervisor's cmd_q.
    BINDINGS = [
        Binding("d", "show_dashboard", "Dashboard"),
        Binding("l", "show_logs", "Logs"),
        Binding("e", "show_events", "Events"),
        Binding("r", "restart", "Restart"),
        Binding("c", "checkpoint", "Checkpoint"),
        Binding("p", "plots", "Plots"),
        Binding("s", "status", "Status"),
        Binding("q", "request_quit", "Quit"),
        Binding("y", "confirm_yes", "Yes", show=False),
        Binding("n", "cancel_quit", "No", show=False),
        # ctrl+c (override Textual's "ctrl+c no longer quits" system binding): first press
        # opens the confirm overlay, a second confirms — so the muscle-memory works but a
        # stray tap doesn't kill the run, and it goes through the clean save path (not the
        # degraded signal path).
        Binding("ctrl+c", "ctrl_c", "Quit", show=False),
    ]

    def __init__(self, state: LauncherState, cmd_q, supervisor=None) -> None:
        super().__init__()
        self._state = state
        self._cmd_q = cmd_q
        self.supervisor = supervisor   # callable() run in a worker thread; None in tests
        self.exit_code = 0
        self._render_error_seen = False

    # ── layout ────────────────────────────────────────────────────────────────
    def compose_body(self) -> ComposeResult:
        with ContentSwitcher(initial="dashboard", id="views"):
            with Vertical(id="dashboard"):
                yield Static(id="topline")
                yield Static(id="badges")
                yield Static("Metrics", classes="section-label")
                with Horizontal(id="metrics-row"):
                    yield DataTable(id="metrics-left")
                    yield DataTable(id="metrics-eval")
                yield Static("Events", classes="section-label")
                yield Static(id="events-dash")
            with VerticalScroll(id="logs"):
                yield Static(id="logs-body")
            with VerticalScroll(id="events"):
                yield Static(id="events-body")
            with Vertical(id="confirm_quit"):
                yield Static(id="confirm-body")

    def on_mount(self) -> None:
        left = self.query_one("#metrics-left", DataTable)
        left.show_header = False
        left.cursor_type = "none"
        left.can_focus = False   # read-only — no focus tint, keys bubble to the app
        left.add_columns("metric", "value")
        ev = self.query_one("#metrics-eval", DataTable)
        ev.cursor_type = "none"
        ev.can_focus = False
        ev.add_columns("", "win rate", "reward")

        self.set_interval(0.5, self._refresh)   # 2 Hz ≈ Rich's refresh_per_second=2
        self._refresh()

        if self.supervisor is not None:
            self._install_signal_handlers()
            self._run_supervisor()

    def _install_signal_handlers(self) -> None:
        """Route SIGHUP (closed tmux/SSH terminal) and SIGTERM (external ``kill``) to the
        clean save-and-exit path, so the *launcher* tears down cleanly instead of dying
        abruptly. The child (which shares the launcher's session) also handles SIGHUP itself
        and checkpoints — these are complementary backstops, so a closed terminal never costs
        a checkpoint or orphans the run."""
        import asyncio
        import signal as _signal
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for sig in (_signal.SIGHUP, _signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._on_shutdown_signal, sig.name)
            except (NotImplementedError, ValueError, RuntimeError, OSError):
                pass  # not on the main thread / platform unsupported — best-effort

    def _on_shutdown_signal(self, signame: str) -> None:
        """Terminal closed or external kill → ask the supervisor to stop the child cleanly."""
        try:
            self._state.add_event(f"📡 {signame} received — saving checkpoint & shutting down…")
        except Exception:  # noqa: BLE001
            pass
        self._cmd_q.put("__quit__")

    @work(thread=True, exclusive=True, group="supervisor")
    def _run_supervisor(self) -> None:
        # Long-lived worker: runs the whole session loop, then asks the app to exit.
        self.supervisor()

    # ── view navigation (app-local, instant) ─────────────────────────────────────
    def action_show_dashboard(self) -> None:
        self.view_mode = "dashboard"

    def action_show_logs(self) -> None:
        self.view_mode = "logs"

    def action_show_events(self) -> None:
        self.view_mode = "events"

    def action_request_quit(self) -> None:
        """`q` opens the confirm overlay (from any view — friendlier than Rich's
        dashboard-only confirm)."""
        self.view_mode = "confirm_quit"

    def action_cancel_quit(self) -> None:
        """`n` cancels the confirm overlay; inert elsewhere."""
        if self.view_mode == "confirm_quit":
            self.view_mode = "dashboard"

    def action_confirm_yes(self) -> None:
        """`y` confirms quit: tell the supervisor to SIGTERM+save the child, return to the
        dashboard so the 'waiting for child to save…' event is visible. Inert elsewhere."""
        if self.view_mode == "confirm_quit":
            self._cmd_q.put("__quit__")
            self.view_mode = "dashboard"

    def action_ctrl_c(self) -> None:
        """Ctrl-C: first press opens the confirm overlay, a second confirms — so the
        quit reflex works without a stray tap killing the run, and it routes through the
        clean save path."""
        if self.view_mode == "confirm_quit":
            self.action_confirm_yes()
        else:
            self.view_mode = "confirm_quit"

    def watch_view_mode(self, _old: str, new: str) -> None:
        """Reactive watcher — switch the visible pane instantly on the event loop."""
        view = new if new in _VIEWS else "dashboard"
        try:
            self.query_one("#views", ContentSwitcher).current = view
        except Exception:  # noqa: BLE001 — widgets may be gone during teardown
            return
        self.sub_title = self._SUBTITLES.get(view, "")

    # ── child-control commands (routed to the supervisor; latency here is irrelevant) ──
    def action_restart(self) -> None:
        self._control("r")

    def action_checkpoint(self) -> None:
        self._control("c")

    def action_plots(self) -> None:
        self._control("p")

    def action_status(self) -> None:
        self._control("s")

    def _control(self, ch: str) -> None:
        # Ignore control keys while the confirm overlay is up — it behaves modally.
        if self.view_mode == "confirm_quit":
            return
        self._cmd_q.put(ch)

    # ── render (event-loop only) ─────────────────────────────────────────────────
    def _refresh(self) -> None:
        # A render glitch must NEVER crash the app: app exit triggers run.py's child reap,
        # which would kill the training run over a cosmetic bug. So swallow render errors,
        # surface one event, and keep ticking — the next snapshot usually renders fine.
        try:
            snap = self._state.snapshot()
            # Repaint every view's *content* so widgets always reflect current state
            # regardless of which is shown (cheap at 2 Hz; keeps tests able to assert any
            # view). The active pane itself is owned by the `view_mode` reactive.
            self._render_dashboard(snap)
            self._render_logs(snap)
            self._render_events(snap)
            self._render_confirm(snap)
        except Exception as e:  # noqa: BLE001 — see above; never let a render fault crash the app
            if not self._render_error_seen:
                self._render_error_seen = True
                try:
                    self._state.add_event(f"⚠️ launcher render error: {e}")
                except Exception:
                    pass

    def _render_dashboard(self, snap) -> None:
        now = time.monotonic()
        elapsed = now - snap.run_start

        top = Text()
        if snap.pid:
            top.append(f"PID {snap.pid}", style="bold")
        else:
            top.append("starting…", style="dim")
        top.append("      ")
        top.append(f"run #{snap.restart_count + 1}", style="cyan")
        top.append("      ")
        top.append(f"{_elapsed_str(elapsed)} elapsed", style="dim")
        top.append("      ")
        if snap.interval_hours > 0 and snap.deadline < float("inf"):
            remaining = snap.deadline - now
            if remaining > 0:
                top.append(f"restart in {_elapsed_str(remaining)}", style="yellow")
            else:
                top.append("awaiting rollout boundary…", style="yellow")
        else:
            top.append("no restart", style="dim")
        self.query_one("#topline", Static).update(top)

        self.query_one("#badges", Static).update(self._badges(snap, now))
        self._render_metrics(snap, now)

        evt = Text()
        for ev in snap.events[-10:]:
            evt.append(ev + "\n", style="dim")
        if not snap.events:
            evt.append("No events yet.", style="dim")
        self.query_one("#events-dash", Static).update(evt)

    def _badges(self, snap, now: float) -> Text:
        sep = "  │  "
        out = Text()
        # git
        h = snap.initial_git_hash
        out.append(f"📌 {h[:8]}", style="green") if h else out.append("git: unknown", style="dim")
        out.append(sep)
        # model
        if snap.run_dir:
            out.append(f"🗂  {os.path.basename(snap.run_dir)}", style="magenta")
        else:
            out.append("model: —", style="dim")
        # restart badge
        rb = self._restart_badge(snap)
        if rb is not None:
            out.append(sep)
            out.append_text(rb)
        # steps + stall
        out.append(sep)
        if snap.metrics_step:
            out.append("steps ", style="dim")
            out.append(f"{snap.metrics_step:,}", style="bold")
        else:
            out.append("waiting for first rollout…", style="dim")
        # ent_coef (the entropy coefficient passed via --ent-coef), when known.
        if snap.ent_coef is not None:
            out.append(sep)
            out.append("ent_coef ", style="dim")
            out.append(f"{snap.ent_coef:g}", style="bold")
        idle = now - snap.last_activity_ts
        if idle > 120:
            out.append(sep)
            out.append(f"⚠ no child output for {_elapsed_str(idle)}", style="bold red")
        return out

    def _restart_badge(self, snap) -> "Text | None":
        n, crashes = snap.restart_count, snap.crash_count
        if not n and not crashes:
            return None
        word = "restart" if n == 1 else "restarts"
        if crashes:
            return Text(f"↻ {n} {word} ({crashes} crash)", style="yellow")
        return Text(f"↻ {n} {word}", style="dim")

    def _render_metrics(self, snap, now: float) -> None:
        metrics = snap.metrics
        left = self.query_one("#metrics-left", DataTable)
        right = self.query_one("#metrics-eval", DataTable)
        left.clear()
        right.clear()

        if not metrics:
            left.add_row(Text("⏳  waiting for first rollout…", style="dim italic"), "")
            return

        # Order keys (preferred order, then alpha), dropping non-actionable per-opponent
        # episode-length keys — identical to ui.py.
        display = {k: v for k, v in metrics.items() if "mean_ep_len_vs_" not in k}
        seen: set = set()
        ordered: list = []
        for k in _METRIC_ORDER:
            if k in display:
                ordered.append(k)
                seen.add(k)
        for k in sorted(display):
            if k not in seen:
                ordered.append(k)

        per_opponent: list = []
        eval_summary: dict = {}
        by_section: dict = {}
        for key in ordered:
            section = key.partition("/")[0]
            if section == "eval":
                if key in _EVAL_SUMMARY:
                    eval_summary[key] = display[key]
                else:
                    per_opponent.append(key)
            else:
                by_section.setdefault(section, []).append(key)

        stale_badge = ""
        if snap.metrics_ts is not None and (now - snap.metrics_ts) > 60:
            stale_badge = f" ({int(now - snap.metrics_ts)}s ago)"

        # ── left column: rollout / time / train (+ any other non-eval sections) ──
        fixed = {"rollout", "eval", "train", "time"}
        sections = ["rollout", "time", "train"] + [s for s in by_section if s not in fixed]
        first = True
        any_left = False
        for sec in sections:
            keys = by_section.get(sec)
            if not keys:
                continue
            any_left = True
            badge = stale_badge if first else ""
            left.add_row(Text(f"{sec}{badge}", style="dim italic"), "")
            first = False
            for key in keys:
                name = key.partition("/")[2]
                left.add_row(
                    Text(f"  {name}", style="dim"),
                    Text(_fmt_metric(key, display[key]), style="bold", justify="right"),
                )
        if not any_left:
            left.add_row(Text("⏳  waiting for first rollout…", style="dim italic"), "")

        # ── right column: eval aggregates + per-opponent ──
        eval_stale = ""
        if snap.eval_metrics_ts is not None and (now - snap.eval_metrics_ts) > 60:
            eval_stale = f" ({int(now - snap.eval_metrics_ts)}s ago)"
        self._fill_eval(right, per_opponent, display, eval_summary, eval_stale)

    def _fill_eval(self, table: DataTable, keys: list, metrics: dict,
                   eval_summary: dict, stale_badge: str) -> None:
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

        def _wr(wr):
            if wr is None:
                return Text("—", style="dim")
            return Text(f"{wr * 100:.1f}%", style=f"bold {gradient_color(wr)}")

        def _rw(rw):
            if rw is None:
                return Text("—", style="dim")
            t = (rw + 30) / 60  # -30 → 0, +30 → 1
            return Text(f"{rw:.1f}", style=f"bold {gradient_color(t)}")

        dur = eval_summary.get("eval/duration_sec")
        dur_str = f" · took {_elapsed_str(dur)}" if dur is not None else ""
        table.add_row(Text(f"eval{dur_str}{stale_badge}", style="dim italic"), "", "")

        if eval_summary:
            table.add_row("  all", _wr(eval_summary.get("eval/win_rate_mean")),
                          _rw(eval_summary.get("eval/mean_reward_mean")))
            table.add_row("  vs Bots", _wr(eval_summary.get("eval/win_rate_vs_bots")),
                          _rw(eval_summary.get("eval/mean_reward_vs_bots")))
            wr_pool = eval_summary.get("eval/win_rate_vs_pool")
            if wr_pool is not None:
                table.add_row("  vs Pool", _wr(wr_pool),
                              _rw(eval_summary.get("eval/mean_reward_vs_pool")))
            table.add_row("", "", "")

        def _row_label(opp: str) -> str:
            # sentinel_<i> is positional (newest→oldest) and maps to a different pool
            # checkpoint each cycle — surface its step (eval/sentinel_step_<i>) so it's clear
            # which snapshot it is (e.g. "vs sentinel_0 (47.0M)"); step 0 is the seed. Lowercase
            # to match the snake_case bot rows. Mirrors ui.py's _row_label.
            if opp.startswith("sentinel_"):
                idx = opp[len("sentinel_"):]
                step = metrics.get(f"eval/sentinel_step_{idx}")
                if step is None:  # not 'if step' — step 0 (the seed) is falsy!
                    return f"  vs {opp}"
                tag = "seed" if step == 0 else f"{step / 1e6:.1f}M"
                return f"  vs {opp} ({tag})"
            return f"  vs {opp}"

        # Random always first, remaining opponents in metric-order.
        if "random" in order:
            order.remove("random")
            order.insert(0, "random")
        rendered_any = False
        sentinel_sep_done = False
        for opp in order:
            # Blank divider between the fixed bot roster and the (rotating) pool sentinels.
            if opp.startswith("sentinel_") and rendered_any and not sentinel_sep_done:
                table.add_row("", "", "")
                sentinel_sep_done = True
            data = opponents[opp]
            table.add_row(_row_label(opp), _wr(data.get("win_rate")), _rw(data.get("reward")))
            rendered_any = True

    def _render_logs(self, snap) -> None:
        text = Text()
        for line in snap.log_lines[-400:]:
            up = line.upper()
            if "🛑" in line or "ERROR" in up:
                style = "bold red"
            elif "⚠" in line or "WARNING" in up:
                style = "yellow"
            elif "✅" in line or "🎉" in line:
                style = "green"
            else:
                style = "dim"
            text.append(line + "\n", style=style)
        if not snap.log_lines:
            text.append("No output yet.", style="dim")
        self.query_one("#logs-body", Static).update(text)
        if self.view_mode == "logs":
            self.query_one("#logs", VerticalScroll).scroll_end(animate=False)

    def _render_events(self, snap) -> None:
        text = Text()
        for ev in snap.events[-200:]:
            text.append(ev + "\n", style="dim")
        if not snap.events:
            text.append("No events yet.", style="dim")
        self.query_one("#events-body", Static).update(text)
        if self.view_mode == "events":
            self.query_one("#events", VerticalScroll).scroll_end(animate=False)

    def _render_confirm(self, snap) -> None:
        body = Text()
        body.append(
            "\n  This will SIGTERM the training child and wait for it to save a checkpoint.\n\n",
            style="dim",
        )
        body.append("  [y] confirm quit (or Ctrl-C)    [n] cancel", style="bold")
        self.query_one("#confirm-body", Static).update(body)
