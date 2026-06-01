import time

import pytest

from main.launcher.state import LauncherSnapshot, LauncherState
from main.launcher.ui import LauncherUI, _elapsed_str, _fmt_val


# ── Helpers ──────────────────────────────────────────────────────────────────

def _snap(**kwargs) -> LauncherSnapshot:
    defaults = dict(
        pid=12345,
        run_start=time.monotonic() - 100,
        deadline=float("inf"),
        restart_count=0,
        crash_count=0,
        interval_hours=3.0,
        view_mode="dashboard",
        metrics={},
        metrics_step=0,
        metrics_ts=None,
        eval_metrics_ts=None,
        last_activity_ts=time.monotonic(),
        log_lines=[],
        events=[],
        initial_git_hash="abc1234",
        run_dir=None,
        ent_coef=None,
    )
    defaults.update(kwargs)
    return LauncherSnapshot(**defaults)


# ── LauncherState ─────────────────────────────────────────────────────────────

class TestLauncherState:
    def test_initial_defaults(self):
        s = LauncherState(interval_hours=3.0)
        assert s.restart_count == 0
        assert s.view_mode == "dashboard"
        assert s.pid is None

    def test_add_log_appears_in_snapshot(self):
        s = LauncherState(interval_hours=3.0)
        s.add_log("hello world")
        snap = s.snapshot()
        assert "hello world" in snap.log_lines

    def test_add_event_includes_timestamp(self):
        s = LauncherState(interval_hours=3.0)
        s.add_event("something happened")
        snap = s.snapshot()
        assert any("something happened" in e for e in snap.events)
        assert any("[" in e for e in snap.events)

    def test_update_metrics_stores_values(self):
        s = LauncherState(interval_hours=3.0)
        s.update_metrics({"rollout/ep_rew_mean": -5.0, "_step": 10000})
        snap = s.snapshot()
        assert snap.metrics["rollout/ep_rew_mean"] == pytest.approx(-5.0)
        assert snap.metrics_step == 10000
        assert snap.metrics_ts is not None

    def test_update_metrics_strips_step_key(self):
        s = LauncherState(interval_hours=3.0)
        s.update_metrics({"time/fps": 1200.0, "_step": 5000})
        snap = s.snapshot()
        assert "_step" not in snap.metrics

    def test_any_child_output_marks_activity(self):
        """The stall watchdog clock advances on ANY sign of life — stdout, event, or
        metrics — so a hung child (none of these) is the only thing that goes stale."""
        for action in (
            lambda st: st.add_log("a line"),
            lambda st: st.add_event("an event"),
            lambda st: st.update_metrics({"time/fps": 1.0, "_step": 1}),
            lambda st: st.mark_activity(),
        ):
            s = LauncherState(interval_hours=3.0)
            # Backdate the clock, then perform the action; it must move forward.
            s._last_activity_ts = time.monotonic() - 1000
            before = s.last_activity_ts
            action(s)
            assert s.last_activity_ts > before
            assert s.snapshot().last_activity_ts == pytest.approx(s.last_activity_ts)

    def test_snapshot_is_isolated_copy(self):
        s = LauncherState(interval_hours=3.0)
        s.add_log("line1")
        snap = s.snapshot()
        s.add_log("line2")
        assert "line2" not in snap.log_lines

    def test_log_lines_maxlen(self):
        # Deep scrollback (5000) so a multi-line traceback isn't pushed out by
        # routine progress chatter; the full stream is also persisted to disk.
        s = LauncherState(interval_hours=3.0)
        for i in range(6000):
            s.add_log(f"line {i}")
        snap = s.snapshot()
        assert len(snap.log_lines) == 5000
        assert snap.log_lines[-1] == "line 5999"      # newest retained
        assert snap.log_lines[0] == "line 1000"       # oldest within the window

    def test_events_capped_at_30(self):
        s = LauncherState(interval_hours=3.0)
        for i in range(40):
            s.add_event(f"event {i}")
        snap = s.snapshot()
        assert len(snap.events) == 30


# ── LauncherUI ────────────────────────────────────────────────────────────────

class TestLauncherUI:
    def test_render_dashboard_empty_state(self):
        ui = LauncherUI()
        result = ui.render(_snap())
        assert result is not None

    def test_render_dashboard_with_metrics(self):
        ui = LauncherUI()
        snap = _snap(
            metrics={
                "rollout/ep_rew_mean": -17.7,
                "time/fps": 1240.0,
                "time/total_timesteps": 27335712.0,
                "train/loss": 14.6,
            },
            metrics_ts=time.monotonic(),
        )
        result = ui.render(snap)
        assert result is not None

    def test_render_logs_view(self):
        ui = LauncherUI()
        snap = _snap(view_mode="logs", log_lines=["line1", "line2", "line3"])
        result = ui.render(snap)
        assert result is not None

    def test_dashboard_surfaces_stall_when_no_output(self):
        """A silent child must visibly show a stall warning on the dashboard — the
        whole point: the TUI 'notices' instead of sitting on 'waiting…' forever."""
        from rich.console import Console
        ui = LauncherUI()
        con = Console(width=240)

        with con.capture() as cap:
            con.print(ui.render(_snap(last_activity_ts=time.monotonic())))
        fresh_text = cap.get()

        with con.capture() as cap:
            con.print(ui.render(_snap(last_activity_ts=time.monotonic() - 600)))
        stale_text = cap.get()

        assert "no child output" not in fresh_text
        assert "no child output" in stale_text

    def test_render_dispatches_all_view_modes(self):
        ui = LauncherUI()
        for mode in ("dashboard", "logs", "confirm_quit"):
            snap = _snap(view_mode=mode)
            assert ui.render(snap) is not None, f"render failed for view_mode={mode}"

    def test_render_confirm_quit(self):
        ui = LauncherUI()
        result = ui.render(_snap(view_mode="confirm_quit"))
        assert result is not None

    def test_render_with_no_pid(self):
        ui = LauncherUI()
        result = ui.render(_snap(pid=None))
        assert result is not None

    def test_render_with_restart_interval(self):
        ui = LauncherUI()
        now = time.monotonic()
        snap = _snap(
            run_start=now - 3600,
            deadline=now + 7200,
            interval_hours=3.0,
            restart_count=2,
        )
        result = ui.render(snap)
        assert result is not None

    def test_restart_badge_hidden_before_first_restart(self):
        from main.launcher.ui import LauncherUI
        assert LauncherUI()._restart_badge(_snap(restart_count=0, crash_count=0)) is None

    def test_restart_badge_shows_count_and_crashes(self):
        """The dashboard must surface how many relaunches happened and how many were
        crash recoveries — the whole point of the feature."""
        from rich.console import Console
        ui = LauncherUI()
        con = Console(width=240)
        with con.capture() as cap:
            con.print(ui.render(_snap(restart_count=3, crash_count=1)))
        text = cap.get()
        assert "3 restarts" in text
        assert "1 crash" in text

    def test_restart_badge_singular_and_no_crash(self):
        from main.launcher.ui import LauncherUI
        badge = LauncherUI()._restart_badge(_snap(restart_count=1, crash_count=0))
        assert "1 restart" in badge and "restarts" not in badge and "crash" not in badge

    def test_render_stale_metrics_badge(self):
        ui = LauncherUI()
        snap = _snap(
            metrics={"time/fps": 1000.0},
            metrics_ts=time.monotonic() - 120,  # 2 minutes stale
        )
        result = ui.render(snap)
        assert result is not None

    def test_metrics_table_unknown_keys_appended(self):
        ui = LauncherUI()
        snap = _snap(metrics={"custom/my_metric": 42.0, "time/fps": 1000.0})
        result = ui.render(snap)
        assert result is not None


# ── Formatting helpers ────────────────────────────────────────────────────────

class TestFormatHelpers:
    def test_fmt_val_large_int_like_float(self):
        result = _fmt_val(27335712.0)
        assert "27,335,712" == result

    def test_fmt_val_small_float_4_sig_figs(self):
        result = _fmt_val(0.015123705)
        assert result == "0.01512"

    def test_fmt_val_negative(self):
        result = _fmt_val(-17.7)
        assert result == "-17.7"

    def test_fmt_val_integer_below_1000(self):
        result = _fmt_val(42.0)
        assert result == "42"

    def test_fmt_val_integer_above_1000(self):
        result = _fmt_val(1380.0)
        assert result == "1,380"

    def test_elapsed_str_seconds_only(self):
        assert _elapsed_str(90) == "01:30"

    def test_elapsed_str_with_hours(self):
        assert _elapsed_str(3700) == "1:01:40"

    def test_elapsed_str_zero(self):
        assert _elapsed_str(0) == "00:00"

    def test_elapsed_str_negative_clamped(self):
        result = _elapsed_str(-5)
        assert result == "00:00"
