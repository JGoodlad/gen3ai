"""Tests for the Textual launcher UI (``app.py``) + supervisor (``run.py``).

Two halves:
  * **Pilot tests** mount ``LauncherApp`` with no supervisor (no child spawned), drive
    ``LauncherState`` directly, and assert the rendered widgets + key→cmd_q forwarding +
    view switching + the confirm-quit/ctrl-c/signal flows.
  * **Loop tests** drive ``_supervise`` deterministically with a mocked child/checkpoint,
    asserting the *returned* exit code (it returns instead of calling ``sys.exit``).

Async tests run via ``asyncio_mode = auto`` (pytest.ini). No real Showdown server, no real
checkpoint, no torch.
"""

import io
import queue
import signal
import subprocess
import time
from contextlib import ExitStack
from unittest.mock import MagicMock, call, patch

from textual.widgets import ContentSwitcher, DataTable, Static

from main.exit_codes import TrainExitCode
from main.launcher.run import (
    _SessionCtx,
    _fatal_config_reason,
    _prepare_session,
    _reap,
    _supervise,
)
from main.launcher.state import LauncherState
from main.launcher.app import LauncherApp, _EVENT_HANG_INDENT, _wrap_hanging


# ── helpers ──────────────────────────────────────────────────────────────────

def _plain(widget: Static) -> str:
    """Plain text of a Static's current content."""
    return str(widget.content)


def _table_text(table: DataTable) -> str:
    """All cell text in a DataTable, newline-joined."""
    out = []
    for r in range(table.row_count):
        for cell in table.get_row_at(r):
            out.append(str(cell))
    return "\n".join(out)


def _populated_state() -> LauncherState:
    """A LauncherState resembling a mid-run snapshot (matches the Rich screenshot)."""
    state = LauncherState(interval_hours=3.0)
    state.pid = 1686225
    state.initial_git_hash = "24e6d7c4abcd"
    state.run_dir = "models/run_20260601_193826"
    state.restart_count = 2
    state.crash_count = 1
    state.deadline = time.monotonic() + 3600  # ~1h to restart
    state.update_metrics({
        "_step": 6029312,
        "rollout/ep_len_mean": 33.33,
        "rollout/ep_rew_mean": -27.26,
        "time/fps": 1121.0,
        "time/total_timesteps": 6029312.0,
        "train/loss": 46.51,
        "train/value_loss": 95.61,
        "eval/win_rate_mean": 0.413,
        "eval/mean_reward_mean": -14.2,
        "eval/win_rate_vs_bots": 0.345,
        "eval/mean_reward_vs_bots": -19.1,
        "eval/win_rate_vs_pool": 0.52,
        "eval/mean_reward_vs_pool": 3.1,
        "eval/duration_sec": 233.0,
        "eval/n_workers": 5.0,   # took = duration_sec / n_workers (per-worker wall-clock)
        # snake_case opponent keys (as emitted post the eval refactor on main).
        "eval/win_rate_vs_random": 0.96,
        "eval/mean_reward_vs_random": 25.3,
        "eval/win_rate_vs_heuristic": 0.33,
        "eval/mean_reward_vs_heuristic": -17.8,
        "eval/win_rate_vs_setup_sweep_v2": 0.35,
        "eval/mean_reward_vs_setup_sweep_v2": -16.9,
        # pool sentinels (positional, newest→oldest) + their step tags
        "eval/win_rate_vs_sentinel_0": 0.47,
        "eval/mean_reward_vs_sentinel_0": -2.0,
        "eval/sentinel_step_0": 47_000_000,
        "eval/win_rate_vs_sentinel_1": 0.61,
        "eval/mean_reward_vs_sentinel_1": 5.5,
        "eval/sentinel_step_1": 0,  # the seed
    })
    state.add_event("🚀 Starting — restart every 3.0h")
    state.add_event("✅ Child PID 1686225 started")
    return state


# ── _wrap_hanging (event soft-wrap with hanging indent) ─────────────────────────

def test_wrap_hanging_indents_continuation_and_never_overflows():
    from rich.cells import cell_len
    # Emoji in the body: cell-aware wrapping must hold (🐴 is 2 cells, not 1).
    ev = ("[16:52:08] 🐴 [STABLE] 1 cross-run opponent(s): ext_ai_v5_5_popart_50m_0607 — "
          "eval greedy; training ≤20% of self-play until mastered (win_rate ≥ 80%)")
    lines = _wrap_hanging(ev, 90).split("\n")
    assert len(lines) > 1                                   # it actually wrapped
    assert all(cell_len(ln) <= 90 for ln in lines)         # no rendered line overflows
    assert all(ln.startswith(" " * _EVENT_HANG_INDENT) for ln in lines[1:])  # hanging indent
    assert lines[0] == ev[:len(lines[0])]                  # first line is the prefix, unindented


def test_wrap_hanging_noop_when_fitting_or_unlaid_out():
    ev = "[16:52:08] 🐴 [STABLE] short"
    assert _wrap_hanging(ev, 400) == ev    # already fits → unchanged (no inserted newlines)
    assert _wrap_hanging(ev, 0) == ev      # width 0 (before first layout) → unchanged
    assert _wrap_hanging(ev, 12) == ev     # too narrow to indent usefully → unchanged


# ── Pilot: rendering parity ────────────────────────────────────────────────────

async def test_skeleton_mounts_all_views():
    app = LauncherApp(LauncherState(interval_hours=3.0), queue.Queue())
    async with app.run_test() as pilot:
        await pilot.pause()
        for vid in ("dashboard", "logs", "events", "confirm_quit", "confirm_force_eval"):
            assert app.query_one(f"#{vid}") is not None
        assert app.query_one("#metrics-left", DataTable) is not None
        assert app.query_one("#metrics-train", DataTable) is not None
        assert app.query_one("#metrics-eval", DataTable) is not None
        assert app.query_one("#views", ContentSwitcher).current == "dashboard"


async def test_dashboard_renders_topline_and_badges():
    app = LauncherApp(_populated_state(), queue.Queue())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._refresh()
        await pilot.pause()
        top = _plain(app.query_one("#topline", Static))
        assert "PID 1686225" in top
        assert "run #3" in top                  # restart_count + 1
        assert "restart in" in top
        badges = _plain(app.query_one("#badges", Static))
        assert "24e6d7c4" in badges              # git hash[:8]
        assert "run_20260601_193826" in badges   # model dir basename
        assert "2 restarts (1 crash)" in badges  # restart badge w/ crash
        assert "6,029,312" in badges             # steps
        assert "ent_coef" not in badges          # ent_coef badge removed from the topline


async def test_dashboard_metrics_tables_populated():
    app = LauncherApp(_populated_state(), queue.Queue())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._refresh()
        await pilot.pause()
        left = _table_text(app.query_one("#metrics-left", DataTable))
        assert "rollout" in left and "time" in left
        assert "ep_rew_mean" in left and "-27.26" in left
        assert "1,121" in left                    # fps formatted with thousands sep
        assert "train" not in left                # train pulled into its own column
        # train/* is its own dedicated column now (the biggest section).
        train = _table_text(app.query_one("#metrics-train", DataTable))
        assert "train" in train and "value_loss" in train and "95.61" in train
        ev = _table_text(app.query_one("#metrics-eval", DataTable))
        assert "all" in ev and "vs Bots" in ev
        assert "vs random" in ev                  # snake_case opponent name
        assert "96.0%" in ev                      # win rate as percent
        assert "25.3" in ev                       # reward, fixed 1-decimal
        assert "took 46s" in ev                   # raw per-worker seconds: int(233.0 / 5)
        # vs Pool now shows BOTH win rate and reward (mirrors ui.py)
        assert "vs Pool" in ev and "52.0%" in ev and "3.1" in ev
        # pool sentinels labeled with their snapshot step (seed for step 0)
        assert "vs sentinel_0 (47.0M)" in ev
        assert "vs sentinel_1 (seed)" in ev


def test_secs_str_switches_to_minutes_past_600s():
    from main.launcher.format import _secs_str
    assert _secs_str(0) == "0s"
    assert _secs_str(94) == "94s"
    assert _secs_str(600) == "600s"          # boundary: still raw seconds
    assert _secs_str(601) == "10m1s"         # just past → XmYs
    assert _secs_str(1949) == "32m29s"
    assert _secs_str(660) == "11m0s"         # exact minute keeps the trailing 0s
    assert _secs_str(-5) == "0s"             # clamped


def _with_distill(state: LauncherState, *, all_distilled: float, frac: float,
                  running: float = 0.0, exhausted: float = 0.0) -> LauncherState:
    """Merge a distill/* metrics block onto an existing state (non-eval → merges)."""
    state.update_metrics({
        "_step": 6_100_000,
        "distill/all_distilled": all_distilled,
        "distill/frac_active_opponents_distilled": frac,
        "distill/n_ready": 5.0,
        "distill/n_running": running,
        "distill/n_exhausted": exhausted,
    })
    return state


async def test_distill_badge_green_when_all_distilled():
    state = _with_distill(_populated_state(), all_distilled=1.0, frac=1.0)
    app = LauncherApp(state, queue.Queue())
    async with app.run_test() as pilot:
        await pilot.pause(); app._refresh(); await pilot.pause()
        badges = _plain(app.query_one("#badges", Static))
        assert "distilled 100%" in badges            # speedup ACTIVE headline


async def test_distill_badge_yellow_while_backfilling():
    state = _with_distill(_populated_state(), all_distilled=0.0, frac=0.6,
                          running=2.0, exhausted=1.0)
    app = LauncherApp(state, queue.Queue())
    async with app.run_test() as pilot:
        await pilot.pause(); app._refresh(); await pilot.pause()
        badges = _plain(app.query_one("#badges", Static))
        assert "distilling 60%" in badges
        assert "2 running" in badges
        assert "1 exhausted" in badges


async def test_no_distill_badge_when_disabled():
    # _populated_state has no distill/* keys → zero footprint (badge absent).
    app = LauncherApp(_populated_state(), queue.Queue())
    async with app.run_test() as pilot:
        await pilot.pause(); app._refresh(); await pilot.pause()
        badges = _plain(app.query_one("#badges", Static))
        assert "distill" not in badges.lower()


async def test_distill_metrics_block_renders_legibly():
    state = _with_distill(_populated_state(), all_distilled=1.0, frac=1.0)
    app = LauncherApp(state, queue.Queue())
    async with app.run_test() as pilot:
        await pilot.pause(); app._refresh(); await pilot.pause()
        left = _table_text(app.query_one("#metrics-left", DataTable))
        assert "distill" in left                      # the section header
        assert "all distilled" in left and "yes" in left   # flag formatted, short label
        assert "distilled" in left and "100.0%" in left    # frac as a percent


async def test_metrics_tables_never_show_a_scrollbar():
    """The dashboard tables size to content and never scroll — even a full self-play +
    --distill-opponents roster (left misc + a dedicated train column + eval with 5 sentinels)
    must not trap the wheel with a scrollbar."""
    state = LauncherState(interval_hours=3.0)
    state.pid = 1
    metrics = {"_step": 76_888_064}
    # non-eval metrics: rollout/time + distill in the left column, the 12 train keys in their
    # own column → 22 rows across the two, the roster that used to overflow a single column
    metrics.update({"rollout/ep_len_mean": 34.3, "rollout/ep_rew_mean": 3.1,
                    "time/fps": 416.0, "time/total_timesteps": 76_888_064.0})
    for k in ("approx_kl", "clip_fraction", "clip_range", "entropy_loss", "explained_variance",
              "learning_rate", "loss", "n_updates", "policy_gradient_loss", "value_loss",
              "clip_fraction_vf", "clip_range_vf"):
        metrics[f"train/{k}"] = 0.5
    metrics.update({"distill/all_distilled": 0, "distill/frac_active_opponents_distilled": 0.5})
    # eval: aggregates + 9 bots + 5 sentinels
    metrics.update({"eval/win_rate_mean": 0.78, "eval/mean_reward_mean": 20.0,
                    "eval/win_rate_vs_bots": 0.76, "eval/mean_reward_vs_bots": 17.0,
                    "eval/win_rate_vs_pool": 0.75, "eval/mean_reward_vs_pool": 18.4})
    for b in ("random", "heuristic", "heuristic2", "staller", "staller_v2", "aggressive",
              "aggressive_v2", "setup_sweep", "setup_sweep_v2"):
        metrics[f"eval/win_rate_vs_{b}"] = 0.8
        metrics[f"eval/mean_reward_vs_{b}"] = 15.0
    for i in range(5):
        metrics[f"eval/win_rate_vs_sentinel_{i}"] = 0.75
        metrics[f"eval/mean_reward_vs_sentinel_{i}"] = 18.0
        metrics[f"eval/sentinel_step_{i}"] = (74 - 10 * i) * 1_000_000
    state.update_metrics(metrics)

    app = LauncherApp(state, queue.Queue())
    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        app._refresh()
        await pilot.pause()
        left = app.query_one("#metrics-left", DataTable)
        train = app.query_one("#metrics-train", DataTable)
        ev = app.query_one("#metrics-eval", DataTable)
        assert train.row_count >= 13                     # 12 train keys + the section header
        assert left.row_count + train.row_count >= 22    # the roster that used to overflow
        assert left.show_vertical_scrollbar is False     # no scrollbar to fight
        assert train.show_vertical_scrollbar is False
        assert ev.show_vertical_scrollbar is False


async def test_eval_cells_use_gradient_color():
    app = LauncherApp(_populated_state(), queue.Queue())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._refresh()
        await pilot.pause()
        ev = app.query_one("#metrics-eval", DataTable)
        styles = []
        for r in range(ev.row_count):
            for cell in ev.get_row_at(r):
                style = getattr(cell, "style", "")
                if style:
                    styles.append(str(style))
        # win-rate / reward cells carry a "bold #rrggbb" gradient style.
        assert any("#" in s for s in styles), styles


async def test_stale_indicator_appears_when_child_silent():
    state = LauncherState(interval_hours=3.0)
    state.pid = 1
    state.update_metrics({"_step": 100, "time/fps": 50.0})
    state._last_activity_ts = time.monotonic() - 600  # 10 min silent (set last)
    app = LauncherApp(state, queue.Queue())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._refresh()
        await pilot.pause()
        assert "no child output" in _plain(app.query_one("#badges", Static))


async def test_no_restart_badge_before_any_restart():
    state = LauncherState(interval_hours=0)  # single run, no restart
    state.pid = 7
    app = LauncherApp(state, queue.Queue())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._refresh()
        await pilot.pause()
        assert "↻" not in _plain(app.query_one("#badges", Static))
        assert "no restart" in _plain(app.query_one("#topline", Static))


# ── Pilot: input + view switching ───────────────────────────────────────────────

async def test_control_keys_forward_to_cmd_q():
    """Child-control keys (r/c/p/s) route to the supervisor's cmd_q; view-nav keys do NOT."""
    q = queue.Queue()
    app = LauncherApp(LauncherState(interval_hours=3.0), q)
    async with app.run_test() as pilot:
        await pilot.pause()
        for key in ["r", "c", "p", "s"]:
            await pilot.press(key)
        await pilot.pause()
        got = []
        while not q.empty():
            got.append(q.get_nowait())
        assert got == ["r", "c", "p", "s"]


async def test_view_nav_is_instant_and_not_routed():
    """View switching happens on the event loop immediately — no cmd_q round-trip, no
    _refresh, no waiting."""
    q = queue.Queue()
    app = LauncherApp(LauncherState(interval_hours=3.0), q)
    async with app.run_test() as pilot:
        await pilot.pause()
        cs = app.query_one("#views", ContentSwitcher)
        assert cs.current == "dashboard"

        await pilot.press("l")
        assert app.view_mode == "logs"
        assert cs.current == "logs"          # switched instantly (same frame)
        await pilot.press("e")
        assert cs.current == "events"
        await pilot.press("d")
        assert cs.current == "dashboard"
        assert q.empty()                     # none of these touched the supervisor


async def test_q_opens_confirm_locally_does_not_quit_or_route():
    """`q` overrides Gen3App's base `q → quit`: it shows the confirm overlay app-locally."""
    q = queue.Queue()
    app = LauncherApp(LauncherState(interval_hours=3.0), q)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("q")
        assert app.is_running                                # NOT quit
        assert app.view_mode == "confirm_quit"               # confirm shown instantly
        assert app.query_one("#views", ContentSwitcher).current == "confirm_quit"
        assert q.empty()                                     # nothing sent to the supervisor yet


async def test_super_c_copy_inherited_not_shadowed():
    """⌘C (super+c) is the base Gen3App copy binding — the launcher's fresh BINDINGS
    (incl. ctrl+c → quit-confirm) must not shadow it or route it to the supervisor."""
    q = queue.Queue()
    app = LauncherApp(LauncherState(interval_hours=3.0), q)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("super+c")
        await pilot.pause()
        assert app.is_running                                # NOT quit
        assert app.view_mode == "dashboard"                  # NOT the confirm overlay
        assert q.empty()                                     # NOT routed to the supervisor


async def test_copy_mode_pauses_refresh_timer_and_is_not_routed():
    """`v` (inherited from Gen3App) enters copy mode: it pauses the 2 Hz refresh timer so a
    repaint can't wipe a terminal selection, and is handled app-locally (not sent to cmd_q)."""
    q = queue.Queue()
    app = LauncherApp(LauncherState(interval_hours=3.0), q)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._refresh_timer is not None
        assert app._refresh_timer._active.is_set()    # ticking before copy mode

        await pilot.press("v")
        await pilot.pause()
        assert app.copy_mode is True
        assert not app._refresh_timer._active.is_set()  # frozen so the selection holds
        assert app.is_running
        assert q.empty()                                # app-local, never routed

        await pilot.press("v")
        await pilot.pause()
        assert app.copy_mode is False
        assert app._refresh_timer._active.is_set()      # ticking again
        assert q.empty()


async def test_confirm_cancel_and_confirm_paths():
    q = queue.Queue()
    app = LauncherApp(LauncherState(interval_hours=3.0), q)
    async with app.run_test() as pilot:
        await pilot.pause()
        # cancel with `n`
        await pilot.press("q")
        assert app.view_mode == "confirm_quit"
        await pilot.press("n")
        assert app.view_mode == "dashboard"
        assert q.empty()
        # confirm with `y` → sentinel to supervisor + back to dashboard
        await pilot.press("q")
        await pilot.press("y")
        assert app.view_mode == "dashboard"
        assert q.get_nowait() == "__quit__"
        assert q.empty()


async def test_force_eval_confirm_flow():
    """`f` opens the force-eval confirm overlay app-locally; `y` routes the `f` control char
    to the supervisor (→ SIGUSR2); `n` cancels without routing."""
    q = queue.Queue()
    app = LauncherApp(LauncherState(interval_hours=3.0), q)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f")
        assert app.is_running                                # not a control key yet — overlay only
        assert app.view_mode == "confirm_force_eval"
        assert app.query_one("#views", ContentSwitcher).current == "confirm_force_eval"
        assert q.empty()                                     # nothing routed until confirmed
        # cancel with `n`
        await pilot.press("n")
        assert app.view_mode == "dashboard"
        assert q.empty()
        # confirm with `y` → routes the "f" control char + back to dashboard
        await pilot.press("f")
        await pilot.press("y")
        assert app.view_mode == "dashboard"
        assert q.get_nowait() == "f"
        assert q.empty()


async def test_force_eval_does_not_interrupt_quit_confirm():
    """`f` is inert while the quit confirm is up — a pending quit must not be derailed."""
    q = queue.Queue()
    app = LauncherApp(LauncherState(interval_hours=3.0), q)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("q")
        assert app.view_mode == "confirm_quit"
        await pilot.press("f")
        assert app.view_mode == "confirm_quit"               # NOT switched to force-eval
        assert q.empty()


async def test_control_keys_modal_during_force_eval_confirm():
    """The force-eval confirm overlay is modal: a stray control key (r/c/p/s) must not route
    to the supervisor mid-decision (parity with the quit confirm)."""
    q = queue.Queue()
    app = LauncherApp(LauncherState(interval_hours=3.0), q)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("f")
        assert app.view_mode == "confirm_force_eval"
        for key in ("r", "c", "p", "s"):
            await pilot.press(key)
        assert app.view_mode == "confirm_force_eval"         # still on the overlay
        assert q.empty()                                     # nothing routed


async def test_ctrl_c_first_confirms_second_quits():
    """Ctrl-C: first press opens the confirm overlay, a second confirms (clean save path)."""
    q = queue.Queue()
    app = LauncherApp(LauncherState(interval_hours=3.0), q)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+c")
        assert app.is_running                       # did NOT die
        assert app.view_mode == "confirm_quit"      # overlay shown
        assert q.empty()                            # nothing sent yet
        await pilot.press("ctrl+c")                 # second press confirms
        assert app.view_mode == "dashboard"
        assert q.get_nowait() == "__quit__"


async def test_shutdown_signal_routes_to_clean_quit():
    """SIGHUP/SIGTERM handler pushes the clean-quit sentinel + surfaces an event."""
    q = queue.Queue()
    state = LauncherState(interval_hours=3.0)
    app = LauncherApp(state, q)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._on_shutdown_signal("SIGHUP")
        assert q.get_nowait() == "__quit__"
        assert any("SIGHUP" in e for e in state.snapshot().events)


async def test_waiting_message_when_no_rollout_metrics_yet():
    """Before the first rollout (eval present, no rollout/time/train), show a clean line."""
    state = LauncherState(interval_hours=3.0)
    state.pid = 1
    state.update_metrics({"_step": 0, "eval/win_rate_mean": 0.5})  # eval only, no rollout
    app = LauncherApp(state, queue.Queue())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._refresh()
        await pilot.pause()
        left = _table_text(app.query_one("#metrics-left", DataTable))
        assert "waiting for first rollout" in left
        assert "—" not in left  # no stray dash row


async def test_control_keys_inert_while_confirming():
    """The confirm overlay is modal: r/c/p/s are ignored until it's dismissed."""
    q = queue.Queue()
    app = LauncherApp(LauncherState(interval_hours=3.0), q)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("q")               # open confirm
        await pilot.press("r")               # should be ignored
        assert q.empty()
        await pilot.press("n")               # dismiss
        await pilot.press("r")               # now it routes
        assert q.get_nowait() == "r"


async def test_render_fault_does_not_crash_app():
    """A bad metric value (non-numeric) breaks numeric formatting; the render guard must
    keep the app alive so run.py's child reap never fires over a cosmetic bug."""
    state = LauncherState(interval_hours=3.0)
    state.pid = 1
    state.update_metrics({"_step": 100, "train/loss": "not-a-number"})  # _fmt_val would raise
    app = LauncherApp(state, queue.Queue())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._refresh()
        await pilot.pause()
        assert app.is_running                # guard caught it
        assert app._render_error_seen        # and surfaced it once
        assert any("render error" in e for e in state.snapshot().events)


async def test_log_lines_colored_by_level():
    state = LauncherState(interval_hours=3.0)
    state.add_log("🛑 ERROR: boom")
    state.add_log("normal progress line")
    app = LauncherApp(state, queue.Queue())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.view_mode = "logs"
        app._refresh()
        await pilot.pause()
        body = app.query_one("#logs-body", Static).content  # rich Text
        # the error line carries a red style span; the plain text is present too.
        assert "boom" in str(body)
        styles = {str(span.style) for span in getattr(body, "spans", [])}
        assert any("red" in s for s in styles), styles


# ── Loop: _supervise exit-code contract (no real child) ──────────────────────────

def _make_proc(returncode: int, pid: int = 99) -> MagicMock:
    proc = MagicMock()
    proc.pid = pid
    proc.returncode = returncode
    proc.stdout = io.BytesIO(b"")
    proc.wait.return_value = None  # proc.wait(timeout=…) returns → loop treats as exited
    return proc


def _ctx(tmp_path, *, interval_hours: float = 0.0) -> _SessionCtx:
    run_dir = str(tmp_path / "run")
    return _SessionCtx(
        child_args=["--debug"],
        child_env={},
        train_script="train.py",
        src_dir="/src",
        run_dir=run_dir,
        run_dir_box=[run_dir],
        session_start=time.time(),
        interval_seconds=interval_hours * 3600,
        grace_seconds=1200.0,
        pin=False,
    )


def _supervise_patches(procs, checkpoint=None):
    """Mock the child spawn + checkpoint discovery so the loop runs without a real child."""
    stack = ExitStack()
    stack.enter_context(patch("main.launcher.child.subprocess.Popen", side_effect=list(procs)))
    stack.enter_context(patch("main.launcher.child.threading.Thread"))
    stack.enter_context(patch("main.launcher.child.os.pipe", return_value=(3, 4)))
    stack.enter_context(patch("main.launcher.child.os.close"))
    stack.enter_context(patch("main.launcher.run._git_hash", return_value="abc1234"))
    stack.enter_context(patch("main.launcher.run.find_latest_checkpoint", return_value=checkpoint))
    stack.enter_context(patch("main.launcher.run._save_crash_log", return_value=None))
    stack.enter_context(patch("main.launcher.run.time.sleep"))
    # The give-up paths register an at-exit crash-log dump; suppress it so a test that
    # injects log lines doesn't leak the child output to the test runner's stderr.
    stack.enter_context(patch("main.launcher.run.atexit.register"))
    return stack


def _supervise_once(ctx, procs, checkpoint=None, *, interval_hours=0.0, max_crash_restarts=3):
    with _supervise_patches(procs, checkpoint=checkpoint):
        return _supervise(
            LauncherState(interval_hours=interval_hours),
            queue.Queue(),
            ctx,
            interval_hours=interval_hours,
            grace_minutes=20.0,
            max_crash_restarts=max_crash_restarts,
        )


class TestSuperviseExitCodes:
    def test_complete_returns_zero(self, tmp_path):
        assert _supervise_once(_ctx(tmp_path), [_make_proc(TrainExitCode.COMPLETE)]) == 0

    def test_interrupted_no_interval_returns_zero(self, tmp_path):
        assert _supervise_once(_ctx(tmp_path), [_make_proc(TrainExitCode.INTERRUPTED)]) == 0

    def test_crash_no_checkpoint_returns_returncode(self, tmp_path):
        assert _supervise_once(_ctx(tmp_path), [_make_proc(TrainExitCode.CRASH)], checkpoint=None) == TrainExitCode.CRASH

    def test_unknown_exit_code_treated_as_crash(self, tmp_path):
        assert _supervise_once(_ctx(tmp_path), [_make_proc(42)], checkpoint=None) == 42


class TestSuperviseCrashRestart:
    def test_crash_auto_restarts_then_completes(self, tmp_path):
        crash = _make_proc(TrainExitCode.CRASH)
        done = _make_proc(TrainExitCode.COMPLETE)
        ckpt = tmp_path / "checkpoint_1000_steps.zip"
        ckpt.write_text("x")
        code = _supervise_once(_ctx(tmp_path), [crash, done], checkpoint=str(ckpt), max_crash_restarts=5)
        assert code == 0
        assert done.wait.called  # the post-crash child really launched

    def test_circuit_breaker_gives_up_after_rapid_crashes(self, tmp_path):
        c1 = _make_proc(TrainExitCode.CRASH)
        c2 = _make_proc(TrainExitCode.CRASH)
        ckpt = tmp_path / "checkpoint_1000_steps.zip"
        ckpt.write_text("x")
        code = _supervise_once(_ctx(tmp_path), [c1, c2], checkpoint=str(ckpt), max_crash_restarts=2)
        assert code == TrainExitCode.CRASH
        assert c2.wait.called  # restarted once before giving up


class TestSuperviseFatalConfig:
    """A non-recoverable config/arch error (ModelVersionError → FATAL_CONFIG, or a
    ``[ModelVersion] FATAL`` signature in the captured output) must NOT auto-restart —
    restarting would hit the identical error. The launcher gives up immediately, even
    when a checkpoint is available (a normal crash WOULD restart from it)."""

    def test_fatal_config_exit_code_does_not_restart(self, tmp_path):
        ckpt = tmp_path / "checkpoint_1000_steps.zip"
        ckpt.write_text("x")
        fatal = _make_proc(TrainExitCode.FATAL_CONFIG)
        nxt = _make_proc(TrainExitCode.COMPLETE)  # must never be reached
        code = _supervise_once(
            _ctx(tmp_path), [fatal, nxt], checkpoint=str(ckpt), max_crash_restarts=5
        )
        assert code == TrainExitCode.FATAL_CONFIG
        assert not nxt.wait.called  # gave up — no second child launched

    def test_fatal_signature_in_log_does_not_restart(self, tmp_path):
        ckpt = tmp_path / "checkpoint_1000_steps.zip"
        ckpt.write_text("x")
        # A generic crash exit (1), but the captured output carries the FATAL signature —
        # the defensive fallback should still classify it as non-recoverable.
        crash = _make_proc(TrainExitCode.CRASH)
        nxt = _make_proc(TrainExitCode.COMPLETE)
        state = LauncherState(interval_hours=0)
        state.add_log("[ModelVersion] FATAL: Architecture family mismatch: saved='a', current='b'.")
        with _supervise_patches([crash, nxt], checkpoint=str(ckpt)):
            code = _supervise(
                state, queue.Queue(), _ctx(tmp_path),
                interval_hours=0, grace_minutes=20.0, max_crash_restarts=5,
            )
        assert code == TrainExitCode.CRASH
        assert not nxt.wait.called

    def test_ordinary_crash_still_restarts(self, tmp_path):
        """Guard: a plain crash with no fatal signal keeps the existing auto-restart."""
        ckpt = tmp_path / "checkpoint_1000_steps.zip"
        ckpt.write_text("x")
        crash = _make_proc(TrainExitCode.CRASH)
        done = _make_proc(TrainExitCode.COMPLETE)
        code = _supervise_once(
            _ctx(tmp_path), [crash, done], checkpoint=str(ckpt), max_crash_restarts=5
        )
        assert code == 0
        assert done.wait.called


class TestFatalConfigReason:
    """Unit-test the pure classifier directly."""

    def test_exit_code_alone_is_sufficient(self):
        assert _fatal_config_reason(int(TrainExitCode.FATAL_CONFIG), []) is not None

    def test_signature_alone_is_sufficient(self):
        reason = _fatal_config_reason(
            int(TrainExitCode.CRASH),
            ["progress chatter", "[ModelVersion] FATAL: vf_coef mismatch", "Fix: resume with ..."],
        )
        assert reason is not None
        assert reason[0].startswith("[ModelVersion] FATAL")
        assert "Fix:" in reason[-1]  # grabs following explanatory lines too

    def test_ordinary_crash_is_none(self):
        assert _fatal_config_reason(int(TrainExitCode.CRASH), ["normal traceback line"]) is None
        assert _fatal_config_reason(int(TrainExitCode.INTERRUPTED), None) is None


class TestSuperviseConfirmQuit:
    """A confirmed quit (the app sends the "__quit__" sentinel) SIGTERMs the child and exits 0.

    This is the path the live smoke didn't hit (its child completed on its own), so cover it
    deterministically. The app owns the confirm overlay locally and only forwards "__quit__"
    once the user pressed y."""

    def test_quit_sentinel_sigterms_child_and_exits_zero(self, tmp_path):
        proc = _make_proc(TrainExitCode.INTERRUPTED)
        # Stay alive one poll (so cmd_q is drained), then report exited.
        proc.wait.side_effect = [subprocess.TimeoutExpired("c", 0.5), None, None, None]
        q = queue.Queue()
        q.put("__quit__")  # what LauncherApp.action_confirm_yes sends
        state = LauncherState(interval_hours=0)
        with _supervise_patches([proc]), patch("main.launcher.run.os.kill") as kill:
            code = _supervise(
                state, q, _ctx(tmp_path),
                interval_hours=0, grace_minutes=20.0, max_crash_restarts=3,
            )
        assert code == 0
        assert any(c.args[1] == signal.SIGTERM for c in kill.call_args_list), kill.call_args_list


# ── Loop: teardown safety net ────────────────────────────────────────────────────

def test_launch_child_stays_in_session():
    """Child is spawned WITHOUT start_new_session so a terminal SIGHUP reaches it and its
    own handler (train_rl_agent) checkpoints; the launcher also handles SIGHUP for a clean
    teardown. (Regression guard against re-detaching, which would bypass the child handler.)"""
    from main.launcher.child import _launch_child
    st = LauncherState(interval_hours=0)
    st.run_dir = None  # _open_child_log → None, no file written
    proc = MagicMock()
    proc.pid = 1
    proc.stdout = io.BytesIO(b"")
    with patch("main.launcher.child.subprocess.Popen", return_value=proc) as popen, \
         patch("main.launcher.child.threading.Thread"), \
         patch("main.launcher.child.os.pipe", return_value=(3, 4)), \
         patch("main.launcher.child.os.close"):
        _launch_child(["--debug"], {}, st, "train.py", "/src")
    assert not popen.call_args.kwargs.get("start_new_session")


class TestReap:
    def test_reap_empty_and_none_are_noops(self):
        _reap([])
        _reap(None)

    def test_reap_skips_dead_child(self):
        proc = MagicMock()
        proc.poll.return_value = 0  # already exited
        with patch("main.launcher.run.os.kill") as k:
            _reap([proc])
        k.assert_not_called()

    def test_reap_sigterms_live_child(self):
        proc = MagicMock()
        proc.pid = 123
        proc.poll.return_value = None  # alive
        proc.wait.return_value = 0     # dies on SIGTERM within grace
        with patch("main.launcher.run.os.kill") as k:
            _reap([proc])
        k.assert_called_once_with(123, signal.SIGTERM)

    def test_reap_escalates_to_sigkill_on_timeout(self):
        proc = MagicMock()
        proc.pid = 5
        proc.poll.return_value = None
        # all 10 one-second grace polls time out → SIGKILL, then a final wait
        proc.wait.side_effect = [subprocess.TimeoutExpired("c", 1)] * 10 + [0]
        with patch("main.launcher.run.os.kill") as k:
            _reap([proc])
        assert k.call_args_list == [call(5, signal.SIGTERM), call(5, signal.SIGKILL)]
