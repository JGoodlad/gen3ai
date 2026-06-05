import io
import json
import os
import signal
import subprocess
import tempfile
import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from main.exit_codes import TrainExitCode
from main.launcher import (
    _TRAIN_SCRIPT,
    _SRC_DIR,
    _PollFlags,
    _dispatch_command,
    _find_model_arg,
    _insert_or_replace_model_arg,
    _insert_or_replace_run_dir_arg,
    _read_checkpoint_git_hash,
    _read_metrics_pipe,
    _strip_launcher_args,
    find_latest_checkpoint,
)
from main.launcher import LauncherState
from main.launcher.run import _find_ent_coef
from main.launcher.worktree import _read_checkpoint_lr


# ── find_latest_checkpoint ───────────────────────────────────────────────────

class TestFindLatestCheckpoint:
    def test_empty_dir_returns_none(self, tmp_path):
        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_single_zip_returned(self, tmp_path):
        p = tmp_path / "checkpoint_1000_steps.zip"
        p.write_text("x")
        assert find_latest_checkpoint(str(tmp_path)) == str(p)

    def test_falls_back_to_step_sort_without_latest_txt(self, tmp_path):
        low = tmp_path / "checkpoint_1000_steps.zip"
        low.write_text("x")
        time.sleep(0.01)
        high = tmp_path / "checkpoint_2000_steps.zip"
        high.write_text("x")
        # Flip mtimes so mtime-only sort would pick 'low'
        now = time.time()
        os.utime(str(high), (now - 10, now - 10))
        os.utime(str(low), (now, now))
        assert find_latest_checkpoint(str(tmp_path)) == str(high)

    def test_ignores_non_zip(self, tmp_path):
        (tmp_path / "model.pt").write_text("x")
        (tmp_path / "notes.txt").write_text("x")
        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_searches_subdirectories(self, tmp_path):
        sub = tmp_path / "run_20240101"
        sub.mkdir()
        deep = sub / "checkpoint_5000_steps.zip"
        deep.write_text("x")
        assert find_latest_checkpoint(str(tmp_path)) == str(deep)

    def test_forced_checkpoint_step_sort(self, tmp_path):
        low = tmp_path / "checkpoint_1000_steps.zip"
        low.write_text("x")
        high = tmp_path / "checkpoint_forced_0000005000_143022.zip"
        high.write_text("x")
        assert find_latest_checkpoint(str(tmp_path)) == str(high)

    def test_step_zero_files_fall_back_to_mtime(self, tmp_path):
        old = tmp_path / "final_model.zip"
        old.write_text("x")
        time.sleep(0.01)
        new = tmp_path / "final_model_interrupted.zip"
        new.write_text("x")
        assert find_latest_checkpoint(str(tmp_path)) == str(new)

    def test_reads_latest_txt_when_run_dir_given(self, tmp_path):
        run = tmp_path / "run_a"
        run.mkdir()
        low = run / "checkpoint_1000_steps.zip"
        low.write_text("x")
        high = run / "checkpoint_2000_steps.zip"
        high.write_text("x")
        (run / "latest.txt").write_text("checkpoint_1000_steps.zip\n")
        # latest.txt points to the lower-step file — should trust it
        assert find_latest_checkpoint(str(tmp_path), run_dir=str(run)) == str(low)

    def test_latest_txt_missing_file_falls_back_to_glob(self, tmp_path):
        run = tmp_path / "run_a"
        run.mkdir()
        good = run / "checkpoint_2000_steps.zip"
        good.write_text("x")
        (run / "latest.txt").write_text("ghost.zip\n")  # points to nonexistent file
        assert find_latest_checkpoint(str(tmp_path), run_dir=str(run)) == str(good)

    def test_min_mtime_filters_old_checkpoints(self, tmp_path):
        old = tmp_path / "checkpoint_9000_steps.zip"
        old.write_text("x")
        now = time.time()
        # backdate the old file so it predates the cutoff
        os.utime(str(old), (now - 100, now - 100))
        new = tmp_path / "checkpoint_100_steps.zip"
        new.write_text("x")
        # new has current mtime; old has higher step count but is too old
        result = find_latest_checkpoint(str(tmp_path), min_mtime=now - 10)
        assert result == str(new)

    def test_min_mtime_returns_none_when_all_filtered(self, tmp_path):
        old = tmp_path / "checkpoint_9000_steps.zip"
        old.write_text("x")
        now = time.time()
        os.utime(str(old), (now - 100, now - 100))
        assert find_latest_checkpoint(str(tmp_path), min_mtime=now - 10) is None

    def test_ignores_artifact_zips_when_no_real_checkpoint(self, tmp_path):
        # A crash at startup, before any real checkpoint, leaves only non-resumable
        # artifact zips: the self-play pool seed, the best-by-eval export, retained
        # eval snapshots. find_latest_checkpoint must return None so the launcher
        # surfaces the crash as the fatal no-checkpoint case — NOT "resume" from the
        # step-0 seed (which mis-derived run_dir to .../snapshots in the TUI badge).
        run = tmp_path / "run_x"
        (run / "snapshots").mkdir(parents=True)
        (run / "snapshots" / "snapshot_000000000000.zip").write_text("x")
        (run / "best_model").mkdir()
        (run / "best_model" / "best_model.zip").write_text("x")
        (run / "eval_traces" / "step_10").mkdir(parents=True)
        (run / "eval_traces" / "step_10" / "snapshot.zip").write_text("x")
        assert find_latest_checkpoint(str(tmp_path)) is None
        assert find_latest_checkpoint(str(tmp_path), run_dir=str(run)) is None

    def test_real_checkpoint_not_shadowed_by_newer_artifact(self, tmp_path):
        # The resumable checkpoint in the run dir wins even when a pool snapshot with a
        # newer mtime sits in snapshots/ — the artifact must never shadow it.
        run = tmp_path / "run_x"
        ckpt = run / "checkpoint_1000_steps.zip"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_text("x")
        time.sleep(0.01)
        snap = run / "snapshots" / "snapshot_000002000000.zip"
        snap.parent.mkdir()
        snap.write_text("x")  # newer mtime than the real checkpoint
        assert find_latest_checkpoint(str(tmp_path)) == str(ckpt)


# ── _insert_or_replace_model_arg ─────────────────────────────────────────────

class TestInsertOrReplaceModelArg:
    def test_inserts_when_absent(self):
        result = _insert_or_replace_model_arg(["--debug", "--steps", "5000"], "models/ckpt.zip")
        assert result == ["--debug", "--steps", "5000", "--model", "models/ckpt.zip"]

    def test_replaces_existing(self):
        result = _insert_or_replace_model_arg(
            ["--model", "old.zip", "--steps", "5000"], "new.zip"
        )
        assert result == ["--model", "new.zip", "--steps", "5000"]

    def test_replaces_equals_form_without_duplicating(self):
        result = _insert_or_replace_model_arg(
            ["--model=old.zip", "--steps", "5000"], "new.zip"
        )
        assert result == ["--model", "new.zip", "--steps", "5000"]

    def test_empty_args(self):
        result = _insert_or_replace_model_arg([], "ckpt.zip")
        assert result == ["--model", "ckpt.zip"]


# ── _insert_or_replace_run_dir_arg ───────────────────────────────────────────

class TestInsertOrReplaceRunDirArg:
    def test_inserts_when_absent(self):
        result = _insert_or_replace_run_dir_arg(["--debug"], "/models/run_x")
        assert result == ["--debug", "--run-dir", "/models/run_x"]

    def test_replaces_existing(self):
        result = _insert_or_replace_run_dir_arg(["--run-dir", "/old", "--debug"], "/new")
        assert result == ["--run-dir", "/new", "--debug"]

    def test_empty_args(self):
        result = _insert_or_replace_run_dir_arg([], "/models/run_x")
        assert result == ["--run-dir", "/models/run_x"]


# ── _find_model_arg ──────────────────────────────────────────────────────────

class TestFindModelArg:
    def test_finds_model(self):
        assert _find_model_arg(["--debug", "--model", "ckpt.zip"]) == "ckpt.zip"

    def test_finds_model_equals_form(self):
        assert _find_model_arg(["--debug", "--model=ckpt.zip"]) == "ckpt.zip"

    def test_returns_none_when_absent(self):
        assert _find_model_arg(["--debug", "--steps", "5000"]) is None

    def test_returns_none_on_empty(self):
        assert _find_model_arg([]) is None


# ── _find_ent_coef ───────────────────────────────────────────────────────────

class TestFindEntCoef:
    def test_finds_space_form(self):
        assert _find_ent_coef(["--ent-coef", "0.058"]) == 0.058

    def test_finds_equals_form(self):
        assert _find_ent_coef(["--ent-coef=0.058"]) == 0.058

    def test_returns_none_when_absent(self):
        assert _find_ent_coef(["--debug"]) is None

    def test_returns_none_on_bad_value(self):
        assert _find_ent_coef(["--ent-coef", "notafloat"]) is None


# ── _read_checkpoint_git_hash ─────────────────────────────────────────────────

class TestReadCheckpointGitHash:
    def test_reads_hash_from_metadata(self, tmp_path):
        (tmp_path / "metadata.json").write_text(json.dumps({"git_hash": "abc123full"}))
        result = _read_checkpoint_git_hash(str(tmp_path / "model.zip"))
        assert result == "abc123full"

    def test_returns_none_when_no_metadata(self, tmp_path):
        result = _read_checkpoint_git_hash(str(tmp_path / "model.zip"))
        assert result is None

    def test_returns_none_when_key_missing(self, tmp_path):
        (tmp_path / "metadata.json").write_text(json.dumps({"saved_at": "2026-01-01"}))
        result = _read_checkpoint_git_hash(str(tmp_path / "model.zip"))
        assert result is None

    def test_sidecar_wins_over_metadata(self, tmp_path):
        # Sidecar is written inside the pinned worktree at save time → authoritative.
        (tmp_path / "checkpoint_500_steps.json").write_text(json.dumps({"git_hash": "sidecar"}))
        (tmp_path / "metadata.json").write_text(json.dumps({"git_hash": "toplevel"}))
        result = _read_checkpoint_git_hash(str(tmp_path / "checkpoint_500_steps.zip"))
        assert result == "sidecar"

    def test_snapshot_history_wins_over_toplevel(self, tmp_path):
        # No sidecar: this checkpoint's history entry beats the top-level git_hash,
        # which reflects the LATEST save, not this (older) checkpoint.
        (tmp_path / "metadata.json").write_text(json.dumps({
            "git_hash": "latest_save",
            "snapshot_history": {
                "checkpoint_500_steps.zip": {"git_hash": "this_ckpt"},
            },
        }))
        result = _read_checkpoint_git_hash(str(tmp_path / "checkpoint_500_steps.zip"))
        assert result == "this_ckpt"

    def test_falls_back_to_toplevel_when_not_in_history(self, tmp_path):
        (tmp_path / "metadata.json").write_text(json.dumps({
            "git_hash": "latest_save",
            "snapshot_history": {"checkpoint_999_steps.zip": {"git_hash": "other"}},
        }))
        result = _read_checkpoint_git_hash(str(tmp_path / "checkpoint_500_steps.zip"))
        assert result == "latest_save"


# ── _read_checkpoint_lr ───────────────────────────────────────────────────────

class TestReadCheckpointLr:
    def test_reads_lr_from_snapshot_history(self, tmp_path):
        # History stores the per-checkpoint value under `lr`; top-level current_lr
        # is the latest save and must NOT shadow an older resumed checkpoint.
        (tmp_path / "metadata.json").write_text(json.dumps({
            "current_lr": 9.9e-4,
            "snapshot_history": {"checkpoint_500_steps.zip": {"lr": 2.5e-5}},
        }))
        result = _read_checkpoint_lr(str(tmp_path / "checkpoint_500_steps.zip"))
        assert result == pytest.approx(2.5e-5)

    def test_sidecar_lr_wins(self, tmp_path):
        (tmp_path / "checkpoint_500_steps.json").write_text(json.dumps({"lr": 1e-5}))
        (tmp_path / "metadata.json").write_text(json.dumps({"current_lr": 9.9e-4}))
        result = _read_checkpoint_lr(str(tmp_path / "checkpoint_500_steps.zip"))
        assert result == pytest.approx(1e-5)

    def test_falls_back_to_toplevel_current_lr(self, tmp_path):
        (tmp_path / "metadata.json").write_text(json.dumps({"current_lr": 9.9e-4}))
        result = _read_checkpoint_lr(str(tmp_path / "checkpoint_500_steps.zip"))
        assert result == pytest.approx(9.9e-4)


# ── _strip_launcher_args ──────────────────────────────────────────────────────

class TestStripLauncherArgs:
    def test_strips_separate_flag_and_value(self):
        args = ["--steps", "5000", "--restart-interval-hours", "3.0", "--debug"]
        assert _strip_launcher_args(args) == ["--steps", "5000", "--debug"]

    def test_strips_equals_form(self):
        args = ["--restart-interval-hours=2.5", "--debug"]
        assert _strip_launcher_args(args) == ["--debug"]

    def test_strips_no_pin(self):
        args = ["--no-pin", "--steps", "5000"]
        assert _strip_launcher_args(args) == ["--steps", "5000"]

    def test_no_op_when_absent(self):
        args = ["--steps", "5000", "--debug"]
        assert _strip_launcher_args(args) == args

    def test_empty(self):
        assert _strip_launcher_args([]) == []


# ── _read_metrics_pipe ───────────────────────────────────────────────────────

class TestReadMetricsPipe:
    def _run_reader(self, lines: list[str]) -> LauncherState:
        """Feed lines into a pipe, run the reader thread, return final state."""
        state = LauncherState(interval_hours=3.0)
        r_fd, w_fd = os.pipe()
        with os.fdopen(w_fd, "w") as w:
            for line in lines:
                w.write(line + "\n")
        # r_fd now has all data + EOF (w is closed)
        t = threading.Thread(target=_read_metrics_pipe, args=(r_fd, state))
        t.start()
        t.join(timeout=2.0)
        return state

    def test_valid_json_updates_state(self):
        state = self._run_reader([
            json.dumps({"rollout/ep_rew_mean": -5.0, "_step": 10000})
        ])
        snap = state.snapshot()
        assert snap.metrics.get("rollout/ep_rew_mean") == pytest.approx(-5.0)
        assert snap.metrics_step == 10000

    def test_multiple_lines_last_wins(self):
        state = self._run_reader([
            json.dumps({"time/fps": 1000.0, "_step": 1}),
            json.dumps({"time/fps": 1500.0, "_step": 2}),
        ])
        snap = state.snapshot()
        assert snap.metrics.get("time/fps") == pytest.approx(1500.0)

    def test_malformed_json_skipped(self):
        state = self._run_reader([
            "not json at all",
            json.dumps({"time/fps": 999.0, "_step": 1}),
        ])
        snap = state.snapshot()
        assert snap.metrics.get("time/fps") == pytest.approx(999.0)

    def test_empty_lines_ignored(self):
        state = self._run_reader(["", "   ", json.dumps({"x": 1.0, "_step": 0})])
        snap = state.snapshot()
        assert "x" in snap.metrics

    def test_eof_exits_cleanly(self):
        state = self._run_reader([])
        # Thread exits without hanging — test passes if join() doesn't timeout
        assert state.snapshot().metrics == {}

    def test_event_payload_adds_event_not_metrics(self):
        state = self._run_reader([
            json.dumps({"_event": "▶️  Resuming at LR 1.00e-04 (checkpoint=1.00e-04)"})
        ])
        snap = state.snapshot()
        assert any("Resuming" in e for e in snap.events)
        assert "_event" not in snap.metrics

    def test_event_payload_does_not_update_metrics(self):
        state = self._run_reader([
            json.dumps({"_event": "some message", "time/fps": 999.0})
        ])
        snap = state.snapshot()
        # Routed as an event — metrics not touched
        assert snap.metrics == {}

    def test_mixed_event_and_metrics(self):
        state = self._run_reader([
            json.dumps({"time/fps": 800.0, "_step": 1000}),
            json.dumps({"_event": "child resumed"}),
            json.dumps({"time/fps": 850.0, "_step": 2000}),
        ])
        snap = state.snapshot()
        assert snap.metrics.get("time/fps") == pytest.approx(850.0)
        assert snap.metrics_step == 2000
        assert any("child resumed" in e for e in snap.events)


# ── _dispatch_command ────────────────────────────────────────────────────────

class TestDispatchCommand:
    def _setup(self):
        proc = MagicMock()
        proc.pid = 12345
        state = LauncherState(interval_hours=3.0)
        flags = _PollFlags()
        deadline = time.monotonic() + 10800
        return proc, state, flags, deadline

    def _dispatch(self, ch, proc=None, state=None, flags=None):
        if proc is None:
            proc, state, flags, deadline = self._setup()
        else:
            deadline = time.monotonic() + 10800
        sent = []
        with patch("main.launcher.input.os.kill", side_effect=lambda pid, sig: sent.append(sig)):
            _dispatch_command(ch, proc, state, flags, deadline, interval_hours=3.0)
        return proc, state, flags, sent

    def test_r_sends_sigterm_and_sets_restart(self):
        _, state, flags, sent = self._dispatch("r")
        assert signal.SIGTERM in sent
        assert flags.restart_requested
        assert flags.sigterm_sent

    def test_r_does_not_double_sigterm(self):
        proc, state, flags, deadline = self._setup()
        flags.sigterm_sent = True
        sent = []
        with patch("main.launcher.input.os.kill", side_effect=lambda pid, sig: sent.append(sig)):
            _dispatch_command("r", proc, state, flags, deadline, interval_hours=3.0)
        assert signal.SIGTERM not in sent

    def test_c_sends_sigusr1(self):
        _, _, _, sent = self._dispatch("c")
        assert signal.SIGUSR1 in sent

    def test_c_handles_dead_child(self):
        proc = MagicMock()
        proc.pid = 99999
        state = LauncherState(interval_hours=3.0)
        flags = _PollFlags()
        with patch("main.launcher.input.os.kill", side_effect=ProcessLookupError):
            _dispatch_command("c", proc, state, flags, float("inf"), 3.0)
        snap = state.snapshot()
        assert any("already exited" in e for e in snap.events)

    def test_q_from_dashboard_enters_confirm_quit(self):
        _, state, flags, sent = self._dispatch("q")
        assert state.view_mode == "confirm_quit"
        assert not flags.quit_requested
        assert not flags.sigterm_sent
        assert signal.SIGTERM not in sent

    def test_q_from_logs_is_noop(self):
        proc, state, flags, deadline = self._setup()
        state.view_mode = "logs"
        sent = []
        with patch("main.launcher.input.os.kill", side_effect=lambda pid, sig: sent.append(sig)):
            _dispatch_command("q", proc, state, flags, deadline, 3.0)
        assert state.view_mode == "logs"
        assert not flags.sigterm_sent
        assert not sent

    def test_l_switches_to_log_view(self):
        _, state, flags, _ = self._dispatch("l")
        assert state.view_mode == "logs"

    def test_d_switches_to_dashboard(self):
        proc, state, flags, deadline = self._setup()
        state.view_mode = "logs"
        with patch("main.launcher.input.os.kill"):
            _dispatch_command("d", proc, state, flags, deadline, 3.0)
        assert state.view_mode == "dashboard"

    def test_s_adds_status_event(self):
        _, state, _, _ = self._dispatch("s")
        snap = state.snapshot()
        assert any("PID" in e for e in snap.events)

    def test_unknown_key_is_ignored(self):
        proc, state, flags, deadline = self._setup()
        sent = []
        with patch("main.launcher.input.os.kill", side_effect=lambda pid, sig: sent.append(sig)):
            _dispatch_command("z", proc, state, flags, deadline, 3.0)
        assert not sent
        assert not flags.sigterm_sent
        assert not flags.quit_requested


class TestCrashLogSnapshot:
    """_save_crash_log persists each crash's traceback to a unique file."""

    def test_save_crash_log_writes_unique_traceback_file(self, tmp_path):
        from main.launcher.run import _save_crash_log
        state = LauncherState(interval_hours=0)
        state.run_dir = str(tmp_path)
        state.add_log("Traceback (most recent call last):")
        state.add_log("RuntimeError: boom")
        p1 = _save_crash_log(str(tmp_path), state, 1)
        p2 = _save_crash_log(str(tmp_path), state, 1)
        assert p1 and p2 and p1 != p2                       # unique token per crash
        assert os.path.basename(p1).startswith("restart_err_") and p1.endswith(".txt")
        assert os.path.basename(os.path.dirname(p1)) == "crashes"   # folded under crashes/
        txt = open(p1).read()
        assert "RuntimeError: boom" in txt                  # full traceback captured
        assert "exit code: 1" in txt                        # header carries the exit code

    def test_save_crash_log_noop_without_run_dir(self):
        from main.launcher.run import _save_crash_log
        state = LauncherState(interval_hours=0)
        assert _save_crash_log(None, state, 1) is None


# ── Child-log persistence + deep scrollback ──────────────────────────────────
class TestChildLogPersistence:
    def test_deep_scrollback_retains_a_full_traceback(self):
        # The buffer must hold far more than a single multi-line traceback so a
        # crash dump isn't truncated by routine progress chatter.
        state = LauncherState(interval_hours=0)
        for i in range(4000):
            state.add_log(f"line {i}")
        lines = state.snapshot().log_lines
        assert len(lines) > 500          # old cap was 500
        assert lines[-1] == "line 3999"

    def test_child_log_path_none_without_run_dir(self):
        from main.launcher.child import child_log_path
        assert child_log_path(None) is None
        assert child_log_path("/models/run_x").endswith("launcher_child.log")

    def test_stdout_reader_streams_to_disk(self, tmp_path):
        # _read_child_stdout writes every line to the log file as it arrives, so a
        # hard child crash still leaves a complete log behind.
        from main.launcher.child import _read_child_stdout
        state = LauncherState(interval_hours=0)
        proc = MagicMock()
        proc.stdout = iter([b"first line\n", b"Traceback (most recent call last):\n", b"  boom\n"])
        log_path = tmp_path / "launcher_child.log"
        with open(log_path, "a", buffering=1) as f:
            _read_child_stdout(proc, state, f)
        contents = log_path.read_text()
        assert "first line" in contents
        assert "Traceback (most recent call last):" in contents and "boom" in contents

    def test_dump_on_exit_writes_buffer_when_stream_absent(self, tmp_path):
        # Fallback path: if streaming never captured anything, the in-memory
        # scrollback is flushed to the model dir on exit.
        from main.launcher.run import _dump_logs_on_exit
        state = LauncherState(interval_hours=0)
        state.run_dir = str(tmp_path)
        state.add_log("important crash detail")
        _dump_logs_on_exit(str(tmp_path), state)
        out = (tmp_path / "launcher_child.log").read_text()
        assert "important crash detail" in out
        assert "session ended" in out

    def test_dump_on_exit_appends_footer_without_duplicating_stream(self, tmp_path):
        # If streaming already wrote the log, the dump only appends a footer (it
        # must NOT re-dump the in-memory buffer on top of the streamed content).
        from main.launcher.run import _dump_logs_on_exit
        log_path = tmp_path / "launcher_child.log"
        log_path.write_text("streamed line A\nstreamed line B\n")
        state = LauncherState(interval_hours=0)
        state.run_dir = str(tmp_path)
        state.add_log("buffer only line")
        _dump_logs_on_exit(str(tmp_path), state)
        out = log_path.read_text()
        assert out.count("streamed line A") == 1
        assert "buffer only line" not in out      # not re-dumped over the stream
        assert "session ended" in out

    def test_dump_on_exit_noop_without_run_dir(self):
        from main.launcher.run import _dump_logs_on_exit
        state = LauncherState(interval_hours=0)
        _dump_logs_on_exit(None, state)   # must not raise
