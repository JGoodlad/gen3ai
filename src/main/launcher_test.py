import os
import signal
import subprocess
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from main.launcher import (
    _insert_or_replace_model_arg,
    _strip_launcher_arg,
    find_latest_checkpoint,
)


class TestFindLatestCheckpoint:
    def test_empty_dir_returns_none(self, tmp_path):
        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_single_zip_returned(self, tmp_path):
        p = tmp_path / "checkpoint_1000.zip"
        p.write_text("x")
        assert find_latest_checkpoint(str(tmp_path)) == str(p)

    def test_returns_newest_by_mtime(self, tmp_path):
        old = tmp_path / "checkpoint_1000.zip"
        old.write_text("x")
        # Ensure distinct mtimes.
        time.sleep(0.01)
        new = tmp_path / "checkpoint_2000.zip"
        new.write_text("x")
        assert find_latest_checkpoint(str(tmp_path)) == str(new)

    def test_ignores_non_zip(self, tmp_path):
        (tmp_path / "model.pt").write_text("x")
        (tmp_path / "notes.txt").write_text("x")
        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_searches_subdirectories(self, tmp_path):
        sub = tmp_path / "run_20240101"
        sub.mkdir()
        deep = sub / "checkpoint_5000.zip"
        deep.write_text("x")
        assert find_latest_checkpoint(str(tmp_path)) == str(deep)

    def test_newest_across_subdirectories(self, tmp_path):
        sub1 = tmp_path / "run_a"
        sub1.mkdir()
        old = sub1 / "checkpoint_1000.zip"
        old.write_text("x")
        time.sleep(0.01)
        sub2 = tmp_path / "run_b"
        sub2.mkdir()
        new = sub2 / "checkpoint_2000.zip"
        new.write_text("x")
        assert find_latest_checkpoint(str(tmp_path)) == str(new)


class TestInsertOrReplaceModelArg:
    def test_inserts_when_absent(self):
        result = _insert_or_replace_model_arg(["--debug", "--steps", "5000"], "models/ckpt.zip")
        assert result == ["--debug", "--steps", "5000", "--model", "models/ckpt.zip"]

    def test_replaces_existing(self):
        result = _insert_or_replace_model_arg(
            ["--model", "old.zip", "--steps", "5000"], "new.zip"
        )
        assert result == ["--model", "new.zip", "--steps", "5000"]

    def test_empty_args(self):
        result = _insert_or_replace_model_arg([], "ckpt.zip")
        assert result == ["--model", "ckpt.zip"]


class TestStripLauncherArg:
    def test_strips_separate_flag_and_value(self):
        args = ["--steps", "5000", "--restart-interval-hours", "3.0", "--debug"]
        assert _strip_launcher_arg(args) == ["--steps", "5000", "--debug"]

    def test_strips_equals_form(self):
        args = ["--restart-interval-hours=2.5", "--debug"]
        assert _strip_launcher_arg(args) == ["--debug"]

    def test_no_op_when_absent(self):
        args = ["--steps", "5000", "--debug"]
        assert _strip_launcher_arg(args) == args

    def test_empty(self):
        assert _strip_launcher_arg([]) == []


class TestCommandDispatch:
    """Test the command-handling logic inside run() by driving cmd_q directly."""

    def _run_one_command(self, cmd: str, interval_hours: float = 3.0):
        """
        Spin up a mock child that exits immediately (rc=0), inject one command,
        and capture the signals sent + prints emitted.
        """
        import queue as qmod
        from main.launcher import run

        cmd_q_real = qmod.Queue()
        cmd_q_real.put(cmd)

        sent_signals = []

        def fake_kill(pid, sig):
            sent_signals.append(sig)

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0
        # First wait(timeout=...) times out so the command queue gets drained.
        # Second wait(timeout=...) returns (process exited). Final bare wait() returns.
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd=[], timeout=1),
            None,
            None,
        ]

        with (
            patch("main.launcher.subprocess.Popen", return_value=mock_proc),
            patch("main.launcher.os.kill", side_effect=fake_kill),
            patch("main.launcher.threading.Thread"),  # suppress stdin thread
            patch("main.launcher.find_latest_checkpoint", return_value=None),
            patch("main.launcher.queue.Queue", return_value=cmd_q_real),
        ):
            with pytest.raises(SystemExit):
                run(["--debug"], interval_hours=0)  # no restart so exits after child

        return sent_signals

    def test_checkpoint_sends_sigusr1(self):
        for cmd in ("c", "checkpoint"):
            sent = self._run_one_command(cmd)
            assert signal.SIGUSR1 in sent, f"SIGUSR1 not sent for command '{cmd}'"

    def test_quit_sends_sigterm(self):
        for cmd in ("q", "quit"):
            sent = self._run_one_command(cmd)
            assert signal.SIGTERM in sent, f"SIGTERM not sent for command '{cmd}'"

    def test_restart_sends_sigterm(self):
        for cmd in ("r", "restart"):
            sent = self._run_one_command(cmd)
            assert signal.SIGTERM in sent, f"SIGTERM not sent for command '{cmd}'"

    def test_status_no_signal(self, capsys):
        sent = self._run_one_command("s")
        assert signal.SIGTERM not in sent
        assert signal.SIGUSR1 not in sent
        out = capsys.readouterr().out
        assert "PID" in out

    def test_unknown_command_no_signal(self):
        sent = self._run_one_command("foobar")
        assert signal.SIGTERM not in sent
        assert signal.SIGUSR1 not in sent


class TestIntervalRestart:
    def test_crash_does_not_restart(self):
        """Non-zero exit code should sys.exit without relaunching."""
        import queue as qmod
        from main.launcher import run

        mock_proc = MagicMock()
        mock_proc.pid = 99
        mock_proc.returncode = 1
        mock_proc.wait.return_value = None

        popen_calls = []

        def fake_popen(*a, **kw):
            popen_calls.append(a)
            return mock_proc

        with (
            patch("main.launcher.subprocess.Popen", side_effect=fake_popen),
            patch("main.launcher.threading.Thread"),
            patch("main.launcher.queue.Queue", return_value=qmod.Queue()),
        ):
            with pytest.raises(SystemExit) as exc_info:
                run(["--debug"], interval_hours=0)

        assert exc_info.value.code == 1
        assert len(popen_calls) == 1  # no second launch

    def test_clean_exit_with_no_restart_interval_exits_zero(self):
        import queue as qmod
        from main.launcher import run

        mock_proc = MagicMock()
        mock_proc.pid = 88
        mock_proc.returncode = 0
        mock_proc.wait.return_value = None

        with (
            patch("main.launcher.subprocess.Popen", return_value=mock_proc),
            patch("main.launcher.threading.Thread"),
            patch("main.launcher.queue.Queue", return_value=qmod.Queue()),
        ):
            with pytest.raises(SystemExit) as exc_info:
                run(["--debug"], interval_hours=0)

        assert exc_info.value.code == 0
