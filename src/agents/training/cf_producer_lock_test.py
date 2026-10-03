"""The cf label PRODUCER's single-instance lock (`cf_producer_lock.py`).

What survived of the deleted trainer-side supply (deletion pass L4): ONE producer per run directory,
held as a kernel `flock` for the producer's lifetime. Each test fails on the revert of the rule it
names — a lock that is never exclusive, a stale file mistaken for a live holder, a holder that is
not released when it closes.
"""
from __future__ import annotations

import os

from agents.training import cf_producer_lock as L
from agents.training.cf_producer_lock import (PRODUCER_EXIT_LOCK_HELD, acquire_producer_lock,
                                              bind_to_parent, live_producer_pid)


def _run_dir(tmp_path) -> str:
    rd = tmp_path / "run"
    rd.mkdir()
    return str(rd)


def test_a_second_claimant_is_refused_while_the_first_holds_the_lock(tmp_path):
    rd = _run_dir(tmp_path)
    fd = acquire_producer_lock(rd)
    assert fd is not None
    try:
        assert live_producer_pid(rd) == os.getpid(), "the holder's pid is the one a refusal names"
        assert acquire_producer_lock(rd) is None, "two producers would share one state file"
    finally:
        os.close(fd)
    again = acquire_producer_lock(rd)
    try:
        assert again is not None, "the lock must be free again once its holder closes"
    finally:
        os.close(again)


def test_a_stale_lock_file_without_a_holder_is_not_a_producer(tmp_path):
    rd = _run_dir(tmp_path)
    (tmp_path / "run" / L.LOCK_NAME).write_text("12345\n")        # a dead producer's file
    assert live_producer_pid(rd) is None
    fd = acquire_producer_lock(rd)
    try:
        assert fd is not None and live_producer_pid(rd) == os.getpid()
    finally:
        os.close(fd)
    assert live_producer_pid(rd) is None


def test_no_lock_file_means_no_producer(tmp_path):
    assert live_producer_pid(_run_dir(tmp_path)) is None


def test_the_parent_binding_reports_whether_the_named_parent_is_still_ours():
    assert bind_to_parent(os.getppid()) is True
    assert bind_to_parent(os.getpid()) is False        # this process is not its own parent


def test_the_lock_held_exit_code_is_the_one_the_producer_cli_returns():
    from agents.training import cf_producer
    assert cf_producer.PRODUCER_EXIT_LOCK_HELD == PRODUCER_EXIT_LOCK_HELD == 4
