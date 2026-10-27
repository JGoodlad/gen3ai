"""H5 fail-fast and H21: a peer's long-tail force-reset line voids the unit, at the FIRST line, anywhere in it.

Measured 2026-10-27 (``measurements/kakuna_determinism_2026-10-27/`` §4): when Metamon's long-tail handler fires on
our turn-250 forfeit, the process LIVES ON and costs the next two games (a 1-turn Kakuna forfeit, then a game on
carried state). The watchdog now stops the half at the first such line; the half's report makes ``peer_clean`` False
for any such line, so the verdict does not hang on whether the peer also recorded an ``error``.

The log text below is the archived peer log's own (n1p_s3503__p72__03's first faulted play, lines 39-42).
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from main.anchors.peers import PeerPlan
from main.anchors.results import unit_counts
from main.anchors.runner import apply_peer_fault, finalize_regime_fields
from main.anchors.session import (
    PEER_FAULT_LINE,
    BattleRecord,
    OurSideState,
    PeerFaultScan,
    SeriesFailure,
    peer_fault_lines,
    watch,
)

CLEAN_LOG = "Challenge mode (challenger): MetaKakuna2 vs Gen3AIAnchor2 for 10 battles\n  Turn    2. | ... \n"
FAULT = ("  Turn  250. | [●⦻⦻⦻⦻⦻][302/302hp]     flygon -   swampert [100%hp][⦻⦻●●●●]"
         "Battle is already finished, call reset\nForce resetting due to long-tail error\n"
         "Battle is already finished, call reset\nForce resetting due to long-tail error\n")


class FakePeer:
    def __init__(self, returncode=None) -> None:
        self.returncode = returncode

    def poll(self):
        return self.returncode


def _plan(tmp_path: Path, text: str) -> PeerPlan:
    log = tmp_path / "peer.log"
    log.write_text(text)
    return PeerPlan(label="metamon:Kakuna", argv=["true"], cwd=tmp_path, env={},
                    ready_pattern=r"Made Challenge Env", log_path=log)


def _record(i: int) -> BattleRecord:
    return BattleRecord(battle_tag=f"battle-gen3ou-{i}", turns=40, won=True, finished=True,
                        hit_forfeit_limit=False, our_team="t", n_decisions=38, n_defaults=0,
                        n_redecides=0, t_finished=time.time())


# ------------------------------------------------------------------------------------------ the scanner
def test_the_scanner_counts_the_line_across_partial_writes(tmp_path: Path) -> None:
    log = tmp_path / "peer.log"
    log.write_text(CLEAN_LOG + "Force resetting due to lo")       # the peer is mid-write
    scan = PeerFaultScan(log)
    assert scan.scan() == 0
    with open(log, "a") as fh:
        fh.write("ng-tail error\nmore\n")
    assert scan.scan() == 1
    assert scan.first_line == 3
    assert peer_fault_lines(log) == 1
    assert peer_fault_lines(tmp_path / "absent.log") == 0


def test_a_last_line_without_a_newline_still_counts_once_the_peer_exited(tmp_path: Path) -> None:
    log = tmp_path / "peer.log"
    log.write_text(CLEAN_LOG + PEER_FAULT_LINE)
    assert peer_fault_lines(log) == 1


# ------------------------------------------------------------------------------------- H5 fail-fast
async def test_the_watchdog_fails_the_half_at_the_first_fault_line_while_the_peer_lives(tmp_path: Path) -> None:
    """THE H5 CASE: 15 of 20 games done, the peer ALIVE (it lives on after the fault) and progressing. Before this
    the watchdog played games 16-20, two of them contaminated; now the half stops at once, by name."""
    plan = _plan(tmp_path, CLEAN_LOG)
    state = OurSideState()
    state.records.extend(_record(i) for i in range(1, 16))
    state.last_progress = time.time()

    async def fault_soon() -> None:
        await asyncio.sleep(0.05)
        with open(plan.log_path, "a") as fh:
            fh.write(FAULT)

    asyncio.ensure_future(fault_soon())
    with pytest.raises(SeriesFailure) as excinfo:
        await asyncio.wait_for(watch(FakePeer(), plan, state, expected=20, first_game_timeout_s=60.0,
                                     progress_timeout_s=60.0, poll_s=0.02), timeout=30.0)
    assert excinfo.value.cause == "peer_long_tail_fault"
    assert "15/20" in excinfo.value.detail and "H5" in excinfo.value.detail


async def test_a_clean_log_never_fires_the_fault(tmp_path: Path) -> None:
    """The control: the same healthy series with no fault line completes exactly as before."""
    plan = _plan(tmp_path, CLEAN_LOG)
    state = OurSideState()

    async def play() -> None:
        for i in range(1, 5):
            await asyncio.sleep(0.03)
            state.records.append(_record(i))
            state.last_progress = time.time()

    asyncio.ensure_future(play())
    await asyncio.wait_for(watch(FakePeer(), plan, state, expected=4, first_game_timeout_s=30.0,
                                 progress_timeout_s=30.0, poll_s=0.02), timeout=30.0)
    assert len(state.records) == 4


# ---------------------------------------------------------------------------------------------- H21
def _report(peer_error_free: bool = True) -> dict:
    report = {"regime_verified_decisions": True, "peer_error_free": peer_error_free}
    finalize_regime_fields(report, peer_rcs=[0])
    return report


def test_a_fault_after_the_last_game_with_no_recorded_error_never_counts(tmp_path: Path) -> None:
    """THE H21 HOLE: a force-reset that RECOVERED (no RecursionError, rc 0) left ``peer_clean`` True before; any
    fault line now makes it False and fails the half, so the unit cannot count."""
    plan = _plan(tmp_path, CLEAN_LOG + FAULT)
    report = _report()
    assert report["peer_clean"] is True                          # what the rc and the error alone said
    failure = apply_peer_fault(report, plan, 20, 20, None)
    assert report["peer_fault_lines"] == 2
    assert report["peer_clean"] is False and report["regime_verified"] is False
    assert failure is not None and failure.cause == "peer_long_tail_fault"
    summary = {"status": "FAILED", "n": 20, "regime_verified_decisions": True, "peer_clean": False,
               "team_source_asymmetry": False}
    assert not unit_counts(summary, games=20).counts


def test_an_earlier_failure_is_kept_and_a_clean_log_changes_nothing(tmp_path: Path) -> None:
    plan = _plan(tmp_path, CLEAN_LOG + FAULT)
    first = SeriesFailure("peer_exited", "rc=1")
    assert apply_peer_fault(_report(), plan, 3, 20, first) is first
    (tmp_path / "clean").mkdir()
    clean = _plan(tmp_path / "clean", CLEAN_LOG)
    report = _report()
    before = dict(report)
    assert apply_peer_fault(report, clean, 20, 20, None) is None
    assert report == dict(before, peer_fault_lines=0)
