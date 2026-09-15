"""THE WATCHDOG — the half of this tool that exists because a dead peer looks like silence.

Hazard H-B, measured 2026-09-14: when our side forfeits at turn 250 the battle ends while Metamon's
env is mid-step; its next `step` raises "Battle is already finished", its long-tail handler
force-resets (sending a fresh challenge) and then **calls itself**, never converging — ~985
recursion levels, then a `RecursionError` that kills the process. A harness without a watchdog then
sits with a live client, no opponent, and nothing at all to report. 4 forfeits in 800 games hit 3 of
16 half-cells.

Every other way a head-to-head dies has the same symptom, and none of them is an exception on our
side:

* a team Showdown REJECTS — the challenge never becomes a battle (de-risk H1, matched-regime H-A);
* a 19-character username — `action.php` returns a refusal the peer accepts as an assertion, logs
  "Successfully logged in", and blocks forever (foul-play H1);
* a challenge aimed at a user who is not yet online — Showdown drops it silently.

So the gate is: **four NAMED causes, each raised as a `SeriesFailure`, never a silent zero.** Every
test here drives the failing branch rather than asserting the happy one, because a watchdog that
cannot fire is indistinguishable from a series that never stalls.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from main.anchors.peers import PeerPlan
from main.anchors.session import (
    BattleRecord,
    OurSideState,
    SeriesFailure,
    await_peer_ready,
    build_team_source,
    load_team_texts,
    log_has,
    log_tail,
    watch,
)


class FakePeer:
    """A stand-in for `subprocess.Popen` with only what the watchdog reads.

    A local object, not a patched module symbol: the watchdog takes the process as a PARAMETER, so
    there is nothing to stub and no way for this to reach the wrong code path.
    """

    def __init__(self, returncode=None) -> None:
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def die(self, rc: int = 1) -> None:
        self.returncode = rc


def _plan(tmp_path: Path, text: str = "") -> PeerPlan:
    log = tmp_path / "peer.log"
    log.write_text(text)
    return PeerPlan(label="metamon:SmallRL", argv=["true"], cwd=tmp_path, env={},
                    ready_pattern=r"Made Challenge Env", log_path=log)


def _record(i: int) -> BattleRecord:
    return BattleRecord(battle_tag=f"battle-gen3ou-{i}", turns=40, won=True, finished=True,
                        hit_forfeit_limit=False, our_team="t", n_decisions=38, n_defaults=0,
                        n_redecides=0, t_finished=time.time())


# ------------------------------------------------------------------------- cause: peer_exited
async def test_a_peer_that_dies_mid_series_is_named_not_silence(tmp_path: Path) -> None:
    """THE H-B CASE. The failure must name the peer, its rc and how far the series got."""
    plan = _plan(tmp_path, "RecursionError: maximum recursion depth exceeded\n")
    proc = FakePeer()
    state = OurSideState()
    state.records.extend(_record(i) for i in (1, 2, 3))
    state.last_progress = time.time()

    async def kill_soon() -> None:
        await asyncio.sleep(0.1)
        proc.die(1)

    asyncio.ensure_future(kill_soon())
    with pytest.raises(SeriesFailure) as excinfo:
        await watch(proc, plan, state, expected=100,
                    first_game_timeout_s=60.0, progress_timeout_s=60.0, poll_s=0.02)
    assert excinfo.value.cause == "peer_exited"
    detail = excinfo.value.detail
    assert "3/100" in detail
    assert "rc=1" in detail
    assert "RecursionError" in detail, "the peer's own log tail must be in the failure"
    assert "H-B" in detail, "the named hazard belongs in the message a human reads"


async def test_a_peer_exit_after_the_expected_count_is_NOT_a_failure(tmp_path: Path) -> None:
    """The control. The peer normally exits first at the end of a series; that is not a death."""
    plan = _plan(tmp_path)
    state = OurSideState()
    state.records.extend(_record(i) for i in range(1, 5))
    await watch(FakePeer(returncode=0), plan, state, expected=4,
                first_game_timeout_s=60.0, progress_timeout_s=60.0, poll_s=0.02)


# ------------------------------------------------------- causes: no_first_game / no_progress
async def test_nothing_finishing_at_all_is_no_first_game(tmp_path: Path) -> None:
    """A rejected team and a dropped challenge both look exactly like this: a live pair, no games."""
    with pytest.raises(SeriesFailure) as excinfo:
        await watch(FakePeer(), _plan(tmp_path, "waiting...\n"), OurSideState(), expected=10,
                    first_game_timeout_s=0.05, progress_timeout_s=600.0, poll_s=0.02)
    assert excinfo.value.cause == "no_first_game"
    assert "0/10" in excinfo.value.detail
    assert "HANG" in excinfo.value.detail


async def test_games_stopping_part_way_is_no_progress_and_carries_the_count(tmp_path: Path) -> None:
    state = OurSideState()
    state.records.extend(_record(i) for i in (1, 2))
    state.last_progress = time.time() - 100.0
    with pytest.raises(SeriesFailure) as excinfo:
        await watch(FakePeer(), _plan(tmp_path), state, expected=10,
                    first_game_timeout_s=600.0, progress_timeout_s=1.0, poll_s=0.02)
    assert excinfo.value.cause == "no_progress"
    assert "2/10" in excinfo.value.detail


async def test_a_series_that_keeps_finishing_games_is_never_failed(tmp_path: Path) -> None:
    """THE POSITIVE CONTROL. A watchdog that fired on a healthy series would be worse than none."""
    state = OurSideState()
    plan = _plan(tmp_path)
    proc = FakePeer()

    async def play() -> None:
        for i in range(1, 5):
            await asyncio.sleep(0.05)
            state.records.append(_record(i))
            state.last_progress = time.time()

    asyncio.ensure_future(play())
    await watch(proc, plan, state, expected=4, first_game_timeout_s=30.0,
                progress_timeout_s=30.0, poll_s=0.02)
    assert len(state.records) == 4


# ------------------------------------------------------------------- cause: peer_never_ready
async def test_a_peer_that_never_prints_its_banner_is_named(tmp_path: Path) -> None:
    """The ACCEPTOR must be online before the challenger's first `/challenge` — Showdown drops a
    challenge aimed at a user who is not online and the challenger then waits forever."""
    with pytest.raises(SeriesFailure) as excinfo:
        await await_peer_ready(FakePeer(), _plan(tmp_path, "loading weights\n"), timeout_s=0.1,
                               poll_s=0.02)
    assert excinfo.value.cause == "peer_never_ready"
    assert "Made Challenge Env" in excinfo.value.detail


async def test_a_peer_that_dies_before_coming_online_is_peer_exited(tmp_path: Path) -> None:
    with pytest.raises(SeriesFailure) as excinfo:
        await await_peer_ready(FakePeer(returncode=2), _plan(tmp_path, "AssertionError: flash\n"),
                               timeout_s=30.0, poll_s=0.02)
    assert excinfo.value.cause == "peer_exited"
    assert "rc=2" in excinfo.value.detail


async def test_readiness_is_the_peers_own_banner_and_nothing_else(tmp_path: Path) -> None:
    await await_peer_ready(FakePeer(), _plan(tmp_path, "...\nMade Challenge Env\n"),
                           timeout_s=5.0, poll_s=0.02)


# ------------------------------------------------------------------------------ the log helpers
def test_log_helpers_answer_honestly_when_there_is_no_log(tmp_path: Path) -> None:
    missing = tmp_path / "nope.log"
    assert log_has(missing, "anything") is False
    assert "no log" in log_tail(missing)


# ------------------------------------------------------------------------------- the team source
def test_load_team_texts_strips_the_trailing_blank_lines_that_hung_a_series(
        tmp_path: Path) -> None:
    """The third line of defence behind the fork fix and the pack guard. Cheap, so it stays."""
    (tmp_path / "a.gen3ou_team").write_text("Skarmory @ Leftovers\n- Spikes\n\n\n")
    (tmp_path / "index.csv").write_text("filename\na.gen3ou_team\n")
    items = load_team_texts(tmp_path)
    assert [name for name, _ in items] == ["a.gen3ou_team"], (
        "index.csv is bookkeeping, not a team — counting it reports a team-source asymmetry that "
        "does not exist"
    )
    assert not items[0][1].endswith("\n")


def test_an_empty_team_directory_is_a_named_failure_not_an_empty_pool(tmp_path: Path) -> None:
    with pytest.raises(SeriesFailure) as excinfo:
        build_team_source({"kind": "dir", "path": str(tmp_path)}, seed=1, draw_log=[])
    assert excinfo.value.cause == "no_teams"


def test_an_unknown_team_source_kind_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SeriesFailure) as excinfo:
        build_team_source({"kind": "telepathy"}, seed=1, draw_log=[])
    assert excinfo.value.cause == "bad_team_source"


def test_a_battle_record_reads_a_tie_as_a_tie_and_not_a_loss(tmp_path: Path) -> None:
    """poke-env's own three flags: `won is None` with `finished` true is a TIE. Metamon's own CSV
    cannot express that (its `won` is a boolean, de-risk H3), which is why OUR side is the
    authority on every result."""
    rec = _record(1)
    assert rec.result == "win"
    assert BattleRecord(**{**rec.__dict__, "won": None}).result == "tie"
    assert BattleRecord(**{**rec.__dict__, "won": False}).result == "loss"
