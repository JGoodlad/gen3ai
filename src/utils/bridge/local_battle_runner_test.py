"""Unit tests for the LocalBattleRunner framing (no Node, no subprocess).

The runner fabricates the room header the sim does not emit (`>battle-…` +
`|init|battle`). These tests pin that framing so `_create_battle` fires exactly
once per side and every later chunk routes to the existing battle.
"""

from utils.bridge.local_battle_runner import _LocalBattleRunner


def test_first_chunk_per_side_gets_init():
    inited = {"p1": False, "p2": False}
    tag = "battle-gen3ou-1"

    first = _LocalBattleRunner._frame(tag, "p1", "|gametype|singles\n|start", inited)
    lines = first.split("\n")
    assert lines[0] == ">battle-gen3ou-1"  # room header -> _handle_message current_room
    assert lines[1] == "|init|battle"       # triggers _create_battle
    assert lines[2] == "|gametype|singles"
    assert inited["p1"] is True


def test_subsequent_chunks_have_no_init():
    inited = {"p1": True, "p2": False}
    tag = "battle-gen3ou-1"

    chunk = _LocalBattleRunner._frame(tag, "p1", "|turn|2", inited)
    assert chunk == ">battle-gen3ou-1\n|turn|2"
    assert "|init|battle" not in chunk


def test_sides_init_independently():
    inited = {"p1": False, "p2": False}
    tag = "battle-gen3ou-3"

    p1_first = _LocalBattleRunner._frame(tag, "p1", "|start", inited)
    p2_first = _LocalBattleRunner._frame(tag, "p2", "|start", inited)
    assert "|init|battle" in p1_first
    assert "|init|battle" in p2_first  # p2's first chunk still gets its own init
    assert inited == {"p1": True, "p2": True}


def test_tag_round_trips_to_create_battle_contract():
    # _create_battle does ">battle-gen3ou-1".split("-") -> [">battle","gen3ou","1"],
    # checks [1] == format, and rebuilds the tag via "-".join(...)[1:].
    inited = {"p1": False, "p2": False}
    framed = _LocalBattleRunner._frame("battle-gen3ou-1", "p1", "|start", inited)
    room_line = framed.split("\n")[0]
    parts = room_line.split("-")
    assert parts[1] == "gen3ou"
    assert "-".join(parts)[1:] == "battle-gen3ou-1"


# --- the per-battle bound: IDLE gap, not total duration -------------------------------------
#
# gen3_battle_progress_deadline_v1. A duration cap on a bridge battle measures the box as much as
# the code, and contention SCALING does not rescue it — MEASURED 2026-08-14, against the parity
# test's then-20 s cap on a box saturated by a `cargo build --release`: **8 of 12 battles scored as
# timeouts (plus a transport error) and the test FAILED**, none of them wedged, all still producing
# protocol chunks; the same test passed warm with no code change. So the detector is the IDLE gap
# between chunks, which is ~load-invariant: contention stretches the gaps, a wedge stops them.
#
# These two tests ARE the evidence for the change — deliberately, because at this box's ordinary
# load nothing timed out either before or after (0 of 12 both ways), so there is no honest
# before/after rate to point at. The mechanism is what is being pinned, and the second test is what
# stops the fix quietly becoming "no bound at all".

import asyncio

import pytest

from utils.contention import ProgressTimeout
from utils.bridge import local_battle_runner as lbr


class _FakeClient:
    """Stands in for a BattleStreamClient — only `progress_count` is read."""

    def __init__(self):
        self.progress_count = 0


def test_a_SLOW_but_progressing_battle_is_not_a_timeout(monkeypatch):
    """THE regression: the battle runs 5x past the idle budget but keeps emitting chunks —
    exactly what a starved-but-healthy battle beside a training run looks like."""
    monkeypatch.setattr(lbr, "_BATTLE_IDLE_BUDGET", 0.4)
    monkeypatch.setattr(lbr, "_PROGRESS_POLL_S", 0.05)
    client = _FakeClient()

    async def _slow_but_alive():
        for _ in range(20):                 # ~2.0 s total vs a 0.4 s idle budget
            await asyncio.sleep(0.1)
            client.progress_count += 1      # a chunk arrived: a sign of life
        return "done"

    async def _run():
        return await lbr._await_battle(_slow_but_alive(), (client, None), "slow battle")

    assert asyncio.run(_run()) == "done"


def test_a_WEDGED_battle_still_times_out(monkeypatch):
    """The other half — a bound that never fires is not a bound."""
    monkeypatch.setattr(lbr, "_BATTLE_IDLE_BUDGET", 0.3)
    monkeypatch.setattr(lbr, "_PROGRESS_POLL_S", 0.05)
    client = _FakeClient()                  # never advances

    async def _wedged():
        await asyncio.sleep(30)

    async def _run():
        return await lbr._await_battle(_wedged(), (client, None), "wedged battle")

    with pytest.raises(ProgressTimeout, match="no progress"):
        asyncio.run(_run())


def test_the_wedged_battle_task_is_cancelled_not_leaked(monkeypatch):
    """A bound that raises but leaves the battle running leaks a bridge subprocess per timeout."""
    monkeypatch.setattr(lbr, "_BATTLE_IDLE_BUDGET", 0.3)
    monkeypatch.setattr(lbr, "_PROGRESS_POLL_S", 0.05)
    client = _FakeClient()
    state = {"cancelled": False}

    async def _wedged():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    async def _run():
        with pytest.raises(ProgressTimeout):
            await lbr._await_battle(_wedged(), (client, None), "wedged battle")

    asyncio.run(_run())
    assert state["cancelled"], "the battle task must be cancelled when the deadline fires"


def test_ProgressTimeout_is_caught_by_existing_timeout_handlers():
    """Callers count timeouts with `except (asyncio.TimeoutError, TimeoutError)`. If the new type
    escaped those, a starved battle would become an uncaught ERROR rather than a counted timeout —
    and the whole doctrine here is that a timeout never becomes a semantic outcome."""
    assert issubclass(ProgressTimeout, TimeoutError)
    assert issubclass(ProgressTimeout, asyncio.TimeoutError)
