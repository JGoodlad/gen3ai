"""The NODE bridge's reject-streak bound (`gen3_node_bridge_reject_bound_v1`).

THE GAP THIS CLOSES. `src/rust_sim/src/bridge.rs` has had `REJECT_STREAK_CAP = 8` since the
rust bridge shipped, for a measured incident: an RL policy is DETERMINISTIC given the same
request, so when its action mask disagrees with the sim about legality it re-sends the SAME
refused choice forever — one child was observed spinning at 46 MB/s of re-requests while its
env's `step()` never returned, wedging the whole vec-env. `local_sim_bridge.js` — the DEFAULT
in-process transport (`--use-bridge=node`) — had no equivalent bound, so the identical wedge
was reachable there.

WHY IT DOES NOT SELF-TERMINATE (probed, ROUND 36 + the round-30 park probe). `Side.emitChoiceError`
re-emits a `|request|` ONLY when the refusal actually CHANGED the request (the maybe-trapped
reveal). A plain `[Invalid choice]` emits the error and NOTHING else — and poke-env answers an
`[Invalid choice]` by immediately re-choosing off the request it already holds. A deterministic
client and the sim therefore ping-pong with ZERO sim progress, unbounded.

WHAT THIS TEST ASSERTS (all against the REAL node bridge subprocess over its real stdio protocol):
  1. `test_repeated_illegal_choice_fails_loud_instead_of_spinning` — the bound TRIPS: a
     deterministically illegal choice (`move 4` on a 2-move mon) re-sent well past the cap
     yields a single `__ERR__` frame naming the side, the offending choice and the refusal,
     within a hard wall-clock budget. PRE-FIX this test HANGS (the bridge relays forever and
     emits only `|error|` chunks), which is exactly the wedge — so the timeout is the pin.
  2. `test_a_legal_choice_after_refusals_still_commits` — the bound does NOT police ordinary
     refusals: a few refused choices followed by a LEGAL one still commits the turn. This is
     the over-eager-cap guard; without it, capping at 0 would pass test 1.
  3. `test_refusals_reset_across_committed_decisions` — the streak resets on PROGRESS, not on
     receiving a re-request. `cap` refusals, then a commit, then `cap` more must NOT trip: a
     model that never reset (or that reset on every re-request, making the cap unreachable)
     fails here. This is the reset-condition pin the spec called the easy thing to get wrong.

Marked `integration` (needs `deps/pokemon-showdown`; no live server, no battles played to end).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
BRIDGE = REPO / "src" / "utils" / "bridge" / "local_sim_bridge.js"
SHOWDOWN = REPO / "deps" / "pokemon-showdown"

# The bridge's cap (`REJECT_STREAK_CAP` in local_sim_bridge.js). Kept in lockstep by
# `test_cap_constant_is_in_lockstep_with_the_source` below so this file cannot silently drift.
CAP = 8

# A deterministic board: p1's lead knows exactly TWO moves, so `move 4` is refused every time
# with a plain `[Invalid choice]` (the shape that emits NO re-request — the wedge shape). Both
# sides are bulky and passive so nothing faints and no boundary resolves on its own.
P1_TEAM = "|snorlax||immunity|splash,bodyslam|Serious|252,252,,,,|N||||"
P2_TEAM = "|snorlax||immunity|splash,bodyslam|Serious|252,252,,,,|N||||"

BUDGET_S = 25.0  # generous: the wedge spins in milliseconds, so a hang blows straight past this


def _requires_showdown() -> None:
    if not (SHOWDOWN / "dist" / "sim" / "index.js").exists():
        pytest.skip("deps/pokemon-showdown dist/ not built (see root CLAUDE.md worktree setup)")


class Bridge:
    """A thin driver for the bridge's real stdio protocol (START/CHOOSE/END → pN/__END__/__ERR__)."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            ["node", str(BRIDGE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(REPO),
            text=True,
            bufsize=1,
        )

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def start(self, seed: str = "1,2,3,4") -> None:
        self.send(
            "START "
            + json.dumps(
                {
                    "formatid": "gen3customgame",
                    "seed": seed,
                    "p1": {"name": "P1", "team": P1_TEAM},
                    "p2": {"name": "P2", "team": P2_TEAM},
                }
            )
        )

    def read_frames_until(self, pred, budget_s: float = BUDGET_S) -> list[str]:
        """Read stdout frames until `pred(frame)` or the budget expires.

        A HANG is the pre-fix failure mode, so the budget is enforced by the caller's
        timeout too — here it produces a readable assertion rather than a silent stall.
        """
        assert self.proc.stdout is not None
        out: list[str] = []
        deadline = time.monotonic() + budget_s
        os.set_blocking(self.proc.stdout.fileno(), False)
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    break
                time.sleep(0.005)
                continue
            frame = line.rstrip("\n")
            out.append(frame)
            if pred(frame):
                return out
        return out

    def close(self) -> None:
        try:
            self.send("END")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def _decode(frame: str) -> str:
    """`pN <b64>` / `__ERR__ <b64>` → the decoded payload ('' for a bare frame)."""
    parts = frame.split(" ", 1)
    if len(parts) != 2:
        return ""
    try:
        return base64.b64decode(parts[1]).decode("utf-8", "replace")
    except Exception:
        return ""


def test_cap_constant_is_in_lockstep_with_the_source() -> None:
    """This file's CAP must equal the bridge's — else tests 1 and 3 test the wrong boundary."""
    src = BRIDGE.read_text()
    assert (
        f"REJECT_STREAK_CAP = {CAP}" in src
    ), f"local_sim_bridge.js's REJECT_STREAK_CAP no longer equals this test's CAP={CAP}"


def test_repeated_illegal_choice_fails_loud_instead_of_spinning() -> None:
    """THE WEDGE PIN: a deterministically-refused choice re-sent past the cap must `__ERR__`.

    PRE-FIX this HANGS — the bridge relays every refusal and the sim, having emitted a plain
    `[Invalid choice]` with no re-request, never advances. The wall-clock budget IS the
    assertion: a spinning bridge blows past it and the test fails on the missing frame.
    """
    _requires_showdown()
    b = Bridge()
    try:
        b.start()
        # Drain the framing + the first request so the battle is genuinely at a move boundary.
        b.read_frames_until(lambda f: "request" in _decode(f), budget_s=10.0)
        # Re-send the SAME illegal choice well past the cap, exactly as a deterministic policy
        # with a disagreeing action mask would.
        for _ in range(CAP + 4):
            b.send("CHOOSE p1 move 4")
        frames = b.read_frames_until(lambda f: f.startswith("__ERR__"))
        errs = [f for f in frames if f.startswith("__ERR__")]
        assert errs, (
            "the node bridge must FAIL LOUD on a no-progress reject loop, not spin — "
            f"got {len(frames)} frames and no __ERR__ within {BUDGET_S}s "
            "(this is the pre-fix wedge: it relays refusals forever)"
        )
        msg = _decode(errs[0])
        assert "reject loop" in msg, f"the __ERR__ must name the condition, got: {msg!r}"
        assert "p1" in msg, f"the __ERR__ must name the offending SIDE, got: {msg!r}"
        assert "move 4" in msg, f"the __ERR__ must name the offending CHOICE, got: {msg!r}"
    finally:
        b.close()


def test_a_legal_choice_after_refusals_still_commits() -> None:
    """THE OVER-EAGER-CAP GUARD: a few refusals then a LEGAL choice must still commit.

    Without this, a cap of 0 (or any model that treats one refusal as fatal) would satisfy the
    wedge pin while breaking the DEFAULT transport for ordinary, legitimate refusals — the
    maybe-trapped probe is a normal two-exchange round.
    """
    _requires_showdown()
    b = Bridge()
    try:
        b.start()
        b.read_frames_until(lambda f: "request" in _decode(f), budget_s=10.0)
        for _ in range(CAP - 1):  # strictly under the cap
            b.send("CHOOSE p1 move 4")
        b.read_frames_until(lambda f: "[Invalid choice]" in _decode(f), budget_s=10.0)
        # Now both sides answer LEGALLY — the turn must resolve.
        b.send("CHOOSE p1 move 1")
        b.send("CHOOSE p2 move 1")
        frames = b.read_frames_until(
            lambda f: "|turn|2" in _decode(f) or "|move|" in _decode(f)
        )
        assert not any(f.startswith("__ERR__") for f in frames), (
            "refusals strictly UNDER the cap must not be fatal — the bound exists to catch a "
            "no-progress LOOP, not to police ordinary refusals"
        )
        assert any(
            "|move|" in _decode(f) or "|turn|2" in _decode(f) for f in frames
        ), "the legal choice must still commit the turn after sub-cap refusals"
    finally:
        b.close()


def test_refusals_reset_across_committed_decisions() -> None:
    """THE RESET-CONDITION PIN: the streak resets on PROGRESS, not on a re-request.

    `cap` refusals → a COMMITTED turn → `cap` more refusals must NOT trip: the commit resets.
    A model that never reset would trip on the second batch; one that reset on every received
    re-request could never trip at all (and would fail the wedge pin instead). This pins the
    exact condition the spec flagged as the easy thing to get wrong.
    """
    _requires_showdown()
    b = Bridge()
    try:
        b.start()
        b.read_frames_until(lambda f: "request" in _decode(f), budget_s=10.0)
        for _ in range(CAP - 1):
            b.send("CHOOSE p1 move 4")
        b.read_frames_until(lambda f: "[Invalid choice]" in _decode(f), budget_s=10.0)
        # A committed decision — this is the ONLY thing that may reset the streak.
        b.send("CHOOSE p1 move 1")
        b.send("CHOOSE p2 move 1")
        b.read_frames_until(lambda f: "|turn|2" in _decode(f), budget_s=15.0)
        # A second batch, again strictly under the cap: must NOT be fatal.
        for _ in range(CAP - 1):
            b.send("CHOOSE p1 move 4")
        frames = b.read_frames_until(lambda f: f.startswith("__ERR__"), budget_s=8.0)
        assert not any(f.startswith("__ERR__") for f in frames), (
            "the reject streak must RESET on a committed decision — two sub-cap batches "
            "separated by a real turn must not accumulate into a false wedge report"
        )
    finally:
        b.close()
