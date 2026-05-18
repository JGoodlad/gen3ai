"""
Replay-based validation of move effectiveness and move-order tracking.

Parses real battle replay HTML files from disk, feeds the raw Showdown protocol
messages into a Battle instance, and verifies that the properties
our_last_effectiveness, opp_last_effectiveness, and we_moved_first match an
independent interpretation of the raw message stream.

This is a second safety net independent of the deterministic effectiveness_test.py
(which uses hand-crafted messages) and the fuzz test (which requires a live server).

Runs in unit-test mode (no server needed).
"""
import logging
import os
import re
from typing import Optional
import pytest

from poke_env.battle.battle import Battle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_repo_root() -> str:
    """Walk up from this file until we find a directory with a 'models/' subdir."""
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):  # stop at filesystem root
        if os.path.isdir(os.path.join(d, "models")):
            return d
        d = os.path.dirname(d)
    return d  # fallback

_REPO_ROOT = _find_repo_root()
REPLAY_DIR = os.path.join(_REPO_ROOT, "models/gen3ou_ppo_new_20260516_223423/replays/step_96")


def _extract_protocol_lines(html_path: str) -> list[str]:
    """Extract raw Showdown protocol lines from a battle replay HTML file."""
    with open(html_path) as f:
        html = f.read()
    lines = []
    for line in html.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and len(stripped) > 2 and stripped != "|":
            lines.append(stripped)
    return lines


def _detect_player_role(lines: list[str], username: str) -> Optional[str]:
    """Find which player role (p1/p2) matches the given username in |player| messages."""
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 4 and parts[1] == "player":
            role, name = parts[2], parts[3]
            if name == username:
                return role
    return None


def _split_into_turns(lines: list[str]) -> list[list[str]]:
    """Split protocol lines into per-turn groups. Each group starts with |turn|N."""
    turns = []
    current: list[str] = []
    for line in lines:
        if line.startswith("|turn|"):
            if current:
                turns.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        turns.append(current)
    return turns


class _RawTurnInterpreter:
    """
    Independent interpreter of raw Showdown protocol for one turn.
    Provides expected values that Battle.our/opp_last_effectiveness should return
    at the start of the NEXT turn.
    """

    def __init__(self, turn_lines: list[str], player_role: str):
        self.player_role = player_role
        # Scan lines
        our_eff: Optional[float] = None
        opp_eff: Optional[float] = None
        we_first: Optional[bool] = None
        move_sides: list[str] = []

        for line in turn_lines:
            parts = line.split("|")
            if len(parts) < 2:
                continue
            mt = parts[1]

            if mt == "move" and len(parts) >= 3:
                mover_prefix = parts[2][:2]  # "p1" or "p2"
                move_sides.append(mover_prefix)

            elif mt == "-supereffective" and len(parts) >= 3:
                defender_prefix = parts[2][:2]
                mult = 2.0
                if defender_prefix == player_role:
                    opp_eff = mult   # opp attacked us, it was SE
                else:
                    our_eff = mult   # we attacked them, it was SE

            elif mt == "-resisted" and len(parts) >= 3:
                defender_prefix = parts[2][:2]
                mult = 0.5
                if defender_prefix == player_role:
                    opp_eff = mult
                else:
                    our_eff = mult

            elif mt == "-immune" and len(parts) >= 3:
                defender_prefix = parts[2][:2]
                mult = 0.0
                if defender_prefix == player_role:
                    opp_eff = mult
                else:
                    our_eff = mult

        if len(move_sides) == 2:
            we_first = (move_sides[0] == player_role)

        self.expected_our_eff = our_eff
        self.expected_opp_eff = opp_eff
        self.expected_we_moved_first = we_first


def _replay_battle(html_path: str) -> list[dict]:
    """
    Replay a battle from an HTML replay file through Battle.parse_message().
    Returns a list of per-turn result dicts containing expected vs actual values.
    """
    lines = _extract_protocol_lines(html_path)

    # Determine which player "we" are from |player|p1|...|
    p1_line = next((l for l in lines if l.startswith("|player|p1|")), None)
    if p1_line is None:
        pytest.skip("Could not find |player|p1| line in replay")
    username = p1_line.split("|")[3]
    player_role = _detect_player_role(lines, username) or "p1"

    b = Battle("battle-replay-test", username, logging.getLogger("test"), gen=3)
    b._player_role = player_role

    turns = _split_into_turns(lines)
    results = []

    # Messages handled at the Player level, never forwarded to parse_message()
    _PLAYER_LEVEL = {"win", "tie", "request", "showteam", "error", "uhtml", "html"}

    for turn_idx, turn_lines in enumerate(turns):
        # Parse all messages for this "turn group"
        for line in turn_lines:
            parts = line.split("|")
            if len(parts) < 2:
                continue
            if parts[1] in _PLAYER_LEVEL:
                continue
            b.parse_message(parts)

        if not turn_lines[0].startswith("|turn|"):
            continue  # skip pre-turn init lines

        # This turn's raw interpreter gives us expected values for the NEXT decision
        raw = _RawTurnInterpreter(turn_lines, player_role)

        # Check if there's a next turn (where properties should be readable)
        if turn_idx + 1 < len(turns):
            # Parse the next turn's opening |turn|N message to advance the counter
            next_turn_lines = turns[turn_idx + 1]
            next_turn_line = next_turn_lines[0]
            if next_turn_line.startswith("|turn|"):
                b.parse_message(next_turn_line.split("|"))

                results.append({
                    "turn": turn_lines[0],
                    "expected_our": raw.expected_our_eff,
                    "expected_opp": raw.expected_opp_eff,
                    "expected_order": raw.expected_we_moved_first,
                    "actual_our": b.our_last_effectiveness,
                    "actual_opp": b.opp_last_effectiveness,
                    "actual_order": b.we_moved_first,
                    "raw_turn_lines": turn_lines,
                })

    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _get_replay_files() -> list[str]:
    if not os.path.isdir(REPLAY_DIR):
        return []
    return [
        os.path.join(REPLAY_DIR, f)
        for f in sorted(os.listdir(REPLAY_DIR))
        if f.endswith("_replay.html")
    ]


@pytest.mark.parametrize("html_path", _get_replay_files())
def test_replay_effectiveness_consistent(html_path: str):
    """
    For each replay, verify that Battle.parse_message() produces effectiveness and
    move-order values that match an independent raw-message interpretation.

    This catches any discrepancy between what Showdown actually sent and what
    our poke-env parser records.
    """
    results = _replay_battle(html_path)
    assert results, f"No turn results extracted from {html_path}"

    mismatches = []
    for r in results:
        # Only validate turns where at least one EXPLICIT effectiveness message fired
        # (|-supereffective|, |-resisted|, or |-immune|). Turns where no such message
        # fired rely on the tentative-1.0 logic which is covered by deterministic tests.
        has_explicit_eff = (
            r["expected_our"] is not None or r["expected_opp"] is not None
        )

        our_ok = (
            r["actual_our"] == r["expected_our"]
            if has_explicit_eff and r["expected_our"] is not None
            else True
        )
        opp_ok = (
            r["actual_opp"] == r["expected_opp"]
            if has_explicit_eff and r["expected_opp"] is not None
            else True
        )
        order_ok = r["actual_order"] == r["expected_order"]
        if not (our_ok and opp_ok and order_ok):
            mismatches.append(r)

    if mismatches:
        lines = [f"\n{len(mismatches)} mismatch(es) in {os.path.basename(html_path)}:"]
        for m in mismatches:
            lines.append(f"  After {m['turn']}:")
            if m["actual_our"] != m["expected_our"]:
                lines.append(
                    f"    our_eff: expected={m['expected_our']} actual={m['actual_our']}"
                )
            if m["actual_opp"] != m["expected_opp"]:
                lines.append(
                    f"    opp_eff: expected={m['expected_opp']} actual={m['actual_opp']}"
                )
            if m["actual_order"] != m["expected_order"]:
                lines.append(
                    f"    we_moved_first: expected={m['expected_order']} actual={m['actual_order']}"
                )
            # Show the problematic turn's raw lines for debugging
            eff_lines = [
                l for l in m["raw_turn_lines"]
                if "supereffective" in l or "resisted" in l or "immune" in l or "|move|" in l
            ]
            if eff_lines:
                lines.append(f"    relevant raw lines: {eff_lines}")
        pytest.fail("\n".join(lines))


def test_replay_coverage():
    """
    Across all available replays, verify that every effectiveness category
    (immune, resisted, normal, SE) appears at least once, confirming the replay
    set is diverse enough to be a useful regression corpus.
    """
    replay_files = _get_replay_files()
    if not replay_files:
        pytest.skip("No replay files found")

    seen_immune = seen_resisted = seen_normal = seen_se = False
    seen_we_first = seen_opp_first = seen_switch_turn = False

    for html_path in replay_files:
        for r in _replay_battle(html_path):
            # Use actual values (includes tentative 1.0 for neutral moves)
            for val in (r["actual_our"], r["actual_opp"]):
                if val == 0.0:
                    seen_immune = True
                elif val == 0.5:
                    seen_resisted = True
                elif val == 1.0:
                    seen_normal = True
                elif val is not None and val >= 2.0:
                    seen_se = True
            # Use expected for order (raw interpretation is correct for move order)
            order = r["expected_order"]
            if order is True:
                seen_we_first = True
            elif order is False:
                seen_opp_first = True
            elif order is None:
                seen_switch_turn = True

    missing = []
    if not seen_immune:
        missing.append("immune (0.0x) — no replay with Ground vs Levitate or Electric vs Volt Absorb?")
    if not seen_resisted:
        missing.append("resisted (0.5x)")
    if not seen_normal:
        missing.append("normal (1.0x)")
    if not seen_se:
        missing.append("super-effective (2.0x+)")
    if not seen_we_first:
        missing.append("we_moved_first=True")
    if not seen_opp_first:
        missing.append("we_moved_first=False")
    if not seen_switch_turn:
        missing.append("we_moved_first=None (switch turn)")

    if missing:
        pytest.fail(
            "Replay corpus is missing coverage for:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )
