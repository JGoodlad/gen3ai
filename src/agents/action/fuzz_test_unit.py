"""Snapshot-immutability simulation (standalone script).

Replaces the old decision-context *latch* race simulation. The protection against a
mid-decision server update is no longer a stash pinned on the battle — it is the
immutable :class:`LegalActions` snapshot captured at observation time. The masker builds
the mask from that snapshot, and the mapper decodes the chosen action against the SAME
snapshot. So even if poke-env processes a background message and mutates ``last_request``
while the model "thinks", the decode is unaffected: it never re-reads the battle.

This script captures a snapshot, then corrupts the underlying request (clears it, or
swaps in a different one), and asserts that decoding every originally-legal action against
the snapshot still succeeds and yields the SAME choice — i.e. the snapshot provides 100%
protection, exactly as the latch used to.
"""

import sys
import os

import numpy as np

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.battle.live_view import LegalActions


class _FakeMove:
    def __init__(self, move_id):
        self.id = move_id


class _FakeMon:
    def __init__(self, species):
        self.species = species


class _FakeBattle:
    """Exposes exactly the surface ``LegalActions.from_battle`` reads."""

    def __init__(self):
        self.turn = 1
        self.last_request = None
        self.team = {}
        self.available_switches = []
        self.available_moves = []
        self.force_switch = False
        self.trapped = False
        self.maybe_trapped = False
        self.wait = False


_SPECIES = ["Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon", "Charizard"]
_MOVES = ["tackle", "growl", "leechseed", "vinewhip"]


def _make_battle(turn: int) -> _FakeBattle:
    """A random legal Gen-3-ish decision: 6 mons, some switchable, 4 moves some disabled,
    occasionally a forced switch."""
    b = _FakeBattle()
    b.turn = turn
    mons = [_FakeMon(s) for s in _SPECIES]
    b.team = {m.species: m for m in mons}

    forced = np.random.random() < 0.1
    if forced:
        b.force_switch = True
        b.available_switches = [mons[i] for i in range(1, 6) if np.random.random() < 0.8]
        b.available_moves = []
        b.last_request = {"forceSwitch": [True], "turn": turn}
        return b

    # Normal move-selection turn.
    b.available_switches = [mons[i] for i in range(1, 6) if np.random.random() < 0.5]
    disabled = [np.random.random() < 0.1 for _ in _MOVES]
    b.available_moves = [
        _FakeMove(m) for m, d in zip(_MOVES, disabled) if not d
    ]
    b.last_request = {
        "active": [{"moves": [
            {"id": m, "disabled": d} for m, d in zip(_MOVES, disabled)
        ]}],
        "turn": turn,
    }
    return b


def run_unit_fuzz(iterations: int = 2000):
    print(f"🚀 Starting Snapshot-Immutability Simulation: {iterations} iterations...")

    total_validations = 0
    snapshot_saves = 0

    for i in range(iterations):
        battle = _make_battle(turn=i + 1)

        # --- Capture the immutable legality snapshot + build the mask from it ---
        legal = LegalActions.from_battle(battle)
        mask = Gen3ActionMasker.mask_from_legal(legal)
        valid_indices = list(np.where(mask == 1)[0])

        # Decode every legal action against the snapshot BEFORE any corruption.
        baseline = {idx: Gen3ActionMapper.action_to_choice(int(idx), legal) for idx in valid_indices}

        # --- The race: corrupt the underlying request while the model "thinks" ---
        corrupted = False
        if np.random.random() < 0.2:
            battle.last_request = None  # server cleared the request
            corrupted = True
        elif np.random.random() < 0.2:
            battle.last_request = _make_battle(turn=i + 1).last_request  # a different request
            corrupted = True

        # --- Decode again against the SAME snapshot — must be identical ---
        try:
            total_validations += 1
            for idx in valid_indices:
                again = Gen3ActionMapper.action_to_choice(int(idx), legal)
                if again != baseline[idx]:
                    raise AssertionError(
                        f"Snapshot decode drifted for action {idx}: "
                        f"{baseline[idx]} -> {again}"
                    )
            if corrupted:
                snapshot_saves += 1
        except Exception as e:
            print(f"\n🛑 [CRITICAL BUG] Snapshot failed to protect turn {i + 1}: {e}")
            sys.exit(1)

    print("\n" + "=" * 40)
    print("✅ Snapshot Immutability Simulation Completed")
    print(f"Total Turn-Decisions Analyzed: {total_validations}")
    print(f"Mid-Decision Corruptions Survived: {snapshot_saves}")
    print("The captured LegalActions snapshot provided 100% protection against "
          "mid-turn request corruption.")
    print("=" * 40)


if __name__ == "__main__":
    run_unit_fuzz()
