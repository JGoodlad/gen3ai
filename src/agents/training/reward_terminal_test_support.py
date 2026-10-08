"""ONE terminal fold, by outcome NAME — the shared helper for terminal-shape tests.

`gen3_winprob_critic_mode_v1`. The four terminal outcomes this reward distinguishes are not four
flags on a config; they are four BOARDS, and each one has a trap that a hand-built case gets
wrong in a way no assertion catches:

* a **timeout** is a FORFEIT-loss at the turn cap (`lost=True`, `turn >= _TIMEOUT_TURN_CAP`), not
  a tie — `gen3_env` issues a `ForfeitBattleOrder` there, so the terminal detects it by TURN COUNT
  and a `finished`-but-not-`lost` board at turn 250 is a different case entirely;
* a **tie** is `finished` with neither `won` nor `lost`, *before* the cap — it shares the decisive
  loss's branch, which is exactly the conflation `victory_value` had to be threaded through;
* a **decisive loss** has to be well before the cap or it reads as the timeout;

Writing those four boards inline is what makes a terminal test a test of the harness. This module
is the one place they live, so every terminal test (`critic_mode_test`: the win INDICATOR and the
historical ±30 / −35 ordering) folds the SAME boards and any disagreement is about the config, never
about the setup. The fold is `reward_config.terminal_breakdown` — the reward's ONE rule, shared with
the prober's core-trace recorder (the per-turn `Gen3RewardManager` that wrapped it is deleted with
the Python battle layer, T27 P6 slice 6d-2; training's terminal is the Rust core's).

Pure test support — imported by tests only, no production consumer.
"""
from __future__ import annotations

from agents.training.reward_config import terminal_breakdown
from agents.training.reward_weights import _TIMEOUT_TURN_CAP

#: The four terminal boards, as the result meta `terminal_breakdown` reads: `(won, lost, finished, turn)`.
_BOARDS = {
    "win":     (True, False, True, 40),
    "loss":    (False, True, True, 40),
    # A pre-cap TIE: finished, nobody won or lost. It shares the decisive loss's branch.
    "tie":     (False, False, True, 40),
    # The 250-turn stall: a forfeit-LOSS detected by the turn count, not by won/lost.
    "timeout": (False, True, True, _TIMEOUT_TURN_CAP),
}

OUTCOMES = tuple(_BOARDS)


def terminal_reward(config, outcome: str) -> float:
    """The `win_loss` TERMINAL term `terminal_breakdown` folds for `outcome` under `config` — since the
    shaped-reward deletion the only field, so it is also the turn's whole reward."""
    if outcome not in _BOARDS:
        raise KeyError(f"unknown terminal outcome {outcome!r} (want one of {OUTCOMES})")
    won, lost, finished, turn = _BOARDS[outcome]
    return float(terminal_breakdown(config, won=won, lost=lost, finished=finished, turn=turn).win_loss)
