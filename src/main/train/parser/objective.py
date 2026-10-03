"""THE OBJECTIVE'S CONSTANTS — values that were trainer FLAGS until deletion pass P11b.

Each had ONE legal value: the win-prob critic is the only trainable critic (`critic_mode`), and its
identity `V(s) == P(win | s)` holds only at gamma 1 with the unit win-indicator terminal (indicator ON,
victory 1.0, draw 0.0), so a `combination_checks` row refused every other value. They are deleted as flags
(`designs/deleted_flags.md`: a typed one is refused WITH its reason) and live here instead, on the
namespace through `parser.set_defaults`, because ~20 readers take them off `args`
(`RewardConfig.from_args`, `model_build`, the sidecar / callbacks gates, `checkargs`).

Why a namespace default and not "readers use the constant": `RewardConfig.from_args` falls back to the
DATACLASS defaults for an attribute that is absent from `args` — the historical 30.0 / -35.0 / signed
terminal, i.e. the deleted shaped-era reward — so a namespace someone built and forgot to fill would
train the wrong objective SILENTLY. A constant that is always on the namespace cannot be forgotten.

What is NOT here: what an old checkpoint RECORDED. `ModelVersion` still records `critic` and the reward
immutables; a recorded NON-production value is refused WITH a reason on a resume / fork (a recorded
shaped critic by `rust_env_setup.refuse_python_era_checkpoint`, deletion pass D4; a recorded reward by
`ModelVersion.check_reward_config`), never silently replaced by these.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict

from agents.model.critic_mode import CRITIC_DEFAULT, WINPROB_GAMMA

#: ``dest`` -> the one value. `objective_constants_test` pins that no member has a parser option and that every
#: member has a `designs/deleted_flags.md` row.
OBJECTIVE_CONSTANTS: Dict[str, Any] = {
    "critic": CRITIC_DEFAULT,
    # the TERMINAL WIN INDICATOR alone, undiscounted: +1.0 on a win, 0.0 on a loss, a tie AND a 250-turn
    # timeout alike, so the return IS 1{win} and V(s) == P(win | s) with no approximation term
    "terminal_indicator": True,
    "victory_value": 1.0,
    "draw_penalty": 0.0,
    "gamma": WINPROB_GAMMA,
}


def set_objective_constants(parser: argparse.ArgumentParser) -> None:
    """Put the constants on every namespace `parser` parses (no option, so nothing can type them)."""
    parser.set_defaults(**OBJECTIVE_CONSTANTS)
