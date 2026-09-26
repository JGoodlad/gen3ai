"""The reward's MAGNITUDES — the terminal's default scale, the timeout cap and the default gamma.

Everything the hand-shaped reward carried (the Φ potentials' weights, every BIAS bonus, tax, clamp
and threshold) was DELETED with the shaped reward path (program_rust_core §4 M3 row, 2026-09-26):
the reward is the TERMINAL alone. What is left is the terminal's own default and the two constants
other modules key on. The per-run, config-driven magnitudes (`victory_value`, `draw_penalty`) live
on ``RewardConfig``.

⚠️ **Changing a value here is a RETRAIN-class change**, not a tuning knob you flip mid-run.
"""
from agents.training.stall import StallConfig as _StallConfig

#: The DEFAULT terminal magnitude (`RewardConfig.victory_value`'s default; pinned equal by
#: `reward_defaults_test.py`). Production runs `--victory-value 1.0 --terminal-indicator`.
VICTORY_VALUE = 30.0

# The turn at which gen3_env forfeits a stalled battle (ForfeitBattleOrder). A terminal at/after this
# turn is a no-progress TIMEOUT (the trainee hit the cap), scored with RewardConfig.draw_penalty —
# kept in sync with the env's stall cap so the reward's timeout test matches where the env forfeits.
_TIMEOUT_TURN_CAP = _StallConfig().threshold

#: The shaped-critic DEFAULT discount (`--gamma` unset, `--critic shaped`) — the historical PPO gamma.
#: The name survives from when the hand potentials required it to equal the PPO gamma; the one PBRS
#: that remains is the win-prob head's (`winprob_pbrs.py`), which reads `model.gamma` directly.
PBRS_GAMMA = 0.9999
