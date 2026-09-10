"""The DENSE AUXILIARY targets (`gen3_dense_aux_v1`) — where the 25 numbers come from.

`agents.model.dense_aux_head` owns the OUTPUT layout; this module owns the LABELS that layout is
scored against, and the two masks that say which of them are real.

**THE BACK-FILL IS THE WIN BIT'S.** These are END-OF-BATTLE facts — who was still standing, at what
HP, how many turns later — copied back to every state of the episode exactly the way
`WinProbLabelCallback` copies the 0/1 outcome, and masked out for the trailing in-progress episode
for exactly the same reason (never trained toward a fabricated label). What differs is that a state
gets 25 of them instead of 1, and 24 of the 25 are facts ABOUT A NAMED ENTITY that the state's own
observation also carries — which is the entire mechanism: a per-entity target puts gradient on the
per-entity axes, where one pooled bit puts none.

**TWO MASKS, ANDed.** A target slot is scored only where BOTH hold:

* **TERMINAL AVAILABILITY** — the slot has an end-of-battle fact at all. Our six always do. The
  opponent's do NOT: a battle can end with three of their party never revealed, and those slots
  have no observed final HP and no observed fainted flag. They are MASKED, never fabricated as
  "alive at full HP" — that label would be wrong roughly as often as it was right and it would be
  wrong in a direction (the un-revealed mons of a team we beat are disproportionately the ones it
  never got to send out).
* **PER-STATE VISIBILITY** — the slot names an entity IN THIS STATE'S OWN OBSERVATION. Opponent
  slot order is REVEAL order (`battle.opponent_team` is populated as mons appear), so the final
  order is a PREFIX-stable extension of every earlier state's: slot i at turn t is the same mon as
  slot i at termination whenever i < (number revealed at t), and is an all-zero absent slot in the
  observation otherwise. Scoring an unrevealed slot would anchor a label to a feature block that
  encodes nothing — the same "target with no feature correspondence" defect the arm exists to fix,
  reintroduced one level down. So the env emits the state's visibility beside the placeholder and
  the callback ANDs it in.

The turns-left output carries no slot, so it is masked by the episode-known bit alone.

**λ DOES NOT REACH THESE TARGETS.** `--win-prob-lambda` blends the win BCE's target backward
through the episode over the critic's own recorded values. There is no counterpart here and there
must not be one: these are TERMINAL FACTS, not returns, and there is no recorded per-state estimate
of "slot 4's final HP" for a recursion to blend. The λ recursion overwrites `win_target` /
`win_mask` and touches no `aux_*` key, so the precedence is structural rather than conventional —
`dense_aux_lambda_test` pins it. Under λ < 1 the two losses simply coexist: the win term regresses
toward a soft λ-return, the aux terms toward the battle's own ending.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np

from agents.model.dense_aux_head import (
    DENSE_AUX_DIM_OUT,
    DENSE_AUX_HP,
    DENSE_AUX_MAX_TURNS,
    DENSE_AUX_SLOTS,
    DENSE_AUX_SURVIVAL,
    DENSE_AUX_TURNS,
)
from agents.observation.base import ObservationEncoder
from agents.observation.constants import TEAM_SIZE

#: The three obs keys. `aux_target` and `aux_mask` are what `train()` reads; `aux_turn` is a REAL
#: present-state value (this state's turn number) that only the label callback consumes — it is
#: half of `turns_left`, and the env is the only place that half exists.
AUX_TARGET_KEY = "aux_target"
AUX_MASK_KEY = "aux_mask"
AUX_TURN_KEY = "aux_turn"


def _team_lists(battle: Any) -> Tuple[List[Any], List[Any]]:
    """`(ours, theirs)` in the OBSERVATION's own order — through the encoder's own accessor.

    Calling `ObservationEncoder.get_team_list` rather than re-deriving `list(battle.team.values())` is
    what makes "the target and the features agree" a fact rather than a comment: if the encoder
    ever changes how it orders a team, this moves with it in the same commit.
    """
    return (ObservationEncoder.get_team_list(battle, is_opponent=False),
            ObservationEncoder.get_team_list(battle, is_opponent=True))


def scale_turns_left(turns_left: float) -> float:
    """`log1p(turns_left) / log1p(DENSE_AUX_MAX_TURNS)`, clipped to [0, 1].

    The log is not decoration. Episode length here is heavily right-skewed with a hard tail at the
    250-turn forfeit cap, so a linear scale would put almost every real target in the bottom tenth
    of the range and leave the head discriminating nothing over the region where nearly all the
    mass is. The clip's upper end can only bind on a state at turn 0 of a maximally long game.
    """
    t = max(0.0, float(turns_left))
    return min(1.0, math.log1p(t) / math.log1p(DENSE_AUX_MAX_TURNS))


def state_visibility(battle: Any) -> np.ndarray:
    """`[DENSE_AUX_DIM_OUT]` — which outputs name an entity THIS STATE'S observation carries.

    1.0 on our occupied party slots (all six, for any legal team) and on the opponent slots
    REVEALED so far; 0.0 on the rest. The survival and HP blocks share one per-slot answer, and the
    turns output is always visible (it names no slot). Emitted every step by `Gen3Env`; the
    callback ANDs it with terminal availability and the episode-known bit.
    """
    vis = np.zeros(DENSE_AUX_DIM_OUT, dtype=np.float32)
    vis[DENSE_AUX_TURNS[0]] = 1.0
    if battle is None:
        return vis
    ours, theirs = _team_lists(battle)
    for i in range(min(TEAM_SIZE, len(ours))):
        vis[DENSE_AUX_SURVIVAL[0] + i] = 1.0
        vis[DENSE_AUX_HP[0] + i] = 1.0
    for i in range(min(TEAM_SIZE, len(theirs))):
        vis[DENSE_AUX_SURVIVAL[0] + TEAM_SIZE + i] = 1.0
        vis[DENSE_AUX_HP[0] + TEAM_SIZE + i] = 1.0
    return vis


def terminal_facts(battle: Any) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """`(target[25], mask[25], terminal_turn)` from the TRAINEE's finished battle view.

    Read off `battle1` at the done step, before the VecEnv auto-resets — the same seam
    `info["win_outcome"]` is published from, and the same view: the opponent's HP here is the
    PUBLICLY known percentage, which is what actually happened and is all a critic could ever be
    asked to predict. Returns `None` when there is no battle to read (never a zero block: an absent
    fact and a zero fact are different, and only the mask may say so).

    The `turns` slot of `target` is left at 0.0 — it is per-STATE (`terminal_turn - this turn`) and
    the callback fills it from `aux_turn`. Its mask bit is set here, because the availability of a
    terminal turn is an episode-level fact like every other entry.
    """
    if battle is None:
        return None
    target = np.zeros(DENSE_AUX_DIM_OUT, dtype=np.float32)
    mask = np.zeros(DENSE_AUX_DIM_OUT, dtype=np.float32)
    ours, theirs = _team_lists(battle)
    for side, team in ((0, ours), (TEAM_SIZE, theirs)):
        for i in range(min(TEAM_SIZE, len(team))):
            mon = team[i]
            if mon is None:
                continue
            k = side + i
            survived = 0.0 if bool(getattr(mon, "fainted", False)) else 1.0
            hp = float(getattr(mon, "current_hp_fraction", 0.0) or 0.0)
            target[DENSE_AUX_SURVIVAL[0] + k] = survived
            # A fainted mon's HP fraction is 0 by definition; taking the two facts from the two
            # accessors independently would let a mid-faint read disagree with itself.
            target[DENSE_AUX_HP[0] + k] = 0.0 if survived == 0.0 else min(1.0, max(0.0, hp))
            mask[DENSE_AUX_SURVIVAL[0] + k] = 1.0
            mask[DENSE_AUX_HP[0] + k] = 1.0
    mask[DENSE_AUX_TURNS[0]] = 1.0
    return target, mask, float(getattr(battle, "turn", 0) or 0)


def ko_counts(survival: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """`(our KOs, their KOs)` DERIVED from a survival row over its UNMASKED slots.

    The count a masked slot cannot contribute to is simply not counted — which is why this is a
    meter and not a loss term (see `dense_aux_head.DENSE_AUX_LAYOUT`'s note).
    """
    s = np.asarray(survival, dtype=np.float64).reshape(-1)[:DENSE_AUX_SLOTS]
    m = np.asarray(mask, dtype=np.float64).reshape(-1)[:DENSE_AUX_SLOTS]
    ours = float((m[:TEAM_SIZE] * (1.0 - s[:TEAM_SIZE])).sum())
    theirs = float((m[TEAM_SIZE:] * (1.0 - s[TEAM_SIZE:])).sum())
    return ours, theirs
