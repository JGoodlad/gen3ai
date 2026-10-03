"""THE CRITIC MODE — which readout is the value function (gen3_winprob_critic_mode_v1).

One string, two values, and it is the answer to a question this tree had two answers to. The
design of record is
[`designs/ai_v12/design_winprob_only_critic.md`](../../../designs/ai_v12/design_winprob_only_critic.md);
its §1 states the problem in one sentence: *the quantity the search's best-measured leaf uses, the
quantity every calibration instrument can score, and the quantity the whole error taxonomy is
written in — is a 0.05-weighted side readout that no gradient reaches from the policy; while the
critic that actually assigns credit predicts a shaped, discounted, PopArt-normalized return that is
commensurable with nothing.*

``shaped`` is that state of affairs: ``_critic_value`` is ``value_net`` in raw return units, and the
win-prob head is an auxiliary BCE. (PopArt, the distributional ``E[Z]`` critic and the aux-BCE
coefficient were DELETED with the shaped critic's levers, deletion pass L1.)

``winprob`` promotes the head: ``V(s) = sigmoid(win_head logit) in [0, 1]``, the value loss IS that
head's BCE against the terminal outcome, and the reward stream is the TERMINAL indicator alone —
so ``V(s)`` is literally ``P(win | s)`` at ``gamma = 1`` with no approximation term.

This module is deliberately **torch-free and import-light**: ``main.checkargs`` promises not to
import torch, and it needs the legal set to validate an argv offline. Everything that knows *which*
modules a mode builds lives at the sites that build them.
"""
from __future__ import annotations

#: The scalar `value_net` in raw return units.
CRITIC_SHAPED = "shaped"

#: The win-prob head IS the critic: `V(s) = sigmoid(logit) in [0, 1]`, trained by BCE against the
#: terminal outcome, with a TERMINAL-indicator reward stream.
CRITIC_WINPROB = "winprob"

#: Every critic a checkpoint can CARRY — what `Gen3DualHeadMaskablePolicy` accepts and what an old
#: `model_config.json` / saved `policy_kwargs` may record. `shaped` stays here so every pre-v109
#: checkpoint (and every shaped-era opponent in a pool) still LOADS.
CRITIC_MODES = (CRITIC_SHAPED, CRITIC_WINPROB)

#: THE critic a run TRAINS: the only trainable one (`shaped` was the Python env core's, deleted in deletion pass
#: U3, 2026-10-02; a shaped CHECKPOINT is refused on a resume / fork, D4: run it PINNED to its own commit). Since
#: deletion pass P11b there is no `--critic` flag — this value is the `critic` constant of every trainer
#: namespace (`main.train.parser.objective`), and the bare `--debug` smoke runs it on the production env core.
CRITIC_DEFAULT = CRITIC_WINPROB

#: What an ABSENT record means — a `model_config.json` / saved `policy_kwargs` / policy attribute that
#: never carried the key was written before `--critic` existed (pre-v109), when the only critic was
#: the shaped one. This is the only possible past and it does NOT follow `CRITIC_DEFAULT`: reading an
#: absent record as the new default would load every pre-v109 checkpoint as a probability critic.
CRITIC_UNRECORDED = CRITIC_SHAPED


def is_winprob(mode: object) -> bool:
    """Is `mode` the win-prob critic? Accepts anything stringable, so a namespace / config / policy
    attribute read with a `getattr(..., 'critic', 'shaped')` default answers without a cast."""
    return str(mode) == CRITIC_WINPROB


#: The win-prob critic's discount — an IDENTITY, not a tuning: with a terminal-only indicator reward
#: and a 250-turn hard cap, V(s) == P(win | s) exactly at gamma = 1. The `gamma` constant of every trainer
#: namespace (`main.train.parser.objective`; the `--gamma` flag was deleted, P11b). A checkpoint's OWN gamma
#: is SB3's (restored on a resume); a shaped-era 0.9999 (`reward_weights.PBRS_GAMMA`) only ever loads.
WINPROB_GAMMA = 1.0
