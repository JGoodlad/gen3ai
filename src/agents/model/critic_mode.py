"""THE CRITIC MODE — which readout is the value function (gen3_winprob_critic_mode_v1).

One string, two values, and it is the answer to a question this tree had two answers to. The
design of record is
[`designs/ai_v12/design_winprob_only_critic.md`](../../../designs/ai_v12/design_winprob_only_critic.md);
its §1 states the problem in one sentence: *the quantity the search's best-measured leaf uses, the
quantity every calibration instrument can score, and the quantity the whole error taxonomy is
written in — is a 0.05-weighted side readout that no gradient reaches from the policy; while the
critic that actually assigns credit predicts a shaped, discounted, PopArt-normalized return that is
commensurable with nothing.*

``shaped`` is that state of affairs, unchanged and byte-identical: ``_critic_value`` is
``value_net`` (or the distributional head's ``E[Z]`` under ``--value-from-dist``), de-normalized
through PopArt into raw shaped-return units, and the win-prob head is an auxiliary BCE folded at
``--win-prob-coef``.

``winprob`` promotes the head: ``V(s) = sigmoid(win_head logit) in [0, 1]``, the value loss IS that
head's BCE against the terminal outcome, and the reward stream is the TERMINAL indicator alone —
so ``V(s)`` is literally ``P(win | s)`` at ``gamma = 1`` with no approximation term. PopArt has no
job (the payoff set is fixed at {win, not-win}, so there is no scale to track) and is refused.

This module is deliberately **torch-free and import-light**: ``main.checkargs`` promises not to
import torch, and it needs the legal set to validate an argv offline. Everything that knows *which*
modules a mode builds lives at the sites that build them.
"""
from __future__ import annotations

#: Today's critic: the scalar `value_net` (or `E[Z]`) in raw shaped-return units, PopArt-pegged.
CRITIC_SHAPED = "shaped"

#: The win-prob head IS the critic: `V(s) = sigmoid(logit) in [0, 1]`, trained by BCE against the
#: terminal outcome, with a TERMINAL-indicator reward stream and no PopArt.
CRITIC_WINPROB = "winprob"

#: The legal set, in `--help` order. `shaped` is first because it is the default.
CRITIC_MODES = (CRITIC_SHAPED, CRITIC_WINPROB)

#: The default. It stays `shaped` until an arm has run — the switch to `winprob` and the fresh-weights
#: `ARCH_SIGNATURE` bump land together, in a later commit, per the design's §5.1.
CRITIC_DEFAULT = CRITIC_SHAPED


def is_winprob(mode: object) -> bool:
    """Is `mode` the win-prob critic? Accepts anything stringable, so a namespace / config / policy
    attribute read with a `getattr(..., 'critic', 'shaped')` default answers without a cast."""
    return str(mode) == CRITIC_WINPROB
