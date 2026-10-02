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

#: The legal set, in `--help` order.
CRITIC_MODES = (CRITIC_SHAPED, CRITIC_WINPROB)

#: The BARE-ARGV default — what a FRESH argv that types no `--critic` resolves to (deletion pass D2,
#: owner 2026-10-02, `designs/ops/deletion_pass_manifest.md` §2.1). `--arch production` applies the same
#: value from the recipe; the bare parser now agrees with it, so the `--debug` smoke runs the production
#: critic on the production env core. Read ONLY where an ARGV is being resolved.
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
#: and a 250-turn hard cap, V(s) == P(win | s) exactly at gamma = 1 (see `--gamma`'s help).
WINPROB_GAMMA = 1.0


def critic_gamma(mode: object) -> float:
    """THE CRITIC -> DISCOUNT PAIRING, declared once: the `--gamma` an UNTYPED flag resolves to
    under `mode`. ``winprob`` -> `WINPROB_GAMMA` (1.0); ``shaped`` -> `reward_weights.PBRS_GAMMA`
    (0.9999, the historical PPO gamma — every shaped and pre-critic run in `models/` trained at it,
    incl. the shaped ladder controls that otherwise took the win-prob reward values).

    Read by `resolve_critic_mode` / `resolve_config` (the launch), `recipe_surface` (a TYPED
    `--critic` under `--arch production` gets ITS critic's discount, never `recipe.fresh`'s) and
    `combination_checks` (`--critic winprob` refuses any other gamma). Lazy import: this module stays
    torch-free, and `reward_weights` is pure constants."""
    if is_winprob(mode):
        return WINPROB_GAMMA
    from agents.training.reward_weights import PBRS_GAMMA
    return float(PBRS_GAMMA)
