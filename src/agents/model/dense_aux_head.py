"""`DenseAuxHead` — the DENSE AUXILIARY readout off `value_pooled` (`gen3_dense_aux_v1`).

Arm 9 of the critic ladder (`designs/research_state/winprob_critic_ladder_2026-09-08.md`).

**THE DEFECT IT IS BUILT AGAINST.** Under `--critic winprob` the value loss is a BCE against ONE
terminal bit copied to every state of the episode, and the head refit
(`winprob_head_refit_2026-09-09`) proved the critic's conditional miscalibration is a TARGET defect
rather than a head defect: only ~10 % of that label's variance lies BETWEEN opponents, so an
early-stopped learner minimising a proper scoring rule shrinks the weak axes — opponent, own team —
toward the marginal although its features carry them. Four 10M levers moved nothing at ±0.01 on bot
resolution (ledger *THE ARMS AT 400 GAMES*).

**THE LITERATURE'S ANSWER, and ours.** KataGo (Wu 2019, §3) reports a large gain in learning
efficiency from AUXILIARY targets that share the win's cause — ownership of every point, and the
final score, beside the win. Our analogue of "ownership of every point" is **per-Pokémon
end-of-battle outcomes**: which of the twelve slots was still standing when the battle ended, at
what HP, and how much longer it lasted. Those are 25 numbers per state instead of one bit, and each
one is a fact ABOUT A NAMED ENTITY — so the gradient they carry runs along exactly the per-entity
axes the win bit cannot separate.

**WHAT THIS MODULE IS.** One hidden layer over `value_pooled` → 25 logits, in the layout
`DENSE_AUX_LAYOUT` below. Every output is a SIGMOID logit trained with binary cross-entropy against
a target in [0, 1] — including the two that are not Bernoulli means (final HP fraction, scaled
turns-left). That uniformity is a decision with three reasons and is defended in
`designs/training/critic_and_value_losses.md`; the short form is that a sigmoid keeps a
bounded-in-[0,1] target in range BY CONSTRUCTION (no clamping, and no MSE-on-a-saturating-sigmoid
vanishing gradient exactly where the HP mass sits, at 0 and 1), BCE with a soft target is a proper
scoring rule for a [0, 1] mean, and one loss family puts all three terms in the same nats scale so
the single coefficient means one thing across them.

**IT IS NOT CALLED BY THE FORWARD.** `CfEvidentialHead`'s contract, for its reason: the head feeds
nothing forward, so a rollout, an eval and the prober pay nothing for it, and pi/vf are
BIT-IDENTICAL at an arbitrary weight in this module rather than merely at init. The training term
(`instrumented_ppo._dense_aux_term`) applies it to the `value_pooled` the minibatch's
`evaluate_actions` forward stashed.

**ITS INPUT IS NOT DETACHED, and that IS the arm.** This is the one place it departs from the four
counterfactual readouts. The whole point of a dense auxiliary target is that its gradient reaches
the TRUNK along the axes the win bit cannot carry — exactly as the win-prob loss does under
`shaping` (which `--critic winprob` implies). "Never touches pi" therefore means what it means for
the win head: the head's OUTPUT never enters the pi path, and its gradient reaches the shared trunk
parameters that both heads read. `dense_aux_head_test.py` asserts the forward half at a large
random weight and the backward half in both directions.

**ZERO-INIT output** (weight and bias) ⇒ every logit is exactly 0 ⇒ p = 0.5 on all 25 outputs ⇒ the
untrained head predicts the honest state of knowledge, and — because the gradient into the hidden
activations is `(p - y) @ W_out` — the trunk feels nothing until `W_out` itself has moved. It is
covered by `restore_identity_init`'s by-observation capture set automatically (the M1 contract),
and the test proves the automatic coverage actually reached it.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from agents.model.arch_constants import D_MODEL, DENSE_AUX_HIDDEN
from agents.observation.constants import TEAM_SIZE

# ── THE TARGET LAYOUT ────────────────────────────────────────────────────────────────────────────
# ONE declaration, imported by the head, by the label builder (`agents.training.dense_aux`), by the
# loss (`instrumented_ppo.value_terms`) and by every test. Slot order is the OBSERVATION's own team
# order — our slots first (`battle.team` insertion order, which is what
# `Gen3ObservationEncoder.get_team_list(battle, is_opponent=False)` returns and what the flat
# vector's OUR-TEAM loop indexes), then the opponent's (`battle.opponent_team` REVEAL order, the
# same list the OPP-TEAM loop indexes). Nothing here may re-derive that order: the label builder
# calls the encoder's own `get_team_list`, so the target and the features cannot disagree.
DENSE_AUX_SLOTS = 2 * TEAM_SIZE                     # 12 — our 6 then theirs 6

#: `[start, stop)` of the SURVIVAL block: did slot k end the battle un-fainted? BCE, targets in
#: {0, 1}.
DENSE_AUX_SURVIVAL = (0, DENSE_AUX_SLOTS)
#: `[start, stop)` of the FINAL-HP block: slot k's HP fraction at termination, in [0, 1]. BCE
#: against the soft target.
DENSE_AUX_HP = (DENSE_AUX_SLOTS, 2 * DENSE_AUX_SLOTS)
#: The single TURNS-LEFT output: `log1p(terminal_turn - this_turn) / log1p(DENSE_AUX_MAX_TURNS)`,
#: clipped to [0, 1]. The log compresses a distribution whose tail is the 250-turn cap; the scale
#: constant is that cap, so the target is 1.0 only for a state at turn 0 of a maximally long game.
DENSE_AUX_TURNS = (2 * DENSE_AUX_SLOTS, 2 * DENSE_AUX_SLOTS + 1)
DENSE_AUX_DIM_OUT = DENSE_AUX_TURNS[1]              # 25

#: The stall cap the scaling uses. It is the sim's own 250-turn forfeit horizon, so a target of 1.0
#: is unreachable in practice and the whole [0, 1] range is spanned by real games.
DENSE_AUX_MAX_TURNS = 250.0

#: `{name: (start, stop)}` — the three terms, for the loss and the metrics.
DENSE_AUX_LAYOUT: Dict[str, Tuple[int, int]] = {
    "survival": DENSE_AUX_SURVIVAL,
    "hp": DENSE_AUX_HP,
    "turns": DENSE_AUX_TURNS,
}

#: The KO COUNTS are DERIVED, never predicted. `6 - sum(survived)` per side is a linear function of
#: the survival block, so a separate output would be a linearly-dependent target that adds no
#: information and one more term to normalise — and it is the ONE target that could not honour the
#: per-slot mask, because a count is a sum over slots some of which are masked out. It is published
#: as a METER (`win_prob/aux_ko_mae_*`) computed from the survival head over the UNMASKED slots.


class DenseAuxHead(torch.nn.Module):
    """`[B, D_MODEL] value_pooled` → `[B, DENSE_AUX_DIM_OUT]` sigmoid logits."""

    def __init__(self, in_dim: int = D_MODEL, hidden: int = DENSE_AUX_HIDDEN,
                 out_dim: int = DENSE_AUX_DIM_OUT):
        super().__init__()
        self.out_dim = int(out_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(in_dim), int(hidden)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden), self.out_dim),
        )
        # Zero-init the OUTPUT layer only: every logit is exactly 0 at a cold start, so the head
        # predicts p = 0.5 everywhere — the honest state of knowledge for a head that has seen no
        # label — and the trunk receives no gradient from it until `W_out` has moved off zero.
        torch.nn.init.zeros_(self.net[2].weight)
        torch.nn.init.zeros_(self.net[2].bias)

    def forward(self, value_pooled: torch.Tensor) -> torch.Tensor:
        if value_pooled.dim() != 2:
            raise ValueError(
                f"DenseAuxHead expected [B, D], got {tuple(value_pooled.shape)} — its only input "
                "is the extractor's `value_pooled` stash.")
        return self.net(value_pooled)               # type: ignore[no-any-return]
