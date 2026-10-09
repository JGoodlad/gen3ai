"""Two NARROW per-entity facts for `--token-encoding static` (`gen3_static_port_v1`,
`designs/endstate/design_static_tokens.md` §12), from the static diagnostic
(`designs/research_state/measurements/static_diag_2026-10-09/`, H2 / H3): under `static` a board fact reaches a mon
token only through attention, and mostly at the trunk's LAST layer; two dynamic facts read worse than under legacy
(our side's Spikes on our mon tokens, R² 0.33 vs 0.46 and the one gap that GREW with training; our active's HP on its
move tokens, R² 0.59 vs 0.80). Each fact is a flag of its own, OFF by default:

* **`--mon-hazard-cost on`** — every mon's token (BOTH sides) gets, as D content (`mon_hazard_features`, [B,12,2]):
  its OWN side's Spikes layers (/3) and the fraction of max HP it would lose switching in. The fraction is the damage
  operator's ONE Spikes entry rule (`DamageOperator.spikes_entry`, which the `x` edge cell reads too): 1/8, 1/6, 1/4
  for 1–3 layers, 0 for a Flying type or Levitate (our ability exact; an opponent's exact where revealed, else its
  species' Smogon P(Levitate) — the belief). Under X5 a hidden slot is priced as its HYPOTHESIS (the op's context).
  An opponent slot with no species (the belief-off ablation only: X5 always holds a hypothesis) reads fraction 0 —
  its Spikes column still says the layers are there.
* **`--move-actor-state on`** — our active's 4 E3 move seats get its CURRENT HP fraction and its status one-hot
  (`move_actor_features`, [B, 1 + CONDITION_DIM]): the actor's state on the tokens of the moves it would use,
  which legacy's move network mixed in and static's move tokens lack. Only the E3 seats: the opponent's E4 threat
  seats never carried their active's state under either encoding, so the diagnostic's logic (a fact static
  REMOVED) does not apply to them.

Both arrive through a ZERO-INIT, bias-free `IsolatedLinear` built LAST (no global RNG draw, skipped by SB3's
orthogonal re-init), so every other parameter's initial bytes equal the flag-off build's and an ON build at init
adds exactly 0. Why ADDED to the token rather than fed into D's MLP: a hidden opponent slot's static token is a
GATHER from the dex table (a pure function of the species, `static_tokens.static_hypothesis_tokens`); a board-
dependent column inside D's MLP would end that. As token content the fact is in place at the trunk's FIRST layer,
which is what the diagnostic's H3 (board facts arrive only at the last layer) asks for.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from agents.observation.constants import CONDITION_DIM, POKEMON_CONDITION_OFFSET, POKEMON_HP_OFFSET, TEAM_SIZE

if TYPE_CHECKING:
    from agents.model.extractor_ctx import ExtractorContext

#: The legal values of the two flags. ``off`` builds nothing.
MON_HAZARD_COST_MODES = ("off", "on")
MOVE_ACTOR_STATE_MODES = ("off", "on")

#: [its side's Spikes layers / 3, the fraction of max HP lost switching in].
MON_HAZARD_DIM = 2
#: [current HP fraction] ⊕ the status one-hot [None, BRN, PAR, SLP, FRZ, PSN, TOX].
MOVE_ACTOR_DIM = 1 + CONDITION_DIM


def mon_hazard_features(op: Any, ctx: 'ExtractorContext', opp_concrete: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[B, 12, MON_HAZARD_DIM]: per mon, its OWN side's Spikes layers (/3: ours for slots 0–5, theirs for 6–11) and
    the HP fraction it would lose switching in (``op.spikes_entry``, the op's one rule). ``ctx`` is the context the
    op prices with (under X5 the hypothesis context). ``opp_concrete`` [B,6] (1 = the slot's species is known or
    hypothesised): None ⇒ revealed = not believed. A non-concrete opponent slot's fraction is 0 (unknown types)."""
    chip_our, chip_opp, _gr_i, _gr_j = op.spikes_entry(ctx)                              # [B,6] ×2
    if opp_concrete is None:
        opp_concrete = (~ctx.opp_believed_mask).to(chip_opp.dtype)
    sp = ctx.spikes_feature.to(chip_our.dtype)                                          # [B,2] /3
    ours = torch.stack([sp[:, 0:1].expand(-1, TEAM_SIZE), chip_our], dim=-1)            # [B,6,2]
    theirs = torch.stack([sp[:, 1:2].expand(-1, TEAM_SIZE), chip_opp * opp_concrete.to(chip_opp.dtype)], dim=-1)
    return torch.cat([ours, theirs], dim=1)


def move_actor_features(ctx: 'ExtractorContext') -> torch.Tensor:
    """[B, MOVE_ACTOR_DIM]: our ACTIVE mon's current HP fraction and status one-hot — the actor of the 4 E3 seats."""
    ar = torch.arange(ctx.batch_size, device=ctx.device)
    row = ctx.pokemon_part[ar, ctx.our_active_idx]                                       # [B, POKEMON_FULL_DIM]
    return torch.cat([row[:, POKEMON_HP_OFFSET:POKEMON_HP_OFFSET + 1],
                      row[:, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + CONDITION_DIM]], dim=-1)
