"""`--obs-facts v1` (`gen3_obs_facts_v1`): the OBS-FACTS block delivered as TOKEN CONTENT (T0).

The observation always carries the block (`agents/observation/obs_facts.py`); the model reads it
only under `--obs-facts v1`, so `off` (production) builds nothing and its forward is the one it was.
Under `v1` each fact is added to the role token of the ENTITY it describes, before the belief stack
and the trunk read them — the per-mon content route (the entity-coverage audit's DYN home):

* SEEN rows (what the opponent has seen of our mon)   → OUR six tokens, row i to slot i;
* VOL rows (Encore / Taunt / Disable / Uproar / trap) → each side's ACTIVE token;
* CHOICE (the opponent active's lock evidence; the stint's first move through the shared move
  embedding — an id never reaches a Linear raw)         → THEIR active token;
* SCREENS (turns left per side)                        → every token of THAT side (side-relative:
  one projection, each side reads its own conditions).

Every projection is ZERO-INIT, so a fresh `v1` build is the `off` network at init (identity-at-init),
and none of it draws from or shifts the global RNG (see `__init__`).
The static-token rebuild re-routes these (SIDE / FIELD tokens, the DYN input); this module is the
lever's own minimal consumer, screened on its own (`design_entity_coverage_audit.md` Decision record).
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from agents.model.arch_constants import D_MODEL
from agents.model.hypothesis_set import IsolatedLinear
from agents.observation.constants import (FACTS_CHOICE_DIM, FACTS_SCREENS, FACTS_SEEN_ROW_DIM,
                                          FACTS_VOL_SIDE_DIM, TEAM_SIZE)

#: The legal `--obs-facts` values.
OBS_FACTS_MODES = ("off", "v1")


class ObsFactsInject(torch.nn.Module):
    """See the module docstring. ``layout`` is the encoder's `get_layout()` (the sub-block offsets)."""

    def __init__(self, layout: Dict[str, Any]):
        super().__init__()
        sub = layout["obs_facts"]
        self._seen = (sub["seen"]["offset"], sub["seen"]["dim"])
        self._choice = (sub["choice"]["offset"], sub["choice"]["dim"])
        self._vol = (sub["vol"]["offset"], sub["vol"]["dim"])
        self._screens = (sub["screens"]["offset"], sub["screens"]["dim"])
        self._n_screens = len(FACTS_SCREENS)
        move_emb = int(layout["move_embedding_dim"])
        # `IsolatedLinear(zero=True)`: exact zeros (identity-at-init), drawn from NO RNG, and not an
        # `nn.Linear`, so SB3's orthogonal re-init neither re-draws them nor shifts its global stream —
        # every OTHER parameter of a `v1` build starts byte-equal to the `off` build at the same seed
        # (the X5 builder's rule, `hypothesis_set.IsolatedLinear`).
        self.seen_proj = IsolatedLinear(FACTS_SEEN_ROW_DIM, D_MODEL, zero=True)
        self.vol_proj = IsolatedLinear(FACTS_VOL_SIDE_DIM, D_MODEL, zero=True)
        # [not_locked_item, not_locked_moves, run] + the first move's embedding
        self.choice_proj = IsolatedLinear(FACTS_CHOICE_DIM - 1 + move_emb, D_MODEL, zero=True)
        self.screens_proj = IsolatedLinear(self._n_screens, D_MODEL, zero=True)

    def forward(self, facts: torch.Tensor, role_tokens: torch.Tensor, our_active_idx: torch.Tensor,
                opp_active_local: torch.Tensor, embeddings: Any) -> torch.Tensor:
        """``facts`` [B, OBS_FACTS_DIM]; ``role_tokens`` [B, 12, D]; the active indices [B]."""
        B = facts.shape[0]
        o, d = self._seen
        seen = facts[:, o:o + d].reshape(B, TEAM_SIZE, FACTS_SEEN_ROW_DIM)
        o, d = self._vol
        vol = facts[:, o:o + d].reshape(B, 2, FACTS_VOL_SIDE_DIM)
        o, d = self._choice
        choice = facts[:, o:o + d]
        o, d = self._screens
        screens = facts[:, o:o + d].reshape(B, 2, self._n_screens)

        batch = torch.arange(B, device=facts.device)
        ours_add = self.seen_proj(seen) + self.screens_proj(screens[:, 0]).unsqueeze(1)    # [B, 6, D]
        opp_add = self.screens_proj(screens[:, 1]).unsqueeze(1).expand(B, TEAM_SIZE, -1)  # [B, 6, D]
        # the actives: their own volatile turns; theirs also the Choice-lock evidence
        our_active = torch.zeros(B, TEAM_SIZE, 1, device=facts.device, dtype=facts.dtype)
        our_active[batch, our_active_idx] = 1.0
        opp_active = torch.zeros(B, TEAM_SIZE, 1, device=facts.device, dtype=facts.dtype)
        opp_active[batch, opp_active_local] = 1.0
        first = embeddings.move_embedding(choice[:, 2].long())                          # [B, move_emb]
        choice_in = torch.cat([choice[:, 0:2], choice[:, 3:4], first], dim=-1)
        ours_add = ours_add + our_active * self.vol_proj(vol[:, 0]).unsqueeze(1)
        opp_add = opp_add + opp_active * (self.vol_proj(vol[:, 1]) + self.choice_proj(choice_in)).unsqueeze(1)
        return role_tokens + torch.cat([ours_add, opp_add], dim=1)
