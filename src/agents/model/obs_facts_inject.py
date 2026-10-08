"""`--obs-facts v1` (`gen3_obs_facts_v1`): the OBS-FACTS block delivered as TOKEN CONTENT (T0).

The observation always carries the block (`agents/observation/obs_facts.py`, the observation's LAST
block at `constants.OFFSET_OBS_FACTS`); the model reads it only under `--obs-facts v1`, so `off`
(production) builds nothing and its forward is the one it was. Under `v1` each fact is added to the
token of the ENTITY it describes, right after the per-mon encoder and before the belief stack and the
trunk read the tokens — the per-mon content route (the entity-coverage audit's DYN home):

* SEEN rows (what the opponent has seen of our mon)   → OUR six tokens, row i to slot i;
* VOL rows (Encore / Taunt / Disable / Uproar / trap) → each side's ACTIVE token;
* CHOICE (the opponent active's lock evidence; the stint's first move through the shared move
  embedding — an id never reaches a Linear raw)         → THEIR active token;
* SCREENS (turns left per side) — a SIDE fact, so its home depends on `--token-encoding`
  (`FACTS_TOKEN_CLASS`): under `legacy` every token of THAT side (side-relative: one projection, each
  side reads its own conditions), legacy's own per-mon board context; under `static` the side's BOARD
  token (`board_tokens.py`'s OUR SIDE / THEIR SIDE), never a per-mon token — the static encoder's rule
  that no board fact enters a mon's token (`designs/endstate/design_static_tokens.md` §1).

Every projection is ZERO-INIT, so a fresh `v1` build is the `off` network at init (identity-at-init),
and none of it draws from or shifts the global RNG (see `__init__`).

A HIDDEN opponent slot's token is its X5 hypothesis token, spliced in AFTER this injection: no fact of
the block describes a hidden mon except its side's screens, which under `legacy` it then does not carry
(the hypothesis token is a function of the species and its row; the trunk's attention reaches the
revealed tokens that do) and under `static` it never would (the side token holds them).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from agents.model.arch_constants import D_MODEL
from agents.model.hypothesis_set import IsolatedLinear
from agents.observation.constants import (FACTS_CHOICE_DIM, FACTS_SCREENS, FACTS_SEEN_ROW_DIM,
                                          FACTS_VOL_SIDE_DIM, TEAM_SIZE)

#: The legal `--obs-facts` values.
OBS_FACTS_MODES = ("off", "v1")

#: Each sub-block's CLASS — where its content may go under `--token-encoding static` (the per-mon column
#: rule of `static_tokens.py`): ``D`` = the mon's own dynamic state (added to that mon's token after the
#: encoder, so the static identity S never reads it); ``SIDE`` = a fact about one side (its board token,
#: never a per-mon token). A new sub-block of the OBS-FACTS block is classified HERE in the same pass —
#: `obs_facts_inject_test.py` fails on an unclassified one.
FACTS_TOKEN_CLASS: Dict[str, str] = {"seen": "D", "choice": "D", "vol": "D", "screens": "SIDE"}


class ObsFactsInject(torch.nn.Module):
    """See the module docstring. ``layout`` is the encoder's `get_layout()` (the sub-block offsets);
    ``token_encoding`` decides where the SIDE-class facts go."""

    def __init__(self, layout: Dict[str, Any], token_encoding: str = "legacy"):
        super().__init__()
        sub = layout["obs_facts"]
        unclassified = sorted(set(sub) - set(FACTS_TOKEN_CLASS))
        if unclassified:
            raise ValueError(f"OBS-FACTS sub-blocks with no FACTS_TOKEN_CLASS: {unclassified}")
        self._seen = (sub["seen"]["offset"], sub["seen"]["dim"])
        self._choice = (sub["choice"]["offset"], sub["choice"]["dim"])
        self._vol = (sub["vol"]["offset"], sub["vol"]["dim"])
        self._screens = (sub["screens"]["offset"], sub["screens"]["dim"])
        self._n_screens = len(FACTS_SCREENS)
        #: `static`: the SIDE-class facts go to the side BOARD tokens (`side_rows`), not the per-mon tokens.
        self.side_to_board = token_encoding == "static"
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

    def _screen_rows(self, facts: torch.Tensor) -> torch.Tensor:
        o, d = self._screens
        return facts[:, o:o + d].reshape(facts.shape[0], 2, self._n_screens)

    def side_rows(self, facts: torch.Tensor) -> torch.Tensor:
        """[B, 2, D_MODEL]: the SIDE-class content (each side's screen turns, one shared projection) for the
        static OUR SIDE / THEIR SIDE board tokens (`TeamTransformer.board_tokens`)."""
        out: torch.Tensor = self.screens_proj(self._screen_rows(facts))
        return out

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

        batch = torch.arange(B, device=facts.device)
        ours_add = self.seen_proj(seen)                                                   # [B, 6, D]
        opp_add: Optional[torch.Tensor] = None
        if not self.side_to_board:
            screens = self._screen_rows(facts)
            ours_add = ours_add + self.screens_proj(screens[:, 0]).unsqueeze(1)
            opp_add = self.screens_proj(screens[:, 1]).unsqueeze(1).expand(B, TEAM_SIZE, -1)
        # the actives: their own volatile turns; theirs also the Choice-lock evidence
        our_active = torch.zeros(B, TEAM_SIZE, 1, device=facts.device, dtype=facts.dtype)
        our_active[batch, our_active_idx] = 1.0
        opp_active = torch.zeros(B, TEAM_SIZE, 1, device=facts.device, dtype=facts.dtype)
        opp_active[batch, opp_active_local] = 1.0
        first = embeddings.move_embedding(choice[:, 2].long())                           # [B, move_emb]
        choice_in = torch.cat([choice[:, 0:2], choice[:, 3:4], first], dim=-1)
        ours_add = ours_add + our_active * self.vol_proj(vol[:, 0]).unsqueeze(1)
        opp_active_add = opp_active * (self.vol_proj(vol[:, 1]) + self.choice_proj(choice_in)).unsqueeze(1)
        opp_add = opp_active_add if opp_add is None else opp_add + opp_active_add
        return role_tokens + torch.cat([ours_add, opp_add], dim=1)
