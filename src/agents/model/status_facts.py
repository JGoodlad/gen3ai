"""THE STATUS FACTS' cure half — `--status-facts exact` (`gen3_endstate_facts_v1`, config v152;
`design_hand_computed_features.md` P2 → §4 rank 4).

`tempo_cost` (production's pair-outcome coordinate) priced a status in TURNS OF OUR CLOCK by the cheapest cure
path it assumed WE would take — our plan, a JUDGMENT by the owner's 2026-10-09 classification test. Whether a
cure path EXISTS is a fact, and so is it on THEIR side (whether our status will stick). `--status-facts exact`
removes `tempo_cost` (and `neutralization`, replaced by the exact burn / paralysis facts in the same two pair
positions, `pair_outcome.STATUS_FACT_COORDS`) and hands every mon, BOTH sides, its cure-availability flags as token
content (zero-init, bias-free `IsolatedLinear(CURE_DIM → D_MODEL)`, built LAST: one-lever init):

* ``cleric_on_side`` — P(a mon of this side that is still alive knows Heal Bell or Aromatherapy) (the mon itself
  included): ours EXACT from our movesets; theirs the noisy-OR over their mons of the mon's presence × its move
  presence (a revealed move 1; a hidden one the fixed-mass move belief; OTHER_species its averaged presence),
  the op's `p_pur_vs_us` convention for "some mon of theirs holds move X". Broadcast to every mon of the side.
  Gen 3: both cure the whole party (`data/moves.ts` healbell / aromatherapy ``onHit`` over ``side.pokemon``).
* ``natural_cure`` — P(the mon's ability is Natural Cure) (ours exact; theirs revealed-exact, else the species'
  Smogon ability prior): the status is shed on switch-out (`data/mods/gen4/abilities.ts` naturalcure).
* ``rest`` — P(the mon knows Rest) (ours exact; theirs its move presence). Rest cures by sleeping itself.
* ``lum_berry`` / ``chesto_berry`` — P(the mon holds it): the item EXACT where it is known (ours always; a revealed
  opponent item; 0 once consumed), the species' Smogon item prior where it is not. Lum cures any major status, Chesto
  sleep, each the moment it lands (`data/items.ts` lumberry / chestoberry ``onUpdate``, both ``gen: 3``).

No turn counts and no weights. Alive-gated; an opponent slot is gated by its concreteness (revealed, or an X5
hypothesis — priced as its species). ONE formula per side, indexed by each mon's own ids: equivariant over a side's
team slots by construction.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import torch

from agents.observation.constants import POKEMON_ITEMS_OFFSET, TEAM_SIZE

if TYPE_CHECKING:
    from agents.model.extractor_ctx import ExtractorContext

#: The per-mon cure-availability columns, in order.
CURE_FACTS: Tuple[str, ...] = ("cleric_on_side", "natural_cure", "rest", "lum_berry", "chesto_berry")
CURE_DIM = len(CURE_FACTS)


def _num(table: Any, ident: str) -> int:
    rec = table.get(ident)
    if rec is None:
        raise ValueError(f"status_facts: {ident!r} did not resolve in the gen-3 data — the flag would silently read "
                         "nothing for it.")
    return int(rec.num)


def build_cure_tables(n_species: int, n_items: int, n_abilities: int, n_moves: int) -> Dict[str, torch.Tensor]:
    """The constant tables (non-persistent buffers): the cure-move indicator columns, the Natural Cure indicator and
    its per-species Smogon marginal, the Lum / Chesto indicators and their per-species Smogon item priors."""
    from agents import gen3_data
    from agents.model.belief_tables import build_item_prior
    from agents.model.damage_tables import build_species_ability_marginal
    moves = torch.zeros(n_moves, 2, dtype=torch.float32)                 # [cleric, rest]
    for mid in ("healbell", "aromatherapy"):
        moves[_num(gen3_data.moves, mid), 0] = 1.0
    moves[_num(gen3_data.moves, "rest"), 1] = 1.0
    nc = torch.zeros(n_abilities, 1, dtype=torch.float32)
    nc[_num(gen3_data.abilities, "naturalcure"), 0] = 1.0
    items = torch.zeros(n_items, 2, dtype=torch.float32)                 # [lum, chesto]
    lum, chesto = _num(gen3_data.items, "lumberry"), _num(gen3_data.items, "chestoberry")
    items[lum, 0] = 1.0
    items[chesto, 1] = 1.0
    prior = build_item_prior(n_species, n_items)
    return {
        "CURE_MOVES": moves,
        "ABILITY_NATURAL_CURE": nc,
        "SPECIES_NATURAL_CURE_PRIOR": build_species_ability_marginal(n_species, nc, neutral=0.0),
        "CURE_ITEMS": items,
        "SPECIES_CURE_ITEM_PRIOR": torch.stack([prior[:, lum], prior[:, chesto]], dim=-1).contiguous(),
    }


class CureFlags(torch.nn.Module):
    """THE cure-availability rule (module docstring). No parameters: constant tables only (no state_dict key, no
    RNG draw). Called with the context the op prices with (X5: the hypothesis context) and the op (its X5 roster)."""

    CURE_MOVES: torch.Tensor
    ABILITY_NATURAL_CURE: torch.Tensor
    SPECIES_NATURAL_CURE_PRIOR: torch.Tensor
    CURE_ITEMS: torch.Tensor
    SPECIES_CURE_ITEM_PRIOR: torch.Tensor

    def __init__(self, layout: Dict[str, Any]) -> None:
        super().__init__()
        for name, t in build_cure_tables(layout["max_species"], layout["max_items"], layout["max_abilities"],
                                         layout["max_moves"]).items():
            self.register_buffer(name, t, persistent=False)
        from agents import gen3_data
        self.healbell_num = _num(gen3_data.moves, "healbell")
        self.aromatherapy_num = _num(gen3_data.moves, "aromatherapy")
        self.rest_num = _num(gen3_data.moves, "rest")

    def _presence(self, w: torch.Tensor) -> torch.Tensor:
        """``[..., 2]`` from a move presence ``w`` ``[..., M]``: [P(Heal Bell or Aromatherapy), P(Rest)] (the two
        cleric moves as independent presences, the noisy-OR the op reads for "holds move X")."""
        cleric = 1.0 - (1.0 - w[..., self.healbell_num]) * (1.0 - w[..., self.aromatherapy_num])
        return torch.stack([cleric, w[..., self.rest_num]], dim=-1)

    def forward(self, ctx: 'ExtractorContext', op: Any, move_belief_logits: torch.Tensor) -> torch.Tensor:
        """[B, 12, CURE_DIM] (`CURE_FACTS`): ours slots 0–5, theirs 6–11."""
        from agents.model.extractor_ctx import ability_known, revealed_ability1_ids
        from agents.model.op_reduction import noisy_or
        pp = ctx.pokemon_part
        dt = pp.dtype
        B = ctx.batch_size
        ar = torch.arange(B, device=ctx.device)
        alive = (ctx.hp_and_active[:, :, 0] > 0).to(dt)                                    # [B,12]
        # --- the moves: ours exact (any of the held move nums), theirs the move presence ---
        # held as a COUNT clamped at 1 (a table read; no max: the F6a rule): [knows a cleric move, knows Rest]
        ours_mv = self.CURE_MOVES[ctx.all_move_ids[:, :TEAM_SIZE]].sum(dim=2).clamp(max=1.0)  # [B,6,2] (exact)
        x5 = op.stash.x5
        if x5 is None:
            w_all = torch.sigmoid(move_belief_logits) * op.HP_CAND_MASK[None, None, :]       # [B,6,M]
            p_mv = self._presence(w_all)                                                    # [B,6,2]
            opp_alive = alive[:, TEAM_SIZE:] * (1.0 - ctx.opp_believed_mask.to(dt))
            concrete = (1.0 - ctx.opp_believed_mask.to(dt))
            slot_pi = opp_alive
            other_cleric = None
        else:
            w = x5.move_w.to(dt)                                                            # [B,6,M]
            p_mv = self._presence(w)                                                        # [B,6,2]
            opp_alive = x5.alive.to(dt)
            concrete = x5.concrete.to(dt)
            slot_pi = x5.slot_pi.to(dt) * opp_alive
            other_cleric = None
            if x5.other is not None and x5.other_any is not None and x5.other_col is not None:
                w_o = x5.other.move_w[ar, x5.other_col].to(dt)                              # [B,M] OTHER's averaged
                p_o = self._presence(w_o)                                                   # [B,2]
                other_cleric = x5.other_any.to(dt) * p_o[:, 0]
        # --- the cleric on each side (the mon itself included), broadcast over the side ---
        ours_cleric = noisy_or(ours_mv[..., 0] * alive[:, :TEAM_SIZE], dim=-1)               # [B] exact 0 / 1
        opp_cleric = noisy_or(slot_pi * p_mv[..., 0], dim=-1)                               # [B]
        if other_cleric is not None:
            opp_cleric = 1.0 - (1.0 - opp_cleric) * (1.0 - other_cleric)
        cleric = torch.cat([ours_cleric[:, None].expand(B, TEAM_SIZE),
                            opp_cleric[:, None].expand(B, TEAM_SIZE)], dim=1)               # [B,12]
        rest = torch.cat([ours_mv[..., 1], p_mv[..., 1]], dim=1)                            # [B,12]
        # --- Natural Cure: known exact, else the species' Smogon marginal (ours are always known) ---
        known = ability_known(ctx).to(dt)
        nc = (known * self.ABILITY_NATURAL_CURE[revealed_ability1_ids(ctx)][..., 0]
              + (1.0 - known) * self.SPECIES_NATURAL_CURE_PRIOR[ctx.species_ids][..., 0])   # [B,12]
        # --- Lum / Chesto: the item where known (0 once consumed), the species prior where not ---
        item_known = pp[..., POKEMON_ITEMS_OFFSET + 1][..., None]
        consumed = pp[..., POKEMON_ITEMS_OFFSET + 2][..., None]
        berry = (item_known * self.CURE_ITEMS[ctx.item_ids] * (1.0 - consumed)
                 + (1.0 - item_known) * self.SPECIES_CURE_ITEM_PRIOR[ctx.species_ids])      # [B,12,2]
        out = torch.stack([cleric, nc, rest, berry[..., 0], berry[..., 1]], dim=-1)          # [B,12,CURE_DIM]
        gate = alive * torch.cat([torch.ones_like(concrete), concrete], dim=1)
        if x5 is not None:
            gate = torch.cat([alive[:, :TEAM_SIZE], opp_alive * concrete], dim=1)
        return out * gate[..., None]


__all__ = ["CURE_FACTS", "CURE_DIM", "build_cure_tables", "CureFlags"]
