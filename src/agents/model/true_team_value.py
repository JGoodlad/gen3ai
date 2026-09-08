"""`TrueTeamValueReadout` — the PRIVILEGED (asymmetric) critic channel (`gen3_value_true_team_v1`).

Arm 5 of the critic ladder (`designs/research_state/winprob_critic_ladder_2026-09-08.md` §L1). Every
other value route in this tree re-reads, re-pools or re-weights the SAME 2501-dim observation both
heads consume; none of them adds information. This one does: it reads the opponent's ACTUAL party —
species, moves, item, ability, derived stats, current HP and status — off a training-and-eval-only
obs key, and delivers a pooled summary into `value_pooled`.

**Why it is a CEILING PROBE and not a shippable feature.** A critic that needs privileged inputs is
not the critic that ships; every downstream use we want (search leaf, distillation target,
calibration instrument) is on the un-privileged net. The arm exists to answer one question — how
much of the win-prob critic's residual error is irreducible uncertainty about the opponent's team? —
by measuring the critic meters with that uncertainty removed. Default OFF, `family=CRITIC`, and it
never enters `designs/production_config.json`.

**vf-ONLY, structurally.** The contribution is yielded from `_value_pooled_routes` and added to
`value_pooled`, and `ProjectionAssembler` returns `pi_combined` as a concat that does not contain
`value_pooled` at all (`vf_combined IS value_pooled`). So `pi` is bit-identical for an ARBITRARY
weight in this module — not merely at init — which is the property that makes the arm readable: a
policy change cannot confound the critic result, because the policy provably cannot see the channel.
`true_team_policy_independence_test.py` asserts it by perturbing the key.

**AUGMENTS rather than REPLACES the belief-keyed opp view, and that is a deliberate choice.**
Replacing the value side's opp tokens with the true team would be the cleaner ceiling in the
abstract, but it would delete the belief route from vf in the same move and confound two changes:
a null could then mean "privilege does not help" OR "the belief route was carrying the signal".
Additive injection leaves every existing value route bit-identical at init and makes the delta
attributable to exactly one thing. It is also the only form the `_value_pooled_routes` seam admits —
additive injection changes no width, so route availability can never mis-size `value_pre_norm`.

**Reuse, not a second encoding.** The block arrives in the obs's OWN per-mon layout
(`TEAM_SIZE × POKEMON_FULL_DIM`, built by `agents.observation.true_team` through the SAME
`PokemonEncoder.encode`), it is sliced by the SAME `slice_pokemon_categoricals` `ObsUnpack` uses,
and its categorical IDs are looked up in the SHARED `Embeddings` tables. What is new here is only
the pooling — six rows to one `D_MODEL` vector.

**Permutation-invariant** over the six rows: one shared per-mon MLP (never a per-slot projection),
then explicit softmax attention. Slot order on this channel is `species num ascending` and carries
no meaning, so an order-sensitive readout would be learning an artifact.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from agents.model.arch_constants import D_MODEL, TTV_DIM, TTV_K
from agents.model.extractor_ctx import slice_pokemon_categoricals
from agents.observation.constants import (
    POKEMON_FULL_DIM,
    POKEMON_SPECIES_KNOWN_OFFSET,
    TEAM_SIZE,
)


def pokemon_id_columns(layout: Dict[str, Any]) -> Tuple[int, ...]:
    """The per-mon block columns that carry EMBEDDING IDS — derived, never restated.

    A raw dex num must never reach a `Linear` (the manifest rule, `observation/constants.py`), so
    the raw half of a per-mon row has to have those columns zeroed. The column set is obtained by
    PROBING `slice_pokemon_categoricals` with a block whose column `c` holds the value `c`: every
    id it returns IS its own column index. That keeps this function in lockstep with the slicer by
    construction — a moved offset or a new id field cannot make the two disagree, which a
    hand-written offset list would do silently on the first layout change.
    """
    probe = torch.arange(POKEMON_FULL_DIM, dtype=torch.float32).view(1, 1, POKEMON_FULL_DIM)
    sliced = slice_pokemon_categoricals(probe, layout)
    cols: set = set()
    for key, val in sliced.items():
        if key.endswith("_ids"):
            cols.update(int(x) for x in val.reshape(-1).tolist())
    return tuple(sorted(c for c in cols if 0 <= c < POKEMON_FULL_DIM))


class TrueTeamValueReadout(torch.nn.Module):
    """`[B, 6, POKEMON_FULL_DIM]` privileged block → `[B, D_MODEL]` added into `value_pooled`."""

    def __init__(self, layout: Dict[str, Any]):
        super().__init__()
        self.layout = layout
        self.num_moves = len(layout['pokemon']['moves']['layout']['slots'])
        id_cols = pokemon_id_columns(layout)
        # A BUFFER, not a python list: it must move with `.to(device)` and survive a state_dict
        # round trip alongside the weights that depend on it.
        self.register_buffer("id_cols", torch.tensor(id_cols, dtype=torch.long))
        row_in = (
            layout['species_embedding_dim']
            + layout['item_embedding_dim']
            + 2 * layout['type_embedding_dim']                       # the mon's type pair
            + 2 * layout['ability_embedding_dim']                    # ability1 + ability2
            + self.num_moves * (layout['move_embedding_dim']
                                + layout['type_embedding_dim'])      # move id + move type, per slot
            + POKEMON_FULL_DIM                                       # the raw row, id columns zeroed
        )
        self.row_in = row_in
        # ONE MLP shared over the six mons — a per-slot projection would reintroduce exactly the
        # positional dependence this channel has no basis for (see the module docstring).
        self.row_mlp = torch.nn.Sequential(
            torch.nn.Linear(row_in, TTV_DIM),
            torch.nn.ReLU(),
            torch.nn.Linear(TTV_DIM, TTV_DIM),
        )
        self.queries = torch.nn.Parameter(torch.randn(TTV_K, TTV_DIM) * (TTV_DIM ** -0.5))
        self.out_proj = torch.nn.Linear(TTV_K * TTV_DIM, D_MODEL)
        # Zero-init ⇒ a cold-start forward adds EXACTLY 0 ⇒ ON is byte-identical to OFF at step 0
        # (the M1 identity-at-init contract). Re-zeroed after SB3's ortho pass by
        # `restore_identity_init`, which captures its set BY OBSERVATION, so this module is covered
        # automatically rather than by being added to a list.
        torch.nn.init.zeros_(self.out_proj.weight)
        torch.nn.init.zeros_(self.out_proj.bias)
        self.out_dim = D_MODEL
        self.last_att: Optional[torch.Tensor] = None    # [B, K, 6] — the diagnostics read

    def forward(self, true_block: torch.Tensor, embeddings: Any) -> torch.Tensor:
        """`true_block` [B, TEAM_SIZE, POKEMON_FULL_DIM] float → `[B, D_MODEL]`.

        Rows whose `species_known` is 0 (an absent slot, or the all-zero block a caller supplies
        when no privileged view exists) are MASKED out of the pool. An all-masked block degrades to
        a uniform average of zero rows rather than a NaN — the explicit-softmax pattern.
        """
        if true_block.dim() != 3 or true_block.shape[1:] != (TEAM_SIZE, POKEMON_FULL_DIM):
            raise ValueError(
                f"TrueTeamValueReadout expected [B, {TEAM_SIZE}, {POKEMON_FULL_DIM}], got "
                f"{tuple(true_block.shape)} — the obs key and this readout disagree about the "
                "per-mon layout, which means the block was not built by "
                "`agents.observation.true_team.build_true_team_block`.")
        block = true_block.to(self.out_proj.weight.dtype)
        ids = slice_pokemon_categoricals(block, self.layout)
        parts = [
            embeddings.species_embedding(ids["species_ids"]),
            embeddings.item_embedding(ids["item_ids"]),
            embeddings.type_embedding(ids["type1_ids"]),
            embeddings.type_embedding(ids["type2_ids"]),
            embeddings.ability_embedding(ids["ability1_ids"]),
            embeddings.ability_embedding(ids["ability2_ids"]),
        ]
        B = block.shape[0]
        # [B, 6, num_moves, emb] → flatten the move axis into the row (the pool is invariant over
        # MONS, not over a mon's own move slots — those are already a fixed, sorted-by-id order).
        parts.append(embeddings.move_embedding(ids["all_move_ids"]).reshape(B, TEAM_SIZE, -1))
        parts.append(embeddings.type_embedding(ids["all_move_type_ids"]).reshape(B, TEAM_SIZE, -1))
        raw = block.clone()
        raw[:, :, self.id_cols] = 0.0        # the manifest rule: no raw dex num reaches a Linear
        parts.append(raw)
        rows = self.row_mlp(torch.cat(parts, dim=2))                       # [B, 6, TTV_DIM]
        masked = block[:, :, POKEMON_SPECIES_KNOWN_OFFSET] < 0.5           # [B, 6] bool
        att = torch.einsum("kd,bnd->bkn", self.queries, rows) * (TTV_DIM ** -0.5)
        att = att.masked_fill(masked[:, None, :], -1e9)
        att = torch.softmax(att, dim=-1)                                   # [B, K, 6]
        self.last_att = att
        out = torch.einsum("bkn,bnd->bkd", att, rows)                      # [B, K, TTV_DIM]
        return self.out_proj(out.reshape(B, TTV_K * TTV_DIM))              # type: ignore[no-any-return]
