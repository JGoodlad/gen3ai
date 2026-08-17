"""Attention pools over the trunk tokens: CLSPool (the pi/vf board summaries) and HiddenOppBeliefPool.

Split out of `features_extractor.py` 2026-08-16 (one responsibility per file); that module
re-exports every name here, so historical import paths still resolve.
"""
from agents.model.extractor_ctx import ExtractorContext
import torch
from typing import Dict, Any, Optional, Tuple
from agents.model.value_threat_inject import (ValueThreatInject)
from agents.model.arch_constants import (D_MODEL,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_FFN_DIM,
)




class CLSPool(torch.nn.Module):
    """Per-side CLS cross-attention pools plus a value-dedicated pool.

    `our_cls`/`their_cls` each attend over their side's 6 post-transformer team tokens
    (fainted slots key-masked) to produce the policy-facing team summaries, and we also
    extract our active Pokémon's refined token. `value_cls` is a third learned query that
    attends over ALL 12 team tokens (both sides) to produce a global "who's winning"
    summary for the value head — a different aggregation than the policy's our-active-centric
    view. History/global information has already flowed into the team tokens via the unified
    transformer, so pooling over the 12 team tokens gives the value query a whole-board read.
    """

    def __init__(self, layout: Dict[str, Any], value_threat_inject_dim: int = 0):
        super().__init__()
        self.our_cls = torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.their_cls = torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.our_cls_attn = torch.nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=TRANSFORMER_N_HEADS, batch_first=True)
        self.their_cls_attn = torch.nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=TRANSFORMER_N_HEADS, batch_first=True)
        self.norm_pool_our = torch.nn.LayerNorm(D_MODEL)
        self.norm_pool_their = torch.nn.LayerNorm(D_MODEL)

        # Value-dedicated CLS pool (Option C): a separate readout for the critic so the
        # shared body's representation is summarised through a value-specific lens.
        self.value_cls = torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.value_cls_attn = torch.nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=TRANSFORMER_N_HEADS, batch_first=True)
        self.norm_pool_value = torch.nn.LayerNorm(D_MODEL)

        # gen3_value_threat_inject_v1: the op's per-our-mon reduced threat row as TOKEN CONTENT on
        # the value pool's copy of our tokens. Owned HERE rather than by the extractor so the
        # augmented tensor is a LOCAL, and no policy-facing read can reach it by construction —
        # that is what makes V1 ("pi bit-identical for arbitrary W_inj") a structural property
        # instead of a discipline the next edit could break.
        self.value_threat_proj = (
            ValueThreatInject(value_threat_inject_dim, D_MODEL)
            if value_threat_inject_dim else None)

    def forward(self, our_team_out: torch.Tensor, their_team_out: torch.Tensor,
                ctx: ExtractorContext,
                threat_rows: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """our/their_team_out [B,6,D_MODEL]; `threat_rows` the op's reduced per-mon rows (required
        only when value_threat_inject is built) → `(our_team_pooled, their_team_pooled,
        our_active_refined, value_pooled)`, each [B, D_MODEL]."""
        batch_size = ctx.batch_size
        our_cls_q   = self.our_cls.expand(batch_size, -1, -1)
        their_cls_q = self.their_cls.expand(batch_size, -1, -1)
        our_pool_out, _   = self.our_cls_attn(our_cls_q,   our_team_out,   our_team_out,
                                              key_padding_mask=ctx.fainted_mask_ours)
        their_pool_out, _ = self.their_cls_attn(their_cls_q, their_team_out, their_team_out,
                                                key_padding_mask=ctx.fainted_mask_opp)
        our_team_pooled   = self.norm_pool_our(our_pool_out).squeeze(1)             # [B, 128]
        their_team_pooled = self.norm_pool_their(their_pool_out).squeeze(1)         # [B, 128]

        batch_idx = torch.arange(batch_size, device=ctx.device)
        our_active_refined = our_team_out[batch_idx, ctx.our_active_idx]            # [B, 128]

        # Value pool: one query over both teams' 12 tokens (fainted slots key-masked).
        # gen3_value_threat_inject_v1: ONLY this pool's copy of our tokens carries the op's
        # per-entity threat magnitude. `our_team_pooled` / `our_active_refined` above, and the
        # pointer head downstream, all read the untouched `our_team_out` — so the policy is
        # provably blind to `W_inj` (V1) and this arm moves the critic alone.
        our_for_value = our_team_out
        if self.value_threat_proj is not None:
            if threat_rows is None:
                raise ValueError(
                    "value_threat_inject is built but the op supplied no reduced rows — the "
                    "DamageOperator must run with reduce_how=VALUE_THREAT_INJECT_REDUCE_HOW so "
                    "`last_reduced_extra` is populated. A silent skip here would make the flag a "
                    "no-op that still passes every shape test.")
            our_for_value = self.value_threat_proj(our_team_out, threat_rows)
        all_team_out = torch.cat([our_for_value, their_team_out], dim=1)            # [B, 12, 128]
        value_cls_q  = self.value_cls.expand(batch_size, -1, -1)
        value_pool_out, _ = self.value_cls_attn(value_cls_q, all_team_out, all_team_out,
                                                key_padding_mask=ctx.all_fainted)
        value_pooled = self.norm_pool_value(value_pool_out).squeeze(1)              # [B, 128]

        return our_team_pooled, their_team_pooled, our_active_refined, value_pooled


class HiddenOppBeliefPool(torch.nn.Module):
    """Hidden-opponent belief: K distinct learned query tokens (DETR object-query / Slot-Attention
    style) that read the post-transformer team tokens and summarise the belief over the opponent's
    still-hidden party.

    Why this exists: once the unrevealed opp slots are unmasked (`attend_unrevealed_opponents`), the
    N unknown slots are *identical* (zeros + `species_known=0`); a permutation-equivariant transformer
    is forced to map identical inputs to identical outputs, so they collapse to one representation —
    the model can know "there are unknowns" but can't represent "slot A leans physical sweeper, slot B
    leans special wall". K learned queries are non-identical *by construction* (independent init), so
    they break that symmetry and can specialise.

    - **K=1** is a single "hidden-opponent CLS" — a set-summary of the whole unrevealed remainder
      (sibling to `our_cls`/`their_cls`/`value_cls`), holding a multimodal belief implicitly without
      ever materialising a phantom mean-mon.
    - **K>1** gives distinct per-slot queries that **coordinate** (the decoder's self-attention lets
      them attend to each other — "board is rock-leaning → I, the 2nd slot, lean rock too" = overload)
      and **read the board** (cross-attention to the 12 team tokens). For K=1 the self-attention is a
      benign near-identity.

    Output `[B, K*D_MODEL]` is concatenated into BOTH the policy and value projection inputs. Hard
    requires `attend_unrevealed_opponents`: with the hidden slots masked there is nothing for the
    belief to summarise. Untrained-capacity caveat: without a dedicated objective (B3 — species-ID /
    BYOL) the RL gradient only weakly shapes these queries; this module is the *structure* those
    objectives later attach to. See `designs/ai_v5/design_offense_and_opponent_belief.md` §B2."""

    def __init__(self, k: int):
        super().__init__()
        if k < 1:
            raise ValueError(f"HiddenOppBeliefPool needs k >= 1, got {k}")
        self.k = k
        # K distinct learned queries — non-identical by construction (independent init) so they do
        # not collapse the way identical zero-slots would. Same 0.02 init scale as the CLSPool queries.
        self.queries = torch.nn.Parameter(torch.randn(1, k, D_MODEL) * 0.02)
        # DETR-style decoder block: query self-attention (coordinate) → cross-attention to the 12
        # team tokens (read the board) → FFN. dropout=0.0 / post-LN match the encoder stack.
        self.decoder = torch.nn.TransformerDecoderLayer(
            d_model=D_MODEL, nhead=TRANSFORMER_N_HEADS, dim_feedforward=TRANSFORMER_FFN_DIM,
            dropout=0.0, activation="relu", batch_first=True, norm_first=False,
        )
        self.norm = torch.nn.LayerNorm(D_MODEL)

    def forward(self, all_team_out: torch.Tensor, all_fainted: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        """all_team_out [B, 12, D_MODEL], all_fainted [B, 12] bool key-mask → [B, K*D_MODEL].

        `all_fainted` always has >=2 False entries (our + opp active are force-unmasked in
        ObsUnpack), so no memory row is fully masked → no attention NaN."""
        queries = self.queries.expand(batch_size, -1, -1)                         # [B, K, D_MODEL]
        belief = self.decoder(queries, all_team_out, memory_key_padding_mask=all_fainted)
        belief = self.norm(belief)                                                # [B, K, D_MODEL]
        return belief.reshape(batch_size, self.k * D_MODEL)                       # [B, K*D_MODEL]
