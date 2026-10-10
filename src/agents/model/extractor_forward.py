"""`ExtractorForward` — the forward PATH: the T0/T1 belief+physics stack and `forward_internal`.

Split out of `features_extractor.py` 2026-08-23 (one responsibility per file). This is the most
consequence-dense code in the tree, so three properties are worth stating where they can be
checked:

* **The stash contract.** `forward_internal` replaces the WHOLE `ExtractorStashes` container at
  ENTRY — that one line is what makes a stale cross-batch read unrepresentable for every field
  at once — and every write goes through `self.stash.<field>`, never a `last_*` name (those are
  read-only properties on `ExtractorApi`, so a stray write raises).
* **The phase ORDER is the contract.** T0 RESOLVE (species / move / spread / HP-type / item
  beliefs) → T1 REASON (the `DamageOperator`, pre-attention) → the trunk → T2 (α/β, the pointer
  cells) → T3 (the pools, the value routes, the side readouts). A consumer moved above its
  producer does not crash; it silently reads a stash from the PREVIOUS forward.
* **`forward` itself stays on `Gen3FeaturesExtractor`**, not here. Both compile flags patch the
  BOUND `fe.forward` and a deliberately-eager pass calls `type(fe).forward`, and
  `instrumented_ppo_test` ASSIGNS `type(fe).forward` — so the concrete class is where that
  attribute has to live for a restore to put it back where it came from.
"""
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

from agents.model.arch_constants import D_MODEL
from agents.model.belief_heads import mask_typeless_hp
from agents.model.extractor_api import ExtractorApi
from agents.model.extractor_ctx import (ExtractorContext, PointerInputs, TOKEN_TYPE_HISTORY,
                                        TOKEN_TYPE_THEIR_TEAM)
from agents.model.extractor_stashes import ExtractorStashes
from agents.model.flat_intent import FlatConsumerOps, append_other, compat_intent_logits, flat_candidates
from agents.model.hypothesis_set import HypothesisSet
from agents.model.hypothesis_encode import gathered_hypothesis_tokens
from agents.model.static_tokens import StaticTokenEncoder, static_hypothesis_tokens
from agents.model.static_facts import (effective_stat_features, mon_hazard_features, move_actor_features,
                                       move_target_features, switch_hazard_features)
from agents.model.hypothesis_tokens import (FixedMassMoves, OppPresence, OpRoster, build_op_roster,
                                            fixed_mass_moves, hypothesis_ctx, key_log_presence,
                                            other_column, other_roster, splice_hypothesis_tokens)
from agents.model.damage_op_layout import _DMG_OMX_IDX_PKO, _SB_SPE
from agents.model.intent_threshold import threshold_probs
from agents.model.move_resolution import gather_ops as gather_move_resolution_ops
from agents.model.pair_outcome import pair_alpha, reduce_pair_in, reduce_pair_in_all
from agents.model.pointer_head import _request_order_move_tokens
from agents.model.team_transformer import _event_reference_cells
from agents.observation.constants import POKEMON_PROTECT_OFFSET, TEAM_SIZE

from agents.model.damage_op import _OUT_SEC_COLS as _OSC
_OUT_SEC_FLINCH_COL = _OSC.index("flinch")   # gen3_intent_conditional_v1: fails at import if dropped


class ExtractorForward(ExtractorApi):
    """The forward path of `Gen3FeaturesExtractor` — see that class."""

    def _typed_hp_posterior(self, opp_tokens: torch.Tensor, ctx: ExtractorContext,
                            raw_move_logits: torch.Tensor
                            ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                       Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compose the raw move posterior into the TYPED-Hidden-Power one → `(typed_logits, presence,
        hp_type_posterior)` (gen3_typed_hp_belief_v1).

        The HP-type head reads THE SAME `opp_tokens` the move head just read, at the same point in the
        forward, so the two halves of `P(HP_t) = presence · P(t)` can never be sourced from differently
        refined tokens. (Before this, the type head lived in `_spread_hp_damage` — which under
        `--move-belief-prefuse` alone runs POST-transformer while the move head runs PRE-transformer, so
        the factors came from two different states of the same slot.)

        Under the **`flat` ABLATION** (`--hp-belief-mode flat`) there is no head: Hidden Power is just
        16 more ordinary move channels that the multi-label move head predicts INDEPENDENTLY, off
        their own real per-typed Smogon usage priors, with no factorisation, no reveal constraint and
        no tracker narrowing. All that survives is masking the bare 237 — which is not a moderation of
        the ablation but a necessity: 237 carries BP 0, so leaving it in the damage candidate set is
        the original "opp HP reads immune" bug, not an arm of the experiment. See the class docstring
        of `HPTypeBelief` for what the ablation is actually testing."""
        if self.hp_type_belief_head is None:                  # flat ablation — HP is an ordinary move
            return mask_typeless_hp(raw_move_logits), None, None, None
        hp_logits, hp_post = self.hp_type_belief_head(opp_tokens, ctx.species_ids[:, TEAM_SIZE:])
        typed, presence = self.hp_type_belief_head.compose_typed_hp(
            raw_move_logits, hp_post,
            ctx.hp_probs[:, TEAM_SIZE:],                     # [B,6,16] tracker narrowing (OPP slots)
            ctx.all_move_ids[:, TEAM_SIZE:, :])              # [B,6,4] revealed ids (rule-out)
        return typed, presence, hp_post, hp_logits

    def _build_hypothesis_species(self, ctx: ExtractorContext, role_pre_belief: torch.Tensor) -> HypothesisSet:
        """gen3_x5_hypothesis_set_v1 (X5 U2) / gen3_x5_belief_tokens_v1 (U3): the T0 hypothesis
        builder's SPECIES half, run BEFORE the move belief (its dex rows become the hidden slots'
        tokens, which the T0 belief heads then read).

        Inputs, all T0: the T0 species prior's LOG-probabilities (recomputed from the same buffers as
        `t0_species_probs`, so the blob path's tensor is untouched), the pre-belief opponent role
        tokens (δ_θ pools the revealed ones), and the TeamTransformer global token's RAW input (δ_θ's
        own projection of it — the transformer's `global_proj` is T1)."""
        from agents.model.t0_species import species_team_prior_logits
        hb = self.hypothesis_builder
        t0 = self.t0_species_prior
        assert hb is not None and t0 is not None
        opp_ids = ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE]
        t0_logp = species_team_prior_logits(
            t0.species_prior_log_marginal, t0.species_prior_log_lift, opp_ids, ctx.opp_believed_mask)
        global_input = torch.cat([ctx.our_ctx_raw, ctx.opp_ctx_raw, ctx.non_matchup_rest], dim=1)
        hs: HypothesisSet = hb.species_set(t0_logp, opp_ids, ctx.opp_believed_mask,
                                           role_pre_belief[:, TEAM_SIZE:], global_input,
                                           self.embeddings.species_embedding)
        return hs

    def _attach_move_group(self, ctx: ExtractorContext, hs: HypothesisSet,
                           mb: Optional[torch.Tensor] = None) -> HypothesisSet:
        """The opponent ACTIVE's move group (U2), after the move belief: its typed posterior at the
        active + the active's species + revealed moves (the active is always revealed). ``mb`` = the
        published move posterior (default: `last_move_belief_logits`)."""
        hb = self.hypothesis_builder
        assert hb is not None
        if mb is None:
            mb = self.last_move_belief_logits
        if mb is None:
            return hs
        opp_ids = ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE]
        bidx = torch.arange(ctx.batch_size, device=ctx.device)
        act = ctx.opp_active_local
        return hb.with_moves(hs, mb[bidx, act], opp_ids[bidx, act],
                             ctx.all_move_ids[:, TEAM_SIZE:, :][bidx, act])

    def _apply_move_belief(self, opp_tokens: torch.Tensor, ctx: ExtractorContext,
                           hctx: Optional[ExtractorContext] = None,
                           hs: Optional[HypothesisSet] = None,
                           ) -> Tuple[torch.Tensor, torch.Tensor, Optional[HypothesisSet],
                                      Optional[FixedMassMoves]]:
        """Predict + reinject the opp moveset into the given opp tokens [B, 6, D] → (enriched, logits).
        ONE call site: PRE-transformer, T0 RESOLVE (gen3_tiered_pipeline_v1 — the POST-transformer
        placement is deleted). The mask selects the slots per move_belief_mode; the
        species/move ids feed prior-fusion (Smogon prior + pin revealed moves certain).

        gen3_typed_hp_belief_v1: the HP-type head + the typed composition run HERE, between the move
        head's read and the reinjection, so the posterior that leaves this method — and therefore the
        one every consumer reads (`last_move_belief_logits`) — is already typed. The reinjection then
        soft-embeds REAL typed moves rather than the typeless 237 row.

        `hctx` (gen3_x5_belief_tokens_v1, X5 only): the hypothesis context —
        a hidden slot holds a CONCRETE species hypothesis, so the move head's prior is THAT species'
        Smogon row (the E10 mixture over the T0 posterior is the blob's stand-in for a hidden slot and
        is not used), and the HP-type head reads the same species. The slot-selection mask and the
        revealed-only HP reinjection read the REAL `ctx`.

        `hs` (fixed_mass, U3 part 3 — ORCHESTRATOR decision on F-X5-26): the opponent ACTIVE's move group
        is built HERE, from the posterior just published, and the reinjection soft-embeds the active's row
        by its DETACHED fixed-mass presence π_m (`FixedMassMoves.w_all`: 1 revealed, π_m, a revealed HP as
        P(t)) instead of its sigmoid inclusion weights — M10's rule, so RL cannot tune the move belief as a
        gate there. The move head keeps training through its own BCE; what is lost is the PPO → move-head
        route through the ACTIVE's reinjection (the other slots' rows keep it). Returns the move-group-
        attached set and its `FixedMassMoves` (None, None with the belief family off)."""
        sctx = hctx if hctx is not None else ctx                 # where the SPECIES / move ids come from
        if self.move_belief_mode == "revealed":
            mb_mask = ~ctx.opp_believed_mask                 # revealed-species slots
        elif self.move_belief_mode == "unrevealed":
            mb_mask = ctx.opp_believed_mask                  # hidden-species slots
        else:                                                # "both"
            mb_mask = torch.ones_like(ctx.opp_believed_mask)
        raw = self.move_belief.move_logits(  # type: ignore[union-attr]
            opp_tokens,
            sctx.species_ids[:, TEAM_SIZE:],                                 # [B, 6]
            sctx.all_move_ids[:, TEAM_SIZE:, :],                             # [B, 6, 4]
            # gen3_hidden_slot_move_mixture_v1 (E10): the hidden slots' prior is the Smogon mixture
            # over the T0 species posterior (None when `t0_species_prior` is off ⇒ the flat row).
            # X5: a hypothesis slot's species is concrete, so no mixture (its own species row).
            hidden_species_probs=(self.stash.t0_species_probs if hctx is None else None),
            opp_believed_mask=(ctx.opp_believed_mask if hctx is None else None))
        logits, presence, hp_post, hp_logits = self._typed_hp_posterior(opp_tokens, sctx, raw)
        # gen3_belief_label_only_v1: register the LIVE tensors for the supervised losses BEFORE
        # publishing. `logits` is the TYPED posterior, so it carries BOTH the move head's and the
        # HP-type head's gradient — which is why the move BCE and the HP CE both keep training under
        # `label_only` while every forward consumer downstream reads the stop-grad publication.
        self.stash.belief_supervision["move_belief_logits"] = logits
        self.stash.belief_supervision["hp_type_logits"] = hp_logits
        self.stash.hp_type_logits = self._publish_belief(hp_logits)
        logits = self._publish_belief(logits)  # type: ignore[assignment]
        hs_m: Optional[HypothesisSet] = None
        fm: Optional[FixedMassMoves] = None
        weights: Optional[torch.Tensor] = None
        if hs is not None:
            hs_m = self._attach_move_group(ctx, hs, logits)
            if hs_m.moves is not None:
                bi = torch.arange(ctx.batch_size, device=ctx.device)
                fm = fixed_mass_moves(hs_m.moves, logits[bi, ctx.opp_active_local])
                act = torch.nn.functional.one_hot(ctx.opp_active_local, TEAM_SIZE).bool().unsqueeze(-1)
                weights = torch.where(act, fm.w_all.to(logits.dtype).unsqueeze(1), torch.sigmoid(logits))
        enriched = self.move_belief.reinject_moves(  # type: ignore[union-attr]
            opp_tokens, mb_mask, self.embeddings.move_embedding, logits, weights=weights)
        # gen3_opp_hp_type_belief_v2: ALSO reinject the presence-gated expected TYPE embedding. This is
        # deliberately not redundant with the move soft-embed above: that one injects believed move
        # IDENTITY (the 355-370 rows), this one injects the believed TYPE in the shared type-embedding
        # space the mon's own types live in — so "this Zapdos threatens ICE" lands in the same geometry
        # attention already uses for type matchups. Revealed slots only. (No head under `flat` — the
        # typed move rows still ride the soft-embed above, which is the point of that ablation.)
        if self.hp_type_belief_head is not None:
            enriched = self.hp_type_belief_head.reinject(
                enriched, hp_post, presence, (~ctx.opp_believed_mask).float(), self.embeddings)  # type: ignore[arg-type]
        return enriched, logits, hs_m, fm

    def _spread_hp_damage(self, opp_tokens: torch.Tensor, ctx: ExtractorContext,
                          hctx: Optional[ExtractorContext] = None,
                          hs: Optional[HypothesisSet] = None,
                          fm: "Optional[FixedMassMoves]" = None,
                          x5r: "Optional[OpRoster]" = None,
                          ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The spread + HP-type belief legs and the FULL DamageOperator, in ONE place.

        `opp_tokens` [B, 6, D] → `(enriched_opp_tokens, damage_block | None)`. ONE call site:
        PRE-transformer (gen3_tiered_pipeline_v1). The beliefs read the raw role tokens, the op runs
        ONCE, and its output both seeds the trunk (see `prefuse_proj`) and feeds every downstream
        consumer. The historical POST-transformer placement — beliefs read from attention-REFINED opp
        tokens — is DELETED.
        Every stash (`last_spread_belief`, `last_hp_type_logits`, `last_move_latent_table`,
        `last_damage_block`) is written here, so the aux losses and the prober read the same tensors.
        """
        # gen3_unified_spread_belief_v1: predict + reinject the opp's hidden SPREAD (revealed slots), and
        # stash the believed stats [B,6,5] for the DamageOperator (consumed at the opp active slot, replacing
        # its hand-coded spread constants) + the speed-supervision loss. Enriches the opp tokens before the
        # CLS pools, like MoveBelief. Hidden slots aren't enriched (their species num 0 → flat prior) and the
        # op only reads the (revealed) active slot.
        # gen3_x5_belief_tokens_v1 (fixed_mass): a hypothesis seat is a CONCRETE species, so the spread
        # and item heads read it with its species (`hctx`) — the spread head enriches revealed AND
        # hypothesis seats (§3.4; unsupervised on hypothesis seats, F-X5-16).
        sctx = hctx if hctx is not None else ctx
        spread_mask = (~ctx.opp_believed_mask if hs is None
                       else (~ctx.opp_believed_mask) | hs.slot_is_hypothesis)
        if self.spread_belief is not None:
            (opp_tokens, _believed, _nat_logits, _ev) = self.spread_belief(
                opp_tokens, spread_mask, sctx.species_ids[:, TEAM_SIZE:])
            # gen3_belief_label_only_v1: the LIVE tensors for the supervised losses, then publish.
            # Cutting `believed` cuts `nature_head`/`ev_head` too — in the generative arm they reach the
            # forward ONLY through it (nat_logits → e_mult → believed → the op; and delta, which the
            # reinject takes, is itself derived from believed). So the nature/EV stashes need no
            # publication of their own; they are registered here for the ONE rule ("a forward-consumed
            # belief head's stashes are published") rather than because a consumer reads them.
            self.stash.belief_supervision["spread_belief"] = _believed
            self.stash.belief_supervision["spread_nature_logits"] = _nat_logits
            self.stash.belief_supervision["spread_ev"] = _ev
            self.stash.spread_belief = self._publish_belief(_believed)
            self.stash.spread_nature_logits = self._publish_belief(_nat_logits)
            self.stash.spread_ev = self._publish_belief(_ev)
        # (no else-clear needed: gen3_extractor_stashes_v1's entry reset left every field None and
        # every supervision key absent)
        # gen3_item_belief_v1 (T0): the hidden-ITEM posterior on the same pre-transformer opp
        # tokens the other T0 beliefs read. The op consumes P(Choice Band) per opp slot (its
        # exactness gating stays op-side); the logits feed the bank's seventh CE row.
        if self.item_belief_head is not None:
            _item_logits, _item_post = self.item_belief_head(
                opp_tokens, sctx.species_ids[:, TEAM_SIZE:])
            self.stash.belief_supervision["item_logits"] = _item_logits
            _item_pub = self._publish_belief(_item_logits)
            self.stash.item_logits = _item_pub
            # the op reads the PUBLICATION (stop-grad under label_only — the one consumer rule),
            # so cutting PPO→belief cuts the value-gradient route through the CB pricing too.
            _item_cb_prob = (torch.softmax(_item_pub, dim=-1)  # type: ignore[arg-type]
                             [:, :, self.damage_op.cb_item_num]
                             if self.damage_op is not None else None)
        else:
            _item_cb_prob = None
        # gen3_speed_physics_v1 (`--speed-physics on` only): the same publication's P(Quick Claw) per opp slot —
        # the op's Quick Claw belief (its exactness gate stays op-side, like CB's). Off passes nothing, and so
        # does a format that BANS Quick Claw (gen3ou, `move_order.quick_claw_live`): no holder exists.
        _op_kw: Dict[str, Any] = {}
        if (self.damage_op is not None and self.damage_op.speed_physics
                and self.damage_op.quick_claw_live and self.item_belief_head is not None):
            _op_kw["item_qc_prob"] = (torch.softmax(self.stash.item_logits, dim=-1)  # type: ignore[arg-type]
                                      [:, :, self.damage_op.qc_item_num])
        # gen3_typed_hp_belief_v1: the opp-HP-TYPE head + its typed composition + its token reinjection all
        # moved UP into `_apply_move_belief`, where the move head reads the same tokens at the same time —
        # so `last_move_belief_logits` is ALREADY typed by the time it reaches here and the op needs no
        # HP-type argument. `last_hp_type_logits` (the aux-CE + prober stash) is written there too.
        # gen3_unified_move_system_v1: the context-free move-latent table — the Stage-3 latent grading aux
        # TARGET (training only; is_grad_enabled-gated, rollout pays nothing) AND
        # (gen3_unified_topk_incoming_v1) the op's top-K candidate latents. The latter must be present in
        # rollout too (the op output feeds both heads), so when topk is on the table is built EVERY forward.
        # One `latent_table()` call, reused for both.
        move_latent_all = None
        # The op's candidate latent table is needed in rollout (not just is_grad_enabled) when the incoming
        # per-move matrix is on — it gathers the per-move latent into the op output (which feeds both heads)
        # — OR (gen3_entity_move_seats_v1) the E4 threat seats are on: they gather the per-candidate latent
        # as the seat identity (and this method runs PRE-transformer under prefuse, which E4 requires — so
        # the stash below is guaranteed to exist by seat-build time). The old `topk_k > 0` disjunct went
        # with the lean top-K block (gen3_op_block_trim_v1): K>0 now IMPLIES `matrices_incoming` (enforced
        # in __init__ and in the op), so it can no longer select a block of its own.
        need_topk_latent = self.damage_op is not None and (
            self.damage_op.matrices_incoming or self.entity_topk_seats > 0)
        if self.move_latent and (torch.is_grad_enabled() or need_topk_latent):
            enc = self.pokemon_encoder.move_latent_encoder
            latent_table = enc.latent_table(self.embeddings)                     # [n_moves, MOVE_LATENT_DIM]
            if torch.is_grad_enabled():
                self.stash.move_latent_table = latent_table                      # grading aux target
            if need_topk_latent:
                # gen3_opp_hp_typed_candidates_v1: the op's candidate axis is C = n_moves — the typed HPs are
                # the real move-nums 355-370, whose latents already carry their type (move_emb[355-370] ⊕ the
                # type emb ⊕ MOVE_ATTR), so a selected HP-Ice candidate gets the genuine typed-move latent. No
                # synthetic append (the old `hp_latent_block` workaround for the 237 collision is obsolete).
                move_latent_all = latent_table                                   # [n_moves, MOVE_LATENT_DIM]
        # gen3_entity_move_seats_v1: LIVE stash for the E4 seat builder (same forward, read in
        # forward_internal right after this returns; live, not detached — the latent gradient rides).
        self.stash.entity_latent_table = move_latent_all if self.entity_topk_seats > 0 else None
        # Differentiable damage op (flag-guarded; None when off): fed the move belief's PREDICTED moves for
        # the opp active. Forward-only, leak-free; its gradient flows back into the move/spread belief heads
        # via last_move_belief_logits / last_spread_belief.
        damage_block = None
        # X5 fixed_mass (U3 part 3): the op reads the HYPOTHESIS context — a hidden slot is priced as its
        # concrete hypothesis — and the roster's per-slot one-hots replace the T0 marginal as its
        # defender belief (the roster carries everything else: alive, per-mon candidates, Beat Up's π / k).
        _opctx = sctx if x5r is not None else ctx
        _sp = x5r.species_probs if x5r is not None else self.stash.t0_species_probs
        if self.damage_op is not None:
            # Optional gradient-checkpointing (same gate as the transformer): the op materialises several
            # [B,6,~416] activations → recompute in backward for ~GBs of VRAM. Bit-exact (no dropout/RNG);
            # a no-op under inference. ctx is a non-tensor arg (use_reentrant=False); the belief tensors carry
            # the grad. move_latent_all (built above) is the op's top-K identity source (None unless topk on).
            if self.damage_op.grad_checkpointing and torch.is_grad_enabled():
                damage_block = checkpoint(self.damage_op, _opctx, self.last_move_belief_logits,
                                          self.last_spread_belief, move_latent_all,
                                          _sp, _item_cb_prob, fm, x5r,
                                          use_reentrant=False, **_op_kw)
            else:
                damage_block = self.damage_op(_opctx, self.last_move_belief_logits, self.last_spread_belief,
                                              move_latent_all, _sp,
                                              item_cb_prob=_item_cb_prob, fixed_moves=fm, x5_roster=x5r,
                                              **_op_kw)
        # Read-only stash for the prober/forensic decode — never read by the forward, so off is unchanged.
        self.stash.damage_block = damage_block
        return opp_tokens, damage_block

    def _other_edge_cells(self, ctx: ExtractorContext, ro: OpRoster, sb: Optional[torch.Tensor],
                          fams: "set[str]") -> Dict[str, torch.Tensor]:
        """X5 fixed_mass (U3 part 3, M3 (c); ORCHESTRATOR F4 (a)): OTHER_species' cells for every family in
        `EdgeBias.OTHER_FAMILIES` — the SAME kernels, run under the OTHER-MODE roster (`other_roster`:
        every hidden slot holds the renormalised tail's AVERAGED defender and attacker), read at a hidden
        slot. The op's per-forward roster is restored afterwards (a `finally`), so no later kernel can read
        the OTHER-mode one."""
        op = self.damage_op
        assert op is not None and ro.other is not None and ro.other_col is not None
        mb = self.last_move_belief_logits
        assert mb is not None
        sp, col, k = ro.other.species_probs, ro.other_col, self.consequence_topk
        out: Dict[str, torch.Tensor] = {}
        op.stash.x5 = ro.other
        try:
            if "d1" in fams:
                out["d1"] = other_column(op.pairwise_outgoing(ctx, sb, species_probs=sp), col, 2)
            if "c1" in fams:
                out["c1"] = other_column(torch.cat([
                    op.pairwise_boost(ctx, sb, species_probs=sp),
                    op.pairwise_boost_incoming(ctx, mb, k_cand=k, species_probs=sp)], dim=-1), col, 2)
            if "c3" in fams:
                out["c3"] = other_column(op.pairwise_recovery(ctx, mb, k_cand=k, species_probs=sp), col, 2)
            if "d4" in fams:
                out["d4"] = other_column(op.pairwise_bench_incoming(ctx, mb, k_bench=k, species_probs=sp), col, 2)
            if "v" in fams:
                out["v"] = other_column(op.pairwise_speed(ctx, sb), col, 2)
        finally:
            op.stash.x5 = ro
        return out

    def _flat_consumer_ops(self, flat: torch.Tensor, k: int, fm: FixedMassMoves, hs: HypothesisSet,
                           ro: Optional[OpRoster], opctx: ExtractorContext,
                           other_d1: Optional[torch.Tensor]) -> FlatConsumerOps:
        """X5 U4 (§3.7): the α / β consumers' operands under the flat pointer — α / β re-expressed from
        the PUBLICATION (`compat_intent_logits`), and each op stash a live consumer reads with OTHER's
        column appended: OTHER_move's (the op's tail contraction, `FixedMassMoves.other_u`) on the seat
        axis, OTHER_species' (the OTHER-mode D1 pass — computed here only when the edge families did
        not — and P(Ghost | OTHER) = `other_tail_probs @ SPECIES_IS_GHOST`, linear and exact) on the
        mon axis. Never a zero row."""
        op = self.damage_op
        assert op is not None and fm.other_u is not None
        alpha, beta = compat_intent_logits(flat, k)
        st = op.stash
        seat_live = torch.cat([op.last_pair_seat_live if op.last_pair_seat_live is not None
                               else fm.seat_on.to(flat.dtype),
                               fm.other_live.to(flat.dtype).unsqueeze(-1)], dim=-1)
        out_cells = out_pko = None
        if st.out_cells is not None:
            if other_d1 is None:
                assert ro is not None and ro.other is not None and ro.other_col is not None
                op.stash.x5 = ro.other
                try:
                    other_d1 = other_column(op.pairwise_outgoing(
                        opctx, self.last_spread_belief, species_probs=ro.other.species_probs),
                        ro.other_col, 2)
                finally:
                    op.stash.x5 = ro
            assert other_d1 is not None
            out_cells = append_other(st.out_cells, other_d1[..., :st.out_cells.shape[-1]], 2, "out_cells")
            assert out_cells is not None
            out_pko = out_cells[..., _DMG_OMX_IDX_PKO]
        ghost = None
        if st.opp_p_ghost is not None:
            ghost = append_other(st.opp_p_ghost, (hs.other_tail_probs.to(op.SPECIES_IS_GHOST.dtype)
                                                  @ op.SPECIES_IS_GHOST).unsqueeze(-1), 1, "opp_p_ghost")
        topk_w = (torch.cat([st.topk_w, fm.other_mass.to(st.topk_w.dtype).unsqueeze(-1)], dim=-1)
                  if st.topk_w is not None else None)
        return FlatConsumerOps(
            alpha=alpha, beta=beta, seat_live=seat_live, other_u=fm.other_u, topk_w=topk_w,
            pair_cells=append_other(st.pair_cells, st.pair_cells_other, 2, "pair_cells"),
            pair_in=append_other(st.pair_in, st.pair_in_other, 2, "pair_in"),
            type_mult=append_other(st.pair_type_mult, st.pair_type_mult_other, 2, "pair_type_mult"),
            out_cells=out_cells, out_pko=out_pko, opp_p_ghost=ghost)

    def _exact_ko_operands(self, opctx: ExtractorContext, n_seats: int) -> Optional[Any]:
        """gen3_endstate_facts_v1 (`--ko-ramp exact`): `intent_threshold.ExactKo` for a threshold on OUR active —
        its exact max HP (the op's own `_active_defender`) and each of the ``n_seats`` seats' crit chance (the op's
        top-K move nums; a seat past them — X5's OTHER_move — reads the base 1/16). None under `ramp`."""
        op = self.damage_op
        if op is None or not op.ko_exact:
            return None
        from agents.model.intent_threshold import ExactKo
        from agents.model.ko_exact import CRIT_P_BASE
        maxhp = op._active_defender(opctx)[2]                                              # [B]
        nums = op.last_topk_idx
        crit = op._crit_p(nums) if nums is not None else None
        B = opctx.batch_size
        if crit is None:
            crit = maxhp.new_full((B, 0), CRIT_P_BASE)
        pad = n_seats - crit.shape[-1]
        if pad > 0:
            crit = torch.cat([crit, crit.new_full((B, pad), CRIT_P_BASE)], dim=-1)
        return ExactKo(maxhp=maxhp, crit_p=crit[:, :n_seats])

    def _op_content_rows(self, opctx: ExtractorContext, sp: Optional[torch.Tensor],
                         cells: Dict[str, Any]) -> torch.Tensor:
        """gen3_static_board_v1 (`--token-encoding static`): [B, 12, D_MODEL] the per-mon OP CONTENT
        (`board_tokens.OpContent`) from the `x` / `g` / `d1` kernels — the SAME cells the edge families
        deliver (re-used from ``cells`` when a family built them this forward; computed here otherwise, on
        the same context and belief, so the two can never disagree on a value)."""
        op, oc = self.damage_op, self.op_content
        assert op is not None and oc is not None
        x = cells.get("x")
        if x is None:
            x = op.pairwise_entry(opctx, self.last_move_belief_logits)  # type: ignore[arg-type]
        g = cells.get("g")
        if g is None:
            g = op.pairwise_schedule(opctx)
        d1 = None
        if oc.outgoing_proj is not None:
            d1 = cells.get("d1")
            if d1 is None:
                d1 = op.pairwise_outgoing(opctx, self.last_spread_belief, species_probs=sp)
        return oc(x, g, d1)  # type: ignore[no-any-return]

    def forward_internal(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the (pi_combined, vf_combined) pre-projection pair by chaining the phases."""
        # gen3_extractor_stashes_v1: replace the WHOLE stash container at ENTRY — no stash (nor a
        # live belief-supervision view, which holds a graph-carrying tensor whose stale read would
        # backprop through a freed or foreign graph) can survive into this forward. This one line
        # is what makes a stale cross-batch read unrepresentable for every field at once.
        self.stash = ExtractorStashes()
        ctx = self.unpack(obs)
        # gen3_t0_species_prior_v1: resolve the hidden opponent slots to a DISCRETE species
        # distribution HERE — still T0, before any T1 consumer — and hand the same tensor to every
        # site that prices an unrevealed defender. One belief computed once: the edge cells and the
        # op block can then never disagree on a value, which is the invariant `pairwise_outgoing`'s
        # docstring already asserts for the physics. None (flag off) ⇒ every consumer falls through
        # to the static usage prior, byte-identically.
        self.stash.t0_species_probs = (
            self.t0_species_prior(ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE],
                                  ctx.opp_believed_mask)
            if self.t0_species_prior is not None else None
        )
        # Expose which opp slots are believed (hidden) so eval/forensic tooling can decode the belief
        # head's per-slot species prediction for exactly those slots. Read-only stash — never read by
        # the forward itself, so the off/baseline output is unchanged.
        self.stash.opp_believed_mask = ctx.opp_believed_mask
        self.stash.opp_active_local = ctx.opp_active_local   # for the prober's belief-row decode
        role_tokens = self.pokemon_encoder(ctx, self.embeddings)
        # gen3_obs_facts_v1 (`--obs-facts v1` only; `off` builds nothing): the OBS-FACTS block as T0 content on
        # the tokens of the entities it describes, before the belief stack reads them (a hidden opponent slot's
        # hypothesis token is spliced in after, and carries none of it). Under `--token-encoding static` the
        # SIDE-class facts go to the side board tokens instead (`_board_side_extra`, read by the trunk).
        _board_side_extra: Optional[torch.Tensor] = None
        if self.obs_facts_inject is not None:
            assert ctx.obs_facts is not None, "obs_facts=v1 on a layout without the OBS-FACTS block"
            role_tokens = self.obs_facts_inject(ctx.obs_facts, role_tokens, ctx.our_active_idx,
                                                ctx.opp_active_local, self.embeddings)
            if self.obs_facts_inject.side_to_board:
                _board_side_extra = self.obs_facts_inject.side_rows(ctx.obs_facts)
        # gen3_x5_belief_tokens_v1 (X5 U3; built with the belief family): the hypothesis set's SPECIES
        # half (δ_θ reads the PRE-belief opponent role tokens — the revealed ones), then the hypothesis
        # TOKENS: THE `pokemon_encoder` on the hypothesis context (hidden slots' rows = their dex rows;
        # every mask the REAL one) + `hypothesis_marker`, spliced into the hidden opponent slots. (The
        # blob path's constant per-position `BeliefSlots` token is DELETED, v144.)
        _hs: Optional[HypothesisSet] = None
        _hctx: Optional[ExtractorContext] = None
        if self.hypothesis_builder is not None:
            _hs = self._build_hypothesis_species(ctx, role_tokens)
            _hctx = hypothesis_ctx(ctx, _hs, self.layout)
            # gen3_x5_hyp_gather_v1: the encoder's species half over the dex table ONCE, gathered by
            # the hypothesis species; the row-level half per row; the rest per OPPONENT slot only
            # (`hypothesis_encode`: the exact split of the per-row pass, up to fp32 reassociation).
            # gen3_static_tokens_v1 (`--token-encoding static`): a hypothesis row reads no board fact, so its
            # WHOLE token is a function of the species — the dex table encoded once and gathered, exactly.
            if isinstance(self.pokemon_encoder, StaticTokenEncoder):
                _opp_hyp = static_hypothesis_tokens(self.pokemon_encoder, self.embeddings, _hs.slot_species,
                                                    self.hypothesis_builder.dex_rows)
            else:
                _opp_hyp = gathered_hypothesis_tokens(self.pokemon_encoder, self.embeddings, ctx,
                                                      _hs.slot_species, self.hypothesis_builder.dex_rows)
            role_tokens = splice_hypothesis_tokens(role_tokens, _opp_hyp, _hs,
                                                   self.hypothesis_builder.hypothesis_marker)
        # T0 RESOLVE — the move belief (gen3_tiered_pipeline_v1). Reinject the predicted opp moveset
        # into the opp ROLE tokens BEFORE the transformer, so the believed moves co-refine with the
        # species/team belief through the attention layers. The logits are stashed here; every
        # downstream consumer (damage op, E4 seats, edge cells, aux loss) reads the same
        # `last_move_belief_logits`. There is no second placement.
        if self.move_belief is not None:
            opp_role, _mb_logits, _hs_m, _fm_m = self._apply_move_belief(
                role_tokens[:, TEAM_SIZE:], ctx, _hctx, _hs)
            self.stash.move_belief_logits = _mb_logits
            role_tokens = torch.cat([role_tokens[:, :TEAM_SIZE], opp_role], dim=1)
        # T0 RESOLVE — X5's hypothesis set: the opponent ACTIVE's move group joins the species half
        # (X5 only; None with the belief family off).
        # X5 U3 part 2: the opponent active's move axis — ONE order for the E4 seats, the op's top-K /
        # pair cells / α seats and the D3 / S3 cells; the fixed-mass π_m as the op's class-M candidate
        # weights (DETACHED, M10). Built inside `_apply_move_belief` (part 3: the active's reinjection
        # reads π_m, F-X5-26), from the same published posterior.
        _fm: Optional[FixedMassMoves] = None
        if _hs is not None:
            if self.move_belief is not None:
                assert _hs_m is not None
                _hs, _fm = _hs_m, _fm_m
            self.stash.hypothesis = _hs
        # X5 U3 part 3: the op's opponent-MON axis (hidden slots as their hypotheses, "alive" from
        # `opp_addressable`, per-mon fixed-mass candidates in ONE order, Beat Up's π / k). The per-mon
        # selection cuts join the rule-8 exclusion (`near_tie_rows`).
        _x5r: Optional[OpRoster] = None
        if (_hs is not None and _hctx is not None and self.damage_op is not None
                and self.last_move_belief_logits is not None):
            _x5r, _hs = build_op_roster(self.hypothesis_builder, ctx, _hctx, _hs, _fm,
                                        self.last_move_belief_logits,
                                        cuts=(self.consequence_topk, self.entity_topk_seats))
            # OTHER_species' physics (M3 (c), F4 (a) / (b)): the OTHER-mode roster + its max-site presence.
            _x5r = other_roster(_x5r, _hs, self.hypothesis_builder, self.move_belief,
                                self.damage_op.BASE_STATS, self.damage_op.SPECIES_TYPE,
                                self.damage_op.SPECIES_SPREAD_PRIOR, int(self.damage_op.CHART.shape[-1]),
                                _SB_SPE, cuts=(self.consequence_topk, self.entity_topk_seats))
            self.stash.hypothesis = _hs
        # T0 RESOLVE (spread/HP-type) → T1 REASON (the op). Run the WHOLE physics stack ONCE, here,
        # PRE-attention: the spread + HP-type beliefs read the raw opp role tokens (the move belief
        # already did, just above), the FULL DamageOperator runs on that belief, and its per-OUR-mon
        # INCOMING rows are injected onto our role tokens through the zero-init `prefuse_proj` — so
        # attention reasons over the physics. `damage_block` is None only when the op is off, in which
        # case there is nothing to inject (and `prefuse_proj` was never built).
        opp_role, damage_block = self._spread_hp_damage(role_tokens[:, TEAM_SIZE:], ctx, _hctx, _hs, _fm,
                                                        _x5r)
        # X5 U3 part 3: every op kernel below reads the hypothesis context + the roster's one-hots.
        _opctx: ExtractorContext = _hctx if (_x5r is not None and _hctx is not None) else ctx
        _sp = _x5r.species_probs if _x5r is not None else self.stash.t0_species_probs
        if damage_block is not None:
            # gen3_op_tensors_views_v1: the op's typed views (set by the forward that just ran)
            # replace every flat-offset slice on the consumer side.
            inc = self.damage_op.last_tensors.incoming_rows  # type: ignore[union-attr]  # per-OUR-mon incoming rows
            if self.op_worst_proj is None:
                role_tokens = torch.cat(
                    [role_tokens[:, :TEAM_SIZE] + self.prefuse_proj(inc), opp_role], dim=1)  # type: ignore[misc]  # residual (0 at init)
            else:
                # gen3_op_reduction_principled_v1: + the noisy-OR KO worst case per our mon (zero-init, 0 at init).
                role_tokens = torch.cat(
                    [role_tokens[:, :TEAM_SIZE] + self.prefuse_proj(inc)  # type: ignore[misc]
                     + self.op_worst_proj(self.damage_op.last_worst_rows), opp_role], dim=1)  # type: ignore[union-attr]
        else:
            role_tokens = torch.cat([role_tokens[:, :TEAM_SIZE], opp_role], dim=1)
        # gen3_entity_move_seats_v1 (v54, Stage 1): build the move ENTITY seats and enter them into
        # the trunk's attention. The E3 permutation (sorted-by-id → request order, by move-num
        # identity) happens HERE, pre-transformer — one permutation, shared by the seats and the
        # pointer head (which now reads the REFINED seats below). E4 gathers the op's pre-transformer
        # candidate weights + latents (`_entity_latent_table`, stashed by `_spread_hp_damage` — the
        # prefuse gate guarantees it ran). Seats append AFTER the global token, so every absolute
        # slice above (team/history/global) is position-stable.
        _tok_req_raw, _move_valid = _request_order_move_tokens(
            self.pokemon_encoder.last_move_tokens, ctx)  # type: ignore[arg-type]
        _seat_tokens, _seat_pad = self.entity_seats(
            _tok_req_raw, _move_valid, ctx, self.damage_op,
            self.last_move_belief_logits,
            self.stash.entity_latent_table, fixed_moves=_fm, x5_roster=_x5r)
        # gen3_static_port_v1 (`--move-actor-state on`, `static_facts.py`): our active's HP + status onto its 4 E3
        # move seats (zero-init, bias-free; an invalid seat stays the zero token it is).
        if self.move_actor_proj is not None:
            _actor = self.move_actor_proj(move_actor_features(ctx).to(_seat_tokens.dtype))            # [B,D]
            _seat_tokens = torch.cat([
                _seat_tokens[:, :4] + _actor[:, None, :] * _move_valid[:, :, None].to(_seat_tokens.dtype),
                _seat_tokens[:, 4:]], dim=1)
        # gen3_probe_facts_v1 (`--move-target-state on`, `static_facts.move_target_features`): THEIR active's HP + status
        # (the target) onto our 4 E3 move seats (zero-init, bias-free; an invalid seat stays the zero token it is).
        if self.move_target_proj is not None:
            _target = self.move_target_proj(move_target_features(ctx).to(_seat_tokens.dtype))         # [B,D]
            _seat_tokens = torch.cat([
                _seat_tokens[:, :4] + _target[:, None, :] * _move_valid[:, :, None].to(_seat_tokens.dtype),
                _seat_tokens[:, 4:]], dim=1)
        _seat_types = self.entity_seats.seat_types(ctx.device)
        # gen3_event_window_v1 (Tier H-B): the event seats join the extra seam LAST, so every
        # front-indexed seat slice (E3 [:4], E4 [4:4+K], the E5 tail) is position-stable, and
        # they take TOKEN_TYPE_HISTORY (the E5 precedent — no token-type table growth).
        if self.history_events is not None:
            if ctx.event_window is None:
                raise RuntimeError(
                    "history_events is on but the obs carries no event_window block — the "
                    "seats would silently attend over nothing.")
            _ev_tokens, _ev_pad = self.history_events(ctx.event_window, self.embeddings)
            _seat_tokens = torch.cat([_seat_tokens, _ev_tokens], dim=1)
            _seat_pad = torch.cat([_seat_pad, _ev_pad], dim=1)
            _seat_types = torch.cat([
                _seat_types,
                torch.full((_ev_tokens.shape[1],), TOKEN_TYPE_HISTORY,
                           dtype=torch.long, device=ctx.device)], dim=0)
        # gen3_edge_bias_trunk_v1 (v56, Stage 2): computed physics as attention EDGES. Cells are
        # built HERE (pre-transformer — d1 from the validated outgoing-matrix kernel at the belief
        # the prefuse stack already produced; d3 from the pre-collapse incoming kernel at the SAME
        # candidate selection the E4 seats just stashed) and delivered to every layer as per-pair
        # per-head additive logit biases via the closure. Zero-init maps ⇒ identity at init.
        _edge_fn = None
        _other_d1: Optional[torch.Tensor] = None     # X5 U4: OTHER_species' D1 cells, reused by out_cells
        _cells: Dict[str, Any] = {}                  # the edge families' cells (the static OPC reuses them)
        if self.edge_bias is not None:
            _fams = self.edge_bias.families
            # The T0 stack computed the spread belief THIS forward, pre-trunk (gen3_tiered_pipeline_v1
            # made that unconditional), so it is always the current one. None when the leg is off —
            # the kernels then use their legacy neutral-bulk constants.
            _sb = self.last_spread_belief
            if "d1" in _fams:
                _cells["d1"] = self.damage_op.pairwise_outgoing(  # type: ignore[union-attr]
                    _opctx, _sb, species_probs=_sp)
            if "c1" in _fams:
                # C1 (outgoing) reuses D1's current-world cells as its delta base when both are
                # on; C1b (incoming) appends the defensive halves — one 6-wide consequence cell.
                _cells["c1"] = torch.cat([
                    self.damage_op.pairwise_boost(_opctx, _sb, base=_cells.get("d1"),  # type: ignore[union-attr]
                                                  species_probs=_sp),
                    self.damage_op.pairwise_boost_incoming(  # type: ignore[union-attr]
                        _opctx, self.last_move_belief_logits, k_cand=self.consequence_topk,  # type: ignore[arg-type]
                        species_probs=_sp),
                ], dim=-1)
            if "c3" in _fams:
                _cells["c3"] = self.damage_op.pairwise_recovery(  # type: ignore[union-attr]
                    _opctx, self.last_move_belief_logits, k_cand=self.consequence_topk,  # type: ignore[arg-type]
                    species_probs=_sp)
            if "c2" in _fams:
                _cells["c2"] = self.damage_op.pairwise_status_consequence(  # type: ignore[union-attr]
                    _opctx, self.last_move_belief_logits, _sb, k_cand=self.consequence_topk,  # type: ignore[arg-type]
                    species_probs=_sp)
            if "c5" in _fams:
                _cells["c5"] = self.damage_op.pairwise_baton(_opctx, _sb)  # type: ignore[union-attr]
            if "s1" in _fams:
                _cells["s1"] = self.damage_op.discrete_outgoing_status(_opctx, per_pair=True)  # type: ignore[union-attr]
            if "d2" in _fams:
                _cells["d2"] = self.damage_op.pairwise_bench_outgoing(_opctx, _sb)  # type: ignore[union-attr]
            if "d3" in _fams:
                # X5 (fixed_mass): priced on the EXTENDED seat axis and contracted onto the K seats
                # (a revealed Hidden Power's seat is its typed mixture).
                _d3_cand = (self.entity_seats.last_cand if _fm is None else (_fm.idx_ext, _fm.w_ext))
                _cells["d3"] = self.damage_op.pairwise_incoming(  # type: ignore[union-attr]
                    _opctx, self.last_move_belief_logits, _d3_cand,  # type: ignore[arg-type]
                    spread_belief=(self.last_spread_belief
                                   if self.damage_op.believed_lean else None),  # type: ignore[union-attr]
                    species_probs=_sp)
                if _fm is not None:
                    _cells["d3"] = _fm.mix_seats(_cells["d3"], dim=1)
            if "d4" in _fams:
                _cells["d4"] = self.damage_op.pairwise_bench_incoming(  # type: ignore[union-attr]
                    _opctx, self.last_move_belief_logits, k_bench=self.consequence_topk,  # type: ignore[arg-type]
                    species_probs=_sp)
            if "g" in _fams:
                _cells["g"] = self.damage_op.pairwise_schedule(_opctx)  # type: ignore[union-attr]
            if "c4" in _fams:
                # gen3_entity_rehome_v1: protect odds live ON the mon slot now — gather OUR
                # active's per-mon protect field (pokemon.py POKEMON_PROTECT_OFFSET).
                _po = ctx.pokemon_part[
                    torch.arange(ctx.batch_size, device=ctx.device), ctx.our_active_idx,
                    POKEMON_PROTECT_OFFSET]
                _cells["c4"] = self.damage_op.pairwise_protect(_opctx, _po)  # type: ignore[union-attr]
            if "x" in _fams:
                _cells["x"] = self.damage_op.pairwise_entry(_opctx, self.last_move_belief_logits)  # type: ignore[arg-type,union-attr]
            if "t" in _fams:
                _cells["t"] = self.damage_op.pairwise_trap(_opctx)  # type: ignore[union-attr]
            if "v" in _fams:
                _cells["v"] = self.damage_op.pairwise_speed(_opctx, _sb)  # type: ignore[union-attr]
            if "h" in _fams:
                # Tier H-A2: the obs-fed pair-history TENDENCY cells — obs order is
                # (opp i, our j); the mon×mon block convention is (our, opp), so permute.
                if ctx.pair_history is None:
                    raise RuntimeError(
                        "edge family 'h' is on but the obs layout carries no pair_history "
                        "block — the family would silently bias on nothing.")
                _cells["h"] = ctx.pair_history.permute(0, 2, 1, 3)
            if "r" in _fams:
                # Tier H-C: STRUCTURAL reference edges — event e's recorded actor/target IS mon
                # m. Species-num equality, SIDE-GATED (a mirror species on the other team must
                # not false-link: the actor lives on the event's own side, the target on the
                # opposite side). PAD rows (valid=0) contribute nothing.
                if ctx.event_window is None or self.history_events is None:
                    raise RuntimeError(
                        "edge family 'r' is on but the event seats are not built "
                        "(--history-events) — the reference edges would have no rows.")
                _cells["r"] = _event_reference_cells(ctx.event_window, ctx.species_ids)
            if "s3" in _fams:
                _cells["s3"] = self.damage_op.discrete_incoming_status(  # type: ignore[union-attr]
                    _opctx, self.last_move_belief_logits, self.entity_seats.last_cand, per_pair=True)  # type: ignore[arg-type]
            _opp_oh = None
            if "d2" in _fams:
                _opp_oh = torch.zeros(ctx.batch_size, TEAM_SIZE, device=ctx.device)
                _opp_oh[torch.arange(ctx.batch_size, device=ctx.device), ctx.opp_active_local] = 1.0
            _base = self.team_transformer._total_tokens
            # X5 fixed_mass (U3 part 3, M3 (c), F-X5-28): OTHER_species' edge column — the op's kernels run
            # once more under the OTHER-MODE roster (every hidden slot holds the tail-averaged mon) and
            # OTHER's cells are read at a hidden slot (`other_column`).
            _ocells: Dict[str, torch.Tensor] = {}
            _oidx = -1
            _olive: Optional[torch.Tensor] = None
            if _x5r is not None and _x5r.other is not None:
                _ocells = self._other_edge_cells(_opctx, _x5r, _sb, _fams)
                _other_d1 = _ocells.get("d1")
                _oidx = self.team_transformer._total_tokens + self.entity_seats.n_seats
                _olive = _x5r.other_live
            _board_seats = self.team_transformer.board_seats
            _edge_fn = lambda bias: self.edge_bias(  # noqa: E731
                bias, _base, _cells, _opp_oh, other_cells=_ocells, other_index=_oidx, other_live=_olive,
                board_seats=_board_seats)
            _c2_edge_cells = _cells.get("c2")
        else:
            _c2_edge_cells = None
        # gen3_intent_move_cell_v1 (G3): the RAW c2-for-the-move-cell operands, computed HERE —
        # still T1, where every other op kernel runs (alpha is T2 and does not exist yet; the
        # weighting happens at the pointer stash below, the same T1-producer/T2-consumer split as
        # `last_pair_cells`). Reuses the c2 edge grid when the edge family already built it this
        # forward — identical function, so the value is the same either way.
        _imc_ops = None
        if (self.intent_move_cell is not None or self.move_resolution_cell is not None) and damage_block is not None:
            _imc_ops = self.damage_op.pointer_intent_status_operands(  # type: ignore[union-attr]
                _opctx, self.last_move_belief_logits, self.last_spread_belief,  # type: ignore[arg-type]
                k_cand=self.consequence_topk, c2_cells=_c2_edge_cells,
                species_probs=_sp,
                # X5 U4 (fixed_mass): + OTHER_move's column (the flat pointer's (K+1)-th seat)
                other_u=(_fm.other_u if (_fm is not None and self.flat_intent_head is not None) else None))
        # gen3_x5_belief_tokens_v1 (fixed_mass): OTHER_species joins the trunk as ONE extra seat right
        # after the entity seats (E3/E4/E5 stay front-indexed; the event seats stay LAST, which the
        # `r` edge family's slice requires), typed THEIR_TEAM, key-masked iff OTHER is masked
        # (structural). Every opponent key carries its log-presence (`key_log_presence`).
        _klp = None
        _other_idx = -1
        if _hs is not None:
            _n_ent = self.entity_seats.n_seats
            _other_idx = self.team_transformer._total_tokens + _n_ent
            _seat_tokens = torch.cat([_seat_tokens[:, :_n_ent], _hs.other_token.unsqueeze(1).to(_seat_tokens.dtype),
                                      _seat_tokens[:, _n_ent:]], dim=1)
            _seat_pad = torch.cat([_seat_pad[:, :_n_ent], (~_hs.other_live).unsqueeze(1),
                                   _seat_pad[:, _n_ent:]], dim=1)
            _seat_types = torch.cat([
                _seat_types[:_n_ent],
                torch.full((1,), TOKEN_TYPE_THEIR_TEAM, dtype=torch.long, device=ctx.device),
                _seat_types[_n_ent:]], dim=0)
            _e5_off = (self.team_transformer._total_tokens + 4 + self.entity_topk_seats
                       if self.entity_seats.tail_seats else None)
            _base_t = self.team_transformer._total_tokens
            _klp = key_log_presence(
                self.team_transformer._total_tokens + _seat_tokens.shape[1], _hs, _other_idx, e5_offset=_e5_off,
                e4=(_fm.seat_logp if (_fm is not None and self.entity_topk_seats > 0) else None),
                e4_offset=_base_t + 4,
                e5_active=(_fm.other_log_mass if _fm is not None else None),
                opp_active_local=ctx.opp_active_local)
        # gen3_static_board_v1 (`--token-encoding static`, audit B2): the per-mon OP CONTENT on BOTH sides —
        # the amounts an edge bias cannot carry (a ratio inside a softmax row): every mon's Spikes chip on entry
        # and end-of-turn ledger, and our active's damage to each of THEIR mons. Added pre-trunk, after the op
        # (T1), beside `prefuse_proj`'s incoming rows on our mons. None under legacy (nothing built).
        if self.op_content is not None:
            role_tokens = role_tokens + self._op_content_rows(_opctx, _sp, _cells)
        # gen3_static_port_v1 (`--mon-hazard-cost on`, `static_facts.py`): every mon's own side's Spikes layers and its
        # switch-in HP cost (the op's ONE entry rule, on the context the op prices with), as D content (zero-init).
        if self.mon_hazard_proj is not None:
            _haz = mon_hazard_features(self.damage_op, _opctx,
                                       _x5r.concrete if _x5r is not None else None)              # [B,12,2]
            role_tokens = role_tokens + self.mon_hazard_proj(_haz.to(role_tokens.dtype))
        # gen3_static_recovery_v1 (`--switch-hazard-cost on`, `static_facts.switch_hazard_features`): each switch
        # target's [our side's Spikes layers / 3, its switch-in HP fraction] (the op's ONE entry rule), appended LAST to
        # the switch pointer cell below (the head's zero-init `switch_extra_proj` reads it).
        _sw_haz: Optional[torch.Tensor] = (switch_hazard_features(self.damage_op, _opctx)
                                           if self.switch_hazard_cost == "on" else None)          # [B,6,2]
        # gen3_static_recovery_v1 (`--eot-residual on`, `eot_residual.py`): every mon's expected END-OF-TURN HP change
        # if it is the one on the field at the end of this turn, per component, as token content (zero-init).
        if self.eot_residual_proj is not None:
            assert self.eot_residual_rule is not None
            _eot = self.eot_residual_rule(_opctx, self.damage_op,
                                          _x5r.concrete if _x5r is not None else None)           # [B,12,EOT_DIM]
            role_tokens = role_tokens + self.eot_residual_proj(_eot.to(role_tokens.dtype))
        # gen3_endstate_facts_v1 (`--status-facts exact`, `status_facts.py`): every mon's cure-availability FACTS (a
        # live cleric on its side, Natural Cure, Rest, a Lum / Chesto Berry), both sides, as token content (zero-init).
        if self.status_cure_proj is not None:
            assert self.status_cure_rule is not None
            _cure = self.status_cure_rule(_opctx, self.damage_op, self.last_move_belief_logits)   # [B,12,CURE_DIM]
            role_tokens = role_tokens + self.status_cure_proj(_cure.to(role_tokens.dtype))
        # gen3_probe_facts_v1 (`--effective-stats on`, `static_facts.effective_stat_features`): each side's ACTIVE mon's
        # stage-applied stats (ours exact, theirs the believed spread) as token content (zero-init).
        if self.effective_stats_proj is not None:
            _eff = effective_stat_features(self.damage_op, _opctx, self.last_spread_belief)      # [B,12,EFF_DIM]
            role_tokens = role_tokens + self.effective_stats_proj(_eff.to(role_tokens.dtype))
        our_team_out, their_team_out, _seat_out = self.team_transformer(
            role_tokens, ctx, self.embeddings,
            extra=(_seat_tokens, _seat_types, _seat_pad),
            edge_bias_fn=_edge_fn, key_log_presence=_klp, board_side_extra=_board_side_extra)
        _presence: Optional[OppPresence] = None
        if _hs is not None:
            assert _seat_out is not None
            _presence = OppPresence(
                slot_log_pi=_hs.slot_log_pi,
                other_out=_seat_out[:, _other_idx - self.team_transformer._total_tokens, :],
                other_log_mass=_hs.other_log_mass, other_live=_hs.other_live)
        # gen3_rank_probe_stash_v1 (K6): the rank probe's trunk readout, by reference (no copy).
        self.stash.trunk_tokens = (our_team_out, their_team_out)
        # Aux belief logits over the refined opp tokens — stashed for the PPO aux loss, NOT fed back
        # into the policy/value path (labels would leak). None when belief is off.
        self.stash.belief_logits = (
            self.belief_head(their_team_out, ctx.species_ids[:, TEAM_SIZE:], ctx.opp_believed_mask)
            if self.belief_head is not None else None
        )
        # (The move belief, the spread/HP-type legs and the DamageOperator all ran PRE-transformer —
        # gen3_tiered_pipeline_v1. `damage_block` and `last_move_belief_logits` were set there and
        # there only; there is no second call site to skip.)
        #
        # CLS pools — derived ONCE, on the final team tokens, so the policy
        # pools, the value pool, and the side/aux readouts below ALL reflect the same state.
        our_team_pooled, their_team_pooled, our_active_refined, value_pooled = self.cls_pool(
            our_team_out, their_team_out, ctx,
            threat_rows=(self.damage_op.last_reduced_extra  # type: ignore[union-attr]
                         if self.value_threat_inject else None),
            presence=_presence,
        )
        # gen3_rank_probe_stash_v1 (K6): the CLS pool's value readout BEFORE the value routes inject
        # below — what the rank probe's `rank/value_cls_*` has always measured.
        self.stash.value_cls = value_pooled
        # gen3_pointer_native_v1 / gen3_entity_move_seats_v1: stash the pointer action head's
        # PER-ENTITY inputs for `Gen3DualHeadMaskablePolicy._get_action_dist_from_latent` — the head
        # itself lives on the policy (its ctx is latent_pi, which doesn't exist here). Move logit k
        # now reads the REFINED E3 seat k (post-attention, d_model-wide — the Stage-1 payoff: the
        # token was refined IN the trunk alongside the board, not just inside PokemonEncoder). The
        # request-order permutation happened ONCE, pre-transformer, at the seat build — order is
        # seat-stable through attention, so seat k is still action logit 6+k; `_move_valid` gates
        # unresolved slots to logit 0 exactly as before (their refined content is attention noise the
        # head never scores). Switch scorer j reads the same (possibly re-attended) board-aware team
        # token every pool reads; the op cells are the same post-gain numbers the projection heads
        # consume (width-0 when the op is off — the head's Linears are built correspondingly
        # narrower, never silently zero-padded).
        # (The blob path's two separate α / β heads are DELETED — the X5 version break, config v144; the flat
        # pointer below is the intent readout.)
        # gen3_x5_flat_pointer_v1 (X5 U4; design §3.7): the FLAT opponent pointer — built with the belief
        # family — one list (their move seats, OTHER_move, each switch target, OTHER_species), one softmax,
        # the detached log π as the logit bias (M10). The consumers below
        # read its RE-EXPRESSION (`_x5i`): α over the K seats + OTHER_move (a priced (K+1)-th seat) +
        # the total switch mass, β over the six slots + OTHER_species, and every seat-axis / mon-axis
        # operand with OTHER's column appended — never a zero row.
        _x5i: Optional[FlatConsumerOps] = None
        if self.flat_intent_head is not None:
            assert _hs is not None and _fm is not None and _presence is not None and _seat_out is not None
            _K = self.entity_topk_seats
            _keep = self.opp_intent_grad_mode == "shaping"
            _ictx = torch.cat([our_team_pooled, their_team_pooled], dim=-1)
            _ftok, _fi = flat_candidates(
                _seat_out, _K, their_team_out, _presence.other_out, _fm, _hs, ctx.opp_addressable,
                ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1], ctx.opp_active_local,
                ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE])
            if not _keep:
                _ftok, _ictx = _ftok.detach(), _ictx.detach()
            _flat = self.flat_intent_head(_ftok, _ictx, _fi.log_pi, _fi.live, _K)
            self.stash.belief_supervision["flat_intent_logits"] = _flat
            _flat_pub = self._publish_belief(_flat)
            assert _flat_pub is not None
            self.stash.flat_intent_logits = _flat_pub
            self.stash.flat_intent = _fi
            _x5i = self._flat_consumer_ops(_flat_pub, _K, _fm, _hs, _x5r, _opctx, _other_d1)
            self.stash.flat_consumer_ops = _x5i
        _tok_req = _seat_out[:, :4, :]
        if self.damage_op is not None and damage_block is not None:
            _mcells, _scells = self.damage_op.pointer_cells(damage_block)
        else:
            _mcells = _tok_req.new_zeros(ctx.batch_size, _tok_req.shape[1], 0)
            _scells = our_team_out.new_zeros(ctx.batch_size, TEAM_SIZE, 0)
        # gen3_intent_move_cell_v1 (G3): alpha consumed on the POLICY side — the c2 re-delivery
        # channels join the pointer MOVE cell HERE, the first point where both operands exist
        # (the op's T1 operand stash from above, and alpha, T2, scored from the seats and pools).
        # α / β are the FLAT POINTER's re-expression (`_x5i`, from the PUBLICATION — stop-grad under
        # `belief_grad_mode=label_only`); None with the belief family off (a consumer that needs them
        # requires opp_intent, and opp_intent builds the flat pointer).
        _al = None if _x5i is None else _x5i.alpha
        _bl = None if _x5i is None else _x5i.beta
        # The four α-REQUIRING consumers below (intent_move_cell, intent_threshold, intent_conditional,
        # switch_branch) read the flat pointer's operands only: each requires opp_intent, and opp_intent builds
        # the flat pointer, so `_x5i` is never None there (the blob path's per-op fallbacks were deleted with
        # it, the X5 version break, config v144, part 2). The three that may run WITHOUT α (pair_outcome_move /
        # _switch, conditional_threat: the R1 `belief_mean` rung) keep both reads.
        if self.intent_move_cell is not None:
            if _x5i is None or _imc_ops is None:
                raise RuntimeError(
                    "intent_move_cell is on but alpha produced no logits or the op stashed no c2 "
                    "operands — the cell would silently contribute nothing, which is "
                    "indistinguishable from a null RESULT.")
            _mcells = torch.cat([_mcells, self.intent_move_cell(_x5i.alpha, *_imc_ops,
                                                                seat_live=_x5i.seat_live)], dim=2)
        # gen3_intent_threshold_v1 (v84): the α-weighted threshold operator, computed ONCE here
        # (the first point where α exists) and consumed by BOTH heads — the move-cell block joins
        # the pointer cells now; the vf block reads the stashed probs at the value tail (a
        # T2-produced tensor read at T3 — the allowed direction). The consumer reads
        # `last_alpha_logits` — the PUBLICATION, stop-grad under `belief_grad_mode=label_only`.
        if self.intent_threshold_move is not None:
            _pair_cells = None if _x5i is None else _x5i.pair_cells
            if _x5i is None or _pair_cells is None:
                raise RuntimeError(
                    "intent_threshold is on but alpha produced no logits or the op stashed no "
                    "pair cells — the thresholds would silently contribute nothing, which is "
                    "indistinguishable from a null RESULT. Requires damage_topk_k>0 (and the "
                    "incoming matrix that computes it).")
            _tp = threshold_probs(
                _x5i.alpha, _pair_cells, self.damage_op.last_pair_gate,  # type: ignore[arg-type,union-attr]
                ctx.our_active_idx, seat_live=_x5i.seat_live,
                exact=self._exact_ko_operands(_opctx, _x5i.alpha.shape[-1] - 1))
            self.stash.thresh_probs = _tp
            _mcells = torch.cat([_mcells, self.intent_threshold_move(
                *_tp, ctx.our_active_req_move_ids)], dim=2)
        # gen3_intent_conditional_v1 (v85): the Counter/flinch/Explosion/Pursuit cells — same
        # T1-producer/T2-consumer split, same publication read.
        if self.intent_conditional is not None:
            _pc = None if _x5i is None else _x5i.pair_cells
            # gen3_x5_version_break_v1 part 5: every op value this cell treats as physics — our moves' high roll
            # (a damage fraction), P(we act first), the flinch chance (probabilities) — is read PRE-gain, in every
            # speed mode: the learned `out_gain` is the projection's adapter, not physics (ONE rule, the
            # move-resolution family's; `--speed-physics on`'s special case is gone).
            _ot = self.damage_op.last_raw_tensors if self.damage_op is not None else None
            _opko = None if _x5i is None else _x5i.out_pko
            _ready = (_x5i is not None and _pc is not None
                      and _ot is not None and _ot.out_per_move is not None
                      and _opko is not None
                      and self.damage_op.last_topk_idx is not None)  # type: ignore[union-attr]
            if not _ready:
                raise RuntimeError(
                    "intent_conditional is on but alpha/the op stashes are missing — the cells "
                    "would silently contribute nothing, which is indistinguishable from a null "
                    "RESULT. Requires damage_topk_k>0 + the incoming matrix + the outgoing "
                    "block.")
            _po = ctx.pokemon_part[
                torch.arange(ctx.batch_size, device=ctx.device), ctx.our_active_idx,
                POKEMON_PROTECT_OFFSET][:, None]
            # gen3_op_lean_forward_v1: the boom cell reads the op's typed PRE-gain pko
            # stash — honest probabilities, present in both render modes (the flat render
            # is serialization, not a source).
            assert _x5i is not None
            _mcells = torch.cat([_mcells, self.intent_conditional(
                _x5i.alpha, _pc, self.damage_op.last_pair_gate,  # type: ignore[union-attr]
                ctx.our_active_idx, self.damage_op.last_topk_idx,  # type: ignore[union-attr]
                _ot.out_per_move[..., 1],  # type: ignore[index,union-attr]
                _ot.out_p_outspeed,  # type: ignore[union-attr]
                _ot.out_secondary[..., _OUT_SEC_FLINCH_COL],  # type: ignore[index,union-attr]
                ctx.our_active_req_move_ids, _po,
                _x5i.beta, _opko,
                ctx.opp_active_local, seat_live=_x5i.seat_live, other_u=_x5i.other_u)],
                dim=2)
        # gen3_pair_outcome_v1 (v93): the UNIFIED outcome vector, α-contracted. The T1 producer
        # (the op) built `pair_in` over the (our mon, their believed seat) grid; here at T2 — the
        # first point where α exists — ONE distribution reduces it, and the row for our ACTIVE
        # defender joins every move cell.
        #
        # α comes from the PUBLICATION when the intent head is on, and from the R1 `belief_mean`
        # rung (α := w/Σw) when it is off. That fallback is what makes this flag independently
        # enableable, and `pair_alpha` documents loudly that presence-belief and usage-belief are
        # NOT the same object — the second is the whole point of the intent head.
        if self.pair_outcome_move is not None:
            _pin = ((self.damage_op.last_pair_in if self.damage_op is not None else None)
                    if _x5i is None else _x5i.pair_in)
            _pw = ((self.damage_op.last_topk_w if self.damage_op is not None else None)
                   if _x5i is None else _x5i.topk_w)
            if _pin is None or _pw is None:
                raise RuntimeError(
                    "pair_outcome_cell is on but the op stashed no unified outcome vector (or no "
                    "top-K belief weights) — the cell would silently contribute nothing, which is "
                    "indistinguishable from a null RESULT. Requires damage_topk_k>0 (and the "
                    "incoming matrix that computes it).")
            _alpha = pair_alpha(_al, _pw,
                                (self.damage_op.last_pair_seat_live  # type: ignore[union-attr]
                                 if _x5i is None else _x5i.seat_live))
            _row = reduce_pair_in(
                _alpha, _pin, self.damage_op.last_pair_gate,  # type: ignore[arg-type,union-attr]
                ctx.our_active_idx)
            _mcells = torch.cat([_mcells, self.pair_outcome_move(_row)], dim=2)
        # gen3_pair_outcome_switch_v1 (v94): the SAME reduction, at EVERY defender, into the
        # pointer SWITCH cell — `design_pair_reduction.md` §2.1's own defect, at its own sink. One
        # α (no J axis ⇒ D3 stays a shape error) producing six rows, each riding its own mon's
        # logit, so the module is equivariant in our team axis by construction.
        if self.pair_outcome_switch is not None:
            _pin_s = ((self.damage_op.last_pair_in if self.damage_op is not None else None)
                      if _x5i is None else _x5i.pair_in)
            _pw_s = ((self.damage_op.last_topk_w if self.damage_op is not None else None)
                     if _x5i is None else _x5i.topk_w)
            _tn_s = self.damage_op.last_topk_idx if self.damage_op is not None else None
            if _pin_s is None or _pw_s is None or _tn_s is None:
                raise RuntimeError(
                    "pair_outcome_switch is on but the op stashed no unified outcome vector (or no "
                    "top-K belief weights / move nums) — the cell would silently contribute "
                    "nothing, which is indistinguishable from a null RESULT. Requires "
                    "damage_topk_k>0 (and the incoming matrix that computes it).")
            _alpha_s = pair_alpha(_al, _pw_s,
                                  (self.damage_op.last_pair_seat_live  # type: ignore[union-attr]
                                   if _x5i is None else _x5i.seat_live))
            _rows = reduce_pair_in_all(
                _alpha_s, _pin_s, self.damage_op.last_pair_gate)  # type: ignore[arg-type,union-attr]
            _scells = torch.cat([_scells, self.pair_outcome_switch(
                _rows, _alpha_s, _tn_s,
                ctx.type1_ids[:, :TEAM_SIZE], ctx.type2_ids[:, :TEAM_SIZE],
                # index 1 of the hazard pair is THEIR side — the layers WE set, which is exactly
                # what their Rapid Spin would remove and a Ghost switch-in would preserve.
                ctx.spikes_feature[:, 1:2],
                **({} if _x5i is None else {"other_u": _x5i.other_u}))], dim=2)
        # gen3_conditional_threat_v1 (v95): OA1 — the SECOND widener of the switch cell. Same α
        # ladder, same (defender, seat) grid, DIFFERENT quantities: the accuracy-folded P(this mon
        # dies) (§0.2(2) — a thin tanh scorer cannot multiply two of its own inputs), the
        # bulk-INDEPENDENT expected type multiplier (the one cell channel `pair_in` never carried),
        # and the two §0.2(3) MARGINS against our own HP. §1.2's λ-weighted `w` is NOT built — see
        # the substitution table in `conditional_threat.py`.
        if self.conditional_threat is not None:
            _ct_pin = ((self.damage_op.last_pair_in if self.damage_op is not None else None)
                       if _x5i is None else _x5i.pair_in)
            _ct_w = ((self.damage_op.last_topk_w if self.damage_op is not None else None)
                     if _x5i is None else _x5i.topk_w)
            _ct_tm = ((self.damage_op.last_pair_type_mult if self.damage_op is not None else None)
                      if _x5i is None else _x5i.type_mult)
            if _ct_pin is None or _ct_w is None or _ct_tm is None:
                raise RuntimeError(
                    "conditional_threat_cell is on but the op stashed no unified outcome vector / "
                    "top-K belief weights / type multiplier — the cell would silently contribute "
                    "nothing, which is indistinguishable from a null RESULT. Requires "
                    "damage_topk_k>0 and the incoming matrix that computes both.")
            _scells = torch.cat([_scells, self.conditional_threat(
                pair_alpha(_al, _ct_w,
                           (self.damage_op.last_pair_seat_live  # type: ignore[union-attr]
                            if _x5i is None else _x5i.seat_live)),
                _ct_pin, _ct_tm, self.damage_op.last_pair_gate,  # type: ignore[union-attr]
                ctx.hp_and_active[:, :TEAM_SIZE, 0])], dim=2)
        # gen3_switch_branch_v1 (v94): OA2 + the Rapid-Spin spinblock + Protect's α-conditioning —
        # the per-request-slot content of the branch in which the OPPONENT switches. The last
        # move-cell rider, and the only one that consumes β forward-side besides v85's boom trade.
        if self.switch_branch is not None:
            _oc = None if _x5i is None else _x5i.out_cells
            _pg = None if _x5i is None else _x5i.opp_p_ghost
            _tn_b = self.damage_op.last_topk_idx if self.damage_op is not None else None
            if _x5i is None or _oc is None or _pg is None or _tn_b is None:
                raise RuntimeError(
                    "switch_branch_cell is on but α/β produced no logits or the op stashed no "
                    "outgoing grid / ghost marginal / top-K selection — the cell would silently "
                    "contribute nothing, which is indistinguishable from a null RESULT. Requires "
                    "opp_intent + damage_matrices_outgoing + damage_topk_k>0 (and the incoming "
                    "matrix that computes the seat axis).")
            _po_b = ctx.pokemon_part[
                torch.arange(ctx.batch_size, device=ctx.device), ctx.our_active_idx,
                POKEMON_PROTECT_OFFSET][:, None]
            _mcells = torch.cat([_mcells, self.switch_branch(
                _x5i.alpha, _x5i.beta, _x5i.seat_live, _tn_b, _oc, _pg,
                ctx.opp_active_local, ctx.our_active_req_move_ids, _po_b,
                # index 0 of the hazard pair is OUR side — what OUR Rapid Spin would remove, and
                # therefore the stake a spinblock destroys.
                ctx.spikes_feature[:, 0:1], other_u=_x5i.other_u)], dim=2)
        # gen3_move_resolution_v1 (v141, `--move-resolution on`): the MOVE-RESOLUTION family — per legal action,
        # P(it resolves as stated) and the seven blocks' FACTS, consolidated (their judgments dropped). The
        # policy retires the seven it replaces, so on a built policy this is the only rider of either cell.
        if self.move_resolution_cell is not None:
            _mr_m, _mr_s = self.move_resolution_cell(
                gather_move_resolution_ops(self, ctx, _al, _bl, _imc_ops, _x5i))
            _mcells = torch.cat([_mcells, _mr_m], dim=2)
            _scells = torch.cat([_scells, _mr_s], dim=2)
        # gen3_static_recovery_v1 (`--switch-hazard-cost on`): the LAST switch-cell columns, read by the pointer head's
        # zero-init `switch_extra_proj` (`pointer_switch_extra_dim`), so the existing scorer's init is untouched.
        if _sw_haz is not None:
            _scells = torch.cat([_scells, _sw_haz.to(_scells.dtype)], dim=2)
        self.stash.pointer_inputs = PointerInputs(
            move_tokens=_tok_req, move_valid=_move_valid, team_tokens=our_team_out,
            move_cells=_mcells, switch_cells=_scells)
        belief = None
        if self.hidden_opp_belief is not None:
            # Same 12-token memory + the single-sourced ctx.all_fainted key-mask the value CLS pools
            # over (all_team_out is a forward activation, cheap to recompute; the MASK carries the
            # NaN-safety invariant and is single-sourced on the context). Computed BEFORE the value
            # routes because the entity pool's `full` rider reads the belief rows.
            all_team_out = torch.cat([our_team_out, their_team_out], dim=1)                 # [B, 12, D]
            belief = self.hidden_opp_belief(all_team_out, ctx.all_fainted, ctx.batch_size,
                                            presence=_presence)
        # ============================================================================
        # gen3_value_pooled_routes_v1 (v89): the value routes INJECT into `value_pooled` —
        # the tensor the dist-head critic actually reads — instead of the post-assembler vf
        # concat, which `--value-from-dist` structurally bypassed (verified on gen-12:
        # `value_entity_pool.out_proj` and the then-live α-reduce projection bit-exact ZERO
        # after 25M steps, while `value_threat_proj` — the one value_pooled route — trained
        # to 0.117). Since the critic-route deletion wave `vf_combined IS value_pooled`, so
        # the SAME tensor feeds `value_net` when the scalar critic is on: one wiring, both
        # parameterizations, and no second vf branch for either to orphan. Every route stays
        # zero-init (cold start adds exactly 0) and vf-only at ANY weight (pi never reads
        # value_pooled). Additive injection changes no width, so route availability can
        # never mis-size `value_pre_norm` — the ede5a88 discovery bug class is gone by
        # construction; the runtime raise guards below keep "on but inputs missing" LOUD.
        # ============================================================================
        for _route_name, _contrib in self._value_pooled_routes(ctx, our_team_out,
                                                               their_team_out, belief,
                                                               damage_block, _presence):
            value_pooled = value_pooled + _contrib
        # Read-only stash of the value-CLS pool (the critic's whole-board "who's winning" summary, the
        # 128-dim value-CLS hint layer). Read by the capacity probes, the ride-along heads and the cf
        # terms. NOT read by the forward → off-path/eval is byte-identical; carries grad on the
        # training pass (a live activation).
        self.stash.value_pooled = value_pooled
        # Auxiliary win-probability readout (flag-guarded; None when off). Reads the whole-board
        # value_pooled and stashes a [B,1] logit for the aux loss + the prober/eval. NOT fed into the
        # assembler (a side readout — the future OUTCOME label can't leak into pi/vf). `read_only` feeds
        # a STOP-GRAD value_pooled (head-only training, no trunk gradient); `shaping` feeds it live.
        # Computed on EVERY forward (one small MLP) so eval/inference can read P(win) too — its cost is
        # negligible and it is never gated off, since the prober reads it under no_grad.
        if self.win_head is not None:
            wp_in = value_pooled if self.win_prob_mode == "shaping" else value_pooled.detach()
            self.stash.win_prob_logits = self.win_head(wp_in)
        if self.policy_query is not None:
            # gen3_policy_readout_trunk_v1 (`--policy-readout trunk`, audit F2): the policy's context is
            # the state query's read of EVERY refined trunk token, in the trunk's own order and under its
            # own key mask (+ its per-key log-presence under fixed_mass), plus the hidden-opponent belief
            # pool's K outputs when built (never masked). No assembler concat, no projection, no tower:
            # `pi_features` IS this [B, D_MODEL] vector (`forward`). The value half is
            # `value_pooled`, exactly what the assembler returns as `vf_combined`.
            assert _seat_out is not None, "the entity seats always join the trunk (E3 is unconditional)"
            # The board: the global token (legacy) or the three board tokens (static, gen3_static_board_v1) —
            # in the trunk's own seat order, so the per-key log-presence `_klp` stays aligned.
            _keys = [our_team_out, their_team_out, self.team_transformer.board_rows(), _seat_out]
            _pads = [ctx.fainted_mask_ours, ctx.fainted_mask_opp,
                     torch.zeros(ctx.batch_size, self.team_transformer.n_board_tokens, dtype=torch.bool,
                                 device=ctx.device), _seat_pad]
            _qlp = _klp
            if belief is not None:
                _brows = belief.view(ctx.batch_size, -1, D_MODEL)
                _keys.append(_brows)
                _pads.append(torch.zeros(ctx.batch_size, _brows.shape[1], dtype=torch.bool,
                                         device=ctx.device))
                if _qlp is not None:
                    _qlp = torch.cat([_qlp, torch.zeros(ctx.batch_size, _brows.shape[1],
                                                        dtype=_qlp.dtype, device=_qlp.device)], dim=1)
            pi_ctx = self.policy_query(torch.cat(_keys, dim=1), torch.cat(_pads, dim=1), _qlp)
            return pi_ctx, value_pooled
        out: Tuple[torch.Tensor, torch.Tensor] = self.assembler(
                             our_team_pooled, their_team_pooled, our_active_refined, value_pooled,
                             ctx, belief)
        return out

    def _value_pooled_routes(self, ctx: ExtractorContext, our_team_out: torch.Tensor,
                             their_team_out: torch.Tensor, belief: Optional[torch.Tensor],
                             damage_block: Optional[torch.Tensor],
                             presence: "Optional[OppPresence]" = None,
                             ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield `(name, [B, D_MODEL] contribution)` for every enabled value route
        (gen3_value_pooled_routes_v1). THE route registry: the gradient-connectivity guard
        (`value_route_gradient_test.py`) iterates exactly this generator, so a route added here
        is covered by construction — and a route added ANYWHERE ELSE is the bug this seam
        exists to prevent. Contract per route: zero-init output projection (cold start adds 0),
        raise when ON but inputs are missing (silence is indistinguishable from a null result).

        FOUR of its five original members were retired by the critic-route deletion wave —
        `intent_value_reduce` (dV 0.3176), `intent_threshold_value` (0.155/0.136), `value_clock`
        (0.2169) and `value_intent` (0.156), all against a 0.39 bar, the first two re-audited at
        2× sample first. `value_entity_pool` is what the audit picked: dV 5.490, **97% of the
        whole critic route joint**. The seam stays at one entry ON PURPOSE — it is the mechanism
        that makes the NEXT route auditable and gradient-guarded the day it is written, and at
        one entry it costs a `for` loop.
        """
        if self.value_entity_pool is not None:
            _op_rows = (self.damage_op.last_tensors.incoming_rows  # type: ignore[union-attr]
                        if (self.damage_op is not None and damage_block is not None) else None)
            _op_alive = ((ctx.hp_and_active[:, :TEAM_SIZE, 0] > 0).float()
                         if _op_rows is not None else None)
            _uvr_kw: Dict[str, object] = {}
            if presence is not None:
                _uvr_kw["presence"] = presence           # X5 fixed_mass: OTHER row + log-π (class E)
            if self.value_entity_pool.full:
                # gen3_static_board_v1: the three refined board tokens take the global row's place (static).
                if self.team_transformer.static_board:
                    _uvr_kw["board_rows"] = self.team_transformer.board_rows()
                else:
                    _uvr_kw["global_row"] = self.team_transformer.last_global_out
                if belief is not None:
                    _uvr_kw["belief_rows"] = belief.view(ctx.batch_size, -1, D_MODEL)
            yield "value_entity_pool", self.value_entity_pool(
                our_team_out, their_team_out, ctx.all_fainted, _op_rows, _op_alive, **_uvr_kw)
