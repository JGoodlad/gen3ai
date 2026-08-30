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
  BOUND `fe.forward` and `cf_terms` calls `type(fe).forward` for a deliberately-eager pass, and
  `instrumented_ppo_test` ASSIGNS `type(fe).forward` — so the concrete class is where that
  attribute has to live for a restore to put it back where it came from.
"""
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

from agents.model.arch_constants import D_MODEL
from agents.model.belief_heads import mask_typeless_hp
from agents.model.extractor_api import ExtractorApi
from agents.model.extractor_ctx import ExtractorContext, PointerInputs, TOKEN_TYPE_HISTORY
from agents.model.extractor_stashes import ExtractorStashes
from agents.model.intent_threshold import threshold_probs
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

    def _apply_move_belief(self, opp_tokens: torch.Tensor,
                           ctx: ExtractorContext) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict + reinject the opp moveset into the given opp tokens [B, 6, D] → (enriched, logits).
        ONE call site: PRE-transformer, T0 RESOLVE (gen3_tiered_pipeline_v1 — the POST-transformer
        placement is deleted). The mask selects the slots per move_belief_mode; the
        species/move ids feed prior-fusion (Smogon prior + pin revealed moves certain).

        gen3_typed_hp_belief_v1: the HP-type head + the typed composition run HERE, between the move
        head's read and the reinjection, so the posterior that leaves this method — and therefore the
        one every consumer reads (`last_move_belief_logits`) — is already typed. The reinjection then
        soft-embeds REAL typed moves rather than the typeless 237 row."""
        if self.move_belief_mode == "revealed":
            mb_mask = ~ctx.opp_believed_mask                 # revealed-species slots
        elif self.move_belief_mode == "unrevealed":
            mb_mask = ctx.opp_believed_mask                  # hidden-species slots
        else:                                                # "both"
            mb_mask = torch.ones_like(ctx.opp_believed_mask)
        raw = self.move_belief.move_logits(  # type: ignore[union-attr]
            opp_tokens,
            ctx.species_ids[:, TEAM_SIZE:],                                  # [B, 6]
            ctx.all_move_ids[:, TEAM_SIZE:, :])                              # [B, 6, 4]
        logits, presence, hp_post, hp_logits = self._typed_hp_posterior(opp_tokens, ctx, raw)
        # gen3_belief_label_only_v1: register the LIVE tensors for the supervised losses BEFORE
        # publishing. `logits` is the TYPED posterior, so it carries BOTH the move head's and the
        # HP-type head's gradient — which is why the move BCE and the HP CE both keep training under
        # `label_only` while every forward consumer downstream reads the stop-grad publication.
        self.stash.belief_supervision["move_belief_logits"] = logits
        self.stash.belief_supervision["hp_type_logits"] = hp_logits
        self.stash.hp_type_logits = self._publish_belief(hp_logits)
        logits = self._publish_belief(logits)  # type: ignore[assignment]
        enriched = self.move_belief.reinject_moves(  # type: ignore[union-attr]
            opp_tokens, mb_mask, self.embeddings.move_embedding, logits)
        # gen3_opp_hp_type_belief_v2: ALSO reinject the presence-gated expected TYPE embedding. This is
        # deliberately not redundant with the move soft-embed above: that one injects believed move
        # IDENTITY (the 355-370 rows), this one injects the believed TYPE in the shared type-embedding
        # space the mon's own types live in — so "this Zapdos threatens ICE" lands in the same geometry
        # attention already uses for type matchups. Revealed slots only. (No head under `flat` — the
        # typed move rows still ride the soft-embed above, which is the point of that ablation.)
        if self.hp_type_belief_head is not None:
            enriched = self.hp_type_belief_head.reinject(
                enriched, hp_post, presence, (~ctx.opp_believed_mask).float(), self.embeddings)  # type: ignore[arg-type]
        return enriched, logits

    def _spread_hp_damage(self, opp_tokens: torch.Tensor, ctx: ExtractorContext
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
        if self.spread_belief is not None:
            (opp_tokens, _believed, _nat_logits, _ev) = self.spread_belief(
                opp_tokens, ~ctx.opp_believed_mask, ctx.species_ids[:, TEAM_SIZE:])
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
                opp_tokens, ctx.species_ids[:, TEAM_SIZE:])
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
        if self.damage_op is not None:
            # Optional gradient-checkpointing (same gate as the transformer): the op materialises several
            # [B,6,~416] activations → recompute in backward for ~GBs of VRAM. Bit-exact (no dropout/RNG);
            # a no-op under inference. ctx is a non-tensor arg (use_reentrant=False); the belief tensors carry
            # the grad. move_latent_all (built above) is the op's top-K identity source (None unless topk on).
            if self.damage_op.grad_checkpointing and torch.is_grad_enabled():
                damage_block = checkpoint(self.damage_op, ctx, self.last_move_belief_logits,
                                          self.last_spread_belief, move_latent_all,
                                          self.stash.t0_species_probs, _item_cb_prob,
                                          use_reentrant=False)
            else:
                damage_block = self.damage_op(ctx, self.last_move_belief_logits, self.last_spread_belief,
                                              move_latent_all, self.stash.t0_species_probs,
                                              item_cb_prob=_item_cb_prob)
        # Read-only stash for the prober/forensic decode — never read by the forward, so off is unchanged.
        self.stash.damage_block = damage_block
        return opp_tokens, damage_block

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
        # In-place hidden-opponent belief: replace the un-revealed opp slots with distinct learned
        # unknown-mon tokens BEFORE the transformer, so the body refines them and every readout
        # attends over them as party members (flag-guarded; None ⇒ baseline zeros).
        if self.belief_slots is not None:
            role_tokens = self.belief_slots(role_tokens, ctx.opp_believed_mask)
        # T0 RESOLVE — the move belief (gen3_tiered_pipeline_v1). Reinject the predicted opp moveset
        # into the opp ROLE tokens BEFORE the transformer, so the believed moves co-refine with the
        # species/team belief through the attention layers. The logits are stashed here; every
        # downstream consumer (damage op, E4 seats, edge cells, aux loss) reads the same
        # `last_move_belief_logits`. There is no second placement.
        if self.move_belief is not None:
            opp_role, _mb_logits = self._apply_move_belief(
                role_tokens[:, TEAM_SIZE:], ctx)
            self.stash.move_belief_logits = _mb_logits
            role_tokens = torch.cat([role_tokens[:, :TEAM_SIZE], opp_role], dim=1)
        # T0 RESOLVE (spread/HP-type) → T1 REASON (the op). Run the WHOLE physics stack ONCE, here,
        # PRE-attention: the spread + HP-type beliefs read the raw opp role tokens (the move belief
        # already did, just above), the FULL DamageOperator runs on that belief, and its per-OUR-mon
        # INCOMING rows are injected onto our role tokens through the zero-init `prefuse_proj` — so
        # attention reasons over the physics. `damage_block` is None only when the op is off, in which
        # case there is nothing to inject (and `prefuse_proj` was never built).
        opp_role, damage_block = self._spread_hp_damage(role_tokens[:, TEAM_SIZE:], ctx)
        if damage_block is not None:
            # gen3_op_tensors_views_v1: the op's typed views (set by the forward that just ran)
            # replace every flat-offset slice on the consumer side.
            inc = self.damage_op.last_tensors.incoming_rows  # type: ignore[union-attr]  # per-OUR-mon incoming rows
            role_tokens = torch.cat(
                [role_tokens[:, :TEAM_SIZE] + self.prefuse_proj(inc), opp_role], dim=1)  # type: ignore[misc]  # residual (0 at init)
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
            self.stash.entity_latent_table)
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
        if self.edge_bias is not None:
            _fams = self.edge_bias.families
            # The T0 stack computed the spread belief THIS forward, pre-trunk (gen3_tiered_pipeline_v1
            # made that unconditional), so it is always the current one. None when the leg is off —
            # the kernels then use their legacy neutral-bulk constants.
            _sb = self.last_spread_belief
            _cells = {}
            if "d1" in _fams:
                _cells["d1"] = self.damage_op.pairwise_outgoing(  # type: ignore[union-attr]
                    ctx, _sb, species_probs=self.stash.t0_species_probs)
            if "c1" in _fams:
                # C1 (outgoing) reuses D1's current-world cells as its delta base when both are
                # on; C1b (incoming) appends the defensive halves — one 6-wide consequence cell.
                _cells["c1"] = torch.cat([
                    self.damage_op.pairwise_boost(ctx, _sb, base=_cells.get("d1"),  # type: ignore[union-attr]
                                                  species_probs=self.stash.t0_species_probs),
                    self.damage_op.pairwise_boost_incoming(  # type: ignore[union-attr]
                        ctx, self.last_move_belief_logits, k_cand=self.consequence_topk),  # type: ignore[arg-type]
                ], dim=-1)
            if "c3" in _fams:
                _cells["c3"] = self.damage_op.pairwise_recovery(  # type: ignore[union-attr]
                    ctx, self.last_move_belief_logits, k_cand=self.consequence_topk)  # type: ignore[arg-type]
            if "c2" in _fams:
                _cells["c2"] = self.damage_op.pairwise_status_consequence(  # type: ignore[union-attr]
                    ctx, self.last_move_belief_logits, _sb, k_cand=self.consequence_topk)  # type: ignore[arg-type]
            if "c5" in _fams:
                _cells["c5"] = self.damage_op.pairwise_baton(ctx, _sb)  # type: ignore[union-attr]
            if "s1" in _fams:
                _cells["s1"] = self.damage_op.discrete_outgoing_status(ctx, per_pair=True)  # type: ignore[union-attr]
            if "d2" in _fams:
                _cells["d2"] = self.damage_op.pairwise_bench_outgoing(ctx, _sb)  # type: ignore[union-attr]
            if "d3" in _fams:
                _cells["d3"] = self.damage_op.pairwise_incoming(  # type: ignore[union-attr]
                    ctx, self.last_move_belief_logits, self.entity_seats.last_cand,  # type: ignore[arg-type]
                    spread_belief=(self.last_spread_belief
                                   if self.damage_op.believed_lean else None))  # type: ignore[union-attr]
            if "d4" in _fams:
                _cells["d4"] = self.damage_op.pairwise_bench_incoming(  # type: ignore[union-attr]
                    ctx, self.last_move_belief_logits, k_bench=self.consequence_topk)  # type: ignore[arg-type]
            if "g" in _fams:
                _cells["g"] = self.damage_op.pairwise_schedule(ctx)  # type: ignore[union-attr]
            if "c4" in _fams:
                # gen3_entity_rehome_v1: protect odds live ON the mon slot now — gather OUR
                # active's per-mon protect field (pokemon.py POKEMON_PROTECT_OFFSET).
                _po = ctx.pokemon_part[
                    torch.arange(ctx.batch_size, device=ctx.device), ctx.our_active_idx,
                    POKEMON_PROTECT_OFFSET]
                _cells["c4"] = self.damage_op.pairwise_protect(ctx, _po)  # type: ignore[union-attr]
            if "x" in _fams:
                _cells["x"] = self.damage_op.pairwise_entry(ctx, self.last_move_belief_logits)  # type: ignore[arg-type,union-attr]
            if "t" in _fams:
                _cells["t"] = self.damage_op.pairwise_trap(ctx)  # type: ignore[union-attr]
            if "v" in _fams:
                _cells["v"] = self.damage_op.pairwise_speed(ctx, _sb)  # type: ignore[union-attr]
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
                    ctx, self.last_move_belief_logits, self.entity_seats.last_cand, per_pair=True)  # type: ignore[arg-type]
            _opp_oh = None
            if "d2" in _fams:
                _opp_oh = torch.zeros(ctx.batch_size, TEAM_SIZE, device=ctx.device)
                _opp_oh[torch.arange(ctx.batch_size, device=ctx.device), ctx.opp_active_local] = 1.0
            _base = self.team_transformer._total_tokens
            _edge_fn = lambda bias: self.edge_bias(bias, _base, _cells, _opp_oh)  # noqa: E731
            _c2_edge_cells = _cells.get("c2")
        else:
            _c2_edge_cells = None
        # gen3_intent_move_cell_v1 (G3): the RAW c2-for-the-move-cell operands, computed HERE —
        # still T1, where every other op kernel runs (alpha is T2 and does not exist yet; the
        # weighting happens at the pointer stash below, the same T1-producer/T2-consumer split as
        # `last_pair_cells`). Reuses the c2 edge grid when the edge family already built it this
        # forward — identical function, so the value is the same either way.
        _imc_ops = None
        if self.intent_move_cell is not None and damage_block is not None:
            _imc_ops = self.damage_op.pointer_intent_status_operands(  # type: ignore[union-attr]
                ctx, self.last_move_belief_logits, self.last_spread_belief,  # type: ignore[arg-type]
                k_cand=self.consequence_topk, c2_cells=_c2_edge_cells)
        our_team_out, their_team_out, _seat_out = self.team_transformer(
            role_tokens, ctx, self.embeddings,
            extra=(_seat_tokens, _seat_types, _seat_pad),
            edge_bias_fn=_edge_fn)
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
        # gen3_pair_value_route_v1 (v95, PV — design_opponent_intent.md §7a(2)): the α-reduced
        # unified outcome row per OUR mon j, as TOKEN CONTENT on the value pool's copy of mon j's
        # token. ⚠️ α is the R1 `belief_mean` rung UNCONDITIONALLY, and that is ORDERING rather than
        # preference: the α/β heads are scored BELOW this line, so the publication does not exist
        # yet. §7a(2) pre-registers exactly this substitution, which separates the DELIVERY claim
        # from the DISTRIBUTION claim — and `pair_alpha` documents loudly that a presence belief and
        # a usage belief are not the same object.
        _pv_rows = None
        if self.pair_value_route:
            _pv_pin = self.damage_op.last_pair_in if self.damage_op is not None else None
            _pv_w = self.damage_op.last_topk_w if self.damage_op is not None else None
            if _pv_pin is None or _pv_w is None:
                raise RuntimeError(
                    "pair_value_route is on but the op stashed no unified outcome vector (or no "
                    "top-K belief weights) — the route would silently contribute nothing, which is "
                    "indistinguishable from a null RESULT. Requires damage_topk_k>0 (and the "
                    "incoming matrix that computes it).")
            _pv_rows = reduce_pair_in_all(
                pair_alpha(None, _pv_w, self.damage_op.last_pair_seat_live),  # type: ignore[union-attr]
                _pv_pin, self.damage_op.last_pair_gate)  # type: ignore[arg-type,union-attr]
        our_team_pooled, their_team_pooled, our_active_refined, value_pooled = self.cls_pool(
            our_team_out, their_team_out, ctx,
            threat_rows=(self.damage_op.last_reduced_extra  # type: ignore[union-attr]
                         if self.value_threat_inject else None),
            pair_rows=_pv_rows,
        )
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
        # gen3_opp_intent_v1: ALPHA (which of their believed moves will they click, or SWITCH) and
        # BETA (if they switch, to whom). Both are POINTER heads over objects that already exist —
        # alpha over the E4 believed-threat seats, beta over their six team tokens — so both are
        # equivariant under permuting what they point at. Supervision-only: the input is DETACHED, so
        # a null result says "the head cannot predict the opponent", not "predicting the opponent
        # perturbed the policy". Stashed for the loss + the prober; never fed forward.
        if self.alpha_head is not None:
            _K = self.entity_topk_seats
            _cand = self.entity_seats.last_cand
            if _cand is None:
                raise RuntimeError(
                    "opp_intent is on but the E4 seat builder stashed no candidate selection — "
                    "alpha's seats and its move-num labels would come from different selections.")
            # gen3_intent_grad_mode_v1. `detached` (default) keeps alpha/beta pure SUPERVISION:
            # a null then says "the head cannot predict the opponent", not "predicting the opponent
            # perturbed the policy" — two very different findings, and the detach is what keeps
            # them apart. `shaping` lets the intent gradient into the trunk, which is the regime
            # step 6 needs (a reduction weighted by alpha is only as good as alpha's read of THIS
            # board) and buys the opposite risk: the aux objective can now fight the RL one. That
            # is why `grad/opp_intent_policy_cosine` ships WITH this flag rather than after it — a
            # persistently negative cosine means the two objectives disagree about the trunk, and
            # without the number a shaping run would just look like a slow one.
            _keep = self.opp_intent_grad_mode == "shaping"
            _seat_feats = _seat_out[:, 4:4 + _K, :]                                # [B,K,D]
            _ictx = torch.cat([our_team_pooled, their_team_pooled], dim=-1)
            if not _keep:
                _seat_feats, _ictx = _seat_feats.detach(), _ictx.detach()
            _seat_nums = _cand[0]                                                  # [B,K] move NUMS
            self.stash.alpha_seat_nums = _seat_nums.detach()
            _alpha = self.alpha_head(_seat_feats, _ictx, seat_valid=(_seat_nums > 0).float())
            # gen3_belief_label_only_v1: alpha is a pure readout UNTIL `--intent-value-reduce`, which
            # appends an alpha-weighted threat term to the CRITIC half (below) — that flag is what makes
            # the value gradient able to reach `alpha_head`, and therefore what puts alpha in the
            # label_only set. Publishing unconditionally keeps the one rule: a forward-consumed belief
            # head's stash IS the publication, so turning the flag on later cannot reopen the route.
            self.stash.belief_supervision["alpha_logits"] = _alpha
            self.stash.alpha_logits = self._publish_belief(_alpha)
            # BETA's candidates: every slot they could legally bring in. Legality is a MASK, never
            # something the head has to learn — an illegal switch-in must be unrepresentable.
            #
            # ⚠️ A REVEALED slot and a BELIEVED slot mean different things by `hp == 0`, and
            # conflating them silently deletes half of beta's job. MEASURED
            # (`tmp/beta_slot_probe.py`, 12 real battles): unrevealed opp slots encode hp EXACTLY
            # 0.000 in 1033/1033 cases — for them 0 means UNKNOWN, not DEAD. Masking on `hp>0`
            # therefore made every hidden mon unaddressable, and the ~46% of switches that bring
            # one (G2a) went from "unsupervised" to "unrepresentable".
            #
            # A believed slot is ALWAYS a legal target, and that is exact rather than heuristic:
            # a Pokemon cannot faint without being revealed, so an unrevealed mon is alive.
            # gen3_opp_addressable_v1: the ADDRESSABILITY half is single-sourced on the context
            # (see ObsUnpack) — beta additionally excludes the current ACTIVE (you cannot switch
            # to the mon already in). Same formula as before, one home for the hp-means-unknown
            # rule.
            _opp_active_flag = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1]   # [B,6]
            _beta_mask = (ctx.opp_addressable & (_opp_active_flag < 0.5)).float()  # [B,6]
            # gen3_intent_conditional_v1 (class B): beta is now PUBLISHED like alpha — the
            # boom trade-value cell consumes it forward-side, so under label_only the policy
            # gradient must be cut at this one boundary while the supervised intent loss keeps
            # the LIVE view (the alpha pattern exactly).
            _beta_live = self.beta_head(  # type: ignore[misc]
                their_team_out.detach(), _ictx, candidate_mask=_beta_mask)
            self.stash.belief_supervision["beta_logits"] = _beta_live
            self.stash.beta_logits = self._publish_belief(_beta_live)
        _tok_req = _seat_out[:, :4, :]
        if self.damage_op is not None and damage_block is not None:
            _mcells, _scells = self.damage_op.pointer_cells(damage_block)
        else:
            _mcells = _tok_req.new_zeros(ctx.batch_size, _tok_req.shape[1], 0)
            _scells = our_team_out.new_zeros(ctx.batch_size, TEAM_SIZE, 0)
        # gen3_intent_move_cell_v1 (G3): alpha consumed on the POLICY side — the c2 re-delivery
        # channels join the pointer MOVE cell HERE, the first point where both operands exist
        # (the op's T1 operand stash from above, and alpha, T2, scored from the seats and pools).
        # The consumer reads `last_alpha_logits` — the PUBLICATION, stop-grad under
        # `belief_grad_mode=label_only` — never a raw stash, so label_only keeps cutting the
        # PPO→alpha_head route through this path exactly as it does for every other consumer.
        if self.intent_move_cell is not None:
            if self.last_alpha_logits is None or _imc_ops is None:
                raise RuntimeError(
                    "intent_move_cell is on but alpha produced no logits or the op stashed no c2 "
                    "operands — the cell would silently contribute nothing, which is "
                    "indistinguishable from a null RESULT.")
            _mcells = torch.cat([_mcells, self.intent_move_cell(
                self.last_alpha_logits, *_imc_ops)], dim=2)
        # gen3_intent_threshold_v1 (v84): the α-weighted threshold operator, computed ONCE here
        # (the first point where α exists) and consumed by BOTH heads — the move-cell block joins
        # the pointer cells now; the vf block reads the stashed probs at the value tail (a
        # T2-produced tensor read at T3 — the allowed direction). The consumer reads
        # `last_alpha_logits` — the PUBLICATION, stop-grad under `belief_grad_mode=label_only`.
        if self.intent_threshold_move is not None:
            _pair_cells = self.damage_op.last_pair_cells if self.damage_op is not None else None
            if self.last_alpha_logits is None or _pair_cells is None:
                raise RuntimeError(
                    "intent_threshold is on but alpha produced no logits or the op stashed no "
                    "pair cells — the thresholds would silently contribute nothing, which is "
                    "indistinguishable from a null RESULT. Requires damage_topk_k>0 (and the "
                    "incoming matrix that computes it).")
            _tp = threshold_probs(
                self.last_alpha_logits, _pair_cells, self.damage_op.last_pair_gate,  # type: ignore[arg-type,union-attr]
                ctx.our_active_idx)
            self.stash.thresh_probs = _tp
            _mcells = torch.cat([_mcells, self.intent_threshold_move(
                *_tp, ctx.our_active_req_move_ids)], dim=2)
        # gen3_intent_conditional_v1 (v85): the Counter/flinch/Explosion/Pursuit cells — same
        # T1-producer/T2-consumer split, same publication read.
        if self.intent_conditional is not None:
            _pc = self.damage_op.last_pair_cells if self.damage_op is not None else None
            _ot = self.damage_op.last_tensors if self.damage_op is not None else None
            _ready = (self.last_alpha_logits is not None and _pc is not None
                      and _ot is not None and _ot.out_per_move is not None
                      and self.damage_op.last_out_pko is not None  # type: ignore[union-attr]
                      and self.last_beta_logits is not None
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
            _mcells = torch.cat([_mcells, self.intent_conditional(
                self.last_alpha_logits, _pc, self.damage_op.last_pair_gate,  # type: ignore[union-attr]
                ctx.our_active_idx, self.damage_op.last_topk_idx,  # type: ignore[union-attr]
                _ot.out_per_move[..., 1], _ot.out_p_outspeed,  # type: ignore[index,union-attr]
                _ot.out_secondary[..., _OUT_SEC_FLINCH_COL],  # type: ignore[index,union-attr]
                ctx.our_active_req_move_ids, _po,
                self.last_beta_logits, self.damage_op.last_out_pko,  # type: ignore[union-attr]
                ctx.opp_active_local)], dim=2)
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
            _pin = self.damage_op.last_pair_in if self.damage_op is not None else None
            _pw = self.damage_op.last_topk_w if self.damage_op is not None else None
            if _pin is None or _pw is None:
                raise RuntimeError(
                    "pair_outcome_cell is on but the op stashed no unified outcome vector (or no "
                    "top-K belief weights) — the cell would silently contribute nothing, which is "
                    "indistinguishable from a null RESULT. Requires damage_topk_k>0 (and the "
                    "incoming matrix that computes it).")
            _alpha = pair_alpha(self.last_alpha_logits, _pw,
                                self.damage_op.last_pair_seat_live)  # type: ignore[union-attr]
            _row = reduce_pair_in(
                _alpha, _pin, self.damage_op.last_pair_gate,  # type: ignore[arg-type,union-attr]
                ctx.our_active_idx)
            _mcells = torch.cat([_mcells, self.pair_outcome_move(_row)], dim=2)
        # gen3_pair_outcome_switch_v1 (v94): the SAME reduction, at EVERY defender, into the
        # pointer SWITCH cell — `design_pair_reduction.md` §2.1's own defect, at its own sink. One
        # α (no J axis ⇒ D3 stays a shape error) producing six rows, each riding its own mon's
        # logit, so the module is equivariant in our team axis by construction.
        if self.pair_outcome_switch is not None:
            _pin_s = self.damage_op.last_pair_in if self.damage_op is not None else None
            _pw_s = self.damage_op.last_topk_w if self.damage_op is not None else None
            _tn_s = self.damage_op.last_topk_idx if self.damage_op is not None else None
            if _pin_s is None or _pw_s is None or _tn_s is None:
                raise RuntimeError(
                    "pair_outcome_switch is on but the op stashed no unified outcome vector (or no "
                    "top-K belief weights / move nums) — the cell would silently contribute "
                    "nothing, which is indistinguishable from a null RESULT. Requires "
                    "damage_topk_k>0 (and the incoming matrix that computes it).")
            _alpha_s = pair_alpha(self.last_alpha_logits, _pw_s,
                                  self.damage_op.last_pair_seat_live)  # type: ignore[union-attr]
            _rows = reduce_pair_in_all(
                _alpha_s, _pin_s, self.damage_op.last_pair_gate)  # type: ignore[arg-type,union-attr]
            _scells = torch.cat([_scells, self.pair_outcome_switch(
                _rows, _alpha_s, _tn_s,
                ctx.type1_ids[:, :TEAM_SIZE], ctx.type2_ids[:, :TEAM_SIZE],
                # index 1 of the hazard pair is THEIR side — the layers WE set, which is exactly
                # what their Rapid Spin would remove and a Ghost switch-in would preserve.
                ctx.spikes_feature[:, 1:2])], dim=2)
        # gen3_conditional_threat_v1 (v95): OA1 — the SECOND widener of the switch cell. Same α
        # ladder, same (defender, seat) grid, DIFFERENT quantities: the accuracy-folded P(this mon
        # dies) (§0.2(2) — a thin tanh scorer cannot multiply two of its own inputs), the
        # bulk-INDEPENDENT expected type multiplier (the one cell channel `pair_in` never carried),
        # and the two §0.2(3) MARGINS against our own HP. §1.2's λ-weighted `w` is NOT built — see
        # the substitution table in `conditional_threat.py`.
        if self.conditional_threat is not None:
            _ct_pin = self.damage_op.last_pair_in if self.damage_op is not None else None
            _ct_w = self.damage_op.last_topk_w if self.damage_op is not None else None
            _ct_tm = self.damage_op.last_pair_type_mult if self.damage_op is not None else None
            if _ct_pin is None or _ct_w is None or _ct_tm is None:
                raise RuntimeError(
                    "conditional_threat_cell is on but the op stashed no unified outcome vector / "
                    "top-K belief weights / type multiplier — the cell would silently contribute "
                    "nothing, which is indistinguishable from a null RESULT. Requires "
                    "damage_topk_k>0 and the incoming matrix that computes both.")
            _scells = torch.cat([_scells, self.conditional_threat(
                pair_alpha(self.last_alpha_logits, _ct_w,
                           self.damage_op.last_pair_seat_live),  # type: ignore[union-attr]
                _ct_pin, _ct_tm, self.damage_op.last_pair_gate,  # type: ignore[union-attr]
                ctx.hp_and_active[:, :TEAM_SIZE, 0])], dim=2)
        # gen3_switch_branch_v1 (v94): OA2 + the Rapid-Spin spinblock + Protect's α-conditioning —
        # the per-request-slot content of the branch in which the OPPONENT switches. The last
        # move-cell rider, and the only one that consumes β forward-side besides v85's boom trade.
        if self.switch_branch is not None:
            _oc = self.damage_op.last_out_cells if self.damage_op is not None else None
            _pg = self.damage_op.last_opp_p_ghost if self.damage_op is not None else None
            _tn_b = self.damage_op.last_topk_idx if self.damage_op is not None else None
            _sl_b = self.damage_op.last_pair_seat_live if self.damage_op is not None else None
            if (self.last_alpha_logits is None or self.last_beta_logits is None
                    or _oc is None or _pg is None or _tn_b is None or _sl_b is None):
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
                self.last_alpha_logits, self.last_beta_logits, _sl_b, _tn_b, _oc, _pg,
                ctx.opp_active_local, ctx.our_active_req_move_ids, _po_b,
                # index 0 of the hazard pair is OUR side — what OUR Rapid Spin would remove, and
                # therefore the stake a spinblock destroys.
                ctx.spikes_feature[:, 0:1])], dim=2)
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
            belief = self.hidden_opp_belief(all_team_out, ctx.all_fainted, ctx.batch_size)
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
                                                               damage_block):
            value_pooled = value_pooled + _contrib
        # Read-only stash of the value-CLS pool (the critic's whole-board "who's winning" summary, the
        # 128-dim FitNets HINT layer). Consumed ONLY by the FitNets value-feature distillation
        # (`instrumented_ppo._value_feat_distill`): both student and teacher forwards leave it here, so the
        # distill loop can regress the student's value_pooled toward each teacher's on the teacher-team
        # states. NOT read by the forward → off-path/eval is byte-identical; carries grad on the student pass
        # (a live activation) so the cosine distill gradient flows into the shared trunk.
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
        # Distributional VALUE readout (flag-guarded; None when off). Same value_pooled the win head
        # reads → per-atom return-distribution logits, stashed for the aux loss + prober/eval. NOT fed
        # into the assembler (a side readout — the value target can't leak into pi/vf). `read_only`
        # feeds a STOP-GRAD value_pooled (head-only training); `shaping` feeds it live. Computed on
        # every forward (one small MLP) so eval/inference can read the distribution too.
        if self.value_dist_head is not None:
            vd_in = value_pooled if self.value_dist_mode == "shaping" else value_pooled.detach()
            self.stash.value_dist_logits = self.value_dist_head(vd_in)
        # gen3_q_winprob_head_v1: the PER-ACTION win-probability readout — the amortized one-ply
        # search leaf (E5 step 1). It scores the SAME per-action tokens the pointer head scores, so
        # it is computed here, AFTER `stash.pointer_inputs` is written and after every value route
        # has landed in `value_pooled` (the head's context is the FINAL summary, the same tensor
        # the win head and the dist critic read).
        #
        # EVERY input is detached — the tokens, the cells and the context alike. `read_only` is the
        # only live mode and there is no `shaping` counterpart, so this head trains its own
        # parameters and provably cannot perturb the trunk: pi/vf are bit-identical at any
        # coefficient, not merely equal in shape. Computed on every forward (one small MLP over 11
        # slots) so a rollout, an eval and the prober can all read P(win|s,a) from the forward that
        # chose the action — which is the entire point of amortizing the search leaf.
        if self.q_winprob_head is not None:
            _pi_in = self.stash.pointer_inputs
            if _pi_in is None:                       # pragma: no cover - structurally unreachable
                raise RuntimeError(
                    "q_winprob_head is built but pointer_inputs was not stashed by this forward — "
                    "the Q head scores the pointer head's own action tokens, so a missing stash "
                    "means the two are wired to different forwards.")
            self.stash.q_winprob_logits = self.q_winprob_head(
                value_pooled.detach(),
                _pi_in.move_tokens.detach(), _pi_in.move_valid.detach(),
                _pi_in.team_tokens.detach(), _pi_in.move_cells.detach(),
                _pi_in.switch_cells.detach())
        out: Tuple[torch.Tensor, torch.Tensor] = self.assembler(
                             our_team_pooled, their_team_pooled, our_active_refined, value_pooled,
                             ctx, belief)
        return out

    def _value_pooled_routes(self, ctx: ExtractorContext, our_team_out: torch.Tensor,
                             their_team_out: torch.Tensor, belief: Optional[torch.Tensor],
                             damage_block: Optional[torch.Tensor]
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
            _uvr_kw = {}
            if self.value_entity_pool.full:
                _uvr_kw["global_row"] = self.team_transformer.last_global_out
                if belief is not None:
                    _uvr_kw["belief_rows"] = belief.view(ctx.batch_size, -1, D_MODEL)
            yield "value_entity_pool", self.value_entity_pool(
                our_team_out, their_team_out, ctx.all_fainted, _op_rows, _op_alive, **_uvr_kw)
