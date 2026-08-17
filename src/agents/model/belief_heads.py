"""Belief heads: the supervised posteriors over the opponent's hidden state — species slots, moves, spread, item, HP type — and the grad-mode contract they publish under.

Split out of `features_extractor.py` 2026-08-16 (one responsibility per file); that module
re-exports every name here, so historical import paths still resolve.
"""
from agents.model.extractor_ctx import Embeddings
import torch
from typing import Dict, Optional, Tuple
from agents.observation.constants import (
    TEAM_SIZE,
)
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
# _PRIOR_FLOOR — the LEGAL-BUT-UNOBSERVED move-prior base (the `--move-candidate-floor` default).
# Legality itself is unconditional; this is only the height of the liftable base a legal-unobserved
# move starts from.
from agents.model.damage_tables import _PRIOR_FLOOR

from agents.model.arch_constants import (D_MODEL,
)




class BeliefSlots(torch.nn.Module):
    """In-place hidden-opponent belief (the live design — supersedes the side-pool `HiddenOppBeliefPool`).

    Instead of summarising the hidden party into K side query tokens (a readout), this REPLACES the
    opponent's un-revealed team slots — which arrive at the encoder as all-zero placeholders (Gen 3
    has no team preview) — with K=TEAM_SIZE **distinct learned "unknown-mon" embeddings**, one per opp
    slot position. The believed mons then sit *in the lineup* and are refined by the SAME 12-token
    `TeamTransformer` and attended over by every downstream readout (`their_cls`, `value_cls`, the
    policy reasoning) — "the model thinks about the hidden mons in latent space" rather than reading a
    side summary.

    Why distinct per-slot params: a permutation-equivariant transformer maps identical inputs to
    identical outputs, so identical zero-slots collapse to one representation (the model can know
    "there are unknowns" but not "slot A leans physical sweeper, slot B special wall"). Independent
    init breaks that symmetry so the slots can specialise — the same trick `HiddenOppBeliefPool` used,
    done in-place. The refined believed tokens are supervised by `BeliefHead` (species + moves),
    which is what makes a slot actually *mean* a Skarmory-shaped wall instead of a generic blob.

    Requires `attend_unrevealed_opponents` (else the believed slots are key-masked out of the
    transformer and never refined). Off ⇒ this module is not built and the opp slots stay zeros
    (baseline arch, byte-for-byte). See `designs/ai_v5/design_offense_and_opponent_belief.md`."""

    def __init__(self):
        super().__init__()
        # One distinct learned token per opponent team-slot position. Same 0.02 init scale as the
        # CLS / belief queries. Slot position is the canonical order the aux labels are matched in.
        self.unknown_slot_emb = torch.nn.Parameter(torch.randn(TEAM_SIZE, D_MODEL) * 0.02)

    def forward(self, role_tokens: torch.Tensor, opp_believed_mask: torch.Tensor) -> torch.Tensor:
        """role_tokens [B, 12, D], opp_believed_mask [B, 6] bool → role_tokens with believed opp
        slots replaced by their learned unknown-token. Revealed (and revealed-then-fainted) opp slots
        keep their encoded token unchanged."""
        batch_size = role_tokens.shape[0]
        our_tokens = role_tokens[:, :TEAM_SIZE, :]
        opp_tokens = role_tokens[:, TEAM_SIZE:, :]                                    # [B, 6, D]
        unknown = self.unknown_slot_emb.unsqueeze(0).expand(batch_size, -1, -1)       # [B, 6, D]
        opp_tokens = torch.where(opp_believed_mask.unsqueeze(-1), unknown, opp_tokens)
        return torch.cat([our_tokens, opp_tokens], dim=1)


class BeliefHead(torch.nn.Module):
    """Auxiliary supervision for the in-place belief slots (the missing "B3" objective).

    Reads the post-transformer opponent team tokens and predicts, per slot, what the hidden mon IS:
    its **species** (cross-entropy) and its **moves** (multi-label BCE). Labels come free from the
    self-play env (it knows the opponent's full team); the loss (computed in `instrumented_ppo`)
    scores ONLY the believed slots, in `BeliefSlots`' canonical slot order. Role is implicit: a
    predicted species routes through the model's existing species/stat/type embeddings, which already
    encode wall-vs-sweeper — so "think Skarmory" supplies the role.

    Returns a dict of logits so a later BYOL/latent-matching target (regress the real hidden mon's
    encoded token) can be added as another key without touching the call sites — the agreed clean
    escalation path off the species+moves v1.

    **Species prior fusion (`species_prior_fusion`, gen3_species_prior_fusion_v1).** Every OTHER belief
    leg in this extractor fuses a PRIOR with a learned DELTA (`MoveBelief.prior_fusion`, plus the
    spread / HP-type / ability legs); the SPECIES head was the one exception — a bare
    `Linear(D_MODEL, n_species)` cold-starting ~uniform over the whole num axis, so `belief/species_acc`
    sat near chance for thousands of updates and every consumer of the posterior read noise (the v36
    expected-latent defender, and the v67 BETA intent head, whose believed-slot resolution asked 110
    rows/update and resolved 0 purely because the posterior was cold). When on, the head's output
    becomes a LEARNED LOG-ODDS DELTA fused additively with a TEAM-COMPOSITION prior:

        species_logits = head_delta + log P(species | the opponent's ALREADY-REVEALED mons)

    The prior is naive Bayes over SMOGON teammate co-occurrence (`build_species_cooccur_prior`,
    the chaos `Teammates` joint — Smogon-sourced since 2026-08-15, never pool-fit) — the
    marginal plus one log-LIFT per revealed teammate — with Species Clause as a hard constraint. It is
    computed ENTIRELY on-GPU from two non-persistent buffers (one `[B,S] @ [S,S]` matmul per forward;
    no host round-trip, no numpy). The delta head is ZERO-INIT, so the cold-start posterior EQUALS the
    prior exactly; OFF reproduces the from-scratch head byte-for-byte."""

    def __init__(self, n_species: int, n_moves: int,
                 species_prior_fusion: bool = False):
        super().__init__()
        self.norm = torch.nn.LayerNorm(D_MODEL)
        self.species_head = torch.nn.Linear(D_MODEL, n_species)
        self.moves_head = torch.nn.Linear(D_MODEL, n_moves)
        self.species_prior_fusion = species_prior_fusion
        if species_prior_fusion:
            from agents.model.damage_tables import build_species_cooccur_prior
            # [n_species] log P(s) + [n_species, n_species] log[P(s|t)/P(s)] — data-derived from the
            # committed pool artifact, recomputable → NON-persistent (the move prior's contract).
            _log_marginal, _log_lift = build_species_cooccur_prior(n_species)
            self.register_buffer("species_prior_log_marginal", _log_marginal, persistent=False)
            self.register_buffer("species_prior_log_lift", _log_lift, persistent=False)
            # Zero-init the head so the cold-start delta is EXACTLY 0 → the fused posterior == the prior
            # at step 0. `restore_identity_init`'s end-of-__init__ observation sweep picks this up
            # automatically, so SB3's ortho pass cannot silently clobber it (ledger M1). Only under
            # fusion; the from-scratch (no-fusion) path keeps the default init unchanged.
            torch.nn.init.zeros_(self.species_head.weight)
            torch.nn.init.zeros_(self.species_head.bias)

    def species_prior_logits(self, opp_species_ids: torch.Tensor,
                             opp_believed_mask: "Optional[torch.Tensor]" = None) -> torch.Tensor:
        """The CONDITIONAL team-composition prior, `[B, 1, n_species]` log-probabilities — broadcast
        over the 6 slots, because "what is in a hidden slot" is a property of the OPPONENT'S TEAM, not
        of which hidden slot you ask about.

        `opp_species_ids` [B,6] national-dex nums (0 = the UNKNOWN sentinel a hidden slot carries);
        `opp_believed_mask` [B,6] bool, True where the slot is still hidden (`ctx.opp_believed_mask`).
        A slot counts as EVIDENCE iff it is revealed AND carries a real num — the two agree by
        construction, and requiring both keeps this single-sourced with the rest of the extractor.

        Naive Bayes, one matmul:

            log P(s | R) ∝ log P(s) + Σ_{r ∈ R} log[ P(s | r) / P(s) ]

        The sum over the revealed set is `revealed_onehot @ log_lift.T`, which is `[B,S] @ [S,S]` —
        deliberately NOT the obvious `log_lift[:, ids]` gather, which would materialize a `[B,6,S]`
        intermediate (157 MB at the production minibatch of 16384) for the same answer.

        SPECIES CLAUSE is applied last and OVERRIDES the evidence: a species already on the board
        cannot also be hiding on the bench. Every entry is FINITE (`SPECIES_CLAUSE_LOGIT`, never
        `-inf`), so the `log_softmax` can never produce a NaN row and the learned delta always has a
        live gradient to lift a floored candidate with.

        The math itself lives in `t0_species.species_team_prior_logits` and is SHARED with the
        T0 resolver (`gen3_t0_species_prior_v1`) — the same belief now also feeds the T1 physics,
        and two copies of a naive-Bayes read would eventually disagree. This method is the
        unsqueeze-to-per-slot wrapper; the body is byte-identical to what it was inline."""
        from agents.model.t0_species import species_team_prior_logits

        return species_team_prior_logits(
            self.species_prior_log_marginal, self.species_prior_log_lift,
            opp_species_ids, opp_believed_mask).unsqueeze(1)            # [B, 1, S]

    def species_logits(self, tokens: torch.Tensor,
                       opp_species_ids: "Optional[torch.Tensor]" = None,
                       opp_believed_mask: "Optional[torch.Tensor]" = None) -> torch.Tensor:
        """tokens [B,6,D] → species logits [B,6,n_species]. Factored out of `forward` (mirrors
        `MoveBelief.move_logits`) so the bidirectional-threat refine can read P(species) over the
        CURRENT opp tokens MID-transformer (per round) and the final op can read it post-transformer —
        without also running the moves/latent heads. `forward` is left byte-identical (it computes its
        own `norm` once and reuses it for both heads); this standalone path is only called by the
        v36 expected-latent-defender (gated off by default), so the baseline forward is unchanged.

        Under `species_prior_fusion` the head output is the learned DELTA and `opp_species_ids` [B,6]
        (national-dex nums) + `opp_believed_mask` [B,6] turn it into the POSTERIOR — exactly the shape
        `MoveBelief.move_logits` takes its `opp_species_ids`/`opp_move_ids` in."""
        read = tokens.detach() if getattr(self, "detach_read", False) else tokens
        logits = self.species_head(self.norm(read))                              # [B, 6, S] (delta)
        if self.species_prior_fusion and opp_species_ids is not None:
            logits = logits + self.species_prior_logits(
                opp_species_ids, opp_believed_mask)                              # posterior = prior ⊕ delta
        return logits

    def species_posterior(self, tokens: torch.Tensor,
                          opp_species_ids: "Optional[torch.Tensor]" = None,
                          opp_believed_mask: "Optional[torch.Tensor]" = None) -> torch.Tensor:
        """tokens [B,6,D] → P(species) [B,6,n_species], the v36 expected-latent defender's input.

        ⚠️ THE SPELLING IS LOAD-BEARING — do not "simplify" this back to `torch.softmax`.

        This one op was the ONLY thing in the whole extractor that Inductor could not codegen, and
        therefore the ONLY reason `--compile-opponents` ever set
        `torch._dynamo.config.suppress_errors`. `torch.softmax` lowers to a `[B,6,n_species]`
        numerator buffer plus a `[B,6,1]` denominator, and the CPU scheduler then trips
        `AssertionError: buf<N>` trying to fuse the division. Reproduced by
        `tmp/inductor_crash_repro.py`; `tmp/softmax_variant_probe.py` shows `.contiguous()`,
        `.clone()`, a 2-D reshape and a hand-rolled `exp / sum` ALL still fail — only the
        `log_softmax().exp()` factoring lowers to a shape Inductor accepts.

        Mathematically identical (`exp(log_softmax(x)) == softmax(x)`, and it goes through the same
        max-subtraction for stability); measured max|Δ| vs eager 5.1e-07, the same order as the
        compile's own float-reassociation noise. Pinned by `inductor_compile_test.py`, which fails
        loudly if a future edit reintroduces the uncompilable form."""
        return torch.log_softmax(
            self.species_logits(tokens, opp_species_ids, opp_believed_mask), dim=-1).exp()

    def forward(self, their_team_out: torch.Tensor,
                opp_species_ids: "Optional[torch.Tensor]" = None,
                opp_believed_mask: "Optional[torch.Tensor]" = None) -> Dict[str, torch.Tensor]:
        """their_team_out [B, 6, D] → {"species": [B,6,n_species], "moves": [B,6,n_moves]}.
        `opp_species_ids` / `opp_believed_mask` are consumed only under `species_prior_fusion`."""
        # gen3_belief_grad_mode_v1: BeliefHead is a pure readout (no reinject), so `detached` simply stops the
        # aux-supervision gradient (species/moves/latent) from reaching the trunk — train the head only.
        read = their_team_out.detach() if getattr(self, "detach_read", False) else their_team_out
        h = self.norm(read)
        species = self.species_head(h)
        if self.species_prior_fusion and opp_species_ids is not None:
            species = species + self.species_prior_logits(opp_species_ids, opp_believed_mask)
        return {"species": species, "moves": self.moves_head(h)}


# The legal `--belief-grad-mode` values — the SINGLE SOURCE for the CLI `choices=`, the extractor's
# ValueError, and `model_version._BELIEF_GRAD_MODE_EFFECT` (pinned to agree by a test). What each one
# cuts is the four-route table in `Gen3FeaturesExtractor.__init__`.
BELIEF_GRAD_MODES = ("shaping", "detached", "label_only")

# gen3_belief_label_only_v1 — the stashes of the FORWARD-CONSUMED belief heads. These are exactly the
# supervised heads whose output reaches pi/vf, and therefore the only ones a policy/value gradient can
# reach BACKWARD: MoveBelief (reinject + the op + the edge cells + the seats), SpreadBelief (reinject +
# the op + the cells), HPTypeBelief (the typed composition + its own type-embedding reinject), and
# AlphaIntentHead (only under `--intent-value-reduce`, which appends an alpha-weighted threat term to
# the CRITIC half — alpha is a pure readout without it, but the flag is what makes it reachable).
#
# THE RULE, and it is per-HEAD rather than per-stash so there is nothing to look up: under `label_only`
# every one of these `last_*` stashes is a STOP-GRAD publication, and a supervised loss must read its
# target through `Gen3FeaturesExtractor.belief_supervision(...)` instead.
#
# The other supervised heads — BeliefHead (species/moves/latent), WinProbHead,
# BetaSwitchHead — are STRUCTURALLY label-only already: they are
# side readouts whose output never re-enters the forward, so no policy gradient can reach them in ANY
# mode and there is nothing here to cut. `belief_label_only_gate_test.py` asserts that rather than
# trusting it, so a head that starts feeding forward fails a test instead of quietly rejoining PPO.
_BELIEF_SUPERVISION_KEYS = frozenset({
    "move_belief_logits", "hp_type_logits",
    "spread_belief", "spread_nature_logits", "spread_ev",
    "alpha_logits", "beta_logits", "item_logits",
})

# Logit at which a REVEALED (certain) move is pinned under prior fusion: sigmoid(10) ≈ 0.99995 ≈ P 1.
_REVEAL_LOGIT = 10.0

# gen3_typed_hp_belief_v1: the logit the bare typeless Hidden Power (num 237) is driven to once the
# typed composition has run — sigmoid(-30) ≈ 9.4e-14, i.e. hard off but FINITE, so the move-belief BCE
# sees a ~0 loss with a ~0 gradient there instead of the NaN a true -inf would produce. 237 survives
# only as the belief's internal PRESENCE channel (read BEFORE this is written); no consumer downstream
# of the composition may treat it as a real move.
_HP_PRESENCE_OFF_LOGIT = -30.0


def mask_typeless_hp(move_logits: torch.Tensor) -> torch.Tensor:
    """Drive the bare typeless Hidden Power channel (num 237) hard-off in a move posterior.

    Shared by BOTH `--hp-belief-mode` arms, because it is not part of what the ablation varies: 237
    carries **BP 0**, so it is not a move at all — leaving it live in the damage candidate set is the
    original "the opponent's Hidden Power reads immune" bug. The ablation varies whether the 16 TYPED
    channels are produced by a presence×type factorisation or predicted independently; both must agree
    that the typeless token is never a candidate."""
    out = move_logits.clone()
    out[..., HIDDEN_POWER_MOVE_NUM] = _HP_PRESENCE_OFF_LOGIT
    return out


class MoveBelief(torch.nn.Module):
    """Predicts each opponent slot's full MOVESET and REINJECTS that prediction back into the slot
    token, so the believed moves actually FLOW into the representation the policy/value heads read —
    not a dead-end aux readout. This is the "make it meaningful" mechanism.

    Per opp slot: a multi-label move head predicts the moveset; the predicted distribution is
    soft-embedded (`sigmoid(logits) @ move_embedding` — the expected moveset embedding), projected
    back to token space, and ADDED to the slot token (a residual, gated to the slots the mode selects).
    The enriched tokens then feed the CLS pools, so the policy reasons about the believed moves. The
    reinject projection is small-init so the enrichment starts ≈0 (no harm before the prediction
    sharpens). The move head's logits are stashed for the aux loss, which supervises against the real
    moveset (revealed slots → direct; hidden slots → Hungarian). `mode` selects which slots are
    enriched + scored: 'revealed' (seen species — predict its UNREVEALED moves, the surprise-OHKO
    new-move gap), 'unrevealed' (hidden species), or 'both'.

    **Prior fusion (`prior_fusion`, the unified two-part belief).** When on, the head's output is treated
    as a LEARNED LOG-ODDS DELTA fused additively with the Smogon move-frequency prior — so the stashed
    `move_logits` carries a proper POSTERIOR over the opponent's moveset, not a from-scratch prediction:
      - REVEALED moves (seen this battle, opp move-id > 0) → pinned CERTAIN (`_REVEAL_LOGIT` ≈ P 1) — the
        "calculate the known moves" half;
      - UNREVEALED moves → `prior_logit(species) + head_delta` — the "pick the unknown, assign a
        probability" half (cold-start = the prior; the head learns the in-battle correction).
    The damage op + the BCE loss both then read ONE coherent posterior (unifying the priors + the learned
    prediction + the damage calc). The prior buffer is a NON-persistent, data-derived lookup; OFF
    reproduces the from-scratch head byte-for-byte. See `designs/ai_v6/design_differentiable_damage_op.md`."""

    def __init__(self, n_moves: int, move_emb_dim: int,
                 prior_fusion: bool = False, n_species: int = 0,
                 move_candidate_floor: float = _PRIOR_FLOOR):
        super().__init__()
        self.move_head = torch.nn.Linear(D_MODEL, n_moves)
        self.reinject = torch.nn.Linear(move_emb_dim, D_MODEL)
        torch.nn.init.normal_(self.reinject.weight, std=0.02)   # start the enrichment ≈0
        torch.nn.init.zeros_(self.reinject.bias)
        self.norm = torch.nn.LayerNorm(D_MODEL)
        self.prior_fusion = prior_fusion
        if prior_fusion:
            from agents.model.damage_tables import build_move_prior_logits
            # [n_species, n_moves] log-odds base rate (data-derived physics, recomputable → non-persistent).
            # LEGALITY IS UNCONDITIONAL (gen3_unconditional_move_legality_v1): a move a species CANNOT
            # LEARN is always ~0 (impossible) — a correctness property, not a toggle. A legal move keeps
            # its TRUE Smogon usage (rare moves stay rare-but-liftable, never pruned); a legal-unobserved
            # move gets `move_candidate_floor` as its small liftable base. The floor is that BASE only —
            # it is no longer an on/off switch (it used to double as one, which is why production's 0.0
            # silently disabled legality altogether).
            self.register_buffer(
                "move_prior_logits",
                build_move_prior_logits(n_species, n_moves, floor=move_candidate_floor),
                persistent=False)
            # Zero-init the head so the cold-start delta is EXACTLY 0 → the fused posterior == the prior at
            # step 0 (the cleanest A/B baseline + matches the docstring claim). Only under fusion; the
            # from-scratch (no-fusion) path keeps the default init unchanged.
            torch.nn.init.zeros_(self.move_head.weight)
            torch.nn.init.zeros_(self.move_head.bias)

    def move_logits(self, opp_tokens: torch.Tensor,
                    opp_species_ids: Optional[torch.Tensor] = None,
                    opp_move_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The POSTERIOR move logits [B,6,M] (NO reinjection) — the head delta, optionally fused with the
        prior + revealed-pinned. Factored out of `forward` so the ITERATIVE refinement path
        (gen3_iterative_damage_v1) can recompute the belief from the MID-transformer opp tokens each round
        without re-running the soft-embed reinjection. When `prior_fusion`, `opp_species_ids` [B,6]
        (national-dex nums) and `opp_move_ids` [B,6,4] (revealed; id>0 ⇒ seen) turn the raw head output into
        the two-part POSTERIOR (prior⊕delta, revealed pinned certain)."""
        # gen3_belief_grad_mode_v1: in `detached` mode the head READS a stop-grad trunk (so neither the
        # supervised belief loss nor the op/policy gradient through this read can reshape the trunk); the
        # reinject WRITE below still rides the LIVE tokens. `getattr` default-False ⇒ unset == byte-identical.
        read = opp_tokens.detach() if getattr(self, "detach_read", False) else opp_tokens
        logits = self.move_head(read)                                            # [B, 6, M] (learned delta)
        if self.prior_fusion and opp_species_ids is not None:
            logits = logits + self.move_prior_logits[opp_species_ids]            # posterior = prior ⊕ delta
            if opp_move_ids is not None:
                # REVEALED moves are certain → pin to a high logit (sigmoid ≈ 1). id 0 = unknown sentinel.
                # BRANCHLESS + SYNC-FREE: the old form did `if bool(valid.any())` then `valid.nonzero()`
                # — TWO host syncs per call, and `move_logits` runs once per refine round + once in
                # `forward` (3×/forward in the production config), stalling the pipeline on the critical
                # path. `scatter_` writes the same mask with no host round-trip. Exactly equivalent: a
                # VALID slot scatters True at its id (always ≥1, since `id > 0` is the validity test); an
                # INVALID slot scatters False at the clamped index 0, which no valid slot can occupy — so
                # the two never collide and the resulting mask is identical, including the all-invalid
                # case (an all-False `revealed` makes the `where` a no-op, exactly like the old skip).
                valid = opp_move_ids > 0                                         # [B, 6, 4]
                ids = opp_move_ids.clamp(0, logits.shape[-1] - 1)                # defensive index clamp
                revealed = torch.zeros_like(logits, dtype=torch.bool)            # [B, 6, M]
                revealed.scatter_(-1, ids, valid)
                logits = torch.where(revealed, _REVEAL_LOGIT, logits)
        return logits

    def reinject_moves(self, opp_tokens: torch.Tensor, apply_mask: torch.Tensor,
                       move_embedding: torch.nn.Embedding,
                       move_logits: torch.Tensor) -> torch.Tensor:
        """Soft-embed a move posterior and residually enrich the opp tokens → [B,6,D].

        Split out of `forward` (gen3_typed_hp_belief_v1) so the caller can interpose the typed-HP
        composition between the head read and the reinjection. That ordering matters: the soft-embed is
        `Σ_m P(m)·move_emb[m]`, so feeding it the RAW posterior would inject `P(HP)·move_emb[237]` — the
        typeless HP row, which carries no type and whose `MOVE_ATTR`/latent row is deliberately all-zero.
        Fed the COMPOSED posterior it instead injects `Σ_t P(HP_t)·move_emb[355+t]`, i.e. the believed
        Hidden Power as a mixture over real typed moves. The enrichment is gated to the selected slots,
        so unselected slots pass through unchanged."""
        # gen3_belief_label_only_v1: under `label_only` the belief is PUBLISHED stop-grad into its own
        # reinjection, so the policy/value gradient cannot reach `move_head` through the token it enriches.
        # Detaching the LOGITS (not `soft_emb`) is deliberate: `move_embedding.weight` is on the far side
        # of the matmul and is NOT a belief parameter, so it keeps learning from the consumer — only the
        # head that PREDICTED the moveset is isolated. `self.reinject` (the consumer-side adapter) likewise
        # trains normally. `getattr` default-False ⇒ unset == byte-identical.
        logits = move_logits.detach() if getattr(self, "publish_detach", False) else move_logits
        soft_emb = torch.sigmoid(logits) @ move_embedding.weight                  # [B, 6, move_emb]
        enriched = opp_tokens + apply_mask.unsqueeze(-1) * self.reinject(soft_emb)
        return self.norm(enriched)

    def forward(self, opp_tokens: torch.Tensor, apply_mask: torch.Tensor,
                move_embedding: torch.nn.Embedding,
                opp_species_ids: Optional[torch.Tensor] = None,
                opp_move_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """opp_tokens [B,6,D], apply_mask [B,6] bool (which slots get the belief), move_embedding table
        → (enriched_tokens [B,6,D], move_logits [B,6,M]). Convenience wrapper over
        `move_logits` + `reinject_moves` with NO typed-HP composition — the extractor drives the two
        steps itself so it can compose in between. Kept for direct/unit use."""
        move_logits = self.move_logits(opp_tokens, opp_species_ids, opp_move_ids)  # [B, 6, M] posterior
        return self.reinject_moves(opp_tokens, apply_mask, move_embedding, move_logits), move_logits


# gen3_nature_ev_belief_v1: the EV-delta head output is scaled by this before adding to the EV prior, so a
# unit logit moves the believed EV by ~_EV_DELTA_SCALE points (the head then learns within the clamped [0,252]).
_EV_DELTA_SCALE = 64.0


class SpreadBelief(torch.nn.Module):
    """Predicts the opponent's hidden SPREAD — the 5 battle-relevant DERIVED stats {atk,def,spa,spd,spe} at
    L100 — per opp slot, and REINJECTS it into the slot token. The THIRD belief leg (moves ✓, species ✓,
    STATS) — `gen3_unified_spread_belief_v1`, mirroring MoveBelief 1:1.

    Two parameterisations (the `nature` flag):
      - ADDITIVE (default): a usage-weighted PRIOR (mean, std per stat, Smogon spreads) ⊕ a learned head DELTA
        in std units → `believed = mean + delta·std`. Simple, but the DERIVED stat is a point estimate that
        sits BETWEEN the nature ×1.1/×0.9 modes → the "over-estimates the largest EV" order-statistic bias.
      - NATURE/EV GENERATIVE (`gen3_nature_ev_belief_v1`, `--spread-belief-nature`): predict a NATURE
        categorical ⊕ its Smogon log-prior + a per-stat EV ⊕ its Smogon prior (the prior-fusion pattern of the
        move/HP-type beliefs), assume IV 31, and COMPUTE `believed = (2·base + 31 + E[EV]/4 + 5)·E[nature_mult]`.
        The nature coupling (exactly one stat ×1.1, one ×0.9 — shared probability mass) + the EV budget are now
        STRUCTURAL, so the head can't inflate every stat → the order-statistic bias is fixed at the source. The
        nature distribution + EV are stashed so the supervised loss (nature CE + EV regression) trains the
        decomposition AND the op (`--spread-belief-nature-marginalize`) can marginalise P(KO) over the natures.

    Either way the output is the same `believed [B,6,5]` DERIVED stat the `DamageOperator` consumes at the opp
    ACTIVE slot (replacing its hand-coded constants), reinjected as a small residual so both heads see it. HP is
    skipped (the op uses the obs HP fraction × a neutral maxhp). Zero-init heads → cold-start == the prior; the
    prior buffers are non-persistent (data-derived) → OFF builds no module (reproduces nothing)."""

    def __init__(self, n_species: int, nature: bool = False):
        super().__init__()
        from agents.model.damage_tables import build_opp_spread_prior, N_SPREAD_STATS
        self.n_stats = N_SPREAD_STATS
        self.nature = nature
        self.reinject = torch.nn.Linear(N_SPREAD_STATS, D_MODEL)
        torch.nn.init.normal_(self.reinject.weight, std=0.02)       # start the enrichment ≈0
        torch.nn.init.zeros_(self.reinject.bias)
        self.norm = torch.nn.LayerNorm(D_MODEL)
        # [n_species, 5, 2] (mean, std) usage prior — non-persistent (recomputable from data/).
        self.register_buffer("spread_prior", build_opp_spread_prior(n_species), persistent=False)
        if not nature:
            self.stat_head = torch.nn.Linear(D_MODEL, N_SPREAD_STATS)   # learned delta in std-units
            torch.nn.init.zeros_(self.stat_head.weight)                 # cold-start delta == 0 → believed == prior
            torch.nn.init.zeros_(self.stat_head.bias)
        else:
            from agents.model.damage_tables import (build_species_nature_prior, build_species_ev_prior,
                                                    build_nature_mult, build_species_base_stats, N_NATURES)
            self.n_natures = N_NATURES
            self.nature_head = torch.nn.Linear(D_MODEL, N_NATURES)      # logit DELTA on the nature log-prior
            torch.nn.init.zeros_(self.nature_head.weight)               # cold-start delta 0 → posterior == prior
            torch.nn.init.zeros_(self.nature_head.bias)
            self.ev_head = torch.nn.Linear(D_MODEL, N_SPREAD_STATS)     # EV DELTA on the EV prior (×_EV_DELTA_SCALE)
            torch.nn.init.zeros_(self.ev_head.weight)
            torch.nn.init.zeros_(self.ev_head.bias)
            self.register_buffer("nature_logprior", build_species_nature_prior(n_species), persistent=False)  # [n,25]
            self.register_buffer("ev_prior", build_species_ev_prior(n_species), persistent=False)             # [n,5]
            self.register_buffer("nature_mult", build_nature_mult(), persistent=False)                        # [25,5]
            self.register_buffer("base_nonhp", build_species_base_stats(n_species), persistent=False)         # [n,5]

    def forward(self, opp_tokens: torch.Tensor, apply_mask: torch.Tensor, opp_species_ids: torch.Tensor):
        """opp_tokens [B,6,D], apply_mask [B,6] bool (which slots get the belief), opp_species_ids [B,6] (nums)
        → (enriched_tokens [B,6,D], believed_stats [B,6,5], nature_logits [B,6,25]|None, ev [B,6,5]|None). The
        residual carries the believed-vs-prior delta into the (selected) token; unselected slots pass through.
        Cold-start (zero deltas) ⇒ believed == the usage prior in BOTH parameterisations."""
        prior = self.spread_prior[opp_species_ids]                  # [B,6,5,2]
        mean, std = prior[..., 0], prior[..., 1]                    # [B,6,5]
        # gen3_belief_grad_mode_v1: `detached` READS a stop-grad trunk for the head(s); reinject below keeps
        # the LIVE `opp_tokens` identity term (so normal policy training still shapes the trunk).
        read = opp_tokens.detach() if getattr(self, "detach_read", False) else opp_tokens
        # gen3_belief_label_only_v1: the reinjection is the ONE forward path that leaves this module from
        # the inside (the returned `believed` is published by the caller), so `label_only` cuts it here.
        # `self.reinject` itself still trains — only the heads that PREDICTED the spread are isolated.
        pub = (lambda t: t.detach()) if getattr(self, "publish_detach", False) else (lambda t: t)
        if not self.nature:
            delta = self.stat_head(read)                            # [B,6,5] (std units; cold == 0)
            believed = (mean + delta * std).clamp(min=1.0)          # [B,6,5] the stat VALUE the op consumes
            enriched = opp_tokens + apply_mask.unsqueeze(-1) * self.reinject(pub(delta))
            return self.norm(enriched), believed, None, None
        # NATURE/EV generative path (gen3_nature_ev_belief_v1): posterior nature ⊕ EV → COMPUTE the derived stat.
        nat_logits = self.nature_logprior[opp_species_ids] + self.nature_head(read)         # [B,6,25] prior⊕delta
        e_mult = torch.softmax(nat_logits, dim=-1) @ self.nature_mult                       # [B,6,5] E[mult]
        ev = (self.ev_prior[opp_species_ids]
              + self.ev_head(read) * _EV_DELTA_SCALE).clamp(0.0, 252.0)                     # [B,6,5]
        base = self.base_nonhp[opp_species_ids]                                             # [B,6,5]
        believed = ((2.0 * base + 31.0 + ev / 4.0 + 5.0) * e_mult).clamp(min=1.0)           # [B,6,5] DERIVED stat
        delta = (believed - mean) / std.clamp(min=1.0)                                      # std-unit residual
        enriched = opp_tokens + apply_mask.unsqueeze(-1) * self.reinject(pub(delta))
        return self.norm(enriched), believed, nat_logits, ev


class ItemBelief(torch.nn.Module):
    """gen3_item_belief_v1 — the opponent's HIDDEN ITEM, per opp slot: a posterior over item
    nums, Smogon prior ⊕ zero-init learned delta (the HPTypeBelief pattern, lean v1 — no token
    reinjection yet).

    Why this belief exists: Gen 3 reveals an item only when it ACTS (Leftovers ticks, a Berry
    fires, Trick) and NEVER reveals Choice Band directly — yet a hidden CB is ×1.5 physical
    Atk. The op's CB-conditional tail priced that with a STATIC per-species usage scalar
    (`SPECIES_CB_PRIOR`); this head replaces that scalar's unrevealed branch with a LEARNED
    read of the trunk's evidence — and the evidence stream now exists: the tokens carry the
    last-action fields and pair tendencies (v79), so "they clicked two DIFFERENT moves ⇒ not
    Choice-locked" is representable, which no static prior can express.

    Cold-start (delta 0) posterior == the Smogon prior exactly. Supervised as the BeliefBank's
    SEVENTH row (CE vs the privileged true item num, revealed slots). Leak-safe: the posterior
    is a model output; the label is training-only. The op consumes only P(item == Choice Band)
    at the active slot, gated by the SAME exactness logic it already had (revealed → 0/1)."""

    def __init__(self, n_species: int, n_items: int):
        super().__init__()
        self.n_items = n_items
        self.norm = torch.nn.LayerNorm(D_MODEL)
        self.item_head = torch.nn.Linear(D_MODEL, n_items)
        torch.nn.init.zeros_(self.item_head.weight)        # cold start: posterior == prior
        torch.nn.init.zeros_(self.item_head.bias)
        from agents.model.damage_tables import build_item_prior
        self.register_buffer("item_prior", build_item_prior(n_species, n_items),
                             persistent=False)

    def forward(self, opp_tokens: torch.Tensor,
                opp_species_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """opp_tokens [B,6,D], opp_species_ids [B,6] nums → (logits, posterior) [B,6,n_items].
        logits = zero-init delta + log(prior[species]); posterior = softmax."""
        read = opp_tokens.detach() if getattr(self, "detach_read", False) else opp_tokens
        delta = self.item_head(self.norm(read))
        prior = self.item_prior[opp_species_ids].clamp_min(1e-6)
        logits = delta + torch.log(prior)
        return logits, torch.softmax(logits, dim=-1)


class HPTypeBelief(torch.nn.Module):
    """Predicts the opponent's hidden HIDDEN-POWER TYPE — per opp slot, a 16-way distribution over
    HIDDEN_POWER_TYPE_ORDER (== the op's HP_TYPE_IDX axis) — FUSED with the Smogon HP-type prior in
    log-odds (mirroring MoveBelief's prior fusion). The `DamageOperator` reads this posterior as its
    typed-HP candidate weights, so an opponent's still-unrevealed Hidden Power is priced as a real ~70-BP
    threat of its most-likely type(s) instead of the all-zero obs `hp_probs` (empty until the opp FIRES HP
    — the "opp HP reads immune" GIGO) or a flat prior. gen3_opp_hp_type_belief_v1, the "force the model to
    guess which Hidden Power it is" head.

    The posterior is stashed at `features_extractor.last_hp_type_logits` and (a) fed to the op as the
    per-slot typed-HP candidate weights — whose damage gradient sharpens this head — and (b) supervised by
    the aux CE loss against the privileged true HP type (from agent2's team). ZERO-INIT so the cold-start
    posterior == the Smogon prior (the clean A/B baseline). Leak-safe: the posterior is a model output; the
    HP-type LABEL rides a training-only obs key read ONLY by the loss, never in pi/vf.

    **gen3_opp_hp_type_belief_v2 (robust):** the belief is ALSO reinjected into the opp token (`reinject`,
    a small-init residual like MoveBelief/SpreadBelief) as the PRESENCE-GATED expected typed-HP embedding —
    so the believed HP type flows into the CLS pools + BOTH heads (attention reasons over "this mon is
    probably an ice/grass HP user"), not only the op's damage block. The presence gate (the move belief's
    P(HP present), ≈1 once `hiddenpower` is revealed — the "presence bit") zeroes the signal when HP is
    unlikely. The posterior stays a full 16-way distribution (it does NOT argmax-collapse), so multiple
    un-ruled-out types remain live candidates the op's top-K simulates distinctly + weights by confidence."""

    def __init__(self, n_species: int, type_emb_dim: int, n_hp: int = 16):
        super().__init__()
        self.norm = torch.nn.LayerNorm(D_MODEL)
        self.type_head = torch.nn.Linear(D_MODEL, n_hp)
        # gen3_typed_hp_belief_v1: the 16 typed-HP dex nums (355-370) in HP_TYPE_ORDER order — the
        # destination of the presence×type composition. Non-persistent (data-derived, recomputable).
        from agents.model.damage_tables import _hp_typed_nums
        self.register_buffer("HP_TYPED_NUMS",
                             torch.tensor(list(_hp_typed_nums()), dtype=torch.long), persistent=False)
        torch.nn.init.zeros_(self.type_head.weight)        # cold-start delta = 0 → posterior == prior
        torch.nn.init.zeros_(self.type_head.bias)
        # gen3_opp_hp_type_belief_v2: the token reinjection of the presence-gated expected typed-HP embedding
        # (the expected type emb = embeddings.hp_soft_type(posterior)). Small-init so the enrichment starts
        # ≈0 (no harm before the belief sharpens), mirroring MoveBelief.reinject / SpreadBelief.reinject.
        self.reinject_proj = torch.nn.Linear(type_emb_dim, D_MODEL)
        torch.nn.init.normal_(self.reinject_proj.weight, std=0.02)
        torch.nn.init.zeros_(self.reinject_proj.bias)
        self.reinject_norm = torch.nn.LayerNorm(D_MODEL)
        from agents.model.damage_tables import build_hp_type_prior
        # [n_species, 16] prob prior (sums to 1) — the log-odds fusion base; non-persistent (data-derived).
        self.register_buffer("hp_prior", build_hp_type_prior(n_species), persistent=False)

    def forward(self, opp_tokens: torch.Tensor,
                opp_species_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """opp_tokens [B,6,D], opp_species_ids [B,6] (national-dex nums) → (logits [B,6,16], posterior
        [B,6,16]). logits = head_delta + log(prior[species]); posterior = softmax. Cold-start (delta 0) ⇒
        posterior == the Smogon prior. The logits feed the CE loss; the posterior feeds the op + reinject."""
        read = opp_tokens.detach() if getattr(self, "detach_read", False) else opp_tokens
        delta = self.type_head(self.norm(read))                             # [B,6,16] (cold == 0)
        prior = self.hp_prior[opp_species_ids].clamp_min(1e-6)              # [B,6,16]
        logits = delta + torch.log(prior)                                  # posterior logit = prior ⊕ delta
        return logits, torch.softmax(logits, dim=-1)

    def compose_typed_hp(self, move_logits: torch.Tensor, posterior: torch.Tensor,
                         obs_hp_probs: torch.Tensor,
                         opp_move_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """**gen3_typed_hp_belief_v1 — the one place a typeless Hidden Power becomes typed.**

        Rewrites the raw move-belief posterior [B,6,M] so that Hidden Power exists ONLY as the 16 real
        typed moves, and returns `(typed_logits, presence)`:

          * ``logits[..., 237]`` → `_HP_PRESENCE_OFF_LOGIT` (sigmoid ≈ 1e-13). The typeless num is a
            belief bookkeeping channel, not a move; nothing downstream — the damage op, the top-K, the
            BCE, the latent grading, the soft-embed reinjection, the prober — may see it as a candidate.
          * ``logits[..., 355+t]`` → ``logit(presence · P(type = t))`` for each of the 16 types.

        **The `Σ_t P(HP_t) = presence` identity is what makes the "a revealed HP must exist somewhere"
        constraint STRUCTURAL rather than a penalty term.** `presence` is read from the raw 237 channel,
        which `MoveBelief.move_logits` pins to `_REVEAL_LOGIT` the moment the opponent is seen using
        `hiddenpower`; so on reveal the 16 typed weights sum to ≈1 and the belief cannot conclude "no
        Hidden Power" no matter what the type head does. It can only be UNSURE ACROSS TYPES — which is
        the honest state, since Gen 3 never discloses the type.

        Two eliminations run first, both of them certain facts rather than learned preferences — the
        "discard the ones that don't make sense" half:

        1. **Rule-out by moveset exhaustion.** If all four of the opponent's moves are revealed and none
           is Hidden Power, it has none: presence → 0. Derived from `opp_move_ids` alone, so it needs no
           extra plumbing and agrees by construction with `HiddenPowerTracker.mark_no_hp`.
        2. **Narrowing by observed effectiveness.** Once the opponent has FIRED Hidden Power, the
           tracker's `obs_hp_probs` has hard-zeroed every type inconsistent with the observed
           effectiveness bucket. Those zeros are CERTAIN physics (not a prior), so the believed type
           distribution is restricted to the survivors and renormalised — the model is forced to spend
           its probability mass only on types that could actually have produced the damage it saw. If
           the belief puts ~no mass on the survivors (an off-meta HP), we fall back to uniform over the
           survivors rather than a ~0 vector, which would silently re-immune the move.

        Shapes: `move_logits` [B,6,M]; `posterior` [B,6,16]; `obs_hp_probs` [B,6,16] (OPP slots);
        `opp_move_ids` [B,6,4]. Differentiable in both `presence` and `posterior`, so the damage
        gradient and the move-belief BCE both sharpen the type head through the typed channels."""
        presence = torch.sigmoid(move_logits[..., HIDDEN_POWER_MOVE_NUM])                   # [B,6]
        # (1) moveset exhaustion — 4 revealed, none of them HP ⇒ certainly no Hidden Power.
        n_revealed = (opp_move_ids > 0).sum(dim=-1)                                     # [B,6]
        hp_seen = (opp_move_ids == HIDDEN_POWER_MOVE_NUM).any(dim=-1)                       # [B,6]
        presence = presence * (~((n_revealed >= opp_move_ids.shape[-1]) & ~hp_seen)).float()
        # (2) effectiveness narrowing — the tracker's zeros are certain, so restrict + renormalise.
        has_obs = obs_hp_probs.sum(-1, keepdim=True) > 0                                # HP has fired
        surv = (obs_hp_probs > 0).float()                                               # certain survivors
        narrowed = posterior * surv
        narrowed = torch.where(narrowed.sum(-1, keepdim=True) > 1e-6, narrowed, surv)   # off-meta fallback
        narrowed = narrowed / narrowed.sum(-1, keepdim=True).clamp_min(1e-6)
        hp_type = torch.where(has_obs, narrowed, posterior)                             # [B,6,16]

        typed_p = (presence.unsqueeze(-1) * hp_type).clamp(1e-9, 1.0 - 1e-6)            # [B,6,16]
        out = move_logits.clone()
        out.index_copy_(-1, self.HP_TYPED_NUMS, torch.logit(typed_p))
        out[..., HIDDEN_POWER_MOVE_NUM] = _HP_PRESENCE_OFF_LOGIT
        return out, presence

    def reinject(self, opp_tokens: torch.Tensor, posterior: torch.Tensor, presence: torch.Tensor,
                 apply_mask: torch.Tensor, embeddings: 'Embeddings') -> torch.Tensor:
        """gen3_opp_hp_type_belief_v2: enrich the opp tokens [B,6,D] with the PRESENCE-GATED expected
        typed-HP embedding, so the believed HP type flows into the CLS pools + both heads (not only the op's
        damage block). `posterior` [B,6,16] (the type belief), `presence` [B,6] (P(HP present) — ≈1 on
        reveal; gates the signal to ≈0 when HP is unlikely), `apply_mask` [B,6] float (which slots to
        enrich — revealed). `embeddings.hp_soft_type(posterior)` = Σ_t P(t)·type_emb[t], the expected type
        embedding. Small-init residual → starts ≈0; LayerNorm'd. Returns the enriched [B,6,D]."""
        # gen3_belief_label_only_v1: both inputs are belief outputs — `posterior` from this head, `presence`
        # from the MOVE head via compose_typed_hp — so `label_only` cuts both, isolating each predictor from
        # the policy gradient. `reinject_proj` (the adapter) still trains.
        if getattr(self, "publish_detach", False):
            posterior, presence = posterior.detach(), presence.detach()
        soft = embeddings.hp_soft_type(posterior)                          # [B,6,type_emb] expected type emb
        gated = presence.unsqueeze(-1) * soft                              # ≈0 when HP unlikely (no spurious)
        enriched = opp_tokens + apply_mask.unsqueeze(-1) * self.reinject_proj(gated)
        return self.reinject_norm(enriched)
