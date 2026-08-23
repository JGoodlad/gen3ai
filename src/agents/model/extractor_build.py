"""`ExtractorBuild` — `Gen3FeaturesExtractor.__init__`: every flag validation, every module.

Split out of `features_extractor.py` 2026-08-23 (one responsibility per file). It is a BASE
CLASS rather than a helper function on purpose: SB3 constructs the extractor as
`features_extractor_class(observation_space, **features_extractor_kwargs)` and ~10 call sites
read `inspect.signature(Gen3FeaturesExtractor.__init__).parameters` as the flag surface, so the
constructor has to stay a real method with a real signature. Inheriting it preserves both
(`Gen3FeaturesExtractor.__init__` IS this function), and it changes no attribute PATH, so the
`state_dict` keys are byte-identical.

⚠️ **THE CONSTRUCTOR IS DELIBERATELY NOT SPLIT FURTHER**, for the same reason
`instrumented_ppo.train()` is not: the property that matters is only checkable by reading it as
ONE straight line. Here that property is MODULE CONSTRUCTION ORDER — SB3 restores optimizer
state POSITIONALLY (the ai_v6_13 "128 vs 5" crash), so a new module must be APPENDED, never
inserted, and several comments in the body say exactly where in this order they sit and why.
Splitting validation from construction would also break `flag_requires_test._guarded_raises`,
which walks THIS function's body for the flag-coupling raises (it resolves the file from
`Gen3FeaturesExtractor.__init__`, so it follows the constructor wherever it lives).
"""
from typing import Any, Dict, Optional, Tuple

import torch
from gymnasium import spaces

from agents.model.arch_constants import (
    CONDITIONAL_THREAT_SWITCH_DIM, D_MODEL, INTENT_COND_MOVE_DIM, INTENT_MOVE_CELL_DIM,
    INTENT_THRESH_MOVE_DIM, PAIR_OUTCOME_MOVE_DIM, PAIR_OUTCOME_SWITCH_DIM,
    PAIR_VALUE_ROUTE_DIM, PROJECTION_DIM, ROLE_TOKEN_SIZE, SWITCH_BRANCH_MOVE_DIM,
)
from agents.model.aux_value_heads import (
    CfEvidentialHead, ShadowValueHead, ValueDistHead, WinProbHead)
from agents.model.belief_heads import (
    BELIEF_GRAD_MODES, BeliefHead, BeliefSlots, HPTypeBelief, ItemBelief, MoveBelief, SpreadBelief)
from agents.model.conditional_threat import ConditionalThreatCell
from agents.model.damage_op import DamageOperator, _DMG_PER_MON
from agents.model.damage_tables import _PRIOR_FLOOR
from agents.model.encoders import PokemonEncoder
from agents.model.extractor_ctx import Embeddings, ObsUnpack
from agents.model.extractor_stashes import ExtractorStashes
from agents.model.intent_conditional import IntentConditionalMoveCell
from agents.model.intent_move_cell import IntentMoveCell
from agents.model.intent_threshold import IntentThresholdMoveCell
from agents.model.opp_intent import AlphaIntentHead, BetaSwitchHead
from agents.model.pair_outcome import PairOutcomeMoveCell, PairOutcomeSwitchCell
from agents.model.pointer_head import EntityMoveSeats
from agents.model.pools import CLSPool, HiddenOppBeliefPool
from agents.model.projection import ProjectionAssembler, compute_projection_widths
from agents.model.switch_branch import SwitchBranchMoveCell
from agents.model.t0_species import T0SpeciesPrior
from agents.model.team_transformer import EdgeBias, EventSeats, TeamTransformer
from agents.model.value_readouts import UnifiedValueReadout
from agents.model.value_threat_inject import (
    VALUE_THREAT_INJECT_REDUCE_HOW, value_threat_inject_dim)
from utils.logging.levels import LogLevel


class ExtractorBuild(torch.nn.Module):
    """The construction half of `Gen3FeaturesExtractor` — see that class for what it builds."""

    # `observation_space` is DELIBERATELY UNREAD. SB3 constructs every features extractor as
    # `features_extractor_class(observation_space, **features_extractor_kwargs)`, so the positional
    # parameter is its construction contract and cannot be dropped — but this extractor sizes
    # everything from `layout` (Gen3ObservationEncoder.get_layout(), the single source of truth
    # for every obs offset), never from the space. Tests and probes exploit that: they pass a
    # plain Box of the right total_dim. Typed `spaces.Space` (not Dict) to record exactly that.
    def __init__(self, observation_space: spaces.Space, layout: Optional[Dict[str, Any]] = None,
                 mappings: Optional[Dict[str, Any]] = None, log_level: LogLevel = LogLevel.QUIET,
                 attend_unrevealed_opponents: bool = False, opp_belief_cls_k: int = 0,
                 opp_belief_slots: bool = False,
                 move_belief_mode: str = "off",
                 damage_op: bool = False, move_prior_fusion: bool = False,
                 win_prob_mode: str = "none",
                 damage_outgoing: bool = False, move_candidate_floor: float = _PRIOR_FLOOR,
                 move_latent: bool = False, spread_belief: bool = False, spread_belief_nature: bool = False,
                 value_dist_mode: str = "none", value_dist_bins: int = 0,
                 value_dist_vmin: float = 0.0, value_dist_vmax: float = 0.0,
                 value_threat_inject: bool = False,
                 opp_intent: bool = False, species_prior_fusion: bool = False,
                 t0_species_prior: bool = False,
                 opp_intent_grad_mode: str = "detached",
                 intent_move_cell: bool = False,
                 intent_threshold: bool = False,
                 intent_conditional: bool = False,
                 pair_outcome_cell: bool = False,
                 pair_outcome_switch: bool = False,
                 switch_branch_cell: bool = False,
                 conditional_threat_cell: bool = False,
                 pair_value_route: bool = False,
                 op_drop_renders: bool = False,
                 op_believed_lean: bool = False,
                 value_entity_pool: bool = False,
                 value_entity_pool_full: bool = False,
                 item_belief: bool = False,
                 history_events: bool = False,
                 damage_topk_k: int = 0,
                 damage_candidate_k: int = 0,
                 entity_topk_seats: int = 0,
                 consequence_topk: int = 6,
                 entity_tail_seats: bool = False,
                 edge_bias_families: str = "off",
                 damage_matrices_outgoing: bool = False, damage_matrices_incoming: bool = False,
                 threat_prob_outspeed: bool = False,
                 hp_belief_mode: str = "composed", belief_grad_mode: str = "shaping",
                 cf_evidential: bool = False,
                 cf_twin_heads: bool = False, cf_shadow_critic: bool = False,
                 ):
        super().__init__()
        # gen3_extractor_stashes_v1 (4b): `layout` is Optional in the SIGNATURE only because SB3
        # builds the extractor from `features_extractor_kwargs`; every real construction passes
        # it and `Embeddings` indexes it immediately — so absence fails loud HERE, with the fix
        # named, instead of as a deep `TypeError: 'NoneType' object is not subscriptable`.
        if layout is None:
            raise ValueError(
                "Gen3FeaturesExtractor requires layout=Gen3ObservationEncoder(mappings)"
                ".get_layout() — pass it via features_extractor_kwargs (SB3) or directly. "
                "The default of None exists only to satisfy SB3's keyword-forwarding "
                "construction contract and is not a usable value.")
        self.layout = layout
        self.mappings = mappings
        self.log_level = log_level
        # Behavioral toggle (no weight-shape change): unmask the opponent's still-hidden
        # party so the transformer attends to it. Version-checked, not in ARCH_SIGNATURE.
        self.attend_unrevealed_opponents = attend_unrevealed_opponents
        # gen3_cpu_damage_deleted_v1: the three --unified-obs ablation masks are gone with their blocks.
        # Hidden-opponent belief: opp_belief_cls_k = number of learned belief query tokens.
        # 0 = OFF (no module, baseline arch — reproduces it byte-for-byte, so no ARCH_SIGNATURE bump);
        # k>0 builds HiddenOppBeliefPool(k) and widens both projection inputs by k*D_MODEL (a
        # WEIGHT-SHAPE change, version-checked like use_popart). k>0 hard-requires the unmask flag:
        # with the hidden slots masked the belief queries would read a board with them deleted.
        if opp_belief_cls_k < 0:
            raise ValueError(f"opp_belief_cls_k must be >= 0 (0 = off), got {opp_belief_cls_k}")
        self.opp_belief_cls_k = opp_belief_cls_k
        if opp_belief_cls_k > 0 and not attend_unrevealed_opponents:
            raise ValueError(
                "opp_belief_cls_k > 0 requires attend_unrevealed_opponents=True — the hidden-opponent "
                "belief queries read the unrevealed opp slots, which are key-masked out unless the "
                "unmask flag is on. Enable --attend-unrevealed-opponents, or set --opp-belief-cls-k 0."
            )

        # gen3_unified_move_system_v1: the mechanics-grounded move latent (structural toggle — widens the
        # move-network input → state_dict-changing; OFF byte-identical). Required by the Stage-3 latent
        # grading aux (the loss reads its latent_table).
        self.move_latent = move_latent

        # gen3_belief_grad_mode_v1 / gen3_belief_label_only_v1 — WHICH gradient arrow between the
        # STATE-prediction belief heads (move / spread / hp-type / the species-moves-latent aux) and the
        # rest of the network is cut. There are FOUR routes, and the modes cut DIFFERENT ones:
        #
        #   route                                              shaping   detached   label_only
        #   A  label loss  -> belief head params                  on         on          on
        #   B  label loss  -> shared trunk (via the head's READ)  on        CUT          on
        #   C  PPO loss    -> belief head params (via the WRITE)  on         on         CUT
        #   D  PPO loss    -> shared trunk (normal training)      on         on          on
        #
        # `detached` cuts B: the heads READ a stop-grad trunk, so no belief gradient reshapes it.
        # `label_only` cuts C: the heads' outputs are PUBLISHED stop-grad to every forward consumer
        # (reinject, the DamageOperator, the edge cells, the seats), so the belief is trained by its
        # SUPERVISED LABELS ALONE — it stays computed, reinjected and consumed exactly as before, but the
        # return can no longer drag it off-calibration. Its read stays LIVE, so the label loss still
        # teaches the trunk to encode hidden state — which is the point: cutting BOTH B and C would leave
        # a probe on a trunk with no incentive to carry the information, still feeding the policy. That
        # fourth combination is deliberately NOT offered.
        #
        # B is applied per-head via `detach_read`, C via `publish_detach` on the extractor (a stop-grad at
        # the ONE publish boundary per head, so a future consumer is cut by construction rather than by
        # remembering). detach() is value-preserving ⇒ the FORWARD is bit-identical in all three modes.
        if belief_grad_mode not in BELIEF_GRAD_MODES:
            raise ValueError(f"belief_grad_mode must be one of {'|'.join(BELIEF_GRAD_MODES)}, "
                             f"got {belief_grad_mode!r}")
        self.belief_grad_mode = belief_grad_mode
        self._belief_detach = (belief_grad_mode == "detached")
        self._belief_label_only = (belief_grad_mode == "label_only")
        # gen3_extractor_stashes_v1: ALL per-forward side values live in ONE typed container,
        # replaced at forward entry (see ExtractorStashes — the shape/consumer docs live on its
        # fields). Reads go through the read-only `last_*` properties below the ctor; writes go
        # through `self.stash.<field>`.
        self.stash = ExtractorStashes()

        # Phase modules.
        self.embeddings = Embeddings(layout)
        self.unpack = ObsUnpack(layout, attend_unrevealed_opponents=attend_unrevealed_opponents)
        # gen3_pointer_native_v1: the pointer action head is THE action head (no flat action_net in this
        # generation), but the MODULE lives on the POLICY (Gen3DualHeadMaskablePolicy._build — its ctx is
        # latent_pi, which does not exist at extractor time). The extractor's side of the contract is the
        # per-forward stash `stash.pointer_inputs` (request-ordered move tokens + valid mask + our team
        # tokens + the op's per-action cells), set unconditionally in forward_internal.
        self.pokemon_encoder = PokemonEncoder(layout, move_latent=move_latent)
        # gen3_entity_move_seats_v1 (v54, Stage 1): move ENTITY seats in the trunk — E3 (our active's
        # 4 request-ordered move tokens, unconditional) + E4 (the opp active's top-`entity_topk_seats`
        # believed threat moves, opt-in). The pointer head then reads the REFINED E3 seats (post-
        # attention, d_model-wide) instead of the raw 32-dim PokemonEncoder tokens. E4's gates: the
        # candidate weights + latent table must exist PRE-transformer, which is exactly the prefuse
        # stack (validated below after those flags are set).
        self.entity_topk_seats = int(entity_topk_seats)
        self.consequence_topk = int(consequence_topk)   # v59: C1b/C2/C3 k_cand + D4 k_bench
        self.entity_tail_seats = bool(entity_tail_seats)
        self.entity_seats = EntityMoveSeats(self.entity_topk_seats, self.entity_tail_seats)
        # gen3_opp_intent_v1: DECLARED here, CONSTRUCTED at the end of __init__ — the MODULES
        # must be appended last (SB3 restores optimizer state POSITIONALLY), while
        # `forward_internal` reads these attributes unconditionally, so they must always exist.
        self.alpha_head: Optional[AlphaIntentHead] = None
        self.beta_head: Optional[BetaSwitchHead] = None
        # gen3_edge_bias_trunk_v1 (v56, Stage 2): computed physics as per-pair per-head attention
        # BIASES (see EdgeBias). "off" builds no module (no state_dict change beyond the layer swap);
        # the maps are zero-init so an ON run is byte-identical to OFF at init. Requirement
        # validation happens below once the source flags are set.
        self.edge_bias_families = str(edge_bias_families or "off")
        self.edge_bias = (EdgeBias(self.edge_bias_families)
                          if self.edge_bias_families != "off" else None)
        self.team_transformer = TeamTransformer(layout)
        # The injection width IS the op reducer's `extra_dim`, computed by the SAME function the
        # reducer uses. It has to come from the pure helper rather than `self.damage_op`, because
        # the op is built ~250 lines BELOW this point and module construction order is load-bearing
        # (SB3 restores optimizer state positionally — reordering to suit this feature would corrupt
        # every resume). A post-construction assert below ties the two together.
        _vti_dim = value_threat_inject_dim() if bool(value_threat_inject) else 0
        # gen3_pair_value_route_v1 (v95, PV): the SECOND token-content injection on the value pool's
        # local copy — Phase A's unified outcome row, which is what the critic has never had in any
        # per-entity currency. Built here (not through the v89 `_value_pooled_routes` seam) because
        # a post-pool additive route would have to collapse the J axis, and the only equivariant
        # collapse is a sum — see `pair_value_route.py` for the whole argument.
        self.cls_pool = CLSPool(layout, value_threat_inject_dim=_vti_dim,
                                pair_value_row_dim=(PAIR_VALUE_ROUTE_DIM
                                                    if bool(pair_value_route) else 0))
        self.hidden_opp_belief = HiddenOppBeliefPool(opp_belief_cls_k) if opp_belief_cls_k > 0 else None
        # In-place hidden-opponent belief (the live design): distinct learned unknown-mon tokens fill
        # the un-revealed opp slots + a species/moves aux head supervises them. OFF reproduces the
        # baseline arch byte-for-byte (no module, opp slots stay zeros). k>0 the side-pool and this are
        # independent flags; the in-place path supersedes the pool. Hard-requires the unmask flag:
        # masked believed slots would never be refined by the transformer.
        self.opp_belief_slots = opp_belief_slots
        if opp_belief_slots and not attend_unrevealed_opponents:
            raise ValueError(
                "opp_belief_slots=True requires attend_unrevealed_opponents=True — the in-place "
                "belief tokens fill the un-revealed opp slots, which are key-masked out of the "
                "transformer unless the unmask flag is on. Enable --attend-unrevealed-opponents."
            )
        # gen3_species_prior_fusion_v1: fuse the TEAM-COMPOSITION species prior into the belief head, so
        # the species posterior starts at the pool base rate conditioned on the opponent's revealed mons
        # instead of ~uniform over the num axis. Two non-persistent buffers + a zero-init delta head, so
        # the state_dict is UNCHANGED — but it is STRUCTURAL all the same: flipping it mid-run silently
        # re-means every species logit (a resumed head trained as a from-scratch predictor would suddenly
        # be read as a delta on a prior it never saw), which is exactly what the version gate is for.
        # Requires opp_belief_slots — there is no BeliefHead to fuse into otherwise.
        self.species_prior_fusion = species_prior_fusion
        if species_prior_fusion and not opp_belief_slots:
            raise ValueError(
                "species_prior_fusion=True requires opp_belief_slots=True — the prior fuses INTO the "
                "BeliefHead's species head, which only exists under the in-place believed slots. Enable "
                "--opp-belief-aux-coef>0 (which turns on opp_belief_slots), or drop "
                "--species-prior-fusion."
            )
        # gen3_t0_species_prior_v1: the SAME team-composition prior, re-homed to T0 so the T1 physics
        # can read it. `BeliefHead` (T2) computes this belief post-transformer and the DamageOperator
        # (T1) runs before it, so the op could never consume the model's own species belief and fell
        # back to the static `SPECIES_USAGE_PRIOR` frequency table. This module is parameter-free
        # (two non-persistent buffers), so the state_dict is unchanged and no optimizer parameter
        # position moves — but it is STRUCTURAL: with it on, every unrevealed-defender damage number
        # is computed against a different distribution. Independent of `species_prior_fusion`: that
        # flag fuses the prior into the T2 aux READOUT, this one feeds the T1 physics, and either is
        # useful without the other.
        # gen3_intent_move_cell_v1 (G3, design_conditional_execution.md): the POLICY-side alpha
        # consumer — the c2 status-consequence family re-delivered through the pointer MOVE cell,
        # alpha-conditioned. Fail-loud on both operands rather than silently degrading: no alpha =>
        # nothing to weight with; no op => no c2 physics to deliver.
        self.intent_move_cell = None
        if intent_move_cell:
            self.intent_move_cell = IntentMoveCell(INTENT_MOVE_CELL_DIM)
            if not opp_intent:
                raise ValueError(
                    "intent_move_cell=True requires opp_intent=True — the c2 re-delivery is "
                    "WEIGHTED BY alpha, and with no alpha head there is no distribution to "
                    "weight with.")
            if not damage_op:
                raise ValueError(
                    "intent_move_cell=True requires damage_op=True — it re-delivers the "
                    "operator's c2 status-consequence physics, which nothing else produces.")
        # gen3_intent_threshold_v1 (v84, design_conditional_execution.md §3.0 step 3): the
        # α-weighted THRESHOLD operator — five mechanics (Focus Punch / Substitute / Endure /
        # Destiny Bond / Endeavor) through the pointer MOVE cell. The flag used to build a SECOND
        # consumer, the p_KO vf route; the critic-route deletion wave retired it (dV 0.155 / 0.136,
        # below the 0.39 bar and registered no-appeal). **The POLICY-side context stays** — that is
        # the half every audit found live, and `intent_threshold_test.py` pins it.
        self.intent_threshold_move = None
        if intent_threshold:
            self.intent_threshold_move = IntentThresholdMoveCell(INTENT_THRESH_MOVE_DIM)
            if not opp_intent:
                raise ValueError(
                    "intent_threshold=True requires opp_intent=True — every threshold form is "
                    "WEIGHTED BY alpha, and with no alpha head there is no distribution to "
                    "weight with.")
            if not damage_op:
                raise ValueError(
                    "intent_threshold=True requires damage_op=True — the operator's per-candidate "
                    "damage/KO cells are the thresholds' only physics source.")
        # gen3_intent_conditional_v1 (v85, design steps 4+7): Counter / Mirror Coat, flinch's
        # (1−α_SWITCH) term, Explosion's execute/into-switch facts, Pursuit's doubling trigger —
        # per-request-slot cells over tensors the op already stashes, α-contracted at T2.
        # gen3_op_lean_forward_v1: believed_lean prices the lean d3 physics from the spread
        # belief — with no SpreadBelief head there is nothing believed to price with.
        if op_believed_lean and not spread_belief:
            raise ValueError(
                "op_believed_lean=True requires spread_belief=True — the lean physics price the "
                "attacker from the believed spread, and without the head the flag would silently "
                "reproduce the de-timid fiction it exists to remove.")
        if op_believed_lean and not damage_op:
            raise ValueError("op_believed_lean=True requires damage_op=True.")
        self.intent_conditional = None
        if intent_conditional:
            self.intent_conditional = IntentConditionalMoveCell(INTENT_COND_MOVE_DIM)
            if not opp_intent:
                raise ValueError(
                    "intent_conditional=True requires opp_intent=True — every cell is "
                    "WEIGHTED BY alpha, and with no alpha head there is no distribution to "
                    "weight with.")
            if not damage_op:
                raise ValueError(
                    "intent_conditional=True requires damage_op=True — the pair cells and the "
                    "outgoing per-move rolls are the cells' only physics source.")
            if not damage_outgoing:
                raise ValueError(
                    "intent_conditional=True requires damage_outgoing=True — the flinch and "
                    "Pursuit cells read the outgoing per-move rolls, p_outspeed and the "
                    "secondary columns, which only the outgoing block computes.")
            if not damage_matrices_outgoing:
                raise ValueError(
                    "intent_conditional=True requires damage_matrices_outgoing=True — the boom "
                    "trade-value cell reads the per-(our move, their mon) pko, which only the "
                    "outgoing matrix computes (an arrival's KO probability has no other source).")
        # gen3_pair_outcome_v1 (v93, design_opponent_intent.md §5.1/§5.3): the UNIFIED per-pair
        # OUTCOME VECTOR — damage AND status AND neutralization AND tempo in one vector in one
        # currency — reduced by ONE alpha over the opponent's believed-move axis and delivered to
        # the pointer MOVE cell. Phase A: the move-cell half only; the switch cell and the
        # beta-conditioned cells are Phase B and are deliberately not built.
        #
        # ⚠️ It requires `damage_op` and NOT `opp_intent`, and the asymmetry is the point. The
        # physics has exactly one source, so no op is fail-loud; but alpha has a shipped fallback
        # (the R1 `belief_mean` rung, alpha := w/Sum w), so the flag is INDEPENDENTLY ENABLEABLE and
        # a run can test the DELIVERY claim (a per-action absolute in the currency the decision
        # needs) separately from the DISTRIBUTION claim (usage belief beats presence belief) —
        # §7a.2's own suggestion. Degrading silently is the thing to avoid, so the fallback is
        # documented, tested, and NOT the same object as alpha.
        self.pair_outcome_move = None
        if pair_outcome_cell:
            self.pair_outcome_move = PairOutcomeMoveCell(PAIR_OUTCOME_MOVE_DIM)
            if not damage_op:
                raise ValueError(
                    "pair_outcome_cell=True requires damage_op=True — the per-(their move, our mon) "
                    "damage cells and the per-pivot status-landing physics are the outcome vector's "
                    "only source; nothing else computes either.")
        # gen3_pair_outcome_switch_v1 (v94, Phase B): the SAME reduced row, per DEFENDER, at the
        # pointer SWITCH cell — the delivery §2.1 says the decision actually needs ("they will
        # click Will-O-Wisp, so bring the Natural Cure mon" is made at the switch logit, whose cell
        # holds ten damage numbers and no status coordinate in any currency). It is the FIRST
        # module to widen the switch cell.
        #
        # It requires `damage_op` and NOT `pair_outcome_cell`, deliberately: the two deliver the
        # same tensor to two different sinks, and making the switch half depend on the move half
        # would mean the phase could never attribute a result to one of them. α uses the same
        # `pair_alpha` ladder (publication, or the R1 belief_mean fallback), so this flag is
        # independently enableable too.
        self.pair_outcome_switch = None
        if pair_outcome_switch:
            self.pair_outcome_switch = PairOutcomeSwitchCell(PAIR_OUTCOME_SWITCH_DIM)
            if not damage_op:
                raise ValueError(
                    "pair_outcome_switch=True requires damage_op=True — the unified outcome "
                    "vector's only producer is the op's pair grid; nothing else computes it.")
        # gen3_switch_branch_v1 (v94, Phase B): OA2 + the Rapid-Spin spinblock + Protect's
        # α-conditioning, all per-request-slot, all the same contraction over the branch in which
        # they SWITCH (`design_conditional_opponent_cells.md` §2 + the owner's two mechanics).
        #
        # ⚠️ Unlike the pair_outcome pair this one REQUIRES `opp_intent`, and the asymmetry is
        # substantive rather than conservative: every coordinate here is conditioned on α_SWITCH or
        # on β, and NEITHER has a fallback. The R1 `belief_mean` rung is a presence belief over
        # their MOVES; it has no switch class at all, so `α_SWITCH` would be identically 0 and the
        # whole cell would read "they never switch" — a claim, not an absence. β has no
        # prior-shaped substitute either. A flag whose fallback silently asserts something false is
        # worse than a flag that says it needs the head.
        self.switch_branch = None
        if switch_branch_cell:
            self.switch_branch = SwitchBranchMoveCell(SWITCH_BRANCH_MOVE_DIM)
            if not (opp_intent and damage_op and damage_matrices_outgoing):
                raise ValueError(
                    "switch_branch_cell=True requires opp_intent=True (α_SWITCH and β have no "
                    "fallback — the R1 belief_mean rung is a presence belief over their MOVES and "
                    "carries no switch class, so every coordinate would read 'they never switch'), "
                    "damage_op=True, and damage_matrices_outgoing=True (OA2's per-(our move, their "
                    "mon) grid is what makes β actionable; there is no other source for 'what my "
                    "move does to the arrival').")
        # gen3_conditional_threat_v1 (v95, Phase C): OA1 — the defensive-pivot coordinates the
        # α-reduced outcome row structurally cannot carry (the accuracy-folded P(this mon dies),
        # the bulk-INDEPENDENT expected type multiplier, and the two §0.2(3) margins), on the
        # pointer SWITCH cell.
        #
        # Requires `damage_op` and `damage_matrices_incoming`, and NOT `pair_outcome_switch`. The
        # matrix requirement is real rather than defensive: it is the ONLY producer of the
        # per-(defender, seat) type multiplier AND of the top-K selection α's seats align to. The
        # independence from Phase B is deliberate for the same reason Phase B is independent of
        # Phase A — the two widen one cell with different quantities, and coupling them would make
        # a measured result unattributable to either. α uses the same `pair_alpha` ladder, so the
        # R1 fallback keeps the flag independently enableable; that fallback is MEANINGFUL here
        # (every coordinate is a "what lands on me if they attack" contraction, so the missing
        # SWITCH mass correctly shrinks it toward zero) rather than the v94 case where it would
        # have asserted something false.
        self.conditional_threat = None
        if conditional_threat_cell:
            self.conditional_threat = ConditionalThreatCell(CONDITIONAL_THREAT_SWITCH_DIM)
            if not (damage_op and damage_matrices_incoming):
                raise ValueError(
                    "conditional_threat_cell=True requires damage_op=True and "
                    "damage_matrices_incoming=True — the incoming matrix is the only producer of "
                    "the per-(our defender, their seat) type multiplier and of the top-K selection "
                    "α's seats align to; nothing else computes either.")
        # gen3_pair_value_route_v1 (v95, Phase C): PV — the same unified outcome row as TOKEN
        # CONTENT on the CRITIC's copy of our tokens (design_opponent_intent.md §7a(2)). The module
        # itself lives on `cls_pool` (so the augmented tensor stays a local and vf-only is
        # structural); this flag records the decision and enforces the dependency.
        #
        # ⚠️ C4 RE-ENTRY CONDITION: any α/β-critic route may be BUILT opt-in but its ENABLING owes
        # the C4-style offline gate first (ledger C6 — the delivery line is EXHAUSTED).
        self.pair_value_route = bool(pair_value_route)
        if self.pair_value_route and not damage_op:
            raise ValueError(
                "pair_value_route=True requires damage_op=True — the injected row IS the op's "
                "unified `pair_in` outcome vector, and nothing else computes it.")
        # (v87's two direct critic routes — `--value-clock` and `--value-intent` — are DELETED in
        # the critic-route deletion wave. `value_intent` read dV 0.156 against a 0.39 bar; its
        # RE-ENTRY CONDITION SURVIVES THE DELETION: any future α/β-to-critic proposal passes the
        # C4-style offline gate FIRST. It is cheap to rebuild through the `_value_pooled_routes`
        # seam — it was deleted because the measurement says the critic does not use it, not
        # because the idea is unsound. `value_clock` read 0.2169 at 2× sample, and C1 had already
        # measured the clock CONTENT reaching the critic through the trunk at ~83% of an HP
        # control's responsiveness — substitution, which is what makes the deletion low-risk.)
        if opp_intent_grad_mode not in ("detached", "shaping"):
            raise ValueError(
                f"opp_intent_grad_mode must be 'detached' or 'shaping', got "
                f"{opp_intent_grad_mode!r}")
        self.opp_intent_grad_mode = opp_intent_grad_mode
        self.t0_species_prior = (T0SpeciesPrior(layout['max_species'])
                                 if t0_species_prior else None)
        self.belief_slots = BeliefSlots() if opp_belief_slots else None
        self.belief_head = (
            BeliefHead(layout['max_species'], layout['max_moves'],
                       species_prior_fusion=species_prior_fusion) if opp_belief_slots else None
        )
        # (`stash.belief_logits` — the per-minibatch aux dict, carries grad — and
        # `stash.opp_believed_mask` are written each forward; see ExtractorStashes.)
        # Move belief (flag-guarded): predict + REINJECT the opp moveset into the slot tokens so the
        # believed moves flow into the policy/value readout. mode ∈ {off, revealed, unrevealed, both}
        # selects which opp slots are enriched + scored. OFF reproduces the baseline arch byte-for-byte.
        if move_belief_mode not in ("off", "revealed", "unrevealed", "both"):
            raise ValueError(f"move_belief_mode must be off|revealed|unrevealed|both, got {move_belief_mode!r}")
        self.move_belief_mode = move_belief_mode
        if move_belief_mode != "off" and not attend_unrevealed_opponents:
            raise ValueError(
                "move_belief_mode != off requires attend_unrevealed_opponents=True — the move belief "
                "reads/enriches the opp slots (incl. hidden ones), which are key-masked unless the "
                "unmask flag is on. Enable --attend-unrevealed-opponents."
            )
        # Prior fusion (the unified two-part belief): fold the Smogon move-frequency prior into the
        # move-belief head as a log-odds residual + pin revealed moves certain. Requires the head to exist
        # (move_belief_mode != off). OFF reproduces the from-scratch head byte-for-byte (no buffer, no
        # forward change) — a forward-behavior toggle (no weight-shape change, version-checked).
        self.move_prior_fusion = move_prior_fusion
        if move_prior_fusion and move_belief_mode == "off":
            raise ValueError(
                "move_prior_fusion=True requires move_belief_mode != off — the prior fuses INTO the "
                "move-belief head's logits; with no head there is nothing to fuse. Set --move-belief-mode "
                "revealed (or both/unrevealed), or disable --move-prior-fusion."
            )
        # gen3_tiered_pipeline_v1: the move belief is reinjected into the opp role tokens BEFORE the
        # TeamTransformer — UNCONDITIONALLY. It is a T0 RESOLVE step: the believed moves co-refine with
        # the species/team belief through the attention layers instead of being grafted on afterwards,
        # and every T1 consumer (the DamageOperator, the E4 seats, the edge cells) reads one posterior
        # computed once. The old POST-transformer call site and its `--move-belief-prefuse` selector are
        # DELETED; a config that recorded `move_belief_prefuse=False` is REFUSED by the v71 migration
        # rather than silently re-ordered.
        self.move_belief = (
            MoveBelief(layout['max_moves'], layout['move_embedding_dim'],
                       prior_fusion=move_prior_fusion, n_species=layout['max_species'],
                       move_candidate_floor=move_candidate_floor)
            if move_belief_mode != "off" else None
        )
        # gen3_unified_spread_belief_v1: the THIRD belief leg — predicts the opp's hidden SPREAD (5 derived
        # stats) per slot, reinjected into the opp token, consumed by the DamageOperator (replacing its
        # hand-coded opp-spread constants). STRUCTURAL toggle (widens nothing in the projection — it enriches
        # the opp token like MoveBelief). Requires move_belief_mode != off only if damage_op is on (the op is
        # the consumer); built whenever the flag is set. Stash for the supervision loss + the op.
        # gen3_nature_ev_belief_v1: --spread-belief-nature swaps the additive point-estimate head for the
        # NATURE/EV generative head (prior-fusion → compute the derived stat) to fix the largest-EV over-estimate.
        # Requires --spread-belief (the head IS the SpreadBelief module). STRUCTURAL (different SpreadBelief params).
        if spread_belief_nature and not spread_belief:
            raise ValueError("spread_belief_nature requires spread_belief=True (it parameterises the "
                             "SpreadBelief head). Enable --spread-belief, or drop --spread-belief-nature.")
        # gen3_nature_ev_belief_v1: the op marginalises P(KO) over the head's nature distribution → requires it.
        self.spread_belief_enabled = spread_belief
        self.spread_belief_nature = spread_belief_nature
        self.spread_belief = SpreadBelief(layout['max_species'], nature=spread_belief_nature) if spread_belief else None
        # gen3_typed_hp_belief_v1 / gen3_hp_belief_ablation_v1: the opponent's Hidden Power is ALWAYS
        # reasoned about as the 16 DISCRETE TYPED moves — the old typeless-BP-0 candidate is gone in both
        # arms, because it was a correctness bug (a 0-damage "immune" reading of a revealed HP), not an
        # ablation. What `hp_belief_mode` varies is HOW the 16 typed channels are produced:
        #
        #   'composed' (DEFAULT) — build `HPTypeBelief` and factor the belief as
        #                          `P(HP_t) = presence · P(type=t)`. Buys the structural constraint
        #                          (a revealed HP must exist as SOME type), the moveset-exhaustion
        #                          rule-out, and the effectiveness narrowing.
        #   'flat'     (ABLATION) — no head. The multi-label move head predicts the 16 typed channels
        #                          INDEPENDENTLY, off their own real per-typed Smogon usage priors, and
        #                          Hidden Power is treated exactly like any other move. No factorisation,
        #                          no constraint, no narrowing.
        #
        # The head is prior-fused + zero-init, so under 'composed' its cold-start posterior IS the Smogon
        # HP-type prior; `--hp-type-belief-coef` controls only whether the privileged CE supervises it on
        # top of the damage + move-BCE gradients it already gets. Neither arm requires `damage_op`: the
        # composition lives in the BELIEF, so the typed posterior reaches the token reinjection, the BCE,
        # the latent grading and the prober even on a run with no damage operator at all.
        self.hp_belief_mode = str(hp_belief_mode)
        if self.hp_belief_mode not in ("composed", "flat"):
            raise ValueError(
                f"hp_belief_mode must be one of composed|flat, got {hp_belief_mode!r}")
        self.hp_type_belief_head = (
            HPTypeBelief(layout['max_species'], layout['type_embedding_dim'])
            if (self.move_belief is not None and self.hp_belief_mode == "composed") else None)
        # gen3_item_belief_v1: the hidden-ITEM posterior (Smogon prior ⊕ zero-init delta), the
        # BeliefBank's seventh head. The op consumes P(Choice Band) at the active slot in place
        # of its static usage scalar; OFF builds nothing (byte-identical).
        self.item_belief_head = (
            ItemBelief(layout['max_species'], layout['max_items']) if item_belief else None)
        # Differentiable damage operator (flag-guarded): consumes the move belief's PREDICTED moves for
        # the opp active and computes the believed-move incoming damage to each of our mons, fed to BOTH
        # heads. OFF reproduces the baseline arch byte-for-byte (no module, projection widths unchanged).
        # Requires a move-belief mode that SCORES the opp active (a revealed mon): revealed|both. Under
        # off/unrevealed the active-slot logits are unsupervised, so the belief gradient story breaks.
        self.damage_op_enabled = damage_op
        if damage_op and move_belief_mode not in ("revealed", "both"):
            raise ValueError(
                "damage_op=True requires move_belief_mode in {revealed, both} — the operator reads the "
                "opp ACTIVE slot's predicted move logits, which are only supervised/reinjected for a "
                "revealed mon. Set --move-belief-mode revealed (or both), or disable --damage-op."
            )
        # OUTGOING direction (our active → opp active, per-move action-aligned): requires the op itself.
        self.damage_outgoing = damage_outgoing
        if damage_outgoing and not damage_op:
            raise ValueError(
                "damage_outgoing=True requires damage_op=True — the outgoing per-move block is emitted by "
                "the DamageOperator. Enable --damage-op (--unified-damage both), or drop the outgoing flag."
            )
        # The DISCRETE incoming move-space K (K = damage_topk_k; 0 = off) — how many of the opp active's
        # most-believed candidate moves the INCOMING MATRIX surfaces individually. Requires the op (it
        # extends it) AND --move-latent (the op gathers each move's LATENT from the MoveLatentEncoder for
        # identity, and the candidate latent table is built only when move_latent).
        self.damage_topk_k = int(damage_topk_k)
        if self.damage_topk_k > 0 and not damage_op:
            raise ValueError(
                "damage_topk_k>0 requires damage_op=True — the discrete incoming block extends the "
                "DamageOperator. Enable --damage-op (--unified-damage), or drop --damage-topk."
            )
        if self.damage_topk_k > 0 and not move_latent:
            raise ValueError(
                "damage_topk_k>0 requires move_latent=True — the block gathers each move's identity "
                "LATENT from the MoveLatentEncoder. Enable --move-latent (--unified-moves), or drop --damage-topk."
            )
        # gen3_op_block_trim_v1: `damage_topk_k` no longer has a block of its own — the v30 LEAN top-K was
        # deleted as a strict subset of the v35 incoming matrix (0 calls/forward in every config that ran
        # the matrix). K is now purely the matrix's width, so K>0 without the matrix would emit NOTHING.
        if self.damage_topk_k > 0 and not damage_matrices_incoming:
            raise ValueError(
                f"damage_topk_k={self.damage_topk_k} requires damage_matrices_incoming=True "
                "(gen3_op_block_trim_v1) — the lean top-K block it used to select was deleted, and K is now "
                "the INCOMING MATRIX's width. Pass --damage-matrices incoming (or both), or --damage-topk 0."
            )
        # gen3_per_move_matrices_v1: the OUTGOING per-move DAMAGE MATRIX (our 4 moves × opp active+revealed
        # bench). Requires damage_op (the op physics). Off byte-identical.
        self.damage_matrices_outgoing = bool(damage_matrices_outgoing)
        if self.damage_matrices_outgoing and not damage_op:
            raise ValueError(
                "damage_matrices_outgoing=True requires damage_op=True — the outgoing per-move damage matrix "
                "is emitted by the DamageOperator. Enable --damage-op (--unified-damage), or drop --damage-matrices."
            )
        # gen3_per_move_matrices_v1: the INCOMING per-move DAMAGE MATRIX. Requires damage_op + move_latent
        # (the latent gather); its K is `damage_topk_k` (default 5).
        self.damage_matrices_incoming = bool(damage_matrices_incoming)
        if self.damage_matrices_incoming and not damage_op:
            raise ValueError(
                "damage_matrices_incoming=True requires damage_op=True — the incoming per-move damage matrix "
                "is emitted by the DamageOperator. Enable --damage-op (--unified-damage), or drop --damage-matrices."
            )
        if self.damage_matrices_incoming and not move_latent:
            raise ValueError(
                "damage_matrices_incoming=True requires move_latent=True — the matrix header gathers each "
                "move's identity LATENT. Enable --move-latent (--unified-moves), or drop the incoming matrix."
            )
        # gen3_topk_candidates_v1: the incoming candidate-sweep cap (0 = full ~400-wide sweep).
        self.damage_candidate_k = int(damage_candidate_k)
        if self.damage_candidate_k and not damage_op:
            raise ValueError("damage_candidate_k>0 requires damage_op=True — it caps the DamageOperator's "
                             "incoming candidate sweep, which only exists when the op is built.")
        # gen3_value_threat_inject_v1: the critic-side magnitude route needs the op's Contract-W
        # reduced rows, and R0 `hard_max` (production) builds NO reducer and stashes nothing — so
        # the flag FORCES the R1 rung on. It is derived, never a second user knob: this arm tests
        # DELIVERY, and a variable rung would confound that with the DISTRIBUTION question.
        self.value_threat_inject = bool(value_threat_inject)
        if self.value_threat_inject and not damage_op:
            raise ValueError(
                "value_threat_inject=True requires damage_op=True — the injected row IS the op's "
                "reduced incoming threat, so with no op there is nothing to inject and the flag "
                "would be a silent no-op.")
        _reduce_how = (VALUE_THREAT_INJECT_REDUCE_HOW if self.value_threat_inject
                       else "hard_max")
        # The incoming matrix's K is damage_topk_k (the one "how many opp moves" knob).
        # `damage_op` is None when the flag is off, and EVERY read of it below sits under a guard
        # on a different, correlated flag (`edge_bias is not None`, `damage_block is not None`,
        # `intent_* is not None`) whose implication this constructor enforces with a raise. That
        # invariant spans two objects, so no narrowing expresses it — hence the
        # `type: ignore[union-attr]` on each read. Same story for the `Optional` OpTensors views.
        self.damage_op = (DamageOperator(layout, outgoing=damage_outgoing, topk_k=self.damage_topk_k,
                                         matrices_outgoing=self.damage_matrices_outgoing,
                                         matrices_incoming=self.damage_matrices_incoming,
                                         prob_outspeed=threat_prob_outspeed,
                                         candidate_k=self.damage_candidate_k,
                                         reduce_how=_reduce_how,
                                         drop_renders=op_drop_renders,
                                         believed_lean=op_believed_lean)
                          if damage_op else None)
        # Tie the two ends together NOW rather than discovering a width mismatch in a forward pass:
        # `cls_pool`'s projection was sized from the pure helper hundreds of lines above, before the
        # Ask the op to KEEP its un-reduced pair cells. Set here, after the op exists, because
        # the flag lives on the op but is owned by consumers built before it. Without this a
        # consumer sees `last_pair_cells is None` and — by design — raises rather than
        # contributing zeros, since a silent no-op reads exactly like a null.
        if (self.intent_threshold_move is not None
                or self.intent_conditional is not None or self.pair_outcome_move is not None
                or self.pair_outcome_switch is not None or self.conditional_threat is not None
                or self.pair_value_route):
            self.damage_op.stash_pair_cells = True  # type: ignore[union-attr]
        # gen3_pair_outcome_v1: and the eight extra coordinates on top of them. Set together with
        # `stash_pair_cells` above, never alone — the damage cells ARE the vector's first six
        # coordinates, so a lone `stash_pair_outcome` would build a narrower vector than
        # `PAIR_OUTCOME_COORDS` declares.
        if (self.pair_outcome_move is not None or self.pair_outcome_switch is not None
                or self.conditional_threat is not None or self.pair_value_route):
            self.damage_op.stash_pair_outcome = True  # type: ignore[union-attr]
        # gen3_switch_branch_v1: the per-opp-slot GHOST marginal the spinblock contracts β against.
        if self.switch_branch is not None:
            self.damage_op.stash_opp_ghost = True  # type: ignore[union-attr]
        # gen3_conditional_threat_v1: the per-(defender, seat) TYPE MULTIPLIER. Not a coordinate of
        # `pair_in` (whose width is a contract three consumers read), so it gets its own seam —
        # a pure `.detach()` of a tensor the incoming matrix already built, so zero extra math.
        if self.conditional_threat is not None:
            self.damage_op.stash_pair_type_mult = True  # type: ignore[union-attr]
        # op existed. If those ever disagree the flag is silently mis-wired, so assert the identity.
        if self.value_threat_inject:
            _built = self.damage_op.pair_reducer.extra_dim  # type: ignore[union-attr]
            if _built != self.cls_pool.value_threat_proj.extra_dim:  # type: ignore[union-attr]
                raise AssertionError(
                    f"value_threat_inject width mismatch: the op's reducer emits {_built} but the "  # type: ignore[union-attr]
                    f"projection was built for {self.cls_pool.value_threat_proj.extra_dim} — "
                    "`value_threat_inject_dim()` has drifted from `PairReducer.extra_dim`.")
        self.threat_prob_outspeed = bool(threat_prob_outspeed)
        # gen3_tiered_pipeline_v1 (was gen3_damage_op_prefuse_v1, v50): ONE damage computation per
        # forward, PRE-attention, and now the ONLY placement. The spread/HP-type beliefs + the FULL op
        # run on the PRE-transformer role tokens (T0 RESOLVE → T1 REASON), the per-OUR-mon incoming rows
        # are injected onto our tokens via the zero-init `prefuse_proj` (identity-at-init), and the same
        # block feeds every downstream consumer. The POST-transformer call site and its
        # `--damage-op-prefuse` selector are DELETED.
        #
        # Its original justification was CPU cost: at B=1 on CPU (the PFSP frozen-opponent regime, which
        # sits on the rollout critical path) the op dominated a dispatch-bound ~6.45 ms forward, against
        # 0.27 ms for the attention layers themselves. The architectural story ("attention now reasons
        # over full-fidelity physics") is SECONDARY and, on this codebase's evidence, unlikely to pay —
        # physics-into-the-trunk measured NULL 3-for-3 (ledger K9/K10) and the lean kernel was already a
        # good proxy for the full op (K10a).
        #
        # Zero-init → the injected residual is EXACTLY 0 at init, so the trunk starts from the same one
        # the baseline transformer sees (identity-at-init; the gradient still flows because the damage
        # feats are non-zero). It carries the FULL per-mon incoming row. NOTE there is no OUTGOING or
        # STATUS trunk residual: those measured null (K10) and would need their own projections.
        self.prefuse_proj = (torch.nn.Linear(_DMG_PER_MON, D_MODEL) if damage_op else None)
        if self.prefuse_proj is not None:
            torch.nn.init.zeros_(self.prefuse_proj.weight)
            torch.nn.init.zeros_(self.prefuse_proj.bias)
        # gen3_entity_move_seats_v1: E4 threat seats need the belief-weighted candidate definition
        # (`DamageOperator.refine_candidates`) + the move latent table, both PRE-transformer — which the
        # tiered order now guarantees whenever the op exists. E3 is unconditional and needs none of this.
        if self.entity_topk_seats > 0 and not (damage_op and move_latent):
            raise ValueError(
                f"entity_topk_seats={self.entity_topk_seats} requires damage_op=True AND "
                "move_latent=True — the E4 threat seats gather the op's pre-transformer candidate "
                "weights and the move latent table. Enable --damage-op + --move-latent "
                "(--unified-moves), or set --entity-topk-seats 0 (E3-only)."
            )
        # gen3_entity_tail_seats_v1 (E5): the tail summarizes the belief the OTHER consumers
        # truncate — it needs the pre-transformer posterior + the op's move buffers, i.e. the same
        # T0/T1 stack as E4.
        if self.entity_tail_seats and not (damage_op and self.entity_topk_seats > 0):
            raise ValueError(
                "entity_tail_seats requires damage_op AND entity_topk_seats > 0 — the tail "
                "is defined relative to the E4 seats' top-K truncation. Enable those, or drop "
                "--entity-tail-seats."
            )
        # gen3_edge_bias_trunk_v1: per-family source requirements. d1 reads the op's outgoing-matrix
        # kernel (our active's moves × opp mons); d3 reads the pre-collapse incoming kernel AT the E4
        # seats' candidate selection, so its rows and the seats must exist together.
        if self.edge_bias is not None:
            fams = self.edge_bias.families
            if ("d1" in fams or "s1" in fams or "c1" in fams or "c2" in fams) \
                    and not (damage_op and damage_outgoing):
                raise ValueError(
                    "edge_bias_families d1/s1/c1/c2 price our active's moves vs the opp team via the op's "
                    "outgoing kernels — require --damage-op AND --damage-outgoing "
                    "(--unified-damage both / --unified-moves both)."
                )
            if "c4" in fams and not damage_op:
                raise ValueError(
                    "edge_bias_families c4 composes the G-family ledger — requires --damage-op."
                )
            if "c3" in fams and not damage_op:
                raise ValueError(
                    "edge_bias_families c3 re-evaluates the believed-hit KO ramp at post-heal "
                    "HP — requires --damage-op (the belief + the op's tables)."
                )
            if "c5" in fams and not damage_op:
                raise ValueError(
                    "edge_bias_families c5 re-runs the switch-in offense kernel under inherited "
                    "stages — requires --damage-op."
                )
            if "g" in fams and not damage_op:
                raise ValueError(
                    "edge_bias_families g reads the op's type tables for the weather-immunity "
                    "legs — requires --damage-op."
                )
            if "x" in fams and not damage_op:
                raise ValueError(
                    "edge_bias_families x reads the pre-transformer composed posterior (Pursuit "
                    "belief) + the op's tables — requires --damage-op."
                )
            if "t" in fams and not damage_op:
                raise ValueError(
                    "edge_bias_families t prices mon↔mon trapping from the op's trap tables — "
                    "requires --damage-op."
                )
            if "v" in fams and not damage_op:
                raise ValueError(
                    "edge_bias_families v prices mon↔mon P(outspeed) from the op's speed machinery — "
                    "requires --damage-op."
                )
            if "d2" in fams and not damage_op:
                raise ValueError(
                    "edge_bias_families d2 prices every our-mon's offense vs the opp active via the "
                    "op's v39 switch-in kernel — requires --damage-op."
                )
            if "d4" in fams and not damage_op:
                raise ValueError(
                    "edge_bias_families d4 prices the opp BENCH's believed threats via the op's "
                    "candidate machinery — requires --damage-op (and a move belief)."
                )
            if ("d3" in fams or "s3" in fams) and self.entity_topk_seats <= 0:
                raise ValueError(
                    "edge_bias_families d3/s3 bias rows are the E4 threat seats — require "
                    "--entity-topk-seats > 0 (which itself requires the prefuse stack)."
                )
            if "r" in fams and not history_events:
                raise ValueError(
                    "edge_bias_families r (Tier H-C reference edges) bias rows are the "
                    "H-B event seats — requires --history-events."
                )
        # Stored on the root so arch_toggles_from_model can thread it to the eval/self-play workers
        # (the move-prior gate is a version-checked forward-behavior toggle).
        self.move_candidate_floor = move_candidate_floor
        self.assembler = ProjectionAssembler(layout)

        # Auxiliary WIN-PROBABILITY head (tri-state `win_prob_mode`): a calibrated P(win|state) readout
        # off `value_pooled`. 'none' = no module (baseline byte-for-byte, NOT in pi/vf so projection dims
        # are unchanged either way). 'read_only' = the head trains its OWN params on a STOP-GRAD
        # value_pooled (a pure, risk-free diagnostic — zero gradient to the trunk). 'shaping' = gradient
        # flows into the shared trunk (the win objective shapes the representation). It is a SIDE readout
        # (stashed for the aux loss + prober, never concatenated into pi/vf — leak-safe). The
        # state_dict-changing toggle is 'none'↔head; the mode itself is resume-immutable (version-checked).
        if win_prob_mode not in ("none", "read_only", "shaping"):
            raise ValueError(f"win_prob_mode must be none|read_only|shaping, got {win_prob_mode!r}")
        self.win_prob_mode = win_prob_mode
        self.win_head = WinProbHead() if win_prob_mode != "none" else None
        # (`stash.win_prob_logits` [B,1] — the aux BCE + prober readout — and `stash.value_pooled`
        # — the FitNets HINT layer `instrumented_ppo._value_feat_distill` reads — are written each
        # forward; NEVER fed into pi/vf, so no label can leak.)

        # Distributional VALUE head (tri-state `value_dist_mode`, v29): an interpretability readout off
        # `value_pooled` emitting `value_dist_bins` logits over the support [vmin, vmax]. 'none' = no
        # module (baseline byte-for-byte, NOT in pi/vf so projection dims are unchanged). 'read_only' =
        # trains its OWN params on a STOP-GRAD value_pooled (a risk-free diagnostic — zero trunk
        # gradient). 'shaping' = its gradient also shapes the shared trunk. SIDE readout (stashed for the
        # aux loss + prober, never in pi/vf — and the value target can't leak). The state_dict-changing
        # toggles are 'none'↔head (the head params) AND the atom count `bins` (the head's output width);
        # both + the mode are resume-immutable (version-checked). See ValueDistHead.
        if value_dist_mode not in ("none", "read_only", "shaping"):
            raise ValueError(f"value_dist_mode must be none|read_only|shaping, got {value_dist_mode!r}")
        if value_dist_mode != "none" and value_dist_bins <= 0:
            raise ValueError(
                f"value_dist_mode={value_dist_mode!r} requires value_dist_bins > 0 (the atom count), "
                f"got {value_dist_bins}"
            )
        if value_dist_mode == "none" and value_dist_bins != 0:
            raise ValueError(
                f"value_dist_bins must be 0 when value_dist_mode == 'none', got {value_dist_bins}"
            )
        self.value_dist_mode = value_dist_mode
        self.value_dist_bins = value_dist_bins
        self.value_dist_vmin = value_dist_vmin
        self.value_dist_vmax = value_dist_vmax
        self.value_dist_head = (
            ValueDistHead(value_dist_bins, value_dist_vmin, value_dist_vmax)
            if value_dist_mode != "none" else None
        )
        # (`stash.value_dist_logits` [B,bins] — the dist-critic/aux/prober readout — is written
        # each forward; NEVER fed into pi/vf.)

        # gen3_unified_value_readout_v1 (v80): the Stage-3 critic entity pool — see the class
        # docstring. With the flag OFF nothing is constructed and every existing parameter keeps
        # its optimizer position; ON is a version-gated arch change (fresh run), where the shift
        # is legitimate. Works with or without the op (the row set shrinks to the 12 team tokens).
        if value_entity_pool_full and not value_entity_pool:
            raise ValueError(
                "value_entity_pool_full=True requires value_entity_pool=True — `full` extends "
                "the pool's row set; there is no pool to extend without the base flag.")
        self.value_entity_pool = (
            UnifiedValueReadout(self.damage_op.per_mon if self.damage_op is not None else 0,
                                full=value_entity_pool_full)
            if value_entity_pool else None)

        # gen3_event_window_v1 (Tier H-B): the event-seat consumer of the obs event window —
        # opt-in (OFF builds nothing, byte-identical); the obs block itself is unconditional.
        self.history_events = EventSeats(layout) if history_events else None
        if history_events and 'event_window_n' not in layout:
            raise ValueError(
                "history_events=True but the obs layout carries no event_window block — "
                "the seats would attend over nothing.")

        self.role_token_size = ROLE_TOKEN_SIZE

        # gen3_belief_grad_mode_v1: stamp the per-head trunk-read detach flag now that every belief head
        # exists. 'shaping' ⇒ all False ⇒ byte-identical. BeliefSlots has no predictive read (it only
        # swaps in learned tokens pre-transformer), so it is intentionally NOT in this list.
        self._stamp_belief_grad_flags()

        # gen3_static_widths_v1: the projection-input widths are STATIC ARITHMETIC — see
        # `compute_projection_widths`. The old mechanism (a construction-time dummy
        # `forward_internal` under `_intent_reduce_discovering`, with zero-fill/skip branches
        # threaded through the runtime forward) is DELETED: since v89 every value route injects
        # additively into `value_pooled`, so no width is emergent, and the discovery pass was
        # the parent of a shipped bug class (ede5a88 — an early return in a discovery branch
        # hid every width appended below it and built the critic 128 dims short). The sweep
        # test `projection_width_test.py` preserves the old mechanism AS THE VERIFIER: it runs
        # a real forward per flag combo and asserts the measured widths equal this arithmetic.
        self.projection_input_dim, self.value_projection_input_dim = compute_projection_widths(
            layout, opp_belief_cls_k=opp_belief_cls_k)

        # Two projection heads, both → PROJECTION_DIM. Pre-projection LayerNorm equalises
        # per-block scales. The value head reads the value-dedicated CLS pool (Option C):
        # the transformer body is shared, but policy and value are summarised + projected
        # through independent paths so the critic isn't fighting the actor over the readout.
        self.projection_dim = PROJECTION_DIM
        self.pre_proj_norm = torch.nn.LayerNorm(self.projection_input_dim)
        self.projection = torch.nn.Linear(self.projection_input_dim, self.projection_dim)
        self.value_pre_norm = torch.nn.LayerNorm(self.value_projection_input_dim)
        self.value_projection = torch.nn.Linear(self.value_projection_input_dim, self.projection_dim)
        self.activation = torch.nn.ReLU()
        # Both heads emit PROJECTION_DIM; SB3 sizes the shared mlp_extractor from this.
        self.features_dim = self.projection_dim

        if log_level >= LogLevel.PERIODIC and mappings:
            from agents.model.observation_debugger import ObservationDebugger
            self._debugger: Optional[ObservationDebugger] = ObservationDebugger(mappings)
        else:
            self._debugger = None

        # gen3_opp_intent_v1: the ALPHA/BETA intent heads. Built LAST (before the identity snapshot)
        # so appending their params cannot shift any existing optimizer position — SB3 restores
        # optimizer state POSITIONALLY (ledger: the ai_v6_13 "128 vs 5" crash), so a new module must
        # always be appended, never inserted.
        self.opp_intent = bool(opp_intent)
        if self.opp_intent and self.entity_topk_seats <= 0:
            raise ValueError(
                "opp_intent=True requires entity_topk_seats>0 — alpha is a POINTER over the E4 "
                "believed-threat move seats, so with no seats there is nothing for it to point at "
                "and the head would silently score an empty set.")
        # Context = both team pools (256). They are read AFTER the op's prefuse injection and the
        # edge families, so our OUTGOING physics is already in `our_team_pooled` — design §3.1's
        # requirement that alpha see our own threat (both sides anticipate; the fixed point is found
        # by self-play training, never solved at inference).
        _intent_ctx = 2 * D_MODEL
        if self.opp_intent:
            self.alpha_head = AlphaIntentHead(D_MODEL, _intent_ctx)
            self.beta_head = BetaSwitchHead(D_MODEL, _intent_ctx)
        # (alpha/beta stashes: read ONLY by the aux loss + the prober; never fed forward — see
        # ExtractorStashes.)

        # gen3_cf_evidential_head_v1 (v98): the EVIDENTIAL Beta readout over P(win|state), for the
        # counterfactual label factory's rung R1 (designs/ai_v10/design_counterfactual_value_
        # grounding.md). STRUCTURAL — the module's params are in the state_dict or they are not —
        # but nothing else: it is NOT called from this forward at all. The training-side loss
        # (`instrumented_ppo._cf_evidential_term`) applies it to the STASHED `value_pooled`,
        # always detached, so pi/vf are bit-identical whether or not it is built.
        # Built LAST, after the intent heads and before the identity snapshot, for the same reason
        # they are: SB3 restores optimizer state POSITIONALLY (the ai_v6_13 "128 vs 5" crash), and
        # building it here also leaves every earlier module's init RNG draw untouched — which is
        # what makes ON-at-coefficient-0 bit-identical to OFF and not merely equal in shape.
        self.cf_evidential = bool(cf_evidential)
        self.cf_evid_head = CfEvidentialHead() if self.cf_evidential else None

        # gen3_cf_twin_heads_v1 (v99): the TWIN WIN-PROB HEADS and the SHADOW CRITIC — the
        # owner-authorized amendment to the R1 pre-registration (ledger 2026-08-22 evening, "Three
        # owner sign-offs", item 3). Both are STRUCTURAL in exactly the `cf_evidential` sense: their
        # params are the state_dict delta and nothing else, because neither is called from this
        # forward. The training-side terms apply them to the STASHED `value_pooled`.
        #
        # WHY TWINS. R1's primary comparison was two RUNS (arm vs control). Two runs differ in every
        # random draw they ever make, and the meter's own measured floor is ~40% of its variance —
        # so a run-to-run difference has to clear noise the design cannot control. Three heads on
        # ONE trunk delete that variance by construction: identical trunk, identical states, and
        # the ONLY difference is which label stream trains each head.
        #   head A = `win_head` above (the CONTROL — the existing on-policy single-outcome BCE, and
        #            it is not touched by any of this: A is not new)
        #   head B = A's loss PLUS the cf-labelled states with SINGLE-OUTCOME labels  → COVERAGE
        #   head C = A's loss PLUS the same states with TIGHT-MC labels               → +PRECISION
        # B−A isolates prioritization/coverage; C−B isolates pure variance reduction. Both twins are
        # `WinProbHead` — the SAME module class and the same capacity as A — because a difference of
        # architectures would be a second explanation for a difference of scores.
        #
        # HEAD-ONLY ALWAYS in v1: both twins read a detached `value_pooled` in every term they take,
        # so this measures the LABEL effect on a trunk that is frozen with respect to them. Trunk
        # exposure and policy transfer stay CROSS-RUN questions (runbook §0a, unamended).
        self.cf_twin_heads = bool(cf_twin_heads)
        if self.cf_twin_heads and self.win_head is None:
            # DECLARED in flag_registry (`requires=("win_prob_mode",)`) and enforced here, which is
            # the registry's contract: a dependency that only the CLI knows is invisible to
            # `checkargs`, so an operator validating a recorded launcher_command would get exit 0 on
            # a command the child then refuses. Head A IS `win_head`; without it the twins have no
            # control objective to mirror and the factorial has no control arm at all — the arm
            # would run and its primary comparison would silently not exist.
            raise ValueError(
                "cf_twin_heads requires win_prob_mode != 'none': heads B and C mirror head A's "
                "on-policy win-prob BCE, and head A is `win_head`, which win_prob_mode='none' does "
                "not build. Set --win-prob-mode read_only|shaping, or drop --cf-twin-heads.")
        self.cf_twin_head_b = WinProbHead() if self.cf_twin_heads else None
        self.cf_twin_head_c = WinProbHead() if self.cf_twin_heads else None
        # The SHADOW CRITIC: a passive value twin on tight-MC `mc_return` labels. Never computes an
        # advantage, never enters GAE, never feeds the forward — the staged promotion path for
        # critic surgery (which owes C4), not the surgery. See `ShadowValueHead`.
        self.cf_shadow_critic = bool(cf_shadow_critic)
        self.cf_shadow_head = ShadowValueHead() if self.cf_shadow_critic else None

        # gen3_identity_init_guard_v1 — SNAPSHOT the identity-at-init contract. See
        # `restore_identity_init` for why this exists; it must be the LAST thing __init__ does, so
        # every module is built and every deliberate zero-init has already been applied.
        self._identity_init_zeroed: Tuple[str, ...] = tuple(
            name for name, mod in self.named_modules()
            if isinstance(mod, torch.nn.Linear) and not bool(mod.weight.any())
        )
