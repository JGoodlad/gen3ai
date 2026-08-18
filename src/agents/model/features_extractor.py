import torch
from torch.utils.checkpoint import checkpoint
import numpy as np
from dataclasses import dataclass, field
from gymnasium import spaces
from typing import Callable, Dict, Any, Iterator, Optional, Sequence, Tuple, NamedTuple
from agents.observation.constants import (
    POKEMON_LAST_ACTION_OFFSET,
    TRACE_INTERVAL,
    TEAM_SIZE,
    GLOBAL_ENV_DIM,
    POKEMON_FULL_DIM,
    POKEMON_HP_PROBS_OFFSET,
    POKEMON_PROTECT_OFFSET,
    POKEMON_SPECIES_KNOWN_OFFSET,
    POKEMON_SPREAD_OFFSET,
    POKEMON_SPREAD_DIM,
    POKEMON_CONDITION_OFFSET,
    POKEMON_SLEEP_BELIEF_OFFSET,
    CLOCK_OFFSET_IN_GLOBAL, CLOCK_DIM,
)
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
from agents.model.value_threat_inject import (VALUE_THREAT_INJECT_REDUCE_HOW, ValueThreatInject,
                                              value_threat_inject_dim)
from agents.model.opp_intent import AlphaIntentHead, BetaSwitchHead
from agents.model.damage_tables import N_SECONDARY as _N_SECONDARY, SECONDARY_COLS as _SECONDARY_COLS
# The LEGAL-BUT-UNOBSERVED move-prior base (the `--move-candidate-floor` default). Legality itself is
# unconditional; this is only the height of the liftable base a legal-unobserved move starts from.
from agents.model.damage_tables import _PRIOR_FLOOR
from agents.action.constants import ACTION_SPACE_SIZE
from utils.logging.levels import LogLevel


# Architecture constants — single source of truth.
# ModelVersion imports these so model_config.json always reflects the live values.
# When you change any of these, also bump MODEL_CONFIG_VERSION in model_version.py.
# gen3_arch_constants_v1: the architecture constants now live in `arch_constants.py` so
# `damage_op.py` can import them without a cycle. Re-exported here UNCHANGED — this module
# remains the documented import surface (`from agents.model.features_extractor import D_MODEL`).
from agents.model.arch_constants import (_INTENT_CELL_FEATURES,
    INTENT_THRESH_MOVE_DIM, INTENT_COND_MOVE_DIM,
    INTENT_MOVE_CELL_DIM, _INTENT_MOVE_CELL_RAW,
    PAIR_OUTCOME_MOVE_DIM, PAIR_OUTCOME_SWITCH_DIM, SWITCH_BRANCH_MOVE_DIM,
    CONDITIONAL_THREAT_SWITCH_DIM, PAIR_VALUE_ROUTE_DIM,
    UVR_K, UVR_DIM, _UVR_N_SOURCES, _UVR_N_SOURCES_FULL,
      # noqa: F401  (re-export
    VALUE_SEED_K,
    VALUE_SEED_DIM,
    ROLE_TOKEN_SIZE,
    PROJECTION_DIM,
    MOVE_NET_HIDDEN,
    MOVE_LATENT_HIDDEN,
    MOVE_LATENT_DIM,
    ROLE_ENCODER_HIDDEN,
    NET_ARCH,
    POINTER_HIDDEN,
    D_MODEL,
    TRANSFORMER_N_LAYERS,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_FFN_DIM,
)

# ============================================================================
# PHASE MODULES — split into sibling files 2026-08-16, one responsibility per file:
# extractor_ctx / encoders / team_transformer / pools / belief_heads /
# aux_value_heads / pointer_head / value_readouts. Everything is re-imported here
# EXPLICITLY because this module is the documented import hub (tests, tmp/ research
# scripts and history all import from it) and the orchestrator consumes most of it.
# ============================================================================
from agents.model.extractor_ctx import (  # noqa: F401
    Embeddings, ExtractorContext, NUM_TOKEN_TYPES, ObsUnpack, PointerInputs, TOKEN_TYPE_GLOBAL, TOKEN_TYPE_HISTORY, TOKEN_TYPE_OUR_MOVE,
    TOKEN_TYPE_OUR_TEAM, TOKEN_TYPE_THEIR_TEAM, TOKEN_TYPE_THEIR_THREAT, locate_active_slot,
    slice_pokemon_categoricals,
)
from agents.model.encoders import (  # noqa: F401
    MoveLatentEncoder, PokemonEncoder,
)
from agents.model.team_transformer import (  # noqa: F401
    BiasedEncoderLayer, EdgeBias, EventSeats, TeamTransformer, _EDGE_C1_CELL, _EDGE_C2_CELL,
    _EDGE_C3_CELL, _EDGE_C4_CELL, _EDGE_C5_CELL, _EDGE_D1_CELL, _EDGE_D2_CELL, _EDGE_D3_CELL,
    _EDGE_D4_CELL, _EDGE_FAMILIES, _EDGE_G_CELL, _EDGE_H_CELL, _EDGE_R_CELL, _EDGE_S1_CELL,
    _EDGE_S3_CELL, _EDGE_T_CELL, _EDGE_V_CELL, _EDGE_X_CELL, _KEY_PAD_NEG, _event_reference_cells,
)
from agents.model.pools import (  # noqa: F401
    CLSPool, HiddenOppBeliefPool,
)
from agents.model.belief_heads import (  # noqa: F401
    BELIEF_GRAD_MODES, BeliefHead, BeliefSlots, HPTypeBelief, ItemBelief, MoveBelief,
    SpreadBelief, _BELIEF_SUPERVISION_KEYS, _EV_DELTA_SCALE, _HP_PRESENCE_OFF_LOGIT,
    _REVEAL_LOGIT, mask_typeless_hp,
)
from agents.model.aux_value_heads import (  # noqa: F401
    ValueDistHead, WinProbHead,
)
from agents.model.pointer_head import (  # noqa: F401
    EntityMoveSeats, PointerNativeActionHead, _request_order_move_tokens,
)
from agents.model.value_readouts import (  # noqa: F401
    MultiSeedValueReadout, UnifiedValueReadout,
)



# Differentiable damage operator (`DamageOperator`) constants — the unified 3-roll + P(KO) damage
# representation (owner-chosen: "the raw rolls the model can read, PLUS the P(KO) the policy uses").
# Per defender, two gen3 type CHANNELS (physical / special) each carry [low_roll, high_roll, crit, pko,
# accuracy]: the three rolls as a fraction of the defender's MAX HP (stationary "how big is the hit" —
# low = the 0.85 roll, high = the max roll, crit = the ×2 screen-ignoring crit), pko = the
# accuracy-discounted P(KO this turn) vs CURRENT HP (= acc·P(KO|hit), the EXACT realized KO probability —
# independent events), and accuracy = the dominant threat's base hit probability. {pko, accuracy} together
# parameterize the full miss/survive/KO outcome distribution with every product PRE-COMPUTED (so the ReLU
# head never has to multiply — the operator does the multiplicative physics, the head reasons additively).
# Then p_outspeed + threat_provenance:
#   [phys_low, phys_high, phys_crit, phys_pko, phys_acc, spec_low, spec_high, spec_crit, spec_pko, spec_acc,
#    p_outspeed, prov]
# Driven by the LEARNED belief instead of the usage prior. NOT modifier-for-modifier parity: the op
# applies type/STAB/ability-immunity/screens/crit/accuracy; it does NOT (yet) apply weather, burn, defender
# boost stages, or fixed-damage/OHKO/HP-relative moves, and p_outspeed is a point estimate (no para/boost
# speed distribution). Those are the documented v2 follow-ups (the gradient story holds without them).
# gen3_damage_op_split_v1: the DamageOperator (1,689 lines) + its constants + the decode mirror
# now live in `damage_op.py` — 39% of this file for one concern. Re-exported UNCHANGED so every
# historical import path still resolves (the prober, model_version, snapshot and the tests all do
# `from agents.model.features_extractor import DamageOperator / decode_damage_block / _DMG_*`).
from agents.model.t0_species import T0SpeciesPrior
from agents.model.intent_value_reduce import IntentValueReduce
from agents.model.intent_move_cell import IntentMoveCell
from agents.model.intent_threshold import (
    IntentThresholdMoveCell, IntentThresholdValue, ThresholdProbs, threshold_probs)
from agents.model.intent_conditional import IntentConditionalMoveCell
from agents.model.pair_outcome import (
    PairOutcomeMoveCell, PairOutcomeSwitchCell, pair_alpha, reduce_pair_in, reduce_pair_in_all)
from agents.model.switch_branch import SwitchBranchMoveCell
from agents.model.conditional_threat import ConditionalThreatCell
from agents.model.pair_value_route import PairValueInject  # noqa: F401  (re-export)
from agents.model.value_routes import ValueClockRoute, ValueIntentRoute
from agents.model.damage_op import _OUT_SEC_COLS as _OSC
_OUT_SEC_FLINCH_COL = _OSC.index("flinch")   # gen3_intent_conditional_v1: fails at import if dropped
from agents.model.damage_op import (  # noqa: F401,E501  (re-export)
    DamageOperator,
    OpTensors,
    _COND_BRN_IDX,
    _COND_PAR_IDX,
    _COND_SLP_IDX,
    _DMG_CB,
    _DMG_CB_PER_MON,
    _DMG_CHANNEL_FEATS,
    _DMG_CHIP_CAP,
    _DMG_CRIT_CAP,
    _DMG_CRIT_P,
    _DMG_EFFECT,
    _DMG_EFFECT_COLS,
    _DMG_IDX_OUTSPEED,
    _DMG_IDX_PHYS_ACC,
    _DMG_IDX_PHYS_CRIT,
    _DMG_IDX_PHYS_HIGH,
    _DMG_IDX_PHYS_LOW,
    _DMG_IDX_PHYS_PKO,
    _DMG_IDX_PROVENANCE,
    _DMG_IDX_SPEC_ACC,
    _DMG_IDX_SPEC_CRIT,
    _DMG_IDX_SPEC_HIGH,
    _DMG_IDX_SPEC_LOW,
    _DMG_IDX_SPEC_PKO,
    _DMG_IMX_CELL,
    _DMG_IMX_HDR_ACC,
    _DMG_IMX_HDR_EFFECT,
    _DMG_IMX_HDR_PHYS,
    _DMG_IMX_HDR_SEC,
    _DMG_IMX_HDR_W,
    _DMG_IMX_HEADER,
    _DMG_N_CHANNELS,
    _DMG_OAX,
    _DMG_OAX_IDX_CRIT,
    _DMG_OAX_IDX_HIGH,
    _DMG_OAX_IDX_LOW,
    _DMG_OAX_IDX_PKO,
    _DMG_OAX_N_MOVES,
    _DMG_OAX_PER_MON,
    _DMG_OAX_PER_MOVE,
    _DMG_OMX,
    _DMG_OMX_CELL,
    _DMG_OMX_IDX_CRIT,
    _DMG_OMX_IDX_HIGH,
    _DMG_OMX_IDX_LOW,
    _DMG_OMX_IDX_MULT,
    _DMG_OMX_IDX_PKO,
    _DMG_OUTGOING,
    _DMG_OUT_N_MOVES,
    _DMG_OUT_PER_MOVE,
    _DMG_OUT_SEC,
    _N_OUT_SECONDARY,
    _OUT_SEC_COLS,
    _OUT_SEC_KEEP,
    _DMG_PARA_SPEED,
    _DMG_PER_MON,
    _DMG_REFINE_K,
    _DMG_ROLL_MIN,
    _DMG_SPEED_SCALE,
    _DMG_SPEED_STD_K,
    _DMG_STATUS,
    _DMG_STATUS_N_MOVES,
    _DMG_STATUS_REFINE,
    _DMG_TOPK_DEFAULT_K,
    _FIRE_TIDX,
    _IMMOBILIZE_STATUS_CATS,
    _SB_ATK,
    _SB_DEF,
    _SB_SPA,
    _SB_SPD,
    _SB_SPE,
    _SECONDARY_MAJOR_N,
    _SECONDARY_TO_STATUS_CAT,
    _SUBSTITUTE_CTX_IDX,
    _WATER_TIDX,
    _dmg_imx_dim,
    decode_damage_block,
)



@dataclass
class ExtractorStashes:
    """Every per-forward SIDE VALUE the extractor exposes, as ONE typed unit
    (`gen3_extractor_stashes_v1` — the OpStashes recipe applied to the orchestrator). The v89
    lesson made the cost of the old pattern concrete: phases communicated through mutable
    `self.last_*` instance stashes, consumers — including other MODULES — read them with
    `getattr`, and nothing type-level connected producer to consumer, so a consumer rewiring
    silently orphaned five value routes for two generations. `forward_internal` replaces the
    whole container at ENTRY, so a stale cross-batch read is unrepresentable for ANY stash.
    Reads stay on the `fe.last_*` properties (the documented consumer surface — the policy's
    pointer head + dist critic, instrumented_ppo's aux losses, the prober and inference all
    keep their spelling); WRITES go through `fe.stash.<field>` — a stray write to a `last_*`
    name now fails loud (AttributeError) instead of silently forking the state.

    Every stash is leak-safe by construction: they are OUTPUTS of the forward, read by aux
    losses / the prober / the policy heads, never fed back as inputs that could carry a label."""
    # --- pointer / action head (read by Gen3DualHeadMaskablePolicy) -------------------------
    pointer_inputs: Optional[PointerInputs] = None   # request-ordered move tokens + valid mask
    #                                                  + our team tokens + the op's per-action cells
    # --- intent heads (T2 publications — stop-grad under belief_grad_mode=label_only) --------
    alpha_logits: Optional[torch.Tensor] = None      # [B,K+1] which believed move (or SWITCH)
    alpha_seat_nums: Optional[torch.Tensor] = None   # [B,K] seat move NUMS (detached; loss labels)
    beta_logits: Optional[torch.Tensor] = None       # [B,6] if they switch, to whom
    thresh_probs: Optional[ThresholdProbs] = None    # T2-computed, read by the vf route at T3
    # --- belief bank (publications; the LIVE views live in `belief_supervision`) -------------
    belief_logits: Optional[Dict[str, torch.Tensor]] = None  # species/moves aux dict (refined opp)
    opp_believed_mask: Optional[torch.Tensor] = None  # [B,6] bool: un-revealed opp slots
    opp_active_local: Optional[torch.Tensor] = None   # [B] opp active idx (prober belief-row decode)
    move_belief_logits: Optional[torch.Tensor] = None  # [B,6,n_moves] TYPED posterior (11 readers)
    move_latent_table: Optional[torch.Tensor] = None   # [n_moves,MOVE_LATENT_DIM] grading target
    spread_belief: Optional[torch.Tensor] = None       # [B,6,5] believed derived stats
    spread_nature_logits: Optional[torch.Tensor] = None  # [B,6,25] (gen3_nature_ev_belief_v1)
    spread_ev: Optional[torch.Tensor] = None           # [B,6,5] believed EVs
    item_logits: Optional[torch.Tensor] = None         # [B,6,n_items] hidden-item posterior
    hp_type_logits: Optional[torch.Tensor] = None      # [B,6,16] HP-type head (aux CE + prober)
    # --- op / physics -------------------------------------------------------------------------
    damage_block: Optional[torch.Tensor] = None        # [B,out_dim] prober decode; never read fwd
    # --- value-side readouts ------------------------------------------------------------------
    value_pooled: Optional[torch.Tensor] = None        # [B,D_MODEL] the FitNets HINT layer
    win_prob_logits: Optional[torch.Tensor] = None     # [B,1] P(win) logit (aux BCE + prober)
    value_dist_logits: Optional[torch.Tensor] = None   # [B,bins] dist-critic atoms (E[Z] source)
    # --- same-forward hand-offs (T0 producer → T1/T2 consumer; internal, no `last_*` name) ----
    t0_species_probs: Optional[torch.Tensor] = None    # T0 species resolve → every T1 pricing site
    entity_latent_table: Optional[torch.Tensor] = None  # LIVE latent table → the E4 seat builder
    # The LIVE (graph-carrying) belief outputs for the SUPERVISED aux losses only — see
    # `belief_supervision()`. Under shaping/detached these are the identical objects the `last_*`
    # stashes hold; under label_only the `last_*` stashes are their stop-grad publications.
    # Container replacement at forward entry is what makes "key absent ⇒ the head did not run
    # this forward" true (these hold graph-carrying tensors, so a stale one would backprop
    # through a freed — or worse, a different minibatch's — graph).
    belief_supervision: Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)


def compute_projection_widths(layout: Dict[str, Any], *, opp_belief_cls_k: int = 0,
                              damage_op: bool = False) -> Tuple[int, int]:
    """The `(pi, vf)` projection-input widths as STATIC ARITHMETIC (gen3_static_widths_v1).

    Mirrors `ProjectionAssembler.forward`'s concat exactly — this is the single place that
    arithmetic lives, and `projection_width_test.py` sweeps flag combos asserting a REAL
    forward's measured widths equal it (the old construction-time discovery forward,
    preserved as the verifier). Only THREE inputs move a width: the layout (the
    `non_matchup_rest` scalar tail), the hidden-opp belief pool (`opp_belief_cls_k`
    queries × D_MODEL, both heads), and the op (whose presence appends the critic's
    multi-seed window, vf only). Every other flag is width-neutral by construction: the
    v89 value routes inject ADDITIVELY into `value_pooled`, the intent cells widen the
    pointer stash (not pi/vf), and the token-stream enrichments change content, not shape.

    Pure — importable and unit-testable without building a model. `damage_op` gates the
    seed window because the assembler builds `MultiSeedValueReadout` iff the op exists,
    and the op (when built) runs every forward, so its `incoming_rows` are always fed.
    """
    from agents.observation.schema import build_schema
    sl = build_schema(layout).slices()
    # The non-matchup scalar tail: global-env block + the 5 raw board scalars — everything
    # between the active contexts and the embedding-ID active_req_moves tail (ObsUnpack).
    non_matchup_rest = sl['reactive.active_req_moves'].start - sl['global_env'].start
    belief = opp_belief_cls_k * D_MODEL
    # pi: our_team_pooled + their_team_pooled + our_active_refined (D_MODEL each) + tail + belief.
    pi = 3 * D_MODEL + non_matchup_rest + belief
    # vf: value_pooled (D_MODEL) + tail + belief + the critic's multi-seed window (op only).
    vf = D_MODEL + non_matchup_rest + belief + ((VALUE_SEED_K * VALUE_SEED_DIM) if damage_op else 0)
    return pi, vf


class ProjectionAssembler(torch.nn.Module):
    """Assembles the pre-projection inputs for BOTH heads.

    Policy input: team pools + our active token + the non-matchup scalar tail.
    Value input: the value-dedicated pool + the same scalar tail (+ the critic's seed window).

    gen3_ctx_dedup_v1: the per-side ENCODED active contexts (`active_ctx_encoder` on the raw
    58-dim boosts+volatiles blocks) are DELETED from both heads. They were duplicated delivery
    with a 1:1 entity-native replacement already live: the E2 injection scatters each side's
    FULL raw ctx block onto its ACTIVE mon's role token (`gen3_entity_rehome_v1`, pinned by
    `e2_ctx_injection_test.py`), and the global token carries both raw blocks as a second
    route — so every scalar the encoder saw reaches both heads through the trunk, entity-
    attached instead of positionally concatenated. `non_matchup_rest` STAYS: its only token
    route is the global token, which no pool reads directly, so the concat is currently its
    one direct path to the heads (no 1:1 replacement — its re-home is a separate decision).
    """

    def __init__(self, layout: Dict[str, Any], seed_per_mon: int = 0):
        super().__init__()
        # gen3_no_concat_v1: the critic's multi-seed window (None when the config has no op —
        # then there are no our_mon rows to read and vf keeps its pooled-only shape).
        self._seed_per_mon = seed_per_mon
        self.seed_readout = MultiSeedValueReadout(seed_per_mon) if seed_per_mon > 0 else None

    def forward(self, our_team_pooled: torch.Tensor, their_team_pooled: torch.Tensor,
                our_active_refined: torch.Tensor, value_pooled: torch.Tensor,
                ctx: ExtractorContext,
                hidden_opp_belief: Optional[torch.Tensor] = None,
                seed_rows: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate the per-head pre-projection inputs → `(pi_combined, vf_combined)` [B, *].
        `hidden_opp_belief` [B, K*D_MODEL] feeds both heads when built; `seed_rows` (the op's typed
        incoming rows [B,6,per_mon]) feeds the critic's multi-seed readout, vf only."""
        pi_parts = [our_team_pooled, their_team_pooled, our_active_refined,
                    ctx.non_matchup_rest]
        vf_parts = [value_pooled, ctx.non_matchup_rest]
        # gen3_no_concat_v1 (v61): THE OP HEAD-CONCAT IS DEAD. The 660-dim flat block no longer
        # enters either head — its measured end-state (gen-4, stratified, 53ef270): net policy
        # dependence +0.00%, all-edges-off ABOVE the concat arm on flips, and the critic's
        # magnitude content decodable without it (act_threat vf r² 0.418 concat-zeroed). The op
        # itself lives on: pointer cells (policy, lossless per-action), prefuse token injection,
        # the D/S/C/V/T/X edge cells, and `last_raw_block` for the probes. The critic's
        # replacement window is the multi-seed readout below (vf only).
        # Hidden-opponent belief (flag-guarded; None when off) feeds BOTH heads — the policy reads
        # the threat over the hidden team, the value reads its winning-ness. Appended last so the
        # off-by-default block layout is unchanged (`compute_projection_widths` sizes the
        # projections; a new width-contributing part must be added THERE too — the sweep test
        # fails on any drift).
        if hidden_opp_belief is not None:
            pi_parts.append(hidden_opp_belief)
            vf_parts.append(hidden_opp_belief)
        # gen3_no_concat_v1 / gen3_op_tensors_views_v1: the multi-seed critic readout over the
        # op's per-our-mon rows — now handed in as the TYPED view (`OpTensors.incoming_rows`,
        # post-gain), so the assembler holds no flat-block offsets. vf ONLY.
        if seed_rows is not None and self.seed_readout is not None:
            alive = (ctx.hp_and_active[:, :TEAM_SIZE, 0] > 0).float()
            vf_parts.append(self.seed_readout(seed_rows, alive))
        pi_combined = torch.cat(pi_parts, dim=1)
        vf_combined = torch.cat(vf_parts, dim=1)
        return pi_combined, vf_combined


class Gen3FeaturesExtractor(torch.nn.Module):
    """Orchestrates the phase modules. Data flow (bracketed phases are flag-gated; all off = the
    baseline ObsUnpack → PokemonEncoder → TeamTransformer → CLSPool → ProjectionAssembler):
        ObsUnpack → PokemonEncoder → [BeliefSlots?] → TeamTransformer → [BeliefHead?] → [MoveBelief?]
          → CLSPool → [HiddenOppBeliefPool?] → [DamageOperator?] → ProjectionAssembler
    then a final pre-projection LayerNorm + Linear + ReLU head per side. `MoveBelief` reinjects the
    believed moveset into the opp tokens BEFORE the pools (so it flows to the heads via cross-attention);
    `DamageOperator` runs AFTER the pools and consumes the move-belief logits, appending its features to
    both projection inputs (it does not enter the token stream). The optional `WinProbHead`
    (`win_prob_mode`) reads `value_pooled` AFTER the pools and stashes a P(win) logit as a SIDE readout
    (never in pi/vf — leak-safe). The embedding tables live in
    `self.embeddings` (shared) and are passed into the phases that need them. See
    `src/agents/model/CLAUDE.md` for the phase-module contract."""

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
                 intent_value_reduce: bool = False,
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
                 value_clock: bool = False,
                 value_intent: bool = False,
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
        # gen3_intent_value_reduce_v1 (step 6). Requires BOTH operands to exist, and both
        # requirements are fail-loud rather than silently degrading: no alpha => nothing to weight
        # with; no op => nothing to weight. `stash_pair_cells` is what makes the op keep the
        # un-reduced tensor at all, so an off run pays nothing.
        self.intent_value_reduce = None
        if intent_value_reduce:
            self.intent_value_reduce = IntentValueReduce(
                TEAM_SIZE, _INTENT_CELL_FEATURES, D_MODEL)
            if not opp_intent:
                raise ValueError(
                    "intent_value_reduce=True requires opp_intent=True — the reduction is WEIGHTED "
                    "BY alpha, and with no alpha head there is no distribution to weight with.")
            if not damage_op:
                raise ValueError(
                    "intent_value_reduce=True requires damage_op=True — it reduces the operator's "
                    "un-reduced per-(our mon, believed move) cells, which nothing else produces.")
        # gen3_intent_move_cell_v1 (G3, design_conditional_execution.md): the POLICY-side alpha
        # consumer — the c2 status-consequence family re-delivered through the pointer MOVE cell,
        # alpha-conditioned. Same fail-loud requirement shape as intent_value_reduce: no alpha =>
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
        # α-weighted THRESHOLD operator — five mechanics through the pointer MOVE cell plus the
        # p_KO critic route (ledger H1). One flag builds BOTH consumers; the shared probs are
        # computed once at the pointer stash (T2) and the vf half reads the stash at T3.
        self.intent_threshold_move = None
        self.intent_threshold_value = None
        if intent_threshold:
            self.intent_threshold_move = IntentThresholdMoveCell(INTENT_THRESH_MOVE_DIM)
            self.intent_threshold_value = IntentThresholdValue(D_MODEL)
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
        # gen3_value_direct_routes_v1 (v87): two direct critic routes, both zero-init vf-tail
        # appends. The clock route has no dependency (the ctx always carries the global block);
        # the intent route consumes the α/β PUBLICATIONS and so requires the intent heads.
        self.value_clock_route = ValueClockRoute() if value_clock else None
        self.value_intent_route = None
        if value_intent:
            if not opp_intent:
                raise ValueError(
                    "value_intent=True requires opp_intent=True — the route feeds the critic "
                    "the α/β posteriors, and with no intent heads there is nothing to feed.")
            self.value_intent_route = ValueIntentRoute(entity_topk_seats)
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
        # gen3_intent_value_reduce_v1 (step 6): ask the op to KEEP its un-reduced cells. Set here,
        # after the op exists, because the flag lives on the op but is owned by a consumer built
        # before it. Without this the consumer sees `last_pair_cells is None` and — by design —
        # raises rather than contributing zeros, since a silent no-op reads exactly like a null.
        if (self.intent_value_reduce is not None or self.intent_threshold_move is not None
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
        # Value-head active readout (weight-shape via flag): adds our_active_refined (D_MODEL) to the
        # value projection. OFF reproduces the baseline value head byte-for-byte (no ARCH_SIGNATURE bump).
        self.assembler = ProjectionAssembler(
            layout,
            seed_per_mon=(self.damage_op.per_mon if self.damage_op is not None else 0))

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
            layout, opp_belief_cls_k=opp_belief_cls_k, damage_op=damage_op)
        # Tie the vf seed-window term to the constructed readout — a drift between the
        # arithmetic's constants and the assembler's module is a construction-time crash here,
        # never a first-forward shape error.
        if self.assembler.seed_readout is not None:
            assert self.assembler.seed_readout.out_dim == VALUE_SEED_K * VALUE_SEED_DIM, (
                f"compute_projection_widths' seed-window term "
                f"({VALUE_SEED_K * VALUE_SEED_DIM}) has drifted from "
                f"MultiSeedValueReadout.out_dim ({self.assembler.seed_readout.out_dim})")

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

        # gen3_identity_init_guard_v1 — SNAPSHOT the identity-at-init contract. See
        # `restore_identity_init` for why this exists; it must be the LAST thing __init__ does, so
        # every module is built and every deliberate zero-init has already been applied.
        self._identity_init_zeroed: Tuple[str, ...] = tuple(
            name for name, mod in self.named_modules()
            if isinstance(mod, torch.nn.Linear) and not bool(mod.weight.any())
        )

    def disable_observation_debugger(self) -> bool:
        """Detach the `ObservationDebugger`. Returns True if one was attached.

        The debugger runs NUMPY assertions inside `forward`. That is fine eagerly, but `torch.compile`
        cannot trace it — dynamo dies building a guard over a numpy bool ("TypeError: 'numpy.bool'
        object cannot be interpreted as an integer"). It is a LEARNER-side diagnostic and a frozen
        opponent has no use for it, so the compile path drops it.

        This is a METHOD rather than the caller reaching in and setting `fe._debugger = None`, so the
        ownership stays here: if the debugger ever gains teardown state, this is the one place that
        has to learn about it."""
        had = self._debugger is not None
        self._debugger = None
        return had

    def restore_identity_init(self) -> int:
        """Re-zero every Linear this extractor deliberately zero-initialised. Returns the count.

        WHY THIS EXISTS. SB3's `ActorCriticPolicy._build()` runs
        ``self.features_extractor.apply(partial(self.init_weights, gain=sqrt(2)))``
        (stable_baselines3/common/policies.py:617-631), and `init_weights` ORTHOGONALLY
        re-initialises EVERY `nn.Linear` it finds. `ortho_init` defaults True and nothing here
        overrides it — so every deliberate zero-init inside the extractor was silently destroyed the
        moment the policy was built, in every real training run.

        That silently falsified a documented invariant for every shipped zero-init feature —
        `prefuse_proj`, the edge-bias family maps and `film_pi`/`film_vf` all claim
        "zero-init ⇒ identity-at-init ⇒ ON starts byte-identical" — and,
        more insidiously, the belief heads' cold-start contract: `MoveBelief.move_head`,
        `SpreadBelief.{stat,nature,ev}_head` and `HPTypeBelief.type_head` are zero-init precisely so
        the cold-start posterior EQUALS the Smogon prior. Clobbered, they start at prior ⊕ noise.

        It went unnoticed because every unit test builds the module or a bare extractor DIRECTLY,
        where the zero-init survives; only SB3-wrapped construction destroys it.

        The set is captured by OBSERVATION at the end of `__init__` (any Linear whose weight is
        all-zero once construction finishes was zero-init'd on purpose) rather than by a hand-kept
        list, so a future zero-init module is protected automatically — the failure mode that
        produced this bug cannot recur by omission. Biases are not tracked: SB3's `init_weights`
        already zeroes every bias.
        """
        by_name = dict(self.named_modules())
        n = 0
        for name in self._identity_init_zeroed:
            mod = by_name.get(name)
            if isinstance(mod, torch.nn.Linear):
                torch.nn.init.zeros_(mod.weight)
                if mod.bias is not None:
                    torch.nn.init.zeros_(mod.bias)
                n += 1
        return n

    def set_belief_grad_mode(self, mode: str) -> None:
        """Apply a belief-grad-mode at RUNTIME (the --allow-belief-grad-mode-change migration path).

        SB3's load reconstructs the extractor from the ZIP's saved policy_kwargs, so a resume that
        passes a different --belief-grad-mode would otherwise be a SILENT NO-OP (the 2026-07-21
        incident: the migration notice printed but the loaded extractor kept 'detached' —
        grad/*_norm_shared stayed exactly 0). The mode lives in THREE places (this attr,
        `_belief_detach`, and the `detach_read` flag stamped on each belief head); this is the ONE
        setter that updates them all — call it post-load on the resume path (a no-op when unchanged).

        gen3_belief_label_only_v1 widened that to FOUR places (`_belief_label_only` and the heads'
        `publish_detach` join the list) — which is exactly why the stamping is now a single
        `_stamp_belief_grad_flags()` shared with `__init__`, rather than a loop duplicated in two
        methods that a future mode could update in only one of."""
        if mode not in BELIEF_GRAD_MODES:
            raise ValueError(f"belief_grad_mode must be one of {'|'.join(BELIEF_GRAD_MODES)}, "
                             f"got {mode!r}")
        changed = mode != getattr(self, "belief_grad_mode", None)
        self.belief_grad_mode = mode
        self._belief_detach = (mode == "detached")
        self._belief_label_only = (mode == "label_only")
        self._stamp_belief_grad_flags()
        if changed:
            print(f"[Gen3FeaturesExtractor] belief_grad_mode APPLIED at runtime -> {mode!r} "
                  f"(detach_read={'on' if self._belief_detach else 'off'}, "
                  f"publish_detach={'on' if self._belief_label_only else 'off'} across the belief heads)")

    def _stamp_belief_grad_flags(self) -> None:
        """Push `belief_grad_mode` down onto the heads — the ONE place either per-head flag is set.

        `detach_read` (cut route B, the trunk read) goes on all four state-prediction heads.
        `publish_detach` (cut route C, the head's own reinjection) goes on the three that HAVE a
        reinjection; `BeliefHead` is a pure readout with nothing to publish, and the extractor-level
        `_publish_belief` covers every consumer that reads a stash rather than being handed the tensor
        by the head. BeliefSlots has no predictive read at all and is intentionally absent.
        """
        _item = getattr(self, "item_belief_head", None)
        for _bh in (self.move_belief, self.spread_belief, self.hp_type_belief_head,
                    _item, self.belief_head):
            if _bh is not None:
                _bh.detach_read = self._belief_detach
        for _bh in (self.move_belief, self.spread_belief, self.hp_type_belief_head,
                    _item):
            if _bh is not None:
                _bh.publish_detach = self._belief_label_only

    # Read-only forwarders for the shared embedding tables — they are a model-level concept
    # and several tests/inspectors reach for them by name. Properties add no state_dict keys.
    @property
    def species_embedding(self) -> torch.nn.Embedding: return self.embeddings.species_embedding
    @property
    def move_embedding(self) -> torch.nn.Embedding: return self.embeddings.move_embedding
    @property
    def item_embedding(self) -> torch.nn.Embedding: return self.embeddings.item_embedding
    @property
    def ability_embedding(self) -> torch.nn.Embedding: return self.embeddings.ability_embedding
    @property
    def type_embedding(self) -> torch.nn.Embedding: return self.embeddings.type_embedding
    @property
    def hp_type_idx_map(self) -> torch.Tensor: return self.embeddings.hp_type_idx_map

    # gen3_extractor_stashes_v1 — the READ surface over the typed stash container (see
    # ExtractorStashes). Every consumer keeps its historical `last_*` spelling — the policy's
    # pointer head + dist critic, instrumented_ppo's aux losses, the prober, inference — and a
    # stray WRITE to any of these names raises AttributeError (no setter) instead of silently
    # forking the state. Writes go through `self.stash.<field>` only.
    @property
    def last_pointer_inputs(self) -> Optional[PointerInputs]: return self.stash.pointer_inputs
    @property
    def last_alpha_logits(self) -> Optional[torch.Tensor]: return self.stash.alpha_logits
    @property
    def last_alpha_seat_nums(self) -> Optional[torch.Tensor]: return self.stash.alpha_seat_nums
    @property
    def last_beta_logits(self) -> Optional[torch.Tensor]: return self.stash.beta_logits
    @property
    def last_belief_logits(self) -> Optional[Dict[str, torch.Tensor]]: return self.stash.belief_logits
    @property
    def last_opp_believed_mask(self) -> Optional[torch.Tensor]: return self.stash.opp_believed_mask
    @property
    def last_opp_active_local(self) -> Optional[torch.Tensor]: return self.stash.opp_active_local
    @property
    def last_move_belief_logits(self) -> Optional[torch.Tensor]: return self.stash.move_belief_logits
    @property
    def last_move_latent_table(self) -> Optional[torch.Tensor]: return self.stash.move_latent_table
    @property
    def last_spread_belief(self) -> Optional[torch.Tensor]: return self.stash.spread_belief
    @property
    def last_spread_nature_logits(self) -> Optional[torch.Tensor]: return self.stash.spread_nature_logits
    @property
    def last_spread_ev(self) -> Optional[torch.Tensor]: return self.stash.spread_ev
    @property
    def last_item_logits(self) -> Optional[torch.Tensor]: return self.stash.item_logits
    @property
    def last_hp_type_logits(self) -> Optional[torch.Tensor]: return self.stash.hp_type_logits
    @property
    def last_damage_block(self) -> Optional[torch.Tensor]: return self.stash.damage_block
    @property
    def last_value_pooled(self) -> Optional[torch.Tensor]: return self.stash.value_pooled
    @property
    def last_win_prob_logits(self) -> Optional[torch.Tensor]: return self.stash.win_prob_logits
    @property
    def last_value_dist_logits(self) -> Optional[torch.Tensor]: return self.stash.value_dist_logits
    # Private per-forward hand-offs with external test readers keep their names too (same
    # read-only discipline; the T0->T1/T2 contract is documented on the dataclass fields).
    @property
    def _thresh_probs(self) -> Optional[ThresholdProbs]: return self.stash.thresh_probs
    @property
    def _entity_latent_table(self) -> Optional[torch.Tensor]: return self.stash.entity_latent_table
    @property
    def _belief_supervision(self) -> Dict[str, Optional[torch.Tensor]]:
        return self.stash.belief_supervision

    # gen3_pointer_native_v1: the pointer head's per-action cell widths — the policy sizes the head's
    # Linears from these at build time (0 when the source op block is off, so a missing block narrows
    # the Linear instead of silently feeding zeros at a learned weight).
    @property
    def pointer_move_token_dim(self) -> int:
        # gen3_entity_move_seats_v1: the head reads the REFINED E3 seats (d_model), no longer the raw
        # 32-dim PokemonEncoder move tokens.
        return D_MODEL

    @property
    def pointer_move_cell_dim(self) -> int:
        # gen3_intent_move_cell_v1: the alpha-conditioned c2 channels widen the move cell when on
        # (the policy sizes the pointer move scorer's in_features from this at build time).
        base = self.damage_op.pointer_move_cell_dim if self.damage_op is not None else 0
        base += INTENT_MOVE_CELL_DIM if self.intent_move_cell is not None else 0
        # gen3_intent_threshold_v1: the five-mechanic threshold channels widen the move cell too.
        base += INTENT_THRESH_MOVE_DIM if self.intent_threshold_move is not None else 0
        # gen3_intent_conditional_v1: the Counter/flinch/Explosion/Pursuit cells likewise.
        base += INTENT_COND_MOVE_DIM if self.intent_conditional is not None else 0
        # gen3_pair_outcome_v1: the α-reduced unified outcome vector at our ACTIVE defender.
        base += PAIR_OUTCOME_MOVE_DIM if self.pair_outcome_move is not None else 0
        # gen3_switch_branch_v1: OA2 + spinblock + Protect's α-conditioning.
        return base + (SWITCH_BRANCH_MOVE_DIM if self.switch_branch is not None else 0)
    @property
    def pointer_switch_cell_dim(self) -> int:
        base = self.damage_op.pointer_switch_cell_dim if self.damage_op is not None else 0
        # gen3_pair_outcome_switch_v1: the FIRST widener of the switch cell — mon j's own α-reduced
        # outcome row + the spin-denial coordinate.
        base += PAIR_OUTCOME_SWITCH_DIM if self.pair_outcome_switch is not None else 0
        # gen3_conditional_threat_v1 (OA1): the SECOND — the four coordinates that row cannot carry.
        return base + (CONDITIONAL_THREAT_SWITCH_DIM
                       if self.conditional_threat is not None else 0)

    def _publish_belief(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Hand a belief output to the FORWARD (reinject / the op / the edge cells / the seats / the
        pointer stash / the prober). Under `label_only` this is a STOP-GRAD copy — gen3_belief_label_only_v1.

        The cut lives at the ONE publish boundary per head rather than at each consumer, and that is the
        whole design: `last_move_belief_logits` alone has eleven downstream readers, so a per-consumer
        rule would be one forgotten site away from silently reopening the route. Publishing instead
        isolates a consumer added TOMORROW by construction.

        Returns the identical object under shaping/detached (and for `None`), so those modes stay
        byte-identical — `detach()` never changes a value, only the graph, so even under `label_only` the
        FORWARD is bit-identical and only the backward differs.
        """
        return t.detach() if (self._belief_label_only and t is not None) else t

    def belief_supervision(self, name: str) -> Optional[torch.Tensor]:
        """The LIVE (graph-carrying) belief output `name`, for the SUPERVISED aux losses ONLY.

        ⚠️ **An aux loss MUST read its target through here, never off the `last_*` attribute.** Under
        `label_only` that attribute is a stop-grad publication (`_publish_belief`), so a loss reading it
        would train nothing at all — and would do so SILENTLY, since the loss value and every metric
        derived from it look exactly the same. The gate test that would catch it is
        `belief_label_only_gate_test.py::test_every_belief_loss_still_trains_its_head`.

        Returns the identical object the `last_*` stash holds under shaping/detached, and `None` for a
        head that is not built (the caller's existing `is None` guards are unchanged).
        """
        if name not in _BELIEF_SUPERVISION_KEYS:
            raise KeyError(
                f"unknown belief supervision key {name!r}; expected one of "
                f"{sorted(_BELIEF_SUPERVISION_KEYS)}. Add the key to _BELIEF_SUPERVISION_KEYS and "
                "register the LIVE tensor where the head's stash is published."
            )
        return self.stash.belief_supervision.get(name)

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
        # PPO→alpha_head route through this path exactly as it does for intent_value_reduce.
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
        # `value_entity_pool.out_proj` and `intent_value_reduce.proj` bit-exact ZERO after
        # 25M steps, while `value_threat_proj` — the one value_pooled route — trained to
        # 0.117). `vf_parts[0] is value_pooled`, so the SAME wiring feeds `value_net` when
        # the scalar critic is on: one wiring, both parameterizations. Every route stays
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
        out: Tuple[torch.Tensor, torch.Tensor] = self.assembler(
                             our_team_pooled, their_team_pooled, our_active_refined, value_pooled,
                             ctx, belief,
                             self.damage_op.last_tensors.incoming_rows  # type: ignore[union-attr]
                             if damage_block is not None else None)
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
        raise when ON but inputs are missing (silence is indistinguishable from a null result)."""
        if self.intent_value_reduce is not None:
            _cells = self.damage_op.last_pair_cells if self.damage_op is not None else None
            if _cells is None or self.last_alpha_logits is None:
                raise RuntimeError(
                    "intent_value_reduce is on but the op stashed no un-reduced cells or alpha "
                    "produced no logits — the term would silently contribute nothing, which is "
                    "indistinguishable from a null RESULT.")
            yield "intent_value_reduce", self.intent_value_reduce(
                self.last_alpha_logits, _cells, self.damage_op.last_pair_gate)  # type: ignore[union-attr]
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
        if self.intent_threshold_value is not None:
            if self._thresh_probs is None:
                raise RuntimeError(
                    "intent_threshold is on but the pointer stash computed no threshold "
                    "probs — the vf route would silently contribute nothing.")
            yield "intent_threshold_value", self.intent_threshold_value(*self._thresh_probs)
        if self.value_clock_route is not None:
            _clock = ctx.non_matchup_rest[:, CLOCK_OFFSET_IN_GLOBAL:CLOCK_OFFSET_IN_GLOBAL + CLOCK_DIM]
            yield "value_clock", self.value_clock_route(_clock)
        if self.value_intent_route is not None:
            if self.last_alpha_logits is None or self.last_beta_logits is None:
                raise RuntimeError(
                    "value_intent is on but alpha/beta produced no logits — the route would "
                    "silently contribute nothing.")
            yield "value_intent", self.value_intent_route(
                self.last_alpha_logits, self.last_beta_logits)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a (pi_features, vf_features) tuple — both [B, PROJECTION_DIM].

        The consuming policy (`Gen3DualHeadMaskablePolicy`) unpacks the tuple and routes
        each half to its own mlp_extractor branch. Standard SB3 policies expect a single
        tensor, so this extractor MUST be paired with that custom policy.
        """
        pi_combined, vf_combined = self.forward_internal(obs)
        if self._debugger is not None:
            self._debugger.on_forward(obs["observation"])
        pi_pre = self.projection(self.pre_proj_norm(pi_combined))
        vf_pre = self.value_projection(self.value_pre_norm(vf_combined))
        pi_features = self.activation(pi_pre)
        vf_features = self.activation(vf_pre)
        return pi_features, vf_features
