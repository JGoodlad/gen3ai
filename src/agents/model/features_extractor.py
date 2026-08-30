"""`agents.model.features_extractor` — `Gen3FeaturesExtractor` and the re-export HUB.

THE documented import surface for the whole model package: every phase module, every damage-op
constant and every architecture constant resolves from here regardless of which file defines it,
because tests, `tmp/` research scripts, the prober and years of history all import from this
name. That is why it carries a file-wide `F401` exemption in `ruff.toml` (PERMANENT, measured:
33 findings without it, nearly all names other modules import back out through the hub).

The class itself is decomposed one-responsibility-per-file and assembled by inheritance, so
`Gen3FeaturesExtractor` is a single `torch.nn.Module` subclass with every attribute PATH — and
therefore every `state_dict` key — exactly where it has always been:

    extractor_stashes.py   `ExtractorStashes`, the per-forward side-value container
    projection.py          the static width arithmetic + `ProjectionAssembler`
    extractor_build.py     `ExtractorBuild`   — `__init__`: flag validation + module construction
    extractor_api.py       `ExtractorApi`     — the `last_*` stash reads and the three setters
    extractor_forward.py   `ExtractorForward` — the T0/T1 stack and `forward_internal`
    features_extractor.py  `Gen3FeaturesExtractor` — the class, and `forward`

`forward` lives HERE, on the concrete class, deliberately: both compile flags patch the BOUND
`fe.forward`, `cf_terms` calls `type(fe).forward` for its always-eager pass, and
`instrumented_ppo_test` ASSIGNS `type(fe).forward` — an attribute defined on a base would be
shadowed by that assignment and never restored to where it came from.

The eight PHASE modules that came out on 2026-08-16 (`extractor_ctx` / `encoders` /
`team_transformer` / `pools` / `belief_heads` / `aux_value_heads` / `pointer_head` /
`value_readouts`) are re-imported below explicitly, for the same hub reason.
"""
import contextlib

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
from agents.model.arch_constants import (
    INTENT_THRESH_MOVE_DIM, INTENT_COND_MOVE_DIM,
    INTENT_MOVE_CELL_DIM, _INTENT_MOVE_CELL_RAW,
    PAIR_OUTCOME_MOVE_DIM, PAIR_OUTCOME_SWITCH_DIM, SWITCH_BRANCH_MOVE_DIM,
    CONDITIONAL_THREAT_SWITCH_DIM, PAIR_VALUE_ROUTE_DIM,
    UVR_K, UVR_DIM, _UVR_N_SOURCES, _UVR_N_SOURCES_FULL,
      # noqa: F401  (re-export
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
    CfEvidentialHead, ShadowValueHead, ValueDistHead, WinProbHead,
)
from agents.model.pointer_head import (  # noqa: F401
    EntityMoveSeats, PointerNativeActionHead, _request_order_move_tokens,
)
from agents.model.q_winprob_head import (  # noqa: F401
    Q_WINPROB_MODES, QWinProbHead,
)
from agents.model.value_readouts import UnifiedValueReadout  # noqa: F401



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
from agents.model.intent_move_cell import IntentMoveCell
from agents.model.intent_threshold import (
    IntentThresholdMoveCell, ThresholdProbs, threshold_probs)
from agents.model.intent_conditional import IntentConditionalMoveCell
from agents.model.pair_outcome import (
    PairOutcomeMoveCell, PairOutcomeSwitchCell, pair_alpha, reduce_pair_in, reduce_pair_in_all)
from agents.model.switch_branch import SwitchBranchMoveCell
from agents.model.conditional_threat import ConditionalThreatCell
from agents.model.pair_value_route import PairValueInject  # noqa: F401  (re-export)
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

from agents.model.extractor_stashes import ExtractorStashes  # noqa: F401
from agents.model.projection import (  # noqa: F401
    ProjectionAssembler, compute_projection_widths,
)
from agents.model.extractor_build import ExtractorBuild  # noqa: F401
from agents.model.extractor_api import ExtractorApi  # noqa: F401
from agents.model.extractor_forward import ExtractorForward, _OUT_SEC_FLINCH_COL  # noqa: F401


class Gen3FeaturesExtractor(ExtractorForward):
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
