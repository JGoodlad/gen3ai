"""The differentiable gen3 DAMAGE OPERATOR — extracted from `features_extractor.py` 2026-08-01.

A pure RELOCATION: same class, same constants, same forward math, byte-for-byte. It was 1,689 of the
extractor's ~4,700 lines (plus its constants) — 39% of the file for one concern — and it depends on
the extractor only through `ctx: 'ExtractorContext'`, which is a STRING forward-reference and so
costs no runtime import. Hence no cycle.

Every public name is re-exported by `features_extractor` so historical import paths
(`from agents.model.features_extractor import DamageOperator, decode_damage_block, _DMG_*`) keep
working — the prober, model_version, snapshot and the tests all use them.

Verified by `tmp/damage_op_equiv_probe.py` (pi/vf/op-block bit-identical) + the full unit suite +
`damage_op_probe_fuzz_test.py` (the constructed-scenario physics oracle) — the only acceptable gate
for a refactor that claims to change nothing.
"""
import torch
from torch.utils.checkpoint import checkpoint
import numpy as np
from dataclasses import dataclass, replace
from gymnasium import spaces
from typing import Callable, Dict, Any, Optional, Sequence, Tuple
from agents.observation.constants import (
    TRACE_INTERVAL,
    TEAM_SIZE,
    GLOBAL_ENV_DIM,
    POKEMON_FULL_DIM,
    POKEMON_HP_PROBS_OFFSET,
    POKEMON_SPECIES_KNOWN_OFFSET,
    POKEMON_SPREAD_OFFSET,
    POKEMON_SPREAD_DIM,
    POKEMON_CONDITION_OFFSET,
    POKEMON_SLEEP_BELIEF_OFFSET,
)
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
from agents.model.team_signature import TEAM_SIGNATURE_DIM, TEAM_SIGNATURE_MOVES
from agents.model.damage_tables import N_SECONDARY as _N_SECONDARY, SECONDARY_COLS as _SECONDARY_COLS
from agents.observation.turn_delta_encoder import (
    TURN_DELTA_DIM,
    EFF_DIM,
    ORDER_DIM,
    TURN_DELTA_EMBEDDED_IDS,
    TURN_DELTA_SCALAR_OFFSETS,
)
from agents.action.constants import ACTION_SPACE_SIZE
from utils.logging.levels import LogLevel
from agents.model.arch_constants import (  # noqa: F401  (re-export)
    ROLE_TOKEN_SIZE,
    PROJECTION_DIM,
    MOVE_NET_HIDDEN,
    MOVE_LATENT_HIDDEN,
    MOVE_LATENT_DIM,
    ROLE_ENCODER_HIDDEN,
    ACTIVE_CTX_HIDDEN,
    NET_ARCH,
    N_HISTORY_TURNS,
    ZARCH_DIM,
    ZARCH_ATOM_HIDDEN,
    D_MODEL,
    TRANSFORMER_N_LAYERS,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_FFN_DIM,
)



_DMG_CHANNEL_FEATS = 5          # [low_roll, high_roll, crit, pko, accuracy] per gen3 type-category channel
_DMG_N_CHANNELS = 2             # physical / special (the gen3 TYPE split)
_DMG_PER_MON = _DMG_N_CHANNELS * _DMG_CHANNEL_FEATS + 2   # + p_outspeed + provenance = 12
# Named per-defender feature offsets (the single source of truth for the op's output slot layout — the
# prober decode + outgoing/safe-switch directions index by these, never a literal).
_DMG_IDX_PHYS_LOW, _DMG_IDX_PHYS_HIGH, _DMG_IDX_PHYS_CRIT, _DMG_IDX_PHYS_PKO, _DMG_IDX_PHYS_ACC = 0, 1, 2, 3, 4
_DMG_IDX_SPEC_LOW, _DMG_IDX_SPEC_HIGH, _DMG_IDX_SPEC_CRIT, _DMG_IDX_SPEC_PKO, _DMG_IDX_SPEC_ACC = 5, 6, 7, 8, 9
_DMG_IDX_OUTSPEED, _DMG_IDX_PROVENANCE = 10, 11
_DMG_ROLL_MIN = 0.85            # lowest of the 16 gen3 damage rolls ((85..100)/100); high roll = 1.0
_DMG_CHIP_CAP = 1.5             # clamp on the roll fractions (a 4× STAB hit otherwise fattens the tail)
_DMG_CRIT_CAP = 3.0            # crit can ×2 a capped high roll → a wider cap
# Opp-active believed-effect threat scalars (belief-weighted MAX over the move belief — see the
# aggregation note in forward; a full-axis noisy-OR over ~400 moves saturated to ~1 from the floor alone),
# order == damage_tables MOVE_EFFECT_COLS: [recovery, status, phaze, boost, hazard, protect]. The
# status/utility axis the damage-only CPU block never had.
_DMG_EFFECT = 6
# gen3_unified_move_system_v1: per-status SECONDARY-effect threat the opp active poses (its damaging
# moves' secondaries — Body Slam para, Rock Slide flinch, Ice Beam freeze), belief-weighted + accuracy-
# folded + ×Serene Grace. 10 scalars, order == damage_tables.SECONDARY_COLS. Appended AFTER the 6 effect
# scalars (the existing per-mon/effect layout is untouched). NO speed coupling — flinch's move-first
# dependence is left to attention (owner decision).
_DMG_INCOMING_SEC = _N_SECONDARY            # 10
# gen3_unified_choice_band_v1: the CB-CONDITIONAL physical tail of the INCOMING threat — per our 6 mons, the
# opp's PHYSICAL [high-roll, P(OHKO)] computed WITH the ×1.5 Choice-Band Atk, then ONE shared `p_cb` scalar
# (P(opp active holds CB)). DECORRELATED from the modal (no-CB) line + p_cb so the head weights them itself
# (OHKO is a nonlinear threshold a mean-field blend would blur — same rationale as the crit-split). The op's
# OUTGOING block separately applies the ×1.5 deterministically (our own item is known). Order:
# [phys_high_cb × 6, phys_pko_cb × 6, p_cb].
_DMG_CB_PER_MON = 2                          # phys_high_cb, phys_pko_cb (physical channel only — CB is phys)
_DMG_CB = _DMG_CB_PER_MON * TEAM_SIZE + 1    # + the shared p_cb scalar = 13
_DMG_CRIT_P = 1.0 / 16.0   # gen3 base crit rate (×2 damage) — the crit roll is exposed as crit_frac
_DMG_SPEED_SCALE = 15.0    # logistic scale for P(outspeed) over the speed-stat difference (~one stage)
_DMG_SPEED_STD_K = 1.702   # gen3_bidir_threat_trunk_v1 (#3): sigmoid≈normal-CDF, so the uncertainty-aware
#                            P(outspeed) divides the speed gap by (believed speed std / 1.702)
# gen3_unified_spread_belief_v1: indices into the SpreadBelief's [atk,def,spa,spd,spe] output (== the
# damage_tables SPREAD_STAT_COLS order). When a spread belief is passed, the op consumes these believed opp
# stat VALUES in place of its hand-coded de-timid/neutral-0-EV constants.
_SB_ATK, _SB_DEF, _SB_SPA, _SB_SPD, _SB_SPE = 0, 1, 2, 3, 4
# gen3_unified_op_physics_v1: the active mons' stat-STAGE boosts (DD/CM/Intimidate…) are the worst
# damage-calc edge case (a +2 sweeper's Atk is doubled). The boost stage is read from the active-context
# block (boosts(14) = 7 stats × 2 [pos/6, neg/6]: atk,def,spa,spd,spe,acc,eva), and the gen3 multiplier
# applied to offense/defense/speed. Mirrors incoming_damage.boost_mult. _PARA_SPEED quarters Speed.
_DMG_PARA_SPEED = 0.25
# Burn (½ physical Atk) + paralysis (×0.25 Speed) read the per-mon condition one-hot (None,BRN,PAR,SLP,…)
# at POKEMON_CONDITION_OFFSET; weather (rain ×1.5 Water/×0.5 Fire; sun the reverse) reads the Water/Fire
# type indices on the TypeEncoder axis (the same axis MOVE_TYPE_IDX rides) + ctx.weather_feature.
_COND_BRN_IDX, _COND_PAR_IDX = 1, 2
from agents.observation.types import TypeEncoder as _TypeEncoder
_WATER_TIDX, _FIRE_TIDX = _TypeEncoder.TYPE_TO_IDX["WATER"], _TypeEncoder.TYPE_TO_IDX["FIRE"]

# OUTGOING direction (our active → opp active): per OUR move, in REQUEST-slot order (== action logits
# 6+k) so the policy head can compare move A vs B directly — the equal-effectiveness tie-break (Earthquake
# vs Meteor Mash into a Rock: same 2× multiplier, different resolved damage). Per move [low, high, crit,
# pko] (accuracy is NOT repeated — our moves' accuracy already rides the action-aligned obs move-block;
# pko still folds it), then one p_outspeed (our active vs opp active), then (gen3_unified_move_system_v1)
# the per-move SECONDARY-effect probabilities — "what status can OUR move cause, with what probability"
# (Thunderbolt 10%/20% para under Serene Grace, ×opp Shield Dust), 4 moves × 10 cols appended LAST.
_DMG_OUT_PER_MOVE = 4
_DMG_OUT_N_MOVES = 4
_DMG_OUT_SEC = _DMG_OUT_N_MOVES * _N_SECONDARY            # 4 moves × 10 secondary probs = 40
_DMG_OUTGOING = _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE + 1 + _DMG_OUT_SEC   # 16 + p_outspeed + 40 = 57

# gen3_unified_status_landing_v1: the OUTGOING per-OUR-move "will my STATUS move land vs THIS opponent"
# block — the GPU replacement for the masked move-effect block's `status_will_land`. Per move (request-slot
# order == action 6+k): P(the status applies) + a `known` bit (the value rests on a CERTAIN block — type
# immunity / already-statused / Sleep-Clause / Substitute / a revealed ability — vs a Smogon-prior estimate).
# Folds accuracy × type immunity × ability immunity (revealed-or-prior) × already-statused × Sleep Clause ×
# Substitute; adds Leech Seed (Grass-immune). NOTE the delta vs the masked CPU block it replaces: it now FOLDS
# base accuracy (Toxic 0.85 / WoW 0.75 — the CPU returned 1−P(ability blocks), accuracy-free), so the value is
# more correct but value-meaning-different on the A/B. UNCOVERED residual: Yawn (delayed sleep — no
# status_inflicted) and a Leech-Seed-already-seeded target (no leechseed volatile read). Shield Dust is NOT
# relevant here (it only scales SECONDARY effects, never a primary status move — see the incoming-secondary
# block). Appended to the outgoing direction (gated on `damage_outgoing`).
_DMG_STATUS_N_MOVES = _DMG_OUT_N_MOVES
_DMG_STATUS = 2 * _DMG_STATUS_N_MOVES                     # 4 P(lands) + 4 known = 8
_COND_SLP_IDX = 3                                         # condition one-hot [None,BRN,PAR,SLP,FRZ,PSN,TOX]
# Substitute's index into the active-context block (our_ctx_raw / opp_ctx_raw = boosts ++ volatiles), DERIVED
# from the obs layout (never hardcoded). A Substitute blocks EVERY status move (incl. Leech Seed) in gen3.
from agents.observation.gen3_effects import VOLATILE_SLOTS as _VOLATILE_SLOTS
from agents.observation.constants import BOOSTS_DIM as _BOOSTS_DIM
_SUBSTITUTE_CTX_IDX = _BOOSTS_DIM + list(_VOLATILE_SLOTS).index("substitute")

# gen3_per_move_matrices_v1 (v32): the OUTGOING per-move DAMAGE MATRIX — our active's 4 moves × the opp's
# 6 mons (active + REVEALED bench). The legacy `_outgoing_block` prices our moves vs the opp ACTIVE only;
# this surfaces "what each of my moves does to each opp mon it could face" so the policy can price a KO on
# a SWITCH-IN (the equal-effectiveness tie-break extended to bench targets). REVEALED-gated: an unrevealed
# opp slot (Gen3 has no team preview) is zeroed — guessing damage off an unknown species is a TODO
# (belief-driven). Grouped by MOVE (action-aligned, request-slot order == action 6+k): per (our move k,
# opp mon d) cell `[low, high, crit, pko, type_mult]`, then a per-opp-mon `revealed` bit (so a 0 column =
# "unrevealed/absent" is distinguishable from "0 damage"). Reuses the `_outgoing_block` physics, broadcast
# over the 6 opp defenders. Design: designs/ai_v6/design_per_move_damage_matrices.md.
_DMG_OMX_CELL = 5                                  # [low, high, crit, pko, type_mult] per (our move, opp mon)
_DMG_OMX_IDX_LOW, _DMG_OMX_IDX_HIGH, _DMG_OMX_IDX_CRIT, _DMG_OMX_IDX_PKO, _DMG_OMX_IDX_MULT = 0, 1, 2, 3, 4
_DMG_OMX = _DMG_OUT_N_MOVES * TEAM_SIZE * _DMG_OMX_CELL + TEAM_SIZE   # 4×6×5 + 6 revealed bits = 126

# gen3_per_move_matrices_v1 (v39): the TRANSPOSED outgoing matrix — our 6 MONS' 4 moves → the opp ACTIVE.
# `_outgoing_block` (and its bench-defender extension `_outgoing_matrix`) price ONLY our CURRENT active as the
# attacker, so on a FORCED SWITCH (our active fainted → `_outgoing_block` zeroes) the policy picks switch-ins
# BLIND to offense. This is the TRANSPOSE of `_outgoing_matrix` (whose DEFENDER axis is the opp's 6 mons): here
# the ATTACKER axis is OUR 6 mons (active + bench), each priced vs the opp ACTIVE only, so the policy knows
# what every candidate switch-in would DO. Per (attacker mon, move) cell `[low, high, crit, pko]` (PARITY with
# `_outgoing_block`'s per-move stack), then a per-attacker `p_outspeed` (speed is per-mon) + an `alive` bit (so
# a fainted/absent attacker reads a distinguishable 0-column, mirroring `_outgoing_matrix`'s `revealed` bit).
# The ACTIVE row reproduces `_outgoing_block` byte-for-byte (its boosts/CB/burn + request-ordered moves); bench
# rows reuse the SAME `_rolls` physics with NEUTRAL boosts (gen3 resets boosts on switch) + the per-mon
# sorted-by-id moves (bench mons have no request order). Layout = ALL cells, then the trailing scalar blocks
# (`p_outspeed[6]` ++ `alive[6]`) — like `_outgoing_matrix`'s `cell.reshape || def_gate`.
_DMG_OAX_PER_MOVE = _DMG_OUT_PER_MOVE              # 4: [low, high, crit, pko]
_DMG_OAX_N_MOVES = _DMG_OUT_N_MOVES               # 4
_DMG_OAX_IDX_LOW, _DMG_OAX_IDX_HIGH, _DMG_OAX_IDX_CRIT, _DMG_OAX_IDX_PKO = 0, 1, 2, 3
# per attacker: 4 moves × [low,high,crit,pko] = 16; the trailing p_outspeed(6) + alive(6) are GLOBAL blocks.
_DMG_OAX_PER_MON = _DMG_OAX_N_MOVES * _DMG_OAX_PER_MOVE                # 16 (the cell block per attacker)
_DMG_OAX = TEAM_SIZE * _DMG_OAX_PER_MON + 2 * TEAM_SIZE               # 6×16 + p_outspeed[6] + alive[6] = 108

# gen3_pointer_native_v1: the per-ACTION cell widths `DamageOperator.pointer_cells` slices out of the flat
# damage block for the pointer action head — the LOSSLESS per-action physics route (move cell k feeds move
# logit 6+k, switch cell j feeds switch logit j; the flat concat delivers the same numbers only post-pool).
# MOVE cell (request-slot k, only when `outgoing`): the `_outgoing_block` damage stack [low,high,crit,pko]
# + the `_status_landing` pair [p_land, known] + the 10 per-move secondary chances = 16.
_PTR_MOVE_CELL = _DMG_OUT_PER_MOVE + 2 + _N_SECONDARY                  # 4 + 2 + 10 = 16
# SWITCH cell (defender/candidate mon j, always when the op is on): the per-mon incoming row (12) + its
# CB-conditional pair [phys_high_cb_j, pko_cb_j] + the shared p_cb (broadcast — the head needs it NEXT TO
# the CB tail it conditions) = 15; + the OAX attacker row (16 cells + p_outspeed_j + alive_j = 18) when
# `matrices_outgoing_all` (the switch-in OFFENSE read).
_PTR_SWITCH_CELL_IN = _DMG_PER_MON + _DMG_CB_PER_MON + 1               # 12 + 2 + 1 = 15
_PTR_SWITCH_CELL_OAX = _DMG_OAX_PER_MON + 2                            # 16 + p_outspeed + alive = 18

# gen3_per_move_matrices_v1 (v33): the INCOMING per-move DAMAGE MATRIX — the ENRICHED evolution of the v30
# top-K block (it SUPERSEDES it; the two don't coexist). For the opp active's top-K most-believed moves: a
# richer per-move HEADER [latent(32), belief, accuracy, is_phys, EXPLICIT effect bits (6: recovery/status/
# phaze/boost/hazard/protect), EXPLICIT per-status secondary chances (10)] — the per-move utility/secondary
# the worst-case p_effect/p_sec opp-active collapse couldn't give (the owner's "phaze/heal/flinch are how
# mid-ladder players reason" signals, un-collapsed) — and a richer per-(move, OUR mon) CELL
# [low, high, crit, pko, type_mult, status_lands] (vs the top-K's [high, pko, status_lands]). Requires
# move_latent (the latent); REUSES damage_topk_k as its K and replaces the lean top-K. Design: design_per_move_damage_matrices.md.
_DMG_IMX_HEADER = MOVE_LATENT_DIM + 3 + _DMG_EFFECT + _N_SECONDARY   # 32 + [belief,acc,is_phys] + 6 + 10 = 51
_DMG_IMX_HDR_W = MOVE_LATENT_DIM                    # belief offset within the header (after the latent)
_DMG_IMX_HDR_ACC = MOVE_LATENT_DIM + 1
_DMG_IMX_HDR_PHYS = MOVE_LATENT_DIM + 2
_DMG_IMX_HDR_EFFECT = MOVE_LATENT_DIM + 3           # 6 effect bits
_DMG_IMX_HDR_SEC = MOVE_LATENT_DIM + 3 + _DMG_EFFECT  # 10 secondary chances
_DMG_IMX_CELL = 6                                   # [low, high, crit, pko, type_mult, status_lands]
(_DMG_IMX_IDX_LOW, _DMG_IMX_IDX_HIGH, _DMG_IMX_IDX_CRIT,
 _DMG_IMX_IDX_PKO, _DMG_IMX_IDX_MULT, _DMG_IMX_IDX_STATUS) = 0, 1, 2, 3, 4, 5


def _dmg_imx_dim(k: int) -> int:
    """Total INCOMING-matrix width for K: the per-move header (K × `_DMG_IMX_HEADER`, shared across our mons)
    ++ the per-(our-mon, move) cell block (`TEAM_SIZE` × k × `_DMG_IMX_CELL`)."""
    return k * _DMG_IMX_HEADER + TEAM_SIZE * k * _DMG_IMX_CELL

# gen3_unified_topk_incoming_v1: the DISCRETE top-K incoming move-space block. The incoming damage op
# collapses the opp active's whole moveset into the worst phys/spec hit per defender (`_chan_max`) — losing
# WHICH move it is + the per-pivot consequences, so the policy can't anticipate the discrete move or pick the
# immune/safe pivot. This block surfaces the opp active's K most-believed CANDIDATE moves individually, each
# carrying its move LATENT identity (gathered from the MoveLatentEncoder — differentiable → sharpens the
# latent) + belief weight (differentiable → sharpens the move belief) + per-OUR-mon damage + a per-pivot
# status-landing scalar (immunity-folded — the Thunder-Wave→Ground safe-switch read). Added ALONGSIDE the
# `_chan_max` worst-case summary (the hybrid the differentiable-op design §4.3 always intended). K is a
# per-model int `damage_topk_k` (0 = off) so `out_dim` scales with it (STRUCTURAL, like `opp_belief_cls_k`).
# Requires `move_latent` (for the latent gather) + `damage_op`. Design: designs/ai_v6/design_topk_incoming_moves.md.
_DMG_TOPK_DEFAULT_K = 5         # default K when enabled (reason about the 4th/5th move = expert-level)
_DMG_TOPK_ID_DIM = MOVE_LATENT_DIM          # 32 — the move-identity latent
_DMG_TOPK_META = 3                          # [belief_w, accuracy, is_phys] (opp-property, per move)
_DMG_TOPK_MOVE = _DMG_TOPK_ID_DIM + _DMG_TOPK_META          # 35 — opp-property feats per top-K move (shared)
_DMG_TOPK_DMG_PER = 3                        # per (our defender, top-K move): [high, pko, status_lands]
# intra-move opp-property offsets
_DMG_TOPK_IDX_LATENT = 0
_DMG_TOPK_IDX_W = _DMG_TOPK_ID_DIM           # 32
_DMG_TOPK_IDX_ACC = _DMG_TOPK_ID_DIM + 1     # 33
_DMG_TOPK_IDX_PHYS = _DMG_TOPK_ID_DIM + 2    # 34
# per-(defender, move) offsets
_DMG_TOPK_IDX_HIGH, _DMG_TOPK_IDX_PKO, _DMG_TOPK_IDX_STATUS = 0, 1, 2
# Map the 6 MAJOR-status secondary columns (par,brn,frz,slp,psn,tox) → the ABILITY_STATUS_BLOCK / status
# category axis (par→1, brn→2, frz→3, slp→4, psn→5, tox→5), for the per-pivot incoming status-landing's
# ability-immunity fold (Limber blocks Body Slam's para, etc.). SECONDARY_COLS order, first 6 cols.
_SECONDARY_MAJOR_N = 6
_SECONDARY_TO_STATUS_CAT = (1, 2, 3, 4, 5, 5)


def _dmg_topk_dim(k: int) -> int:
    """Total top-K block width for a given K: the opp-property block (K × `_DMG_TOPK_MOVE`, shared across
    defenders) ++ the per-(defender, move) block (`TEAM_SIZE` × k × `_DMG_TOPK_DMG_PER`). = K·53."""
    return k * _DMG_TOPK_MOVE + TEAM_SIZE * k * _DMG_TOPK_DMG_PER


# gen3_iterative_damage_v1: the ITERATIVE damage-refinement primitive. The full op runs ONCE post-transformer
# (a one-shot read of the FINAL belief). This recomputes a LEAN per-our-mon incoming-damage summary BETWEEN
# transformer layers — as the opp token (hence the move belief read from it) is enriched by attention — and
# injects it back onto our-mon tokens, so each layer attends over physics derived from the CURRENT belief
# (physics-in-the-loop, not one-shot). `_DMG_REFINE_FEATS` = the per-defender summary the refinement injects:
# `[phys_high, spec_high, phys_pko, spec_pko]` — the worst-case damage magnitude + accuracy-folded P(KO) per
# gen3 type channel (decorrelated: damage magnitude vs KO probability). `_DMG_REFINE_K` = the candidates the
# lean kernel scores per round (the opp active's top-K most-believed moves — ~50× cheaper than the full
# ~416-candidate axis, so the per-round recompute is cheap; the heavy full sweep still runs once in `forward`).
_DMG_REFINE_FEATS = 4
_DMG_REFINE_K = 8

# gen3_bidir_threat_trunk_v1: the OUTGOING half of the in-trunk threat refine — the SYMMETRIC mirror of the
# incoming refine. `discrete_outgoing` computes how hard OUR active's 4 known moves hit each of the opp's 6
# mons and injects the per-opp-mon summary onto the OPP token slice (so attention reasons over "how
# threatened is each opp mon by us", not just "how threatened are we"). `_DMG_OUT_REFINE` = the same 4-feat
# `[phys_high, spec_high, phys_pko, spec_pko]` per-opp-mon summary. REVEALED opp mons use real types/bulk +
# full P(KO); UNREVEALED mons use the EXPECTED-LATENT read (E[mult] via SPECIES_EXP_MULT, E[bulk]/E[maxhp]
# via SPECIES_SPREAD_PRIOR / E[base_hp], marginalized over the move-belief's P(species)) with P(KO) NULLED
# (a full-HP switch-in is ~never OHKO'd — owner decision, drops the Jensen-threshold complexity).
_DMG_OUT_REFINE = 4

# gen3_status_trunk_v1: STATUS-LANDING into the trunk (the last CPU-obs deprecation gap). Status immunity
# (type × ability × already-statused × Sleep-Clause × Substitute) is a deterministic MECHANICS fact — the
# same class as type effectiveness, which we COMPUTE — and learning it would force attention to correlate
# non-local info (the move's status intent on one token, the defender's types+ability on another). So we
# COMPUTE it and inject a per-defender summary into the trunk, both directions. `_DMG_STATUS_REFINE` = the
# 2-scalar per-defender summary `[p_major, p_immobilize]`: p_major = P(any major status lands), p_immobilize
# = P(an ACTION-DENYING status lands = paralysis/freeze/sleep). The major-vs-immobilize split makes the
# trunk signal SELF-CONTAINED ("I'll be immobilized" vs "I'll be chipped") so the policy needn't cross-
# reference which move. Dedicated-move path only (damaging-move secondaries ride the heads' secondary block).
_DMG_STATUS_REFINE = 2
_IMMOBILIZE_STATUS_CATS = (1, 3, 4)   # MOVE_STATUS_CAT values for paralysis / freeze / sleep (action-denying)


class DamageOperator(torch.nn.Module):
    """Fixed, differentiable gen3 damage calculator run in the GPU forward pass, fed by the
    move-belief head's PREDICTED moves — the "compute the physics, learn the belief" op
    (`designs/ai_v6/design_differentiable_damage_op.md`).

    For the opponent ACTIVE mon (always a revealed mon under move-belief mode revealed/both), it
    computes the incoming damage its *believed* moveset would deal to each of our 6 mons, then
    aggregates per (defender, gen3-type-channel) into the believed-move threat. The damage is a
    differentiable function of the move-belief weights `w_m = sigmoid(last_move_belief_logits[active])`,
    so the gradient sharpens the belief toward the moves that actually threaten KOs. This replaces
    the FIXED usage-prior the CPU `incoming_damage.py` block must use (the belief doesn't exist at
    obs-build time) with the model's LEARNED belief.

    Per defender d, `_DMG_PER_MON` (12) features: per channel (physical / special — the gen3 TYPE split) the
    3-roll `[low, high, crit]` + accuracy-folded `pko` + `acc` (10 = 5×2), plus P(outspeed) and the threat
    provenance. Aggregation is a HARD max over the
    channel's believed candidates (= `incoming_damage`'s max-over-candidates; differentiable via the
    argmax subgradient — the dominant move's belief weight gets gradient — without the candidate-count
    dilution a low-temperature soft-max would suffer over ~400 moves). Plus `_DMG_EFFECT` opp-active
    believed-EFFECT scalars (belief-weighted MAX of the belief × per-move status/utility flags: recovery,
    status, phaze, boost, hazard, protect) — the status-threat axis the damage-only CPU block never had.
    Plus (gen3_unified_move_system_v1) `_DMG_INCOMING_SEC` per-STATUS secondary scalars — the opp active's
    DAMAGING-move secondaries (Body Slam para, Rock Slide flinch, Ice Beam freeze): realized
    `max_m(w_m·chance·acc) × Serene Grace(opp)`, accuracy folded, NO speed coupling (flinch's move-first
    dependence is left to attention). When `outgoing`, each of OUR 4 moves ALSO carries its secondary
    probabilities (`chance·acc × Serene Grace(us) × Shield Dust(opp)`) — "what status can this move cause,
    with what probability". Order == damage_tables.SECONDARY_COLS.
    Hidden Power needs NO special handling here (gen3_typed_hp_belief_v1): the move-belief posterior
    arrives already composed into the 16 typed move-nums 355-370, each a real BP-70 typed row, so HP-Ice
    and HP-Grass are priced as ordinary distinct moves with their own effectiveness. The only HP-aware
    line left in the op is masking the bare 237 presence channel out of the candidate set.

    Stats: our defenders use their REAL spread (IVs/EVs/nature reconstructed from the obs spread block —
    they are revealed); the hidden-spread attacker uses a fixed de-timid offensive assumption (252 EV,
    31 IV, +nature ×1.1), mirroring `incoming_damage`'s offensive-stat tail. The smooth (un-floored)
    L100 stat + damage formula keeps everything differentiable (the byte-exact floored kernel is the
    proof's; the forward only needs the gradient).

    When `outgoing`, the op ALSO appends the per-OUR-move outgoing damage block AND the
    gen3_unified_status_landing_v1 STATUS-LANDING block (per move: P(a dedicated status move lands vs the opp
    active — type/ability/already-statused/Sleep-Clause/Substitute folded, Leech Seed Grass-immune) + a
    `known` bit) — the GPU home for the masked move-effect `status_will_land`.

    Leak-safe: reads only the PREDICTED belief + public obs (our HP/types, the opp active's revealed
    species/types/condition/sub) — never a privileged label. Output `[B, self.out_dim]` (= incoming +,
    when outgoing, the damage + status-landing blocks) is appended to BOTH projection heads. Zeroed (incl.
    gradient) when there is no opponent active and per fainted defender.
    Lookup tables are registered as non-persistent float32 buffers (pure physics, recomputable from
    `data/`)."""

    per_mon = _DMG_PER_MON
    n_effect = _DMG_EFFECT
    n_incoming_sec = _DMG_INCOMING_SEC
    incoming_dim = TEAM_SIZE * _DMG_PER_MON + _DMG_EFFECT + _DMG_INCOMING_SEC + _DMG_CB

    def __init__(self, layout: Dict[str, Any], outgoing: bool = False, topk_k: int = 0,
                 matrices_outgoing: bool = False, matrices_incoming: bool = False,
                 matrices_outgoing_all: bool = False,
                 prob_outspeed: bool = False,
                 candidate_k: int = 0):
        super().__init__()
        # gen3_topk_candidates_v1: cap the incoming candidate sweep at the K most-believed opponent
        # moves (0 = the full ~400-wide sweep, byte-identical). No tail-risk bound — the truncated
        # mass is simply dropped, which is the tradeoff under test.
        self.damage_candidate_k = int(candidate_k)
        # gen3_bidir_threat_trunk_v1 (#3): use a SOFT P(our_spe > opp_spe) over the believed speed mean±std
        # (SPECIES_SPREAD_PRIOR) instead of the hard point-estimate comparison. Forward-behavior toggle (no
        # new params; values only). Stored for the forward / _outgoing_block p_outspeed computation.
        self.prob_outspeed = bool(prob_outspeed)
        from agents.model.damage_tables import (
            build_damage_buffers, HIDDEN_POWER_NUM, HIDDEN_POWER_BP,
            CHOICE_BAND_ITEM_NUM, CHOICE_BAND_PHYS_MULT,
        )
        bufs = build_damage_buffers(layout['max_moves'], layout['max_species'], layout['max_abilities'])
        for name, tensor in bufs.items():
            # Non-persistent: deterministic physics from data/, not learned weights → keep them out of
            # every checkpoint (and out of the state_dict, so a load never demands them).
            self.register_buffer(name, tensor, persistent=False)
        self.hp_num = HIDDEN_POWER_NUM
        self.hp_bp = float(HIDDEN_POWER_BP)
        # gen3_typed_hp_belief_v1: HP reaches the op as 16 ORDINARY typed-move candidates (nums
        # HP_TYPED_NUMS = 355-370, real BP/type in the buffers) — the presence×type composition happens
        # UPSTREAM in `HPTypeBelief.compose_typed_hp`, so the op holds no HP-type source of its own (the
        # old `hp_type_fix` / `SPECIES_HP_PRIOR` pair is gone, along with the per-call-site divergence it
        # allowed). `HP_CAND_MASK` (a non-persistent buffer) now zeros only the bare typeless 237, the
        # BP-0 presence channel that is never a damage candidate.
        self.cb_item_num = CHOICE_BAND_ITEM_NUM            # gen3_unified_choice_band_v1: Choice Band item num
        self.cb_phys_mult = float(CHOICE_BAND_PHYS_MULT)   # ×1.5 physical Atk
        # gen3_unified_topk_incoming_v1: secondary-col → status-category map for the per-pivot incoming
        # status-landing's ability-immunity fold (non-persistent — pure constant).
        self.register_buffer("_SEC_CAT_IDX", torch.tensor(_SECONDARY_TO_STATUS_CAT, dtype=torch.long),
                             persistent=False)
        # PADDED SPARSE INDEX for the two opp-active belief×flag maxes in `forward` (`p_effect` / `p_sec`).
        # Both are `max_m (w_m · table_mk)` over ALL ~400 candidate moves, but the tables are extremely
        # sparse (≤21 of 400 nonzero rows per effect column, ≤19 per secondary column), so the dense form
        # materialised a [B,400,6] / [B,400,10] tensor to produce 6 / 10 numbers. Gathering only the
        # nonzero rows is EXACT — `w ≥ 0` (sigmoid) and `MOVE_ACCURACY ≥ 0`, so every dropped term is a
        # product with table value 0, which can never be the max unless the column is entirely zero, and
        # the pad lane (a zeroed value slot) reproduces exactly that 0. Same op count, ~19× less traffic.
        for _name, _table in (("EFF", self.MOVE_EFFECT_FLAGS), ("SEC", self.MOVE_SECONDARY)):
            _cols = _table.shape[1]
            _width = max(1, int((_table > 0).sum(dim=0).max().item()))
            _idx = torch.zeros(_cols, _width, dtype=torch.long)
            _val = torch.zeros(_cols, _width, dtype=_table.dtype)
            for _k in range(_cols):
                _rows = (_table[:, _k] > 0).nonzero(as_tuple=True)[0]
                _idx[_k, :_rows.numel()] = _rows
                _val[_k, :_rows.numel()] = _table[_rows, _k]
            self.register_buffer(f"_{_name}_IDX", _idx, persistent=False)      # [cols, width]
            self.register_buffer(f"_{_name}_VAL", _val, persistent=False)      # [cols, width] (0 = pad)
        # OUTGOING direction (our active → opp active, per-move action-aligned): off by default. When on,
        # the op ALSO emits the _DMG_OUTGOING block (widens out_dim → both projections auto-size).
        self.outgoing = outgoing
        # gen3_unified_topk_incoming_v1: the discrete top-K incoming block (0 = off). When >0 the op ALSO
        # emits the `_dmg_topk_dim(topk_k)` block (widens out_dim → both projections auto-size). Requires the
        # caller to pass `move_latent_all` to forward (enforced at the extractor: needs --move-latent).
        self.topk_k = topk_k
        # gen3_per_move_matrices_v1: the OUTGOING per-move DAMAGE MATRIX (our 4 moves × opp active+revealed
        # bench). Off by default; when on the op ALSO emits the `_DMG_OMX` block (widens out_dim → both
        # projections auto-size). Requires the op's physics buffers (always present). The legacy single-active
        # `_outgoing_block` (`outgoing`) is a SUBSET — running the matrix supersedes it (a run uses one).
        self.matrices_outgoing = matrices_outgoing
        # gen3_per_move_matrices_v1: the INCOMING per-move DAMAGE MATRIX = the ENRICHED top-K. It REUSES the
        # shared `topk_k` as its K (so ONE knob — `--damage-topk K` — tunes both the lean top-K and the rich
        # matrix: try 4/5/6) and REPLACES the lean top-K block at that K (so the lean block is suppressed —
        # never both). Defaults K to _DMG_TOPK_DEFAULT_K (5) if topk_k unset. Requires move_latent (the latent
        # gather), enforced at the extractor.
        self.matrices_incoming = matrices_incoming
        self.matrices_incoming_k = (topk_k if topk_k > 0 else _DMG_TOPK_DEFAULT_K) if matrices_incoming else 0
        # gen3_per_move_matrices_v1 (v39): the TRANSPOSED outgoing matrix — our 6 mons' moves → opp active (the
        # switch-in offense read). Off by default; when on the op ALSO emits the `_DMG_OAX` block (widens
        # out_dim → both projections auto-size). Appended LAST (after the v34 outgoing matrix). Requires the
        # op's physics buffers (always present). The legacy single-active `_outgoing_block` is the ACTIVE row.
        self.matrices_outgoing_all = matrices_outgoing_all
        # The lean top-K block is emitted ONLY when topk_k>0 AND the rich matrix is OFF (the matrix replaces it).
        _lean_topk = topk_k if (topk_k > 0 and not matrices_incoming) else 0
        # The OUTGOING direction carries the per-move damage block + the gen3_unified_status_landing_v1
        # status-landing block (both action-aligned, our active → opp). Off ⇒ neither → baseline byte-identical.
        self.out_dim = (self.incoming_dim + (_DMG_OUTGOING + _DMG_STATUS if outgoing else 0)
                        + (_dmg_topk_dim(_lean_topk) if _lean_topk > 0 else 0)
                        + (_DMG_OMX if matrices_outgoing else 0)
                        + (_dmg_imx_dim(self.matrices_incoming_k) if matrices_incoming else 0)
                        + (_DMG_OAX if matrices_outgoing_all else 0))
        # Runtime grad-checkpointing flag (set per run by --grad-checkpointing via
        # _apply_grad_checkpointing) — recompute the op in backward, trading idle-GPU compute for the
        # ~GBs of [B,6,C]-over-~416-candidate activations this op materialises at batch 16384. No-op
        # under inference (gated on is_grad_enabled). Bit-exact (no dropout/RNG in the op).
        self.grad_checkpointing = False
        # Learnable per-channel adapter (the "structure to learn" — answers the review's M3): a gain on
        # each of the out_dim output channels, INITIALISED to put the heterogeneous physics channels on a
        # comparable scale (chip≤1.5, crit_delta≤1/16, the rest in [0,1]) so the shared pre_proj_norm
        # doesn't bury the small ones, then trained. ×only (no bias) → preserves the no-threat zeros (the
        # has_opp / defender_alive gates stay clean). OFF = no module, so this never touches the baseline.
        gain = torch.ones(self.out_dim)
        # per-mon block, 12 feats: [phys_low, phys_high, phys_crit, phys_pko, phys_acc, spec_low,
        # spec_high, spec_crit, spec_pko, spec_acc, p_outspeed, provenance] → pre-scale the rolls onto
        # ~[0,1]: low/high (cap 1.5) ÷1.5, crit (cap 3.0) ÷3.0; pko/acc/outspeed/provenance already [0,1].
        per_mon_init = torch.tensor([1.0 / 1.5, 1.0 / 1.5, 1.0 / 3.0, 1.0, 1.0,
                                     1.0 / 1.5, 1.0 / 1.5, 1.0 / 3.0, 1.0, 1.0, 1.0, 1.0])
        gain[:TEAM_SIZE * self.per_mon] = per_mon_init.repeat(TEAM_SIZE)
        # gen3_unified_choice_band_v1: the CB block tail [phys_high_cb×6, phys_pko_cb×6, p_cb] — scale the
        # CB high-roll like the other high rolls (cap 1.5 → ÷1.5); pko/p_cb already in [0,1] (stay 1.0).
        _cb0 = TEAM_SIZE * self.per_mon + _DMG_EFFECT + _DMG_INCOMING_SEC
        gain[_cb0:_cb0 + TEAM_SIZE] = 1.0 / 1.5                       # the phys_high_cb sub-block
        if outgoing:
            # outgoing block: per move [low, high, crit, pko] (same roll scaling), then p_outspeed.
            out_move_init = torch.tensor([1.0 / 1.5, 1.0 / 1.5, 1.0 / 3.0, 1.0])
            gain[self.incoming_dim:self.incoming_dim + _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE] = \
                out_move_init.repeat(_DMG_OUT_N_MOVES)
            # the trailing p_outspeed, the per-move secondary block, and the gen3_unified_status_landing_v1
            # status block (p_land/known) all stay at gain 1.0 — they are already probabilities in [0,1].
        if _lean_topk > 0:
            # gen3_unified_topk_incoming_v1: the LEAN top-K block tail (only when the rich matrix is OFF). The
            # opp-property sub-block (K × 35: latent ++ [belief, acc, is_phys]) stays at gain 1.0 — the latent
            # is LayerNorm-normalized downstream and belief/acc/is_phys are already [0,1]. The
            # per-(defender,move) sub-block (6·K × [high, pko, status_lands]) scales only the high-roll
            # (cap 1.5 → ÷1.5); pko/status are [0,1].
            _topk0 = self.incoming_dim + (_DMG_OUTGOING + _DMG_STATUS if outgoing else 0)
            _dmg0 = _topk0 + _lean_topk * _DMG_TOPK_MOVE
            _per = torch.tensor([1.0 / 1.5, 1.0, 1.0])               # [high, pko, status_lands]
            gain[_dmg0:_dmg0 + TEAM_SIZE * _lean_topk * _DMG_TOPK_DMG_PER] = _per.repeat(TEAM_SIZE * _lean_topk)
        if matrices_outgoing:
            # gen3_per_move_matrices_v1: the outgoing-matrix tail. Per (move, opp mon) cell
            # [low, high, crit, pko, type_mult] — scale low/high (÷1.5), crit (÷3.0), type_mult (cap 4× → ÷4);
            # pko already [0,1]. The trailing 6 `revealed` bits stay 1.0.
            _omx0 = (self.incoming_dim + (_DMG_OUTGOING + _DMG_STATUS if outgoing else 0)
                     + (_dmg_topk_dim(_lean_topk) if _lean_topk > 0 else 0))
            _cell_init = torch.tensor([1.0 / 1.5, 1.0 / 1.5, 1.0 / 3.0, 1.0, 1.0 / 4.0])  # low,high,crit,pko,mult
            gain[_omx0:_omx0 + _DMG_OUT_N_MOVES * TEAM_SIZE * _DMG_OMX_CELL] = \
                _cell_init.repeat(_DMG_OUT_N_MOVES * TEAM_SIZE)
        if matrices_incoming:
            # gen3_per_move_matrices_v1: the incoming-matrix tail. The per-move header (K × [latent(32),
            # belief, acc, is_phys, effect(6), secondary(10)]) stays gain 1.0 (latent normalized downstream;
            # the rest are [0,1]). The per-(mon, move) cell [low,high,crit,pko,type_mult,status] scales
            # low/high (÷1.5), crit (÷3), type_mult (cap 4× → ÷4); pko/status are [0,1].
            _imx0 = (self.incoming_dim + (_DMG_OUTGOING + _DMG_STATUS if outgoing else 0)
                     + (_dmg_topk_dim(_lean_topk) if _lean_topk > 0 else 0)   # 0 when matrices_incoming (lean off)
                     + (_DMG_OMX if matrices_outgoing else 0)
                     + self.matrices_incoming_k * _DMG_IMX_HEADER)            # after the per-move header
            _imx_cell = torch.tensor([1.0 / 1.5, 1.0 / 1.5, 1.0 / 3.0, 1.0, 1.0 / 4.0, 1.0])
            gain[_imx0:_imx0 + TEAM_SIZE * self.matrices_incoming_k * _DMG_IMX_CELL] = \
                _imx_cell.repeat(TEAM_SIZE * self.matrices_incoming_k)
        if matrices_outgoing_all:
            # gen3_per_move_matrices_v1 (v39): the TRANSPOSED outgoing-matrix tail. Per (attacker mon, move)
            # cell [low, high, crit, pko] — scale low/high (÷1.5), crit (÷3.0); pko already [0,1]. The trailing
            # p_outspeed[6] + alive[6] blocks stay at gain 1.0 (already in [0,1]). Placed AFTER every prior
            # block (incoming → outgoing → lean topk → omx → imx) — append LAST, all prior offsets untouched.
            _oax0 = (self.incoming_dim + (_DMG_OUTGOING + _DMG_STATUS if outgoing else 0)
                     + (_dmg_topk_dim(_lean_topk) if _lean_topk > 0 else 0)
                     + (_DMG_OMX if matrices_outgoing else 0)
                     + (_dmg_imx_dim(self.matrices_incoming_k) if matrices_incoming else 0))
            _oax_move = torch.tensor([1.0 / 1.5, 1.0 / 1.5, 1.0 / 3.0, 1.0])   # low,high,crit,pko
            gain[_oax0:_oax0 + TEAM_SIZE * _DMG_OAX_N_MOVES * _DMG_OAX_PER_MOVE] = \
                _oax_move.repeat(TEAM_SIZE * _DMG_OAX_N_MOVES)
        self.out_gain = torch.nn.Parameter(gain)
        # gen3_unified_topk_incoming_v1: detached side stashes for the prober (the selected candidate
        # indices + their belief weights) → exact move-name decode. None when topk off / before a forward.
        self.last_topk_idx: Optional[torch.Tensor] = None
        self.last_topk_w: Optional[torch.Tensor] = None

    def _opp_candidate_weights(self, ctx: 'ExtractorContext',
                               move_belief_logits: torch.Tensor) -> torch.Tensor:
        """Build the opp-active candidate belief weights ``w`` [B, n_moves] — the SINGLE source for all op
        candidate sites (``forward`` + the lean ``discrete_incoming`` / ``discrete_incoming_status`` refine
        kernels).

        **gen3_typed_hp_belief_v1 — HP arrives already typed.** The move-belief posterior this receives is
        the COMPOSED one (`HPTypeBelief.compose_typed_hp`, run once per forward next to the move-belief
        head): the 16 typed nums 355-370 already carry ``P(HP present)·P(HP type)`` and the bare typeless
        237 has been driven to a hard-off logit. So the op does NO Hidden-Power reasoning of its own — it
        prices HP-Ice and HP-Grass as the ordinary typed moves they are, off real BP/type rows, exactly
        like Thunderbolt.

        This replaces the old in-op scatter (`w[237]` × a locally-sourced type distribution). That version
        had two defects this removes structurally: the type SOURCE was chosen per call site — ``forward``
        passed the learned posterior while ``refine_candidates`` did not, so the between-layers refine
        kernels silently priced HP off the Smogon prior while the head block priced it off the learned
        belief — and in the (then default) `off` mode the source was the obs ``hp_probs``, which is
        all-zero until the opponent actually FIRES Hidden Power, so a REVEALED HP was priced as
        nonexistent. There is now exactly one HP posterior per forward and every consumer reads it.

        The mask keeps the bare 237 out of the candidate set (belt-and-braces: the composition already
        drives it to ~0) since it is a belief bookkeeping channel with BP 0, never a real move."""
        B, device = ctx.batch_size, ctx.device
        ar = torch.arange(B, device=device)
        w = torch.sigmoid(move_belief_logits[ar, ctx.opp_active_local])               # [B, n_moves] (typed)
        return w * self.HP_CAND_MASK[None, :]                                         # zero the 237 presence channel

    def _chan_max(self, value: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        """Max over a channel's candidates = the most-threatening believed move (exactly
        `incoming_damage.py`'s max-over-candidates). `value` [B,6,C] (≥0), `channel_mask` [1,1,C]
        (1=on-channel). Off-channel candidates are zeroed; since values are ≥0, `amax` returns the max
        on-channel value (or 0 if the channel has no threat). Differentiable via the argmax subgradient —
        the dominant move's belief weight gets gradient — and crucially NOT diluted by the ~400 zero-score
        candidates the way a low-temperature soft-max would be."""
        return (value * channel_mask).amax(dim=-1)

    @staticmethod
    def _boost_mult(stage: torch.Tensor) -> torch.Tensor:
        """gen3_unified_op_physics_v1: gen3 stat-stage multiplier (atk/def/spa/spd/spe). stage≥0 →
        (2+stage)/2, stage<0 → 2/(2−stage), clamped to [−6,6]. Mirrors incoming_damage.boost_mult."""
        s = stage.clamp(-6.0, 6.0)
        return torch.where(s >= 0, (2.0 + s) / 2.0, 2.0 / (2.0 - s))

    @staticmethod
    def _boost_stages(ctx_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                       torch.Tensor, torch.Tensor]:
        """Read the active mon's [atk,def,spa,spd,spe] boost STAGES ([B] each) from its active-context
        block (boosts = 7 stats × 2 dims [max(0,stage)/6, max(0,−stage)/6]); stage = (pos − neg)·6."""
        b = ctx_raw
        return ((b[:, 0] - b[:, 1]) * 6.0, (b[:, 2] - b[:, 3]) * 6.0, (b[:, 4] - b[:, 5]) * 6.0,
                (b[:, 6] - b[:, 7]) * 6.0, (b[:, 8] - b[:, 9]) * 6.0)

    @staticmethod
    def _weather_mult(weather_feature: torch.Tensor, is_water: torch.Tensor,
                      is_fire: torch.Tensor) -> torch.Tensor:
        """gen3_unified_op_physics_v1: gen3 weather BP modifier — rain (weather idx 2) ×1.5 Water / ×0.5
        Fire; sun (idx 1) ×1.5 Fire / ×0.5 Water; else 1.0. `is_water`/`is_fire` are broadcast-compatible
        per-candidate type-match flags. Sandstorm/Hail have no BP effect (gen3). Mirrors
        incoming_damage.weather_damage_mult."""
        sun = weather_feature[:, 1:2]                                                # [B,1]
        rain = weather_feature[:, 2:3]                                               # [B,1]
        return 1.0 + rain * (0.5 * is_water - 0.5 * is_fire) + sun * (0.5 * is_fire - 0.5 * is_water)

    def _rolls(self, dmg_ns, screen, maxhp, cur_hp, acc, eps: float = 1e-6):
        """The single source of the 3-roll + accuracy-folded-P(KO) physics — BOTH the incoming kernel
        and the outgoing block call this (the DRY core). From pre-screen max-roll damage ``dmg_ns`` + the
        DEFENDER's ``screen`` multiplier + ``maxhp``/``cur_hp`` + per-candidate ``acc`` (all broadcast-
        compatible) → ``(high_frac, low_frac, crit_frac, ko_ramp)``: the max-roll / 0.85-roll / ×2-crit
        damage as a fraction of MAX HP (gen3 crit ignores screens → ×2 the PRE-screen damage; clamped,
        "damage IF it lands"), and the accuracy-discounted P(KO this turn) vs CURRENT HP (``acc·P(KO|hit)``,
        the exact realized KO probability — accuracy and the roll are independent events)."""
        # `screen=None` means "no screen multiplier" — skips a full-tensor multiply by an all-ones
        # tensor (the coarse refine path allocated one every round). `x * 1.0 == x` exactly in IEEE-754
        # for every finite value, so the two forms are bit-identical.
        dmg = dmg_ns if screen is None else dmg_ns * screen               # post-screen max-roll
        inv = 1.0 / (maxhp + eps)
        high = (dmg * inv).clamp(max=_DMG_CHIP_CAP)
        low = (_DMG_ROLL_MIN * dmg * inv).clamp(max=_DMG_CHIP_CAP)
        crit = (2.0 * dmg_ns * inv).clamp(max=_DMG_CRIT_CAP)
        ko = acc * torch.clamp((dmg - cur_hp) / (0.15 * dmg + eps), 0.0, 1.0)
        return high, low, crit, ko

    def _nature_marg_ko(self, ko_ramp, high_frac, maxhp, cur_hp, acc_all, phys_all, fixed_all, nat_probs,
                        eps: float = 1e-6):
        """gen3_nature_ev_belief_v1: MARGINALISE the incoming per-(defender, candidate) P(KO) `ko_ramp` over
        the opp active's believed NATURE distribution (`--spread-belief-nature-marginalize`). The op consumes a
        single believed offense (= base_neutral·E[nature_mult]); P(KO) is a nonlinear THRESHOLD, so the
        mean-field read (`ko` at E[mult]) blurs the ×1.1/×0.9 asymmetry. Each candidate uses exactly ONE
        offensive stat (atk for physical, spa for special), so a 3-point quadrature over THAT stat's nature
        effect {reduce ×0.9, neither ×1.0, boost ×1.1} is EXACT (no cross-stat correlation to lose):

            P(KO)_marg = Σ_case P(stat in case)·acc·clamp((dmg·case_mult/E[mult] − cur)/(0.15·dmg·case_mult/
                         E[mult] + eps), 0, 1)

        where `dmg = high_frac·maxhp` reconstructs the believed post-screen max-roll (the cap only bites on
        overkill, which saturates P(KO)=1 either way). Differentiable in `nat_probs` → the op's KO gradient also
        sharpens the nature head. Fixed-damage candidates are nature-INVARIANT → kept at `ko_ramp` untouched.

        Shapes: `ko_ramp`/`high_frac` [B,n_def,C]; `maxhp`/`cur_hp` [B,n_def]; `acc_all`/`phys_all`/`fixed_all`
        **[B,C]** (per-batch-row since gen3_topk_candidates_v1 — the candidate set is this row's top-K);
        `nat_probs` [B,25] (softmax over natures at the opp active). Returns marginalised ko [B,n_def,C]."""
        is_boost = (self.NATURE_MULT == 1.1).float()                          # [25,5]
        is_reduce = (self.NATURE_MULT == 0.9).float()
        pboost = nat_probs @ is_boost                                         # [B,5] P(stat boosted) per stat
        preduce = nat_probs @ is_reduce                                       # [B,5] P(stat reduced)
        e_mult = (1.0 + 0.1 * pboost - 0.1 * preduce).clamp(min=eps)          # [B,5] E[nature mult] (head's)
        is_phys_c = phys_all                                                  # [B,C]

        def _stat(t):                                                        # [B,5] → [B,C] atk if phys else spa
            return t[:, _SB_ATK:_SB_ATK + 1] * is_phys_c + t[:, _SB_SPA:_SB_SPA + 1] * (1.0 - is_phys_c)
        pb, pr, em = _stat(pboost), _stat(preduce), _stat(e_mult)            # [B,C] each
        pn = (1.0 - pb - pr).clamp(min=0.0)                                   # P(neither)
        dmg = (high_frac * maxhp[:, :, None]).clamp(min=eps)                  # [B,n,C] reconstructed believed dmg
        cur = cur_hp[:, :, None]                                              # [B,n,1]
        acc = acc_all[:, None, :]                                             # [B,1,C]

        def _ramp(r):                                                        # r [B,C] offense ratio vs believed
            d = dmg * r[:, None, :]
            return acc * torch.clamp((d - cur) / (0.15 * d + eps), 0.0, 1.0)  # [B,n,C]
        # `dmg` already folds the head's E[mult] (believed offense = base_neutral·E[mult] → high_frac ∝ it), and
        # `em` here == that SAME E[mult]; so `dmg·case_mult/em = base_neutral·case_mult·(physics)` — the em
        # CANCELS, leaving the per-case offense exactly. The nature-posterior gradient therefore flows ONLY
        # through the case WEIGHTS (pr/pn/pb), a clean single path (NOT double-counted) — do NOT detach `em`
        # (that would un-cancel it and bias the gradient). base_neutral depends on nat_probs only via the EV head.
        ko_marg = (pr[:, None, :] * _ramp(0.9 / em)
                   + pn[:, None, :] * _ramp(1.0 / em)
                   + pb[:, None, :] * _ramp(1.1 / em))                        # [B,n,C]
        keep = (fixed_all > 0).float()[:, None, :]                           # fixed-damage → nature-invariant
        return keep * ko_ramp + (1.0 - keep) * ko_marg

    def _damage_rolls(self, atk: torch.Tensor, spa: torch.Tensor, at1: torch.Tensor, at2: torch.Tensor,
                      def_stat: torch.Tensor, spd_stat: torch.Tensor, maxhp: torch.Tensor,
                      cur_hp: torch.Tensor, t1d: torch.Tensor, t2d: torch.Tensor, ability1: torch.Tensor,
                      reflect: torch.Tensor, light_screen: torch.Tensor,
                      bp_all: torch.Tensor, mty_all: torch.Tensor, phys_all: torch.Tensor,
                      acc_all: torch.Tensor, fixed_all: torch.Tensor, weather_mult: torch.Tensor,
                      eps: float = 1e-6):
        """Role-parameterized gen3 single-hit damage per ``(defender, candidate)`` — the shared
        physics kernel every DIRECTION reuses (incoming opp→our-6, outgoing our→opp, safe-switch).
        Roles are passed in rather than hardcoded so the SAME math serves attacker/defender swaps.

        Shapes: ``atk``/``spa``/``at1``/``at2`` are ``[B]`` (one attacker); ``def_stat``/``spd_stat``/
        ``maxhp``/``cur_hp``/``t1d``/``t2d``/``ability1`` are ``[B, n_def]``; ``reflect``/``light_screen``
        are ``[B, 1]`` (the DEFENDER's side screens); ``bp_all``/``mty_all``/``phys_all`` are ``[C]``
        (the candidate move axis incl. the 16 typed Hidden Powers); ``acc_all`` is ``[C]`` (per-candidate
        base hit probability). Returns ``(high_frac, low_frac, crit_frac, ko_ramp)``, each ``[B, n_def,
        C]``: the max-roll / 0.85-roll / ×2-crit damage as a fraction of the defender's MAX HP (clamped —
        damage IF it lands), and the **accuracy-discounted** modal no-crit P(KO) vs CURRENT HP
        (``acc · P(KO|hit)`` — so an inaccurate move reads a lower KO-this-turn risk). Pure /
        differentiable (no learned params) — the shared physics every direction reuses."""
        # EFFECTIVENESS is folded in TYPE space (19 wide) BEFORE the candidate gather. The chart and
        # ability multipliers are per (defender, TYPE), so gathering each to the ~400-wide candidate
        # axis and multiplying THERE redid the same arithmetic C/19 ≈ 21× over. Multiplying the three
        # [B,n,19] tables first and gathering ONCE is BIT-IDENTICAL (same three factors, same
        # association order; a gather is exact) and drops 2 of 3 full-width gathers + 2 [B,n,C] muls.
        # Defender ABILITY immunity/resist (Levitate 0× Ground, Flash Fire 0× Fire, Thick Fat 0.5×
        # Fire/Ice) rides the same fold.
        # gen3_topk_candidates_v1: the per-candidate args are [B,C] (PER-BATCH-ROW), because the
        # candidate set is now the top-K of THIS row's move belief — different battles in a batch have
        # different opponents, so a batch-shared candidate list would be wrong. Every per-candidate
        # index therefore gathers instead of broadcasting. (Single call site, so the contract change
        # is local.)
        n_def = t1d.shape[1]
        eff19 = self.CHART[t1d] * self.CHART[t2d] * self.ABILITY_DAMAGE_MULT[ability1]           # [B,n,19]
        eff = eff19.gather(2, mty_all[:, None, :].expand(-1, n_def, -1))                        # [B,n,C]
        # ATTACK / DEFENCE selection by category. `phys·x + (1−phys)·y` over the candidate axis is a
        # GATHER wearing a multiply's clothes — `phys_all` is exactly 0/1, so the blend only ever
        # returns one of two values (and `1·x + 0·y == x` exactly in IEEE-754 for finite stats).
        # Indexing a 2-wide stack instead is value-identical AND moves the DIVISION off the candidate
        # axis: the reciprocal is taken on [B,n,2] and gathered, rather than ~400 divides per
        # (batch, defender). The reciprocal-then-multiply is the one FP-ordering change here.
        pidx = (phys_all > 0.5).long()                                                          # [B,C] 1=phys
        A = torch.stack((spa, atk), dim=-1).gather(1, pidx)                                     # [B,C]
        inv_d = (1.0 / (torch.stack((spd_stat, def_stat), dim=-1) + eps)) \
            .gather(2, pidx[:, None, :].expand(-1, n_def, -1))                                  # [B,n,C]
        is_stab = ((mty_all == at1[:, None]) | (mty_all == at2[:, None])).float()               # [B,C]
        stab = 1.0 + 0.5 * is_stab                                                              # [B,C]
        # DEFENDER-side screens: Reflect halves physical incoming, Light Screen halves special.
        # gen3 CRIT IGNORES screens, so the crit roll below uses the pre-screen damage (dmg_ns).
        screen = 1.0 - 0.5 * (reflect * phys_all + light_screen * (1.0 - phys_all))            # [B,C]
        # Every remaining per-candidate factor (STAB, the BP-0 gate, weather, the 0.925 constant) is
        # per (batch, candidate) — folding them into the [B,n,C] tensor ONE AT A TIME cost four
        # full-width multiplies where one does. Combine on the cheap [B,C] axis, apply once. The `/50`
        # and the 42 likewise fold into a [B,C] numerator, removing a second full-width division.
        bp_gate = (bp_all > 0).float()                                  # [B,C]; reused by the CB tail
        pre = stab * bp_gate * weather_mult * 0.925                     # [B,C] (weather: rain/sun BP)
        eff_pre = eff * pre[:, None, :]                                 # [B,n,C] shared by both cores
        core = ((42.0 / 50.0) * bp_all * A)[:, None, :] * inv_d + 2.0   # [B,n,C]
        dmg_ns = core * eff_pre                                         # [B,n,C] pre-screen
        # Final 3 rolls + accuracy-folded P(KO) via the shared formula (DRY — same as the outgoing block).
        high, low, crit, ko = self._rolls(dmg_ns, screen[:, None, :], maxhp[:, :, None], cur_hp[:, :, None],
                                          acc_all[:, None, :], eps)
        # gen3_unified_op_physics_v1: FIXED-damage moves (Seismic Toss / Night Shade = 100, Dragon Rage 40,
        # Sonic Boom 20) ignore Atk/Def/roll/crit but RESPECT type/ability immunity. Override the rolls with
        # the constant fraction (all three rolls equal — no variance), gated to 0 where `eff<=0` (Fighting
        # Seismic Toss → 0 vs Ghost; Ghost Night Shade → 0 vs Normal). Otherwise the BP-0 formula reads ~0.
        # gen3_unified_choice_band_v1: the CB-CONDITIONAL physical rolls — recompute with the physical Atk
        # ×1.5 at the STAT level (A_cb), so `core = k·A+2`'s +2 floor isn't itself ×1.5'd (the exact physics,
        # consistent with the outgoing block which scales our_atk). Special candidates unchanged. Only `high_cb`
        # / `ko_cb` are used (the op aggregates the PHYSICAL channel); the fixed-damage override below is
        # applied to them too (fixed damage is CB-independent → reads identically).
        A_cb = torch.stack((spa, atk + 0.5 * atk), dim=-1).gather(1, pidx)              # [B,C] physical Atk ×1.5
        # Only `high_cb` + `ko_cb` are aggregated (the special channel is CB-invariant), so compute them
        # INLINE rather than via _rolls — skips the unused low/crit rolls (~2×[B,n,C] of activations the
        # grad-checkpoint backward recompute would otherwise double; matters at batch 16384). `dmg_cb` folds
        # the defender screen in (post-screen), matching _rolls' high/ko exactly.
        dmg_cb = (((42.0 / 50.0) * bp_all * A_cb)[:, None, :] * inv_d + 2.0) \
            * eff_pre * screen[:, None, :]                              # [B,n,C] post-screen (reuses eff_pre)
        inv_cb = 1.0 / (maxhp[:, :, None] + eps)
        high_cb = (dmg_cb * inv_cb).clamp(max=_DMG_CHIP_CAP)
        ko_cb = acc_all[:, None, :] * torch.clamp(
            (dmg_cb - cur_hp[:, :, None]) / (0.15 * dmg_cb + eps), 0.0, 1.0)
        is_fixed = (fixed_all > 0)[:, None, :]                                         # [B,1,C]
        not_immune = (eff > 0).float()                                                # [B,n,C] type+ability gate
        fixed_frac = (fixed_all[:, None, :] / (maxhp[:, :, None] + eps)) * not_immune
        fixed_ko = acc_all[:, None, :] * (fixed_all[:, None, :] >= cur_hp[:, :, None]).float() * not_immune
        high = torch.where(is_fixed, fixed_frac, high)
        low = torch.where(is_fixed, fixed_frac, low)
        crit = torch.where(is_fixed, fixed_frac, crit)
        ko = torch.where(is_fixed, fixed_ko, ko)
        high_cb = torch.where(is_fixed, fixed_frac, high_cb)
        ko_cb = torch.where(is_fixed, fixed_ko, ko_cb)
        return high, low, crit, ko, high_cb, ko_cb

    def _p_outspeed(self, our_spe: torch.Tensor, opp_spe: torch.Tensor,
                    opp_spe_std: Optional[torch.Tensor] = None) -> torch.Tensor:
        """P(our mon outspeeds the opp active). LEGACY: a logistic over the speed gap at a FIXED scale.
        gen3_bidir_threat_trunk_v1 (#3, `prob_outspeed`): UNCERTAINTY-AWARE — divide the gap by the believed
        speed STD (sigmoid ≈ normal CDF ⇒ divisor = std/1.702), so a high-variance opp speed reads closer to
        0.5 and a well-pinned one reads sharp. All args broadcast together."""
        if self.prob_outspeed and opp_spe_std is not None:
            return torch.sigmoid((our_spe - opp_spe) / (opp_spe_std / _DMG_SPEED_STD_K + 1e-6))
        return torch.sigmoid((our_spe - opp_spe) / _DMG_SPEED_SCALE)

    def _outgoing_block(self, ctx: 'ExtractorContext',
                        spread_belief: Optional[torch.Tensor] = None) -> torch.Tensor:
        """OUR active → opp active, PER MOVE in REQUEST-slot order (== action logits 6+k), so the policy
        head can compare move A vs B directly — the equal-effectiveness tie-break (Earthquake vs Meteor
        Mash into a Rock: same 2× multiplier, different resolved damage). Our moves are KNOWN (no belief —
        a hard one-hot), LEGALITY-MASKED via the action mask (Choice-lock / Disable / Taunt / no-PP); the
        opp DEFENDER's bulk is hidden → a NEUTRAL 0-EV estimate (not max-bulk, which would under-price our
        KOs); opp ability immunity is revealed-or-none; OPP-side screens apply. Output `[B, _DMG_OUTGOING]`:
        per move `[low, high, crit, pko]` + one `p_outspeed`. Reuses the shared `_rolls` formula. Leak-safe
        (public obs only); gated to 0 when there is no opp active OR our active is fainted/absent. Our moves
        are certain → no move-belief gradient (correct: we don't learn our own moves), but differentiable
        in the smooth stat/damage formula."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        our_act = ctx.our_active_idx                                 # [B] our active slot (0..5)
        opp_act = TEAM_SIZE + ctx.opp_active_local                   # [B] opp active global slot
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B]
        our_alive = (ctx.hp_and_active[ar, our_act, 0] > 0).float()  # [B] our active must exist + be alive
        gate = (has_opp * our_alive)[:, None]                        # [B,1]

        # --- our 4 moves in REQUEST-slot order (action logits 6+k), legality-masked ---
        # gen3_op_move_align_v1: read the request-ordered obs slice (NOT all_move_ids[our_act], which is
        # sorted-by-id), so slot k's output ↔ action 6+k. `legal` is the CURRENT-decision choosability in
        # request order (was ctx.move_mask = prev-turn, sorted-by-id — a stale + misordered gate).
        move_ids = ctx.our_active_req_move_ids                       # [B,4] request order
        move_ty = ctx.our_active_req_move_type_ids                   # [B,4] resolved type (incl our HP type)
        legal = ctx.our_active_req_move_legal                        # [B,4] currently-legal (Choice/Disable/PP)
        is_hp = (move_ids == self.hp_num)
        bp = torch.where(is_hp, torch.full_like(move_ty, self.hp_bp, dtype=torch.float32),
                         self.MOVE_BP[move_ids])                     # [B,4] HP → 70 (else dex BP; status → 0)
        phys = self.TYPE_IS_PHYS[move_ty]                          # [B,4] gen3 category by resolved type
        acc = self.MOVE_ACCURACY[move_ids]                        # [B,4] (HP num → 1.0 default)
        usable = legal * (bp > 0).float()                         # [B,4] gate to legal damaging moves

        # --- our active attacker (real spread) ---
        a_base = self.BASE_STATS[ctx.species_ids[ar, our_act]]     # [B,6] [hp,atk,def,spa,spd,spe]
        spread = ctx.pokemon_part[ar, our_act,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spread[:, 0:6] * 31.0
        ev = spread[:, 6:12] * 252.0
        nat = spread[:, 13:18]                                     # [B,5] [atk,def,spa,spd,spe]
        our_atk = (2.0 * a_base[:, 1] + iv[:, 1] + ev[:, 1] / 4.0 + 5.0) * nat[:, 0]   # [B]
        our_spa = (2.0 * a_base[:, 3] + iv[:, 3] + ev[:, 3] / 4.0 + 5.0) * nat[:, 2]   # [B]
        our_spe = (2.0 * a_base[:, 5] + iv[:, 5] + ev[:, 5] / 4.0 + 5.0) * nat[:, 4]   # [B]
        # gen3_unified_op_physics_v1: OUR active's offensive + speed stat-stage boosts (we attack here) +
        # BURN (½ phys atk) + PARALYSIS (×0.25 speed).
        o_b_atk, o_b_def, o_b_spa, o_b_spd, o_b_spe = self._boost_stages(ctx.our_ctx_raw)
        our_burn = ctx.pokemon_part[ar, our_act, POKEMON_CONDITION_OFFSET + _COND_BRN_IDX]   # [B]
        our_para = ctx.pokemon_part[ar, our_act, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]   # [B]
        # gen3_unified_choice_band_v1: OUR Choice Band ×1.5 physical Atk (our item is KNOWN → deterministic,
        # not a belief). Composes multiplicatively with boosts/burn below; physical only (CB doesn't touch SpA).
        our_cb = (ctx.item_ids[ar, our_act] == self.cb_item_num).float()                     # [B]
        our_atk = our_atk * torch.where(our_cb > 0.5, our_atk.new_tensor(self.cb_phys_mult),
                                        our_atk.new_tensor(1.0))
        our_atk = our_atk * self._boost_mult(o_b_atk) * torch.where(
            our_burn > 0.5, our_atk.new_tensor(0.5), our_atk.new_tensor(1.0))
        our_spa = our_spa * self._boost_mult(o_b_spa)
        our_spe = our_spe * self._boost_mult(o_b_spe) * torch.where(
            our_para > 0.5, our_spe.new_tensor(_DMG_PARA_SPEED), our_spe.new_tensor(1.0))
        at1 = ctx.type1_ids[ar, our_act]                          # [B] our types (STAB)
        at2 = ctx.type2_ids[ar, our_act]

        # --- opp active defender (revealed species/types; ability revealed-or-none) ---
        # Bulk: the SpreadBelief's learned def/spd if provided (gen3_unified_spread_belief_v1), else the
        # legacy NEUTRAL 0-EV estimate (not max-bulk, which would under-price our KOs). maxhp stays the
        # neutral estimate either way (HP EVs vary little + the obs HP fraction carries relative HP).
        d_base = self.BASE_STATS[ctx.species_ids[ar, opp_act]]     # [B,6]
        bs = spread_belief[ar, ctx.opp_active_local] if spread_belief is not None else None  # [B,5] or None
        opp_def = bs[:, _SB_DEF] if bs is not None else (2.0 * d_base[:, 2] + 31.0 + 5.0)    # [B]
        opp_spd = bs[:, _SB_SPD] if bs is not None else (2.0 * d_base[:, 4] + 31.0 + 5.0)
        opp_maxhp = 2.0 * d_base[:, 0] + 31.0 + 110.0
        opp_spe = bs[:, _SB_SPE] if bs is not None else (2.0 * d_base[:, 5] + 31.0 + 5.0)   # believed / neutral
        # gen3_unified_op_physics_v1: OPP active's DEFENSIVE + speed boosts (it's the defender here) + its
        # paralysis (×0.25 speed, for p_outspeed).
        p_b_atk, p_b_def, p_b_spa, p_b_spd, p_b_spe = self._boost_stages(ctx.opp_ctx_raw)
        opp_para = ctx.pokemon_part[ar, opp_act, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]   # [B]
        opp_def = opp_def * self._boost_mult(p_b_def)
        opp_spd = opp_spd * self._boost_mult(p_b_spd)
        opp_spe = opp_spe * self._boost_mult(p_b_spe) * torch.where(
            opp_para > 0.5, opp_spe.new_tensor(_DMG_PARA_SPEED), opp_spe.new_tensor(1.0))
        opp_cur_hp = ctx.hp_and_active[ar, opp_act, 0] * opp_maxhp  # [B] obs HP frac × est. max HP
        t1d = ctx.type1_ids[ar, opp_act]                          # [B]
        t2d = ctx.type2_ids[ar, opp_act]
        opp_ability = ctx.ability1_ids[ar, opp_act]              # [B] (0 if unrevealed → no immunity mult)

        # --- gen3 damage per move (defender = opp active, candidates = our 4 moves), via the shared rolls ---
        eff = self.CHART[t1d[:, None], move_ty] * self.CHART[t2d[:, None], move_ty]   # [B,4]
        eff = eff * self.ABILITY_DAMAGE_MULT[opp_ability].gather(1, move_ty)          # [B,4] defender immunity
        A = phys * our_atk[:, None] + (1.0 - phys) * our_spa[:, None]                 # [B,4]
        D = phys * opp_def[:, None] + (1.0 - phys) * opp_spd[:, None]                 # [B,4]
        is_stab = ((move_ty == at1[:, None]) | (move_ty == at2[:, None])).float()
        stab = 1.0 + 0.5 * is_stab
        core = 42.0 * bp * A / (D + eps) / 50.0 + 2.0
        weather_mult = self._weather_mult(ctx.weather_feature, (move_ty == _WATER_TIDX).float(),
                                          (move_ty == _FIRE_TIDX).float())             # [B,4] rain/sun
        dmg_ns = core * stab * eff * 0.925 * usable * weather_mult                    # [B,4] (non-usable → 0)
        opp_reflect = ctx.screen_feature[:, 1:2]                                      # OPP-side screens
        opp_ls = ctx.screen_feature[:, 3:4]
        screen = 1.0 - 0.5 * (opp_reflect * phys + opp_ls * (1.0 - phys))             # [B,4]
        high, low, crit, ko = self._rolls(dmg_ns, screen, opp_maxhp[:, None], opp_cur_hp[:, None], acc, eps)
        # gen3_unified_op_physics_v1: OUR fixed-damage moves (Seismic Toss into the opp), immunity-gated +
        # legality-gated (usable). Mirrors the incoming kernel's override.
        fixed = self.MOVE_FIXED_DAMAGE[move_ids] * usable                            # [B,4] (0 if illegal)
        is_fixed = fixed > 0
        not_immune = (eff > 0).float()                                               # [B,4] type+ability gate
        fixed_frac = (fixed / (opp_maxhp[:, None] + eps)) * not_immune
        fixed_ko = acc * (fixed >= opp_cur_hp[:, None]).float() * not_immune
        high = torch.where(is_fixed, fixed_frac, high)
        low = torch.where(is_fixed, fixed_frac, low)
        crit = torch.where(is_fixed, fixed_frac, crit)
        ko = torch.where(is_fixed, fixed_ko, ko)
        opp_spe_std = self.SPECIES_SPREAD_PRIOR[ctx.species_ids[ar, opp_act], _SB_SPE, 1]   # [B] (#3)
        p_outspeed = self._p_outspeed(our_spe, opp_spe, opp_spe_std)                  # [B]

        # gen3_unified_move_system_v1: per OUR move, "what status can it cause + with what probability".
        # realized P(effect k | move) = chance_mk × acc_m × Serene Grace(our active) × Shield Dust(opp
        # active), gated to legal moves (status moves carry 0 secondary → naturally zeroed). Order ==
        # SECONDARY_COLS. [B,4,10].
        our_serene = self.ABILITY_SECONDARY_MULT[ctx.ability1_ids[ar, our_act]]        # [B] our active
        opp_block = self.ABILITY_SECONDARY_BLOCK[opp_ability]                          # [B] opp Shield Dust
        sec = self.MOVE_SECONDARY[move_ids]                                            # [B,4,10] base chance
        sec = sec * (acc * legal)[:, :, None] * (our_serene * opp_block)[:, None, None]
        sec = sec.clamp(max=1.0)                                                        # [B,4,10]

        per_move = torch.stack([low, high, crit, ko], dim=-1)                          # [B,4,4]
        block = torch.cat([per_move.reshape(B, -1), p_outspeed[:, None], sec.reshape(B, -1)], dim=1)  # [B, _DMG_OUTGOING]
        return block * gate

    def _outgoing_matrix(self, ctx: 'ExtractorContext',
                         spread_belief: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_per_move_matrices_v1: OUR active's 4 moves → the opp's 6 mons (active + REVEALED bench). The
        bench extension of `_outgoing_block` (which prices only the opp active): per (our move k, opp mon d)
        a `[low, high, crit, pko, type_mult]` cell so the policy prices a KO on a SWITCH-IN, plus a per-opp-mon
        `revealed` bit. REVEALED-gated — an UNREVEALED opp slot (Gen3 no team preview) is zeroed (its species/
        types/bulk are unknown; belief-driven damage there is a TODO). Reuses the `_outgoing_block` physics
        (attacker = our active with CB/boost/burn; OPP-side screens; per-defender bulk = SpreadBelief or the
        neutral 0-EV estimate; only the opp ACTIVE carries boosts — bench is reset), broadcast over the 6
        defenders. Output `[B, _DMG_OMX]` (grouped by move, action-aligned). Gated to 0 with no opp / fainted
        or absent our active. Leak-safe (revealed species + our known moves only)."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        our_act = ctx.our_active_idx
        opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B]
        our_alive = (ctx.hp_and_active[ar, our_act, 0] > 0).float()                     # [B]
        gate = (has_opp * our_alive)[:, None]                                           # [B,1]

        # --- our 4 moves in REQUEST-slot order (action 6+k), legality-masked (== _outgoing_block) ---
        # gen3_op_move_align_v1: request-ordered obs slice + current-decision legality (see _outgoing_block).
        move_ids = ctx.our_active_req_move_ids                                          # [B,4] request order
        move_ty = ctx.our_active_req_move_type_ids                                      # [B,4]
        legal = ctx.our_active_req_move_legal                                           # [B,4]
        is_hp = (move_ids == self.hp_num)
        bp = torch.where(is_hp, torch.full_like(move_ty, self.hp_bp, dtype=torch.float32),
                         self.MOVE_BP[move_ids])                                        # [B,4]
        phys = self.TYPE_IS_PHYS[move_ty]                                               # [B,4]
        acc = self.MOVE_ACCURACY[move_ids]                                              # [B,4]
        usable = legal * (bp > 0).float()                                               # [B,4]

        # --- our active attacker (real spread; CB ×1.5 phys, offensive boosts, burn) — same as _outgoing_block ---
        a_base = self.BASE_STATS[ctx.species_ids[ar, our_act]]
        spr = ctx.pokemon_part[ar, our_act, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spr[:, 0:6] * 31.0; ev = spr[:, 6:12] * 252.0; nat = spr[:, 13:18]
        our_atk = (2.0 * a_base[:, 1] + iv[:, 1] + ev[:, 1] / 4.0 + 5.0) * nat[:, 0]
        our_spa = (2.0 * a_base[:, 3] + iv[:, 3] + ev[:, 3] / 4.0 + 5.0) * nat[:, 2]
        o_b_atk, _odf, o_b_spa, _osd, _ose = self._boost_stages(ctx.our_ctx_raw)
        our_burn = ctx.pokemon_part[ar, our_act, POKEMON_CONDITION_OFFSET + _COND_BRN_IDX]
        our_cb = (ctx.item_ids[ar, our_act] == self.cb_item_num).float()
        our_atk = our_atk * torch.where(our_cb > 0.5, our_atk.new_tensor(self.cb_phys_mult), our_atk.new_tensor(1.0))
        our_atk = our_atk * self._boost_mult(o_b_atk) * torch.where(
            our_burn > 0.5, our_atk.new_tensor(0.5), our_atk.new_tensor(1.0))
        our_spa = our_spa * self._boost_mult(o_b_spa)
        at1 = ctx.type1_ids[ar, our_act]; at2 = ctx.type2_ids[ar, our_act]              # [B] (STAB)
        A = phys * our_atk[:, None] + (1.0 - phys) * our_spa[:, None]                   # [B,4]

        # --- opp 6 defenders (bulk = SpreadBelief or neutral 0-EV; boosts only on the active slot) ---
        d_base = self.BASE_STATS[ctx.species_ids[:, opp]]                               # [B,6,6]
        if spread_belief is not None:
            opp_def = spread_belief[:, :, _SB_DEF]; opp_spd = spread_belief[:, :, _SB_SPD]   # [B,6]
        else:
            opp_def = 2.0 * d_base[..., 2] + 31.0 + 5.0; opp_spd = 2.0 * d_base[..., 4] + 31.0 + 5.0
        opp_maxhp = 2.0 * d_base[..., 0] + 31.0 + 110.0                                 # [B,6]
        _pa, p_b_def, _ps, p_b_spd, _pe = self._boost_stages(ctx.opp_ctx_raw)
        def_boost = torch.ones_like(opp_def); def_boost[ar, ctx.opp_active_local] = self._boost_mult(p_b_def)
        spd_boost = torch.ones_like(opp_spd); spd_boost[ar, ctx.opp_active_local] = self._boost_mult(p_b_spd)
        opp_def = opp_def * def_boost; opp_spd = opp_spd * spd_boost
        opp_hp_frac = ctx.hp_and_active[:, opp, 0]                                      # [B,6]
        opp_cur_hp = opp_hp_frac * opp_maxhp                                            # [B,6]
        t1d = ctx.type1_ids[:, opp]; t2d = ctx.type2_ids[:, opp]                        # [B,6]
        opp_ability = ctx.ability1_ids[:, opp]                                          # [B,6]
        revealed = (~ctx.opp_believed_mask).float()                                     # [B,6] species known
        def_gate = revealed * (opp_hp_frac > 0).float()                                 # [B,6] real switchable target

        # --- type effectiveness eff[B,4,6] = CHART[t1d]·CHART[t2d]·ability_mult, gathered by move type ---
        T = self.CHART.shape[-1]
        mty_e = move_ty[:, :, None, None].expand(B, _DMG_OUT_N_MOVES, TEAM_SIZE, 1)      # [B,4,6,1]
        def _gather_type(table_per_def):                                                 # table [B,6,T] → [B,4,6]
            t = table_per_def[:, None].expand(B, _DMG_OUT_N_MOVES, TEAM_SIZE, T)
            return torch.gather(t, 3, mty_e).squeeze(-1)
        eff = (_gather_type(self.CHART[t1d]) * _gather_type(self.CHART[t2d])
               * _gather_type(self.ABILITY_DAMAGE_MULT[opp_ability]))                    # [B,4,6]
        # --- gen3 damage per (move, defender) → [B,4,6] (the _outgoing_block physics, broadcast over 6) ---
        D = phys[:, :, None] * opp_def[:, None, :] + (1.0 - phys)[:, :, None] * opp_spd[:, None, :]   # [B,4,6]
        is_stab = ((move_ty == at1[:, None]) | (move_ty == at2[:, None])).float()        # [B,4]
        stab = (1.0 + 0.5 * is_stab)[:, :, None]                                          # [B,4,1]
        core = 42.0 * bp[:, :, None] * A[:, :, None] / (D + eps) / 50.0 + 2.0             # [B,4,6]
        weather = self._weather_mult(ctx.weather_feature, (move_ty == _WATER_TIDX).float(),
                                     (move_ty == _FIRE_TIDX).float())[:, :, None]         # [B,4,1]
        dmg_ns = core * stab * eff * 0.925 * usable[:, :, None] * weather                 # [B,4,6]
        opp_reflect = ctx.screen_feature[:, 1:2]; opp_ls = ctx.screen_feature[:, 3:4]     # OPP-side screens
        screen = (1.0 - 0.5 * (opp_reflect * phys + opp_ls * (1.0 - phys)))[:, :, None]   # [B,4,1]
        high, low, crit, ko = self._rolls(dmg_ns, screen, opp_maxhp[:, None, :],
                                          opp_cur_hp[:, None, :], acc[:, :, None], eps)    # each [B,4,6]
        # fixed-damage moves (Seismic Toss into a defender): immunity + legality gated, CB-invariant.
        fixed = (self.MOVE_FIXED_DAMAGE[move_ids] * usable)[:, :, None]                   # [B,4,1]
        is_fixed = fixed > 0
        not_immune = (eff > 0).float()                                                   # [B,4,6]
        fixed_frac = (fixed / (opp_maxhp[:, None, :] + eps)) * not_immune
        fixed_ko = acc[:, :, None] * (fixed >= opp_cur_hp[:, None, :]).float() * not_immune
        high = torch.where(is_fixed, fixed_frac, high); low = torch.where(is_fixed, fixed_frac, low)
        crit = torch.where(is_fixed, fixed_frac, crit); ko = torch.where(is_fixed, fixed_ko, ko)

        cell = torch.stack([low, high, crit, ko, eff], dim=-1)                            # [B,4,6,_DMG_OMX_CELL]
        cell = cell * (usable[:, :, None, None] * def_gate[:, None, :, None])             # gate move-legal × real target
        out = torch.cat([cell.reshape(B, _DMG_OUT_N_MOVES * TEAM_SIZE * _DMG_OMX_CELL), def_gate], dim=1)  # [B,_DMG_OMX]
        return out * gate

    def _outgoing_attacker_matrix(self, ctx: 'ExtractorContext',
                                  spread_belief: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_per_move_matrices_v1 (v39): the TRANSPOSE of `_outgoing_matrix` — our 6 MONS' 4 moves → the opp
        ACTIVE. On a FORCED SWITCH our active is fainted, so `_outgoing_block`/`_outgoing_matrix` (which only
        price the current active attacker) ZERO and the policy picks switch-ins BLIND to offense; this prices
        every candidate switch-in's offense vs the opp active. The ACTIVE row reproduces `_outgoing_block`
        byte-for-byte (its boosts/CB/burn + request-ordered moves + the same opp-active defender + the same
        `_rolls` kernel); bench rows reuse the SAME `_rolls` physics with NEUTRAL boosts (gen3 resets boosts on
        switch) + the per-mon sorted-by-id moves (bench mons have no current-decision request order). Output
        `[B, _DMG_OAX]` = all (attacker, move) cells `[low,high,crit,pko]` ++ a per-attacker `p_outspeed[6]` ++
        an `alive[6]` gate. Gated to 0 with no opp active; each attacker gated by its alive bit. Leak-safe
        (public obs + the believed opp spread only — same inputs as `_outgoing_block`)."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        our = slice(0, TEAM_SIZE)
        opp_act = TEAM_SIZE + ctx.opp_active_local                     # [B] opp active global slot
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B] (== _outgoing_block)

        # --- our 6 attackers' moves: per-mon sorted-by-id block, ACTIVE row OVERWRITTEN with the request slice ---
        # The active uses the SAME request-ordered slice _outgoing_block reads (==action 6+k) → byte-for-byte
        # parity on the active row; bench mons (no request order) use all_move_ids[:, :TEAM_SIZE] (sorted-by-id).
        move_ids = ctx.all_move_ids[:, our].clone()                    # [B,6,4] sorted-by-id
        move_ty = ctx.all_move_type_ids[:, our].clone()                # [B,6,4]
        legal = torch.ones(B, TEAM_SIZE, _DMG_OAX_N_MOVES, device=device)   # bench: all moves available
        move_ids[ar, ctx.our_active_idx] = ctx.our_active_req_move_ids        # active → request order (parity)
        move_ty[ar, ctx.our_active_idx] = ctx.our_active_req_move_type_ids
        legal[ar, ctx.our_active_idx] = ctx.our_active_req_move_legal         # active → current-decision legality
        is_hp = (move_ids == self.hp_num)
        bp = torch.where(is_hp, torch.full_like(move_ty, self.hp_bp, dtype=torch.float32),
                         self.MOVE_BP[move_ids])                       # [B,6,4]
        phys = self.TYPE_IS_PHYS[move_ty]                              # [B,6,4]
        acc = self.MOVE_ACCURACY[move_ids]                            # [B,6,4]
        usable = legal * (bp > 0).float()                            # [B,6,4] legal damaging moves

        # --- our 6 attackers (real spread; CB ×1.5 phys, burn; boosts only on the active slot, bench reset) ---
        a_base = self.BASE_STATS[ctx.species_ids[:, our]]            # [B,6,6] [hp,atk,def,spa,spd,spe]
        spread = ctx.pokemon_part[:, our,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]   # [B,6,SPREAD]
        iv = spread[..., 0:6] * 31.0
        ev = spread[..., 6:12] * 252.0
        nat = spread[..., 13:18]                                     # [B,6,5] [atk,def,spa,spd,spe]
        our_atk = (2.0 * a_base[..., 1] + iv[..., 1] + ev[..., 1] / 4.0 + 5.0) * nat[..., 0]   # [B,6]
        our_spa = (2.0 * a_base[..., 3] + iv[..., 3] + ev[..., 3] / 4.0 + 5.0) * nat[..., 2]   # [B,6]
        our_spe = (2.0 * a_base[..., 5] + iv[..., 5] + ev[..., 5] / 4.0 + 5.0) * nat[..., 4]   # [B,6]
        # Boosts: the ACTIVE row carries our_ctx_raw's stages; bench rows neutral (mult 1.0) — gen3 resets on
        # switch (mirrors _outgoing_matrix's defender-boost handling exactly).
        o_b_atk, _odf, o_b_spa, _osd, o_b_spe = self._boost_stages(ctx.our_ctx_raw)   # active only, [B]
        atk_boost = torch.ones(B, TEAM_SIZE, device=device); atk_boost[ar, ctx.our_active_idx] = self._boost_mult(o_b_atk)
        spa_boost = torch.ones(B, TEAM_SIZE, device=device); spa_boost[ar, ctx.our_active_idx] = self._boost_mult(o_b_spa)
        spe_boost = torch.ones(B, TEAM_SIZE, device=device); spe_boost[ar, ctx.our_active_idx] = self._boost_mult(o_b_spe)
        # Burn / Choice Band compose PER MON (each mon's own KNOWN condition/item).
        our_burn = ctx.pokemon_part[:, our, POKEMON_CONDITION_OFFSET + _COND_BRN_IDX]   # [B,6]
        our_para = ctx.pokemon_part[:, our, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]   # [B,6]
        our_cb = (ctx.item_ids[:, our] == self.cb_item_num).float()                     # [B,6]
        our_atk = our_atk * torch.where(our_cb > 0.5, our_atk.new_tensor(self.cb_phys_mult), our_atk.new_tensor(1.0))
        our_atk = our_atk * atk_boost * torch.where(our_burn > 0.5, our_atk.new_tensor(0.5), our_atk.new_tensor(1.0))
        our_spa = our_spa * spa_boost
        our_spe = our_spe * spe_boost * torch.where(our_para > 0.5, our_spe.new_tensor(_DMG_PARA_SPEED),
                                                    our_spe.new_tensor(1.0))
        at1 = ctx.type1_ids[:, our]; at2 = ctx.type2_ids[:, our]                        # [B,6] (STAB)
        A = phys * our_atk[:, :, None] + (1.0 - phys) * our_spa[:, :, None]             # [B,6,4]

        # --- opp ACTIVE defender (identical to _outgoing_block: believed/neutral bulk, defensive boosts) ---
        d_base = self.BASE_STATS[ctx.species_ids[ar, opp_act]]        # [B,6]
        bs = spread_belief[ar, ctx.opp_active_local] if spread_belief is not None else None   # [B,5] or None
        opp_def = bs[:, _SB_DEF] if bs is not None else (2.0 * d_base[:, 2] + 31.0 + 5.0)     # [B]
        opp_spd = bs[:, _SB_SPD] if bs is not None else (2.0 * d_base[:, 4] + 31.0 + 5.0)
        opp_maxhp = 2.0 * d_base[:, 0] + 31.0 + 110.0
        opp_spe = bs[:, _SB_SPE] if bs is not None else (2.0 * d_base[:, 5] + 31.0 + 5.0)
        _pa, p_b_def, _ps, p_b_spd, p_b_spe = self._boost_stages(ctx.opp_ctx_raw)
        opp_para = ctx.pokemon_part[ar, opp_act, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]   # [B]
        opp_def = opp_def * self._boost_mult(p_b_def)
        opp_spd = opp_spd * self._boost_mult(p_b_spd)
        opp_spe = opp_spe * self._boost_mult(p_b_spe) * torch.where(
            opp_para > 0.5, opp_spe.new_tensor(_DMG_PARA_SPEED), opp_spe.new_tensor(1.0))
        opp_cur_hp = ctx.hp_and_active[ar, opp_act, 0] * opp_maxhp    # [B]
        t1d = ctx.type1_ids[ar, opp_act]; t2d = ctx.type2_ids[ar, opp_act]   # [B]
        opp_ability = ctx.ability1_ids[ar, opp_act]                  # [B] (0 if unrevealed)

        # --- type effectiveness eff[B,6,4] = CHART[t1d]·CHART[t2d]·ability_mult (single defender, gathered) ---
        eff = (self.CHART[t1d][:, None, None, :].expand(B, TEAM_SIZE, _DMG_OAX_N_MOVES, self.CHART.shape[-1])
               .gather(3, move_ty[..., None]).squeeze(-1))            # [B,6,4]
        eff = eff * (self.CHART[t2d][:, None, None, :].expand(B, TEAM_SIZE, _DMG_OAX_N_MOVES, self.CHART.shape[-1])
                     .gather(3, move_ty[..., None]).squeeze(-1))
        eff = eff * (self.ABILITY_DAMAGE_MULT[opp_ability][:, None, None, :]
                     .expand(B, TEAM_SIZE, _DMG_OAX_N_MOVES, self.ABILITY_DAMAGE_MULT.shape[-1])
                     .gather(3, move_ty[..., None]).squeeze(-1))      # [B,6,4] defender immunity

        # --- gen3 damage per (attacker, move) → [B,6,4] (the _outgoing_block physics, single opp defender) ---
        D = phys * opp_def[:, None, None] + (1.0 - phys) * opp_spd[:, None, None]       # [B,6,4]
        is_stab = ((move_ty == at1[:, :, None]) | (move_ty == at2[:, :, None])).float() # [B,6,4]
        stab = 1.0 + 0.5 * is_stab
        core = 42.0 * bp * A / (D + eps) / 50.0 + 2.0                                   # [B,6,4]
        # gen3 weather BP modifier (== _weather_mult): sun/rain are [B,1] → broadcast as [B,1,1] over [B,6,4].
        is_water = (move_ty == _WATER_TIDX).float(); is_fire = (move_ty == _FIRE_TIDX).float()   # [B,6,4]
        sun = ctx.weather_feature[:, 1:2, None]; rain = ctx.weather_feature[:, 2:3, None]        # [B,1,1]
        weather = 1.0 + rain * (0.5 * is_water - 0.5 * is_fire) + sun * (0.5 * is_fire - 0.5 * is_water)  # [B,6,4]
        dmg_ns = core * stab * eff * 0.925 * usable * weather                           # [B,6,4]
        opp_reflect = ctx.screen_feature[:, 1:2]; opp_ls = ctx.screen_feature[:, 3:4]   # OPP-side screens [B,1]
        screen = 1.0 - 0.5 * (opp_reflect[:, :, None] * phys + opp_ls[:, :, None] * (1.0 - phys))  # [B,6,4]
        high, low, crit, ko = self._rolls(dmg_ns, screen, opp_maxhp[:, None, None],
                                          opp_cur_hp[:, None, None], acc, eps)           # each [B,6,4]
        # fixed-damage moves (Seismic Toss into the opp active): immunity + legality gated, CB-invariant.
        fixed = self.MOVE_FIXED_DAMAGE[move_ids] * usable                              # [B,6,4]
        is_fixed = fixed > 0
        not_immune = (eff > 0).float()                                                 # [B,6,4]
        fixed_frac = (fixed / (opp_maxhp[:, None, None] + eps)) * not_immune
        fixed_ko = acc * (fixed >= opp_cur_hp[:, None, None]).float() * not_immune
        high = torch.where(is_fixed, fixed_frac, high); low = torch.where(is_fixed, fixed_frac, low)
        crit = torch.where(is_fixed, fixed_frac, crit); ko = torch.where(is_fixed, fixed_ko, ko)

        # --- p_outspeed per attacker (our_spe [B,6] vs the shared believed opp speed) ---
        opp_spe_std = self.SPECIES_SPREAD_PRIOR[ctx.species_ids[ar, opp_act], _SB_SPE, 1]   # [B]
        p_outspeed = self._p_outspeed(our_spe, opp_spe[:, None].expand(B, TEAM_SIZE),
                                      opp_spe_std[:, None].expand(B, TEAM_SIZE))             # [B,6]

        # --- assemble + gate ---
        per_move = torch.stack([low, high, crit, ko], dim=-1)                           # [B,6,4,4]
        alive = (ctx.hp_and_active[:, our, 0] > 0).float()                              # [B,6] attacker exists+alive
        per_move = per_move * (usable[:, :, :, None] * alive[:, :, None, None])         # gate legal × alive attacker
        p_outspeed = p_outspeed * alive
        out = torch.cat([per_move.reshape(B, TEAM_SIZE * _DMG_OAX_N_MOVES * _DMG_OAX_PER_MOVE),
                         p_outspeed, alive], dim=1)                                     # [B, _DMG_OAX]
        return out * has_opp[:, None]                                                   # zeroed when no opp active

    # ------------------------------------------------------------------ pointer-native action head cells
    @property
    def pointer_move_cell_dim(self) -> int:
        """Per-request-slot cell width for the pointer MOVE scorer (0 when the outgoing direction is off —
        the head's Linear in_features are fixed by the build-time toggle set, the op's own convention)."""
        return _PTR_MOVE_CELL if self.outgoing else 0

    @property
    def pointer_switch_cell_dim(self) -> int:
        """Per-candidate-mon cell width for the pointer SWITCH scorer."""
        return _PTR_SWITCH_CELL_IN + (_PTR_SWITCH_CELL_OAX if self.matrices_outgoing_all else 0)

    def pointer_cells(self, damage_block: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """gen3_pointer_native_v1: slice the flat damage block into PER-ACTION cells for the pointer head —
        the op owns its layout, so the offsets live here (mirroring `decode_damage_block`, the SoT mirror)
        and the consumer never hardcodes an index.

        Returns ``(move_cells [B,4,pointer_move_cell_dim], switch_cells [B,6,pointer_switch_cell_dim])``:
          * move cell k (REQUEST-slot order == action logit 6+k, the `gen3_op_move_align_v1` guarantee —
            `_outgoing_block`/`_status_landing` read `ctx.our_active_req_move_ids`, the same id source the
            pointer token permutation matches against): `[low, high, crit, pko, p_land, known, sec×10]`.
          * switch cell j: the incoming per-defender row (12) + `[phys_high_cb_j, pko_cb_j, p_cb]` +
            (when `matrices_outgoing_all`) the OAX attacker row `[cells×16, p_outspeed_j, alive_j]`.
        Pure slicing of the SAME tensor the projection heads consume (post-gain), so the pointer path and
        the flat concat can never disagree on a value."""
        B = damage_block.shape[0]
        # --- switch cells: incoming per-mon rows + the CB tail (+ the OAX attacker rows) ---
        inc = damage_block[:, :TEAM_SIZE * _DMG_PER_MON].reshape(B, TEAM_SIZE, _DMG_PER_MON)
        cb0 = TEAM_SIZE * _DMG_PER_MON + _DMG_EFFECT + _DMG_INCOMING_SEC
        high_cb = damage_block[:, cb0:cb0 + TEAM_SIZE]                                   # [B,6]
        pko_cb = damage_block[:, cb0 + TEAM_SIZE:cb0 + 2 * TEAM_SIZE]                    # [B,6]
        p_cb = damage_block[:, cb0 + 2 * TEAM_SIZE:cb0 + 2 * TEAM_SIZE + 1]              # [B,1] shared
        switch_parts = [inc, high_cb[:, :, None], pko_cb[:, :, None],
                        p_cb[:, None, :].expand(B, TEAM_SIZE, 1)]
        if self.matrices_outgoing_all:
            oax0 = self.out_dim - _DMG_OAX                    # OAX is appended LAST (the v39 contract)
            cells = damage_block[:, oax0:oax0 + TEAM_SIZE * _DMG_OAX_PER_MON].reshape(
                B, TEAM_SIZE, _DMG_OAX_PER_MON)
            posp0 = oax0 + TEAM_SIZE * _DMG_OAX_PER_MON
            p_outspeed = damage_block[:, posp0:posp0 + TEAM_SIZE]                        # [B,6]
            alive = damage_block[:, posp0 + TEAM_SIZE:posp0 + 2 * TEAM_SIZE]             # [B,6]
            switch_parts += [cells, p_outspeed[:, :, None], alive[:, :, None]]
        switch_cells = torch.cat(switch_parts, dim=2)                                    # [B,6,Cs]
        # --- move cells: the outgoing damage stack + status landing + per-move secondaries ---
        if not self.outgoing:
            return damage_block.new_zeros(B, _DMG_OUT_N_MOVES, 0), switch_cells
        ob = self.incoming_dim
        per_move = damage_block[:, ob:ob + _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE].reshape(
            B, _DMG_OUT_N_MOVES, _DMG_OUT_PER_MOVE)                                      # move-major [B,4,4]
        sec0 = ob + _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE + 1                             # skip p_outspeed
        sec = damage_block[:, sec0:sec0 + _DMG_OUT_SEC].reshape(B, _DMG_OUT_N_MOVES, _N_SECONDARY)
        st0 = ob + _DMG_OUTGOING                                                         # the status block
        p_land = damage_block[:, st0:st0 + _DMG_STATUS_N_MOVES]                          # [B,4]
        known = damage_block[:, st0 + _DMG_STATUS_N_MOVES:st0 + 2 * _DMG_STATUS_N_MOVES]  # [B,4]
        move_cells = torch.cat([per_move, p_land[:, :, None], known[:, :, None], sec], dim=2)
        return move_cells, switch_cells                                                  # [B,4,16], [B,6,Cs]

    def _status_landing(self, ctx: 'ExtractorContext') -> torch.Tensor:
        """gen3_unified_status_landing_v1: per OUR move (REQUEST-slot order == action 6+k), P(a dedicated
        STATUS move applies to the opp active) + a `known` bit — the GPU home for the masked move-effect
        block's `status_will_land`. The status MOVES the outgoing DAMAGE block can't price (BP 0 → usable 0).

        P(lands) = is_status_move · accuracy · (1−type_immune) · (1−ability_block) · (1−already_block)
                   · (1−sleep_clause_block), gated to 0 with no opp active / our active dead. Where:
          • type_immune  — per-MOVE gen3 rule (Thunder Wave→Ground, Toxic/Poison→Steel/Poison, Will-O-Wisp
            →Fire, **Leech Seed→Grass**), max over the opp active's two types.
          • ability_block — REVEALED opp ability → exact `ABILITY_STATUS_BLOCK`; UNREVEALED → the species
            Smogon-prior marginal `SPECIES_STATUS_BLOCK_PRIOR` (Snorlax Toxic ≈0.14 Immunity-dominated).
          • already_block — the opp active already carries a major status (can't double-apply); NOT Leech Seed.
          • sleep_clause_block — a SLEEP move fails if ANY opp mon is already asleep via a NON-Rest source
            (`sleep_is_deterministic==0`). Rest self-sleep does NOT consume our cap (the user's rule). The
            per-mon Rest flag is the existing gen3_sleep_wake_belief_v1 `sleep_is_deterministic` (reused).
          • has_sub — the opp active behind a Substitute blocks EVERY status move (incl. Leech Seed); read
            from the public Substitute volatile in `ctx.opp_ctx_raw` at `_SUBSTITUTE_CTX_IDX`.
        `known` = the value rests on CERTAIN (public) info — a type/already-statused/Sleep-Clause/Substitute
        hard block OR a revealed ability — vs a Smogon-prior estimate. No move-belief gradient (OUR moves are
        certain). UNCOVERED residual: Yawn (delayed sleep, no status_inflicted), Leech-Seed-already-seeded."""
        B, device = ctx.batch_size, ctx.device
        ar = torch.arange(B, device=device)
        our_act = ctx.our_active_idx                                  # [B]
        opp_act = TEAM_SIZE + ctx.opp_active_local                    # [B] opp-active global slot
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B]
        our_alive = (ctx.hp_and_active[ar, our_act, 0] > 0).float()   # [B]
        gate = (has_opp * our_alive)[:, None]                         # [B,1]

        # gen3_op_move_align_v1: request-ordered obs slice so p_land[k] ↔ action 6+k (was
        # all_move_ids[our_act], sorted-by-id → the output was positionally misaligned with the actions).
        move_ids = ctx.our_active_req_move_ids                        # [B,4] request order
        inflicts = self.MOVE_INFLICTS_STATUS[move_ids]               # [B,4]
        acc = self.MOVE_ACCURACY[move_ids]                          # [B,4] (Toxic .85, WoW .75, T-Wave 1, …)
        sidx = self.MOVE_STATUS_CAT[move_ids]                       # [B,4] long (0 = not a status move)
        is_sleep = self.MOVE_IS_SLEEP[move_ids]                     # [B,4]
        blocked_if_statused = self.MOVE_BLOCKED_IF_STATUSED[move_ids]  # [B,4] (0 for Leech Seed)

        # type immunity (per move) — max over the opp active's two types.
        t1 = ctx.type1_ids[ar, opp_act]                            # [B]
        t2 = ctx.type2_ids[ar, opp_act]
        ti = self.MOVE_STATUS_TYPE_IMMUNE[move_ids]                 # [B,4,N_TYPE_IDX]
        type_immune = torch.maximum(ti.gather(2, t1[:, None, None].expand(B, 4, 1)).squeeze(2),
                                    ti.gather(2, t2[:, None, None].expand(B, 4, 1)).squeeze(2))  # [B,4]

        # ability immunity — revealed → exact; unrevealed (id 0) → the species Smogon-prior marginal.
        opp_ability = ctx.ability1_ids[ar, opp_act]                # [B] (0 if unrevealed)
        opp_species = ctx.species_ids[ar, opp_act]                 # [B]
        abl_rev = self.ABILITY_STATUS_BLOCK[opp_ability].gather(1, sidx)           # [B,4]
        abl_prior = self.SPECIES_STATUS_BLOCK_PRIOR[opp_species].gather(1, sidx)   # [B,4]
        revealed = (opp_ability > 0).float()[:, None]              # [B,1]
        ability_block = revealed * abl_rev + (1.0 - revealed) * abl_prior          # [B,4]

        # already-statused (opp active) — any non-None status bit → blocks a MAJOR status (not Leech Seed).
        opp_cond = ctx.pokemon_part[ar, opp_act,
                                    POKEMON_CONDITION_OFFSET + 1:POKEMON_CONDITION_OFFSET + 7]  # [B,6]
        already_statused = (opp_cond.sum(dim=1) > 0.5).float()[:, None]            # [B,1]
        already_block = already_statused * blocked_if_statused                     # [B,4]

        # Sleep Clause — ANY opp mon asleep via a NON-Rest source consumes our one-sleep cap.
        opp_slp = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE, POKEMON_CONDITION_OFFSET + _COND_SLP_IDX]  # [B,6]
        opp_rest = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE, POKEMON_SLEEP_BELIEF_OFFSET]  # [B,6] is_rest
        nonrest_sleep = opp_slp * (1.0 - opp_rest)                                 # [B,6]
        sleep_clause = (nonrest_sleep.sum(dim=1) > 0.5).float()[:, None]           # [B,1]
        sleep_block = sleep_clause * is_sleep                                      # [B,4]

        # Substitute — the opp active behind a Sub blocks EVERY status move (incl. Leech Seed) in gen3. Read
        # the public Substitute volatile from the opp active context (boosts ++ volatiles). Applies to ALL
        # inflicting moves (not just majors), so it folds in as a flat per-channel factor below.
        has_sub = (ctx.opp_ctx_raw[:, _SUBSTITUTE_CTX_IDX] > 0.5).float()[:, None]  # [B,1]

        p_land = (inflicts * acc * (1.0 - type_immune) * (1.0 - ability_block)
                  * (1.0 - already_block) * (1.0 - sleep_block) * (1.0 - has_sub))  # [B,4]
        # `known` = the value rests on CERTAIN info (a hard block — type/already-statused/Sleep-Clause/
        # Substitute, all PUBLIC — or a revealed ability) vs a Smogon-prior estimate.
        certain = torch.clamp(type_immune + already_block + sleep_block + has_sub + revealed, max=1.0)  # [B,4]
        known = inflicts * certain                                                 # [B,4]
        return torch.cat([p_land, known], dim=1) * gate                            # [B, _DMG_STATUS]

    def _incoming_status_lands(self, ctx: 'ExtractorContext', topk_idx: torch.Tensor,
                               high_topk: torch.Tensor) -> torch.Tensor:
        """gen3_unified_topk_incoming_v1: per (OUR defender d, top-K move k), P(move k applies a status to
        defender d) — the immunity-folded per-pivot safe-switch read (Thunder Wave → a Ground pivot = 0).
        Combines two mutually-exclusive paths, taking the max:
          • DEDICATED status move (Thunder Wave/Toxic/Will-O-Wisp/Spore/Leech Seed, BP 0): `inflicts · acc ·
            (1−type_immune@our_def_types) · (1−ability_block@our_def_ability) · (1−already)` — the
            per-MOVE type immunity (Thunder Wave→Ground, Toxic→Steel/Poison, WoW→Fire, Leech Seed→Grass)
            evaluated at OUR DEFENDER's types (the incoming mirror of `_status_landing`'s opp lookup).
          • DAMAGING-move MAJOR-status SECONDARY (Body Slam para, Ice Beam frz): `max_col(chance_col ·
            (1−ability_block[cat(col)])) · acc · Serene-Grace(opp) · 1[damage lands on this pivot] ·
            (1−already)` — gated by `high_topk>0`, so a pivot immune to the DAMAGE (Ghost vs Body Slam)
            shows 0 status risk too. gen3 has no type-based para/freeze immunity beyond that gate.
        All inputs are buffers + OUR-side public obs (types/ability/condition known) + the opp's revealed
        Serene Grace → w-INDEPENDENT (the belief gradient rides `w_topk`, not this). HP candidates carry no
        status (extended with zeros). v2 residual: incoming Sleep-Clause / our-Substitute (the owner's named
        case is type immunity)."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        K = topk_idx.shape[1]
        ar = torch.arange(B, device=device)
        opp_act = TEAM_SIZE + ctx.opp_active_local
        n_type = self.MOVE_STATUS_TYPE_IMMUNE.shape[1]
        # --- candidate-axis (C = n_moves; the typed HP nums 355-370 carry no status/secondary — all-zero
        # in these buffers, verified) move-status attributes → gather top-K (gen3_opp_hp_typed_candidates_v1) ---
        inflicts = self.MOVE_INFLICTS_STATUS[topk_idx]                                        # [B,K]
        acc = self.MOVE_ACCURACY[topk_idx]                                                    # [B,K]
        sidx = self.MOVE_STATUS_CAT[topk_idx]                                                 # [B,K]
        blocked = self.MOVE_BLOCKED_IF_STATUSED[topk_idx]                                     # [B,K]
        ti = self.MOVE_STATUS_TYPE_IMMUNE[topk_idx]                                           # [B,K,n_type]
        sec = self.MOVE_SECONDARY[topk_idx]                                                   # [B,K,10]

        # --- our 6 defenders' (known) types / ability / already-statused ---
        t1d = ctx.type1_ids[:, :TEAM_SIZE]                                                    # [B,6]
        t2d = ctx.type2_ids[:, :TEAM_SIZE]
        abl = self.ABILITY_STATUS_BLOCK[ctx.ability1_ids[:, :TEAM_SIZE]]                       # [B,6,7]
        our_cond = ctx.pokemon_part[:, :TEAM_SIZE,
                                    POKEMON_CONDITION_OFFSET + 1:POKEMON_CONDITION_OFFSET + 7]  # [B,6,6]
        already = (our_cond.sum(-1) > 0.5).float()                                            # [B,6]

        # --- DEDICATED status move landing: type immunity @ our defender types (max over the 2 types) ---
        ti_dk = ti[:, None, :, :].expand(B, TEAM_SIZE, K, n_type)                              # [B,6,K,n_type]
        ti1 = torch.gather(ti_dk, 3, t1d[:, :, None, None].expand(B, TEAM_SIZE, K, 1)).squeeze(-1)
        ti2 = torch.gather(ti_dk, 3, t2d[:, :, None, None].expand(B, TEAM_SIZE, K, 1)).squeeze(-1)
        t_imm = torch.maximum(ti1, ti2)                                                       # [B,6,K]
        abl_block = torch.gather(abl, 2, sidx[:, None, :].expand(B, TEAM_SIZE, K))             # [B,6,K]
        already_block = already[:, :, None] * blocked[:, None, :]                             # [B,6,K]
        dedicated = (inflicts[:, None, :] * acc[:, None, :] * (1.0 - t_imm)
                     * (1.0 - abl_block) * (1.0 - already_block))                             # [B,6,K]

        # --- DAMAGING-move MAJOR-status SECONDARY (gated by the damage actually landing on this pivot) ---
        opp_serene = self.ABILITY_SECONDARY_MULT[ctx.ability1_ids[ar, opp_act]]               # [B]
        sec_major = sec[..., :_SECONDARY_MAJOR_N]                                             # [B,K,6]
        abl_per_col = abl[..., self._SEC_CAT_IDX]                                             # [B,6,6] per status cat
        sec_land = (sec_major[:, None, :, :] * (1.0 - abl_per_col)[:, :, None, :]).amax(dim=-1)  # [B,6,K]
        damage_gate = (high_topk > eps).float()                                               # [B,6,K]
        secondary = (sec_land * acc[:, None, :] * opp_serene[:, None, None]
                     * damage_gate * (1.0 - already[:, :, None])).clamp(max=1.0)              # [B,6,K]
        return torch.maximum(dedicated, secondary)                                            # [B,6,K]

    def _topk_block(self, ctx: 'ExtractorContext', w_all: torch.Tensor, high_frac: torch.Tensor,
                    ko_ramp: torch.Tensor, acc_all: torch.Tensor, phys_all: torch.Tensor,
                    move_latent_all: torch.Tensor, has_opp: torch.Tensor,
                    defender_alive: torch.Tensor, cand_nums: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_unified_topk_incoming_v1: the DISCRETE top-K incoming block. Selects the opp active's K
        most-believed CANDIDATE moves (over `w_all` [B,C], indices DETACHED) and surfaces each individually:
          • opp-property (shared across defenders): the move LATENT (gathered from `move_latent_all` [C,32]
            — DIFFERENTIABLE → sharpens the MoveLatentEncoder) ++ [belief_w (DIFFERENTIABLE → sharpens the
            move belief), accuracy, is_phys].
          • per (our defender d, move k): [high-roll, P(KO), status_lands] — `high`/`pko` gathered from the
            RAW (w-independent) physics rolls (`0` for a damage-immune pivot), `status_lands` immunity-folded
            (§`_incoming_status_lands`; `0` for a status-immune pivot). The two immunity kinds the safest
            switches turn on.
        Decorrelated by design (damage/status are w-independent physics; the belief gradient rides `w_topk`,
        the latent gradient rides `latent_topk`). Meaningful-K gate: once all 4 opp-active moves are revealed
        the moveset is closed → the 5th+ slot is zeroed (nothing left to reason about). Gated to 0 with no
        opp active / per fainted defender. Output `[B, _dmg_topk_dim(K)]`."""
        B, device = ctx.batch_size, ctx.device
        K = self.topk_k
        ar = torch.arange(B, device=device)
        # --- select the K most-believed candidates (selection DETACHED; gathered values differentiable) ---
        topk_idx = w_all.detach().topk(K, dim=-1).indices                          # [B,K]
        w_topk = w_all.gather(-1, topk_idx)                                        # [B,K] → belief gradient
        # gen3_topk_candidates_v1: `topk_idx` indexes the (possibly TRUNCATED) candidate axis, so any
        # gather into the FULL move space — the latent table, the effect/secondary buffers, the type
        # ids, the status-landing physics and the prober's stash — must go through `cand_nums` to get
        # the REAL move-num. None ⇒ no truncation ⇒ the reduced index IS the move-num.
        real_idx = topk_idx if cand_nums is None else cand_nums.gather(-1, topk_idx)   # [B,K] move-nums
        self.last_topk_idx = real_idx.detach()                                     # prober: exact move names
        self.last_topk_w = w_topk.detach()
        # --- opp-property: latent (→ MoveLatentEncoder gradient) + belief + accuracy + is_phys ---
        latent_topk = move_latent_all[real_idx]                                    # [B,K,32] differentiable
        acc_topk = acc_all.gather(-1, topk_idx)                                    # [B,K] (buffer, no grad)
        phys_topk = phys_all.gather(-1, topk_idx)                                  # [B,K] (buffer, no grad)
        # --- per (defender, move) damage: gather the RAW physics rolls (w-INDEPENDENT → decorrelated) ---
        idxd = topk_idx[:, None, :].expand(B, TEAM_SIZE, K)                        # [B,6,K]
        high_topk = high_frac.gather(-1, idxd)                                     # [B,6,K] (0 if dmg-immune)
        pko_topk = ko_ramp.gather(-1, idxd)                                        # [B,6,K]
        status_topk = self._incoming_status_lands(ctx, real_idx, high_topk)        # [B,6,K] (0 if status-immune)
        # --- meaningful-K gate: a gen3 mon has exactly 4 moves, so once all 4 are revealed the moveset is
        # closed → zero the 5th+ slot (nothing left to reason about); <4 revealed → all K slots live. ---
        opp_act = TEAM_SIZE + ctx.opp_active_local
        n_revealed = (ctx.all_move_ids[ar, opp_act] > 0).sum(-1)                   # [B] revealed opp-move count
        slot_live = ((torch.arange(K, device=device)[None, :] < 4)
                     | (n_revealed[:, None] < 4)).float()                          # [B,K]
        # --- assemble + gate (opp-property gated by has_opp×slot_live; damage also by defender_alive) ---
        opp_prop = torch.cat([latent_topk, w_topk[..., None], acc_topk[..., None],
                              phys_topk[..., None]], dim=-1)                        # [B,K,_DMG_TOPK_MOVE]
        opp_prop = opp_prop * (has_opp[:, None, None] * slot_live[:, :, None])
        per_def = torch.stack([high_topk, pko_topk, status_topk], dim=-1)          # [B,6,K,3]
        per_def = per_def * (has_opp[:, None, None, None] * defender_alive[:, :, None, None]
                             * slot_live[:, None, :, None])
        return torch.cat([opp_prop.reshape(B, K * _DMG_TOPK_MOVE),
                          per_def.reshape(B, TEAM_SIZE * K * _DMG_TOPK_DMG_PER)], dim=1)

    def _incoming_matrix(self, ctx: 'ExtractorContext', w_all: torch.Tensor, low_frac: torch.Tensor,
                         high_frac: torch.Tensor, crit_frac: torch.Tensor, ko_ramp: torch.Tensor,
                         acc_all: torch.Tensor, phys_all: torch.Tensor, move_latent_all: torch.Tensor,
                         has_opp: torch.Tensor, defender_alive: torch.Tensor,
                         matrix_k: int, cand_nums: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_per_move_matrices_v1: the INCOMING per-move DAMAGE MATRIX — the ENRICHED top-K block (replaces
        it). For the opp active's top-`matrix_k` most-believed candidates (selection DETACHED): a per-move
        HEADER [latent(32), belief w (→ sharpens the belief), accuracy, is_phys, EXPLICIT effect bits (6,
        from MOVE_EFFECT_FLAGS — recovery/status/phaze/boost/hazard/protect, per move, un-collapsed), EXPLICIT
        secondary chances (10, from MOVE_SECONDARY)] and a per-(OUR mon, move) CELL [low, high, crit, pko,
        type_mult, status_lands]. Damage rolls GATHER from the SAME validated `_damage_rolls` tensors as the
        worst-case block (so an immune pivot reads 0); type_mult is the effectiveness at OUR defender's types;
        status_lands reuses `_incoming_status_lands`. Decorrelated (belief gradient rides `w`, latent rides the
        gather). Meaningful-K gate (zero the 5th+ slot once all 4 opp moves are revealed). HP candidates carry
        zero effect/secondary (extended with zeros). Output `[B, _dmg_imx_dim(matrix_k)]`."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        K = matrix_k
        ar = torch.arange(B, device=device)
        topk_idx = w_all.detach().topk(K, dim=-1).indices                          # [B,K] (detached selection)
        w_topk = w_all.gather(-1, topk_idx)                                        # [B,K] → belief gradient
        # gen3_topk_candidates_v1: `topk_idx` indexes the (possibly TRUNCATED) candidate axis, so any
        # gather into the FULL move space — the latent table, the effect/secondary buffers, the type
        # ids, the status-landing physics and the prober's stash — must go through `cand_nums` to get
        # the REAL move-num. None ⇒ no truncation ⇒ the reduced index IS the move-num.
        real_idx = topk_idx if cand_nums is None else cand_nums.gather(-1, topk_idx)   # [B,K] move-nums
        self.last_topk_idx = real_idx.detach()                                     # prober: exact move names
        self.last_topk_w = w_topk.detach()
        # --- per-move header: latent (→ MoveLatentEncoder grad) + belief + accuracy + is_phys + effect + secondary ---
        latent_topk = move_latent_all[real_idx]                                    # [B,K,32] differentiable
        acc_topk = acc_all.gather(-1, topk_idx)                                    # [B,K]
        phys_topk = phys_all.gather(-1, topk_idx)                                  # [B,K]
        # HP at the typed nums 355-370 carries no effect/secondary (all-zero in these buffers, verified);
        # C = n_moves (gen3_opp_hp_typed_candidates_v1 — the typed HP are ordinary move-num candidates).
        eff_flags = self.MOVE_EFFECT_FLAGS[real_idx]                               # [B,K,6]
        sec = self.MOVE_SECONDARY[real_idx]                                        # [B,K,10]
        # --- per-(defender, move) cell: gather the RAW physics rolls (w-INDEPENDENT) + type_mult + status ---
        idxd = topk_idx[:, None, :].expand(B, TEAM_SIZE, K)                        # [B,6,K]
        low_topk = low_frac.gather(-1, idxd)                                       # [B,6,K]
        high_topk = high_frac.gather(-1, idxd)
        crit_topk = crit_frac.gather(-1, idxd)
        pko_topk = ko_ramp.gather(-1, idxd)
        # type_mult @ OUR defenders' types/ability for the top-K move types (the immune/resist pivot read)
        mty_topk = self.MOVE_TYPE_IDX[real_idx]                                    # [B,K]
        idx2 = mty_topk[:, None, :].expand(B, TEAM_SIZE, K)                         # [B,6,K]
        t1d = ctx.type1_ids[:, :TEAM_SIZE]; t2d = ctx.type2_ids[:, :TEAM_SIZE]
        amul = self.ABILITY_DAMAGE_MULT[ctx.ability1_ids[:, :TEAM_SIZE]]            # [B,6,T]
        type_mult = (torch.gather(self.CHART[t1d], 2, idx2) * torch.gather(self.CHART[t2d], 2, idx2)
                     * torch.gather(amul, 2, idx2))                                 # [B,6,K]
        status_topk = self._incoming_status_lands(ctx, real_idx, high_topk)        # [B,6,K]
        # --- meaningful-K gate (== _topk_block): once all 4 opp moves revealed, the 5th+ slot is closed ---
        opp_act = TEAM_SIZE + ctx.opp_active_local
        n_revealed = (ctx.all_move_ids[ar, opp_act] > 0).sum(-1)                   # [B]
        slot_live = ((torch.arange(K, device=device)[None, :] < 4)
                     | (n_revealed[:, None] < 4)).float()                          # [B,K]
        header = torch.cat([latent_topk, w_topk[..., None], acc_topk[..., None],
                            phys_topk[..., None], eff_flags, sec], dim=-1)          # [B,K,_DMG_IMX_HEADER]
        header = header * (has_opp[:, None, None] * slot_live[:, :, None])
        cell = torch.stack([low_topk, high_topk, crit_topk, pko_topk, type_mult, status_topk], dim=-1)  # [B,6,K,6]
        cell = cell * (has_opp[:, None, None, None] * defender_alive[:, :, None, None]
                       * slot_live[:, None, :, None])
        return torch.cat([header.reshape(B, K * _DMG_IMX_HEADER),
                          cell.reshape(B, TEAM_SIZE * K * _DMG_IMX_CELL)], dim=1)

    def refine_candidates(self, ctx: 'ExtractorContext',
                          move_belief_logits: torch.Tensor,
                          k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """The SHARED candidate selection for the between-layers refine kernels → `(topk_idx, w_topk)`.

        `discrete_incoming` and `discrete_incoming_status` are called from the same refine round with the
        SAME `move_belief_logits` object, so each was independently rebuilding an identical `[B, n_moves]`
        sigmoid + typed-HP scatter and an identical top-K — 4 redundant candidate builds and 2 redundant
        top-Ks per forward in the production config. Hoisting it here lets the caller compute once and
        pass the result to both (they still fall back to computing it when called standalone).

        `k` overrides the default `_DMG_REFINE_K` — the E4 entity-seat builder
        (`gen3_entity_move_seats_v1`) reuses this selection at its own `entity_topk_seats` K, so the
        seats and the refine kernels share ONE candidate definition (the index selection stays
        DETACHED; the gathered weights stay differentiable so the belief gradient rides them)."""
        w_all = self._opp_candidate_weights(ctx, move_belief_logits)                     # [B, n_moves]
        K = min(_DMG_REFINE_K if k is None else int(k), w_all.shape[1])
        topk_idx = w_all.detach().topk(K, dim=-1).indices                                # [B,K] (DETACHED)
        return topk_idx, w_all.gather(-1, topk_idx)                                      # → belief gradient

    def discrete_incoming(self, ctx: 'ExtractorContext',
                          move_belief_logits: torch.Tensor,
                          cand: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """gen3_iterative_damage_v1: a LEAN per-our-mon incoming-threat summary for the ITERATIVE refinement
        (recomputed BETWEEN transformer layers, fed `move_belief_logits` re-read from the MID-transformer opp
        tokens). For the opp active's top-`_DMG_REFINE_K` most-believed candidate moves (the ~50×-cheaper
        primitive — K≈8 vs the full ~416 axis), compute the believed worst-case incoming damage to each of
        our 6 mons and reduce to `[B, TEAM_SIZE, _DMG_REFINE_FEATS]` = `[phys_high, spec_high, phys_pko,
        spec_pko]` (the per-channel max-roll fraction + accuracy-folded P(KO), belief-weighted via the SAME
        `_chan_max` hard-max the full op uses). Decorrelated: the damage physics is w-INDEPENDENT, the belief
        weighting rides `w_topk` (the gradient sharpens `move_head` each round). v1 uses the LEGACY de-timid
        attacker offense (NO spread belief / boost / burn / weather / fixed-damage — the coarse refinement
        signal; the FINAL post-transformer op carries the full physics and is authoritative). Reuses the
        shared `_rolls` formula + the op's physics buffers. Gated to 0 with no opp active / per fainted
        defender (no `/0`)."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        opp_act = TEAM_SIZE + ctx.opp_active_local                                        # [B] global opp-active
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()    # [B]
        # --- Attacker = opp active (legacy de-timid offense; the coarse refinement signal) ---
        a_base = self.BASE_STATS[ctx.species_ids[ar, opp_act]]                            # [B,6]
        off_const = 31.0 + 252.0 / 4.0 + 5.0
        atk = (2.0 * a_base[:, 1] + off_const) * 1.1                                      # [B]
        spa = (2.0 * a_base[:, 3] + off_const) * 1.1                                      # [B]
        at1 = ctx.type1_ids[ar, opp_act]                                                  # [B]
        at2 = ctx.type2_ids[ar, opp_act]
        # --- Defenders = our 6 (REAL spread reconstructed from the obs, like forward) ---
        d_base = self.BASE_STATS[ctx.species_ids[:, :TEAM_SIZE]]                          # [B,6,6]
        spread = ctx.pokemon_part[:, :TEAM_SIZE,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spread[..., 0:6] * 31.0
        ev = spread[..., 6:12] * 252.0
        nat = spread[..., 13:18]                                                          # [B,6,5]
        def_stat = (2.0 * d_base[..., 2] + iv[..., 2] + ev[..., 2] / 4.0 + 5.0) * nat[..., 1]   # [B,6]
        spd_stat = (2.0 * d_base[..., 4] + iv[..., 4] + ev[..., 4] / 4.0 + 5.0) * nat[..., 3]   # [B,6]
        maxhp = 2.0 * d_base[..., 0] + iv[..., 0] + ev[..., 0] / 4.0 + 110.0              # [B,6]
        hp_frac = ctx.hp_and_active[:, :TEAM_SIZE, 0]                                     # [B,6]
        cur_hp = hp_frac * maxhp
        defender_alive = (hp_frac > 0).float()                                           # [B,6]
        t1d = ctx.type1_ids[:, :TEAM_SIZE]                                               # [B,6]
        t2d = ctx.type2_ids[:, :TEAM_SIZE]
        ability = ctx.ability1_ids[:, :TEAM_SIZE]                                        # [B,6]
        # --- Belief at the opp active → w [B, n_moves] (same source as forward; the lean refine passes no
        # learned posterior → the prior FLOOR resolves the typed-HP belief, scattered onto 355-370; the bare
        # 237 is masked — gen3_opp_hp_typed_candidates_v1) ---
        # --- Candidate axis attributes: C = n_moves (the typed HP 355-370 carry real BP/type; no append) ---
        bp_all = self.MOVE_BP                                                            # [n_moves]
        mty_all = self.MOVE_TYPE_IDX                                                     # [n_moves]
        phys_all = self.MOVE_PHYS                                                        # [n_moves]
        acc_all = self.MOVE_ACCURACY                                                     # [n_moves]
        # --- SELECT the top-K most-believed candidates (selection DETACHED; gathered values differentiable).
        # Reused from the caller when the sibling status kernel already built it (`refine_candidates`). ---
        topk_idx, w_topk = cand if cand is not None else self.refine_candidates(ctx, move_belief_logits)
        K = topk_idx.shape[1]
        bp_k = bp_all[topk_idx]                                                          # [B,K]
        mty_k = mty_all[topk_idx]                                                        # [B,K] (long, TypeEncoder)
        phys_k = phys_all[topk_idx]                                                      # [B,K]
        acc_k = acc_all[topk_idx]                                                        # [B,K]
        # --- gen3 damage for the K candidates × 6 defenders → [B,6,K] (the lean per-K mirror of _damage_rolls) ---
        idxd = mty_k[:, None, :].expand(B, TEAM_SIZE, K)                                  # [B,6,K] type indices
        eff = torch.gather(self.CHART[t1d], 2, idxd) * torch.gather(self.CHART[t2d], 2, idxd)  # [B,6,K]
        amul = self.ABILITY_DAMAGE_MULT[ability]                                          # [B,6,T] defender ability
        eff = eff * torch.gather(amul, 2, idxd)                                           # [B,6,K] fold immunity
        A = phys_k * atk[:, None] + (1.0 - phys_k) * spa[:, None]                         # [B,K]
        D = (phys_k[:, None, :] * def_stat[:, :, None]
             + (1.0 - phys_k)[:, None, :] * spd_stat[:, :, None])                         # [B,6,K]
        is_stab = ((mty_k == at1[:, None]) | (mty_k == at2[:, None])).float()             # [B,K]
        stab = 1.0 + 0.5 * is_stab                                                        # [B,K]
        core = 42.0 * bp_k[:, None, :] * A[:, None, :] / (D + eps) / 50.0 + 2.0           # [B,6,K]
        dmg_ns = core * stab[:, None, :] * eff * 0.925                                    # [B,6,K] pre-screen
        dmg_ns = dmg_ns * (bp_k > 0).float()[:, None, :]                                  # kill the +2 floor on BP-0
        reflect, light_screen = ctx.screen_feature[:, 0:1], ctx.screen_feature[:, 2:3]    # [B,1] OUR-side screens
        screen = 1.0 - 0.5 * (reflect * phys_k + light_screen * (1.0 - phys_k))           # [B,K]
        high, _low, _crit, ko = self._rolls(dmg_ns, screen[:, None, :], maxhp[:, :, None],
                                            cur_hp[:, :, None], acc_k[:, None, :], eps)     # each [B,6,K]
        # --- per (defender, channel) HARD max of the belief-weighted roll/KO over the K candidates ---
        wb = w_topk[:, None, :]                                                           # [B,1,K]
        phys_mask = phys_k[:, None, :]                                                    # [B,1,K]
        spec_mask = 1.0 - phys_mask
        phys_high = (wb * high * phys_mask).amax(dim=-1)                                  # [B,6]
        spec_high = (wb * high * spec_mask).amax(dim=-1)
        phys_pko = (wb * ko * phys_mask).amax(dim=-1)
        spec_pko = (wb * ko * spec_mask).amax(dim=-1)
        feats = torch.stack([phys_high, spec_high, phys_pko, spec_pko], dim=-1)           # [B,6,_DMG_REFINE_FEATS]
        return feats * defender_alive[:, :, None] * has_opp[:, None, None]                # gates

    def discrete_outgoing(self, ctx: 'ExtractorContext',
                          species_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_bidir_threat_trunk_v1: the SYMMETRIC mirror of `discrete_incoming` for the OUTGOING direction.
        Computes how hard OUR active's 4 KNOWN moves hit each of the opp's 6 mons and reduces to
        `[B, TEAM_SIZE, _DMG_OUT_REFINE]` = `[phys_high, spec_high, phys_pko, spec_pko]` (per-channel
        best-over-our-moves max-roll fraction + accuracy-folded P(KO)) — injected onto the OPP token slice so
        attention reasons over "how threatened is each opp mon by us".

        Two defender regimes, selected per slot by `ctx.opp_believed_mask`:
          - REVEALED opp mon: real types (CHART) + revealed-ability immunity + a NEUTRAL bulk estimate (its
            sentinel-free base stats), gated by its real HP; FULL P(KO).
          - UNREVEALED opp mon (`species_probs` given): the EXPECTED-LATENT read — `E[mult]` via the
            `SPECIES_EXP_MULT` matmul with `P(species)` (folds type chart × the expected ability immunity),
            `E[def/spd]` via `SPECIES_SPREAD_PRIOR` means and `E[maxhp]` via `E[base_hp]` (the sentinel
            species 0 has zero base stats, so EVERYTHING must come from the belief, not `d_base`), assumed
            full-HP (a switch-in), with **P(KO) NULLED** (owner: ~never OHKO a full-HP switch-in). When
            `species_probs` is None, unrevealed slots are zeroed (the legacy revealed-gating).

        Coarse like `discrete_incoming` (our real base spread, no boosts/CB/screens/weather — the final
        `_outgoing_matrix` carries the full physics). Decorrelated: the `E[mult]` gradient rides
        `P(species)` (sharpening the species belief), the damage magnitude is belief-independent. Gated to 0
        with no opp active / no/own-fainted our active."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        our_act = ctx.our_active_idx                                              # [B]
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B]
        our_alive = (ctx.hp_and_active[ar, our_act, 0] > 0).float()               # [B]
        gate = has_opp * our_alive                                                # [B]
        # --- our 4 KNOWN moves (coarse: real base spread, no boosts/CB) ---
        # gen3_op_move_align_v1: read the request-ordered obs slice + current-decision legality (consistent
        # with the per-move-output methods). The output max-pools over our moves so the ORDER is invariant,
        # but the legality GATE must be current (was ctx.move_mask = prev-turn, sorted-by-id — a stale gate).
        move_ids = ctx.our_active_req_move_ids                                    # [B,4] request order
        move_ty = ctx.our_active_req_move_type_ids                                # [B,4] resolved (incl HP)
        legal = ctx.our_active_req_move_legal                                     # [B,4] current choosability
        is_hp = (move_ids == self.hp_num)
        bp = torch.where(is_hp, torch.full_like(move_ty, self.hp_bp, dtype=torch.float32),
                         self.MOVE_BP[move_ids])                                  # [B,4]
        phys = self.TYPE_IS_PHYS[move_ty]                                         # [B,4]
        acc = self.MOVE_ACCURACY[move_ids]                                        # [B,4]
        usable = legal * (bp > 0).float()                                         # [B,4] legal damaging moves
        a_base = self.BASE_STATS[ctx.species_ids[ar, our_act]]                    # [B,6]
        spread = ctx.pokemon_part[ar, our_act,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spread[:, 0:6] * 31.0
        ev = spread[:, 6:12] * 252.0
        nat = spread[:, 13:18]                                                    # [B,5]
        our_atk = (2.0 * a_base[:, 1] + iv[:, 1] + ev[:, 1] / 4.0 + 5.0) * nat[:, 0]   # [B]
        our_spa = (2.0 * a_base[:, 3] + iv[:, 3] + ev[:, 3] / 4.0 + 5.0) * nat[:, 2]   # [B]
        A = phys * our_atk[:, None] + (1.0 - phys) * our_spa[:, None]             # [B,4]
        at1 = ctx.type1_ids[ar, our_act]
        at2 = ctx.type2_ids[ar, our_act]
        is_stab = ((move_ty == at1[:, None]) | (move_ty == at2[:, None])).float()  # [B,4]
        stab = 1.0 + 0.5 * is_stab                                                # [B,4]
        # --- opp 6 defenders ---
        opp_species = ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE]                 # [B,6]
        d_base = self.BASE_STATS[opp_species]                                     # [B,6,6] (sentinel → 0s)
        opp_hp_frac = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0]            # [B,6]
        defender_alive = (opp_hp_frac > 0).float()                               # [B,6] (revealed slots)
        believed = ctx.opp_believed_mask.float()                                  # [B,6] 1.0 = UNREVEALED
        t1d = ctx.type1_ids[:, TEAM_SIZE:2 * TEAM_SIZE]                           # [B,6]
        t2d = ctx.type2_ids[:, TEAM_SIZE:2 * TEAM_SIZE]
        ability = ctx.ability1_ids[:, TEAM_SIZE:2 * TEAM_SIZE]                    # [B,6]
        mt = move_ty[:, None, :].expand(B, TEAM_SIZE, 4)                          # [B,6,4] our move types
        # --- type effectiveness eff[B,6,4]: REVEALED via CHART; UNREVEALED via expected mult ---
        eff_rev = (torch.gather(self.CHART[t1d], 2, mt) * torch.gather(self.CHART[t2d], 2, mt)
                   * torch.gather(self.ABILITY_DAMAGE_MULT[ability], 2, mt))      # [B,6,4]
        # --- bulk + maxhp: REVEALED neutral 0-EV; UNREVEALED expected (sentinel base = 0 → all from belief) ---
        neutral_def = 2.0 * d_base[..., 2] + 31.0 + 5.0                           # [B,6]
        neutral_spd = 2.0 * d_base[..., 4] + 31.0 + 5.0
        neutral_maxhp = 2.0 * d_base[..., 0] + 31.0 + 110.0
        opp_def, opp_spd, opp_maxhp, eff, alive_gate = (neutral_def, neutral_spd, neutral_maxhp,
                                                        eff_rev, defender_alive)
        if species_probs is not None:
            b = (believed > 0.5)
            e_mult = species_probs @ self.SPECIES_EXP_MULT                        # [B,6,N_TYPE_IDX]
            eff_unrev = torch.gather(e_mult, 2, mt)                               # [B,6,4]
            eff = torch.where(b[:, :, None], eff_unrev, eff_rev)
            means = self.SPECIES_SPREAD_PRIOR[..., 0]                             # [n_species,5] (atk,def,spa,spd,spe)
            e_bulk = species_probs @ means                                        # [B,6,5]
            e_base_hp = species_probs @ self.BASE_STATS[:, 0]                      # [B,6] expected base HP
            opp_def = torch.where(b, e_bulk[..., _SB_DEF], neutral_def)
            opp_spd = torch.where(b, e_bulk[..., _SB_SPD], neutral_spd)
            opp_maxhp = torch.where(b, 2.0 * e_base_hp + 31.0 + 110.0, neutral_maxhp)
            # an UNREVEALED mon is an assumed full-HP switch-in (its obs HP frac is 0/placeholder) → force alive
            alive_gate = torch.where(b, torch.ones_like(defender_alive), defender_alive)
        else:
            eff = eff_rev * (1.0 - believed)[:, :, None]                          # legacy: zero unrevealed
        D = (phys[:, None, :] * opp_def[:, :, None]
             + (1.0 - phys)[:, None, :] * opp_spd[:, :, None])                    # [B,6,4]
        opp_cur_hp = opp_hp_frac * opp_maxhp                                       # [B,6] (unrevealed: 0, pko nulled)
        core = 42.0 * bp[:, None, :] * A[:, None, :] / (D + eps) / 50.0 + 2.0     # [B,6,4]
        dmg_ns = core * stab[:, None, :] * eff * 0.925                            # [B,6,4]
        dmg_ns = dmg_ns * (bp > 0).float()[:, None, :]                            # kill the +2 floor on BP-0
        # coarse: no screens ⇒ pass None instead of allocating a [B,1,4] all-ones tensor and multiplying
        # the whole damage tensor by 1.0 (bit-identical; see `_rolls`).
        high, _low, _crit, ko = self._rolls(dmg_ns, None, opp_maxhp[:, :, None],
                                            opp_cur_hp[:, :, None], acc[:, None, :], eps)  # each [B,6,4]
        # --- per (defender, channel) best-over-our-moves of the usable rolls ---
        um = usable[:, None, :]                                                   # [B,1,4]
        phys_m = phys[:, None, :]
        spec_m = 1.0 - phys_m
        phys_high = (high * phys_m * um).amax(dim=-1)                             # [B,6]
        spec_high = (high * spec_m * um).amax(dim=-1)
        phys_pko = (ko * phys_m * um).amax(dim=-1)
        spec_pko = (ko * spec_m * um).amax(dim=-1)
        # P(KO) NULLED for UNREVEALED defenders (full-HP switch-in is ~never OHKO'd)
        revealed_slot = 1.0 - believed                                            # [B,6]
        phys_pko = phys_pko * revealed_slot
        spec_pko = spec_pko * revealed_slot
        feats = torch.stack([phys_high, spec_high, phys_pko, spec_pko], dim=-1)   # [B,6,_DMG_OUT_REFINE]
        return feats * alive_gate[:, :, None] * gate[:, None, None]               # gates

    def discrete_incoming_status(self, ctx: 'ExtractorContext',
                                 move_belief_logits: torch.Tensor,
                                 cand: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """gen3_status_trunk_v1 (INCOMING): per OUR mon, the belief-weighted `[P(major status lands),
        P(immobilizing status lands = para/frz/slp)]` from the opp active's top-`_DMG_REFINE_K` believed
        DEDICATED status moves (Thunder Wave / Toxic / Will-O-Wisp / Spore / Leech Seed). The "will I get
        statused" anticipation signal — injected onto OUR-mon tokens (the incoming mirror of the damage
        refine). Reuses the `_incoming_status_lands` DEDICATED-move immunity physics (type @ OUR def types,
        ability block, already-statused); the damaging-move secondary-para path stays at the heads. The
        major-vs-immobilize split is the decorrelation that matters for a SWITCH (a Ground pivot reads 0
        T-Wave immobilize even if it eats Toxic). Belief-weighted hard-max over K → the per-round gradient
        rides `w_topk` and sharpens the move belief toward status threats. `[B, TEAM_SIZE, _DMG_STATUS_REFINE]`."""
        B, device = ctx.batch_size, ctx.device
        ar = torch.arange(B, device=device)
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()   # [B]
        n_type = self.MOVE_STATUS_TYPE_IMMUNE.shape[1]
        # candidate selection — the SAME detached top-K over the move belief as discrete_incoming. C =
        # n_moves: the typed HP nums 355-370 are ordinary candidates (the belief scattered onto them, bare
        # 237 masked) and carry NO status (all-zero in these buffers, verified) — gen3_opp_hp_typed_candidates_v1.
        # Shared with `discrete_incoming` for the same round when the caller passes it (`refine_candidates`).
        topk_idx, w_topk = cand if cand is not None else self.refine_candidates(ctx, move_belief_logits)
        K = topk_idx.shape[1]
        inflicts = self.MOVE_INFLICTS_STATUS[topk_idx]                                    # [B,K]
        acc = self.MOVE_ACCURACY[topk_idx]                                                # [B,K]
        sidx = self.MOVE_STATUS_CAT[topk_idx]                                             # [B,K]
        blocked = self.MOVE_BLOCKED_IF_STATUSED[topk_idx]                                 # [B,K]
        ti = self.MOVE_STATUS_TYPE_IMMUNE[topk_idx]                                       # [B,K,n_type]
        # our 6 defenders' KNOWN types / ability-block / already-statused / alive
        t1d = ctx.type1_ids[:, :TEAM_SIZE]
        t2d = ctx.type2_ids[:, :TEAM_SIZE]
        abl = self.ABILITY_STATUS_BLOCK[ctx.ability1_ids[:, :TEAM_SIZE]]                   # [B,6,N_STATUS_CAT]
        our_cond = ctx.pokemon_part[:, :TEAM_SIZE,
                                    POKEMON_CONDITION_OFFSET + 1:POKEMON_CONDITION_OFFSET + 7]
        already = (our_cond.sum(-1) > 0.5).float()                                        # [B,6]
        defender_alive = (ctx.hp_and_active[:, :TEAM_SIZE, 0] > 0).float()                # [B,6]
        ti_dk = ti[:, None, :, :].expand(B, TEAM_SIZE, K, n_type)
        ti1 = torch.gather(ti_dk, 3, t1d[:, :, None, None].expand(B, TEAM_SIZE, K, 1)).squeeze(-1)
        ti2 = torch.gather(ti_dk, 3, t2d[:, :, None, None].expand(B, TEAM_SIZE, K, 1)).squeeze(-1)
        t_imm = torch.maximum(ti1, ti2)                                                   # [B,6,K]
        abl_block = torch.gather(abl, 2, sidx[:, None, :].expand(B, TEAM_SIZE, K))         # [B,6,K]
        already_block = already[:, :, None] * blocked[:, None, :]                         # [B,6,K]
        land = (inflicts[:, None, :] * acc[:, None, :] * (1.0 - t_imm)
                * (1.0 - abl_block) * (1.0 - already_block))                              # [B,6,K]
        is_immob = sum((sidx == c) for c in _IMMOBILIZE_STATUS_CATS).float().clamp(max=1.0)              # [B,K]
        w_b = w_topk[:, None, :]
        p_major = (w_b * land).amax(dim=-1)                                               # [B,6]
        p_immob = (w_b * land * is_immob[:, None, :]).amax(dim=-1)                         # [B,6]
        feats = torch.stack([p_major, p_immob], dim=-1)                                   # [B,6,_DMG_STATUS_REFINE]
        return feats * defender_alive[:, :, None] * has_opp[:, None, None]

    def discrete_outgoing_status(self, ctx: 'ExtractorContext') -> torch.Tensor:
        """gen3_status_trunk_v1 (OUTGOING): per OPP mon (REVEALED-gated), the `[P(major status from OUR
        active's status moves lands), P(immobilizing status lands)]` — the in-trunk home for the masked
        move-effect block's `status_will_land`, extended over the opp's 6 mons (the active is ALWAYS
        revealed = the deprecation requirement; revealed bench = bonus; unrevealed zeroed in v1). Reuses the
        `_status_landing` immunity physics (type @ opp types, ability revealed-exact else species prior,
        already-statused, Sleep-Clause, Substitute @ the active slot) per OUR move, reduced by category over
        our 4 moves. OUR moves are KNOWN → no belief gradient. `[B, TEAM_SIZE, _DMG_STATUS_REFINE]`."""
        B, device = ctx.batch_size, ctx.device
        ar = torch.arange(B, device=device)
        our_act = ctx.our_active_idx
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()
        our_alive = (ctx.hp_and_active[ar, our_act, 0] > 0).float()
        gate = has_opp * our_alive                                                        # [B]
        n_type = self.MOVE_STATUS_TYPE_IMMUNE.shape[1]
        # our 4 status moves (gen3_op_move_align_v1: the request-ordered obs slice, NOT the sorted-by-id
        # all_move_ids[our_act] — consistent with every other our-move op read). Output max-pools over the
        # 4 moves so the ORDER is invariant here; no legality gate (parity with _status_landing + the CPU
        # move-effect block, which both KEEP disabled moves — legality is the action mask's job).
        move_ids = ctx.our_active_req_move_ids                                             # [B,4] request order
        inflicts = self.MOVE_INFLICTS_STATUS[move_ids]                                     # [B,4]
        acc = self.MOVE_ACCURACY[move_ids]                                                # [B,4]
        sidx = self.MOVE_STATUS_CAT[move_ids]                                             # [B,4]
        is_sleep = self.MOVE_IS_SLEEP[move_ids]                                           # [B,4]
        blocked = self.MOVE_BLOCKED_IF_STATUSED[move_ids]                                 # [B,4]
        ti = self.MOVE_STATUS_TYPE_IMMUNE[move_ids]                                       # [B,4,n_type]
        is_immob = sum((sidx == c) for c in _IMMOBILIZE_STATUS_CATS).float().clamp(max=1.0)              # [B,4]
        # opp 6 defenders
        opp_t1 = ctx.type1_ids[:, TEAM_SIZE:2 * TEAM_SIZE]
        opp_t2 = ctx.type2_ids[:, TEAM_SIZE:2 * TEAM_SIZE]
        opp_ability = ctx.ability1_ids[:, TEAM_SIZE:2 * TEAM_SIZE]                         # [B,6]
        opp_species = ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE]                          # [B,6]
        revealed_slot = (1.0 - ctx.opp_believed_mask.float())                             # [B,6] 1 = revealed
        defender_alive = (ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0] > 0).float()    # [B,6]
        ti_dm = ti[:, None, :, :].expand(B, TEAM_SIZE, 4, n_type)                          # [B,6,4,n_type]
        timm1 = torch.gather(ti_dm, 3, opp_t1[:, :, None, None].expand(B, TEAM_SIZE, 4, 1)).squeeze(-1)
        timm2 = torch.gather(ti_dm, 3, opp_t2[:, :, None, None].expand(B, TEAM_SIZE, 4, 1)).squeeze(-1)
        t_imm = torch.maximum(timm1, timm2)                                               # [B,6,4]
        ab_rev = torch.gather(self.ABILITY_STATUS_BLOCK[opp_ability], 2,
                              sidx[:, None, :].expand(B, TEAM_SIZE, 4))                    # [B,6,4]
        ab_pri = torch.gather(self.SPECIES_STATUS_BLOCK_PRIOR[opp_species], 2,
                              sidx[:, None, :].expand(B, TEAM_SIZE, 4))                    # [B,6,4]
        is_rev = (opp_ability > 0).float()[:, :, None]                                     # [B,6,1]
        ability_block = is_rev * ab_rev + (1.0 - is_rev) * ab_pri                          # [B,6,4]
        opp_cond = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE,
                                    POKEMON_CONDITION_OFFSET + 1:POKEMON_CONDITION_OFFSET + 7]
        already = (opp_cond.sum(-1) > 0.5).float()                                        # [B,6]
        already_block = already[:, :, None] * blocked[:, None, :]                         # [B,6,4]
        # Sleep-Clause (global): any opp asleep via a non-Rest source → our sleep moves fail
        opp_slp = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE, POKEMON_CONDITION_OFFSET + _COND_SLP_IDX]
        opp_rest = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE, POKEMON_SLEEP_BELIEF_OFFSET]
        sleep_clause = ((opp_slp * (1.0 - opp_rest)).sum(-1) > 0.5).float()[:, None, None]  # [B,1,1]
        sleep_block = sleep_clause * is_sleep[:, None, :]                                  # [B,6,4]
        # Substitute — only the opp ACTIVE slot can hold a Sub (blocks every status move)
        has_sub = (ctx.opp_ctx_raw[:, _SUBSTITUTE_CTX_IDX] > 0.5).float()                  # [B]
        is_active = torch.zeros(B, TEAM_SIZE, device=device)
        is_active[ar, ctx.opp_active_local] = 1.0
        sub_block = (has_sub[:, None] * is_active)[:, :, None]                             # [B,6,1]
        land = (inflicts[:, None, :] * acc[:, None, :] * (1.0 - t_imm) * (1.0 - ability_block)
                * (1.0 - already_block) * (1.0 - sleep_block) * (1.0 - sub_block))         # [B,6,4]
        p_major = land.amax(dim=-1)                                                        # [B,6]
        p_immob = (land * is_immob[:, None, :]).amax(dim=-1)                               # [B,6]
        feats = torch.stack([p_major, p_immob], dim=-1)                                    # [B,6,_DMG_STATUS_REFINE]
        return feats * revealed_slot[:, :, None] * defender_alive[:, :, None] * gate[:, None, None]

    def forward(self, ctx: 'ExtractorContext', move_belief_logits: torch.Tensor,
                spread_belief: Optional[torch.Tensor] = None,
                move_latent_all: Optional[torch.Tensor] = None,
                spread_nature_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = ctx.batch_size
        device = ctx.device
        eps = 1e-6
        ar = torch.arange(B, device=device)
        opp_act = TEAM_SIZE + ctx.opp_active_local                         # [B] global opp-active slot
        # gen3_unified_spread_belief_v1: the believed opp-active stats [B,5] (atk,def,spa,spd,spe), or None
        # (→ the legacy hand-coded de-timid offense / neutral bulk constants below).
        sb = spread_belief[ar, ctx.opp_active_local] if spread_belief is not None else None

        # No-opp-active gate (forced switch / battle start / dummy zero-obs): zero the whole block.
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B]

        # --- Attacker = opp active (revealed species; hidden spread → fixed 252 EV / 31 IV / ×1.1) ---
        a_base = self.BASE_STATS[ctx.species_ids[ar, opp_act]]            # [B,6] [hp,atk,def,spa,spd,spe]
        off_const = 31.0 + 252.0 / 4.0 + 5.0                              # IV + EV/4 + 5 (legacy de-timid)
        atk = sb[:, _SB_ATK] if sb is not None else (2.0 * a_base[:, 1] + off_const) * 1.1   # [B] believed/legacy
        spa = sb[:, _SB_SPA] if sb is not None else (2.0 * a_base[:, 3] + off_const) * 1.1   # [B]
        # gen3_unified_op_physics_v1: fold the OPP active's OFFENSIVE stat-stage boosts (Dragon Dance /
        # Calm Mind / Swords Dance) into its offense — a +2 sweeper's Atk is doubled (the worst
        # damage-calc edge case). Stages read from the opp active-context; our-side read below for defence.
        opp_b_atk, opp_b_def, opp_b_spa, opp_b_spd, opp_b_spe = self._boost_stages(ctx.opp_ctx_raw)
        atk = atk * self._boost_mult(opp_b_atk)
        spa = spa * self._boost_mult(opp_b_spa)
        # gen3_unified_op_physics_v1: BURN halves the opp attacker's PHYSICAL attack (atk only; spa unhurt).
        opp_burn = ctx.pokemon_part[ar, opp_act, POKEMON_CONDITION_OFFSET + _COND_BRN_IDX]   # [B]
        atk = atk * torch.where(opp_burn > 0.5, atk.new_tensor(0.5), atk.new_tensor(1.0))
        at1 = ctx.type1_ids[ar, opp_act]                                 # [B] TypeEncoder axis
        at2 = ctx.type2_ids[ar, opp_act]

        # --- Defenders = our 6 mons (revealed → REAL spread reconstructed from the obs) ---
        d_base = self.BASE_STATS[ctx.species_ids[:, :TEAM_SIZE]]          # [B,6,6]
        spread = ctx.pokemon_part[:, :TEAM_SIZE,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spread[..., 0:6] * 31.0                                      # [B,6,6] [hp,atk,def,spa,spd,spe]
        ev = spread[..., 6:12] * 252.0
        nat = spread[..., 13:18]                                          # [B,6,5] [atk,def,spa,spd,spe]
        def_stat = (2.0 * d_base[..., 2] + iv[..., 2] + ev[..., 2] / 4.0 + 5.0) * nat[..., 1]   # [B,6]
        spd_stat = (2.0 * d_base[..., 4] + iv[..., 4] + ev[..., 4] / 4.0 + 5.0) * nat[..., 3]   # [B,6]
        # OUR ACTIVE defender's DEFENSIVE boosts (only the active mon carries boosts in gen3 — bench reset).
        our_b_atk, our_b_def, our_b_spa, our_b_spd, our_b_spe = self._boost_stages(ctx.our_ctx_raw)
        def_boost = torch.ones_like(def_stat); def_boost[ar, ctx.our_active_idx] = self._boost_mult(our_b_def)
        spd_boost = torch.ones_like(spd_stat); spd_boost[ar, ctx.our_active_idx] = self._boost_mult(our_b_spd)
        def_stat = def_stat * def_boost
        spd_stat = spd_stat * spd_boost
        maxhp = 2.0 * d_base[..., 0] + iv[..., 0] + ev[..., 0] / 4.0 + 110.0                    # [B,6]
        hp_frac = ctx.hp_and_active[:, :TEAM_SIZE, 0]                     # [B,6]
        cur_hp = hp_frac * maxhp                                          # [B,6]
        defender_alive = (hp_frac > 0).float()                           # [B,6]
        t1d = ctx.type1_ids[:, :TEAM_SIZE]                               # [B,6]
        t2d = ctx.type2_ids[:, :TEAM_SIZE]

        # Per-move belief over the real move-nums (UNMASKED — used by the believed-EFFECT block below; the
        # bare num-237 carries no effect flags, so it's not masked here). The damage CANDIDATE weights (with
        # the bare-237 presence channel masked) come from `_opp_candidate_weights` after the attribute build.
        w = torch.sigmoid(move_belief_logits[ar, ctx.opp_active_local])   # [B, n_moves]
        # --- Candidate set: C = n_moves. The 16 typed Hidden Powers are ORDINARY move-num candidates
        # (355-370, real BP 70 + type) already carrying P(present)·P(type) from the composed posterior;
        # the bare 237 (BP 0) is the masked presence channel — gen3_typed_hp_belief_v1. ---
        bp_all = self.MOVE_BP                                                                   # [n_moves]
        mty_all = self.MOVE_TYPE_IDX                                                            # [n_moves]
        phys_all = self.MOVE_PHYS                                                               # [n_moves]
        acc_all = self.MOVE_ACCURACY                                                            # [n_moves]
        fixed_all = self.MOVE_FIXED_DAMAGE                                                      # [n_moves]
        # Fixed-damage moves read BP 0 → derived category STATUS → MOVE_PHYS 0; route them onto their TYPE's
        # channel instead (Seismic Toss=Fighting=phys, Night Shade=Ghost=phys), matching the outgoing block.
        phys_all = torch.where(fixed_all > 0, self.TYPE_IS_PHYS[mty_all], phys_all)             # [n_moves]
        # gen3_unified_op_physics_v1: per-candidate WEATHER BP modifier (rain/sun × Water/Fire), [B,n_moves].
        weather_mult = self._weather_mult(ctx.weather_feature, (mty_all == _WATER_TIDX).float()[None, :],
                                          (mty_all == _FIRE_TIDX).float()[None, :])             # [B,n_moves]
        # gen3_typed_hp_belief_v1: the candidate belief weights — the typed HPs already carry
        # P(present)·P(type) from the composed posterior; only the bare-237 presence channel is masked.
        w_all = self._opp_candidate_weights(ctx, move_belief_logits)                            # [B, n_moves]

        # gen3_topk_candidates_v1: TRUNCATE the candidate axis to the top-K of the MOVE BELIEF, no
        # tail bound. The op used to price ALL ~400 move-nums per defender even though the opponent
        # runs four moves — the belief already says which candidates matter, so the sweep spent ~96%
        # of its work on candidates whose weight makes them irrelevant to every `max` downstream.
        # Selection is per-batch-row (each battle has its own opponent) and DETACHED; the gathered
        # WEIGHTS stay differentiable, so the belief gradient still rides the surviving candidates.
        # `cand_nums` maps reduced index -> real move-num, so the top-K / matrix blocks and the
        # prober's `last_topk_idx` keep reporting REAL moves. k=0 keeps the full sweep (byte-identical).
        cand_nums = None
        if self.damage_candidate_k > 0 and self.damage_candidate_k < w_all.shape[-1]:
            cand_nums = w_all.detach().topk(self.damage_candidate_k, dim=-1).indices          # [B,K]
            w_all = w_all.gather(-1, cand_nums)                                              # [B,K] differentiable
            bp_all = bp_all[cand_nums]                                                       # [B,K]
            mty_all = mty_all[cand_nums]
            phys_all = phys_all[cand_nums]
            acc_all = acc_all[cand_nums]
            fixed_all = fixed_all[cand_nums]
            weather_mult = weather_mult.gather(-1, cand_nums)                                # [B,K]
        else:
            # No truncation: broadcast the 1-D buffers to the [B,C] contract `_damage_rolls` now takes.
            _B1 = w_all.shape[0]
            bp_all = bp_all.expand(_B1, -1) if bp_all.dim() == 1 else bp_all
            mty_all = mty_all.expand(_B1, -1) if mty_all.dim() == 1 else mty_all
            phys_all = phys_all.expand(_B1, -1) if phys_all.dim() == 1 else phys_all
            acc_all = acc_all.expand(_B1, -1) if acc_all.dim() == 1 else acc_all
            fixed_all = fixed_all.expand(_B1, -1) if fixed_all.dim() == 1 else fixed_all

        # --- gen3 damage per (defender, candidate), all differentiable in w (the shared physics
        # kernel — incoming roles: attacker = opp active, defenders = our 6, OUR-side screens) ---
        our_reflect = ctx.screen_feature[:, 0:1]                                                # [B,1]
        our_light_screen = ctx.screen_feature[:, 2:3]                                           # [B,1]
        high_frac, low_frac, crit_frac, ko_ramp, high_cb, ko_cb = self._damage_rolls(
            atk, spa, at1, at2, def_stat, spd_stat, maxhp, cur_hp, t1d, t2d,
            ctx.ability1_ids[:, :TEAM_SIZE], our_reflect, our_light_screen,
            bp_all, mty_all, phys_all, acc_all, fixed_all, weather_mult, eps)

        # gen3_nature_ev_belief_v1 (--spread-belief-nature-marginalize): the believed offense folds in a single
        # E[nature_mult], so the nonlinear P(KO) THRESHOLD blurs the ×1.1/×0.9 asymmetry. Marginalise it over
        # the believed nature distribution (3-point quadrature on the candidate's one offensive stat — exact).
        # The magnitude rolls (high/low/crit) stay at the believed mean (linear → mean-field exact).
        if spread_nature_logits is not None:
            nat_probs = torch.softmax(spread_nature_logits[ar, ctx.opp_active_local], dim=-1)   # [B,25]
            ko_ramp = self._nature_marg_ko(ko_ramp, high_frac, maxhp, cur_hp, acc_all, phys_all,
                                           fixed_all, nat_probs, eps)
            ko_cb = self._nature_marg_ko(ko_cb, high_cb, maxhp, cur_hp, acc_all, phys_all,
                                         fixed_all, nat_probs, eps)

        # --- per (defender, channel): HARD max of the belief-weighted roll/KO over the candidates ---
        # The dominant believed move owns each channel (the candidate-count-robust max, NOT a diluting
        # soft-max over ~400 moves). low/high/crit are monotone in damage → the same dominant move; pko is
        # its KO probability. Each feature is `max_c w_c · value_c` on the channel.
        wb = w_all[:, None, :]                                           # [B,1,C] (belief, broadcast over defenders)
        phys_mask = phys_all[:, None, :]                                 # [B,1,C]
        spec_mask = 1.0 - phys_mask
        # Hoist the belief-weighted rolls ONCE. Each `wb * <roll>` was previously computed TWICE (once
        # per channel on its own line) and `wb * high_frac` a THIRD time below as `wf` — 5 redundant
        # [B,6,C] multiplies per forward. Keeping the per-channel MASKED tensors additionally lets
        # `_chan_acc` reuse the exact tensor its channel max was taken from. Same operands, same order,
        # same masks ⇒ bit-identical.
        wl, wh, wc, wk = wb * low_frac, wb * high_frac, wb * crit_frac, wb * ko_ramp   # [B,6,C] each
        wh_p, wh_s = wh * phys_mask, wh * spec_mask
        phys_low, spec_low = (wl * phys_mask).amax(dim=-1), (wl * spec_mask).amax(dim=-1)
        phys_high, spec_high = wh_p.amax(dim=-1), wh_s.amax(dim=-1)
        phys_crit, spec_crit = (wc * phys_mask).amax(dim=-1), (wc * spec_mask).amax(dim=-1)
        phys_pko, spec_pko = (wk * phys_mask).amax(dim=-1), (wk * spec_mask).amax(dim=-1)
        # gen3_unified_choice_band_v1: the CB-CONDITIONAL physical tail — the PHYSICAL-channel high-roll +
        # P(OHKO) computed with the opp Atk ×1.5. Same hard-max aggregation over the believed candidates;
        # special channel is CB-invariant so only the physical max is exposed (paired with p_cb below).
        phys_high_cb = self._chan_max(wb * high_cb, phys_mask)                                   # [B,6]
        phys_pko_cb = self._chan_max(wb * ko_cb, phys_mask)                                      # [B,6]
        # PER-CHANNEL accuracy + PROVENANCE of the dominant (max belief-weighted high-roll) believed move.
        # accuracy is gathered COHERENTLY at the channel's dominant-damage move (the one the rolls describe),
        # so {pko, accuracy} parameterize that threat's full outcome distribution. provenance is the dominant
        # move's belief weight (1.0 ≈ a REVEALED/pinned move, <1.0 = a usage-prior GUESS). argmax detached;
        # the gathered (acc fixed-buffer / belief weight) values carry the right gradient.
        acc_exp = acc_all[:, None, :].expand(-1, TEAM_SIZE, -1)                                  # [B,6,C]

        # `wfc` is the SAME masked tensor whose amax already produced this channel's `phys_high`/
        # `spec_high` above, so both are passed in rather than recomputed (the old form rebuilt the
        # product AND re-ran the amax per channel).
        def _chan_acc(wfc, chan_max):
            dom = wfc.argmax(dim=-1, keepdim=True)                                               # [B,6,1]
            acc = torch.gather(acc_exp, -1, dom).squeeze(-1)                                     # [B,6]
            return torch.where(chan_max > eps, acc, torch.zeros_like(acc))                       # 0 if no threat
        phys_acc = _chan_acc(wh_p, phys_high)
        spec_acc = _chan_acc(wh_s, spec_high)

        dom_idx = wh.argmax(dim=-1, keepdim=True)                                                # [B,6,1] (overall)
        provenance = torch.gather(w_all[:, None, :].expand(-1, TEAM_SIZE, -1), -1, dom_idx).squeeze(-1)
        provenance = torch.where(wh.amax(dim=-1) > eps, provenance, torch.zeros_like(provenance))
        # P(outspeed): our mon's REAL speed vs the opp active's fast-tail speed (252/+nat) — a per-mon
        # point estimate (paralysis/boosts not modelled in v1). Logistic over the stat difference.
        our_spe = (2.0 * d_base[..., 5] + iv[..., 5] + ev[..., 5] / 4.0 + 5.0) * nat[..., 4]     # [B,6]
        opp_spe = sb[:, _SB_SPE] if sb is not None else (2.0 * a_base[:, 5] + off_const) * 1.1   # [B] believed/legacy
        # gen3_unified_op_physics_v1: SPEED boosts (Agility / DD) + PARALYSIS (×0.25) on the active mons fold
        # into p_outspeed — so "I set up DD → I outspeed" / "I paralyze them → I outspeed" are both priced.
        opp_para = ctx.pokemon_part[ar, opp_act, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]       # [B]
        opp_spe = opp_spe * self._boost_mult(opp_b_spe) * torch.where(
            opp_para > 0.5, opp_spe.new_tensor(_DMG_PARA_SPEED), opp_spe.new_tensor(1.0))
        our_para = ctx.pokemon_part[ar, ctx.our_active_idx, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]  # [B]
        our_spe_mult = torch.ones_like(our_spe)                                                  # [B,6]
        our_spe_mult[ar, ctx.our_active_idx] = self._boost_mult(our_b_spe) * torch.where(
            our_para > 0.5, our_para.new_tensor(_DMG_PARA_SPEED), our_para.new_tensor(1.0))
        our_spe = our_spe * our_spe_mult
        opp_spe_std = self.SPECIES_SPREAD_PRIOR[ctx.species_ids[ar, opp_act], _SB_SPE, 1]        # [B] (#3)
        p_outspeed = self._p_outspeed(our_spe, opp_spe[:, None], opp_spe_std[:, None])           # [B,6]

        # Slot order == the named _DMG_IDX_* offsets: [phys_low, phys_high, phys_crit, phys_pko, phys_acc,
        #               spec_low, spec_high, spec_crit, spec_pko, spec_acc, outspeed, prov]
        feats = torch.stack([phys_low, phys_high, phys_crit, phys_pko, phys_acc,
                             spec_low, spec_high, spec_crit, spec_pko, spec_acc,
                             p_outspeed, provenance], dim=-1)                                     # [B,6,12]
        feats = feats * defender_alive[:, :, None] * has_opp[:, None, None]                       # gates

        # --- opp-active believed-EFFECT threat: belief weight of the most-believed move of each category ---
        # p_k = max_m (w_m·flag_mk) = P(the single most-believed move of category k) — the SAME collapse-free
        # max the chip/pko channels use. (A full-axis noisy-OR over ~400 moves saturated to ~1 from the floor
        # alone — the candidate-count dilution `_chan_max` exists to avoid.) [recovery, status, phaze, boost,
        # hazard, protect]; over the num moves only (HP is damaging → flags 0).
        # SPARSE form (see the _EFF_IDX/_EFF_VAL buffers): the dense `w[:,:,None] * FLAGS[None]` built a
        # [B, 400, 6] tensor to reduce to [B,6]. Only ≤21 of 400 moves carry any effect flag, so gather
        # those rows instead — exact (w ≥ 0, flags ∈ {0,1}, pad lanes contribute 0).
        w_eff = w[:, self._EFF_IDX] * self._EFF_VAL[None, :, :]          # [B, K, width]
        p_effect = w_eff.amax(dim=-1) * has_opp[:, None]                # [B, K], gated

        # gen3_unified_move_system_v1: per-STATUS secondary threat from the opp active's DAMAGING moves
        # (Body Slam para, Rock Slide flinch, Ice Beam freeze — the axis the binary `status` flag missed).
        # realized P(effect k) = max_m (w_m · chance_mk · acc_m) × Serene Grace(opp active). Accuracy is
        # folded (a secondary only fires on a hit — the same physics-in-the-op principle as pko: e.g. Zap
        # Cannon's 100% para × 50% acc → 0.5). NO speed coupling — flinch's move-first dependence is left
        # to attention (owner decision). Order == damage_tables.SECONDARY_COLS. (Defender Shield Dust is a
        # rare v2 follow-up — the effect block is opp-active-level, not per-defender.)
        # SPARSE form, as for p_effect above: [B,400,10] → [B,10,width≤19]. `w·acc` is formed first (the
        # same product the dense line made), then only the rows with a nonzero chance are gathered.
        w_acc = w * self.MOVE_ACCURACY[None, :]                                                  # [B, M]
        w_sec = w_acc[:, self._SEC_IDX] * self._SEC_VAL[None, :, :]                              # [B,10,width]
        opp_serene = self.ABILITY_SECONDARY_MULT[ctx.ability1_ids[ar, opp_act]]                  # [B]
        p_sec = (w_sec.amax(dim=-1) * opp_serene[:, None]).clamp(max=1.0) * has_opp[:, None]     # [B,10]

        # gen3_unified_choice_band_v1: P(opp active holds Choice Band), collapsed to 0/1 once its item is
        # revealed (item_id==CB → 1; any OTHER revealed item → 0; unrevealed id==0 → the species usage prior).
        # The op's outgoing block applies CB ×1.5 deterministically for OUR known item; here it's a belief.
        opp_item = ctx.item_ids[ar, opp_act]                                                     # [B]
        cb_prior = self.SPECIES_CB_PRIOR[ctx.species_ids[ar, opp_act]]                           # [B]
        revealed_cb = (opp_item == self.cb_item_num).float()                                     # [B]
        unrevealed = (opp_item == 0).float()                                                     # [B] all-zero id
        p_cb = (revealed_cb + (1.0 - revealed_cb) * unrevealed * cb_prior) * has_opp             # [B]
        # CB-conditional physical tail, gated like the modal per-mon feats (alive defender + opp present).
        cb_gate = defender_alive * has_opp[:, None]                                              # [B,6]
        cb_block = torch.cat([phys_high_cb * cb_gate, phys_pko_cb * cb_gate, p_cb[:, None]], dim=1)  # [B, _DMG_CB]

        block = torch.cat([feats.reshape(B, TEAM_SIZE * self.per_mon), p_effect, p_sec, cb_block], dim=1)  # [B, incoming_dim]
        # OUTGOING (our active → opp active, per-move action-aligned): appended after the incoming block
        # when enabled (widens out_dim; both projections auto-size). Reuses the shared `_rolls` physics.
        # gen3_unified_status_landing_v1: the per-OUR-move status-landing block rides the SAME outgoing
        # direction (status moves the damage block can't price), so it's appended right after it.
        if self.outgoing:
            block = torch.cat([block, self._outgoing_block(ctx, spread_belief),
                               self._status_landing(ctx)], dim=1)  # [B, out_dim]
        # gen3_unified_topk_incoming_v1: the DISCRETE top-K incoming block (opp active's K most-believed
        # moves, each with its latent identity + per-pivot damage/status). Appended LAST so the existing
        # incoming/outgoing offsets are untouched. Needs the candidate latent table (real ⊕ typed-HP),
        # built in forward_internal and passed in — the extractor hard-requires --move-latent when topk>0.
        if self.topk_k > 0 and not self.matrices_incoming:
            # The LEAN top-K block — emitted only when the rich incoming matrix is OFF (the matrix replaces
            # it at the same K). gen3_per_move_matrices_v1.
            if move_latent_all is None:
                raise ValueError("damage_topk_k>0 requires move_latent_all (the candidate latent table); "
                                 "the extractor must build it (requires --move-latent).")
            block = torch.cat([block, self._topk_block(
                ctx, w_all, high_frac, ko_ramp, acc_all, phys_all, move_latent_all,
                has_opp, defender_alive, cand_nums=cand_nums)], dim=1)  # [B, out_dim]
        # gen3_per_move_matrices_v1: the OUTGOING per-move DAMAGE MATRIX (our 4 moves × opp active+revealed
        # bench). Appended LAST so the existing incoming/outgoing/topk offsets are untouched.
        if self.matrices_outgoing:
            block = torch.cat([block, self._outgoing_matrix(ctx, spread_belief)], dim=1)  # [B, out_dim]
        # gen3_per_move_matrices_v1: the INCOMING per-move DAMAGE MATRIX (enriched top-K; mutually exclusive
        # with the lean topk block — the matrix REPLACES it at the same K). Appended LAST. Reuses the already-computed
        # rolls (low/high/crit/ko_ramp) + the candidate latent table.
        if self.matrices_incoming:
            if move_latent_all is None:
                raise ValueError("matrices_incoming requires move_latent_all (the candidate latent table); "
                                 "the extractor must build it (requires --move-latent).")
            block = torch.cat([block, self._incoming_matrix(
                ctx, w_all, low_frac, high_frac, crit_frac, ko_ramp, acc_all, phys_all, move_latent_all,
                has_opp, defender_alive, self.matrices_incoming_k, cand_nums=cand_nums)], dim=1)  # [B, out_dim]
        # gen3_per_move_matrices_v1 (v39): the TRANSPOSED outgoing matrix (our 6 mons' moves → opp active — the
        # switch-in offense read). Appended LAST so every existing offset is untouched.
        if self.matrices_outgoing_all:
            block = torch.cat([block, self._outgoing_attacker_matrix(ctx, spread_belief)], dim=1)  # [B, out_dim]
        # Read-only stash of the PRE-gain physics (the interpretable damage fractions / P(KO) / accuracy),
        # for the prober/forensic decode — the learned out_gain only rescales for the projection.
        self.last_raw_block = block.detach()
        return block * self.out_gain                                    # learnable per-channel adapter (×only)


# Effect-scalar column order (== damage_tables.MOVE_EFFECT_COLS) for the prober decode.
_DMG_EFFECT_COLS = ("recovery", "status", "phaze", "boost", "hazard", "protect")



def decode_damage_block(row, *, outgoing: bool, topk_k: int = 0, team_size: int = TEAM_SIZE,
                        matrices_outgoing: bool = False, matrices_incoming_k: int = 0,
                        matrices_outgoing_all: bool = False):
    """Decode ONE `DamageOperator.last_raw_block[i]` row (the PRE-gain physics) into a human-readable dict
    for the prober / forensic tooling — the single source of truth for the operator's output layout, mirrored
    by the TUI. Uses the named `_DMG_IDX_*` offsets: per our mon the incoming threat
    `[low,high,crit,pko,acc]×{phys,spec} + p_outspeed + provenance` in **TEAM-SLOT order** (`incoming[i]` =
    our team slot i — the op reads our defenders as `ctx.species_ids[:, :TEAM_SIZE]`; the active mon is
    whichever slot carries the active flag, NOT necessarily slot 0 — the bench slots are the safe-switch
    reads), then the 6 opp-active believed-EFFECT scalars, then (if `outgoing`) our 4
    moves' outgoing damage `[low,high,crit,pko]` + p_outspeed + per-move secondary, then the
    gen3_unified_status_landing_v1 `status_landing` block — per move `{p_land, known}` (request-slot order =
    action logits 6+k), then (if `topk_k>0`) the gen3_unified_topk_incoming_v1 DISCRETE top-K incoming block
    `incoming_topk` (the opp active's K most-believed moves, each with its latent + belief + per-pivot
    damage/status)."""
    r = [float(x) for x in row]

    def _chan(b):
        return {"low": r[b], "high": r[b + 1], "crit": r[b + 2], "pko": r[b + 3], "acc": r[b + 4]}

    incoming = []
    for i in range(team_size):
        b = i * _DMG_PER_MON
        incoming.append({
            "phys": _chan(b + _DMG_IDX_PHYS_LOW), "spec": _chan(b + _DMG_IDX_SPEC_LOW),
            "p_outspeed": r[b + _DMG_IDX_OUTSPEED], "provenance": r[b + _DMG_IDX_PROVENANCE],
        })
    eb = team_size * _DMG_PER_MON
    sb = eb + _DMG_EFFECT                              # incoming per-status secondary base
    cbb = sb + _DMG_INCOMING_SEC                       # gen3_unified_choice_band_v1: CB block base
    out = {"incoming": incoming,
           "effect": {c: r[eb + j] for j, c in enumerate(_DMG_EFFECT_COLS)},
           "incoming_secondary": {c: r[sb + j] for j, c in enumerate(_SECONDARY_COLS)},
           # CB-conditional physical tail (per our mon) + the shared P(opp holds Choice Band).
           "choice_band": {"phys_high_cb": [r[cbb + i] for i in range(team_size)],
                           "phys_pko_cb": [r[cbb + team_size + i] for i in range(team_size)],
                           "p_cb": r[cbb + 2 * team_size]},
           "outgoing": None, "incoming_topk": None}
    base = cbb + _DMG_CB                               # end of the incoming block
    if outgoing:
        ob = base                                      # outgoing damage base (after the CB block)
        osb = ob + _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE + 1   # outgoing per-move secondary base (after p_outspeed)
        slb = ob + _DMG_OUTGOING                        # status-landing base (after the whole outgoing block)
        out["outgoing"] = {
            "moves": [{"low": r[ob + k * _DMG_OUT_PER_MOVE], "high": r[ob + k * _DMG_OUT_PER_MOVE + 1],
                       "crit": r[ob + k * _DMG_OUT_PER_MOVE + 2], "pko": r[ob + k * _DMG_OUT_PER_MOVE + 3]}
                      for k in range(_DMG_OUT_N_MOVES)],
            "p_outspeed": r[ob + _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE],
            "secondary": [{c: r[osb + k * _N_SECONDARY + j] for j, c in enumerate(_SECONDARY_COLS)}
                          for k in range(_DMG_OUT_N_MOVES)],
        }
        # gen3_unified_status_landing_v1: per-OUR-move P(status lands) + a `known` bit (request-slot order).
        out["status_landing"] = [{"p_land": r[slb + k], "known": r[slb + _DMG_STATUS_N_MOVES + k]}
                                 for k in range(_DMG_STATUS_N_MOVES)]
        base = ob + _DMG_OUTGOING + _DMG_STATUS         # end of the outgoing+status block
    if topk_k > 0:
        # gen3_unified_topk_incoming_v1: opp-property block (K × [latent(32), belief, acc, is_phys]), then
        # the per-(defender, move) block (team_size × K × [high, pko, status_lands]).
        mb = base                                      # opp-property base
        db = mb + topk_k * _DMG_TOPK_MOVE              # per-defender damage base
        moves = []
        for k in range(topk_k):
            o = mb + k * _DMG_TOPK_MOVE
            moves.append({"latent": [r[o + j] for j in range(_DMG_TOPK_ID_DIM)],
                          "belief": r[o + _DMG_TOPK_IDX_W], "accuracy": r[o + _DMG_TOPK_IDX_ACC],
                          "is_phys": r[o + _DMG_TOPK_IDX_PHYS]})
        per_def = []
        for i in range(team_size):
            rows = []
            for k in range(topk_k):
                o = db + (i * topk_k + k) * _DMG_TOPK_DMG_PER
                rows.append({"high": r[o + _DMG_TOPK_IDX_HIGH], "pko": r[o + _DMG_TOPK_IDX_PKO],
                             "status_lands": r[o + _DMG_TOPK_IDX_STATUS]})
            per_def.append(rows)
        out["incoming_topk"] = {"moves": moves, "per_defender": per_def}
        base = db + team_size * topk_k * _DMG_TOPK_DMG_PER     # end of the top-K block
    out["outgoing_matrix"] = None
    if matrices_outgoing:
        # gen3_per_move_matrices_v1: our 4 moves × opp 6 mons (grouped by move), cell
        # [low,high,crit,pko,type_mult], then 6 per-opp-mon `revealed` bits.
        mvs = []
        for k in range(_DMG_OUT_N_MOVES):
            defs = []
            for dft in range(team_size):
                o = base + (k * team_size + dft) * _DMG_OMX_CELL
                defs.append({"low": r[o + _DMG_OMX_IDX_LOW], "high": r[o + _DMG_OMX_IDX_HIGH],
                             "crit": r[o + _DMG_OMX_IDX_CRIT], "pko": r[o + _DMG_OMX_IDX_PKO],
                             "type_mult": r[o + _DMG_OMX_IDX_MULT]})
            mvs.append(defs)
        rb = base + _DMG_OUT_N_MOVES * team_size * _DMG_OMX_CELL
        out["outgoing_matrix"] = {"moves": mvs, "revealed": [r[rb + dft] for dft in range(team_size)]}
        base = rb + team_size                              # end of the outgoing matrix
    out["incoming_matrix"] = None
    if matrices_incoming_k > 0:
        # gen3_per_move_matrices_v1: the INCOMING matrix — per-move header (K × [latent(32), belief, acc,
        # is_phys, effect(6), secondary(10)]), then the per-(our-mon, move) cell block (team_size × K ×
        # [low, high, crit, pko, type_mult, status_lands]).
        K = matrices_incoming_k
        hb = base                                          # header base
        cb2 = hb + K * _DMG_IMX_HEADER                     # per-(mon, move) cell base
        moves = []
        for k in range(K):
            o = hb + k * _DMG_IMX_HEADER
            moves.append({"latent": [r[o + j] for j in range(MOVE_LATENT_DIM)],
                          "belief": r[o + _DMG_IMX_HDR_W], "accuracy": r[o + _DMG_IMX_HDR_ACC],
                          "is_phys": r[o + _DMG_IMX_HDR_PHYS],
                          "effect": [r[o + _DMG_IMX_HDR_EFFECT + j] for j in range(_DMG_EFFECT)],
                          "secondary": [r[o + _DMG_IMX_HDR_SEC + j] for j in range(_N_SECONDARY)]})
        per_def = []
        for i in range(team_size):
            rows = []
            for k in range(K):
                o = cb2 + (i * K + k) * _DMG_IMX_CELL
                rows.append({"low": r[o + _DMG_IMX_IDX_LOW], "high": r[o + _DMG_IMX_IDX_HIGH],
                             "crit": r[o + _DMG_IMX_IDX_CRIT], "pko": r[o + _DMG_IMX_IDX_PKO],
                             "type_mult": r[o + _DMG_IMX_IDX_MULT], "status_lands": r[o + _DMG_IMX_IDX_STATUS]})
            per_def.append(rows)
        out["incoming_matrix"] = {"moves": moves, "per_defender": per_def}
        base = cb2 + team_size * matrices_incoming_k * _DMG_IMX_CELL    # end of the incoming matrix
    out["outgoing_matrix_all"] = None
    if matrices_outgoing_all:
        # gen3_per_move_matrices_v1 (v39): the TRANSPOSED outgoing matrix — our `team_size` mons × 4 moves
        # → the opp ACTIVE: all (attacker, move) cells [low,high,crit,pko], then the trailing per-attacker
        # `p_outspeed` block + the `alive` block.
        ab = base                                          # attacker-cell base
        pb = ab + team_size * _DMG_OAX_N_MOVES * _DMG_OAX_PER_MOVE   # p_outspeed base
        lb = pb + team_size                                # alive base
        attackers = []
        for i in range(team_size):
            mvs = []
            for k in range(_DMG_OAX_N_MOVES):
                o = ab + (i * _DMG_OAX_N_MOVES + k) * _DMG_OAX_PER_MOVE
                mvs.append({"low": r[o + _DMG_OAX_IDX_LOW], "high": r[o + _DMG_OAX_IDX_HIGH],
                            "crit": r[o + _DMG_OAX_IDX_CRIT], "pko": r[o + _DMG_OAX_IDX_PKO]})
            attackers.append({"moves": mvs, "p_outspeed": r[pb + i], "alive": r[lb + i]})
        out["outgoing_matrix_all"] = {"attackers": attackers}
    return out


