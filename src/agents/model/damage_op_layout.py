"""The DamageOperator flat-block LAYOUT — every `_DMG_*` offset/width constant, the `OpTensors`
named views (the one slicer consumers hold instead of offsets), and `decode_damage_block`.

Split out of `damage_op.py` 2026-08-16 (one responsibility per file): the operator computes the
physics; this module states the shape contract. `damage_op.py` re-exports every name here, so
historical import paths still resolve.
"""
import torch
from dataclasses import dataclass
from typing import Any, Dict, Optional
from agents.observation.constants import (
    TEAM_SIZE,
)
from agents.model.damage_tables import (N_SECONDARY as _N_SECONDARY,
                                        SECONDARY_COLS as _SECONDARY_COLS)
from agents.model.arch_constants import (  # noqa: F401  (re-export)
    ROLE_TOKEN_SIZE,
    PROJECTION_DIM,
    MOVE_NET_HIDDEN,
    MOVE_LATENT_HIDDEN,
    MOVE_LATENT_DIM,
    ROLE_ENCODER_HIDDEN,
    NET_ARCH,
    D_MODEL,
    TRANSFORMER_N_LAYERS,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_FFN_DIM,
)





_DMG_CHANNEL_FEATS = 5          # [low_roll, high_roll, crit, pko, accuracy] per gen3 type-category channel
_DMG_N_CHANNELS = 2             # physical / special (the gen3 TYPE split)
_DMG_PER_MON = _DMG_N_CHANNELS * _DMG_CHANNEL_FEATS + 2   # + p_outspeed + provenance = 12
# gen3_pair_reduce_v1: the per-(defender, candidate) cell channels handed to the Contract-W/L
# reduction rungs — [low, high, crit, ko, acc, is_phys] (see the forward's cells_pr stack).
_PAIR_REDUCE_N_CHANNELS = 6
# Named per-defender feature offsets (the single source of truth for the op's output slot layout — the
# prober decode + outgoing/safe-switch directions index by these, never a literal).
_DMG_IDX_PHYS_LOW, _DMG_IDX_PHYS_HIGH, _DMG_IDX_PHYS_CRIT, _DMG_IDX_PHYS_PKO, _DMG_IDX_PHYS_ACC = 0, 1, 2, 3, 4
_DMG_IDX_SPEC_LOW, _DMG_IDX_SPEC_HIGH, _DMG_IDX_SPEC_CRIT, _DMG_IDX_SPEC_PKO, _DMG_IDX_SPEC_ACC = 5, 6, 7, 8, 9
_DMG_IDX_OUTSPEED, _DMG_IDX_PROVENANCE = 10, 11
_DMG_ROLL_MIN = 0.85            # lowest of the 16 gen3 damage rolls ((85..100)/100); high roll = 1.0
_DMG_CHIP_CAP = 1.5             # clamp on the roll fractions (a 4× STAB hit otherwise fattens the tail)
_DMG_CRIT_CAP = 3.0            # crit can ×2 a capped high roll → a wider cap
# Opp-active believed-effect threat scalars (belief-weighted MAX over the move belief), order ==
# damage_tables MOVE_EFFECT_COLS: [recovery, status, phaze, boost, hazard, protect]. `_DMG_EFFECT` is
# the WIDTH of that column set; it survives as the per-move header field of the v35 INCOMING MATRIX
# (`_DMG_IMX_HDR_EFFECT`), which is where these bits are actually read.
#
# gen3_op_block_trim_v1 (2026-08-03): the opp-active-level COLLAPSE of both this effect block and the
# v24 per-status incoming SECONDARY block is DELETED from the incoming direction. Ledger P1/P4: the two
# are the two least-used channels in the whole op — a per-block zero→masked-KL ablation on 4000 real eval
# states put `incoming effect` at **1.2%** and `incoming secondary` at **0.1%** of the zero-whole-op
# ceiling — because both are opp-active-level scalars with NO DEFENDER AXIS, and v35's `_incoming_matrix`
# already carries the same facts PER MOVE (`_DMG_IMX_HDR_EFFECT` / `_DMG_IMX_HDR_SEC`) and PER DEFENDER
# (`status_lands`, measured KL 0.0005 vs the collapse's 0.1446). Deleting them is 16 dims off both
# projection heads and removes the whole unmasked-belief `w` read from the forward.
_DMG_EFFECT = 6
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
# NAMED stat indices — two DIFFERENT layouts coexist and a bare integer silently crosses them
# (the 2026-08-06 GIGO: pairwise_speed/pairwise_boost read index 4 = SPD as "speed" — both
# generations trained their V edge on Special Defense; the main op's index-5 paths were right).
# BASE_STATS / the obs spread iv/ev blocks: [hp, atk, def, spa, spd, spe]; the obs NATURE
# block ( _NATURE_STAT_ORDER ): [atk, def, spa, spd, spe]. Use these names, never integers.
_BS_HP, _BS_ATK, _BS_DEF, _BS_SPA, _BS_SPD, _BS_SPE = 0, 1, 2, 3, 4, 5
_NAT_ATK, _NAT_DEF, _NAT_SPA, _NAT_SPD, _NAT_SPE = 0, 1, 2, 3, 4
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
# gen3_op_block_trim_v1: the outgoing secondary block prices only the columns a gen3 move OUR side can
# actually run is able to inflict. `slp` has NO damaging move in gen3 at all (0 of ~400), and `psn`/`tox`
# are carried only by moves the team pool does not run — MEASURED over the 773 `data/teams/` teams:
# slp 0 teams, tox 0 teams, psn 1 team (0.1%). So those three columns were 12 dims of structural zero on
# every forward (4 moves × 3), and the ledger-P1 ablation prices the whole outgoing-secondary family well
# under the 1.3% these three sit inside. The INCOMING side keeps all 10 columns wherever it still reads
# them (`_DMG_IMX_HDR_SEC`, `_incoming_status_lands`) — the OPPONENT is not restricted to our team pool.
_OUT_SEC_DROP = ("slp", "psn", "tox")
_OUT_SEC_COLS = tuple(c for c in _SECONDARY_COLS if c not in _OUT_SEC_DROP)   # 7, SECONDARY_COLS order
_OUT_SEC_KEEP = tuple(_SECONDARY_COLS.index(c) for c in _OUT_SEC_COLS)       # into the 10-wide table
_N_OUT_SECONDARY = len(_OUT_SEC_COLS)                                        # 7
_DMG_OUT_SEC = _DMG_OUT_N_MOVES * _N_OUT_SECONDARY        # 4 moves × 7 secondary probs = 28
_DMG_OUTGOING = _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE + 1 + _DMG_OUT_SEC   # 16 + p_outspeed + 28 = 45

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
_LEECH_SEED_CTX_SLOT = list(_VOLATILE_SLOTS).index("leechseed")   # gen3_edge_bias_trunk_v1 (G)

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
# + the `_status_landing` pair [p_land, known] + the live per-move secondary chances = 13.
_PTR_MOVE_CELL = _DMG_OUT_PER_MOVE + 2 + _N_OUT_SECONDARY              # 4 + 2 + 7 = 13
# SWITCH cell (defender/candidate mon j, always when the op is on): the per-mon incoming row (12) + its
# CB-conditional pair [phys_high_cb_j, pko_cb_j] + the shared p_cb (broadcast — the head needs it NEXT TO
# the CB tail it conditions) = 15; + the OAX attacker row (16 cells + p_outspeed_j + alive_j = 18) when
_PTR_SWITCH_CELL_IN = _DMG_PER_MON + _DMG_CB_PER_MON + 1               # 12 + 2 + 1 = 15

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

# `damage_topk_k` is the ONE "how many opponent moves does the discrete block reason about" knob. It used
# to gate TWO blocks: the v30 LEAN top-K (`_topk_block`) and the v35 rich `_incoming_matrix`, with the
# matrix suppressing the lean one at the same K.
#
# gen3_op_block_trim_v1 (2026-08-03): the LEAN block is DELETED. It was strictly a subset of the matrix
# (header `[latent, belief, acc, is_phys]` ⊂ the matrix header, cell `[high, pko, status_lands]` ⊂ the
# matrix cell `[low, high, crit, pko, type_mult, status_lands]`), so every production config running
# `--damage-matrices incoming` suppressed it — the ledger-P1 cProfile measured `_topk_block` at **0 calls
# per forward** in that build, i.e. dead code carried through the op's hottest file. `damage_topk_k > 0`
# now means exactly "emit the incoming matrix at K"; the op REFUSES the combination that used to select
# the lean block rather than silently emitting nothing.
_DMG_TOPK_DEFAULT_K = 6         # default K when enabled (owner 2026-08-06: K=6 everywhere — 4 real
                                # moves + 2 surprise-candidate slots of truncation insurance)
# Map the 6 MAJOR-status secondary columns (par,brn,frz,slp,psn,tox) → the ABILITY_STATUS_BLOCK / status
# category axis (par→1, brn→2, frz→3, slp→4, psn→5, tox→5), for the per-pivot incoming status-landing's
# ability-immunity fold (Limber blocks Body Slam's para, etc.). SECONDARY_COLS order, first 6 cols.
_SECONDARY_MAJOR_N = 6
_SECONDARY_TO_STATUS_CAT = (1, 2, 3, 4, 5, 5)


# gen3_iterative_damage_v1: the ITERATIVE damage-refinement primitive. The full op runs ONCE post-transformer
# (a one-shot read of the FINAL belief). This recomputes a LEAN per-our-mon incoming-damage summary BETWEEN
# transformer layers — as the opp token (hence the move belief read from it) is enriched by attention — and
# injects it back onto our-mon tokens, so each layer attends over physics derived from the CURRENT belief
# `[phys_high, spec_high, phys_pko, spec_pko]` — the worst-case damage magnitude + accuracy-folded P(KO) per
# gen3 type channel (decorrelated: damage magnitude vs KO probability). `_DMG_REFINE_K` = the candidates the
# lean kernel scores per round (the opp active's top-K most-believed moves — ~50× cheaper than the full
# ~416-candidate axis, so the per-round recompute is cheap; the heavy full sweep still runs once in `forward`).
_DMG_REFINE_K = 8

# gen3_bidir_threat_trunk_v1: the OUTGOING half of the in-trunk threat refine — the SYMMETRIC mirror of the
# incoming refine. `discrete_outgoing` computes how hard OUR active's 4 known moves hit each of the opp's 6
# mons and injects the per-opp-mon summary onto the OPP token slice (so attention reasons over "how
# threatened is each opp mon by us", not just "how threatened are we"). `_DMG_OUT_REFINE` = the same 4-feat
# `[phys_high, spec_high, phys_pko, spec_pko]` per-opp-mon summary. REVEALED opp mons use real types/bulk +
# full P(KO); UNREVEALED mons use the EXPECTED-LATENT read (E[mult] via SPECIES_EXP_MULT, E[bulk]/E[maxhp]
# via SPECIES_SPREAD_PRIOR / E[base_hp], marginalized over the move-belief's P(species)) with P(KO) NULLED
# (a full-HP switch-in is ~never OHKO'd — owner decision, drops the Jensen-threshold complexity).

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


@dataclass
class OpTensors:
    """Typed, named VIEWS over the op's flat output block (`gen3_op_tensors_views_v1`,
    design_op_tensors.md steps 1-2).

    ONE place slices the layout (`DamageOperator.tensors_from_block`); every consumer reads a
    named field instead of re-deriving offsets — the slice-offset bug class becomes
    unrepresentable at the consumer. Zero-copy: every field is a view/reshape of the SAME
    storage `flat` owns, so gradients and post-`out_gain` scaling are untouched and the flat
    block remains the serialization (`decode_damage_block` / the prober keep reading it).

    Region inventory (widths under the production config):
      * `incoming_rows` + the CB tail — the per-OUR-mon incoming physics (dims 0..84);
        consumed by the prefuse injection, the pointer SWITCH cells and the critic's seed
        readout.
      * the outgoing + status-landing groups (dims 85..137, `None` when `outgoing` is off) —
        consumed by the pointer MOVE cells.
      * `incoming_matrix` / `outgoing_matrix` / OAX — opaque render regions kept ONLY as the
        serialization tail (no forward consumer since the head-concat's deletion,
        `gen3_no_concat_v1`); dropping them from the forward is design_op_tensors.md step 3
        (retrain-class: it shrinks `out_gain`).
    """
    flat: torch.Tensor                          # [B, out_dim] — post-gain serialization
    incoming_rows: torch.Tensor                 # [B, 6, per_mon]
    cb_high: torch.Tensor                       # [B, 6]
    cb_pko: torch.Tensor                        # [B, 6]
    p_cb: torch.Tensor                          # [B, 1] (shared scalar)
    out_per_move: Optional[torch.Tensor]        # [B, 4, 4] [low,high,crit,pko], request order
    out_p_outspeed: Optional[torch.Tensor]      # [B, 1]
    out_secondary: Optional[torch.Tensor]       # [B, 4, 7] (_OUT_SEC_COLS order)
    status_p_land: Optional[torch.Tensor]       # [B, 4]
    status_known: Optional[torch.Tensor]        # [B, 4]
    outgoing_matrix: Optional[torch.Tensor]     # [B, 126] opaque render (off in production)
    incoming_matrix: Optional[torch.Tensor]     # [B, imx_dim] opaque render





def decode_damage_block(row: torch.Tensor, *, outgoing: bool, team_size: int = TEAM_SIZE,
                        matrices_outgoing: bool = False, matrices_incoming_k: int = 0,
) -> Dict[str, Any]:
    """Decode ONE `DamageOperator.last_raw_block[i]` row (the PRE-gain physics) into a human-readable dict
    for the prober / forensic tooling — the single source of truth for the operator's output layout, mirrored
    by the TUI. Uses the named `_DMG_IDX_*` offsets: per our mon the incoming threat
    `[low,high,crit,pko,acc]×{phys,spec} + p_outspeed + provenance` in **TEAM-SLOT order** (`incoming[i]` =
    our team slot i — the op reads our defenders as `ctx.species_ids[:, :TEAM_SIZE]`; the active mon is
    whichever slot carries the active flag, NOT necessarily slot 0 — the bench slots are the safe-switch
    reads), then the `choice_band` tail, then (if `outgoing`) our 4 moves' outgoing damage
    `[low,high,crit,pko]` + p_outspeed + per-move secondary (the 7 live `_OUT_SEC_COLS`), then the
    gen3_unified_status_landing_v1 `status_landing` block — per move `{p_land, known}` (request-slot order =
    action logits 6+k), then the matrix blocks.

    gen3_op_block_trim_v1: the opp-active `effect` / `incoming_secondary` scalars and the lean
    `incoming_topk` block no longer exist in the row — the per-move/per-defender `incoming_matrix`
    (`matrices_incoming_k`) is where those facts live."""
    r = [float(x) for x in row]

    def _chan(b: int) -> Dict[str, float]:
        return {"low": r[b], "high": r[b + 1], "crit": r[b + 2], "pko": r[b + 3], "acc": r[b + 4]}

    incoming = []
    for i in range(team_size):
        b = i * _DMG_PER_MON
        incoming.append({
            "phys": _chan(b + _DMG_IDX_PHYS_LOW), "spec": _chan(b + _DMG_IDX_SPEC_LOW),
            "p_outspeed": r[b + _DMG_IDX_OUTSPEED], "provenance": r[b + _DMG_IDX_PROVENANCE],
        })
    cbb = team_size * _DMG_PER_MON                     # gen3_unified_choice_band_v1: CB block base
    out = {"incoming": incoming,
           # CB-conditional physical tail (per our mon) + the shared P(opp holds Choice Band).
           "choice_band": {"phys_high_cb": [r[cbb + i] for i in range(team_size)],
                           "phys_pko_cb": [r[cbb + team_size + i] for i in range(team_size)],
                           "p_cb": r[cbb + 2 * team_size]},
           "outgoing": None}
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
            "secondary": [{c: r[osb + k * _N_OUT_SECONDARY + j] for j, c in enumerate(_OUT_SEC_COLS)}
                          for k in range(_DMG_OUT_N_MOVES)],
        }
        # gen3_unified_status_landing_v1: per-OUR-move P(status lands) + a `known` bit (request-slot order).
        out["status_landing"] = [{"p_land": r[slb + k], "known": r[slb + _DMG_STATUS_N_MOVES + k]}
                                 for k in range(_DMG_STATUS_N_MOVES)]
        base = ob + _DMG_OUTGOING + _DMG_STATUS         # end of the outgoing+status block
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
    return out
