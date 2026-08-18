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
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
from agents.observation.constants import (
    TEAM_SIZE,
    POKEMON_SPREAD_OFFSET,
    POKEMON_SPREAD_DIM,
    POKEMON_CONDITION_OFFSET,
    POKEMON_COUNTER_OFFSET,  # noqa: F401  (re-export — consequence_edges_test imports it from here)
    )
# re-export — op_block_split_audit.py reads `D._N_SECONDARY` off this module
from agents.model.damage_tables import N_SECONDARY as _N_SECONDARY  # noqa: F401
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

# The flat-block layout (every _DMG_* constant, OpTensors, decode_damage_block) lives in
# `damage_op_layout.py` (split 2026-08-16); re-imported EXPLICITLY because this module is
# the historical import surface and the operator uses nearly all of it.
from agents.model.damage_op_layout import (  # noqa: F401
    OpTensors, _BOOSTS_DIM, _BS_ATK, _BS_DEF, _BS_HP, _BS_SPA, _BS_SPD, _BS_SPE, _COND_BRN_IDX,
    _COND_PAR_IDX, _COND_SLP_IDX, _DMG_CB, _DMG_CB_PER_MON, _DMG_CHANNEL_FEATS, _DMG_CHIP_CAP,
    _DMG_CRIT_CAP, _DMG_CRIT_P, _DMG_EFFECT, _DMG_IDX_OUTSPEED, _DMG_IDX_PHYS_ACC,
    _DMG_IDX_PHYS_CRIT, _DMG_IDX_PHYS_HIGH, _DMG_IDX_PHYS_LOW, _DMG_IDX_PHYS_PKO,
    _DMG_IDX_PROVENANCE, _DMG_IDX_SPEC_ACC, _DMG_IDX_SPEC_CRIT, _DMG_IDX_SPEC_HIGH,
    _DMG_IDX_SPEC_LOW, _DMG_IDX_SPEC_PKO, _DMG_IMX_CELL, _DMG_IMX_HDR_ACC, _DMG_IMX_HDR_EFFECT,
    _DMG_IMX_HDR_PHYS, _DMG_IMX_HDR_SEC, _DMG_IMX_HDR_W, _DMG_IMX_HEADER, _DMG_IMX_IDX_CRIT,
    _DMG_IMX_IDX_HIGH, _DMG_IMX_IDX_LOW, _DMG_IMX_IDX_MULT, _DMG_IMX_IDX_PKO, _DMG_IMX_IDX_STATUS,
    _DMG_N_CHANNELS, _DMG_OAX, _DMG_OAX_IDX_CRIT, _DMG_OAX_IDX_HIGH, _DMG_OAX_IDX_LOW,
    _DMG_OAX_IDX_PKO, _DMG_OAX_N_MOVES, _DMG_OAX_PER_MON, _DMG_OAX_PER_MOVE, _DMG_OMX,
    _DMG_OMX_CELL, _DMG_OMX_IDX_CRIT, _DMG_OMX_IDX_HIGH, _DMG_OMX_IDX_LOW, _DMG_OMX_IDX_MULT,
    _DMG_OMX_IDX_PKO, _DMG_OUTGOING, _DMG_OUT_N_MOVES, _DMG_OUT_PER_MOVE, _DMG_OUT_SEC,
    _DMG_PARA_SPEED, _DMG_PER_MON, _DMG_REFINE_K, _DMG_ROLL_MIN, _DMG_SPEED_SCALE,
    _DMG_SPEED_STD_K, _DMG_STATUS, _DMG_STATUS_N_MOVES, _DMG_STATUS_REFINE, _DMG_TOPK_DEFAULT_K,
    _FIRE_TIDX, _IMMOBILIZE_STATUS_CATS, _LEECH_SEED_CTX_SLOT, _NAT_ATK, _NAT_DEF, _NAT_SPA,
    _NAT_SPD, _NAT_SPE, _N_OUT_SECONDARY, _OUT_SEC_COLS, _OUT_SEC_DROP, _OUT_SEC_KEEP,
    _PAIR_REDUCE_N_CHANNELS, _PTR_MOVE_CELL, _PTR_SWITCH_CELL_IN, _SB_ATK, _SB_DEF, _SB_SPA,
    _SB_SPD, _SB_SPE, _SECONDARY_MAJOR_N, _SECONDARY_TO_STATUS_CAT, _SUBSTITUTE_CTX_IDX,
    _TypeEncoder, _VOLATILE_SLOTS, _WATER_TIDX, _dmg_imx_dim, decode_damage_block,
)




@dataclass
class OpStashes:
    """Every per-forward SIDE VALUE the op exposes, as ONE typed unit (`gen3_op_stashes_v1` —
    the OpTensors discipline applied to the stash surface). The forward replaces the whole
    container at ENTRY, so a stale batch is unrepresentable for ANY stash — the guarantee
    `last_w_all` had to hand-build (and `last_topk_idx` quietly lacked: it used to survive a
    forward in which the matrix path didn't run). Reads stay on the `op.last_*` properties (the
    documented surface, like the extractor's re-exports); WRITES go through `op.stash.<field>`
    — a stray write to a `last_*` name now fails loud instead of silently forking the state."""
    topk_idx: Optional[torch.Tensor] = None          # [B,K] move NUMS (detached; prober decode)
    topk_cand_idx: Optional[torch.Tensor] = None     # [B,K] candidate-axis indices (detached)
    topk_w: Optional[torch.Tensor] = None            # [B,K] belief weights (detached)
    w_all: Optional[torch.Tensor] = None             # [B,n_moves] candidate weights (LIVE — dedup)
    pair_cells: Optional[torch.Tensor] = None        # [B,J,K,F] un-reduced cells (alpha consumers)
    pair_gate: Optional[torch.Tensor] = None         # [B,J,1] alive x has_opp
    # gen3_pair_outcome_v1: the UNIFIED outcome vector — `pair_cells` (damage) concatenated with the
    # eight status/neutralization/tempo coordinates, on the SAME (defender, seat) axes. ONE tensor,
    # because one alpha cannot weight two (design_opponent_intent.md §5.1).
    pair_in: Optional[torch.Tensor] = None           # [B,J,K,_PAIR_OUTCOME_RAW]
    pair_seat_live: Optional[torch.Tensor] = None    # [B,K] the meaningful-K gate (unmodeled seats)
    reduced_extra: Optional[torch.Tensor] = None     # [B,6,extra] the pair-reduce rung output
    out_pko: Optional[torch.Tensor] = None           # [B,4,6] per-(our move, their mon) pko, PRE-gain
    # gen3_switch_branch_v1 (v94): the SAME outgoing grid un-collapsed — `[low, high, crit, pko,
    # type_mult]` per (our request move, their mon). `out_pko` is one channel of it and keeps its
    # name for the v85 consumers; OA2 needs the magnitude and the effectiveness too, and
    # re-slicing the flat render at a consumer would both break the one-slicer rule and read the
    # learned-GAIN-scaled render values instead of the honest physics.
    out_cells: Optional[torch.Tensor] = None         # [B,4,6,_DMG_OMX_CELL] PRE-gain
    # gen3_switch_branch_v1 (v94): P(their slot j is Ghost) — the revealed types where the slot is
    # revealed, the hidden-team species posterior marginalised through SPECIES_IS_GHOST where it is
    # not. Leak-free (publications + the Smogon prior only), and computed HERE because the op owns
    # BOTH the species posterior and the type table; a consumer deriving it would re-implement both.
    opp_p_ghost: Optional[torch.Tensor] = None       # [B,6]
    raw_block: Optional[torch.Tensor] = None         # [B,out_dim] PRE-gain block (prober decode)
    tensors: Optional['OpTensors'] = None            # the post-gain typed views

from agents.model.damage_op_blocks import DamageOperatorBlocks
from agents.model.damage_op_pairwise import DamageOperatorPairwise
from agents.model.pair_outcome import GHOST_TYPE_IDX as _GHOST_TIDX, PAIR_OUTCOME_IDX

if TYPE_CHECKING:  # no runtime import — `ctx` is only ever passed in, never constructed here
    from agents.model.extractor_ctx import ExtractorContext


class DamageOperator(DamageOperatorPairwise, DamageOperatorBlocks, torch.nn.Module):
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
    dilution a low-temperature soft-max would suffer over ~400 moves), then the `_DMG_CB` Choice-Band tail.
    gen3_op_block_trim_v1: the opp-active-level believed-EFFECT (6) and per-STATUS SECONDARY (10) scalars
    that used to follow the per-mon block are DELETED — they are defender-axis-free collapses that ledger
    P1 measured at 1.2% / 0.1% of the whole-op ceiling, and `_incoming_matrix` carries the same facts per
    move AND per defender. When `outgoing`, each of OUR 4 moves carries its secondary probabilities
    (`chance·acc × Serene Grace(us) × Shield Dust(opp)`) — "what status can this move cause, with what
    probability" — over the 7 live `_OUT_SEC_COLS` (SECONDARY_COLS minus slp/psn/tox, which no pool team's
    moveset can inflict).
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
    incoming_dim = TEAM_SIZE * _DMG_PER_MON + _DMG_CB

    def __init__(self, layout: Dict[str, Any], outgoing: bool = False, topk_k: int = 0,
                 matrices_outgoing: bool = False, matrices_incoming: bool = False,
                 prob_outspeed: bool = False,
                 candidate_k: int = 0,
                 reduce_how: str = "hard_max",
                 drop_renders: bool = False,
                 believed_lean: bool = False):
        super().__init__()
        # gen3_op_lean_forward_v1 (v86, design_op_tensors step 3): `drop_renders` removes the three
        # RENDER regions (outgoing matrix / incoming matrix / OAX) from the flat forward block — they
        # have had no forward consumer since the head-concat's deletion (gen3_no_concat_v1); every
        # value a consumer needs survives as a typed stash (`last_topk_idx`, `last_pair_cells`,
        # `last_out_pko`). The matrices' SELECTION machinery still runs; only the serialization tail
        # goes, so out_dim (and out_gain) shrink while every surviving offset is UNCHANGED (the
        # renders always appended last). `believed_lean` prices the LEAN d3 physics
        # (`_incoming_rolls`) with the believed spread instead of the legacy de-timid attacker —
        # the B-spread correctness fix applied to the last de-timid site the edges read.
        self.drop_renders = bool(drop_renders)
        self.believed_lean = bool(believed_lean)
        # gen3_pair_reduce_v1 (design_pair_reduction.md §8.1 steps 3-4): the Contract-W/L reduction
        # rungs, built BESIDE the legacy hard-max block (add-beside, §9 — never replacing it).
        # 'hard_max' (production default) builds NOTHING — no params, no state_dict keys, no forward
        # work: byte-identical. Any other rung populates `last_reduced_extra` [B,6,extra_dim] each
        # forward; DELIVERY (switch cell / prefuse / seed rows) + config wiring is gen-6 work.
        from agents.model.pair_reduce import build_pair_reducer, PAIR_REDUCE_HOWS
        if reduce_how not in PAIR_REDUCE_HOWS:
            raise ValueError(f"DamageOperator reduce_how={reduce_how!r} — one of {PAIR_REDUCE_HOWS}")
        self.reduce_how = reduce_how
        self.pair_reducer = build_pair_reducer(reduce_how, n_channels=_PAIR_REDUCE_N_CHANNELS)
        # Step 6's seam flag: the UN-reduced cells are stashed only when a downstream alpha
        # consumer asked for them. Off => None and zero extra work.
        self.stash_pair_cells: bool = False
        # gen3_pair_outcome_v1: the same seam for the UNIFIED outcome vector. Setting it implies
        # `stash_pair_cells` (the damage half IS the first six coordinates), and the extractor sets
        # both together — but the implication is asserted in the forward rather than assumed, since
        # a consumer that set only this one would otherwise get a silently narrower vector.
        self.stash_pair_outcome: bool = False
        # gen3_switch_branch_v1 (v94): the seam for the per-opp-slot GHOST marginal. Off => None
        # and zero extra work; on, it costs one `[B|B,6, n_species] @ [n_species]` matvec beside
        # the species reads `_outgoing_matrix` already does.
        self.stash_opp_ghost: bool = False
        # gen3_op_stashes_v1: ALL per-forward side values live in ONE typed container, replaced
        # at forward entry (see OpStashes). The individual docs moved onto the dataclass fields;
        # the provenance tags (candidate dedup, lean forward, tensors views) are in CHANGELOG.
        self.stash = OpStashes()
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
            CHOICE_BAND_ITEM_NUM, CHOICE_BAND_PHYS_MULT, CURSE_MOVE_NUM, TOXIC_MOVE_NUM,
            REST_MOVE_NUM, BATON_PASS_MOVE_NUM,
        )
        from agents.observation.sleep_belief import expected_free_turns
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
        self.curse_num = CURSE_MOVE_NUM                    # C1: the runtime non-Ghost Curse branch
        self.toxic_num = TOXIC_MOVE_NUM                    # C2: tox vs plain psn (they share cat 5)
        # C2 sleep consequence: E[free turns] endpoints DERIVED from the verified hazard tables
        # (sleep_belief.expected_free_turns — one source), marginalised per-mon over P(Early Bird).
        self.sleep_free_noeb = float(expected_free_turns(False, 0.0))    # 2.5
        self.sleep_free_eb = float(expected_free_turns(False, 1.0))      # 1.0
        # C3's Rest self-sleep cost (owner-prioritized): Rest sleep is DETERMINISTIC — time=3
        # fixed → EXACTLY 2 lost turns (1 with Early Bird), and our own ability is KNOWN.
        self.rest_num = REST_MOVE_NUM
        self.rest_sleep_noeb = float(expected_free_turns(True, 0.0))     # 2.0 exactly
        self.rest_sleep_eb = float(expected_free_turns(True, 1.0))       # 1.0 exactly
        self.baton_num = BATON_PASS_MOVE_NUM                             # C5's receiver-axis edge
        # gen3_unified_topk_incoming_v1: secondary-col → status-category map for the per-pivot incoming
        # status-landing's ability-immunity fold (non-persistent — pure constant).
        self.register_buffer("_SEC_CAT_IDX", torch.tensor(_SECONDARY_TO_STATUS_CAT, dtype=torch.long),
                             persistent=False)
        # gen3_op_block_trim_v1: the OUR-side secondary columns the outgoing block prices. `slp`/`psn`/`tox`
        # are dropped — no gen3 damaging move inflicts sleep at all, and the psn/tox carriers appear on
        # 1 / 0 of the 773 pool teams, so those 12 dims were structural zeros. Non-persistent (pure data).
        self.register_buffer("_OUT_SEC_KEEP_IDX", torch.tensor(_OUT_SEC_KEEP, dtype=torch.long),
                             persistent=False)
        # OUTGOING direction (our active → opp active, per-move action-aligned): off by default. When on,
        # the op ALSO emits the _DMG_OUTGOING block (widens out_dim → both projections auto-size).
        self.outgoing = outgoing
        # The discrete incoming move-space K (0 = off): how many of the opp active's most-believed candidate
        # moves the `_incoming_matrix` surfaces INDIVIDUALLY. Requires the caller to pass `move_latent_all`
        # to forward (enforced at the extractor: needs --move-latent).
        self.topk_k = topk_k
        # gen3_per_move_matrices_v1: the OUTGOING per-move DAMAGE MATRIX (our 4 moves × opp active+revealed
        # bench). Off by default; when on the op ALSO emits the `_DMG_OMX` block (widens out_dim → both
        # projections auto-size). Requires the op's physics buffers (always present). The legacy single-active
        # `_outgoing_block` (`outgoing`) is a SUBSET — running the matrix supersedes it (a run uses one).
        self.matrices_outgoing = matrices_outgoing
        # gen3_per_move_matrices_v1: the INCOMING per-move DAMAGE MATRIX — the ONLY consumer of `topk_k`
        # since gen3_op_block_trim_v1 deleted the lean top-K block it superseded. K = `topk_k` (try 4/5/6),
        # defaulting to _DMG_TOPK_DEFAULT_K (5) if unset. Requires move_latent (the latent gather), enforced
        # at the extractor.
        self.matrices_incoming = matrices_incoming
        self.matrices_incoming_k = (topk_k if topk_k > 0 else _DMG_TOPK_DEFAULT_K) if matrices_incoming else 0
        if topk_k > 0 and not matrices_incoming:
            # Fail loud rather than silently emit nothing: pre-trim this combination selected the lean
            # top-K block, which no longer exists. `--damage-topk K` now means "the incoming matrix at K".
            raise ValueError(
                f"damage_topk_k={topk_k} requires matrices_incoming=True (gen3_op_block_trim_v1). The lean "
                "top-K block was deleted — the v35 INCOMING MATRIX is its strict superset and was already "
                "suppressing it in every production config (measured 0 calls/forward). Pass "
                "--damage-matrices incoming (or both) alongside --damage-topk, or set --damage-topk 0.")
        # The OUTGOING direction carries the per-move damage block + the gen3_unified_status_landing_v1
        # status-landing block (both action-aligned, our active → opp). Off ⇒ neither → baseline byte-identical.
        _renders = not drop_renders                     # gen3_op_lean_forward_v1: the serialization tail
        self.out_dim = (self.incoming_dim + (_DMG_OUTGOING + _DMG_STATUS if outgoing else 0)
                        + (_DMG_OMX if matrices_outgoing and _renders else 0)
                        + (_dmg_imx_dim(self.matrices_incoming_k)
                           if matrices_incoming and _renders else 0))
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
        _cb0 = TEAM_SIZE * self.per_mon
        gain[_cb0:_cb0 + TEAM_SIZE] = 1.0 / 1.5                       # the phys_high_cb sub-block
        if outgoing:
            # outgoing block: per move [low, high, crit, pko] (same roll scaling), then p_outspeed.
            out_move_init = torch.tensor([1.0 / 1.5, 1.0 / 1.5, 1.0 / 3.0, 1.0])
            gain[self.incoming_dim:self.incoming_dim + _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE] = \
                out_move_init.repeat(_DMG_OUT_N_MOVES)
            # the trailing p_outspeed, the per-move secondary block, and the gen3_unified_status_landing_v1
            # status block (p_land/known) all stay at gain 1.0 — they are already probabilities in [0,1].
        if matrices_outgoing and _renders:
            # gen3_per_move_matrices_v1: the outgoing-matrix tail. Per (move, opp mon) cell
            # [low, high, crit, pko, type_mult] — scale low/high (÷1.5), crit (÷3.0), type_mult (cap 4× → ÷4);
            # pko already [0,1]. The trailing 6 `revealed` bits stay 1.0.
            _omx0 = self.incoming_dim + (_DMG_OUTGOING + _DMG_STATUS if outgoing else 0)
            _cell_init = torch.tensor([1.0 / 1.5, 1.0 / 1.5, 1.0 / 3.0, 1.0, 1.0 / 4.0])  # low,high,crit,pko,mult
            gain[_omx0:_omx0 + _DMG_OUT_N_MOVES * TEAM_SIZE * _DMG_OMX_CELL] = \
                _cell_init.repeat(_DMG_OUT_N_MOVES * TEAM_SIZE)
        if matrices_incoming and _renders:
            # gen3_per_move_matrices_v1: the incoming-matrix tail. The per-move header (K × [latent(32),
            # belief, acc, is_phys, effect(6), secondary(10)]) stays gain 1.0 (latent normalized downstream;
            # the rest are [0,1]). The per-(mon, move) cell [low,high,crit,pko,type_mult,status] scales
            # low/high (÷1.5), crit (÷3), type_mult (cap 4× → ÷4); pko/status are [0,1].
            _imx0 = (self.incoming_dim + (_DMG_OUTGOING + _DMG_STATUS if outgoing else 0)
                     + (_DMG_OMX if matrices_outgoing else 0)
                     + self.matrices_incoming_k * _DMG_IMX_HEADER)            # after the per-move header
            _imx_cell = torch.tensor([1.0 / 1.5, 1.0 / 1.5, 1.0 / 3.0, 1.0, 1.0 / 4.0, 1.0])
            gain[_imx0:_imx0 + TEAM_SIZE * self.matrices_incoming_k * _DMG_IMX_CELL] = \
                _imx_cell.repeat(TEAM_SIZE * self.matrices_incoming_k)
        self.out_gain = torch.nn.Parameter(gain)


    # gen3_op_stashes_v1 — the READ surface over the typed stash container (the re-export
    # convention: reads stay valid everywhere; writes must go through `self.stash`).
    @property
    def last_topk_idx(self) -> Optional[torch.Tensor]: return self.stash.topk_idx
    @property
    def last_topk_cand_idx(self) -> Optional[torch.Tensor]: return self.stash.topk_cand_idx
    @property
    def last_topk_w(self) -> Optional[torch.Tensor]: return self.stash.topk_w
    @property
    def last_w_all(self) -> Optional[torch.Tensor]: return self.stash.w_all
    @property
    def last_pair_cells(self) -> Optional[torch.Tensor]: return self.stash.pair_cells
    @property
    def last_pair_gate(self) -> Optional[torch.Tensor]: return self.stash.pair_gate
    @property
    def last_pair_in(self) -> Optional[torch.Tensor]: return self.stash.pair_in
    @property
    def last_pair_seat_live(self) -> Optional[torch.Tensor]: return self.stash.pair_seat_live
    @property
    def last_reduced_extra(self) -> Optional[torch.Tensor]: return self.stash.reduced_extra
    @property
    def last_out_pko(self) -> Optional[torch.Tensor]: return self.stash.out_pko
    @property
    def last_out_cells(self) -> Optional[torch.Tensor]: return self.stash.out_cells
    @property
    def last_opp_p_ghost(self) -> Optional[torch.Tensor]: return self.stash.opp_p_ghost
    @property
    def last_raw_block(self) -> Optional[torch.Tensor]: return self.stash.raw_block
    @property
    def last_tensors(self) -> Optional['OpTensors']: return self.stash.tensors

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
        passed the learned posterior while ``refine_candidates`` did not, so the candidate-selection
        consumers silently priced HP off the Smogon prior while the head block priced it off the learned
        belief — and in the (then default) `off` mode the source was the obs ``hp_probs``, which is
        all-zero until the opponent actually FIRES Hidden Power, so a REVEALED HP was priced as
        nonexistent. There is now exactly one HP posterior per forward and every consumer reads it.

        The mask keeps the bare 237 out of the candidate set (belt-and-braces: the composition already
        drives it to ~0) since it is a belief bookkeeping channel with BP 0, never a real move."""
        B, device = ctx.batch_size, ctx.device
        ar = torch.arange(B, device=device)
        w = torch.sigmoid(move_belief_logits[ar, ctx.opp_active_local])               # [B, n_moves] (typed)
        return w * self.HP_CAND_MASK[None, :]                                         # zero the 237 presence channel

    def _chan_max(self, value: torch.Tensor, channel_mask: torch.Tensor,
                  how: str = "hard_max") -> torch.Tensor:
        """THE arity-2 → arity-1 REDUCTION SITE (design_op_tensors.md §3.2 — `REDUCE(pair_in,
        over=MOVE_AXIS, how=…)`). Every "collapse the believed-move axis onto a defender" in the
        op routes here, so the roadmap's separately-designed alternatives are SETTINGS of this
        one knob, not features: `hard_max` (today — the belief-weighted amax below),
        `belief_weighted_mean` (the un-maxed marginal), `conditional(λ)` (OA1),
        `learned_attention(k)` (PV-as-reduction). Only `hard_max` is implemented; adding a
        setting is an A/B at THIS call site with no new plumbing (the §5 step-5 gate).

        hard_max: `value` [B,6,C] (≥0), `channel_mask` [1,1,C] (1=on-channel). Off-channel
        candidates zeroed; `amax` returns the max on-channel belief-weighted value (or 0).
        Differentiable via the argmax subgradient — the dominant move's belief weight gets the
        gradient — and NOT diluted the way a low-temperature softmax over a wide candidate
        sweep would be (historically ~400 candidates; K=6 today, so that objection is ~6-way
        and much weaker than when written)."""
        if how != "hard_max":
            raise NotImplementedError(f"REDUCE how={how!r} — only 'hard_max' is implemented at "
                                      "this legacy site. The Contract-W/L rungs (belief_mean / "
                                      "learned / deepsets / multi) live in pair_reduce.py, built "
                                      "via DamageOperator(reduce_how=…) and stashed on "
                                      "last_reduced_extra — design_pair_reduction.md §8.1")
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

    def _rolls(self, dmg_ns: torch.Tensor, screen: Optional[torch.Tensor], maxhp: torch.Tensor,
               cur_hp: torch.Tensor, acc: torch.Tensor,
               eps: float = 1e-6) -> Tuple[torch.Tensor, ...]:
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

    def _damage_rolls(self, atk: torch.Tensor, spa: torch.Tensor, at1: torch.Tensor, at2: torch.Tensor,
                      def_stat: torch.Tensor, spd_stat: torch.Tensor, maxhp: torch.Tensor,
                      cur_hp: torch.Tensor, t1d: torch.Tensor, t2d: torch.Tensor, ability1: torch.Tensor,
                      reflect: torch.Tensor, light_screen: torch.Tensor,
                      bp_all: torch.Tensor, mty_all: torch.Tensor, phys_all: torch.Tensor,
                      acc_all: torch.Tensor, fixed_all: torch.Tensor, weather_mult: torch.Tensor,
                      eps: float = 1e-6) -> Tuple[torch.Tensor, ...]:
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

    # ------------------------------------------------------------------ pointer-native action head cells
    @property
    def pointer_move_cell_dim(self) -> int:
        """Per-request-slot cell width for the pointer MOVE scorer (0 when the outgoing direction is off —
        the head's Linear in_features are fixed by the build-time toggle set, the op's own convention)."""
        return _PTR_MOVE_CELL if self.outgoing else 0

    @property
    def pointer_switch_cell_dim(self) -> int:
        """Per-candidate-mon cell width for the pointer SWITCH scorer."""
        return _PTR_SWITCH_CELL_IN

    def tensors_from_block(self, damage_block: torch.Tensor) -> OpTensors:
        """THE single slicer of the flat block's layout (`gen3_op_tensors_views_v1`) — every
        region as a named zero-copy view (see `OpTensors`). A pure function of the passed
        tensor: it serves the post-gain forward output, constructed test rows, and the
        offset-parity gates alike (`decode_damage_block` stays the human-readable mirror)."""
        B = damage_block.shape[0]
        inc = damage_block[:, :TEAM_SIZE * _DMG_PER_MON].reshape(B, TEAM_SIZE, _DMG_PER_MON)
        cb0 = TEAM_SIZE * _DMG_PER_MON
        cb_high = damage_block[:, cb0:cb0 + TEAM_SIZE]                                   # [B,6]
        cb_pko = damage_block[:, cb0 + TEAM_SIZE:cb0 + 2 * TEAM_SIZE]                    # [B,6]
        p_cb = damage_block[:, cb0 + 2 * TEAM_SIZE:cb0 + 2 * TEAM_SIZE + 1]              # [B,1] shared
        out_per_move = out_p_outspeed = out_secondary = None
        status_p_land = status_known = None
        pos = self.incoming_dim
        if self.outgoing:
            out_per_move = damage_block[:, pos:pos + _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE].reshape(
                B, _DMG_OUT_N_MOVES, _DMG_OUT_PER_MOVE)                                  # move-major [B,4,4]
            sp0 = pos + _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE
            out_p_outspeed = damage_block[:, sp0:sp0 + 1]                                # [B,1]
            out_secondary = damage_block[:, sp0 + 1:sp0 + 1 + _DMG_OUT_SEC].reshape(
                B, _DMG_OUT_N_MOVES, _N_OUT_SECONDARY)                                   # [B,4,7]
            st0 = pos + _DMG_OUTGOING
            status_p_land = damage_block[:, st0:st0 + _DMG_STATUS_N_MOVES]               # [B,4]
            status_known = damage_block[
                :, st0 + _DMG_STATUS_N_MOVES:st0 + 2 * _DMG_STATUS_N_MOVES]              # [B,4]
            pos += _DMG_OUTGOING + _DMG_STATUS
        outgoing_matrix = None
        if self.matrices_outgoing and not self.drop_renders:
            outgoing_matrix = damage_block[:, pos:pos + _DMG_OMX]
            pos += _DMG_OMX
        incoming_matrix = None
        if self.matrices_incoming and not self.drop_renders:
            imx_dim = _dmg_imx_dim(self.matrices_incoming_k)
            incoming_matrix = damage_block[:, pos:pos + imx_dim]
            pos += imx_dim
        if pos != self.out_dim:
            raise RuntimeError(
                f"OpTensors layout walk ended at {pos}, out_dim is {self.out_dim} — a region "
                "was added to the flat block without a view here (the layout has ONE owner).")
        return OpTensors(
            flat=damage_block, incoming_rows=inc, cb_high=cb_high, cb_pko=cb_pko, p_cb=p_cb,
            out_per_move=out_per_move, out_p_outspeed=out_p_outspeed,
            out_secondary=out_secondary, status_p_land=status_p_land,
            status_known=status_known, outgoing_matrix=outgoing_matrix,
            incoming_matrix=incoming_matrix)

    def pointer_cells(self, damage_block: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """gen3_pointer_native_v1: the flat damage block as PER-ACTION cells for the pointer head.
        Since `gen3_op_tensors_views_v1` the slicing lives in `tensors_from_block` (the layout's
        one owner); this just assembles the per-action concats from the named views.

        Returns ``(move_cells [B,4,pointer_move_cell_dim], switch_cells [B,6,pointer_switch_cell_dim])``:
          * move cell k (REQUEST-slot order == action logit 6+k, the `gen3_op_move_align_v1` guarantee —
            `_outgoing_block`/`_status_landing` read `ctx.our_active_req_move_ids`, the same id source the
            pointer token permutation matches against): `[low, high, crit, pko, p_land, known, sec×7]`
            (the 7 live secondary columns — see `_OUT_SEC_COLS`).
          * switch cell j: the incoming per-defender row (12) + `[phys_high_cb_j, pko_cb_j, p_cb]` +
        Pure slicing of the SAME tensor the seed readout consumes (post-gain), so the pointer path and
        the critic's window can never disagree on a value."""
        B = damage_block.shape[0]
        t = self.tensors_from_block(damage_block)
        # --- switch cells: incoming per-mon rows + the CB tail ---
        switch_parts = [t.incoming_rows, t.cb_high[:, :, None], t.cb_pko[:, :, None],
                        t.p_cb[:, None, :].expand(B, TEAM_SIZE, 1)]
        switch_cells = torch.cat(switch_parts, dim=2)                                    # [B,6,Cs]
        # --- move cells: the outgoing damage stack + status landing + per-move secondaries ---
        if not self.outgoing:
            return damage_block.new_zeros(B, _DMG_OUT_N_MOVES, 0), switch_cells
        # The `if not self.outgoing` early-return above is exactly the condition under which
        # `tensors_from_block` leaves these four views None — an invariant across two objects
        # that no narrowing can express.
        move_cells = torch.cat([t.out_per_move, t.status_p_land[:, :, None],  # type: ignore[index,list-item]
                                t.status_known[:, :, None], t.out_secondary], dim=2)  # type: ignore[index,list-item]
        return move_cells, switch_cells                                                  # [B,4,13], [B,6,Cs]

    def forward(self, ctx: 'ExtractorContext', move_belief_logits: torch.Tensor,
                spread_belief: Optional[torch.Tensor] = None,
                move_latent_all: Optional[torch.Tensor] = None,
                species_probs: Optional[torch.Tensor] = None,
                item_cb_prob: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the full post-gain damage block [B, out_dim] from the beliefs. `move_belief_logits`
        [B,6,M] (already typed-HP composed), `spread_belief` [B,6,5] believed opp stats, `move_latent_all`
        [n_moves,MOVE_LATENT_DIM] (top-K identity source), `species_probs` [B,6,S] the T0 species prior,
        `item_cb_prob` [B,6] P(Choice Band). Also populates every per-forward stash on `self.stash`."""
        self.stash = OpStashes()          # gen3_op_stashes_v1: ONE reset, no stash can go stale
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
        self.stash.w_all = w_all                     # gen3_op_candidate_dedup_v1: same-forward reuse

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
        def _chan_acc(wfc: torch.Tensor, chan_max: torch.Tensor) -> torch.Tensor:
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

        # gen3_pair_reduce_v1: the Contract-W/L rungs read the SAME per-(defender, candidate) cell
        # tensors the hard max just reduced — [low, high, crit, ko, acc, is_phys] per (j, c) — plus
        # the belief w, and stash their per-defender rows. None (production 'hard_max') skips
        # entirely; nothing below this block changes at any `reduce_how`.
        # gen3_intent_value_reduce_v1 (step 6): the SAME un-reduced cells the pair-reducer consumes
        # are stashed so a LATER tier can reduce them by alpha. This is the only shape step 6 can
        # take: alpha is scored from the E4 seats and the CLS pools, both computed DOWNSTREAM of
        # this operator, so it cannot weight a reduction that happens here — the design doc's
        # "swap `_chan_max`'s how=" is unbuildable for that reason, not for want of a knob.
        # Stashing is free (no math, no gradient path) and the op's own reduction is untouched.
        if self.pair_reducer is not None or self.stash_pair_cells:
            cells_pr = torch.stack([
                low_frac, high_frac, crit_frac, ko_ramp,
                acc_exp, phys_mask.expand(-1, TEAM_SIZE, -1)], dim=-1)                            # [B,6,C,6]
            _gate = defender_alive[:, :, None] * has_opp[:, None, None]
            # The alignment CANNOT happen here: `cells_pr`'s C axis is the FULL candidate move
            # space, alpha's seats are the TOP-K, and the top-K index is not computed until
            # `_incoming_matrix` runs further down this same forward. Hold the raw cells and align
            # after it — pairing the two axes positionally would mis-weight every term while every
            # shape check still passed (`op move-order` bug class).
            _pr_cells_raw, _pr_gate_raw = (cells_pr, _gate) if self.stash_pair_cells else (None, None)
            self.stash.reduced_extra = ((self.pair_reducer(w_all, cells_pr) * _gate)
                                       if self.pair_reducer is not None else None)
        # (no else-clear needed: gen3_op_stashes_v1's entry reset already left every field None)

        # gen3_op_block_trim_v1: the opp-active-level believed-EFFECT scalars (6) and per-STATUS SECONDARY
        # scalars (10) that used to sit here are DELETED. Both were belief-weighted maxes with NO DEFENDER
        # AXIS, and the ledger-P1 zero→masked-KL ablation measured them at 1.2% / 0.1% of the whole-op
        # ceiling — the two least-used channels in the operator. The same facts survive PER MOVE and PER
        # DEFENDER in `_incoming_matrix`'s header (`_DMG_IMX_HDR_EFFECT` / `_DMG_IMX_HDR_SEC`) and cell
        # (`status_lands`), which is where the head actually reads them (KL 0.0005 vs the collapse's 0.1446).
        # Deleting them also removes the whole UNMASKED-belief read (`w = sigmoid(...)`) from the forward —
        # the candidate weights `w_all` above are now the op's only belief read.

        # gen3_unified_choice_band_v1: P(opp active holds Choice Band), collapsed to 0/1 once its item is
        # revealed (item_id==CB → 1; any OTHER revealed item → 0; unrevealed id==0 → the species usage prior).
        # The op's outgoing block applies CB ×1.5 deterministically for OUR known item; here it's a belief.
        opp_item = ctx.item_ids[ar, opp_act]                                                     # [B]
        # gen3_item_belief_v1: when the ITEM BELIEF runs, its published P(item==CB) at the ACTIVE
        # slot replaces the static usage scalar in the UNREVEALED branch — the exactness gating
        # (revealed → 0/1) stays HERE either way, so the belief only ever moves the prior factor.
        # None (flag off / every historical caller) is byte-identical to the static prior.
        if item_cb_prob is not None:
            cb_prior = item_cb_prob[ar, ctx.opp_active_local]                                    # [B]
        else:
            cb_prior = self.SPECIES_CB_PRIOR[ctx.species_ids[ar, opp_act]]                       # [B]
        revealed_cb = (opp_item == self.cb_item_num).float()                                     # [B]
        unrevealed = (opp_item == 0).float()                                                     # [B] all-zero id
        p_cb = (revealed_cb + (1.0 - revealed_cb) * unrevealed * cb_prior) * has_opp             # [B]
        # CB-conditional physical tail, gated like the modal per-mon feats (alive defender + opp present).
        cb_gate = defender_alive * has_opp[:, None]                                              # [B,6]
        cb_block = torch.cat([phys_high_cb * cb_gate, phys_pko_cb * cb_gate, p_cb[:, None]], dim=1)  # [B, _DMG_CB]

        block = torch.cat([feats.reshape(B, TEAM_SIZE * self.per_mon), cb_block], dim=1)  # [B, incoming_dim]
        # OUTGOING (our active → opp active, per-move action-aligned): appended after the incoming block
        # when enabled (widens out_dim; both projections auto-size). Reuses the shared `_rolls` physics.
        # gen3_unified_status_landing_v1: the per-OUR-move status-landing block rides the SAME outgoing
        # direction (status moves the damage block can't price), so it's appended right after it.
        if self.outgoing:
            block = torch.cat([block, self._outgoing_block(ctx, spread_belief),
                               self._status_landing(ctx)], dim=1)  # [B, out_dim]
        # gen3_per_move_matrices_v1: the OUTGOING per-move DAMAGE MATRIX (our 4 moves × opp active+revealed
        # bench). Appended LAST so the existing incoming/outgoing offsets are untouched.
        # gen3_switch_branch_v1 (v94): P(their slot j is Ghost). Independent of the matrices (it
        # reads only the species posterior and the revealed types), so it sits outside the block
        # below — but it is the SAME `unrevealed_species_probs` those reads use, which is what
        # makes "the arrival that blocks Rapid Spin" and "the arrival our move lands on" the same
        # belief rather than two.
        if self.stash_opp_ghost:
            _sp = self.unrevealed_species_probs(ctx, species_probs)      # [B,S] | [B,6,S]
            _pg = _sp @ self.SPECIES_IS_GHOST                            # [B]   | [B,6]
            if _pg.dim() == 1:
                _pg = _pg[:, None].expand(B, TEAM_SIZE)
            _og = slice(TEAM_SIZE, 2 * TEAM_SIZE)
            _rev_ghost = ((ctx.type1_ids[:, _og] == _GHOST_TIDX)
                          | (ctx.type2_ids[:, _og] == _GHOST_TIDX)).float()          # [B,6]
            # A REVEALED slot's typing is fact, so the belief is used only where there is nothing
            # to know — the same revealed/unrevealed split `_outgoing_matrix` applies to bulk.
            self.stash.opp_p_ghost = torch.where(ctx.opp_believed_mask, _pg, _rev_ghost)
        if self.matrices_outgoing:
            _omx = self._outgoing_matrix(ctx, spread_belief, species_probs=species_probs)
            # gen3_op_lean_forward_v1: the typed pko stash consumers read (pre-gain — honest
            # probabilities, not the learned-gain-scaled render values).
            # gen3_switch_branch_v1: the WHOLE cell grid is stashed and `out_pko` becomes a view of
            # it — one reshape, not two, so the OA2 magnitudes and the v85 boom pko can never
            # describe different worlds.
            self.stash.out_cells = _omx[:, :_DMG_OUT_N_MOVES * TEAM_SIZE * _DMG_OMX_CELL].reshape(
                B, _DMG_OUT_N_MOVES, TEAM_SIZE, _DMG_OMX_CELL)
            self.stash.out_pko = self.stash.out_cells[..., _DMG_OMX_IDX_PKO]
            if not self.drop_renders:
                block = torch.cat([block, _omx], dim=1)
        # gen3_per_move_matrices_v1: the INCOMING per-move DAMAGE MATRIX — the op's ONLY discrete
        # incoming move-space block since gen3_op_block_trim_v1 deleted the lean top-K it superseded.
        # Appended LAST. Reuses the already-computed rolls (low/high/crit/ko_ramp) + the candidate latent table.
        if self.matrices_incoming:
            if move_latent_all is None:
                raise ValueError("matrices_incoming requires move_latent_all (the candidate latent table); "
                                 "the extractor must build it (requires --move-latent).")
            _imx = self._incoming_matrix(
                ctx, w_all, low_frac, high_frac, crit_frac, ko_ramp, acc_all, phys_all, move_latent_all,
                has_opp, defender_alive, self.matrices_incoming_k, cand_nums=cand_nums)
            # gen3_op_lean_forward_v1: under drop_renders the matrix's SELECTION side effects
            # (last_topk_idx / last_topk_cand_idx — the axis alpha's seats align to) are the whole
            # point of the call; only the flat render is dropped.
            if not self.drop_renders:
                block = torch.cat([block, _imx], dim=1)  # [B, out_dim]
        # gen3_intent_value_reduce_v1 (step 6): NOW the top-K candidate index exists, so the held
        # cells can be gathered onto alpha's seat axis. Done in the op because the op owns both
        # objects; a consumer doing it would be guessing at an internal ordering.
        if self.stash_pair_cells and _pr_cells_raw is not None:
            _ci = self.last_topk_cand_idx
            if _ci is None:
                raise RuntimeError(
                    "stash_pair_cells is on but no top-K candidate index was recorded — the cells "
                    "cannot be aligned to alpha's seats. Step 6 requires damage_topk_k>0 (and the "
                    "incoming matrix that computes it).")
            _idx = _ci[:, None, :, None].expand(
                _pr_cells_raw.shape[0], _pr_cells_raw.shape[1], _ci.shape[-1],
                _pr_cells_raw.shape[-1])
            self.stash.pair_cells = _pr_cells_raw.gather(2, _idx)              # [B,J,K,F]
            self.stash.pair_gate = _pr_gate_raw
            # gen3_pair_outcome_v1: the UNIFIED outcome vector. The damage cells just aligned to
            # alpha's seat axis are its first six coordinates; the eight status / neutralization /
            # tempo coordinates are computed on the SAME `last_topk_idx` selection and concatenated,
            # so `pair_in` is ONE tensor over ONE (defender, seat) grid. That unification IS the
            # prerequisite design_opponent_intent.md §5.1 names: "damage and status are computed in
            # two functions with two reductions, and one alpha cannot weight two tensors."
            if self.stash_pair_outcome:
                _ti = self.last_topk_idx
                if _ti is None:
                    raise RuntimeError(
                        "stash_pair_outcome is on but no top-K move-num selection was recorded — "
                        "the status coordinates have no seat axis to be computed on. Requires "
                        "damage_topk_k>0 (and the incoming matrix that computes it).")
                _dmg = self.stash.pair_cells
                _extra = self.pair_outcome_coords(
                    ctx, _ti, _dmg[..., PAIR_OUTCOME_IDX["high"]],
                    our_spe, opp_spe, opp_spe_std, d_base)
                self.stash.pair_in = torch.cat([_dmg, _extra], dim=-1)         # [B,J,K,RAW]
                # (`pair_seat_live` — the unmodeled-seat mask alpha must spend no mass on — is
                # stashed by `_incoming_matrix`, which is where the meaningful-K gate is already
                # computed. One computation, one home.)
                if self.stash.pair_seat_live is None:
                    raise RuntimeError(
                        "stash_pair_outcome is on but the incoming matrix stashed no seat-liveness "
                        "gate — an unmodeled 5th+ seat would be given alpha mass it cannot carry.")

        # gen3_per_move_matrices_v1 (v39): the TRANSPOSED outgoing matrix (our 6 mons' moves → opp active — the
        # switch-in offense read). Appended LAST so every existing offset is untouched.
        # Read-only stash of the PRE-gain physics (the interpretable damage fractions / P(KO) / accuracy),
        # for the prober/forensic decode — the learned out_gain only rescales for the projection.
        self.stash.raw_block = block.detach()
        gained = block * self.out_gain                                  # learnable per-channel adapter (×only)
        # gen3_op_tensors_views_v1: the typed named views over the post-gain block, computed ONCE
        # here so every same-forward consumer (prefuse injection, seed readout) reads a field
        # instead of an offset. Zero-copy — `gained` is still the returned serialization.
        self.stash.tensors = self.tensors_from_block(gained)
        return gained


# Effect-column order (== damage_tables.MOVE_EFFECT_COLS) — the layout of the INCOMING MATRIX's per-move
# `_DMG_IMX_HDR_EFFECT` header field, for the prober decode. (The opp-active-level collapse that used to
# read this directly is gone — gen3_op_block_trim_v1.)
_DMG_EFFECT_COLS = ("recovery", "status", "phaze", "boost", "hazard", "protect")
