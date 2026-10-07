"""K9(b)'s DECLARED inventory of every DISCRETE operation in the policy forward
(`gen3_behaviour_tie_exclusion_v1`, owner 2026-10-01).

WHY. The policy forward is piecewise-discontinuous: it SELECTS (topk, argmax) and THRESHOLDS
(``x > eps``, ``fixed >= cur_hp``). Where a selection or threshold sits within a rounding error of
its cutoff, the rollout's forward (T2: compiled, at its bucket shape) and the learner's (eager, at
the probe's batch) can resolve it differently, and log pi(a|s) jumps on that row (the K9(b) tail
sweep: 3 rows in 3.54M, fp64 gaps 5e-8 .. 1.7e-7, `designs/research_state/measurements/
k9_behaviour_tail/`). K9(b)'s fp32 behaviour check therefore EXCLUDES every row whose smallest
margin over the MARGIN sites below is under a measured epsilon, and judges every other row
deterministically (`agents/training/rust_rollout/consistency.py`; the recorder is
`agents/training/rust_rollout/tie_margins.py`).

THE RULE. Every selection call (topk / argmax / argmin / sort / argsort / kthvalue / msort /
max-or-min with a dim), every comparison in value position and every float -> int cast in a
`FORWARD_MODULES` module is declared here, keyed by (module, its source as `ast.unparse` prints
it), as exactly one of:

* a MARGIN site — the operand is a SCORE (it moves with the weights, or is float arithmetic whose
  cutoff a rounding error can cross): its per-row margin is computed by its `Rule`;
* an EXACT site — the operands are bit-identical in every forward of the same row (an observation
  read, a constant table, integers, a value gathered at a declared selection's index, ...), with the
  reason (`REASONS`).

`selection_sites_test.py` scans the AST and FAILS on an undeclared node (a new topk, argmax or
comparison anywhere in a forward module), on a stale declaration, and on a line where one op
could be either class; it also runs the production forward under a few-ulp weight jitter and
fails when an EXACT site's operands move (a score declared exact). The keys are SOURCE TEXT, so an
edit to a declared line fails the scan until it is re-declared — deliberately (owner: correctness
beats churn).

MARGINS are RELATIVE (|a - b| / max(|a|, |b|)): fp32 rounding is relative to the magnitude of the
values compared, and a relative margin is what the measured epsilon is stated in.
"""
from __future__ import annotations

import ast
import functools
from pathlib import Path
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

#: The modules (``agents/model/<name>.py``) whose code the policy forward runs — measured on the
#: production surface (every file a torch op of `evaluate_actions` was called from;
#: `selection_sites_test` re-measures it and fails on a module outside this list).
FORWARD_MODULES: Tuple[str, ...] = (
    "aux_value_heads", "belief_heads", "conditional_threat", "damage_kinds", "damage_op", "damage_op_blocks",
    "damage_op_pairwise", "encoders", "extractor_ctx", "extractor_forward", "features_extractor",
    "flat_intent", "hypothesis_encode", "hypothesis_set", "hypothesis_tokens",
    "intent_conditional", "intent_move_cell", "intent_threshold", "masked_categorical", "opp_intent",
    "pair_outcome", "pair_reduce", "pointer_head", "policy", "pools", "projection", "switch_branch",
    "t0_species", "team_transformer", "value_readouts", "value_threat_inject",
)

#: Top-level functions of a forward module that are NOT the forward (loss, label and metric helpers):
#: every node inside them is exempt (`REASONS["LABEL"]`).
LABEL_FUNCS: FrozenSet[Tuple[str, str]] = frozenset({
    ("opp_intent", f) for f in (
        "match_seats_to_move_num", "resolve_believed_slot_by_content", "info_gain_nats",
        "intent_label_weights", "set_valued_switch_loss", "intent_losses", "_alpha_subset_metrics",
        "_beta_subset_metrics", "switch_coverage_metrics", "render_alpha")} | {
    # gen3_x5_hypothesis_set_v1 (X5 U2): the set-supervision loss + label helpers and the rule-8
    # near-tie READER (`near_tie_rows`, a check-side helper, never the forward).
    ("hypothesis_set", f) for f in (
        "label_multi_hot", "set_bce", "belief_head_team_scores", "hypothesis_moves_bce", "near_tie_rows",
        # gen3_x5_belief_tokens_v1 (U3): the hypothesis-seat move TARGETS (a label builder)
        "hypothesis_moves_targets")} | {
    # gen3_x5_flat_pointer_v1 (X5 U4): the flat pointer's LABEL builder and its human render.
    ("flat_intent", f) for f in ("flat_intent_targets", "render_flat")})


class Rule(NamedTuple):
    """How a MARGIN site's per-row margin is computed (`tie_margins`; every margin is RELATIVE):

    * ``topk``     — the gap between the k-th and (k+1)-th value along the selection's dim;
    * ``argmax``   — the gap between the top-1 and top-2 value along the dim, over the slots whose
                     top-1 exceeds ``gate`` (a slot whose max is at or below ``gate`` is MASKED by the
                     site's own gate comparison — declared as its own MARGIN site — so its argmax
                     cannot reach log pi; ``gate`` 0 = every slot);
    * ``threshold`` — |x - t| elementwise, t the comparison's other operand; ``zero_exact`` declares
                     that an element where BOTH are exactly 0 is a structural zero (a product with
                     an exact 0 mask / table entry / observation), not a tie;
    * ``threshold_self`` — a ``x >= t`` whose ``t`` is one of ``x``'s own values (a top-k cutoff):
                     the element(s) that ARE ``t`` are skipped unless ``t`` occurs more than once
                     (two candidates exactly at the cutoff are a tie);
    * ``sort_head`` — an ascending (stable) SORT whose first ``head`` positions are consumed in
                     order (X5's one ordering, gen3_x5_belief_tokens_v1): the smallest gap between
                     ADJACENT sorted keys among the first ``head``, counting only pairs whose BOTH
                     keys are genuine ``−π`` values in ``[−1, 0]`` — a pair touching a structural key
                     (``+inf`` non-candidate, ``−2`` pinned revealed move) cannot flip — and, with
                     ``zero_exact``, not two exact zeros (a structural π = 0 row).

    ``consumed`` (``sort_head`` only; `gen3_behaviour_tie_consumed_v1`, 2026-10-06): the name of the issuing
    frame's local that declares how the CALLER reads the order (`hypothesis_set.stable_order`):
    a long tensor (per row) = that many leading positions read IN ORDER, the rest only as a set — the pairs up
    to and across it count; a `hypothesis_set.SetCuts` = the prefix read as a SET at each cut — only the pair
    straddling a cut counts; None = every pair of the head. A pair its reader cannot reorder anything by is no
    tie. ``""`` = no declaration (every pair of the head).

    ``payload`` (``argmax`` only; `gen3_behaviour_tie_identity_v1`, 2026-10-05): the names of the issuing
    frame's local tensors the selected index GATHERS — everything the selection reaches (a tensor of fewer
    dims than the operand is per-row, broadcast over its slot dims). The margin is then the gap to the
    nearest candidate whose payload DIFFERS: two candidates whose every payload value is bit-identical
    select the same values, so a tie between them cannot move log pi. An undeclared consumer of the index
    would make this unsound, so `selection_sites_test` reads the source and requires each payload site's
    index to be consumed ONLY by a ``gather`` of a declared payload. ``()`` = no identity clearance."""
    kind: str
    gate: float = 0.0
    zero_exact: bool = False
    why: str = ""
    head: int = 0
    payload: Tuple[str, ...] = ()
    consumed: str = ""


RULE_KINDS = ("topk", "argmax", "threshold", "threshold_self", "sort_head")

#: Every MARGIN site: (module, source) -> rule.
MARGIN: Dict[Tuple[str, str], Rule] = {
    # --- the believed-candidate SELECTIONS: topk over the composed move belief sigmoid(logits) * mask
    ("pointer_head", "w_all.topk(K, dim=-1)"): Rule("topk", why="E5 tail seats: each opp mon's top-K"),
    ("pointer_head", "w_all >= topv[..., -1:].clamp(min=1e-09)"): Rule(
        "threshold_self", why="E5 in-top-K mask at the K-th value of the same tensor (ties incl.)"),
    ("damage_op_pairwise", "w_all.detach().topk(K, dim=-1)"): Rule(
        "topk", why="per opp mon j: its top-K believed candidates (C1b/C3 attackers, bench incoming)"),
    ("damage_op_blocks", "w_all.detach().topk(K, dim=-1)"): Rule(
        "topk", why="the opp active's top-K candidates (the per-move matrix, the seat refine)"),
    ("damage_op", "w_all.detach().topk(self.damage_candidate_k, dim=-1)"): Rule(
        "topk", why="the candidate-axis truncation (damage_candidate_k > 0 only)"),
    # --- the dominant-move ARGMAX (accuracy / provenance read at the max belief-weighted high roll)
    #     gen3_behaviour_tie_identity_v1: each gathers ONE payload at its index (the move's accuracy; the
    #     move's belief weight), so a tie between two moves with bit-identical payloads is not a hazard.
    ("damage_op", "wfc.argmax(dim=-1, keepdim=True)"): Rule(
        "argmax", gate=1e-6, why="per-channel dominant move; masked when chan_max <= eps", payload=("acc_exp",)),
    ("damage_op", "wh.argmax(dim=-1, keepdim=True)"): Rule(
        "argmax", gate=1e-6, why="overall dominant move; masked when wh.amax <= eps", payload=("w_all",)),
    ("pair_reduce", "(w * ref).argmax(dim=-1, keepdim=True)"): Rule(
        "argmax", why="alpha_hard_max (Contract-W reducer; not the production path)"),
    # --- THRESHOLDS on scores
    ("damage_op", "chan_max > eps"): Rule("threshold", zero_exact=True, why="the 858 argmax's gate"),
    ("damage_op", "wh.amax(dim=-1) > eps"): Rule("threshold", zero_exact=True, why="the 864 argmax's gate"),
    ("damage_op_blocks", "high_topk > eps"): Rule(
        "threshold", zero_exact=True, why="a status secondary needs the damage to land"),
    ("belief_heads", "narrowed.sum(-1, keepdim=True) > 1e-06"): Rule(
        "threshold", zero_exact=True, why="the off-meta Hidden Power fallback"),
    ("pair_reduce", "total > PAIR_REDUCE_EPS"): Rule("threshold", zero_exact=True, why="belief-mean reducer"),
    ("pair_reduce", "w.sum(dim=-1, keepdim=True) > PAIR_REDUCE_EPS"): Rule(
        "threshold", zero_exact=True, why="learned reducer"),
    ("intent_conditional", "high_k > 0"): Rule(
        "threshold", zero_exact=True, why="damaging seat (0 exactly for a status / immune move)"),
    ("intent_threshold", "high_k > 0"): Rule(
        "threshold", zero_exact=True, why="landing damage breaks Focus Punch (0 exactly for status / immune)"),
    # --- THRESHOLDS on weight-free float arithmetic (the current HP is a product of an observed fraction
    #     and a computed max HP; a fixed-damage KO at exactly the remaining HP is a genuine near-tie)
    #     gen3_nonformula_damage_v1: ONE site now — every kernel reaches it through damage_kinds.
    ("damage_kinds", "fixed >= tgt_cur_hp"): Rule("threshold", zero_exact=True),
    # --- gen3_x5_belief_tokens_v1 (X5 U3, `--belief-tokens fixed_mass` only): THE one stable order of
    #     the hypothesis set (species, and the active's move group — one source line). Since U3 log π
    #     reads it: hypothesis rank j fills the j-th hidden slot (k <= 5, so positions 0..5) and the
    #     move seats are positions 0..K-1 (K = 6): every adjacent pair among the first 7 is a boundary
    #     (`hypothesis_set.near_tie_rows` is the same rule, at the module's own eps).
    #     gen3_behaviour_tie_consumed_v1 (2026-10-06): each caller DECLARES how it reads the order — the
    #     species order in order up to k (ranks >= k are OTHER's tail, a set), the op's per-mon orders
    #     (`build_op_roster`, `other_roster`) as a SET before each cut (only the cut pair counts), the move
    #     group in order (None: every pair of the head).
    ("hypothesis_set", "torch.argsort(neg, dim=-1, stable=True)"): Rule(
        "sort_head", head=7, zero_exact=True, consumed="consumed",
        why="X5's one order: hypothesis slots / move seats in order, the seat boundary"),
}

#: Why an EXACT site's operands are bit-identical in every forward of the same row.
REASONS: Dict[str, str] = {
    "OBS": "a float read from the observation (an HP fraction, a 0/1 flag, a float-coded id), or a sum / "
           "difference / product of such exact values, against a constant",
    "TABLE": "a constant table value, a gather of one by an exact index, or a sum / product of exact table "
             "entries (type-chart multipliers, 0/1 bits, fractions summed in one fixed order), against a constant",
    "INT": "integer or bool operands",
    "SELECTED": "a table / observation value GATHERED at a declared selection's index: exact given the "
                "selection, and a flip of that selection is the selection site's own margin",
    "ID": "identity matching over integer-valued 0/1 indicators (which sorted slot holds this move num)",
    "PYTHON": "a Python scalar comparison (a config value), not a tensor op",
    "NOT_LOGP": "does not reach log pi(a|s) (the greedy-action readout)",
    "LABEL": "a loss / label / metric helper (`LABEL_FUNCS`), not the policy forward",
    "MAX_VALUE": "an argmax whose index is read ONLY by a gather of its OWN operand (`damage_op.max_by_index`, "
                 "gen3_fm_index_max_v1): the gathered value is the operand's maximum, bit-identical to `amax` "
                 "of it in every forward, so which near-tied candidate wins never reaches log pi — the site "
                 "is exactly as continuous as the `amax` it spells (`selection_sites_test` pins the gather)",
    "BISECT": "a fixed-step bisection's direction test under no_grad (X5's fixed-size construction): "
              "either branch keeps the root inside the bracket, so a flip at a rounding error moves the "
              "converged root by at most the final bracket width — a continuous, ulp-scale change, never a "
              "log pi jump",
}

#: Every EXACT site: module -> reason -> sources.
EXACT: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "belief_heads": {
        "OBS": ("obs_hp_probs > 0", "obs_hp_probs.sum(-1, keepdim=True) > 0"),
        "INT": ("n_revealed >= opp_move_ids.shape[-1]", "opp_move_ids == HIDDEN_POWER_MOVE_NUM", "opp_move_ids > 0"),
    },
    "damage_op": {
        "OBS": ("hp_frac > 0", "opp_burn > 0.5", "opp_para > 0.5", "our_para > 0.5", "s >= 0"),
        "TABLE": ("bp_all > 0", "phys_all > 0.5"),
        "SELECTED": ("bu_all > 0",),            # gen3_beatup_exact_v1: the 0/1 Beat Up bit at the candidate index
        "MAX_VALUE": ("x.detach().argmax(dim=-1, keepdim=True)",),   # gen3_fm_index_max_v1 (fixed_mass only)
        "INT": ("(phys_all > 0.5).long()", "ctx.type1_ids[:, _og] == _GHOST_TIDX",
                "ctx.type2_ids[:, _og] == _GHOST_TIDX", "move_ty == _ELECTRIC_TIDX", "move_ty == _FIRE_TIDX",
                "move_ty == _WATER_TIDX", "mty_all == at1[:, None]", "mty_all == at2[:, None]", "opp_item == 0",
                "opp_item == self.cb_item_num"),
    },
    "damage_op_blocks": {
        "OBS": ("(opp_slp * (1.0 - opp_rest)).sum(-1) > 0.5", "ctx.hp_and_active[:, :TEAM_SIZE, 0] > 0",
                "ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0] > 0", "ctx.hp_and_active[:, our, 0] > 0",
                "ctx.hp_and_active[ar, our_act, 0] > 0", "ctx.opp_ctx_raw[:, _SUBSTITUTE_CTX_IDX] > 0.5",
                "has_nc > 0.5", "has_rest > 0.5", "hp_frac > 0", "nonrest_sleep.sum(dim=1) > 0.5",
                "opp_cond.sum(-1) > 0.5", "opp_cond.sum(dim=1) > 0.5", "opp_hp_frac > 0", "opp_para > 0.5",
                "our_burn > 0.5", "our_cb > 0.5", "our_cond.sum(-1) > 0.5", "our_para > 0.5",
                "(~ctx.opp_believed_mask).long()"),
        "TABLE": ("has_cure > 0.5", "has_other_cleric > 0.5", "self.MOVE_TYPE_IDX.long()",
                  "live_cleric.sum(dim=-1, keepdim=True) - live_cleric > 0.5"),
        "SELECTED": ("bp_k > 0", "ded.sum(dim=-1, keepdim=True) > 0.5", "sec_tot > eps"),
        "INT": ("ctx.all_move_ids[ar, opp_act] > 0", "ctx.item_ids[:, our] == self.cb_item_num",
                "ctx.item_ids[ar, our_act] == self.cb_item_num", "move_ids == self.hp_num",
                "move_ty == _ELECTRIC_TIDX", "move_ty == _FIRE_TIDX", "move_ty == _WATER_TIDX",
                "move_ty == at1[:, :, None]", "move_ty == at1[:, None]", "move_ty == at2[:, :, None]",
                "move_ty == at2[:, None]", "move_ty.long()", "mty_k == at1[:, None]", "mty_k == at2[:, None]",
                "n_revealed[:, None] < 4", "opp_ability > 0", "our_moves == self.rest_num", "sidx == c",
                "torch.arange(K, device=device)[None, :] < 4"),
    },
    "damage_op_pairwise": {
        "OBS": ("(ctx.spikes_feature * 3.0).round()", "(ctx.spikes_feature * 3.0).round().long()",
                "ctx.hp_and_active[:, :TEAM_SIZE, 0] > 0", "ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0] > 0",
                "ctx.hp_and_active[:, opp, 0] > 0", "ctx.hp_and_active[:, sl, 0] > 0",
                "ctx.hp_and_active[ar, ctx.our_active_idx, 0] > 0", "hp_frac > 0",
                "hp_frac[:, None] > hp_cost", "hp_j > 0", "leech_active > 0.5", "opp_burn > 0.5"),
        "TABLE": ("deltas.abs().sum(-1) > 0", "frac > 0", "hp_cost <= 0",
                  "self.TYPE_IS_GHOST[_at1] + self.TYPE_IS_GHOST[_at2] > 0.0", "wh > 0"),
        "SELECTED": ("bp_k > 0", "nf_k[0] + nf_k[1] + nf_k[2] > 0"),
        "INT": ("ctx.all_move_ids[:, :TEAM_SIZE] == pur", "ctx.our_active_req_move_ids == self.baton_num",
                "ctx.our_active_req_move_ids == self.curse_num", "ctx.our_active_req_move_ids == self.rest_num",
                "ctx.our_active_req_move_ids.long()", "ids == self.toxic_num", "items == LEFTOVERS_NUM",
                "move_ids == n", "mty_k == at1[:, :, None]", "mty_k == at2[:, :, None]", "mty_k.long()",
                "mty_k == ctx.type1_ids[:, opp][:, :, None]", "mty_k == ctx.type2_ids[:, opp][:, :, None]",
                "opp_ab > 0", "opp_ability > 0", "sidx != LEECH_SEED_CAT", "sidx == 1", "sidx == 2", "sidx == 5",
                "sidx == _SLP_STATUS_CAT", "sidx > 0", "types1 == ti", "types2 == ti"),
    },
    "damage_kinds": {
        # gen3_nonformula_damage_v1: the declared non-formula / HP-dependent-BP kinds (the attacker's
        # and the target's HP are observed fractions × computed max HP; the kinds are table bits).
        "OBS": ("ctx.hp_and_active[:, sl, 0] > 0", "ratio < thr", "status < 0.5", "tgt_cur_hp > 0", "bp > 0"),
        "TABLE": ("fixed + target_frac + endeavor > 0", "hp_scaled > 0", "flail > 0", "eff > 0",
                  "nonformula > 0", "op.MOVE_BEATUP[move_ids] > 0", "target_frac >= 1.0"),
        # gen3_beatup_exact_v1: Beat Up's ally filter reads observed HP fractions and 0/1 condition flags
        # (a fainted / hidden slot is exactly 0, a status flag exactly 0 or 1 — no near-tie), and the
        # move flag is a 0/1 table bit gathered at an exact / already-declared index.
        "SELECTED": ("is_bu > 0",),
    },
    "encoders": {
        "INT": ("ctx.all_move_ids == HIDDEN_POWER_MOVE_NUM",),
    },
    "extractor_ctx": {
        "OBS": ("_opp_hp > 0.0", "hp_and_active[:, 0:TEAM_SIZE, 0] == 0",
                "hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0] == 0",
                "pokemon_part[:, :, POKEMON_LAST_ACTION_OFFSET].long()", "pokemon_part[:, :, ability1_idx].long()",
                "pokemon_part[:, :, ability2_idx].long()", "pokemon_part[:, :, item_idx].long()",
                "pokemon_part[:, :, slot_idx + _type_off].long()", "pokemon_part[:, :, slot_idx].long()",
                "pokemon_part[:, :, species_idx].long()",
                "pokemon_part[:, :, types_info['offset'] + types_layout['type1']['offset']].long()",
                "pokemon_part[:, :, types_info['offset'] + types_layout['type2']['offset']].long()",
                "species_known_opp < 0.5", "species_known_opp > 0.5", "torch.argmax(active_flags, dim=1)",
                "x[:, _arm.start + _arm_per:_arm.start + 2 * _arm_per].long()",
                "x[:, _arm.start:_arm.start + _arm_per].long()"),
    },
    "extractor_forward": {
        "OBS": ("_opp_active_flag < 0.5", "ctx.hp_and_active[:, :TEAM_SIZE, 0] > 0"),
        "INT": ("_seat_nums > 0",),
        "PYTHON": ("self.entity_topk_seats > 0", "self.opp_intent_grad_mode == 'shaping'"),
    },
    "intent_conditional": {
        "INT": ("req_move_ids[..., None] == self.gate_nums", "req_move_ids[..., None] == self.protect_only_nums",
                "topk_nums[..., None] == self.protect_nums"),
    },
    "intent_threshold": {
        "INT": ("req_move_ids[..., None] == self.mech_nums",),
    },
    "masked_categorical": {
        "INT": ("actions.long()",),
        "NOT_LOGP": ("torch.argmax(probs(logp), dim=1)",),
    },
    "opp_intent": {
        "OBS": ("candidate_mask < 0.5", "candidate_mask > 0.5", "seat_valid < 0.5"),
        "INT": ("(candidate_mask > 0.5).sum(dim=-1) == 0",),
    },
    "pair_outcome": {
        "INT": ("our_type1 == GHOST_TYPE_IDX", "our_type2 == GHOST_TYPE_IDX", "topk_nums[..., None] == self.spin_num",
                # X5 U4 (`seat_in_set`): move-num membership + its index cast
                "nums[..., None] == set_nums", "set_nums.long()"),
    },
    # gen3_x5_flat_pointer_v1 (X5 U4, fixed_mass): the flat pointer's candidate set — integer ids, the
    # hypothesis set's structural counts / ranks, and the observation's active flag. No selection: the
    # move seats come from THE one order (its boundaries are already `near_tie_rows`'), OTHER_move /
    # OTHER_species are whole sets.
    "flat_intent": {
        "OBS": ("opp_active_flag < 0.5",),
        "INT": ("hs.slot_species.long()", "opp_species_ids.long()", "mv.seat_nums == _HP_NUM",
                "hs.slot_species > 0", "fm.seat_nums.long()", "hs.rank >= sp.k.unsqueeze(-1)", "sp.k > 0",
                "torch.arange(F, device=tokens.device) <= other_move_col(k)"),
    },
    "pointer_head": {
        "ID": ("match.float().argmax(-1)", "move_valid < 0.5"),
        "INT": ("ctx.our_active_req_move_ids.long()", "req_ids[:, :, None] > 0",
                "sorted_ids[:, None, :] == req_ids[:, :, None]"),
    },
    "switch_branch": {
        "INT": ("req_move_ids[..., None] == self.protect_nums", "req_move_ids[..., None] == self.spin_num"),
    },
    # gen3_x5_hypothesis_set_v1 (X5 U2, `--belief-tokens fixed_mass` only — the module never runs on the
    # production `blob` surface). In U2 the hypothesis set is STASHED and read only by the presence BCE
    # and the readers, so its three float-operand discrete ops are NOT_LOGP. 🚨 U3 wires the tokens into
    # the trunk / the op — log π then reads them, and these three MUST be re-declared: the selection
    # argsort as a MARGIN rule on the k-th / (k+1)-th π gap (§3.1's rule-8 exclusion,
    # `hypothesis_set.near_tie_rows`, including OTHER's tail-mean cutoff and the move seats); the
    # bisection's `total > k_t` is NOT a discontinuity (τ converges to the same root either way, to the
    # dtype's resolution) and needs a reason of its own; `denom > 0` is a structural-count gate.
    "hypothesis_set": {
        "BISECT": ("total > k_t",),
        "TABLE": ("logits > cut",),
        "INT": ("k.long()", "opp_species_ids.clamp(0, S - 1).long()", "opp_believed_mask.bool().sum(-1).long()",
                "revealed_ids.clamp(0, M - 1).long()", "revealed_ids > 0", "revealed.sum(-1).long()",
                "n > 0", "k > 0", "k < n", "k >= n", "at > 0", "at < n_avail", "ids > 0",
                "num >= TYPED_HP_NUMS[0]", "num <= TYPED_HP_NUMS[-1]", "num != HIDDEN_POWER_MOVE_NUM",
                "j.unsqueeze(0) < k.unsqueeze(-1)", "species.clamp(0, self.n_species - 1).long()",
                "jj.unsqueeze(0) < torch.minimum(n_seatable, torch.full_like(n_seatable, K)).unsqueeze(-1)",
                "jj.unsqueeze(0) < r.unsqueeze(-1)", "opp_species_ids > 0",
                "rank >= k.unsqueeze(-1)", "rank >= K",
                "k_m > 0", "r < K", "believed.long()",
                "i >= lo.unsqueeze(-1)", "i < at.unsqueeze(-1)", "i + 1 < n_avail.unsqueeze(-1)"),
    },
    # gen3_x5_belief_tokens_v1 (X5 U3, fixed_mass only): structural integer tests on the move group
    # (a seat's num is the revealed Hidden Power; the group's mass k_m is positive). U3 part 3 (the op's
    # opponent-MON roster): the per-mon candidate masks are num-range tests, the hidden-team marginal is
    # gated on the integer count k, and the bench E5 tail is a RANK test on the one order — the order
    # itself is `hypothesis_set.stable_order` (the declared sort_head MARGIN site, re-used, not copied).
    # gen3_x5_hyp_gather_v1: the gathered hypothesis encoding — the Hidden Power slot test on the dex
    # table's integer move ids (as `encoders`), and the hypothesis species (already integer).
    "hypothesis_encode": {
        "INT": ("ids['all_move_ids'] == HIDDEN_POWER_MOVE_NUM", "slot_species.long()"),
    },
    "hypothesis_tokens": {
        "INT": ("moves.seat_nums == HP", "pres.k > 0", "species.clamp(0, S - 1).long()",
                "num >= _TYPED_HP[0]", "num <= _TYPED_HP[-1]", "num != HP", "k > 0",
                "ro.move_rank >= K",
                # OTHER's tables (U3 part 3): the species-type table and the first hidden slot (an integer
                # argmax over a bool mask — exact by type)
                "species_type.long()", "t.unsqueeze(0) == st[:, 0:1]", "t.unsqueeze(0) == st[:, 1:2]",
                "hyp.long()", "torch.argmax(hyp.long(), dim=-1)"),
    },
    "t0_species": {
        "OBS": ("onehot > 0",),
        "INT": ("ids > 0", "opp_species_ids.clamp(0, n_species - 1).long()"),
    },
    "team_transformer": {
        "OBS": ("-eside == ss", "actor == sm", "actor > 0", "eside == ss", "ev[:, :, C.ACTOR_SPECIES].long()",
                "ev[:, :, C.CALLER].long()", "ev[:, :, C.CANT].long()", "ev[:, :, C.DENIAL].long()",
                "ev[:, :, C.ENTRY].long()", "ev[:, :, C.FAINT_CAUSE].long()", "ev[:, :, C.ITEM_TRANSITION].long()",
                "ev[:, :, C.MOVE].long()", "ev[:, :, C.REL_SPECIES].long()", "ev[:, :, C.STATUS].long()",
                "ev[:, :, C.STAT].long()", "ev[:, :, C.TARGET_SPECIES].long()", "ev[:, :, C.TYPE].long()",
                "ev[:, :, C.VALID] < 0.5", "ev[:, :, C.VALID] > 0.5", "rel == sm", "rel > 0", "rside == ss",
                "tgt == sm", "tgt > 0"),
    },
    "value_readouts": {
        "OBS": ("op_alive.clamp(max=1.0) < 0.5",),
    },
}


# ------------------------------------------------------------------------------------------- the scan
SELECTION_ATTRS = frozenset({"topk", "argmax", "argmin", "sort", "argsort", "kthvalue", "msort"})
CAST_ATTRS = frozenset({"long", "int", "round", "floor", "ceil", "trunc"})
COMPARE_CALLS = frozenset({"gt", "ge", "lt", "le", "eq", "ne", "greater", "greater_equal", "less", "less_equal"})
_INT_DTYPES = ("long", "int", "int8", "int16", "int32", "int64", "uint8", "bool")
_CMP_OPS = {ast.Gt: "gt", ast.GtE: "ge", ast.Lt: "lt", ast.LtE: "le", ast.Eq: "eq", ast.NotEq: "ne"}


class Node(NamedTuple):
    module: str
    func: str          # the enclosing top-level function / class.method qualname
    kind: str          # "sel" | "cmp" | "cast"
    ops: Tuple[str, ...]   # sel / cast: the method name; cmp: each operator ("gt", "ge", ...)
    src: str
    line: int
    end: int


def module_path(module: str) -> Path:
    return Path(__file__).resolve().parent / f"{module}.py"


def _qualnames(tree: ast.AST) -> Dict[int, str]:
    out: Dict[int, str] = {}

    def walk(n: ast.AST, q: List[str]) -> None:
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(c, q + [c.name])
            else:
                out[id(c)] = ".".join(q)
                walk(c, q)
    walk(tree, [])
    return out


def _is_int_dtype(n: ast.AST) -> bool:
    s = ast.unparse(n)
    return any(s.endswith(t) for t in _INT_DTYPES)


def scan_source(module: str, source: str) -> List[Node]:
    """Every discrete node of ``source`` (module docs): selection calls, value-position comparisons (not
    an ``if`` / ``while`` / ``assert`` / conditional-expression / comprehension test; no ``is`` / ``in``),
    comparison calls (``x.gt(y)``, ``torch.ge(...)``) and float -> int casts (``.long()``, ``.round()``,
    ``.to(torch.long)``, ...)."""
    tree = ast.parse(source)
    tests: set = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            tests.update(id(x) for x in ast.walk(n.test))
        elif isinstance(n, ast.comprehension):
            tests.update(id(x) for c in n.ifs for x in ast.walk(c))
    qn = _qualnames(tree)
    out: List[Node] = []
    for n in ast.walk(tree):
        kind: Optional[str] = None
        ops: Tuple[str, ...] = ()
        if isinstance(n, ast.Compare):
            if id(n) in tests or any(isinstance(o, (ast.Is, ast.IsNot, ast.In, ast.NotIn)) for o in n.ops):
                continue
            kind, ops = "cmp", tuple(_CMP_OPS.get(type(o), "?") for o in n.ops)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            a = n.func.attr
            if a in SELECTION_ATTRS or (a in ("max", "min") and (n.args or any(k.arg == "dim" for k in n.keywords))):
                kind, ops = "sel", (a,)
            elif a in COMPARE_CALLS and n.args:
                kind, ops = "cmp", (a,)
            elif a in CAST_ATTRS and not n.args:
                kind, ops = "cast", (a,)
            elif a in ("to", "type") and (any(_is_int_dtype(x) for x in n.args)
                                          or any(k.arg == "dtype" and _is_int_dtype(k.value) for k in n.keywords)):
                kind, ops = "cast", (a,)
        if kind is not None:
            line = int(getattr(n, "lineno", 0))
            out.append(Node(module, qn.get(id(n), ""), kind, ops, ast.unparse(n), line,
                            int(getattr(n, "end_lineno", line) or line)))
    return out


def scan(module: str) -> List[Node]:
    return scan_source(module, module_path(module).read_text())


class Declared(NamedTuple):
    exact: Optional[str]       # the REASONS key, or None for a MARGIN site
    rule: Optional[Rule]


def declaration(module: str, src: str, func: str = "") -> Optional[Declared]:
    """What ``src`` in ``module`` is declared as (None = UNDECLARED)."""
    if (module, func.split(".")[0]) in LABEL_FUNCS:
        return Declared("LABEL", None)
    rule = MARGIN.get((module, src))
    if rule is not None:
        return Declared(None, rule)
    for reason, srcs in EXACT.get(module, {}).items():
        if src in srcs:
            return Declared(reason, None)
    return None


_RUNTIME_OPS = {"gt": "gt", "__gt__": "gt", "greater": "gt", "ge": "ge", "__ge__": "ge", "greater_equal": "ge",
                "lt": "lt", "__lt__": "lt", "less": "lt", "le": "le", "__le__": "le", "less_equal": "le",
                "eq": "eq", "__eq__": "eq", "ne": "ne", "__ne__": "ne",
                "__rgt__": "lt", "__rge__": "le", "__rlt__": "gt", "__rle__": "ge"}


def runtime_op(name: str) -> Optional[Tuple[str, str]]:
    """A torch function name as the recorder sees it -> (kind, op), or None when it is not discrete."""
    if name in _RUNTIME_OPS:
        return "cmp", _RUNTIME_OPS[name]
    if name in SELECTION_ATTRS or name in ("max", "min"):
        return "sel", name
    if name in CAST_ATTRS or name in ("to", "type"):
        return "cast", name
    return None


class Resolved(NamedTuple):
    declared: Optional[Declared]   # None = undeclared
    src: str


@functools.lru_cache(maxsize=None)
def line_map(module: str) -> Dict[Tuple[int, str], Resolved]:
    """(line, kind) -> the declaration of the discrete node(s) spanning that line, for the runtime
    recorder. A line where nodes of one kind differ in class is AMBIGUOUS (the scan test refuses it)."""
    out: Dict[Tuple[int, str], Resolved] = {}
    for n in scan(module):
        d = declaration(module, n.src, n.func)
        for ln in range(n.line, n.end + 1):
            key = (ln, n.kind)
            prev = out.get(key)
            if prev is None or (prev.declared is not None and d is not None and prev.declared == d):
                out[key] = Resolved(d, n.src if prev is None else prev.src)
            elif prev.declared != d:
                # Different classes on one line: prefer a MARGIN declaration (conservative); the scan test
                # refuses this shape anyway.
                if d is not None and d.rule is not None:
                    out[key] = Resolved(d, n.src)
    return out


def resolve(module: str, line: int, kind: str) -> Optional[Resolved]:
    """The declaration the recorder applies to an op of ``kind`` executed at ``module``:``line`` (None
    when the line holds no scanned node of that kind — e.g. a library call's internal op)."""
    if module not in FORWARD_MODULES:
        return None
    return line_map(module).get((int(line), kind))


def ambiguous_lines(module: str) -> List[str]:
    """Lines where nodes of the same kind are declared differently (the recorder could not tell them apart)."""
    seen: Dict[Tuple[int, str], set] = {}
    for n in scan(module):
        d = declaration(module, n.src, n.func)
        cls = None if d is None else ("MARGIN" if d.rule is not None else "EXACT")
        for ln in range(n.line, n.end + 1):
            seen.setdefault((ln, n.kind), set()).add(cls)
    return [f"{module}.py:{ln} ({k})" for (ln, k), c in sorted(seen.items()) if len(c) > 1]
