"""`check_compatible` -- the gate that runs on EVERY load.

Resume, frozen eval opponents, the self-play pool, distillation teachers: all of them pass a
saved config through here before the weights are touched. Its members are the fields whose
mismatch would be a wrong ANSWER rather than a loud failure, so a field belongs here only if
flipping it changes the state_dict or the forward. The resume-only and opponent-only gates
(where a frozen opponent must NOT be rejected) live in `resume_checks.py`.

`flag_registry_test.py::test_resume_immutable_flags_are_excluded_from_check_compatible` reads
this method with `inspect.getsource(ModelVersion.check_compatible)`, which follows the function
to whichever module defines it -- so the move keeps that scan pointed at the real text.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from agents.model.model_version.constants import ModelVersionError
from agents.model.model_version.fields import ModelVersionFields

if TYPE_CHECKING:
    from agents.model.model_version.spec import ModelVersion


class ModelVersionCompatibility(ModelVersionFields):
    """The `ModelVersion.check_compatible` half of the class."""

    def check_compatible(self, saved: ModelVersion) -> None:
        """Raises ModelVersionError if saved is incompatible with self (current code).
        Call as: current_version.check_compatible(saved_version).
        """
        # Architecture family — hard stop if different
        if self.arch_signature != saved.arch_signature:
            raise ModelVersionError(
                f"Architecture family mismatch: saved='{saved.arch_signature}', "
                f"current='{self.arch_signature}'.\n"
                "These models use structurally different networks and cannot be loaded interchangeably.\n"
                "Start a fresh training run, or use subprocess isolation for league play."
            )

        # Weight-relevant fields — all must match exactly
        _WEIGHT_FIELDS = {
            "total_dim", "active_context_dim",
            "species_embedding_dim", "max_species",
            "move_embedding_dim", "max_moves",
            "item_embedding_dim", "max_items",
            "ability_embedding_dim", "max_abilities",
            "type_embedding_dim", "max_types",
            "role_token_size", "projection_dim",
            "move_net_hidden", "role_encoder_hidden",
            "net_arch",
        }
        current = asdict(self)
        saved_d = asdict(saved)
        mismatches = [
            f"  {k}: saved={saved_d[k]!r}, current={current[k]!r}"
            for k in sorted(_WEIGHT_FIELDS)
            if current[k] != saved_d.get(k)
        ]
        if mismatches:
            raise ModelVersionError(
                "Model weight-shape mismatch — cannot load saved model with current architecture.\n"
                "Mismatched fields:\n" + "\n".join(mismatches) + "\n\n"
                "Fix: restore matching constants, or start a fresh training run."
            )

        # Feature toggle — value-checked (not weight-shape) but STRUCTURAL: PopArt adds value-head
        # buffers + normalized output, so loading a use_popart mismatch breaks the state_dict on
        # EVERY load. Unlike vf_coef / reward-config (value-meaning, resume-only) it lives here in
        # check_compatible (gates eval / pool / distill loads too), with a dedicated message.
        if self.use_popart != saved.use_popart:
            raise ModelVersionError(
                f"PopArt mismatch: saved={saved.use_popart}, current={self.use_popart}.\n"
                "PopArt changes the value head's parameterization (normalized output + running "
                "mu/sigma buffers), so it cannot be toggled on a resumed model.\n"
                "Resume with the matching --use-popart setting, or start a fresh training run."
            )

        # Behavioral toggle — value-checked (not weight-shape): unmasking the opponent's hidden
        # party changes the transformer's key_padding_mask (policy AND value forward). The state_dict
        # is identical either way, but a resume that flips it would feed the policy a different mask
        # than it trained under. Lives here (gates resume) with a dedicated message; same-run
        # pool/sentinel/distill snapshots carry the same value so they pass trivially.
        if self.attend_unrevealed_opponents != saved.attend_unrevealed_opponents:
            raise ModelVersionError(
                f"attend_unrevealed_opponents mismatch: saved={saved.attend_unrevealed_opponents}, "
                f"current={self.attend_unrevealed_opponents}.\n"
                "Unmasking the opponent's hidden party changes the transformer mask the policy was "
                "trained under, so it cannot be toggled on a resumed model.\n"
                "Resume with the matching --attend-unrevealed-opponents setting, or start a fresh run."
            )

        # Structural toggle — like use_popart it changes the state_dict (k>0 adds HiddenOppBeliefPool +
        # widens both projection Linears by k*D_MODEL), so a mismatch breaks the load. As a plain int,
        # EVERY distinct value is a weight-shape change (incl. 0↔N = adding/removing the module), so one
        # unconditional comparison gates it — no separate on/off field.
        if self.opp_belief_cls_k != saved.opp_belief_cls_k:
            raise ModelVersionError(
                f"opp_belief_cls_k mismatch: saved={saved.opp_belief_cls_k}, current={self.opp_belief_cls_k}.\n"
                "The number of hidden-opponent belief query tokens (0 = off) sets the projection width, "
                "so it is a weight-shape parameter and cannot change on an existing model.\n"
                f"Resume with --opp-belief-cls-k {saved.opp_belief_cls_k}, or start a fresh training run."
            )

        # Structural toggle — adds our_active_refined to the value projection (widens it by D_MODEL), so
        # a mismatch breaks the value head's state_dict. Like use_popart it gates EVERY load.

        # Structural toggle — ON adds the BeliefHead + per-slot unknown-mon embeddings to the
        # state_dict (the in-place hidden-opponent belief). Like use_popart it gates EVERY load; the
        # training-only opp_belief_aux_coef is deliberately NOT checked (it touches no forward pass).
        if self.opp_belief_slots != saved.opp_belief_slots:
            raise ModelVersionError(
                f"opp_belief_slots mismatch: saved={saved.opp_belief_slots}, current={self.opp_belief_slots}.\n"
                "The hidden-opponent belief-aux module (learned unknown-mon slot tokens + BeliefHead) "
                "changes the state_dict, so it cannot be toggled on an existing model.\n"
                "Resume with the matching --opp-belief-aux-coef setting, or start a fresh training run."
            )

        # Structural toggle — the MoveBelief module (move head + reinject Linear) is in the state_dict
        # AND its mode changes the trained forward (which slots are enriched). Gated as a STRING, every
        # load; the training-only move_belief_coef is NOT checked.
        if self.move_belief_mode != saved.move_belief_mode:
            raise ModelVersionError(
                f"move_belief_mode mismatch: saved={saved.move_belief_mode!r}, current={self.move_belief_mode!r}.\n"
                "The move-belief module (predict+reinject the opp moveset) changes the state_dict and the "
                "forward, so the mode cannot change on an existing model.\n"
                "Resume with the matching --move-belief-mode setting, or start a fresh training run."
            )

        # Structural toggle — the DamageOperator appends a believed-damage block to BOTH projection
        # inputs, so toggling it changes both projection Linears' shapes. Like value_active_readout it
        # gates EVERY load with a dedicated bool compare; OFF = baseline byte-for-byte.
        if self.damage_op != saved.damage_op:
            raise ModelVersionError(
                f"damage_op mismatch: saved={saved.damage_op}, current={self.damage_op}.\n"
                "The differentiable damage operator widens both projection heads, so it changes the "
                "state_dict and cannot be toggled on an existing model.\n"
                "Resume with the matching --damage-op setting, or start a fresh training run."
            )

        # Structural toggle (weight-shape, like damage_op): the OUTGOING per-move block widens both
        # projection Linears, so toggling it changes the state_dict.
        if self.damage_outgoing != saved.damage_outgoing:
            raise ModelVersionError(
                f"damage_outgoing mismatch: saved={saved.damage_outgoing}, current={self.damage_outgoing}.\n"
                "The outgoing per-move damage block widens both projection heads, so it changes the "
                "state_dict and cannot be toggled on an existing model.\n"
                "Resume with the matching --unified-damage setting, or start a fresh training run."
            )

        # v24 STRUCTURAL toggle (weight-shape, like damage_op): the MoveLatentEncoder widens the
        # move-network input, so toggling it changes the state_dict.
        if self.move_latent != saved.move_latent:
            raise ModelVersionError(
                f"move_latent mismatch: saved={saved.move_latent}, current={self.move_latent}.\n"
                "The MoveLatentEncoder concatenates a per-move latent into the move network, so it changes "
                "the state_dict and cannot be toggled on an existing model.\n"
                "Resume with the matching --move-latent setting, or start a fresh training run."
            )

        # v25 STRUCTURAL toggle (like opp_belief_slots): the SpreadBelief module adds params, so toggling
        # it changes the state_dict.
        if self.spread_belief != saved.spread_belief:
            raise ModelVersionError(
                f"spread_belief mismatch: saved={saved.spread_belief}, current={self.spread_belief}.\n"
                "The SpreadBelief module (the hidden-spread belief head) changes the state_dict and cannot "
                "be toggled on an existing model.\n"
                "Resume with the matching --spread-belief setting, or start a fresh training run."
            )

        # v40 STRUCTURAL toggle (gen3_nature_ev_belief_v1): the nature/EV generative head has DIFFERENT
        # SpreadBelief params (nature_head + ev_head vs the additive stat_head), so toggling it changes the
        # state_dict.
        if self.spread_belief_nature != saved.spread_belief_nature:
            raise ModelVersionError(
                f"spread_belief_nature mismatch: saved={saved.spread_belief_nature}, "
                f"current={self.spread_belief_nature}.\n"
                "The nature/EV generative head reparameterises SpreadBelief (its params differ from the "
                "additive head), so it changes the state_dict and cannot be toggled on an existing model.\n"
                "Resume with the matching --spread-belief-nature setting, or start a fresh training run."
            )

        # v40 FORWARD-BEHAVIOR toggle (gen3_nature_ev_belief_v1, like move_prior_fusion): no new params, but a
        # mid-run flip feeds the op a different (marginalised vs mean-field) forward.

        # v25 FORWARD-BEHAVIOR toggles (like mask_incoming_damage_obs): each zeros a now-subsumed obs region
        # from the model's view → a different forward the policy/value trained under (state_dict identical).

        # Forward-behavior float (no weight-shape change, like move_prior_fusion): the LEGAL-BUT-UNOBSERVED
        # base of the move prior. Legality itself is unconditional (v65) and not comparable — only the
        # height of this floor is a choice, and changing it changes the belief the policy/value/op trained
        # under. A pre-v65 checkpoint recorded 0.0 (which used to mean "legality OFF") and will land here
        # against the current 0.02 default: that rejection is the POINT, not a bug to migrate away.
        # NOTE (MIGRATION_FLOOR): every pre-v65 config is now refused at _migrate_config's floor
        # before it can reach this check, so the saved==0.0 message below is reachable only from a
        # hand-built v67+ config (the extractor itself refuses floors below _MIN_PRIOR_FLOOR).
        # Kept as defence in depth — behavior deliberately unchanged.
        if self.move_candidate_floor != saved.move_candidate_floor:
            raise ModelVersionError(
                f"move_candidate_floor mismatch: saved={saved.move_candidate_floor}, "
                f"current={self.move_candidate_floor}.\n"
                "This is the legal-but-unobserved base of the move prior; changing it changes the belief "
                "the policy trained under, so it cannot be changed on a resumed model.\n"
                + (
                    "saved=0.0 predates v65 (gen3_unconditional_move_legality_v1), where 0.0 meant "
                    "'no legality gate' — a prior that gave phantom mass to moves a species cannot "
                    "learn. That is no longer representable: legality is unconditional now. This "
                    "checkpoint cannot be resumed; start a fresh training run.\n"
                    if saved.move_candidate_floor == 0.0 else
                    "Resume with the matching --move-candidate-floor, or start a fresh training run.\n"
                )
            )

        # Forward-behavior toggle (no weight-shape change, like attend_unrevealed_opponents): fusing the
        # move prior changes the belief the policy/value/damage-op trained under, so a resume that flips
        # it would feed a different forward. The state_dict is identical either way (the prior buffer is
        # non-persistent), so this is a train/eval-consistency gate, not a loadability one.
        if self.move_prior_fusion != saved.move_prior_fusion:
            raise ModelVersionError(
                f"move_prior_fusion mismatch: saved={saved.move_prior_fusion}, current={self.move_prior_fusion}.\n"
                "Fusing the Smogon move prior into the belief changes the forward the policy trained "
                "under, so it cannot be toggled on a resumed model.\n"
                "Resume with the matching --move-prior-fusion setting, or start a fresh training run."
            )

        # (v32 `move_belief_prefuse` and v50 `damage_op_prefuse` had gates here. Both are DELETED at
        # v71: the PRE-transformer placement is the only one, so there is no longer a second forward to
        # be inconsistent with. A saved config that recorded either at a non-production value is
        # REFUSED by the v71 migration, which is louder than this gate was.)

        # v49 gen3_topk_candidates_v1 — FORWARD-BEHAVIOR (no weight-shape change): truncating the
        # op's candidate axis changes the damage the policy/value trained under. Unconditional int
        # compare (the `damage_topk_k` pattern), so 0<->K and K<->M both fail.
        if self.damage_candidate_k != saved.damage_candidate_k:
            raise ModelVersionError(
                f"damage_candidate_k mismatch: saved={saved.damage_candidate_k}, "
                f"current={self.damage_candidate_k}.\n"
                "Capping the DamageOperator's candidate sweep changes the incoming damage the model "
                "trained under, so it cannot be toggled on a saved model.\n"
                "Load with the matching --damage-candidate-k, or start a fresh training run."
            )

        # v51 gen3_pointer_native_v1: the pointer head is unconditional (no gate) — a pre-generation
        # checkpoint fails the ARCH_SIGNATURE family check above, which is the intended loud break.

        # v54 gen3_entity_move_seats_v1 — STRUCTURAL int (the damage_topk_k pattern): >0 adds
        # `threat_seat_proj` (state_dict) and K threat seats to every attention pass (forward), so
        # 0<->K and K<->M both fail. The unconditional E3 seats ride the ARCH_SIGNATURE, not this.
        if self.consequence_topk != saved.consequence_topk:
            raise ValueError(
                f"consequence_topk mismatch: saved={saved.consequence_topk}, "
                f"current={self.consequence_topk} — the consequence kernels' candidate axis is a "
                "forward-behavior toggle (the worst-case max covers a different candidate set); "
                "load with matching --consequence-topk."
            )
        if self.entity_topk_seats != saved.entity_topk_seats:
            raise ModelVersionError(
                f"entity_topk_seats mismatch: saved={saved.entity_topk_seats}, "
                f"current={self.entity_topk_seats}.\n"
                "The E4 threat-move seats add a projection (threat_seat_proj) and change every "
                "attention pass's seat count, so the weights are not interchangeable.\n"
                "Load with the matching --entity-topk-seats, or start a fresh training run."
            )

        # v56 gen3_edge_bias_trunk_v1 — STRUCTURAL str (the win_prob_mode pattern): a family adds its
        # zero-init bias map (state_dict) and its cells to every attention pass (forward), so any
        # mismatch — off<->on or a different family set — fails.
        if self.edge_bias_families != saved.edge_bias_families:
            raise ModelVersionError(
                f"edge_bias_families mismatch: saved={saved.edge_bias_families!r}, "
                f"current={self.edge_bias_families!r}.\n"
                "The edge-bias families add per-family map parameters and change the attention "
                "biases the model trained under, so the weights are not interchangeable.\n"
                "Load with the matching --edge-bias-families, or start a fresh training run."
            )

        # v57 gen3_entity_tail_seats_v1 — STRUCTURAL bool: adds tail_proj/tail_marker (state_dict)
        # and 6 seats to every attention pass (forward).
        if self.entity_tail_seats != saved.entity_tail_seats:
            raise ModelVersionError(
                f"entity_tail_seats mismatch: saved={saved.entity_tail_seats}, "
                f"current={self.entity_tail_seats}.\n"
                "The E5 tail-threat seats add parameters and change every attention pass's seat "
                "count, so the weights are not interchangeable.\n"
                "Load with the matching --entity-tail-seats, or start a fresh training run."
            )

        # Structural + resume-IMMUTABLE toggle — gated as a STRING so BOTH 'none'↔head (a state_dict
        # change: the WinProbHead params) AND read_only↔shaping (same params, but flipping the trunk
        # gradient flow mid-run is a silent training change the user chose to forbid) FATAL on a
        # mismatch. Like move_belief_mode it gates EVERY load; same-run pool/sentinel snapshots carry the
        # identical mode so they pass trivially. The training-only win_prob_coef is NOT checked.
        # gen3_hp_belief_ablation_v1 (v53, like win_prob_mode): 'composed' builds HPTypeBelief and
        # 'flat' does not (a state_dict change), and the two produce different typed-HP posteriors (a
        # forward change). A STRING compare gates both. The training-only hp_type_belief_coef is NOT
        # checked.
        if self.hp_belief_mode != saved.hp_belief_mode:
            raise ModelVersionError(
                f"hp_belief_mode mismatch: saved={saved.hp_belief_mode!r}, "
                f"current={self.hp_belief_mode!r}.\n"
                "How the opponent's typed Hidden-Power belief is produced is fixed for a run's "
                "lifetime: 'composed' adds the HPTypeBelief head and the presence x type "
                "factorisation, 'flat' predicts the 16 typed channels independently.\n"
                "Resume with the matching --hp-belief-mode, or start a fresh training run."
            )
        if self.win_prob_mode != saved.win_prob_mode:
            raise ModelVersionError(
                f"win_prob_mode mismatch: saved={saved.win_prob_mode!r}, current={self.win_prob_mode!r}.\n"
                "The win-probability head is fixed for a run's lifetime: adding/removing it changes the "
                "state_dict, and switching read_only↔shaping flips whether its loss shapes the shared "
                "trunk (a silent mid-run training change).\n"
                "Resume with the matching --win-prob-mode setting, or start a fresh training run."
            )


        # v29 distributional VALUE head (like win_prob_mode): the MODE gates none↔head (the
        # ValueDistHead params) AND read_only↔shaping (grad-flow); the BIN COUNT is the head's output
        # Linear width. Both are weight-shape/forward changes → FATAL on a resume mismatch. The support
        # (vmin/vmax) is value-meaning → resume-only check_value_dist, not here.
        # (gen3_seed_quantile_v1's `seed_quantile` gate is DELETED at v78 with the head itself.)
        # gen3_value_threat_inject_v1 (v64): the injection projection is a state_dict-changing
        # module, AND the flag switches the op's reducer on, so a flip is doubly incompatible.
        if self.value_threat_inject != saved.value_threat_inject:
            raise ModelVersionError(
                f"value_threat_inject mismatch: saved={saved.value_threat_inject}, "
                f"current={self.value_threat_inject}.\n"
                "The critic's threat-injection projection is fixed for a run's lifetime: adding or "
                "removing it changes the state_dict, and the flag also switches the DamageOperator's "
                "pair reduction from hard_max to belief_mean — so a mid-run flip would change what "
                "the critic reads AND which modules exist.\n"
                "Resume with the matching --value-threat-inject setting, or start a fresh training run."
            )
        # gen3_opp_intent_v1 (v68): the alpha/beta pointer heads are state_dict-changing modules.
        if self.opp_intent != saved.opp_intent:
            raise ModelVersionError(
                f"opp_intent mismatch: saved={saved.opp_intent}, current={self.opp_intent}.\n"
                "The opponent-intent heads are fixed for a run's lifetime: adding or removing them "
                "changes the state_dict.\n"
                "Resume with the matching --opp-intent-coef setting, or start a fresh training run."
            )
        # gen3_t0_species_prior_v1 (v72): the state_dict is IDENTICAL either way (the co-occurrence
        # tables are non-persistent buffers and the module has no parameters), so — exactly as with
        # species_prior_fusion below — this compare is the ONLY thing that can reject a mid-run flip.
        # Nothing about the shapes would ever complain, while every unrevealed-defender damage number
        # the policy and critic were trained against would silently change under them.
        # gen3_intent_grad_mode_v1 (v73): flipping this mid-run changes what the shared trunk is
        # being trained to do, with no shape anywhere to notice.
        # gen3_intent_move_cell_v1 (v77): widens the pointer move scorer's in_features (a policy
        # state_dict change), so a mismatch would be shape-caught — this names the cause instead.
        if self.intent_move_cell != saved.intent_move_cell:
            raise ModelVersionError(
                f"intent_move_cell mismatch: saved={saved.intent_move_cell}, "
                f"current={self.intent_move_cell}.\n"
                "The G3 alpha-conditioned c2 move-cell channels widen the pointer move scorer, "
                "so the flag is fixed for a run's lifetime.\n"
                "Resume with the matching --intent-move-cell, or start a fresh run."
            )
        # gen3_unified_value_readout_v1 (v80): widens the value projection (a state_dict change),
        # so a mismatch would be shape-caught — this names the cause instead.
        if self.value_entity_pool != saved.value_entity_pool:
            raise ModelVersionError(
                f"value_entity_pool mismatch: saved={saved.value_entity_pool}, "
                f"current={self.value_entity_pool}.\n"
                "The unified critic entity pool widens the value projection, so the flag is "
                "fixed for a run's lifetime.\n"
                "Resume with the matching --value-entity-pool, or start a fresh run."
            )
        # gen3_unified_value_readout_v2 (v82): grows the pool's source table (state_dict).
        if self.value_entity_pool_full != saved.value_entity_pool_full:
            raise ModelVersionError(
                f"value_entity_pool_full mismatch: saved={saved.value_entity_pool_full}, "
                f"current={self.value_entity_pool_full}.\n"
                "The full row set grows the pool's source-embedding table, so the flag is "
                "fixed for a run's lifetime.\n"
                "Resume with the matching --value-entity-pool-full, or start a fresh run."
            )
        # gen3_item_belief_v1 (v83): builds the ItemBelief module (a state_dict change).
        if self.item_belief != saved.item_belief:
            raise ModelVersionError(
                f"item_belief mismatch: saved={saved.item_belief}, "
                f"current={self.item_belief}.\n"
                "The item-belief head adds trunk modules, so the flag is fixed for a run's "
                "lifetime.\n"
                "Resume with the matching --item-belief, or start a fresh run."
            )
        # gen3_pair_outcome_v1 (v93): one zero-init projection + a pointer-move-cell width
        # change (state_dict).
        if self.pair_outcome_cell != saved.pair_outcome_cell:
            raise ModelVersionError(
                f"pair_outcome_cell mismatch: saved={saved.pair_outcome_cell}, "
                f"current={self.pair_outcome_cell}.\n"
                "The unified outcome vector widens the pointer move cell, so the flag is fixed "
                "for a run's lifetime.\n"
                "Resume with the matching --pair-outcome-cell, or start a fresh run."
            )
        # gen3_pair_outcome_switch_v1 (v94): one zero-init projection + a pointer-SWITCH-cell
        # width change (state_dict).
        if self.pair_outcome_switch != saved.pair_outcome_switch:
            raise ModelVersionError(
                f"pair_outcome_switch mismatch: saved={saved.pair_outcome_switch}, "
                f"current={self.pair_outcome_switch}.\n"
                "The per-defender outcome row widens the pointer SWITCH cell, so the flag is "
                "fixed for a run's lifetime.\n"
                "Resume with the matching --pair-outcome-switch, or start a fresh run."
            )
        # gen3_switch_branch_v1 (v94): one zero-init projection + a pointer-move-cell width
        # change (state_dict).
        if self.switch_branch_cell != saved.switch_branch_cell:
            raise ModelVersionError(
                f"switch_branch_cell mismatch: saved={saved.switch_branch_cell}, "
                f"current={self.switch_branch_cell}.\n"
                "The OA2 / spinblock / Protect-mass cell widens the pointer move cell, so the "
                "flag is fixed for a run's lifetime.\n"
                "Resume with the matching --switch-branch-cell, or start a fresh run."
            )
        # gen3_conditional_threat_v1 (v95): one zero-init projection + a pointer-SWITCH-cell
        # width change (state_dict).
        if self.conditional_threat_cell != saved.conditional_threat_cell:
            raise ModelVersionError(
                f"conditional_threat_cell mismatch: saved={saved.conditional_threat_cell}, "
                f"current={self.conditional_threat_cell}.\n"
                "OA1's conditional-threat coordinates widen the pointer SWITCH cell, so the flag "
                "is fixed for a run's lifetime.\n"
                "Resume with the matching --conditional-threat-cell, or start a fresh run."
            )
        # gen3_pair_value_route_v1 (v95): one zero-init D_MODEL projection inside CLSPool. It
        # injects ADDITIVELY, so NO width moves anywhere and nothing shape-based can see the
        # difference except the extra state_dict key — the version gate carries this one.
        if self.pair_value_route != saved.pair_value_route:
            raise ModelVersionError(
                f"pair_value_route mismatch: saved={saved.pair_value_route}, "
                f"current={self.pair_value_route}.\n"
                "PV adds a zero-init injection into the critic's copy of our team tokens, so the "
                "flag is fixed for a run's lifetime.\n"
                "Resume with the matching --pair-value-route, or start a fresh run."
            )
        # gen3_intent_threshold_v1 (v84): two zero-init projections + width changes (state_dict).
        if self.intent_threshold != saved.intent_threshold:
            raise ModelVersionError(
                f"intent_threshold mismatch: saved={saved.intent_threshold}, "
                f"current={self.intent_threshold}.\n"
                "The threshold operator widens the pointer move cell and the critic, so the "
                "flag is fixed for a run's lifetime.\n"
                "Resume with the matching --intent-threshold, or start a fresh run."
            )
        # gen3_intent_conditional_v1 (v85): a zero-init projection + a pointer-cell width change.
        if self.intent_conditional != saved.intent_conditional:
            raise ModelVersionError(
                f"intent_conditional mismatch: saved={saved.intent_conditional}, "
                f"current={self.intent_conditional}.\n"
                "The mechanic cells widen the pointer move cell, so the flag is fixed for a "
                "run's lifetime.\n"
                "Resume with the matching --intent-conditional, or start a fresh run."
            )
        # gen3_op_lean_forward_v1 (v86): out_gain shape / d3 forward math.
        if self.op_drop_renders != saved.op_drop_renders:
            raise ModelVersionError(
                f"op_drop_renders mismatch: saved={saved.op_drop_renders}, "
                f"current={self.op_drop_renders}.\n"
                "The lean forward block shrinks out_gain, so the flag is fixed for a run's "
                "lifetime.\nResume with the matching --op-drop-renders, or start a fresh run."
            )
        if self.op_believed_lean != saved.op_believed_lean:
            raise ModelVersionError(
                f"op_believed_lean mismatch: saved={saved.op_believed_lean}, "
                f"current={self.op_believed_lean}.\n"
                "The believed-lean d3 physics are a forward-math change with no shape, so this "
                "gate is the ONLY thing that rejects a mismatched resume.\n"
                "Resume with the matching --op-believed-lean, or start a fresh run."
            )
        # gen3_event_window_v1 (v81): builds the EventSeats consumer (a state_dict change).
        if self.history_events != saved.history_events:
            raise ModelVersionError(
                f"history_events mismatch: saved={saved.history_events}, "
                f"current={self.history_events}.\n"
                "The H-B event seats add trunk modules, so the flag is fixed for a run's "
                "lifetime.\nResume with the matching --history-events, or start a fresh run."
            )
        if self.opp_intent_grad_mode != saved.opp_intent_grad_mode:
            raise ModelVersionError(
                f"opp_intent_grad_mode mismatch: saved={saved.opp_intent_grad_mode!r}, "
                f"current={self.opp_intent_grad_mode!r}.\n"
                "Whether the opponent-intent objective shapes the trunk is fixed for a run's "
                "lifetime.\nResume with the matching --opp-intent-grad-mode, or start a fresh run."
            )
        if self.t0_species_prior != saved.t0_species_prior:
            raise ModelVersionError(
                f"t0_species_prior mismatch: saved={saved.t0_species_prior}, "
                f"current={self.t0_species_prior}.\n"
                "The T0 species belief is fixed for a run's lifetime: it decides whether the physics "
                "prices an unrevealed opponent from the model's own team-composition belief or from "
                "the static gen3ou usage prior. Flipping it re-means every damage number against a "
                "hidden slot.\n"
                "Resume with the matching --t0-species-prior setting, or start a fresh training run."
            )
        # gen3_species_prior_fusion_v1 (v69): the state_dict is IDENTICAL either way (the co-occurrence
        # tables are non-persistent buffers), so this compare is the ONLY thing standing between a
        # resume and a silently re-meant species head — ON reads the head's output as a DELTA on the
        # team-composition prior, OFF reads the same numbers as the whole prediction.
        if self.species_prior_fusion != saved.species_prior_fusion:
            raise ModelVersionError(
                f"species_prior_fusion mismatch: saved={saved.species_prior_fusion}, "
                f"current={self.species_prior_fusion}.\n"
                "The species belief's prior fusion is fixed for a run's lifetime: flipping it changes "
                "what the species head's output MEANS (delta-on-prior vs. the full prediction), and "
                "nothing in the weights would catch it.\n"
                "Resume with the matching --species-prior-fusion setting, or start a fresh training run."
            )
        if self.value_dist_mode != saved.value_dist_mode:
            raise ModelVersionError(
                f"value_dist_mode mismatch: saved={saved.value_dist_mode!r}, current={self.value_dist_mode!r}.\n"
                "The distributional value head is fixed for a run's lifetime: adding/removing it changes "
                "the state_dict, and switching read_only↔shaping flips whether its loss shapes the shared "
                "trunk (a silent mid-run training change).\n"
                "Resume with the matching --value-dist-mode setting, or start a fresh training run."
            )
        # gen3_cf_evidential_head_v1 (v98): the evidential Beta readout's params are the state_dict
        # delta — a resume that flips this has either no weights for the head or weights with no
        # home. The head is never called by the forward, so NOTHING else would catch it: a mismatch
        # would load "successfully" and the term would silently supervise a freshly-random head (or
        # not exist at all) for the rest of the run.
        if self.cf_evidential != saved.cf_evidential:
            raise ModelVersionError(
                f"cf_evidential mismatch: saved={saved.cf_evidential}, current={self.cf_evidential}.\n"
                "The counterfactual EVIDENTIAL head is fixed for a run's lifetime: adding or removing "
                "it changes the state_dict, and because it is never called by the forward there is no "
                "shape error downstream that would catch the mismatch.\n"
                "Resume with the matching --cf-evidential setting, or start a fresh training run."
            )
        # gen3_cf_twin_heads_v1 (v99): the same argument as v98, twice. A head the forward never
        # calls produces NO shape error anywhere, so this bool compare is the only thing between a
        # flipped flag and a run that silently supervises freshly-random twins — or that loses the
        # within-run paired comparison the whole amendment exists for — for the rest of its life.
        if self.cf_twin_heads != saved.cf_twin_heads:
            raise ModelVersionError(
                f"cf_twin_heads mismatch: saved={saved.cf_twin_heads}, current={self.cf_twin_heads}.\n"
                "The counterfactual TWIN win-prob heads are fixed for a run's lifetime: building them "
                "changes the state_dict, and because they are never called by the forward there is no "
                "shape error downstream that would catch the mismatch. The arm's primary comparison is "
                "a WITHIN-RUN paired head difference, which a mid-run flip destroys.\n"
                "Resume with the matching --cf-twin-heads setting, or start a fresh training run."
            )
        if self.cf_shadow_critic != saved.cf_shadow_critic:
            raise ModelVersionError(
                f"cf_shadow_critic mismatch: saved={saved.cf_shadow_critic}, "
                f"current={self.cf_shadow_critic}.\n"
                "The SHADOW critic head is fixed for a run's lifetime: building it changes the "
                "state_dict, and because it is never called by the forward there is no shape error "
                "downstream that would catch the mismatch.\n"
                "Resume with the matching --cf-shadow-critic setting, or start a fresh training run."
            )
        # gen3_q_winprob_head_v1 (v107): the PER-ACTION win-prob head's params are the state_dict
        # delta. Unlike the four cf readouts above this one IS called by the forward — but it
        # writes only a STASH, so a mismatch still produces no shape error anywhere in pi/vf: a
        # resume that dropped the flag would load "successfully" and quietly stop publishing the
        # readout (and stop training it), while a resume that ADDED it would supervise a freshly
        # random head as if it were the run's trained one. A string compare is the only gate.
        if self.q_winprob_mode != saved.q_winprob_mode:
            raise ModelVersionError(
                f"q_winprob_mode mismatch: saved={saved.q_winprob_mode!r}, "
                f"current={self.q_winprob_mode!r}.\n"
                "The PER-ACTION win-probability head is fixed for a run's lifetime: building it "
                "changes the state_dict, and because it only writes a side stash there is no shape "
                "error downstream that would catch the mismatch.\n"
                "Resume with the matching --q-winprob-mode setting, or start a fresh training run."
            )
        if self.value_dist_bins != saved.value_dist_bins:
            raise ModelVersionError(
                f"value_dist_bins mismatch: saved={saved.value_dist_bins}, current={self.value_dist_bins}.\n"
                "The atom count is the value-dist head's output width — a different N is a weight-shape "
                "change.\n"
                "Resume with the matching --value-dist-bins setting, or start a fresh training run."
            )
        # gen3_unified_topk_incoming_v1 (v30): the discrete incoming move-space K scales the
        # DamageOperator out_dim → both projection in_features. Every distinct K (incl. 0↔N = adding/
        # removing the block) is a weight-shape change → a single unconditional int compare gates it
        # (like opp_belief_cls_k / value_dist_bins).
        if self.damage_topk_k != saved.damage_topk_k:
            raise ModelVersionError(
                f"damage_topk_k mismatch: saved={saved.damage_topk_k}, current={self.damage_topk_k}.\n"
                "The top-K incoming block's K is the number of opp moves surfaced — it scales the damage "
                "operator's output (hence both projection widths), so any change is a weight-shape "
                "mismatch.\n"
                "Resume with the matching --damage-topk setting, or start a fresh training run."
            )
        # gen3_per_move_matrices_v1 (v32): the outgoing per-move damage matrix widens the op out_dim → both
        # projection in_features. Toggling it is a weight-shape change (like damage_op).
        if self.damage_matrices_outgoing != saved.damage_matrices_outgoing:
            raise ModelVersionError(
                f"damage_matrices_outgoing mismatch: saved={saved.damage_matrices_outgoing}, "
                f"current={self.damage_matrices_outgoing}.\n"
                "The outgoing per-move damage matrix widens the damage operator's output (hence both "
                "projection widths), so toggling it is incompatible with a saved checkpoint.\n"
                "Resume with the matching --damage-matrices setting, or start a fresh training run."
            )
        # gen3_per_move_matrices_v1 (v33): the incoming per-move matrix widens the op out_dim → both
        # projection in_features (and supersedes topk). Toggling it is a weight-shape change (like damage_op).
        if self.damage_matrices_incoming != saved.damage_matrices_incoming:
            raise ModelVersionError(
                f"damage_matrices_incoming mismatch: saved={saved.damage_matrices_incoming}, "
                f"current={self.damage_matrices_incoming}.\n"
                "The incoming per-move damage matrix widens the damage operator's output (hence both "
                "projection widths), so toggling it is incompatible with a saved checkpoint.\n"
                "Resume with the matching --damage-matrices setting, or start a fresh training run."
            )
        # gen3_bidir_threat_trunk_v1 (v36): the uncertainty-aware P(outspeed) is a version-gated
        # forward-behavior toggle — fresh-only.
        if self.threat_prob_outspeed != saved.threat_prob_outspeed:
            raise ModelVersionError(
                f"threat_prob_outspeed mismatch: saved={saved.threat_prob_outspeed}, "
                f"current={self.threat_prob_outspeed}.\n"
                "It changes the P(outspeed) forward (uncertainty-aware scale), a version-checked "
                "forward-behavior change. Resume with the matching flag, or start a fresh run."
            )
