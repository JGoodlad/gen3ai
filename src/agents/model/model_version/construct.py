"""`from_layout_and_policy_kwargs` -- the one place a live `ModelVersion` is built from a run.

A mixin over `ModelVersionFields` so the constructor's 250 lines of keyword plumbing do not sit
between the field block and the gates.
"""
from __future__ import annotations

from typing import Any, Dict, Self

from agents.model.model_version.constants import ARCH_SIGNATURE, MODEL_CONFIG_VERSION
from agents.model.model_version.fields import ModelVersionFields


class ModelVersionConstruction(ModelVersionFields):
    """The `ModelVersion.from_layout_and_policy_kwargs` half of the class."""

    @classmethod
    def from_layout_and_policy_kwargs(
        cls,
        layout: Dict[str, Any],
        policy_kwargs: Dict[str, Any],
        vf_coef: float = 0.5,
        reward_config: Any = None,               # duck-typed: read only via getattr(_, default)
        value_tail_weight: float = 0.0,
        opp_belief_aux_coef: float = 0.0,
        move_belief_coef: float = 0.0,
        win_prob_coef: float = 1.0,
        move_belief_latent_coef: float = 0.0,
        spread_belief_coef: float = 0.0,
        value_dist_coef: float = 1.0,
        hp_type_belief_coef: float = 0.0,
        item_belief_coef: float = 0.0,
        td_aux_coef: float = 0.0,
        win_prob_pbrs_coef: float = 0.0,
        win_prob_pbrs_source: "str | None" = None,
        policy_grad_coef: float = 1.0,
        intent_label_bot_weight: float = 1.0,
        cf_records: bool = False,
        cf_records_keep: int = 512,
        cf_winprob_coef: float = 0.0,
        cf_head_only: bool = True,
        cf_label_lag_steps: int = 150_000,
        cf_label_likelihood: str = "binomial",
        cf_evidential_coef: float = 0.0,
        cf_evidential_reg: float = 1e-3,
        cf_twin_coef: float = 0.0,
        cf_shadow_coef: float = 0.0,
        capacity_telemetry: bool = False,
        canary_reset_steps: int = 1_000_000,
        capacity_cosine_every: int = 50,
        capacity_velocity_every: int = 50,
        distill_target: str = "kl",
        distill_topk: int = 1,
        distill_gate: str = "none",
        distill_gate_tau: float = 0.0,
        distill_beta: float = 1.0,
        rank_tripwire: str = "warn",
        rank_tripwire_drop: float = 0.20,
    ) -> Self:
        from agents.model.features_extractor import (
            ROLE_TOKEN_SIZE,
            PROJECTION_DIM,
            MOVE_NET_HIDDEN,
            ROLE_ENCODER_HIDDEN,
            NET_ARCH,
        )
        return cls(
            config_version=MODEL_CONFIG_VERSION,
            arch_signature=ARCH_SIGNATURE,
            species_embedding_dim=layout["species_embedding_dim"],
            max_species=layout["max_species"],
            move_embedding_dim=layout["move_embedding_dim"],
            max_moves=layout["max_moves"],
            item_embedding_dim=layout["item_embedding_dim"],
            max_items=layout["max_items"],
            ability_embedding_dim=layout["ability_embedding_dim"],
            max_abilities=layout["max_abilities"],
            type_embedding_dim=layout["type_embedding_dim"],
            max_types=layout["max_types"],
            total_dim=layout["total_dim"],
            active_context_dim=layout["active_context_dim"],
            role_token_size=ROLE_TOKEN_SIZE,
            projection_dim=PROJECTION_DIM,
            move_net_hidden=list(MOVE_NET_HIDDEN),
            role_encoder_hidden=list(ROLE_ENCODER_HIDDEN),
            net_arch=list(policy_kwargs.get("net_arch", NET_ARCH)),
            vf_coef=vf_coef,
            bias_additivity=float(getattr(reward_config, "bias_additivity", 1.0)),
            mat_alive_weight=float(getattr(reward_config, "mat_alive_weight", 1.25)),
            bias_redesign=bool(getattr(reward_config, "bias_redesign", False)),
            switch_bias_weight=float(getattr(reward_config, "switch_bias_weight", 0.0)),
            # The two getattr fallbacks below track the RewardConfig defaults (owner decision
            # 2026-08-18): a version built with reward_config=None must record the composition a
            # default run actually trains with, not the superseded one.
            draw_penalty=float(getattr(reward_config, "draw_penalty", -35.0)),
            self_ko_hp_penalty=float(getattr(reward_config, "self_ko_hp_penalty", 0.0)),
            drop_redundant_bias=bool(getattr(reward_config, "drop_redundant_bias", False)),
            drop_switch_bias=bool(getattr(reward_config, "drop_switch_bias", False)),
            all_shaping_pbrs=bool(getattr(reward_config, "all_shaping_pbrs", True)),
            stall_pbrs=bool(getattr(reward_config, "stall_pbrs", False)),
            no_progress_penalty=float(getattr(reward_config, "no_progress_penalty", 0.15)),
            # gen3_clean_world_config_v1 — the CLEAN-WORLD switches; the fallbacks track the
            # RewardConfig defaults, i.e. today's behaviour.
            hand_shaping=bool(getattr(reward_config, "hand_shaping", True)),
            pbrs_material=bool(getattr(reward_config, "pbrs_material", True)),
            pbrs_belief=bool(getattr(reward_config, "pbrs_belief", True)),
            victory_value=float(getattr(reward_config, "victory_value", 30.0)),
            progress_decision_tense=bool(getattr(reward_config, "progress_decision_tense", False)),
            progress_switch_freeze=bool(getattr(reward_config, "progress_switch_freeze", False)),
            use_popart=bool(policy_kwargs.get("use_popart", False)),
            attend_unrevealed_opponents=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "attend_unrevealed_opponents", False)
            ),
            opp_belief_cls_k=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("opp_belief_cls_k", 0)
            ),
            opp_belief_slots=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("opp_belief_slots", False)
            ),
            move_belief_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("move_belief_mode", "off")
            ),
            damage_op=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_op", False)
            ),
            damage_outgoing=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_outgoing", False)
            ),
            move_candidate_floor=float(
                policy_kwargs.get("features_extractor_kwargs", {}).get("move_candidate_floor", 0.02)
            ),
            move_latent=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("move_latent", False)
            ),
            spread_belief=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("spread_belief", False)
            ),
            spread_belief_nature=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("spread_belief_nature", False)
            ),
            move_prior_fusion=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("move_prior_fusion", False)
            ),
            damage_candidate_k=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_candidate_k", 0)
            ),
            consequence_topk=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("consequence_topk", 6)
            ),
            entity_topk_seats=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("entity_topk_seats", 0)
            ),
            edge_bias_families=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("edge_bias_families", "off")
            ),
            entity_tail_seats=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("entity_tail_seats", False)
            ),
            win_prob_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("win_prob_mode", "none")
            ),
            value_dist_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("value_dist_mode", "none")
            ),
            value_dist_bins=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("value_dist_bins", 0)
            ),
            value_threat_inject=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("value_threat_inject", False)
            ),
            opp_intent=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("opp_intent", False)
            ),
            t0_species_prior=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("t0_species_prior", False)
            ),
            opp_intent_grad_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "opp_intent_grad_mode", "detached")
            ),
            intent_move_cell=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "intent_move_cell", False)
            ),
            value_entity_pool=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "value_entity_pool", False)
            ),
            history_events=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "history_events", False)
            ),
            value_entity_pool_full=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "value_entity_pool_full", False)
            ),
            item_belief=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "item_belief", False)
            ),
            intent_threshold=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "intent_threshold", False)
            ),
            intent_conditional=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "intent_conditional", False)
            ),
            pair_outcome_cell=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "pair_outcome_cell", False)
            ),
            pair_outcome_switch=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "pair_outcome_switch", False)
            ),
            switch_branch_cell=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "switch_branch_cell", False)
            ),
            conditional_threat_cell=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "conditional_threat_cell", False)
            ),
            pair_value_route=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "pair_value_route", False)
            ),
            op_drop_renders=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "op_drop_renders", False)
            ),
            op_believed_lean=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "op_believed_lean", False)
            ),
            species_prior_fusion=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("species_prior_fusion", False)
            ),
            cf_evidential=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("cf_evidential", False)
            ),
            cf_twin_heads=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("cf_twin_heads", False)
            ),
            cf_shadow_critic=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("cf_shadow_critic", False)
            ),
            value_dist_vmin=float(
                policy_kwargs.get("features_extractor_kwargs", {}).get("value_dist_vmin", 0.0)
            ),
            value_dist_vmax=float(
                policy_kwargs.get("features_extractor_kwargs", {}).get("value_dist_vmax", 0.0)
            ),
            damage_topk_k=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_topk_k", 0)
            ),
            damage_matrices_outgoing=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_matrices_outgoing", False)
            ),
            damage_matrices_incoming=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_matrices_incoming", False)
            ),
            threat_prob_outspeed=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("threat_prob_outspeed", False)
            ),
            hp_belief_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("hp_belief_mode", "composed")
            ),
            belief_grad_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("belief_grad_mode", "shaping")
            ),
            value_from_dist=bool(policy_kwargs.get("value_from_dist", False)),
            hp_type_belief_coef=float(hp_type_belief_coef),
            item_belief_coef=float(item_belief_coef),
            td_aux_coef=float(td_aux_coef),
            win_prob_pbrs_coef=float(win_prob_pbrs_coef),
            win_prob_pbrs_source=(str(win_prob_pbrs_source) if win_prob_pbrs_source else None),
            policy_grad_coef=float(policy_grad_coef),
            intent_label_bot_weight=float(intent_label_bot_weight),
            cf_records=bool(cf_records),
            cf_records_keep=int(cf_records_keep),
            cf_winprob_coef=float(cf_winprob_coef),
            cf_head_only=bool(cf_head_only),
            cf_label_lag_steps=int(cf_label_lag_steps),
            cf_label_likelihood=str(cf_label_likelihood),
            cf_evidential_coef=float(cf_evidential_coef),
            cf_evidential_reg=float(cf_evidential_reg),
            cf_twin_coef=float(cf_twin_coef),
            cf_shadow_coef=float(cf_shadow_coef),
            capacity_telemetry=bool(capacity_telemetry),
            canary_reset_steps=int(canary_reset_steps),
            capacity_cosine_every=int(capacity_cosine_every),
            capacity_velocity_every=int(capacity_velocity_every),
            distill_target=str(distill_target),
            distill_topk=int(distill_topk),
            distill_gate=str(distill_gate),
            distill_gate_tau=float(distill_gate_tau),
            distill_beta=float(distill_beta),
            rank_tripwire=str(rank_tripwire),
            rank_tripwire_drop=float(rank_tripwire_drop),
            value_tail_weight=float(value_tail_weight),
            opp_belief_aux_coef=float(opp_belief_aux_coef),
            move_belief_coef=float(move_belief_coef),
            win_prob_coef=float(win_prob_coef),
            move_belief_latent_coef=float(move_belief_latent_coef),
            spread_belief_coef=float(spread_belief_coef),
            value_dist_coef=float(value_dist_coef),
        )
