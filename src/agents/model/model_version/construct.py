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
        opp_belief_aux_coef: float = 0.0,
        move_belief_coef: float = 0.0,
        move_belief_latent_coef: float = 0.0,
        spread_belief_coef: float = 0.0,
        hp_type_belief_coef: float = 0.0,
        item_belief_coef: float = 0.0,
        arch_source: "str | None" = None,
        policy_grad_coef: float = 1.0,
        intent_label_bot_weight: float = 1.0,
        fork_fraction: float = 0.0,
        fork_branches: int = 3,
        fork_contested_gap: float = 0.40,
        fork_contested_absv: float = 0.0,
        fork_max_per_battle: int = 1,
        fork_crn: str = "dice_and_draws",
        eval_sentinel_greedy: bool = True,
        promote_threshold: float = 0.55,
        capacity_telemetry: bool = False,
        canary_reset_steps: int = 1_000_000,
        capacity_cosine_every: int = 50,
        capacity_velocity_every: int = 50,
        rank_tripwire: str = "warn",
        rank_tripwire_drop: float = 0.20,
        policy_gae_lambda: float = 0.80,
        diagnostics_every: int = 1,
        opp_intent_coef: float = 0.0,
        eval_mirrored_pairs: bool = False,
        promotion_sprt: bool = False,
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
            # The fallbacks track the RewardConfig defaults: a version built with reward_config=None
            # records what a default run actually trains with.
            draw_penalty=float(getattr(reward_config, "draw_penalty", -35.0)),
            victory_value=float(getattr(reward_config, "victory_value", 30.0)),
            progress_decision_tense=bool(getattr(reward_config, "progress_decision_tense", False)),
            progress_switch_freeze=bool(getattr(reward_config, "progress_switch_freeze", False)),
            terminal_indicator=bool(getattr(reward_config, "terminal_indicator", False)),
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
            # gen3_ridealong_heads_v1 (v126): the four ride-along declarations ride the extractor
            # kwargs (the policy builds the heads from them).
            ridealong_ensemble=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("ridealong_ensemble", 0)),
            ridealong_rnd=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("ridealong_rnd", False)),
            ridealong_adv=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("ridealong_adv", 0)),
            ridealong_opp=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("ridealong_opp", 0)),
            ridealong_rnd_variants=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("ridealong_rnd_variants",
                                                                       "off")),
            # gen3_x5_hypothesis_set_v1 (v136): X5's belief representation.
            belief_tokens=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("belief_tokens", "blob")),
            # gen3_oracle_reveal_v1 (v137): the DIAGNOSTIC observation mode (resume-immutable).
            oracle_reveal=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("oracle_reveal", "off")),
            damage_topk_k=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_topk_k", 0)
            ),
            damage_matrices_outgoing=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_matrices_outgoing", False)
            ),
            damage_matrices_incoming=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_matrices_incoming", False)
            ),
            hp_belief_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("hp_belief_mode", "composed")
            ),
            belief_grad_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("belief_grad_mode", "shaping")
            ),
            # gen3_winprob_critic_mode_v1: a POLICY kwarg, not an extractor one — the critic ROUTE
            # lives in Gen3DualHeadMaskablePolicy.
            critic=str(policy_kwargs.get("critic", "shaped")),
            hp_type_belief_coef=float(hp_type_belief_coef),
            item_belief_coef=float(item_belief_coef),
            arch_source=(str(arch_source) if arch_source else None),
            policy_grad_coef=float(policy_grad_coef),
            intent_label_bot_weight=float(intent_label_bot_weight),
            fork_fraction=float(fork_fraction or 0.0),
            fork_branches=int(fork_branches or 3),
            fork_contested_gap=float(fork_contested_gap or 0.40),
            fork_contested_absv=float(fork_contested_absv or 0.0),
            fork_max_per_battle=int(fork_max_per_battle or 1),
            fork_crn=str(fork_crn or "dice_and_draws"),
            eval_sentinel_greedy=bool(eval_sentinel_greedy),
            promote_threshold=float(promote_threshold),
            capacity_telemetry=bool(capacity_telemetry),
            canary_reset_steps=int(canary_reset_steps),
            capacity_cosine_every=int(capacity_cosine_every),
            capacity_velocity_every=int(capacity_velocity_every),
            rank_tripwire=str(rank_tripwire),
            rank_tripwire_drop=float(rank_tripwire_drop),
            policy_gae_lambda=float(policy_gae_lambda),
            diagnostics_every=int(diagnostics_every),
            opp_intent_coef=float(opp_intent_coef),
            eval_mirrored_pairs=bool(eval_mirrored_pairs),
            promotion_sprt=bool(promotion_sprt),
            opp_belief_aux_coef=float(opp_belief_aux_coef),
            move_belief_coef=float(move_belief_coef),
            move_belief_latent_coef=float(move_belief_latent_coef),
            spread_belief_coef=float(spread_belief_coef),
        )
