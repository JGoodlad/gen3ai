"""The live identity constants, the error type, and the reward-immutable field tables.

`MODEL_CONFIG_VERSION` and `ARCH_SIGNATURE` are the two numbers every other module in this
package compares against, so they live alone in the leaf of the import graph: nothing here
imports a sibling, which is what lets `migrations`, `fields`, `compat` and `resume_checks` all
read them without a cycle.

The per-version narratives above each constant are HISTORY and are preserved verbatim.
"""
from typing import Any, Dict


# Bump this whenever the ModelVersion schema changes (fields added/renamed/removed).
# Also add a migration case in _migrate_config().
#
# The per-version narrative (v3 -> v88: what each field means, which gate enforces it, and
# why) lives in designs/CHANGELOG.md under 'The MODEL_CONFIG_VERSION narrative' — moved
# there 2026-08-16; it is history, and this file keeps only the live machinery.
# v89 (gen3_value_pooled_routes_v1): the five value routes (intent_value_reduce v74,
#   value_entity_pool v80/82, intent_threshold's vf half v84, value_clock/value_intent v87 —
#   four of the five are DELETED at v96; `value_entity_pool` is the one that carried)
#   INJECT into `value_pooled` instead of the post-assembler vf concat, which
#   `--value-from-dist` structurally bypassed (gen-12 proof: their zero-init projections
#   bit-exact ZERO after 25M steps). Route out-widths become D_MODEL and the vf concat
#   narrows for flag-ON configs, so a <v89 checkpoint recording ANY of them ON carries
#   shapes the surviving code cannot load — REFUSED (the v75 rule); OFF stamps forward.
# v92 (gen3_td_consistency_aux_v1): `td_aux_coef` — the TD-consistency auxiliary's weight. A
#   TRAINING-only loss coefficient (the opp_belief_aux_coef class): recorded for provenance and
#   for flagless-resume read-back, never gated. A pre-v92 config defaults it to 0.0 = OFF.
#   ⚠️ Built as v90 and RENUMBERED: v90 (gen3_frame_deletion_v1) and v91
#   (gen3_event_semantics_v1) landed while this sat on a branch. No ARCH_SIGNATURE bump —
#   the term is computed in the PPO step, never in the extractor forward, so a coef-0 build
#   is byte-identical and there is nothing for `check_compatible` to gate.
# v95 (substrate Phase C): `conditional_threat_cell` (OA1 — the defensive-pivot coordinates on the
#   pointer SWITCH cell) and `pair_value_route` (PV — the α-reduced outcome row as TOKEN CONTENT on
#   the critic's copy of our team tokens). Both opt-in, zero-init, OFF byte-identical. The same bump
#   carries `gen3_status_economy_v1`, which AMENDS `tempo_cost`'s coordinate semantics under the
#   existing `pair_outcome_*` flags (the Natural Cure ability + the bench-cleric path become undo
#   paths; the reduction becomes a MIN over available paths). No ARCH_SIGNATURE bump — with the
#   flags OFF the forward is byte-identical — but a <v95 config recording either pair_outcome flag
#   ON is REFUSED rather than migrated, since it trained against different numbers.
# v96 (gen3_critic_route_wave_v1) — THE CRITIC-ROUTE DELETION WAVE. Seven audited-dead critic
#   routes are deleted in one pass, and with them the whole post-assembler vf tail:
#     * the v61 MultiSeedValueReadout + seed_diagnostics + the `value_seeds/*` TB contract
#       (dV 0.0000 bit-exact, gen-13 AND gen-14)
#     * the hidden-opp belief's VF half ONLY — its PI half flips 39.6% of argmaxes and STAYS
#     * the `non_matchup_rest` VF concat (0.0000; C1 measured the content substituting
#       through the global token) — its PI concat STAYS
#     * `value_intent` (0.156) · `intent_threshold`'s vf half (0.155/0.136) ·
#       `intent_value_reduce` (0.3176 at 2x) · `value_clock` (0.2169 at 2x), all vs a 0.39 bar
#   Three FIELDS go with them (`intent_value_reduce`, `value_clock`, `value_intent`); the other
#   four were unconditional or ride a surviving flag. `vf_combined` is now `value_pooled` alone,
#   which is what bumps ARCH_SIGNATURE: `value_projection` narrows and `assembler.seed_readout.*`
#   leaves the state_dict, and NOTHING in the config records either, so the signature is the only
#   gate that can reject a pre-v96 checkpoint with a diagnosis instead of an opaque torch error.
# v97 (gen3_intent_label_bot_weight_v1): `intent_label_bot_weight` — the per-sample weight on the
#   opponent-intent (alpha/beta) LABELS produced against a heuristic BOT. A TRAINING-only loss
#   weight (the td_aux_coef class): recorded for provenance and for flagless-resume read-back,
#   never gated. It scales a loss term computed in the PPO step and touches no forward pass, so a
#   default (1.0) build is bit-identical and there is nothing for `check_compatible` to compare.
#   A pre-v97 config defaults it to 1.0 = OFF. No ARCH_SIGNATURE bump.
# v98 (gen3_cf_evidential_head_v1): `cf_evidential` — the EVIDENTIAL Beta readout over P(win|state)
#   off `value_pooled` (the counterfactual label factory's rung R1). STRUCTURAL, exactly the
#   win_prob_mode / value_dist_mode precedent: building it adds the head's params to the state_dict,
#   so a resume that flips it has either no weights for the head or orphan weights, and a bool
#   compare in check_compatible is the gate. It is never called from the extractor forward (the
#   training-side loss applies it to the stashed `value_pooled`, always detached), so OFF is
#   byte-for-byte the baseline and ON-at-coefficient-0 is bit-identical in pi/vf too — hence NO
#   ARCH_SIGNATURE bump, the optional-side-head rule. A pre-v98 config defaults it to False = OFF.
#   Its two coefficients (`cf_evidential_coef`, `cf_evidential_reg`) are TRAINING-only argparse in
#   the `--opd-coef` / `--cf-winprob-coef` class and appear nowhere here.
# v99 (gen3_cf_twin_heads_v1): `cf_twin_heads` + `cf_shadow_critic` — the TWIN WIN-PROB HEADS and
#   the passive SHADOW CRITIC (the owner-authorized amendment to the signed R1 pre-registration;
#   ledger 2026-08-22 evening, "Three owner sign-offs" item 3). Two STRUCTURAL bools in exactly the
#   v98 mould: each builds modules whose params ARE the state_dict delta, neither is ever called
#   from the extractor forward (the training-side terms apply them to the stashed `value_pooled`,
#   always detached), so OFF is byte-for-byte the baseline and ON-at-coefficient-0 is bit-identical
#   in pi/vf. NO ARCH_SIGNATURE bump — optional side heads, obs family unchanged. Both are gated by
#   a bool compare in check_compatible, and the gate is the ONLY thing that can catch a flipped
#   flag, because a head the forward never calls produces no shape error anywhere. A pre-v99 config
#   defaults BOTH to False = OFF (not a guess: the modules did not exist). Their coefficients
#   (`cf_twin_coef`, `cf_shadow_coef`) are TRAINING-only argparse in the `--opd-coef` class and
#   appear nowhere here. TWO fields in ONE bump because they ship as one amendment.
# v100 (gen3_cf_coef_provenance_v1): the TEN counterfactual COEFFICIENTS (cf_records,
#   cf_records_keep, cf_winprob_coef, cf_head_only, cf_label_lag_steps, cf_label_likelihood,
#   cf_evidential_coef, cf_evidential_reg, cf_twin_coef, cf_shadow_coef) leave the `--opd-coef`
#   genre for the td_aux_coef one: all TRAINING-only (a loss in the PPO step; no forward read, no
#   weight shape) ⇒ RECORDED for provenance + flagless-resume read-back, NEVER gated.
#   ⚠️ THE DEFECT IS SILENT: an R1 arm resumed without re-typing `--cf-winprob-coef 1.0` kept
#   training and simply stopped applying the term it was launched to measure. Their three
#   STRUCTURAL companions (cf_evidential v98, cf_twin_heads/cf_shadow_critic v99) were already
#   recorded and GATED, so a resume could keep the head and lose the coefficient driving it.
#   A pre-v100 config defaults each to its ARGPARSE default — not a guess: the fields did not
#   exist, so that is what every such run ran with. No ARCH_SIGNATURE bump, no floor change.
# v101 (gen3_capacity_telemetry_v1): the FOUR live-capacity-telemetry knobs (capacity_telemetry,
#   canary_reset_steps, capacity_cosine_every, capacity_velocity_every) — v100's shape exactly:
#   TRAINING-only, RECORDED for provenance + flagless-resume read-back, NEVER gated. They are
#   weaker than v100's even: a cf coefficient at least scales a loss, whereas these fold nothing
#   into `loss` and write no `.grad`, so a run's parameter updates are bit-identical on or off.
#   They are recorded anyway because a DIAGNOSTIC whose provenance is unrecoverable is a number
#   nobody can interpret six months later — `metadata.json`'s `cli_args` is overwritten by every
#   resuming process, so `model_config.json` is the only durable record of what a run measured.
#   A pre-v101 config defaults each to its ARGPARSE default (not a guess: the fields did not
#   exist). No ARCH_SIGNATURE bump — the canary head is owned by the PPO object, not the
#   extractor, so no state_dict key and no forward changes.
# v102 (gen3_policy_grad_coef_v1): `policy_grad_coef` — the weight on the PPO POLICY-GRADIENT term itself
#   (`policy_grad_coef * policy_loss`; scales ONLY the clipped surrogate — never entropy, never the value
#   term, never an aux). A TRAINING-only loss coefficient, the td_aux_coef class exactly: recorded
#   for provenance and for flagless-resume read-back, never gated. 1.0 = the upstream expression
#   (byte-identical — the unscaled tensor is used); 0.0 = arm F's pure-distill/aux phase, the
#   value it exists for (design_advantage_gated_distillation.md §5 needed a way to run PPO with
#   the policy-gradient term OFF, and no flag could zero it). A pre-v102 config defaults it to
#   1.0 = upstream — not a guess: the term entered at an implicit 1.0 in every run ever made.
#   No ARCH_SIGNATURE bump — computed in the PPO step, never in the extractor forward.
# v103 (gen3_distill_target_gate_v1): the ADVANTAGE-GATED / ACTION-FORM DISTILLATION family + the
#   RANK TRIPWIRE (design_advantage_gated_distillation.md §3.1/§3.3/§4.1/§7.1). Seven TRAINING-only
#   knobs, the td_aux_coef class exactly — recorded for provenance + flagless-resume read-back,
#   never gated. `distill_target` ("kl" = the untouched full-distribution KL, the byte-identical
#   default; "action" = the teacher's top-K probabilities renormalized over the legal set —
#   `distill_topk`=1 ⇒ pure argmax CE, the §3.3 axis no arm had ever manipulated; K >= n_actions
#   recovers the KL), `distill_gate` "advantage" + `distill_gate_tau` (rung (a): a row fires only
#   where the teacher's argmax disagrees with the SAMPLED action AND the student's own NORMALIZED
#   advantage reads it as a mistake, Â < -τ — so the distill gradient pushes a logit PPO is
#   already pushing down, by construction), `distill_beta` (the AWR |Â| temperature, mirroring
#   search_teacher_beta), and `rank_tripwire`/`rank_tripwire_drop` (§4.1: the rank/policy_pr
#   EMA-vs-own-baseline watchdog, default "warn"; "abort" may stop learn() cleanly — it changes
#   WHEN training ends, never what a step computes). A pre-v103 config defaults each to its
#   argparse default — not a guess: "kl" is the one loss every such run trained with, and the
#   tripwire did not exist. No ARCH_SIGNATURE bump — nothing here touches a forward pass or a
#   weight shape.
# v104 (gen3_winprob_pbrs_v1; ai_v12 route 1 — designs/ai_v12/design_winprob_behavior_coupling.md):
#   `win_prob_pbrs_coef` — POTENTIAL-BASED REWARD SHAPING from the win-prob head. Every
#   transition's reward gains `coef * (gamma*phi(s') - phi(s))`, phi = the DETACHED sigmoid of the
#   win-prob logit, applied trainer-side to the rollout buffer before GAE. It is the FIRST knob in
#   this family that edits the REWARD STREAM rather than a loss term — worth saying, because the
#   provenance class is nevertheless td_aux_coef's exactly: no forward pass reads it, no weight
#   shape depends on it, so it is recorded for provenance + flagless-resume read-back and never
#   gated. 0.0 = OFF and the shaping module is not even imported (byte-identical). A pre-v104
#   config defaults it to 0.0 — not a guess: the flag did not exist, so no run could have used it.
#   No ARCH_SIGNATURE bump — the reward stream is not the network.
# v105 (gen3_clean_world_config_v1 + gen3_winprob_pbrs_source_v1): FIVE keys for the CLEAN-WORLD
#   arm. Four are resume-immutable VALUE-meaning reward fields — `hand_shaping` (the master
#   off-switch for all eight hand PBRS potentials AND the whole BIAS class, the composition
#   `--no-all-shaping-pbrs` could not reach because that flag is ALSO `_bias_term_active`'s master
#   gate), `pbrs_material` / `pbrs_belief` (the two potentials that had no flag at all) and
#   `victory_value` (the ±30 terminal, promoted off a module constant so ±1 is reachable by flag).
#   The fifth, `win_prob_pbrs_source`, is TRAINING-only provenance: the frozen checkpoint whose
#   win-prob head supplies φ. Every default IS the pre-v105 behaviour, so the migration is a plain
#   setdefault and a flagless run is byte-identical. No ARCH_SIGNATURE bump — no weight shape moves.
MODEL_CONFIG_VERSION = 105

# The one-line effect of each `belief_grad_mode`, for the migration notice. Keyed by the SAME strings
# as `features_extractor.BELIEF_GRAD_MODES` (which owns the legal set + the ValueError); the two are
# pinned to agree by `belief_grad_mode_test.py::test_every_mode_has_a_migration_notice`, so a fourth
# mode cannot ship with a silently generic notice.
_BELIEF_GRAD_MODE_EFFECT = {
    "shaping": "the belief-aux gradient now SHAPES the shared trunk, and PPO trains the heads.",
    "detached": "the belief-aux gradient now STOPS at the heads (the trunk is stop-grad on the read).",
    "label_only": "the belief heads are now trained by their SUPERVISED LABELS ALONE — no policy/value "
                  "gradient reaches them (their outputs are published stop-grad to every consumer).",
}

# Change this when the neural architecture changes structurally in a way that makes
# weights from a different signature incompatible (e.g. adding LSTM, replacing attention).
# Same-family dim changes (role_token_size 128→256) don't need a new signature —
# check_compatible() catches those via the dim fields.
#
# The signature-by-signature history (v2 -> gen3_ctx_dedup_v1: what broke weight
# compatibility each time, and why) lives in designs/CHANGELOG.md under 'The
# ARCH_SIGNATURE narrative' — moved there 2026-08-16.
ARCH_SIGNATURE = "gen3_critic_route_wave_v1"
class ModelVersionError(Exception):
    pass


# The resume-IMMUTABLE reward hparams, in the order `check_reward_config` reports them, each mapped
# to the value a config-shaped object is read with when it lacks the field. The DEFAULTS here track
# `agents.training.reward_manager.RewardConfig` and are pinned against it by
# `src/main/reward_defaults_test.py` — a divergence would make an absent field mean one thing to the
# reward and another to the version record, which is the drift class this whole file guards.
_REWARD_IMMUTABLE_FIELDS: Dict[str, Any] = {
    "bias_additivity": 1.0,
    "mat_alive_weight": 1.25,
    "bias_redesign": False,
    "switch_bias_weight": 0.0,
    "draw_penalty": -35.0,
    "self_ko_hp_penalty": 0.0,
    "drop_redundant_bias": False,
    "drop_switch_bias": False,
    "all_shaping_pbrs": True,
    "stall_pbrs": False,
    "no_progress_penalty": 0.15,
    # gen3_clean_world_config_v1 — the CLEAN-WORLD switches. Every default is today's behaviour.
    "hand_shaping": True,
    "pbrs_material": True,
    "pbrs_belief": True,
    "victory_value": 30.0,
}

# field -> the CLI flag that sets it. Bools use the BoolFlag `--no-` negation (the documented
# opt-out spelling); floats take their value positionally.
_REWARD_FIELD_FLAGS: Dict[str, str] = {
    "bias_additivity": "--bias-additivity",
    "mat_alive_weight": "--mat-alive-weight",
    "bias_redesign": "--bias-redesign",
    "switch_bias_weight": "--switch-bias-weight",
    "draw_penalty": "--draw-penalty",
    "self_ko_hp_penalty": "--self-ko-hp-penalty",
    "drop_redundant_bias": "--drop-redundant-bias",
    "drop_switch_bias": "--drop-switch-bias",
    "all_shaping_pbrs": "--all-shaping-pbrs",
    "stall_pbrs": "--stall-pbrs",
    "no_progress_penalty": "--no-progress-penalty",
    "hand_shaping": "--hand-shaping",
    "pbrs_material": "--pbrs-material",
    "pbrs_belief": "--pbrs-belief",
    "victory_value": "--victory-value",
}


def _reward_flag_repr(name: str, value: Any) -> str:
    """The exact CLI text that would set reward field `name` to `value` — what a resume must re-pass.

    Bools render as the bare flag or its `--no-` negation rather than `--flag false`: both parse
    (BoolFlag takes a value too), but the negation is the spelling the help text and the docs use,
    and an error message that teaches a second spelling costs more than it saves.
    """
    flag = _REWARD_FIELD_FLAGS[name]
    if isinstance(value, bool):
        return flag if value else f"--no-{flag[2:]}"
    return f"{flag} {value!r}"
