"""The migration chain: `MIGRATION_FLOOR`, the signature pairing, and `_migrate_config`.

⚠️ The `PRE-FLOOR MIGRATION HISTORY (v2-v95)` comment inside `_migrate_config` is a DELIBERATE
ARCHIVE, not dead prose: it records what each deleted `if version < N` branch injected or popped,
which is the only surviving statement of what an archived checkpoint's config meant. It is
preserved here verbatim and must not be trimmed.
"""
from agents.model.model_version.constants import ARCH_SIGNATURE, ModelVersionError


# The migration floor: the first MODEL_CONFIG_VERSION stamped with the current ARCH_SIGNATURE.
# Every `if version < N` migration branch with N <= this floor could only ever produce a config
# that the arch_signature gate — run by every consumer immediately after migration (snapshot.py's
# load paths, the fixed-opponent pool, train resume) — rejects anyway, so `_migrate_config`
# refuses pre-floor configs outright instead of walking dead branches.
# ⚠️ When ARCH_SIGNATURE next changes, raise this floor to the new signature's first stamped
# version IN THE SAME COMMIT (and append the pairing to SIGNATURE_FIRST_VERSION below) —
# migration_floor_test.py fails if the two drift apart.
MIGRATION_FLOOR = 96

# The signature → first-stamped-version pairing the floor is derived from. Append-only: add the
# new signature's row when it lands. migration_floor_test.py asserts
# MIGRATION_FLOOR == SIGNATURE_FIRST_VERSION[ARCH_SIGNATURE].
SIGNATURE_FIRST_VERSION = {
    "gen3_deadline_clock_v1": 67,
    "gen3_ctx_dedup_v1": 76,
    "gen3_frame_deletion_v1": 90,
    "gen3_event_semantics_v1": 91,
    "gen3_critic_route_wave_v1": 96,
}
def _migrate_config(data: dict) -> dict:
    """Apply incremental forward-migrations to bring an old config up to the current schema.

    Configs older than MIGRATION_FLOOR (the first version stamped with the current
    ARCH_SIGNATURE) are REFUSED outright rather than migrated: every loader that consumes a
    migrated config gates on `arch_signature` immediately afterwards (snapshot.py's load paths,
    the fixed-opponent pool, train resume), so a pre-floor migration could only ever produce a
    config that check_compatible rejects a moment later. Every `if version < N` branch with
    N <= MIGRATION_FLOOR was therefore dead code and has been deleted; the history each one
    recorded is preserved in the comment block below (and, in narrative form, in the
    version-history comments above MODEL_CONFIG_VERSION and in designs/CHANGELOG.md).

    What is left is exactly two kinds of live code: the version-INDEPENDENT sanitizers (they
    run at EVERY version, because a stale key must leave the dict whatever vintage wrote it —
    `cls(**data)` TypeErrors on one), and the genuinely post-floor `if version < N` branches
    with N > MIGRATION_FLOOR.
    """
    version = data.get("config_version", 1)
    if version < MIGRATION_FLOOR:
        raise ModelVersionError(
            f"config_version {version} is a PRE-GENERATION checkpoint: it predates "
            f"v{MIGRATION_FLOOR}, the first MODEL_CONFIG_VERSION stamped with the current "
            f"ARCH_SIGNATURE ({ARCH_SIGNATURE!r}). A checkpoint from an earlier generation "
            "cannot be loaded by this code — its weights were trained against an architecture "
            "this codebase no longer contains, and no config migration can bridge that.\n"
            "To re-probe it, use the git_hash recorded in the checkpoint's own metadata.json "
            "(git checkout <git_hash> and probe from there — the prober prints exactly this "
            "diagnosis, with the hash, for archived runs)."
        )
    # ----------------------------------------------------------------------------------------
    # PRE-FLOOR MIGRATION HISTORY (v2–v95) — documentation, not code. The executable branches
    # were deleted once MIGRATION_FLOOR rose past them. What each one injected (setdefault) or
    # removed (POP), verbatim from the deleted code:
    #   v2:  n_history_turns=1 (old models used a single TurnDelta).
    #   v3:  vf_coef=0.5 (the SB3 default every pre-flag run trained with).
    #   v4:  reward-config hparams — bias_additivity=1.0, mat_alive_weight=1.25,
    #        bias_redesign=False (pre-flag runs used the single-variable defaults).
    #   v5:  switch_bias_weight=0.0 (the switch-bias lever; absent = OFF).
    #   v6:  use_popart=False.
    #   v7:  draw_penalty=-30.0 (old runs scored a tie/timeout as a decisive loss).
    #   v8:  attend_unrevealed_opponents=False (old models key-masked unrevealed opp slots).
    #   v9:  opp_belief_cls_k=0 (no belief module); POPped the interim never-shipped
    #        `opp_belief_cls` bool.
    #   v10: value_active_readout=False (old value heads did not read the active-mon token).
    #   v11: value_tail_weight=0.0 (plain MSE value loss).
    #   v12: self_ko_hp_penalty=0.0 (the symmetric material PBRS priced a healthy
    #        Explosion/Self-Destruct trade at ~0).
    #   v13: drop_redundant_bias=False, drop_switch_bias=False (old runs kept every BIAS term).
    #   v14: all_shaping_pbrs=False, no_progress_penalty=0.15 (end-state PBRS switch).
    #   v15: stall_pbrs=False (the companion switch).
    #   v16: opp_belief_slots=False, opp_belief_aux_coef=0.0 (in-place hidden-opp belief-aux).
    #   v17: move_belief_mode="off", move_belief_coef=0.0 (no MoveBelief module).
    #   v18: opp_belief_latent=False, opp_belief_latent_coef=0.0 (BeliefHead latent predictor).
    #   v19: damage_op=False (no DamageOperator).
    #   v20: move_prior_fusion=False (no unified-move-belief prior fusion).
    #   v21: stamp — the incoming-damage-obs ablation toggle (its field went at v48).
    #   v22: win_prob_mode="none", win_prob_coef=1.0 (the tri-state win-probability head).
    #   v23: damage_outgoing=False, move_candidate_floor=0.0 (incoming-only op + the un-gated
    #        0.02-floor-less legacy prior).
    #   v24: move_latent=False, move_belief_latent_coef=0.0 (MoveLatentEncoder).
    #   v25: spread_belief=False, spread_belief_coef=0.0.
    #   v26: stamp — gen3_unified_op_physics_v1 (op physics parity: boosts/burn/weather/para/
    #        fixed-damage, intrinsic to damage_op; values-only).
    #   v27: stamp — gen3_unified_status_landing_v1: the op's OUTGOING direction gained the
    #        per-OUR-move status-landing block, Leech Seed's Grass immunity, Sleep Clause and
    #        Substitute-blocks-status. Intrinsic to damage_outgoing (out_dim grew, so a v26
    #        damage_outgoing checkpoint failed the SB3 load_state_dict projection in_features —
    #        the projection input dim is runtime-discovered, not a ModelVersion field).
    #   v28: stamp — gen3_unified_choice_band_v1: the op priced Choice Band (our known CB ×1.5
    #        physical Atk deterministically OUTGOING; a DECORRELATED CB-conditional physical tail
    #        [phys_high_cb + P(OHKO|CB)] + shared p_cb INCOMING — OHKO is a nonlinear threshold a
    #        mean-field ×(1+0.5·p_cb) would blur). Intrinsic to damage_op.
    #   v29: value_dist_mode="none", value_dist_bins=0, value_dist_vmin=0.0, value_dist_vmax=0.0,
    #        value_dist_coef=1.0 (the distributional value head, Phase A).
    #   v30: damage_topk_k=0 (gen3_unified_topk_incoming_v1 — the discrete top-K incoming block;
    #        STRUCTURAL int gated in check_compatible).
    #   v31: stamp — `damage_reattend` added (deleted at v71; no setdefault, so an ABSENT key
    #        stayed absent and the v71 judge only saw RECORDED values).
    #   v32: stamp — `move_belief_prefuse` added (same v71 treatment).
    #   v33: stamp — gen3_iterative_damage_v1: `damage_refine_rounds` (deleted at v70).
    #   v34: damage_matrices_outgoing=False (gen3_per_move_matrices_v1, our 4 moves × opp active
    #        + revealed bench).
    #   v35: damage_matrices_incoming=False (the incoming per-move matrix over the enriched
    #        top-K; reuses damage_topk_k as its K).
    #   v36: threat_prob_outspeed=False (gen3_bidir_threat_trunk_v1; its siblings
    #        threat_refine_outgoing / threat_unrevealed_outgoing were deleted at v70).
    #   v37: stamp — gen3_status_trunk_v1: `threat_status_refine` (deleted at v70).
    #   v38: hp_type_belief_coef=0.0 (gen3_opp_hp_type_belief_v1; the v38 mode key was POPped by
    #        the v52 migration).
    #   v39: damage_matrices_outgoing_all=False (the TRANSPOSED outgoing matrix — our 6 mons'
    #        moves → opp active).
    #   v40: spread_belief_nature=False (the SpreadBelief nature/EV generative head).
    #   v41: belief_grad_mode="shaping" (gen3_belief_grad_mode_v1; detach() is value-preserving,
    #        so 'shaping' reproduced the v40 forward AND backward byte-for-byte).
    #   v42: stamp — turn-history depth cut N_HISTORY_TURNS 10 → 7 (obs 3469 → 2992); the
    #        obs-dim weight-field check carried the rejection.
    #   v43: pubval_mode="none", pubval_coef=0.0 (gen3_pubval_aux_v1).
    #   v44: zarch_film="off", zarch_dim=0, zarch_recon_coef=0.0, zarch_vicreg_coef=0.0
    #        (gen3_zarch_film_v1).
    #   v45: value_from_dist=False (gen3_dist_critic_v1 Phase B; resume-immutable via
    #        check_value_from_dist, excluded from check_compatible).
    #   v46: zarch_lut="off", zarch_lut_teams=0 (gen3_zarch_lut_v1).
    #   v47: stamp — gen3_belief_single_compute_v1: `move_belief_single_compute` (deleted at v70).
    #   v48: gen3_cpu_damage_deleted_v1 — POPped the three `--unified-obs` ablation fields
    #        (mask_incoming_damage_obs, mask_active_move_scalars_obs, mask_move_effects_obs)
    #        along with the obs blocks they masked (obs 2992 → 2889). POP rather than setdefault:
    #        `from_json_file` does `cls(**data)`, so a stale key would TypeError before the clear
    #        obs-dim rejection.
    #   v49: damage_candidate_k=0 (the op candidate cap; the `pointer_head` bool it also added
    #        was deleted at v51).
    #   v50: stamp — `damage_op_prefuse` added (deleted at v71).
    #   v51: gen3_pointer_native_v1 — POPped `pointer_head` (the pointer head became THE action
    #        head, unconditionally; POP for the v48 reason).
    #   v52: gen3_typed_hp_belief_v1 — POPped `hp_type_belief_mode` (the opponent's Hidden Power
    #        is composed into the 16 typed move-nums inside the belief, so the tri-state had
    #        nothing left to gate; the belief's forward math changed while projection widths did
    #        not, so the ARCH_SIGNATURE bump was the only rejection).
    #   v53: hp_belief_mode="composed" (gen3_hp_belief_ablation_v1; 'flat' is the opt-in
    #        ablation, 'composed' reproduced the v52 forward exactly).
    #   v54: entity_topk_seats=0 (gen3_entity_move_seats_v1; E3 rode the ARCH_SIGNATURE).
    #   v55: stamp — gen3_op_block_trim_v1 (the op's output shrank by 28 dims; the deleted lean
    #        top-K block re-meant `damage_topk_k`; signature carried the break).
    #   v56: edge_bias_families="off" (gen3_edge_bias_trunk_v1; the layer swap itself rode the
    #        ARCH_SIGNATURE).
    #   v57: entity_tail_seats=False (gen3_entity_tail_seats_v1).
    #   v58: stamp — the SpD-as-speed GIGO fix (pairwise_speed/pairwise_boost read stat index 4
    #        as "speed"; values-only — a pre-v58 checkpoint's v_map/c1_map trained against the
    #        buggy feature).
    #   v59: consequence_topk=4 (pre-v59 models trained with the hardcoded k_cand/k_bench = 4).
    #   v60: stamp — gen3_entity_rehome_v1 (signature carried the break).
    #   v61: stamp — gen3_no_concat_v1 (the head-concat deletion + the multi-seed critic
    #        readout; signature carried the break).
    #   v62: value_seed_vicreg_coef=0.0 (gen3_seed_vicreg_v1; resume-immutable, the vf_coef
    #        class).
    #   v63: seed_quantile=False (gen3_seed_quantile_v1; structural, one shared Linear).
    #   v64: value_threat_inject=False (gen3_value_threat_inject_v1; structural, one shared
    #        zero-init Linear on the value pool's copy of our tokens).
    #   v65: stamp — gen3_unconditional_move_legality_v1. `move_candidate_floor` was deliberately
    #        left EXACTLY as recorded (0.0 on every pre-v65 run): 0.0 used to mean "legality
    #        OFF", is no longer a valid value, and check_compatible rejects it against the new
    #        0.02 default with a dedicated message rather than silently migrating 0.0 → 0.02
    #        (which would load a checkpoint whose policy trained on a prior that gave phantom
    #        mass to unlearnable moves).
    #   v66: gen3_nature_marginalize_removed_v1 — POPped `spread_belief_nature_marginalize`.
    #        The kernel (`DamageOperator._nature_marg_ko`) was DELETED, not defaulted off:
    #        measured on gen-8's own checkpoint over 1,075,200 ALIVE (defender, candidate)
    #        cells, |ΔP(KO)| vs mean-field was p50/p90/p95 = 0.00000, p99 = 0.00047, mean
    #        0.0003, and only 0.39% of cells moved by >0.02 — the nature posterior is peaked
    #        (top-1 mass 0.75, entropy 0.64 of 3.22 nats), so integrating over it ≈ evaluating
    #        at its mode (ledger K1's shape: sound theory, absent magnitude). It also computed a
    #        WRONG answer on empty slots: `dmg.clamp(min=eps)` turned zero damage into 1e-6, and
    #        with cur_hp also 0 the ramp gave 1e-6/(0.15e-6+1e-6) = 0.8696 — a spurious 87% KO on
    #        a slot with nothing in it (gated downstream, believed harmless, but a trap). The
    #        KEPT half is `--spread-belief-nature`, the generative nature/EV head — the actual
    #        correctness fix (it makes the largest-EV order-statistic bias structurally
    #        unrepresentable), and why `belief/spread_largest_bias` closed -26 → -12.8 on gen-8.
    #   v67: stamp — gen3_deadline_clock_v1: the obs CLOCK group 1 → 3 scalars (GLOBAL_ENV_DIM
    #        18 → 20, obs 2667 → 2669). No model_config field — the break is in the obs width +
    #        weight shapes, which ARCH_SIGNATURE carries. The first version of the
    #        gen3_deadline_clock_v1 generation (the gen-8/gen-9 world).
    #   v68: gen3_opp_intent_v1 — opp_intent=False (the alpha/beta intent heads).
    #   v69: gen3_species_prior_fusion_v1 — species_prior_fusion=False.
    #   v70: gen3_refine_loop_removed_v1 — POPped the refine loop's five dead fields
    #        (damage_refine_rounds, threat_refine_outgoing, threat_unrevealed_outgoing,
    #        threat_status_refine, move_belief_single_compute — 0-rounds/unreachable/INERT in
    #        every production config; the expected-latent outgoing math itself was re-homed,
    #        unconditional, by gen3_unrevealed_outgoing_prior_v1).
    #   v71: gen3_tiered_pipeline_v1 — the PRE-transformer placement became the only one;
    #        POPped move_belief_prefuse/damage_op_prefuse/damage_reattend, REFUSING (not
    #        defaulting) any config that recorded the deleted POST placement, because that
    #        toggle changed no weight shape and nothing downstream could catch a silent pop.
    #   v72: gen3_t0_species_prior_v1 — t0_species_prior=False.
    #   v73: gen3_intent_grad_mode_v1 — opp_intent_grad_mode="detached".
    #   v74: gen3_intent_value_reduce_v1 — intent_value_reduce=False.
    #   v75: SimSiam latent belief DELETED — POPped opp_belief_latent/opp_belief_latent_coef,
    #        REFUSING a config that recorded opp_belief_latent=True (it carried PARAMETERS the
    #        live extractor has no home for; the zip-kwargs sanitizer keeps that judgment).
    #   v76: stamp — gen3_ctx_dedup_v1 (signature carried the break; the floor sat here until
    #        gen3_frame_deletion_v1 raised it to 90).
    #   v77: intent_move_cell=False (gen3_intent_move_cell_v1, the G3 alpha-conditioned c2
    #        move-cell channels).
    #   v78: gen3_flag_surface_p1_v1 — POPped the zarch family + the seed-pressure pair
    #        (zarch_film, zarch_dim, zarch_lut, zarch_lut_teams, zarch_recon_coef,
    #        zarch_vicreg_coef, seed_quantile, value_seed_vicreg_coef), REFUSING a config that
    #        recorded zarch_film != "off" or seed_quantile=True (both built MODULES — the v75
    #        rule). The z_arch/FiLM conditioning line closed on the LUT arm's +0.024,
    #        CI [-0.016, +0.064]; both seed pressures capped at ~1-D of k=4, so the shared
    #        readout, not the coefficient, was the binding constraint.
    #   v79: stamp — gen3_pair_history_v1 (the H-A obs widening; total_dim + the widened encoder
    #        shapes carried the break, and the "h" edge family rides edge_bias_families).
    #   v80: value_entity_pool=False (gen3_unified_value_readout_v1).
    #   v81: history_events=False (gen3_event_window_v1; the H-B obs widening itself was
    #        weight-field-caught on total_dim).
    #   v82: value_entity_pool_full=False (gen3_unified_value_readout_v2).
    #   v83: item_belief=False (gen3_item_belief_v1).
    #   v84: intent_threshold=False (gen3_intent_threshold_v1).
    #   v85: intent_conditional=False (gen3_intent_conditional_v1).
    #   v86: op_drop_renders=False, op_believed_lean=False (gen3_op_lean_forward_v1).
    #   v87: stamp — gen3_value_direct_routes_v1 introduced value_clock/value_intent. Both FIELDS
    #        died at v96, so the branch had nothing left to default in; the v96 sanitizer below
    #        POPs them instead.
    #   v88: gen3_dead_flag_purge_v1 — the stamp only. Its POP/REFUSE half is version-INDEPENDENT
    #        and survives below.
    #   v89: gen3_value_pooled_routes_v1 — REFUSED a <v89 config recording intent_value_reduce /
    #        value_entity_pool / intent_threshold / value_clock / value_intent ON: the routes were
    #        re-homed from the vf-tail concat into `value_pooled`, so their projection shapes and
    #        the vf concat width changed. OFF stamped forward (the routes built nothing).
    #   v90: gen3_frame_deletion_v1 — POPped n_history_turns. The 7x159 TurnDelta lag frames and
    #        the 11-dim prev-turn action mask left the observation (3529 -> 2437, net -1092 after
    #        the H-B window's new cant column); a `_WEIGHT_FIELDS` break on total_dim plus an
    #        ARCH_SIGNATURE bump, so check_compatible refused a pre-v90 checkpoint first.
    #   v91: stamp — gen3_event_semantics_v1: the H-B event row gained faint_cause_id (col 20) and
    #        item_transition (col 21), closing the last two coverage gaps the frame-deletion audit
    #        found (obs 2437 -> 2501). Width change ⇒ total_dim + signature carried the break.
    #   v92: td_aux_coef=0.0 (gen3_td_consistency_aux_v1; TRAINING-only, no forward/weight shape).
    #   v93: pair_outcome_cell=False (gen3_pair_outcome_v1).
    #   v94: pair_outcome_switch=False, switch_branch_cell=False (substrate Phase B).
    #   v95: conditional_threat_cell=False, pair_value_route=False (substrate Phase C). The same
    #        bump carried gen3_status_economy_v1, which AMENDED `tempo_cost`'s coordinate semantics
    #        under the existing pair_outcome_* flags (Natural Cure + the bench-cleric path became
    #        undo paths; the reduction became a MIN over available paths rather than a MAX) — so
    #        the branch REFUSED a <v95 config recording either pair_outcome flag ON, since it
    #        trained against different numbers. Neither flag was ever enabled in a run, so that
    #        guard was latent, never a migration path.
    #   v96: gen3_critic_route_wave_v1 — the stamp only. Its POP/REFUSE half for
    #        intent_value_reduce / value_clock / value_intent is version-INDEPENDENT and survives
    #        below.
    # ----------------------------------------------------------------------------------------
    # ---- VERSION-INDEPENDENT SANITIZERS -----------------------------------------------------
    # These are NOT migration branches and are not floored away: they run at EVERY version,
    # because `from_json_file` does `cls(**data)` and a stale key TypeErrors there. Each POPs a
    # deleted field, and applies the v75 rule to its value — a recorded ON value named
    # parameters/widths the surviving code cannot rebuild ⇒ REFUSE with the re-read diagnosis;
    # OFF pops silently (the module was never built, so the weights are unaffected).
    #
    # v88 (gen3_dead_flag_purge_v1):
    for _dead, _ok in (("value_active_readout", False),
                       ("damage_matrices_outgoing_all", False),
                       ("pubval_mode", "none")):
        if _dead in data:
            _rec = data.pop(_dead)
            _bad = (_rec != _ok if isinstance(_ok, str) else bool(_rec) is not _ok)
            if _bad:
                raise ModelVersionError(
                    f"{_dead}={_rec!r} is no longer supported (gen3_dead_flag_purge_v1): the "
                    f"only supported value is {_ok!r}. This checkpoint trained under a forward "
                    "that no longer exists; re-read it from the git_hash in its metadata.json.")
    data.pop("pubval_coef", None)   # training-only (INERT for a forward) ⇒ any value pops silently
    # v96 (gen3_critic_route_wave_v1) — the critic-route deletion wave. Three fields LEFT the
    # config, so they must be POPped or a later `cls(**data)` TypeErrors on the stale key. The
    # parallel judgment on the PICKLED side is `snapshot._DEAD_FEK_JUDGED`, which has no floor to
    # hide behind; this must not be the only place the decision is written down.
    for _dead in ("intent_value_reduce", "value_clock", "value_intent"):
        if _dead in data:
            if data.pop(_dead):
                raise ModelVersionError(
                    f"{_dead}=True is no longer supported (gen3_critic_route_wave_v1, config "
                    "v96): the critic route behind it is DELETED, so this checkpoint's weights "
                    "include a projection the current extractor has no home for.\n"
                    "Every route in the wave was audited dead, or below the 0.39 dV bar twice, "
                    "on the end-of-run critic_route_audit; `--value-entity-pool` — which carries "
                    "97% of the critic's route dependence — is the successor.\n"
                    "To re-read this checkpoint, use the git_hash in its own metadata.json.")
    # ---- POST-FLOOR MIGRATION BRANCHES (N > MIGRATION_FLOOR) --------------------------------
    # v97 (gen3_intent_label_bot_weight_v1) — a TRAINING-only loss weight, so a pre-v97 checkpoint
    # trained with it OFF and the field simply defaults in. No forward, no weight shape, no gate:
    # provenance + flagless-resume read-back only. Same shape as v92's since-floored td_aux_coef
    # branch, and REACHABLE where that one no longer is: a v96 checkpoint is exactly at the floor
    # and its config genuinely lacks the field.
    if version < 97:
        data.setdefault("intent_label_bot_weight", 1.0)
        data["config_version"] = 97
    # v98 (gen3_cf_evidential_head_v1) — a STRUCTURAL toggle, so unlike v97 this one is gated. It
    # still DEFAULTS rather than refusing: a pre-v98 checkpoint could not have built the head (the
    # module did not exist), so False is not a guess, it is the only possible past. The refusal
    # direction is handled by check_compatible, which fires the moment a live run's True meets a
    # migrated False.
    if version < 98:
        data.setdefault("cf_evidential", False)
        data["config_version"] = 98
    # v99 (gen3_cf_twin_heads_v1) — two STRUCTURAL toggles, both gated like v98's. Same reasoning
    # for defaulting rather than refusing: a pre-v99 checkpoint could not have built either module,
    # so False is the only possible past, and the refusal direction is check_compatible's the moment
    # a live run's True meets a migrated False.
    if version < 99:
        data.setdefault("cf_twin_heads", False)
        data.setdefault("cf_shadow_critic", False)
        data["config_version"] = 99
    # v100 (gen3_cf_coef_provenance_v1) — ten TRAINING-only coefficients ⇒ v97's shape, not
    # v98/v99's: no gate, no refusal, just a default (the ARGPARSE one, which is what every
    # pre-v100 run necessarily ran with). Any recorded value migrates untouched.
    if version < 100:
        for _k, _v in (("cf_records", False), ("cf_records_keep", 512),
                       ("cf_winprob_coef", 0.0), ("cf_head_only", True),
                       ("cf_label_lag_steps", 150_000), ("cf_label_likelihood", "binomial"),
                       ("cf_evidential_coef", 0.0), ("cf_evidential_reg", 1e-3),
                       ("cf_twin_coef", 0.0), ("cf_shadow_coef", 0.0)):
            data.setdefault(_k, _v)
        data["config_version"] = 100
    # v101 (gen3_capacity_telemetry_v1) — four TRAINING-only DIAGNOSTIC knobs ⇒ v100's shape: no
    # gate, no refusal, just a default (the ARGPARSE one, which is what every pre-v101 run
    # necessarily ran with — the telemetry did not exist). Any recorded value migrates untouched.
    if version < 101:
        for _k, _v in (("capacity_telemetry", False), ("canary_reset_steps", 1_000_000),
                       ("capacity_cosine_every", 50), ("capacity_velocity_every", 50)):
            data.setdefault(_k, _v)
        data["config_version"] = 101
    return data
