"""Phase 1 — CONFIG RESOLUTION: turn a parsed argv into a coherent, validated run configuration.

Three jobs, in order, and the order is load-bearing:

* **DESUGAR** the umbrella flags (`--unified-moves` -> `--unified-damage` -> the component
  toggles, `--damage-matrices` -> its two bools) BEFORE `_resolve`, so they are not None-filled
  from a saved checkpoint.
* **`_resolve`** every version-checked structural toggle: `None` (not passed) INHERITS the saved
  value, so the documented flagless resume (`--model X --steps N`) keeps the architecture it was
  trained with instead of falling back to OFF and FATAL-ing at `check_compatible`.
* **VALIDATE** — for a training-only coefficient, the `parser.error` here is the ONLY gate there
  is (nothing version-checks it), which is why the checks are exhaustive rather than a sample.

`args` is MUTATED in place; the handful of values that are not attributes of `args` come back in
`ResolvedRunConfig`.
"""
import dataclasses
import sys
from typing import Any

from agents.model.damage_tables import _MIN_PRIOR_FLOOR, _PRIOR_FLOOR
from agents.training.watchdog import start_orphan_watchdog
from main.launcher.ipc import emit
from main.train.checkpoint_state import _load_saved_version
from main.exit_codes import TrainExitCode
from main.train.compile_flags import (
    resolve_compile_opponents_preload, resolve_compile_trainer_auto,
)
from main.train.constants import (
    CF_DUTY_CYCLE_FLOOR, DEFAULT_DISTILL_TEAM_BIAS, cf_label_duty_cycle,
    checkpoint_interval_env_steps, checkpoint_save_freq_vec_calls,
)
from poke_env import LocalhostServerConfiguration
from poke_env.ps_client.server_configuration import localhost_server_configuration
from utils.logging.levels import LogLevel


@dataclasses.dataclass(frozen=True)
class ResolvedRunConfig:
    """The three resolved values that are NOT attributes of `args`."""

    server_config: Any
    annealing_mode: bool
    log_level: LogLevel


def _announce_cf_duty_cycle(args) -> None:
    """PRINT the counterfactual label DUTY CYCLE, and REFUSE a starved one.

    THE DEFECT THIS MAKES UNREPRESENTABLE (`ai_v9_29_rev1_0823`, 2026-08-23). The label producer
    can only stamp labels with the step of the newest `checkpoints/` zip, and `cf_label_buffer`
    expires a row more than `--cf-label-lag-steps` behind the live policy. So the two flags define
    a fraction — and NOBODY WAS COMPUTING IT. At the hardcoded 50 000 VEC-CALL cadence and
    `--n-envs 48` the checkpoint interval is 2 400 000 env steps against a 150 000-step bound: a
    6.25% duty cycle, observed as **6 labels ingested against 255 expired in two hours**, with
    every counter on both sides reading healthy (the producer was producing; the buffer was
    expiring; neither knew about the other's number).

    So the number is now PRINTED on every launch that has both halves on, healthy or not — a
    quantity nobody computes is how this shipped — and refused below the floor. A refusal exits
    `FATAL_CONFIG` rather than `parser.error`, because restarting would hit the identical config
    every time and the launcher must give up rather than loop.

    `--debug` is exempt: a smoke has one env and runs for thousands of steps, so its duty cycle is
    an artifact of the smoke rather than a statement about the recipe.
    """
    on = bool((args.cf_twin_coef and args.cf_twin_coef > 0)
              or (args.cf_winprob_coef and args.cf_winprob_coef > 0))
    if not (on and args.cf_records):
        return
    n_envs = 1 if args.debug else int(args.n_envs)
    every = getattr(args, "checkpoint_every_steps", None)
    vec_calls = checkpoint_save_freq_vec_calls(every, n_envs)
    interval = checkpoint_interval_env_steps(every, n_envs)
    duty = cf_label_duty_cycle(args.cf_label_lag_steps, interval)
    shown = "unbounded (--cf-label-lag-steps 0 = labels never expire)" if duty == float("inf") \
        else f"{duty:.1%}"
    line = (f"🧾 [CF] label DUTY CYCLE {shown} — --cf-label-lag-steps "
            f"{args.cf_label_lag_steps:,} / {interval:,} env-steps between checkpoints "
            f"({vec_calls:,} vec-calls x {n_envs} envs)")
    if args.debug:
        emit(line + "  [--debug: the floor is not enforced]")
        return
    if duty >= CF_DUTY_CYCLE_FLOOR:
        emit(line)
        return
    print(
        f"\n[CF] FATAL: the counterfactual label path is STARVED BY CONSTRUCTION.\n"
        f"  --cf-label-lag-steps         : {args.cf_label_lag_steps:,} env steps\n"
        f"  checkpoint interval          : {interval:,} env steps "
        f"({vec_calls:,} vec-calls x {n_envs} envs)\n"
        f"  --checkpoint-every-steps     : "
        f"{'(unset — the 50000-vec-call default)' if every is None else format(every, ',')}\n"
        f"  => DUTY CYCLE                : {shown}  (floor {CF_DUTY_CYCLE_FLOOR:.0%})\n"
        f"  The producer stamps every label with the newest checkpoint's step, so outside that\n"
        f"  window EVERY label it writes is expired by the buffer on arrival. Two remedies, and\n"
        f"  either alone is enough:\n"
        # Both remedies are printed WITHOUT thousands separators: they are copy-pasteable argv
        # values, and `--checkpoint-every-steps 600,000` is an argparse error.
        f"    * checkpoint MORE OFTEN: --checkpoint-every-steps "
        f"{max(1, int(args.cf_label_lag_steps / CF_DUTY_CYCLE_FLOOR))} or less\n"
        f"    * widen the staleness bound: --cf-label-lag-steps "
        f"{max(1, int(CF_DUTY_CYCLE_FLOOR * interval))} or more (a label then supervises a\n"
        f"      policy further from the one that produced it — the cost this bound exists to cap)\n",
        file=sys.stderr, flush=True)
    sys.exit(int(TrainExitCode.FATAL_CONFIG))


def resolve_config(args, parser) -> ResolvedRunConfig:
    """Desugar, inherit and validate `args` in place. Returns the values that live outside it."""
    # --- Resolve `--use-bridge` into the two internal fields ------------------------------------
    # ONE knob now: `--use-bridge {off,node,rust}`, defaulting to `rust` (serverless training AND
    # eval). It splits into `args.use_showdown_bridge` (a plain bool = "bridge enabled?", read at
    # every transport site) + `args.bridge_impl` (the "node"|"rust" child selector, read only at
    # spawn). `off` keeps a bridge_impl of "node" so a websocket run still has a well-formed value
    # for the offline/search paths that take one.
    #
    # The DEPRECATED `--use-showdown-bridge` boolean alias is DELETED. It meant `--use-bridge=node`,
    # which is no longer the default, so keeping it would have made "the legacy flag" silently mean
    # "the slower impl" — pass `--use-bridge=node` explicitly for that.
    _use_bridge = getattr(args, "use_bridge", "rust")
    args.bridge_impl = "node" if _use_bridge == "off" else _use_bridge
    args.use_showdown_bridge = _use_bridge != "off"

    # --- Resolve resumable structural toggles (None sentinel = "not passed on the CLI") ---
    # Each version-checked structural toggle defaults to None so a FLAGLESS resume can INHERIT the
    # saved value (the documented `--model … --steps …` command), instead of falling back to OFF and
    # FATALing at check_compatible (saved-ON vs current-default-OFF). An EXPLICIT flag that flips a
    # toggle still FATALs at load (desirable). A fresh run (no --model) → the toggle's OFF default.
    _saved_ver = _load_saved_version(args.model) if args.model else None
    if args.model and _saved_ver is None:
        print("[Resume] WARNING: saved model_config.json unreadable — structural toggles fall back to "
              "their OFF defaults and may FATAL at the version check; pass them explicitly if needed.")
    _popart_explicit = args.use_popart is not None
    _coef_explicit = args.opp_belief_aux_coef is not None
    _hp_coef_explicit = args.hp_type_belief_coef is not None   # before _resolve fills the 0.05 default

    # --unified-moves is the umbrella over the WHOLE move system: it sets --unified-damage to the same
    # level (so the op/belief/outgoing desugar below runs) AND turns on the move latent + its grading.
    # Applied BEFORE the --unified-damage desugar so the level flows through. v24.
    #
    # DEFAULT-ON (2026-08-04, owner decision): the unified move system is the model — every production
    # config since v24 runs it, and the off path is an ablation baseline, not a supported configuration.
    # A None (flagless) invocation resolves to:
    #   * FRESH run → 'both' (the full system), with a printed note;
    #   * RESUME (--model) → NO desugar — the component toggles stay None and _resolve below inherits the
    #     checkpoint's saved arch verbatim (the same flagless-resume contract every structural toggle
    #     follows), so a resume can never be version-FATALed by a default. A launcher restart that
    #     forwarded the original explicit flag is likewise unchanged.
    # An EXPLICIT 'off' still works (fresh ablation baselines need it) but is DEPRECATED and warns.
    if args.unified_moves is None:
        if args.model:
            args.unified_moves = "off"     # no desugar — inherit the saved component toggles via _resolve
        else:
            args.unified_moves = "both"
            print("[Arch] --unified-moves defaults to 'both' (the unified move system is the model; "
                  "pass --unified-moves off explicitly for the DEPRECATED ablation baseline).")
    elif args.unified_moves == "off":
        print("[Arch] DEPRECATED: --unified-moves off — the non-unified path is an ablation baseline "
              "only (no move belief, no damage op, no discrete move-space). It keeps working, but new "
              "features target the unified system.")
    if getattr(args, "unified_moves", "off") != "off":
        if getattr(args, "unified_damage", "off") == "off":
            args.unified_damage = args.unified_moves
        if args.move_latent is None:
            args.move_latent = True
        if args.move_belief_latent_coef is None:
            args.move_belief_latent_coef = 0.05
        # gen3_unified_topk_incoming_v1: the umbrella also turns on the DISCRETE top-K incoming block at the
        # default K (the deps — damage_op + move_latent — are satisfied above/below). An explicit
        # --damage-topk wins (incl. --damage-topk 0 to A/B it off under --unified-moves).
        if args.damage_topk_k is None:
            from agents.model.features_extractor import _DMG_TOPK_DEFAULT_K
            args.damage_topk_k = _DMG_TOPK_DEFAULT_K


    # --unified-damage desugars into the component flags BEFORE _resolve (so they aren't None-filled from a
    # saved version). When not 'off' it forces damage_op + prior fusion + (for 'both') the outgoing block,
    # and defaults the move-belief mode to 'revealed' unless the user set it explicitly (so
    # `--unified-damage both --move-belief-mode both` still guesses unrevealed mons' moves).
    if getattr(args, "unified_damage", "off") != "off":
        if args.move_belief_mode is None:
            args.move_belief_mode = "revealed"
        args.damage_op = True
        args.move_prior_fusion = True
        args.damage_outgoing = (args.unified_damage == "both")

    # Explicit CLARITY knob: "predict the moves of mons we haven't even SEEN". OFF
    # (--no-predict-unrevealed-mon-moves) zeros BOTH hidden-mon move-prediction paths — the
    # hidden-opponent BeliefHead's moves-BCE (`opp_belief_moves_weight` → 0) AND any MoveBelief
    # unrevealed leg (`move_belief_mode` 'unrevealed'/'both' → 'revealed'). The REVEALED-mon move belief
    # (predict a SEEN mon's unseen slots) and the SPECIES belief on hidden mons are UNTOUCHED. A desugar
    # into existing fields (no new version field); unset/True preserves the current behavior.
    if getattr(args, "predict_unrevealed_mon_moves", None) is False:
        args.opp_belief_moves_weight = 0.0
        if args.move_belief_mode in ("unrevealed", "both"):
            args.move_belief_mode = "revealed"

    # gen3_per_move_matrices_v1: --damage-matrices desugars to the two bool toggles BEFORE _resolve (so a
    # resume inherits them). None ⇒ let _resolve inherit/default; an explicit value wins. The INCOMING matrix
    # is the ENRICHED top-K — it REUSES --damage-topk K as its K (the one "how many opp moves" knob) and
    # REPLACES the lean top-K block at that K. Default the K to _DMG_TOPK_DEFAULT_K if unset (so it works
    # standalone); an explicit --damage-topk (or --unified-moves' default) wins.
    if getattr(args, "damage_matrices", None) is not None:
        args.damage_matrices_outgoing = args.damage_matrices in ("outgoing", "both")
        args.damage_matrices_incoming = args.damage_matrices in ("incoming", "both")
        if args.damage_matrices_incoming and not args.damage_topk_k:
            from agents.model.features_extractor import _DMG_TOPK_DEFAULT_K   # local: needed without --unified-moves
            args.damage_topk_k = _DMG_TOPK_DEFAULT_K     # the matrix's K = --damage-topk (default 5)
    else:
        if not hasattr(args, "damage_matrices_outgoing"):
            args.damage_matrices_outgoing = None
        if not hasattr(args, "damage_matrices_incoming"):
            args.damage_matrices_incoming = None

    def _resolve(name, default):
        if getattr(args, name) is None:
            setattr(args, name, getattr(_saved_ver, name, default) if _saved_ver is not None else default)
    _resolve("use_popart", False)
    _resolve("opp_belief_cls_k", 0)
    _resolve("opp_belief_aux_coef", 0.0)
    _resolve("move_belief_mode", "off")        # v17 structural (version-checked, fresh-only)
    _resolve("move_belief_coef", 0.0)          # training-only (inherited like opp_belief_aux_coef)
    _resolve("damage_op", False)               # v19 structural (version-checked, fresh-only)
    _resolve("damage_outgoing", False)         # v23 structural (version-checked, fresh-only)
    _resolve("move_candidate_floor", _PRIOR_FLOOR)  # v65 forward-behavior (version-checked, fresh-only)
    _resolve("move_latent", False)             # v24 structural (version-checked, fresh-only)
    _resolve("move_belief_latent_coef", 0.0)   # training-only (inherited like move_belief_coef)
    _resolve("spread_belief", False)           # v25 structural (version-checked, fresh-only)
    _resolve("spread_belief_nature", False)    # v40 structural (version-checked, fresh-only)
    _resolve("spread_belief_coef", 0.0)        # training-only (inherited like move_belief_coef)
    _resolve("move_prior_fusion", False)       # v20 forward-behavior (version-checked, fresh-only)
    _resolve("damage_candidate_k", 0)          # v49 forward-behavior (version-checked, fresh-only)
    _resolve("entity_topk_seats", 0)           # v54 structural int (version-checked, fresh-only)
    _resolve("consequence_topk", 6)            # v59 forward-behavior int (version-checked)
    _resolve("edge_bias_families", "off")      # v56 structural str (version-checked, fresh-only)
    _resolve("entity_tail_seats", False)       # v57 structural bool (version-checked, fresh-only)
    _resolve("win_prob_mode", "none")          # v22 structural + resume-immutable (version-checked)
    _resolve("win_prob_coef", 1.0)             # training-only (inherited like opp_belief_aux_coef)
    _resolve("value_dist_mode", "none")        # v29 structural + resume-immutable (version-checked)
    _resolve("value_dist_bins", 0)             # v29 structural (atom count; version-checked)
    _resolve("value_dist_vmin", 0.0)           # v29 resume-immutable support (version-checked)
    _resolve("value_dist_vmax", 0.0)           # v29 resume-immutable support (version-checked)
    _resolve("value_dist_coef", 1.0)           # training-only (inherited like win_prob_coef)
    _resolve("td_aux_coef", 0.0)               # v90 training-only (inherited like win_prob_coef)
    _resolve("win_prob_pbrs_coef", 0.0)        # v104 training-only (inherited like td_aux_coef)
    _resolve("policy_grad_coef", 1.0)                   # v102 training-only (inherited like td_aux_coef; 1.0 = upstream)
    _resolve("value_threat_inject", False)     # v64 structural bool (version-checked, fresh-only)
    _resolve("opp_intent_coef", 0.0)           # v67 training-only coef; the HEADS are structural
    _resolve("beta_setvalued_coef", 0.0)       # training-only coef; no module, no version gate
    _resolve("intent_label_bot_weight", 1.0)   # v97 training-only (inherited like win_prob_coef)
    # (`opp_intent_grad_mode` had a `_resolve` here until 2026-08-23. It is config_only now —
    #  no argparse dest to inherit FROM, so a resolve line would be dead. Frozen "detached".)
    _resolve("intent_move_cell", False)        # v77 structural, version-checked (G3)
    _resolve("value_entity_pool", False)       # v80 structural, version-checked (Stage-3 T3)
    _resolve("history_events", False)          # v81 structural, version-checked (Tier H-B)
    _resolve("value_entity_pool_full", False)  # v82 structural, version-checked (full row set)
    _resolve("item_belief", False)             # v83 structural, version-checked (gen3_item_belief_v1)
    _resolve("pair_outcome_cell", False)   # v93 structural, version-checked (gen3_pair_outcome_v1)
    _resolve("pair_outcome_switch", False)  # v94 structural, version-checked (gen3_pair_outcome_switch_v1)
    _resolve("switch_branch_cell", False)   # v94 structural, version-checked (gen3_switch_branch_v1)
    _resolve("conditional_threat_cell", False)  # v95 structural, version-checked (gen3_conditional_threat_v1)
    _resolve("pair_value_route", False)     # v95 structural, version-checked (gen3_pair_value_route_v1)
    _resolve("intent_threshold", False)        # v84 structural, version-checked (gen3_intent_threshold_v1)
    _resolve("intent_conditional", False)      # v85 structural, version-checked (gen3_intent_conditional_v1)
    _resolve("op_drop_renders", False)         # v86 structural, version-checked (gen3_op_lean_forward_v1)
    _resolve("op_believed_lean", False)        # v86 structural, version-checked (gen3_op_lean_forward_v1)
    # THE COUNTERFACTUAL FAMILY — structural heads AND their coefficients, all inherited.
    # The three STRUCTURAL bools are version-checked; the ten coefficients below are the
    # td_aux_coef class (config v100, gen3_cf_coef_provenance_v1): recorded for provenance,
    # never gated, and read back here so a flagless resume keeps the arm it was launched as.
    # ⚠️ Every one of these argparse entries MUST default to None or the `_resolve` line is a
    # no-op — that is exactly how these three sat here reading False for two versions.
    # `flag_registry_test.test_cli_flags_argparse_default_is_none` is now that gate.
    _resolve("cf_evidential", False)           # v98 structural, version-checked (gen3_cf_evidential_head_v1)
    _resolve("cf_twin_heads", False)           # v99 structural, version-checked (gen3_cf_twin_heads_v1)
    _resolve("cf_shadow_critic", False)        # v99 structural, version-checked (gen3_cf_twin_heads_v1)
    _resolve("cf_records", False)              # v100 training-only (inherited like td_aux_coef)
    _resolve("cf_records_keep", 512)           # v100 training-only
    _resolve("cf_winprob_coef", 0.0)           # v100 training-only
    _resolve("cf_head_only", True)             # v100 training-only
    _resolve("cf_label_lag_steps", 150_000)    # v100 training-only
    _resolve("cf_label_likelihood", "binomial")  # v100 training-only
    _resolve("cf_evidential_coef", 0.0)        # v100 training-only
    _resolve("cf_evidential_reg", 1e-3)        # v100 training-only
    _resolve("cf_twin_coef", 0.0)              # v100 training-only
    _resolve("cf_shadow_coef", 0.0)            # v100 training-only
    # gen3_capacity_telemetry_v1 — the live saturation early-warnings. The td_aux_coef class:
    # recorded for provenance, never gated, and read back here so a flagless resume (or a
    # hand-typed one between launcher restarts) keeps logging the run's own `capacity/*` series.
    _resolve("capacity_telemetry", False)      # v101 training-only diagnostic (no loss, no grad)
    _resolve("canary_reset_steps", 1_000_000)  # v101 training-only
    _resolve("capacity_cosine_every", 50)      # v101 training-only
    _resolve("capacity_velocity_every", 50)    # v101 training-only
    _resolve("species_prior_fusion", False)    # v68 structural bool (version-checked, fresh-only)
    _resolve("t0_species_prior", False)        # v72 structural bool (version-checked, fresh-only)
    _resolve("search_teacher_coef", 0.0)       # training-only AWR weight (inherited on flagless resume)
    _resolve("search_teacher_value_coef", 0.0)  # training-only off-policy value term (default OFF)
    _resolve("search_teacher_beta", 1.0)       # training-only AWR temperature
    _resolve("search_teacher_batch_size", 256)  # training-only per-train() correction sample
    _resolve("opd_coef", 0.0)                  # training-only OPD KL weight (inherited on flagless resume)
    _resolve("distill_coef", 0.0)              # training-only exploiter-distillation KL weight (inherited on resume)
    _resolve("distill_value_coef", 0.0)        # training-only exploiter VALUE-distillation MSE weight (inherited on resume)
    _resolve("distill_value_feat_coef", 0.0)   # training-only FitNets value-FEATURE distill cosine weight (inherited on resume)
    # gen3_distill_target_gate_v1 (config v103) — the action-form/top-K distill target, the
    # advantage gate, and the rank tripwire. The td_aux_coef class: recorded for provenance,
    # never gated, read back here so a flagless resume keeps the arm it was launched as.
    _resolve("distill_target", "kl")           # v103 training-only TARGET FORM ("kl" = byte-identical)
    _resolve("distill_topk", 1)                # v103 training-only top-K (1 = argmax CE)
    _resolve("distill_gate", "none")           # v103 training-only JUDGE (rung a)
    _resolve("distill_gate_tau", 0.0)          # v103 training-only gate threshold (normalized-adv units)
    _resolve("distill_beta", 1.0)              # v103 training-only AWR |adv| temperature
    _resolve("rank_tripwire", "warn")          # v103 training-only diagnostic (§4.1; no loss, no grad)
    _resolve("rank_tripwire_drop", 0.20)       # v103 training-only TRIP threshold (fractional drop)
    _resolve("opd_beta", 1.0)                  # training-only OPD softmax temperature β
    _resolve("damage_topk_k", 0)               # v30 structural int (top-K incoming; version-checked, fresh-only)
    _resolve("damage_matrices_outgoing", False)  # v32 structural (outgoing damage matrix; version-checked, fresh-only)
    _resolve("damage_matrices_incoming", False)  # v33 structural (incoming damage matrix; version-checked, fresh-only)
    # gen3_op_block_trim_v1: --damage-topk K now sizes the INCOMING MATRIX and nothing else — the v30 LEAN
    # top-K block it used to select is DELETED (a strict subset of the matrix, which already suppressed it
    # in every production config; the ledger-P1 cProfile measured it at 0 calls/forward). So K>0 implies the
    # matrix. When the user gave no explicit --damage-matrices (the --unified-moves path, which auto-sets
    # K=5) turn the incoming matrix ON rather than let K>0 mean "emit nothing"; an EXPLICIT
    # --damage-matrices off/outgoing next to K>0 is a contradiction and errors below.
    if args.damage_topk_k and args.damage_topk_k > 0 and not args.damage_matrices_incoming:
        if getattr(args, "damage_matrices", None) is None:
            args.damage_matrices_incoming = True
            print("[Arch] --damage-topk implies the INCOMING per-move damage matrix (gen3_op_block_trim_v1: "
                  f"the lean top-K block was deleted) — enabling it at K={args.damage_topk_k}.")
    _resolve("threat_prob_outspeed", False)      # v36 forward-behavior (prob outspeed; version-checked, fresh-only)
    _resolve("belief_grad_mode", "shaping")    # v41 resume-immutable training hparam (vf_coef class; flagless resume inherits)
    _resolve("value_from_dist", False)         # v45 Phase B: dist head is the critic (resume-immutable; flagless resume inherits)
    _resolve("hp_belief_mode", "composed")     # v53 STRUCTURAL (version-checked, fresh-only)
    _resolve("hp_type_belief_coef", 0.05)      # training-only (inherited like spread_belief_coef)
    _resolve("item_belief_coef", 0.05)         # training-only (inherited like hp_type_belief_coef)
    # Phase B (v45): the dist head can only BE the critic if it's a live, trunk-shaping head.
    if args.value_from_dist and args.value_dist_mode != "shaping":
        parser.error("--value-from-dist requires --value-dist-mode shaping (the distributional head must "
                     "be a live critic that shapes the trunk; got value_dist_mode="
                     f"{args.value_dist_mode!r}).")
    # PopArt INHERITED on a flagless resume → adopt its required `--clip-range-vf none` (the saved
    # popart run necessarily used it), so the explicit-config check below doesn't block the resume.
    if args.use_popart and not _popart_explicit and _saved_ver is not None and args.clip_range_vf is not None:
        args.clip_range_vf = None
    # Friendly belief-resume notes (inheriting vs an explicit flip).
    if args.model and _saved_ver is not None:
        _sc = getattr(_saved_ver, "opp_belief_aux_coef", 0.0) or 0.0
        if not _coef_explicit and _sc > 0.0:
            print(f"[Belief] resume: inheriting saved --opp-belief-aux-coef {_sc:g} (pass it explicitly to override).")
        elif _coef_explicit and (_sc > 0.0) != (args.opp_belief_aux_coef > 0.0):
            print(f"[Belief] WARNING: --opp-belief-aux-coef {args.opp_belief_aux_coef:g} flips the belief head "
                  f"vs the saved checkpoint (coef {_sc:g}); a weight-shape change → will FATAL on load.")

    if args.use_popart and args.clip_range_vf is not None:
        # Require value clipping to be EXPLICITLY off with PopArt — a self-documenting config beats a
        # silent override. PopArt normalizes the value targets so clipping is unnecessary; and because
        # the value head returns de-normalized values an active clip would clip in UN-normalized units.
        parser.error(
            "--use-popart requires an explicit '--clip-range-vf none' (it defaults to 0.5). PopArt "
            "normalizes the value targets so value clipping is unnecessary — and an active clip "
            "would clip in un-normalized units and cripple the critic. Pass --clip-range-vf none."
        )
    if not 0.0 <= args.stable_opponent_selfplay_share <= 1.0:
        parser.error("--stable-opponent-selfplay-share must be a fraction in [0, 1]")
    if args.exploiter and args.self_play:
        parser.error("--exploiter trains vs ONE fixed target as the sole opponent — it is mutually "
                     "exclusive with --self-play. Drop --self-play (the exploiter needs no pool).")
    if args.exploiter_keep_bots and not args.exploiter:
        parser.error("--exploiter-keep-bots only applies in exploiter mode — pass --exploiter <target> "
                     "too (it mixes the bots in ALONGSIDE that target).")
    if args.warmstart_consensus and not args.exploiter:
        parser.error("--warmstart-consensus builds an EXPLOITER init (a disagreement-gated consensus of "
                     "teacher exploiters, sharp-on-agree / flat-on-disagree) and only applies in exploiter "
                     "mode — pass --exploiter <target>. It is deliberately NOT available for "
                     "generalist / self-play training, whose objective is to ABSORB per-team divergence "
                     "(--distill-teacher), the OPPOSITE of distilling the consensus.")
    if not 0.0 <= args.exploiter_bot_fraction <= 1.0:
        parser.error("--exploiter-bot-fraction must be a fraction in [0, 1]")
    if args.exploiter_temp_start is not None:
        if not args.exploiter:
            parser.error("--exploiter-temp-start only applies in exploiter mode — pass --exploiter "
                         "<target> too (it anneals THAT target's play temperature).")
        if args.exploiter_temp_start <= 0.0 or args.exploiter_temp_end <= 0.0:
            parser.error("--exploiter-temp-start / --exploiter-temp-end must be > 0 (a softmax "
                         "temperature; the opponent's logits are divided by it).")
        if not 0.0 <= args.exploiter_temp_anneal_frac <= 1.0:
            parser.error("--exploiter-temp-anneal-frac must be a fraction in [0, 1]")
        if args.exploiter_temp_mode == "ratchet":
            if not 0.0 < args.exploiter_temp_ratchet_factor < 1.0:
                parser.error("--exploiter-temp-ratchet-factor must be in (0, 1) (it multiplies the "
                             "temperature DOWN each ratchet).")
            if not 0.0 < args.exploiter_temp_ratchet_wr < 1.0:
                parser.error("--exploiter-temp-ratchet-wr must be a win-rate in (0, 1).")
            if args.exploiter_temp_ratchet_games < 1:
                parser.error("--exploiter-temp-ratchet-games must be >= 1.")
            if args.exploiter_temp_start <= args.exploiter_temp_end:
                parser.error("--exploiter-temp-mode ratchet needs --exploiter-temp-start > "
                             "--exploiter-temp-end (it ratchets the temp DOWN from start toward end).")
    # gen3_exploiter_pool_ladder_v1 — the POOL-LADDER opponent curriculum. Same shape of dependency
    # as --exploiter-keep-bots above: it only means anything in exploiter mode, and it names rungs
    # that end at the --exploiter target, so a ladder with no target is a config with no terminus.
    if args.exploiter_ladder:
        if not args.exploiter:
            parser.error("--exploiter-ladder only applies in exploiter mode — pass --exploiter "
                         "<target> too (the ladder's TERMINAL rung IS that target; without it the "
                         "curriculum has no destination).")
        if not 0.0 < args.exploiter_ladder_gate < 1.0:
            parser.error("--exploiter-ladder-gate must be a win-rate in (0, 1).")
        if args.exploiter_ladder_window < 1:
            parser.error("--exploiter-ladder-window must be >= 1.")
        if args.exploiter_ladder_rungs < 1:
            parser.error("--exploiter-ladder-rungs must be >= 1 (the number of auto: rungs drawn "
                         "BEFORE the --exploiter target is appended).")
    elif args.exploiter_ladder_rungs < 1:
        parser.error("--exploiter-ladder-rungs must be >= 1.")
    if args.exploiter_temp_start is None and args.exploiter_temp_mode == "ratchet":
        parser.error("--exploiter-temp-mode ratchet requires --exploiter-temp-start (the initial/max "
                     "temperature to ratchet down from — set it HIGH, e.g. 5.0).")
    if args.opp_belief_cls_k < 0:
        parser.error("--opp-belief-cls-k must be >= 0 (0 = off)")
    if args.opp_belief_aux_coef < 0.0:
        parser.error("--opp-belief-aux-coef must be >= 0 (0 = off)")
    if args.move_belief_coef is not None and args.move_belief_coef < 0.0:
        parser.error("--move-belief-coef must be >= 0 (0 = off)")
    if args.win_prob_coef is not None and args.win_prob_coef < 0.0:
        # A negative coef would INVERT the BCE gradient (train the head/trunk to MAXIMISE error).
        # win_prob_coef is training-only (not version-locked), so guard it here — the only gate.
        parser.error("--win-prob-coef must be >= 0 (0 = off; the mode controls on/off)")
    if args.value_dist_mode != "none":
        # The atom count is the head's output width; the support must be a real interval. Self-documenting
        # config: require both explicitly when the head is on (no magic defaults for a versioned param).
        if not args.value_dist_bins or args.value_dist_bins <= 0:
            parser.error("--value-dist-mode requires --value-dist-bins > 0 (the atom count; recommended 32)")
        if not (args.value_dist_vmax > args.value_dist_vmin):
            parser.error("--value-dist-mode requires --value-dist-vmax > --value-dist-vmin (the atom support)")
    elif args.value_dist_bins:
        parser.error("--value-dist-bins is set but --value-dist-mode is none — pass a mode, or drop the bins")
    if args.value_dist_coef is not None and args.value_dist_coef < 0.0:
        # A negative coef would INVERT the CE gradient. value_dist_coef is training-only (not
        # version-locked), so guard it here — the only gate.
        parser.error("--value-dist-coef must be >= 0 (0 = off; the mode controls on/off)")
    if args.td_aux_coef is not None and args.td_aux_coef < 0.0:
        # A negative coef would INVERT the consistency gradient (train the critic to MAXIMISE its own
        # Bellman residual). td_aux_coef is training-only (not version-locked), so guard it here.
        parser.error("--td-aux-coef must be >= 0 (0 = off)")
    if args.win_prob_pbrs_coef is not None and args.win_prob_pbrs_coef < 0.0:
        # A negative coef INVERTS the potential — the policy would be rewarded for driving its own
        # P(win) DOWN. The invariance theorem still holds (φ' = −φ is a valid potential), which is
        # exactly why this cannot be caught downstream: it would train, converge, and be wrong.
        parser.error("--win-prob-pbrs-coef must be >= 0 (0 = off)")
    if args.win_prob_pbrs_coef and args.win_prob_pbrs_coef > 0 and args.win_prob_mode == "none":
        # The POTENTIAL IS the win-prob head. Under `none` the head is not BUILT, so there is nothing
        # to read and the shaping would be a silent no-op — the invisible-regression class. Fail at
        # config time with the fix in the message, not at the first rollout end.
        parser.error("--win-prob-pbrs-coef > 0 requires --win-prob-mode read_only|shaping — the PBRS "
                     "potential φ(s) IS the win-prob head's output, and --win-prob-mode none builds "
                     "no head. Pass a mode, or drop the shaping coefficient.")
    if args.policy_grad_coef is not None and args.policy_grad_coef < 0.0:
        # A negative coef would ASCEND the PPO surrogate — train the policy to be maximally wrong.
        # 0.0 (arm F's pure-distill/aux phase) is the intended floor. policy_grad_coef is training-only
        # (not version-locked), so guard it here — the only gate.
        parser.error("--policy-grad-coef must be >= 0 (1 = upstream PPO; 0 = no policy-gradient term)")
    if args.intent_label_bot_weight is not None and args.intent_label_bot_weight < 0.0:
        # A negative weight would train alpha/beta to be MAXIMALLY wrong about bots — the opposite
        # of "train on them less". 0.0 (ignore bot rows entirely) is the intended floor.
        # Training-only (not version-locked), so this parser check is the only gate.
        parser.error("--intent-label-bot-weight must be >= 0 (0 = train on no bot rows; 1 = off)")
    if args.opd_coef is not None and args.opd_coef < 0.0:
        parser.error("--opd-coef must be >= 0 (0 = off)")
    if args.opd_coef and args.opd_coef > 0 and not args.search_teacher:
        # OPD distils the beam's π' from the SAME correction buffer the search-teacher fills (its workers
        # build π'), so it can't run standalone.
        parser.error("--opd-coef > 0 requires --search-teacher (OPD distils the search-teacher's "
                     "correction buffer; its workers build the π' targets)")
    # gen3_cf_label_plumbing_v1 — training-only, so these parser checks are the ONLY gate.
    if args.cf_winprob_coef is not None and args.cf_winprob_coef < 0.0:
        parser.error("--cf-winprob-coef must be >= 0 (0 = off)")
    if args.cf_winprob_coef and args.cf_winprob_coef > 0 and args.win_prob_mode == "none":
        # There is no head to supervise: `win_prob_mode none` means WinProbHead is not BUILT. Fail
        # at the CLI rather than let a live coefficient silently fold nothing for a whole run.
        parser.error("--cf-winprob-coef > 0 requires --win-prob-mode read_only|shaping — the "
                     "counterfactual labels supervise the WIN-PROB head, which 'none' does not build")
    if args.cf_label_lag_steps is not None and args.cf_label_lag_steps < 0:
        parser.error("--cf-label-lag-steps must be >= 0 (0 = never expire)")
    if args.cf_records_keep is not None and args.cf_records_keep < 1:
        parser.error("--cf-records-keep must be >= 1")
    # gen3_capacity_telemetry_v1 — training-only diagnostics, so these parser checks are the ONLY
    # gate. A reset interval of 0 would re-seed a target on EVERY minibatch, which is not a slower
    # canary but a different (and meaningless) instrument, so it is refused rather than clamped.
    if args.canary_reset_steps is not None and args.canary_reset_steps < 1:
        parser.error("--canary-reset-steps must be >= 1 (it is the ENV-step interval between "
                     "plasticity-canary resets; there is no 'off' value — use "
                     "--no-capacity-telemetry to turn the whole instrument off)")
    if args.capacity_cosine_every is not None and args.capacity_cosine_every < 0:
        parser.error("--capacity-cosine-every must be >= 0 (0 = skip the half-batch cosine)")
    if args.capacity_velocity_every is not None and args.capacity_velocity_every < 0:
        parser.error("--capacity-velocity-every must be >= 0 (0 = skip the feature-velocity probe)")
    # gen3_cf_evidential_head_v1 — the coefficients are training-only, so (as above) these parser
    # checks are the ONLY gate. `--cf-evidential` itself IS version-gated, hence not checked here.
    if args.cf_evidential_coef is not None and args.cf_evidential_coef < 0.0:
        parser.error("--cf-evidential-coef must be >= 0 (0 = off)")
    if args.cf_evidential_reg is not None and args.cf_evidential_reg < 0.0:
        parser.error("--cf-evidential-reg must be >= 0 (0 = no KL pull toward Beta(1,1))")
    if args.cf_evidential_coef and args.cf_evidential_coef > 0 and not args.cf_evidential:
        # There is no head to supervise, and unlike the win-prob case the head cannot be added to a
        # run later — it is a state_dict change. Refuse at the CLI rather than fold nothing for a
        # whole run and then FATAL the resume that tries to fix it.
        parser.error("--cf-evidential-coef > 0 requires --cf-evidential — the evidential term "
                     "supervises a head that flag BUILDS, and it is a structural (version-gated) "
                     "toggle that cannot be turned on mid-run")
    # gen3_cf_twin_heads_v1 — the coefficients are training-only (parser checks are the ONLY gate);
    # the two structural flags are version-gated, so only their CROSS-flag requirements land here.
    if args.cf_twin_coef is not None and args.cf_twin_coef < 0.0:
        parser.error("--cf-twin-coef must be >= 0 (0 = off)")
    if args.cf_shadow_coef is not None and args.cf_shadow_coef < 0.0:
        parser.error("--cf-shadow-coef must be >= 0 (0 = off)")
    if args.cf_twin_coef and args.cf_twin_coef > 0 and not args.cf_twin_heads:
        # Same reasoning as --cf-evidential-coef: the heads are a state_dict change, so they cannot
        # be added mid-run to rescue a live coefficient — the mistake would cost the whole run AND
        # FATAL the resume that tried to fix it.
        parser.error("--cf-twin-coef > 0 requires --cf-twin-heads — the twin heads are a "
                     "state_dict change (v99, version-gated) and cannot be added to a run that "
                     "did not start with them.")
    if args.cf_twin_heads and args.win_prob_mode == "none":
        # Heads B and C mirror head A's on-policy BCE, and head A is `win_head`. With
        # --win-prob-mode none there is no head A, so the factorial has no control arm: B and C
        # would carry only their cf folds and B−A would be undefined. Refuse at the CLI rather than
        # produce an arm whose primary comparison silently does not exist.
        parser.error("--cf-twin-heads requires --win-prob-mode read_only|shaping — the twins "
                     "mirror head A's on-policy BCE, and --win-prob-mode none builds no head A, so "
                     "the arm's control arm would not exist.")
    if args.cf_shadow_coef and args.cf_shadow_coef > 0 and not args.cf_shadow_critic:
        parser.error("--cf-shadow-coef > 0 requires --cf-shadow-critic — the shadow head is a "
                     "state_dict change (v99, version-gated) and cannot be added to a run that "
                     "did not start with it.")
    if args.cf_records and not args.use_showdown_bridge:
        # The record is a `__RECON__` frame off the bridge child's stdout; the websocket transport
        # never produces one, so the flag would be a silent no-op.
        parser.error("--cf-records requires the in-process bridge (--use-bridge node|rust) — the "
                     "reconstruction record is a bridge frame; a websocket run emits none")
    if getattr(args, "checkpoint_every_steps", None) is not None and args.checkpoint_every_steps < 1:
        parser.error("--checkpoint-every-steps must be >= 1 (it is an ENV-STEP interval; there is "
                     "no 'off' value — omit the flag for the historical 50000-vec-call cadence)")
    _announce_cf_duty_cycle(args)
    if args.distill_coef is not None and args.distill_coef < 0.0:
        parser.error("--distill-coef must be >= 0 (0 = off)")
    if args.distill_value_coef is not None and args.distill_value_coef < 0.0:
        parser.error("--distill-value-coef must be >= 0 (0 = off)")
    if args.distill_value_coef and args.distill_value_coef > 0 and not (args.distill_coef and args.distill_coef > 0):
        parser.error("--distill-value-coef > 0 requires --distill-coef > 0 — the value distillation is "
                     "coherent only because the policy KL drives π_student→π_teacher on those states, "
                     "making V_teacher the right target (V^π is policy-relative).")
    if args.distill_value_feat_coef is not None and args.distill_value_feat_coef < 0.0:
        parser.error("--distill-value-feat-coef must be >= 0 (0 = off)")
    if (args.distill_value_feat_coef and args.distill_value_feat_coef > 0
            and not (args.distill_coef and args.distill_coef > 0)):
        parser.error("--distill-value-feat-coef > 0 requires --distill-coef > 0 — the FitNets value-feature "
                     "match is coherent only because the policy KL drives π_student→π_teacher on those states, "
                     "making the teacher's value_pooled the right target (V^π is policy-relative).")
    # gen3_exploiter_distill_v1: parse --distill-teacher into (teacher_path, [team_files]) GROUPS once,
    # stored on args for the teambuilder + model-setup to reuse. Preferred form =
    # 'TEACHER:TEAM[,TEAM...][;TEACHER2:...]' — ';' separates TEACHERS, ',' separates that teacher's TEAMS,
    # so ONE multi-team teacher (a --trainee-teams z-cluster exploiter) binds to all its teams without being
    # repeated N times (which would cost N identical teacher forwards per batch). The legacy comma-separated
    # pair form ('T1:a.txt,T2:b.txt') still parses (a comma segment containing ':' starts a new teacher).
    #
    # `--distill-team-bias` carries a None argparse default so a TYPED value is distinguishable from
    # the unset flag; the guard below refuses a typed bias with no teacher, and could not exist if
    # every flagless run arrived carrying 0.4. Resolved here, before any reader.
    _team_bias_explicit = args.distill_team_bias is not None
    if args.distill_team_bias is None:
        args.distill_team_bias = DEFAULT_DISTILL_TEAM_BIAS
    #
    # THE COEFFICIENT GATES THE LOSS, NOT THE BOOKKEEPING (gen3_distill_bias_at_coef0_v1). The pairs
    # are parsed whenever --distill-teacher is given, at ANY coefficient — because `--distill-team-bias`
    # (the trainee's team distribution) reads them, and a CONTROL arm is precisely "the same teacher
    # teams, the same bias, no loss". Gating the parse on the coefficient made that arm silently
    # UNBIASED: run `ai_v9_58_R2CTRL_0827` recorded `--distill-teacher <5> --distill-coef 0
    # --distill-team-bias 0.4` and trained at an EFFECTIVE bias of 0.0, so the capstone's
    # "team-bias constancy" design was violated by the config layer, invisibly, in both metadata and
    # argv. What DOES stay coefficient-gated is everything that costs something or changes a tensor:
    # the teacher model LOADING (main.train.model_build), the loss fold (instrumented_ppo), and the
    # training-only `distill_mask` obs key (main.train.matchup_setup — emitting it at coef 0 would
    # change the observation SPACE of a run that folds no distill term).
    args._distill_pairs = []
    _items = [x.strip() for x in (args.distill_teacher or "").split(",") if x.strip()]
    if args.distill_coef and args.distill_coef > 0 and not _items:
        parser.error("--distill-coef > 0 requires --distill-teacher (as 'TEACHER:TEAM[,TEAM...]' groups)")
    if _team_bias_explicit and args.distill_team_bias > 0 and not _items:
        # The bias is a bias TOWARD THE TEACHER TEAMS; with no teacher there is no team to bias
        # toward, so the flag would be a silent no-op — the exact failure this whole block exists to
        # make impossible. (Not reachable from the unset flag: the argparse default is None and the
        # 0.4 resolution happens above, so only a TYPED value lands here.)
        parser.error("--distill-team-bias > 0 requires --distill-teacher — the bias points at the "
                     "TEACHER TEAMS ('TEACHER:TEAM[,TEAM...]' groups) and there is nothing to bias "
                     "toward without one; the flag would be a silent no-op. Drop it, pass "
                     "--distill-team-bias 0, or name the teacher(s).")
    if _items:
        if ":" not in _items[0]:
            # The bare-list + parallel --distill-teacher-team form is DELETED (no run ever passed it;
            # verified across every models/*/metadata.json 2026-08-16). One form, no misalignment.
            #
            # The check is on the FIRST segment, which is the one that distinguishes the two forms.
            # It used to be `all(":" in x for x in _items)`, which also rejected the DOCUMENTED
            # multi-team group 'T1:a.txt,b.txt' (a teacher's 2nd and later teams are comma segments
            # with no colon by construction) — unless another ';'-joined teacher happened to follow,
            # which is the only reason the multi-team form was ever seen to work. A later bare
            # segment with no preceding teacher is still refused, by `parse_distill_teacher_spec`.
            parser.error("--distill-teacher takes 'TEACHER:TEAM[,TEAM...]' colon groups — the bare "
                         "teacher list (with the deleted --distill-teacher-team) is no longer accepted")
        from agents.training.distill_spec import parse_distill_teacher_spec
        from agents.training.matchup_spec import read_recorded_trainee_teams
        try:
            # 'TEACHER:*' → EXACTLY the teams that teacher trained on, from its own recorded
            # provenance (single source of truth — a hand-typed list could mismatch and fire the
            # distill mask where the teacher is off-distribution, silently).
            args._distill_pairs = parse_distill_teacher_spec(
                args.distill_teacher, resolve_wildcard=read_recorded_trainee_teams)
        except (ValueError, FileNotFoundError) as _e:
            parser.error(str(_e))
    # gen3_distill_target_gate_v1 (design §7.5): the action-form family's dependency graph.
    # Checked on the RESOLVED values (after `_resolve`), so an incoherent combination is refused
    # whether it was typed on this launch or inherited from the checkpoint's recorded config.
    if args.distill_target == "action" and not (args.distill_coef and args.distill_coef > 0):
        parser.error("--distill-target action requires --distill-coef > 0 — the target form is a "
                     "property of the distill term; without the term there is nothing to shape")
    if args.distill_topk < 1:
        parser.error("--distill-topk must be >= 1 (1 = argmax CE; K >= n_actions recovers the KL)")
    if args.distill_topk != 1 and args.distill_target != "action":
        parser.error("--distill-topk requires --distill-target action — the top-K dial "
                     "parameterizes the action-form target; the 'kl' path has no K")
    if args.distill_gate != "none" and args.distill_target != "action":
        parser.error("--distill-gate requires --distill-target action (design §7.5: the gate "
                     "rides the action-form term)")
    if args.distill_gate_tau != 0.0 and args.distill_gate != "advantage":
        parser.error("--distill-gate-tau requires --distill-gate advantage — tau is the advantage "
                     "gate's threshold")
    if args.distill_beta <= 0.0:
        parser.error("--distill-beta must be > 0 (an AWR temperature)")
    if not (0.0 < args.rank_tripwire_drop < 1.0):
        parser.error("--rank-tripwire-drop must be in (0, 1) — a fractional drop from baseline")
    if args._distill_pairs and (args.trainee_team or args.trainee_teams):
        # Keyed on the PAIRS, not the coefficient (gen3_distill_bias_at_coef0_v1): the team bias now
        # applies at coef 0 too, and it REPLACES the trainee teambuilder — so at coef 0 a pin would
        # not merely be redundant, it would be silently DISCARDED. Refuse instead.
        parser.error("--distill-teacher is mutually exclusive with --trainee-team/--trainee-teams: "
                     "distillation biases the trainee toward the teacher teams via --distill-team-bias "
                     "while keeping the pool for rehearsal; a hard pin would remove the rehearsal (and "
                     "cause forgetting), and the bias would override the pin anyway")
    if args.move_belief_mode in ("unrevealed", "both") and not (args.opp_belief_aux_coef > 0.0):
        # FAIL LOUD on a nonsensical config: 'unrevealed'/'both' score the HIDDEN opp slots, but without
        # the species-belief head (--opp-belief-aux-coef>0) those slots are never filled with learned
        # unknown-mon tokens — they stay encoder placeholders (~zeros). Predicting a hidden mon's moveset
        # from an empty token (with no representation of WHICH mon it is) is meaningless. 'revealed' mode is
        # exempt: it scores REVEALED slots, which carry real role-tokens regardless of the belief head.
        parser.error(
            f"--move-belief-mode {args.move_belief_mode} scores the opponent's HIDDEN slots, which are "
            "only filled with learned unknown-mon tokens when the species-belief head is on. Add "
            "--opp-belief-aux-coef <coef> (>0), or use --move-belief-mode revealed (seen mons only)."
        )
    if args.damage_op and args.move_belief_mode not in ("revealed", "both"):
        # FAIL LOUD: the damage operator reads the opp ACTIVE slot's PREDICTED move logits, which are
        # only supervised/reinjected for a REVEALED mon (revealed|both). Under off/unrevealed the
        # active-slot logits are an unsupervised readout and the belief-gradient story breaks.
        parser.error(
            "--damage-op requires --move-belief-mode revealed (or both): the operator is fed the opp "
            "active's predicted moves, which are only supervised for a revealed mon. Set "
            "--move-belief-mode revealed, or drop --damage-op."
        )
    if args.move_prior_fusion and args.move_belief_mode == "off":
        # FAIL LOUD: prior fusion folds the Smogon prior INTO the move-belief head's logits; with no
        # head (--move-belief-mode off) there is nothing to fuse.
        parser.error(
            "--move-prior-fusion requires --move-belief-mode != off (revealed|unrevealed|both): the prior "
            "fuses into the move-belief head's logits. Set --move-belief-mode revealed, or drop "
            "--move-prior-fusion."
        )
    if args.species_prior_fusion and not (args.opp_belief_aux_coef and args.opp_belief_aux_coef > 0):
        # FAIL LOUD: the species prior fuses INTO BeliefHead's species head, and that head only exists
        # under the in-place believed slots (which --opp-belief-aux-coef>0 is what turns on).
        parser.error(
            "--species-prior-fusion requires --opp-belief-aux-coef > 0: the team-composition prior "
            "fuses into the BeliefHead's species head, which is only built under the hidden-opponent "
            "belief slots. Set --opp-belief-aux-coef, or drop --species-prior-fusion."
        )
    if args.damage_candidate_k and not args.damage_op:
        # FAIL LOUD at the CLI (not at extractor build, which happens only after the run has already
        # tried to stand up a server): the cap narrows the DamageOperator's candidate axis, which
        # only exists when the op is built.
        parser.error(
            "--damage-candidate-k requires --damage-op (it caps the damage operator's incoming "
            "candidate sweep, which only exists when the op is built). Add --damage-op / "
            "--unified-damage, or drop --damage-candidate-k."
        )
    if args.damage_candidate_k and args.damage_candidate_k < 0:
        parser.error("--damage-candidate-k must be >= 0 (0 = the full candidate sweep).")
    if args.damage_outgoing and not args.damage_op:
        # The outgoing per-move block is emitted by the DamageOperator → the op must exist.
        parser.error(
            "--damage-outgoing requires --damage-op (the outgoing block is part of the damage operator). "
            "Use --unified-damage both, or add --damage-op."
        )
    if args.entity_topk_seats and args.entity_topk_seats > 0 and not (
            args.damage_op and args.move_latent):
        # gen3_entity_move_seats_v1: the E4 seats gather the op's PRE-transformer candidate weights
        # and the move latent table — both of which the tiered order produces whenever the op is on.
        parser.error(
            "--entity-topk-seats > 0 requires --damage-op AND --move-latent (--unified-moves): "
            "the E4 threat seats gather the op's pre-transformer candidate weights + move latents. "
            "Add those flags, or set --entity-topk-seats 0 (E3-only)."
        )
    if args.entity_tail_seats and not (args.damage_op
                                       and args.entity_topk_seats and args.entity_topk_seats > 0):
        parser.error("--entity-tail-seats requires --damage-op AND --entity-topk-seats > 0 "
                     "(the tail is defined relative to the E4 seats' truncation).")
    _ebf = args.edge_bias_families
    if _ebf and _ebf != "off":
        # The family vocabulary is the EXTRACTOR'S, single-sourced — a hand-copied set here
        # silently rejected the v79 `h` family at launch (caught by the flag-on bridge smoke:
        # the extractor knew `h`, the CLI did not, so a `,h` launch died in argparse).
        from agents.model.features_extractor import _EDGE_FAMILIES as _valid
        _fams = {"d1", "d3"} if _ebf == "d" else set(_ebf.split(","))
        if _fams - set(_valid):
            parser.error(f"--edge-bias-families: unknown families {sorted(_fams - set(_valid))} "
                         f"(valid: off, d [= d1,d3 frozen], or a comma list of {sorted(_valid)})")
        if (_fams & {"d1", "s1", "c1", "c2"}) and not (args.damage_op and args.damage_outgoing):
            parser.error("--edge-bias-families d1/s1/c1/c2 require --damage-op AND --damage-outgoing "
                         "(--unified-damage both / --unified-moves both).")
        if "x" in _fams and not args.damage_op:
            parser.error("--edge-bias-families x requires --damage-op "
                         "(the Pursuit belief comes from the op's pre-transformer posterior).")
        if (_fams & {"d2", "d4", "v", "t", "g", "c4", "c3", "c5"}) and not args.damage_op:
            parser.error("--edge-bias-families d2/d4/v/t/g/c4/c3/c5 require --damage-op (the op's kernels/buffers).")
        if (_fams & {"d3", "s3"}) and not (args.entity_topk_seats and args.entity_topk_seats > 0):
            parser.error("--edge-bias-families d3/s3 require --entity-topk-seats > 0 (the bias rows "
                         "ARE the E4 threat seats).")
    if not (_MIN_PRIOR_FLOOR <= args.move_candidate_floor < 1.0):
        # gen3_unconditional_move_legality_v1: the floor is the LEGAL-BUT-UNOBSERVED base, and a value at
        # or below the "impossible" probability collapses the legality distinction it exists to preserve.
        # 0.0 in particular is what a pre-v65 resume carries — it used to mean "legality OFF".
        parser.error(
            f"--move-candidate-floor {args.move_candidate_floor} is out of range: it is the "
            f"LEGAL-BUT-UNOBSERVED base of the move prior and must satisfy "
            f"{_MIN_PRIOR_FLOOR} <= value < 1.0 (default {_PRIOR_FLOOR}).\n"
            "Move legality is unconditional and has no off switch; 0.0 is no longer meaningful. "
            "If this came from resuming a pre-v65 checkpoint, that model's belief is incompatible — "
            "start a fresh run."
        )
    if args.move_candidate_floor != _PRIOR_FLOOR and not args.move_prior_fusion:
        # A NON-DEFAULT floor with no prior fusion is a silently-ignored flag: the floor is only read when
        # the fused prior is built. (The default is not flagged — it is just the default.)
        parser.error(
            "--move-candidate-floor requires --move-prior-fusion (it sets the floor of the FUSED move "
            "prior, which only exists under fusion). Enable --move-prior-fusion (or --unified-damage), "
            "or drop --move-candidate-floor."
        )
    if args.damage_topk_k and args.damage_topk_k > 0 and not args.damage_op:
        # The discrete incoming move-space block extends the DamageOperator.
        parser.error(
            "--damage-topk requires --damage-op (the discrete incoming block extends the damage operator). "
            "Use --unified-damage / --unified-moves, or add --damage-op, or set --damage-topk 0."
        )
    if args.damage_topk_k and args.damage_topk_k > 0 and not args.move_latent:
        # The block gathers each candidate move's identity LATENT from the MoveLatentEncoder.
        parser.error(
            "--damage-topk requires --move-latent (the block gathers each move's identity latent "
            "from the MoveLatentEncoder). Use --unified-moves, or add --move-latent, or set --damage-topk 0."
        )
    if args.damage_topk_k and args.damage_topk_k > 0 and not args.damage_matrices_incoming:
        # gen3_op_block_trim_v1: only reachable when --damage-matrices was passed EXPLICITLY as
        # off/outgoing (the implicit case is auto-enabled above). K would size a block that isn't emitted.
        parser.error(
            f"--damage-topk {args.damage_topk_k} contradicts --damage-matrices {args.damage_matrices}: K is "
            "the INCOMING matrix's width, and the lean top-K block it used to select was deleted "
            "(gen3_op_block_trim_v1). Use --damage-matrices incoming/both, or set --damage-topk 0."
        )
    if getattr(args, "damage_matrices_outgoing", False) and not args.damage_op:
        # gen3_per_move_matrices_v1: the outgoing damage matrix is emitted by the DamageOperator.
        parser.error(
            "--damage-matrices outgoing requires --damage-op (the matrix is emitted by the damage operator). "
            "Use --unified-damage / --unified-moves, or add --damage-op, or set --damage-matrices off."
        )
    if getattr(args, "damage_matrices_incoming", False):
        # gen3_per_move_matrices_v1: the incoming matrix needs the op + the move latent, and SUPERSEDES top-K.
        if not args.damage_op:
            parser.error(
                "--damage-matrices incoming requires --damage-op (the matrix is emitted by the damage "
                "operator). Use --unified-damage / --unified-moves, or add --damage-op."
            )
        if not args.move_latent:
            parser.error(
                "--damage-matrices incoming requires --move-latent (the matrix header gathers each move's "
                "identity latent). Use --unified-moves, or add --move-latent."
            )
    # gen3_bidir_threat_trunk_v1 (v36): the uncertainty-aware P(outspeed).
    if getattr(args, "threat_prob_outspeed", False) and not args.damage_op:
        parser.error(
            "--threat-prob-outspeed requires --damage-op (the P(outspeed) feature lives in the damage operator)."
        )
    if args.move_belief_latent_coef and not args.move_latent:
        # The latent grading reads the MoveLatentEncoder's latent table → the encoder must exist.
        parser.error(
            "--move-belief-latent-coef requires --move-latent (the grading reads its per-move latent "
            "table). Enable --move-latent (or --unified-moves), or set --move-belief-latent-coef 0."
        )
    if args.move_belief_latent_coef and args.move_belief_mode not in ("revealed", "both"):
        # The grading scores the move belief on REVEALED slots (slot==species), like the move-belief BCE.
        parser.error(
            "--move-belief-latent-coef requires --move-belief-mode revealed (or both): it grades the "
            "move belief on revealed slots. Set --move-belief-mode revealed (or --unified-moves), or set "
            "--move-belief-latent-coef 0."
        )
    if args.spread_belief_coef and not args.spread_belief:
        # The supervision reads the spread belief's believed stats (last_spread_belief) → the module must exist.
        parser.error(
            "--spread-belief-coef requires --spread-belief (it supervises the believed opp spread). "
            "Enable --spread-belief, or set --spread-belief-coef 0."
        )
    if args.spread_belief_nature and not args.spread_belief:
        # gen3_nature_ev_belief_v1: --spread-belief-nature parameterises the SpreadBelief module → it must exist.
        parser.error(
            "--spread-belief-nature requires --spread-belief (it reparameterises the SpreadBelief head). "
            "Enable --spread-belief, or drop --spread-belief-nature."
        )

    # One server config, built from --showdown-port and threaded to every Showdown client
    # (training-env players in spawn workers, eval, and self-play). Default port: 8000.
    server_config = (
        LocalhostServerConfiguration
        if args.showdown_port is None
        else localhost_server_configuration(args.showdown_port)
    )
    if args.use_showdown_bridge:
        emit(f"🌉 Transport: in-process BattleStream bridge [{args.bridge_impl}] for BOTH training "
             "and eval (no Showdown server needed — --showdown-port ignored)")
        if args.bridge_impl == "rust":
            # One-time startup warning naming the Rust bridge's honest remaining scope limits (the
            # offline search/replay drivers are still Node-only; an INCOMPLETE modeled move set that
            # fail-louds) — resolve/build the binary NOW so a missing toolchain fails loudly at
            # startup, not deep inside the first env reset.
            from utils.bridge.sim_bridge_bin import (
                warn_rust_deferrals, resolve_and_publish_sim_bridge_bin)
            warn_rust_deferrals(emit)
            # Build ONCE here and PUBLISH the path (POKESIM_SIM_BRIDGE_BIN) so every
            # SubprocVecEnv env worker / eval-worker subprocess inherits a ready binary
            # instead of racing its own `cargo build` on first spawn.
            _rust_bin = resolve_and_publish_sim_bridge_bin()
            emit(f"🦀 [BRIDGE=rust] sim_bridge binary (prebuilt, published to children): {_rust_bin}")
            # The search-TEACHER used to be hard-blocked here. That guard is GONE
            # (`gen3_rust_search_driver_v1` / `gen3_rust_replay_driver_v1`): the Rust
            # `search_driver` binary now serves BOTH offline verb families, and
            # `SearchTeacherCallback(impl=args.bridge_impl)` threads this run's engine into the
            # worker subprocesses, so a rust run's teacher no longer silently falls back to node.
            #
            # For the record, since it cost someone an investigation: the guard's ORIGINAL reason —
            # that the search-teacher needs the sim's own byte-identical `input_log` — was simply
            # FALSE. Nothing reads the record's committed-choice lines. The only readers are
            # `replay_kernels.js::writeStart` and `ReconstructionRecord.start_options()` /
            # `.players()`, all of which touch only the `>start` / `>player` lines, which the rust
            # record renders exactly. The real blocker was always the missing DRIVER, and that is
            # what got built.
            if getattr(args, "search_teacher", False) or getattr(args, "teacher_persistent", False):
                emit("🦀 [BRIDGE=rust] search-teacher on the RUST offline drivers "
                     "(search_driver binary serves open_root/expand_many + replay/reroll/"
                     "reroll_many). Gated by: better_line node≡rust candidate values bit-identical, "
                     "search_clone_parity (clone ≡ reroll_many at the obs), and the counterfactual "
                     "confirm leg — each run on rust. NOT yet gated: a full multi-cycle teacher run "
                     "end-to-end on rust. Fall back with --use-bridge=node if a cycle misbehaves.")
    else:
        emit(f"🔌 Showdown server: {server_config.websocket_url}")

    annealing_mode = args.anneal_lr_start_steps is not None
    if annealing_mode:
        if args.anneal_min_lr is None:
            print("[AnnealLR] ERROR: --anneal-min-lr is required when --anneal-lr-start-steps is set")
            sys.exit(1)
        if args.anneal_lr_start_steps >= args.steps:
            print(f"[AnnealLR] ERROR: --anneal-lr-start-steps ({args.anneal_lr_start_steps:,}) "
                  f"must be less than --steps ({args.steps:,})")
            sys.exit(1)

    if args.hp_type_belief_coef and args.move_belief_mode == "off":
        # The CE supervises the HPTypeBelief head's posterior (last_hp_type_logits), and the head is built
        # only alongside a move belief (it composes P(HP present) from the move posterior's 237 channel).
        # EXPLICIT coef + no belief = a real contradiction → error. But the coef DEFAULTS to 0.05
        # (_resolve), so on the DEPRECATED `--unified-moves off` ablation baseline the un-passed default
        # would make the flag fail out of the box — the same shape as the `--hp-belief-mode flat` case
        # below, resolved the same way: AUTO-ZERO with a loud note.
        if _hp_coef_explicit:
            parser.error(
                "--hp-type-belief-coef requires a move belief (--move-belief-mode != off / --unified-moves): "
                "the HP-type head composes P(HP present) out of the move posterior. Enable the move belief, "
                "or set --hp-type-belief-coef 0."
            )
        print("[HPBelief] no move belief (--unified-moves off): auto-zeroing the default "
              "--hp-type-belief-coef (the HP-type head is built only alongside a move belief).")
        args.hp_type_belief_coef = 0.0
    if args.hp_type_belief_coef and args.hp_belief_mode == "flat":
        # The `flat` ablation builds NO HPTypeBelief head, so there is no posterior for the CE to
        # supervise. AUTO-ZERO with a loud note rather than erroring:
        # --hp-type-belief-coef defaults to 0.05, so erroring would make
        # `--hp-belief-mode flat` fail out of the box — a hostile flag to run an ablation with. The
        # note keeps it from being a SILENT no-op, which is the failure that actually matters here.
        print("[HPBelief] --hp-belief-mode flat: auto-zeroing --hp-type-belief-coef (the ablation "
              "builds no HP-type head, so there is no posterior for the CE to supervise). The 16 "
              "typed HP channels are still predicted + supervised by the move-belief BCE.")
        args.hp_type_belief_coef = 0.0
    # The preload IS the opponent compile, moved into the forkserver — so it FOLLOWS
    # --compile-opponents by default (both ship ON). Only an EXPLICIT --compile-opponents-preload
    # alongside --no-compile-opponents is a contradiction worth erroring on; the same pairing
    # arrived at by defaults must silently resolve to "no compile at all", or --no-compile-opponents
    # (the documented fallback) would itself become a usage error.
    try:
        args.compile_opponents_preload = resolve_compile_opponents_preload(
            args.compile_opponents_preload, args.compile_opponents)
    except ValueError as exc:
        parser.error(str(exc))
    if args.item_belief_coef and not args.item_belief:
        # The CE supervises the ItemBelief head's posterior (last_item_logits) and the head exists
        # only under --item-belief; a coef with no head would be a silent no-op (the bank's row
        # sees no stash and returns None), so make the config honest instead of quietly inert.
        print("[ItemBelief] --item-belief off: auto-zeroing --item-belief-coef (the head the CE "
              "supervises is not built; pass --item-belief to enable it).")
        args.item_belief_coef = 0.0
    log_level = LogLevel[args.log_level.upper()]

    # Automatically enable deep traces if --debug is set
    if args.debug:
        log_level = LogLevel.DEBUG
        # Default a smoke run to CPU so it never contends with a live GPU training run.
        # Only the "auto" default is overridden — an explicit --device cpu|cuda still wins.
        # Set before any args.device consumer (pool/opponent/model build are all downstream).
        if args.device == "auto":
            args.device = "cpu"
        # A --debug smoke run is a short-lived child of the launching shell/agent and
        # uses DummyVecEnv (no SubprocVecEnv worker watchdog). If its parent dies it gets
        # orphaned, and a hung smoke (e.g. a vanished 9XXX server) then lingers for days as
        # a zombie. Exit if reparented. Started here — before team/env/server setup — so a
        # hang anywhere in startup is covered too. Real (launcher-managed) runs keep a live
        # parent and are unaffected.
        start_orphan_watchdog(label="debug-smoke")

    # --compile-trainer's AUTO default, resolved AFTER --debug has had its say on the device (a
    # --debug run with no --device is cpu, and must stay a pure-CPU minute-long smoke). An explicit
    # --compile-trainer / --no-compile-trainer always wins; the auto value only fills the None.
    #
    # ⚠️ THE SECOND HALF IS NOT OPTIONAL. `check_shape_stability` REFUSES two configs outright
    # (--async-rollout, and a rollout that does not divide by --batch-size), so a device-only
    # default would convert two classes of command that work today into a FATAL_CONFIG exit —
    # the same failure the cpu conditioning above exists to avoid, one flag over. A DEFAULT yields
    # to the config the user actually typed and says why; an EXPLICIT --compile-trainer still hits
    # the refusal, loudly, because there the user asked for something impossible.
    if args.compile_trainer is None:
        args.compile_trainer, _ct_why = resolve_compile_trainer_auto(
            device=args.device, debug=args.debug, n_steps=args.n_steps, n_envs=args.n_envs,
            batch_size=args.batch_size, async_rollout=bool(getattr(args, "async_rollout", False)))
        if _ct_why:
            emit("⚡ --compile-trainer would be ON by default here, but this config cannot take "
                 f"it — leaving it OFF rather than refusing to launch. Reason: {_ct_why} "
                 "(pass --compile-trainer explicitly to make this a hard error instead.)")
        if args.compile_trainer:
            # LOUD, and at STARTUP rather than only after the compile succeeds: with the default
            # ON, every plain cuda run now trades away the ObservationDebugger, and a trade nobody
            # typed a flag for is exactly the kind that has to announce itself.
            emit("⚡ --compile-trainer ON by default (device=cuda) — ~1.75x on the PPO train step. "
                 "⚠️ this DROPS the ObservationDebugger (dynamo cannot trace its numpy asserts); "
                 "pass --no-compile-trainer to keep it. Compile failure is FATAL by design.")

    return ResolvedRunConfig(server_config=server_config, annealing_mode=annealing_mode,
                             log_level=log_level)
