"""Phase 5 — THE MODEL, and the `learn()` call itself.

Two paths, and they are deliberately parallel rather than merged: a RESUME (`--model`) loads a
checkpoint through `load_model_snapshot`'s version gate and re-applies every training-only
coefficient on top, while a FRESH run constructs `InstrumentedMaskablePPO` and derives its
`ModelVersion` from the same `build_extractor_arch_kwargs` table. Both then compile, round-trip,
save a snapshot, wire the signal handlers, seed the pool, start the watchdog and train.
"""
import os
import sys
import traceback

import torch

from agents.model.extractor_arch import build_extractor_arch_kwargs
from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH
from agents.model.model_version import ModelVersion, ModelVersionError
from agents.model.policy import Gen3DualHeadMaskablePolicy, POLICY_ACTIVATION_FN
from agents.model.snapshot import load_model_snapshot, record_checkpoint, save_model_snapshot
import agents.training.cuda_ledger as _cuda_ledger
import agents.training.update_fit as _update_fit
from agents.training.instrumented_ppo.device_batches import DEFAULT_MODE as _devb_default_mode
from agents.training.distill_anchor_callback import save_anchor_ref_beside
from agents.observation.state_encoder import Gen3ObservationEncoder
from agents.training.adaptive_lr_callback import TwoPhaseLRCallback
from agents.training.dose import kl_controller_snapshot
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.reward_manager import reward_config_digest
from agents.training.watchdog import start_subprocess_watchdog
from main.exit_codes import TrainExitCode, exit_code_for
from main.launcher.ipc import emit, send_event
from main.train.constants import _ABORT_EVAL_DRAIN_SEC
from main.train.checkpoint_state import _validate_or_reset_optimizer_state
from main.train.fork_lr import apply_fork_lr_pin, read_recorded_pin, resolve_fork_lr
from main.train.lifecycle import (
    _apply_grad_checkpointing, _arm_compile_sentinel, _arm_learner_lifecycle, _maybe_compile_trainer,
    _run_roundtrip_test,
    _setup_signal_handlers,
)
from main.train.run_io import (
    _attach_run_tb_logger, _model_hparams, _run_arch_toggles, _run_lineage, _write_latest_txt,
)
from utils.logging.levels import LogLevel


def attach_cf_labels(model, *, args, _cf_labels_dir, reward_config):
    """Attach the counterfactual label buffer + its coefficients (both build paths).

    The coefficients are set UNCONDITIONALLY (they are class defaults otherwise, and a resume
    must be able to turn the term OFF as well as on); the BUFFER is only built when the coef is
    live, so an off run never touches the filesystem. Training-only: RECORDED in
    `model_config.json` for provenance + flagless-resume read-back since config v100
    (`gen3_cf_coef_provenance_v1`), never checked by `check_compatible`.

    `cf_records` / `cf_records_keep` are set here too even though the RING is built in the env
    workers, not off `model` — `lifecycle._run_roundtrip_test` stamps its ModelVersion off the
    model object, so a field only argparse knows about would record as its default there."""
    model.cf_records = bool(args.cf_records)
    model.cf_records_keep = int(args.cf_records_keep)
    model.cf_winprob_coef = float(args.cf_winprob_coef or 0.0)
    model.cf_head_only = bool(args.cf_head_only)
    model.cf_label_lag_steps = int(args.cf_label_lag_steps)
    model.cf_label_likelihood = str(args.cf_label_likelihood)
    model.cf_evidential_coef = float(args.cf_evidential_coef or 0.0)
    model.cf_evidential_reg = float(args.cf_evidential_reg or 0.0)
    model.cf_twin_coef = float(args.cf_twin_coef or 0.0)
    model.cf_shadow_coef = float(args.cf_shadow_coef or 0.0)
    # gen3_q_winprob_head_v1 (v107): the per-action Q head's two coefficients. Set on the MODEL
    # unconditionally, BEFORE the `_cf_labels_dir` early-out, for the reason the block above is
    # ordered this way: `run_io`/`lifecycle` read them off the model into `model_config.json`, so a
    # field only argparse knows about would record as its default there.
    model.q_winprob_coef = float(args.q_winprob_coef or 0.0)
    model.q_winprob_onpolicy_coef = float(args.q_winprob_onpolicy_coef or 0.0)
    if not _cf_labels_dir:
        return
    from agents.training.cf_label_buffer import CfLabelBuffer
    os.makedirs(_cf_labels_dir, exist_ok=True)
    _obs_space = model.observation_space["observation"]
    # gen3_cf_twin_heads_v1: the SHADOW critic's `mc_return` labels are SHAPED returns, so they
    # are only this run's labels if the producer used this run's reward. The digest is handed to
    # the buffer, which drops a mismatching `mc_return` (never the row) and counts it. Passed
    # ONLY when the shadow head is live — a run without one has nothing to protect and should
    # not reject rows over a field it does not read.
    _cf_reward_sha1 = (reward_config_digest(reward_config)
                       if float(args.cf_shadow_coef or 0.0) > 0 else None)
    model._cf_buffer = CfLabelBuffer(
        _cf_labels_dir, obs_dim=int(_obs_space.shape[0]),
        lag_bound=int(args.cf_label_lag_steps), reward_sha1=_cf_reward_sha1)
    if model.cf_winprob_coef > 0:
        emit(f"🎯 [CF] win-prob grounding ON: coef={model.cf_winprob_coef:g} "
             f"likelihood={model.cf_label_likelihood} head_only={model.cf_head_only} "
             f"lag={model.cf_label_lag_steps} ← {_cf_labels_dir}")
    if model.cf_evidential_coef > 0:
        emit(f"🎲 [CF] EVIDENTIAL Beta head ON: coef={model.cf_evidential_coef:g} "
             f"reg={model.cf_evidential_reg:g} (always-detached readout) "
             f"← {_cf_labels_dir}")
    if model.cf_twin_coef > 0:
        emit(f"👯 [CF] TWIN win-prob heads ON: coef={model.cf_twin_coef:g} "
             f"(A=control / B=single-outcome / C=tight-MC, head-only always) "
             f"← {_cf_labels_dir}. Read cf/twin_b_coverage FIRST — a producer shipping no "
             f"outcome_label makes B==A and turns C−B into C−A with no other tell.")
    if model.cf_shadow_coef > 0:
        emit(f"🩻 [CF] SHADOW critic ON: coef={model.cf_shadow_coef:g} "
             f"(passive mc_return readout — no advantage, no GAE) "
             f"reward_sha1={(_cf_reward_sha1 or '')[:12]} ← {_cf_labels_dir}")
    if model.q_winprob_coef > 0 or model.q_winprob_onpolicy_coef > 0:
        emit(f"🎯 [Q] PER-ACTION win-prob head ON: cf_coef={model.q_winprob_coef:g} "
             f"onpolicy_coef={model.q_winprob_onpolicy_coef:g} (read_only — every input "
             f"detached) ← {_cf_labels_dir}. Read q_winprob/label_coverage FIRST: a producer "
             f"shipping no per-action `q_labels` trains the head on nothing, and if the on-policy "
             f"coefficient is the only live one the head learns ONLY where the policy already "
             f"goes — the starvation trap (p≈0.002 preferred-alternative rate).")


# ── The training-hparam passthroughs: ONE declared table, applied on BOTH build paths ──
#
# These sixty-odd `model.<x> = args.<x>` lines existed VERBATIM TWICE — once on the resume path
# and once on the fresh path — differing only in their trailing comments. Two copies of a list
# whose whole job is to be COMPLETE is the failure mode worth designing against here: a new
# coefficient added to one branch and not the other produces a run that silently trains with the
# class default on resume (or on fresh) and nothing anywhere reports it, because every one of
# these is a plain attribute with a plausible default. One table, applied by one function, makes
# that class unrepresentable rather than merely unlikely.
#
# Everything here is TRAINING-ONLY and resume-MUTABLE: not version-locked, never consulted by
# `check_compatible`. The resume-IMMUTABLE one (`vf_coef`) is enforced separately, before this runs.

_PLAIN = None          # model.<x> = args.<x>
_F0 = "f0"             # model.<x> = float(args.<x> or 0.0)  — None/"" coerce to 0.0
_F0_OPT = "f0?"        # ...and tolerate a namespace that has no such dest at all

_TRAINING_HPARAMS: "tuple[tuple[str, str | None], ...]" = (
    ("grad_accum_steps",              _PLAIN),   # 1 = off; effective batch = batch_size·K
    ("opp_belief_aux_coef",           _PLAIN),   # hidden-opp belief aux loss (0.0 = off)
    ("opp_belief_moves_weight",       _PLAIN),   # species_CE + w·moves_BCE
    ("move_belief_coef",              _PLAIN),   # move-belief reinjection loss (0.0 = off)
    ("move_belief_latent_coef",       _PLAIN),   # move-latent grading loss (0.0 = off)
    ("spread_belief_coef",            _PLAIN),   # spread-belief speed supervision (0.0 = off)
    ("hp_type_belief_coef",           _PLAIN),   # HP-type CE (0.0 = no direct CE)
    ("item_belief_coef",              _PLAIN),   # item CE (0.0 = no direct CE)
    ("td_aux_coef",                   _PLAIN),   # TD-consistency aux (0.0 = byte-identical)
    ("policy_grad_coef",                       _PLAIN),   # policy-gradient term weight (1.0 = upstream)
    ("intent_label_bot_weight",       _PLAIN),   # gen3_intent_label_bot_weight_v1 (1.0 = off)
    ("win_prob_strata_weight",        _PLAIN),   # gen3_winprob_strata_weight_v1 (0.0 = bit-identical)
    # gen3_fork_v1 — the FORK ARM. All six _PLAIN: they are read off the model by
    # `ForkArmCallback` once per rollout and never enter a forward pass or a weight shape.
    ("fork_fraction",                 _PLAIN),   # 0.0 = OFF and bit-identical
    ("fork_branches",                 _PLAIN),   # 3 = top-2 + one uniformly random legal action
    ("fork_contested_gap",            _PLAIN),   # the top-2 logit-gap QUANTILE (inert at 0.0)
    ("fork_contested_absv",           _PLAIN),   # the |V-0.5| band; 0.0 = off (see the flag)
    ("fork_max_per_battle",           _PLAIN),   # forks per episode slice (inert at 0.0)
    ("fork_crn",                      _PLAIN),   # dice | dice_and_draws (inert at 0.0)
    # SEARCH-TEACHER (coef 0 / flag absent = byte-identical). The buffer is filled by the
    # SearchTeacherCallback from worker shards; the AWR aux loss in train() samples it.
    ("search_teacher_coef",           _PLAIN),
    ("search_teacher_value_coef",     _PLAIN),
    ("search_teacher_beta",           _PLAIN),
    ("search_teacher_batch_size",     _PLAIN),
    # OPD (on-policy self-distillation). Requires --search-teacher: it fills the SAME
    # _correction_buffer, its workers building π'.
    ("opd_coef",                      _PLAIN),
    ("opd_beta",                      _PLAIN),
    # gen3_exploiter_distill_v1 — the KL/value weights; the TEACHERS are loaded below.
    ("distill_coef",                  _F0),
    ("distill_value_coef",            _F0),
    ("distill_value_feat_coef",       _F0),      # gen3_exploiter_value_feat_distill_v1
    # gen3_distill_target_gate_v1 — the action-form/top-K target + advantage gate (v103).
    # (`rank_tripwire`/`rank_tripwire_drop` are CALLBACK config, not model attrs — see
    # main.train.callbacks — so they are deliberately not rows here.)
    ("distill_target",                _PLAIN),
    ("distill_topk",                  _PLAIN),
    ("distill_gate",                  _PLAIN),
    ("distill_gate_tau",              _PLAIN),
    ("distill_beta",                  _PLAIN),
    ("opp_intent_coef",               _F0_OPT),
    ("beta_setvalued_coef",           _F0_OPT),
    # gen3_capacity_telemetry_v1 — the live saturation early-warnings. Folds NO loss term and
    # writes no `.grad`, so an ON run's parameter updates are bit-identical to an OFF one; these
    # four only decide whether the `capacity/*` scalars exist and at what cadence.
    ("capacity_telemetry",            _PLAIN),
    ("canary_reset_steps",            _PLAIN),
    ("capacity_cosine_every",         _PLAIN),
    ("capacity_velocity_every",       _PLAIN),
)


def apply_training_hparams(model, args, *, mappings, attach_cf_labels) -> None:
    """Apply every training-only hparam to `model`. Called from BOTH build paths, identically.

    The table above covers the passthroughs. The things that follow it are NOT
    passthroughs and so are deliberately not table rows: two DERIVED booleans (a different arg
    name, and a predicate over a coefficient), one DERIVED float (a diagnostic denominator read
    off a different flag), and the distill-teacher load, which does I/O and
    can FATAL. Keeping the derived ones in code rather than inventing a table dialect to hold
    them is the point — the table stays a list of names, which is the thing that has to be
    reviewable at a glance for completeness.
    """
    for name, how in _TRAINING_HPARAMS:
        if how is _F0_OPT:
            setattr(model, name, float(getattr(args, name, 0.0) or 0.0))
        elif how is _F0:
            setattr(model, name, float(getattr(args, name) or 0.0))
        else:
            setattr(model, name, getattr(args, name))

    # gen3_capacity_telemetry_v1: SAY SO at launch. Two of this instrument's properties are
    # counter-intuitive enough that a silent ON is a misreading waiting to happen — the canary's
    # state does not survive a resume, and every scalar is a TREND rather than a level.
    from agents.training.instrumented_ppo.capacity_terms import capacity_startup_banner
    _cap_line = capacity_startup_banner(model)
    if _cap_line:
        emit(_cap_line)

    # gen3_fork_lr_pin_v1 — DOSE PROVENANCE. `--lr` is INERT on a resume (the optimizer's saved LR
    # wins), so a reader of `metadata.json` has to be able to see BOTH what the flag said and what
    # the optimizer is doing. `agents.training.dose.dose_block` reads this off the model; argparse is
    # the only place it exists, and this function is the one place both build paths meet.
    if getattr(args, "lr", None) is not None:
        model._dose_lr_flag = float(args.lr)

    # DERIVED, not passthrough: the arg is the flag, the attribute is the predicate.
    model._search_teacher_on = bool(args.search_teacher)
    # gen3_diagnostics_cadence_v1: `--diagnostics-every` (resolved: a fresh run's default, or the
    # parent's recorded value on a flagless resume), plus the two LOAD-BEARING exemptions DERIVED
    # from the consumers' own flags — each predicate is the one that registers that consumer
    # (`main.train.callbacks`), so a registered consumer can never read a thinned series.
    # (A namespace without the dest — a hand-built test namespace — keeps the class default 1.)
    model.diagnostics_every = int(getattr(args, "diagnostics_every", None) or 1)
    # M5 Lane G — K9(b) behaviour-policy consistency (`--behaviour-check`): resolved per env core
    # (`rust_env_setup.resolve_env_core_args`: fatal on both env cores — M5 Lane K9). A namespace
    # without the dest keeps it off.
    model.behaviour_check = str(getattr(args, "behaviour_check", None) or "off")
    # gen3_device_batch_mode_v1 (`--device-batch`): how train()'s micro-batches reach the device —
    # bit-identical in every mode; a namespace without the dest keeps the module default.
    model.device_batch_mode = str(getattr(args, "device_batch", None) or _devb_default_mode)
    model.rank_probe_every_update = getattr(args, "rank_tripwire", "warn") != "off"
    model.noise_terms_every_update = getattr(args, "adaptive_batch", "off") not in ("off", "total")
    model._opd_on = bool(args.opd_coef and args.opd_coef > 0)

    # gen3_cf_label_plumbing_v1: counterfactual win-prob grounding (coef 0 = byte-identical).
    attach_cf_labels(model)

    # gen3_fork_v1: the FORK ARM injects branch transitions into the rollout buffer, which the
    # stock buffer has nowhere to put — `get()` iterates exactly `buffer_size * n_envs`. Installed
    # HERE because this function is the one place both build paths meet and it runs AFTER
    # `_setup_model` built the buffer being replaced. OFF ⇒ not even imported.
    if (float(getattr(args, "fork_fraction", 0.0) or 0.0) > 0.0
            and getattr(args, "env_core", "python") != "rust"):   # rust: branch rows ride the FIFO (forks.md §14)
        from agents.training.fork_buffer import install_fork_buffer
        install_fork_buffer(model)

    # gen3_exploiter_distill_v1: attach the frozen per-team teachers (foreign exploiters) on the
    # training device. OFF (coef 0 / no teacher) → the list stays empty so the loss block is
    # skipped (byte-identical). A bad path FATALs config, never a crash-restart loop.
    #
    # THE COEFFICIENT IS THE GATE, NOT THE PAIRS (gen3_distill_bias_at_coef0_v1). Since the fix,
    # `_distill_pairs` is populated at ANY coefficient — the team bias reads it — so a coef-0
    # CONTROL arm now arrives here with N teachers named. It must still load NONE of them: N frozen
    # networks of RAM and a forward per minibatch, to be multiplied by a coefficient of zero.
    model._distill_teachers = []          # teacher-id = index + 1
    if args.distill_coef and args.distill_coef > 0 and getattr(args, "_distill_pairs", None):
        from agents.model.snapshot import (
            current_model_version as _cmv_d, load_foreign_opponent as _lfo_d)
        from agents.training.fixed_opponent_pool import resolve_model_ref as _rmr_d
        _cv_d = _cmv_d(mappings, **_run_arch_toggles(args))
        # WHICH FILE each teacher resolved to, printed per teacher (gen3_last_snapshot_resolution_v1).
        # A bare run dir names a RUN, not a file, and until 2026-09-06 it silently meant the
        # BOT-WIN-RATE `best_model` export — which for 2 of 8 R5F teachers was a ~0.93M-step
        # checkpoint rather than the ~2.93M final, with nothing recording it (ledger 2026-09-06,
        # probe H8). The rung + num_timesteps are stated so a reader never infers them from layout.
        _teacher_prov: "list[str]" = []
        for _tp, _tf in args._distill_pairs:
            try:
                _ref_d = _rmr_d(_tp, None)              # run-dir/zip/@step → the resolved file
                _tm_d, _ = _lfo_d(_ref_d.zip_path, current_version=_cv_d,
                                  device=str(model.device), config_path=_ref_d.config_path)
                _tm_d.policy.set_training_mode(False)
                model._distill_teachers.append(_tm_d)
                _teacher_prov.append(f"{_tp} -> {_ref_d.describe()}")
            except Exception as _e_d:  # noqa: BLE001 — bad/incompatible teacher weights
                print(f"\n[Distill] FATAL: could not load --distill-teacher {_tp}: {_e_d}")
                sys.stdout.flush()
                os._exit(int(TrainExitCode.FATAL_CONFIG))
        emit(f"🧪 [DISTILL] {len(model._distill_teachers)} teacher(s) attached on {model.device} "
             f"(order = teacher-id 1..{len(model._distill_teachers)})")
        for _i_d, _line_d in enumerate(_teacher_prov, start=1):
            emit(f"   teacher {_i_d}: {_line_d}")

    if args.search_teacher:
        from agents.training.teacher.buffer import CorrectionBuffer
        model._correction_buffer = CorrectionBuffer(args.search_teacher_buffer_size)


def _start_rust_env(env, model) -> None:
    """M5 Lane G (`--env-core rust`): the Rust env's STARTUP (core, T2, arena) from the model's own
    policy and hyperparameters. It must run BEFORE `--compile-trainer`: T2 deep-copies the policy as its
    slot templates, and a copy taken after the compile would carry the patched `forward` bound to the
    LEARNER's extractor. A no-op on the python env core."""
    startup = getattr(env, "startup", None)
    if callable(startup):
        startup(model)


async def build_and_train(*, args, env, mappings, model_dir, cli_args, log_level, n_envs,
                          reward_config, reward_composition, annealing_mode, _async_rollout,
                          _shutdown_event, _run_eval, _effective_max_lr,
                          callbacks, eval_callback, lr_callback, adaptive_ppo_callback,
                          graceful_restart_callback,
                          _attach_cf_labels, _maybe_seed_pool, evaluate_model_random) -> None:
    """Load or construct the model, then run (and finish) the training job."""
    # THE FORK PARENT, captured BEFORE the consensus warm-start may re-point `args.model` at
    # `<run>/warmstart/…` — that init is built FROM the parent, so recording it would make the run
    # its own ancestor. See `run_io._run_lineage` / `agents.training.lineage`.
    _fork_source_model = args.model
    # gen3_exploiter_consensus_warmstart_v1: build (ONCE) the disagreement-gated consensus warm-start and
    # re-point --model at it, so the exploiter inits from a competent, archetype-NEUTRAL base (sharp where
    # the teacher exploiters AGREE, high-entropy where they FORK). Exploiter-only (guarded at parse).
    # IDEMPOTENT under launcher restarts: skip entirely once ANY training checkpoint exists (the normal
    # resume path continues from it); otherwise (re)use the built warm-start as the init. The warm-start is
    # arch-identical to --model (its model_config.json is copied), so the resume-immutable checks above,
    # computed on the original --model, stay valid.
    if args.warmstart_consensus:
        _ws_has_ckpt = (os.path.isdir(os.path.join(model_dir, "checkpoints"))
                        and any(f.endswith(".zip")
                                for f in os.listdir(os.path.join(model_dir, "checkpoints"))))
        if _ws_has_ckpt:
            print("🌱 [WARMSTART] training checkpoint already present → skipping consensus warm-start "
                  "(resuming trained state).", flush=True)
        else:
            from agents.training.warmstart import run_consensus_warmstart, warmstart_failed
            from agents.training.fixed_opponent_pool import _resolve_zip_and_config as _ws_resolve
            # Imported HERE. In the monolith this name reached the warm-start block only because
            # the `--exploiter` branch (which the flag requires) happened to have imported it
            # locally hundreds of lines earlier — a latent NameError one guard away.
            from agents.model.snapshot import current_model_version as _current_model_version
            _ws_dir = os.path.join(model_dir, "warmstart")
            _ws_ckpt = os.path.join(_ws_dir, "warmstart_consensus.zip")
            if not os.path.exists(_ws_ckpt):
                # gen3_supply_guard_v2: a warm-start failure is FATAL_CONFIG, never CRASH. The
                # warm-start is rebuilt FROM SCRATCH on every restart (no checkpoint exists yet), so a
                # restart replays the identical failure — and one that fails after the launcher's
                # 10-minute rapid-crash window (battles + BC run longer than that) reset the breaker
                # every time and crash-looped without bound. A partial artifact is removed so it can
                # never be mistaken for a built init.
                try:
                    _ws_s_zip, _ws_s_cfg, _ = _ws_resolve(args.model, None)
                    _ws_teachers = {}
                    for _wi, _wt in enumerate([x.strip() for x in args.warmstart_consensus.split(",") if x.strip()]):
                        _wz, _wcfg, _ = _ws_resolve(_wt, None)
                        _ws_teachers[f"t{_wi + 1}"] = (_wz, _wcfg)
                    _ws_cv = _current_model_version(mappings, **_run_arch_toggles(args))
                    print(f"🌱 [WARMSTART] disagreement-gated consensus of {len(_ws_teachers)} teacher(s) "
                          f"→ exploiter init ({args.warmstart_battles} battles, {args.warmstart_bc_steps} BC "
                          f"steps)", flush=True)
                    await run_consensus_warmstart(_ws_s_zip, _ws_s_cfg, _ws_teachers, _ws_dir, _ws_cv, mappings,
                                                  battles=args.warmstart_battles, bc_steps=args.warmstart_bc_steps,
                                                  device=str(args.device))
                except Exception as _ws_e:  # noqa: BLE001 — re-raised as the typed FATAL below
                    raise warmstart_failed(_ws_e, _ws_ckpt) from _ws_e
            args.model = _ws_ckpt        # init training from the warm start

    if args.model:
        model_path = args.model
        if not os.path.exists(model_path) and not model_path.endswith(".zip"):
            potential_paths = [
                os.path.join("models", "goldens", model_path),
                os.path.join("models", "goldens", model_path, "final_model"),
                os.path.join("models", "goldens", model_path, "final_model.zip"),
            ]
            for p in potential_paths:
                if os.path.exists(p) or os.path.exists(p + ".zip"):
                    model_path = p
                    break

        # Build the current-code version for compatibility check
        _load_encoder = Gen3ObservationEncoder(mappings)
        # ONE source of truth for every version-checked arch toggle
        # (agents.model.extractor_arch.build_extractor_arch_kwargs). The fresh-run path below
        # builds the SAME dict from the SAME table, so a new v51 toggle cannot land on one
        # path and not the other — which would make a resume version-check an arch it did not
        # build.
        _load_extractor_kwargs = build_extractor_arch_kwargs(
            args, base=_load_encoder.get_features_extractor_kwargs())
        _load_policy_kwargs = {
            "features_extractor_class": Gen3FeaturesExtractor,
            "features_extractor_kwargs": _load_extractor_kwargs,
            "net_arch": NET_ARCH,
            # gen3_policy_activation_pin_v1: carried for PARITY with the fresh-run dict below, so
            # the two policy_kwargs sites cannot drift. It has no effect on the resume itself —
            # ModelVersion records no activation field, and SB3 rebuilds the loaded policy from the
            # ZIP's OWN saved policy_kwargs, not from this dict.
            "activation_fn": POLICY_ACTIVATION_FN,
            "critic": args.critic,  # gen3_winprob_critic_mode_v1: WHICH readout is the critic
        }
        current_version = ModelVersion.from_layout_and_policy_kwargs(
            _load_extractor_kwargs["layout"], _load_policy_kwargs, vf_coef=args.vf_coef,
            reward_config=reward_config,
            opp_belief_aux_coef=args.opp_belief_aux_coef,
            move_belief_coef=args.move_belief_coef,
            move_belief_latent_coef=args.move_belief_latent_coef,
            spread_belief_coef=args.spread_belief_coef,
            hp_type_belief_coef=args.hp_type_belief_coef,
            item_belief_coef=args.item_belief_coef,
            td_aux_coef=args.td_aux_coef,
            arch_source=getattr(args, "arch_source", None),
            policy_grad_coef=args.policy_grad_coef,
            intent_label_bot_weight=args.intent_label_bot_weight,
            win_prob_strata_weight=args.win_prob_strata_weight,
            policy_gae_lambda=args.policy_gae_lambda,
            diagnostics_every=args.diagnostics_every,
            opp_intent_coef=float(args.opp_intent_coef or 0.0),
            fork_fraction=args.fork_fraction,
            fork_branches=args.fork_branches,
            fork_contested_gap=args.fork_contested_gap,
            fork_contested_absv=args.fork_contested_absv,
            fork_max_per_battle=args.fork_max_per_battle,
            fork_crn=args.fork_crn,
            cf_records=args.cf_records,
            cf_records_keep=args.cf_records_keep,
            cf_winprob_coef=args.cf_winprob_coef,
            cf_head_only=args.cf_head_only,
            cf_label_lag_steps=args.cf_label_lag_steps,
            cf_label_likelihood=args.cf_label_likelihood,
            cf_evidential_coef=args.cf_evidential_coef,
            cf_evidential_reg=args.cf_evidential_reg,
            cf_twin_coef=args.cf_twin_coef,
            cf_shadow_coef=args.cf_shadow_coef,
            q_winprob_coef=args.q_winprob_coef,
            q_winprob_onpolicy_coef=args.q_winprob_onpolicy_coef,
            # gen3_eval_sentinel_greedy_default_v1 (v112): the EVAL REGIME + the gate it
            # derives, so a flagless resume reads its own regime back instead of crossing
            # an opponent-regime boundary. Both are RESOLVED by `resolve_config`.
            eval_sentinel_greedy=args.eval_sentinel_greedy,
            promote_threshold=args.promote_threshold,
            eval_mirrored_pairs=args.eval_mirrored_pairs,
            promotion_sprt=args.promotion_sprt,
            capacity_telemetry=args.capacity_telemetry,
            canary_reset_steps=args.canary_reset_steps,
            capacity_cosine_every=args.capacity_cosine_every,
            capacity_velocity_every=args.capacity_velocity_every,
            distill_target=args.distill_target,
            distill_topk=args.distill_topk,
            distill_gate=args.distill_gate,
            distill_gate_tau=args.distill_gate_tau,
            distill_beta=args.distill_beta,
            rank_tripwire=args.rank_tripwire,
            rank_tripwire_drop=args.rank_tripwire_drop,
            teacher_scan_limit=args.teacher_scan_limit,
        )

        print(f"Loading existing model from {model_path}")
        try:
            model = load_model_snapshot(
                model_path,
                env=env,
                current_version=current_version,
                device=args.device,
                enforce_vf_coef=args.vf_coef,  # FATAL if the run was started with a different vf_coef
                enforce_reward_config=reward_config,  # FATAL if victory_value/terminal_indicator/draw_penalty drift
                enforce_belief_grad_mode=args.belief_grad_mode,  # FATAL if the belief-trunk-grad mode drifts (v41)
                allow_belief_grad_mode_change=args.allow_belief_grad_mode_change,  # intentional migration
            )
            # gen3_belief_grad_mode_v1 MIGRATION FIX: SB3 reconstructs the extractor from the ZIP's
            # saved policy_kwargs, so the requested mode must be APPLIED to the live extractor
            # post-load (else --allow-belief-grad-mode-change is a silent no-op — the 2026-07-21
            # incident, visible as grad/*_norm_shared == 0 under 'shaping'). No-op when unchanged.
            model.policy.features_extractor.set_belief_grad_mode(args.belief_grad_mode)
        except ModelVersionError as e:
            print(f"\n[ModelVersion] FATAL: {e}")
            sys.stdout.flush()  # os._exit() skips buffer flushing — make sure the reason reaches the log
            # Non-recoverable: an arch-family / vf_coef / reward-config mismatch fails the
            # SAME way on every retry. Exit with FATAL_CONFIG so the launcher gives up
            # immediately instead of auto-restarting into the identical error.
            os._exit(int(TrainExitCode.FATAL_CONFIG))
        # Guard: if a param-reorder refactor since this checkpoint desynced the position-keyed Adam
        # state, REMAP the momentum to the current params BY NAME (reading the saved order from the
        # checkpoint zip), so a reorder — same-shape (silent) or different-shape (the
        # gen3_nature_ev_belief_v1 SpreadBelief crash) — is corrected instead of scrambled. Before any
        # LR read. model_path is the resolved checkpoint zip used for the load just above.
        _validate_or_reset_optimizer_state(model, model_path)
        model.ent_coef = args.ent_coef          # resume-only: the fresh path passes it to the ctor
        # Every training-only hparam, from the one table shared with the fresh path below.
        apply_training_hparams(model, args, mappings=mappings,
                               attach_cf_labels=_attach_cf_labels)
        # K9(b): where a behaviour violation's row dump is appended (`consistency.VIOLATION_DUMP`).
        model.behaviour_dump_dir = model_dir
        model.vf_coef = args.vf_coef  # == the saved value (enforced above); set explicitly for parity
        # gen3_policy_gae_lambda_v1: `--policy-gae-lambda`, resolved by `_resolve` (a flagless resume
        # INHERITS the parent's recorded value; a pre-v123 parent migrates to the old hardcoded 0.80).
        model.gae_lambda = float(args.policy_gae_lambda)
        # Resume-path LR setup. Phase determines whether we read from the
        # optimizer (Phase 1, KL-driven) or compute the cosine (Phase 2).
        saved_lr: float | None = None  # only set in branches that read it
        if annealing_mode:
            t = model.num_timesteps
            if lr_callback.phase(t) == 1:
                # Phase 1: KL-driven adaptation continues from the optimizer's saved
                # LR. --lr is a fresh-run seed only; on resume it's ignored so the
                # controller can keep whatever rate Phase 1 had settled on. The
                # saved LR is still clamped into [min_lr, max_lr] in case the user
                # tightened the bounds between restarts.
                saved_lr = model.policy.optimizer.param_groups[0]["lr"]
                resume_lr = saved_lr
                resume_lr_clamped = max(args.min_lr, min(resume_lr, _effective_max_lr))
                if resume_lr_clamped != resume_lr:
                    print(
                        f"[TwoPhaseLR] Clamping resume LR {resume_lr:.2e} → {resume_lr_clamped:.2e} "
                        f"to fit [{args.min_lr:.2e}, {_effective_max_lr:.2e}] (saved={saved_lr:.2e})."
                    )
                resume_lr = resume_lr_clamped
                model.lr_schedule = lambda _: resume_lr
                lr_callback._current_lr = resume_lr
                lr_detail = f"Phase 1 adaptive, saved={saved_lr:.2e} (arg --lr={args.lr:.2e} ignored on resume)"
                send_event(
                    f"▶️ Resuming TwoPhaseLR Phase 1 at LR {resume_lr:.2e}, step {t:,} "
                    f"(anneal_start={args.anneal_lr_start_steps:,})"
                )
            else:
                # Phase 2: cosine decay. handoff_lr was passed to the constructor
                # from the sidecar; fall back to the optimizer's LR if missing
                # (legacy run that pre-dates handoff_lr persistence).
                if lr_callback.handoff_lr is None:
                    lr_callback._handoff_lr = model.policy.optimizer.param_groups[0]["lr"]
                    print(
                        f"[TwoPhaseLR] No persisted handoff_lr on resume; "
                        f"using optimizer LR {lr_callback._handoff_lr:.2e} as cosine start."
                    )
                resume_lr = lr_callback._cosine_lr_at(t)
                model.lr_schedule = lambda _: resume_lr
                lr_callback._current_lr = resume_lr
                lr_detail = (
                    f"Phase 2 cosine, handoff={lr_callback.handoff_lr:.2e} → "
                    f"min={args.anneal_min_lr:.2e}"
                )
                send_event(
                    f"▶️ Resuming TwoPhaseLR Phase 2 (cosine) at LR {resume_lr:.2e}, step {t:,} "
                    f"(handoff={lr_callback.handoff_lr:.2e}, target={args.anneal_min_lr:.2e} at {args.steps:,})"
                )
        else:
            # Pure adaptive: resume LR from optimizer state. --lr is a fresh-run
            # seed only; on resume the controller keeps whatever rate it had
            # settled on. The saved LR is still clamped into [min_lr, max_lr]
            # in case the user tightened the bounds between restarts.
            saved_lr = model.policy.optimizer.param_groups[0]["lr"]
            resume_lr = saved_lr
            resume_lr_clamped = max(args.min_lr, min(resume_lr, _effective_max_lr))
            if resume_lr_clamped != resume_lr:
                print(
                    f"[AdaptiveLR] Clamping resume LR {resume_lr:.2e} → {resume_lr_clamped:.2e} "
                    f"to fit [{args.min_lr:.2e}, {_effective_max_lr:.2e}] (saved={saved_lr:.2e})."
                )
            resume_lr = resume_lr_clamped
            model.lr_schedule = lambda _: resume_lr
            adaptive_ppo_callback._current_lr = resume_lr
            lr_detail = f"saved={saved_lr:.2e} (arg --lr={args.lr:.2e} ignored on resume)"
            send_event(f"▶️ Resuming at LR {resume_lr:.2e}, epochs {args.n_epochs} (checkpoint LR={saved_lr:.2e})")
        # gen3_fork_lr_pin_v1 — `--fork-lr`. LAST WORD on the LR, deliberately after both branches
        # above: under `--fork-lr-freeze` a Phase-2 resume would otherwise have just installed a
        # cosine value into the controller's `_current_lr`, and the freeze holds whatever is there.
        # The fork-vs-restart discrimination (and the reason a periodic restart must NOT re-pin) is
        # in `main.train.fork_lr`.
        _fork_decision = resolve_fork_lr(
            fork_lr=getattr(args, "fork_lr", None),
            fork_lr_freeze=bool(getattr(args, "fork_lr_freeze", False)),
            model_path=model_path, model_dir=model_dir)
        if _fork_decision.apply or getattr(args, "fork_lr", None) is not None:
            emit(f"🎚️ [ForkLR] {_fork_decision.reason}")
        if _fork_decision.apply:
            model._fork_lr_pin = apply_fork_lr_pin(
                model, _fork_decision, lr_callback=lr_callback,
                min_lr=args.min_lr, max_lr=_effective_max_lr, source_model=model_path)
            resume_lr = model.policy.optimizer.param_groups[0]["lr"]
            lr_detail = (f"PINNED by --fork-lr to {resume_lr:.2e}"
                         + (" and FROZEN" if _fork_decision.frozen else ""))
        else:
            # Carry any recorded pin forward so a restart's metadata still states what this run's
            # step size was set to and by what — a provenance block that evaporates on restart is
            # worse than none, because the run reads as if it had never been pinned.
            model._fork_lr_pin = read_recorded_pin(model_dir)
        # PLAIN DATA, taken AFTER the pin so a `--fork-lr-freeze` is captured. Never the callback
        # itself — it back-references the model and SB3's Logger, which cloudpickle cannot save.
        model._dose_kl = kl_controller_snapshot(lr_callback)
        model.n_epochs = args.n_epochs   # resume-only: the fresh path passes it to the ctor
        # (`grad_accum_steps` was set here too, and again on the fresh path — it is now one row
        #  in `_TRAINING_HPARAMS`, applied on both. Nothing between there and here reads it.)
        model.clip_range = lambda _: args.clip_range
        # None must stay a bare None (disabled), not `lambda _: None` — SB3 / the
        # instrumented update branch on `clip_range_vf is None`, and a callable is not None.
        model.clip_range_vf = None if args.clip_range_vf is None else (lambda _: args.clip_range_vf)

        if args.eval_only:
            await evaluate_model_random(model)
            return
        else:
            remaining_steps = args.steps - model.num_timesteps
            if remaining_steps <= 0:
                print(f"Training already complete ({model.num_timesteps:,} / {args.steps:,} steps)")
                sys.exit(TrainExitCode.COMPLETE)
            print(f"Continuing Training (Steps: {remaining_steps:,} remaining of {args.steps:,}, LR: {resume_lr:.2e} ({lr_detail}))")
            # gen3_winprob_critic_mode_v1: `--gamma` is INERT ON A RESUME, exactly like `--lr` —
            # SB3 restores the checkpoint's own gamma, so the argv's value never reaches GAE. STATE
            # it rather than let a resumed run silently discount differently from what its command
            # says, and RE-POINT the reward config's copy at the value actually in force (it is
            # recorded, and hashed into `reward_config_digest`).
            if abs(float(reward_config.gamma) - float(model.gamma)) > 1e-12:
                print(f"[Resume] gamma: using the checkpoint's {float(model.gamma):g} "
                      f"(arg --gamma={float(reward_config.gamma):g} ignored on resume, like --lr); "
                      f"the reward config's copy follows it.")
                reward_config.gamma = float(model.gamma)
            _ledger = _cuda_ledger.start(model.device)   # gen3_cuda_ledger_v1: where the card goes
            _start_rust_env(env, model)   # M5 Lane G: BEFORE the trainer's compile step
            _ledger.mark("rust env core (T2 slots, staging, arena; python core: none)")
            _maybe_compile_trainer(model, args)
            _run_roundtrip_test(model, _load_extractor_kwargs["layout"], _load_policy_kwargs, debug=args.debug)
            _apply_grad_checkpointing(model, args.grad_checkpointing)
            _arm_compile_sentinel(model, args)   # gen3_compile_sentinel_v1: reset, prewarm, lock
            _ledger.mark("compiled regions: gate + prewarm + lock")
            _arm_learner_lifecycle(model, args)  # K6 gen3_learner_freeze_v1: declare, then freeze guard
            _ledger.mark("optimizer state declared (Adam m, v) + lifecycle")
            _ledger.report(model_dir)
            # gen3_update_fit_v1: RUN one dry update (restored exactly) and refuse a first update that
            # would not fit with the declared headroom — at startup, never an OOM at update 1.
            _fit = _update_fit.check_update_fits(model, model_dir)
            if _fit:
                print(_fit, flush=True)
            model._async_rollout = _async_rollout   # route collect_rollouts to the non-barrier path
            # gen3_run_lineage_v1 — written ONCE at fork creation and preserved by every later save.
            # `None` on a same-run restart, which is what keeps the recorded parent immutable.
            _lineage = _run_lineage(args, model_dir, model_path=_fork_source_model,
                                    fork_step=int(getattr(model, "num_timesteps", 0) or 0))
            save_model_snapshot(model_dir, current_version, hparams=_model_hparams(model), cli_args=cli_args,
                                reward_composition=reward_composition, lineage=_lineage)
            # gen3_tb_inherit_v1 — a FORK inherits its parent's scalar curves (steps <= fork_step),
            # so its TensorBoard reads from step 0 instead of starting mid-air. Driven off the very
            # block just recorded above: `_lineage` is non-None ONLY on a fork (build_lineage returns
            # None on a same-run restart via `fork_lr.is_same_run_checkpoint`), and the parent +
            # fork_step are read out of it — so the curve a fork inherits and the parent it claims
            # cannot disagree. Runs BEFORE _attach_run_tb_logger so the prefix is in place before the
            # run's own writer opens. Never raises: it returns a reason instead (see the module).
            from agents.training.tb_inherit import inherit_from_lineage as _inherit_tb
            print(_inherit_tb(model_dir, _lineage,
                              enabled=bool(getattr(args, "tb_inherit", True))).describe())

            _abort_fn = _setup_signal_handlers(
                model, model_dir, _shutdown_event, current_version,
                lambda: model.policy.optimizer.param_groups[0]["lr"],
                lambda: model.n_epochs,
                handoff_lr_fn=(
                    (lambda: lr_callback.handoff_lr)
                    if isinstance(lr_callback, TwoPhaseLRCallback) else None
                ),
                eval_drain_fn=(
                    (lambda: eval_callback.drain(timeout=_ABORT_EVAL_DRAIN_SEC))
                    if (eval_callback is not None and hasattr(eval_callback, "drain")) else None
                ),
            )
            if eval_callback is not None:
                eval_callback.abort_fn = _abort_fn
            graceful_restart_callback.abort_fn = _abort_fn

            # Seed the pool from these weights iff self-play is active and the pool is empty
            # (no env rebuild — workers re-scan the dir on demand). No-op when below threshold
            # or the pool already has snapshots. Then start the worker watchdog before rollouts.
            _maybe_seed_pool(model)
            start_subprocess_watchdog(env, label="train_env", shutdown_event=_shutdown_event)

            _attach_run_tb_logger(model, model_dir)  # TB → <model_dir>/tb/ (resumes append to it)
            try:
                # `remaining_steps`, NOT `args.steps`. SB3's `_setup_learn` does
                # `total_timesteps += self.num_timesteps` whenever `reset_num_timesteps=False`, so
                # passing the ABSOLUTE target here re-adds the steps already trained and silently
                # doubles the budget: a resume at 24.08M with --steps 25M retargeted to ~49M. The run
                # printed "915,520 remaining of 25,000,000" and kept going 1M steps past the target —
                # the message was computed correctly and then not used. gen-9 hit this too (it was at
                # 26M against a 25M budget and had to be killed by hand); gen-10 reached 26.05M.
                model.learn(total_timesteps=remaining_steps, callback=callbacks,
                            reset_num_timesteps=False)
            except Exception as e:
                # A genuine training error (NOT the graceful restart — that path os._exit(15)s and
                # never reaches here). Print the FULL traceback so the crash is diagnosable, save the
                # exception weights for forensics, then RE-RAISE: the old code swallowed the error and
                # fell through to the normal save + "Training complete" + final eval, masking a fatal
                # crash as a clean completion (so the launcher saw exit-0 and never auto-restarted).
                # Re-raising surfaces it as a non-zero exit → launcher restarts from the last checkpoint
                # (resilience) instead of silently ending the run with a fake final win rate.
                print(f"Training interrupted by exception: {e}")
                traceback.print_exc()
                final_path = os.path.join(model_dir, "final_model_exception")
                model.save(final_path)
                _write_latest_txt(model_dir, "final_model_exception.zip")
                raise

            # Training is over: stand the worker watchdog down BEFORE the final save/eval, so a
            # train-env worker SIGTERM'd in teardown (exitcode -15) is not an `exit 1` crash.
            _shutdown_event.set()
            final_path = os.path.join(model_dir, "final_model")
            model.save(final_path)
            _write_latest_txt(model_dir, "final_model.zip")
            save_anchor_ref_beside(model, final_path + ".zip")   # see run_io; no-op without a moving anchor
            save_model_snapshot(os.path.dirname(final_path), current_version, hparams=_model_hparams(model), cli_args=cli_args,
                                reward_composition=reward_composition, lineage=_lineage)
            print(f"Training complete. Model saved to {final_path}")
            best_model_dir = os.path.join(model_dir, "best_model")
            if os.path.isdir(best_model_dir):
                save_model_snapshot(best_model_dir, current_version, hparams=_model_hparams(model), cli_args=cli_args,
                                reward_composition=reward_composition, lineage=_lineage)
            if _run_eval and getattr(args, "final_eval", False):
                await evaluate_model_random(model)
    else:
        print(f"Starting NEW Training (Parallel x{n_envs}, Batch: {args.batch_size}, Epochs: {args.n_epochs})")
        # model_dir and unique_id are now pre-defined earlier in main()
        
        # Initialize a dummy encoder to get the handoff kwargs
        # Initialize a dummy encoder to get the layout handoff kwargs, then layer every
        # version-checked arch toggle on top via the SHARED table (agents.model.extractor_arch)
        # that the resume path above also uses.
        temp_encoder = Gen3ObservationEncoder(mappings)
        extractor_kwargs = build_extractor_arch_kwargs(
            args, base=temp_encoder.get_features_extractor_kwargs())

        policy_kwargs = {
            "features_extractor_class": Gen3FeaturesExtractor,
            "features_extractor_kwargs": extractor_kwargs,
            "net_arch": [512, 512],
            # gen3_policy_activation_pin_v1: pin the tower's nonlinearity instead of inheriting
            # sb3-contrib's signature default. Same value the default gives today (nn.Tanh), so
            # this is behaviour-neutral — see agents.model.policy.POLICY_ACTIVATION_FN for why an
            # unpinned activation is invisible to check_compatible.
            "activation_fn": POLICY_ACTIVATION_FN,
            "optimizer_class": torch.optim.AdamW,
            "optimizer_kwargs": {"weight_decay": args.weight_decay, "eps": 1e-5},
            # gen3_winprob_critic_mode_v1: 'shaped' (the default) is byte-identical to every
            # generation to date; 'winprob' routes _critic_value to sigmoid(win_head logit).
            "critic": args.critic,
        }
        
        # --- Model Initialization ---
        total_rollout_size = args.n_steps * n_envs
        if args.batch_size > total_rollout_size:
            print(f"Note: Capping batch_size from {args.batch_size} to {total_rollout_size} to match rollout capacity.")
            args.batch_size = total_rollout_size

        model = InstrumentedMaskablePPO(
            Gen3DualHeadMaskablePolicy,
            env,
            verbose=1,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=float(args.policy_gae_lambda),   # gen3_policy_gae_lambda_v1 (default 0.80)
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            device=args.device,
            seed=args.seed,
            policy_kwargs=policy_kwargs
        )

        # Every training-only hparam, from the one table shared with the resume path above.
        apply_training_hparams(model, args, mappings=mappings,
                               attach_cf_labels=_attach_cf_labels)
        model.behaviour_dump_dir = model_dir   # K9(b) row dumps (as on the resume path above)
        model._dose_kl = kl_controller_snapshot(lr_callback)   # plain data, never the callback
        model._fork_lr_pin = None          # a FRESH run cannot be pinned — `--fork-lr` is refused there
        version = ModelVersion.from_layout_and_policy_kwargs(
            extractor_kwargs["layout"], policy_kwargs, vf_coef=args.vf_coef,
            reward_config=reward_config,
            opp_belief_aux_coef=args.opp_belief_aux_coef,
            move_belief_coef=args.move_belief_coef,
            move_belief_latent_coef=args.move_belief_latent_coef,
            spread_belief_coef=args.spread_belief_coef,
            hp_type_belief_coef=args.hp_type_belief_coef,
            item_belief_coef=args.item_belief_coef,
            td_aux_coef=args.td_aux_coef,
            arch_source=getattr(args, "arch_source", None),
            policy_grad_coef=args.policy_grad_coef,
            intent_label_bot_weight=args.intent_label_bot_weight,
            win_prob_strata_weight=args.win_prob_strata_weight,
            policy_gae_lambda=args.policy_gae_lambda,
            diagnostics_every=args.diagnostics_every,
            opp_intent_coef=float(args.opp_intent_coef or 0.0),
            fork_fraction=args.fork_fraction,
            fork_branches=args.fork_branches,
            fork_contested_gap=args.fork_contested_gap,
            fork_contested_absv=args.fork_contested_absv,
            fork_max_per_battle=args.fork_max_per_battle,
            fork_crn=args.fork_crn,
            cf_records=args.cf_records,
            cf_records_keep=args.cf_records_keep,
            cf_winprob_coef=args.cf_winprob_coef,
            cf_head_only=args.cf_head_only,
            cf_label_lag_steps=args.cf_label_lag_steps,
            cf_label_likelihood=args.cf_label_likelihood,
            cf_evidential_coef=args.cf_evidential_coef,
            cf_evidential_reg=args.cf_evidential_reg,
            cf_twin_coef=args.cf_twin_coef,
            cf_shadow_coef=args.cf_shadow_coef,
            q_winprob_coef=args.q_winprob_coef,
            q_winprob_onpolicy_coef=args.q_winprob_onpolicy_coef,
            # gen3_eval_sentinel_greedy_default_v1 (v112): the EVAL REGIME + the gate it
            # derives, so a flagless resume reads its own regime back instead of crossing
            # an opponent-regime boundary. Both are RESOLVED by `resolve_config`.
            eval_sentinel_greedy=args.eval_sentinel_greedy,
            promote_threshold=args.promote_threshold,
            eval_mirrored_pairs=args.eval_mirrored_pairs,
            promotion_sprt=args.promotion_sprt,
            capacity_telemetry=args.capacity_telemetry,
            canary_reset_steps=args.canary_reset_steps,
            capacity_cosine_every=args.capacity_cosine_every,
            capacity_velocity_every=args.capacity_velocity_every,
            distill_target=args.distill_target,
            distill_topk=args.distill_topk,
            distill_gate=args.distill_gate,
            distill_gate_tau=args.distill_gate_tau,
            distill_beta=args.distill_beta,
            rank_tripwire=args.rank_tripwire,
            rank_tripwire_drop=args.rank_tripwire_drop,
            teacher_scan_limit=args.teacher_scan_limit,
        )
        # (A `PBRS_GAMMA == model.gamma` assert lived here while the reward folded hand potentials;
        # it went with them in the shaped-reward deletion, 2026-09-26.)
        _ledger = _cuda_ledger.start(model.device)   # gen3_cuda_ledger_v1: where the card goes
        _start_rust_env(env, model)   # M5 Lane G: BEFORE the trainer's compile step
        _ledger.mark("rust env core (T2 slots, staging, arena; python core: none)")
        _maybe_compile_trainer(model, args)
        _run_roundtrip_test(model, extractor_kwargs["layout"], policy_kwargs, debug=args.debug)
        _apply_grad_checkpointing(model, args.grad_checkpointing)
        _arm_compile_sentinel(model, args)   # gen3_compile_sentinel_v1: reset, prewarm, lock
        _ledger.mark("compiled regions: gate + prewarm + lock")
        _arm_learner_lifecycle(model, args)  # K6 gen3_learner_freeze_v1: declare, then freeze guard
        _ledger.mark("optimizer state declared (Adam m, v) + lifecycle")
        _ledger.report(model_dir)
        # gen3_update_fit_v1: RUN one dry update (restored exactly) and refuse a first update that
        # would not fit with the declared headroom — at startup, never an OOM at update 1.
        _fit = _update_fit.check_update_fits(model, model_dir)
        if _fit:
            print(_fit, flush=True)
        model._async_rollout = _async_rollout   # route collect_rollouts to the non-barrier path
        # gen3_run_lineage_v1 — a FRESH run states the explicit null form (`fork_parent: null,
        # role: "fresh"`), because "no block" and "no parent" are different facts.
        _lineage = _run_lineage(args, model_dir, model_path=None,
                                fork_step=int(getattr(model, "num_timesteps", 0) or 0))
        save_model_snapshot(model_dir, version, hparams=_model_hparams(model), cli_args=cli_args,
                                reward_composition=reward_composition, lineage=_lineage)

        _abort_fn = _setup_signal_handlers(
            model, model_dir, _shutdown_event, version,
            lambda: model.policy.optimizer.param_groups[0]["lr"],
            lambda: model.n_epochs,
            handoff_lr_fn=(
                (lambda: lr_callback.handoff_lr)
                if isinstance(lr_callback, TwoPhaseLRCallback) else None
            ),
            eval_drain_fn=(
                (lambda: eval_callback.drain(timeout=_ABORT_EVAL_DRAIN_SEC))
                if (eval_callback is not None and hasattr(eval_callback, "drain")) else None
            ),
        )
        if eval_callback is not None:
            eval_callback.abort_fn = _abort_fn
        graceful_restart_callback.abort_fn = _abort_fn

        # Seed the pool from these weights iff self-play is active and the pool is empty
        # (no env rebuild — workers re-scan the dir on demand). Then start the worker watchdog.
        _maybe_seed_pool(model)
        start_subprocess_watchdog(env, label="train_env", shutdown_event=_shutdown_event)

        _attach_run_tb_logger(model, model_dir)  # TB → <model_dir>/tb/
        try:
            if log_level >= LogLevel.DETAILED:
                from sb3_contrib.common.maskable.utils import is_masking_supported
                print(f"✅ [DEBUG] Masking supported for env: {is_masking_supported(env)}")
            # FRESH run: `num_timesteps` is 0, so SB3's `total_timesteps += num_timesteps` is a
            # no-op and the absolute target is correct here. The RESUME site must pass the REMAINING
            # budget instead — see the note there.
            model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=False)
        except Exception as e:
            print("\n" + "🛑" * 30)
            print(f"🛑 TRAINING CRASHED: {e}")
            print("🛑" * 30)
            traceback.print_exc()
            # Stop immediately, do not proceed to evaluation. CRASH (1) unless the error is one a
            # restart would REPLAY (`exit_codes.exit_code_for`: a non-finite learner → 4, a starved
            # external supply → 5), exactly as the entry point's fail-fast handlers map it — a
            # literal 1 here turned every FATAL raised inside a FRESH run's learn() into a restart.
            os._exit(exit_code_for(e))

        # Training is over: stand the worker watchdog down (see the resume path's note).
        _shutdown_event.set()
        final_path = os.path.join(model_dir, "final_model")
        model.save(final_path)
        _write_latest_txt(model_dir, "final_model.zip")
        _final_handoff = lr_callback.handoff_lr if isinstance(lr_callback, TwoPhaseLRCallback) else None
        record_checkpoint(model_dir, final_path + ".zip", adaptive_ppo_callback.current_lr, model.n_epochs, hparams=_model_hparams(model), handoff_lr=_final_handoff)
        save_anchor_ref_beside(model, final_path + ".zip")   # see run_io; no-op without a moving anchor
        save_model_snapshot(os.path.dirname(final_path), version, hparams=_model_hparams(model), cli_args=cli_args,
                                reward_composition=reward_composition, lineage=_lineage)
        print(f"Training complete. Model saved to {final_path}")
        best_model_dir = os.path.join(model_dir, "best_model")
        if os.path.isdir(best_model_dir):
            save_model_snapshot(best_model_dir, version, hparams=_model_hparams(model), cli_args=cli_args,
                                reward_composition=reward_composition, lineage=_lineage)
        if _run_eval and getattr(args, "final_eval", False):
            await evaluate_model_random(model)
