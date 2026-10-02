"""Run LIFECYCLE: the things done to a live model once, around `learn()`.

Grad checkpointing, the trainer compile, the save/reload round-trip smoke test, and the signal
handlers (SIGINT/SIGTERM/SIGHUP checkpoint-and-exit, SIGUSR1 forced checkpoint, SIGUSR2 forced
eval) that turn a kill into a clean, checkpoint-saving shutdown.
"""
import os
import signal
import sys
from datetime import datetime

from agents.model.extra_obs_keys import synthetic_obs
from agents.model.model_version import ModelVersion
from agents.model.snapshot import load_model_snapshot, record_checkpoint, save_model_snapshot
from agents.training.distill_anchor_callback import save_anchor_ref_beside
from agents.training.eval_callback import request_forced_eval
from main.exit_codes import TrainExitCode
from main.launcher.ipc import send_event
from main.train.run_io import _model_hparams, _write_latest_txt


def _apply_grad_checkpointing(model, enabled: bool) -> None:
    """Toggle gradient checkpointing on the live model's transformer body.

    Runtime-only and bit-exact (dropout=0 + use_reentrant=False): it never enters the
    saved checkpoint or the version check, so it is set fresh each run from
    ``--grad-checkpointing`` regardless of what a resumed checkpoint was trained with.
    Trades one extra transformer forward in the backward pass (on the otherwise-idle GPU)
    for ~5GB less activation VRAM. A no-op under inference (no_grad).
    """
    if not enabled:
        return
    from agents.model.features_extractor import TeamTransformer, DamageOperator
    n = 0
    for module in model.policy.modules():
        if isinstance(module, (TeamTransformer, DamageOperator)):
            module.grad_checkpointing = True
            n += 1
    print(f"[GradCheckpoint] enabled on {n} transformer block(s) "
          f"(bit-exact; trades idle-GPU compute for ~5GB activation VRAM)")


def _declare_compile_cache(args, model_dir: str) -> None:
    """K3 (gen3_hermetic_compile_cache_v1): declare `<run>/compile_cache/` as this process tree's
    ONLY compile cache — EMPTY at a fresh launch or a fork, reused only on this run's own restart
    when its stamp (code commit + torch + compile-config row) matches. Must run the moment the run
    dir exists: before any compile, env worker, forkserver, eval subprocess or T2 service, every one
    of which inherits the three environment variables it sets. `args.model` here is the ARGV's
    (before a consensus warm-start re-points it at `<run>/warmstart/`, which is an init, not this
    run's progress — `is_same_run_checkpoint` says so)."""
    from agents.model.compile_cache import CompileCacheError, prepare_run_cache
    from main.launcher.ipc import emit
    from main.train.fork_lr import is_same_run_checkpoint
    try:
        prepare_run_cache(model_dir,
                          same_run_resume=is_same_run_checkpoint(getattr(args, "model", None) or "",
                                                                 model_dir),
                          emit=emit)
    except CompileCacheError as exc:
        print(f"\n[CompileCache] FATAL: {exc}", file=sys.stderr, flush=True)
        send_event(f"[CompileCache] FATAL: {exc}")
        sys.exit(TrainExitCode.FATAL_CONFIG)


def _maybe_compile_trainer(model, args) -> None:
    """Apply `--compile-trainer` to the LEARNER, or die trying (see `agents.model.compile_trainer`).

    Placed BEFORE `_run_roundtrip_test` on purpose: that test is a save -> reload -> forward, so
    running it after the compile turns it into a free gate on the one thing that would silently
    corrupt every checkpoint of the run — a compiled callable leaking into the saved state_dict.
    """
    from agents.model.compile_trainer import (CompileTrainerError, check_shape_stability,
                                               preflight_compile_trainer)
    try:
        if getattr(args, "compile_trainer", False):
            # Decidable at startup, so decide it at startup: a config that would feed the compiled
            # extractor an unbounded set of batch shapes ends in a SILENT eager fallback.
            check_shape_stability(
                n_steps=int(getattr(args, "n_steps", 0) or 0),
                n_envs=int(getattr(args, "n_envs", 0) or 0),
                batch_size=int(getattr(args, "batch_size", 0) or 0),
                async_rollout=bool(getattr(args, "async_rollout", False)),
            )
        # `send_event`, NOT `emit`: emit() falls back to print() when there is no launcher pipe, and
        # compile_trainer already prints to stdout — so passing emit duplicated every line in a
        # standalone run. send_event is event-only, so the launcher panel still gets it and a
        # standalone run says it once.
        preflight_compile_trainer(model, getattr(args, "compile_trainer", False), emit=send_event)
    except CompileTrainerError as exc:
        print(f"\n[CompileTrainer] FATAL: {exc}", file=sys.stderr, flush=True)
        send_event(f"[CompileTrainer] FATAL: {exc}")   # stderr above; this is the launcher panel
        sys.exit(TrainExitCode.FATAL_CONFIG)


def _arm_compile_sentinel(model, args) -> None:
    """gen3_compile_sentinel_v1 — `agents.model.compile_control`'s phases for a compiled learner:
    reset dynamo, install + gate the DECLARED REGIONS (R0, R1), prewarm every declared signature,
    lock, and attach the per-rollout / per-update checks.

    Placed AFTER `_apply_grad_checkpointing` (the forward reads that attribute, so anything compiled
    before it is a stale cache entry) and before `learn()`. No-op when the learner is not compiled.
    """
    from agents.model.compile_trainer import CompileTrainerError, arm_compile_sentinel
    if not getattr(args, "compile_trainer", False):
        return
    if getattr(args, "debug", False) and getattr(args, "compile_opponents", False):
        # --debug is ONE DummyVecEnv: the opponents live IN THIS PROCESS, and each pool snapshot
        # compiles `Gen3FeaturesExtractor.forward` lazily — the SAME code objects (the SAME
        # cache_size_limit slots) as the learner, and after the lock. Refuse at startup rather than
        # die at the first snapshot load with a sentinel FATAL that names the learner's frame.
        msg = ("[CompileSentinel] FATAL: --debug with --compile-trainer AND --compile-opponents "
               "compiles opponents inside the learner process, on the learner's own dynamo code "
               "objects, after the compile lock. Pass --no-compile-opponents with --debug "
               "--compile-trainer (production's SubprocVecEnv compiles opponents in the workers).")
        print(f"\n{msg}", file=sys.stderr, flush=True)
        send_event(msg)
        sys.exit(TrainExitCode.FATAL_CONFIG)
    try:
        arm_compile_sentinel(model, n_envs=int(getattr(model, "n_envs", 0) or args.n_envs),
                             batch_size=int(model.batch_size), emit=send_event)
    except CompileTrainerError as exc:
        print(f"\n[CompileSentinel] FATAL: {exc}", file=sys.stderr, flush=True)
        send_event(f"[CompileSentinel] FATAL: {exc}")
        sys.exit(TrainExitCode.FATAL_CONFIG)


def _arm_learner_lifecycle(model, args) -> None:
    """K6 (gen3_learner_freeze_v1) — the learner's DECLARED LIFECYCLE: the last startup step.

    Declares what the steady state would otherwise acquire lazily (every Adam/AdamW optimizer's
    state; the capacity canary under `--capacity-telemetry`), then attaches the FREEZE GUARD
    (`agents.training.learner_lifecycle`): it freezes the learner's object graph on entry to the
    first rollout and fails any new optimizer / parameter / module / buffer / optimizer-state entry
    after it with a typed `LazyAcquisitionError` (FATAL_CONFIG, not restarted). Always on — compiled
    or not, CPU or CUDA: it costs one identity walk per update. Placed AFTER `_arm_compile_sentinel`
    so its wrappers are the outermost (the freeze precedes the compile sentinel's own first-rollout
    work, and its check follows the sentinel's)."""
    from agents.training.learner_lifecycle import (LazyAcquisitionError, attach,
                                                   declare_learner_startup)
    try:
        declare_learner_startup(model, emit=send_event)
    except LazyAcquisitionError as exc:
        print(f"\n{exc}", file=sys.stderr, flush=True)
        send_event(str(exc).splitlines()[0][:500])
        sys.exit(TrainExitCode.FATAL_CONFIG)
    attach(model, emit=send_event)


def _run_roundtrip_test(model, layout: dict, policy_kwargs: dict, debug: bool = False) -> None:
    """Startup smoke test: save → reload → zero forward pass → assert output shape.

    Catches serialization failures at second 5, not hour 50. Raises on any failure.
    """
    import shutil
    import tempfile
    import torch
    from agents.model.features_extractor import PROJECTION_DIM

    version = ModelVersion.from_layout_and_policy_kwargs(
        layout, policy_kwargs, vf_coef=float(model.vf_coef),
        opp_belief_aux_coef=float(getattr(model, "opp_belief_aux_coef", 0.0)),
        move_belief_coef=float(getattr(model, "move_belief_coef", 0.0)),
        move_belief_latent_coef=float(getattr(model, "move_belief_latent_coef", 0.0)),
        spread_belief_coef=float(getattr(model, "spread_belief_coef", 0.0)),
        td_aux_coef=float(getattr(model, "td_aux_coef", 0.0)),
        policy_grad_coef=float(getattr(model, "policy_grad_coef", 1.0)),
        intent_label_bot_weight=float(getattr(model, "intent_label_bot_weight", 1.0)),
        win_prob_strata_weight=float(getattr(model, "win_prob_strata_weight", 0.0)),
        win_prob_lambda=float(getattr(model, "win_prob_lambda", 1.0)),
        policy_gae_lambda=float(model.gae_lambda),
        diagnostics_every=int(getattr(model, "diagnostics_every", 1)),
        opp_intent_coef=float(getattr(model, "opp_intent_coef", 0.0) or 0.0),
        win_prob_lambda_truncated=str(getattr(model, "win_prob_lambda_truncated", "bootstrap")),
        win_prob_rollout_target=float(getattr(model, "win_prob_rollout_target", 0.0)),
        win_prob_rollout_r=int(getattr(model, "win_prob_rollout_r", 8)),
        win_prob_rollout_mode=str(getattr(model, "win_prob_rollout_mode", "replace")),
        win_prob_rollout_weight=float(getattr(model, "win_prob_rollout_weight", 1.0)),
        fork_fraction=float(getattr(model, "fork_fraction", 0.0)),
        fork_branches=int(getattr(model, "fork_branches", 3)),
        fork_contested_gap=float(getattr(model, "fork_contested_gap", 0.4)),
        fork_contested_absv=float(getattr(model, "fork_contested_absv", 0.0)),
        fork_max_per_battle=int(getattr(model, "fork_max_per_battle", 1)),
        fork_crn=str(getattr(model, "fork_crn", 'dice_and_draws')),
        win_prob_dense_aux=float(getattr(model, "win_prob_dense_aux", 0.0)),
        cf_records=bool(getattr(model, "cf_records", False)),
        cf_records_keep=int(getattr(model, "cf_records_keep", 512)),
        cf_winprob_coef=float(getattr(model, "cf_winprob_coef", 0.0)),
        cf_head_only=bool(getattr(model, "cf_head_only", True)),
        cf_label_lag_steps=int(getattr(model, "cf_label_lag_steps", 150_000)),
        cf_label_likelihood=str(getattr(model, "cf_label_likelihood", "binomial")),
        cf_evidential_coef=float(getattr(model, "cf_evidential_coef", 0.0)),
        cf_evidential_reg=float(getattr(model, "cf_evidential_reg", 1e-3)),
        cf_twin_coef=float(getattr(model, "cf_twin_coef", 0.0)),
        cf_shadow_coef=float(getattr(model, "cf_shadow_coef", 0.0)),
        q_winprob_coef=float(getattr(model, "q_winprob_coef", 0.0)),
        q_winprob_onpolicy_coef=float(getattr(model, "q_winprob_onpolicy_coef", 0.0)),
        capacity_telemetry=bool(getattr(model, "capacity_telemetry", False)),
        canary_reset_steps=int(getattr(model, "canary_reset_steps", 1_000_000)),
        capacity_cosine_every=int(getattr(model, "capacity_cosine_every", 50)),
        capacity_velocity_every=int(getattr(model, "capacity_velocity_every", 50)),
        # gen3_distill_target_gate_v1 (v103): the five loss knobs live on the model; the two
        # rank-tripwire knobs are callback config (not model attrs) and default here — this
        # version only feeds the round-trip smoke, and neither is gated.
        distill_target=str(getattr(model, "distill_target", "kl")),
        distill_topk=int(getattr(model, "distill_topk", 1)),
        distill_gate=str(getattr(model, "distill_gate", "none")),
        distill_gate_tau=float(getattr(model, "distill_gate_tau", 0.0)),
        distill_beta=float(getattr(model, "distill_beta", 1.0)),
    )
    total_dim = layout["total_dim"]
    tmpdir = tempfile.mkdtemp(prefix="roundtrip_")
    try:
        zip_path = os.path.join(tmpdir, "roundtrip_model")
        model.save(zip_path)
        save_model_snapshot(tmpdir, version, git_hash="roundtrip-test")
        reloaded = load_model_snapshot(
            zip_path + ".zip",
            env=model.get_env(),
            current_version=version,
            device=str(model.device),
        )
        dev = next(reloaded.policy.parameters()).device
        # The round-trip smoke builds its OWN obs dict, so it owes every flag-gated Dict key this
        # extractor's forward reads — the privileged value route RAISES on a missing
        # `opp_true_team` (a silent skip is the gen-12 dead-tail bug). Built from the DECLARED
        # registry rather than by hand: `agents.model.extra_obs_keys`, whose docstring records the
        # launch the hand-built version cost.
        dummy_obs = synthetic_obs(reloaded.policy.features_extractor, total_dim,
                                  device=dev, action_mask=True)
        with torch.no_grad():
            pi_features, vf_features = reloaded.policy.features_extractor(dummy_obs)
        assert pi_features.shape == (1, PROJECTION_DIM), (
            f"Round-trip test: unexpected policy-feature shape {pi_features.shape}, expected (1, {PROJECTION_DIM})"
        )
        assert vf_features.shape == (1, PROJECTION_DIM), (
            f"Round-trip test: unexpected value-feature shape {vf_features.shape}, expected (1, {PROJECTION_DIM})"
        )
        if debug:
            print(f"[ModelVersion] Round-trip smoke test PASSED (pi+vf shape: {tuple(pi_features.shape)})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _setup_signal_handlers(model, model_dir, shutdown_event, version, current_lr_fn, current_epochs_fn, handoff_lr_fn=None, eval_drain_fn=None):
    """Wire SIGINT/SIGTERM/SIGHUP/SIGUSR1/SIGUSR2. Returns the abort_training closure so it
    can be passed to eval callbacks as their canonical "die cleanly" path.

    ``handoff_lr_fn`` is optional; when present it returns the TwoPhaseLR
    callback's current handoff_lr (or None while still in Phase 1) so the
    cosine starting LR is persisted alongside the SIGTERM checkpoint.

    ``eval_drain_fn`` is optional; when present it is called AFTER the checkpoint
    is safely saved to wait (briefly, bounded) for an in-flight subprocess eval so
    its results land before exit. Bounded so the child still exits inside the
    launcher's SIGKILL grace — the checkpoint is already safe regardless.
    """

    def _handoff() -> "float | None":
        return handoff_lr_fn() if handoff_lr_fn is not None else None

    def abort_training(reason: str) -> None:
        """Single canonical abort path — works from any thread.

        Saves a full checkpoint (with metadata + latest.txt), then exits with
        TrainExitCode.INTERRUPTED (15) so the launcher restarts the run.
        Uses os._exit() rather than sys.exit() so it terminates the whole
        process even when called from a background eval thread.
        """
        shutdown_event.set()
        print(f"\n[ABORT] {reason}")
        try:
            path = os.path.join(model_dir, "final_model_interrupted")
            model.save(path)
            _write_latest_txt(model_dir, "final_model_interrupted.zip")
            lr = current_lr_fn()
            epochs = current_epochs_fn()
            hparams = _model_hparams(model)
            save_model_snapshot(model_dir, version, current_lr=lr, current_epochs=epochs, hparams=hparams)
            record_checkpoint(model_dir, path + ".zip", lr, epochs, hparams=hparams, handoff_lr=_handoff())
            save_anchor_ref_beside(model, path + ".zip")   # see run_io; no-op without a moving anchor
            print(f"[ABORT] Checkpoint saved → {path}.zip")
        except Exception as e:
            print(f"[ABORT] Save failed: {e}")
        # Checkpoint is safe; now wait for any in-flight eval to FINISH so its results
        # land in metadata.json before we exit (bounded by _ABORT_EVAL_DRAIN_SEC, which
        # fits inside the scheduled-restart grace window).
        if eval_drain_fn is not None:
            try:
                eval_drain_fn()
            except Exception as e:
                print(f"[ABORT] eval drain failed: {e}")
        os._exit(int(TrainExitCode.INTERRUPTED))

    def _forced_checkpoint(sig, frame):
        step = model.num_timesteps
        name = f"checkpoint_forced_{step:010d}_{datetime.now().strftime('%H%M%S')}"
        # Forced checkpoints are resumable checkpoints → they live under checkpoints/
        # alongside the periodic ones; latest.txt records the run-relative path.
        ckpt_dir = os.path.join(model_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = os.path.join(ckpt_dir, name)
        model.save(ckpt)
        _write_latest_txt(model_dir, os.path.join("checkpoints", name + ".zip"))
        record_checkpoint(
            model_dir,
            ckpt + ".zip",
            current_lr_fn(),
            current_epochs_fn(),
            hparams=_model_hparams(model),
            handoff_lr=_handoff(),
        )
        save_anchor_ref_beside(model, ckpt + ".zip")       # see run_io; no-op without a moving anchor
        print(f"\n💾 [CHECKPOINT] Forced save → {ckpt}.zip")

    def _forced_eval(sig, frame):
        # Signal context: just flag the request (request_forced_eval is async-signal-safe).
        # The active eval callback picks it up on its next _on_step — and REJECTS it if a
        # cycle is already running. Driven by the launcher's "force eval" button (SIGUSR2).
        request_forced_eval()

    signal.signal(signal.SIGINT,  lambda sig, frame: abort_training("SIGINT received"))
    signal.signal(signal.SIGTERM, lambda sig, frame: abort_training("SIGTERM received"))
    # SIGHUP = the controlling terminal/window closed. The launcher spawns the child in the
    # SAME session (no start_new_session), so closing the tmux window SIGHUPs the whole group;
    # without this handler the child died mid-iteration with NO checkpoint (lost ~1h once).
    # Route it to the same graceful checkpoint-then-INTERRUPTED path as SIGTERM so an accidental
    # window close costs nothing. (Running the launcher under `nohup` also prevents the SIGHUP;
    # this is the in-code backstop for when it isn't.)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, lambda sig, frame: abort_training("SIGHUP received (terminal/window closed)"))
    signal.signal(signal.SIGUSR1, _forced_checkpoint)
    # SIGUSR2 = the launcher's "force eval" button — run an off-cadence eval cycle now.
    signal.signal(signal.SIGUSR2, _forced_eval)
    return abort_training
