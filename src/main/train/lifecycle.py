"""Run LIFECYCLE: the things done to a live model once, around `learn()`.

Grad checkpointing, the trainer compile, the save/reload round-trip smoke test, and the signal
handlers (SIGINT/SIGTERM/SIGHUP checkpoint-and-exit and SIGUSR1 forced checkpoint, each at the next
SAFE POINT — `deferred_abort` —, SIGUSR2 forced eval) that turn a kill into a clean, checkpoint-saving
shutdown.
"""
import os
import signal
import sys
from datetime import datetime

from agents.model.extra_obs_keys import synthetic_obs
from agents.model.model_version import ModelVersion
from agents.model.snapshot import load_model_snapshot, record_checkpoint, save_model_snapshot
from agents.training.eval_callback import request_forced_eval
from main.exit_codes import TrainExitCode
from main.launcher.ipc import send_event
from main.train.deferred_abort import DeferredAbort
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
    (the ARGV's `--model`, so `is_same_run_checkpoint` judges the checkpoint the user named, not a derived
    path)."""
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
            # Decidable at startup, so decide it at startup: an update that does not divide into full
            # micro-batches would hand the one compiled learner graph a second, undeclared shape (the
            # collector refuses it at the first buffer build; this refuses it before the model
            # compiles). The REAL update size: the model's own n_steps / batch_size (a resume restores
            # the checkpoint's, so the argv's are inert there) with the argv's target
            # (P10-E, F9 — it used to judge n_steps * n_envs, which the target overrides).
            from main.train.compile_flags import update_rows_for
            check_shape_stability(
                update_rows=update_rows_for(
                    rollout_target_samples=getattr(args, "rollout_target_samples", 0),
                    n_steps=int(getattr(model, "n_steps", None) or getattr(args, "n_steps", 0) or 0),
                    n_envs=int(getattr(model, "n_envs", None) or getattr(args, "n_envs", 0) or 0)),
                batch_size=int(getattr(model, "batch_size", None) or getattr(args, "batch_size", 0) or 0),
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
    reset dynamo, install + gate the DECLARED REGION (R1, the micro-step), prewarm its declared
    signature, lock, and attach the per-rollout / per-update checks.

    Placed AFTER `_apply_grad_checkpointing` (the forward reads that attribute, so anything compiled
    before it is a stale cache entry) and before `learn()`. No-op when the learner is not compiled.
    """
    from agents.model.compile_trainer import CompileTrainerError, arm_compile_sentinel
    if not getattr(args, "compile_trainer", False):
        return
    try:
        arm_compile_sentinel(model, batch_size=int(model.batch_size), emit=send_event)
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
        policy_grad_coef=float(getattr(model, "policy_grad_coef", 1.0)),
        intent_label_bot_weight=float(getattr(model, "intent_label_bot_weight", 1.0)),
        policy_gae_lambda=float(model.gae_lambda),
        diagnostics_every=int(getattr(model, "diagnostics_every", 1)),
        opp_intent_coef=float(getattr(model, "opp_intent_coef", 0.0) or 0.0),
        fork_fraction=float(getattr(model, "fork_fraction", 0.0)),
        fork_branches=int(getattr(model, "fork_branches", 3)),
        fork_contested_gap=float(getattr(model, "fork_contested_gap", 0.4)),
        fork_contested_absv=float(getattr(model, "fork_contested_absv", 0.0)),
        fork_max_per_battle=int(getattr(model, "fork_max_per_battle", 1)),
        fork_crn=str(getattr(model, "fork_crn", 'dice_and_draws')),
        capacity_telemetry=bool(getattr(model, "capacity_telemetry", False)),
        canary_reset_steps=int(getattr(model, "canary_reset_steps", 1_000_000)),
        capacity_cosine_every=int(getattr(model, "capacity_cosine_every", 50)),
        capacity_velocity_every=int(getattr(model, "capacity_velocity_every", 50)),
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
        # extractor's forward reads (none today; a missing one RAISES — a silent skip is the
        # gen-12 dead-tail bug). Built from the DECLARED
        # registry rather than by hand: `agents.model.extra_obs_keys`, whose docstring records the
        # launch the hand-built version cost.
        dummy_obs = synthetic_obs(reloaded.policy.features_extractor, total_dim,
                                  device=dev, action_mask=True)
        with torch.no_grad():
            pi_features, vf_features = reloaded.policy.features_extractor(dummy_obs)
        # gen3_policy_readout_trunk_v1 (audit F2): the policy-feature width is the extractor's to state —
        # PROJECTION_DIM under `--policy-readout tower`, D_MODEL under `trunk` (the state query's read).
        pi_dim = int(getattr(reloaded.policy.features_extractor, "policy_ctx_dim", PROJECTION_DIM))
        assert pi_features.shape == (1, pi_dim), (
            f"Round-trip test: unexpected policy-feature shape {pi_features.shape}, expected (1, {pi_dim})"
        )
        # The value half is `value_pooled` itself (no projection since the version break, audit F1).
        vf_dim = int(reloaded.policy.features_extractor.vf_features_dim)
        assert vf_features.shape == (1, vf_dim), (
            f"Round-trip test: unexpected value-feature shape {vf_features.shape}, expected (1, {vf_dim})"
        )
        if debug:
            print(f"[ModelVersion] Round-trip smoke test PASSED (pi shape: {tuple(pi_features.shape)}, "
                  f"vf shape: {tuple(vf_features.shape)})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _setup_signal_handlers(model, model_dir, shutdown_event, version, current_lr_fn, current_epochs_fn,
                           handoff_lr_fn=None, *, exit_fn=os._exit, wait_fn=None,
                           start_watchdog=True):
    """Wire SIGINT/SIGTERM/SIGHUP/SIGUSR1/SIGUSR2. Returns the run's `DeferredAbort`
    (`main.train.deferred_abort`, gen3_deferred_abort_v1): CALLED with a reason it is the canonical
    "save a checkpoint and exit 15" path (the graceful restart's), and its `safe_point()` runs a
    signal-requested abort. Both run ONLY on the main thread at a safe point — the stop signals just
    record the request (P10 review F1: the old in-handler dump deadlocked on TensorBoard's lock, and
    the in-handler save could tear a checkpoint mid-update). SIGUSR1's forced checkpoint is recorded the
    same way and saved at the next safe point, without an exit (P10-A2).

    ``handoff_lr_fn`` is optional; when present it returns the TwoPhaseLR
    callback's current handoff_lr (or None while still in Phase 1) so the
    cosine starting LR is persisted alongside the SIGTERM checkpoint.

    ``exit_fn`` / ``wait_fn`` / ``start_watchdog`` are the test seams (`DeferredAbort`'s).
    """

    def _handoff() -> "float | None":
        return handoff_lr_fn() if handoff_lr_fn is not None else None

    def _commit_abort(reason: str) -> None:
        """The abort's body — at a safe point, on the main thread (`DeferredAbort.abort` claims the exit
        first, then exits 15 after this returns).

        Saves a full checkpoint (with metadata + latest.txt). There is no eval to drain: the eval cycle is
        blocking and in process, and a stop signal inside it abandons the partial cycle at its safe point.
        """
        shutdown_event.set()
        print(f"\n[ABORT] {reason}", flush=True)
        # The pending scalars (P3, gen3_final_update_dump_v1): the loop dumps BEFORE each update, so the
        # last update's `train/*` are still pending — this IS that iteration's dump, at the step the
        # rollout reached (`OwnedLoop.dump_logs`). Safe now: a safe point is never inside a dump.
        try:
            model.dump_logs()
        except Exception as e:
            print(f"[ABORT] pending-scalar dump failed: {e}")
        try:
            path = os.path.join(model_dir, "final_model_interrupted")
            model.save(path)
            _write_latest_txt(model_dir, "final_model_interrupted.zip")
            lr = current_lr_fn()
            epochs = current_epochs_fn()
            hparams = _model_hparams(model)
            save_model_snapshot(model_dir, version, current_lr=lr, current_epochs=epochs, hparams=hparams)
            record_checkpoint(model_dir, path + ".zip", lr, epochs, hparams=hparams, handoff_lr=_handoff())
            print(f"[ABORT] Checkpoint saved → {path}.zip")
        except Exception as e:
            print(f"[ABORT] Save failed: {e}")
        sys.stdout.flush()

    def _forced_checkpoint() -> None:
        """SIGUSR1's save — at a safe point, on the main thread (`DeferredAbort.request_checkpoint`,
        gen3_deferred_checkpoint_v1); training continues after it, and a failed save is REPORTED, never
        raised into the loop (an operator's key must not end the run)."""
        step = model.num_timesteps
        name = f"checkpoint_forced_{step:010d}_{datetime.now().strftime('%H%M%S')}"
        # Forced checkpoints are resumable checkpoints → they live under checkpoints/
        # alongside the periodic ones; latest.txt records the run-relative path.
        ckpt_dir = os.path.join(model_dir, "checkpoints")
        ckpt = os.path.join(ckpt_dir, name)
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
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
        except Exception as e:
            print(f"\n💾 [CHECKPOINT] Forced save FAILED ({type(e).__name__}: {e}) — training continues",
                  flush=True)
            return
        print(f"\n💾 [CHECKPOINT] Forced save → {ckpt}.zip", flush=True)

    abort = DeferredAbort(_commit_abort, checkpoint=_forced_checkpoint, exit_fn=exit_fn, wait_fn=wait_fn)

    def _forced_eval(sig, frame):
        # Signal context: just flag the request (request_forced_eval is async-signal-safe).
        # The active eval callback picks it up on its next _on_step — and REJECTS it if a
        # cycle is already running. Driven by the launcher's "force eval" button (SIGUSR2).
        request_forced_eval()

    # The stop signals only RECORD the request (`DeferredAbort.request`: attribute stores + raw
    # `os.write`s); the main thread runs the abort at its next safe point (gen3_deferred_abort_v1).
    signal.signal(signal.SIGINT,  lambda sig, frame: abort.request("SIGINT received"))
    signal.signal(signal.SIGTERM, lambda sig, frame: abort.request("SIGTERM received"))
    # SIGHUP = the controlling terminal/window closed. The launcher spawns the child in the
    # SAME session (no start_new_session), so closing the tmux window SIGHUPs the whole group;
    # without this handler the child died mid-iteration with NO checkpoint (lost ~1h once).
    # Route it to the same graceful checkpoint-then-INTERRUPTED path as SIGTERM so an accidental
    # window close costs nothing. (Running the launcher under `nohup` also prevents the SIGHUP;
    # this is the in-code backstop for when it isn't.)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, lambda sig, frame: abort.request("SIGHUP received (terminal/window closed)"))
    # SIGUSR1 = the launcher's "forced checkpoint" key — RECORDED, saved at the next safe point (P10-A2:
    # the in-handler save could tear a checkpoint mid-update, exactly as the abort's could).
    signal.signal(signal.SIGUSR1, lambda sig, frame: abort.request_checkpoint("SIGUSR1 received"))
    # SIGUSR2 = the launcher's "force eval" button — run an off-cadence eval cycle now.
    signal.signal(signal.SIGUSR2, _forced_eval)
    if start_watchdog:
        abort.start()
    return abort
