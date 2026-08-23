"""Run LIFECYCLE: the things done to a live model once, around `learn()`.

Grad checkpointing, the trainer compile, the save/reload round-trip smoke test, and the signal
handlers (SIGINT/SIGTERM/SIGHUP checkpoint-and-exit, SIGUSR1 forced checkpoint, SIGUSR2 forced
eval) that turn a kill into a clean, checkpoint-saving shutdown.
"""
import os
import signal
import sys
from datetime import datetime

from agents.model.model_version import ModelVersion
from agents.model.snapshot import load_model_snapshot, record_checkpoint, save_model_snapshot
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


def _maybe_compile_trainer(model, args) -> None:
    """Apply `--compile-trainer` to the LEARNER, or die trying (see `agents.model.compile_trainer`).

    Placed BEFORE `_run_roundtrip_test` on purpose: that test is a save -> reload -> forward, so
    running it after the compile turns it into a free gate on the one thing that would silently
    corrupt every checkpoint of the run — a compiled callable leaking into the saved state_dict.
    """
    from agents.model.compile_trainer import (CompileTrainerError, check_shape_stability,
                                               compile_trainer_extractor)
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
        compile_trainer_extractor(model, getattr(args, "compile_trainer", False),
                                  emit=send_event)
    except CompileTrainerError as exc:
        print(f"\n[CompileTrainer] FATAL: {exc}", file=sys.stderr, flush=True)
        send_event(f"[CompileTrainer] FATAL: {exc}")   # stderr above; this is the launcher panel
        sys.exit(TrainExitCode.FATAL_CONFIG)


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
        value_tail_weight=float(getattr(model, "value_tail_weight", 0.0)),
        opp_belief_aux_coef=float(getattr(model, "opp_belief_aux_coef", 0.0)),
        move_belief_coef=float(getattr(model, "move_belief_coef", 0.0)),
        win_prob_coef=float(getattr(model, "win_prob_coef", 1.0)),
        move_belief_latent_coef=float(getattr(model, "move_belief_latent_coef", 0.0)),
        spread_belief_coef=float(getattr(model, "spread_belief_coef", 0.0)),
        value_dist_coef=float(getattr(model, "value_dist_coef", 1.0)),
        td_aux_coef=float(getattr(model, "td_aux_coef", 0.0)),
        intent_label_bot_weight=float(getattr(model, "intent_label_bot_weight", 1.0)),
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
        dummy_obs = {
            "observation": torch.zeros(1, total_dim, device=dev),
            "action_mask": torch.ones(1, 11, dtype=torch.int8, device=dev),
        }
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
