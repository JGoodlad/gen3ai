import multiprocessing
import traceback
import functools
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Hardened path injection for worker reliability
import sys
import os
script_path = os.path.abspath(__file__)
main_dir = os.path.dirname(script_path)
src_dir = os.path.dirname(main_dir)
root_dir = os.path.dirname(src_dir)
for d in [root_dir, src_dir, main_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

import asyncio
import json
import random
import argparse
import signal
import threading
import torch
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.model.model_version import ModelVersion, ModelVersionError
from agents.model.snapshot import save_model_snapshot, load_model_snapshot, read_checkpoint_metadata, record_checkpoint
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.inference.player import RLPlayer
from utils.teambuilder import Gen3Teambuilder
from utils.team_loader import TeamLoader
from agents.training.eval_callback import (
    PerOpponentEvalCallback, opponent_name, request_forced_eval,
    _EVAL_SUBPROCESS_CONCURRENCY, EVAL_SHARD_GAMES,
)
from agents.training.graceful_restart_callback import GracefulRestartCallback
from agents.training.snapshot_pool import (
    SnapshotPool, heuristic_fraction, HEURISTIC_FLOOR, SELF_PLAY_START, SELF_PLAY_FULL,
)
from agents.training.selfplay_callback import SelfPlayCallback
from agents.training.wrappers import MaskableAgentWrapper, STABLE_CHALLENGE_SHARE
from agents.training.gen3_env import Gen3Env
from utils.bridge.bridge_session import attach_bridge_transport
from utils.bridge.local_battle_runner import run_local_battles
from agents.training.reward_manager import Gen3RewardManager
from agents.training.stall import StallConfig
from agents.training.watchdog import start_subprocess_watchdog, start_orphan_watchdog
from agents.training.adaptive_lr_callback import AdaptivePPOCallback, TwoPhaseLRCallback
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.async_vec_env import AsyncSubprocVecEnv
from agents.training.metrics_exporter_callback import MetricsExporterCallback
from utils.logging.levels import LogLevel
from main.exit_codes import TrainExitCode
from main.launcher.ipc import send_event, emit

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from agents.opponents import (
    Gen3StallerPlayer, Gen3AggressivePlayer, Gen3SetupSweepPlayer,
    Gen3StallerV2Player, Gen3AggressiveV2Player, Gen3SetupSweepV2Player,
    Gen3HeuristicV2Player,
)
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.ps_client.server_configuration import localhost_server_configuration

BATTLE_FORMAT = "gen3ou"
CLIP_RANGE_DEFAULT = 0.15


def optional_float(s: str) -> float | None:
    """argparse `type=` converter for an optional float (`float | None`).

    Returns `None` for the sentinels `none`/`null`/`""` (case-insensitive),
    otherwise parses a float. A bad value raises `ValueError`, which argparse
    turns into a clean usage error. Used by `--clip-range-vf` so `none`
    disables value-function clipping (SB3 branches on `clip_range_vf is None`).
    """
    if s.strip().lower() in ("none", "null", ""):
        return None
    return float(s)


_BOOL_TRUE = ("true", "t", "yes", "y", "1", "on")
_BOOL_FALSE = ("false", "f", "no", "n", "0", "off")


def str2bool(s: str) -> bool:
    """Parse a human boolean: true/false, yes/no, 1/0, on/off (case-insensitive)."""
    v = s.strip().lower()
    if v in _BOOL_TRUE:
        return True
    if v in _BOOL_FALSE:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean ({'/'.join(_BOOL_TRUE)} or {'/'.join(_BOOL_FALSE)}), got {s!r}")


class BoolFlag(argparse.Action):
    """Boolean flag accepting BOTH the bare/`--no-` form AND an explicit value.

    Registers a generated `--no-<flag>` for every `--<flag>` (like
    argparse.BooleanOptionalAction) but ALSO takes an optional value:
        --foo               -> True
        --no-foo            -> False
        --foo true | false  -> parsed (also yes/no, 1/0, on/off; --foo=false too)
    Passing a value to the negation (`--no-foo true`) is a usage error.
    """

    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        opts, self._negatives = [], set()
        for opt in option_strings:
            opts.append(opt)
            if opt.startswith("--"):
                neg = "--no-" + opt[2:]
                opts.append(neg)
                self._negatives.add(neg)
        super().__init__(option_strings=opts, dest=dest, nargs="?", default=default,
                         required=required, help=help, metavar="{true,false}")

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self._negatives:
            if values is not None:
                raise argparse.ArgumentError(
                    self, f"{option_string} is a negation and does not take a value")
            setattr(namespace, self.dest, False)
        elif values is None:            # bare `--foo`
            setattr(namespace, self.dest, True)
        else:                           # `--foo <value>` / `--foo=<value>`
            setattr(namespace, self.dest, str2bool(values))


def _load_saved_version(model_path: str):
    """Best-effort read of a checkpoint's saved ModelVersion (its model_config.json).

    Returns the ModelVersion, or **None** when the config is missing/unreadable — so a caller can
    distinguish "could not determine" from a real value (rather than silently fail-safe to a default
    and then FATAL at the version check). Used to let a flagless resume INHERIT every version-checked
    structural toggle (use_popart / value_active_readout / opp_belief_cls_k / attend_unrevealed_opponents)
    + the belief coef, so the documented `--model … --steps …` resume works uniformly."""
    try:
        from agents.model.snapshot import _resolve_paths
        from agents.model.model_version import ModelVersion
        _, cfg_dir = _resolve_paths(model_path)
        cfg = os.path.join(cfg_dir, "model_config.json")
        if not os.path.exists(cfg):
            # The checkpoint may live in <run>/checkpoints/ while the run-level
            # model_config.json stays at the run root — search the parent too (mirroring
            # load_model_snapshot). Without this, a flagless resume of a toggle-ON run reads
            # no saved version, falls back to OFF defaults, and FATALs at the arch check.
            parent_cfg = os.path.join(os.path.dirname(cfg_dir), "model_config.json")
            if os.path.exists(parent_cfg):
                cfg = parent_cfg
        if os.path.exists(cfg):
            return ModelVersion.from_json_file(cfg)
    except Exception as e:
        print(f"[Resume] WARNING: could not read saved model_config.json from {model_path}: {e}")
    return None


def _run_arch_toggles(args) -> dict:
    """The architecture TOGGLES of THIS run, for current_model_version so the version gate compares
    like-for-like against the run's own (toggle-ON) pool/stable-opponent snapshots. Without these, a
    belief-ON / popart / attend-unrevealed run would FATAL on every snapshot it is meant to protect."""
    return dict(
        attend_unrevealed_opponents=args.attend_unrevealed_opponents,
        opp_belief_cls_k=args.opp_belief_cls_k,
        opp_belief_slots=(args.opp_belief_aux_coef > 0.0),
        value_active_readout=args.value_active_readout,
        use_popart=args.use_popart,
        move_belief_mode=args.move_belief_mode,
        opp_belief_latent=(args.opp_belief_latent_coef > 0.0),
        damage_op=args.damage_op,
        damage_reattend=args.damage_reattend,
        damage_outgoing=args.damage_outgoing,
        move_candidate_floor=args.move_candidate_floor,
        move_latent=args.move_latent,
        spread_belief=args.spread_belief,
        spread_belief_nature=args.spread_belief_nature,
        spread_belief_nature_marginalize=args.spread_belief_nature_marginalize,
        move_prior_fusion=args.move_prior_fusion,
        move_belief_prefuse=args.move_belief_prefuse,
        mask_incoming_damage_obs=args.mask_incoming_damage_obs,
        mask_active_move_scalars_obs=args.mask_active_move_scalars_obs,
        mask_move_effects_obs=args.mask_move_effects_obs,
        win_prob_mode=args.win_prob_mode,
        value_dist_mode=args.value_dist_mode,
        value_dist_bins=args.value_dist_bins,
        value_dist_vmin=args.value_dist_vmin,
        value_dist_vmax=args.value_dist_vmax,
        damage_topk_k=args.damage_topk_k,
        damage_refine_rounds=args.damage_refine_rounds,
        damage_matrices_outgoing=args.damage_matrices_outgoing,
        damage_matrices_incoming=args.damage_matrices_incoming,
        damage_matrices_outgoing_all=args.damage_matrices_outgoing_all,
        threat_refine_outgoing=args.threat_refine_outgoing,
        threat_unrevealed_outgoing=args.threat_unrevealed_outgoing,
        threat_prob_outspeed=args.threat_prob_outspeed,
        threat_status_refine=args.threat_status_refine,
        hp_type_belief_mode=args.hp_type_belief_mode,
    )


def _model_hparams(model) -> dict:
    clip_range_vf = float(model.clip_range_vf(1.0)) if model.clip_range_vf is not None else -1.0
    opt = model.policy.optimizer
    return {
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "ent_coef": float(model.ent_coef),
        "vf_coef": float(model.vf_coef),
        "opp_belief_aux_coef": float(getattr(model, "opp_belief_aux_coef", 0.0)),
        "move_belief_coef": float(getattr(model, "move_belief_coef", 0.0)),
        "move_belief_latent_coef": float(getattr(model, "move_belief_latent_coef", 0.0)),
        "spread_belief_coef": float(getattr(model, "spread_belief_coef", 0.0)),
        "hp_type_belief_coef": float(getattr(model, "hp_type_belief_coef", 0.0)),
        "opp_belief_latent_coef": float(getattr(model, "opp_belief_latent_coef", 0.0)),
        "win_prob_coef": float(getattr(model, "win_prob_coef", 1.0)),
        "value_dist_coef": float(getattr(model, "value_dist_coef", 1.0)),
        "batch_size": model.batch_size,
        "grad_accum_steps": int(getattr(model, "grad_accum_steps", 1)),
        "n_steps": model.n_steps,
        "clip_range": float(model.clip_range(1.0)),
        "clip_range_vf": clip_range_vf,
        "optimizer": type(opt).__name__,
        "weight_decay": opt.param_groups[0].get("weight_decay", 0.0),
    }


def _write_latest_txt(model_dir: str, name: str) -> None:
    """Atomically record the most-recent checkpoint in <model_dir>/latest.txt.

    ``name`` is resolved RELATIVE to ``model_dir`` (the run root): periodic + forced
    checkpoints live under ``checkpoints/`` so their name is run-relative
    (``checkpoints/checkpoint_123_steps.zip``); the final-model singletons stay at the
    run root so their name is a bare basename (``final_model.zip``). Every reader joins
    it back with the run dir (``os.path.join(run_dir, name)``), so both forms resolve.
    """
    latest = os.path.join(model_dir, "latest.txt")
    tmp = latest + ".tmp"
    with open(tmp, "w") as f:
        f.write(name + "\n")
    os.replace(tmp, latest)


def _attach_run_tb_logger(model, model_dir: str) -> str:
    """Route SB3's logger to ``<model_dir>/tb/`` (stdout + tensorboard).

    SB3's ``learn(tb_log_name=...)`` always appends a ``_<N>`` run-id, so it can't
    write a bare ``tb/`` dir. Configuring the logger ourselves and ``set_logger``-ing
    it bypasses that (``_custom_logger`` makes ``learn`` skip its own logger setup),
    landing the run's TensorBoard data inside its own model dir — co-located with the
    checkpoints (NOT a separate top-level ``tensorboard/`` tree). The path is
    cwd-relative via ``model_dir``, the same basis the checkpoints use, so it lands in
    the main repo even under the launcher's worktree pin; and promoting a run to a
    golden (``mv models/run_X models/_goldens/<name>``) carries its curves along. Point
    ``tensorboard --logdir models`` to see every run + golden, each named by its dir.
    """
    from stable_baselines3.common.logger import configure as _sb3_configure
    tb_dir = os.path.join(model_dir, "tb")
    fmts = ["stdout", "tensorboard"] if model.verbose >= 1 else ["tensorboard"]
    model.set_logger(_sb3_configure(tb_dir, fmts))
    return tb_dir


class _HparamLogCallback(BaseCallback):
    """Logs static hyperparameters to TensorBoard once at training start."""

    def __init__(self, ent_coef: float):
        super().__init__()
        self._ent_coef = ent_coef

    def _on_training_start(self) -> None:
        self.logger.record("hparams/ent_coef", self._ent_coef)
        self.logger.record("hparams/gamma", self.model.gamma)
        self.logger.record("hparams/gae_lambda", self.model.gae_lambda)
        self.logger.record("hparams/vf_coef", float(self.model.vf_coef))
        self.logger.dump(self.num_timesteps)

    def _on_step(self) -> bool:
        return True


class _TrackingCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that keeps latest.txt up to date and writes per-checkpoint metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_lr_fn = None
        self._current_epochs_fn = None
        # Optional: returns the current TwoPhaseLR handoff_lr (or None).
        self._handoff_lr_fn = None
        # SB3 writes the .zip into self.save_path, which we point at <run>/checkpoints/.
        # latest.txt + metadata.json are run-LEVEL, so derive the run root (the parent
        # of the checkpoints/ subdir; == save_path if it isn't one, e.g. legacy/tests).
        self._run_dir = (
            os.path.dirname(self.save_path)
            if os.path.basename(os.path.normpath(self.save_path)) == "checkpoints"
            else self.save_path
        )

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            # SB3 just wrote the .zip into self.save_path (<run>/checkpoints/). latest.txt
            # records the run-RELATIVE path (checkpoints/checkpoint_<N>_steps.zip) and the
            # per-checkpoint sidecar lands next to the .zip; metadata.json (snapshot_history)
            # stays at the run root (self._run_dir).
            ckpt_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            _write_latest_txt(self._run_dir, os.path.relpath(ckpt_path, self._run_dir))
            if self._current_lr_fn is not None and self._current_epochs_fn is not None:
                handoff_lr = self._handoff_lr_fn() if self._handoff_lr_fn is not None else None
                record_checkpoint(
                    self._run_dir,
                    ckpt_path,
                    self._current_lr_fn(),
                    self._current_epochs_fn(),
                    hparams=_model_hparams(self.model),
                    handoff_lr=handoff_lr,
                )
        return result


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


def _run_roundtrip_test(model, layout: dict, policy_kwargs: dict, debug: bool = False) -> None:
    """Startup smoke test: save → reload → zero forward pass → assert output shape.

    Catches serialization failures at second 5, not hour 50. Raises on any failure.
    """
    import shutil
    import tempfile
    import torch
    import numpy as np
    from agents.model.features_extractor import PROJECTION_DIM

    version = ModelVersion.from_layout_and_policy_kwargs(
        layout, policy_kwargs, vf_coef=float(model.vf_coef),
        value_tail_weight=float(getattr(model, "value_tail_weight", 0.0)),
        opp_belief_aux_coef=float(getattr(model, "opp_belief_aux_coef", 0.0)),
        move_belief_coef=float(getattr(model, "move_belief_coef", 0.0)),
        opp_belief_latent_coef=float(getattr(model, "opp_belief_latent_coef", 0.0)),
        win_prob_coef=float(getattr(model, "win_prob_coef", 1.0)),
        move_belief_latent_coef=float(getattr(model, "move_belief_latent_coef", 0.0)),
        spread_belief_coef=float(getattr(model, "spread_belief_coef", 0.0)),
        value_dist_coef=float(getattr(model, "value_dist_coef", 1.0)),
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


# Wait for an in-flight subprocess eval to FINISH on a graceful restart so its
# results land before exit. A scheduled restart is self-initiated by
# GracefulRestartCallback at a rollout boundary, and the launcher won't force-kill
# until the child overruns the deadline by --restart-grace-minutes (20 min default),
# so a 10-min drain fits. The checkpoint is saved first either way, so even the
# pathological forced-SIGTERM case (child already overran → ~90s SIGKILL) is safe —
# it only risks losing the in-flight eval, never the checkpoint.
_ABORT_EVAL_DRAIN_SEC = 600.0


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


async def main():
    # --- Pre-flight Checks ---
    try:
        import tensorboard
    except ImportError:
        print("\n" + "🛑" * 30)
        print("🛑 ERROR: Tensorboard is NOT installed.")
        print("🛑 Training requires tensorboard for professional logging.")
        print("🛑 Please run: pip install tensorboard")
        print("🛑" * 30 + "\n")
        os._exit(1)

    # --- Fail-Fast Handlers ---
    def global_exception_handler(exctype, value, tb):
        print("\n" + "🛑" * 20)
        print("🛑 FATAL ERROR DETECTED - FAILING FAST")
        print("🛑" * 20)
        traceback.print_exception(exctype, value, tb)
        os._exit(1) # Force immediate termination of all threads

    sys.excepthook = global_exception_handler
    
    def asyncio_exception_handler(loop, context):
        msg = context.get("exception", context["message"])
        print(f"\n🛑 Asyncio Error: {msg}")
        os._exit(1)
        
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(asyncio_exception_handler)

    parser = argparse.ArgumentParser(description="Train or Evaluate Gen 3 OU RL Agent")
    
    # --- Operational Flags ---
    parser.add_argument("--model", type=str, help="Path to existing model to load")
    parser.add_argument("--run-dir", type=str, help="Run folder to write checkpoints into (set by launcher on resume)")
    parser.add_argument("--eval-only", action=BoolFlag, default=False, help="Skip training and only evaluate")
    parser.add_argument("--steps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--debug", action=BoolFlag, default=False, help="Use DummyVecEnv (1 env) for debugging")
    parser.add_argument("--debug-eval", "--debug_eval", dest="debug_eval", action=BoolFlag, default=False,
                        help="Run evaluation under --debug. By default a --debug smoke run skips ALL eval "
                             "(both the periodic eval callback AND the final win-rate eval) so it needs no "
                             "eval opponents / Showdown eval connection and stays light on CPU. Pass "
                             "--debug-eval to exercise the eval pipeline in a smoke run. No effect on real "
                             "(non-debug) runs, which always eval.")
    parser.add_argument("--n-envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--async-rollout", "--async_rollout", dest="async_rollout",
                        action=BoolFlag, default=False,
                        help="Non-barrier async rollout collection: keep every env worker "
                             "continuously in-flight and forward whichever are ready, instead of "
                             "barriering on the slowest env each step (AsyncSubprocVecEnv + an "
                             "on-policy async collect_rollouts that overlaps the GPU forward with "
                             "CPU env-stepping). Off by default; ignored under --debug. With async, "
                             "right-size --n-envs nearer the core count (16) rather than oversubscribing.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, or auto)")
    parser.add_argument("--showdown-port", type=int, default=None,
                        help="Local Showdown server port (default 8000). Sets the port for the trainee, "
                             "eval, and self-play clients. Start the server on the matching port, "
                             "e.g. npm run showdown -- <port>.")
    parser.add_argument("--use-showdown-bridge", action=BoolFlag, default=False,
                        help="Use the in-process BattleStream bridge instead of a websocket "
                             "Showdown server: each training env owns a local sim subprocess and "
                             "eval/self-play play in-process via run_local_battles — no server, no "
                             "port, no /challenge connection storm, deterministic delivery. Covers "
                             "BOTH training AND eval, so a run needs no Showdown server at all. "
                             "Default False (websocket).")
    parser.add_argument(
        "--self-play-use-cpu",
        action=BoolFlag,
        default=True,
        help="Load self-play opponent snapshots on CPU instead of the training device. "
             "Default True: avoids one CUDA context per SubprocVecEnv worker (~300-600 MB each), "
             "which would otherwise OOM the GPU at high --n-envs. Opponent inference is batch-1 "
             "no_grad, so CPU is plenty fast. Pass --no-self-play-use-cpu to load them on --device.",
    )
    parser.add_argument("--eval-battles", type=int, default=100, help="Battles per evaluation opponent")
    parser.add_argument("--eval-concurrency", type=int, default=100, help="Concurrent battles during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log-level", type=str, default="periodic", choices=["quiet", "periodic", "detailed", "debug"], help="Logging verbosity level")

    # --- Hyperparameter Flags (Optimized for GPU) ---
    parser.add_argument("--batch-size", type=int, default=4096, help="PPO mini-batch size")
    parser.add_argument("--grad-accum-steps", "--grad_accum_steps", dest="grad_accum_steps",
                        type=int, default=1,
                        help="Gradient accumulation: sum the gradients of K --batch-size MICRO-batches "
                             "and step the optimizer ONCE per group of K, giving the EXACT gradient of a "
                             "(batch_size·K) batch at the GPU-memory cost of batch_size (only one "
                             "micro-batch's activations are ever held). 1 = OFF (one step per minibatch, "
                             "byte-identical to stock). Use it to keep a large effective batch when the "
                             "full minibatch OOMs: e.g. --batch-size 4096 --grad-accum-steps 4 ≈ "
                             "--batch-size 16384 at ¼ the activation peak. A train-loop knob (not "
                             "version-locked); pass it on every resume like --batch-size.")
    parser.add_argument("--n-epochs", type=int, default=5, help="PPO optimization epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate (AdaptiveLRCallback adjusts from here)")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Hard lower bound on adaptive LR")
    parser.add_argument("--max-lr", type=float, default=None, help="Hard upper bound on adaptive LR (default: 2× --lr)")
    parser.add_argument("--anneal-lr-start-steps", type=int, default=None,
                        help="Absolute global step at which cosine LR decay begins. "
                             "Duration = --steps minus this value. Pass the same value on every resume.")
    parser.add_argument("--anneal-min-lr", type=float, default=None,
                        help="LR floor for annealing (required with --anneal-lr-start-steps). "
                             "Separate from --min-lr used by AdaptivePPO.")
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy coefficient (exploration bonus)")
    parser.add_argument("--defensive-entropy-boost", "--defensive_entropy_boost", dest="defensive_entropy_boost",
                        type=float, default=1.0,
                        help="STATE-CONDITIONED entropy boost (gen3_defensive_entropy_v1): multiply the "
                             "per-decision entropy bonus by this factor ON decisions where the active mon has a "
                             "productive defensive move legal (HP-recovery with HP to restore, or a self/team "
                             "status-cure with a status to clear). Keeps the policy EXPLORING defensive moves "
                             "(Recover/Soft-Boiled/Wish/Refresh/Heal Bell) instead of collapsing to attacking, "
                             "WITHOUT touching the reward (no stall incentive — the draw penalty + no-progress "
                             "clock stay the guardrail; the model only keeps healing if the returns reward it). "
                             "1.0 = OFF (byte-identical). Try 3.0. TRAINING-only (not version-locked).")
    parser.add_argument("--defensive-entropy-anneal-frac", "--defensive_entropy_anneal_frac",
                        dest="defensive_entropy_anneal_frac", type=float, default=0.0,
                        help="Anneal --defensive-entropy-boost linearly back to 1.0 over this FRACTION of total "
                             "--steps (e.g. 0.5 = boost fades to off by the halfway point). 0.0 = constant boost "
                             "(default). Lets exploration fade as the policy learns defensive value.")
    parser.add_argument("--vf-coef", "--vf_coef", dest="vf_coef", type=float, default=0.5,
                        help="PPO value-loss coefficient (default 0.5, the SB3 default). Fixed for a "
                             "run's lifetime: it is recorded in model_config.json and resuming with a "
                             "different value is a FATAL error (it silently rescales the value head's "
                             "gradient on the shared trunk — tune it on a fresh run). Watch "
                             "grad/value_policy_logratio (the aux-independent value-vs-policy balance).")
    parser.add_argument("--value-tail-weight", "--value_tail_weight", dest="value_tail_weight",
                        type=float, default=0.0,
                        help="Tail-weighted value loss β∈[0,1] (default 0.0 = plain MSE, byte-identical). "
                             ">0 blends in the CVaR of the worst ~10%% value misses: (1-β)·MSE + β·CVaR, "
                             "so the critic prioritises the big over-claim craters it under-prices (a "
                             "probe found VF→incoming-KO AUC 0.79 vs the policy's 0.90). Symmetric in "
                             "error sign → V stays unbiased (GAE advantages unaffected). Watch "
                             "eval/td_resid_tail fall. Resume-immutable (recorded + FATAL to change).")
    # --- Reward config (design_markovian_reward_and_features.md). Resume-immutable, value-checked. ---
    parser.add_argument("--bias-additivity", "--bias_additivity", dest="bias_additivity", type=float,
                        default=1.0, help="BIAS-class additive↔telescoping knob λ∈[0,1] (default 1.0 = "
                        "fully additive, byte-identical to today's biases). 0.0 = fully telescoping "
                        "(pure PBRS hint). Per-run constant (NOT annealed). Resume-immutable.")
    parser.add_argument("--mat-alive-weight", "--mat_alive_weight", dest="mat_alive_weight", type=float,
                        default=1.25, help="Material PBRS Φ_mat per-mon-alive weight (default 1.25). "
                        "Resume-immutable.")
    parser.add_argument("--no-progress-penalty", "--no_progress_penalty", dest="no_progress_penalty",
                        type=float, default=0.15, help="Flat per-no-progress-window penalty magnitude "
                        "(default 0.15; only charged when --bias-redesign).")
    parser.add_argument("--bias-redesign", "--bias_redesign", dest="bias_redesign", action=BoolFlag,
                        default=False, help="Enable the staged BIAS redesign: the no-progress clock "
                        "replaces the anti-spam taxes + the obs-keyed reframes apply. Default OFF = the "
                        "single-variable run (material clutch-fix only). Pass --no-bias-redesign (or "
                        "--bias-redesign false) to set it off explicitly. Resume-immutable.")
    parser.add_argument("--switch-bias-weight", "--switch_bias_weight", dest="switch_bias_weight",
                        type=float, default=0.0, help="Belief-risk-scaled stay-into-KO BIAS lever for "
                        "the under-switch pathology (design_reward_switching.md §7). 0.0 = OFF "
                        "(default; behavior unchanged). >0 taxes staying in a high-P(KO) spot when a "
                        "safe pivot exists (−w·risk) + rewards escaping it. BIAS-class, so it also "
                        "rides --bias-additivity (λ=1 additive vs λ=0 telescoping A/B). Resume-immutable.")
    parser.add_argument("--draw-penalty", "--draw_penalty", dest="draw_penalty", type=float,
                        default=-30.0, help="Terminal reward for a DRAW / 250-turn timeout (no "
                        "winner). Default -30.0 = same as a decisive loss (behavior unchanged). Set "
                        "more negative (e.g. -35) to make stalling to the turn cap strictly worse "
                        "than losing cleanly — discourages no-progress stall-wars. A decisive loss "
                        "stays -30. Resume-immutable (recorded + value-checked in model_config.json).")
    parser.add_argument("--self-ko-hp-penalty", "--self_ko_hp_penalty", dest="self_ko_hp_penalty",
                        type=float, default=0.0,
                        help="Decision-time-HP-scaled penalty (-w*hp) for self-KOing a mon via "
                        "Explosion/Self-Destruct. Default 0.0 = OFF (behavior unchanged). The symmetric "
                        "material PBRS prices a healthy 1-for-1 trade at ~0, so the critic learns to "
                        "value a full-HP self-KO positively and the policy throws away healthy mons "
                        "(measured: ~38%% of explosions are on >=80%%-HP mons). >0 (e.g. 2.5) charges "
                        "the squandered HP, sparing legitimate low-HP sac-for-KO. Resume-immutable "
                        "(recorded + value-checked in model_config.json).")
    parser.add_argument("--drop-redundant-bias", "--drop_redundant_bias", dest="drop_redundant_bias",
                        action=BoolFlag, default=False, help="De-bias cleanup: zero BIAS terms REDUNDANT "
                        "with an existing PBRS/terminal term — stall_tax (covered by the no-progress clock "
                        "+ --draw-penalty; it also taxed winning long games on raw turn count) and "
                        "matchup_penalty (the same incoming-KO threat signal as pbrs_belief, but additive "
                        "not telescoping). Default OFF = byte-identical. Resume-immutable, value-checked.")
    parser.add_argument("--drop-switch-bias", "--drop_switch_bias", dest="drop_switch_bias",
                        action=BoolFlag, default=False, help="De-bias cleanup: zero the HAND-CODED "
                        "switch-strategy subsidy (switch_base, switch_bouncing_tax, escape_threat_switch, "
                        "se_switch, pivot_*, sleep_in/out) — switching value is LEARNABLE from Φ_mat + "
                        "pbrs_belief + win/loss, so hand-rewarding it distorts the objective. Default OFF "
                        "= byte-identical. Resume-immutable, value-checked.")
    parser.add_argument("--all-shaping-pbrs", "--all_shaping_pbrs", dest="all_shaping_pbrs",
                        action=BoolFlag, default=False, help="END-STATE PBRS, 'everything but stall': "
                        "fold Φ_hazard/Φ_boost/Φ_opp_boosts + Φ_status (telescoping, policy-invariant) "
                        "and ZERO every BIAS term EXCEPT the anti-stall tilt no_progress_tax — so all "
                        "non-stall shaping is policy-invariant (the bad turn-ramp stall_tax is zeroed). "
                        "Default OFF = byte-identical. Pair with --stall-pbrs for a FULLY-PBRS reward, or "
                        "use alone to keep the no_progress stall tilt as the one acknowledged BIAS. "
                        "Resume-immutable, value-checked.")
    parser.add_argument("--stall-pbrs", "--stall_pbrs", dest="stall_pbrs",
                        action=BoolFlag, default=False, help="END-STATE PBRS, 'stall': fold Φ_progress "
                        "(telescoping anti-stall over the turns_since_progress clock) and ZERO "
                        "no_progress_tax + stall_tax, so the anti-stall signal is policy-invariant too. "
                        "Default OFF. Run --all-shaping-pbrs WITH --stall-pbrs ⇒ the whole BIAS class is "
                        "zero (TERMINAL + PBRS only); WITHOUT it ⇒ keep the no_progress stall tilt as "
                        "insurance against stall-regression (watch the stall-rate canary). Resume-"
                        "immutable, value-checked.")
    parser.add_argument("--clip-range", type=float, default=CLIP_RANGE_DEFAULT, help="PPO policy clip range (default 0.15)")
    parser.add_argument("--clip-range-vf", type=optional_float, default=0.5, help="Value function clip range; pass 'none' to disable clipping (thesis used 0.0184)")
    parser.add_argument("--use-popart", "--use_popart", dest="use_popart", action=BoolFlag, default=None,
                        help="Enable PopArt value-target normalization (adaptive (mu,sigma) on the "
                             "value head; keeps the value gradient O(1) so it stops swamping the "
                             "shared trunk). Requires an explicit --clip-range-vf none (value "
                             "clipping is unnecessary with normalization). Version-checked: cannot "
                             "be toggled on a resumed model.")
    parser.add_argument("--attend-unrevealed-opponents", "--attend_unrevealed_opponents",
                        dest="attend_unrevealed_opponents", action=BoolFlag, default=None,
                        help="Keep the opponent's still-hidden party (unrevealed mons — Gen 3 has no "
                             "team preview) ATTENDABLE in the transformer instead of key-masking them "
                             "identically to fainted mons. Lets the body reason about the hidden team. "
                             "No weight-shape change; version-checked, so it cannot be toggled on a "
                             "resumed model. Off by default (clean A/B baseline).")
    parser.add_argument("--opp-belief-cls-k", "--opp_belief_cls_k", dest="opp_belief_cls_k",
                        type=int, default=None,
                        help="Hidden-opponent belief: number of distinct learned query tokens (DETR "
                             "object-query style) that summarise the unrevealed opp party and feed both "
                             "heads. 0 = OFF (default, baseline arch). 1 = a single 'hidden-opponent CLS' "
                             "set-summary; >1 = N distinct per-slot queries that coordinate + specialise. "
                             "k>0 REQUIRES --attend-unrevealed-opponents (else the queries read a board "
                             "with the hidden mons masked out) and is a weight-shape change (version-"
                             "checked, cannot change on a resume). NOTE: without a dedicated aux objective "
                             "(B3 — species-ID / BYOL) the RL gradient only weakly shapes these queries.")
    parser.add_argument("--opp-belief-aux-coef", "--opp_belief_aux_coef",
                        dest="opp_belief_aux_coef", type=float, default=None,
                        help="In-place hidden-opponent BELIEF AUX (the B3 objective). 0.0 = OFF (default). "
                             ">0 turns ON opp_belief_slots (fills the un-revealed opp team slots with "
                             "distinct learned unknown-mon tokens refined in-lineup by the transformer + a "
                             "BeliefHead) and AUTO-FORCES --attend-unrevealed-opponents, and adds "
                             "coef*(species_CE + moves_BCE) over the believed slots to the PPO loss. The "
                             "slot module is weight-shape (version-checked); the coef itself is a "
                             "TRAINING-only hparam like --ent-coef (NOT resume-locked). The privileged "
                             "belief obs labels exist only when >0.")
    parser.add_argument("--opp-belief-moves-weight", "--opp_belief_moves_weight",
                        dest="opp_belief_moves_weight", type=float, default=1.0,
                        help="Relative weight of the moves multi-label BCE vs the species CE inside the "
                             "belief aux term (aux = species_CE + w·moves_BCE; both on a per-believed-slot "
                             "scale). Default 1.0 — species dominates; raise to up-weight move prediction. "
                             "TRAINING-only, like --opp-belief-aux-coef. Ignored when the coef is 0. The "
                             "explicit --[no-]predict-unrevealed-mon-moves knob below is the clear on/off.")
    parser.add_argument("--predict-unrevealed-mon-moves", "--predict_unrevealed_mon_moves",
                        dest="predict_unrevealed_mon_moves", action=BoolFlag, default=None,
                        help="EXPLICIT clarity knob: should the model predict the MOVES of opponent mons it "
                             "has NOT even seen (the hidden bench)? Default (unset) = yes (current behavior). "
                             "--no-predict-unrevealed-mon-moves turns it OFF — zeros BOTH hidden-mon "
                             "move-prediction paths: the BeliefHead's hidden-slot moves-BCE "
                             "(--opp-belief-moves-weight → 0) AND any MoveBelief unrevealed leg "
                             "(--move-belief-mode 'unrevealed'/'both' → 'revealed'). The REVEALED-mon move "
                             "belief (a SEEN mon's unseen slots) and the SPECIES belief on hidden mons are "
                             "UNTOUCHED. A desugar into existing fields — no version field.")
    parser.add_argument("--move-belief-mode", "--move_belief_mode", dest="move_belief_mode",
                        choices=("off", "revealed", "unrevealed", "both"), default=None,
                        help="MOVE-belief REINJECTION: predict each opp mon's moveset and FLOW it back into "
                             "the slot token (soft move-embedding added before the CLS pools), so the policy/"
                             "value heads reason about the believed moves — not a dead-end readout. 'off' "
                             "(default) = no module (baseline byte-for-byte). 'revealed' = seen mons only "
                             "(predict their still-UNREVEALED moves — the defensible, surprise-OHKO lever). "
                             "'unrevealed' = hidden mons (Hungarian-matched, omniscient — REQUIRES "
                             "--opp-belief-aux-coef>0, else the hidden slots are empty placeholders). 'both' "
                             "= all slots (also requires it). STRUCTURAL (a new head; version-"
                             "checked, fresh-only — cannot change on a resume) and AUTO-FORCES "
                             "--attend-unrevealed-opponents. Supervised by privileged labels (the model's own "
                             "full team), training-only. The known-vs-unknown axis is the defensible-vs-"
                             "omniscient A/B.")
    parser.add_argument("--move-belief-coef", "--move_belief_coef", dest="move_belief_coef",
                        type=float, default=None,
                        help="Loss weight for the move-belief head (move_belief_coef * BCE over the scored "
                             "opp slots), like --opp-belief-aux-coef. 0.0 = no supervised pull (the module "
                             "still reinjects, but only RL gradient shapes it). TRAINING-only (not version-"
                             "locked). Ignored when --move-belief-mode off.")
    parser.add_argument("--opp-belief-latent-coef", "--opp_belief_latent_coef",
                        dest="opp_belief_latent_coef", type=float, default=None,
                        help="LATENT-belief escalation. 0.0 = OFF (default). >0 turns ON opp_belief_latent "
                             "(adds an asymmetric SimSiam predictor to the BeliefHead) and adds "
                             "coef*(cosine-to-encoder-role-token + VICReg) over the believed slots: each "
                             "slot's refined token is regressed toward the STOP-GRAD pokemon_encoder "
                             "role-token of the TRUE hidden mon — graded identity supervision the hard "
                             "species CE can't give. REQUIRES --opp-belief-aux-coef>0 (the believed slots + "
                             "species head + Hungarian assignment it rides). The predictor is weight-shape "
                             "(version-checked, fresh-only); the coef is TRAINING-only like --opp-belief-aux-"
                             "coef. The privileged belief_target_slots obs key exists only when >0.")
    parser.add_argument("--value-active-readout", "--value_active_readout", dest="value_active_readout",
                        action=BoolFlag, default=None,
                        help="Route the active mon's refined token (our_active_refined) into the VALUE "
                             "head's projection. The dual-head value readout pools the whole board but "
                             "DROPS the active-mon view the policy head keeps — a probe found the critic "
                             "predicts an incoming self-KO at AUC 0.79 vs the policy's 0.90, under-pricing "
                             "the V-tail. Widens the value projection by D_MODEL (weight-shape, version-"
                             "checked, cannot change on a resume). Off by default (clean A/B baseline).")
    parser.add_argument("--damage-op", "--damage_op", dest="damage_op",
                        action=BoolFlag, default=None,
                        help="Differentiable GPU damage operator: compute the believed-move incoming "
                             "damage the opp ACTIVE would deal to each of our mons, fed by the MOVE "
                             "belief's predicted moves (sigmoid logits), and append it to BOTH heads. "
                             "Differentiable, so gradients sharpen the move belief toward real KO "
                             "threats; replaces the CPU obs block's fixed usage-prior with the LEARNED "
                             "belief. STRUCTURAL (widens both projections; version-checked, fresh-only). "
                             "REQUIRES --move-belief-mode revealed|both (it reads the opp active's "
                             "predicted logits, supervised only for a revealed mon). Off by default.")
    parser.add_argument("--damage-reattend", "--damage_reattend", dest="damage_reattend",
                        action=BoolFlag, default=None,
                        help="Re-attend the team transformer to the computed damage: project the op's "
                             "per-OUR-mon incoming-damage block onto the 6 our-team tokens, run ONE more "
                             "encoder layer (our↔opp), then re-derive the CLS pools — so attention now reasons "
                             "OVER the physics and the pi/vf pools are damage-AWARE board summaries instead of "
                             "damage-blind ones (today the damage block is a post-pool concat no attention "
                             "sees). NOTE: a BOARD-level enrichment, NOT first-class per-candidate switch "
                             "scoring (the bench tokens are pooled back to one vector; that needs a per-bench "
                             "pointer head, a follow-up). Identity-at-init ⇒ ON starts ≈ the --damage-op "
                             "baseline. STRUCTURAL (adds modules; version-checked, fresh-only; projection "
                             "widths unchanged). REQUIRES --damage-op. PopArt strongly recommended (the extra "
                             "shared-trunk layer worsens value-grad contention). Off by default.")
    parser.add_argument("--unified-damage", "--unified_damage", dest="unified_damage",
                        choices=["off", "incoming", "both"], default="off",
                        help="ONE knob for the unified damage system (desugars into the component flags at "
                             "parse time): 'off' = baseline; 'incoming' = move belief (revealed) + prior "
                             "fusion + the GPU damage op (opp active → our 6 mons, incl. the safe-switch "
                             "bench rows); 'both' = also the OUTGOING per-move block (our active → opp "
                             "active, action-aligned — the equal-effectiveness tie-break). Overrides "
                             "--move-belief-mode / --damage-op / --move-prior-fusion / --damage-outgoing "
                             "when not 'off'. Pair with --move-candidate-floor (the learnset/rarity gate) "
                             "and --move-belief-mode both (to also guess unrevealed mons' moves).")
    parser.add_argument("--damage-outgoing", "--damage_outgoing", dest="damage_outgoing",
                        action=BoolFlag, default=None,
                        help="OUTGOING per-move damage direction (our active → opp active), in REQUEST-slot "
                             "order so the policy head can compare move A vs B directly (the "
                             "equal-effectiveness tie-break: Earthquake vs Brick Break into a Rock). "
                             "STRUCTURAL (widens both projections; version-checked, fresh-only). REQUIRES "
                             "--damage-op. Off by default. (Usually set via --unified-damage both.)")
    parser.add_argument("--move-candidate-floor", "--move_candidate_floor", dest="move_candidate_floor",
                        type=float, default=None,
                        help="LEGALITY-only move-prior gate. 0.0 = OFF (legacy flat 0.02-floor prior). >0 "
                             "drives moves a species CANNOT learn to ~0 (removes the phantom-threat noise the "
                             "flat floor invented), while a legal move keeps its TRUE Smogon usage (rare techs "
                             "stay rare-but-liftable, never pruned — so surprise-move anticipation survives) "
                             "and a legal-unobserved move gets this small floor as a liftable base (try 0.02 "
                             "or smaller). Forward-behavior toggle (version-checked, fresh-only). REQUIRES "
                             "--move-prior-fusion (it gates the fused prior). Off by default.")
    parser.add_argument("--move-prior-fusion", "--move_prior_fusion", dest="move_prior_fusion",
                        action=BoolFlag, default=None,
                        help="Unified two-part move belief: fuse the Smogon move-frequency PRIOR into the "
                             "move-belief head as a log-odds residual (posterior = prior + learned delta) "
                             "and PIN revealed moves certain — so the belief the damage op + BCE loss read "
                             "is one coherent posterior (priors ⊕ prediction unified), anchored at the "
                             "prior at cold-start. Forward-behavior toggle (no weight-shape change; "
                             "version-checked, fresh-only). REQUIRES --move-belief-mode != off. Off by default.")
    parser.add_argument("--move-belief-prefuse", "--move_belief_prefuse", dest="move_belief_prefuse",
                        action=BoolFlag, default=None,
                        help="Reinject the move belief BEFORE the team transformer instead of after, so the "
                             "predicted opp moves co-refine with the species/team belief through the 2 "
                             "attention layers (the believed moveset participates in attention, rather than "
                             "being grafted onto the already-refined tokens). Same MoveBelief module/params — "
                             "only the call timing differs. Forward-behavior toggle (no weight-shape change; "
                             "version-checked, fresh-only). REQUIRES --move-belief-mode != off. Off by default.")
    parser.add_argument("--mask-incoming-damage-obs", "--mask_incoming_damage_obs",
                        dest="mask_incoming_damage_obs", action=BoolFlag, default=None,
                        help="Unified-architecture ABLATION: zero the 51-dim incoming-damage / OHKO obs "
                             "block out of the MODEL's view (the block stays in the obs at a fixed dim; "
                             "the reward PBRS still reads the belief from live_view). Use WITH --damage-op "
                             "to A/B whether the learned belief→damage op replaces the CPU usage-prior "
                             "collapse — no code deleted, fully reversible. Forward-behavior toggle "
                             "(no weight-shape change; version-checked, fresh-only). Off by default.")
    parser.add_argument("--win-prob-mode", "--win_prob_mode", dest="win_prob_mode",
                        choices=("none", "read_only", "shaping"), default=None,
                        help="Auxiliary WIN-PROBABILITY head: a calibrated P(win|state) readout off the "
                             "value pool, supervised by the Monte-Carlo episode outcome (win=1/loss=0) — "
                             "the shaped critic's V is expected RETURN, not win odds, so this gives an "
                             "interpretable P(win) (and ΔP(win) per move). 'none' (default) = no module "
                             "(baseline byte-for-byte). 'read_only' = the head trains on a STOP-GRAD value "
                             "pool — a pure, risk-free diagnostic that CANNOT perturb the policy. 'shaping' "
                             "= its gradient also shapes the shared trunk (the win objective improves the "
                             "representation; A/B it vs read_only). STRUCTURAL + resume-IMMUTABLE "
                             "(version-checked: any change FATALs on resume). The head is a SIDE readout "
                             "(never in pi/vf — leak-safe).")
    parser.add_argument("--win-prob-coef", "--win_prob_coef", dest="win_prob_coef",
                        type=float, default=None,
                        help="Loss weight for the win-prob head's BCE (win_prob_coef * BCE), like "
                             "--opp-belief-aux-coef. Default 1.0. TRAINING-only (not version-locked; "
                             "inherited on a flagless resume). Ignored when --win-prob-mode none. Lower it "
                             "if 'shaping' fights the policy (watch grad/win_prob_share).")
    parser.add_argument("--value-dist-mode", "--value_dist_mode", dest="value_dist_mode",
                        choices=("none", "read_only", "shaping"), default=None,
                        help="Distributional VALUE head (v29): an interpretability readout off the value "
                             "pool emitting --value-dist-bins logits over [--value-dist-vmin, "
                             "--value-dist-vmax] — softmax = the critic's predicted RETURN DISTRIBUTION "
                             "(sharp=confident, wide=uncertain, bimodal=coinflip), reviewable per-decision "
                             "in the prober. 'none' (default) = no module (baseline byte-for-byte). "
                             "'read_only' = the head trains on a STOP-GRAD value pool (a risk-free "
                             "diagnostic that CANNOT perturb the policy). 'shaping' = its gradient also "
                             "shapes the shared trunk. STRUCTURAL + resume-IMMUTABLE (version-checked). A "
                             "SIDE readout (never in pi/vf — leak-safe). "
                             "Design: designs/ai_v6/design_distributional_value_critic.md.")
    parser.add_argument("--value-dist-bins", "--value_dist_bins", dest="value_dist_bins",
                        type=int, default=None,
                        help="Atom count for --value-dist-mode (the head's output width; weight-shape, "
                             "version-checked). Recommended 32 (readable). Required > 0 when the mode is "
                             "on; ignored (must be 0) when none.")
    parser.add_argument("--value-dist-vmin", "--value_dist_vmin", dest="value_dist_vmin",
                        type=float, default=None,
                        help="Lower edge of the value-dist atom support (the return range the atoms span). "
                             "Resume-immutable (version-checked). Required when --value-dist-mode is on.")
    parser.add_argument("--value-dist-vmax", "--value_dist_vmax", dest="value_dist_vmax",
                        type=float, default=None,
                        help="Upper edge of the value-dist atom support. Resume-immutable "
                             "(version-checked). Required when --value-dist-mode is on (must be > vmin).")
    parser.add_argument("--value-dist-coef", "--value_dist_coef", dest="value_dist_coef",
                        type=float, default=None,
                        help="Loss weight for the value-dist head's HL-Gauss CE (value_dist_coef * CE), "
                             "like --win-prob-coef. Default 1.0. TRAINING-only (not version-locked; "
                             "inherited on a flagless resume). Ignored when --value-dist-mode none. Lower "
                             "it if 'shaping' fights the policy (watch grad/value_dist_share / "
                             "grad/value_dist_policy_cosine — this head's own shared-trunk pull).")
    parser.add_argument("--move-latent", "--move_latent", dest="move_latent",
                        action=BoolFlag, default=None,
                        help="MoveLatentEncoder (gen3_unified_move_system_v1): a context-free, "
                             "mechanics-grounded per-move latent (move/type embeddings + structured "
                             "MOVE_ATTR — BP / category / accuracy / priority / drain / per-status secondary "
                             "chances) concatenated into the move network, so the model reads a richer move "
                             "identity AND the SAME latent is the similarity-grading target (Rock Slide ~= "
                             "Hidden Power Rock). STRUCTURAL (widens the move-network input; version-checked, "
                             "fresh-only). Off by default.")
    parser.add_argument("--move-belief-latent-coef", "--move_belief_latent_coef",
                        dest="move_belief_latent_coef", type=float, default=None,
                        help="Latent-space grading weight for the move belief: coef * (cosine of the "
                             "predicted move distribution's expected move-latent toward the true moveset's "
                             "mean latent + VICReg floor) on revealed slots — the soft complement to the "
                             "per-ID BCE so near-moves grade as near. REQUIRES --move-latent (reads its "
                             "latent table) and a move-belief mode that scores revealed slots. TRAINING-only "
                             "(not version-locked; inherited on a flagless resume). 0.0 = OFF.")
    parser.add_argument("--unified-moves", "--unified_moves", dest="unified_moves",
                        choices=["off", "incoming", "both"], default="off",
                        help="ONE knob for the WHOLE unified move system: sets --unified-damage to the same "
                             "level (move belief + prior fusion + the GPU damage op, incl. its per-status "
                             "secondary/Serene-Grace effects; 'both' adds the outgoing direction) AND turns "
                             "on --move-latent + a default --move-belief-latent-coef 0.05 + the DISCRETE "
                             "top-K incoming block (--damage-topk, default K=5). Compose the pieces by hand "
                             "for finer control (e.g. --damage-topk 0 to A/B it off under --unified-moves).")
    parser.add_argument("--damage-topk", "--damage_topk", dest="damage_topk_k",
                        type=int, default=None,
                        help="DISCRETE top-K incoming move-space block (gen3_unified_topk_incoming_v1): K = "
                             "the number of the opp ACTIVE's most-believed CANDIDATE moves surfaced "
                             "INDIVIDUALLY (vs the worst-case max collapse that loses WHICH move it is). Per "
                             "move: its move LATENT identity (gathered from the MoveLatentEncoder — "
                             "differentiable → sharpens the latent) + belief weight (→ sharpens the move "
                             "belief) + accuracy + is_phys, then per OUR mon [high-roll, P(KO), "
                             "status_lands] — the discrete-move + per-pivot read (incl. damage-immunity AND "
                             "status-immunity = 0, e.g. Thunder-Wave→Ground) that makes 'anticipate the move "
                             "/ pick the safe switch' decidable. 0 = off. STRUCTURAL int (scales both "
                             "projections; version-checked, fresh-only). REQUIRES --damage-op + --move-latent. "
                             "AUTO-set to 5 by --unified-moves (the moveset is 4, so the 5th slot is the "
                             "surprise/uncertain candidate); the 5th is zeroed once all 4 opp moves are "
                             "revealed. Default off (set by --unified-moves, or pass explicitly).")
    parser.add_argument("--damage-refine-rounds", "--damage_refine_rounds", dest="damage_refine_rounds",
                        type=int, default=None,
                        help="ITERATIVE damage refinement (gen3_iterative_damage_v1): N = the number of "
                             "transformer layers (capped by the layer count) before which the DamageOperator's "
                             "LEAN discrete incoming damage is RECOMPUTED from the CURRENT (being-enriched) opp "
                             "tokens — re-reading the move belief — and injected back onto our-mon tokens via a "
                             "zero-init refine_proj (identity at init). So each attention layer reasons over "
                             "physics derived from the FRESHEST belief (physics-in-the-loop), and the per-round "
                             "read sharpens the move-belief head — instead of the one-shot post-transformer op. "
                             "0 = off (baseline forward byte-for-byte). STRUCTURAL int (version-checked, "
                             "fresh-only). REQUIRES --damage-op (the op physics + the move belief). NOT auto-set "
                             "by --unified-moves — an explicit A/B lever. Default off.")
    parser.add_argument("--damage-matrices", "--damage_matrices", dest="damage_matrices",
                        choices=["off", "incoming", "outgoing", "both"], default=None,
                        help="Per-move DAMAGE MATRICES (gen3_per_move_matrices_v1). 'outgoing': OUR 4 moves × "
                             "the opp's 6 mons (active + REVEALED bench) — per (move, opp mon) "
                             "[low,high,crit,pko,type_mult] + a revealed bit (price a KO on a SWITCH-IN). "
                             "'incoming': the ENRICHED top-K — per opp move a header [latent, belief, acc, "
                             "is_phys, EXPLICIT effect bits(6), secondary chances(10)] + per (OUR mon, move) "
                             "cell [low,high,crit,pko,type_mult,status_lands] (the un-collapsed evolution of "
                             "--damage-topk; it REUSES --damage-topk K as its K — one knob, try 4/5/6, default "
                             "5 — and REPLACES the lean top-K block at that K; requires --move-latent). "
                             "'both' = incoming + outgoing. Unrevealed opp slots zeroed (belief-driven = TODO). "
                             "STRUCTURAL (version-checked, fresh-only). REQUIRES --damage-op. 'off' (default) = "
                             "baseline byte-identical.")
    parser.add_argument("--damage-matrices-outgoing-all", "--damage_matrices_outgoing_all",
                        dest="damage_matrices_outgoing_all", action=BoolFlag, default=None,
                        help="The TRANSPOSED outgoing matrix (gen3_per_move_matrices_v1, v39): OUR 6 MONS' 4 "
                             "moves → the opp ACTIVE — per (attacker mon, move) [low,high,crit,pko] + a "
                             "per-attacker p_outspeed + an alive bit. The transpose of --damage-matrices "
                             "outgoing (which prices our ACTIVE's moves vs the opp's 6 mons): on a FORCED SWITCH "
                             "the active is fainted so the single-active outgoing block zeroes and the policy "
                             "picks switch-ins BLIND to offense — this prices every candidate switch-in's "
                             "offense. The ACTIVE row reproduces the single-active block byte-for-byte (parity); "
                             "bench rows reuse the SAME physics with NEUTRAL boosts (gen3 resets on switch). "
                             "STRUCTURAL (version-checked, fresh-only). REQUIRES --damage-op. Default off "
                             "(byte-identical).")
    # gen3_bidir_threat_trunk_v1 (v36): the bidirectional in-trunk threat field (#1/#2/#3).
    parser.add_argument("--threat-refine-outgoing", "--threat_refine_outgoing", dest="threat_refine_outgoing",
                        action=BoolFlag, default=None,
                        help="#1 OUTGOING threat into the TRUNK (gen3_bidir_threat_trunk_v1): inject a per-opp-mon "
                             "outgoing-threat residual (how hard OUR active hits each opp mon) onto the OPP tokens "
                             "via a zero-init outgoing_proj, riding the SAME between-layers refine loop — so "
                             "attention reasons over BOTH threat directions, not just incoming. STRUCTURAL "
                             "(version-checked, fresh-only). REQUIRES --damage-op AND --damage-refine-rounds>0. "
                             "Default off (byte-identical).")
    parser.add_argument("--threat-unrevealed-outgoing", "--threat_unrevealed_outgoing",
                        dest="threat_unrevealed_outgoing", action=BoolFlag, default=None,
                        help="#2 EXPECTED-LATENT defender: price the outgoing residual's UNREVEALED opp columns by "
                             "marginalizing the move-belief's P(species) through SPECIES_EXP_MULT (type chart × "
                             "expected ability immunity — Levitate/Water&Volt Absorb/Flash Fire) + SPECIES_SPREAD_"
                             "PRIOR (E[bulk]); P(KO) NULLED (a full-HP switch-in is ~never OHKO'd). FORWARD-behavior "
                             "(version-checked, fresh-only). REQUIRES --threat-refine-outgoing (+ a belief head, "
                             "--opp-belief-aux-coef>0, for P(species)). Default off.")
    parser.add_argument("--threat-prob-outspeed", "--threat_prob_outspeed", dest="threat_prob_outspeed",
                        action=BoolFlag, default=None,
                        help="#3 UNCERTAINTY-AWARE P(outspeed): divide the speed gap by the believed speed STD "
                             "(SPECIES_SPREAD_PRIOR; sigmoid≈normal-CDF) instead of a fixed scale — a high-variance "
                             "opp speed reads ~0.5, a pinned one reads sharp. FORWARD-behavior (version-checked, "
                             "fresh-only). REQUIRES --damage-op. Default off (byte-identical).")
    parser.add_argument("--threat-status-refine", "--threat_status_refine", dest="threat_status_refine",
                        action=BoolFlag, default=None,
                        help="STATUS-LANDING into the TRUNK (gen3_status_trunk_v1, the last CPU-obs deprecation "
                             "gap): two zero-init residuals riding the refine loop — INCOMING ('will I be "
                             "statused' onto OUR tokens, from the opp active's believed status moves) + OUTGOING "
                             "('can I status this opp mon' onto OPP tokens, revealed-gated, from our status moves), "
                             "each [P(major), P(immobilize=para/frz/slp)] computed by reusing the v27 status-landing "
                             "physics (type × ability × already × Sleep-Clause × Substitute). Status immunity is a "
                             "computed MECHANICS fact handed over (not learned across non-local tokens) — completes "
                             "the FULL --unified-obs deprecation. STRUCTURAL (version-checked, fresh-only). REQUIRES "
                             "--damage-op AND --damage-refine-rounds>0. Default off (byte-identical).")
    parser.add_argument("--spread-belief", "--spread_belief", dest="spread_belief",
                        action=BoolFlag, default=None,
                        help="SpreadBelief (gen3_unified_spread_belief_v1): the THIRD belief leg — predict "
                             "the opponent's hidden SPREAD (the 5 derived stats atk/def/spa/spd/spe) per "
                             "slot from a usage PRIOR + a learned head, reinject into the opp token, and "
                             "feed the DamageOperator so it consumes BELIEVED opp stats instead of its "
                             "hand-coded de-timid/neutral constants (offense, bulk, speed). STRUCTURAL "
                             "(version-checked, fresh-only). Off by default.")
    parser.add_argument("--spread-belief-coef", "--spread_belief_coef", dest="spread_belief_coef",
                        type=float, default=None,
                        help="Spread-belief SUPERVISION weight (gen3_unified_spread_belief_v1): coef * "
                             "smooth_l1(believed derived stats {atk,def,spa,spd,spe}, TRUE derived stats) "
                             "over the REVEALED opp slots, so the SpreadBelief head LEARNS the opponent's "
                             "hidden EV spread (privileged training-only label from agent2's own team) "
                             "instead of sitting at the usage-mean prior (which over-estimates the largest-EV "
                             "stat → mis-priced damage/outspeed). The DamageOperator then prices damage "
                             "against the opponent's REAL bulk/offense/speed. 0.0 = OFF (byte-identical loss; "
                             "the head gets only the indirect op-damage gradient). REQUIRES --spread-belief. "
                             "TRAINING-only (not version-locked); metrics ride belief/spread_* "
                             "(mae, largest_bias→0, n_slots).")
    parser.add_argument("--spread-belief-nature", "--spread_belief_nature", dest="spread_belief_nature",
                        action=BoolFlag, default=None,
                        help="NATURE/EV generative spread head (gen3_nature_ev_belief_v1): swap SpreadBelief's "
                             "additive point-estimate for a head that predicts a NATURE categorical ⊕ its "
                             "Smogon prior + per-stat EVs ⊕ their prior (prior-fusion), assumes IV 31, and "
                             "COMPUTES the derived stat. The nature coupling (one stat ×1.1, one ×0.9) + the EV "
                             "budget are STRUCTURAL → the head can't inflate every stat, fixing the "
                             "'over-estimates the largest EV' order-statistic bias at the source. Supervised by "
                             "nature CE + EV regression (privileged inverted label) folded at --spread-belief-coef; "
                             "metrics ride belief/natureev_* (nature_acc, ev_mae). STRUCTURAL (version-checked, "
                             "fresh-only). REQUIRES --spread-belief. Off by default.")
    parser.add_argument("--spread-belief-nature-marginalize", "--spread_belief_nature_marginalize",
                        dest="spread_belief_nature_marginalize", action=BoolFlag, default=None,
                        help="Op-side NATURE MARGINALIZATION (gen3_nature_ev_belief_v1): the DamageOperator "
                             "marginalises the nonlinear P(KO)/damage over the believed nature distribution "
                             "(compute-then-blend over the top natures) instead of using E[nature_mult] — "
                             "restores the ×1.1/×0.9 asymmetry in the KO threshold. FORWARD-BEHAVIOR "
                             "(version-checked, fresh-only). REQUIRES --spread-belief-nature. Off by default.")
    parser.add_argument("--hp-type-belief", "--hp_type_belief", dest="hp_type_belief_mode",
                        choices=["off", "prior", "learned"], default=None,
                        help="Opponent HIDDEN-POWER-TYPE belief + the typed-HP candidate FIX "
                             "(gen3_opp_hp_type_belief_v1) — fixes the DamageOperator showing the opp's "
                             "Hidden Power as 0-damage/'immune' (the bare typeless num-237 candidate out-ranked "
                             "the 16 typed rows + the obs hp_probs is empty until HP fires). 'off' = legacy "
                             "(the bug). 'prior' = mask the bare-237 + floor the typed-HP belief on the Smogon "
                             "HP-type prior (forward-behavior change, NO new params). 'learned' = ALSO add the "
                             "HPTypeBelief head (prior ⊕ learned delta), whose posterior the op consumes + the "
                             "aux CE supervises — the 'force the model to guess which Hidden Power it is' head, "
                             "so the top-K surfaces the 2-3 most-likely typed HPs with real damage. STRUCTURAL "
                             "(version-checked, fresh-only); REQUIRES --damage-op. Off by default.")
    parser.add_argument("--hp-type-belief-coef", "--hp_type_belief_coef", dest="hp_type_belief_coef",
                        type=float, default=None,
                        help="HP-type-belief SUPERVISION weight (gen3_opp_hp_type_belief_v1): coef * "
                             "cross_entropy(HPTypeBelief posterior, TRUE opp HP type) over the REVEALED opp "
                             "slots that run Hidden Power (privileged training-only label from agent2's team — "
                             "Gen 3 never reveals the opp HP type). 0.0 = OFF (the head gets only the indirect "
                             "op-damage gradient + sits at the Smogon prior). Only meaningful with "
                             "--hp-type-belief learned. TRAINING-only (not version-locked); metrics ride "
                             "belief/hptype_* (acc, n_slots). Suggested 0.05.")
    parser.add_argument("--unified-obs", "--unified_obs", dest="unified_obs",
                        action=BoolFlag, default=False,
                        help="DISABLE the redundant CPU obs blocks the unified GPU path now subsumes (ONE "
                             "master switch): zeros the incoming-damage block (→ --damage-op), the "
                             "active-move power/multiplier scalars (→ the op's outgoing block, so requires "
                             "--unified-damage both), and the 44-dim move-effect block (→ MOVE_ATTR/the move "
                             "latent + the op effect axes). Each region stays in the obs vector (dim "
                             "unchanged); the reward PBRS still reads them. Pair with --unified-moves both + "
                             "--spread-belief to run pure-unified. Granular --mask-*-obs flags underneath.")
    parser.add_argument("--mask-active-move-scalars-obs", "--mask_active_move_scalars_obs",
                        dest="mask_active_move_scalars_obs", action=BoolFlag, default=None,
                        help="Granular: zero the active-move power+multiplier scalars from the model's view "
                             "(subsumed by the op's outgoing block; requires --damage-outgoing). Part of "
                             "--unified-obs.")
    parser.add_argument("--mask-move-effects-obs", "--mask_move_effects_obs",
                        dest="mask_move_effects_obs", action=BoolFlag, default=None,
                        help="Granular: zero the 44-dim move-effect block from the model's view (subsumed "
                             "by MOVE_ATTR/the move latent + the op effect axes; pair with --move-latent + "
                             "--damage-op). Part of --unified-obs.")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per environment per rollout")
    parser.add_argument("--grad-checkpointing", "--grad_checkpointing", dest="grad_checkpointing",
                        action=BoolFlag, default=False,
                        help="Gradient-checkpoint the transformer encoder layers during the PPO "
                             "update (bit-exact; trades one extra forward on the idle GPU for "
                             "~5GB less activation VRAM). Off by default; safe to toggle per run.")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="AdamW weight decay (L2 regularisation). Default 1e-5 is conservative for PPO.")

    # --- Subprocess eval ---
    parser.add_argument("--eval-workers", "--eval_workers", dest="eval_workers", type=int, default=5,
                        help="Number of parallel eval-worker subprocesses per cycle (default 5 for bot "
                             "eval; self-play doubles this to 10). Workers work-steal opponents from a "
                             "shared pool, so uneven per-opponent cost self-balances. Capped at the "
                             "opponent count.")
    parser.add_argument("--eval-device", "--eval_device", dest="eval_device", type=str, default="cpu",
                        help="Device for the eval-worker subprocess inference (default cpu, to decouple from the training GPU).")
    parser.add_argument("--eval-concurrency-per-worker", "--eval_concurrency_per_worker",
                        dest="eval_concurrency_per_worker", type=int, default=_EVAL_SUBPROCESS_CONCURRENCY,
                        help="Battles each eval worker overlaps at once within its claimed opponent (default 1 = "
                             "sequential). Single-thread asyncio latency-hiding (not multi-core): overlaps the "
                             "bridge/server I/O wait with other battles' forwards. A single-core bridge benchmark "
                             "measured ~2x decisions/sec at 3 on spare cores (less under live training contention); "
                             "the plateau is ~3. Cross-opponent parallelism is still --eval-workers.")
    parser.add_argument("--eval-shard-games", "--eval_shard_games",
                        dest="eval_shard_games", type=int, default=EVAL_SHARD_GAMES,
                        help="Games per work-steal shard unit (battle-level work-stealing, default 25 → ~4 shards "
                             "per opponent). Each opponent's eval games split into chunks any idle worker can drain, "
                             "so one straggler no longer pins a whole opponent on a single worker — the long tail "
                             "collapses to one shard. Smaller = finer tail collapse but more player builds / (on "
                             "websocket) more connection churn; the in-process bridge (--use-showdown-bridge) is "
                             "preferred for fine shards. >= the per-opponent game count disables sharding (one shard "
                             "per opponent = the original opponent-level behaviour).")
    parser.add_argument("--keep-eval-snapshots", "--keep_eval_snapshots", dest="keep_eval_snapshots",
                        type=int, default=10,
                        help="Retain the N most-recent eval weight snapshots in eval_traces/step_<N>/snapshot.zip "
                             "so the prober can reload the bit-exact model that produced a cycle's traces "
                             "(~27MB each; default 10 ≈ 270MB). 0 only writes the identity manifest; the prober "
                             "then falls back to the nearest persisted checkpoint.")
    parser.add_argument("--keep-eval-trace-steps", "--keep_eval_trace_steps", dest="keep_eval_trace_steps",
                        type=int, default=20,
                        help="The trainer grooms the forensic traces it writes: after each eval cycle it "
                             "keeps only the N most-recent eval step dirs under eval_traces/ (0 = keep all). "
                             "`python -m main.prober.groom` is the manual fallback for finished runs.")
    parser.add_argument("--keep-stalls", "--keep_stalls", dest="keep_stalls", type=int, default=50,
                        help="Bound the run's stalls/ dir: each eval cycle keep only the N most-recent "
                             "stall_*.html replays (0 = keep all). `python -m agents.training.artifact_retention` "
                             "is the manual fallback / cross-run sweep.")
    parser.add_argument("--keep-crashes", "--keep_crashes", dest="keep_crashes", type=int, default=10,
                        help="Bound the run's crashes/ dir: each eval cycle keep only the N most-recent "
                             "launcher restart_err_*.txt files (0 = keep all).")
    parser.add_argument("--self-play", action=BoolFlag, default=False, help="Enable self-play snapshot pool as training opponents")
    parser.add_argument("--distill-opponents", "--distill_opponents", dest="distill_opponents",
                        action=BoolFlag, default=False,
                        help="Distill self-play opponents into a cheaper network for faster rollouts "
                             "(all-or-nothing: backfill the whole pool on enable, then atomic switch; "
                             "fail-closed gate + auto-revert). See designs/ai_v5/distill_integration.md.")
    parser.add_argument("--snapshot-dir", type=str, default=None, help="Pool directory (default: <run_dir>/snapshots)")
    parser.add_argument("--promote-threshold", type=float, default=None,
                        help="Win rate vs. pool to trigger snapshot promotion. Default 0.65 with "
                             "stochastic sentinels; auto-lowered to 0.55 under --eval-sentinel-greedy "
                             "(greedy-vs-greedy removes the temperature handicap, so a genuinely-ahead "
                             "trainee wins the pool by a smaller margin — 0.65 would freeze the pool). "
                             "An explicit value always wins.")
    parser.add_argument("--eval-sentinel-greedy", "--eval_sentinel_greedy", dest="eval_sentinel_greedy",
                        action=BoolFlag, default=False,
                        help="Eval the self-play pool sentinels GREEDY (argmax) instead of stochastic. "
                             "Removes the greedy-trainee-vs-stochastic-sentinel handicap so win_rate_vs_pool "
                             "/ snapshot ELO reflect real best-vs-best skill (≈50%% vs a recent self, ramping "
                             "with sentinel age) instead of a flat temperature offset. Eval-only — TRAINING "
                             "opponents stay stochastic. Metric discontinuity vs prior cycles; pair with the "
                             "auto-lowered --promote-threshold (0.55).")
    parser.add_argument("--self-play-temp", type=float, default=1.0,
                        help="Sampling temperature for self-play TRAINING opponents (they sample, "
                             "not argmax, so the learner faces the policy's full action distribution). "
                             "1.0 = the policy's own distribution; >1 flatter/more random; lower → toward "
                             "greedy. Eval opponents stay deterministic regardless.")
    # ── Bot-mix curriculum (#2): keep the coverage-punishing bots in the TRAINING mix ──
    parser.add_argument("--bot-weights", "--bot_weights", dest="bot_weights", type=str, default=None,
                        help="Bias the per-episode HEURISTIC opponent pick toward chosen archetypes, "
                             "e.g. 'aggressive_v2=3,heuristic2=3'. Unlisted bots default to weight 1.0. "
                             "Names: heuristic, heuristic2, staller, staller_v2, aggressive, aggressive_v2, "
                             "setup_sweep, setup_sweep_v2. Omitted → uniform (current behavior). Only biases "
                             "WHICH heuristic an episode draws; the pool-vs-heuristic fraction is unaffected.")
    parser.add_argument("--heuristic-floor", "--heuristic_floor", dest="heuristic_floor",
                        type=float, default=None,
                        help="Minimum fraction of training episodes vs real bots once self-play saturates "
                             f"(default {HEURISTIC_FLOOR:g}). Raise it (e.g. 0.25) to keep a bigger permanent "
                             "bot slice so the coverage blindspot keeps getting exercised under self-play.")
    parser.add_argument("--self-play-start-wr", "--self_play_start_wr", dest="self_play_start_wr",
                        type=float, default=None,
                        help=f"win_rate_vs_bots at which self-play begins to ramp in (default {SELF_PLAY_START:g}).")
    parser.add_argument("--self-play-full-wr", "--self_play_full_wr", dest="self_play_full_wr",
                        type=float, default=None,
                        help=f"win_rate_vs_bots at which self-play reaches the floor (default {SELF_PLAY_FULL:g}); "
                             "raise it to ramp slower / stay bot-heavier for longer.")
    # ── Stable (cross-run) opponents: load a model from ANOTHER run as a fixed opponent ──
    parser.add_argument("--stable-opponents", "--stable_opponents", dest="stable_opponents",
                        type=str, default=None,
                        help="Foreign model(s) from ANOTHER run to use as fixed eval opponents, "
                             "comma-separated. Simplest form is just the run dir: "
                             "'models/ai_v5_5_popart_N_0607' — the opponent is then labelled by that "
                             "dir name (ai_v5_5_popart_N_0607). Optional per-entry suffixes: "
                             "'@<step>' picks a specific checkpoint (default: best_model); "
                             "':<name>' renames it. (Per-opponent weights are NOT supported yet — "
                             "they only matter for the training mix, which is Stage 2.) Each model "
                             "must share this run's arch_signature (= observation layout) — a "
                             "mismatch is a startup FATAL surfaced to the TUI. Default None (off).")
    parser.add_argument("--stable-opponent-temp", "--stable_opponent_temp", dest="stable_opponent_temp",
                        type=float, default=1.0,
                        help="TRAINING-mix play temperature for stable opponents (default 1.0 = the "
                             "policy's own distribution). Stochastic (not greedy) so a fixed opponent "
                             "is a moving target — harder to over-exploit. (In EVAL they always play "
                             "greedy/temp-0 for a clean win-rate yardstick.)")
    parser.add_argument("--stable-opponent-mastered-wr", "--stable_opponent_mastered_wr",
                        dest="stable_opponent_mastered_wr", type=float, default=0.80,
                        help="Win rate at which a stable opponent is considered MASTERED and moves "
                             "from the challenge bucket (played alongside the self-play pool) to the "
                             "coverage floor (played alongside the bots) — it 'becomes another bot'. "
                             "Default 0.80. One-way per run. Only active under --self-play.")
    parser.add_argument("--stable-opponent-selfplay-share", "--stable_opponent_selfplay_share",
                        dest="stable_opponent_selfplay_share", type=float,
                        default=STABLE_CHALLENGE_SHARE,
                        help="Fraction of SELF-PLAY (challenge) episodes spent vs stable opponents — "
                             "the rest go to the self-play pool. Caps how much a fixed opponent "
                             "occupies training so a single one can't dominate; multiple un-mastered "
                             f"stable opponents SHARE this slice. Default {STABLE_CHALLENGE_SHARE:g}. "
                             "Only active under --self-play.")

    args = parser.parse_args()

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

    # --unified-moves is the umbrella over the WHOLE move system: it sets --unified-damage to the same
    # level (so the op/belief/outgoing desugar below runs) AND turns on the move latent + its grading.
    # Applied BEFORE the --unified-damage desugar so the level flows through. v24.
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

    # --unified-obs is the master DISABLE-redundant switch: flip on the three obs-ablation masks (each only
    # where the GPU path subsumes it — active-move scalars need the outgoing op). v25.
    if getattr(args, "unified_obs", False):
        if args.mask_incoming_damage_obs is None:
            args.mask_incoming_damage_obs = True
        if args.mask_active_move_scalars_obs is None:
            args.mask_active_move_scalars_obs = True
        if args.mask_move_effects_obs is None:
            args.mask_move_effects_obs = True

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
    _resolve("value_active_readout", False)
    _resolve("attend_unrevealed_opponents", False)
    _resolve("opp_belief_cls_k", 0)
    _resolve("opp_belief_aux_coef", 0.0)
    _resolve("move_belief_mode", "off")        # v17 structural (version-checked, fresh-only)
    _resolve("move_belief_coef", 0.0)          # training-only (inherited like opp_belief_aux_coef)
    _resolve("opp_belief_latent_coef", 0.0)    # training-only (inherited like opp_belief_aux_coef)
    _resolve("damage_op", False)               # v19 structural (version-checked, fresh-only)
    _resolve("damage_reattend", False)         # v31 structural (version-checked, fresh-only)
    _resolve("damage_outgoing", False)         # v23 structural (version-checked, fresh-only)
    _resolve("move_candidate_floor", 0.0)      # v23 forward-behavior (version-checked, fresh-only)
    _resolve("move_latent", False)             # v24 structural (version-checked, fresh-only)
    _resolve("move_belief_latent_coef", 0.0)   # training-only (inherited like move_belief_coef)
    _resolve("spread_belief", False)           # v25 structural (version-checked, fresh-only)
    _resolve("spread_belief_nature", False)    # v40 structural (version-checked, fresh-only)
    _resolve("spread_belief_nature_marginalize", False)  # v40 forward-behavior (version-checked, fresh-only)
    _resolve("spread_belief_coef", 0.0)        # training-only (inherited like move_belief_coef)
    _resolve("mask_active_move_scalars_obs", False)  # v25 forward-behavior (version-checked, fresh-only)
    _resolve("mask_move_effects_obs", False)         # v25 forward-behavior (version-checked, fresh-only)
    _resolve("move_prior_fusion", False)       # v20 forward-behavior (version-checked, fresh-only)
    _resolve("move_belief_prefuse", False)     # v32 forward-behavior (version-checked, fresh-only)
    _resolve("mask_incoming_damage_obs", False)  # v21 forward-behavior (version-checked, fresh-only)
    _resolve("win_prob_mode", "none")          # v22 structural + resume-immutable (version-checked)
    _resolve("win_prob_coef", 1.0)             # training-only (inherited like opp_belief_aux_coef)
    _resolve("value_dist_mode", "none")        # v29 structural + resume-immutable (version-checked)
    _resolve("value_dist_bins", 0)             # v29 structural (atom count; version-checked)
    _resolve("value_dist_vmin", 0.0)           # v29 resume-immutable support (version-checked)
    _resolve("value_dist_vmax", 0.0)           # v29 resume-immutable support (version-checked)
    _resolve("value_dist_coef", 1.0)           # training-only (inherited like win_prob_coef)
    _resolve("damage_topk_k", 0)               # v30 structural int (top-K incoming; version-checked, fresh-only)
    _resolve("damage_refine_rounds", 0)        # v31 structural int (iterative refine; version-checked, fresh-only)
    _resolve("damage_matrices_outgoing", False)  # v32 structural (outgoing damage matrix; version-checked, fresh-only)
    _resolve("damage_matrices_incoming", False)  # v33 structural (incoming damage matrix; version-checked, fresh-only)
    _resolve("damage_matrices_outgoing_all", False)  # v39 structural (transposed outgoing matrix; version-checked, fresh-only)
    _resolve("threat_refine_outgoing", False)    # v36 structural (outgoing→trunk; version-checked, fresh-only)
    _resolve("threat_unrevealed_outgoing", False)  # v36 forward-behavior (expected-latent; version-checked, fresh-only)
    _resolve("threat_prob_outspeed", False)      # v36 forward-behavior (prob outspeed; version-checked, fresh-only)
    _resolve("threat_status_refine", False)      # v37 structural (status→trunk; version-checked, fresh-only)
    _resolve("hp_type_belief_mode", "off")     # v38 structural + resume-immutable (version-checked, fresh-only)
    _resolve("hp_type_belief_coef", 0.0)       # training-only (inherited like spread_belief_coef)
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
    if args.opp_belief_cls_k < 0:
        parser.error("--opp-belief-cls-k must be >= 0 (0 = off)")
    if args.opp_belief_cls_k > 0 and not args.attend_unrevealed_opponents:
        parser.error(
            "--opp-belief-cls-k > 0 requires --attend-unrevealed-opponents — the hidden-opponent belief "
            "queries read the unrevealed opp slots, which are key-masked unless the unmask flag is on. "
            "Add --attend-unrevealed-opponents (or set --opp-belief-cls-k 0)."
        )
    if args.opp_belief_aux_coef < 0.0:
        parser.error("--opp-belief-aux-coef must be >= 0 (0 = off)")
    if args.opp_belief_aux_coef > 0.0:
        # coef>0 turns on the in-place BeliefHead (a weight-shape change) which REQUIRES the unmask
        # flag (the believed slots must be attendable to be refined). Auto-enable it so a single flag
        # suffices; the model side hard-gates opp_belief_slots on attend_unrevealed_opponents.
        args.attend_unrevealed_opponents = True
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
    if args.move_belief_mode != "off":
        # The MoveBelief module reads/refines the opp slots, so (like the BeliefHead) it requires the
        # unrevealed slots to be attendable — auto-enable the unmask flag (the model side hard-gates
        # move_belief_mode!=off on attend_unrevealed_opponents).
        args.attend_unrevealed_opponents = True
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
    if args.opp_belief_latent_coef is not None and args.opp_belief_latent_coef < 0.0:
        parser.error("--opp-belief-latent-coef must be >= 0 (0 = off)")
    if args.opp_belief_latent_coef > 0.0 and not (args.opp_belief_aux_coef > 0.0):
        # The latent head attaches to the BeliefHead over the believed slots AND rides the species-CE
        # Hungarian assignment (computed only when --opp-belief-aux-coef>0). Without it there is no
        # species head, no believed-slot fill, and no per-minibatch assignment to match the latent on.
        parser.error(
            "--opp-belief-latent-coef > 0 requires --opp-belief-aux-coef > 0 — the latent predictor "
            "attaches to the BeliefHead and reuses its Hungarian slot↔mon assignment. Enable "
            "--opp-belief-aux-coef <coef> (>0), or set --opp-belief-latent-coef 0."
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
    if args.move_belief_prefuse and args.move_belief_mode == "off":
        # FAIL LOUD: prefuse moves the move-belief REINJECTION before the transformer; with no head
        # (--move-belief-mode off) there is no reinjection to move.
        parser.error(
            "--move-belief-prefuse requires --move-belief-mode != off (revealed|unrevealed|both): it "
            "moves the move-belief reinjection before the transformer. Set --move-belief-mode revealed, "
            "or drop --move-belief-prefuse."
        )
    if args.damage_outgoing and not args.damage_op:
        # The outgoing per-move block is emitted by the DamageOperator → the op must exist.
        parser.error(
            "--damage-outgoing requires --damage-op (the outgoing block is part of the damage operator). "
            "Use --unified-damage both, or add --damage-op."
        )
    if args.damage_reattend and not args.damage_op:
        # The re-attend layer reads the operator's per-mon incoming-damage block → the op must exist.
        parser.error(
            "--damage-reattend requires --damage-op (the re-attend layer reads the operator's incoming "
            "damage block). Use --unified-damage (with --damage-op), or add --damage-op, or drop "
            "--damage-reattend."
        )
    if args.damage_reattend and not args.use_popart:
        # SOFT warn (not a hard error — reattend runs without PopArt, just riskier): the extra shared-trunk
        # layer routes the value gradient through more of the trunk, which the value loss already dominates.
        print("⚠ --damage-reattend without --use-popart: the extra shared-trunk re-attend layer worsens the "
              "value-gradient contention on the trunk (the value MSE already swamps it at γ≈0.9999). PopArt "
              "is strongly recommended — add --use-popart and watch grad/value_policy_logratio.", file=sys.stderr)
    if args.move_candidate_floor and not args.move_prior_fusion:
        # The learnset/rarity gate prunes the FUSED prior; with no prior fusion there is no prior to gate.
        parser.error(
            "--move-candidate-floor requires --move-prior-fusion (it gates the fused move prior). "
            "Enable --move-prior-fusion (or --unified-damage), or drop --move-candidate-floor."
        )
    if args.damage_topk_k and args.damage_topk_k > 0 and not args.damage_op:
        # gen3_unified_topk_incoming_v1: the top-K incoming block extends the DamageOperator.
        parser.error(
            "--damage-topk requires --damage-op (the top-K incoming block extends the damage operator). "
            "Use --unified-damage / --unified-moves, or add --damage-op, or set --damage-topk 0."
        )
    if args.damage_topk_k and args.damage_topk_k > 0 and not args.move_latent:
        # The block gathers each top-K move's identity LATENT from the MoveLatentEncoder.
        parser.error(
            "--damage-topk requires --move-latent (the top-K block gathers each move's identity latent "
            "from the MoveLatentEncoder). Use --unified-moves, or add --move-latent, or set --damage-topk 0."
        )
    if args.damage_refine_rounds and args.damage_refine_rounds > 0 and not args.damage_op:
        # gen3_iterative_damage_v1: the refinement recomputes the DamageOperator's lean incoming damage
        # between transformer layers (and re-reads the move belief, which --damage-op requires).
        parser.error(
            "--damage-refine-rounds requires --damage-op (the iterative refinement recomputes the damage "
            "operator's lean incoming threat between transformer layers). Use --unified-damage / "
            "--unified-moves, or add --damage-op, or set --damage-refine-rounds 0."
        )
    if getattr(args, "damage_matrices_outgoing", False) and not args.damage_op:
        # gen3_per_move_matrices_v1: the outgoing damage matrix is emitted by the DamageOperator.
        parser.error(
            "--damage-matrices outgoing requires --damage-op (the matrix is emitted by the damage operator). "
            "Use --unified-damage / --unified-moves, or add --damage-op, or set --damage-matrices off."
        )
    if getattr(args, "damage_matrices_outgoing_all", False) and not args.damage_op:
        # gen3_per_move_matrices_v1 (v39): the TRANSPOSED outgoing matrix is emitted by the DamageOperator.
        parser.error(
            "--damage-matrices-outgoing-all requires --damage-op (the matrix is emitted by the damage operator). "
            "Use --unified-damage / --unified-moves, or add --damage-op, or drop --damage-matrices-outgoing-all."
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
    # gen3_bidir_threat_trunk_v1 (v36): the bidirectional in-trunk threat field.
    if getattr(args, "threat_refine_outgoing", False):
        if not args.damage_op:
            parser.error(
                "--threat-refine-outgoing requires --damage-op (the outgoing physics is the damage operator). "
                "Use --unified-damage / --unified-moves, or add --damage-op."
            )
        if not (args.damage_refine_rounds and args.damage_refine_rounds > 0):
            parser.error(
                "--threat-refine-outgoing requires --damage-refine-rounds>0 — the outgoing residual rides the "
                "SAME between-layers refine loop. Set --damage-refine-rounds N."
            )
    if getattr(args, "threat_unrevealed_outgoing", False):
        if not getattr(args, "threat_refine_outgoing", False):
            parser.error(
                "--threat-unrevealed-outgoing requires --threat-refine-outgoing (it only enriches the outgoing "
                "residual's UNREVEALED columns with the expected-latent defender)."
            )
        if not (args.opp_belief_aux_coef and args.opp_belief_aux_coef > 0):
            parser.error(
                "--threat-unrevealed-outgoing requires --opp-belief-aux-coef>0 — the expected-latent defender "
                "reads P(species) from the hidden-opponent belief head (BeliefHead.species_logits)."
            )
    if getattr(args, "threat_prob_outspeed", False) and not args.damage_op:
        parser.error(
            "--threat-prob-outspeed requires --damage-op (the P(outspeed) feature lives in the damage operator)."
        )
    if getattr(args, "threat_status_refine", False):
        if not args.damage_op:
            parser.error(
                "--threat-status-refine requires --damage-op (the status-landing physics is the damage operator). "
                "Use --unified-damage / --unified-moves, or add --damage-op."
            )
        if not (args.damage_refine_rounds and args.damage_refine_rounds > 0):
            parser.error(
                "--threat-status-refine requires --damage-refine-rounds>0 — the status residuals ride the SAME "
                "between-layers refine loop. Set --damage-refine-rounds N."
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
    if args.spread_belief_nature_marginalize and not args.spread_belief_nature:
        # The op marginalises over the NATURE distribution the generative head produces → that head must be on.
        parser.error(
            "--spread-belief-nature-marginalize requires --spread-belief-nature (the op marginalises over the "
            "generative head's nature distribution). Enable --spread-belief-nature, or drop the flag."
        )
    if args.hp_type_belief_mode != "off" and not args.damage_op:
        # The typed-HP candidates the fix masks/floors live in the DamageOperator (also enforced at the
        # extractor build — this is the friendlier CLI message).
        parser.error(
            "--hp-type-belief != off requires --damage-op (the typed-HP candidates it fixes are the "
            "DamageOperator's). Add --damage-op (--unified-damage), or set --hp-type-belief off."
        )
    if args.hp_type_belief_coef and args.hp_type_belief_mode != "learned":
        # The CE supervises the HPTypeBelief head's posterior (last_hp_type_logits) → the head must exist.
        parser.error(
            "--hp-type-belief-coef requires --hp-type-belief learned (it supervises the learned HP-type "
            "head). Set --hp-type-belief learned, or set --hp-type-belief-coef 0."
        )
    if args.mask_active_move_scalars_obs and not args.damage_outgoing:
        # Zeroing the active-move power/multiplier scalars only makes sense once the op's OUTGOING block
        # replaces them; without it the model loses the per-move signal with no substitute.
        parser.error(
            "--mask-active-move-scalars-obs requires --damage-outgoing (--unified-damage both): the op's "
            "outgoing per-move damage is what replaces the zeroed obs scalars. Add --unified-damage both, "
            "or drop --mask-active-move-scalars-obs / --unified-obs."
        )
    if args.mask_incoming_damage_obs and not args.damage_op:
        # The 51-dim incoming-damage/OHKO block is subsumed by the differentiable DamageOperator's incoming
        # rolls; masking it without the op leaves the model with NO incoming-damage signal at all.
        parser.error(
            "--mask-incoming-damage-obs requires --damage-op (--unified-damage): the op's incoming damage "
            "block is what replaces the zeroed obs block. Add --unified-damage, or drop "
            "--mask-incoming-damage-obs / --unified-obs."
        )
    if args.mask_move_effects_obs and not args.move_latent:
        # The 44-dim per-OUR-move effect block's STRUCTURAL identity (is_boost/heal/protect/phaze/hazard,
        # cures_self/team, per-status secondary chances) is carried into the model only via the move
        # latent's MOVE_ATTR; masking it without --move-latent erases that signal with no substitute.
        parser.error(
            "--mask-move-effects-obs requires --move-latent (--unified-moves): the move latent's MOVE_ATTR "
            "is what carries the per-move effect identity once the obs block is zeroed. Add --unified-moves, "
            "or drop --mask-move-effects-obs / --unified-obs."
        )
    if args.mask_move_effects_obs and not args.damage_outgoing:
        # The block also carried `status_will_land`; its GPU replacement is the op's OUTGOING status-landing
        # block (gen3_unified_status_landing_v1), which only exists with the outgoing direction. Without it
        # the model would lose the "will my Toxic/WoW/Spore/Leech Seed land" signal entirely.
        parser.error(
            "--mask-move-effects-obs requires --damage-outgoing (--unified-damage both / --unified-moves "
            "both): the op's outgoing status-landing block is what replaces the zeroed `status_will_land`. "
            "Add --unified-damage both, or drop --mask-move-effects-obs / --unified-obs."
        )
    if args.mask_move_effects_obs:
        # With --move-latent + --damage-outgoing the structural identity (MOVE_ATTR) AND status_will_land
        # (the op status-landing block, incl. Sleep Clause + Leech Seed + Substitute) are GPU-replaced. The
        # remaining UNCOVERED residual: the recovery MAGNITUDE / Rest-cure detail (the op effect block carries
        # a single recovery scalar), Yawn (delayed sleep), and a Leech-Seed-already-seeded target. Note it.
        print("[NOTE] --mask-move-effects-obs: status_will_land is now GPU-replaced (op status-landing block, "
              "incl. Sleep Clause + Leech Seed + Substitute). Residual uncovered: recovery magnitude/Rest-cure, "
              "Yawn, Leech-Seed-already-seeded.")
    log_level = LogLevel[args.log_level.upper()]

    # One server config, built from --showdown-port and threaded to every Showdown client
    # (training-env players in spawn workers, eval, and self-play). Default port: 8000.
    server_config = (
        LocalhostServerConfiguration
        if args.showdown_port is None
        else localhost_server_configuration(args.showdown_port)
    )
    if args.use_showdown_bridge:
        emit("🌉 Transport: in-process BattleStream bridge for BOTH training and eval "
             "(no Showdown server needed — --showdown-port ignored)")
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

    # Load all teams using the new TeamLoader
    loader = TeamLoader()
    sample_teams = loader.get_sample_teams()
    all_teams = loader.get_all_teams()
    
    emit(f"📦 {len(sample_teams)} sample teams (bias) / {len(all_teams)} total loaded")

    # Trainee draws from the full pool, but 50% of the time uses a sample team.
    # This exposes the agent to diverse team compositions while keeping a stable anchor.
    trainee_teambuilder = Gen3Teambuilder(all_teams, bias_teams=sample_teams, bias_prob=0.1)
    opponent_teambuilder = Gen3Teambuilder(all_teams)

    mappings = load_mappings()
    
    # Training heuristic opponents — ALL eight archetype bots (both v1 and v2 of each).
    # They play differently and the extra playstyle diversity is the point. Random is NOT
    # here (it's the eval-only "is the model broken" floor).
    OPPONENT_CLASSES = [
        SimpleHeuristicsPlayer,
        Gen3HeuristicV2Player,
        Gen3StallerPlayer,
        Gen3StallerV2Player,
        Gen3AggressivePlayer,
        Gen3AggressiveV2Player,
        Gen3SetupSweepPlayer,
        Gen3SetupSweepV2Player,
    ]
    print(f"[Opponents] training pool = {len(OPPONENT_CLASSES)} bots "
          f"({', '.join(opponent_name(c) for c in OPPONENT_CLASSES)})")

    # Resolve --bot-weights (name=weight) into a roster-aligned vector (unlisted → 1.0). None →
    # uniform (current behavior, byte-for-byte). Validated here so a typo fails fast at startup.
    _bot_weight_vec = None
    if args.bot_weights:
        _overrides = {}
        for tok in args.bot_weights.split(","):
            if not tok.strip():
                continue
            name, sep, val = tok.partition("=")
            if not sep:
                print(f"[Opponents] ERROR: --bot-weights token '{tok}' is not name=weight")
                sys.exit(1)
            _overrides[name.strip()] = float(val)
        _valid = {opponent_name(c) for c in OPPONENT_CLASSES}
        _bad = set(_overrides) - _valid
        if _bad:
            print(f"[Opponents] ERROR: unknown --bot-weights names {sorted(_bad)} "
                  f"(valid: {sorted(_valid)})")
            sys.exit(1)
        _bot_weight_vec = [_overrides.get(opponent_name(c), 1.0) for c in OPPONENT_CLASSES]
        print(f"[Opponents] heuristic weights = "
              f"{ {opponent_name(c): w for c, w in zip(OPPONENT_CLASSES, _bot_weight_vec)} }")

    # Resolve + VALIDATE --stable-opponents (cross-run fixed opponents) at startup. Each foreign
    # model must share THIS run's arch_signature (= observation layout) — a mismatch is a
    # NON-RECOVERABLE config error: exit FATAL_CONFIG so the launcher gives up immediately (the
    # same path a checkpoint arch mismatch takes) and the TUI shows the fatal, instead of
    # auto-restarting into the identical failure.
    _fixed_opponents = []
    if args.stable_opponents:
        from agents.training.fixed_opponent_pool import resolve_stable_opponents
        from agents.model.snapshot import (
            current_model_version as _current_model_version, load_foreign_opponent)
        _cv_stable = _current_model_version(mappings, **_run_arch_toggles(args))
        try:
            _fixed_opponents = resolve_stable_opponents(
                args.stable_opponents, _cv_stable, default_temperature=args.stable_opponent_temp,
            )
            # Validate the WEIGHTS actually load here in the main process (resolve only reads the
            # config). A valid config + corrupt/unreadable zip would otherwise pass the gate and
            # crash every env worker → crash-restart loop. Load once on CPU and discard.
            for _e in _fixed_opponents:
                load_foreign_opponent(_e.zip_path, current_version=_cv_stable, device="cpu",
                                      config_path=_e.config_path)
        except (ModelVersionError, FileNotFoundError, ValueError) as e:
            print(f"\n[StableOpponent] FATAL: {e}")
            sys.stdout.flush()  # os._exit() skips buffer flushing — make sure the reason reaches the log
            os._exit(int(TrainExitCode.FATAL_CONFIG))
        except Exception as e:  # noqa: BLE001 — a corrupt/unreadable foreign weights zip
            print(f"\n[StableOpponent] FATAL: failed to load stable opponent weights: {e}")
            sys.stdout.flush()
            os._exit(int(TrainExitCode.FATAL_CONFIG))
        # emit() → the launcher Events panel (like the [SELFPLAY] startup lines); print()s standalone.
        _stable_labels = ", ".join(e.label for e in _fixed_opponents)
        if args.self_play:
            emit(f"🐴 [STABLE] {len(_fixed_opponents)} cross-run opponent(s): {_stable_labels} — "
                 f"eval greedy; training ≤{args.stable_opponent_selfplay_share:.0%} of self-play until "
                 f"mastered (win_rate ≥ {args.stable_opponent_mastered_wr:.0%})")
        else:
            emit(f"🐴 [STABLE] {len(_fixed_opponents)} cross-run opponent(s): {_stable_labels} — "
                 "EVAL-ONLY (no --self-play, so they don't join the training mix)")

    # Curriculum (transition + floor) effective values: CLI override or the module defaults.
    _heuristic_floor = args.heuristic_floor if args.heuristic_floor is not None else HEURISTIC_FLOOR
    _sp_start_wr = args.self_play_start_wr if args.self_play_start_wr is not None else SELF_PLAY_START
    _sp_full_wr = args.self_play_full_wr if args.self_play_full_wr is not None else SELF_PLAY_FULL
    if (_heuristic_floor, _sp_start_wr, _sp_full_wr) != (HEURISTIC_FLOOR, SELF_PLAY_START, SELF_PLAY_FULL):
        print(f"[Opponents] self-play curriculum: start_wr={_sp_start_wr:g} full_wr={_sp_full_wr:g} "
              f"heuristic_floor={_heuristic_floor:g} "
              f"(defaults {SELF_PLAY_START:g}/{SELF_PLAY_FULL:g}/{HEURISTIC_FLOOR:g})")

    # Promotion gate: regime-aware default — 0.55 under greedy sentinels (the temperature handicap
    # is gone, so a genuinely-ahead trainee wins the pool by a smaller margin and 0.65 would freeze
    # the pool), else the original 0.65. An explicit --promote-threshold always wins.
    _promote_threshold = (args.promote_threshold if args.promote_threshold is not None
                          else (0.55 if args.eval_sentinel_greedy else 0.65))
    if args.eval_sentinel_greedy:
        print(f"[Opponents] eval sentinels GREEDY (best-vs-best pool/ELO signal) — "
              f"promote_threshold={_promote_threshold:g}")

    def create_training_env_random(idx, stall_config=None, opponent_device="auto",
                                   opponent_version=None, snapshot_dir=None,
                                   self_play_fraction=0.0, self_play=False,
                                   heuristic_weights=None, stable_opponents=None):
        def _init():
            try:
                ts = datetime.now().strftime('%H%M%S')
                env_username = f"RLAgent{idx}{ts}"

                env_log_level = log_level if idx == 0 else LogLevel.QUIET

                env = Gen3Env(
                    mappings,
                    battle_format=BATTLE_FORMAT,
                    team=trainee_teambuilder,
                    log_level=env_log_level,
                    stall_config=stall_config,
                    reward_fn=reward_factory(log_level=env_log_level),
                    server_configuration=server_config,
                    account_configuration1=AccountConfiguration(env_username, "password"),
                    # Bridge mode: don't open websockets — the in-process sim is the transport.
                    start_listening=not args.use_showdown_bridge,
                    # TRAINING-only privileged belief labels (only the trainee env; the model side
                    # gates the BeliefHead on the same coef>0 signal). Eval/self-play opponents play
                    # via RLPlayer, not Gen3Env, so they never emit them.
                    emit_belief_labels=(args.opp_belief_aux_coef > 0.0),
                    move_belief_mode=args.move_belief_mode,
                    emit_belief_target=(args.opp_belief_latent_coef > 0.0),
                    emit_win_target=(args.win_prob_mode != "none"),
                    # SPREAD-belief supervision (gen3_unified_spread_belief_v1): emit the privileged
                    # true-spread label only when the loss will consume it (coef>0; the CLI guards that
                    # --spread-belief-coef requires --spread-belief, so the head is present to supervise).
                    emit_spread_labels=(args.spread_belief and args.spread_belief_coef > 0.0),
                    # HP-TYPE-belief supervision (gen3_opp_hp_type_belief_v1): emit the privileged true-HP-
                    # type label only under the learned head + a non-zero CE coef (the CLI guards both).
                    emit_hp_type_labels=(args.hp_type_belief_mode == "learned" and args.hp_type_belief_coef > 0.0),
                    # DEFENSIVE-exploration flag (gen3_defensive_entropy_v1): emit only when the boost is on, so
                    # the state-conditioned entropy term in the PPO loss can read it. Off = no key, no cost.
                    emit_defensive_opportunity=(args.defensive_entropy_boost > 1.0),
                )
                if args.use_showdown_bridge:
                    # Swap the two _EnvPlayer agents' websocket transport for a local
                    # BattleStream subprocess. Everything above the transport (obs, reward,
                    # mask, wrappers) is unchanged — see utils/bridge/bridge_session.py.
                    attach_bridge_transport(env, battle_format=BATTLE_FORMAT)

                # Opponents are pure DECISION FUNCTIONS over env.battle2 (env.agent1/agent2 do
                # the networking), so build them start_listening=False — no idle connections,
                # and we can hold several per env for live per-episode selection. The heuristic
                # roster is always built; self-play adds a per-worker pool + one reusable pool
                # RLPlayer whose .model is swapped per episode (see MaskableAgentWrapper).
                heuristic_opponents = [
                    cls(
                        battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                        server_configuration=server_config,
                        account_configuration=AccountConfiguration(f"Opp{idx}h{i}{ts}", "password"),
                        start_listening=False,
                    )
                    for i, cls in enumerate(OPPONENT_CLASSES)
                ]

                pool = pool_player = None
                if self_play and snapshot_dir is not None:
                    pool = SnapshotPool(
                        pool_dir=snapshot_dir, current_version=opponent_version,
                        device=opponent_device,
                    )
                    # Distilled opponents are rebuilt from the obs layout on load (env-side).
                    if getattr(args, "distill_opponents", False):
                        pool.set_distill_layout(Gen3ObservationEncoder(mappings).get_layout())
                    # model=None placeholder — the wrapper swaps in a sampled snapshot before
                    # ever using it. Stochastic + temperature so the learner trains against the
                    # policy's full action distribution (richer, less exploitable than argmax).
                    # Strict (crash-over-corruption) on a stale decision: the launcher restarts.
                    pool_player = RLPlayer(
                        model=None, team=opponent_teambuilder, battle_format=BATTLE_FORMAT,
                        server_configuration=server_config, mappings=mappings,
                        account_configuration=AccountConfiguration(f"Opp{idx}p{ts}", "password"),
                        start_listening=False,
                        stochastic=True, temperature=args.self_play_temp,
                    )

                # Stable cross-run opponents — one reusable RLPlayer each, loaded ONCE per worker
                # (foreign models don't change, so no per-episode reload). They join the TRAINING
                # mix only under self-play (the challenge/pool bucket); each plays stochastically at
                # its temperature (harder to over-exploit). Un-mastered → challenge peer of the pool;
                # mastered (pushed via set_stable_mastered) → floor peer of the bots.
                stable_players, stable_labels = [], []
                if self_play and stable_opponents:
                    from agents.model.snapshot import load_foreign_opponent
                    for e in stable_opponents:
                        opp_model, _ = load_foreign_opponent(
                            e.zip_path, current_version=opponent_version,
                            device=opponent_device, config_path=e.config_path)
                        stable_players.append(RLPlayer(
                            model=opp_model, team=opponent_teambuilder, battle_format=BATTLE_FORMAT,
                            server_configuration=server_config, mappings=mappings,
                            account_configuration=AccountConfiguration(
                                f"Opp{idx}s{len(stable_players)}{ts}", "password"),
                            start_listening=False,
                            stochastic=True, temperature=e.temperature,
                        ))
                        stable_labels.append(e.label)

                wrapped = MaskableAgentWrapper(
                    env, heuristic_opponents=heuristic_opponents, pool=pool,
                    pool_player=pool_player, self_play_fraction=self_play_fraction, rng_seed=idx,
                    heuristic_weights=heuristic_weights,
                    stable_players=stable_players, stable_labels=stable_labels,
                    stable_challenge_share=args.stable_opponent_selfplay_share,
                )

                # FORCE OVERRIDE: SingleAgentWrapper hardcodes 10 for gen3ou. We need 11.
                # Also ensure it propagates our Dict observation space natively.
                wrapped.action_space = env.action_space
                wrapped.observation_space = env.observation_space

                return Monitor(wrapped)
            except Exception as e:
                print(f"🛑 ERROR IN WORKER {idx}: {e}")
                traceback.print_exc()
                raise e
        return _init

    # --- Directory Setup ---
    if args.run_dir:
        model_dir = args.run_dir                                     # launcher-managed resume
    else:
        model_dir = f"models/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(model_dir, exist_ok=True)
    # Full CLI namespace (JSON-safe) → persisted into metadata.json for run provenance.
    cli_args = json.loads(json.dumps(vars(args), default=str))
    if not args.run_dir:
        with open(os.path.join(model_dir, "command.txt"), "w") as f:
            f.write(" ".join(sys.argv))
        
    stall_cfg = StallConfig(output_dir=os.path.join(model_dir, "stalls"))
    # Per-run reward config (design §1). gamma MUST == the PPO gamma (asserted post-build below); the
    # factory passes it to every env's reward manager. Default = the single-variable run.
    from agents.training.reward_manager import RewardConfig
    # Single construction site (gamma == InstrumentedMaskablePPO(gamma=0.9999), asserted below). Every
    # reward CLI flag flows in by name → training, eval, and the version record all use ONE config.
    reward_config = RewardConfig.from_args(args)
    reward_factory = functools.partial(Gen3RewardManager, config=reward_config)

    # Running parallel environments
    n_envs = 1 if args.debug else args.n_envs
    # --async-rollout swaps the barriered SubprocVecEnv for AsyncSubprocVecEnv (per-env in-flight
    # stepping + drain-safe env_method). Only when not --debug (DummyVecEnv has one env, no barrier).
    _async_rollout = args.async_rollout and not args.debug
    if args.debug:
        EnvClass = DummyVecEnv
    elif _async_rollout:
        EnvClass = AsyncSubprocVecEnv
    else:
        EnvClass = SubprocVecEnv

    emit(f"⚙️ Initializing {n_envs} envs ({EnvClass.__name__})"
         + (" — non-barrier async rollout" if _async_rollout else ""))

    _shutdown_event = threading.Event()

    # --- Self-Play Pool Setup ---
    # The pool is a directory the env workers read off disk (each builds its own SnapshotPool).
    # The heuristic-vs-pool split is NOT fixed per process anymore: every env picks its opponent
    # per-episode from a LIVE self_play_fraction that the eval callback updates each eval (see
    # MaskableAgentWrapper). The initial fraction comes from the persisted win rate (summary.json)
    # so a resumed run starts at the right ramp level instead of cold-starting at 0%.
    _pool: SnapshotPool | None = None
    _opp_version = None  # ModelVersion threaded into opponent snapshot loads (set when self-play on)
    _snapshot_dir = None
    _initial_self_play_fraction = 0.0
    if args.self_play:
        from pathlib import Path as _Path
        from agents.model.snapshot import current_model_version as _current_model_version

        _snapshot_dir = _Path(args.snapshot_dir) if args.snapshot_dir else _Path(model_dir) / "snapshots"
        _cv = _current_model_version(mappings, **_run_arch_toggles(args))
        _opp_version = _cv
        _pool = SnapshotPool(pool_dir=_snapshot_dir, current_version=_cv, device=args.device)
        if getattr(args, "distill_opponents", False):
            _pool.set_distill_layout(Gen3ObservationEncoder(mappings).get_layout())
        _persisted_wr = _pool.load_persisted_win_rate()
        _initial_self_play_fraction = 1.0 - heuristic_fraction(
            _persisted_wr, floor=_heuristic_floor, start=_sp_start_wr, full=_sp_full_wr)
        emit(
            f"🎮 [SELFPLAY] Pool has {len(_pool)} snapshots, win_rate_vs_bots={_persisted_wr:.2%} "
            f"→ self_play_fraction={_initial_self_play_fraction:.0%} (live, per-episode)"
        )

    opponent_device = "cpu" if args.self_play_use_cpu else args.device
    if args.self_play:
        emit(
            f"🧠 [SELFPLAY] Opponent snapshots load on '{opponent_device}' "
            f"({'CPU — avoids per-worker CUDA contexts' if args.self_play_use_cpu else 'training device'})"
        )

    def _make_factories():
        return [
            create_training_env_random(
                i, stall_config=stall_cfg, opponent_device=opponent_device,
                opponent_version=_opp_version,
                snapshot_dir=str(_snapshot_dir) if _snapshot_dir is not None else None,
                self_play_fraction=_initial_self_play_fraction, self_play=args.self_play,
                heuristic_weights=_bot_weight_vec,
                stable_opponents=_fixed_opponents,
            )
            for i in range(n_envs)
        ]

    env_factories = _make_factories()
    env = EnvClass(env_factories)
    # NOTE: the subprocess watchdog is started LATER, just before model.learn() — nothing steps
    # the env before then (model construction/load only reads its spaces).

    def _maybe_seed_pool(model):
        """Seed the pool from the loaded weights iff self-play is active (fraction>0 → win rate
        ≥ SELF_PLAY_START) AND the pool is empty — so the seed is captured from a *competent*
        model, never random/weak. No env rebuild: the worker pools re-scan the dir on demand
        (and lazily whenever they see it empty). The eval callback also seeds when the model
        first crosses the threshold mid-run."""
        if not (args.self_play and _pool is not None):
            return
        if _initial_self_play_fraction > 0 and _pool.is_empty():
            _pool.seed(model)
            emit(f"🌱 [SELFPLAY] Seeded pool from current weights "
                 f"(win rate ≥ threshold → self_play_fraction={_initial_self_play_fraction:.0%})")

    async def evaluate_model_random(model):
        ts = datetime.now().strftime('%H%M%S')
        n = args.eval_battles
        # Bridge eval: build every player start_listening=False and play in-process via
        # run_local_battles (no server). Same flag as training; lets --debug + bridge run serverless.
        _eval_sl = not args.use_showdown_bridge
        print(f"\nFinal Evaluation (Session {ts}, Battles: {n}, Concurrency: {args.eval_concurrency})...")

        rl_player = RLPlayer(
            model=model,
            team=trainee_teambuilder,
            battle_format=BATTLE_FORMAT,
            server_configuration=server_config,
            mappings=mappings,
            account_configuration=AccountConfiguration(f"RLFinal{ts}", "password"),
            max_concurrent_battles=args.eval_concurrency,
            stochastic=False,  # final eval = greedy policy
            start_listening=_eval_sl,
        )

        final_opponents = [
            (opponent_name(RandomPlayer), RandomPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"FinalRand{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
                start_listening=_eval_sl,
            )),
            (opponent_name(SimpleHeuristicsPlayer), SimpleHeuristicsPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"FinalHeur{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
                start_listening=_eval_sl,
            )),
            (opponent_name(Gen3StallerPlayer), Gen3StallerPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"FinalStall{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
                start_listening=_eval_sl,
            )),
            (opponent_name(Gen3AggressivePlayer), Gen3AggressivePlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"FinalAggr{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
                start_listening=_eval_sl,
            )),
            (opponent_name(Gen3SetupSweepPlayer), Gen3SetupSweepPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"FinalSetup{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
                start_listening=_eval_sl,
            )),
        ]
        for _cls, _uname in [
            (Gen3HeuristicV2Player, f"FinalHeur2{ts}"),
            (Gen3StallerV2Player, f"FinalStallV2{ts}"),
            (Gen3AggressiveV2Player, f"FinalAggrV2{ts}"),
            (Gen3SetupSweepV2Player, f"FinalSetupV2{ts}"),
        ]:
            final_opponents.append((opponent_name(_cls), _cls(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=server_config,
                account_configuration=AccountConfiguration(_uname, "password"),
                max_concurrent_battles=args.eval_concurrency,
                start_listening=_eval_sl,
            )))

        win_rates: dict[str, float] = {}
        for name, opponent in final_opponents:
            if rl_player.n_finished_battles > 0:
                rl_player.reset_battles()
            print(f"  vs {name} [{n} battles]...")
            start_time = datetime.now()
            if args.use_showdown_bridge:
                # Overlap games like the server does; cap the Node-process fan-out (eval_concurrency
                # defaults to 100, which would spawn 100 sim children).
                await run_local_battles(rl_player, opponent, n,
                                        concurrency=min(args.eval_concurrency, 8))
            else:
                await rl_player.battle_against(opponent, n_battles=n)
            duration = datetime.now() - start_time
            wr = rl_player.n_won_battles / rl_player.n_finished_battles
            win_rates[name] = wr
            print(f"  Win rate vs {name}: {wr * 100:.1f}%  [{duration}]")
            model.logger.record(f"eval_final/win_rate_vs_{name}", wr)

        aggregate = sum(win_rates.values()) / len(win_rates)
        model.logger.record("eval_final/win_rate_mean", aggregate)
        model.logger.dump(model.num_timesteps)
        print(f"\nFinal aggregate win rate: {aggregate * 100:.1f}%")

    # --- Callback Setup (Shared) ---
    # Periodic checkpoints land in <run>/checkpoints/ (SB3 makedirs it); the callback
    # keeps latest.txt + metadata.json at the run root (derived from save_path).
    checkpoint_callback = _TrackingCheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="checkpoint",
    )

    # --lr must lie within [--min-lr, --max-lr]. This is the user-facing contract
    # for both pure-adaptive runs and TwoPhaseLR Phase 1 — KL adaptation reads
    # args.lr as the seed for fresh runs and as the cap for resumes, so it has
    # to be a valid in-band LR. Enforced before any callback is constructed.
    _effective_max_lr = args.max_lr if args.max_lr is not None else args.lr * 2.0
    if not (args.min_lr <= args.lr <= _effective_max_lr):
        print(f"[AdaptiveLR] ERROR: --lr {args.lr:.2e} is outside "
              f"[--min-lr {args.min_lr:.2e}, --max-lr {_effective_max_lr:.2e}]")
        sys.exit(1)

    # If resuming and the checkpoint has already crossed into the cosine phase,
    # read the persisted handoff_lr so the new callback can pick up the same
    # cosine starting point. None means "still in Phase 1, KL-driven."
    resumed_handoff_lr: float | None = None
    if annealing_mode and args.model and os.path.exists(args.model):
        try:
            _meta = read_checkpoint_metadata(args.model)
            _h = _meta.get("handoff_lr")
            if isinstance(_h, (int, float)):
                resumed_handoff_lr = float(_h)
        except Exception as e:
            print(f"[TwoPhaseLR] WARNING: failed to read handoff_lr from {args.model}: {e}")

    if annealing_mode:
        lr_callback = TwoPhaseLRCallback(
            initial_lr=args.lr,
            total_steps=args.steps,
            anneal_start_steps=args.anneal_lr_start_steps,
            anneal_min_lr=args.anneal_min_lr,
            min_lr=args.min_lr,
            max_lr=args.max_lr,
            handoff_lr=resumed_handoff_lr,
        )
    else:
        lr_callback = AdaptivePPOCallback(
            initial_lr=args.lr,
            min_lr=args.min_lr,
            max_lr=args.max_lr,
        )
    # Keep alias so references below still resolve during the resume path.
    adaptive_ppo_callback = lr_callback
    checkpoint_callback._current_lr_fn = lambda: model.policy.optimizer.param_groups[0]["lr"]
    checkpoint_callback._current_epochs_fn = lambda: model.n_epochs
    # Only TwoPhaseLRCallback exposes a handoff_lr; AdaptivePPOCallback does not.
    checkpoint_callback._handoff_lr_fn = (
        (lambda: lr_callback.handoff_lr) if isinstance(lr_callback, TwoPhaseLRCallback) else None
    )
    graceful_restart_callback = GracefulRestartCallback()
    callbacks = [checkpoint_callback, lr_callback, MetricsExporterCallback(), _HparamLogCallback(args.ent_coef), graceful_restart_callback]
    # Win-probability head: captures each episode's win/loss outcome during collection + back-fills the
    # rollout buffer's MC label before train() (only when the head is on → a default run pays nothing).
    if args.win_prob_mode != "none":
        from agents.training.win_prob_callback import WinProbLabelCallback
        callbacks.append(WinProbLabelCallback())
    eval_callback = None
    # A --debug smoke run skips ALL eval by default — the periodic eval callback below AND the
    # final win-rate eval — so it needs no eval opponents / Showdown eval connection and stays
    # light on CPU. --debug-eval opts back in. Real (non-debug) runs are unaffected (always True).
    _run_eval = (not args.debug) or args.debug_eval

    # On resume, the last eval lives in the resumed checkpoint's metadata.json (a different
    # dir from this fresh run) — point the eval callback at it so the TUI shows the most
    # recent eval immediately instead of a blank panel until the next cycle.
    _resume_meta = None
    if args.model:
        _ckpt_dir = args.model if os.path.isdir(args.model) else os.path.dirname(args.model)
        # metadata.json is run-LEVEL (at the run root); a relocated checkpoint lives in
        # <run>/checkpoints/, so strip a trailing checkpoints/ to find it.
        if _ckpt_dir and os.path.basename(os.path.normpath(_ckpt_dir)) == "checkpoints":
            _ckpt_dir = os.path.dirname(_ckpt_dir)
        if _ckpt_dir:
            _resume_meta = os.path.join(_ckpt_dir, "metadata.json")

    if args.self_play and _pool is not None and _run_eval:
        # Self-play eval mirrors the bot-eval frozen-snapshot SUBPROCESS pattern
        # (non-blocking): the workers work-steal the bot roster AND up to 5 pool sentinels,
        # play a frozen snapshot, and the parent collects + promotes on a later poll. The
        # worker rebuilds opponents / teambuilders / mappings itself from the data dir, so
        # nothing live is constructed here. Under --debug it runs only with --debug-eval
        # (fast eval cadence), so `--self-play --debug --debug-eval` against a 9XXX server
        # exercises seed → pool eval → promotion; a plain --debug smoke skips it.
        eval_callback = SelfPlayCallback(
            pool=_pool,
            model_dir=model_dir,
            server_config=server_config,
            showdown_port=args.showdown_port,
            use_showdown_bridge=args.use_showdown_bridge,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            promote_threshold=_promote_threshold,
            self_play_temp=args.self_play_temp,
            # Greedy-vs-greedy pool eval (best-vs-best signal); default off keeps the live run's
            # win_rate_vs_pool / ELO continuous until opted in.
            eval_sentinel_greedy=args.eval_sentinel_greedy,
            # Curriculum knobs (#2): the live per-episode fraction the callback pushes each eval
            # uses these, matching the env's initial fraction computed above.
            heuristic_floor=_heuristic_floor,
            self_play_start_wr=_sp_start_wr,
            self_play_full_wr=_sp_full_wr,
            # Self-play eval is ~2x the inference of bot eval — the 5 sentinel matchups run
            # the model for BOTH players (trainee + sentinel), vs bot matchups where only the
            # trainee infers. So double the work-stealing pool to keep wall-clock comparable
            # (5 bot-eval workers → 10 here).
            n_workers=args.eval_workers * 2,
            eval_device=args.eval_device,
            eval_concurrency=args.eval_concurrency_per_worker,
            eval_shard_games=args.eval_shard_games,
            distill_opponents=args.distill_opponents,
            distill_device=args.eval_device,  # CPU by default → no GPU contention with training
            keep_eval_snapshots=args.keep_eval_snapshots,
            keep_eval_trace_steps=args.keep_eval_trace_steps,
            keep_stalls=args.keep_stalls,
            keep_crashes=args.keep_crashes,
            resume_eval_metadata=_resume_meta,
            fixed_opponents=_fixed_opponents,
            stable_opponent_mastered_wr=args.stable_opponent_mastered_wr,
            # Reporting-only: lets the callback REPORT the exact per-episode opponent-mix fractions
            # (train/selfplay_fraction = pool, train/stable_fraction, train/nonbot_fraction) the env
            # wrapper's selection implies — no change to selection. The capped stable challenge share,
            # the bot-weight vector, and the floor bot-roster size all live only in the wrapper / here.
            stable_challenge_share=args.stable_opponent_selfplay_share,
            bot_weight_vec=_bot_weight_vec,
            floor_roster_count=len(OPPONENT_CLASSES),
            debug=args.debug,
        )
        callbacks.append(eval_callback)
    elif _run_eval:
        # Bot eval runs in a frozen-snapshot subprocess (non-blocking, CPU). The
        # worker rebuilds opponents/teambuilders/mappings itself from the data
        # dir, so nothing live is constructed here.
        eval_callback = PerOpponentEvalCallback(
            model_dir=model_dir,
            server_config=server_config,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            n_workers=args.eval_workers,
            eval_device=args.eval_device,
            eval_concurrency=args.eval_concurrency_per_worker,
            eval_shard_games=args.eval_shard_games,
            showdown_port=args.showdown_port,
            use_showdown_bridge=args.use_showdown_bridge,
            resume_eval_metadata=_resume_meta,
            keep_eval_snapshots=args.keep_eval_snapshots,
            keep_eval_trace_steps=args.keep_eval_trace_steps,
            keep_stalls=args.keep_stalls,
            keep_crashes=args.keep_crashes,
            fixed_opponents=_fixed_opponents,
        )
        callbacks.append(eval_callback)

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
        _load_extractor_kwargs = _load_encoder.get_features_extractor_kwargs()
        # Behavioral mask toggle — version-checked vs the saved model_config.json. The resumed
        # policy is rebuilt from the zip's own kwargs, so this only feeds current_version: a
        # resume with a different value FATALs rather than silently ignoring the flag.
        _load_extractor_kwargs["attend_unrevealed_opponents"] = args.attend_unrevealed_opponents
        _load_extractor_kwargs["opp_belief_cls_k"] = args.opp_belief_cls_k
        # Belief-slots arch toggle — version-checked vs the saved config (a resume that flips the
        # belief head on/off FATALs, same machinery as opp_belief_cls_k). coef>0 is the enable signal.
        _load_extractor_kwargs["opp_belief_slots"] = (args.opp_belief_aux_coef > 0.0)
        _load_extractor_kwargs["value_active_readout"] = args.value_active_readout
        # Move-belief mode — version-checked vs the saved config (fresh-only; a resume that changes it
        # FATALs, same machinery as opp_belief_slots).
        _load_extractor_kwargs["move_belief_mode"] = args.move_belief_mode
        # Latent-belief arch toggle — version-checked vs the saved config (fresh-only). coef>0 enables.
        _load_extractor_kwargs["opp_belief_latent"] = (args.opp_belief_latent_coef > 0.0)
        # Damage-operator toggle — version-checked vs the saved config (fresh-only).
        _load_extractor_kwargs["damage_op"] = args.damage_op
        _load_extractor_kwargs["damage_reattend"] = args.damage_reattend       # v31 (version-checked)
        _load_extractor_kwargs["damage_outgoing"] = args.damage_outgoing       # v23 (version-checked)
        _load_extractor_kwargs["move_candidate_floor"] = args.move_candidate_floor  # v23 (version-checked)
        _load_extractor_kwargs["move_latent"] = args.move_latent               # v24 (version-checked)
        _load_extractor_kwargs["spread_belief"] = args.spread_belief           # v25 (version-checked)
        _load_extractor_kwargs["spread_belief_nature"] = args.spread_belief_nature  # v40 (version-checked)
        _load_extractor_kwargs["spread_belief_nature_marginalize"] = args.spread_belief_nature_marginalize  # v40
        # Move-prior fusion — version-checked vs the saved config (fresh-only).
        _load_extractor_kwargs["move_prior_fusion"] = args.move_prior_fusion
        # Move-belief pre-fuse (reinject before the transformer) — version-checked (fresh-only). v32.
        _load_extractor_kwargs["move_belief_prefuse"] = args.move_belief_prefuse
        # Incoming-damage-obs ablation — version-checked vs the saved config (fresh-only).
        _load_extractor_kwargs["mask_incoming_damage_obs"] = args.mask_incoming_damage_obs
        _load_extractor_kwargs["mask_active_move_scalars_obs"] = args.mask_active_move_scalars_obs  # v25
        _load_extractor_kwargs["mask_move_effects_obs"] = args.mask_move_effects_obs                # v25
        # Win-probability head mode — version-checked vs the saved config (resume-IMMUTABLE; any change
        # FATALs, same machinery as move_belief_mode).
        _load_extractor_kwargs["win_prob_mode"] = args.win_prob_mode
        # Distributional value head (v29) — mode + atom count version-checked in check_compatible; the
        # support (vmin/vmax) is resume-immutable, enforced below via enforce_value_dist.
        _load_extractor_kwargs["value_dist_mode"] = args.value_dist_mode
        _load_extractor_kwargs["value_dist_bins"] = args.value_dist_bins
        _load_extractor_kwargs["value_dist_vmin"] = args.value_dist_vmin
        _load_extractor_kwargs["value_dist_vmax"] = args.value_dist_vmax
        # Discrete top-K incoming block (v30) — K version-checked in check_compatible (scales projections).
        _load_extractor_kwargs["damage_topk_k"] = args.damage_topk_k
        _load_extractor_kwargs["damage_refine_rounds"] = args.damage_refine_rounds   # v31 (version-checked)
        _load_extractor_kwargs["damage_matrices_outgoing"] = args.damage_matrices_outgoing  # v32 (version-checked)
        _load_extractor_kwargs["damage_matrices_incoming"] = args.damage_matrices_incoming  # v33 (version-checked)
        _load_extractor_kwargs["damage_matrices_outgoing_all"] = args.damage_matrices_outgoing_all  # v39 (version-checked)
        _load_extractor_kwargs["threat_refine_outgoing"] = args.threat_refine_outgoing      # v36 (version-checked)
        _load_extractor_kwargs["threat_unrevealed_outgoing"] = args.threat_unrevealed_outgoing  # v36
        _load_extractor_kwargs["threat_prob_outspeed"] = args.threat_prob_outspeed          # v36 (version-checked)
        _load_extractor_kwargs["threat_status_refine"] = args.threat_status_refine          # v37 (version-checked)
        _load_extractor_kwargs["hp_type_belief_mode"] = args.hp_type_belief_mode            # v38 (version-checked)
        _load_policy_kwargs = {
            "features_extractor_class": Gen3FeaturesExtractor,
            "features_extractor_kwargs": _load_extractor_kwargs,
            "net_arch": NET_ARCH,
            "use_popart": args.use_popart,  # version-checked vs the saved model_config.json
        }
        current_version = ModelVersion.from_layout_and_policy_kwargs(
            _load_extractor_kwargs["layout"], _load_policy_kwargs, vf_coef=args.vf_coef,
            reward_config=reward_config, value_tail_weight=args.value_tail_weight,
            opp_belief_aux_coef=args.opp_belief_aux_coef,
            move_belief_coef=args.move_belief_coef,
            opp_belief_latent_coef=args.opp_belief_latent_coef,
            win_prob_coef=args.win_prob_coef,
            move_belief_latent_coef=args.move_belief_latent_coef,
            spread_belief_coef=args.spread_belief_coef,
            value_dist_coef=args.value_dist_coef,
            hp_type_belief_coef=args.hp_type_belief_coef,
        )

        print(f"Loading existing model from {model_path}")
        try:
            model = load_model_snapshot(
                model_path,
                env=env,
                current_version=current_version,
                device=args.device,
                enforce_vf_coef=args.vf_coef,  # FATAL if the run was started with a different vf_coef
                enforce_reward_config=reward_config,  # FATAL if bias_additivity/mat_alive_weight/redesign drift
                enforce_value_tail_weight=args.value_tail_weight,  # FATAL if the value-loss tail weight drifts
                enforce_value_dist=(args.value_dist_vmin, args.value_dist_vmax),  # FATAL if the dist support drifts
            )
        except ModelVersionError as e:
            print(f"\n[ModelVersion] FATAL: {e}")
            sys.stdout.flush()  # os._exit() skips buffer flushing — make sure the reason reaches the log
            # Non-recoverable: an arch-family / vf_coef / reward-config mismatch fails the
            # SAME way on every retry. Exit with FATAL_CONFIG so the launcher gives up
            # immediately instead of auto-restarting into the identical error.
            os._exit(int(TrainExitCode.FATAL_CONFIG))
        model.ent_coef = args.ent_coef
        model.value_tail_weight = args.value_tail_weight  # == saved (enforced above); set for the loop
        model.opp_belief_aux_coef = args.opp_belief_aux_coef  # training hparam (not version-locked; resume-mutable)
        model.opp_belief_moves_weight = args.opp_belief_moves_weight
        model.move_belief_coef = args.move_belief_coef  # move-belief loss weight (training-only; resume-mutable)
        model.move_belief_latent_coef = args.move_belief_latent_coef  # move-latent grading weight (training-only)
        model.spread_belief_coef = args.spread_belief_coef  # spread-belief speed-supervision weight (training-only)
        model.defensive_entropy_boost = args.defensive_entropy_boost            # gen3_defensive_entropy_v1 (training-only)
        model.defensive_entropy_anneal_frac = args.defensive_entropy_anneal_frac
        model.hp_type_belief_coef = args.hp_type_belief_coef  # HP-type CE weight (training-only; mode none = off)
        model.opp_belief_latent_coef = args.opp_belief_latent_coef  # latent-belief loss weight (training-only)
        model.win_prob_coef = args.win_prob_coef  # win-prob loss weight (training-only; resume-mutable)
        model.value_dist_coef = args.value_dist_coef  # value-dist HL-Gauss loss weight (training-only; resume-mutable)
        model.vf_coef = args.vf_coef  # == the saved value (enforced above); set explicitly for parity
        model.gae_lambda = 0.80
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
        model.n_epochs = args.n_epochs
        model.grad_accum_steps = args.grad_accum_steps   # grad accumulation (1 = off); a train-loop knob, re-applied each resume
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
            _run_roundtrip_test(model, _load_extractor_kwargs["layout"], _load_policy_kwargs, debug=args.debug)
            _apply_grad_checkpointing(model, args.grad_checkpointing)
            model._async_rollout = _async_rollout   # route collect_rollouts to the non-barrier path
            save_model_snapshot(model_dir, current_version, hparams=_model_hparams(model), cli_args=cli_args)

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
                model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=False)
            except Exception as e:
                print(f"Training interrupted by exception: {e}")
                final_path = os.path.join(model_dir, "final_model_exception")
                model.save(final_path)
                _write_latest_txt(model_dir, "final_model_exception.zip")

            final_path = os.path.join(model_dir, "final_model")
            model.save(final_path)
            _write_latest_txt(model_dir, "final_model.zip")
            save_model_snapshot(os.path.dirname(final_path), current_version, hparams=_model_hparams(model), cli_args=cli_args)
            print(f"Training complete. Model saved to {final_path}")
            best_model_dir = os.path.join(model_dir, "best_model")
            if os.path.isdir(best_model_dir):
                save_model_snapshot(best_model_dir, current_version, hparams=_model_hparams(model), cli_args=cli_args)
            if _run_eval:
                await evaluate_model_random(model)
    else:
        print(f"Starting NEW Training (Parallel x{n_envs}, Batch: {args.batch_size}, Epochs: {args.n_epochs})")
        # model_dir and unique_id are now pre-defined earlier in main()
        
        # Initialize a dummy encoder to get the handoff kwargs
        temp_encoder = Gen3ObservationEncoder(mappings)
        extractor_kwargs = temp_encoder.get_features_extractor_kwargs()
        extractor_kwargs["log_level"] = log_level
        # Unmask the opponent's still-hidden party (default off). SB3 forwards this to
        # Gen3FeaturesExtractor; from_layout_and_policy_kwargs records it in model_config.json.
        extractor_kwargs["attend_unrevealed_opponents"] = args.attend_unrevealed_opponents
        # Hidden-opponent belief (k=0 = off; k>0 requires the unmask flag, validated above). SB3 forwards
        # this to Gen3FeaturesExtractor; from_layout_and_policy_kwargs records it in model_config.json.
        extractor_kwargs["opp_belief_cls_k"] = args.opp_belief_cls_k
        # In-place hidden-opponent BELIEF AUX (weight-shape): coef>0 builds the BeliefHead + learned
        # unknown-mon slot tokens (auto-forces attend_unrevealed_opponents above). The coef itself is a
        # TRAINING hparam set on the model below; this bool is the version-checked arch toggle.
        extractor_kwargs["opp_belief_slots"] = (args.opp_belief_aux_coef > 0.0)
        # Value-head active readout (default off): routes our_active_refined into the value projection.
        extractor_kwargs["value_active_readout"] = args.value_active_readout
        # Move-belief reinjection (off|revealed|unrevealed|both; weight-shape). != off builds the MoveBelief
        # head + auto-forces attend_unrevealed_opponents above. The coef is a TRAINING hparam set below;
        # the MODE is the version-checked arch toggle.
        extractor_kwargs["move_belief_mode"] = args.move_belief_mode
        # Latent-belief escalation (weight-shape): coef>0 builds the BeliefHead latent predictor. The
        # coef is a TRAINING hparam set below; this bool is the version-checked arch toggle.
        extractor_kwargs["opp_belief_latent"] = (args.opp_belief_latent_coef > 0.0)
        # Differentiable damage operator (weight-shape): widens both projection heads with the
        # believed-move incoming-damage block. Requires move_belief_mode revealed|both (validated above).
        extractor_kwargs["damage_op"] = args.damage_op
        # Damage re-attend (structural; adds a damage→token projection + encoder layer, re-derives the
        # pools). Requires damage_op (validated above). Projection widths unchanged. v31.
        extractor_kwargs["damage_reattend"] = args.damage_reattend
        # Outgoing per-move direction (weight-shape): our active → opp active, action-aligned. Requires
        # damage_op (validated above). v23.
        extractor_kwargs["damage_outgoing"] = args.damage_outgoing
        # Learnset + rarity-cap move-prior gate (forward-behavior): 0.0 = legacy floor; >0 prunes illegal /
        # sub-floor moves. Requires move_prior_fusion (validated above). No weight-shape change. v23.
        extractor_kwargs["move_candidate_floor"] = args.move_candidate_floor
        # MoveLatentEncoder (weight-shape): the context-free mechanics-grounded move latent concatenated
        # into the move network. The latent-grading coef is a TRAINING hparam set below; this bool is the
        # version-checked arch toggle. v24 (gen3_unified_move_system_v1).
        extractor_kwargs["move_latent"] = args.move_latent
        # SpreadBelief (weight-shape): predict+reinject the opp's hidden spread; the op consumes it. The coef
        # is a TRAINING hparam set below; this bool is the version-checked arch toggle. v25.
        extractor_kwargs["spread_belief"] = args.spread_belief
        # gen3_nature_ev_belief_v1 (v40): the NATURE/EV generative head (structural) + the op-side nature
        # marginalization (forward-behavior). Both version-checked, fresh-only; OFF byte-identical.
        extractor_kwargs["spread_belief_nature"] = args.spread_belief_nature
        extractor_kwargs["spread_belief_nature_marginalize"] = args.spread_belief_nature_marginalize
        # --unified-obs disable-redundant masks (forward-behavior): zero a now-subsumed obs region. v25.
        extractor_kwargs["mask_active_move_scalars_obs"] = args.mask_active_move_scalars_obs
        extractor_kwargs["mask_move_effects_obs"] = args.mask_move_effects_obs
        # Unified move belief (forward-behavior): fuse the Smogon move prior into the belief head. Requires
        # move_belief_mode != off (validated above). No weight-shape change (non-persistent prior buffer).
        extractor_kwargs["move_prior_fusion"] = args.move_prior_fusion
        # Move-belief pre-fuse (forward-behavior): reinject the move belief BEFORE the transformer so the
        # believed moves co-refine through attention. Requires move_belief_mode != off (validated above).
        # No weight-shape change (same MoveBelief module/params, different call timing). v32.
        extractor_kwargs["move_belief_prefuse"] = args.move_belief_prefuse
        # Unified-architecture ablation (forward-behavior): zero the incoming-damage obs block from the
        # model's view. No weight-shape change. Independent A/B knob (typically paired with --damage-op).
        extractor_kwargs["mask_incoming_damage_obs"] = args.mask_incoming_damage_obs
        # Win-probability head (none|read_only|shaping; structural + resume-immutable). 'none' = no module
        # (baseline byte-for-byte). The coef is a TRAINING hparam set on the model below; the MODE is the
        # version-checked arch toggle.
        extractor_kwargs["win_prob_mode"] = args.win_prob_mode
        # Distributional VALUE head (v29; none|read_only|shaping + atom count + support). Interpretability
        # side readout off value_pooled — never in pi/vf. 'none' = no module (baseline byte-for-byte). Mode
        # + bins are version-checked (check_compatible); the support is resume-immutable (check_value_dist).
        extractor_kwargs["value_dist_mode"] = args.value_dist_mode
        extractor_kwargs["value_dist_bins"] = args.value_dist_bins
        extractor_kwargs["value_dist_vmin"] = args.value_dist_vmin
        extractor_kwargs["value_dist_vmax"] = args.value_dist_vmax
        # gen3_unified_topk_incoming_v1 (v30): the DISCRETE top-K incoming block's K (0 = off). STRUCTURAL
        # (scales both projections; version-checked). Requires --damage-op + --move-latent (validated above).
        extractor_kwargs["damage_topk_k"] = args.damage_topk_k
        extractor_kwargs["damage_refine_rounds"] = args.damage_refine_rounds
        extractor_kwargs["damage_matrices_outgoing"] = args.damage_matrices_outgoing
        extractor_kwargs["damage_matrices_incoming"] = args.damage_matrices_incoming
        extractor_kwargs["damage_matrices_outgoing_all"] = args.damage_matrices_outgoing_all
        extractor_kwargs["threat_refine_outgoing"] = args.threat_refine_outgoing
        extractor_kwargs["threat_unrevealed_outgoing"] = args.threat_unrevealed_outgoing
        extractor_kwargs["threat_prob_outspeed"] = args.threat_prob_outspeed
        extractor_kwargs["threat_status_refine"] = args.threat_status_refine
        extractor_kwargs["hp_type_belief_mode"] = args.hp_type_belief_mode

        policy_kwargs = {
            "features_extractor_class": Gen3FeaturesExtractor,
            "features_extractor_kwargs": extractor_kwargs,
            "net_arch": [512, 512],
            "optimizer_class": torch.optim.AdamW,
            "optimizer_kwargs": {"weight_decay": args.weight_decay, "eps": 1e-5},
            "use_popart": args.use_popart,  # builds the PopArtNormalizer in the policy; recorded in model_config.json
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
            gamma=0.9999,
            gae_lambda=0.80,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            device=args.device,
            seed=args.seed,
            policy_kwargs=policy_kwargs
        )

        model.value_tail_weight = args.value_tail_weight   # tail-weighted value loss (0.0 = plain MSE)
        model.grad_accum_steps = args.grad_accum_steps     # grad accumulation (1 = off; effective batch = batch_size·K)
        model.opp_belief_aux_coef = args.opp_belief_aux_coef  # hidden-opp belief aux loss (0.0 = off)
        model.opp_belief_moves_weight = args.opp_belief_moves_weight  # species_CE + w·moves_BCE
        model.move_belief_coef = args.move_belief_coef  # move-belief reinjection loss (0.0 = off)
        model.move_belief_latent_coef = args.move_belief_latent_coef  # move-latent grading loss (0.0 = off)
        model.spread_belief_coef = args.spread_belief_coef  # spread-belief speed-supervision loss (0.0 = off)
        model.defensive_entropy_boost = args.defensive_entropy_boost            # gen3_defensive_entropy_v1 (training-only)
        model.defensive_entropy_anneal_frac = args.defensive_entropy_anneal_frac
        model.hp_type_belief_coef = args.hp_type_belief_coef  # HP-type CE loss (0.0 = off; mode none = off)
        model.opp_belief_latent_coef = args.opp_belief_latent_coef  # latent-belief loss (0.0 = off)
        model.win_prob_coef = args.win_prob_coef  # win-prob head BCE loss (mode none = off)
        model.value_dist_coef = args.value_dist_coef  # value-dist HL-Gauss loss (mode none = off)
        version = ModelVersion.from_layout_and_policy_kwargs(
            extractor_kwargs["layout"], policy_kwargs, vf_coef=args.vf_coef,
            reward_config=reward_config, value_tail_weight=args.value_tail_weight,
            opp_belief_aux_coef=args.opp_belief_aux_coef,
            move_belief_coef=args.move_belief_coef,
            opp_belief_latent_coef=args.opp_belief_latent_coef,
            win_prob_coef=args.win_prob_coef,
            move_belief_latent_coef=args.move_belief_latent_coef,
            spread_belief_coef=args.spread_belief_coef,
            value_dist_coef=args.value_dist_coef,
            hp_type_belief_coef=args.hp_type_belief_coef,
        )
        # PBRS_GAMMA must equal the PPO gamma for both potentials to be policy-invariant (design §7.1).
        # The reward manager is built before the model (in the env factory), so assert here where both
        # exist. A non-default --gamma would silently break PBRS — make it a fast startup crash.
        from agents.training.reward_manager import PBRS_GAMMA as _PBRS_GAMMA
        assert abs(_PBRS_GAMMA - float(model.gamma)) < 1e-12 and abs(reward_config.gamma - float(model.gamma)) < 1e-12, (
            f"PBRS_GAMMA ({_PBRS_GAMMA}) / reward_config.gamma ({reward_config.gamma}) must equal "
            f"model.gamma ({model.gamma}) — PBRS is only policy-invariant when they match."
        )
        _run_roundtrip_test(model, extractor_kwargs["layout"], policy_kwargs, debug=args.debug)
        _apply_grad_checkpointing(model, args.grad_checkpointing)
        model._async_rollout = _async_rollout   # route collect_rollouts to the non-barrier path
        save_model_snapshot(model_dir, version, hparams=_model_hparams(model), cli_args=cli_args)

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
            model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=False)
        except Exception as e:
            print("\n" + "🛑" * 30)
            print(f"🛑 TRAINING CRASHED: {e}")
            print("🛑" * 30)
            traceback.print_exc()
            os._exit(1) # Stop immediately, do not proceed to evaluation

        final_path = os.path.join(model_dir, "final_model")
        model.save(final_path)
        _write_latest_txt(model_dir, "final_model.zip")
        _final_handoff = lr_callback.handoff_lr if isinstance(lr_callback, TwoPhaseLRCallback) else None
        record_checkpoint(model_dir, final_path + ".zip", adaptive_ppo_callback.current_lr, model.n_epochs, hparams=_model_hparams(model), handoff_lr=_final_handoff)
        save_model_snapshot(os.path.dirname(final_path), version, hparams=_model_hparams(model), cli_args=cli_args)
        print(f"Training complete. Model saved to {final_path}")
        best_model_dir = os.path.join(model_dir, "best_model")
        if os.path.isdir(best_model_dir):
            save_model_snapshot(best_model_dir, version, hparams=_model_hparams(model), cli_args=cli_args)
        if _run_eval:
            await evaluate_model_random(model)

if __name__ == "__main__":
    asyncio.run(main())
