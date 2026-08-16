import multiprocessing
import os as _os
import traceback
import functools

# ── BLAS THREAD PINNING — must run BEFORE torch is imported anywhere ──────────────────────────
# Each SubprocVecEnv worker runs a full CPU opponent forward; with the library default (one thread
# per core) N workers spawn N×cores competing threads. Measured on a 16-core box, 8 neural-opponent
# envs run DIRECTLY (no launcher): load average 110 and **6 fps**, vs 231 fps with these pinned — a
# ~38× cliff that dwarfs every other measured throughput lever.
#
# `launcher/child.py` already exports these for production, so runs under the launcher were never
# affected — but `python src/main/train_rl_agent.py …` is a DOCUMENTED entry point (root CLAUDE.md
# "Training — run directly") and had no such protection. Setting them here covers both paths; workers
# inherit them through `spawn`. `setdefault` so an explicit override still wins, and the env-worker
# `_init` pins `torch.set_num_threads(1)` independently in case one does.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

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

from agents.model.features_extractor import BELIEF_GRAD_MODES, Gen3FeaturesExtractor, NET_ARCH
# The move prior's LEGAL-BUT-UNOBSERVED base (--move-candidate-floor default) and its lower bound.
# Move legality itself is unconditional — these only set how high a legal-unobserved move starts.
from agents.model.damage_tables import _PRIOR_FLOOR, _MIN_PRIOR_FLOOR
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.model.model_version import ModelVersion, ModelVersionError
from agents.model.extractor_arch import build_extractor_arch_kwargs
from agents.model.compile_prewarm import prewarm_extractor_compile
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
from agents.training.matchup_spec import MatchupSpec
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
    structural toggle (use_popart / opp_belief_cls_k / move_belief_mode / damage_op)
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


def _read_saved_optimizer_state(checkpoint_path: str, opt_name_set):
    """Read ``(saved_optimizer_state_dict, saved_param_names)`` straight from an SB3 checkpoint zip.

    ``saved_param_names`` is the saved registration ORDER of the parameters the saved optimizer
    indexed (params only), recovered by filtering the saved ``policy.pth`` state_dict keys to
    ``opt_name_set`` (the names of the params in the CURRENT optimizer). The param subsequence of a
    module's ``state_dict()`` keys is exactly its ``named_parameters()`` order, and the optimizer
    indexes those params 0..N-1 in that order, so ``saved_param_names[i]`` is the param that owns the
    saved optimizer state entry ``i``. Raises on any structural surprise (missing members, count
    mismatch) so the caller falls back to the shape-only guard instead of remapping on bad data."""
    import io
    import zipfile
    with zipfile.ZipFile(checkpoint_path) as z:
        members = set(z.namelist())
        if not {"policy.optimizer.pth", "policy.pth"} <= members:
            raise FileNotFoundError("checkpoint missing policy.optimizer.pth / policy.pth")
        saved_opt = torch.load(io.BytesIO(z.read("policy.optimizer.pth")),
                               map_location="cpu", weights_only=False)
        saved_policy = torch.load(io.BytesIO(z.read("policy.pth")),
                                  map_location="cpu", weights_only=False)
    saved_param_names = [k for k in saved_policy.keys() if k in opt_name_set]
    n_saved_opt = sum(len(g.get("params", [])) for g in saved_opt.get("param_groups", []))
    if len(saved_param_names) != n_saved_opt:
        raise ValueError(f"saved param-name count ({len(saved_param_names)}) != saved optimizer "
                         f"param count ({n_saved_opt}); cannot safely map momentum by name")
    return saved_opt, saved_param_names


def _remap_optimizer_state_by_name(opt, current_named_params, saved_opt, saved_param_names) -> dict:
    """Rebuild ``opt``'s per-param momentum so each CURRENT param receives the momentum that was saved
    for a param of the SAME NAME, regardless of registration order — closing the same-shape-reorder
    blind spot a shape check cannot see. ``current_named_params`` is ``(name, param)`` in OPTIMIZER
    index order. A name whose shape changed, or that is new, gets fresh zero-init momentum; a vanished
    saved name is ignored. Mutates ``opt`` via ``load_state_dict`` (torch casts each entry to its
    param's device/dtype). Returns a counts dict for logging/tests. Pure given its inputs → unit-tested."""
    saved_state = saved_opt.get("state", {}) or {}
    saved_idx_of = {nm: i for i, nm in enumerate(saved_param_names)}
    corrected, counts = {}, {"carried": 0, "reordered": 0, "dropped_shape": 0, "fresh": 0}
    for j, (name, param) in enumerate(current_named_params):
        si = saved_idx_of.get(name)
        entry = saved_state.get(si) if si is not None else None
        if entry is None:
            counts["fresh"] += 1                          # new param, or one that carried no momentum
            continue
        ea = entry.get("exp_avg")
        if ea is not None and tuple(ea.shape) != tuple(param.shape):
            counts["dropped_shape"] += 1                  # name reused at a different shape → drop
            continue
        corrected[j] = entry
        counts["carried"] += 1
        if si != j:
            counts["reordered"] += 1
    sd = opt.state_dict()                                 # current param_groups (correct indices) ...
    sd["state"] = corrected                               # ... with momentum re-keyed to current positions
    opt.load_state_dict(sd)                               # torch casts each entry to its param's device/dtype
    return counts


def _shape_only_reset_optimizer_state(model) -> None:
    """Fallback guard, used only when the checkpoint zip can't be read for a name-keyed remap: drop
    ALL momentum if ANY param's saved exp_avg/exp_avg_sq shape disagrees with the live param (proof
    the position-keyed state is misaligned). Same-shape permutations are UNDETECTABLE here — which is
    exactly why the name-keyed remap is preferred whenever the zip is readable."""
    opt = getattr(getattr(model, "policy", None), "optimizer", None)
    if opt is None:
        return
    name_of = {id(p): n for n, p in model.policy.named_parameters()}
    bad = []
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p)
            if not st:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                t = st.get(key)
                if t is not None and tuple(t.shape) != tuple(p.shape):
                    bad.append(f"{name_of.get(id(p), '?')} param{tuple(p.shape)} {key}{tuple(t.shape)}")
    if bad:
        from collections import defaultdict
        print(f"[Resume] WARNING: optimizer momentum is MISALIGNED with current parameters "
              f"({len(bad)} shape mismatch(es)) — a parameter-reorder refactor since this checkpoint "
              f"was saved desynced the position-keyed Adam state. RESETTING optimizer momentum "
              f"(fresh zero-init; LR/param_groups preserved). Mismatches: " + "; ".join(bad[:8]))
        sys.stdout.flush()
        opt.state = defaultdict(dict)


def _validate_or_reset_optimizer_state(model, checkpoint_path: str = None) -> None:
    """Realign a resumed AdamW optimizer state to the CURRENT parameters BY NAME.

    SB3/torch save+load the optimizer state BY PARAMETER POSITION, not by name, so a refactor that
    REORDERS a module's parameters between the save and the resume (e.g. v40's `SpreadBelief.__init__`
    building `reinject`/`norm` before `stat_head`) silently misassigns the saved per-param momentum
    (`exp_avg`/`exp_avg_sq`) to the WRONG params: a DIFFERENT-shape reorder crashes `AdamW.step()`
    later (the gen3_nature_ev_belief_v1 bug), and — worse, because it is silent — a SAME-shape reorder
    is invisible to a shape check and quietly corrupts momentum.

    Fix: remap BY NAME. We read the saved optimizer state + the saved parameter NAME ORDER straight
    from the checkpoint zip and rebuild `opt.state` so each current param receives exactly the momentum
    that was saved for its name, regardless of registration order (a param whose name is new or whose
    shape changed gets fresh zero-init). This SUPERSEDES a shape-only reset and closes the
    same-shape-reorder blind spot, so "append new params LAST" is no longer load-bearing for optimizer
    correctness. Falls back to the legacy shape-only reset if the zip can't be read (defensive — never
    crash a resume). No-op (momentum carried verbatim) when the saved order already matches current."""
    opt = getattr(getattr(model, "policy", None), "optimizer", None)
    if opt is None:
        return
    if checkpoint_path:
        try:
            opt_param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
            named = [(n, p) for n, p in model.policy.named_parameters() if id(p) in opt_param_ids]
            id_to_name = {id(p): n for n, p in named}
            # (name, param) in the EXACT optimizer index order — robust even if some named params
            # are excluded from the optimizer (e.g. a frozen param).
            current = [(id_to_name.get(id(p)), p)
                       for group in opt.param_groups for p in group["params"]]
            saved_opt, saved_param_names = _read_saved_optimizer_state(
                checkpoint_path, {n for n, _ in named})
            if len(saved_param_names) != len(current):
                raise ValueError(f"saved param count ({len(saved_param_names)}) != current "
                                 f"({len(current)})")
            counts = _remap_optimizer_state_by_name(opt, current, saved_opt, saved_param_names)
            if counts["reordered"] or counts["dropped_shape"]:
                print(f"[Resume] Optimizer momentum remapped BY NAME: {counts['carried']} carried "
                      f"({counts['reordered']} were REORDERED since save → corrected), "
                      f"{counts['dropped_shape']} dropped on shape change, {counts['fresh']} fresh. "
                      f"Position-keyed desync prevented.")
                sys.stdout.flush()
            return
        except Exception as e:
            print(f"[Resume] WARNING: name-keyed optimizer remap unavailable ({e}); falling back to "
                  f"the shape-only guard.")
            sys.stdout.flush()
    _shape_only_reset_optimizer_state(model)


def _resolve_fresh_model_dir(run_name, exploiter_label, model_arg):
    """Pick the run directory for a run whose --run-dir is NOT set (i.e. not a launcher-managed
    resume). Precedence: an explicit --run-name → ``models/<name>``; else, in exploiter mode, a
    derived ``models/exploiter_vs_<target>``; else a date-stamped ``models/run_<timestamp>`` (the
    legacy default). A NAMED dir is validated as a single safe path component, and we refuse to start
    a FRESH run on top of an EXISTING run (one carrying a metadata.json) — unless --model resumes from
    INSIDE that very dir — so naming a run after e.g. the live run can't silently clobber it. Returns
    the dir (or exits with a clear FATAL). Pure given its args → unit-tested."""
    import re
    if run_name:
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", run_name):
            print(f"\n[RunName] FATAL: --run-name {run_name!r} must be a single name "
                  f"(letters/digits/._-), with no slashes or path traversal.")
            sys.exit(1)
        model_dir = os.path.join("models", run_name)
    elif exploiter_label:
        model_dir = os.path.join("models", "exploiter_vs_" + exploiter_label.removeprefix("ext_"))
    else:
        return f"models/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # always unique → no guard
    # Clobber guard: a named fresh run must not write into a DIFFERENT existing run's dir.
    resuming_into_it = bool(model_arg) and os.path.abspath(model_arg).startswith(
        os.path.abspath(model_dir) + os.sep)
    if os.path.exists(os.path.join(model_dir, "metadata.json")) and not resuming_into_it:
        print(f"\n[RunName] FATAL: {model_dir!r} is already a run (it has a metadata.json). Pick a "
              f"different --run-name, or pass --model <a checkpoint inside it> to resume that run.")
        sys.exit(1)
    return model_dir


def _run_arch_toggles(args) -> dict:
    """The architecture TOGGLES of THIS run, for current_model_version so the version gate compares
    like-for-like against the run's own (toggle-ON) pool/stable-opponent snapshots. Without these, a
    belief-ON / popart / attend-unrevealed run would FATAL on every snapshot it is meant to protect.

    Sourced from `agents.model.flag_registry` via `arch_toggles_from_args`, NOT hand-listed: this
    dict and `build_extractor_arch_kwargs` used to be two independently maintained lists of the same
    toggles, and a toggle added to one and not the other means the gate compares an architecture the
    run does not build. `use_popart` is appended by hand because it is a policy_kwarg rather than an
    extractor kwarg, so it is out of the registry's scope."""
    from agents.model.extractor_arch import arch_toggles_from_args
    return {**arch_toggles_from_args(args), "use_popart": args.use_popart}


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
        "win_prob_coef": float(getattr(model, "win_prob_coef", 1.0)),
        "pubval_coef": float(getattr(model, "pubval_coef", 0.0)),
        "value_dist_coef": float(getattr(model, "value_dist_coef", 1.0)),
        "search_teacher_coef": float(getattr(model, "search_teacher_coef", 0.0)),
        "search_teacher_value_coef": float(getattr(model, "search_teacher_value_coef", 0.0)),
        "search_teacher_beta": float(getattr(model, "search_teacher_beta", 1.0)),
        "search_teacher_batch_size": int(getattr(model, "search_teacher_batch_size", 256)),
        "opd_coef": float(getattr(model, "opd_coef", 0.0)),
        "distill_coef": float(getattr(model, "distill_coef", 0.0)),
        "distill_value_coef": float(getattr(model, "distill_value_coef", 0.0)),
        "distill_value_feat_coef": float(getattr(model, "distill_value_feat_coef", 0.0)),
        "opd_beta": float(getattr(model, "opd_beta", 1.0)),
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
    import numpy as np
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
        pubval_coef=float(getattr(model, "pubval_coef", 0.0)),
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


# gen3_smoke_eval_scale_v1: a short run is a SMOKE, and a smoke's final eval is a formality.
# Measured: the final eval is 9 opponents x --eval-battles games; at 100 that is ~900 battles and
# ran past a 300s timeout on a loaded box, printing "Training complete" BEFORE it started — which
# reads exactly like a hang and cost real debugging time. Scaling it for short runs removes the
# tax; the honest banner below removes the confusion.
SMOKE_STEPS = 100_000
SMOKE_EVAL_BATTLES = 5
DEFAULT_EVAL_BATTLES = 100


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
    parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, default=None,
                        help="A MEMORABLE name for a fresh run → writes to models/<name>/ instead of "
                             "a date-stamped models/run_<timestamp>/. Must be a single name "
                             "(letters/digits/._-, no slashes). Refuses to overwrite an existing run "
                             "of that name (pick another, or --model to resume it). Ignored when "
                             "--run-dir is set (launcher resume). For --exploiter, defaults to "
                             "'exploiter_vs_<target>' if you don't name it.")
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
    parser.add_argument("--use-bridge", type=str, default="rust",
                        choices=["off", "node", "rust"],
                        help="In-process BattleStream bridge transport for BOTH training AND eval "
                             "(no Showdown server, no port, no /challenge storm, deterministic). "
                             "'rust' (DEFAULT) = the byte-compatible src/rust_sim sim_bridge binary "
                             "(built via cargo; override with POKESIM_SIM_BRIDGE_BIN) — measured "
                             "1.41x node's throughput at --n-envs 48 with a ~25x smaller child "
                             "(9 MB RSS vs ~224 MB). 'node' = the Node local_sim_bridge.js, kept as "
                             "the explicit A/B arm and for the parity harness. 'off' = the websocket "
                             "transport, which needs a running Showdown server on --showdown-port. "
                             "NOTE: 'rust' now emits __RECON__ (gen3_bridge_recon_record_v1, on a "
                             "seedless battle too) and supports resumeReseed "
                             "(gen3_bridge_resume_reseed_v1), so the forensic reconstruction and "
                             "counterfactual paths work on rust. The OFFLINE search/replay drivers "
                             "are on rust too (gen3_rust_search_driver_v1 / "
                             "gen3_rust_replay_driver_v1 — one search_driver binary serves both "
                             "verb families), so --search-teacher no longer requires 'node'; the "
                             "run's impl is threaded into the teacher workers. 'rust' also "
                             "fail-louds on an unmodeled move.")
    parser.add_argument(
        "--self-play-use-cpu",
        action=BoolFlag,
        default=True,
        help="Load self-play opponent snapshots on CPU instead of the training device. "
             "Default True: avoids one CUDA context per SubprocVecEnv worker (~300-600 MB each), "
             "which would otherwise OOM the GPU at high --n-envs. Opponent inference is batch-1 "
             "no_grad, so CPU is plenty fast. Pass --no-self-play-use-cpu to load them on --device.",
    )
    parser.add_argument("--eval-battles", type=int, default=None,
                        help="Battles per FINAL-evaluation opponent. Default 100, but AUTO-SCALED "
                             f"down to {SMOKE_EVAL_BATTLES} when --steps < {SMOKE_STEPS:,} (a smoke "
                             "run), because a 9-opponent x 100-battle final eval costs many minutes "
                             "and a 2k-step policy produces no signal worth that. An explicit value "
                             "always wins.")
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
                        help="The LEGAL-BUT-UNOBSERVED base probability of the fused move prior (default "
                             "0.02). This is NOT an on/off switch: move LEGALITY is UNCONDITIONAL — a move a "
                             "species CANNOT learn always gets ~0 prior mass, and a legal move always keeps "
                             "its TRUE Smogon usage (rare techs stay rare-but-liftable, never pruned, so "
                             "surprise-move anticipation survives). This flag only sets how high a LEGAL move "
                             "with no recorded usage starts, so in-battle evidence can still lift it. Must be "
                             ">= 0.001 (0.0 would make legal-unobserved indistinguishable from impossible). "
                             "Forward-behavior value (version-checked, fresh-only); only read under "
                             "--move-prior-fusion, which is what builds the prior.")
    parser.add_argument("--move-prior-fusion", "--move_prior_fusion", dest="move_prior_fusion",
                        action=BoolFlag, default=None,
                        help="Unified two-part move belief: fuse the Smogon move-frequency PRIOR into the "
                             "move-belief head as a log-odds residual (posterior = prior + learned delta) "
                             "and PIN revealed moves certain — so the belief the damage op + BCE loss read "
                             "is one coherent posterior (priors ⊕ prediction unified), anchored at the "
                             "prior at cold-start. Forward-behavior toggle (no weight-shape change; "
                             "version-checked, fresh-only). REQUIRES --move-belief-mode != off. Off by default.")
    parser.add_argument("--t0-species-prior", "--t0_species_prior",
                        dest="t0_species_prior", action=BoolFlag, default=None,
                        help="T0 SPECIES belief for the physics (gen3_t0_species_prior_v1, v72): price "
                             "unrevealed opponent mons from the model's own team-composition belief "
                             "(naive-Bayes over the revealed team, Species-Clause floored) instead of "
                             "the STATIC gen3ou usage prior. The belief already existed at T2 "
                             "(BeliefHead) where the T1 DamageOperator could not read it; this "
                             "re-homes it to T0. Parameter-free, no state_dict change. STRUCTURAL and "
                             "version-checked: it re-means every damage number against a hidden slot, "
                             "so it cannot be flipped on resume.")
    parser.add_argument("--species-prior-fusion", "--species_prior_fusion",
                        dest="species_prior_fusion", action=BoolFlag, default=None,
                        help="SPECIES belief prior fusion (gen3_species_prior_fusion_v1, v68): fuse a "
                             "TEAM-COMPOSITION prior into BeliefHead's species head as a log-prob "
                             "residual (posterior = prior + learned delta), the same two-part shape "
                             "--move-prior-fusion gives the move belief. The prior is naive Bayes over "
                             "pairwise co-occurrence in the data/teams/ pool — 'given the opponent mons "
                             "already revealed, what is likely in a hidden slot' — with Species Clause "
                             "as a hard constraint. The species head was the ONE belief leg with no "
                             "prior, so it cold-started ~uniform over ~400 nums. Measured on the pool, "
                             "5-fold held out: top-1 0.106 with nothing revealed, and with 3 revealed "
                             "0.189 conditional vs 0.156 marginal-only (top-3 0.449 vs 0.345) — vs "
                             "~0.0025 for uniform. The delta head is ZERO-INIT, so the cold-start "
                             "posterior EQUALS the prior. Adds NO parameters (the co-occurrence tables "
                             "are non-persistent buffers), but STRUCTURAL + version-checked all the "
                             "same: flipping it re-means every species logit. REQUIRES "
                             "--opp-belief-aux-coef>0. Off by default (byte-identical).")
    parser.add_argument("--compile-opponents", "--compile_opponents", dest="compile_opponents",
                        action="store_true", default=False,
                        help="torch.compile each frozen SELF-PLAY OPPONENT's feature extractor in the "
                             "env workers (CPU, B=1 — the measured 68%% of rollout worker time). "
                             "Measured 6.53x on the real forward; value-preserving to ~5e-7 with 0/16 "
                             "argmax flips. This is the CPU/ROLLOUT half; --compile-trainer is the "
                             "GPU/LEARNER half and they are independent. RUNTIME PERF KNOB: not "
                             "versioned, not in check_compatible, NOT inherited on resume — re-pass it "
                             "each launch. Hides CUDA in the (CPU) workers first, because compiling in "
                             "a CUDA-visible process costs ~252 MiB of card per worker.")
    parser.add_argument("--compile-opponents-strict", "--compile_opponents_strict",
                        dest="compile_opponents_strict", action="store_true", default=False,
                        help="Turn a failed or ineffective OPPONENT compile into a hard error instead "
                             "of a warning. Without --compile-opponents this does nothing. Falling "
                             "back to eager is a ~6.5x regression on the opponent forward that is "
                             "otherwise invisible (the run just produces fewer steps/hour forever), so "
                             "use this when you would rather fail at startup than discover it in the "
                             "FPS graph a day later. (--compile-trainer needs no such flag: it is "
                             "ALWAYS fail-loud, see its help.)")
    parser.add_argument("--compile-trainer", "--compile_trainer", dest="compile_trainer",
                        action="store_true", default=False,
                        help="torch.compile the LEARNER's feature extractor — the GPU forward AND "
                             "backward that the PPO train step runs. Measured on v76 at the production "
                             "shape (batch 4096, PopArt on, real MaskablePPO path): "
                             "155.1 -> 88.5 ms per minibatch = 1.75x, i.e. ~+62%% end-to-end FPS at the "
                             "~89%% train share. CUDA ONLY and FAIL-LOUD by design — a silent fall back "
                             "to eager would be an invisible 1.75x regression, and the CPU backward "
                             "provably does not lower (Inductor's C++ backend refuses an atomic_add "
                             "scatter). RUNTIME PERF KNOB: not versioned, NOT inherited on resume — "
                             "re-pass it each launch, like --grad-checkpointing.")
    parser.add_argument("--consequence-topk", "--consequence_topk", dest="consequence_topk",
                        type=int, default=None,
                        help="v59: the CONSEQUENCE kernels' believed-candidate axis — C1b/C2/C3's "
                             "k_cand + D4's k_bench in one knob (how many candidates the belief-"
                             "weighted worst-case max covers per opp mon). Default 6 (4 real moves "
                             "+ 2 surprise slots; pre-v59 models trained at 4). FORWARD-BEHAVIOR "
                             "(no params) but version-checked — a frozen opponent's forward "
                             "changes with it.")
    parser.add_argument("--entity-topk-seats", "--entity_topk_seats", dest="entity_topk_seats",
                        type=int, default=None,
                        help="gen3_entity_move_seats_v1 (v54, Stage 1 of the entity generation): the E4 "
                             "THREAT-MOVE seat count — the opp active's top-K believed candidate moves "
                             "enter the trunk as attention SEATS ([move latent ⊕ belief w ⊕ acc ⊕ "
                             "is_phys] per seat; the op's refine_candidates definition, one source). "
                             "0 (default) = E3-only: our active's 4 request-ordered move seats, which "
                             "are UNCONDITIONAL in this generation (the pointer head reads the REFINED "
                             "seats). STRUCTURAL int (version-checked, fresh-only). >0 REQUIRES "
                             "--damage-op + --move-latent (--unified-moves).")
    parser.add_argument("--entity-tail-seats", "--entity_tail_seats", dest="entity_tail_seats",
                        action=BoolFlag, default=None,
                        help="gen3_entity_tail_seats_v1 (v57, E5): 6 per-opp-mon TAIL-THREAT seats — "
                             "the truncation insurance summarizing the beyond-top-K belief mass every "
                             "candidate consumer drops ([p_tail, worst_phys, worst_spec, revealed]). "
                             "STRUCTURAL (version-checked, fresh-only). REQUIRES --damage-op "
                             "AND --entity-topk-seats > 0.")
    parser.add_argument("--edge-bias-families", "--edge_bias_families", dest="edge_bias_families",
                        type=str, default=None,
                        help="gen3_edge_bias_trunk_v1 (v56, Stage 2 of the entity generation): deliver "
                             "computed physics as per-pair per-head additive ATTENTION BIASES. 'off' "
                             "(default) | 'd' (= d1,d3) | a comma list. d1 = our active's moves x the "
                             "opp's 6 mons (the outgoing-matrix kernel) at the (E3 seat, opp-mon seat) "
                             "pairs — requires --damage-op + --damage-outgoing; d3 = the opp's top-K "
                             "believed moves x our 6 mons (the pre-collapse incoming kernel, the SAME "
                             "candidates as the E4 seats) at the (E4 seat, our-mon seat) pairs — "
                             "requires --entity-topk-seats > 0. c1 = the CONSEQUENCE edge: post-"
                             "setup-move damage/outspeed DELTAS (SD/DD/CM/Agility hypothetical "
                             "kernel re-runs) at the (E3 setup seat, opp-mon) pairs — requires "
                             "--damage-op + --damage-outgoing. Zero-init maps: identity at init. "
                             "STRUCTURAL (version-checked, fresh-only). The op head-concat stays "
                             "(deprecation playbook: bias-ablation audit before deletion).")
    parser.add_argument("--damage-candidate-k", "--damage_candidate_k", dest="damage_candidate_k",
                        type=int, default=None,
                        help="Cap the DamageOperator's INCOMING candidate sweep at the K most-believed "
                             "opponent moves (0 = the full ~400-wide sweep, byte-identical). NO tail "
                             "bound - the truncated mass is DROPPED, so a rare-but-lethal candidate "
                             "below rank K is simply not priced (the on-policy probe measured top-16 "
                             "owning 94.2%% of channels, with misses BIMODAL). Payoff is learner-side: "
                             "measured +11.4%% forward / +63.5%% op at B=256, but only +0.3%% at B=1 "
                             "(the CPU opponent is dispatch-bound, not tensor-size bound). "
                             "Forward-behavior (version-checked, fresh-only). REQUIRES --damage-op.")
    # gen3_pointer_native_v1: --pointer-head is GONE — the pointer head is THE action head,
    # unconditionally (no flat action_net exists in this generation; see Gen3DualHeadMaskablePolicy).
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
    parser.add_argument("--pubval-mode", "--pubval_mode", dest="pubval_mode",
                        choices=("none", "read_only", "shaping"), default=None,
                        help="Auxiliary PUBLIC-VALUE head (gen3_pubval_aux_v1): regress a value-pool readout "
                             "toward the FROZEN human-replay-calibrated public value V_pub = P(win | PUBLIC "
                             "board) (data/gen3_pubval.json — 164k rated gen3ou games, held-out AUC ~0.73, "
                             "calibrated; regenerate via `python -m agents.training.pubval_calibration`). The "
                             "value-INDEPENDENT exogenous signal: a dense per-step target that tells the trunk "
                             "WHEN the game swung (credit assignment), priced by HUMAN outcomes instead of the "
                             "self-play bootstrap. 'none' (default) = no module (byte-for-byte). 'read_only' = "
                             "head-only training on a STOP-GRAD value pool (a learnability probe: CAN the trunk "
                             "linearly carry V_pub?). 'shaping' = the human positional prior also shapes the "
                             "shared trunk (the credit-assignment experiment). STRUCTURAL + resume-IMMUTABLE "
                             "(version-checked). SIDE readout — never in pi/vf, never in GAE (V^human ≠ V^π).")
    parser.add_argument("--pubval-coef", "--pubval_coef", dest="pubval_coef",
                        type=float, default=None,
                        help="Loss weight for the pubval head's soft-target BCE (pubval_coef * BCE), like "
                             "--win-prob-coef. Default 0.1. TRAINING-only (not version-locked; inherited on a "
                             "flagless resume). Ignored when --pubval-mode none. Lower it if 'shaping' fights "
                             "the policy (watch grad/pubval_share).")
    # --- SEARCH-AS-TEACHER (offline ExIt plateau-breaker; designs/ai_v6/design_search_teacher.md) ---
    # All TRAINING-only (no version bump; coef 0 / flag absent = byte-identical). The coefs are
    # _resolve'd (flagless-resume-inherited); the operational knobs are forwarded by the launcher.
    parser.add_argument("--search-teacher", "--search_teacher", dest="search_teacher",
                        action="store_true",
                        help="Enable the search-teacher: each cycle, search + rollout-confirm the worst "
                             "falsify-flagged loss craters (EXACT reloaded opponent), CI-gate strictly-"
                             "better corrections, and distil them into the policy via an AWR aux loss. "
                             "Non-blocking (subprocess workers). Recommended at PLATEAU. Re-pass on resume.")
    parser.add_argument("--search-teacher-coef", "--search_teacher_coef", dest="search_teacher_coef",
                        type=float, default=None,
                        help="AWR policy-distillation weight (search_teacher_coef * advantage-weighted CE "
                             "toward the verified-better action). Default 0.0 = OFF (loss byte-identical). "
                             "Training-only (inherited on a flagless resume). Watch grad/searchteacher_share "
                             "+ teacher/agree_rate.")
    parser.add_argument("--search-teacher-value-coef", "--search_teacher_value_coef",
                        dest="search_teacher_value_coef", type=float, default=None,
                        help="OFF by default (0.0) — the off-policy value term (the search value is V^π*, "
                             "which biases the GAE critic). Only for the joint-ExIt A/B.")
    parser.add_argument("--search-teacher-beta", "--search_teacher_beta", dest="search_teacher_beta",
                        type=float, default=None, help="AWR temperature β (default 1.0).")
    # ON-POLICY SELF-DISTILLATION (OPD) — upgrades the distillation TARGET from the single action A*
    # (AWR) to the FULL improved distribution π' via KL(π' ‖ π_student). Training-only, modelled EXACTLY
    # on --search-teacher-coef (0 = byte-identical; NOT version-locked). REQUIRES --search-teacher (it
    # fills the correction buffer + its workers build π'). A run carries BOTH targets → A/B AWR vs KL.
    parser.add_argument("--opd-coef", "--opd_coef", dest="opd_coef", type=float, default=None,
                        help="ON-POLICY SELF-DISTILLATION weight (opd_coef * KL(π' ‖ π_student) toward the "
                             "beam's improved distribution). Default 0.0 = OFF (loss byte-identical). "
                             "Requires --search-teacher. Training-only (inherited on a flagless resume). "
                             "Watch grad/opd_share + opd/kl / opd/agree_rate.")
    parser.add_argument("--opd-beta", "--opd_beta", dest="opd_beta", type=float, default=None,
                        help="OPD softmax temperature β for π' over the per-action backed-up values "
                             "(default 1.0). Higher β → flatter target.")
    # EXPLOITER DISTILLATION (gen3_exploiter_distill_v1) — pour a frozen per-team SPECIALIST (an
    # --exploiter checkpoint) into the generalist via an ON-POLICY KL, masked to the states where the
    # trainee pilots the teacher's team; the other (pool) states are the anti-forgetting rehearsal.
    # Training-only (0 = byte-identical; NOT version-locked). designs/learning/generalist_specialist_amortization_gap.md
    parser.add_argument("--distill-teacher", "--distill_teacher", dest="distill_teacher", type=str, default=None,
                        help="Frozen exploiter teacher(s) to distil into the trainee, as "
                             "'TEACHER:TEAM' pairs (KL(π_teacher ‖ π_student) on that teacher's team states). "
                             "TEACHER = a checkpoint dir/.zip, TEAM = its Showdown team file. Comma-separated "
                             "for N teachers (joint multi-teacher distillation), e.g. "
                             "'models/expA:data/teams/specialist/a.txt,models/expB:data/teams/specialist/b.txt'. "
                             "The colon pairing binds each teacher to its team — no misalignment possible.")
    parser.add_argument("--distill-teacher-team", "--distill_teacher_team", dest="distill_teacher_team",
                        type=str, default=None,
                        help="DEPRECATED (back-compat for in-flight runs) — a parallel team list for the bare "
                             "(colon-less) --distill-teacher form. Prefer the 'TEACHER:TEAM' pair form instead.")
    parser.add_argument("--distill-coef", "--distill_coef", dest="distill_coef", type=float, default=None,
                        help="Exploiter-distillation KL weight (default 0.0 = OFF, loss byte-identical). "
                             "Requires --distill-teacher + --distill-teacher-team. Training-only (inherited on "
                             "a flagless resume). Watch distill/kl ↓ + distill/agree_rate ↑ + grad/distill_share.")
    parser.add_argument("--distill-value-coef", "--distill_value_coef", dest="distill_value_coef",
                        type=float, default=None,
                        help="VALUE-distillation weight (gen3_exploiter_value_distill_v1): also pour the "
                             "teacher's per-team VALUE into the student — MSE(V_teacher, V_student) on the "
                             "teacher-team states, in the PopArt-normalized frame. Default 0.0 = OFF "
                             "(byte-identical; no teacher predict_values forward). Requires --distill-coef > 0 "
                             "(the policy KL validates the value target). Training-only, inherited on resume. "
                             "The A/B lever for 'does distilling the value enrich it' — watch distill/value_mse ↓ "
                             "and the value_cls effective-rank probe rise. Distributional-value distill is future.")
    parser.add_argument("--distill-value-feat-coef", "--distill_value_feat_coef", dest="distill_value_feat_coef",
                        type=float, default=None,
                        help="FITNETS VALUE-FEATURE distillation weight (gen3_exploiter_value_feat_distill_v1): "
                             "match the teacher's INTERMEDIATE 128-dim value-CLS pool (the hint layer) instead of "
                             "the collapsed scalar V — 1−cos(value_pooled_student, value_pooled_teacher) on the "
                             "teacher-team states, so the trunk inherits the teacher's per-team value STRUCTURE "
                             "(scalar value-distill CRYSTALLIZES the critic — value_cls rank DROPS). Default 0.0 = "
                             "OFF (byte-identical; no teacher value_pooled read). Requires --distill-coef > 0. "
                             "Training-only, inherited on resume. Composes with / is an A/B alternative to "
                             "--distill-value-coef — watch distill/value_feat_cos ↓ + the value_cls rank probe.")
    parser.add_argument("--distill-team-bias", "--distill_team_bias", dest="distill_team_bias",
                        type=float, default=0.4,
                        help="Fraction of trainee episodes biased to the teacher's team (rest = pool "
                             "rehearsal). Default 0.4. Only used when --distill-coef > 0.")
    parser.add_argument("--search-teacher-batch-size", "--search_teacher_batch_size",
                        dest="search_teacher_batch_size", type=int, default=None,
                        help="Corrections sampled per train() for the AWR forward (default 256).")
    parser.add_argument("--search-teacher-buffer-size", "--search_teacher_buffer_size",
                        dest="search_teacher_buffer_size", type=int, default=20000,
                        help="Correction ring capacity (recency; default 20000).")
    parser.add_argument("--teacher-search-budget", "--teacher_search_budget", dest="teacher_search_budget",
                        type=int, default=200, help="Candidates searched per cycle (budget cap; default 200).")
    parser.add_argument("--teacher-confirm-rollouts", "--teacher_confirm_rollouts",
                        dest="teacher_confirm_rollouts", type=int, default=8,
                        help="Monte-Carlo confirm games per candidate for the Wilson-CI strictly-better gate.")
    parser.add_argument("--teacher-search-workers", "--teacher_search_workers",
                        dest="teacher_search_workers", type=int, default=3,
                        help="Search-teacher worker subprocesses per cycle (default 3).")
    parser.add_argument("--teacher-search-freq", "--teacher_search_freq", dest="teacher_search_freq",
                        type=int, default=0, help="Steps between search-teacher cycles (0 = use the eval freq).")
    parser.add_argument("--teacher-persistent", "--teacher_persistent", dest="teacher_persistent",
                        action="store_true",
                        help="PERSISTENT-pool mode (the supply lever): long-lived workers GENERATE their "
                             "own fresh losses (frozen trainee vs current opponents) and search them "
                             "CONTINUOUSLY, dripping corrections into the buffer — instead of the bursty "
                             "per-cycle eval-trace scan. Higher, fresher supply; recommended once enabled.")
    parser.add_argument("--teacher-refresh-steps", "--teacher_refresh_steps", dest="teacher_refresh_steps",
                        type=int, default=500_000,
                        help="Persistent mode: re-freeze the trainee snapshot the workers use every N "
                             "steps (so long-lived workers track the moving policy). Default 500k.")
    parser.add_argument("--teacher-gen-battles", "--teacher_gen_battles", dest="teacher_gen_battles",
                        type=int, default=12, help="Persistent mode: battles generated per worker iteration.")
    parser.add_argument("--intent-value-reduce", "--intent_value_reduce",
                        dest="intent_value_reduce", action=BoolFlag, default=None,
                        help="STEP 6 (gen3_intent_value_reduce_v1): CONSUME alpha. Reduces the "
                             "operator's un-reduced per-(our mon, believed move) cells by alpha "
                             "into an expected-incoming-threat row per mon, appended to the "
                             "CRITIC's features through a zero-init projection. The op itself is "
                             "untouched (still hard-max) — alpha is scored downstream of it and "
                             "cannot weight its internal reduction. Requires --opp-intent-coef>0 "
                             "and --damage-op. STRUCTURAL, version-checked.")
    parser.add_argument("--intent-move-cell", "--intent_move_cell",
                        dest="intent_move_cell", action=BoolFlag, default=None,
                        help="G3 (gen3_intent_move_cell_v1, design_conditional_execution.md): the "
                             "POLICY-side alpha consumer — the c2 status-consequence family "
                             "re-delivered through the pointer MOVE cell as a per-action absolute, "
                             "alpha-conditioned (burn/sleep channels become unrenormalized "
                             "alpha-expectations over the op's top-K seat candidates; the seat "
                             "mass rides as a decorrelated alpha_stay channel). Zero-init "
                             "projection => identity at init. Requires --opp-intent-coef>0, "
                             "--damage-op and --damage-topk-k>0. STRUCTURAL, version-checked.")
    parser.add_argument("--value-entity-pool", "--value_entity_pool",
                        dest="value_entity_pool", action=BoolFlag, default=None,
                        help="gen3_unified_value_readout_v1 (v80, design_unified_belief.md §3 / "
                             "Stage-3 T3-DELIVER): ONE attention pool over the critic's entity "
                             "rows — the 12 post-transformer team tokens + the op's per-our-mon "
                             "incoming rows — K learned queries, per-source type embeddings, "
                             "ZERO-INIT output projection riding vf only (the policy is untouched "
                             "at any weight). The designed successor of the bolt-on vf routes the "
                             "critic_route_audit adjudicates. Works with or without --damage-op "
                             "(the row set shrinks to the team tokens). STRUCTURAL, "
                             "version-checked.")
    parser.add_argument("--opp-intent-grad-mode", "--opp_intent_grad_mode",
                        dest="opp_intent_grad_mode", choices=["detached", "shaping"], default=None,
                        help="Whether alpha/beta's gradient reaches the shared trunk "
                             "(gen3_intent_grad_mode_v1). 'detached' (default) = pure supervision, "
                             "so a null indicts the HEAD rather than the policy. 'shaping' lets the "
                             "intent objective shape the representation — watch "
                             "grad/opp_intent_policy_cosine: persistently negative means it is "
                             "FIGHTING the RL objective for the trunk. STRUCTURAL, version-checked.")
    parser.add_argument("--beta-setvalued-coef", "--beta_setvalued_coef",
                        dest="beta_setvalued_coef", type=float, default=None,
                        help="SET-VALUED partial credit for beta on switch-ins we did not believe "
                             "(gen3_beta_setvalued_v1). Today those rows are MASKED, discarding a "
                             "true fact: they brought a mon we had not revealed. This grades the "
                             "coarse call -log(sum of believed-slot mass) without asserting WHICH "
                             "member, which is the part we cannot label. Scales on top of "
                             "--opp-intent-coef. 0.0 = OFF (byte-identical). Training-only.")
    parser.add_argument("--opp-intent-coef", "--opp_intent_coef", dest="opp_intent_coef",
                        type=float, default=0.0,
                        help="OPPONENT-INTENT aux (gen3_opp_intent_v1, v67): supervise ALPHA — a "
                             "distribution over the opponent's K believed threat-move seats PLUS "
                             "SWITCH — and BETA — which of their mons comes in — against what they "
                             "ACTUALLY did. Both are POINTER heads (equivariant over their moves / "
                             "their bench) and see a DETACHED input, so a null says the head cannot "
                             "predict them rather than that predicting them hurt the policy. "
                             "Measured headroom (gen-8): the belief's top-K contains their move 85.8%% "
                             "of the time but ranks it first only 51.8%% — 34pp of mis-ranked mass. "
                             "Requires --entity-topk-seats>0. 0.0 = OFF (no heads, byte-identical). "
                             "STRUCTURAL + version-checked; the coef itself is training-only.")
    parser.add_argument("--value-threat-inject", "--value_threat_inject",
                        dest="value_threat_inject", action="store_true", default=False,
                        help="CRITIC THREAT INJECTION (gen3_value_threat_inject_v1, v64): add the "
                             "DamageOperator's alpha-weighted incoming-threat row for each of OUR "
                             "mons to that mon's token on the VALUE POOL's copy only, so value_cls "
                             "pools per-entity threat MAGNITUDES instead of the softmax RATIOS the "
                             "d3 edge family can carry. vf-ONLY: the policy reads the unaugmented "
                             "tokens, so pi is bit-identical at any weight (gated). Forces the op's "
                             "pair reduction to the R1 belief_mean rung (hard_max builds no reducer "
                             "and would leave nothing to inject). Zero-init => ON starts identical "
                             "to OFF. STRUCTURAL + version-checked: fixed for a run's lifetime.")
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
                        choices=["off", "incoming", "both"], default=None,
                        help="ONE knob for the WHOLE unified move system: sets --unified-damage to the same "
                             "level (move belief + prior fusion + the GPU damage op, incl. its per-status "
                             "secondary/Serene-Grace effects; 'both' adds the outgoing direction) AND turns "
                             "on --move-latent + a default --move-belief-latent-coef 0.05 + the DISCRETE "
                             "incoming move-space at K=5 (--damage-topk, which implies --damage-matrices "
                             "incoming). DEFAULT: 'both' on a FRESH run (the unified system IS the model — "
                             "without it the op has no belief to price and the policy loses the whole "
                             "believed-move threat read); a RESUME (--model) inherits the checkpoint's saved "
                             "component toggles verbatim, so old configs keep working. 'off' is DEPRECATED — "
                             "it survives only as an explicit ablation baseline and warns at startup. Compose "
                             "the pieces by hand for finer control (e.g. --damage-topk 0 to A/B the discrete "
                             "move-space off under --unified-moves).")
    parser.add_argument("--damage-topk", "--damage_topk", dest="damage_topk_k",
                        type=int, default=None,
                        help="K for the DISCRETE incoming move-space: the number of the opp ACTIVE's "
                             "most-believed CANDIDATE moves the INCOMING per-move damage matrix surfaces "
                             "INDIVIDUALLY (vs the worst-case max collapse that loses WHICH move it is) — "
                             "per move its LATENT identity + belief + acc + is_phys + per-move effect/"
                             "secondary bits, then per OUR mon [low, high, crit, P(KO), type_mult, "
                             "status_lands], the read that makes 'anticipate the move / pick the safe "
                             "switch' decidable (damage-immunity AND status-immunity both = 0, e.g. "
                             "Thunder-Wave→Ground). 0 = off. STRUCTURAL int (scales both projections; "
                             "version-checked, fresh-only). REQUIRES --damage-op + --move-latent, and "
                             "IMPLIES --damage-matrices incoming (gen3_op_block_trim_v1 deleted the lean "
                             "top-K block K used to select — the matrix is its strict superset, and the "
                             "profiler measured the lean block at 0 calls/forward). AUTO-set to 5 by "
                             "--unified-moves (the moveset is 4, so the 5th slot is the surprise candidate); "
                             "the 5th is zeroed once all 4 opp moves are revealed. Default off.")
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
    # gen3_bidir_threat_trunk_v1 (v36): the uncertainty-aware P(outspeed).
    parser.add_argument("--threat-prob-outspeed", "--threat_prob_outspeed", dest="threat_prob_outspeed",
                        action=BoolFlag, default=None,
                        help="#3 UNCERTAINTY-AWARE P(outspeed): divide the speed gap by the believed speed STD "
                             "(SPECIES_SPREAD_PRIOR; sigmoid≈normal-CDF) instead of a fixed scale — a high-variance "
                             "opp speed reads ~0.5, a pinned one reads sharp. FORWARD-behavior (version-checked, "
                             "fresh-only). REQUIRES --damage-op. Default off (byte-identical).")
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
    parser.add_argument("--hp-belief-mode", "--hp_belief_mode", dest="hp_belief_mode",
                        choices=["composed", "flat"], default=None,
                        help="How the opponent's 16 TYPED Hidden-Power channels are produced "
                             "(gen3_hp_belief_ablation_v1). BOTH arms reason over discrete TYPED HP "
                             "(355-370) and mask the typeless BP-0 num 237 — that is not the variable, "
                             "it is the 'opp HP reads immune' bug. "
                             "'composed' (DEFAULT) factors the belief as P(HP_t) = presence x P(type), "
                             "which makes 'a REVEALED Hidden Power must exist as SOME type' structural "
                             "(Sum_t P(HP_t) = presence, reveal-pinned), and applies the two certain-fact "
                             "eliminations: moveset exhaustion (4 moves seen, none is HP => ruled out) and "
                             "effectiveness narrowing (the HiddenPowerTracker's hard zeros). "
                             "'flat' is the ABLATION: no HPTypeBelief head — the multi-label move head "
                             "predicts the 16 typed channels INDEPENDENTLY off their own real per-typed "
                             "Smogon usage priors, i.e. Hidden Power is treated exactly like any other "
                             "move, with no factorisation, no constraint and no narrowing. Use it to "
                             "measure what the factorisation is worth. STRUCTURAL (version-checked, "
                             "fresh-only).")
    parser.add_argument("--hp-type-belief-coef", "--hp_type_belief_coef", dest="hp_type_belief_coef",
                        type=float, default=None,
                        help="HP-type-belief SUPERVISION weight (gen3_opp_hp_type_belief_v1): coef * "
                             "cross_entropy(HPTypeBelief posterior, TRUE opp HP type) over the REVEALED opp "
                             "slots that run Hidden Power (privileged training-only label from agent2's team — "
                             "Gen 3 never reveals the opp HP type). 0.0 = the head still runs and still gets "
                             "the op's damage gradient + the move-belief BCE through its typed channels; it "
                             "just has no direct CE, so it stays near the Smogon prior. gen3_typed_hp_belief_v1 "
                             "removed the old --hp-type-belief mode flag: the head is UNCONDITIONAL whenever "
                             "there is a move belief, because its 'off' state made the model reason over a "
                             "typeless BP-0 Hidden Power and priced a REVEALED HP as nonexistent. "
                             "TRAINING-only (not version-locked); metrics ride belief/hptype_* (acc, n_slots).")
    parser.add_argument("--value-from-dist", "--value_from_dist", dest="value_from_dist",
                        action=BoolFlag, default=None,
                        help="Phase B (gen3_dist_critic_v1): make the DISTRIBUTIONAL value head the critic "
                             "— GAE/bootstrap/deployment read E[Z] and the HL-Gauss CE is the primary value "
                             "loss (vf_coef weight); the scalar value_net freezes as a fallback. Requires "
                             "--value-dist-mode shaping. Resume-immutable (the belief-grad-mode class); flip "
                             "on a warm-started run with --allow-value-from-dist-change.")
    parser.add_argument("--allow-value-from-dist-change", "--allow_value_from_dist_change",
                        dest="allow_value_from_dist_change", action="store_true", default=False,
                        help="Permit the INTENTIONAL Phase-B critic-source migration on resume (the v45 gate "
                             "otherwise FATALs a drift). The offline probe confirmed E[Z]≈V, so the swap is "
                             "near-seamless. Loud notice; next save records the new mode. Needed once.")
    parser.add_argument("--allow-belief-grad-mode-change", "--allow_belief_grad_mode_change",
                        dest="allow_belief_grad_mode_change", action="store_true", default=False,
                        help="Permit an INTENTIONAL belief-grad-mode migration on resume (the v41 gate "
                             "otherwise makes a drift FATAL). detach() is value-preserving, so flipping "
                             "shaping<->detached on a converged checkpoint is weight-safe — only future "
                             "gradients change. Prints a loud notice; the next checkpoint save records "
                             "the new mode, so this flag is needed once per migration.")
    parser.add_argument("--belief-grad-mode", "--belief_grad_mode", dest="belief_grad_mode",
                        choices=list(BELIEF_GRAD_MODES), default=None,
                        help="gen3_belief_grad_mode_v1: WHICH gradient arrow between the STATE-prediction "
                             "belief heads (move / spread / hp-type / the species-moves-latent aux) and the "
                             "rest of the net is cut. THE TWO NON-DEFAULT MODES CUT OPPOSITE ARROWS. "
                             "'shaping' (default) = nothing cut: the heads READ the live trunk, so their "
                             "supervised + reinject gradients reshape it, and PPO trains the heads. "
                             "'detached' = they READ a STOP-GRAD trunk, so NO belief gradient reshapes the "
                             "trunk — it can't drag the trunk toward predicting hidden state at the policy's "
                             "expense (eliminates belief->trunk interference). "
                             "'label_only' (gen3_belief_label_only_v1) = the opposite cut: the heads' outputs "
                             "are PUBLISHED stop-grad to every forward consumer, so NO policy/value gradient "
                             "reaches a belief head's PARAMETERS and the belief is trained by its supervised "
                             "labels ALONE. The belief is still computed, reinjected and consumed by the op — "
                             "the policy reads it, it just can't push it off-calibration. Its trunk READ stays "
                             "live, so the label loss still teaches the trunk to encode hidden state (cutting "
                             "both would leave a probe on a trunk with no reason to carry the information, "
                             "still feeding the policy — that combination is deliberately not offered). "
                             "In ALL modes detach() is value-preserving, so the FORWARD is bit-identical and "
                             "only the training gradient differs. RESUME-IMMUTABLE (like --vf-coef, "
                             "version-checked on resume only — a frozen opponent's forward is unaffected). The "
                             "win-aligned heads (--win-prob-mode / --value-dist-mode) keep their own read_only.")
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
                             "websocket) more connection churn; the in-process bridge (--use-bridge, the default) is "
                             "preferred for fine shards. >= the per-opponent game count disables sharding (one shard "
                             "per opponent = the original opponent-level behaviour).")
    parser.add_argument("--eval-games", "--eval_games", dest="eval_games", type=int, default=None,
                        help="Games per OPPONENT per eval cycle (default: the module EVAL_GAMES, 100). "
                             "Per-cell 95%% CI: n=100 -> +/-0.098, n=200 -> +/-0.069 — raise for tighter "
                             "sentinel/promotion reads at proportionally more eval compute (work-stolen "
                             "across --eval-workers, off the training path). Shards per opponent = "
                             "eval-games / --eval-shard-games.")
    parser.add_argument("--snapshot-ladder-games", "--snapshot_ladder_games",
                        dest="snapshot_ladder_games", type=int, default=100,
                        help="Frozen-snapshot ELO ladder: games per pair for the per-promotion "
                             "round-robin tax (0 = disable). On each promotion a DETACHED bridge "
                             "subprocess plays the new frozen snapshot vs the current pool and "
                             "appends to <run>/snapshot_ladder/games.jsonl (measured once, kept "
                             "forever) — a dense, high-resolution internal ladder the saturated "
                             "bots can't provide. Off the training path.")
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
    # ── PFSP / league-lite (prioritized fictitious self-play) — both OFF by default (byte-identical) ──
    parser.add_argument("--pfsp-scale", "--pfsp_scale", dest="pfsp_scale", type=float, default=0.0,
                        help="PFSP hardness weighting for self-play pool sampling (default 0.0 = off, "
                             "pure recency). >0 oversamples the pool selves the trainee is LOSING to "
                             "(weight ×(1 + pfsp_scale·(1−win_rate))) while never starving the ones it "
                             "beats — turns the recency window into a prioritised curriculum. The live "
                             "per-snapshot win-rates are measured at each self-play eval (EMA-smoothed) "
                             "and pushed to the training envs. Try 1.0–2.0. Pairs with --pool-spread so "
                             "PFSP has a diverse ladder of selves, not a recent-selves echo chamber.")
    parser.add_argument("--n-sentinels", "--n_sentinels", dest="n_sentinels", type=int, default=5,
                        help="Number of evenly-spaced pool snapshots eval'd as sentinels each self-play "
                             "cycle (default 5). Each gets a FRESH win-rate, which is what --pfsp-scale "
                             "weights the pool by — so a higher count re-prioritises MORE of the pool per "
                             "cycle (cuts the 'only ~¼ of the pool re-measured' staleness on a deep pool). "
                             "Cost: each extra sentinel is +100 games/cycle, work-stolen by the doubled "
                             "eval pool; eval is non-blocking + skip-while-running so it self-throttles. "
                             "Pairs with a larger --max-snapshots. Training-only (not version-locked).")
    parser.add_argument("--pool-spread", "--pool_spread", dest="pool_spread",
                        action=BoolFlag, default=False,
                        help="Self-play pool retention: keep a temporally-DIVERSE ladder (newest + "
                             "oldest + an even interior spread) instead of the oldest-evicted sliding "
                             "window, so PFSP (--pfsp-scale) has a real range of past selves to "
                             "up-weight. Default off = the legacy sliding window (byte-identical).")
    # ── Team-side PFSP: variance-weighted TEAM sampling by self-play win-rate (OFF by default) ──
    parser.add_argument("--team-pfsp", "--team_pfsp", dest="team_pfsp",
                        choices=["off", "measure", "var", "onesided"], default="off",
                        help="Per-team self-play win-rate tracking for the trainee's pool teams (default "
                             "off = uniform random.choice, byte-identical). 'measure' TRACKS + persists "
                             "the per-team self-play win-rate to <run>/team_winrates.json (the offline "
                             "'which team is the generalist weakest on → next exploiter target' artifact) "
                             "WITHOUT biasing sampling. 'var' additionally weights each pool team by floor "
                             "+ p*(1-p) (p = the win-rate EMA, seed 0.5), capped at --team-pfsp-cap x the "
                             "uniform share — so the trainee drills the teams it wins ~half the time (max "
                             "variance) and stops over-sampling the ones it crushes / always loses. "
                             "'onesided' keeps the LOSING side at MAX weight instead — w(p)=0.25 for p<0.5, "
                             "else p*(1-p) (continuous at 0.5): every sub-50%% team stays maximally sampled "
                             "and only mastery retires a team (under the z_arch/FiLM conditioning "
                             "hypothesis the weak tail is the learnable headroom, so 'truly lost' is the "
                             "claim under test, not a sampling prior). Measured on SELF-PLAY pool battles "
                             "only (bots excluded). Training-only, NOT version-locked.")
    parser.add_argument("--team-pfsp-cap", "--team_pfsp_cap", dest="team_pfsp_cap",
                        type=float, default=3.0,
                        help="Over-representation cap for --team-pfsp: no team is sampled more than "
                             "this multiple of the uniform share (weight ≤ cap×mean(raw)). Default 3.0.")
    parser.add_argument("--team-pfsp-floor", "--team_pfsp_floor", dest="team_pfsp_floor",
                        type=float, default=0.05,
                        help="Weight floor for --team-pfsp (raw_i = floor + p*(1-p)) so a fully-won / "
                             "fully-lost team is never starved to zero. Default 0.05.")
    parser.add_argument("--team-block-episodes", "--team_block_episodes", dest="team_block_episodes",
                        type=int, default=1,
                        help="Hold each drawn TRAINEE team for N consecutive episodes before redrawing "
                             "(1 = off, byte-identical). The per-team gradient-density counter to the "
                             "measured FiLM sample starvation (film/noise_scale ~8-9x the batch): at "
                             "~64 (~one rollout of episodes) per-update per-team density rises ~15x AND "
                             "blocks span an update boundary, so an env replays its team right after "
                             "that team's gradient landed (the exploiter-style learn-and-retest loop). "
                             "Composes with --team-pfsp (weights apply at each redraw; outcomes "
                             "attribute to the blocked team). Trainee side only; training-only, NOT "
                             "version-locked, resume-forwarded.")
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
    parser.add_argument("--stable-opponent-pfsp", "--stable_opponent_pfsp",
                        dest="stable_opponent_pfsp", action="store_true",
                        help="DYNAMIC stable-opponent selection: within the capped stable challenge "
                             "slice, pick WEIGHTED by how much the trainee is LOSING to each "
                             "(1 - win_rate) instead of uniformly — spend the exploiter budget on the "
                             "axis it's failing worst, and let each fade as it's mastered. The TOTAL "
                             "pool-vs-stable share is unchanged. Training-only; OFF = uniform "
                             "(byte-identical). Pairs with a raised --stable-opponent-selfplay-share.")
    parser.add_argument("--exploiter", dest="exploiter", type=str, default=None,
                        help="EXPLOITER MODE: train against ONE fixed foreign model as the SOLE "
                             "opponent every episode — the league 'exploiter' role (learn to beat a "
                             "specific target, e.g. the current main agent). Takes a run dir / "
                             "checkpoint spec exactly like --stable-opponents (e.g. "
                             "'models/ai_v6_13_outgoing_dmg_0620'), must share this run's "
                             "arch_signature (startup FATAL otherwise). This is a clean opponent-mix "
                             "front-end: it needs NO --self-play / --stable-opponents / share "
                             "fiddling — just point it at the target. Mutually exclusive with "
                             "--self-play. Recommended: init the exploiter from a strong checkpoint "
                             "(--model <target's checkpoint>) so it has a baseline to exploit from. "
                             "Default None (off).")
    parser.add_argument("--warmstart-consensus", "--warmstart_consensus", dest="warmstart_consensus",
                        type=str, default=None,
                        help="EXPLOITER MODE (requires --exploiter): before training, build a competent, "
                             "archetype-NEUTRAL warm start by disagreement-gated CONSENSUS distillation of "
                             "N mature teacher exploiters (comma-separated run-dirs) into --model (the "
                             "generalist init), then init the exploiter from it. The BC target is SHARP "
                             "where the teachers AGREE (universal decisions inherited) and FLAT where they "
                             "DISAGREE (archetype forks left high-entropy → the new exploiter specializes "
                             "FREELY, unbiased). Built ONCE into <run>/warmstart/ (idempotent across "
                             "launcher restarts; skipped once a training checkpoint exists). Deliberately "
                             "NOT valid for generalist/self-play runs (whose job is the opposite — absorb "
                             "divergence via --distill-teacher). See agents.training.warmstart. Default off.")
    parser.add_argument("--warmstart-battles", dest="warmstart_battles", type=int, default=200,
                        help="On-policy battles to collect for the --warmstart-consensus BC dataset (200).")
    parser.add_argument("--warmstart-bc-steps", dest="warmstart_bc_steps", type=int, default=4000,
                        help="BC gradient steps for --warmstart-consensus (early-stops on gated-KL; 4000).")
    parser.add_argument("--exploiter-keep-bots", dest="exploiter_keep_bots", action="store_true",
                        help="EXPLOITER MODE (requires --exploiter): mix the heuristic bots BACK IN "
                             "alongside the exploiter target instead of playing the target as the sole "
                             "opponent. Per episode, the target is faced with prob "
                             "(1 - --exploiter-bot-fraction), else a random floor/heuristic bot. Lets a "
                             "from-scratch specialist keep a bot floor while it learns to beat one strong "
                             "target. Off (default) = the target is the sole opponent (byte-identical).")
    parser.add_argument("--exploiter-bot-fraction", dest="exploiter_bot_fraction", type=float,
                        default=0.5,
                        help="Under --exploiter-keep-bots, the per-episode probability of facing a "
                             "heuristic bot instead of the exploiter target (default 0.5). The exploiter "
                             "target is faced with the complementary probability (1 - this).")
    parser.add_argument("--exploiter-temp-start", dest="exploiter_temp_start", type=float, default=None,
                        help="EXPLOITER MODE (requires --exploiter): ANNEAL the target opponent's sampling "
                             "temperature over training — a difficulty curriculum via opponent STOCHASTICITY. "
                             "Setting this (a positive float, e.g. 2.0) starts the target at this temperature "
                             "(flatter logits → noisier/weaker play, so a from-scratch trainee can win some "
                             "games and get a learning signal) and linearly anneals it to --exploiter-temp-end "
                             "over --exploiter-temp-anneal-frac of training, held after. None (default) = OFF: "
                             "the target plays at --stable-opponent-temp the whole run (byte-identical). "
                             "Training-only (not version-locked; forwarded verbatim on resume, where the anneal "
                             "continues from the resumed step).")
    parser.add_argument("--exploiter-temp-end", dest="exploiter_temp_end", type=float, default=1.0,
                        help="EXPLOITER MODE: the target opponent's temperature at the END of the anneal window "
                             "(default 1.0 = the policy's own distribution, i.e. the target's true strength as a "
                             "stochastic training opponent). Only used when --exploiter-temp-start is set. Set "
                             "below 1.0 to push the target toward greedy (harder) by the end.")
    parser.add_argument("--exploiter-temp-anneal-frac", dest="exploiter_temp_anneal_frac", type=float,
                        default=0.2,
                        help="EXPLOITER MODE: fraction of total --steps over which to linearly anneal the target "
                             "temperature from --exploiter-temp-start to --exploiter-temp-end (default 0.2 = the "
                             "first 20%% of training; held at the end temp after). 0 = constant at "
                             "--exploiter-temp-start (a fixed hotter opponent, no anneal). Only used in the FIXED "
                             "temp mode (--exploiter-temp-mode fixed).")
    parser.add_argument("--exploiter-temp-mode", dest="exploiter_temp_mode",
                        choices=["fixed", "ratchet"], default="fixed",
                        help="EXPLOITER MODE (with --exploiter-temp-start): how the target temperature is "
                             "controlled. 'fixed' (default) = the linear time schedule (--exploiter-temp-anneal-frac). "
                             "'ratchet' = DYNAMIC win-rate-driven: start at --exploiter-temp-start (set it HIGH, e.g. "
                             "5.0, so early games are trivially winnable) and ratchet the temperature DOWN toward "
                             "--exploiter-temp-end only when the trainee's measured TRAINING win-rate vs the target "
                             "clears --exploiter-temp-ratchet-wr — a ONE-WAY auto-curriculum that tracks the trainee's "
                             "competence frontier (never weakens the target, so no comfort-trap). Resume-safe (the "
                             "ratcheted temp is persisted to <run>/exploiter_temp_state.json).")
    parser.add_argument("--exploiter-temp-ratchet-wr", dest="exploiter_temp_ratchet_wr", type=float,
                        default=0.55,
                        help="RATCHET mode: the trainee TRAINING-WR vs the target at which the temperature ratchets "
                             "DOWN (harder). Default 0.55 (keeps play near the ~0.5 max-advantage-signal zone). "
                             "Measured per window of --exploiter-temp-ratchet-games target games.")
    parser.add_argument("--exploiter-temp-ratchet-factor", dest="exploiter_temp_ratchet_factor",
                        type=float, default=0.9,
                        help="RATCHET mode: multiply the temperature by this (<1) on each ratchet (default 0.9 = 10%% "
                             "harder steps). Floored at --exploiter-temp-end.")
    parser.add_argument("--exploiter-temp-ratchet-games", dest="exploiter_temp_ratchet_games",
                        type=int, default=500,
                        help="RATCHET mode: min target-games per decision window before a ratchet check (default 500 "
                             "— the noise guard; larger = smoother/slower).")
    parser.add_argument("--trainee-team", dest="trainee_team", type=str, default=None,
                        help="SPECIALIST MODE: pin the TRAINEE's team pool to the ONE team in this file "
                             "(a Showdown EXPORT string, like data/teams/sample/*.txt), so the agent "
                             "always plays that exact 6-mon team. The OPPONENTS still draw the full "
                             "diverse pool. Use to train a single-team specialist (e.g. --trainee-team "
                             "data/teams/specialist/tss_starmie.txt). Default None = the full trainee "
                             "pool (byte-identical).")
    parser.add_argument("--trainee-teams", dest="trainee_teams", type=str, default=None,
                        help="MULTI-TEAM SPECIALIST MODE: pin the TRAINEE's team pool to the SMALL "
                             "FIXED SET of teams in these files (comma-separated Showdown-export paths), "
                             "sampled UNIFORMLY per episode — a z-near multi-team exploiter (the "
                             "1-vs-3-team A/B). Opponents still draw the full pool. Mutually exclusive "
                             "with --trainee-team; under --exploiter EVERY member must be a sample team. "
                             "Default None.")
    parser.add_argument("--allow-nonsample-trainee", dest="allow_nonsample_trainee", action="store_true",
                        help="RESEARCH override: skip the exploiter vetted-SAMPLE gate so --trainee-team(s) "
                             "may pin NON-sample POOL teams (anchor on a sample, nearest neighbors from all "
                             "719 pool teams → a tighter z-cluster than the 32 samples allow). For FiLM "
                             "capacity / count-vs-diversity studies; NOT for a teacher you'll distil as-is. "
                             "Training-only, not version-locked. Default off (gate enforced).")

    args = parser.parse_args()
    if getattr(args, "trainee_teams", None) and getattr(args, "trainee_team", None):
        parser.error("--trainee-teams (multi-team pin) is mutually exclusive with --trainee-team "
                     "(single-team pin) — use one or the other.")

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
    _resolve("pubval_mode", "none")            # v43 structural + resume-immutable (version-checked)
    _resolve("pubval_coef", 0.1)               # training-only (inherited like win_prob_coef)
    _resolve("value_dist_mode", "none")        # v29 structural + resume-immutable (version-checked)
    _resolve("value_dist_bins", 0)             # v29 structural (atom count; version-checked)
    _resolve("value_dist_vmin", 0.0)           # v29 resume-immutable support (version-checked)
    _resolve("value_dist_vmax", 0.0)           # v29 resume-immutable support (version-checked)
    _resolve("value_dist_coef", 1.0)           # training-only (inherited like win_prob_coef)
    _resolve("value_threat_inject", False)     # v64 structural bool (version-checked, fresh-only)
    _resolve("opp_intent_coef", 0.0)           # v67 training-only coef; the HEADS are structural
    _resolve("beta_setvalued_coef", 0.0)       # training-only coef; no module, no version gate
    _resolve("opp_intent_grad_mode", "detached")  # v73 structural, version-checked
    _resolve("intent_value_reduce", False)     # v74 structural, version-checked (step 6)
    _resolve("intent_move_cell", False)        # v77 structural, version-checked (G3)
    _resolve("value_entity_pool", False)       # v80 structural, version-checked (Stage-3 T3)
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
    elif args.exploiter_temp_mode == "ratchet":
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
    if args.pubval_coef is not None and args.pubval_coef < 0.0:
        parser.error("--pubval-coef must be >= 0 (0 = off; the mode controls on/off)")
    if args.pubval_mode != "none":
        # gen3_pubval_aux_v1: fail FAST if the frozen V_pub artifact is missing/stale — a run that
        # discovered this at env-build time would crash every worker instead of erroring once here.
        from agents.training.pubval import PubValModel
        try:
            PubValModel.load()
        except (FileNotFoundError, ValueError, KeyError) as e:
            parser.error(f"--pubval-mode {args.pubval_mode}: {e}")
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
    if args.opd_coef is not None and args.opd_coef < 0.0:
        parser.error("--opd-coef must be >= 0 (0 = off)")
    if args.opd_coef and args.opd_coef > 0 and not args.search_teacher:
        # OPD distils the beam's π' from the SAME correction buffer the search-teacher fills (its workers
        # build π'), so it can't run standalone.
        parser.error("--opd-coef > 0 requires --search-teacher (OPD distils the search-teacher's "
                     "correction buffer; its workers build the π' targets)")
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
    # Oldest form = bare teacher list + a parallel --distill-teacher-team (kept so in-flight runs resume).
    args._distill_pairs = []
    if args.distill_coef and args.distill_coef > 0:
        _items = [x.strip() for x in (args.distill_teacher or "").split(",") if x.strip()]
        if not _items:
            parser.error("--distill-coef > 0 requires --distill-teacher (as 'TEACHER:TEAM[,TEAM...]' groups)")
        if any(":" in x for x in _items):                       # PREFERRED: colon groups
            if args.distill_teacher_team:
                parser.error("--distill-teacher uses 'TEACHER:TEAM[,TEAM...]' groups — do NOT also pass the "
                             "deprecated --distill-teacher-team")
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
        else:                                                   # LEGACY: bare list + parallel --distill-teacher-team
            print("[Distill] WARNING: bare --distill-teacher + --distill-teacher-team is DEPRECATED; "
                  "prefer 'TEACHER:TEAM' colon pairs in --distill-teacher.")
            _teams = [t.strip() for t in (args.distill_teacher_team or "").split(",") if t.strip()]
            if len(_teams) != len(_items):
                parser.error(f"legacy --distill-teacher ({len(_items)}) / --distill-teacher-team ({len(_teams)}) "
                             "must be equal-length — or use the 'TEACHER:TEAM' pair form in --distill-teacher")
            args._distill_pairs = [(_t, [_tm]) for _t, _tm in zip(_items, _teams)]
    if args.distill_coef and args.distill_coef > 0 and (args.trainee_team or args.trainee_teams):
        parser.error("--distill-coef is mutually exclusive with --trainee-team/--trainee-teams: "
                     "distillation biases the trainee toward the teacher team via --distill-team-bias "
                     "while keeping the pool for rehearsal; a hard pin would remove the rehearsal (and "
                     "cause forgetting)")
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

    # Load all teams using the new TeamLoader
    loader = TeamLoader()
    sample_teams = loader.get_sample_teams()
    all_teams = loader.get_all_teams()
    
    emit(f"📦 {len(sample_teams)} sample teams (bias) / {len(all_teams)} total loaded")

    # THE MATCHUP — declared ONCE (`MatchupSpec.from_args`, designs/ai_v8/design_matchup_config.md)
    # and consumed everywhere: BOTH teambuilders come from the spec (trainee/opponent independent BY
    # CONSTRUCTION — the mirror-bug class), the eval callbacks get the trainee pin from it, the
    # Events panel echoes it, and metadata.json records it (+ spec_hash, the measurement-regime tag).
    # SPECIALIST MODE (--trainee-team) pins ONLY the trainee source; opponents keep the full pool.
    matchup = MatchupSpec.from_args(args)
    # EXPLOITER team-source guarantee: an exploiter may ONLY EVER pilot a vetted sample team (the
    # curated, tournament-proven set) — never a bulk-downloaded `other` team. FATAL otherwise (a
    # deliberate startup gate, like the stable-opponent arch check). Non-exploiter / unpinned runs
    # are unaffected; the existing TSS specialist pin IS a sample team, so it passes.
    if getattr(args, "allow_nonsample_trainee", False):
        # RESEARCH override: skip the vetted-sample gate so an exploiter can pilot whole-POOL z-near
        # teams (anchor on a sample, nearest neighbors from all 719 teams). Use for capacity studies
        # (count-vs-diversity of the FiLM cluster), NOT for a teacher you intend to distil as-is.
        print("⚠️ [Exploiter] --allow-nonsample-trainee: SKIPPING the vetted-sample gate — trainee may "
              "pilot non-sample pool teams (research/capacity mode).")
    else:
        try:
            from agents.training.matchup_spec import validate_exploiter_trainee_is_sample
            validate_exploiter_trainee_is_sample(matchup, sample_teams)
        except ValueError as _e:
            print(f"\n[Exploiter] FATAL: {_e}")
            sys.stdout.flush()
            os._exit(int(TrainExitCode.FATAL_CONFIG))
    # → eval callbacks (trainee_team_str). Read from EVAL_trainee_teams (not trainee_teams) so the
    # distillation path evals on the TAUGHT teams; a `pin_multi` source yields a LIST (eval samples
    # among them, exactly as training does), a single pin yields the raw export, else None = pool.
    _ets = matchup.eval_trainee_teams
    _specialist_team_str = (list(_ets.pin_strs) if _ets.kind == "pin_multi" and _ets.pin_strs
                            else _ets.pin_str)
    # Team-side PFSP threads ONLY into the TRAINEE builder (opponent teams aren't win-rate-sampled);
    # "off" (default) is byte-identical construction.
    trainee_teambuilder = matchup.trainee_teams.build(
        all_teams, sample_teams,
        team_pfsp=args.team_pfsp, team_pfsp_cap=args.team_pfsp_cap,
        team_pfsp_floor=args.team_pfsp_floor)
    # Team-blocked episodes: hold each drawn trainee team for N consecutive episodes — the
    # per-team gradient-density counter to the measured FiLM sample starvation. Trainee side ONLY
    # (opponent draws stay per-episode); 1 = off, byte-identical. Training-only, not version-locked.
    if args.team_block_episodes > 1:
        trainee_teambuilder.set_block_episodes(args.team_block_episodes)
    opponent_teambuilder = matchup.opponent_teams.build(all_teams, sample_teams)
    # gen3_exploiter_distill_v1: bias the trainee toward the teacher's team (rest = pool rehearsal) so it
    # gets enough distillation signal, and precompute the teacher team's species id-set for the env's
    # per-state `distill_mask`. --distill-coef is mutually exclusive with --trainee-team (a pin would
    # defeat the mixed-team rehearsal that guards against forgetting).
    args._distill_species = None
    if getattr(args, "_distill_pairs", None):
        from poke_env.teambuilder.teambuilder import Teambuilder as _TB
        from poke_env.data.normalize import to_id_str as _to_id
        _species_sets, _team_strs = [], []
        for _tp, _tfs in args._distill_pairs:
            _sets = []
            for _tf in _tfs:
                with open(_tf, encoding="utf-8") as _df:
                    _s = _df.read()
                _team_strs.append(_s)
                # poke-env parks the species in `nickname` when the export has no nickname → fall back to it.
                _sets.append(frozenset(_to_id(m.species or m.nickname) for m in _TB.parse_showdown_team(_s)))
            _species_sets.append(_sets)
        # list (per TEACHER, teacher-id = index+1) of LISTS of species-frozensets (that teacher's teams) —
        # a multi-team teacher's KL fires on ANY of its teams (the env matches `cur in sp_list`).
        args._distill_species = _species_sets
        # Bias the trainee across ALL N teacher teams (bias_prob total, split evenly); rest = pool rehearsal.
        trainee_teambuilder = Gen3Teambuilder(all_teams, bias_teams=_team_strs,
                                              bias_prob=args.distill_team_bias,
                                              team_pfsp=args.team_pfsp,
                                              team_pfsp_cap=args.team_pfsp_cap,
                                              team_pfsp_floor=args.team_pfsp_floor)
        emit(f"🧪 [DISTILL] {len(args._distill_pairs)} teacher(s) / {len(_team_strs)} team(s), "
             f"coef={args.distill_coef} | trainee biased {args.distill_team_bias:.0%} across all "
             f"{len(_team_strs)} teacher team(s); rest = pool rehearsal")
        for _i, (_tp, _tfs) in enumerate(args._distill_pairs, start=1):
            emit(f"   [{_i}] {_tp} ← {len(_tfs)} team(s): "
                 f"{', '.join(os.path.basename(_f) for _f in _tfs)}")
    for _ln in matchup.summary_lines():
        emit(_ln)
    if matchup.trainee_teams.kind == "pin_multi":
        _tt = matchup.trainee_teams
        emit(f"🎯 [MULTI-SPECIALIST] trainee pinned to {len(_tt.pin_strs)} teams (sampled uniformly): "
             f"{', '.join(os.path.basename(f) for f in _tt.pin_files)} (opponents keep the full pool)")
    elif _specialist_team_str:
        _spec_mons = [ln.split("@")[0].split("(")[0].strip()
                      for ln in _specialist_team_str.splitlines()
                      if ln.strip() and "@" in ln]
        emit(f"🎯 [SPECIALIST] trainee pinned to ONE team from {args.trainee_team}: "
             f"{', '.join(_spec_mons)} (opponents keep the full pool)")

    # RESUME MATCHUP-DRIFT GUARD: matchup flags (--trainee-team/--exploiter/--bot-weights/…) are
    # NOT resume-immutable — a mid-run curriculum change is legitimate — but it must never be
    # SILENT: a resume whose declared matchup differs from what the run last recorded overwrites
    # cli_args and changes the training distribution. Warn LOUDLY with the field diff; the new era
    # is appended to metadata `matchup_history` at the next save (save_model_snapshot), so the
    # run's full regime timeline survives. (A launcher restart forwards flags verbatim → no drift.)
    if args.model:
        from agents.model.snapshot import read_recorded_matchup
        from agents.training.matchup_spec import describe_drift
        _rec_hash, _rec_spec = read_recorded_matchup(args.model)
        if _rec_hash and _rec_hash != matchup.spec_hash():
            emit(f"⚠️ [MATCHUP DRIFT] this resume declares matchup {matchup.spec_hash()} but the "
                 f"run last recorded {_rec_hash} — the TRAINING DISTRIBUTION IS CHANGING mid-run. "
                 "Metrics across the change are NOT comparable (a new era lands in "
                 "metadata.json:matchup_history).")
            for _d in describe_drift(_rec_spec, matchup.to_dict()):
                emit(f"   ⚠️ {_d}")

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
        # A specialist opponent shows its fold-back pin (it pilots ITS OWN team, training + eval).
        _stable_labels = ", ".join(
            e.label + (f" [pilots ITS OWN pin: {os.path.basename(e.team_file)}]" if e.team_str else "")
            for e in _fixed_opponents)
        if args.self_play:
            emit(f"🐴 [STABLE] {len(_fixed_opponents)} cross-run opponent(s): {_stable_labels} — "
                 f"eval greedy; training ≤{args.stable_opponent_selfplay_share:.0%} of self-play until "
                 f"mastered (win_rate ≥ {args.stable_opponent_mastered_wr:.0%})")
        else:
            emit(f"🐴 [STABLE] {len(_fixed_opponents)} cross-run opponent(s): {_stable_labels} — "
                 "EVAL-ONLY (no --self-play, so they don't join the training mix)")

    # EXPLOITER mode (--exploiter): resolve the single fixed target the SAME way as a stable opponent
    # (run-dir/checkpoint spec → arch-gated FixedOpponentEntry), validating its weights load here so a
    # corrupt zip FATALs once up front instead of crashing every env worker. The env factory builds one
    # RLPlayer from it per worker; the wrapper then uses it as the sole training opponent. (Mutual
    # exclusivity with --self-play is enforced at arg-parse time above.)
    _exploiter_entry = None
    if args.exploiter:
        from agents.training.fixed_opponent_pool import resolve_stable_opponents
        from agents.model.snapshot import (
            current_model_version as _current_model_version, load_foreign_opponent)
        _cv_expl = _current_model_version(mappings, **_run_arch_toggles(args))
        # gen3_exploiter_temp_anneal_v1: when annealing the target's temperature, START it at
        # --exploiter-temp-start (so the very first episodes are already at the curriculum's hot temp,
        # before ExploiterTempAnnealCallback's first per-rollout push); else the fixed
        # --stable-opponent-temp (unchanged default).
        _expl_temp0 = (args.exploiter_temp_start if args.exploiter_temp_start is not None
                       else args.stable_opponent_temp)
        try:
            _resolved = resolve_stable_opponents(args.exploiter, _cv_expl,
                                                 default_temperature=_expl_temp0)
            if len(_resolved) != 1:
                raise ValueError(f"--exploiter takes exactly ONE target model, got {len(_resolved)}")
            _exploiter_entry = _resolved[0]
            load_foreign_opponent(_exploiter_entry.zip_path, current_version=_cv_expl, device="cpu",
                                  config_path=_exploiter_entry.config_path)  # validate weights load
        except (ModelVersionError, FileNotFoundError, ValueError) as e:
            print(f"\n[Exploiter] FATAL: {e}")
            sys.stdout.flush()
            os._exit(int(TrainExitCode.FATAL_CONFIG))
        except Exception as e:  # noqa: BLE001 — corrupt/unreadable foreign weights zip
            print(f"\n[Exploiter] FATAL: failed to load exploiter target weights: {e}")
            sys.stdout.flush()
            os._exit(int(TrainExitCode.FATAL_CONFIG))
        if args.exploiter_temp_start is None:
            _temp_desc = f"temp {args.stable_opponent_temp:g}"
        elif args.exploiter_temp_mode == "ratchet":
            _temp_desc = (f"temp {args.exploiter_temp_start:g}→{args.exploiter_temp_end:g} WR-RATCHETED "
                          f"(harder when train-WR ≥ {args.exploiter_temp_ratchet_wr:.0%})")
        else:
            _temp_desc = (f"temp {args.exploiter_temp_start:g}→{args.exploiter_temp_end:g} annealed over "
                          f"{args.exploiter_temp_anneal_frac:.0%} of training")
        if args.exploiter_keep_bots:
            emit(f"🥊 [EXPLOITER] training vs {_exploiter_entry.label} ({_temp_desc}) "
                 f"with the heuristic bots MIXED IN: per episode P(target)={1 - args.exploiter_bot_fraction:.0%}, "
                 f"P(bot)={args.exploiter_bot_fraction:.0%}. Goal: learn to beat the target while keeping a bot floor.")
        else:
            emit(f"🥊 [EXPLOITER] training vs {_exploiter_entry.label} as the SOLE opponent every episode "
                 f"({_temp_desc}; no self-play/pool/bots). Goal: learn to beat it.")
        if _exploiter_entry.team_str:
            emit(f"   target pilots ITS OWN pinned team ({os.path.basename(_exploiter_entry.team_file)}) "
                 "— the fold-back contract")

    # Opponent-parity Proposal A: the exploiter target AUTO-registers as an eval opponent, so the
    # verdict metric (eval/win_rate_vs_ext_<target>) exists without remembering to duplicate the
    # target in --stable-opponents. Dedup-guarded — the historical both-flags recipe is unchanged.
    # Training-mix side is untouched (exploiter mode excludes --self-play → the entry is eval-only).
    if _exploiter_entry is not None:
        from agents.training.fixed_opponent_pool import register_exploiter_for_eval
        _fixed_opponents, _expl_registered = register_exploiter_for_eval(
            _fixed_opponents, _exploiter_entry)
        if _expl_registered:
            emit(f"🥊 [EXPLOITER] target auto-registered for eval as {_exploiter_entry.label} "
                 f"(greedy verdict metric eval/win_rate_vs_{_exploiter_entry.label})")

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
                                   heuristic_weights=None, stable_opponents=None,
                                   exploiter_entry=None):
        def _init():
            try:
                # Defensive per-worker pin (the module-level env vars are the primary guard, but they
                # are `setdefault` — an explicit OMP_NUM_THREADS=8 for the learner must not turn every
                # worker's B=1 opponent forward into an N-thread contender). Mirrors
                # snapshot_ladder.py's pin for the same reason: B=1 CPU inference gets nothing from
                # intra-op parallelism, and the parallelism that matters is ACROSS workers.
                import torch as _torch
                _torch.set_num_threads(1)
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
                    emit_win_target=(args.win_prob_mode != "none"),
                    emit_pubval_target=(args.pubval_mode != "none"),
                    # SPREAD-belief supervision (gen3_unified_spread_belief_v1): emit the privileged
                    # true-spread label only when the loss will consume it (coef>0; the CLI guards that
                    # --spread-belief-coef requires --spread-belief, so the head is present to supervise).
                    emit_spread_labels=(args.spread_belief and args.spread_belief_coef > 0.0),
                    emit_opp_intent_labels=(getattr(args, 'opp_intent_coef', 0.0) > 0.0),
                    # HP-TYPE-belief supervision (gen3_typed_hp_belief_v1): emit the privileged true-HP-type
                    # label only when the CE will consume it (the head itself is unconditional under a move
                    # belief; the CLI guards that the coef implies one).
                    emit_hp_type_labels=(args.move_belief_mode != "off" and args.hp_belief_mode == "composed"
                                         and args.hp_type_belief_coef > 0.0),
                    # DEFENSIVE-exploration flag (gen3_defensive_entropy_v1): emit only when the boost is on, so
                    # the state-conditioned entropy term in the PPO loss can read it. Off = no key, no cost.
                    emit_defensive_opportunity=(args.defensive_entropy_boost > 1.0),
                    # EXPLOITER DISTILLATION (gen3_exploiter_distill_v1): the teacher team's species id-set
                    # (None unless --distill-coef>0). The env emits `distill_mask`=1 on states where the
                    # trainee pilots this team — the only states the distillation KL folds. None → no key.
                    distill_team_species=getattr(args, "_distill_species", None),
                    # The OPPONENT side's real team source (agent2 does the networking for every
                    # per-episode opponent; the rotated Players are decision-functions whose own
                    # builders are inert). Without this, PokeEnv fed `team=` (the TRAINEE builder)
                    # to BOTH sides — a --trainee-team pin made every battle a single-team MIRROR.
                    opponent_team=opponent_teambuilder,
                )
                if args.use_showdown_bridge:
                    # Swap the two _EnvPlayer agents' websocket transport for a local
                    # BattleStream subprocess. Everything above the transport (obs, reward,
                    # mask, wrappers) is unchanged — see utils/bridge/bridge_session.py.
                    attach_bridge_transport(env, battle_format=BATTLE_FORMAT,
                                            impl=args.bridge_impl)

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
                        pfsp_scale=getattr(args, "pfsp_scale", 0.0),
                        pool_spread=getattr(args, "pool_spread", False),
                        compile_extractor=args.compile_opponents,
                        compile_hide_cuda=True,       # spawned env worker — never take a CUDA context
                        compile_strict=args.compile_opponents_strict,
                    )
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
                stable_players, stable_labels, stable_teams = [], [], []
                if self_play and stable_opponents:
                    from agents.model.snapshot import (load_foreign_opponent,
                                                        maybe_compile_extractor)
                    from utils.teambuilder import Gen3Teambuilder as _G3TB
                    for e in stable_opponents:
                        opp_model, _ = load_foreign_opponent(
                            e.zip_path, current_version=opponent_version,
                            device=opponent_device, config_path=e.config_path)
                        # hide_cuda=True: this runs in a spawned env worker, where a CUDA context
                        # would cost ~252 MiB of card per worker (the June 48× OOM).
                        maybe_compile_extractor(opp_model, args.compile_opponents,
                                                label=f"stable:{e.label}", hide_cuda=True,
                                                strict=args.compile_opponents_strict)
                        # Fold-back: a specialist opponent pilots ITS OWN pinned team (entry
                        # team_str from its run's metadata); the wrapper switches agent2._team to
                        # this builder on the episodes it plays. None = pool pilot (generalist).
                        _pin_tb = _G3TB(list(e.team_strs)) if e.team_strs else None   # multi-team specialist samples among ITS OWN teams
                        stable_players.append(RLPlayer(
                            model=opp_model, team=(_pin_tb or opponent_teambuilder),
                            battle_format=BATTLE_FORMAT,
                            server_configuration=server_config, mappings=mappings,
                            account_configuration=AccountConfiguration(
                                f"Opp{idx}s{len(stable_players)}{ts}", "password"),
                            start_listening=False,
                            stochastic=True, temperature=e.temperature,
                        ))
                        stable_labels.append(e.label)
                        stable_teams.append(_pin_tb)

                # EXPLOITER mode: one fixed target loaded ONCE per worker → the sole opponent. Same
                # foreign-load path as a stable opponent; stochastic at the stable-opponent temp so
                # it stays a moving target (harder to over-exploit a frozen target's quirks).
                exploiter_player = None
                exploiter_team = None
                if exploiter_entry is not None:
                    from agents.model.snapshot import (load_foreign_opponent,
                                                        maybe_compile_extractor)
                    from utils.teambuilder import Gen3Teambuilder as _G3TB
                    _ex_model, _ = load_foreign_opponent(
                        exploiter_entry.zip_path, current_version=opponent_version,
                        device=opponent_device, config_path=exploiter_entry.config_path)
                    maybe_compile_extractor(_ex_model, args.compile_opponents,
                                            label="exploiter-target", hide_cuda=True,
                                            strict=args.compile_opponents_strict)
                    # Fold-back: an exploiter-of-a-specialist faces the target ON ITS OWN pinned team.
                    exploiter_team = (_G3TB(list(exploiter_entry.team_strs))
                                      if exploiter_entry.team_strs else None)
                    exploiter_player = RLPlayer(
                        model=_ex_model, team=(exploiter_team or opponent_teambuilder),
                        battle_format=BATTLE_FORMAT,
                        server_configuration=server_config, mappings=mappings,
                        account_configuration=AccountConfiguration(f"Opp{idx}x{ts}", "password"),
                        start_listening=False,
                        stochastic=True, temperature=exploiter_entry.temperature,
                    )

                wrapped = MaskableAgentWrapper(
                    env, heuristic_opponents=heuristic_opponents, pool=pool,
                    pool_player=pool_player, self_play_fraction=self_play_fraction, rng_seed=idx,
                    heuristic_weights=heuristic_weights,
                    stable_players=stable_players, stable_labels=stable_labels,
                    stable_challenge_share=args.stable_opponent_selfplay_share,
                    stable_pfsp=args.stable_opponent_pfsp,
                    exploiter_player=exploiter_player,
                    # Fold-back per-opponent teams: pinned builders (or None) parallel to
                    # stable_players, the exploiter target's pin, and the pool builder to restore
                    # on unpinned episodes. All-None → the wrapper never touches agent2._team.
                    stable_teams=stable_teams, exploiter_team=exploiter_team,
                    opponent_pool_team=opponent_teambuilder,
                    # keep-bots: the heuristic roster (always built above) is mixed back in
                    # per-episode alongside the exploiter target. No-op unless exploiter_player is set.
                    exploiter_keep_bots=args.exploiter_keep_bots,
                    exploiter_bot_fraction=args.exploiter_bot_fraction,
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
        # A memorable --run-name (models/<name>), an exploiter default (models/exploiter_vs_<target>),
        # or the legacy date-stamp — with a guard against clobbering an existing run.
        model_dir = _resolve_fresh_model_dir(
            args.run_name,
            _exploiter_entry.label if _exploiter_entry is not None else None,
            args.model)

    os.makedirs(model_dir, exist_ok=True)
    # Full CLI namespace (JSON-safe) → persisted into metadata.json for run provenance.
    cli_args = json.loads(json.dumps(vars(args), default=str))
    # Matchup provenance (designs/ai_v8/design_matchup_config.md): the DECLARED matchup + its hash
    # ride into metadata.json beside the flags, so a run's measurement regime is auditable — two
    # eras with different hashes (e.g. the pre-fix OOD-eval era) are not metric-comparable.
    cli_args["_matchup_spec"] = matchup.to_dict()
    cli_args["_matchup_spec_hash"] = matchup.spec_hash()
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

    # --compile-opponents: warm the SHARED on-disk Inductor cache in THIS process before any env
    # worker exists, so the workers all hit it warm instead of racing on a cold one (measured
    # 59.6 s -> 30.1 s wall for 16 workers). Uses the same arch table as the model build below, so
    # the cached codegen is for the graph the workers will actually run.
    #
    # An earlier attempt went further — `set_forkserver_preload` on a module that compiles at
    # import, so the graph is traced ONCE and every worker inherits it by fork (0.12 s per worker).
    # It is NOT viable and must not be retried without fixing the cause: forking is only safe from a
    # SINGLE-THREADED process, and the forkserver ends up with at least two extra threads —
    # Inductor's 16-way parallel-codegen pool (which survives the compile) and poke-env's global
    # asyncio loop thread, started at import by `agents.model.features_extractor`'s transitive
    # poke-env dependency. A 48-env run with that preload forked 2 workers instead of 48 and hung
    # forever, parent blocked in `unix_stream_data_wait`, box at 0.2 load. Guarded by
    # `compile_prewarm_test.py`.
    if args.compile_opponents and not args.debug:
        prewarm_extractor_compile(build_extractor_arch_kwargs(args), mappings)

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
        _pool = SnapshotPool(pool_dir=_snapshot_dir, current_version=_cv, device=args.device,
                             pfsp_scale=args.pfsp_scale, pool_spread=args.pool_spread)
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

    # Exploiter mode is NOT self-play, so the self-play block above left _opp_version=None — but the
    # env factory still needs it to arch-gate the exploiter target's foreign load. Set it here.
    if _exploiter_entry is not None and _opp_version is None:
        from agents.model.snapshot import current_model_version as _current_model_version
        _opp_version = _current_model_version(mappings, **_run_arch_toggles(args))

    def _make_factories():
        return [
            create_training_env_random(
                i, stall_config=stall_cfg, opponent_device=opponent_device,
                opponent_version=_opp_version,
                snapshot_dir=str(_snapshot_dir) if _snapshot_dir is not None else None,
                self_play_fraction=_initial_self_play_fraction, self_play=args.self_play,
                heuristic_weights=_bot_weight_vec,
                stable_opponents=_fixed_opponents,
                exploiter_entry=_exploiter_entry,
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
        _n_opp = 9
        print(f"\nFinal Evaluation (Session {ts}, Battles: {n}, Concurrency: {args.eval_concurrency})...")
        # Say the QUANTITY of work out loud. "Training complete" is printed by the caller BEFORE
        # this runs, so without a line here a still-working process is indistinguishable from a hung
        # one — six timeouts were spent proving exactly that.
        print(f"  ~{_n_opp * n} battles ({_n_opp} opponents x {n}). Training IS finished and the "
              f"model is saved; this is the post-training measurement and it can take minutes.",
              flush=True)

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
                                        concurrency=min(args.eval_concurrency, 8),
                                        impl=args.bridge_impl)
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
    # gen3_exploiter_temp_anneal_v1: control the EXPLOITER target's sampling temperature over training
    # (a difficulty curriculum via opponent stochasticity — hot/weak early → true strength later),
    # pushed to every env's exploiter RLPlayer via env_method each rollout. Registered ONLY when
    # --exploiter-temp-start is set → an off run makes no push (byte-identical). Training-only.
    # 'fixed' = linear time schedule; 'ratchet' = dynamic win-rate-driven one-way ratchet.
    if args.exploiter and args.exploiter_temp_start is not None:
        from agents.training.exploiter_temp_callback import (
            ExploiterTempAnnealCallback, ExploiterTempRatchetCallback)
        if args.exploiter_temp_mode == "ratchet":
            callbacks.append(ExploiterTempRatchetCallback(
                temp_start=args.exploiter_temp_start, temp_end=args.exploiter_temp_end,
                threshold=args.exploiter_temp_ratchet_wr, factor=args.exploiter_temp_ratchet_factor,
                min_games=args.exploiter_temp_ratchet_games, run_dir=model_dir))
        else:
            callbacks.append(ExploiterTempAnnealCallback(
                temp_start=args.exploiter_temp_start, temp_end=args.exploiter_temp_end,
                anneal_frac=args.exploiter_temp_anneal_frac))
    # Win-probability head: captures each episode's win/loss outcome during collection + back-fills the
    # rollout buffer's MC label before train() (only when the head is on → a default run pays nothing).
    if args.win_prob_mode != "none":
        from agents.training.win_prob_callback import WinProbLabelCallback
        callbacks.append(WinProbLabelCallback())
    # Team-side PFSP: variance-weighted TEAM sampling by self-play win-rate. Registered ONLY when on
    # → an off run adds no callback and makes no env_method calls (byte-identical). Training-only.
    if args.team_pfsp != "off":
        from agents.training.team_pfsp_callback import TeamPFSPCallback
        callbacks.append(TeamPFSPCallback(cap=args.team_pfsp_cap, floor=args.team_pfsp_floor,
                                          mode=args.team_pfsp, persist_dir=model_dir))
    # SEARCH-TEACHER: each cycle, search + confirm the worst loss craters and distil verified-better
    # corrections into model._correction_buffer (the AWR aux loss samples it). Non-blocking subprocess
    # workers; off by default (the buffer fills nothing → coef-0 loss is byte-identical regardless).
    if args.search_teacher:
        from agents.training.teacher.callback import SearchTeacherCallback
        callbacks.append(SearchTeacherCallback(
            run_dir=model_dir,
            freq_steps=(args.teacher_search_freq if args.teacher_search_freq > 0 else 2_000_000),
            budget=args.teacher_search_budget, n_workers=args.teacher_search_workers,
            confirm_rollouts=args.teacher_confirm_rollouts,
            persistent=args.teacher_persistent, refresh_steps=args.teacher_refresh_steps,
            n_battles=args.teacher_gen_battles,
            # OPD: when --opd-coef>0 the workers ALSO build the improved distribution π' (the KL target).
            opd_build_pi_target=bool(args.opd_coef and args.opd_coef > 0), opd_beta=args.opd_beta,
            # The workers' sim engine follows the run's --use-bridge impl (no separate flag);
            # "node" when the bridge is off, which is the historical behaviour.
            impl=args.bridge_impl,
            verbose=1))
    eval_callback = None
    # A --debug smoke run skips ALL eval by default — the periodic eval callback below AND the
    # final win-rate eval — so it needs no eval opponents / Showdown eval connection and stays
    # light on CPU. --debug-eval opts back in. Real (non-debug) runs are unaffected (always True).
    _run_eval = (not args.debug) or args.debug_eval
    # gen3_smoke_eval_scale_v1: resolve --eval-battles. An explicit value ALWAYS wins; otherwise a
    # short run gets the smoke count. Keyed on --steps rather than on a new --smoke flag so it needs
    # nothing remembered at the call site, and it cannot quietly weaken a real run: at 15M steps the
    # condition is false and the default stays 100.
    if args.eval_battles is None:
        _smoke = int(getattr(args, "steps", 0) or 0) < SMOKE_STEPS
        args.eval_battles = SMOKE_EVAL_BATTLES if _smoke else DEFAULT_EVAL_BATTLES
        if _smoke and _run_eval:
            print(f"[SmokeEval] --steps {args.steps:,} < {SMOKE_STEPS:,}: final eval scaled to "
                  f"{SMOKE_EVAL_BATTLES} battles/opponent (pass --eval-battles N to override). "
                  f"These win rates are NOT a measurement — they only prove the eval path runs.")

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
            eval_games=args.eval_games,
            snapshot_ladder_games=args.snapshot_ladder_games,
            model_dir=model_dir,
            server_config=server_config,
            showdown_port=args.showdown_port,
            use_showdown_bridge=args.use_showdown_bridge,
            compile_extractor=args.compile_opponents,
            bridge_impl=args.bridge_impl,
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
            # Self-play eval is ~2x the inference of bot eval — the sentinel matchups
            # (--n-sentinels) run the model for BOTH players (trainee + sentinel), vs bot matchups
            # where only the trainee infers. So double the work-stealing pool to keep wall-clock
            # comparable (5 bot-eval workers → 10 here); raise --eval-workers too if --n-sentinels
            # is pushed high so the extra sentinel shards still drain promptly.
            n_workers=args.eval_workers * 2,
            eval_device=args.eval_device,
            eval_concurrency=args.eval_concurrency_per_worker,
            eval_shard_games=args.eval_shard_games,
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
            stable_pfsp=args.stable_opponent_pfsp,
            bot_weight_vec=_bot_weight_vec,
            floor_roster_count=len(OPPONENT_CLASSES),
            # PFSP: when >0 the callback EMA-smooths the per-sentinel win-rates each eval and pushes
            # them to the env pools so sampling oversamples the selves we're losing to (0.0 = off).
            pfsp_scale=args.pfsp_scale,
            n_sentinels=args.n_sentinels,
            debug=args.debug,
            # --trainee-team pin → eval measures the trainee ON ITS OWN TEAM (None = default pool).
            trainee_team_str=_specialist_team_str,
        )
        callbacks.append(eval_callback)
    elif _run_eval:
        # Bot eval runs in a frozen-snapshot subprocess (non-blocking, CPU). The
        # worker rebuilds opponents/teambuilders/mappings itself from the data
        # dir, so nothing live is constructed here.
        eval_callback = PerOpponentEvalCallback(
            model_dir=model_dir,
            eval_games=args.eval_games,
            server_config=server_config,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            n_workers=args.eval_workers,
            eval_device=args.eval_device,
            eval_concurrency=args.eval_concurrency_per_worker,
            eval_shard_games=args.eval_shard_games,
            showdown_port=args.showdown_port,
            use_showdown_bridge=args.use_showdown_bridge,
            compile_extractor=args.compile_opponents,
            bridge_impl=args.bridge_impl,
            resume_eval_metadata=_resume_meta,
            keep_eval_snapshots=args.keep_eval_snapshots,
            keep_eval_trace_steps=args.keep_eval_trace_steps,
            keep_stalls=args.keep_stalls,
            keep_crashes=args.keep_crashes,
            fixed_opponents=_fixed_opponents,
            # --trainee-team pin → eval measures the trainee ON ITS OWN TEAM (None = default pool).
            trainee_team_str=_specialist_team_str,
        )
        callbacks.append(eval_callback)

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
            from agents.training.warmstart import run_consensus_warmstart
            from agents.training.fixed_opponent_pool import _resolve_zip_and_config as _ws_resolve
            _ws_dir = os.path.join(model_dir, "warmstart")
            _ws_ckpt = os.path.join(_ws_dir, "warmstart_consensus.zip")
            if not os.path.exists(_ws_ckpt):
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
            "use_popart": args.use_popart,  # version-checked vs the saved model_config.json
            "value_from_dist": args.value_from_dist,  # Phase B: dist head is the critic (resume-immutable)
        }
        current_version = ModelVersion.from_layout_and_policy_kwargs(
            _load_extractor_kwargs["layout"], _load_policy_kwargs, vf_coef=args.vf_coef,
            reward_config=reward_config, value_tail_weight=args.value_tail_weight,
            opp_belief_aux_coef=args.opp_belief_aux_coef,
            move_belief_coef=args.move_belief_coef,
            win_prob_coef=args.win_prob_coef,
            pubval_coef=args.pubval_coef,
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
                enforce_belief_grad_mode=args.belief_grad_mode,  # FATAL if the belief-trunk-grad mode drifts (v41)
                allow_belief_grad_mode_change=args.allow_belief_grad_mode_change,  # intentional migration
                enforce_value_from_dist=args.value_from_dist,  # FATAL if the Phase-B critic source drifts (v45)
                allow_value_from_dist_change=args.allow_value_from_dist_change,
            )
            # gen3_belief_grad_mode_v1 MIGRATION FIX: SB3 reconstructs the extractor from the ZIP's
            # saved policy_kwargs, so the requested mode must be APPLIED to the live extractor
            # post-load (else --allow-belief-grad-mode-change is a silent no-op — the 2026-07-21
            # incident, visible as grad/*_norm_shared == 0 under 'shaping'). No-op when unchanged.
            model.policy.features_extractor.set_belief_grad_mode(args.belief_grad_mode)
            # gen3_dist_critic_v1 (Phase B) MIGRATION FIX: same silent-no-op class — the loaded policy
            # is rebuilt from the ZIP's saved policy_kwargs (a pre-v45 checkpoint lacks value_from_dist),
            # so apply the requested source to the live policy post-load (no-op when unchanged).
            model.policy.set_value_from_dist(args.value_from_dist)
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
        model.ent_coef = args.ent_coef
        model.value_tail_weight = args.value_tail_weight  # == saved (enforced above); set for the loop
        model.opp_belief_aux_coef = args.opp_belief_aux_coef  # training hparam (not version-locked; resume-mutable)
        model.opp_belief_moves_weight = args.opp_belief_moves_weight
        model.move_belief_coef = args.move_belief_coef  # move-belief loss weight (training-only; resume-mutable)
        model.move_belief_latent_coef = args.move_belief_latent_coef  # move-latent grading weight (training-only)
        model.spread_belief_coef = args.spread_belief_coef  # spread-belief speed-supervision weight (training-only)
        model.defensive_entropy_boost = args.defensive_entropy_boost            # gen3_defensive_entropy_v1 (training-only)
        model.defensive_entropy_anneal_frac = args.defensive_entropy_anneal_frac
        model.hp_type_belief_coef = args.hp_type_belief_coef  # HP-type CE weight (training-only)
        model.win_prob_coef = args.win_prob_coef  # win-prob loss weight (training-only; resume-mutable)
        model.pubval_coef = args.pubval_coef  # pubval loss weight (training-only; resume-mutable)
        model.value_dist_coef = args.value_dist_coef  # value-dist HL-Gauss loss weight (training-only; resume-mutable)
        # SEARCH-TEACHER (training-only; coef 0 / flag absent = byte-identical). Buffer is filled by the
        # SearchTeacherCallback from worker shards; the AWR aux loss in train() samples it.
        model.search_teacher_coef = args.search_teacher_coef
        model.search_teacher_value_coef = args.search_teacher_value_coef
        model.search_teacher_beta = args.search_teacher_beta
        model.search_teacher_batch_size = args.search_teacher_batch_size
        model._search_teacher_on = bool(args.search_teacher)
        # OPD (on-policy self-distillation): training-only (coef 0 = byte-identical, NOT version-locked).
        # Requires --search-teacher (it fills the SAME _correction_buffer, its workers building π').
        model.opd_coef = args.opd_coef
        model.opp_intent_coef = float(getattr(args, 'opp_intent_coef', 0.0) or 0.0)
        model.beta_setvalued_coef = float(getattr(args, 'beta_setvalued_coef', 0.0) or 0.0)
        # gen3_exploiter_distill_v1: attach the frozen per-team teacher (foreign exploiter) on the training
        # device + set the KL weight (training-only). OFF (coef 0 / no teacher) → _distill_teacher stays
        # None so the loss block is skipped (byte-identical). A bad path FATALs config (no crash-restart loop).
        model.distill_coef = float(args.distill_coef or 0.0)
        model.distill_value_coef = float(args.distill_value_coef or 0.0)
        model.distill_value_feat_coef = float(args.distill_value_feat_coef or 0.0)  # gen3_exploiter_value_feat_distill_v1
        model._distill_teachers = []   # gen3_exploiter_distill_v1: N frozen per-team teachers (teacher-id = index+1)
        if args.distill_coef and args.distill_coef > 0 and getattr(args, "_distill_pairs", None):
            from agents.model.snapshot import (
                current_model_version as _cmv_d, load_foreign_opponent as _lfo_d)
            from agents.training.fixed_opponent_pool import _resolve_zip_and_config as _rzc_d
            _cv_d = _cmv_d(mappings, **_run_arch_toggles(args))
            for _tp, _tf in args._distill_pairs:
                try:
                    _zip_d, _cfg_d, _ = _rzc_d(_tp, None)   # run-dir → (zip, config)
                    _tm_d, _ = _lfo_d(_zip_d, current_version=_cv_d,
                                      device=str(model.device), config_path=_cfg_d)
                    _tm_d.policy.set_training_mode(False)
                    model._distill_teachers.append(_tm_d)
                except Exception as _e_d:  # noqa: BLE001 — bad/incompatible teacher weights
                    print(f"\n[Distill] FATAL: could not load --distill-teacher {_tp}: {_e_d}")
                    sys.stdout.flush()
                    os._exit(int(TrainExitCode.FATAL_CONFIG))
            emit(f"🧪 [DISTILL] {len(model._distill_teachers)} teacher(s) attached on {model.device} "
                 f"(order = teacher-id 1..{len(model._distill_teachers)})")
        model.opd_beta = args.opd_beta
        model._opd_on = bool(args.opd_coef and args.opd_coef > 0)
        if args.search_teacher:
            from agents.training.teacher.buffer import CorrectionBuffer
            model._correction_buffer = CorrectionBuffer(args.search_teacher_buffer_size)
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
            _maybe_compile_trainer(model, args)
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
        # Initialize a dummy encoder to get the layout handoff kwargs, then layer every
        # version-checked arch toggle on top via the SHARED table (agents.model.extractor_arch)
        # that the resume path above also uses.
        temp_encoder = Gen3ObservationEncoder(mappings)
        extractor_kwargs = build_extractor_arch_kwargs(
            args, base=temp_encoder.get_features_extractor_kwargs(), log_level=log_level)

        policy_kwargs = {
            "features_extractor_class": Gen3FeaturesExtractor,
            "features_extractor_kwargs": extractor_kwargs,
            "net_arch": [512, 512],
            "optimizer_class": torch.optim.AdamW,
            "optimizer_kwargs": {"weight_decay": args.weight_decay, "eps": 1e-5},
            "use_popart": args.use_popart,  # builds the PopArtNormalizer in the policy; recorded in model_config.json
            "value_from_dist": args.value_from_dist,  # Phase B: GAE reads E[Z]; recorded in model_config.json
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
        model.hp_type_belief_coef = args.hp_type_belief_coef  # HP-type CE loss (0.0 = no direct CE)
        model.win_prob_coef = args.win_prob_coef  # win-prob head BCE loss (mode none = off)
        model.pubval_coef = args.pubval_coef  # pubval head soft-BCE (mode none = off)
        model.value_dist_coef = args.value_dist_coef  # value-dist HL-Gauss loss (mode none = off)
        # SEARCH-TEACHER (training-only; coef 0 / flag absent = byte-identical). See the resume site.
        model.search_teacher_coef = args.search_teacher_coef
        model.search_teacher_value_coef = args.search_teacher_value_coef
        model.search_teacher_beta = args.search_teacher_beta
        model.search_teacher_batch_size = args.search_teacher_batch_size
        model._search_teacher_on = bool(args.search_teacher)
        # OPD (on-policy self-distillation): training-only (coef 0 = byte-identical, NOT version-locked).
        # Requires --search-teacher (it fills the SAME _correction_buffer, its workers building π').
        model.opd_coef = args.opd_coef
        model.opp_intent_coef = float(getattr(args, 'opp_intent_coef', 0.0) or 0.0)
        model.beta_setvalued_coef = float(getattr(args, 'beta_setvalued_coef', 0.0) or 0.0)
        # gen3_exploiter_distill_v1: attach the frozen per-team teacher (foreign exploiter) on the training
        # device + set the KL weight (training-only). OFF (coef 0 / no teacher) → _distill_teacher stays
        # None so the loss block is skipped (byte-identical). A bad path FATALs config (no crash-restart loop).
        model.distill_coef = float(args.distill_coef or 0.0)
        model.distill_value_coef = float(args.distill_value_coef or 0.0)
        model.distill_value_feat_coef = float(args.distill_value_feat_coef or 0.0)  # gen3_exploiter_value_feat_distill_v1
        model._distill_teachers = []   # gen3_exploiter_distill_v1: N frozen per-team teachers (teacher-id = index+1)
        if args.distill_coef and args.distill_coef > 0 and getattr(args, "_distill_pairs", None):
            from agents.model.snapshot import (
                current_model_version as _cmv_d, load_foreign_opponent as _lfo_d)
            from agents.training.fixed_opponent_pool import _resolve_zip_and_config as _rzc_d
            _cv_d = _cmv_d(mappings, **_run_arch_toggles(args))
            for _tp, _tf in args._distill_pairs:
                try:
                    _zip_d, _cfg_d, _ = _rzc_d(_tp, None)   # run-dir → (zip, config)
                    _tm_d, _ = _lfo_d(_zip_d, current_version=_cv_d,
                                      device=str(model.device), config_path=_cfg_d)
                    _tm_d.policy.set_training_mode(False)
                    model._distill_teachers.append(_tm_d)
                except Exception as _e_d:  # noqa: BLE001 — bad/incompatible teacher weights
                    print(f"\n[Distill] FATAL: could not load --distill-teacher {_tp}: {_e_d}")
                    sys.stdout.flush()
                    os._exit(int(TrainExitCode.FATAL_CONFIG))
            emit(f"🧪 [DISTILL] {len(model._distill_teachers)} teacher(s) attached on {model.device} "
                 f"(order = teacher-id 1..{len(model._distill_teachers)})")
        model.opd_beta = args.opd_beta
        model._opd_on = bool(args.opd_coef and args.opd_coef > 0)
        if args.search_teacher:
            from agents.training.teacher.buffer import CorrectionBuffer
            model._correction_buffer = CorrectionBuffer(args.search_teacher_buffer_size)
        version = ModelVersion.from_layout_and_policy_kwargs(
            extractor_kwargs["layout"], policy_kwargs, vf_coef=args.vf_coef,
            reward_config=reward_config, value_tail_weight=args.value_tail_weight,
            opp_belief_aux_coef=args.opp_belief_aux_coef,
            move_belief_coef=args.move_belief_coef,
            win_prob_coef=args.win_prob_coef,
            pubval_coef=args.pubval_coef,
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
        _maybe_compile_trainer(model, args)
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
            # FRESH run: `num_timesteps` is 0, so SB3's `total_timesteps += num_timesteps` is a
            # no-op and the absolute target is correct here. The RESUME site must pass the REMAINING
            # budget instead — see the note there.
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
