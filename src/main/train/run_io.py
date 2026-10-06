"""The run DIRECTORY and its bookkeeping: where a run writes, and what it records as it goes.

`_resolve_fresh_model_dir` picks the directory; `_write_latest_txt` / `_attach_run_tb_logger` /
`_model_hparams` / `_TrackingCheckpointCallback` are what keep it current while training runs.
`_run_arch_toggles` is here because it, too, is provenance — the toggle set a snapshot is gated on.
"""
import os
import sys
from datetime import datetime
from typing import Optional

from agents.training.loop_callbacks import BaseCallback

from agents.model.snapshot import record_checkpoint
from agents.training.dose import dose_block
from agents.training.lineage import build_lineage
from main.train.constants import checkpoint_due
from utils.torch_state_guard import INIT_NUM_THREADS_ATTR


def _resolve_fresh_model_dir(run_name, exploiter_label, model_arg):
    """Pick the run directory for a run whose --run-dir is NOT set (i.e. not a launcher-managed
    resume). Precedence: an explicit --run-name → ``<archive>/<name>``; else, in exploiter mode, a
    derived ``<archive>/<era>_exploiter_vs_<target>``; else a date-stamped
    ``<archive>/<era>_run_<timestamp>`` (``<era>_`` = ``utils.era.CURRENT_ERA``'s two-letter code, e.g.
    ``rb_``; an explicit name is accepted as typed, with a warning when it lacks the era prefix). ``<archive>`` is ``utils.paths.run_archive_dir()`` — ``$GEN3AI_MODELS_DIR`` or the
    MAIN checkout's ``models/`` — NEVER a cwd-relative ``models/``: from a worktree that directory is
    deleted silently with the worktree (2026-09-23, eight runs). No archive → FATAL_CONFIG. A NAMED dir is
    validated as a single safe path component, and we refuse to start a FRESH run on top of an EXISTING
    run (one carrying a metadata.json) — unless --model resumes from INSIDE that very dir — so naming a
    run after e.g. the live run can't silently clobber it. Returns the (absolute) dir, or exits with a
    clear FATAL. Pure given its args and the archive → unit-tested."""
    import re
    from main.exit_codes import TrainExitCode
    from utils.era import prefixed, run_name_warning
    from utils.paths import RunArchiveError, new_run_dir
    if run_name:
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", run_name):
            print(f"\n[RunName] FATAL: --run-name {run_name!r} must be a single name "
                  f"(letters/digits/._-), with no slashes or path traversal.")
            sys.exit(1)
        leaf = run_name   # accepted AS TYPED (utils.era.run_name_warning says why)
        warning = run_name_warning(run_name)
        if warning:
            print(warning)
    elif exploiter_label:
        leaf = prefixed("exploiter_vs_" + exploiter_label.removeprefix("ext_"))
    else:
        # always unique → no clobber guard; the era prefix is the DEFAULT name's (utils.era)
        leaf = prefixed(f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    try:
        model_dir = new_run_dir(leaf)
    except RunArchiveError as exc:
        print(f"\n[RunArchive] FATAL: {exc}", file=sys.stderr)
        sys.exit(int(TrainExitCode.FATAL_CONFIG))
    if not (run_name or exploiter_label):
        return model_dir
    # Clobber guard: a named fresh run must not write into a DIFFERENT existing run's dir.
    resuming_into_it = bool(model_arg) and os.path.abspath(model_arg).startswith(
        os.path.abspath(model_dir) + os.sep)
    if os.path.exists(os.path.join(model_dir, "metadata.json")) and not resuming_into_it:
        print(f"\n[RunName] FATAL: {model_dir!r} is already a run (it has a metadata.json). Pick a "
              f"different --run-name, or pass --model <a checkpoint inside it> to resume that run.")
        sys.exit(1)
    return model_dir


def _resolve_model_dir(run_dir, run_name, exploiter_label, model_arg):
    """THE trainer's run directory — the ONE decision, so a test can drive it.

    ``--run-dir`` (the launcher's resume / fork dir, or one typed by hand) is used as given but
    CHECKED: a dir inside a linked worktree's OWN ``models/`` dies silently with the worktree
    (2026-09-23, eight runs), so it exits ``FATAL_CONFIG`` here — the first moment it is known, before
    anything exists. Otherwise ``_resolve_fresh_model_dir`` (``--run-name`` / exploiter / minted, all
    in ``utils.paths.run_archive_dir()``, the main checkout's ``models/``). Always absolute."""
    if not run_dir:
        return _resolve_fresh_model_dir(run_name, exploiter_label, model_arg)
    from main.exit_codes import TrainExitCode
    from utils.paths import RunArchiveError, checked_run_dir
    try:
        return checked_run_dir(run_dir)
    except RunArchiveError as exc:
        print(f"\n[RunArchive] FATAL: {exc}", file=sys.stderr)
        sys.exit(int(TrainExitCode.FATAL_CONFIG))


def _run_lineage(args, model_dir: str, *, model_path, fork_step) -> "dict | None":
    """THE LINEAGE SEAM — the immutable `lineage` block for THIS process, or None on a restart.

    All of the work lives in `agents.training.lineage`; this is the one line that knows which
    argparse fields carry the fork's parent and exploiter target. `None` means "a same-run restart
    contributes nothing", and `save_model_snapshot` preserves whatever the run already recorded
    (the same existing-value-wins rule `original_command` uses).

    `model_path` must be the `--model` as PARSED (the fork parent), never a path a later step
    re-pointed `args.model` at — recording a derived init would make the run its own ancestor.

    `pool_seeded_from` rides IN this block rather than beside it: the block is written once at fork
    creation and frozen thereafter (`save_model_snapshot`: the existing value always wins), and the
    pool is seeded earlier in the same process, so the fact is available exactly when the block is
    built. It is a SIBLING key inside the block, never an edit to `fork_parent` — the immutability
    mechanism forbids rewriting a recorded block, so a later restart cannot add or change it.
    """
    block = build_lineage(model_path=model_path, model_dir=model_dir,
                          exploiter=getattr(args, "exploiter", None),
                          fork_step=fork_step)
    if block is not None:
        from agents.training.pool_seed import pool_dir_for, read_seed_record
        record = read_seed_record(pool_dir_for(model_dir))
        if record is not None:
            block["pool_seeded_from"] = record
    return block


def _run_arch_toggles(args) -> dict:
    """The architecture TOGGLES of THIS run, for current_model_version so the version gate compares
    like-for-like against the run's own (toggle-ON) pool/stable-opponent snapshots. Without these, a
    belief-ON / attend-unrevealed run would FATAL on every snapshot it is meant to protect.

    Sourced from `agents.model.flag_registry` via `arch_toggles_from_args`, NOT hand-listed: this
    dict and `build_extractor_arch_kwargs` used to be two independently maintained lists of the same
    toggles, and a toggle added to one and not the other means the gate compares an architecture the
    run does not build. `critic` is appended by hand because it is a policy_kwarg rather than an
    extractor kwarg, so it is out of the registry's scope."""
    from agents.model.extractor_arch import arch_toggles_from_args
    return {**arch_toggles_from_args(args), "critic": args.critic}


def _matmul_precision() -> str:
    import torch
    return str(torch.get_float32_matmul_precision())


def _model_hparams(model) -> dict:
    """The per-checkpoint hparam block."""
    clip_range_vf = float(model.clip_range_vf(1.0)) if model.clip_range_vf is not None else -1.0
    opt = model.policy.optimizer
    out = {
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "ent_coef": float(model.ent_coef),
        "vf_coef": float(model.vf_coef),
        "opp_belief_aux_coef": float(getattr(model, "opp_belief_aux_coef", 0.0)),
        "move_belief_coef": float(getattr(model, "move_belief_coef", 0.0)),
        "move_belief_latent_coef": float(getattr(model, "move_belief_latent_coef", 0.0)),
        "spread_belief_coef": float(getattr(model, "spread_belief_coef", 0.0)),
        "hp_type_belief_coef": float(getattr(model, "hp_type_belief_coef", 0.0)),
        "item_belief_coef": float(getattr(model, "item_belief_coef", 0.0)),
        "policy_grad_coef": float(getattr(model, "policy_grad_coef", 1.0)),
        "intent_label_bot_weight": float(getattr(model, "intent_label_bot_weight", 1.0)),
        "fork_fraction": float(getattr(model, "fork_fraction", 0.0)),
        "fork_branches": int(getattr(model, "fork_branches", 3)),
        "fork_contested_gap": float(getattr(model, "fork_contested_gap", 0.4)),
        "fork_contested_absv": float(getattr(model, "fork_contested_absv", 0.0)),
        "fork_max_per_battle": int(getattr(model, "fork_max_per_battle", 1)),
        "fork_crn": str(getattr(model, "fork_crn", 'dice_and_draws')),
        "batch_size": model.batch_size,
        "grad_accum_steps": int(getattr(model, "grad_accum_steps", 1)),
        "n_steps": model.n_steps,
        "clip_range": float(model.clip_range(1.0)),
        "clip_range_vf": clip_range_vf,
        "optimizer": type(opt).__name__,
        "weight_decay": opt.param_groups[0].get("weight_decay", 0.0),
        # gen3_matmul_precision_v1: what THIS (the trainer) process's fp32 matmuls actually ran at —
        # read from torch rather than the argv, so the record is the realized value. Always `highest`
        # now (TF32 was retired, deletion pass K2); kept as provenance, and it is what a resume reads to
        # refuse a run that recorded `high` (`model_version.retired_levers`).
        "matmul_precision": _matmul_precision(),
        # gen3_fork_lr_pin_v1 — THE DOSE. `lr x n_epochs / (batch_size*grad_accum_steps)`, plus the
        # provenance a reader needs to know whether that LR was chosen or inherited. Nested rather
        # than flattened so `python -m main.dose` reads one key and cannot collide with an hparam
        # name. metadata.json ONLY — never model_config.json, which is the weight-shape record
        # `check_compatible` reads.
        "dose": dose_block(model),
    }
    # The torch thread count the run's FRESH build ran at (`construct_fresh_learner`, X5 A/B §7.4
    # precondition). Carried on the model (it rides every checkpoint), ABSENT for a checkpoint that predates
    # the record — UNKNOWN, never a guessed 1. `save_model_snapshot` makes the first recorded value immutable.
    _bt = getattr(model, INIT_NUM_THREADS_ATTR, None)
    if isinstance(_bt, int) and not isinstance(_bt, bool):
        out[INIT_NUM_THREADS_ATTR] = _bt
    # The step this snapshot was taken at. Recorded because `pin_history` (metadata.json's
    # append-only "which commit ran which steps") needs a step for its span boundaries, and
    # every production save already routes through here. `getattr` because the test doubles
    # that call this are plain hparam holders, not SB3 models.
    _steps = getattr(model, "num_timesteps", None)
    if _steps is not None:
        out["num_timesteps"] = int(_steps)
    # M5 Lane G — WHICH ENV CORE this process's rollouts ran on (the env core): the Rust core's stamp
    # (front end, build stamp, trigger, T2 backend + buckets, the keyed draw), or the bare `rust` for a save
    # with no collector (a fresh-checkpoint tool). A python-era record (`python`) is read by D4. Written on
    # every save like `matmul_precision`, so each checkpoint's sidecar names the core that produced it.
    _ec = getattr(model, "_env_core_stamp", None)
    out["env_core"] = ({k: v for k, v in _ec.items() if k != "summary"} if isinstance(_ec, dict)
                       else {"env_core": "rust"})
    # gen3_oracle_reveal_v1 (v137) — the OBSERVATION MODE this process's rows were built at (a diagnostic: `off` is
    # production). model_config.json carries it as the version-gated field; this is the per-checkpoint provenance
    # a reader of metadata.json sees beside `env_core`. The extractor holds the mode the policy was built under.
    out["oracle_reveal"] = str(getattr(getattr(getattr(model, "policy", None), "features_extractor", None),
                                       "oracle_reveal", "off") or "off")
    return out


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
    """Route the learner's logger (`agents.training.train_logger`) to ``<model_dir>/tb/`` (stdout +
    tensorboard).

    ``set_logger`` marks it custom, so ``learn``'s setup keeps it (the owned loop serves no
    ``tensorboard_log`` run-id directories), landing the run's TensorBoard data inside its own model dir — co-located with the
    checkpoints (NOT a separate top-level ``tensorboard/`` tree). The path is
    cwd-relative via ``model_dir``, the same basis the checkpoints use, so it lands in
    the main repo even under the launcher's worktree pin; and promoting a run to a
    golden (``mv models/run_X models/_goldens/<name>``) carries its curves along. Point
    ``tensorboard --logdir models`` to see every run + golden, each named by its dir.
    """
    from agents.training.train_logger import configure
    tb_dir = os.path.join(model_dir, "tb")
    fmts = ["stdout", "tensorboard"] if model.verbose >= 1 else ["tensorboard"]
    model.set_logger(configure(tb_dir, fmts))
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


class DoseLogCallback(BaseCallback):
    """Publish the DOSE to TensorBoard every rollout — `train/dose_rate` + `train/effective_batch`.

    `train/learning_rate` alone cannot be compared across runs: the same LR at
    `batch_size 2048 x grad_accum 16` and at `2048 x 2` differ 8x in optimizer steps per env step,
    and that product is what predicted a distillation fold's collateral (ledger M7). Recording the
    product LIVE means a run's dose curve exists even when its checkpoint sidecars are groomed away.

    `effective_batch` is emitted beside it because the rate alone is ambiguous — a falling
    `dose_rate` is a KL controller annealing or an operator having raised `--grad-accum-steps` on a
    restart, and only the second moves this line.
    """

    def _on_rollout_end(self) -> None:
        block = dose_block(self.model)
        rate = block.get("dose_rate_now")
        if rate is not None:
            self.logger.record("train/dose_rate", float(rate))
        if block.get("effective_batch"):
            self.logger.record("train/effective_batch", int(block["effective_batch"]))

    def _on_step(self) -> bool:
        return True


class _TrackingCheckpointCallback(BaseCallback):
    """The periodic checkpointer: saves at TOTAL-ENV-STEP boundaries, keeps latest.txt up to date
    and writes per-checkpoint metadata.

    🚨 IT DOES NOT COUNT CALLS (sb3's `CheckpointCallback`, its base until deletion pass U4, saved on
    `n_calls % save_freq`). A callback call was N env steps on the deleted sync Python core, a WAVE
    under the deleted async collector, and is N trainee decisions on the Rust collector — a different
    interval on each. The save lands at the first call whose `num_timesteps` reaches the next multiple
    of `interval_env_steps` (`constants.checkpoint_due`), the rule the eval callbacks use.

    ON A RESTART the boundaries are GLOBAL multiples of the interval: the anchor is the step the
    process resumed at (`_on_training_start`), so the next save is the next multiple above it, not
    `resume step + interval` as SB3's per-process call counter gave.
    """

    def __init__(self, *, interval_env_steps: int, save_path: str, name_prefix: str = "checkpoint",
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.interval_env_steps = max(1, int(interval_env_steps))
        # The `num_timesteps` the last boundary test saw; set at `learn()` start (a resume's step).
        self._last_step: Optional[int] = None
        self._current_lr_fn = None
        self._current_epochs_fn = None
        # Optional: returns the current TwoPhaseLR handoff_lr (or None).
        self._handoff_lr_fn = None
        # The .zip goes into self.save_path, which we point at <run>/checkpoints/.
        # latest.txt + metadata.json are run-LEVEL, so derive the run root (the parent
        # of the checkpoints/ subdir; == save_path if it isn't one, e.g. legacy/tests).
        self._run_dir = (
            os.path.dirname(self.save_path)
            if os.path.basename(os.path.normpath(self.save_path)) == "checkpoints"
            else self.save_path
        )

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_training_start(self) -> None:
        self._last_step = int(self.model.num_timesteps)

    def _due(self, now: int) -> bool:
        """Advance the anchor on EVERY call, so one boundary saves exactly once. With no anchor (a
        caller that never ran `learn()`'s training start) the first call only sets it."""
        last = now if self._last_step is None else self._last_step
        self._last_step = now
        return checkpoint_due(last, now, self.interval_env_steps)

    def _on_step(self) -> bool:
        # The MODEL's counter, not the callback's `num_timesteps` mirror (that one is refreshed only
        # by the public `on_step`); the file is named by the same number the boundary was tested on.
        now = int(self.model.num_timesteps)
        if self._due(now):
            ckpt_path = os.path.join(self.save_path, f"{self.name_prefix}_{now}_steps.zip")
            self.model.save(ckpt_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {ckpt_path}")
            # The .zip is in self.save_path (<run>/checkpoints/). latest.txt
            # records the run-RELATIVE path (checkpoints/checkpoint_<N>_steps.zip) and the
            # per-checkpoint sidecar lands next to the .zip; metadata.json (snapshot_history)
            # stays at the run root (self._run_dir).
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
        return True
