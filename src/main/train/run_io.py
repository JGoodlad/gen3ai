"""The run DIRECTORY and its bookkeeping: where a run writes, and what it records as it goes.

`_resolve_fresh_model_dir` picks the directory; `_write_latest_txt` / `_attach_run_tb_logger` /
`_model_hparams` / `_TrackingCheckpointCallback` are what keep it current while training runs.
`_run_arch_toggles` is here because it, too, is provenance — the toggle set a snapshot is gated on.
"""
import os
import sys
from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from agents.model.snapshot import record_checkpoint
from agents.training.dose import dose_block
from agents.training.lineage import build_lineage


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


def _run_lineage(args, model_dir: str, *, model_path, fork_step) -> "dict | None":
    """THE LINEAGE SEAM — the immutable `lineage` block for THIS process, or None on a restart.

    All of the work lives in `agents.training.lineage`; this is the one line that knows which
    argparse fields carry the fork's parent, teachers and target. `None` means "a same-run restart
    contributes nothing", and `save_model_snapshot` preserves whatever the run already recorded
    (the same existing-value-wins rule `original_command` uses).

    `model_path` must be the PRE-WARM-START `--model`: the consensus warm-start re-points
    `args.model` at `<run>/warmstart/warmstart_consensus.zip`, which is an INIT built from the real
    parent, not the parent itself — recording it would make the run its own ancestor.
    """
    return build_lineage(model_path=model_path, model_dir=model_dir,
                         exploiter=getattr(args, "exploiter", None),
                         distill_teacher=getattr(args, "distill_teacher", None),
                         fork_step=fork_step)


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
        "item_belief_coef": float(getattr(model, "item_belief_coef", 0.0)),
        "td_aux_coef": float(getattr(model, "td_aux_coef", 0.0)),
        "policy_grad_coef": float(getattr(model, "policy_grad_coef", 1.0)),
        "intent_label_bot_weight": float(getattr(model, "intent_label_bot_weight", 1.0)),
        "win_prob_coef": float(getattr(model, "win_prob_coef", 1.0)),
        "value_dist_coef": float(getattr(model, "value_dist_coef", 1.0)),
        "search_teacher_coef": float(getattr(model, "search_teacher_coef", 0.0)),
        "search_teacher_value_coef": float(getattr(model, "search_teacher_value_coef", 0.0)),
        "search_teacher_beta": float(getattr(model, "search_teacher_beta", 1.0)),
        "search_teacher_batch_size": int(getattr(model, "search_teacher_batch_size", 256)),
        "opd_coef": float(getattr(model, "opd_coef", 0.0)),
        "cf_winprob_coef": float(getattr(model, "cf_winprob_coef", 0.0)),
        "cf_label_likelihood": str(getattr(model, "cf_label_likelihood", "binomial")),
        "cf_evidential_coef": float(getattr(model, "cf_evidential_coef", 0.0)),
        "cf_evidential_reg": float(getattr(model, "cf_evidential_reg", 0.0)),
        "cf_twin_coef": float(getattr(model, "cf_twin_coef", 0.0)),
        "cf_shadow_coef": float(getattr(model, "cf_shadow_coef", 0.0)),
        "q_winprob_coef": float(getattr(model, "q_winprob_coef", 0.0)),
        "q_winprob_onpolicy_coef": float(getattr(model, "q_winprob_onpolicy_coef", 0.0)),
        "distill_coef": float(getattr(model, "distill_coef", 0.0)),
        "distill_value_coef": float(getattr(model, "distill_value_coef", 0.0)),
        "distill_value_feat_coef": float(getattr(model, "distill_value_feat_coef", 0.0)),
        "distill_target": str(getattr(model, "distill_target", "kl")),
        "distill_topk": int(getattr(model, "distill_topk", 1)),
        "distill_gate": str(getattr(model, "distill_gate", "none")),
        "distill_gate_tau": float(getattr(model, "distill_gate_tau", 0.0)),
        "distill_beta": float(getattr(model, "distill_beta", 1.0)),
        "opd_beta": float(getattr(model, "opd_beta", 1.0)),
        "batch_size": model.batch_size,
        "grad_accum_steps": int(getattr(model, "grad_accum_steps", 1)),
        "n_steps": model.n_steps,
        "clip_range": float(model.clip_range(1.0)),
        "clip_range_vf": clip_range_vf,
        "optimizer": type(opt).__name__,
        "weight_decay": opt.param_groups[0].get("weight_decay", 0.0),
        # gen3_fork_lr_pin_v1 — THE DOSE. `lr x n_epochs / (batch_size*grad_accum_steps)`, plus the
        # provenance a reader needs to know whether that LR was chosen or inherited. Nested rather
        # than flattened so `python -m main.dose` reads one key and cannot collide with an hparam
        # name. metadata.json ONLY — never model_config.json, which is the weight-shape record
        # `check_compatible` reads.
        "dose": dose_block(model),
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
