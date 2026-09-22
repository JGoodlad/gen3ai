"""`--fork-lr` / `--fork-lr-freeze` — pin a FORK's step size, and ONLY a fork's.

THE PROBLEM. On a resume `--lr` is INERT: `model_build`'s resume path restores the checkpoint's
optimizer LR and prints `(arg --lr=… ignored on resume)`, because the KL controller is supposed to
keep whatever rate it had settled on across a launcher restart. That is right for a restart and
wrong for a FORK — a distillation fold launched off a parent checkpoint silently inherits the
PARENT's annealed rate, so three folds launched with identical flags ran at a median 5.8e-5,
2.8e-5 and 1.0e-4. The controller's inherited state was a hidden confound in every fold comparison.

THE DISCRIMINATION RULE, and why it is the one to key off. The launcher re-invokes the SAME argv
every `--restart-interval-hours`, resuming the SAME run dir — so a flag that fires "on resume"
fires on every restart, and a `--fork-lr` applied that way would reset the LR every few hours
forever. What separates the two cases is WHERE the resumed checkpoint lives:

    a FORK      →  `--model` names a checkpoint OUTSIDE the target run dir
    a RESTART   →  `--model` names a checkpoint THIS run wrote (`<run>/checkpoints/*.zip`, or a
                   `<run>/*.zip` for the legacy root-checkpoint layout)

That is not a new invention — it is the same predicate `run_io._resolve_fresh_model_dir` already
uses for its clobber guard (`resuming_into_it`), and the same one the LAUNCHER uses in
`launcher/checkpoint.py::resolve_fork_resume_model` to decide whether a restart should re-init from
the source or continue in place. The launcher additionally SWAPS `--model` to the fork's own
checkpoint once the fork has made progress, so on restart #2 of a fork the rule reads RESTART for
the same reason it reads RESTART for a plain resume.

`<run>/warmstart/…` is deliberately NOT a same-run checkpoint even though it is inside the run dir:
the consensus warm-start is an INIT built from foreign teachers, which is a fork by every meaning
that matters here.

THE FREEZE IS DIFFERENT. `--fork-lr` is a one-time event (pin the rate at the moment of forking,
then let the controller work). `--fork-lr-freeze` is a PROPERTY OF THE RUN — a fold that wants a
constant, recordable dose wants it constant for the whole fold — so it DOES survive every restart,
re-read from the pin recorded in `metadata.json`'s `dose` block (or from the argv, which a launcher
restart reproduces verbatim).
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, Optional


@dataclasses.dataclass(frozen=True)
class ForkLrDecision:
    """What to do about `--fork-lr` on this process. `reason` is printed, always."""

    apply: bool
    lr: Optional[float]
    frozen: bool
    reason: str


def is_same_run_checkpoint(model_path: str, model_dir: str) -> bool:
    """Is `model_path` a checkpoint THIS run produced — i.e. a periodic/crash RESTART?

    True for `<run>/checkpoints/<any>.zip` and for a bare `<run>/<name>.zip` (the legacy
    root-checkpoint layout, plus `final_model*.zip`). False for anything outside the run dir, and
    false for a nested dir that is not `checkpoints/` — notably `<run>/warmstart/`, which holds a
    freshly-built INIT rather than this run's own training progress.
    """
    if not model_path or not model_dir:
        return False
    mp = os.path.abspath(model_path)
    md = os.path.abspath(model_dir)
    if not mp.startswith(md + os.sep):
        return False
    rel = os.path.relpath(mp, md)
    parts = rel.split(os.sep)
    return len(parts) == 1 or parts[0] == "checkpoints"


def read_recorded_pin(model_dir: str) -> Optional[Dict[str, Any]]:
    """The `dose.fork_lr_pin` block from `<model_dir>/metadata.json`, or None.

    Best-effort by design: a missing/unreadable/older metadata.json means "no pin on record", which
    is the same answer as a run that never had one. This is how the FREEZE survives a restart.
    """
    path = os.path.join(model_dir, "metadata.json")
    try:
        with open(path) as f:
            meta = json.load(f)
    except Exception:  # noqa: BLE001 — absent, truncated, or not ours
        return None
    dose = meta.get("dose")
    pin = dose.get("fork_lr_pin") if isinstance(dose, dict) else None
    return dict(pin) if isinstance(pin, dict) else None


def resolve_fork_lr(*, fork_lr: Optional[float], fork_lr_freeze: bool,
                    model_path: str, model_dir: str) -> ForkLrDecision:
    """Decide whether to pin this process's LR, and to what. Pure — unit-tested directly.

    The four outcomes:
      * FORK + `--fork-lr`        → apply the pin (and the freeze, if asked).
      * FORK, no `--fork-lr`      → nothing (today's behaviour exactly).
      * RESTART, run is FROZEN    → re-apply the recorded pin and hold it; the freeze is a
                                    property of the RUN, so it must not evaporate on a restart.
      * RESTART, not frozen       → nothing. This is the whole point of the fork/restart split:
                                    re-pinning here would reset the controller every few hours.
    """
    same_run = is_same_run_checkpoint(model_path, model_dir)
    if same_run:
        recorded = read_recorded_pin(model_dir)
        if recorded and recorded.get("frozen") and recorded.get("lr") is not None:
            return ForkLrDecision(
                True, float(recorded["lr"]), True,
                f"same-run restart — the run's FROZEN pin persists (recorded "
                f"{float(recorded['lr']):.2e})")
        if fork_lr is not None and fork_lr_freeze:
            return ForkLrDecision(
                True, float(fork_lr), True,
                "same-run restart — --fork-lr-freeze is a property of the run and persists")
        if fork_lr is not None:
            return ForkLrDecision(
                False, None, False,
                f"same-run restart — --fork-lr {float(fork_lr):.2e} NOT re-applied (a periodic "
                f"restart keeps the controller's adapted rate; pass --fork-lr-freeze to pin it "
                f"for the whole run)")
        return ForkLrDecision(False, None, False, "same-run restart, no --fork-lr")
    if fork_lr is None:
        return ForkLrDecision(False, None, False, "fork, but no --fork-lr (LR inherited from the "
                                                  "parent checkpoint, as always)")
    return ForkLrDecision(
        True, float(fork_lr), bool(fork_lr_freeze),
        f"FORK from a checkpoint outside {model_dir} — pinning LR to {float(fork_lr):.2e}"
        + (" and FREEZING the KL controller" if fork_lr_freeze else ""))


def clamp_pin(lr: float, *, min_lr: float, max_lr: float) -> float:
    """The pin still respects `[--min-lr, --max-lr]` — a bound the user set is a bound."""
    return max(float(min_lr), min(float(lr), float(max_lr)))


def build_pin_record(decision: ForkLrDecision, *, applied_lr: float, source_model: str,
                     num_timesteps: int) -> Dict[str, Any]:
    """The `dose.fork_lr_pin` block. Written every save; read back by `read_recorded_pin`."""
    return {
        "lr": float(applied_lr),
        "frozen": bool(decision.frozen),
        "applied_at_step": int(num_timesteps),
        "source_model": str(source_model),
        "reason": decision.reason,
    }


def apply_fork_lr_pin(model, decision: ForkLrDecision, *, lr_callback, min_lr: float,
                      max_lr: float, source_model: str) -> Optional[Dict[str, Any]]:
    """Install the pin on a loaded model. Returns the record to stash, or None when not applying.

    THREE places have to agree or the pin is a no-op somewhere:
      * the OPTIMIZER's `param_groups[0]["lr"]` — what the next step actually uses, and what the
        checkpoint sidecar records as this run's LR;
      * `model.lr_schedule` — what SB3's `_update_learning_rate` re-installs at the top of every
        `train()`, so setting only the optimizer would be overwritten on the first update;
      * the KL controller's `_current_lr` — its multiplicative ladder starts from wherever it
        thinks it is, so seeding it from the checkpoint's rate would walk straight back there.
    """
    if not decision.apply or decision.lr is None:
        return None
    lr = clamp_pin(decision.lr, min_lr=min_lr, max_lr=max_lr)
    for group in model.policy.optimizer.param_groups:
        group["lr"] = lr
    model.lr_schedule = lambda _: lr
    if lr_callback is not None:
        lr_callback._current_lr = lr
        if decision.frozen and hasattr(lr_callback, "freeze_at"):
            lr_callback.freeze_at(lr)
    return build_pin_record(decision, applied_lr=lr, source_model=source_model,
                            num_timesteps=int(getattr(model, "num_timesteps", 0) or 0))


# --------------------------------------------------------------------------------------------
# gen3_fork_lr_inherit_guard_v1 — a FORK of a FROZEN parent must NAME its own dose
# --------------------------------------------------------------------------------------------
#: Every spelling the parent's recorded command may use for the two flags this guard reads.
_FREEZE_FLAGS = ("--fork-lr-freeze",)
_FORK_LR_FLAGS = ("--fork-lr",)


@dataclasses.dataclass(frozen=True)
class FrozenParent:
    """A fork parent whose LR was PINNED and FROZEN — the thing this guard refuses to inherit from.

    `lr` is the parent's frozen value (None only when the parent froze without a recoverable
    number, which the message says out loud rather than inventing one). `source` names WHICH
    recorded field answered, because "recorded in the command" and "recorded in the dose block"
    are different warrants and a reader is entitled to know which one fired.
    """

    run_dir: str
    run_name: str
    lr: Optional[float]
    source: str
    role: Optional[str]


@dataclasses.dataclass(frozen=True)
class InheritVerdict:
    """The guard's answer. `refuse` is the only status the LAUNCH path acts on; `line` is printed
    by BOTH surfaces, so a silent pass is never mute about whether the parent was read at all."""

    status: str
    line: str
    refuse: bool = False
    parent: Optional[FrozenParent] = None


def _as_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def frozen_parent_pin(parent_run_dir: Optional[str]) -> Optional[FrozenParent]:
    """Was the run at `parent_run_dir` trained at a PINNED, FROZEN LR? — read, never re-derived.

    🚨 **The optimisation block is NOT in `model_config.json`.** That file's 144 keys carry no
    `fork_lr`, `learning_rate`, `batch_size`, `n_epochs` or `grad_accum_steps` — it is a
    weight-SHAPE record — so the obvious shape of this guard (overlay the argv on the parent's
    config and read the resolved value) cannot work. The answer lives only in `metadata.json`, in
    two recorded places, and this function reads both in preference order:

      1. the IMMUTABLE ``original_command`` — ``--fork-lr-freeze`` present ⇒ frozen, with the value
         from its ``--fork-lr``. This is the operator's stated intent, written once at fork creation
         and preserved verbatim across every restart, and it is the only source that works on a run
         predating the ``dose`` block;
      2. the ``dose`` block's ``lr_frozen`` / ``fork_lr`` — what the run ACTUALLY ran at, rewritten
         every save. Used only to confirm a freeze the command did not state (a freeze re-applied
         from a recorded pin on a restart), and only when ``lineage`` also says the run was a FORK
         rather than a fresh one.

    Returns None for a run that was not frozen, for a run whose metadata cannot be read, and for a
    parent that is not on this box — **an unreadable parent is "unknown", never "frozen"**. The
    caller reports that case rather than refusing on it: a guard that cannot see the evidence must
    not pretend it did.
    """
    if not parent_run_dir or not os.path.isdir(parent_run_dir):
        return None
    from agents.training import lineage

    name = os.path.basename(os.path.normpath(parent_run_dir))
    cmd = lineage.read_original_command(parent_run_dir)
    role = lineage.role_of(parent_run_dir, warn=False)
    meta_path = os.path.join(parent_run_dir, "metadata.json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:  # noqa: BLE001 — absent, truncated, or not ours
        meta = {}
    dose = meta.get("dose") if isinstance(meta.get("dose"), dict) else {}

    if lineage.command_has_flag(cmd, *_FREEZE_FLAGS):
        lr = _as_float(lineage.command_flag_value(cmd, *_FORK_LR_FLAGS))
        if lr is None:
            lr = _as_float(dose.get("fork_lr")) or _as_float(dose.get("lr_now"))
        return FrozenParent(parent_run_dir, name, lr, "original_command", role)

    # The command did not say so. A recorded FROZEN dose still counts, but only on a run that
    # `lineage` calls a fork — a fresh run cannot have inherited a pin, and treating one as frozen
    # would refuse every ordinary fork of every ordinary run.
    if dose.get("lr_frozen") is True and role not in (None, "fresh"):
        lr = _as_float(dose.get("fork_lr")) or _as_float(dose.get("lr_now"))
        return FrozenParent(parent_run_dir, name, lr, "dose_block", role)
    return None


def check_inherited_fork_lr(*, model_path: Optional[str], model_dir: Optional[str],
                            fork_lr: Optional[float], fork_lr_freeze: bool,
                            allow: bool = False) -> InheritVerdict:
    """🚨 **A FORK INHERITS THE PARENT'S LR BUT NOT ITS FREEZE.**

    `--fork-lr-freeze` pins a rate and holds the KL controller at it for the whole run. Fork that
    run and name no `--fork-lr` of your own and SB3 restores the parent's optimizer state — the
    frozen NUMBER arrives, the freeze does not, and a live controller starts annealing away from a
    value that was chosen precisely because it should not move. The run's dose is then neither the
    parent's nor one anybody selected, and it is **not stationary within the run**, so `main.dose`
    can only report its median.

    That is not hypothetical. The three era-2 exploiters (2026-09-20, `7afa2b34`) forked the
    plateau parent's frozen `2.80e-05` without naming `--fork-lr`, their controllers annealed it to
    `8.36e-05`, and all three ran at a median `5.5e-05` — a dose of `8.392e-9`, **0.39×** the v8
    reference, against the era-1 exploiters' **1.78×**. A 4.5× gap in the campaign's own step size,
    discovered after the GPU time was spent, on argvs that three gates had passed.

    Pure apart from reading the parent's `metadata.json`. Returns a verdict for BOTH surfaces:
    `main.checkargs` prints `line` and refuses nothing; the launch path turns `refuse` into a
    startup `FATAL_CONFIG`.
    """
    if not model_path:
        return InheritVerdict("fresh", "[ForkLR] no --model: a fresh run has no parent LR to inherit")
    if model_dir and is_same_run_checkpoint(model_path, model_dir):
        return InheritVerdict(
            "restart", "[ForkLR] same-run RESTART — the LR is this run's own, nothing is inherited")
    if fork_lr is not None or fork_lr_freeze:
        named = " ".join(
            ([f"--fork-lr {float(fork_lr):.4g}"] if fork_lr is not None else [])
            + (["--fork-lr-freeze"] if fork_lr_freeze else []))
        return InheritVerdict("named", f"[ForkLR] ✓ this fork names its own dose ({named})")

    from agents.training import lineage

    parent_dir = lineage.run_dir_of(model_path)
    parent = frozen_parent_pin(parent_dir)
    if parent is None:
        if not parent_dir or not os.path.isdir(parent_dir):
            return InheritVerdict(
                "parent_unreadable",
                f"[ForkLR] ⚠️ the fork parent's run dir is not on this box ({parent_dir or model_path}) "
                f"— whether its LR was FROZEN is UNKNOWN, so this check could not run. If it was, "
                f"name --fork-lr explicitly.")
        return InheritVerdict(
            "parent_unfrozen",
            f"[ForkLR] ✓ the fork parent ({os.path.basename(os.path.normpath(parent_dir))}) did not "
            f"run at a frozen LR — the inherited rate is an ordinary annealed one")

    shown = f"{parent.lr:.2e}" if parent.lr is not None else "an unrecorded value"
    fix = (f"--fork-lr {parent.lr:.6g} --fork-lr-freeze" if parent.lr is not None
           else "--fork-lr <the rate you mean> --fork-lr-freeze")
    msg = (
        f"this launch FORKS `{parent.run_name}`, whose LR was PINNED and FROZEN at {shown} "
        f"(recorded in its metadata.json `{parent.source}`), and this argv names NEITHER "
        f"--fork-lr NOR --fork-lr-freeze. A fork inherits the parent's optimizer LR but NOT its "
        f"freeze, so the KL controller will start live at {shown} and anneal away from it: the "
        f"run's dose is then neither the parent's nor one you chose, and it is not stationary "
        f"within the run. That is exactly what the era-2 exploiters did on 2026-09-20 — 2.80e-05 "
        f"→ 8.36e-05, median 5.5e-05, 0.39× the v8 reference against era-1's 1.78×, a 4.5× gap "
        f"nobody registered. THE FIX: name the dose you mean — `{fix}` reproduces the parent's — "
        f"or pass --allow-inherited-fork-lr to state deliberately that this run inherits an "
        f"UNFROZEN {shown} and will say so wherever its dose is reported.")
    if allow:
        return InheritVerdict(
            "override",
            f"[ForkLR] ⚠️ --allow-inherited-fork-lr: inheriting {parent.run_name}'s FROZEN "
            f"{shown} WITHOUT its freeze. The KL controller runs live from there — read the run's "
            f"realized dose with `python -m main.dose` and state it wherever the number is "
            f"reported.", parent=parent)
    return InheritVerdict("REFUSE", f"[ForkLR] FATAL: {msg}", refuse=True, parent=parent)


def enforce_inherited_fork_lr(args: Any, model_dir: Optional[str]) -> InheritVerdict:
    """The LAUNCH-path wrapper: print the verdict, and `os._exit(FATAL_CONFIG)` on a refusal.

    Called from `train_rl_agent.main` the moment `model_dir` is known and BEFORE it is created, so
    a refusal leaves no run directory behind — the same property the `[Untaught]` guard has.
    """
    verdict = check_inherited_fork_lr(
        model_path=getattr(args, "model", None), model_dir=model_dir,
        fork_lr=getattr(args, "fork_lr", None),
        fork_lr_freeze=bool(getattr(args, "fork_lr_freeze", False)),
        allow=bool(getattr(args, "allow_inherited_fork_lr", False)))
    if verdict.refuse:
        import sys
        from main.exit_codes import TrainExitCode
        print(f"\n{verdict.line}")
        sys.stdout.flush()
        os._exit(int(TrainExitCode.FATAL_CONFIG))
    if verdict.status in ("override", "parent_unreadable"):
        print(verdict.line)
    return verdict
