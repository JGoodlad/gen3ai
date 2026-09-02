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
