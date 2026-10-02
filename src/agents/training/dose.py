"""THE DOSE — how hard a run is actually updating, as one recorded number.

WHY THIS EXISTS. `--lr` is INERT on a resume: the resume path restores the checkpoint's optimizer
LR (`main.train.model_build`, the `(arg --lr=… ignored on resume)` line), so a fork inherits
whatever rate the PARENT's KL controller had annealed to. Three distillation folds launched with
the same `--lr` therefore ran at three different rates — a median 5.8e-5, 2.8e-5 and 1.0e-4 — and
nothing in any of those runs said so. The controller's inherited state was a hidden confound in
every fold comparison the program has made.

The quantity that predicts a fold's collateral is not the LR alone but the DOSE:

    updates_per_env_step = n_epochs * optimizer_steps_per_epoch / rollout_rows
    dose_rate            = lr * updates_per_env_step

i.e. the learning rate times how many optimizer steps one env step buys. When the rollout divides
by the effective batch (`batch_size * grad_accum_steps`) this is `n_epochs / effective_batch`; when
it does not, the learner's short last group is still a FULL-weight step (K10(c), 2026-09-30), so the
step count is ceil(ceil(rows / batch_size) / grad_accum_steps), not rows / effective_batch. `grad_accum_steps`
belongs in the denominator because K micro-batches are summed into ONE optimizer step, so the
effective batch is `batch_size * K` and the step count falls by K. Two runs at the same `--lr`
differ by 6x in dose when one accumulates 16 micro-batches and the other 2 — which is exactly what
separated the v8 fold (2.15e-8) from every gen-era fold that followed it.

PURE and torch-free by design: `main.dose` reads the same functions over saved sidecar JSON on
runs whose checkpoints no longer load, and `main.train.run_io` calls `dose_block` on a live model.

⚠️ Nothing here may hold a LIVE object off the model. Everything `dose_block` reads is either a
primitive on the model or a plain dict stashed there — see `kl_controller_snapshot` for the save
that a live callback reference breaks.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def effective_batch(batch_size: int, grad_accum_steps: int = 1) -> int:
    """The batch one optimizer step actually sees: `batch_size * K`."""
    return int(batch_size) * max(1, int(grad_accum_steps or 1))


def optimizer_steps_per_epoch(*, rollout_rows: int, batch_size: int, grad_accum_steps: int = 1) -> int:
    """The optimizer steps ONE epoch over `rollout_rows` actually takes (K10(c), 2026-09-30).

    The learner cuts the rollout into ceil(rows / batch_size) micro-batches and steps once per
    `grad_accum_steps` of them, flushing a SHORT trailing group as one more step — which it rescales
    to its own micro count (`instrumented_ppo/ppo.py`, the partial-group flush), i.e. a FULL-weight
    step normalised by its real rows. So 98,304 rows at 2,048 x 32 take TWO steps an epoch, not the
    1.5 that `rows / effective_batch` says."""
    b = int(batch_size)
    if b <= 0 or int(rollout_rows) <= 0:
        raise ValueError(f"rollout rows and batch size must be positive (got {rollout_rows}, {batch_size})")
    k = max(1, int(grad_accum_steps or 1))
    micros = -(-int(rollout_rows) // b)
    return -(-micros // k)


def updates_per_env_step(*, batch_size: int, grad_accum_steps: int = 1,
                         n_epochs: int, rollout_rows: Optional[int] = None) -> float:
    """Optimizer steps per collected env step. With `rollout_rows` known: the steps the learner really
    takes, `n_epochs * optimizer_steps_per_epoch / rollout_rows` (K10(c)); without it, the old
    `n_epochs / effective_batch`, which is EXACT only when the rollout divides by the effective batch
    and UNDER-counts a ragged last step (3/4 of the truth at 98,304 / 65,536)."""
    eff = effective_batch(batch_size, grad_accum_steps)
    if eff <= 0:
        raise ValueError(f"effective batch must be positive (got batch_size={batch_size}, "
                         f"grad_accum_steps={grad_accum_steps})")
    if rollout_rows:
        steps = optimizer_steps_per_epoch(rollout_rows=rollout_rows, batch_size=batch_size,
                                          grad_accum_steps=grad_accum_steps)
        return float(n_epochs) * steps / float(rollout_rows)
    return float(n_epochs) / float(eff)


def dose_rate(*, lr: float, batch_size: int, grad_accum_steps: int = 1,
              n_epochs: int, rollout_rows: Optional[int] = None) -> float:
    """`lr * updates_per_env_step` — the number to compare two folds on."""
    return float(lr) * updates_per_env_step(
        batch_size=batch_size, grad_accum_steps=grad_accum_steps, n_epochs=n_epochs,
        rollout_rows=rollout_rows)


def kl_controller_snapshot(callback: Any) -> Optional[Dict[str, Any]]:
    """A PLAIN-DATA snapshot of the LR controller's configuration, or None when there isn't one.

    🚨 A snapshot rather than the callback itself, and that is not a style choice. `model.save()`
    cloudpickles the model's `__dict__`, and an LR callback holds a back-reference to the model and
    to SB3's `Logger` — which carries a `_contextvars.Context` and CANNOT be pickled. Stashing the
    live object on the model therefore breaks EVERY save in the run, at the pre-train round-trip
    smoke (observed, 2026-09-01). Same hazard the `_cf_buffer` exclusion documents; here the
    fix is not to hold the object at all, which needs no exclusion list to stay true.

    Read DUCK-TYPED rather than by class, so it covers `AdaptivePPOCallback`, `TwoPhaseLRCallback`
    and a test double alike. `anneal_start_steps` rides along because `phase` is a function of the
    STEP, which is only known at save time.
    """
    if callback is None:
        return None
    out: Dict[str, Any] = {"frozen": bool(getattr(callback, "frozen", False))}
    for name in ("target_kl", "kl_factor", "lr_factor", "min_lr", "max_lr"):
        val = getattr(callback, name, None)
        if val is not None:
            out[name] = float(val)
    anneal_start = getattr(callback, "_anneal_start_steps", None)
    if anneal_start is not None:
        out["anneal_start_steps"] = int(anneal_start)
    return out


def kl_controller_block(snapshot: Optional[Dict[str, Any]], *,
                        num_timesteps: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """`kl_controller_snapshot`'s output plus the `phase` that only the current step can decide.

    A FROZEN controller reports `frozen` because neither the KL ladder nor the cosine is in charge;
    a two-phase one reports which side of `anneal_start_steps` the run is on; anything else is the
    plain KL-adaptive controller.
    """
    if not snapshot:
        return None
    out = {k: v for k, v in snapshot.items() if k not in ("frozen", "anneal_start_steps")}
    anneal_start = snapshot.get("anneal_start_steps")
    if snapshot.get("frozen"):
        phase = "frozen"
    elif anneal_start is not None and num_timesteps is not None:
        phase = f"twophase_{1 if int(num_timesteps) < int(anneal_start) else 2}"
    else:
        phase = "adaptive"
    return {"phase": phase, **out}


def dose_block(model: Any) -> Dict[str, Any]:
    """The `dose` block recorded into metadata.json (and every checkpoint sidecar).

    Every field is read off the LIVE model, never off `args`, with two exceptions that argparse
    alone knows and `apply_training_hparams` stashes on the model: `_dose_lr_flag` (what `--lr`
    said, so a reader can SEE that it was inert) and `_fork_lr_pin` (the applied `--fork-lr`
    record). Missing attributes degrade to a `None` field rather than raising — this block is
    provenance, and a save must never fail because a diagnostic could not be computed.

    Provenance rule (root CLAUDE.md): this belongs in `metadata.json`, NEVER in
    `model_config.json`, which is the weight-shape record `check_compatible` reads.
    """
    lr_now: Optional[float]
    try:
        lr_now = float(model.policy.optimizer.param_groups[0]["lr"])
    except Exception:  # noqa: BLE001 — provenance, not a gate
        lr_now = None
    batch_size = int(getattr(model, "batch_size", 0) or 0)
    grad_accum = max(1, int(getattr(model, "grad_accum_steps", 1) or 1))
    n_epochs = int(getattr(model, "n_epochs", 0) or 0)
    eff = effective_batch(batch_size, grad_accum)
    # the rollout the learner is handed: n_steps x n_envs (the Rust collector sets n_steps to its
    # update target / n_envs, so this is its declared target too)
    ns, ne_ = getattr(model, "n_steps", None), getattr(model, "n_envs", None)
    rows = (int(ns) * int(ne_) if all(isinstance(x, int) and not isinstance(x, bool) for x in (ns, ne_))
            else 0)
    rows = rows if rows > 0 else 0
    ups = (updates_per_env_step(batch_size=batch_size, grad_accum_steps=grad_accum,
                                n_epochs=n_epochs, rollout_rows=rows or None) if eff > 0 else None)
    steps = (optimizer_steps_per_epoch(rollout_rows=rows, batch_size=batch_size, grad_accum_steps=grad_accum)
             if (rows and batch_size > 0) else None)
    pin = getattr(model, "_fork_lr_pin", None)
    kl_snapshot = getattr(model, "_dose_kl", None)
    frozen = bool((kl_snapshot or {}).get("frozen", False)) or bool((pin or {}).get("frozen", False))
    block: Dict[str, Any] = {
        "lr_now": lr_now,
        "lr_flag": (float(getattr(model, "_dose_lr_flag", 0.0))
                    if getattr(model, "_dose_lr_flag", None) is not None else None),
        "fork_lr": (float(pin["lr"]) if pin and pin.get("lr") is not None else None),
        "lr_frozen": frozen,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "effective_batch": eff,
        "n_epochs": n_epochs,
        "rollout_rows": rows or None,
        "optimizer_steps_per_epoch": steps,
        "updates_per_env_step": ups,
        "dose_rate_now": (lr_now * ups) if (lr_now is not None and ups is not None) else None,
        "kl_controller": kl_controller_block(
            kl_snapshot, num_timesteps=getattr(model, "num_timesteps", None)),
    }
    if pin:
        block["fork_lr_pin"] = dict(pin)
    return block
