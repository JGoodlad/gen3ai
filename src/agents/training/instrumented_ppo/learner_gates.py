"""K9(b)'s dispatch and K9(c) — the learner's in-loop GIGO gates (M5 Lane K).

**K9(b) BEHAVIOUR-POLICY CONSISTENCY** (`--behaviour-check`, default ``fatal``) has ONE implementation:
Lane G's pre-loop probe (``rust_rollout/consistency.py``), its own learner forward before the epoch loop
on the ``[n_steps, n_envs]`` layout, age-bucketed by the rows' policy versions (a buffer with no version
record is judged as every row current). `behaviour_gate_mode` says whether ``train()`` runs it. The
python env core's variant — the FIRST micro-batch's own forward compared in the loop, for a buffer with
no versions — was deleted with that core's last user (deletion pass U4): every buffer now comes from
the Rust collector, which stamps versions. The statistic, its bar and the tie exclusion are
``consistency.BEHAVIOUR_GATE`` / ``consistency.enforce_behaviour`` (``designs/training/learner_gates.md``).

**K9(c) FAIL-CLOSED NON-FINITE LOSS / GRADIENT.** Before this, nothing in ``train()`` looked: a NaN
loss back-propagated NaN gradients, ``clip_grad_norm_`` (``error_if_nonfinite=False``) scaled every
gradient by ``max_norm / NaN`` in place, and ``optimizer.step()`` wrote NaN into every parameter and
into Adam's moments — silently, with the run continuing on a dead network. Two checks, both before
the optimizer can move anything:

* `check_loss_finite` — once per micro-batch, on the ASSEMBLED loss (every fold included), naming the
  term(s) that went non-finite (the names are the grad-balance probe's, plus policy/entropy/value);
* `clip_grad_norm_checked` — at every optimizer step, the pre-clip total norm the step already
  reads, with ``error_if_nonfinite`` so a NaN/Inf norm raises BEFORE the in-place scaling, naming the
  parameters whose gradient is non-finite. This one also covers a finite loss whose BACKWARD produced
  a NaN (and the grad-accumulation flush's rescale).

plus `check_buffer_finite` (once per update, before any forward: rewards, values, log-probs, advantages, returns and
every float label key) and `check_kl_finite` (the approx-KL can be Inf under a finite loss). The rest of
the audit — the sites that absorbed a NaN BEFORE the total — is ``designs/training/learner_gates.md``.

All raise `NonFiniteLearnerError` BEFORE the optimizer step, so the weights a crash-save writes are the
last finite ones. Every message starts with `LEARNER_FATAL_TAG` (``[Learner] FATAL``) and the class name
is ``main.exit_codes.NonFiniteLearnerError``: the trainer exits ``TrainExitCode.FATAL_NONFINITE`` (4)
and the launcher does NOT restart (cutover-prep, ``743008c1``). This module only raises.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import torch as th

from main.exit_codes import NonFiniteLearnerError


#: The tag every non-finite FATAL's message starts with (`nonfinite`), beside the class itself —
#: ``main.exit_codes.NonFiniteLearnerError`` (a ``FloatingPointError``), which the trainer's fail-fast
#: handlers map to ``TrainExitCode.FATAL_NONFINITE`` (4) through ``exit_code_for`` and the launcher does
#: NOT restart (cutover-prep, ``743008c1``). This module only raises; it defines no class of its own.
LEARNER_FATAL_TAG = "[Learner] FATAL"


def nonfinite(message: str) -> NonFiniteLearnerError:
    """The K9(c) error for ``message``, tagged `LEARNER_FATAL_TAG` once: ``raise nonfinite("...")``."""
    message = str(message)
    return NonFiniteLearnerError(message if message.startswith(LEARNER_FATAL_TAG)
                                 else f"{LEARNER_FATAL_TAG} {message}")


def behaviour_gate_mode(model: Any) -> str:
    """``"probe"`` (Lane G's pre-loop probe — every ``--behaviour-check`` but ``off``) or ``"off"``."""
    mode = str(getattr(model, "behaviour_check", "off") or "off")
    return "off" if mode == "off" else "probe"


def _nonfinite(t: Any) -> bool:
    if isinstance(t, th.Tensor):
        return not bool(th.isfinite(t.detach()).all())
    return isinstance(t, float) and not math.isfinite(t)


def check_loss_finite(loss: th.Tensor, terms: Dict[str, Any], *, epoch: int, micro: int) -> None:
    """K9(c): raise `NonFiniteLearnerError` if the assembled ``loss`` is NaN/Inf (module docs)."""
    if bool(th.isfinite(loss.detach()).all()):
        return
    bad = [k for k, v in terms.items() if v is not None and _nonfinite(v)]
    named = ", ".join(bad) if bad else ("none of the named terms — an unnamed fold (e.g. the set-valued "
                                        "beta term) or the sum itself overflowed")
    raise nonfinite(
        f"[K9(c)] NON-FINITE LOSS {float(loss.detach()):g} at epoch {epoch} (micro-batch {micro} of this "
        f"update), before its backward — non-finite term(s): {named}. Refusing to apply it (the optimizer would write NaN into "
        "every parameter).")


def clip_grad_norm_checked(policy: Any, max_norm: float, *, epoch: int) -> float:
    """``clip_grad_norm_`` with K9(c) fail-closed: the pre-clip total norm (the value the fold already
    logs as ``grad_norms``), or `NonFiniteLearnerError` naming the parameters whose gradient is NaN/Inf.

    ``error_if_nonfinite=True`` raises BEFORE the in-place scaling (torch's default scales every
    gradient by ``max_norm / NaN``, which would erase the attribution), so the gradients read below are
    the ones the backward produced. The norm and the clipping arithmetic are unchanged."""
    params = list(policy.parameters())
    try:
        norm = th.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=True)
    except RuntimeError as exc:
        bad = [name for name, p in policy.named_parameters()
               if p.grad is not None and not bool(th.isfinite(p.grad).all())]
        if not bad:
            raise
        raise nonfinite(
            f"[K9(c)] NON-FINITE GRADIENT at epoch {epoch}, before optimizer.step() — {len(bad)} parameter(s) "
            f"with a NaN/Inf gradient, first: {bad[:8]}. The loss was finite (checked per micro-batch), so a "
            "BACKWARD produced it. Refusing the step.") from exc
    return float(norm)


def check_kl_finite(approx_kl: float, *, epoch: int) -> None:
    """K9(c): the per-micro-batch approx-KL must be finite. It CAN be non-finite under a finite loss (an
    overflowed ratio on a positive-advantage row takes the clipped branch), and a NaN/Inf reading pins the
    KL→LR controller's EMA for the rest of the run (`adaptive_lr_callback`)."""
    if not math.isfinite(approx_kl):
        raise nonfinite(
            f"[K9(c)] NON-FINITE approx-KL {approx_kl} at epoch {epoch}: a row's probability ratio overflowed "
            "(the policy moved a taken action's probability by more than e^88) — refusing to continue.")


#: The buffer's own trained arrays (``values`` / ``log_probs`` are the behaviour quantities PPO reads).
_BUFFER_ARRAYS = ("rewards", "values", "log_probs", "advantages", "returns")


def check_buffer_finite(buf: Any) -> None:
    """K9(c): once per update and before any forward — every trained array of
    the rollout buffer and every FLOAT label key of its observation dict is finite. The flat
    ``observation`` row is excluded (it is a policy INPUT: a NaN there reaches the loss through the
    forward, and a full scan is ~0.3 s per update at production size) — as are integer keys (a NaN
    cannot be stored in them). Cost at production size: ~4 x 98,304 floats + the labels, sub-ms."""
    bad: List[str] = []
    for name in _BUFFER_ARRAYS:
        arr = getattr(buf, name, None)
        if arr is not None and not np.isfinite(np.asarray(arr)).all():
            bad.append(f"{name} ({int((~np.isfinite(np.asarray(arr))).sum())} rows)")
    obs = getattr(buf, "observations", None)
    if isinstance(obs, dict):
        for k, v in obs.items():
            if k == "observation":
                continue
            a = np.asarray(v)
            if a.dtype.kind == "f" and not np.isfinite(a).all():
                bad.append(f"obs[{k!r}] ({int((~np.isfinite(a)).sum())} values)")
    if bad:
        raise nonfinite(
            f"[K9(c)] NON-FINITE ROLLOUT BUFFER before the update: {', '.join(bad)}. A NaN reward / value / "
            "label is garbage in (the collector, GAE or a label callback produced it) — refusing to train "
            "on it.")
