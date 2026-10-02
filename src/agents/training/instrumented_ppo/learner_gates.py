"""K9(b) on the PYTHON env core and K9(c) — the learner's in-loop GIGO gates (M5 Lane K).

**K9(b) BEHAVIOUR-POLICY CONSISTENCY, python path** (`--behaviour-check`, default ``fatal`` on both
env cores). Lane G's probe (``rust_rollout/consistency.py``) runs its OWN learner forward before the
epoch loop because a Rust-collected buffer carries rows of OLDER policy versions, which only the
``[n_steps, n_envs]`` layout (before ``get()`` shuffles it) can tell apart. A buffer that carries no
version record — every python-core rollout — is played entirely by the weights the learner holds, so
the check needs no forward of its own: it reads the FIRST micro-batch's ``evaluate_actions`` output
(epoch 0, before any optimizer step can have run) against the rollout's stored ``old_log_prob`` — one
host read per update — and a failure is a typed `BehaviourMismatch`. The STATISTIC and its bar are
keyed by the run's float32 matmul precision (``consistency.BEHAVIOUR_GATES``, measured —
``designs/training/learner_gates.md``): at fp32 DETERMINISTIC — the rows whose forward sits within a
rounding error of a discrete selection / threshold cutoff are excluded (`behaviour_margins_first_micro`:
one no-grad eager forward of the micro-batch under `tie_margins.TieMargins`, before its own forward),
every other row must have ``|Δ|`` < 1e-4 (FATAL at once), and the excluded share must stay under its
ceiling; under TF32 BOTH the micro-batch's
``p99`` < 3.6e-3 (global faults) and its ``max`` < 0.071 (localized gross faults) — the TF32 max is
PERSISTENT: one violation warns loudly and dumps the offending rows (``<run_dir>/behaviour_violations.jsonl``);
FATAL when it recurs on 4 consecutive updates (``TF32_MAX_PERSISTENCE``, from the measured tail). One table and one enforcement
(``consistency.enforce_behaviour``), read by both implementations. Which implementation runs is decided
ONCE per ``train()`` by `behaviour_gate_mode`, so a buffer never pays both.

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

from agents.training.rust_rollout.consistency import behaviour_statistic, enforce_behaviour
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
    """``"probe"`` (Lane G's pre-loop probe: the buffer carries per-row policy versions), ``"in_loop"``
    (every row is current: the first micro-batch's forward is compared), or ``"off"``."""
    mode = str(getattr(model, "behaviour_check", "off") or "off")
    if mode == "off":
        return "off"
    return "probe" if getattr(model, "_rust_row_versions", None) is not None else "in_loop"


def behaviour_margins_first_micro(model: Any, observations: Any, actions: Any, masks: Any) -> Any:
    """K9(b), python path: the first micro-batch's TIE MARGINS (`tie_margins.selection_gaps`: one no-grad,
    eager, train-mode forward of the learner under the recorder) — called BEFORE that micro-batch's own
    forward, so the stashes the fold reads afterwards are that forward's, and before any optimizer step,
    so the weights are the ones that played the rows. None when the run's precision judges every row
    (no tie exclusion: TF32)."""
    from agents.training.rust_rollout.consistency import checked_margins, tie_eps
    from agents.training.rust_rollout.tie_margins import TieMargins

    if tie_eps() <= 0:
        return None
    acts = actions.reshape(-1).long()
    rec = TieMargins(int(acts.shape[0]))
    was = model.policy.training
    model.policy.set_training_mode(True)
    try:
        with th.no_grad(), rec:
            model.policy.evaluate_actions(observations, acts, action_masks=masks)
    finally:
        model.policy.set_training_mode(was)
    checked_margins(model, rec)
    return rec.margin, rec.site


def check_behaviour_first_micro(model: Any, log_prob: th.Tensor, old_log_prob: th.Tensor,
                                actions: Any = None, masks: Any = None, margins: Any = None) -> float:
    """K9(b), python path (module docs): the first micro-batch's recomputed log-probs vs the stored
    behaviour log-probs, judged and enforced by the ONE precision-keyed gate
    (``consistency.enforce_behaviour`` — the tie exclusion, persistence, the row dump, FATAL / warn).
    ``margins`` is `behaviour_margins_first_micro`'s ``(margin, site)`` — required at fp32. Records
    ``behaviour/*``; ``actions`` / ``masks`` (the micro-batch's) are read only for a dump. Returns the max |Δ|."""
    d = (log_prob.detach().reshape(-1).double() - old_log_prob.detach().reshape(-1).double()).abs()
    a = d.cpu().numpy()                                   # the one host read (a micro-batch of floats)
    worst = float(a.max()) if a.size else 0.0
    metrics = {"behaviour/max_abs_dlogp_current": worst,
               "behaviour/p99_abs_dlogp_current": behaviour_statistic(a, "p99"),
               "behaviour/rows_current": float(a.size), "behaviour/rows_probed": float(a.size)}
    logger = getattr(model, "logger", None)
    try:
        metrics.update(enforce_behaviour(
            model, a, where=f"the first micro-batch ({a.size} rows, every one played by the weights the "
                            "learner holds now)", actions=actions, masks=masks,
            margins=None if margins is None else margins[0], sites=None if margins is None else margins[1]))
    finally:
        if logger is not None:
            for k, v in metrics.items():
                logger.record(k, v)
        model._behaviour_probe_metrics = metrics
    return worst


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
