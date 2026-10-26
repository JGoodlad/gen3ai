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
  term(s) that went non-finite (the names are the grad-balance probe's, plus policy/entropy/value).
  Since `gen3_batched_host_reads_v1` the verdict of a micro-batch whose values nothing reads earlier is
  made from its deferred host read (`check_loss_finite_read`) at the optimizer step it joins — after its
  backward, still before that step: the backward writes only ``.grad``, and the raise means no step
  ever applies it, so the weights and Adam's moments are the last finite ones either way;
* `clip_grad_norm_checked` — at every optimizer step, the pre-clip total norm the step already
  reads (in the same transfer as the deferred reads), judged on the host so a NaN/Inf norm raises
  BEFORE the in-place scaling, naming the parameters whose gradient is non-finite. This one also covers a finite loss whose BACKWARD produced
  a NaN (and the grad-accumulation flush's rescale).

plus `check_buffer_finite` (once per update, before any forward: rewards, values, log-probs, advantages, returns and
every float label key) and `check_kl_finite` (the approx-KL can be Inf under a finite loss).

**THE FULL-MICRO-BATCH RULE** (`gen3_learner_full_micro_v1`, TECH_DEBT F-LG-12 part 2): `check_full_micro_batches`
refuses, once per update and before any forward, a buffer whose row count does not divide by the micro-batch —
`RaggedMicroBatchError` (a ``FatalConfigError`` → ``FATAL_CONFIG``; not a K9(c) non-finite fatal). A short last micro-batch was weighted
as if full-size by the accumulation (the eager path's former "bounded mis-weighting"); the Rust collector never
emits one (`RustCollector._ensure_buffer`, the trigger's quantum) and compiled R1 refuses one, so the eager
learner now holds the same contract instead of trusting its caller. The rest of
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

from main.exit_codes import FatalConfigError, NonFiniteLearnerError


#: The tag every non-finite FATAL's message starts with (`nonfinite`), beside the class itself —
#: ``main.exit_codes.NonFiniteLearnerError`` (a ``FloatingPointError``), which the trainer's fail-fast
#: handlers map to ``TrainExitCode.FATAL_NONFINITE`` (4) through ``exit_code_for`` and the launcher does
#: NOT restart (cutover-prep, ``743008c1``). This module only raises; its one class is `RaggedMicroBatchError`.
LEARNER_FATAL_TAG = "[Learner] FATAL"


class RaggedMicroBatchError(FatalConfigError):
    """`check_full_micro_batches`: an update whose rows do not divide by the learner's micro-batch — a
    deterministic configuration defect, so ``TrainExitCode.FATAL_CONFIG`` (no restart loop)."""


def check_full_micro_batches(buf: Any, batch_size: int) -> None:
    """Every micro-batch of this update is FULL (`gen3_learner_full_micro_v1`): ``buffer_size * n_envs`` is a
    multiple of ``batch_size``. Once per update, before any forward. A short last micro-batch would join an
    accumulation group as if it held ``batch_size`` rows (``loss / accum``), over-weighting each of its rows —
    the same contract the Rust collector (`RustCollector._ensure_buffer`) and compiled R1 already enforce."""
    rows = int(buf.buffer_size) * int(buf.n_envs)
    micro = int(batch_size)
    if micro <= 0 or rows % micro:
        raise RaggedMicroBatchError(
            f"{LEARNER_FATAL_TAG} an update of {rows:,} rows does not divide by the micro-batch {micro:,}: "
            f"the last micro-batch would hold {rows % micro if micro > 0 else rows:,} rows and be weighted as "
            "if full (gen3_learner_full_micro_v1). Pick a --batch-size that divides the update (the Rust "
            "collector's trigger quantum guarantees it).")


class LabelVocabError(FatalConfigError):
    """`check_label_vocab_buffer`: a rollout label id outside its head's vocabulary — the label↔embedding num
    pipeline is corrupt (deterministic: ``TrainExitCode.FATAL_CONFIG``, no restart loop)."""


#: The Hidden Power type head's class count (``HIDDEN_POWER_TYPE_ORDER``, the [B, 6, 16] logits).
_N_HP_TYPES = 16


def check_label_vocab_buffer(buf: Any, extractor: Any) -> None:
    """``gen3_label_vocab_raise_v1``: once per update, before any forward, every categorical LABEL the rollout
    buffer carries lies in its head's vocabulary — the believed species / moves (``max_species`` /
    ``max_moves``), the item belief (``max_items``) and the Hidden Power type (16), the latter two at their
    SCORED slots (``belief_bank_static.check_label_vocab``). The compiled micro-step cannot raise on data (a
    static-shape program), so this host check on the buffer's numpy arrays is the gate; it used to exist only as
    a function nothing called, and the item / hp-type twin clamped. Cost: a max / min over a few int arrays."""
    obs = getattr(buf, "observations", None)
    layout = getattr(extractor, "layout", None)
    if not isinstance(obs, dict) or not isinstance(layout, dict):
        return
    from agents.training.belief_bank_static import check_label_vocab

    try:
        check_label_vocab(obs, n_species=layout.get("max_species"), n_moves=layout.get("max_moves"),
                          n_items=layout.get("max_items"), n_hp_types=_N_HP_TYPES)
    except ValueError as exc:
        raise LabelVocabError(f"{LEARNER_FATAL_TAG} [labels] {exc}") from exc


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
    if isinstance(t, np.ndarray):
        return not bool(np.isfinite(t).all())
    return isinstance(t, float) and not math.isfinite(t)


def check_loss_finite(loss: th.Tensor, terms: Dict[str, Any], *, epoch: int, micro: int) -> None:
    """K9(c): raise `NonFiniteLearnerError` if the assembled ``loss`` is NaN/Inf (module docs)."""
    if bool(th.isfinite(loss.detach()).all()):
        return
    raise_nonfinite_loss(float(loss.detach()), terms, epoch=epoch, micro=micro, where="before its backward")


def check_loss_finite_read(loss_finite: bool, loss_value: float, terms: Dict[str, Any], *, epoch: int,
                           micro: int) -> None:
    """K9(c), from the HOST READ of a micro-batch (`micro_step.unpack`; ``terms`` are float32 host
    arrays): the same verdict and message as `check_loss_finite`, made at the optimizer step the
    micro-batch joins (`gen3_batched_host_reads_v1`) — still BEFORE that step moves anything."""
    if loss_finite:
        return
    raise_nonfinite_loss(loss_value, terms, epoch=epoch, micro=micro,
                         where="after its backward, before the optimizer step it joins")


def raise_nonfinite_loss(loss_value: float, terms: Dict[str, Any], *, epoch: int, micro: int,
                         where: str) -> None:
    bad = [k for k, v in terms.items() if v is not None and _nonfinite(v)]
    named = ", ".join(bad) if bad else ("none of the named terms — an unnamed fold (e.g. the set-valued "
                                        "beta term) or the sum itself overflowed")
    raise nonfinite(
        f"[K9(c)] NON-FINITE LOSS {loss_value:g} at epoch {epoch} (micro-batch {micro} of this "
        f"update), {where} — non-finite term(s): {named}. Refusing to apply it (the optimizer would write NaN into "
        "every parameter).")


def clip_grad_norm_checked(policy: Any, max_norm: float, *, epoch: int, reads: Any = None) -> float:
    """``clip_grad_norm_`` with K9(c) fail-closed: the pre-clip total norm (the value the fold already
    logs as ``grad_norms``), or `NonFiniteLearnerError` naming the parameters whose gradient is NaN/Inf.

    torch's own two halves, `get_total_norm` then `clip_grads_with_norm_` (exactly what
    ``clip_grad_norm_`` runs), with the norm READ IN BETWEEN: a NaN/Inf norm raises BEFORE the in-place
    scaling (torch's default scales every gradient by ``max_norm / NaN``, which would erase the
    attribution), so the gradients named below are the ones the backward produced. The norm and the
    clipping arithmetic are unchanged.

    ``reads`` (`host_reads.HostReadQueue`, `gen3_batched_host_reads_v1`): the norm is read in the SAME
    transfer as every read the update deferred to this step — the micro-batches' metrics and their K9(c)
    loss / KL verdicts, whose callbacks run (and may raise) BEFORE the norm is judged, the order the inline
    checks ran in. One host sync per optimizer step instead of one per read."""
    params = list(policy.parameters())
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        if reads is not None:
            reads.drain()
        return 0.0
    with th.no_grad():
        total = th.nn.utils.get_total_norm(grads, 2.0, error_if_nonfinite=False, foreach=None)
    if reads is not None and total.dtype == th.float32:
        host = reads.drain([total])
        assert host is not None
        norm = float(host[0])
    else:
        if reads is not None:
            reads.drain()
        norm = float(total)
    if not math.isfinite(norm):
        bad = [name for name, p in policy.named_parameters()
               if p.grad is not None and not bool(th.isfinite(p.grad).all())]
        if not bad:
            raise RuntimeError(f"The total norm of order 2.0 for gradients from `parameters` is non-finite "
                               f"({norm}), but no parameter's gradient is — refusing the step")
        raise nonfinite(
            f"[K9(c)] NON-FINITE GRADIENT at epoch {epoch}, before optimizer.step() — {len(bad)} parameter(s) "
            f"with a NaN/Inf gradient, first: {bad[:8]}. The loss was finite (checked per micro-batch), so a "
            "BACKWARD produced it. Refusing the step.")
    with th.no_grad():
        th.nn.utils.clip_grads_with_norm_(params, max_norm, total, foreach=None)
    return norm


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
