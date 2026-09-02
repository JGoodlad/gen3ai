"""PER-LOSS-TERM gradient noise scale — *whose* noise is the total `train/noise_scale` reading?

`train/noise_scale` (`noise_scale.py`) estimates McCandlish's critical batch
`B_simple = tr(Σ)/|G|²` from the TOTAL gradient. On this tree the total gradient is not the PPO
policy gradient: it is the policy term plus the value term plus the entropy bonus plus a dozen
DENSE supervised auxiliaries (belief heads, win-prob, spread/nature/HP-type, value-dist, TD-aux,
the counterfactual family) plus, on a fold, the distillation KL. A supervised head's per-example
gradients AGREE far more than a policy-gradient's do (its target is a label, not an advantage), so
a large aux share DEFLATES the measured `tr(Σ)/|G|²` — and the run reads "over-batched" while the
term you actually care about is starved. That confound is the whole reason this module exists.

WHAT IT DOES. On one sampled `train()` call, over the SAME two batch sizes the total estimator
already uses for free (one micro-batch `B = batch_size`, and the accumulated first group
`B = batch_size·accum`), it accumulates the gradient of each GROUP of loss terms separately and
feeds the two squared norms through the SAME pure `_noise_scale_estimate` two-point solve. Five
groups: `policy` (the clipped surrogate as folded), `value` (`vf_coef·value_loss`), `entropy`
(`ent_coef·ent_loss_used`), `aux` (every belief / win-prob / value-dist / TD-aux / counterfactual
/ search-teacher / OPD term), `distill` (the `--distill-coef` family).

HOW IT IS WIRED, and why it cannot change the update. The tagger is threaded through the fold as
`loss = loss + _ntg.add("aux", term)`: `add` RETURNS ITS ARGUMENT UNCHANGED, so the loss
expression is the one that was there before, tensor-for-tensor. The gradients come from
`torch.autograd.grad(..., retain_graph=True)`, which never writes `.grad` — structurally the same
read-only probe `grad_balance_metrics` has run per-term on every `train()` for generations (which
is also why `--compile-trainer` is not a new risk here: the compiled backward is already called
repeatedly with `retain_graph` by that probe). When the feature is off, the null tagger's `add` is
a two-instruction passthrough and no gradient is taken at all.

WHAT IT COSTS. `len(groups)` extra backward traversals per micro-batch, on `accum` micro-batches,
on one `train()` call in `_NOISE_PER_TERM_EVERY`. Peak extra memory is
`len(active groups) × Σ|params|` (one gradient accumulator per group, freed at the end of the
call). The measured overhead and the resulting default live in `src/agents/training/CLAUDE.md`.

FAILURE IS SELF-DISABLING. Anything the probe raises retires it for the rest of the call with one
printed line and leaves the training step untouched: a diagnostic must never take a run down.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch as th

#: The gradient groups, in reporting order. `aux` is deliberately ONE bucket rather than one entry
#: per head — `grad/<term>_share` already breaks the heads out individually, and the question this
#: module answers ("do the dense supervised losses deflate the total reading?") is answered by the
#: aggregate. `distill` is separated from `aux` because it is the one group that comes and goes
#: with a fold and whose dose is the thing being tuned.
NOISE_TERM_GROUPS: Tuple[str, ...] = ("policy", "value", "entropy", "aux", "distill")


def per_term_enabled(model: object) -> bool:
    """Is the per-term probe on for this process?

    `$GEN3AI_NOISE_SCALE_PER_TERM` (``0``/``false``/``off`` to disable) wins over the
    `PpoHyperparameters.noise_scale_per_term` class default. It is an ENV knob rather than a CLI
    flag on purpose: the probe changes no training math, so it never belongs in
    `model_config.json`, and an env var is the one switch a launcher resume cannot forget to
    forward.
    """
    env = os.environ.get("GEN3AI_NOISE_SCALE_PER_TERM")
    if env is not None:
        return env.strip().lower() not in ("", "0", "false", "off", "no")
    return bool(getattr(model, "noise_scale_per_term", True))


def _sq_norm(grads: Iterable[Optional[th.Tensor]]) -> float:
    """‖g‖² over a gradient tuple, tolerating the `None`s `allow_unused=True` returns."""
    acc = None
    for g in grads:
        if g is None:
            continue
        s = g.detach().pow(2).sum()
        acc = s if acc is None else acc + s
    return float(acc) if acc is not None else 0.0


class NullTermTagger:
    """The OFF path: `add` hands the term straight back, `flush_micro` does nothing.

    A singleton rather than a `None` check at ~25 fold sites — the call sites read
    `loss = loss + _ntg.add("aux", term)` whether or not anything is being measured, so the fold
    order (which is a contract, see `ppo.train`'s docstring) stays readable as one straight line.
    """

    collecting = False
    micros = 0
    probe_seconds = 0.0
    failed = False

    def add(self, group: str, term):    # noqa: D102 - passthrough
        return term

    def flush_micro(self) -> None:      # noqa: D102 - no-op
        return None

    def result(self, accum: int) -> Dict[str, Tuple[float, float]]:   # noqa: D102
        return {}

    def release(self) -> None:          # noqa: D102
        return None


NULL_TAGGER = NullTermTagger()


class PerTermNoiseSampler:
    """Accumulates per-GROUP gradients across the first accumulation group of one `train()` call.

    Lifecycle, per sampled `train()`:
      * `add(group, term)` during each minibatch's fold — records the tensor, returns it unchanged.
      * `flush_micro()` immediately BEFORE `(loss/accum).backward()` — one `autograd.grad` per
        group with live terms; the first micro-batch's squared norms are kept as the
        `B = batch_size` point, and every micro-batch's gradient is summed into a per-group
        accumulator.
      * `result(accum)` after the group closes — `(g_small_sq, g_big_sq)` per group, where
        `g_big_sq = ‖Σ_i g_i / accum‖²` mirrors exactly what `.grad` holds for the total after the
        accum micro-batches of `(loss/accum).backward()`.
      * `release()` — drop the accumulators.
    """

    collecting = True

    def __init__(self, params: Sequence[th.nn.Parameter],
                 groups: Sequence[str] = NOISE_TERM_GROUPS) -> None:
        self._params: List[th.nn.Parameter] = [p for p in params if p.requires_grad]
        self._groups: Tuple[str, ...] = tuple(groups)
        self._pending: Dict[str, List[th.Tensor]] = {}
        self._accum: Dict[str, List[Optional[th.Tensor]]] = {}
        self.small_sq: Dict[str, float] = {}
        self.micros: int = 0
        self.probe_seconds: float = 0.0
        self.failed: bool = False

    def add(self, group: str, term):
        """Tag `term` as belonging to `group` and RETURN IT UNCHANGED (the fold must not move)."""
        if isinstance(term, th.Tensor) and term.requires_grad:
            self._pending.setdefault(group, []).append(term)
        return term

    def flush_micro(self) -> None:
        """Take this micro-batch's per-group gradients. Call BEFORE the real `backward()`."""
        pending, self._pending = self._pending, {}
        if self.failed or not self._params:
            return
        t0 = time.perf_counter()
        try:
            first = self.micros == 0
            for group in self._groups:
                terms = pending.get(group)
                if not terms:
                    continue
                total = terms[0] if len(terms) == 1 else sum(terms[1:], terms[0])
                grads = th.autograd.grad(total, self._params, retain_graph=True,
                                         allow_unused=True)
                buf = self._accum.get(group)
                if buf is None:
                    self._accum[group] = [None if g is None else g.detach().clone() for g in grads]
                else:
                    for i, g in enumerate(grads):
                        if g is None:
                            continue
                        buf[i] = g.detach().clone() if buf[i] is None else buf[i] + g.detach()
                if first:
                    self.small_sq[group] = _sq_norm(grads)
            self.micros += 1
        except Exception as exc:                              # pragma: no cover - defensive
            self.failed = True
            self._accum = {}
            self.small_sq = {}
            print("⚠️ [NOISE] per-term noise-scale probe disabled for this run "
                  f"({type(exc).__name__}: {exc}). Training is unaffected; set "
                  "GEN3AI_NOISE_SCALE_PER_TERM=0 to silence.", flush=True)
        finally:
            self.probe_seconds += time.perf_counter() - t0

    def result(self, accum: int) -> Dict[str, Tuple[float, float]]:
        """`{group: (g_small_sq, g_big_sq)}` — empty unless the whole first group was sampled.

        A partial group (the KL early-stop discards one) carries no `B = batch_size·accum` point,
        so it yields nothing rather than a wrong second point — the same discipline the total
        estimator applies by leaving `noise_g_big_sq` None.
        """
        if self.failed or self.micros < accum or accum < 2:
            return {}
        inv = 1.0 / float(accum * accum)
        out: Dict[str, Tuple[float, float]] = {}
        for group, buf in self._accum.items():
            if group not in self.small_sq:
                continue    # first appeared on a later micro-batch — no matched small-batch point
            out[group] = (self.small_sq[group], _sq_norm(buf) * inv)
        return out

    def release(self) -> None:
        """Drop the per-group gradient accumulators (the probe's only non-trivial memory)."""
        self._accum = {}
        self._pending = {}
