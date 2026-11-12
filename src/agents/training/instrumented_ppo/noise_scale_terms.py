"""PER-LOSS-TERM gradient noise scale — *whose* noise is the total `train/noise_scale` reading?

`train/noise_scale` (`noise_scale.py`) estimates McCandlish's critical batch
`B_simple = tr(Σ)/|G|²` from the TOTAL gradient. On this tree the total gradient is not the PPO
policy gradient: it is the policy term plus the value term plus the entropy bonus plus a dozen
DENSE supervised auxiliaries (belief heads, win-prob, spread/nature/HP-type, opponent intent). A supervised head's per-example
gradients AGREE far more than a policy-gradient's do (its target is a label, not an advantage), so
a large aux share DEFLATES the measured `tr(Σ)/|G|²` — and the run reads "over-batched" while the
term you actually care about is starved. That confound is the whole reason this module exists.

WHAT IT DOES. On one sampled `train()` call, over the SAME two batch sizes the total estimator
already uses for free (one micro-batch `B = batch_size`, and the accumulated first group
`B = batch_size·accum`), it accumulates the gradient of each GROUP of loss terms separately and
feeds the two squared norms through the SAME pure `_noise_scale_estimate` two-point solve. Four
groups: `policy` (the clipped surrogate as folded), `value` (`vf_coef·value_loss`), `entropy`
(`ent_coef·ent_loss_used`), `aux` (every belief / win-prob / intent term).

HOW IT IS WIRED, and why it cannot change the update. R1 returns each loss term beside the loss
(`micro_step`'s `terms` / `term_groups`); the fold hands them to the tagger (`tag_terms` → `add`,
which RETURNS ITS ARGUMENT UNCHANGED), so the loss expression is the one that was there before,
tensor-for-tensor. The gradients come from `torch.autograd.grad`, which never writes `.grad`. When
the feature is off, the null tagger's `add` is a two-instruction passthrough and no gradient is
taken at all.

WHOSE GRAPH (gen3_probe_own_graph_v1). In the learner each group's gradient is taken from the
group's OWN re-run of R1 on the same rows at the same parameters (`flush_micro(reforward)`,
`probe_forward.ProbeForward`), AFTER the micro-batch's real `backward()` and BEFORE the optimizer
step, with nothing retained: the graph is freed as it is walked, so the probe's peak is a training
forward's, not a retained graph plus a second backward's transient on top of it (that sum was the
learner's peak — 9,106 MiB at d_model 256 against 7,139 for the forward — and the term that kept
the wide arm from fitting the card: `designs/training/learner_lifecycle.md`). The value is the same
function of the same inputs as a gradient read off the micro-batch's own retained graph, which
`flush_micro()` with no ``reforward`` still does for a direct caller and is the reference the
equivalence tests compare against (`probe_own_graph_test.py`).

WHAT IT COSTS. Per micro-batch, one R1 forward and one backward traversal per live group, on
`accum` micro-batches, on one `train()` call in `--diagnostics-every` (`diagnostics_cadence.py`).
Extra memory is `len(active groups) × Σ|params|` (one gradient accumulator per group, freed at the
end of the call) beside one forward's graph. The measured overhead and the resulting default live
in `designs/training/step_size_and_batch.md`; the live meter is `train/noise_per_term_ms`.

A FAILURE IS LOUD. Nothing is caught here: whatever the probe raises — an out-of-memory error
included — leaves `train()`. The startup update-fit check runs this probe on the first (always
diagnostic) update's shape before any rollout (`update_fit.dry_update`), so a probe that does not
fit the card is an `UpdateWontFit` at startup, not a metric that quietly stops being written for
one arm of a comparison (the per-call disable this replaced did exactly that at d_model 256).
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch as th

from agents.training.batched_reads import read_floats
from agents.training.grad_balance import ProbeGraphError

#: The gradient groups, in reporting order. `aux` is deliberately ONE bucket rather than one entry
#: per head — `grad/<term>_share` already breaks the heads out individually, and the question this
#: module answers ("do the dense supervised losses deflate the total reading?") is answered by the
#: aggregate.
NOISE_TERM_GROUPS: Tuple[str, ...] = ("policy", "value", "entropy", "aux")

#: `train/noise_per_term_groups`: how many groups the sampled update folded (0 = the first accumulation
#: group was not completed, so no group has a second batch-size point). Written on every sampled update.
GROUPS_SAMPLED_TAG = "train/noise_per_term_groups"


def state_tags(group: str) -> Tuple[str, str]:
    """The two tags a sampled update ALWAYS writes for a group it measured: the smoothed `|G_g|²` and
    `tr(Σ_g)` the three derived tags are computed from. They are the group's whole estimator state, so
    a derived tag that is withheld is explained by the same dump (gen3_noise_term_keys_v1)."""
    return f"train/noise_scale_g2_{group}", f"train/noise_scale_trsigma_{group}"


def derived_tags(group: str) -> Tuple[str, str, str]:
    """`(B_simple, B_simple / effective batch, |G_g|² / |G_total|²)` — each written only while its
    VALUE gate holds (`NoiseScaleDiagnostics._fold_per_term_noise`): the first two need both smoothed
    estimates positive (`|G_g|²` above 1e-12), the share a positive `|G_g|²` and a readable total."""
    return (f"train/noise_scale_{group}", f"train/noise_scale_ratio_{group}",
            f"train/noise_scale_share_{group}")


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


def sq_norm(grads: Iterable[Optional[th.Tensor]]) -> float:
    """‖g‖² over a gradient tuple, tolerating the `None`s `allow_unused=True` returns."""
    acc = sq_norm_t(grads)
    return float(acc) if acc is not None else 0.0


def sq_norm_t(grads: Iterable[Optional[th.Tensor]]) -> Optional[th.Tensor]:
    """`sq_norm` left ON THE DEVICE (None for no gradient): the same sum in the same order, read later
    in one batch with the others (`batched_reads.read_floats`, gen3_batched_host_reads_v1)."""
    acc = None
    for g in grads:
        if g is None:
            continue
        s = g.detach().pow(2).sum()
        acc = s if acc is None else acc + s
    return acc


#: The original private spelling, kept so nothing that reached for it breaks.
_sq_norm = sq_norm


def sum_terms(terms: Sequence[th.Tensor]) -> th.Tensor:
    """`Σ terms` as ONE tensor, summed in list order so the graph is built identically every call."""
    return terms[0] if len(terms) == 1 else sum(terms[1:], terms[0])


def term_gradient(terms: Sequence[th.Tensor], params: Sequence[th.nn.Parameter],
                  *, retain_graph: bool = True) -> Tuple[Optional[th.Tensor], ...]:
    """`d(Σ terms)/d params` WITHOUT writing a single `.grad` — the one way this tree reads a
    per-term gradient.

    `torch.autograd.grad` returns the gradients rather than accumulating them, so a caller can read
    a loss term's own pull while the real `backward()` that follows is completely unaffected.
    `PerTermNoiseSampler` (which measures each group's noise) is its caller; a second implementation
    would be a second answer to "what is this group's gradient".
    """
    return th.autograd.grad(sum_terms(terms), params,
                            retain_graph=retain_graph, allow_unused=True)


def _is_live(term: Any) -> bool:
    """Does ``term`` carry a gradient the sampler can take (a tensor on a graph)?"""
    return isinstance(term, th.Tensor) and term.requires_grad


def tag_terms(tagger: Any, terms: Mapping[str, Any], term_groups: Mapping[str, str],
              present: Mapping[str, bool]) -> None:
    """Hand one R1 forward's loss terms to ``tagger``, each under its group — the ONE rule for which
    terms a micro-batch's per-term gradient covers, used for the real forward and for every probe
    re-run of it: a term R1 registered (``terms``) and scored on these rows (``present``; a name the
    read does not carry counts as scored)."""
    for name, term in terms.items():
        if present.get(name, True):
            tagger.add(term_groups[name], term)


class _GroupCollector:
    """`tag_terms`' sink for a probe re-run: the live terms by group, nothing else."""

    def __init__(self) -> None:
        self.groups: Dict[str, List[th.Tensor]] = {}

    def add(self, group: str, term):
        if _is_live(term):
            self.groups.setdefault(group, []).append(term)
        return term


def grouped_terms(terms: Mapping[str, Any], term_groups: Mapping[str, str],
                  present: Mapping[str, bool]) -> Dict[str, List[th.Tensor]]:
    """One forward's live loss terms by group, by `tag_terms`' rule."""
    col = _GroupCollector()
    tag_terms(col, terms, term_groups, present)
    return col.groups


class NullTermTagger:
    """The OFF path: `add` hands the term straight back, `flush_micro` does nothing.

    A singleton rather than a `None` check at ~25 fold sites — the call sites read
    `loss = loss + _ntg.add("aux", term)` whether or not anything is being measured, so the fold
    order (which is a contract, see `ppo.train`'s docstring) stays readable as one straight line.
    """

    collecting = False
    micros = 0
    probe_seconds = 0.0

    def add(self, group: str, term):    # noqa: D102 - passthrough
        return term

    def flush_micro(self, reforward: Optional[Callable[[], Dict[str, List[th.Tensor]]]] = None) -> None:
        return None                     # noqa: D102 - no-op: no forward is re-run, no gradient taken

    def result(self, accum: int) -> Dict[str, Tuple[float, float]]:   # noqa: D102
        return {}

    def release(self) -> None:          # noqa: D102
        return None


NULL_TAGGER = NullTermTagger()


class PerTermNoiseSampler:
    """Accumulates per-GROUP gradients across the first accumulation group of one `train()` call.

    Lifecycle, per sampled `train()`:
      * `add(group, term)` during each minibatch's fold (`tag_terms`) — records the tensor, returns it
        unchanged.
      * `flush_micro(reforward)` once per micro-batch — one `autograd.grad` per group with live terms;
        the first micro-batch's squared norms are kept as the `B = batch_size` point, and every
        micro-batch's gradient is summed into a per-group accumulator. With ``reforward`` (the
        learner: after the real `backward()`, before the optimizer step) each group's gradient comes
        from its own re-run of the forward and nothing is retained; without it the gradients are read
        off the graph the tagged terms hold, which must then still be alive (call BEFORE `backward()`).
      * `result(accum)` after the group closes — `(g_small_sq, g_big_sq)` per group, where
        `g_big_sq = ‖Σ_i g_i / accum‖²` mirrors exactly what `.grad` holds for the total after the
        accum micro-batches of `(loss/accum).backward()`.
      * `release()` — drop the accumulators.

    It catches nothing (module docs, "A FAILURE IS LOUD").
    """

    collecting = True

    def __init__(self, params: Sequence[th.nn.Parameter],
                 groups: Sequence[str] = NOISE_TERM_GROUPS) -> None:
        self._params: List[th.nn.Parameter] = [p for p in params if p.requires_grad]
        self._groups: Tuple[str, ...] = tuple(groups)
        self._pending: Dict[str, List[th.Tensor]] = {}
        self._accum: Dict[str, List[Optional[th.Tensor]]] = {}
        #: each group's first-micro-batch ‖g‖², ON THE DEVICE until `result` reads them all at once
        self.small_sq: Dict[str, Optional[th.Tensor]] = {}
        self.micros: int = 0
        self.probe_seconds: float = 0.0

    def add(self, group: str, term):
        """Tag `term` as belonging to `group` and RETURN IT UNCHANGED (the fold must not move)."""
        if _is_live(term):
            self._pending.setdefault(group, []).append(term)
        return term

    def flush_micro(self, reforward: Optional[Callable[[], Dict[str, List[th.Tensor]]]] = None) -> None:
        """Take this micro-batch's per-group gradients.

        ``reforward()`` re-runs the micro-batch's forward and returns its live terms by group
        (`grouped_terms`); it is called once per live group and that call's graph is consumed by the
        group's one gradient (``retain_graph=False``). A group the real forward tagged and the re-run
        does not produce is a `ProbeGraphError`. ``reforward=None``: differentiate the tagged terms on
        the graph they hold, retained (the reference path — call it BEFORE the real `backward()`)."""
        pending, self._pending = self._pending, {}
        if not self._params:
            return
        t0 = time.perf_counter()
        try:
            first = self.micros == 0
            for group in self._groups:
                terms = pending.get(group)
                if not terms:
                    continue
                if reforward is None:
                    grads = term_gradient(terms, self._params)
                else:
                    fresh = reforward().get(group)
                    if not fresh or len(fresh) != len(terms):
                        raise ProbeGraphError(
                            f"the per-term noise probe's own forward produced {len(fresh or ())} live "
                            f"{group!r} term(s) where the micro-batch's forward tagged {len(terms)}: a re-run "
                            "of the micro-step on the same rows must register the same terms "
                            "(gen3_probe_own_graph_v1)")
                    grads = term_gradient(fresh, self._params, retain_graph=False)
                    del fresh
                buf = self._accum.get(group)
                if buf is None:
                    self._accum[group] = [None if g is None else g.detach().clone() for g in grads]
                else:
                    for i, g in enumerate(grads):
                        if g is None:
                            continue
                        buf[i] = g.detach().clone() if buf[i] is None else buf[i] + g.detach()
                if first:
                    self.small_sq[group] = sq_norm_t(grads)
            self.micros += 1
        finally:
            self.probe_seconds += time.perf_counter() - t0

    def result(self, accum: int) -> Dict[str, Tuple[float, float]]:
        """`{group: (g_small_sq, g_big_sq)}` — empty unless the whole first group was sampled.

        A partial group (the KL early-stop discards one) carries no `B = batch_size·accum` point,
        so it yields nothing rather than a wrong second point — the same discipline the total
        estimator applies by leaving `noise_g_big_sq` None.
        """
        if self.micros < accum or accum < 2:
            return {}
        inv = 1.0 / float(accum * accum)
        # first appeared on a later micro-batch — no matched small-batch point: skipped
        groups = [g for g in self._accum if g in self.small_sq]
        host = read_floats([self.small_sq[g] for g in groups]
                           + [sq_norm_t(self._accum[g]) for g in groups])    # ONE host read
        n = len(groups)
        return {g: (host[i], host[n + i] * inv) for i, g in enumerate(groups)}

    def release(self) -> None:
        """Drop the per-group gradient accumulators (the probe's only non-trivial memory)."""
        self._accum = {}
        self._pending = {}
