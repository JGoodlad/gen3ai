"""SOURCE-SEPARATED DISTILLATION ANCHORING (`gen3_distill_grad_project_v1`) — constrain the
DISTILL gradient's effect on off-slice behaviour, and leave PPO's gradient completely free.

WHY THIS EXISTS, AND WHY THE OUTPUT ANCHOR CANNOT DO IT. The 2026-09-01 gift/decay pair
(`ledger.md` → *"v8's GIFT IS A TRANSIENT HUMP"* and *"WHAT v8's LAST 2.5M UNDID"*) measured that a
fold does **two things in two directions**:

  * **the GIFT** — an early off-slice habit change ORTHOGONAL to the taught content (cosine 0.14
    with the teachers' fingerprint), PPO-driven, landed by ~+3M, worth **+5-10pp on UNTAUGHT
    teams**, and 92% INTACT at +15M;
  * **the LEAK** — the teachers' taught content itself arriving on untaught boards through shared
    weights, PARALLEL to the teachers' fingerprint (cos +0.559, p 0.0015, at ~1/3 amplitude),
    harmless on taught teams and worth **-5.66pp [-12.1, -0.2]** on untaught ones.

`distill_anchor.py` penalises OUTPUT displacement from a reference on off-slice states. At the
output there is nothing left to separate: gift and leak are both "the student's off-slice policy
has moved away from the parent", so a fixed-parent anchor **taxes the gift exactly as hard as the
leak** — the ledger's design consequence (a) in as many words.

**At UPDATE time the two ARE separable, because they have different SOURCES.** The total update is
`g_ppo + g_distill`, and this tree already computes those two objects separately (the
`_ntg.add("distill", ...)` tags, whose gradients `PerTermNoiseSampler` takes every sampled
`train()`). The gift is PPO's; the leak is the distill term's. So: leave `g_ppo` untouched and
remove from `g_distill` only the components that move off-slice behaviour.

**THE MECHANISM — orthogonal gradient projection.** Sample `m` off-slice rows from the minibatch
(`--distill-anchor-proj-samples`, default 16). For each, take `grad log pi(a* | s)` where `a*` is
the student's own argmax over the LEGAL set — the direction that most directly moves that state's
behaviour. Orthonormalise those `m` vectors (modified Gram-Schmidt with one re-orthogonalisation
pass, near-degenerate directions dropped) into a basis `Q`, and step with

    g_total = g_ppo + P_perp g_distill,    P_perp g = g - sum_j q_j <q_j, g>

To first order the distill term then changes the sampled off-slice log-probabilities by **zero**,
while its component along every direction that only moves TAUGHT states survives in full. PPO's
gradient is never read, never projected, never scaled.

**PRECEDENT.** This is Orthogonal Gradient Descent (Farajtabar et al. 2020, *"Orthogonal Gradient
Descent for Continual Learning"*, AISTATS) and Gradient Projection Memory (Saha et al. 2021, ICLR)
applied at a different seam: those project the NEW-task gradient off a memory of OLD-task gradient
directions to stop forgetting across tasks; here the two "tasks" are two TERMS of one loss at one
timestep — the taught slice and everything else — and the constraint basis is rebuilt from the live
minibatch every step rather than banked.

🚨 **THE FIRST-ORDER LIMITATION, stated once and not to be forgotten.** A projection at step `t`
kills the distill term's INSTANTANEOUS effect on the sampled off-slice log-probs. It does NOT bound
the ACCUMULATED displacement: the constraint set is resampled every step, curvature carries the
policy off the tangent plane, and a systematic second-order drift can still add up over millions of
steps. That is exactly what the OUTPUT anchor bounds — so the two COMPOSE rather than compete, and
`--distill-anchor-coef > 0` beside `--distill-anchor-mode grad_project` is a supported (and
recommended-for-a-long-fold) combination: the projection removes the per-step leak, the output
anchor catches what leaks past it. `distill/collateral_kl_vs_parent` is what says whether it did.

**WHAT IS PROJECTED.** The TEACHER terms only — the policy KL, the scalar value MSE and the FitNets
hint. The output anchor term is NOT projected even though it rides the same `distill` noise-scale
group: its whole job is to act on off-slice outputs, so projecting it off the off-slice behaviour
subspace would delete it and make the composition above vacuous. The distinction is made at the
call site in `ppo.py` (`_dgp.add(...)` wraps the three teacher terms and not the anchor), not by a
group name, so it is readable where it matters.

**PER MICRO-BATCH, NOT PER ACCUMULATION GROUP** — a deliberate choice, and at the default
`--grad-accum-steps 1` the two are IDENTICAL, so the distinction exists only in the accumulate
regime. Three reasons:

  1. **Memory.** The `m` constraint vectors are FULL-PARAMETER-SIZED (`m x |theta|` floats). Holding
     them across an accumulation group multiplies that by `accum` — and `--grad-accum-steps` exists
     precisely to cut the memory peak, so a per-group projection would fight the one flag it most
     needs to compose with.
  2. **Seam size.** Per micro-batch needs ONE call after `backward()`. Per group would additionally
     need an apply before `clip_grad_norm_` in BOTH step sites (the in-loop step and the trailing
     partial-group flush) and a reset on the KL early-stop discard. `ppo.py` is ~1.8k lines against
     a hard 2,000-line gate; a seam has to earn its lines.
  3. **It is the CONSERVATIVE one.** `sum_i P_perp_i g_i` removes at most what
     `P_perp_union (sum_i g_i)` would (each per-micro span is a subset of the union), and it
     constrains each distill gradient by the off-slice states it was actually computed alongside.

🚨 **COST — IT IS NOT CHEAP, AND THE FIRST DRAFT OF THIS PARAGRAPH SAID IT WAS.** The `m`
constraint backwards run over a graph built from ONLY the `m` sampled rows (the observations are
sliced before the forward), so each is ~`m/B` of a full backward *in FLOPs* — but on CPU at small
`m` the Gen3 extractor is dispatch-bound, not FLOP-bound, so the per-call overhead does not shrink
with the row count and the FLOP argument does not survive contact with a clock.

**MEASURED** (CPU `--debug` fold, `--n-steps 512 --batch-size 128`, `m = 16`, the real 2501-dim
extractor, box carrying a live fleet; 2026-09-01):

| meter | grad_project | off_slice monitor-only |
|---|---|---|
| `distill/proj_ms` (per micro-batch) | **426 - 644 ms** | — |
| `train/train_ms` (per `train()`) | **12.9 - 19.1 s** | **4.2 - 5.9 s** |

i.e. the projection is roughly **55-70% of `train()`** and ~2.5-3x the step, in that configuration.
The share should FALL as `--batch-size` rises (the numerator is per-row work, the denominator is
per-batch) and on a GPU, **but that has not been measured** — read `distill/proj_ms` against
`train/train_ms` on your own arm before assuming it. Peak extra memory is `(m + 2) x |theta|`
floats, freed at the end of the micro-batch.

⚠️ **AND `proj_removed_frac` CAME OUT AT 0.75 - 0.89, WHICH IS THE FINDING, NOT A BUG.** A random
vector's projection onto a random 16-dim subspace of a ~2M-dim space would keep ~1e-5 of its energy.
Removing ~80% means the distill gradient and the off-slice behaviour gradients share their dominant
directions almost entirely — which is the "shared weights carry the taught content onto untaught
boards" mechanism, seen directly at the update. It also states this method's ceiling plainly: where
a direction BOTH teaches and leaks, a first-order projection cannot keep the teaching, and at
`m = 16` most of the teacher term's magnitude goes with the leak (`distill/kl` stayed HIGHER in the
projected arm of that smoke — less absorbed, exactly as that reading predicts). `proj_rank` came out
at ~15.4 of 16, so the sampled directions are near-independent and lowering `m` trades removal for
coverage rather than de-duplicating anything.

**FAILURE IS SELF-DISABLING**, the `PerTermNoiseSampler` convention: anything raised inside the
projection retires it for the rest of the `train()` call with one printed line and leaves `.grad`
untouched. A regulariser must never take a run down — but note the consequence honestly: a retired
projector trains as a plain unprojected fold, which is why it prints.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import torch as th

from agents.training.instrumented_ppo.noise_scale_terms import term_gradient

#: `--distill-anchor-mode grad_project`. Kept here rather than in `distill_anchor.py` so the mode
#: name and the machinery it selects live together; `ANCHOR_MODES` imports it back.
GRAD_PROJECT_MODE = "grad_project"

#: Default `m` — how many off-slice rows constrain each step (`--distill-anchor-proj-samples`).
DEFAULT_PROJ_SAMPLES = 16

#: Modified-Gram-Schmidt drop threshold, RELATIVE to the incoming vector's own norm. A direction
#: whose residual falls below this after two orthogonalisation passes is treated as already spanned.
#: 1e-4 rather than machine epsilon on purpose: the vectors are `|theta|`-dimensional float32, where
#: cancellation error in a single MGS pass is O(eps*sqrt(|theta|)) ~ 1e-4 at |theta| ~ 2e6 — a
#: tighter threshold would keep numerical noise as a "constraint" and project the gradient onto it.
GS_REL_TOL = 1e-4


def flatten_grads(grads: Sequence[Optional[th.Tensor]],
                  params: Sequence[th.nn.Parameter]) -> th.Tensor:
    """One flat float vector over `params`, with `allow_unused`'s `None`s read as zeros."""
    return th.cat([(th.zeros_like(p) if g is None else g).reshape(-1)
                   for g, p in zip(grads, params)])


def orthonormalize(vectors: Sequence[th.Tensor], tol: float = GS_REL_TOL) -> List[th.Tensor]:
    """MODIFIED Gram-Schmidt with ONE re-orthogonalisation pass -> an orthonormal basis.

    Modified (subtract each accepted direction in sequence off the RUNNING residual) rather than
    classical (subtract all projections of the ORIGINAL vector at once), because the classical form
    loses orthogonality catastrophically on nearly-dependent inputs — which is the normal case here:
    two off-slice states in one minibatch often want the same behavioural change.

    The second pass is the standard "twice is enough" (Giraud et al. 2005): after one pass the
    residual's orthogonality error is O(eps*kappa), after two it is O(eps). Without it a duplicated
    constraint survives with a relative residual ~1e-4 in float32 at production `|theta|` and is
    kept as a spurious extra dimension. Vectors whose residual falls below `tol * ||v||` are
    DROPPED — the returned list is therefore shorter than the input whenever the sampled states
    agree.
    """
    basis: List[th.Tensor] = []
    for v in vectors:
        n0 = float(v.norm())
        if n0 <= 0.0:
            continue
        r = v.to(th.float32)
        for _ in range(2):                       # twice is enough
            for q in basis:
                r = r - q * th.dot(q, r)
        nr = float(r.norm())
        if nr <= tol * n0:
            continue                             # already spanned — a genuine duplicate lands here
        basis.append(r / nr)
    return basis


def project_out(g: th.Tensor, basis: Sequence[th.Tensor]) -> th.Tensor:
    """The component of `g` INSIDE `span(basis)` — i.e. `g - P_perp g`, the part to be REMOVED.

    Returned as the removal rather than the survivor because that is what the caller subtracts from
    an already-accumulated `.grad`: `.grad` holds `g_ppo + g_distill`, and subtracting this makes it
    `g_ppo + P_perp g_distill` without ever touching `g_ppo`.
    """
    removed = th.zeros_like(g)
    for q in basis:
        removed = removed + q * th.dot(q, g)
    return removed


def offslice_rows(distill_mask: th.Tensor, m: int,
                  generator: th.Generator) -> Optional[th.Tensor]:
    """Up to `m` row indices where `distill_mask == 0` (no teacher pins this state's team).

    Drawn WITHOUT replacement from a projector-owned `th.Generator`, never the global RNG: this
    feature must not perturb the stream that the rollout permutation and every seeded-arm comparison
    read. `None` when the minibatch holds no off-slice row (nothing to constrain).
    """
    off = (distill_mask.reshape(-1) < 0.5).nonzero(as_tuple=False).reshape(-1)
    n = int(off.numel())
    if n == 0:
        return None
    if n <= m:
        return off
    pick = th.randperm(n, generator=generator)[:m].to(off.device)
    return off[pick]


def behaviour_constraints(policy, observations: Dict[str, th.Tensor],
                          action_masks: th.Tensor, rows: th.Tensor,
                          params: Sequence[th.nn.Parameter]) -> List[th.Tensor]:
    """`grad_theta log pi(a* | s)` for each sampled off-slice state `s`, as flat vectors.

    ONE forward over the SLICED observations (the `m` rows only, not the whole minibatch), then one
    `autograd.grad` per row through that small graph — so the whole battery costs about `m/B` of a
    full-batch backward, which is what makes a per-step projection affordable at all.

    `a*` is the student's own argmax over the LEGAL set. It is the single direction whose movement
    IS the behaviour change on that board (the ledger's untaught meters are argmax-play meters), and
    it needs no label, no reference forward and no second policy.
    """
    obs = {k: v[rows] for k, v in observations.items()}
    logits = policy.get_distribution(obs).distribution.logits
    neg = (action_masks[rows].to(logits.dtype) - 1.0) * 1e9
    logp = th.log_softmax(logits + neg, dim=-1)
    star = logp.detach().argmax(-1)
    out: List[th.Tensor] = []
    n = int(rows.numel())
    for j in range(n):
        grads = th.autograd.grad(logp[j, star[j]], params,
                                 retain_graph=(j < n - 1), allow_unused=True)
        out.append(flatten_grads(grads, params))
    return out


class NullGradProjector:
    """The OFF path — `add` hands the term straight back and every hook is a no-op.

    A singleton rather than a `None` check at four call sites, so `ppo.py`'s fold still reads as one
    straight line whether or not anything is being projected (the `NULL_TAGGER` convention).
    """

    active = False
    failed = False

    def add(self, term):            # noqa: D102 - passthrough
        return term

    def before_backward(self, policy, rollout_data) -> None:   # noqa: D102 - no-op
        return None

    def after_backward(self, accum: int) -> None:              # noqa: D102 - no-op
        return None


NULL_PROJECTOR = NullGradProjector()


class DistillGradProjector:
    """Projects each micro-batch's DISTILL gradient off the off-slice behaviour subspace.

    Lifecycle, per micro-batch:
      * `add(term)` during the distill fold — records the tensor, RETURNS IT UNCHANGED.
      * `before_backward(policy, rollout_data)` immediately before `(loss/accum).backward()` —
        takes `g_distill` with `autograd.grad` (which writes no `.grad`), builds the constraint
        basis, and stashes the removal vector.
      * `after_backward(accum)` immediately after — subtracts `removal/accum` from `.grad`, matching
        the `1/accum` scaling the real backward applied, then drops every buffer.

    Between those two calls nothing is mutated, so an exception anywhere retires the projector with
    `.grad` in exactly the state the unprojected fold would have left it.
    """

    active = True

    def __init__(self, params: Sequence[th.nn.Parameter], metrics_out: Dict[str, List[float]],
                 *, samples: int = DEFAULT_PROJ_SAMPLES, tol: float = GS_REL_TOL,
                 seed: int = 0) -> None:
        self._params: List[th.nn.Parameter] = [p for p in params if p.requires_grad]
        self._metrics = metrics_out
        self._m = max(1, int(samples))
        self._tol = float(tol)
        self._gen = th.Generator().manual_seed(int(seed))
        self._pending: List[th.Tensor] = []
        self._removal: Optional[th.Tensor] = None
        self.failed = False

    # ------------------------------------------------------------------ the fold-side seam
    def add(self, term):
        """Tag `term` as DISTILL-sourced and RETURN IT UNCHANGED (the loss must not move)."""
        if isinstance(term, th.Tensor) and term.requires_grad:
            self._pending.append(term)
        return term

    # ------------------------------------------------------------------ the step-side seam
    def before_backward(self, policy, rollout_data) -> None:
        """Compute the removal vector. Call BEFORE `(loss/accum).backward()` — graph must be alive."""
        pending, self._pending = self._pending, []
        self._removal = None
        if self.failed or not pending or not self._params:
            return
        t0 = time.perf_counter()
        try:
            g = flatten_grads(term_gradient(pending, self._params), self._params)
            g_sq = float(g.pow(2).sum())
            dmask = rollout_data.observations.get("distill_mask")
            rows = None if dmask is None else offslice_rows(dmask, self._m, self._gen)
            basis: List[th.Tensor] = []
            removed_sq = 0.0
            if rows is not None and g_sq > 0.0:
                basis = orthonormalize(
                    behaviour_constraints(policy, rollout_data.observations,
                                          rollout_data.action_masks, rows, self._params),
                    self._tol)
                removal = project_out(g, basis)
                removed_sq = float(removal.pow(2).sum())
                if removed_sq > 0.0:
                    self._removal = removal
            self._metrics.setdefault("proj_rank", []).append(float(len(basis)))
            self._metrics.setdefault("proj_removed_frac", []).append(
                removed_sq / g_sq if g_sq > 0.0 else 0.0)
            self._metrics.setdefault("proj_constraint_rows", []).append(
                0.0 if rows is None else float(rows.numel()))
        except Exception as exc:                              # pragma: no cover - defensive
            self._fail(exc)
        finally:
            self._metrics.setdefault("proj_ms", []).append(
                1000.0 * (time.perf_counter() - t0))

    def after_backward(self, accum: int) -> None:
        """Apply the removal to the accumulated `.grad`. Call AFTER `(loss/accum).backward()`.

        `.grad` holds `(g_ppo + g_distill)/accum` for this micro-batch on top of whatever earlier
        micro-batches of the group put there; subtracting `removal/accum` leaves
        `(g_ppo + P_perp g_distill)/accum` — PPO's contribution bit-for-bit as the backward made it.

        A parameter with a `None` `.grad` is MATERIALISED when the removal touches it: `P_perp g` is
        a genuine vector in the constraint span, which can have components on parameters `g_distill`
        itself never reached, and silently dropping them would apply a different operator than the
        one this module documents.
        """
        removal, self._removal = self._removal, None
        if self.failed or removal is None:
            return
        try:
            scale = 1.0 / float(max(1, int(accum)))
            off = 0
            with th.no_grad():
                for p in self._params:
                    n = p.numel()
                    chunk = removal[off:off + n].view_as(p)
                    off += n
                    if p.grad is None:
                        if float(chunk.abs().max()) == 0.0:
                            continue
                        p.grad = th.zeros_like(p)
                    p.grad.sub_(chunk, alpha=scale)
        except Exception as exc:                              # pragma: no cover - defensive
            self._fail(exc)

    # ------------------------------------------------------------------ failure
    def _fail(self, exc: BaseException) -> None:
        self.failed = True
        self._pending = []
        self._removal = None
        print("⚠️ [DISTILL-PROJ] gradient projection DISABLED for the rest of this train() "
              f"({type(exc).__name__}: {exc}). The fold continues UNPROJECTED — i.e. as an "
              "ordinary off-slice fold with no source separation at all.", flush=True)


def make_projector(model, metrics_out: Dict[str, List[float]],
                   params: Sequence[th.nn.Parameter]):
    """`DistillGradProjector` under `--distill-anchor-mode grad_project`, else `NULL_PROJECTOR`.

    The ONE construction site `ppo.py` calls. Off => the null singleton, whose `add` is a two-
    instruction passthrough and whose hooks do nothing, so a run without the mode pays nothing and
    its update is bit-identical.
    """
    if str(getattr(model, "distill_anchor_mode", "off_slice")) != GRAD_PROJECT_MODE:
        return NULL_PROJECTOR
    return DistillGradProjector(
        params, metrics_out,
        samples=int(getattr(model, "distill_anchor_proj_samples", DEFAULT_PROJ_SAMPLES) or
                    DEFAULT_PROJ_SAMPLES),
        seed=int(getattr(model, "seed", 0) or 0))
