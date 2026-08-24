"""The probe's ONE runtime perf knob: a compiled extractor for the B=1 LIVE decision only.

**Two forward paths run in this process and they want opposite things.** The live decision — ours
and, in the mirror, the opponent's — is a single row, and a single-row CPU forward is dispatch-bound:
``torch.compile`` fuses it and wins ~5.5x. The SEARCH scores its arms in one wide batch instead, and
there ``torch.compile`` LOSES, badly.

Measured 2026-08-23 on the live ``ai_v9_29_rev1_0823`` checkpoint, eager vs compiled interleaved,
BLAS pinned to one thread (``predict_values``, ms per call):

===== ========= ========== =========
    B  eager ms  compiled   speedup
===== ========= ========== =========
    1     27.95       5.13     5.45x
    8     31.88      11.44     2.79x
   32     62.58      38.70     1.62x
   64    110.52     755.51     0.15x
  128    199.89     817.87     0.24x
  256    390.88     903.17     0.43x
===== ========= ========== =========

Two independent reasons the wide half is not merely "less good":

* **the compiled kernel is 2-7x SLOWER above B=32** — dynamo falls back to a dynamic-shape graph
  whose lowering does not vectorize this extractor's scatter;
* **every new batch size pays a fresh TRACE** — 105 s at B=8, 120 s at B=32, 78 s at B=64 before it
  stabilized. The search realizes a different arm count at nearly every decision (50-300 here), so a
  compiled batch path would re-trace approximately forever.

So this routes on batch size: B==1 to the compiled graph, everything else to the untouched eager
one. That is also why the knob is **OFF by default**. Compiling perturbs the forward at the 1e-6
level (measured max|Δ| 7.6e-06 at B=64), and an argmax over near-tied actions can flip on that —
which would change the CONTROL arm's own chosen moves and make tonight's rows incomparable with
last night's. The probe's rule is that an optimization which can change a choice is opt-in and
labelled, never a default.

It does NOT buy search WIDTH: the live decision happens before the per-decision deadline starts, so
none of its cost is inside the budget. What it buys is games per hour — and most of all on the
``base`` (policy-alone) cell, whose entire cost IS that forward. **Measured there: 60 mirror
battles in 250.1 s eager vs 143.6 s compiled — 1.74x END TO END, with the one-off ~40 s trace
already inside the compiled figure** (two interleaved repeats each way, ranges disjoint:
255.7/244.5 against 138.8/148.5). Marginal per battle it is ~2.5x. A search cell sees far less,
because there the live forward is a single-digit share of a wall the search dominates.

**Empirically it did not move an outcome, and that is still evidence rather than a guarantee.**
Same seed, same 20 mirror battles, base arm: **20/20 identical in result, turn count and decision
count** between eager and compiled. That is why the flag exists and why it is not the default —
the probe's rows have to stay comparable across relaunches, and "measured not to bite on 20
battles" is not the same claim as "cannot bite".
"""

from __future__ import annotations

from typing import Any, Optional

#: Batch size the compiled graph is specialized for. Everything else routes to eager.
COMPILED_BATCH = 1


def batch_of(obs: Any) -> Optional[int]:
    """The leading dim of a Dict-obs forward's ``observation``, or ``None`` if unreadable.

    Unreadable routes to EAGER, deliberately: eager is the reference implementation, so the
    fall-back of an unrecognized input can only ever be correct."""
    try:
        x = obs["observation"] if isinstance(obs, dict) else obs
        return int(x.shape[0])
    except Exception:                                    # noqa: BLE001
        return None


def compile_b1_extractor(model, *, enabled: bool = True, label: str = "search_dividend") -> bool:
    """``torch.compile`` the extractor for B=1 and leave every wider forward eager.

    Returns True if the compiled path is live. Never raises: ``maybe_compile_extractor`` already
    reverts and warns on a failed or too-slow compile, and this adds only the batch router on top.
    """
    if not enabled:
        return False
    fe = getattr(getattr(model, "policy", None), "features_extractor", None)
    if fe is None:
        return False

    from agents.model.compile_opponents import maybe_compile_extractor

    eager = fe.forward                                   # the untouched bound method
    if not maybe_compile_extractor(model, True, label=label, hide_cuda=True):
        return False
    compiled = fe.forward                                # the guarded compiled callable

    def routed(obs: Any) -> Any:
        return compiled(obs) if batch_of(obs) == COMPILED_BATCH else eager(obs)

    fe.forward = routed
    _warm_live_signature(model)
    return True


def _warm_live_signature(model) -> None:
    """Force the trace the FIRST real decision would otherwise pay for.

    ``maybe_compile_extractor`` warms with ``{"observation": zeros(1, D)}`` — ONE key. Every call
    this process makes carries ``action_mask`` too, and dynamo guards on a dict's KEY SET exactly as
    hard as on shape and dtype, so a graph that looks warm re-traces on the first live forward
    (measured 19.5 s on the counterfactual producer, 2026-08-23). Both live entry points are warmed:
    ``get_distribution`` (the policy's action) and ``predict_values`` (the critic's).
    """
    import torch as th

    try:
        space = model.observation_space
        obs = {"observation": th.zeros(1, int(space["observation"].shape[0]), dtype=th.float32),
               "action_mask": th.ones(1, int(space["action_mask"].shape[0]), dtype=th.float32)}
        with th.no_grad():
            model.policy.get_distribution(obs)
            model.policy.predict_values(obs)
    except Exception as exc:                             # noqa: BLE001
        print(f"[search_dividend] compiled-graph warm-up skipped ({type(exc).__name__}: "
              f"{str(exc)[:160]}) — the first live decision will trace instead", flush=True)
