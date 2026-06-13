"""Gradient-balance + value-scale diagnostics for the shared-trunk PPO.

The dual-head feature extractor shares ONE transformer trunk between the policy and value
heads (see ``src/agents/model/CLAUDE.md``). Both losses' gradients flow into that shared
trunk and *compete*: if the value loss dominates (large / unclipped, big-return scale) it
swamps the trunk and the policy barely updates. These pure helpers measure that competition
**directly**, so reducing ``vf_coef`` or adding return normalization (PopArt) can be tuned to
a number instead of inferred indirectly from ``approx_kl`` / ``clip_fraction``.

Two probes, both cheap and both run once per ``train()`` call:

* :func:`grad_balance_metrics` — the value-vs-policy *pull* on the shared trunk (the pressure
  gauge). ``value_share`` ~0.5 = balanced, →1 = value swamps the trunk; ``value_policy_logratio``
  is the same imbalance on a linear, non-saturating log scale (0 = balanced, >0 = value dominates)
  so a fix landing is *visible* where ``value_share`` sits pinned near 1; ``policy_value_cosine``
  <0 = the heads drag the trunk in opposing directions (structural conflict, ``vf_coef``-free).
* :func:`value_scale_metrics` — the return / value-prediction *scale* (PopArt prep). These are
  exactly the ``(μ, σ)`` an adaptive return normalizer tracks; watch them to SEE the value scale
  drift (reward annealing / policy improvement) that a static ``vf_coef`` can't follow.

All functions are pure (no SB3, no logging) so they unit-test without a training loop;
``InstrumentedMaskablePPO.train()`` calls them and records the results via the standard logger
(→ TensorBoard + launcher TUI).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch as th
from torch import nn

# Phase submodules of Gen3FeaturesExtractor whose parameters are TRULY SHARED — both the
# policy-loss and value-loss gradients write the *same* tensors. This is the contested
# representation the pressure metric is about.
#
# Deliberately EXCLUDED (head-private — counting them would dilute the balance with
# non-competing gradient components):
#   - ``cls_pool``: holds head-private query/attention params (``our_cls``/``their_cls`` feed
#     only the policy; ``value_cls`` feeds only the value head). The value gradient still flows
#     *through* ``value_cls_attn`` to reach ``team_transformer`` (which IS shared and IS
#     measured); we just don't count the head-private query params themselves.
#   - ``pre_proj_norm``/``projection`` (policy head) + ``value_pre_norm``/``value_projection``
#     (value head): the dedicated readouts, by definition not shared.
# (``unpack`` is stateless — no params — so its absence here is a no-op.)
# Single source of truth: if the extractor's phase attribute names change, update this tuple.
SHARED_TRUNK_PHASES = ("embeddings", "pokemon_encoder", "team_transformer", "assembler")


def shared_trunk_parameters(features_extractor: nn.Module) -> List[nn.Parameter]:
    """The shared-trunk parameters of the dual-head extractor (the contested representation).

    Selected by the :data:`SHARED_TRUNK_PHASES` allow-list over ``named_parameters()`` so a
    renamed or newly-added head module can never silently leak into the "shared" set. Returns
    ``[]`` for a non-Gen3 extractor (e.g. a unit-test stub or a different policy), which the
    caller treats as "skip the gradient probe".
    """
    prefixes = tuple(p + "." for p in SHARED_TRUNK_PHASES)
    return [
        param
        for name, param in features_extractor.named_parameters()
        if name.startswith(prefixes) and param.requires_grad
    ]


def _flat_grads(loss: th.Tensor, params: Sequence[nn.Parameter]) -> th.Tensor:
    """Flattened gradient of ``loss`` w.r.t. ``params`` (zeros for params it doesn't reach).

    Uses ``autograd.grad(retain_graph=True)`` so it is a **read-only** probe: it does NOT
    populate ``.grad`` and leaves the graph intact for the real ``loss.backward()`` that
    follows in the training step. ``allow_unused=True`` tolerates a param outside this loss's
    subgraph (substituted with zeros) rather than raising.
    """
    if not params:
        return th.zeros(1)
    grads = th.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return th.cat([
        (g if g is not None else th.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, params)
    ])


def grad_balance_metrics(
    policy_term: th.Tensor,
    value_term: th.Tensor,
    shared_params: Sequence[nn.Parameter],
    aux_term: "th.Tensor | None" = None,
) -> Dict[str, float]:
    """Value-vs-policy gradient competition on the shared trunk.

    ``policy_term`` / ``value_term`` are the **weighted** loss contributions exactly as they
    enter the combined loss — ``policy_term = policy_loss + ent_coef*entropy_loss`` and
    ``value_term = vf_coef*value_loss`` — so the reported norms are the *actual* pull each head
    exerts on the trunk at the current coefficients. ``value_share`` therefore scales with
    ``vf_coef`` (that is what makes it a tuning target: dial ``vf_coef`` so ``value_share`` sits
    near ~0.5). ``value_policy_logratio`` = ``log10(‖g_value‖/‖g_policy‖)`` is the same imbalance
    in a *linear*, non-saturating form (0.0 = balanced, >0 = value dominates) — the legible gauge
    for watching a fix land where ``value_share`` is pinned near 1. Cosine is scale-invariant, so
    it is the ``vf_coef``-independent structural-conflict signal (<0 ⟹ the heads pull the trunk in
    opposing directions).

    Returns ``{grad/value_share, grad/value_policy_logratio, grad/policy_value_cosine,
    grad/policy_norm_shared, grad/value_norm_shared}``. **Must be called while the graph is alive**
    (before ``loss.backward()``).
    """
    g_pi = _flat_grads(policy_term, shared_params)
    g_vf = _flat_grads(value_term, shared_params)
    n_pi = float(g_pi.norm())
    n_vf = float(g_vf.norm())
    cosine = float((g_pi @ g_vf).item() / (n_pi * n_vf)) if n_pi > 0.0 and n_vf > 0.0 else 0.0
    total = n_pi + n_vf
    # log10 of the value/policy pull ratio. ``value_share`` saturates near 1 when the value head
    # dominates (0.985 vs 0.99 vs 0.995 all look alike but are 66× / 99× / 199×), so it barely
    # moves while a fix lands; the log-ratio is *linear* in the imbalance (now ≈ +1.8 at ~66:1,
    # 0.0 = balanced, <0 = policy dominates), making it the legible gauge for watching PopArt /
    # a vf_coef change pull the trunk back toward balance. Guarded 0.0 when either norm is zero
    # (a unit-test detached-graph artifact; in real training both are strictly positive) — same
    # convention as ``cosine``; note 0.0 here doubles as the "balanced" value, which is fine since
    # the no-signal case never occurs live.
    logratio = float(np.log10(n_vf / n_pi)) if n_pi > 0.0 and n_vf > 0.0 else 0.0
    out = {
        "grad/policy_norm_shared": n_pi,
        "grad/value_norm_shared": n_vf,
        "grad/value_share": (n_vf / total) if total > 0.0 else 0.0,
        "grad/value_policy_logratio": logratio,
        "grad/policy_value_cosine": cosine,
    }
    # Belief-aux pull on the SAME shared trunk (the in-place belief tokens flow through the team
    # transformer, so the aux writes the same params). ``belief_share`` = ‖g_aux‖ / (‖g_pi‖+‖g_vf‖+
    # ‖g_aux‖) is the principled "is the aux DOMINATING the trunk" signal (watch it sit ~5-15%; a
    # spike toward ~1 with a degrading policy = the aux is fighting the actor → lower
    # ``opp_belief_aux_coef``). ``belief_policy_cosine`` <0 ⟹ the aux pulls the trunk against the
    # policy. ``aux_term`` is the WEIGHTED contribution (opp_belief_aux_coef·aux) exactly as it enters
    # the loss, so the share scales with the coef — that is what makes it the tuning target.
    if aux_term is not None:
        g_aux = _flat_grads(aux_term, shared_params)
        n_aux = float(g_aux.norm())
        total3 = n_pi + n_vf + n_aux
        out["grad/belief_norm_shared"] = n_aux
        out["grad/belief_share"] = (n_aux / total3) if total3 > 0.0 else 0.0
        out["grad/belief_policy_cosine"] = (
            float((g_aux @ g_pi).item() / (n_aux * n_pi)) if n_aux > 0.0 and n_pi > 0.0 else 0.0
        )
    return out


def value_scale_metrics(returns, values) -> Dict[str, float]:
    """Return- and value-prediction scale stats — the inputs PopArt's ART half would track.

    ``return_mean`` / ``return_std`` are exactly the ``(μ, σ)`` an adaptive return normalizer
    estimates; ``return_abs_max`` shows the tail magnitude; ``value_pred_std`` is the value
    head's actual output spread (does it really span ±tens?). Watch these to SEE the non-
    stationary value scale drift as the reward is annealed / the policy improves — the signal a
    static ``vf_coef`` can't track and PopArt would.

    Pure NumPy; ``returns`` / ``values`` are the full rollout buffer's arrays (any shape — they
    are flattened). Returns an empty dict for empty input rather than emitting NaNs.
    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    out: Dict[str, float] = {}
    if r.size:
        out["train/return_mean"] = float(r.mean())
        out["train/return_std"] = float(r.std())
        out["train/return_abs_max"] = float(np.abs(r).max())
    if v.size:
        out["train/value_pred_std"] = float(v.std())
    return out
