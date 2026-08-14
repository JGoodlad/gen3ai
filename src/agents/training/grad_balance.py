"""Gradient-balance + value-scale diagnostics for the shared-trunk PPO.

The dual-head feature extractor shares ONE transformer trunk between the policy and value
heads (see ``src/agents/model/CLAUDE.md``). Both losses' gradients flow into that shared
trunk and *compete* — and so does EVERY auxiliary head whose gradient reaches the trunk
(species/move/latent/move-latent belief, the win-prob head under ``shaping``, the
distributional value head under ``shaping``). With more than two competitors the question is
no longer "value vs policy" but "is ANY term crowding out the rest", so the probe measures
each term's pull on **one common denominator** — a pie that sums to ~1 and is directly
comparable across terms. These pure helpers measure that competition **directly**, so
reducing ``vf_coef`` / an aux coef or adding return normalization (PopArt) can be tuned to a
number instead of inferred indirectly from ``approx_kl`` / ``clip_fraction``.

Two probes, both cheap and both run once per ``train()`` call:

* :func:`grad_balance_metrics` — the per-term *pull* on the shared trunk (the pressure gauge).
  Every ``grad/<term>_share`` is that term's gradient norm over the SAME total
  (``policy + value + Σ aux``), so ``grad/policy_share`` + ``grad/value_share`` +
  ``grad/aux_share`` ≈ 1 and a term swamping the trunk is visible as its share climbing.
  ``grad/value_policy_logratio`` is the *aux-independent* value-vs-policy imbalance (a pure
  ratio of the two RL norms — the PopArt / ``vf_coef`` tuning gauge); ``grad/<term>_policy_cosine``
  <0 = that term drags the trunk against the policy (structural conflict, coef-free).
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
    aux_terms: "Dict[str, th.Tensor] | None" = None,
) -> Dict[str, float]:
    """Per-term gradient competition on the shared trunk, on ONE common denominator.

    ``policy_term`` / ``value_term`` are the **weighted** loss contributions exactly as they
    enter the combined loss — ``policy_term = policy_loss + ent_coef*entropy_loss`` and
    ``value_term = vf_coef*value_loss``. ``aux_terms`` maps a short name → that auxiliary's
    **weighted** contribution (``coef * aux_loss``) exactly as it entered the loss, e.g.
    ``{"species_belief": …, "move_belief": …, "move_latent": …, "win_prob": …,
    "value_dist": …}`` — pass only the terms that are ACTIVE this minibatch (an empty / ``None``
    dict means "RL heads only", the upstream-identical 2-way case).

    Every ``grad/<term>_share`` is that term's shared-trunk gradient norm divided by the SAME
    total ``T = ‖g_pi‖ + ‖g_vf‖ + Σ‖g_aux‖`` — so the shares are mutually comparable, sum to
    ~1.0 (``policy_share + value_share + aux_share``), and a term crowding out the rest is read
    off directly. (The shares are an **L1-of-norms** proxy — an upper bound, not a variance
    decomposition, since ``‖a+b‖ ≠ ‖a‖+‖b‖`` — but the SAME convention for every term, which is
    the comparability that matters.) The norms scale with their coefficients, which is what makes
    each share a tuning target: dial a coef so its share sits where you want it.

    Returns the always-present block
    ``{grad/policy_share, grad/value_share, grad/policy_norm_shared, grad/value_norm_shared,
    grad/value_policy_logratio, grad/policy_value_cosine}`` — where ``value_policy_logratio`` =
    ``log10(‖g_value‖/‖g_policy‖)`` is the **aux-independent** value-vs-policy imbalance (0 =
    balanced, >0 = value dominates, <0 = policy dominates: the legible PopArt / ``vf_coef`` knob,
    unaffected by how many auxiliaries are on) and ``policy_value_cosine`` <0 = the two RL heads
    drag the trunk in opposing directions. For each ``aux_terms`` entry it adds
    ``grad/<name>_share``, ``grad/<name>_norm_shared`` and ``grad/<name>_policy_cosine`` (<0 = that
    aux fights the policy), plus a single ``grad/aux_share`` = Σ aux shares (the total non-RL draw —
    one number for "are the scaffolds collectively crowding the RL heads"). **Must be called while
    the graph is alive** (before ``loss.backward()``).
    """
    g_pi = _flat_grads(policy_term, shared_params)
    g_vf = _flat_grads(value_term, shared_params)
    n_pi = float(g_pi.norm())
    n_vf = float(g_vf.norm())

    # Per-aux gradient + norm — each is a separately-reported scaffold pulling the SAME trunk.
    aux_g: Dict[str, th.Tensor] = {}
    aux_n: Dict[str, float] = {}
    for name, term in (aux_terms or {}).items():
        g = _flat_grads(term, shared_params)
        aux_g[name] = g
        aux_n[name] = float(g.norm())

    # ONE common denominator = the FULL trunk pull (policy + value + every reported scaffold), so
    # every `*_share` is on the same scale, they sum to ~1, and any term crowding out the rest is
    # directly visible. Guarded 0.0 when the total is zero (a unit-test all-detached artifact; in
    # real training n_pi/n_vf are strictly positive).
    total = n_pi + n_vf + sum(aux_n.values())

    def _share(n: float) -> float:
        return (n / total) if total > 0.0 else 0.0

    def _cos_vs_policy(g: th.Tensor, n: float) -> float:
        # Guarded 0.0 when either norm is zero (a detached-graph artifact; live both are >0).
        return float((g @ g_pi).item() / (n * n_pi)) if n > 0.0 and n_pi > 0.0 else 0.0

    out: Dict[str, float] = {
        "grad/policy_norm_shared": n_pi,
        "grad/value_norm_shared": n_vf,
        "grad/policy_share": _share(n_pi),
        "grad/value_share": _share(n_vf),
        # log10 of the value/policy pull RATIO — AUX-INDEPENDENT (a pure ratio of the two RL norms,
        # unchanged by how many auxiliaries are reported), linear & non-saturating (0 = balanced,
        # >0 = value dominates, <0 = policy dominates). The legible gauge for watching PopArt / a
        # vf_coef change pull the value/policy balance back — `value_share` now moves with the aux
        # count (it is value's slice of the WHOLE pie), so the ratio is the cleaner balance signal.
        "grad/value_policy_logratio": (
            float(np.log10(n_vf / n_pi)) if n_pi > 0.0 and n_vf > 0.0 else 0.0
        ),
        # Scale-invariant (hence vf_coef-independent) structural-conflict signal between the two RL
        # heads (<0 ⟹ policy and value pull the trunk in opposing directions). Guarded 0.0 on a
        # zero norm (unit-test detached artifact).
        "grad/policy_value_cosine": (
            float((g_pi @ g_vf).item() / (n_pi * n_vf)) if n_pi > 0.0 and n_vf > 0.0 else 0.0
        ),
    }
    for name, n in aux_n.items():
        out[f"grad/{name}_norm_shared"] = n
        out[f"grad/{name}_share"] = _share(n)
        out[f"grad/{name}_policy_cosine"] = _cos_vs_policy(aux_g[name], n)
    if aux_n:
        # Total non-RL scaffold draw on the trunk (Σ aux shares) — the rollup of every aux term, so
        # one curve answers "are the auxiliaries collectively crowding out policy/value".
        out["grad/aux_share"] = _share(sum(aux_n.values()))
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
