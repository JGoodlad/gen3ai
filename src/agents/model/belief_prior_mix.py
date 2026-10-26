"""THE BELIEF PRIOR MIX — an INFERENCE-ONLY anchor of the opponent belief to its Smogon prior (lever 1 (a) of
``designs/model/generalisation_review_2026-10-26.md``; D0f at λ = 1).

Each of the three LEARNED belief legs is a Smogon prior plus a learned delta, and on an unfamiliar opponent the
delta is confidently wrong (the review's P5). This module replaces each leg's distribution ``b`` with

    b_λ = (1 − λ) · b + λ · prior

BEFORE anything downstream reads it, so the hypothesis tokens, the damage operator, the reinjected tokens, the
policy and the critic all act on the mixed belief:

* **species** (X5's hypothesis set, ``HypothesisBuilder.species_set``): the fixed-size presences π (Σ = k over the
  candidates) are mixed with the presences of the prior alone (the delta zeroed), so they still sum to k; the
  hypothesis order, the hidden slots' concrete species, OTHER and every π consumer follow from the mixed π;
* **moves** (``MoveBelief.move_logits``): each move's Bernoulli marginal σ(prior ⊕ δ) is mixed with σ(prior), in
  log space, and returned as a logit; revealed moves are pinned after the mix, exactly as before;
* **items** (``ItemBelief.forward``): the categorical posterior softmax(log prior + δ) is mixed with
  softmax(log prior) and returned as log-probabilities.

These are the three deltas D0f zeroes (``design_diversity_teams_opponents.md`` §3.3). The spread and Hidden Power
type heads are NOT mixed (not in D0f's definition; the review measured neither).

EXACT ENDPOINTS, by construction rather than by float luck: λ = 0 runs the unchanged code path (bit-identical to a
model that never heard of this module), and λ = 1 returns the prior-only value itself (bit-identical to the deltas
zeroed in the weights, D0f's surgery). ``belief_prior_mix_test.py`` pins both on a real ``MaskablePPO`` policy.

INFERENCE ONLY. λ is a plain attribute on the three modules (never a ``state_dict`` entry, never a config field):
:func:`set_belief_prior_mix` is a MODEL-LOAD option (``main.anchors --our-belief-prior-mix``; ``main.reads`` model
``load``), and a forward that builds an autograd graph with λ > 0 RAISES (:func:`refuse_with_grad`): the deltas were
trained on the unmixed belief, and a learner step on the mixed one is a different experiment nobody registered.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, List, Tuple

import torch
import torch.nn.functional as F

#: The attribute each mixed module carries (class default 0.0 = off).
ATTR = "belief_prior_mix"


class BeliefPriorMixError(RuntimeError):
    """The mix cannot be applied as asked (a missing leg, λ out of range, or a graph-building forward)."""


def check_lambda(lam: Any) -> float:
    """λ as a float in [0, 1]; anything else is a :class:`BeliefPriorMixError` (NaN included)."""
    try:
        x = float(lam)
    except (TypeError, ValueError):
        raise BeliefPriorMixError(f"belief prior mix λ {lam!r} is not a number") from None
    if not (0.0 <= x <= 1.0):                         # NaN fails both comparisons
        raise BeliefPriorMixError(f"belief prior mix λ = {x!r} is outside [0, 1]")
    return x


def refuse_with_grad(lam: float, where: str) -> None:
    """λ > 0 in a forward that records autograd: refused (module docstring, INFERENCE ONLY)."""
    if lam > 0.0 and torch.is_grad_enabled():
        raise BeliefPriorMixError(
            f"{where}: belief_prior_mix = {lam:g} in a forward that builds a gradient. The mix is an "
            "INFERENCE-ONLY model-load option (torch.no_grad / inference_mode); it never trains.")


def _logs(lam: float) -> Tuple[float, float]:
    return math.log1p(-lam), math.log(lam)


def mix_bernoulli_logits(post: torch.Tensor, prior: torch.Tensor, lam: float) -> torch.Tensor:
    """logit((1 − λ)·σ(post) + λ·σ(prior)), elementwise, computed in log space (stable at |logit| ≫ 1).
    ``0 < λ < 1`` (the endpoints are the callers' exact branches)."""
    a, b = _logs(lam)
    p32, q32 = post.float(), prior.float()
    log_p = torch.logaddexp(a + F.logsigmoid(p32), b + F.logsigmoid(q32))
    log_not_p = torch.logaddexp(a + F.logsigmoid(-p32), b + F.logsigmoid(-q32))
    return (log_p - log_not_p).to(post.dtype)


def mix_categorical_logits(post: torch.Tensor, prior: torch.Tensor, lam: float) -> torch.Tensor:
    """log((1 − λ)·softmax(post) + λ·softmax(prior)) over the last axis: normalised log-probabilities, so their
    softmax IS the mixture. ``0 < λ < 1``."""
    a, b = _logs(lam)
    out = torch.logaddexp(a + torch.log_softmax(post.float(), dim=-1), b + torch.log_softmax(prior.float(), dim=-1))
    return out.to(post.dtype)


def mix_presence(post: Any, prior: Any, lam: float) -> Any:
    """The species presence mixed: π = (1 − λ)·π_post + λ·π_prior on the live candidates (both sum to k there, so the
    mixture does too) and log π = log of that; every structural field (V, k, n, live, full) is the two presences'
    common one, and the full / off-V / masked rows keep their structural values. ``logits`` stays the posterior's
    (the training BCE's graph field, read by no policy or critic consumer). ``0 < λ < 1``."""
    import dataclasses

    live_c = post.cand & post.live.unsqueeze(-1)
    a, b = _logs(lam)
    pi = torch.where(live_c, (1.0 - lam) * post.pi + lam * prior.pi, post.pi)
    log_pi = torch.where(live_c, torch.logaddexp(a + post.log_pi, b + prior.log_pi), post.log_pi)
    return dataclasses.replace(post, pi=pi, log_pi=log_pi)


def mixed_modules(extractor: Any) -> List[Tuple[str, Any]]:
    """``[(name, module)]``: the three legs the mix reaches on ``extractor``; a missing one is refused (a model
    without X5's hypothesis builder, a fused move prior or the item belief cannot be anchored as registered)."""
    out = []
    for name in ("hypothesis_builder", "move_belief", "item_belief_head"):
        m = getattr(extractor, name, None)
        if m is None:
            raise BeliefPriorMixError(f"the extractor has no {name}: the belief prior mix needs all three legs "
                                      "(X5 hypothesis set, move belief with prior fusion, item belief)")
        out.append((name, m))
    if not getattr(extractor.move_belief, "prior_fusion", False):
        raise BeliefPriorMixError("move_belief has no prior fusion: there is no Smogon prior to mix toward")
    return out


def _extractors(model: Any) -> Iterable[Any]:
    """The distinct features extractors of a model / policy / extractor (SB3 aliases them; one shared object)."""
    pol = getattr(model, "policy", model)
    cands = [getattr(pol, n, None) for n in ("features_extractor", "pi_features_extractor", "vf_features_extractor")]
    found = {id(m): m for m in cands if m is not None}
    if not found:
        found = {id(model): model}
    return found.values()


def set_belief_prior_mix(model: Any, lam: Any) -> float:
    """Set λ on every mixed leg of ``model`` (an SB3 model, its policy, or a features extractor) and read it back;
    returns the applied λ. λ = 0 restores the unchanged forward."""
    x = check_lambda(lam)
    for fe in _extractors(model):
        for _name, m in mixed_modules(fe):
            setattr(m, ATTR, x)
    got = belief_prior_mix_of(model)
    if got != x:
        raise BeliefPriorMixError(f"belief prior mix read back {got!r}, not the {x!r} set")
    return x


def belief_prior_mix_of(model: Any) -> float:
    """The λ ``model`` runs at (0.0 for a model with none set); legs that disagree are refused."""
    vals = set()
    for fe in _extractors(model):
        for name in ("hypothesis_builder", "move_belief", "item_belief_head"):
            m = getattr(fe, name, None)
            if m is not None:
                vals.add(float(getattr(m, ATTR, 0.0)))
    if len(vals) > 1:
        raise BeliefPriorMixError(f"the belief legs run at different λ {sorted(vals)}")
    return vals.pop() if vals else 0.0
