"""Regression guard for the ONE op that Inductor could not codegen.

BACKGROUND. `--compile-extractor` used to set `torch._dynamo.config.suppress_errors = True`. That
looked like defensive hygiene but was actually working around a single failure: with
`--threat-unrevealed-outgoing` (v36 #2) on, the softmax over species logits lowered to a
`[B,6,n_species]` numerator plus a `[B,6,1]` denominator and the Inductor CPU scheduler asserted
(`AssertionError: buf<N>`) trying to fuse the division. Suppression converted that into a per-frame
eager fallback, so the production config compiled only PARTIALLY (3.6x instead of 6.5x) — and every
unrelated backend failure in the process became silent too.

`BeliefHead.species_posterior` fixes it by spelling the same math as `log_softmax(...).exp()`, which
lowers to a form Inductor accepts. `tmp/softmax_variant_probe.py` shows `.contiguous()`, `.clone()`,
a 2-D reshape and a hand-rolled `exp / sum` all still FAIL, so the working spelling is not obvious
and is very easy to "simplify" back into a broken one.

The fast tests pin the MATH. The compile tests pin the COMPILE and run BY DEFAULT — the whole point
is to catch "the model no longer compiles" at code time rather than as a silent ~6.5x slower opponent
forward in a live run. They cost ~20 s on a warm Inductor cache. Opt out only when you need a fast
loop:

    GEN3AI_SKIP_COMPILE_TESTS=1 pytest src/ -q
"""
import inspect
import os

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.features_extractor import BeliefHead, Gen3FeaturesExtractor
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

# Opt-OUT, not opt-in: a model that stops compiling is a ~6.5x regression that is invisible at
# runtime, so the default has to be "the test suite catches it".
_SKIP_COMPILE = os.environ.get("GEN3AI_SKIP_COMPILE_TESTS") == "1"
_skip_compile = pytest.mark.skipif(_SKIP_COMPILE, reason="GEN3AI_SKIP_COMPILE_TESTS=1")


def test_species_posterior_matches_plain_softmax():
    """The rewrite must be MATH-neutral: `exp(log_softmax(x)) == softmax(x)`."""
    torch.manual_seed(0)
    head = BeliefHead.__new__(BeliefHead)          # bypass __init__; we only exercise the spelling
    logits = torch.randn(3, 6, 400) * 7.0          # wide scale: catches a non-stable factoring
    head.species_logits = lambda tokens: logits    # type: ignore[assignment]
    got = BeliefHead.species_posterior(head, torch.zeros(3, 6, 8))
    want = torch.softmax(logits, dim=-1)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()
    assert torch.allclose(got.sum(-1), torch.ones(3, 6), atol=1e-5)


def test_species_posterior_is_stable_for_large_logits():
    """`log_softmax().exp()` keeps softmax's max-subtraction, so a naive `exp/sum` blowup must not
    reappear if someone re-spells this."""
    head = BeliefHead.__new__(BeliefHead)
    logits = torch.tensor([[[1e4, 1e4 + 1.0, -1e4]]])
    head.species_logits = lambda tokens: logits    # type: ignore[assignment]
    got = BeliefHead.species_posterior(head, torch.zeros(1, 1, 8))
    assert torch.isfinite(got).all()
    assert pytest.approx(1.0, abs=1e-5) == float(got.sum())


# The LITERAL production arch (the ai_v8_03 shape). `threat_unrevealed_outgoing` is the flag that
# used to crash Inductor, but it only crashes IN THIS COMPANY — it needs the belief stack that
# produces species logits, so the guard has to build the whole thing. Filtered against the current
# signature so an arch change drops stale keys instead of erroring.
_PRODUCTION_ARCH = dict(
    attend_unrevealed_opponents=True, belief_grad_mode="shaping",
    damage_matrices_incoming=True, damage_matrices_outgoing=True,
    damage_matrices_outgoing_all=True, damage_op=True, damage_outgoing=True,
    damage_reattend=False, damage_refine_rounds=2, damage_topk_k=5,
    move_belief_mode="revealed", move_belief_prefuse=False,
    move_candidate_floor=0.02, move_latent=True, move_prior_fusion=True,
    opp_belief_latent=True, opp_belief_slots=True, spread_belief=True,
    spread_belief_nature=True, spread_belief_nature_marginalize=True,
    threat_prob_outspeed=True, threat_refine_outgoing=True, threat_status_refine=True,
    threat_unrevealed_outgoing=True, value_active_readout=True, value_dist_bins=51,
    value_dist_mode="shaping", value_dist_vmax=12.0, value_dist_vmin=-12.0,
    win_prob_mode="shaping", zarch_dim=32, zarch_film="heads",
)


def _build_production_extractor():
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {a: b for a, b in _PRODUCTION_ARCH.items() if a in sig}
    torch.manual_seed(0)
    return Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kw).eval(), layout


@_skip_compile
def test_production_arch_compiles_without_suppression():
    """THE regression. `--threat-unrevealed-outgoing` is what crashed Inductor; with suppression OFF
    a reintroduced bad spelling raises `BackendCompilerFailed` here instead of silently costing half
    the speedup in production."""
    torch.set_num_threads(1)
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    fe, layout = _build_production_extractor()
    obs = {"observation": torch.rand(1, layout["total_dim"],
                                     generator=torch.Generator().manual_seed(7))}
    with torch.no_grad():
        ref = fe(obs)
        got = torch.compile(fe.forward)(obs)
    err = max(float((a - b).abs().max()) for a, b in zip(ref, got))
    assert err < 1e-5, f"compiled output diverged from eager: {err:.2e}"


@_skip_compile
def test_production_arch_compiles_to_one_graph():
    """Graph breaks are how a 6.5x quietly becomes a 1.2x — the win depends on ONE fused graph."""
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    fe, layout = _build_production_extractor()
    obs = {"observation": torch.zeros(1, layout["total_dim"])}
    explained = torch._dynamo.explain(fe.forward)(obs)
    assert explained.graph_break_count == 0, explained.break_reasons
    assert explained.graph_count == 1
