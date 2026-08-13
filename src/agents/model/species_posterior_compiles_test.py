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
    head.species_logits = lambda tokens, *a, **k: logits    # type: ignore[assignment]
    got = BeliefHead.species_posterior(head, torch.zeros(3, 6, 8))
    want = torch.softmax(logits, dim=-1)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()
    assert torch.allclose(got.sum(-1), torch.ones(3, 6), atol=1e-5)


def test_species_posterior_is_stable_for_large_logits():
    """`log_softmax().exp()` keeps softmax's max-subtraction, so a naive `exp/sum` blowup must not
    reappear if someone re-spells this."""
    head = BeliefHead.__new__(BeliefHead)
    logits = torch.tensor([[[1e4, 1e4 + 1.0, -1e4]]])
    head.species_logits = lambda tokens, *a, **k: logits    # type: ignore[assignment]
    got = BeliefHead.species_posterior(head, torch.zeros(1, 1, 8))
    assert torch.isfinite(got).all()
    assert pytest.approx(1.0, abs=1e-5) == float(got.sum())


# ⚠️ THE PRODUCTION ARCH IS LOADED, NEVER HAND-PINNED. This test used to carry a literal
# `_PRODUCTION_ARCH = dict(...)` frozen at the ai_v8_03 shape (refine_rounds=2, no prefuse, no
# edge families) — and it ROTTED silently: for three generations the default-on compile gate
# faithfully compiled a DEAD graph while the real production graph drifted, which is exactly how
# the gen-4 launch's CompilePrewarm failure (`gen3_unrevealed_outgoing_prior_v1`'s [B,6,S]
# expand mis-vectorizing on Inductor CPU) shipped past a green suite. The kwargs now come from
# `designs/production_config.json` — the SAME single source `delivery_graph`, the arch viewer
# and the prewarm-era eval workers key on — filtered by the live constructor signature exactly
# like `delivery_graph.build_graph` and `compile_prewarm` do. The `arch_signature` assertion
# makes a stale config FAIL LOUD here instead of quietly testing yesterday's model.
_PRODUCTION_CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                  "designs", "production_config.json")


def _build_production_extractor():
    import json

    from agents.model.model_version import ARCH_SIGNATURE
    from agents.model.damage_tables import sanitize_historical_move_floor

    with open(_PRODUCTION_CONFIG) as fh:
        cfg = json.load(fh)
    assert cfg.get("arch_signature") == ARCH_SIGNATURE, (
        f"designs/production_config.json is STALE (arch_signature "
        f"{cfg.get('arch_signature')!r} != live {ARCH_SIGNATURE!r}) — this compile gate would "
        f"be testing a dead graph. Refresh the config from the production run's "
        f"model_config.json (the _PRODUCTION_ARCH-dict rot class this assertion exists for)."
    )
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {a: b for a, b in cfg.items() if a in sig}
    # v65 (gen3_unconditional_move_legality_v1): the committed config is a VERBATIM pre-v65 run copy
    # and records move_candidate_floor 0.0, which the validated range now rejects. Reconcile at
    # construction — the same seam delivery_graph uses — rather than editing the historical record.
    sanitize_historical_move_floor(kw)
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
