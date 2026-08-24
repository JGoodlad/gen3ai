"""Unit tests for the B=1 compile router.

The measurement it encodes is in ``perf.py``: compiled is 5.45x at B=1 and 0.15-0.43x at the widths
the search actually realizes. So the ONE property that matters is that a wide forward never reaches
the compiled graph — a router that leaked would turn the probe's arm scoring 2-7x slower while every
log line still said "compiled", which is the invisible-regression shape this project keeps eating.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from main.search_dividend import perf


class _FE:
    def __init__(self):
        self.calls = []

    def forward(self, obs):
        self.calls.append(("eager", perf.batch_of(obs)))
        return "eager"


def _model(fe, dim=4, n_act=3):
    return types.SimpleNamespace(
        policy=types.SimpleNamespace(features_extractor=fe),
        observation_space={"observation": types.SimpleNamespace(shape=(dim,)),
                           "action_mask": types.SimpleNamespace(shape=(n_act,))})


def _obs(b):
    return {"observation": np.zeros((b, 4), dtype=np.float32),
            "action_mask": np.ones((b, 3), dtype=np.float32)}


@pytest.mark.parametrize("b", [1, 2, 8, 64, 300])
def test_batch_of_reads_the_leading_dim(b):
    assert perf.batch_of(_obs(b)) == b
    assert perf.batch_of(np.zeros((b, 4))) == b


def test_batch_of_returns_None_rather_than_guessing():
    """Unreadable must route to EAGER, and eager is selected by ``!= COMPILED_BATCH``."""
    assert perf.batch_of({"nope": 1}) is None
    assert perf.batch_of(None) is None
    assert perf.batch_of(None) != perf.COMPILED_BATCH


def test_disabled_is_a_no_op():
    fe = _FE()
    assert perf.compile_b1_extractor(_model(fe), enabled=False) is False
    assert "forward" not in vars(fe), "no router was installed"
    assert fe.forward(_obs(1)) == "eager"


def test_a_model_with_no_extractor_declines():
    assert perf.compile_b1_extractor(types.SimpleNamespace(policy=None), enabled=True) is False
    assert perf.compile_b1_extractor(types.SimpleNamespace(), enabled=True) is False


def test_a_refused_compile_leaves_the_forward_untouched(monkeypatch):
    fe = _FE()
    monkeypatch.setattr("agents.model.compile_opponents.maybe_compile_extractor",
                        lambda *a, **k: False)
    assert perf.compile_b1_extractor(_model(fe), enabled=True) is False
    assert "forward" not in vars(fe), "no router was installed"
    assert fe.forward(_obs(1)) == "eager"


def _install_fake_compile(monkeypatch, fe, warmed):
    """Stand in for `maybe_compile_extractor`: patch a marker callable onto `fe.forward`."""
    def fake(model, enabled, label="", hide_cuda=False, strict=False):
        def compiled(obs):
            fe.calls.append(("compiled", perf.batch_of(obs)))
            return "compiled"
        fe.forward = compiled
        return True
    monkeypatch.setattr("agents.model.compile_opponents.maybe_compile_extractor", fake)
    monkeypatch.setattr(perf, "_warm_live_signature", lambda m: warmed.append(m))


def test_B1_routes_to_the_compiled_graph_and_everything_wider_stays_eager(monkeypatch):
    fe, warmed = _FE(), []
    _install_fake_compile(monkeypatch, fe, warmed)
    assert perf.compile_b1_extractor(_model(fe), enabled=True) is True
    assert fe.forward(_obs(1)) == "compiled"
    for b in (2, 8, 64, 128, 300):
        assert fe.forward(_obs(b)) == "eager", f"B={b} must not reach the compiled graph"
    assert [c for c, _ in fe.calls] == ["compiled"] + ["eager"] * 5
    assert warmed, "the live call signature must be warmed, or the first decision re-traces"


def test_an_unreadable_obs_routes_to_eager(monkeypatch):
    fe, warmed = _FE(), []
    _install_fake_compile(monkeypatch, fe, warmed)
    perf.compile_b1_extractor(_model(fe), enabled=True)
    assert fe.forward({"not_an_obs": 1}) == "eager"


def test_the_flag_exists_and_defaults_OFF():
    """OFF is the measurement, not an oversight: compiling perturbs the forward at ~1e-6 and an
    argmax over near-tied actions can flip on it."""
    from main.search_dividend.__main__ import build_parser

    args = build_parser().parse_args(["model"])
    assert args.compile_extractor is False
    assert build_parser().parse_args(["model", "--compile-extractor"]).compile_extractor is True
