"""gen3_precision_keyed_parity_v1 — T2's greedy near-tie band and its ladder cap are KEYED BY the float32
matmul precision the gate runs at, from ONE measured table (`parity_probe.PRECISION_BARS`).

THE DEFECT (2026-09-30, the VacuousParity unit's GPU read): under ``--matmul-precision high`` (TF32) a
FRESH production policy was REFUSED at T2 startup — "greedy action differs from eager on 1 decisive
rows (top-2 margin > 0.001)". The greedy rule's band was the fp32 log-prob bar at every precision,
but TF32's healthy compiled-vs-eager |Δ log π| reaches 1.4e-2 on T2's own fixture rows (0.040 on the
learner's), so a row whose top-2 margin is a few 1e-3 flips legitimately. The fp32 band was also the
bar itself rather than 2x it (TECH_DEBT row, Lane E's rule is 2x).

These run on CPU: CPU matmuls ignore TF32, so setting the precision exercises exactly the KEYING (the
band and the cap the gate picks), with the served output crafted. Each test names what its revert does.
"""
from __future__ import annotations

import pytest
import torch

from agents.inference.service import (
    InferenceService, ParityFailure, ServiceSpec, SlotGroupSpec, VacuousParity, policy_reference,
)
from agents.inference.service.fixtures import perturbed_fresh_policy
from agents.inference.service.parity import fixture_rows, judge
from agents.model import compile_trainer as ct
from agents.model import parity_probe as pp
from utils.torch_state_guard import torch_globals

#: The window a TF32 near-tie lives in: decisive at fp32, a near-tie at TF32.
_FP32_BAND = pp.tie_band("highest")
_TF32_BAND = pp.tie_band("high")


@pytest.fixture(scope="module")
def policy():
    """A perturbed fresh production policy (informative: its log-probs vary within rows)."""
    with torch_globals(num_threads=2):
        yield perturbed_fresh_policy(0)


def _rows(policy, n):
    obs, mask = fixture_rows(policy.observation_space["observation"].shape[0], n)
    o, m = torch.as_tensor(obs), torch.as_tensor(mask)
    logp, value = policy_reference(policy, o, m)
    return o, m, logp, value


def _margins(logp):
    top2 = logp.topk(2, dim=-1).values
    return torch.where(torch.isfinite(top2[:, 1]), top2[:, 0] - top2[:, 1],
                       torch.full_like(top2[:, 0], float("inf")))


# ---------------------------------------------------------------- the table


def test_ONE_table_the_band_is_2x_each_precisions_bar_and_fp32_is_the_compile_gates_bar():
    """Revert the band to the bare bar ⇒ the first assertion fails (the TECH_DEBT inconsistency)."""
    for prec, (bar, cap) in pp.PRECISION_BARS.items():
        assert pp.tie_band(prec) == 2.0 * bar, prec
        assert 0.0 < cap <= pp.PERTURB_MAX_SCALE, prec
    assert pp.PRECISION_BARS["highest"][0] == ct._FP32_TOL["legal_logprob"], \
        "the fp32 bar must be the compile gate's legal log-prob bar, read from one place"
    assert pp.PRECISION_BARS["high"][1] < pp.PRECISION_BARS["highest"][1], \
        "TF32's ladder cap is tighter than fp32's (its 0.1 rungs drift past the TF32 rule)"
    assert all(sc <= pp.PRECISION_BARS["high"][1] for sc, _ in pp.ladder_at("high"))
    assert any(sc > pp.PRECISION_BARS["high"][1] for sc, _ in pp.ladder_at("highest"))


def test_an_UNMEASURED_precision_is_refused_not_judged_with_a_guess(policy):
    o, m, logp, value = _rows(policy, 8)
    with torch_globals(float32_matmul_precision="medium"):
        with pytest.raises(ParityFailure, match="no measured parity bars"):
            judge(where="medium", policy=policy, obs=o, mask=m,
                  served=(logp, value, logp.argmax(-1)))


# ---------------------------------------------------------------- judge, crafted served output


def _near_tie_flip(logp, margin):
    """Greedy = eager's SECOND choice on every row whose top-2 margin is inside the TF32 window."""
    window = (margin > _FP32_BAND) & (margin < _TF32_BAND)
    greedy = logp.argmax(-1).clone()
    greedy[window] = logp.topk(2, dim=-1).indices[window, 1]
    return greedy, window


def test_a_LEGITIMATE_TF32_near_tie_flip_is_a_counted_tie_under_TF32_and_a_failure_under_fp32(policy):
    """Revert the keying (one fp32 band everywhere) ⇒ the TF32 judge raises like the fp32 one."""
    o, m, logp, value = _rows(policy, 128)
    greedy, window = _near_tie_flip(logp, _margins(logp))
    assert int(window.sum()) >= 3, "precondition: the fixture must carry rows in the TF32 window"
    with torch_globals(float32_matmul_precision="high"):
        rep = judge(where="tf32", policy=policy, obs=o, mask=m, served=(logp, value, greedy))
    assert rep.near_ties >= int(window.sum()), rep.line()
    with torch_globals(float32_matmul_precision="highest"):
        with pytest.raises(ParityFailure, match=r"greedy action differs .*'highest' near-tie band"):
            judge(where="fp32", policy=policy, obs=o, mask=m, served=(logp, value, greedy))


@pytest.mark.parametrize("precision", ["highest", "high"])
def test_a_REAL_greedy_divergence_above_the_band_fails_at_BOTH_precisions(policy, precision):
    """The served action is a legal action more than the TF32 band below eager's top: a wrong choice
    at any precision. Revert the near-top check (or widen the band without bound) ⇒ TF32 passes it."""
    o, m, logp, value = _rows(policy, 128)
    legal_low = torch.where(m, logp, torch.full_like(logp, float("inf"))).argmin(-1)
    gap = logp.max(-1).values - logp.gather(1, legal_low.view(-1, 1)).squeeze(1)
    far = gap > 2.0 * _TF32_BAND
    assert int(far.sum()) >= 1, "precondition: some row has a legal action far below its top"
    greedy = torch.where(far, legal_low, logp.argmax(-1))
    with torch_globals(float32_matmul_precision=precision):
        with pytest.raises(ParityFailure, match="greedy action differs|not one of eager's near-top"):
            judge(where=precision, policy=policy, obs=o, mask=m, served=(logp, value, greedy))


# ---------------------------------------------------------------- the service end to end (CPU eager)


def _flip_near_ties(real):
    """A served path whose greedy picks eager's SECOND action on rows inside the TF32 window — what a
    healthy TF32 graph may legitimately do on a near-tie."""
    def decide(module, obs, mask):
        logp, value, greedy = real(module, obs, mask)
        flipped, _ = _near_tie_flip(logp, _margins(logp))
        return logp, value, flipped
    return decide


def _service(template, **kw):
    spec = ServiceSpec(groups=(SlotGroupSpec("pool", 1, template),), device="cpu", backend="eager",
                       buckets=(2, 8), **kw)
    return InferenceService(spec).startup()


def test_a_TF32_SERVICE_with_legitimate_near_tie_flips_starts_up_and_an_fp32_one_refuses(
        monkeypatch, policy):
    """The fresh-launch refusal, fixed at the service level. Revert the keying ⇒ the TF32 startup
    raises "greedy action differs" exactly like the fp32 one."""
    import agents.inference.service.engine as engine_mod
    monkeypatch.setattr(engine_mod, "decide", _flip_near_ties(engine_mod.decide))
    with torch_globals(float32_matmul_precision="high"):
        svc = _service(policy)
        assert sum(r.near_ties for r in svc.startup_reports) > 0
    with torch_globals(float32_matmul_precision="highest"):
        with pytest.raises(ParityFailure, match="greedy action differs"):
            _service(policy)


def test_the_TF32_ladder_skips_rungs_above_its_cap_and_says_so():
    """A collapsed critic that fp32 judges at rung (0.1, 3) is REFUSED at TF32 — whose cap is 0.05 —
    with the skipped rungs named, rather than judged on a rung where TF32's own drift crosses its
    bars. Revert the cap ⇒ TF32 climbs to 0.1 and passes it."""
    from main.fresh_checkpoint import build_fresh_model
    with torch_globals(num_threads=2):
        p = build_fresh_model(0)[0].policy.eval()
        head = p.features_extractor.win_head.net[3]
        with torch.no_grad():
            head.bias.fill_(-9.0)
            head.weight.mul_(0.01)
        with torch_globals(float32_matmul_precision="high"):
            with pytest.raises(VacuousParity, match=r"8 declared rung\(s\) above matmul precision "
                                                    r"'high''s scale cap 0\.05 were not tried"):
                _service(p)
        with torch_globals(float32_matmul_precision="highest"):
            _service(p)                    # control: fp32 climbs to (0.1, 3) and judges it
