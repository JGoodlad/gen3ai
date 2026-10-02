"""T2's greedy near-tie band and its precision refusal — fp32 matmul precision 'highest' is the ONLY
precision (TF32 was retired, deletion pass K2), so the bars are two constants in `parity_probe`
(`LOGPROB_BAR`, `NEAR_TIE_BAND` = 2x it), and a process at any other precision is REFUSED, never judged
with a guess.

These run on CPU. The band tests craft the eager log-probs (a stub of `parity.policy_reference`, which
`judge` reads at call time) so a row's top-2 margin is exact; each test names what its revert does.
"""
from __future__ import annotations

import pytest
import torch

from agents.inference.service import (
    InferenceService, ParityFailure, ServiceSpec, SlotGroupSpec, policy_reference,
)
from agents.inference.service import parity
from agents.inference.service.fixtures import perturbed_fresh_policy
from agents.inference.service.parity import fixture_rows, judge
from agents.model import compile_trainer as ct
from agents.model import parity_probe as pp
from utils.torch_state_guard import torch_globals


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


# ---------------------------------------------------------------- the constants


def test_the_band_is_2x_the_compile_gates_log_prob_bar_read_from_one_place():
    """Revert the band to the bare bar ⇒ the first assertion fails (the TECH_DEBT inconsistency); a
    second copy of the bar in the compile gate ⇒ the second does."""
    assert pp.NEAR_TIE_BAND == 2.0 * pp.LOGPROB_BAR
    assert pp.LOGPROB_BAR == ct._FP32_TOL["legal_logprob"], \
        "the fp32 bar must be the compile gate's legal log-prob bar, read from one place"
    assert all(0.0 < sc <= pp.PERTURB_MAX_SCALE for sc, _ in pp.PERTURB_LADDER)
    for gone in ("PRECISION_BARS", "precision_bars", "tie_band", "ladder_at"):
        assert not hasattr(pp, gone), f"{gone}: the precision-keyed table is deleted with TF32"


def test_the_measured_precision_is_accepted_and_every_other_is_refused_with_a_message():
    assert pp.MEASURED_PRECISION == "highest" == torch.get_float32_matmul_precision()
    assert pp.unmeasured_precision() is None
    for other in ("high", "medium"):
        with torch_globals(float32_matmul_precision=other):
            msg = pp.unmeasured_precision()
        assert msg is not None and repr(other) in msg and "no measured parity bars" in msg, other


@pytest.mark.parametrize("precision", ["high", "medium"])
def test_a_non_fp32_precision_is_refused_by_the_parity_judge_not_judged_with_a_guess(policy, precision):
    """A TF32 request ('high') is refused like any unmeasured precision. Revert the refusal in `judge`
    ⇒ it judges at the fp32 bars and either passes a TF32 graph it never measured or fails it by
    chance."""
    o, m, logp, value = _rows(policy, 8)
    with torch_globals(float32_matmul_precision=precision):
        with pytest.raises(ParityFailure, match="no measured parity bars"):
            judge(where=precision, policy=policy, obs=o, mask=m,
                  served=(logp, value, logp.argmax(-1)))


# ---------------------------------------------------------------- judge, crafted eager log-probs


def _pick_row_with_two_legal(m):
    rows = (m.sum(-1) >= 2).nonzero().flatten()
    assert rows.numel() >= 1, "precondition: the fixture has a row with two legal actions"
    return int(rows[0])


def _crafted(monkeypatch, policy, margin):
    """Eager log-probs whose row ``r`` has its top-2 margin set to exactly ``margin``, served
    unchanged — plus the served greedy that picks eager's SECOND action on that row (and eager's
    argmax everywhere else). Returns ``(obs, mask, served, row)``."""
    o, m, logp, value = _rows(policy, 16)
    r = _pick_row_with_two_legal(m)
    logp = logp.clone()
    top2 = logp[r].topk(2)
    logp[r, top2.indices[1]] = top2.values[0] - margin
    greedy = logp.argmax(-1).clone()
    greedy[r] = top2.indices[1]
    monkeypatch.setattr(parity, "policy_reference", lambda *_a, **_k: (logp, value))
    return o, m, (logp, value, greedy), r


def test_a_flip_INSIDE_the_near_tie_band_is_a_counted_tie_and_one_OUTSIDE_it_fails(monkeypatch, policy):
    """The greedy rule: the served argmax equals eager's wherever eager's top-2 margin exceeds
    `NEAR_TIE_BAND`; inside it a flip is a counted near-tie. Revert the band to 0 ⇒ the first judge
    raises; widen it without bound ⇒ the second passes."""
    o, m, served, r = _crafted(monkeypatch, policy, pp.NEAR_TIE_BAND / 2.0)
    assert float(_margins(served[0])[r]) < pp.NEAR_TIE_BAND
    rep = judge(where="inside", policy=policy, obs=o, mask=m, served=served)
    assert rep.near_ties >= 1, rep.line()

    o, m, served, r = _crafted(monkeypatch, policy, pp.NEAR_TIE_BAND * 2.5)
    assert float(_margins(served[0])[r]) > pp.NEAR_TIE_BAND
    with pytest.raises(ParityFailure, match=r"greedy action differs .*near-tie band"):
        judge(where="outside", policy=policy, obs=o, mask=m, served=served)


def test_a_REAL_greedy_divergence_far_below_eagers_top_fails(policy):
    """The served action is a legal action more than the band below eager's top: a wrong choice. Revert
    the near-top check ⇒ it passes as a 'near-tie'."""
    o, m, logp, value = _rows(policy, 128)
    legal_low = torch.where(m, logp, torch.full_like(logp, float("inf"))).argmin(-1)
    gap = logp.max(-1).values - logp.gather(1, legal_low.view(-1, 1)).squeeze(1)
    far = gap > 2.0 * pp.NEAR_TIE_BAND
    assert int(far.sum()) >= 1, "precondition: some row has a legal action far below its top"
    greedy = torch.where(far, legal_low, logp.argmax(-1))
    with pytest.raises(ParityFailure, match="greedy action differs|not one of eager's near-top"):
        judge(where="far", policy=policy, obs=o, mask=m, served=(logp, value, greedy))


# ---------------------------------------------------------------- the service end to end (CPU eager)


def _service(template, **kw):
    spec = ServiceSpec(groups=(SlotGroupSpec("pool", 1, template),), device="cpu", backend="eager",
                       buckets=(2, 8), **kw)
    return InferenceService(spec).startup()


def test_a_service_started_at_a_non_fp32_precision_is_refused_at_startup(policy):
    """A TF32 service cannot start: the startup parity gate refuses the precision. Revert the refusal
    ⇒ the 'high' startup succeeds on a gate whose bars were never measured there."""
    _service(policy)                                           # control: fp32 starts
    with torch_globals(float32_matmul_precision="high"):
        with pytest.raises(ParityFailure, match="no measured parity bars"):
            _service(policy)


def test_a_service_whose_served_greedy_flips_a_decisive_row_is_refused(monkeypatch, policy):
    """The fresh-launch gate end to end. Revert the greedy check ⇒ the startup passes a service that
    serves a different action than the policy's own eager path."""
    import agents.inference.service.engine as engine_mod
    real = engine_mod.decide

    def decide(module, obs, mask):
        logp, value, greedy = real(module, obs, mask)
        flipped = greedy.clone()
        decisive = _margins(logp) > pp.NEAR_TIE_BAND
        assert bool(decisive.any()), "precondition: some served row is decisive"
        flipped[decisive] = logp.topk(2, dim=-1).indices[decisive, 1]
        return logp, value, flipped

    monkeypatch.setattr(engine_mod, "decide", decide)
    with pytest.raises(ParityFailure, match="greedy action differs"):
        _service(policy)
