"""The parity gates and SELECTION TIES (`gen3_t2_parity_tie_judge_v1`, root rule 8), on the CPU: T2's `parity.judge` and
the compiled-opponent gate (`opponent_parity`), which share ONE tie rule (`tie_margins.resolve_ties`) with K9(b).

X60-T s3501 died `FATAL_CONFIG` (2026-11-06): its compiled b256 graph resolved a near-tied selection on two fixture
rows the other way from eager (opponent slot 3's move order, Hidden Power Grass vs Ice at the K = 6 move-seat cut,
relative margin 1.6e-6), so |dlogp| read 0.14 on a forward that was correct. On the CPU the failing weights' eager with
that pair flipped (K9(b)'s `TieFlip`) equals the GPU's served rows to 1.2e-5.

Here the same shape is planted on a perturbed fresh policy: the "served" output IS the eager forward with one tied row's
selection resolved the other way. The gate must PASS it (before the fix it FAILED: revert `_resolve_ties` and the first
test fails on legal_logprob), and must still FAIL a drift on a row at no tie, and a tied row that matches neither
resolution.

The epsilon. A fresh policy's ties at K9(b)'s `FP32_TIE_EPS` (2e-4) are exact ties whose flip moves nothing (measured: the
largest flip there reads 2e-4 on legal log-probs, under the bar), so the planted row needs a wider one: the tests set
`consistency.FP32_TIE_EPS` to `TEST_EPS`, which the gate reads at call time (the last test pins that). Rule 8 in the test
itself: the planted row's margin is at most HALF the epsilon, its flip moves its legal log-probs by at least 10x the
bar, and the drifted control row's margin is at least 10x the epsilon.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.inference.service.decision import policy_reference
from agents.inference.service.fixtures import perturbed_fresh_policy
from agents.inference.service.parity import fixture_rows, judge
from agents.inference.service.spec import ParityFailure
from agents.model.parity_probe import LOGPROB_BAR
from agents.training.rust_rollout import consistency
from agents.training.rust_rollout.tie_margins import TieFlip, TieMargins
from utils.torch_state_guard import torch_globals

pytestmark = pytest.mark.slow  # builds production-arch policies and runs several eager forwards of 64 rows

ROWS = 64
TEST_EPS = 1e-2
#: (seed, scale) of the perturbed fresh policies searched, in order, for a planted row. Measured 2026-11-06 (CPU, main
#: 2e63e5f0e): seed 1 at scale 0.3 holds one (fixture row 14, margin 2.2e-3, flip |dlogp| 0.79); scale 0.05 never does.
SEARCH = ((1, 0.3), (0, 0.3), (2, 0.3), (3, 0.3))


def _legal_gap(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return torch.where(m, (a - b).abs(), torch.zeros_like(a)).amax(-1)


def _plant(policy, o, m):
    rec = TieMargins(ROWS, near_eps=TEST_EPS)
    with rec:
        e_lp, e_v = policy_reference(policy, o, m)
    rec.check()
    margin = np.asarray(rec.margin)
    far = [int(r) for r in np.flatnonzero(margin >= 10 * TEST_EPS)]
    for r in [int(r) for r in np.flatnonzero(rec.flippable() & (margin <= TEST_EPS / 2))]:
        mode = TieFlip(ROWS, rec.flip_plan([r]), rec.call_sites)
        with mode:
            f_lp, f_v = policy_reference(policy, o, m)
        mode.check_applied()
        if far and float(_legal_gap(f_lp, e_lp, m)[r]) >= 10 * LOGPROB_BAR:
            return (e_lp, e_v), (f_lp, f_v), r, far[0]
    return None


@pytest.fixture(scope="module")
def planted():
    """(policy, obs, mask, eager (logp, V), flipped (logp, V), the flipped row, a row 10x clear of every cutoff)."""
    with torch_globals(num_threads=2):
        for seed, scale in SEARCH:
            policy = perturbed_fresh_policy(seed, scale=scale)
            obs, mask = fixture_rows(policy.observation_space["observation"].shape[0], ROWS)
            o, m = torch.as_tensor(obs), torch.as_tensor(mask)
            got = _plant(policy, o, m)
            if got is not None:
                yield (policy, o, m) + got
                return
    pytest.fail(f"no perturbed fresh policy of {SEARCH} holds a flippable row at a margin <= {TEST_EPS / 2:g} whose "
                f"flip moves its legal log-probs by >= {10 * LOGPROB_BAR:g}")


def _served(lp: torch.Tensor, v: torch.Tensor):
    return lp, v, lp.argmax(-1)


def test_a_row_at_the_OTHER_resolution_of_a_selection_tie_PASSES_and_is_counted(planted, monkeypatch):
    policy, o, m, (e_lp, e_v), (f_lp, f_v), row, _far = planted
    monkeypatch.setattr(consistency, "FP32_TIE_EPS", TEST_EPS)
    assert float(_legal_gap(f_lp, e_lp, m)[row]) >= 10 * LOGPROB_BAR       # the as-is judgement cannot pass it
    others = torch.ones(ROWS, dtype=torch.bool)
    others[row] = False
    assert float(_legal_gap(f_lp, e_lp, m)[others].max()) == 0.0, "the flip moved a row it did not flip"
    with torch_globals(num_threads=2):
        rep = judge(where="planted tie", policy=policy, obs=o, mask=m, served=_served(f_lp, f_v))
    assert rep.tie_flipped == 1, rep.line()
    assert "selection ties 1 flip-judged" in rep.line()


def test_a_drift_on_a_row_at_NO_tie_still_fails(planted, monkeypatch):
    policy, o, m, (e_lp, e_v), (f_lp, f_v), row, far = planted
    monkeypatch.setattr(consistency, "FP32_TIE_EPS", TEST_EPS)
    assert far != row
    bad = f_lp.clone()
    bad[far] = torch.where(m[far], bad[far] + 5 * LOGPROB_BAR, bad[far])
    with torch_globals(num_threads=2), pytest.raises(ParityFailure, match=r"(?s)legal_logprob.*selection ties resolved"):
        judge(where="drift at no tie", policy=policy, obs=o, mask=m, served=_served(bad, f_v))


def test_a_tied_row_matching_NEITHER_resolution_fails(planted, monkeypatch):
    policy, o, m, (e_lp, e_v), _flipped, row, _far = planted
    monkeypatch.setattr(consistency, "FP32_TIE_EPS", TEST_EPS)
    bad = e_lp.clone()
    bad[row] = torch.where(m[row], bad[row] + 5 * LOGPROB_BAR, bad[row])
    with torch_globals(num_threads=2), pytest.raises(ParityFailure, match="legal_logprob"):
        judge(where="neither", policy=policy, obs=o, mask=m, served=_served(bad, e_v))


def test_the_tie_rule_reads_K9bs_epsilon_at_call_time(planted, monkeypatch):
    """The epsilon is K9(b)'s, read at call time: at 0 no row is tied, so the planted flip FAILS."""
    policy, o, m, _eager, (f_lp, f_v), _row, _far = planted
    monkeypatch.setattr(consistency, "FP32_TIE_EPS", 0.0)
    with torch_globals(num_threads=2), pytest.raises(ParityFailure, match="legal_logprob"):
        judge(where="eps 0", policy=policy, obs=o, mask=m, served=_served(f_lp, f_v))


def test_the_OPPONENT_gate_shares_the_rule(planted, monkeypatch):
    """`opponent_parity` (B=1 rows): a "compiled" extractor that resolves the planted row's tie the other way PASSES,
    flip-judged; with the rule off (epsilon 0) the same forward is REFUSED — the fail-on-revert witness."""
    from types import SimpleNamespace

    from agents.model import opponent_parity as op
    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.compile_trainer import CompileTrainerError

    policy, _o, _m, _eager, _flipped, row, _far = planted
    assert row < op.ROWS, f"the planted row {row} is outside the opponent gate's first {op.ROWS} rows"
    monkeypatch.setattr(consistency, "FP32_TIE_EPS", TEST_EPS)
    monkeypatch.setattr(op, "_PASSED", {})
    model = SimpleNamespace(policy=policy)
    fe = policy.features_extractor
    original = fe.forward
    x, mk = load_parity_rows(policy.observation_space["observation"].shape[0])
    with torch_globals(num_threads=2):
        rec = TieMargins(1, near_eps=TEST_EPS)
        with torch.no_grad(), rec:
            op._arm(model, fe, original, x[row:row + 1], mk[row:row + 1])
        rec.check()
        assert rec.flippable()[0], "the planted row is not flippable at B=1"
        plan, sites = rec.flip_plan([0]), rec.call_sites
        target = torch.as_tensor(x[row:row + 1], dtype=torch.float32)

        def compiled(obs):
            if torch.equal(obs["observation"], target):
                with TieFlip(1, plan, sites):
                    return original(obs)
            return original(obs)

        line = op.check_opponent_parity(model, original, compiled, label="planted tie")
        assert "selection ties: 1 flip-judged" in line, line
        monkeypatch.setattr(consistency, "FP32_TIE_EPS", 0.0)
        monkeypatch.setattr(op, "_PASSED", {})
        with pytest.raises(CompileTrainerError, match="DISAGREES"):
            op.check_opponent_parity(model, original, compiled, label="eps 0")
