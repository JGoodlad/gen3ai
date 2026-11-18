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
`consistency.FP32_TIE_EPS` to `TEST_EPS`, which the gate reads at call time (the fourth test pins that). Rule 8 in the
test itself: the planted row's margin is at most HALF the epsilon and at least a TENTH of it (5x the production
epsilon, so no fp32 forward resolves the pair the other way by rounding), its flip moves its legal log-probs by at
least 10x the bar, and the drifted control row's margin is at least 10x the epsilon.

The tie is PLANTED, not found (2026-11-18). A pair whose swap moves a log-prob is rare: on 108 policies (12 seeds x 9
scales around 0.3) 3 held one inside the band, and the first version's four hand-measured seeds lost theirs when the
model's numerics moved (config v168, `3b1733f50`: seed 1's row 14 went from margin 2.2e-3 to 8.9e-3, its flip still
0.17, and all five tests failed in the fixture until this was written). So `_plant_seed` takes a LEAD, a row whose one
near pair at `LEAD_EPS` has a flip that moves the log-probs, and solves for the perturbation scale that puts the
pair's signed margin in the middle of the band (a secant; the pair's gap is smooth in the scale), then judges that
policy by the same acceptance. A lead survives a change of the numerics that moves its margin anywhere inside
`LEAD_EPS` (v168 moved row 14's by 6.7e-3); `SEEDS` records what was measured, on this tree and on two older ones.
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
#: The planted margin's band (module docs): clear of an exact tie and of the epsilon, the secant aims at `AIM`.
TIE_FLOOR, TIE_CEIL, AIM = TEST_EPS / 10, TEST_EPS / 2, 0.3 * TEST_EPS
#: A lead is a row with ONE near pair at this wider epsilon whose flip moves its legal log-probs by 10x the bar.
LEAD_EPS = 3 * TEST_EPS
#: The perturbation scale a seed starts from (0.05, the parity probe's own, holds no flip that moves a log-prob), the
#: secant's first step, how far it may travel and how many policies one seed may build.
SCALE, FIRST_STEP, SPAN, STEPS = 0.3, 5e-4, 0.02, 6
#: The seeds tried in order. Measured 2026-11-18 (CPU, 2 threads, main ee884a9c1): seed 1 plants fixture row 14 at
#: scale 0.3005 (margin 1.5e-3, flip |dlogp| 0.041) and seed 9 row 11 at 0.300114 (3.0e-3, 0.25); no other seed of
#: the 32 holds a lead. The same two plant on the trees before v168 (`1bab9e87a`) and before v165 (`2e63e5f0e`),
#: where seed 0 does too (row 2). An exhausted search costs 49 s.
SEEDS = range(32)


def _legal_gap(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return torch.where(m, (a - b).abs(), torch.zeros_like(a)).amax(-1)


def _flips(policy, o, m, eps):
    """One eager forward and one forward with EVERY row flippable at ``eps`` flipped (rows do not interact):
    (eager (logp, V), the recorder, {row: (margin, the flip's max legal |dlogp|, its FlipStep)})."""
    rec = TieMargins(ROWS, near_eps=eps)
    with rec:
        eager = policy_reference(policy, o, m)
    rec.check()
    rows = [int(r) for r in np.flatnonzero(rec.flippable())]
    if not rows:
        return eager, rec, {}
    plan = rec.flip_plan(rows)
    mode = TieFlip(ROWS, plan, rec.call_sites)
    with mode:
        f_lp, _ = policy_reference(policy, o, m)
    mode.check_applied()
    gap = _legal_gap(f_lp, eager[0], m)
    steps = {r: st for at_call in plan.values() for r, st in at_call}
    return eager, rec, {r: (float(rec.margin[r]), float(gap[r]), steps[r]) for r in rows}


def _plant(policy, o, m):
    """The acceptance: (eager, that row ALONE flipped, the row, a control row), or None."""
    (e_lp, e_v), rec, flips = _flips(policy, o, m, TEST_EPS)
    far = [int(r) for r in np.flatnonzero(np.asarray(rec.margin) >= 10 * TEST_EPS)]
    for r, (margin, gap, _step) in sorted(flips.items()):
        if far and TIE_FLOOR <= margin <= TIE_CEIL and gap >= 10 * LOGPROB_BAR:
            mode = TieFlip(ROWS, rec.flip_plan([r]), rec.call_sites)
            with mode:
                f_lp, f_v = policy_reference(policy, o, m)
            mode.check_applied()
            return (e_lp, e_v), (f_lp, f_v), r, far[0]
    return None


def _lead(policy, o, m, lead):
    """((row, call, the pair), its SIGNED margin): the first call names the lead (the smallest margin whose flip
    moves the log-probs), a later one re-reads the SAME pair, negative once its two candidates have swapped."""
    _eager, _rec, flips = _flips(policy, o, m, LEAD_EPS)
    moving = {r: f for r, f in flips.items() if f[1] >= 10 * LOGPROB_BAR}
    if lead is None:
        if not moving:
            return None
        row = min(moving, key=lambda r: (moving[r][0], r))
        return (row, moving[row][2].call, moving[row][2].witness), moving[row][0]
    row, call, pair = lead
    margin, _gap, step = moving.get(row, (0.0, 0.0, None))
    if step is None or step.call != call or set(step.witness) != set(pair):
        return None
    return lead, margin if step.witness == pair else -margin


def _plant_seed(seed):
    """(policy, obs, mask) + `_plant`'s tuple at the scale that puts one seed's lead pair in the band, or None."""
    scale, lead, seen = SCALE, None, []
    for _ in range(STEPS):
        policy = perturbed_fresh_policy(seed, scale=scale)
        obs, mask = fixture_rows(policy.observation_space["observation"].shape[0], ROWS)
        o, m = torch.as_tensor(obs), torch.as_tensor(mask)
        got = _plant(policy, o, m)
        if got is not None:
            return (policy, o, m) + got
        read = _lead(policy, o, m, lead)
        if read is None:
            return None
        lead, signed = read
        seen.append((scale, signed - AIM))
        if len(seen) == 1:
            scale += FIRST_STEP
            continue
        (s0, g0), (s1, g1) = seen[-2:]
        if g1 == g0:
            return None
        scale = s1 - g1 * (s1 - s0) / (g1 - g0)
        if abs(scale - SCALE) > SPAN:
            return None
    return None


@pytest.fixture(scope="module")
def planted():
    """(policy, obs, mask, eager (logp, V), flipped (logp, V), the flipped row, a row 10x clear of every cutoff)."""
    with torch_globals(num_threads=2):
        for seed in SEEDS:
            got = _plant_seed(seed)
            if got is not None:
                yield got
                return
    pytest.fail(f"no seed of {SEEDS} at scale {SCALE:g} plants a flippable row at a margin in [{TIE_FLOOR:g}, "
                f"{TIE_CEIL:g}] whose flip moves its legal log-probs by >= {10 * LOGPROB_BAR:g}: the model's numerics "
                "moved every lead out of the secant's reach; re-measure SEEDS (module docs)")


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
    rows = max(op.ROWS, row + 1)          # the gate's own rows, and as many more as reach the planted one
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

        line = op.check_opponent_parity(model, original, compiled, label="planted tie", rows=rows)
        assert "selection ties: 1 flip-judged" in line, line
        monkeypatch.setattr(consistency, "FP32_TIE_EPS", 0.0)
        monkeypatch.setattr(op, "_PASSED", {})
        with pytest.raises(CompileTrainerError, match="DISAGREES"):
            op.check_opponent_parity(model, original, compiled, label="eps 0", rows=rows)
