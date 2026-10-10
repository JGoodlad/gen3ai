"""`--ko-ramp exact` (`ko_exact.roll_ko_prob`) is the 16-roll P(KO) sum in CLOSED FORM (`gen3_ko_exact_closed_v1`;
owner 2026-10-10: "optimise the 16 rolls, since it's a known formula with a crit chance", then "remove the old one").
The 16-roll sum itself survives HERE only, as the REFERENCE ORACLE (`_sum16`, the shipped loop of config v152 verbatim);
`--ko-ramp exact_closed`, the value that briefly carried the closed form beside it, is RETIRED (last test section).

The op's damage is CONTINUOUS (no per-roll floor — `ko_exact`'s docstring), so the 16 terms
``x_r = (top · r/100 − lo) / w`` are an exact arithmetic sequence and ``Σ clamp(x_r, 0, 1)`` has an exact closed form:
two run counts and one arithmetic-series sum. There is no flooring error to quantify — the closed form and the sum
compute the SAME real function, for our exact HP (±½ HP) and for their HP-percentage bin alike — so the only
difference between the two spellings is floating-point rounding, and every bar below is that rounding's bound.

THE ROUNDING BOUND (u = eps / 2, the unit roundoff; ``k1 = 1 + (|top| + |lo|) / w``, the problem's condition number —
how much a relative perturbation of the inputs moves a term ``x_r``). The 16-roll sum: each term is a product, a
difference and a quotient of the inputs (≤ 4 u k1 + 2 u, its constants ``r / 100`` and ``1 / 0.925`` rounded
included), clamped (1-Lipschitz, so a term near an edge keeps its error), then summed left to right (partial sums ≤ 16:
≤ 136 u), ÷ 16 exactly → ``|sum − exact| ≤ 4 u k1 + 10.5 u``. The closed form: ``x_start`` is one term with a possibly
large quotient (≤ 5 u k1), the step ``|top| · 0.01 / w`` is ≤ 5 u relative (≤ 0.05 u k1 absolute) times a mean index
≤ 15 (≤ 1 u k1 with its rounding); the middle mean is in [0, 1] (+ u); count × mean and the add (≤ 32 u), ÷ 16
exactly → ≤ 6 u k1 + 4 u; a COUNT off by one at a rounding error moves a term that lies within its own error of the
edge (≤ 7 u k1 per edge, two edges, ÷ 16) → ``|closed − exact| ≤ 7 u k1 + 4 u``. Hence the bars, with slack:

* each spelling vs the EXACT RATIONAL value (`fractions.Fraction` from the float inputs): ≤ 6 eps k1;
* closed vs the 16-roll sum: ≤ 8 eps k1 — which is ≤ 1e-12 in float64 wherever k1 ≤ 563, so the 1e-12 bar holds BY
  THE BOUND on that domain and is asserted there directly. Realistic inputs reach k1 ~ 2,900 (our HP as a fraction of
  a 714-HP mon under an overkill hit: w = 1 / maxhp), where the bound is 5.2e-12; there NEITHER spelling is
  guaranteed to 1e-12 against the exact value (the 16-roll sum's own error is the 4 u k1 term), so the asserted bar
  is the bound itself. Measured on this grid (descriptive, not asserted): max |closed − sum| 6.0e-15 over the four
  realistic regimes, 6.5e-13 over the adversarial set, worst |closed − sum| / (eps k1) 0.70 against the bound's 8.

The bars are ABSOLUTE on a probability in [0, 1] (a relative bar on a probability near 0 bounds nothing either
spelling can meet: an input a rounding error from a step reads ~1e-17 in one and 0 in the other). Non-finite inputs
are OUTSIDE the domain (excluded by rule, never by chance): an infinite damage makes the closed form's count quotient
inf / inf = NaN where the sum reads 1 — no op kernel produces a non-finite damage.

Gradients: the counts are piecewise constant (detached), so the closed form's gradient is the sum's term by term —
asserted equal off the kinks, the kink-adjacent inputs excluded BY RULE (a term within 64 eps k1 of 0 or 1, the
radius inside which a count can flip by rounding). AT a kink the 16-roll sum returns torch.clamp's INCLUSIVE
subgradient (a term exactly at 0 or 1 passes its gradient); the closed form returns the inclusive one when the count
quotient lands exactly on the integer, else the one-sided derivative from the side its rounding put the term on —
both valid subgradients, pinned below.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from typing import Any, Dict, List, Tuple

import pytest
import torch

from agents.model import ko_exact as KE
from agents.model.ko_exact import (CRIT_P_BASE, MEAN_ROLL, N_ROLLS, ROLLS, closed_sum, ko_given_hit, roll_ko_prob,
                                   run_counts)

F64 = torch.float64
EPS64 = torch.finfo(F64).eps
EPS32 = torch.finfo(torch.float32).eps
#: the 1e-12 domain: 8 eps64 k1 <= 1e-12  <=>  k1 <= 563 (the bound, not a measured cut)
K1_MAX_1E12 = 1e-12 / (8.0 * EPS64)

Grid = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _sum16(mean_dmg: torch.Tensor, hp_lo: torch.Tensor, hp_hi: torch.Tensor) -> torch.Tensor:
    """THE REFERENCE ORACLE: the 16-roll enumeration `ko_exact.roll_ko_prob` shipped at config v152 (before the closed
    form replaced it), verbatim — each roll's KO resolved over the observed HP interval, summed, ÷ 16."""
    top = mean_dmg / MEAN_ROLL
    width = (hp_hi - hp_lo).clamp_min(KE._TINY)
    total = torch.zeros_like(top + hp_lo)
    for r in ROLLS:
        total = total + ((top * (r / 100.0) - hp_lo) / width).clamp(0.0, 1.0)
    return total / float(N_ROLLS)


def _k1(m: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, tiny: float) -> torch.Tensor:
    """1 + (|top| + |lo|) / w in float64 — the condition number the bound scales with."""
    m, lo, hi = m.to(F64), lo.to(F64), hi.to(F64)
    w = (hi - lo).clamp_min(tiny)
    return 1.0 + ((m / MEAN_ROLL).abs() + lo.abs()) / w


def _cat(parts: List[Grid]) -> Grid:
    return (torch.cat([p[0].reshape(-1) for p in parts]), torch.cat([p[1].reshape(-1) for p in parts]),
            torch.cat([p[2].reshape(-1) for p in parts]))


def _random_grid(n: int, seed: int) -> Grid:
    """Four regimes, seeded: OUR exact HP in HP points (±½ HP) and in fractions of max HP, THEIR percentage bin in HP
    points (and the full-HP ±½ HP case), and a wide signed box (negative damage, negative / inverted widths)."""
    g = torch.Generator().manual_seed(seed)

    def u(lo: float, hi: float, k: int = n) -> torch.Tensor:
        return torch.empty(k, dtype=F64).uniform_(lo, hi, generator=g)
    maxhp = torch.randint(100, 715, (n,), generator=g).to(F64)
    hp = torch.floor(u(0.0, 1.0) * maxhp) + 1.0
    ours_pts = (u(0.0, 2.5) * maxhp, hp - 0.5, hp + 0.5)
    ours_frac = (u(0.0, 3.0), hp / maxhp - 0.5 / maxhp, hp / maxhp + 0.5 / maxhp)
    p = torch.randint(1, 101, (n,), generator=g).to(F64)
    cur = p / 100.0 * maxhp
    full = p >= 100
    theirs = (u(0.0, 2.5) * maxhp, torch.where(full, cur - 0.5, cur - 0.01 * maxhp), torch.where(full, cur + 0.5, cur))
    lo_w = u(-20.0, 500.0)
    wide = (u(-50.0, 500.0), lo_w, lo_w + u(-2.0, 50.0))
    return _cat([ours_pts, ours_frac, theirs, wide])


def _adversarial_grid() -> Grid:
    """Steps EXACTLY at the clamp edges (a term computed by the 16-roll sum's own expression lands on 0, or next to
    1), top = 0 (step 0), lo above every roll, hi below every roll, w at and below the clamp floor, negative damage,
    and a huge condition number (w at the floor with HP-sized operands)."""
    parts: List[Grid] = []
    t = torch.arange(1, 401, dtype=F64) * 0.5                          # 100-roll damages 0.5 … 200
    mean = t * MEAN_ROLL
    top = mean / MEAN_ROLL                                             # what the sum recomputes
    for r in ROLLS:
        edge0 = top * (r / 100.0)                                      # x_r == 0 exactly in the sum's arithmetic
        for w in (1.0, 3.71, 0.0037):
            parts.append((mean, edge0, edge0 + w))
            parts.append((mean, edge0 - w, edge0))                     # x_r == 1 up to the subtraction's rounding
    zero = torch.zeros(1, dtype=F64)
    for lo in (-3.0, -1.0, -0.5, -1e-300, 0.0, 1e-300, 0.3, 1.0, 5.0):
        for w in (1.0, 1e-7, 0.0, -1.0, 7.5):
            parts.append((zero, zero + lo, zero + lo + w))
    m = torch.linspace(0.0, 900.0, 61, dtype=F64)
    parts.append((m, m * 0 + 1e6, m * 0 + 1e6 + 1.0))                  # lo above every roll
    parts.append((m, m * 0 - 1e6, m * 0 - 1e6 + 1.0))                  # hi below every roll
    for d in (0.0, 1e-9, -3.0, KE._TINY, 2e-6):                        # w at / around / below the clamp floor
        parts.append((m, m * 0 + 300.0, m * 0 + 300.0 + d))
        parts.append((m, m * 0 + 0.37, m * 0 + 0.37 + d))
    parts.append((-m, m * 0 - 200.0, m * 0 - 190.0))                    # negative damage (top < 0)
    parts.append((-m, -m * 0.9, -m * 0.9 + 1.0))
    parts.append((m, m * 0.8, m * 0.8 + KE._TINY))                     # huge k1: HP-sized operands, w at the floor
    return _cat(parts)


def _exact(m: float, lo: float, hi: float, tiny: float) -> float:
    """The exact real value for the float inputs: Fraction arithmetic, r / 100 and ÷ 0.925 exact."""
    top = Fraction(m) / Fraction(37, 40)
    w = max(Fraction(hi) - Fraction(lo), Fraction(tiny))
    s = Fraction(0)
    for r in ROLLS:
        x = (top * Fraction(r, 100) - Fraction(lo)) / w
        s += min(max(x, Fraction(0)), Fraction(1))
    return float(s / N_ROLLS)


def _tiny(dtype: torch.dtype) -> float:
    return float(torch.tensor(KE._TINY, dtype=dtype).item())


# ============================================================================================== VALUES
@pytest.fixture(scope="module")
def grid64() -> Grid:
    return _cat([_random_grid(25_000, 7), _adversarial_grid()])


def test_the_closed_form_equals_the_16_roll_sum_in_float64_within_the_rounding_bound(grid64: Grid) -> None:
    """Every finite input of the dense seeded grid + the adversarial set: |closed − sum| ≤ 8 eps k1, and ≤ 1e-12 on
    the k1 ≤ 563 domain (the bound itself puts it there). Fails on a wrong count (an off-by-one run reads a whole term
    of error), a wrong mean index, a dropped high run, a mis-ordered sequence for top < 0, or a w-floor mismatch."""
    m, lo, hi = grid64
    b, c = _sum16(m, lo, hi), roll_ko_prob(m, lo, hi)
    k1 = _k1(m, lo, hi, _tiny(F64))
    assert torch.isfinite(b).all() and torch.isfinite(c).all()
    err = (c - b).abs()
    assert bool((err <= 8.0 * EPS64 * k1).all()), f"worst slack {float((err / (8.0 * EPS64 * k1)).max()):.3f}"
    dom = k1 <= K1_MAX_1E12
    assert int(dom.sum()) > 0.5 * m.numel()                            # non-vacuous: most of the grid is in it
    assert float(err[dom].max()) <= 1e-12


def test_both_spellings_meet_the_exact_rational_value(grid64: Grid) -> None:
    """Against the EXACT value (Fraction arithmetic on the float inputs), over every adversarial input and a
    deterministic stride of the random ones: each spelling within 6 eps k1 — the closed form is as accurate as the
    16-roll sum, not merely close to it."""
    m, lo, hi = grid64
    n_adv = _adversarial_grid()[0].numel()
    idx = torch.cat([torch.arange(0, m.numel() - n_adv, 37), torch.arange(m.numel() - n_adv, m.numel())])
    m, lo, hi = m[idx], lo[idx], hi[idx]
    tiny = _tiny(F64)
    ex = torch.tensor([_exact(float(a), float(b_), float(c_), tiny) for a, b_, c_ in zip(m, lo, hi)], dtype=F64)
    bound = 6.0 * EPS64 * _k1(m, lo, hi, tiny)
    assert bool(((_sum16(m, lo, hi) - ex).abs() <= bound).all())
    assert bool(((roll_ko_prob(m, lo, hi) - ex).abs() <= bound).all())


def test_float32_is_within_the_stated_bound(grid64: Grid) -> None:
    """The training dtype: the same bound at fp32's eps — |closed32 − sum32| ≤ 8 eps32 k1, and each within 6 eps32 k1
    of the exact value (the float64 sum on the same fp32 inputs, whose own error ≤ 6 eps64 k1 is added). k1 is read
    from the fp32 inputs. The summation ORDER differs (16 sequential adds vs count × mean), which is what the bound
    prices; it is not a tolerance chosen by measurement."""
    m, lo, hi = (t.to(torch.float32) for t in grid64)
    b, c = _sum16(m, lo, hi), roll_ko_prob(m, lo, hi)
    k1 = _k1(m, lo, hi, _tiny(torch.float32))
    ref = _sum16(m.to(F64), lo.to(F64), hi.to(F64))
    ref_err = 6.0 * EPS64 * k1
    assert bool(((c.to(F64) - b.to(F64)).abs() <= 8.0 * EPS32 * k1).all())
    assert bool(((c.to(F64) - ref).abs() <= 6.0 * EPS32 * k1 + ref_err).all())
    assert bool(((b.to(F64) - ref).abs() <= 6.0 * EPS32 * k1 + ref_err).all())


def test_degenerate_inputs_read_what_the_16_roll_sum_reads() -> None:
    """Spelled out: top = 0 reads clamp(−lo / w) (all 16 terms equal), lo above every roll reads 0, hi below every roll
    reads 1, a zero or negative width is the 1e-6 floor (a step), negative damage reverses the sequence (10 rolls at
    1, the 95-roll at ½) — each spelling equal to the hand value to rounding (÷ 0.925 is inexact)."""
    cases = [(0.0, -0.25, 0.75, 0.25), (0.0, 0.5, 1.5, 0.0), (0.0, -2.0, -1.0, 1.0), (0.0, -0.5, -0.5, 1.0),
             (92.5, 1e6, 1e6 + 1.0, 0.0), (92.5, -1e6, -1e6 + 1.0, 1.0), (92.5, 90.5, 90.5, 10.0 / 16.0),
             (92.5, 90.5, 89.5, 10.0 / 16.0), (-92.5, -95.5, -94.5, 10.5 / 16.0)]
    for m, lo, hi, want in cases:
        args = (torch.tensor(m, dtype=F64), torch.tensor(lo, dtype=F64), torch.tensor(hi, dtype=F64))
        assert float(_sum16(*args)) == pytest.approx(want, abs=1e-13), (m, lo, hi)
        assert float(roll_ko_prob(*args)) == pytest.approx(want, abs=1e-13), (m, lo, hi)
    # the width FLOOR is the sum's own (1e-6): a zero width with the 90-roll half a floor-width above lo reads ½ there
    mean = torch.tensor(92.5, dtype=F64)
    lo_t = (mean / MEAN_ROLL) * 0.9 - 0.5 * KE._TINY
    for fn in (_sum16, roll_ko_prob):
        assert float(fn(mean, lo_t, lo_t)) == pytest.approx(10.5 / 16.0, abs=1e-6), fn.__name__


# ========================================================================= CONTINUITY ACROSS A COUNT STEP
def test_the_sum_is_continuous_across_every_count_step(grid64: Grid) -> None:
    """`selection_sites` declares the two counts COUNT_CONTINUOUS: the term that changes class at a step sits ON the
    edge, so (a) the closed sum at the neighbouring count differs from it by that boundary term only (≤ its rounding
    when the term is computed on the edge), and (b) moving lo one ulp across an exact edge — where the count FLIPS —
    moves the value by ≤ 16 eps k1 (the true function moves ≤ ulp(lo) / w ≤ eps k1). Non-vacuous: the counts flip on
    hundreds of the adversarial edges."""
    m, lo, hi = _adversarial_grid()
    keep = torch.isfinite(m) & ((hi - lo) > 1e-3) & (m > 0)          # the on-edge lattice (w well above the floor)
    m, lo, hi = m[keep], lo[keep], hi[keep]
    top = m / MEAN_ROLL
    w = (hi - lo).clamp_min(KE._TINY)
    x_start = torch.minimum((top * 0.85 - lo) / w, (top * 1.0 - lo) / w)
    step = top.abs() * 0.01 / w
    n0, n1 = run_counts(x_start, step)
    k1 = _k1(m, lo, hi, _tiny(F64))
    base = closed_sum(x_start, step, n0, n1)
    for d0, d1 in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        a0, a1 = (n0 + d0).clamp(0, N_ROLLS), (n1 + d1).clamp(0, N_ROLLS)
        ok = a1 >= a0
        # the boundary term the neighbouring count moves between runs, and its distance from the edge it crosses
        j = torch.minimum(n0, a0) if d0 != 0.0 else torch.minimum(n1, a1)
        xj = x_start + step * j
        edge_gap = (xj if d0 != 0.0 else xj - 1.0).abs()
        moved = (closed_sum(x_start, step, a0, a1) - base).abs()
        assert bool((moved[ok] <= edge_gap[ok] + 32.0 * EPS64 * N_ROLLS * k1[ok]).all())
    flips = 0
    for direction in (float("inf"), float("-inf")):
        lo2 = torch.nextafter(lo, torch.full_like(lo, direction))
        w2 = (hi - lo2).clamp_min(KE._TINY)
        xs2 = torch.minimum((top * 0.85 - lo2) / w2, (top * 1.0 - lo2) / w2)
        a0, a1 = run_counts(xs2, top.abs() * 0.01 / w2)
        flips += int(((a0 != n0) | (a1 != n1)).sum())
        dv = (roll_ko_prob(m, lo2, hi) - roll_ko_prob(m, lo, hi)).abs()
        assert bool((dv <= 16.0 * EPS64 * k1).all()), float((dv / (16.0 * EPS64 * k1)).max())
    assert flips >= 100, f"only {flips} count flips — the continuity check saw no step"


# ============================================================================================== GRADIENTS
def _grads(fn: Any, m: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    a, b, c = (t.detach().clone().requires_grad_(True) for t in (m, lo, hi))
    fn(a, b, c).sum().backward()
    return a.grad, b.grad, c.grad


def _off_kink(m: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Every term at least 64 eps k1 from 0 and from 1 (the radius inside which a count can flip by rounding), and the
    width off its clamp floor by the same relative radius — the rule that excludes a kink-adjacent input."""
    tiny = _tiny(F64)
    k1 = _k1(m, lo, hi, tiny)
    w = (hi - lo).clamp_min(tiny)
    rho = 64.0 * EPS64 * k1
    top = m / MEAN_ROLL
    ok = ((hi - lo) - tiny).abs() > rho * tiny
    for r in ROLLS:
        x = (top * (r / 100.0) - lo) / w
        ok &= (x.abs() > rho) & ((x - 1.0).abs() > rho)
    return ok


def test_gradients_equal_the_16_roll_sums_off_the_kinks(grid64: Grid) -> None:
    """d/d(mean damage, lo, hi) of the closed form == the 16-roll sum's (float64 autograd), on every grid input at
    least 64 eps k1 from a kink (excluded by rule), incl. top = 0 (`torch.minimum`'s tie splits 0.85 / 1.0 into the
    mean roll's 0.925; |top|'s derivative is 0) and top < 0. Bound: 64 eps k1 / w (the closed form forms x_start and
    step · mean separately, a cancellation of two O(k1) terms, ÷ w)."""
    m, lo, hi = grid64
    keep = _off_kink(m, lo, hi)
    assert int(keep.sum()) > 0.5 * m.numel()
    m, lo, hi = m[keep], lo[keep], hi[keep]
    gb, gc = _grads(_sum16, m, lo, hi), _grads(roll_ko_prob, m, lo, hi)
    w = (hi - lo).clamp_min(_tiny(F64))
    tol = 64.0 * EPS64 * _k1(m, lo, hi, _tiny(F64)) / w
    for name, a, b in zip(("mean_dmg", "hp_lo", "hp_hi"), gb, gc):
        assert bool(((a - b).abs() <= tol).all()), (name, float(((a - b).abs() / tol).max()))
    z = m == 0
    assert int(z.sum()) > 0 and bool((gb[0][z] != 0).any())          # top = 0 with a live middle run is covered


def test_at_an_exact_kink_each_spelling_returns_a_valid_subgradient() -> None:
    """A term EXACTLY at 0 (in the 16-roll sum's arithmetic): the sum returns torch.clamp's INCLUSIVE subgradient (the
    term passes its gradient); the closed form returns the inclusive one or the one-sided derivative from the side its
    count's rounding put the term on — pinned as: the sum's gradient == the inclusive derivative, the closed form's ∈
    {left, right}, where left / right are the sum's gradient one tenth of a term's spacing to either side (no other
    kink in between)."""
    mean = torch.tensor(92.5, dtype=F64)
    top = mean / MEAN_ROLL
    seen = set()
    for r in ROLLS[1:-1]:
        lo = top * (r / 100.0)
        hi = lo + 0.25                                                # the next roll is 4 spacings away
        g_sum = _grads(_sum16, mean, lo, hi)[0]
        g_closed = _grads(roll_ko_prob, mean, lo, hi)[0]
        nudge = 0.1 * float(top) / 100.0
        g_left = _grads(_sum16, mean, lo - nudge, hi - nudge)[0]     # the edge term inside the run
        g_right = _grads(_sum16, mean, lo + nudge, hi + nudge)[0]    # the edge term below 0
        assert float(g_sum) == pytest.approx(float(g_left), rel=1e-12)
        close = [s for s, g in (("in", g_left), ("out", g_right)) if abs(float(g_closed - g)) <= 1e-12 * abs(float(g))]
        assert close, (r, float(g_closed), float(g_left), float(g_right))
        seen.update(close)
    assert seen, "no kink was evaluated"


# ============================================================================================ THE CRIT MIX
def test_ko_given_hit_mixes_both_calls_through_the_closed_form() -> None:
    """The crit mix prices the no-crit AND the crit roll through `roll_ko_prob` (bit-equal to the explicit mix), and
    is within the bound of the same mix over the ORACLE (the mix is a convex combination, so the bound carries)."""
    g = torch.Generator().manual_seed(4)
    m = torch.rand(512, generator=g, dtype=F64) * 300.0
    crit = 2.0 * m * (1.0 + torch.rand(512, generator=g, dtype=F64))
    lo = torch.rand(512, generator=g, dtype=F64) * 250.0
    hi = lo + 1.0
    c = torch.where(torch.rand(512, generator=g) > 0.5, 1.0 / 8.0, 1.0 / 16.0).to(F64)
    want = (1.0 - c) * roll_ko_prob(m, lo, hi) + c * roll_ko_prob(crit, lo, hi)
    assert torch.equal(ko_given_hit(m, crit, lo, hi, c), want)
    base = (1.0 - CRIT_P_BASE) * roll_ko_prob(m, lo, hi) + CRIT_P_BASE * roll_ko_prob(crit, lo, hi)
    assert torch.equal(ko_given_hit(m, crit, lo, hi), base)
    oracle = (1.0 - c) * _sum16(m, lo, hi) + c * _sum16(crit, lo, hi)
    k1 = torch.maximum(_k1(m, lo, hi, _tiny(F64)), _k1(crit, lo, hi, _tiny(F64)))
    assert bool(((ko_given_hit(m, crit, lo, hi, c) - oracle).abs() <= 8.0 * EPS64 * k1).all())


# =============================================================================================== COMPILE
def test_the_closed_form_compiles_fullgraph_with_no_break() -> None:
    """`torch.compile(fullgraph=True)` (CPU, dynamo's `eager` backend: graphs and breaks are dynamo's, independent of
    the backend) of the closed form and of the crit mix through it, forward and backward: a graph break is a compile
    ERROR under fullgraph, and the compiled values / gradients equal eager's bit for bit (the eager backend runs the
    same kernels)."""
    torch._dynamo.reset()
    try:
        f = torch.compile(roll_ko_prob, fullgraph=True, backend="eager")
        mix = torch.compile(ko_given_hit, fullgraph=True, backend="eager")
        g = torch.Generator().manual_seed(9)
        m = (torch.rand(4, 6, 5, generator=g) * 300.0).requires_grad_(True)
        lo = torch.rand(4, 6, 1, generator=g) * 250.0
        hi = lo + 1.0
        out = f(m, lo, hi)
        assert torch.equal(out, roll_ko_prob(m, lo, hi))
        (gc,) = torch.autograd.grad(out.sum(), m)
        (ge,) = torch.autograd.grad(roll_ko_prob(m, lo, hi).sum(), m)
        assert torch.equal(gc, ge)
        c = torch.full((4, 1, 5), 1.0 / 16.0)
        assert torch.equal(mix(m, 2.0 * m, lo, hi, c), ko_given_hit(m, 2.0 * m, lo, hi, c))
    finally:
        torch._dynamo.reset()


# =========================================================================================== THE MODEL
_POLICIES: Dict[str, Any] = {}
#: Where each exact-KO site runs: the END-STATE arm (the closing test's overlay) prices the op's KO and move
#: resolution's Pursuit / Substitute; `--move-resolution on` retires intent_threshold, so its Substitute break is
#: reached on the PRODUCTION surface + `--ko-ramp exact` alone.
SURFACE_SITES = {"endstate": {"_ko_exact", "_exact_ko_ours"}, "production": {"_ko_exact", "sub_break_given_hit"}}


def _policy(surface: str) -> Any:
    """A real SB3 policy on ``surface`` ('endstate' = the arm's overlay; 'production') at `--ko-ramp exact`."""
    if surface not in _POLICIES:
        from agents.model.endstate_facts_test import _toggles
        from agents.model.identity_init_test import _build_real_policy
        from main.train.arch_arms import arm_overlay
        over = dict(arm_overlay("endstate")) if surface == "endstate" else {}
        over["ko_ramp"] = "exact"
        _POLICIES[surface] = _build_real_policy(**_toggles(**over))[0].policy
    return _POLICIES[surface]


@pytest.fixture(scope="module")
def rows() -> torch.Tensor:
    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    obs, _mask = load_parity_rows(Gen3ObservationEncoder(load_mappings()).dimension)
    return torch.as_tensor(obs)


@pytest.mark.parametrize("surface", sorted(SURFACE_SITES))
def test_every_ko_site_of_the_forward_meets_the_oracle_within_the_bound(
        surface: str, rows: torch.Tensor, monkeypatch: pytest.MonkeyPatch) -> None:
    """On the compile-parity fixture's real rows at `--ko-ramp exact`: every P(KO) call goes through `roll_ko_prob`
    (a spy on the module global `ko_given_hit` reads at call time), from every site the surface runs — the op's
    `_ko_exact` (`_rolls` and the Choice-Band `ko_cb`), move resolution's `_exact_ko_ours` (Pursuit, Substitute;
    END-STATE) and intent_threshold's `sub_break_given_hit` (PRODUCTION + the flag) — and on each call's REAL operands
    the value is within 8 eps32 k1 of the 16-roll ORACLE. Fails if a site prices its KO by anything else."""
    real = KE.roll_ko_prob
    calls: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def spy(m: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        out = real(m, lo, hi)
        calls.append((sys._getframe(2).f_code.co_name, m.detach(), lo.detach(), hi.detach(), out.detach()))
        return out

    monkeypatch.setattr(KE, "roll_ko_prob", spy)
    fe = _policy(surface).features_extractor
    with torch.no_grad():
        fe({"observation": rows})
    sites = {c[0] for c in calls}
    assert sites == SURFACE_SITES[surface], sites
    for site, m, lo, hi, out in calls:
        b = _sum16(m, lo, hi)
        k1 = _k1(*torch.broadcast_tensors(m, lo, hi), _tiny(m.dtype))
        assert bool(((out.to(F64) - b.to(F64)).abs() <= 8.0 * EPS32 * k1).all()), site


def test_the_tie_recorder_is_clean_on_the_endstate_arm(rows: torch.Tensor) -> None:
    """K9(b): the END-STATE forward at `--ko-ramp exact` runs NO undeclared discrete op, and the closed form's counts
    (float floor / ceil, declared COUNT_CONTINUOUS) are never a recorded site — they exclude no row."""
    from agents.training.rust_rollout.tie_margins import TieMargins
    fe = _policy("endstate").features_extractor
    rec = TieMargins(rows.shape[0])
    with torch.no_grad(), rec:
        fe({"observation": rows})
    rec.check()
    assert rec.sites_seen and not any(s.startswith("ko_exact.py") for s in rec.sites_seen), rec.sites_seen


# ===================================================================================== THE RETIRED VALUE
def test_the_retired_exact_closed_value_is_refused_with_its_reason_everywhere() -> None:
    """`--ko-ramp exact_closed` (legal only at 6e0a1a7a) is refused WITH its reason at every door a value can come
    through: the parser (before argparse's bare "invalid choice"), a `model_config.json` (`_migrate_config`), a zip's
    pickled extractor kwargs (`sanitize_dead_extractor_kwargs`) and the constructor — never a KeyError or a bare
    "one of". `exact` and `ramp` pass every door."""
    import argparse

    from agents.model.model_version import ModelVersionError
    from agents.model.model_version.migrations import _migrate_config
    from agents.model.model_version.retired_levers import RETIRED_VALUES, refuse_retired_values
    from agents.model.snapshot import sanitize_dead_extractor_kwargs
    from main.train.parser.clean_world import add_clean_world_flags

    assert [(r.field, r.value) for r in RETIRED_VALUES] == [("ko_ramp", "exact_closed")]
    p = argparse.ArgumentParser(exit_on_error=False)
    add_clean_world_flags(p)
    with pytest.raises(argparse.ArgumentError, match="exact_closed was RETIRED .*pass --ko-ramp exact"):
        p.parse_args(["--ko-ramp", "exact_closed"])
    assert p.parse_args(["--ko-ramp", "exact"]).ko_ramp == "exact"
    with pytest.raises(ModelVersionError, match="exact_closed was RETIRED"):
        refuse_retired_values({"ko_ramp": "exact_closed"})
    with pytest.raises(ModelVersionError, match="exact_closed was RETIRED"):
        _migrate_config({"config_version": 154, "ko_ramp": "exact_closed"})
    with pytest.raises(ModelVersionError, match="exact_closed was RETIRED"):
        sanitize_dead_extractor_kwargs({"ko_ramp": "exact_closed"})
    for ok in ("exact", "ramp"):
        refuse_retired_values({"ko_ramp": ok})
        sanitize_dead_extractor_kwargs({"ko_ramp": ok})
    from agents.model.endstate_facts_test import _toggles
    from agents.model.identity_init_test import _build_real_policy
    with pytest.raises(ValueError, match="exact_closed was RETIRED"):
        _build_real_policy(**_toggles(ko_ramp="exact_closed"))


def test_the_history_doc_lists_exactly_the_retired_values() -> None:
    """`designs/deleted_flags.md` §4 (the flag lives, a VALUE is gone) lists exactly `RETIRED_VALUES`, each with a
    citation — the doc and the refusal table cannot drift."""
    import re

    from agents.model.model_version.retired_levers import RETIRED_VALUES
    from utils.paths import repo_path
    text = repo_path("designs", "deleted_flags.md").read_text(encoding="utf-8")
    sec = text[text.index("## 4. RETIRED VALUES"):]
    sec = sec[:sec.index("\n---")] if "\n---" in sec else sec
    rows = re.findall(r"^\| --([a-z0-9-]+) ([A-Za-z0-9_]+) \| (.+?) \|", sec, flags=re.M)
    assert {(f, v) for f, v, _c in rows} == {(r.flag, r.value) for r in RETIRED_VALUES}
    assert all(re.search(r"gen3_[a-z0-9_]+|20\d\d-\d\d-\d\d|[0-9a-f]{8}", c) for _f, _v, c in rows)
