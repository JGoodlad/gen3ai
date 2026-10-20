"""FP32_ALT — X5's τ solve (`gen3_tau_fp32_alt_v1`, config v160; `designs/endstate/design_x5_belief_tokens.md` §3.2).

What each test pins, and what reverting it would break:

* ACCURACY WHATEVER THE SLOPE: on saturated rows (one or three candidates 12–150 nats clear of the rest, k = 1 / 3:
  the X26 A3 / A6 τ class and far past it) the default ``fp32_alt`` solve lands within 2 fp32 ulp of the root,
  against an independent float64 reference (numpy, a different library and loop) cross-checked once against 60-digit
  mpmath. PRECONDITION, asserted: each row past a 23-nat lead IS ill-conditioned — the naive fp32 band ulp(k) / S'(τ)
  exceeds 1e-3 and the naive fp32 solve (``fp32_naive``) misses by > 1e-4. Reverting the split residual (to the naive
  sum in fp32) fails it;
* ONE τ UNDER ANY SUMMATION ORDER (the two-program emulation of the A3 diagnosis): the naive fp32 sum spreads over 32
  candidate orders by > 1e-3 (asserted), ``fp32_alt`` by <= 2 ulp, ``fp64`` by <= 1 ulp;
* THE REAL SECOND PROGRAM, CPU AND CUDA (`slow`): inductor's compiled fp32_alt τ against eager on the saturated rows —
  within 2 ulp on the CPU (C++ kernel), within `CUDA_TAU_MAX_ULPS` on CUDA (Triton: the production rollout's kernel; a GPU
  test, it skips naming why without a free card). The CUDA case is the GPU-window cell `tau_cuda_parity`'s first half
  (`REVIEW.md` F2); `compiled_vs_eager_tau` is shared with that cell;
* NO FLOAT64 ON THE fp32 PATH: the traced graph of the ``fp32_alt`` solve on fp32 scores holds no float64 tensor (the
  GPU cost reason), the ``fp64`` one does (the probe has teeth); both trace as ONE graph equal to eager;
* THE RECORD: one default everywhere (``TAU_SOLVE_DEFAULT``, the ModelVersion field, the extractor kwarg, the registry
  row, the flag's choices); a pre-v160 config migrates to fp32_alt; a flagless fork / resume inherits the recorded
  solve and a typed one wins; ``fp32_naive`` is refused by every model surface;
* EVERY CONSTRUCTION RUNS THE MODEL'S SOLVE: on the production learner (the K9 golden's build) each
  ``fixed_size_tau`` call of a forward — the species set, the active's move group, the per-slot move presence, OTHER's
  mixture — and the BeliefHead set BCE receive the builder's solve, and the post-load setter switches all of them.
  A call site that forgot to pass it would run the default under ``fp64`` and fail.
"""
from __future__ import annotations

import inspect
import math
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from agents.model import hypothesis_set as HS
from agents.model.extractor_compiles_test import _cuda_skip_reason
from agents.model.hypothesis_set import TAU_SOLVE_DEFAULT, TAU_SOLVE_NAIVE, TAU_SOLVES, fixed_size_tau

N = 400


def _saturated(lead: float, k: int = 1, seed: int = 62) -> torch.Tensor:
    """[1, N] fp32 scores: ``k`` candidates at 0, 0.5, … and the other N − k drawn N(−lead, 2), shifted so the best of
    them sits exactly ``lead`` nats below 0 — at the root the top candidates' π are ≈ 1 − e^{−lead/2}."""
    g = torch.Generator().manual_seed(seed)
    a = -lead + 2.0 * torch.randn(N, generator=g, dtype=torch.float64)
    a = a - (a.max() + lead)
    for j in range(k):
        a[17 + j] = 0.5 * j
    return a.float().unsqueeze(0)


def _ref_tau(s32: torch.Tensor, k: int) -> float:
    """The root of Σσ(s + τ) = k for the fp32 scores, in numpy float64: 200 bisection steps on the SPLIT residual
    (relative-accurate terms against an exact integer — float64 resolves it to ~1e-15 for any lead under ~1400 nats).
    Shares no code with `fixed_size_tau`."""
    s = s32.double().numpy().reshape(-1)
    lo, hi = -400.0, 400.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        x = s + mid
        t = 1.0 / (1.0 + np.exp(np.abs(x)))                  # σ(−|x|)
        resid = float(np.where(x > 0, -t, t).sum())
        if resid > k - int((x > 0).sum()):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _ulp32(x: float) -> float:
    return math.ulp(float(np.float32(abs(x)))) * 2.0 ** 29     # float64's ulp of that value x 2^29 = fp32's


def _band(s32: torch.Tensor, tau: float, k: int) -> float:
    pi = torch.sigmoid(s32.double() + tau)
    return float(torch.finfo(torch.float32).eps) * k / float((pi * (1 - pi)).sum())


def test_the_float64_reference_is_the_exact_root():
    """The reference's own check: 60-digit mpmath on one 100-nat row (where float64's NAIVE sum is already wrong)."""
    mpmath = pytest.importorskip("mpmath")
    s32 = _saturated(100.0)
    with mpmath.workdps(60):
        xs = [mpmath.mpf(float(v)) for v in s32.reshape(-1)]
        lo, hi = mpmath.mpf(-400), mpmath.mpf(400)
        for _ in range(110):
            mid = (lo + hi) / 2
            if sum(1 / (1 + mpmath.exp(-(x + mid))) for x in xs) > 1:
                hi = mid
            else:
                lo = mid
        exact = float((lo + hi) / 2)
    assert abs(_ref_tau(s32, 1) - exact) < 1e-9


@pytest.mark.parametrize("k", (1, 3))
@pytest.mark.parametrize("lead", (12.0, 23.0, 40.0, 100.0, 150.0))
def test_fp32_alt_lands_within_two_ulp_of_the_root_whatever_the_slope(lead: float, k: int):
    s32 = _saturated(lead, k)
    cand, kk = torch.ones(1, N, dtype=torch.bool), torch.tensor([k])
    ref = _ref_tau(s32, k)
    if lead >= 23.0:          # PRECONDITION: the row is ill-conditioned for the naive fp32 sum (the τ class)
        assert _band(s32, ref, k) > 1e-3
        assert abs(float(fixed_size_tau(s32, cand, kk, solve=TAU_SOLVE_NAIVE)) - ref) > 1e-4
    got = fixed_size_tau(s32, cand, kk)                               # the DEFAULT solve
    assert got.dtype == torch.float32
    assert abs(float(got) - ref) <= 2 * _ulp32(ref), (lead, k, float(got), ref)


def test_fp32_alt_is_the_default_and_fp64_is_still_the_float64_solve(monkeypatch):
    """T62's solve stays available bit for bit: ``fp64`` on fp32 scores IS the float64 solve rounded once."""
    assert TAU_SOLVE_DEFAULT == "fp32_alt" and TAU_SOLVES == ("fp32_alt", "fp64")
    s32 = _saturated(23.0)
    cand, kk = torch.ones(1, N, dtype=torch.bool), torch.tensor([1])
    t64 = fixed_size_tau(s32, cand, kk, solve="fp64")
    assert t64.dtype == torch.float32
    assert torch.equal(t64, fixed_size_tau(s32.double(), cand, kk, solve="fp64").float())
    with pytest.raises(ValueError, match="unknown solve"):
        fixed_size_tau(s32, cand, kk, solve="bf16")


@pytest.mark.parametrize("solve,max_ulps", (("fp32_alt", 2), ("fp64", 1)))
def test_two_programs_land_on_one_tau_under_any_summation_order(solve: str, max_ulps: int):
    """THE τ-class mechanism (X26 A3 / A6): programs that sum the constraint in different orders (T2's compiled
    rollout, the eager learner) — emulated by permuting the candidate axis — must land on the same τ."""
    s32 = _saturated(23.0)
    cand, kk = torch.ones(1, N, dtype=torch.bool), torch.tensor([1])
    perms = [torch.randperm(N, generator=torch.Generator().manual_seed(i)) for i in range(32)]

    def spread(how: str) -> float:
        taus = torch.cat([fixed_size_tau(s32[:, p], cand, kk, solve=how) for p in perms])
        return float(taus.max() - taus.min())

    assert spread(TAU_SOLVE_NAIVE) > 1e-3                                       # PRECONDITION: the naive sum spreads
    assert spread(solve) <= max_ulps * _ulp32(_ref_tau(s32, 1))


def _traced_dtypes(solve: str, s32: torch.Tensor, cand: torch.Tensor, k: torch.Tensor):
    import torch._dynamo as dynamo
    graphs: List[Any] = []

    def backend(gm: Any, example_inputs: Any) -> Any:
        graphs.append(gm)
        return gm.forward

    fn = torch.compile(lambda a, b, c: fixed_size_tau(a, b, c, solve=solve), backend=backend, fullgraph=True,
                       dynamic=False)
    try:
        got = fn(s32, cand, k)
    finally:
        dynamo.reset()
    dts = {n.meta["example_value"].dtype for g in graphs for n in g.graph.nodes
           if isinstance(n.meta.get("example_value"), torch.Tensor)}
    return got, dts, len(graphs)


@pytest.mark.parametrize("solve", TAU_SOLVES)
def test_each_solve_traces_as_one_graph_equal_to_eager_and_fp32_alt_holds_no_float64(solve: str):
    """`--compile-trainer` and T2 compile the forward with `fixed_size_tau` inside: ``fullgraph=True`` raises on a break.
    The fp32_alt graph on fp32 scores holds NO float64 tensor (float64 runs at ~1/64 rate on consumer GPUs — the
    reason FP32_ALT exists); the fp64 graph does (the dtype probe sees a float64 op when there is one)."""
    s32 = _saturated(23.0).repeat(3, 1)
    s32[1] = s32[1].flip(-1)
    cand, k = torch.ones(3, N, dtype=torch.bool), torch.tensor([1, 1, 5])
    got, dts, n_graphs = _traced_dtypes(solve, s32, cand, k)
    assert n_graphs == 1 and got.dtype == torch.float32
    assert torch.equal(got, fixed_size_tau(s32, cand, k, solve=solve))
    assert (torch.float64 in dts) == (solve == "fp64"), dts


def _tau_class_rows() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The saturated rows the compiled-vs-eager test judges: leads 23 / 40 / 100 nats, k = 1 and 3 (CPU tensors)."""
    rows = torch.cat([_saturated(lead, k) for lead in (23.0, 40.0, 100.0) for k in (1, 3)])
    return rows, torch.ones(rows.shape[0], N, dtype=torch.bool), torch.tensor([1, 3] * 3)


def compiled_vs_eager_tau(rows: torch.Tensor, cand: torch.Tensor, k: torch.Tensor, device: str,
                          solve: str = "fp32_alt") -> tuple[torch.Tensor, torch.Tensor]:
    """``(compiled, eager)`` fp32 τ of ``fixed_size_tau`` on ``device``, both returned on the CPU: inductor's fused
    reduction (C++ on the CPU, Triton on CUDA, ``dynamic=False``) against ATen's. SHARED by this file's test and by the
    GPU-window cell `tau_cuda_parity` (`research_state/measurements/gpu_window_battery_2026-10-20/`), which runs it on
    X26's captured τ-class rows too."""
    import torch._dynamo as dynamo

    from agents.model.inductor_compile import compile_on
    rows, cand, k = rows.to(device), cand.to(device), k.to(device)
    fn = compile_on(lambda a, b, c: fixed_size_tau(a, b, c, solve=solve), device=device, backend="inductor",
                    dynamic=False)
    try:
        comp = fn(rows, cand, k)
    finally:
        dynamo.reset()
    return comp.cpu(), fixed_size_tau(rows, cand, k, solve=solve).cpu()


#: THE CUDA BAR (preregistered before any CUDA datum, `tau_fp32_alt_2026-10-20/REVIEW.md` F2): the compiled Triton τ is
#: within 4 fp32 ulp of ATen's. The CPU bar is 2 (measured 0 on every violator). The proof's worst case is 2c <= 4.8e-5
#: (F3) for ANY summation order, so a CUDA miss between 4 ulp (~8e-6) and that is a HOLD to read, not a refutation.
CUDA_TAU_MAX_ULPS = 4
_CUDA_SKIP = _cuda_skip_reason()
_TAU_RECORD_ENV = "TAU_CUDA_PARITY_RECORD"


def _record_cuda_tau(comp: torch.Tensor, eager: torch.Tensor, ulps: List[float], max_ulps: int) -> None:
    """The GPU-window cell reads this: when ``$TAU_CUDA_PARITY_RECORD`` names a file, the CUDA case writes its numbers
    there BEFORE it asserts (a failing run keeps its record)."""
    import json
    import os
    path = os.environ.get(_TAU_RECORD_ENV)
    if not path:
        return
    d = (comp.double() - eager.double()).abs()
    with open(path, "w") as fh:
        json.dump({"test": "fp32_alt_compiled_by_inductor_matches_eager_on_the_tau_class_row", "device": "cuda",
                   "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": int(d.numel()),
                   "bar_ulps": max_ulps, "max_abs_dtau": float(d.max()), "max_ulps": max(ulps),
                   "per_row_abs_dtau": [float(x) for x in d], "per_row_ulps": ulps,
                   "tau_eager": [float(x) for x in eager]}, fh, indent=1)


@pytest.mark.slow
@pytest.mark.parametrize("device,max_ulps", (
    ("cpu", 2),
    pytest.param("cuda", CUDA_TAU_MAX_ULPS, marks=pytest.mark.skipif(_CUDA_SKIP is not None, reason=_CUDA_SKIP or ""))))
def test_fp32_alt_compiled_by_inductor_matches_eager_on_the_tau_class_row(device: str, max_ulps: int):
    """A REAL second program (inductor's fused reduction against ATen's), not the permutation emulation: on the
    saturated rows the compiled fp32_alt τ is within ``max_ulps`` ulp of eager (the naive sum's gap was up to 1.7e-2 on
    X26's violators, `research_state/measurements/tau_fp32_alt_2026-10-20/`). The CPU case is the C++ kernel; the CUDA
    case (GPU tier: `GEN3AI_TEST_ALLOW_GPU=1` under a lease, ~2 min) is the PRODUCTION pair's Triton kernel, which the
    review (F2) found unmeasured. A skip names why (no card, or someone else on it) and is NOT a pass."""
    rows, cand, k = _tau_class_rows()
    comp, eager = compiled_vs_eager_tau(rows, cand, k, device)
    ulps = [abs(float(comp[i]) - float(eager[i])) / _ulp32(float(eager[i])) for i in range(rows.shape[0])]
    if device == "cuda":
        _record_cuda_tau(comp, eager, ulps, max_ulps)
    for i, u in enumerate(ulps):
        assert u <= max_ulps, (i, u)


# ------------------------------------------------------------- the review's pins (REVIEW.md, 2026-10-20)
def _sparse_saturated(n: int, k: int, gap: float) -> torch.Tensor:
    """[1, N] fp32 scores with only ``n`` candidates (the first n columns; the move vocabulary's shape): ``k`` of
    them at 0, 0.25, … and the other ``n − k`` exactly ``gap`` nats below 0. Columns ≥ n are filler (masked)."""
    a = torch.zeros(N, dtype=torch.float32)
    a[:k] = 0.25 * torch.arange(k, dtype=torch.float32)
    a[k:n] = -gap
    return a.unsqueeze(0)


@pytest.mark.parametrize("n,k,gap", ((2, 1, 30.0), (4, 3, 30.0), (5, 4, 25.0), (8, 4, 40.0), (380, 6, 40.0)))
def test_fp32_alt_is_accurate_on_small_and_full_candidate_sets_with_k_up_to_six(n: int, k: int, gap: float):
    """THE CLAIM "WHATEVER THE SLOPE" on the shapes the k = 1 / 3 dense rows above do not reach: the move group's
    SMALL candidate sets (n = 2 … 8 of 400 columns, k_m = n − 1 or 4: most terms in H, the integer k − |H| near 0)
    and a species-shaped k = 6 row (380 of 400 candidates). PRECONDITION, asserted: the naive fp32 sum misses each by
    > 1e-3 (measured 1.2e-2 – 4.9 nats). Reverting the split residual fails it."""
    s32 = _sparse_saturated(n, k, gap)
    cand = torch.zeros(1, N, dtype=torch.bool)
    cand[0, :n] = True
    kk = torch.tensor([k])
    ref = _ref_tau(s32[:, :n], k)                     # the reference sees ONLY the candidates
    assert abs(float(fixed_size_tau(s32, cand, kk, solve=TAU_SOLVE_NAIVE)) - ref) > 1e-3
    got = fixed_size_tau(s32, cand, kk)
    assert abs(float(got) - ref) <= 2 * _ulp32(ref), (n, k, float(got), ref)


@pytest.mark.parametrize("solve", TAU_SOLVES)
def test_a_non_candidate_never_reaches_tau_whatever_its_score(solve: str):
    """A non-candidate's score is never read (module docstring: "a non-candidate is −inf"): NaN, ±inf, ±1e30 or the
    masked key's −1e9 in the masked columns give the SAME τ, bit for bit, as zeros there. The fp32_alt loop replaces
    them by −inf BEFORE x = a + τ (so NaN − inf never forms); reverting that mask to the raw scores fails it."""
    base = torch.cat([_saturated(23.0), _saturated(40.0, 3), _sparse_saturated(8, 4, 40.0)])
    cand = torch.ones(3, N, dtype=torch.bool)
    cand[:, 100::7] = False                            # 43 masked columns per row, interleaved with the candidates
    cand[2, 8:] = False
    k = torch.tensor([1, 3, 4])
    clean = torch.where(cand, base, torch.zeros(()))
    want = fixed_size_tau(clean, cand, k, solve=solve)
    assert bool(torch.isfinite(want).all())
    for junk in (math.nan, math.inf, -math.inf, 1e30, -1e30, -1e9):
        dirty = torch.where(cand, base, torch.full((), junk))
        assert torch.equal(fixed_size_tau(dirty, cand, k, solve=solve), want), junk


@pytest.mark.parametrize("k", (1, 3))
def test_bf16_scores_are_solved_in_fp32_and_only_tau_is_rounded(k: int):
    """``solve_dt = promote(scores.dtype, float32)``: a bf16 score row is solved in fp32 and τ rounded to bf16 ONCE,
    exactly the fp32 solve of the upcast scores, rounded (no production caller passes bf16 today — the trunk's bf16
    region ends before δ_θ and every belief head — but the docstring promises it). Solving in bf16 fails it."""
    sb = torch.cat([_saturated(23.0, k), _saturated(40.0, k, seed=7)]).bfloat16()
    cand, kk = torch.ones(2, N, dtype=torch.bool), torch.tensor([k, k])
    got = fixed_size_tau(sb, cand, kk)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, fixed_size_tau(sb.float(), cand, kk).bfloat16())


# ---------------------------------------------------------------------------------------------------- the record
def test_one_default_everywhere_and_the_naive_sum_is_refused_by_every_model_surface():
    from agents.model.extractor_build import ExtractorBuild
    from agents.model.flag_registry import BY_NAME, Klass, Tier
    from agents.model.model_version.fields import ModelVersionFields
    from main.train.parser import build_parser

    row = BY_NAME["tau_solve"]
    assert (row.default, row.tier, row.klass) == (TAU_SOLVE_DEFAULT, Tier.CLI, Klass.TRAINING_COEF)
    assert ModelVersionFields.__dataclass_fields__["tau_solve"].default == TAU_SOLVE_DEFAULT
    init = inspect.signature(ExtractorBuild.__init__)
    assert init.parameters["tau_solve"].default == TAU_SOLVE_DEFAULT
    p = build_parser()
    assert p.parse_args([]).tau_solve is None and p.parse_args(["--tau-solve", "fp64"]).tau_solve == "fp64"
    for bad in (TAU_SOLVE_NAIVE, "float64"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--tau-solve", bad])
        with pytest.raises(ValueError, match="measurement baseline only"):
            HS.check_tau_solve(bad)


def test_a_pre_v160_config_migrates_to_fp32_alt_and_a_recorded_solve_is_kept():
    from agents.model.model_version.constants import MODEL_CONFIG_VERSION
    from agents.model.model_version.migrations import _migrate_config
    assert MODEL_CONFIG_VERSION >= 160
    old = _migrate_config({"config_version": 159})
    assert old["tau_solve"] == "fp32_alt" and old["config_version"] >= 160
    assert _migrate_config({"config_version": 159, "tau_solve": "fp64"})["tau_solve"] == "fp64"


def test_a_flagless_fork_INHERITS_the_recorded_solve_and_a_typed_one_wins():
    import main.train.config as cfg
    from agents.model.model_version import ModelVersion
    from agents.model.resume_reapply import SETTERS, resolved_run_level
    from main.train.config import inherit_saved_flag
    from main.train.parser import build_parser
    saved = ModelVersion.__new__(ModelVersion)
    saved.tau_solve = "fp64"
    args = build_parser().parse_args([])
    assert inherit_saved_flag(args, saved, "tau_solve", "fp32_alt") is True and args.tau_solve == "fp64"
    typed = build_parser().parse_args(["--tau-solve", "fp32_alt"])
    assert inherit_saved_flag(typed, saved, "tau_solve", "fp32_alt") is False and typed.tau_solve == "fp32_alt"
    assert '_resolve("tau_solve", "fp32_alt")' in inspect.getsource(cfg)
    assert SETTERS["tau_solve"] == "set_tau_solve" and resolved_run_level(args)["tau_solve"] == "fp64"


# ------------------------------------------------------------------------------- every construction, one solve
class _SolveTap:
    """Records the ``solve`` every `fixed_size_tau` call of a block receives (looked up as a module global)."""

    def __init__(self) -> None:
        self.orig = HS.fixed_size_tau
        self.calls: List[str] = []

    def __enter__(self) -> "_SolveTap":
        def wrapped(*a: Any, **kw: Any) -> torch.Tensor:
            self.calls.append(kw.get("solve", "<positional or default>"))
            return self.orig(*a, **kw)
        HS.fixed_size_tau = wrapped        # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> None:
        HS.fixed_size_tau = self.orig      # type: ignore[assignment]


@pytest.fixture(scope="module")
def golden_learner():
    from agents.training import learner_golden as LG
    model = LG.build_learner()
    LG.load_buffer_into(model)
    return model


def _obs(model: Any, n: int = 32) -> Dict[str, torch.Tensor]:
    buf = model.rollout_buffer
    return {k: torch.as_tensor(np.asarray(v).reshape(-1, *np.asarray(v).shape[2:])[:n]) for k, v in
            buf.observations.items()}


def test_every_construction_of_the_production_forward_and_the_set_bce_runs_the_models_solve(golden_learner):
    from agents.training.belief_bank_static import hypothesis_set_terms
    pol = golden_learner.policy
    fe = pol.features_extractor
    hb = fe.hypothesis_builder
    assert hb is not None and hb.tau_solve == TAU_SOLVE_DEFAULT
    obs = _obs(golden_learner)
    try:
        for solve in ("fp64", TAU_SOLVE_DEFAULT):          # the setter first, then back to the default
            fe.set_tau_solve(solve)
            assert hb.tau_solve == solve
            with _SolveTap() as tap, torch.no_grad():
                pol.get_distribution(obs)
            # species set + the active's move group + the per-slot move presence + OTHER's mixture, at least
            assert len(tap.calls) >= 4 and set(tap.calls) == {solve}, tap.calls
            hs, bl = fe.last_hypothesis, fe.last_belief_logits
            assert hs.species.solve == solve
            with _SolveTap() as tap2, torch.no_grad():        # the BeliefHead set BCE re-runs the construction
                hypothesis_set_terms(hs, bl, obs["belief_species"], obs["belief_moves"], 1.0)
            assert tap2.calls and set(tap2.calls) == {solve}, tap2.calls
    finally:
        fe.set_tau_solve(TAU_SOLVE_DEFAULT)
    with pytest.raises(ValueError):
        fe.set_tau_solve(TAU_SOLVE_NAIVE)
    with pytest.raises(ValueError):
        HS.HypothesisBuilder(fe.layout, global_input_dim=50, n_move_seats=6, tau_solve=TAU_SOLVE_NAIVE)
