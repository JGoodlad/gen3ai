"""`torch.compile` the LEARNER's feature extractor — the GPU forward AND backward of the PPO step.

WHY THIS IS A SEPARATE MODULE FROM `snapshot.maybe_compile_extractor`, rather than a flag on it.
The two paths want OPPOSITE things at every decision, and folding them together is how the
`hide_cuda` bug happened (it used to be INFERRED from `torch.cuda.is_initialized()`, which was
correct only by accident of the call sites):

| | frozen OPPONENT (`--compile-opponents`) | LEARNER (`--compile-trainer`) |
|---|---|---|
| device | CPU, and it HIDES cuda so 48 workers do not each take ~252 MiB of card | CUDA, and hiding it would defeat the entire point |
| grad | inference only — grad-enabled calls route to EAGER on purpose | grad-enabled is the ONLY case that matters |
| on failure | warn and fall back to eager; `--compile-opponents-strict` opts into raising | ALWAYS raise |
| batch | B=1, launch-bound | the production minibatch (4096) |

MEASURED (2026-08-14, v76 `gen3_ctx_dedup_v1`, RTX 3080 Ti, the real
`MaskablePPO -> ActorCriticPolicy._build()` path, gen-9's own `cli_args`: batch 4096, PopArt on;
`policy.evaluate_actions` forward+backward, arms interleaved, 3 pairs):

    eager                  155.1 ms
    compiled extractor      88.5 ms   1.753x
    compiled evaluate_actions 88.5 ms 1.757x

At the ~89% train share of production wall at 10 epochs that is **~+62% end-to-end FPS**.

**We compile the EXTRACTOR, not `evaluate_actions`, and the numbers above are why.** The two scopes
measure the same to within 0.004x — the mlp_extractor, the pointer action head and the value head
contribute nothing measurable — so the whole-policy scope buys nothing for strictly more graph, and
more graph means more surface for SB3's distribution objects and the mask path to break on. Take the
identical win with the smaller blast radius.

FAIL-LOUD IS THE POINT, and it is not symmetric with the opponent path. A silent eager fallback here
is a 1.75x regression that no metric surfaces: the run trains correctly and simply produces ~38%
fewer steps per hour, forever. The opponent path can afford `strict` to be opt-in because it prints
a `[CompileExtractor]` line either way; here there is nothing to notice, so the only safe default is
to refuse to start.

CPU IS REJECTED, not attempted. `extractor_compiles_test.py` pins the reason as a measured fact: the
CPU BACKWARD does not lower — Inductor's C++ backend asserts on the damage op's `atomic_add` scatter
(`codegen/cpp.py: assert mode is None`). So `--compile-trainer --device cpu` cannot work, and saying
so at startup beats a confusing backend traceback ten minutes in.
"""
from __future__ import annotations

import contextlib
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

from agents.model.compile_gate_probe import (GradCoverageError, coverage_verdict, gate_loss,
                                             grad_parameters, per_param_grad_errors)
from agents.model.compile_parity_fixture import ParityFixtureError, load_parity_rows
from agents.model.parity_probe import (PERTURB_SEED, PRECISION_BARS, VacuousParityError,
                                       fresh_reason, ladder_at, perturbed_parameters,
                                       require_informative, rung_seed)


class CompileTrainerError(RuntimeError):
    """Raised when `--compile-trainer` cannot deliver a compiled learner. Always fatal."""


class VacuousCompileParityError(CompileTrainerError, VacuousParityError):
    """The parity gate was asked to judge a quantity that does not vary (gen3_fresh_parity_probe_v1)
    — e.g. a FRESH policy's legal log-probs, constant per row under the zero-init pointer head. A
    `CompileTrainerError`, so the launcher treats it as config-fatal like any other gate failure."""


# A compiled learner must beat eager by at least this, or something is wrong with the assumption
# rather than with the measurement: the arch measured 1.75x, so anything at or below parity means the
# graph fragmented or the backend fell back per-frame. Deliberately loose — this is a "did it do
# ANYTHING" tripwire, not a performance assertion, because a busy box can compress the ratio.
_MIN_SPEEDUP = 1.05

# How many forward+backward passes to time per arm when validating. Small: this runs at startup on
# the critical path, and the effect it checks for is a ~1.75x, not a 2% one.
_VALIDATE_REPS = 3

# The batch the startup validation compiles at. Small on purpose: see the note in
# `compile_trainer_extractor`. NOT the production batch, and the log line says so.
_VALIDATE_BATCH = 64

# A compiled forward that disagrees with eager by more than this is a wrong kernel, not a speedup.
# This is THE rule at matmul precision 'highest' (full fp32, the default) and it is unchanged there.
_MAX_NUMERIC_DRIFT = 1e-4

# gen3_tf32_parity_gate_v1 — the rule under REDUCED matmul precision (`--matmul-precision high`,
# TF32 tensor cores). There a fixed 1e-4 is the wrong question: eager and compiled each run their
# own TF32 kernels (cuBLAS vs Inductor/Triton, different tiling and accumulation order), so the two
# legitimately differ by TF32 rounding (~10-bit mantissa), which on this extractor is ~1e-2 absolute
# — the ai_v14_04_lbat_t32 launch died on exactly that (7.62e-03). So at reduced precision BOTH arms
# are measured against an fp32 EAGER REFERENCE (same weights, same obs, matmul precision temporarily
# 'highest') and the compiled arm must be no worse than K x what eager TF32 itself pays:
#
#     e_eager = max|eager_tf32    - ref_fp32|      (the rounding TF32 costs a CORRECT graph)
#     e_comp  = max|compiled_tf32 - ref_fp32|
#     PASS iff e_comp <= _TF32_K * e_eager + _TF32_EPS
#
# K is justified by measurement in `designs/training/compile_flags.md` (the ratio
# e_comp/e_eager on the real production extractor); EPS is the fp32 gate's own tolerance, so a
# graph that is exact under TF32 (e_eager == 0, e.g. no matmul on the path) is held to the SAME bar
# as at 'highest', never a looser one. A wrong kernel is a disagreement with the fp32 reference that
# does not shrink with precision; TF32 rounding cannot hide one larger than K x e_eager.
_TF32_K = 4.0
_TF32_EPS = _MAX_NUMERIC_DRIFT


def check_shape_stability(*, n_steps: int, n_envs: int, batch_size: int,
                          async_rollout: bool) -> None:
    """Refuse a config that would feed the compiled extractor an UNBOUNDED set of batch shapes.

    MEASURED BACKGROUND (2026-08-14), because the naive reading of this is wrong. Recompiles here
    are NORMAL and must not be an error: `share_features_extractor=True` means one extractor serves
    both paths, so `fe.forward` is called at batch=`n_envs` during rollout and batch=`batch_size`
    during train, alternating forever. Dynamo handles that — alternating two shapes converges after
    ~6 calls to a fixed set of graphs and then never recompiles again (17 graphs; steady state 8.8 ms
    at batch 48 / 74 ms at 512). So `torch._dynamo.config.error_on_recompile = True` would crash a
    perfectly healthy run on its second call, and `automatic_dynamic_shapes` is what makes the
    two-shape case work at all rather than being the hazard.

    THE ACTUAL HAZARD is dynamo's `cache_size_limit` (8). Exceed it for one code object and dynamo
    silently falls back to EAGER — which is exactly the invisible ~1.75x regression this whole flag
    exists to prevent, arriving with no error and no metric that would show it. Two configs get you
    there, and both are decidable at startup:

      * a REMAINDER minibatch — `n_steps*n_envs` not divisible by `batch_size` adds a third shape
        (and every epoch replays it), for no benefit;
      * `--async-rollout`, which forwards whatever set of envs is READY, so the rollout batch VARIES
        by construction — an unbounded shape set, guaranteed to exhaust the cache.

    Raises `CompileTrainerError`; pure, so both rules are testable without a GPU.
    """
    if async_rollout:
        raise CompileTrainerError(
            "--compile-trainer is incompatible with --async-rollout.\n"
            "The async collector forwards whichever envs are READY, so the rollout batch size VARIES "
            "every step. torch.compile keys on shape, so that is an unbounded set of graphs: dynamo "
            f"blows its cache_size_limit ({_dynamo_cache_limit()}) and SILENTLY falls back to eager "
            "— a ~1.75x regression with no error and nothing in any metric to show it.\n"
            "Pick one: drop --async-rollout (measured +14% at n_envs=64) and keep --compile-trainer "
            "(measured +62%), or drop --compile-trainer.")

    rollout = int(n_steps) * int(n_envs)
    if batch_size and rollout % int(batch_size) != 0:
        raise CompileTrainerError(
            f"--compile-trainer needs a rollout that divides evenly into minibatches, but "
            f"n_steps*n_envs = {n_steps}*{n_envs} = {rollout} leaves a remainder of "
            f"{rollout % int(batch_size)} against --batch-size {batch_size}.\n"
            "That remainder minibatch is a THIRD batch shape, replayed every epoch, which spends "
            f"dynamo's cache_size_limit ({_dynamo_cache_limit()}) faster and buys nothing — and "
            "exhausting it makes dynamo fall back to eager SILENTLY.\n"
            f"Adjust --batch-size to a divisor of {rollout} (e.g. "
            f"{_largest_divisor_at_most(rollout, int(batch_size))}), or drop --compile-trainer.")


def _dynamo_cache_limit() -> int:
    from agents.model.compile_control import cache_size_limit   # the one torch._dynamo adapter
    return cache_size_limit()


def _largest_divisor_at_most(n: int, cap: int) -> int:
    """A concrete suggestion beats 'pick a divisor' — the error should not make you do arithmetic."""
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def check_speedup(eager_ms: float, comp_ms: float) -> float:
    """Pure verdict: the compiled arm must actually be faster. Returns the speedup or raises.

    Separated out so it is testable without a GPU — the rule is the contract, and a rule that can
    only be exercised on a box with a free card is a rule that gets exercised rarely.
    """
    speedup = eager_ms / comp_ms if comp_ms > 0 else 0.0
    if speedup < _MIN_SPEEDUP:
        raise CompileTrainerError(
            f"--compile-trainer: compiled is NOT faster ({eager_ms:.1f} -> {comp_ms:.1f} ms, "
            f"{speedup:.2f}x < {_MIN_SPEEDUP}x). That means the graph fragmented or the backend fell "
            f"back per-frame rather than compiling — the measured figure for this arch is ~1.75x. "
            f"Failing rather than running a compile that costs startup time and buys nothing. Check "
            f"for a new graph break (`agents/model/extractor_compiles_test.py` asserts 1 graph / "
            f"0 breaks).")
    return speedup


def check_numerics(err: float, *, precision: str = "highest", eager_err: Optional[float] = None,
                   what: str = "features", tol: Optional[float] = None) -> str:
    """Pure verdict: a compile that changes the numbers is not a speedup. Returns the log line.

    ``precision`` is the RESOLVED `torch.get_float32_matmul_precision()` the arms ran at; ``what``
    names the quantity (``features``, ``legal_logprob``, ``value``, ``train features``).

    * ``'highest'`` (full fp32, the default): ``err`` is max|compiled - eager| and must be below
      ``tol`` (default 1e-4, the features bar, UNCHANGED). Looser than the CPU path's 1e-5 because
      cuBLAS may pick a different reduction order for the fused kernels; 1e-4 still catches a wrong
      kernel while tolerating a reordered correct one. ``eager_err`` is ignored.
    * anything else (``'high'`` = TF32): ``err`` is e_comp = max|compiled - fp32_ref| and
      ``eager_err`` is e_eager = max|eager - fp32_ref|; PASS iff e_comp <= K*e_eager + EPS. A missing
      or non-finite ``eager_err`` FAILS: a reduced-precision parity claim without its fp32 reference
      is no claim at all.

    Every comparison is written ``not (x <= tol)`` so a NaN FAILS rather than sailing through.
    """
    if precision == "highest":
        bar = _MAX_NUMERIC_DRIFT if tol is None else float(tol)
        rule = f"{what}: fp32 max|compiled-eager| {err:.2e} < {bar:g}"
        if not (err < bar):
            raise CompileTrainerError(
                f"--compile-trainer: the compiled extractor DISAGREES with eager on {what} "
                f"(max|delta| {err:.2e} > {bar:g}, matmul precision 'highest'). A faster wrong "
                f"model is not a win — investigate before re-enabling.")
        return rule
    if eager_err is None or not (eager_err >= 0.0) or eager_err == float("inf"):
        raise CompileTrainerError(
            f"--compile-trainer: at matmul precision {precision!r} the parity gate needs the eager "
            f"arm's own error against an fp32 reference (got {eager_err!r}) for {what}. Refusing "
            f"to enable the compile unvalidated.")
    bar = _TF32_K * eager_err + _TF32_EPS
    rule = (f"{what}: e_comp=max|compiled-fp32ref| {err:.2e} <= {_TF32_K:g} x "
            f"e_eager=max|eager-fp32ref| {eager_err:.2e} + {_TF32_EPS:g} = {bar:.2e} "
            f"(matmul precision {precision!r}, TF32)")
    if not (err <= bar):
        raise CompileTrainerError(
            f"--compile-trainer: the compiled extractor DISAGREES with the fp32 reference beyond "
            f"TF32 rounding — {rule} FAILED (ratio e_comp/e_eager "
            f"{(err / eager_err) if eager_err > 0 else float('inf'):.2f}). Eager at the same "
            f"precision stays within e_eager, so this is a wrong kernel, not rounding. A faster "
            f"wrong model is not a win — investigate before re-enabling.")
    return rule


def _max_abs_delta(a: Tuple["torch.Tensor", "torch.Tensor"],
                   b: Tuple["torch.Tensor", "torch.Tensor"]) -> float:
    """max|a - b| over the (pi, vf) pair, computed in fp32."""
    return max(float((x.float() - y.float()).abs().max()) for x, y in zip(a, b))


def fp32_reference(fn: Callable[[Any], Any], obs: Any) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """One no-grad EAGER forward at matmul precision 'highest', restoring the caller's precision.

    The reference the reduced-precision gate measures both arms against. Restored in ``finally`` so
    an exception cannot leave the trainer silently running at a precision its argv did not ask for.
    """
    prev = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad():
            pi, vf = fn(obs)
        return pi.clone(), vf.clone()
    finally:
        torch.set_float32_matmul_precision(prev)


def parity_verdict(*, eager: Tuple["torch.Tensor", "torch.Tensor"],
                   compiled: Tuple["torch.Tensor", "torch.Tensor"],
                   reference: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = None,
                   precision: Optional[str] = None) -> str:
    """The numerics gate over the arms' OUTPUTS. Raises `CompileTrainerError` or returns the rule line.

    ``precision`` defaults to the process's current `torch.get_float32_matmul_precision()`. At
    'highest' ``reference`` is not needed (and not read); at any other precision it is REQUIRED.
    """
    precision = precision or torch.get_float32_matmul_precision()
    if precision == "highest":
        return check_numerics(_max_abs_delta(compiled, eager), precision=precision)
    if reference is None:
        raise CompileTrainerError(
            f"--compile-trainer: matmul precision {precision!r} but no fp32 reference was "
            f"measured — refusing to validate the compile against a tolerance meant for fp32.")
    return check_numerics(_max_abs_delta(compiled, reference), precision=precision,
                          eager_err=_max_abs_delta(eager, reference))


# gen3_compile_parity_real_obs_v1 — the DECISION-level tolerances at matmul precision 'highest'.
# MEASURED after the trunk split (2026-09-28, 3,840 real eval-trace rows, ai_v14_01_base @72M and
# final): legal log-prob max|d| 2.7e-05, win-prob |dV| max 1.0e-06, pi_features max 6.9e-05; the
# broken single graph read 7.39 / 0.33 / 11.8. Each bar sits >= 10x over the healthy maximum and
# >= 1000x under the defect.
_FP32_TOL = {"features": _MAX_NUMERIC_DRIFT, "legal_logprob": PRECISION_BARS["highest"][0],
             "value": 1e-4}
# The train graph: cosine between the compiled and eager gradients of the gate's own loss over the
# extractor's parameters. Healthy after the split: 1.000000 (rel err 4.7e-07); the defect: 0.778.
_MIN_GRAD_COSINE = 0.9999
# gen3_gate_grad_coverage_v1 — the PER-PARAMETER gradient rule. The global cosine above is dominated
# by the largest gradients: on the perturbed production policy (CPU, 64 fixture rows) DROPPING the
# whole gradient through the pointer head's move cells left it at 1.0000 (the path's parameters are
# a sliver of the norm), while the per-parameter relative error ||c_p - e_p|| / ||e_p|| read 0.10 on
# `damage_op.out_gain`. So every parameter whose eager gradient norm is above
# `_PARAM_GRAD_FLOOR` x the largest one is ALSO held to a per-parameter bar. TWO bars, because the
# healthy noise differs by 100x between the two weight regimes (RTX 3080 Ti, torch 2.5.1, 64 rows,
# fp32, 2026-09-29; `designs/training/compile_flags.md`, "The train graph's coverage"):
#   * the FRESH-weights perturbed pass (every fresh launch): healthy max 7.3e-06 over 6 seeds;
#     a 10% backward error on the pointer's move cells reads 1.01e-02 -> bar 1e-3;
#   * REAL (trained) weights: healthy max 9.3e-04 on ai_v14_01_base/final, and perturbed trained
#     states showed isolated 4.9e-02 outliers (the alpha seat scorer) -> bar 0.2, which still
#     refuses any >=20% backward error on a path (a DROPPED path reads ~1.0).
_MAX_PARAM_GRAD_REL = 1e-3
_MAX_PARAM_GRAD_REL_TRAINED = 0.2
_PARAM_GRAD_FLOOR = 1e-3


def _parity_obs(obs_dim: int, batch: int, device: Any) -> Tuple[Dict[str, "torch.Tensor"], Any]:
    """The committed REAL rows as the gate's obs dict + their legal-action masks (numpy bool).

    Rows are repeated when ``batch`` exceeds the fixture. Raises `ParityFixtureError` (turned into
    a `CompileTrainerError` by the caller) when the fixture is missing or stale — never zeros.
    """
    import numpy as np

    rows, mask = load_parity_rows(obs_dim)
    idx = np.arange(int(batch)) % len(rows)
    return ({"observation": torch.as_tensor(rows[idx], device=device)}, mask[idx])


class _matmul_precision:
    """Temporarily set fp32 matmul precision; restored even on an exception."""

    def __init__(self, value: str):
        self.value = value
        self.prev = torch.get_float32_matmul_precision()

    def __enter__(self) -> None:
        self.prev = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision(self.value)

    def __exit__(self, *exc: Any) -> None:
        torch.set_float32_matmul_precision(self.prev)


def _readout(model: Any, fe: Any, obs: Any, legal_mask: Any) -> Dict[str, "torch.Tensor"]:
    """One no-grad forward through whatever `fe.forward` is installed, read at the DECISION level.

    ``features`` — (pi_features ‖ vf_features). When the policy is the Gen3 dual-head policy (it
    exposes the pointer-head seam and the critic read), also ``legal_logprob`` — the MASKED
    policy's log-probabilities on legal actions (0 elsewhere), i.e. exactly what the rollout
    samples from and PPO's ratio reads — and ``value``, the critic's value (the win-prob sigmoid
    under ``critic='winprob'``). A stand-in policy yields ``features`` only.
    """
    policy: Any = getattr(model, "policy", None)
    with torch.no_grad():
        pi, vf = fe(obs)
        out = {"features": torch.cat([pi.flatten(1), vf.flatten(1)], dim=1).float().clone()}
        if all(hasattr(policy, a) for a in ("mlp_extractor", "_critic_value",
                                            "_get_action_dist_from_latent")):
            lp = policy.mlp_extractor.forward_actor(pi)
            lv = policy.mlp_extractor.forward_critic(vf)
            out["value"] = policy._critic_value(lv).float().flatten().clone()
            dist = policy._get_action_dist_from_latent(lp)
            dist.apply_masking(legal_mask)
            logp = dist.distribution.logits.float()
            legal = torch.as_tensor(legal_mask, device=logp.device, dtype=torch.bool)
            out["legal_logprob"] = torch.where(legal, logp, torch.zeros_like(logp)).clone()
    return out


def _train_step(model: Any, fe: Any, obs: Any, legal_mask: Any) -> Dict[str, "torch.Tensor"]:
    """The TRAIN graph: one forward+backward of the gate's PROBE loss
    (`compile_gate_probe.gate_loss`: the features, the masked legal log-probs, V and every
    graph-carrying stash tensor — gen3_gate_grad_coverage_v1), read as the forward features, the
    flattened gradient over every POLICY parameter (extractor AND heads; `grad_parameters`), each
    parameter's size (``grad_sizes``, for the per-parameter rule) and its max|grad|
    (``grad_absmax``, for the coverage guard)."""
    params = [p for _, p in grad_parameters(model, fe)]
    for p in params:
        p.grad = None
    loss, pi, vf = gate_loss(model, fe, obs, legal_mask)
    loss.backward()
    gs = [(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten()
          for p in params]
    grad = torch.cat(gs)
    feats = torch.cat([pi.detach().flatten(1), vf.detach().flatten(1)], dim=1).float().clone()
    for p in params:
        p.grad = None
    return {"features": feats, "grad": grad,
            "grad_sizes": torch.tensor([g.numel() for g in gs], dtype=torch.long),
            "grad_absmax": torch.stack([g.abs().max() if g.numel() else g.new_zeros(())
                                        for g in gs]).cpu()}


def _cos(a: "torch.Tensor", b: "torch.Tensor") -> float:
    """Cosine in FLOAT64: over ~10^7 fp32 entries an fp32 dot/norm reads > 1 (1.0005 measured on
    the production policy's gradient), which would put the 0.9999 bar inside rounding noise."""
    a, b = a.detach().double(), b.detach().double()
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 and nb == 0.0:
        return 1.0
    return float(torch.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def _require_informative(quantities: Dict[str, "torch.Tensor"], bars: Dict[str, float],
                         where: str) -> None:
    try:
        require_informative(quantities, bars, where=where)
    except VacuousParityError as exc:
        raise VacuousCompileParityError(str(exc)) from exc


def decision_verdicts(*, eager: Dict[str, "torch.Tensor"], compiled: Dict[str, "torch.Tensor"],
                      reference: Optional[Dict[str, "torch.Tensor"]] = None,
                      precision: Optional[str] = None, allow_vacuous: bool = False) -> List[str]:
    """The no-grad parity gate over each readout quantity. Raises `CompileTrainerError` or returns
    one rule line per quantity. At 'highest' each quantity uses its `_FP32_TOL` bar on
    max|compiled - eager|; at reduced precision the TF32 rule (`check_numerics`) against the fp32
    ``reference``, which is then REQUIRED.

    FAIL-CLOSED ON A VACUOUS COMPARISON (gen3_fresh_parity_probe_v1): every compared quantity of the
    EAGER arm must vary by more than its own bar (`parity_probe.spread`), else
    `VacuousCompileParityError` — a fresh policy's legal log-probs are constant per row and pass any
    miscompile. ``allow_vacuous=True`` is for a caller that has ALREADY judged the same graph on a
    perturbed, informative copy of the weights (the gate's fresh path) and only it."""
    precision = precision or torch.get_float32_matmul_precision()
    if not allow_vacuous:
        _require_informative({k: v for k, v in eager.items() if k in _FP32_TOL}, _FP32_TOL,
                             "--compile-trainer parity (decision readout)")
    if precision != "highest" and reference is None:
        raise CompileTrainerError(
            f"--compile-trainer: matmul precision {precision!r} but no fp32 reference was "
            f"measured — refusing to validate the compile against a tolerance meant for fp32.")
    lines = []
    for key in ("features", "legal_logprob", "value"):
        if key not in eager:
            continue
        if key not in compiled:
            raise CompileTrainerError(f"--compile-trainer: the compiled arm produced no {key!r}")
        if precision == "highest":
            err = float((compiled[key] - eager[key]).abs().max())
            lines.append(check_numerics(err, precision=precision, what=key, tol=_FP32_TOL[key]))
        else:
            assert reference is not None
            lines.append(check_numerics(
                float((compiled[key] - reference[key]).abs().max()), precision=precision,
                eager_err=float((eager[key] - reference[key]).abs().max()), what=key))
    return lines


def _param_verdict(compiled: Dict[str, "torch.Tensor"], eager: Dict[str, "torch.Tensor"],
                   param_names: Optional[List[str]], *, allow_vacuous: bool = False,
                   bar: float = _MAX_PARAM_GRAD_REL) -> Optional[str]:
    """The per-parameter gradient rule (fp32 only). None when the arms carry no ``grad_sizes``
    (a hand-built verdict input)."""
    if "grad_sizes" not in eager or "grad_sizes" not in compiled:
        return None
    sizes = [int(x) for x in eager["grad_sizes"].tolist()]
    if sizes != [int(x) for x in compiled["grad_sizes"].tolist()]:
        raise CompileTrainerError("--compile-trainer: the two arms' gradients cover different "
                                  "parameter sets — the gate is mis-wired")
    errs = per_param_grad_errors(compiled["grad"], eager["grad"], sizes,
                                 floor_frac=_PARAM_GRAD_FLOOR)
    if not errs:
        if allow_vacuous:
            return None
        raise VacuousCompileParityError("--compile-trainer parity (train graph): no parameter's "
                                        "gradient is above the per-parameter floor")
    worst_i, worst = max(errs, key=lambda t: (not (t[1] == t[1]), t[1]))   # NaN sorts worst
    name = param_names[worst_i] if param_names and worst_i < len(param_names) else f"#{worst_i}"
    rule = (f"per-param grad rel err max {worst:.2e} ({name}) over {len(errs)} params "
            f"<= {bar:g}")
    if not (worst <= bar):
        bad = [(param_names[i] if param_names and i < len(param_names) else f"#{i}", e)
               for i, e in errs if not (e <= bar)]
        raise CompileTrainerError(
            f"--compile-trainer: the compiled TRAIN graph's gradient DISAGREES with eager on "
            f"{len(bad)} parameter(s) — {rule} FAILED (e.g. "
            f"{', '.join(f'{n} {e:.2e}' for n, e in bad[:5])}). A backward miscompile in a path "
            f"the global cosine cannot resolve. A faster wrong model is not a win — investigate "
            f"before re-enabling.")
    return rule


def train_verdict(*, eager: Dict[str, "torch.Tensor"], compiled: Dict[str, "torch.Tensor"],
                  reference: Optional[Dict[str, "torch.Tensor"]] = None,
                  precision: Optional[str] = None, allow_vacuous: bool = False,
                  param_names: Optional[List[str]] = None,
                  param_bar: float = _MAX_PARAM_GRAD_REL) -> str:
    """The TRAIN-graph parity gate: forward features (the numerics rule) AND the gradient's cosine.

    At 'highest': cos(compiled_grad, eager_grad) >= 0.9999. At reduced precision:
    (1 - cos(compiled, fp32)) <= K * (1 - cos(eager, fp32)) + 1e-4 — TF32's own angular error on
    the eager arm, scaled by the same K as the value rule. The eager arm's features must vary
    across rows and its gradient must be non-zero (`VacuousCompileParityError` otherwise; an
    all-zero gradient has cosine 1.0 with anything's zero).

    gen3_gate_grad_coverage_v1: when the arms carry the per-parameter reads (`_train_step` always
    does), the eager arm must ALSO give a non-zero gradient to all but
    `compile_gate_probe.MAX_ZERO_GRAD_FRACTION` of the parameters (`VacuousCompileParityError`
    otherwise), and at fp32 every parameter above the floor is held to `_MAX_PARAM_GRAD_REL`."""
    precision = precision or torch.get_float32_matmul_precision()
    coverage = None
    if not allow_vacuous:
        _require_informative({"features": eager["features"], "grad": eager["grad"]},
                             {"features": _MAX_NUMERIC_DRIFT, "grad": 0.0},
                             "--compile-trainer parity (train graph)")
        if "grad_absmax" in eager:
            try:
                coverage = coverage_verdict(eager["grad_absmax"], param_names,
                                            where="--compile-trainer parity (train graph)")
            except GradCoverageError as exc:
                raise VacuousCompileParityError(str(exc)) from exc
    if precision == "highest":
        feat = check_numerics(float((compiled["features"] - eager["features"]).abs().max()),
                              precision=precision, what="train features")
        cos = _cos(compiled["grad"], eager["grad"])
        if not (cos >= _MIN_GRAD_COSINE):
            raise CompileTrainerError(
                f"--compile-trainer: the compiled TRAIN graph's gradient DISAGREES with eager — "
                f"cosine {cos:.6f} < {_MIN_GRAD_COSINE} on the gate's own loss over the "
                f"extractor's parameters. PPO would be stepping along the wrong direction. A "
                f"faster wrong model is not a win — investigate before re-enabling.")
        per_param = _param_verdict(compiled, eager, param_names, allow_vacuous=allow_vacuous,
                                   bar=param_bar)
        return "; ".join(x for x in (feat, f"grad cosine {cos:.6f} >= {_MIN_GRAD_COSINE}",
                                     per_param, coverage) if x)
    if reference is None:
        raise CompileTrainerError(
            f"--compile-trainer: matmul precision {precision!r} but the train graph has no fp32 "
            f"reference — refusing to validate it against a tolerance meant for fp32.")
    feat = check_numerics(float((compiled["features"] - reference["features"]).abs().max()),
                          precision=precision,
                          eager_err=float((eager["features"] - reference["features"]).abs().max()),
                          what="train features")
    d_comp = 1.0 - _cos(compiled["grad"], reference["grad"])
    d_eager = 1.0 - _cos(eager["grad"], reference["grad"])
    bar = _TF32_K * max(d_eager, 0.0) + _TF32_EPS
    rule = (f"grad 1-cos vs fp32: compiled {d_comp:.2e} <= {_TF32_K:g} x eager {d_eager:.2e} "
            f"+ {_TF32_EPS:g} = {bar:.2e}")
    if not (d_comp <= bar):
        raise CompileTrainerError(
            f"--compile-trainer: the compiled TRAIN graph's gradient DISAGREES with the fp32 "
            f"reference beyond TF32 rounding at matmul precision {precision!r} — {rule} FAILED. "
            f"A faster wrong model is not a win — investigate before re-enabling.")
    return "; ".join(x for x in (feat, rule, coverage) if x)


def resolve_device(fe: Any) -> "torch.device":
    """The learner's device, as its own function so the CPU refusal has a seam to test through."""
    # `fe` is deliberately `Any` (the extractor arrives through SB3), so `.parameters()` is too.
    return next(fe.parameters()).device  # type: ignore[no-any-return]


def _one_step(fe: Any, obs: Any) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """One forward + backward through the extractor, the shape the PPO step actually runs."""
    fe.zero_grad(set_to_none=True)
    pi, vf = fe(obs)
    (pi.square().mean() + vf.square().mean()).backward()
    return pi, vf


def _time_steps(fe: Any, obs: Any, reps: int) -> float:
    for _ in range(2):                       # warm: the first call pays tracing + codegen
        _one_step(fe, obs)
    if obs["observation"].is_cuda:
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        _one_step(fe, obs)
        if obs["observation"].is_cuda:
            torch.cuda.synchronize()         # async: else we time the LAUNCH, not the work
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


_Arm = Tuple[Dict[str, "torch.Tensor"], Dict[str, "torch.Tensor"],
             Optional[Dict[str, "torch.Tensor"]], Optional[Dict[str, "torch.Tensor"]]]


def _gate_arm(model: Any, fe: Any, obs: Any, legal_mask: Any, precision: str) -> _Arm:
    """One arm of the gate through whatever `fe.forward` is installed: ``(read, train, read32,
    train32)`` — the decision readout and the train step at the process precision, plus the same
    two at 'highest' when that precision is reduced (else None). For the EAGER arm the fp32 pair is
    the reference; for the COMPILED arm it is the same trace's fp32 graph."""
    read, train = _readout(model, fe, obs, legal_mask), _train_step(model, fe, obs, legal_mask)
    read32 = train32 = None
    if precision != "highest":
        with _matmul_precision("highest"):
            read32 = _readout(model, fe, obs, legal_mask)
            train32 = _train_step(model, fe, obs, legal_mask)
    return read, train, read32, train32


def _arm_vacuity(arm: _Arm, precision: str, param_names: Optional[List[str]]) -> Optional[str]:
    """``None`` when ``arm`` (an EAGER arm) is informative on every rule `_arm_verdicts` applies, else
    the vacuity refusal's text. Judged by running the rules on the arm against ITSELF: every numeric
    delta is exactly 0, so the only thing that can raise is a vacuity guard."""
    try:
        _arm_verdicts(arm, arm, precision, param_names=param_names, param_bar=_MAX_PARAM_GRAD_REL)
    except VacuousCompileParityError as exc:
        return str(exc).split(". A quantity")[0]
    return None


def _arm_verdicts(eager: _Arm, comp: _Arm, precision: str, *,
                  allow_vacuous: bool = False,
                  param_names: Optional[List[str]] = None,
                  param_bar: float = _MAX_PARAM_GRAD_REL_TRAINED) -> List[str]:
    """Every rule over one (eager, compiled) pair of arms. Raises `CompileTrainerError`."""
    e_read, e_train, ref_read, ref_train = eager
    c_read, c_train, c_read32, c_train32 = comp
    rules = decision_verdicts(eager=e_read, compiled=c_read, reference=ref_read,
                              precision=precision, allow_vacuous=allow_vacuous)
    rules.append(train_verdict(eager=e_train, compiled=c_train, reference=ref_train,
                               precision=precision, allow_vacuous=allow_vacuous,
                               param_names=param_names, param_bar=param_bar))
    if c_read32 is not None and ref_read is not None and ref_train is not None \
            and c_train32 is not None:
        rules += ["[same graph at fp32] " + r for r in decision_verdicts(
            eager=ref_read, compiled=c_read32, precision="highest", allow_vacuous=allow_vacuous)]
        rules.append("[same graph at fp32] " + train_verdict(
            eager=ref_train, compiled=c_train32, precision="highest",
            allow_vacuous=allow_vacuous, param_names=param_names, param_bar=param_bar))
    return rules


#: Batch sizes the COMPILED learner forward never serves (`gen3_batch1_eager_v1`, Lane K): a call at
#: one of these batch sizes runs the extractor's EAGER forward (same parameters, same autograd).
#:
#: WHY batch 1. On torch 2.8.0+cu126 a batch-1 CUDA eval/no-grad graph of the production extractor
#: does not LOWER (Triton `CompilationError`, "'constexpr_type' object has no attribute 'is_block'" on
#: a fully-constant `tl.broadcast_to` index — the K1 finding; batch 2 and 4 compile, and 2.5.1
#: compiles batch 1). Batch 1 reaches the learner process twice, both OFF the hot path: the
#: truncated-episode `predict_values` under `--critic shaped` (rare) and the trainer's in-process
#: FINAL EVALUATION (after `learn()` returns). Eager costs ~18 ms per batch-1 forward and nothing
#: else; padding to batch 2 would have to slice every per-forward extractor stash back to one row.
#: So batch 1 is never a compiled signature — on either torch, one rule, nothing version-keyed — and
#: the declared signature table (the prewarm) contains none. `compile_trainer_test` pins the routing;
#: the CUDA test `compile_batch1_cuda_test` (2.8, `slow`, GPU tier) fails on its revert.
EAGER_BATCHES = frozenset({1})


def _rows(obs: Any) -> int:
    x: Any = obs.get("observation") if isinstance(obs, dict) else obs
    if x is None and isinstance(obs, dict):
        x = next(iter(obs.values()))
    return int(x.shape[0])


def route_small_batches_eager(fe: Any, compiled: Callable[..., Any]) -> Callable[..., Any]:
    """The callable `compile_trainer_extractor` installs as ``fe.forward``: ``compiled`` for every
    batch except `EAGER_BATCHES`, which run the class's own (eager) forward. Pure dispatch — no
    numerics, no extra dynamo frame (the batch-size test runs in Python, outside the graph)."""
    cls_forward = type(fe).forward

    def forward(obs: Any) -> Any:
        if _rows(obs) in EAGER_BATCHES:
            return cls_forward(fe, obs)
        return compiled(obs)
    forward._gen3_compiled = compiled                                   # type: ignore[attr-defined]
    return forward


def compile_trainer_extractor(model: Any, enabled: bool, *, batch: Optional[int] = None,
                              emit: Optional[Callable[[str], None]] = None) -> Optional[float]:
    """Compile `model.policy.features_extractor.forward` in place. Returns the measured speedup.

    Returns None when `enabled` is False (a true no-op — nothing is touched, so an off run is
    byte-identical). Raises `CompileTrainerError` on ANY failure, including a compile that does not
    actually go faster.

    `emit` is an optional one-arg callable for the launcher event stream; stderr is used regardless.

    Patches the BOUND `fe.forward`, never the module. `torch.compile(module)` would wrap it in an
    `OptimizedModule` and prefix every `state_dict` key with `_orig_mod.`, which would land in the
    next checkpoint and make it unloadable by anything else — the same reason the opponent path
    patches the bound method. `save_model_snapshot` -> `model.save()` writes `policy.state_dict()`,
    so the keys are what ends up on disk; `compile_trainer_test.py` pins that they are unchanged and
    that a save/reload round-trip still works.
    """
    if not enabled:
        return None
    # K3 (gen3_hermetic_compile_cache_v1): the learner compiles into the RUN's own cache (declared
    # by the trainer before this), or a private one — never torch's shared default.
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("learner compile")

    # VALIDATE AT A SMALL, SAFE BATCH — and label the number with the shape it was measured at.
    #
    # This was `model.batch_size` for exactly one afternoon, and it BROKE STARTUP — a gen-10 launch
    # that had been running happily at 935 fps refused to start. The cause is now KNOWN and it is not
    # subtle: **validating at the train batch needs MORE GPU memory than training itself does.**
    # Validation runs the arm eager AND compiled in one process, with Inductor's compile workspace on
    # top; training only ever needs one of them. At batch 4096 that exceeds the card and the allocator
    # fails — surfacing first as a mystifying `CUDA error: invalid configuration argument`, and as a
    # plain `OutOfMemoryError` once the obs was valid enough to get further. So the small batch is not
    # a shortcut around an unexplained bug; it is the only shape this check can afford.
    #
    # The validation exists to answer "did the compile WORK", not "how fast is it in production", and
    # a small batch answers that. The cost is one extra graph shape at startup, which is cheap:
    # dynamo converges over repeated alternation and then stops recompiling (measured — see
    # `check_shape_stability`). The honesty problem was never the batch; it was reporting a
    # batch-64 ratio as if it were the production one. So the SHAPE IS NAMED in the log line.
    if batch is None:
        batch = _VALIDATE_BATCH

    def _say(msg: str) -> None:
        print(msg, flush=True)
        if emit is not None:
            try:
                emit(msg)
            except Exception:
                pass                          # a diagnostic must never break the run

    policy = getattr(model, "policy", None)
    fe = getattr(policy, "features_extractor", None)
    if fe is None:
        raise CompileTrainerError(
            "--compile-trainer: this policy has no `features_extractor`. The flag compiles the "
            "Gen3 extractor specifically; it cannot be used with a stock SB3 policy.")

    device = resolve_device(fe)
    if device.type != "cuda":
        raise CompileTrainerError(
            f"--compile-trainer requires CUDA, but the model is on {device.type!r}.\n"
            "This is not a conservatism: the CPU BACKWARD provably does not lower — Inductor's C++ "
            "backend asserts on the damage operator's atomic_add scatter "
            "(`codegen/cpp.py: assert mode is None`), pinned by "
            "`agents/model/extractor_compiles_test.py::test_cpu_backward_still_does_not_compile`.\n"
            "Pass --device cuda, or drop --compile-trainer. (--compile-opponents is the CPU-side "
            "flag and is unaffected.)")

    obs_dim = None
    for attr in ("obs_dim", "observation_dim"):
        obs_dim = getattr(fe, attr, None)
        if isinstance(obs_dim, int):
            break
    if not isinstance(obs_dim, int):
        layout = getattr(fe, "layout", None)
        obs_dim = (layout or {}).get("total_dim") if isinstance(layout, dict) else None
    if not isinstance(obs_dim, int):
        raise CompileTrainerError(
            "--compile-trainer: could not determine the extractor's observation width, so the "
            "compile could not be validated. Refusing to enable it unvalidated.")

    was_training = fe.training
    fe.train()                                 # the backward path is what we are compiling
    # gen3_compile_parity_real_obs_v1: REAL observation rows, never zeros. Until 2026-09-28 this
    # probe was an all-zero obs, on which the single-graph CUDA compile agreed with eager to 4.8e-7
    # while it was off by 7.65 on real rows (70.9% argmax agreement, gradient cosine 0.778) — zero
    # obs exercise no masking, no top-K seat selection and no edge family. The rows are a committed
    # fixture of reproducible bridge-battle states (`compile_parity_fixture`); a missing or stale
    # fixture REFUSES rather than falling back to zeros.
    obs, legal_mask = _parity_obs(obs_dim, batch, device)
    # …and every FLAG-GATED Dict key this extractor's forward reads, from the declared registry.
    # dynamo guards on a dict's KEY SET, so an under-built warmup is either a crash (the privileged
    # value route RAISES on a missing `opp_true_team` — the ai_v12_14_ladder_truevalue launch) or a
    # full re-trace on the first live batch. Never add a key here by hand.
    from agents.model.extra_obs_keys import zero_extra_obs
    obs.update(zero_extra_obs(fe, batch=batch, device=device))

    original = fe.forward
    precision = torch.get_float32_matmul_precision()
    # gen3_fresh_parity_probe_v1: the module whose parameters a FRESH-weights pass perturbs — the
    # whole policy (extractor AND heads) when there is one, else the extractor.
    probe_module = policy if isinstance(policy, torch.nn.Module) else fe
    param_names = [n for n, _ in grad_parameters(model, fe)]
    fresh: Optional[str] = None
    # gen3_compile_sentinel_v1: phase 1 of `compile_control` — the gate compiles FREELY, with the
    # cache-limit detector already listening (it also sets suppress_errors=False: a partial compile
    # must be LOUD). Phase 2 (`reset`, dropping every graph compiled here) and 3-4 (prewarm, lock)
    # run in `arm_compile_sentinel`, after grad checkpointing is applied and before `learn()`.
    from agents.model.compile_control import control
    with control(emit).gate():
        try:
            eager_ms = _time_steps(fe, obs, _VALIDATE_REPS)
            # The EAGER arm: the deployed function's decision readout and its train-step gradient,
            # plus (reduced precision only) the same two at fp32 — the reference both arms are
            # measured against. All taken BEFORE the compile is installed (eager by construction).
            eager_arm = _gate_arm(model, fe, obs, legal_mask, precision)
            # gen3_fresh_parity_probe_v1: FRESH weights (the zero-init pointer head makes every
            # legal log-prob -log(n_legal)) cannot judge the decision readout. Detect it on the
            # eager arm and ALSO run both arms on a seeded perturbation of the SAME parameters —
            # in place, restored bit-exactly, private RNG — so the graph judged is the one that
            # ships. `decision_verdicts` refuses a vacuous comparison, so this cannot be skipped.
            fresh = fresh_reason({k: v for k, v in eager_arm[0].items() if k in _FP32_TOL},
                                 _FP32_TOL)
            p_eager_arm = None
            p_scale: Optional[float] = None
            p_seed, p_k = PERTURB_SEED, 0
            if fresh is not None:
                # gen3_parity_perturb_ladder_v1: climb the DECLARED ladder of (scale, seed) rungs
                # to the first whose EAGER arm is informative on every rule; a COLLAPSED critic (a
                # saturated win-prob head) can stay vacuous on V at the first, fresh-weights rung.
                # None ⇒ refuse: the ladder never licenses a vacuous pass.
                tried = []
                try:
                    ladder = ladder_at(precision)
                except KeyError as exc:
                    raise CompileTrainerError(f"--compile-trainer: {exc}") from exc
                for scale, k in ladder:
                    with perturbed_parameters(probe_module, seed=rung_seed(k), scale=scale):
                        arm = _gate_arm(model, fe, obs, legal_mask, precision)
                    why = _arm_vacuity(arm, precision, param_names)
                    if why is None:
                        p_eager_arm, p_scale, p_seed, p_k = arm, float(scale), rung_seed(k), int(k)
                        break
                    tried.append(f"scale {scale:g} seed+{k}: {why}")
                if p_eager_arm is None:
                    raise VacuousCompileParityError(
                        f"--compile-trainer parity: VACUOUS on the real weights ({fresh}) and on "
                        f"every rung of the perturbation ladder {ladder} ((scale, seed offset) "
                        f"at or under matmul precision {precision!r}'s scale cap; seed "
                        f"{PERTURB_SEED}) — {'; '.join(tried)}. Refusing: a comparison that does "
                        f"not vary cannot tell a miscompile from a match.")

            compiled = torch.compile(original)
            # `wrap_compiled` records a lock rejection raised through the learner forward before it
            # propagates (sticky: a caller's `except Exception` cannot hide it — torch >= 2.8's
            # start callback no longer sees a rejected recompile). Pass-through otherwise.
            fe.forward = route_small_batches_eager(fe, control(emit).wrap_compiled(compiled))
            comp_ms = _time_steps(fe, obs, _VALIDATE_REPS)
            # gen3_tf32_parity_gate_v1: under reduced precision the TF32 rule above can only resolve
            # a defect larger than ~K x TF32's own rounding (measured: a DROPPED projection bias passes
            # it). So `_gate_arm` ALSO runs the same compiled callable at 'highest' — dynamo guards on
            # the TF32 flag, so that is a separate fp32 graph of the SAME trace — and it is held to the
            # strict fp32 bars against the fp32 eager reference. Every precision-independent graph
            # defect (a wrong fusion, a dropped term, the single-graph miscompile) fails there at
            # full resolution.
            comp_arm = _gate_arm(model, fe, obs, legal_mask, precision)
            p_comp_arm = None
            if p_scale is not None:                         # same seed + scale -> same weights
                with perturbed_parameters(probe_module, seed=p_seed, scale=p_scale):
                    p_comp_arm = _gate_arm(model, fe, obs, legal_mask, precision)
        except (CompileTrainerError, ParityFixtureError) as exc:
            fe.forward = original
            if isinstance(exc, ParityFixtureError):
                raise CompileTrainerError(f"--compile-trainer: {exc}") from exc
            raise
        except Exception as exc:
            fe.forward = original
            # Include the TRACEBACK, not just str(exc). "Bisect the op" is useless advice without a
            # stack, and the one failure this has actually seen in the wild (a CUDA "invalid
            # configuration argument") carries its whole diagnosis in the frames — `str(exc)` alone
            # names no op, no shape and no file.
            import traceback as _tb
            _stack = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))[-3000:]
            raise CompileTrainerError(
                f"--compile-trainer: the learner's extractor FAILED to compile — "
                f"{type(exc).__name__}: {exc}\n\n--- traceback (last frames) ---\n{_stack}\n"
                "This is fatal by design: falling back to eager here is a ~1.75x throughput regression "
                "that nothing in the run would surface. Either fix the op that will not lower (bisect "
                "it — see `src/agents/model/CLAUDE.md`, the species_posterior precedent, where the whole "
                "'torch cannot compile our model' story was ONE op), or drop --compile-trainer."
            ) from exc
        finally:
            fe.zero_grad(set_to_none=True)
            probe_module.zero_grad(set_to_none=True)   # the probe loss reaches the heads too
            if not was_training:
                fe.eval()

    try:
        if p_eager_arm is not None and p_comp_arm is not None:
            # The informative verdict FIRST, with the vacuity guard ON: a perturbation that still
            # left a quantity constant refuses the launch rather than passing it.
            rules = [f"[fresh weights, seeded perturbation scale={p_scale:g} seed+{p_k}] " + r
                     for r in _arm_verdicts(p_eager_arm, p_comp_arm, precision,
                                            param_names=param_names,
                                            param_bar=_MAX_PARAM_GRAD_REL)]
            rules += _arm_verdicts(eager_arm, comp_arm, precision, allow_vacuous=True,
                                   param_names=param_names)
        else:
            rules = _arm_verdicts(eager_arm, comp_arm, precision, param_names=param_names)
        speedup = check_speedup(eager_ms, comp_ms)
    except CompileTrainerError:
        fe.forward = original          # never leave a rejected compile installed
        raise
    if fresh is not None:
        _say(f"[CompileTrainer] FRESH weights ({fresh} on the fixture — vacuous on their own): the "
             f"parity gate ALSO ran on a seeded perturbation of every policy parameter (seed "
             f"{p_seed}, scale {p_scale:g} — the first informative (scale, seed offset) rung of "
             f"{ladder_at(precision)}; restored bit-exactly, private RNG)")
    # Said on EVERY passing launch (a failure says it in the raised message): which rules ran, at
    # which precision, and the numbers — so a TF32 run records how close to its bar it sat.
    _say(f"[CompileTrainer] parity PASS on {batch} REAL obs rows — " + " | ".join(rules))

    prod = int(getattr(model, "batch_size", 0) or 0)
    shape_note = (f" — VALIDATION shape only; the production batch is {prod} and the measured "
                  f"speedup there is ~1.75x, NOT this number"
                  if prod and prod != batch else "")
    _say(f"[CompileTrainer] ON — learner fwd+bwd {eager_ms:.1f} -> {comp_ms:.1f} ms "
         f"({speedup:.2f}x) at batch {batch} on {device}{shape_note}")
    return speedup


# ------------------------------------------------------------------------------------------------
# gen3_compile_sentinel_v1 — phases 2-4 of `compile_control`: reset, prewarm, attach (lock later)
# ------------------------------------------------------------------------------------------------

def _prewarm_obs(model: Any, batch: int) -> Dict[str, "torch.Tensor"]:
    """A batch shaped EXACTLY like the rollout's `obs_as_tensor(self._last_obs)`: every key of the
    policy's observation space (dynamo guards the dict), `observation` from the committed REAL rows,
    `action_mask` from their masks, every other key zeros of its space's shape and dtype."""
    import numpy as np

    policy = model.policy
    device = resolve_device(policy.features_extractor)
    space = policy.observation_space
    rows, mask = load_parity_rows(int(space.spaces["observation"].shape[0]))
    idx = np.arange(int(batch)) % len(rows)
    out: Dict[str, "torch.Tensor"] = {}
    for key, sub in space.spaces.items():
        if key == "observation":
            arr = rows[idx]
        elif key == "action_mask" and tuple(sub.shape) == tuple(mask.shape[1:]):
            arr = mask[idx].astype(sub.dtype)
        else:
            arr = np.zeros((int(batch),) + tuple(sub.shape or ()), dtype=sub.dtype)
        out[key] = torch.as_tensor(arr, device=device)
    return out


def production_prewarm_calls(model: Any, *, n_envs: int,
                             batch_size: int) -> List[Tuple[str, Callable[[], None]]]:
    """Every compiled-extractor signature the learner process reaches in production, in the order
    production first reaches them (rollout first, so its graph specializes on n_envs as it would).

    Found by reading every learner-process caller (designs/training/compile_flags.md, "The compile
    sentinel" — the late-shape table):
      * rollout `policy(obs)` + end-of-rollout `predict_values`: EVAL, no-grad, batch n_envs;
      * the update's `evaluate_actions`: TRAIN, grad, batch batch_size (fwd + bwd);
      * `rank_probe` (first minibatch of every update): TRAIN, no-grad, batch batch_size;
      * a second TRAIN/grad size (batch_size // 2 — the capacity half-batch cosine) so the train
        graph is dynamic in batch before the lock: variable-size train callers (td-aux, the
        distill-anchor fallback) then reuse it;
    A batch in `EAGER_BATCHES` (batch 1: the `--critic shaped` truncation value, the final
    evaluation, a `--debug` single-env rollout) never reaches the compiled graph, so it is never
    prewarmed — `gen3_batch1_eager_v1`.
    """
    policy = model.policy

    def _eval_nograd(b: int) -> Callable[[], None]:
        def run() -> None:
            was = policy.training
            policy.set_training_mode(False)
            try:
                with torch.no_grad():
                    policy.extract_features(_prewarm_obs(model, b))
            finally:
                policy.set_training_mode(was)
        return run

    def _train(b: int, grad: bool) -> Callable[[], None]:
        def run() -> None:
            was = policy.training
            policy.set_training_mode(True)
            try:
                if grad:
                    pi, vf = policy.extract_features(_prewarm_obs(model, b))
                    (pi.float().square().mean() + vf.float().square().mean()).backward()
                else:
                    with torch.no_grad():
                        policy.extract_features(_prewarm_obs(model, b))
            finally:
                policy.zero_grad(set_to_none=True)
                policy.set_training_mode(was)
        return run

    calls: List[Tuple[str, Callable[[], None]]] = [
        (f"rollout eval/no-grad B={n_envs}", _eval_nograd(n_envs)),
        (f"update train/grad B={batch_size}", _train(batch_size, True)),
        (f"rank-probe train/no-grad B={batch_size}", _train(batch_size, False)),
    ]
    half = max(2, int(batch_size) // 2)
    if half != int(batch_size):
        calls.append((f"capacity train/grad B={half}", _train(half, True)))
    return [c for c, b in zip(calls, (int(n_envs), int(batch_size), int(batch_size), half))
            if b not in EAGER_BATCHES]


def arm_compile_sentinel(model: Any, *, n_envs: int, batch_size: int,
                         emit: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Phases 2-4 for a compiled learner: reset the gate's graphs, prewarm every production
    signature, and attach the lock (after the first rollout + update) and the per-rollout /
    per-update checks. A no-op (returns None) when the learner is not compiled.

    MUST run after `_apply_grad_checkpointing` (the forward reads `grad_checkpointing`, so a graph
    compiled before it is a stale entry) and before `learn()`. Raises `CompileSentinelError`.
    """
    fe = getattr(getattr(model, "policy", None), "features_extractor", None)
    if fe is None or "forward" not in vars(fe):
        return None
    from agents.model.compile_control import control
    ctl = control(emit)

    def _say(msg: str) -> None:
        print(msg, flush=True)
        if emit is not None:
            try:
                emit(msg)
            except Exception:
                pass

    t0 = time.perf_counter()
    _say(ctl.reset())
    calls = production_prewarm_calls(model, n_envs=int(n_envs), batch_size=int(batch_size))
    line = ctl.prewarm(calls)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _say(f"{line} — reset + prewarm took {time.perf_counter() - t0:.1f}s")
    ctl.attach(model)
    return line


@contextlib.contextmanager
def eager_extractor(fe: Any) -> Iterator[None]:
    """Route `fe` to its EAGER forward for the block (a compiled `fe.forward` is an INSTANCE
    attribute over the class method; removing it for the block exposes the class method).

    For the learner-process callers whose signature can FIRST appear after the lock and cannot be
    pre-warmed — a different obs KEY SET or a batch that may be 1 (search-teacher/OPD, fork-arm
    scoring, the distill grad-projection). Same parameters, same autograd; eager numerics.
    """
    compiled = vars(fe).get("forward") if fe is not None and hasattr(fe, "__dict__") else None
    if compiled is None:                          # not compiled (or no extractor): nothing to route
        yield
        return
    del fe.forward
    try:
        yield
    finally:
        fe.forward = compiled
