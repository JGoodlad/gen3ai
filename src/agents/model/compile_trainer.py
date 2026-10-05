"""`--compile-trainer` — the learner process's compile: its startup preflight, the parity VERDICTS
every compile gate shares, and the compile SENTINEL that installs, gates, prewarms and locks the
learner's DECLARED REGION (`agents.model.compile_regions`: R1 the micro-step — the only compiled
learner surface; the compiled rollout region R0 was deleted, P10-E, owner-approved 2026-10-03).

ONE COMPILED SURFACE (gen3_one_gate_per_region_v1; the torch-2.5.1 extractor-only compile and its own
parity gate were DELETED in the post-switch deletion pass, 2026-10-02). HEAD runs torch >= 2.8 only
(`utils.torch_floor`): a run trained on 2.5.1 resumes PINNED to its own commit, which still carries
that compile. Here the learner compiles as its declared regions at `arm_compile_sentinel`, after grad
checkpointing and before `learn()`; `preflight_compile_trainer` only refuses, at the trainer's compile
step, a learner the regions cannot serve.

FAIL-LOUD IS THE POINT. A silent eager fallback is a ~1.75-2x throughput regression that no metric
surfaces: the run trains correctly and simply produces fewer steps per hour, forever. So every failure
here raises `CompileTrainerError` (FATAL_CONFIG), never warns.

CPU IS REJECTED, not attempted. The compiled learner is gated and measured on CUDA only. (On torch
2.5.1 the CPU backward did not even lower — Inductor's C++ backend asserted on the damage op's
`atomic_add` scatter; on 2.8 it lowers, per `extractor_compiles_test`'s per-torch pin, but no CPU
compiled learner has been gated.)

THE COMPILED STATE IS OFF THE MODULE. The region is stored on the model (`_compiled_micro_step`),
never as a patched module or an `OptimizedModule` — so `state_dict` keys (and every checkpoint) are
unchanged.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from agents.model.compile_gate_probe import per_param_grad_errors
from agents.model.compile_gate_probe import grad_parameters  # noqa: F401 — read as `ct.grad_parameters`
from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.parity_probe import LOGPROB_BAR, VacuousParityError, require_informative


class CompileTrainerError(RuntimeError):
    """Raised when `--compile-trainer` cannot deliver a compiled learner. Always fatal."""


class VacuousCompileParityError(CompileTrainerError, VacuousParityError):
    """The parity gate was asked to judge a quantity that does not vary (gen3_fresh_parity_probe_v1)
    — e.g. a FRESH policy's legal log-probs, constant per row under the zero-init pointer head. A
    `CompileTrainerError`, so the launcher treats it as config-fatal like any other gate failure."""


# A compiled forward that disagrees with eager by more than this is a wrong kernel, not a speedup. fp32
# matmul precision 'highest' is the ONLY precision (TF32 was retired, deletion pass K2), so this is THE rule.
_MAX_NUMERIC_DRIFT = 1e-4


def check_shape_stability(*, update_rows: int, batch_size: int) -> None:
    """Refuse a config whose UPDATE does not divide into full micro-batches.

    ``update_rows`` is the REAL update size — the rows one update trains on:
    `--rollout-target-samples` (0 = ``n_steps * n_envs``; `rust_rollout.trigger.update_rows`, the one
    definition the collector's own trigger uses). It is NOT ``n_steps * n_envs`` alone: a production
    recipe sets the target independently, and judging the product of two flags that no longer set the
    update size would switch `--compile-trainer` off for a perfectly valid run (P10-E, F9).

    WHY A REMAINDER IS REFUSED. R1 (`compile_regions`) has ONE declared micro-batch signature,
    `batch_size` rows, compiled with `fullgraph=True` and locked at startup. A remainder micro-batch is
    a SECOND shape, replayed every epoch: it would be a new compiled signature after the lock (the
    sentinel's FATAL) or, with a ragged eager route, a silent ~2x slowdown for that micro-step. The
    route is deleted (`gen3_r1_no_ragged_v1`): the collector refuses such an update at every buffer
    build (`RustCollector._ensure_buffer`) and R1's dispatcher refuses the call. This check says so at
    STARTUP, decidable from the argv, with a concrete batch size that fits.

    Raises `CompileTrainerError`; pure, so the rule is testable without a GPU.
    """
    rows = int(update_rows)
    if batch_size and rows % int(batch_size) != 0:
        raise CompileTrainerError(
            f"--compile-trainer needs an update that divides evenly into micro-batches, but an update "
            f"trains on {rows:,} rows (--rollout-target-samples, or n_steps*n_envs when it is 0) "
            f"and that leaves a remainder of {rows % int(batch_size):,} against "
            f"--batch-size {batch_size}.\n"
            "That remainder micro-batch is a SECOND batch shape the one compiled learner graph "
            "does not declare, replayed every epoch — and the collector refuses such an update.\n"
            f"Adjust --batch-size to a divisor of {rows:,} (e.g. "
            f"{_largest_divisor_at_most(rows, int(batch_size))}) or --rollout-target-samples to a "
            f"multiple of lcm(--batch-size, --n-envs), or drop --compile-trainer.")


def _largest_divisor_at_most(n: int, cap: int) -> int:
    """A concrete suggestion beats 'pick a divisor' — the error should not make you do arithmetic."""
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def check_numerics(err: float, *, what: str = "features", tol: Optional[float] = None) -> str:
    """Pure verdict: a compile that changes the numbers is not a speedup. Returns the log line.

    ``err`` is max|compiled - eager| at fp32 matmul precision 'highest' and must be below ``tol`` (default
    1e-4, the features bar). Looser than the CPU path's 1e-5 because cuBLAS may pick a different reduction
    order for the fused kernels; 1e-4 still catches a wrong kernel while tolerating a reordered correct
    one. ``what`` names the quantity (``features``, ``legal_logprob``, ``value``, ``train features``).

    Written ``not (x < tol)`` so a NaN FAILS rather than sailing through.
    """
    bar = _MAX_NUMERIC_DRIFT if tol is None else float(tol)
    rule = f"{what}: fp32 max|compiled-eager| {err:.2e} < {bar:g}"
    if not (err < bar):
        raise CompileTrainerError(
            f"--compile-trainer: the compiled extractor DISAGREES with eager on {what} "
            f"(max|delta| {err:.2e} > {bar:g}, matmul precision 'highest'). A faster wrong "
            f"model is not a win — investigate before re-enabling.")
    return rule


# gen3_compile_parity_real_obs_v1 — the DECISION-level tolerances at matmul precision 'highest'.
# MEASURED after the trunk split (2026-09-28, 3,840 real eval-trace rows, ai_v14_01_base @72M and
# final): legal log-prob max|d| 2.7e-05, win-prob |dV| max 1.0e-06, pi_features max 6.9e-05; the
# broken single graph read 7.39 / 0.33 / 11.8. Each bar sits >= 10x over the healthy maximum and
# >= 1000x under the defect.
_FP32_TOL = {"features": _MAX_NUMERIC_DRIFT, "legal_logprob": LOGPROB_BAR,
             "value": 1e-4}
# The train graph: cosine between the compiled and eager gradients of the gate's own loss over the
# extractor's parameters. Healthy after the split: 1.000000 (rel err 4.7e-07); the defect: 0.778.
_MIN_GRAD_COSINE = 0.9999
# gen3_gate_grad_coverage_v1 — the PER-PARAMETER gradient rule. The global cosine above is dominated
# by the largest gradients: on the perturbed production policy (CPU, 64 fixture rows) DROPPING the
# whole gradient through the pointer head's move cells left it at 1.0000 (the path's parameters are
# a sliver of the norm), while the per-parameter relative error ||c_p - e_p|| / ||e_p|| read 0.10 on
# `damage_op.out_gain`. So every parameter whose eager gradient norm is above
# `_PARAM_GRAD_FLOOR` x the largest one is ALSO held to a per-parameter bar. The BAR is the caller's:
# it depends on the weight regime and on the loss it judges — region R1's is `compile_regions.
# R1_PARAM_BAR` (`gen3_r1_param_bar_by_regime_v1`), and `train_verdict` / `_param_verdict` REFUSE a
# per-parameter comparison without one (P10-E: the retired extractor gate's 1e-3 / 0.2 TF32-era
# bars were the silent default of a caller that forgot to pass its own, and are deleted).
_PARAM_GRAD_FLOOR = 1e-3


def fixture_index(batch: int, n_rows: int, slice_: int = 0) -> "np.ndarray":
    """Which committed fixture rows a batch of ``batch`` takes: slice 0 = every row in order, tiled;
    slice 1 = the SECOND HALF of the rows, tiled — an independent batch of the same shape (the
    canary's confirmation re-run, `compile_canary`)."""
    import numpy as np

    if int(slice_) == 0 or n_rows < 2:
        return np.arange(int(batch)) % n_rows
    half = n_rows // 2
    return half + np.arange(int(batch)) % (n_rows - half)


def _parity_obs(obs_dim: int, batch: int, device: Any,
                slice_: int = 0) -> Tuple[Dict[str, "torch.Tensor"], Any]:
    """The committed REAL rows as the gate's obs dict + their legal-action masks (numpy bool).

    Rows are repeated when ``batch`` exceeds the fixture. Raises `ParityFixtureError` (turned into
    a `CompileTrainerError` by the caller) when the fixture is missing or stale — never zeros.
    """
    rows, mask = load_parity_rows(obs_dim)
    idx = fixture_index(int(batch), len(rows), slice_)
    return ({"observation": torch.as_tensor(rows[idx], device=device)}, mask[idx])


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
                      allow_vacuous: bool = False) -> List[str]:
    """The no-grad parity gate over each readout quantity. Raises `CompileTrainerError` or returns
    one rule line per quantity: each quantity uses its `_FP32_TOL` bar on max|compiled - eager|.

    FAIL-CLOSED ON A VACUOUS COMPARISON (gen3_fresh_parity_probe_v1): every compared quantity of the
    EAGER arm must vary by more than its own bar (`parity_probe.spread`), else
    `VacuousCompileParityError` — a fresh policy's legal log-probs are constant per row and pass any
    miscompile. ``allow_vacuous=True`` is for a caller that has ALREADY judged the same graph on a
    perturbed, informative copy of the weights (the gate's fresh path) and only it."""
    if not allow_vacuous:
        _require_informative({k: v for k, v in eager.items() if k in _FP32_TOL}, _FP32_TOL,
                             "--compile-trainer parity (decision readout)")
    lines = []
    for key in ("features", "legal_logprob", "value"):
        if key not in eager:
            continue
        if key not in compiled:
            raise CompileTrainerError(f"--compile-trainer: the compiled arm produced no {key!r}")
        err = float((compiled[key] - eager[key]).abs().max())
        lines.append(check_numerics(err, what=key, tol=_FP32_TOL[key]))
    return lines


def _param_verdict(compiled: Dict[str, "torch.Tensor"], eager: Dict[str, "torch.Tensor"],
                   param_names: Optional[List[str]], *, allow_vacuous: bool = False,
                   bar: Optional[float] = None,
                   bar_by_index: Optional[Dict[int, float]] = None) -> Optional[str]:
    """The per-parameter gradient rule (fp32 only). None when the arms carry no ``grad_sizes``
    (a hand-built verdict input). The ``bar`` is REQUIRED once they do: there is no default bar
    (`_PARAM_GRAD_FLOOR`'s comment), so a caller that forgot the regime-selected one is a typed
    refusal, never a quietly stricter or looser gate. ``bar_by_index`` (parameter index -> bar) is a
    caller-DECLARED per-parameter exception to ``bar`` (region R1's unmoved zero-init parameters,
    `compile_regions.unmoved_parameters`, judged at the FRESH bar); the rule line counts them."""
    if "grad_sizes" not in eager or "grad_sizes" not in compiled:
        return None
    if bar is None:
        raise CompileTrainerError(
            "--compile-trainer: the per-parameter gradient rule needs the caller's own bar "
            "(`param_bar` — region R1's is `compile_regions.R1_PARAM_BAR[regime]`); there is no "
            "default (the retired extractor gate's 1e-3 was TF32-era and would judge R1's healthy "
            "fp32 noise as a miscompile)")
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
    over = dict(bar_by_index or {})

    def bar_of(i: int) -> float:
        return float(over.get(i, bar))

    def nm(i: int) -> str:
        return param_names[i] if param_names and i < len(param_names) else f"#{i}"
    main = [(i, e) for i, e in errs if i not in over]
    excepted = [(i, e) for i, e in errs if i in over]
    rule = ""
    if main:
        worst_i, worst = max(main, key=lambda t: (not (t[1] == t[1]), t[1]))   # NaN sorts worst
        rule = (f"per-param grad rel err max {worst:.2e} ({nm(worst_i)}) over {len(main)} params "
                f"<= {bar:g}")
    if excepted:
        xi, xw = max(excepted, key=lambda t: (not (t[1] == t[1]), t[1]))
        rule = (rule + "; " if rule else "") + (
            f"{len(excepted)} declared-exception param(s) max {xw:.2e} ({nm(xi)}) <= {bar_of(xi):g}")
    bad = [(nm(i), e) for i, e in errs if not (e <= bar_of(i))]
    if bad:
        raise CompileTrainerError(
            f"--compile-trainer: the compiled TRAIN graph's gradient DISAGREES with eager on "
            f"{len(bad)} parameter(s) — {rule} FAILED (e.g. "
            f"{', '.join(f'{n} {e:.2e}' for n, e in bad[:5])}). A backward miscompile in a path "
            f"the global cosine cannot resolve. A faster wrong model is not a win — investigate "
            f"before re-enabling.")
    return rule


def train_verdict(*, eager: Dict[str, "torch.Tensor"], compiled: Dict[str, "torch.Tensor"],
                  allow_vacuous: bool = False, param_names: Optional[List[str]] = None,
                  param_bar: Optional[float] = None,
                  param_bar_by_index: Optional[Dict[int, float]] = None) -> str:
    """The TRAIN-graph parity gate: forward features (the numerics rule) AND the gradient's cosine:
    cos(compiled_grad, eager_grad) >= 0.9999. The eager arm's features must vary across rows and its
    gradient must be non-zero (`VacuousCompileParityError` otherwise; an all-zero gradient has cosine 1.0
    with anything's zero).

    When the arms carry per-parameter sizes (``grad_sizes``, region R1's arms do), every parameter above
    the floor is ALSO held to ``param_bar`` (`_param_verdict`) — REQUIRED then, with no default."""
    if not allow_vacuous:
        _require_informative({"features": eager["features"], "grad": eager["grad"]},
                             {"features": _MAX_NUMERIC_DRIFT, "grad": 0.0},
                             "--compile-trainer parity (train graph)")
    feat = check_numerics(float((compiled["features"] - eager["features"]).abs().max()),
                          what="train features")
    cos = _cos(compiled["grad"], eager["grad"])
    if not (cos >= _MIN_GRAD_COSINE):
        raise CompileTrainerError(
            f"--compile-trainer: the compiled TRAIN graph's gradient DISAGREES with eager — "
            f"cosine {cos:.6f} < {_MIN_GRAD_COSINE} on the gate's own loss over the "
            f"extractor's parameters. PPO would be stepping along the wrong direction. A "
            f"faster wrong model is not a win — investigate before re-enabling.")
    per_param = _param_verdict(compiled, eager, param_names, allow_vacuous=allow_vacuous,
                               bar=param_bar, bar_by_index=param_bar_by_index)
    return "; ".join(x for x in (feat, f"grad cosine {cos:.6f} >= {_MIN_GRAD_COSINE}",
                                 per_param) if x)


def resolve_device(fe: Any) -> "torch.device":
    """The learner's device, as its own function so the CPU refusal has a seam to test through."""
    # `fe` is deliberately `Any` (the extractor arrives through SB3), so `.parameters()` is too.
    return next(fe.parameters()).device  # type: ignore[no-any-return]


# ------------------------------------------------------------------------------------------------
# gen3_compile_sentinel_v1 — `compile_control`'s phases for the declared region: reset, install +
# gate, prewarm, lock, attach
# ------------------------------------------------------------------------------------------------

def _prewarm_obs(model: Any, batch: int, slice_: int = 0) -> Dict[str, "torch.Tensor"]:
    """A batch of ``batch`` rows over every key of the policy's observation space: `observation` from
    the committed REAL rows, `action_mask` from their masks, every other key zeros of its space's
    shape and dtype (`compile_regions.weights_regime`'s eager freshness read)."""
    import numpy as np

    policy = model.policy
    device = resolve_device(policy.features_extractor)
    space = policy.observation_space
    rows, mask = load_parity_rows(int(space.spaces["observation"].shape[0]))
    idx = fixture_index(int(batch), len(rows), slice_)
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


def preflight_compile_trainer(model: Any, enabled: bool, *,
                              emit: Optional[Callable[[str], None]] = None) -> None:
    """The trainer's compile step for `--compile-trainer`: refuse a learner the declared regions
    cannot serve, and say what will be compiled. Compiles NOTHING — the regions are installed, gated
    and prewarmed at `arm_compile_sentinel` (after grad checkpointing, the forward reads it).

    A no-op when `enabled` is False (nothing is touched, so an off run is byte-identical). Raises
    `CompileSentinelError` on a torch the sentinel does not support or whose internals drifted
    (`compile_control.require_supported_torch`), and `CompileTrainerError` on a policy without a features extractor, a non-CUDA learner, or a learner
    without the micro-step (`_micro_static`: the instrumented PPO's R1 declaration)."""
    if not enabled:
        return
    # The sentinel's torch check FIRST — decidable before anything else, and the compile path is the
    # only one that runs it (P10-D): an unsupported / drifted torch is a `CompileSentinelError`, i.e.
    # the caller's FATAL_CONFIG with the reason, never an import-time crash.
    from agents.model import compile_control as _cc
    _cc.require_supported_torch()
    policy = getattr(model, "policy", None)
    fe = getattr(policy, "features_extractor", None)
    if fe is None:
        raise CompileTrainerError(
            "--compile-trainer: this policy has no `features_extractor`. The flag compiles the "
            "Gen3 learner's declared regions; it cannot be used with a stock SB3 policy.")
    device = resolve_device(fe)
    if device.type != "cuda":
        raise CompileTrainerError(
            f"--compile-trainer requires CUDA, but the model is on {device.type!r}.\n"
            "The compiled learner (the declared region, its startup gate, the canary and the "
            "measured speedup) exists on CUDA only; a CPU compiled learner has never been gated. "
            "Pass --device cuda, or drop --compile-trainer.")
    if not hasattr(model, "_micro_static"):
        raise CompileTrainerError(
            "--compile-trainer: this learner has no micro-step (`_micro_static`), so its declared "
            "region R1 cannot be built. The learner compiles ONLY as its declared region (R1 the "
            "micro-step; the extractor-only compile was deleted 2026-10-02, the rollout region R0 "
            "2026-10-03) — use the instrumented learner, or drop --compile-trainer.")
    msg = ("⚡ [CompileTrainer] ON — the learner compiles as its DECLARED REGION (R1, the micro-step) "
           "at the compile sentinel, gated there (gen3_one_gate_per_region_v1)")
    print(msg, flush=True)
    if emit is not None:
        try:
            emit(msg)
        except Exception:
            pass                          # a diagnostic must never break the run


def arm_compile_sentinel(model: Any, *, batch_size: int,
                         emit: Optional[Callable[[str], None]] = None) -> str:
    """Install, gate, prewarm and LOCK the learner's declared region, and attach the per-rollout /
    per-update checks. Called only for a compiled learner (`--compile-trainer`).

    K8 (gen3_declared_regions_v1): R1 (the micro-step), `fullgraph=True`, gated against eager on real
    rows, prewarmed at exactly its declared signature, then locked (K6: before the first real
    iteration). MUST run after `_apply_grad_checkpointing` (the forward reads `grad_checkpointing`,
    so a graph compiled before it is a stale entry) and before `learn()`. Raises
    `CompileTrainerError` (incl. `CompileSentinelError`).
    """
    fe = getattr(getattr(model, "policy", None), "features_extractor", None)
    if fe is None or not hasattr(model, "_micro_static"):
        raise CompileTrainerError(
            "--compile-trainer: the compile sentinel needs the instrumented learner (a features "
            "extractor and the micro-step `_micro_static`) — the declared region is the only "
            "compiled learner surface")
    from agents.model.compile_control import control, set_strict_errors
    # `install` pins the compile config (K1b's `donated_buffer=False` — without it the first
    # retain_graph probe crashes: measured on the consolidation's launch proof, 2026-10-01) and
    # registers the cache-limit detector and the after-lock compile counter. Idempotent.
    ctl = control(emit).install()
    set_strict_errors()

    def _say(msg: str) -> None:
        print(msg, flush=True)
        if emit is not None:
            try:
                emit(msg)
            except Exception:
                pass

    t0 = time.perf_counter()
    _say(ctl.reset())
    from agents.model import compile_regions as _cr
    _say(f"[CompileRegions] installing {_cr.install(model, emit=emit)} (fullgraph=True, static "
         f"shapes; the rank probe reads R1, the optimizer step R3 is EAGER by declaration)")
    _cr.gate_regions(model, batch_size=int(batch_size), say=_say)
    line = ctl.prewarm(_cr.prewarm_calls(model, batch_size=int(batch_size)))
    _say(_cr.assert_inventory(model))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _say(f"{line} — reset + prewarm took {time.perf_counter() - t0:.1f}s")
    # K6: LOCK NOW — before the first real iteration. Every signature the steady state may reach is
    # in the declared table just prewarmed; anything else is an undeclared signature, a typed FATAL
    # naming its failing guard, never a warm-up iteration's silent absorption.
    ctl.lock("the end of startup (the region gate, the reset and the prewarm of every declared "
             "signature)")
    # K6's IN-RUN PARITY CANARY: the startup gate proves the graph at t=0, the canary at t=N.
    from agents.model.compile_canary import CANARY_EVERY, CANARY_FIRST, CompileCanary
    ctl.canary = CompileCanary(model, batch_size=int(batch_size), emit=emit)
    _say(f"🐤 [CompileCanary] armed: R1 (the learner micro-step) compiled vs eager on the K9 learner "
         f"golden's real labelled rows at update {CANARY_FIRST}, then every {CANARY_EVERY} updates "
         f"(loss + every policy gradient, B={int(batch_size)}), at the startup gate's bars; a "
         f"disagreement is confirmed before it FATALs")
    ctl.attach(model)
    return line
