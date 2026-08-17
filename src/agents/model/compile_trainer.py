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

import time
from typing import Any, Callable, Optional, Tuple

import torch


class CompileTrainerError(RuntimeError):
    """Raised when `--compile-trainer` cannot deliver a compiled learner. Always fatal."""


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
_MAX_NUMERIC_DRIFT = 1e-4


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
    try:
        import torch._dynamo.config as _c
        return int(getattr(_c, "cache_size_limit", 8))
    except Exception:
        return 8


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


def check_numerics(err: float) -> None:
    """Pure verdict: a compile that changes the numbers is not a speedup.

    Tolerance is looser than the CPU path's 1e-5 because cuBLAS/cuDNN may pick a different reduction
    order or TF32 for the fused kernels; 1e-4 still catches a wrong kernel while tolerating a
    differently-ordered correct one.
    """
    if not (err < _MAX_NUMERIC_DRIFT):
        raise CompileTrainerError(
            f"--compile-trainer: the compiled extractor DISAGREES with eager (max|delta| {err:.2e} "
            f"> {_MAX_NUMERIC_DRIFT:g}). A faster wrong model is not a win — investigate before "
            f"re-enabling.")


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

    # The ObservationDebugger runs NUMPY assertions inside `forward`; dynamo cannot trace them at
    # all (it dies building a guard over a numpy bool), so this is compile-or-debugger, not both.
    # It attaches at log_level >= PERIODIC — i.e. it IS on in production — so this is a real
    # trade-off rather than a debug-mode detail, and it gets its own line rather than happening
    # quietly. The opponent path drops it for the same reason.
    dropped_debugger = False
    if hasattr(fe, "disable_observation_debugger"):
        dropped_debugger = bool(fe.disable_observation_debugger())

    was_training = fe.training
    fe.train()                                 # the backward path is what we are compiling
    # ZEROS, not `torch.rand` — matching `snapshot._zero_obs` on the opponent path, and for a reason
    # this module learned the hard way. A random float vector is NOT a valid observation: the
    # ObservationDebugger rejects it outright ("CRITICAL INTEGRITY FAILURE: multiple active Pokemon"),
    # so it can drive the forward down branches no real battle reaches. An all-zero obs is the
    # canonical "nothing known" state — every categorical id is 0, every flag clear — which is
    # structurally legal, and it was ALSO what turned a mystifying `CUDA error: invalid configuration
    # argument` into the honest `OutOfMemoryError` underneath it when this was being debugged.
    obs = {"observation": torch.zeros(batch, obs_dim, device=device)}

    original = fe.forward
    try:
        torch._dynamo.config.suppress_errors = False   # a partial compile must be LOUD, not silent
        eager_ms = _time_steps(fe, obs, _VALIDATE_REPS)
        with torch.no_grad():
            ref_pi, ref_vf = fe(obs)
            ref_pi, ref_vf = ref_pi.clone(), ref_vf.clone()

        compiled = torch.compile(original)
        fe.forward = compiled
        comp_ms = _time_steps(fe, obs, _VALIDATE_REPS)
        with torch.no_grad():
            got_pi, got_vf = fe(obs)
    except CompileTrainerError:
        fe.forward = original
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
        if not was_training:
            fe.eval()

    err = max(float((ref_pi - got_pi).abs().max()), float((ref_vf - got_vf).abs().max()))
    try:
        check_numerics(err)
        speedup = check_speedup(eager_ms, comp_ms)
    except CompileTrainerError:
        fe.forward = original          # never leave a rejected compile installed
        raise

    prod = int(getattr(model, "batch_size", 0) or 0)
    shape_note = (f" — VALIDATION shape only; the production batch is {prod} and the measured "
                  f"speedup there is ~1.75x, NOT this number"
                  if prod and prod != batch else "")
    _say(f"[CompileTrainer] ON — learner fwd+bwd {eager_ms:.1f} -> {comp_ms:.1f} ms "
         f"({speedup:.2f}x) at batch {batch} on {device}{shape_note}")
    if dropped_debugger:
        _say("[CompileTrainer] the ObservationDebugger was DROPPED to allow the compile — dynamo "
             "cannot trace its numpy asserts. You lose that per-forward obs-integrity check for "
             "this run; every other guard is unaffected.")
    return speedup
