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
from typing import Optional

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

# A compiled forward that disagrees with eager by more than this is a wrong kernel, not a speedup.
_MAX_NUMERIC_DRIFT = 1e-4


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


def resolve_device(fe) -> "torch.device":
    """The learner's device, as its own function so the CPU refusal has a seam to test through."""
    return next(fe.parameters()).device


def _one_step(fe, obs):
    """One forward + backward through the extractor, the shape the PPO step actually runs."""
    fe.zero_grad(set_to_none=True)
    pi, vf = fe(obs)
    (pi.square().mean() + vf.square().mean()).backward()
    return pi, vf


def _time_steps(fe, obs, reps: int) -> float:
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


def compile_trainer_extractor(model, enabled: bool, *, batch: int = 64,
                              emit=None) -> Optional[float]:
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
    obs = {"observation": torch.rand(batch, obs_dim, device=device)}

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
        raise CompileTrainerError(
            f"--compile-trainer: the learner's extractor FAILED to compile — "
            f"{type(exc).__name__}: {exc}\n"
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

    _say(f"[CompileTrainer] ON — learner fwd+bwd {eager_ms:.1f} -> {comp_ms:.1f} ms "
         f"({speedup:.2f}x) at batch {batch} on {device}")
    if dropped_debugger:
        _say("[CompileTrainer] the ObservationDebugger was DROPPED to allow the compile — dynamo "
             "cannot trace its numpy asserts. You lose that per-forward obs-integrity check for "
             "this run; every other guard is unaffected.")
    return speedup
