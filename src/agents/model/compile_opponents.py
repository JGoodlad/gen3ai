"""Compiled CPU OPPONENTS — `torch.compile` for frozen self-play / eval / teacher models.

Split out of `snapshot.py` 2026-08-16: loading a checkpoint and compiling its extractor are
different responsibilities, and every consumer of one that never needed the other was importing
both. This is the OPPONENT half (CPU, B=1, warn-and-fall-back); the LEARNER half — CUDA,
fail-loud — is `compile_trainer.py`, and the reasons they cannot be one module are at the top of
that file. `compile_prewarm.py` warms the shared on-disk Inductor cache this module reads.

Import surface: `snapshot.py` re-exports every public name here, so historical import paths
(`from agents.model.snapshot import maybe_compile_extractor`) still resolve.
"""
from __future__ import annotations

import os
import sys
import time

# Shared Inductor cache. Under `spawn` every worker re-imports and re-traces from scratch, but a
# SHARED on-disk cache turns all but the first process's CODEGEN into a hit (measured 19.1s cold ->
# 5.8s warm). This is the only compile artifact that crosses a process boundary: `torch.compile`
# returns a live Python object, so the compiled callable itself can never be handed to a spawned
# child — see `prewarm_extractor_compile`.
DEFAULT_INDUCTOR_CACHE_DIR = "/tmp/gen3ai_inductor_cache"

# A compile must beat eager by at least this much to be kept. Not a safety margin against noise so
# much as a floor on "worth the risk at all": anything under it means dynamo overhead is eating the
# fusion win, which is what a partially-traced graph looks like.
_MIN_COMPILE_SPEEDUP = 1.05
_TIMING_REPS = 12
_TIMING_WARMUP = 3

# Set once a compile has been MEASURED to pay off in this process. The validation answers "does this
# extractor's code object compile to something faster?", and `torch.compile` keys on exactly that
# code object — so the answer cannot differ for a second model in the same process. Consumers that
# load models in a LOOP (the search-teacher worker rebuilds its opponent every iteration; an eval
# worker walks several opponents) would otherwise re-pay ~15 eager forwards each time for an answer
# they already have. Deliberately process-local: a fresh process re-validates, because that is where
# a genuinely different outcome (a cold cache, a failing backend) could show up.
_COMPILE_VALIDATED = False


def _compile_warn(msg: str) -> None:
    """Report a compile problem LOUDLY.

    Falling back to eager is a ~6.5x regression on the opponent forward and a measured ~24% loss of
    end-to-end training throughput, but it is invisible — the run simply produces fewer steps per
    hour, forever, and looks healthy. So a failure goes to stderr AND (under the launcher) into the
    event stream, where it surfaces in the TUI rather than scrolling past in a worker's stdout."""
    line = f"⚠️ [CompileExtractor] {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        from main.launcher.ipc import emit
        emit(line)
    except Exception:
        pass                                          # standalone / no launcher pipe: stderr is enough


def _inductor_cache_dir() -> str:
    """Set (once) and return the shared Inductor cache dir. Deliberately NOT an import-time side
    effect — `snapshot.py` is imported by the prober, eval workers and offline tooling that never
    compile anything, and a module that mutates the environment on import is the kind of thing that
    is impossible to reason about later. Inductor reads this when it first codegens, which is always
    inside `maybe_compile_extractor`, so setting it here is early enough."""
    return os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", DEFAULT_INDUCTOR_CACHE_DIR)


class CompileExtractorError(RuntimeError):
    """Raised under `--compile-opponents-strict` when a compile does not deliver its speedup."""


def maybe_compile_extractor(model, enabled: bool, label: str = "opponent",
                            hide_cuda: bool = False, strict: bool = False) -> bool:
    """`torch.compile` a frozen model's FEATURE EXTRACTOR for CPU inference. Returns True if applied.

    WHY THE EXTRACTOR AND NOT THE OP. The 2026-06-30 attempt compiled only `DamageOperator.forward`
    inside `policy.get_distribution` and measured **0.70× (slower)** — dynamo overhead on a graph that
    still ran ~10k eager dispatches around it. Compiling the WHOLE extractor instead gives one fused
    graph: measured **6.5× at B=1 on CPU** on the literal production config (6.37 -> 0.98 ms), 1 graph
    / 0 graph breaks, values within 5.1e-7 of eager.

    THREE PROPERTIES THAT MAKE THIS CHEAP (all measured, `tmp/compile_share_probe.py`):
      * the compile is keyed on the CODE OBJECT, not the module instance — a second extractor built in
        the same process compiles in **0.00s**, so every later pool snapshot is free;
      * parameters are graph INPUTS, not baked constants — `load_state_dict` of a different checkpoint
        does NOT recompile (dynamo frame count unchanged);
      * we patch the BOUND `fe.forward`, never the module — `torch.compile(module)` would prefix every
        state_dict key with `_orig_mod.` and break resume/load.

    `hide_cuda` MUST be True in an env worker and False in the learner. Compiling even a CPU-device
    model inside a CUDA-VISIBLE process INITIALISES CUDA and takes ~252 MiB of card; ×48 workers is
    ~12 GB and is exactly the OOM that killed the June attempt. This used to be INFERRED from
    `torch.cuda.is_initialized()` as a proxy for "am I a worker" — which was correct only by accident
    of the call sites, and would have silently blinded the learner's GPU the first time anyone called
    this from the main process before CUDA was touched. It is now the caller's explicit declaration.

    NOTE ON `suppress_errors`: this deliberately does NOT set it. It used to, because ONE op
    crashed Inductor codegen (`BeliefHead.species_posterior`, now fixed) — and globally
    suppressing backend errors to work
    around it meant every OTHER compile failure also became a silent per-frame eager fallback. A
    failure here should be loud, caught, and reported. Late failures are handled by
    `_eager_fallback_on_error`, which degrades that one model instead of the whole process.
    """
    if not enabled:
        return False
    import torch

    fe = getattr(getattr(model, "policy", None), "features_extractor", None)
    if fe is None:
        return False
    if hide_cuda:
        if torch.cuda.is_initialized():
            # Refuse rather than pretend: the context already exists, so hiding the device now buys
            # nothing and the caller has mis-declared which process it is in.
            msg = (f"{label}: DISABLED — hide_cuda=True but this process has already initialised "
                   f"CUDA; refusing to compile (it would add a ~252 MiB context per worker).")
            _compile_warn(msg)
            if strict:
                raise CompileExtractorError(msg)
            return False
        os.environ["CUDA_VISIBLE_DEVICES"] = ""       # no per-worker CUDA context (the June OOM)
    cache_dir = _inductor_cache_dir()
    if hasattr(fe, "disable_observation_debugger"):
        fe.disable_observation_debugger()             # numpy asserts inside forward; dynamo can't trace

    global _COMPILE_VALIDATED
    revalidate = not _COMPILE_VALIDATED
    original = fe.forward
    warmup = _compile_warmup_obs(fe)
    try:
        eager_ms = _time_forward(fe.forward, warmup) if revalidate else 0.0
        compiled = torch.compile(fe.forward)
        # Force the compile HERE, inside the try — `torch.compile` is lazy, so without this the real
        # compilation happens on the first live decision, far outside this handler.
        with torch.no_grad():
            compiled(warmup)
        comp_ms = _time_forward(compiled, warmup) if revalidate else 0.0
    except Exception as e:                            # by default never take a run down for a perf knob
        fe.forward = original
        msg = f"{label}: DISABLED — {type(e).__name__}: {str(e)[:200]}"
        _compile_warn(msg)
        if strict:
            raise CompileExtractorError(msg) from e
        return False

    if not revalidate:
        # Already proven in this process — keep it without re-timing. STILL LOG IT: a silent success
        # is indistinguishable from "never ran" in a run log, and that is not hypothetical — the
        # eval-worker opponent compile looked missing for exactly this reason until it was verified
        # by instrumenting the call. Coverage you cannot see is coverage you will doubt.
        fe.forward = _eager_fallback_on_error(compiled, original, label)
        print(f"[CompileExtractor] {label}: ON (reused this process's validated compile)", flush=True)
        return True

    speedup = eager_ms / comp_ms if comp_ms > 0 else 0.0
    if speedup < _MIN_COMPILE_SPEEDUP:
        # Compiling can LOSE: the June attempt measured 0.70× because dynamo overhead exceeded the
        # fusion win on a fragmented graph. Measure, then keep or revert — never assume.
        fe.forward = original
        msg = (f"{label}: REVERTED to eager — compiled {comp_ms:.2f} ms vs eager {eager_ms:.2f} ms "
               f"({speedup:.2f}x) is below the {_MIN_COMPILE_SPEEDUP:.2f}x floor; the graph is "
               f"probably fragmented. Expect roughly a {1.0 / max(speedup, 1e-9):.1f}x slower "
               f"opponent forward than a healthy compile gives.")
        _compile_warn(msg)
        if strict:
            raise CompileExtractorError(msg)
        return False

    fe.forward = _eager_fallback_on_error(compiled, original, label)
    _COMPILE_VALIDATED = True
    print(f"[CompileExtractor] {label}: ON — {eager_ms:.2f} -> {comp_ms:.2f} ms "
          f"({speedup:.1f}x, cache {cache_dir})", flush=True)
    return True


def _eager_fallback_on_error(compiled, original, label: str):
    """Wrap the compiled callable so it degrades to eager instead of killing the caller.

    TWO things it guards:

    1. A LATE compile failure. `torch.compile` guards on input properties, so a shape/dtype it has
       not seen can trigger a fresh trace at CALL time — long after the load-time try/except
       returned. Opponent inference is always B=1 so that should not happen, but "should not" is not
       a guarantee worth a crashed 3-hour run. This is the targeted replacement for the old global
       `suppress_errors=True`: same never-crash property, scoped to ONE model, and it SAYS so
       instead of silently running eager while claiming to be compiled.

    2. GRAD-ENABLED calls. The compiled artifact here is built for INFERENCE. Under `requires_grad`
       dynamo hands the graph to AOTAutograd, which must also lower the BACKWARD — and Inductor's
       CPU backward codegen fails on this model's scatter/`index_add` (the HP-type belief); that is
       the documented reason the June `--compile-damage-op` integration was inference-only. Every
       frozen-opponent consumer runs under `no_grad`, but the PROBER does not: gradient saliency
       backprops through this same extractor. So route grad-enabled calls to eager. Value-identical
       either way, and it keeps `maybe_compile_extractor` safe to apply to any non-training model
       rather than only the ones we have audited for no_grad."""
    import torch

    state = {"failed": False}

    def guarded(obs):
        if state["failed"] or torch.is_grad_enabled():
            return original(obs)
        try:
            return compiled(obs)
        except Exception as e:
            state["failed"] = True
            _compile_warn(f"{label}: FELL BACK to eager (this model is now ~6x slower) — "
                          f"{type(e).__name__}: {str(e)[:200]}")
            return original(obs)

    return guarded


def _time_forward(fn, obs, reps: int = _TIMING_REPS) -> float:
    """min-of-N ms for one forward. min, not mean: contention only ever ADDS time, and this runs at
    worker startup while other workers are still spawning."""
    import torch
    with torch.no_grad():
        for _ in range(_TIMING_WARMUP):
            fn(obs)
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(obs)
            best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _compile_warmup_obs(fe) -> dict:
    """A zero observation of the right width — enough to force compilation of the B=1 graph."""
    import torch
    layout = getattr(fe, "layout", None)
    dim = layout["total_dim"] if layout else fe.observation_space.shape[0]
    return {"observation": torch.zeros(1, dim)}
