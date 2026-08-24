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
import shutil
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Tuple

# Shared Inductor cache. Under `spawn` every worker re-imports and re-traces from scratch, but a
# SHARED on-disk cache turns all but the first process's CODEGEN into a hit (measured 19.1s cold ->
# 5.8s warm). This is the only compile artifact that crosses a process boundary: `torch.compile`
# returns a live Python object, so the compiled callable itself can never be handed to a spawned
# child — see `prewarm_extractor_compile`.
DEFAULT_INDUCTOR_CACHE_DIR = "/tmp/gen3ai_inductor_cache"

# A compile must beat eager by at least this much to be kept. A floor on "worth the risk at all",
# not a safety margin against noise — the noise is handled by the MEASUREMENT below, not by moving
# this number. (It was proposed to drop it to 0.7 after the 2026-08-24 launch failures; that would
# have widened a broken instrument instead of fixing it. The gate was uninformative in BOTH
# directions — see `_measure_arms`.)
_MIN_COMPILE_SPEEDUP = 1.05

# The measurement under the floor. `_TIMING_SAMPLES` medians per arm, each sample a min over
# `_TIMING_REPS` forwards, with the arms ALTERNATED (see `_measure_arms`). Total per arm =
# _TIMING_WARMUP + _TIMING_SAMPLES * _TIMING_REPS forwards.
_TIMING_SAMPLES = 5
_TIMING_REPS = 4
_TIMING_WARMUP = 3

# QUORUM. A single worker landing below the floor is a reading, not a diagnosis: 48 workers
# measuring the same model on the same box spread 7.7x on the EAGER arm alone (14.9-115.7 ms,
# 2026-08-24), and the same checkpoint that scored 0.78x in one worker scored 6.3x median across
# 48/48 minutes later. So under `--compile-opponents-strict` a below-floor reading is fatal only
# when a MAJORITY-ISH fraction of the reporting workers agree — a systemic failure — and never on
# one worker's draw. `_QUORUM_MIN_REPORTS` keeps the very first readings from deciding anything.
_QUORUM_REVERT_FRACTION = 0.25
_QUORUM_MIN_REPORTS = 4

# Where the cross-process tally lives. Set by `arm_compile_quorum` in the trainer BEFORE any env
# worker exists, and inherited by every spawn/forkserver child through the environment. Unset (the
# prober, a standalone eval worker, a test) ⇒ the tally is process-local and the quorum degenerates
# to "this process" — documented in `_record_verdict`.
COMPILE_QUORUM_ENV = "GEN3AI_COMPILE_QUORUM_DIR"

# The process-local fallback tally. Deliberately module-level and NOT reset per call: a process that
# validates several opponents should accumulate, exactly as the file tally does across processes.
_LOCAL_TALLY = {"reverts": 0, "total": 0}

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
    """Raised under `--compile-opponents-strict` when the compile path fails.

    Two different conditions, deliberately not symmetric:

    * a compile that ERRORS (backend crash, a mis-declared `hide_cuda`) is fatal in THIS process
      immediately — it is a fact, not a reading;
    * a compile that merely measures below the floor is fatal only on a QUORUM (`_quorum_is_fatal`),
      because a single timing verdict was measured to be worth nothing (see `_measure_arms`)."""


def maybe_compile_extractor(model: Any, enabled: bool, label: str = "opponent",
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

    HOW THE KEEP/REVERT DECISION IS MADE (rewritten 2026-08-24 after it killed three launches on
    timing noise): both arms are warmed identically, then timed ALTERNATED, and the verdict is the
    ratio of their MEDIANS — see `_measure_arms` for the measured spreads that forced this. Under
    `strict`, a below-floor reading warns with its numbers and is fatal only when a quorum of the
    reporting compiles agree (`_quorum_is_fatal`).

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
    eager_series: List[float] = []
    comp_series: List[float] = []
    try:
        compiled = torch.compile(fe.forward)
        # Force the compile HERE, inside the try — `torch.compile` is lazy, so without this the real
        # compilation happens on the first live decision, far outside this handler.
        with torch.no_grad():
            compiled(warmup)
        if revalidate:
            # BOTH arms are timed together, alternated, after an IDENTICAL warm-up. Timing them
            # one-after-the-other is what made the old gate uninformative: the eager arm ran first,
            # cold, while the box was still spawning 47 other workers.
            eager_series, comp_series = _measure_arms(original, compiled, warmup)
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

    eager_ms = statistics.median(eager_series)
    comp_ms = statistics.median(comp_series)
    speedup = eager_ms / comp_ms if comp_ms > 0 else 0.0
    if speedup < _MIN_COMPILE_SPEEDUP:
        # Compiling can LOSE: the June attempt measured 0.70x, dynamo overhead exceeding the fusion
        # win. Measure, then keep or revert — never assume. What this branch may NOT do is say WHY:
        # a ratio of two timings cannot separate a fragmented graph from a busy box, and the text
        # that asserted "the graph is probably fragmented" sent three launch failures down the wrong
        # investigation. Report the measurement; let the reader diagnose.
        fe.forward = original
        reverts, total = _record_verdict(reverted=True)
        msg = (f"{label}: REVERTED to eager — median eager {eager_ms:.2f} ms vs median compiled "
               f"{comp_ms:.2f} ms = {speedup:.2f}x, below the {_MIN_COMPILE_SPEEDUP:.2f}x floor. "
               f"{_describe_measurement(eager_series, comp_series)} "
               f"This opponent's forward now runs eager. "
               f"Quorum so far: {reverts}/{total} compiles below the floor.")
        _compile_warn(msg)
        if strict and _quorum_is_fatal(reverts, total):
            raise CompileExtractorError(
                f"{msg} FATAL under --compile-opponents-strict: more than "
                f"{_QUORUM_REVERT_FRACTION:.0%} of the {total} compiles that have reported are "
                f"below the floor, which is a systemic failure rather than one worker's draw.")
        return False

    fe.forward = _eager_fallback_on_error(compiled, original, label)
    _COMPILE_VALIDATED = True
    _record_verdict(reverted=False)
    print(f"[CompileExtractor] {label}: ON — median {eager_ms:.2f} -> {comp_ms:.2f} ms "
          f"({speedup:.1f}x, cache {cache_dir})", flush=True)
    return True


def _eager_fallback_on_error(compiled: Callable[[Any], Any], original: Callable[[Any], Any],
                             label: str) -> Callable[[Any], Any]:
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

    def guarded(obs: Any) -> Any:
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


def _warm_arm(fn: Callable[[Any], Any], obs: Any, calls: int = _TIMING_WARMUP) -> None:
    """Run `calls` untimed forwards. Called on BOTH arms before EITHER is timed — the single most
    important property of this measurement. The old gate timed eager first and compiled second, so
    the eager arm paid every first-touch cost (allocator, page faults, the process's own spawn
    contention) that the compiled arm then measured warm. That asymmetry is free speedup for the
    compiled arm in one regime and free slowdown in another, and both directions were observed."""
    import torch
    with torch.no_grad():
        for _ in range(calls):
            fn(obs)


def _time_forward(fn: Callable[[Any], Any], obs: Any, reps: int = _TIMING_REPS) -> float:
    """ONE sample: min-of-`reps` ms for a single forward. min, not mean, WITHIN a sample: contention
    only ever ADDS time. Does NOT warm — `_measure_arms` warms both arms up front so neither can be
    measured cold while the other is warm."""
    import torch
    with torch.no_grad():
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(obs)
            best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _measure_arms(eager: Callable[[Any], Any], compiled: Callable[[Any], Any],
                  obs: Any) -> Tuple[List[float], List[float]]:
    """MEDIAN-OF-N, ALTERNATED. Returns (eager samples, compiled samples), both in ms.

    Why this shape, from the 2026-08-24 production failures. The old gate compared ONE eager timing
    to ONE compiled timing, and killed three launches on the ratio. Measured on the same model and
    the same box across 48 workers: the EAGER arm alone spread 14.94-115.71 ms (7.7x), the compiled
    arm 2.08-17.90 ms. A single pair therefore reports whichever end of each spread it happened to
    land on — one worker scored 0.78x (FATAL under strict) while the same checkpoint scored 6.3x
    median across 48/48 minutes later, and one failure landed at EXACTLY the 1.05x floor. The gate
    was equally uninformative the other way: a cold-measured eager arm lets a genuinely broken
    compile pass at 29x, so "0 workers below the floor" was never evidence of health.

    Three properties, each fixing one half of that:

    * MEDIAN of `_TIMING_SAMPLES` samples per arm — a single outlier sample cannot move the verdict,
      where a single outlier timing decided it before.
    * ALTERNATION — the arms are interleaved sample-by-sample rather than run back-to-back, so a
      load regime that drifts during the measurement (this box usually carries a trainer, and 47
      sibling workers are compiling at the same moment) hits both arms alike instead of landing
      entirely on whichever arm ran during it. The per-round ORDER also flips, so neither arm
      permanently owns the "first call after the other arm ran" position.
    * an identical warm-up on both arms BEFORE any timing (`_warm_arm`).

    It costs `_TIMING_SAMPLES * _TIMING_REPS` = 20 forwards per arm against the old 12, i.e. ~8 more
    eager forwards per process — tens of ms next to the ~30 s compile it is validating."""
    _warm_arm(eager, obs)
    _warm_arm(compiled, obs)
    eager_ms: List[float] = []
    comp_ms: List[float] = []
    for i in range(_TIMING_SAMPLES):
        if i % 2 == 0:
            eager_ms.append(_time_forward(eager, obs))
            comp_ms.append(_time_forward(compiled, obs))
        else:
            comp_ms.append(_time_forward(compiled, obs))
            eager_ms.append(_time_forward(eager, obs))
    return eager_ms, comp_ms


def _describe_measurement(eager_series: List[float], comp_series: List[float]) -> str:
    """The measurement, spelled out. A verdict that prints only its own conclusion cannot be
    second-guessed by the person reading the log at 2 a.m., and this one was wrong three times."""
    fmt = lambda xs: "[" + ", ".join(f"{x:.2f}" for x in xs) + "]"   # noqa: E731
    return (f"Samples (ms, alternated, min-of-{_TIMING_REPS} each): "
            f"eager {fmt(eager_series)} compiled {fmt(comp_series)}.")


# ── The quorum ────────────────────────────────────────────────────────────────────────────────

def arm_compile_quorum(run_dir: str | None = None) -> str:
    """Create a FRESH shared tally directory for this process tree and publish it in `os.environ`.

    Call ONCE in the trainer, before the vec env exists: every `SubprocVecEnv` worker, forkserver
    child and Popen'd eval worker inherits the environment, so they all tally into one place and
    `--compile-opponents-strict` can ask "how many of us reverted?" instead of "did I revert?".
    The directory is CLEARED here, so each launcher restart starts a fresh count rather than being
    protected by the previous window's verdicts."""
    base = (os.path.join(run_dir, ".compile_quorum") if run_dir
            else os.path.join(tempfile.gettempdir(), f"gen3ai_compile_quorum_{os.getpid()}"))
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base, exist_ok=True)
    os.environ[COMPILE_QUORUM_ENV] = base
    return base


def _record_verdict(reverted: bool) -> Tuple[int, int]:
    """Record this compile's verdict and return `(reverts, total)` OBSERVED SO FAR.

    One empty file per verdict, named `<pid>-<ns>.{ok,revert}`: a create-and-count needs no lock, no
    server and no cleanup, and a worker that cannot write (read-only FS, a deleted dir) falls back to
    the process-local tally rather than failing a compile over bookkeeping.

    ⚠️ THE LIMIT, stated rather than hidden: this is a PREFIX estimate. A worker sees only the
    verdicts written before it looked, so the fraction it reads is "of the workers that have reported
    so far", not of all N. Consequences, both deliberate: an isolated bad reading can never be fatal
    (it is 1 of a growing denominator, and the first `_QUORUM_MIN_REPORTS` decide nothing), while a
    systemic failure trips as soon as enough workers have agreed — which is the asymmetry strict mode
    wants. Within one run the tally also spans the whole process tree and the whole restart window,
    so a healthy startup does dilute a later mid-run regression; the alternative (a real barrier
    across 48 spawned workers) is cross-process plumbing this perf knob does not justify."""
    d = os.environ.get(COMPILE_QUORUM_ENV)
    if d:
        try:
            os.makedirs(d, exist_ok=True)
            suffix = "revert" if reverted else "ok"
            with open(os.path.join(d, f"{os.getpid()}-{time.time_ns()}.{suffix}"), "w"):
                pass
            names = os.listdir(d)
            return sum(1 for n in names if n.endswith(".revert")), len(names)
        except OSError:
            pass                                      # unwritable/vanished: the local tally is honest too
    _LOCAL_TALLY["total"] += 1
    _LOCAL_TALLY["reverts"] += int(reverted)
    return _LOCAL_TALLY["reverts"], _LOCAL_TALLY["total"]


def _quorum_is_fatal(reverts: int, total: int) -> bool:
    """Is a below-floor reading a SYSTEMIC failure? Strict mode's only fatal condition for the timing
    gate. `>` not `>=` on the fraction, so exactly 1-of-4 (the boundary a 48-worker run reaches early
    and often) warns rather than kills — boundary artifacts are how this gate failed before."""
    return total >= _QUORUM_MIN_REPORTS and reverts > _QUORUM_REVERT_FRACTION * total


def _compile_warmup_obs(fe: Any) -> Dict[str, Any]:
    """A zero observation of the right width — enough to force compilation of the B=1 graph.

    It carries `action_mask` as well as `observation`, and the mask is FLOAT32, because dynamo guards
    on a dict's KEY SET and on dtype exactly as hard as it guards on shape. Every real opponent call
    arrives through `policy.get_distribution` with both keys and a preprocessed float mask, so
    warming with `observation` alone leaves the first LIVE decision to re-trace the whole extractor —
    measured at 19.5 s against a 3.8 ms steady state in the cf producer (53870dd), where it was
    charged to whichever record happened to run first."""
    import torch
    layout = getattr(fe, "layout", None)
    dim = layout["total_dim"] if layout else fe.observation_space.shape[0]
    return {"observation": torch.zeros(1, dim, dtype=torch.float32),
            "action_mask": torch.ones(1, _n_actions(fe), dtype=torch.float32)}


def _n_actions(fe: Any) -> int:
    """The action-mask width, from the extractor's own space when it has a Dict one."""
    space = getattr(fe, "observation_space", None)
    try:
        return int(space["action_mask"].shape[0])     # type: ignore[index]
    except Exception:
        from agents.action.constants import ACTION_SPACE_SIZE
        return int(ACTION_SPACE_SIZE)
