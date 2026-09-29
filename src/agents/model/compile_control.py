"""COMPILE CONTROL — the one adapter over `torch._dynamo`, and the sentinel that makes
`--compile-trainer` unable to silently churn dynamo's cache or fall back to eager.

gen3_compile_sentinel_v1. The owner (2026-09-28): "I just can't stand the idea that that could be
subtly happening to us. Is there no way to force it to raise or reject?" — and: reach into torch
internals only BEHIND ONE CLEAN INTERFACE, with explicit phases. This module is that interface.
**It is the only runtime module that touches `torch._dynamo`** (the lone exception is
`team_transformer`'s in-graph `graph_break()`, which is model code dynamo traces, not control).

THE PHASES (one `CompileControl` per process — `control()`):

  1. `gate()`      — the startup parity gate (`compile_trainer_extractor`) compiles FREELY. The
                     cache-limit detector is already listening: a limit hit here is fatal too.
  2. `reset()`     — `torch._dynamo.reset()` right after the gate, BEFORE production warm-up. WHY
                     (measured, `designs/training/compile_flags.md` "The compile sentinel"): dynamo's
                     cache is keyed per CODE OBJECT (`Gen3FeaturesExtractor.forward.__code__`, the
                     trunk-split resume frames, ...), shared by every instance and every
                     `torch.compile` wrapper of that code. The gate's graphs (train/grad and
                     train/no-grad at batch 64, and under TF32 the same pair again at 'highest')
                     therefore spend the SAME `cache_size_limit` slots production needs — a separate
                     compiled callable would NOT isolate them in torch 2.5.1. Nothing the gate
                     compiled is a production signature (production rolls out in EVAL mode at
                     n_envs), so the reset loses nothing.
  3. `prewarm(calls)` — run every production signature NOW, so caching happens when WE say, not
                     whenever a late caller first arrives.
  4. `lock(where)` — after the first real rollout + update: `error_on_recompile = True`; a compile
                     START callback counts every frame compile attempted from here on. Any late
                     recompile, first compile or cache-limit hit is a typed FATAL
                     (`CompileSentinelError` -> `os._exit(FATAL_CONFIG)`; the launcher does not
                     restart it).
  5. `stats()`     — graphs total, per-code cache entries, hits, post-lock compiles -> TB scalars.

THE TWO SILENT FAILURES, verified in torch 2.5.1 source:
  * CACHE-LIMIT FALLBACK — `convert_frame._compile` logs ONE `log.warning("torch._dynamo hit
    config.%s ...")` and raises `CacheLimitExceeded`/`unimplemented`, a SOFT failure
    `ConvertFrame.__call__` swallows: that frame runs EAGER from then on. A logging handler that
    raises is swallowed by `logging`, so `_CacheLimitHandler` records a sticky flag and the trainer
    CHECKS it at every rollout end and update end.
  * LATE RECOMPILES — each costs a stall and spends the cache toward the fallback.
    `error_on_recompile` raises `RecompileError` at the call site (raised raw, before `_compile`'s
    own try); the start callback is the backstop for what it cannot see — the FIRST compile of a
    never-seen code object, and (2.5.1 only) a `RecompileError` a caller's `except Exception`
    swallowed; `wrap_compiled` covers the swallow on every version.

TORCH VERSION GUARD. The internals used here are pinned per version in `_SUPPORTED`; an unknown
torch REFUSES (typed FATAL) rather than run a sentinel whose semantics may have moved. Each internal
is exercised on the installed torch by `compile_control_test.py`'s CONTRACT tests, which run on BOTH
supported versions. 2.5.1+cu121 locks with `error_on_recompile`; 2.8.0+cu126 (Lane K1,
`designs/endstate/program_rust_core.md`) locks with `torch.compiler.set_stance("fail_on_recompile")`
(the `"stance"` mode), which rejects ANY cache miss at the call site. On 2.8 the start callback runs
after the recompile check and so never sees a rejection; `wrap_compiled` (the learner forward is
installed through it) records rejections so a swallowed one is still fatal at the next check.
`fullgraph=True` is K6's, not done here.

What it deliberately does NOT do: change a number. `error_on_recompile` acts only on the recompile
path; the handler and the callback only count; `prewarm` runs under `torch.random.fork_rng` and
zeroes the gradients it made. `compile_control_test` pins identical outputs + gradients on vs off.
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
import statistics
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from agents.model.compile_trainer import CompileTrainerError

# EXACT torch version (local tag included) -> lock mechanism. "error_on_recompile" = config flag +
# log-warning detector + compile-start callback (2.5.1, contract-tested). "stance" =
# `torch.compiler.set_stance("fail_on_recompile")` + the same detector and callback (Lane K1,
# torch 2.8.0+cu126, contract-tested 2026-09-28 — `compile_control_test` runs on BOTH envs).
_SUPPORTED: Dict[str, str] = {"2.5.1+cu121": "error_on_recompile", "2.8.0+cu126": "stance"}

# What `stance` mode's rejection raises (torch 2.8 `eval_frame._callback_from_stance`): a plain
# RuntimeError, NOT a `RecompileError` — `find_recompile_error` matches it by this text.
_STANCE_REJECT_TEXT = "Detected recompile when torch.compile stance is 'fail_on_recompile'"

# THE SOURCE-HASH DRIFT TRIPWIRE (the `instrumented_ppo._verify_upstream_unchanged` pattern, for
# torch): SHA256 of `inspect.getsource(...)` of EVERY torch internal this adapter's semantics rest
# on, at the finest granularity `inspect` allows, per exact torch version. Verified at this module's
# import (so in every process that uses --compile-trainer) and by `compile_control_test`. A HASH
# says the code changed; the behavioural CONTRACT tests say the behaviour changed — both are kept.
#   cache_size.*                  — how entries are counted against the limit, and the verdict
#   guards.get_and_maybe_log_...  — the function that raises RecompileError on error_on_recompile
#   convert_frame._compile        — emits "torch._dynamo hit config.%s" (+ run_start_callbacks,
#                                   the recompile check, the CacheLimitExceeded/unimplemented path)
#   ConvertFrame.__call__         — the SOFT-failure swallow that makes the limit hit silent
#   ConvertFrameAssert.__call__   — `input_codes.add(code)`, what `cache_entries_by_code` walks
#   Tracker / _debug_get_cache_entry_list — the per-code entry read
#   CompilationCallbackHandler    — the compile-start callback registry
#   OutputGraph.compile_and_call_fx_graph — increments counters['stats']['unique_graphs']
#   torch._dynamo.reset / torch.compiler.reset — the post-gate reset
# TORCH 2.8 (Lane K1, re-read 2026-09-28) — what MOVED, and what the row hashes instead:
#   cache_size.exceeds_cache_size_limit -> exceeds_recompile_limit; config.cache_size_limit is now an
#     ALIAS of config.recompile_limit, and the warning names `config.recompile_limit (8)` /
#     `accumulated_recompile_limit`. The soft failure is `RecompileLimitExceeded` (was
#     CacheLimitExceeded), still swallowed by ConvertFrame.__call__ -> the frame runs EAGER.
#   guards.get_and_maybe_log_recompilation_reason -> ..._reasons (plural); still raises
#     RecompileError under error_on_recompile.
#   the compile-START callback now runs inside `compile_inner`, i.e. AFTER the recompile check and
#     the limit check (2.5.1: at the top of `_compile`) — so a REJECTED recompile no longer fires it —
#     and it now takes a `CallbackArgs` and also fires for LAZY_BACKWARD / TRITON_AUTOTUNING /
#     CUDAGRAPH_RECORDING triggers (all late compile work, all counted after the lock).
#   NEW: eval_frame._callback_from_stance (the "fail_on_recompile" branch) + eval_frame._set_stance
#     + decorators.set_stance — what `stance` mode rests on.
_SOURCE_HASHES: Dict[str, Dict[str, str]] = {
    "2.5.1+cu121": {
        "torch._dynamo.cache_size.compute_cache_size":
            "09c173a0698890766340687e01355f0b4b020770a644a21500e73e7886043482",
        "torch._dynamo.cache_size.is_recompilation":
            "b99d3c1f88fa3f80ec61ddcaa3c75d19d780b0a5245a3b9f07e00188799f8842",
        "torch._dynamo.cache_size.exceeds_cache_size_limit":
            "f7e81a56003515efd1b184a177eb071b03f88cf9d401dfd286a78da8bf08c532",
        "torch._dynamo.guards.get_and_maybe_log_recompilation_reason":
            "d72a0c856834814224f263e33be191d8a2b49450fa6f52d6d1378abb53f7c765",
        "torch._dynamo.convert_frame._compile":
            "2daa45669438e5e1653652d41d74458cea0f5dcb75dcf245de0693e582ba0645",
        "torch._dynamo.convert_frame.ConvertFrame.__call__":
            "48a63a2d50cb23f02c96ee6b4435ee2250b58c299d2ae040d961fc4954bc67b2",
        "torch._dynamo.convert_frame.ConvertFrameAssert.__call__":
            "8f372d0705bd7e49eb94cec4bb286f6392a47f74cdb1bc52c62fff1c946cafa7",
        "torch._dynamo.convert_frame.Tracker":
            "b2d821e2a67708098d684c2f17f732db0ba8a4901c063a70142fa972fb06ee29",
        "torch._dynamo.eval_frame._debug_get_cache_entry_list":
            "87fbc89bc7caa0f766c40336cbcf9f283205e2d04726e618e44bad268d3e5e0c",
        "torch._dynamo.callback.CompilationCallbackHandler":
            "98780974244d341eccf8dd23d656a6b1ae1c68e5475ac9c686992061e6471aa1",
        "torch._dynamo.output_graph.OutputGraph.compile_and_call_fx_graph":
            "e88a9e9b8571e12cea33c913712f6258d53cf70874d0dabfe4ccc8822ca9d7ba",
        "torch._dynamo.reset":
            "e4997fcbc979934efa007509d39b8913ff4dd0023162de0b0a8c1225d164a7f2",
        "torch.compiler.reset":
            "ae358f0c70af992ce82f8b4ea5e25e8800221a841c8078841335f98c34e13092",
    },
    "2.8.0+cu126": {
        "torch._dynamo.cache_size.compute_cache_size":
            "8c4f31bf7bfaebb8f69a761d58f96f8fdef06ee3cb860887b5458af2e41b2935",
        "torch._dynamo.cache_size.is_recompilation":
            "b99d3c1f88fa3f80ec61ddcaa3c75d19d780b0a5245a3b9f07e00188799f8842",
        "torch._dynamo.cache_size.exceeds_recompile_limit":
            "55a9866b420fb1bf58f4517d51b3a4e8cb593e85508310096503e49a19c637e0",
        "torch._dynamo.guards.get_and_maybe_log_recompilation_reasons":
            "8ce6be66f4e7e46d8d06053da39b8531ed2c1626dbb6ec1ca6c2ae7c97f535c6",
        "torch._dynamo.convert_frame._compile":
            "9c06510380dc7d32cbf44b144990107819602b60cb23e9b6c7070ebb88210529",
        "torch._dynamo.convert_frame.ConvertFrame.__call__":
            "6a8ed9a68161aa826c0ce2e5772edfad382eee2822257f6ad0dd6bed5ca24eef",
        "torch._dynamo.convert_frame.ConvertFrameAssert.__call__":
            "dd854a3ba6d2d84b0ed8f2bcf68ab35f776af92dfe97233a2c886ba495a3eeca",
        "torch._dynamo.convert_frame.Tracker":
            "5d102b1c017588532b961b4b999bb03bbcfdd1d20127b2095eefd1617c883ffd",
        "torch._dynamo.eval_frame._debug_get_cache_entry_list":
            "5ccaff8b8c1b7d40c4543ba8eef308c357bdfd835ac2e2180dc0cd22428d1691",
        "torch._dynamo.callback.CompilationCallbackHandler":
            "f6ed0187be8dc06c4957bb66584cbb537f258f4ee2c982cb3231c821c4aeb410",
        "torch._dynamo.output_graph.OutputGraph.compile_and_call_fx_graph":
            "d64581c0eba11b32e5c5a5c03433bafde2de6602d9f0e0b8c8550a00f94420d1",
        "torch._dynamo.reset":
            "0c3e45397def4debeb93673514a4db39e6e4cd6873dd24961c9593349eb9c2d1",
        "torch.compiler.reset":
            "ae358f0c70af992ce82f8b4ea5e25e8800221a841c8078841335f98c34e13092",
        "torch._dynamo.eval_frame._callback_from_stance":
            "ceac5c59c840964d6962b08228a9fa621367bd75c4b6507a1259c802f4867dcd",
        "torch._dynamo.eval_frame._set_stance":
            "b4abd0bdeab1342f21be96ae0bdf3b033e793398d2784877a4f100b17ba0ad14",
        "torch._dynamo.decorators.set_stance":
            "9ca4f2d3cc7aa6da5421e9723cb517ca44fcef8c010f6edbb9bc60b7d4ccb13b",
        "torch.compiler.set_stance":
            "88da974587d1867ee52e4fddca8f5696bef4e104a591dacdc2e891a9ed9b8963",
    },
}

# The logger `convert_frame` writes the cache-limit warning to (`logging.getLogger(__name__)`).
_CONVERT_FRAME_LOGGER = "torch._dynamo.convert_frame"
_LIMIT_RE = re.compile(r"torch\._dynamo hit config\.(\w+)")

# The behavioural backstop (independent of dynamo internals): steady-state train_ms above this
# multiple of its post-lock baseline for this many consecutive updates is a loud WARNING, never
# fatal — box contention (an eval burst, a peer's job) produces the same signature.
REGRESSION_RATIO = 1.4
REGRESSION_CONSECUTIVE = 3
BASELINE_UPDATES = 5

FATAL_TAG = "[CompileSentinel] FATAL"


class CompileSentinelError(CompileTrainerError):
    """A compiled learner recompiled after lock, hit dynamo's cache limit, or runs on a torch the
    sentinel does not support / whose internals drifted. Always fatal (`FATAL_CONFIG`)."""


# --------------------------------------------------------------------------- version guard
_RERECORD = ("re-read these functions in the installed torch, re-run "
             "`src/agents/model/compile_control_test.py` (the behavioural contract tests) on it, "
             "then re-record the row with `python -m agents.model.compile_control --record` "
             "(Lane K1 hits this by design when moving to torch >= 2.8: add the 'stance' mode).")


def lock_mode(v: Optional[str] = None) -> str:
    """The lock mechanism for this EXACT torch version, or raise `CompileSentinelError`."""
    ver = str(v or torch.__version__)
    mode = _SUPPORTED.get(ver)
    if mode is None:
        raise CompileSentinelError(
            f"--compile-trainer: the compile sentinel does not support torch {ver!r} (supported: "
            f"{sorted(_SUPPORTED)}). It reaches into torch._dynamo internals whose semantics can "
            f"move between versions, so it REFUSES rather than guard a run it may not see — "
            f"{_RERECORD}")
    return mode


def _resolve(qualname: str) -> Any:
    import importlib
    parts = qualname.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(qualname)


def source_hashes(qualnames: Sequence[str]) -> Dict[str, str]:
    """{qualname: SHA256 of inspect.getsource} on the INSTALLED torch ('<missing: ...>' if gone)."""
    import hashlib
    import inspect
    out: Dict[str, str] = {}
    for q in qualnames:
        try:
            out[q] = hashlib.sha256(inspect.getsource(_resolve(q)).encode("utf-8")).hexdigest()
        except Exception as exc:                  # a renamed / removed internal IS drift
            out[q] = f"<missing: {type(exc).__name__}: {exc}>"
    return out


def verify_torch_internals(version: Optional[str] = None) -> None:
    """The drift tripwire. Raises `CompileSentinelError` naming every drifted qualname with both
    hashes, or on a torch version with no recorded row."""
    ver = str(version or torch.__version__)
    expected = _SOURCE_HASHES.get(ver)
    if expected is None:
        raise CompileSentinelError(
            f"[CompileControl] DRIFT: no recorded torch-internals row for torch {ver!r} (recorded: "
            f"{sorted(_SOURCE_HASHES)}) — {_RERECORD}")
    actual = source_hashes(sorted(expected))
    drifted = [q for q in sorted(expected) if actual[q] != expected[q]]
    if drifted:
        lines = "\n".join(f"  {q}\n    expected {expected[q]}\n    actual   {actual[q]}"
                          for q in drifted)
        raise CompileSentinelError(
            f"[CompileControl] DRIFT DETECTED on torch {ver}: {len(drifted)} torch internal(s) the "
            f"compile sentinel depends on changed source:\n{lines}\nACTION REQUIRED: {_RERECORD}")


# --------------------------------------------------------------------------- thin dynamo shims
def cache_size_limit() -> int:
    """dynamo's per-code-object entry limit (`recompile_limit` on torch >= 2.8, where
    `cache_size_limit` is its alias; `cache_size_limit` on 2.5.1)."""
    cfg = torch._dynamo.config
    try:
        return int(getattr(cfg, "recompile_limit", None) or cfg.cache_size_limit)
    except Exception:
        return 8


def set_strict_errors() -> None:
    """A partial compile must be LOUD: never let dynamo swallow a compile error into eager."""
    torch._dynamo.config.suppress_errors = False


def dynamo_graphs_total() -> int:
    """Process-wide FX graphs dynamo compiled (`counters['stats']['unique_graphs']`; not reset)."""
    try:
        from torch._dynamo.utils import counters
        return int(counters["stats"]["unique_graphs"])
    except Exception:
        return -1


def cache_entries_by_code() -> Dict[str, int]:
    """{`name (file:line)`: resident cache entries} over every code object dynamo was handed.

    The per-code entry count is what `cache_size_limit` bounds (per ID-matched `self`; the learner
    has one extractor). `convert_frame.input_codes` + `eval_frame._debug_get_cache_entry_list`,
    both contract-tested; on API drift -> {} and callers say "unknown", never 0.
    """
    try:
        from torch._dynamo import convert_frame, eval_frame
        out: Dict[str, int] = {}
        for ref in list(convert_frame.input_codes.seen):
            code = ref()
            if code is None:
                continue
            n = len(eval_frame._debug_get_cache_entry_list(code))
            if n:
                key = f"{code.co_name} ({code.co_filename.rsplit('/', 1)[-1]}:{code.co_firstlineno})"
                out[key] = out.get(key, 0) + n
        return out
    except Exception:
        return {}


def _compiling_code_name() -> str:
    """Best-effort: the code object dynamo is about to compile, read off `_compile`'s frame."""
    try:
        f: Any = sys._getframe(1)
        while f is not None:
            if f.f_code.co_name == "_compile" and f.f_code.co_filename.endswith("convert_frame.py"):
                code = f.f_locals.get("code")
                if code is not None:
                    return f"{code.co_name} ({code.co_filename}:{code.co_firstlineno})"
            f = f.f_back
    except Exception:
        pass
    return "<unknown frame>"


def find_recompile_error(exc: BaseException) -> Optional[BaseException]:
    """The lock's rejection in `exc`'s cause/context chain, if any: a `RecompileError`
    (`error_on_recompile`) or the `fail_on_recompile` stance's RuntimeError (torch >= 2.8)."""
    try:
        from torch._dynamo.exc import RecompileError
    except Exception:
        return None
    cur: Optional[BaseException] = exc
    for _ in range(16):
        if cur is None:
            break
        if isinstance(cur, RecompileError) or (
                isinstance(cur, RuntimeError) and _STANCE_REJECT_TEXT in str(cur)):
            return cur
        cur = cur.__cause__ or cur.__context__
    return None


class _CacheLimitHandler(logging.Handler):
    """Recognises dynamo's cache-limit warning. Never raises (logging would swallow it anyway)."""

    def __init__(self, ctl: "CompileControl"):
        super().__init__(level=logging.WARNING)
        self._ctl = ctl

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        m = _LIMIT_RE.search(msg)
        if m is None and "hit config." not in str(record.msg):
            return
        self._ctl._on_limit_hit(m.group(1) if m else "unknown_limit", msg)


# --------------------------------------------------------------------------- the control
class CompileControl:
    """Process-wide phases over dynamo for a `--compile-trainer` run. See the module docstring."""

    def __init__(self, emit: Optional[Callable[[str], None]] = None, *,
                 torch_version: Optional[str] = None):
        self.mode = lock_mode(torch_version)
        self._emit = emit
        self.phase = "new"
        self.installed = False
        self.locked = False
        self.lock_where: Optional[str] = None
        self.graphs_at_lock: Optional[int] = None
        self.entries_at_lock: Dict[str, int] = {}
        self.entries_after_gate: Dict[str, int] = {}
        self.prewarmed: List[str] = []
        self.compile_starts = 0
        self.compiles_after_lock = 0
        self.rejected_after_lock = 0
        self.after_lock_frames: List[str] = []
        self.limit_hits: List[Tuple[str, str]] = []
        self.watch = TrainMsWatch()
        self._handler: Optional[_CacheLimitHandler] = None
        self._prev_level: Optional[int] = None
        self._prev_error_on_recompile: Optional[bool] = None
        self._stance_set = False

    # -- wiring --------------------------------------------------------------------------------
    def install(self) -> "CompileControl":
        if self.installed:
            return self
        import torch._dynamo  # noqa: F401  (registers the loggers)
        from torch._dynamo.callback import callback_handler
        lg = logging.getLogger(_CONVERT_FRAME_LOGGER)
        # A handler only sees records the logger lets through: never let a TORCH_LOGS setting that
        # raised the level above WARNING make the warning (and so the detector) vanish.
        if lg.getEffectiveLevel() > logging.WARNING:
            self._prev_level = lg.level
            lg.setLevel(logging.WARNING)
        self._handler = _CacheLimitHandler(self)
        lg.addHandler(self._handler)
        callback_handler.register_start_callback(self._on_compile_start)
        self._prev_error_on_recompile = bool(torch._dynamo.config.error_on_recompile)
        self.installed = True
        return self

    def uninstall(self) -> None:
        if not self.installed:
            return
        from torch._dynamo.callback import callback_handler
        lg = logging.getLogger(_CONVERT_FRAME_LOGGER)
        if self._handler is not None:
            lg.removeHandler(self._handler)
        if self._prev_level is not None:
            lg.setLevel(self._prev_level)
        with contextlib.suppress(ValueError):
            callback_handler.remove_start_callback(self._on_compile_start)
        if self._prev_error_on_recompile is not None:
            torch._dynamo.config.error_on_recompile = self._prev_error_on_recompile
        self._unset_stance()
        self.installed = False
        self.locked = False

    # -- dynamo hooks (never raise) ------------------------------------------------------------
    def _on_compile_start(self, args: Any = None) -> None:
        # torch 2.5.1 calls this with no argument; torch 2.8 passes a `CallbackArgs` whose
        # `callback_trigger` is DYNAMO / LAZY_BACKWARD / TRITON_AUTOTUNING / CUDAGRAPH_RECORDING.
        self.compile_starts += 1
        if self.locked:
            self.compiles_after_lock += 1
            if len(self.after_lock_frames) < 8:
                trig = getattr(getattr(args, "callback_trigger", None), "name", None)
                name = _compiling_code_name()
                self.after_lock_frames.append(f"{name} [{trig}]" if trig else name)

    def _on_rejected(self, exc: BaseException) -> None:
        """The lock REJECTED a compile at a watched call site. Sticky, so a caller's
        `except Exception` cannot hide it (on torch 2.8 the start callback never sees a rejected
        recompile: it runs after the recompile check)."""
        self.rejected_after_lock += 1
        if len(self.after_lock_frames) < 8:
            self.after_lock_frames.append(f"rejected: {str(exc).splitlines()[0][:200]}")

    def wrap_compiled(self, compiled: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a compiled callable so a lock rejection raised through it is RECORDED (sticky)
        before it propagates. `compile_trainer_extractor` installs the learner forward through
        this. Pure pass-through otherwise (no numerics, no extra dynamo frame)."""
        ctl = self

        def watched(*a: Any, **k: Any) -> Any:
            try:
                return compiled(*a, **k)
            except BaseException as exc:
                if ctl.locked and find_recompile_error(exc) is not None:
                    ctl._on_rejected(exc)
                raise
        watched._compile_control_inner = compiled  # type: ignore[attr-defined]
        return watched

    def _unset_stance(self) -> None:
        if getattr(self, "_stance_set", False):
            torch.compiler.set_stance("default")  # type: ignore[attr-defined]
            self._stance_set = False

    def _on_limit_hit(self, limit_type: str, msg: str) -> None:
        self.limit_hits.append((limit_type, msg))

    # -- phase 1: the gate ---------------------------------------------------------------------
    @contextlib.contextmanager
    def gate(self) -> Iterator["CompileControl"]:
        """The startup parity gate compiles freely inside this block (detector already on)."""
        self.install()
        set_strict_errors()
        self.phase = "gate"
        try:
            yield self
        finally:
            self.entries_after_gate = cache_entries_by_code()
        self.check("the startup parity gate")

    # -- phase 2: reset ------------------------------------------------------------------------
    def reset(self) -> str:
        """Drop every dynamo cache entry (the gate's, and any throwaway in-process compile)."""
        self.check("before the post-gate reset")
        torch._dynamo.reset()
        self.phase = "reset"
        before = sum(self.entries_after_gate.values()) if self.entries_after_gate else 0
        return (f"[CompileControl] reset after the gate: dropped {before} cache entries the gate "
                f"left on the production code objects (now {sum(cache_entries_by_code().values())})")

    # -- phase 3: prewarm ----------------------------------------------------------------------
    def prewarm(self, calls: Sequence[Tuple[str, Callable[[], None]]]) -> str:
        """Run each labelled production signature once, RNG-neutral. Returns the log line."""
        self.phase = "prewarm"
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        for label, fn in calls:
            with torch.random.fork_rng(devices=devices):
                fn()
            self.prewarmed.append(label)
            self.check(f"prewarm ({label})")
        ent = cache_entries_by_code()
        worst = max(ent.values()) if ent else -1
        return (f"[CompileControl] prewarmed {len(calls)} production signature(s) "
                f"[{', '.join(self.prewarmed)}]: {dynamo_graphs_total()} graphs in process, max "
                f"{worst} cache entries per code object (limit {cache_size_limit()})")

    # -- phase 4: lock -------------------------------------------------------------------------
    def lock(self, where: str) -> str:
        """Freeze the graph set: from here any dynamo compile in this process is a FATAL."""
        self.check(f"lock ({where})")
        self.graphs_at_lock = dynamo_graphs_total()
        self.entries_at_lock = cache_entries_by_code()
        set_strict_errors()                  # a swallowed RecompileError is still counted, not eaten
        if self.mode == "error_on_recompile":
            torch._dynamo.config.error_on_recompile = True
        else:                                # "stance" (torch >= 2.8, Lane K1): any cache MISS on a
            # compiled callable raises at the call site — a recompile AND a never-seen frame
            torch.compiler.set_stance("fail_on_recompile")  # type: ignore[attr-defined]
            self._stance_set = True
        self.locked = True
        self.lock_where = where
        self.phase = "locked"
        line = self.lock_line()
        self._say(line)
        return line

    def lock_line(self) -> str:
        ent = self.entries_at_lock
        limit = cache_size_limit()
        if ent:
            worst_code, worst = max(ent.items(), key=lambda kv: kv[1])
            per = (f"max {worst} cache entries per code object (limit {limit}, headroom "
                   f"{limit - worst}) at {worst_code}; {len(ent)} code objects hold "
                   f"{sum(ent.values())} entries")
        else:
            per = "per-code cache entries UNKNOWN (dynamo introspection unavailable)"
        return (f"🧊 [COMPILE LOCK] after {self.lock_where}: {per}; {self.graphs_at_lock} graphs "
                f"compiled in this process (incl. the gate's); mode {self.mode}. Any further "
                f"dynamo compile or a cache-limit hit now stops the run with a typed exit.")
        # NB: this HEALTHY line must carry none of scripts/ops/watch_run.sh's failure words
        # (FATAL / Traceback / OutOfMemory) — the stock watcher greps the child log for them and
        # exited on the old wording (2026-09-28, T32b). Pinned by compile_control_test.

    # -- the check -----------------------------------------------------------------------------
    def violation(self) -> Optional[str]:
        """None when healthy, else the FATAL's text."""
        if self.limit_hits:
            kind, msg = self.limit_hits[0]
            return (f"dynamo hit config.{kind} ({len(self.limit_hits)} hit(s)) during phase "
                    f"'{self.phase}' — that frame now runs EAGER, silently. First warning:\n{msg}")
        if self.compiles_after_lock or self.rejected_after_lock:
            return (f"{self.compiles_after_lock} dynamo compile(s) started and "
                    f"{self.rejected_after_lock} rejected AFTER the compile lock "
                    f"({self.lock_where}); graphs at lock {self.graphs_at_lock}, now "
                    f"{dynamo_graphs_total()}. Frames: {self.after_lock_frames}")
        return None

    def check(self, where: str) -> None:
        v = self.violation()
        if v is not None:
            raise CompileSentinelError(fatal_text(where, v))

    @contextlib.contextmanager
    def guard(self, where: str) -> Iterator[None]:
        """A `RecompileError` inside becomes `CompileSentinelError`; the sticky flags are checked on
        the way out."""
        try:
            yield
        except BaseException as exc:
            rec = find_recompile_error(exc)
            if rec is not None:
                raise CompileSentinelError(fatal_text(
                    where, f"dynamo RECOMPILED after the compile lock ({self.lock_where}):\n"
                           f"{rec}")) from exc
            raise
        self.check(where)

    # -- phase 5: stats ------------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        ent = cache_entries_by_code()
        out = {
            "compile/graphs_total": float(dynamo_graphs_total()),
            "compile/recompiles_after_lock": float(self.compiles_after_lock
                                                   + self.rejected_after_lock),
            "compile/cache_limit_hits": float(len(self.limit_hits)),
            "compile/max_cache_entries_per_code": float(max(ent.values())) if ent else -1.0,
            "compile/locked": 1.0 if self.locked else 0.0,
        }
        out.update(self.watch.scalars())
        return out

    # -- trainer wiring ------------------------------------------------------------------------
    def attach(self, model: Any) -> None:
        """Wrap `model.collect_rollouts` and `model.train` (INSTANCE attributes over the bound
        methods SB3's `learn()` calls) and `model.learn` (to RELEASE the lock when training ends —
        the final evaluation runs in-process on the compiled forward): guard both, lock after the
        first `train()`, record the
        `compile/*` scalars and the train_ms backstop at the update cadence, and turn any violation
        into `os._exit(FATAL_CONFIG)` — an exception inside `learn()` would be a restartable CRASH
        (`model_build`'s generic `except`), and a restart would replay the same failure."""
        orig_collect = model.collect_rollouts
        orig_train = model.train
        orig_learn = model.learn
        ctl = self

        def collect_rollouts(*a: Any, **k: Any) -> Any:
            try:
                with ctl.guard("rollout end"):
                    return orig_collect(*a, **k)
            except CompileSentinelError as exc:
                fatal_exit(str(exc))

        def train(*a: Any, **k: Any) -> Any:
            first = not ctl.locked
            try:
                with ctl.guard("update end"):
                    out = orig_train(*a, **k)
                if first:
                    ctl.lock("the first rollout + update")
            except CompileSentinelError as exc:
                fatal_exit(str(exc))
            # the lock update itself is not steady state (it may carry iteration 1's compiles), so
            # the train_ms baseline starts at the NEXT update
            ctl.record(model, observe_train_ms=not first)
            return out

        def learn(*a: Any, **k: Any) -> Any:
            # The lock's scope is TRAINING. After `learn()` the trainer runs its FINAL EVALUATION
            # in-process on the same compiled forward (batch 1, no-grad — new signatures by
            # design); measured 2026-09-28: the locked sentinel broke it with RecompileError. So
            # release on the way out, however `learn()` ends.
            try:
                return orig_learn(*a, **k)
            finally:
                ctl.release("learn() returned")

        model.collect_rollouts = collect_rollouts
        model.train = train
        model.learn = learn
        model._compile_control = self

    def release(self, why: str) -> None:
        """End the lock (training is over): restore `error_on_recompile`, stop counting."""
        if not self.locked:
            return
        self.locked = False
        self.phase = "released"
        prev = self._prev_error_on_recompile
        torch._dynamo.config.error_on_recompile = bool(prev) if prev is not None else False
        self._unset_stance()
        self._say(f"🧊 [COMPILE LOCK] released — {why}; {self.compiles_after_lock} compile(s) "
                  f"after the lock during training (must be 0); {self.rejected_after_lock} rejected "
                  f"(must be 0).")

    def record(self, model: Any, *, observe_train_ms: bool = True) -> None:
        """TB scalars + the train_ms backstop, after an update. Never raises."""
        try:
            logger = model.logger
            tm = getattr(logger, "name_to_value", {}).get("train/train_ms")
            if self.locked and observe_train_ms and tm is not None:
                line = self.watch.observe(float(tm))
                if line is not None:
                    self._say(line)
            for k, v in self.stats().items():
                logger.record(k, v)
        except Exception:
            pass

    def _say(self, msg: str) -> None:
        print(msg, flush=True)
        if self._emit is not None:
            with contextlib.suppress(Exception):
                self._emit(msg)


def fatal_text(where: str, detail: str) -> str:
    return (f"{FATAL_TAG} at {where}: {detail}\n"
            "--compile-trainer's contract is ONE fixed set of compiled graphs after the first "
            "iteration. A late recompile or a cache-limit hit means the learner is recompiling (a "
            "stall per event) or running EAGER (~1.75x slower), silently. Fatal by design; the "
            "launcher does NOT restart it (EXIT_FATAL_CONFIG). Diagnose with TORCH_LOGS=recompiles "
            "on a short fork; see designs/training/compile_flags.md ('The compile sentinel').")


def fatal_exit(msg: str) -> None:
    """Print the FATAL (stdout + stderr + launcher panel) and `os._exit(FATAL_CONFIG)`."""
    from main.exit_codes import TrainExitCode
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            print(f"\n{msg}", file=stream, flush=True)
    with contextlib.suppress(Exception):
        from main.launcher.ipc import send_event
        send_event(msg.splitlines()[0][:500])
    os._exit(int(TrainExitCode.FATAL_CONFIG))


# --------------------------------------------------------------------------- process singleton
_CONTROL: Optional[CompileControl] = None


def control(emit: Optional[Callable[[str], None]] = None) -> CompileControl:
    """The process's `CompileControl` (created on first use; raises on an unsupported torch)."""
    global _CONTROL
    if _CONTROL is None:
        _CONTROL = CompileControl(emit)
    elif emit is not None and _CONTROL._emit is None:
        _CONTROL._emit = emit
    return _CONTROL


def _reset_control_for_tests() -> None:
    global _CONTROL
    if _CONTROL is not None:
        _CONTROL.uninstall()
    _CONTROL = None


# --------------------------------------------------------------------------- behavioural backstop
class TrainMsWatch:
    """Independent of dynamo: steady-state `train_ms` vs its post-lock baseline. WARN only.

    The baseline is the median of the first `BASELINE_UPDATES` post-lock updates, then FROZEN (a
    running median would follow a regression down). `observe` returns the loud line when
    `train_ms > REGRESSION_RATIO x baseline` for `REGRESSION_CONSECUTIVE` updates in a row (once
    per episode), else None; `flag` is 1.0 while in that state.
    """

    def __init__(self, ratio: float = REGRESSION_RATIO, consecutive: int = REGRESSION_CONSECUTIVE,
                 baseline_n: int = BASELINE_UPDATES):
        self.ratio, self.consecutive, self.baseline_n = ratio, consecutive, baseline_n
        self._warm: List[float] = []
        self.baseline: Optional[float] = None
        self.streak = 0
        self.flag = 0.0
        self.last_ratio: Optional[float] = None

    def observe(self, train_ms: float) -> Optional[str]:
        if not (train_ms > 0):
            return None
        if self.baseline is None:
            self._warm.append(float(train_ms))
            if len(self._warm) >= self.baseline_n:
                self.baseline = float(statistics.median(self._warm))
            return None
        self.last_ratio = train_ms / self.baseline
        if self.last_ratio > self.ratio:
            self.streak += 1
        else:
            self.streak = 0
            self.flag = 0.0
        if self.streak == self.consecutive:
            self.flag = 1.0
            return (f"⚠️ [COMPILE REGRESSION?] train_ms {train_ms:.0f} is {self.last_ratio:.2f}x "
                    f"the post-lock baseline {self.baseline:.0f} for {self.streak} consecutive "
                    f"updates (bar {self.ratio}x). The sentinel saw NO recompile, so either the box "
                    f"is contended (check `uptime`, a peer job, an eval burst) or dynamo is doing "
                    f"something the sentinel cannot see — investigate before trusting throughput.")
        return None

    def scalars(self) -> Dict[str, float]:
        out = {"compile/regression_flag": self.flag}
        if self.last_ratio is not None:
            out["compile/train_ms_vs_lock_baseline"] = float(self.last_ratio)
        return out


# --------------------------------------------------------------------------- import-time tripwire
# Every process that imports this adapter is a --compile-trainer process (it is imported lazily by
# `compile_trainer` only when the learner compiles), so verifying HERE covers every such process.
# `CompileSentinelError` is a `CompileTrainerError`, so the startup path turns it into FATAL_CONFIG.
if __name__ != "__main__":
    verify_torch_internals()
else:  # pragma: no cover — the re-record helper: `python -m agents.model.compile_control --record`
    import json as _json
    # the installed torch's own row if it has one, else the NEWEST row's qualnames (then edit the
    # renamed ones by hand — a '<missing: ...>' value names each)
    _rows = sorted(_SOURCE_HASHES.get(torch.__version__) or list(_SOURCE_HASHES.values())[-1])
    if "--record" in sys.argv:
        print(_json.dumps({torch.__version__: source_hashes(_rows)}, indent=4))
