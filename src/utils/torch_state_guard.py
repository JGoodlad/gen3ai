"""TORCH GLOBAL-STATE GUARD — a test that changes process-global numeric state and leaves it
changed FAILS, naming the global, its before/after values and the test (`gen3_torch_state_guard_v1`).

WHY (Lane E, `c256dd95`, F-LJ-6). A harness called `torch.set_num_threads(4)` and never restored it.
The next test in the same process built fresh policy weights at a different intra-op thread count:
94 of 721 tensors moved by up to 7.5e-6, an EXACT-tie argmax flipped, and a compiled-vs-eager parity
test failed only when it ran AFTER other tests. Nothing about that failure pointed at its cause. A
process-global torch setting is an input to every later computation in the process, so a leak turns
test ORDER into an input — the pass/fail of test B becomes a function of which tests happened to run
before it in the same xdist worker. That is GIGO in its worst form: silent, order-dependent, and
green in isolation.

THE CONTRACT.
  * `DECLARED` below is the ONE list of globals the guard watches. Add a row here and nowhere else.
  * The root `conftest.py` snapshots it around every test (an autouse fixture: the test body and its
    function-scoped fixtures), at every MODULE boundary (module/class-scoped fixtures, which set up
    before any function fixture can look), and around every test module's IMPORT (collection-time
    leaks, which happen before any test runs).
  * A difference FAILS. The guard NEVER restores the value for you: restoring would hide the leak,
    and the leak is the bug. The fix is at the source — a `try/finally` or a context manager
    (`torch_globals(...)` below, or `torch.random.fork_rng`, or `config.patch(...)`).
  * There is NO allowlist, by the same principle as the file-size gate's empty one. A test that
    must change a global restores it. `GEN3AI_SKIP_TORCH_STATE_GUARD=1` turns the whole guard off.

WHAT IS DELIBERATELY NOT HERE.
  * The global RNGs (`torch`, `numpy`, `random`). Every test CONSUMES them, so their state changes by
    design; "a later test relies on an earlier test's seed" is a property of the READER (it must
    seed its own generator), not something a before/after diff can see.
  * Things with no getter (`torch.set_flush_denormal`, `torch.set_printoptions`).
  * In-place mutation of a container NESTED inside a dynamo/inductor config value (one level of
    list/dict/set is copied; deeper is not).

COST. Reads only modules already in `sys.modules` — it never imports a subsystem to inspect it
(`prime()` imports `torch` once per process so the core rows always have a "before"). A module that
first appears DURING a test is compared against its declared DEFAULT. Measured per snapshot in the
`__main__` block below; the report is in `designs/ops/testing.md`.
"""
from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

SKIP_ENV = "GEN3AI_SKIP_TORCH_STATE_GUARD"

#: A value that could not be read (the reader raised). Two unreadable reads compare equal.
_UNREADABLE = "<unreadable>"


@dataclass(frozen=True)
class Global:
    """One watched global: its human name, the module that must already be imported for it to be
    read, how to read it from that module, and (optionally) the value a FRESH import of that module
    has — used when the module first appears during the window being checked."""
    name: str
    module: str
    read: Callable[[Any], Any]
    default: Optional[Callable[[Any], Any]] = None


def _attr(path: str) -> Callable[[Any], Any]:
    """A reader for a dotted attribute path below the module (``"backends.cudnn.benchmark"``)."""
    parts = path.split(".")

    def read(mod: Any) -> Any:
        obj = mod
        for p in parts:
            obj = getattr(obj, p)
        return obj() if callable(obj) and not isinstance(obj, type) else obj
    return read


def _call(name: str) -> Callable[[Any], Any]:
    def read(mod: Any) -> Any:
        return getattr(mod, name)()
    return read


def _copy1(v: Any) -> Any:
    """One level of container copy, so an in-place `config.x.append(...)` is still a difference."""
    if isinstance(v, (list, dict, set)):
        return type(v)(v) if not isinstance(v, dict) else dict(v)
    return v


def _config_values(cfg: Any) -> Dict[str, Any]:
    """Every key of a torch `ConfigModule`, resolved WITHOUT `getattr` (which, on 2.8, writes a
    deepcopy of a container default into the entry as a side effect). 2.5 keeps plain values in
    `_config`; 2.8 keeps `_ConfigEntry` objects whose effective value is resolved below."""
    out: Dict[str, Any] = {}
    unset = _unset_sentinel()
    for k, e in cfg._config.items():
        if hasattr(e, "user_override"):
            v = e.env_value_force
            if v is unset:
                v = e.user_override
            if v is unset:
                v = e.env_value_default
            if v is unset:
                v = e.default
        else:
            v = e
        out[k] = _copy1(v)
    return out


def _config_defaults(cfg: Any) -> Dict[str, Any]:
    """What a fresh import of the config module holds (measured equal to `_config_values` right
    after import, on both torch 2.5.1 and 2.8.0 — see `torch_state_guard_test`)."""
    out: Dict[str, Any] = {}
    default = getattr(cfg, "_default", None)
    unset = _unset_sentinel()
    for k, e in cfg._config.items():
        if hasattr(e, "user_override"):
            v = e.env_value_force
            if v is unset:
                v = e.env_value_default
            if v is unset:
                v = e.default
        else:
            v = default[k] if default is not None and k in default else e
        out[k] = _copy1(v)
    return out


_NO_SENTINEL = object()


def _unset_sentinel() -> Any:
    mod = sys.modules.get("torch.utils._config_module")
    return getattr(mod, "_UNSET_SENTINEL", _NO_SENTINEL)


def _stance(mod: Any) -> Any:
    return getattr(mod, "_stance", None)


def _stance_default(mod: Any) -> Any:
    st = getattr(mod, "_stance", None)
    return None if st is None else type(st)()


def _callbacks(mod: Any) -> Tuple[int, int]:
    h = mod.callback_handler
    return len(h.start_callbacks), len(h.end_callbacks)


def _compile_control(mod: Any) -> Tuple[bool, bool]:
    c = getattr(mod, "_CONTROL", None)
    return (bool(c is not None and c.installed), bool(c is not None and c.locked))


# ------------------------------------------------------------------------------ THE DECLARED LIST
# One row per process-global the guard watches. The torch core rows need `torch` (imported once by
# `prime()`); every other row is read only if its module is already imported.
DECLARED: Tuple[Global, ...] = (
    Global("torch.get_num_threads()", "torch", _call("get_num_threads")),
    Global("torch.get_num_interop_threads()", "torch", _call("get_num_interop_threads")),
    Global("torch.get_float32_matmul_precision()", "torch", _call("get_float32_matmul_precision")),
    Global("torch.get_default_dtype()", "torch", _call("get_default_dtype")),
    Global("torch.get_default_device()", "torch", _call("get_default_device")),
    Global("torch.are_deterministic_algorithms_enabled()", "torch",
           _call("are_deterministic_algorithms_enabled")),
    Global("torch.is_deterministic_algorithms_warn_only_enabled()", "torch",
           _call("is_deterministic_algorithms_warn_only_enabled")),
    Global("torch.is_grad_enabled()", "torch", _call("is_grad_enabled")),
    Global("torch.is_inference_mode_enabled()", "torch", _call("is_inference_mode_enabled")),
    Global("torch.is_anomaly_enabled()", "torch", _call("is_anomaly_enabled")),
    Global("torch.is_anomaly_check_nan_enabled()", "torch", _call("is_anomaly_check_nan_enabled")),
    Global("torch.is_warn_always_enabled()", "torch", _call("is_warn_always_enabled")),
    Global("torch.backends.cuda.matmul.allow_tf32", "torch", _attr("backends.cuda.matmul.allow_tf32")),
    Global("torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction", "torch",
           _attr("backends.cuda.matmul.allow_fp16_reduced_precision_reduction")),
    Global("torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction", "torch",
           _attr("backends.cuda.matmul.allow_bf16_reduced_precision_reduction")),
    Global("torch.backends.cudnn.allow_tf32", "torch", _attr("backends.cudnn.allow_tf32")),
    Global("torch.backends.cudnn.benchmark", "torch", _attr("backends.cudnn.benchmark")),
    Global("torch.backends.cudnn.deterministic", "torch", _attr("backends.cudnn.deterministic")),
    Global("torch.backends.cudnn.enabled", "torch", _attr("backends.cudnn.enabled")),
    Global("torch.backends.mkldnn.enabled", "torch", _attr("backends.mkldnn.enabled")),
    # --- compile state: dynamo / inductor / functorch config, the compiler stance, the callbacks
    Global("torch._dynamo.config", "torch._dynamo.config", _config_values, _config_defaults),
    Global("torch._inductor.config", "torch._inductor.config", _config_values, _config_defaults),
    Global("torch._functorch.config", "torch._functorch.config", _config_values, _config_defaults),
    Global("torch.compiler.config", "torch.compiler.config", _config_values, _config_defaults),
    Global("torch._dynamo.eval_frame._stance (torch.compiler.set_stance)", "torch._dynamo.eval_frame",
           _stance, _stance_default),
    Global("torch._dynamo.callback (start, end) callback counts", "torch._dynamo.callback",
           _callbacks, lambda mod: (0, 0)),
    Global("agents.model.compile_control._CONTROL (installed, locked)", "agents.model.compile_control",
           _compile_control, lambda mod: (False, False)),
    # --- numpy's floating-point error mode (np.seterr)
    Global("numpy.geterr()", "numpy", _call("geterr"),
           lambda mod: {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}),
)

Snapshot = Dict[str, Any]


def enabled() -> bool:
    return not os.environ.get(SKIP_ENV)


def prime() -> None:
    """Import `torch` once per process, so its core rows always have a BEFORE. Without it, a test
    that is the first to import torch AND leaks a thread count would compare against nothing."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pass


def _read(g: Global, mod: Any) -> Any:
    try:
        return g.read(mod)
    except Exception:
        return _UNREADABLE


def snapshot() -> Snapshot:
    """The current value of every DECLARED global whose module is imported. Never imports."""
    mods = sys.modules
    out: Snapshot = {}
    for g in DECLARED:
        mod = mods.get(g.module)
        if mod is not None:
            out[g.name] = _read(g, mod)
    return out


def diff(before: Snapshot, after: Snapshot) -> List[Tuple[str, Any, Any]]:
    """``[(name, before, after), ...]`` for every global that differs. A config module is diffed
    key by key (only the moved keys are named). A row absent BEFORE (its module was imported inside
    the window) is compared against its declared default; with no default it is not judged."""
    out: List[Tuple[str, Any, Any]] = []
    by_name = {g.name: g for g in DECLARED}
    for name, a in after.items():
        if name in before:
            b = before[name]
        else:
            g = by_name[name]
            mod = sys.modules.get(g.module)
            if g.default is None or mod is None:
                continue
            try:
                b = g.default(mod)
            except Exception:
                continue
            name = f"{name} [imported during this window; compared to its import-time default]"
        if isinstance(a, dict) and isinstance(b, dict) and name.split(" ")[0].endswith("config"):
            for k in sorted(set(a) | set(b)):
                if not _same(b.get(k, _UNREADABLE), a.get(k, _UNREADABLE)):
                    out.append((f"{name.split(' ')[0]}.{k}", b.get(k), a.get(k)))
        elif not _same(b, a):
            out.append((name, b, a))
    return out


def _same(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:
        return repr(a) == repr(b)


FIX_HINT = (
    "A test (or a fixture, or an import) that changes process-global torch/numeric state must "
    "RESTORE it — `try/finally`, `utils.torch_state_guard.torch_globals(num_threads=..., "
    "float32_matmul_precision=...)`, `torch._dynamo.config.patch(...)`, or a yield fixture that "
    "restores at its own scope's end. The guard did NOT restore it for you: every later test in "
    "this process now runs under the leaked value. Declared list: src/utils/torch_state_guard.py; "
    f"`{SKIP_ENV}=1` turns the guard off (it has no allowlist).")


def describe(where: str, leaks: List[Tuple[str, Any, Any]]) -> str:
    lines = [f"TORCH GLOBAL-STATE LEAK — {where} left {len(leaks)} process-global(s) changed:"]
    for name, b, a in leaks:
        lines.append(f"    {name}: {b!r} -> {a!r}")
    lines.append(FIX_HINT)
    return "\n".join(lines)


@contextlib.contextmanager
def torch_globals(*, num_threads: Optional[int] = None,
                  float32_matmul_precision: Optional[str] = None,
                  allow_tf32: Optional[bool] = None) -> Iterator[None]:
    """Set the named torch globals for the block and RESTORE all four (thread count, fp32 matmul
    precision, cuda-matmul and cudnn TF32) to what was found, even on an exception. With no
    arguments it only restores — the shape for code that sets them itself (a CLI `main()` run
    in-process). The shape every test that pins a thread count or precision should use."""
    import torch
    prev_threads = torch.get_num_threads()
    prev_prec = torch.get_float32_matmul_precision()
    prev_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        if num_threads is not None:
            torch.set_num_threads(int(num_threads))
        if float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(float32_matmul_precision)
        if allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev_tf32
        torch.set_float32_matmul_precision(prev_prec)
        torch.set_num_threads(prev_threads)


if __name__ == "__main__":             # the per-snapshot cost the testing doc quotes
    import timeit
    prime()
    import torch._dynamo  # noqa: F401  (the worst case: every config module imported)
    import torch._inductor.config  # noqa: F401
    import torch._functorch.config  # noqa: F401
    n = 2000
    s = timeit.timeit(snapshot, number=n) / n
    b = snapshot()
    d = timeit.timeit(lambda: diff(b, snapshot()), number=n) / n
    print(f"snapshot {s * 1e6:.1f} us; snapshot+diff {d * 1e6:.1f} us; rows {len(b)}")
