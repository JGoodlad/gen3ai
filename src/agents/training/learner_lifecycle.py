"""K6 — the learner's DECLARED LIFECYCLE: the FREEZE GUARD (`gen3_learner_freeze_v1`, M5 Lane K).

THE PRINCIPLE (owner, 2026-09-28; `designs/endstate/program_rust_core.md`, "M5 DESIGN PRINCIPLE — a
DECLARED LIFECYCLE"): a training process is a long-lived server. STARTUP declares and acquires every
resource the steady state will use, then FREEZES; the STEADY STATE acquires nothing. Until this
module the principle was unenforced in the learner: the K8 inventory found the ride-along heads'
Adam built lazily on the first update — by luck, not by any check.

WHAT IS FROZEN, and when. `LearnerFreeze.freeze()` runs at the END of startup — on entry to the first
`collect_rollouts` (after the trainer compile, its parity gate, the prewarm and `learn()`'s own setup,
i.e. after every `_on_training_start`), before any real rollout or update. It snapshots the LEARNER'S
OBJECT GRAPH:

  * every `nn.Module` and every `nn.Parameter` of the policy (extractor, heads, aux heads, the
    ride-along heads) and of any other module hanging off the model, by OBJECT IDENTITY (a parameter
    REPLACED under the same name is a new object the optimizer does not hold — the dead-param class
    `policy._build` documents);
  * every buffer NAME (a buffer re-assigned in place under its own name is not an acquisition);
  * every `torch.optim.Optimizer` reachable from the model, its param groups (which parameter objects
    each holds) and which parameters already carry optimizer STATE (`declare_optimizer_state` creates
    it at startup, so a state entry born later is a lazy acquisition too).

After every update and every rollout `check()` compares; anything NEW is a typed, single-shot FATAL —
`LazyAcquisitionError` (a `FatalConfigError`: exit `FATAL_CONFIG`, the launcher does not restart),
naming the object (its attribute path, class, shape) and its CONSTRUCTION SITE. These are the
DETERMINISTIC cases (orchestrator, 2026-09-30): a healthy run never trips them, so there is no
tolerance and no warn tier. (CUDA memory is NOT judged here: fragmentation makes it stochastic, so
it is a separate leak detector that warns by default — `cuda_memory_trend`.)

HOW A SITE IS KNOWN. While frozen, three recorders run (each only RECORDS — never raises — so an
object built for something other than the learner, e.g. an in-process opponent under `--debug`, is
recorded and ignored unless it joins the learner's graph):
  * torch's GLOBAL module / parameter / buffer REGISTRATION hooks (`torch.nn.modules.module.
    register_module_*_registration_hook`) record the stack of every registration;
  * `torch.optim.Optimizer.__init__` is wrapped to record the stack of every optimizer built;
  * a GLOBAL optimizer STEP pre-hook (`register_optimizer_step_pre_hook`) RAISES at the point an
    optimizer outside the frozen set first steps — before it can move a weight — and also records
    the violation STICKY, so a caller's `except Exception` cannot hide it from the next `check()`.
A construction with no record (made before the freeze, or outside any registration) says so.

WHAT IT DOES NOT DO: change a number. The snapshot and the checks read identities; the hooks record.
`declare_optimizer_state` changes the optimizer's state only from "absent" to exactly what torch's
lazy first step creates (`declare_optimizer_state_test` pins a pre-declared update bit-identical to a
lazy one, parameters and moments).
"""
from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import torch

from main.exit_codes import FatalConfigError

FATAL_TAG = "[LearnerLifecycle] FATAL"
#: The memory half's tag: a STOP the launcher restarts (capped), not a FATAL configuration error.
STOP_TAG = "[LearnerLifecycle] STOP"

#: Frames kept per recorded construction site (innermost first after the filter below).
_SITE_FRAMES = 6
#: Path fragments of frames that are never the interesting site (torch internals, this module).
_SKIP_FRAME_FRAGMENTS = ("/torch/", "/site-packages/", "learner_lifecycle.py", "<frozen ")


class LazyAcquisitionError(FatalConfigError):
    """A learner resource (optimizer / parameter / module / buffer / optimizer state) was acquired
    AFTER the freeze. Deterministic, so single-shot: exit `FATAL_CONFIG` (3), not restarted."""


class CudaMemoryLeakError(RuntimeError):
    """The memory half (`gen3_cuda_memory_trend_v1`): a SUSTAINED growth of the learner process's
    live CUDA memory projects an out-of-memory inside the declared horizon (`cuda_memory_trend`'s
    STOP). Raised at an update's end, so the trainer's exception handler saves
    `final_model_exception.zip` (the checkpoint) before the process exits `FATAL_CUDA_LEAK` (6) —
    NOT a configuration error: a fresh process clears a leak, so the launcher RESTARTS from that
    checkpoint, at most `exit_codes.CUDA_LEAK_RESTART_CAP` times per session, then stops for good."""


# ------------------------------------------------------------------------------------------- sites
def _site(skip: int = 2) -> str:
    """A compact construction site: the innermost repo frames of the current stack, innermost
    first. Walks the frames directly (no source-line lookup), so a recorder firing on every module
    registration of an in-process model load stays cheap."""
    try:
        f: Any = sys._getframe(skip)
    except Exception:                                      # pragma: no cover — introspection only
        return "<site unavailable>"
    frames: List[Tuple[str, int, str]] = []
    while f is not None and len(frames) < 60:
        frames.append((f.f_code.co_filename, f.f_lineno, f.f_code.co_name))
        f = f.f_back
    keep = [fr for fr in frames if not any(s in fr[0] for s in _SKIP_FRAME_FRAGMENTS)]
    keep = keep[:_SITE_FRAMES] or frames[:_SITE_FRAMES]
    return " <- ".join(f"{fn.split('/src/', 1)[-1]}:{ln} {name}" for fn, ln, name in keep)


class _SiteRecorder:
    """Records the construction / registration site of objects created while installed."""

    def __init__(self) -> None:
        from torch.utils.weak import WeakIdKeyDictionary
        self.sites: Any = WeakIdKeyDictionary()
        self._handles: List[Any] = []
        self._orig_opt_init: Optional[Callable[..., None]] = None
        self.installed = False

    def site_of(self, obj: Any) -> Optional[str]:
        try:
            got = self.sites.get(obj)
            return None if got is None else str(got)
        except Exception:                                  # pragma: no cover
            return None

    def _record(self, obj: Any, what: str) -> None:
        try:
            if obj is not None and obj not in self.sites:
                self.sites[obj] = f"{what} at {_site(3)}"
        except Exception:                                  # never break a construction
            pass

    def install(self) -> None:
        if self.installed:
            return
        import torch.nn.modules.module as _mm
        rec = self

        def on_param(module: Any, name: str, param: Any) -> None:
            rec._record(param, f"registered as parameter {name!r} of {type(module).__name__}")
            rec._record(module, f"{type(module).__name__} (gained parameter {name!r})")

        def on_module(module: Any, name: str, sub: Any) -> None:
            rec._record(sub, f"registered as submodule {name!r} of {type(module).__name__}")

        def on_buffer(module: Any, name: str, buf: Any) -> None:
            rec._record(module, f"{type(module).__name__} (gained buffer {name!r})")

        self._handles = [_mm.register_module_parameter_registration_hook(on_param),
                         _mm.register_module_module_registration_hook(on_module),
                         _mm.register_module_buffer_registration_hook(on_buffer)]
        orig = torch.optim.Optimizer.__init__
        self._orig_opt_init = orig

        def opt_init(opt_self: Any, *a: Any, **k: Any) -> None:
            rec._record(opt_self, f"{type(opt_self).__name__} constructed")
            orig(opt_self, *a, **k)

        opt_init._gen3_lifecycle_wrapper = True             # type: ignore[attr-defined]
        torch.optim.Optimizer.__init__ = opt_init           # type: ignore[method-assign,assignment]
        self.installed = True

    def uninstall(self) -> None:
        if not self.installed:
            return
        for h in self._handles:
            with contextlib.suppress(Exception):
                h.remove()
        self._handles = []
        if self._orig_opt_init is not None:
            torch.optim.Optimizer.__init__ = self._orig_opt_init  # type: ignore[method-assign]
            self._orig_opt_init = None
        self.installed = False


# ------------------------------------------------------------------------------------------- graph
_CONTAINERS = (list, tuple, set, frozenset)


def _children(obj: Any) -> Iterator[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            yield f"[{k!r}]", v
    elif isinstance(obj, _CONTAINERS):
        for i, v in enumerate(list(obj)):
            yield f"[{i}]", v


def learner_objects(model: Any) -> Tuple[Dict[str, torch.nn.Module], Dict[str, torch.optim.Optimizer]]:
    """The learner's ROOT modules and optimizers, by attribute path.

    Roots: `model.policy` (the whole module tree: extractor, heads, aux and ride-along heads), plus
    every `nn.Module` / `Optimizer` held directly by the model, the policy or a policy submodule's
    instance dict — directly or one container deep (a list / tuple / dict of optimizers, e.g. the
    ride-along variants'), and one attribute deep into a repo-owned helper object the model holds
    (e.g. a telemetry object holding its own head and optimizer)."""
    modules: Dict[str, torch.nn.Module] = {}
    opts: Dict[str, torch.optim.Optimizer] = {}
    seen: Set[int] = set()
    policy = getattr(model, "policy", None)
    if isinstance(policy, torch.nn.Module):
        modules["policy"] = policy
        seen.add(id(policy))

    def visit(path: str, obj: Any, depth: int) -> None:
        if obj is None or id(obj) in seen:
            return
        if isinstance(obj, torch.optim.Optimizer):
            seen.add(id(obj))
            opts[path] = obj
            return
        if isinstance(obj, torch.nn.Module):
            seen.add(id(obj))
            if not any(obj is m or _is_submodule(obj, m) for m in modules.values()):
                modules[path] = obj
            return
        if depth <= 0:
            return
        if isinstance(obj, (dict,) + _CONTAINERS):
            seen.add(id(obj))
            for k, v in _children(obj):
                visit(path + k, v, depth - 1)
            return
        mod = type(obj).__module__ or ""
        if mod.startswith(("agents.", "main.")) and hasattr(obj, "__dict__"):
            seen.add(id(obj))
            for k, v in list(vars(obj).items()):
                visit(f"{path}.{k}", v, depth - 1)

    holders: List[Tuple[str, Any]] = [("model", model)]
    if isinstance(policy, torch.nn.Module):
        holders.append(("policy", policy))
        for name, sub in policy.named_modules():
            if sub is not policy:
                holders.append((f"policy.{name}", sub))
    for hpath, holder in holders:
        try:
            items = list(vars(holder).items())
        except TypeError:
            continue
        for k, v in items:
            if k in ("_modules", "_parameters", "_buffers", "env", "logger", "rollout_buffer",
                     "_last_obs", "_last_episode_starts", "observation_space", "action_space"):
                continue
            visit(f"{hpath}.{k}", v, 2)
    return modules, opts


def _is_submodule(obj: torch.nn.Module, root: torch.nn.Module) -> bool:
    return any(obj is m for m in root.modules())


@dataclass
class FrozenSet:
    """The identities frozen at `freeze()`."""
    modules: Dict[int, str] = field(default_factory=dict)        # id -> qualified path
    params: Dict[int, str] = field(default_factory=dict)         # id -> qualified name
    param_ids_by_name: Dict[str, int] = field(default_factory=dict)
    buffers: Set[str] = field(default_factory=set)               # qualified buffer names
    optimizers: Dict[int, str] = field(default_factory=dict)     # id -> attribute path
    groups: Dict[int, List[Tuple[int, ...]]] = field(default_factory=dict)  # opt id -> per-group ids
    state: Dict[int, Set[int]] = field(default_factory=dict)     # opt id -> param ids with state


def _walk(model: Any) -> Tuple[FrozenSet, Dict[int, Any]]:
    """The current graph as a `FrozenSet`, plus id -> object for naming."""
    mods, opts = learner_objects(model)
    fs = FrozenSet()
    objs: Dict[int, Any] = {}
    for root_path, root in mods.items():
        for name, m in root.named_modules():
            q = root_path if not name else f"{root_path}.{name}"
            fs.modules.setdefault(id(m), q)
            objs[id(m)] = m
        for name, p in root.named_parameters(remove_duplicate=False):
            q = f"{root_path}.{name}"
            fs.params.setdefault(id(p), q)
            fs.param_ids_by_name[q] = id(p)
            objs[id(p)] = p
        for name, _b in root.named_buffers(remove_duplicate=False):
            fs.buffers.add(f"{root_path}.{name}")
    for path, opt in opts.items():
        fs.optimizers[id(opt)] = path
        objs[id(opt)] = opt
        fs.groups[id(opt)] = [tuple(id(p) for p in g["params"]) for g in opt.param_groups]
        fs.state[id(opt)] = {id(p) for p, st in opt.state.items() if st}
        for g in opt.param_groups:
            for p in g["params"]:
                objs.setdefault(id(p), p)
    return fs, objs


def _describe(obj: Any) -> str:
    if isinstance(obj, torch.nn.Parameter) or torch.is_tensor(obj):
        return f"{type(obj).__name__}{tuple(obj.shape)} {str(obj.dtype).replace('torch.', '')}"
    if isinstance(obj, torch.optim.Optimizer):
        n = sum(len(g["params"]) for g in obj.param_groups)
        return f"{type(obj).__name__} over {n} parameter tensor(s)"
    return type(obj).__name__


# ------------------------------------------------------------------------------- optimizer state
_DECLARABLE = ("Adam", "AdamW")


def declare_optimizer_state(opt: torch.optim.Optimizer) -> int:
    """Create the optimizer STATE of every trainable parameter that has none, EXACTLY as torch's
    lazy first-step init would — so the steady state never allocates it. Returns how many entries
    were created.

    Adam / AdamW only (their lazy init is all-zero moments and step 0; SGD's momentum buffer, for
    one, is initialised from the first gradient, so zeros would be wrong — refused, typed). The
    mechanism is torch's own init: one `step()` over ZERO gradients for exactly the stateless
    parameters (every other parameter's `.grad` is None for the call, so torch skips it), after which
    the parameters are copied back BIT-EXACTLY (AdamW's decoupled weight decay moves a weight even on
    a zero gradient) and every state tensor is zeroed and the step counter reset to 0 — i.e. the
    state the first real step would have created at its entry. Must run with no graph alive."""
    if type(opt).__name__ not in _DECLARABLE:
        raise LazyAcquisitionError(
            f"{FATAL_TAG}: cannot declare the state of a {type(opt).__name__} at startup — only "
            f"{_DECLARABLE} have an all-zero lazy init this helper reproduces exactly.")
    fresh = [p for g in opt.param_groups for p in g["params"]
             if p.requires_grad and not opt.state.get(p)]
    if not fresh:
        return 0
    fresh_ids = {id(p) for p in fresh}
    everyone = [p for g in opt.param_groups for p in g["params"]]
    saved_grads = {id(p): p.grad for p in everyone}
    saved_vals = [p.detach().clone() for p in fresh]
    try:
        with torch.no_grad():
            for p in everyone:
                p.grad = torch.zeros_like(p) if id(p) in fresh_ids else None
            opt.step()
            for p, v in zip(fresh, saved_vals):
                p.copy_(v)
            for p in fresh:
                st = opt.state.get(p) or {}
                for k, v in list(st.items()):
                    if k == "step":
                        if torch.is_tensor(v):
                            v.zero_()
                        else:
                            st[k] = 0
                    elif torch.is_tensor(v):
                        v.zero_()
    finally:
        for p in everyone:
            p.grad = saved_grads.get(id(p))
    return len(fresh)


# ------------------------------------------------------------------------------------- the freeze
class LearnerFreeze:
    """Snapshot the learner's object graph, then fail any acquisition after it. See module docs."""

    def __init__(self, model: Any, *, emit: Optional[Callable[[str], None]] = None):
        self.model = model
        self._emit = emit
        self.frozen: Optional[FrozenSet] = None
        self.where: Optional[str] = None
        self.recorder = _SiteRecorder()
        self.violations: List[str] = []           # sticky (the step pre-hook's)
        self._step_hook: Any = None
        self.checks = 0
        self.memory: Optional[CudaMemoryWatch] = None   # the memory half, set by `attach`

    # -- lifecycle ------------------------------------------------------------------------------
    def freeze(self, where: str) -> str:
        self.frozen, _ = _walk(self.model)
        self.where = where
        self.recorder.install()
        frozen_opts = set(self.frozen.optimizers)
        freeze = self

        def step_pre_hook(opt: Any, args: Any, kwargs: Any) -> None:
            if id(opt) in frozen_opts:
                return None
            site = freeze.recorder.site_of(opt) or "constructed before the freeze, held outside the learner graph"
            msg = (f"an optimizer outside the frozen set STEPPED after the freeze ({freeze.where}): "
                   f"{_describe(opt)}; built: {site}; stepped at {_site(2)}")
            freeze.violations.append(msg)
            raise LazyAcquisitionError(f"{FATAL_TAG}: {msg}\n{_HOWTO}")

        from torch.optim.optimizer import register_optimizer_step_pre_hook
        self._step_hook = register_optimizer_step_pre_hook(step_pre_hook)
        fs = self.frozen
        n_state = sum(len(v) for v in fs.state.values())
        line = (f"🧊 [LEARNER FREEZE] {where}: {len(fs.modules)} modules, {len(fs.params)} parameters, "
                f"{len(fs.buffers)} buffers, {len(fs.optimizers)} optimizer(s) "
                f"[{', '.join(sorted(fs.optimizers.values()))}] with {n_state} state entries. From "
                f"here a new optimizer / parameter / module / buffer / optimizer-state entry in the "
                f"learner stops the run with a typed exit.")
        self._say(line)
        return line

    def release(self, why: str = "") -> None:
        if self._step_hook is not None:
            with contextlib.suppress(Exception):
                self._step_hook.remove()
            self._step_hook = None
        self.recorder.uninstall()
        if self.frozen is not None and why:
            self._say(f"🧊 [LEARNER FREEZE] released — {why}; {self.checks} checks passed.")
        self.frozen = None

    # -- the check ------------------------------------------------------------------------------
    def violations_now(self) -> List[str]:
        """Every acquisition since the freeze (empty when healthy). Pure read."""
        if self.frozen is None:
            return []
        out = list(self.violations)
        cur, objs = _walk(self.model)
        old = self.frozen

        def site(oid: int) -> str:
            s = self.recorder.site_of(objs.get(oid))
            return s or "no recorded site (built before the freeze, or outside any registration)"

        for oid, q in cur.modules.items():
            if oid not in old.modules:
                out.append(f"NEW MODULE {q} = {_describe(objs.get(oid))}; built: {site(oid)}")
        for oid, q in cur.params.items():
            if oid not in old.params:
                was = old.param_ids_by_name.get(q)
                what = "REPLACED PARAMETER" if was is not None else "NEW PARAMETER"
                out.append(f"{what} {q} = {_describe(objs.get(oid))}; built: {site(oid)}")
        for b in sorted(cur.buffers - old.buffers):
            out.append(f"NEW BUFFER {b}")
        for oid, path in cur.optimizers.items():
            if oid not in old.optimizers:
                out.append(f"NEW OPTIMIZER {path} = {_describe(objs.get(oid))}; built: {site(oid)}")
                continue
            if cur.groups[oid] != old.groups.get(oid):
                out.append(f"PARAM GROUPS CHANGED on optimizer {path}: "
                           f"{[len(g) for g in old.groups.get(oid, [])]} -> "
                           f"{[len(g) for g in cur.groups[oid]]} tensors per group")
            born = cur.state[oid] - old.state.get(oid, set())
            if born:
                names = [cur.params.get(i) or old.params.get(i) or f"<id {i}>" for i in born]
                out.append(f"OPTIMIZER STATE created after the freeze on {path} for {len(born)} "
                           f"parameter(s): {sorted(names)[:6]} — declare it at startup "
                           f"(`learner_lifecycle.declare_optimizer_state`)")
        return out

    def check(self, where: str) -> None:
        bad = self.violations_now()
        self.checks += 1
        if bad:
            raise LazyAcquisitionError(fatal_text(where, self.where, bad))

    def _say(self, msg: str) -> None:
        print(msg, flush=True)
        if self._emit is not None:
            with contextlib.suppress(Exception):
                self._emit(msg)


_HOWTO = ("The learner's steady state acquires NOTHING (the declared lifecycle, "
          "designs/endstate/program_rust_core.md): build it at startup — in the model build, "
          "`_setup_model`, or a `@lifecycle_decl.startup_builder` the trainer runs before the freeze "
          "— and declare optimizer state with `learner_lifecycle.declare_optimizer_state`. Fatal by "
          "design; the launcher does NOT restart it (FATAL_CONFIG). See "
          "designs/training/learner_lifecycle.md.")


def fatal_text(where: str, frozen_at: Optional[str], bad: List[str]) -> str:
    lines = "\n".join(f"  - {b}" for b in bad[:12])
    more = f"\n  … and {len(bad) - 12} more" if len(bad) > 12 else ""
    return (f"{FATAL_TAG} at {where}: {len(bad)} resource(s) acquired after the learner froze "
            f"({frozen_at}):\n{lines}{more}\n{_HOWTO}")


# ------------------------------------------------------------------------------ startup + wiring
def declare_learner_startup(model: Any, *, emit: Optional[Callable[[str], None]] = None) -> str:
    """The trainer's LAST startup step before the freeze: acquire what the steady state would
    otherwise build lazily. Today: every Adam/AdamW optimizer's state (`declare_optimizer_state`)
    and, under `--capacity-telemetry`, the canary head + its optimizer (`CapacityTerms.
    _capacity_declare`). Returns the log line."""
    declare = getattr(model, "_capacity_declare", None)
    cap = declare() if callable(declare) else None
    _, opts = learner_objects(model)
    made: List[str] = []
    for path, opt in sorted(opts.items()):
        if type(opt).__name__ in _DECLARABLE:
            n = declare_optimizer_state(opt)
            if n:
                made.append(f"{path} +{n}")
    line = (f"🧊 [LEARNER STARTUP] declared optimizer state: {', '.join(made) or 'none needed'}"
            + (f"; capacity telemetry: {cap}" if cap else ""))
    print(line, flush=True)
    if emit is not None:
        with contextlib.suppress(Exception):
            emit(line)
    return line


# ------------------------------------------------------------------------------ the memory half
class CudaMemoryWatch:
    """The call site of K6's CUDA memory TREND (`agents.training.cuda_memory_trend`, a leak detector
    for a clean early stop — never a gate): one `MemoryTrend` per process, started at the freeze;
    a sample at the two quiescent points of every update (after the rollout, after `train()`); at
    every window close the projection is LOGGED (one line) and recorded to TB (`lifecycle/cuda_*`);
    a STOP verdict raises `CudaMemoryLeakError`. Inert off CUDA. The window line goes to the run's
    log (``say``, stdout); a WARN that changes and the STOP also go to ``emit`` (the launcher's event
    channel). ``sampler`` is the test seam (default `cuda_memory_trend.sample_cuda`)."""

    def __init__(self, model: Any, *, emit: Optional[Callable[[str], None]] = None,
                 device: Any = None, sampler: Optional[Callable[..., Any]] = None,
                 say: Optional[Callable[[str], None]] = None) -> None:
        dev = device if device is not None else getattr(model, "device", None)
        self.device = dev if dev is not None and torch.device(dev).type == "cuda" else None
        self.model = model
        self.emit = emit
        self.say: Callable[[str], None] = say or (lambda m: print(m, flush=True))
        self.sampler = sampler
        self.trend: Any = None
        self.updates = 0

    def start(self) -> None:
        if self.device is not None and self.trend is None:
            from agents.training.cuda_memory_trend import MemoryTrend
            self.trend = MemoryTrend()

    def observe(self, phase: str) -> None:
        if self.trend is None:
            return
        from agents.training import cuda_memory_trend as cmt
        if phase == "post_update":
            self.updates += 1
        sampler = self.sampler or cmt.sample_cuda
        sample = sampler(self.device, update=self.updates, phase=phase)
        v = self.trend.observe(sample)
        # gen3_cuda_ledger_v1: the PEAK of the step that just ended — the update's (post_update) or
        # the rollout's (post_rollout) — every update, not only at a window close.
        logger = getattr(self.model, "_logger", None)
        if logger is not None and getattr(sample, "peak_allocated", -1) >= 0:
            what = "update" if phase == "post_update" else "rollout"
            logger.record(f"lifecycle/cuda_{what}_peak_alloc_mib", sample.peak_allocated / (1 << 20))
            logger.record(f"lifecycle/cuda_{what}_peak_reserved_mib", sample.peak_reserved / (1 << 20))
            logger.record(f"lifecycle/cuda_{what}_end_alloc_mib", sample.allocated / (1 << 20))
        if logger is not None:
            # every sample, one stable tag set whatever the window state (gen3_reserved_after_freeze_v1)
            for k, x in self.trend.sample_scalars(sample).items():
                logger.record(k, x)
        if v.window_closed:
            logger = getattr(self.model, "_logger", None)
            if logger is not None:
                for k, x in self.trend.tb_scalars().items():
                    logger.record(k, x)
            self.say(v.message)                        # the projection, every window, in the run log
        elif v.level == "WARN" and v.changed:
            self.say(v.message)
        if v.level in ("WARN", "STOP") and v.changed and self.emit is not None:
            self.emit(v.message)
        if v.level == "STOP":
            raise CudaMemoryLeakError(f"{STOP_TAG} — CUDA MEMORY LEAK: {v.message}")


def attach(model: Any, *, emit: Optional[Callable[[str], None]] = None,
           memory: Optional[CudaMemoryWatch] = None) -> LearnerFreeze:
    """Wire the freeze guard onto ``model``'s loop — the `learner_freeze` owner of the DECLARED hook
    table (`agents/training/loop_hooks.py`, OUTERMOST by the table's order; on a duck-typed model
    with no table, instance-attribute wrappers — attach AFTER `compile_control` there):

      * the FIRST `collect_rollouts` entry FREEZES (startup is over: the compile gate, the prewarm
        and `learn()`'s own `_on_training_start` have all run);
      * every `collect_rollouts` and `train` exit CHECKS (`LazyAcquisitionError`, a
        `FatalConfigError`: the trainer's fail-fast handler exits FATAL_CONFIG, not restarted);
      * `learn()`'s exit RELEASES (the in-process final evaluation is not the steady state);
      * on CUDA, the MEMORY half (`CudaMemoryWatch`) starts at the freeze and samples after every
        rollout and update — a projected OOM is `CudaMemoryLeakError` (exit FATAL_CUDA_LEAK after
        the trainer's handler saved `final_model_exception.zip`; the launcher restarts it, capped).

    `_learner_freeze` and the table are in `_excluded_save_params` (see the hub)."""
    freeze = LearnerFreeze(model, emit=emit)
    memory = memory if memory is not None else CudaMemoryWatch(model, emit=emit)   # says to stdout
    import contextlib

    from agents.training.loop_hooks import install

    @contextlib.contextmanager
    def collect() -> Any:
        if freeze.frozen is None and not getattr(freeze, "_released", False):
            freeze.freeze("the first rollout of learn()")
            memory.start()
        yield
        freeze.check("rollout end")
        memory.observe("post_rollout")

    @contextlib.contextmanager
    def update() -> Any:
        yield
        freeze.check("update end")
        memory.observe("post_update")

    @contextlib.contextmanager
    def learn() -> Any:
        try:
            yield
        finally:
            freeze._released = True                     # type: ignore[attr-defined]
            freeze.release("learn() returned")

    # gen3_declared_loop_hooks_v1: the loop's DECLARED hook table (outermost owner), or — for a
    # duck-typed model with no table — the same bodies as instance-attribute wrappers.
    install(model, "learner_freeze", {"collect": collect, "update": update, "learn": learn})
    model._learner_freeze = freeze
    freeze.memory = memory
    return freeze
