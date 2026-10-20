"""K8 — DECLARED COMPILE REGIONS (`gen3_declared_regions_v1`, M5 Lane K; owner 2026-09-30).

THE RULE. Every compiled learner-process callable is a named REGION in ONE declaration table
(`REGIONS`), compiled with `fullgraph=True` — so a graph break anywhere INSIDE a region is a compile
ERROR at startup naming dynamo's reason and file:line — at exactly its DECLARED signatures (shape ×
train/eval × grad), prewarmed at startup and then LOCKED (K6). The only legal "breaks" are the region
boundaries: plain Python between regions. A deliberate split is two regions, never an allowed break.
Rejected (Decision record): a pinned graph-break COUNT (a count hides a moved break) and break sites
pinned by file:line (line keys churn on every edit). Owner: correctness beats churn.

THE TABLE (`REGIONS`):

  R0  (DELETED, P10-E, owner-approved 2026-10-03)  The compiled ROLLOUT forward. On the only env core
                           the Rust collector serves every rollout forward through the T2 inference
                           service (`agents/inference/service`, its own gate on every served bucket),
                           so the learner process never called R0 — yet it was compiled, gated,
                           prewarmed and canaried at every launch (a sizing run's log: 240 compiled
                           region calls per update, all R1's). The learner's EAGER
                           `Gen3DualHeadMaskablePolicy.rollout_core` / `forward` stay for the
                           callers that are not the hot path (`predict`, the prober, the eval workers,
                           tests).
  R1  learner_micro_step   `instrumented_ppo.micro_step.micro_step`: `evaluate_actions` + fold steps
                           1–3a (`gen3_learner_micro_step_v1`). train / grad / batch batch_size; its
                           backward is AOTAutograd's, from the same graph. ONE declared row count:
                           the collector refuses an update that does not divide by the micro-batch
                           (`RustCollector._ensure_buffer`; the combination check
                           `rollout_target_on_the_quantum`; `compile_trainer.check_shape_stability` at
                           startup), so a ragged micro-batch never reaches R1 — and one that does
                           (a mis-wired caller) is REFUSED by the dispatcher, never run eager or
                           compiled as a new signature (`gen3_r1_no_ragged_v1`).
  R2  rank_probe         — NO forward at all (K8, `gen3_rank_device_v1`): the effective-rank probe
                           reads R1's OWN forward's stashes on the first micro-batch (trunk tokens,
                           value CLS, the projected pi / vf features) and computes the four spectra
                           on the device (`rank_metrics.effective_rank_t`), one host read.
  R3  optimizer_step       — NOT compiled, by decision: `clip_grad_norm_` + AdamW + `zero_grad`
                           measured 0.04 s per update (accumulation 32 — K8 inventory), and the
                           KL→LR controller moves the learning rate every update, so a compiled step
                           keyed on lr would be a new signature per update (or need lr as a tensor and
                           AdamW's capturable path) for no measurable gain.
  *   eager, declared      the batch build, label alignment, the per-micro host read, the EAGER TAIL
                           of the fold (the ride-along update; no loss term is left in it), every
                           diagnostic probe, logging.

TORCH. Regions are a torch 2.8 feature, and HEAD runs torch >= 2.8 only (`utils.torch_floor`): on
2.5.1 the (since deleted) `forward_guard`'s weakref lookup and other constructs break `fullgraph=True`. The regions are
the ONLY compiled learner surface (the 2.5.1 extractor-only compile was deleted 2026-10-02); a run
trained on 2.5.1 resumes pinned to its own commit, which still carries that compile.

THE GATE. `gate_regions` holds the compiled region to eager at startup on REAL rows — R1 ALWAYS on
the K9 learner golden's real labelled buffer (`gen3_r1_golden_rows_always_v1`, P10-C): a key this run
declares that the golden lacks is filled with its DECLARED placeholder (`fill_value`: the label
inventory's `host_const` value, e.g. the fork arm's `fork_pg_m` = 1.0, or the extra-obs-keys
registry's zeros), a golden key this run does not declare is dropped (its term is off here), and
anything else REFUSES. There is no zero-label fallback any more: it blanked the critic / intent /
belief losses, so ~48 of their parameters fell under the per-parameter floor unjudged (176 judged vs
219 on the golden rows) on every `--fork-fraction > 0` run. When a key was filled, R1's judged
parameter set is ALSO held to the golden rows' own (the same weights, the filled keys removed and the
static resolved without them): a parameter the golden judges CLEARLY that this run's rows do not judge
is a FATAL (`judged_set_shrinkage`). R1's loss and the gradient over
every policy parameter (cosine ≥ 0.9999 and the per-parameter rule; on a FRESH launch also on a seeded
perturbation, whose legal log-probs are not constant). Every rung judges the golden rows MOVED OFF THE PPO
CLIP BOUNDARY on its own weights (`relocate_clip_boundary`, `gen3_r1_clip_margin_v1`, T36; standing rule 8), so no
verdict depends on one row's 0/1 clip switch. The gate runs at fp32 matmul precision 'highest' only — the one precision (TF32 was retired, deletion pass
K2); a process at any other precision is refused. A disagreement is a `CompileTrainerError`
(FATAL_CONFIG).
"""
from __future__ import annotations

import contextlib
import hashlib
import math
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from agents.model import inductor_compile as _ic
from agents.model import compile_trainer as ct
from agents.model.compile_gate_probe import per_param_grad_errors
from agents.model import region_calls as RC
from agents.training.lifecycle_decl import startup_builder


class Signature(NamedTuple):
    batch: str                       # "batch_size"
    mode: str                        # "eval" | "train"
    grad: bool
    #: Which compiled callable serves it: "compiled" (the region's one compile) or "graphed" (the same
    #: function compiled again with Inductor's cudagraph trees, `--learner-cudagraphs on`).
    route: str = "compiled"
    #: The trainer flag that DECLARES this signature (None = always declared). A gated signature exists
    #: only on a run whose flag is on (`declared_signatures`).
    gate: Optional[str] = None


class Region(NamedTuple):
    name: str
    callable: str
    signatures: Tuple[Signature, ...]
    compiled: bool
    why: str


REGIONS: Tuple[Region, ...] = (
    Region("R1_learner_micro_step", "instrumented_ppo.micro_step.micro_step",
           (Signature("batch_size", "train", True),
            # T25 lever C (`gen3_r1_graphed_route_v1`, `--learner-cudagraphs on`): the SAME signature
            # replayed from cudagraph trees on every micro-step that carries no retain_graph probe; the
            # "compiled" route above serves the probe micro-steps (`r1_route_for`).
            Signature("batch_size", "train", True, "graphed", "learner_cudagraphs")), True,
           "the accumulation group + optimizer step are per GROUP; the host read follows it"),
    Region("R2_rank_probe", "rank_metrics.rank_probe_from_stash (no forward)", (), False,
           "reads R1's own forward's stashes; the spectra are device tensors, one host read"),
    Region("R3_optimizer_step", "clip_grad_norm_ + AdamW.step + zero_grad", (), False,
           "0.04 s/update measured; the KL->LR controller moves lr every update"),
)


#: T25 perf levers B / C (`gen3_r1_inductor_presets_v1`; `designs/research_state/measurements/perf_phase_2026-10-10/
#: PLAN.md`): NAMED Inductor option sets for R1's compile, passed as `torch.compile(..., options=)` so they reach R1
#: only (never T2's graphs, never a global config). A preset is set on the learner as `_r1_inductor_preset` BEFORE
#: `install` (default ``"default"``: no options, the production compile byte for byte); presets join with ``+``.
#: All are numerics-neutral in intent — `coordesc` / `combo` re-tile / co-schedule kernels (an fp32 summation order
#: may move within fp32 noise, which the startup gate and the canary judge as for any compile) and `cudagraphs`
#: replays the same kernels from a captured graph — but each is a MEASUREMENT lever until its paired benchmark and
#: the gates have run on the GPU (no trainer flag yet).
R1_INDUCTOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "coordesc": {"coordinate_descent_tuning": True},     # tile / warp search per pointwise & reduction kernel
    "combo": {"combo_kernels": True},                    # horizontal fusion of independent small kernels
    "cudagraphs": {"triton.cudagraphs": True},           # cudagraph trees over R1's forward + backward
}


#: The CPU-ONLY Inductor options (`gen3_cpu_inductor_scalar_v1`, 2026-10-15) — defined in, and applied by, THE one
#: compile seam `agents.model.inductor_compile` (`compile_on`), which every compile site in `src/` goes through
#: (`gen3_cpu_compile_sites_v1`). Re-exported here for R1's readers. A CPU learner's R1 carries it; a CUDA learner's
#: kwargs are unchanged byte for byte (pinned by `cpu_inductor_tile2d_test`). The trainer's `--compile-trainer`
#: refuses a CPU device, so on R1 it serves the TEST SEAMS that compile R1 on CPU (`compiled_first_update_test`).
CPU_INDUCTOR_OPTIONS: Dict[str, Any] = _ic.CPU_INDUCTOR_OPTIONS


def learner_device_type(model: Any) -> str:
    """``"cpu"`` / ``"cuda"`` / ... — the device type ``model`` (an SB3 learner) trains on: its `device`, else
    its policy's first parameter's."""
    dev = getattr(model, "device", None)
    if dev is None:
        dev = next(iter(model.policy.parameters())).device
    return torch.device(dev).type


def cpu_inductor_options(model: Any, backend: Optional[str] = None) -> Dict[str, Any]:
    """`CPU_INDUCTOR_OPTIONS` when R1 compiles on a CPU learner through Inductor (``backend`` None or
    ``"inductor"``); ``{}`` for a CUDA learner and for any other backend (an `eager` / `aot_eager` stand-in takes
    no Inductor options)."""
    return _ic.inductor_options(learner_device_type(model), backend)


def r1_inductor_options(preset: str) -> Dict[str, Any]:
    """The merged Inductor options of ``preset`` (names from `R1_INDUCTOR_PRESETS`, joined by ``+``); an
    unknown name raises naming the known ones."""
    out: Dict[str, Any] = {}
    for name in str(preset or "default").split("+"):
        if name not in R1_INDUCTOR_PRESETS:
            raise ValueError(f"unknown R1 Inductor preset {name!r}; known: {sorted(R1_INDUCTOR_PRESETS)}")
        out.update(R1_INDUCTOR_PRESETS[name])
    return out


def compiled_regions() -> Tuple[Region, ...]:
    return tuple(r for r in REGIONS if r.compiled)


def learner_cudagraphs_on(model: Any = None) -> bool:
    """Is the GRAPHED R1 route declared on ``model`` (`--learner-cudagraphs on`, set by
    `compile_trainer.arm_compile_sentinel` before `install`)? False for None / an unset model."""
    return bool(getattr(model, "_learner_cudagraphs", False)) if model is not None else False


def declared_signatures(model: Any = None) -> Tuple[Signature, ...]:
    """The compiled signatures ``model``'s run declares: every ungated one, and a gated one only when
    its flag is on for ``model`` (None = no flag on: the production table)."""
    on = {"learner_cudagraphs": learner_cudagraphs_on(model)}
    return tuple(s for r in compiled_regions() for s in r.signatures
                 if s.gate is None or on.get(s.gate, False))


def declared_signature_count(model: Any = None) -> int:
    """How many compiled graphs the table declares for ``model``'s run — one dynamo cache entry each on
    R1's code object (`assert_inventory`)."""
    return len(declared_signatures(model))


def r1_route_for(*, retained_graph: bool) -> str:
    """WHICH R1 callable a micro-step takes (`gen3_r1_graphed_route_v1`): ``"compiled"`` when the
    micro-step will call autograd through R1's graph a SECOND time — a `retain_graph=True` probe before
    its `backward()` (the per-term noise sampler's flush, the grad-balance probe) — else ``"graphed"``.
    Decided BEFORE the call from the diagnostics plan and the probes' own state
    (`instrumented_ppo.diagnostics_cadence.micro_retains_graph`): deterministic, never a try / except.
    WHY: cudagraph trees record ONE backward per forward; a second traversal of the graph dies in the
    recording ("graph recording observed an input tensor deallocate during graph recording that did not
    occur during replay", F-PB-2) — so no retain_graph probe may ever run on the graphed route."""
    return "compiled" if retained_graph else "graphed"


# ------------------------------------------------------------------------------------- install
def install(model: Any, *, backend: Optional[str] = None,
            emit: Optional[Callable[[str], None]] = None,
            graphed_backend: Optional[Any] = None) -> List[str]:
    """Compile R1 (`fullgraph=True`, static shapes) and install it as ``model._compiled_micro_step``
    (read by `TrainSetup._micro_region`). Any instance-level extractor forward (none is installed on
    the learner since the extractor-only compile's deletion) is removed first, so the region traces
    the extractor's own forward and every other caller runs it eager. Returns the installed region
    names. Compilation happens LAZILY at the region's first call — the gate and the prewarm make that
    first call at startup.

    Under `--learner-cudagraphs on` (``model._learner_cudagraphs``, gen3_r1_graphed_route_v1) R1 is
    compiled a second time as the GRAPHED route (``model._compiled_micro_step_graphed``): the same
    kwargs plus Inductor's ``triton.cudagraphs``. ``graphed_backend`` is a TEST SEAM (CPU has no
    cudagraphs): that route compiles with it instead, without options."""
    from agents.model.compile_control import control
    from agents.training.instrumented_ppo.micro_step import micro_step
    policy = model.policy
    fe = policy.features_extractor
    # K3 (gen3_hermetic_compile_cache_v1): the regions compile into the RUN's own cache (declared
    # by the trainer at startup; idempotent here), never a cache torch would pick by default.
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("learner regions compile")
    if "forward" in vars(fe):
        del fe.forward
    ctl = control(emit)
    kw: Dict[str, Any] = {"fullgraph": True, "dynamic": False}
    if backend is not None:
        kw["backend"] = backend
    preset = str(getattr(model, "_r1_inductor_preset", "default") or "default")
    opts = r1_inductor_options(preset)   # + the CPU workaround, added by `compile_on` on a CPU learner only
    dev = learner_device_type(model)
    graphed = learner_cudagraphs_on(model)
    if graphed and "cudagraphs" in preset.split("+"):
        raise ct.CompileTrainerError(
            f"--learner-cudagraphs on with the R1 Inductor preset {preset!r}: the preset graphs R1's ONE "
            f"compile, so the retain_graph probes would run through cudagraph trees again (F-PB-2). The "
            f"flag declares its own graphed route; drop `cudagraphs` from the preset "
            f"(gen3_r1_graphed_route_v1)")
    # a non-default preset (T25 levers B / C) or a CPU learner (gen3_cpu_inductor_scalar_v1) passes `options`;
    # CUDA's default passes none
    r1c = ctl.wrap_compiled(_ic.compile_on(micro_step, device=dev, options=opts, **kw))
    rows = int(getattr(model, "batch_size", 0) or 0)
    model._compiled_micro_step = _r1_dispatcher(r1c, rows, "compiled")
    names = ["R1_learner_micro_step"]
    if hasattr(model, "_compiled_micro_step_graphed"):   # a re-install drops an earlier graphed route
        model._compiled_micro_step_graphed = None
    if graphed:
        # T25 lever C (`gen3_r1_graphed_route_v1`): R1 compiled a SECOND time, the same function and the
        # same options plus cudagraph trees — a second DECLARED signature on R1's code object (dynamo keys
        # a cache entry by its compile wrapper), gated, prewarmed and locked like the first.
        if graphed_backend is not None:              # the CPU test seam (no options: a stand-in)
            r1g = ctl.wrap_compiled(_ic.compile_on(micro_step, device=dev, **dict(kw, backend=graphed_backend)))
        else:
            r1g = ctl.wrap_compiled(_ic.compile_on(micro_step, device=dev, options={
                **opts, **R1_INDUCTOR_PRESETS["cudagraphs"]}, **kw))
        model._compiled_micro_step_graphed = _r1_dispatcher(r1g, rows, "graphed")
        names.append("R1_learner_micro_step[graphed]")
        from agents.model.compile_control import cudagraph_skips
        model._cudagraph_skips_at_install = cudagraph_skips()
    # the R1 signature this run DECLARES (its levers from the resolved config) — every update is
    # held to it (`check_r1_declared`)
    model._r1_declared = r1_declaration(model)
    return names


def _r1_dispatcher(r1c: Callable[..., Any], rows: int, route: str) -> Callable[..., Any]:
    """The installed R1 callable of one ROUTE: the row-count refusal, the no-silent-eager check and
    the route counters (``R1_compiled`` for every compiled call; ``R1_graphed`` for the graphed one)."""

    def r1(policy: Any, obs: Any, actions: Any, *rest: Any) -> Any:
        # R1's ONE declared signature is `batch_size` rows (`gen3_r1_no_ragged_v1`). The collector
        # refuses an update that does not divide by the micro-batch, so another row count is a
        # mis-wired caller: REFUSED here, naming the cause — never an eager run (a silent ~2x
        # slowdown) and never a second compiled signature (the sentinel's FATAL after the lock, a
        # silent extra graph before it).
        if int(actions.shape[0]) != rows:
            raise ct.CompileTrainerError(
                f"--compile-trainer region R1: a micro-batch of {int(actions.shape[0])} rows, but the "
                f"declared signature is {rows} (--batch-size). The collector guarantees full "
                f"micro-batches (an update that does not divide by the batch size is refused at "
                f"startup and at every buffer build), so this call is mis-wired — R1 has no ragged "
                f"route (gen3_r1_no_ragged_v1)")
        before = RC.EAGER_BODY["R1"]
        out = r1c(policy, obs, actions, *rest)
        _ran_compiled("R1", before)
        RC.count("R1_compiled")
        if route == "graphed":
            RC.count("R1_graphed")
        return out
    r1._gen3_compiled = r1c                              # type: ignore[attr-defined]
    r1._gen3_route = route                               # type: ignore[attr-defined]
    return r1


def r1_routes(model: Any) -> List[Tuple[str, Callable[..., Any]]]:
    """Every installed R1 route as ``(route, callable)`` — the graphed one FIRST (it serves ~every
    micro-step, so the gate and the canary judge it first), then the compiled one."""
    out: List[Tuple[str, Callable[..., Any]]] = []
    g = getattr(model, "_compiled_micro_step_graphed", None)
    if g is not None:
        out.append(("graphed", g))
    c = getattr(model, "_compiled_micro_step", None)
    if c is not None:
        out.append(("compiled", c))
    return out


def _ran_compiled(region: str, before: int) -> None:
    """The compiled route of ``region`` must not have executed the region's Python body: if it did,
    dynamo ran it EAGER (disabled, a swallowed error, a skipped frame) — a typed FATAL, never a
    silent slowdown (`agents.model.region_calls`)."""
    if RC.EAGER_BODY[region] != before:
        raise ct.CompileTrainerError(
            f"--compile-trainer: the declared region {region} ran EAGER on its COMPILED route — "
            f"dynamo did not execute the compiled graph (torch._dynamo.config.disable "
            f"{getattr(torch._dynamo.config, 'disable', None)}, suppress_errors "
            f"{torch._dynamo.config.suppress_errors}). A partly uncompiled learner is a silent ~2x "
            f"slowdown: stopped (gen3_no_silent_eager_v1)")


def assert_inventory(model: Any) -> str:
    """The compiled-region INVENTORY equals the declaration, in the run (startup, after the
    prewarm): one cache entry on R1's code object. Other code objects are not judged (they belong to
    other components). Raises `CompileTrainerError`; returns the log line."""
    from agents.model.compile_control import cache_entries_by_code
    skip = graphed_route_skipped(model)          # the graphed route's own failure, named first
    if skip is not None:
        raise ct.CompileTrainerError(skip)
    ent = cache_entries_by_code()
    want = {"micro_step": declared_signature_count(model)}
    got = {name: sum(v for k, v in ent.items() if k.split(" ")[0] == name) for name in want}
    if got != want:
        raise ct.CompileTrainerError(
            f"--compile-trainer: the compiled-region INVENTORY differs from the declaration — "
            f"cache entries {got}, declared {want} (gen3_no_silent_eager_v1)")
    if not learner_cudagraphs_on(model):
        return "[CompileRegions] inventory == declaration: R1 1 graph (fullgraph=True)"
    return ("[CompileRegions] inventory == declaration: R1 2 graphs (fullgraph=True) — the compiled "
            "route + the GRAPHED route (cudagraph trees, --learner-cudagraphs on; 0 cudagraph skips)")


def graphed_route_skipped(model: Any) -> Optional[str]:
    """The FATAL's text when Inductor declined to cudagraph a graph since the graphed route was installed
    (its `cudagraph_skips` counter moved: an input mutation, a CPU tensor, …) — the graphed route would
    then run UN-graphed, a silent no-op lever (`gen3_r1_graphed_route_v1`, the no-silent-eager rule
    applied to graphs). None when healthy or when no graphed route is declared."""
    if not learner_cudagraphs_on(model):
        return None
    from agents.model.compile_control import cudagraph_skips
    base = getattr(model, "_cudagraph_skips_at_install", None)
    now = cudagraph_skips()
    if base is None or now <= int(base):
        return None
    return (f"--learner-cudagraphs on: Inductor SKIPPED cudagraphs {now - int(base)} time(s) since R1's "
            f"graphed route was installed (torch._dynamo counters['inductor']['cudagraph_skips']), so "
            f"the graphed route runs UN-graphed — a silent no-op lever. Refusing "
            f"(gen3_r1_graphed_route_v1). Diagnose with TORCH_LOGS=perf_hints on a short launch.")


def uninstall(model: Any) -> None:
    model._compiled_micro_step = None
    if getattr(model, "_compiled_micro_step_graphed", None) is not None:
        model._compiled_micro_step_graphed = None
    model._r1_declared = None


def installed(model: Any) -> bool:
    return getattr(model, "_compiled_micro_step", None) is not None


# ------------------------------------------------------------------------------ signature inputs
class R1Batch(NamedTuple):
    obs: Dict[str, torch.Tensor]
    actions: torch.Tensor
    action_masks: torch.Tensor
    old_log_prob: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    source: str
    filled: Tuple[str, ...] = ()      # space keys the golden lacks, at their DECLARED placeholder
    #: per batch row, the committed fixture row it tiles (`compile_trainer.fixture_index`): the canary records the
    #: worst row of every arm as a FIXTURE row (gen3_r1_yardstick_margin_v1), so a tail can be named afterwards
    fixture_rows: Optional[np.ndarray] = None


def fill_value(key: str, space: Any) -> Optional[np.ndarray]:
    """ONE row of ``key``'s DECLARED placeholder (``space``'s shape and dtype), or None when nothing
    declares one. The declarations: the training-label inventory's ``host_const`` rows (the constant
    the env writes on every row — the fork arm's `fork_pg_m` = 1.0, a MULTIPLIER on the policy term,
    so the golden rows' policy loss is unchanged) and the extra-obs-keys registry (the all-zero
    block, the honest "unavailable" encoding a real emitter supplies)."""
    from agents.model.extra_obs_keys import BY_KEY as EXTRA
    from utils.rust_env.label_inventory import BY_KEY as LABELS
    row = LABELS.get(key)
    if row is not None and row.rust == "host_const" and row.const is not None:
        return np.full(tuple(space.shape), row.const, dtype=space.dtype)
    extra = EXTRA.get(key)
    if extra is not None:
        return np.asarray(extra.zeros(), dtype=space.dtype).reshape(tuple(space.shape))
    return None


class GoldenRows(NamedTuple):
    data: Dict[str, np.ndarray]                # the committed buffer (``obs:<key>`` + the PPO fields)
    filled: Dict[str, np.ndarray]              # space key the golden lacks -> one placeholder row
    dropped: Tuple[str, ...]                   # golden obs keys this run's space does not declare


def golden_rows(model: Any) -> GoldenRows:
    """The K9 learner golden's committed REAL buffer, fitted to this run's observation space — or a
    `CompileTrainerError` (FATAL_CONFIG) naming why it cannot be. R1 is never judged on other rows."""
    from agents.training.learner_golden import BUFFER_PATH
    try:
        with np.load(BUFFER_PATH) as z:
            data = {k: z[k] for k in z.files}
    except Exception as exc:
        raise ct.CompileTrainerError(
            f"--compile-trainer: region R1's startup gate reads the K9 learner golden's buffer "
            f"({BUFFER_PATH}) and could not: {type(exc).__name__}: {exc}") from exc
    space = model.policy.observation_space.spaces
    have = {k[4:]: data[k] for k in data if k.startswith("obs:")}
    bad = [f"{k}: golden {tuple(have[k].shape[2:])}, run {tuple(space[k].shape)}"
           for k in sorted(set(have) & set(space)) if tuple(have[k].shape[2:]) != tuple(space[k].shape)]
    filled: Dict[str, np.ndarray] = {}
    undeclared: List[str] = []
    for k in sorted(set(space) - set(have)):
        v = fill_value(k, space[k])
        if v is None:
            undeclared.append(k)
        else:
            filled[k] = v
    if bad or undeclared:
        raise ct.CompileTrainerError(
            "--compile-trainer: region R1's startup gate judges the compiled micro-step on the K9 "
            "learner golden's REAL labelled rows, and this run's observation space does not fit them"
            + (f" — shape mismatch ({'; '.join(bad)}): the observation layout changed without "
               f"`python -m agents.training.learner_golden rebuild-buffer`" if bad else "")
            + (f" — key(s) {undeclared} the golden lacks have NO declared placeholder (a "
               f"`host_const` row in `utils.rust_env.label_inventory`, or an `extra_obs_keys` row): "
               f"declare one, or rebuild the golden buffer with the key" if undeclared else "")
            + ". Judging R1 on other rows (the old zero-label fallback) leaves the label-driven "
              "terms' parameters unjudged (P10-C, gen3_r1_golden_rows_always_v1).")
    return GoldenRows(data, filled, tuple(sorted(set(have) - set(space))))


def r1_batch(model: Any, batch: int, slice_: int = 0) -> R1Batch:
    """A micro-batch shaped EXACTLY like `rollout_buffer.get(batch_size)`'s (every key of the
    observation space, the buffer's dtypes): the golden's real labelled rows tiled to ``batch``, a key
    the golden lacks at its declared placeholder (`golden_rows`; a run they cannot fit REFUSES)."""
    dev = ct.resolve_device(model.policy.features_extractor)
    space = model.policy.observation_space.spaces
    # the buffer's own action dtype (a Discrete space stores int64): a different dtype is a different
    # signature — measured: the sentinel named `tensor 'actions' dtype mismatch` after the lock.
    adt = torch.as_tensor(np.zeros(1, dtype=model.rollout_buffer.actions.dtype)).dtype
    g = golden_rows(model)
    data = g.data
    n = int(data["actions"].reshape(-1).shape[0])
    idx = ct.fixture_index(int(batch), n, slice_)

    def flat(a: np.ndarray) -> np.ndarray:
        rows: np.ndarray = a.reshape(-1, *a.shape[2:])[idx]
        return rows
    obs = {k: torch.as_tensor(flat(data["obs:" + k]).astype(space[k].dtype), device=dev)
           for k in space if k not in g.filled}
    for k, row in g.filled.items():
        obs[k] = torch.as_tensor(np.broadcast_to(row, (len(idx), *row.shape)).copy(), device=dev)
    obs = {k: obs[k] for k in space}                     # the space's key order
    f = {k: torch.as_tensor(flat(data[k]), device=dev)
         for k in ("actions", "log_probs", "values", "advantages", "returns", "action_masks")}
    src = "the K9 learner golden's real labelled buffer"
    if g.filled:
        src += " + " + ", ".join(f"{k}={float(np.asarray(v).reshape(-1)[0]):g} (declared placeholder)"
                                 for k, v in g.filled.items())
    if g.dropped:
        src += f"; golden keys not in this run's space: {list(g.dropped)}"
    return R1Batch(obs, f["actions"].to(adt).reshape(-1, 1), f["action_masks"].float(),
                   f["log_probs"].float().reshape(-1), f["values"].float().reshape(-1),
                   f["advantages"].float().reshape(-1), f["returns"].float().reshape(-1),
                   src, tuple(g.filled), np.asarray(idx))


class R1Declaration(NamedTuple):
    """R1's DECLARED signature beyond the batch's shapes (`gen3_r1_declared_levers_v1`): the static
    flags (`MicroStatic`) and the observation KEY set. Resolved from the run's config and its buffer's
    key set at startup (`r1_declaration`) — the same call `train()` makes — and recorded on the model
    by `install`; `check_r1_declared` holds every update to it."""
    static: Any
    obs_keys: Tuple[str, ...]


def _r1_static(model: Any) -> Any:
    """The `MicroStatic` exactly as `train()` resolves it (`TrainSetup._micro_static`)."""
    return model._micro_static(model._resolve_fold_flags())


def r1_declaration(model: Any) -> R1Declaration:
    return R1Declaration(_r1_static(model), tuple(sorted(model.policy.observation_space.spaces)))


def check_r1_declared(model: Any, st: Any) -> None:
    """Raise `CompileSentinelError` (a typed FATAL; `compile_control.attach` turns it into
    FATAL_CONFIG) when THIS update's R1 inputs are not the startup declaration — a lever, a coefficient
    or an observation key that moved. Never relaxed into a recompile: the fix is to resolve the value
    in `TrainSetup._micro_static`, from the config and the buffer's key set, before the lock."""
    decl: Optional[R1Declaration] = getattr(model, "_r1_declared", None)
    if decl is None:
        return
    diffs = [f"{k}: declared {a!r}, this update {b!r}"
             for k, a, b in zip(st._fields, decl.static, st) if a != b]
    obs_keys = tuple(sorted(model.rollout_buffer.observations)) if isinstance(
        model.rollout_buffer.observations, dict) else decl.obs_keys
    if obs_keys != decl.obs_keys:
        diffs.append(f"observation keys: undeclared {sorted(set(obs_keys) - set(decl.obs_keys))}, "
                     f"missing {sorted(set(decl.obs_keys) - set(obs_keys))}")
    if diffs:
        from agents.model.compile_control import CompileSentinelError
        raise CompileSentinelError(
            "R1 learner_micro_step: this update's signature is NOT the one declared at startup "
            "(gen3_r1_declared_levers_v1) — " + "; ".join(diffs) + ". Every lever a run will ever "
            "turn on is declared from its resolved config before the compile lock: resolve it in "
            "`TrainSetup._micro_static`, never relax the lock.")


def _r1_advantages(model: Any, b: R1Batch) -> torch.Tensor:
    """The advantage argument R1 receives in ``train()`` for ``b``'s rows: the batch's own, or — on a DECOUPLED-PPO
    run (X14 `--overlap-collect`; its static turns R1's normalisation off) — the same rows normalised the way
    ``train()`` hands them over at w = 1 (`overlap.decoupled_advantages`), so every gate judges R1 on the inputs a
    production update gives it (`gen3_overlap_norm_then_weight_v1`)."""
    from agents.training.rust_rollout.overlap import decoupled, decoupled_advantages
    if not decoupled(model):
        return b.advantages
    return decoupled_advantages(b.advantages, torch.ones_like(b.advantages), bool(model.normalize_advantage))


def _r1_args(model: Any, b: R1Batch) -> Tuple[Any, ...]:
    return (model.policy, b.obs, b.actions, b.action_masks,
            b.old_log_prob, b.old_values, _r1_advantages(model, b), b.returns, _r1_static(model))


def _r1_arm(model: Any, fn: Callable[..., Any], args: Tuple[Any, ...]) -> Dict[str, torch.Tensor]:
    """One R1 call + backward: the loss and the flat gradient over every policy parameter."""
    names_params = ct.grad_parameters(model, model.policy.features_extractor)
    params = [p for _, p in names_params]
    for p in params:
        p.grad = None
    out = fn(*args)
    out.loss.backward()
    gs = [(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten()
          for p in params]
    for p in params:
        p.grad = None
    # `.clone()`: on the graphed route the loss is cudagraph-pool memory the next replay overwrites
    # (`gen3_r1_graphed_route_v1`); the flat gradient is a fresh `cat` already.
    return {"loss": out.loss.detach().float().reshape(1).clone(), "grad": torch.cat(gs),
            "grad_sizes": torch.tensor([g.numel() for g in gs], dtype=torch.long),
            "log_prob": out.log_prob.detach().float().reshape(-1).clone()}


def _r1_accum_arm(model: Any, fn: Callable[..., Any], args_list: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
    """``len(args_list)`` R1 calls + backwards ACCUMULATED into one ``.grad`` (as `train()`'s accumulation
    group does, `.grad` cleared once before the first): the summed loss and the flat accumulated gradient."""
    names_params = ct.grad_parameters(model, model.policy.features_extractor)
    params = [p for _, p in names_params]
    for p in params:
        p.grad = None
    losses, lps = [], []
    for args in args_list:
        out = fn(*args)
        out.loss.backward()
        losses.append(out.loss.detach().float().reshape(1).clone())
        lps.append(out.log_prob.detach().float().reshape(-1).clone())
    gs = [(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten()
          for p in params]
    flat = torch.cat(gs)
    for p in params:
        p.grad = None
    return {"loss": torch.stack(losses).sum().reshape(1), "grad": flat,
            "grad_sizes": torch.tensor([g.numel() for g in gs], dtype=torch.long), "log_prob": torch.cat(lps)}


def _routes() -> Tuple[int, int]:
    """(R1's eager-body executions, R1's compiled-route dispatches) so far — `region_calls`' counters."""
    return int(RC.EAGER_BODY["R1"]), int(RC.peek().get("R1_compiled", 0))


def _r1_pair(model: Any, args: Tuple[Any, ...], fn: Optional[Callable[..., Any]] = None
             ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """``(compiled, eager)`` arms of ONE R1 parity comparison, with their INDEPENDENCE proved
    (`gen3_gate_independent_arms_v1`, F-XC-4): the compiled arm must go through the installed
    dispatcher's COMPILED route exactly once and execute R1's Python body ZERO times, and the eager arm
    must execute R1's Python body exactly once and dispatch the compiled route ZERO times. Anything else
    — the compiled slot holding the eager function (a thing compared with itself), a dispatcher that
    fell back to eager, an "eager" arm that dispatched the compiled graph — is a typed refusal, never a
    verdict. Counters, not identities: they read what each arm actually EXECUTED (`region_calls`) —
    so a wrapper around the installed dispatcher (a test's planted fault) is still the compiled arm,
    and the bare eager function in the compiled slot is refused (it runs the eager body). ``fn`` is the
    route under judgement (`r1_routes`); None = the installed compiled route."""
    from agents.training.instrumented_ppo.micro_step import micro_step
    if fn is None:
        fn = getattr(model, "_compiled_micro_step", None)
    if fn is None:
        raise ct.GateArmsNotIndependentError(
            "--compile-trainer region R1: no compiled micro-step is installed, so the parity gate has no "
            "COMPILED arm (gen3_gate_independent_arms_v1)")
    e0, c0 = _routes()
    comp = _r1_arm(model, fn, args)
    _mem("compiled arm")
    e1, c1 = _routes()
    eager = _r1_arm(model, micro_step, args)
    _mem("eager arm")
    e2, c2 = _routes()
    if (e1 - e0, c1 - c0) != (0, 1) or (e2 - e1, c2 - c1) != (1, 0):
        raise ct.GateArmsNotIndependentError(
            f"--compile-trainer region R1: the parity gate's two arms are NOT independent — the "
            f"compiled arm ran R1's eager body {e1 - e0}x and its compiled route {c1 - c0}x (want 0 / 1), "
            f"the eager arm ran the eager body {e2 - e1}x and the compiled route {c2 - c1}x (want 1 / 0). "
            f"A gate whose two sides share their arithmetic proves nothing: refusing "
            f"(gen3_gate_independent_arms_v1)")
    return comp, eager


@contextlib.contextmanager
def _fatal_compile_errors() -> Iterator[None]:
    """A region that cannot compile `fullgraph=True` — a graph break inside it — is a typed STARTUP
    FATAL naming dynamo's reason and the user frame (`CompileTrainerError` -> FATAL_CONFIG), never a
    crash the launcher would restart into the identical failure."""
    try:
        yield
    except ct.CompileTrainerError:
        raise
    except Exception as exc:
        from agents.model.compile_parity_fixture import ParityFixtureError
        if isinstance(exc, ParityFixtureError):             # a missing / stale fixture REFUSES
            raise ct.CompileTrainerError(f"--compile-trainer: {exc}") from exc
        try:
            from torch._dynamo.exc import TorchDynamoException
        except Exception:                                   # pragma: no cover
            TorchDynamoException = ()                       # type: ignore[assignment,misc]
        if isinstance(exc, TorchDynamoException) or "torch._dynamo" in type(exc).__module__:
            codegen = ("\nNOT a graph break: the BACKEND's code generation failed (Inductor's C++ compile). On a CPU "
                       "learner see `designs/training/compile_flags.md` \"The CPU Inductor tile2d codegen bug\" "
                       "(gen3_cpu_inductor_scalar_v1) — R1 on CPU carries `compile_regions.CPU_INDUCTOR_OPTIONS`, and a CPU compiled "
                       "learner is a test seam only (`--compile-trainer` refuses a CPU device)."
                       if "CppCompileError" in str(exc) else "")
            raise ct.CompileTrainerError(
                f"--compile-trainer: a DECLARED REGION does not compile as one graph "
                f"(`fullgraph=True`) — {type(exc).__name__}: {str(exc)[:3000]}\nA graph break inside a "
                f"region is an error by declaration (K8, gen3_declared_regions_v1): make the construct "
                f"traceable, or split the region in `compile_regions.REGIONS` deliberately.{codegen}") from exc
        raise


# ------------------------------------------------------------------------------------ the gate
def gate_regions(model: Any, *, batch_size: int, say: Callable[[str], None] = print) -> List[str]:
    """Hold R1 to eager at startup (module docstring). Raises `CompileTrainerError`."""
    from agents.model.parity_probe import unmeasured_precision
    refusal = unmeasured_precision()
    if refusal is not None:
        raise ct.CompileTrainerError(f"--compile-trainer: {refusal}")
    policy = model.policy
    fe = policy.features_extractor
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    names = [n for n, _ in ct.grad_parameters(model, fe)]
    rules: List[str] = []
    was = policy.training
    mem = GateMemory(model)
    try:
        with _fatal_compile_errors(), torch.random.fork_rng(devices=devices), gate_memory(mem):
            # ---- R1: train / grad / batch_size, on real labelled rows
            regime = weights_regime(model)
            b = r1_batch(model, int(batch_size))
            policy.set_training_mode(True)
            args = _r1_args(model, b)
            eager, _, r1_rules = r1_rungs(model, args, names, regime, source=b.source)
            rules.extend(r1_rules[:1])
            if b.filled:
                rules.append("R1 " + _r1_judged_set_rule(model, b, eager, names))
            rules.extend(r1_rules[1:])
            # T25 lever C (`gen3_r1_graphed_route_v1`): the GRAPHED route gets the same rungs, plus the
            # one thing a single fwd/bwd cannot show — a gradient ACCUMULATED across two replays (the
            # graph pool's outputs are overwritten by the next replay; `.grad` must not alias them).
            if learner_cudagraphs_on(model):
                for route, fn in r1_routes(model):
                    if route != "compiled":
                        rules.extend(r1_rungs(model, args, names, regime, source=b.source,
                                              fn=fn, route=route)[2])
                rules.append(_r1_accum_rule(model, names, regime, int(batch_size)))
    finally:
        policy.set_training_mode(was)
        for p in policy.parameters():
            p.grad = None
        release_gate_memory(model)                       # gen3_gate_releases_memory_v1, on a refusal too
    from agents.model.trunk_precision import trunk_precision_of
    prec = trunk_precision_of(policy)
    line = (f"[CompileRegions] parity PASS ({'fp32' if prec == 'fp32' else prec + ' trunk: the bf16 rule'}) — "
            + " | ".join(rules))
    say(line)
    if mem.rows:
        say(mem.line())
    return rules


# --------------------------------------------------------------- the gate's MEMORY (gen3_gate_memory_v1)
MiB = float(1 << 20)


class GateMemory:
    """WHERE R1's gate / canary puts the card (gen3_gate_memory_v1, 2026-10-15): after each step of a rung (the
    relocation forward, the compiled arm, the eager arm, the fp32 reference arm) the allocated bytes and the allocated
    HIGH-WATER mark so far — so a step that RETAINS memory (allocated stays up) and the step that set the peak (the
    high-water rises during it) are both read off one line. It never resets the allocator's peak counters (the CUDA
    ledger and the update-fit check read them across the gate). A no-op off CUDA."""

    def __init__(self, model: Any) -> None:
        dev = ct.resolve_device(model.policy.features_extractor)
        self.device = dev if dev.type == "cuda" and torch.cuda.is_available() else None
        self.rows: List[Tuple[str, float, float, float]] = []    # (step, allocated, high-water, reserved) MiB
        self.start = (0.0, 0.0)
        if self.device is not None:
            torch.cuda.synchronize(self.device)
            self.start = (torch.cuda.memory_allocated(self.device) / MiB,
                          torch.cuda.max_memory_allocated(self.device) / MiB)

    def mark(self, step: str) -> None:
        if self.device is None:
            return
        torch.cuda.synchronize(self.device)
        self.rows.append((step, torch.cuda.memory_allocated(self.device) / MiB,
                          torch.cuda.max_memory_allocated(self.device) / MiB,
                          torch.cuda.memory_reserved(self.device) / MiB))

    def peak_above_start(self) -> float:
        """The gate's transient: its allocated high-water above what was allocated when it started (MiB)."""
        return max([self.start[1]] + [r[2] for r in self.rows]) - self.start[0]

    def line(self) -> str:
        a0, h0 = self.start
        steps = ", ".join(f"{s} {a:,.0f}/{h:,.0f}" for s, a, h, _ in self.rows)
        return (f"🧮 [CompileRegions] gate memory (MiB, allocated after / high-water): start {a0:,.0f}/{h0:,.0f}; "
                f"{steps}; the gate's transient above its start {self.peak_above_start():,.0f} MiB "
                f"(gen3_gate_memory_v1)")


_GATE_MEMORY: Optional[GateMemory] = None


def _mem(step: str) -> None:
    if _GATE_MEMORY is not None:
        _GATE_MEMORY.mark(step)


@contextlib.contextmanager
def gate_memory(mem: GateMemory) -> Iterator[GateMemory]:
    """Record the R1 rungs run inside the block into ``mem`` (`GateMemory`)."""
    global _GATE_MEMORY
    prev, _GATE_MEMORY = _GATE_MEMORY, mem
    try:
        yield mem
    finally:
        _GATE_MEMORY = prev


def release_gate_memory(model: Any) -> int:
    """gen3_gate_releases_memory_v1: drop the autograd graph a gate / canary forward left STASHED on the policy
    (`consistency.release_autograd_stashes`: every stash keeps its value, detached; the next forward overwrites it).
    Returns how many stashes held a graph. The allocator's CACHED blocks are released by the caller that owns the
    card's story (`compile_trainer.arm_compile_sentinel`, before the update-fit check measures)."""
    from agents.training.rust_rollout.consistency import release_autograd_stashes
    return release_autograd_stashes(model.policy)


def _r1_accum_rule(model: Any, names: List[str], regime: str, batch_size: int) -> str:
    """`_r1_accum_once`, re-run ONCE at M' when a YARDSTICK arm broke the margin on the first read (as `_r1_judge`;
    gen3_r1_yardstick_margin_v1)."""
    margin: Optional[float] = None
    events: List[Dict[str, Any]] = []
    while True:
        try:
            return _r1_accum_once(model, names, regime, batch_size, margin, events)
        except _YardstickOutOfMargin as y:
            events.append(y.event)
            margin = float(y.event["rerun_margin"])
            release_gate_memory(model)


def _r1_accum_once(model: Any, names: List[str], regime: str, batch_size: int, margin: Optional[float],
                   events: List[Dict[str, Any]]) -> str:
    """The GRAPHED route's ACCUMULATION rung (`gen3_r1_graphed_route_v1`): two micro-batches (the golden
    rows' slices 0 and 1 — different rows, so the two gradients differ) through the graphed route with
    ONE `.grad` accumulating across both, against the same through eager, at ``regime``'s bar (the
    unmoved parameters at the fresh bar, as `r1_rungs`' live rung). A `.grad` that aliased the first
    replay's pool output would read 2·g₂ for g₁ + g₂ — an O(1) error no bar passes. Deterministic. Arms
    independent by the route counters, as `_r1_pair`."""
    from agents.training.instrumented_ppo.micro_step import micro_step
    g = getattr(model, "_compiled_micro_step_graphed", None)
    if g is None:
        raise ct.GateArmsNotIndependentError(
            "--learner-cudagraphs on but no graphed R1 route is installed — mis-wired "
            "(gen3_r1_graphed_route_v1)")
    fixes = [relocate_clip_boundary(model, _r1_args(model, r1_batch(model, int(batch_size), s)), margin)
             for s in (0, 1)]
    args_list = [a for a, _ in fixes]
    # the accumulated arms' log-probs are the two micro-batches' concatenated: so is the fixture's record
    parts = [x for _, x in fixes]
    fix = ClipRelocation(torch.cat([x.old_log_prob for x in parts]), torch.cat([x.moved for x in parts]),
                         torch.cat([x.ref_log_prob for x in parts]), torch.cat([x.advantages for x in parts]),
                         parts[0].clip_range, parts[0].margin, min(x.nearest_unmoved for x in parts),
                         _rerun_source(events[0]) if events else parts[0].shift_source,
                         torch.cat([_raw_old(x) for x in parts]))
    tail = "".join("; " + yardstick_line(e) for e in events)
    e0, g0 = int(RC.EAGER_BODY["R1"]), int(RC.peek().get("R1_graphed", 0))
    graphed = _r1_accum_arm(model, g, args_list)
    e1, g1 = int(RC.EAGER_BODY["R1"]), int(RC.peek().get("R1_graphed", 0))
    eager = _r1_accum_arm(model, micro_step, args_list)
    e2, g2 = int(RC.EAGER_BODY["R1"]), int(RC.peek().get("R1_graphed", 0))
    if (e1 - e0, g1 - g0) != (0, 2) or (e2 - e1, g2 - g1) != (2, 0):
        raise ct.GateArmsNotIndependentError(
            f"--compile-trainer region R1[graphed]: the accumulation rung's arms are NOT independent — "
            f"graphed arm eager body {e1 - e0}x / graphed route {g1 - g0}x (want 0 / 2), eager arm "
            f"{e2 - e1}x / {g2 - g1}x (want 2 / 0) (gen3_gate_independent_arms_v1)")
    unmoved = unmoved_parameters(model) if regime != "fresh" else []
    idx = {n: i for i, n in enumerate(names)}
    excepted = {idx[n]: R1_PARAM_BAR["fresh"] for n in unmoved}
    label = f"{regime} weights, 2 micro-batches ACCUMULATED into one .grad (golden slices 0 + 1)"
    from agents.model.trunk_precision import fp32_scope, trunk_precision_of
    if trunk_precision_of(model.policy) == "bf16":
        # the bf16 rule (`_r1_verdict_bf16`): the same two micro-batches accumulated through the fp32 EAGER
        # reference, proved eager by the counters like the rung's eager arm
        with fp32_scope(model.policy):
            ref = _r1_accum_arm(model, micro_step, args_list)
        e3, g3 = int(RC.EAGER_BODY["R1"]), int(RC.peek().get("R1_graphed", 0))
        if (e3 - e2, g3 - g2) != (2, 0):
            raise ct.GateArmsNotIndependentError(
                f"--compile-trainer region R1[graphed] (bf16 rule): the accumulation rung's fp32 REFERENCE arm ran "
                f"the eager body {e3 - e2}x / the graphed route {g3 - g2}x (want 2 / 0)")
        m_line = _clip_margin_rule(fix, {"graphed": graphed, "eager": eager, "fp32 reference": ref},
                                   rerun=not events)
        gl = golden_row_losses({"ref": ref, "eager": eager, "comp": graphed}, fix, args_list)
        return "R1[graphed] " + _r1_verdict_bf16(eager, graphed, ref, names, regime, label=label,
                                                 bar_by_index=excepted,
                                                 losses=(gl["ref"], gl["eager"], gl["comp"])) + "; " + m_line + tail
    m_line = _clip_margin_rule(fix, {"graphed": graphed, "eager": eager}, rerun=not events)
    return ("R1[graphed] " + _r1_verdict(eager, graphed, names, regime, bar_by_index=excepted, label=label)
            + "; " + m_line + tail)


# ------------------------------------------------------------- R1's judged set vs the golden's own
#: The band around the per-parameter floor (`compile_trainer._PARAM_GRAD_FLOOR`, relative to the
#: largest per-parameter gradient norm) inside which a parameter is NOT compared
#: (`gen3_r1_judged_set_v1`, P10-C): the golden rows must judge it CLEARLY (ratio > band x floor) for
#: its absence from this run's judged set (ratio <= floor) to count, so only a >= band-fold drop in
#: its relative gradient can trip the check — never the rounding between two arithmetically equal
#: expressions (the fork mask at 1.0 is `sum / n` where the golden's is `mean`). Measured on the
#: production learner at the golden's seed (CPU, B = 256): the smallest judged ratio was 1.01e-3,
#: one hair above the floor — exactly the parameter a bandless comparison would flip on.
JUDGED_SET_BAND = 2.0


def _grad_ratios(arm: Dict[str, torch.Tensor]) -> List[float]:
    sizes = [int(x) for x in arm["grad_sizes"].tolist()]
    norms = [float(g.norm()) for g in torch.split(arm["grad"].float(), sizes)]
    top = max(norms) if norms else 0.0
    return [n / top if top > 0.0 else 0.0 for n in norms]


def judged_set_shrinkage(reference: Dict[str, torch.Tensor], run: Dict[str, torch.Tensor],
                         names: List[str], *, floor: Optional[float] = None,
                         band: float = JUDGED_SET_BAND) -> Tuple[List[str], int, int, int]:
    """``(missing, clear, judged, banded)``: the parameters the REFERENCE arm judges clearly
    (gradient-norm ratio > ``band`` x ``floor``) that the RUN arm does not judge at all (ratio <=
    ``floor``, the per-parameter rule's own skip); how many the reference judges clearly; how many
    the run judges; how many the reference judges inside the band (not compared). Eager arms of
    `_r1_arm`."""
    fl = float(ct._PARAM_GRAD_FLOOR if floor is None else floor)
    ref, cur = _grad_ratios(reference), _grad_ratios(run)
    if len(ref) != len(cur):
        raise ct.CompileTrainerError("--compile-trainer region R1: the golden-alone reference and "
                                     "this run's arm cover different parameter sets — mis-wired")
    clear = [i for i, r in enumerate(ref) if r > band * fl]
    missing = [names[i] if i < len(names) else f"#{i}" for i in clear if not cur[i] > fl]
    banded = sum(1 for r in ref if fl < r <= band * fl)
    return missing, len(clear), sum(1 for r in cur if r > fl), banded


def _r1_reference_args(model: Any, b: R1Batch) -> Tuple[Any, ...]:
    """R1's arguments on the golden rows ALONE: the filled keys removed from the batch, and the
    static resolved as `train()` resolves it from a buffer that does not declare them (the fold's
    predicates read the BUFFER's key set — the fork mask's is `fork_pg_m in observations`)."""
    keys = set(b.filled)
    buf = model.rollout_buffer
    saved = buf.observations
    try:
        if isinstance(saved, dict):
            buf.observations = {k: v for k, v in saved.items() if k not in keys}
        st = _r1_static(model)
    finally:
        buf.observations = saved
    obs = {k: v for k, v in b.obs.items() if k not in keys}
    return (model.policy, obs, b.actions, b.action_masks,
            b.old_log_prob, b.old_values, _r1_advantages(model, b), b.returns, st)


def _r1_judged_set_rule(model: Any, b: R1Batch, run: Dict[str, torch.Tensor], names: List[str]) -> str:
    """REFUSE when filling ``b.filled`` shrank R1's judged parameter set below the golden rows' own
    (`judged_set_shrinkage`): a placeholder that blanks a term would leave its parameters unjudged —
    the P10-C defect class. One extra EAGER micro-step, only on a run whose space the golden lacks a
    key of (never on the production surface). Returns the gate-line rule."""
    from agents.training.instrumented_ppo.micro_step import micro_step
    try:
        # the golden rows alone, relocated off the clip boundary like the run's arm (`relocate_clip_boundary`), so the
        # two judged sets differ only by the filled keys
        ref = _r1_arm(model, micro_step, relocate_clip_boundary(model, _r1_reference_args(model, b))[0])
    except ct.CompileTrainerError:
        raise
    except Exception as exc:
        raise ct.CompileTrainerError(
            f"--compile-trainer region R1: cannot judge the golden rows ALONE (without the filled "
            f"key(s) {list(b.filled)}) to compare R1's judged parameter set — {type(exc).__name__}: "
            f"{exc}. A filled key whose term the fold does not gate on the buffer's key set needs its "
            f"own reference (P10-C)") from exc
    missing, clear, judged, banded = judged_set_shrinkage(ref, run, names)
    if missing:
        raise ct.CompileTrainerError(
            f"--compile-trainer region R1: filling {list(b.filled)} with the declared placeholder "
            f"SHRANK the judged parameter set — {len(missing)} parameter(s) the golden rows judge "
            f"clearly (> {JUDGED_SET_BAND:g}x the floor) are not judged on this run's rows (e.g. "
            f"{', '.join(missing[:8])}). The startup gate and the canary would pass a backward "
            f"miscompile there unseen: refusing (P10-C, gen3_r1_judged_set_v1). Fix the placeholder "
            f"(it must not blank a term) or rebuild the golden buffer with the key.")
    return (f"judged set >= the golden rows' own: all {clear} parameters they judge clearly "
            f"(> {JUDGED_SET_BAND:g}x floor) are judged here ({judged} judged; {banded} inside the "
            f"band, not compared) [filled {list(b.filled)}]")


# ------------------------------------------------------------------ R1's gradient bar, by regime
#: R1's PER-PARAMETER gradient bar, BY WEIGHT REGIME (`gen3_r1_param_bar_by_regime_v1`, 2026-10-01).
#: The extractor gate's 1e-3 / 0.2 (`compile_trainer._MAX_PARAM_GRAD_REL[_TRAINED]`) were measured on
#: its PROBE loss over 64 fixture rows; R1 is the real PPO micro-step over the production micro-batch,
#: and its healthy compiled-vs-eager disagreement is far larger and comes from EITHER side. Under the
#: 1e-3 bar every fp32 resume of arm C (2.47e-3) and every fresh fp32 launch at B = 2048 (1.0e-2 to
#: 2.5e-2) FATAL'd at startup. MEASURED by a matched-noise control
#: (`designs/research_state/measurements/k6_k8/r1_noise/`; torch 2.8, CUDA, fp32 'highest', B = 2048
#: of the gate's own rows, 4 fresh seeds + 3 perturbed-fresh + 18 trained checkpoints): against a
#: float64 eager reference, the compiled gradient was never more than 2x the worse of CUDA eager's and
#: CPU eager's own fp32 error, except the 17 hidden-opp-belief decoder parameters of one state, at
#: <= 5.5e-4 (README); the largest gate reading per regime is `R1_HEALTHY_MAX`, and on arm C's final
#: weights it is CUDA EAGER's own error (2.47e-3 vs float64; the compiled gradient 1.7e-5).
#: Bar = `R1_BAR_K` x that maximum. A 10% backward error on a path reads ~0.1, so the TRAINED bar
#: refuses it with 10x margin; the FRESH bar does not, which is why a fresh launch ALSO runs R1 on a
#: seeded perturbation at the trained bar (`_r1_perturbed`; perturbed-fresh weights read <= 2.3e-5).
R1_HEALTHY_MAX = {"fresh": 2.51e-2, "trained": 2.47e-3}
R1_BAR_K = 4.0
R1_PARAM_BAR = {k: R1_BAR_K * v for k, v in R1_HEALTHY_MAX.items()}


#: The model attribute holding the FRESH BUILD's per-parameter init fingerprints
#: (`gen3_r1_unmoved_init_v1`): ``{policy parameter name: param_sha256(value at the fresh build)}``.
#: Plain JSON data, so SB3's ``save`` writes it into every checkpoint's ``data`` and ``load`` restores it
#: — it travels with the WEIGHTS (restart, resume, fork), and nothing ever recomputes it after a load.
INIT_RECORD_ATTR = "param_init_sha256"


def param_sha256(p: torch.Tensor) -> str:
    """sha256 over a tensor's dtype, shape and raw bytes (device-independent: hashed from a CPU copy, one
    parameter at a time). Equal digests <=> bit-identical values (``-0.0`` and ``0.0`` differ)."""
    t = p.detach().cpu().contiguous().reshape(-1)          # host copy FIRST: no device allocation
    h = hashlib.sha256(f"{t.dtype}|{tuple(p.shape)}|".encode())
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@startup_builder
def record_param_init(model: Any) -> Dict[str, str]:
    """Record the FRESH BUILD's init fingerprint of every policy parameter on ``model``
    (`INIT_RECORD_ATTR`) and return it. Called ONCE, by the trainer's fresh construction
    (`main.train.model_build.construct_fresh_learner`), before anything trains — a declared startup
    acquisition (K6), never a lazy one. CPU only (Python strings: ~0.1 KB per parameter; the GPU is not
    touched beyond a transient per-parameter device-to-host copy)."""
    rec = {n: param_sha256(p) for n, p in model.policy.named_parameters()}
    setattr(model, INIT_RECORD_ATTR, rec)
    return rec


def init_record(model: Any) -> Optional[Dict[str, str]]:
    """The run's init record, or None when its checkpoint carries none (saved before
    `gen3_r1_unmoved_init_v1`, or built outside the trainer's fresh construction)."""
    rec = getattr(model, INIT_RECORD_ATTR, None)
    return rec if isinstance(rec, dict) and rec else None


def unmoved_rule(model: Any) -> str:
    """Which rule `unmoved_parameters` applies to ``model`` — named on every rule line it shapes."""
    rec = init_record(model)
    return ("bit-identical to the init record, or exactly 0.0" if rec is not None
            else "exactly 0.0 only: no init record in this checkpoint")


def unmoved_parameters(model: Any) -> List[str]:
    """The judged parameters (`compile_trainer.grad_parameters`) training has NEVER MOVED
    (`gen3_r1_unmoved_param_v1`, generalised by `gen3_r1_unmoved_init_v1`): a parameter is UNMOVED iff

      * every element is BIT-IDENTICAL to its value at the run's fresh build — its digest equals the
        init record's (`INIT_RECORD_ATTR`, written by `record_param_init`, carried in the checkpoint), or
      * every element is exactly 0.0 (the zero rule — the only rule on a checkpoint with no init record).

    WHY. `weights_regime` classifies the MODEL; gradient CONDITIONING is a property of each parameter.
    A head training has never moved still has FRESH-weights conditioning when the rest of the model is
    trained, and the trained bar then reads its healthy fp32 noise as a miscompile. The case that
    FATAL'd (2026-10-04, `rb_x5ab_oracle_sp_s1001`, update 10): under `--oracle-reveal` the blob arm's
    species belief labels are all PAD, so `belief_head` receives NO gradient in training — its
    zero-init `species_head` stays exactly 0.0, and so do its ortho-init `moves_head.weight`, its
    LayerNorm and `belief_slots.unknown_slot_emb` (bit-identical to the fresh build at seed 1001;
    `designs/research_state/measurements/oracle_canary_2026-10-04/`) — while the gate's golden rows
    still supervise it. The species head's reading, 1.40e-2, was the FRESH regime's own.

    DETERMINISTIC: both tests are categorical (digest equality; every element == 0.0) — no tolerance,
    so no input sits within a rounding error of the rule's boundary. A parameter training has moved by
    any amount (weight decay included) is judged at the model's regime, as before. The zero rule
    stays with a record: an all-zero parameter has the fresh conditioning whatever its history."""
    rec = init_record(model) or {}
    out: List[str] = []
    for n, p in ct.grad_parameters(model, model.policy.features_extractor):
        if p.numel() == 0:
            continue
        if not bool(p.detach().any()) or (n in rec and param_sha256(p) == rec[n]):
            out.append(n)
    return out


def r1_rungs(model: Any, args: Tuple[Any, ...], names: List[str], regime: str, *,
             source: Optional[str] = None, fn: Optional[Callable[..., Any]] = None,
             route: Optional[str] = None, watch: Optional["GradWatch"] = None
             ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[str]]:
    """Region R1's verdict rungs, shared by the startup gate and the in-run canary. Returns the live
    rung's EAGER and COMPILED arms (the judged-set rule and the canary's cosine read them) and one
    rule line per rung; raises `CompileTrainerError` on a disagreement.

      1. the LIVE weights at ``regime``'s bar — except the UNMOVED parameters (`unmoved_parameters`:
         bit-identical to the init record, or exactly 0.0), which are judged at the FRESH bar there
         (their conditioning is fresh whatever the model's regime);
      2. FRESH regime: every parameter perturbed (`_r1_perturbed`), judged at the TRAINED bar;
      3. TRAINED regime with unmoved parameters: ONLY those perturbed off their init, by name-keyed seeded
         noise (`parity_probe.perturbed_parameters(only=…)`, restored bit-exactly), and EVERY
         parameter judged at the TRAINED bar (`_r1_unmoved_perturbed`) — so the unmoved parameters'
         backward paths are still held to the tight bar, on weights where their gradient is
         well-conditioned. A real miscompile there is caught by this rung.

    ``fn`` / ``route``: the R1 route under judgement (`r1_routes`; None = the installed compiled route,
    rule lines prefixed ``R1``; a named route prefixes ``R1[<route>]``). ``watch``: a `GradWatch` the LIVE rung fills
    with its watch-list's per-parameter numbers (the canary's growth-curve record), also when the rung refuses.
    """
    unmoved = unmoved_parameters(model) if regime != "fresh" else []
    pre = "R1 " if route is None else f"R1[{route}] "
    idx = {n: i for i, n in enumerate(names)}
    excepted = {idx[n]: R1_PARAM_BAR["fresh"] for n in unmoved}
    tail = f" [{source}]" if source else ""
    if unmoved:
        tail += f" [unmoved rule: {unmoved_rule(model)}]"
    eager, comp, line = _r1_judge(model, args, names, regime, fn, bar_by_index=excepted, watch=watch)
    rules = [pre + line + tail]
    if regime == "fresh":
        rules.append(pre + _r1_perturbed(model, args, names, fn))
    elif unmoved:
        rules.append(pre + _r1_unmoved_perturbed(model, args, names, unmoved, fn))
    return eager, comp, rules


def _r1_judge(model: Any, args: Tuple[Any, ...], names: List[str], regime: str,
              fn: Optional[Callable[..., Any]] = None, *,
              label: Optional[str] = None, bar_by_index: Optional[Dict[int, float]] = None,
              watch: Optional["GradWatch"] = None
              ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
    """ONE rung: the independent compiled / eager pair on the CURRENT weights, judged by the rule of the
    policy's trunk precision — fp32: `_r1_verdict` (compiled vs eager at the fp32 bars); bf16: THE bf16 PARITY
    RULE (`_r1_verdict_bf16`, with the fp32 eager REFERENCE arm `_r1_reference`). Returns (eager, compiled,
    the rule line). ``fn``: the R1 route under judgement (`r1_routes`; None = the installed compiled route).

    A YARDSTICK arm out of margin on the first read (`YARDSTICK_ARMS`, gen3_r1_yardstick_margin_v1) is recorded as an
    event (``watch``, the rule line) and the WHOLE rung re-runs ONCE at M' (`_clip_margin_rule`), judged there: the
    first read's verdict is discarded (its gradient may carry the 0/1 clip switch the margin exists to exclude)."""
    from agents.model.trunk_precision import trunk_precision_of
    margin: Optional[float] = None
    events: List[Dict[str, Any]] = []
    raw_args = args
    while True:
        # rule 8: the rows off the PPO clip boundary (T36), at the model's margin or the yardstick re-run's M'
        args, fix = relocate_clip_boundary(model, raw_args, margin)
        if events:
            fix = fix._replace(shift_source=_rerun_source(events[0]))
        comp, eager = _r1_pair(model, args, fn)
        arms: Dict[str, Dict[str, torch.Tensor]] = {"compiled": comp, "eager": eager}
        line, err = "", None
        try:
            if trunk_precision_of(model.policy) == "bf16":
                ref = _r1_reference(model, args)
                arms["fp32 reference"] = ref
                gl = golden_row_losses({"ref": ref, "eager": eager, "comp": comp}, fix, [args])
                line = _r1_verdict_bf16(eager, comp, ref, names, regime, label=label, bar_by_index=bar_by_index,
                                        watch=watch, losses=(gl["ref"], gl["eager"], gl["comp"]))
            else:
                line = _r1_verdict(eager, comp, names, regime, label=label, bar_by_index=bar_by_index, watch=watch)
        except (ct.NonFiniteGateArmError, ct.GateArmsNotIndependentError):
            raise                                             # named precisely already
        except ct.CompileTrainerError as exc:
            err = exc
        # the margin's precondition is read whatever the verdict (the watch record is complete), and an arm that broke
        # it is named as ITSELF — its gradient verdict would be the 0/1 switch this fixture exists to exclude
        try:
            m_line = _clip_margin_rule(fix, arms, watch, rerun=not events)
        except _YardstickOutOfMargin as y:
            events.append(y.event)
            if watch is not None:
                watch.add_yardstick(y.event)
            margin = float(y.event["rerun_margin"])
            del comp, eager, arms
            release_gate_memory(model)
            continue
        except ct.CompileTrainerError as m_err:
            if err is not None:
                raise m_err from err                          # the verdict's own refusal rides as the cause
            raise
        if err is not None:
            raise err
        return eager, comp, line + "; " + m_line + "".join("; " + yardstick_line(e) for e in events)


def _r1_unmoved_perturbed(model: Any, args: Tuple[Any, ...], names: List[str],
                          unmoved: List[str], fn: Optional[Callable[..., Any]] = None) -> str:
    """TRAINED weights with unmoved parameters: R1 again with ONLY those parameters moved off their
    init (the declared ladder's first rung, name-keyed noise ADDED to the live value; every other
    parameter at its live value), EVERY parameter judged at the TRAINED bar
    (`gen3_r1_unmoved_param_v1`, `gen3_r1_unmoved_init_v1`)."""
    from agents.model.parity_probe import PERTURB_LADDER, perturbed_parameters, rung_seed
    scale, k = PERTURB_LADDER[0]
    policy_names = [n for n, _ in model.policy.named_parameters()]
    only = [n for n in unmoved if n in policy_names]
    if len(only) != len(unmoved):
        raise ct.CompileTrainerError("--compile-trainer region R1: an unmoved parameter is not a "
                                     "parameter of the policy — mis-wired")
    with perturbed_parameters(model.policy, seed=rung_seed(k), scale=scale, only=only):
        _, _, line = _r1_judge(model, args, names, "trained", fn,
                               label=(f"trained weights, the {len(only)} unmoved param(s) perturbed "
                                      f"scale={scale:g} seed+{k} (e.g. {only[0]})"))
    return line


def weights_regime(model: Any, rows: int = 8) -> str:
    """``"fresh"`` when the policy's legal log-probs on the committed real-obs fixture are constant
    within every row — the zero-init pointer head of a fresh launch — else ``"trained"`` (a resume,
    a fork, a crash-restart, the canary at update N). One eager no-grad forward; no compile."""
    from agents.model.parity_probe import vacuous_keys
    policy = model.policy
    was = policy.training
    obs = ct._prewarm_obs(model, int(rows))
    _, mask = ct._parity_obs(int(obs["observation"].shape[-1]), int(rows),
                             ct.resolve_device(policy.features_extractor))
    policy.set_training_mode(False)
    from agents.model.trunk_precision import fp32_scope
    try:
        with fp32_scope(policy):              # the regime is a property of the WEIGHTS, read at the fp32 function
            e = _decision_readout(model, obs, mask)
    finally:
        policy.set_training_mode(was)
    bar = {"legal_logprob": ct._FP32_TOL["legal_logprob"]}
    return "fresh" if vacuous_keys({"legal_logprob": e["legal_logprob"]}, bar) else "trained"


def _r1_perturbed(model: Any, args: Tuple[Any, ...], names: List[str],
                  fn: Optional[Callable[..., Any]] = None) -> str:
    """FRESH weights only: R1 again on the declared ladder's first rung — a seeded perturbation of
    every policy parameter, in place and restored bit-exactly (`parity_probe.perturbed_parameters`)
    — judged at the TRAINED bar. On fresh weights a few ill-conditioned gradients (the uniform belief
    heads) need the looser fresh bar, under which a backward defect could hide; the perturbed weights
    are trained-like and get the tight one (the deleted extractor gate's fresh-weights rule)."""
    from agents.model.parity_probe import PERTURB_LADDER, perturbed_parameters, rung_seed
    scale, k = PERTURB_LADDER[0]
    with perturbed_parameters(model.policy, seed=rung_seed(k), scale=scale):
        _, _, line = _r1_judge(model, args, names, "trained", fn,
                               label=f"fresh weights, seeded perturbation scale={scale:g} seed+{k}")
    return line


def _r1_verdict(eager: Dict[str, torch.Tensor], comp: Dict[str, torch.Tensor], names: List[str],
                regime: str, label: Optional[str] = None,
                bar_by_index: Optional[Dict[int, float]] = None, watch: Optional["GradWatch"] = None) -> str:
    if watch is not None:
        watch.fill_fp32(eager, comp, names, regime, bar_by_index)
    e_loss, c_loss = float(eager["loss"]), float(comp["loss"])
    if not (np.isfinite(e_loss) and np.isfinite(c_loss)):
        raise ct.CompileTrainerError(f"--compile-trainer region R1: non-finite loss (eager {e_loss}, "
                                     f"compiled {c_loss})")
    rel = abs(c_loss - e_loss) / max(1.0, abs(e_loss))
    if not rel <= ct._MAX_NUMERIC_DRIFT:
        raise ct.CompileTrainerError(
            f"--compile-trainer region R1: the compiled micro-step's LOSS disagrees with eager — "
            f"{c_loss!r} vs {e_loss!r} (rel {rel:.2e} > {ct._MAX_NUMERIC_DRIFT:g})")
    loss_rule = f"loss rel {rel:.2e} <= {ct._MAX_NUMERIC_DRIFT:g}"
    eager_t = {"features": eager["loss"], "grad": eager["grad"], "grad_sizes": eager["grad_sizes"]}
    comp_t = {"features": comp["loss"], "grad": comp["grad"], "grad_sizes": comp["grad_sizes"]}
    grad_rule = ct.train_verdict(eager=eager_t, compiled=comp_t, allow_vacuous=True,
                                 param_names=names, param_bar=R1_PARAM_BAR[regime],
                                 param_bar_by_index=bar_by_index)
    return f"[{label or regime + ' weights'}] {loss_rule}; {grad_rule}"


# --------------------------------------------- R1's FIXTURE off the PPO CLIP BOUNDARY (standing rule 8, T36)
#: THE DEFECT IT REMOVES (`gen3_r1_clip_margin_v1`, T36; the 2026-10-15 `rb_toy_ent_s4001` canary FATAL,
#: `designs/training/compile_flags.md` "The bf16 canary FATAL on trained weights"). R1 is judged on the K9 golden's
#: committed rows, whose OLD log-probs are a near-uniform policy's; on trained weights their PPO ratios spread
#: 0.002 … 6.9. The clipped surrogate SWITCHES a row's whole policy gradient on or off at the clip boundary that bites
#: its advantage's sign (A > 0 at ρ = 1 + clip, A < 0 at ρ = 1 − clip; the other boundary never switches it). A row
#: within rounding of that boundary carries its full gradient in one arm and none in the other, so the per-parameter
#: statistic becomes a 0/1 lottery: two HEALTHY eager bf16 arms read the canary's ratio above its bar of 12 in 18 % of
#: 992 dithered pairs (max 31x) at the FATAL's weights. A verdict is never allowed to depend on that (rule 8).
#:
#: THE FIX: before judging, every row whose fp32 REFERENCE ratio sits within `R1_CLIP_MARGIN` (log-prob units) of its
#: switching boundary gets its OLD log-prob moved so the reference sits 2 x the margin INSIDE the ACTIVE side
#: (`clip_relocation`). Every rung of the startup gate and of the canary does it on its own weights
#: (`relocate_clip_boundary`, in `_r1_judge` and the graphed accumulation rung). MOVED, not excluded: the batch keeps
#: its declared shape and every row (excluding one would change the advantage normalisation and drop that row's
#: whole policy gradient from the check), each moved row's gradient is judged IN FULL by every arm, and a row far from
#: its boundary keeps the golden's label. Deterministic: a function of the reference's own forward on the same rows.
#: The relocation's own threshold is not a verdict boundary: a row just inside it is moved to 2 M, a row just outside
#: stays at >= M, and either way no arm whose shift is <= M can switch it.
#:
#: THE PRECONDITION, checked on every rung (`_clip_margin_rule`): every judged arm's largest |Δ log π| from the
#: reference is <= the margin. Then no row can switch (moved rows sit 2 M away, the others > M), and that is a
#: CATEGORICAL statement; a COMPILED arm that moves a log-prob farther is refused by name (a forward disagreement of
#: >= 4x the measured healthy maximum — never a lottery read through a gradient ratio). A YARDSTICK arm that does is
#: an event and the rung re-runs at M' (`YARDSTICK_ARMS`, gen3_r1_yardstick_margin_v1, below).
#:
#: The largest |Δ log π| (the taken action's log-prob) a JUDGED arm has been measured to move a fixture row from the
#: fp32 eager reference. PROVISIONAL — CPU and EAGER only (2026-10-15): eager bf16 vs eager fp32, max over (a) the
#: `rb_toy_ent_s4001` census (62 checkpoints 1M-62M, B = 64 untiled, `main.ops.diag_bf16_canary cpu`) 0.161 (median of
#: the per-checkpoint max 0.063), (b) the production-surface census (10 states) 0.055, (c) the 32-seed dithered null at
#: the FATAL's weights (`compile_regions_clip_margin_fatal_test`) 0.173 — rounded UP to 2 significant digits as the
#: bf16 bars are; fp32 arms ~1e-6. The COMPILED bf16 arm's shift on CUDA is read by the GPU diagnostic
#: (`forward_vs_E32.Cbf.log_prob_max_abs`) and printed on every R1 rule line (`clip margin: … max |dlogp|`): if it
#: exceeds this value, raise it here (and the margin with it) — for a LIVE run, whose pin cannot move, `--canary-shift-max`
#: (T37, below) carries the raise through a same-run resume.
R1_LOGPROB_SHIFT_MAX = 0.18
#: THE MARGIN = R1_BAR_K (the gate's safety factor over a healthy maximum, as every R1 bar) x that shift = 0.72.
R1_CLIP_MARGIN = R1_BAR_K * R1_LOGPROB_SHIFT_MAX


#: THE CHECK-ONLY OVERRIDE (`--canary-shift-max`, T37, `gen3_canary_shift_max_v1`; `designs/endstate/design_x26_readiness.md`
#: §4.1 item 2, §4.2, F28). A LIVE run's code pin cannot move at a restart (`worktree.resolve_pin` refuses it), so a
#: raised `R1_LOGPROB_SHIFT_MAX` in code can reach a run only through a FORK; this flag carries the same number through
#: a same-run RESUME instead. It is the margin's INPUT and nothing else: it moves where the gate relocates its fixture
#: rows and where it refuses an arm, never a bar of the parity rule, never the training update (`canary_shift_max_test`
#: holds one update bit-identical at two values). It can only RAISE the shift above `R1_LOGPROB_SHIFT_MAX`: a lower
#: margin has no measured basis (0.18 is the largest healthy shift measured), and `resolve_shift_max` refuses one.
SHIFT_MAX_FLAG = "--canary-shift-max"


class CanaryShiftMaxError(ValueError):
    """A `--canary-shift-max` value the gate cannot honour: not a finite number, or BELOW `R1_LOGPROB_SHIFT_MAX` (the
    flag only raises it; a lower bar has no measured basis). Typed so the parser, `resolve_config` and the gate
    refuse it with one reason."""


def resolve_shift_max(value: Any) -> float:
    """The shift (log-prob) the margin is built from: ``None`` is the code's `R1_LOGPROB_SHIFT_MAX`; a typed value
    is returned as a float if it is finite and at least that constant, else `CanaryShiftMaxError` naming why."""
    if value is None:
        return float(R1_LOGPROB_SHIFT_MAX)
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise CanaryShiftMaxError(f"{SHIFT_MAX_FLAG} {value!r} is not a number") from None
    if not math.isfinite(v):
        raise CanaryShiftMaxError(f"{SHIFT_MAX_FLAG} {value!r} is not finite")
    if v < R1_LOGPROB_SHIFT_MAX:
        raise CanaryShiftMaxError(
            f"{SHIFT_MAX_FLAG} {v:g} is BELOW the code's R1_LOGPROB_SHIFT_MAX {R1_LOGPROB_SHIFT_MAX:g}, the largest "
            f"healthy shift measured: the flag only RAISES the margin (4 x it) — a lower bar has no measured basis, "
            f"and it would make the gate refuse arms the measurements call healthy")
    return v


def shift_max_of(model: Any) -> float:
    """The effective shift on ``model``: its `_canary_shift_max` (set by `apply_training_hparams` from the RESOLVED
    flag, never pickled), else the code's default. A model without the attribute (a duck-typed double, a reader)
    reads the default."""
    return resolve_shift_max(getattr(model, "_canary_shift_max", None))


def shift_max_source(model: Any) -> str:
    """Where `shift_max_of` came from, for the rule line and the canary record."""
    return "default" if getattr(model, "_canary_shift_max", None) is None else SHIFT_MAX_FLAG


def clip_margin_of(model: Any) -> float:
    """The margin on ``model``: `R1_BAR_K` x `shift_max_of(model)` (= `R1_CLIP_MARGIN` unless overridden)."""
    return R1_BAR_K * shift_max_of(model)


class ClipRelocation(NamedTuple):
    """One rung's relocated fixture: the new old log-probs, which batch rows moved, the reference's log-prob and its
    clip-switching side per row (True = the row's policy gradient is ACTIVE), the nearest UNMOVED row's distance."""
    old_log_prob: torch.Tensor
    moved: torch.Tensor                 # bool, per batch row
    ref_log_prob: torch.Tensor          # float64, CPU
    advantages: torch.Tensor            # float64, CPU: the micro-step's own (normalised) advantages
    clip_range: float
    margin: float
    nearest_unmoved: float              # log-prob distance of the closest row left in place (inf when none)
    shift_source: str = "default"       # where the shift behind `margin` came from: "default" | "--canary-shift-max"
    #: the golden's OWN old log-probs before the relocation (float64, CPU): the LOSS comparisons read the policy term
    #: on these (`golden_row_loss`, gen3_r1_loss_on_golden_rows_v1); None = no raw rows recorded (a hand-built record)
    raw_old_log_prob: Optional[torch.Tensor] = None

    @property
    def shift_max(self) -> float:
        """The shift the margin was built from (`margin` = `R1_BAR_K` x it)."""
        return self.margin / R1_BAR_K

    def line(self) -> str:
        n, b = int(self.moved.sum()), int(self.moved.numel())
        near = f"{self.nearest_unmoved:.3f}" if np.isfinite(self.nearest_unmoved) else "-"
        return (f"clip margin: {n}/{b} boundary rows relocated (M {self.margin:.3f} = {R1_BAR_K:g} x shift_max "
                f"{self.shift_max:.4g} [{self.shift_source}]; nearest unmoved {near})")


def _switch_log_bound(adv: torch.Tensor, clip_range: float) -> torch.Tensor:
    """Per row, log of the clip boundary that SWITCHES its policy gradient: log(1 + clip) for A > 0, log(1 - clip)
    otherwise (A = 0 rows carry no policy gradient, so their bound is never read)."""
    return torch.where(adv > 0, torch.full_like(adv, math.log(1.0 + clip_range)),
                       torch.full_like(adv, math.log(1.0 - clip_range)))


def clip_active(log_prob: torch.Tensor, old_log_prob: torch.Tensor, adv: torch.Tensor,
                clip_range: float) -> torch.Tensor:
    """Per row: does the clipped surrogate pass this row's policy gradient? (A > 0: ρ <= 1 + clip; A < 0: ρ >= 1 - clip
    — `clamp` passes its gradient AT its bounds; A = 0: False.) `micro_step`'s ``min(ρA, clip(ρ)A)`` branch, read in
    log space in float64 (the fixture never holds a row within the margin of a bound, so the log-vs-ratio rounding
    cannot decide it)."""
    lr = log_prob.detach().double().cpu().reshape(-1) - old_log_prob.detach().double().cpu().reshape(-1)
    a = adv.detach().double().cpu().reshape(-1)
    logb = _switch_log_bound(a, clip_range)
    return torch.where(a > 0, lr <= logb, torch.where(a < 0, lr >= logb, torch.zeros_like(lr, dtype=torch.bool)))


#: THE LOSS IS READ ON THE GOLDEN'S OWN ROWS (`gen3_r1_loss_on_golden_rows_v1`, 2026-10-17; `designs/training/
#: compile_flags.md` "bf16 drift on a sharpening policy"). The relocation above exists for the GRADIENT, whose policy
#: part switches 0/1 at the clip bound. The LOSS does not switch: min(A rho, A clip(rho)) is CONTINUOUS in rho, so a
#: loss comparison needs no relocation, and the relocation distorts it: it moves an A < 0 row to rho = 0.85 e^(2M)
#: (3.59 at M = 0.72), so that row's loss carries rho |A| x its log-prob shift — 4.2x its weight at the bound, and
#: e^(2 dM) more again under a raised `--canary-shift-max`. `BF16_DRIFT_BARS` / `BF16_ENVELOPE_K` were DERIVED on the
#: golden's own rows (window D, 2026-10-10), before the relocation existed (T36, 2026-10-15). On X26 (`rb_x26_s2601`)
#: one such row (|A| 3.23, the fixture's largest; rho |A| 11.6) carried 88-100 % of the startup gate's loss drift
#: from 116M on (6.68e-3 = 48 % of the bar at the 121M restart; 2 % of the CPU null draws past the bar), while the same
#: weights read on the golden's own rows stay at <= 55 % of it. So every LOSS comparison of the bf16 rule (the DRIFT
#: and the ENVELOPE) reads each arm's loss with its policy term re-evaluated on the golden's OWN old log-probs: the
#: statistic the bars were derived on. Host-side, exact (the policy term is a function of the arm's own log-probs, the
#: normalised advantages and the old log-probs — `micro_step`'s formula, in float64), no extra forward. The GRADIENT
#: comparisons stay on the relocated rows (rule 8).
def _raw_old(fix: "ClipRelocation") -> torch.Tensor:
    if fix.raw_old_log_prob is None:
        raise ct.CompileTrainerError("--compile-trainer region R1: the clip relocation carries no raw old log-probs — "
                                     "the loss comparison cannot be read on the golden's own rows (mis-wired)")
    return fix.raw_old_log_prob.detach().double().cpu().reshape(-1)


def _policy_term(log_prob: torch.Tensor, old: torch.Tensor, adv: torch.Tensor, clip_range: float, coef: float,
                 weights: Optional[torch.Tensor], n_micro: int) -> float:
    """`micro_step`'s policy term (`policy_grad_coef` x -mean(min(A rho, A clip(rho))), the fork mask's weighted mean
    when ``weights``), in float64, summed over ``n_micro`` equal micro-batches (the accumulation rung's loss is the
    SUM of its micro-batches' losses)."""
    lp = log_prob.detach().double().cpu().reshape(-1)
    rho = torch.exp(lp - old)
    per = torch.min(adv * rho, adv * rho.clamp(1.0 - clip_range, 1.0 + clip_range))
    w = torch.ones_like(per) if weights is None else weights.detach().double().cpu().reshape(-1)
    total = 0.0
    for p_, w_ in zip(per.chunk(int(n_micro)), w.chunk(int(n_micro))):
        total += float(-(p_ * w_).sum() / (w_.sum().clamp(min=1.0) if weights is not None else float(p_.numel())))
    return float(coef) * total


def golden_row_loss(arm: Dict[str, torch.Tensor], fix: "ClipRelocation", st: Any,
                    weights: Optional[torch.Tensor] = None, n_micro: int = 1) -> float:
    """``arm``'s loss with its PPO policy term read on the golden's OWN old log-probs instead of the relocated ones
    (the block comment above): L - pg(relocated old) + pg(raw old), each from the arm's own log-probs. Only the policy
    term reads the old log-probs, so this is the loss the same forward would give on the golden's own rows."""
    adv = fix.advantages.detach().double().cpu().reshape(-1)
    raw, moved = _raw_old(fix), fix.old_log_prob.detach().double().cpu().reshape(-1)
    coef = float(getattr(st, "policy_grad_coef", 1.0))
    w = weights if bool(getattr(st, "fork_pg_mask", False)) else None
    lp = arm["log_prob"]
    return (float(arm["loss"].sum()) - _policy_term(lp, moved, adv, fix.clip_range, coef, w, n_micro)
            + _policy_term(lp, raw, adv, fix.clip_range, coef, w, n_micro))


def _pg_weights(args_list: List[Tuple[Any, ...]]) -> Optional[torch.Tensor]:
    """The fork arm's per-row policy-term mask over ``args_list``'s micro-batches (None when the obs lacks it)."""
    from agents.training.fork_arm import PG_MASK_KEY
    ms = [a[1].get(PG_MASK_KEY) for a in args_list]
    return None if any(m is None for m in ms) else torch.cat([m.reshape(-1) for m in ms])


def golden_row_losses(arms: Dict[str, Dict[str, torch.Tensor]], fix: "ClipRelocation",
                      args_list: List[Tuple[Any, ...]]) -> Dict[str, float]:
    """`golden_row_loss` for every arm (one rung's micro-batches ``args_list``, the static last in each)."""
    st = args_list[0][-1]
    w = _pg_weights(args_list) if bool(getattr(st, "fork_pg_mask", False)) else None
    return {k: golden_row_loss(v, fix, st, w, len(args_list)) for k, v in arms.items()}


def clip_relocation(ref_log_prob: torch.Tensor, old_log_prob: torch.Tensor, adv: torch.Tensor, clip_range: float,
                    margin: float = R1_CLIP_MARGIN, shift_source: str = "default") -> ClipRelocation:
    """The pure relocation (module comment above): a row whose reference log-ratio is within ``margin`` of (or AT)
    its switching boundary — ``|log ρ − log b| <= margin`` — and whose advantage is non-zero gets ``old`` = ref − target,
    so its reference log-ratio is ``target`` = log b ∓ 2 margin, inside the ACTIVE side. Every other row keeps ``old``.
    The returned old log-probs are on ``old_log_prob``'s device and dtype."""
    lp = ref_log_prob.detach().double().cpu().reshape(-1)
    o = old_log_prob.detach().double().cpu().reshape(-1)
    a = adv.detach().double().cpu().reshape(-1)
    if not (lp.shape == o.shape == a.shape):
        raise ct.CompileTrainerError(f"--compile-trainer region R1: the clip relocation's inputs disagree in shape "
                                     f"(log-prob {tuple(lp.shape)}, old {tuple(o.shape)}, advantages {tuple(a.shape)})")
    logb = _switch_log_bound(a, float(clip_range))
    dist = ((lp - o) - logb).abs()
    near = dist.le(float(margin)) & a.ne(0)
    target = torch.where(a > 0, logb - 2.0 * float(margin), logb + 2.0 * float(margin))
    new = torch.where(near, lp - target, o)
    rest = dist[~near & a.ne(0)]
    return ClipRelocation(new.to(dtype=old_log_prob.dtype, device=old_log_prob.device).reshape(old_log_prob.shape),
                          near, lp, a, float(clip_range), float(margin),
                          float(rest.min()) if rest.numel() else float("inf"), shift_source, o)


def relocate_clip_boundary(model: Any, args: Tuple[Any, ...], margin: Optional[float] = None
                           ) -> Tuple[Tuple[Any, ...], ClipRelocation]:
    """R1's arguments with the fixture moved off the clip boundary on ``model``'s CURRENT weights, and the record.
    ``margin`` None (every caller in the gate) is the model's own, `clip_margin_of` (`--canary-shift-max`, T37).
    The reference is R1's own eager body at the fp32 base classes (`trunk_precision.fp32_scope`; a no-op on an fp32
    trunk) — one forward, gradient mode as the arms run it (nothing saved for a backward that never runs), no backward;
    it runs R1's eager body once, outside every arm's independence window. ``args`` is `_r1_args`' tuple (old log-probs
    at index 4, the static last)."""
    from agents.model.trunk_precision import fp32_scope
    from agents.training.instrumented_ppo.micro_step import micro_step
    from agents.training.rust_rollout.consistency import _drop_saved, _never_unpacked
    st = args[-1]
    # gen3_gate_releases_memory_v1: this forward is NEVER backpropagated, so autograd keeps NOTHING for it
    # (`saved_tensors_hooks` that drop every saved tensor — K9(b)'s probe recipe, `gen3_probe_releases_graph_v1`: it
    # changes what autograd keeps, never a kernel or a value). Before, its outputs were STASHED on the policy (the
    # extractor's `stash` container, the masked distribution) and through them its fp32 graph — every saved activation
    # of a grad-mode fp32 forward at the micro size, 4,486 MiB on the X26 surface — stayed alive under the compiled arm
    # that followed, until that arm's forward replaced the stashes: the gate's and the canary's peak rose from 5.4 to
    # 8.2 GiB (the 2026-10-15 `rb_xs_s2701` UpdateWontFit; `compile_regions_memory_test`).
    with fp32_scope(model.policy), torch.autograd.graph.saved_tensors_hooks(_drop_saved, _never_unpacked):
        out = micro_step(*args)
    fix = clip_relocation(out.log_prob.detach(), args[4], out.advantages.detach(), float(st.clip_range),
                          clip_margin_of(model) if margin is None else margin, shift_max_source(model))
    del out
    release_gate_memory(model)
    _mem("relocation forward")
    return args[:4] + (fix.old_log_prob,) + args[5:], fix


#: THE YARDSTICK ARMS (`gen3_r1_yardstick_margin_v1`, 2026-10-19; the X34 u1700 UNCONFIRMED,
#: `designs/research_state/measurements/x34_canary_u1700_2026-10-19/`). The margin rule reads every arm against the fp32
#: reference, but only the COMPILED arm ("compiled", "graphed": any arm not named here) is under judgement; eager (bf16
#: or fp32) and the fp32 reference are the YARDSTICK it is measured against. At u1700 EAGER bf16 moved a fixture row by
#: 0.856 > M 0.72 while compiled read 0.52 with 0 clip switches: a bf16 conditioning tail (the fp32 function alone moves
#: a row by up to 0.54 under a half-ulp weight dither), which the old rule refused as "compiled DISAGREES with eager"
#: and counted toward the canary's confirm / FATAL ladder. Now a yardstick out of margin is a recorded EVENT, never a
#: refusal on its own: the rung re-runs ONCE at M' = max(M, `YARDSTICK_HEADROOM` x the largest YARDSTICK shift) and is
#: judged there. The compiled arm's shift never sets M', so a miscompile cannot widen its own bar; a compiled arm out
#: of M (with every yardstick inside) is refused exactly as before. Deterministic (rule 8): the arms' forwards do not
#: read the old log-probs, so the re-run reads the same shifts (CUDA eager / compiled are run-to-run bit-identical,
#: window D) and the yardstick sits 1/1.25 inside M'. A yardstick STILL out at M' did not reproduce its own forward:
#: that is refused (`YardstickRefusal`) and the canary's ladder counts it.
YARDSTICK_ARMS: Tuple[str, ...] = ("eager", "fp32 reference")
#: M' headroom over the yardstick's own shift. **UNVERIFIED** — a proposal with no CUDA census behind it (u1700 README).
YARDSTICK_HEADROOM = 1.25


class YardstickRefusal(ct.CompileTrainerError):
    """A refusal whose failing arm is a YARDSTICK (`YARDSTICK_ARMS`) or the bf16 DRIFT rule (eager bf16 vs the fp32
    reference) — never the compiled arm. Typed so the canary words it as itself, never "compiled DISAGREES with eager";
    it still refuses (and the canary's ladder still counts it): a yardstick that does not reproduce its own forward, or
    a bf16 function that left the fp32 one, leaves nothing to judge the compiled arm by."""


class _YardstickOutOfMargin(Exception):
    """Internal signal: a yardstick arm broke the margin on a rung's FIRST read; the rung re-runs at
    ``event["rerun_margin"]``. Never escapes `_r1_judge` / `_r1_accum_rule`."""

    def __init__(self, event: Dict[str, Any]) -> None:
        super().__init__(f"yardstick out of margin: {event}")
        self.event = event


def _nan_inf(d: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(d, nan=float("inf"))


def _margin_reads(fix: ClipRelocation, arms: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Tuple[float, int, int]]:
    """Per arm: (its largest |Δ log π| from the relocation's reference, the batch row that carries it (-1 on an empty
    batch), how many rows' clip branch it switches). A non-finite shift reads +inf."""
    act_ref = clip_active(fix.ref_log_prob, fix.old_log_prob, fix.advantages, fix.clip_range)
    reads: Dict[str, Tuple[float, int, int]] = {}
    for name, arm in arms.items():
        lp = arm["log_prob"].detach().double().cpu().reshape(-1)
        if lp.shape != fix.ref_log_prob.shape:
            raise ct.CompileTrainerError(f"--compile-trainer region R1: the {name} arm's log-probs "
                                         f"{tuple(lp.shape)} do not match the fixture's {tuple(fix.ref_log_prob.shape)}")
        d = _nan_inf((lp - fix.ref_log_prob).abs())
        row = int(d.argmax()) if d.numel() else -1
        shift = float(d[row]) if d.numel() else 0.0
        shift = shift if np.isfinite(shift) else float("inf")
        flips = int((clip_active(lp, fix.old_log_prob, fix.advantages, fix.clip_range) != act_ref).sum())
        reads[name] = (shift, row, flips)
    return reads


def _same_precision(arms: Dict[str, Dict[str, torch.Tensor]]) -> Optional[Tuple[str, float, int]]:
    """THE SAME-PRECISION FORWARD (gen3_r1_yardstick_margin_v1): (the compiled arm, max |log π_compiled − log π_eager|,
    its batch row) — what the compiler did, without bf16's conditioning (CPU Inductor read 0.017-0.028 at X34 268M where
    each arm sat 0.10-0.13 from fp32). RECORDED with NO bar until a CUDA census; None without both arms."""
    judged = [n for n in arms if n not in YARDSTICK_ARMS]
    if "eager" not in arms or not judged:
        return None
    a = arms[judged[0]]["log_prob"].detach().double().cpu().reshape(-1)
    b = arms["eager"]["log_prob"].detach().double().cpu().reshape(-1)
    if a.shape != b.shape or not a.numel():
        return None
    d = _nan_inf((a - b).abs())
    row = int(d.argmax())
    v = float(d[row])
    return judged[0], (v if np.isfinite(v) else float("inf")), row


def _rerun_source(event: Dict[str, Any]) -> str:
    """The re-run's margin source, for the rule line and the record (`ClipRelocation.shift_source`)."""
    arm, d = max(event["out_of_margin"].items(), key=lambda kv: kv[1]["shift"])
    return f"yardstick re-run: {YARDSTICK_HEADROOM:g} x {arm} {d['shift']:.4g}"


def _g(v: Any, spec: str = ".3g") -> str:
    """A number as ``spec``; a recorded non-finite value (`GradWatch.record` writes it as its string) as itself."""
    return format(v, spec) if isinstance(v, (int, float)) else str(v)


def yardstick_line(event: Dict[str, Any]) -> str:
    """One yardstick event (live, or as `GradWatch.record` wrote it), for a rule line and the canary's log."""
    arms = ", ".join(f"{n} {_g(d['shift'])} (row {d['row']})" for n, d in event["out_of_margin"].items())
    return (f"YARDSTICK out of margin, NOT counted (gen3_r1_yardstick_margin_v1): {arms} > M {_g(event['margin'], '.3f')} "
            f"on the first read (compiled {_g(event['compiled_shift'])}) — re-run once at M' "
            f"{_g(event['rerun_margin'], '.3f')} = max(M, {YARDSTICK_HEADROOM:g} x the yardstick's shift; the compiled "
            f"arm's never sets it)")


def _clip_margin_rule(fix: ClipRelocation, arms: Dict[str, Dict[str, torch.Tensor]],
                      watch: Optional["GradWatch"] = None, *, rerun: bool = False) -> str:
    """The precondition (module comment above): every judged arm's largest |Δ log π| from the relocation's
    reference is <= the margin, and — its consequence, checked rather than assumed — no arm's clip branch differs from
    the reference's on any row. Returns the rule-line fragment, else:

      * ``rerun`` and a YARDSTICK arm (`YARDSTICK_ARMS`) out of margin: `_YardstickOutOfMargin`, carrying the event
        and M' (the caller re-runs the rung there; the compiled arm is judged only at M');
      * a COMPILED arm out of margin: `CompileTrainerError` naming it (as before gen3_r1_yardstick_margin_v1);
      * a yardstick out of margin without ``rerun`` (the re-run itself, or a direct caller): `YardstickRefusal`."""
    reads = _margin_reads(fix, arms)
    same = _same_precision(arms)
    if watch is not None:
        watch.fill_margin(fix, reads, same)
    out = [n for n, (sh, _, fl) in reads.items() if not sh <= fix.margin or fl]
    yard = [n for n in out if n in YARDSTICK_ARMS]
    judged = [n for n in out if n not in YARDSTICK_ARMS]
    if yard and rerun:
        widest = max(reads[n][0] for n in yard)
        m2 = max(fix.margin, YARDSTICK_HEADROOM * widest)
        if np.isfinite(m2) and m2 > fix.margin:
            comp = [reads[n][0] for n in reads if n not in YARDSTICK_ARMS]
            raise _YardstickOutOfMargin({
                "margin": float(fix.margin), "rerun_margin": float(m2), "headroom": YARDSTICK_HEADROOM,
                "out_of_margin": {n: {"shift": reads[n][0], "row": reads[n][1], "clip_switches": reads[n][2]}
                                  for n in yard},
                "first_read_max_abs_dlogp": {n: r[0] for n, r in reads.items()},
                "first_read_worst_row": {n: r[1] for n, r in reads.items()},
                "compiled_shift": max(comp) if comp else float("nan")})
    for name in judged + yard:
        shift, row, flips = reads[name]
        if name in YARDSTICK_ARMS:
            raise YardstickRefusal(
                f"--compile-trainer region R1: the YARDSTICK {name} arm — NOT the compiled one — moved a fixture "
                f"row's log-prob by up to {shift:.3g} (row {row}) from the fp32 eager reference and switched {flips} "
                f"row(s)' PPO clip branch, past the margin {fix.margin:.3g} [{fix.shift_source}]"
                + (" — set by its OWN first read: the yardstick did not reproduce its own forward"
                   if fix.shift_source.startswith("yardstick") else "")
                + ". The compiled arm cannot be judged against it (gen3_r1_yardstick_margin_v1).")
        raise ct.CompileTrainerError(
            f"--compile-trainer region R1: the {name} arm moved a fixture row's log-prob by up to {shift:.3g} "
            f"(row {row}) from the fp32 eager reference and switched {flips} row(s)' PPO clip branch (the clip margin "
            f"is {fix.margin:.3g} = {R1_BAR_K:g} x the shift {fix.shift_max:.4g} [{fix.shift_source}]; the largest "
            f"healthy shift measured is R1_LOGPROB_SHIFT_MAX {R1_LOGPROB_SHIFT_MAX:g}). The gate's rows are "
            f"relocated off the clip boundary by that margin so no verdict depends "
            f"on a 0/1 switch (standing rule 8, gen3_r1_clip_margin_v1): a forward this far from the reference is "
            f"refused as itself. If this arm is HEALTHY, the margin is too small — re-measure "
            f"(main.ops.diag_bf16_canary) and raise the shift: {SHIFT_MAX_FLAG} <v> on a RESUME of this run "
            f"(check-only, it can only raise; T37), or R1_LOGPROB_SHIFT_MAX in code for the next pin; never "
            f"judge across a switch.")
    line = (fix.line() + "; max |dlogp| " + ", ".join(f"{k} {v[0]:.2e} (row {v[1]})" for k, v in reads.items())
            + f" <= {fix.margin:.3f}, 0 clip-branch switches")
    if same is not None:
        line += f"; same-precision |{same[0]} - eager| {same[1]:.2e} (row {same[2]}; recorded, no bar)"
    return line


# ------------------------------------------------------------------ the canary's WATCH-LIST record
class GradWatch:
    """A DECLARED watch-list of parameters (name prefixes) whose per-check numbers R1's live rung records whatever
    the verdict (`gen3_canary_watch_v1`, 2026-10-15): the bf16 rule's per-parameter ratio, the compiled distance and
    eager bf16's yardstick (or the fp32 rule's compiled-vs-eager error and bar), the check's worst parameter and its
    global numbers. The canary writes it to `canary_verdicts.jsonl`, so a parameter family's GROWTH CURVE is
    recoverable from a run (the `rb_toy_ent_s4001` FATAL logged only each check's worst parameter, so the pointer
    head's ratio at the earlier canaries is unrecoverable). Pure bookkeeping on numbers the rule computes anyway."""

    def __init__(self, prefixes: Tuple[str, ...]) -> None:
        self.prefixes = tuple(prefixes)
        self.rule: Optional[str] = None
        self.params: Dict[str, Dict[str, Any]] = {}
        self.worst: Optional[Dict[str, Any]] = None
        self.globals: Dict[str, float] = {}
        self.margin: Optional[Dict[str, Any]] = None
        #: per batch row, the committed fixture row it tiles (`R1Batch.fixture_rows`; set by the canary), so a worst row is
        #: recorded as a FIXTURE row too; None = batch rows only
        self.fixture_rows: Optional[List[int]] = None
        self.yardstick: List[Dict[str, Any]] = []        # gen3_r1_yardstick_margin_v1: the rung's yardstick events

    def _fixture(self, row: int) -> Optional[int]:
        fr = self.fixture_rows
        return int(fr[row]) if fr is not None and 0 <= row < len(fr) else None

    def fill_margin(self, fix: "ClipRelocation", reads: Dict[str, Tuple[float, int, int]],
                    same: Optional[Tuple[str, float, int]] = None) -> None:
        """The rung's clip-boundary fixture (`gen3_r1_clip_margin_v1`): rows relocated, the margin, the nearest unmoved
        row's distance and every arm's largest |Δ log π| from the reference — so a run's records carry the compiled
        arm's shift (the margin's provisional input) at every canary — plus (gen3_r1_yardstick_margin_v1) the WORST
        ROW of every arm (batch row, and fixture row when known) and the same-precision compiled-vs-eager gap."""
        self.margin = {"relocated_rows": int(fix.moved.sum()), "rows": int(fix.moved.numel()),
                       "margin": float(fix.margin), "shift_max": float(fix.shift_max),
                       "shift_max_source": str(fix.shift_source), "nearest_unmoved": float(fix.nearest_unmoved),
                       "max_abs_dlogp": {k: float(v[0]) for k, v in reads.items()},
                       "worst_row": {k: int(v[1]) for k, v in reads.items()}}
        if self.fixture_rows is not None:
            self.margin["worst_fixture_row"] = {k: self._fixture(int(v[1])) for k, v in reads.items()}
        if same is not None:
            self.margin["same_precision"] = {"arms": f"{same[0]} - eager", "max_abs_dlogp": float(same[1]),
                                             "row": int(same[2]), "fixture_row": self._fixture(int(same[2])),
                                             "bar": None}

    def add_yardstick(self, event: Dict[str, Any]) -> None:
        """A yardstick-out-of-margin event (`_clip_margin_rule`), its rows also as fixture rows when known."""
        ev = dict(event)
        ev["out_of_margin"] = {n: dict(d, fixture_row=self._fixture(int(d["row"])))
                               for n, d in event["out_of_margin"].items()}
        self.yardstick.append(ev)

    def wants(self, name: str) -> bool:
        return bool(self.prefixes) and name.startswith(self.prefixes)

    def fill_bf16(self, err_c: Dict[int, float], err_e: Dict[int, float], names: List[str], regime: str,
                  over: Dict[int, float], globals_: Dict[str, float]) -> None:
        self.rule, self.globals = "bf16", {k: float(v) for k, v in globals_.items()}
        best = (-1.0, -1)
        for i, nm in enumerate(names):
            if i in err_c:
                floor = float(over.get(i, R1_PARAM_BAR[regime]))
                ratio = err_c[i] / max(err_e[i], floor)
                ratio = ratio if ratio == ratio else float("inf")
                if ratio > best[0]:
                    best = (ratio, i)
                if self.wants(nm):
                    self.params[nm] = {"ratio": ratio, "compiled": err_c[i], "eager": err_e[i], "floor": floor}
            elif self.wants(nm):
                self.params[nm] = {"judged": False}
        if best[1] >= 0:
            i = best[1]
            self.worst = {"param": names[i], "ratio": best[0], "compiled": err_c[i], "eager": err_e[i]}

    def fill_fp32(self, eager: Dict[str, torch.Tensor], comp: Dict[str, torch.Tensor], names: List[str],
                  regime: str, over: Optional[Dict[int, float]]) -> None:
        sizes = [int(x) for x in eager["grad_sizes"].tolist()]
        err = dict(per_param_grad_errors(comp["grad"], eager["grad"], sizes, floor_frac=ct._PARAM_GRAD_FLOOR))
        ov = dict(over or {})
        self.rule = "fp32"
        for i, nm in enumerate(names):
            if self.wants(nm):
                self.params[nm] = ({"compiled_vs_eager": err[i], "bar": float(ov.get(i, R1_PARAM_BAR[regime]))}
                                   if i in err else {"judged": False})
        if err:
            j = max(err, key=lambda x: err[x])
            self.worst = {"param": names[j], "compiled_vs_eager": err[j]}

    def record(self) -> Dict[str, Any]:
        """The JSON-able record (4 significant digits; a non-finite value as its string)."""
        def r(v: Any) -> Any:
            if isinstance(v, float):
                return float(f"{v:.4g}") if np.isfinite(v) else str(v)
            if isinstance(v, dict):
                return {k: r(x) for k, x in v.items()}
            return v
        out: Dict[str, Any] = {"rule": self.rule, "params": r(self.params), "worst": r(self.worst),
                               "global": r(self.globals), "clip_margin": r(self.margin)}
        if self.yardstick:                                  # only when one occurred: other records are unchanged
            out["yardstick_out_of_margin"] = [r(e) for e in self.yardstick]
        return out


# ------------------------------------------------------------------ THE bf16 PARITY RULE
#: R1 under `--trunk-precision bf16` (`gen3_r1_bf16_parity_v1`; T25 lever A; the derivation is
#: `designs/research_state/measurements/perf_phase_2026-10-10/PLAN.md` §2a, its numbers `results/bf16_parity_bars.json`,
#: re-derived by `compile_regions_bf16_test`). A bf16 compiled micro-step cannot equal a bf16 EAGER one to fp32's bars:
#: each arm rounds at different points (Inductor fuses and upcasts where eager rounds every op's output to bf16), so the
#: two differ by bf16 ROUNDING (unit roundoff 2^-8 relative, ~32,768x fp32's), and the fp32 rule refuses it on the
#: loss alone (window B: loss rel 9.7e-4 > 1e-4). So the rule measures BOTH bf16 arms against the fp32 EAGER
#: REFERENCE (the production function, the same weights and rows: `_r1_reference`, under `trunk_precision.fp32_scope`)
#: and asks two questions, every one a deterministic comparison:
#:
#:   DRIFT    is the bf16 FUNCTION still near the fp32 one?  eager-bf16 vs the reference — the loss's relative
#:            error and the gradient's global relative L2 error — each under `BF16_DRIFT_BARS` (R1_BAR_K x the
#:            healthy maximum over the measured states). Catches bf16 losing the model (an overflow, a trunk whose
#:            activations outgrew bf16's 8-bit mantissa): what the fp32 gate could never see, because both its arms
#:            were fp32.
#:   ENVELOPE is the COMPILED bf16 graph no farther from the reference than eager bf16 is?  For the loss, the global
#:            gradient and EVERY judged parameter p: d(compiled, ref) <= K x max(d(eager, ref), floor), the floor
#:            being the fp32 gate's own bar for that quantity (global sqrt(2(1 - 0.9999)) = the fp32 cosine bar as
#:            a relative L2; per parameter `R1_PARAM_BAR[regime]`, an unmoved parameter's at the FRESH bar as in the
#:            fp32 rule) — except the scalar LOSS, floored at bf16's unit roundoff 2^-9 (`BF16_LOSS_FLOOR`) — so a
#:            compiled bf16 graph is never held tighter than an fp32 one, and a quantity whose eager error lands
#:            near zero by chance does not shrink its own bar. K = R1_BAR_K x the healthy maximum of that ratio
#:            (`BF16_ENVELOPE_K`).
#:
#: THE TEETH, stated (PLAN.md §2a): a backward miscompile on a parameter is refused only when it exceeds K x that
#: parameter's own bf16 error — bf16's gradient noise is the resolution limit, coarser than the fp32 rule's by the
#: ratio of the two errors (`results/bf16_parity_bars.json` "teeth"). A 10 % backward error is NOT refused; the exact
#: per-parameter resolution, its confirmation by planted faults, and the 2026-10-15 decision to keep the fp32 reference
#: (a same-precision compiled-vs-eager-bf16 rule measured NO sharper: Inductor's fused kernels round differently from
#: eager bf16, so the two bf16 arms differ by bf16 rounding too) are `designs/training/compile_flags.md` "The bf16
#: gate's resolution" and `compile_regions_resolution_test`. `None` would mean NOT MEASURED: a bf16 R1 REFUSED.
#: DERIVED (window D, 2026-10-10, `E_bf16_envelope`: RTX 3080 Ti, torch 2.8, `--arch endstate`, B = 2,048 of the gate's
#: rows, 20 reads = fresh + 6 ladder-perturbed + 3 bf16-trained states x 2 slices, each arm twice — both bf16 arms were
#: run-to-run bit-identical): healthy max drift loss 3.26e-3, gradient rel-L2 0.169; healthy max envelope ratio loss
#: 1.35, gradient 1.58, per parameter 2.94. Each bar = R1_BAR_K x that, rounded UP to 2 significant digits.
BF16_DRIFT_BARS: Optional[Dict[str, float]] = {"loss": 0.014, "grad_rel_l2": 0.68}
BF16_ENVELOPE_K: Optional[Dict[str, float]] = {"loss": 5.4, "grad_rel_l2": 6.4, "param": 12.0}
#: the global gradient's floor: the fp32 rule's cosine bar as a relative L2 (|a - b| / |b| = sqrt(2 (1 - cos)) for
#: equal norms)
BF16_GRAD_FLOOR = float(np.sqrt(2.0 * (1.0 - ct._MIN_GRAD_COSINE)))
#: the loss's floor: bf16's UNIT ROUNDOFF, 2^-9 (round-to-nearest, an 8-bit significand), not fp32's 1e-4. The loss is
#: an fp32 reduction over values that each carry the trunk's bf16 output rounding, so a relative disagreement under u
#: is below what a bf16 forward resolves — and a SCALAR's eager error can land near zero by chance (window D: eager bf16
#: read 2.5e-5 on one read where compiled read 8.5e-4, both ~u), which would make an fp32 floor an 8.5x "ratio" of
#: pure rounding luck. Measured eager-bf16 loss errors: 2.5e-5 .. 3.3e-3 over 20 GPU reads.
BF16_LOSS_FLOOR = max(ct._MAX_NUMERIC_DRIFT, 2.0 ** -9)


def _r1_reference(model: Any, args: Tuple[Any, ...]) -> Dict[str, torch.Tensor]:
    """The bf16 rule's REFERENCE arm: R1's EAGER body with every trunk round at its fp32 base class
    (`trunk_precision.fp32_scope`) — the production function on the same weights and rows. Proved eager like the
    gate's own eager arm: R1's Python body exactly once, the compiled route zero times (a compiled call under the
    base classes would be a new signature)."""
    from agents.model.trunk_precision import fp32_scope
    from agents.training.instrumented_ppo.micro_step import micro_step
    e0, c0 = _routes()
    with fp32_scope(model.policy):
        ref = _r1_arm(model, micro_step, args)
    _mem("fp32 reference arm")
    e1, c1 = _routes()
    if (e1 - e0, c1 - c0) != (1, 0):
        raise ct.GateArmsNotIndependentError(
            f"--compile-trainer region R1 (bf16 rule): the fp32 REFERENCE arm ran the eager body {e1 - e0}x and "
            f"the compiled route {c1 - c0}x (want 1 / 0) — refusing (gen3_gate_independent_arms_v1)")
    return ref


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(b))


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / ||b|| in float64 (NaN when ``b`` is all zero or either side is non-finite)."""
    a64, b64 = a.detach().double(), b.detach().double()
    nb = float(b64.norm())
    out = float((a64 - b64).norm()) / nb if nb > 0 else float("nan")
    return out if np.isfinite(out) else float("nan")


def bf16_measured() -> Optional[str]:
    """``None`` when the bf16 rule has measured bars, else the refusal text."""
    if BF16_DRIFT_BARS is None or BF16_ENVELOPE_K is None:
        return ("--trunk-precision bf16: the compiled learner's bf16 PARITY RULE has no measured bars yet "
                "(compile_regions.BF16_DRIFT_BARS / BF16_ENVELOPE_K are None) — refusing to judge with a guess")
    return None


def _r1_verdict_bf16(eager: Dict[str, torch.Tensor], comp: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor],
                     names: List[str], regime: str, label: Optional[str] = None,
                     bar_by_index: Optional[Dict[int, float]] = None, watch: Optional["GradWatch"] = None,
                     losses: Optional[Tuple[float, float, float]] = None) -> str:
    """THE bf16 PARITY RULE (the block comment above): DRIFT (eager bf16 vs the fp32 reference) and ENVELOPE
    (compiled bf16 vs the reference, within K x eager bf16's own error, floored at the fp32 bar). ``losses``: the
    (reference, eager, compiled) losses every LOSS comparison reads — the gate passes each arm's loss on the golden's
    OWN rows (`golden_row_loss`, gen3_r1_loss_on_golden_rows_v1), the arms' own (relocated-row) losses are then
    reported beside them; None reads the arms' own. Raises `CompileTrainerError`; returns the rule line."""
    refusal = bf16_measured()
    if refusal is not None:
        raise ct.CompileTrainerError(refusal)
    assert BF16_DRIFT_BARS is not None and BF16_ENVELOPE_K is not None
    ct.require_finite_arms(eager=eager, compiled=comp, param_names=names)
    ct.require_finite_arms(eager=ref, compiled=eager, param_names=names)
    head = f"[{label or regime + ' weights'}] bf16 rule:"
    own = (float(ref["loss"].sum()), float(eager["loss"].sum()), float(comp["loss"].sum()))
    l_ref, l_e, l_c = own if losses is None else (float(losses[0]), float(losses[1]), float(losses[2]))
    if not all(np.isfinite(v) for v in own + (l_ref, l_e, l_c)):
        raise ct.CompileTrainerError(f"--compile-trainer region R1 (bf16 rule): non-finite loss (reference "
                                     f"{l_ref}, eager {l_e}, compiled {l_c}; the arms' own {own})")
    d_loss_e, d_loss_c = _rel(l_e, l_ref), _rel(l_c, l_ref)
    # the relocated rows' drift, REPORTED (the statistic before gen3_r1_loss_on_golden_rows_v1), never judged
    reloc = (_rel(own[1], own[0]), _rel(own[2], own[0]))
    rows = "" if losses is None else (f" [loss on the golden's own rows; relocated rows: drift {reloc[0]:.2e}, "
                                      f"envelope {reloc[1]:.2e}]")
    d_g_e, d_g_c = _rel_l2(eager["grad"], ref["grad"]), _rel_l2(comp["grad"], ref["grad"])
    sizes = [int(x) for x in ref["grad_sizes"].tolist()]
    err_c = dict(per_param_grad_errors(comp["grad"], ref["grad"], sizes, floor_frac=ct._PARAM_GRAD_FLOOR))
    err_e = dict(per_param_grad_errors(eager["grad"], ref["grad"], sizes, floor_frac=ct._PARAM_GRAD_FLOOR))
    over = dict(bar_by_index or {})
    if watch is not None:                                   # the record is complete whichever check refuses
        watch.fill_bf16(err_c, err_e, names, regime, over,
                        {"drift_loss": d_loss_e, "drift_grad": d_g_e, "envelope_loss": d_loss_c,
                         "envelope_grad": d_g_c, "drift_loss_relocated": reloc[0],
                         "envelope_loss_relocated": reloc[1]})
    # ---- DRIFT: the bf16 function vs the fp32 one
    for what, v in (("loss", d_loss_e), ("grad_rel_l2", d_g_e)):
        if not v <= BF16_DRIFT_BARS[what]:
            raise YardstickRefusal(
                f"--compile-trainer region R1 (bf16 rule): the bf16 EAGER micro-step has DRIFTED from the fp32 "
                f"function — {what} {v:.2e} > {BF16_DRIFT_BARS[what]:g} (eager bf16 vs the fp32 eager reference, "
                f"same weights and rows){rows if what == 'loss' else ''}). bf16 no longer represents this model: run "
                f"--trunk-precision fp32.")
    # ---- ENVELOPE: compiled bf16 no farther from the reference than K x eager bf16 (floored at the fp32 bar)
    checks = (("loss", d_loss_c, d_loss_e, BF16_LOSS_FLOOR),
              ("grad_rel_l2", d_g_c, d_g_e, BF16_GRAD_FLOOR))
    for what, dc, de, floor in checks:
        bar = BF16_ENVELOPE_K[what] * max(de, floor)
        if not dc <= bar:
            raise ct.CompileTrainerError(
                f"--compile-trainer region R1 (bf16 rule): the COMPILED bf16 micro-step is farther from the fp32 "
                f"reference than eager bf16 allows — {what}: compiled {dc:.2e} > {BF16_ENVELOPE_K[what]:g} x "
                f"max(eager {de:.2e}, floor {floor:g}) = {bar:.2e}{rows if what == 'loss' else ''}. A faster wrong "
                f"model is not a win.")
    if not err_c:
        raise ct.VacuousCompileParityError("--compile-trainer region R1 (bf16 rule): no parameter's reference "
                                           "gradient is above the per-parameter floor")
    k = BF16_ENVELOPE_K["param"]
    worst = (-1.0, -1, 0.0, 0.0)
    bad = []
    for i, dc in err_c.items():
        floor_i = float(over.get(i, R1_PARAM_BAR[regime]))
        ratio = dc / max(err_e[i], floor_i)
        if not (ratio == ratio):                            # NaN sorts worst and fails
            ratio = float("inf")
        if ratio > worst[0]:
            worst = (ratio, i, dc, err_e[i])
        if not ratio <= k:
            bad.append((names[i] if i < len(names) else f"#{i}", dc, err_e[i]))
    if bad:
        raise ct.CompileTrainerError(
            f"--compile-trainer region R1 (bf16 rule): the compiled TRAIN graph's gradient is farther from the fp32 "
            f"reference than {k:g} x eager bf16's own error on {len(bad)} parameter(s) (e.g. "
            + ", ".join(f"{n} compiled {c:.2e} vs eager {e:.2e}" for n, c, e in bad[:5])
            + "). A backward miscompile under bf16's rounding — investigate before re-enabling.")
    wn = names[worst[1]] if 0 <= worst[1] < len(names) else f"#{worst[1]}"
    return (f"{head} drift loss {d_loss_e:.2e} <= {BF16_DRIFT_BARS['loss']:g}, grad {d_g_e:.2e} <= "
            f"{BF16_DRIFT_BARS['grad_rel_l2']:g}; envelope loss {d_loss_c:.2e} (eager {d_loss_e:.2e}){rows}, grad "
            f"{d_g_c:.2e} (eager {d_g_e:.2e}); per-param worst ratio {worst[0]:.2f} <= {k:g} ({wn}: compiled "
            f"{worst[2]:.2e} vs eager {worst[3]:.2e}) over {len(err_c)} params")


def _decision_readout(model: Any, obs: Any, mask: Any) -> Dict[str, torch.Tensor]:
    """The EAGER decision readout (legal log-probs, V) of the policy's own rollout core on ``obs`` —
    what `weights_regime` reads to tell a fresh launch's constant legal log-probs from trained ones."""
    with torch.no_grad():
        values, logp = model.policy.rollout_core(obs, mask)
        legal = torch.as_tensor(mask, device=logp.device, dtype=torch.bool)
        return {"value": values.float().flatten().clone(),
                "legal_logprob": torch.where(legal, logp.float(), torch.zeros_like(logp.float())).clone()}


# ---------------------------------------------------------------------------------- the prewarm
def prewarm_calls(model: Any, *, batch_size: int) -> List[Tuple[str, Callable[[], None]]]:
    """The declared signatures, as prewarm calls (`compile_control.prewarm`): R1 at its update
    signature (forward + backward), on real rows — once per declared ROUTE (`r1_routes`: the compiled
    one, and the graphed one under `--learner-cudagraphs on`)."""
    policy = model.policy

    def _call(fn: Callable[..., Any]) -> Callable[[], None]:
        def _r1() -> None:
            was = policy.training
            policy.set_training_mode(True)
            try:
                with _fatal_compile_errors():
                    _r1_arm(model, fn, _r1_args(model, r1_batch(model, int(batch_size))))
            finally:
                policy.set_training_mode(was)
                policy.zero_grad(set_to_none=True)
        return _r1

    calls = [(f"R1 learner_micro_step train/grad B={int(batch_size)}", _call(model._compiled_micro_step))]
    g = getattr(model, "_compiled_micro_step_graphed", None)
    if g is not None:
        calls.append((f"R1[graphed] learner_micro_step train/grad B={int(batch_size)}", _call(g)))
    return calls
