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
2.5.1 `forward_guard`'s weakref lookup and other constructs break `fullgraph=True`. The regions are
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
perturbation, whose legal log-probs are not constant). The gate runs at fp32 matmul precision 'highest' only — the one precision (TF32 was retired, deletion pass
K2); a process at any other precision is refused. A disagreement is a `CompileTrainerError`
(FATAL_CONFIG).
"""
from __future__ import annotations

import contextlib
import hashlib
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from agents.model import compile_trainer as ct
from agents.model import region_calls as RC
from agents.training.lifecycle_decl import startup_builder


class Signature(NamedTuple):
    batch: str                       # "batch_size"
    mode: str                        # "eval" | "train"
    grad: bool


class Region(NamedTuple):
    name: str
    callable: str
    signatures: Tuple[Signature, ...]
    compiled: bool
    why: str


REGIONS: Tuple[Region, ...] = (
    Region("R1_learner_micro_step", "instrumented_ppo.micro_step.micro_step",
           (Signature("batch_size", "train", True),), True,
           "the accumulation group + optimizer step are per GROUP; the host read follows it"),
    Region("R2_rank_probe", "rank_metrics.rank_probe_from_stash (no forward)", (), False,
           "reads R1's own forward's stashes; the spectra are device tensors, one host read"),
    Region("R3_optimizer_step", "clip_grad_norm_ + AdamW.step + zero_grad", (), False,
           "0.04 s/update measured; the KL->LR controller moves lr every update"),
)


def compiled_regions() -> Tuple[Region, ...]:
    return tuple(r for r in REGIONS if r.compiled)


def declared_signature_count() -> int:
    """How many compiled graphs the table declares."""
    return sum(len(r.signatures) for r in compiled_regions())


# ------------------------------------------------------------------------------------- install
def install(model: Any, *, backend: Optional[str] = None,
            emit: Optional[Callable[[str], None]] = None) -> List[str]:
    """Compile R1 (`fullgraph=True`, static shapes) and install it as ``model._compiled_micro_step``
    (read by `TrainSetup._micro_region`). Any instance-level extractor forward (none is installed on
    the learner since the extractor-only compile's deletion) is removed first, so the region traces
    the extractor's own forward and every other caller runs it eager. Returns the installed region
    names. Compilation happens LAZILY at the region's first call — the gate and the prewarm make that
    first call at startup."""
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
    r1c = ctl.wrap_compiled(torch.compile(micro_step, **kw))
    rows = int(getattr(model, "batch_size", 0) or 0)

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
        return out
    r1._gen3_compiled = r1c                              # type: ignore[attr-defined]
    model._compiled_micro_step = r1
    # the R1 signature this run DECLARES (its levers from the resolved config) — every update is
    # held to it (`check_r1_declared`)
    model._r1_declared = r1_declaration(model)
    return ["R1_learner_micro_step"]


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
    ent = cache_entries_by_code()
    want = {"micro_step": 1}
    got = {name: sum(v for k, v in ent.items() if k.split(" ")[0] == name) for name in want}
    if got != want:
        raise ct.CompileTrainerError(
            f"--compile-trainer: the compiled-region INVENTORY differs from the declaration — "
            f"cache entries {got}, declared {want} (gen3_no_silent_eager_v1)")
    return "[CompileRegions] inventory == declaration: R1 1 graph (fullgraph=True)"


def uninstall(model: Any) -> None:
    model._compiled_micro_step = None
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
                   src, tuple(g.filled))


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


def _r1_args(model: Any, b: R1Batch) -> Tuple[Any, ...]:
    return (model.policy, b.obs, b.actions, b.action_masks,
            b.old_log_prob, b.old_values, b.advantages, b.returns, _r1_static(model))


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
    return {"loss": out.loss.detach().float().reshape(1), "grad": torch.cat(gs),
            "grad_sizes": torch.tensor([g.numel() for g in gs], dtype=torch.long)}


def _routes() -> Tuple[int, int]:
    """(R1's eager-body executions, R1's compiled-route dispatches) so far — `region_calls`' counters."""
    return int(RC.EAGER_BODY["R1"]), int(RC.peek().get("R1_compiled", 0))


def _r1_pair(model: Any, args: Tuple[Any, ...]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """``(compiled, eager)`` arms of ONE R1 parity comparison, with their INDEPENDENCE proved
    (`gen3_gate_independent_arms_v1`, F-XC-4): the compiled arm must go through the installed
    dispatcher's COMPILED route exactly once and execute R1's Python body ZERO times, and the eager arm
    must execute R1's Python body exactly once and dispatch the compiled route ZERO times. Anything else
    — the compiled slot holding the eager function (a thing compared with itself), a dispatcher that
    fell back to eager, an "eager" arm that dispatched the compiled graph — is a typed refusal, never a
    verdict. Counters, not identities: they read what each arm actually EXECUTED (`region_calls`) —
    so a wrapper around the installed dispatcher (a test's planted fault) is still the compiled arm,
    and the bare eager function in the compiled slot is refused (it runs the eager body)."""
    from agents.training.instrumented_ppo.micro_step import micro_step
    fn = getattr(model, "_compiled_micro_step", None)
    if fn is None:
        raise ct.CompileTrainerError(
            "--compile-trainer region R1: no compiled micro-step is installed, so the parity gate has no "
            "COMPILED arm (gen3_gate_independent_arms_v1)")
    e0, c0 = _routes()
    comp = _r1_arm(model, fn, args)
    e1, c1 = _routes()
    eager = _r1_arm(model, micro_step, args)
    e2, c2 = _routes()
    if (e1 - e0, c1 - c0) != (0, 1) or (e2 - e1, c2 - c1) != (1, 0):
        raise ct.CompileTrainerError(
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
            raise ct.CompileTrainerError(
                f"--compile-trainer: a DECLARED REGION does not compile as one graph "
                f"(`fullgraph=True`) — {type(exc).__name__}: {str(exc)[:3000]}\nA graph break inside a "
                f"region is an error by declaration (K8, gen3_declared_regions_v1): make the construct "
                f"traceable, or split the region in `compile_regions.REGIONS` deliberately.") from exc
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
    try:
        with _fatal_compile_errors(), torch.random.fork_rng(devices=devices):
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
    finally:
        policy.set_training_mode(was)
        for p in policy.parameters():
            p.grad = None
    line = "[CompileRegions] parity PASS (fp32) — " + " | ".join(rules)
    say(line)
    return rules


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
            b.old_log_prob, b.old_values, b.advantages, b.returns, st)


def _r1_judged_set_rule(model: Any, b: R1Batch, run: Dict[str, torch.Tensor], names: List[str]) -> str:
    """REFUSE when filling ``b.filled`` shrank R1's judged parameter set below the golden rows' own
    (`judged_set_shrinkage`): a placeholder that blanks a term would leave its parameters unjudged —
    the P10-C defect class. One extra EAGER micro-step, only on a run whose space the golden lacks a
    key of (never on the production surface). Returns the gate-line rule."""
    from agents.training.instrumented_ppo.micro_step import micro_step
    try:
        ref = _r1_arm(model, micro_step, _r1_reference_args(model, b))
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
             source: Optional[str] = None
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
    """
    unmoved = unmoved_parameters(model) if regime != "fresh" else []
    comp, eager = _r1_pair(model, args)
    idx = {n: i for i, n in enumerate(names)}
    excepted = {idx[n]: R1_PARAM_BAR["fresh"] for n in unmoved}
    tail = f" [{source}]" if source else ""
    if unmoved:
        tail += f" [unmoved rule: {unmoved_rule(model)}]"
    rules = ["R1 " + _r1_verdict(eager, comp, names, regime, bar_by_index=excepted) + tail]
    if regime == "fresh":
        rules.append("R1 " + _r1_perturbed(model, args, names))
    elif unmoved:
        rules.append("R1 " + _r1_unmoved_perturbed(model, args, names, unmoved))
    return eager, comp, rules


def _r1_unmoved_perturbed(model: Any, args: Tuple[Any, ...], names: List[str],
                          unmoved: List[str]) -> str:
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
        comp, eager = _r1_pair(model, args)
    return _r1_verdict(eager, comp, names, "trained",
                       label=(f"trained weights, the {len(only)} unmoved param(s) perturbed "
                              f"scale={scale:g} seed+{k} (e.g. {only[0]})"))


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
    try:
        e = _decision_readout(model, obs, mask)
    finally:
        policy.set_training_mode(was)
    bar = {"legal_logprob": ct._FP32_TOL["legal_logprob"]}
    return "fresh" if vacuous_keys({"legal_logprob": e["legal_logprob"]}, bar) else "trained"


def _r1_perturbed(model: Any, args: Tuple[Any, ...], names: List[str]) -> str:
    """FRESH weights only: R1 again on the declared ladder's first rung — a seeded perturbation of
    every policy parameter, in place and restored bit-exactly (`parity_probe.perturbed_parameters`)
    — judged at the TRAINED bar. On fresh weights a few ill-conditioned gradients (the uniform belief
    heads) need the looser fresh bar, under which a backward defect could hide; the perturbed weights
    are trained-like and get the tight one (the deleted extractor gate's fresh-weights rule)."""
    from agents.model.parity_probe import PERTURB_LADDER, perturbed_parameters, rung_seed
    scale, k = PERTURB_LADDER[0]
    with perturbed_parameters(model.policy, seed=rung_seed(k), scale=scale):
        comp, eager = _r1_pair(model, args)
    return _r1_verdict(eager, comp, names, "trained",
                       label=f"fresh weights, seeded perturbation scale={scale:g} seed+{k}")


def _r1_verdict(eager: Dict[str, torch.Tensor], comp: Dict[str, torch.Tensor], names: List[str],
                regime: str, label: Optional[str] = None,
                bar_by_index: Optional[Dict[int, float]] = None) -> str:
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
    signature (forward + backward), on real rows."""
    policy = model.policy

    def _r1() -> None:
        was = policy.training
        policy.set_training_mode(True)
        try:
            with _fatal_compile_errors():
                _r1_arm(model, model._compiled_micro_step, _r1_args(model, r1_batch(model, int(batch_size))))
        finally:
            policy.set_training_mode(was)
            policy.zero_grad(set_to_none=True)

    return [(f"R1 learner_micro_step train/grad B={int(batch_size)}", _r1)]
