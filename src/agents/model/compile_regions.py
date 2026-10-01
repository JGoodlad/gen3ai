"""K8 — DECLARED COMPILE REGIONS (`gen3_declared_regions_v1`, M5 Lane K; owner 2026-09-30).

THE RULE. Every compiled learner-process callable is a named REGION in ONE declaration table
(`REGIONS`), compiled with `fullgraph=True` — so a graph break anywhere INSIDE a region is a compile
ERROR at startup naming dynamo's reason and file:line — at exactly its DECLARED signatures (shape ×
train/eval × grad), prewarmed at startup and then LOCKED (K6). The only legal "breaks" are the region
boundaries: plain Python between regions. A deliberate split is two regions, never an allowed break.
Rejected (Decision record): a pinned graph-break COUNT (a count hides a moved break) and break sites
pinned by file:line (line keys churn on every edit). Owner: correctness beats churn.

THE TABLE (`REGIONS`):

  R0  rollout_forward      `Gen3DualHeadMaskablePolicy.rollout_core`: extractor + MLP towers +
                           pointer head + critic read + FUNCTIONAL masking -> (values, masked log-
                           probs). eval / no-grad / batch n_envs. The action draw stays eager (the
                           same `multinomial` call, the same RNG stream). Batch 1 (the in-process
                           final evaluation) runs the eager core (`gen3_batch1_eager_v1`). Serves the
                           python env core's rollout; on `--env-core rust` T2 serves rollouts.
  R1  learner_micro_step   `instrumented_ppo.micro_step.micro_step`: `evaluate_actions` + fold steps
                           1–3a (`gen3_learner_micro_step_v1`). train / grad / batch batch_size; its
                           backward is AOTAutograd's, from the same graph. A RAGGED micro-batch (any
                           other row count: the last of an epoch over an uneven rollout) runs the
                           same function EAGER — a declared route, never a new signature.
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
                           of the fold (dense aux, the CF-twin mirror, value-dist, distill + anchor,
                           search-teacher, OPD, TD-aux, the counterfactual block — none on the
                           production surface), every diagnostic probe, logging.

TORCH. Regions are a torch 2.8 feature (owner 2026-09-30: 2.8 is the target, 2.5.1 is legacy for
pinned resumes): on 2.5.1 `forward_guard`'s weakref lookup and other constructs break
`fullgraph=True`, so `regions_supported()` is False there and `--compile-trainer` keeps the legacy
extractor-only compile — the pinned commit's behaviour.

THE GATE. `gate_regions` holds each compiled region to eager at startup on REAL rows (R1 on the K9
learner golden's real labelled buffer when the run's observation keys match it — the production
surface — else on the committed real-obs fixture with zero labels): R1's loss and the gradient over
every policy parameter (cosine ≥ 0.9999 and the per-parameter rule), R0's decision readout (legal
log-probs, V; on a seeded perturbation of FRESH weights, whose legal log-probs are constant). Under
`--matmul-precision high` the TF32 rule against an eager fp32 reference. A disagreement is a
`CompileTrainerError` (FATAL_CONFIG).
"""
from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from agents.model import compile_trainer as ct
from agents.model import region_calls as RC


class Signature(NamedTuple):
    batch: str                       # "n_envs" | "batch_size"
    mode: str                        # "eval" | "train"
    grad: bool


class Region(NamedTuple):
    name: str
    callable: str
    signatures: Tuple[Signature, ...]
    compiled: bool
    why: str


REGIONS: Tuple[Region, ...] = (
    Region("R0_rollout_forward", "Gen3DualHeadMaskablePolicy.rollout_core",
           (Signature("n_envs", "eval", False),), True,
           "the env step is the boundary; the action draw stays eager (same RNG stream)"),
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


def declared_signature_count(n_envs: int) -> int:
    """How many compiled graphs the table declares (a batch in `EAGER_BATCHES` is not compiled)."""
    n = 0
    for r in compiled_regions():
        for s in r.signatures:
            if not (s.batch == "n_envs" and int(n_envs) in ct.EAGER_BATCHES):
                n += 1
    return n


def regions_supported(version: Optional[str] = None) -> bool:
    """Declared regions need torch 2.8 (2.5.1 cannot compile them `fullgraph=True`)."""
    return str(version or torch.__version__).startswith("2.8")


# ------------------------------------------------------------------------------------- install
def _rollout_core(policy: Any, obs: Any, action_masks: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    out: Tuple[torch.Tensor, torch.Tensor] = policy.rollout_core(obs, action_masks)
    return out


def install(model: Any, *, backend: Optional[str] = None,
            emit: Optional[Callable[[str], None]] = None) -> List[str]:
    """Compile R0 and R1 (`fullgraph=True`, static shapes) and install them: R1 as
    ``model._compiled_micro_step`` (read by `TrainSetup._micro_region`), R0 in the policy module's
    weak registry (`policy._ROLLOUT_REGIONS`). Any instance-level compiled extractor forward (the
    startup gate's) is removed first, so the regions trace the extractor's own forward and every
    other caller runs it eager. Returns the installed region names. Compilation happens LAZILY at
    each region's first call — the gate and the prewarm make that first call at startup."""
    from agents.model.compile_control import control
    from agents.model.policy import _ROLLOUT_REGIONS
    from agents.training.instrumented_ppo.micro_step import micro_step
    policy = model.policy
    fe = policy.features_extractor
    # K3 (gen3_hermetic_compile_cache_v1): the regions compile into the RUN's own cache (declared by
    # the trainer at startup; idempotent here), never a cache torch would pick by default.
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

    def r1(policy: Any, popart: Any, obs: Any, actions: Any, *rest: Any) -> Any:
        # R1's ONE declared signature is `batch_size` rows. A RAGGED micro-batch (the last one of an
        # epoch when the rollout does not divide evenly — fork rows, an uneven sizing) takes the
        # DECLARED EAGER route, the same function: never a new compiled signature after the lock.
        # Its declared size is ONE per epoch (`rollout_buffer.get` yields full batches, then the
        # rest); more in one update is not a ragged tail any more (gen3_no_silent_eager_v1).
        if int(actions.shape[0]) != rows:
            n = RC.count("R1_eager_ragged")
            cap = max(1, int(getattr(model, "n_epochs", 1) or 1))
            if ctl.locked and n > cap:
                raise ct.CompileTrainerError(
                    f"--compile-trainer region R1: {n} RAGGED micro-batches in one update (declared: "
                    f"at most one per epoch, {cap}) — R1 is running EAGER on the hot path "
                    f"({int(actions.shape[0])} rows vs the declared {rows}); a silent ~2x slowdown, "
                    f"stopped (gen3_no_silent_eager_v1)")
            return micro_step(policy, popart, obs, actions, *rest)
        before = RC.EAGER_BODY["R1"]
        out = r1c(policy, popart, obs, actions, *rest)
        _ran_compiled("R1", before)
        RC.count("R1_compiled")
        return out
    r1._gen3_compiled = r1c                              # type: ignore[attr-defined]
    r0c = ctl.wrap_compiled(torch.compile(_rollout_core, **kw))

    def r0(pol: Any, obs: Any, action_masks: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        out: Tuple[torch.Tensor, torch.Tensor]
        # ONE declared mask input: a bool tensor on the learner's device (`gen3_r0_mask_dtype_v1`).
        # The rollout hands the policy the env's numpy masks — int8 from the env workers — while the
        # gate and the prewarm passed numpy bool, and dynamo guards a numpy input's dtype: the first
        # real rollout was an UNDECLARED signature, the sentinel's FATAL at rollout end (found by the
        # launcher restart proof, 2026-10-01). `masked_categorical` reads any of them as
        # `as_tensor(m, dtype=bool)`, so normalising here changes no number.
        if action_masks is not None:
            action_masks = torch.as_tensor(action_masks, dtype=torch.bool,
                                           device=ct.resolve_device(pol.features_extractor))
        if ct._rows(obs) in ct.EAGER_BATCHES:            # gen3_batch1_eager_v1
            RC.count("R0_eager_batch1")                  # declared, counted, logged per update
            out = pol.rollout_core(obs, action_masks)
        else:
            before = RC.EAGER_BODY["R0"]
            out = r0c(pol, obs, action_masks)
            _ran_compiled("R0", before)
            RC.count("R0_compiled")
        return out

    model._compiled_micro_step = r1
    _ROLLOUT_REGIONS[policy] = r0
    return ["R0_rollout_forward", "R1_learner_micro_step"]


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


def assert_inventory(model: Any, n_envs: int) -> str:
    """The compiled-region INVENTORY equals the declaration, in the run (startup, after the
    prewarm): one cache entry on R1's code object, one on R0's (none when `n_envs` is a declared
    eager batch). Other code objects are not judged (they belong to other components). Raises
    `CompileTrainerError`; returns the log line."""
    from agents.model.compile_control import cache_entries_by_code
    ent = cache_entries_by_code()
    want = {"micro_step": 1, "_rollout_core": 0 if int(n_envs) in ct.EAGER_BATCHES else 1}
    got = {name: sum(v for k, v in ent.items() if k.split(" ")[0] == name) for name in want}
    if got != want:
        raise ct.CompileTrainerError(
            f"--compile-trainer: the compiled-region INVENTORY differs from the declaration — "
            f"cache entries {got}, declared {want} (gen3_no_silent_eager_v1)")
    return (f"[CompileRegions] inventory == declaration: R1 1 graph, R0 {want['_rollout_core']} "
            f"graph(s) (fullgraph=True)")


def uninstall(model: Any) -> None:
    from agents.model.policy import _ROLLOUT_REGIONS
    model._compiled_micro_step = None
    _ROLLOUT_REGIONS.pop(model.policy, None)


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


def _golden_rows(model: Any) -> Optional[Dict[str, np.ndarray]]:
    """The K9 learner golden's committed REAL buffer, when its observation keys and shapes match this
    run's (the production surface) — else None."""
    try:
        from agents.training.learner_golden import BUFFER_PATH
        with np.load(BUFFER_PATH) as z:
            data = {k: z[k] for k in z.files}
    except Exception:
        return None
    space = model.policy.observation_space.spaces
    have = {k[4:]: data[k] for k in data if k.startswith("obs:")}
    if set(have) != set(space) or any(
            tuple(have[k].shape[2:]) != tuple(space[k].shape) for k in space):
        return None
    return data


def r1_batch(model: Any, batch: int, slice_: int = 0) -> R1Batch:
    """A micro-batch shaped EXACTLY like `rollout_buffer.get(batch_size)`'s (every key of the
    observation space, the buffer's dtypes): the golden's real labelled rows tiled to ``batch`` when
    they fit this run, else the real-obs fixture with zero labels and seeded PPO quantities."""
    dev = ct.resolve_device(model.policy.features_extractor)
    space = model.policy.observation_space.spaces
    # the buffer's own action dtype (a Discrete space stores int64): a different dtype is a different
    # signature — measured: the sentinel named `tensor 'actions' dtype mismatch` after the lock.
    adt = torch.as_tensor(np.zeros(1, dtype=model.rollout_buffer.actions.dtype)).dtype
    data = _golden_rows(model)
    if data is not None:
        n = int(data["actions"].reshape(-1).shape[0])
        idx = ct.fixture_index(int(batch), n, slice_)

        def flat(a: np.ndarray) -> np.ndarray:
            rows: np.ndarray = a.reshape(-1, *a.shape[2:])[idx]
            return rows
        obs = {k: torch.as_tensor(flat(data["obs:" + k]).astype(space[k].dtype), device=dev)
               for k in space}
        f = {k: torch.as_tensor(flat(data[k]), device=dev)
             for k in ("actions", "log_probs", "values", "advantages", "returns", "action_masks")}
        return R1Batch(obs, f["actions"].to(adt).reshape(-1, 1), f["action_masks"].float(),
                       f["log_probs"].float().reshape(-1), f["values"].float().reshape(-1),
                       f["advantages"].float().reshape(-1), f["returns"].float().reshape(-1),
                       "the K9 learner golden's real labelled buffer")
    obs = ct._prewarm_obs(model, int(batch), slice_)
    _, mask = ct._parity_obs(int(obs["observation"].shape[-1]), int(batch), dev, slice_)
    m = torch.as_tensor(mask, device=dev)
    g = torch.Generator(device="cpu").manual_seed(20260930)
    acts = torch.argmax(m.float() + 0.01 * torch.rand(m.shape, generator=g).to(dev), dim=-1)
    b = int(batch)
    return R1Batch(obs, acts.to(adt).reshape(-1, 1), m.float(),
                   (-torch.rand(b, generator=g) * 2.0).to(dev), torch.rand(b, generator=g).to(dev),
                   torch.randn(b, generator=g).to(dev), torch.rand(b, generator=g).to(dev),
                   "the committed real-obs fixture (zero labels)")


def _r1_args(model: Any, b: R1Batch) -> Tuple[Any, ...]:
    f = model._resolve_fold_flags()
    st = model._micro_static(f, getattr(model.policy, "popart", None), None, False)
    var = model._micro_var(st, None)
    return (model.policy, getattr(model.policy, "popart", None), b.obs, b.actions, b.action_masks,
            b.old_log_prob, b.old_values, b.advantages, b.returns, var, st)


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
def gate_regions(model: Any, *, n_envs: int, batch_size: int,
                 say: Callable[[str], None] = print) -> List[str]:
    """Hold R1 and R0 to eager at startup (module docstring). Raises `CompileTrainerError`."""
    policy = model.policy
    fe = policy.features_extractor
    precision = torch.get_float32_matmul_precision()
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
            from agents.training.instrumented_ppo.micro_step import micro_step
            comp = _r1_arm(model, model._compiled_micro_step, args)
            eager = _r1_arm(model, micro_step, args)
            ref = None
            if precision != "highest":
                with ct._matmul_precision("highest"):
                    ref = _r1_arm(model, micro_step, args)
            rules.append("R1 " + _r1_verdict(eager, comp, ref, precision, names, regime)
                         + f" [{b.source}]")
            if regime == "fresh":
                rules.append("R1 " + _r1_perturbed(model, args, precision, names))
            # ---- R0: eval / no-grad / n_envs, the decision readout
            if int(n_envs) not in ct.EAGER_BATCHES:
                rules += ["R0 " + r for r in _r0_verdicts(model, int(n_envs), precision)]
    finally:
        policy.set_training_mode(was)
        for p in policy.parameters():
            p.grad = None
    line = f"[CompileRegions] parity PASS ({precision}) — " + " | ".join(rules)
    say(line)
    return rules


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
        e = _r0_readout(model, _rollout_core, obs, mask)
    finally:
        policy.set_training_mode(was)
    bar = {"legal_logprob": ct._FP32_TOL["legal_logprob"]}
    return "fresh" if vacuous_keys({"legal_logprob": e["legal_logprob"]}, bar) else "trained"


def _r1_perturbed(model: Any, args: Tuple[Any, ...], precision: str, names: List[str]) -> str:
    """FRESH weights only: R1 again on the declared ladder's first rung — a seeded perturbation of
    every policy parameter, in place and restored bit-exactly (`parity_probe.perturbed_parameters`)
    — judged at the TRAINED bar. On fresh weights a few ill-conditioned gradients (the uniform belief
    heads) need the looser fresh bar, under which a backward defect could hide; the perturbed weights
    are trained-like and get the tight one (the extractor gate's and R0's own fresh-weights rule)."""
    from agents.model.parity_probe import ladder_at, perturbed_parameters, rung_seed
    from agents.training.instrumented_ppo.micro_step import micro_step
    scale, k = ladder_at(precision)[0]
    with perturbed_parameters(model.policy, seed=rung_seed(k), scale=scale):
        comp = _r1_arm(model, model._compiled_micro_step, args)
        eager = _r1_arm(model, micro_step, args)
        ref = None
        if precision != "highest":
            with ct._matmul_precision("highest"):
                ref = _r1_arm(model, micro_step, args)
    return _r1_verdict(eager, comp, ref, precision, names, "trained",
                       label=f"fresh weights, seeded perturbation scale={scale:g} seed+{k}")


def _r1_verdict(eager: Dict[str, torch.Tensor], comp: Dict[str, torch.Tensor],
                ref: Optional[Dict[str, torch.Tensor]], precision: str, names: List[str],
                regime: str, label: Optional[str] = None) -> str:
    e_loss, c_loss = float(eager["loss"]), float(comp["loss"])
    if not (np.isfinite(e_loss) and np.isfinite(c_loss)):
        raise ct.CompileTrainerError(f"--compile-trainer region R1: non-finite loss (eager {e_loss}, "
                                     f"compiled {c_loss})")
    if precision == "highest":
        rel = abs(c_loss - e_loss) / max(1.0, abs(e_loss))
        if not rel <= ct._MAX_NUMERIC_DRIFT:
            raise ct.CompileTrainerError(
                f"--compile-trainer region R1: the compiled micro-step's LOSS disagrees with eager — "
                f"{c_loss!r} vs {e_loss!r} (rel {rel:.2e} > {ct._MAX_NUMERIC_DRIFT:g})")
        loss_rule = f"loss rel {rel:.2e} <= {ct._MAX_NUMERIC_DRIFT:g}"
    else:
        assert ref is not None
        r_loss = float(ref["loss"])
        loss_rule = ct.check_numerics(abs(c_loss - r_loss), precision=precision,
                                      eager_err=abs(e_loss - r_loss), what="R1 loss")
    eager_t = {"features": eager["loss"], "grad": eager["grad"], "grad_sizes": eager["grad_sizes"]}
    comp_t = {"features": comp["loss"], "grad": comp["grad"], "grad_sizes": comp["grad_sizes"]}
    ref_t = (None if ref is None else
             {"features": ref["loss"], "grad": ref["grad"], "grad_sizes": ref["grad_sizes"]})
    grad_rule = ct.train_verdict(eager=eager_t, compiled=comp_t, reference=ref_t,
                                 precision=precision, allow_vacuous=True, param_names=names,
                                 param_bar=R1_PARAM_BAR[regime])
    return f"[{label or regime + ' weights'}] {loss_rule}; {grad_rule}"


def _r0_readout(model: Any, fn: Callable[..., Any], obs: Any, mask: Any) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        values, logp = fn(model.policy, obs, mask)
        legal = torch.as_tensor(mask, device=logp.device, dtype=torch.bool)
        return {"value": values.float().flatten().clone(),
                "legal_logprob": torch.where(legal, logp.float(), torch.zeros_like(logp.float())).clone()}


def _r0_verdicts(model: Any, n_envs: int, precision: str) -> List[str]:
    from agents.model.parity_probe import fresh_reason, ladder_at, perturbed_parameters, rung_seed
    from agents.model.policy import _ROLLOUT_REGIONS
    policy = model.policy
    policy.set_training_mode(False)
    obs = ct._prewarm_obs(model, n_envs)
    _, mask = ct._parity_obs(int(obs["observation"].shape[-1]), n_envs, ct.resolve_device(
        policy.features_extractor))
    comp_fn = _ROLLOUT_REGIONS[policy]

    def arms() -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        c = _r0_readout(model, comp_fn, obs, mask)
        e = _r0_readout(model, _rollout_core, obs, mask)
        r = None
        if precision != "highest":
            with ct._matmul_precision("highest"):
                r = _r0_readout(model, _rollout_core, obs, mask)
        return c, e, r

    c, e, r = arms()
    tol = {k: v for k, v in ct._FP32_TOL.items() if k in e}
    fresh = fresh_reason(e, tol)
    if fresh is None:
        return ct.decision_verdicts(eager=e, compiled=c, reference=r, precision=precision)
    for scale, k in ladder_at(precision):
        with perturbed_parameters(policy, seed=rung_seed(k), scale=scale):
            c, e, r = arms()
            if fresh_reason(e, tol) is None:
                return [f"[fresh weights, perturbation scale={scale:g} seed+{k}] " + x
                        for x in ct.decision_verdicts(eager=e, compiled=c, reference=r,
                                                      precision=precision)]
    raise ct.CompileTrainerError(f"--compile-trainer region R0: the decision readout is vacuous on the "
                                 f"weights and on every perturbation rung ({fresh}) — refusing")


# ---------------------------------------------------------------------------------- the prewarm
def prewarm_calls(model: Any, *, n_envs: int, batch_size: int) -> List[Tuple[str, Callable[[], None]]]:
    """The declared signatures, as prewarm calls (`compile_control.prewarm`): R0 at its rollout
    signature, R1 at its update signature (forward + backward), each on real rows."""
    from agents.model.policy import _ROLLOUT_REGIONS
    policy = model.policy
    calls: List[Tuple[str, Callable[[], None]]] = []

    def _r0() -> None:
        was = policy.training
        policy.set_training_mode(False)
        try:
            obs = ct._prewarm_obs(model, int(n_envs))
            _, mask = ct._parity_obs(int(obs["observation"].shape[-1]), int(n_envs),
                                     ct.resolve_device(policy.features_extractor))
            with _fatal_compile_errors(), torch.no_grad():
                _ROLLOUT_REGIONS[policy](policy, obs, mask)
        finally:
            policy.set_training_mode(was)

    def _r1() -> None:
        was = policy.training
        policy.set_training_mode(True)
        try:
            with _fatal_compile_errors():
                _r1_arm(model, model._compiled_micro_step, _r1_args(model, r1_batch(model, int(batch_size))))
        finally:
            policy.set_training_mode(was)
            policy.zero_grad(set_to_none=True)

    if int(n_envs) not in ct.EAGER_BATCHES:
        calls.append((f"R0 rollout_forward eval/no-grad B={int(n_envs)}", _r0))
    calls.append((f"R1 learner_micro_step train/grad B={int(batch_size)}", _r1))
    return calls

