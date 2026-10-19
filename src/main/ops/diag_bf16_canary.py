"""DIAGNOSTIC — the bf16 CANARY FATAL on TRAINED weights (`rb_toy_ent_s4001`, update 700, 2026-10-15; X26 BLOCKER).

THE EVENT. The toy control `rb_toy_ent_s4001` (`--arch production` + 44 ablations: entity tokens + attention, no damage
op, no move resolution; bf16 trunk; `--compile-trainer`; pinned `3899277a`) ended FATAL at 72,658,984 steps: the
in-run canary (`agents.model.compile_canary`) refused region R1 under THE bf16 PARITY RULE on three pointer-head
parameters — `pointer_head.move_proj.weight` compiled 4.03e-01 from the fp32 reference vs eager bf16's 2.82e-02 (ratio
14.3 > K = 12), `move_proj.bias` 3.26e-01 / 2.59e-02, `move_score.weight` 3.03e-01 / 2.41e-02 — confirmed on the same
rows and on the second half of the fixture in the same update. Every earlier canary passed (worst ratios 1.1-2.9).

WHAT THIS MODULE READS (designs/training/compile_flags.md "The bf16 canary FATAL on trained weights" ranks the
hypotheses each read can kill). All on the canary's OWN rows — `compile_regions.r1_batch` (the K9 learner golden's 64
committed labelled rows tiled to B = 2,048; slice 0 = the gate's rows, slice 1 = the confirmation's second half) —
through the run's own `MicroStatic`, on the run's own checkpoints, built at THIS checkout (strict load):

* ARMS (`ARMS`): E32 eager fp32 (the canary's REFERENCE), Ebf eager bf16 (its yardstick), and on CUDA Cbf compiled bf16
  (the production `compile_regions.install`), C32 compiled fp32 (a second learner at fp32), Cbf_emul compiled bf16 with
  Inductor's `emulate_precision_casts`, Cbf_aot compiled bf16 through `aot_eager` (dynamo + AOTAutograd, eager kernels).
* PER-PARAMETER DISTANCES between every pair of arms over the pointer head (`pairwise`), the canary's own statistic
  replayed (`canary_stat`: d(Cbf, E32) / max(d(Ebf, E32), floor) over every judged parameter) and its verdict through
  `compile_regions._r1_verdict_bf16` itself.
* THE FORWARD vs THE BACKWARD (`implied_dz` / `implied_read`). Every arm returns its masked log-probs, so the gradient
  of the PPO + entropy loss with respect to the logits it IMPLIES is computable from its own forward; pushed through the
  eager fp32 pointer head it must reproduce the arm's pointer-head gradient up to activation rounding. A backward that
  disagrees with its own forward (a backward miscompile) is told apart from a forward whose logits moved.
* CONDITIONING (`row_decomposition`): the pointer-head gradient split per fixture row — kappa = sum_rows |g_row| /
  |sum_rows g_row|, the factor by which cancellation between rows turns a per-row rounding error into a relative error
  of the sum — and which rows carry an arm's error. The rows are OFF-POLICY on trained weights (the golden's old
  log-probs are a near-uniform policy's), so the PPO ratios are far from 1 (`ratio_stats`).
* THE NULL (`dither_null`): N seeded multiplicative weight dithers (relative sigma 2^-9, half a bf16 ulp — it re-draws
  which way each bf16 rounding falls and moves the fp32 function by ~0.2 %), each with its OWN fp32 reference: the
  distribution of each arm's error and how often a HEALTHY pair of arms reads above the canary's K.
* ACTIVATIONS (`activation_stats`, `weight_norms`): latent / move-token magnitudes, the move_proj pre-tanh magnitude and
  saturation, the masked log-prob range, max-prob, entropy; the pointer head's weight norms — per checkpoint.

    python -m main.ops.diag_bf16_canary cpu --run rb_toy_ent_s4001 --ckpts 5M,15M,30M,45M,60M,66M,70M,72M,fatal --out D
    python -m main.ops.diag_bf16_canary gpu --run rb_toy_ent_s4001 --out D        # under the lease: scripts/ops/diag_bf16_canary.sh
    python -m main.ops.diag_bf16_canary verdict --out D                           # VERDICT.md from D/*.json

AFTER THE FIX (T36, `gen3_r1_clip_margin_v1`). The arms above still read the golden's RAW rows (the defect's
reproduction). Two reads now run PRODUCTION's fixed code instead: `relocated` (and the null's `relocated` half) moves
the rows through `compile_regions.clip_relocation` at `R1_CLIP_MARGIN`, and `canary_replay` is the canary's own
`_check`, which relocates on every rung — so at the FATAL's weights the replay must PASS, the relocated null must read
P(ratio > K) = 0 for the Ebf pairs AND for Cbf, Cbf must switch no relocated row, and `forward_vs_E32.Cbf.
log_prob_max_abs` must stay within `compile_regions.R1_LOGPROB_SHIFT_MAX` (the margin's provisional, CPU-only input).
The canary's confirmation is the two DISJOINT halves (slices 2 and 1), no longer slice 0 again + slice 1.
THE ROWS (`--buffer`, default `auto`): a FATAL is a property of a (weights, rows) state, and legacy deletion U3 rebuilt
the committed K9 golden after `rb_toy_ent_s4001`'s FATAL; `auto` reads the run's PINNED rows when they are committed
(`src/agents/model/testdata/k9_golden_buffer_<run>_pin.npz` — s4001's are), so the raw reads reproduce the FATAL's state.

INCREMENTAL: each checkpoint's read is its own JSON (`<out>/<mode>_<label>.json`), written as soon as it finishes; a
finished one is skipped on a re-run unless `--force`. A WALL BUDGET (`--budget-min`) stops starting new checkpoints
(never a semantic outcome: the summary says INCOMPLETE). Nothing here trains or writes under `models/`.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
import zipfile
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch

RUN = "rb_toy_ent_s4001"
#: the canary's micro-batch (the run's --batch-size): R1's ONE declared signature
BATCH = 2048
#: the batch THIS process reads (`--batch`; the GPU mode keeps `BATCH`, R1's declared signature). The fixture is 64
#: rows tiled, so a CPU read at 64 sees every row once: the clip decisions and the margins are EXACT, the gradient
#: differs only by the advantage normalisation's Bessel factor (std over 64 vs over 2,048 tiled rows, 0.8 %)
RUN_BATCH = BATCH
#: the parameter family the FATAL named
WATCH = "pointer_head."
#: the dither's relative sigma: half a bf16 ulp (2^-8 relative spacing)
DITHER_REL = 2.0 ** -9
FATAL_ZIP = "final_model_canary_fatal.zip"
_CKPT = re.compile(r"checkpoint_(\d+)_steps\.zip$")
#: the arms and what each is (the JSON carries this table)
ARMS: Dict[str, str] = {
    "E32": "eager, trunk fp32 (the canary's fp32 REFERENCE, `fp32_scope`)",
    "Ebf": "eager, trunk bf16 (the canary's yardstick)",
    "Cbf": "compiled (Inductor), trunk bf16 — the production compile (`compile_regions.install`)",
    "C32": "compiled (Inductor), trunk fp32 — a second learner at fp32, the same weights",
    "Cbf_emul": "compiled (Inductor) bf16 with `emulate_precision_casts` (rounds where eager rounds)",
    "Cbf_aot": "compiled bf16 through `aot_eager` (dynamo + AOTAutograd, eager kernels: no Inductor codegen)",
    "Cbf_pheager": "compiled (Inductor) bf16 with the POINTER HEAD run eager (`torch.compiler.disable` on its forward: "
                   "a graph break, fullgraph=False) — a third learner, the same weights",
}
CPU_ARMS = ("E32", "Ebf")
#: the localisation arms: one that fails to compile or run is RECORDED (`failed_arms`) and dropped; E32 / Ebf / Cbf
#: are the read itself and stop it
OPTIONAL_ARMS = frozenset({"C32", "Cbf_emul", "Cbf_aot", "Cbf_pheager"})
GPU_ARMS = ("E32", "Ebf", "Cbf", "C32", "Cbf_emul", "Cbf_aot", "Cbf_pheager")
DEFAULT_CPU_CKPTS = "5M,15M,30M,45M,60M,66M,70M,72M,fatal"
DEFAULT_GPU_CKPTS = "fatal,72M,70M,66M,60M,45M,30M,15M"


class DiagError(RuntimeError):
    """A precondition of the diagnostic does not hold (the read would not mean what it says)."""


# ----------------------------------------------------------------------------------------- the run and its weights
def resolve_run(spec: str) -> Path:
    """A run NAME (under the main checkout's `models/`) or a run DIRECTORY."""
    p = Path(spec)
    if p.is_dir():
        return p.resolve()
    from utils.paths import main_models_dir
    m = main_models_dir()
    if m is None or not (m / spec).is_dir():
        raise DiagError(f"no run {spec!r} (models/ archive: {m})")
    return (m / spec).resolve()


def checkpoint_steps(run: Path) -> Dict[int, Path]:
    """{steps: zip} of `<run>/checkpoints/checkpoint_<N>_steps.zip`."""
    out: Dict[int, Path] = {}
    ck = run / "checkpoints"
    if ck.is_dir():
        for p in ck.iterdir():
            m = _CKPT.search(p.name)
            if m:
                out[int(m.group(1))] = p
    return out


def resolve_checkpoint(run: Path, spec: str) -> Tuple[str, Path]:
    """``(label, zip)`` for a checkpoint SPEC: ``fatal`` (the canary's own weights, `final_model_canary_fatal.zip`),
    ``<x>M`` (the checkpoint NEAREST x million steps, within 0.6M — a further one is refused, never substituted) or a
    path to a zip."""
    s = spec.strip()
    if s == "fatal":
        p = run / FATAL_ZIP
        if not p.is_file():
            raise DiagError(f"{run.name} has no {FATAL_ZIP}")
        return "fatal", p
    m = re.fullmatch(r"(\d+(?:\.\d+)?)M", s)
    if m:
        want = float(m.group(1)) * 1e6
        steps = checkpoint_steps(run)
        if not steps:
            raise DiagError(f"{run.name} has no checkpoints/")
        best = min(steps, key=lambda n: abs(n - want))
        if abs(best - want) > 0.6e6:
            raise DiagError(f"no checkpoint of {run.name} within 0.6M steps of {s} (nearest {best:,})")
        return s, steps[best]
    p = Path(s)
    if p.is_file():
        return p.stem, p
    raise DiagError(f"checkpoint spec {spec!r}: not 'fatal', '<x>M' or a zip path")


def policy_state(zip_path: Path) -> Dict[str, torch.Tensor]:
    with zipfile.ZipFile(zip_path) as z:
        sd: Dict[str, torch.Tensor] = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
    return sd


#: set by `--allow-fp32-run`: judge an fp32-TRAINED run's weights under the bf16 rule (the read
#: `verify_compile_gate_cuda.sh` makes on the `production` baseline; a bf16 fork of it meets exactly this)
ALLOW_FP32_RUN = False


def build_learner(zip_path: Path, device: str, *, precision: str = "bf16", batch: Optional[int] = None) -> Any:
    """The run's learner AT THIS CHECKOUT: the trainer's own resume resolution (`--model <zip>`: every flag inherited
    from the run's `model_config.json`), the learner golden's seeded build over that namespace, the checkpoint's
    policy loaded STRICT, the trunk at ``precision``, moved to ``device``, `batch_size` = the canary's B."""
    from agents.model.trunk_precision import set_trunk_precision, trunk_precision_of
    from agents.training import learner_golden as LG
    from main.ops.cuda_build_validate import resolved_args
    args = resolved_args(["--model", str(zip_path)], "cpu")
    if getattr(args, "trunk_precision", None) != "bf16" and not ALLOW_FP32_RUN:
        raise DiagError(f"{zip_path}: the run's resolved --trunk-precision is {getattr(args, 'trunk_precision', None)!r}, "
                        "not bf16 — this diagnostic is about the bf16 rule (--allow-fp32-run judges its weights "
                        "under the bf16 rule anyway)")
    model = LG.build_learner(args=args)
    set_trunk_precision(model.policy, precision)
    if trunk_precision_of(model.policy) != precision:
        raise DiagError(f"the learner's trunk is {trunk_precision_of(model.policy)}, wanted {precision}")
    model.policy.load_state_dict(policy_state(zip_path), strict=True)
    dev = torch.device(device)
    model.policy.to(dev)
    model.device = dev
    model.rollout_buffer.device = dev
    model.batch_size = int(batch or RUN_BATCH)
    model.policy.set_training_mode(True)
    return model


def load_weights(models: Sequence[Any], zip_path: Path) -> None:
    sd = policy_state(zip_path)
    for m in models:
        m.policy.load_state_dict(sd, strict=True)


# ------------------------------------------------------------------------------------------------------------ arms
class ArmOut(NamedTuple):
    loss: float
    grad: torch.Tensor         # flat, every policy parameter (`grad_parameters` order), float32 on the CPU
    logp: torch.Tensor         # [B, 11] the masked normalised log-probs the arm computed
    log_prob: torch.Tensor     # [B]
    entropy: torch.Tensor      # [B]
    adv: torch.Tensor          # [B] the normalised advantages


def param_table(model: Any) -> Tuple[List[str], List[int]]:
    from agents.model import compile_trainer as ct
    named = ct.grad_parameters(model, model.policy.features_extractor)
    return [n for n, _ in named], [int(p.numel()) for _, p in named]


def run_arm(model: Any, fn: Callable[..., Any], call_args: Tuple[Any, ...], *, fp32_trunk: bool) -> ArmOut:
    """One R1 call + backward (as `compile_regions._r1_arm`), keeping the forward's outputs. ``fp32_trunk``: run under
    `fp32_scope` (the reference arm; NEVER with a compiled ``fn``, which traced the bf16 classes)."""
    from agents.model import compile_trainer as ct
    from agents.model.trunk_precision import fp32_scope
    params = [p for _, p in ct.grad_parameters(model, model.policy.features_extractor)]
    for p in params:
        p.grad = None
    scope = fp32_scope(model.policy) if fp32_trunk else contextlib.nullcontext()
    with scope:
        out = fn(*call_args)
        out.loss.backward()
    gs = [(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten().cpu() for p in params]
    for p in params:
        p.grad = None
    return ArmOut(float(out.loss.detach().float()), torch.cat(gs), out.logp.detach().float().cpu().clone(),
                  out.log_prob.detach().float().cpu().clone(), out.entropy.detach().float().cpu().clone(),
                  out.advantages.detach().float().cpu().clone())


def split(flat: torch.Tensor, names: Sequence[str], sizes: Sequence[int]) -> Dict[str, torch.Tensor]:
    return dict(zip(names, torch.split(flat, list(sizes))))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / ||b|| in float64; NaN when ``b`` is zero."""
    a64, b64 = a.detach().double().flatten(), b.detach().double().flatten()
    nb = float(b64.norm())
    return float((a64 - b64).norm()) / nb if nb > 0 else float("nan")


def pairwise(arms: Dict[str, ArmOut], names: Sequence[str], sizes: Sequence[int],
             prefix: str = WATCH) -> Dict[str, Dict[str, float]]:
    """``{"X|Y": {param: d(X, Y) = ||X_p - Y_p|| / ||Y_p||}}`` over every ordered pair (X != Y) and every parameter
    named ``prefix*``, plus ``"_global"`` (the whole flat gradient)."""
    sp = {k: split(v.grad, names, sizes) for k, v in arms.items()}
    out: Dict[str, Dict[str, float]] = {}
    for x in arms:
        for y in arms:
            if x == y:
                continue
            row = {n: rel_l2(sp[x][n], sp[y][n]) for n in names if n.startswith(prefix)}
            row["_global"] = rel_l2(arms[x].grad, arms[y].grad)
            out[f"{x}|{y}"] = row
    return out


def canary_stat(comp: ArmOut, eager: ArmOut, ref: ArmOut, names: Sequence[str], sizes: Sequence[int],
                regime: str) -> Dict[str, Any]:
    """THE CANARY'S PER-PARAMETER STATISTIC, exactly as `compile_regions._r1_verdict_bf16` computes it (the same
    judged set, floor and K; unmoved-parameter bar overrides are not modelled — none is on the pointer head):
    ``{"k", "worst": [...top 8 by ratio], "watch": {param: {...}}, "over_k": [names]}``."""
    from agents.model import compile_regions as cr
    from agents.model import compile_trainer as ct
    from agents.model.compile_gate_probe import per_param_grad_errors
    k = float(cr.BF16_ENVELOPE_K["param"]) if cr.BF16_ENVELOPE_K else float("nan")
    floor = float(cr.R1_PARAM_BAR[regime])
    err_c = dict(per_param_grad_errors(comp.grad, ref.grad, list(sizes), floor_frac=ct._PARAM_GRAD_FLOOR))
    err_e = dict(per_param_grad_errors(eager.grad, ref.grad, list(sizes), floor_frac=ct._PARAM_GRAD_FLOOR))
    rows = []
    for i, dc in err_c.items():
        de = err_e[i]
        rows.append({"param": names[i], "compiled": dc, "eager": de, "floor": floor, "ratio": dc / max(de, floor)})
    rows.sort(key=lambda r: -r["ratio"])
    return {"k": k, "floor": floor, "judged": len(rows), "worst": rows[:8],
            "watch": {r["param"]: r for r in rows if r["param"].startswith(WATCH)},
            "over_k": [r["param"] for r in rows if not r["ratio"] <= k]}


def canary_verdict(comp: ArmOut, eager: ArmOut, ref: ArmOut, names: Sequence[str], sizes: Sequence[int],
                   regime: str) -> Dict[str, str]:
    """The bf16 rule ITSELF (`compile_regions._r1_verdict_bf16`) on these arms: ``{"verdict": "pass"|"refused",
    "text": the rule line or the refusal}``."""
    from agents.model import compile_regions as cr
    from agents.model import compile_trainer as ct
    gs = torch.tensor([int(x) for x in sizes], dtype=torch.long)

    def d(a: ArmOut) -> Dict[str, torch.Tensor]:
        return {"loss": torch.tensor([a.loss]), "grad": a.grad, "grad_sizes": gs}
    try:
        return {"verdict": "pass", "text": cr._r1_verdict_bf16(d(eager), d(comp), d(ref), list(names), regime)}
    except ct.CompileTrainerError as exc:
        return {"verdict": "refused", "text": str(exc)}


# --------------------------------------------------------------------------------- the forward vs the backward
def ppo_active(arm: ArmOut, old_log_prob: torch.Tensor, st: Any) -> torch.Tensor:
    """[B] bool: the rows whose PPO clipped surrogate passes a gradient (`micro_step`'s min(A rho, A clip(rho)):
    rho inside [1 - clip, 1 + clip], or the unclipped branch is the smaller one). A row whose rho sits within
    rounding of 1 +/- clip with the advantage's sign pointing out of the range FLIPS between carrying its full
    gradient and carrying none — a decision boundary on the gradient (standing rule 8)."""
    a_n = arm.adv.double()
    rho = (arm.log_prob.double() - old_log_prob.detach().double().cpu()).exp()
    lo, hi = 1.0 - float(st.clip_range), 1.0 + float(st.clip_range)
    inside = (rho >= lo) & (rho <= hi)
    return inside | (a_n * rho < a_n * rho.clamp(lo, hi))


def implied_dz(arm: ArmOut, actions: torch.Tensor, old_log_prob: torch.Tensor, masks: torch.Tensor,
               st: Any, active: Optional[torch.Tensor] = None) -> torch.Tensor:
    """dL/dz [B, 11] (float64) — the gradient of R1's POLICY + ENTROPY terms with respect to the pointer head's
    logits, IMPLIED by the arm's own forward (its masked log-probs, its normalised advantages): the PPO clipped
    surrogate (`micro_step`: -mean(min(A rho, A clip(rho))), rho = exp(log pi - old)) and -ent_coef * mean(H). Only
    those terms reach the pointer head (the critic and the win-prob head read the value features). ``active``: hold
    the clip branch to another arm's decisions (`ppo_active`) — the FROZEN-BRANCH read."""
    if st.fork_pg_mask:
        raise DiagError("the fork policy mask is on: the implied-gradient formula does not model it")
    legal = masks.bool().cpu()
    logp = arm.logp.double()
    p = torch.where(legal, logp.exp(), torch.zeros_like(logp))
    lp = torch.where(legal, logp, torch.zeros_like(logp))
    h = -(p * lp).sum(-1)
    a_n = arm.adv.double()
    rho = (arm.log_prob.double() - old_log_prob.detach().double().cpu()).exp()
    if active is None:
        active = ppo_active(arm, old_log_prob, st)
    b = float(logp.shape[0])
    coef = -(a_n * rho * active.double()) / b * float(st.policy_grad_coef)
    onehot = torch.nn.functional.one_hot(actions.long().reshape(-1).cpu(), logp.shape[-1]).double()
    dz = coef[:, None] * (onehot - p) + (float(st.ent_coef) / b) * p * (lp + h[:, None])
    return torch.where(legal, dz, torch.zeros_like(dz))


def head_inputs(model: Any, obs: Dict[str, torch.Tensor], *, fp32_trunk: bool) -> Tuple[torch.Tensor, ...]:
    """The pointer head's six inputs (latent_pi, request-order move tokens, move_valid, team tokens, move cells,
    switch cells), EAGER, no grad, at the trunk precision asked."""
    from agents.model.trunk_precision import fp32_scope
    pol = model.policy
    scope = fp32_scope(pol) if fp32_trunk else contextlib.nullcontext()
    with scope, torch.no_grad():
        pi, _ = pol.extract_features(obs)
        lat = pol.mlp_extractor.forward_actor(pi)
        tok, valid, team, mcells, scells = pol.features_extractor.last_pointer_inputs
    return tuple(t.detach() for t in (lat, tok, valid, team, mcells, scells))


def head_names(model: Any) -> List[str]:
    return [WATCH + n for n, _ in model.policy.pointer_head.named_parameters()]


def head_grads(model: Any, inputs: Tuple[torch.Tensor, ...], dzs: Sequence[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    """The pointer head's parameter gradient for each injected logit gradient in ``dzs`` (one fp32 forward of the head
    on ``inputs``, one `autograd.grad` per dz): ``[{param: grad}]``."""
    ph = model.policy.pointer_head
    named = list(ph.named_parameters())
    params = [p for _, p in named]
    out: List[Dict[str, torch.Tensor]] = []
    with torch.enable_grad():
        logits = ph(*inputs)
        for i, dz in enumerate(dzs):
            gs = torch.autograd.grad(logits, params, grad_outputs=dz.to(logits.dtype).to(logits.device),
                                     retain_graph=i + 1 < len(dzs), allow_unused=True)
            out.append({WATCH + n: (g if g is not None else torch.zeros_like(p)).detach().double().cpu()
                        for (n, p), g in zip(named, gs)})
    return out


def implied_read(model: Any, arms: Dict[str, ArmOut], call_args: Tuple[Any, ...], names: Sequence[str],
                 sizes: Sequence[int]) -> Dict[str, Any]:
    """For every arm X: G(dz_X) = the eager fp32 head's gradient under X's IMPLIED logit gradient, and
    * ``self``: d(G(dz_X), grad_X) per head parameter — small when X's backward is consistent with its own forward
      (only the activations differ); LARGE with a small ``dz`` difference is a BACKWARD defect;
    * ``vs_ref``: d(G(dz_X), G(dz_E32)) — how far X's FORWARD alone moves the head's gradient;
    * ``dz_vs_ref``: ||dz_X - dz_E32|| / ||dz_E32|| (the logit-gradient difference itself).
    ``validity``: d(G(dz_E32), grad_E32) — the formula's own check (must be ~1e-6; above 1e-3 the read is INVALID)."""
    st = call_args[-1]
    obs, actions, masks, old = call_args[1], call_args[2], call_args[3], call_args[4]
    inputs = head_inputs(model, obs, fp32_trunk=True)
    order = list(arms)
    act_ref = ppo_active(arms["E32"], old, st)
    dzs = [implied_dz(arms[k], actions, old, masks, st) for k in order]
    dzf = [implied_dz(arms[k], actions, old, masks, st, active=act_ref) for k in order]
    allg = head_grads(model, inputs, dzs + dzf)
    gs = dict(zip(order, allg[:len(order)]))
    gf = dict(zip(order, allg[len(order):]))
    dz_ref = dzs[order.index("E32")]
    judged = judged_params(arms["E32"], names, sizes)
    out: Dict[str, Any] = {"judged": [n for n in gs["E32"] if n in judged]}
    for k, dz in zip(order, dzs):
        g = split(arms[k].grad, names, sizes)
        flips = torch.nonzero(ppo_active(arms[k], old, st) != act_ref).flatten()
        out[k] = {"self": {n: rel_l2(gs[k][n], g[n]) for n in gs[k] if n in judged},
                  "vs_ref": {n: rel_l2(gs[k][n], gs["E32"][n]) for n in gs[k] if n in judged},
                  "frozen_branch_vs_ref": {n: rel_l2(gf[k][n], gf["E32"][n]) for n in gf[k] if n in judged},
                  "dz_vs_ref": rel_l2(dz, dz_ref),
                  "branch_flips": {"batch_rows": int(flips.numel()),
                                   "rows": _flip_rows(arms[k], arms["E32"], old, st, flips)}}
    worst = max(out["E32"]["self"].values()) if out["E32"]["self"] else float("nan")
    out["validity"] = {"worst_self_E32": worst, "valid": bool(worst <= 1e-3),
                       "note": "over the canary's judged head parameters (an unjudged one, e.g. a never-legal "
                               "struggle scorer, carries a rounding-residual gradient the formula sets to 0)"}
    return out


def _flip_rows(arm: ArmOut, ref: ArmOut, old: torch.Tensor, st: Any, flips: torch.Tensor,
               cap: int = 8) -> List[Dict[str, Any]]:
    """The first ``cap`` DISTINCT flipped rows (dedup by the reference's rho: the fixture's rows are tiled)."""
    out: List[Dict[str, Any]] = []
    seen = set()
    rho_r = (ref.log_prob.double() - old.detach().double().cpu()).exp()
    rho_x = (arm.log_prob.double() - old.detach().double().cpu()).exp()
    for b in flips.tolist():
        key = round(float(rho_r[b]), 9)
        if key in seen:
            continue
        seen.add(key)
        out.append({"batch_row": int(b), "adv_norm": float(ref.adv[b]), "rho_ref": float(rho_r[b]),
                    "rho_arm": float(rho_x[b]), "clip_bounds": [1.0 - float(st.clip_range), 1.0 + float(st.clip_range)]})
        if len(out) >= cap:
            break
    return out


def judged_params(ref: ArmOut, names: Sequence[str], sizes: Sequence[int]) -> set:
    """The canary's judged set: every parameter whose reference gradient norm is above `_PARAM_GRAD_FLOOR` x the
    largest per-parameter norm (`compile_gate_probe.per_param_grad_errors`)."""
    from agents.model import compile_trainer as ct
    norms = {n: float(g.double().norm()) for n, g in split(ref.grad, names, sizes).items()}
    top = max(norms.values()) if norms else 0.0
    return {n for n, v in norms.items() if top > 0 and v > ct._PARAM_GRAD_FLOOR * top}


# --------------------------------------------------------------------------------------- conditioning (per row)
def _row_groups(n_rows_batch: int, slice_: int, n_fixture: int) -> Dict[int, torch.Tensor]:
    from agents.model.compile_trainer import fixture_index
    idx = torch.as_tensor(fixture_index(n_rows_batch, n_fixture, slice_))
    return {int(u): (idx == u) for u in torch.unique(idx).tolist()}


def row_decomposition(model: Any, arms: Dict[str, ArmOut], call_args: Tuple[Any, ...], slice_: int,
                      n_fixture: int, top: int = 5) -> Dict[str, Any]:
    """The pointer head's gradient (E32's implied one) split per FIXTURE ROW: per head parameter, kappa =
    sum_u |g_u| / |sum_u g_u| (pg + entropy, and the policy term alone), the top rows' shares with the row's facts,
    and for every other arm X the rows that carry its error: |G_u(dz_X) - G_u(dz_E32)| / |sum_u G_u(dz_E32)|."""
    st = call_args[-1]
    obs, actions, masks, old = call_args[1], call_args[2], call_args[3], call_args[4]
    groups = _row_groups(int(actions.shape[0]), slice_, n_fixture)
    inputs = head_inputs(model, obs, fp32_trunk=True)
    st_pg = st._replace(ent_coef=0.0)
    dz_ref = implied_dz(arms["E32"], actions, old, masks, st)
    dz_pg = implied_dz(arms["E32"], actions, old, masks, st_pg)
    others = [k for k in arms if k != "E32"]
    dz_x = {k: implied_dz(arms[k], actions, old, masks, st) for k in others}
    us = sorted(groups)
    dzs: List[torch.Tensor] = []
    for u in us:
        m = groups[u][:, None].double()
        dzs += [dz_ref * m, dz_pg * m] + [(dz_x[k] - dz_ref) * m for k in others]
    per = head_grads(model, inputs, dzs)
    stride = 2 + len(others)
    facts = _row_facts(arms["E32"], actions, old, masks, st, groups)
    out: Dict[str, Any] = {"rows": len(us), "params": {}}
    for n in per[0]:
        g_u = torch.stack([per[i * stride][n].flatten() for i in range(len(us))])
        g_pg = torch.stack([per[i * stride + 1][n].flatten() for i in range(len(us))])
        tot = g_u.sum(0)
        nt = float(tot.norm())
        norms = g_u.norm(dim=1)
        kappa = float(norms.sum()) / nt if nt > 0 else float("inf")
        kappa_pg = float(g_pg.norm(dim=1).sum()) / max(float(g_pg.sum(0).norm()), 1e-300)
        order = torch.argsort(norms, descending=True)[:top].tolist()
        rec: Dict[str, Any] = {"kappa": kappa, "kappa_policy_term": kappa_pg, "sum_norm": nt,
                               "top_rows": [{"row": us[i], "share_of_abs": float(norms[i] / norms.sum()),
                                             **facts[us[i]]} for i in order], "error_rows": {}}
        for j, k in enumerate(others):
            e_u = torch.stack([per[i * stride + 2 + j][n].flatten() for i in range(len(us))])
            en = e_u.norm(dim=1) / max(nt, 1e-300)
            eo = torch.argsort(en, descending=True)[:top].tolist()
            rec["error_rows"][k] = {"total_rel": float(e_u.sum(0).norm()) / max(nt, 1e-300),
                                    "rows": [{"row": us[i], "rel_contribution": float(en[i]), **facts[us[i]]}
                                             for i in eo]}
        out["params"][n] = rec
    return out


def _row_facts(arm: ArmOut, actions: torch.Tensor, old: torch.Tensor, masks: torch.Tensor, st: Any,
               groups: Dict[int, torch.Tensor]) -> Dict[int, Dict[str, Any]]:
    legal = masks.bool().cpu()
    a = actions.long().reshape(-1).cpu()
    rho = (arm.log_prob.double() - old.detach().double().cpu()).exp()
    lo, hi = 1.0 - float(st.clip_range), 1.0 + float(st.clip_range)
    adv = arm.adv.double()
    out: Dict[int, Dict[str, Any]] = {}
    for u, m in groups.items():
        b = int(torch.nonzero(m)[0])
        lp = arm.logp[b].double()
        lg = legal[b]
        active = bool((lo <= rho[b] <= hi) or (adv[b] * rho[b] < adv[b] * rho[b].clamp(lo, hi)))
        out[u] = {"action": int(a[b]), "kind": "switch" if int(a[b]) < 6 else ("move" if int(a[b]) < 10 else "struggle"),
                  "n_legal": int(lg.sum()), "adv_norm": float(adv[b]), "rho": float(rho[b]), "active": active,
                  "logp_action": float(lp[a[b]]), "max_p": float(lp[lg].exp().max()),
                  "legal_logp_range": float(lp[lg].max() - lp[lg].min()), "tiles": int(m.sum())}
    return out


def ratio_stats(arm: ArmOut, old: torch.Tensor, st: Any, uniq: Sequence[int],
                fixture_rows: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """The PPO ratio on the fixture's UNIQUE rows (``uniq``: one batch index per row): rho = exp(log pi - the
    golden's old log-prob) — off-policy on trained weights — the clipped share, the rows near a clip boundary
    (within 1 % of 1 +/- clip_range), the active (gradient-carrying) rows."""
    ix = torch.as_tensor(list(uniq))
    rho = (arm.log_prob.double() - old.detach().double().cpu()).exp()[ix]
    adv = arm.adv.double()[ix]
    lo, hi = 1.0 - float(st.clip_range), 1.0 + float(st.clip_range)
    inside = (rho >= lo) & (rho <= hi)
    active = inside | (adv * rho < adv * rho.clamp(lo, hi))
    near = ((rho / lo - 1).abs() < 0.01) | ((rho / hi - 1).abs() < 0.01)
    # the boundary that SWITCHES a row's gradient: A > 0 at 1 + clip, A < 0 at 1 - clip (the other one never does);
    # the margin is in log-prob units, the currency of the arms' forward differences (`forward_vs_E32`)
    b = torch.where(adv > 0, torch.full_like(rho, hi), torch.full_like(rho, lo))
    margin = (rho.log() - b.log()).abs()
    order = torch.argsort(margin)[:5].tolist()
    q = [0.0, 0.1, 0.5, 0.9, 1.0]
    return {"rows": int(rho.numel()), "rho_quantiles": dict(zip(map(str, q), np.quantile(rho.numpy(), q).tolist())),
            "inside_clip": int(inside.sum()), "active": int(active.sum()), "near_boundary": int(near.sum()),
            "max_active_abs_adv_rho": float((adv.abs() * rho * active.double()).max()),
            "switching_margin_logp": [{"fixture_row": int(fixture_rows[i]) if fixture_rows else None,
                                       "batch_row": int(uniq[i]), "margin": float(margin[i]), "rho": float(rho[i]),
                                       "adv_norm": float(adv[i]), "active": bool(active[i])} for i in order]}


#: THE CANARY FIX's margin, in log-prob units — PRODUCTION's (`compile_regions.R1_CLIP_MARGIN`, T36: R1_BAR_K x the
#: largest |log pi_bf16 - log pi_fp32| measured; the GPU read records the compiled arms' own in `forward_vs_E32`). The
#: diagnostic's first reads (2026-10-15, before T36) used 0.5.
def _relocate_margin() -> float:
    from agents.model import compile_regions as cr
    return float(cr.R1_CLIP_MARGIN)


def relocate_old_log_prob(ref: ArmOut, old: torch.Tensor, st: Any,
                          margin: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """THE CANARY FIX (T36, standing rule 8), applied to a COPY of the batch through PRODUCTION's own function
    (`compile_regions.clip_relocation`, which the startup gate and the canary run on every rung): every row whose fp32
    REFERENCE ratio sits within ``margin`` (log-prob units; default production's) of the clip boundary that switches its
    gradient (A > 0: 1 + clip; A < 0: 1 - clip) gets its old log-prob moved so the reference ratio sits 2 x ``margin``
    inside the ACTIVE side. A row far from its boundary keeps the golden's old log-prob. Returns (the new old log-probs
    on ``old``'s device, {"moved_batch_rows", "margin", "nearest_unmoved"})."""
    from agents.model import compile_regions as cr
    m = _relocate_margin() if margin is None else float(margin)
    fix = cr.clip_relocation(ref.log_prob, old, ref.adv, float(st.clip_range), m)
    return fix.old_log_prob, {"moved_batch_rows": int(fix.moved.sum()), "margin": m,
                              "nearest_unmoved": fix.nearest_unmoved}


def relocated_read(ctx: "Context", slice_: int, arms: Dict[str, ArmOut], call_args: Tuple[Any, ...],
                   regime: str) -> Tuple[Dict[str, Any], Dict[str, ArmOut]]:
    """Every arm again on the RELOCATED batch: the canary's statistic and verdict, the branch flips (expected 0) and
    the watched parameters' distances — the T36 fix (production's relocation), validated on the same weights."""
    st = call_args[-1]
    new_old, info = relocate_old_log_prob(arms["E32"], call_args[4], st)
    arms_r, args_r = ctx.run_arms(slice_, old_override=new_old)
    act = ppo_active(arms_r["E32"], args_r[4], st)
    out: Dict[str, Any] = {"info": info,
                           "branch_flips": {a: int((ppo_active(v, args_r[4], st) != act).sum())
                                            for a, v in arms_r.items() if a != "E32"},
                           "pairwise": {k: v for k, v in pairwise(arms_r, ctx.names, ctx.sizes).items()
                                        if k.endswith("|E32") or k.endswith("|Ebf")}}
    for comp in [a for a in ("Cbf", "Cbf_emul", "Cbf_aot", "Cbf_pheager") if a in arms_r]:
        out.setdefault("canary_stat", {})[comp] = canary_stat(arms_r[comp], arms_r["Ebf"], arms_r["E32"],
                                                               ctx.names, ctx.sizes, regime)
    if "Cbf" in arms_r:
        out["canary_verdict"] = canary_verdict(arms_r["Cbf"], arms_r["Ebf"], arms_r["E32"], ctx.names, ctx.sizes,
                                               regime)
    return out, arms_r


# ------------------------------------------------------------------------------------------- activations, weights
def _q(t: torch.Tensor, qs: Sequence[float] = (0.5, 0.9, 0.99, 1.0)) -> Dict[str, float]:
    v = t.detach().double().flatten().cpu().numpy()
    return {str(q): float(np.quantile(v, q)) for q in qs} if v.size else {}


def activation_stats(model: Any, obs: Dict[str, torch.Tensor], masks: torch.Tensor, uniq: Sequence[int], *,
                     fp32_trunk: bool) -> Dict[str, Any]:
    """The pointer head's inputs and internals on the fixture's unique rows (eager, no grad)."""
    from agents.model import masked_categorical as mc
    lat, tok, valid, team, mcells, scells = head_inputs(model, obs, fp32_trunk=fp32_trunk)
    ph = model.policy.pointer_head
    ix = torch.as_tensor(list(uniq), device=lat.device)
    with torch.no_grad():
        c = ph.ctx_proj(lat)
        x_move = ph.move_proj(torch.cat([tok, mcells], dim=-1)) + c[:, None, :]
        s_in = torch.cat([team, scells[..., :ph._switch_base_dim]], dim=-1)
        x_sw = ph.switch_proj(s_in) + c[:, None, :]
        logits = ph(lat, tok, valid, team, mcells, scells)
        logp = mc.masked_logits(logits, masks, int(logits.shape[-1]))
    legal = masks.bool()[ix]
    lgt = logits[ix]
    lp = logp[ix]
    v = valid[ix].bool()
    xm = x_move[ix][v]
    big = torch.finfo(lp.dtype).max
    lp_max = torch.where(legal, lp, torch.full_like(lp, -big)).max(-1).values
    lp_min = torch.where(legal, lp, torch.full_like(lp, big)).min(-1).values
    p = torch.where(legal, lp.exp(), torch.zeros_like(lp))
    ent = -(p * torch.where(legal, lp, torch.zeros_like(lp))).sum(-1)
    return {
        "latent_pi_abs": _q(lat[ix].abs()), "move_token_abs": _q(tok[ix][v].abs()),
        "move_preact_abs": _q(xm.abs()), "move_saturated_gt3": float((xm.abs() > 3).double().mean()),
        "move_saturated_gt5": float((xm.abs() > 5).double().mean()),
        "switch_preact_abs": _q(x_sw[ix].abs()), "switch_saturated_gt3": float((x_sw[ix].abs() > 3).double().mean()),
        "legal_logit_abs": _q(lgt[legal].abs()), "legal_logp_range": _q(lp_max - lp_min),
        "max_p": _q(lp_max.exp(), (0.1, 0.5, 0.9, 1.0)), "entropy": _q(ent, (0.0, 0.1, 0.5, 0.9)),
    }


def weight_norms(model: Any) -> Dict[str, Dict[str, float]]:
    return {WATCH + n: {"norm": float(p.detach().double().norm()), "max_abs": float(p.detach().abs().max())}
            for n, p in model.policy.pointer_head.named_parameters()}


# ------------------------------------------------------------------------------------------------- the null
@contextlib.contextmanager
def dithered(models: Sequence[Any], seed: int, rel: float = DITHER_REL) -> Iterator[None]:
    """Every policy parameter of every model in ``models`` (identical weights) multiplied by (1 + rel * N(0, 1)) —
    the SAME name-keyed noise on each — and restored BIT-EXACTLY on exit."""
    saved = [{n: p.detach().clone() for n, p in m.policy.named_parameters()} for m in models]
    try:
        with torch.no_grad():
            for m in models:
                for n, p in m.policy.named_parameters():
                    g = torch.Generator().manual_seed((zlib.crc32(n.encode()) ^ (seed * 2654435761)) & 0x7FFFFFFF)
                    noise = torch.randn(p.shape, generator=g, dtype=torch.float32)
                    p.mul_(1.0 + rel * noise.to(p.device))
        yield
    finally:
        with torch.no_grad():
            for m, sv in zip(models, saved):
                for n, p in m.policy.named_parameters():
                    p.copy_(sv[n])


def null_summary(per_seed: List[Dict[str, Dict[str, float]]], k: float, floor: float) -> Dict[str, Any]:
    """From per-seed ``{arm: {param: d(arm, E32_seed)}}``: per arm and head parameter the error quantiles; and the
    canary statistic's null — for the ``Ebf`` pairs (i != j: d_Ebf_i / max(d_Ebf_j, floor)), and when ``Cbf`` is
    present its own per-seed statistic d_Cbf_i / max(d_Ebf_i, floor) — with the share above ``k``. A parameter a seed
    does not carry (`_dither` records only the parameters the canary JUDGES on that seed: above the per-parameter
    floor of its own reference) is left out of that seed's numbers — the canary never compares it there."""
    out: Dict[str, Any] = {"seeds": len(per_seed), "k": k, "floor": floor, "errors": {}, "ratio_null": {}}
    if not per_seed:
        return out
    arms = list(per_seed[0])
    params = sorted({n for s in per_seed for a in s for n in s[a]})
    for a in arms:
        out["errors"][a] = {n: _q(torch.tensor(v), (0.0, 0.1, 0.5, 0.9, 1.0)) for n in params
                            if (v := [s[a][n] for s in per_seed if n in s.get(a, {})])}
    for n in params:
        e = [s["Ebf"][n] for s in per_seed if n in s.get("Ebf", {})] if "Ebf" in arms else []
        pairs = [e[i] / max(e[j], floor) for i in range(len(e)) for j in range(len(e)) if i != j]
        rec: Dict[str, Any] = {"seeds_judged": len(e)}
        if pairs:
            rec["Ebf_pairs"] = {"n": len(pairs), "p_over_k": float(np.mean(np.array(pairs) > k)),
                                "quantiles": _q(torch.tensor(pairs), (0.5, 0.9, 0.99, 1.0))}
        if "Cbf" in arms:
            r = [s["Cbf"][n] / max(s["Ebf"][n], floor) for s in per_seed
                 if n in s.get("Cbf", {}) and n in s.get("Ebf", {})]
            if r:
                rec["Cbf_vs_Ebf"] = {"n": len(r), "p_over_k": float(np.mean(np.array(r) > k)),
                                     "quantiles": _q(torch.tensor(r), (0.5, 0.9, 0.99, 1.0))}
        out["ratio_null"][n] = rec
    return out


# ------------------------------------------------------------------------------------------------ the context
class Context:
    """The learners and the arm callables of one process (CPU: E32 / Ebf; CUDA: every arm in `GPU_ARMS`)."""

    def __init__(self, first_zip: Path, device: str, arms: Sequence[str], say: Callable[[str], None],
                 smoke_backend: Optional[str] = None) -> None:
        """``smoke_backend`` (CPU only, a CODE-PATH smoke): every compiled arm through that dynamo backend (e.g.
        ``eager``) instead of Inductor — the compiled arms then equal eager, so their numbers mean nothing; what it
        proves is that every route of the GPU read RUNS before a GPU window is spent on it."""
        if smoke_backend and device == "cuda":
            raise DiagError("--smoke-backend is a CPU code-path smoke; the GPU read compiles with Inductor")
        self.device = device
        self.smoke_backend = smoke_backend
        self.say = say
        t0 = time.perf_counter()
        self.model = build_learner(first_zip, device)
        self.names, self.sizes = param_table(self.model)
        self.st = None
        self.arm_names = list(arms)
        self.model32: Optional[Any] = None
        self.model_ph: Optional[Any] = None
        self.failed: Dict[str, str] = {}
        self.fns: Dict[str, Tuple[Any, Callable[..., Any], bool]] = {}
        from agents.training.instrumented_ppo.micro_step import micro_step
        self.fns["E32"] = (self.model, micro_step, True)
        self.fns["Ebf"] = (self.model, micro_step, False)
        compiled = [a for a in arms if a not in ("E32", "Ebf")]
        if compiled:
            if smoke_backend:
                self.say(f"[diag] compiled arms {compiled} through the {smoke_backend!r} backend: a CODE-PATH smoke "
                         "(compiled == eager arithmetic; their numbers mean nothing)")
            elif device != "cuda":
                self.say(f"[diag] compiled arms {compiled} on the CPU: Inductor's C++ backend, NOT the CUDA (Triton) "
                         "kernels the run trained on — it reads the graph-level path (dynamo, AOTAutograd, the "
                         "partitioner, decompositions), not the GPU codegen")
            from agents.model.compile_cache import ensure_hermetic_cache
            ensure_hermetic_cache("diag_bf16_canary")   # K3: never torch's shared default dir
            from agents.model import compile_control as cc
            from agents.model import compile_regions as cr
            cc.control().install()
            torch._dynamo.config.cache_size_limit = max(int(torch._dynamo.config.cache_size_limit), 64)
            kw: Dict[str, Any] = {"fullgraph": True, "dynamic": False}
            sb: Dict[str, Any] = {"backend": smoke_backend} if smoke_backend else {}
            if "Cbf" in compiled:
                cr.install(self.model, backend=smoke_backend)
                self.fns["Cbf"] = (self.model, self.model._compiled_micro_step, False)
            if "C32" in compiled:
                self.model32 = build_learner(first_zip, device, precision="fp32")
                cr.install(self.model32, backend=smoke_backend)
                self.fns["C32"] = (self.model32, self.model32._compiled_micro_step, False)
            if "Cbf_emul" in compiled:
                emul: Dict[str, Any] = sb or {"options": {"emulate_precision_casts": True}}
                self.fns["Cbf_emul"] = (self.model, torch.compile(micro_step, **emul, **kw), False)
            if "Cbf_aot" in compiled:
                self.fns["Cbf_aot"] = (self.model, torch.compile(micro_step, backend="aot_eager", **kw), False)
            if "Cbf_pheager" in compiled:
                # the pointer head EAGER inside an otherwise compiled R1: an instance-level disabled forward is a
                # graph break (so fullgraph=False), on its OWN learner so no other arm's guards see the patch
                self.model_ph = build_learner(first_zip, device)
                ph = self.model_ph.policy.pointer_head
                ph.forward = torch.compiler.disable(ph.forward)
                self.fns["Cbf_pheager"] = (self.model_ph, torch.compile(micro_step, fullgraph=False, dynamic=False,
                                                                        **sb), False)
        self.say(f"[diag] learner(s) built on {device} in {time.perf_counter() - t0:.1f}s; arms {self.arm_names}")

    @property
    def models(self) -> List[Any]:
        return [m for m in (self.model, self.model32, self.model_ph) if m is not None]

    def batch_args(self, model: Any, slice_: int) -> Tuple[Tuple[Any, ...], Any]:
        from agents.model import compile_regions as cr
        b = cr.r1_batch(model, RUN_BATCH, slice_)
        return cr._r1_args(model, b), b

    def run_arms(self, slice_: int, which: Optional[Sequence[str]] = None,
                 old_override: Optional[torch.Tensor] = None) -> Tuple[Dict[str, ArmOut], Tuple[Any, ...]]:
        """Every arm on fixture slice ``slice_``; ``old_override`` replaces the batch's old log-probs (the
        relocated fixture, `relocate_old_log_prob`) — an input tensor, so no compiled arm recompiles."""
        arms: Dict[str, ArmOut] = {}
        args0: Tuple[Any, ...] = ()
        for a in list(which or self.arm_names):
            if a not in self.arm_names:
                continue                                   # dropped after a failure (`self.failed`)
            model, fn, fp32 = self.fns[a]
            call_args, _ = self.batch_args(model, slice_)
            if old_override is not None:
                call_args = call_args[:4] + (old_override.to(call_args[4].device),) + call_args[5:]
            if model is self.model:
                args0 = call_args
            t0 = time.perf_counter()
            try:
                arms[a] = run_arm(model, fn, call_args, fp32_trunk=fp32)
            except Exception as exc:
                if a not in OPTIONAL_ARMS:
                    raise
                # a LOCALISATION arm that cannot compile or run is a recorded finding, never the window's end
                self.failed[a] = f"{type(exc).__name__}: {str(exc)[:600]}"
                self.arm_names.remove(a)
                self.say(f"[diag]   arm {a} FAILED and is dropped: {self.failed[a][:200]}")
                continue
            if self.device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if dt > 20:
                self.say(f"[diag]   arm {a} slice {slice_}: {dt:.0f}s (first call compiles)")
        return arms, args0


# ---------------------------------------------------------------------------------------------- one checkpoint
def n_fixture_rows() -> int:
    from agents.training.learner_golden import BUFFER_PATH
    with np.load(BUFFER_PATH) as z:
        return int(z["actions"].reshape(-1).shape[0])


def read_checkpoint(ctx: Context, label: str, zip_path: Path, *, dither_n: int, slices: Sequence[int] = (0, 1)
                    ) -> Dict[str, Any]:
    """Everything for one checkpoint (see the module docstring)."""
    from agents.model import compile_regions as cr
    t0 = time.perf_counter()
    load_weights(ctx.models, zip_path)
    model = ctx.model
    regime = cr.weights_regime(model)
    nfx = n_fixture_rows()
    rec: Dict[str, Any] = {"label": label, "zip": str(zip_path), "regime": regime, "device": ctx.device,
                           "arms": {a: ARMS[a] for a in ctx.arm_names}, "weights": weight_norms(model), "slices": {}}
    st: Any = None
    for sl in slices:
        arms, call_args = ctx.run_arms(sl)
        st = call_args[-1]
        grp = sorted(_row_groups(RUN_BATCH, sl, nfx).items())
        uniq = [int(torch.nonzero(m)[0]) for _, m in grp]
        fx_rows = [int(u) for u, _ in grp]
        srec: Dict[str, Any] = {
            "loss": {a: v.loss for a, v in arms.items()},
            "pairwise": pairwise(arms, ctx.names, ctx.sizes),
            "forward_vs_E32": {a: _forward_diff(v, arms["E32"], call_args[3], uniq) for a, v in arms.items()
                               if a != "E32"},
            "ratio": ratio_stats(arms["E32"], call_args[4], st, uniq, fx_rows),
            "implied": implied_read(model, arms, call_args, ctx.names, ctx.sizes),
            "rows": row_decomposition(model, arms, call_args, sl, nfx),
            "activations": {"fp32": activation_stats(model, call_args[1], call_args[3], uniq, fp32_trunk=True),
                            "bf16": activation_stats(model, call_args[1], call_args[3], uniq, fp32_trunk=False)},
        }
        for comp in [a for a in ("Cbf", "Cbf_emul", "Cbf_aot", "Cbf_pheager") if a in arms]:
            srec.setdefault("canary_stat", {})[comp] = canary_stat(arms[comp], arms["Ebf"], arms["E32"],
                                                                    ctx.names, ctx.sizes, regime)
        if "Cbf" in arms:
            srec["canary_verdict"] = canary_verdict(arms["Cbf"], arms["Ebf"], arms["E32"], ctx.names, ctx.sizes,
                                                    regime)
            srec["canary_replay"] = _canary_replay(model, sl)
        if "C32" in arms:
            # the fp32 rule's own question: compiled fp32 vs eager fp32 (a precision-independent miscompile)
            srec["fp32_rule"] = _fp32_rule(arms["C32"], arms["E32"], ctx.names, ctx.sizes, regime)
        srec["relocated"], _ = relocated_read(ctx, sl, arms, call_args, regime)
        rec["slices"][str(sl)] = srec
        ctx.say(f"[diag] {label} slice {sl}: " + _headline(srec))
    if dither_n > 0:
        rec["null"] = _dither(ctx, dither_n, regime)
    rec["failed_arms"] = dict(ctx.failed)
    rec["r1_cache_entries"] = _cache_entries()
    rec["seconds"] = time.perf_counter() - t0
    rec["st"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                 for k, v in st._asdict().items()} if st is not None else None
    return rec


def _cache_entries() -> Optional[int]:
    """How many dynamo cache entries hold R1's code object (`micro_step`): one per separately compiled arm (+ the
    graph-break arm's pieces live on other code objects) — the check that no arm silently reused another's graph."""
    try:
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list
        from agents.training.instrumented_ppo.micro_step import micro_step
        return len(_debug_get_cache_entry_list(micro_step.__code__))
    except Exception:                                          # a report field, never a reason to stop
        return None


def _forward_diff(arm: ArmOut, ref: ArmOut, masks: torch.Tensor, uniq: Sequence[int]) -> Dict[str, float]:
    ix = torch.as_tensor(list(uniq))
    legal = masks.bool().cpu()[ix]
    d = (arm.logp[ix].double() - ref.logp[ix].double()).abs()[legal]
    return {"legal_logp_max_abs": float(d.max()), "legal_logp_p99_abs": float(np.quantile(d.numpy(), 0.99)),
            "log_prob_max_abs": float((arm.log_prob[ix].double() - ref.log_prob[ix].double()).abs().max()),
            "entropy_max_abs": float((arm.entropy[ix].double() - ref.entropy[ix].double()).abs().max())}


def _fp32_rule(c32: ArmOut, e32: ArmOut, names: Sequence[str], sizes: Sequence[int], regime: str) -> Dict[str, Any]:
    from agents.model import compile_regions as cr
    from agents.model import compile_trainer as ct
    from agents.model.compile_gate_probe import per_param_grad_errors
    err = dict(per_param_grad_errors(c32.grad, e32.grad, list(sizes), floor_frac=ct._PARAM_GRAD_FLOOR))
    bar = float(cr.R1_PARAM_BAR[regime])
    worst = sorted(((v, names[i]) for i, v in err.items()), reverse=True)[:8]
    a, b = c32.grad.double(), e32.grad.double()
    cos = float((a * b).sum() / (a.norm() * b.norm()))
    return {"bar": bar, "cosine": cos, "loss_rel": abs(c32.loss - e32.loss) / max(1.0, abs(e32.loss)),
            "worst": [{"param": n, "rel": v} for v, n in worst], "over_bar": [n for v, n in worst if not v <= bar],
            "watch": {names[i]: v for i, v in err.items() if names[i].startswith(WATCH)}}


def _canary_replay(model: Any, slice_: int) -> Dict[str, str]:
    """The canary's OWN check (`CompileCanary._check`: `compile_regions.r1_rungs` through the installed compiled
    route) on this slice — the reproduction of the run's verdict at this checkout, verbatim."""
    from agents.model import compile_trainer as ct
    from agents.model.compile_canary import CompileCanary
    try:
        rules = CompileCanary(model, batch_size=RUN_BATCH)._check(slice_, {})
        return {"verdict": "pass", "text": " | ".join(rules)}
    except ct.CompileTrainerError as exc:
        return {"verdict": "refused", "text": f"{type(exc).__name__}: {exc}"}


def _headline(srec: Dict[str, Any]) -> str:
    pw = srec["pairwise"]
    n = WATCH + "move_proj.weight"
    parts = []
    for pair in ("Ebf|E32", "Cbf|E32", "Cbf|Ebf", "C32|E32", "Cbf_emul|Ebf", "Cbf_aot|Ebf", "Cbf_pheager|Ebf"):
        if pair in pw:
            parts.append(f"d({pair.replace('|', ',')}) {pw[pair].get(n, float('nan')):.3g}")
    kap = srec["rows"]["params"].get(n, {}).get("kappa", float("nan"))
    cv = srec.get("canary_verdict", {}).get("verdict", "-")
    return f"move_proj.weight: {', '.join(parts)}; kappa {kap:.3g}; canary {cv}"


def _dither(ctx: Context, n: int, regime: str) -> Dict[str, Any]:
    """The dithered null on slice 0: per seed, every arm's head error vs that seed's own E32, the number of PPO
    clip-branch FLIPS against E32 (batch rows), and the error with the branch HELD to E32's decisions (the implied
    gradient through the eager fp32 head) — so a bimodal error is attributed, or not, to the clip boundary."""
    from agents.model import compile_regions as cr
    per_seed: List[Dict[str, Dict[str, float]]] = []
    per_seed_r: List[Dict[str, Dict[str, float]]] = []
    raw: List[Dict[str, Any]] = []
    for s in range(n):
        with dithered(ctx.models, seed=1 + s):
            arms, call_args = ctx.run_arms(0)
            st = call_args[-1]
            obs, actions, masks, old = call_args[1], call_args[2], call_args[3], call_args[4]
            inputs = head_inputs(ctx.model, obs, fp32_trunk=True)
            act_ref = ppo_active(arms["E32"], old, st)
            order = list(arms)
            gf = dict(zip(order, head_grads(ctx.model, inputs, [
                implied_dz(arms[a], actions, old, masks, st, active=act_ref) for a in order])))
            _, arms_r = relocated_read(ctx, 0, arms, call_args, regime)
        sp = {k: split(v.grad, ctx.names, ctx.sizes) for k, v in arms.items()}
        spr = {k: split(v.grad, ctx.names, ctx.sizes) for k, v in arms_r.items()}
        # only what the canary JUDGES on this seed (above the per-parameter floor of its own reference): a head
        # parameter below it (the struggle score, ~1e-9 of the largest gradient) reads pure rounding as a ratio
        jd, jdr = judged_params(arms["E32"], ctx.names, ctx.sizes), judged_params(arms_r["E32"], ctx.names, ctx.sizes)
        per_seed.append({a: {nm: rel_l2(sp[a][nm], sp["E32"][nm]) for nm in sp[a]
                             if nm.startswith(WATCH) and nm in jd} for a in arms if a != "E32"})
        per_seed_r.append({a: {nm: rel_l2(spr[a][nm], spr["E32"][nm]) for nm in spr[a]
                               if nm.startswith(WATCH) and nm in jdr} for a in arms_r if a != "E32"})
        raw.append({a: {"flips": int((ppo_active(arms[a], old, st) != act_ref).sum()),
                        "err": {nm: rel_l2(sp[a][nm], sp["E32"][nm]) for nm in _RAW_PARAMS},
                        "err_frozen_branch": {nm: rel_l2(gf[a][nm], gf["E32"][nm]) for nm in _RAW_PARAMS}}
                    for a in arms if a != "E32"})
    k = float(cr.BF16_ENVELOPE_K["param"]) if cr.BF16_ENVELOPE_K else float("nan")
    out = null_summary(per_seed, k, float(cr.R1_PARAM_BAR[regime]))
    out["relocated"] = null_summary(per_seed_r, k, float(cr.R1_PARAM_BAR[regime]))
    out["per_seed"] = raw
    out["flip_attribution"] = flip_attribution(raw)
    return out


#: the per-seed raw record's parameters (the three the FATAL named)
_RAW_PARAMS = (WATCH + "move_proj.weight", WATCH + "move_proj.bias", WATCH + "move_score.weight")


def flip_attribution(raw: List[Dict[str, Any]], param: str = WATCH + "move_proj.weight") -> Dict[str, Any]:
    """Per arm: the head error's median over the seeds WITH a clip-branch flip vs WITHOUT, and the frozen-branch
    error's maximum — if the flips carry the bad mode, the with-flip median is large and the frozen-branch error
    stays in the good mode for every seed."""
    out: Dict[str, Any] = {}
    arms = list(raw[0]) if raw else []
    for a in arms:
        w = [r[a]["err"][param] for r in raw if r[a]["flips"] > 0]
        wo = [r[a]["err"][param] for r in raw if r[a]["flips"] == 0]
        fr = [r[a]["err_frozen_branch"][param] for r in raw]
        out[a] = {"seeds_with_flip": len(w), "seeds_without": len(wo),
                  "median_err_with_flip": float(np.median(w)) if w else None,
                  "median_err_without": float(np.median(wo)) if wo else None,
                  "max_err_frozen_branch": float(max(fr)) if fr else None,
                  "max_err_without_flip": float(max(wo)) if wo else None}
    return out


# ------------------------------------------------------------------------------------------------------ verdict
def _label_key(label: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)M", label or "")
    return float(m.group(1)) if m else (1e9 if label == "fatal" else 2e9)


def render_verdict(out_dir: Path) -> str:
    """VERDICT.md — the one-page template, filled from every checkpoint read (`<mode>_<label>.json`) in ``out_dir``,
    in step order per mode. Every column is a field of the JSON; the checklist names the read that decides each
    hypothesis (designs/training/compile_flags.md "The bf16 canary FATAL on trained weights")."""
    recs: Dict[str, Any] = {}
    for p in sorted(out_dir.glob("*_*.json")):
        if p.name.startswith(("cpu_", "gpu_")):
            with contextlib.suppress(OSError, ValueError):
                r = json.loads(p.read_text())
                if isinstance(r, dict) and "slices" in r:
                    recs[p.stem] = r
    n = WATCH + "move_proj.weight"

    def g(d: Any, *ks: str) -> Any:
        for k in ks:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    def f(x: Any) -> str:
        return "-" if x is None else (f"{x:.3g}" if isinstance(x, float) else str(x))
    lines = ["# bf16 canary FATAL on trained weights — diagnostic verdict", "",
             f"Reads: {len(recs)}. Arms: " + "; ".join(f"**{k}** {v}" for k, v in ARMS.items()), "",
             "## Per checkpoint (slice 0, `pointer_head.move_proj.weight`; d = relative L2 to the second arm)", "",
             "| read | d(Ebf,E32) | d(Cbf,E32) | d(C32,E32) | d(Cbf_emul,Ebf) | d(Cbf_aot,Ebf) | d(Cbf_pheager,Ebf) | "
             "canary ratio | verdict | Cbf flips | Cbf frozen-branch d | min switching margin | relocated ratio / "
             "verdict | null P(ratio>K) Ebf pairs / Cbf | relocated null | kappa | pre-act p99 |",
             "|" + "---|" * 18]
    for name, r in sorted(recs.items(), key=lambda kv: (kv[0].split("_")[0], _label_key(kv[1].get("label", "")))):
        s0 = g(r, "slices", "0") or {}
        pw = s0.get("pairwise", {})
        cs = g(s0, "canary_stat", "Cbf", "watch", n) or {}
        nul = g(r, "null", "ratio_null", n) or {}
        nulr = g(r, "null", "relocated", "ratio_null", n) or {}
        rel = g(s0, "relocated") or {}
        margins = g(s0, "ratio", "switching_margin_logp") or []
        lines.append("| " + " | ".join([
            name, f(g(pw, "Ebf|E32", n)), f(g(pw, "Cbf|E32", n)), f(g(pw, "C32|E32", n)),
            f(g(pw, "Cbf_emul|Ebf", n)), f(g(pw, "Cbf_aot|Ebf", n)), f(g(pw, "Cbf_pheager|Ebf", n)),
            f(cs.get("ratio")), f(g(s0, "canary_verdict", "verdict")),
            f(g(s0, "implied", "Cbf", "branch_flips", "batch_rows")),
            f(g(s0, "implied", "Cbf", "frozen_branch_vs_ref", n)),
            f(margins[0]["margin"]) if margins else "-",
            f"{f(g(rel, 'canary_stat', 'Cbf', 'watch', n, 'ratio'))} / {f(g(rel, 'canary_verdict', 'verdict'))}",
            f"{f(g(nul, 'Ebf_pairs', 'p_over_k'))} / {f(g(nul, 'Cbf_vs_Ebf', 'p_over_k'))}",
            f"{f(g(nulr, 'Ebf_pairs', 'p_over_k'))} / {f(g(nulr, 'Cbf_vs_Ebf', 'p_over_k'))}",
            f(g(s0, "rows", "params", n, "kappa")), f(g(s0, "activations", "fp32", "move_preact_abs", "0.99"))]) + " |")
    lines += ["", "## Verdict (fill from the table; each line names the read that decides it)", "",
              "- [ ] **Reproduced** on the canary's weights (`gpu_fatal`: `canary_replay` / `canary_verdict` "
              "refused, naming the run's parameters at its numbers): ...",
              "- [ ] **The clip-boundary flip** (the CPU finding): Cbf flips a row's PPO clip branch against E32 "
              "(`implied.Cbf.branch_flips`), its FROZEN-branch error is in the no-flip mode (`frozen_branch_vs_ref`), "
              "the null's bad mode is the flip seeds (`null.flip_attribution`): ...",
              "- [ ] **(i) precision-independent miscompile**: d(C32,E32) on the pointer head vs the fp32 bar "
              "(`fp32_rule.over_bar` empty = KILLED): ...",
              "- [ ] **(ii) Inductor bf16 numerics**: Cbf's frozen-branch error and `implied.Cbf.self` (its backward "
              "against its own forward) vs Ebf's; Cbf_emul / Cbf_aot / Cbf_pheager vs Ebf: ...",
              "- [ ] **(iii) a yardstick unusually small**: d(Ebf,E32) at the FATAL vs `null.errors.Ebf` (its no-flip "
              "mode): ...",
              "- [ ] **(iv) the fixture**: the rows within rounding of their switching clip boundary "
              "(`ratio.switching_margin_logp`); the RELOCATED fixture (the T36 fix, production's) passes and its null has "
              "P(ratio > K) = 0: ...",
              "- [ ] **Recommendation for X26**: ...", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------- main
def _say(line: str) -> None:
    print(line, flush=True)


def _write(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def pinned_rows_dir() -> Path:
    """Where a run's PINNED K9 rows live (the buffer its canary judged, when the committed golden has since been
    rebuilt): `src/agents/model/testdata/`."""
    from utils.paths import src_path
    return src_path("agents", "model", "testdata")


def select_buffer(run: Path, spec: str) -> Path:
    """The K9 golden buffer every read of this process judges, installed as `learner_golden.BUFFER_PATH` (which
    `compile_regions.golden_rows` reads). The FATAL is a property of a (weights, ROWS) state: legacy deletion U3 rebuilt
    the committed golden (`33a1c3b9`) after `rb_toy_ent_s4001`'s canary FATAL'd on the rows of its pin, and on the new
    rows those weights hold no boundary row as close. ``auto`` reads the run's pinned rows when committed
    (`testdata/k9_golden_buffer_<run>_pin.npz`), else the committed golden."""
    from agents.training import learner_golden as LG
    if spec == "auto":
        pinned = pinned_rows_dir() / f"k9_golden_buffer_{run.name}_pin.npz"
        path = pinned if pinned.is_file() else Path(LG.BUFFER_PATH)
    elif spec == "golden":
        path = Path(LG.BUFFER_PATH)
    else:
        path = Path(spec)
    if not path.is_file():
        raise DiagError(f"--buffer {spec!r}: no such file {path}")
    LG.BUFFER_PATH = path
    return path


def run_mode(mode: str, a: argparse.Namespace) -> int:
    global ALLOW_FP32_RUN, RUN_BATCH
    ALLOW_FP32_RUN = bool(getattr(a, "allow_fp32_run", False))
    RUN_BATCH = int(getattr(a, "batch", None) or BATCH)
    if mode == "gpu" and RUN_BATCH != BATCH:
        raise DiagError(f"the gpu mode reads R1's declared signature, B = {BATCH} (got --batch {RUN_BATCH})")
    torch.set_float32_matmul_precision("highest")
    if mode == "gpu":
        if not torch.cuda.is_available():
            raise DiagError("the gpu mode needs CUDA (run it through scripts/ops/diag_bf16_canary.sh under the lease)")
        torch.cuda.set_device(0)
    run = resolve_run(a.run)
    buffer = select_buffer(run, getattr(a, "buffer", "auto") or "auto")
    _say(f"[diag] the rows judged: {buffer}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = [s for s in a.ckpts.split(",") if s.strip()]
    ckpts = [resolve_checkpoint(run, s) for s in specs]
    arms = GPU_ARMS if mode == "gpu" else CPU_ARMS
    if a.arms:
        arms = tuple(x for x in a.arms.split(",") if x)
        bad = [x for x in arms if x not in ARMS]
        if bad or "E32" not in arms or "Ebf" not in arms:
            raise DiagError(f"--arms must name known arms and include E32 and Ebf: {arms}")
    t_start = time.perf_counter()
    ctx = Context(ckpts[0][1], "cuda" if mode == "gpu" else "cpu", arms, _say,
                  smoke_backend=getattr(a, "smoke_backend", None) or None)
    meta = {"mode": mode, "run": str(run), "torch": torch.__version__, "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if mode == "gpu" else "cpu",
            "threads": torch.get_num_threads(), "batch": RUN_BATCH, "dither_rel": DITHER_REL,
            "commit": _commit(), "argv": sys.argv[1:], "started": time.time(), "buffer": str(buffer)}
    _write(out / f"{mode}_meta.json", meta)
    done, skipped, incomplete = [], [], []
    for i, (label, zp) in enumerate(ckpts):
        path = out / f"{mode}_{label}.json"
        if path.exists() and not a.force:
            skipped.append(label)
            continue
        if (time.perf_counter() - t_start) / 60.0 > a.budget_min:
            incomplete.append(label)
            continue
        dn = a.dither if i < a.dither_first else a.dither_rest
        rec = read_checkpoint(ctx, label, zp, dither_n=dn)
        rec["meta"] = meta
        _write(path, rec)
        done.append(label)
        _say(f"[diag] wrote {path} ({rec['seconds']:.0f}s)")
    status = "COMPLETE" if not incomplete else "INCOMPLETE (wall budget)"
    _write(out / f"{mode}_status.json", {"status": status, "done": done, "skipped": skipped,
                                          "not_started": incomplete,
                                          "elapsed_s": time.perf_counter() - t_start})
    (out / "VERDICT.md").write_text(render_verdict(out))
    _say(f"[diag] {mode}: {status}; done {done}, skipped {skipped}, not started {incomplete} -> {out}/VERDICT.md")
    return 0 if not incomplete else 3


def _commit() -> str:
    try:
        from utils.git import head_sha
        return str(head_sha())
    except Exception:                                         # a report field, never a reason to stop
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main.ops.diag_bf16_canary", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for mode, ck, d1, dr, budget in (("cpu", DEFAULT_CPU_CKPTS, 0, 0, 600.0), ("gpu", DEFAULT_GPU_CKPTS, 32, 8, 14.0)):
        s = sub.add_parser(mode)
        s.add_argument("--run", default=RUN, help="run name under models/ or a run directory")
        s.add_argument("--ckpts", default=ck, help="comma list: fatal | <x>M | a zip path (the FIRST builds the learner)")
        s.add_argument("--out", required=True)
        s.add_argument("--arms", default="", help=f"comma list from {list(ARMS)} (default: every arm of the mode)")
        s.add_argument("--dither", type=int, default=d1, help="null seeds for the first --dither-first checkpoints")
        s.add_argument("--dither-first", type=int, default=2)
        s.add_argument("--dither-rest", type=int, default=dr, help="null seeds for the remaining checkpoints")
        s.add_argument("--budget-min", type=float, default=budget,
                       help="stop STARTING checkpoints after this many minutes (the status says INCOMPLETE)")
        s.add_argument("--force", action="store_true", help="redo checkpoints whose JSON exists")
        if mode == "cpu":
            s.add_argument("--smoke-backend", default="",
                           help="a CODE-PATH smoke of the compiled arms on CPU through this dynamo backend (eager)")
        s.add_argument("--batch", type=int, default=BATCH,
                       help="rows per read (cpu: 64 reads the fixture's rows once — exact clip decisions, ~30x faster "
                            "on the production surface; gpu: must stay the declared 2048)")
        s.add_argument("--allow-fp32-run", action="store_true",
                       help="judge an fp32-trained run's weights under the bf16 rule (as the production baseline's "
                            "gate read does)")
        s.add_argument("--buffer", default="auto",
                       help="the K9 golden buffer whose rows every read judges: 'auto' = the run's PINNED rows when "
                            "src/agents/model/testdata/k9_golden_buffer_<run>_pin.npz exists, else the committed "
                            "golden; 'golden' = the committed golden; or an .npz path")
    v = sub.add_parser("verdict")
    v.add_argument("--out", required=True)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.mode == "verdict":
        out = Path(a.out)
        (out / "VERDICT.md").write_text(render_verdict(out))
        print(out / "VERDICT.md")
        return 0
    try:
        return run_mode(a.mode, a)
    except DiagError as exc:
        print(f"[diag] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
