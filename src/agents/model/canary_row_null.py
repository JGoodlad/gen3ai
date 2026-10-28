"""THE PER-ROW DITHER NULL (`gen3_canary_row_null_shadow_v1`, 2026-10-25; BINDING since `gen3_canary_row_bar_v1`,
2026-10-28).

WHY. R1's clip-margin rule held every arm to ONE global margin M (`R1_BAR_K` x `R1_LOGPROB_SHIFT_MAX`, 0.72), from a
CPU census of a toy run. On X26-lineage weights the fp32 function ALONE moves some fixture row by 0.80-1.23 under a
rounding-sized weight dither, and the compiled arm read 0.712 at the X26 continuation's u1000. The fix is a PER-ROW bar
from each row's own dither null (`designs/research_state/measurements/canary_row_null_design_2026-10-25/`). This module
computes that null at every canary, on its own weights and its own kernels; it was a recorded-only shadow until the W0
CUDA census set k, and now `canary_row_bar` turns it into the compiled arm's per-row bars (stage 1). This module itself
still judges nothing: a skip, a truncation or an error is recorded and the bars fall back to the cap.

WHAT IT COMPUTES (the design's rule 1). On the K9 golden's UNIQUE fixture rows (B = the
fixture's row count, batch row == fixture row: E32 at the canary's tiled B equals E32 at 64 to 2.9e-5):

  * the fp32 REFERENCE: `evaluate_actions_functional` under `trunk_precision.fp32_scope`, no grad, train mode, fp32
    matmul precision 'highest' — log π(taken) per row;
  * ``ROW_NULL_SEEDS`` dithers: every policy parameter multiplied by (1 + ``ROW_NULL_REL`` x N(0, 1)), the noise
    NAME-KEYED and SEEDED (`noise_seed`: `main.ops.diag_bf16_canary.dithered`'s formula, so on CPU the draws are its
    draws) from ONE generator on the parameters' device, re-seeded per parameter — FIXED seeds 0..K-1, so the same
    weights give the same null. Each parameter is rewritten from a saved copy and RESTORED BIT-EXACTLY in a
    ``finally`` (the pattern of the canary's own perturbed rungs, `parity_probe.perturbed_parameters`);
  * per row: ``null`` = max over the seeds of |log π_s − log π|; per arm (from the scheduled check's live rung,
    `compile_regions.GradWatch.row_dlogp`, the max over the row's tiles) the shift and its ratio to
    max(null, ``ROW_NULL_FLOOR``).

THE BUDGET. Target <= 1-2 s per canary on the GPU (the design's estimate, UNVERIFIED: a B = 64 fp32 forward is
launch-bound). The reference forward runs twice (``ref_repeat_max_abs``: the fp32 forward's own run-to-run
difference, 0 when it is deterministic; the second, warm call is the one timed); when (K + 1) x it would pass ``cap_s`` (``ROW_NULL_CAP_S``,
``$GEN3AI_CANARY_ROW_NULL_CAP_S``; 0 turns the shadow off) the seeds are SKIPPED with the reason recorded, and a run
that overruns mid-way stops and records the seeds it finished (``truncated``). Every wall time is in the record.

WHAT IT DOES NOT DO: change training. Under `torch.random.fork_rng`; training mode, the fp32 matmul precision and every
parameter are restored; nothing is backpropagated; the policy's stashes are released (`release_gate_memory`); any
error is RECORDED (``error``), never raised — the per-row bars then fall back to the cap (`canary_row_bar.from_null`).
"""
from __future__ import annotations

import os
import time
import zlib
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

#: The design's K (16: the calibration's seed count). The record keeps every seed's per-row value, so a K-sensitivity
#: (open question 3: the max over K is heavy-tailed, a jump row rests on one seed) is recoverable from any record.
ROW_NULL_SEEDS = 16
#: The dither: relative 2^-9 (half a bf16 ulp), `diag_bf16_canary.DITHER_REL`.
ROW_NULL_REL = 2.0 ** -9
#: The ratio's floor f (the design's 0.02; on CPU every floor from 0.01 to 0.1 gave the same maximum).
ROW_NULL_FLOOR = 0.02
#: The cap on the whole null's wall time per canary (seconds). Production target <= 1-2 s on the GPU; a CPU learner's
#: B = 64 forward is 0.3-9 s, so there the cap usually SKIPS (recorded).
ROW_NULL_CAP_S = 5.0
CAP_ENV = "GEN3AI_CANARY_ROW_NULL_CAP_S"


def cap_seconds() -> float:
    """The configured cap: ``$GEN3AI_CANARY_ROW_NULL_CAP_S`` (a non-negative number; 0 = the shadow OFF), else
    `ROW_NULL_CAP_S`. A malformed value raises (a cap is never guessed)."""
    v = os.environ.get(CAP_ENV, "").strip()
    if not v:
        return ROW_NULL_CAP_S
    try:
        c = float(v)
    except ValueError as exc:
        raise ValueError(f"${CAP_ENV}={v!r} is not a number of seconds") from exc
    if not (c >= 0.0 and np.isfinite(c)):
        raise ValueError(f"${CAP_ENV}={v!r} is not a non-negative number of seconds")
    return c


def noise_seed(name: str, seed: int) -> int:
    """The NAME-KEYED seed of parameter ``name``'s noise on dither ``seed`` — `diag_bf16_canary.dithered`'s formula."""
    return (zlib.crc32(name.encode()) ^ (int(seed) * 2654435761)) & 0x7FFFFFFF


def _r(v: float) -> Any:
    """4 significant digits (a non-finite value as its string), as `GradWatch.record`."""
    v = float(v)
    return float(f"{v:.4g}") if np.isfinite(v) else str(v)


def _q(a: np.ndarray) -> Dict[str, Any]:
    if not a.size:
        return {}
    fin = np.where(np.isfinite(a), a, np.inf)
    return {"p50": _r(np.quantile(fin, 0.5)), "p90": _r(np.quantile(fin, 0.9)), "max": _r(fin.max()),
            "argmax_row": int(np.argmax(fin))}


def _ref_logp(policy: Any, b: Any) -> torch.Tensor:
    """log π(taken) per row: the fp32 reference forward (eager, no grad), float64 on the CPU."""
    from agents.model.trunk_precision import fp32_scope
    with fp32_scope(policy), torch.no_grad():
        _v, lp, _e, _l, _m = policy.evaluate_actions_functional(b.obs, b.actions.long().flatten(), b.action_masks)
    ref: torch.Tensor = lp.detach().double().cpu().reshape(-1)
    return ref


def compute(model: Any, *, k: Optional[int] = None, rel: float = ROW_NULL_REL, cap_s: Optional[float] = None,
            clock: Callable[[], float] = time.perf_counter) -> Dict[str, Any]:
    """The per-row null on ``model``'s CURRENT weights (module docstring). Returns the raw result —
    ``{"null": ndarray[n], "per_seed": ndarray[s, n], ...}`` — or ``{"skipped": why}`` / ``{"error": what}``; never
    raises, never changes the model (parameters bit-exact, training mode, the matmul precision, the global RNG).
    ``k`` None = `ROW_NULL_SEEDS`; ``cap_s`` None = `cap_seconds()`."""
    from agents.model import compile_regions as cr
    k = ROW_NULL_SEEDS if k is None else int(k)
    cap = cap_seconds() if cap_s is None else float(cap_s)
    if cap <= 0.0:
        return {"skipped": f"off (cap {cap:g} s; ${CAP_ENV})"}
    policy: Any = getattr(model, "policy", None)
    if not hasattr(policy, "evaluate_actions_functional") or not hasattr(policy, "observation_space"):
        return {"skipped": "the policy has no evaluate_actions_functional / observation space (not a production learner)"}
    out: Dict[str, Any] = {"k": int(k), "rel": float(rel), "cap_s": float(cap)}
    was_training = bool(policy.training)
    prec = torch.get_float32_matmul_precision()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    saved: Dict[str, torch.Tensor] = {}
    t0 = clock()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.set_float32_matmul_precision("highest")
            policy.set_training_mode(True)
            n = int(cr.golden_rows(model).data["actions"].reshape(-1).shape[0])
            b = cr.r1_batch(model, n, 0)                     # B = n: batch row == fixture row
            if b.fixture_rows is not None and list(b.fixture_rows) != list(range(n)):
                return {"skipped": f"the slice-0 batch of {n} rows is not the fixture rows in order"}
            ref = _ref_logp(policy, b)
            t1 = clock()
            again = _ref_logp(policy, b)          # warm (the first call may pay a lazy init) and a repeat check
            out["ref_forward_s"] = clock() - t1
            out["ref_repeat_max_abs"] = float((again - ref).abs().max()) if ref.numel() else 0.0
            out["rows"] = n
            if (k + 1) * out["ref_forward_s"] > cap:
                out["skipped"] = (f"projected {(k + 1) * out['ref_forward_s']:.3g} s = (K {k} + 1) x the reference "
                                  f"forward {out['ref_forward_s']:.3g} s > the cap {cap:g} s (${CAP_ENV})")
                return out
            params = list(policy.named_parameters())
            with torch.no_grad():
                saved = {nm: p.detach().clone() for nm, p in params}
            gens: Dict[str, torch.Generator] = {}
            seeds: List[np.ndarray] = []
            for s in range(int(k)):
                with torch.no_grad():
                    for nm, p in params:
                        dev = str(p.device)
                        g = gens.get(dev)
                        if g is None:
                            g = gens[dev] = torch.Generator(device=p.device)
                        g.manual_seed(noise_seed(nm, s))
                        noise = torch.randn(p.shape, generator=g, dtype=torch.float32, device=p.device)
                        p.copy_(saved[nm] * (1.0 + rel * noise))
                seeds.append((_ref_logp(policy, b) - ref).abs().numpy())
                if clock() - t0 > cap and s + 1 < k:
                    out["truncated"] = f"the cap {cap:g} s was passed after {s + 1} of {k} seeds"
                    break
            per = np.stack(seeds) if seeds else np.zeros((0, n))
            out["per_seed"] = per
            out["null"] = per.max(axis=0) if seeds else np.zeros(n)
            return out
    except Exception as exc:                                 # recorded, never a verdict (module docstring)
        out["error"] = f"{type(exc).__name__}: {exc}"[:500]
        return out
    finally:
        if saved:
            with torch.no_grad():
                for nm, p in policy.named_parameters():
                    if nm in saved:
                        p.copy_(saved[nm])
        policy.set_training_mode(was_training)
        torch.set_float32_matmul_precision(prec)
        try:
            cr.release_gate_memory(model)
        except Exception:                                    # noqa: BLE001 — a toy policy has no stashes
            pass
        out["seconds"] = clock() - t0


def record(raw: Dict[str, Any], row_dlogp: Optional[Dict[str, np.ndarray]] = None,
           floor: float = ROW_NULL_FLOOR, judged: bool = False) -> Dict[str, Any]:
    """The JSONL record (`canary_verdicts.jsonl` ``row_null``) of `compute`'s result and the scheduled check's per-row
    |Δ log π| per arm (fixture-row indexed; None / absent when the check never reached the margin rule). ``judged``:
    the per-row bars came from this null (`canary_row_bar`, gen3_canary_row_bar_v1)."""
    rec: Dict[str, Any] = {"shadow": not judged, "judged": bool(judged)}
    for key in ("k", "rel", "cap_s", "rows", "ref_repeat_max_abs", "skipped", "truncated", "error"):
        if key in raw:
            rec[key] = raw[key]
    for key in ("seconds", "ref_forward_s"):
        if key in raw:
            rec[key] = round(float(raw[key]), 4)
    null = raw.get("null")
    if null is None:
        return rec
    null = np.asarray(null, dtype=np.float64)
    rec["floor"] = float(floor)
    rec["null_per_row"] = [_r(x) for x in null]
    rec["null"] = _q(null)
    per = np.asarray(raw.get("per_seed", np.zeros((0, null.size))))
    rec["seeds_done"] = int(per.shape[0])
    rec["per_seed"] = [[_r(x) for x in row] for row in per]
    rec["seed_worst_row"] = [int(np.argmax(row)) for row in per]
    arms: Dict[str, Any] = {}
    for arm, d in (row_dlogp or {}).items():
        d = np.asarray(d, dtype=np.float64)
        if d.shape != null.shape:
            arms[arm] = {"error": f"per-row shift has {d.size} rows, the null {null.size}"}
            continue
        ratio = d / np.maximum(null, floor)
        arms[arm] = {"dlogp_per_row": [_r(x) for x in d], "ratio_per_row": [_r(x) for x in ratio],
                     "dlogp": _q(d), "ratio": _q(ratio)}
    if arms:
        rec["arms"] = arms
    return rec
