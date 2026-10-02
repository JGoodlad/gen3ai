"""The DECISION-level parity check for the compiled CPU OPPONENT (gen3_opponent_compile_parity_v1).

WHY. `--compile-opponents` (`compile_opponents.maybe_compile_extractor`) compiled every frozen
self-play / eval / ladder model's extractor and checked only its SPEED, on an all-zero
observation. Nothing compared a compiled opponent's decisions with eager at all — so a CPU Inductor
miscompile would have handed the learner a wrong opponent (a wrong self-play distribution, a wrong
eval number) with every gate green. The learner's own compile gate learned
the same lesson twice (zero obs hid a 7.65 miscompile; fresh weights hid the pointer head).

WHAT IT CHECKS, at the learner gate's bars (`compile_trainer._FP32_TOL`, `decision_verdicts`):
``features`` (pi ‖ vf), the MASKED legal log-probabilities and ``V``, eager vs the installed compiled
callable, on `ROWS` rows of the committed REAL-obs fixture (`compile_parity_fixture`), each fed at
B=1 with the float ``action_mask`` and every declared extra key — EXACTLY the key set, dtypes and
shape the opponent's warm-up compiled and every live decision reuses, so the check judges the graph
that plays and triggers no recompile. On FRESH weights (the zero-init pointer head makes every legal
log-prob ``-log(n_legal)``) both arms ALSO run on a seeded in-place perturbation
(`parity_probe.perturbed_parameters`, restored bit-exactly) and that verdict is judged with the
vacuity guard ON, exactly as the learner gate does.

WHAT IT RAISES. The learner gate's own typed errors: `compile_trainer.CompileTrainerError`
(`VacuousCompileParityError` for a comparison that cannot fail). NOT the opponent path's
warn-and-fall-back: a timing miss costs throughput, a parity miss is a WRONG opponent (GIGO), and a
CPU miscompile is systemic — the same code object in every worker — so it raises regardless of
`--compile-opponents-strict`. `maybe_compile_extractor` uninstalls the compile before re-raising.

COST AND CADENCE — once per DISTINCT WEIGHTS per process, never per game. Measured (CPU, one
thread, 2026-09-29): the check costs ~0.28 s on trained weights and ~0.55 s on fresh ones (the
perturbed pass doubles it), against a 106 s cold / ~6 s warm-cache compile on the first load and a
~0.3 s reused-compile load after. That is small per LOAD, but one consumer loads PER GAME: the
eval worker re-`MaskablePPO.load`s its snapshot opponent every game batch. So a PASS is
cached process-locally under a blake2b fingerprint of the policy's ``state_dict`` bytes plus its
train/eval mode (`weights_fingerprint`, ~18 ms for the 7.0M-parameter production policy): the
compiled callable is shared per process by dynamo's code-object cache and the parameters are graph
inputs, so the same weights through the same graph cannot give a different verdict. A different
checkpoint (or a changed weight) is a different key and is checked. A failure is never cached.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List

import torch

#: Fixture rows the check reads, each at B=1. Enough to carry multi-action rows, switches and every
#: phase of the reproducible battles; small because it runs per model load on the worker's startup.
ROWS = 16

#: Process-local PASS verdicts, keyed by `weights_fingerprint` (see the module docstring).
_PASSED: Dict[str, str] = {}


def weights_fingerprint(policy: torch.nn.Module) -> str:
    """blake2b over every ``state_dict`` entry's name, shape, dtype and BYTES, plus the train/eval
    mode of every submodule (a dropout/mode switch is a different graph)."""
    h = hashlib.blake2b(digest_size=16)
    for name, v in policy.state_dict().items():
        t = v.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            t = t.float()
        h.update(f"{name}|{tuple(t.shape)}|{t.dtype}|".encode())
        h.update(t.numpy().tobytes())
    h.update(bytes(int(m.training) for m in policy.modules()))
    return h.hexdigest()


def _obs_row(fe: Any, row: Any, mask_row: Any) -> Dict[str, torch.Tensor]:
    """One B=1 decision obs shaped exactly like the compile warm-up's (`_compile_warmup_obs`)."""
    from agents.model.extra_obs_keys import zero_extra_obs
    obs = {"observation": torch.as_tensor(row[None], dtype=torch.float32),
           "action_mask": torch.as_tensor(mask_row[None], dtype=torch.float32)}
    obs.update(zero_extra_obs(fe))
    return obs


def _arm(model: Any, fe: Any, forward: Callable[[Any], Any], rows: Any, mask: Any
         ) -> Dict[str, torch.Tensor]:
    """The decision readout (`compile_trainer._readout`) row by row through ``forward``."""
    from agents.model.compile_trainer import _readout
    had = "forward" in vars(fe)
    prev = vars(fe).get("forward")
    fe.forward = forward
    try:
        per: List[Dict[str, torch.Tensor]] = [
            _readout(model, fe, _obs_row(fe, rows[i], mask[i]), mask[i:i + 1])
            for i in range(len(rows))]
    finally:
        if had:
            fe.forward = prev
        else:
            del fe.forward
    return {k: torch.cat([p[k] for p in per], dim=0) for k in per[0]}


def check_opponent_parity(model: Any, original: Callable[[Any], Any],
                          compiled: Callable[[Any], Any], *, label: str = "opponent",
                          rows: int = ROWS) -> str:
    """Judge ``compiled`` against ``original`` (eager) at the decision level. Returns the PASS line;
    raises `CompileTrainerError` (or its `VacuousCompileParityError`) otherwise. Leaves
    ``fe.forward`` exactly as it found it and the weights bit-identical."""
    from agents.model.compile_gate_probe import has_policy_heads
    from agents.model.compile_parity_fixture import ParityFixtureError, load_parity_rows
    from agents.model.compile_trainer import (_FP32_TOL, CompileTrainerError,
                                              VacuousCompileParityError, decision_verdicts)
    from agents.model.parity_probe import (PERTURB_LADDER, fresh_reason, perturbed_parameters,
                                           rung_seed)

    policy = getattr(model, "policy", None)
    fe = getattr(policy, "features_extractor", None)
    where = f"--compile-opponents parity ({label})"
    if fe is None or not has_policy_heads(policy) or not isinstance(policy, torch.nn.Module):
        raise CompileTrainerError(
            f"{where}: not a Gen3 dual-head policy, so its compiled decisions cannot be validated "
            f"— refusing to install an unvalidated compile.")
    layout = getattr(fe, "layout", None)
    dim = layout["total_dim"] if isinstance(layout, dict) else fe.observation_space.shape[0]
    try:
        all_rows, all_mask = load_parity_rows(int(dim))
    except ParityFixtureError as exc:
        raise CompileTrainerError(f"{where}: {exc}") from exc
    x, m = all_rows[:rows], all_mask[:rows]
    key = f"{rows}|{type(fe).__module__}.{type(fe).__qualname__}|{weights_fingerprint(policy)}"
    if key in _PASSED:
        return f"{_PASSED[key]} [cached: these exact weights already passed in this process]"

    with torch.no_grad():
        eager = _arm(model, fe, original, x, m)
        comp = _arm(model, fe, compiled, x, m)
        fresh = fresh_reason({k: v for k, v in eager.items() if k in _FP32_TOL}, _FP32_TOL)
        lines: List[str] = []
        try:
            if fresh is not None:
                # gen3_parity_perturb_ladder_v1: the first INFORMATIVE rung of the declared ladder
                # judges (a collapsed critic can stay vacuous on V at the fresh-weights scale); a
                # rung that is still vacuous moves on, a real divergence raises at once, and no
                # informative rung re-raises the last vacuity refusal.
                for n_rung, (scale, k) in enumerate(PERTURB_LADDER):
                    with perturbed_parameters(policy, seed=rung_seed(k), scale=scale):
                        p_eager = _arm(model, fe, original, x, m)
                        p_comp = _arm(model, fe, compiled, x, m)
                    try:
                        rules = decision_verdicts(eager=p_eager, compiled=p_comp)
                    except VacuousCompileParityError:
                        if n_rung == len(PERTURB_LADDER) - 1:
                            raise
                        continue
                    lines += [f"[fresh weights, seeded perturbation scale={scale:g} seed+{k}] " + r
                              for r in rules]
                    break
            lines += decision_verdicts(eager=eager, compiled=comp, allow_vacuous=fresh is not None)
        except CompileTrainerError as exc:
            raise type(exc)(f"{where}: {exc}") from exc
    line = f"parity PASS on {len(x)} REAL obs rows at B=1 — " + " | ".join(lines)
    _PASSED[key] = line
    return line
