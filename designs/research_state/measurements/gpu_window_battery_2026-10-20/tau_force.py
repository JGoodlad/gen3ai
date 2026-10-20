"""The τ-solve FORCING seam of the FP32_ALT cells (`tau_decision_cell.py`, `tau_update_cell.py`).

A model's solve is ``features_extractor.hypothesis_builder.tau_solve``, and no model accepts ``fp32_naive`` (the
pre-T62 sum, measurement only). So a cell forces its arm at the ONE call site every model path shares:
`hypothesis_set.fixed_mass_presence` calls ``fixed_size_tau`` through the MODULE GLOBAL. `install` replaces that global,
BEFORE anything is built or compiled, by a wrapper that drops the caller's ``solve`` and passes ``solve=<arm>``.

The wrapper COUNTS its calls without changing the compiled program: an eager call bumps ``EAGER[arm]``; a call dynamo
traces runs `_note_trace` through ``comptime`` (at trace time, never in the graph), so the counter adds no guard and
no recompile (a plain counter mutation inside the traced region recompiles on EVERY call: probed 2026-10-20, torch
2.8). `report` FAILS a cell whose wrapper was replaced mid-cell or was never reached by an arm (a vacuous arm).

    import tau_force as TF; TF.install(HS, "fp32_alt"); ...; rep = TF.report(HS, ("fp32_alt",)); rep["violations"]
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch._dynamo.comptime import comptime

#: Every arm the forcing accepts: the pre-T62 naive fp32 sum (the cost baseline), FP32_ALT (the default), T62's fp64.
#: This battery's cells force only the first two (fp64 was read by the 2026-10-19 battery).
ARMS = ("fp32_naive", "fp32_alt", "fp64")
STATE: dict[str, Any] = {"arm": None, "fn": None}
EAGER: dict[str, int] = {a: 0 for a in ARMS}
TRACED: dict[str, int] = {a: 0 for a in ARMS}


def _note_trace(ctx: Any) -> None:            # comptime: no free variables allowed, so module globals only
    TRACED[STATE["arm"]] += 1


def check_api(HS: Any) -> list[str]:
    """What this checkout's `hypothesis_set` lacks for the forcing (empty = ready)."""
    bad = []
    if hasattr(HS, "TAU_SOLVE_DTYPE"):
        bad.append("hypothesis_set.TAU_SOLVE_DTYPE still exists (a T62 checkout, not FP32_ALT)")
    if tuple(getattr(HS, "TAU_SOLVES", ())) != ("fp32_alt", "fp64"):
        bad.append(f"hypothesis_set.TAU_SOLVES is {getattr(HS, 'TAU_SOLVES', None)!r}, want ('fp32_alt', 'fp64')")
    if getattr(HS, "TAU_SOLVE_NAIVE", None) != "fp32_naive":
        bad.append(f"hypothesis_set.TAU_SOLVE_NAIVE is {getattr(HS, 'TAU_SOLVE_NAIVE', None)!r}, want 'fp32_naive'")
    fn = getattr(HS, "fixed_size_tau", None)
    if fn is None or not {"n_iter", "solve"} <= set(inspect.signature(fn).parameters):
        bad.append("hypothesis_set.fixed_size_tau lacks the n_iter / solve parameters")
    src = inspect.getsource(HS.fixed_mass_presence) if hasattr(HS, "fixed_mass_presence") else ""
    if "fixed_size_tau(" not in src:
        bad.append("hypothesis_set.fixed_mass_presence no longer calls fixed_size_tau through the module global")
    return bad


def install(HS: Any, arm: str) -> Callable[..., torch.Tensor]:
    """Replace ``HS.fixed_size_tau`` by the forcing wrapper (once per process); its arm is ``arm`` until `set_arm`."""
    if STATE["fn"] is not None:
        raise RuntimeError("tau_force.install called twice in one process")
    bad = check_api(HS)
    if bad:
        raise RuntimeError("; ".join(bad))
    orig = HS.fixed_size_tau

    def forced(scores: torch.Tensor, cand: torch.Tensor, k: torch.Tensor, n_iter: int = HS.BISECTION_ITERS,
               solve: Any = None) -> torch.Tensor:
        a = STATE["arm"]
        if torch.compiler.is_compiling():
            comptime(_note_trace)
        else:
            EAGER[a] += 1
        r: torch.Tensor = orig(scores, cand, k, n_iter, solve=a)
        return r

    set_arm(arm)
    STATE["fn"] = forced
    HS.fixed_size_tau = forced
    return forced


def set_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (one of {ARMS})")
    STATE["arm"] = arm


def intact(HS: Any) -> bool:
    return STATE["fn"] is not None and HS.fixed_size_tau is STATE["fn"]


def report(HS: Any, arms: Sequence[str]) -> dict[str, Any]:
    """The forcing record a cell writes into its result; ``violations`` non-empty = the cell is NOT ok."""
    v = []
    if not intact(HS):
        v.append("hypothesis_set.fixed_size_tau was replaced mid-cell (the forcing wrapper is not installed)")
    for a in arms:
        if EAGER[a] + TRACED[a] == 0:
            v.append(f"arm {a}: the wrapper was called 0 times (a vacuous arm)")
    return {"arms": list(arms), "eager_calls": {a: EAGER[a] for a in arms},
            "trace_calls": {a: TRACED[a] for a in arms}, "intact": intact(HS), "violations": v}


def cpu_reach_check(HS: Any, arms: Sequence[str], learner: bool, seed: int = 2601, rows: int = 8) -> str:
    """CPU only: install the wrapper, build the seeded fresh policy at X26's surface (the cells' build), and for each
    arm run the path the cell times -- T2's eager decision forward (``learner=False``) or the learner's
    `evaluate_actions_functional` forward + backward (``learner=True``) -- asserting the wrapper was REACHED by that
    arm; then a dynamo trace (``backend="eager"``, ``fullgraph=True``) of `fixed_mass_presence` per arm, asserting
    one trace is counted and a second call neither re-traces nor graph-breaks. Raises on any miss."""
    import shlex

    from agents.inference.service.decision import DecisionModule
    from agents.inference.service.engine import decide
    from agents.inference.service.parity import fixture_rows
    from agents.training.gpu_bench_suite import PRODUCTION_ARGV
    from agents.training.gpu_bench_update import resolved_args
    from main.fresh_checkpoint import build_fresh_model

    install(HS, arms[0])
    args = resolved_args(shlex.split(PRODUCTION_ARGV), "cpu")
    model, _, _ = build_fresh_model(seed, args=args)
    pol = model.policy
    attr = pol.features_extractor.hypothesis_builder.tau_solve
    D = int(model.observation_space["observation"].shape[0])
    o_np, m_np = fixture_rows(D, rows)
    o, m = torch.as_tensor(o_np).float(), torch.as_tensor(m_np)
    out = []
    for a in arms:
        set_arm(a)
        before = EAGER[a]
        if learner:
            pol.train()
            acts = m.float().argmax(-1)
            vals, logp, _, _, _ = pol.evaluate_actions_functional({"observation": o, "action_mask": m}, acts, m)
            (vals.sum() + logp.sum()).backward()
            pol.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                decide(DecisionModule(pol.eval()).eval(), o, m)
        hits = EAGER[a] - before
        if hits <= 0:
            raise RuntimeError(f"arm {a}: the {'learner' if learner else 'decision'} path never reached the wrapper")
        out.append(f"{a} {hits} eager calls")
    g = torch.Generator().manual_seed(0)
    s = torch.randn(4, 30, generator=g) * 4
    cand = torch.rand(4, 30, generator=g) > 0.2
    k = torch.full((4,), 3, dtype=torch.long)
    for a in arms:
        set_arm(a)
        torch._dynamo.reset()
        graphs = [0]

        def be(gm: Any, ex: Any, graphs: list[int] = graphs) -> Any:
            graphs[0] += 1
            return gm.forward
        c = torch.compile(lambda x, y, z: HS.fixed_mass_presence(x, y, z).pi, backend=be, fullgraph=True)
        t0 = TRACED[a]
        c(s, cand, k)
        c(s, cand, k)
        if TRACED[a] - t0 != 1 or graphs[0] != 1:
            raise RuntimeError(f"arm {a}: dynamo traced the wrapper {TRACED[a] - t0} times into {graphs[0]} graphs "
                               "over two calls (want 1 and 1: a counter that re-traces would perturb the timing)")
    torch._dynamo.reset()
    rep = report(HS, arms)
    if rep["violations"]:
        raise RuntimeError("; ".join(rep["violations"]))
    return (f"wrapper reached on the {'learner forward+backward' if learner else 'decision forward'} "
            f"(B{rows}, model tau_solve attr {attr!r}): {', '.join(out)}; one trace, no re-trace, per arm")
