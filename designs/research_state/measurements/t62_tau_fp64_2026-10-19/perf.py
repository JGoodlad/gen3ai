"""T62 COST on CPU (no GPU lease): the float64 tau solve against the pre-T62 fp32 solve.

1. MICRO: `fixed_size_tau` alone at the production call shapes -- the species call [B, 400] and the per-slot move
   call [B, 6, M] -- eager and inductor-compiled (fullgraph=True: a graph break would raise).
2. FORWARD: the whole T2 decision forward (`DecisionModule`) on fixture rows at X26's checkpoint_280000170, eager
   and inductor-compiled, OLD (`TAU_SOLVE_DTYPE` patched to float32) vs NEW. Interleaved ABAB timing, median.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src CUDA_VISIBLE_DEVICES= python3 perf.py --out perf.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from typing import Any, Callable, Dict, List

import numpy as np
import torch as th

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("t62_equivalence", os.path.join(_HERE, "equivalence.py"))
EQ = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(EQ)  # type: ignore[union-attr]


def _set(dt: th.dtype) -> None:
    from agents.model import hypothesis_set as HS
    HS.TAU_SOLVE_DTYPE = dt


def abab(fns: Dict[str, Callable[[], Any]], reps: int) -> Dict[str, float]:
    """Median wall ms per call, the arms interleaved (each arm sets its own solve dtype before the call)."""
    t: Dict[str, List[float]] = {k: [] for k in fns}
    for _ in range(reps):
        for k, f in fns.items():
            t0 = time.perf_counter()
            f()
            t[k].append((time.perf_counter() - t0) * 1e3)
    return {k: float(np.median(v)) for k, v in t.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=15)
    a = ap.parse_args()
    th.set_num_threads(int(os.environ.get("T62_THREADS", "3")))
    import torch._dynamo as dynamo
    from agents.model import hypothesis_set as HS
    res: Dict[str, Any] = {"threads": th.get_num_threads(), "torch": th.__version__}
    g = th.Generator().manual_seed(0)
    # ---- 1. micro
    for name, shape in (("species", (512, 400)), ("moves", (512, 6, 371))):
        s = (th.randn(*shape, generator=g) * 4).float()
        cand = th.rand(*shape, generator=g) > 0.1
        k = th.randint(1, 5, shape[:-1], generator=g)
        row: Dict[str, Any] = {"shape": list(shape)}
        for dt_name, dt in (("old", th.float32), ("new", th.float64)):
            _set(dt)
            dynamo.reset()
            c = th.compile(HS.fixed_size_tau, fullgraph=True, backend="inductor", dynamic=False)
            t0 = time.perf_counter()
            c(s, cand, k)
            row[f"compile_s_{dt_name}"] = round(time.perf_counter() - t0, 1)
            row[f"compiled_ms_{dt_name}"] = abab({"x": lambda: c(s, cand, k)}, a.reps)["x"]
            row[f"eager_ms_{dt_name}"] = abab({"x": lambda: HS.fixed_size_tau(s, cand, k)}, a.reps)["x"]
            dynamo.reset()
        _set(th.float64)
        print(f"[micro {name}] {row}", flush=True)
        res[f"micro_{name}"] = row
    # ---- 2. the whole decision forward
    from agents.inference.service.decision import DecisionModule
    from main.capacity import load_policy
    from utils.rust_env.fixture_battles import play_rows
    rows = play_rows(8, seed=6201)
    fo = np.concatenate([o for o, _ in rows]).astype(np.float32)
    fm = np.concatenate([m for _, m in rows]).astype(bool)
    m, _c, _ = load_policy(EQ.P280, EQ.P280_DIR, "cpu")
    dm = DecisionModule(m.policy).eval()
    for B in (64, 256):
        idx = np.arange(B) % len(fo)
        o, mk = th.as_tensor(fo[idx]), th.as_tensor(fm[idx])
        row = {"batch": B}
        with th.no_grad():
            def run(dt: th.dtype, f: Callable[..., Any]) -> Callable[[], Any]:
                def _r() -> Any:
                    _set(dt)
                    return f(o, mk)
                return _r
            row["eager_ms"] = abab({"old": run(th.float32, dm), "new": run(th.float64, dm)}, a.reps)
            comp = {}
            for dt_name, dt in (("old", th.float32), ("new", th.float64)):
                _set(dt)
                dynamo.reset()
                from agents.model.compile_regions import CPU_INDUCTOR_OPTIONS     # torch 2.8.0 CPU tile2d codegen bug
                c = th.compile(DecisionModule(m.policy).eval(), backend="inductor", dynamic=False, options=dict(CPU_INDUCTOR_OPTIONS))
                t0 = time.perf_counter()
                c(o, mk)
                row[f"compile_s_{dt_name}"] = round(time.perf_counter() - t0, 1)
                comp[dt_name] = abab({"x": run(dt, c)}, a.reps)["x"]
                dynamo.reset()
            row["compiled_ms"] = comp
        _set(th.float64)
        print(f"[forward B={B}] {row}", flush=True)
        res[f"forward_B{B}"] = row
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
