"""FP32_ALT COST on CPU (no GPU lease): the fp32 split-residual tau solve against the pre-T62 naive fp32 sum and T62's
float64 solve. ONE torch thread (the clean CPU read; the box runs other agents' jobs), arms INTERLEAVED, medians.

1. MICRO: `fixed_size_tau` alone at the production call shapes -- the species call [512, 400] and the per-slot move
   call [512, 6, 371] -- eager and inductor-compiled (fullgraph=True: a graph break would raise).
2. FORWARD, EAGER: the whole T2 decision forward (`DecisionModule`) on fixture rows at X26's checkpoint_280000170, at
   B 1 / 16 / 64 / 256, the hypothesis builder's ``tau_solve`` set per call.
3. FORWARD, COMPILED (--compiled): the same forward compiled by inductor at B 64, one compile per arm.

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
_spec = importlib.util.spec_from_file_location("fp32alt_equivalence", os.path.join(_HERE, "equivalence.py"))
EQ = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(EQ)  # type: ignore[union-attr]
ARMS = ("fp32_naive", "fp32_alt", "fp64")


def interleaved(fns: Dict[str, Callable[[], Any]], reps: int) -> Dict[str, float]:
    """Median wall ms per call, the arms interleaved (ABCABC...), after one warm call each."""
    for f in fns.values():
        f()
    t: Dict[str, List[float]] = {k: [] for k in fns}
    for _ in range(reps):
        for k, f in fns.items():
            t0 = time.perf_counter()
            f()
            t[k].append((time.perf_counter() - t0) * 1e3)
    return {k: float(np.median(v)) for k, v in t.items()}


def ratios(ms: Dict[str, float]) -> Dict[str, float]:
    return {f"{k}/fp32_naive": round(v / ms["fp32_naive"], 3) for k, v in ms.items() if k != "fp32_naive"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--compiled", action="store_true")
    a = ap.parse_args()
    th.set_num_threads(int(os.environ.get("TAU_THREADS", "1")))
    import torch._dynamo as dynamo
    from agents.model import hypothesis_set as HS
    from agents.model.compile_regions import CPU_INDUCTOR_OPTIONS     # torch 2.8.0 CPU tile2d codegen bug
    res: Dict[str, Any] = {"threads": th.get_num_threads(), "torch": th.__version__, "reps": a.reps}
    g = th.Generator().manual_seed(0)
    # ---- 1. micro
    for name, shape in (("species", (512, 400)), ("moves", (512, 6, 371))):
        s = (th.randn(*shape, generator=g) * 4).float()
        cand = th.rand(*shape, generator=g) > 0.1
        k = th.randint(1, 5, shape[:-1], generator=g)
        row: Dict[str, Any] = {"shape": list(shape)}
        row["eager_ms"] = interleaved({arm: (lambda arm=arm: HS.fixed_size_tau(s, cand, k, solve=arm)) for arm in ARMS},
                                      a.reps)
        row["compile_s"] = {}
        for arm in ARMS:
            dynamo.reset()
            c = th.compile(lambda x, y, z, arm=arm: HS.fixed_size_tau(x, y, z, solve=arm), fullgraph=True,
                           backend="inductor", dynamic=False, options=dict(CPU_INDUCTOR_OPTIONS))
            t0 = time.perf_counter()
            c(s, cand, k)
            row["compile_s"][arm] = round(time.perf_counter() - t0, 1)
            row.setdefault("compiled_ms", {})[arm] = interleaved({"x": lambda c=c: c(s, cand, k)}, a.reps)["x"]
        dynamo.reset()
        row["eager_ratio"], row["compiled_ratio"] = ratios(row["eager_ms"]), ratios(row["compiled_ms"])
        print(f"[micro {name}] eager {row['eager_ms']} {row['eager_ratio']} | compiled {row['compiled_ms']} "
              f"{row['compiled_ratio']}", flush=True)
        res[f"micro_{name}"] = row
    # ---- 2. the whole decision forward, eager
    from agents.inference.service.decision import DecisionModule
    from main.capacity import load_policy
    from utils.rust_env.fixture_battles import play_rows
    rows = play_rows(8, seed=6201)
    fo = np.concatenate([o for o, _ in rows]).astype(np.float32)
    fm = np.concatenate([m for _, m in rows]).astype(bool)
    m, _c, _ = load_policy(EQ.P280, EQ.P280_DIR, "cpu")
    pol = m.policy
    dm = DecisionModule(pol).eval()

    def run(arm: str, f: Callable[..., Any], o: th.Tensor, mk: th.Tensor) -> Callable[[], Any]:
        def _r() -> Any:
            EQ.TS.set_solve(pol, arm)
            return f(o, mk)
        return _r

    with th.no_grad():
        for B in (1, 16, 64, 256):
            idx = np.arange(B) % len(fo)
            o, mk = th.as_tensor(fo[idx]), th.as_tensor(fm[idx])
            ms = interleaved({arm: run(arm, dm, o, mk) for arm in ARMS}, a.reps if B < 256 else max(5, a.reps // 2))
            res[f"forward_eager_B{B}"] = {"ms": ms, "ratio": ratios(ms)}
            print(f"[forward eager B={B}] {ms} {ratios(ms)}", flush=True)
            with open(a.out, "w") as fh:
                json.dump(res, fh, indent=1)
        # ---- 3. compiled
        if a.compiled:
            B = 64
            idx = np.arange(B) % len(fo)
            o, mk = th.as_tensor(fo[idx]), th.as_tensor(fm[idx])
            row = {"batch": B, "compile_s": {}, "ms": {}}
            for arm in ARMS:
                dynamo.reset()
                EQ.TS.set_solve(pol, arm)
                c = th.compile(DecisionModule(pol).eval(), backend="inductor", dynamic=False,
                               options=dict(CPU_INDUCTOR_OPTIONS))
                t0 = time.perf_counter()
                c(o, mk)
                row["compile_s"][arm] = round(time.perf_counter() - t0, 1)
                row["ms"][arm] = interleaved({"x": run(arm, c, o, mk)}, a.reps)["x"]
                dynamo.reset()
            row["ratio"] = ratios(row["ms"])
            res[f"forward_compiled_B{B}"] = row
            print(f"[forward compiled B={B}] {row}", flush=True)
    EQ.TS.set_solve(pol, "fp32_alt")
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
