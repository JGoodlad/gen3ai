"""Which X5 pieces carry the cost: the extractor's COMPILED forward + backward at the learner's micro-batch,
one ablation arm per PROCESS (X5 cost ablation, 2026-10-04). MEASUREMENT ONLY — every arm except `blob`
and `fm` patches the code IN THIS PROCESS (monkeypatch, no file changes) and computes something
DIFFERENT from the design; an arm's number is the cost of what it removes, never a candidate design.

    GEN3AI_TEST_ALLOW_GPU=1 python ablate.py --arm <arm> --obs real_obs.npz --batch 2048 --out res.json [--profile]

The program: the production-surface learner (`production_args()`, `--belief-tokens fixed_mass` for every
arm but `blob`), the gates' perturbation (`parity_probe.perturb_`, so δ_θ is not at its zero init),
`torch.compile(fe, dynamic=False, fullgraph=True)` of the extractor forward on REAL rows, fwd + backward of
(pi.sum() + vf.sum()) — `compile_ab.py --bwd` of the U2 checks, unchanged — timed as the median of
REPS blocks of N steps after warm-up (CUDA-synchronised), plus one eager warm-up call before the compile.

ARMS (each removes ONE piece from `fm` by making its result DEAD or CONSTANT, so Inductor drops it and
its backward; the patch is named):
  blob         production (no X5)
  fm           `--belief-tokens fixed_mass` as built at the measured commit
  no_other     OTHER's second op pass: `ExtractorForward._other_edge_cells` returns zero cells of the
               real shapes (recorded on the eager call) — the OTHER-mode roster's kernels go dead
  no_hypenc    the hypothesis `PokemonEncoder` pass: `splice_hypothesis_tokens` is fed the REAL pass's
               tokens as the hypothesis tokens, so the second encoder pass (and its backward) is dead
  no_bisect    every fixed-size construction's 64 bisection steps: `fixed_size_tau` at 0 steps (τ = the
               bracket midpoint; same graph otherwise) — species, active moves, per-mon [B,6,M], OTHER's
               moves, and any in the graph
  no_argsort   every stable sort: `stable_order` returns the identity order (no argsort) — the species
               order, the active's move order, the per-mon [B,6,M] order, OTHER's order
  no_ext       the K+16 seat axis: `fixed_mass_moves` returns idx_ext = the K seats, w_ext = seat_w,
               mix = I_K (the 16 typed-HP columns are not priced)
  no_keybias   the trunk's log-π key bias: `key_log_presence` returns None (the pools keep theirs)
  no_hypatk    the kept hypothesis-attacker work: the roster's `alive` excludes hypothesis slots (dense
               kernels — expected ≈ 0: gating multiplies, it does not skip)
  no_permon    the per-mon move construction alone: `slot_move_presence`'s fixed-size construction
               replaced by σ(logits)·cand (no bisection) and its order by the identity
  all_off      no_other + no_hypenc + no_bisect + no_argsort + no_ext + no_keybias (the residual vs blob is
               what is left: the OTHER trunk seat, δ_θ, the op on the hypothesis context, the pools' bias …)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from collections import deque

import numpy as np
import torch

PIECES = {
    "no_other": ["other"], "no_hypenc": ["hypenc"], "no_bisect": ["bisect"], "no_argsort": ["argsort"],
    "no_ext": ["ext"], "no_keybias": ["keybias"], "no_hypatk": ["hypatk"], "no_permon": ["permon"],
    "all_off": ["other", "hypenc", "bisect", "argsort", "ext", "keybias"],
}
ARMS = ["blob", "fm"] + list(PIECES)


def build(arm: str):
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from agents.training.rust_rollout.build import trainee_spaces
    from agents.training.rust_vec_env import RustVecEnv
    from main.fresh_checkpoint import _production_policy_kwargs
    from main.train.production_args import production_args
    from utils.torch_state_guard import single_thread_build

    args = production_args()
    if arm != "blob":
        args.belief_tokens = "fixed_mass"
    obs, act = trainee_spaces(args)
    env = RustVecEnv(n_envs=4, observation_space=obs, action_space=act, build=lambda m: None)
    _a, _l, pk = _production_policy_kwargs(args)
    torch.manual_seed(0)
    with single_thread_build():
        model = InstrumentedMaskablePPO(Gen3DualHeadMaskablePolicy, env, n_steps=16, batch_size=16,
                                        n_epochs=1, device="cpu", seed=0, policy_kwargs=pk, verbose=0)
    model.ep_info_buffer = deque(maxlen=100)
    return model


def apply_patches(pieces) -> None:
    import agents.model.extractor_forward as EF
    import agents.model.hypothesis_set as HS
    import agents.model.hypothesis_tokens as HT

    if "other" in pieces:
        orig = EF.ExtractorForward._other_edge_cells
        shapes = {}

        def other_cells(self, ctx, ro, sb, fams):
            b = int(ro.alive.shape[0])
            if b not in shapes:              # the EAGER warm-up call at this batch records the real shapes
                out = orig(self, ctx, ro, sb, fams)
                shapes[b] = {k: (tuple(v.shape), v.dtype) for k, v in out.items()}
                return out
            return {k: torch.zeros(s, dtype=d, device=ro.alive.device) for k, (s, d) in shapes[b].items()}
        EF.ExtractorForward._other_edge_cells = other_cells
    if "hypenc" in pieces:
        orig_s = HT.splice_hypothesis_tokens
        EF.splice_hypothesis_tokens = lambda rt, rh, hs, mk: orig_s(rt, rt, hs, mk)
    if "bisect" in pieces:
        orig_t = HS.fixed_size_tau
        HS.fixed_size_tau = lambda scores, cand, k, n_iter=0: orig_t(scores, cand, k, 0)
    if "argsort" in pieces:
        def ident(key, cand):
            return torch.arange(key.shape[-1], device=key.device).expand(key.shape).contiguous()
        HS.stable_order = ident
    if "ext" in pieces:
        orig_f = HT.fixed_mass_moves

        def fm_noext(moves, logits):
            fm = orig_f(moves, logits)
            K = fm.seat_nums.shape[1]
            B = fm.seat_nums.shape[0]
            eye = torch.eye(K, dtype=fm.mix.dtype, device=fm.mix.device).expand(B, -1, -1)
            return dataclasses.replace(fm, idx_ext=fm.seat_nums, w_ext=fm.seat_w, mix=eye)
        EF.fixed_mass_moves = fm_noext
    if "keybias" in pieces:
        EF.key_log_presence = lambda *a, **k: None
    if "hypatk" in pieces:
        orig_r = HT.build_op_roster

        def roster_noatk(*a, **k):
            ro, hs = orig_r(*a, **k)
            return dataclasses.replace(ro, alive=ro.alive * (~ro.hyp).to(ro.alive.dtype)), hs
        EF.build_op_roster = roster_noatk
    if "permon" in pieces:
        orig_smp = HT.slot_move_presence
        orig_bor = HT.build_op_roster

        def cheap_presence(scores, cand, k, n_iter=None):
            pi = torch.where(cand, torch.sigmoid(scores.detach()), torch.zeros_like(scores))
            live = (k > 0) & (k < cand.sum(-1))
            return HS.Presence(logits=scores, pi=pi, log_pi=pi, cand=cand, k=k, n=cand.sum(-1), live=live,
                               full=~live)

        def smp(hb, move_logits, species, revealed_ids):
            real = HS.fixed_mass_presence
            HS.fixed_mass_presence = cheap_presence
            try:
                return orig_smp(hb, move_logits, species, revealed_ids)
            finally:
                HS.fixed_mass_presence = real
        HT.slot_move_presence = smp

        def bor(*a, **k):
            real = HS.stable_order
            HS.stable_order = lambda key, cand: torch.arange(key.shape[-1], device=key.device).expand(key.shape).contiguous()
            try:
                return orig_bor(*a, **k)
            finally:
                HS.stable_order = real
        EF.build_op_roster = bor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--n", type=int, default=20, help="steps per timing block")
    ap.add_argument("--reps", type=int, default=5, help="timing blocks (median reported)")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--t2-batches", default="",
                    help="also time the NO-GRAD forward at these batches (T2's buckets, e.g. 8,64,256) as T2 serves it: "
                         "compiled (fullgraph, static shape) and CAPTURED in a CUDA graph, timed by replay")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from agents.model.compile_cache import ensure_hermetic_cache
    from agents.model.parity_probe import PERTURB_SCALE, PERTURB_SEED, perturb_

    root = ensure_hermetic_cache(f"x5 cost ablation {a.arm}")
    apply_patches(PIECES.get(a.arm, []))
    model = build(a.arm)
    pol = model.policy
    pol.eval()
    perturb_(pol, seed=PERTURB_SEED, scale=PERTURB_SCALE)
    pol.to("cuda")
    fe = pol.features_extractor
    with np.load(a.obs) as z:
        obs = {k[4:]: torch.as_tensor(z[k][:a.batch]).to("cuda") for k in z.files if k.startswith("obs:")}
    n_rows = next(iter(obs.values())).shape[0]
    assert n_rows == a.batch, f"bank has {n_rows} rows < batch {a.batch}"

    def fn(b):
        return fe(b)

    def step(f):
        pi, vf = f(obs)
        (pi.float().sum() + vf.float().sum()).backward()
        pol.zero_grad(set_to_none=True)

    res = {"arm": a.arm, "pieces_removed": PIECES.get(a.arm, []), "batch": a.batch, "torch": torch.__version__}
    with torch.enable_grad():
        step(fn)                                     # eager warm-up (records the no_other shapes)
        torch.cuda.synchronize()
        comp = torch.compile(fn, dynamic=False, fullgraph=True)
        t0 = time.time()
        step(comp)
        torch.cuda.synchronize()
        res["first_call_compile_s"] = round(time.time() - t0, 1)
        for _ in range(10):
            step(comp)
        torch.cuda.synchronize()
        blocks = []
        for _ in range(a.reps):
            t = time.time()
            for _ in range(a.n):
                step(comp)
            torch.cuda.synchronize()
            blocks.append((time.time() - t) / a.n * 1000)
        res["compiled_ms_blocks"] = [round(x, 3) for x in blocks]
        res["compiled_ms_median"] = round(statistics.median(blocks), 3)
        res["peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if a.profile:
            from torch.profiler import ProfilerActivity, profile
            n_it = 3
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(n_it):
                    step(comp)
                torch.cuda.synchronize()
            rows = {}
            for e in prof.key_averages():
                us = getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
                if us > 0:
                    rows[e.key] = (us / n_it / 1000.0, e.count / n_it)

            def cls(name: str) -> str:
                n = name.lower()
                if "fmha" in n or "flash" in n or "attention" in n or "sdpa" in n:
                    return "attention"
                if "gemm" in n or "cutlass" in n or "cublas" in n or "sgemm" in n:
                    return "gemm"
                if n.startswith("triton_"):
                    return "triton_reduction" if ("red" in n.split("_")[1] or "per" in n.split("_")[1]) else "triton_pointwise"
                if "sort" in n or "radix" in n or "cub::" in n:
                    return "sort"
                if "memcpy" in n or "memset" in n:
                    return "memcpy"
                return "other"
            by = {}
            for k, (ms, c) in rows.items():
                by.setdefault(cls(k), [0.0, 0.0])
                by[cls(k)][0] += ms
                by[cls(k)][1] += c
            res["kernel_class_ms_per_step"] = {k: [round(v[0], 3), round(v[1], 1)] for k, v in sorted(by.items())}
            res["kernel_total_ms_per_step"] = round(sum(v[0] for v in by.values()), 3)
            res["kernel_launches_per_step"] = round(sum(v[1] for v in by.values()), 1)
            top = sorted(rows.items(), key=lambda kv: -kv[1][0])[:30]
            res["top_kernels_ms_per_step"] = [[k[:140], round(v[0], 3), round(v[1], 1)] for k, v in top]
    if a.t2_batches:
        res["t2_graph_replay_ms"] = {}
        for tb in [int(x) for x in a.t2_batches.split(",")]:
            ob = {k: v[:tb].clone() for k, v in obs.items()}
            torch._dynamo.reset()
            with torch.no_grad():
                fn(ob)                      # eager warm-up at this batch (records the no_other shapes)
            f2 = torch.compile(fn, dynamic=False, fullgraph=True)
            with torch.no_grad():
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        f2(ob)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    f2(ob)
                for _ in range(20):
                    g.replay()
                torch.cuda.synchronize()
                bl = []
                for _ in range(5):
                    t = time.time()
                    for _ in range(200):
                        g.replay()
                    torch.cuda.synchronize()
                    bl.append((time.time() - t) / 200 * 1000)
            res["t2_graph_replay_ms"][str(tb)] = {"blocks": [round(x, 4) for x in bl],
                                                   "median": round(statistics.median(bl), 4)}
            del g
    from torch._dynamo.utils import counters
    res["graph_breaks"] = dict(counters.get("graph_break", {}))
    res["unique_graphs"] = int(counters["stats"].get("unique_graphs", 0))
    res["cache_root"] = root
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "top_kernels_ms_per_step"}))


if __name__ == "__main__":
    main()
