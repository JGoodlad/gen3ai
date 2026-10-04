"""X5 hypothesis set: CUDA eager vs COMPILED vs CPU eager parity, and the stable-argsort tie order
(`--belief-tokens fixed_mass`; U2 hand-off check 2, 2026-10-04).

    GEN3AI_TEST_ALLOW_GPU=1 python cuda_parity.py --obs real_obs.npz --out parity.json [--shapes 8,64,256,2048]

WHAT IT COMPARES. The production-surface `fixed_mass` learner (built exactly as `hypothesis_set_test`'s
`_unperturbed_learner`: one build thread, no test perturbation), then the extractor's forward on REAL
rows (`gen_real_obs.py`: the Rust collector, 4,096 rows), reading `fe.last_hypothesis`:

  ref   CPU  eager fp32       (the reference)
  geag  CUDA eager fp32
  gcmp  CUDA compiled fp32    (`torch.compile(fn, dynamic=False, fullgraph=True)`, T2's flags, a private
                               hermetic Inductor cache; one graph per static shape, T2's buckets
                               8 / 64 / 256, plus 2,048 = the learner micro-batch, grad mode)

on three weight sets: COLD (as built: delta_theta's last layer is zero, so pi is the Smogon prior's
fixed-size marginal), PERT (`parity_probe.perturb_`, the gates' own seeded perturbation, scale 0.05:
every head informative) and STRESS (PERT + delta_theta's output layer at scale 0.5, so delta moves
the scores by several nats and the prior no longer dominates the ordering).

THE BARS (registered here BEFORE the run; the code's own deterministic tolerances):
  continuous probabilities / masses (pi, hyp_pi, OTHER mass, P(any tail), seat pi, tail probs):
        max |delta| <= 1e-5          (= the fp32 sum-pi-equals-k tolerance of the construction)
  log-presence (slot_log_pi, OTHER log-mass), finite entries:  max |delta| <= 1e-4
  embeddings (OTHER token, tail mean):                          max |delta| <= 1e-4  (the compile gate's
        fp32 feature bar)
  sum-pi residual |sum pi - k|:  <= 1e-5 on every device / mode
  exact: k, n, live, full, other_live (integers / structure)
  DISCRETE (selection): the seated hypotheses through the boundary (order[:k+1]), hyp_species,
        slot_species, rank of the seated species, and the move seats (nums / live / revealed) must be
        IDENTICAL on every row NOT in `near_tie_rows` (rule 8, SELECTION_TIE_EPS = 1e-6) of EITHER
        computation. The excluded count is reported. A mismatch on a non-excluded row FAILS.

THE TIE TEST (`ties`). 2,048 synthetic rows x 400 candidates whose scores take only 12 distinct values
(so ~33 candidates share each pi EXACTLY and the seat boundary lands inside a tie block most rows):
`stable_order` and `fixed_mass_presence` on CPU eager, CUDA eager and CUDA compiled must give the SAME
full permutation (equal pi -> lower num first, every row). Gaps between distinct values are >= 0.3 nats,
so no row is a rounding-error near-tie and nothing is excluded.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from agents.model.hypothesis_set import (BISECTION_ITERS, SELECTION_TIE_EPS, fixed_mass_presence,
                                         near_tie_rows, stable_order)
from agents.model.parity_probe import PERTURB_SCALE, PERTURB_SEED, perturb_

TOL_PROB = 1e-5
TOL_LOG = 1e-4
TOL_EMB = 1e-4
TOL_SUM = 1e-5


# ------------------------------------------------------------------------------------------ build
def build_fixed_mass_policy():
    from collections import deque

    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from agents.training.rust_rollout.build import trainee_spaces
    from agents.training.rust_vec_env import RustVecEnv
    from main.fresh_checkpoint import _production_policy_kwargs
    from main.train.production_args import production_args
    from utils.torch_state_guard import single_thread_build

    args = production_args()
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


def load_obs(path: str) -> Dict[str, torch.Tensor]:
    with np.load(path) as z:
        return {k[4:]: torch.as_tensor(z[k]) for k in z.files if k.startswith("obs:")}


def batch(obs: Dict[str, torch.Tensor], lo: int, hi: int, device: str) -> Dict[str, torch.Tensor]:
    return {k: v[lo:hi].to(device) for k, v in obs.items()}


# ------------------------------------------------------------------------------------- extraction
FIELDS_PROB = ("pi", "hyp_pi", "other_mass", "other_any", "tail_probs", "mv_seat_pi", "mv_other_mass",
               "mv_pi")
FIELDS_LOG = ("slot_log_pi", "other_log_mass", "mv_other_log_mass")
FIELDS_EMB = ("other_token", "other_tail_mean")
FIELDS_EXACT = ("k", "n", "live", "full", "other_live", "mv_k", "mv_other_live", "mv_r")
FIELDS_SEL = ("order_top", "hyp_species", "slot_species", "slot_is_hyp", "rank_seated", "mv_seat_nums",
              "mv_seat_live", "mv_seat_rev")


def extract(fe, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """One extractor forward, then every HypothesisSet field the checks read, as plain tensors (the
    compile target: the whole extractor forward is traced, the stash read inside the graph)."""
    fe(obs)
    hs = fe.last_hypothesis
    sp = hs.species
    out = {
        "pi": sp.pi, "k": sp.k, "n": sp.n, "live": sp.live, "full": sp.full,
        "hyp_pi": hs.hyp_pi, "other_mass": hs.other_mass, "other_any": hs.other_any,
        "tail_probs": hs.other_tail_probs, "other_live": hs.other_live,
        "slot_log_pi": hs.slot_log_pi, "other_log_mass": hs.other_log_mass,
        "other_token": hs.other_token, "other_tail_mean": hs.other_tail_mean,
        "hyp_species": hs.hyp_species, "slot_species": hs.slot_species,
        "slot_is_hyp": hs.slot_is_hypothesis,
        "order_top": hs.order[:, :7], "rank": hs.rank,
        "species_tie_gap": hs.species_tie_gap,
    }
    mv = hs.moves
    if mv is not None:
        out.update({"mv_seat_pi": mv.seat_pi, "mv_other_mass": mv.other_mass,
                    "mv_other_log_mass": mv.other_log_mass, "mv_pi": mv.presence.pi,
                    "mv_k": mv.presence.k, "mv_other_live": mv.other_live, "mv_r": mv.r,
                    "mv_seat_nums": mv.seat_nums, "mv_seat_live": mv.seat_live,
                    "mv_seat_rev": mv.seat_revealed, "mv_tie_gap": mv.tie_gap})
    return out


def run_chunks(fn: Callable, obs: Dict[str, torch.Tensor], bsz: int, device: str,
               grad: bool = False) -> Dict[str, torch.Tensor]:
    """Run ``fn`` over ``obs`` in chunks of exactly ``bsz`` rows (static shapes; N must divide)."""
    n = next(iter(obs.values())).shape[0]
    assert n % bsz == 0, (n, bsz)
    parts: List[Dict[str, torch.Tensor]] = []
    for lo in range(0, n, bsz):
        b = batch(obs, lo, lo + bsz, device)
        if grad:
            r = fn(b)
        else:
            with torch.no_grad():
                r = fn(b)
        parts.append({k: v.detach().cpu() for k, v in r.items()})
    return {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}


# ----------------------------------------------------------------------------------- comparisons
def compare(ref: Dict[str, torch.Tensor], got: Dict[str, torch.Tensor], k_seat: torch.Tensor
            ) -> Dict[str, Any]:
    """The registered comparison of two computations of the hypothesis set. Returns the numbers and
    the per-field PASS flags; ``excluded`` is the rule-8 near-tie row count (the union of both sides)."""
    near = (ref["species_tie_gap"] < SELECTION_TIE_EPS) | (got["species_tie_gap"] < SELECTION_TIE_EPS)
    if "mv_tie_gap" in ref:
        near = near | (ref["mv_tie_gap"] < SELECTION_TIE_EPS) | (got["mv_tie_gap"] < SELECTION_TIE_EPS)
    keep = ~near
    res: Dict[str, Any] = {"rows": int(near.numel()), "excluded_near_tie": int(near.sum())}
    ok = True

    def mx(a, b):
        return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0

    for f in FIELDS_PROB:
        if f in ref:
            d = mx(ref[f], got[f]); res[f] = d; ok &= d <= TOL_PROB
    for f in FIELDS_LOG:
        if f in ref:
            fin = (ref[f] > -1e8) & (got[f] > -1e8)
            same_mask = bool(((ref[f] > -1e8) == (got[f] > -1e8)).all())
            d = mx(ref[f][fin], got[f][fin]); res[f] = d; res[f + "_mask_equal"] = same_mask
            ok &= (d <= TOL_LOG) and same_mask
    for f in FIELDS_EMB:
        d = mx(ref[f], got[f]); res[f] = d; ok &= d <= TOL_EMB
    for f in FIELDS_EXACT:
        if f in ref:
            e = bool(torch.equal(ref[f], got[f])); res[f + "_equal"] = e; ok &= e
    # selection (rule 8): identical on every row not near-tied
    kk = k_seat.clamp(max=6)
    pos = torch.arange(7).unsqueeze(0)
    thru = pos < (kk + 1).unsqueeze(-1)                                   # order[:k+1]
    sel_ok = {}
    sel_ok["order_top"] = bool((((ref["order_top"] == got["order_top"]) | ~thru).all(-1) | near).all())
    seated_ref = ref["rank"] < kk.unsqueeze(-1)                           # [B,S] the k seated species
    sel_ok["rank_seated"] = bool((((ref["rank"] == got["rank"]) | ~seated_ref).all(-1) | near).all())
    for f in ("hyp_species", "slot_species", "slot_is_hyp"):
        sel_ok[f] = bool(((ref[f] == got[f]).all(-1) | near).all())
    for f in ("mv_seat_nums", "mv_seat_live", "mv_seat_rev"):
        if f in ref:
            sel_ok[f] = bool(((ref[f] == got[f]).all(-1) | near).all())
    # how many NON-excluded rows disagree, per selection field (0 expected)
    mism = {f: int(((~(ref[f] == got[f]).reshape(ref[f].shape[0], -1).all(-1)) & keep).sum())
            for f in ("hyp_species", "slot_species", "mv_seat_nums") if f in ref}
    # the rows rule 8 EXCLUDES are not judged; report anyway how many still agree exactly (informational)
    if bool(near.any()):
        agree = torch.ones(int(near.sum()), dtype=torch.bool)
        for f in ("hyp_species", "slot_species", "mv_seat_nums"):
            if f in ref:
                agree &= (ref[f][near] == got[f][near]).reshape(int(near.sum()), -1).all(-1)
        res["excluded_rows_still_identical_selection"] = int(agree.sum())
    res["selection_ok"] = sel_ok
    res["selection_mismatch_rows_kept"] = mism
    ok &= all(sel_ok.values())
    # sum-pi residual (per computation, informational + bar)
    for tag, d in (("ref", ref), ("got", got)):
        live = d["live"]
        resid = (d["pi"].double().sum(-1) - d["k"].double()).abs()
        res[f"sum_pi_resid_{tag}"] = float(resid[live].max()) if bool(live.any()) else 0.0
        ok &= res[f"sum_pi_resid_{tag}"] <= TOL_SUM
    res["PASS"] = bool(ok)
    return res


def informative(ref: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Spread of the compared quantities on this weight set (a parity check on a quantity with no
    spread is vacuous — `parity_probe.require_informative`'s point)."""
    live = ref["live"]
    pi = ref["pi"][live]
    return {"live_rows": int(live.sum()), "pi_max": float(pi.max()), "pi_row_max_std": float(ref["pi"].max(-1).values[live].std()),
            "other_mass_min": float(ref["other_mass"][ref["other_live"]].min()),
            "other_mass_max": float(ref["other_mass"][ref["other_live"]].max()),
            "other_mass_std": float(ref["other_mass"][ref["other_live"]].std())}


# ---------------------------------------------------------------------------------- the tie test
def tie_inputs(B: int = 2048, N: int = 400, seed: int = 11) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    levels = torch.arange(12, dtype=torch.float32) * 0.37 - 2.0
    scores = levels[torch.randint(0, 12, (B, N), generator=g)]
    r = torch.randint(0, 7, (B,), generator=g)
    cand = torch.zeros(B, N, dtype=torch.bool)
    cand[:, 1:387] = True                                                  # nums 1..386
    for b in range(B):                                                    # r revealed nums removed
        drop = torch.randperm(386, generator=g)[: int(r[b])] + 1
        cand[b, drop] = False
    k = 6 - r
    return scores, cand, k


def tie_fn(scores, cand, k):
    p = fixed_mass_presence(scores, cand, k)
    return p.pi, stable_order(p.pi, p.cand), p.live


def tie_test(device_cuda: str) -> Dict[str, Any]:
    scores, cand, k = tie_inputs()
    res: Dict[str, Any] = {}
    ref_pi, ref_ord, ref_live = tie_fn(scores, cand, k)
    # tie census on the reference: the seat boundary (positions k-1 | k) inside an exact tie block
    srt = torch.gather(ref_pi, 1, ref_ord)
    b = torch.arange(srt.shape[0])
    kk = k.clamp(1, 385)
    straddle = (srt[b, kk - 1] == srt[b, kk]) & ref_live
    res["rows"] = int(srt.shape[0]); res["live_rows"] = int(ref_live.sum())
    res["rows_with_boundary_inside_an_exact_tie_block"] = int(straddle.sum())
    res["distinct_pi_values_per_row_median"] = float(torch.tensor([len(torch.unique(ref_pi[i][cand[i]])) for i in range(0, 2048, 64)]).median())
    # CPU eager must itself be the lower-num-first order on every tie block
    first_of_tie = True
    for i in range(0, 2048, 7):
        nc = int(cand[i].sum())
        o = ref_ord[i][:nc]                                   # the candidates' positions (non-candidates last)
        p = ref_pi[i][o]
        same = p[1:] == p[:-1]
        first_of_tie &= bool((o[1:][same] > o[:-1][same]).all())
    res["cpu_ties_resolve_to_lower_num"] = bool(first_of_tie)
    dev = torch.device(device_cuda)
    s_g, c_g, k_g = scores.to(dev), cand.to(dev), k.to(dev)
    with torch.no_grad():
        e_pi, e_ord, e_live = tie_fn(s_g, c_g, k_g)
        comp = torch.compile(tie_fn, dynamic=False, fullgraph=True)
        t0 = time.time()
        c_pi, c_ord, c_live = comp(s_g, c_g, k_g)
        torch.cuda.synchronize()
        res["tie_compile_wall_s"] = round(time.time() - t0, 2)
    res["cuda_eager_order_equals_cpu"] = bool(torch.equal(e_ord.cpu(), ref_ord))
    res["cuda_compiled_order_equals_cpu"] = bool(torch.equal(c_ord.cpu(), ref_ord))
    res["cuda_eager_order_equals_cuda_compiled"] = bool(torch.equal(e_ord, c_ord))
    res["cuda_eager_pi_maxdiff_vs_cpu"] = float((e_pi.cpu().double() - ref_pi.double()).abs().max())
    res["cuda_compiled_pi_maxdiff_vs_cpu"] = float((c_pi.cpu().double() - ref_pi.double()).abs().max())
    # exact ties stay exact on every device (equal scores -> bit-equal pi within one computation)
    def ties_exact(pi, o):
        bad = 0
        for i in range(0, 2048, 5):
            m = cand[i]
            sc = scores[i][m]; pv = pi[i].cpu()[m]
            for lv in torch.unique(sc):
                if len(torch.unique(pv[sc == lv])) != 1:
                    bad += 1
        return bad
    res["tie_blocks_not_bit_equal_cpu"] = ties_exact(ref_pi, ref_ord)
    res["tie_blocks_not_bit_equal_cuda_eager"] = ties_exact(e_pi, e_ord)
    res["tie_blocks_not_bit_equal_cuda_compiled"] = ties_exact(c_pi, c_ord)
    # a pure sort check too: argsort(stable) of a key with huge exact-tie blocks, CPU vs CUDA, eager and compiled
    key = torch.randint(0, 5, (2048, 400), generator=torch.Generator().manual_seed(3)).float()
    cm = torch.ones(2048, 400, dtype=torch.bool)
    o_cpu = stable_order(key, cm)
    o_cu = stable_order(key.to(dev), cm.to(dev)).cpu()
    o_cc = torch.compile(stable_order, dynamic=False, fullgraph=True)(key.to(dev), cm.to(dev)).cpu()
    res["raw_sort_5_levels_cuda_eager_equals_cpu"] = bool(torch.equal(o_cpu, o_cu))
    res["raw_sort_5_levels_cuda_compiled_equals_cpu"] = bool(torch.equal(o_cpu, o_cc))
    res["PASS"] = bool(res["cpu_ties_resolve_to_lower_num"] and res["cuda_eager_order_equals_cpu"]
                       and res["cuda_compiled_order_equals_cpu"] and res["raw_sort_5_levels_cuda_eager_equals_cpu"]
                       and res["raw_sort_5_levels_cuda_compiled_equals_cpu"]
                       and res["tie_blocks_not_bit_equal_cpu"] == 0 and res["tie_blocks_not_bit_equal_cuda_eager"] == 0
                       and res["tie_blocks_not_bit_equal_cuda_compiled"] == 0)
    return res


# ----------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="8,64,256,2048")
    ap.add_argument("--skip-compiled", action="store_true")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "no CUDA (set GEN3AI_TEST_ALLOW_GPU=1 under the GPU lease)"
    shapes = [int(s) for s in a.shapes.split(",")]
    out: Dict[str, Any] = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                           "bisection_iters": BISECTION_ITERS, "selection_tie_eps": SELECTION_TIE_EPS,
                           "bars": {"prob": TOL_PROB, "log": TOL_LOG, "emb": TOL_EMB, "sum_pi": TOL_SUM}}
    obs = load_obs(a.obs)
    n = next(iter(obs.values())).shape[0]
    print(f"[parity] {n} real rows; torch {torch.__version__}; {out['device']}", flush=True)

    model = build_fixed_mass_policy()
    pol = model.policy
    pol.eval()
    fe = pol.features_extractor
    assert fe.belief_tokens == "fixed_mass"
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("x5 cuda parity")

    base_state = {k: v.clone() for k, v in pol.state_dict().items()}

    def set_weights(kind: str) -> None:
        pol.to("cpu")                      # perturbation noise is drawn on CPU (a private generator)
        pol.load_state_dict(base_state)
        if kind in ("pert", "stress"):
            perturb_(pol, seed=PERTURB_SEED, scale=PERTURB_SCALE)
        if kind == "stress":
            hb = fe.hypothesis_builder
            g = torch.Generator().manual_seed(77)
            with torch.no_grad():
                hb.delta_out.weight.add_(torch.randn(hb.delta_out.weight.shape, generator=g) * 0.5)
                hb.delta_out.bias.add_(torch.randn(hb.delta_out.bias.shape, generator=g) * 0.5)

    # the compiled graphs: parameters are graph INPUTS, so switching weight sets neither recompiles
    # nor escapes the graph. Compile once per shape on CUDA; the policy lives on CUDA for those runs.
    # CPU references run first (the policy on CPU), per weight set.
    refs: Dict[str, Dict[str, torch.Tensor]] = {}
    for kind in ("cold", "pert", "stress"):
        set_weights(kind)
        t0 = time.time()
        refs[kind] = run_chunks(lambda b: extract(fe, b), obs, 256, "cpu")
        print(f"[parity] CPU eager ref {kind}: {time.time() - t0:.1f}s; {informative(refs[kind])}", flush=True)
        out.setdefault("informative", {})[kind] = informative(refs[kind])
        r = refs[kind]
        out["informative"][kind]["revealed_count_hist"] = torch.bincount(6 - r["k"], minlength=7).tolist()
        if "mv_k" in r:
            out["informative"][kind]["move_group_r_hist"] = torch.bincount(r["mv_r"], minlength=5).tolist()
    pol.to("cuda")
    fe_cuda = pol.features_extractor

    results: Dict[str, Any] = {}
    compiled_fns: Dict[Tuple[int, bool], Callable] = {}
    for kind in ("cold", "pert", "stress"):
        set_weights(kind)
        pol.to("cuda")
        # CUDA eager at 256 (no static-shape concern)
        eag = run_chunks(lambda b: extract(fe_cuda, b), obs, 256, "cuda")
        results[f"{kind}/cuda_eager_vs_cpu_eager"] = compare(refs[kind], eag, refs[kind]["k"])
        print(f"[parity] {kind} cuda_eager_vs_cpu: PASS={results[f'{kind}/cuda_eager_vs_cpu_eager']['PASS']} "
              f"excluded={results[f'{kind}/cuda_eager_vs_cpu_eager']['excluded_near_tie']}", flush=True)
        if a.skip_compiled:
            continue
        for bsz in shapes:
            grad = bsz == 2048
            key = (bsz, grad)
            if key not in compiled_fns:
                def fn(b, _fe=fe_cuda):
                    return extract(_fe, b)
                compiled_fns[key] = torch.compile(fn, dynamic=False, fullgraph=True)
                # compile (first call) timed separately
                b0 = batch(obs, 0, bsz, "cuda")
                torch.cuda.synchronize(); t0 = time.time()
                if grad:
                    compiled_fns[key](b0)
                else:
                    with torch.no_grad():
                        compiled_fns[key](b0)
                torch.cuda.synchronize()
                out.setdefault("compile_wall_s", {})[f"B{bsz}{'_grad' if grad else ''}"] = round(time.time() - t0, 1)
                print(f"[parity] compiled B={bsz} grad={grad}: first call {time.time() - t0:.1f}s", flush=True)
            got = run_chunks(compiled_fns[key], obs, bsz, "cuda", grad=grad)
            tag = f"{kind}/cuda_compiled_B{bsz}{'_grad' if grad else ''}"
            results[tag + "_vs_cuda_eager"] = compare(eag, got, eag["k"])
            results[tag + "_vs_cpu_eager"] = compare(refs[kind], got, refs[kind]["k"])
            print(f"[parity] {tag}: vs cuda eager PASS={results[tag + '_vs_cuda_eager']['PASS']} "
                  f"(excl {results[tag + '_vs_cuda_eager']['excluded_near_tie']}); vs cpu PASS="
                  f"{results[tag + '_vs_cpu_eager']['PASS']}", flush=True)
    out["comparisons"] = results
    out["ties"] = tie_test("cuda")
    print(f"[parity] ties: {out['ties']}", flush=True)
    out["ALL_PASS"] = bool(all(v["PASS"] for v in results.values()) and out["ties"]["PASS"])
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[parity] ALL_PASS={out['ALL_PASS']} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
