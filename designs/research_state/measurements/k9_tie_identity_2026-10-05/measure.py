"""K9(b) tie IDENTITY measurement (2026-10-05): which cutoff excludes each row, and is the tie VALUE-IDENTICAL
(resolving it the other way changes nothing the forward computes) or GENUINELY DISTINCT?

CPU only (no GPU lease). Per arm, a real complete-game rollout on the Rust collector (self-check build,
T2 eager on CPU, p2 a seeded random policy on an external route, seeded pool teams), then the learner's
probe forward on every row under the tie-margin recorder, then FLIP forwards: at every declared MARGIN
call, each row's near-tied elements (margin < FP32_TIE_EPS) are resolved the OTHER way (a top-k's k-th
and (k+1)-th candidates swapped, an argmax's top-2 taken, a threshold's bool flipped, a sorted pair
swapped). A row whose full masked log-prob vector is BIT-IDENTICAL under every flip variant is a tie
that cannot move log pi; otherwise it is genuinely distinct.

Variants: JOINT (every near-tied element of the row flipped at once) and one SINGLE variant per call
index (only that call's elements). Every row with no flip must reproduce bit-for-bit in every variant
(the row-independence / determinism check; a failure is reported, never tolerated). The flips classify
the rows the rule BEFORE `gen3_behaviour_tie_identity_v1` excluded (payload identity stripped,
`old_rule`); ``new_rule`` reports the production recorder's exclusion (payload identity + the
SELECTION-FREE certificate) on the same rows.

    PYTHONPATH=src python measure.py --arm oracle_full --weights fresh --out result_oracle_full_fresh.json
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th
from torch.overrides import TorchFunctionMode

from agents.model import selection_sites as SS
from agents.training.rust_rollout import tie_margins as TM
from agents.training.rust_rollout.consistency import FP32_TIE_EPS

ARMS = {"blob": {}, "oracle_full": {"oracle_reveal": "full"}, "oracle_species": {"oracle_reveal": "species"},
        "fixed_mass": {"belief_tokens": "fixed_mass"}}


_CAPTURE: Optional[List[Tuple[Any, ...]]] = None
_site_margin = TM.site_margin


def _capturing_site_margin(rule, name, args, kwargs, payload=None, out=None):
    g = _site_margin(rule, name, args, kwargs, payload, out)
    if _CAPTURE is not None and g is not None:
        _CAPTURE.append((rule, name, args, kwargs))
    return g


TM.site_margin = _capturing_site_margin     # the recorder calls it by module attribute


@contextlib.contextmanager
def old_rule():
    """The rule BEFORE gen3_behaviour_tie_identity_v1: no payload identity on any MARGIN site."""
    saved = dict(SS.MARGIN)
    for k, r in list(SS.MARGIN.items()):
        if r.payload:
            SS.MARGIN[k] = r._replace(payload=())
    SS.line_map.cache_clear()
    try:
        yield
    finally:
        SS.MARGIN.clear()
        SS.MARGIN.update(saved)
        SS.line_map.cache_clear()


class Rec(TM.TieMargins):
    """TieMargins + per MARGIN call: site, rule, per-row min margin, and the op's operands."""

    def __init__(self, rows: int) -> None:
        super().__init__(rows, keep_calls=True)
        self.detail: List[Dict[str, Any]] = []

    def finish(self, captured) -> None:
        assert len(captured) == len(self.calls), (len(captured), len(self.calls))
        for (site, per), (rule, name, args, kwargs) in zip(self.calls, captured):
            self.detail.append({"site": site, "name": name, "rule": rule, "per": per, "args": args,
                                "kwargs": kwargs})


def tie_class(rule: SS.Rule, name: str, args, kwargs, row: int) -> str:
    """The class of row ``row``'s smallest-margin element at this call: exact-zero / exact-nonzero / near."""
    x = args[0].detach().double()[row:row + 1]
    a2 = tuple([x] + [a[row:row + 1] if isinstance(a, th.Tensor) and a.dim() > 0 and a.shape[0] == args[0].shape[0]
                      else a for a in args[1:]])
    g = TM.site_margin(rule, name, a2, kwargs)
    m = float(g.reshape(-1).min())
    if m > 0:
        return "near"
    # an exact tie: are the tied values 0?
    if rule.kind in ("topk",):
        k = int(TM._arg(a2, kwargs, 1, "k", 1))
        v = th.topk(x, k + 1, dim=-1).values
        gg = TM._rel(v[..., k - 1], v[..., k]).reshape(-1)
        j = int(th.argmin(gg))
        val = float(v[..., k - 1].reshape(-1)[j])
        return "exact0" if val == 0 else f"exact({val:.6g})"
    if rule.kind == "argmax":
        v = th.topk(x, 2, dim=int(TM._arg(a2, kwargs, 1, "dim", -1))).values
        return "exact0" if float(v.reshape(-1, 2)[int(th.argmin(g.reshape(-1)))][0]) == 0 else "exact"
    return "exact"


class Flip(TorchFunctionMode):
    """Re-run the forward; at MARGIN call index c, resolve the rows' near-tied elements the other way."""

    def __init__(self, rows: int, plan: Dict[int, np.ndarray], eps: float) -> None:
        super().__init__()
        self.rows, self.plan, self.eps = rows, plan, eps
        self.c = -1
        self.sites: List[str] = []
        self.changed = np.zeros(rows, dtype=bool)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        r = SS.runtime_op(getattr(func, "__name__", ""))
        if r is None or not args or not isinstance(args[0], th.Tensor):
            return out
        kind, _op = r
        name = getattr(func, "__name__", "")
        if kind == "sel":
            if not args[0].is_floating_point() or (name in ("max", "min") and not isinstance(out, tuple)):
                return out
        elif kind == "cmp":
            if not (isinstance(out, th.Tensor) and out.dtype == th.bool):
                return out
            other = args[1] if len(args) > 1 else kwargs.get("other")
            if not (args[0].is_floating_point() or (isinstance(other, th.Tensor) and other.is_floating_point())):
                return out
        else:
            if not (args[0].is_floating_point() and isinstance(out, th.Tensor)
                    and not out.is_floating_point() and out.dtype != th.bool):
                return out
        where = TM._caller()
        if where is None:
            return out
        res = SS.resolve(where[0], where[1], kind)
        if res is None or res.declared is None or res.declared.rule is None:
            return out
        rule = res.declared.rule
        g = _site_margin(rule, name, args, kwargs)
        if g is None:
            return out
        self.c += 1
        self.sites.append(f"{where[0]}.py:{where[1]} {name}")
        rows = self.plan.get(self.c)
        if rows is None or rows.size == 0:
            return out
        sel = th.zeros(self.rows, dtype=th.bool)
        sel[th.as_tensor(rows)] = True
        new = flip(rule, name, args, kwargs, out, g, sel, self.eps)
        a = out[1] if isinstance(out, tuple) else out
        b = new[1] if isinstance(new, tuple) else new
        ch = (a != b).reshape(self.rows, -1).any(dim=1).cpu().numpy()
        self.changed |= ch
        return new


def flip(rule, name, args, kwargs, out, g, sel, eps):
    x = args[0]
    B = x.shape[0]
    selb = sel.to(x.device).reshape((B,) + (1,) * (g.dim() - 1))
    near = (g < eps) & selb
    if rule.kind == "topk":
        k = int(TM._arg(args, kwargs, 1, "k", 1))
        dim = int(TM._arg(args, kwargs, 2, "dim", -1))
        assert dim in (-1, x.dim() - 1)
        vals, idx = out[0].clone(), out[1].clone()
        taken = th.zeros_like(x, dtype=th.bool).scatter_(-1, out[1], True)
        xm = th.where(taken, th.full_like(x, -float("inf")), x)
        av, ai = xm.max(dim=-1)                    # the best candidate NOT selected
        nk = near[..., 0]
        vals[..., k - 1] = th.where(nk, av, vals[..., k - 1])
        idx[..., k - 1] = th.where(nk, ai, idx[..., k - 1])
        return th.return_types.topk((vals, idx))
    if rule.kind == "argmax":
        dim = int(TM._arg(args, kwargs, 1, "dim", -1))
        assert dim in (-1, x.dim() - 1) and name == "argmax"
        o = out.clone()
        oi = o if o.dim() == x.dim() else o.unsqueeze(-1)
        taken = th.zeros_like(x, dtype=th.bool).scatter_(-1, oi, True)
        alt = th.where(taken, th.full_like(x, -float("inf")), x).argmax(dim=-1, keepdim=True)
        nk = near if near.dim() == x.dim() else near.unsqueeze(-1)
        res = th.where(nk[..., :1], alt, oi)
        return res if o.dim() == x.dim() else res.squeeze(-1)
    if rule.kind == "sort_head":
        # out = argsort indices [B, ..., N]; swap every adjacent near pair of each selected row's head
        N = x.shape[-1]
        o = out.clone().reshape(-1, N)
        v = th.sort(x.detach().double(), dim=-1).values.reshape(-1, N)
        per_row = v.shape[0] // B
        head = min(int(rule.head), N)
        a_, c_ = v[:, :head - 1], v[:, 1:head]
        gen = (a_ >= -1) & (a_ <= 0) & (c_ >= -1) & (c_ <= 0)
        if rule.zero_exact:
            gen &= ~((a_ == 0) & (c_ == 0))
        near2 = gen & (TM._rel(a_, c_) < eps) & sel.to(x.device).repeat_interleave(per_row)[:, None]
        for r_, p_ in th.nonzero(near2).tolist():
            o[r_, p_], o[r_, p_ + 1] = o[r_, p_ + 1].clone(), o[r_, p_].clone()
        return o.reshape(out.shape)
    if rule.kind == "threshold_self":
        t = args[1]
        xx, tt = th.broadcast_tensors(x, t)
        is_t = xx == tt
        mult = is_t.sum(-1, keepdim=True)
        # near (x != t): flip; multiplicity: drop the LAST element equal to t
        o = out.clone()
        fl = near & ~is_t
        last = th.zeros_like(is_t)
        pos = th.arange(x.shape[-1], device=x.device).expand_as(xx)
        lastpos = th.where(is_t, pos, th.full_like(pos, -1)).amax(-1, keepdim=True)
        last = (pos == lastpos) & (mult > 1) & selb
        o = th.where(fl | last, ~o, o)
        return o
    # threshold
    o = out.clone()
    return th.where(near.expand_as(o) if near.shape != o.shape else near, ~o, o)


def build(arm: str, weights: str, seed: int, n_envs: int, n_steps: int, run_seed: int):
    from agents.training.learner_golden import _one_thread
    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_rollout.build import RustEnvDecl, trainee_spaces
    from agents.training.rust_vec_env import RustVecEnv
    from main.train.production_args import production_args

    TK.build_selfcheck()
    args = production_args()
    for k, v in ARMS[arm].items():
        setattr(args, k, v)
    obs, act = trainee_spaces(args)
    decl = RustEnvDecl(n_envs=n_envs, threads=2, front="ffi", profile="selfcheck", n_steps=n_steps,
                       micro_batch=n_steps, device="cpu", backend="eager", run_seed=run_seed, gamma=1.0,
                       gae_lambda=0.8, oracle_reveal=ARMS[arm].get("oracle_reveal", "off"))
    p2 = TK.RandomP2(5)
    env = RustVecEnv(n_envs=n_envs, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl, p2=p2, builder=TK.pool_builder()))
    with _one_thread():
        if weights == "fresh":
            from agents.model.policy import Gen3DualHeadMaskablePolicy
            from agents.training.instrumented_ppo import InstrumentedMaskablePPO
            from main.fresh_checkpoint import _production_policy_kwargs
            _a, _l, pk = _production_policy_kwargs(args)
            th.manual_seed(seed)
            model = InstrumentedMaskablePPO(Gen3DualHeadMaskablePolicy, env, n_steps=n_steps, batch_size=n_steps,
                                            n_epochs=1, gamma=1.0, gae_lambda=0.8, device="cpu", seed=seed,
                                            policy_kwargs=pk, verbose=0)
            model.ep_info_buffer = collections.deque(maxlen=100)
            model.ep_success_buffer = collections.deque(maxlen=100)
        else:
            model = TK.fresh_model(env, n_steps=n_steps, batch_size=n_steps, seed=seed, perturb_seed=1234,
                                   policy_args=args, perturb_keyed=arm != "blob")
    col = env.startup(model)
    t0 = time.time()
    if not col.collect(model, TK.NullCallback(), model.rollout_buffer):
        raise RuntimeError("collector stopped")
    print(f"collected in {time.time() - t0:.0f}s", flush=True)
    return model, env


def forward(model, obs, acts, masks, mode):
    from stable_baselines3.common.utils import obs_as_tensor
    with th.no_grad(), mode:
        _v, lp, _e = model.policy.evaluate_actions(obs_as_tensor(obs, model.device), acts, action_masks=masks)
    logits = model.policy._last_pi_distribution
    logits = getattr(getattr(logits, "distribution", logits), "logits")
    return lp.detach().double().numpy(), logits.detach().double().numpy()


def same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """per row: bit-identical (NaN == NaN at the same position, -inf == -inf)."""
    eq = (a == b) | (np.isnan(a) & np.isnan(b))
    return eq.reshape(a.shape[0], -1).all(axis=1)


def main(argv=None) -> int:
    global _CAPTURE
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--weights", default="fresh", choices=["fresh", "perturbed"])
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--run-seed", type=int, default=1001)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    model, env = build(a.arm, a.weights, a.seed, a.n_envs, a.n_steps, a.run_seed)
    try:
        buf = model.rollout_buffer
        n_steps, n_envs = int(buf.buffer_size), int(buf.n_envs)
        N = n_steps * n_envs
        eps = FP32_TIE_EPS
        model.policy.set_training_mode(True)
        by_site = collections.Counter()
        by_site_class = collections.Counter()
        n_ex = n_cleared = n_cleared_joint_only = n_multi = 0
        determinism_fail = 0
        n_noop = 0
        n_new = n_new_payload = 0
        new_sites: collections.Counter = collections.Counter()
        new_payload_by_verdict: collections.Counter = collections.Counter()
        per_row: List[Dict[str, Any]] = []
        absd_all = []
        for s in range(0, N, a.chunk):
            f = np.arange(s, min(N, s + a.chunk))
            t, e = f // n_envs, f % n_envs
            obs = {k: v[t, e] for k, v in buf.observations.items()}
            acts = th.as_tensor(buf.actions[t, e].reshape(-1)).long()
            masks = th.as_tensor(buf.action_masks[t, e])
            # the NEW rule (production): payload identity + the selection-free certificate
            from agents.training.rust_rollout.consistency import selection_free
            rec_new = TM.TieMargins(f.size)
            forward(model, obs, acts, masks, rec_new)
            rec_new.check()
            ex_new = (rec_new.margin < eps) & (not selection_free(model.policy))
            ex_payload = rec_new.margin < eps
            n_new += int(ex_new.sum())
            n_new_payload += int(ex_payload.sum())
            for r in np.flatnonzero(ex_payload):
                new_sites[rec_new.site[int(r)]] += 1
            ctx = old_rule()
            ctx.__enter__()
            rec = Rec(f.size)
            _CAPTURE = []
            lp0, full0 = forward(model, obs, acts, masks, rec)
            cap, _CAPTURE = _CAPTURE, None
            rec.check()
            rec.finish(cap)
            absd_all.append(np.abs(lp0 - buf.log_probs[t, e].astype(np.float64)))
            # per row: which calls are near-tied
            calls_of: Dict[int, List[int]] = collections.defaultdict(list)
            for ci, d in enumerate(rec.detail):
                for r in np.flatnonzero(d["per"] < eps):
                    calls_of[int(r)].append(ci)
            # determinism of the baseline itself
            _lpb, fullb = forward(model, obs, acts, masks, Flip(f.size, {}, eps))
            assert same(full0, fullb).all(), "the baseline forward is not bit-reproducible"
            # JOINT variant
            plan: Dict[int, List[int]] = collections.defaultdict(list)
            for r, cs in calls_of.items():
                for ci in cs:
                    plan[ci].append(r)
            fm = Flip(f.size, {c: np.asarray(v) for c, v in plan.items()}, eps)
            _lpj, fullj = forward(model, obs, acts, masks, fm)
            assert fm.sites == [d["site"] for d in rec.detail], "flip forward's call sequence diverged"
            okj = same(full0, fullj)
            noop = np.zeros(f.size, dtype=bool)
            for r in calls_of:
                noop[r] = not fm.changed[r]
            n_noop += int(noop.sum())
            okj &= ~noop          # a flip that changed nothing proves nothing: the row stays excluded
            untouched = np.ones(f.size, dtype=bool)
            untouched[list(calls_of)] = False
            determinism_fail += int((~okj & untouched).sum())
            # SINGLE variants: one per call index with near-tied rows
            ok_all = okj.copy()
            for ci, rows in plan.items():
                fs = Flip(f.size, {ci: np.asarray(rows)}, eps)
                _lps, fulls = forward(model, obs, acts, masks, fs)
                oks = same(full0, fulls)
                determinism_fail += int((~oks & untouched).sum())
                for r in rows:
                    ok_all[r] &= oks[r]
            for r, cs in calls_of.items():
                n_ex += 1
                sites = sorted({rec.detail[c]["site"] for c in cs})
                cls = sorted({tie_class(rec.detail[c]["rule"], rec.detail[c]["name"],
                                        rec.detail[c]["args"], rec.detail[c]["kwargs"], r) for c in cs})
                if len(cs) > 1:
                    n_multi += 1
                verdict = "identical" if ok_all[r] else "distinct"
                if ok_all[r]:
                    n_cleared += 1
                elif okj[r]:
                    n_cleared_joint_only += 1
                key = " + ".join(sites)
                by_site[(key, verdict)] += 1
                by_site_class[(key, "/".join(cls), verdict)] += 1
                if ex_payload[r]:
                    new_payload_by_verdict[verdict] += 1
                per_row.append({"row": int(f[r]), "sites": sites, "n_calls": len(cs), "classes": cls,
                                "verdict": verdict, "joint_identical": bool(okj[r])})
            print(f"chunk {s}: excluded {len(calls_of)} / {f.size}", flush=True)
            ctx.__exit__(None, None, None)
            assert not (ex_payload & ~np.isin(np.arange(f.size), list(calls_of))).any(), "new rule excluded a new row"
            del rec
        absd = np.concatenate(absd_all)
        out = {"arm": a.arm, "weights": a.weights, "seed": a.seed, "run_seed": a.run_seed, "rows": N, "eps": eps,
               "excluded_before": n_ex, "excluded_frac_before": n_ex / N,
               "cleared_identical": n_cleared, "excluded_after": n_ex - n_cleared,
               "excluded_frac_after": (n_ex - n_cleared) / N,
               "rows_multi_call": n_multi, "joint_identical_but_single_distinct": n_cleared_joint_only,
               "determinism_failures_untouched_rows": determinism_fail, "flip_noop_rows": n_noop,
               "max_abs_dlogp_vs_stored": float(absd.max()),
               "by_site": [{"sites": k[0], "verdict": k[1], "rows": v} for k, v in by_site.most_common()],
               "by_site_class": [{"sites": k[0], "class": k[1], "verdict": k[2], "rows": v}
                                 for k, v in by_site_class.most_common()],
               "new_rule": {"excluded_payload_only": n_new_payload, "excluded_frac_payload_only": n_new_payload / N,
                            "excluded": n_new, "excluded_frac": n_new / N,
                            "payload_only_rows_by_flip_verdict": dict(new_payload_by_verdict),
                            "payload_only_rows_by_min_site": dict(new_sites.most_common())},
               "torch": th.__version__}
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(json.dumps({k: v for k, v in out.items() if k not in ("by_site_class",)}, indent=1))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
