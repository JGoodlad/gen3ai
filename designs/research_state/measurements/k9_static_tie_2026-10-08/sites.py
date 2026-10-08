"""K9(b) static-screen u1480: per PRODUCTION-excluded row, which stable_order CALLER / pair / cut, and does
resolving ONLY the production-counted near pairs the other way move the full masked log-probs?

Run with cwd = the checkout under test; args: --weights-from <pt> --out <json> [--n-envs --n-steps --run-seed]."""
import collections
import importlib.util
import json
import os
import sys

import numpy as np
import torch as th

ROOT = os.getcwd()
p = os.path.join(ROOT, "designs/research_state/measurements/k9_early_probe_2026-10-06/measure.py")
spec = importlib.util.spec_from_file_location("k9_early", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
old = m.old
old.ARMS["static_fm"] = {"belief_tokens": "fixed_mass", "token_encoding": "static"}
TM, SS = old.TM, old.SS
from agents.training.rust_rollout.consistency import FP32_TIE_EPS as EPS  # noqa: E402

_CAP = None
_LAST = {"caller": None}
_orig_sm = TM.site_margin
_orig_cd = TM.consumed_decl


def _cd(rule, frame, site):
    c = _orig_cd(rule, frame, site)
    _LAST["caller"] = None if frame is None or frame.f_back is None else \
        f"{frame.f_back.f_code.co_name}:{frame.f_back.f_lineno}"
    _LAST["consumed"] = c
    loc = frame.f_back.f_locals if frame is not None and frame.f_back is not None else {}
    ctx = loc.get("ctx")
    _LAST["addr"] = None if ctx is None else ctx.opp_addressable.detach().clone()
    _LAST["hp"] = None if ctx is None else ctx.hp_and_active[:, 6:12, 0].detach().clone()
    hs = loc.get("hs")
    _LAST["hyp"] = None if hs is None or ctx is None else hs.slot_is_hypothesis.detach().clone()
    _LAST["spc"] = None if hs is None or ctx is None else hs.slot_species.detach().clone()
    _LAST["bel"] = None if ctx is None else ctx.opp_believed_mask.bool().detach().clone()
    _LAST["act"] = None if ctx is None else ctx.opp_active_local.detach().clone()
    return c


def _sm(rule, name, args, kwargs, payload=None, out=None, consumed=None):
    g = _orig_sm(rule, name, args, kwargs, payload, out, consumed)
    if _CAP is not None and g is not None:
        caller = _LAST["caller"] if rule.kind == "sort_head" else None
        _CAP.append({"rule": rule, "name": name, "x": args[0].detach().double().clone(), "kwargs": kwargs,
                     "args": args, "consumed": consumed, "caller": caller, "addr": _LAST.get("addr") if caller else None, "hp": _LAST.get("hp") if caller else None,
                     "hyp": _LAST.get("hyp") if caller else None, "bel": _LAST.get("bel") if caller else None,
                     "act": _LAST.get("act") if caller else None, "spc": _LAST.get("spc") if caller else None,
                     "out": (out[1] if isinstance(out, tuple) else out).clone() if out is not None else None})
        _LAST["caller"] = None
    return g


TM.site_margin = _sm
TM.consumed_decl = _cd


def counted_near(rule, x, consumed, B):
    """[B*per, head-1] the production-counted near pairs of a sort_head call (mirrors site_margin)."""
    N = x.shape[-1]
    v = th.sort(x, dim=-1).values.narrow(-1, 0, min(int(rule.head), N))
    a, b = v[..., :-1], v[..., 1:]
    gen = (a >= -1) & (a <= 0) & (b >= -1) & (b <= 0)
    if rule.zero_exact:
        gen &= ~((a == 0) & (b == 0))
    if consumed is not None:
        i = th.arange(a.shape[-1])
        if isinstance(consumed, th.Tensor):
            gen &= i < consumed.long().unsqueeze(-1)
        else:
            gen &= th.isin(i + 1, th.tensor([int(c) for c in consumed]))
    near = gen & (TM._rel(a, b) < EPS)
    return near.reshape(-1, near.shape[-1]), v.reshape(-1, v.shape[-1])


_orig_flip = old.flip


def _prod_flip(rule, name, args, kwargs, out, g, sel, eps):
    """old.flip, with a sort_head resolved on the PRODUCTION-counted near pairs only (its consumed read
    from the issuing stable_order frame)."""
    if rule.kind != "sort_head":
        return _orig_flip(rule, name, args, kwargs, out, g, sel, eps)
    import sys as _s
    f = _s._getframe(1)
    while f is not None and f.f_code.co_name != "stable_order":
        f = f.f_back
    assert f is not None, "sort_head flip outside stable_order"
    consumed = f.f_locals.get("consumed")
    x = args[0].detach().double()
    B = sel.shape[0]
    near, _v = counted_near(rule, x, consumed, B)
    per = near.shape[0] // B
    near &= sel.repeat_interleave(per)[:, None]
    N = x.shape[-1]
    o = out.clone().reshape(-1, N)
    for r_, p_ in th.nonzero(near).tolist():
        o[r_, p_], o[r_, p_ + 1] = o[r_, p_ + 1].clone(), o[r_, p_].clone()
    return o.reshape(out.shape)


old.flip = _prod_flip


class PFlip(old.Flip):
    def __init__(self, rows, plan, caps=None):
        super().__init__(rows, plan, EPS)


def main():
    global _CAP
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--run-seed", type=int, default=1001)
    ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()
    m._WEIGHTS_FROM = a.weights_from
    model, env = m.build("static_fm", None, 1001, a.n_envs, a.n_steps, a.run_seed)
    try:
        buf = model.rollout_buffer
        n_steps, n_envs = int(buf.buffer_size), int(buf.n_envs)
        N = n_steps * n_envs
        model.policy.set_training_mode(True)
        tally = collections.Counter()
        pair_tally = collections.Counter()
        examples = []
        n_ex = 0
        det_fail = 0
        for s in range(0, N, a.chunk):
            f = np.arange(s, min(N, s + a.chunk))
            t, e = f // n_envs, f % n_envs
            obs = {k: v[t, e] for k, v in buf.observations.items()}
            acts = th.as_tensor(buf.actions[t, e].reshape(-1)).long()
            masks = th.as_tensor(buf.action_masks[t, e])
            rec = TM.TieMargins(f.size, keep_calls=True)
            _CAP = []
            lp0, full0 = old.forward(model, obs, acts, masks, rec)
            caps, _CAP = _CAP, None
            rec.check()
            assert len(caps) == len(rec.calls), (len(caps), len(rec.calls))
            ex = np.flatnonzero(rec.margin < EPS)
            n_ex += ex.size
            calls_of = collections.defaultdict(list)
            for ci, (site, per) in enumerate(rec.calls):
                for r in np.flatnonzero(per < EPS):
                    calls_of[int(r)].append(ci)
            plan = collections.defaultdict(list)
            for r, cs in calls_of.items():
                for ci in cs:
                    plan[ci].append(r)
            # baseline determinism
            _l, fb = old.forward(model, obs, acts, masks, PFlip(f.size, {}, caps))
            assert old.same(full0, fb).all()
            fj = PFlip(f.size, {c: np.asarray(v) for c, v in plan.items()}, caps)
            _l, fullj = old.forward(model, obs, acts, masks, fj)
            assert fj.sites == [c[0] for c in rec.calls], "flip call sequence diverged"
            okj = old.same(full0, fullj)
            okj &= np.array([fj.changed[i] or i not in calls_of for i in range(f.size)])  # a no-op flip proves nothing
            untouched = np.ones(f.size, dtype=bool)
            untouched[list(calls_of)] = False
            det_fail += int((~okj & untouched).sum())
            ok_all = okj.copy()
            for ci, rows in plan.items():
                fs = PFlip(f.size, {ci: np.asarray(rows)}, caps)
                _l, fulls = old.forward(model, obs, acts, masks, fs)
                oks = old.same(full0, fulls)
                det_fail += int((~oks & untouched).sum())
                for r in rows:
                    ok_all[r] &= oks[r]
            for r, cs in calls_of.items():
                desc = []
                for ci in cs:
                    cap = caps[ci]
                    site = rec.calls[ci][0]
                    if cap["rule"].kind == "sort_head":
                        near, v = counted_near(cap["rule"], cap["x"], cap["consumed"], f.size)
                        per = near.shape[0] // f.size
                        nr = near[r * per:(r + 1) * per]
                        pos = th.nonzero(nr).tolist()
                        cons = cap["consumed"]
                        ctag = ("k" if isinstance(cons, th.Tensor) else ("cuts" + str(tuple(cons)) if cons is not None
                                                                          else "none"))
                        order = cap["out"].reshape(-1, cap["x"].shape[-1])[r * per:(r + 1) * per]
                        xs = cap["x"].reshape(-1, cap["x"].shape[-1])[r * per:(r + 1) * per]
                        for sub, pp in pos:
                            ia, ib = int(order[sub, pp]), int(order[sub, pp + 1])
                            d = {"caller": cap["caller"], "consumed": ctag, "sub": sub, "pair": pp,
                                 "ka": float(xs[sub, ia]), "kb": float(xs[sub, ib]), "ia": ia, "ib": ib}
                            if cap["addr"] is not None:
                                d["addr"] = bool(cap["addr"][r, sub]); d["hp"] = float(cap["hp"][r, sub])
                                d["bel"] = bool(cap["bel"][r, sub]); d["act"] = int(cap["act"][r]) == sub
                                d["hyp"] = None if cap["hyp"] is None else bool(cap["hyp"][r, sub])
                                d["spc"] = None if cap["spc"] is None else int(cap["spc"][r, sub])
                                pair_tally[(f"addr={d['addr']} hp>0={d['hp'] > 0} bel={d['bel']} hyp={d['hyp']} act={d['act']}",
                                            "", "", "id" if ok_all[r] else "DIST")] += 1
                            if isinstance(cons, th.Tensor):
                                d["k"] = int(cons.reshape(-1)[r]) if cons.dim() == 1 else None
                            desc.append(d)
                            pair_tally[(cap["caller"], ctag, pp, "id" if ok_all[r] else "DIST")] += 1
                    else:
                        desc.append({"site": site})
                callers = sorted({(d.get("caller") or d.get("site")) for d in desc})
                tally[(" + ".join(c.split(":")[0] if c else "?" for c in callers),
                       "identical" if ok_all[r] else "distinct")] += 1
                if len(examples) < 400:
                    dj = np.abs(np.nan_to_num(full0[r], neginf=0) - np.nan_to_num(fullj[r], neginf=0)).max()
                    examples.append({"dlogp_joint": float(dj), "env": int(e[r]), "row": int(f[r]), "verdict": "identical" if ok_all[r] else "distinct",
                                     "joint_identical": bool(okj[r]), "desc": desc})
            print(f"chunk {s}: excluded {ex.size}", flush=True)
        out = {"rows": N, "excluded": n_ex, "excluded_frac": n_ex / N, "determinism_failures": det_fail,
               "by_caller": [{"callers": k[0], "verdict": k[1], "rows": v} for k, v in tally.most_common()],
               "by_pair": [{"caller": k[0], "consumed": k[1], "pair": k[2], "verdict": k[3], "n": v}
                           for k, v in pair_tally.most_common()],
               "examples": examples}
        json.dump(out, open(a.out, "w"), indent=1)
        print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=1))
    finally:
        env.close()


if __name__ == "__main__":
    main()
