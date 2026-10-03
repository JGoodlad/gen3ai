"""X5 Tier 0: the ORACLE COUNTERFACTUAL on the physics channel. CPU only, no training, no checkpoint.

For every banked decision with >= 1 hidden opponent mon (and an opponent active + a live active of ours, where the
op is not gated to zero), it prices the hidden opponent mons under each REPRESENTATION, with the production op's
own formulas (kernels.py, bit-exact against the op in parity.py), and compares with the TRUTH:

  R0      today's blob: every hidden seat = the AVERAGED defender over the belief marginal (production: T0); the
          hidden attacker is gated off (threat 0); P(KO) nulled.
  R1_H    X5, M3 (c): the top-H species by pi as concrete mons (weight pi), plus OTHER with mass k - sum(named pi),
          priced as the averaged defender AND the averaged attacker (kernels.averaged_attacker) over the renormalised
          tail of pi; OTHER's P(KO) nulled (today's hidden-path semantics, which (c) inherits).
  R2_H    X5, M3 (d): the same named list, OTHER with NO physics (contributes nothing).
  OP_H    PERFECT OTHER PHYSICS: OTHER keeps its belief mass (k - sum named pi) but is priced as the TRUE hidden
          mons the list missed (their mean; their worst member for the max-type reads). Mass-conserving, so
          1 - err(OP_H)/err(R1_H) is the share of R1_H's error that OTHER's tail representation causes.
  OO_H    the ORACLE-OTHER counterfactual: the named list as in R1_H, and OTHER replaced by the TRUE hidden mons the
          list missed (weight 1 each): OTHER perfect in mass AND physics. NOT mass-conserving, and the named list's
          excess mass equals OTHER's deficit, so the two errors partly cancel inside R1_H and removing one can
          raise the total: read OO_H as the literal-oracle bound, OP_H as the decomposition.
  Rinf    every species named (no OTHER, no averaging anywhere): the expectation of the exact physics under the
          belief. The floor ANY budget can reach under that belief.
  T       the truth: the actual hidden mons, weight 1 each (pi = 1 on the truth, OTHER empty).

Budgets H: 1:1 (= k = 6 - r), k + 6, 12, 18. pi = the logistic fixed-size marginals (design §3.2), for three
beliefs: the Smogon prior (cold start) and the revision's two pool-memorising proxies (proxies.py).

Per-entity features (fractions; reported in percentage points):
  out[k]   our move k's max-roll damage on a full-HP switch-in, % of ITS max HP;
  ko[k]    acc x P(our move k OHKOs that full-HP switch-in);
  in       the switch-in's worst hit on our active (D4 cell, max of phys/spec presence-scaled max over its top-6
           candidates), % of OUR max HP;
  bulkP/S  E[maxhp] x E[def] / E[spd] (the op's defender bulk inputs), reported as relative error %;
  margin   best_out x our_hp_frac - in  (> 0: we KO it in fewer hits than it needs; speed and integer hits ignored).
Summaries over the hidden set: the presence-weighted MEAN, (1/k) sum_e w_e f_e (what an expectation-type reduction
reads), and for `in` also the presence-scaled MAX, max_e min(1, w_e) f_e (M2 = C; OTHER's presence capped at 1 —
the note does not say how a class-M site weighs a mass above 1: FINDING). For comparison only, `in_emax` is M2
option A: the expected max under independent presence (noisy-OR), sum_j v_(j) p_(j) prod_{l<j} (1 - p_(l)) over the
entities sorted by value (OTHER at presence min(1, mass)); its target is the truth's max.

Decision proxies (flip = the representation's answer differs from the truth's):
  P1 best move     argmax over our usable damaging moves (>= 2) of the mean out[k];
  P2 safe (max)    in_max < our current HP fraction;
  P2e safe (mean)  in_mean < our current HP fraction;
  P2a safe (E max) in_emax < our current HP fraction (M2 option A, for comparison);
  P3 KO race       margin_mean > 0.
Rule 8: a decision is EXCLUDED from a proxy when the truth's OR the representation's value lies within BAND = 1 pp
of HP of the proxy's boundary (P1: top-1 minus top-2 < BAND); the excluded share is reported per cell.
"""
from __future__ import annotations

import sys

if __name__ == "__main__" and "--check" in sys.argv:      # the readout gate's input check: no torch, no compute
    from utils.paths import repo_path
    _need = [repo_path("designs", "research_state", "measurements", "m5_laneS", "bank_v1", f)
             for f in ("manifest.json", "battles.jsonl.gz", "decisions.jsonl.gz")]
    _need += [repo_path("data", "teams", "gen3_team_archetypes.json"),
              repo_path("designs", "research_state", "measurements", "x5_revision_2026-10-03", "out", "m1m3.log")]
    _here = __import__("pathlib").Path(__file__).resolve().parent
    _need += [_here / f for f in ("common.py", "kernels.py", "proxies.py", "parity.py")]
    _miss = [str(p) for p in _need if not p.exists()]
    print("missing:" if _miss else "all inputs resolve", *_miss, sep="\n  ")
    sys.exit(1 if _miss else 0)

import argparse
import gzip
import hashlib
import json
from collections import defaultdict

import numpy as np
import torch

import kernels as KN
from common import OUT, THREADS, build_extractor, load_rows, opp_true_team, unpack
from proxies import Beliefs, fixed_size
from agents.observation.constants import TEAM_SIZE

BAND = 0.01
BUDGETS = ("1:1", "k+6", "12", "18")
BELIEFS = ("prior", "memo_nb", "memo_exact")
FEATS = ("out", "ko", "in_mean", "in_max", "in_emax", "bulkP", "bulkS", "margin")
PROXIES = ("P1", "P2", "P2e", "P2a", "P3")
CHUNK = 256
N_BOOT = 400


def budget_H(name: str, k: torch.Tensor) -> torch.Tensor:
    return {"1:1": k, "k+6": k + 6, "12": torch.full_like(k, 12), "18": torch.full_like(k, 18)}[name]


def averaged_entity(op, fx, tabs, att, dfn, p32, K, attacker=True, null_ko=True):
    """Features of ONE averaged entity per decision for the species marginal p32 [B,S]. attacker=False is R0: today
    every hidden attacker is gated off, so its threat is exactly 0."""
    h, ko = KN.outgoing(op, att, KN.averaged_defender(op, p32))          # [B,4,1]
    if attacker:
        a = KN.averaged_attacker(op, fx, tabs, p32, K)
        w, _p, _c = KN.incoming(op, dfn, a)                               # [B,1]
    else:
        w = torch.zeros(h.shape[0], 1)
    dd = KN.averaged_defender(op, p32)
    out = h[..., 0].double()
    best = out.max(-1).values
    return dict(out=out, ko=torch.zeros_like(out) if null_ko else ko[..., 0].double(), inw=w[:, 0].double(),
                bulkP=(dd["maxhp"] * dd["def_"])[:, 0].double(), bulkS=(dd["maxhp"] * dd["spd"])[:, 0].double(),
                margin=best * dfn["hp_frac"].double() - w[:, 0].double())


def summarise(w, wavg, k, F, A):
    """w [B,S] weights on concrete species, wavg [B] weight on the averaged entity A (or None) -> summaries."""
    kd = k.double()
    def mean(Fs, Fa, axis_extra=False):
        if axis_extra:
            s = torch.einsum("bs,bms->bm", w, Fs)
            if A is not None:
                s = s + wavg[:, None] * Fa
            return s / kd[:, None]
        s = (w * Fs).sum(-1)
        if A is not None:
            s = s + wavg * Fa
        return s / kd
    inmax = (w.clamp(max=1.0) * F["inw"]).max(-1).values
    pres, vals = w.clamp(max=1.0), F["inw"]
    if A is not None:
        a_max = A.get("inw_max", A["inw"])          # OP_H: OTHER's worst member for the max-type reads
        inmax = torch.maximum(inmax, wavg.clamp(max=1.0) * a_max)
        pres = torch.cat([pres, wavg.clamp(max=1.0)[:, None]], 1); vals = torch.cat([vals, a_max[:, None]], 1)
    # M2 option A for comparison: the expected max under independent presence (noisy-OR), sorted by value
    o = torch.sort(vals, dim=-1, descending=True, stable=True).indices
    pv, vv = pres.gather(1, o), vals.gather(1, o)
    surv = torch.cumprod(torch.cat([torch.ones_like(pv[:, :1]), 1.0 - pv[:, :-1]], 1), 1)
    inemax = (vv * pv * surv).sum(-1)
    return dict(out=mean(F["out"], A["out"] if A else None, True), ko=mean(F["ko"], A["ko"] if A else None, True),
                in_mean=mean(F["inw"], A["inw"] if A else None), in_max=inmax, in_emax=inemax,
                bulkP=mean(F["bulkP"], A["bulkP"] if A else None), bulkS=mean(F["bulkS"], A["bulkS"] if A else None),
                margin=mean(F["margin"], A["margin"] if A else None))


def decide(sm, hp, U):
    """-> {proxy: (decision [B] (long), distance to boundary [B])}."""
    out = sm["out"].masked_fill(~U, -1e9)
    top2 = out.topk(2, -1)
    p1 = (top2.indices[:, 0], top2.values[:, 0] - top2.values[:, 1])
    return {"P1": p1, "P2": ((sm["in_max"] < hp).long(), (sm["in_max"] - hp).abs()),
            "P2e": ((sm["in_mean"] < hp).long(), (sm["in_mean"] - hp).abs()),
            "P2a": ((sm["in_emax"] < hp).long(), (sm["in_emax"] - hp).abs()),
            "P3": ((sm["margin"] > 0).long(), sm["margin"].abs())}


def errors(sm, st, U):
    nU = U.sum(-1)
    nan = torch.full(nU.shape, float("nan"), dtype=torch.float64)
    e = {}
    e["out"] = torch.where(nU > 0, ((sm["out"] - st["out"]).abs() * U).sum(-1) / nU.clamp(min=1) * 100, nan)
    e["ko"] = torch.where(nU > 0, ((sm["ko"] - st["ko"]).abs() * U).sum(-1) / nU.clamp(min=1) * 100, nan)
    e["in_mean"] = (sm["in_mean"] - st["in_mean"]).abs() * 100
    e["in_max"] = (sm["in_max"] - st["in_max"]).abs() * 100
    e["in_emax"] = (sm["in_emax"] - st["in_max"]).abs() * 100      # the truth's max (presence 1) is the target
    e["bulkP"] = (sm["bulkP"] - st["bulkP"]).abs() / st["bulkP"] * 100
    e["bulkS"] = (sm["bulkS"] - st["bulkS"]).abs() / st["bulkS"] * 100
    e["margin"] = (sm["margin"] - st["margin"]).abs() * 100
    return e


def run(limit=None):
    torch.set_num_threads(THREADS)
    torch.manual_seed(0)
    bank, rows, masks, gate = load_rows()
    truth_team = opp_true_team(bank)
    fx, op = build_extractor()
    K = fx.consequence_topk
    bel = Beliefs()
    fits = bel.check_fits()
    dec = bank.decisions
    battle_ids = sorted({d["battle"] for d in dec})
    bidx = {b: i for i, b in enumerate(battle_ids)}
    N = len(dec) if limit is None else limit
    rec = defaultdict(list)          # (belief, rep) -> list of per-decision arrays
    meta = defaultdict(list)
    counts = dict(total=N, no_hidden=0, gated=0, used=0)
    with torch.no_grad():
        dt = KN.defender_tables(op)
        tabs = KN.attacker_tables(op, fx, K)
        S = dt["maxhp"].shape[0]
        for c0 in range(0, N, CHUNK):
            sl = slice(c0, min(N, c0 + CHUNK))
            ctx = unpack(fx, rows[sl], masks[sl])
            hid = ctx.opp_believed_mask
            att = KN.our_attacker(op, ctx)
            nh = hid.sum(-1)
            counts["no_hidden"] += int((nh == 0).sum())
            keep = (nh > 0) & (att["gate"] > 0)
            counts["gated"] += int(((nh > 0) & (att["gate"] == 0)).sum())
            if not keep.any():
                continue
            from common import select_ctx
            ki = torch.nonzero(keep).flatten()
            ctx = select_ctx(ctx, ki)
            hid = ctx.opp_believed_mask
            ids = ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE]
            att = KN.our_attacker(op, ctx)
            dfn = KN.our_defender(op, ctx)
            B = ctx.batch_size
            k = hid.sum(-1)
            drows = [dec[c0 + int(i)] for i in ki]
            # the truth
            y = torch.zeros(B, S, dtype=torch.float64)
            for b, d in enumerate(drows):
                true = truth_team[d["battle"]][d["side"]]
                revealed = [int(x) for x in ids[b][~hid[b]].tolist()]
                if not set(revealed) <= set(true):
                    raise RuntimeError(f"{d['id']}: revealed {revealed} not in the true team {true}")
                th = sorted(set(true) - set(revealed))
                if len(th) != int(k[b]):
                    raise RuntimeError(f"{d['id']}: {len(th)} true hidden vs {int(k[b])} hidden slots")
                y[b, th] = 1.0
            # per-species features (exact physics, every species)
            all_h, all_ko = KN.outgoing(op, att, dt)                         # [B,4,S]
            inw, _inp, _ = KN.incoming(op, dfn, tabs)                        # [B,S]
            out = all_h.double(); best = out.max(1).values
            F = dict(out=out, ko=all_ko.double(), inw=inw.double(),
                     bulkP=(dt["maxhp"] * dt["def_"]).double()[None].expand(B, S),
                     bulkS=(dt["maxhp"] * dt["spd"]).double()[None].expand(B, S),
                     margin=best * dfn["hp_frac"].double()[:, None] - inw.double())
            U = (att["usable"] > 0) & ((att["bp"] > 0) | (att["fixed"] > 0))
            hp = dfn["hp_frac"].double()
            st = summarise(y, None, k, F, None)
            dT = decide(st, hp, U)
            lb, valid = bel.logbeliefs(ids, hid, pad_to=S)
            t0 = fx.t0_species_prior(ids, hid)
            for b, d in enumerate(drows):
                meta["battle"].append(bidx[d["battle"]]); meta["r"].append(6 - int(k[b]))
                meta["id"].append(d["id"]); meta["nU"].append(int(U[b].sum()))
                meta["hp"].append(float(hp[b])); meta["opp_class"].append(d["opp_class"])
            for name in BELIEFS:
                a = lb[name]
                pi = fixed_size(a, k, valid)
                p0 = t0 if name == "prior" else a.exp().float()
                order = torch.sort(pi, dim=-1, descending=True, stable=True).indices
                rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(S).expand(B, S))
                reps = {"R0": (torch.zeros_like(pi), k.double(), averaged_entity(op, fx, tabs, att, dfn, p0, K,
                                                                                attacker=False)),
                        "Rinf": (pi, None, None)}
                for bn in BUDGETS:
                    H = budget_H(bn, k)
                    named = rank < H[:, None]
                    wn = pi * named
                    mO = (k.double() - wn.sum(-1)).clamp(min=0.0)
                    q = (pi * ~named); q = (q / q.sum(-1, keepdim=True).clamp_min(1e-300)).float()
                    reps[f"R1_{bn}"] = (wn, mO, averaged_entity(op, fx, tabs, att, dfn, q, K))
                    reps[f"R2_{bn}"] = (wn, None, None)
                    reps[f"OO_{bn}"] = (wn + y * ~named, None, None)
                    # OP_H: OTHER keeps its mass but its PHYSICS is the truth's — the mean (max, for the
                    # max-type reads) of the TRUE hidden mons the list missed; no missed mon => OTHER weight 0.
                    miss = y * ~named
                    nm = miss.sum(-1)
                    def mmean(Fs, nm=nm, miss=miss):
                        if Fs.dim() == 3:
                            return torch.einsum("bs,bms->bm", miss, Fs) / nm.clamp(min=1)[:, None]
                        return (miss * Fs).sum(-1) / nm.clamp(min=1)
                    Aop = dict(out=mmean(F["out"]), ko=mmean(F["ko"]), inw=mmean(F["inw"]),
                               inw_max=(miss * F["inw"]).max(-1).values, bulkP=mmean(F["bulkP"]),
                               bulkS=mmean(F["bulkS"]), margin=mmean(F["margin"]))
                    reps[f"OP_{bn}"] = (wn, torch.where(nm > 0, mO, torch.zeros_like(mO)), Aop)
                    meta[f"{name}_other_mass_{bn}"].extend(mO.tolist())
                    meta[f"{name}_recall_{bn}"].extend(((y * named).sum(-1) / k.double()).tolist())
                    # rule-8 read at the selection boundary (reported, not excluded: one stable ordering)
                    srt = torch.gather(pi, 1, order)
                    hb = H.clamp(max=S - 1)
                    gap = (srt.gather(1, (hb - 1)[:, None]) - srt.gather(1, hb[:, None]))[:, 0].abs()
                    meta[f"{name}_neartie_{bn}"].extend((gap < 1e-6).long().tolist())
                for rn, (w, wavg, A) in reps.items():
                    sm = summarise(w, wavg, k, F, A)
                    e = errors(sm, st, U)
                    dR = decide(sm, hp, U)
                    arr = [e[f] for f in FEATS]
                    for p in PROXIES:
                        incl = (dT[p][1] >= BAND) & (dR[p][1] >= BAND)
                        if p == "P1":
                            incl = incl & (U.sum(-1) >= 2)
                        flip = (dT[p][0] != dR[p][0]).double()
                        arr.append(torch.where(incl, flip, torch.full_like(flip, -1.0)))
                    rec[(name, rn)].append(torch.stack([x.double() for x in arr], 1).numpy())
            counts["used"] += B
            print(f"  {min(N, c0 + CHUNK)}/{N}  used {counts['used']}", flush=True)
    data = {key: np.concatenate(v, 0) for key, v in rec.items()}
    meta = {k_: np.asarray(v) for k_, v in meta.items()}
    return data, meta, counts, fits, gate, len(battle_ids)


# ------------------------------------------------------------------------------------------------ aggregation
def boot_weights(battle, n_battles, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n_battles, np.full(n_battles, 1.0 / n_battles), size=n_boot).astype(np.float64)
    return W[:, battle]                                            # [R, N]


def agg_cell(X, sel, Wd):
    """X [N, nF+nP]; sel bool [N] -> point + CI for each feature mean and proxy flip rate, + excluded share."""
    nF = len(FEATS)
    out = {}
    Xs = X[sel]; Ws = Wd[:, sel]
    for j, f in enumerate(FEATS):
        v = Xs[:, j]
        ok = np.isfinite(v)
        pt = float(v[ok].mean()) if ok.any() else None
        bs = (Ws[:, ok] @ v[ok]) / Ws[:, ok].sum(1)
        out[f] = dict(mean=pt, ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                      median=float(np.median(v[ok])), p90=float(np.quantile(v[ok], .9)))
    for j, p in enumerate(PROXIES):
        v = Xs[:, nF + j]
        inc = v >= 0
        n_inc = int(inc.sum())
        rate = float(v[inc].mean()) if n_inc else None
        bs = (Ws[:, inc] @ v[inc]) / np.maximum(Ws[:, inc].sum(1), 1e-12)
        out[p] = dict(flip=rate, ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], n=n_inc,
                      excluded=float(1 - n_inc / max(1, len(v))))
    return out, Xs, Ws


def ratio_ci(num_x, den_x, Ws):
    """1 - mean(num)/mean(den) with a paired battle bootstrap (num, den per decision, finite)."""
    ok = np.isfinite(num_x) & np.isfinite(den_x)
    num_x, den_x, Ws = num_x[ok], den_x[ok], Ws[:, ok]
    if len(num_x) == 0 or den_x.mean() <= 0:
        return [None, None, None]
    pt = 1.0 - num_x.mean() / den_x.mean()
    bn = (Ws @ num_x) / Ws.sum(1); bd = (Ws @ den_x) / Ws.sum(1)
    b = 1.0 - bn / np.where(bd > 0, bd, np.nan)
    return [float(pt), float(np.nanquantile(b, .025)), float(np.nanquantile(b, .975))]


def aggregate(data, meta, n_battles):
    battle = meta["battle"]; r = meta["r"]
    Wd = boot_weights(battle, n_battles)
    res = {}
    for (name, rn), X in sorted(data.items()):
        cells = {}
        for rsel in ["all", 1, 2, 3, 4, 5]:
            sel = np.ones(len(r), bool) if rsel == "all" else (r == rsel)
            if sel.sum() == 0:
                continue
            cells[str(rsel)], _, _ = agg_cell(X, sel, Wd)
            cells[str(rsel)]["n"] = int(sel.sum())
        res.setdefault(name, {})[rn] = cells
    # OTHER share, Jensen share, recovery (pooled + per r), paired bootstrap
    derived = {}
    for name in BELIEFS:
        dn = {}
        for rsel in ["all", 1, 2, 3, 4, 5]:
            sel = np.ones(len(r), bool) if rsel == "all" else (r == rsel)
            if sel.sum() == 0:
                continue
            Ws = Wd[:, sel]
            X0 = data[(name, "R0")][sel]; Xi = data[(name, "Rinf")][sel]
            cell = {}
            for j, f in enumerate(FEATS):
                d = {}
                for bn in BUDGETS:
                    X1 = data[(name, f"R1_{bn}")][sel][:, j]; XO = data[(name, f"OO_{bn}")][sel][:, j]
                    X2 = data[(name, f"R2_{bn}")][sel][:, j]
                    XP = data[(name, f"OP_{bn}")][sel][:, j]
                    d[bn] = dict(other_share=ratio_ci(XP, X1, Ws), oracle_other_share=ratio_ci(XO, X1, Ws),
                                 jensen_share=ratio_ci(Xi[:, j], X1, Ws),
                                 recovered_of_R0=ratio_ci(X1, X0[:, j], Ws),
                                 recovered_of_R0_d=ratio_ci(X2, X0[:, j], Ws))
                    ok = np.isfinite(X1) & np.isfinite(X0[:, j])
                    gap = X0[ok, j].mean() - Xi[ok, j].mean()
                    d[bn]["recovered_of_gap_R0_to_Rinf"] = (float((X0[ok, j].mean() - X1[ok].mean()) / gap)
                                                            if gap > 0 else None)
                d["Rinf_recovered_of_R0"] = ratio_ci(Xi[:, j], X0[:, j], Ws)
                cell[f] = d
            dn[str(rsel)] = cell
        derived[name] = dn
    # flip rates on the COMMON included set (a decision counts only if no representation excluded it), so every
    # representation is compared on the same decisions; point + battle-clustered CI, pooled
    nF = len(FEATS)
    common = {}
    for name in BELIEFS:
        reps = [rn for (b_, rn) in data if b_ == name]
        cm = {}
        for j, p in enumerate(PROXIES):
            inc = np.all(np.stack([data[(name, rn)][:, nF + j] >= 0 for rn in reps]), 0)
            cp = {"n": int(inc.sum()), "share_of_decisions": float(inc.mean())}
            Ws = Wd[:, inc]
            base = data[(name, "R1_1:1")][inc, nF + j]
            for rn in sorted(reps):
                v = data[(name, rn)][inc, nF + j]
                bs = (Ws @ v) / Ws.sum(1)
                cp[rn] = [float(v.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
                dv = v - base                                   # paired: rep minus (c) at 1:1
                bd = (Ws @ dv) / Ws.sum(1)
                cp["delta_vs_R1_1:1"] = cp.get("delta_vs_R1_1:1", {})
                cp["delta_vs_R1_1:1"][rn] = [float(dv.mean()), float(np.quantile(bd, .025)),
                                             float(np.quantile(bd, .975))]
            cm[p] = cp
        # the same paired deltas on every feature's mean |err| (decisions where both are finite)
        fd = {}
        for j, f in enumerate(FEATS):
            base = data[(name, "R1_1:1")][:, j]
            fd[f] = {}
            for rn in sorted(reps):
                v = data[(name, rn)][:, j]
                ok = np.isfinite(v) & np.isfinite(base)
                dv = v[ok] - base[ok]
                bd = (Wd[:, ok] @ dv) / Wd[:, ok].sum(1)
                fd[f][rn] = [float(dv.mean()), float(np.quantile(bd, .025)), float(np.quantile(bd, .975))]
        cm["feature_delta_vs_R1_1:1"] = fd
        common[name] = cm
    # belief-level reads: OTHER share of hidden mass and recall by budget and r
    mass = {}
    for name in BELIEFS:
        m = {}
        for bn in BUDGETS:
            om = meta[f"{name}_other_mass_{bn}"]; rc = meta[f"{name}_recall_{bn}"]; nt = meta[f"{name}_neartie_{bn}"]
            m[bn] = {str(rr): dict(other_share=float((om[r == rr] / (6 - rr)).mean()), recall=float(rc[r == rr].mean()),
                                   near_ties=int(nt[r == rr].sum()), n=int((r == rr).sum()))
                     for rr in range(1, 6) if (r == rr).any()}
            m[bn]["frac_other_mass_gt1"] = float((om > 1.0).mean())
        mass[name] = m
    return res, derived, mass, common


def rounded(o, nd=6):
    if isinstance(o, float):
        return float(f"{o:.{nd}g}") if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: rounded(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rounded(v, nd) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    data, meta, counts, fits, gate, n_battles = run(a.limit)
    res, derived, mass, common = aggregate(data, meta, n_battles)
    r = meta["r"]
    summary = dict(
        counts=counts, n_battles=n_battles, r_hist={str(x): int((r == x).sum()) for x in range(1, 6)},
        opp_class_hist={c: int((meta["opp_class"] == c).sum()) for c in sorted(set(meta["opp_class"].tolist()))},
        belief_fits=fits, rows_gate={k_: gate[k_] for k_ in ("recorded_rows_checked", "byte_equal", "obs_as_recorded")},
        band=BAND, budgets=BUDGETS, feats=FEATS, proxies=PROXIES, n_boot=N_BOOT,
        cells=res, derived=derived, belief_mass=mass, flips_common_set=common,
        p1_structural_exclusion=float((meta["nU"] < 2).mean()))
    tag = a.tag
    js = json.dumps(rounded(summary), indent=1, sort_keys=True) + "\n"
    (OUT / f"tier0{tag}.json").write_text(js)
    # per-decision rows: a decision index + one row per (decision, belief, representation); gzip with mtime 0
    dl = ["idx,id,battle_idx,r,opp_class,hp_frac,n_usable_damaging"]
    for i in range(len(meta["id"])):
        dl.append(f"{i},{meta['id'][i]},{meta['battle'][i]},{meta['r'][i]},{meta['opp_class'][i]},"
                  f"{meta['hp'][i]:.4f},{meta['nU'][i]}")
    lines = [",".join(["idx", "belief", "rep"] + list(FEATS) + list(PROXIES))]
    for (name, rn), X in sorted(data.items()):
        if name != "prior":          # the committed rows are the Smogon-prior belief's (the memorising proxies are
            continue                 # aggregated in tier0.json and regenerate byte-equal from this script)
        for i in range(len(X)):
            vals = [str(i), name, rn] + [f"{x:.2f}" if np.isfinite(x) else "" for x in X[i, :len(FEATS)]]
            vals += [str(int(x)) for x in X[i, len(FEATS):]]
            lines.append(",".join(vals))
    raw = ("\n".join(lines) + "\n").encode()
    for fn, payload in ((f"decisions{tag}.csv.gz", ("\n".join(dl) + "\n").encode()), (f"rows{tag}.csv.gz", raw)):
        with open(OUT / fn, "wb") as f:
            with gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as g:
                g.write(payload)
    print("json sha", hashlib.sha256(js.encode()).hexdigest()[:16], "rows sha", hashlib.sha256(raw).hexdigest()[:16])


if __name__ == "__main__":
    main()
