"""Analyse the move-leg dumps: the meter's units, raw vs fixed-mass ("closed") scores, mass, and the split of the
gap by whether the pool's sets of that species carry the true move. numpy only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

D = Path.home() / "gen3ai_archive/move_leg_diag_2026-11-10"
HP, TYPED = 237, np.arange(355, 371)
PANELS = ("home72", "heldout72_hpfix", "forum72", "p72")
DRAWS, SEED = 10_000, 20261109


def lsig(x):
    return -np.logaddexp(0.0, -x)


def sig(x):
    return np.exp(lsig(x))


def tau(x, cand, k, iters=64):
    """τ [U] with Σ_cand σ(x + τ) = k (bisection on the provable bracket), float64."""
    n = cand.sum(-1)
    ks = np.minimum(np.maximum(k, 1), np.maximum(n, 2) - 1).astype(np.float64)
    a_max = np.where(cand, x, -np.inf).max(-1)
    a_min = np.where(cand, x, np.inf).min(-1)
    a_max = np.where(n > 0, a_max, 0.0); a_min = np.where(n > 0, a_min, 0.0)
    base = np.log(ks) - np.log(np.maximum(n, 2) - ks)
    lo, hi = base - a_max, base - a_min
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        tot = np.where(cand, sig(x + mid[:, None]), 0.0).sum(-1)
        above = tot > ks
        hi = np.where(above, mid, hi); lo = np.where(above, lo, mid)
    return 0.5 * (lo + hi)


def boot_index(n):
    return np.random.default_rng(SEED).integers(0, n, size=(DRAWS, n))


def team_mean(val, team, n_teams, w=None):
    """Per-team mean of per-unit ``val`` (NaN = no unit)."""
    w = np.ones_like(val) if w is None else w
    s = np.bincount(team, weights=val * w, minlength=n_teams)
    c = np.bincount(team, weights=w, minlength=n_teams)
    return np.where(c > 0, s / np.maximum(c, 1e-300), np.nan)


def ci(tm, idx):
    m = float(np.nanmean(tm))
    draws = np.nanmean(tm[idx], axis=1)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return m, float(lo), float(hi)


def fmt(c):
    return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"


def load(label, panel):
    z = np.load(D / "dump" / f"{label}__{panel}.npz")
    t = np.load(D / "dump" / f"{label}__tables.npz")
    return {k: z[k] for k in z.files}, {k: t[k] for k in t.files}


def units(z, t):
    U, Mv = z["post"].shape
    sp = z["species"]
    ar = np.arange(U)
    shown = np.zeros((U, Mv), bool)
    ids = np.where(z["shown_flag"] & (z["shown"] > 0), z["shown"], 0)
    for j in range(4):
        shown[ar, np.clip(ids[:, j], 0, Mv - 1)] |= ids[:, j] > 0
    shown[:, 0] = False
    rev_model = np.zeros((U, Mv), bool)                     # the model's rule: id > 0
    for j in range(4):
        rev_model[ar, np.clip(z["shown"][:, j], 0, Mv - 1)] |= z["shown"][:, j] > 0
    rev_model[:, 0] = False
    truth = np.zeros((U, Mv), bool)
    for j in range(4):
        m = z["truth"][:, j]
        truth[ar, np.clip(m, 0, Mv - 1)] |= m >= 0
    hp = np.zeros(Mv, bool); hp[HP] = True; hp[TYPED] = True
    truth_all = truth.copy()
    truth = truth & ~hp[None] & ~shown
    cand = t["meter_legal"][np.clip(sp, 0, 399)] & ~hp[None] & ~shown
    ok = truth.any(-1) & ~(truth & ~cand).any(-1)
    typed = np.zeros(Mv, bool); typed[TYPED] = True
    hp_rev = rev_model[:, HP]
    cand_model = (t["move_legal"][np.clip(sp, 0, 399)] & t["move_valid"][None] & ~rev_model
                  & (np.arange(Mv) != HP)[None] & ~(typed[None] & hp_rev[:, None]))
    r = rev_model.sum(-1)
    return dict(ok=ok, truth=truth, truth_all=truth_all, shown=shown, cand=cand, cand_model=cand_model, r=r, rev_model=rev_model,
                k_m=np.maximum(4 - r, 0), illegal=truth.any(-1) & (truth & ~cand).any(-1),
                rev_mismatch=int((shown != rev_model).any(-1).sum()))


def main():
    labels = sys.argv[1].split(",")
    pool = np.load(D / "pool_table.npz")
    n_sets, count = pool["n_sets"], pool["count"]
    pool_sets = {tuple(int(v) for v in row) for row in pool["sets"]}
    out = {}
    for label in labels:
        for panel in PANELS:
            z, t = load(label, panel)
            u = units(z, t)
            ok = u["ok"]
            nt = int(z["n_teams"])
            idx = boot_index(nt)
            team = z["team"][ok]
            truth, cand, cm = u["truth"][ok], u["cand"][ok], u["cand_model"][ok]
            k_m = u["k_m"][ok]
            nu = truth.sum(-1)
            res = {"units": int(ok.sum()), "illegal": int(u["illegal"].sum()), "rev_mismatch": u["rev_mismatch"]}
            X = {v: z[v][ok].astype(np.float64) for v in ("post", "prior")}
            live = (k_m > 0) & (k_m < cm.sum(-1))
            sc = {}
            for v in ("post", "prior"):
                x = X[v]
                tv = tau(x, cm, k_m)
                xc = x + tv[:, None]
                sc[v] = dict(
                    raw=(np.where(truth, lsig(x), 0).sum(-1) / nu),
                    closed=(np.where(truth, lsig(xc), 0).sum(-1) / nu),
                    mass=np.where(cm, sig(x), 0).sum(-1),
                    mass_meter=np.where(cand, sig(x), 0).sum(-1),
                    ll_raw=np.where(cand, np.where(truth, lsig(x), lsig(-x)), 0).sum(-1),
                    ll_closed=np.where(cand, np.where(truth, lsig(xc), lsig(-xc)), 0).sum(-1),
                    tau=tv)
            res["live_share"] = float(live.mean())
            for key in ("raw", "closed", "ll_raw", "ll_closed"):
                tp, tq = team_mean(sc["post"][key], team, nt), team_mean(sc["prior"][key], team, nt)
                res[key] = {"post": ci(tp, idx), "prior": ci(tq, idx), "diff": ci(tp - tq, idx)}
            res["mass"] = {"post": ci(team_mean(sc["post"]["mass"], team, nt), idx),
                           "prior": ci(team_mean(sc["prior"]["mass"], team, nt), idx),
                           "target_4_minus_r": ci(team_mean(k_m.astype(float), team, nt), idx),
                           "truth_nonhp_unshown": ci(team_mean(nu.astype(float), team, nt), idx),
                           "post_meter": ci(team_mean(sc["post"]["mass_meter"], team, nt), idx),
                           "prior_meter": ci(team_mean(sc["prior"]["mass_meter"], team, nt), idx)}
            res["tau"] = {"post": ci(team_mean(sc["post"]["tau"], team, nt), idx),
                          "prior": ci(team_mean(sc["prior"]["tau"], team, nt), idx)}
            # by number of revealed moves
            res["by_r"] = {}
            for r in range(0, 4):
                m = u["r"][ok] == r
                if m.sum() == 0:
                    continue
                e = {"units": int(m.sum())}
                for key in ("raw", "closed"):
                    d = sc["post"][key][m] - sc["prior"][key][m]
                    e[key] = ci(team_mean(d, team[m], nt), idx)
                e["mass_post"] = float(sc["post"]["mass"][m].mean()); e["mass_prior"] = float(sc["prior"]["mass"][m].mean())
                res["by_r"][r] = e
            # per true move: the pool status of (species, move)
            uu, mm = np.nonzero(truth)
            sp = z["species"][ok][uu]
            ns, ct = n_sets[sp], count[sp, mm]
            cls = np.where(ns == 0, 0, np.where(ct > 0, 1, 2))      # 0 species off pool, 1 attested, 2 unattested
            d_raw = lsig(X["post"][uu, mm]) - lsig(X["prior"][uu, mm])
            xc_post = X["post"] + sc["post"]["tau"][:, None]; xc_prior = X["prior"] + sc["prior"]["tau"][:, None]
            d_closed = lsig(xc_post[uu, mm]) - lsig(xc_prior[uu, mm])
            delta = X["post"][uu, mm] - X["prior"][uu, mm]
            res["by_class"] = {}
            w_unit = 1.0 / nu[uu]
            for c, name in enumerate(("species_off_pool", "move_attested", "move_unattested")):
                m = cls == c
                e = {"share_of_true_moves": float(m.mean()), "n": int(m.sum())}
                if m.sum():
                    e["raw"] = ci(team_mean(d_raw[m], team[uu][m], nt), idx)
                    e["closed"] = ci(team_mean(d_closed[m], team[uu][m], nt), idx)
                    e["delta_logit"] = float(delta[m].mean())
                    e["prior_p"] = float(sig(X["prior"][uu, mm][m]).mean())
                    e["post_p"] = float(sig(X["post"][uu, mm][m]).mean())
                    # contribution to the unit-level raw gap (sum over moves of d/|U| per unit, averaged over units)
                    e["contrib_raw"] = float((d_raw[m] * w_unit[m]).sum() / len(nu))
                    e["contrib_closed"] = float((d_closed[m] * w_unit[m]).sum() / len(nu))
                res["by_class"][name] = e
            # the full true set attested in the pool
            ta = u["truth_all"][ok]
            spu = z["species"][ok]
            att = np.zeros(len(spu), bool)
            for i in range(len(spu)):
                ms = sorted(int(x) for x in np.flatnonzero(ta[i]))
                att[i] = (int(spu[i]), *ms, *([-1] * (4 - len(ms)))) in pool_sets
            res["by_set"] = {}
            for name, m in (("set_in_pool", att), ("set_not_in_pool", ~att)):
                e = {"share_of_units": float(m.mean())}
                if m.sum():
                    for key in ("raw", "closed"):
                        e[key] = ci(team_mean(sc["post"][key][m] - sc["prior"][key][m], team[m], nt), idx)
                res["by_set"][name] = e
            out[f"{label}/{panel}"] = res
            print(f"\n== {label} / {panel}: {res['units']} units, truth-illegal {res['illegal']}, reveal-rule mismatch {res['rev_mismatch']}, live {res['live_share']:.3f}")
            for key in ("raw", "closed", "ll_raw", "ll_closed"):
                r_ = res[key]
                print(f"  {key:10s} post {r_['post'][0]:+.3f} prior {r_['prior'][0]:+.3f}  post-prior {fmt(r_['diff'])}")
            ms = res["mass"]
            print(f"  mass over the model's candidates: post {ms['post'][0]:.3f} prior {ms['prior'][0]:.3f} target 4-r {ms['target_4_minus_r'][0]:.3f} | non-HP unshown truths {ms['truth_nonhp_unshown'][0]:.3f}; meter cands post {ms['post_meter'][0]:.3f} prior {ms['prior_meter'][0]:.3f}")
            print(f"  tau: post {res['tau']['post'][0]:+.3f} prior {res['tau']['prior'][0]:+.3f}")
            for r, e in res["by_r"].items():
                print(f"  r={r}: units {e['units']:6d} raw {fmt(e['raw'])} closed {fmt(e['closed'])} mass post {e['mass_post']:.2f} prior {e['mass_prior']:.2f} (target {4 - r})")
            for name, e in res["by_class"].items():
                if e["n"]:
                    print(f"  {name:18s} share {e['share_of_true_moves']:.3f} raw {fmt(e['raw'])} closed {fmt(e['closed'])} "
                          f"contrib raw {e['contrib_raw']:+.3f} closed {e['contrib_closed']:+.3f} | delta {e['delta_logit']:+.2f} p prior {e['prior_p']:.3f} post {e['post_p']:.3f}")
            for name, e in res["by_set"].items():
                if "raw" in e:
                    print(f"  {name:16s} share {e['share_of_units']:.3f} raw {fmt(e['raw'])} closed {fmt(e['closed'])}")
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
