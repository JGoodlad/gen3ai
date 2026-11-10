"""Second pass: what the move residual IS. Per panel: counterfactual predictors scored against the prior (the
residual shrunk, clipped, one-sided, its static (species, move) part; the pool's own marginal and set-completer),
the share of the residual that is a static offset, and how far two checkpoints' residuals agree on the same rows.
numpy only.   analyze2.py <label[,label...]> <out.json>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import D, PANELS, boot_index, ci, fmt, load, lsig, sig, team_mean, units  # noqa: E402

HP, TYPED_LO, TYPED_HI = 237, 355, 370


def norm_hp(m):
    return HP if TYPED_LO <= m <= TYPED_HI else m


def pool_predictors(sp, rev, pp, pool, kappa=1.0):
    n_sets, count = pool["n_sets"], pool["count"]
    marg = (count[sp] + kappa * pp) / (n_sets[sp][:, None] + kappa)
    by_sp = {}
    for row, c in zip(pool["sets"], pool["set_count"]):
        ms = [int(m) for m in row[1:] if m >= 0]
        by_sp.setdefault(int(row[0]), []).append((frozenset(norm_hp(m) for m in ms), ms, int(c)))
    cond = np.empty_like(pp)
    n_cons = np.zeros(len(sp))
    cache = {}
    for i in range(len(sp)):
        R = frozenset(int(m) for m in np.flatnonzero(rev[i]))
        key = (int(sp[i]), R)
        if key not in cache:
            cnt = np.zeros(pp.shape[1]); tot = 0
            for msn, ms, c in by_sp.get(key[0], ()):
                if R <= msn:
                    cnt[ms] += c; tot += c
            cache[key] = (cnt, tot)
        cnt, tot = cache[key]
        cond[i] = (cnt + kappa * pp[i]) / (tot + kappa)
        n_cons[i] = tot
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(p, 1e-6, 1 - 1e-6))
    return lg(marg), lg(cond), n_cons


def score(x, truth, cand, nu):
    return (np.where(truth, lsig(x), 0).sum(-1) / nu,
            np.where(cand, np.where(truth, lsig(x), lsig(-x)), 0).sum(-1))


def cell_means(delta, cand, sp, Mv=400):
    uu, mm = np.nonzero(cand)
    key = sp[uu] * Mv + mm
    s = np.bincount(key, weights=delta[uu, mm], minlength=400 * Mv)
    c = np.bincount(key, minlength=400 * Mv)
    return (s / np.maximum(c, 1)).reshape(400, Mv), c.reshape(400, Mv)


def main():
    labels = sys.argv[1].split(",")
    pool = dict(np.load(D / "pool_table.npz"))
    out = {}
    static_home = {}
    deltas = {}
    for label in labels:
        for panel in PANELS:
            z, t = load(label, panel)
            u = units(z, t)
            ok = u["ok"]
            nt = int(z["n_teams"]); idx = boot_index(nt)
            team = z["team"][ok]; sp = z["species"][ok].astype(np.int64)
            truth, cand = u["truth"][ok], u["cand"][ok]
            nu = truth.sum(-1)
            post, prior = z["post"][ok].astype(np.float64), z["prior"][ok].astype(np.float64)
            delta = post - prior
            deltas[(label, panel)] = (delta.astype(np.float32), cand, sig(prior) >= 0.05)
            base = score(prior, truth, cand, nu)
            res = {}

            def rel(x, name):
                r_, l_ = score(x, truth, cand, nu)
                res[name] = {"raw": ci(team_mean(r_ - base[0], team, nt), idx),
                             "ll": ci(team_mean(l_ - base[1], team, nt), idx)}

            rel(post, "model")
            for a in (0.25, 0.5, 0.75):
                rel(prior + a * delta, f"shrink_{a}")
            for c in (1.0, 2.0):
                rel(prior + np.clip(delta, -c, c), f"clip_{c}")
            rel(prior + np.maximum(delta, 0), "raise_only")
            rel(prior + np.minimum(delta, 0), "lower_only")
            cm, cc = cell_means(delta, cand, sp)
            if panel == "home72":
                static_home[label] = cm
            rel(prior + cm[sp], "static_in_panel")
            rel(prior + static_home[label][sp], "static_from_home72")
            rel(prior + delta - cm[sp], "context_only_in_panel")
            marg, cond, n_cons = pool_predictors(sp, u["rev_model"][ok], sig(prior), pool)
            rel(marg, "pool_marginal")
            rel(cond, "pool_set_completer")
            # the share of the residual's mean square that is a static (species, move) offset
            uu, mm = np.nonzero(cand)
            d = delta[uu, mm]
            st = cm[sp[uu], mm]
            big = sig(prior[uu, mm]) >= 0.05
            res["delta"] = {
                "rms": float(np.sqrt((d ** 2).mean())), "rms_p05": float(np.sqrt((d[big] ** 2).mean())),
                "static_share": float((st ** 2).sum() / (d ** 2).sum()),
                "static_share_p05": float((st[big] ** 2).sum() / (d[big] ** 2).sum()),
                "mean_on_truth": float(delta[truth].mean()), "mean_off_truth": float(delta[cand & ~truth].mean()),
                "sd_on_truth": float(delta[truth].std()),
                "share_truth_lowered_1": float((delta[truth] < -1).mean()),
                "share_truth_lowered_3": float((delta[truth] < -3).mean()),
            }
            dc = (cond - prior)[uu, mm]; dm = (marg - prior)[uu, mm]
            res["corr"] = {"completer": float(np.corrcoef(d[big], dc[big])[0, 1]),
                           "marginal": float(np.corrcoef(d[big], dm[big])[0, 1]),
                           "units_with_consistent_pool_set": float((n_cons > 0).mean())}
            # split by whether the pool holds a set of this species consistent with the reveals
            for name, m in (("pool_consistent", n_cons > 0), ("pool_inconsistent", n_cons == 0)):
                if m.sum():
                    r_, l_ = score(post, truth, cand, nu)
                    res[name] = {"share": float(m.mean()),
                                 "raw": ci(team_mean((r_ - base[0])[m], team[m], nt), idx)}
            out[f"{label}/{panel}"] = res
            print(f"\n== {label} / {panel}")
            for name in ("model", "shrink_0.25", "shrink_0.5", "shrink_0.75", "clip_1.0", "clip_2.0", "raise_only",
                         "lower_only", "static_in_panel", "static_from_home72", "context_only_in_panel",
                         "pool_marginal", "pool_set_completer"):
                print(f"  {name:22s} raw {fmt(res[name]['raw'])}   ll {fmt(res[name]['ll'])}")
            dl = res["delta"]
            print(f"  delta rms {dl['rms']:.2f} (p>=.05: {dl['rms_p05']:.2f}) static share {dl['static_share']:.2f} "
                  f"(p>=.05: {dl['static_share_p05']:.2f}) | on truths mean {dl['mean_on_truth']:+.2f} sd {dl['sd_on_truth']:.2f} "
                  f"lowered>1 {dl['share_truth_lowered_1']:.3f} >3 {dl['share_truth_lowered_3']:.3f}; off truths mean {dl['mean_off_truth']:+.2f}")
            print(f"  corr(delta, completer-prior) {res['corr']['completer']:+.2f}  corr(delta, pool marginal-prior) "
                  f"{res['corr']['marginal']:+.2f}  units with a consistent pool set {res['corr']['units_with_consistent_pool_set']:.3f}")
            for name in ("pool_consistent", "pool_inconsistent"):
                if name in res:
                    print(f"  {name:18s} share {res[name]['share']:.3f} raw {fmt(res[name]['raw'])}")
    # cross-checkpoint agreement on the same rows
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            for panel in PANELS:
                da, cand, big = deltas[(a, panel)]; db = deltas[(b, panel)][0]
                m = cand & big
                x, y = da[m].astype(np.float64), db[m].astype(np.float64)
                e = {"corr": float(np.corrcoef(x, y)[0, 1]), "rms_a": float(np.sqrt((x ** 2).mean())),
                     "rms_b": float(np.sqrt((y ** 2).mean())), "rms_diff": float(np.sqrt(((x - y) ** 2).mean()))}
                out[f"agree/{a}/{b}/{panel}"] = e
                print(f"agree {a} vs {b} on {panel}: corr {e['corr']:+.3f} rms {e['rms_a']:.2f} / {e['rms_b']:.2f} rms of the difference {e['rms_diff']:.2f}")
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
