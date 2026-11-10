"""Third pass: the candidate levers as post-hoc transforms of the residual, scored against the prior on three
scores: raw (truths-only mean log sigma, the meter's moves_logp), ll (the full Bernoulli log-likelihood over the
meter's candidates, a proper score) and closed (truths-only on the count-normalised presence the consumers read).
Plus the calibration slope: the residual scale alpha that maximises ll per panel. numpy only.
analyze3.py <label[,label...]> <out.json>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import PANELS, boot_index, ci, fmt, load, lsig, sig, tau, team_mean, units  # noqa: E402


def mix(post, prior, lam):
    a, b = np.log1p(-lam), np.log(lam)
    lp = np.logaddexp(a + lsig(post), b + lsig(prior))
    ln = np.logaddexp(a + lsig(-post), b + lsig(-prior))
    return lp - ln


def main():
    labels = sys.argv[1].split(",")
    out = {}
    for label in labels:
        for panel in PANELS:
            z, t = load(label, panel)
            u = units(z, t)
            ok = u["ok"]
            nt = int(z["n_teams"]); idx = boot_index(nt)
            team = z["team"][ok]
            truth, cand, cm = u["truth"][ok], u["cand"][ok], u["cand_model"][ok]
            k_m = u["k_m"][ok]
            nu = truth.sum(-1)
            live = (k_m > 0) & (k_m < cm.sum(-1))
            post, prior = z["post"][ok].astype(np.float64), z["prior"][ok].astype(np.float64)
            delta = post - prior

            def sc(x):
                raw = np.where(truth, lsig(x), 0).sum(-1) / nu
                ll = np.where(cand, np.where(truth, lsig(x), lsig(-x)), 0).sum(-1)
                tt = tau(x, cm, k_m)
                closed = np.where(truth, lsig(x + tt[:, None]), 0).sum(-1) / nu
                return raw, ll, closed

            base = sc(prior)
            res = {}

            def rel(x, name):
                s = sc(x)
                res[name] = {"raw": ci(team_mean(s[0] - base[0], team, nt), idx),
                             "ll": ci(team_mean(s[1] - base[1], team, nt), idx),
                             "closed": ci(team_mean((s[2] - base[2])[live], team[live], nt), idx)}

            rel(post, "model")
            for c in (0.5, 1.0, 1.5, 2.0):
                rel(prior + c * np.tanh(delta / c), f"tanh_{c}")
            for lam in (0.25, 0.5):
                rel(mix(post, prior, lam), f"floor_{lam}")
            grid = np.arange(0.0, 1.5001, 0.125)
            lls = []
            for a in grid:
                x = prior + a * delta
                ll = np.where(cand, np.where(truth, lsig(x), lsig(-x)), 0).sum(-1)
                lls.append(float(np.nanmean(team_mean(ll - base[1], team, nt))))
            res["slope"] = {"alpha": [float(a) for a in grid], "ll": lls, "best_alpha": float(grid[int(np.argmax(lls))])}
            out[f"{label}/{panel}"] = res
            print(f"\n== {label} / {panel}   (calibration slope: ll is best at alpha = {res['slope']['best_alpha']:.3f}, "
                  f"ll there {max(lls):+.3f}; at 1.0 {lls[8]:+.3f})")
            for name in ("model", "tanh_0.5", "tanh_1.0", "tanh_1.5", "tanh_2.0", "floor_0.25", "floor_0.5"):
                print(f"  {name:12s} raw {fmt(res[name]['raw'])}  ll {fmt(res[name]['ll'])}  closed {fmt(res[name]['closed'])}")
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
