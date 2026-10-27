"""REVIEW F's independent checks of X60's redesign (design_human_replay_lever.md §6, e1512c36).

CPU, seeded, numpy + scipy, about a minute. Run from the repo root:
    python designs/research_state/measurements/human_replay_quality_2026-10-25/review_f_checks.py
It prints one JSON document (REVIEW_F_x60.md quotes it).

Written from §6.3's text, NOT from x60_redesign_2026-10-26/power.py, so that a shared bug would not reproduce itself.
- ``primary``: Δ_F's registered rule (look 1 = 3 v 3, futility only; look 2 = 5 v 5; binomial + 2 max(1.0 pp, σ̂)² / n;
  the Welch guard), at 1,200 F games a final, over σ ∈ {0, 1.0, 2.0, 2.75 (the measured one-sided upper bound)} pp.
- ``contrast_path``: T − D1 on H + F under the global null, with the seed count chosen by the primary's look 1
  (§6.4: "3 v 3 if look 1 ended it, else 5 v 5") and T's F units 0–29 NESTED in its primary units 0–59.
- ``kakuna_specific_guard``: P(I_F point ≤ 0) for a true in-house gain x at the measured per-final S SD (5 v 5).
- ``r72_mde``: the Welch 80 % MDE of an in-house 3 v 3 and 5 v 5 read at the measured per-final S SD.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import chi2, norm, t as tdist

SEED = 61026
R = 60_000
Z = 1.959963985
FLOOR = 0.010
BAND = 0.030
TIE = 1e-9
rng = np.random.default_rng(SEED)


def interval(on, off, n, panels=1):
    """§6.3's interval for ON − OFF. on/off: (R, k, K) per-seed-final rates, n games per (final, panel)."""
    k = on.shape[1]
    pa, pc = on.mean(1), off.mean(1)
    d = pa.mean(1) - pc.mean(1)
    vbin = (pa * (1 - pa) / (n * k) + pc * (1 - pc) / (n * k)).sum(1) / panels ** 2
    ma, mc = on.mean(2), off.mean(2)
    w = (ma.var(1, ddof=1) + mc.var(1, ddof=1)) / 2
    b = ((on * (1 - on) / n).sum(2).mean(1) + (off * (1 - off) / n).sum(2).mean(1)) / 2 / panels ** 2
    df = 2 * (k - 1)
    hw = Z * np.sqrt(vbin + 2 * np.maximum(FLOOR ** 2, w - b) / k)
    guard = w * df / chi2.ppf(0.95, df) - b > FLOOR ** 2
    va, vc = ma.var(1, ddof=1) / k, mc.var(1, ddof=1) / k
    dfw = (va + vc) ** 2 / np.maximum((va ** 2 + vc ** 2) / (k - 1), 1e-30)
    return d, np.where(guard, tdist.ppf(0.975, dfw) * np.sqrt(va + vc), hw)


def finals(base, eff, sigma, n, k=5):
    u = rng.normal(0.0, sigma, (R, k, 1))
    return rng.binomial(n, np.clip(np.asarray(base) + np.asarray(eff) + u, 0.01, 0.99)) / n


def primary() -> dict:
    out = {}
    for sigma in (0.0, 0.01, 0.02, 0.0275):
        for delta in (0.0, 0.03):
            on, off = finals([0.49], [delta], sigma, 1200), finals([0.49], [0.0], sigma, 1200)
            d1, h1 = interval(on[:, :3], off[:, :3], 1200)
            d2, h2 = interval(on, off, 1200)
            stop = d1 <= TIE
            lo, hi = np.where(stop, d1 - h1, d2 - h2), np.where(stop, d1 + h1, d2 + h2)
            gain = ~stop & (lo > TIE)
            loss = hi < -TIE
            nug = ~gain & ~loss & (hi < BAND - TIE)
            out[f"sigma {sigma * 100:.2f} pp, delta {delta * 100:+.0f} pp"] = {
                "GAIN": round(float(gain.mean()), 3), "NO-USEFUL-GAIN": round(float(nug.mean()), 3),
                "stop_at_look1": round(float(stop.mean()), 3)}
    return out


def contrast_path() -> dict:
    out = {}
    for sigma in (0.0, 0.01, 0.02):
        u_t, u_d, u_c = (rng.normal(0.0, sigma, (R, 5)) for _ in range(3))
        pt_h, pt_f = np.clip(0.46 + u_t, .01, .99), np.clip(0.49 + u_t, .01, .99)
        pd_h, pd_f = np.clip(0.46 + u_d, .01, .99), np.clip(0.49 + u_d, .01, .99)
        tf_a, tf_b = rng.binomial(600, pt_f), rng.binomial(600, pt_f)      # T's F units 0-29 and 30-59
        t_prim = ((tf_a + tf_b) / 1200)[..., None]
        c_prim = (rng.binomial(1200, np.clip(0.49 + u_c, .01, .99)) / 1200)[..., None]
        stop = interval(t_prim[:, :3], c_prim[:, :3], 1200)[0] <= TIE
        tc = np.stack([rng.binomial(600, pt_h) / 600, tf_a / 600], -1)
        dc = np.stack([rng.binomial(600, pd_h) / 600, rng.binomial(600, pd_f) / 600], -1)
        d3, h3 = interval(tc[:, :3], dc[:, :3], 600, panels=2)
        d5, h5 = interval(tc, dc, 600, panels=2)
        d, h = np.where(stop, d3, d5), np.where(stop, h3, h5)
        lb, ub = d - h > TIE, d + h < -TIE
        out[f"sigma {sigma * 100:.1f} pp"] = {
            "P(look-1 stop)": round(float(stop.mean()), 3),
            "LADDER-BETTER overall / | stop / | continue": [round(float(x), 3) for x in
                                                            (lb.mean(), lb[stop].mean(), lb[~stop].mean())],
            "FORUM-BETTER overall / | stop / | continue": [round(float(x), 3) for x in
                                                           (ub.mean(), ub[stop].mean(), ub[~stop].mean())]}
    return out


def kakuna_specific_guard() -> dict:
    out = {}
    for sd in (0.37, 0.60, 0.86):
        se = (2 * sd ** 2 / 5) ** 0.5
        out[f"per-final S SD {sd} pp"] = {f"true in-house gain {x} pp": round(float(norm.cdf(-x / se)), 3)
                                          for x in (0.0, 0.25, 0.5, 1.0)}
    return out


def r72_mde() -> dict:
    out = {}
    for sd in (0.37, 0.60, 0.86):
        row = {}
        for k in (3, 5):
            se = (2 * sd ** 2 / k) ** 0.5
            df = 2 * (k - 1)
            row[f"{k} v {k}"] = round(float((tdist.ppf(0.975, df) + tdist.ppf(0.80, df)) * se), 2)
        out[f"per-final S SD {sd} pp"] = row
    return out


if __name__ == "__main__":
    print(json.dumps({"seed": SEED, "reps": R, "primary": primary(), "contrast_path": contrast_path(),
                      "kakuna_specific_guard_P(I_F point <= 0)": kakuna_specific_guard(),
                      "r72_welch_mde80_pp": r72_mde()}, indent=1))
