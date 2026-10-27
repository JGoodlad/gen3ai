"""X60 REDESIGN: the measured seed noise and the power of the registered rule (design_human_replay_lever.md §6).

CPU, seeded, deterministic, numpy + scipy, under a minute. Run from the repo root:
    python designs/research_state/measurements/x60_redesign_2026-10-26/power.py
It prints one JSON document; the design quotes ``results/power.json`` (the same output).

Part 1, the MEASURED noise (no simulation). The seed-to-seed spread of P*+20M fork finals against Kakuna, from the
committed per-seed counts of C0 and D1 (5 finals each, D1's look-2 input) and D3 (3 finals, its look-1 cells), at
7 units (140 games) per (seed final, panel). Per panel set: the pooled within-arm sample variance W of the per-seed
equal-weight means (df 10), the mean per-seed binomial variance b, phi = W / b, the point sigma^2 = W - b and its
one-sided 95 % UPPER bound W df / chi2_0.05(df) - b.

Part 2, the REGISTERED rule (the design's §6.3), simulated exactly, against the rule the draft registered (D1's AM-5
point phi with the sign rule; 'd1_point') and REVIEW D's milder fix ('d1_t': the same with t at df). A seed final's
score on panel k is Binomial(N_k, p_k + delta_k + u), u ~ N(0, sigma) per (arm, seed final), shared across panels.
- 'x60' (registered): look 1 (3 seeds) is a FUTILITY look only: delta-hat <= 0 ends the read with look 1's class
  (never GAIN); else look 2 (5 seeds) decides. The variance of ON - OFF is the stratified binomial part of each arm
  plus 2 max(SIGMA_F, sigma-hat)^2 / n_seeds, sigma-hat^2 = W - b from the two arms' per-seed means (df 2 (n - 1));
  if sigma-hat^2's one-sided 95 % LOWER bound exceeds SIGMA_F^2, a Welch t interval on the per-seed means decides
  instead. No sign rule. Classes: GAIN iff LB > 0; LOSS iff UB < 0; NO-USEFUL-GAIN iff 0 <= UB < +BAND (a gain
  of BAND is excluded); else NOT-RESOLVED. 'equivalent' (the interval inside [-BAND, +BAND]) is reported beside.
- 'd1_point' / 'd1_t': main.x35.d1 as it stands (Wald stand-in for Newcombe at p ~ 0.5), GAIN possible at look 1
  with >= 2 of 3 seeds up, CONTINUE iff UNDECIDED with delta-hat > 0, look 2 needs >= 3 of 5 seeds up.
Part 3, the contrasts: H-TEAMS - D1 on H + F (one look, 5 v 5 seeds, the x60 variance; classes T-BETTER / D1-BETTER
/ NO-DIFFERENCE inside +-BAND / NOT-RESOLVED), and the H-BC placebo read CONDITIONAL on an H-BC primary GAIN, on
FRESH games (the selection bias it leaves). Part 4: the in-house (Welch, 5 v 5 finals) MDE at the measured S spread.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, t as tdist

SEED = 20261026
Z = 1.959963985
REPS = 40_000
SIGMA_F = 0.010           # the registered seed-SD floor (1.0 pp of a final's score)
BAND = 0.03               # NO-DIFFERENCE band (+-3 pp), D1's NULL band
TIE = 1e-9
HERE = Path(__file__).resolve().parent
MEAS = HERE.parent
BASE = {"F": 0.49, "H": 0.46, "P72": 0.47}   # C0's own rates at D1's read (F 342/700), rounded


# ---------------------------------------------------------------- part 1: measured noise
def measured() -> dict:
    d1 = json.loads((MEAS / "x35_breakers_stage_2026-10-17/d1_read/read_look2_input.json").read_text())["h"]
    d3 = json.loads((MEAS / "x41_d3_setmut_2026-10-25/read/cells.json").read_text())["anchor"]["h"]["cells"]
    arms = {"C0": d1["off"], "D1": d1["on"],
            "D3": {k.split("_s")[1]: {p: [x["wins"], x["n"]] for p, x in v.items()} for k, v in d3.items()}}
    name = {"F": "forum72", "P72": "p72", "H": "heldout72_hpfix"}
    out = {"source": "C0 + D1 (5 finals each, d1_read/read_look2_input.json), D3 (3 finals, look-1 cells.json); "
                     "140 games per (seed final, panel)"}
    for ps in (("F",), ("F", "P72"), ("H", "F"), ("H", "F", "P72")):
        num = den = 0.0
        bs = []
        means = {}
        for a, seeds in arms.items():
            m = []
            for s in sorted(seeds):
                c = seeds[s]
                pr = [c[name[p]][0] / c[name[p]][1] for p in ps]
                n = [c[name[p]][1] for p in ps]
                m.append(float(np.mean(pr)))
                bs.append(sum(x * (1 - x) / g for x, g in zip(pr, n)) / len(ps) ** 2)
            num += np.var(m, ddof=1) * (len(m) - 1)
            den += len(m) - 1
            means[a] = [round(100 * x, 1) for x in m]
        w, b, df = num / den, float(np.mean(bs)), int(den)
        out["+".join(ps)] = {"per_seed_means_pp": means, "W_sd_pp": round(100 * w ** 0.5, 2),
                             "binomial_sd_pp": round(100 * b ** 0.5, 2), "phi": round(w / b, 2), "df": df,
                             "sigma_point_pp": round(100 * max(0.0, w - b) ** 0.5, 2),
                             "sigma_upper95_pp": round(100 * max(0.0, w * df / chi2.ppf(0.05, df) - b) ** 0.5, 2)}
    return out


# ---------------------------------------------------------------- part 2: the rules
def finals(rng, panels, n, eff, sigma, seeds=5):
    """(REPS, seeds, K) win fractions of one arm; n[k] games per (seed final, panel k)."""
    p = np.array([BASE[k] for k in panels]) + np.asarray(eff)
    u = rng.normal(0.0, sigma, (REPS, seeds, 1))
    return rng.binomial(np.asarray(n)[None, None, :], np.clip(p[None, None, :] + u, 0.01, 0.99)) / np.asarray(n)


def x60_interval(a, c, n):
    """The registered interval of ON (a) - OFF (c): (delta-hat, half-width, welch_used). a, c: (REPS, ns, K)."""
    n = np.asarray(n, float)
    ns, k = a.shape[1], a.shape[2]
    pa, pc = a.mean(1), c.mean(1)
    d = pa.mean(1) - pc.mean(1)
    vbin = ((pa * (1 - pa) / (n * ns)).sum(1) + (pc * (1 - pc) / (n * ns)).sum(1)) / k ** 2
    ma, mc = a.mean(2), c.mean(2)
    w = (ma.var(1, ddof=1) + mc.var(1, ddof=1)) / 2
    b = (((a * (1 - a)) / n).sum(2).mean(1) + ((c * (1 - c)) / n).sum(2).mean(1)) / 2 / k ** 2
    df = 2 * (ns - 1)
    s2 = np.maximum(SIGMA_F ** 2, w - b)
    hw = Z * np.sqrt(vbin + 2 * s2 / ns)
    guard = w * df / chi2.ppf(0.95, df) - b > SIGMA_F ** 2
    va, vc = ma.var(1, ddof=1) / ns, mc.var(1, ddof=1) / ns
    dfw = (va + vc) ** 2 / np.maximum(va ** 2 / (ns - 1) + vc ** 2 / (ns - 1), 1e-30)
    hw_w = tdist.ppf(0.975, dfw) * np.sqrt(va + vc)
    return d, np.where(guard, hw_w, hw), guard


def classes(d, hw, band=BAND, gain_ok=None, one_sided=True):
    """GAIN / LOSS / NO-USEFUL-GAIN (UB < +band; one_sided) or NO-DIFF (inside +-band; two-sided) / NOT-RESOLVED."""
    lo, hi = d - hw, d + hw
    gain = lo > TIE if gain_ok is None else (lo > TIE) & gain_ok
    small = (hi < band - TIE) if one_sided else (lo > -band + TIE) & (hi < band - TIE)
    return np.where(gain, "GAIN", np.where(hi < -TIE, "LOSS",
                    np.where(small, "NO-USEFUL-GAIN" if one_sided else "NO-DIFF", "NOT-RESOLVED")))


def rule_x60(a, c, n, band=BAND):
    d1, h1, _ = x60_interval(a[:, :3], c[:, :3], n)
    d2, h2, g2 = x60_interval(a, c, n)
    c1 = classes(d1, h1, band)
    c1 = np.where(c1 == "GAIN", "NOT-RESOLVED", c1)          # never GAIN at look 1 (cannot occur with d1 <= 0)
    stop = d1 <= TIE
    final = np.where(stop, c1, classes(d2, h2, band))
    lo, hi = np.where(stop, d1 - h1, d2 - h2), np.where(stop, d1 + h1, d2 + h2)
    equiv = (lo > -band + TIE) & (hi < band - TIE)
    return final, stop, h2, g2, equiv


def rule_d1(a, c, n, mode, band=BAND):
    """main.x35.d1 as it stands; its NULL is the two-sided band."""
    n = np.asarray(n, float)
    k = a.shape[2]
    res = []
    for ns, sign_min in ((3, 2), (5, 3)):
        x, y = a[:, :ns], c[:, :ns]
        px, py = x.mean(1), y.mean(1)
        d = px.mean(1) - py.mean(1)
        v = ((px * (1 - px) + py * (1 - py)) / (n * ns)).sum(1) / k ** 2

        def ratio(z):
            return z.mean(2).var(1, ddof=1) / (((z * (1 - z)) / n).sum(2).mean(1) / k ** 2)
        phi = np.maximum(1.0, (ratio(x) + ratio(y)) / 2)
        q = Z if mode == "point" else tdist.ppf(0.975, 2 * (ns - 1))
        hw = q * np.sqrt(v * phi)
        sign = ((x.mean(2) - y.mean(2)) > TIE).sum(1) >= sign_min
        res.append((classes(d, hw, band, sign, one_sided=False), d))
    (c1, d1), (c2, _) = res
    c1 = np.where(c1 == "NOT-RESOLVED", "UNDECIDED", c1)
    c2 = np.where(c2 == "NOT-RESOLVED", "UNDECIDED", c2)
    return np.where((c1 == "UNDECIDED") & (d1 > TIE), c2, c1)


def share(cls, names=("GAIN", "LOSS", "NO-USEFUL-GAIN", "NO-DIFF", "NOT-RESOLVED", "UNDECIDED")):
    return {k: round(float((cls == k).mean()), 3) for k in names if (cls == k).any()}


def primary(rng) -> dict:
    out = {}
    for n in (1200, 1600, 2000):
        for sigma in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
            for delta in (0.0, 0.02, 0.03, 0.04):
                a = finals(rng, ("F",), [n], [delta], sigma)
                c = finals(rng, ("F",), [n], [0.0], sigma)
                fin, stop, h2, g2, eq = rule_x60(a, c, [n])
                key = f"F alone, {n} games/final, sigma {sigma * 100:.1f} pp, delta {delta * 100:+.0f} pp"
                row = {"x60": share(fin), "x60_equivalent_reported": round(float(eq.mean()), 3),
                       "x60_stop_at_look1": round(float(stop.mean()), 3),
                       "x60_halfwidth_look2_pp": round(float(np.median(h2)) * 100, 2),
                       "x60_welch_guard_fires": round(float(g2.mean()), 3)}
                if n == 1200:
                    row["d1_point (the draft)"] = share(rule_d1(a, c, [n], "point"))
                    row["d1_t (REVIEW D's fix)"] = share(rule_d1(a, c, [n], "t"))
                out[key] = row
    return out


def contrast_teams(rng) -> dict:
    """H-TEAMS - D1 on H + F, equal weight, 5 v 5 seeds, one look, the x60 variance, at n_c games per (seed final,
    panel) on BOTH arms (the same unit seeds 0 .. n_c / 20 - 1; H-TEAMS's primary F units beyond them are not used)."""
    out = {}
    for nc in (600, 1200):
        for sigma in (0.0, 0.01, 0.015, 0.02):
            for delta in (0.0, 0.02, 0.03, 0.04):
                a = finals(rng, ("H", "F"), [nc, nc], [delta, delta], sigma)
                c = finals(rng, ("H", "F"), [nc, nc], [0.0, 0.0], sigma)
                d, hw, g = x60_interval(a, c, [nc, nc])
                cls = classes(d, hw, one_sided=False)
                out[f"n_c {nc}, sigma {sigma * 100:.1f} pp, delta {delta * 100:+.0f} pp"] = {
                    **share(cls), "halfwidth_pp": round(float(np.median(hw)) * 100, 2)}
    return out


def placebo(rng) -> dict:
    """H-BC's placebo, played only after an H-BC primary GAIN: H-BC's FRESH games (new unit seeds; the same trained
    finals, so the same seed effects u) against P-BC's, on F at 1,200 a final; the x60 interval, one look.
    'labels matter' = LB > 0. Under H0 both arms carry the same true gain e."""
    out = {}
    n = 1200
    for sigma in (0.0, 0.01, 0.015):
        for e_on, e_pl in ((0.02, 0.02), (0.03, 0.03), (0.04, 0.04), (0.04, 0.01), (0.05, 0.02)):
            p = BASE["F"]
            u_on = rng.normal(0.0, sigma, (REPS, 5, 1))
            on = rng.binomial(n, np.clip(p + e_on + u_on, 0.01, 0.99)) / n
            off = finals(rng, ("F",), [n], [0.0], sigma)
            fin = rule_x60(on, off, [n])[0]
            gain = fin == "GAIN"
            on_fresh = rng.binomial(n, np.clip(p + e_on + u_on, 0.01, 0.99)) / n
            pl = finals(rng, ("F",), [n], [e_pl], sigma)
            d, hw, _ = x60_interval(on_fresh, pl, [n])
            lm = (d - hw > TIE)
            d0, hw0, _ = x60_interval(on, pl, [n])           # the same games re-used (the rejected design)
            lm0 = (d0 - hw0 > TIE)
            out[f"sigma {sigma * 100:.1f} pp, H-BC {e_on * 100:+.0f} pp, P-BC {e_pl * 100:+.0f} pp"] = {
                "P(H-BC GAIN)": round(float(gain.mean()), 3),
                "P(labels matter | GAIN), fresh games": round(float(lm[gain].mean()), 3) if gain.any() else None,
                "P(labels matter | GAIN), primary games re-used": round(float(lm0[gain].mean()), 3) if gain.any() else None}
    return out


def inhouse() -> dict:
    """The in-house S meter (X35's population score vs the frozen R, 7 x 500 mirrored pairs a final): its measured
    per-final SD (C0 and D1, five finals each, home72) and the Welch 5 v 5 MDE at 80 % power (two-sided 95 %)."""
    p = json.loads((MEAS / "x35_breakers_stage_2026-10-17/d1_read/read_look2_input.json").read_text())["p"]
    sd = {a: float(np.std([p[a][s] for s in sorted(p[a])], ddof=1)) for a in ("off", "on")}
    out = {"per_final_sd_pp": {"C0": round(sd["off"], 2), "D1": round(sd["on"], 2)}}
    for s1, s2, tag in ((sd["off"], sd["off"], "both arms at C0's SD"), (sd["off"], sd["on"], "C0's and D1's SD"),
                        (sd["on"], sd["on"], "both arms at D1's SD")):
        se = np.sqrt((s1 ** 2 + s2 ** 2) / 5)
        df = (s1 ** 2 / 5 + s2 ** 2 / 5) ** 2 / ((s1 ** 2 / 5) ** 2 / 4 + (s2 ** 2 / 5) ** 2 / 4)
        out[f"MDE80_pp ({tag})"] = round(float((tdist.ppf(0.975, df) + tdist.ppf(0.80, df)) * se), 2)
    return out


def eight_seeds(rng) -> dict:
    """REVIEW D's D-B1 item 3, costed: the same rule at 8 seeds a arm (look 1 at 3, look 2 at 8; C0 would need three
    new seeds too), F alone, 1,200 games a final. For comparison only; the design registers 5 seeds, no extension."""
    out = {}
    for sigma in (0.0, 0.01, 0.015, 0.02):
        for delta in (0.0, 0.03):
            a = finals(rng, ("F",), [1200], [delta], sigma, seeds=8)
            c = finals(rng, ("F",), [1200], [0.0], sigma, seeds=8)
            fin = rule_x60(a, c, [1200])[0]
            out[f"8 seeds, sigma {sigma * 100:.1f} pp, delta {delta * 100:+.0f} pp"] = share(fin)
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    r = {"seed": SEED, "reps": REPS, "sigma_floor_pp": SIGMA_F * 100, "band_pp": BAND * 100, "base": BASE,
         "measured_noise": measured(), "primary": primary(rng), "contrast_teams_minus_d1": contrast_teams(rng),
         "placebo_after_gain": placebo(rng), "inhouse": inhouse(),
         "eight_seeds_for_comparison": eight_seeds(rng)}
    s = json.dumps(r, indent=1)
    print(s)
    if "--write" in sys.argv:
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / "power.json").write_text(s + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
