"""X60-T PREREG DRAFT: the CPU analyses REVIEW F requires before route T registers (F-M4, F-M6, F-M5, F-M8, F-M2, F-M1).

CPU only, seeded, deterministic, numpy + scipy; read-only on ``~/gen3ai_archive`` and ``models/``. Run from the repo root:

    export PYTHONPATH=$PYTHONPATH:src
    python designs/research_state/measurements/x60t_h_teams_2026-11-06/analysis.py --write

It writes ``results/analysis.json`` (the PREREG quotes it). Every simulated verdict uses the REGISTERED rule as
``x60_redesign_2026-10-26/power.py`` codes it: this file IMPORTS ``finals`` and ``rule_x60`` (and the constants they
read) and never re-implements them (the brief: "do not fork its logic").

Parts (each REVIEW F finding's "fix" step):
- ``fm4``: sigma-hat, the seed-to-seed SD of a P*+20M final's F rate, from X61P's 10 finals at 60 F units (1,200 games)
  each, with §6.3's estimator (W - b, df 2 (n - 1)); its one-sided 95 % UPPER bound; fix 3's seed-count decision
  (UB > 1.5 pp => 8 seeds an arm); the power grid at 5 and 8 seeds through power.py's rule.
- ``fm6``: the meters each arm has; the exchange rate between the Kakuna F meter and the in-house meter (what can be
  measured: the only SAME-panel pair, home72, N1-P - C0); the I_F guard's two error rates at a grid of rates; the
  end-to-end ADOPT-CANDIDATE simulation (Delta_F through power.py, I_F, Delta_P, goal (a)).
- ``fm5``: R72's per-team draw dose in 20M (D1 doc §4.1's 570k games) by corpus size N, and R72's Welch MDE at 3 v 3,
  5 v 5 and 8 v 8 from the measured per-final S SDs.
- ``fm8``: measured per-game / per-set CPU rates and realised wall throughput from X61P's and D3's read states; GPU-h
  from the X35 dose records; the cost table.
- ``fm1``: sha256 of the design doc at e1512c361 and of its §6 text; whether §6 moved since.
- ``fm2``: the engine at the registered play commit (main.reads.engine), and where C0's banked cells sit.
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm, t as tdist

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent
REPO = MEAS.parents[2]
ARCH = Path(os.environ.get("GEN3AI_ARCHIVE_DIR", os.path.expanduser("~/gen3ai_archive")))
SEED = 20261106
REPS = 40_000

# ---- power.py, imported (never re-implemented) ------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location("x60_power", MEAS / "x60_redesign_2026-10-26" / "power.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
P.REPS = REPS


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ================================================================== F-M4: sigma-hat from X61P's 10 finals
def x61p_cells() -> dict:
    c = json.loads((ARCH / "x61p_n1p_read/look2/cells.json").read_text())
    f = {lab: (v["forum72"]["wins"], v["forum72"]["n"]) for lab, v in c["anchor"]["primary"]["cells"].items()}
    rep = {lab: {p: (x["wins"], x["n"]) for p, x in v.items()} for lab, v in c["anchor"]["reported"]["cells"].items()}
    s = {lab: v["s_pp"] for lab, v in c["s"]["p"]["nodes"].items()}
    se = {lab: v["se_pp"] for lab, v in c["s"]["p"]["nodes"].items()}
    return {"F": f, "reported": rep, "S_home72": s, "S_home72_se": se,
            "regime_F": c["anchor"]["primary"]["regime_ids"], "engine": c["anchor"]["primary"]["engine"],
            "S_regime": c["s"]["p"].get("imports", {}).get("regime_id"), "spec_sha256": c["spec_sha256"]}


def sigma_hat(arms: dict) -> dict:
    """§6.3's estimator: W = the mean of the arms' sample variances of their per-seed F rates (equal seeds per arm, so
    the df-pooled variance), b = the mean per-seed binomial variance p (1 - p) / n, df = sum (n_a - 1);
    sigma^2 = W - b; one-sided 95 % UPPER bound W df / chi2_0.05(df) - b; the guard's LOWER bound W df / chi2_0.95 - b."""
    num = den = 0.0
    bs, per = [], {}
    for a, cells in arms.items():
        p = np.array([w / n for w, n in cells])
        n = np.array([n for _, n in cells], float)
        num += p.var(ddof=1) * (len(p) - 1)
        den += len(p) - 1
        bs += list(p * (1 - p) / n)
        per[a] = {"rates_pp": [round(100 * x, 2) for x in p], "sd_pp": round(100 * p.std(ddof=1), 3),
                  "mean_pp": round(100 * p.mean(), 3)}
    w, b, df = num / den, float(np.mean(bs)), int(den)
    up = w * df / chi2.ppf(0.05, df) - b
    lo = w * df / chi2.ppf(0.95, df) - b
    return {"per_arm": per, "W_sd_pp": round(100 * w ** 0.5, 3), "binomial_sd_pp": round(100 * b ** 0.5, 3),
            "phi": round(w / b, 3), "df": df, "sigma2_point_pp2": round(1e4 * (w - b), 4),
            "sigma_point_pp": round(100 * max(0.0, w - b) ** 0.5, 3),
            "sigma_upper95_pp": round(100 * max(0.0, up) ** 0.5, 3),
            "sigma2_lower95_pp2": round(1e4 * lo, 4),
            "guard_would_fire_at_floor_1pp": bool(lo > P.SIGMA_F ** 2)}


def primary_grid(rng, sigmas_pp, seeds) -> dict:
    """power.py's registered rule (finals + rule_x60, look 1 at 3 seeds, look 2 at ``seeds``)."""
    out = {}
    for s in sigmas_pp:
        for delta in (0.0, 0.02, 0.03, 0.04):
            a = P.finals(rng, ("F",), [1200], [delta], s / 100, seeds=seeds)
            c = P.finals(rng, ("F",), [1200], [0.0], s / 100, seeds=seeds)
            fin, stop, h2, g2, eq = P.rule_x60(a, c, [1200])
            out[f"{seeds} seeds, sigma {s:.2f} pp, delta {delta * 100:+.0f} pp"] = {
                **P.share(fin), "stop_at_look1": round(float(stop.mean()), 3),
                "halfwidth_look2_pp": round(float(np.median(h2)) * 100, 2),
                "welch_guard_fires": round(float(g2.mean()), 3)}
    return out


def fm4(rng) -> dict:
    x = x61p_cells()
    c0 = [x["F"][f"c0_s350{i}"] for i in range(1, 6)]
    n1 = [x["F"][f"n1p_s350{i}"] for i in range(1, 6)]
    pooled = sigma_hat({"C0": c0, "N1-P": n1})
    c0_only = sigma_hat({"C0": c0})
    n1_only = sigma_hat({"N1-P": n1})
    ub = pooled["sigma_upper95_pp"]
    decision = "8 seeds an arm" if ub > 1.5 + 1e-9 else "5 seeds an arm"
    sig = sorted({0.0, pooled["sigma_point_pp"], 1.0, 1.5, round(ub, 2), 2.0, 2.75})
    grid5 = primary_grid(rng, sig, 5)
    grid8 = primary_grid(rng, sig, 8)
    return {"source": f"X61P look 2 cells.json (spec {x['spec_sha256'][:12]}), forum72 regime {x['regime_F']}, "
                      f"engine {x['engine']}; 60 units = 1,200 games a final; ties are non-wins",
            "pooled_C0_N1P": pooled, "C0_only": c0_only, "N1P_only": n1_only,
            "fix3_rule": "UB (pooled, one-sided 95 %) > 1.5 pp => 8 seeds an arm; decided on control-only data "
                         "(X61P's arms, no X60 data)",
            "fix3_decision": decision, "power_5_seeds": grid5, "power_8_seeds": grid8,
            "base_F_rate_used": P.BASE["F"],
            "base_F_rate_measured_C0_X61P": round(sum(w for w, _ in c0) / sum(n for _, n in c0), 4)}


# ================================================================== F-M6: the exchange rate and the I_F guard
def welch(a_mean, a_var, na, b_mean, b_var, nb):
    va, vb = a_var / na, b_var / nb
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / np.maximum(va ** 2 / (na - 1) + vb ** 2 / (nb - 1), 1e-30)
    return a_mean - b_mean, se, df


def inventory() -> dict:
    """Which arms carry which meter (read from the banked records; no S group on forum72 exists anywhere)."""
    s_forum = []
    for f in list(glob.glob(str(ARCH / "**/spec.json"), recursive=True)) + \
            list(glob.glob(str(MEAS / "**/read_spec*.json"), recursive=True)):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        for g, v in (d.get("s") or {}).items():
            if isinstance(v, dict) and v.get("panel") == "forum72":
                s_forum.append(f"{f}:{g}")
    return {
        "S_groups_on_forum72_anywhere": s_forum,
        "arms": {
            "D1 (rb_x35_d1wide)": "Kakuna F/H/P72 5 x 140 (E1); S home72 5 v 5 (D1 P read); S pool 3 v 3; no S on F",
            "D3 (rb_x35_d3mut)": "Kakuna F/H/P72 5 x 140 (look 2, E1); S home72 5 (vs C0's banked); no S on F",
            "N1-P (rb_x35_n1p)": "Kakuna F 5 x 1,200 + home72/H/P72 5 x 140 (E3); S home72 5 v 5; no S on F",
            "X26c (rb_x26c_s2601 @480M)": "Kakuna H/F/P72 vs P* (60 units); S home72 vs X26 @380M (not vs P*): "
                                          "the two meters have DIFFERENT baselines; no S on F",
            "belief prior mix (lambda 0.25 / 1.0)": "Kakuna only (an inference-time load option; no in-house read)",
        },
        "I_F_available_for": [],
    }


def same_panel_home72() -> dict:
    """The only SAME-panel pair of meters in the archive: N1-P - C0 on home72, Kakuna (7 units a final) and S (vs R)."""
    x = x61p_cells()
    k = {a: [x["reported"][f"{a}_s350{i}"]["home72"] for i in range(1, 6)] for a in ("c0", "n1p")}
    kw = {a: sum(w for w, _ in v) for a, v in k.items()}
    kn = {a: sum(n for _, n in v) for a, v in k.items()}
    pk = {a: kw[a] / kn[a] for a in k}
    dk = 100 * (pk["n1p"] - pk["c0"])
    se_k = 100 * np.sqrt(sum(pk[a] * (1 - pk[a]) / kn[a] for a in k))
    s = {a: np.array([x["S_home72"][f"{a}_s350{i}"] for i in range(1, 6)]) for a in ("c0", "n1p")}
    ds, se_s, df_s = welch(s["n1p"].mean(), s["n1p"].var(ddof=1), 5, s["c0"].mean(), s["c0"].var(ddof=1), 5)
    # the ratio ΔS / ΔK by simulation of both estimates' sampling distributions (Fieller-like percentile)
    rng = np.random.default_rng(SEED + 6)
    dk_d = rng.normal(dk, se_k, 200_000)
    ds_d = ds + se_s * rng.standard_t(df_s, 200_000)
    ok = np.abs(dk_d) > 0.25
    r = ds_d[ok] / dk_d[ok]
    return {"kakuna_home72": {"c0": f"{kw['c0']}/{kn['c0']}", "n1p": f"{kw['n1p']}/{kn['n1p']}",
                              "delta_pp": round(dk, 2), "se_pp": round(float(se_k), 2)},
            "S_home72": {"delta_pp": round(float(ds), 2), "se_pp": round(float(se_s), 2), "df": round(float(df_s), 1),
                         "sd_c0_pp": round(float(s["c0"].std(ddof=1)), 3),
                         "sd_n1p_pp": round(float(s["n1p"].std(ddof=1)), 3)},
            "rate_point_dS_over_dK": round(float(ds / dk), 2),
            "rate_80pct_range_simulated": [round(float(np.percentile(r, 10)), 2), round(float(np.percentile(r, 90)), 2)],
            "reading": "one arm, one panel (home72, in-pool), a NEGATIVE effect: the ratio's 80 % range spans "
                       "zero-to-several, so it does not measure the F exchange rate"}


def lever_signatures() -> dict:
    """Cross-panel, per lever: Kakuna F (and U) against in-house S on home72 (P). NOT an exchange rate: the meters sit
    on different panels; reported to show the two can move in opposite directions under a breadth lever."""
    out = {}
    d1 = json.loads((MEAS / "x35_breakers_stage_2026-10-17/d1_read/read_look2_input.json").read_text())
    for arm, on, off, pk in (("D1 - C0 (E1)", d1["h"]["on"], d1["h"]["off"], d1["p"]),):
        f_on = sum(on[s]["forum72"][0] for s in on) / sum(on[s]["forum72"][1] for s in on)
        f_off = sum(off[s]["forum72"][0] for s in off) / sum(off[s]["forum72"][1] for s in off)
        s_on = np.array([pk["on"][k] for k in sorted(pk["on"])])
        s_off = np.array([pk["off"][k] for k in sorted(pk["off"])])
        out[arm] = {"dK_F_pp": round(100 * (f_on - f_off), 2), "dS_home72_pp": round(float(s_on.mean() - s_off.mean()), 2),
                    "sd_S_on_pp": round(float(s_on.std(ddof=1)), 3), "sd_S_off_pp": round(float(s_off.std(ddof=1)), 3)}
    d3 = json.loads((MEAS / "x41_d3_setmut_2026-10-25/read/look2_cells.json").read_text())
    d3f = [v["forum72"] for v in d3["anchor"]["h"]["cells"].values()]
    d3s = np.array([v["s_pp"] for v in d3["s"]["p"]["nodes"].values()])
    c0f = d1["h"]["off"]
    f_off = sum(c0f[s]["forum72"][0] for s in c0f) / sum(c0f[s]["forum72"][1] for s in c0f)
    s_off = np.array([d1["p"]["off"][k] for k in sorted(d1["p"]["off"])])
    out["D3 - C0 (D3 E1 look 2; C0 D1's banked E1)"] = {
        "dK_F_pp": round(100 * (sum(x["wins"] for x in d3f) / sum(x["n"] for x in d3f) - f_off), 2),
        "dS_home72_pp": round(float(d3s.mean() - s_off.mean()), 2), "sd_S_on_pp": round(float(d3s.std(ddof=1)), 3)}
    x = x61p_cells()
    fk = {a: sum(x["F"][f"{a}_s350{i}"][0] for i in range(1, 6)) / 6000 for a in ("c0", "n1p")}
    sk = {a: np.array([x["S_home72"][f"{a}_s350{i}"] for i in range(1, 6)]) for a in ("c0", "n1p")}
    out["N1-P - C0 (E3)"] = {"dK_F_pp": round(100 * (fk["n1p"] - fk["c0"]), 2),
                             "dS_home72_pp": round(float(sk["n1p"].mean() - sk["c0"].mean()), 2),
                             "sd_S_on_pp": round(float(sk["n1p"].std(ddof=1)), 3),
                             "sd_S_off_pp": round(float(sk["c0"].std(ddof=1)), 3)}
    return out


def structural_rate(p_k=0.50, p_s=0.50) -> float:
    """Under a common logit-skill shift (the lever adds delta logits to the trainee's play on F against ANY opponent),
    dP/d(delta) = p (1 - p) on each meter, so the in-house gain per Kakuna pp is p_s (1 - p_s) / (p_k (1 - p_k))."""
    return p_s * (1 - p_s) / (p_k * (1 - p_k))


def guard_errors(rng, rates, sds, n=5) -> dict:
    """I_F guard: Welch across n v n finals. 'point': pass iff the point estimate > 0 (the design). 'lb90': pass iff
    the one-sided 90 % LB > 0 (fix 3). Error 1 = the LEAK (pass when the true in-house gain is 0); error 2 = the COST
    (fail when the Kakuna F gain is a true +3 pp and the in-house gain is rate x 3 pp)."""
    out = {}
    for sd in sds:
        for true_if in [0.0] + [3.0 * r for r in rates]:
            a = rng.normal(true_if, sd, (REPS, n))
            b = rng.normal(0.0, sd, (REPS, n))
            d, se, df = welch(a.mean(1), a.var(1, ddof=1), n, b.mean(1), b.var(1, ddof=1), n)
            lb90 = d - tdist.ppf(0.90, df) * se
            out[f"sd {sd:.2f} pp, true I_F {true_if:+.2f} pp, {n} v {n}"] = {
                "pass_point": round(float((d > 1e-9).mean()), 3), "pass_lb90": round(float((lb90 > 1e-9).mean()), 3)}
    return out


def end_to_end(rng, seeds, sigma_pp, dk, rate, sd_on, sd_off, dp_true, home_cost, rule) -> dict:
    """P(ADOPT-CANDIDATE) and P(TRADE) for one scenario. Delta_F: power.py's finals + rule_x60. I_F: Welch over the
    look-2 finals (point or lb90). Delta_P: D1's rule, Welch two-sided 97.5 %, NON-INFERIOR iff LB > -2.0 pp.
    goal (a): ON's Kakuna home72, 7 units (140 games) a final pooled over the look-2 finals, Wilson LB > 0.50 at a true
    rate 0.560 (C0's, X61P) + home_cost. G-F / G-E / G-C are NOT modelled (taken as PASS). The meters' sampling is
    drawn independently given the truth (a FINDING: seed effects shared across meters would correlate them)."""
    a = P.finals(rng, ("F",), [1200], [dk / 100], sigma_pp / 100, seeds=seeds)
    c = P.finals(rng, ("F",), [1200], [0.0], sigma_pp / 100, seeds=seeds)
    fin = P.rule_x60(a, c, [1200])[0]
    gain = fin == "GAIN"
    ia = rng.normal(rate * dk, sd_on, (REPS, seeds))
    ib = rng.normal(0.0, sd_off, (REPS, seeds))
    d, se, df = welch(ia.mean(1), ia.var(1, ddof=1), seeds, ib.mean(1), ib.var(1, ddof=1), seeds)
    i_pass = (d > 1e-9) if rule == "point" else (d - tdist.ppf(0.90, df) * se > 1e-9)
    pa = rng.normal(dp_true, sd_on, (REPS, seeds))
    pb = rng.normal(0.0, sd_off, (REPS, seeds))
    d, se, df = welch(pa.mean(1), pa.var(1, ddof=1), seeds, pb.mean(1), pb.var(1, ddof=1), seeds)
    noninf = d - tdist.ppf(1 - 0.025 / 2, df) * se > -2.0 + 1e-9
    ng = 140 * seeds
    x = rng.binomial(ng, 0.560 + home_cost / 100, REPS)
    ph, z = x / ng, norm.ppf(0.975)
    wil_lb = (ph + z * z / (2 * ng) - z * np.sqrt(ph * (1 - ph) / ng + z * z / (4 * ng * ng))) / (1 + z * z / ng)
    goal_a = wil_lb > 0.50 + 1e-9
    adopt = gain & i_pass & noninf & goal_a
    trade = gain & i_pass & ~(noninf & goal_a)
    kspec = gain & ~i_pass
    return {"P(GAIN)": round(float(gain.mean()), 3), "P(I_F pass | GAIN)": round(float(i_pass[gain].mean()), 3)
            if gain.any() else None, "P(Delta_P NON-INFERIOR)": round(float(noninf.mean()), 3),
            "P(goal a)": round(float(goal_a.mean()), 3), "P(ADOPT-CANDIDATE)": round(float(adopt.mean()), 3),
            "P(TRADE)": round(float(trade.mean()), 3), "P(KAKUNA-SPECIFIC)": round(float(kspec.mean()), 3)}


def fm6(rng, seeds_registered) -> dict:
    sp = same_panel_home72()
    rates = [0.25, 0.5, 1.0, 1.8]
    sds = [0.36, 0.56, 0.86]
    out = {"inventory": inventory(), "same_panel_home72_N1P_minus_C0": sp,
           "lever_signatures_cross_panel": lever_signatures(),
           "structural_rate_logit": {"p_k 0.50, p_s 0.50": round(structural_rate(0.50, 0.50), 3),
                                     "p_k 0.504, p_s 0.57": round(structural_rate(0.504, 0.57), 3)},
           "rates_grid": rates, "sds_grid_pp": sds,
           "guard_errors_5v5": guard_errors(rng, rates, sds, 5)}
    if seeds_registered == 8:
        out["guard_errors_8v8"] = guard_errors(rng, rates, sds, 8)
    e2e = {}
    sig = 1.0
    for rule in ("point", "lb90"):
        for rate in (0.0, 0.25, 0.5, 1.0):
            for dp, hc in ((0.0, 0.0), (-1.0, -2.0), (-1.6, -2.0), (-3.3, -4.0)):
                for sd_on in (0.56, 0.86):
                    key = (f"{rule}, {seeds_registered} seeds, sigma {sig} pp, dK_F +3 pp, rate {rate}, "
                           f"dP {dp:+.1f} pp, home72 Kakuna {hc:+.1f} pp, sd_on {sd_on} / sd_off 0.36")
                    e2e[key] = end_to_end(rng, seeds_registered, sig, 3.0, rate, sd_on, 0.36, dp, hc, rule)
    out["end_to_end_adopt"] = e2e
    # the null: dK_F = 0 everywhere (false ADOPT)
    out["end_to_end_null"] = {rule: end_to_end(rng, seeds_registered, sig, 0.0, 1.0, 0.56, 0.36, 0.0, 0.0, rule)
                              for rule in ("point", "lb90")}
    return out


# ================================================================== F-M5: R72's dose and MDE
def mde80(s1, s2, n):
    """Welch, two-sided 95 %, 80 % power; the formula power.py's ``inhouse`` uses (there with D1's SDs, fixed)."""
    se = np.sqrt((s1 ** 2 + s2 ** 2) / n)
    df = (s1 ** 2 / n + s2 ** 2 / n) ** 2 / ((s1 ** 2 / n) ** 2 / (n - 1) + (s2 ** 2 / n) ** 2 / (n - 1))
    return float((tdist.ppf(0.975, df) + tdist.ppf(0.80, df)) * se)


def fm5() -> dict:
    games_20m = 570_000          # D1 doc §4.1's rule (design_diversity_teams_opponents.md: "a 20M fork plays ≈ 570k games")
    share = 0.5
    per_side = games_20m * share
    dose = {f"N {n}": {"draws_per_team_per_side": round(per_side / n, 1),
                       "draws_per_team_both_sides": round(2 * per_side / n, 1),
                       "vs a pool team per side in C0 (≈ 790)": round(per_side / n / 790, 3),
                       "vs P*'s cumulative pilots per pool team (≈ 9,300)": round(per_side / n / 9300, 4)}
            for n in (1500, 2000, 2500, 3000)}
    sds = {"C0 (X61P, home72)": 0.36, "C0 (D1 read)": 0.37, "N1-P (X61P)": 0.56, "D1": 0.86}
    mde = {}
    for nm, s in sds.items():
        mde[nm] = {f"{n} v {n}": round(mde80(s, s, n), 2) for n in (3, 5, 8)}
        mde[nm]["3 v 5 (look-1 ON vs C0's five)"] = round(float(
            (lambda se, df: (tdist.ppf(0.975, df) + tdist.ppf(0.8, df)) * se)(
                np.sqrt(s * s / 3 + 0.36 ** 2 / 5),
                (s * s / 3 + 0.36 ** 2 / 5) ** 2 / ((s * s / 3) ** 2 / 2 + (0.36 ** 2 / 5) ** 2 / 4))), 2)
    # the expected R72 effect: GUESS brackets from H-CTX's home premium (P* on home72 vs F against Kakuna: +7.5 pp at
    # ≈ 9,300 pilots a pool team), scaled by exposure (linear, and log(1 + draws)), Kakuna pp
    prem, pilots = 7.5, 9300
    guess = {f"N {n}": {"linear_pp": round(prem * (per_side / n) / pilots, 2),
                        "log_pp": round(prem * np.log1p(per_side / n) / np.log1p(pilots), 2)}
             for n in (1500, 2000, 2500, 3000)}
    return {"games_per_20M": games_20m, "share": share, "dose": dose, "per_final_S_sd_pp": sds, "mde80_pp": mde,
            "expected_effect_GUESS_kakuna_pp": guess,
            "note": "the in-house size of R72's effect = the Kakuna-scale GUESS x the exchange rate (fm6), unmeasured"}


# ================================================================== F-M8: measured rates and cost
def anchor_rates(state: Path) -> dict:
    walls, games = [], 0
    for f in glob.glob(str(state / "anchor/*/*/units.jsonl")):
        for line in Path(f).read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") == "end" and e.get("counted") and e.get("wall_s"):
                walls.append(float(e["wall_s"]))
    sup = [json.loads(x) for x in (state / "anchor_supervisor.jsonl").read_text().splitlines() if x.strip()]
    t0 = min(e["t"] for e in sup if e["event"] == "start")
    t1 = max(e["t"] for e in sup if e["event"] == "done") if any(e["event"] == "done" for e in sup) else None
    conc = max([e.get("concurrency", 1) for e in sup if e["event"] == "start"] or [1])
    w = np.array(walls)
    games = 20 * len(w)
    return {"units_played_here": int(len(w)), "games": games,
            "s_per_game_median": round(float(np.median(w)) / 20, 1) if len(w) else None,
            "s_per_game_mean": round(float(w.mean()) / 20, 1) if len(w) else None,
            "s_per_game_p90": round(float(np.percentile(w, 90)) / 20, 1) if len(w) else None,
            "lane_h_per_1000_games_mean": round(float(w.sum()) / 3600 / games * 1000, 2) if len(w) else None,
            "concurrency": conc,
            "lane_span_h": round((t1 - t0) / 3600, 1) if t1 else None,
            "realised_games_per_wall_h": round(games / ((t1 - t0) / 3600), 1) if t1 else None}


def s_rates(state: Path) -> dict:
    sup = [json.loads(x) for x in (state / "s_supervisor.jsonl").read_text().splitlines() if x.strip()]
    lane_s, per_label, t_start, open_unit = 0.0, {}, None, None
    for e in sup:
        if e["event"] == "unit":
            open_unit = e
        elif e["event"] == "unit_end" and open_unit is not None:
            dur = e["t"] - open_unit["t"]
            ls = dur * len(open_unit["engines"])
            lane_s += ls
            lab = open_unit["engines"][0].split(":")[0]
            per_label[lab] = per_label.get(lab, 0.0) + ls
            open_unit = None
    starts = [e["t"] for e in sup if e["event"] == "start"]
    dones = [e["t"] for e in sup if e["event"] == "done"]
    sets = len(per_label)
    return {"sets_played_here": sets, "lane_h_per_set_mean": round(lane_s / 3600 / sets, 2) if sets else None,
            "lane_h_per_set": {k: round(v / 3600, 2) for k, v in per_label.items()},
            "lane_span_h_incl_pauses": round((max(dones) - min(starts)) / 3600, 1) if dones else None}


def gpu_h() -> dict:
    recs = {}
    for f in sorted(glob.glob(str(MEAS / "x35_breakers_stage_2026-10-17/dose/rb_x35_*.json"))):
        d = json.loads(Path(f).read_text())
        if d.get("last_step", 0) >= 300_000_170 and d["run"].split("_s")[0] in (
                "rb_x35_c0", "rb_x35_d1wide", "rb_x35_d3mut", "rb_x35_n1p"):
            recs[d["run"]] = {"gpu_h": d["gpu_h"], "pin": d["pin"][:9], "restarts": d.get("restarts")}
    g = np.array([v["gpu_h"] for v in recs.values()])
    return {"runs": recs, "n": len(g), "mean": round(float(g.mean()), 3), "min": float(g.min()),
            "max": float(g.max())}


def fm8() -> dict:
    rates = {"X61P look 1 (E3, 46b9e572 -> de15dfc7b)": {"anchor": anchor_rates(ARCH / "x61p_n1p_read/look1"),
                                                          "s": s_rates(ARCH / "x61p_n1p_read/look1")},
             "X61P look 2 (E3, de15dfc7b)": {"anchor": anchor_rates(ARCH / "x61p_n1p_read/look2"),
                                             "s": s_rates(ARCH / "x61p_n1p_read/look2")},
             "D3 look 1 (E2)": {"anchor": anchor_rates(ARCH / "x41_d3_read/look1"),
                                "s": s_rates(ARCH / "x41_d3_read/look1")},
             "D3 look 2 (E1 hybrid)": {"anchor": anchor_rates(ARCH / "x41_d3_read/look2"),
                                       "s": s_rates(ARCH / "x41_d3_read/look2")}}
    g = gpu_h()
    # the per-game and per-set rates the cost uses: X61P's (the registered harness, the registered engine)
    k_s = np.mean([rates[k]["anchor"]["s_per_game_mean"] for k in rates if k.startswith("X61P")])
    s_h = np.mean([rates[k]["s"]["lane_h_per_set_mean"] for k in rates if k.startswith("X61P")])
    kak_hi = max(rates[k]["anchor"]["s_per_game_mean"] for k in rates)
    thr = [rates[k]["anchor"]["realised_games_per_wall_h"] for k in rates if rates[k]["anchor"]["realised_games_per_wall_h"]]
    thr_x61p = [rates[k]["anchor"]["realised_games_per_wall_h"] for k in rates if k.startswith("X61P")]

    def plan(seeds):
        """The games and sets by READ TIER (F-M8 fix 3: a trimmed read still decides). C0 s3501-3503's F and
        reported cells and P nodes import from X61P look 1; s3504-3505's sit in look 2's state, and a group takes
        ONE import, so they are re-played unless a two-state import is built (FINDING). The T - D1 contrast uses
        ON's first five finals (m1: always 5 v 5)."""
        new_c0 = 3 if seeds == 8 else 0
        tiers = {
            "T1 primary F + goal (a) + Delta_P": {
                "kakuna": {"ON F 0-59": 1200 * seeds, "ON home72 0-6 (goal a)": 140 * seeds,
                           "C0 F 0-59, s3504-3505 re-played": 2 * 1200, "C0 F 0-59, new seeds": 1200 * new_c0,
                           "determinism guard (one imported C0 cell re-played)": 20},
                "sets": {"ON P": seeds, "C0 P s3504-3505 re-played": 2, "C0 P new seeds": new_c0}},
            "T2 I_F + R72": {"kakuna": {},
                             "sets": {"ON I_F": seeds, "ON R72": seeds, "C0 I_F": 5 + new_c0, "C0 R72": 5 + new_c0}},
            "T3 reported (H / P72 / home72 at 7 units)": {
                "kakuna": {"ON P72 0-6": 140 * seeds, "C0 H/P72/home72 0-6, s3504-3505 re-played": 2 * 420,
                           "C0 H/P72/home72 0-6, new seeds": 420 * new_c0}, "sets": {}},
            "T4 T - D1 contrast (H + F units 0-29, 5 v 5)": {
                "kakuna": {"ON H 0-29 (s3501-3505)": 5 * 600, "D1 H + F 0-29 re-played at E3": 5 * 2 * 600},
                "sets": {}},
        }
        rows, tot_g, tot_s = {}, 0, 0
        for name, t in tiers.items():
            ng, ns = sum(t["kakuna"].values()), sum(t["sets"].values())
            tot_g += ng
            tot_s += ns
            rows[name] = {**t, "kakuna_games": ng, "inhouse_sets": ns,
                          "kakuna_lane_h": round(ng * k_s / 3600, 1), "inhouse_lane_h": round(ns * s_h, 1),
                          "kakuna_wall_h_at_X61P_throughput": [round(ng / max(thr_x61p), 1),
                                                               round(ng / min(thr_x61p), 1)]}
        gpu = {"ON look 1 (3 seeds)": round(3 * g["mean"], 1), "ON look 2": round((seeds - 3) * g["mean"], 1),
               "new C0 seeds": round(new_c0 * g["mean"], 1), "smoke": 0.2}
        return {"tiers": rows, "kakuna_total": tot_g, "inhouse_total": tot_s,
                "cpu_lane_h": {"kakuna_at_X61P_rate": round(tot_g * k_s / 3600, 1),
                               "kakuna_at_slowest_measured": round(tot_g * kak_hi / 3600, 1),
                               "inhouse_at_X61P_rate": round(tot_s * s_h, 1)},
                "wall_h_kakuna_at_realised_throughput": [round(tot_g / max(thr), 1), round(tot_g / min(thr), 1)],
                "wall_h_kakuna_at_X61P_throughput": [round(tot_g / max(thr_x61p), 1), round(tot_g / min(thr_x61p), 1)],
                "wall_h_inhouse_two_lanes_at_X61P_rate": round(tot_s * s_h / 2, 1),
                "gpu_h": {**gpu, "total": round(sum(gpu.values()), 1)},
                "look1_futility_stop_saves": {
                    "gpu_h": round((seeds - 3) * g["mean"] + new_c0 * g["mean"], 1),
                    "note": "a look-1 stop plays ON's three finals only (F, home72, P72, I_F, R72, P) and no look-2 "
                            "C0 re-play; T - D1 is read only at 5 v 5, so it is not played after a look-1 stop "
                            "(see PREREG §9)"}}

    return {"rates": rates, "gpu_h_records": g, "kakuna_s_per_game_used": round(float(k_s), 1),
            "kakuna_s_per_game_slowest_state": kak_hi, "inhouse_lane_h_per_set_used": round(float(s_h), 2),
            "plan_5_seeds": plan(5), "plan_8_seeds": plan(8)}


# ================================================================== F-M1 / F-M2: the freeze and the harness
def fm1() -> dict:
    blob = subprocess.run(["git", "-C", str(REPO), "show", "e1512c361:designs/endstate/design_human_replay_lever.md"],
                          capture_output=True, check=True).stdout
    txt = blob.decode()
    i, j = txt.index("## 6. X60"), txt.index("## 7. Appendix")
    head = (REPO / "designs/endstate/design_human_replay_lever.md").read_text()
    hi, hj = head.index("## 6. X60"), head.index("## 7. Appendix")
    return {"file_sha256_at_e1512c361": hashlib.sha256(blob).hexdigest(),
            "section6_sha256_at_e1512c361": hashlib.sha256(txt[i:j].encode()).hexdigest(),
            "section6_sha256_at_HEAD": hashlib.sha256(head[hi:hj].encode()).hexdigest(),
            "section6_unchanged_since": txt[i:j] == head[hi:hj]}


def fm2() -> dict:
    out = {"play_commit": "de15dfc7b6d19d6f0e4a3724f269e26e7aa22957"}
    try:
        from main.reads import engine as EN
        out["engine_at_play_commit"] = EN.engine_at("de15dfc7b6d19d6f0e4a3724f269e26e7aa22957")
        out["engine_E3_matches_X61P_cells"] = out["engine_at_play_commit"] == x61p_cells()["engine"]
    except Exception as exc:          # the check is reported, never assumed
        out["engine_at_play_commit"] = f"UNCHECKED: {type(exc).__name__}: {exc}"
    l1 = sorted(os.listdir(ARCH / "x61p_n1p_read/look1/anchor/primary"))
    l2 = sorted(os.listdir(ARCH / "x61p_n1p_read/look2/anchor/primary"))
    out["C0_F_cells_played_in"] = {"look1": [x for x in l1 if x.startswith("c0")],
                                   "look2": [x for x in l2 if x.startswith("c0")]}
    out["look1_spec_sha256"] = _sha256_file(ARCH / "x61p_n1p_read/look1/spec.json")
    out["look2_spec_sha256"] = _sha256_file(ARCH / "x61p_n1p_read/look2/spec.json")
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    r = {"seed": SEED, "reps": REPS, "power_py": str(MEAS / "x60_redesign_2026-10-26/power.py"),
         "power_py_sha256": _sha256_file(MEAS / "x60_redesign_2026-10-26/power.py")}
    r["fm4"] = fm4(rng)
    seeds = 8 if r["fm4"]["fix3_decision"].startswith("8") else 5
    r["fm6"] = fm6(rng, seeds)
    r["fm5"] = fm5()
    r["fm8"] = fm8()
    r["fm1"] = fm1()
    r["fm2"] = fm2()
    s = json.dumps(r, indent=1, default=str)
    print(s)
    if "--write" in sys.argv:
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / "analysis.json").write_text(s + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
