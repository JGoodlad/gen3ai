"""L1, the POLICY-ANCHORED acting rules on the banked honest rows: FIT on Tier 1, READ held out on E2 and X26
(`../PREREG.md` §3-§5). Pure numpy over the cache `l1l5_data.py build` writes; no model, no engine, no playout.

    python -I l1_read.py selfcheck    # the rules' unit checks on synthetic states (no bank read)
    python -I l1_read.py read         # → ../result_l1.json (+ the verdict lines on stdout)

THE RULES (one parameter each; every rule picks among the legal actions in the POLICY's order, ties within TIE = 1e-6 to
the earliest, the registered tie rule). Q(a) is the scorer's per-action value over the honest worlds (LEAF-K8: the mean
of the 8 world outcomes; TRUNC-4-K8: the mean of the 8 4-turn values; CRITIC: the mean of the 8 one-ply critic values),
log π(a) the policy's masked log-softmax at the root:
  ARGMAX         the registered rule (the module's W4): argmax Q
  PIKL(λ)        argmax Q(a) + λ·log π(a) — the MODE of π'(a) ∝ π(a)·exp(Q(a)/λ) (piKL; Ataraxos's reverse-KL step to
                 the move network at α = 0); λ = 0 is ARGMAX, λ = ∞ the policy's own top choice
  GUMBEL(c)      argmax logit(a) + (c_visit + N_max)·c·q̃(a), q̃ = Q min-max rescaled over the legal actions,
                 c_visit 50, N_max 8 (`main.value_search.rules.gumbel_scores`, no Gumbel noise); c = 1 is the
                 module's default, c = 0 the policy's top choice
  THRESH(δ)      play ARGMAX's pick a* only if Q(a*) − Q(top1) > δ (SPARTA-style deviation), else top1; δ = 0 is ARGMAX
IMP = T(the rule's pick) − T(the policy's top1), T the referee (the true world's seeds 16..39), the honest-search gain
metric of every bank.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "result_l1.json"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1] / "hidden_info_tier1_2026-10-13"))
import t1_stats as ST  # noqa: E402  (the study's bootstrap: 2,000 draws, seed 20261011, battle-clustered)

TIE = 1e-6
EPS = 1e-9
INF = math.inf
C_VISIT, N_MAX = 50.0, 8.0
PRIMARY = ("LEAF-K8", "TRUNC-4-K8")
SCORERS = ("LEAF-K8", "TRUNC-4-K8", "CRITIC")
KEY = {"LEAF-K8": "W", "TRUNC-4-K8": "X4", "CRITIC": "crit"}
RULES = ("PIKL", "GUMBEL", "THRESH")
#: each grid runs from the LEAST to the MOST anchored value (the fit's tie-break takes the later entry)
GRIDS: Dict[str, List[float]] = {
    "PIKL": [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5,
             1.0, INF],
    "GUMBEL": [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0],
    "THRESH": [0.0, 0.005, 0.01, 0.02, 1 / 32, 3 / 32, 5 / 32, 7 / 32, 9 / 32, 11 / 32, 13 / 32, 15 / 32, 19 / 32,
               23 / 32, INF],
}
LEVEL_VERDICT = 100.0 - 5.0 / 3.0          # Bonferroni over the three rules (PREREG §4.2)
EDGE = 0.0005
MATERIAL = 0.005


# ------------------------------------------------------------------------------------------------
# the rules (pure)
# ------------------------------------------------------------------------------------------------

def first_max(s: np.ndarray) -> int:
    m = float(np.max(s))
    return int(np.argmax(s >= m - TIE))


def log_softmax(logit: np.ndarray) -> np.ndarray:
    z = logit - logit.max()
    return z - math.log(float(np.exp(z).sum()))


def rescale(q: np.ndarray) -> np.ndarray:
    lo, hi = float(q.min()), float(q.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(q)
    return (q - lo) / (hi - lo)


def pick(rule: str, x: float, q: np.ndarray, logit: np.ndarray) -> int:
    """The index (policy order; 0 = the policy's top1) the rule plays."""
    if rule == "ARGMAX":
        return first_max(q)
    if rule == "PIKL":
        if x == INF:
            return 0
        return first_max(q + x * log_softmax(logit))
    if rule == "GUMBEL":
        return first_max(logit + (C_VISIT + N_MAX) * x * rescale(q))
    if rule == "THRESH":
        if x == INF:
            return 0
        a = first_max(q)
        return a if q[a] - q[0] > x + EPS else 0
    raise ValueError(rule)


def pikl_sampled(lam: float, q: np.ndarray, logit: np.ndarray) -> np.ndarray:
    """π'(a) ∝ π(a)·exp(Q(a)/λ) (the SAMPLED piKL policy; λ = ∞ is π itself, λ = 0 the uniform argmax set)."""
    lp = log_softmax(logit)
    if lam == INF:
        z = lp
    elif lam == 0.0:
        z = np.where(q >= q.max() - TIE, 0.0, -np.inf)
    else:
        z = lp + q / lam
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


# ------------------------------------------------------------------------------------------------
# frames
# ------------------------------------------------------------------------------------------------

class Frame:
    """States of one read with their bootstrap (battle-clustered; strata = the checkpoint for Tier 1 / E2)."""

    def __init__(self, recs: List[dict], strata: bool):
        self.recs = recs
        self.n = len(recs)
        self.bt = ST.Boot([r["battle"] for r in recs], [r["stratum_bt"] for r in recs] if strata else None)

    def imp(self, src: str, z: str, rule: str, x: float) -> np.ndarray:
        out = np.full(self.n, np.nan)
        for i, r in enumerate(self.recs):
            s = r.get(src)
            if s is None:
                continue
            q = np.asarray(s[KEY[z]], dtype=np.float64).mean(1)
            out[i] = r["Tp"][pick(rule, x, q, r["logit"])] - r["Tp"][0]
        return out

    def imp_sampled(self, src: str, z: str, lam: float) -> np.ndarray:
        out = np.full(self.n, np.nan)
        for i, r in enumerate(self.recs):
            s = r.get(src)
            if s is None:
                continue
            q = np.asarray(s[KEY[z]], dtype=np.float64).mean(1)
            out[i] = float(pikl_sampled(lam, q, r["logit"]) @ r["Tp"]) - r["Tp"][0]
        return out


def summ(bt, vec, sel=None) -> dict:
    est, v = bt.mean(vec, sel)
    lo95, hi95 = ST.ci(v, 95.0)
    lov, hiv = ST.ci(v, LEVEL_VERDICT)
    m = np.isfinite(vec) if sel is None else (np.isfinite(vec) & sel)
    return {"mean": round(est, 6), "ci95": [_r(lo95), _r(hi95)], "ci_verdict": [_r(lov), _r(hiv)], "n": int(m.sum())}


def _r(x):
    return None if x is None else round(float(x), 6)


def gx(x: float):
    return "inf" if x == INF else x


# ------------------------------------------------------------------------------------------------
# the fit and the read
# ------------------------------------------------------------------------------------------------

def fit(F: Frame, src: str, z: str, rule: str) -> Tuple[float, List[dict]]:
    """The grid value maximising the pooled mean IMP (states weighted equally); a tie within 1e-12 → the MORE anchored
    value (the later grid entry)."""
    curve = []
    best, bx = -INF, None
    for x in GRIDS[rule]:
        v = F.imp(src, z, rule, x)
        mu = float(np.nanmean(v))
        curve.append({"x": gx(x), "mean": round(mu, 6), "n": int(np.isfinite(v).sum())})
        if mu >= best - 1e-12:
            best, bx = max(mu, best), x
    return bx, curve


def read_rule(H: Frame, src: str, z: str, rule: str, x: float) -> dict:
    v = H.imp(src, z, rule, x)
    a = H.imp(src, z, "ARGMAX", 0.0)
    return {"x": gx(x), "IMP": summ(H.bt, v), "minus_ARGMAX": summ(H.bt, v - a),
            "deviates_from_top1_share": round(float(np.nanmean(_dev(H, src, z, rule, x))), 4)}


def _dev(H: Frame, src: str, z: str, rule: str, x: float) -> np.ndarray:
    out = np.full(H.n, np.nan)
    for i, r in enumerate(H.recs):
        s = r.get(src)
        if s is not None:
            out[i] = float(pick(rule, x, np.asarray(s[KEY[z]], dtype=np.float64).mean(1), r["logit"]) != 0)
    return out


def frames(recs: List[dict]) -> Dict[str, Frame]:
    for r in recs:
        r["stratum_bt"] = r["cp"]
    t1 = [r for r in recs if r["bank"] == "tier1"]
    e2 = [r for r in recs if r["bank"] == "e2"]
    x26 = [r for r in recs if r["bank"] == "x26"]
    out = {"tier1": Frame(t1, True), "e2": Frame(e2, True), "x26": Frame(x26, False)}
    for cp in ("end2", "prod"):
        out[f"e2:{cp}"] = Frame([r for r in e2 if r["cp"] == cp], False)
        out[f"tier1:{cp}"] = Frame([r for r in t1 if r["cp"] == cp], False)
    for lab in sorted({r["cp"] for r in x26}):
        out[f"x26:{lab}"] = Frame([r for r in x26 if r["cp"] == lab], False)
    return out


HELD = ("e2", "x26")


def verdict(primary: Dict[str, Dict[str, Dict[str, dict]]]) -> dict:
    """PREREG §4.3. ``primary[rule][bank][scorer]`` = the held-out summary at the FITTED value."""
    pays = {R: {b: all(ST.gt(primary[R][b][z]["IMP"]["ci_verdict"][0], EDGE) for z in PRIMARY) for b in HELD}
            for R in RULES}
    cells_clear = [f"{R}|{b}|{z}" for R in RULES for b in HELD for z in PRIMARY
                   if ST.gt(primary[R][b][z]["IMP"]["ci_verdict"][0], EDGE)]
    bounded = all(ST.lt(primary[R][b][z]["IMP"]["ci_verdict"][1], MATERIAL) for R in RULES for b in HELD for z in PRIMARY)
    winners = [R for R in RULES if all(pays[R][b] for b in HELD)]
    if winners:
        v = "PAYS"
    elif cells_clear:
        v = "PAYS-SCOPED"
    elif bounded:
        v = "NULL"
    else:
        v = "NOT DETECTED"
    return {"L1": v, "rules_paying_on_both_banks": winners, "pays_per_rule_bank": pays, "clear_cells": cells_clear,
            "every_upper_below_material": bounded}


def read() -> dict:
    import l1l5_data as L

    cache = L.load_cache()
    gigo = gates(json.loads((L.CACHE / "cache.json").read_text()))
    fr = frames(cache["recs"])
    out: dict = {"status": "L1 READ (PREREG §3-§5)", "grids": {R: [gx(x) for x in g] for R, g in GRIDS.items()},
                 "level_verdict": LEVEL_VERDICT, "n": {k: f.n for k, f in fr.items()}}
    # §3: the fit on Tier 1 (arm F), per rule and scorer
    fitted: Dict[str, Dict[str, float]] = {}
    out["fit"] = {}
    for R in RULES:
        fitted[R] = {}
        out["fit"][R] = {}
        for z in SCORERS:
            x, curve = fit(fr["tier1"], "F", z, R)
            fitted[R][z] = x
            out["fit"][R][z] = {"fitted": gx(x), "tier1_curve": curve}
    (HERE.parent / "fit_l1.json").write_text(json.dumps(out["fit"], indent=1, sort_keys=True) + "\n")   # §3.1: first
    out["gigo"] = gigo
    # §4: the PRIMARY held-out read
    primary = {R: {b: {z: read_rule(fr[b], "F", z, R, fitted[R][z]) for z in SCORERS} for b in HELD} for R in RULES}
    out["primary"] = primary
    out["ARGMAX_heldout"] = {b: {z: summ(fr[b].bt, fr[b].imp("F", z, "ARGMAX", 0.0)) for z in SCORERS} for b in HELD}
    out["verdict"] = verdict(primary)
    # §5: SECONDARIES (descriptive, no verdict)
    sec: dict = {}
    sec["per_checkpoint"] = {k: {R: {z: read_rule(f, "F", z, R, fitted[R][z])["IMP"] for z in PRIMARY} for R in RULES}
                             for k, f in fr.items() if ":" in k}
    sec["heldout_curves (an upper envelope, NOT a verdict)"] = {
        b: {R: {z: [{"x": gx(x), **summ(fr[b].bt, fr[b].imp("F", z, R, x))} for x in GRIDS[R]] for z in PRIMARY}
            for R in RULES} for b in HELD}
    sec["anchored_rollout_minus_anchored_critic"] = {
        b: {R: {z: summ(fr[b].bt, fr[b].imp("F", z, R, fitted[R][z]) - fr[b].imp("F", "CRITIC", R, fitted[R]["CRITIC"]))
                for z in PRIMARY} for R in RULES} for b in HELD}
    by = {}
    for b in HELD:
        f = fr[b]
        cells = {f"k {k}": np.array([(r["k"] if r["k"] <= 5 else 6) == k for r in f.recs]) for k in (1, 2, 3, 4, 5, 6)}
        cells.update({f"opp {o}": np.array([r["opp"] == o for r in f.recs]) for o in sorted({r["opp"] for r in f.recs})})
        by[b] = {c: {R: {z: summ(f.bt, f.imp("F", z, R, fitted[R][z]), m) for z in PRIMARY} for R in RULES}
                 for c, m in cells.items() if m.sum() >= 30}
    sec["by_cell"] = by
    samp = {}
    for z in PRIMARY:
        x, curve = _fit_sampled(fr["tier1"], z)
        samp[z] = {"fitted": gx(x), "tier1_curve": curve,
                   "heldout": {b: summ(fr[b].bt, fr[b].imp_sampled("F", z, x)) for b in HELD},
                   "policy_sampled_heldout": {b: summ(fr[b].bt, fr[b].imp_sampled("F", z, INF)) for b in HELD}}
    sec["pikl_sampled"] = samp
    orc = {}
    for R in RULES:
        orc[R] = {}
        for z in PRIMARY:
            x, curve = fit(fr["tier1"], "O", z, R)
            orc[R][z] = {"fitted_on_O": gx(x), "tier1_O_curve": curve,
                         "heldout_O_at_O_fit": {b: summ(fr[b].bt, fr[b].imp("O", z, R, x)) for b in HELD},
                         "heldout_O_at_F_fit": {b: summ(fr[b].bt, fr[b].imp("O", z, R, fitted[R][z])) for b in HELD},
                         "heldout_O_ARGMAX": {b: summ(fr[b].bt, fr[b].imp("O", z, "ARGMAX", 0.0)) for b in HELD}}
    sec["oracle_world"] = orc
    fn = {}
    for R in RULES:
        fn[R] = {}
        for z in PRIMARY:
            x, _c = fit(fr["tier1"], "FN", z, R)
            fn[R][z] = {"fitted_on_FN": gx(x), "x26_FN_at_FN_fit": summ(fr["x26"].bt, fr["x26"].imp("FN", z, R, x)),
                        "x26_FN_at_F_fit": summ(fr["x26"].bt, fr["x26"].imp("FN", z, R, fitted[R][z])),
                        "x26_FN_ARGMAX": summ(fr["x26"].bt, fr["x26"].imp("FN", z, "ARGMAX", 0.0))}
    sec["FN_network_belief_worlds"] = fn
    out["secondary (descriptive)"] = sec
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def gates(log: dict) -> dict:
    """PREREG §1 G1 / G3 (G2 raised inside the loader): refuse the read on any failure."""
    bad = []
    for cp, n in log["tier1"]["per_cp"].items():
        if n["argmax_mismatch"]:
            bad.append(f"tier1 {cp}: {n['argmax_mismatch']} ARGMAX mismatches")
    for lab, n in log["x26"]["per_anchor"].items():
        if n["argmax_mismatch"]:
            bad.append(f"x26 {lab}: {n['argmax_mismatch']} ARGMAX mismatches")
    for z, c in log["e2"]["crosscheck"].items():
        if c["mine_n"] != c["registered"]["n"] or abs(c["mine_mean"] - c["registered"]["mean"]) > 5e-6:
            bad.append(f"e2 {z}: {c['mine_mean']} n {c['mine_n']} vs registered {c['registered']}")
    want = {"tier1": 1477, "e2": 1108, "x26": 1148}
    if log["n"] != want:
        bad.append(f"counts {log['n']} != {want}")
    if bad:
        raise SystemExit("GIGO gate FAILED: " + "; ".join(bad))
    return {"G1_G3": "PASS", "counts": log["n"], "cache_sha256": log["cache_sha256"]}


def _fit_sampled(F: Frame, z: str) -> Tuple[float, List[dict]]:
    curve, best, bx = [], -INF, None
    for x in GRIDS["PIKL"]:
        mu = float(np.nanmean(F.imp_sampled("F", z, x)))
        curve.append({"x": gx(x), "mean": round(mu, 6)})
        if mu >= best - 1e-12:
            best, bx = max(mu, best), x
    return bx, curve


# ------------------------------------------------------------------------------------------------
# selfcheck (synthetic; no bank)
# ------------------------------------------------------------------------------------------------

def selfcheck() -> None:
    sys.path.insert(0, str(HERE.parents[4] / "src"))          # this checkout's search module (``-I`` drops PYTHONPATH)
    from main.value_search import rules as VR

    rng = np.random.default_rng(7)
    for _ in range(2000):
        A = int(rng.integers(2, 10))
        logit = np.sort(rng.normal(0, 2, A))[::-1]
        q = rng.integers(0, 9, A) / 8.0 if rng.random() < 0.5 else rng.random(A)
        # the limits
        assert pick("PIKL", 0.0, q, logit) == pick("ARGMAX", 0, q, logit) == pick("THRESH", 0.0, q, logit)
        assert pick("PIKL", INF, q, logit) == 0 and pick("THRESH", INF, q, logit) == 0
        assert pick("GUMBEL", 0.0, q, logit) == 0 or abs(logit[pick("GUMBEL", 0.0, q, logit)] - logit[0]) <= TIE
        # GUMBEL matches the search module's own scores
        g = VR.gumbel_scores(logit, {i: float(q[i]) for i in range(A)}, N_MAX)
        mine = logit + (C_VISIT + N_MAX) * 0.1 * rescale(q)
        g01 = {a: float(logit[a]) + VR.sigma(VR.rescale({i: float(q[i]) for i in range(A)})[a], N_MAX, c_scale=0.1)
               for a in range(A)}
        assert np.allclose(mine, [g01[a] for a in range(A)])
        assert np.allclose(logit + (C_VISIT + N_MAX) * 1.0 * rescale(q), [g[a] for a in range(A)])
        # PIKL is the mode of the sampled π'
        lam = float(rng.choice([0.01, 0.05, 0.2]))
        p = pikl_sampled(lam, q, logit)
        assert p[pick("PIKL", lam, q, logit)] >= p.max() - 1e-9
        # monotone anchoring: THRESH never deviates at a larger δ where it did not at a smaller one
        dev = [pick("THRESH", d, q, logit) != 0 for d in GRIDS["THRESH"]]
        assert all(not dev[i + 1] or dev[i] for i in range(len(dev) - 1))
    # the verdict mapping on toy summaries
    def cell(lo, hi):
        return {"IMP": {"ci_verdict": [lo, hi]}}
    allnull = {R: {b: {z: cell(-0.004, 0.004) for z in SCORERS} for b in HELD} for R in RULES}
    assert verdict(allnull)["L1"] == "NULL"
    wide = {R: {b: {z: cell(-0.004, 0.02) for z in SCORERS} for b in HELD} for R in RULES}
    assert verdict(wide)["L1"] == "NOT DETECTED"
    one = json.loads(json.dumps(wide))
    one["PIKL"]["e2"]["LEAF-K8"] = cell(0.002, 0.02)
    assert verdict(one)["L1"] == "PAYS-SCOPED"
    for b in HELD:
        for z in PRIMARY:
            one["PIKL"][b][z] = cell(0.002, 0.02)
    assert verdict(one)["L1"] == "PAYS"
    edge = json.loads(json.dumps(wide))
    for b in HELD:
        for z in PRIMARY:
            edge["GUMBEL"][b][z] = cell(0.0005, 0.02)         # ON the boundary: not clear (rule 8)
    assert verdict(edge)["L1"] == "NOT DETECTED"
    print("selfcheck OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["selfcheck"]:
        selfcheck()
    elif sys.argv[1:] == ["read"]:
        res = read()
        print(json.dumps({"fitted": {R: {z: res["fit"][R][z]["fitted"] for z in SCORERS} for R in RULES},
                          "verdict": res["verdict"]}, indent=1))
    else:
        raise SystemExit("usage: l1_read.py selfcheck|read")
