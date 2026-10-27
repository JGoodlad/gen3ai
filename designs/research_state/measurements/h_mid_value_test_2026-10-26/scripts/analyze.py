"""H-MID: the critic's calibration, the policy given the value, and the critic's discrimination, home72 vs U.

Usage (no model, no torch; any interpreter with numpy):
    python analyze.py --games GAMES.jsonl.gz --recon kakuna=K.jsonl.gz [--recon pm=P.jsonl.gz] --out results.json

``GAMES`` = the piloting-gap forensics' per-game rows (D(t), the result, the cell); each ``--recon`` = the output of
``reconstruct.py`` for one read. Everything here is fixed by PREREG.md (§3 the fidelity gate, §4 states and weights,
§5 measures, §6 the verdict); the verdict is evaluated once, on the Kakuna read.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict

import numpy as np

B = 2000
SEED = 20261026
TS = [1, 5, 10, 15, 20, 30]
V_EDGES = [0.2, 0.4, 0.6, 0.8]
EPS = 1e-6
BOUNDARY = 0.001
READ_CELLS = {
    "kakuna": ("x51", {"pstar_home72": "home72", "pstar_hpfix": "H", "pstar_forum72": "F", "pstar_p72": "P72"}),
    "pm": ("mm", {"pm_home72": "home72", "pm_hpfix": "H", "pm_forum72": "F", "pm_p72": "P72"}),
}


def is_level(d: float) -> bool:
    """The forensics' ``conditional`` level: within one mon (-1 < D < 1), the population of its 0.609 / 0.501."""
    return -1.0 < d < 1.0


def state_at(g: dict, t: int) -> float:
    """The forensics' material lead at the start of turn t (``analyze.state_at``, copied)."""
    best = None
    for s in g["snaps"]:
        if s[0] <= t:
            best = s
        else:
            break
    if best is None or g["turns"] < t:
        best = g["final"]
    return best[1] - best[3]


# ------------------------------------------------------------------------------------------------ loading
def load(games_path: str, recon: dict) -> dict:
    fz = {}
    with gzip.open(games_path, "rt") as fh:
        for line in fh:
            g = json.loads(line)
            fz[(g["read"], g["unit"], g["tag"])] = g
    reads = {}
    for name, path in recon.items():
        read_key, cells = READ_CELLS[name]
        with gzip.open(path, "rt") as fh:
            rows = [json.loads(x) for x in fh]
        fid = {"games": len(rows), "excluded": 0, "excluded_detail": [], "decisions": 0, "argmax_match": 0,
               "disagree_small_margin": 0, "protocol_identical": 0, "frames_ok": 0, "ties": 0, "result_mismatch": 0}
        games = []
        for r in rows:
            fid["protocol_identical"] += int(bool(r.get("protocol_identical")))
            fid["frames_ok"] += int(bool(r.get("frames_ok")))
            fid["decisions"] += r.get("n_dec", 0)
            fid["argmax_match"] += r.get("n_match", 0)
            fid["disagree_small_margin"] += r.get("n_dec", 0) - r.get("n_match", 0) - r.get("n_bad_disagree", 0)
            if not r.get("ok"):
                fid["excluded"] += 1
                fid["excluded_detail"].append({k: r.get(k) for k in ("unit", "tag", "err", "protocol_identical",
                                                                      "frames_ok", "n_bad_disagree")})
                continue
            g = fz.get((read_key, r["unit"], r["tag"]))
            if g is None:
                raise SystemExit(f"GIGO: no forensics row for {read_key}/{r['unit']}/{r['tag']}")
            if r["result"] is None or abs(float(g["result"]) - float(r["result"])) > 1e-9:
                fid["result_mismatch"] += 1
                raise SystemExit(f"GIGO: result differs for {r['unit']}/{r['tag']}: forensics {g['result']} "
                                 f"replay {r['result']}")
            cell = r["unit"].rsplit("_", 1)[0]
            g2 = {"unit": r["unit"], "panel": cells[cell], "y": float(r["result"]), "turns": g["turns"],
                  "snaps": g["snaps"], "final": g["final"], "dec": r["decisions"]}
            fid["ties"] += int(g2["y"] == 0.5)
            games.append(g2)
        n = fid["games"]
        fid["argmax_agreement"] = fid["argmax_match"] / max(1, fid["decisions"])
        fid["excluded_share"] = fid["excluded"] / max(1, n)
        fid["VOID"] = bool(fid["excluded_share"] > 0.01 or fid["argmax_agreement"] < 0.995)
        reads[name] = {"fidelity": fid, "games": games}
    return reads


def v_at(g: dict, t: int):
    """V at P*'s first decision with turn >= t, in a game that reached t; None otherwise."""
    if g["turns"] < t:
        return None
    for d in g["dec"]:
        if d[1] >= t:
            return d[2]
    return None


# ------------------------------------------------------------------------------------------------ groups
class Group:
    """One state set: per-state V, y, weight (U's panels equal), and units for the within-cell bootstrap."""

    def __init__(self, v, y, panel, unit):
        self.v, self.y = np.asarray(v, float), np.asarray(y, float)
        self.panel = np.asarray(panel)
        by_panel = defaultdict(int)
        for p in panel:
            by_panel[p] += 1
        k = len(by_panel)
        self.w = np.array([1.0 / (k * by_panel[p]) for p in panel]) if len(panel) else np.zeros(0)
        units = defaultdict(list)
        for i, (p, u) in enumerate(zip(panel, unit)):
            units[(p, u)].append(i)
        self.cells = defaultdict(list)
        for (p, _u), idx in units.items():
            self.cells[p].append(np.array(idx, dtype=int))

    def resample(self, rng) -> np.ndarray:
        out = []
        for units in self.cells.values():
            pick = rng.integers(0, len(units), len(units))
            out.extend(units[j] for j in pick)
        return np.concatenate(out) if out else np.zeros(0, dtype=int)


def wauc(v, y, w) -> float:
    """Weighted AUC of v for y (ties in v count 1/2; y = 0.5 is dropped)."""
    keep = y != 0.5
    v, y, w = v[keep], y[keep], w[keep]
    if not len(v) or y.min() == y.max():
        return float("nan")
    uniq, inv = np.unique(v, return_inverse=True)
    posw = np.bincount(inv, weights=w * (y == 1), minlength=len(uniq))
    negw = np.bincount(inv, weights=w * (y == 0), minlength=len(uniq))
    below = np.cumsum(negw) - negw
    return float(np.sum(posw * (below + 0.5 * negw)) / (posw.sum() * negw.sum()))


def set_stats(grp: Group, idx: np.ndarray) -> dict:
    v, y, w, panel = grp.v[idx], grp.y[idx], grp.w[idx], grp.panel[idx]
    sw = w.sum()
    W = float(np.sum(w * y) / sw)
    Vb = float(np.sum(w * v) / sw)
    vc = np.clip(v, EPS, 1 - EPS)
    brier = float(np.sum(w * (v - y) ** 2) / sw)
    ll = float(np.sum(w * -(y * np.log(vc) + (1 - y) * np.log(1 - vc))) / sw)
    aucs = [wauc(v[panel == p], y[panel == p], w[panel == p]) for p in sorted(set(panel.tolist()))]
    aucs = [a for a in aucs if not math.isnan(a)]
    b = np.searchsorted(V_EDGES, v, side="right")
    bins = []
    for k in range(len(V_EDGES) + 1):
        m = b == k
        bins.append((float(w[m].sum() / sw), float(np.sum(w[m] * y[m]) / w[m].sum()) if m.any() else float("nan"),
                     float(np.sum(w[m] * v[m]) / w[m].sum()) if m.any() else float("nan")))
    hi = v >= 0.6
    return {"W": W, "V": Vb, "bias": Vb - W, "brier": brier, "logloss": ll,
            "auc": float(np.mean(aucs)) if aucs else float("nan"),
            "W_at_V_ge_0.6": float(np.sum(w[hi] * y[hi]) / w[hi].sum()) if hi.any() else float("nan"),
            "share_V_ge_0.6": float(w[hi].sum() / sw), "_bins": bins}


def contrast(sh: dict, su: dict) -> dict:
    out = {"G_y": sh["W"] - su["W"], "G_V": sh["V"] - su["V"], "d_bias": su["bias"] - sh["bias"],
           "d_brier": su["brier"] - sh["brier"], "d_logloss": su["logloss"] - sh["logloss"],
           "d_auc_h_minus_U": sh["auc"] - su["auc"],
           "d_W_at_V_ge_0.6": sh["W_at_V_ge_0.6"] - su["W_at_V_ge_0.6"]}
    num = den = dropped = 0.0
    for (wh, yh, _), (_wu, yu, _) in zip(sh["_bins"], su["_bins"]):
        if math.isnan(yh) or math.isnan(yu):
            dropped += wh
            continue
        num += wh * (yh - yu)
        den += wh
    out["matched_V_gap"] = num / den if den else float("nan")
    out["matched_V_dropped_share"] = dropped
    return out


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        out[prefix + k] = v
    return out


def compare(gh: Group, gu: Group, rng) -> dict:
    """Point + 95 % percentile interval for every set statistic and contrast (both groups resampled)."""
    ih, iu = np.arange(len(gh.v)), np.arange(len(gu.v))
    sh, su = set_stats(gh, ih), set_stats(gu, iu)
    point = {**flat(sh, "home72."), **flat(su, "U."), **contrast(sh, su)}
    reps = defaultdict(list)
    for _ in range(B):
        xh, xu = gh.resample(rng), gu.resample(rng)
        a, b = set_stats(gh, xh), set_stats(gu, xu)
        for k, v in {**flat(a, "home72."), **flat(b, "U."), **contrast(a, b)}.items():
            reps[k].append(v)
    out = {}
    for k, p in point.items():
        r = np.array([x for x in reps[k] if not math.isnan(x)])
        lo, hi = (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))) if len(r) else (float("nan"),) * 2
        out[k] = [round(p, 4), round(lo, 4), round(hi, 4)]
    out["n_home72"], out["n_U"] = len(gh.v), len(gu.v)
    out["reliability"] = {"edges": V_EDGES,
                          "home72": [[round(x, 4) for x in b] for b in sh["_bins"]],
                          "U": [[round(x, 4) for x in b] for b in su["_bins"]],
                          "cols": ["share", "W", "mean V"]}
    return out


def groups_at(games: list, t: int, level_only: bool):
    sets = {"home72": ([], [], [], []), "U": ([], [], [], [])}
    for g in games:
        v = v_at(g, t)
        if v is None:
            continue
        if level_only and not is_level(state_at(g, t)):
            continue
        s = sets["home72" if g["panel"] == "home72" else "U"]
        s[0].append(v), s[1].append(g["y"]), s[2].append(g["panel"]), s[3].append(g["unit"])
    return Group(*sets["home72"]), Group(*sets["U"])


def drift_groups(games: list, t0: int = 10, t1: int = 20):
    """V(min(t1, T)) - V(t0) with V(T) = y, on level-at-t0 states."""
    sets = {"home72": ([], [], [], []), "U": ([], [], [], [])}
    for g in games:
        v0 = v_at(g, t0)
        if v0 is None or not is_level(state_at(g, t0)):
            continue
        v1 = v_at(g, t1) if g["turns"] >= t1 else g["y"]
        if v1 is None:
            continue
        s = sets["home72" if g["panel"] == "home72" else "U"]
        s[0].append(v1 - v0), s[1].append(g["y"]), s[2].append(g["panel"]), s[3].append(g["unit"])
    return Group(*sets["home72"]), Group(*sets["U"])


def drift_compare(gh: Group, gu: Group, rng) -> dict:
    def m(grp, idx):
        return float(np.sum(grp.w[idx] * grp.v[idx]) / grp.w[idx].sum())
    ph, pu = m(gh, np.arange(len(gh.v))), m(gu, np.arange(len(gu.v)))
    rh, ru, rd = [], [], []
    for _ in range(B):
        xh, xu = gh.resample(rng), gu.resample(rng)
        a, b = m(gh, xh), m(gu, xu)
        rh.append(a), ru.append(b), rd.append(b - a)

    def ci(p, r):
        return [round(p, 4), round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]
    return {"home72": ci(ph, rh), "U": ci(pu, ru), "U_minus_home72": ci(pu - ph, rd)}


def all_decisions_auc(games: list, rng) -> dict:
    """AUC over every decision, each game weighted 1 (each decision 1 / n_dec), per panel, U = mean of three."""
    per = {}
    for g in games:
        per.setdefault(g["panel"], []).append(g)

    def auc_of(gs):
        v = np.concatenate([[d[2] for d in g["dec"]] for g in gs])
        y = np.concatenate([[g["y"]] * len(g["dec"]) for g in gs])
        w = np.concatenate([[1.0 / len(g["dec"])] * len(g["dec"]) for g in gs])
        return wauc(v, y, w)

    def units(gs):
        u = defaultdict(list)
        for g in gs:
            u[g["unit"]].append(g)
        return list(u.values())

    pu = {p: units(gs) for p, gs in per.items()}

    def stat(sample):
        h = auc_of(sample["home72"])
        uu = float(np.mean([auc_of(sample[p]) for p in ("H", "F", "P72")]))
        return h, uu, h - uu
    pt = stat(per)
    reps = []
    for _ in range(B // 4):            # the all-decision AUC is the costliest statistic: B / 4 replicates
        s = {p: [g for j in rng.integers(0, len(us), len(us)) for g in us[j]] for p, us in pu.items()}
        reps.append(stat(s))
    reps = np.array(reps)
    names = ["home72", "U", "home72_minus_U"]
    return {n: [round(pt[i], 4), round(float(np.percentile(reps[:, i], 2.5)), 4),
                round(float(np.percentile(reps[:, i], 97.5)), 4)] for i, n in enumerate(names)} | {"B": B // 4}


# ------------------------------------------------------------------------------------------------ verdict
def above(x: float, thr: float) -> bool:
    """x >= thr, with the boundary rule: within BOUNDARY of thr does not satisfy."""
    return x >= thr + BOUNDARY


def gt0(x: float) -> bool:
    return x > BOUNDARY


def covers0(lo: float, hi: float) -> bool:
    return lo < -BOUNDARY and hi > BOUNDARY


def verdict(p: dict) -> dict:
    db, bu, gv = p["d_bias"], p["U.bias"], p["G_V"]
    miscal = above(db[0], 0.05) and gt0(db[1])
    policy = covers0(db[1], db[2]) and above(gv[0], 0.05) and gt0(gv[1])
    v = "CRITIC-MISCALIBRATED-ON-U" if miscal else "POLICY-CANNOT-ACT" if policy else "INCONCLUSIVE"
    return {"verdict": v, "inputs": {"d_bias": db, "G_V": gv, "U.bias (reported, not decided on)": bu},
            "conditions": {"miscal": miscal, "policy": policy}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--recon", action="append", required=True, help="name=path, name in {kakuna, pm}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    reads = load(a.games, dict(x.split("=", 1) for x in a.recon))
    res = {"prereg": "PREREG.md", "B": B, "seed": SEED, "reads": {}}
    for name, rd in reads.items():
        out = {"fidelity": rd["fidelity"]}
        if rd["fidelity"]["VOID"]:
            out["VOID"] = True
            res["reads"][name] = out
            continue
        games = rd["games"]
        out["by_turn"] = {}
        for t in TS:
            out["by_turn"][str(t)] = {
                "level": compare(*groups_at(games, t, True), rng),
                "all": compare(*groups_at(games, t, False), rng)}
        out["drift_level10_to_20"] = drift_compare(*drift_groups(games), rng)
        out["auc_all_decisions"] = all_decisions_auc(games, rng)
        prim = out["by_turn"]["10"]["level"]
        out["verdict_rule"] = verdict(prim)
        res["reads"][name] = out
    k = res["reads"].get("kakuna", {})
    if "verdict_rule" in k:
        prim = k["by_turn"]["10"]["level"]
        t10 = k["by_turn"]["10"]["all"]
        res["VERDICT"] = k["verdict_rule"]["verdict"]
        res["predictions"] = {
            "P1_dBrier_ge_0.02": bool(above(prim["d_brier"][0], 0.02) and gt0(prim["d_brier"][1])),
            "P2_miscalibrated_and_G_V_le_0.04": bool(res["VERDICT"] == "CRITIC-MISCALIBRATED-ON-U"
                                                     and prim["G_V"][0] <= 0.04),
            "P3_auc_t10_all_h_minus_U_ge_0.03": bool(above(t10["d_auc_h_minus_U"][0], 0.03)),
        }
    else:
        res["VERDICT"] = "NOT READ (the primary read is VOID)"
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"VERDICT": res["VERDICT"], "predictions": res.get("predictions"),
                      "fidelity": {n: {k: v for k, v in r["fidelity"].items() if k != "excluded_detail"}
                                   for n, r in res["reads"].items()}}, indent=1))


if __name__ == "__main__":
    main()
