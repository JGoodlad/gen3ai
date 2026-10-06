"""The meters, the bootstrap and the verdict (README §3-§4). Reads ``rows/scores.npz`` and the blob
policy's root read (``rows/reads/blob.probs.npz``), writes ``result.json`` and ``result.md``.

    python score.py
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows"
LANES = HERE.parent / "m5_laneS"
EPS = 0.1
STARVE = 0.01
TIE = 1e-6
BOUND = 0.0005
FLAT = 0.02
N_BOOT = 2000
SEED = 20261006
KS = (1, 4, 8)


def load_decisions():
    with gzip.open(LANES / "bank_v1" / "decisions.jsonl.gz", "rt") as f:
        return [json.loads(x) for x in f]


def root_policy(decs):
    z = np.load(ROWS / "reads" / "blob.probs.npz")
    import hashlib

    if str(z["ids_sha256"]) != hashlib.sha256("\n".join(d["id"] for d in decs).encode()).hexdigest():
        raise SystemExit("the root read was made on a different bank")
    return {d["id"]: (z["probs"][i].astype(np.float64), z["logits"][i].astype(np.float64)) for i, d in enumerate(decs)}


def contested_ids(decs, pol, subset_ids):
    """README §3: free, turn 2-40, >= 3 legal, top-2 masked-logit gap < the 40th percentile over all
    ELIGIBLE subset turns (train and held-out alike)."""
    by = {d["id"]: d for d in decs}
    gaps = {}
    for i in subset_ids:
        d = by[i]
        if d["kind"] != "free" or not (2 <= d["turn"] <= 40) or d["n_legal"] < 3:
            continue
        m = np.array([c == "1" for c in d["mask"]])
        lg = np.sort(pol[i][1][m])[::-1]
        gaps[i] = float(lg[0] - lg[1])
    thr = float(np.percentile(list(gaps.values()), 40))
    return {i for i, g in gaps.items() if g < thr}, thr, len(gaps)


def turn_tables(sc):
    """{turn id: {"battle", "actions": [a], "truth": [n_a, 24], "scorers": {name: [n_a]}}}"""
    names = [k for k in sc.files if k == "BASE" or k.startswith("REFIT_") or k.startswith("LEAF_")]
    t: dict = {}
    for r in range(len(sc["ids"])):
        did = str(sc["ids"][r])
        e = t.setdefault(did, {"battle": str(sc["battle"][r]), "actions": [], "truth": [],
                               "scorers": {n: [] for n in names}})
        e["actions"].append(int(sc["action"][r]))
        e["truth"].append(sc["truth"][r])
        for n in names:
            e["scorers"][n].append(float(sc[n][r]))
    for e in t.values():
        e["truth"] = np.stack(e["truth"])
        e["scorers"] = {n: np.array(v) for n, v in e["scorers"].items()}
    return t, names


def classes(truth: np.ndarray):
    """Lane S on the truth seeds: a*, gap, SE, near-best, dominated (per action)."""
    mean = truth.mean(1)
    best = int(np.argmax(mean))
    d = truth[best][None, :] - truth
    gap = d.mean(1)
    se = d.std(1, ddof=1) / np.sqrt(truth.shape[1])
    return best, gap, gap <= EPS, (gap - 1.96 * se) > EPS


def pairs_for(e, kind, pol_probs=None, pol_logits=None, mask_actions=None):
    """[(i, j, sign of truth Δ)] for one turn; TIES (Δ == 0 exactly; Δ is a multiple of 1/24) dropped."""
    tm = e["truth"].mean(1)
    n = len(e["actions"])
    out = []
    if kind == "all":
        cand = [(i, j) for i in range(n) for j in range(i + 1, n)]
    elif kind == "top12":
        lg = np.array([pol_logits[a] for a in e["actions"]])
        o = np.argsort(-lg, kind="stable")
        cand = [(int(o[0]), int(o[1]))]
    else:
        best, gap, near, dom = classes(e["truth"])
        pr = np.array([pol_probs[a] for a in e["actions"]])
        starved = near & (pr < STARVE)
        cand = []
        for i in np.flatnonzero(starved):
            for j in range(n):
                if j == i:
                    continue
                if kind == "starved" or (kind == "starved_vs_best" and j == best) or \
                        (kind == "starved_vs_dominated" and dom[j]):
                    cand.append((int(min(i, j)), int(max(i, j))))
        cand = sorted(set(cand))
    for i, j in cand:
        dt = (e["truth"][i] - e["truth"][j]).mean()
        q = round(dt * 24)
        if q == 0:
            continue
        out.append((i, j, 1 if q > 0 else -1))
    return out


def correct(s, i, j, sign):
    d = s[i] - s[j]
    if abs(d) < TIE:
        return 0.5
    return 1.0 if (d > 0) == (sign > 0) else 0.0


def meter(tabs, names, turn_ids, kind, pol):
    """Per battle: [n pairs, sum correct per scorer]; returns (per_battle dict, names)."""
    pb = defaultdict(lambda: np.zeros(1 + len(names)))
    for did in turn_ids:
        e = tabs[did]
        pr, lg = pol[did]
        for i, j, sg in pairs_for(e, kind, pr, lg):
            v = pb[e["battle"]]
            v[0] += 1
            for k, n in enumerate(names):
                v[1 + k] += correct(e["scorers"][n], i, j, sg)
    return pb


def boot(pb, names, contrasts):
    bats = sorted(pb)
    M = np.stack([pb[b] for b in bats]) if bats else np.zeros((0, 1 + len(names)))
    tot = M.sum(0)
    n = tot[0]
    acc = {nm: (tot[1 + k] / n if n else float("nan")) for k, nm in enumerate(names)}
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(bats), size=(N_BOOT, len(bats))) if bats else None
    res = {"n_pairs": int(n), "n_battles": len(bats), "acc": {}, "contrast": {}}
    if draws is None or n == 0:
        return res
    S = M[draws].sum(1)                      # [N_BOOT, 1 + names]
    A = S[:, 1:] / S[:, :1]
    for k, nm in enumerate(names):
        lo, hi = np.percentile(A[:, k], [2.5, 97.5])
        res["acc"][nm] = [acc[nm], float(lo), float(hi)]
    for a, b in contrasts:
        ia, ib = names.index(a), names.index(b)
        d = A[:, ia] - A[:, ib]
        lo, hi = np.percentile(d, [2.5, 97.5])
        res["contrast"][f"{a} - {b}"] = [acc[a] - acc[b], float(lo), float(hi)]
    return res


def verdict(m1):
    c8 = m1["contrast"]["REFIT_K8_s0 - BASE"]
    lf = m1["contrast"]["LEAF_K8 - BASE"]
    closed = c8[0] / lf[0] if lf[0] != 0 else float("nan")
    if lf[1] <= BOUND:
        v = "NO ROOM (read void)"
    elif c8[1] > BOUND:
        v = "LABEL-LIMITED" if closed >= 0.5 else "PARTIAL"
    elif c8[1] >= -FLAT and c8[2] <= FLAT:
        v = "FLAT"
    else:
        v = "NOT DETECTED, NOT SHOWN FLAT"
    boundary = any(abs(x - y) < BOUND for x in (c8[1], lf[1]) for y in (0.0,)) or \
        abs(c8[1] + FLAT) < BOUND or abs(c8[2] - FLAT) < BOUND
    return {"verdict": v, "closed": closed, "boundary_flag": bool(boundary)}


def main() -> int:
    import sys

    pre = "smoke_" if "--smoke" in sys.argv else ""
    decs = load_decisions()
    pol = root_policy(decs)
    sc = np.load(ROWS / f"{pre}scores.npz")
    tabs, names = turn_tables(sc)
    sub_ids = json.loads((ROWS / "subset_train.json").read_text())["ids"] + \
        json.loads((ROWS / "subset_held.json").read_text())["ids"]
    cont, thr, n_elig = contested_ids(decs, pol, sub_ids)
    by = {d["id"]: d for d in decs}
    held = sorted(tabs)
    held_cont = [i for i in held if i in cont]
    held_free = [i for i in held if by[i]["kind"] == "free"]
    contrasts = [("REFIT_K8_s0", "BASE"), ("LEAF_K8", "BASE"), ("REFIT_K8_s0", "REFIT_K1_s0"),
                 ("REFIT_K1_s0", "BASE"), ("REFIT_K4_s0", "BASE"), ("LEAF_K4", "BASE"), ("LEAF_K1", "BASE"),
                 ("LEAF_K8", "REFIT_K8_s0")]
    out = {"contested_threshold_logit_gap": thr, "n_eligible": n_elig, "n_contested_all": len(cont),
           "n_held_turns": len(held), "n_held_contested": len(held_cont), "n_held_free": len(held_free),
           "meters": {}}
    for key, ids, kind in (("M1_all_contested", held_cont, "all"), ("M2_top1_top2", held_cont, "top12"),
                           ("M3_starved", held_free, "starved"), ("M3a_starved_vs_best", held_free, "starved_vs_best"),
                           ("M3b_starved_vs_dominated", held_free, "starved_vs_dominated")):
        out["meters"][key] = boot(meter(tabs, names, ids, kind, pol), names, contrasts)
    out["verdict"] = verdict(out["meters"]["M1_all_contested"])
    fr = json.loads((ROWS / f"{pre}fit_report.json").read_text())
    out["fit_report"] = fr
    ((ROWS / "smoke_result.json") if pre else (HERE / "result.json")).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"verdict": out["verdict"], "M1": out["meters"]["M1_all_contested"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
