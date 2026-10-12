"""Score the replication's arms: the referee T(a), IMP per arm and scorer, the registered contrasts, and §12.3's
B2-B4 for M (design_hidden_info_binding.md §1.1, §6.3, §12.3). Written from the spec; reuses no scoring code.

    python -m replica.score --rows ~/gen3ai_archive/hidden_info_replica/rows --out <dir>

- REFEREE T(a): the mean over truth seeds 16-39 of the win-scale outcome of action a in the TRUE battle (win 1,
  loss 0, tie or truncation 1/2), re-derived from the Phase 1 rows' outcome strings (``out[a][k]`` is seed k:
  '+' win, '-' loss, '0' tie / truncation). Nothing else in the Phase 1 rows is read.
- LEAF-K: the mean outcome of an action over worlds 0..K-1 (a fair arm: world j on dice seed j) or dice seeds
  0..K-1 (the oracle O). Its argmax over ALL legal actions; a tie (|difference| < 1e-6) goes to the action the
  policy ranks higher (the checkpoint's masked logits, from the registered states file).
- IMP = mean over states of T(argmax) - T(top1). CIs: battle-clustered bootstrap, 2,000 draws,
  ``numpy.random.default_rng(20261011)``, percentile 95 %, every contrast PAIRED on the same draws; pooled over
  checkpoints by resampling battles within each checkpoint.
- G = IMP_O - IMP_F; s_M = (IMP_{F+M} - IMP_F) / G; n_M = (IMP_O - IMP_{O-M}) / G (ratios of means on the same
  draws), computed only when G's lower bound > 0.02. G_P = IMP_O - IMP_P.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ARCHIVE = Path(os.path.expanduser("~/gen3ai_archive/value_disc_pilot"))
HERE = Path(__file__).resolve().parent.parent
STATES = HERE.parent / "value_disc_pilot_2026-10-11"
CPS = ("end2", "prod")
ARMS = ("O", "P", "F", "F+M", "O-M")
TRUTH_SEEDS = range(16, 40)
WIN = {"+": 1.0, "-": 0.0, "0": 0.5}
BOUNDARY = 0.0005
N_BOOT = 2000


def referee(cp: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for f in sorted(glob.glob(str(ARCHIVE / cp / "rows" / "*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in out:
                raise RuntimeError(f"{cp}: two Phase 1 rows for {r['id']}")
            out[r["id"]] = {a: float(np.mean([WIN[s[k]] for k in TRUTH_SEEDS])) for a, s in r["out"].items()}
    return out


def policy_order(state: dict, legal: Sequence[str]) -> List[str]:
    lg = state["logits"]
    return sorted(legal, key=lambda a: (-lg[int(a)], int(a)))


def leaf_argmax(outcomes: Dict[str, List[float]], k: int, order: List[str]) -> str:
    """argmax of the mean win-scale outcome over the first k worlds / seeds, ties to the policy's order."""
    best, bv = None, None
    for a in order:
        v = float(np.mean([(x + 1.0) / 2.0 for x in outcomes[a][:k]]))
        if bv is None or v > bv + 1e-6:
            best, bv = a, v
    return best


def load_rows(rows_dir: Path, cp: str, arm: str) -> Dict[str, dict]:
    """One row per state. A state run twice (a failed state RE-RUN after a builder fix that only turns failures
    into worlds, e.g. the shiny flag) keeps its ok row; two ok rows for one state is an error."""
    out: Dict[str, dict] = {}
    for f in sorted(glob.glob(str(rows_dir / cp / arm / "*.jsonl"))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                old = out.get(r["id"])
                if old is not None and old.get("ok") and r.get("ok"):
                    raise RuntimeError(f"{cp} {arm} {r['id']}: two ok rows")
                if old is None or not old.get("ok"):
                    out[r["id"]] = r
    return out


def per_state(rows_dir: Path, cp: str, ks=(1, 4, 8), subset: str = "") -> dict:
    states = json.loads((STATES / f"states_{cp}.json").read_text())["states"]
    if subset:                       # a REGISTERED subset (the interim read): interim_ids_<cp>.txt
        keep = set((HERE / f"{subset}_ids_{cp}.txt").read_text().split())
        states = {k: v for k, v in states.items() if k in keep}
    T = referee(cp)
    res: dict = {"imp": defaultdict(dict), "ok": defaultdict(set), "battle": {}, "excluded": defaultdict(dict)}
    for sid, st in states.items():
        res["battle"][sid] = st["battle"]
    for arm in ARMS:
        rows = load_rows(rows_dir, cp, arm)
        for sid, st in states.items():
            r = rows.get(sid)
            if r is None:
                continue
            if not r.get("ok"):
                res["excluded"][arm][sid] = r.get("reason") or "not ok"
                continue
            legal = sorted(r["outcomes"], key=int)
            if sorted(T[sid], key=int) != legal:
                res["excluded"][arm][sid] = "legal set differs from the referee's"
                continue
            if any(len(v) != 8 for v in r["outcomes"].values()):
                res["excluded"][arm][sid] = "not 8 outcomes per action"
                continue
            order = policy_order(st, legal)
            top1 = order[0]
            for k in ks:
                a = leaf_argmax(r["outcomes"], k, order)
                res["imp"][(arm, f"LEAF-K{k}")][sid] = T[sid][a] - T[sid][top1]
            res["ok"][arm].add(sid)
    return res


def boot_indices(battles_by_cp: Dict[str, List[str]], rng: np.random.Generator):
    """One bootstrap draw: battles resampled with replacement WITHIN each checkpoint."""
    return {cp: rng.choice(len(bs), size=len(bs), replace=True) for cp, bs in battles_by_cp.items()}


def contrast_table(data: Dict[str, dict], cps: Sequence[str], scorer: str, n_boot: int = N_BOOT) -> dict:
    """Every registered quantity on the states where ALL five arms are ok (the common set), pooled over ``cps``
    with states weighted equally, plus each arm's IMP on its own ok set."""
    common = {cp: sorted(set.intersection(*[data[cp]["ok"][a] for a in ARMS])) for cp in cps}
    # per-battle sums so a bootstrap draw is a battle resample
    bat: Dict[str, List[str]] = {}
    sums: Dict[str, Dict[str, np.ndarray]] = {}
    counts: Dict[str, np.ndarray] = {}
    for cp in cps:
        bs = sorted({data[cp]["battle"][s] for s in common[cp]})
        bat[cp] = bs
        ix = {b: i for i, b in enumerate(bs)}
        counts[cp] = np.zeros(len(bs))
        sums[cp] = {a: np.zeros(len(bs)) for a in ARMS}
        for s in common[cp]:
            i = ix[data[cp]["battle"][s]]
            counts[cp][i] += 1
            for a in ARMS:
                sums[cp][a][i] += data[cp]["imp"][(a, scorer)][s]

    def stats_from(w: Dict[str, np.ndarray]) -> dict:
        n = sum(float((counts[cp] * w[cp]).sum()) for cp in cps)
        imp = {a: sum(float((sums[cp][a] * w[cp]).sum()) for cp in cps) / n for a in ARMS}
        G = imp["O"] - imp["F"]
        return {"IMP": imp, "G": G, "G_P": imp["O"] - imp["P"], "dFM": imp["F+M"] - imp["F"],
                "dOM": imp["O"] - imp["O-M"],
                "s_M": (imp["F+M"] - imp["F"]) / G if G != 0 else float("nan"),
                "n_M": (imp["O"] - imp["O-M"]) / G if G != 0 else float("nan")}

    point = stats_from({cp: np.ones(len(bat[cp])) for cp in cps})
    rng = np.random.default_rng(20261011)
    draws = []
    for _ in range(n_boot):
        idx = boot_indices(bat, rng)
        w = {cp: np.bincount(idx[cp], minlength=len(bat[cp])).astype(float) for cp in cps}
        draws.append(stats_from(w))

    def ci(get) -> List[float]:
        v = np.array([get(d) for d in draws], dtype=float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2

    out = {"n_states": {cp: len(common[cp]) for cp in cps}, "n_battles": {cp: len(bat[cp]) for cp in cps},
           "IMP": {a: {"point": point["IMP"][a], "ci": ci(lambda d, a=a: d["IMP"][a])} for a in ARMS}}
    for key in ("G", "G_P", "dFM", "dOM"):
        out[key] = {"point": point[key], "ci": ci(lambda d, k=key: d[k])}
    g_lo = out["G"]["ci"][0]
    if g_lo > 0.02 + BOUNDARY:
        for key in ("s_M", "n_M"):
            out[key] = {"point": point[key], "ci": ci(lambda d, k=key: d[k])}
    else:
        out["s_M"] = out["n_M"] = {"point": None, "ci": None, "note": "no gap to apportion (G lower <= 0.02)"}
    return out


def verdict_b2_b4(t: dict) -> dict:
    def clear_gt(x, thr):
        return x is not None and x > thr + BOUNDARY

    def clear_ge(x, thr):
        return x is not None and x >= thr + BOUNDARY

    g_lo = t["G"]["ci"][0]
    b2 = clear_gt(g_lo, 0.02)
    out = {"B2_G_lower_gt_0.02": b2}
    if t["s_M"]["point"] is None:
        out.update({"B3_sufficiency": False, "B4_necessity": False, "note": "shares not computed (B2 fails)"})
        return out
    out["B3_sufficiency"] = clear_ge(t["s_M"]["point"], 0.50) and clear_ge(t["s_M"]["ci"][0], 0.25)
    out["B4_necessity"] = clear_ge(t["n_M"]["point"], 0.50) and clear_ge(t["n_M"]["ci"][0], 0.25)
    out["boundary_flags"] = [name for name, x, thr in (("G_lower", g_lo, 0.02), ("s_M", t["s_M"]["point"], 0.5),
                                                       ("s_M_lower", t["s_M"]["ci"][0], 0.25),
                                                       ("n_M", t["n_M"]["point"], 0.5),
                                                       ("n_M_lower", t["n_M"]["ci"][0], 0.25))
                             if x is not None and abs(x - thr) <= BOUNDARY]
    return out


def own_set_imps(data: Dict[str, dict], cp: str, scorer: str) -> dict:
    out = {}
    rng = np.random.default_rng(20261011)
    for a in ARMS:
        ids = sorted(data[cp]["ok"][a])
        if not ids:
            continue
        bs = sorted({data[cp]["battle"][s] for s in ids})
        ix = {b: i for i, b in enumerate(bs)}
        sm, ct = np.zeros(len(bs)), np.zeros(len(bs))
        for s in ids:
            i = ix[data[cp]["battle"][s]]
            sm[i] += data[cp]["imp"][(a, scorer)][s]
            ct[i] += 1
        pt = sm.sum() / ct.sum()
        bd = []
        for _ in range(N_BOOT):
            w = np.bincount(rng.choice(len(bs), size=len(bs)), minlength=len(bs))
            bd.append((sm * w).sum() / (ct * w).sum())
        out[a] = {"n": len(ids), "point": float(pt), "ci": [float(np.percentile(bd, 2.5)),
                                                             float(np.percentile(bd, 97.5))]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default="", help="\"interim\": only the registered interim states")
    a = ap.parse_args(argv)
    rows = Path(os.path.expanduser(a.rows))
    data = {cp: per_state(rows, cp, subset=a.subset) for cp in CPS}
    result: dict = {"schema": "hir_replica_result_v1", "excluded": {}, "own_set": {}, "common": {}, "verdicts": {}}
    for cp in CPS:
        result["excluded"][cp] = {arm: {"n": len(v), "reasons": _count(v.values())}
                                  for arm, v in data[cp]["excluded"].items()}
        result["own_set"][cp] = {sc: own_set_imps(data, cp, sc) for sc in ("LEAF-K1", "LEAF-K4", "LEAF-K8")}
    for name, cps in (("end2", ("end2",)), ("prod", ("prod",)), ("pooled", CPS)):
        if not all(data[cp]["ok"].get(arm) for cp in cps for arm in ARMS):
            result["common"][name] = "an arm has no rows yet"
            continue
        result["common"][name] = {sc: contrast_table(data, cps, sc) for sc in ("LEAF-K8", "LEAF-K4", "LEAF-K1")}
        result["verdicts"][name] = verdict_b2_b4(result["common"][name]["LEAF-K8"])
    per_state_out = {cp: {sid: {f"{arm}|{sc}": round(v, 6) for (arm, sc), d in data[cp]["imp"].items()
                                for s2, v in d.items() if s2 == sid}
                          for sid in sorted(data[cp]["battle"])} for cp in CPS}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    (out / "imp_per_state.json").write_text(json.dumps(per_state_out, sort_keys=True) + "\n")
    print(json.dumps(result["verdicts"], indent=1))
    return 0


def _count(xs) -> dict:
    c: Dict[str, int] = defaultdict(int)
    for x in xs:
        c[str(x).split(":")[0][:60]] += 1
    return dict(c)


if __name__ == "__main__":
    raise SystemExit(main())
