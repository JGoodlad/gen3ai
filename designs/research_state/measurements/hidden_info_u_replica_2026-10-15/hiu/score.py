"""This build's OWN read: IMP per arm and scorer, s_U, n_U, s_M, n_M, R3's two paired numerators, with intervals;
the identities; the exclusions; the per-state IMPs; the class marginals of the drawn worlds.

    python -m hiu.score --root ~/gen3ai_archive/hidden_info_u_replica --out <this folder>

No reference value is read or printed: this is the build's read only (the comparison belongs to a third party).

- REFEREE T(a): mean over truth seeds 16-39 of the win-scale outcome (win 1, loss 0, tie / truncation 1/2) of action
  a in the TRUE battle, from the Phase 1 rows' outcome strings (``out[a][k]``: '+', '-', '0').
- LEAF-K: an action's mean win-scale outcome over worlds 0..K-1 (O: dice seeds 0..K-1); its argmax over every legal
  action, a tie (|difference| < 1e-6) to the action the policy ranks higher (the registered masked logits).
- IMP(state, arm) = T(argmax) - T(top1). An arm's IMP = the mean over the COMMON set (states where all six arms are
  ok), every contrast paired on it.
- G = O - F; s_U = (F+U - F) / G; n_U = (O - (O-U)) / G; s_M = (F+M - F) / G; n_M = (O - (O-M)) / G;
  R3 reveal-only = (F+U - F) - (F+M - F); R3 hide-only = (O - (O-U)) - (O - (O-M)).
- CIs: battle-clustered percentile bootstrap, 2,000 draws, ``numpy.random.default_rng(20261015)``, every quantity
  from the same draws (paired); a pooled read resamples battles within each checkpoint, states weighted equally.
- Rule 8: a bound within +-0.005 of 0 (the pre-registration's band for the numerators) is flagged ``boundary``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PILOT = Path(os.path.expanduser("~/gen3ai_archive/value_disc_pilot"))
HERE = Path(__file__).resolve().parent.parent
STATES = HERE.parent / "value_disc_pilot_2026-10-11"
CPS = ("end2", "prod")
ARMS = ("O", "F", "F+U", "F+M", "O-M", "O-U")
TRUTH_SEEDS = range(16, 40)
WIN = {"+": 1.0, "-": 0.0, "0": 0.5}
N_BOOT = 2000
SEED = 20261015
BAND = 0.005
SCORERS = ("LEAF-K8", "LEAF-K4", "LEAF-K1")


def population(sid: str) -> str:
    """The opponent population of a state: ``sentinel_*`` opponents are our own snapshots (the trainer's eval
    labels a snapshot ``sentinel_<i>``); every other label is a scripted bot."""
    opp = sid.split("/")[2]
    return "snapshots" if opp.startswith("sentinel_") else "bots"


def referee(cp: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for f in sorted(glob.glob(str(PILOT / cp / "rows" / "*.jsonl"))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r["id"] in out:
                    raise RuntimeError(f"{cp}: two Phase 1 rows for {r['id']}")
                out[r["id"]] = {a: float(np.mean([WIN[s[k]] for k in TRUTH_SEEDS])) for a, s in r["out"].items()}
    return out


DUPLICATES: Counter = Counter()


def load_rows(root: Path, cp: str, arm: str) -> Dict[str, dict]:
    """One row per state over every worker's file. Builds and playouts are deterministic per (arm, state), so a
    state written twice (two workers racing a dead worker's claim) must carry the same ok flag and outcomes; the
    first row (file name order) is kept, and a disagreement raises."""
    out: Dict[str, dict] = {}
    for p in sorted((root / "rows" / cp / arm).glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                old = out.get(r["id"])
                if old is not None:
                    DUPLICATES[f"{cp}|{arm}"] += 1
                    if (old.get("ok"), old.get("outcomes")) != (r.get("ok"), r.get("outcomes")):
                        raise RuntimeError(f"{cp} {arm} {r['id']}: two rows that disagree")
                    continue
                out[r["id"]] = r
    return out


def leaf_argmax(outcomes: Dict[str, List[float]], k: int, order: List[str]) -> str:
    best, bv = None, None
    for a in order:
        v = float(np.mean([(x + 1.0) / 2.0 for x in outcomes[a][:k]]))
        if bv is None or v > bv + 1e-6:
            best, bv = a, v
    return best


def per_state(root: Path, cp: str) -> dict:
    states = json.loads((STATES / f"states_{cp}.json").read_text())["states"]
    T = referee(cp)
    res: dict = {"imp": defaultdict(dict), "ok": defaultdict(set), "battle": {}, "excluded": defaultdict(dict),
                 "rows": {}, "k_seen": {}}
    for sid, st in states.items():
        res["battle"][sid] = st["battle"]
    for arm in ARMS:
        rows = load_rows(root, cp, arm)
        res["rows"][arm] = rows
        for sid, st in states.items():
            r = rows.get(sid)
            if r is None:
                continue
            if "k_seen" in r:
                res["k_seen"][sid] = r["k_seen"]
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
            lg = st["logits"]
            order = sorted(legal, key=lambda a: (-lg[int(a)], int(a)))
            for sc in SCORERS:
                k = int(sc.split("K")[1])
                res["imp"][(arm, sc)][sid] = T[sid][leaf_argmax(r["outcomes"], k, order)] - T[sid][order[0]]
            res["ok"][arm].add(sid)
    return res


def _quantities(imp: Dict[str, float]) -> Dict[str, float]:
    G = imp["O"] - imp["F"]
    dFU, dOU = imp["F+U"] - imp["F"], imp["O"] - imp["O-U"]
    dFM, dOM = imp["F+M"] - imp["F"], imp["O"] - imp["O-M"]
    nan = float("nan")
    return {"G": G, "dFU": dFU, "dOU": dOU, "dFM": dFM, "dOM": dOM,
            "s_U": dFU / G if G else nan, "n_U": dOU / G if G else nan,
            "s_M": dFM / G if G else nan, "n_M": dOM / G if G else nan,
            "R3_reveal_only": dFU - dFM, "R3_hide_only": dOU - dOM}


def table(data: Dict[str, dict], sel: Dict[str, Sequence[str]], scorer: str) -> dict:
    """Every quantity on the given per-checkpoint state lists (each a subset of the common set)."""
    cps = [cp for cp in sel if sel[cp]]
    if not cps:
        return {"n_states": 0}
    bat, sums, counts = {}, {}, {}
    for cp in cps:
        bs = sorted({data[cp]["battle"][s] for s in sel[cp]})
        ix = {b: i for i, b in enumerate(bs)}
        bat[cp] = bs
        counts[cp] = np.zeros(len(bs))
        sums[cp] = {a: np.zeros(len(bs)) for a in ARMS}
        for s in sel[cp]:
            i = ix[data[cp]["battle"][s]]
            counts[cp][i] += 1
            for a in ARMS:
                sums[cp][a][i] += data[cp]["imp"][(a, scorer)][s]

    def stats(w):
        n = sum(float((counts[cp] * w[cp]).sum()) for cp in cps)
        imp = {a: sum(float((sums[cp][a] * w[cp]).sum()) for cp in cps) / n for a in ARMS}
        return {"IMP": imp, **_quantities(imp)}

    point = stats({cp: np.ones(len(bat[cp])) for cp in cps})
    rng = np.random.default_rng(SEED)
    draws = [stats({cp: np.bincount(rng.choice(len(bat[cp]), size=len(bat[cp])), minlength=len(bat[cp]))
                    .astype(float) for cp in cps}) for _ in range(N_BOOT)]

    def ci(get):
        v = np.array([get(d) for d in draws], dtype=float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]

    out = {"n_states": {cp: len(sel[cp]) for cp in cps}, "n_battles": {cp: len(bat[cp]) for cp in cps},
           "IMP": {a: {"point": point["IMP"][a], "ci": ci(lambda d, a=a: d["IMP"][a])} for a in ARMS}}
    for key in ("G", "dFU", "dOU", "dFM", "dOM", "s_U", "n_U", "s_M", "n_M", "R3_reveal_only", "R3_hide_only"):
        out[key] = {"point": point[key], "ci": ci(lambda d, k=key: d[k])}
    for key in ("R3_reveal_only", "R3_hide_only", "dFU", "dOU", "dFM", "dOM"):
        lo, hi = out[key]["ci"]
        out[key]["excludes_0_clear"] = bool(lo is not None and (lo > BAND or hi < -BAND))
        out[key]["boundary"] = bool(lo is not None and (abs(lo) <= BAND or abs(hi) <= BAND))
    return out


def identities(data: Dict[str, dict]) -> dict:
    """At k = 6 (every opponent mon seen): F+U = F and O-U = O, outcome for outcome (every action, every world)."""
    out = {}
    for cp in CPS:
        d = data[cp]
        rec = {"k6_states": 0, "FU_eq_F": 0, "FU_ne_F": [], "FU_or_F_missing": 0,
               "OU_eq_O": 0, "OU_ne_O": [], "OU_or_O_missing": 0}
        for sid, k in d["k_seen"].items():
            if k != 6:
                continue
            rec["k6_states"] += 1
            for a, b, eq, ne, miss in (("F+U", "F", "FU_eq_F", "FU_ne_F", "FU_or_F_missing"),
                                       ("O-U", "O", "OU_eq_O", "OU_ne_O", "OU_or_O_missing")):
                ra, rb = d["rows"][a].get(sid), d["rows"][b].get(sid)
                if ra is None or rb is None:      # not run yet
                    rec[miss] += 1
                    continue
                if not (ra.get("ok") and rb.get("ok")):
                    if bool(ra.get("ok")) != bool(rb.get("ok")):
                        ne_reason = f"ok differs: {a} {bool(ra and ra.get('ok'))} {b} {bool(rb and rb.get('ok'))}"
                        rec[ne].append([sid, ne_reason])
                    else:
                        rec[miss] += 1
                    continue
                if ra["outcomes"] == rb["outcomes"]:
                    rec[eq] += 1
                else:
                    rec[ne].append([sid, "outcomes differ"])
        rec["HOLD"] = not rec["FU_ne_F"] and not rec["OU_ne_O"]
        out[cp] = rec
    return out


def marginals(data: Dict[str, dict]) -> dict:
    """Per checkpoint and arm, over every ok state's 8 worlds: the never-seen slots' species / item / ability /
    nature frequencies (class U) and the seen mons' drawn move / item / ability frequencies, for a TV test."""
    out: dict = {}
    for cp in CPS:
        out[cp] = {}
        for arm in ARMS:
            if arm == "O":
                continue
            c: Dict[str, Counter] = defaultdict(Counter)
            n = 0
            for sid in data[cp]["ok"][arm]:
                r = data[cp]["rows"][arm][sid]
                us = set(r.get("u_slots", []))
                for w in r["worlds"]:
                    n += 1
                    for i, mon in enumerate(w["team"].split("]")):
                        f = mon.split("|")
                        sp = (f[1] or f[0]).lower()
                        if i in us:
                            c["U_species"][sp] += 1
                            c["U_item"][f[2]] += 1
                            c["U_ability"][f[3]] += 1
                            c["U_nature"][f[5]] += 1
                            for m in f[4].split(","):
                                c["U_move"][m] += 1
                        else:
                            for m in f[4].split(","):
                                c["seen_move"][sp + ":" + m] += 1
                            c["seen_item"][sp + ":" + f[2]] += 1
                            c["seen_ability"][sp + ":" + f[3]] += 1
            out[cp][arm] = {"n_worlds": n, **{k: dict(v.most_common()) for k, v in c.items()}}
    return out


def tv_permutation_test(a: Sequence[str], b: Sequence[str], n_perm: int = 2000, seed: int = SEED) -> dict:
    """Total variation between two samples of categorical draws and its permutation null (labels shuffled): the
    p-value is the share of permuted TVs >= the observed one. Draws within a world / state are not independent, so
    for a clustered null permute whole clusters (pass cluster-level lists of draws flattened per cluster)."""
    def tv(x, y):
        cx, cy = Counter(x), Counter(y)
        return 0.5 * sum(abs(cx[k] / len(x) - cy[k] / len(y)) for k in set(cx) | set(cy))
    obs = tv(a, b)
    pool = list(a) + list(b)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pool))
        x = [pool[i] for i in p[:len(a)]]
        y = [pool[i] for i in p[len(a):]]
        ge += tv(x, y) >= obs - 1e-12
    return {"tv": obs, "p": (ge + 1) / (n_perm + 1), "n_a": len(a), "n_b": len(b)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/gen3ai_archive/hidden_info_u_replica")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    root = Path(os.path.expanduser(a.root))
    data = {cp: per_state(root, cp) for cp in CPS}
    common = {cp: sorted(set.intersection(*[data[cp]["ok"][x] for x in ARMS])) for cp in CPS}
    result: dict = {"schema": "hiu_result_v1", "rows_present": {}, "excluded": {}, "common_n": {},
                    "reads": {}, "identities": identities(data)}
    for cp in CPS:
        result["rows_present"][cp] = {arm: len(data[cp]["rows"][arm]) for arm in ARMS}
        result["excluded"][cp] = {arm: {"n": len(v), "reasons": dict(Counter(str(x).split(":")[0][:60]
                                                                             for x in v.values())),
                                        "ids": v} for arm, v in data[cp]["excluded"].items()}
        result["common_n"][cp] = len(common[cp])
    result["duplicate_rows"] = dict(DUPLICATES)
    reads = {"end2": {"end2": common["end2"], "prod": []}, "prod": {"end2": [], "prod": common["prod"]},
             "pooled": common}
    for pop in ("bots", "snapshots"):
        for cp in CPS:
            reads[f"{cp}|{pop}"] = {c: ([s for s in common[c] if population(s) == pop] if c == cp else [])
                                    for c in CPS}
        reads[f"pooled|{pop}"] = {c: [s for s in common[c] if population(s) == pop] for c in CPS}
    for name, sel in reads.items():
        result["reads"][name] = {sc: table(data, sel, sc) for sc in SCORERS}
    per = {cp: {sid: {f"{arm}|{sc}": round(v, 6) for (arm, sc), dd in data[cp]["imp"].items()
                      for s2, v in dd.items() if s2 == sid} for sid in sorted(data[cp]["battle"])} for cp in CPS}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    (out / "imp_per_state.json").write_text(json.dumps(per, sort_keys=True) + "\n")
    (out / "marginals.json").write_text(json.dumps(marginals(data), sort_keys=True) + "\n")
    print(json.dumps({"common_n": result["common_n"], "identities": {cp: {k: v for k, v in r.items()
                                                                         if not isinstance(v, list)}
                                                                     for cp, r in result["identities"].items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
