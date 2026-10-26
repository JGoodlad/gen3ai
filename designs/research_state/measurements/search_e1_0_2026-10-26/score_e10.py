"""E1-0's READ (``PREREGISTRATION.md`` §4-§7): the picks ALREADY BANKED by the maturity battery (X26 225M, interim LEAN),
world-sampler stage 1 and stage 1c, re-scored on the BOT-stratum states under (a) the SELF-referee as banked and (b) the
BOT referee (``play_e10.py --mode bot``).

HOW: each experiment's OWN scoring code runs unchanged (``score_mat.score_anchor``, ``score_s1.score``, and stage 1c's
``read_b`` per-state rule, replicated from ``read_1c.py`` with its imported ``l1l5_data`` / ``l1_read``), with ONE
substitution: Tier 1's ``t1_data.truth_arrays`` sees, for a bot-stratum state in mode (b), the truth row's referee
seeds 16..39 replaced by the bot referee's outcomes. No scorer reads seeds 16..39 (``t1_data``: the referee T is the
only reader), so every pick is the banked pick, and the substitution changes T alone; G-PICKS checks it.

    python score_e10.py [--selftest]      -> result_e10.json (this folder); --selftest: the rules on planted data

Pure numpy over the archives (read only). Outcomes are printed only as the final record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent
MATD = MEAS / "hidden_info_maturity_x26_2026-10-18"
S1D = MEAS / "world_sampler_stage1_2026-10-19"
C1D = MEAS / "search_1c_2026-10-20"
L1S = MEAS / "search_l1_l5_2026-10-20" / "scripts"
for _p in (L1S, C1D, S1D, MATD):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(HERE))
import play_e10 as P  # noqa: E402  (ARCH, the registered order; no engine at import)

ANCHOR = "x26_225M"
SEED = 20261026
NB = 2000
EDGE = 0.0005
GAME_BAR = 0.0018          # the design's SELF-REFEREE-MISREADS magnitude (design_search_next.md, E1's rules)
GROSS_BAR = 0.015          # the review's F8: E1-0 resolves a gross misreading, about +-0.015
PRIMARY = ("MAT:F:LEAF-K8", "MAT:O:LEAF-K8", "S1:C_pool-F_legal:LEAF-K8", "1C:PIKL:C128L:LEAF-K8")
LEVEL_P = 100.0 * (1.0 - 0.05 / len(PRIMARY))        # Bonferroni over the primary family: 98.75 %
LEVEL_S = 95.0
G_IDENT_TOL = 1.5e-5       # the published json rounds to 5 decimals
MAX_REFUSED = 0.02


# ------------------------------------------------------------------------------------------- statistics

class Boot:
    """Battle-clustered bootstrap: ``C[r, b]`` = how often battle b is drawn in replicate r (every cell on the SAME
    draws, so differences are paired)."""

    def __init__(self, battles: List[str], seed: int = SEED, n: int = NB):
        self.ub = sorted(set(battles))
        ix = {b: i for i, b in enumerate(self.ub)}
        self.ix = ix
        nb = len(self.ub)
        rng = np.random.default_rng(seed)
        self.C = np.zeros((n, nb))
        draws = rng.integers(0, nb, size=(n, nb))
        for r in range(n):
            np.add.at(self.C[r], draws[r], 1.0)

    def mean(self, battles: List[str], x: np.ndarray) -> Tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        bi = np.array([self.ix[b] for b in battles], dtype=np.int64)
        S = np.bincount(bi, weights=x, minlength=len(self.ub))
        N = np.bincount(bi, minlength=len(self.ub)).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = (self.C @ S) / (self.C @ N)
        return (float(S.sum() / N.sum()) if N.sum() else float("nan")), vec


def ci(vec: np.ndarray, level: float) -> List[Optional[float]]:
    v = vec[np.isfinite(vec)]
    if len(v) < 20:
        return [None, None]
    a = (100.0 - level) / 2.0
    lo, hi = np.percentile(v, [a, 100.0 - a])
    return [float(lo), float(hi)]


def summ(est: float, vec: np.ndarray, level: float, n: int) -> dict:
    lo, hi = ci(vec, level)
    return {"mean": round(est, 5), "ci": [None if lo is None else round(lo, 5), None if hi is None else round(hi, 5)],
            "level": level, "n": n}


def sign_class(c: List[Optional[float]]) -> str:
    """POSITIVE iff lower > +EDGE; NEGATIVE iff upper < -EDGE; else NULL (rule 8's edge)."""
    if c[0] is not None and c[0] > EDGE:
        return "POSITIVE"
    if c[1] is not None and c[1] < -EDGE:
        return "NEGATIVE"
    return "NULL"


def cell_read(self_v: dict, bot_v: dict, battle: Dict[str, str], bt: Boot, level: float) -> Optional[dict]:
    ids = sorted(i for i in self_v if i in bot_v and np.isfinite(self_v[i]) and np.isfinite(bot_v[i]))
    if not ids:
        return None
    bs = [battle[i] for i in ids]
    a = np.array([self_v[i] for i in ids])
    b = np.array([bot_v[i] for i in ids])
    ea, va = bt.mean(bs, a)
    eb, vb = bt.mean(bs, b)
    ed, vd = bt.mean(bs, b - a)
    out = {"self": summ(ea, va, level, len(ids)), "bot": summ(eb, vb, level, len(ids)),
           "bot_minus_self": summ(ed, vd, level, len(ids)),
           "changed_pick_states": int(np.sum((a != 0) | (b != 0)))}
    nz = (a != 0) | (b != 0)
    out["corr_state_imp"] = (round(float(np.corrcoef(a[nz], b[nz])[0, 1]), 4)
                             if nz.sum() >= 3 and np.std(a[nz]) > 0 and np.std(b[nz]) > 0 else None)
    out["class"] = {"self": sign_class(out["self"]["ci"]), "bot": sign_class(out["bot"]["ci"])}
    return out


def disagrees(c: dict) -> bool:
    """RELABEL condition for one cell: D's interval excludes 0 (rule 8's edge) with |D| > GAME_BAR, or the two
    referees' sign classes are OPPOSITE (one POSITIVE, the other NEGATIVE)."""
    d = c["bot_minus_self"]
    lo, hi = d["ci"]
    excl = (lo is not None and lo > EDGE) or (hi is not None and hi < -EDGE)
    opp = {c["class"]["self"], c["class"]["bot"]} == {"POSITIVE", "NEGATIVE"}
    return bool((excl and abs(d["mean"]) > GAME_BAR) or opp)


def inside_gross(c: dict) -> bool:
    lo, hi = c["bot_minus_self"]["ci"]
    return lo is not None and hi is not None and lo > -GROSS_BAR + EDGE and hi < GROSS_BAR - EDGE


def verdict(prim: Dict[str, dict]) -> str:
    if any(disagrees(c) for c in prim.values()):
        return "RELABEL-NEEDED"
    if all(inside_gross(c) for c in prim.values()):
        return "REFEREE-AGREES"
    return "UNDECIDED"


# ------------------------------------------------------------------------------------------- spearman

def _rank(x: np.ndarray) -> np.ndarray:
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x), dtype=np.float64)
    for v in np.unique(x):           # average ranks over ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 3:
        return None
    ra, rb = _rank(np.asarray(a, float)), _rank(np.asarray(b, float))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------------------------------- the substitution

class Sub:
    """The ONE change to the banked scoring: in mode "bot", a bot-stratum truth row's seeds 16..39 are the bot
    referee's. Every pick the scorers make is recorded (G-PICKS)."""

    def __init__(self, bot_rows: Dict[str, dict]):
        import t1_data as D

        self.D = D
        self.bot_rows = bot_rows
        self.mode = "self"
        self.picks: List[int] = []
        self._ta, self._pick = D.truth_arrays, D.pick
        D.truth_arrays = self.truth_arrays
        D.pick = self.pick

    def truth_arrays(self, r: dict, meta: dict) -> dict:
        if self.mode == "bot":
            b = self.bot_rows.get(r["id"])
            if b is not None:
                if sorted(b["out"], key=int) != sorted(r["out"], key=int):
                    raise SystemExit(f"{r['id']}: the bot row's actions differ from the truth row's")
                r = dict(r)
                r["out"] = {a: r["out"][a][:16] + b["out"][a] + r["out"][a][40:] for a in r["out"]}
        return self._ta(r, meta)

    def pick(self, score, order):
        i = self._pick(score, order)
        self.picks.append(int(i))
        return i


def cells_mat() -> Dict[str, Dict[str, float]]:
    import score_mat as SM

    SM.E.set_bank("interim")
    recs = SM.score_anchor(ANCHOR, False)["recs"]
    out: Dict[str, Dict[str, float]] = {}
    for r in recs:
        if r.get("tie") or not r["src"].get("O", {}).get("ok"):
            continue
        for s in ("O", "F", "FN"):
            e = r["src"].get(s, {})
            if e.get("ok"):
                for z, v in e["imp"].items():
                    out.setdefault(f"MAT:{s}:{z}", {})[r["id"]] = v
        if r["src"].get("F", {}).get("ok"):
            out.setdefault("MAT:G_F:LEAF-K8", {})[r["id"]] = r["src"]["O"]["imp"]["LEAF-K8"] - r["src"]["F"]["imp"]["LEAF-K8"]
    return out


def cells_s1() -> Dict[str, Dict[str, float]]:
    import s1_common as E1
    import score_s1 as SS

    tab = SS.score(False)
    out: Dict[str, Dict[str, float]] = {}
    srcs = ["O", "F", "FN", *E1.ARMS, "F_rolls_w"]
    for r in tab["recs"]:
        if r["tie"] or not r["src"].get("O", {}).get("ok"):
            continue
        for s in srcs:
            e = r["src"].get(s, {})
            if e.get("ok"):
                for z, v in e["imp"].items():
                    out.setdefault(f"S1:{s}:{z}", {})[r["id"]] = v
        base = r["src"].get("F_legal", {})
        for x in E1.PRIMARY:
            e = r["src"].get(x, {})
            if e.get("ok") and base.get("ok"):
                for z in ("LEAF-K8", "TRUNC-4-K8"):
                    out.setdefault(f"S1:{x}-F_legal:{z}", {})[r["id"]] = e["imp"][z] - base["imp"][z]
    return out


def cells_1c(sub: Sub) -> Dict[str, Dict[str, float]]:
    """``read_1c.read_b``'s per-state IMP (its ``imp`` closure), on read_b's own state set and arms."""
    import c1c_common as K
    import s1_common as E1
    import l1_read as L1
    import l1l5_data as LD

    ra = json.loads((C1D / "result_1c_a.json").read_text())
    play = {arm: K.read_by_id(K.ARCH / "play" / arm / "rows.jsonl") for arm in ra["bar"]["play"]}
    rows_by = K.read_by_id(K.ARCH / "rows" / "rows.jsonl")
    order = E1.ordered_ids()
    rows = [rows_by[i] for i in order if i in rows_by]
    ids = [r["id"] for r in rows if all(r["id"] in play[x] for x in play)]
    truth: Dict[str, dict] = {}
    for p in sorted((E1.MAT_ARCH / "rows" / E1.ANCHOR / "O").glob("shard_*.jsonl")):
        truth.update({r["id"]: r for r in K.read_rows(p)})
    roots = {r["id"]: r for r in K.read_rows(E1.MAT_ARCH / "roots" / f"{E1.ANCHOR}.jsonl")}
    import read_1c as R1

    key = {"LEAF-K8": "W", "TRUNC-4-K8": "X4", "CRITIC": "crit"}
    out: Dict[str, Dict[str, float]] = {}
    for did in ids:
        r1, rt = truth.get(did), roots.get(did)
        if r1 is None or not r1.get("ok") or rt is None:
            continue
        tp = LD.truth_part(r1, rt["logits"], rt["top"])
        Tp, logit = tp["rec"]["Tp"], tp["rec"]["logit"]
        for a in play:
            rw = play[a].get(did)
            if rw is None or not rw.get("ok"):
                continue
            s = LD.world_part(rw, tp)
            for rule in ("PIKL", "ARGMAX"):
                x = R1.L1_FIT[rule] if rule != "ARGMAX" else None
                for z, k in key.items():
                    q = np.asarray(s[k], dtype=np.float64).mean(1)
                    i = L1.pick(rule, x[z] if x else 0.0, q, logit)
                    sub.picks.append(int(i))
                    out.setdefault(f"1C:{rule}:{a}:{z}", {})[did] = float(Tp[i] - Tp[0])
    return out


def all_cells(sub: Sub) -> Dict[str, Dict[str, float]]:
    c = {}
    c.update(cells_mat())
    c.update(cells_s1())
    c.update(cells_1c(sub))
    return c


# ------------------------------------------------------------------------------------------- the read

def g_ident(cells: Dict[str, Dict[str, float]], strata: Dict[str, str]) -> dict:
    """The self-mode extraction reproduces the PUBLISHED bot-stratum points (the extraction is the banked one)."""
    pub = []
    rm = json.loads((MATD / "result_mat.json").read_text())["per_anchor"][ANCHOR]["G_F"]["stratum:bot"]
    pub.append(("MAT:G_F:LEAF-K8", rm["mean"], rm["n"]))
    rs = json.loads((S1D / "result_s1.json").read_text())["PRIMARY"]
    for x, blk in rs.items():
        for z, v in blk["per_stratum"]["bot"].items():
            pub.append((f"S1:{x}-F_legal:{z}", v["mean"], v["n"]))
    rc = json.loads((C1D / "result_1c_b.json").read_text())["table"]
    for rule in ("PIKL", "ARGMAX"):
        for a, blk in rc[rule].items():
            for z, v in blk.items():
                b = v["by_stratum"]["bot"]
                pub.append((f"1C:{rule}:{a}:{z}", b["mean"], b["n"]))
    checks, fails = [], 0
    for name, mean, n in pub:
        v = [x for i, x in cells.get(name, {}).items() if strata.get(i) == "bot"]
        got = float(np.mean(v)) if v else float("nan")
        ok = len(v) == n and abs(got - mean) <= G_IDENT_TOL
        fails += int(not ok)
        checks.append({"cell": name, "published": mean, "n_pub": n, "got": round(got, 6), "n": len(v), "ok": ok})
    return {"checked": len(checks), "failed": fails, "pass": fails == 0, "checks": checks}


def t_agreement(truth: Dict[str, dict], bot_rows: Dict[str, dict], roots: Dict[str, dict]) -> dict:
    """The state-level RANK agreement of the two referees' T over every legal action, and their top-T agreement."""
    import t1_data as D

    rhos, top_eq, lev_self, lev_bot = [], [], [], []
    for did, b in bot_rows.items():
        r1 = truth[did]
        acts = sorted(r1["out"], key=int)
        ts = np.array([np.mean([D.DEC[c] for c in r1["out"][a][16:40]]) for a in acts]) * 0.5 + 0.5
        tb = np.array([np.mean([D.DEC[c] for c in b["out"][a]]) for a in acts]) * 0.5 + 0.5
        lev_self.append(float(ts.mean()))
        lev_bot.append(float(tb.mean()))
        rho = spearman(ts, tb)
        if rho is not None:
            rhos.append(rho)
        lg = np.array([roots[did]["logits"][int(a)] for a in acts], dtype=np.float64)
        order = sorted(range(len(acts)), key=lambda i: (-lg[i], int(acts[i])))
        top_eq.append(int(D.pick(ts, order) == D.pick(tb, order)))
    r = np.array(rhos)
    return {"states": len(bot_rows), "spearman_states": len(r),
            "spearman_mean": round(float(r.mean()), 4) if len(r) else None,
            "spearman_median": round(float(np.median(r)), 4) if len(r) else None,
            "spearman_share_positive": round(float(np.mean(r > 0)), 4) if len(r) else None,
            "top_T_action_agrees": round(float(np.mean(top_eq)), 4),
            "mean_T_self": round(float(np.mean(lev_self)), 4), "mean_T_bot": round(float(np.mean(lev_bot)), 4)}


def guards(ids: List[str], truth: Dict[str, dict], bot: Dict[str, dict], selfr: Dict[str, dict], dec: dict) -> dict:
    g: dict = {}
    want = [i for i in ids if (truth.get(i) or {}).get("ok")]
    g["states"] = len(ids)
    g["with_ok_truth"] = len(want)
    g["bot_rows"] = sum(1 for i in want if i in bot)
    ok = [i for i in want if (bot.get(i) or {}).get("ok")]
    g["bot_refused"] = sum(1 for i in want if i in bot and not bot[i].get("ok"))
    g["G-COVER"] = g["bot_rows"] == len(want) and g["bot_refused"] <= MAX_REFUSED * len(want)
    g["G-TOKENS"] = all(bot[i]["tokens_ok"] for i in ok)
    g["G-BOT-SIDE"] = all(bot[i]["fwd_rows"][1 if dec[i]["side"] == "p1" else 0] == 0 for i in ok)
    g["G-BOT-NAME"] = all(bot[i]["bot"] == P.bot_name(P.E.bank_states()[i]) and bot[i]["bot_seed"] == P.bot_seed(i)
                          for i in ok)
    mism, chars, states_bad = 0, 0, 0
    for i in want:
        s = selfr.get(i)
        if s is None or not s.get("ok"):
            states_bad += 1
            continue
        bad = 0
        for a, v in truth[i]["out"].items():
            w = s["out"].get(a, "")
            chars += len(v[16:40])
            bad += sum(1 for x, y in zip(v[16:40], w) if x != y) + abs(len(v[16:40]) - len(w))
        mism += bad
        states_bad += int(bad > 0)
    g["G-REPRO"] = {"states_replayed": sum(1 for i in want if (selfr.get(i) or {}).get("ok")),
                    "states_differing": states_bad, "chars": chars, "chars_differing": mism,
                    "pass": states_bad == 0 and chars > 0}
    g["pass"] = bool(g["G-COVER"] and g["G-TOKENS"] and g["G-BOT-SIDE"] and g["G-BOT-NAME"] and g["G-REPRO"]["pass"])
    return g


def read() -> dict:
    E = P.E
    E.set_bank("interim")
    meta = E.bank_states()
    ids = [i for i in E.mode_ids("lean") if meta[i]["stratum"] == "bot"]
    strata = {i: meta[i]["stratum"] for i in E.mode_ids("lean")}
    battle = {i: meta[i]["battle"] for i in E.mode_ids("lean")}
    truth = E.read_dir(E.rows_dir(ANCHOR, "O"), "shard_*.jsonl")
    roots = E.read_dir(E.d("roots"), f"{ANCHOR}.jsonl")
    bot_all = E.read_dir(P.ARCH / "rows" / "bot", "rows.jsonl")
    self_all = E.read_dir(P.ARCH / "rows" / "self", "rows.jsonl")
    from main.policy_spectrum.bank import load_bank

    dec = {d["id"]: d for d in load_bank(E.bank_path()).decisions}
    res: dict = {"anchor": ANCHOR, "bank": "interim LEAN", "stratum": "bot", "seed": SEED, "draws": NB,
                 "primary": list(PRIMARY), "level_primary": LEVEL_P}
    res["guards"] = guards(ids, truth, bot_all, self_all, dec)
    bot_ok = {i: r for i, r in bot_all.items() if r.get("ok") and i in ids}
    sub = Sub(bot_ok)
    sub.mode = "self"
    c_self = all_cells(sub)
    p_self = list(sub.picks)
    res["G-IDENT"] = g_ident(c_self, strata)
    sub.picks = []
    sub.mode = "bot"
    c_bot = all_cells(sub)
    res["G-PICKS"] = {"picks": len(p_self), "pass": p_self == sub.picks}
    res["VOID"] = not (res["guards"]["pass"] and res["G-IDENT"]["pass"] and res["G-PICKS"]["pass"])
    bt = Boot([battle[i] for i in bot_ok])
    cells = {}
    for name in sorted(c_self):
        a = {i: v for i, v in c_self[name].items() if i in bot_ok}
        b = {i: v for i, v in c_bot.get(name, {}).items() if i in bot_ok}
        c = cell_read(a, b, battle, bt, LEVEL_P if name in PRIMARY else LEVEL_S)
        if c is not None:
            cells[name] = c
    res["primary_cells"] = {n: cells[n] for n in PRIMARY}
    res["primary_disagrees"] = {n: disagrees(cells[n]) for n in PRIMARY}
    res["secondary_cells"] = {n: c for n, c in cells.items() if n not in PRIMARY}
    res["secondary_disagrees"] = sorted(n for n, c in res["secondary_cells"].items() if disagrees(c))
    ms = [(c["self"]["mean"], c["bot"]["mean"]) for c in cells.values()]
    res["rank_agreement"] = {
        "T_over_actions": t_agreement(truth, bot_ok, roots),
        "cells_spearman_of_means": round(spearman(np.array([m[0] for m in ms]), np.array([m[1] for m in ms])), 4),
        "cells": len(ms)}
    res["verdict"] = "VOID" if res["VOID"] else verdict(res["primary_cells"])
    return res


def selftest() -> None:
    """The rules on planted data: a planted instrument shift is RELABEL-NEEDED, an identity is REFEREE-AGREES."""
    rng = np.random.default_rng(1)
    n = 150
    battle = {f"s{i}": f"b{i // 2}" for i in range(n)}
    bt = Boot(list(battle.values()))
    base = {k: float(rng.choice([0.0, 0.0, 0.0, 0.05, -0.03])) for k in battle}
    same = cell_read(base, dict(base), battle, bt, LEVEL_P)
    assert not disagrees(same) and inside_gross(same)
    shifted = cell_read(base, {k: v - 0.04 for k, v in base.items()}, battle, bt, LEVEL_P)
    assert disagrees(shifted)
    small = cell_read(base, {k: v - 0.001 for k, v in base.items()}, battle, bt, LEVEL_P)
    assert not disagrees(small)            # excludes 0 but |D| < the game bar
    pos = {k: 0.03 + 0.01 * rng.standard_normal() for k in battle}
    flip = cell_read(pos, {k: -v for k, v in pos.items()}, battle, bt, LEVEL_P)
    assert flip["class"] == {"self": "POSITIVE", "bot": "NEGATIVE"} and disagrees(flip)
    noisy = cell_read(base, {k: v + 0.2 * rng.standard_normal() for k, v in base.items()}, battle, bt, LEVEL_P)
    assert verdict({"x": same}) == "REFEREE-AGREES" and verdict({"x": same, "y": shifted}) == "RELABEL-NEEDED"
    assert verdict({"x": same, "y": noisy}) in ("UNDECIDED", "RELABEL-NEEDED")
    assert spearman(np.array([1, 2, 3.0]), np.array([3, 2, 1.0])) == -1.0
    print("selftest: pass")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    res = read()
    text = json.dumps(res, indent=1, sort_keys=True, default=float)
    (HERE / "result_e10.json").write_text(text + "\n")
    print("verdict:", res["verdict"], "| sha256", hashlib.sha256(text.encode()).hexdigest()[:12])


if __name__ == "__main__":
    main()
