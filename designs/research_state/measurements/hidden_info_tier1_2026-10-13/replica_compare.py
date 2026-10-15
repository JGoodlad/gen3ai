"""THIRD-PARTY COMPARISON of the hidden-info study's two implementations (registration §12.2 R-REPLICATE, §13.1).

The ORIGINAL Tier 1 (this folder's frozen blinded table + arm key) against the BLIND REPLICA
(`../hidden_info_replica_2026-10-11/`: `imp_per_state.json`, `rows/*.jsonl.gz`, `marginals.json`). Written by a third
party that ran neither. Read-only on both; it changes no verdict of either.

    PYTHONPATH=<worktree>/src python -I replica_compare.py   → replica_comparison.json (this folder)

What it computes (every comparison LIKE WITH LIKE: the same states, one battle-clustered bootstrap for both, pooled =
battles resampled within each checkpoint; the original's `t1_stats.Boot`):
1. IDENTITY: per state, IMP_O (LEAF-K1/K4/K8) of the two implementations (the same truth seeds 0-7 and referee: they
   must be EQUAL); the replica's O rows against the Phase 1 outcome strings; both implementations' IMP re-derived from
   their per-world outcomes by ONE scorer (`score_blind.imp_of`).
2. The COMMON SET C = the registered states where BOTH implementations hold O, P, F, F+M, O-M. On C: IMP per arm, G,
   G_P, sigma_M, nu_M, s_M, n_M, the §12.3 conditions computable from these arms (B2, B3, B4, B8, R1, the M-only cascade),
   each implementation separately and the PAIRED cross-implementation differences.
3. The AGREEMENT criteria (stated in AGREE below before this script was first run).
4. The VIEW: what each implementation says the searched side had seen (moves per seen mon), state by state.
5. MARGINALS on C (§12.2's TV < 0.02), one classification frame (the replica's `seen`) applied to both world sets,
   with a split-half noise floor.
6. TIES and the POST-HOC dose curve: the share of states whose honest LEAF-K8 argmax is tied, and M8 (the TRUE world
   at weight p, the implementation's own F worlds at 1 - p) for BOTH implementations on C (post hoc for the replica:
   it registered no dose arm).
7. The LEARNSET HANDICAP (the replica's move prior filtered on the pre-`64a695dc` table): the missing pairs in the
   true opponent teams by class, the honest worlds' hit rates on those moves, and a difference-in-differences estimate
   of the handicap's effect on IMP_F and IMP_O-M (affected vs unaffected states).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REP = HERE.parent / "hidden_info_replica_2026-10-11"
sys.path.insert(0, str(HERE))
import t1_common as T  # noqa: E402
import t1_data as D  # noqa: E402
import t1_stats as ST  # noqa: E402
import unblind_t1 as U  # noqa: E402
from score_blind import imp_of  # noqa: E402

ARMS = ("O", "P", "F", "F+M", "O-M")
LEAF = "LEAF-K8"
SCORERS = ("LEAF-K1", "LEAF-K4", "LEAF-K8")
GRID = np.array([0.0, 0.001, 0.01, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
LEARNSET_FIX = "64a695dc"
# the replica writes IMP rounded to 6 decimals; a real IMP difference is a multiple of 1/48 (the referee: 24 seeds, a
# tie = 1/2), so "equal" = within 1e-5 (rule 8: no input lies anywhere near this boundary; the max is recorded)
TOL = 1e-5

# Pre-stated agreement criteria (written before the first run of this script; the headline numbers of BOTH READMEs had
# been read, so this is not blind).
AGREE = {
    "brief": "AGREE iff the two implementations' 95 % intervals OVERLAP and every R-rule condition computed from the "
             "quantity reads the SAME (rule 8 applied to each); DISAGREE otherwise; NOT REPLICATED where the replica "
             "has no such quantity",
    "registered_12_2": "§12.2: point estimates of every registered IMP-scale contrast differ by < 0.005 on the same "
                       "states, the per-class marginals' TV < 0.02, and every verdict the same",
    "supplementary": "the PAIRED cross-implementation difference (orig - replica, same states, same draws): its 95 % "
                     "interval contains 0 (descriptive; not a registered criterion)",
}


# The third party's JUDGEMENT per verdict (rule 1 / B1 / §13.2), from the computed reads below; the reasons are in
# replica_comparison.md §8. Kept here so the json carries it beside the numbers it rests on.
RULE1 = {
    "R-GIGO (Tier-1 worlds)": "PARTLY MET (instrument level: truth, referee, scorer identical; views 1164/1164)",
    "G (B2)": "MET",
    "R-MOVES = MOVES-REFUTED (R3)": "PARTLY MET (the M half replicates; R3's U half NOT replicated: no U arms)",
    "R-MOVES B6 / B7 (dose, placebo)": "NOT MET (no D50 / D90 / W90)",
    "R-RIVALS U": "NOT MET",
    "R-RECOVER": "NOT MET",
    "R-EVAL": "NOT MET (LEAF scorers only)",
    "R-STAGE-*": "NOT MET (the k-gradient sub-read DIFFERS)",
    "R-DOSE + the post-hoc tie-break": "PARTLY MET (descriptive: the M8 curve and its step reproduce)",
    "Amendment 3": "NOT MET",
    "FN": "NOT MET",
    "items / abilities / spreads add little (descriptive)": "PARTLY MET (P - F AGREE; BOUNDARY on the §12.2 point rule)",
    "R-REPLICATE (§12.2) as a whole": "NOT MET (nu_M point rule; TV < 0.02 below the noise floor; no replica verdict)",
}


def S(est, vec, n=None):
    return ST.summary(est, vec, n)


def lo(v):
    return ST.ci(v)[0]


def hi(v):
    return ST.ci(v)[1]


# ------------------------------------------------------------------------------------------------ loading

def load_jsonl_gz(p: Path):
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def replica_rows(cp: str, arm: str) -> dict:
    out = {}
    for r in load_jsonl_gz(REP / "rows" / f"{cp}_{arm}.jsonl.gz"):
        if r.get("ok"):
            if r["id"] in out:
                raise RuntimeError(f"replica {cp} {arm} {r['id']}: two ok rows")
            out[r["id"]] = r
    return out


def bank_truth(cp: str) -> dict:
    """{decision id: (side, the opponent's packed team)} from the pilot's bank."""
    bdir = T.PILOT_ARCH / cp / "bank"
    battles = {b["battle_id"]: b for b in load_jsonl_gz(bdir / "battles.jsonl.gz")}
    out = {}
    for d in load_jsonl_gz(bdir / "decisions.jsonl.gz"):
        b = battles[d["battle"]]
        opp = "p2" if d["side"] == "p1" else "p1"
        out[d["id"]] = b[opp]["team"]
    return out


def tid(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def unpack(packed: str):
    out = []
    for chunk in packed.split("]"):
        f = chunk.split("|")
        out.append({"name": f[0], "species": tid(f[1] or f[0]), "item": f[2], "ability": f[3],
                    "moves": [m for m in f[4].split(",") if m], "nature": f[5], "evs": f[6], "ivs": f[8]})
    return out


def base_move(m: str) -> str:
    return "hiddenpower" if m.startswith("hiddenpower") else m


def learnset_missing() -> set:
    root = HERE.parents[3]
    old = json.loads(subprocess.run(["git", "-C", str(root), "show", f"{LEARNSET_FIX}~1:data/pokemon/gen3_learnset.json"],
                                    check=True, capture_output=True, text=True).stdout)
    new = json.loads(subprocess.run(["git", "-C", str(root), "show", f"{LEARNSET_FIX}:data/pokemon/gen3_learnset.json"],
                                    check=True, capture_output=True, text=True).stdout)
    return {(s, m) for s, ms in new.items() for m in ms if m not in set(old.get(s, []))}


# ------------------------------------------------------------------------------------------------ the frame

def build():
    tab, name, sha = U.load()
    rep_imp = json.loads((REP / "imp_per_state.json").read_text())
    states = []
    checks = {"O_identity": {}, "replica_O_vs_phase1": {}, "rescore": {}}
    for cp in T.CPS:
        meta = D.stage_meta(cp)
        r1 = D.phase1_rows(cp, "registered")
        rrows = {a: replica_rows(cp, a) for a in ARMS}
        ow = {a: D.code_worlds(cp, name[a]) for a in ("F", "F+M", "O-M")}
        truth_team = bank_truth(cp)
        o_eq = Counter()
        rO_eq = Counter()
        resc = Counter()
        for rec in tab[cp]["recs"]:
            if rec["set"] != "registered":
                continue
            did = rec["id"]
            ri = rep_imp[cp].get(did, {})
            st = {"cp": cp, "id": did, "battle": rec["battle"], "k": rec["k"], "Tp": rec["Tp"],
                  "orig": {}, "rep": {}, "orig_ok": {}, "rep_ok": {}}
            for a in ARMS:
                src = "O" if a == "O" else ("P" if a == "P" else name[a])
                s = rec["src"].get(src, {})
                st["orig_ok"][a] = bool(s.get("ok"))
                st["orig"][a] = {z: s["imp"][z] for z in SCORERS} if s.get("ok") else None
                st["rep_ok"][a] = f"{a}|{LEAF}" in ri
                st["rep"][a] = {z: ri[f"{a}|{z}"] for z in SCORERS} if st["rep_ok"][a] else None
            if st["orig"]["O"] and st["rep"]["O"]:
                for z in SCORERS:
                    dz = abs(st["orig"]["O"][z] - st["rep"]["O"][z])
                    o_eq[f"{z}_equal" if dz < TOL else f"{z}_differ"] += 1
                    o_eq["max_abs_diff_e6"] = max(o_eq["max_abs_diff_e6"], round(dz * 1e6))
            # truth arrays in policy order (Phase 1)
            S1 = D.truth_arrays(r1[did], meta[did])
            order = np.array(S1["order"])
            acts = S1["acts"]
            st["Tm"] = S1["W"][order][:, :8].mean(1)
            # the replica's O rows against Phase 1 seeds 0-7
            ro = rrows["O"].get(did)
            if ro is not None:
                Wr = D.w(np.array([ro["outcomes"][a] for a in acts]))
                rO_eq["equal" if np.array_equal(Wr, S1["W"][:, :8]) else "differ"] += 1
            # per-world means of each implementation's world arms, policy order
            st["H"] = {"orig": {}, "rep": {}}
            for a, code in (("F", name["F"]), ("F+M", name["F+M"]), ("O-M", name["O-M"])):
                wv = rec["w"].get(code)
                if wv is not None:
                    st["H"]["orig"][a] = wv["W"].astype(np.float64).mean(1)
            for a in ("P", "F", "F+M", "O-M"):
                rr = rrows[a].get(did)
                if rr is not None:
                    if sorted(rr["outcomes"], key=int) != acts:
                        raise RuntimeError(f"replica {cp} {a} {did}: actions differ from Phase 1")
                    Wr = D.w(np.array([rr["outcomes"][x] for x in acts]))
                    st["H"]["rep"][a] = Wr[order].mean(1)
            # re-derive LEAF-K8 IMP with ONE scorer for both implementations
            for who in ("orig", "rep"):
                for a, h in st["H"][who].items():
                    got = float(imp_of(h[None], rec["Tp"])[0])
                    want = st[who][a][LEAF] if st[who][a] else None
                    if want is not None:
                        resc[f"{who}:{a}:{'equal' if abs(got - want) < TOL else 'differ'}"] += 1
                        resc[f"{who}:max_abs_diff_e6"] = max(resc[f"{who}:max_abs_diff_e6"],
                                                             round(abs(got - want) * 1e6))
            # the views and the worlds (for the view check, the marginals and the handicap)
            st["rep_seen"] = (rrows["F"].get(did) or rrows["F+M"].get(did) or {}).get("seen")
            st["orig_revealed"] = None
            st["orig_teams"] = {}
            for a in ("F", "F+M", "O-M"):
                wr = ow[a].get(did)
                if wr and wr.get("ok"):
                    st["orig_teams"][a] = wr["teams"]
                    if st["orig_revealed"] is None:
                        st["orig_revealed"] = wr.get("revealed")
            st["rep_teams"] = {a: [w["team"] for w in rrows[a][did]["worlds"]] for a in ("F", "F+M", "O-M")
                               if did in rrows[a]}
            st["true_team"] = truth_team[did]
            states.append(st)
        checks["O_identity"][cp] = dict(o_eq)
        checks["replica_O_vs_phase1"][cp] = dict(rO_eq)
        checks["rescore"][cp] = dict(resc)
    return states, checks, sha


# ------------------------------------------------------------------------------------------------ reads

def frames(states, sel):
    sub = [s for s, m in zip(states, sel) if m]
    out = {"pooled": (sub, ST.Boot([s["battle"] for s in sub], [s["cp"] for s in sub]))}
    for cp in T.CPS:
        c = [s for s in sub if s["cp"] == cp]
        out[cp] = (c, ST.Boot([s["battle"] for s in c]))
    return out


def arr(sub, who, arm, z=LEAF):
    return np.array([s[who][arm][z] if s[who][arm] else np.nan for s in sub])


def m_read(sub, bt, who, z=LEAF):
    o, p, f, fm, om = (arr(sub, who, a, z) for a in ARMS)
    out = {"IMP": {a: S(*bt.mean(x), int(np.isfinite(x).sum())) for a, x in zip(ARMS, (o, p, f, fm, om))}}
    G = bt.mean(o - f)
    out["G"] = S(*G)
    out["G_P"] = S(*bt.mean(o - p))
    out["P_minus_F"] = S(*bt.mean(p - f))       # what telling the seen mons' items / abilities / spreads buys
    out["sigma_M"] = S(*bt.mean(fm - f))
    out["nu_M"] = S(*bt.mean(o - om))
    sM = bt.ratio(fm - f, o - f)
    nM = bt.ratio(o - om, o - f)
    out["s_M"] = S(*sM)
    out["n_M"] = S(*nM)
    vec = {"IMP": {a: bt.mean(x) for a, x in zip(ARMS, (o, p, f, fm, om))}, "G": G, "G_P": bt.mean(o - p),
           "sigma_M": bt.mean(fm - f), "nu_M": bt.mean(o - om), "s_M": sM, "n_M": nM}
    return out, vec


def conditions(vec, per_cp_vecs):
    """The §12.3 conditions computable from O, F, F+M, O-M (rule 8 via t1_stats)."""
    sM, nM, G = vec["s_M"], vec["n_M"], vec["G"]
    c = {"B2": ST.gt(lo(G[1]), 0.02),
         "B3": ST.ge(sM[0], 0.5) and ST.ge(lo(sM[1]), 0.25),
         "B4": ST.ge(nM[0], 0.5) and ST.ge(lo(nM[1]), 0.25),
         "B8": all(ST.gt(lo(v["sigma_M"][1]), 0.0005) and ST.gt(lo(v["nu_M"][1]), 0.0005) for v in per_cp_vecs),
         "B8_per_cp": {i: {"sigma_lower_gt": ST.gt(lo(v["sigma_M"][1]), 0.0005),
                           "nu_lower_gt": ST.gt(lo(v["nu_M"][1]), 0.0005)} for i, v in zip(T.CPS, per_cp_vecs)},
         "R1": ST.lt(hi(sM[1]), 0.25) and ST.lt(hi(nM[1]), 0.25),
         "s_upper_lt_0.25": ST.lt(hi(sM[1]), 0.25), "s_lower_gt_0": ST.gt(lo(sM[1]), 0.0),
         "n_upper_lt_0.25": ST.lt(hi(nM[1]), 0.25), "n_lower_ge_0.25": ST.ge(lo(nM[1]), 0.25),
         "n_lower_gt_0": ST.gt(lo(nM[1]), 0.0),
         "B8_sigma_both_cps": all(ST.gt(lo(v["sigma_M"][1]), 0.0005) for v in per_cp_vecs),
         "B8_nu_both_cps": all(ST.gt(lo(v["nu_M"][1]), 0.0005) for v in per_cp_vecs),
         "NNS_pattern": ST.ge(lo(nM[1]), 0.25) and ST.lt(hi(sM[1]), 0.25),
         "SHARED_pattern": ST.gt(lo(sM[1]), 0.0) or ST.gt(lo(nM[1]), 0.0)}
    if not c["B2"]:
        v = "NO GAP TO APPORTION"
    elif c["R1"]:
        v = "MOVES-REFUTED (R1)"
    elif c["NNS_pattern"]:
        v = "MOVES-NECESSARY-NOT-SUFFICIENT"
    elif c["SHARED_pattern"]:
        v = "MOVES-SHARED"
    else:
        v = "MOVES-NOT-DETECTED"
    c["M_only_cascade"] = v + " (M-only: R2, R3, B5-B7 need arms the replica did not run)"
    return c


def point_rule(a, b):
    """§12.2's "point estimates differ by < 0.005", rule 8 applied (within ±0.0005 of 0.005 = BOUNDARY)."""
    d = abs(a["mean"] - b["mean"])
    return "AGREE" if d < 0.005 - ST.EDGE else ("DISAGREE" if d > 0.005 + ST.EDGE else "BOUNDARY")


def overlap(a, b):
    return a["ci"][0] <= b["ci"][1] and b["ci"][0] <= a["ci"][1]


def compare(states, sel):
    fr = frames(states, sel)
    res = {"n": {k: len(v[0]) for k, v in fr.items()}, "battles": {k: v[1].nb for k, v in fr.items()}}
    vecs = {}
    for k, (sub, bt) in fr.items():
        res[k] = {}
        vecs[k] = {}
        for who in ("orig", "rep"):
            res[k][who], vecs[k][who] = m_read(sub, bt, who)
        # paired cross-implementation differences on the same draws
        d = {}
        for who in ("orig", "rep"):
            vecs[k][who]["P_minus_F"] = bt.mean(arr(sub, who, "P") - arr(sub, who, "F"))
        for q in ("G", "G_P", "P_minus_F", "sigma_M", "nu_M", "s_M", "n_M"):
            a, b = vecs[k]["orig"][q], vecs[k]["rep"][q]
            d[q] = S(a[0] - b[0], a[1] - b[1])
        for arm in ARMS:
            a, b = vecs[k]["orig"]["IMP"][arm], vecs[k]["rep"]["IMP"][arm]
            d[f"IMP_{arm}"] = S(a[0] - b[0], a[1] - b[1])
        res[k]["orig_minus_rep"] = d
    for who in ("orig", "rep"):
        res["pooled"][who]["conditions"] = conditions(vecs["pooled"][who], [vecs[cp][who] for cp in T.CPS])
    # the agreement table (pooled)
    P = res["pooled"]
    cond_of = {"G": ["B2"], "s_M": ["B3", "s_upper_lt_0.25", "s_lower_gt_0"],
               "n_M": ["B4", "n_upper_lt_0.25", "n_lower_ge_0.25", "n_lower_gt_0"],
               "sigma_M": ["B8_sigma_both_cps"], "nu_M": ["B8_nu_both_cps"], "G_P": [], "P_minus_F": []}
    table = {}
    for q in ("G", "G_P", "P_minus_F", "sigma_M", "nu_M", "s_M", "n_M"):
        a, b = P["orig"][q], P["rep"][q]
        same = all(P["orig"]["conditions"][c] == P["rep"]["conditions"][c] for c in cond_of[q])
        ov = overlap(a, b)
        dd = P["orig_minus_rep"][q]
        row = {"orig": a, "rep": b, "intervals_overlap": ov, "conditions": cond_of[q],
               "conditions_same": same, "brief_criterion": "AGREE" if (ov and same) else "DISAGREE",
               "paired_diff": dd, "paired_diff_ci_contains_0": dd["ci"][0] <= 0 <= dd["ci"][1]}
        if q in ("G", "G_P", "P_minus_F", "sigma_M", "nu_M"):
            row["registered_12_2_point"] = point_rule(a, b)
        table[q] = row
    for arm in ARMS:
        a, b = P["orig"]["IMP"][arm], P["rep"]["IMP"][arm]
        dd = P["orig_minus_rep"][f"IMP_{arm}"]
        table[f"IMP_{arm}"] = {"orig": a, "rep": b, "intervals_overlap": overlap(a, b),
                               "brief_criterion": "AGREE" if overlap(a, b) else "DISAGREE", "paired_diff": dd,
                               "paired_diff_ci_contains_0": dd["ci"][0] <= 0 <= dd["ci"][1],
                               "registered_12_2_point": point_rule(a, b)}
    res["agreement_pooled"] = table
    return res, fr


def per_state_agreement(states, sel):
    sub = [s for s, m in zip(states, sel) if m]
    out = {}
    for a in ("P", "F", "F+M", "O-M"):
        x, y = arr(sub, "orig", a), arr(sub, "rep", a)
        out[a] = {"pearson": round(float(np.corrcoef(x, y)[0, 1]), 4),
                  "share_identical_imp": round(float(np.mean(np.abs(x - y) < 1e-9)), 4)}
    return out


# ------------------------------------------------------------------------------------------------ views

def view_check(states, sel):
    """Seen-mon move sets: the replica's `seen` (by nickname) against the original's `revealed` (by species)."""
    c = Counter()
    examples = []
    for s, m in zip(states, sel):
        if not m or s["rep_seen"] is None or s["orig_revealed"] is None:
            c["not_comparable"] += 1
            continue
        rv = {v["species"]: sorted({base_move(x) for x in v["moves"]}) for v in s["rep_seen"].values()}
        ov = {tid(k): sorted({base_move(x) for x in v}) for k, v in s["orig_revealed"].items()}
        if set(rv) != set(ov):
            c["species_differ"] += 1
            if len(examples) < 5:
                examples.append({"id": s["id"], "rep": rv, "orig": ov})
            continue
        diff = {sp: (rv[sp], ov[sp]) for sp in rv if rv[sp] != ov[sp]}
        if diff:
            c["moves_differ"] += 1
            if len(examples) < 8:
                examples.append({"id": s["id"], "diff_rep_vs_orig": diff})
        else:
            c["equal"] += 1
    return {"counts": dict(c), "examples": examples}


# ------------------------------------------------------------------------------------------------ marginals

def classify(teams, seen):
    tabs = defaultdict(Counter)
    for team in teams:
        for m in unpack(team):
            s = seen.get(m["name"])
            if s is None or s["species"] != m["species"]:
                tabs["U.species"][m["species"]] += 1
                tabs["U.item"][m["item"]] += 1
                tabs["U.ability"][m["ability"]] += 1
                tabs["U.nature"][m["nature"]] += 1
                for mv in m["moves"]:
                    tabs["U.move"][f"{m['species']}:{mv}"] += 1
                continue
            rev = set(s["moves"])
            for mv in m["moves"]:
                if mv in rev:
                    continue
                if mv.startswith("hiddenpower") and "hiddenpower" in rev:
                    tabs["M.bare_hp_type"][mv[len("hiddenpower"):]] += 1
                    continue
                tabs["M.move"][f"{m['species']}:{mv}"] += 1
            if s.get("item") is None:
                tabs["I.item"][f"{m['species']}:{m['item']}"] += 1
            if s.get("ability") is None:
                tabs["A.ability"][f"{m['species']}:{m['ability']}"] += 1
            tabs["S.nature"][f"{m['species']}:{m['nature']}"] += 1
    return tabs


def tv(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return float("nan")
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in set(a) | set(b))


def marginals(states, sel, missing=None):
    out = {}
    for arm in ("F", "F+M", "O-M"):
        tabs = {"orig": defaultdict(Counter), "rep": defaultdict(Counter)}
        halves = {"orig": [defaultdict(Counter), defaultdict(Counter)], "rep": [defaultdict(Counter), defaultdict(Counter)]}
        for s, m in zip(states, sel):
            if not m or s["rep_seen"] is None:
                continue
            h = int(hashlib.sha256(s["battle"].encode()).hexdigest(), 16) % 2
            for who, teams in (("orig", s["orig_teams"].get(arm)), ("rep", s["rep_teams"].get(arm))):
                if not teams:
                    continue
                t = classify(teams, s["rep_seen"])
                for k, v in t.items():
                    tabs[who][k].update(v)
                    halves[who][h][k].update(v)
        if missing is not None:
            # the same tables with every pre-evolution / Sketch pair the replica's table lacked removed (renormalised):
            # what is left is the difference NOT explained by the learnset handicap
            for who in ("orig", "rep"):
                for tb in [tabs[who]] + halves[who]:
                    for k in ("M.move", "U.move"):
                        tb[k + "_excl_missing"] = Counter(
                            {key: n for key, n in tb[k].items()
                             if (key.split(":")[0], base_move(key.split(":")[1])) not in missing})
        res = {}
        for k in sorted(set(tabs["orig"]) | set(tabs["rep"])):
            res[k] = {"TV": round(tv(tabs["orig"][k], tabs["rep"][k]), 4),
                      "noise_floor_split_half_orig": round(tv(halves["orig"][0][k], halves["orig"][1][k]), 4),
                      "noise_floor_split_half_rep": round(tv(halves["rep"][0][k], halves["rep"][1][k]), 4),
                      "draws": [sum(tabs["orig"][k].values()), sum(tabs["rep"][k].values())],
                      "lt_0.02": tv(tabs["orig"][k], tabs["rep"][k]) < 0.02}
        out[arm] = res
    return out


def fm_exact(states, sel):
    """Arm DEFINITION check: in F+M every seen mon's move list should be its TRUE list. Share of seen-mon sets whose
    moves equal the truth's exactly, and equal up to Hidden Power's TYPE (gen 3: the type is the IVs', class S)."""
    out = {}
    for who, key in (("orig", "orig_teams"), ("rep", "rep_teams")):
        c = Counter()
        for s, m in zip(states, sel):
            if not m or s["rep_seen"] is None or "F+M" not in s[key]:
                continue
            true = {x["name"]: x for x in unpack(s["true_team"])}
            for team in s[key]["F+M"]:
                for x in unpack(team):
                    sv = s["rep_seen"].get(x["name"])
                    if sv is None or sv["species"] != x["species"]:
                        continue
                    t = true[x["name"]]["moves"]
                    c["sets"] += 1
                    c["exact"] += set(x["moves"]) == set(t)
                    c["exact_up_to_hp_type"] += {base_move(y) for y in x["moves"]} == {base_move(y) for y in t}
        out[who] = {k: (round(v / c["sets"], 4) if k != "sets" else v) for k, v in c.items()}
    return out


# ------------------------------------------------------------------------------------------------ ties / dose

def ties_and_dose(states, sel):
    fr = frames(states, sel)
    out = {}
    for k, (sub, bt) in fr.items():
        out[k] = {}
        for who in ("orig", "rep"):
            tied = np.array([float(np.sum(s["H"][who]["F"] >= s["H"][who]["F"].max() - D.TIE) > 1) for s in sub])
            cur = np.array([imp_of(GRID[:, None] * s["Tm"][None] + (1 - GRID)[:, None] * s["H"][who]["F"][None],
                                   s["Tp"]) for s in sub])
            res = {"tied_share": S(*bt.mean(tied)),
                   "M8_curve": {f"{p:g}": S(*bt.mean(cur[:, j])) for j, p in enumerate(GRID)},
                   "jump_0_to_0.001": S(*bt.mean(cur[:, 1] - cur[:, 0])),
                   "rise_0.001_to_0.1": S(*bt.mean(cur[:, 3] - cur[:, 1])),
                   "rise_0.1_to_1": S(*bt.mean(cur[:, -1] - cur[:, 3]))}
            out[k][who] = res
            out[k][f"_{who}_curve"] = cur
        d = out[k]["_orig_curve"] - out[k]["_rep_curve"]
        out[k]["orig_minus_rep"] = {f"{p:g}": S(*bt.mean(d[:, j])) for j, p in enumerate(GRID)}
        del out[k]["_orig_curve"], out[k]["_rep_curve"]
    return out


# ------------------------------------------------------------------------------------------------ the handicap

def handicap(states, sel, missing):
    from agents.gen3_data import priors

    c = Counter()
    per_state_aff_seen = []
    per_state_aff_any = []
    hits = {"orig": Counter(), "rep": Counter()}
    pair_count = Counter()
    for s, m in zip(states, sel):
        true = unpack(s["true_team"])
        seen = s["rep_seen"] or {}
        aff_seen = aff_unseen = aff_revealed = False
        hidden_seen_moves = []
        for mon in true:
            sv = seen.get(mon["name"])
            is_seen = sv is not None and sv["species"] == mon["species"]
            rev = {base_move(x) for x in sv["moves"]} if is_seen else set()
            for mv in mon["moves"]:
                miss = (mon["species"], base_move(mv)) in missing
                if miss:
                    pair_count[f"{mon['species']}:{base_move(mv)}"] += 1 if m else 0
                if is_seen:
                    if base_move(mv) in rev:
                        aff_revealed |= miss
                    else:
                        aff_seen |= miss
                        hidden_seen_moves.append((mon["name"], mon["species"], base_move(mv), miss))
                else:
                    aff_unseen |= miss
        if m:
            c["states"] += 1
            c["seen_hidden_missing"] += aff_seen
            c["never_seen_missing"] += aff_unseen
            c["revealed_missing"] += aff_revealed
            c["any_missing_in_true_team"] += bool(aff_seen or aff_unseen or aff_revealed)
            c["hidden_missing_(seen_or_never_seen)"] += bool(aff_seen or aff_unseen)
            # hit rates of the honest F worlds on the seen mons' hidden TRUE moves
            for who, teams in (("orig", s["orig_teams"].get("F")), ("rep", s["rep_teams"].get("F"))):
                if not teams:
                    continue
                for team in teams:
                    wm = {x["name"]: {base_move(y) for y in x["moves"]} for x in unpack(team)}
                    for nm, sp, mv, miss in hidden_seen_moves:
                        key = "missing_pair" if miss else "other"
                        hits[who][f"{key}:slots"] += 1
                        hits[who][f"{key}:hits"] += mv in wm.get(nm, set())
            per_state_aff_seen.append(aff_seen)
            per_state_aff_any.append(bool(aff_seen or aff_unseen))
    hr = {who: {k: (round(h[f"{k}:hits"] / h[f"{k}:slots"], 4) if h[f"{k}:slots"] else None, h[f"{k}:slots"])
                for k in ("missing_pair", "other")} for who, h in hits.items()}
    # Smogon prior mass of the missing pairs held by the true teams (how often an unfiltered draw would hold them)
    mass = {}
    for pr, n in pair_count.most_common(15):
        sp, mv = pr.split(":")
        mass[pr] = {"states": n, "smogon_inclusion": round(float(priors.moves(sp).get(mv, 0.0)), 3)}
    return {"counts": dict(c), "hit_rate_F_seen_hidden_true_moves": hr, "top_pairs_in_true_teams": mass,
            "missing_pairs_total": len(missing)}, np.array(per_state_aff_seen), np.array(per_state_aff_any)


def did(states, sel, aff, arm):
    """Difference in differences of (orig - rep) IMP_arm between affected and unaffected states (pooled frame)."""
    sub = [s for s, m in zip(states, sel) if m]
    bt = ST.Boot([s["battle"] for s in sub], [s["cp"] for s in sub])
    d = arr(sub, "orig", arm) - arr(sub, "rep", arm)
    a = bt.mean(d, aff)
    u = bt.mean(d, ~aff)
    share = float(np.mean(aff))
    return {"affected_n": int(aff.sum()), "unaffected_n": int((~aff).sum()), "share_affected": round(share, 4),
            "orig_minus_rep_affected": S(*a), "orig_minus_rep_unaffected": S(*u),
            "DiD": S(a[0] - u[0], a[1] - u[1]),
            "implied_bias_on_pooled_IMP_rep": S(-share * (a[0] - u[0]), -share * (a[1] - u[1]))}


def main():
    states, checks, sha = build()
    C = np.array([all(s["orig_ok"][a] and s["rep_ok"][a] for a in ARMS) for s in states])
    rep_common = np.array([all(s["rep_ok"][a] for a in ARMS) for s in states])
    out = {"status": "THIRD-PARTY COMPARISON (§12.2 / §13.1); read-only on both implementations",
           "table_sha256": sha, "criteria": AGREE, "rule1_status": RULE1, "checks": checks,
           "sets": {"registered": len(states), "replica_common": int(rep_common.sum()), "C_both": int(C.sum()),
                    "C_by_cp": {cp: int(sum(1 for s, m in zip(states, C) if m and s["cp"] == cp)) for cp in T.CPS},
                    "orig_excluded_inside_replica_common": int((rep_common & ~C).sum())}}
    res, _ = compare(states, C)
    out["common_set"] = res
    # the selection between the published reads: the replica on the states IT built but the original excluded
    xs = rep_common & ~C
    fx = frames(states, xs)
    sub_x, bt_x = fx["pooled"]
    out["replica_only_states"] = {"n": int(xs.sum()),
                                  "rep": m_read(sub_x, bt_x, "rep")[0],
                                  "orig_nu_M": S(*bt_x.mean(arr(sub_x, "orig", "O") - arr(sub_x, "orig", "O-M"))),
                                  "orig_ok_counts": {a: int(sum(s["orig_ok"][a] for s, m in zip(states, xs) if m))
                                                     for a in ARMS}}
    out["per_state"] = per_state_agreement(states, C)
    # the other leaf scorers both implementations computed from the same playouts (reported, not registered)
    fr = frames(states, C)
    # a SUB-READ of R-STAGE-VOI on k (registered on every Tier-1S state; here the registered states of C only, which
    # the replica holds): G by reveal-count bin and the OLS slope of (O - F) on z-scored k, per implementation
    sub, bt = fr["pooled"]
    kz = np.array([s["k"] for s in sub], dtype=np.float64)
    kz = (kz - kz.mean()) / kz.std()
    kb = np.array([T.k_bin(s["k"]) for s in sub])
    out["G_by_k_on_C"] = {}
    for who in ("orig", "rep"):
        g = arr(sub, who, "O") - arr(sub, who, "F")
        beta, bs = bt.ols(g, kz[:, None])
        out["G_by_k_on_C"][who] = {"slope_per_sd": S(float(beta[1]), bs[:, 1]),
                                   "bins": {b: S(*bt.mean(g, kb == b), int((kb == b).sum()))
                                            for b in sorted(set(kb.tolist()))}}
    dg = (arr(sub, "orig", "O") - arr(sub, "orig", "F")) - (arr(sub, "rep", "O") - arr(sub, "rep", "F"))
    beta, bs = bt.ols(dg, kz[:, None])
    out["G_by_k_on_C"]["orig_minus_rep"] = {"slope_per_sd": S(float(beta[1]), bs[:, 1]),
                                            "bins": {b: S(*bt.mean(dg, kb == b)) for b in sorted(set(kb.tolist()))}}
    # the same split for nu_M (O - O-M): where only the seen mons' moves are re-drawn, the handicap acts alone
    dn = arr(sub, "rep", "O-M") - arr(sub, "orig", "O-M")
    out["G_by_k_on_C"]["IMP_O-M_rep_minus_orig"] = {b: S(*bt.mean(dn, kb == b)) for b in sorted(set(kb.tolist()))}
    out["other_scorers_pooled_on_C"] = {
        z: {who: {q: m_read(fr["pooled"][0], fr["pooled"][1], who, z)[0][q] for q in ("G", "s_M", "n_M")}
            for who in ("orig", "rep")} for z in ("LEAF-K1", "LEAF-K4")}
    out["view"] = {"on_C": view_check(states, C),
                   "all_registered": view_check(states, np.ones(len(states), bool))}
    missing = learnset_missing()
    out["marginals_on_C"] = marginals(states, C, missing)
    out["F+M_seen_sets_exact"] = fm_exact(states, C)
    out["ties_and_dose_on_C"] = ties_and_dose(states, C)
    h, aff_seen, aff_any = handicap(states, C, missing)
    h["DiD_IMP_O-M_by_seen_hidden_missing"] = did(states, C, aff_seen, "O-M")
    h["DiD_IMP_F_by_any_hidden_missing"] = did(states, C, aff_any, "F")
    h["DiD_IMP_F+M_by_any_hidden_missing"] = did(states, C, aff_any, "F+M")
    h["DiD_IMP_P_by_any_hidden_missing"] = did(states, C, aff_any, "P")
    out["learnset_handicap_on_C"] = h
    (HERE / "replica_comparison.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    P = res["pooled"]
    print(json.dumps({"sets": out["sets"], "checks": checks, "agreement": res["agreement_pooled"],
                      "orig_cond": P["orig"]["conditions"], "rep_cond": P["rep"]["conditions"]}, indent=1,
                     default=float)[:12000])


if __name__ == "__main__":
    main()
