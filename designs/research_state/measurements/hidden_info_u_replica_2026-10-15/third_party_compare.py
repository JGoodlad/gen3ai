"""THIRD-PARTY COMPARISON: the BLIND U-arm replication (this folder, `hiu/`) against the ORIGINAL Tier 1
(`../hidden_info_tier1_2026-10-13/`), under this folder's PREREGISTRATION.md (`8dd25833`, committed before any arm
output of the blind build existed). Written by a third party that built neither. Read-only on both.

    PYTHONPATH=<worktree>/src python third_party_compare.py   → third_party_comparison.json (this folder)

It mirrors `../hidden_info_tier1_2026-10-13/replica_compare.py` (the earlier third-party comparison, `bc1e203b`):
every comparison LIKE WITH LIKE on the same states, ONE battle-clustered bootstrap for both implementations
(`t1_stats.Boot`: 2,000 draws, seed 20261011, battles resampled within each checkpoint when pooled), every
cross-implementation difference PAIRED on the same draws, rule 8 via `t1_stats` (a bound within ±0.0005 of its
boundary is not clear).

1. INSTRUMENT: IMP_O equal state by state; the blind build's O rows against the Phase 1 outcome strings; both
   implementations' IMPs re-derived from their per-world outcomes by ONE scorer (`score_blind.imp_of`); the
   searched side's view (seen mons and revealed moves) compared state by state.
2. IDENTITIES re-checked from the rows (both implementations): at k = 6, F+U = F and O-U = O outcome for outcome.
3. The COMMON SET C = the registered states where BOTH implementations hold O, F, F+M, F+U, O-M, O-U. On C: IMP per
   arm, G, sigma / nu / s / n for U and M, R3's two numerators (reveal-only F+U - F+M, hide-only O-M - O-U) and its
   registered ratio form; the agreement table; per checkpoint, per opponent population, per both.
4. ROBUSTNESS: C without the Hidden-Power-spelling states (the blind guard's false positive), without the pending
   states (spec choice U1), and the other leaf scorers.
5. TV: per-class marginals of the drawn worlds on C, with a PERMUTATION NULL (implementation labels swapped per
   state; the pre-registration's replacement for §12.2's TV < 0.02), and the slot-placement read (spec choice U2).
6. The LEARNSET HANDICAP: the F worlds' hit rate on the seen mons' hidden true moves, split by the pairs the
   pre-`64a695dc` table lacked (the handicap the earlier comparison found must be gone).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ORIG = HERE.parent / "hidden_info_tier1_2026-10-13"
ROOT = Path(os.path.expanduser("~/gen3ai_archive/hidden_info_u_replica"))
sys.path.insert(0, str(ORIG))
import replica_compare as RC  # noqa: E402
import t1_common as T  # noqa: E402
import t1_data as D  # noqa: E402
import t1_stats as ST  # noqa: E402
import unblind_t1 as U  # noqa: E402
from score_blind import imp_of  # noqa: E402

ARMS = ("O", "F", "F+M", "F+U", "O-M", "O-U")
LEAF = "LEAF-K8"
SCORERS = ("LEAF-K1", "LEAF-K4", "LEAF-K8")
TOL = 1e-5              # the blind build writes IMP rounded to 6 decimals; a real IMP step is a multiple of 1/48
PERM = 2000
PERM_SEED = 20261016
ALPHA = 0.05
P_EDGE = 0.005          # rule 8 for the permutation p: within ±0.005 of 0.05 is BOUNDARY (resolution 1/2001)

# Pre-stated criteria (written before this script first ran; the blind README's Results and the original's
# result_t1.md / red_team.md had been read, so this is NOT blind).
CRITERIA = {
    "registered": "PREREGISTRATION.md: AGREE iff overlapping 95 % intervals (s_U, n_U, R3's two paired numerators) "
                  "AND the same R3 conditions AND |Δ| < 0.005 on sigma_U and nu_U (rule 8 at the boundary)",
    "reading_of_rule_8": "PRIMARY: the study's registered rule 8 (t1_stats.EDGE = 0.0005): an R3 numerator's lower "
                         "bound must exceed +0.0005; |Δ| < 0.0045 AGREE, > 0.0055 DISAGREE, else BOUNDARY (not "
                         "clear, counts as not agreeing). SENSITIVITY: the blind build's reading (a bound within "
                         "±0.005 of 0 is not clear), reported beside it",
    "R3_conditions": "R3 s-side: lower(F+U - F+M) > 0; n-side: lower(O-M - O-U) > 0; R3 = both. Also the "
                     "registered RATIO form (unblind_t1.r_moves: lower((F+U - F+M) / G) > 0 and lower((O-M - O-U) / "
                     "G) > 0) on C, which must read the same",
    "TV": "per class and arm: TV of the pooled draws, its null by swapping the implementation labels of each state's "
          "8 worlds with probability 1/2 (2,000 permutations, seed 20261016); AGREE iff p >= 0.05 (BOUNDARY within "
          "±0.005); Holm-adjusted p reported over the classes of one arm. Teeth: every class an arm keeps TRUE must "
          "read TV 0 exactly",
    "identity_ruling": "an identity FAILS on a state iff both rows exist and an outcome differs; a missing row is an "
                       "exclusion whose CAUSE is ruled on (a defect of the U construction = failure; a defect "
                       "independent of it = exclusion), and the comparison must be robust to dropping the states",
}


# The third party's JUDGEMENTS, from the computed reads below (reasons in third_party_comparison.md). Kept here so the
# json carries them beside the numbers they rest on.
IDENTITY_RULING = {
    "reading": "EXCLUSION (a missing row), NOT an identity failure; the build is VALID",
    "why": [
        "no outcome differs on any k = 6 state where both rows exist: F+U = F on 114 + 94, O-U = O on 129 + 101",
        "the 2 missing O-U rows were refused by the build's consistency guard BEFORE any playout: no outcome exists",
        ("the guard's cause is independent of the U construction: ONE true set in all 1,200 states' banks spells Hidden "
        "Power untyped (prod Charizard, blank IVs = all 31 = Hidden Power Dark); the engine reports it typed "
        "(hiddenpowerdark), the packed truth untyped (hiddenpower). The guard raised on exactly the 4 states holding "
        "that set (F's identity check of the TRUE team on all 4; F+M, O-M, O-U on the 2 at k = 6), and on no other"),
        ("the ORIGINAL played the same 2 states' O-U (the true team, Hidden Power spelled typed) and reproduced the "
        "truth outcome for outcome: the two spellings are the same move"),
        ("the 2 states lie outside the common set anyway (the blind F failed on them by the same guard), so C, and every "
        "number compared, is unchanged by the ruling; dropping all 4 states, or both whole battles, leaves C at 978"),
    ],
    "caveat": "the registered check as scored by hiu.score reads HOLD = false on prod; this ruling reads the "
              "registration's intent (outcome for outcome) and is the third party's, stated openly",
}
RULE1 = {
    "R-MOVES = MOVES-REFUTED (R3), its U half": "REPLICATED (MET) at the registered scorer LEAF-K8 on the registered "
    "mix: the pre-registered criterion AGREES on C. SCOPED: not under the +-0.005-band reading (the original's own "
    "s-side lower bound +0.0020 sits inside it), not at LEAF-K4 / K1 (the R3 conditions flip in opposite directions), "
    "not on end2 alone or on the snapshot-opponent states (R3 fails there in BOTH implementations)",
    "R-MOVES, its M half": "PARTLY MET (B3 fails in both, s_M <= 0.13; but n_M now differs: blind 0.54 holds B4 on C, "
    "original 0.38 fails it; paired difference excludes 0)",
    "R-RIVALS U = U-SHARED": "PARTLY MET (the same verdict on C; B8 differs: the original's end2 sigma_U lower bound "
    "-0.0012)",
    "the opponent-population split (red team's strongest attack)": "REPLICATED (R3 holds on bot states and fails on "
    "snapshot states in both; the bot - snapshot difference is clear on both sides in both)",
    "R-REPLICATE (§12.2) as a whole": "NOT MET (the M side's point rule fails on nu_M / IMP_O-M / sigma_M; the "
    "permutation-null TV test DIFFERS on M.move, S.nature, U.move and the first never-seen slot's species)",
    "Amendment 3, R-STAGE-CLASS, R-DOSE (U parts)": "NOT COMPARED here (outside the pre-registration)",
}
CONFIDENCE = {
    "hidden TEAM information binds honest search (G ~ +0.07)": "~95 % (unchanged; G +0.068 / +0.063 on C)",
    "the moves are THE binding constraint": "~5 % (unchanged; B3 fails in both)",
    "the moves are a material share (n_M >= 0.25)": "~90 % (both lower bounds >= 0.27)",
    "the never-seen Pokemon are the largest single class on the registered mix (LEAF-K8)": "~85 % (was 80 / 70)",
    "... on competent-opponent (snapshot) states": "~30 % (was 35; R3 fails there in both implementations)",
    "U's dominance is a property of the bot-game states": "~80 % (the population split replicates)",
}


def pop_of(did: str) -> str:
    return "snapshots" if did.split("/")[2].startswith("sentinel_") else "bots"


def S(est, vec, n=None):
    return ST.summary(est, vec, n)


def lo(v):
    return ST.ci(v)[0]


def hi(v):
    return ST.ci(v)[1]


# ------------------------------------------------------------------------------------------------ loading

def blind_rows(cp: str, arm: str) -> dict:
    out = {}
    for p in sorted((ROOT / "rows" / cp / arm).glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                old = out.get(r["id"])
                if old is not None and (old.get("ok"), old.get("outcomes")) != (r.get("ok"), r.get("outcomes")):
                    raise RuntimeError(f"blind {cp} {arm} {r['id']}: two rows that disagree")
                out.setdefault(r["id"], r)
    return out


def untyped_hp(packed: str) -> bool:
    return any("hiddenpower" in m["moves"] for m in RC.unpack(packed))


def build():
    tab, name, sha = U.load()
    bimp = json.loads((HERE / "imp_per_state.json").read_text())
    states = []
    chk = {"O_identity": {}, "blind_O_vs_phase1": {}, "rescore": {}, "identities": {}, "pop_label": Counter()}
    for cp in T.CPS:
        meta = D.stage_meta(cp)
        r1 = D.phase1_rows(cp, "registered")
        brows = {a: blind_rows(cp, a) for a in ARMS}
        ow = {a: D.code_worlds(cp, name[a]) for a in ("F", "F+M", "F+U", "O-M", "O-U")}
        truth_team = RC.bank_truth(cp)
        o_eq, bo_eq, resc = Counter(), Counter(), Counter()
        idn = {"blind": Counter(), "orig": Counter(), "blind_fail": [], "orig_fail": [], "blind_missing": [],
               "orig_missing": []}
        for rec in tab[cp]["recs"]:
            if rec["set"] != "registered":
                continue
            did = rec["id"]
            bi = bimp[cp].get(did, {})
            chk["pop_label"][(pop_of(did), rec["opp_class"].startswith("sentinel_"))] += 1
            st = {"cp": cp, "id": did, "battle": rec["battle"], "k": rec["k"], "Tp": rec["Tp"], "pop": pop_of(did),
                  "orig": {}, "blind": {}, "orig_ok": {}, "blind_ok": {}}
            for a in ARMS:
                s = rec["src"].get(name[a], {})
                st["orig_ok"][a] = bool(s.get("ok"))
                st["orig"][a] = {z: s["imp"][z] for z in SCORERS} if s.get("ok") else None
                st["blind_ok"][a] = f"{a}|{LEAF}" in bi
                st["blind"][a] = {z: bi[f"{a}|{z}"] for z in SCORERS} if st["blind_ok"][a] else None
            if st["orig"]["O"] and st["blind"]["O"]:
                for z in SCORERS:
                    dz = abs(st["orig"]["O"][z] - st["blind"]["O"][z])
                    o_eq[f"{z}_equal" if dz < TOL else f"{z}_differ"] += 1
            S1 = D.truth_arrays(r1[did], meta[did])
            order = np.array(S1["order"])
            acts = S1["acts"]
            W1 = S1["W"][order][:, :8]          # every array below is in the POLICY's order (index 0 = top1)
            # the blind build's per-world arrays (its rows are keyed by action; re-ordered to the policy's order)
            Hb = {}
            for a in ARMS:
                rr = brows[a].get(did)
                if rr is not None and rr.get("ok"):
                    if sorted(rr["outcomes"], key=int) != acts:
                        raise RuntimeError(f"blind {cp} {a} {did}: actions differ from Phase 1")
                    Hb[a] = D.w(np.array([rr["outcomes"][x] for x in acts]))[order]
            if "O" in Hb:
                bo_eq["equal" if np.array_equal(Hb["O"], W1) else "differ"] += 1
            # the original's table stores its world arrays in the policy's order already (replica_compare reads
            # them the same way)
            Ho = {"O": W1}
            for a in ("F", "F+M", "F+U", "O-M", "O-U"):
                wv = rec["w"].get(name[a])
                if wv is not None and st["orig_ok"][a]:
                    Ho[a] = wv["W"].astype(np.float64)
            # ONE scorer for both implementations
            for who, H in (("orig", Ho), ("blind", Hb)):
                for a, h in H.items():
                    for z in SCORERS:
                        K = int(z.split("K")[1])
                        got = float(imp_of(h[:, :K].mean(1)[None], rec["Tp"])[0])
                        want = st[who][a][z] if st[who][a] else None
                        if want is not None:
                            resc[f"{who}:{'equal' if abs(got - want) < TOL else 'differ'}"] += 1
            # the registered identities at k = 6, both implementations, from the per-world outcomes
            if rec["k"] == 6:
                for who, H, ok in (("blind", Hb, st["blind_ok"]), ("orig", Ho, st["orig_ok"])):
                    for a, b in (("F+U", "F"), ("O-U", "O")):
                        tag = f"{a}=={b}"
                        if a in H and b in H:
                            same = np.array_equal(H[a], H[b])
                            idn[who][tag + (":equal" if same else ":differ")] += 1
                            if not same:
                                idn[f"{who}_fail"].append([did, tag])
                        else:
                            idn[who][tag + ":missing"] += 1
                            if (a in H) != (b in H):
                                idn[f"{who}_missing"].append([did, tag, f"{a} ok {a in H}", f"{b} ok {b in H}"])
            # views and worlds (marginals, the handicap)
            fr = brows["F"].get(did) or brows["O-U"].get(did) or {}
            st["seen"] = fr.get("seen")
            st["pending"] = bool(fr.get("pending"))
            st["orig_revealed"] = next((ow[a][did].get("revealed") for a in ("F", "F+M", "O-M", "O-U", "F+U")
                                        if did in ow[a] and ow[a][did].get("ok")), None)
            st["orig_teams"] = {a: ow[a][did]["teams"] for a in ow if did in ow[a] and ow[a][did].get("ok")}
            st["blind_teams"] = {a: [w["team"] for w in brows[a][did]["worlds"]] for a in ARMS
                                 if a != "O" and did in brows[a] and brows[a][did].get("ok")}
            st["blind_u_slots"] = fr.get("u_slots")
            st["true_team"] = truth_team[did]
            st["untyped_hp"] = untyped_hp(truth_team[did])
            states.append(st)
        chk["O_identity"][cp] = dict(o_eq)
        chk["blind_O_vs_phase1"][cp] = dict(bo_eq)
        chk["rescore"][cp] = dict(resc)
        chk["identities"][cp] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in idn.items()}
    chk["pop_label"] = {f"{k[0]}|sentinel_label={k[1]}": v for k, v in chk["pop_label"].items()}
    return states, chk, sha


# ------------------------------------------------------------------------------------------------ reads

def frames(states, sel):
    sub = [s for s, m in zip(states, sel) if m]
    out = {"pooled": (sub, ST.Boot([s["battle"] for s in sub], [s["cp"] for s in sub]))}
    for cp in T.CPS:
        c = [s for s in sub if s["cp"] == cp]
        if c:
            out[cp] = (c, ST.Boot([s["battle"] for s in c]))
    return out


def arr(sub, who, arm, z=LEAF):
    return np.array([s[who][arm][z] if s[who][arm] else np.nan for s in sub])


QS = ("G", "sigma_U", "nu_U", "s_U", "n_U", "R3_reveal_only", "R3_hide_only", "R3_ratio_s", "R3_ratio_n",
      "sigma_M", "nu_M", "s_M", "n_M")


def read(sub, bt, who, z=LEAF):
    o, f, fm, fu, om, ou = (arr(sub, who, a, z) for a in ARMS)
    v = {"IMP": {a: bt.mean(x) for a, x in zip(ARMS, (o, f, fm, fu, om, ou))},
         "G": bt.mean(o - f), "sigma_U": bt.mean(fu - f), "nu_U": bt.mean(o - ou),
         "s_U": bt.ratio(fu - f, o - f), "n_U": bt.ratio(o - ou, o - f),
         "R3_reveal_only": bt.mean(fu - fm), "R3_hide_only": bt.mean(om - ou),
         "R3_ratio_s": bt.ratio(fu - fm, o - f), "R3_ratio_n": bt.ratio(om - ou, o - f),
         "sigma_M": bt.mean(fm - f), "nu_M": bt.mean(o - om), "s_M": bt.ratio(fm - f, o - f),
         "n_M": bt.ratio(o - om, o - f)}
    out = {"IMP": {a: S(*x) for a, x in v["IMP"].items()}, **{q: S(*v[q]) for q in QS}}
    return out, v


def conds(v, per_cp=None, band=None):
    """The R3 / R-RIVALS-U conditions computable on C. band=None: rule 8 as registered (gt(x, 0) = x > 0.0005);
    band=0.005: the blind build's reading (a bound within ±0.005 of 0 is not clear)."""
    def pos(x):
        lw = lo(x[1])
        return ST.gt(lw, 0.0) if band is None else (lw is not None and lw > band)
    c = {"R3_s": pos(v["R3_reveal_only"]), "R3_n": pos(v["R3_hide_only"]),
         "R3_ratio_s": pos(v["R3_ratio_s"]), "R3_ratio_n": pos(v["R3_ratio_n"]),
         "B2": ST.gt(lo(v["G"][1]), 0.02),
         "B3_U": ST.ge(v["s_U"][0], 0.5) and ST.ge(lo(v["s_U"][1]), 0.25),
         "B4_U": ST.ge(v["n_U"][0], 0.5) and ST.ge(lo(v["n_U"][1]), 0.25),
         "s_U_lower_gt_0": ST.gt(lo(v["s_U"][1]), 0.0), "n_U_lower_gt_0": ST.gt(lo(v["n_U"][1]), 0.0),
         "s_U_upper_lt_0.25": ST.lt(hi(v["s_U"][1]), 0.25), "n_U_upper_lt_0.25": ST.lt(hi(v["n_U"][1]), 0.25),
         "B3_M": ST.ge(v["s_M"][0], 0.5) and ST.ge(lo(v["s_M"][1]), 0.25),
         "B4_M": ST.ge(v["n_M"][0], 0.5) and ST.ge(lo(v["n_M"][1]), 0.25),
         "n_M_lower_ge_0.25": ST.ge(lo(v["n_M"][1]), 0.25), "s_M_upper_lt_0.25": ST.lt(hi(v["s_M"][1]), 0.25)}
    c["R3"] = c["R3_s"] and c["R3_n"]
    c["R3_ratio"] = c["R3_ratio_s"] and c["R3_ratio_n"]
    if per_cp:
        c["B8_U"] = all(ST.gt(lo(p["sigma_U"][1]), 0.0005) and ST.gt(lo(p["nu_U"][1]), 0.0005) for p in per_cp)
    if c["B2"] and c["B3_U"] and c["B4_U"] and v["s_U"][0] >= v["s_M"][0] and v["n_U"][0] >= v["n_M"][0] \
            and c.get("B8_U", False):
        c["R-RIVALS_U"] = "U-BINDING"
    elif c["s_U_upper_lt_0.25"] and c["n_U_upper_lt_0.25"]:
        c["R-RIVALS_U"] = "U-REFUTED"
    elif c["s_U_lower_gt_0"] or c["n_U_lower_gt_0"]:
        c["R-RIVALS_U"] = "U-SHARED"
    else:
        c["R-RIVALS_U"] = "U-NOT-DETECTED"
    return c


COND_OF = {"s_U": ["B3_U", "s_U_lower_gt_0", "s_U_upper_lt_0.25"],
           "n_U": ["B4_U", "n_U_lower_gt_0", "n_U_upper_lt_0.25"],
           "R3_reveal_only": ["R3_s", "R3"], "R3_hide_only": ["R3_n", "R3"],
           "R3_ratio_s": ["R3_ratio_s"], "R3_ratio_n": ["R3_ratio_n"],
           "sigma_U": ["B8_U"], "nu_U": ["B8_U"], "G": ["B2"],
           "s_M": ["B3_M", "s_M_upper_lt_0.25"], "n_M": ["B4_M", "n_M_lower_ge_0.25"],
           "sigma_M": [], "nu_M": []}
PRIMARY = ("s_U", "n_U", "R3_reveal_only", "R3_hide_only", "sigma_U", "nu_U")


def compare(states, sel, z=LEAF, with_cp=True):
    fr = frames(states, sel)
    res = {"n": {k: len(v[0]) for k, v in fr.items()}, "battles": {k: v[1].nb for k, v in fr.items()}}
    vecs = {}
    for k, (sub, bt) in fr.items():
        res[k] = {}
        vecs[k] = {}
        for who in ("orig", "blind"):
            res[k][who], vecs[k][who] = read(sub, bt, who, z)
        res[k]["orig_minus_blind"] = {q: S(vecs[k]["orig"][q][0] - vecs[k]["blind"][q][0],
                                           vecs[k]["orig"][q][1] - vecs[k]["blind"][q][1]) for q in QS}
        res[k]["orig_minus_blind"].update({f"IMP_{a}": S(vecs[k]["orig"]["IMP"][a][0] - vecs[k]["blind"]["IMP"][a][0],
                                                         vecs[k]["orig"]["IMP"][a][1] - vecs[k]["blind"]["IMP"][a][1])
                                           for a in ARMS})
    per = {who: [vecs[cp][who] for cp in T.CPS if cp in vecs] if with_cp else None for who in ("orig", "blind")}
    for who in ("orig", "blind"):
        res["pooled"][who]["conditions"] = conds(vecs["pooled"][who], per[who])
        res["pooled"][who]["conditions_band_0.005"] = conds(vecs["pooled"][who], per[who], band=0.005)
        for cp in T.CPS:
            if cp in vecs:
                res[cp][who]["conditions"] = conds(vecs[cp][who])
    return res, vecs


R3_CONDS = ("R3_s", "R3_n", "R3", "R3_ratio_s", "R3_ratio_n", "R3_ratio")


def agreement(res):
    """The REGISTERED criterion (PREREGISTRATION.md): every primary quantity's intervals overlap, the R3 conditions
    read the same, and |Δ| < 0.005 on sigma_U and nu_U (rule 8 at 0.005). The other rule conditions computed from a
    quantity (R-RIVALS U's B3 / B4 / B8, the M side's B3 / B4 / knife edge) are SECONDARY: the earlier comparison's
    broader criterion, reported, not part of this registered verdict."""
    P = res["pooled"]
    r3_same = all(P["orig"]["conditions"][c] == P["blind"]["conditions"][c] for c in R3_CONDS)
    r3_same5 = all(P["orig"]["conditions_band_0.005"][c] == P["blind"]["conditions_band_0.005"][c]
                   for c in R3_CONDS)
    table = {}
    for q in QS:
        a, b = P["orig"][q], P["blind"][q]
        ov = RC.overlap(a, b)
        cs = COND_OF[q]
        same = all(P["orig"]["conditions"][c] == P["blind"]["conditions"][c] for c in cs)
        dd = P["orig_minus_blind"][q]
        row = {"orig": a, "blind": b, "intervals_overlap": ov, "paired_diff": dd,
               "paired_diff_ci_contains_0": dd["ci"][0] <= 0 <= dd["ci"][1],
               "secondary_rule_conditions": cs, "secondary_rule_conditions_same": same,
               "secondary_brief_criterion": "AGREE" if (ov and same) else "DISAGREE"}
        if q in ("sigma_U", "nu_U", "sigma_M", "nu_M", "G", "R3_reveal_only", "R3_hide_only"):
            row["abs_delta_point_rule"] = RC.point_rule(a, b)
        if q in PRIMARY:
            ok = ov and r3_same and (q not in ("sigma_U", "nu_U") or row["abs_delta_point_rule"] == "AGREE")
            row["registered_criterion"] = "AGREE" if ok else (
                "BOUNDARY" if ov and r3_same and row.get("abs_delta_point_rule") == "BOUNDARY" else "DISAGREE")
            ok5 = ov and r3_same5 and (q not in ("sigma_U", "nu_U") or row["abs_delta_point_rule"] == "AGREE")
            row["registered_criterion_band_0.005"] = "AGREE" if ok5 else "DISAGREE"
        table[q] = row
    for arm in ARMS:
        a, b = P["orig"]["IMP"][arm], P["blind"]["IMP"][arm]
        dd = P["orig_minus_blind"][f"IMP_{arm}"]
        table[f"IMP_{arm}"] = {"orig": a, "blind": b, "intervals_overlap": RC.overlap(a, b), "paired_diff": dd,
                               "abs_delta_point_rule": RC.point_rule(a, b)}
    prim = {q: table[q]["registered_criterion"] for q in PRIMARY}
    prim5 = {q: table[q]["registered_criterion_band_0.005"] for q in PRIMARY}
    return table, {"primary": prim, "R3_conditions_same": r3_same,
                   "R3_conditions": {who: {c: P[who]["conditions"][c] for c in R3_CONDS} for who in ("orig", "blind")},
                   "verdict": "AGREE" if all(v == "AGREE" for v in prim.values()) else "DISAGREE",
                   "sensitivity_band_0.005": {
                       "primary": prim5, "R3_conditions_same": r3_same5,
                       "R3_conditions": {who: {c: P[who]["conditions_band_0.005"][c] for c in R3_CONDS}
                                         for who in ("orig", "blind")},
                       "verdict": "AGREE" if all(v == "AGREE" for v in prim5.values()) else "DISAGREE"},
                   "secondary_R-RIVALS_U": {who: P[who]["conditions"]["R-RIVALS_U"] for who in ("orig", "blind")}}


def headline(res):
    """The compact per-frame read: R3 numerators and the U / M shares, both implementations."""
    out = {}
    for k in res:
        if k in ("n", "battles"):
            continue
        out[k] = {"n": res["n"][k], "battles": res["battles"][k]}
        for who in ("orig", "blind"):
            out[k][who] = {q: res[k][who][q] for q in ("G", "s_U", "n_U", "s_M", "n_M", "R3_reveal_only",
                                                        "R3_hide_only")}
            if "conditions" in res[k][who]:
                out[k][who]["R3"] = {c: res[k][who]["conditions"][c] for c in ("R3_s", "R3_n", "R3")}
        out[k]["orig_minus_blind"] = {q: res[k]["orig_minus_blind"][q] for q in ("R3_reveal_only", "R3_hide_only",
                                                                                 "s_U", "n_U")}
    return out


def by_population(states, C):
    out = {}
    for pop in ("bots", "snapshots"):
        sel = C & np.array([s["pop"] == pop for s in states])
        res, _ = compare(states, sel, with_cp=False)
        out[pop] = headline(res)
    # the bot - snapshot difference of the absolute numerators, within each implementation (pooled, paired draws)
    sub, bt = frames(states, C)["pooled"]
    isbot = np.array([s["pop"] == "bots" for s in sub])
    diff = {}
    for who in ("orig", "blind"):
        _o, _f, fm, fu, om, ou = (arr(sub, who, a) for a in ARMS)
        d = {}
        for q, x in (("R3_reveal_only", fu - fm), ("R3_hide_only", om - ou)):
            b, s = bt.mean(x, isbot), bt.mean(x, ~isbot)
            d[q] = S(b[0] - s[0], b[1] - s[1])
        diff[who] = d
    out["bots_minus_snapshots"] = diff
    return out


# ------------------------------------------------------------------------------------------------ marginals / TV

def hp(m: str) -> str:
    return "hiddenpower" if m.startswith("hiddenpower") else m


def classify(team: str, seen: dict, u_slots) -> dict:
    """Per-class category lists of ONE world (Hidden Power normalised to its base: its type is a spread fact)."""
    seen_by_name = {n: v for n, v in seen.items()}
    out = defaultdict(list)
    first_u = min(u_slots) if u_slots else None
    for i, m in enumerate(RC.unpack(team)):
        s = seen_by_name.get(m["name"])
        if s is None or s["species"] != m["species"]:
            out["U.species"].append(m["species"])
            out["U.item"].append(m["item"])
            out["U.ability"].append(m["ability"])
            out["U.nature"].append(m["nature"])
            out["U.move"].extend(f"{m['species']}:{hp(x)}" for x in m["moves"])
            if i == first_u:
                out["U.first_slot_species"].append(m["species"])
            continue
        rev = {hp(x) for x in s["moves"]}
        out["M.move"].extend(f"{m['species']}:{hp(x)}" for x in m["moves"] if hp(x) not in rev)
        # the same without Hidden Power (whose presence in a drawn set depends on how a build ties its TYPE to the
        # IVs: the spec's open M-or-S question)
        out["M.move_excl_hp"].extend(f"{m['species']}:{x}" for x in m["moves"]
                                     if hp(x) not in rev and hp(x) != "hiddenpower")
        if s.get("item") is None:
            out["I.item"].append(f"{m['species']}:{m['item']}")
        if s.get("ability") is None:
            out["A.ability"].append(f"{m['species']}:{m['ability']}")
        out["S.nature"].append(f"{m['species']}:{m['nature']}")
    return out


def tv_counts(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def perm_test(A: np.ndarray, B: np.ndarray, rng, cats=None) -> dict:
    """A, B [states, categories]: per-state counts of the two implementations. Null: swap each state's labels
    with probability 1/2 (paired design: under H0 the two world sets of one state are exchangeable)."""
    obs = tv_counts(A.sum(0), B.sum(0))
    Dm = A - B
    sa, sb = A.sum(0), B.sum(0)
    null = np.empty(PERM)
    for r in range(PERM):
        m = rng.random(len(A)) < 0.5
        x = Dm[m].sum(0)
        null[r] = tv_counts(sa - x, sb + x)
    p = (1 + int((null >= obs - 1e-12).sum())) / (PERM + 1)
    return {"TV": round(obs, 4), "null_median": round(float(np.median(null)), 4),
            "null_q95": round(float(np.quantile(null, 0.95)), 4), "p": round(p, 4),
            "draws": [int(sa.sum()), int(sb.sum())], "states": len(A),
            "top_contributors": [] if cats is None else [
                {"cat": cats[i], "orig": round(float(sa[i] / sa.sum()), 4), "blind": round(float(sb[i] / sb.sum()), 4)}
                for i in np.argsort(-np.abs(sa / sa.sum() - sb / sb.sum()))[:8]]}


def marginals(states, C):
    rng = np.random.default_rng(PERM_SEED)
    keep_true = {"F+U": ("U.",), "F+M": ("M.",), "O-M": ("U.", "I.", "A.", "S."), "O-U": ("M.", "I.", "A.", "S.")}
    out = {}
    for arm in ("F", "F+M", "F+U", "O-M", "O-U"):
        per = []
        for s, m in zip(states, C):
            if not m or s["seen"] is None or arm not in s["orig_teams"] or arm not in s["blind_teams"]:
                continue
            co, cb = Counter(), Counter()
            for who, teams, c in (("orig", s["orig_teams"][arm], co), ("blind", s["blind_teams"][arm], cb)):
                for t in teams:
                    for k, v in classify(t, s["seen"], s["blind_u_slots"]).items():
                        c.update((k, x) for x in v)
            per.append((co, cb))
        classes = sorted({k for co, cb in per for (k, _) in list(co) + list(cb)})
        res = {}
        for cl in classes:
            cats = sorted({x for co, cb in per for (k, x) in list(co) + list(cb) if k == cl})
            ix = {c: i for i, c in enumerate(cats)}
            A = np.zeros((len(per), len(cats)))
            B = np.zeros((len(per), len(cats)))
            for i, (co, cb) in enumerate(per):
                for (k, x), n in co.items():
                    if k == cl:
                        A[i, ix[x]] += n
                for (k, x), n in cb.items():
                    if k == cl:
                        B[i, ix[x]] += n
            if A.sum() == 0 or B.sum() == 0:
                continue
            r = perm_test(A, B, rng, cats)
            r["tooth_true_class"] = cl.startswith(keep_true.get(arm, ()))
            r["verdict"] = ("BOUNDARY" if abs(r["p"] - ALPHA) <= P_EDGE else ("AGREE" if r["p"] >= ALPHA
                                                                                else "DIFFER"))
            res[cl] = r
        # Holm over the non-tooth classes of this arm
        test = sorted([(v["p"], k) for k, v in res.items() if not v["tooth_true_class"]])
        m_ = len(test)
        running = 0.0
        for j, (p, k) in enumerate(test):
            running = max(running, min(1.0, (m_ - j) * p))
            res[k]["p_holm"] = round(running, 4)
        out[arm] = res
    return out


# ------------------------------------------------------------------------------------------------ the handicap

def handicap(states, C, missing):
    hits = {"orig": Counter(), "blind": Counter()}
    for s, m in zip(states, C):
        if not m or s["seen"] is None:
            continue
        seen = s["seen"]
        hidden = []
        for mon in RC.unpack(s["true_team"]):
            sv = seen.get(mon["name"])
            if sv is None or sv["species"] != mon["species"]:
                continue
            rev = {hp(x) for x in sv["moves"]}
            for mv in mon["moves"]:
                if hp(mv) not in rev:
                    hidden.append((mon["name"], hp(mv), (mon["species"], hp(mv)) in missing))
        for who, teams in (("orig", s["orig_teams"].get("F")), ("blind", s["blind_teams"].get("F"))):
            for team in teams or []:
                wm = {x["name"]: {hp(y) for y in x["moves"]} for x in RC.unpack(team)}
                for nm, mv, miss in hidden:
                    key = "missing_pair" if miss else "other"
                    hits[who][f"{key}:slots"] += 1
                    hits[who][f"{key}:hits"] += mv in wm.get(nm, set())
    return {who: {k: {"hit_rate": round(h[f"{k}:hits"] / h[f"{k}:slots"], 4) if h[f"{k}:slots"] else None,
                      "slots": h[f"{k}:slots"]} for k in ("missing_pair", "other")} for who, h in hits.items()}


def view_check(states, C):
    c = Counter()
    for s, m in zip(states, C):
        if not m or s["seen"] is None or s["orig_revealed"] is None:
            c["not_comparable"] += 1
            continue
        bv = {v["species"]: sorted({hp(x) for x in v["moves"]}) for v in s["seen"].values()}
        ov = {RC.tid(k): sorted({hp(x) for x in v}) for k, v in s["orig_revealed"].items()}
        c["equal" if bv == ov else "differ"] += 1
    return dict(c)


# ------------------------------------------------------------------------------------------------ main

def main():
    states, chk, sha = build()
    both = np.array([all(s["orig_ok"][a] and s["blind_ok"][a] for a in ARMS) for s in states])
    out = {"status": "THIRD-PARTY COMPARISON under PREREGISTRATION.md (8dd25833); read-only on both builds",
           "table_sha256": sha, "criteria": CRITERIA, "identity_ruling": IDENTITY_RULING, "rule1_status": RULE1,
           "confidence_third_party": CONFIDENCE, "checks": chk}
    out["sets"] = {"registered": len(states), "C_both": int(both.sum()),
                   "C_by_cp": {cp: int(sum(1 for s, m in zip(states, both) if m and s["cp"] == cp)) for cp in T.CPS},
                   "orig_all_six": int(sum(all(s["orig_ok"][a] for a in ARMS) for s in states)),
                   "blind_all_six": int(sum(all(s["blind_ok"][a] for a in ARMS) for s in states)),
                   "untyped_hp_states": int(sum(s["untyped_hp"] for s in states)),
                   "untyped_hp_states_in_C": int(sum(s["untyped_hp"] for s, m in zip(states, both) if m)),
                   "pending_states_in_C": int(sum(s["pending"] for s, m in zip(states, both) if m)),
                   "C_by_pop": {p: int(sum(1 for s, m in zip(states, both) if m and s["pop"] == p))
                                for p in ("bots", "snapshots")}}
    res, _ = compare(states, both)
    out["common_set"] = res
    out["agreement_pooled"], out["verdict"] = agreement(res)
    out["per_checkpoint"] = headline(res)
    out["per_population"] = by_population(states, both)
    out["per_population_x_checkpoint"] = {}
    for pop in ("bots", "snapshots"):
        for cp in T.CPS:
            sel = both & np.array([s["pop"] == pop and s["cp"] == cp for s in states])
            r, _ = compare(states, sel, with_cp=False)
            out["per_population_x_checkpoint"][f"{cp}|{pop}"] = headline(r)["pooled"]
    # robustness: the identity-issue states and the spec choices
    guard_battles = {s["battle"] for s in states if s["untyped_hp"]}
    rob = {}
    for nm, sel in (("drop_untyped_hp_team_states", both & ~np.array([s["untyped_hp"] for s in states])),
                    ("drop_whole_battles_with_untyped_hp", both & ~np.array([s["battle"] in guard_battles
                                                                             for s in states])),
                    ("drop_pending_states_U1", both & ~np.array([s["pending"] for s in states]))):
        r, _ = compare(states, sel)
        _t, v = agreement(r)
        rob[nm] = {"n": int(sel.sum()), "verdict": v,
                   "pooled": {who: {q: r["pooled"][who][q] for q in PRIMARY} for who in ("orig", "blind")},
                   "R3": {who: {c: r["pooled"][who]["conditions"][c] for c in ("R3_s", "R3_n", "R3")}
                          for who in ("orig", "blind")}}
    out["robustness"] = rob
    out["other_scorers_pooled"] = {}
    for z in ("LEAF-K4", "LEAF-K1"):
        r, _ = compare(states, both, z=z)
        _t, v = agreement(r)
        out["other_scorers_pooled"][z] = {"verdict": v, **headline(r)["pooled"]}
    # each implementation on its OWN all-six set (selection differs; descriptive)
    own = {}
    for who in ("orig", "blind"):
        sel = np.array([all(s[f"{who}_ok"][a] for a in ARMS) for s in states])
        r, _ = compare(states, sel)
        own[who] = {"n": int(sel.sum()), **{q: r["pooled"][who][q] for q in PRIMARY},
                    "R3": {c: r["pooled"][who]["conditions"][c] for c in ("R3_s", "R3_n", "R3")}}
    out["own_all_six_sets"] = own
    out["view_on_C"] = view_check(states, both)
    out["marginals_on_C"] = marginals(states, both)
    out["learnset_handicap_F_hit_rates_on_C"] = handicap(states, both, RC.learnset_missing())
    (HERE / "third_party_comparison.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps({"sets": out["sets"], "checks": {k: v for k, v in chk.items() if k != "identities"},
                      "identities": chk["identities"], "verdict": out["verdict"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
