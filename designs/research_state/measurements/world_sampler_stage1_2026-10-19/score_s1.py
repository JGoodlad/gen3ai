"""STAGE 1's TABLE (``PREREGISTRATION.md`` §6, part 1): per state, the referee T (the TRUE world's held-out seeds 16..39
at X26 225M), and for every source (the truth O; the banked F and FN; every stage-1 arm; F_rolls likelihood-weighted)
whether it holds a valid row and each scorer's IMP; F_legal's belief-dose curves (M8: the true world at weight p,
F_legal's worlds at 1 − p) that TODAY reads; and per arm the WORLD meters from the world rows (the exact-world mass on
U and M, the never-seen species hit, the near-copy rate). Pure numpy over the archives; Tier 1's ``t1_data``
(``truth_arrays``, ``world_arrays``, ``scorer_values``, ``pick``) and ``score_blind.dk_family`` are IMPORTED, never
copied, so every number is computed as Tier 1 and the maturity battery computed theirs.

    python score_s1.py [--partial]      -> <archive>/score/table.pkl (+ table.sha256); --partial prints counts only

A state enters the table only when it is COMPLETE: every arm holds a play row (ok, an exclusion or a defect). No outcome
is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1_common as E  # noqa: E402

for _p in (E.T1, E.BRC, E.PILOT):
    if str(_p) not in sys.path:
        sys.path.insert(1, str(_p))
if str(E.REPO / "src") not in sys.path:
    sys.path.insert(1, str(E.REPO / "src"))

import score_blind as SB  # noqa: E402  (Tier 1's dose family)
import t1_data as D  # noqa: E402  (Tier 1's per-state arrays)

KEEP = ("LEAF-K8", "LEAF-K4", "TRUNC-4-K8", "CRITIC")
M8 = ("M8|LEAF-K8", "M8|TRUNC-4-K8")
NEAR = 4


def canon(m: str) -> str:
    return "hiddenpower" if m.startswith("hiddenpower") else m


def mons(team: str) -> Dict[str, frozenset]:
    from agents.info_levers.packed import parse_set, species_id, typed_moves

    out = {}
    for e in team.split("]"):
        f = parse_set(e)
        out[species_id(f)] = frozenset(canon(m) for m in typed_moves(f))
    return out


def world_meters(wrow: dict, true_team: str) -> dict:
    """Per arm and state, over its 8 worlds: the share exact on U (the never-seen species SET) and on M (every seen
    mon's whole move set, modulo the Hidden Power type), on both (q_SM, the exact-world mass), the never-seen species
    hit (set-based), and the share of worlds sharing >= 4 mon sets with the true team (a near copy)."""
    tru = mons(true_team)
    seen = set(wrow.get("revealed") or {})
    u_true = frozenset(s for s in tru if s not in seen)
    u_ex, m_ex, um, hit, near = [], [], [], [], []
    for t in wrow["teams"]:
        w = mons(t)
        u_w = frozenset(s for s in w if s not in seen)
        ue = u_w == u_true
        me = all(w.get(s) == tru[s] for s in seen if s in tru)
        u_ex.append(ue)
        m_ex.append(me)
        um.append(ue and me)
        if u_true:
            hit.append(len(u_w & u_true) / len(u_true))
        near.append(sum(1 for s, mv in w.items() if tru.get(s) == mv) >= NEAR)
    out = {"U_exact": float(np.mean(u_ex)), "M_exact": float(np.mean(m_ex)), "qSM": float(np.mean(um)),
           "hit": float(np.mean(hit)) if hit else float("nan"), "near4": float(np.mean(near)), "k": len(seen)}
    if wrow.get("log_w"):
        from main.value_search.rules import world_weights

        wt = world_weights(wrow["log_w"], E.J)
        out["ess"] = float(1.0 / np.sum(wt ** 2))
        out["qSM_w"] = float(np.dot(wt, np.asarray(um, dtype=np.float64)))
    return out


def weighted_values(Sw: dict, log_w: List[float]) -> Dict[str, np.ndarray]:
    """F_rolls' scorers with each world at its self-normalised likelihood weight (``rules.world_weights``: the same
    function ``search_decision(weights="likelihood")`` uses)."""
    from main.value_search.rules import world_weights

    wt = world_weights(log_w, E.J)
    return {"LEAF-K8": Sw["W"][:, :E.J] @ wt, "TRUNC-4-K8": Sw["Xd"][4][:, :E.J] @ wt,
            "CRITIC": Sw["crit"][:, :E.J] @ wt}


def complete_ids(rows: Dict[str, Dict[str, dict]]) -> List[str]:
    """The registered order's states holding a play row in EVERY arm."""
    return [i for i in E.ordered_ids() if all(i in rows[a] for a in E.ARMS)]


def score(partial: bool = False) -> dict:
    meta = E.mat_states()
    roots = E.read_by_id(E.MAT_ARCH / "roots" / f"{E.ANCHOR}.jsonl")
    truth: Dict[str, dict] = {}
    for p in sorted((E.MAT_ARCH / "rows" / E.ANCHOR / "O").glob("shard_*.jsonl")):
        truth.update(E.read_by_id(p))
    banked = {s: E.read_by_id(E.MAT_ARCH / "rows" / E.ANCHOR / s / "rows_0_of_1.jsonl") for s in ("F", "FN")}
    rows = {a: E.read_by_id(E.rows_path(a)) for a in E.ARMS}
    wrows = {a: E.read_by_id(E.worlds_path(a)) for a in E.ARMS}
    ids = complete_ids(rows)
    order = E.ordered_ids()
    cnt = {"complete": len(ids), "registered": len(order),
           "is_prefix": ids == order[:len(ids)],
           "arms": {a: {"ok": 0, "excluded": 0, "defect": 0, "refused": 0} for a in E.ARMS}}
    if partial:
        return {"counts": cnt}
    from main.policy_spectrum.bank import load_bank

    bank = load_bank(E.MAT_ARCH / "bank")
    dec = {d["id"]: d for d in bank.decisions}
    import l2_s1 as L2

    fit = json.loads((E.HERE / "alpha_fit.json").read_text())
    llu = L2.LLU(fit["ALPHA"], {r["id"]: r for r in E.read_rows(E.MAT_ARCH / "fn_belief" / f"{E.ANCHOR}.jsonl")})
    recs = []
    for did in ids:
        m, rt, r1 = meta[did], roots[did], truth[did]
        rec = {"id": did, "battle": m["battle"], "tie": bool(rt["tie"]), "src": {}, "dk": {}, "worlds": {},
               **{k: m.get(k) for k in ("k", "k_bin", "turn", "turn_bin", "material", "mat_bin", "stratum")}}
        if not r1.get("ok"):
            rec["src"]["O"] = {"ok": False}
            recs.append(rec)
            continue
        S1 = D.truth_arrays(r1, {"logits": rt["logits"], "top": rt["top"]})
        Tv, t1 = S1["T"], S1["top1"]
        rec["n_legal"] = len(S1["acts"])

        def put(name: str, vals: Dict[str, np.ndarray]) -> None:
            rec["src"][name] = {"ok": True, "imp": {z: float(Tv[D.pick(v, S1["order"])] - Tv[t1])
                                                    for z, v in vals.items() if z in KEEP}}

        put("O", D.scorer_values(S1))
        from main.policy_spectrum.truth import to_log

        b = bank.battles[bank.battle_index[dec[did]["battle"]]]
        true_team = to_log(b)["teams"][1 if dec[did]["side"] == "p1" else 0]
        for s in ("F", "FN"):
            rw = banked[s].get(did)
            if rw is not None and rw.get("ok"):
                put(s, D.scorer_values(D.world_arrays(rw, S1["acts"], S1["turn"])))
            else:
                rec["src"][s] = {"ok": False}
        for a in E.ARMS:
            rw = rows[a][did]
            c = cnt["arms"][a]
            if not rw.get("ok"):
                c["defect" if rw.get("defect") else "excluded" if rw.get("excluded") else "refused"] += 1
                rec["src"][a] = {"ok": False, "excluded": bool(rw.get("excluded")), "defect": bool(rw.get("defect")),
                                 "error": str(rw.get("error"))[:200]}
                continue
            c["ok"] += 1
            Sw = D.world_arrays(rw, S1["acts"], S1["turn"])
            put(a, D.scorer_values(Sw))
            wr = wrows[a].get(did)
            if wr is not None and wr.get("ok"):
                rec["worlds"][a] = world_meters(wr, true_team)
                # AMENDMENT A1 (L2): the arm's calibration proxies on its own worlds / its species proposal
                rec["worlds"][a]["ll8"] = L2.ll8(wr["teams"], true_team, wr.get("revealed") or {})
                rec["worlds"][a]["llu"] = llu(a, did, true_team, wr.get("revealed") or {})
                rec["worlds"][a]["prop"] = wr.get("prop")
                rec["worlds"][a]["cost_build_cpu_s"] = wr["cost"]["cpu_s"]
            rec["worlds"].setdefault(a, {})["cost_play_cpu_s"] = (rw["cost"]["play"]["cpu_s"]
                                                                  + rw["cost"]["crit"]["cpu_s"])
            if a == "F_rolls":
                if list(rw.get("log_w") or []) != list(wr.get("log_w") or []):
                    raise SystemExit(f"{did}: F_rolls' play row and world row carry different weights")
                put("F_rolls_w", weighted_values(Sw, rw["log_w"]))
            if a == "F_legal":
                fam = SB.dk_family(S1, Sw, None)
                rec["dk"]["F_legal"] = {k: np.asarray(fam[k], dtype=np.float64) for k in M8}
        recs.append(rec)
    return {"counts": cnt, "recs": recs, "ids_sha256": hashlib.sha256("".join(ids).encode()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    tab = score(a.partial)
    if a.partial:
        print(json.dumps(tab["counts"], indent=1, sort_keys=True))
        return
    b = pickle.dumps(tab, protocol=4)
    out = E.d("score") / "table.pkl"
    out.write_bytes(b)
    sha = hashlib.sha256(b).hexdigest()
    (E.d("score") / "table.sha256").write_text(f"{sha}  table.pkl\n")
    print("table:", out, sha, json.dumps(tab["counts"], sort_keys=True))


def load_table() -> dict:
    p = E.d("score") / "table.pkl"
    b = p.read_bytes()
    want = (E.d("score") / "table.sha256").read_text().split()[0]
    if hashlib.sha256(b).hexdigest() != want:
        raise SystemExit(f"{p} does not match its sha256 file")
    return pickle.loads(b)


def first_ok(x: Optional[dict]) -> bool:
    return bool(x and x.get("ok"))


if __name__ == "__main__":
    main()
