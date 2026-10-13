"""THE k-STRATIFIED SUPPLEMENT's ids (§12.13.2, frozen). Pure Python + numpy; reads NO outcome (pool.npz is opened for
its `ids` and `gap` alone; the census carries no outcome).

    python select_supplement.py end2 prod        # any interpreter with numpy; no engine, no model

The rule, per checkpoint:
  candidates = the eligible pool (census.npz, whose ids equal pool.npz's in order)
               MINUS the registered 600 (`states_<cp>.json`)
               MINUS top-2 ties (pool gap < 1e-6; rule 8);
  the quintile-edge exclusion of the registered sample is NOT applied: the supplement is not stratified by gap;
  per k bin, in the order of `t1_common.K_BINS`: the bin's candidates ordered by sha256("hib_stage_supp_v1:" + id),
  taken in that order, skipping a candidate whose battle already holds 1 supplemental state in THIS bin or 3
  supplemental states in all (counted across the bins processed so far), until the bin holds SUPP_TARGET[cp][bin].
A bin that cannot be filled STOPS the script (the rule is never relaxed).

Writes supplement_<cp>.json (this folder): the rule, the census sha it read, the ids per bin, the counts.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_common as T

TIE_EPS = 1e-6
RULE = ("design_hidden_info_binding.md §12.13.2: from the checkpoint's eligible pool (the last three eval cycles; free "
        "move round, turn 2-40, >= 3 legal, gate ① passed), EXCLUDING the registered 600 and top-2 ties (pool gap < "
        "1e-6, rule 8; the quintile-edge exclusion is NOT applied: the supplement is not stratified by gap), per k bin "
        "in the order 1-2, 3, 4, 5, 6, candidates in sha256('hib_stage_supp_v1:' + id) order, at most 1 supplemental "
        "state per battle per k bin and 3 per battle in all, until the bin holds SUPP_TARGET[cp][bin].")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def select(cp: str) -> dict:
    cen_path = T.ARCH / cp / "census.npz"
    cen_doc = json.loads((HERE / f"census_{cp}.json").read_text())
    sha = sha256_file(cen_path)
    if sha != cen_doc["census_sha256"] or not cen_doc["teeth"]["pass"]:
        raise SystemExit(f"{cp}: census.npz sha {sha} != census_{cp}.json's, or its teeth did not pass")
    cen = np.load(cen_path)
    ids, battle, k = cen["ids"], cen["battle"], cen["k"]
    pool = np.load(T.PILOT_ARCH / cp / "pool.npz")
    pool_ids, gap = pool["ids"], pool["gap"]                 # ids and gap ONLY (no outcome is read)
    if len(ids) != len(pool_ids) or not (ids == pool_ids).all():
        raise SystemExit(f"{cp}: census ids are not pool.npz's ids in order")
    reg = set(T.registered_states(cp))
    if not reg <= set(ids.tolist()):
        raise SystemExit(f"{cp}: a registered id is missing from the census")
    is_reg = np.array([x in reg for x in ids.tolist()])
    tie = gap < TIE_EPS
    cand = ~is_reg & ~tie
    kb = np.array([T.k_bin(x) for x in k])
    per_bin_bat, per_bat = Counter(), Counter()
    chosen, avail = {}, {}
    for b in T.K_BINS:
        want = T.SUPP_TARGET[cp][b]
        idx = sorted(np.flatnonzero(cand & (kb == b)).tolist(), key=lambda i: T.supp_key(str(ids[i])))
        avail[b] = len(idx)
        got = []
        for i in idx:
            if len(got) >= want:
                break
            bid = str(battle[i])
            if per_bin_bat[(b, bid)] >= T.SUPP_PER_BATTLE_PER_BIN or per_bat[bid] >= T.SUPP_PER_BATTLE:
                continue
            per_bin_bat[(b, bid)] += 1
            per_bat[bid] += 1
            got.append(str(ids[i]))
        if len(got) < want:
            raise SystemExit(f"{cp}: k bin {b} holds {len(got)} of {want} under the rule ({len(idx)} candidates); "
                             "STOP (the rule is not relaxed)")
        chosen[b] = sorted(got)
    all_ids = [x for b in T.K_BINS for x in chosen[b]]
    assert len(all_ids) == len(set(all_ids)) and not (set(all_ids) & reg)
    return {"cp": cp, "run": T.CPS[cp]["run"], "pin": T.CPS[cp]["pin"], "rule": RULE,
            "salt": T.SUPP_SALT, "per_battle_per_bin": T.SUPP_PER_BATTLE_PER_BIN,
            "per_battle": T.SUPP_PER_BATTLE, "target": T.SUPP_TARGET[cp],
            "census_npz": str(cen_path), "census_sha256": sha,
            "pool_size": len(ids), "excluded_registered": int(is_reg.sum()),
            "excluded_ties": int(tie.sum()), "excluded_ties_not_registered": int((tie & ~is_reg).sum()),
            "quintile_edge_exclusion_applied": False,
            "quintile_edge_exclusion_note": "not applied: the supplement is not stratified by gap (pool.npz's "
                                            "`excluded` counts the registered sample's rule-8 edge + tie states)",
            "pool_rule8_excluded_flag_count": int(pool["excluded"].sum()),
            "candidates_per_bin": avail, "selected_per_bin": {b: len(chosen[b]) for b in T.K_BINS},
            "selected": len(all_ids), "battles": len(per_bat),
            "battles_with_n_supp": dict(sorted(Counter(per_bat.values()).items())),
            "ids_per_bin": chosen}


def main():
    cps = sys.argv[1:] or list(T.CPS)
    for cp in cps:
        doc = select(cp)
        (HERE / f"supplement_{cp}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(cp, json.dumps({k: doc[k] for k in ("candidates_per_bin", "selected_per_bin", "selected", "battles",
                                                  "excluded_ties", "excluded_registered")}))


if __name__ == "__main__":
    main()
