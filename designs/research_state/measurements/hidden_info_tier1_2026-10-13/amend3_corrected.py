"""AMENDMENT 3's descriptive block, RE-DERIVED after the world-builder record fix (`8702a56f`, ledger GIGO FIX 2026-10-15).

The defect: the builder's never-seen fact RECORDS were paired with the wrong slot after the gender placement, so the
stored ``species_fact.match`` (what ``score_blind.fact_hits`` summed into ``hits[code][2]``) compared a species with the
truth of ANOTHER slot. The worlds themselves (whole packed entries) were right: no playout, IMP, G or share moves; only
the hit-rate proxy ``unblind_t1.hit_proxy`` (-> ``class_dose``) reads the records.

This script, on the FROZEN table (sha checked by ``unblind_t1.load``) and the stored world records:

1. REPRODUCES the published block: the table's ``hits`` equal ``fact_hits`` of the stored records for every state x arm,
   and ``class_dose`` on the unmodified table deep-equals ``result_t1.json``'s "CLASS-DOSE (Amendment 3)";
2. recomputes ``hits[code][2]`` SET-BASED from the stored world teams: the never-seen slots of a world whose species is a
   TRUE never-seen species (the truth's team minus the revealed species), the definition of the fixed
   ``species_fact.match``; ``hits[code][3]`` (the never-seen slot count) is audited equal to the stored one;
3. re-runs ``class_dose`` with only ``hits[code][2]`` replaced and writes both blocks to ``amend3_corrected.json``.

    PYTHONPATH=<this checkout>/src python -I amend3_corrected.py         # reads ~/gen3ai_archive (banks + worlds)

Nothing else is touched; no registered verdict is read or changed.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import t1_common as T  # noqa: E402
import t1_data as D  # noqa: E402
import unblind_t1 as U  # noqa: E402
from score_blind import fact_hits  # noqa: E402

from main.policy_spectrum.bank import load_bank  # noqa: E402
from main.policy_spectrum.truth import to_log  # noqa: E402
from main.value_search.world_draws import parse_set, toid  # noqa: E402


def species_of(team: str) -> list:
    return [toid(p[1] or p[0]) for p in (parse_set(e) for e in team.split("]"))]


def true_species(banks: dict, did: str) -> list:
    for bank in banks.values():
        d = bank.dec_by_id.get(did)
        if d is not None:
            b = bank.battles[bank.battle_index[d["battle"]]]
            return species_of(to_log(b)["teams"][1 if d["side"] == "p1" else 0])
    raise KeyError(did)


def set_based(wr: dict, truth_sp: list) -> tuple:
    """(never-seen slots holding a TRUE never-seen species, never-seen slots), summed over the worlds."""
    rev = set(wr["revealed"])
    true_never = {s for s in truth_sp if s not in rev}
    sh = sn = 0
    for team in wr["teams"]:
        for s in species_of(team):
            if s not in rev:
                sn += 1
                sh += int(s in true_never)
    return sh, sn


def main() -> None:
    tab, name, sha = U.load()
    fresh = copy.deepcopy(tab)
    corrected = copy.deepcopy(tab)
    audit = {"states": 0, "records": 0, "table_hits_equal_fact_hits": 0, "sn_equal": 0, "sh_changed": 0,
             "sh_published_total": 0, "sh_corrected_total": 0, "sn_total": 0}
    bad = []
    for cp in T.CPS:
        banks = {w: load_bank(T.bank_path(cp, w)) for w in ("registered", "supp")}
        for b in banks.values():
            b.dec_by_id = {d["id"]: d for d in b.decisions}
        worlds = {c: D.code_worlds(cp, c) for c in D.coded_dirs(cp)}
        for rec, crec in zip(fresh[cp]["recs"], corrected[cp]["recs"]):
            audit["states"] += 1
            tsp = None
            for code, h in rec["hits"].items():
                wr = worlds[code][rec["id"]]
                audit["records"] += 1
                if fact_hits(wr) == list(h):
                    audit["table_hits_equal_fact_hits"] += 1
                else:
                    bad.append((cp, rec["id"], code, "table hits != fact_hits(stored record)"))
                if not (wr.get("facts") and wr.get("ok", True)):
                    continue
                tsp = tsp or true_species(banks, rec["id"])
                sh, sn = set_based(wr, tsp)
                if sn == h[3]:
                    audit["sn_equal"] += 1
                else:
                    bad.append((cp, rec["id"], code, f"never-seen slots {sn} != stored {h[3]}"))
                audit["sh_published_total"] += h[2]
                audit["sh_corrected_total"] += sh
                audit["sn_total"] += sn
                audit["sh_changed"] += int(sh != h[2])
                crec["hits"][code] = [h[0], h[1], sh, h[3]]
    if bad:
        raise SystemExit(f"AUDIT FAILED ({len(bad)}): {bad[:5]}")

    out = {"table_sha256": sha, "audit": audit}
    for label, t in (("published_reproduced", fresh), ("corrected", corrected)):
        fr_all = U.frames(t, "all")
        out[label] = U.class_dose(fr_all, name)
    pub = json.loads((HERE / "result_t1.json").read_text())["CLASS-DOSE (Amendment 3)"]
    out["reproduction_deep_equal_result_t1"] = (json.loads(json.dumps(out["published_reproduced"], default=str)) == pub)
    (HERE / "amend3_corrected.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    if "--merge" in sys.argv:
        # keep the published block under its own key (superseded, not overwritten); add the corrected one beside it
        res = json.loads((HERE / "result_t1.json").read_text())
        if not out["reproduction_deep_equal_result_t1"]:
            raise SystemExit("refusing to merge: the published block was not reproduced")
        res["CLASS-DOSE (Amendment 3) CORRECTED"] = out["corrected"]
        res["CLASS-DOSE (Amendment 3) CORRECTION NOTE"] = (
            "The block under 'CLASS-DOSE (Amendment 3)' is the PUBLISHED one, superseded by 8702a56f (the builder's "
            "never-seen fact records were mis-keyed). 'CLASS-DOSE (Amendment 3) CORRECTED' is amend3_corrected.py's "
            "re-derivation (frozen table sha " + sha + "). No registered verdict changes.")
        (HERE / "result_t1.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"audit": audit, "reproduction_deep_equal_result_t1": out["reproduction_deep_equal_result_t1"]},
                     indent=1))
    for label in ("published_reproduced", "corrected"):
        a = out[label]["all"]["U"]
        print(label, "U proxy", a["TODAY_proxy"], "p0/PI", a["PI"])
        for ax, cells in out[label]["axes"].items():
            print(" ", ax, {b: c.get("CLASS-FIRST", {}).get("from_today_proxy", {}).get("verdict") for b, c in cells.items()})


if __name__ == "__main__":
    main()
