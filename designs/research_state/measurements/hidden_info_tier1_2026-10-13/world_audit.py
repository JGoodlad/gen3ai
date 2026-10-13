"""R-GIGO FOR THE TIER-1 WORLDS (registration §12.1 with Amendment 1 (d); no outcome is read). Over EVERY stored world
row of every arm (HEAD; the information-lever audit suite's own checks, `main.info_audit.checks`, applied to the rows):

  * ``provenance`` per spec arm (a fact told without a lever, a mass-1 truth lever not the truth, a revealed fact not
    the truth, a placebo equal to the truth): 0 violations;
  * ``leak:moves:seen:drawn`` (a seen mon's DRAWN moves hit the truth no more often than their exact CP inclusion,
    z ≤ 4) and ``dose:moves:seen:told`` (a told move lever hits exactly as often as its mixture says, |z| ≤ 4), per arm;
  * ``legality``: no world breaks a gen3ou rule the true team does not (`worlds.team_problems`, re-checked here);
  * the PAD-PIN re-gate (`pin_regate_<cp>_<code>.json`: every sampled world passes the strict gate at the engine that
    plays it, the planted item swap caught) and the pipeline IDENTITY (`identity_V_oracle.json`);
  * exclusions per arm (a state the builder could not fit), with their reasons; F's share against the 10 % rule (§6.1).

    run_head.sh 8 2h world_audit.py [--out world_audit.json]

R-GIGO (Tier 1) = PASS iff every check passes. The descriptive per-class truth-match tables are written beside.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_common as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="world_audit.json")
    a = ap.parse_args()
    from agents.info_levers.consumers import SEARCH
    from main.info_audit import checks as CK
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import to_log
    from main.value_search.worlds import team_problems

    key = T.load_key()
    res = {"arms": {}, "pin_regate": {}, "identity": None, "checks": []}
    for cp in T.CPS:
        banks = {w: load_bank(T.bank_path(cp, w)) for w in ("registered", "supp")}
        true_team = {}
        for w, bk in banks.items():
            for d in bk.decisions:
                b = bk.battles[bk.battle_index[d["battle"]]]
                lg = to_log(b)
                true_team[d["id"]] = lg["teams"][1 if d["side"] == "p1" else 0]
        for lab, arm in T.ARMS.items():
            code = key[lab]
            rows = T.read_dir(T.worlds_dir(cp, code), "worlds_*.jsonl")
            st = res["arms"].setdefault(lab, {"rows": 0, "ok": 0, "excluded": 0, "crash": 0, "reasons": {},
                                              "provenance_violations": 0, "legality_violations": 0,
                                              "padded_worlds": 0, "worlds": 0, "tries_mean": None,
                                              "build_cpu_s": 0.0, "_tries": []})
            lk = st.setdefault("_leak", CK.Leakage())
            spec = SEARCH.validate(arm["spec"])
            for did, r in rows.items():
                st["rows"] += 1
                st["build_cpu_s"] += r.get("cost", {}).get("cpu_s", 0.0)
                if not r.get("ok"):
                    st["excluded"] += 1
                    st["crash"] += int(bool(r.get("crash")))
                    why = str(r.get("error", ""))
                    tag = ("no honest draw fits" if "no honest draw" in why else "worlds not built in tries"
                           if "not built in" in why else why.split(":")[0][:60])
                    st["reasons"][tag] = st["reasons"].get(tag, 0) + 1
                    continue
                st["ok"] += 1
                st["_tries"].append(sum(r["tries"]) / len(r["tries"]))
                W = SimpleNamespace(facts=r["facts"], samplers=r.get("samplers", []))
                if arm["kind"] == "spec":
                    pv = CK.provenance(spec, W)
                    st["provenance_violations"] += int(pv.get("violations", 0))
                lk.add(W)
                tp = team_problems(true_team[did])
                for tw in r["teams"]:
                    st["worlds"] += 1
                    st["padded_worlds"] += int("*" in tw)
                    if team_problems(tw) - tp:
                        st["legality_violations"] += 1
            for code_file in T.HERE.glob(f"pin_regate_{cp}_{code}.json"):
                res["pin_regate"][f"{cp}:{lab}"] = json.loads(code_file.read_text())
    ok = True
    for lab, st in res["arms"].items():
        lk = st.pop("_leak")
        tr = st.pop("_tries")
        st["tries_mean"] = round(sum(tr) / len(tr), 3) if tr else None
        st["share_excluded"] = round(st["excluded"] / max(1, st["rows"]), 4)
        st["build_cpu_s"] = round(st["build_cpu_s"], 1)
        st["leakage"] = lk.verdicts()
        st["match_table"] = lk.table() if hasattr(lk, "table") else {}
        for v in st["leakage"]:
            res["checks"].append({"arm": lab, **v})
            ok &= bool(v.get("pass", v.get("ok", True)))
        if st["provenance_violations"] or st["legality_violations"] or st["crash"]:
            ok = False
    for k, v in res["pin_regate"].items():
        ok &= v.get("verdict") == "PASS"
    idp = T.HERE / "identity_V_oracle.json"
    res["identity"] = json.loads(idp.read_text()).get("verdict") if idp.exists() else "NOT RUN"
    ok &= res["identity"] == "PASS"
    res["F_excluded_over_10pct"] = res["arms"].get("F", {}).get("share_excluded", 0) > 0.10
    res["verdict"] = "PASS" if ok and res["pin_regate"] else "FAIL"
    (T.HERE / a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"verdict": res["verdict"], "identity": res["identity"],
                      "pin_regate": {k: v.get("verdict") for k, v in res["pin_regate"].items()},
                      "arms": {k: {x: v[x] for x in ("rows", "ok", "excluded", "provenance_violations",
                                                     "legality_violations")} for k, v in res["arms"].items()}},
                     indent=1))


if __name__ == "__main__":
    main()
