"""Stress of the lean screen's one non-identity class: a refusal (ok:false) the FULL replay raises AFTER the screen's
stop. Replays deliberately wrong opponent teams (whole teams from other bank battles, and single-slot swaps) through
the lenient screen, full and screened, and compares the gate verdicts battle by battle."""
import json
import re
import random
import sys
import time

from agents.battle import core_replay as CR
from main.value_search import worlds as W

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from probe import gate_of  # noqa: E402


def main(spec, outp):
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.bank_test import BANK_V1
    from main.policy_spectrum.truth import cmd_index, to_log

    bank = B.load_bank(BANK_V1)
    rnd = random.Random(spec.get("seed", 0))
    pool = [t for bt in bank.battles for t in (bt.p1["team"], bt.p2["team"])]
    decs = [d for d in bank.decisions if spec["lo"] <= d["turn"] <= spec["hi"] and d.get("kind") == "free"]
    rnd.shuffle(decs)
    t0 = time.time()
    tot = {"decisions": 0, "battles": 0, "stopped": 0, "full_notok": 0, "lean_notok": 0, "post_stop_refusal": 0,
           "mismatch": 0, "errors": []}
    seen = set()
    for d in decs:
        if time.time() - t0 > spec["budget_s"]:
            break
        if d["battle"] in seen:
            continue
        seen.add(d["battle"])
        bat = bank.battles[bank.battle_index[d["battle"]]]
        log, at = to_log(bat), cmd_index(bat, d["side"], d["n"])
        try:
            root = W.prepare_root(log, at, d["side"], "stress|" + d["id"], False)
        except Exception as e:
            tot["errors"].append(f"prepare {d['id']}: {type(e).__name__}: {str(e)[:120]}")
            continue
        g = gate_of(root.viewer, root.n, root.ref, root.ref_proto, f"p{root.oi + 1}")
        teams = []
        for _ in range(spec["whole"]):
            teams.append(rnd.choice(pool))
        ents = root.team.split("]")
        for _ in range(spec["swap"]):
            other = rnd.choice(pool).split("]")
            i = rnd.randrange(len(ents))
            e = list(ents)
            have = {W.species_id(W.parse_set(x)) for x in ents if x}
            cand = [o for o in other if o and W.species_id(W.parse_set(o)) not in have]
            if not cand:
                continue
            e[i] = rnd.choice(cand)
            teams.append("]".join(e))

        def cmds_of(t):
            return [c + ["if_open"] if c[0] != "forcelose" else c for c in W.render(root.tmpl, t)]
        recs = []
        for k, t in enumerate(teams):
            try:
                recs.append(root.rec(t, f"x{k}", cmds_of(t)))
            except Exception:          # a template slot the substitute team cannot render (fewer moves): skip it
                continue
        if not recs:
            continue
        try:
            full = CR.run_core(recs, trackers=True, obs=True)
            lean = CR.run_core(recs, trackers=True, obs=True, screen={"viewer": root.viewer, "lines": root.ref_proto})
        except RuntimeError as e:
            tot["errors"].append(f"run {d['id']}: {str(e)[:300]}")
            continue
        tot["decisions"] += 1
        for r, rf, rl in zip(recs, full, lean):
            tot["battles"] += 1
            tot["stopped"] += bool(rl.get("stopped"))
            tot["full_notok"] += not rf["ok"]
            tot["lean_notok"] += not rl["ok"]
            if not rf["ok"]:
                k = re.sub(r"[0-9]+", "#", (rf.get("error") or ""))[:90]
                k = "species clause (team construction)" if "Species Clause" in k else k
                tot.setdefault("err_kinds", {})
                tot["err_kinds"][k] = tot["err_kinds"].get(k, 0) + 1
            if not rf["ok"] and rl["ok"]:
                tot["post_stop_refusal"] += 1
            gf, gl = g(rf), g(rl)
            if gf[:3] != gl[:3]:
                tot["mismatch"] += 1
                with open(outp + ".cases", "a") as f:
                    f.write(json.dumps({"decision": d["id"], "at": at, "side": d["side"], "team": r.p2["team"]
                                        if root.oi == 1 else r.p1["team"], "full": gf, "lean": gl,
                                        "full_err": rf.get("error")}) + "\n")
    tot["wall_s"] = round(time.time() - t0, 1)
    with open(outp, "a") as f:
        f.write(json.dumps({"spec": spec, **tot}) + "\n")
    print(json.dumps({"spec": spec, **{k: v for k, v in tot.items() if k != "errors"}, "n_errors": len(tot["errors"]),
                      "errors": tot["errors"][:5], "err_kinds": tot.get("err_kinds")}), flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]), sys.argv[2])
