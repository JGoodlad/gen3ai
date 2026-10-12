"""ROOT FACTS for the registered descriptive reads (design doc §11, D2-D4 and the §10.4 tags): per registered state,
the TRUE prefix replayed at the checkpoint's pin (``agents.battle.core_replay.run_core``, trackers on), and from the
SEARCHED side's own protocol stream up to its root decision: the opponent species it has seen with the moves it saw
each use (the pilot's ``determinize.reveals``), the opponent's ACTIVE species, Spikes layers on each side, and the
two actives' stat stages (parsed from -boost / -unboost / -setboost / -clearallboost / -clearboost / -clearnegativeboost,
reset on switch / drag: a DESCRIPTIVE tag, not an engine read). No model, no playout, no outcome.

    run_at_pin.sh <cp> 1 6 30m root_facts.py   →  root_facts_<cp>.json (this folder)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "value_disc_pilot_2026-10-11"
sys.path.insert(0, str(PILOT))
import common as C  # noqa: E402
import determinize as D  # noqa: E402

STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


def parse_facts(res, viewer: int, n: int):
    ent = [e for e in res["trackers"][viewer] if "obs" in e]
    after = int(ent[n]["after"])
    me, op = f"p{viewer + 1}", f"p{2 - viewer}"
    active = {me: None, op: None}
    boosts = {me: {}, op: {}}
    spikes = {me: 0, op: 0}
    for side, lines in res["chunks"][: after + 1]:
        if side != viewer:
            continue
        for line in lines:
            p = line.split("|")
            if len(p) < 3:
                continue
            tag = p[1]
            who = p[2][:2] if len(p[2]) >= 2 else ""
            if tag in ("switch", "drag") and len(p) > 3 and who in active:
                active[who] = D.toid(p[3].split(",")[0])
                boosts[who] = {}
            elif tag in ("-boost", "-unboost", "-setboost") and len(p) > 4 and who in boosts:
                st, k = p[3], int(p[4])
                cur = boosts[who].get(st, 0)
                new = k if tag == "-setboost" else cur + (k if tag == "-boost" else -k)
                boosts[who][st] = max(-6, min(6, new))
            elif tag == "-clearallboost":
                boosts = {me: {}, op: {}}
            elif tag == "-clearboost" and who in boosts:
                boosts[who] = {}
            elif tag == "-clearnegativeboost" and who in boosts:
                boosts[who] = {k: v for k, v in boosts[who].items() if v > 0}
            elif tag in ("-sidestart", "-sideend") and len(p) > 3 and "spikes" in D.toid(p[3]):
                s = p[2][:2]
                if s in spikes:
                    spikes[s] = min(3, spikes[s] + 1) if tag == "-sidestart" else 0
    return {"opp_active": active[op], "our_active": active[me],
            "opp_boosts": {k: v for k, v in boosts[op].items() if v},
            "our_boosts": {k: v for k, v in boosts[me].items() if v},
            "spikes_ours": spikes[me], "spikes_theirs": spikes[op]}


def main():
    c = C.cp()
    C.check_pin(c)
    from agents.battle.core_replay import RecordedBattle, run_core
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index

    bank = load_bank(C.out_dir(c) / "bank")
    out = {}
    decs = sorted(bank.decisions, key=lambda x: x["id"])
    recs, meta = [], []
    for d in decs:
        b = bank.battles[bank.battle_index[d["battle"]]]
        at = cmd_index(b, d["side"], d["n"])
        recs.append(RecordedBattle(label=d["id"], format_id=b.format_id, seed=b.seed, p1=dict(b.p1), p2=dict(b.p2),
                                   commands=[list(x) for x in b.commands[:at]]))
        meta.append((d, b))
    res = []
    for i in range(0, len(recs), 32):
        res.extend(run_core(recs[i:i + 32], trackers=True, obs=True))
    for (d, b), r in zip(meta, res):
        if not r["ok"]:
            out[d["id"]] = {"ok": False, "error": str(r.get("error"))[:300]}
            continue
        viewer = 0 if d["side"] == "p1" else 1
        seen, _rows = D.reveals(r, viewer, d["n"])
        team = (b.p2 if d["side"] == "p1" else b.p1)["team"]
        sets = [D.parse_set(e) for e in team.split("]")]
        true = {D.toid(f[1] or f[0]): f[4].split(",") for f in sets}
        unseen = {}
        for sid, mv in seen.items():
            if sid not in true:
                unseen[sid] = None            # a seen species not on the true team: a FINDING (should never happen)
                continue
            unseen[sid] = sum(1 for m in true[sid]
                              if not (m in mv or (m.startswith("hiddenpower") and "hiddenpower" in mv)))
        f = parse_facts(r, viewer, d["n"])
        out[d["id"]] = {"ok": True, "seen": {k: sorted(v) for k, v in seen.items()}, "unseen_moves": unseen,
                        "n_seen": len(seen), "n_unrevealed": len(true) - len(seen), **f,
                        "opp_active_unseen_moves": unseen.get(f["opp_active"])}
    bad = [k for k, v in out.items() if not v.get("ok")]
    miss = [k for k, v in out.items() if v.get("ok") and any(x is None for x in v["unseen_moves"].values())]
    (HERE / f"root_facts_{c}.json").write_text(json.dumps({"cp": c, "states": out, "refused": bad,
                                                           "seen_not_on_team": miss}, sort_keys=True) + "\n")
    print(json.dumps({"cp": c, "states": len(out), "refused": len(bad), "seen_not_on_team": len(miss)}))


if __name__ == "__main__":
    main()
