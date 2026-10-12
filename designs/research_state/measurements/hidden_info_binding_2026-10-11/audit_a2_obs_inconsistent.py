"""A2 follow-up (no outcome read): the PLAYED worlds whose SEARCHED side's protocol differs from the truth's (A2's
``searched_protocol_kinds``). Per state: which kinds, how many of its 8 worlds, and for a Focus Punch tightening
(``-singleturn ... move: Focus Punch``) the mon, whether ``determinize.reveals`` counted Focus Punch as seen, and the
mon's condition at the root (the opponent's last request: alive or fainted), i.e. whether the world's wrong set can
matter in the playouts. Also: every public line kind in the searched side's prefix that names an opponent move
outside ``|move|`` lines (what ``reveals`` does not parse), counted over ALL played states.

    run_at_pin.sh <cp> 1 4 30m audit_a2_obs_inconsistent.py
→ here ``audit_a2_obs_inconsistent_<cp>.json``.
"""
import collections
import json

import audit_lib as A

C, D = A.C, A.D


def main():
    c = A.cp()
    from agents.battle.core_replay import run_core
    from main.policy_spectrum.truth import cmd_index

    bk = A.bank(c)
    out = {"audit": "A2 follow-up: worlds inconsistent with the searched side's protocol", "cp": c, "by_dir": {}}
    unparsed = collections.Counter()
    unparsed_ex = collections.defaultdict(list)
    done_unparsed = set()
    for det_dir in A.AUDIT_DIRS:
        rows = C.read_rows(A.audit_dir(c) / f"a2_rows_{det_dir}.jsonl")
        W = A.world_rows(c, det_dir)
        states = []
        for r in rows:
            if not r.get("ok"):
                continue
            ws = [w for w in r["worlds"] if w.get("searched_protocol_kinds")]
            d, b = A.decision(bk, r["id"])
            side = d["side"]
            opp = "p2" if side == "p1" else "p1"
            v = 0 if side == "p1" else 1
            at = cmd_index(b, side, d["n"])
            need_replay = bool(ws) or r["id"] not in done_unparsed
            if not need_replay:
                continue
            rt = run_core([A.rec(b, "t", opp, None, b.commands[:at])], trackers=True, obs=True)[0]
            lines = [ln for s, ls in rt["chunks"] if s == v for ln in ls]
            oprefix = f"p{2 - v}a: "
            if r["id"] not in done_unparsed:
                done_unparsed.add(r["id"])
                for ln in lines:
                    f = ln.split("|")
                    if len(f) > 2 and f[1] != "move" and f[2].startswith(oprefix) and "move: " in ln:
                        key = f"{f[1]}|{ln.split('move: ')[1].split('|')[0]}"
                        unparsed[f[1]] += 1
                        if len(unparsed_ex[f[1]]) < 5:
                            unparsed_ex[f[1]].append({"id": r["id"], "line": ln[:160], "key": key})
            if not ws:
                continue
            kinds = collections.Counter(k for w in ws for k in w["searched_protocol_kinds"])
            fp = [ln for ln in lines if ln.startswith("|-singleturn|" + oprefix) and "Focus Punch" in ln]
            req = [ln for s, ls in rt["chunks"] if s == 1 - v for ln in ls if ln.startswith("|request|")]
            cond = {}
            if req:
                for p in json.loads(req[-1][len("|request|"):])["side"]["pokemon"]:
                    cond[D.toid(p["details"].split(",")[0])] = p["condition"]
            wr = W[r["id"]]
            fp_mons = sorted({D.toid(ln.split("|")[2][len(oprefix):]) for ln in fp})
            states.append({"id": r["id"], "turn": r["turn"], "worlds_inconsistent": len(ws), "kinds": dict(kinds),
                           "focus_punch_tightening_seen_by_searched_side": fp_mons,
                           "focuspunch_counted_as_revealed": {m: "focuspunch" in set(wr["revealed"].get(m, [])) for m in fp_mons},
                           "root_condition_of_those_mons": {m: cond.get(m) for m in fp_mons},
                           "notes": [w["notes"][0][:220] for w in ws[:2]]})
        fp_states = [s for s in states if s["kinds"].get("-singleturn")]
        out["by_dir"][det_dir] = {
            "states_with_inconsistent_worlds": len(states),
            "worlds_inconsistent": sum(s["worlds_inconsistent"] for s in states),
            "states_by_kind": dict(collections.Counter(k for s in states for k in s["kinds"])),
            "focus_punch_states": len(fp_states),
            "focus_punch_states_mon_alive_at_root": sum(1 for s in fp_states
                                                        if any(v_ and "fnt" not in v_ for v_ in s["root_condition_of_those_mons"].values())),
            "states": states,
        }
    out["public_lines_naming_an_opponent_move_outside_move_lines (all played states, det and det_b)"] = dict(unparsed)
    out["examples_of_such_lines"] = dict(unparsed_ex)
    A.write_json(f"audit_a2_obs_inconsistent_{c}.json", out)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "states"} for k, v in out["by_dir"].items()}, indent=1))
    print(json.dumps(out["public_lines_naming_an_opponent_move_outside_move_lines (all played states, det and det_b)"]))


if __name__ == "__main__":
    main()
