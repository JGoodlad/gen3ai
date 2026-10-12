"""TEETH for the v2 builder's two new checks (Amendment 8; no outcome read): planted GENDER defects must fail the gate.
For 10 states per checkpoint with a v2 world (``det2``, world 0):
- G1 "a draw removed": a never-seen mon on a TRUE draw slot gets an explicit gender, so the world costs one turn-0
  gender draw fewer (the pre-root dice shift) — must fail the rows / protocol gate;
- G2 "a seen gender flipped": a SEEN gender-ratio mon gets the explicit OPPOSITE of the gender our side saw — the
  rows cannot see it; the protocol gate must.
Both are replayed through the pin's core exactly as the builder's gate does (rows of every searched-side decision up
to the root, and every protocol line it was sent).

    run_at_pin.sh <cp> 1 4 30m teeth_v2_gender.py   →  teeth_v2_gender_<cp>.json
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import determinize as D  # noqa: E402


def main():
    c = C.cp()
    C.check_pin(c)
    from agents.battle.core_replay import RecordedBattle, run_core
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index

    od = C.out_dir(c)
    bank = load_bank(od / "bank")
    dmap = {d["id"]: d for d in bank.decisions}
    P = D.Priors(Path(os.environ["VDP_PIN"]) / "data" / "pokemon")
    W = {r["id"]: r for p in sorted((od / "det2").glob("worlds_*.jsonl")) for r in C.read_rows(p) if r.get("ok")}
    res = {"cp": c, "G1_draw_removed": {"planted": 0, "caught": 0, "by": {}},
           "G2_seen_gender_flipped": {"planted": 0, "caught": 0, "by": {}}, "examples": []}

    def gate(b, d, at, opp, team, cmds):
        p = {"p1": dict(b.p1), "p2": dict(b.p2)}
        p[opp] = dict(p[opp], team=team)
        r = run_core([RecordedBattle(label="t", format_id=b.format_id, seed=b.seed, p1=p["p1"], p2=p["p2"],
                                     commands=[list(x) for x in cmds])], trackers=True, obs=True)[0]
        if not r["ok"]:
            return None, None
        v = 0 if d["side"] == "p1" else 1
        try:
            return D.reveals(r, v, d["n"])[1], D.protocol_lines(r, v, d["n"])
        except RuntimeError:
            return None, None

    for did in sorted(W):
        if res["G1_draw_removed"]["planted"] >= 10 and res["G2_seen_gender_flipped"]["planted"] >= 10:
            break
        w = W[did]
        d = dmap[did]
        b = bank.battles[bank.battle_index[d["battle"]]]
        opp = "p2" if d["side"] == "p1" else "p1"
        at = cmd_index(b, d["side"], d["n"])
        true_team = getattr(b, opp)["team"]
        ref = gate(b, d, at, opp, true_team, b.commands[:at])
        team, cmds = w["teams"][0], w["cmds"][0]
        assert gate(b, d, at, opp, team, cmds) == ref, f"{did}: the archived v2 world does not pass its own gate"
        ents = team.split("]")
        tids = [D.toid(D.parse_set(e)[1] or D.parse_set(e)[0]) for e in true_team.split("]")]
        seen = set(w["revealed"])
        draws = D.gender_draw_slots(P, true_team)
        plants = []
        g1 = [i for i in sorted(draws) if tids[i] not in seen]
        if g1 and res["G1_draw_removed"]["planted"] < 10:
            plants.append(("G1_draw_removed", g1[0], "M"))
        g2 = [i for i in sorted(draws) if tids[i] in seen]
        if g2 and res["G2_seen_gender_flipped"]["planted"] < 10:
            # the gender our side saw: the truth's switch details for that mon (from the reference protocol)
            nm = D.parse_set(ents[g2[0]])[0]
            g_seen = next((ln.split("|")[3].split(", ")[1] for ln in ref[1]
                           if ln.split("|")[1:2] in (["switch"], ["drag"]) and ln.split("|")[3].split(",")[0] == nm
                           and len(ln.split("|")[3].split(", ")) > 1 and ln.split("|")[3].split(", ")[1] in ("M", "F")),
                          None)
            if g_seen:
                plants.append(("G2_seen_gender_flipped", g2[0], "F" if g_seen == "M" else "M"))
        for kind, slot, g in plants:
            f = D.parse_set(ents[slot])
            f[7] = g
            bad = list(ents)
            bad[slot] = "|".join(f)
            got = gate(b, d, at, opp, "]".join(bad), cmds)
            res[kind]["planted"] += 1
            if got == ref:
                why = "MISSED"
            elif got[0] is None:
                why = "refused"
            elif got[0] != ref[0]:
                why = "rows"
            else:
                why = "protocol_only"
            res[kind]["caught"] += why != "MISSED"
            res[kind]["by"][why] = res[kind]["by"].get(why, 0) + 1
            if len(res["examples"]) < 6:
                res["examples"].append({"id": did, "kind": kind, "slot": slot, "gender": g, "caught_by": why})
    res["pass"] = all(res[k]["planted"] >= 10 and res[k]["caught"] == res[k]["planted"]
                      for k in ("G1_draw_removed", "G2_seen_gender_flipped"))
    (C.HERE / f"teeth_v2_gender_{c}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}))


if __name__ == "__main__":
    main()
