"""E2 §5 (VALIDITY, descriptive, no outcome): every world the E2 HEAD builder accepted must also pass the strict gate at
the PAD pin's own engine — Tier 1's `pin_regate.py` on E2's worlds (its helpers imported, never copied): the world's
prefix replayed by the pad pin's ``core_events --trackers --obs`` gives the searched side byte-equal observation frames
at every one of its decisions up to the root AND byte-equal protocol lines vs the TRUE prefix. TEETH as Tier 1: a seen
mon's REVEALED move replaced by Splash must be caught every time (required); the pad-strip and item-swap plants are
descriptive.

    run_e2.sh <cp>_pad 1 6 4h pin_regate_e2.py --arm F [--n 0]      # --n 0 = every accepted world of the arm

Writes ``pin_regate_e2_<cp>_<arm>.json`` (this folder).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_common as E  # noqa: E402

E.add_paths()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    cp = os.environ.get("VDP_CP")
    if os.environ.get("HIB_E2_KIND") != "pad" or cp not in E.CPS:
        raise SystemExit("run at a pad pin (run_e2.sh <cp>_pad)")
    import pin_regate as PR                         # Tier 1's (its viewer_view and plants)
    from agents.battle.core_replay import RecordedBattle, run_core
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index

    bank = load_bank(E.bank_path(cp))
    dec = {d["id"]: d for d in bank.decisions}
    sub = ("V_" + E.arm_dir(a.arm[2:])) if a.arm.startswith("V:") else E.arm_dir(a.arm)
    rows = E.read_dir(E.cp_dir(cp) / "worlds" / sub, "worlds_*.jsonl")
    ids = sorted(i for i, r in rows.items() if r.get("ok"))
    if a.n:
        ids = ids[:: max(1, len(ids) // a.n)][: a.n]
    res = {"cp": cp, "arm": a.arm, "pin_head": os.environ.get("HIB_E2_CHECKOUT_HEAD"), "states": len(ids), "worlds": 0,
           "fail": 0, "padded_worlds": 0, "examples": []}
    for did in ids:
        w = rows[did]
        d = dec[did]
        b = bank.battles[bank.battle_index[d["battle"]]]
        side = d["side"]
        viewer = 0 if side == "p1" else 1
        at = cmd_index(b, side, d["n"])
        if at != int(w["at"]):
            raise SystemExit(f"{did}: root command index {at} != the builder's {w['at']}")

        def rec(team_opp, cmds, label):
            p1, p2 = dict(b.p1), dict(b.p2)
            (p2 if side == "p1" else p1)["team"] = team_opp
            return RecordedBattle(label=label, format_id=b.format_id, seed=b.seed, p1=p1, p2=p2,
                                  commands=[list(x) for x in cmds])

        true_opp = (b.p2 if side == "p1" else b.p1)["team"]
        jobs, kinds = [rec(true_opp, b.commands[:at], f"{did}#true")], ["true"]
        for j, (tw, cw) in enumerate(zip(w["teams"], w["cmds"])):
            jobs.append(rec(tw, cw, f"{did}#w{j}"))
            kinds.append(("world", j))
            if "*" in tw:
                jobs.append(rec(PR.strip_pad(tw), cw, f"{did}#w{j}#strip"))
                kinds.append(("strip", j))
            if j == 0:
                jobs.append(rec(PR.swap_seen_item(tw, set(w["revealed"])), cw, f"{did}#w{j}#item"))
                kinds.append(("item", j))
                mv = PR.swap_revealed_move(tw, w["revealed"])
                if mv is not None:
                    jobs.append(rec(mv, cw, f"{did}#w{j}#move"))
                    kinds.append(("move", j))
        out = run_core(jobs, trackers=True, obs=True)
        ref = PR.viewer_view(out[0], viewer, d["n"]) if out[0]["ok"] else None
        if ref is None:
            raise SystemExit(f"{did}: the TRUE prefix does not replay at the pad pin")
        for kind, r in zip(kinds[1:], out[1:]):
            got = PR.viewer_view(r, viewer, d["n"]) if r["ok"] else None
            same = got is not None and got[0] == ref[0] and got[1] == ref[1]
            if kind[0] == "world":
                res["worlds"] += 1
                res["padded_worlds"] += int("*" in w["teams"][kind[1]])
                if not same:
                    res["fail"] += 1
                    if len(res["examples"]) < 10:
                        res["examples"].append(f"{did} world {kind[1]}: "
                                               f"{'refused ' + str(r.get('error'))[:200] if not r['ok'] else 'differs'}")
            else:
                key = f"plant_{kind[0]}"
                res.setdefault(key, {"tried": 0, "caught": 0})
                res[key]["tried"] += 1
                res[key]["caught"] += int(not same)
    mv = res.get("plant_move", {"tried": 0, "caught": 0})
    res["verdict"] = "PASS" if (res["worlds"] and res["fail"] == 0 and mv["tried"] and mv["caught"] == mv["tried"]) \
        else "FAIL"
    s = json.dumps(res, indent=1, sort_keys=True)
    print(s)
    (E.HERE / f"pin_regate_e2_{cp}_{sub}.json").write_text(s + "\n")


if __name__ == "__main__":
    main()
