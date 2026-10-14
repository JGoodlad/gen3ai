"""VALIDITY (no outcome): every world the HEAD builder accepted must ALSO pass the strict gate at the PAD pin's own engine
(the engine that plays it): the world's prefix, replayed by the pad pin's ``core_events --trackers --obs``, gives the
searched side byte-equal observation frames at every one of its decisions up to the root AND byte-equal protocol lines
(everything that side was sent) vs the TRUE prefix replayed by the same binary. This is what makes "built at HEAD, played
at the pin" sound (README §2): the HEAD gate and the pin engine agree on every accepted world.

TEETH (planted): REQUIRED — a seen opponent mon's REVEALED move replaced by Splash (play named it, so the searched side
saw it: always observable; every plant must be caught). DESCRIPTIVE — (a) every gender-PAD marker stripped (``M*`` →
``M``: the named gender kept, the turn-0 draw no longer spent; the dice shift whenever the slot drew); (b) a seen mon's
item replaced by another item (caught only when the item acted or was named before the root: on the first read 18 of 19,
the uncaught one an item that never showed, as it should be).

    run_pin.sh <cp>_pad 1 6 2h pin_regate.py --code Tnn [--states registered] [--n 60] [--out pin_regate_<cp>_<code>.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_common as T  # noqa: E402

sys.path.insert(1, str(T.PILOT))
import common as C  # noqa: E402


def viewer_view(res: dict, viewer: int, n: int):
    ent = [e for e in res["trackers"][viewer] if "obs" in e]
    if len(ent) <= n:
        return None
    after = int(ent[n]["after"])
    lines = [ln for side, ls in res["chunks"][: after + 1] if side == viewer for ln in ls]
    return [e["obs"]["b64"] for e in ent[: n + 1]], lines


def strip_pad(team: str) -> str:
    out = []
    for e in team.split("]"):
        f = e.split("|")
        if len(f) > 7 and f[7].endswith("*"):
            f[7] = f[7][:-1]
        out.append("|".join(f))
    return "]".join(out)


def swap_seen_item(team: str, seen: set) -> str:
    out = []
    done = False
    for e in team.split("]"):
        f = e.split("|")
        sid = "".join(ch for ch in (f[1] or f[0]).lower() if ch.isalnum())
        if not done and sid in seen:
            f[2] = "lumberry" if f[2].lower() != "lumberry" else "leftovers"
            done = True
        out.append("|".join(f))
    return "]".join(out)


def swap_revealed_move(team: str, revealed: dict):
    """A seen mon's REVEALED move (play named it, so the searched side saw it) replaced by Splash: always observable.
    None when no seen mon has a revealed move held under that id in the world's set."""
    out, done = [], False
    for e in team.split("]"):
        f = e.split("|")
        sid = "".join(ch for ch in (f[1] or f[0]).lower() if ch.isalnum())
        mv = f[4].split(",") if len(f) > 4 else []
        rev = set(revealed.get(sid, []))
        hit = next((i for i, m in enumerate(mv) if m in rev and m != "splash"), None)
        if not done and hit is not None:
            mv[hit] = "splash"
            f[4] = ",".join(mv)
            done = True
        out.append("|".join(f))
    return "]".join(out) if done else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--states", default="registered", choices=("registered", "supp"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cp = C.cp()
    if os.environ.get("HIB_T1_PIN_KIND") != "pad":
        raise SystemExit("run at a pad pin")
    from agents.battle.core_replay import RecordedBattle, run_core
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index

    bank = load_bank(T.bank_path(cp, a.states))
    rows = T.read_dir(T.worlds_dir(cp, a.code), f"worlds_{a.states}_*.jsonl")
    ids = sorted(i for i, r in rows.items() if r.get("ok"))
    ids = ids[:: max(1, len(ids) // a.n)][: a.n]
    res = {"cp": cp, "code": a.code, "states": len(ids), "worlds": 0, "fail": 0, "padded_worlds": 0,
           "plant_strip": {"tried": 0, "caught": 0}, "plant_item": {"tried": 0, "caught": 0}, "examples": []}
    for did in ids:
        w = rows[did]
        d = next(x for x in bank.decisions if x["id"] == did)
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
        jobs = [rec(true_opp, b.commands[:at], f"{did}#true")]
        kinds = ["true"]
        for j, (tw, cw) in enumerate(zip(w["teams"], w["cmds"])):
            jobs.append(rec(tw, cw, f"{did}#w{j}"))
            kinds.append(("world", j))
            if "*" in tw:
                jobs.append(rec(strip_pad(tw), cw, f"{did}#w{j}#strip"))
                kinds.append(("strip", j))
            if j == 0:
                jobs.append(rec(swap_seen_item(tw, set(w["revealed"])), cw, f"{did}#w{j}#item"))
                kinds.append(("item", j))
                mv = swap_revealed_move(tw, w["revealed"])
                if mv is not None:
                    jobs.append(rec(mv, cw, f"{did}#w{j}#move"))
                    kinds.append(("move", j))
        out = run_core(jobs, trackers=True, obs=True)
        ref = viewer_view(out[0], viewer, d["n"]) if out[0]["ok"] else None
        if ref is None:
            raise SystemExit(f"{did}: the TRUE prefix does not replay at the pad pin")
        for kind, r in zip(kinds[1:], out[1:]):
            got = viewer_view(r, viewer, d["n"]) if r["ok"] else None
            same = got is not None and got[0] == ref[0] and got[1] == ref[1]
            if kind[0] == "world":
                res["worlds"] += 1
                res["padded_worlds"] += int("*" in w["teams"][kind[1]])
                if not same:
                    res["fail"] += 1
                    if len(res["examples"]) < 10:
                        res["examples"].append(f"{did} world {kind[1]}: {'refused ' + str(r.get('error'))[:200] if not r['ok'] else 'differs'}")
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
    if a.out:
        (HERE / a.out).write_text(s + "\n")


if __name__ == "__main__":
    main()
