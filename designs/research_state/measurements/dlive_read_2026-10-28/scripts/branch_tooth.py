"""The deviation-only live read's PRE-PLAY TOOTH (PREREG §5): forced branches of the S-U GATE's banked raw games (unit
seeds 100-109, outside this read), on the registered engine, through the read's own code (``live_branch.plan_branches``
then ``python -m main.x35.kakuna run``). Three real S′ deviations (the re-gate's scored picks) and two NULL controls,
across both halves of a series (games 0, 2 and 10):

* every unit counts, and G-BRANCH passed live at its decision;
* each branch's protocol equals its raw game's through the branch turn (``main.reads.prefix.compare``);
* each null control replays its raw game WHOLE, byte for byte, with the same result and length. A one-game
  ``--deal-game G`` unit thereby reproduces game G of a whole 20-game unit (the SOP's UNVERIFIED note), on the engine.

No branch outcome is printed (the read's states are fresh; these are not among them).

    python branch_tooth.py plan  --out STATE     # then: python -m main.x35.kakuna run --state STATE
    python branch_tooth.py check --out STATE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from main.value_search import live_branch as LB

GATE = Path.home() / "gen3ai_archive/su_gate_2026-10-27"
REGATE = Path.home() / "gen3ai_archive/su_regate_rolls_2026-10-28/scored.jsonl"
RAW = GATE / "anchor/gate/pstar"
#: (cell, games kept, quota): the first two S′ deviations of an H unit in order, and one in the second half of a p72 unit
PICK = (("pstar__heldout72_hpfix__100", 0, 2), ("pstar__p72__101", 10, 1))


def plan(out: Path) -> None:
    sel = [json.loads(x) for x in (GATE / "gate/selected.jsonl").read_text().splitlines()]
    sc = {r["key"]: r for r in (json.loads(x) for x in REGATE.read_text().splitlines())}
    subset, quota = [], {}
    for cell, from_game, q in PICK:
        rows = [dict(r, raw_state=str(RAW)) for r in sel if r["cell"] == cell and r["index"] >= from_game]
        subset += rows
        quota[rows[0]["panel"]] = q
    scored, devs = [], {}
    for r in sorted(subset, key=LB.order_key):
        if devs.get(r["panel"], 0) >= quota[r["panel"]]:
            continue
        scored.append(sc[r["key"]])
        devs[r["panel"]] = devs.get(r["panel"], 0) + int(LB.is_deviation(sc[r["key"]]))
    for m in LB.plan_branches(subset, scored, quota, [RAW], out):
        print(m["unit"], m["kind"], f"forced {m['action']} (pi {m['pi_argmax']})")


def check(out: Path) -> int:
    rows = LB.collect(out)
    bad = []
    for r in rows:
        ok = (r["status"] == "counted" and not r["g_branch_fails"] and r["capture"]["ok"]
              and r.get("forced") == r["action"] and r.get("branch_index") == r["game"])
        if r["kind"] == "null":
            ok = ok and r["y_branch"] == r["y_raw"] and r["branch_turns"] == r["raw_turns"]
        print(json.dumps({"unit": r["unit"], "kind": r["kind"], "status": r["status"], "g_branch_fails":
                          r["g_branch_fails"], "capture": r.get("capture"), "cpu_s": r["cpu_s"], "tooth": ok}))
        bad += [] if ok else [r["unit"]]
    print(json.dumps({"units": len(rows), "tooth": "PASS" if not bad else "FAIL", "failed": bad}))
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", choices=("plan", "check"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.verb == "plan":
        plan(Path(a.out))
        return 0
    return check(Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
