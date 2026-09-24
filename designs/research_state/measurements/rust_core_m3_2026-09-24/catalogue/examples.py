"""Concrete REPRODUCTIONS for the catalogue's label cases: the first decisions (battle, viewer, turn,
the chunk index) where the α/β label reads a given way in a given opponent case, with the native
window that shows what really happened and the per-side protocol lines of that window.

    python examples.py --keys 0 40 --per-case 2       # MILESTONE random keys [0, 40)

Each battle is replayed with the key recipe (``rust_core_parity.play``), so every example is
reproducible from its key alone. The label is the core's, which slice T holds equal to training's.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys

from catalogue import classify

WANT = ("label:opp_dragged:1", "label:opp_replacement_only:1", "label:opp_refused:0", "label:opp_refused:1",
        "label:opp_baton_pass:0", "label:opp_baton_pass:2", "label:opp_called_move:0", "label:opp_denied_fainted_first:2",
        "label:opp_denied_turn_cut:2", "label:opp_encored_same_turn:0", "faint_destinybond", "faint_perishsong",
        "item_via_trick", "hazard_cleared", "clock_false_progress")


def main(argv=None) -> int:
    from agents.battle import rust_core_parity as P

    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs=2, type=int, default=[0, 40])
    ap.add_argument("--per-case", type=int, default=1)
    a = ap.parse_args(argv)
    logging.getLogger("poke-env").setLevel(logging.ERROR)
    found = collections.defaultdict(list)
    for key in range(*a.keys):
        lv = P.play(key, tag="Ex")
        (res,) = P.run_core([lv.recorded], trackers=True)
        chunks = P.core_chunks(res)
        for vi, viewer in enumerate(res.get("trackers") or []):
            prev_after = -1
            for cap in viewer:
                if "window" not in cap:
                    continue
                c = classify(cap["window"], cap["trackers"]["label"], cap["trackers"])
                side = "p1" if vi == 0 else "p2"
                lines = [ln for j, (s, t) in enumerate(chunks) if s == side and prev_after < j <= cap["after"]
                         for ln in t.split("\n") if not ln.startswith("|request|")]
                prev_after = cap["after"]
                for k in WANT:
                    if c.get(k) and len(found[k]) < a.per_case:
                        found[k].append({"battle": f"random_{key}", "viewer": side, "after": cap["after"],
                                         "label": cap["trackers"]["label"], "lines": lines,
                                         "window": [{x: y for x, y in act.items() if x != "effects"} for act in cap["window"]]})
        if all(len(found[k]) >= a.per_case for k in WANT):
            break
    for k in WANT:
        print(f"\n===== {k}: {len(found[k])} example(s)")
        for ex in found[k]:
            print(json.dumps({x: ex[x] for x in ("battle", "viewer", "after", "label")}))
            for ln in ex["lines"]:
                print("   ", ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
