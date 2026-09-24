"""Supplement to ``catalogue.py``: WHICH refusals get a non-masked α/β label, and does the label name
the refused move (a correct intent: the opponent DID choose it) or something else? And for every
window where the opponent was DRAGGED and the label reads SWITCH: what else the opponent did in the
window, and whether the label names the dragged mon or the one it chose. Over the
MILESTONE seeded-random keys (the policy half is omitted — its battles are slow to replay and the
catalogue's own table already carries its counts). One pass, no durable rows: it re-derives a
breakdown of counts ``rows.jsonl`` already holds in aggregate.

    python refused_breakdown.py > refused_breakdown.txt
"""

from __future__ import annotations

import collections
import logging
import sys


def main() -> int:
    from agents.battle import rust_core_parity as P

    logging.getLogger("poke-env").setLevel(logging.ERROR)
    out: collections.Counter = collections.Counter()
    for r in P.MILESTONE_RANDOM_KEYS:
        for key in r:
            lv = P.play(key, tag="Cat")
            (res,) = P.run_core([lv.recorded], trackers=True)
            for viewer in res.get("trackers") or []:
                for cap in viewer:
                    w = cap.get("window")
                    if not w:
                        continue
                    lab = cap["trackers"]["label"]
                    drags = [a for a in w if a["kind"] == "drag" and a["to"][0] == "opp"]
                    if drags and lab is not None and lab["kind"] == 1:
                        chosen = [a for a in w if a["kind"] == "switch" and a["to"][0] == "opp"]
                        refused = any(a["kind"] == "cant" and a["mon"][0] == "opp" and not a.get("then_moved") for a in w)
                        what = ("a chosen/replacement switch" if chosen else "a refusal" if refused else "neither")
                        names = ("the DRAGGED mon" if lab["switch_species"] == drags[-1]["to"][1] else
                                 "the chosen mon" if chosen and lab["switch_species"] == chosen[-1]["to"][1] else "another mon")
                        out[f"DRAG + SWITCH label | the opponent's own action: {what} | the label names {names}"] += 1
                    cants = [a for a in w if a["kind"] == "cant" and a["mon"][0] == "opp" and not a.get("then_moved")]
                    if not cants or lab is None or lab["kind"] == 2:
                        continue
                    c = cants[-1]
                    dragged = any(a["kind"] == "drag" and a["to"][0] == "opp" for a in w)
                    if lab["kind"] == 0:
                        named = "the refused move" if c.get("move_id") and lab["move_id"] == c["move_id"] else "another move"
                        out[f"MOVE label | cant {c['reason']} | names {named}"] += 1
                    else:
                        out[f"SWITCH label | cant {c['reason']} | {'dragged after' if dragged else 'no drag'}"] += 1
    for k, n in out.most_common():
        print(f"{n:6d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
