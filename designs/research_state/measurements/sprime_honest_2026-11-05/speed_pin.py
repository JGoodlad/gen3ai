"""S'-HONEST design input 5 (2026-11-05, OUTCOME-BLIND): the size of the SPEED-TIE / dice-silent seed-reuse leak
(L1a of design_sprime_honest.md) on S'-LIVE's OWN decisions, by the info-audit's counterfactual flips
(``main.info_audit.checks.seed_reuse``: one hidden fact of one seen opponent mon changed — its Speed by +/-1 point, its
unrevealed item to a neutral one or a dice item, its unrevealed ability to a dice-rolling alternative — gated exactly
as a world is, on the TRUE seed, against a passing unflipped control). The decisions: a fixed-seed random sample of the
hashed re-score sample's kept logs (every search-path kind, excluded ones included).

    PYTHONPATH=src python3 speed_pin.py <S'-LIVE state>/anchor/u/pstar_s/cells 400 speed_pin.json
"""
import glob
import json
import sys

import numpy as np

from main.info_audit import checks as C
from main.value_search import worlds as W


def main(cells, nmax, out):
    recs = []
    for f in sorted(f for f in glob.glob(cells + "/*/search_decisions.jsonl") if ".partial." not in f):
        for line in open(f):
            if '"rescore_sample": true' in line:
                recs.append(json.loads(line))
    rng = np.random.default_rng(20261105)
    pick = [recs[i] for i in sorted(rng.choice(len(recs), size=min(nmax, len(recs)), replace=False))]
    P = W.Prior()
    rows, fails = [], 0
    for d in pick:
        try:
            root = W.prepare_root(d["log"], d["at"], d["side"], d["key"], root_choice_in_log=False)
            v = C.seed_reuse(root, P)
        except Exception as e:                       # noqa: BLE001  (counted, never a silent skip)
            fails += 1
            rows.append({"key": d["key"], "error": f"{type(e).__name__}: {str(e)[:160]}"})
            continue
        keep = ("pass", "speed_mons", "speed_pinned", "item_mons", "item_dice_rejected", "item_dice_tested",
                "item_mass_removed", "ability_mons", "ability_rejected", "replays")
        rows.append(dict({k: v.get(k) for k in keep}, key=d["key"], turn=d["turn"],
                         searched=bool(d.get("searched"))))
    ok = [r for r in rows if "error" not in r]
    summary = {"decisions": len(rows), "failed": fails, "control_pass": sum(1 for r in ok if r["pass"]),
               "speed_pairs": sum(r["speed_mons"] for r in ok), "speed_pinned_pairs": sum(r["speed_pinned"] for r in ok),
               "decisions_with_a_speed_pin": sum(1 for r in ok if r["speed_pinned"]),
               "item_dice_rejected": sum(r["item_dice_rejected"] for r in ok),
               "item_dice_tested": sum(r["item_dice_tested"] for r in ok),
               "ability_rejected": sum(r["ability_rejected"] for r in ok),
               "ability_mons": sum(r["ability_mons"] for r in ok)}
    summary["share_decisions_speed_pinned"] = round(summary["decisions_with_a_speed_pin"] / max(1, len(ok)), 4)
    summary["share_pairs_speed_pinned"] = round(summary["speed_pinned_pairs"] / max(1, summary["speed_pairs"]), 4)
    summary["item_mass_removed_per_item_mon"] = round(sum(r["item_mass_removed"] for r in ok) / max(1, sum(r["item_mons"] for r in ok)), 8)
    json.dump({"summary": summary, "decisions": rows}, open(out, "w"), indent=1, sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
