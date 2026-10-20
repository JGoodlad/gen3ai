"""B9 (opponent model, alpha): tabulate ``main.belief_roles read`` JSONs of X26 checkpoints on the Lane S bank.

    python3 b9_summary.py <belief_roles dir> <labels comma-separated> <out.json> <out.md>

Per checkpoint (on-pool rows, i.e. all 20,712; the bank's opponents are ai_v14-era pool snapshots, bots and
exploiters): alpha's intent log loss of the opponent's realised next action (all / move / switch, with the reader's
own 95 % interval, NOT paired across checkpoints), by opponent class, the OTHER calibration error, the presence Brier
and the role read R1. ``intent_logloss_conditional`` (the X5 adoption gate) needs a blob reference set and is not
available for X26 (no blob run), so the AS-BUILT log loss is the read. Descriptive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(d: str, labels: str, out_json: str, out_md: str) -> int:
    rows = {}
    for lab in labels.split(","):
        j = json.load(open(Path(d) / f"{lab}.json"))
        on = j["on_pool"]
        it = on["intent"]
        rows[lab] = {
            "intent_logloss_all": it["all"]["logloss"], "ci95_all": it["all"]["logloss_ci95"],
            "intent_logloss_move": it["move"]["logloss"], "intent_logloss_switch": it["switch"]["logloss"],
            "n_labeled": it["all"]["n_labeled"],
            "by_opp_class": {k: v.get("logloss") for k, v in it["by_opp_class"].items()},
            "other_mean_err": j["per_run"]["other_mean_err"],
            "presence_brier": j["per_run"]["presence_brier"],
            "presence_resolution": j["per_run"]["presence_resolution"],
            "roles_r1_weighted_abs_delta": j["per_run"]["roles_r1_weighted_abs_delta"],
            "excluded_rule8": it["excluded_rule8"],
        }
    Path(out_json).write_text(json.dumps({"schema": "x26_collapse_battery_b9_v1", "rows": rows}, indent=1))
    L = ["| X26 @ | intent log loss, all [95 %] | move | switch | vs pool snapshots | vs bots | vs exploiters | OTHER mean err | presence Brier | role R1 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, r in rows.items():
        c = r["by_opp_class"]
        f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
        L.append(f"| {lab} | {r['intent_logloss_all']:.3f} [{r['ci95_all'][0]:.3f}, {r['ci95_all'][1]:.3f}] | "
                 f"{f(r['intent_logloss_move'])} | {f(r['intent_logloss_switch'])} | {f(c.get('pool_snapshot'))} | "
                 f"{f(c.get('bot'))} | {f(c.get('exploiter'))} | {r['other_mean_err']:+.3f} | {r['presence_brier']:.3f} | "
                 f"{r['roles_r1_weighted_abs_delta']:.3f} |")
    Path(out_md).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
