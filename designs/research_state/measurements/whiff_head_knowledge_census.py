"""PROBE L, stage 1 — the MODEL-FREE whiff census + the alpha readout on EVERY decision.

For one step of one run, over an fnmatch opponent scope:
  * run `main.prober.loops.analyze_battle` on every battle (raw Showdown protocol, never the
    rendered timeline) -> BaitEvents (turn, arrival, move, kind, whiff, loop_step, reclick, inv,
    chosen_prob, delta_v, delta_win_prob) and PivotReads (alpha/beta per voluntary pivot);
  * additionally emit the alpha SWITCH probability for EVERY `move_selection` decision in the
    battle, tagged by what that decision turned out to be:
        whiff_pivot   - they pivoted, we moved into it, it whiffed   (the population)
        hit_pivot     - they pivoted, we moved into it, it connected (the matched control)
        no_pivot      - no voluntary opponent pivot on that turn      (the baseline)
    This is what makes "alpha is ELEVATED at whiff decisions" a comparison rather than a number.

Writes one jsonl row per battle.

Run:
  nice -n 15 python tmp/whiff_census.py --step step_24000000 --opponent 'sentinel_*'
  (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from main.prober.loops import analyze_battle, chosen_prob, norm_id  # noqa: E402
from main.prober.engine.protocol import parse_protocol_log  # noqa: E402


def protocol_lines(base: str) -> "list[str]":
    """The raw Showdown protocol out of the trace's `*_replay.html` sibling — the SAME parser
    `ProbeSession._protocol_lines` uses, so the census means what the shipped detector means."""
    path = base + "_replay.html"
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return list(parse_protocol_log(f.read()))
    except Exception:  # noqa: BLE001 — best-effort, mirrors the session
        return []


def alpha_switch_p(inv: dict) -> "float | None":
    alpha = (inv.get("opp_intent") or {}).get("alpha") or []
    return next((float(a["p"]) for a in alpha if a.get("name") == "SWITCH"), None)


def alpha_top_is_switch(inv: dict) -> "bool | None":
    alpha = (inv.get("opp_intent") or {}).get("alpha") or []
    return bool(alpha[0].get("name") == "SWITCH") if alpha else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--step", default="step_24000000")
    ap.add_argument("--opponent", default="sentinel_*")
    ap.add_argument("--near-zero-frac", type=float, default=0.01)
    ap.add_argument("--out", default="tmp/whiff_census.jsonl")
    a = ap.parse_args(argv)

    run = a.run or "/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823"
    root = os.path.join(run, "eval_traces", a.step)
    opps = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and fnmatch.fnmatchcase(d, a.opponent))
    skipped: collections.Counter = collections.Counter()
    n_battles = 0
    with open(a.out, "w", buffering=1) as fh:
        for opp in opps:
            odir = os.path.join(root, opp)
            for name in sorted(os.listdir(odir)):
                if not name.endswith("_summary.json"):
                    continue
                base = os.path.join(odir, name[: -len("_summary.json")])
                lines = protocol_lines(base)
                if not lines:
                    skipped["no_replay_html"] += 1
                    continue
                with open(base + "_summary.json") as f:
                    summary = json.load(f)
                npz_path = base + "_states.npz"
                if not os.path.exists(npz_path):
                    skipped["no_npz"] += 1
                    continue
                with np.load(npz_path) as z:
                    values = list(z["values"]) if "values" in z.files else None
                    win_probs = list(z["win_probs"]) if "win_probs" in z.files else None
                    n_legal = (np.asarray(z["action_mask"], dtype=bool).sum(axis=1).tolist()
                               if "action_mask" in z.files else None)
                    has_actions = "actions" in z.files
                invs = summary.get("invocations", [])
                outcome = ((summary.get("meta") or {}).get("result") or "?")
                fold = analyze_battle(lines, invs, outcome=outcome,
                                      n_turns=(summary.get("meta") or {}).get("turns"),
                                      values=values, win_probs=win_probs,
                                      near_zero_frac=a.near_zero_frac)
                if fold.skipped:
                    skipped[fold.skipped.split(":")[0]] += 1
                    continue
                n_battles += 1

                bait_by_turn = {b.turn: b for b in fold.baits}
                read_by_turn = {r.turn: r for r in fold.reads}
                # alpha on EVERY move_selection decision, tagged by its bait status
                decisions = []
                for k, inv in enumerate(invs):
                    if inv.get("phase") != "move_selection" or inv.get("turn") is None:
                        continue
                    t = int(inv["turn"])
                    b = bait_by_turn.get(t)
                    if b is None:
                        tag = "no_pivot" if t not in read_by_turn else "pivot_not_moved_into"
                    elif b.whiff:
                        tag = "whiff_pivot"
                    else:
                        tag = "hit_pivot"
                    decisions.append({
                        "inv": k, "turn": t, "tag": tag,
                        "alpha_switch_p": alpha_switch_p(inv),
                        "alpha_top_is_switch": alpha_top_is_switch(inv),
                        "chosen_prob": chosen_prob(inv),
                        "chosen": inv.get("chosen"),
                        "our": norm_id((inv.get("our") or {}).get("species")),
                        "opp": norm_id((inv.get("opp") or {}).get("species")),
                        "n_legal": (int(n_legal[k]) if n_legal is not None and k < len(n_legal)
                                    else None),
                    })

                fh.write(json.dumps({
                    "base": base, "opponent": opp, "step": a.step,
                    "outcome": fold.outcome, "our_side": fold.our_side,
                    "turns": fold.n_turns, "n_decisions": fold.n_decisions,
                    "has_recon": os.path.exists(base + "_reconstruction.json"),
                    "has_actions_array": bool(has_actions),
                    "moved_into_pivots": fold.moved_into_pivots,
                    "whiffs": fold.whiffs, "whiff_kinds": fold.whiff_kinds,
                    "reclicks": fold.reclicks, "loop_battle": fold.loop_battle,
                    "worst_loop": fold.worst_loop, "loops": [dict(g) for g in fold.loops],
                    "baits": [{"turn": b.turn, "arrival": b.arrival, "move": b.move,
                               "kind": b.kind, "whiff": b.whiff, "loop_step": b.loop_step,
                               "reclick": b.reclick, "inv": b.inv,
                               "chosen_prob": b.chosen_prob, "delta_v": b.delta_v,
                               "delta_win_prob": b.delta_win_prob} for b in fold.baits],
                    "reads": [{"turn": r.turn, "arrival": r.arrival, "first_time": r.first_time,
                               "arrival_revealed": r.arrival_revealed,
                               "slot_correct": r.slot_correct,
                               "species_correct": r.species_correct,
                               "alpha_top_is_switch": r.alpha_top_is_switch,
                               "alpha_switch_p": r.alpha_switch_p,
                               "loop_step": r.loop_step} for r in fold.reads],
                    "decisions": decisions,
                }) + "\n")

    print(json.dumps({"root": root, "opponents": opps, "battles": n_battles,
                      "skipped": dict(skipped), "out": a.out}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
