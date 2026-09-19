"""POST-HOC: how much information does OMNISCIENCE actually add in gen 3, measured rather than
argued — and two descriptive facts about the captured boards.

PREDICTION.md P5 registered in advance that omniscience is NEARLY FREE for this evaluator family:
an UNREVEALED opponent mon has never been on the field, so it is at full HP, alive and unstatused,
which is exactly what the one-sided `Φ_mat` already assumes for a declared-but-unseen slot. The
only residue is that the opponent's on-field HP reaches us as a ROUNDED percentage. This prints the
size of that residue, per successor, so the claim is a measurement.

    python3 posthoc_omni.py --hand <hand dir> --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    d_hp, d_alive, d_tempo, unrev, seen_n, turns = [], [], [], [], [], []
    hp_true_unrev_full = 0
    n = 0
    # Per-fork: does every branch's successor sit at the SAME turn? A branch that KOs and forces a
    # replacement re-decides inside the fork turn, so its "successor" can be one ply shallower than
    # its sibling's. It is not a confound for this control (the head was scored on the identical
    # states) but it IS a property of every published level on this instrument.
    by_fork, lag = {}, []
    for hp_path in sorted(glob.glob(os.path.join(args.hand, "hand_s*.jsonl"))):
        for line in open(hp_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break
            if not r.get("obs_match"):
                continue
            b = r["board"]
            seen, true = b["opp_seen"], b.get("opp_true")
            if true is None:
                continue
            n += 1
            nu = max(0, seen["declared"] - seen["n_known"])
            one_sided_hp = seen["hp_known"] + nu           # unrevealed counted FULL HP
            one_sided_alive = seen["alive_known"] + nu
            d_hp.append(one_sided_hp - true["hp_known"])
            d_alive.append(one_sided_alive - true["alive_known"])
            d_tempo.append(seen["tempo"] - true["tempo"])
            unrev.append(nu)
            seen_n.append(seen["n_known"])
            turns.append(b["turn"])
            key = (r["shard"], r["battle"], r["inv"])
            by_fork.setdefault(key, {})[r["branch"]] = b["turn"]
            lag.append(b["turn"] - r["fork_turn"])
            # is every UNREVEALED opponent mon really untouched? alive_known + unrevealed == true
            # alive is the testable form of that claim.
            hp_true_unrev_full += int(one_sided_alive == true["alive_known"])

    d_hp = np.asarray(d_hp)
    out = {
        "n_successors": n,
        "opp_hp_sum_one_sided_minus_true": {
            "mean": float(d_hp.mean()), "mean_abs": float(np.abs(d_hp).mean()),
            "sd": float(d_hp.std()), "p95_abs": float(np.percentile(np.abs(d_hp), 95)),
            "max_abs": float(np.abs(d_hp).max()),
        },
        "opp_alive_one_sided_minus_true": {
            "mean": float(np.mean(d_alive)), "mean_abs": float(np.mean(np.abs(d_alive))),
            "max_abs": int(np.max(np.abs(d_alive))),
            "exactly_equal_rate": hp_true_unrev_full / max(1, n),
        },
        "opp_tempo_status_seen_minus_true": {
            "mean": float(np.mean(d_tempo)), "mean_abs": float(np.mean(np.abs(d_tempo))),
            "max_abs": int(np.max(np.abs(d_tempo))),
        },
        "opp_mons_revealed": {"mean": float(np.mean(seen_n)),
                              "hist": {str(k): int(v) for k, v in
                                       zip(*np.unique(seen_n, return_counts=True))}},
        "opp_mons_unrevealed": {"mean": float(np.mean(unrev))},
        "successor_turn": {"mean": float(np.mean(turns)),
                           "median": float(np.median(turns))},
        "successor_turn_minus_fork_turn": {
            "hist": {str(k): int(v) for k, v in zip(*np.unique(lag, return_counts=True))}},
        "branch_successors_at_same_turn": {
            "n_forks_with_3": int(sum(1 for v in by_fork.values() if len(v) == 3)),
            "rate_all_same_turn": float(np.mean(
                [len(set(v.values())) == 1 for v in by_fork.values() if len(v) == 3])),
        },
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
