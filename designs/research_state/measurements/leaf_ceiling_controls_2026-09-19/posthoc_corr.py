"""POST-HOC: how much of the learned head IS the hand evaluator?

Spearman rank correlation between arm W's `sigmoid(win_prob_logits)` and each hand scorer, over
the 14,570 scored successor states. A high rank correlation says the 75M-step critic and a
hand-written potential sum order the same states the same way — which is the mechanism behind
their reading the same pairwise accuracy.

    python3 posthoc_corr.py --forks <dir> --snapshot <zip> --hand <dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_MEAS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("paired_refit_discrimination_2026-09-14", "fork_arm_read_2026-09-16",
           "offline_leaf_fit_2026-09-18", "leaf_ceiling_controls_2026-09-19"):
    sys.path.insert(0, os.path.join(_MEAS, _d))


def _spearman(x, y):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forks", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    import refit
    from forks import load_model
    from fit_heads import forward_feats
    from score_controls import HAND_NAMES, load_hand

    rows, succ, sm = refit.load_forks(args.forks)
    model = load_model(args.snapshot)
    pooled, _pre, V0 = forward_feats(model, succ, sm, want_prepool=False)
    tab, _pairs = refit.build_table(rows, pooled, V0)
    hand, _turn, _st = load_hand(args.hand, args.forks)
    v = tab["V0"]
    out = {"n": int(len(v)), "spearman_vs_headW": {}, "auc_vs_outcome": {}}
    from refit import auc
    out["auc_vs_outcome"]["headW"] = auc(v, tab["y"])
    for nm in HAND_NAMES:
        h = hand[nm][tab["idx"]]
        m = np.isfinite(h)
        out["spearman_vs_headW"][nm] = _spearman(v[m], h[m])
        out["auc_vs_outcome"][nm] = auc(h[m], tab["y"][m])
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
