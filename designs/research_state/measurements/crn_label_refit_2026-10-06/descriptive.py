"""The registered DESCRIPTIVE reads (README §3, last line): M1 accuracy split by pair category, and
ECE of each head scorer against the truth mean (held-out actions, 15 equal-mass bins on the ±1 scale
mapped to [0, 1]). Appends to ``result.json`` under ``descriptive``.

    python descriptive.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import score as S

HERE = Path(__file__).resolve().parent
SCORERS = ("BASE", "REFIT_K1_s0", "REFIT_K8_s0", "LEAF_K8")


def pair_class(cats, a, b):
    ca, cb = cats.get(str(a), "?"), cats.get(str(b), "?")
    sw = (ca == "switch") + (cb == "switch")
    return {0: "move-move", 1: "move-switch", 2: "switch-switch"}[sw]


def ece(pred01, y01, bins=15):
    o = np.argsort(pred01, kind="stable")
    tot = 0.0
    for part in np.array_split(o, bins):
        tot += len(part) * abs(pred01[part].mean() - y01[part].mean())
    return tot / len(pred01)


def main() -> int:
    decs = S.load_decisions()
    by = {d["id"]: d for d in decs}
    pol = S.root_policy(decs)
    sc = np.load(S.ROWS / "scores.npz")
    tabs, names = S.turn_tables(sc)
    sub = json.loads((S.ROWS / "subset_train.json").read_text())["ids"] + \
        json.loads((S.ROWS / "subset_held.json").read_text())["ids"]
    cont, _, _ = S.contested_ids(decs, pol, sub)
    acc = defaultdict(lambda: defaultdict(list))
    for did in sorted(tabs):
        if did not in cont:
            continue
        e = tabs[did]
        pr, lg = pol[did]
        for i, j, sg in S.pairs_for(e, "all", pr, lg):
            pc = pair_class(by[did]["cats"], e["actions"][i], e["actions"][j])
            for n in SCORERS:
                acc[pc][n].append(S.correct(e["scorers"][n], i, j, sg))
    split = {pc: {"n_pairs": len(v[SCORERS[0]]), **{n: float(np.mean(v[n])) for n in SCORERS}}
             for pc, v in sorted(acc.items())}
    truth01 = (sc["truth"].mean(1) + 1.0) / 2.0
    eces = {n: float(ece((sc[n] + 1.0) / 2.0, truth01)) for n in ("BASE", "REFIT_K1_s0", "REFIT_K4_s0", "REFIT_K8_s0")}
    res = json.loads((HERE / "result.json").read_text())
    res["descriptive"] = {"M1_by_pair_class": split, "ece_vs_truth_mean_held_actions": eces}
    (HERE / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["descriptive"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
