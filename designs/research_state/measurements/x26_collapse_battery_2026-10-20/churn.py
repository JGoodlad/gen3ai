"""(f) POLICY CHURN and the read-overlap check, from the policy reads on X26's bank (no forward passes).

    python3 churn.py <reads dir> <labels comma-separated, in step order> <out.json> [--overlap <pilot reads dir>]

Churn: the share of the bank's decisions with >= 2 legal actions (all 7,028: the bank holds no single-legal one) whose greedy action differs between two
checkpoints, and the mean total-variation distance of their action distributions; for each consecutive pair and for
every label against the last. Battle-clustered bootstrap 95 % (2,000 resamples, seed 20261020) on the flip share.

--overlap: for every label present in both dirs, the max |p_this - p_pilot| and the greedy-flip share between this
read and the pilot's read of the SAME checkpoint (the pilot read at 63f97a57; this unit at its own commit). A
non-zero difference is the reader's drift across commits (e.g. config v160's tau solve), not training.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np


def load(d: Path, lab: str) -> np.ndarray:
    return np.load(d / f"{lab}.probs.npz")["probs"].astype(np.float64)


def main(argv: list[str]) -> int:
    reads, labels, out = Path(argv[0]), argv[1].split(","), argv[2]
    overlap = Path(argv[argv.index("--overlap") + 1]) if "--overlap" in argv else None
    bank = Path(__file__).resolve().parents[1] / "collapse_pilot_2026-10-17" / "bank_x26"
    dec = [json.loads(line) for line in gzip.open(bank / "decisions.jsonl.gz", "rt")]
    battles = np.array([d["battle"] for d in dec])
    free = np.array([d["mask"].count("1") >= 2 for d in dec])
    ub, inv = np.unique(battles[free], return_inverse=True)
    rng = np.random.default_rng(20261020)
    boot_idx = [rng.integers(0, len(ub), len(ub)) for _ in range(2000)]
    P = {lab: load(reads, lab)[free] for lab in labels}

    def flip(a: str, b: str) -> dict:
        f = (P[a].argmax(1) != P[b].argmax(1)).astype(float)
        tv = 0.5 * np.abs(P[a] - P[b]).sum(1)
        per_b = np.bincount(inv, weights=f, minlength=len(ub))
        n_b = np.bincount(inv, minlength=len(ub)).astype(float)
        bs = [per_b[i].sum() / n_b[i].sum() for i in boot_idx]
        return {"flip_share": round(float(f.mean()), 4), "ci95": [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])],
                "mean_tv": round(float(tv.mean()), 4)}

    res: dict = {"schema": "x26_collapse_battery_churn_v1", "n_free": int(free.sum()), "consecutive": {}, "vs_last": {}}
    for a, b in zip(labels, labels[1:]):
        res["consecutive"][f"{a}->{b}"] = flip(a, b)
    for a in labels[:-1]:
        res["vs_last"][f"{a}->{labels[-1]}"] = flip(a, labels[-1])
    if overlap is not None:
        res["overlap_vs_pilot"] = {}
        for lab in labels:
            mine, pilot = reads / f"{lab}.probs.npz", overlap / f"{lab}.probs.npz"
            if pilot.exists() and mine.exists() and mine.resolve() != pilot.resolve():
                a, b = load(reads, lab), load(overlap, lab)
                res["overlap_vs_pilot"][lab] = {"max_abs_prob_diff": float(np.abs(a - b).max()),
                                                "greedy_flip_share_all": float((a.argmax(1) != b.argmax(1)).mean())}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
