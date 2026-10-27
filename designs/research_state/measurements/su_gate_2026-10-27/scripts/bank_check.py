"""The S-U gate test's GIGO TOOTH (PREREG §4): before any U state is read, the gate's scorer re-scores the S-U review's
bank rows and must reproduce the review's printed numbers EXACTLY (`../../su_headroom_2026-10-27/review_results.json`,
`noise`; REVIEW.md B2): sign accuracy 0.5132 [0.4468, 0.5801] n 228 at P* and 0.5798 [0.5184, 0.6447] n 238 at 25M;
IMP (= L1's registered cell) +0.0001 [-0.0100, +0.0111] and +0.0193 [+0.0083, +0.0305]; the deviation rate 0.4019 /
0.4145.

What it shares with the gate (`main.value_search.sign_gate`): the acting rule (`pikl_pick` = `main.value_search.rules`'
piKL pick, the one the live search plays), the sign value (`sign_value`) and the estimator (`cluster_ratio_boot`), run at
the review's B = 1,000 and seed 20261027 (one panel = the review's `boot`, draw for draw). The bank is loaded through L1's
registered loader (sha256-checked table and common ids, 719 states per anchor). Pure numpy, READ-ONLY on every input.

    PYTHONPATH=src python bank_check.py     (CPU, < 1 min, ~1 GB; writes bank_check.json beside this folder)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent.parent
sys.path.insert(0, str(MEAS / "search_l1_plateau_2026-10-25"))
import l1_plateau as P  # noqa: E402  L1's registered plateau loader

from main.value_search.sign_gate import cluster_ratio_boot, pikl_pick, sign_value  # noqa: E402

LAM = 0.015
B, SEED = 1000, 20261027
REVIEW = json.loads((MEAS / "su_headroom_2026-10-27" / "review_results.json").read_text())["noise"]


def r4(x: float) -> float:
    return round(float(x), 4)


def main() -> int:
    log: dict = {}
    recs, _ = P.load(log)
    for k in ("table_sha256", "n_common", "common_ids_sha256"):
        if log[k] != REVIEW["loader"][k]:
            raise SystemExit(f"loader {k}: {log[k]} != the review's {REVIEW['loader'][k]}")
    out: dict = {"loader": {k: log[k] for k in ("table_sha256", "n_common", "common_ids_sha256")}, "B": B, "seed": SEED}
    bad = []
    for lab in P.LABELS:
        g = {"dev": {}, "sign": {}, "imp": {}}
        for r in (r for r in recs if r["cp"] == lab):
            q = np.asarray(r["F"]["X4"], dtype=float).mean(1)
            T = np.asarray(r["Tp"], dtype=float)
            p8 = pikl_pick(q, r["logit"], LAM)
            for k, v in (("dev", float(p8 != 0)), ("sign", sign_value(T[p8], T[0]) if p8 != 0 else float("nan")),
                         ("imp", float(T[p8] - T[0]))):
                g[k].setdefault(r["battle"], []).append(v)
        res = {}
        for k, rk in (("dev", "dev8"), ("sign", "sign"), ("imp", "imp")):
            b = cluster_ratio_boot({lab: g[k]}, [lab], B=B, seed=SEED)
            mine = [r4(b["est"]), r4(b["lo"]), r4(b["hi"]), b["n"]]
            res[k] = {"mine": mine, "review": REVIEW[lab][rk], "equal": mine == REVIEW[lab][rk]}
            if mine != REVIEW[lab][rk]:
                bad.append(f"{lab} {k}: {mine} != the review's {REVIEW[lab][rk]}")
        out[lab] = res
    out["gate"] = "PASS" if not bad else "FAIL"
    out["failures"] = bad
    (HERE.parent / "bank_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
