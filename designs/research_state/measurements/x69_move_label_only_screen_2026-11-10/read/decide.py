"""X69 light screen: SCREEN.md §6 applied to the meter's table, with an independent recomputation.

Frozen record of how §9's numbers were made (2026-11-11); not a tool. Usage:
    python decide.py <scores dir> <table.json> <out decide.json>

1. Reads `table.json` (written by `python -m main.offpool_belief table --versus C0`) and takes, per arm and panel,
   `moves_ll` `minus_C0` (mean, lo, hi): the registered d = arm - C0.
2. Recomputes d BY HAND from the eight score files with numpy alone (team means, C0 = the mean over its five
   finals inside each team, paired team bootstrap: 10,000 draws, default_rng([20261109, n_teams]), percentile 95 %)
   and requires it to equal the table's to 1e-12.
3. Applies §6: an arm passes iff LB(H) >= +0.6, LB(F) > 0 and LB(home72) >= -0.2; a bound within 1e-6 of its
   threshold is NOT met. PROMOTE the first passing arm in the order B, A, AB; else DROP.
"""
import json
import sys
from pathlib import Path

import numpy as np

SEED, DRAWS, EPS = 20261109, 10_000, 1e-6
ARMS = {"B": "x69b_s6902", "A": "x69a_s6901", "AB": "x69ab_s6903"}          # §6.2's order
C0 = ["c0_s3501", "c0_s3502", "c0_s3503", "c0_s3504", "c0_s3505"]
PANELS = {"H": "heldout72_hpfix", "F": "forum72", "home72": "home72"}
ROWS = {"heldout72_hpfix": "21545e87b1fb", "forum72": "0caf6037fee3", "home72": "5207d8c100df"}
# (threshold, strict): strict = "> t" (F), else ">= t"; either way a bound within EPS of t is not met.
BARS = {"H": (0.6, False), "F": (0.0, True), "home72": (-0.2, False)}


def team_means(doc, panel, metric="moves_ll", variant="post"):
    b = doc["panels"][panel]
    s, n = np.asarray(b["sums"][metric][variant], float), np.asarray(b["units"]["moves"], float)
    out = np.full(s.shape, np.nan)
    np.divide(s, n, out=out, where=n > 0)
    return out


def paired(a, b):
    v = a - b
    assert not np.isnan(v).any(), "a team without a moves unit: the by-hand path assumes none"
    idx = np.random.default_rng([SEED, len(v)]).integers(0, len(v), size=(DRAWS, len(v)))
    boot = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def met(lb, bar, strict):
    if abs(lb - bar) <= EPS:
        return False
    return lb > bar


def main(scores_dir, table_json, out):
    scores = {k: json.loads((Path(scores_dir) / f"{k}.json").read_text()) for k in [*ARMS.values(), *C0]}
    table = json.loads(Path(table_json).read_text())
    assert table["versus"] == "C0" and table["bootstrap"] == {"draws": DRAWS, "seed": SEED, "unit": "team"}
    for doc in scores.values():
        for p, sha in ROWS.items():
            assert doc["panels"][p]["bank"]["rows_sha256"].startswith(sha), (doc["label"], p)
            assert len(doc["panels"][p]["team_labels"]) == 72
    res = {"rule": "SCREEN.md §6", "seed": SEED, "draws": DRAWS, "eps": EPS, "arms": {}}
    for arm, label in ARMS.items():
        cell = {}
        for short, panel in PANELS.items():
            c0 = np.stack([team_means(scores[c], panel) for c in C0]).mean(axis=0)
            hand = paired(team_means(scores[label], panel), c0)
            t = table["panels"][panel]["metrics"]["moves_ll"]["arms"][arm]["minus_C0"]
            tab = (t["mean"], t["lo"], t["hi"])
            assert all(abs(x - y) <= 1e-12 for x, y in zip(hand, tab)), (arm, panel, hand, tab)
            bar, strict = BARS[short]
            cell[short] = {"d": tab[0], "lo": tab[1], "hi": tab[2], "bar": bar, "strict": strict,
                           "within_eps_of_bar": abs(tab[1] - bar) <= EPS, "met": met(tab[1], bar, strict),
                           "by_hand_equal": True}
        cell["passes"] = all(cell[s]["met"] for s in PANELS)
        res["arms"][arm] = cell
    first = next((a for a in ARMS if res["arms"][a]["passes"]), None)
    res["outcome"] = f"PROMOTE {first}" if first else "DROP"
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    for arm, cell in res["arms"].items():
        print(arm, " ".join(f"{s}: {cell[s]['d']:+.3f} [{cell[s]['lo']:+.3f}, {cell[s]['hi']:+.3f}] "
                            f"{'met' if cell[s]['met'] else 'NOT met'}" for s in PANELS),
              "PASSES" if cell["passes"] else "does not pass")
    print("OUTCOME:", res["outcome"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
