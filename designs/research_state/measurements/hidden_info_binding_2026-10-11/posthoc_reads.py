"""POST-HOC descriptive breakdowns (NOT registered; written AFTER descriptive_reads.json was read, to understand D1's
split): the pooled gap g(s) = IMP_O − IMP_P (LEAF-K8, Phase 2 Smogon worlds) by unrevealed = 0 vs ≥ 1 crossed with
the oracle pick's kind, and within unrevealed = 0 by the opponent active's unseen move slots and by the seen mons'
total unseen move slots. Labelled POST-HOC wherever quoted; no verdict reads it.

    python -I posthoc_reads.py  →  posthoc_reads.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "value_disc_pilot_2026-10-11"))
import descriptive_reads as DR  # noqa: E402
import score as S1  # noqa: E402


def main():
    recs = []
    for cp in DR.CPS:
        _st, rs, _ex = DR.per_state(cp, "det")
        recs += rs
    bt = S1.Boot([r["battle"] for r in recs], [1.0] * len(recs))
    g = np.array([r["imp_o"]["ARGMAX|LEAF-K8"] - r["imp_p"]["ARGMAX|LEAF-K8"] for r in recs])
    out = {}
    out.update(DR.strata(bt, recs, g, "unrev0_vs_1plus_x_pick",
                         lambda r: ("unrev0" if r["unrevealed"] == 0 else "unrev1+") + "/" +
                         ("move" if r["pick_o"] >= 6 else "switch")))
    sub = [r for r in recs if r["unrevealed"] == 0]
    bts = S1.Boot([r["battle"] for r in sub], [1.0] * len(sub))
    gs = np.array([r["imp_o"]["ARGMAX|LEAF-K8"] - r["imp_p"]["ARGMAX|LEAF-K8"] for r in sub])
    out.update(DR.strata(bts, sub, gs, "unrev0_by_active_unseen",
                         lambda r: None if r["active_unseen_moves"] is None else
                         ("0-1" if r["active_unseen_moves"] <= 1 else ("2" if r["active_unseen_moves"] == 2 else "3-4"))))
    out.update(DR.strata(bts, sub, gs, "unrev0_by_seen_unseen_moves",
                         lambda r: "<=10" if r["seen_unseen_moves"] <= 10 else
                         ("11-14" if r["seen_unseen_moves"] <= 14 else "15+")))
    (HERE / "posthoc_reads.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for k, v in out.items():
        print(k)
        for lab, x in v.items():
            print(f"   {lab:16s} n={x['n']:4d} G {x['G']['mean']:+.3f} {x['G']['ci']} | O {x['imp_o']['mean']:+.3f} "
                  f"P {x['imp_p']['mean']:+.3f}")


if __name__ == "__main__":
    main()
