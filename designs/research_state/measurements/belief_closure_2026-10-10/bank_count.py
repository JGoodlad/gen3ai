"""How often does the FOUR-MOVE closure matter? Counts, over a probe-battery bank's decision rows, the
decisions where at least one opponent mon has all four of its moves revealed (read straight off the
observation's per-mon move-id slots, the cells the extractor reads as ``all_move_ids``).

Read-only; runs on CPU in seconds:

    python designs/research_state/measurements/belief_closure_2026-10-10/bank_count.py \
        ~/gen3ai_archive/probe_battery/bank_v1
"""
from __future__ import annotations

import json
import sys

import numpy as np


def count(bank: str) -> dict:
    L = json.load(open(f"{bank}/obs_layout.json"))
    rows = np.load(f"{bank}/rows.npy", mmap_mode="r")
    p = L["parts"]["opp_team"]
    per = p["reshape"][1]
    pk = L["pokemon"]
    mo = pk["moves"]["offset"]
    slots = [s["offset"] for s in pk["moves"]["layout"]["slots"]]
    hp = pk["hp"]["offset"]
    opp = np.asarray(rows[:, p["start"]:p["end"]]).reshape(len(rows), 6, per)
    ids = np.stack([opp[:, :, mo + s] for s in slots], -1).round().astype(int)   # [N,6,4]
    nrev = (ids > 0).sum(-1)                                                      # [N,6]
    alive = opp[:, :, hp] > 0
    act = opp[:, :, per - 1] > 0.5        # the per-mon tail's last cell is the ACTIVE flag
    four = nrev == 4
    out = dict(
        bank=bank, decisions=int(len(rows)),
        frac_any_opp_mon_4_revealed=float(four.any(1).mean()),
        frac_any_ALIVE_opp_mon_4_revealed=float((four & alive).any(1).mean()),
        frac_opp_ACTIVE_4_revealed=float((four & act).any(1).mean()),
        frac_alive_BENCH_mon_4_revealed=float((four & alive & ~act).any(1).mean()),
        frac_any_ALIVE_opp_mon_3plus_revealed=float(((nrev >= 3) & alive).any(1).mean()),
        rows_with_exactly_one_active=float((act.sum(1) == 1).mean()),
        hist_max_revealed_per_decision=np.bincount(nrev.max(1), minlength=5).tolist(),
        mean_mons_with_4_per_decision=float(four.sum(1).mean()),
    )
    return out


if __name__ == "__main__":
    print(json.dumps(count(sys.argv[1]), indent=1))
