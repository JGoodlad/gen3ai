"""X65 learnability probe (2026-11-06, DESCRIPTIVE): choosing the CONTROL target. On every 8th label row of the H
panel (4,003 rows), each candidate policy's greedy action (the gate's forward, ``delivery._forward``) against P*'s:
how often it departs from P*, and how often it agrees with S'. The control is the candidate whose departure rate is
nearest S''s own (37.6 % on these rows). CPU, nice 19, cores 5-7,13-15, under mem_cap.

    PYTHONPATH=src python3 control_pick.py <run>/checkpoints/<ckpt>.zip [...]      # paths under main's models/
"""
import sys, numpy as np
from pathlib import Path
from main.x65 import bank as BK, delivery as DV
bank = Path.home() / "gen3ai_archive/x65_exit0_2026-11-05/bank"
z = BK.load_rows(bank, ["heldout72_hpfix"], ("obs", "mask", "label", "played", "pi_argmax"), labels=["pstar_s"])
l = np.flatnonzero(z["label"])[::8]
obs, mask = z["obs"][l], z["mask"][l]
M = Path("/home/goodlad/dev/gen3ai/models")
base = DV.masked_argmax(DV._forward(M / "rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip", obs, mask), mask)
print("rows", len(l), "P* offline == pi_argmax", (base == z["pi_argmax"][l]).mean(), "S' dep", (z["played"][l] != z["pi_argmax"][l]).mean())
for c in sys.argv[1:]:
    a = DV.masked_argmax(DV._forward(M / c, obs, mask), mask)
    dep = a != base
    print(c, "dep vs P*", round(dep.mean(), 4), "agree S'", round((a == z["played"][l]).mean(), 4),
          "at S' deps", round((a == z["played"][l])[z["played"][l] != base].mean(), 4), flush=True)
