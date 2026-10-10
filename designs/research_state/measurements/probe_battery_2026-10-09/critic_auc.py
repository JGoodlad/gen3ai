"""The critic's OWN win-prob vs the game outcome on the bank (the reference for outcome_win's probe)."""
import json
import numpy as np
from pathlib import Path
from main.probe_battery import probes as P
from main.probe_battery.bank import load_decisions
from main.probe_battery.report import mean_ci
A = Path("/home/goodlad/gen3ai_archive/probe_battery")
decs = load_decisions(A / "bank_v1")
y = np.array([np.nan if d["winner"] not in ("p1", "p2") else float(d["winner"] == f"p{int(d['side']) + 1}") for d in decs])
ok = np.isfinite(y)
out = {}
for arm, labs in (("legacy", [f"L{i}" for i in range(1, 9)]), ("static", [f"S{i}" for i in range(1, 8)])):
    aucs, briers, ent = [], [], []
    for lab in labs:
        c = np.load(A / "caps_v1" / f"{lab}.npz")
        w = c["win_prob"]
        aucs.append(P.auc(y[ok], w[ok]))
        briers.append(float(np.mean((w[ok] - y[ok]) ** 2)))
        p = c["probs"].astype(np.float64)
        ent.append(float(np.mean(-(p * np.log(np.clip(p, 1e-12, 1))).sum(1))))
    out[arm] = {"critic_auc": mean_ci(aucs), "critic_brier": mean_ci(briers), "policy_entropy": mean_ci(ent)}
print(json.dumps(out, indent=1))
