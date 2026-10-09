"""Step 3 (H2 / H3): linear probes from captured trunk tokens to battle facts. DESCRIPTIVE.

For one capture (`cap/<label>.npz`, from capture.py) and every (seat, fact) pair below, at each of
three depths (trunk INPUT, after layer 1, after layer 2): a ridge probe (RidgeCV, alphas 1/10/100,
on standardised float32 tokens) cross-validated by BATTLE (GroupKFold 5, battles in bank order — no
shuffling, so the split is deterministic), out-of-fold predictions pooled → R² for a continuous fact,
ROC-AUC of the ridge score for a binary one. Rows where the seat is undefined (no unique active / no
alive bench mon) are dropped. Writes probe/<label>.json, then REWRITES the capture without its
`tok` array (disk).

Seats: OA our active mon token, TA their active mon token, OB our first alive bench mon token,
M0 our active's first move seat (E3).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ARCH = Path("/home/goodlad/gen3ai_archive/static_diag_2026-10-09")
POK, OUR0, OPP0, CTX0, G, R = 122, 0, 732, 1464, 1584, 1604
HP, COND, ACTIVE = 67, 16, 121
SEATS = {"OA": 0, "TA": 1, "OB": 2, "M0": 3}


def facts(rows, oa, ta, ob):
    """name -> (values [N], kind, mask [N]) — read off the obs rows at the pin's layout."""
    N = len(rows)
    ar = np.arange(N)
    mon = lambda base, i, off: rows[ar, base + np.maximum(i, 0) * POK + off]
    cond = lambda base, i: rows[ar[:, None], base + np.maximum(i, 0)[:, None] * POK + COND + np.arange(7)[None, :]]
    f = {}
    f["hp_self_OA"] = (mon(OUR0, oa, HP), "c")
    f["hp_self_TA"] = (mon(OPP0, ta, HP), "c")
    f["hp_self_OB"] = (mon(OUR0, ob, HP), "c")
    f["status_self_OA"] = ((cond(OUR0, oa)[:, 1:].sum(1) > 0.5).astype(float), "b")
    f["status_self_TA"] = ((cond(OPP0, ta)[:, 1:].sum(1) > 0.5).astype(float), "b")
    f["weather_any"] = ((1.0 - rows[:, G + 0]) > 0.5).astype(float), "b"
    f["spikes_our_side"] = (rows[:, G + 7], "c")
    f["spikes_their_side"] = (rows[:, G + 8], "c")
    f["screen_our_side"] = (((rows[:, G + 12] + rows[:, G + 14]) > 0.5).astype(float), "b")
    f["clock_elapsed"] = (rows[:, G + 9], "c")
    f["fainted_ours"] = (rows[:, R + 0], "c")
    f["fainted_theirs"] = (rows[:, R + 1], "c")
    f["our_active_boosted"] = ((np.abs(rows[:, CTX0:CTX0 + 14]).sum(1) > 1e-6).astype(float), "b")
    f["their_active_boosted"] = ((np.abs(rows[:, CTX0 + 60:CTX0 + 74]).sum(1) > 1e-6).astype(float), "b")
    f["their_active_hp"] = (mon(OPP0, ta, HP), "c")
    f["our_active_hp"] = (mon(OUR0, oa, HP), "c")
    return f


# (seat, fact) pairs: the fact's home under each arm is in the README's table.
PAIRS = [
    ("OA", "hp_self_OA"), ("TA", "hp_self_TA"), ("OB", "hp_self_OB"),
    ("OA", "status_self_OA"), ("TA", "status_self_TA"),
    ("OA", "weather_any"), ("TA", "weather_any"), ("OB", "weather_any"), ("M0", "weather_any"),
    ("OA", "spikes_our_side"), ("OB", "spikes_our_side"), ("TA", "spikes_their_side"),
    ("OA", "screen_our_side"), ("OB", "screen_our_side"),
    ("OA", "clock_elapsed"), ("OB", "clock_elapsed"), ("M0", "clock_elapsed"),
    ("OA", "fainted_ours"), ("OB", "fainted_ours"), ("OA", "fainted_theirs"),
    ("OA", "our_active_boosted"), ("OB", "our_active_boosted"), ("TA", "their_active_boosted"),
    ("OA", "their_active_hp"), ("TA", "our_active_hp"), ("M0", "our_active_hp"),
]


def probe(X, y, kind, groups):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=(1.0, 10.0, 100.0)).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    if kind == "c":
        return float(r2_score(y, oof))
    return float(roc_auc_score(y, oof))


def main(label):
    cap_path = ARCH / "cap" / f"{label}.npz"
    cap = dict(np.load(cap_path))
    if "tok" not in cap:
        print(f"[probe] {label}: no tok (already probed)")
        return 0
    rows = np.load(ARCH / "bank_rows.npz")["rows"]
    meta = json.loads((ARCH / "bank_meta.json").read_text())
    battles = {b: i for i, b in enumerate(dict.fromkeys(d["battle"] for d in meta))}
    groups = np.array([battles[d["battle"]] for d in meta])
    oa, ta, ob = cap["our_active"], cap["their_active"], cap["our_bench"]
    valid = {"OA": oa >= 0, "TA": ta >= 0, "OB": ob >= 0, "M0": oa >= 0}
    F = facts(rows, oa, ta, ob)
    tok = cap["tok"].astype(np.float32)
    out = {"label": label, "metric": "R2 (continuous) / AUC (binary), out-of-fold, GroupKFold(5) by battle",
           "probes": {}}
    for seat, fact in PAIRS:
        y, kind = F[fact]
        m = valid[seat] & ((ta >= 0) if fact in ("hp_self_TA", "status_self_TA", "their_active_hp") else True) \
            & ((oa >= 0) if fact in ("hp_self_OA", "status_self_OA", "our_active_hp") else True)
        yy, gg = y[m], groups[m]
        if kind == "b" and (yy.min() == yy.max() or min(yy.mean(), 1 - yy.mean()) < 0.01):
            continue
        res = {"kind": kind, "n": int(m.sum()), "base_rate": float(yy.mean())}
        for di, dn in enumerate(("in", "L1", "L2")):
            res[dn] = round(probe(tok[m, SEATS[seat], di], yy, kind, gg), 4)
        out["probes"][f"{seat}:{fact}"] = res
    (ARCH / "probe").mkdir(exist_ok=True)
    (ARCH / "probe" / f"{label}.json").write_text(json.dumps(out, indent=1))
    cap.pop("tok")
    np.savez(cap_path, **cap)
    print(f"[probe] {label}: {len(out['probes'])} probes written; tok stripped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
