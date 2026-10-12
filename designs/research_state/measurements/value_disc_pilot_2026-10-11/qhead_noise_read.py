"""POST-HOC, DESCRIPTIVE (Phase 3's null, written after R-QHEAD was read; TRAINING states only, no evaluation state):
how much learnable signal do the K = 2 oracle labels carry?

- NOISE SHARE: with two playouts per action, the per-action noise variance is estimated without bias by ½·(o₁ − o₂)²/2
  (the variance of a K = 2 mean); its share of the centred label variance bounds the R² any predictor can reach
  (ceiling = 1 − noise share, on these labels);
- SIMPLE PREDICTORS, fit on the fit split and scored on the validation split (the Q head's own split): the centred
  label regressed (one slope, through 0) on the centred policy logit; on the ride-along A head; and on both.

    python -I qhead_noise_read.py end2   →  qhead_noise_read.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_qhead as Q  # noqa: E402

DEC = {"+": 1.0, "-": 0.0, "0": 0.5}


def main():
    cp = sys.argv[1]
    F = Q.load_feats(cp, "qbank")
    lab = Q.load_labels(cp)
    keep, Y = Q.label_matrix(F, lab)
    F = Q.subset(F, keep)
    battles = [str(b) for b in F["battle"]]
    val_b = Q.split(battles)
    is_val = np.array([b in val_b for b in battles])
    mask = F["mask"].astype(bool)
    n = mask.sum(1, keepdims=True)

    def cen(x):
        x = np.where(mask, x, 0.0)
        return (x - x.sum(1, keepdims=True) / n) * mask
    yc = cen(np.nan_to_num(Y))
    noise = np.zeros_like(Y)
    for i, did in enumerate(F["ids"]):
        for a, s in lab[str(did)].items():
            o = [DEC[ch] for ch in s]
            noise[i, int(a)] = (o[0] - o[1]) ** 2 / 4.0          # unbiased var of the K = 2 mean
    # the centred noise: each action's noise enters its centred label with weight (1 − 1/n)
    nshare = float(((noise * mask) * (1 - 1 / n)).sum() / (yc ** 2).sum())
    lg = cen(np.where(mask, F["logits"], 0.0).astype(np.float64))
    ad = cen(F["adv"].astype(np.float64))
    out = {"cp": cp, "states": int(len(keep)), "pairs": int(mask.sum()),
           "centred_label_var": round(float((yc ** 2).sum() / mask.sum()), 5),
           "noise_share_of_centred_var": round(nshare, 4), "r2_ceiling_on_these_labels": round(1 - nshare, 4)}
    tr, va = ~is_val, is_val
    for name, X in (("logit", [lg]), ("A head", [ad]), ("logit + A head", [lg, ad])):
        A = np.stack([x[tr][mask[tr]] for x in X], 1)
        b = np.linalg.lstsq(A, yc[tr][mask[tr]], rcond=None)[0]
        P = sum(c * x for c, x in zip(b, X))
        r2 = 1 - ((P[va] - yc[va])[mask[va]] ** 2).sum() / (yc[va][mask[va]] ** 2).sum()
        out[f"val_r2[{name}]"] = round(float(r2), 5)
        out[f"slope[{name}]"] = [round(float(x), 5) for x in b]
    (Q.HERE / "qhead_noise_read.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
