"""Which labels could have taught X68-FT its raises? Row-weighted label rates of watched move ids in the
training bank's two label tensors, and the species that carry the raises in the dump."""
import sys, os, numpy as np
A = os.path.expanduser("~/gen3ai_archive/offpool_belief_2026-11-09/x68")
D = os.path.expanduser("~/gen3ai_archive/move_leg_diag_2026-11-10/dump")
W = [253, 325, 222, 203, 214, 189, 102, 249, 201]
for b in ("train_0", "val"):
    z = np.load(f"{A}/{b}/rows.npz")
    km, bm, bs = z["known_moves"], z["belief_moves"], z["belief_species"]
    print(b, "rows", km.shape[0], "known_moves", km.shape, "belief_moves", bm.shape, "belief_species", bs.shape)
    for name, t in (("known", km), ("belief", bm)):
        v = t[t >= 0]
        cnt = np.bincount(v, minlength=400)
        print(f"  {name}: labelled entries {v.size}; watched:", {w: int(cnt[w]) for w in W},
              "typedHP(355-370):", int(cnt[355:371].sum()), "id237:", int(cnt[237]))
        top = np.argsort(-cnt)[:12]
        print("   top ids:", [(int(i), int(cnt[i])) for i in top])
for lab in ("pstar", "x68ft"):
    z = np.load(f"{D}/{lab}__p72.npz")
    t = np.load(f"{D}/{lab}__tables.npz")
    legal = t["meter_legal"] if "meter_legal" in t.files else t["move_legal"]
    sp, post, prior, shown = z["species"], z["post"], z["prior"], z["shown"]
    sig = lambda x: 1 / (1 + np.exp(-x))
    for w in (253, 325, 222, 214):
        cand = legal[sp, w] > 0
        for s4 in range(shown.shape[1]):
            cand &= shown[:, s4] != w
        if not cand.any():
            print(lab, w, "no candidates"); continue
        us = np.unique(sp[cand], return_counts=True)
        rows = []
        for s, c in zip(*us):
            m = cand & (sp == s)
            rows.append((int(s), int(c), float(sig(prior[m, w]).mean()), float(sig(post[m, w]).mean())))
        rows.sort(key=lambda r: -r[1])
        print(lab, "move", w, "n_cand", int(cand.sum()), "species(count, prior p, post p):",
              [(s, c, round(a, 3), round(b_, 3)) for s, c, a, b_ in rows[:10]])
