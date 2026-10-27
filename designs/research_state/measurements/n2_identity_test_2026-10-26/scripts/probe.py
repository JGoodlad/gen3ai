"""PROBE (P* forward, CPU): the PRIMARY usage-draw swap and S3's mechanics-preserving perturbations on the side-0 rows
of the archived collections. Writes units.npz (one row per (row, target, draw)) and s3.json + probe_meta.json."""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

import n2common as N
from agents.observation import constants as K

C = N.C
PANELS = ("home72", "F", "H", "P72")
SLOT = K.POKEMON_FULL_DIM
SET_SPANS = [(K.POKEMON_ITEMS_OFFSET, K.POKEMON_ITEMS_OFFSET + 3), (K.POKEMON_ABILITIES_OFFSET, K.POKEMON_ABILITIES_OFFSET + 4),
             (K.POKEMON_MOVES_OFFSET, K.POKEMON_MOVES_OFFSET + 44), (K.POKEMON_SPREAD_OFFSET, K.POKEMON_SPREAD_OFFSET + 18),
             (K.POKEMON_HP_BLOCK_OFFSET, K.POKEMON_HP_BLOCK_OFFSET + K.POKEMON_HP_BLOCK_DIM)]
SET_COLS = np.concatenate([np.arange(a, b) for a, b in SET_SPANS])
CHUNK = 128


def kl(p, q, m):
    p, q = np.clip(p[m], 1e-9, 1), np.clip(q[m], 1e-9, 1)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def eligible_slots(o):
    slots = o[:6 * SLOT].reshape(6, SLOT)
    act = int(np.argmax(slots[:, K.POKEMON_ACTIVE_OFFSET]))
    seen = o[K.OFFSET_OBS_FACTS:K.OFFSET_OBS_FACTS + K.FACTS_SEEN_DIM].reshape(6, K.FACTS_SEEN_ROW_DIM)
    out = []
    for b in range(6):
        if b == act or slots[b, K.POKEMON_HP_OFFSET] < 1.0 or seen[b].any():
            continue
        if slots[b, K.POKEMON_CONDITION_OFFSET + 1:K.POKEMON_CONDITION_OFFSET + K.CONDITION_DIM].any():
            continue
        out.append(b)
    return slots, out


def main() -> None:
    from perturb import pattern
    from static_probe import signature
    assert K.FACTS_SEEN_ROW_DIM == 7 and K.FACTS_SEEN_DIM == 42
    dz = np.load(N.OUT / "draws.npz")
    vec = {k: v for k, v in zip(dz["keys"].tolist(), dz["vecs"])}
    orig = {p: dz[f"orig_{p}"] for p in PANELS}
    meta = json.load(open(N.OUT / "draws_meta.json"))
    sp_of = meta["species_of_slot"]
    model = C.load_model()
    stats = Counter()

    base_obs, base_mask, pert_obs, units, s3_jobs = [], [], [], [], []
    for gi, p in enumerate(PANELS):
        z = N.load_rows(p)
        group = 0 if p == "home72" else 1
        rows = np.flatnonzero(z["side"] == 0)
        s3_row_counter = 0
        for r in rows:
            o = z["obs"][r]
            slots, elig = eligible_slots(o)
            i = int(z["p1team"][r])
            row_targets = []
            for s in elig:
                spc = int(round(slots[s, K.POKEMON_SPECIES_OFFSET]))
                b = sp_of[p][i].index(spc)
                draws = [d for d in range(3) if f"draw|{p}|{i}|{b}|{d}" in vec]
                if not draws:
                    stats[f"{p}_no_target_or_draw"] += 1        # wrong familiarity, ineligible species, or no draw
                    continue
                if not np.array_equal(orig[p][i, b, SET_COLS], slots[s, SET_COLS]):
                    stats[f"{p}_splice_guard_fail"] += 1
                    continue
                stats[f"{p}_targets"] += 1
                row_targets.append((s, b, draws))
            if not row_targets:
                continue
            ri = len(base_obs)
            base_obs.append(o)
            base_mask.append(z["mask"][r])
            for s, b, draws in row_targets:
                cur = signature(slots[s])
                for d in draws:
                    v = vec[f"draw|{p}|{i}|{b}|{d}"]
                    po = o.copy()
                    po[s * SLOT + SET_COLS] = v[SET_COLS]
                    pat = pattern(cur, signature(v))
                    units.append((ri, len(pert_obs), group, gi, int(z["game"][r]), s, spc_of(slots, s), pat[0], pat[1], d))
                    pert_obs.append(po)
            # S3: every 4th eligible row per group, its first target
            if s3_row_counter % 4 == 0:
                s, b, _ = row_targets[0]
                for kind in ("reverse", "nickname", "ev_equiv"):
                    key = f"s3|{p}|{i}|{b}|{kind}"
                    if key in vec:
                        po = o.copy()
                        po[s * SLOT + SET_COLS] = vec[key][SET_COLS]
                        s3_jobs.append((ri, group, kind, s, bool(np.array_equal(po, o)), po))
            s3_row_counter += 1

    def run(obs_list, mask_list):
        P, W = [], []
        for a in range(0, len(obs_list), CHUNK):
            pr, w, _ = C.forward(model, np.stack(obs_list[a:a + CHUNK]), np.stack(mask_list[a:a + CHUNK]))
            P.append(pr)
            W.append(w)
        return np.concatenate(P), np.concatenate(W)

    pb, wb = run(base_obs, base_mask)
    u = np.array(units, dtype=np.int64)
    pp, wp = run(pert_obs, [base_mask[ri] for ri in u[:, 0]])
    m = [base_mask[ri] for ri in u[:, 0]]
    klv = np.array([kl(pb[ri], pp[j], m[j]) for j, ri in enumerate(u[:, 0])])
    flip = (pp.argmax(1) != pb[u[:, 0]].argmax(1)).astype(np.int8)
    dv = wp - wb[u[:, 0]]
    dsw = pp[np.arange(len(u)), u[:, 5]] - pb[u[:, 0], u[:, 5]]
    np.savez_compressed(N.OUT / "units.npz", row=u[:, 0], group=u[:, 2], panel=u[:, 3], game=u[:, 4], slot=u[:, 5],
                        species=u[:, 6], n_moves=u[:, 7], item=u[:, 8], draw=u[:, 9], kl=klv, flip=flip, dv=dv,
                        dswitch=dsw)
    # S3
    s3 = {}
    if s3_jobs:
        ps, ws = run([j[5] for j in s3_jobs], [base_mask[j[0]] for j in s3_jobs])
        for kind in ("reverse", "nickname", "ev_equiv"):
            for grp in (0, 1):
                ii = [k for k, j in enumerate(s3_jobs) if j[2] == kind and j[1] == grp]
                if not ii:
                    continue
                kls = [kl(pb[s3_jobs[k][0]], ps[k], base_mask[s3_jobs[k][0]]) for k in ii]
                s3[f"{kind}_{'pool' if grp == 0 else 'unfamiliar'}"] = {
                    "n": len(ii), "obs_identical_frac": round(float(np.mean([s3_jobs[k][4] for k in ii])), 4),
                    "max_abs_dprob": float(max(np.abs(ps[k] - pb[s3_jobs[k][0]]).max() for k in ii)),
                    "kl_mean": float(np.mean(kls)), "kl_max": float(np.max(kls)),
                    "flip": round(float(np.mean([ps[k].argmax() != pb[s3_jobs[k][0]].argmax() for k in ii])), 4),
                    "abs_dv_mean": float(np.mean([abs(ws[k] - wb[s3_jobs[k][0]]) for k in ii]))}
    json.dump(s3, open(N.OUT / "s3.json", "w"), indent=1)
    json.dump({"stats": dict(stats), "n_rows": len(base_obs), "n_units": int(len(u))},
              open(N.OUT / "probe_meta.json", "w"), indent=1)
    print(json.dumps({"stats": dict(stats), "n_units": int(len(u)), "s3": s3}, indent=1))


def spc_of(slots, s) -> int:
    return int(round(slots[s, K.POKEMON_SPECIES_OFFSET]))


if __name__ == "__main__":
    main()
    sys.exit(0)
