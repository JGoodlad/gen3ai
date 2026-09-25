"""POST-REGISTRATION additions to the belief win-rate A/B (declared in README §5; NOT in PREDICTION.md).

    python post_registration.py <rows_dir> <out_dir>      # writes <out_dir>/post_registration.json

Written after the registered numbers were read. Nothing here can change the registered verdict.
  A. Owner addition: the side read at k = 3 and k = 5 revealed as NAMED strata, plus the pool - ladder
     difference in action-change rate (overall and at k = 3 / 5), battle-index bootstrap drawn jointly
     over both columns (ratio of sums; no correlation across battles).
  B. Owner addition: can the WIN-RATE effect be stratified by k? The only per-battle k in the rows is
     the LEARNED trajectory's final reveal count K_final (PRIOR-ONLY rows carry no side read). This
     tabulates WR_learned / WR_prior / E by K_final so the reader can see why that stratifier is a
     selection on the control arm's own outcome and not an effect at k.
  C. Sensitivity of the registered contrasts to the 4 dropped (INCONCLUSIVE) indices: each dropped
     index keeps its finished primary cells and its FAILED cells take the extreme 0/1 fill in each
     direction for each contrast; the point + CI are recomputed at n = 3,000.
  D. Sensitivity to the TURN CAP: a battle that reaches turn 250 ends with BOTH sides stall-forfeiting,
     and the sim always processes p1's FORCELOSE first (`gen3_cf_draw_at_cap_v1`), so the rows record
     it as OUR loss. It is a draw by construction. D re-scores every capped battle (turns >= 250) as
     1/2 and recomputes the registered primary + secondary contrasts.
"""
import itertools
import json
import os
import sys

import numpy as np

import analyze
from analyze import PRIMARY, ci, score, boot_weights

SEED = analyze.SEED + 7


def side_arrays(rows, idx, col):
    cnt = np.zeros((len(idx), 7))
    ch = {"all": np.zeros((len(idx), 7)), "move": np.zeros((len(idx), 7))}
    for a, i in enumerate(idx):
        for k, _off, s_all, s_mv in rows[(col, "learned", i)]["side"]:
            cnt[a, k] += 1
            ch["all"][a, k] += s_all
            ch["move"][a, k] += s_mv
    return cnt, ch


def part_a(rows, idx):
    cp, hp = side_arrays(rows, idx, "pool")
    cl, hl = side_arrays(rows, idx, "ladder")
    rng = np.random.default_rng(SEED)
    ws = list(boot_weights(len(idx), rng))
    out = {}
    for scope in ("all", "move"):
        for kk in ("3", "5", "all"):
            sel = (lambda m: m.sum(1)) if kk == "all" else (lambda m, k=int(kk): m[:, k])
            c_p, h_p, c_l, h_l = sel(cp), sel(hp[scope]), sel(cl), sel(hl[scope])
            rp = np.concatenate([(w @ h_p) / (w @ c_p) for w in ws])
            rl = np.concatenate([(w @ h_l) / (w @ c_l) for w in ws])
            pt_p, pt_l = h_p.sum() / c_p.sum(), h_l.sum() / c_l.sum()
            d = rp - rl
            c_ = ci(d)
            out[f"{scope}:k={kk}"] = {
                "pool": {"rate": float(pt_p), "ci": ci(rp), "n_decisions": int(c_p.sum())},
                "ladder": {"rate": float(pt_l), "ci": ci(rl), "n_decisions": int(c_l.sum())},
                "pool_minus_ladder": {"point": float(pt_p - pt_l), "ci": c_,
                                      "read": analyze.tag(pt_p - pt_l, c_)}}
    return out


def part_b(rows, idx):
    out = {}
    for col in ("pool", "ladder"):
        strata = {}
        for i in idx:
            K = max(s[0] for s in rows[(col, "learned", i)]["side"])
            strata.setdefault(K, []).append(i)
        per = {}
        for K in sorted(strata):
            ii = strata[K]
            wl = float(np.mean([score(rows[(col, "learned", i)]) for i in ii]))
            wp = float(np.mean([score(rows[(col, "prior", i)]) for i in ii]))
            per[str(K)] = {"n_battles": len(ii), "WR_learned": wl, "WR_prior": wp, "E_point": wp - wl}
        out[col] = per
    return out


def contrasts(Y):
    """Y [n, 4] in PRIMARY order -> (E_pool, E_ladder, DiD)."""
    e_pool = Y[:, 1] - Y[:, 0]
    e_lad = Y[:, 3] - Y[:, 2]
    return {"E_pool": e_pool, "E_ladder": e_lad, "DiD": e_lad - e_pool}


def part_c(rows, idx_ok, dropped):
    base = np.array([[score(rows[(c[0], c[1], i)]) for c in PRIMARY] for i in idx_ok])
    out = {"dropped_indices": dropped}
    rng = np.random.default_rng(SEED + 1)
    n = len(idx_ok) + len(dropped)
    ws = list(boot_weights(n, rng))
    for name in ("E_pool", "E_ladder", "DiD"):
        res = {}
        for side, sign in (("worst_low", -1), ("worst_high", +1)):
            # choose, per dropped index, the 0/1 fill of the 4 cells that pushes this contrast furthest
            # per dropped index: the primary cells that DID finish keep their observed score; only the
            # failed cells are free, and each takes the 0/1 fill that pushes this contrast furthest
            fills = []
            for i in dropped:
                known = [score(rows[(c[0], c[1], i)]) if rows[(c[0], c[1], i)]["status"] == "ok" else None
                         for c in PRIMARY]
                free = [j for j, v in enumerate(known) if v is None]
                cands = []
                for f in itertools.product((0.0, 1.0), repeat=len(free)):
                    row = list(known)
                    for j, v in zip(free, f):
                        row[j] = v
                    cands.append(row)
                fills.append(max(cands, key=lambda r: sign * contrasts(np.array([r]))[name][0]))
            Y = np.vstack([base, np.array(fills)])
            v = contrasts(Y)[name]
            samp = np.concatenate([w @ v / n for w in ws])
            res[side] = {"point": float(v.mean()), "ci": ci(samp)}
        out[name] = res
    return out


CAP = 250


def part_d(rows, idx):
    cells = PRIMARY + analyze.SECONDARY

    def sc(r):
        return 0.5 if (r["turns"] or 0) >= CAP else score(r)
    Y = np.array([[sc(rows[(c[0], c[1], i)]) for c in cells] for i in idx])
    capped = {f"{c[0]}:{c[1]}": int(sum(1 for i in idx if (rows[(c[0], c[1], i)]["turns"] or 0) >= CAP))
              for c in cells}
    capped_recorded_winner = sorted({rows[(c[0], c[1], i)]["winner"] for c in cells for i in idx
                                     if (rows[(c[0], c[1], i)]["turns"] or 0) >= CAP})
    n = len(idx)
    rng = np.random.default_rng(SEED + 2)
    samp = np.concatenate([w @ Y / n for w in boot_weights(n, rng)])
    col = {f"{c[0]}:{c[1]}": j for j, c in enumerate(cells)}
    defs = {"E_pool": [("pool:prior", 1), ("pool:learned", -1)],
            "E_ladder": [("ladder:prior", 1), ("ladder:learned", -1)],
            "DiD": [("ladder:prior", 1), ("ladder:learned", -1), ("pool:prior", -1), ("pool:learned", 1)],
            "E^move_pool": [("pool:move", 1), ("pool:learned", -1)],
            "E^move_ladder": [("ladder:move", 1), ("ladder:learned", -1)],
            "DiD^move": [("ladder:move", 1), ("ladder:learned", -1), ("pool:move", -1), ("pool:learned", 1)]}
    pt = Y.mean(0)
    out = {"capped_per_cell": capped, "capped_recorded_winner_values": capped_recorded_winner,
           "wr": {k: float(pt[j]) for k, j in col.items()}, "contrasts": {}}
    for name, terms in defs.items():
        p_ = float(sum(s_ * pt[col[k]] for k, s_ in terms))
        c_ = ci(sum(s_ * samp[:, col[k]] for k, s_ in terms))
        out["contrasts"][name] = {"point": p_, "ci": c_, "read": analyze.tag(p_, c_)}
    return out


def main(rows_dir, out_dir):
    rows = analyze.load(rows_dir)
    idx_all = sorted({k[2] for k in rows})
    cells = PRIMARY + analyze.SECONDARY
    ok = [i for i in idx_all if all(rows[(c[0], c[1], i)]["status"] == "ok" for c in cells)]
    dropped = [i for i in idx_all if i not in set(ok)]
    res = {"A_side_read_named_strata": part_a(rows, ok),
           "B_wr_by_learned_K_final": part_b(rows, ok),
           "C_drop_sensitivity": part_c(rows, ok, dropped),
           "D_turn_cap_as_draw": part_d(rows, ok)}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "post_registration.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
