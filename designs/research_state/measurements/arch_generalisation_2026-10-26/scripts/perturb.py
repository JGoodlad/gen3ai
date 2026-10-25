"""PERTURBATION probes on familiar (home72) decision rows, P* forward:

(a) BENCH SET SWAP — one never-yet-seen, full-HP bench mon of ours gets another set of ITS species: a FAMILIAR set
    (a pool set) or an UNFAMILIAR one (an H / F / P72 set the pool never had), the two matched on the difference
    pattern from the current set (moves differing, item differs). Only that slot's set columns change (item,
    abilities, the four move slots, spread, Hidden-Power block); species, state and every other block stay.
(b) BENCH PERMUTATION — our bench slots cyclically permuted (team slots, pair-history our axis, OBS-FACTS seen rows,
    the switch mask); an equivariant network moves the switch probabilities with their mons and nothing else.
Writes perturb.json."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import common as C
from agents.observation import constants as K
from static_probe import SLOT, signature, set_distance, turn1_slots

SET_SPANS = [(K.POKEMON_ITEMS_OFFSET, K.POKEMON_ITEMS_OFFSET + 3), (K.POKEMON_ABILITIES_OFFSET, K.POKEMON_ABILITIES_OFFSET + 4),
             (K.POKEMON_MOVES_OFFSET, K.POKEMON_MOVES_OFFSET + 44), (K.POKEMON_SPREAD_OFFSET, K.POKEMON_SPREAD_OFFSET + 18),
             (K.POKEMON_HP_BLOCK_OFFSET, K.POKEMON_HP_BLOCK_OFFSET + K.POKEMON_HP_BLOCK_DIM)]
OUR = K.OFFSET_OUR_TEAM


def pattern(cur, alt):
    sd = set_distance(cur, alt)
    it, spd = int(cur[2] != alt[2]), int(cur[4] != alt[4])
    return (sd - it - spd, it)


def cur_sig(slot):
    return signature(slot)


def swapped_in(cur, alt):
    """The alt set's moves the current set lacks (multiset difference)."""
    rest = list(cur[1])
    out = []
    for m in alt[1]:
        if m in rest:
            rest.remove(m)
        else:
            out.append(int(m))
    return out


def kl(p, q, m):
    p, q = np.clip(p[m], 1e-9, 1), np.clip(q[m], 1e-9, 1)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def main(n_rows: int = 600):
    from utils.team_sources import packed_teams
    lib = C.lib()
    model = C.load_model()
    tag = __import__("os").environ.get("PROBE_TAG", "")
    z = dict(np.load(C.OUT / f"rows_home72{tag}.npz"))
    pool = turn1_slots(packed_teams("pool"), lib).reshape(-1, SLOT)
    unf = np.concatenate([turn1_slots(C.pack(C.panel_texts(p)), lib).reshape(-1, SLOT) for p in ("H", "F", "P72")])
    psig = [signature(s) for s in pool]
    pool_set = set(psig)
    fam_by, unf_by = defaultdict(dict), defaultdict(dict)
    for s, v in zip(psig, pool):
        fam_by[s[0]].setdefault(s, v)
    for s, v in zip([signature(s) for s in unf], unf):
        if s not in pool_set:
            unf_by[s[0]].setdefault(s, v)
    seen_off = K.OFFSET_OBS_FACTS + 0
    seen_rows, seen_dim = 6, 7
    obs_all, mask_all = z["obs"], z["mask"]
    order = np.random.default_rng(1).permutation(len(obs_all))      # spread the sample over every game
    base_rows, fam_rows, unf_rows, perm_rows, meta = [], [], [], [], []
    for r in order:
        if len(base_rows) >= n_rows:
            break
        o = obs_all[r]
        slots = o[OUR:OUR + 6 * SLOT].reshape(6, SLOT)
        act = int(np.argmax(slots[:, K.POKEMON_ACTIVE_OFFSET]))
        seen = o[seen_off:seen_off + seen_rows * seen_dim].reshape(6, seen_dim)
        pick = None
        for b in range(6):
            if b == act or slots[b, K.POKEMON_HP_OFFSET] < 1.0 or seen[b].any():
                continue
            if slots[b, K.POKEMON_CONDITION_OFFSET + 1:K.POKEMON_CONDITION_OFFSET + K.CONDITION_DIM].any():
                continue
            cur = signature(slots[b])
            fams, unfs = defaultdict(list), defaultdict(list)
            for sg, v in sorted(fam_by[cur[0]].items(), key=lambda kv: str(kv[0])):
                if sg != cur:
                    fams[pattern(cur, sg)].append((sg, v))
            for sg, v in sorted(unf_by[cur[0]].items(), key=lambda kv: str(kv[0])):
                unfs[pattern(cur, sg)].append((sg, v))
            common = sorted(k for k in fams if k in unfs and k[0] >= 1)
            if common:
                k0 = common[0]
                h = (int(r) * 2654435761) % (2 ** 32)
                pick = (b, k0, fams[k0][h % len(fams[k0])], unfs[k0][(h // 7) % len(unfs[k0])])
                break
        bench = [b for b in range(6) if b != act]
        if pick is None or len(bench) < 2:
            continue
        b, pat, (fs, fv), (us, uv) = pick
        o_f, o_u = o.copy(), o.copy()
        for a0, a1 in SET_SPANS:
            o_f[OUR + b * SLOT + a0:OUR + b * SLOT + a1] = fv[a0:a1]
            o_u[OUR + b * SLOT + a0:OUR + b * SLOT + a1] = uv[a0:a1]
        # (b) cyclic bench permutation: slot bench[k] takes the content of bench[k-1]
        sigma = list(range(6))
        for k, s in enumerate(bench):
            sigma[s] = bench[k - 1]          # new slot s holds old slot sigma[s]
        o_p = o.copy()
        o_p[OUR:OUR + 6 * SLOT] = slots[sigma].reshape(-1)
        ph = o[K.OFFSET_PAIR_HISTORY:K.OFFSET_PAIR_HISTORY + K.PAIR_HISTORY_DIM].reshape(6, 6, -1)
        o_p[K.OFFSET_PAIR_HISTORY:K.OFFSET_PAIR_HISTORY + K.PAIR_HISTORY_DIM] = ph[:, sigma].reshape(-1)
        o_p[seen_off:seen_off + seen_rows * seen_dim] = seen[sigma].reshape(-1)
        m = mask_all[r]
        m_p = m.copy()
        m_p[:6] = m[:6][sigma]
        base_rows.append((o, m))
        fam_rows.append((o_f, m))
        unf_rows.append((o_u, m))
        perm_rows.append((o_p, m_p, sigma))
        meta.append({"row": int(r), "game": int(z["game"][r]), "slot": b, "pattern": pat, "species": int(fs[0]),
                     "fam_in": swapped_in(cur_sig(slots[b]), fs), "unf_in": swapped_in(cur_sig(slots[b]), us)})

    def run(rows):
        P, W = [], []
        for a in range(0, len(rows), 64):
            ch = rows[a:a + 64]
            p, w, _ = C.forward(model, np.stack([x[0] for x in ch]), np.stack([x[1] for x in ch]))
            P.append(p)
            W.append(w)
        return np.concatenate(P), np.concatenate(W)
    pb, wb = run(base_rows)
    pf, wf = run(fam_rows)
    pu, wu = run(unf_rows)
    pp, wp = run(perm_rows)
    res = {"n": len(base_rows)}
    for name, (p2, w2) in {"familiar_set": (pf, wf), "unfamiliar_set": (pu, wu)}.items():
        kls = [kl(pb[i], p2[i], base_rows[i][1]) for i in range(len(pb))]
        res[name] = {"kl_mean": round(float(np.mean(kls)), 4), "kl_median": round(float(np.median(kls)), 4),
                     "abs_dv": round(float(np.mean(np.abs(w2 - wb))), 4),
                     "signed_dv": round(float(np.mean(w2 - wb)), 4),
                     "flip": round(float(np.mean(p2.argmax(1) != pb.argmax(1))), 4),
                     "switch_to_swapped_mon_dprob": round(float(np.mean(
                         [p2[i][meta[i]["slot"]] - pb[i][meta[i]["slot"]] for i in range(len(pb))])), 4)}
    # paired difference unfamiliar - familiar, with a bootstrap over rows (seeded)
    kf = np.array([kl(pb[i], pf[i], base_rows[i][1]) for i in range(len(pb))])
    ku = np.array([kl(pb[i], pu[i], base_rows[i][1]) for i in range(len(pb))])
    dvf, dvu = np.abs(wf - wb), np.abs(wu - wb)
    rng = np.random.default_rng(0)
    games = np.array([m["game"] for m in meta])
    ug = np.unique(games)
    by_g = {g: np.flatnonzero(games == g) for g in ug}
    bs_kl, bs_dv = [], []
    for _ in range(2000):
        gg = rng.choice(ug, len(ug))
        ii = np.concatenate([by_g[g] for g in gg])
        bs_kl.append(np.mean(ku[ii] - kf[ii]))
        bs_dv.append(np.mean(dvu[ii] - dvf[ii]))
    res["n_games"] = int(len(ug))
    res["n_distinct_swaps"] = {"familiar": len({(m["species"], tuple(m["fam_in"])) for m in meta}),
                               "unfamiliar": len({(m["species"], tuple(m["unf_in"])) for m in meta})}

    def ci(x):
        return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]
    res["unf_minus_fam"] = {"kl": [round(float(np.mean(ku - kf)), 4), ci(bs_kl)],
                            "abs_dv": [round(float(np.mean(dvu - dvf)), 4), ci(bs_dv)],
                            "bootstrap": "2,000 resamples of GAMES (rows cluster by game)"}
    # the swapped-in moves' Smogon usage P(move in set | species): the rarest one, binned
    prior = model.policy.features_extractor.move_belief.move_prior_probs.numpy()

    def usage(m, key):
        mv = m[key]
        return min(float(prior[m["species"], x]) for x in mv) if mv else 1.0
    res["by_swapped_in_usage"] = {}
    for lo, hi in ((0.3, 2.0), (0.05, 0.3), (0.0, 0.05)):
        fi = [i for i, m in enumerate(meta) if lo <= usage(m, "fam_in") < hi]
        ui = [i for i, m in enumerate(meta) if lo <= usage(m, "unf_in") < hi]
        res["by_swapped_in_usage"][f"[{lo},{hi})"] = {
            "n_fam": len(fi), "kl_fam": round(float(kf[fi].mean()), 4) if fi else None,
            "dv_fam": round(float(dvf[fi].mean()), 4) if fi else None,
            "n_unf": len(ui), "kl_unf": round(float(ku[ui].mean()), 4) if ui else None,
            "dv_unf": round(float(dvu[ui].mean()), 4) if ui else None}
    res["by_pattern"] = {}
    for pat in sorted({tuple(m["pattern"]) for m in meta}):
        ii = [i for i, m in enumerate(meta) if tuple(m["pattern"]) == pat]
        res["by_pattern"][f"moves{pat[0]}_item{pat[1]}"] = {"n": len(ii), "kl_fam": round(float(kf[ii].mean()), 4),
                                                           "kl_unf": round(float(ku[ii].mean()), 4)}
    # permutation
    errs, dvs, kls = [], [], []
    for i, (o_p, m_p, sigma) in enumerate(perm_rows):
        expect = pb[i].copy()
        expect[:6] = pb[i][:6][sigma]
        errs.append(float(np.abs(pp[i] - expect).max()))
        dvs.append(abs(float(wp[i] - wb[i])))
        kls.append(kl(expect, pp[i], m_p))
    res["bench_permutation"] = {"max_abs_dprob": round(max(errs), 6), "mean_abs_dprob_max": round(float(np.mean(errs)), 6),
                                "max_abs_dv": round(max(dvs), 6), "mean_kl": round(float(np.mean(kls)), 6),
                                "flip": round(float(np.mean([pp[i].argmax() != np.argmax(
                                    np.r_[pb[i][:6][perm_rows[i][2]], pb[i][6:]]) for i in range(len(pp))])), 4)}
    json.dump({"result": res, "meta": meta}, open(C.OUT / f"perturb{tag}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 600)
