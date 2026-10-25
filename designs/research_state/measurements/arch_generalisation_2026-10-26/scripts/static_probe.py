"""STATIC probes (no games): (1) the capacity census of P*; (2) where the static identity S's first layer gets its
input variance (identity embeddings vs compositional features); (3) the geometry of OUR-side S tokens: does an
unfamiliar set land as far from the pool's sets of its species as its set difference predicts?

Turn-1 rows: one reset per team (p1 = the team), the p1 observation's six own slots (every set field known).
Writes static_probe.json beside the scripts' parent."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

import common as C
from agents.observation import constants as K

SLOT = K.POKEMON_FULL_DIM
MOVE0 = K.POKEMON_MOVES_OFFSET
MSD = 11


def turn1_slots(packed, lib):
    """[T,6,SLOT] p1's own six slots at the first decision of a game with each team as p1."""
    from utils.rust_env import ffi
    from utils.rust_env import protocol as P
    n = 48
    out = []
    for a in range(0, len(packed), n):
        chunk = packed[a:a + n]
        teams = chunk + [packed[(a + n + 7) % len(packed)]]
        m = len(chunk)
        spec = P.spec_json(n=m, threads=1, teams=teams, names=("argena", "argenb"), turn_limit=250,
                           refusal_budget=4, bank_dir=None, opponents=[{"kind": "external"}])
        core = ffi.FfiCore(spec, lib=lib)
        c = core.cols
        for e in range(m):
            c["ep_team"][e] = [e, m]
            c["ep_seed"][e] = [1, 2, 3, 4]
            c["ep_opp"][e] = 0
        core.reset()
        assert c["need"][:m, 0].all()
        obs = c["obs"][:m, 0].copy()
        core.close()
        out.append(obs[:, K.OFFSET_OUR_TEAM:K.OFFSET_OUR_TEAM + 6 * SLOT].reshape(m, 6, SLOT))
    return np.concatenate(out)


def signature(slot):
    """A set's identity from its slot: species, the sorted move ids, item, abilities, the 18 spread columns."""
    mv = tuple(sorted(int(round(slot[MOVE0 + MSD * j])) for j in range(4)))
    sp = tuple(np.round(slot[K.POKEMON_SPREAD_OFFSET:K.POKEMON_SPREAD_OFFSET + 18], 3).tolist())
    return (int(round(slot[K.POKEMON_SPECIES_OFFSET])), mv, int(round(slot[K.POKEMON_ITEMS_OFFSET])),
            int(round(slot[K.POKEMON_ABILITIES_OFFSET])), sp)


def set_distance(a, b):
    """#moves differing (0-4) + item differs + spread differs (each 0/1), same species assumed."""
    ma, mb = list(a[1]), list(b[1])
    common = 0
    for m in ma:
        if m in mb:
            mb.remove(m)
            common += 1
    return (4 - common) + int(a[2] != b[2]) + int(a[4] != b[4])


def s_inputs(pe, emb, pp):
    """S's input blocks for slots pp [1,N,SLOT] (our side: every move legal), by name, in s_in order."""
    from agents.model.extractor_ctx import slice_pokemon_categoricals
    ids = slice_pokemon_categoricals(pp, pe.layout)
    # reuse encode for the move set; recompute the S blocks the same way encode does
    B, N = pp.shape[0], pp.shape[1]
    legal = pp.new_ones(B, N, pe.num_moves)
    act = pp.new_zeros(B, N, pe._active_ctx_dim)
    parts = pe.encode(pp, ids, emb, legal, act)
    rem, known, _ = pe._move_columns(pp)
    mv_emb = emb.move_embedding(ids["all_move_ids"])
    from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
    mv_type = emb.type_embedding(ids["all_move_type_ids"])
    is_hp = (ids["all_move_ids"] == HIDDEN_POWER_MOVE_NUM).unsqueeze(-1)
    soft = emb.hp_soft_type(ids["hp_probs"]).unsqueeze(2).expand(-1, -1, pe.num_moves, -1)
    mv_type = torch.where(is_hp, soft, mv_type)
    blocks = {
        "species_emb": emb.species_embedding(ids["species_ids"]),
        "stats": pe._actual_stats(pp, ids["species_ids"]),
        "item_emb": emb.item_embedding(ids["item_ids"]),
        "item_known": pp[..., pe._item_known:pe._item_known + 1],
        "type_set_emb": emb.type_embedding(ids["type1_ids"]) + emb.type_embedding(ids["type2_ids"]),
        "ability_embs": torch.cat([emb.ability_embedding(ids["ability1_ids"]),
                                   emb.ability_embedding(ids["ability2_ids"])], -1),
        "ability_dom_known": torch.cat([pp[..., pe._ab_dom:pe._ab_dom + 1],
                                        pp[..., pe._ab_known:pe._ab_known + 1]], -1),
        "move_set_pool": None,
        "species_known": pp[..., K.POKEMON_SPECIES_KNOWN_OFFSET:K.POKEMON_SPECIES_KNOWN_OFFSET + 1],
        "hp_block": pp[..., K.POKEMON_HP_BLOCK_OFFSET:K.POKEMON_HP_BLOCK_OFFSET + K.POKEMON_HP_BLOCK_DIM],
    }
    move_blocks = {"move_emb": mv_emb, "move_type_emb": mv_type, "move_feats": rem, "move_known": known,
                   "hp_probs_slot": _hp_slot(pe, pp)}
    if pe.move_latent:
        move_blocks["move_latent"] = pe.move_latent_encoder(mv_emb, mv_type, ids["all_move_ids"])
    return parts, blocks, move_blocks


def variance_shares(W, blocks, order):
    """Each block's own contribution to the first layer's pre-activation variance, trace(W_b Cov_b W_b^T),
    as a share of their sum (cross-block covariance left out; reported with the total for scale)."""
    col = 0
    res = {}
    for name in order:
        x = blocks[name]
        d = x.shape[-1]
        Wb = W[:, col:col + d]
        col += d
        X = x.reshape(-1, d).double()
        X = X - X.mean(0, keepdim=True)
        cov = X.T @ X / max(len(X) - 1, 1)
        res[name] = float(torch.trace(Wb.double() @ cov @ Wb.double().T))
    assert col == W.shape[1], (col, W.shape)
    tot = sum(res.values())
    return {k: round(v / tot, 4) for k, v in res.items()}, tot


def main():
    from utils.team_sources import packed_teams
    lib = C.lib()
    model = C.load_model()
    fe = model.policy.features_extractor
    pe, emb = fe.pokemon_encoder, fe.embeddings
    out = {}
    # ---- (1) census
    census = {name: sum(p.numel() for p in mod.parameters()) for name, mod in fe.named_children()}
    census = dict(sorted(census.items(), key=lambda kv: -kv[1]))
    pol = {"features_extractor": sum(p.numel() for p in fe.parameters()),
           "mlp_extractor": sum(p.numel() for p in model.policy.mlp_extractor.parameters()),
           "total_policy": sum(p.numel() for p in model.policy.parameters())}
    emb_tab = {n: list(m.weight.shape) for n, m in emb.named_children() if hasattr(m, "weight")}
    pe_parts = {n: sum(p.numel() for p in m.parameters()) for n, m in pe.named_children()}
    out["census"] = {"extractor_children": census, "policy": pol, "embedding_tables": emb_tab,
                     "static_encoder_children": pe_parts}
    # ---- turn-1 own slots
    groups = {"pool": packed_teams("pool")}
    for p in ("home72", "H", "F", "P72"):
        groups[p] = C.pack(C.panel_texts(p))
    slots = {g: turn1_slots(t, lib) for g, t in groups.items()}
    sigs = {g: [signature(s) for s in v.reshape(-1, SLOT)] for g, v in slots.items()}
    pool_sigs = set(sigs["pool"])
    pool_species = {s[0] for s in sigs["pool"]}
    pool_sm = {(s[0], s[1]) for s in sigs["pool"]}
    pool_moves_item = {(s[0], s[1], s[2]) for s in sigs["pool"]}
    nov = {}
    for g in ("home72", "H", "F", "P72"):
        ss = sigs[g]
        nov[g] = {"n_mons": len(ss),
                  "species_in_pool": round(np.mean([s[0] in pool_species for s in ss]), 3),
                  "moveset_in_pool": round(np.mean([(s[0], s[1]) in pool_sm for s in ss]), 3),
                  "moves_item_in_pool": round(np.mean([(s[0], s[1], s[2]) in pool_moves_item for s in ss]), 3),
                  "exact_set_in_pool": round(np.mean([s in pool_sigs for s in ss]), 3)}
    out["set_novelty"] = nov
    # ---- S tokens + (2) variance shares (pool rows)
    with torch.no_grad():
        S = {}
        for g, v in slots.items():
            pp = torch.tensor(v.reshape(1, -1, SLOT), dtype=torch.float32)
            parts, blocks, mblocks = s_inputs(pe, emb, pp)
            S[g] = parts.static[0].numpy()
            if g == "pool":
                # move set pool: what encode summed (recompute from the move network path)
                morder = ["move_emb", "move_type_emb", "move_feats", "move_known", "hp_probs_slot"] + \
                    (["move_latent"] if pe.move_latent else [])
                mv = pe.move_network(torch.cat([mblocks[k] for k in morder], -1))
                B, N = pp.shape[:2]
                mi = mv.reshape(B * N, 4, -1)
                d, _ = pe.move_self_attn(mi, mi, mi)
                blocks["move_set_pool"] = pe.move_self_norm(mi + d).reshape(B, N, 4, -1).sum(2)
                order = ["species_emb", "stats", "item_emb", "item_known", "type_set_emb", "ability_embs",
                         "ability_dom_known", "move_set_pool", "species_known", "hp_block"]
                W = pe.role_encoder[0].weight
                shares, tot = variance_shares(W, blocks, order)
                mW = pe.move_network[0].weight
                mshares, mtot = variance_shares(mW, mblocks, morder)
                if pe.move_latent:
                    ids = __import__("agents.model.extractor_ctx", fromlist=["x"]).slice_pokemon_categoricals(
                        pp, pe.layout)
                    lb = {"move_emb": mblocks["move_emb"], "move_type_emb": mblocks["move_type_emb"],
                          "MOVE_ATTR": pe.move_latent_encoder.MOVE_ATTR[ids["all_move_ids"]]}
                    lshares, ltot = variance_shares(pe.move_latent_encoder.mlp[0].weight, lb,
                                                    ["move_emb", "move_type_emb", "MOVE_ATTR"])
                    out["move_latent_first_layer_variance_share"] = {"shares": lshares, "total": ltot}
                out["S_first_layer_variance_share"] = {"role_encoder": shares, "role_total": tot,
                                                       "move_network": mshares, "move_total": mtot}
                # check our recomputed pool equals encode's (the static token from our blocks)
                s_in = torch.cat([blocks[k] for k in order], -1)
                err = float((pe.role_encoder(s_in) - parts.static).abs().max())
                out["S_recompute_max_abs_err"] = err
                assert err < 1e-4, err
    # ---- (3) geometry
    pool_by_species = defaultdict(list)
    seen = set()
    for i, s in enumerate(sigs["pool"]):
        if s in seen:
            continue
        seen.add(s)
        pool_by_species[s[0]].append((s, S["pool"][i]))
    rows = []

    def nn(sig, vec, exclude_identical):
        best = None
        for psig, pvec in pool_by_species.get(sig[0], []):
            if exclude_identical and psig == sig:
                continue
            d = float(np.linalg.norm(vec - pvec))
            sd = set_distance(sig, psig)
            if best is None or d < best[0]:
                best = (d, sd)
        return best
    # pool, leave-one-out (unique sets)
    for sp, lst in pool_by_species.items():
        for psig, pvec in lst:
            b = nn(psig, pvec, True)
            if b is not None:
                rows.append({"group": "pool_loo", "d": b[0], "sd_of_nn": b[1], "norm": float(np.linalg.norm(pvec))})
    for g in ("H", "F", "P72"):
        done = set()
        for i, s in enumerate(sigs[g]):
            if s in pool_sigs or s in done or s[0] not in pool_by_species:
                continue
            done.add(s)
            b = nn(s, S[g][i], False)
            # the set distance to the CLOSEST pool set by set distance (not by S distance)
            sd_min = min(set_distance(s, psig) for psig, _ in pool_by_species[s[0]])
            rows.append({"group": g, "d": b[0], "sd_of_nn": b[1], "sd_min": sd_min,
                         "norm": float(np.linalg.norm(S[g][i]))})
    for r in rows:
        if r["group"] == "pool_loo":
            r["sd_min"] = r["sd_of_nn"]
    geo = {}
    for g in ("pool_loo", "H", "F", "P72"):
        rr = [r for r in rows if r["group"] == g]
        geo[g] = {"n_sets": len(rr), "median_d_nn": round(float(np.median([r["d"] for r in rr])), 3),
                  "median_norm": round(float(np.median([r["norm"] for r in rr])), 3)}
        by = {}
        for k in range(0, 7):
            ds = [r["d"] for r in rr if r["sd_of_nn"] == k]
            if len(ds) >= 5:
                by[str(k)] = {"n": len(ds), "median_d": round(float(np.median(ds)), 3)}
        geo[g]["by_set_distance_of_nn"] = by
    # ratio: d_nn / set distance, for sd >= 1 (an S distance per unit of set difference)
    for g in geo:
        rr = [r["d"] / r["sd_of_nn"] for r in rows if r["group"] == g and r["sd_of_nn"] >= 1]
        geo[g]["median_d_per_set_unit"] = round(float(np.median(rr)), 3) if rr else None
    out["S_geometry"] = geo
    # matched set-difference PATTERNS: every (unique set, pool set of its species) pair, by
    # (moves differing, item differs, spread differs); S distance median per pattern, per group
    def pattern(a, b):
        sd = set_distance(a, b)
        it, spd = int(a[2] != b[2]), int(a[4] != b[4])
        return (sd - it - spd, it, spd)
    pat = defaultdict(lambda: defaultdict(list))
    for g in ("pool", "H", "F", "P72"):
        done = set()
        for i, sg in enumerate(sigs[g]):
            if sg in done or sg[0] not in pool_by_species or (g != "pool" and sg in pool_sigs):
                continue
            done.add(sg)
            for psig, pvec in pool_by_species[sg[0]]:
                if psig == sg:
                    continue
                pat[g][pattern(sg, psig)].append(float(np.linalg.norm(S[g][i] - pvec)))
    out["S_distance_by_difference_pattern"] = {
        g: {f"moves{k[0]}_item{k[1]}_spread{k[2]}": {"n_pairs": len(v), "median_d": round(float(np.median(v)), 3)}
            for k, v in sorted(pat[g].items()) if len(v) >= 10}
        for g in pat}
    json.dump(out, open(C.OUT / "static_probe.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("move_latent_first_layer_variance_share", "S_distance_by_difference_pattern")}, indent=1))


def _hp_slot(pe, pp):
    from agents.model.extractor_ctx import slice_pokemon_categoricals
    from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
    ids = slice_pokemon_categoricals(pp, pe.layout)
    hp = ids["hp_probs"]
    is_hp = (ids["all_move_ids"] == HIDDEN_POWER_MOVE_NUM).unsqueeze(-1)
    return torch.where(is_hp, hp.unsqueeze(2).expand(-1, -1, pe.num_moves, -1), torch.zeros((), dtype=hp.dtype))


if __name__ == "__main__":
    main()
