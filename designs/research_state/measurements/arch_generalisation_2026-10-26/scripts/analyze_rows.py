"""ROW probes on collect.py's rows (P* forward, CPU): belief posterior vs its own Smogon prior (moves on revealed
opponent mons, hidden species, hidden items), the critic's calibration, and identity-vs-feature MEAN ABLATIONS of the
static identity S. Usage: analyze_rows.py <panel> [n_ablation_rows]. Writes rows_<panel>_analysis.json."""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import torch

import common as C
from agents.observation import constants as K
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM

SLOT = K.POKEMON_FULL_DIM
MOVE0 = K.POKEMON_MOVES_OFFSET
HP_FAMILY = set([HIDDEN_POWER_MOVE_NUM] + list(range(355, 371)))
CHUNK = 64


# ----------------------------------------------------------------------------------- the patched S encoder
def patched_encode(ablate: str, means: dict):
    """StaticTokenEncoder.encode (static_tokens.py:181-245) verbatim except ONE input replaced by a constant mean."""
    from agents.model.arch_constants import MOVE_NET_HIDDEN

    def encode(self, pp, ids, embeddings, move_legal, active_ctx):
        B, N = pp.shape[0], pp.shape[1]
        nm = self.num_moves
        rem, known, cur_pp = self._move_columns(pp)
        mv_emb = embeddings.move_embedding(ids["all_move_ids"])
        if ablate == "move_id":
            real = (ids["all_move_ids"] > 0).unsqueeze(-1)
            mv_emb = torch.where(real, means["move_emb"].expand_as(mv_emb), mv_emb)
        mv_type = embeddings.type_embedding(ids["all_move_type_ids"])
        hp_probs = ids["hp_probs"]
        is_hp = (ids["all_move_ids"] == HIDDEN_POWER_MOVE_NUM).unsqueeze(-1)
        soft = embeddings.hp_soft_type(hp_probs).unsqueeze(2).expand(-1, -1, nm, -1)
        mv_type = torch.where(is_hp, soft, mv_type)
        hp_per_slot = torch.where(is_hp, hp_probs.unsqueeze(2).expand(-1, -1, nm, -1),
                                  torch.zeros((), dtype=hp_probs.dtype))
        if ablate == "move_feats":
            real = (ids["all_move_ids"] > 0).unsqueeze(-1)
            rem = torch.where(real, means["move_rem"].expand_as(rem), rem)
        blocks = [mv_emb, mv_type, rem, known, hp_per_slot]
        if self.move_latent:
            le = self.move_latent_encoder
            attr = le.MOVE_ATTR[ids["all_move_ids"]]
            if ablate == "move_feats":
                attr = torch.where(real, means["move_attr"].expand_as(attr), attr)
            blocks.append(le.mlp(torch.cat([mv_emb, mv_type, attr], dim=-1)))
        mv = self.move_network(torch.cat(blocks, dim=-1))
        mv_in = mv.reshape(B * N, nm, MOVE_NET_HIDDEN[1])
        mv_delta, _ = self.move_self_attn(mv_in, mv_in, mv_in)
        m_static = self.move_self_norm(mv_in + mv_delta).reshape(B, N, nm, MOVE_NET_HIDDEN[1])
        move_set = m_static.sum(dim=2)
        sp_emb = embeddings.species_embedding(ids["species_ids"])
        realmon = (ids["species_ids"] > 0).unsqueeze(-1)
        if ablate == "species_id":
            sp_emb = torch.where(realmon, means["species_emb"].expand_as(sp_emb), sp_emb)
        stats = self._actual_stats(pp, ids["species_ids"])
        if ablate == "stats":
            stats = torch.where(realmon, means["stats"].expand_as(stats), stats)
        it_emb = embeddings.item_embedding(ids["item_ids"])
        if ablate == "item_id":
            it_emb = torch.where((ids["item_ids"] > 0).unsqueeze(-1), means["item_emb"].expand_as(it_emb), it_emb)
        s_in = torch.cat([
            sp_emb, stats, it_emb,
            pp[..., self._item_known:self._item_known + 1],
            embeddings.type_embedding(ids["type1_ids"]) + embeddings.type_embedding(ids["type2_ids"]),
            embeddings.ability_embedding(ids["ability1_ids"]), embeddings.ability_embedding(ids["ability2_ids"]),
            pp[..., self._ab_dom:self._ab_dom + 1], pp[..., self._ab_known:self._ab_known + 1],
            move_set,
            pp[..., K.POKEMON_SPECIES_KNOWN_OFFSET:K.POKEMON_SPECIES_KNOWN_OFFSET + 1],
            pp[..., K.POKEMON_HP_BLOCK_OFFSET:K.POKEMON_HP_BLOCK_OFFSET + K.POKEMON_HP_BLOCK_DIM],
        ], dim=-1)
        static = self.role_encoder(s_in)
        mdyn = torch.cat([cur_pp, move_legal.unsqueeze(-1).to(pp.dtype)], dim=-1)
        move_tokens = m_static + self.move_dyn_proj(mdyn)
        pool = torch.relu(self.move_dyn_pool(torch.cat([m_static, mdyn], dim=-1))).sum(dim=2)
        la = K.POKEMON_LAST_ACTION_OFFSET
        d_in = torch.cat([
            pp[..., K.POKEMON_HP_OFFSET:K.POKEMON_HP_OFFSET + 1],
            pp[..., K.POKEMON_CONDITION_OFFSET:K.POKEMON_CONDITION_OFFSET + K.CONDITION_DIM],
            pp[..., K.POKEMON_COUNTER_OFFSET:K.POKEMON_COUNTER_OFFSET + K.POKEMON_COUNTER_DIM],
            pp[..., K.POKEMON_SLEEP_BELIEF_OFFSET:K.POKEMON_SLEEP_BELIEF_OFFSET + K.POKEMON_SLEEP_BELIEF_DIM],
            pp[..., K.POKEMON_RECENCY_OFFSET:K.POKEMON_RECENCY_OFFSET + K.POKEMON_RECENCY_DIM],
            pp[..., K.POKEMON_PROTECT_OFFSET:K.POKEMON_PROTECT_OFFSET + 1],
            embeddings.move_embedding(ids["last_move_ids"]),
            pp[..., la + 1:la + K.POKEMON_LAST_ACTION_DIM],
            pp[..., K.POKEMON_TRAPPED_OFFSET:K.POKEMON_MAYBE_TRAPPED_OFFSET + 1],
            pp[..., K.POKEMON_ACTIVE_OFFSET:K.POKEMON_ACTIVE_OFFSET + 1],
            pp[..., self._item_consumed:self._item_consumed + 1],
            active_ctx, pool,
        ], dim=-1)
        dynamic = self.dynamic_encoder(d_in)
        from agents.model.static_tokens import StaticParts
        return StaticParts(tokens=static + dynamic, static=static, dynamic=dynamic, move_tokens=move_tokens)
    return encode


def ablation_means(model, obs):
    """Constant means over every REAL mon slot (both sides) of ``obs``: species / item / move embeddings, the
    13-d stat input, the move remnants and MOVE_ATTR."""
    from agents.model.extractor_ctx import slice_pokemon_categoricals
    fe = model.policy.features_extractor
    pe, emb = fe.pokemon_encoder, fe.embeddings
    pp = torch.tensor(obs[:, :12 * SLOT].reshape(1, -1, SLOT), dtype=torch.float32)
    ids = slice_pokemon_categoricals(pp, pe.layout)
    with torch.no_grad():
        real = ids["species_ids"][0] > 0
        mreal = ids["all_move_ids"][0] > 0
        ireal = ids["item_ids"][0] > 0
        rem, _, _ = pe._move_columns(pp)
        m = {"species_emb": emb.species_embedding(ids["species_ids"][0][real]).mean(0),
             "stats": pe._actual_stats(pp, ids["species_ids"])[0][real].mean(0),
             "item_emb": emb.item_embedding(ids["item_ids"][0][ireal]).mean(0),
             "move_emb": emb.move_embedding(ids["all_move_ids"][0][mreal]).mean(0),
             "move_rem": rem[0][mreal].mean(0)}
        if pe.move_latent:
            m["move_attr"] = pe.move_latent_encoder.MOVE_ATTR[ids["all_move_ids"][0][mreal]].mean(0)
    return m


# ----------------------------------------------------------------------------------- beliefs
def capture_hypothesis(fe):
    """Wrap HypothesisBuilder.species_set to keep its inputs (the T0 log-prior and the candidate set)."""
    hb = fe.hypothesis_builder
    orig = hb.species_set
    box = {}

    def species_set(t0_log_prior, opp_species_ids, opp_believed_mask, *a, **kw):
        box["t0"], box["sp"], box["bel"] = t0_log_prior.detach(), opp_species_ids.detach(), opp_believed_mask.detach()
        return orig(t0_log_prior, opp_species_ids, opp_believed_mask, *a, **kw)
    hb.species_set = species_set
    return box


def prior_presence(fe, box):
    from agents.model.hypothesis_set import fixed_mass_presence, species_candidates
    hb = fe.hypothesis_builder
    believed = box["bel"].bool()
    cand, k = species_candidates(hb.species_valid, box["sp"], believed)
    return fixed_mass_presence(box["t0"], cand, k, solve=hb.tau_solve).pi, cand, k


def opp_slots(obs):
    return obs[:, K.OFFSET_OPP_TEAM:K.OFFSET_OPP_TEAM + 6 * SLOT].reshape(-1, 6, SLOT)


def main(panel: str, n_abl: int = 640) -> None:
    tag = __import__("os").environ.get("PROBE_TAG", "")
    z = dict(np.load(C.OUT / f"rows_{panel}{tag}.npz"))
    zh = dict(np.load(C.OUT / f"rows_home72{tag}.npz"))
    model = C.load_model()
    fe = model.policy.features_extractor
    box = capture_hypothesis(fe)
    obs, mask = z["obs"], z["mask"]
    n = len(obs)
    side = z["side"]
    won = (z["winner"] == side + 1).astype(float)
    tie = z["winner"] == 0
    win_p = np.zeros(n)
    probs_all = np.zeros((n, 11))
    mv_rows = []          # (side, prec_post, prec_prior, bce_post, bce_prior)
    sp_rows = []          # (side, hit_post, hit_prior, ll_post, ll_prior)
    it_rows = []          # (side, ce_post, ce_prior, acc_post, acc_prior)
    mprior = fe.move_belief.move_prior_logits
    iprior = fe.item_belief_head.item_prior
    legal_floor = float(torch.logit(torch.tensor(2e-6)))
    for a in range(0, n, CHUNK):
        sl = slice(a, a + CHUNK)
        p, w, _ = C.forward(model, obs[sl], mask[sl])
        probs_all[sl], win_p[sl] = p, w
        ml = fe.last_move_belief_logits.numpy()
        il = torch.log_softmax(fe.last_item_logits, -1).numpy()
        hyp = fe.last_hypothesis
        pi_post = hyp.species.pi.numpy()
        pi_prior, cand, _ = prior_presence(fe, box)
        pi_prior, cand = pi_prior.numpy(), cand.numpy()
        os_ = opp_slots(obs[sl])
        km, bs, il_lab, il_m = z["known_moves"][sl], z["belief_species"][sl], z["item_label"][sl], z["item_mask"][sl]
        for r in range(len(os_)):
            sd = int(side[a + r])
            # moves on REVEALED opp mons
            for i in range(6):
                if km[r, i, 0] < 0:
                    continue
                spc = int(round(os_[r, i, K.POKEMON_SPECIES_OFFSET]))
                truth = {int(m) for m in km[r, i] if m >= 0} - HP_FAMILY
                rev = {int(round(os_[r, i, MOVE0 + 11 * j])) for j in range(4) if os_[r, i, MOVE0 + 11 * j + 6] > 0.5}
                U = truth - rev
                if not U or spc <= 0:
                    continue
                pr = mprior[spc].numpy()
                candm = np.flatnonzero(pr > legal_floor)
                candm = np.array([m for m in candm if m not in rev and m not in HP_FAMILY])
                if not set(U) <= set(candm.tolist()):
                    continue        # a true move the learnset gate calls illegal: excluded (counted below)
                y = np.isin(candm, list(U)).astype(float)
                res = []
                for lg in (ml[r, i, candm], pr[candm]):
                    top = candm[np.argsort(-lg, kind="stable")[:len(U)]]
                    prec = len(set(top.tolist()) & U) / len(U)
                    pz = 1 / (1 + np.exp(-lg))
                    pz = np.clip(pz, 1e-7, 1 - 1e-7)
                    bce = float(-(y * np.log(pz) + (1 - y) * np.log(1 - pz)).sum() / max(len(U), 1))
                    res += [prec, bce]
                mv_rows.append((sd, res[0], res[2], res[1], res[3]))
            # hidden species
            H = {int(s) for s in bs[r] if s >= 0}
            if H:
                kkr = len(H)
                c_ = cand[r]
                res = []
                for pi in (pi_post[r], pi_prior[r]):
                    pc = np.where(c_, pi, -1.0)
                    top = np.argsort(-pc, kind="stable")[:kkr]
                    res += [len(set(top.tolist()) & H) / kkr,
                            float(np.mean([np.log(max(pi[s], 1e-12)) for s in H]))]
                sp_rows.append((sd, res[0], res[2], res[1], res[3]))
            # hidden items (revealed slot, item not yet known in the obs)
            for i in range(6):
                if il_m[r, i] < 0.5 or il_lab[r, i] <= 0:
                    continue
                if os_[r, i, K.POKEMON_ITEMS_OFFSET + 1] > 0.5:
                    continue
                spc = int(round(os_[r, i, K.POKEMON_SPECIES_OFFSET]))
                lab = int(il_lab[r, i])
                lp_prior = torch.log_softmax(torch.log(iprior[spc].clamp_min(1e-6)), -1).numpy()
                it_rows.append((sd, -il[r, i, lab], -lp_prior[lab], float(il[r, i].argmax() == lab),
                                float(lp_prior.argmax() == lab)))
    out = {"panel": panel, "n_rows": int(n), "n_games": int(len(np.unique(z["game"])))}

    def summarise(rows, names):
        res = {}
        arr = np.array(rows) if rows else np.zeros((0, len(names) + 1))
        for sd in (0, 1):
            s = arr[arr[:, 0] == sd] if len(arr) else arr
            res[f"side{sd}"] = {"n": int(len(s)), **{nm: round(float(s[:, j + 1].mean()), 4) if len(s) else None
                                                     for j, nm in enumerate(names)}}
        return res
    out["moves_revealed_mons"] = summarise(mv_rows, ["prec_post", "prec_prior", "bce_post", "bce_prior"])
    out["hidden_species"] = summarise(sp_rows, ["hit_post", "hit_prior", "ll_post", "ll_prior"])
    out["hidden_items"] = summarise(it_rows, ["ce_post", "ce_prior", "acc_post", "acc_prior"])
    # critic calibration (ties excluded)
    cal = {}
    for sd in (0, 1):
        m = (side == sd) & ~tie
        g = z["game"][m]
        cal[f"side{sd}"] = {"n_rows": int(m.sum()), "n_games": int(len(np.unique(g))),
                            "mean_v": round(float(win_p[m].mean()), 4), "mean_outcome_rowwt": round(float(won[m].mean()), 4),
                            "brier": round(float(((win_p[m] - won[m]) ** 2).mean()), 4)}
        # per-game: the first decision's value, and the game outcome
        firsts = [np.flatnonzero(m & (z["game"] == gg))[0] for gg in np.unique(g)]
        cal[f"side{sd}"]["first_v_mean"] = round(float(win_p[firsts].mean()), 4)
        cal[f"side{sd}"]["game_win_rate"] = round(float(won[firsts].mean()), 4)
    out["critic"] = cal
    # mean ablations of S on a deterministic subsample
    means = ablation_means(model, zh["obs"][::7])
    idx = np.linspace(0, n - 1, num=min(n_abl, n)).round().astype(int)
    pe = fe.pokemon_encoder
    orig = pe.encode
    abl = {}
    try:
        for name in ("species_id", "move_id", "item_id", "stats", "move_feats"):
            pe.encode = types.MethodType(patched_encode(name, means), pe)
            res = {0: [], 1: []}
            for a in range(0, len(idx), CHUNK):
                ii = idx[a:a + CHUNK]
                p, w, _ = C.forward(model, obs[ii], mask[ii])
                pb, wb = probs_all[ii], win_p[ii]
                for k, r in enumerate(ii):
                    mm = mask[r]
                    kl = float(np.sum(pb[k][mm] * (np.log(np.clip(pb[k][mm], 1e-9, 1)) - np.log(np.clip(p[k][mm], 1e-9, 1)))))
                    res[int(side[r])].append((kl, abs(w[k] - wb[k]), float(p[k].argmax() != pb[k].argmax())))
            abl[name] = {f"side{s}": {"n": len(v), "kl": round(float(np.mean([x[0] for x in v])), 4),
                                       "abs_dv": round(float(np.mean([x[1] for x in v])), 4),
                                       "flip": round(float(np.mean([x[2] for x in v])), 4)} for s, v in res.items()}
        # identity check: the patched encoder with NO ablation reproduces the base forward
        pe.encode = types.MethodType(patched_encode("none", means), pe)
        p, w, _ = C.forward(model, obs[idx[:CHUNK]], mask[idx[:CHUNK]])
        out["patched_identity_max_abs_dprob"] = float(np.abs(p - probs_all[idx[:CHUNK]]).max())
    finally:
        pe.encode = orig
    out["S_mean_ablation"] = abl
    json.dump(out, open(C.OUT / f"rows_{panel}{tag}_analysis.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 640)
