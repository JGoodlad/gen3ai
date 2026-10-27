"""S4: identity vs mechanics mean-ablation of OUR six mons' S inputs (P* forward, CPU), on every row probe.py used.

IDENTITY  = the species, move and item id embeddings of our six mons -> a constant mean;
MECHANICS = our six mons' 13 stat columns, the move mechanics columns and MOVE_ATTR -> a constant mean.
The opponent's tokens and the hypothesis rows are untouched (only an encode over the 12 board mons is patched, on
mons 0-5). The encoder is the review's verbatim copy of `StaticTokenEncoder.encode` (`analyze_rows.patched_encode`),
here with an our-side mask; its no-ablation identity against the live forward is checked (VOID if > 1e-5).
Writes s4.npz (per row: group, panel, game, kl and flip of each arm) and s4_meta.json."""
from __future__ import annotations

import json
import types

import numpy as np
import torch

import n2common as N
from agents.observation import constants as K

C = N.C
PANELS = ("home72", "F", "H", "P72")
SLOT = K.POKEMON_FULL_DIM
CHUNK = 128
ARMS = {"identity": ("species_emb", "move_emb", "item_emb"), "mechanics": ("stats", "move_rem", "move_attr")}


def our_encode(ablate: tuple, means: dict, orig_encode):
    """The review's patched encode, generalised: every name in ``ablate`` replaced by its mean on OUR mons only."""
    from agents.model.arch_constants import MOVE_NET_HIDDEN
    from agents.model.static_tokens import StaticParts
    from agents.observation.moves import HIDDEN_POWER_MOVE_NUM

    def encode(self, pp, ids, embeddings, move_legal, active_ctx):
        B, Nn = pp.shape[0], pp.shape[1]
        if Nn != 12 or not ablate:
            return orig_encode(pp, ids, embeddings, move_legal, active_ctx)
        ours = torch.zeros(Nn, dtype=torch.bool)
        ours[:6] = True
        o2 = ours.view(1, Nn, 1)
        o3 = ours.view(1, Nn, 1, 1)
        nm = self.num_moves
        rem, known, cur_pp = self._move_columns(pp)
        realmv = (ids["all_move_ids"] > 0).unsqueeze(-1) & o3
        mv_emb = embeddings.move_embedding(ids["all_move_ids"])
        if "move_emb" in ablate:
            mv_emb = torch.where(realmv, means["move_emb"].expand_as(mv_emb), mv_emb)
        mv_type = embeddings.type_embedding(ids["all_move_type_ids"])
        hp_probs = ids["hp_probs"]
        is_hp = (ids["all_move_ids"] == HIDDEN_POWER_MOVE_NUM).unsqueeze(-1)
        soft = embeddings.hp_soft_type(hp_probs).unsqueeze(2).expand(-1, -1, nm, -1)
        mv_type = torch.where(is_hp, soft, mv_type)
        hp_per_slot = torch.where(is_hp, hp_probs.unsqueeze(2).expand(-1, -1, nm, -1),
                                  torch.zeros((), dtype=hp_probs.dtype))
        if "move_rem" in ablate:
            rem = torch.where(realmv, means["move_rem"].expand_as(rem), rem)
        blocks = [mv_emb, mv_type, rem, known, hp_per_slot]
        if self.move_latent:
            le = self.move_latent_encoder
            attr = le.MOVE_ATTR[ids["all_move_ids"]]
            if "move_attr" in ablate:
                attr = torch.where(realmv, means["move_attr"].expand_as(attr), attr)
            blocks.append(le.mlp(torch.cat([mv_emb, mv_type, attr], dim=-1)))
        mv = self.move_network(torch.cat(blocks, dim=-1))
        mv_in = mv.reshape(B * Nn, nm, MOVE_NET_HIDDEN[1])
        mv_delta, _ = self.move_self_attn(mv_in, mv_in, mv_in)
        m_static = self.move_self_norm(mv_in + mv_delta).reshape(B, Nn, nm, MOVE_NET_HIDDEN[1])
        move_set = m_static.sum(dim=2)
        sp_emb = embeddings.species_embedding(ids["species_ids"])
        realmon = (ids["species_ids"] > 0).unsqueeze(-1) & o2
        if "species_emb" in ablate:
            sp_emb = torch.where(realmon, means["species_emb"].expand_as(sp_emb), sp_emb)
        stats = self._actual_stats(pp, ids["species_ids"])
        if "stats" in ablate:
            stats = torch.where(realmon, means["stats"].expand_as(stats), stats)
        it_emb = embeddings.item_embedding(ids["item_ids"])
        if "item_emb" in ablate:
            it_emb = torch.where((ids["item_ids"] > 0).unsqueeze(-1) & o2, means["item_emb"].expand_as(it_emb), it_emb)
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
        return StaticParts(tokens=static + dynamic, static=static, dynamic=dynamic, move_tokens=move_tokens)
    return encode


def our_means(model, obs):
    """Constant means over OUR real mons of ``obs`` (six slots per row)."""
    from agents.model.extractor_ctx import slice_pokemon_categoricals
    fe = model.policy.features_extractor
    pe, emb = fe.pokemon_encoder, fe.embeddings
    pp = torch.tensor(obs[:, :6 * SLOT].reshape(1, -1, SLOT), dtype=torch.float32)
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


def main() -> None:
    from probe import kl
    un = np.load(N.OUT / "units.npz")
    model = C.load_model()
    pe = model.policy.features_extractor.pokemon_encoder
    # rebuild probe.py's row list (same order): the side-0 rows with >= 1 target, in panel order
    all_side0 = []
    import probe as PR
    dz = np.load(N.OUT / "draws.npz")
    vec_keys = set(dz["keys"].tolist())
    orig = {p: dz[f"orig_{p}"] for p in PANELS}
    sp_of = json.load(open(N.OUT / "draws_meta.json"))["species_of_slot"]
    obs_l, mask_l, meta_l = [], [], []
    for gi, p in enumerate(PANELS):
        z = N.load_rows(p)
        rows = np.flatnonzero(z["side"] == 0)
        all_side0.append(z["obs"][rows])
        for r in rows:
            o = z["obs"][r]
            slots, elig = PR.eligible_slots(o)
            i = int(z["p1team"][r])
            ok = False
            for s in elig:
                b = sp_of[p][i].index(int(round(slots[s, K.POKEMON_SPECIES_OFFSET])))
                if any(f"draw|{p}|{i}|{b}|{d}" in vec_keys for d in range(3)) and \
                        np.array_equal(orig[p][i, b, PR.SET_COLS], slots[s, PR.SET_COLS]):
                    ok = True
                    break
            if ok:
                obs_l.append(o)
                mask_l.append(z["mask"][r])
                meta_l.append((0 if p == "home72" else 1, gi, int(z["game"][r])))
    assert len(obs_l) == int(un["row"].max()) + 1, (len(obs_l), int(un["row"].max()) + 1)
    means = our_means(model, np.concatenate(all_side0))
    obs, mask = np.stack(obs_l), np.stack(mask_l)

    def run():
        P = []
        for a in range(0, len(obs), CHUNK):
            P.append(C.forward(model, obs[a:a + CHUNK], mask[a:a + CHUNK])[0])
        return np.concatenate(P)
    base = run()
    orig_encode = pe.encode
    out = {}
    try:
        pe.encode = types.MethodType(our_encode(("__none__",), means, orig_encode), pe)
        # identity check: the copied encoder with nothing ablated reproduces the live forward
        p0 = np.concatenate([C.forward(model, obs[a:a + CHUNK], mask[a:a + CHUNK])[0] for a in range(0, 256, CHUNK)])
        ident = float(np.abs(p0 - base[:256]).max())
        for arm, names in ARMS.items():
            pe.encode = types.MethodType(our_encode(names, means, orig_encode), pe)
            pa = run()
            out[f"{arm}_kl"] = np.array([kl(base[k], pa[k], mask[k]) for k in range(len(obs))])
            out[f"{arm}_flip"] = (pa.argmax(1) != base.argmax(1)).astype(np.int8)
    finally:
        pe.encode = orig_encode
    m = np.array(meta_l)
    np.savez_compressed(N.OUT / "s4.npz", group=m[:, 0], panel=m[:, 1], game=m[:, 2], **out)
    json.dump({"patched_identity_max_abs_dprob": ident, "n_rows": len(obs)}, open(N.OUT / "s4_meta.json", "w"), indent=1)
    print({"identity_check": ident, "n_rows": len(obs),
           **{f"{k}_g{g}": round(float(v[m[:, 0] == g].mean()), 4) for k, v in out.items() for g in (0, 1)}})


if __name__ == "__main__":
    main()
