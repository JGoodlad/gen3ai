"""MIXING probe: the belief posterior shrunk toward its own Smogon prior, p_mix = (1 − λ)·posterior + λ·prior, scored
on FAMILIAR opponents (side 0 of every panel; side 1 of home72) and UNFAMILIAR ones (side 1 of F / H / P72). λ = 0 is
P*'s belief, λ = 1 the prior alone. A mechanism check for an anchoring lever, offline only (the policy is not re-run
on the mixed belief). Writes mix_probe.json."""
from __future__ import annotations

import json

import numpy as np
import torch

import common as C
from agents.observation import constants as K
from analyze_rows import CHUNK, HP_FAMILY, MOVE0, capture_hypothesis, opp_slots, prior_presence

LAMBDAS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)


def main():
    model = C.load_model()
    fe = model.policy.features_extractor
    box = capture_hypothesis(fe)
    mprior = fe.move_belief.move_prior_logits
    iprior = fe.item_belief_head.item_prior
    legal_floor = float(torch.logit(torch.tensor(2e-6)))
    acc = {g: {"species_ll": {l: [] for l in LAMBDAS}, "species_hit": {l: [] for l in LAMBDAS},
               "moves_prec": {l: [] for l in LAMBDAS}, "moves_bce": {l: [] for l in LAMBDAS},
               "item_ce": {l: [] for l in LAMBDAS}, "item_acc": {l: [] for l in LAMBDAS}}
           for g in ("familiar", "unfamiliar")}
    for panel in ("home72", "F", "H", "P72"):
        z = dict(np.load(C.OUT / f"rows_{panel}.npz"))
        obs, mask, side = z["obs"], z["mask"], z["side"]
        for a in range(0, len(obs), CHUNK):
            sl = slice(a, a + CHUNK)
            C.forward(model, obs[sl], mask[sl])
            ml = torch.sigmoid(fe.last_move_belief_logits).numpy()
            ip = torch.softmax(fe.last_item_logits, -1).numpy()
            pi_post = fe.last_hypothesis.species.pi.numpy()
            pi_prior, cand, _ = prior_presence(fe, box)
            pi_prior, cand = pi_prior.numpy(), cand.numpy()
            os_ = opp_slots(obs[sl])
            km, bs, il_lab, il_m = z["known_moves"][sl], z["belief_species"][sl], z["item_label"][sl], z["item_mask"][sl]
            for r in range(len(os_)):
                g = "unfamiliar" if (panel != "home72" and side[a + r] == 1) else "familiar"
                A = acc[g]
                H = {int(s) for s in bs[r] if s >= 0}
                if H:
                    for l in LAMBDAS:
                        pi = (1 - l) * pi_post[r] + l * pi_prior[r]
                        pc = np.where(cand[r], pi, -1.0)
                        top = np.argsort(-pc, kind="stable")[:len(H)]
                        A["species_hit"][l].append(len(set(top.tolist()) & H) / len(H))
                        A["species_ll"][l].append(float(np.mean([np.log(max(pi[s], 1e-12)) for s in H])))
                for i in range(6):
                    if km[r, i, 0] < 0:
                        continue
                    spc = int(round(os_[r, i, K.POKEMON_SPECIES_OFFSET]))
                    truth = {int(m) for m in km[r, i] if m >= 0} - HP_FAMILY
                    rev = {int(round(os_[r, i, MOVE0 + 11 * j])) for j in range(4) if os_[r, i, MOVE0 + 11 * j + 6] > 0.5}
                    U = truth - rev
                    if not U or spc <= 0:
                        continue
                    prl = mprior[spc].numpy()
                    candm = np.array([m for m in np.flatnonzero(prl > legal_floor) if m not in rev and m not in HP_FAMILY])
                    if not set(U) <= set(candm.tolist()):
                        continue
                    y = np.isin(candm, list(U)).astype(float)
                    pp, pr = ml[r, i, candm], 1 / (1 + np.exp(-prl[candm]))
                    for l in LAMBDAS:
                        pz = np.clip((1 - l) * pp + l * pr, 1e-7, 1 - 1e-7)
                        top = candm[np.argsort(-pz, kind="stable")[:len(U)]]
                        A["moves_prec"][l].append(len(set(top.tolist()) & U) / len(U))
                        A["moves_bce"][l].append(float(-(y * np.log(pz) + (1 - y) * np.log(1 - pz)).sum() / len(U)))
                for i in range(6):
                    if il_m[r, i] < 0.5 or il_lab[r, i] <= 0 or os_[r, i, K.POKEMON_ITEMS_OFFSET + 1] > 0.5:
                        continue
                    spc = int(round(os_[r, i, K.POKEMON_SPECIES_OFFSET]))
                    lab = int(il_lab[r, i])
                    pri = torch.softmax(torch.log(iprior[spc].clamp_min(1e-6)), -1).numpy()
                    for l in LAMBDAS:
                        pm = (1 - l) * ip[r, i] + l * pri
                        A["item_ce"][l].append(-float(np.log(max(pm[lab], 1e-12))))
                        A["item_acc"][l].append(float(pm.argmax() == lab))
    out = {g: {m: {str(l): round(float(np.mean(v)), 4) for l, v in d.items()} for m, d in A.items()}
           for g, A in acc.items()}
    out["n"] = {g: {m: len(next(iter(d.values()))) for m, d in A.items()} for g, A in acc.items()}
    json.dump(out, open(C.OUT / "mix_probe.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
