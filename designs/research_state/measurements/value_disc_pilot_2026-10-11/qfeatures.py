"""PHASE 3, step 3 (Amendment 7): the FROZEN checkpoint's root features for the Q head, one CPU forward per state at
the pin (eager, the strict loader). For every state of a bank (``bank`` = the registered evaluation states, ``qbank`` =
the training states):

- ``logits`` [11] (the raw policy logits, as Phase 1's root forward), ``v`` (the critic's P(win)), ``mask`` [11];
- ``pooled`` [D] (``value_pooled``: the critic's whole-board summary, the context the ride-along A head reads);
- the pointer head's own per-action inputs (``PointerInputs``): ``mtok`` [4,D] / ``mcell`` [4,Cm] (moves, request
  order), ``team`` [6,D] / ``scell`` [6,Cs] (switches), ``mvalid`` [4];
- ``hid`` [11,H]: the pointer head's per-action hidden layer (tanh(proj(entity ⊕ cells) + ctx_proj(latent_pi)),
  the layer its zero-init scorers read), recomputed from those inputs and CHECKED: the head's scorers applied to it
  reproduce ``logits`` up to one constant per row (``logits`` are the distribution's masked, normalised
  log-probabilities, as in Phase 1) within 1e-4, else the script refuses;
- ``adv`` [11]: the ride-along A head's member mean (π-centred; ``RideAlongHeads.readout``): the network's OWN
  per-action advantage readout, trained on-policy (a no-rollout comparator).

    run_at_pin.sh end2 2 12 1h qfeatures.py --bank bank|qbank   → archive ``<cp>/qfeat_<bank>.npz``
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

BATCH = 128


def pointer_hidden(ph, lat_pi, p):
    import torch

    c = ph.ctx_proj(lat_pi)
    m = torch.tanh(ph.move_proj(torch.cat([p.move_tokens, p.move_cells], dim=-1)) + c[:, None, :])
    sc = p.switch_cells
    if ph.switch_extra_proj is None:
        s = torch.tanh(ph.switch_proj(torch.cat([p.team_tokens, sc], dim=-1)) + c[:, None, :])
    else:
        base = ph._switch_base_dim
        s = torch.tanh(ph.switch_proj(torch.cat([p.team_tokens, sc[..., :base]], dim=-1))
                       + ph.switch_extra_proj(sc[..., base:]) + c[:, None, :])
    g = torch.tanh(c)
    logits = torch.cat([ph.switch_score(s).squeeze(-1), ph.move_score(m).squeeze(-1) * p.move_valid,
                        ph.struggle_score(g)], dim=-1)
    hid = torch.cat([s, m, g[:, None, :]], dim=1)                       # [B,11,H] in the action-space order
    return hid, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", choices=("bank", "qbank"), required=True)
    a = ap.parse_args()
    c = C.cp()
    C.check_pin(c)
    import torch

    from agents.model.extra_obs_keys import zero_extra_obs
    from agents.model.ridealong_heads import RideAlongBatch
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.replay import replay
    from main.policy_spectrum.spectrum import masked_probs

    od = C.out_dir(c)
    bank = load_bank(od / a.bank)
    decs = sorted(bank.decisions, key=lambda x: x["id"])
    bids = sorted({d["battle"] for d in decs})
    reps = dict(zip(bids, replay([bank.battles[bank.battle_index[b]].recorded() for b in bids], workers=2)))
    dd = [reps[d["battle"]].decisions[0 if d["side"] == "p1" else 1][d["n"]] for d in decs]
    rows = np.stack([np.array(x.row, dtype=np.float32) for x in dd])
    masks = np.stack([np.asarray(x.mask, dtype=np.uint8) for x in dd])
    cols = {k: [] for k in ("logits", "v", "pooled", "mtok", "mcell", "team", "scell", "mvalid", "hid", "adv")}
    worst = 0.0
    with inference_globals(C.threads()):
        model = C.load_model(c)
        pol = model.policy
        fe = pol.features_extractor
        heads = getattr(pol, "ridealong", None)
        for i in range(0, len(rows), BATCH):
            mk = masks[i:i + BATCH]
            ob = {"observation": torch.from_numpy(np.ascontiguousarray(rows[i:i + BATCH])),
                  "action_mask": torch.tensor(mk.astype(np.float32))}
            ob.update(zero_extra_obs(fe, batch=len(mk), device="cpu"))
            with torch.no_grad():
                pi_f, vf_f = pol.extract_features(ob)
                lat_pi = pol.mlp_extractor.forward_actor(pi_f)
                v = pol._critic_value(vf_f).reshape(-1)
                logits = pol._get_action_dist_from_latent(lat_pi).distribution.logits
                p = fe.last_pointer_inputs
                hid, lg2 = pointer_hidden(pol.pointer_head, lat_pi, p)
                legal = torch.from_numpy(mk.astype(bool))
                # the distribution's logits are the masked, NORMALISED log-probabilities: equal to the head's raw
                # scores up to one constant per row over the legal actions
                d = torch.where(legal, lg2 - logits, torch.zeros_like(lg2))
                d = d - d.sum(1, keepdim=True) / legal.sum(1, keepdim=True)
                worst = max(worst, float(torch.where(legal, d, torch.zeros_like(d)).abs().max()))
                adv = torch.zeros_like(logits)
                if heads is not None and getattr(heads, "adv", None) is not None:
                    pi = torch.from_numpy(masked_probs(logits.numpy(), mk).astype(np.float32))
                    b = RideAlongBatch.detached(obs=ob["observation"], pooled=fe.last_value_pooled, pointer=tuple(p),
                                                pi=pi, logits=logits, legal=legal, values=v)
                    adv = heads.readout(b)["adv_mean"]
            cols["logits"].append(logits.numpy())
            cols["v"].append(v.numpy())
            cols["pooled"].append(fe.last_value_pooled.numpy())
            cols["mtok"].append(p.move_tokens.numpy())
            cols["mcell"].append(p.move_cells.numpy())
            cols["team"].append(p.team_tokens.numpy())
            cols["scell"].append(p.switch_cells.numpy())
            cols["mvalid"].append(p.move_valid.numpy())
            cols["hid"].append(hid.numpy())
            cols["adv"].append(adv.numpy())
    if worst > 1e-4:
        raise SystemExit(f"the recomputed pointer hidden does not reproduce the logits (max |Δ| {worst:.2e})")
    out = {k: np.concatenate(v, 0).astype(np.float32) for k, v in cols.items()}
    np.savez_compressed(od / f"qfeat_{a.bank}.npz", ids=np.array([d["id"] for d in decs]),
                        battle=np.array([d["battle"] for d in decs]), mask=masks, **out)
    print(f"[qfeatures] {c} {a.bank}: {len(decs)} states, pointer-hidden check max |Δ| {worst:.2e}, "
          + ", ".join(f"{k} {tuple(v.shape[1:])}" for k, v in out.items()), flush=True)


if __name__ == "__main__":
    main()
