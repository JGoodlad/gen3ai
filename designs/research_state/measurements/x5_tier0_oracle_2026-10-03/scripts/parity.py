"""Parity of the vectorised kernels (kernels.py) against the PRODUCTION op functions, on real banked states.

  P0  T0 species prior: proxies.Beliefs 'prior' == the extractor's T0SpeciesPrior renormalised over the valid set.
  P1  R0 exactly: `_outgoing_matrix(ctx, species_probs=T0)` hidden columns == outgoing(averaged_defender(T0)).
  P2  one-hot hidden defenders: `_outgoing_matrix` with a [B,6,S] one-hot override == the all-species table.
  P3  a tail-distribution override (renormalised T0 minus its top 3) == outgoing(averaged_defender(tail)).
  P4  species types: SPECIES_TYPE[revealed species] == the obs (type1, type2) as a set, on every revealed slot.
  P5  hypothesis attackers: `pairwise_bench_incoming` with a hidden slot FAKE-REVEALED as species s (species id,
      types, HP 1, believed False, its composed cold-start move row) == incoming(attacker_tables)[s].
  P6  REAL revealed bench attackers: `pairwise_bench_incoming` on the unmodified state with the production
      cold-start move posterior (MoveBelief.move_logits + _typed_hp_posterior on zero tokens: every head delta is
      zero-init) == incoming() fed the same row.
Writes out/parity.json; exits non-zero if any check exceeds its tolerance.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

import kernels as KN
from common import OUT, THREADS, build_extractor, load_rows, unpack
from proxies import Beliefs
from agents.observation.constants import TEAM_SIZE

TOL = 2e-5          # absolute, on damage fractions / probabilities (fp32 reassociation only)


def main():
    torch.set_num_threads(THREADS)
    torch.manual_seed(0)
    bank, rows, masks, gate = load_rows()
    fx, op = build_extractor()
    K = fx.consequence_topk
    gen = np.random.default_rng(0)
    pick = np.sort(gen.choice(len(rows), 512, replace=False))
    res = {}
    with torch.no_grad():
        ctx = unpack(fx, rows[pick], masks[pick])
        B = ctx.batch_size
        hid = ctx.opp_believed_mask
        ids = ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE]
        t0 = fx.t0_species_prior(ids, hid)
        S = t0.shape[1]
        bel = Beliefs()
        lb, valid = bel.logbeliefs(ids, hid, pad_to=S)
        t0v = torch.where(valid, t0.double(), torch.zeros_like(t0.double()))
        t0v = t0v / t0v.sum(-1, keepdim=True)
        res["P0_t0_vs_prior_maxabs"] = float((t0v - lb["prior"].exp()).abs().max())

        att = KN.our_attacker(op, ctx)
        real = op._outgoing_matrix(ctx, None, species_probs=t0)[:, :4 * 6 * 5].reshape(B, 4, 6, 5)
        mine_h, mine_ko = KN.outgoing(op, att, KN.averaged_defender(op, t0))
        hcols = hid[:, None, :].expand(B, 4, 6)
        res["P1_R0_high_maxabs"] = float((real[..., 1] - mine_h.expand(B, 4, 6))[hcols].abs().max())
        res["P1_R0_ko_real_is_zero"] = float(real[..., 3][hcols].abs().max())
        res["P1_n_hidden_cells"] = int(hcols.sum())

        dt = KN.defender_tables(op)
        all_h, all_ko = KN.outgoing(op, att, dt)                                  # [B,4,S]
        sp = torch.zeros(B, 6, S)
        choice = torch.zeros(B, 6, dtype=torch.long)
        for b in range(B):
            cand = torch.nonzero(valid[b] & (op.BASE_STATS[:, 0] > 0)).flatten()
            for j in range(6):
                s = int(cand[gen.integers(len(cand))]); choice[b, j] = s; sp[b, j, s] = 1.0
        real1 = op._outgoing_matrix(ctx, None, species_probs=sp)[:, :4 * 6 * 5].reshape(B, 4, 6, 5)
        mine1 = torch.gather(all_h, 2, choice[:, None, :].expand(B, 4, 6))
        res["P2_onehot_high_maxabs"] = float((real1[..., 1] - mine1)[hcols].abs().max())

        top3 = t0v.topk(3, -1).indices
        tail = t0v.clone().scatter_(1, top3, 0.0); tail = (tail / tail.sum(-1, keepdim=True)).float()
        real2 = op._outgoing_matrix(ctx, None, species_probs=tail)[:, :4 * 6 * 5].reshape(B, 4, 6, 5)
        mine2, _ = KN.outgoing(op, att, KN.averaged_defender(op, tail))
        res["P3_tail_high_maxabs"] = float((real2[..., 1] - mine2.expand(B, 4, 6))[hcols].abs().max())

        rev = ~hid
        st = op.SPECIES_TYPE[ids]
        obs_t = torch.stack([ctx.type1_ids[:, TEAM_SIZE:], ctx.type2_ids[:, TEAM_SIZE:]], -1)
        same = (torch.sort(st, -1).values == torch.sort(obs_t, -1).values).all(-1)
        res["P4_types_mismatch_revealed"] = int((~same & rev).sum())
        res["P4_n_revealed"] = int(rev.sum())

        tabs = KN.attacker_tables(op, fx, K)
        dfn = KN.our_defender(op, ctx)
        _w, _p, cells_all = KN.incoming(op, dfn, tabs)                           # [B,S,4]
        ctx2 = fake_reveal(ctx, op, choice, hid)
        sml = KN.species_move_logits(fx)
        mbl = sml[ctx2.species_ids[:, TEAM_SIZE:]]                               # [B,6,M]
        real5 = op.pairwise_bench_incoming(ctx2, mbl, k_bench=K)                 # [B,6i,6j,4]
        ar = torch.arange(B)
        r5 = real5[ar, ctx.our_active_idx]                                       # [B,6j,4]
        m5 = cells_all[ar[:, None], choice]                                      # [B,6j,4]
        nact = torch.ones(B, 6, dtype=torch.bool); nact[ar, ctx.opp_active_local] = False
        sel = hid & nact
        res["P5_hyp_attacker_maxabs"] = float((r5 - m5)[sel].abs().max())
        res["P5_n_cells"] = int(sel.sum())
        res["P5_frac_nonzero"] = float((m5[sel].abs().sum(-1) > 0).float().mean())

        zeros = torch.zeros(B, 6, fx.move_belief.move_head.in_features)
        raw = fx.move_belief.move_logits(zeros, ids, ctx.all_move_ids[:, TEAM_SIZE:, :],
                                         hidden_species_probs=t0, opp_believed_mask=hid)
        typed, _pres, _hp, _hl = fx._typed_hp_posterior(zeros, ctx, raw)
        real6 = op.pairwise_bench_incoming(ctx, typed, k_bench=K)[ar, ctx.our_active_idx]   # [B,6j,4]
        alive = ctx.hp_and_active[:, TEAM_SIZE:, 0] > 0
        sel6 = rev & alive & nact
        diffs = []
        for j in range(6):
            w_all = torch.sigmoid(typed[:, j]) * op.HP_CAND_MASK[None, :]
            idx = w_all.topk(K, -1).indices
            sj = ids[:, j]
            mty = op.MOVE_TYPE_IDX[idx]
            stab = 1.0 + 0.5 * ((mty == ctx.type1_ids[:, TEAM_SIZE + j:TEAM_SIZE + j + 1])
                                | (mty == ctx.type2_ids[:, TEAM_SIZE + j:TEAM_SIZE + j + 1])).float()
            a = dict(w=w_all.gather(-1, idx), bp=op.MOVE_BP[idx], mty=mty, phys=op.MOVE_PHYS[idx],
                     acc=op.MOVE_ACCURACY[idx], stab=stab, atk=tabs["atk"][sj], spa=tabs["spa"][sj],
                     _per_decision=True)
            _w6, _p6, c6 = KN.incoming(op, dfn, a)
            diffs.append((real6[:, j] - c6[:, 0]).abs().max(-1).values)
        d6 = torch.stack(diffs, 1)
        res["P6_revealed_attacker_maxabs"] = float(d6[sel6].max()) if sel6.any() else None
        res["P6_n_cells"] = int(sel6.sum())
        res["P6_frac_nonzero"] = float((real6[sel6].abs().sum(-1) > 0).float().mean()) if sel6.any() else None
    res["tolerance_abs"] = TOL
    res["sample"] = {"n": int(len(pick)), "seed": 0, "rows_gate": gate["obs_as_recorded"]}
    bad = [k for k, v in res.items() if k.endswith("maxabs") and v is not None and v > TOL] + \
        ([] if res["P4_types_mismatch_revealed"] == 0 else ["P4"])
    res["verdict"] = "PASS" if not bad else f"FAIL {bad}"
    (OUT / "parity.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res, indent=1, sort_keys=True))
    sys.exit(0 if not bad else 1)


def fake_reveal(ctx, op, choice, hid):
    """A copy of ctx with every HIDDEN opp slot j turned into a REVEALED, full-HP mon of species choice[:, j]."""
    import dataclasses
    sid = ctx.species_ids.clone(); t1 = ctx.type1_ids.clone(); t2 = ctx.type2_ids.clone()
    hpa = ctx.hp_and_active.clone()
    opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
    sid[:, opp] = torch.where(hid, choice, sid[:, opp])
    st = op.SPECIES_TYPE[choice]
    t1[:, opp] = torch.where(hid, st[..., 0], t1[:, opp]); t2[:, opp] = torch.where(hid, st[..., 1], t2[:, opp])
    hpa[:, opp, 0] = torch.where(hid, torch.ones_like(hpa[:, opp, 0]), hpa[:, opp, 0])
    return dataclasses.replace(ctx, species_ids=sid, type1_ids=t1, type2_ids=t2, hp_and_active=hpa,
                               opp_believed_mask=torch.zeros_like(hid))


if __name__ == "__main__":
    main()
