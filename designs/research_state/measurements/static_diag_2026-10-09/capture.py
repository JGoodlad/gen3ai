"""Step 2: one checkpoint on the whole Lane S bank (20,712 decisions), CPU forwards at the pin, with
read-only hooks on the trunk's two `BiasedEncoderLayer`s. DESCRIPTIVE / exploratory.

Writes ONE small npz per checkpoint to ~/gen3ai_archive/static_diag_2026-10-09/cap/<label>.npz:
  probs [N,11] f16, win_prob [N] f32,
  attn_ent  [N, L, H, Q]  f16  — attention entropy (nats) of query seat-group Q per layer / head,
                                 Q = (our active, their active, our bench mean, E3 mean, board mean)
  attn_mass [N, L, H, 2, G] f16 — for the our-active and their-active query: the attention mass on
                                 key groups G = (our mons, their mons, board, E3, E4/E5/other, events)
  probe tokens (float16) for 4 seats × 3 depths (trunk input, after layer 1, after layer 2):
     seats = (our active, their active, our first alive bench mon, E3 seat 0)  -> tok [N, 4, 3, 128]
  (≈ 20.7k × 12 × 128 × 2 B ≈ 64 MB per checkpoint; deleted after the probe step reads it — the disk
   is near full, see the README's FINDING.)
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ARCH = Path("/home/goodlad/gen3ai_archive/static_diag_2026-10-09")
POK = 122
OUR0, OPP0 = 0, 732
ACTIVE_OFF, HP_OFF = 121, 67


def seat_layout(fe, n_tokens):
    tt = fe.team_transformer
    es = getattr(fe, "entity_seats", None)
    base = int(tt._total_tokens)
    n_e3 = 4 if es is not None else 0
    k_e4 = int(getattr(es, "topk_seats", 0) or 0) if es is not None else 0
    n_tail = 6 if (es is not None and getattr(es, "tail_seats", False)) else 0
    has_other = getattr(fe, "hypothesis_builder", None) is not None
    n_ev = n_tokens - base - n_e3 - k_e4 - n_tail - (1 if has_other else 0)
    assert n_ev >= 0
    return base, n_e3, n_tokens - n_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    import torch

    from agents.model.extra_obs_keys import zero_extra_obs
    from main.policy_spectrum.reader import inference_globals, load_checkpoint

    d = np.load(ARCH / "bank_rows.npz")
    rows, masks = d["rows"], d["masks"]
    N = len(rows)
    our_act = rows[:, [OUR0 + i * POK + ACTIVE_OFF for i in range(6)]] > 0.5
    opp_act = rows[:, [OPP0 + i * POK + ACTIVE_OFF for i in range(6)]] > 0.5
    our_hp = rows[:, [OUR0 + i * POK + HP_OFF for i in range(6)]]
    oa = np.where(our_act.sum(1) == 1, our_act.argmax(1), -1)
    ta = np.where(opp_act.sum(1) == 1, opp_act.argmax(1), -1)
    bench_ok = (~our_act) & (our_hp > 0)
    ob = np.where(bench_ok.any(1), bench_ok.argmax(1), -1)

    out_dir = ARCH / "cap"
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()
    with inference_globals(a.threads):
        model = load_checkpoint(Path(a.ckpt))
        pol = model.policy
        fe = pol.features_extractor
        layers = [m for m in fe.modules() if type(m).__name__ == "BiasedEncoderLayer"]
        assert len(layers) == 2, len(layers)
        cur = {"x": [], "bias": [], "out": []}

        def pre(mod, args, kwargs):
            cur["x"].append(args[0])
            cur["bias"].append(args[1] if len(args) > 1 else kwargs.get("bias"))

        def post(mod, args, kwargs, out):
            cur["out"].append(out)

        hs = []
        for L in layers:
            hs.append(L.register_forward_pre_hook(pre, with_kwargs=True))
            hs.append(L.register_forward_hook(post, with_kwargs=True))
        probs = np.zeros((N, 11), np.float16)
        winp = np.zeros(N, np.float32)
        ent = np.zeros((N, 2, 4, 5), np.float16)
        mass = np.zeros((N, 2, 4, 2, 6), np.float16)
        tok = np.zeros((N, 4, 3, 128), np.float16)
        lay = None
        try:
            for s in range(0, N, a.batch):
                for v in cur.values():
                    v.clear()
                sl = slice(s, s + a.batch)
                mt = torch.as_tensor(masks[sl].astype(np.float32))
                obs = {"observation": torch.as_tensor(rows[sl]), "action_mask": mt}
                obs.update(zero_extra_obs(fe, batch=len(mt), device=torch.device("cpu")))
                with torch.no_grad():
                    dist = pol.get_distribution(obs)
                    lg = dist.distribution.logits
                    lg = torch.where(mt.bool(), lg, torch.full_like(lg, -1e8))
                    probs[sl] = torch.softmax(lg, 1).numpy().astype(np.float16)
                    winp[sl] = torch.sigmoid(fe.last_win_prob_logits.reshape(-1)).numpy()
                    B, n, dm = cur["x"][0].shape
                    if lay is None:
                        lay = seat_layout(fe, n)
                        base, n_e3, ev0 = lay
                        print(f"[cap] {a.label}: n_tokens {n} base {base} E3 {base}..{base + n_e3 - 1} "
                              f"events from {ev0}", flush=True)
                    groups = [(0, 6), (6, 12), (12, base), (base, base + n_e3), (base + n_e3, ev0), (ev0, n)]
                    idx = np.arange(s, min(s + a.batch, N))
                    bi = torch.arange(len(idx))
                    oa_b = torch.as_tensor(np.maximum(oa[idx], 0))
                    ta_b = torch.as_tensor(6 + np.maximum(ta[idx], 0))
                    ob_b = torch.as_tensor(np.maximum(ob[idx], 0))
                    alive_ours = torch.as_tensor(bench_ok[idx])                       # bench & alive
                    for li, L in enumerate(layers):
                        x, bias = cur["x"][li], cur["bias"][li]
                        H, hd = L.n_heads, L.head_dim
                        qkv = L.in_proj(x).reshape(B, n, 3, H, hd)
                        q, k = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2)
                        w = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(hd) + bias, dim=-1)  # B,H,n,n
                        e = -(w * torch.log(w.clamp_min(1e-12))).sum(-1)                     # B,H,n
                        e_oa = e[bi, :, oa_b]
                        e_ta = e[bi, :, ta_b]
                        bm = alive_ours.float()[:, None, :]
                        e_bench = (e[:, :, 0:6] * bm).sum(-1) / bm.sum(-1).clamp_min(1)
                        e_e3 = e[:, :, base:base + n_e3].mean(-1)
                        e_board = e[:, :, 12:base].mean(-1)
                        ent[idx, li] = torch.stack([e_oa, e_ta, e_bench, e_e3, e_board], -1).numpy()
                        for qi, qs in enumerate((oa_b, ta_b)):
                            wq = w[bi, :, qs]                                                  # B,H,n
                            mass[idx, li, :, qi] = torch.stack(
                                [wq[..., g0:g1].sum(-1) for g0, g1 in groups], -1).numpy()
                    depths = [cur["x"][0], cur["out"][0], cur["out"][1]]
                    for di, T in enumerate(depths):
                        tok[idx, 0, di] = T[bi, oa_b].numpy()
                        tok[idx, 1, di] = T[bi, ta_b].numpy()
                        tok[idx, 2, di] = T[bi, ob_b].numpy()
                        tok[idx, 3, di] = T[:, base].numpy()
        finally:
            for h in hs:
                h.remove()
    np.savez(out_dir / f"{a.label}.npz", probs=probs, win_prob=winp, attn_ent=ent, attn_mass=mass, tok=tok,
             our_active=oa, their_active=ta, our_bench=ob, layout=np.array(lay))
    print(f"[cap] {a.label}: {N} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
