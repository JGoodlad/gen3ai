"""CPU evidence for the s6302 T2 parity FATAL (README): the failing weights' legal log-probs under one-ulp weight noise,
the tied move pair on fixture rows 54/55, and every row's selection gap over ALL opponent mons vs the roster's own
(addressable-masked) gap that `near_tie_rows` reads. Run at the pin (PYTHONPATH=<worktree at 39eaa326>/src), card hidden:

    CUDA_VISIBLE_DEVICES="" <python> cpu_tie_probe.py <run_dir> <ckpt.zip> [<ckpt.zip> ...]
"""
from __future__ import annotations

import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_gate import N_FIXTURE, recording_gaps  # noqa: E402

TRIALS = 6
ROW, MON, PAIR = 54, 5, (202, 280)


def probe(ckpt: str, run_dir: str) -> None:
    import agents.model.extractor_forward as ef
    import agents.model.hypothesis_tokens as ht
    from agents.inference.service.decision import policy_reference
    from agents.inference.service.parity import fixture_rows
    from agents.model.hypothesis_set import near_tie_rows
    from main.capacity import load_policy

    model, _, _ = load_policy(ckpt, run_dir, "cpu")
    pol = model.policy.eval()
    sp = pol.observation_space
    D = sp["observation"].shape[0] if hasattr(sp, "spaces") else sp.shape[0]
    o, m = fixture_rows(D, N_FIXTURE)
    obs, mask = torch.as_tensor(o), torch.as_tensor(m)
    cap = {}
    orig = ht.build_op_roster

    def rec(hb, ctx, hctx, hs, fm, move_logits, cuts):
        ro, hs2 = orig(hb, ctx, hctx, hs, fm, move_logits, cuts)
        cap.update(ro=ro, hs=hs2, ml=move_logits.detach().clone(), addr=ctx.opp_addressable.clone(), cuts=cuts)
        return ro, hs2

    ef.build_op_roster = rec
    try:
        with recording_gaps(N_FIXTURE) as gap_all:
            lp0, _ = policy_reference(pol, obs, mask)
    finally:
        ef.build_op_roster = orig
    legal = lambda lp: torch.where(mask, lp, torch.zeros_like(lp))  # noqa: E731
    ro, ml = cap["ro"], cap["ml"]
    a, b = PAIR
    print(f"[{os.path.basename(ckpt)}] cuts {cap['cuts']}")
    print(f"  row {ROW} mon {MON} (addressable {bool(cap['addr'][ROW, MON])}): logits {a}/{b} "
          f"{float(ml[ROW, MON, a]):.7f} / {float(ml[ROW, MON, b]):.7f} (equal {bool(ml[ROW, MON, a] == ml[ROW, MON, b])}); "
          f"ranks {int(ro.move_rank[ROW, MON, a])} / {int(ro.move_rank[ROW, MON, b])}")
    nt = near_tie_rows(cap["hs"], eps=2e-4)
    print(f"  row {ROW}: gap over ALL mons {float(gap_all[ROW]):.2e}; near_tie_rows(eps 2e-4) flags it: {bool(nt[ROW])}")
    print(f"  rows with gap(all mons) <= 2e-4: {[int(r) for r in torch.nonzero(gap_all <= 2e-4).flatten()]}; "
          f"near_tie_rows(2e-4): {[int(r) for r in torch.nonzero(nt).flatten()]}")
    gen = torch.Generator().manual_seed(0)
    for t in range(TRIALS):
        p2 = copy.deepcopy(pol)
        with torch.no_grad():
            for p in p2.parameters():
                p.mul_(1 + (2.0 ** -23) * torch.randn(p.shape, generator=gen))
        lpk, _ = policy_reference(p2, obs, mask)
        e = (legal(lpk) - legal(lp0)).abs()
        r, act = divmod(int(e.argmax()), e.shape[1])
        print(f"  one-ulp weight noise trial {t}: max|dlogp| {float(e.max()):.2e} at row {r} action {act} "
              f"(eager logp {float(lp0[r, act]):.3f}); rows > 1e-3: {[int(x) for x in torch.nonzero(e.max(1).values > 1e-3).flatten()]}")
        del p2


if __name__ == "__main__":
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision("highest")
    for c in sys.argv[2:]:
        probe(c, sys.argv[1])
