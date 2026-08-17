"""Split the `in_matrix` (⑥) dependence into HEADERS vs CELLS — and finer.

Extends `tmp/incoming_conditional_probe.py` (same collect / threat gate / metrics / shuffle
control, so numbers are directly comparable) with ONE change: the intervention takes an INDEX
SET rather than a contiguous (lo, hi) slice, which is what lets us cut *within* the per-move
header and *by cell channel*.

Why: ⑥'s 522 dims decompose as 306 headers (6 moves x 51) + 216 cells (6 mons x 6 moves x 6),
and its 24.2%/9.6M zero-arm zeroes BOTH together. Three questions that split answers:

  1. HEADERS vs CELLS — 210 of the 306 header dims are already E4 seat content
     (`threat_seat_proj` input == the header's first 35 dims, same `refine_candidates`
     selection). If headers-only reads ~0, those dims are deletable duplication.
  2. Which header fields matter — latent(32) / [w,acc,is_phys](3) are in E4; effect(6) and
     secondary(10) have no entity home at all.
  3. THE ORPHAN TEST — cell channels `low`(0) and `crit`(2) are carried by NO entity route
     (not E4, not d3 `[high,pko,eff,is_phys,w]`, not s3). The three cell arms are each
     exactly 72 dims, so the zero-perturbation size is width-MATCHED across them — an
     apples-to-apples comparison the block-level arms never had.

Cell layout (from `decode_damage_block`): cell(i=our mon, k=move) at
`cells_base + (i*K + k)*_DMG_IMX_CELL`, channels [low, high, crit, pko, type_mult, status_lands].
"""
import argparse
import json
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import torch

if TYPE_CHECKING:
    import numpy as np

from agents.model import damage_op as D
from agents.observation.constants import TEAM_SIZE
from agents.action.constants import SWITCH_START, SWITCH_END, MOVE_START, MOVE_END


def collect(patterns: Sequence[str], max_states: int, seed: int = 0,
            ) -> Tuple["np.ndarray", "np.ndarray", Dict]:
    # gen3_audit_state_sampler_v1: STRATIFIED, deterministic (shared `audit_states` sampler).
    # The old sorted-glob + break-at-cap drew every state from ONE lexically-first step dir —
    # the §2.5.1 defect this probe's own caveats documented against itself.
    from agents.model.audit_states import collect_states

    return collect_states(patterns, max_states, seed=seed)


def arms(op: Any) -> Dict[str, torch.Tensor]:
    """→ {name: LongTensor of column indices into the op block}. Anchors first (for
    reproduction against the committed 40M numbers), then the new decomposition."""
    assert op.matrices_incoming, "probe requires --damage-matrices incoming"
    K = op.matrices_incoming_k
    H, C = D._DMG_IMX_HEADER, D._DMG_IMX_CELL

    # locate the incoming matrix inside the block, exactly as block_slices() does
    cur = op.incoming_dim
    if op.outgoing:
        cur += D._DMG_OUTGOING + D._DMG_STATUS
    if op.matrices_outgoing:
        cur += D._DMG_OMX
    imx_lo = cur
    hdr_lo = imx_lo
    cells_lo = hdr_lo + K * H
    imx_hi = cells_lo + TEAM_SIZE * K * C
    assert imx_hi - imx_lo == D._dmg_imx_dim(K)

    def rng(lo: int, hi: int) -> torch.Tensor:
        return torch.arange(lo, hi, dtype=torch.long)

    def hdr_field(off: int, width: int) -> torch.Tensor:
        """`width` columns at `off` within EVERY move's 51-dim header."""
        return torch.tensor([hdr_lo + k * H + off + j
                             for k in range(K) for j in range(width)], dtype=torch.long)

    def cell_chans(chans: Sequence[int]) -> torch.Tensor:
        """`chans` channels of EVERY (our mon, move) cell."""
        return torch.tensor([cells_lo + (i * K + k) * C + c
                             for i in range(TEAM_SIZE) for k in range(K) for c in chans],
                            dtype=torch.long)

    a: Dict[str, torch.Tensor] = {}
    # --- anchors (must reproduce the committed 40M run) ---
    a["FULL_CONCAT"] = rng(0, op.out_dim)
    a["in_matrix"] = rng(imx_lo, imx_hi)
    # --- Q1: the top-level split ---
    a["imx_HEADERS"] = rng(hdr_lo, cells_lo)
    a["imx_CELLS"] = rng(cells_lo, imx_hi)
    # --- Q2: inside the header (latent + w/acc/phys are ALREADY E4 seat content) ---
    a["hdr_latent~E4"] = hdr_field(0, D.MOVE_LATENT_DIM)
    a["hdr_w_acc_phys~E4"] = hdr_field(D._DMG_IMX_HDR_W, 3)
    a["hdr_effect"] = hdr_field(D._DMG_IMX_HDR_EFFECT, D._DMG_EFFECT)
    a["hdr_secondary"] = hdr_field(D._DMG_IMX_HDR_SEC, D._N_SECONDARY)
    # --- Q3: cell channels, WIDTH-MATCHED at 72 dims each ---
    a["cell_LOW_CRIT(orphan)"] = cell_chans((D._DMG_IMX_IDX_LOW, D._DMG_IMX_IDX_CRIT))
    a["cell_high_pko~d3"] = cell_chans((D._DMG_IMX_IDX_HIGH, D._DMG_IMX_IDX_PKO))
    a["cell_mult_status~d3s3"] = cell_chans((D._DMG_IMX_IDX_MULT, D._DMG_IMX_IDX_STATUS))
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--site", choices=("assembler", "op"), default="assembler",
                    help="WHERE to intervene. 'assembler' (legacy default): the assembler's "
                         "damage_block argument — under gen3_no_concat_v1 (v61+) this reaches "
                         "ONLY the critic's seed-readout rows, so use it to isolate the vf "
                         "route. 'op': the DamageOperator's returned block, before any consumer "
                         "binds it — covers pointer cells (pi) + seed rows (vf) + the stash; "
                         "the honest post-concat site for design_pair_reduction.md §8.1 step 0.")
    ap.add_argument("--max-states", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pko", type=float, default=0.5)
    ap.add_argument("--safe", type=float, default=0.35)
    ap.add_argument("--out", default=None)
    # `build_arch_viewer._load_overlays` keys the viewer's dependence overlay off a `provenance`
    # block and MERGES overlays by (generation, step) — omit these and the entry renders as
    # "None@0M" and merges with nothing.
    ap.add_argument("--generation", default=None, help="e.g. gen-3 (for the viewer overlay)")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (for the overlay)")
    ap.add_argument("--note", default=None, help="provenance note surfaced in the viewer")
    a = ap.parse_args()

    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings
    model, _ = load_foreign_opponent(a.checkpoint,
                                     current_version=current_model_version(load_mappings()),
                                     device="cpu")
    policy = model.policy
    fe = policy.features_extractor
    op = fe.damage_op
    assert op is not None, "checkpoint has no damage op"
    obs_np, masks_np, coverage = collect(a.states, a.max_states)
    N = len(obs_np)
    dev = next(policy.parameters()).device

    stash: Dict[str, Any] = {}

    def _unpack_hook(_m: Any, _inp: Any, out: Any) -> None:
        stash["ctx"] = out
    h = fe.unpack.register_forward_hook(_unpack_hook)

    def forward_all(cols: Optional[torch.Tensor] = None,
                    mode: str = "zero") -> Tuple[torch.Tensor, ...]:
        """cols = LongTensor of op-block columns to intervene on at the assembler, or None.

        'zero'    — set those columns to 0 (the deletion counterfactual). OFF-MANIFOLD, so it
                    measures reliance PLUS the shock of an impossible input.
        'shuffle' — permute them ACROSS states (same width, same marginals, state-specificity
                    destroyed). This is group PERMUTATION IMPORTANCE and it is THE DEPENDENCE
                    ESTIMATE — quote this column, not `zero`, and not `zero - shuffle`.

        `zero - shuffle` is printed below only as a diagnostic: it is the size of the
        zero-ablation artifact, NOT a width-fair measure of reliance. (A 2026-08-08 draft of
        `designs/ai_v9/design_op_tensors.md` quoted it as one; see that doc's §2.5.5 retraction.)
        """
        def _perturb(db: torch.Tensor) -> torch.Tensor:
            db = db.clone()
            if mode == "zero":
                db[:, cols] = 0.0
            else:
                n = db.shape[0]
                perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
                db[:, cols] = db[perm][:, cols]
            return db

        hook = None
        if cols is not None and a.site == "assembler":
            # LEGACY SITE (pre-v61 semantics): patch what the assembler consumes of the block.
            # Under gen3_no_concat_v1 that is only the per-mon rows (the seed readout), so this
            # site measures the CRITIC route alone and reads structural zeros for every arm
            # outside dims [0, TEAM_SIZE*per_mon) — the gen-5 step-0 run that motivated --site.
            # gen3_op_tensors_views_v1: the assembler now receives the TYPED rows view, so the
            # perturbation runs in FLAT col coordinates on a clone of the block and the rows
            # view of THAT clone is substituted — same semantics, same col-space.
            def _pre(_module: Any, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
                _rows: Any = (fe.damage_op.last_tensors.incoming_rows
                              if fe.damage_op.last_tensors is not None else None)
                if args and args[-1] is not None and args[-1] is _rows:
                    flat = _perturb(fe.last_damage_block.clone())
                    rows = flat[:, :TEAM_SIZE * op.per_mon].reshape(
                        flat.shape[0], TEAM_SIZE, op.per_mon)
                    return args[:-1] + (rows,)
                return args
            hook = fe.assembler.register_forward_pre_hook(_pre)
        elif cols is not None:
            # --site op: patch the op's RETURN VALUE, before ANY consumer binds it — the local
            # `damage_block` the extractor stashes (last_damage_block), slices for the pointer
            # cells, and hands the assembler are all this one tensor. Covers every BLOCK-mediated
            # route (pi pointer cells + vf seed rows). NOT covered: routes fed from the op's
            # INTERNAL tensors before the block is assembled (prefuse token injection, E4 seat
            # content, d3/s3 edge biases) — those are separately measured by edge_ablation_audit.
            def _post(_module: Any, _inp: Any, out: torch.Tensor) -> torch.Tensor:
                return _perturb(out)
            hook = op.register_forward_hook(_post)
        ps, vs, raws, acts, hps = [], [], [], [], []
        try:
            with torch.no_grad():
                for i in range(0, N, a.batch):
                    mk = torch.as_tensor(masks_np[i:i + a.batch], device=dev)
                    ob = {"observation": torch.as_tensor(obs_np[i:i + a.batch], device=dev),
                          "action_mask": mk.float()}
                    dist = policy.get_distribution(ob)
                    # SB3 types `.distribution` as the multi-discrete union; the maskable
                    # CATEGORICAL path this policy takes always yields the single one with logits.
                    lg = dist.distribution.logits.clone()  # type: ignore[union-attr]
                    lg[~mk] = -1e9
                    ps.append(torch.softmax(lg, -1))
                    vs.append(policy.predict_values(ob).squeeze(-1))
                    if cols is None:
                        raws.append(op.last_raw_block.clone())
                        ctx = stash["ctx"]
                        acts.append(ctx.hp_and_active[:, :, -1].clone())
                        hps.append(ctx.hp_and_active[:, :, 0].clone())
        finally:
            if hook is not None:
                hook.remove()
        if cols is None:
            return (torch.cat(ps), torch.cat(vs), torch.cat(raws),
                    torch.cat(acts), torch.cat(hps))
        return torch.cat(ps), torch.cat(vs)

    base_p, base_v, raw, active, hp = forward_all(None)
    h.remove()

    # --- the THREAT gate, identical to incoming_conditional_probe ---
    pm = op.per_mon
    idx = torch.arange(N)
    our_act = active[:, :TEAM_SIZE].argmax(1)
    has_act = active[:, :TEAM_SIZE].max(1).values > 0.5
    permon = raw[:, :TEAM_SIZE * pm].view(N, TEAM_SIZE, pm)
    pko = torch.maximum(permon[:, :, D._DMG_IDX_PHYS_LOW + 3],
                        permon[:, :, D._DMG_IDX_SPEC_LOW + 3])
    outspeed = permon[:, :, D._DMG_IDX_OUTSPEED]
    act_pko = pko[idx, our_act]
    act_outspeed = outspeed[idx, our_act]
    alive = hp[:, :TEAM_SIZE] > 0
    bench = alive.clone()
    bench[idx, our_act] = False
    big = torch.full_like(pko, 9.9)
    bench_pko = torch.where(bench, pko, big).min(1).values
    can_switch = torch.as_tensor(masks_np[:, SWITCH_START:SWITCH_END]).any(1)
    can_move = torch.as_tensor(masks_np[:, MOVE_START:MOVE_END]).any(1)
    threat = (has_act & (act_outspeed < 0.5) & (act_pko >= a.pko)
              & can_switch & can_move & (bench_pko <= a.safe))
    masks_t = torch.as_tensor(masks_np, device=dev)

    def metrics(p: torch.Tensor, v: torch.Tensor, sel: torch.Tensor) -> Dict[str, float]:
        eps = 1e-9
        kl = ((base_p * ((base_p + eps).log() - (p + eps).log())) * masks_t.float()).sum(-1)
        flip = (base_p.argmax(-1) != p.argmax(-1)).float()
        dv = (base_v - v).abs()
        return {"n": int(sel.sum()), "kl_mean": float(kl[sel].mean()),
                "flip_rate": float(flip[sel].mean()), "dv_mean": float(dv[sel].mean())}

    allsel = torch.ones(N, dtype=torch.bool)
    rows: Dict[str, Dict[str, Any]] = {}
    for name, cols in arms(op).items():
        p, v = forward_all(cols, "zero")
        ps, vs = forward_all(cols, "shuffle")
        rows[name] = {"width": int(cols.numel()),
                      "all": metrics(p, v, allsel), "threat": metrics(p, v, threat),
                      "shuf_all": metrics(ps, vs, allsel),
                      "shuf_threat": metrics(ps, vs, threat)}

    print(f"\nstates: {N}   THREAT states: {int(threat.sum())} "
          f"({100 * threat.float().mean():.1f}%)")
    print("  SHUF = permutation importance = the dependence. 'artifact' = zero - shuf "
          "(the off-manifold cost of zeroing), NOT a reliance measure.")
    hdr = (f"{'arm':>24} {'w':>5} | {'SHUF flip':>9} {'ZERO flip':>9} {'artifact':>9} "
           f"| {'kl shuf':>8} {'kl zero':>8} | {'|dV| shuf':>9} {'|dV| zero':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in rows.items():
        al, sa = r["all"], r["shuf_all"]
        print(f"{name:>24} {r['width']:>5} | {100 * sa['flip_rate']:>8.2f}% "
              f"{100 * al['flip_rate']:>8.2f}% "
              f"{100 * (al['flip_rate'] - sa['flip_rate']):>8.2f}% "
              f"| {sa['kl_mean']:>8.5f} {al['kl_mean']:>8.5f} "
              f"| {sa['dv_mean']:>9.3f} {al['dv_mean']:>9.3f}")
    if a.out:
        json.dump({"provenance": {"generation": a.generation, "step": a.step, "n_states": N,
                                  "date": None,
                                  "producer": "src/agents/model/op_block_split_audit.py",
                                  "note": a.note,
                                  "sampling": coverage},
                   "meta": {"checkpoint": a.checkpoint, "n_states": N,
                            "n_threat": int(threat.sum()), "pko": a.pko, "safe": a.safe,
                            "probe": "op_block_split_audit.py", "site": a.site},
                   "blocks": rows}, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
