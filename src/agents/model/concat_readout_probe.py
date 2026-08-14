"""The §9.1 DISCRIMINATING ARMS (design_op_tensors.md) — is the concat's residual value
READOUT bandwidth, or delivery the entity routes lack?

Three pre-registered arm families on one stratified state sample:

1. **Per-head concat zeroing** — zero the op's flat block in the PI projection input only,
   the VF input only, and both. Sizes the two halves of the old two-route precondition
   directly (the block enters `ProjectionAssembler.forward` once for each head, appended
   last — the probe composes each arm from two assembler calls, one with the block zeroed,
   taking pi from one and vf from the other; exact, no internal replication).
2. **Redundancy arm** — zero the E4 threat-seat tokens alone, the concat's `imx_HEADERS`
   slice alone, and both together. The headers are bit-for-bit the E4 seat input content, so:
   flat-alone ≈ both  ⇒ the entity route goes unused (readout hypothesis CONFIRMED);
   both ≫ each alone  ⇒ the routes are complementary (delivery still matters).
3. **Conditional coverage** — the coverage-probe targets (labels EXACT from the op's own
   in_matrix cells) refit on pi/vf features WITH the concat zeroed at the projections:
   does the critic's one measured magnitude read (`act_threat` vf r² ≈ 0.33 at baseline)
   collapse without the concat?

Usage:
  python -m agents.model.concat_readout_probe <ckpt.zip> --states '<glob>' \
      [--max-states 4000] [--out report.json] [--generation gen-4]
"""
from __future__ import annotations

import argparse
import contextlib
import json

import numpy as np
import torch

from agents.model.audit_states import collect_states
from agents.observation.constants import TEAM_SIZE

# The §2.1 offset table (code-derived; re-derived here from the live constants at runtime).
from agents.model import damage_op as D


def _imx_header_slice(op):
    """(lo, hi) of the incoming-matrix HEADERS inside the flat op block."""
    cur = op.incoming_dim
    if op.outgoing:
        cur += D._DMG_OUTGOING + D._DMG_STATUS
    if op.matrices_outgoing:
        cur += D._DMG_OMX
    assert op.matrices_incoming, "probe requires the incoming matrix"
    return cur, cur + op.matrices_incoming_k * D._DMG_IMX_HEADER


@contextlib.contextmanager
def _assembler_arm(fe, mode: str, hdr: "tuple[int, int] | None" = None):
    """mode ∈ {'off_pi', 'off_vf', 'off_both', 'headers_off', 'none'} — patch the assembler."""
    if mode == "none":
        yield
        return
    orig = fe.assembler.forward

    def patched(our_p, their_p, our_act, val_p, ctx, hidden_opp_belief=None, seed_rows=None):
        # gen3_op_tensors_views_v1: the assembler consumes the TYPED incoming-rows view. The
        # imx headers therefore never reach this site at all — `headers_off` is structurally a
        # no-op here (it already was post-gen3_no_concat_v1, when the headers fell outside the
        # rows), kept only so historical report shapes still render.
        if seed_rows is None or mode == "headers_off":
            return orig(our_p, their_p, our_act, val_p, ctx,
                        hidden_opp_belief=hidden_opp_belief, seed_rows=seed_rows)
        z = torch.zeros_like(seed_rows)
        pi_z, vf_z = orig(our_p, their_p, our_act, val_p, ctx,
                          hidden_opp_belief=hidden_opp_belief, seed_rows=z)
        if mode == "off_both":
            return pi_z, vf_z
        pi_r, vf_r = orig(our_p, their_p, our_act, val_p, ctx,
                          hidden_opp_belief=hidden_opp_belief, seed_rows=seed_rows)
        return (pi_z, vf_r) if mode == "off_pi" else (pi_r, vf_z)

    fe.assembler.forward = patched
    try:
        yield
    finally:
        fe.assembler.forward = orig


@contextlib.contextmanager
def _e4_seats_off(fe):
    """Zero the E4 threat-seat tokens in the trunk's `extra` pack (E3 [0:4], E4 [4:4+K],
    E5 tail after — layout per EntityMoveSeats)."""
    tt = fe.team_transformer
    orig = tt.forward
    k = fe.entity_seats.topk_seats

    def patched(*args, extra=None, **kw):
        if extra is not None and k > 0:
            tokens, types, pad = extra
            tokens = tokens.clone()
            tokens[:, 4:4 + k] = 0.0
            extra = (tokens, types, pad)
        return orig(*args, extra=extra, **kw)

    tt.forward = patched
    try:
        yield
    finally:
        tt.forward = orig


@torch.no_grad()
def _measure(policy, obs_np, masks_np, batch):
    ps, vs = [], []
    for i in range(0, len(obs_np), batch):
        mk = torch.as_tensor(masks_np[i:i + batch])
        ob = {"observation": torch.as_tensor(obs_np[i:i + batch]), "action_mask": mk.float()}
        dist = policy.get_distribution(ob)
        lg = dist.distribution.logits.clone()
        lg[~mk] = -1e9
        ps.append(torch.softmax(lg, -1))
        vs.append(policy.predict_values(ob).squeeze(-1))
    return torch.cat(ps), torch.cat(vs)


def _vs_base(p, v, bp, bv):
    flips = (p.argmax(1) != bp.argmax(1)).float().mean().item()
    kl = (bp.clamp_min(1e-9) * (bp.clamp_min(1e-9).log() - p.clamp_min(1e-9).log())).sum(1)
    return {"flip_rate": round(flips, 5), "kl_mean": round(kl.mean().item(), 6),
            "dv_mean": round((v - bv).abs().mean().item(), 4)}


@torch.no_grad()
def _features(policy, obs_np, masks_np, batch):
    pis, vfs = [], []
    for i in range(0, len(obs_np), batch):
        ob = {"observation": torch.as_tensor(obs_np[i:i + batch]),
              "action_mask": torch.as_tensor(masks_np[i:i + batch]).float()}
        pi, vf = policy.extract_features(ob)
        pis.append(pi.clone())
        vfs.append(vf.clone())
    return torch.cat(pis).numpy(), torch.cat(vfs).numpy()


def _coverage_fit(policy, fe, obs_np, masks_np, batch):
    """The coverage-probe targets (labels exact from the op's in_matrix) on pi/vf features."""
    from main.prober.engine import fit_probe
    op, K = fe.damage_op, fe.damage_op.matrices_incoming_k
    hdr_lo, hdr_hi = _imx_header_slice(op)
    cells_dim = TEAM_SIZE * K * D._DMG_IMX_CELL

    stash = {}
    h = fe.unpack.register_forward_hook(lambda m, i, o: stash.__setitem__("ctx", o))
    imxs, acts, hps = [], [], []
    with torch.no_grad():
        for i in range(0, len(obs_np), batch):
            ob = {"observation": torch.as_tensor(obs_np[i:i + batch]),
                  "action_mask": torch.as_tensor(masks_np[i:i + batch]).float()}
            policy.extract_features(ob)
            blk = op.last_raw_block
            imxs.append(blk[:, hdr_lo:hdr_hi + cells_dim].clone())
            ctx = stash["ctx"]
            acts.append(ctx.hp_and_active[:, :, -1].clone())
            hps.append(ctx.hp_and_active[:, :, 0].clone())
    h.remove()
    imx, active, hp = torch.cat(imxs), torch.cat(acts), torch.cat(hps)
    N = len(imx)
    hdr = imx[:, :K * D._DMG_IMX_HEADER].view(N, K, D._DMG_IMX_HEADER)
    cells = imx[:, K * D._DMG_IMX_HEADER:].view(N, TEAM_SIZE, K, D._DMG_IMX_CELL)
    w = hdr[:, :, D._DMG_IMX_HDR_W]
    wpko = cells[:, :, :, D._DMG_IMX_IDX_PKO] * w.unsqueeze(1)
    threat = wpko.max(dim=2).values
    alive = hp[:, :TEAM_SIZE] > 0
    our_act = active[:, :TEAM_SIZE].argmax(1)
    idx = torch.arange(N)
    keep = (active[:, :TEAM_SIZE].max(1).values > 0.5).numpy()
    labels = {
        "act_threat": ("regression", threat[idx, our_act].numpy()),
        "n_threatened": ("regression", ((threat >= 0.5) & alive).sum(1).float().numpy()),
    }
    out = {}
    for arm in ("baseline", "concat_off"):
        cm = _assembler_arm(fe, "off_both") if arm == "concat_off" else contextlib.nullcontext()
        with cm:
            pi_f, vf_f = _features(policy, obs_np, masks_np, batch)
        out[arm] = {}
        for name, (task, y) in labels.items():
            yk = y[keep]
            out[arm][name] = {
                "pi": fit_probe(pi_f[keep], yk, task)["overall"],
                "vf": fit_probe(vf_f[keep], yk, task)["overall"],
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--max-states", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=None)
    ap.add_argument("--generation", default=None)
    ap.add_argument("--note", default=None)
    a = ap.parse_args()

    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings
    model, _ = load_foreign_opponent(
        a.checkpoint, current_version=current_model_version(load_mappings()), device="cpu")
    policy, fe = model.policy, model.policy.features_extractor
    assert fe.damage_op is not None and fe.entity_seats is not None
    hdr = _imx_header_slice(fe.damage_op)

    obs_np, masks_np, coverage = collect_states(a.states, a.max_states)
    bp, bv = _measure(policy, obs_np, masks_np, a.batch)

    arms = {}
    for name, mode in (("concat_off_pi", "off_pi"), ("concat_off_vf", "off_vf"),
                       ("concat_off_both", "off_both"), ("imx_headers_off", "headers_off")):
        with _assembler_arm(fe, mode, hdr):
            p, v = _measure(policy, obs_np, masks_np, a.batch)
        arms[name] = _vs_base(p, v, bp, bv)
        print(f"{name:>22}: {arms[name]}")
    with _e4_seats_off(fe):
        p, v = _measure(policy, obs_np, masks_np, a.batch)
    arms["e4_seats_off"] = _vs_base(p, v, bp, bv)
    print(f"{'e4_seats_off':>22}: {arms['e4_seats_off']}")
    with _e4_seats_off(fe), _assembler_arm(fe, "headers_off", hdr):
        p, v = _measure(policy, obs_np, masks_np, a.batch)
    arms["e4_and_headers_off"] = _vs_base(p, v, bp, bv)
    print(f"{'e4_and_headers_off':>22}: {arms['e4_and_headers_off']}")

    cov = _coverage_fit(policy, fe, obs_np, masks_np, a.batch)
    print("coverage:", json.dumps(cov, default=str)[:400])

    if a.out:
        json.dump({
            "provenance": {"generation": a.generation, "checkpoint": a.checkpoint,
                           "n_states": int(len(obs_np)), "date": "2026-08-09",
                           "producer": "src/agents/model/concat_readout_probe.py",
                           "note": a.note, "sampling": coverage},
            "arms": arms, "conditional_coverage": cov,
        }, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
