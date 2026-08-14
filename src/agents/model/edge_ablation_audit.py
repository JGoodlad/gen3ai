"""The per-family edge-bias ABLATION AUDIT — Stage 2's verdict instrument
(`designs/ai_v9/design_generation_roadmap.md` §3 Stage 2: "per-family bias ablation" decides
which edge families stay and gates the op head-concat deletion).

For a trained checkpoint, each enabled edge family's map is temporarily ZEROED (zero bias ==
that family absent — the identity-at-init property in reverse) and the policy distribution is
re-measured over a set of REAL decision states:

  * ``kl_mean`` / ``kl_p95`` — masked KL(baseline ‖ ablated) over legal actions
  * ``flip_rate``            — argmax action changes
  * ``dv_mean``              — mean |ΔV| on the critic

A family whose numbers sit at ~0 after training is decorative (its map stayed ≈0 or attention
ignores it); a large one is load-bearing. The ``all`` row ablates every family at once.

Two additional OP-CONCAT arms answer the question the family rows can't — the roadmap's
head-concat DELETION counterfactual: ``concat`` zeroes the op's damage block at the
`ProjectionAssembler` ONLY (edges, the prefuse token injection and the pointer head's op cells
all stay — exactly what deleting the concat would leave), and ``concat_cells`` additionally
zeroes the pointer cells (the op fully absent from the heads — the modern P1 ceiling). A small
``concat`` number on a TRAINED edge checkpoint is the redundancy evidence deletion needs.

States come from eval-trace ``states.npz`` files (the arrays the run's eval recorder writes and
the prober reads — pass one or more paths/globs). There is deliberately NO random-obs mode for
real audits: random vectors are not on-distribution states and would understate every number.

Usage:
  python -m agents.model.edge_ablation_audit <checkpoint.zip> --states 'models/run_x/eval_traces/**/states.npz' [--max-states 4096] [--out report.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys

import numpy as np
import torch


def _collect_states(patterns, max_states: int, seed: int = 0):
    # gen3_audit_state_sampler_v1: STRATIFIED, deterministic sampling (the shared
    # `audit_states.collect_states` — two-level round-robin over step dirs × opponents +
    # a seeded row subsample). The old sorted-glob + break-at-cap drew every state from ONE
    # lexically-first step dir (step_10000032 < step_2000016) while labelling itself a pool
    # average — design_op_tensors.md §2.5.1. Returns (obs, masks, coverage); callers write
    # `coverage` into the report so a reader can VERIFY the spread instead of trusting it.
    from agents.model.audit_states import collect_states

    obs, masks, coverage = collect_states(patterns, max_states, seed=seed)
    if not masks.any(axis=1).all():
        raise ValueError("a state decoded to ZERO legal actions — the logits→mask recovery is wrong "
                         "for this trace format; inspect the npz")
    return obs, masks, coverage


@torch.no_grad()
def _measure(policy, obs_t: dict, masks: torch.Tensor):
    """→ (probs [N, A] masked-renormalized, values [N])."""
    dist = policy.get_distribution(obs_t)
    logits = dist.distribution.logits.clone()
    logits[~masks] = -1e9
    probs = torch.softmax(logits, dim=-1)
    values = policy.predict_values(obs_t).squeeze(-1)
    return probs, values


def _masked_kl(p: torch.Tensor, q: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    eps = 1e-9
    kl = (p * ((p + eps).log() - (q + eps).log()))
    return (kl * masks.float()).sum(-1)


@torch.no_grad()
def audit(policy, obs_np: np.ndarray, masks_np: np.ndarray, batch: int = 512) -> dict:
    fe = policy.features_extractor
    eb = getattr(fe, "edge_bias", None)
    if eb is None:
        raise ValueError("checkpoint has edge_bias_families='off' — nothing to audit")
    fams = sorted(eb.families)
    device = next(policy.parameters()).device
    maps = {f: getattr(eb, f"{f}_map") for f in fams}

    def _forward_all():
        ps, vs = [], []
        for i in range(0, len(obs_np), batch):
            mk = torch.as_tensor(masks_np[i:i + batch], device=device)
            # The eval/opponent obs contract (RLPlayer → get_distribution): observation + action_mask.
            ob = {"observation": torch.as_tensor(obs_np[i:i + batch], device=device),
                  "action_mask": mk.float()}
            p, v = _measure(policy, ob, mk)
            ps.append(p); vs.append(v)
        return torch.cat(ps), torch.cat(vs)

    base_p, base_v = _forward_all()
    masks_t = torch.as_tensor(masks_np, device=device)
    report = {}

    def _ablate(names):
        saved = {n: (maps[n].weight.detach().clone(), maps[n].bias.detach().clone()) for n in names}
        for n in names:
            maps[n].weight.zero_(); maps[n].bias.zero_()
        p, v = _forward_all()
        for n in names:
            maps[n].weight.copy_(saved[n][0]); maps[n].bias.copy_(saved[n][1])
            assert torch.equal(maps[n].weight, saved[n][0]), "restore must be bitwise"  # noqa: S101
        kl = _masked_kl(base_p, p, masks_t)
        return {
            "kl_mean": float(kl.mean()),
            "kl_p95": float(kl.quantile(0.95)),
            "flip_rate": float((base_p.argmax(-1) != p.argmax(-1)).float().mean()),
            "dv_mean": float((base_v - v).abs().mean()),
        }

    for f in fams:
        report[f] = _ablate([f])
    report["all"] = _ablate(fams)

    # --- the op-concat arms (the DELETION counterfactual the family rows can't measure) ---
    # The roadmap's op-concat deletion question is about REDUNDANCY: with the edges + prefuse
    # injection + pointer cells all still live, how much does the policy still lean on the raw
    # 807-dim block at the projection concat? `concat` zeroes damage_block at the
    # ProjectionAssembler ONLY (exactly what deleting the concat would remove); `concat_cells`
    # ALSO zeroes the pointer head's op cells (the op fully absent from the heads — the modern
    # P1 ceiling). Neither touches the edge biases or the prefuse token injection.
    if getattr(fe, "damage_op", None) is not None and fe.last_damage_block is not None:
        pa = fe.assembler

        def _zero_db_hook(_module, args):
            # gen3_op_tensors_views_v1: the assembler receives the TYPED incoming-rows view
            # (`OpTensors.incoming_rows`), not the flat block — match it by identity against
            # the op's own stash so a signature drift here fails the identity check loudly
            # (the arm asserts it fired, below) instead of silently measuring nothing.
            _rows = (fe.damage_op.last_tensors.incoming_rows
                     if fe.damage_op.last_tensors is not None else None)
            if args and args[-1] is not None and args[-1] is _rows:
                _zero_db_hook.fired = True
                return args[:-1] + (torch.zeros_like(args[-1]),)
            return args
        _zero_db_hook.fired = False

        def _measure_with(zero_cells: bool):
            h = pa.register_forward_pre_hook(_zero_db_hook)
            orig_cells = fe.damage_op.pointer_cells
            if zero_cells:
                fe.damage_op.pointer_cells = lambda db: tuple(
                    torch.zeros_like(t) for t in orig_cells(db))
            try:
                _zero_db_hook.fired = False
                p, v = _forward_all()
                if not _zero_db_hook.fired:
                    raise RuntimeError(
                        "the concat-arm assembler hook never matched its argument — the "
                        "assembler's seed-rows identity drifted and the arm measured nothing.")
            finally:
                h.remove()
                fe.damage_op.pointer_cells = orig_cells
            kl = _masked_kl(base_p, p, masks_t)
            return {
                "kl_mean": float(kl.mean()),
                "kl_p95": float(kl.quantile(0.95)),
                "flip_rate": float((base_p.argmax(-1) != p.argmax(-1)).float().mean()),
                "dv_mean": float((base_v - v).abs().mean()),
            }

        report["concat"] = _measure_with(zero_cells=False)
        report["concat_cells"] = _measure_with(zero_cells=True)
        # The hook is identity when removed — re-measuring base must reproduce it bitwise.
        chk_p, chk_v = _forward_all()
        assert torch.equal(chk_p, base_p) and torch.equal(chk_v, base_v), \
            "op-arm restore failed — the hook/patch leaked into the baseline"  # noqa: S101
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--states", nargs="+", required=True,
                    help="states.npz path(s)/glob(s) — the eval-trace format (obs + action_mask)")
    ap.add_argument("--max-states", type=int, default=4096)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings
    # Inference-only load (the churn_probe pattern): the opponent loader's arch-FAMILY check is
    # exactly the right strictness for auditing any in-generation checkpoint from newer code.
    model, _ver = load_foreign_opponent(args.checkpoint,
                                        current_version=current_model_version(load_mappings()),
                                        device="cpu")
    obs_np, masks_np, coverage = _collect_states(args.states, args.max_states)
    report = audit(model.policy, obs_np, masks_np)
    meta = {"checkpoint": args.checkpoint, "n_states": int(len(obs_np)),
            "sampling": coverage}   # per-step/per-opponent sampled counts — verify, don't trust
    out = {"meta": meta, "families": report}
    hdr = f"{'family':>8} {'kl_mean':>10} {'kl_p95':>10} {'flip%':>7} {'|dV|':>8}"
    print(hdr)
    for f, r in report.items():
        print(f"{f:>8} {r['kl_mean']:>10.5f} {r['kl_p95']:>10.5f} "
              f"{100 * r['flip_rate']:>6.2f}% {r['dv_mean']:>8.4f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
