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

One additional OP arm answers the question the family rows can't: ``concat_cells`` zeroes the
pointer head's op CELLS, i.e. the op fully absent from the heads — the modern P1 ceiling. On
gen-14 it reads KL 0.5682 / flips 0.3105, the single largest policy dependence in this report,
which is why it is a live tripwire rather than scaffolding.

**Its twin ``concat`` is DELETED, and the reason is worth keeping.** That arm was built for the
v61 op head-concat deletion counterfactual, and it worked by zeroing the assembler's LAST
positional argument. The concat died at v61; from v76 that argument was ``seed_rows``, so for
three generations ``concat`` silently measured the MULTI-SEED CRITIC READOUT under the name of a
block that no longer existed — and duly reported 0.0000 on every axis (gen-14, n=12,391),
identical to the dedicated ``seed`` arm in ``critic_route_audit``. The critic-route deletion wave
deleted the seed readout, so the arm now has no subject at all. **An instrument that outlives its
subject does not go quiet; it re-points at whatever is left at that offset and keeps printing
numbers.** Same lesson as the allowlist entry that outlived its own fix.

States come from eval-trace ``states.npz`` files (the arrays the run's eval recorder writes and
the prober reads — pass one or more paths/globs). There is deliberately NO random-obs mode for
real audits: random vectors are not on-distribution states and would understate every number.

Usage:
  python -m agents.model.edge_ablation_audit <checkpoint.zip> --states 'models/run_x/eval_traces/**/states.npz' [--max-states 4096] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

import numpy as np
import torch


def _collect_states(patterns: Sequence[str], max_states: int, seed: int = 0,
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
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
def _measure(policy: Any, obs_t: dict, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
def audit(policy: Any, obs_np: np.ndarray, masks_np: np.ndarray, batch: int = 512) -> dict:
    fe = policy.features_extractor
    eb = getattr(fe, "edge_bias", None)
    if eb is None:
        raise ValueError("checkpoint has edge_bias_families='off' — nothing to audit")
    fams = sorted(eb.families)
    device = next(policy.parameters()).device
    maps = {f: getattr(eb, f"{f}_map") for f in fams}

    def _forward_all() -> tuple[torch.Tensor, torch.Tensor]:
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
    report: dict[str, dict[str, float]] = {}

    def _ablate(names: Sequence[str]) -> dict[str, float]:
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

    # --- the op arm the family rows can't measure -----------------------------------------
    # `concat_cells` zeroes the pointer head's op CELLS: the op fully absent from the heads (the
    # modern P1 ceiling). It does NOT touch the edge biases or the prefuse token injection, so
    # what it isolates is exactly the per-action absolute channel. Its `concat` twin — which
    # zeroed the assembler's trailing positional argument — is deleted; see the module docstring
    # for why an arm that outlived its subject is worse than no arm.
    if getattr(fe, "damage_op", None) is not None and fe.last_damage_block is not None:
        orig_cells = fe.damage_op.pointer_cells
        fired = {"v": False}

        def _zeroed(db: Any) -> Any:
            fired["v"] = True
            return tuple(torch.zeros_like(t) for t in orig_cells(db))

        fe.damage_op.pointer_cells = _zeroed
        try:
            p, v = _forward_all()
        finally:
            fe.damage_op.pointer_cells = orig_cells
        if not fired["v"]:
            raise RuntimeError(
                "the concat_cells arm's pointer_cells patch was never called — the op's pointer "
                "surface moved and the arm measured nothing.")
        kl = _masked_kl(base_p, p, masks_t)
        report["concat_cells"] = {
            "kl_mean": float(kl.mean()),
            "kl_p95": float(kl.quantile(0.95)),
            "flip_rate": float((base_p.argmax(-1) != p.argmax(-1)).float().mean()),
            "dv_mean": float((base_v - v).abs().mean()),
        }
        # The patch is identity when restored — re-measuring base must reproduce it bitwise.
        chk_p, chk_v = _forward_all()
        assert torch.equal(chk_p, base_p) and torch.equal(chk_v, base_v), \
            "op-arm restore failed — the patch leaked into the baseline"  # noqa: S101
    return report


def main(argv: Sequence[str] | None = None) -> int:
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
