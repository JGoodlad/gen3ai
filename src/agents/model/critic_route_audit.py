"""The CRITIC-ROUTE consolidation audit (cleanup journey Phase 3 / T3) — pre-registered arms.

Measures, on a TRAINED checkpoint over stratified eval-trace states, how much the policy and
critic lean on each of the parallel critic magnitude routes carried since the concat's death:

  arm `seed`        — zero the assembler's `seed_rows` (the MultiSeedValueReadout window)
  arm `threat`      — zero `CLSPool`'s `threat_rows` (the --value-threat-inject token content)
  arm `hidden_opp`  — zero the 768-dim HiddenOppBeliefPool concat: `_both`, `_pi`, `_vf`
  arm `entity_pool` — zero the unified entity pool's output (gen3_unified_value_readout_v1,
                      the Stage-3 successor route; present only on a --value-entity-pool run)
  arm `intent_reduce` — zero the α-weighted IntentValueReduce vf term (v74; present only on an
                      --intent-value-reduce run — gen-9 through gen-11 all train it)
  arm `event_seats` — key-mask ALL H-B event seats (the design's seat USAGE audit in ablation
                      form; present only on a --history-events run; NOT a zero-init route —
                      nonzero at init is expected, the verdict is read on a TRAINED run)
  arm `nmr`         — zero `non_matchup_rest` at the assembler (the LAST positional head
                      concat, both heads; the Phase-3 item-2 deletion evidence — separate
                      from all_off, which stays the belief/magnitude-route joint)
  arm `all_off`     — every present magnitude route together (the joint ceiling; nmr excluded)

Each arm reports masked KL / argmax flips (policy) and |dV| (critic) against the unablated
forward, the same instrument family as `edge_ablation_audit` (whose state sampling and KL
helpers this reuses). The DIST head is deliberately NOT an arm: with `value_from_dist` the
distribution IS the critic's output parameterization, not a removable input route — its
verdict comes from quantile-coverage calibration, not ablation.

⚠️ Run this from the RUN'S OWN pinned worktree (`git worktree add <dir> <metadata git_hash>`),
copying this file in if the run predates it — a checkpoint is only loadable under the code
that trained it. All APIs used here exist since v64 (threat_rows) / v76 (seed_rows).

Usage:
  export PYTHONPATH=$PYTHONPATH:src
  python -m agents.model.critic_route_audit <checkpoint.zip> \\
      --states 'models/<run>/eval_traces/**/*_states.npz' [--max-states 6000] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from agents.model.edge_ablation_audit import _collect_states, _masked_kl, _measure


@torch.no_grad()
def _forward_all(policy, obs_np, masks_np, batch=512):
    """The edge_ablation_audit forward contract verbatim: `policy.get_distribution` +
    `policy.predict_values` (which internally respect PopArt and value_from_dist — the exact
    reason not to hand-roll the value path here)."""
    device = next(policy.parameters()).device
    has_mask_key = "action_mask" in getattr(policy.observation_space, "spaces", {})
    ps, vs = [], []
    for i in range(0, len(obs_np), batch):
        mk = torch.as_tensor(masks_np[i:i + batch], device=device).bool()
        ob = {"observation": torch.as_tensor(obs_np[i:i + batch], device=device)}
        if has_mask_key:                    # real run policies; test fixtures may be Box-only
            ob["action_mask"] = mk.float()
        p, v = _measure(policy, ob, mk)
        ps.append(p.cpu())
        vs.append(v.cpu())
    return torch.cat(ps), torch.cat(vs)


class _Arms:
    """Context managers zeroing one route each; identity-checked to fail loud on drift."""

    def __init__(self, fe):
        self.fe = fe

    def seed(self):
        fe = self.fe
        hooks = []

        def _pre(_m, args):
            _rows = (fe.damage_op.last_tensors.incoming_rows
                     if getattr(fe.damage_op, "last_tensors", None) is not None else None)
            if args and args[-1] is not None and args[-1] is _rows:
                _pre.fired = True
                return args[:-1] + (torch.zeros_like(args[-1]),)
            return args
        _pre.fired = False
        hooks.append(fe.assembler.register_forward_pre_hook(_pre))
        return hooks, _pre

    def threat(self):
        fe = self.fe
        hooks = []

        def _pre(_m, args, kwargs):
            tr = kwargs.get("threat_rows")
            if tr is not None:
                _pre.fired = True
                kwargs = dict(kwargs)
                kwargs["threat_rows"] = torch.zeros_like(tr)
            return args, kwargs
        _pre.fired = False
        hooks.append(fe.cls_pool.register_forward_pre_hook(_pre, with_kwargs=True))
        return hooks, _pre

    def event_seats(self):
        """Key-mask ALL H-B event seats (force the pad mask True) — the design's "usage audit
        on the event seats" in the house ablation form: if the trunk learned to attend over
        the event window, masking it moves pi (KL/flips) and/or vf (|dV|); a dead-zero reading
        on a TRAINED run means the seats never came alive. Present only on a
        --history-events run. Affects BOTH heads (a trunk-level source)."""
        fe = self.fe
        marker = {"fired": False}

        def _hook(_m, _args, output):
            tokens, pad = output
            marker["fired"] = True
            return tokens, torch.ones_like(pad)
        return [fe.history_events.register_forward_hook(_hook)], marker

    def intent_reduce(self):
        """Zero the α-weighted IntentValueReduce term (v74, vf-only zero-init concat) — the
        critic-side α consumer gen-9+ trains. Its arm belongs in the same §2 consolidation as
        seed/threat: the runbook's Phase-3 list names it, and without an arm the route would be
        adjudicated by argument instead of measurement."""
        fe = self.fe
        marker = {"fired": False}

        def _hook(_m, _args, output):
            marker["fired"] = True
            return torch.zeros_like(output)
        return [fe.intent_value_reduce.register_forward_hook(_hook)], marker

    def nmr(self):
        """Zero `ctx.non_matchup_rest` at the assembler — the LAST positional head concat
        (global env + board scalars, both heads). Its content also rides the global token
        through the trunk, so this arm measures the DIRECT-shortcut dependency the Phase-3
        item-2 deletion needs evidence for (cleanup journey §5.2)."""
        import dataclasses

        fe = self.fe
        marker = {"fired": False}

        def _pre(_m, args):
            ctx = args[4]
            nmr_t = getattr(ctx, "non_matchup_rest", None)
            if nmr_t is None:
                return args
            marker["fired"] = True
            new_ctx = dataclasses.replace(ctx, non_matchup_rest=torch.zeros_like(nmr_t))
            return args[:4] + (new_ctx,) + args[5:]
        return [fe.assembler.register_forward_pre_hook(_pre)], marker

    def entity_pool(self):
        """Zero the unified entity pool's OUTPUT (gen3_unified_value_readout_v1) — vf-only by
        construction, so this arm reads pure |dV| with structurally-zero KL/flips."""
        fe = self.fe
        marker = {"fired": False}

        def _hook(_m, _args, output):
            marker["fired"] = True
            return torch.zeros_like(output)
        return [fe.value_entity_pool.register_forward_hook(_hook)], marker

    def hidden_opp(self, mode: str):
        """mode ∈ {'both', 'pi', 'vf'} — patch the assembler's belief argument per head."""
        fe = self.fe
        orig = fe.assembler.forward
        marker = {"fired": False}

        def patched(our_p, their_p, our_act, val_p, ctx, hidden_opp_belief=None, seed_rows=None):
            if hidden_opp_belief is None:
                return orig(our_p, their_p, our_act, val_p, ctx,
                            hidden_opp_belief=None, seed_rows=seed_rows)
            marker["fired"] = True
            z = torch.zeros_like(hidden_opp_belief)
            if mode == "both":
                return orig(our_p, their_p, our_act, val_p, ctx,
                            hidden_opp_belief=z, seed_rows=seed_rows)
            pi_z, vf_z = orig(our_p, their_p, our_act, val_p, ctx,
                              hidden_opp_belief=z, seed_rows=seed_rows)
            pi_r, vf_r = orig(our_p, their_p, our_act, val_p, ctx,
                              hidden_opp_belief=hidden_opp_belief, seed_rows=seed_rows)
            return (pi_z, vf_r) if mode == "pi" else (pi_r, vf_z)

        fe.assembler.forward = patched
        class _H:
            def remove(self_inner):
                fe.assembler.forward = orig
        return [_H()], marker


def _assert_fired(name: str, markers) -> None:
    """The staleness guard: an arm whose hook never matched measured NOTHING while producing a
    plausible report — the exact silent-failure mode this audit cannot afford once a year of
    refactors separates it from the forward it patches."""
    for mk in markers:
        fired = mk["fired"] if isinstance(mk, dict) else mk.fired
        if not fired:
            raise RuntimeError(f"arm {name}: a hook never matched its argument — "
                               "the route identity drifted and the arm measured nothing.")


def audit(policy, obs_np, masks_np, batch=512) -> dict:
    fe = policy.features_extractor
    arms = _Arms(fe)
    base_p, base_v = _forward_all(policy, obs_np, masks_np, batch)
    masks_t = torch.as_tensor(masks_np)
    report = {}

    def _run(name, hook_sets):
        hooks, markers = [], []
        for hs, mk in hook_sets:
            hooks += hs
            markers.append(mk)
        try:
            p, v = _forward_all(policy, obs_np, masks_np, batch)
        finally:
            for h in hooks:
                h.remove()
        _assert_fired(name, markers)
        kl = _masked_kl(base_p, p, masks_t)
        report[name] = {
            "kl_mean": float(kl.mean()), "kl_p95": float(kl.quantile(0.95)),
            "flip_rate": float((base_p.argmax(-1) != p.argmax(-1)).float().mean()),
            "dv_mean": float((base_v - v).abs().mean()),
        }

    if getattr(fe.assembler, "seed_readout", None) is not None:
        _run("seed", [arms.seed()])
    if getattr(fe, "value_threat_inject", False):
        _run("threat", [arms.threat()])
    if getattr(fe, "hidden_opp_belief", None) is not None:
        for mode in ("both", "pi", "vf"):
            _run(f"hidden_opp_{mode}", [arms.hidden_opp(mode)])
    if getattr(fe, "value_entity_pool", None) is not None:
        _run("entity_pool", [arms.entity_pool()])
    if getattr(fe, "intent_value_reduce", None) is not None:
        _run("intent_reduce", [arms.intent_reduce()])
    if getattr(fe, "history_events", None) is not None:
        _run("event_seats", [arms.event_seats()])
    # Always present (unconditional obs content): the last positional head concat. Deliberately
    # NOT part of all_off — that arm is the belief/magnitude-route joint; this one answers the
    # separate Phase-3 item-2 question (can the direct shortcut die once the global token
    # carries the content through the trunk).
    _run("nmr", [arms.nmr()])
    all_sets = []
    if getattr(fe.assembler, "seed_readout", None) is not None:
        all_sets.append(arms.seed())
    if getattr(fe, "value_threat_inject", False):
        all_sets.append(arms.threat())
    if getattr(fe, "hidden_opp_belief", None) is not None:
        all_sets.append(arms.hidden_opp("both"))
    if getattr(fe, "value_entity_pool", None) is not None:
        all_sets.append(arms.entity_pool())
    if getattr(fe, "intent_value_reduce", None) is not None:
        all_sets.append(arms.intent_reduce())
    if all_sets:
        _run("all_off", all_sets)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--max-states", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    obs, masks, coverage = _collect_states(a.states, a.max_states)
    # Inference-only load, the edge_ablation_audit pattern (arch-FAMILY gate, no env needed).
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings
    model, _ver = load_foreign_opponent(a.checkpoint,
                                        current_version=current_model_version(load_mappings()),
                                        device="cpu")
    rep = {"checkpoint": os.path.abspath(a.checkpoint), "n_states": int(len(obs)),
           "coverage": coverage, "arms": audit(model.policy, obs, masks, a.batch)}
    txt = json.dumps(rep, indent=2)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
