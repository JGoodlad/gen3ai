"""The CRITIC-ROUTE consolidation audit (cleanup journey Phase 3 / T3) — pre-registered arms.

Measures, on a TRAINED checkpoint over stratified eval-trace states, how much the policy and
critic lean on each of the parallel critic magnitude routes carried since the concat's death:

  arm `threat`      — zero `CLSPool`'s `threat_rows` (the --value-threat-inject token content)
  arm `pair_value`  — zero `CLSPool`'s `pair_rows` (the v95 PV shelf; the C4 gate its enabling owes)
  arm `hidden_opp`  — zero the HiddenOppBeliefPool concat (POLICY-side since the wave)
  arm `entity_pool` — zero the unified entity pool's output (gen3_unified_value_readout_v1,
                      the Stage-3 successor route; present only on a --value-entity-pool run)
  arm `event_seats` — key-mask ALL H-B event seats (the design's seat USAGE audit in ablation
                      form; present only on a --history-events run; NOT a zero-init route —
                      nonzero at init is expected, the verdict is read on a TRAINED run)
  arm `nmr`         — zero `non_matchup_rest` at the assembler (the LAST positional head
                      concat; POLICY-side since the wave deleted its vf half)
  arm `all_off`     — every present magnitude route together (the joint ceiling; nmr excluded)

**THE INSTRUMENT IS SMALLER THAN IT WAS, and the missing arms are the finding.** The
critic-route deletion wave retired `seed` (dV 0.0000 bit-exact, twice), `intent_reduce` (0.3176 at
2× sample), the `vr_*` arms for `intent_threshold_value` / `value_clock` / `value_intent`, and the
`_vf` half of `hidden_opp`. An arm whose subject no longer exists is not a null reading, it is a
DEAD INSTRUMENT — leaving it would either fabricate a 0.0 row or trip `_assert_fired` on every
run, and both read as measurements. The generic `value_route` arm STAYS (it now covers one route)
because that is the mechanism by which the next route is auditable the day it is written.

Each arm reports masked KL / argmax flips (policy) and |dV| (critic) against the unablated
forward, the same instrument family as `edge_ablation_audit` (whose state sampling and KL
helpers this reuses). The DIST head is deliberately NOT an arm: with `value_from_dist` the
distribution IS the critic's output parameterization, not a removable input route — its
verdict comes from quantile-coverage calibration, not ablation.

⚠️ Run this from the RUN'S OWN pinned worktree (`git worktree add <dir> <metadata git_hash>`),
copying this file in if the run predates it — a checkpoint is only loadable under the code
that trained it — and after the critic-route deletion wave a PRE-v96 checkpoint additionally
needs its own copy of this file, because the arms it deletes cannot be reconstructed from HEAD.

Usage:
  python -m agents.model.critic_route_audit <checkpoint.zip> \\
      --states 'models/<run>/eval_traces/**/*_states.npz' [--max-states 6000] [--out report.json]
  (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import json
import os
from typing import TYPE_CHECKING, Any, Sequence

import torch

from agents.model.edge_ablation_audit import _collect_states, _masked_kl, _measure

if TYPE_CHECKING:
    import numpy as np


@torch.no_grad()
def _forward_all(policy: Any, obs_np: "np.ndarray", masks_np: "np.ndarray",
                 batch: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __init__(self, fe: Any) -> None:
        self.fe = fe

    def threat(self) -> tuple[list[Any], Any]:
        fe = self.fe
        hooks = []

        def _pre(_m: Any, args: tuple[Any, ...],
                 kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
            tr = kwargs.get("threat_rows")
            if tr is not None:
                # hand-set did-it-match flag on the hook function object; mypy models a
                # function as attribute-less.
                _pre.fired = True  # type: ignore[attr-defined]
                kwargs = dict(kwargs)
                kwargs["threat_rows"] = torch.zeros_like(tr)
            return args, kwargs
        _pre.fired = False  # type: ignore[attr-defined]
        hooks.append(fe.cls_pool.register_forward_pre_hook(_pre, with_kwargs=True))
        return hooks, _pre

    def pair_value(self) -> tuple[list[Any], Any]:
        """gen3_pair_value_route_v1 (v95, PV) — zero the α-reduced UNIFIED outcome rows on their way
        into `CLSPool`, exactly as `threat()` does for v64's damage-only rows.

        This arm IS the C4-style offline gate the route's re-entry condition names: it prices what
        the critic actually leans on the status / neutralization / tempo currency for, in the same
        |dV| units as every other route, on one trained run. vf-only by construction, so a nonzero
        KL/flip reading here would itself be a finding (it would mean the injection leaked into pi).
        """
        fe = self.fe
        hooks = []

        def _pre(_m: Any, args: tuple[Any, ...],
                 kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
            pr = kwargs.get("pair_rows")
            if pr is not None:
                _pre.fired = True  # type: ignore[attr-defined]
                kwargs = dict(kwargs)
                kwargs["pair_rows"] = torch.zeros_like(pr)
            return args, kwargs
        _pre.fired = False  # type: ignore[attr-defined]
        hooks.append(fe.cls_pool.register_forward_pre_hook(_pre, with_kwargs=True))
        return hooks, _pre

    def event_seats(self) -> tuple[list[Any], dict[str, bool]]:
        """Key-mask ALL H-B event seats (force the pad mask True) — the design's "usage audit
        on the event seats" in the house ablation form: if the trunk learned to attend over
        the event window, masking it moves pi (KL/flips) and/or vf (|dV|); a dead-zero reading
        on a TRAINED run means the seats never came alive. Present only on a
        --history-events run. Affects BOTH heads (a trunk-level source)."""
        fe = self.fe
        marker = {"fired": False}

        def _hook(_m: Any, _args: Any, output: Any) -> Any:
            tokens, pad = output
            marker["fired"] = True
            return tokens, torch.ones_like(pad)
        return [fe.history_events.register_forward_hook(_hook)], marker

    def value_route(self, name: str) -> tuple[list[Any], dict[str, bool]]:
        """Zero ONE v89 `value_pooled` route by NAME, at the `_value_pooled_routes` registry seam.

        Deliberately generic rather than five bespoke arms. The seam already yields
        `(name, contribution)` for every enabled route, so keying off that name means the arm set
        cannot drift from the route set — a route added tomorrow is auditable the day it exists,
        which is exactly what did NOT happen for `value_clock` / `value_intent` /
        `intent_threshold_value`: they shipped in v87/v84, trained live through all of gen-13, and
        had no arm at all when the §2 verdict was computed.

        It also gives a free cross-check on the one route that ALSO has a bespoke arm
        (`entity_pool`) — two independent mechanisms reading one route should agree, and a
        disagreement means one of them is measuring the wrong thing.

        Implemented by wrapping the bound generator (the seam is a method, not a module, so there
        is no hook to register). `fired` is set only when the named route is actually yielded, so
        a typo or a disabled route fails the `_assert_fired` staleness guard rather than silently
        reporting a null.
        """
        fe = self.fe
        marker = {"fired": False}
        original = fe._value_pooled_routes

        def _wrapped(*a: Any, **k: Any) -> Any:
            for rname, contrib in original(*a, **k):
                if rname == name:
                    marker["fired"] = True
                    yield rname, torch.zeros_like(contrib)
                else:
                    yield rname, contrib

        fe._value_pooled_routes = _wrapped

        class _Restore:
            def remove(self_inner) -> None:
                fe._value_pooled_routes = original
        return [_Restore()], marker

    def frames(self) -> tuple[list[Any], dict[str, bool]]:
        """Zero the 7xN TurnDelta LAG-FRAME block AND the prev-turn action mask **at the obs
        input** — the DENOMINATOR the H-B `event_seats` arm is measured against.

        This is the only arm that ablates OBS CONTENT rather than a module output, because the
        thing under test is content, not a route: gen-14's headline change deletes these 1124
        dims outright, and "the seats carry at least as much as the frames they replace" is
        unanswerable without measuring the frames the same way the seats were measured.

        Zeroing at INPUT (a pre-hook on the extractor) rather than at some consumer is what makes
        the two arms comparable: `event_seats` key-masks the whole seat block, so its reading is
        "everything downstream of that content, gone". Zeroing a mid-pipeline tensor instead would
        measure one consumer and silently exclude the others (the frames feed the history tokens
        AND the global token).

        Offsets come from the layout's validated slice map, never a literal — the obs has moved
        twice in three generations (2669 -> 2921 -> 3529) and a hardcoded 1556 would now zero the
        middle of the event window while reporting a frames number.
        """
        from agents.observation.schema import build_schema
        fe = self.fe
        marker = {"fired": False}
        sl = build_schema(fe.layout).slices()
        # gen3_frame_deletion_v1 DELETED the blocks this arm zeroes. The arm stays because it is
        # the instrument that LICENSED that deletion (gen-13.5 §4, dV 1.3015 vs event_seats
        # 2.7714) and every archived pre-v90 checkpoint is still auditable with it — but on a
        # post-deletion layout there is nothing to zero, and silently reporting dV 0.0 would read
        # as "the frames were worthless" rather than "the frames are gone". RAISE instead: an
        # arm that cannot measure its subject must say so, not return a number.
        missing = [k for k in ("turn_history", "prev_action_mask") if k not in sl]
        if missing:
            raise RuntimeError(
                f"the `frames` arm has no subject on this architecture: {missing} absent from the "
                f"obs schema (gen3_frame_deletion_v1 deleted them at v90). Run this arm against a "
                f"pre-v90 checkpoint, from the commit its metadata.json git_hash names.")
        spans = [sl["turn_history"], sl["prev_action_mask"]]

        def _pre(_m: Any, args: Any, kwargs: Any) -> Any:
            obs = kwargs.get("observations")
            if obs is None and args:
                obs = args[0]
            if not isinstance(obs, dict) or "observation" not in obs:
                return args, kwargs
            x = obs["observation"].clone()
            for s in spans:
                x[..., s] = 0.0
            marker["fired"] = True
            new = dict(obs)
            new["observation"] = x
            if kwargs.get("observations") is not None:
                kwargs = dict(kwargs)
                kwargs["observations"] = new
                return args, kwargs
            return (new,) + tuple(args[1:]), kwargs
        return [fe.register_forward_pre_hook(_pre, with_kwargs=True)], marker

    def nmr(self) -> tuple[list[Any], dict[str, bool]]:
        """Zero `ctx.non_matchup_rest` at the assembler — the LAST positional head concat
        (global env + board scalars). Its content also rides the global token through the trunk,
        so this arm measures the DIRECT-shortcut dependency the Phase-3 item-2 deletion needs
        evidence for (cleanup journey §5.2). Since the critic-route wave deleted its VF half on a
        0.0000 reading, the arm is POLICY-side: a nonzero `dv_mean` here would be a finding."""
        import dataclasses

        fe = self.fe
        marker = {"fired": False}
        i = _arg_index(fe.assembler.forward, "ctx")

        def _pre(_m: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            ctx = args[i]
            nmr_t = getattr(ctx, "non_matchup_rest", None)
            if nmr_t is None:
                return args
            marker["fired"] = True
            new_ctx = dataclasses.replace(ctx, non_matchup_rest=torch.zeros_like(nmr_t))
            return args[:i] + (new_ctx,) + args[i + 1:]
        return [fe.assembler.register_forward_pre_hook(_pre)], marker

    def entity_pool(self) -> tuple[list[Any], dict[str, bool]]:
        """Zero the unified entity pool's OUTPUT (gen3_unified_value_readout_v1) — vf-only by
        construction, so this arm reads pure |dV| with structurally-zero KL/flips."""
        fe = self.fe
        marker = {"fired": False}

        def _hook(_m: Any, _args: Any, output: Any) -> Any:
            marker["fired"] = True
            return torch.zeros_like(output)
        return [fe.value_entity_pool.register_forward_hook(_hook)], marker

    def hidden_opp(self) -> tuple[list[Any], dict[str, bool]]:
        """Zero the assembler's belief argument.

        This used to take a `mode ∈ {'both','pi','vf'}` and run the assembler TWICE to split the
        heads. **The split is now structural rather than instrumental**: the pool feeds only pi,
        because gen-14 measured the vf half at dV 0.0000 while the pi half flipped 39.6% of
        argmaxes, and the wave deleted exactly the dead half. Keeping a three-mode arm would
        report `_both` and `_pi` as two different numbers that are now identically equal, which is
        the shape of a reading nobody can interpret."""
        fe = self.fe
        marker = {"fired": False}
        i = _arg_index(fe.assembler.forward, "hidden_opp_belief")

        def _pre(_m: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            if len(args) <= i or args[i] is None:
                return args
            marker["fired"] = True
            return args[:i] + (torch.zeros_like(args[i]),) + args[i + 1:]
        return [fe.assembler.register_forward_pre_hook(_pre)], marker


def _arg_index(fn: Any, param: str) -> int:
    """The POSITION of `param` in `fn`'s signature, resolved at hook-registration time.

    A `register_forward_pre_hook` without `with_kwargs` only ever sees positional args, so an
    arm that patches one has to know an index — but it must not HARDCODE one. The `_assert_fired`
    staleness guard catches an argument that DISAPPEARS (the hook stops matching); it cannot
    catch an argument INSERTED before the subject, which silently re-points the arm at the new
    occupant while every marker still fires. That is the `concat` failure exactly, one level in.
    Resolving the index from the live signature by NAME makes both directions loud: a rename
    raises here, an insertion moves the index with the subject.
    """
    import inspect

    params = list(inspect.signature(fn).parameters)
    if param not in params:
        raise RuntimeError(
            f"the audit's arm binds `{param}`, which is not an argument of {fn.__qualname__} "
            f"(it takes {params}) — the route identity drifted and the arm would measure the "
            "wrong thing rather than nothing.")
    return params.index(param)


def _assert_fired(name: str, markers: Sequence[Any]) -> None:
    """The staleness guard: an arm whose hook never matched measured NOTHING while producing a
    plausible report — the exact silent-failure mode this audit cannot afford once a year of
    refactors separates it from the forward it patches."""
    for mk in markers:
        fired = mk["fired"] if isinstance(mk, dict) else mk.fired
        if not fired:
            raise RuntimeError(f"arm {name}: a hook never matched its argument — "
                               "the route identity drifted and the arm measured nothing.")


def _route_is_enabled(fe: Any, name: str) -> bool:
    """Is this v89 value route BUILT on `fe`? Keyed on the module the seam actually calls, so a
    renamed attribute fails here (arm absent, visible in the report) rather than silently."""
    attr = {"value_entity_pool": "value_entity_pool"}[name]
    return getattr(fe, attr, None) is not None


def audit(policy: Any, obs_np: "np.ndarray", masks_np: "np.ndarray", batch: int = 512) -> dict:
    fe = policy.features_extractor
    arms = _Arms(fe)
    base_p, base_v = _forward_all(policy, obs_np, masks_np, batch)
    masks_t = torch.as_tensor(masks_np)
    report: dict[str, dict[str, float]] = {}
    # Arms whose SUBJECT this architecture no longer has. Kept OUT of `report`, whose
    # values are homogeneous dV/KL rows — a skip reason living there would either break
    # every consumer's `["kl_mean"]` or, worse, get coerced to a 0.0 that reads as a
    # measurement. Absent-with-a-reason and measured-as-zero must never look alike.
    skipped: dict[str, str] = {}

    def _run(name: str, hook_sets: Sequence[tuple[list[Any], Any]]) -> None:
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

    if getattr(fe, "value_threat_inject", False):
        _run("threat", [arms.threat()])
    if getattr(fe.cls_pool, "pair_value_proj", None) is not None:
        _run("pair_value", [arms.pair_value()])
    if getattr(fe, "hidden_opp_belief", None) is not None:
        _run("hidden_opp", [arms.hidden_opp()])
    if getattr(fe, "value_entity_pool", None) is not None:
        _run("entity_pool", [arms.entity_pool()])
    if getattr(fe, "history_events", None) is not None:
        _run("event_seats", [arms.event_seats()])
    # Every v89 value_pooled route, keyed off the registry seam itself so the arm set cannot drift
    # from the route set. `vr_value_entity_pool` deliberately DUPLICATES the bespoke `entity_pool`
    # arm — two mechanisms reading one route should agree, and a disagreement means one of them
    # measures the wrong thing. The seam has ONE member since the deletion wave; the loop stays
    # because its whole value is covering the NEXT one automatically.
    for _vr in ("value_entity_pool",):
        if _route_is_enabled(fe, _vr):
            _run(f"vr_{_vr}", [arms.value_route(_vr)])
    # The §4 DENOMINATOR. Run unconditionally — the frames are unconditional obs content, and the
    # comparison is only meaningful when both arms come from the SAME states in the SAME pass.
    # Deliberately NOT in all_off: that arm is the value-route joint, this one answers "would the
    # deletion cost anything", which is a different question about a different substrate.
    # gen3_frame_deletion_v1 deleted this arm's subject at v90. The ARM still raises when asked
    # directly (it must never fabricate a dV for a block that is not there), so the RUNNER checks
    # availability instead of catching — "skipped, and here is why" is a reportable state, whereas
    # a 0.0 in the results table would read as "the frames were worthless" to anyone scanning it.
    from agents.observation.schema import build_schema as _bs
    _sl = _bs(fe.layout).slices()
    if "turn_history" in _sl and "prev_action_mask" in _sl:
        _run("frames", [arms.frames()])
    else:
        skipped["frames"] = ("no subject on this architecture — the lag frames were deleted at "
                             "v90 (gen3_frame_deletion_v1); audit a pre-v90 checkpoint from its "
                             "own git_hash to measure them")
    # Always present (unconditional obs content): the last positional head concat. Deliberately
    # NOT part of all_off — that arm is the belief/magnitude-route joint; this one answers the
    # separate Phase-3 item-2 question (can the direct shortcut die once the global token
    # carries the content through the trunk).
    _run("nmr", [arms.nmr()])
    all_sets = []
    if getattr(fe, "value_threat_inject", False):
        all_sets.append(arms.threat())
    if getattr(fe.cls_pool, "pair_value_proj", None) is not None:
        all_sets.append(arms.pair_value())
    if getattr(fe, "hidden_opp_belief", None) is not None:
        all_sets.append(arms.hidden_opp())
    if getattr(fe, "value_entity_pool", None) is not None:
        all_sets.append(arms.entity_pool())
    if all_sets:
        _run("all_off", all_sets)
    for _k, _why in skipped.items():
        # LOUD, and deliberately not a row: `report` holds measurements, and an arm that could
        # not be measured must not occupy a slot in it under any encoding (0.0, NaN, or a string).
        print(f"  [audit] arm {_k!r} SKIPPED — {_why}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
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
