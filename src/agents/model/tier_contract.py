"""The TIER-ORDERING CONTRACT (`gen3_tiered_pipeline_v1`).

The forward pass resolves the game in the order the game itself resolves in:

    T0 RESOLVE   what is on the board?          species/move/spread/HP-type belief
    T1 REASON    what follows from that board?  the DamageOperator, the seats, the edges, attention
    T2 DECIDE    what will they do, and what     the post-attention readouts: belief aux, the pools,
                 does that make my moves worth?  the α/β intent heads
    T3 DELIVER   one contract, two pools         the assembler + the readouts off the pools

Until now that ordering held **by construction** — it was a property of how `forward_internal`
happened to be written, and nothing stopped a future head from reading α at T0 or a future T0 leg
from consuming a pooled board summary. This module makes it an ASSERTED INVARIANT, in the same
spirit as the leak-safety assertions in `delivery_graph_test.py`.

**Two checks, and the honest boundary between them.**

1. **ORDER** — every tier-declared entry point is entered in NON-DECREASING tier order within one
   forward. This is the direct regression guard on the thing that was just made unconditional: a
   re-introduced POST-transformer `move_belief` call site is a T0 entry after a T1 entry, and the
   check fails on it. Falsifiable, and `tier_contract_test.py` proves it by planting exactly that
   violation and asserting the checker reports it.

2. **PROVENANCE** — no tier-declared entry point receives, as an argument, a tensor whose STORAGE
   was produced by a strictly LATER tier. Keyed on `untyped_storage().data_ptr()` rather than on
   tensor identity, so it sees through `.detach()`, slices and other views — which matters, because
   the α head's input is a detached slice and a "just read α at T0" regression would arrive
   detached. Producing tensors are kept alive for the duration of the trace so a freed storage
   cannot be recycled into a false positive.

**What this CANNOT catch, stated plainly.** It is a check on DATA FLOW, not on meaning. It sees a
later tier's tensor arriving at an earlier tier. It does NOT see an earlier tier *recomputing*
something intent-like from raw tokens — a fresh `Linear(tokens) -> "which move will they click"`
at T0 produces a brand-new storage with no provenance, and no mechanical check can tell that
tensor from any other learned projection. That failure mode is a design review, not a test. What
the contract does buy is that such a head cannot then be FED the real α, and cannot run out of
order — which is the failure that actually shipped in this codebase (two placements of one
computation, selected by flag).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# --------------------------------------------------------------------------- the declaration

TIER_NAMES: Dict[int, str] = {
    0: "T0 RESOLVE",
    1: "T1 REASON",
    2: "T2 DECIDE",
    3: "T3 DELIVER",
}

#: extractor attribute -> tier. A module that is absent or None on a given config is skipped;
#: a module that is PRESENT but undeclared fails `test_every_child_module_declares_a_tier`, so
#: adding a phase forces a tier decision rather than silently escaping the contract.
TIER_OF: Dict[str, int] = {
    # T0 RESOLVE — the opponent's hidden state, from the Smogon prior ⊕ a learned delta.
    "pokemon_encoder": 0,
    # `t0_species_prior` is the counterpart of `belief_head`'s T2 note below, and the reason this
    # contract earns its keep: the SAME team-composition species belief is computed in both places,
    # and the tier is the whole difference. At T2 it is a training-only readout the physics cannot
    # reach — which is why the DamageOperator (T1) priced unrevealed defenders from a STATIC usage
    # table for as long as it did. Declared T0, it is a resolve step the op consumes directly.
    "t0_species_prior": 0,
    "belief_slots": 0,
    "move_belief": 0,
    "hp_type_belief_head": 0,
    "spread_belief": 0,
    # gen3_item_belief_v1 (v83): the hidden-item posterior — a resolve step exactly like the
    # species/spread/HP-type beliefs; the op (T1) consumes its p_cb publication.
    "item_belief_head": 0,
    # T1 REASON — physics over the resolved state, then attention over tokens that carry it.
    "damage_op": 1,
    "entity_seats": 1,
    "edge_bias": 1,
    "team_transformer": 1,
    # T2 DECIDE — the post-attention readouts.
    #
    # `belief_head` sits here DELIBERATELY, and the T0/T2 split of the species belief
    # (`BeliefSlots` injects unknown-mon tokens pre-trunk; `BeliefHead` reads refined tokens
    # post-trunk) is legitimate rather than a second resolve path: `BeliefHead` is a
    # TRAINING-ONLY side readout whose output is stashed for the aux loss and never fed forward.
    # Declaring it T2 is what records that fact — if it ever started feeding a T0/T1 consumer,
    # the provenance check would fail.
    "belief_head": 2,
    "cls_pool": 2,
    "alpha_head": 2,
    "beta_head": 2,
    # gen3_intent_move_cell_v1: the POLICY-side alpha consumer — weights the op's T1 c2 operand
    # stash by the T2 alpha publication into the pointer MOVE cell. T2 by the same logic as
    # `alpha_head` itself: it answers "what are my moves worth", post-attention.
    "intent_move_cell": 2,
    # gen3_intent_threshold_v1 (v84): the α-weighted threshold operator, at the pointer stash
    # beside intent_move_cell (T2). Its p_KO CRITIC half (`intent_threshold_value`, T3) was
    # deleted by the critic-route deletion wave; the move cell is the whole flag now.
    "intent_threshold_move": 2,
    # gen3_intent_conditional_v1 (v85): the Counter/flinch/Explosion/Pursuit cells — the same
    # pointer-stash placement as intent_move_cell.
    "intent_conditional": 2,
    # gen3_pair_outcome_v1 (v93): the α-reduced unified outcome vector, delivered to the pointer
    # MOVE cell. Same T1-producer (the op builds `pair_in`) / T2-consumer (α contracts it) split
    # as every other α cell — T2 is where α first exists.
    "pair_outcome_move": 2,
    # gen3_pair_outcome_switch_v1 / gen3_switch_branch_v1 (v94, Phase B): the same
    # T1-producer / T2-consumer split — the op builds `pair_in` and the outgoing grid,
    # α and β first exist at T2, and both cells run at the pointer stash.
    "pair_outcome_switch": 2,
    "switch_branch": 2,
    # gen3_conditional_threat_v1 (v95, Phase C — OA1): same split again. The op stashes `pair_in`
    # AND the per-(defender, seat) type multiplier at T1; α first exists at T2, and the cell runs
    # at the pointer stash. (PV, its Phase-C partner, is NOT listed: it lives under `cls_pool` as
    # a child module rather than as a top-level extractor child, so `cls_pool`'s own tier covers
    # it — which is also what makes its vf-only property structural.)
    "conditional_threat": 2,
    # T3 DELIVER — the pooled contract and the readouts taken off it.
    "hidden_opp_belief": 3,
    "assembler": 3,
    "win_head": 3,
    "value_dist_head": 3,
    # gen3_cf_evidential_head_v1 (v98): the evidential Beta readout, off the same `value_pooled` the
    # other two readouts take — T3 DELIVER by the same argument. It is the ONE tiered child the
    # extractor forward never calls (the training-side cf loss applies it to the STASHED
    # value_pooled), so the tracer will never see it; the declaration is here so a future config
    # that builds it does not trip `test_every_child_module_declares_a_tier`.
    "cf_evid_head": 3,
    # gen3_cf_twin_heads_v1 (v99): the twin win-prob heads and the shadow critic — the same three
    # facts as `cf_evid_head`. They read `value_pooled`, so T3 DELIVER; the forward never calls
    # them (the training-side terms apply them to the STASHED value_pooled), so the tracer will
    # never see them either; and the declaration exists so a config that builds them does not trip
    # `test_every_child_module_declares_a_tier`.
    "cf_twin_head_b": 3,
    "cf_twin_head_c": 3,
    "cf_shadow_head": 3,
    # gen3_q_winprob_head_v1 (v107): the PER-ACTION win-probability readout. T3 DELIVER — it reads
    # `value_pooled` AND the T2/T3 pointer stash, so it is strictly downstream of both, and it
    # publishes a readout rather than feeding any consumer. Unlike the four cf heads above the
    # forward DOES call it, so the tracer sees it, and its entry being T3 is what asserts nothing
    # earlier can consume its output.
    "q_winprob_head": 3,
    # gen3_unified_value_readout_v1 (v80): the Stage-3 unified critic entity pool, injected into
    # `value_pooled` through the v89 seam. Since the critic-route deletion wave it is the ONLY
    # member of that seam — and the only DELIVER-stage critic route other than the two `cls_pool`
    # token-content injections.
    "value_entity_pool": 3,
    # gen3_event_window_v1 (v81): event seats are TRUNK INPUT built pre-transformer from the
    # obs block — T1 REASON, exactly like entity_seats.
    "history_events": 1,
}

#: Children that own no tier: embedding tables, the obs unpacker, and the root projections that
#: sit AFTER `forward_internal` returns. Anything not listed here must appear in `TIER_OF`.
UNTIERED_CHILDREN = frozenset({
    "embeddings",           # shared tables — a resource, not a phase
    "unpack",               # ObsUnpack: produces the tier-0 INPUT, ahead of every tier
    "prefuse_proj",         # the T1->trunk residual, owned by the root rather than by a phase
    "pre_proj_norm", "projection", "value_pre_norm", "value_projection", "activation",
    "value_threat_proj",    # lives under cls_pool in production; listed for the LUT-fork paths
})

#: Modules invoked through NAMED METHODS rather than `__call__`, and therefore invisible to the
#: forward hooks. `MoveBelief` in particular is never called — `_apply_move_belief` reaches into
#: `move_logits` / `reinject_moves` — so without this it would silently escape both checks.
#: `damage_op`'s physics kernels are enumerated mechanically (every `pairwise_*` plus the named
#: discrete/candidate entry points) rather than hand-listed, so a new kernel is covered on arrival.
_NAMED_ENTRY_POINTS: Dict[str, Tuple[str, ...]] = {
    "move_belief": ("move_logits", "reinject_moves"),
    "hp_type_belief_head": ("compose_typed_hp", "reinject"),
    "spread_belief": (),
    "damage_op": ("refine_candidates", "discrete_outgoing_status", "discrete_incoming_status",
                  "pointer_intent_status_operands"),
}


def _entry_methods(name: str, module: Any) -> Tuple[str, ...]:
    """The named methods to instrument for `name` (on top of `__call__`)."""
    extra = _NAMED_ENTRY_POINTS.get(name, ())
    if name == "damage_op":
        extra = extra + tuple(sorted(
            m for m in dir(type(module)) if m.startswith("pairwise_")))
    return tuple(m for m in extra if callable(getattr(module, m, None)))


# --------------------------------------------------------------------------- the trace

@dataclasses.dataclass(frozen=True)
class TierViolation:
    kind: str            # "order" | "provenance"
    module: str
    tier: int
    detail: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (f"[{self.kind}] {self.module} ({TIER_NAMES.get(self.tier, self.tier)}): "
                f"{self.detail}")


@dataclasses.dataclass
class TierTrace:
    order: List[Tuple[str, int]]                 # (entry point, tier) in call order
    violations: List[TierViolation]

    @property
    def ok(self) -> bool:
        return not self.violations


def _storages(obj: Any, out: List[Tuple[int, torch.Tensor]], depth: int = 0) -> None:
    """Collect `(storage_ptr, tensor)` from a nested arg/return structure.

    `nn.Module` values are skipped (a module argument carries parameters, not dataflow), as are
    tensors with no storage. Recursion is depth-bounded so a cyclic structure cannot hang a test.
    """
    if depth > 4:
        return
    if isinstance(obj, torch.Tensor):
        try:
            out.append((obj.untyped_storage().data_ptr(), obj))
        except Exception:
            pass
        return
    if isinstance(obj, torch.nn.Module):
        return
    if isinstance(obj, (tuple, list)):
        for v in obj:
            _storages(v, out, depth + 1)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _storages(v, out, depth + 1)
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            _storages(getattr(obj, f.name, None), out, depth + 1)


class _Tracer:
    def __init__(self) -> None:
        self.order: List[Tuple[str, int]] = []
        self.violations: List[TierViolation] = []
        self.produced: Dict[int, int] = {}      # storage ptr -> earliest producing tier
        self._prev: Optional[Tuple[str, int]] = None   # per-FORWARD, reset at each boundary
        self._keepalive: List[torch.Tensor] = []
        self._handles: List[Any] = []
        self._patched: List[Tuple[Any, str]] = []

    def begin_forward(self) -> None:
        """ORDER is a within-forward property; PROVENANCE deliberately spans forwards.

        Resetting only `_prev` is what makes a STALE STASH (a T0 leg reading last forward's α)
        a pure provenance failure rather than a confusing order failure — and that hazard is
        real here: `_sb = self.last_spread_belief` used to be guarded precisely because a
        non-prefuse config would have read the PREVIOUS forward's value.
        """
        self._prev = None

    # -- the two checks -----------------------------------------------------
    def _enter(self, name: str, tier: int, args: Sequence[Any], kwargs: Dict[str, Any]) -> None:
        if self._prev is not None and self._prev[1] > tier:
            prev, prev_tier = self._prev
            self.violations.append(TierViolation(
                "order", name, tier,
                f"entered after {prev} ({TIER_NAMES.get(prev_tier, prev_tier)}) — the tier order "
                f"must be non-decreasing within one forward"))
        self._prev = (name, tier)
        self.order.append((name, tier))
        found: List[Tuple[int, torch.Tensor]] = []
        _storages(list(args), found)
        _storages(kwargs, found)
        for ptr, _t in found:
            src = self.produced.get(ptr)
            if src is not None and src > tier:
                self.violations.append(TierViolation(
                    "provenance", name, tier,
                    f"received a tensor produced at {TIER_NAMES.get(src, src)} — a "
                    f"{TIER_NAMES.get(tier, tier)} module may read only earlier tiers"))
                break

    def _exit(self, tier: int, output: Any) -> None:
        found: List[Tuple[int, torch.Tensor]] = []
        _storages(output, found)
        for ptr, t in found:
            prev = self.produced.get(ptr)
            if prev is None or tier < prev:
                self.produced[ptr] = tier
            self._keepalive.append(t)          # a freed storage must not be recycled into a lie

    # -- instrumentation ----------------------------------------------------
    def instrument(self, fe: Any) -> None:
        for name, tier in sorted(TIER_OF.items(), key=lambda kv: kv[1]):
            module = getattr(fe, name, None)
            if module is None:
                continue
            self._hook_call(name, tier, module)
            for meth in _entry_methods(name, module):
                self._patch_method(f"{name}.{meth}", tier, module, meth)

    def _hook_call(self, name: str, tier: int, module: Any) -> None:
        if not isinstance(module, torch.nn.Module):
            return

        def pre(_m: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any],
                _n: str = name, _t: int = tier) -> None:
            self._enter(_n, _t, args, kwargs)
            return None

        def post(_m: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any,
                 _t: int = tier) -> None:
            self._exit(_t, output)
            return None

        self._handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        self._handles.append(module.register_forward_hook(post, with_kwargs=True))

    def _patch_method(self, label: str, tier: int, module: Any, meth: str) -> None:
        bound = getattr(module, meth)

        def wrapper(*args: Any, _b: Any = bound, _l: str = label, _t: int = tier,
                    **kwargs: Any) -> Any:
            self._enter(_l, _t, args, kwargs)
            out = _b(*args, **kwargs)
            self._exit(_t, out)
            return out

        setattr(module, meth, wrapper)
        self._patched.append((module, meth))

    def restore(self) -> None:
        for h in self._handles:
            h.remove()
        for module, meth in self._patched:
            try:
                delattr(module, meth)          # drop the instance attr, re-exposing the class method
            except AttributeError:             # pragma: no cover - defensive
                pass
        self._handles.clear()
        self._patched.clear()
        self._keepalive.clear()


def trace_tiers(fe: Any, obs: Dict[str, torch.Tensor], forwards: int = 1) -> TierTrace:
    """Run `forwards` × `forward_internal` with the contract instrumented and report what it saw.

    `forwards=2` is the setting that can see a STALE-STASH leak (an earlier tier reading a later
    tier's tensor from the PREVIOUS forward), which a single forward structurally cannot: within
    one monotone forward a later tier's output does not exist yet.
    """
    tracer = _Tracer()
    tracer.instrument(fe)
    try:
        with torch.no_grad():
            for _ in range(max(1, forwards)):
                tracer.begin_forward()
                fe.forward_internal(obs)
    finally:
        tracer.restore()
    return TierTrace(order=tracer.order, violations=tracer.violations)


def assert_tier_contract(fe: Any, obs: Dict[str, torch.Tensor],
                         forwards: int = 2) -> TierTrace:
    """`trace_tiers`, raising `AssertionError` on any violation. Returns the trace when clean."""
    trace = trace_tiers(fe, obs, forwards=forwards)
    if trace.violations:
        seen = " -> ".join(f"{n}[{t}]" for n, t in trace.order)
        raise AssertionError(
            "tier-ordering contract violated:\n  "
            + "\n  ".join(str(v) for v in trace.violations)
            + f"\nobserved order: {seen}")
    return trace


def declared_tier(name: str) -> Optional[int]:
    """The declared tier of an extractor attribute, or None if it owns no tier."""
    return TIER_OF.get(name)
