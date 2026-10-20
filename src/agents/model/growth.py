"""DECLARED, FUNCTION-PRESERVING GROWTH AT A FORK (B-GROW, `gen3_grow_fork_v1`;
`designs/endstate/design_bot_ideas_screen.md` §4, `designs/model/versioning.md` "Declared growth at a fork").

A FORK may name fields of the network that GROW from its parent's recorded value (`--grow trunk_layers=4`,
`main.train.grow_fork`). Every other field must match the parent exactly, as on any fork. The child starts as the
parent's FUNCTION, bit for bit: the parent's parameters load STRICTLY (the normal loader, at the parent's own
architecture), then the new parameters are appended, each initialised so that it adds exactly zero.

THE GROWABLE FIELDS (one table, `GROWABLE`):

* ``trunk_layers`` — BUILT. ``N_parent -> N_child`` appends ``N_child - N_parent`` `trunk_depth.IdentityInitRound`s
  after the existing rounds: pre-LN residual blocks whose ``out_proj`` and ``linear2`` start at exactly zero (weight
  and bias), so each new round adds 0 to the residual stream. The new rounds are the SAME modules a fresh build at
  ``N_child`` would construct at those positions (`trunk_depth.build_extra_rounds`, the private seed), so a grown
  child and a fresh child differ only in the weights the parent trained.
* ``trunk_ffn_by_kind`` / ``opp_intent_grad_mode`` — DECLARED, NOT BUILT (B-KFFN, B-INT). Naming one is a typed
  refusal that names the build that adds it.

FIELD-GENERIC (`design_growth_builds_kffn_int.md` §3). Each BUILT row carries its own hooks — ``parse``, ``validate``,
``current`` (the value the loaded network is BUILT at), ``grow`` (in place; returns the new parameters) and an optional
``gate`` — and nothing outside the row names its field. :func:`apply_growth` loops over the declared fields in
``GROWABLE``'s declaration ORDER: one probe of the parent, every grower, ONE optimizer rebuild over the union of the new
parameters, one probe of the child; the forward bit-identity check and every row's extra gate read those probes.

THE OPTIMIZER RULE (F-BI-3, established 2026-10-19). A fork carries the parent's WHOLE optimizer state: SB3 loads the
zip's ``policy.optimizer.pth`` and `main.train.checkpoint_state._validate_or_reset_optimizer_state` realigns every
AdamW entry (``exp_avg``, ``exp_avg_sq``, ``step``) to the current parameters BY NAME, with the param group's
hyperparameters (the LR the parent's controller had reached included). Growth runs AFTER that realignment and keeps
it: :func:`regrow_optimizer` builds a new optimizer of the same class over the policy's parameters in module order
(the order a fresh child's `Gen3DualHeadMaskablePolicy._build` uses), copies every param group key, and carries each
PARENT parameter's state object unchanged (by identity: the parent's parameters are the child's). A NEW parameter
gets NO state, so AdamW initialises it at its first step: zero moments and step 0, i.e. its own bias correction.

THE FORK-TIME GATE. :func:`apply_growth` evaluates the parent and the grown child on REAL observation rows (the
committed compile-parity fixture, 64 rows of reproducible Rust-core battles) on FRESH CPU builds (the parent's
kwargs, then the child's, each loaded strictly with the live weights), fp32, eager, and refuses
(`GrowthError`, a `ModelVersionError`, so the trainer exits FATAL_CONFIG) unless every action logit and every value
is BIT-IDENTICAL. The CPU builds are what make "bit-identical" a deterministic check on a CUDA learner (a GPU kernel
need not reproduce itself bit for bit), and the child's strict load proves the grown state dict is exactly a fresh
child build's key set and shapes.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch

from agents.model.model_version.constants import ModelVersionError

#: The lineage stamp's / the `growth` record's schema.
SCHEMA = "gen3_grow_fork_v1"


class GrowthError(ModelVersionError):
    """A declared growth the loader must refuse, or a grown network that is not its parent's function."""


class Grown(NamedTuple):
    """What one row's grower appended: the new ``(name, parameter)`` pairs in module order, and ``copies`` —
    ``{id(new parameter): the existing parameter it is a COPY of}`` for a row whose new parameters copy existing ones
    (the optimizer then copies that parameter's AdamW state: the row's ``optimizer_rule``); empty when they start
    fresh."""

    new: List[Tuple[str, torch.nn.Parameter]]
    copies: Dict[int, torch.nn.Parameter]


#: A row's extra gate: ``(parent probe, child probe, (obs, mask)) -> report keys`` merged into the fork-time gate's
#: report (the forward bit-identity check is every row's, run once on the same probes); raises `GrowthError`.
GateHook = Callable[[Any, Any, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class Growable:
    """One field a fork may grow, with its hooks (module docstring, FIELD-GENERIC). A BUILT row carries every hook but
    the optional ``gate``; an unbuilt row is only its declaration."""

    field: str
    built: bool
    build: str          # the build that adds (or added) the growth
    what: str           # one line: what the growth appends
    #: ``text -> the typed child value`` (`GrowthError` on a bad value)
    parse: Optional[Callable[[str], Any]] = None
    #: ``(parent, child) -> None``: refuses a step this row cannot initialise as the identity (equal is not judged)
    validate: Optional[Callable[[Any, Any], None]] = None
    #: ``policy -> the field's value as BUILT`` (what the loaded network actually is)
    current: Optional[Callable[[Any], Any]] = None
    #: ``(policy, child) -> Grown``: grows the policy IN PLACE (modules, the extractor's attribute and kwargs)
    grow: Optional[Callable[[Any, Any], Grown]] = None
    gate: Optional[GateHook] = None
    #: the row's addendum to `OPTIMIZER_RULE`, recorded in the stamp (empty = the shared rule alone)
    optimizer_rule: str = ""


#: The optimizer rule, as recorded in the lineage stamp and the `growth` record.
OPTIMIZER_RULE = ("every parent parameter's AdamW state (exp_avg, exp_avg_sq, step) carried unchanged after the "
                  "fork's by-name realignment; every NEW parameter starts with no state (zero moments, step 0); the "
                  "param group (LR included) carried")

#: Bit-identity is the bar (a new round adds an exact 0); this is how many rows the gate reads.
GATE_ROWS_SOURCE = "agents/model/compile_parity_obs.npz (agents.model.compile_parity_fixture)"


# ------------------------------------------------------------------------------------------ the declaration
def require_built(field: str) -> Growable:
    """The table row for ``field``, or `GrowthError` naming why it cannot grow."""
    row = GROWABLE.get(field)
    if row is None:
        raise GrowthError(f"--grow {field}: not a growable field (growable: {', '.join(sorted(GROWABLE))})")
    if not row.built:
        raise GrowthError(
            f"--grow {field}: DECLARED growable but NOT BUILT — it needs {row.build} ({row.what}). Until that build "
            "lands, a fork cannot change it.")
    return row


def parse_value(field: str, text: str) -> Any:
    """The typed child value of ``field`` (the row's ``parse``)."""
    row = require_built(field)
    assert row.parse is not None, f"built growable row {field} has no parse hook"
    return row.parse(text)


def validate_step(field: str, parent: Any, child: Any) -> None:
    """``parent -> child`` is a growth of ``field`` this code can initialise as the identity (equal is NOT a
    growth: the caller decides whether an unchanged value is a restart or a meaningless flag)."""
    row = require_built(field)
    assert row.validate is not None, f"built growable row {field} has no validate hook"
    row.validate(parent, child)


def optimizer_rule(fields: Any) -> str:
    """The optimizer rule a growth of ``fields`` records: `OPTIMIZER_RULE`, plus each declared row's addendum."""
    extra = [f"{f}: {GROWABLE[f].optimizer_rule}" for f in GROWABLE if f in fields and GROWABLE[f].optimizer_rule]
    return OPTIMIZER_RULE + ("".join(f"; {e}" for e in extra))


def validate_declared(saved: Any, current: Any, growth: Mapping[str, Mapping[str, Any]]) -> None:
    """The loader's half: every declared field is built, the saved config records the declared PARENT value, the
    current version is the declared CHILD value, and the step is a growth. ``growth`` is ``{field: {"parent": p,
    "child": c}}`` (the lineage stamp's ``fields`` block)."""
    if not growth:
        raise GrowthError("a declared growth names no field")
    for field, step in growth.items():
        require_built(field)
        p, c = step.get("parent"), step.get("child")
        if getattr(saved, field) != p:
            raise GrowthError(f"declared growth {field} {p!r} -> {c!r}, but the checkpoint's model_config.json "
                              f"records {field}={getattr(saved, field)!r}")
        if getattr(current, field) != c:
            raise GrowthError(f"declared growth {field} {p!r} -> {c!r}, but this run builds "
                              f"{field}={getattr(current, field)!r}")
        if p == c:
            raise GrowthError(f"declared growth {field} {p!r} -> {c!r} changes nothing")
        validate_step(field, p, c)


# ------------------------------------------------------------------------------------------ the model
def _transformer(policy: Any) -> Any:
    tt = getattr(getattr(policy, "features_extractor", None), "team_transformer", None)
    if tt is None or not hasattr(tt, "trunk_layers"):
        raise GrowthError("the policy has no features_extractor.team_transformer with a trunk depth to grow")
    return tt


def trunk_layers_of(policy: Any) -> int:
    return int(_transformer(policy).trunk_layers)


def _parse_trunk_layers(text: str) -> int:
    from agents.model.trunk_depth import TRUNK_LAYERS_CHOICES
    try:
        v = int(str(text).strip())
    except ValueError:
        raise GrowthError(f"--grow trunk_layers={text!r}: not an integer") from None
    if v not in TRUNK_LAYERS_CHOICES:
        raise GrowthError(f"--grow trunk_layers={v}: the trunk depth must be one of {TRUNK_LAYERS_CHOICES}")
    return v


def _validate_trunk_layers(parent: Any, child: Any) -> None:
    if int(child) < int(parent):
        raise GrowthError(
            f"--grow trunk_layers={child}: the parent records {parent} rounds — growth only ADDS rounds (a "
            "trained round cannot be removed as an identity). Start a fresh run to train a shallower trunk.")


def _grow_trunk_layers_row(policy: Any, child: Any) -> Grown:
    return Grown(grow_trunk_layers(policy, int(child)), {})


def grow_trunk_layers(policy: Any, trunk_layers: int) -> List[Tuple[str, torch.nn.Parameter]]:
    """Append identity-init rounds IN PLACE so the trunk has ``trunk_layers`` rounds. The existing rounds (and every
    other module) are untouched objects; the new ones are `build_extra_rounds(trunk_layers)`'s modules at the new
    positions, moved to the trunk's device. Returns the new ``(name, parameter)`` pairs in module order."""
    from agents.model.arch_constants import TRANSFORMER_N_LAYERS
    from agents.model.trunk_depth import build_extra_rounds

    fe = policy.features_extractor
    tt = _transformer(policy)
    old = int(tt.trunk_layers)
    validate_step("trunk_layers", old, trunk_layers)
    if int(trunk_layers) == old:
        raise GrowthError(f"grow_trunk_layers: the trunk already has {old} rounds")
    fresh = build_extra_rounds(int(trunk_layers))
    assert fresh is not None
    existing = list(tt.extra_rounds) if tt.extra_rounds is not None else []
    if len(existing) != old - TRANSFORMER_N_LAYERS:
        raise GrowthError(f"the trunk records {old} rounds but holds {len(existing)} extra round(s), "
                          f"not {old - TRANSFORMER_N_LAYERS}")
    ref = next(tt.parameters())
    added = [fresh[i].to(device=ref.device) for i in range(len(existing), len(fresh))]
    tt.extra_rounds = torch.nn.ModuleList(existing + added)
    tt.trunk_layers = int(trunk_layers)
    fe.trunk_layers = int(trunk_layers)
    for kw in (getattr(policy, "features_extractor_kwargs", None),):
        if isinstance(kw, dict):
            kw["trunk_layers"] = int(trunk_layers)
    added_ids = {id(p) for m in added for p in m.parameters()}
    return [(n, p) for n, p in policy.named_parameters() if id(p) in added_ids]


def regrow_optimizer(policy: Any, new_params: Sequence[torch.nn.Parameter],
                     copies: Optional[Mapping[int, torch.nn.Parameter]] = None) -> Dict[str, int]:
    """Rebuild ``policy.optimizer`` over the grown policy (module docstring, THE OPTIMIZER RULE). Returns counts:
    ``carried`` (parent parameters whose state moved over), ``no_state`` (parent parameters that had none),
    ``fresh`` (new parameters, no state) and — only when a row declares ``copies`` (``{id(new): source}``) —
    ``copied``: new parameters given an independent CLONE of their source's AdamW state (``exp_avg``,
    ``exp_avg_sq``, ``step``); a copy whose source has no state is ``fresh``."""
    old = getattr(policy, "optimizer", None)
    if old is None:
        return {"carried": 0, "no_state": 0, "fresh": len(new_params)}
    if len(old.param_groups) != 1:
        raise GrowthError(f"the policy optimizer has {len(old.param_groups)} param groups; growth carries exactly one")
    old_ids = {id(p) for p in old.param_groups[0]["params"]}
    new_ids = {id(p) for p in new_params}
    params = [p for p in policy.parameters() if id(p) in old_ids or id(p) in new_ids]
    if len(params) != len(old_ids) + len(new_ids):
        raise GrowthError(f"growth could not place every optimizer parameter: {len(params)} found, "
                          f"{len(old_ids)} old + {len(new_ids)} new expected")
    group = {k: v for k, v in old.param_groups[0].items() if k != "params"}
    opt = type(old)(params, lr=group["lr"])
    opt.defaults = dict(old.defaults)
    opt.param_groups[0].update(group)
    counts = {"carried": 0, "no_state": 0, "fresh": len(new_ids)}
    if copies:
        counts["copied"] = 0
    for p in params:
        if id(p) not in old_ids:
            src = (copies or {}).get(id(p))
            st = old.state.get(src) if src is not None else None
            if st:
                opt.state[p] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in st.items()}
                counts["copied"] += 1
                counts["fresh"] -= 1
            continue
        st = old.state.get(p)
        if st:
            opt.state[p] = st
            counts["carried"] += 1
        else:
            counts["no_state"] += 1
    policy.optimizer = opt
    return counts


# ------------------------------------------------------------------------------------------ the gate
def cpu_probe(model: Any) -> Any:
    """A FRESH CPU build of ``model``'s policy from its CURRENT ``policy_kwargs`` (no optimizer, built inside
    `isolated_global_rng`, so the global streams do not move), loaded STRICTLY with the live policy's weights, fp32,
    eval mode. After a growth this is also the proof that the grown state dict is exactly a fresh child build's."""
    from agents.model.trunk_precision import set_trunk_precision
    from agents.training.global_rng_guard import isolated_global_rng
    from agents.training.instrumented_ppo.inference import _no_optimizer

    pol = model.policy
    pk = {**model.policy_kwargs, "optimizer_class": _no_optimizer, "optimizer_kwargs": {}}
    with isolated_global_rng():
        probe = model.policy_class(pol.observation_space, pol.action_space, model.lr_schedule, **pk)
    probe.load_state_dict({k: v.detach().to("cpu") for k, v in pol.state_dict().items()}, strict=True)
    set_trunk_precision(probe, "fp32")
    probe.set_training_mode(False)
    return probe


def forward_record(policy: Any, obs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """``{"logits": [N, A], "values": [N, 1]}`` — the masked action logits and the critic's value."""
    with torch.no_grad():
        o = {"observation": torch.as_tensor(obs), "action_mask": torch.as_tensor(mask.astype(np.int8))}
        logits = policy.get_distribution(o, action_masks=mask).distribution.logits.numpy().copy()
        values = policy.predict_values(o).numpy().copy()
    return {"logits": logits, "values": values}


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    fin = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return float("inf")
    return float(np.abs(a[fin] - b[fin]).max()) if fin.any() else 0.0


def gate_rows(policy: Any) -> Tuple[np.ndarray, np.ndarray]:
    """The REAL rows the fork-time gate reads (`compile_parity_fixture.load_parity_rows`; a stale or missing fixture
    is its typed refusal)."""
    from agents.model.compile_parity_fixture import load_parity_rows

    obs_space = policy.observation_space
    dim = int((obs_space["observation"] if hasattr(obs_space, "spaces") else obs_space).shape[0])
    return load_parity_rows(dim)


def apply_growth(model: Any, growth: Mapping[str, Mapping[str, Any]], *,
                 rows: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
    """Grow ``model`` (an SB3 algorithm whose policy was loaded STRICTLY at the parent's architecture) IN PLACE by
    ``growth`` (``{field: {"parent": p, "child": c}}``), rebuild its optimizer by THE OPTIMIZER RULE and run THE
    FORK-TIME GATE. The declared fields grow in ``GROWABLE``'s declaration order (depth before the FFN experts, so a
    new round's zero FFN is what gets copied). Returns the report recorded beside the stamp; raises `GrowthError` on any
    refusal."""
    policy = model.policy
    rows_of: Dict[str, Growable] = {}
    for field, step in growth.items():
        row = require_built(field)
        assert row.current is not None and row.grow is not None, f"built growable row {field} lacks a hook"
        have = row.current(policy)
        if have != step.get("parent"):
            raise GrowthError(f"declared growth {field} {step.get('parent')!r} -> {step.get('child')!r}, but the "
                              f"loaded network has {field}={have!r}")
        rows_of[field] = row
    order = [f for f in GROWABLE if f in rows_of]
    obs, mask = rows if rows is not None else gate_rows(policy)
    parent_probe = cpu_probe(model)
    ref = forward_record(parent_probe, obs, mask)
    new_named: List[Tuple[str, torch.nn.Parameter]] = []
    copies: Dict[int, torch.nn.Parameter] = {}
    pk = getattr(model, "policy_kwargs", None)
    for field in order:
        child = growth[field]["child"]
        grown = rows_of[field].grow(policy, child)       # type: ignore[misc]
        new_named += grown.new
        copies.update(grown.copies)
        if isinstance(pk, dict) and isinstance(pk.get("features_extractor_kwargs"), dict):
            pk["features_extractor_kwargs"][field] = child
    counts = regrow_optimizer(policy, [p for _, p in new_named], copies)
    # The child's ACTUAL weights (new parameters included), loaded STRICTLY into a fresh build at the child's kwargs.
    child_probe = cpu_probe(model)
    got = forward_record(child_probe, obs, mask)
    d_logits = _max_abs_diff(ref["logits"], got["logits"])
    d_values = _max_abs_diff(ref["values"], got["values"])
    identical = bool(np.array_equal(ref["logits"], got["logits"]) and np.array_equal(ref["values"], got["values"]))
    report: Dict[str, Any] = {
        "function_check": {
            "rows": int(obs.shape[0]), "rows_source": GATE_ROWS_SOURCE, "device": "cpu", "dtype": "float32",
            "mode": "eager", "max_abs_d_logits": d_logits, "max_abs_d_value": d_values, "bit_identical": identical,
        },
        "optimizer_state": dict(counts),
        "new_parameters": {"tensors": len(new_named), "elements": int(sum(p.numel() for _, p in new_named))},
    }
    if not identical:
        raise GrowthError(
            f"the grown network is NOT its parent's function on {obs.shape[0]} real rows (fp32 eager CPU): max |d "
            f"logits| {d_logits:.3g}, max |d value| {d_values:.3g}. A growth must start as the identity — a new "
            "parameter's init is not function-preserving. Refused.")
    for field in order:
        gate = rows_of[field].gate
        if gate is not None:
            report.update(gate(parent_probe, child_probe, (obs, mask)))
    return report


# ------------------------------------------------------------------------------------------ the table
GROWABLE: Dict[str, Growable] = {
    "trunk_layers": Growable(
        "trunk_layers", True, "B-GROW",
        "appends identity-init pre-LN trunk rounds (trunk_depth.IdentityInitRound: out_proj and linear2 zero)",
        parse=_parse_trunk_layers, validate=_validate_trunk_layers, current=trunk_layers_of,
        grow=_grow_trunk_layers_row),
    "trunk_ffn_by_kind": Growable(
        "trunk_ffn_by_kind", False, "B-KFFN",
        "per-token-kind FFN experts upcycled as copies of the shared FFN (design_bot_ideas_screen.md §2.2)"),
    "opp_intent_grad_mode": Growable(
        "opp_intent_grad_mode", False, "B-INT",
        "the opponent-intent objective's trunk gradient switch (design_bot_ideas_screen.md §2.3)"),
}
