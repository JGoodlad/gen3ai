"""`ModelFlag.requires` and the constructor must know exactly the same dependencies — both ways.

WHY BOTH DIRECTIONS. A declaration that nothing checks is a comment, and a check nothing declares is
invisible. The registry's whole thesis is that a fact written in two places drifts, so `requires`
would be worth very little if it were merely *asserted* to match `Gen3FeaturesExtractor.__init__`:

  * FORWARD — for every declared `flag requires dep`, build the extractor with `flag` on (plus its
    transitive closure, so nothing ELSE is missing) and `dep` off, and demand a `ValueError` naming
    `dep`. A dependency that has quietly stopped being enforced fails here.
    The same test also builds the closure-satisfied config UNVIOLATED, which is the stronger half:
    it proves the declared `requires` is SUFFICIENT, so an undeclared dependency of that flag shows
    up as a build failure rather than as silence.
  * REVERSE — parse the constructor and collect every `raise ValueError` guarded by a condition that
    mentions two or more registry flags. Each such coupling must be declared in `requires`, or
    listed in `BESPOKE_COUPLINGS` below with a reason it CANNOT be. A new hand-written raise about a
    flag pair therefore fails until someone decides which it is.

WHAT STAYS BESPOKE, and why the carve-out is narrow. `requires` can only say "this flag must be
ENABLED". Two constructor checks are strictly stronger and keep their hand-written form — but note
that only one of them is exempted: `damage_op` still DECLARES `move_belief_mode` (the weaker true
statement), because a weaker truth in the registry is better than a blank. Only
`edge_bias_families`, whose requirement is a function of WHICH family letters are selected, has no
true flag-level statement to make at all.
"""
import ast
import inspect
import os
from typing import Dict, FrozenSet, List, Set, Tuple

import gymnasium as gym
import numpy as np
import pytest

from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.flag_registry import BY_NAME, REGISTRY, is_enabled, requirement_closure
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

# ------------------------------------------------------------------- the ON/OFF values under test
# OFF is derived, never listed: a mode string's off state IS its registry default (`'off'` /
# `'none'`), and a bool/width's is `False`/`0`. ON needs help only where "on" is not `True`.
_ON_OVERRIDE: Dict[str, object] = {
    "move_belief_mode": "revealed",     # the value damage_op's stronger check also accepts
    "value_dist_mode": "read_only",
    "value_dist_bins": 4,
    "opp_belief_cls_k": 2,
    "entity_topk_seats": 2,
    "damage_topk_k": 2,
    "damage_candidate_k": 2,
}


def _off_value(name: str) -> object:
    d = BY_NAME[name].default
    if isinstance(d, str):
        return d if d in ("off", "none") else "off"
    if isinstance(d, bool):
        return False
    if isinstance(d, int):
        return 0
    raise AssertionError(f"{name}: no OFF value convention for default {d!r}")


def _on_value(name: str) -> object:
    v = _ON_OVERRIDE.get(name, True)
    assert is_enabled(v), f"{name}: chosen ON value {v!r} reads as disabled"
    return v


@pytest.fixture(scope="module")
def base_kwargs():
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    return {"observation_space": space, "layout": layout, "mappings": mappings}


# Scalars a flag needs at a legal MAGNITUDE, which `requires` cannot express and deliberately does
# not try to: "enabled" is a switch predicate, and these are value RELATIONS between two numbers.
# This one was found by the positive control below on its first run — `value_dist_mode != 'none'`
# also needs `value_dist_vmax > value_dist_vmin`, enforced inside `ValueDistHead` rather than in the
# constructor, so neither `requires` nor the reverse scan (which reads only `__init__`) can see it.
# Listed here so the positive control tests the flag rather than tripping over its bounds.
_VALUE_RELATIONS: Dict[str, Dict[str, object]] = {
    "value_dist_mode": {"value_dist_vmin": 0.0, "value_dist_vmax": 1.0},
}


def _config_for(flag: str) -> Dict[str, object]:
    """`flag` on plus everything its declared closure says it needs — nothing else."""
    cfg: Dict[str, object] = {flag: _on_value(flag)}
    for dep in requirement_closure(flag):
        cfg[dep] = _on_value(dep)
    for name in list(cfg):
        cfg.update(_VALUE_RELATIONS.get(name, {}))
    return cfg


_PAIRS = [(f.name, dep) for f in REGISTRY for dep in f.requires]
_SUBJECTS = sorted({f.name for f in REGISTRY if f.requires})


@pytest.mark.parametrize("flag", _SUBJECTS)
def test_the_declared_closure_is_sufficient(flag, base_kwargs):
    """POSITIVE control: satisfying only what `requires` names must actually build.

    This is the half that catches an INCOMPLETE declaration. A dependency the constructor enforces
    but the registry never learned about shows up here as the constructor refusing a config the
    registry called complete.
    """
    cfg = _config_for(flag)
    try:
        Gen3FeaturesExtractor(**base_kwargs, **cfg)
    except ValueError as exc:
        pytest.fail(
            f"{flag}'s declared requirement closure is INCOMPLETE — building "
            f"{sorted(cfg)} still raises:\n  {exc}\n"
            f"Add the missing flag to ModelFlag({flag!r}).requires in flag_registry.py.")


@pytest.mark.parametrize("flag,dep", _PAIRS, ids=[f"{f}-needs-{d}" for f, d in _PAIRS])
def test_each_declared_requirement_is_enforced(flag, dep, base_kwargs):
    """NEGATIVE control: removing exactly one declared dependency must be refused."""
    cfg = _config_for(flag)
    cfg[dep] = _off_value(dep)
    with pytest.raises(ValueError) as exc:
        Gen3FeaturesExtractor(**base_kwargs, **cfg)
    assert dep in str(exc.value), (
        f"{flag} with {dep}={cfg[dep]!r} raised, but the message never names {dep}:\n"
        f"  {exc.value}\nA declared dependency must fail with a message the reader can act on.")


# ----------------------------------------------------------- REVERSE: the constructor's own raises
# Locals/attributes that ARE a registry flag under another spelling. Without these the edge-family
# raises read as single-flag checks and would slip past the coupling rule entirely.
_ALIASES = {"edge_bias": "edge_bias_families", "fams": "edge_bias_families",
            "_fams": "edge_bias_families"}

# Constructor raises that couple two registry flags and CANNOT become a `requires` row. Keyed by the
# exact set of flags in the guard, so widening a raise to a third flag re-opens the decision.
BESPOKE_COUPLINGS: Dict[FrozenSet[str], str] = {
    frozenset({"edge_bias_families", "damage_op"}):
        "per-FAMILY: c3/c4/c5/g/x/t/v/d2/d4 each need the op, but `h` needs nothing — so no "
        "flag-level statement about edge_bias_families is true.",
    frozenset({"edge_bias_families", "damage_op", "damage_outgoing"}):
        "per-FAMILY: only d1/s1/c1/c2 need the outgoing block.",
    frozenset({"edge_bias_families", "entity_topk_seats"}):
        "per-FAMILY: d3/s3 ARE the E4 seats' bias rows; the other 15 families are indifferent.",
    frozenset({"edge_bias_families", "history_events"}):
        "per-FAMILY: only `r` (Tier H-C reference edges) rides the H-B event seats.",
}


def _flag_names_in(expr: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Name):
            cand = _ALIASES.get(node.id, node.id)
            if cand in BY_NAME:
                out.add(cand)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "self":
            cand = _ALIASES.get(node.attr, node.attr)
            if cand in BY_NAME:
                out.add(cand)
    return out


def _guarded_raises() -> List[Tuple[int, Set[str], str]]:
    """`(lineno, registry flags in the enclosing conditions, the conditions' source)` per raise."""
    path = inspect.getsourcefile(Gen3FeaturesExtractor)
    tree = ast.parse(open(path).read())
    ctor = next(b for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                and n.name == "Gen3FeaturesExtractor"
                for b in n.body if isinstance(b, ast.FunctionDef) and b.name == "__init__")

    found: List[Tuple[int, Set[str], str]] = []

    def walk(node: ast.AST, guards: List[ast.expr]) -> None:
        if isinstance(node, ast.If):
            for st in node.body:
                walk(st, guards + [node.test])
            for st in node.orelse:
                walk(st, guards)
            return
        if isinstance(node, ast.Raise):
            names: Set[str] = set()
            for g in guards:
                names |= _flag_names_in(g)
            found.append((node.lineno, names, " and ".join(ast.unparse(g) for g in guards)))
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.Lambda)):
                continue          # a nested helper's raises are not ctor-level validation
            walk(child, guards)

    for st in ctor.body:
        walk(st, [])
    return found


def test_the_constructor_declares_no_coupling_the_registry_lacks():
    """Every ctor raise about two-or-more registry flags is declared, or explicitly bespoke."""
    undeclared = []
    for lineno, names, guard in _guarded_raises():
        if len(names) < 2:
            continue              # a value-domain check ("must be one of …") couples nothing
        if frozenset(names) in BESPOKE_COUPLINGS:
            continue
        # Declared iff ONE of the flags names all the others as its dependencies.
        if any(set(names) - {n} <= set(BY_NAME[n].requires) for n in names):
            continue
        undeclared.append(f"  {os.path.basename(inspect.getsourcefile(Gen3FeaturesExtractor))}"
                          f":{lineno} couples {sorted(names)}\n      guard: {guard}")
    assert not undeclared, (
        "constructor raises that couple registry flags with no matching declaration:\n"
        + "\n".join(undeclared)
        + "\nEither add `requires=(...)` to the owning ModelFlag in flag_registry.py, or — if the "
          "dependency is per-VALUE and no flag-level statement is true — add the flag set to "
          "BESPOKE_COUPLINGS in this file with the reason.")


def test_every_bespoke_coupling_still_exists():
    """A stale exemption is as bad as a missing one: it hides the next real coupling behind it."""
    live = {frozenset(names) for _, names, _ in _guarded_raises() if len(names) >= 2}
    stale = sorted(sorted(k) for k in BESPOKE_COUPLINGS if k not in live)
    assert not stale, (
        f"BESPOKE_COUPLINGS entries no longer match any constructor raise: {stale}. The check was "
        "deleted or reshaped — remove the exemption, or (if it became a plain flag dependency) "
        "declare it in flag_registry and drop it from here.")


def test_off_and_on_values_round_trip():
    """The ON/OFF convention this file tests with is the registry's own, not a private one."""
    for f in REGISTRY:
        if not (f.requires or any(f.name in g.requires for g in REGISTRY)):
            continue
        assert not is_enabled(_off_value(f.name)), f"{f.name}: OFF value reads as enabled"
        assert is_enabled(_on_value(f.name)), f"{f.name}: ON value reads as disabled"
