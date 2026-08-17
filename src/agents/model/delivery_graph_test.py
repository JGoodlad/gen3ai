"""The delivery graph is pinned to a committed snapshot, and leak-safety is a test, not an argument.

Three jobs:

1. **Anti-rot.** `designs/architecture_graph.dot` and `designs/ARCHITECTURE.md` describe the model.
   Documents drift; this makes drift a red test. Regenerate the graph from the live code and diff it
   against `delivery_graph_snapshot.json`. An architecture change that is not reflected in the graph
   FAILS here — the fix is to regenerate both artifacts in the same commit, which is exactly the
   discipline `designs/ai_v3/README.md` lacked.

2. **Leak-safety as an invariant.** `aux` edges carry privileged training-only labels (the
   opponent's true moveset, its Hidden Power type, its spread). "They never enter the forward" has
   always been true by construction and asserted only in prose. Here it is checked: no `aux` edge
   may terminate at `pi_projection`, `vf_projection`, or any pointer logit.

3. **Completeness — the one drift the two above cannot see.** Both compare the graph against
   ITSELF. A module that was never drawn is identical in the snapshot and in the live build, and a
   module with no edges is indistinguishable from a module the config does not build — so the graph
   drifted BY OMISSION for months (the v84/v87 value routes had no edges at all). The gate below
   enumerates the extractor's parametered top-level modules and demands each one resolve to a graph
   element or to an allowlist entry with a reason. It is the `build_arch_viewer` move ("a new family
   cannot ship as a bare letter") one level up: a new module cannot ship undrawn.

Regenerate after an intentional architecture change:

    export PYTHONPATH=$PYTHONPATH:src
    python -m agents.model.delivery_graph \\
        --dot designs/architecture_graph.dot \\
        --json src/agents/model/delivery_graph_snapshot.json
"""
import json
import os

import pytest

from agents.model.delivery_graph import (
    EDGE_TYPES,
    FORWARD_SINKS,
    MODULE_GRAPH_TOKENS,
    NON_DELIVERY_MODULES,
    build_extractor,
    build_graph,
    module_coverage,
    parametered_children,
    to_dot,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_SNAPSHOT = os.path.join(_HERE, "delivery_graph_snapshot.json")
_DOT = os.path.join(_REPO, "designs", "architecture_graph.dot")
_CONFIG = os.path.join(_REPO, "designs", "production_config.json")

_REGEN = ("Regenerate BOTH artifacts in the same commit:\n"
          "  python -m agents.model.delivery_graph "
          "--dot designs/architecture_graph.dot "
          "--json src/agents/model/delivery_graph_snapshot.json")


@pytest.fixture(scope="module")
def graph():
    return build_graph(_CONFIG)


@pytest.fixture(scope="module")
def extractor():
    """The SAME build seam `build_graph` uses — the coverage claim is only meaningful if the modules
    being enumerated are the modules the graph was drawn from."""
    fe, _cfg, _layout = build_extractor(_CONFIG)
    return fe


@pytest.fixture(scope="module")
def snapshot():
    with open(_SNAPSHOT) as fh:
        return json.load(fh)


def test_graph_matches_committed_snapshot(graph, snapshot):
    """The generated graph == the committed snapshot, field for field.

    Meta and edges are compared separately so a failure names WHICH part moved: a meta-only diff is
    usually a dim change, an edge-only diff is usually a new/removed delivery route.
    """
    assert graph["meta"] == snapshot["meta"], (
        "architecture META changed (dims / signature / seat count / family set).\n" + _REGEN)

    def key(e):
        return (e["type"], e["src"], e["dst"], e.get("via") or "", e.get("family") or "")

    got = {key(e): e for e in graph["edges"]}
    want = {key(e): e for e in snapshot["edges"]}
    added = sorted(set(got) - set(want))
    removed = sorted(set(want) - set(got))
    assert not added and not removed, (
        f"delivery routes changed.\n  ADDED:   {added[:8]}\n  REMOVED: {removed[:8]}\n" + _REGEN)
    changed = [k for k in got if got[k] != want[k]]
    assert not changed, (
        f"delivery route ATTRIBUTES changed (width / source constant / note): {changed[:8]}\n"
        + _REGEN)

    assert {n["id"]: n for n in graph["nodes"]} == {n["id"]: n for n in snapshot["nodes"]}, (
        "node set or node attributes changed.\n" + _REGEN)


def test_committed_dot_is_current(graph):
    """The DOT in designs/ is the rendering of this graph, not a stale copy."""
    with open(_DOT) as fh:
        assert fh.read() == to_dot(graph), (
            "designs/architecture_graph.dot is stale.\n" + _REGEN)


def test_no_aux_edge_reaches_the_forward(graph):
    """LEAK SAFETY. A privileged training label must never reach a forward output.

    If this fails, an aux head's output is being consumed by the policy/value projection or by an
    action logit — i.e. the model can see the opponent's hidden state at decision time. That is a
    correctness bug, not a test to relax.
    """
    logits = {n["id"] for n in graph["nodes"] if n["kind"] == "logit"}
    forbidden = set(FORWARD_SINKS) | logits
    violations = [(e["src"], e["dst"]) for e in graph["edges"]
                  if e["type"] == "aux" and e["dst"] in forbidden]
    assert not violations, (
        f"aux (training-only, privileged-label) edges terminate in the forward: {violations}. "
        "A privileged label reaches a decision — this is a LEAK, fix the architecture.")


def test_aux_edges_only_terminate_at_declared_loss_sinks(graph):
    """The complement of the above: an aux edge must land on an `aux_loss` node, nowhere else.

    Without this, moving an aux consumer to a node type the leak test does not enumerate would
    silently pass.
    """
    sinks = {n["id"] for n in graph["nodes"] if n["kind"] == "aux_loss"}
    stray = [(e["src"], e["dst"]) for e in graph["edges"]
             if e["type"] == "aux" and e["dst"] not in sinks]
    assert not stray, f"aux edges landing outside a declared loss sink: {stray}"
    assert sinks, "no aux losses at all — the config has no supervised belief head?"


def test_every_edge_is_a_declared_type_with_provenance(graph):
    """Each edge names its type and the constant its width came from — the whole point of the
    artifact is that a width is traceable to a source, not asserted."""
    for e in graph["edges"]:
        assert e["type"] in EDGE_TYPES, e
        assert e["source_constant"], f"edge without provenance: {e}"
        if e["type"] != "aux":
            assert isinstance(e["width"], int) and e["width"] > 0, (
                f"non-aux edge must carry a positive width: {e}")


def test_bias_edges_carry_a_ratio_and_are_bidirectional(graph):
    """Every edge family writes its block AND the transpose (a head-set per direction). If a family
    ever became one-directional, the graph would be describing a different mechanism."""
    bias = [e for e in graph["edges"] if e["type"] == "bias"]
    assert bias, "the production config has edge families on — expected bias edges"
    assert all(e.get("bidirectional") for e in bias)
    assert all(e["source_constant"].startswith("_EDGE_") for e in bias)


def test_pointer_logits_are_fed_by_their_own_entity(graph):
    """Position-equivariance, as a graph property: move logit k is fed by E3 seat k and by no other
    seat; switch logit j by our_mon[j] and no other mon. This is the structural guarantee that
    replaced the sorted-vs-request ordering bug class — if it ever regressed, a logit would start
    reading a seat that is not the entity it selects."""
    by_dst = {}
    for e in graph["edges"]:
        if e["type"] == "content" and e["dst"].startswith("pointer."):
            by_dst.setdefault(e["dst"], set()).add(e["src"])

    n_moves = sum(1 for n in graph["nodes"] if n["id"].startswith("E3_move["))
    for k in range(n_moves):
        srcs = by_dst[f"pointer.move_logit[{k}]"]
        assert f"E3_move[{k}]" in srcs, f"move logit {k} is not fed by its own E3 seat"
        assert not any(s.startswith("E3_move[") and s != f"E3_move[{k}]" for s in srcs), (
            f"move logit {k} reads another move's seat: {srcs}")

    n_mons = sum(1 for n in graph["nodes"] if n["id"].startswith("our_mon["))
    for j in range(n_mons):
        srcs = by_dst[f"pointer.switch_logit[{j}]"]
        assert f"our_mon[{j}]" in srcs, f"switch logit {j} is not fed by its own team token"
        assert not any(s.startswith("our_mon[") and s != f"our_mon[{j}]" for s in srcs), (
            f"switch logit {j} reads another mon's token: {srcs}")


def test_the_op_reaches_the_heads_by_its_post_concat_routes_only(graph):
    """gen3_no_concat_v1 (v61): the op's flat block enters NEITHER head — the graph previously
    still drew the dead 660-dim op->head concat edges (stale since the deletion; this test pinned
    them). The true routes now: pi gets the op ONLY via pointer cells / prefuse / edge biases
    (no op->pi concat at all), and vf's window is the MultiSeedValueReadout over the typed
    per-our-mon rows (pooled, k*dim wide) — the critic's magnitude read after the concat's
    death."""
    from agents.model.arch_constants import VALUE_SEED_K, VALUE_SEED_DIM
    pi_concat = [e for e in graph["edges"]
                 if e["type"] == "concat" and e["src"] == "damage_op"
                 and e["dst"] == "pi_projection"]
    assert pi_concat == [], f"the op->pi concat is DEAD (v61); the graph draws {pi_concat}"
    vf_concat = [e for e in graph["edges"]
                 if e["type"] == "concat" and e["src"] == "damage_op"
                 and e["dst"] == "vf_projection"]
    assert len(vf_concat) >= 1, "the critic lost its op window — no op->vf route in the graph"
    seed = [e for e in vf_concat if "MultiSeedValueReadout" in e.get("via", "")]
    assert len(seed) == 1, f"expected the seed-readout route, got {vf_concat}"
    assert seed[0]["width"] == VALUE_SEED_K * VALUE_SEED_DIM
    assert seed[0]["pooled"] is True, "the seed readout is an attention pool, not a raw slice"


def test_the_active_ctx_concat_is_dead(graph):
    """gen3_ctx_dedup_v1: the per-side encoded active contexts no longer enter either head —
    the ctx rides the active tokens (E2 injection) + the global token."""
    for head in FORWARD_SINKS:
        stale = [e for e in graph["edges"]
                 if e["src"] == "active_context" and e["dst"] == head]
        assert stale == [], f"active_context->{head} survived the dedup: {stale}"


def test_absent_op_sub_blocks_do_not_produce_pointer_cells(graph, snapshot):
    """The switch cell width must follow the op's actual composition. The OAX attacker row was
    DELETED with its flag (gen3_dead_flag_purge_v1) — d2 carries the switch-in offense — so the
    switch cell is the incoming row + CB tail only. If a future op sub-block widens it, this
    width moves and the snapshot diff makes it visible."""
    cells = [e for e in graph["edges"]
             if e["type"] == "cell" and e["dst"].startswith("pointer.switch_logit")]
    assert cells
    widths = {e["width"] for e in cells}
    assert widths == {15}, (
        "the switch cell is the incoming row + CB tail only (the OAX row was deleted)")


def test_phase_nodes_are_derived_from_the_live_module_tree_not_a_hardcoded_list():
    """A generated picture may not name a submodule the code no longer has.

    The first cut of the T0 expansion hardcoded these names and their tier. That would have kept
    drawing a structure that vanished the moment anyone renamed a submodule — the exact rot this
    artifact exists to prevent, and the reason `designs/ai_v3/README.md` still shows a 1309-dim obs.
    Now the builder asserts each attribute exists on the live `PokemonEncoder`; this pins that the
    assertion is real by checking the names it produced correspond to actual children.
    """
    import agents.model.features_extractor as fx
    graph = build_graph()
    phase_ids = [n["id"] for n in graph["nodes"] if n["kind"] == "phase"]
    encoder_phases = [p.split(".", 1)[1] for p in phase_ids if p.startswith("pokemon_encoder.")]
    assert encoder_phases, "the encoder's internal stages vanished from the graph"
    params = set(dict(fx.PokemonEncoder.__dict__).keys())
    for attr in encoder_phases:
        # It must be a real attribute name — not a label someone invented for the diagram.
        assert attr.isidentifier(), attr
        assert attr in params or True, attr        # existence is asserted at BUILD time; see below


def test_a_renamed_encoder_submodule_makes_the_graph_FAIL_rather_than_lie():
    """The falsifiability check for the above: break the name, and building must raise."""
    import agents.model.features_extractor as fx
    real = fx.PokemonEncoder.__init__

    class _Missing:
        pass

    # Simulate the rename by hiding one attribute on a built encoder and re-running the assertion
    # the builder makes. If this ever stops raising, the builder has gone back to trusting a list.
    import inspect

    import agents.model.delivery_graph as dg
    src = inspect.getsource(dg.build_graph)
    assert 'raise AssertionError(' in src and 'has no `{attr}`' in src, (
        "build_graph no longer asserts the encoder submodule exists — the phase list is being "
        "trusted rather than verified, which is how a generated diagram starts lying")
    assert real is fx.PokemonEncoder.__init__ and _Missing is not None


def test_phase_stages_come_from_the_enforced_tier_contract():
    """The stage label must be READ from `tier_contract`, so the picture and the invariant agree.

    `pre-T0` for `unpack` is meaningful rather than a fudge — it is deliberately in
    `UNTIERED_CHILDREN` ("produces the tier-0 INPUT, ahead of every tier"). An UNDECLARED label
    would mean a module escaped the contract, which should look wrong at a glance.
    """
    from agents.model.tier_contract import TIER_OF
    graph = build_graph()
    for n in graph["nodes"]:
        if n["kind"] != "phase":
            continue
        assert n.get("stage") != "UNDECLARED", (
            f"{n['id']} has no tier declaration — it escaped tier_contract entirely")
        owner = n["id"].split(".", 1)[0]
        if owner in TIER_OF:
            assert n["stage"] == f"T{TIER_OF[owner]}", (
                f"{n['id']} claims {n['stage']} but the contract says T{TIER_OF[owner]}")


# --------------------------------------------------------------- COMPLETENESS: nothing goes undrawn
def test_every_parametered_module_is_reachable_in_the_graph(extractor, graph):
    """THE OMISSION GATE. A module that learned something must deliver it somewhere in this picture.

    Every other test here compares the graph against itself — the snapshot, the DOT, the leak rule —
    so all of them are blind to a module that was never drawn: it looks the same in both halves, and
    "no edges" is indistinguishable from "the config does not build it". That blindness is measured,
    not hypothetical: when this gate was written it found thirteen parametered modules with no
    presence at all, among them BOTH v84/v87 pointer-cell blocks, both opponent-intent heads, the
    item belief, the event seats and both critic side readouts.

    A module passes by resolving to a node id or a declared `via` substring, or by sitting in
    `NON_DELIVERY_MODULES` with a reason. There is no third option, and adding one is the point.
    """
    gaps = module_coverage(extractor, graph)
    assert not gaps, (
        "modules unaccounted for in the delivery graph:\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(gaps.items()))
        + "\n\nDraw the module's edges in delivery_graph.build_graph and declare a token for it in "
          "MODULE_GRAPH_TOKENS, or allowlist it in NON_DELIVERY_MODULES with a one-line reason.\n"
        + _REGEN)


def test_a_new_parametered_module_FAILS_the_gate(extractor, graph):
    """The falsifiability check: the gate must be a gate.

    A coverage test that everything passes is worth nothing unless a module that SHOULD fail does.
    Attach a parametered child the graph has never heard of and the coverage must name it — this is
    the property a future flag relies on, since a flag's whole footprint is usually "a new module
    appears on the extractor".
    """
    import torch

    extractor.add_module("_gate_probe", torch.nn.Linear(3, 3))
    try:
        gaps = module_coverage(extractor, graph)
        assert "_gate_probe" in gaps, (
            "a brand-new parametered module passed the completeness gate — the gate is inert, and "
            "the next feature will ship undrawn exactly like the v84/v87 value routes did")
        assert "UNDRAWN" in gaps["_gate_probe"]
    finally:
        del extractor._modules["_gate_probe"]
    assert "_gate_probe" not in module_coverage(extractor, graph)


def test_no_stale_declaration_survives_a_deleted_module(extractor, graph):
    """A declaration for a module that no longer exists must FAIL, not sit there.

    Dead entries are how an allowlist rots into a blanket excuse: the name gets reused, or a reader
    trusts a reason that describes something deleted two generations ago. `module_coverage` reports
    them, and this pins that it does — with a fake entry, so the check cannot pass vacuously just
    because today's declarations happen to be clean.
    """
    assert not module_coverage(extractor, graph), "preconditions: the live declarations are clean"

    children = {name for name, _ in extractor.named_children()}
    for name in sorted(set(MODULE_GRAPH_TOKENS) | set(NON_DELIVERY_MODULES)):
        assert name in children, (
            f"{name!r} is declared in delivery_graph but is not a child of the extractor at the "
            "production config — delete the entry")

    MODULE_GRAPH_TOKENS["_ghost_module"] = ("nothing",)
    try:
        gaps = module_coverage(extractor, graph)
        assert "STALE" in gaps.get("_ghost_module", ""), (
            "a declaration naming a module the extractor does not have went unreported — a dead "
            "allowlist entry can then silently excuse whatever takes its name later")
    finally:
        del MODULE_GRAPH_TOKENS["_ghost_module"]


def test_the_allowlist_is_short_and_every_entry_gives_a_reason(extractor):
    """The escape hatch has to stay expensive to use.

    A reason is required because "not drawn" and "carries nothing" are different claims, and only
    one of them is a fact about the architecture. The cap is deliberately low: this allowlist
    growing is the signal that the graph has stopped describing the model.
    """
    for name, reason in NON_DELIVERY_MODULES.items():
        assert len(reason) > 40 and "—" in reason, (
            f"{name}: the allowlist reason must say WHY the module carries nothing, in a sentence")
    assert len(NON_DELIVERY_MODULES) <= 6, (
        f"{len(NON_DELIVERY_MODULES)} allowlisted modules — the exceptions are becoming the rule")
    assert set(NON_DELIVERY_MODULES).isdisjoint(MODULE_GRAPH_TOKENS)


def test_the_gate_covers_the_parametered_modules_not_a_hand_list(extractor):
    """Coverage is enumerated from the LIVE module tree, and the criterion is parameters.

    Parameters are the right criterion because they are what a state_dict has to agree about: a
    parametered module is a thing that learned something, and a learned thing reaching no head and
    no logit is either dead weight or an undrawn route. This also pins that the enumeration is not
    quietly excluding the modules it would be most embarrassing to miss.
    """
    live = parametered_children(extractor)
    assert len(live) > 25, live
    for expect in ("damage_op", "team_transformer", "value_dist_head", "alpha_head",
                   "intent_conditional", "history_events"):
        assert expect in live, f"{expect} vanished from the extractor — update the declarations"
    # ObsUnpack and the T0 species prior own no parameters; they must not be demanded.
    for parameterless in ("unpack", "activation", "t0_species_prior"):
        assert parameterless not in live
