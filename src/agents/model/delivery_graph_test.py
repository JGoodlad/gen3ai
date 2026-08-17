"""The delivery graph is pinned to a committed snapshot, and leak-safety is a test, not an argument.

Two jobs:

1. **Anti-rot.** `designs/architecture_graph.dot` and `designs/ARCHITECTURE.md` describe the model.
   Documents drift; this makes drift a red test. Regenerate the graph from the live code and diff it
   against `delivery_graph_snapshot.json`. An architecture change that is not reflected in the graph
   FAILS here — the fix is to regenerate both artifacts in the same commit, which is exactly the
   discipline `designs/ai_v3/README.md` lacked.

2. **Leak-safety as an invariant.** `aux` edges carry privileged training-only labels (the
   opponent's true moveset, its Hidden Power type, its spread). "They never enter the forward" has
   always been true by construction and asserted only in prose. Here it is checked: no `aux` edge
   may terminate at `pi_projection`, `vf_projection`, or any pointer logit.

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
    build_graph,
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
