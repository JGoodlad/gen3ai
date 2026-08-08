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


def test_the_op_block_reaches_both_heads_at_full_width(graph):
    """The op's head concat is the only channel that can deliver an absolute MAGNITUDE to the heads
    (a bias is a softmax-normalised ratio). Both heads must read it, unpooled, at the op's full
    out_dim — narrowing it silently would be a capability change, not a refactor."""
    op_dim = graph["meta"]["op_out_dim"]
    for head in FORWARD_SINKS:
        concat = [e for e in graph["edges"]
                  if e["type"] == "concat" and e["src"] == "damage_op" and e["dst"] == head]
        assert len(concat) == 1, f"expected exactly one op->{head} concat edge, got {concat}"
        assert concat[0]["width"] == op_dim
        assert concat[0]["pooled"] is False


def test_absent_op_sub_blocks_do_not_produce_pointer_cells(graph, snapshot):
    """The switch cell width must follow the op's actual toggles. In the production config
    `damage_matrices_outgoing_all` is off, so the OAX attacker row does NOT exist and the switch
    logit gets no offense read — the exact fact a stale doc misreported. If someone turns the
    matrix on, this width moves and the snapshot diff makes it visible."""
    cells = [e for e in graph["edges"]
             if e["type"] == "cell" and e["dst"].startswith("pointer.switch_logit")]
    assert cells
    widths = {e["width"] for e in cells}
    assert len(widths) == 1
    const = cells[0]["source_constant"]
    if "OAX" in const:
        assert widths == {33}, "OAX row present — expected the 15+18 switch cell"
    else:
        assert widths == {15}, (
            "without matrices_outgoing_all the switch cell is the incoming row + CB tail only")
