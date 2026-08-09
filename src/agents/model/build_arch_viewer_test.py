"""Tests for the generated architecture viewer.

The viewer's whole justification is that it CANNOT drift from the code (see
`build_arch_viewer.__doc__`). These assert the properties that justification rests on: the payload
is the graph (not a copy of it), it is deterministic, it is self-contained, and the two claims the
artifact makes in its own UI — leak-safety and contamination labelling — are computed from data
rather than asserted.
"""
from __future__ import annotations

import json
import os
import re

import pytest

from agents.model import build_arch_viewer as B

_HTML_PATH = B._DEFAULT_OUT


@pytest.fixture(scope="module")
def payload():
    return B.build_payload(B._load_graph(None))


@pytest.fixture(scope="module")
def rendered(payload):
    return B.render(payload)


def _embedded(html: str) -> dict:
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert m, "the embedded payload block is missing"
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_payload_is_the_graph_not_a_copy(payload):
    """Every node and edge of the snapshot survives into the viewer, with endpoints resolving."""
    with open(B._SNAPSHOT) as fh:
        graph = json.load(fh)
    assert len(payload["nodes"]) == len(graph["nodes"])
    assert len(payload["edges"]) == len(graph["edges"])
    assert {n["id"] for n in payload["nodes"]} == {n["id"] for n in graph["nodes"]}
    ids = {n["id"] for n in payload["nodes"]}
    for e in payload["edges"]:
        assert e["src"] in ids and e["dst"] in ids, e


def test_every_edge_type_has_a_semantic_class(payload):
    """A new edge type must be classified deliberately, not defaulted into a hue."""
    for e in payload["edges"]:
        assert e["cls"] in payload["class_order"], e
        assert e["carries"], e


def test_leak_safety_is_computed_not_asserted(payload):
    """No aux edge may terminate at a forward sink — and the badge must read it off the data."""
    violations = [e["id"] for e in payload["edges"]
                  if e["type"] == "aux" and (e["dst"] in B.FORWARD_SINKS
                                             or e["dst"].startswith("pointer."))]
    assert payload["leak_safety"]["violations"] == violations
    assert payload["leak_safety"]["ok"] is (not violations)
    assert payload["leak_safety"]["ok"], "an aux edge reaches the forward — that is a real leak"


def test_contamination_is_keyed_on_generation_not_audit_date(payload):
    """The speed-stat GIGO is a property of the TRAINED model.

    gen-2's audit is dated on the fix date (2026-08-06) while gen-2 trained entirely against the
    buggy feature, so a date-based rule reports its `v` row as trustworthy. This pins the
    generation-based rule that replaced it.
    """
    by_gen = {o["generation"]: o for o in payload["overlays"]}
    for gen in B._GIGO_GENERATIONS:
        if gen in by_gen:
            assert by_gen[gen]["contaminated"] == ["v", "c1"], gen
            assert by_gen[gen]["contamination_note"]
    for gen, o in by_gen.items():
        if gen not in B._GIGO_GENERATIONS:
            assert o["contaminated"] == [], gen


def test_overlays_merge_the_two_producers(payload):
    """Edge-family rows and op-block rows for one checkpoint must land on ONE selector entry.

    Kept separate, choosing the op-block file would leave every bias edge unmeasured with no hint
    that the family rows exist.
    """
    gen3 = [o for o in payload["overlays"] if o["generation"] == "gen-3"]
    assert gen3, "no gen-3 overlay"
    o = gen3[0]
    assert o["families"], "family rows missing after the merge"
    assert o["blocks"], "op-block rows missing after the merge"
    assert len(o["sources"]) > 1, "the merge did not actually combine files"
    assert o["sources"] == sorted(o["sources"])


def test_render_is_deterministic(payload):
    assert B.render(payload) == B.render(payload)


def test_rendered_is_self_contained(rendered):
    """file:// means no fetch of local files; only the documented CDN tags may reach the network."""
    assert "fetch(" not in rendered and "XMLHttpRequest" not in rendered
    urls = set(re.findall(r'https?://[^"\')\s]+', rendered))
    assert urls, "expected the documented CDN script tags"
    assert all("cdn.jsdelivr.net" in u for u in urls), urls


def test_payload_survives_html_embedding(rendered, payload):
    """`</script>` inside the JSON would truncate the block; the escape must round-trip."""
    assert "</script>" not in rendered.split('id="payload"')[1].split("</script>")[0]
    assert _embedded(rendered)["leak_safety"] == payload["leak_safety"]


def test_committed_artifact_is_current(rendered):
    """The .html is a build artifact; if it is stale the viewer is lying about the architecture."""
    if not os.path.exists(_HTML_PATH):
        pytest.skip("viewer not built in this tree")
    with open(_HTML_PATH) as fh:
        assert fh.read() == rendered, (
            "designs/architecture_viewer.html is stale — regenerate with "
            "`python -m agents.model.build_arch_viewer`")


def test_every_bias_family_is_expanded_not_left_as_a_code(payload):
    """`d2` / `c1` / `s3` must never reach the screen as a bare code.

    Half the value of the artifact is being readable aloud without going and looking a code up,
    so a family added to the extractor has to arrive with a label. The cell definition is parsed
    rather than restated, and is checked against the graph's own width so the two cannot drift.
    """
    families = payload["families"]
    used = {e["family"] for e in payload["edges"] if e["type"] == "bias"}
    missing = sorted(used - set(families))
    assert not missing, f"bias families with no label: {missing} — add them to FAMILY_LABEL"
    for fam in sorted(used):
        meta = families[fam]
        assert meta["label"], fam
        assert meta["cells"], f"{fam}: no cell contents parsed from features_extractor.py"
        widths = {e["width"] for e in payload["edges"]
                  if e["type"] == "bias" and e["family"] == fam}
        assert widths == {meta["width"]}, (
            f"{fam}: graph says width {widths}, `_EDGE_{fam.upper()}_CELL` says {meta['width']}")


def test_only_the_cytoscape_tag_is_fetched(rendered):
    """Layout is hand-placed, so no layout plugin may creep back in.

    A ranker is the wrong tool on this graph — 388 of 487 edges are seat->seat attention bias,
    i.e. peers, and ranking them produced a diagonal smear with sub-pixel labels. Re-adding
    dagre would be the first step back to that, and it costs two more network dependencies on an
    artifact whose whole point is that it opens from `file://`.
    """
    tags = re.findall(r'<script src="([^"]+)"', rendered)
    assert len(tags) == 1, tags
    assert "cytoscape" in tags[0] and "dagre" not in tags[0], tags


def test_every_node_kind_and_seat_group_has_a_layout_band(rendered, payload):
    """Every node kind must be named by the template's `bandOf()` / `SEAT_BANDS`.

    Unplaced nodes are not lost — they go to a MISC band the header calls out — but a kind added
    to `delivery_graph` still deserves a deliberate column rather than the fallback, and the
    warning is only visible to someone who happens to open the page.
    """
    script = rendered.rsplit("<script>", 1)[1]
    for kind in sorted({n["kind"] for n in payload["nodes"]}):
        assert f"'{kind}'" in script, (
            f"node kind {kind!r} is not placed by bandOf() — it lands in the MISC band")
    for group in sorted({re.sub(r"\[\d+\]$", "", n["id"])
                         for n in payload["nodes"] if n["kind"] == "seat"}):
        assert f"'{group}'" in script, f"seat group {group!r} is missing from SEAT_BANDS"


def test_doc_links_point_at_real_headings(payload):
    """A wrong deep link is worse than none: every mapped section must exist in ARCHITECTURE.md."""
    path = os.path.join(B._REPO, "designs", "ARCHITECTURE.md")
    if not os.path.exists(path):
        pytest.skip("ARCHITECTURE.md not present")
    with open(path) as fh:
        doc = fh.read()
    for section in {n["doc_section"] for n in payload["nodes"] if n.get("doc_section")}:
        assert section in doc, f"ARCHITECTURE.md has no heading {section!r}"


def test_concept_links_point_at_real_headings(payload):
    for e in payload["edges"]:
        path = os.path.join(B._REPO, e["concept_file"])
        assert os.path.exists(path), e["concept_file"]
        with open(path) as fh:
            assert e["concept_heading"] in fh.read(), e["concept_heading"]
