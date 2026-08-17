"""GENERATED interactive architecture viewer — the delivery digraph you can actually interrogate.

WHY THIS EXISTS. `delivery_graph.py` already answers "who delivers what, to whom, through which
physical channel" as JSON + DOT. DOT is fine for a printout and useless for the two questions that
actually come up in a design conversation:

  1. "What does the critic see?" / "What does the switch logit consume?"  — a PATH query.
  2. "Which of these routes does the policy actually LEAN on?"            — a MEASUREMENT overlay.

A static rendering of 36 seats and 487 edges cannot answer either; it is a hairball. This module
emits a single self-contained HTML file where both are one interaction: pick a node and see only
the subgraph that reaches it; pick a checkpoint and see edge width scale with measured dependence.

WHY GENERATED, NOT DRAWN. `designs/ai_v3/README.md` is the in-repo proof that hand-drawn
architecture diagrams rot — it still shows a 1309-dim obs. Everything here is derived from
`delivery_graph.py`'s output and from `designs/research_state/measurements/*.json`, so the viewer
cannot disagree with the code without `delivery_graph_test.py` failing first.

WHY THE DATA IS EMBEDDED. The artifact is opened with `file://` (no server, no build step).
`fetch()` of a sibling JSON is blocked by the file:// origin policy, so the graph, the overlays and
the deep-links are inlined as JSON literals. The only network dependency is the single CDN <script>
tag for cytoscape; with no network the page still loads and reports the failure in place rather
than rendering a blank canvas, and `--vendor` inlines that bundle too for a copy that needs no
network at all (a separate output — the committed artifact stays CDN-linked so `--check` has one
thing to compare against). Layout is hand-placed rather than ranked, so no
layout plugin is fetched — see the layout block in the template for why a ranker is the wrong tool
on a graph that is 80% peer edges.

THE ENCODING IS THE ARGUMENT. Edge HUE carries the semantic class, because that is the claim the
architecture rests on:

    RATIO     (bias)                     an attention bias is softmax-normalised. It can move
                                         *who attends to whom*; it cannot transmit "53% of max HP".
    ABSOLUTE  (content / concat / cell)  token content, the head concat, and per-action pointer
                                         cells are the only channels that carry a magnitude.
    NONE      (aux)                      training-only supervision. Must never reach a forward sink.

Three hues, not five, is a deliberate consequence of validating the palette rather than eyeballing
it: a node-link diagram is an ALL-PAIRS colour problem (any two edges can cross anywhere), and at
five slots the normal-vision floor FAILs (magenta<->orange dE 12.9 < 15). Cutting to the three
semantic classes passes every check in both modes, and the three ABSOLUTE sub-channels are then
separated by line style — which is a better artifact anyway, because the hue now encodes the thesis.
Palette provenance is in `PALETTE_PROVENANCE` below; re-run the validator if you change it.

LAYOUT. The Python here builds the payload and renders; it carries no markup. The front end is
three real files in `arch_viewer_assets/` (`viewer.html` is the shell, with `__CSS__`, `__JS__`
and `__PAYLOAD__` placeholders), and the long-lived server is `arch_viewer_serve.py`. Splitting
those out is what makes `node --check viewer.js` a test rather than a thing someone remembers.

Usage:
    python -m agents.model.build_arch_viewer                       # -> designs/architecture_viewer.html
    python -m agents.model.build_arch_viewer --out /tmp/v.html
    python -m agents.model.build_arch_viewer --config <run>/model_config.json   # regenerate live
    python -m agents.model.build_arch_viewer --check                # exit 1 if the artifact is stale
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_SNAPSHOT = os.path.join(_HERE, "delivery_graph_snapshot.json")
_MEASUREMENTS = os.path.join(_REPO, "designs", "research_state", "measurements")
_DEFAULT_OUT = os.path.join(_REPO, "designs", "architecture_viewer.html")

# --------------------------------------------------------------------------------------------
# Palette. Validated with the dataviz skill's validator, --pairs all (the honest pairlist for a
# node-link diagram), both modes. Do not hand-edit without re-running it.
# --------------------------------------------------------------------------------------------
PALETTE_PROVENANCE = {
    "validator": "dataviz/scripts/validate_palette.js --pairs all",
    "light": {
        "slots": ["#2a78d6", "#eb6834", "#1baf7a"],
        "result": "ALL CHECKS PASS (worst CVD dE 9.2 deutan, worst normal dE 24.0); "
                  "WARN contrast #1baf7a 2.74 < 3:1 -> relief satisfied by the always-on legend "
                  "and the table view",
    },
    "dark": {
        "slots": ["#3987e5", "#d95926", "#199e70"],
        "result": "ALL CHECKS PASS, no warnings (worst CVD dE 9.4 deutan, worst normal dE 20.9)",
    },
    "rejected": [
        "5 slots (blue/orange/aqua/yellow/magenta) light --pairs all: FAIL normal-vision floor "
        "#e87ba4<->#eb6834 dE 12.9 < 15",
        "blue/orange/violet dark: FAIL CVD dE 1.9 protan and normal dE 9.8 vs #3987e5",
        "blue/orange/green light: FAIL CVD dE 3.2 protan vs #eb6834",
    ],
}

# Semantic class per edge type. Hue is assigned to the CLASS (fixed order, never cycled); the three
# ABSOLUTE sub-channels are separated by line style, which is legitimate secondary encoding and
# happens to state the architecture's own claim.
EDGE_CLASS = {
    "content": "absolute",
    "concat": "absolute",
    "cell": "absolute",
    "bias": "ratio",
    "aux": "none",
}
CLASS_ORDER = ["absolute", "ratio", "none"]      # fixed categorical order
CLASS_CARRIES = {
    "absolute": "an ABSOLUTE (a real magnitude, e.g. 53% of max HP)",
    "ratio": "a RATIO only (softmax-normalised attention weight)",
    "none": "nothing in the forward (training-only supervision)",
}
EDGE_DASH = {                                     # secondary encoding, within class
    "content": None,                              # solid  - lands in the residual stream
    "concat": [10, 5],                            # dashed - lands at the head input
    "cell": [2, 3, 9, 3],                         # dash-dot - lands per-action at the logit
    "bias": [6, 4],                               # dashed - a modulation, not a payload
    "aux": [1, 4],                                # dotted - never in the forward
}
EDGE_TYPE_BLURB = {
    "content": "token content — written into the residual stream; any attender reads it linearly",
    "concat": "head concat — unpooled, straight into the projection input",
    "cell": "pointer cell — per-action, read affinely at one logit",
    "bias": "attention edge bias — steers who attends to whom; carries no magnitude",
    "aux": "training-only supervision — privileged label, never in the forward",
}
# Node kind -> shape. Deliberately NOT hue: a second categorical colour system competing with the
# edge classes is a known failure mode, so kind is carried by shape + border only.
NODE_SHAPE = {
    "seat": "round-rectangle",
    "operator": "hexagon",
    "belief_head": "diamond",
    "head": "round-rectangle",
    "logit": "ellipse",
    "input": "cut-rectangle",
    "aux_loss": "octagon",
}
FORWARD_SINKS = ("pi_projection", "vf_projection")

# --------------------------------------------------------------------------------------------
# Bias families. `d2` / `c1` / `s3` are opaque two-character codes, and an artifact whose job is to
# be read aloud in a design conversation must never send you off to look one up — so the viewer
# expands every one of them, everywhere the code appears.
#
# The split is deliberate: the CELL CONTENTS are PARSED out of `team_transformer.py` (those lines
# ARE the definition, and `edge_bias_test.py` pins them against the operator's own methods), so a
# cell-content change reaches the viewer without anyone remembering to update it. Only the one-line
# "what is this family FOR" phrase is curated here, because the code states that as a variable name
# and nothing else. `build_arch_viewer_test.py` fails if a family in the graph has no entry, so a
# new family cannot ship as a bare letter.
# --------------------------------------------------------------------------------------------
_EXTRACTOR = os.path.join(_HERE, "team_transformer.py")   # the _EDGE_*_CELL lines moved here 2026-08-16
FAMILY_LABEL = {
    "h": "pair-history tendencies (switch-ins / clicks into this mon, exposure, recency)",
    "d1": "our active's move vs each opp mon",
    "d2": "our bench's offense vs their ACTIVE",
    "d3": "their believed move vs each of our mons",
    "d4": "the opp BENCH's believed threat",
    "s1": "will our status move land on that mon",
    "s3": "will their believed status move land on us",
    "v": "speed order, our mon vs opp mon",
    "t": "trapping probability, both directions",
    "x": "switch-in cost — entry chip, pursuit, grounded",
    "g": "end-of-turn residuals — leftovers, weather, status tick, leech",
    "c1": "post-setup consequence deltas, offensive AND defensive",
    "c2": "what LANDING a status move would do",
    "c3": "does healing beat their KO",
    "c4": "the turn a successful Protect banks",
    "c5": "Baton-Pass receiver axis",
}


def _family_cells() -> Dict[str, Dict[str, Any]]:
    """Parse `_EDGE_<FAM>_CELL = <width>  # [cells] per (row, col) - note` from the extractor."""
    with open(_EXTRACTOR) as fh:
        lines = fh.read().splitlines()
    out: Dict[str, Dict[str, Any]] = {}
    fam = None
    for line in lines:
        head = re.match(r"^_EDGE_(\w+)_CELL\s*=\s*(\d+)\s*#\s?(.*)$", line)
        if head:
            fam = head.group(1).lower()
            out[fam] = {"width": int(head.group(2)), "comment": head.group(3).strip()}
            continue
        cont = re.match(r"^\s+#\s?(.*)$", line)                 # wrapped continuation comment
        if fam and cont:
            out[fam]["comment"] = (out[fam]["comment"] + " " + cont.group(1).strip()).strip()
            continue
        fam = None
    for meta in out.values():
        c = meta.pop("comment")
        cells = re.search(r"\[(.*?)\]", c)
        per = re.search(r"per \(([^)]*)\)", c)
        note = re.search(r"\u2014\s*(.*)$", c)
        meta["cells"] = cells.group(1) if cells else None
        meta["per"] = per.group(1) if per else None
        meta["note"] = note.group(1).strip() if note else None
    return out


def _families() -> Dict[str, Dict[str, Any]]:
    """One record per family: the curated label plus the parsed cell definition."""
    cells = _family_cells()
    return {fam: {"label": label, **cells.get(fam, {})} for fam, label in FAMILY_LABEL.items()}

# Generations whose ENTIRE training predates the 2026-08-06 speed-stat fix (v58). Contamination
# is a property of the trained model, not of the audit date — see `_load_overlays`.
_GIGO_GENERATIONS = ("gen-1", "gen-2")

# ARCHITECTURE.md section per node kind / id prefix. Small and verified by `--check`; a wrong link
# is worse than none, so anything unmapped stays null.
DOC_SECTION = {
    "damage_op": "4. The `DamageOperator` output block",
    "pi_projection": "3.1 / 3.2 The head inputs — GENERATED",
    "vf_projection": "3.1 / 3.2 The head inputs — GENERATED",
    "move_belief": "2.1 Order of operations",
    "hp_type_belief": "2.1 Order of operations",
}
DOC_SECTION_PREFIX = [
    ("pointer.", "3.3 The action head is the pointer head — there is no flat `action_net`"),
    ("E3_move", "2.3 The 36-token sequence"),
    ("E4_threat", "2.3 The 36-token sequence"),
    ("E5_tail", "2.3 The 36-token sequence"),
    ("our_mon", "2.3 The 36-token sequence"),
    ("opp_mon", "2.3 The 36-token sequence"),
    ("history", "1.6 Turn history — 7 slots × 159 dims"),
    ("loss.", "7. Training-only obs keys — the leak-safety list"),
]
CONCEPT_NOTE = {
    "ratio": ("designs/learning/entity_tokens_biases_pointers.md",
              "What each route can physically CARRY — routing vs magnitude"),
    "absolute": ("designs/learning/entity_tokens_biases_pointers.md",
                 "How to actually DELIVER a magnitude (the three questions)"),
    "none": ("designs/learning/entity_tokens_biases_pointers.md",
             "The sorting rule"),
}


def _load_graph(config: Optional[str]) -> Dict[str, Any]:
    """The committed snapshot by default; regenerate live when a config is named."""
    if config:
        from agents.model import delivery_graph
        return delivery_graph.build_graph(config)                       # type: ignore[attr-defined]
    with open(_SNAPSHOT) as fh:
        return json.load(fh)


def _norm_metrics(d: Dict[str, Any]) -> Dict[str, float]:
    """Pull the four comparable metrics out of whatever shape a probe wrote."""
    out = {}
    for k in ("kl_mean", "kl_p95", "flip_rate", "dv_mean"):
        v = d.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _load_overlays() -> List[Dict[str, Any]]:
    """Normalise every measurement file into one overlay-per-checkpoint structure.

    Two producers write different shapes and they are NOT interchangeable:
      * edge_ablation_audit.py -> {"families": {fam: metrics}}  incl. the 'all'/'concat'/
        'concat_cells' arms. Arm semantics: zero the family's map (its identity-at-init state).
      * incoming_conditional_probe.py -> {"blocks": {block: {"width", "all"/"threat"/"shuf_all"/
        "shuf_threat": metrics}}}. Here the ZERO arm and the SHUFFLE arm answer different
        questions — zeroing a 522-dim block is a far larger perturbation than shuffling it across
        states, so the shuffle arm is the width-fair one and is what we surface by default.
    """
    overlays: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(_MEASUREMENTS, "*.json"))):
        with open(path) as fh:
            raw = json.load(fh)
        prov = raw.get("provenance", {})
        if not isinstance(prov, dict):
            # Some measurements record provenance as a one-line STRING (the oracle/baseline
            # class). A malformed neighbour must never take down the whole viewer build.
            prov = {"note": str(prov)}
        base = {
            "id": os.path.basename(path)[:-5],
            "file": os.path.relpath(path, _REPO),
            "generation": prov.get("generation"),
            "step": prov.get("step"),
            "n_states": prov.get("n_states") or raw.get("meta", {}).get("n_states"),
            "date": prov.get("date"),
            "producer": prov.get("producer"),
            "note": prov.get("note"),
            "families": {},
            "blocks": {},
            "arms": {},
        }
        # The speed-stat GIGO: `pairwise_speed` (edge `v`) and C1's outspeed channel read stat
        # index 4 — SPECIAL DEFENSE — as "speed" until the 2026-08-06 fix (v58).
        #
        # Keying this on the AUDIT DATE would be wrong, and wrong in the direction that matters:
        # contamination is a property of the TRAINED MODEL, not of when someone measured it.
        # gen-2's audit is dated 2026-08-06 (the fix date) yet gen-2 trained entirely against the
        # buggy feature, so a date rule silently reports its `v` row as trustworthy. Both
        # pre-fix generations are named explicitly instead.
        base["contaminated"] = (["v", "c1"] if base["generation"] in _GIGO_GENERATIONS else [])
        base["contamination_note"] = (
            "`v` (pairwise_speed) and C1's outspeed channel read Special Defense as speed through "
            "this generation's entire training run — fixed 2026-08-06 (v58). These rows measure "
            "that the CHANNEL carries signal, not that the speed physics is correct."
            if base["contaminated"] else None)

        for fam, m in (raw.get("families") or {}).items():
            metrics = _norm_metrics(m)
            if not metrics:
                continue
            (base["arms"] if fam in ("all", "concat", "concat_cells") else base["families"])[fam] = metrics
        for blk, m in (raw.get("blocks") or {}).items():
            if not isinstance(m, dict):
                continue
            base["blocks"][blk] = {
                "width": m.get("width"),
                "zero_all": _norm_metrics(m.get("all", {})),
                "zero_threat": _norm_metrics(m.get("threat", {})),
                "shuffle_all": _norm_metrics(m.get("shuf_all", {})),
                "shuffle_threat": _norm_metrics(m.get("shuf_threat", {})),
            }
        if base["families"] or base["blocks"] or base["arms"]:
            overlays.append(base)

    # MERGE by (generation, step). The two producers write DIFFERENT halves of the same
    # checkpoint's picture — the edge audit knows the per-family rows, the op-block probe knows the
    # concat sub-blocks — and kept separate they make the selector actively misleading: picking
    # "gen-3 @9.6M op blocks" would leave every bias edge unmeasured, and picking the edge audit
    # would leave the concat on a coarser arm, with no hint that the other half exists.
    merged: Dict[Any, Dict[str, Any]] = {}
    for o in overlays:
        key = (o["generation"], o["step"])
        if key not in merged:
            o["sources"] = [o["file"]]
            merged[key] = o
            continue
        m = merged[key]
        m["sources"].append(o["file"])
        for section in ("families", "blocks", "arms"):
            for k, v in o[section].items():
                m[section].setdefault(k, v)          # first writer wins; both are recorded above
        # keep the widest sample and the fullest identity
        if (o.get("n_states") or 0) > (m.get("n_states") or 0):
            m["n_states"] = o["n_states"]
        if len(o["id"]) < len(m["id"]):
            m["id"] = o["id"]
    out = list(merged.values())
    for o in out:
        o["sources"] = sorted(o["sources"])
        o["id"] = f"{o['generation']}@{(o['step'] or 0) // 1000000}M"
    out.sort(key=lambda o: (o.get("date") or "", o.get("step") or 0, o["id"]))
    return out


def _doc_section(node_id: str) -> Optional[str]:
    if node_id in DOC_SECTION:
        return DOC_SECTION[node_id]
    for pre, sec in DOC_SECTION_PREFIX:
        if node_id.startswith(pre):
            return sec
    return None


def build_payload(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Graph + overlays + derived facts, everything the page needs, deterministic."""
    nodes, edges = graph["nodes"], graph["edges"]
    ids = {n["id"] for n in nodes}
    for e in edges:                       # fail loud rather than render a broken graph
        if e["src"] not in ids or e["dst"] not in ids:
            raise ValueError(f"edge references an unknown node: {e}")

    enriched_nodes = []
    for n in sorted(nodes, key=lambda x: x["id"]):
        m = dict(n)
        m["doc_section"] = _doc_section(n["id"])
        m["shape"] = NODE_SHAPE.get(n["kind"], "round-rectangle")
        enriched_nodes.append(m)

    enriched_edges = []
    for i, e in enumerate(sorted(edges, key=lambda x: (x["src"], x["dst"], x["type"],
                                                       x.get("family") or ""))):
        m = dict(e)
        m["id"] = f"e{i}"
        m["cls"] = EDGE_CLASS[e["type"]]
        m["carries"] = CLASS_CARRIES[m["cls"]]
        m["blurb"] = EDGE_TYPE_BLURB[e["type"]]
        note, heading = CONCEPT_NOTE[m["cls"]]
        m["concept_file"], m["concept_heading"] = note, heading
        enriched_edges.append(m)

    # Leak-safety, computed from the data rather than asserted: no `aux` edge may terminate at a
    # forward sink or any pointer logit.
    violations = [e["id"] for e in enriched_edges
                  if e["type"] == "aux" and (e["dst"] in FORWARD_SINKS
                                             or e["dst"].startswith("pointer."))]
    return {
        "meta": graph["meta"],
        "nodes": enriched_nodes,
        "edges": enriched_edges,
        "overlays": _load_overlays(),
        "palette": PALETTE_PROVENANCE,
        "class_order": CLASS_ORDER,
        "class_carries": CLASS_CARRIES,
        "edge_dash": EDGE_DASH,
        "edge_blurb": EDGE_TYPE_BLURB,
        "families": _families(),
        "leak_safety": {"violations": violations, "ok": not violations},
        # A content-derived identity for the architecture itself. Deterministic (so the committed
        # artifact stays byte-stable) and enough to answer "am I looking at the same graph as
        # last time" at a glance. Deliberately NOT a timestamp: git does not preserve mtimes, so
        # an embedded date would make `--check` fail on any fresh clone.
        "graph_sha256": hashlib.sha256(
            json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12],
    }


_ASSETS = os.path.join(_HERE, "arch_viewer_assets")


def _asset(name: str) -> str:
    """Read one of the viewer's front-end assets.

    The markup, CSS and JS live as REAL FILES rather than as a string literal in here. That is not
    tidiness: 675 lines of CSS+JS inside a Python string get no syntax highlighting, no linter and
    no parser, so `viewer.js` could only be checked by generating the HTML and cutting the script
    block back out of it by hand. As files, `node --check` is a test (see
    `build_arch_viewer_test.py`), and the `#sheetbar` bug — a rule placed after the media query
    that needed it, so `display:none` won and the bottom sheet shipped with no way to close it —
    is the kind of thing an editor flags for free.

    Read per call, never cached: `--serve` re-renders per request, so editing a stylesheet or the
    JS shows up on the next reload with no restart and without even the module swap in
    `_live_module`.
    """
    with open(os.path.join(_ASSETS, name), encoding="utf-8") as fh:
        return fh.read()


def _shell() -> str:
    """The markup with its CSS and JS inlined — one self-contained document, as file:// needs."""
    return (_asset("viewer.html")
            .replace("__CSS__", _asset("viewer.css").rstrip("\n"))
            .replace("__JS__", _asset("viewer.js").rstrip("\n")))


def render(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    blob = blob.replace("</", "<\\/")          # cannot terminate the <script> block
    return _shell().replace("__PAYLOAD__", blob)


# The one <script src> in the markup. Read from the asset so `--vendor`, the tests and the page
# can never disagree about which URL is the page's single network dependency.
def _cdn() -> tuple:
    m = re.search(r'<script src="([^"]+)"></script>', _asset("viewer.html"))
    if not m:
        raise SystemExit("viewer.html no longer has the cytoscape <script src> tag")
    return m.group(0), m.group(1)


def vendor(html: str) -> str:
    """Inline the cytoscape bundle so the artifact needs NO network at all.

    The committed `.html` deliberately stays on the CDN tag: it keeps the file small enough to
    diff, and the page already degrades loudly rather than blankly when the script is missing.
    `--vendor` is for the offline copy — a laptop on a plane, an air-gapped box — where "loudly"
    is not good enough. It is a SEPARATE output, never the committed artifact, so `--check` has
    exactly one thing to compare against.
    """
    import urllib.request

    tag, url = _cdn()
    try:
        with urllib.request.urlopen(url, timeout=30) as fh:
            js = fh.read().decode("utf-8")
    except Exception as exc:                    # noqa: BLE001 - any failure means no vendored copy
        raise SystemExit(f"--vendor could not fetch {url}: {exc}\n"
                         "It needs network access ONCE, to inline the bundle.")
    # A `</script>` anywhere in the bundle would terminate the block early.
    js = js.replace("</script>", "<\\/script>")
    return html.replace(tag, f"<script>{js}</script>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None,
                    help="a run's model_config.json — regenerate the graph live instead of using "
                         "the committed snapshot")
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if --out differs from what would be generated (drift gate)")
    ap.add_argument("--vendor", action="store_true",
                    help="inline the cytoscape bundle for a fully offline copy (needs network "
                         "once, and an explicit --out — it is never the committed artifact)")
    ap.add_argument("--serve", action="store_true",
                    help="serve the viewer, re-rendered from disk on every request (the headless "
                         "path: put a tunnel in front instead of copying a file around)")
    ap.add_argument("--port", type=int, default=6007,
                    help="port for --serve (default 6007, beside tensorboard's 6006)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address for --serve (default loopback — front it with the tunnel)")
    a = ap.parse_args(argv)

    if a.serve:
        from agents.model.arch_viewer_serve import serve      # imports a socket server; lazy
        return serve(a.port, a.host)

    if a.vendor and a.check:
        ap.error("--vendor and --check are mutually exclusive: --check compares against the "
                 "committed CDN-linked artifact")
    if a.vendor and os.path.abspath(a.out) == os.path.abspath(_DEFAULT_OUT):
        ap.error("--vendor needs an explicit --out; the committed artifact stays CDN-linked "
                 "so that --check has one thing to compare against")

    html = render(build_payload(_load_graph(a.config)))
    if a.vendor:
        html = vendor(html)
    if a.check:
        if not os.path.exists(a.out):
            print(f"MISSING {a.out}")
            return 1
        with open(a.out) as fh:
            cur = fh.read()
        if cur != html:
            print(f"STALE {a.out} — regenerate with: python -m agents.model.build_arch_viewer")
            return 1
        print(f"OK {a.out} is current")
        return 0
    with open(a.out, "w") as fh:
        fh.write(html)
    n = html.count('"id":')
    print(f"wrote {a.out}  ({len(html):,} bytes, {n} embedded ids)")
    print(f"open it with:  file://{os.path.abspath(a.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
