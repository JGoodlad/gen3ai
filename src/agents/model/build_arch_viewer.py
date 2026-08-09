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
import sys
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
# The split is deliberate: the CELL CONTENTS are PARSED out of `features_extractor.py` (those lines
# ARE the definition, and `edge_bias_test.py` pins them against the operator's own methods), so a
# cell-content change reaches the viewer without anyone remembering to update it. Only the one-line
# "what is this family FOR" phrase is curated here, because the code states that as a variable name
# and nothing else. `build_arch_viewer_test.py` fails if a family in the graph has no entry, so a
# new family cannot ship as a bare letter.
# --------------------------------------------------------------------------------------------
_EXTRACTOR = os.path.join(_HERE, "features_extractor.py")
FAMILY_LABEL = {
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
    "pi_projection": "3.1 `pi_projection` — 1131 → 512",
    "vf_projection": "3.2 `vf_projection` — 875 → 512",
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


_HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gen3AI — delivery digraph</title>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<style>
:root{--surface-1:#fcfcfb;--surface-2:#f2f2ef;--text-primary:#0b0b0b;--text-secondary:#52514e;
 --text-muted:#77756f;--line:#d9d8d3;--absolute:#2a78d6;--ratio:#eb6834;--none:#1baf7a;
 --unmeasured:#b8b6b0;--good:#1a7f4b;--bad:#c0392b;}
html[data-theme=dark]{--surface-1:#1a1a19;--surface-2:#242423;--text-primary:#fff;
 --text-secondary:#c3c2b7;--text-muted:#8e8d85;--line:#3a3a38;--absolute:#3987e5;--ratio:#d95926;
 --none:#199e70;--unmeasured:#55544f;--good:#4ec98a;--bad:#e66767;}
*{box-sizing:border-box}
body{margin:0;font:13px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 background:var(--surface-1);color:var(--text-primary)}
header{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:10px 14px;
 border-bottom:1px solid var(--line);background:var(--surface-2)}
h1{font-size:14px;margin:0;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--text-muted);font-size:11.5px}
.ctl{display:flex;gap:6px;align-items:center}
label{color:var(--text-secondary);font-size:11.5px}
select,button,input{font:inherit;font-size:12px;padding:4px 7px;border:1px solid var(--line);
 border-radius:6px;background:var(--surface-1);color:var(--text-primary)}
button{cursor:pointer}button:hover{background:var(--surface-2)}
button[aria-pressed=true]{background:var(--text-primary);color:var(--surface-1);
 border-color:var(--text-primary)}
#wrap{display:flex;height:calc(100vh - 52px)}
#cy{flex:1;min-width:0}
aside{width:352px;border-left:1px solid var(--line);overflow:auto;padding:12px 14px;
 background:var(--surface-2)}
aside h2{font-size:12.5px;margin:0 0 2px;letter-spacing:-.01em}
aside .kind{color:var(--text-muted);font-size:11px;margin-bottom:10px}
dl{margin:0 0 12px;display:grid;grid-template-columns:auto 1fr;gap:3px 10px;font-size:12px}
dt{color:var(--text-muted)}dd{margin:0;word-break:break-word}
.sec{margin:14px 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
 color:var(--text-muted);border-top:1px solid var(--line);padding-top:9px}
.chip{display:inline-flex;align-items:center;gap:5px;padding:1px 7px;border-radius:99px;
 border:1px solid var(--line);font-size:11px;margin:2px 3px 2px 0}
.sw{width:9px;height:9px;border-radius:2px;flex:none}
.elist{font-size:11.5px;max-height:238px;overflow:auto}
.elist div{padding:2px 0;border-bottom:1px dotted var(--line);color:var(--text-secondary)}
.elist .ghdr{border-bottom:none;padding:5px 0 1px}
.elist div b{color:var(--text-primary);font-weight:550}
.mut{color:var(--text-muted)}
.aged{color:var(--ratio);font-weight:600}
#prov code{font-size:11px}
#legend{position:absolute;left:12px;bottom:12px;background:var(--surface-2);border:1px solid
 var(--line);border-radius:8px;padding:9px 11px;font-size:11.5px;max-width:430px}
#legend b{font-weight:650}
#legend .row{display:flex;gap:8px;align-items:center;margin:3px 0}
#legend svg{flex:none}
.badge{padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600}
.badge.ok{background:var(--good);color:#fff}.badge.no{background:var(--bad);color:#fff}
#tip{position:absolute;pointer-events:none;background:var(--surface-1);border:1px solid var(--line);
 border-radius:7px;padding:7px 9px;font-size:11.5px;max-width:330px;box-shadow:0 4px 14px #0002;
 display:none;z-index:9}
#tbl{display:none;flex:1;overflow:auto;padding:10px 14px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
td .mut{white-space:normal}
th{position:sticky;top:0;background:var(--surface-2);cursor:pointer;color:var(--text-secondary)}
.famdef{border-top:1px solid var(--line);padding-top:6px;margin-top:6px}
.note{border:1px solid var(--line);border-left:3px solid var(--text-muted);border-radius:6px;
 padding:7px 9px;font-size:11.5px;color:var(--text-secondary);margin:6px 0}
.warn{background:color-mix(in srgb,var(--ratio) 15%,transparent);border:1px solid var(--ratio);border-radius:6px;padding:6px 9px;
 font-size:11.5px;margin:8px 0}
#err{padding:22px;font-size:13px;line-height:1.6}
</style></head><body>
<header>
  <h1>Gen3AI delivery digraph</h1>
  <span class="sub" id="hdr"></span>
  <span class="ctl"><label for="ck">checkpoint</label><select id="ck"></select></span>
  <span class="ctl"><label for="mt">metric</label><select id="mt">
    <option value="flip_rate">argmax flips</option><option value="kl_mean">masked KL</option>
    <option value="dv_mean">|ΔV| (critic)</option></select></span>
  <span class="ctl"><label for="bf">bias family</label><select id="bf"></select></span>
  <span class="ctl"><label for="fo">focus</label><select id="fo"></select>
    <select id="dir"><option value="anc">→ what feeds it</option>
      <option value="desc">what it feeds →</option><option value="both">both</option></select></span>
  <button id="clr">clear focus</button>
  <button id="tg" aria-pressed="false">table</button>
  <button id="dk" aria-pressed="true">dark</button>
  <span id="leak"></span>
  <span class="sub" id="prov"></span>
</header>
<div id="wrap"><div id="cy"></div><div id="tbl"></div><aside id="side"></aside></div>
<div id="legend"></div><div id="tip"></div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const $ = s => document.querySelector(s);
const CLS_VAR = {absolute:'--absolute', ratio:'--ratio', none:'--none'};
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

if (typeof cytoscape === 'undefined') {
  /* Marked on the BODY, not just written into it: the failure text also appears verbatim in this
     script's own source, so anything reading the DOM cannot tell "the page failed" from "the page
     contains the code that would say so". The attribute is unambiguous. */
  document.body.dataset.cytoscape = 'missing';
  document.body.innerHTML = '<div id="err"><b>cytoscape failed to load.</b><br>' +
    'This page needs the one CDN &lt;script&gt; tag at the top (cytoscape). ' +
    'You are probably offline. The data itself is embedded and intact — open the ' +
    '<code>#payload</code> script block, or use <code>designs/architecture_graph.dot</code>.</div>';
  throw new Error('cytoscape missing');
}

/* ---------- measurement lookup ------------------------------------------------------------ */
let OV = D.overlays.length ? D.overlays[D.overlays.length - 1] : null;
let METRIC = 'flip_rate';
/* Map an edge to its measured value under the selected overlay.
   bias   -> the family arm (each family's map zeroed).
   concat -> the op-block table's FULL_CONCAT, shuffle arm (width-fair); falls back to the
             edge-audit 'concat' arm when this checkpoint only has that.
   cell   -> only measured jointly as 'concat_cells'; reported, never faked per-edge. */
function measure(e) {
  if (!OV) return null;
  if (e.type === 'bias' && e.family && OV.families[e.family])
    return {v: OV.families[e.family][METRIC], src: 'family ' + e.family + ' (map zeroed)',
            bad: (OV.contaminated || []).includes(e.family)};
  if (e.type === 'concat' && e.src === 'damage_op') {
    const b = OV.blocks && OV.blocks.FULL_CONCAT;
    if (b && b.shuffle_all && b.shuffle_all[METRIC] != null)
      return {v: b.shuffle_all[METRIC], src: 'op block FULL_CONCAT (shuffle arm)', bad: false};
    if (OV.arms.concat) return {v: OV.arms.concat[METRIC], src: 'concat arm (zeroed)', bad: false};
  }
  if (e.type === 'cell' && OV.arms.concat_cells)
    return {v: OV.arms.concat_cells[METRIC], src: 'concat_cells arm (joint, not per-edge)',
            bad: false};
  return null;
}
/* measured -> px. sqrt so the small families stay visible against a max ~400x larger, and a
   deliberately modest ceiling: the widest channel here (the op's per-action cell route) is real
   and IS the policy's biggest dependency, but at 10px it becomes an opaque band that hides the
   seats it crosses. 6.4px still reads as "much the fattest" without erasing its neighbours. */
function scaleW(v) {
  if (v == null) return 1.1;
  const all = D.edges.map(measure).filter(Boolean).map(m => m.v).filter(x => x != null);
  const mx = Math.max(...all, 1e-9);
  return 1.0 + 5.4 * Math.sqrt(Math.max(v, 0) / mx);
}

/* ---------- graph ------------------------------------------------------------------------- */
const els = [
  ...D.nodes.map(n => ({data: {...n, label: n.id}})),
  ...D.edges.map(e => ({data: {...e, source: e.src, target: e.dst}})),
];
/* The stylesheet is a FUNCTION of the current CSS variables, not a literal, because cytoscape
   resolves `cssv()` once at construction and then owns those values on its own canvas. Baked in,
   a theme applied after construction — which is what a `#dark=1` deep link does — leaves every
   node painted in the other theme's palette on the new background. Rebuilt, both entry points
   (toggle and deep link) go through the same path. */
const sheet = () => [
    {selector: 'node', style: {
      'shape': 'data(shape)', 'label': 'data(label)', 'font-size': 11,
      'text-valign': 'center', 'color': cssv('--text-primary'),
      'background-color': cssv('--surface-2'), 'border-width': 1.4,
      'border-color': cssv('--line'), 'width': 'label', 'height': 'label',
      'padding': '7px', 'text-wrap': 'wrap', 'text-max-width': '104px'}},
    {selector: 'node[kind="head"], node[kind="operator"]', style: {
      'border-width': 2.6, 'font-weight': 'bold', 'font-size': 12}},
    {selector: 'node[kind="logit"]', style: {'font-size': 10}},
    {selector: 'edge', style: {
      'curve-style': 'bezier', 'width': 1.1, 'opacity': .82,
      'target-arrow-shape': 'triangle', 'arrow-scale': .55,
      'line-color': cssv('--unmeasured'), 'target-arrow-color': cssv('--unmeasured')}},
    {selector: 'edge[cls="absolute"]', style: {'line-color': cssv('--absolute'),
      'target-arrow-color': cssv('--absolute')}},
    {selector: 'edge[cls="ratio"]', style: {'line-color': cssv('--ratio'),
      'target-arrow-color': cssv('--ratio')}},
    {selector: 'edge[cls="none"]', style: {'line-color': cssv('--none'),
      'target-arrow-color': cssv('--none')}},
    /* Opacity states are CLASSES, deliberately ordered: an element `.style()` bypass outranks
       every stylesheet rule, so anything that fades an edge has to live here or focus-dimming
       silently loses to it. Later rule wins on a tie, so `.dim` is last. */
    {selector: 'edge.contam', style: {'opacity': .34}},
    {selector: 'edge.faint', style: {'opacity': .2}},
    {selector: 'edge.faint.contam', style: {'opacity': .1}},
    {selector: '.dim', style: {'opacity': .05, 'text-opacity': .12}},
    {selector: '.hid', style: {'display': 'none'}},
    {selector: '.sel', style: {'border-width': 3, 'border-color': cssv('--text-primary')}},
];
const cy = cytoscape({
  container: $('#cy'), elements: els, wheelSensitivity: .2, layout: {name: 'preset'},
  style: sheet(),
});
function applyTheme() { cy.style().fromJson(sheet()).update(); dashes(); restyle(); }
function dashes() {
  D.edges.forEach(e => {                   // dash = sub-channel (secondary encoding)
    const d = D.edge_dash[e.type];
    if (d) cy.$id(e.id).style({'line-style': 'dashed', 'line-dash-pattern': d});
  });
}
dashes();

/* ---------- layout -------------------------------------------------------------------------
   Hand-placed columns, NOT a layered ranker. 388 of the 487 edges are attention-bias edges
   running seat->seat WITHIN one 36-token sequence: peer edges, not pipeline stages. Handing
   those to dagre asks it to put 36 mutual peers in a rank order, and it answers with a diagonal
   smear whose fit-zoom drives every label sub-pixel. The pipeline the model actually has is
   short and known, so the columns are assigned outright and the seats pack into three bands.

   Bands come from a RULE on `kind`, never from an id list: a node added to the snapshot lands
   somewhere deliberate, and anything the rule does not recognise goes to a trailing MISC band
   that the header calls out rather than being quietly dropped or piled on the origin. This also
   covers the seats that carry no edge at all — `history[0..6]` have NONE, and `global` only
   RECEIVES bias (28 in, 0 out) — which a ranker has no opinion about and would drop wherever. */
const SEAT_BANDS = [['our_mon', 'opp_mon'],
                    ['E3_move', 'E4_threat', 'E5_tail'],
                    ['global', 'history']];
const MISC_BAND = 7;
const seatGroup = id => id.replace(/\[\d+\]$/, '');
function bandOf(n) {
  if (n.kind === 'input' || n.kind === 'belief_head') return 0;
  if (n.kind === 'operator') return 1;
  if (n.kind === 'seat') {
    const i = SEAT_BANDS.findIndex(b => b.includes(seatGroup(n.id)));
    return i < 0 ? MISC_BAND : 2 + i;
  }
  if (n.kind === 'head') return 5;
  if (n.kind === 'logit' || n.kind === 'aux_loss') return 6;
  return MISC_BAND;
}
const BANDS = Array.from({length: MISC_BAND + 1}, () => []);
D.nodes.forEach(n => BANDS[bandOf(n)].push(n));
/* Reading order inside a band: sources before beliefs, seats in declared group order, forward
   sinks before the training-only losses. */
const ordKey = n => (n.kind === 'aux_loss' ? '9' : n.kind === 'belief_head' ? '5' : '1') +
  (n.kind === 'seat' ? String(SEAT_BANDS.flat().indexOf(seatGroup(n.id))).padStart(2, '0') : '') +
  n.id;
BANDS.forEach(b => b.sort((a, c) => ordKey(a) < ordKey(c) ? -1 : 1));
(() => {
  /* The three seat bands are ONE logical column (the 36-token sequence) split only because 36
     rows would not fit; they get a tight gap so they read as one block, and the stage boundaries
     around them get a wide one. */
  const GAP_WIDE = 92, GAP_TIGHT = 34, ROW = 13, GROUP_GAP = 26;
  let x = 0, prevBand = -1;
  BANDS.forEach((band, bi) => {
    if (!band.length) return;
    if (prevBand >= 0) x += (prevBand >= 2 && prevBand <= 3 && bi <= 4) ? GAP_TIGHT : GAP_WIDE;
    prevBand = bi;
    const w = Math.max(...band.map(n => cy.$id(n.id).outerWidth()));
    let h = 0, prev = null;
    const ys = band.map(n => {                       // stack, with a gap where the group changes
      const g = n.kind === 'seat' ? seatGroup(n.id) : n.kind;
      if (prev !== null && g !== prev) h += GROUP_GAP;
      prev = g;
      const nh = cy.$id(n.id).outerHeight(), y = h + nh / 2;
      h += nh + ROW;
      return y;
    });
    const off = -(h - ROW) / 2;                      // every band centred on the same axis
    band.forEach((n, i) => cy.$id(n.id).position({x: x + w / 2, y: off + ys[i]}));
    x += w;
  });
})();
cy.fit(undefined, 24);

/* Which bias family is on show: 'all' (every family, faint, so the skeleton reads through 388
   overlapping edges), 'none' (hidden), or one family id at full strength. The family IS the unit
   the measurement is defined on, so this is the control that turns the overlay from a wash of
   colour into a readable per-family answer. */
let BIASSEL = 'all';
function restyle() {
  cy.batch(() => {
    D.edges.forEach(e => {
      const m = measure(e), el = cy.$id(e.id);
      el.style('width', scaleW(m ? m.v : null));
      const c = m ? cssv(CLS_VAR[e.cls]) : cssv('--unmeasured');
      el.style('line-color', c); el.style('target-arrow-color', c);
      const solo = e.type === 'bias' && BIASSEL !== 'all' && BIASSEL !== 'none';
      el.toggleClass('contam', !!(m && m.bad));
      el.toggleClass('faint', e.type === 'bias' && BIASSEL === 'all');
      el.toggleClass('hid', e.type === 'bias' && (BIASSEL === 'none'
                                                  || (solo && e.family !== BIASSEL)));
    });
  });
}

/* ---------- focus / path filter ----------------------------------------------------------- */
let FOCUS = null, DIR = 'anc';
function applyFocus() {
  cy.elements().removeClass('dim sel');
  if (!FOCUS) return;
  const n = cy.$id(FOCUS); if (!n.length) return;
  let keep = n.union(DIR === 'anc' ? n.predecessors()
                   : DIR === 'desc' ? n.successors()
                   : n.predecessors().union(n.successors()));
  cy.elements().difference(keep).addClass('dim');
  n.addClass('sel');
}

/* ---------- side panel -------------------------------------------------------------------- */
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function row(k, v) { return v == null || v === '' ? '' : `<dt>${esc(k)}</dt><dd>${v}</dd>`; }
const fmt = v => METRIC === 'flip_rate' ? (v * 100).toFixed(2) + '%' : v.toFixed(5);

/* A bias family is never shown as a bare code. `d2` means nothing to a reader; "d2 — our bench's
   offense vs their ACTIVE" means something immediately, and the full cell definition is one hover
   away. Every place a family id appears goes through one of these. */
const FAM = D.families || {};
const famLabel = f => (FAM[f] && FAM[f].label) || '';
const famName = f => famLabel(f) ? `${f} — ${famLabel(f)}` : f;
const famCells = f => {
  const m = FAM[f];
  if (!m || !m.cells) return '';
  return `[${m.cells}]` + (m.per ? ` per (${m.per})` : '') + (m.width ? ` · ${m.width} dims` : '');
};
const famTitle = f => {
  const m = FAM[f];
  if (!m) return f;
  return [famName(f), famCells(f), m.note].filter(Boolean).join('\n');
};

/* Forward reachability, by channel class. Built once; 58 nodes / 487 edges makes this trivial.
   `aux` is excluded from every route set — a training-only edge reaching a forward sink is the
   leak the badge exists to rule out, so it must never count as a way to get somewhere. */
const ADJ = {};
D.edges.forEach(e => (ADJ[e.src] = ADJ[e.src] || []).push(e));
const CONTENT_ROUTES = new Set(['content', 'concat', 'cell']);
const ANY_FORWARD = new Set(['content', 'concat', 'cell', 'bias']);
function reach(id, types) {
  const seen = new Set(), stack = [id];
  while (stack.length) {
    for (const e of ADJ[stack.pop()] || []) {
      if (!types.has(e.type) || seen.has(e.dst)) continue;
      seen.add(e.dst); stack.push(e.dst);
    }
  }
  return seen;
}
/* Where this token's information can actually GO, split by channel class. A bias edge DOES move
   information — it changes the attention weights, so it changes the pooled token — it just
   cannot carry a magnitude, so collapsing the two into one "reachable" would lose the whole
   point of the diagram.

   The `no route` verdict needs its caveat stated IN THE PANEL, not just here. This graph models
   EXPLICIT channels; ordinary self-attention among the 36 seats is not an edge in it. So a seat
   with no modelled channel (`history[0..6]` have no edge at all; `global` only receives bias) is
   "not tracked by this graph", NOT "unused by the model" — and on an artifact whose job is to be
   read aloud in a design conversation, that difference is exactly the kind of thing that would
   otherwise get repeated back as a finding. */
function delivery(id) {
  const viaContent = reach(id, CONTENT_ROUTES), viaAny = reach(id, ANY_FORWARD);
  const hit = (s, k) => k.endsWith('.') ? [...s].some(x => x.startsWith(k)) : s.has(k);
  const out = [['pi_projection', 'pi_projection'], ['vf_projection', 'vf_projection'],
               ['pointer logits', 'pointer.']]
    .filter(([, k]) => k !== id)
    .map(([label, k]) => row('→ ' + label,
      hit(viaContent, k) ? '<b>content</b> — concat / cell / token'
      : hit(viaAny, k) ? 'attention <b>bias only</b> — steers, carries no magnitude'
      : '<span class="mut">no route</span>')).join('');
  return out ? `<div class="sec">what it can deliver to</div><dl>${out}</dl>` : '';
}
/* Said in the panel, because "no route" on its own reads as "this token does nothing". */
const NO_CHANNEL_NOTE = '<div class="note">This seat carries <b>no modelled channel</b>. It still '
  + 'sits in the 36-token sequence and takes part in ordinary self-attention — the delivery graph '
  + 'tracks explicit channels (content / concat / cell / bias / aux) and does not model plain '
  + 'attention as an edge. Read this as <b>“not tracked here”</b>, not “unused by the model”.</div>';
/* Every bias family acting on this token, ranked by measured dependence at the selected
   checkpoint. This is the per-token form of the overlay: "what conditions this seat, and how
   much does the policy actually move when you switch it off". */
function familiesOn(id) {
  const n = {};
  D.edges.forEach(e => {
    if (e.type === 'bias' && (e.src === id || e.dst === id)) n[e.family] = (n[e.family] || 0) + 1;
  });
  const keys = Object.keys(n);
  if (!keys.length) return '';
  const rows = keys.map(f => ({
    f, n: n[f],
    v: OV && OV.families[f] ? OV.families[f][METRIC] : null,
    bad: !!(OV && (OV.contaminated || []).includes(f)),
  })).sort((a, b) => (b.v == null ? -1 : b.v) - (a.v == null ? -1 : a.v));
  return `<div class="sec">bias families on this token (${keys.length})</div>
    <div class="elist">${rows.map(r => `<div title="${esc(famTitle(r.f))}">
      <b>${esc(r.f)}</b> <span class="mut">×${r.n}</span> —
      ${r.v == null ? '<i>unmeasured here</i>'
        : fmt(r.v) + (r.bad ? ' <span class="mut">⚠ contaminated</span>' : '')}
      <div class="mut">${esc(famLabel(r.f))}</div></div>`).join('')}
    </div>`;
}
/* What the panel is currently showing, so a checkpoint or metric change re-renders it. Left
   static, the panel keeps displaying the numbers of a checkpoint you already navigated away
   from — stale figures beside a selector that says otherwise. */
let SEL = null;
function refreshPanel() {
  if (!SEL) return;
  (SEL.kind === 'node' ? showNode : showEdge)(SEL.id);
}
function showNode(id) {
  const n = D.nodes.find(x => x.id === id); if (!n) return;
  SEL = {kind: 'node', id};
  const ins = D.edges.filter(e => e.dst === id), outs = D.edges.filter(e => e.src === id);
  /* One row per edge, and NO cap: the list scrolls. The previous `.slice(0, 14)` printed a
     "×15" chip above 14 rows, which is the exact shape of a silent truncation. */
  const grp = es => D.class_order.map(c => {
    const k = es.filter(e => e.cls === c); if (!k.length) return '';
    return `<div class="ghdr"><span class="chip"><span class="sw"
      style="background:var(${CLS_VAR[c]})"></span>${c} ×${k.length}</span></div>` +
      k.map(e => `<div>${esc(e.type)} ${e.src === id ? '→' : '←'}
        <b>${esc(e.src === id ? e.dst : e.src)}</b></div>`).join(''); }).join('');
  $('#side').innerHTML = `<h2>${esc(id)}</h2><div class="kind">${esc(n.kind)}</div><dl>
    ${row('token index', n.index)}${row('token type', n.token_type)}
    ${row('action index', n.action_index)}${row('out dim', n.out_dim)}
    ${row('in / out', n.in_features ? n.in_features + ' → ' + n.out_features : null)}
    ${row('incoming dim', n.incoming_dim)}${row('label key', n.label_key)}</dl>
    ${ins.length + outs.length ? delivery(id) + familiesOn(id)
        : '<div class="sec">what it can deliver to</div>' + NO_CHANNEL_NOTE}
    <div class="sec">incoming (${ins.length})</div><div class="elist">${grp(ins) || '—'}</div>
    <div class="sec">outgoing (${outs.length})</div><div class="elist">${grp(outs) || '—'}</div>
    ${n.doc_section ? `<div class="sec">reference</div>
      <div>designs/ARCHITECTURE.md § <b>${esc(n.doc_section)}</b></div>` : ''}
    <div class="sec">focus</div>
    <div><button onclick="setFocus('${id}','anc')">what feeds it</button>
         <button onclick="setFocus('${id}','desc')">what it feeds</button></div>`;
}
function showEdge(id) {
  const e = D.edges.find(x => x.id === id); if (!e) return;
  SEL = {kind: 'edge', id};
  const m = measure(e);
  $('#side').innerHTML = `<h2>${esc(e.src)} → ${esc(e.dst)}</h2>
    <div class="kind">${esc(e.type)} · carries ${esc(e.carries)}</div>
    <dl>${row('channel', esc(e.blurb))}${row('width', e.width)}
    ${row('family', e.family ? `<b>${esc(e.family)}</b> — ${esc(famLabel(e.family))}` : null)}
    ${row('cells', e.family && famCells(e.family)
      ? `<code>${esc(famCells(e.family))}</code>` : null)}
    ${row('family note', e.family && FAM[e.family] && FAM[e.family].note
      ? esc(FAM[e.family].note) : null)}
    ${row('via', e.via ? esc(e.via) : null)}
    ${row('source constant', e.source_constant ? '<code>' + esc(e.source_constant) + '</code>' : null)}
    ${row('bidirectional', e.bidirectional ? 'yes' : null)}
    ${row('zero-init', e.zero_init ? 'yes — identity at init' : null)}
    ${row('note', e.note ? esc(e.note) : null)}</dl>
    ${m ? `<div class="sec">measured</div><dl>
       ${row(METRIC, fmt(m.v))}
       ${row('arm', esc(m.src))}${row('checkpoint', esc(OV.id))}</dl>
       ${m.bad ? `<div class="warn"><b>contaminated.</b> ${esc(OV.contamination_note)}</div>` : ''}`
     : '<div class="sec">measured</div><div class="kind">no measurement for this edge at the ' +
       'selected checkpoint — rendered in the neutral “unmeasured” style, never as zero.</div>'}
    <div class="sec">concept</div><div>${esc(e.concept_file)}<br>§ <b>${esc(e.concept_heading)}</b></div>`;
}

/* ---------- legend, table, controls -------------------------------------------------------- */
function legend() {
  const line = (c, dash) => `<svg width="34" height="8"><line x1="1" y1="4" x2="33" y2="4"
    stroke="var(${CLS_VAR[c]})" stroke-width="2.4" ${dash ? `stroke-dasharray="${dash.join(' ')}"` : ''}/></svg>`;
  $('#legend').innerHTML = `<b>Edge colour = what the channel physically carries</b>
    <div class="row">${line('absolute', null)}<span><b>absolute</b> — ${esc(D.class_carries.absolute)}</span></div>
    <div class="row" style="margin-left:16px">${line('absolute', D.edge_dash.concat)}<span>concat (head input) · ${line('absolute', D.edge_dash.cell)} cell (per-action) · solid = token content</span></div>
    <div class="row">${line('ratio', D.edge_dash.bias)}<span><b>ratio</b> — ${esc(D.class_carries.ratio)}</span></div>
    <div class="row">${line('none', D.edge_dash.aux)}<span><b>training-only</b> — ${esc(D.class_carries.none)}</span></div>
    <div class="row"><svg width="34" height="8"><line x1="1" y1="4" x2="33" y2="4"
      stroke="var(--unmeasured)" stroke-width="2.4"/></svg><span>no measurement at this checkpoint</span></div>
    <div class="row"><span class="sub">thickness = <span id="lgm"></span>; node shape = kind</span></div>
    ${BIASSEL !== 'all' && BIASSEL !== 'none' ? `<div class="row famdef"><span>
      showing <b>${esc(BIASSEL)}</b> — ${esc(famLabel(BIASSEL))}<br>
      <span class="mut">${esc(famCells(BIASSEL))}</span></span></div>` : ''}`;
  $('#lgm').textContent = $('#mt').selectedOptions[0].textContent;
}
function table() {
  const rows = D.edges.map(e => ({e, m: measure(e)}))
    .sort((a, b) => ((b.m && b.m.v) || -1) - ((a.m && a.m.v) || -1));
  $('#tbl').innerHTML = `<table><thead><tr><th>from</th><th>to</th><th>channel</th>
    <th>carries</th><th>width</th><th>family</th><th>${esc(METRIC)}</th><th>arm</th></tr></thead>
    <tbody>${rows.map(({e, m}) => `<tr><td>${esc(e.src)}</td><td>${esc(e.dst)}</td>
    <td>${esc(e.type)}</td><td>${esc(e.cls)}</td><td>${e.width ?? ''}</td>
    <td title="${esc(e.family ? famTitle(e.family) : '')}">${e.family
      ? esc(e.family) + ' <span class="mut">' + esc(famLabel(e.family)) + '</span>' : ''}</td>
    <td>${m ? fmt(m.v) : '—'}</td>
    <td>${m ? esc(m.src) + (m.bad ? ' ⚠ contaminated' : '') : ''}</td></tr>`).join('')}</tbody></table>`;
}
function header() {
  const m = D.meta;
  $('#hdr').textContent = `${m.arch_signature} · cfg v${m.config_version} · obs ${m.obs_dim} · ` +
    `${m.n_tokens} tokens · op ${m.op_out_dim} · ${D.nodes.length} nodes / ${D.edges.length} edges` +
    (BANDS[MISC_BAND].length ? ` · ⚠ ${BANDS[MISC_BAND].length} unplaced (` +
      BANDS[MISC_BAND].map(n => n.kind).join(', ') + ') — extend bandOf()' : '');
  /* Provenance, so staleness is something you SEE rather than something you have to reason
     about. The hash identifies the architecture (same hash = same graph as last time). The age
     only appears when served, because a static file has no way to know it — and it is the
     snapshot's age, NOT the page's: the page rebuilds per request, the graph underneath it does
     not. 14 days is a nudge, not a rule. */
  const prov = D.snapshot_built;
  $('#prov').innerHTML = `graph <code>${esc(D.graph_sha256 || '?')}</code>` + (prov
    ? ` · snapshot <span class="${prov.days >= 14 ? 'aged' : ''}">${prov.days}d old</span>` : '');
  $('#prov').title = 'graph_sha256 identifies the architecture itself.\n' + (prov
    ? `The delivery-graph snapshot was last rebuilt ${prov.iso} (${prov.source}).\n` +
      'The page re-renders on every request; the snapshot under it only changes when someone ' +
      'regenerates it after an architecture change.'
    : 'Snapshot age is only shown when served — a file:// copy cannot know it.');
  const L = D.leak_safety;
  $('#leak').innerHTML = `<span class="badge ${L.ok ? 'ok' : 'no'}">leak-safety ` +
    `${L.ok ? 'PASS' : 'FAIL (' + L.violations.length + ')'}</span>`;
  $('#leak').title = 'computed from the embedded data: no aux edge may terminate at ' +
    'pi_projection, vf_projection, or a pointer logit';
}
function setFocus(id, dir) { FOCUS = id; DIR = dir || DIR; $('#fo').value = id || '';
  $('#dir').value = DIR; applyFocus(); hash(); }
function hash() {
  const p = new URLSearchParams();
  if (FOCUS) { p.set('focus', FOCUS); p.set('dir', DIR); }
  p.set('ck', OV ? OV.id : ''); p.set('m', METRIC); p.set('bias', BIASSEL);
  p.set('theme', document.documentElement.dataset.theme);
  location.hash = p.toString();
}
function fromHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  /* The theme is pinned EXPLICITLY, not as a "dark=1" flag on top of a default. Dark is the
     default here, and a flag-over-default link silently changes meaning the day the default
     moves. */
  const th = p.get('theme') === 'light' ? 'light' : p.get('theme') === 'dark' ? 'dark' : null;
  if (th) { document.documentElement.dataset.theme = th;
            $('#dk').setAttribute('aria-pressed', String(th === 'dark')); }
  if (p.get('m')) METRIC = p.get('m');
  if (p.get('bias')) BIASSEL = p.get('bias');
  const ck = p.get('ck'); if (ck) { const o = D.overlays.find(x => x.id === ck); if (o) OV = o; }
  FOCUS = p.get('focus') || null; DIR = p.get('dir') || 'anc';
}

$('#ck').innerHTML = D.overlays.map(o => `<option value="${o.id}">${o.generation || o.id} @ ` +
  `${o.step ? (o.step / 1e6) + 'M' : '?'} · n=${o.n_states || '?'} · ${o.date || ''}</option>`).join('');
const FAMS = [...new Set(D.edges.filter(e => e.type === 'bias').map(e => e.family))].sort();
const famCount = f => D.edges.filter(e => e.type === 'bias' && e.family === f).length;
$('#bf').innerHTML = `<option value="all">all ${FAMS.length} families (faint)</option>` +
  FAMS.map(f => `<option value="${f}">${esc(famName(f))} (${famCount(f)})</option>`).join('') +
  '<option value="none">none (skeleton only)</option>';
$('#fo').innerHTML = '<option value="">— none —</option>' +
  D.nodes.map(n => `<option value="${n.id}">${n.id}</option>`).join('');
fromHash();
$('#ck').value = OV ? OV.id : ''; $('#mt').value = METRIC; $('#fo').value = FOCUS || '';
$('#dir').value = DIR; $('#bf').value = BIASSEL;
$('#bf').onchange = e => { BIASSEL = e.target.value; restyle(); legend(); hash(); };
$('#ck').onchange = e => { OV = D.overlays.find(o => o.id === e.target.value); restyle(); legend();
  table(); refreshPanel(); hash(); };
$('#mt').onchange = e => { METRIC = e.target.value; restyle(); legend(); table(); refreshPanel();
  hash(); };
$('#fo').onchange = e => setFocus(e.target.value || null);
$('#dir').onchange = e => { DIR = e.target.value; applyFocus(); hash(); };
$('#clr').onclick = () => setFocus(null);
$('#tg').onclick = e => { const on = e.target.getAttribute('aria-pressed') !== 'true';
  e.target.setAttribute('aria-pressed', on); $('#tbl').style.display = on ? 'block' : 'none';
  $('#cy').style.display = on ? 'none' : 'block'; $('#legend').style.display = on ? 'none' : 'block';
  if (on) table(); };
$('#dk').onclick = e => { const on = e.target.getAttribute('aria-pressed') !== 'true';
  e.target.setAttribute('aria-pressed', on);
  document.documentElement.dataset.theme = on ? 'dark' : 'light';
  applyTheme(); applyFocus(); legend(); hash(); };
cy.on('tap', 'node', ev => showNode(ev.target.id()));
cy.on('tap', 'edge', ev => showEdge(ev.target.id()));
cy.on('tap', ev => { if (ev.target === cy) $('#side').innerHTML =
  '<div class="kind">Click a node or an edge. Use <b>focus</b> to isolate everything that feeds ' +
  'a sink — e.g. <code>vf_projection</code> answers “what does the critic see”.</div>'; });
cy.on('mouseover', 'edge', ev => { const e = ev.target.data(), m = measure(e);
  const t = $('#tip'); t.style.display = 'block';
  t.innerHTML = `<b>${esc(e.src)} → ${esc(e.dst)}</b><br>${esc(e.type)} · ${esc(e.carries)}` +
    (e.width ? `<br>width ${e.width}` : '') +
    (e.family ? `<br>family <b>${esc(e.family)}</b> — ${esc(famLabel(e.family))}` +
      (famCells(e.family) ? `<br><span class="mut">${esc(famCells(e.family))}</span>` : '') : '') +
    (m ? `<br><b>${esc(METRIC)}</b> ${fmt(m.v)}${m.bad ? ' ⚠' : ''}`
       : '<br><i>unmeasured here</i>'); });
cy.on('mouseout', 'edge', () => $('#tip').style.display = 'none');
document.addEventListener('mousemove', e => { const t = $('#tip');
  t.style.left = Math.min(e.clientX + 14, innerWidth - 345) + 'px';
  t.style.top = (e.clientY + 16) + 'px'; });
header(); applyTheme(); legend(); applyFocus();   // applyTheme covers the `#dark=1` deep link
/* A deep link restores the panel too, not just the dimming — otherwise a shared
   `#focus=vf_projection` lands on a filtered graph with an empty panel beside it. */
if (FOCUS) showNode(FOCUS);
else $('#side').innerHTML = '<div class="kind">Click any node — a token seat, the operator, a ' +
  'head, a logit — for its channels, what it can deliver to, and the bias families acting on ' +
  'it. Use <b>focus</b> to isolate everything that feeds a sink: <code>vf_projection</code> ' +
  'answers “what does the critic see”.</div>';

/* A machine-readable record that the page actually RENDERED, for the headless check in
   `build_arch_viewer_render_integration_test.py`. Without it a render test can only assert the
   file was served; with it, it can assert the script ran to completion, that every node got a
   position, and — the bug this exists for — that the theme reached the CYTOSCAPE CANVAS and not
   merely the CSS, which is a separate palette that a deep-linked theme once missed. */
document.body.dataset.ready = '1';
document.body.dataset.nodes = String(cy.nodes().length);
document.body.dataset.positioned =
  String(cy.nodes().filter(n => n.position('x') !== 0 || n.position('y') !== 0).length);
document.body.dataset.nodeBg = cy.nodes()[0].style('background-color');
</script></body></html>
"""


def render(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    blob = blob.replace("</", "<\\/")          # cannot terminate the <script> block
    return _HTML.replace("__PAYLOAD__", blob)


# The one <script src> in the template. Named here so `--vendor` and the tests agree on it.
_CDN_TAG = re.search(r'<script src="([^"]+)"></script>', _HTML).group(0)
_CDN_URL = re.search(r'<script src="([^"]+)"></script>', _HTML).group(1)


def vendor(html: str) -> str:
    """Inline the cytoscape bundle so the artifact needs NO network at all.

    The committed `.html` deliberately stays on the CDN tag: it keeps the file small enough to
    diff, and the page already degrades loudly rather than blankly when the script is missing.
    `--vendor` is for the offline copy — a laptop on a plane, an air-gapped box — where "loudly"
    is not good enough. It is a SEPARATE output, never the committed artifact, so `--check` has
    exactly one thing to compare against.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(_CDN_URL, timeout=30) as fh:
            js = fh.read().decode("utf-8")
    except Exception as exc:                    # noqa: BLE001 - any failure means no vendored copy
        raise SystemExit(f"--vendor could not fetch {_CDN_URL}: {exc}\n"
                         "It needs network access ONCE, to inline the bundle.")
    # A `</script>` anywhere in the bundle would terminate the block early.
    js = js.replace("</script>", "<\\/script>")
    return html.replace(_CDN_TAG, f"<script>{js}</script>")


def _snapshot_provenance() -> Optional[Dict[str, Any]]:
    """When was the snapshot last regenerated? Served pages only.

    Not embedded in the committed artifact on purpose: git does not preserve mtimes, so a date
    baked into the HTML would differ on every clone and break the byte-comparison `--check`
    depends on. A live server, though, is sitting on a real checkout and can just look.

    Prefers the snapshot's last COMMIT date over its mtime — mtime tells you when the file was
    written (a `git pull` rewrites it), the commit date tells you when the architecture was
    actually last rebuilt, which is the question being asked.
    """
    import datetime
    import subprocess

    iso, source = None, "mtime"
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cI", "--", _SNAPSHOT],
                             cwd=_REPO, capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            iso, source = out.stdout.strip(), "git"
    except Exception:                              # noqa: BLE001 - git absent / not a checkout
        pass
    if iso is None:
        try:
            iso = datetime.datetime.fromtimestamp(
                os.path.getmtime(_SNAPSHOT)).astimezone().isoformat()
        except OSError:
            return None
    try:
        when = datetime.datetime.fromisoformat(iso)
        days = (datetime.datetime.now(when.tzinfo) - when).days
    except ValueError:
        return None
    return {"iso": iso, "days": days, "source": source}


def _live_module(_state={}):
    """Hand back THIS module, reloaded whenever its source changes. No restart, ever.

    The graph snapshot, the measurement files and the `_EDGE_*_CELL` block are already re-read per
    request, but the HTML template and `FAMILY_LABEL` are module globals bound at import — so
    without this, editing the generator kept serving the old page until someone restarted the unit.
    Now every input is live and the service only ever needs restarting for a Python upgrade.

    Loaded into a FRESH module object and swapped in only on success. `importlib.reload()` would be
    the obvious call and is the wrong one: it re-executes into the EXISTING namespace, so a file
    caught mid-save — half a function, a syntax error — corrupts the module you are still serving
    from. A fresh object either builds or is discarded, and the old one keeps working either way.
    """
    import importlib.util

    mtime = os.path.getmtime(__file__)
    if _state and _state["mtime"] == mtime:
        return _state["mod"]
    if not _state:                                 # first call: adopt the running module
        _state.update(mod=sys.modules[__name__], mtime=mtime)
        return _state["mod"]
    spec = importlib.util.spec_from_file_location(__name__ + "_live", __file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                   # a broken edit raises here; _state is untouched
    _state.update(mod=mod, mtime=mtime)            # ...so we only commit to a module that built
    return mod


def serve(port: int, host: str = "127.0.0.1") -> int:
    """Serve the viewer, RE-RENDERED FROM DISK on every request.

    This is the answer to "how does the endpoint stay up to date", and it answers it by removing
    the question. Nothing is uploaded and nothing is cached: each GET rebuilds the payload from the
    checkout this process is running in, so the page cannot be more stale than the working tree.
    A deployed copy — Pages, an rsync, a file dropped on a webserver — is a SECOND artifact that
    rots the moment the first one moves, which is the failure this whole module exists to prevent.

    Cheap enough to do per-request: the default path reads JSON and one source file, never imports
    torch, and renders in ~6 ms (measured). (`--config` does import `delivery_graph`, which is why
    the served build always uses the committed snapshot.)

    The RENDERING path is fully live, including this file — see `_live_module`: the snapshot, the
    measurements, the `_EDGE_*_CELL` block, the HTML template and `FAMILY_LABEL` all reach a
    request with no restart. What does NOT hot-reload is the server itself: `serve` and `Handler`
    are already-bound closures, so a change to the ROUTES or to what the handler injects needs
    `systemctl --user restart gen3ai-model-viewer`. Measured, not assumed — an architecture change
    reached the live endpoint untouched while a new handler-injected field did not.

    ETag + `Cache-Control: no-cache` rather than `no-store`: the browser revalidates on every load
    (so it can never show you a stale architecture) but gets a 304 and zero bytes back when nothing
    has changed. `no-store` would forbid the cache entirely and re-send 300 KB for every refresh,
    which buys nothing — the freshness comes from re-rendering, not from refusing to cache.

    Binds to LOOPBACK by default. The intended front door is the existing Cloudflare Tunnel, which
    connects from the same box; binding to 0.0.0.0 would put the architecture and its audit numbers
    on the LAN with no auth, which is not a decision this flag should make for you.

    Routes:
      /            the viewer
      /graph.json  the raw payload — the better thing to hand an AI than 300 KB of HTML
      /healthz     "ok" plus the node/edge counts, for a probe
    """
    import http.server
    import traceback

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "gen3ai-arch-viewer"

        def _send(self, code, body, ctype, etag=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            if etag:
                self.send_header("ETag", etag)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _body(self, body, ctype):
            """Serve with an ETag, answering 304 when the caller already has these bytes."""
            etag = '"' + hashlib.sha256(body).hexdigest()[:32] + '"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send(200, body, ctype, etag)

        def do_GET(self):                          # noqa: N802 - stdlib naming
            route = self.path.split("?")[0].split("#")[0]
            try:
                mod = _live_module()
                if route in ("/", "/index.html"):
                    payload = mod.build_payload(mod._load_graph(None))
                    payload["snapshot_built"] = mod._snapshot_provenance()
                    self._body(mod.render(payload).encode(), "text/html; charset=utf-8")
                elif route == "/graph.json":
                    payload = mod.build_payload(mod._load_graph(None))
                    payload["snapshot_built"] = mod._snapshot_provenance()
                    self._body(json.dumps(payload, indent=1).encode(),
                               "application/json; charset=utf-8")
                elif route == "/healthz":
                    p = mod.build_payload(mod._load_graph(None))
                    self._send(200,
                               f"ok {len(p['nodes'])} nodes {len(p['edges'])} edges\n".encode(),
                               "text/plain; charset=utf-8")
                else:
                    self._send(404, b"not found\n", "text/plain; charset=utf-8")
            except Exception:                      # noqa: BLE001
                # Fail LOUD and in place. A 500 with the traceback is worth far more than a page
                # that quietly keeps showing the last architecture that happened to build — and it
                # is what you want mid-edit, when the file on disk genuinely is broken.
                self._send(500, ("the viewer failed to render from this checkout:\n\n"
                                 + traceback.format_exc()).encode(),
                           "text/plain; charset=utf-8")

        do_HEAD = do_GET                           # noqa: N815 - stdlib naming

        def log_message(self, fmt, *args):         # noqa: A002 - stdlib signature
            pass                                   # one line per request is just noise here

    srv = http.server.ThreadingHTTPServer((host, port), Handler)
    print(f"serving the delivery digraph on http://{host}:{port}")
    print(f"  rendering live from {_REPO} — no build artifact, no restart on change")
    print("  routes: /  ·  /graph.json  ·  /healthz")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


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
