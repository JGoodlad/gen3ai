"""The ``main.reads`` rule ``b1_fresh_cell``: ONE fresh three-seed cell against ONE banked BOUNDARY-1 cell, read by
BOUNDARY-1's own head-to-head statistic and classes, one arm a side.

The registration is the frame ``designs/research_state/measurements/fresh_cell_frame_2026-11-17/FRAME.md`` (§3, §4),
which a card's ``PREREG.md`` adopts. This module is the rule and nothing else: two pure halves, ``inputs`` (the look's
plan and its ledger cells -> the input document) and ``decide`` (the input document -> the verdict), so a decision is
re-checked from ``inputs.json`` alone (``python -m main.b1.fresh recheck``). The kit that plans, plays and reads the
cross is :mod:`main.b1.fresh`.

**The design** (:class:`FreshDesign`). Two sides, :data:`FRESH` and :data:`BANKED`, ``n`` runs each (3 by the frame;
any equal ``n >= 2`` is accepted by the same code). A cell is one fresh run against one banked run, ``n x n`` of them,
STORED from the fresh side (the fresh run is the player); a cell stored from the other side is refused, never flipped.

**The statistic** is :func:`main.b1.cross.simple_effect` on the fresh side against the banked side, as coded: on a view,
``D = mean(the n x n cell scores) - 50``; ``V = s2_fresh / n + s2_banked / n``, each ``s2`` the within-side sample
variance of that side's per-run means (a fresh run's mean over the banked runs; a banked run's: the fresh side's mean
score against it), each floored at :data:`main.b1.cross.FLOOR_PP` squared; ``df = 2 (n - 1)``; the two-sided
:data:`main.b1.cross.LEVEL` interval by Student's t. It is B1's factor statistic (``factor_stat``) with ONE arm a side
in place of two: with one ON and one OFF arm, ``s2_rows`` and ``s2_cols`` are each one arm's floored variance, ``2n``
becomes ``n`` and ``4 (n - 1)`` becomes ``2 (n - 1)``. At ``n = 3``: 4 df, t = 2.132, a half-width never below 3.06 pp.

**The classes** are read on view ``N`` (:data:`DECIDING_VIEW`) alone; every other view of
:data:`main.b1.cross.VIEWS` is REPORTED.

* ``100M``: :func:`main.b1.cross.classify` at ``100M``, B1 §5.3's table in its order: WIN (lower end > 0), LOSS (upper
  end < 0), NO-USEFUL-GAIN (upper end < +4.0), else UNDECIDED.
* ``50M``: the frame's continuation rule (§4), which reads the POINT estimate: CONTINUE iff ``D > 0`` by more than
  :data:`main.b1.cross.EPS_PP`, else STOPPED-AT-50M (it carries no class). A STOPPED-AT-50M read whose upper end is
  below B1's harm line (:data:`main.b1.cross.HARM_PP`) is also labelled HARM-AT-50M, a description.
* Any bound or estimate within :data:`main.b1.cross.EPS_PP` of the threshold it is compared with fails that comparison
  and is named in ``at_boundary`` (root rule 8, B1's ``_clears``).

**INCONCLUSIVE, never a class** (B1 §5 and §10 as coded, unchanged): any cell missing, listed twice, stored from the
wrong side, outside the design, not OK by the ledger, holding other than the registered mirrored pairs on its panel
(:data:`main.b1.cross.PAIRS`), or with more than a quarter of its games drawn (:func:`main.b1.cross.scores_from_cells`);
any row not played as registered (another commit than the pin, another panel's team set, a second device or backend:
``main.b1.look.row_problems``, passed in by the kit).

**Label order.** The rule reads the fresh side against the banked side and is not symmetric in that choice: reading the
same cells the other way round negates ``D`` and mirrors the interval (so a WIN would read as a LOSS). It IS invariant
to the order of the runs within a side, to the order of the cells and panels in the input, and to which side a cell
was stored from once it is oriented (the tests hold each).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from main.b1 import cross as X

RULE = "b1_fresh_cell"
SCHEMA = "gen3_read_b1_fresh_cell_v1"
FRESH, BANKED = "fresh", "banked"
SIDES: Tuple[str, str] = (FRESH, BANKED)
#: The looks the frame reads (§4): 50M decides whether to continue, 100M decides.
LOOKS: Tuple[str, ...] = ("50M", "100M")
DECIDING_VIEW = X.N_VIEW
CONTINUE = X.CONTINUE
STOPPED_AT_50M = "STOPPED-AT-50M"
HARM_AT_50M = "HARM-AT-50M"
INCONCLUSIVE = "INCONCLUSIVE"
KEYS = ("look",)
RULE_DOCS = {RULE: "one fresh 3-seed cell against one banked BOUNDARY-1 cell (fresh_cell_frame_2026-11-17/FRAME.md "
                   "§3-§4): main.b1.cross.simple_effect on N (floored within-side variances, t on 2(n-1) df, 90 %); "
                   "100M WIN / LOSS / NO-USEFUL-GAIN / UNDECIDED by main.b1.cross.classify, 50M CONTINUE iff the "
                   "estimate > +0.005 pp else STOPPED-AT-50M; decided by the kit (python -m main.b1.fresh read), "
                   "never by a main.reads spec"}


def _err(msg: str) -> Exception:
    from main.reads.rules import RuleInputError

    return RuleInputError(msg)


# ------------------------------------------------------------------------------------------------- the design
@dataclass(frozen=True)
class FreshDesign:
    """side -> its runs (run names), in the plan's order. The cells are (fresh run, banked run): the fresh side plays."""

    runs: Mapping[str, Tuple[str, ...]]

    def __post_init__(self) -> None:
        if set(self.runs) != set(SIDES):
            raise X.CrossInputError(f"a fresh-cell design holds the sides {SIDES}, got {sorted(self.runs)}")
        sizes = {s: len(self.runs[s]) for s in SIDES}
        if len(set(sizes.values())) != 1 or min(sizes.values()) < 2:
            raise X.CrossInputError(f"every side needs the same number (>= 2) of runs, got {sizes}")
        flat = [r for s in SIDES for r in self.runs[s]]
        if len(set(flat)) != len(flat):
            raise X.CrossInputError("a run appears twice in the design")

    @property
    def n(self) -> int:
        return len(self.runs[FRESH])

    def cells(self) -> List[X.Cell]:
        return [(p, o) for p in self.runs[FRESH] for o in self.runs[BANKED]]


def design_of(runs: Mapping[str, Sequence[str]]) -> FreshDesign:
    return FreshDesign({s: tuple(rs) for s, rs in runs.items()})


# ------------------------------------------------------------------------------------------ statistic, classes
def stat(design: FreshDesign, scores: X.Scores, view: str = DECIDING_VIEW) -> X.Stat:
    """The registered statistic on already-blended ``scores`` (module docstring): B1's simple effect, fresh vs banked."""
    return X.simple_effect(design, scores, FRESH, BANKED, view=view)


def classify(look: str, st: X.Stat) -> Dict[str, Any]:
    """``{"outcome", "at_boundary", "labels"}`` at ``look`` from the statistic on the deciding view."""
    if look not in LOOKS:
        raise X.CrossInputError(f"look {look!r} is not one of {LOOKS}")
    if look == "100M":
        v = X.classify("100M", st)
        return {"outcome": v.outcome, "at_boundary": list(v.at_boundary), "labels": []}
    near: List[str] = []
    ahead, nb = X._clears(st.estimate, 0.0, above=True)
    if nb:
        near.append(f"CONTINUE (the estimate): {st.estimate!r} within {X.EPS_PP} pp of 0")
    if ahead:
        return {"outcome": CONTINUE, "at_boundary": near, "labels": []}
    harm, nh = X._clears(st.hi, X.HARM_PP, above=False)
    if nh:
        near.append(f"HARM-AT-50M (upper end): {st.hi!r} within {X.EPS_PP} pp of {X.HARM_PP}")
    return {"outcome": STOPPED_AT_50M, "at_boundary": near, "labels": [HARM_AT_50M] if harm else []}


# ---------------------------------------------------------------------------------------------- the rule halves
def check(p: Mapping[str, Any], *, models: Any = None, anchor: Any = None, s: Any = None) -> List[str]:
    """The registered parameters: ``{"look": "50M" | "100M"}`` and nothing else. A ``main.reads`` spec cannot carry
    this rule: ``main.reads`` has no head-to-head lane, and a spec with neither an anchor nor an S group is refused at
    load, so any spec naming it is refused here (the kit decides it)."""
    probs: List[str] = []
    if anchor or s or models:
        probs.append(f"{RULE} is decided by the B1 kit (python -m main.b1.fresh read), not by a main.reads spec: "
                     "main.reads plays no head-to-head cross")
    if not isinstance(p, Mapping):
        return probs + [f"{RULE}: params must be a mapping"]
    extra = sorted(set(p) - set(KEYS))
    if extra:
        probs.append(f"{RULE}: unknown params {extra}")
    if p.get("look") not in LOOKS:
        probs.append(f"{RULE}: look {p.get('look')!r} is not one of {LOOKS}")
    return probs


def _draws(cells: Sequence[Any]) -> Dict[str, Any]:
    games = sum(int(c.games) for c in cells)
    draws = sum(int(c.d) for c in cells)
    worst = max((int(c.d) / int(c.games) for c in cells if int(c.games) > 0), default=0.0)
    return {"games": games, "draws": draws, "share": (draws / games if games else None), "worst_cell_share": worst}


def inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any = None, *,
           pairs: Mapping[str, int] = X.PAIRS) -> Dict[str, Any]:
    """The input document from the kit's look: ``cells`` = ``{"look": <look.json>, "cells_by_panel": {panel: [ledger
    Cell]}, "row_reasons": [str]}``. ``pairs`` (tests) overrides the registered pairs a cell. ``spec`` is unused."""
    bad = check(p)
    if bad:
        raise _err("; ".join(bad))
    doc = cells.get("look")
    if not isinstance(doc, Mapping):
        raise _err(f"{RULE}: no look document")
    if doc.get("look") != p["look"]:
        raise _err(f"{RULE}: the look document is the {doc.get('look')!r} look, the params read {p['look']!r}")
    runs = {s: [f["run"] for f in doc["runs"][s]] for s in SIDES}
    design = design_of(runs)
    run_of = {f["sha256"]: f["run"] for s in SIDES for f in doc["runs"][s]}
    by_panel = cells.get("cells_by_panel") or {}
    reasons: List[str] = [str(r) for r in cells.get("row_reasons") or ()]
    scores: Dict[str, List[List[Any]]] = {}
    for panel in X.PANELS:
        got, why = X.scores_from_cells(by_panel.get(panel, ()), run_of, design, int(pairs[panel]))
        if got is None:
            reasons.extend(f"{panel}: {w}" for w in why)
        else:
            scores[panel] = [[a, b, got[(a, b)]] for a, b in design.cells()]
    return {"schema": SCHEMA, "rule": RULE, "look": doc["look"], "card": doc.get("card"), "pin": doc.get("pin"),
            "runs": runs, "pairs": {k: int(pairs[k]) for k in X.PANELS},
            "checkpoints": {f["run"]: {"steps": f["steps"], "sha256": f["sha256"]} for s in SIDES
                            for f in doc["runs"][s]},
            "draws": {panel: _draws(by_panel.get(panel, ())) for panel in X.PANELS},
            "reasons": reasons, "scores": scores if not reasons else {}}


def decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    """The verdict from the input document alone (module docstring)."""
    bad = check(p)
    if bad:
        raise _err("; ".join(bad))
    if doc.get("schema") != SCHEMA or doc.get("look") != p["look"]:
        raise _err(f"{RULE}: the input document is not a {SCHEMA} of the {p['look']} look")
    head: Dict[str, Any] = {"schema": SCHEMA, "rule": RULE, "look": doc["look"], "card": doc.get("card"),
                            "pin": doc.get("pin"), "runs": {s: list(doc["runs"][s]) for s in SIDES},
                            "floor_pp": X.FLOOR_PP, "level": X.LEVEL, "read_on": DECIDING_VIEW,
                            "draws": doc.get("draws")}
    if doc.get("reasons"):
        return dict(head, decision={"verdict": INCONCLUSIVE, "outcome": INCONCLUSIVE, "at_boundary": [],
                                    "labels": [], "reasons": list(doc["reasons"])})
    design = design_of(doc["runs"])
    by_panel: Dict[str, Dict[X.Cell, float]] = {}
    for panel in X.PANELS:
        rows = doc["scores"].get(panel)
        if rows is None:
            raise _err(f"{RULE}: no scores for panel {panel}")
        by_panel[panel] = {(str(a), str(b)): float(v) for a, b, v in rows}
    stats = {v: stat(design, X.blend(design, by_panel, v), v) for v in X.VIEWS}
    st = stats[DECIDING_VIEW]
    cls = classify(str(doc["look"]), st)
    blended = X.blend(design, by_panel, DECIDING_VIEW)
    words = (f"{cls['outcome']} on {DECIDING_VIEW}: fresh - banked = {st.estimate:+.2f} pp "
             f"[{st.lo:+.2f}, {st.hi:+.2f}] ({int(round(100 * st.level))} %, {st.df} df)")
    return dict(head, n_per_side=design.n, rows=list(design.runs[FRESH]), cols=list(design.runs[BANKED]),
                h_pp=[[X.oriented(blended, r, c) for c in design.runs[BANKED]] for r in design.runs[FRESH]],
                views={v: s.as_dict() for v, s in stats.items()},
                decision={"verdict": cls["outcome"], "outcome": cls["outcome"], "at_boundary": cls["at_boundary"],
                          "labels": cls["labels"], "words": words, "reasons": []})


def read(look_doc: Mapping[str, Any], cells_by_panel: Mapping[str, Sequence[Any]], row_reasons: Sequence[str] = (),
         *, pairs: Mapping[str, int] = X.PAIRS) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(inputs, verdict)`` of a played look (``pairs``: tests only)."""
    p = {"look": str(look_doc["look"])}
    doc = inputs({"look": look_doc, "cells_by_panel": cells_by_panel, "row_reasons": list(row_reasons)}, p,
                 pairs=pairs)
    return doc, decide(doc, p)

