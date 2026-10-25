"""The REGISTERED interval and decision rules, by name (``designs/tools/reads.md`` §5). Pure functions; no game.

A read names ONE rule in its spec (``decision.rule``). Each :class:`Rule` has two halves, both pure:

* ``inputs(cells, params, spec)`` -- the rule's INPUT DOCUMENT from the assembled cells (``main.reads.assemble``):
  exactly the JSON the rule reads, written beside the record, so a decision is re-checkable from that file alone;
* ``decide(doc, params)`` -- the verdict document, as recorded.

THE ESTIMATORS are not re-derived here. Each is the one function a registered read already uses, re-exported under
one name so a new read imports this module and nothing else: Welch / R1's classes (:mod:`main.x35.readout`), the
equal-weight stratified Wilson, Newcombe-of-strata and AM-5's overdispersion φ̂ (:mod:`main.x35.d1`), Wilson and
Newcombe method 10 at z = 1.959963985 (:mod:`main.anchors.results`) and at z = 1.96 (``best_response_gap``). Rule 8 is
:func:`clears` (``readout``'s: within :data:`RULE8_EPS` of a boundary is never a crossing). Migrated here from the
per-read scripts (they had no home in ``src/``): :func:`cell_se` and :func:`s_se` (the X26 drift pre-read's declared
per-final interval of S), :func:`paired_unit_diff` (x26c's descriptive unit-paired difference) and
:func:`guards_not_read` (the orchestrator's A1 ruling (a): the guards gate only a QUALIFYING primary).

THE RULES (:data:`RULES`):

* ``x35_r1`` -- X35's R1 / R2 / guards / verdict (``readout.read``), on S per seed final (SPRT, LRF, A1 ...);
* ``d1`` -- D1's (X38 / X41) H on the unfamiliar Kakuna panels widened by φ̂, and P (``d1.read_d1``);
* ``anchor_trajectory`` -- x26c PREREG §5: the goal-(b) bar on an anchor score, CLIMBING / FLAT on its Newcombe
  difference against a comparator, and the in-house S "rising" read against a banked node.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from main.anchors.results import Z95, newcombe, wilson
from main.x35 import d1 as D
from main.x35 import readout as R
from main.x35.d1 import newcombe_strat, overdispersion, strat_score, widen, wilson_pn
from main.x35.readout import RULE8_EPS, classify, welch_interval

__all__ = ["RULES", "RULE8_EPS", "Rule", "RuleInputError", "Z95", "cell_se", "classify", "clears", "decide",
           "guards_not_read", "inputs", "newcombe", "newcombe_diff_ci", "newcombe_strat", "overdispersion",
           "paired_unit_diff", "s_se", "strat_score", "welch_interval", "widen", "wilson", "wilson_pn"]


class RuleInputError(ValueError):
    """The cells cannot form this rule's input (a missing group, side or seed): named, never a default."""


def newcombe_diff_ci(w1: int, n1: int, w2: int, n2: int, z: float = 1.96) -> Tuple[float, float, float]:
    """``best_response_gap.newcombe_diff_ci`` (R2 / G-E's meter, z = 1.96)."""
    from agents.training.best_response_gap import newcombe_diff_ci as _n

    return _n(w1, n1, w2, n2, z)


def clears(stat: float, boundary: float, *, above: bool = True) -> Tuple[bool, bool]:
    """``(crosses, near)``: ``stat`` > ``boundary`` (``above``) or < it; within :data:`RULE8_EPS` is ``near`` and never
    a crossing (standing rule 8; ``readout``'s convention, which x26c's ``lb > bar + EPS`` equals)."""
    return R._clears(stat, boundary, above=above)


def cell_se(pairs: Sequence[int]) -> float:
    """The per-pair SE of one mirrored cell's mean score from its pentanomial (pair score i/4, i = 0..4)."""
    n = sum(pairs)
    if n <= 0:
        raise RuleInputError("a cell with no completed pair has no SE")
    mu = sum(c * i for i, c in enumerate(pairs)) / (4.0 * n)
    var = sum(c * (i / 4.0 - mu) ** 2 for i, c in enumerate(pairs)) / n
    return math.sqrt(var / n)


def s_se(per_member_se_pp: Sequence[float]) -> float:
    """se(S) = sqrt(Σ se_m²) / K over R's K members (pp): the pre-read's declared per-final interval."""
    if not per_member_se_pp:
        raise RuleInputError("no member SE")
    return math.sqrt(math.fsum(x * x for x in per_member_se_pp)) / len(per_member_se_pp)


def paired_unit_diff(per_unit: Mapping[Tuple[str, str, int], float], on: str, off: str, panels: Sequence[str],
                     unit_seeds: Sequence[int], z: float = Z95) -> Optional[Dict[str, Any]]:
    """DESCRIPTIVE: the panel-equal-weight mean of unit-paired win-rate differences (same panel, same unit seed),
    with a normal interval on the per-panel SEs (x26c ``verdict.py``); ``None`` when a panel has < 2 pairs."""
    by: Dict[str, List[float]] = {k: [] for k in panels}
    for k in panels:
        for seed in unit_seeds:
            a, b = per_unit.get((on, k, seed)), per_unit.get((off, k, seed))
            if a is not None and b is not None:
                by[k].append(a - b)
    if not all(len(v) > 1 for v in by.values()):
        return None
    K = len(panels)
    m = sum(sum(v) / len(v) for v in by.values()) / K
    var = sum((sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) / len(v) for v in by.values()) / (K * K)
    return {"d": m, "lo": m - z * math.sqrt(var), "hi": m + z * math.sqrt(var), "pairs": {k: len(v) for k, v in by.items()}}


def guards_not_read(v: Dict[str, Any], ruling: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The A1 ruling (a): on a NON-qualifying primary the guards were not played, so each named guard is recorded
    INCONCLUSIVE with the ruling's reason (``ruling = {"guards": [...], "why": str, "absent": {guard: why}}``);
    ``absent`` replaces a guard's reasons and keeps its status. A qualifying primary is left untouched (its
    guards decide, and an unplayed one is INCONCLUSIVE by the rule itself)."""
    if not ruling:
        return v
    primary = v.get("h") if "h" in v else v.get("r1")
    qualifies = (primary or {}).get("class") == R.GAIN or v.get("verdict") in (R.ADOPT, R.TRADE)
    if qualifies:
        return v
    for n in ruling.get("guards") or ():
        v["guards"][n].update(reasons=[str(ruling["why"])], status=R.INCONCLUSIVE)
    for n, why in (ruling.get("absent") or {}).items():
        v["guards"][n].update(reasons=[str(why)])
    return v


def _jsonable(d: Any) -> Any:
    """The record's own form (what ``json.dump(..., default=str)`` writes), so a decision compares to a file."""
    return json.loads(json.dumps(d, default=str))


# ------------------------------------------------------------------------------------------------ helpers
def _seed_side(spec: Any, labels: Sequence[str], on: str, off: str) -> Dict[str, Dict[str, str]]:
    """``{"on": {seed: label}, "off": {...}}`` by the spec's arm / seed; two finals of a side on one seed refuse."""
    out: Dict[str, Dict[str, str]] = {"on": {}, "off": {}}
    for lab in labels:
        m = spec.models[lab]
        side = "on" if m.arm == on else "off" if m.arm == off else None
        if side is None:
            continue
        if str(m.seed) in out[side]:
            raise RuleInputError(f"two {m.arm} finals share seed {m.seed}")
        out[side][str(m.seed)] = lab
    return out


def _group(cells: Mapping[str, Any], lane: str, name: Any) -> Dict[str, Any]:
    g = (cells.get(lane) or {}).get(name)
    if g is None:
        raise RuleInputError(f"no assembled {lane} group {name!r}")
    return dict(g)


def imported_s(entry: Mapping[str, Any], *, want_refpop: Optional[str] = None) -> Dict[str, Optional[float]]:
    """``{seed: S_pp}`` of ``entry["arm"]`` from ANOTHER read's committed S results (``finals`` rows of a g1 / d1s /
    sprt ``results.json``), its sha256 checked (``entry = {path, sha256, arm}``): a banked control, never re-played."""
    from main.reads.models import sha256_file, spec_path

    p = spec_path(str(entry["path"]))
    got = sha256_file(p)
    if got != entry["sha256"]:
        raise RuleInputError(f"imported S {p}: sha256 {got[:12]} != the spec's {str(entry['sha256'])[:12]}")
    doc = json.loads(p.read_text())
    if want_refpop is not None and doc.get("refpop_content_sha256") != want_refpop:
        # sprt_read/assemble.py's STOP: "the two reads used different R"
        raise RuleInputError(f"imported S {p}: R content {str(doc.get('refpop_content_sha256'))[:12]} is not this "
                             f"read's {want_refpop[:12]}")
    rows = doc["finals"]
    return {str(r["seed"]): (r["s_pp"] if not r.get("inconclusive") else None) for r in rows
            if r["arm"] == entry["arm"]}


def _check_keys(params: Mapping[str, Any], need: Sequence[str], allowed: Sequence[str]) -> List[str]:
    probs = [f"missing {k!r}" for k in need if k not in params]
    probs += [f"unknown {k!r}" for k in sorted(set(params) - set(allowed))]
    return probs


# ------------------------------------------------------------------------------------------------ x35_r1
_R1_KEYS = ("arm", "look", "on", "off", "s_group", "import_off", "dose", "guards_not_read")


def _r1_check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    probs = _check_keys(p, ("arm", "look", "on", "off", "s_group"), _R1_KEYS)
    if p.get("s_group") in s and s[p["s_group"]].pairs < R.MIN_PAIRS:
        probs.append(f"S group pairs {s[p['s_group']].pairs} < the registered {R.MIN_PAIRS} per cell")
    if p.get("arm") not in R.ARMS:
        probs.append(f"arm {p.get('arm')!r} is none of {sorted(R.ARMS)}")
    if p.get("look") not in R.LOOK_SEEDS:
        probs.append(f"look {p.get('look')!r} is none of {sorted(R.LOOK_SEEDS)}")
    if p.get("s_group") not in s:
        probs.append(f"s_group {p.get('s_group')!r} is not an S group")
    return probs


def _r1_inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    g = _group(cells, "s", p["s_group"])
    sides = _seed_side(spec, list(g["nodes"]), p["on"], p["off"])
    r1: Dict[str, Any] = {side: {seed: g["nodes"][lab]["s_pp"] for seed, lab in sorted(sides[side].items())}
                          for side in ("on", "off")}
    if p.get("import_off"):
        if r1["off"]:
            raise RuleInputError("import_off with OFF finals also played in the S group")
        r1["off"] = imported_s(p["import_off"], want_refpop=spec.s[p["s_group"]].refpop_content_sha256)
    if g.get("inconclusive"):
        r1["inconclusive"] = list(g["inconclusive"])          # readout: an INCONCLUSIVE reason, never a score
    doc: Dict[str, Any] = {"arm": p["arm"], "look": p["look"], "r1": r1}
    if p.get("dose"):
        doc["dose"] = dict(p["dose"])
    return doc


def _r1_decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    return guards_not_read(_jsonable(R.read(doc).as_dict()), p.get("guards_not_read"))


# ------------------------------------------------------------------------------------------------ d1
_D1_KEYS = ("arm", "look", "on", "off", "anchor_group", "s_group", "guards_not_read")


def _d1_check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    probs = _check_keys(p, ("arm", "look", "on", "off", "anchor_group", "s_group"), _D1_KEYS)
    if p.get("s_group") in s and s[p["s_group"]].pairs < R.MIN_PAIRS:
        probs.append(f"S group pairs {s[p['s_group']].pairs} < the registered {R.MIN_PAIRS} per cell")
    if p.get("arm") not in D.PANEL_ROLES:
        probs.append(f"arm {p.get('arm')!r} is none of {sorted(D.PANEL_ROLES)}")
    if p.get("look") not in R.LOOK_SEEDS:
        probs.append(f"look {p.get('look')!r} is none of {sorted(R.LOOK_SEEDS)}")
    a = anchor.get(p.get("anchor_group"))
    if a is None:
        probs.append(f"anchor_group {p.get('anchor_group')!r} is not an anchor group")
    elif tuple(a.teamsets) != D.PANELS:
        probs.append(f"anchor group teamsets {list(a.teamsets)} are not D1's panels {list(D.PANELS)}")
    if p.get("s_group") not in s:
        probs.append(f"s_group {p.get('s_group')!r} is not an S group")
    return probs


def h_input(group: Mapping[str, Any], spec: Any, on: str, off: str) -> Dict[str, Any]:
    """D1's H (``kakuna.d1h_input``'s shape) from an assembled anchor group: each seed final's own counts per
    panel over COUNTED units, ties and units beside."""
    out: Dict[str, Any] = {"on": {}, "off": {}, "ties": {"on": {}, "off": {}}, "units": {"on": {}, "off": {}}}
    sides = _seed_side(spec, list(group["cells"]), on, off)
    for side in ("on", "off"):
        for seed, lab in sorted(sides[side].items()):
            if not any(c["units"] or c["skipped"] for c in group["cells"][lab].values()):
                continue                                      # nothing played (``d1h_input`` has no such final)
            for ts, c in sorted(group["cells"][lab].items()):
                out[side].setdefault(seed, {})[ts] = [int(c["wins"]), int(c["n"])]
                out["ties"][side].setdefault(seed, {})[ts] = int(c["ties"])
                out["units"][side].setdefault(seed, {})[ts] = {"counted": int(c["units"]),
                                                               "skipped": list(c["skipped"])}
    return out


def p_input(group: Mapping[str, Any], spec: Any, on: str, off: str) -> Dict[str, Any]:
    """D1's P (``d1.p_input``'s shape) from an assembled S group on a registered panel."""
    sides = _seed_side(spec, list(group["nodes"]), on, off)
    out: Dict[str, Any] = {"on": {}, "off": {}, "inconclusive": list(group["inconclusive"]), "per_final": {},
                           "panel": group["panel"], "team_set": group["team_set"]}
    for side in ("on", "off"):
        for seed, lab in sorted(sides[side].items()):
            node = group["nodes"][lab]
            out[side][seed] = node["s_pp"]
            out["per_final"][lab] = node["population_score"]
    return out


def _d1_inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    return {"arm": p["arm"], "look": p["look"],
            "h": h_input(_group(cells, "anchor", p["anchor_group"]), spec, p["on"], p["off"]),
            "p": p_input(_group(cells, "s", p["s_group"]), spec, p["on"], p["off"])}


def _d1_decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    return guards_not_read(_jsonable(D.read_d1(doc).as_dict()), p.get("guards_not_read"))


# --------------------------------------------------------------------------------------- anchor_trajectory
_AT_KEYS = ("on", "off", "anchor_group", "panels", "unit_seeds", "min_unit_frac", "goal_bar", "subset_below",
            "s1", "s_baseline", "k9b", "verdict_text")
#: the verdict's words (a read may register its own, e.g. x26c's "goal (b)")
AT_VERDICT_TEXT = {"met": "GOAL MET", "not_met": "goal not met"}
VOID = "VOID"


def _at_check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    probs = _check_keys(p, ("on", "off", "anchor_group", "goal_bar", "min_unit_frac"), _AT_KEYS)
    a = anchor.get(p.get("anchor_group"))
    if a is None:
        probs.append(f"anchor_group {p.get('anchor_group')!r} is not an anchor group")
    for k in ("on", "off"):
        if p.get(k) not in models:
            probs.append(f"{k} {p.get(k)!r} is not a model label")
    s1 = p.get("s1")
    if s1 is not None and (not isinstance(s1, dict) or s1.get("group") not in s or s1.get("label") not in models):
        probs.append("s1: {group: <S group>, label: <model>}")
    b = p.get("s_baseline")
    if s1 is not None and not (isinstance(b, dict) and isinstance(b.get("s_pp"), (int, float))
                               and isinstance(b.get("se_pp"), (int, float))):
        probs.append("s_baseline: {s_pp, se_pp, source} (the banked node S1 is read against)")
    return probs


def _at_inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    g = _group(cells, "anchor", p["anchor_group"])
    a = spec.anchor[p["anchor_group"]]
    doc: Dict[str, Any] = {"on": p["on"], "off": p["off"], "panels": list(p.get("panels") or a.teamsets),
                           "unit_seeds": list(p.get("unit_seeds") or a.unit_seeds),
                           "units": list(g["units"]),
                           "s1": None, "k9b_verdict": None}
    doc["units_planned"] = len(doc["panels"]) * len(doc["unit_seeds"])
    if p.get("s1"):
        sg = _group(cells, "s", p["s1"]["group"])
        node = sg["nodes"][p["s1"]["label"]]
        bad = node["inconclusive"] or sg.get("inconclusive")
        doc["s1"] = None if bad else {"s_pp": node["s_pp"], "se_pp": node["se_pp"]}
    if p.get("k9b"):
        from main.reads.models import spec_path

        doc["k9b_verdict"] = json.loads(spec_path(str(p["k9b"])).read_text())["verdict"]
    return doc


def anchor_block(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    """x26c ``verdict.kakuna_block``: per label the stratified score over COUNTED units, Δ = Newcombe of strata,
    the unit-paired difference and the low-seed subset (both descriptive)."""
    on, off, panels = doc["on"], doc["off"], list(doc["panels"])
    pooled: Dict[str, Dict[str, Dict[str, Any]]] = {}
    per: Dict[Tuple[str, str, int], float] = {}
    for u in doc["units"]:
        c = pooled.setdefault(u["label"], {}).setdefault(u["panel"], {"wins": 0, "ties": 0, "n": 0, "units": 0,
                                                                          "skipped": []})
        if not u["counted"]:
            c["skipped"].append(u["unit"])
            continue
        c["units"] += 1
        c["wins"] += int(u["wins"])
        c["ties"] += int(u.get("ties", 0))
        c["n"] += int(u["n"])
        per[(u["label"], u["panel"], int(u["unit_seed"]))] = int(u["wins"]) / int(u["n"])
    out: Dict[str, Any] = {}
    for lab in (on, off):
        cells = pooled.get(lab) or {}
        missing = [k for k in panels if k not in cells]
        if missing:
            raise RuleInputError(f"{lab}: no units on {missing}")
        ss = strat_score({k: (cells[k]["wins"], cells[k]["n"]) for k in panels}, panels)
        out[lab] = dict(ss.as_dict(), units={k: cells[k]["units"] for k in panels},
                        skipped={k: cells[k]["skipped"] for k in panels}, ties={k: cells[k]["ties"] for k in panels},
                        units_counted=sum(cells[k]["units"] for k in panels))
    # the panels are passed: x26c's verdict.py omitted them, correct only because its panels were d1's default
    d, lo, hi = newcombe_strat(strat_score({k: (pooled[on][k]["wins"], pooled[on][k]["n"]) for k in panels}, panels),
                               strat_score({k: (pooled[off][k]["wins"], pooled[off][k]["n"]) for k in panels}, panels))
    out["delta"] = {"d": d, "lo": lo, "hi": hi}
    pu = paired_unit_diff(per, on, off, panels, doc["unit_seeds"])
    if pu is not None:
        out["paired_units"] = pu
    below = p.get("subset_below")
    if below is not None:
        sub = {}
        for lab in (on, off):
            w = {k: [0, 0] for k in panels}
            for u in doc["units"]:
                if u["label"] == lab and int(u["unit_seed"]) < int(below) and u["counted"]:
                    w[u["panel"]][0] += int(u["wins"])
                    w[u["panel"]][1] += int(u["n"])
            if all(n for _x, n in w.values()):
                sub[lab] = strat_score({k: (v[0], v[1]) for k, v in w.items()}, panels).as_dict()
        out[f"seeds_0_{int(below) - 1}"] = sub
    return out


def _at_decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    """x26c PREREG §5 (``verdict.decide``): VOID on a non-PASS K9(b) or < ``min_unit_frac`` units counted; goal
    met iff the ON score's Wilson LB clears ``goal_bar``; CLIMBING iff Δ's LB clears 0, else FLAT; S1 RISING iff
    (S1 − baseline)'s normal LB clears 0 (se's added in quadrature), read only when FLAT (``d3_revisit``)."""
    k = anchor_block(doc, p)
    on, off = doc["on"], doc["off"]
    reasons = []
    if p.get("k9b") and doc.get("k9b_verdict") != "PASS":
        reasons.append(f"K9(b) {doc.get('k9b_verdict')}")
    for lab in (on, off):
        if k[lab]["units_counted"] < float(p["min_unit_frac"]) * int(doc["units_planned"]) - RULE8_EPS:
            reasons.append(f"{lab}: {k[lab]['units_counted']} / {doc['units_planned']} units counted "
                           f"(< {float(p['min_unit_frac']):.0%})")
    if reasons:
        return {"kakuna": k, "decision": {"verdict": VOID, "reasons": reasons}}
    goal = clears(k[on]["wilson95"][0], float(p["goal_bar"]))[0]
    climbing = clears(k["delta"]["lo"], 0.0)[0]
    dec: Dict[str, Any] = {"goal_met": goal, "trajectory": "CLIMBING" if climbing else "FLAT"}
    s1, base = doc.get("s1"), p.get("s_baseline")
    if p.get("s1") is not None:
        if s1 is not None and s1.get("se_pp") is not None:
            d = s1["s_pp"] - base["s_pp"]
            se = math.sqrt(s1["se_pp"] ** 2 + base["se_pp"] ** 2)
            rising = clears(d - Z95 * se, 0.0)[0]
            dec["s1"] = {"s_pp": s1["s_pp"], "se_pp": s1["se_pp"], "baseline": dict(base), "d_vs_baseline": d,
                         "lo": d - Z95 * se, "hi": d + Z95 * se, "rising": rising}
            if not climbing:
                dec["d3_revisit"] = rising
        else:
            dec["s1"] = None
            if not climbing:
                dec["d3_revisit"] = "UNREAD (S1 missing)"
    words = dict(AT_VERDICT_TEXT, **(p.get("verdict_text") or {}))
    dec["verdict"] = f"{words['met'] if goal else words['not_met']}; {dec['trajectory']}"
    return {"kakuna": k, "decision": dec}


# ------------------------------------------------------------------------------------------------ registry
@dataclass(frozen=True)
class Rule:
    name: str
    doc: str
    check_params: Callable[..., List[str]]
    inputs: Callable[[Mapping[str, Any], Mapping[str, Any], Any], Dict[str, Any]]
    decide: Callable[[Mapping[str, Any], Mapping[str, Any]], Dict[str, Any]]
    #: an X35 stage record (top-level arm / look / verdict / primary; ``batch.read_record`` loads it)
    x35_record: bool

    def verdict_of(self, decision: Mapping[str, Any]) -> str:
        v = decision.get("verdict") if self.x35_record else (decision.get("decision") or {}).get("verdict")
        if not isinstance(v, str) or not v:
            raise RuleInputError(f"{self.name}: the decision carries no verdict")
        return v


RULES: Dict[str, Rule] = {
    "x35_r1": Rule("x35_r1", "X35 R1 / R2 / guards (main.x35.readout.read)", _r1_check, _r1_inputs, _r1_decide, True),
    "d1": Rule("d1", "D1 H (unfamiliar Kakuna panels, φ̂-widened) + P (main.x35.d1.read_d1)", _d1_check, _d1_inputs,
               _d1_decide, True),
    "anchor_trajectory": Rule("anchor_trajectory", "x26c PREREG §5: goal bar, CLIMBING / FLAT, S1 rising",
                              _at_check, _at_inputs, _at_decide, False),
}


def inputs(rule: str, cells: Mapping[str, Any], params: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    return RULES[rule].inputs(cells, params, spec)


def decide(rule: str, doc: Mapping[str, Any], params: Mapping[str, Any]) -> Dict[str, Any]:
    return RULES[rule].decide(doc, params)
