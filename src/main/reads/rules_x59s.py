r"""X59-S's registered reads as ``main.reads`` rules (``x59s_matched_students_2026-10-25/PREREG.md`` §4–§5,
Amendments 2 and 4). The statistics are :mod:`main.human_data.x59s_read`'s, unchanged; this module only gathers
their inputs from the assembled Kakuna cells and the read's sha-recorded artefacts.

* ``x59s_floor`` -- the FLOOR PILOT (units 0-3 of H, F and P72 per primary student): every arm's mean U rate
  ``< floor_bar`` fires the floor (the in-house rule then decides). It releases ONLY that boolean (A2.11 m3), with
  VOID 6 (a repeat of one student's unit under a second label must equal the original GAME FOR GAME) and the cells
  that lost more than one unit (VOID 7, reported).
* ``x59s`` -- the FULL READ: each primary student's U counts over the named groups (the pilot's imported, the rest
  played), σ̂ and the three contrasts (:func:`x59s_read.read`), the pilot's floor boolean, the in-house
  head-to-heads (:func:`x59s_read.inhouse`), the floor references, the yardsticks and FR-2's class, through
  :func:`x59s_read.verdict` (Amendment 2's overrides). VOID 3 (and the learned-nothing guard), VOID 5 (the identity
  unit), VOID 6 (from the pilot's record) and VOID 7 (two cells with more than one uncounted unit) refuse a verdict
  (``VOID``). Descriptive, read by no rule: Δ per panel, Q-H10 (each arm minus P\*'s X51 twins, unit-paired), each
  arm's student SD, the yardstick table, m5's flags.

A student's rate is the equal-weight mean of its three panels' (wins / games); a tie is a non-win (X51's rule).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

FLOOR_FIRES = "FLOOR FIRES"
FLOOR_HOLDS = "FLOOR DOES NOT FIRE"
VOID = "VOID"
ARMS = ("S-H", "S-P", "S-S")
ARTEFACTS = ("students", "floor_refs", "inhouse", "fr2", "identity", "pilot_record")
_FLOOR_KEYS = ("group", "panels", "arms", "floor_bar", "repeat")
_FULL_KEYS = ("groups", "panels", "arms", "artefacts", "x51")


def _err(msg: str) -> Exception:
    from main.reads.rules import RuleInputError

    return RuleInputError(msg)


def _check_arms(p: Mapping[str, Any], models: Any, groups: Sequence[Any]) -> List[str]:
    probs: List[str] = []
    arms = p.get("arms") or {}
    if set(arms) != set(ARMS):
        return [f"arms: exactly {list(ARMS)}"]
    k = {len(v) for v in arms.values()}
    if len(k) != 1 or k.pop() < 2:
        probs.append("arms: the same number (>= 2) of students in every arm")
    for a, labs in arms.items():
        for lab in labs:
            if lab not in models:
                probs.append(f"arms.{a}: {lab!r} is not a model")
            for g in groups:
                if lab not in g.models:
                    probs.append(f"arms.{a}: {lab!r} is not played by group {g.name!r}")
        seeds = [models[lab].seed for lab in labs if lab in models]
        if len(set(seeds)) != len(seeds):
            probs.append(f"arms.{a}: two students share a seed")
    for g in groups:
        if set(g.teamsets) != set(p.get("panels") or []):
            probs.append(f"group {g.name!r} plays {list(g.teamsets)}, not the panels {p.get('panels')}")
    return probs


def counts(cells: Mapping[str, Any], groups: Sequence[str], labels: Sequence[str], panels: Sequence[str]
           ) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{label: {panel: {wins, n, units, planned, uncounted}}}`` summed over ``groups`` (counted units only)."""
    from main.reads.rules import _group

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for g in groups:
        cg = _group(cells, "anchor", g)["cells"]
        for lab in labels:
            for ts in panels:
                c = cg[lab][ts]
                o = out.setdefault(lab, {}).setdefault(ts, {"wins": 0, "n": 0, "units": 0, "planned": 0})
                o["wins"] += int(c["wins"])
                o["n"] += int(c["n"])
                o["units"] += int(c["units"])
                o["planned"] += int(c["planned"])
    for by in out.values():
        for o in by.values():
            o["uncounted"] = o["planned"] - o["units"]
    return out


def _arms_counts(doc: Mapping[str, Any]) -> Dict[str, List[Dict[str, Tuple[int, int]]]]:
    st = doc["students"]
    return {a: [{ts: (int(st[lab][ts]["wins"]), int(st[lab][ts]["n"])) for ts in doc["panels"]} for lab in labs]
            for a, labs in doc["arms"].items()}


def _short_cells(doc: Mapping[str, Any]) -> List[str]:
    return sorted(f"{lab}/{ts}" for lab, by in doc["students"].items() for ts, c in by.items() if c["uncounted"] > 1)


def _no_games(doc: Mapping[str, Any]) -> List[str]:
    return sorted(f"{lab}/{ts}" for lab, by in doc["students"].items() for ts, c in by.items() if not c["n"])


# ------------------------------------------------------------------------------------------- the floor pilot
def _floor_check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    from main.reads.rules import _check_keys

    probs = _check_keys(p, _FLOOR_KEYS, _FLOOR_KEYS)
    if p.get("group") not in anchor:
        return probs + [f"group {p.get('group')!r} is not an anchor group"]
    probs += _check_arms(p, models, [anchor[p["group"]]])
    r = p.get("repeat") or {}
    if r.get("group") not in anchor or r.get("label") not in models or r.get("of") not in models:
        return probs + ["repeat: {group, label, of}: an anchor group playing a second label of one student"]
    rg, og = anchor[r["group"]], anchor[p["group"]]
    if list(rg.models) != [r["label"]]:
        probs.append(f"repeat: group {r['group']!r} must play exactly {r['label']!r}")
    if r["of"] not in og.models:
        probs.append(f"repeat.of {r['of']!r} is not played by the pilot group")
    if not (set(rg.teamsets) <= set(og.teamsets) and set(rg.unit_seeds) <= set(og.unit_seeds)):
        probs.append("repeat: its units must be units of the pilot group")
    if models[r["label"]].ref.sha256 is None or models[r["label"]].ref.sha256 != models[r["of"]].ref.sha256:
        probs.append("repeat: the label must pin the SAME bytes as the student it repeats")
    if not 0.0 < float(p.get("floor_bar", 0)) < 1.0:
        probs.append("floor_bar in (0, 1)")
    return probs


def repeat_checks(cells: Mapping[str, Any], r: Mapping[str, Any], of_group: str) -> List[Dict[str, Any]]:
    """VOID 6: each unit of the repeat group against the SAME unit of the student it repeats, game for game: the
    5-tuples and each game's choices from both captures (:func:`main.reads.assemble.compare_choices`)."""
    from main.reads.assemble import compare_choices
    from main.reads.rules import _group

    orig = {(u["panel"], int(u["unit_seed"])): u for u in _group(cells, "anchor", of_group)["units"]
            if u["label"] == r["of"]}
    out = []
    for u in _group(cells, "anchor", r["group"])["units"]:
        o = orig.get((u["panel"], int(u["unit_seed"])))
        tuple_eq = bool(o is not None and u.get("games") is not None and u["games"] == o.get("games"))
        ch = {"choices": "not compared", "differing": [], "why": "the 5-tuples differ"}
        if tuple_eq and o is not None:
            ch = compare_choices(Path(u["cell_dir"]) if u.get("cell_dir") else None,
                                 Path(o["cell_dir"]) if o.get("cell_dir") else None, o["games"],
                                 unavailable="the original unit has no cell dir")
        out.append({"unit": u["unit"], "of": None if o is None else o["unit"], "counted": bool(u["counted"]),
                    "of_counted": bool(o and o["counted"]),
                    "games_equal": tuple_eq and ch["choices"] != "differ", "tuple_equal": tuple_eq,
                    "choices": ch["choices"], "choices_differing": ch["differing"], "choices_why": ch["why"]})
    return out


def _floor_inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    labs = [lab for a in ARMS for lab in p["arms"][a]]
    return {"panels": list(p["panels"]), "arms": {a: list(p["arms"][a]) for a in ARMS},
            "students": counts(cells, [p["group"]], labs, p["panels"]),
            "repeat": repeat_checks(cells, p["repeat"], p["group"])}


def floor_fires(doc: Mapping[str, Any], bar: float) -> bool:
    """Every arm's mean U rate is below ``bar`` (rule 8: a mean within 1e-9 of the bar does NOT fire)."""
    from main.human_data.x59s_read import TIE, student_rate

    arms = _arms_counts(doc)
    return all(sum(student_rate(s) for s in v) / len(v) < bar - TIE for v in arms.values())


def _floor_decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    rep = doc["repeat"]
    if not rep:
        reasons.append("VOID 6: the repeat unit was not played")
    for c in rep:
        if not (c["counted"] and c["of_counted"]):
            reasons.append(f"VOID 6: {c['unit']} or its original {c['of']} is not counted")
        elif not c["games_equal"]:
            reasons.append(f"VOID 6: {c['unit']} does not reproduce {c['of']} game for game")
    empty = _no_games(doc)
    if empty:
        reasons.append(f"cells with no counted game: {empty}")
    short = _short_cells(doc)
    if reasons:
        return {"floor": None, "short_cells": short, "decision": {"verdict": VOID, "reasons": reasons}}
    fires = floor_fires(doc, float(p["floor_bar"]))
    # A2.11 m3: the pilot releases ONLY its boolean (no arm rate is written into the decision).
    return {"floor": fires, "short_cells": short, "repeat_pass": True,
            "decision": {"verdict": FLOOR_FIRES if fires else FLOOR_HOLDS}}


# ------------------------------------------------------------------------------------------------ the full read
def _full_check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    from main.reads.rules import _check_keys

    probs = _check_keys(p, _FULL_KEYS, _FULL_KEYS)
    gs = p.get("groups") or []
    if not gs or any(g not in anchor for g in gs):
        return probs + [f"groups {gs}: anchor groups of this spec"]
    probs += _check_arms(p, models, [anchor[g] for g in gs])
    seeds = [set(anchor[g].unit_seeds) for g in gs]
    if any(a & b for i, a in enumerate(seeds) for b in seeds[i + 1:]):
        probs.append("groups: their unit seeds must be disjoint (a unit is counted once)")
    art = p.get("artefacts") or {}
    if set(art) != set(ARTEFACTS) or not all(isinstance(v, str) for v in art.values()):
        probs.append(f"artefacts: a path for each of {list(ARTEFACTS)}")
    x = p.get("x51") or {}
    if not (isinstance(x.get("path"), str) and isinstance(x.get("sha256"), str)
            and set(x.get("cells") or {}) == set(p.get("panels") or [])):
        probs.append("x51: {path, sha256, cells: {panel: read_cell}} for every panel (Q-H10, descriptive)")
    return probs


def _artefact(path: str) -> Tuple[Dict[str, Any], str]:
    from main.reads.models import spec_path
    from utils.hashing import sha256_file

    pp = spec_path(path)
    if not pp.is_file():
        raise _err(f"artefact {path}: missing (it is made before the read is decided)")
    return json.loads(pp.read_text()), sha256_file(pp)


def _unit_rates(cells: Mapping[str, Any], groups: Sequence[str], labels: Sequence[str]
                ) -> Dict[str, Dict[Tuple[str, int], float]]:
    from main.reads.rules import _group

    out: Dict[str, Dict[Tuple[str, int], float]] = {lab: {} for lab in labels}
    for g in groups:
        for u in _group(cells, "anchor", g)["units"]:
            if u["label"] in out and u["counted"] and u.get("n"):
                out[u["label"]][(u["panel"], int(u["unit_seed"]))] = int(u["wins"]) / int(u["n"])
    return out


def _full_inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    from main.reads.rules import banked_units

    labs = [lab for a in ARMS for lab in p["arms"][a]]
    doc: Dict[str, Any] = {"panels": list(p["panels"]), "arms": {a: list(p["arms"][a]) for a in ARMS},
                           "students": counts(cells, p["groups"], labs, p["panels"]), "artefacts": {}}
    doc["artefact"] = {}
    for k, path in p["artefacts"].items():
        body, sha = _artefact(path)
        doc["artefacts"][k] = {"path": path, "sha256": sha}
        doc["artefact"][k] = body
    from main.reads.rules import _group

    doc["engines"] = {g: _group(cells, "anchor", g).get("engine") for g in p["groups"]}
    bank = banked_units(p["x51"])
    rates = _unit_rates(cells, p["groups"], labs)
    doc["unit_rates"] = {lab: {f"{ts}/{s}": r for (ts, s), r in by.items()} for lab, by in rates.items()}
    doc["x51_units"] = {ts: {str(s): int(u["wins"]) / int(u["n"]) for s, u in by.items() if u["n"]}
                        for ts, by in bank.items()}
    return doc


def qh10(doc: Mapping[str, Any], arm: str) -> Dict[str, Any]:
    r"""Q-H10 (descriptive): the arm's mean over students of (student unit rate − P\*'s X51 twin unit rate), per
    (panel, unit seed), then equal weight over panels; a normal interval over the unit seeds within each panel."""
    per_panel: Dict[str, Dict[str, Any]] = {}
    for ts in doc["panels"]:
        bank = doc["x51_units"][ts]
        diffs = []
        for s, b in sorted(bank.items(), key=lambda kv: int(kv[0])):
            xs = [doc["unit_rates"][lab].get(f"{ts}/{s}") for lab in doc["arms"][arm]]
            xs = [x for x in xs if x is not None]
            if xs:
                diffs.append(sum(xs) / len(xs) - b)
        m = sum(diffs) / len(diffs) if diffs else float("nan")
        v = sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1) / len(diffs) if len(diffs) > 1 else float("nan")
        per_panel[ts] = {"delta": m, "var": v, "units": len(diffs)}
    P = len(doc["panels"])
    d = sum(x["delta"] for x in per_panel.values()) / P
    se = math.sqrt(sum(x["var"] for x in per_panel.values())) / P
    return {"delta": d, "lo": d - 1.959963985 * se, "hi": d + 1.959963985 * se, "per_panel": per_panel}


def _full_decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    from main.human_data import x59s_read as X

    body = doc["artefact"]
    reasons: List[str] = []
    sdoc = body["students"]
    reasons += [f"VOID 3: {r}" for r in sdoc.get("void3") or []]
    if not body["identity"].get("pass"):
        reasons.append(f"VOID 5: the identity unit does not reproduce X51 ({body['identity'].get('unit')})")
    pilot = body["pilot_record"]
    pdec = pilot.get("decision") or {}
    floor = pdec.get("floor")
    if not isinstance(floor, bool):
        reasons.append(f"VOID 6 / the pilot: its record carries no floor boolean ({(pdec.get('decision') or {})})")
    short = _short_cells(doc)
    if len(short) >= 2:
        reasons.append(f"VOID 7: {len(short)} cells lost more than one unit {short}")
    empty = _no_games(doc)
    if empty:
        reasons.append(f"cells with no counted game: {empty}")
    engines = set(doc.get("engines", {}).values()) | set(body["floor_refs"].get("engines") or [])
    if len(engines) != 1 or None in engines:
        reasons.append(f"one engine per read (A4.1): the Kakuna groups and floor references played on {sorted(map(str, engines))}")
    fr2 = body["fr2"].get("class")
    if fr2 not in X.FR2_CLASSES:
        reasons.append(f"FR-2 carries no class ({fr2!r}): Delta_data is not reported without it (A2.5)")
    ih = body["inhouse"]
    if ih.get("short") or any(r is None for v in (ih.get("rates") or {}).values() for r in v):
        reasons.append(f"the in-house head-to-heads are short: {ih.get('short')}")
    fl = body["floor_refs"]
    if fl.get("uncounted") or any(r is None for r in (fl.get("rates") or {}).values()):
        reasons.append(f"the floor references are short: {fl.get('uncounted')}")
    out: Dict[str, Any] = {"short_cells": short}
    if reasons:
        out["decision"] = {"verdict": VOID, "reasons": reasons}
        return out
    arms = _arms_counts(doc)
    read_ = X.read(arms)
    read_["floor_on_all_units"] = read_["floor"]
    read_["floor"] = bool(floor)                     # §4: the floor is the PILOT's (units 0-3), not the full read's
    names = {"delta_data": X.CONTRASTS["delta_data"][2], "delta_labels": X.CONTRASTS["delta_labels"][2]}
    inh = {c: X.inhouse(ih["rates"][c], names[c]) for c in names}
    lab_of = {lab: a for a, labs in doc["arms"].items() for lab in labs}
    yard = {lab: s["yardsticks"] for lab, s in sdoc["students"].items() if lab in lab_of}
    sp = [yard[lab]["selfplay_val_pstar_labels"] for lab in doc["arms"]["S-P"]]
    sh = [yard[lab]["selfplay_val_pstar_labels"] for lab in doc["arms"]["S-H"]]
    v = X.verdict(read_, inh, dict(fl["rates"]), sp, sh, fr2)
    rate = {a: [X.student_rate(s) for s in arms[a]] for a in ARMS}
    per_panel = {}
    for ts in doc["panels"]:
        m = {a: sum(w / g for w, g in (s[ts] for s in arms[a])) / len(arms[a]) for a in ARMS}
        per_panel[ts] = {"arm_means": m, "delta_data": m["S-H"] - m["S-S"], "delta_labels": m["S-H"] - m["S-P"],
                         "delta_states": m["S-P"] - m["S-S"]}
    out.update({
        "engine": next(iter(engines)), "read": read_, "inhouse": inh, "floor_rates": fl["rates"], "fr2": fr2, "verdict_detail": v,
        "descriptive": {
            "arm_rates": {a: {"mean": sum(r) / len(r), "sd": math.sqrt(X._var(r)), "students": r}
                          for a, r in rate.items()},
            "per_panel": per_panel,
            "qh10_vs_pstar_x51": {a: qh10(doc, a) for a in ARMS},
            "yardsticks": {lab: {"arm": lab_of[lab], **y} for lab, y in yard.items()},
            "val_rose_last_epoch": sorted(lab for lab, s in sdoc["students"].items()
                                          if lab in lab_of and s.get("val_rose_last_epoch"))}})

    def _c(name: str) -> str:
        c = read_[name]
        return (f"{name} {v['classes'].get(name, c['class'])} {100 * c['delta']:+.2f} pp "
                f"[{100 * c['lo']:+.2f}, {100 * c['hi']:+.2f}]")

    words = [_c("delta_data"), _c("delta_labels"),
             f"delta_states {read_['delta_states']['class']} {100 * read_['delta_states']['delta']:+.2f} pp",
             f"FR-2 {fr2}"]
    if v["uninformative"]:
        words.insert(0, "UNINFORMATIVE")
    if read_["floor"]:
        words.insert(0, "FLOOR (in-house decides)")
    out["decision"] = {"verdict": "; ".join(words), "classes": v["classes"], "actions": v["actions"],
                       "uninformative": v["uninformative"]}
    return out


RULE_DOCS = {
    "x59s_floor": "X59-S's floor pilot: every arm's U mean < floor_bar fires the in-house fallback; VOID 6 repeat",
    "x59s": "X59-S's full read: x59s_read.read / inhouse / verdict on the Kakuna cells and the read's artefacts",
}
