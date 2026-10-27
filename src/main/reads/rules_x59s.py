r"""X59-S's registered reads as ``main.reads`` rules (``x59s_matched_students_2026-10-25/PREREG.md`` §4–§5,
Amendments 2, 4 and 5). The statistics are :mod:`main.human_data.x59s_read`'s; this module only gathers
their inputs from the assembled Kakuna cells and the read's sha-recorded artefacts.

* ``x59s_floor`` -- the floor pilot as first registered (every arm's mean U rate ``< floor_bar``). SUPERSEDED before
  it was ever decided by Amendment 5 A5.1 (REVIEW H B1): kept so the pilot state's spec still loads.
* ``x59s_gate`` -- the per-contrast floor gate (A5.1) on the pilot and repeat groups IMPORTED from the pilot state: a
  contrast is UNINFORMATIVE iff both its arms' pilot means are <= the better floor reference + ``band``; it releases
  ONLY those booleans and whether the read continues (A2.11 m3), with VOID 6 (the repeat, game for game) and one
  engine for the pilot and the floor references.
* ``x59s`` -- the FULL READ: each primary student's U counts over the named groups, σ̂ and the three contrasts with
  A5.2's widened interval (:func:`x59s_read.read`), the in-house head-to-heads (:func:`x59s_read.inhouse`), the gate's
  booleans, forum72's own contrasts, the yardsticks, S-H's fit, the re-encode share and FR-2's class, through
  :func:`x59s_read.verdict` (Amendments 2 and 5). VOID 3, 5, 6 (the gate's), 7, a second engine (the in-house core
  stamps included, m5), a missing FR-2 class or re-encode share, or short in-house cells refuse a verdict (``VOID``).
  Descriptive, read by no rule: each panel's contrasts with intervals, the paired per-seed contrasts, the MDE at the
  measured interval, the student-to-P\* gap (secondary, cross-engine), the arms' SDs, the yardsticks, m5's flags.

A student's rate is the equal-weight mean of its three panels' (wins / games); a tie is a non-win (X51's rule).
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

FLOOR_FIRES = "FLOOR FIRES"
FLOOR_HOLDS = "FLOOR DOES NOT FIRE"
VOID = "VOID"
ARMS = ("S-H", "S-P", "S-S")
ARTEFACTS = ("students", "floor_refs", "inhouse", "fr2", "identity", "gate_record", "m3_reencode")
_FLOOR_KEYS = ("group", "panels", "arms", "floor_bar", "repeat")
_GATE_KEYS = ("group", "panels", "arms", "floor_refs", "band", "repeat")
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
    """VOID 6: each unit of the repeat group against the SAME unit of the student it repeats, game for game."""
    from main.reads.rules import _group

    orig = {(u["panel"], int(u["unit_seed"])): u for u in _group(cells, "anchor", of_group)["units"]
            if u["label"] == r["of"]}
    out = []
    for u in _group(cells, "anchor", r["group"])["units"]:
        o = orig.get((u["panel"], int(u["unit_seed"])))
        out.append({"unit": u["unit"], "of": None if o is None else o["unit"], "counted": bool(u["counted"]),
                    "of_counted": bool(o and o["counted"]),
                    "games_equal": bool(o is not None and u.get("games") is not None and u["games"] == o.get("games"))})
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


# ------------------------------------------------------------------------- the gate (Amendment 5 A5.1, replaces it)
def _gate_check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    from main.reads.rules import _check_keys

    probs = _check_keys(p, _GATE_KEYS, _GATE_KEYS)
    fp = dict(p)
    fp.pop("floor_refs", None)
    fp.pop("band", None)
    fp["floor_bar"] = 0.5                       # the shared checks of the group, the arms and the repeat
    probs += [x for x in _floor_check(fp, models=models, anchor=anchor, s=s) if "missing" not in x
              and "unknown" not in x]
    if not isinstance(p.get("floor_refs"), str):
        probs.append("floor_refs: the path of x59s_ops floor's JSON")
    if not 0.0 < float(p.get("band", 0) or 0) < 1.0:
        probs.append("band in (0, 1)")
    return probs


def _gate_inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    doc = _floor_inputs(cells, p, spec)
    body, sha = _artefact(p["floor_refs"])
    doc["floor_refs"] = {"path": p["floor_refs"], "sha256": sha, "rates": body.get("rates"),
                         "uncounted": body.get("uncounted"), "engines": body.get("engines")}
    from main.reads.rules import _group

    doc["engines"] = {g: _group(cells, "anchor", g).get("engine") for g in (p["group"], p["repeat"]["group"])}
    return doc


def _gate_decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    """A5.1: one UNINFORMATIVE boolean per contrast and whether the rest of the read plays, and NOTHING else (A2.11
    m3); VOID 6 (the repeat, game for game) and one engine for the pilot and the references."""
    from main.human_data import x59s_read as X

    reasons: List[str] = []
    for c in doc["repeat"]:
        if not (c["counted"] and c["of_counted"]):
            reasons.append(f"VOID 6: {c['unit']} or its original {c['of']} is not counted")
        elif not c["games_equal"]:
            reasons.append(f"VOID 6: {c['unit']} does not reproduce {c['of']} game for game")
    if not doc["repeat"]:
        reasons.append("VOID 6: the repeat unit was not played")
    fl = doc["floor_refs"]
    if fl.get("uncounted") or not fl.get("rates") or any(r is None for r in fl["rates"].values()):
        reasons.append(f"the floor references are short: {fl.get('uncounted')}")
    engines = set(doc["engines"].values()) | set(fl.get("engines") or [])
    if len(engines) != 1 or None in engines:
        reasons.append(f"one engine per read (A4.1): {sorted(map(str, engines))}")
    if _no_games(doc):
        reasons.append(f"cells with no counted game: {_no_games(doc)}")
    short = _short_cells(doc)
    if reasons:
        return {"gate": None, "short_cells": short, "decision": {"verdict": VOID, "reasons": reasons}}
    g = X.gate(_arms_counts(doc), dict(fl["rates"]), float(p["band"]))
    words = ", ".join(f"{k} {'UNINFORMATIVE' if v else 'informative'}" for k, v in g["uninformative"].items())
    return {"gate": {"uninformative": g["uninformative"], "continue": g["continue"]}, "short_cells": short,
            "repeat_pass": True, "engine": next(iter(engines)),
            "decision": {"verdict": f"{'CONTINUE' if g['continue'] else 'STOP'}: {words}"}}


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


def _fit_drop(sdoc: Mapping[str, Any], labels: Sequence[str]) -> float:
    """A5.6: the mean over ``labels`` of the drop in validation NLL over the last quarter epoch (``val_hist``'s last
    two points; positive = still falling)."""
    drops = [float(sdoc["students"][lab]["val_hist"][-2][1]) - float(sdoc["students"][lab]["val_hist"][-1][1])
             for lab in labels]
    return sum(drops) / len(drops)


def _panel_contrasts(arms: Mapping[str, List[Dict[str, Tuple[int, int]]]], panels: Sequence[str], sd: Dict[str, float]
                     ) -> Dict[str, Dict[str, Any]]:
    """Per panel, each contrast with A5.2's interval on that panel's games alone (descriptive, no claim)."""
    from main.human_data import x59s_read as X

    out: Dict[str, Dict[str, Any]] = {}
    for ts in panels:
        one = {a: [{ts: s[ts]} for s in v] for a, v in arms.items()}
        out[ts] = {name: {**X.contrast(one[a], one[b], sd, names), "stamp": "descriptive, no claim"}
                   for name, (a, b, names) in X.CONTRASTS.items()}
    return out


def _paired(arms: Mapping[str, List[Dict[str, Tuple[int, int]]]]) -> Dict[str, Dict[str, Any]]:
    """m3: the per-seed paired contrast (students share an init per seed), a t interval df k - 1; descriptive."""
    from scipy.stats import t as tdist

    from main.human_data import x59s_read as X

    out = {}
    for name, (a, b, _) in X.CONTRASTS.items():
        d = [X.student_rate(x) - X.student_rate(y) for x, y in zip(arms[a], arms[b])]
        m = sum(d) / len(d)
        hw = tdist.ppf(0.975, len(d) - 1) * math.sqrt(X._var(d) / len(d))
        out[name] = {"delta": m, "lo": m - hw, "hi": m + hw, "stamp": "descriptive, no claim"}
    return out


def _full_decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    from main.human_data import x59s_read as X

    body = doc["artefact"]
    reasons: List[str] = []
    sdoc = body["students"]
    reasons += [f"VOID 3: {r}" for r in sdoc.get("void3") or []]
    if not body["identity"].get("pass"):
        reasons.append(f"VOID 5: the identity unit does not reproduce X51 ({body['identity'].get('unit')})")
    gdec = body["gate_record"].get("decision") or {}
    gate = gdec.get("gate")
    if not (isinstance(gate, dict) and isinstance(gate.get("uninformative"), dict)):
        reasons.append(f"VOID 6 / the gate: its record carries no per-contrast booleans ({gdec.get('decision')})")
    short = _short_cells(doc)
    if len(short) >= 2:
        reasons.append(f"VOID 7: {len(short)} cells lost more than one unit {short}")
    empty = _no_games(doc)
    if empty:
        reasons.append(f"cells with no counted game: {empty}")
    ih = body["inhouse"]
    engines = (set(doc.get("engines", {}).values()) | set(body["floor_refs"].get("engines") or [])
               | ({gdec.get("engine")} if gdec.get("engine") else set()))
    if len(engines) != 1 or None in engines:
        reasons.append(f"one engine per read (A4.1): the Kakuna groups, the gate and the floor references played on "
                       f"{sorted(map(str, engines))}")
    ih_env = set(ih.get("core_stamps") or [])
    if len(engines) == 1 and ih_env != {str(next(iter(engines))).split(";")[-1]}:
        reasons.append(f"the in-house head-to-heads' core stamps {sorted(ih_env)} are not the read's one engine (m5)")
    fr2 = body["fr2"].get("class")
    if fr2 not in X.FR2_CLASSES:
        reasons.append(f"FR-2 carries no class ({fr2!r}): Delta_data is not reported without it (A2.5)")
    if ih.get("short") or any(r is None for v in (ih.get("rates") or {}).values() for r in v):
        reasons.append(f"the in-house head-to-heads are short: {ih.get('short')}")
    m3 = body["m3_reencode"].get("share_differing")
    if not isinstance(m3, (int, float)):
        reasons.append("A5.7: the re-encode check carries no share_differing")
    out: Dict[str, Any] = {"short_cells": short}
    if reasons:
        out["decision"] = {"verdict": VOID, "reasons": reasons}
        return out
    arms = _arms_counts(doc)
    read_ = X.read(arms)
    sd = X.seed_sd(arms)
    names = {"delta_data": X.CONTRASTS["delta_data"][2], "delta_labels": X.CONTRASTS["delta_labels"][2]}
    inh = {c: X.inhouse(ih["rates"][c], names[c]) for c in names}
    lab_of = {lab: a for a, labs in doc["arms"].items() for lab in labs}
    yard_all = {lab: s["yardsticks"] for lab, s in sdoc["students"].items() if lab in lab_of}
    yard = {a: [yard_all[lab]["selfplay_val_pstar_labels"] for lab in doc["arms"][a]] for a in ARMS}
    panels = _panel_contrasts(arms, doc["panels"], sd)
    f_delta = {c: panels["forum72"][c]["delta"] for c in names}
    fit = _fit_drop(sdoc, doc["arms"]["S-H"])
    v = X.verdict(read_, inh, dict(gate["uninformative"]), f_delta, yard, fr2, fit, float(m3))
    rate = {a: [X.student_rate(s) for s in arms[a]] for a in ARMS}
    mde = {name: (1.959963985 + 0.841621234) * read_[name]["halfwidth"] / 1.959963985 for name in X.CONTRASTS}
    out.update({
        "engine": next(iter(engines)), "read": read_, "inhouse": inh, "gate": gate, "fr2": fr2,
        "sh_fit_drop": fit, "reencode_share": m3, "verdict_detail": v,
        "descriptive": {
            "stamp": "descriptive, no claim (A5.9 m14)",
            "arm_rates": {a: {"mean": sum(r) / len(r), "sd": math.sqrt(X._var(r)), "students": r}
                          for a, r in rate.items()},
            "per_panel": panels, "paired_per_seed": _paired(arms), "mde80_at_measured": mde,
            "student_to_pstar_gap_secondary_cross_engine": {a: qh10(doc, a) for a in ARMS},
            "yardsticks": {lab: {"arm": lab_of[lab], **y} for lab, y in yard_all.items()},
            "val_rose_last_epoch": sorted(lab for lab, s in sdoc["students"].items()
                                          if lab in lab_of and s.get("val_rose_last_epoch")),
            "forum72_regime": "the legacy Kakuna peer pack (before b0441623e); no later F read is set against it"}})

    def _c(name: str) -> str:
        c = read_[name]
        base = ", ".join(f"{a} {m:.3f}" for a, m in zip(X.CONTRASTS[name][:2], c["arm_means"]))
        cls = v["classes"].get(name, c["class"])
        if cls == X.UNINF:
            cls = "UNINFORMATIVE (both arms at the floor)"
        return (f"{name} {cls} {100 * c['delta']:+.2f} pp [{100 * c['lo']:+.2f}, {100 * c['hi']:+.2f}] "
                f"(Kakuna {v['kakuna_classes'][name]}; {base}; in-house {100 * inh[name]['delta']:+.2f} pp "
                f"[{100 * inh[name]['lo']:+.2f}, {100 * inh[name]['hi']:+.2f}])")

    words = [_c("delta_data"), _c("delta_labels"),
             f"delta_states {read_['delta_states']['class']} {100 * read_['delta_states']['delta']:+.2f} pp", f"FR-2 {fr2}"]
    out["decision"] = {"verdict": "; ".join(words), "classes": v["classes"], "actions": v["actions"],
                       "uninformative": v["uninformative"]}
    return out


RULE_DOCS = {
    "x59s_gate": "X59-S's per-contrast floor gate (Amendment 5 A5.1) on the imported pilot; VOID 6 repeat; booleans only",
    "x59s_floor": "X59-S's floor pilot: every arm's U mean < floor_bar fires the in-house fallback; VOID 6 repeat",
    "x59s": "X59-S's full read: x59s_read.read / inhouse / verdict on the Kakuna cells and the read's artefacts",
}
