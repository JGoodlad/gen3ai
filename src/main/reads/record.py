"""The READ RECORD: one schema-checked JSON per read (``designs/tools/reads.md`` §6).

    {"schema": "gen3_read_record_v1", "read", "rule", "spec_sha256", "prereg", "verdict", "created",
     "models": {label: {arm, seed, sha256, step, source}},          every model, by the bytes that were played
     "regimes": {"anchor": {group: id}, "s": {group: id}},          ALWAYS present: a record without one refuses
     "ledgers": {group: {family, digest, rows}},                    the S reads' eval-ledger digests
     "code": {commit, dirty, pin},                                  the playing checkout (``code.json``)
     "repins": [{from_commit, to_commit, playpath_digest, by, why, time, ...}],   each playing commit a repin accepted
                                                                    (``main.reads.repin``; [] = the read played at ``code``)
     "cells_inconclusive": [...],                                   what assemble could not count, verbatim
     "reported_incomplete": {"<lane>.<group>": [...]},              REPORTED-only groups not complete (never block)
     "void": {label: {run, k9b, batch2, pass}} | null,              the registered VOID gate, every run PASS
     "imports": {"<lane>.<group>": {state, spec_sha256, ...}},      cells imported from an earlier read's state
     "completeness": {"class": "COMPLETE"} |                        every planned unit of every decided group was played, or
                    {"class": "INCOMPLETE_BY_STOP", "opt_in": "decision.allow_partial_at_stop",   the spec REGISTERED a reduced read
                     "groups": {"anchor.<g>": {planned, played, unplayed, counted}},               and the lane ended at its
                     "stops": {"anchor": {event: "stop_at", time, left, stop_epoch}}}              registered stop (absent = COMPLETE)
     "inputs": {...}, "decision": {...}}                            the rule's input and output, as decided

An X35-stage rule (``Rule.x35_record``: ``x35_r1``, ``d1``) also carries its decision's keys at the TOP level (``arm``,
``look``, ``verdict``, ``r1`` / ``h`` / ``p``, ``guards``, ``reasons``, ``reported``), so ``main.x35.batch.read_record``
and ``final_class`` load the file unchanged when it is copied to ``<stage>/reads/<arm>_look<k>.json``. The research
ledger cites a record by its path and :func:`summary_line`.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCHEMA = "gen3_read_record_v1"
COMPLETE = "COMPLETE"
#: The class of a record decided on the counted units of a lane that ended at its REGISTERED stop with units unplayed; only
#: a spec with ``decision.allow_partial_at_stop: true`` can write it (:data:`OPT_IN`).
INCOMPLETE_BY_STOP = "INCOMPLETE_BY_STOP"
OPT_IN = "decision.allow_partial_at_stop"
REQUIRED = ("schema", "read", "rule", "spec_sha256", "verdict", "created", "models", "regimes", "ledgers", "code",
            "cells_inconclusive", "inputs", "decision")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RID = re.compile(r"^[0-9a-f]{16}$")


class RecordError(ValueError):
    """A record that is not a record (every problem listed)."""


def build(spec: Any, cells: Mapping[str, Any], inputs: Mapping[str, Any], decision: Mapping[str, Any],
          code: Mapping[str, Any], *, now: Optional[str] = None,
          void: Optional[Mapping[str, Any]] = None, repins: Sequence[Mapping[str, Any]] = (),
          partial: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """``partial``: ``None`` (every planned unit played) or ``{groups, stops}`` of :func:`main.reads.api.record` (the decided
    anchor groups with units unplayed, and each lane's registered-stop evidence): the record is then
    :data:`INCOMPLETE_BY_STOP`, which the spec must have registered (``allow_partial_at_stop``), and only those groups'
    unplayed units are left out of ``cells_inconclusive``."""
    from main.reads.assemble import conclusive, reported_incomplete
    from main.reads.rules import RULES

    rule = RULES[spec.rule]
    reported = rule.reported(spec.params)
    comp: Dict[str, Any] = {"class": COMPLETE}
    if partial:
        if not getattr(spec, "allow_partial_at_stop", False):
            raise RecordError(f"a {INCOMPLETE_BY_STOP} record needs the spec's {OPT_IN}: true (registered before any "
                              "counted game); this spec registers none")
        comp = {"class": INCOMPLETE_BY_STOP, "opt_in": OPT_IN, "groups": dict(partial["groups"]),
                "stops": dict(partial["stops"])}
    allow = [tuple(k.split(".", 1)) for k in (comp.get("groups") or {})]
    models = {lab: {k: e.get(k) for k in ("arm", "seed", "sha256", "step", "source")}
              for lab, e in (cells.get("models") or {}).items()}
    rec: Dict[str, Any] = {
        "schema": SCHEMA, "read": spec.read, "rule": spec.rule, "spec_sha256": spec.sha256, "prereg": spec.prereg,
        "verdict": rule.verdict_of(decision), "created": now or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": models,
        "regimes": {lane: {g: grp.get("regime_id") for g, grp in (cells.get(lane) or {}).items()}
                    for lane in ("anchor", "s")},
        "ledgers": {g: {"family": grp.get("family"), "digest": grp.get("ledger_digest"), "rows": grp.get("rows")}
                    for g, grp in (cells.get("s") or {}).items()},
        "code": dict(code), "repins": [dict(r) for r in repins],
        "cells_inconclusive": conclusive(cells, reported=reported, allow_unplayed=allow),
        "completeness": comp,
        "reported_incomplete": reported_incomplete(cells, reported),
        "void": None if void is None else dict(void),
        "imports": {f"{lane}.{g}": grp["imports"] for lane in ("anchor", "s")
                    for g, grp in (cells.get(lane) or {}).items() if grp.get("imports")},
        "inputs": dict(inputs), "decision": dict(decision)}
    if rule.x35_record:
        for k, v in decision.items():
            if k != "verdict":
                rec[k] = v
    probs = check(rec)
    if probs:
        raise RecordError(f"the record does not check: {probs}")
    return rec


def check(rec: Mapping[str, Any]) -> List[str]:
    """Every way ``rec`` is not a read record (empty = it is)."""
    from main.reads.rules import RULES

    probs = [f"missing {k!r}" for k in REQUIRED if k not in rec]
    if probs:
        return probs
    if rec["schema"] != SCHEMA:
        probs.append(f"schema {rec['schema']!r}")
    rule = RULES.get(str(rec["rule"]))
    if rule is None:
        return probs + [f"rule {rec['rule']!r} is not registered"]
    try:
        if rule.verdict_of(rec["decision"]) != rec["verdict"]:
            probs.append(f"verdict {rec['verdict']!r} is not the decision's {rule.verdict_of(rec['decision'])!r}")
    except ValueError as exc:
        probs.append(str(exc))
    if not rec["models"]:
        probs.append("no model")
    for lab, m in rec["models"].items():
        if not (isinstance(m.get("sha256"), str) and _SHA256.match(m["sha256"])):
            probs.append(f"model {lab}: no sha256 (it was never resolved)")
    groups = [(lane, g, rid) for lane in ("anchor", "s") for g, rid in (rec["regimes"].get(lane) or {}).items()]
    if not groups:
        probs.append("no cell group")
    incomplete = rec.get("reported_incomplete") or {}
    for lane, g, rid in groups:
        if f"{lane}.{g}" in incomplete and rid is None:
            continue                     # a REPORTED-only group recorded INCOMPLETE (its reasons say why)
        if not (isinstance(rid, str) and _RID.match(rid)):
            probs.append(f"{lane} group {g}: regime id {rid!r} (the regime is ALWAYS recorded)")
    for g, led in rec["ledgers"].items():
        if not led.get("digest"):
            probs.append(f"S group {g}: no ledger digest")
    if rec["cells_inconclusive"]:
        # a verdict decided on cells assemble could not count is not a verdict (a wrong regime, unplayed units)
        probs.append(f"cells not conclusive: {list(rec['cells_inconclusive'])[:5]}")
    from main.reads.rules import BATCH2_VOID_ARM, void_failures

    if rec.get("void") is not None and void_failures(rec["void"]):
        probs.append(f"the registered VOID gate fails: {void_failures(rec['void'])}")
    if rule.x35_record and rec.get("arm") in BATCH2_VOID_ARM and not rec.get("void"):
        probs.append(f"arm {rec.get('arm')}: no VOID gate recorded (V-S1 / V-S2 / LR-F and K9(b) gate this read)")
    if not (isinstance(rec["code"], dict) and rec["code"].get("commit")):
        probs.append("code: no playing commit")
    probs += _check_completeness(rec.get("completeness"))
    from main.reads.repin import REQUIRED as REPIN_KEYS

    for i, r in enumerate(rec.get("repins") or ()):                # absent on a record written before repin existed
        miss = [k for k in REPIN_KEYS if not (isinstance(r, dict) and r.get(k))]
        if miss:
            probs.append(f"repins[{i}]: lacks {miss}")
    if rule.x35_record:
        for k in ("arm", "look"):
            if k not in rec or rec[k] != rec["decision"].get(k):
                probs.append(f"x35 record: top-level {k!r} is not the decision's")
    return probs


def _check_completeness(comp: Any) -> List[str]:
    """Every way a record's ``completeness`` is not one (absent = a record written before it existed = COMPLETE)."""
    if comp is None:
        return []
    if not isinstance(comp, dict) or comp.get("class") not in (COMPLETE, INCOMPLETE_BY_STOP):
        return [f"completeness: {comp!r} is neither {COMPLETE} nor {INCOMPLETE_BY_STOP}"]
    if comp["class"] == COMPLETE:
        return [] if not comp.get("groups") else ["completeness: COMPLETE with unplayed groups"]
    probs = []
    if comp.get("opt_in") != OPT_IN:
        probs.append(f"completeness: {INCOMPLETE_BY_STOP} without {OPT_IN} (the spec must register a reduced read)")
    groups, stops = comp.get("groups") or {}, comp.get("stops") or {}
    if not groups:
        probs.append(f"completeness: {INCOMPLETE_BY_STOP} names no group with units unplayed")
    for k, g in groups.items():
        if not (isinstance(g, dict) and int(g.get("unplayed") or 0) > 0):
            probs.append(f"completeness.groups[{k}]: no unplayed unit")
        if k.split(".", 1)[0] not in stops:
            probs.append(f"completeness.groups[{k}]: its lane has no recorded stop")
    for lane, st in stops.items():
        if not (isinstance(st, dict) and st.get("event") == "stop_at"):
            probs.append(f"completeness.stops[{lane}]: not a registered stop (stop_at)")
    return probs


def write(rec: Mapping[str, Any], path: Path) -> Path:
    probs = check(rec)
    if probs:
        raise RecordError(f"{path}: refused: {probs}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return path


def load(path: Path) -> Dict[str, Any]:
    rec = json.loads(Path(path).read_text())
    probs = check(rec)
    if probs:
        raise RecordError(f"{path}: {probs}")
    return dict(rec)


def summary_line(rec: Mapping[str, Any]) -> str:
    """The one line a ledger entry quotes: read, verdict, regimes, model sha prefixes."""
    regs = ", ".join(f"{lane}.{g}={rid}" for lane in ("anchor", "s") for g, rid in sorted(rec["regimes"][lane].items()))
    mods = ", ".join(f"{lab}={m['sha256'][:8]}" for lab, m in sorted(rec["models"].items()))
    comp = rec.get("completeness") or {}
    part = ""
    if comp.get("class") == INCOMPLETE_BY_STOP:
        n = {k: (g["played"], g["planned"]) for k, g in comp["groups"].items()}
        part = f" [{INCOMPLETE_BY_STOP}: " + ", ".join(f"{k} {a} of {b} units played" for k, (a, b) in sorted(n.items())) + "]"
    return f"{rec['read']}: {rec['verdict']}{part} (rule {rec['rule']}; regimes {regs}; models {mods})"
