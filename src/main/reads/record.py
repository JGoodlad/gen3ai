"""The READ RECORD: one schema-checked JSON per read (``designs/tools/reads.md`` §6).

    {"schema": "gen3_read_record_v1", "read", "rule", "spec_sha256", "prereg", "verdict", "created",
     "models": {label: {arm, seed, sha256, step, source}},          every model, by the bytes that were played
     "regimes": {"anchor": {group: id}, "s": {group: id}},          ALWAYS present: a record without one refuses
     "ledgers": {group: {family, digest, rows}},                    the S reads' eval-ledger digests
     "code": {commit, dirty, pin},                                  the playing checkout (``code.json``)
     "cells_inconclusive": [...],                                   what assemble could not count, verbatim
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
from typing import Any, Dict, List, Mapping, Optional

SCHEMA = "gen3_read_record_v1"
REQUIRED = ("schema", "read", "rule", "spec_sha256", "verdict", "created", "models", "regimes", "ledgers", "code",
            "cells_inconclusive", "inputs", "decision")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RID = re.compile(r"^[0-9a-f]{16}$")


class RecordError(ValueError):
    """A record that is not a record (every problem listed)."""


def build(spec: Any, cells: Mapping[str, Any], inputs: Mapping[str, Any], decision: Mapping[str, Any],
          code: Mapping[str, Any], *, now: Optional[str] = None) -> Dict[str, Any]:
    from main.reads.assemble import conclusive
    from main.reads.rules import RULES

    rule = RULES[spec.rule]
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
        "code": dict(code), "cells_inconclusive": conclusive(cells), "inputs": dict(inputs),
        "decision": dict(decision)}
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
    for lane, g, rid in groups:
        if not (isinstance(rid, str) and _RID.match(rid)):
            probs.append(f"{lane} group {g}: regime id {rid!r} (the regime is ALWAYS recorded)")
    for g, led in rec["ledgers"].items():
        if not led.get("digest"):
            probs.append(f"S group {g}: no ledger digest")
    if rec["cells_inconclusive"]:
        # a verdict decided on cells assemble could not count is not a verdict (a wrong regime, unplayed units)
        probs.append(f"cells not conclusive: {list(rec['cells_inconclusive'])[:5]}")
    if not (isinstance(rec["code"], dict) and rec["code"].get("commit")):
        probs.append("code: no playing commit")
    if rule.x35_record:
        for k in ("arm", "look"):
            if k not in rec or rec[k] != rec["decision"].get(k):
                probs.append(f"x35 record: top-level {k!r} is not the decision's")
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
    return f"{rec['read']}: {rec['verdict']} (rule {rec['rule']}; regimes {regs}; models {mods})"
