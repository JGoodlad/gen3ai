"""The verbs, as functions (the CLI's and a driver's one import): ``plan`` / ``run`` / ``status`` / ``assemble`` /
``decide`` / ``record``. Each reads and writes only the state directory (``main.reads.layout``)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from main.reads import lanectl as LC
from main.reads import layout as LY
from main.reads.assemble import AssembleError, assemble as _assemble, unplayed_groups
from main.reads.models import ModelError
from main.reads.playpath import PlayPathError
from main.reads.prefix import PrefixError
from main.reads.record import RecordError, build as build_record, load as load_record, summary_line, write
from main.reads.repin import RepinError, load_repins, repin
from main.reads.rules import RULES, RuleInputError, void_failures, void_gate as rules_void_gate
from main.reads.runner import RunError, load_state, plan, run, status
from main.reads.spec import SpecError, load as load_spec
from main.reads.speccheck import check_spec
from main.reads.imports import ReadImportError
from main.reads.supervisors import SupervisorError
from main.x35.kakuna import KakunaError

__all__ = ["REFUSALS", "assemble", "check_spec", "decide", "hold", "load_record", "load_repins", "load_spec", "pid", "plan",
           "record", "release", "repin", "run", "status", "summary_line", "void_gate"]

#: Every typed refusal a verb raises (the CLI's exit 2).
REFUSALS = (SpecError, ModelError, RunError, AssembleError, RuleInputError, RecordError, LY.StateError, KakunaError,
            SupervisorError, ReadImportError, RepinError, PlayPathError, LC.LaneError, PrefixError)
hold, release = LC.hold, LC.release


def pid(state: Path, lane: str, *, wait_s: float = 0.0) -> int:
    """The pid of ``lane``'s LIVE supervisor as it recorded itself (``<state>/<lane>/supervisor.pid``, checked against
    ``/proc``), waiting up to ``wait_s`` for a supervisor that is still starting; a refusal when there is none."""
    return LC.wait_pid(LY.check_state_dir(state), lane, wait_s)


def _write(path: Path, doc: Any) -> None:
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def assemble(state: Path) -> Dict[str, Any]:
    spec = load_state(state)
    cells = _assemble(spec, LY.check_state_dir(state))
    _write(LY.cells_file(state), cells)
    return cells


def void_gate(state: Path) -> Optional[Dict[str, Any]]:
    """The spec's registered VOID gate (``rules.void_gate``), written to ``void.json``; a non-PASS run REFUSES (a VOID
    seed enters no read). ``None`` when the spec registers none."""
    spec = load_state(state)
    gate = rules_void_gate(spec)
    if gate is None:
        return None
    _write(LY.void_file(state), gate)
    bad = void_failures(gate)
    if bad:
        raise RuleInputError(f"the registered VOID gate fails, so this read is not decided: {bad} (a VOID seed enters "
                             f"no read; re-run it at the same pin and seed; {LY.void_file(state)})")
    return gate


def decide(state: Path) -> Dict[str, Any]:
    """The registered rule on the assembled cells (``assemble`` first if ``cells.json`` is absent), after the
    registered VOID gate (:func:`void_gate`)."""
    void_gate(state)
    spec = load_state(state)
    cf = LY.cells_file(state)
    cells = json.loads(cf.read_text()) if cf.exists() else assemble(state)
    doc = RULES[spec.rule].inputs(cells, spec.params, spec)
    _write(LY.inputs_file(state), doc)
    dec = RULES[spec.rule].decide(doc, spec.params)
    _write(LY.decision_file(state), dec)
    return dec


def partial_at_stop(state: Path, spec: Any, cells: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """``None`` when every planned unit of every decided anchor group was played, or when the spec did not register a
    reduced read (the record then refuses the unplayed units, as it always did). With ``decision.allow_partial_at_stop``:
    ``{groups, stops}`` for :func:`main.reads.record.build`, the stop evidence of each lane that has units unplayed
    (:func:`main.reads.lanectl.stop_evidence`: no live supervisor and a last event of ``stop_at``); a lane not at a registered
    stop is a refusal."""
    if not spec.allow_partial_at_stop:
        return None
    groups = unplayed_groups(cells, RULES[spec.rule].reported(spec.params))
    if not groups:
        return None
    return {"groups": groups, "stops": {lane: LC.stop_evidence(state, lane)
                                        for lane in sorted({k.split(".", 1)[0] for k in groups})}}


def record(state: Path, *, out: Optional[Path] = None) -> Dict[str, Any]:
    """Re-assemble, re-decide and write the schema-checked record (a missing regime id refuses). Units left unplayed refuse
    it, unless the spec registered ``decision.allow_partial_at_stop`` and the lane ended at its registered stop: the record
    is then :data:`main.reads.record.INCOMPLETE_BY_STOP` (:func:`partial_at_stop`)."""
    spec = load_state(state)
    cells = assemble(state)
    dec = decide(state)
    doc = json.loads(LY.inputs_file(state).read_text())
    code = json.loads(LY.code_file(state).read_text())
    vf = LY.void_file(state)
    gate = json.loads(vf.read_text()) if spec.params.get("void") else None   # decide just wrote it
    rec = build_record(spec, cells, doc, dec, code, void=gate, repins=load_repins(state),
                       partial=partial_at_stop(state, spec, cells))
    write(rec, LY.record_file(state))
    if out is not None:
        write(rec, out)
    return rec
