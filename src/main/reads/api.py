"""The verbs, as functions (the CLI's and a driver's one import): ``plan`` / ``run`` / ``status`` / ``assemble`` /
``decide`` / ``record``. Each reads and writes only the state directory (``main.reads.layout``)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from main.reads import layout as LY
from main.reads.assemble import AssembleError, assemble as _assemble
from main.reads.models import ModelError
from main.reads.record import RecordError, build as build_record, load as load_record, summary_line, write
from main.reads.rules import RULES, RuleInputError
from main.reads.runner import RunError, load_state, plan, run, status
from main.reads.spec import SpecError, load as load_spec
from main.x35.kakuna import KakunaError

__all__ = ["REFUSALS", "assemble", "decide", "load_record", "load_spec", "plan", "record", "run", "status",
           "summary_line"]

#: Every typed refusal a verb raises (the CLI's exit 2).
REFUSALS = (SpecError, ModelError, RunError, AssembleError, RuleInputError, RecordError, LY.StateError, KakunaError)


def _write(path: Path, doc: Any) -> None:
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def assemble(state: Path) -> Dict[str, Any]:
    spec = load_state(state)
    cells = _assemble(spec, LY.check_state_dir(state))
    _write(LY.cells_file(state), cells)
    return cells


def decide(state: Path) -> Dict[str, Any]:
    """The registered rule on the assembled cells (``assemble`` first if ``cells.json`` is absent)."""
    spec = load_state(state)
    cf = LY.cells_file(state)
    cells = json.loads(cf.read_text()) if cf.exists() else assemble(state)
    doc = RULES[spec.rule].inputs(cells, spec.params, spec)
    _write(LY.inputs_file(state), doc)
    dec = RULES[spec.rule].decide(doc, spec.params)
    _write(LY.decision_file(state), dec)
    return dec


def record(state: Path, *, out: Optional[Path] = None) -> Dict[str, Any]:
    """Re-assemble, re-decide and write the schema-checked record (a missing regime id refuses)."""
    spec = load_state(state)
    cells = assemble(state)
    dec = decide(state)
    doc = json.loads(LY.inputs_file(state).read_text())
    code = json.loads(LY.code_file(state).read_text())
    rec = build_record(spec, cells, doc, dec, code)
    write(rec, LY.record_file(state))
    if out is not None:
        write(rec, out)
    return rec
