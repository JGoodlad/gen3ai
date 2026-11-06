"""``--human-data`` AT THE TRAINER (``human_data_v1``): the one decision the trainer, ``main.checkargs`` and the
launcher's ``--dry-run`` share. Grammar and record: ``agents.info_levers.human_data``; the corpus behind a ``teams``
clause: ``utils.team_corpora.human``; design: ``designs/endstate/human_data_architecture.md`` §6.

:func:`decide` (pure reads: the source run's ``metadata.json`` and, for ``teams``, the corpus's files) answers:

1. **The resume / fork refusal.** A ``--model`` launch whose SOURCE run carries a clause other than ``off`` (its
   ``metadata.json`` ``cli_args``, or its lineage, or an ancestor's recorded in that lineage) REFUSES unless
   ``--human-data`` is typed (the clause or ``off``): nothing inherits or drops a lever silently. The launcher's
   restarts re-launch the RESUME-role argv, which keeps the typed flag (it is not fresh-only), so a 6-h restart
   neither trips the refusal nor drops the lever.
2. **What the surface honours.** Training honours ``off`` and ``teams``; ``bc`` / ``kl`` are refused by name.
3. **One lever per run.** A ``teams`` clause beside a ``--team-mix`` (typed, or restored by a same-run resume) is
   refused.
4. **The corpus.** ``teams:<rowset>@<share>`` loads and verifies the corpus against its committed stub.

Every refusal is ``FATAL_CONFIG`` at the trainer (:class:`HumanDataRefused`). ``off`` opens no corpus path.
:func:`lineage_keys` is what the immutable lineage block of a NEW run (fresh or fork) carries.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.info_levers import human_data as HD
from main.exit_codes import FatalConfigError


class HumanDataRefused(FatalConfigError):
    """A ``--human-data`` launch the trainer will not run → ``FATAL_CONFIG`` (3): a restart would hit it identically."""


@dataclass
class Decision:
    record: Optional[Dict[str, Any]]            # None when refused
    refusal: Optional[str] = None
    source: Optional[Dict[str, Any]] = None     # what the --model source run carries (None: nothing / no --model)
    lines: List[str] = field(default_factory=list)


def _metadata(model_path: Optional[str]) -> Dict[str, Any]:
    from agents.training.lineage import run_dir_of
    from utils.paths import resolve_archive_ref

    try:
        run_dir = run_dir_of(resolve_archive_ref(model_path)) if model_path else None
        with open(os.path.join(run_dir, "metadata.json")) as f:
            meta = json.load(f)
    except (OSError, TypeError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _canonical(value: Any) -> Optional[str]:
    """A recorded clause (a record block or the typed text) → its canonical text, or None for off / absent / junk."""
    if isinstance(value, dict):
        return str(value["canonical"]) if HD.is_on(value) else None
    if isinstance(value, str):
        try:
            c = HD.parse(value)
        except HD.HumanDataError:
            return value          # unparseable here (a later grammar?): still a clause, never treated as off
        return None if c.is_off else c.canonical
    return None


def source_clause(model_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """The human-data clause the ``--model`` source run carries (``{"canonical", "where"}``), or None. Read from
    its ``metadata.json``: the last process's resolved record (``cli_args._human_data``) and typed text
    (``cli_args.human_data``), the lineage's own clause, and the first human-data ancestor the lineage names."""
    meta = _metadata(model_path)
    cli = meta.get("cli_args") if isinstance(meta.get("cli_args"), dict) else {}
    lin = meta.get("lineage") if isinstance(meta.get("lineage"), dict) else {}
    anc = lin.get("human_data_ancestor") if isinstance(lin.get("human_data_ancestor"), dict) else None
    for where, value in (("metadata.json cli_args._human_data", cli.get("_human_data")),
                         ("metadata.json cli_args.human_data", cli.get("human_data")),
                         ("metadata.json lineage.human_data", lin.get("human_data")),
                         ("metadata.json lineage.human_data_ancestor", (anc or {}).get("record"))):
        c = _canonical(value)
        if c is not None:
            return {"canonical": c, "where": where, "ancestor": anc}
    return None


def decide(args, *, load_corpus: bool = True) -> Decision:
    """The launch's human-data decision over a resolved (or checkargs-effective) namespace. Pure but for reads."""
    typed = getattr(args, "human_data", None)
    model = getattr(args, "model", None)
    src = source_clause(model) if model else None
    if typed is None and src is not None:
        return Decision(None, (f"--model {model}: its run carries the human-data clause {src['canonical']} "
                               f"({src['where']}); a resume or fork must TYPE --human-data (that clause to keep it, or "
                               "off to drop it), so no lever is inherited or dropped silently"), src)
    try:
        clause = HD.honoured_or_refuse(HD.parse(typed), "train")
    except HD.HumanDataError as e:
        return Decision(None, str(e), src)
    if clause.lever == "teams" and getattr(args, "team_mix", None):
        return Decision(None, (f"--human-data {clause.canonical} beside --team-mix {args.team_mix}: one team lever "
                               "per run (the clause IS a team mix, sides both)"), src)
    corpus = None
    if clause.lever == "teams" and load_corpus:
        from utils.team_corpora import CorpusError
        from utils.team_corpora import human as H

        try:
            corpus = H.identity(clause.rowset, H.load(clause.rowset))
        except CorpusError as e:
            return Decision(None, f"--human-data {clause.canonical}: {e}", src)
    rec = HD.record(clause, corpus)
    lines = [HD.stamp_line(rec)]
    if src is not None and src["canonical"] != clause.canonical:
        lines.append(f"   ⚠️ the --model source carried {src['canonical']} ({src['where']}); this launch TYPED "
                     f"{clause.canonical}: the lever changes here")
    return Decision(rec, None, src, lines)


def resolve(args, emit=print) -> Dict[str, Any]:
    """The trainer's call (before the matchup is declared): refuse, or stamp the record on ``args._human_data`` (so
    ``metadata.json:cli_args`` carries it) and emit the startup stamp."""
    d = decide(args)
    if d.refusal is not None:
        raise HumanDataRefused(f"[HUMAN-DATA] FATAL: {d.refusal}")
    args._human_data = d.record
    for ln in d.lines:
        emit(ln)
    return d.record


def lineage_keys(args, model: Optional[str]) -> Dict[str, Any]:
    """The sibling keys a NEW run's immutable lineage block carries: ``human_data`` (this run's clause, when not
    ``off``) and ``human_data_ancestor`` (the nearest ancestor that trained under a clause: the fork parent itself,
    else the one the parent's lineage names), so the refusal follows a lever down every fork. ``model`` is the fork
    parent AS PARSED (``run_io._run_lineage``'s ``model_path``; None for a fresh run)."""
    out: Dict[str, Any] = {}
    rec = getattr(args, "_human_data", None)
    if HD.is_on(rec):
        out["human_data"] = dict(rec)
    if model:
        meta = _metadata(model)
        cli = meta.get("cli_args") if isinstance(meta.get("cli_args"), dict) else {}
        lin = meta.get("lineage") if isinstance(meta.get("lineage"), dict) else {}
        own = cli.get("_human_data") if HD.is_on(cli.get("_human_data")) else lin.get("human_data")
        if HD.is_on(own):
            from agents.training.lineage import run_dir_of
            out["human_data_ancestor"] = {"run": os.path.basename(run_dir_of(model) or ""), "record": dict(own)}
        elif isinstance(lin.get("human_data_ancestor"), dict):
            out["human_data_ancestor"] = dict(lin["human_data_ancestor"])
    return out


def report_lines(args) -> "tuple[List[str], Optional[str]]":
    """``main.checkargs`` / ``--dry-run``: the lines to print and the refusal (or None)."""
    d = decide(args)
    if d.refusal is not None:
        return [f"human data                     : ✗ REFUSED — {d.refusal}"], d.refusal
    rec = d.record or {}
    head = ("off (no human data)" if not HD.is_on(rec)
            else f"{rec['canonical']}  hash {rec['hash']}  (training honours teams only; eval / play stay off)")
    return [f"human data                     : {head}"] + [f"      {ln.strip()}" for ln in d.lines], None
