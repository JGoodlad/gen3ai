"""THE HUMAN-DATA SWITCH (``--human-data``, ``human_data_v1``): one clause saying which human data, if any, a run
trains on. Design: ``designs/endstate/human_data_architecture.md`` §6 (the ONE grammar, provenance, the refusals);
the experiments that use it: ``designs/endstate/design_human_replay_lever.md`` §6.4 (X60-T, ``teams``) and the R9 card
in ``designs/research_state/EXPERIMENT_BACKLOG.md`` (``init`` / ``opp``).

THE GRAMMAR (the architecture doc's §6 is the one place it is defined; ``LEVERS`` must equal its heads, pinned by
``src/human_data_doc_gate_test.py``)::

    off | teams:<rowset>@<share> | init:<model>@<dose> | opp:<model>@<share>

* ``teams``: the rowset's human-built team corpus joins the self-play team draw on BOTH sides with probability
  ``share`` in [0, 1], the whole corpus, exactly as ``--team-mix <corpus>@ALL:<share> --team-mix-sides both`` would.
  ``share`` 0 is the NULL DOSE: the corpus is verified and stamped but never drawn (the builder installs no mix and
  consumes no RNG, so its played bits are OFF's).
* ``init``: a FRESH run starts from a registered human-replay clone (``utils.human_models``): the parameters the
  clone's training moved are copied into the fresh network; everything else keeps the run's own fresh init. ``dose``
  is 1 (copy) or 0 (the NULL DOSE: the model is verified and stamped, nothing is copied).
* ``opp``: the registered clone, FROZEN, joins the self-play opponents through the stable-opponent path
  (``--stable-opponents <clone zip>:<id> --stable-opponent-selfplay-share <share>``); ``share`` in [0, 1] is the cap
  on the self-play episodes it takes; 0 is the NULL DOSE (verified and stamped, no opponent installed).

Every head is BUILT (:data:`NOT_BUILT` is empty); ``bc`` (dropped by X59-S's read) and ``kl`` (never routed) left the
grammar when ``init`` and ``opp`` took their rows (the doc's decision record, 2026-11-10).

Which surface honours what (:data:`HONOURED`): training honours every head; eval and play honour ``off`` only (no
human decision is ever evaluated or played by this switch).

THE CANONICAL TEXT is ``off`` or ``<lever>:<id>@<param>`` with the parameter printed at up to four decimals (a
parameter must be a whole multiple of 1/10,000, so ``0.50`` and ``.5`` are one clause). THE HASH is
``sha256("human_data_v1:" + canonical)[:16]``. :func:`record` is the provenance block every consumer writes (the
resolved namespace's ``_human_data``, ``metadata.json``'s ``cli_args`` and lineage) and :func:`stamp_line` the
startup stamp the PREREG's VOID check reads.

Pure: no file is read here. The corpus behind ``teams:<rowset>`` is :mod:`utils.team_corpora.human`'s.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

SCHEMA = "human_data_v1"
OFF = "off"
#: The lever heads of the one grammar, in its order.
LEVERS = ("teams", "init", "opp")
#: Which lever heads each surface honours (``off`` is honoured everywhere).
HONOURED: Dict[str, frozenset] = {"train": frozenset(LEVERS), "eval": frozenset(), "play": frozenset()}
#: Heads in the grammar whose consumer does not exist yet: refused by name, never silently ignored.
NOT_BUILT: frozenset = frozenset()
#: Heads whose target is a ROWSET id / a registered human-trained MODEL id (``utils.human_models``).
ROWSET_LEVERS = frozenset({"teams"})
MODEL_LEVERS = frozenset({"init", "opp"})
GRAMMAR = "off | teams:<rowset>@<share> | init:<model>@<dose> | opp:<model>@<share>"
#: The parameter's resolution: whole multiples of 1/10,000 (a value between two is refused, never rounded).
PARAM_QUANTUM = 10_000

_CLAUSE_RE = re.compile(r"^(?P<lever>[a-z]+):(?P<target>[A-Za-z0-9_][A-Za-z0-9_.-]*)@(?P<param>[^@:\s]+)$")


class HumanDataError(ValueError):
    """A clause outside the grammar, or one a surface does not honour."""


@dataclass(frozen=True)
class Clause:
    """One parsed ``--human-data`` clause. ``lever`` is ``off`` or a head of :data:`LEVERS`."""

    lever: str
    target: Optional[str] = None
    param: Optional[float] = None

    @property
    def is_off(self) -> bool:
        return self.lever == OFF

    @property
    def canonical(self) -> str:
        if self.is_off:
            return OFF
        return f"{self.lever}:{self.target}@{format_param(self.param)}"

    @property
    def hash(self) -> str:
        return clause_hash(self.canonical)

    @property
    def label(self) -> str:
        """``off`` or ``<lever>:<id>`` (the clause without its dose)."""
        return OFF if self.is_off else f"{self.lever}:{self.target}"

    @property
    def rowset(self) -> Optional[str]:
        return self.target if self.lever in ROWSET_LEVERS else None

    @property
    def model(self) -> Optional[str]:
        return self.target if self.lever in MODEL_LEVERS else None


def format_param(value: float) -> str:
    """The canonical parameter text: up to four decimals, trailing zeros dropped (``0.5``, ``1``, ``0``)."""
    q = round(float(value) * PARAM_QUANTUM)
    whole, frac = divmod(int(q), PARAM_QUANTUM)
    return str(whole) if frac == 0 else f"{whole}.{frac:04d}".rstrip("0")


def clause_hash(canonical: str) -> str:
    return hashlib.sha256((SCHEMA + ":" + canonical).encode()).hexdigest()[:16]


def parse(text: Optional[str]) -> Clause:
    """``--human-data``'s text → :class:`Clause` (``None`` / empty = ``off``). Syntax only: no rowset or model id is
    looked up here (the trainer's startup does that, so ``main.checkargs`` and a pinned parser need no data)."""
    raw = OFF if text is None else str(text).strip()
    if raw in ("", OFF):
        return Clause(OFF)
    m = _CLAUSE_RE.match(raw)
    if not m:
        raise HumanDataError(f"--human-data {text!r}: expected {GRAMMAR}")
    lever, target, praw = m.group("lever"), m.group("target"), m.group("param")
    if lever not in LEVERS:
        raise HumanDataError(f"--human-data {text!r}: unknown lever {lever!r} (the grammar's heads: "
                             f"{', '.join(LEVERS)}; or off)")
    try:
        value = float(praw)
    except ValueError:
        raise HumanDataError(f"--human-data {text!r}: the parameter {praw!r} is not a number") from None
    if not math.isfinite(value) or value < 0.0:
        raise HumanDataError(f"--human-data {text!r}: the parameter must be finite and >= 0, got {praw}")
    if lever in ("teams", "opp") and value > 1.0:
        raise HumanDataError(f"--human-data {text!r}: a {lever} share must be in [0, 1], got {praw}")
    if lever == "init" and value not in (0.0, 1.0):
        raise HumanDataError(f"--human-data {text!r}: an init dose is 1 (copy the clone) or 0 (the null dose), "
                             f"got {praw}")
    scaled = value * PARAM_QUANTUM
    if abs(scaled - round(scaled)) > 1e-6:
        raise HumanDataError(f"--human-data {text!r}: the parameter must be a whole multiple of 1/{PARAM_QUANTUM:,} "
                             f"(got {praw}); write it at four decimals or fewer")
    return Clause(lever, target, round(scaled) / PARAM_QUANTUM)


def honoured_or_refuse(clause: Clause, surface: str) -> Clause:
    """Return ``clause`` when ``surface`` (``train`` / ``eval`` / ``play``) honours it; raise
    :class:`HumanDataError` naming why otherwise. Never a silent fallback to ``off``."""
    if surface not in HONOURED:
        raise HumanDataError(f"unknown human-data surface {surface!r} (known: {', '.join(HONOURED)})")
    if clause.is_off or clause.lever in HONOURED[surface]:
        return clause
    if clause.lever in NOT_BUILT:
        raise HumanDataError(f"--human-data {clause.canonical}: the {clause.lever!r} lever is NOT BUILT "
                             "(designs/endstate/human_data_architecture.md §6)")
    raise HumanDataError(f"--human-data {clause.canonical}: {surface} honours only 'off' (no human data is ever "
                         f"{'evaluated' if surface == 'eval' else 'played'})")


def record(clause: Clause, corpus: Optional[Mapping[str, Any]] = None,
           human_model: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The provenance block of ``clause``. ``corpus`` is the verified corpus identity a ``teams`` clause resolved to
    (:func:`utils.team_corpora.human.identity`); ``human_model`` the verified model identity an ``init`` / ``opp``
    clause resolved to (:func:`utils.human_models.load`), whose rowset the record then names; ``None`` for ``off``."""
    out: Dict[str, Any] = {"schema": SCHEMA, "label": clause.label, "canonical": clause.canonical,
                           "hash": clause.hash, "lever": clause.lever}
    if not clause.is_off:
        out.update({"target": clause.target, "param": clause.param, "rowset": clause.rowset,
                    "model": clause.model})
    if corpus is not None:
        out["corpus"] = dict(corpus)
    if human_model is not None:
        out["human_model"] = dict(human_model)
        out["rowset"] = human_model.get("rowset")
    return out


def is_null_dose(rec: Optional[Mapping[str, Any]]) -> bool:
    """A recorded clause other than ``off`` whose parameter is 0: verified and stamped, never applied."""
    return is_on(rec) and float(rec.get("param") or 0.0) == 0.0  # type: ignore[union-attr]


def is_on(rec: Optional[Mapping[str, Any]]) -> bool:
    """Does a recorded block (or ``None``) carry a lever other than ``off``?"""
    return isinstance(rec, Mapping) and str(rec.get("canonical", OFF)) != OFF


def stamp_line(rec: Mapping[str, Any]) -> str:
    """The startup stamp (``[HUMAN-DATA] lever=… rowset=… rows_sha=…``); X60's VOID check reads its first three
    fields, so they keep this order and spelling."""
    corpus = rec.get("corpus") or {}
    hm = rec.get("human_model") or {}
    rows_sha = corpus.get("data_sha256") or hm.get("rows_manifest_sha256")
    head = (f"[HUMAN-DATA] lever={rec.get('lever', OFF)} rowset={rec.get('rowset') or '-'} "
            f"rows_sha={rows_sha or '-'}")
    tail = f" clause={rec.get('canonical', OFF)} hash={rec.get('hash')}"
    if corpus:
        tail += f" manifest_sha={corpus.get('manifest_sha256')} n={corpus.get('n')}"
        if rec.get("lever") == "teams" and float(rec.get("param") or 0.0) == 0.0:
            tail += " NULL-DOSE (loaded and stamped, never drawn)"
    if hm:
        tail += f" model={hm.get('id')} zip_sha={hm.get('checkpoint_sha256')}"
        if float(rec.get("param") or 0.0) == 0.0:
            tail += " NULL-DOSE (verified and stamped, never applied)"
    return head + tail
