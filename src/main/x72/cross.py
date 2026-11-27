"""X72's CROSSES: plan, driver and reader of the ROUND cross (PREREG §5), whose ``h.json`` is what ``x72_rule.py
decide-round`` reads, and of §9's two REPORTED END crosses (``--end``, the last section here).

After round k (1..4): rows = the three round-k finals, columns = the three round-(k-1) finals (round 0 = the banked
swap finals), every row against every column, **9 cells x 4,000 mirrored pairs**. Every file is taken through
:func:`main.x72.finals.load_final`: an ARCHIVE COPY re-hashed against its ``finals.json`` and judged by its own
record, never a run dir's live file.

**Protocol: X70's deciding cross, unchanged** (:func:`play_argv`): ``main.h2h play-many`` at the PLAY PIN
(:data:`PLAY_PIN`), both sides greedy, batches of 500 pairs, schedule seed 0, the players' eval team builder (no
panel), ``--oracle-reveal-mode off``, purpose ``ab``, label ``x72_swap_plateau``, family ``x72_round_h2h``, request
``x72_r<k>``. The games and the family registration run from the play-pin checkout on ITS code (``cwd`` and
``PYTHONPATH``); the plan and the read run on this checkout's (they exist at no pin), through the same
``main.h2h.cross.look_cells`` / ``matrix_from_cells`` X70's reader called. A cell's score is the ledger cell's
(wins + half the draws over the games), in pp, from the ROW's side.

**The read** (:func:`read`) writes ``<work dir>/h.json`` ALWAYS, ``cells.json`` and ``read.json``:

* every cell present, ``OK`` and holding EXACTLY 4,000 completed pairs: ``h_pp`` is the 3 x 3 (rows = lineages
  7341..7343 at round k, columns = the same lineages at round k-1), exit 0;
* otherwise ``INCONCLUSIVE``: ``h_pp`` is ``[]`` and ``inconclusive`` lists every reason (a missing cell, a non-OK
  cell, a duplicate, fewer OR more than 4,000 pairs), exit 6. The frozen ``decide-round`` answers an empty matrix
  with ``INCONCLUSIVE`` and ``HOLD``, so the chain may hand it ``h.json`` in both cases.

**REFUSED, nothing read, nothing written (exit 7; a HOLD, the look is not spent):** the request's rows name more
than one device / backend (one venue a cross, X70 §14.2), a commit other than the play pin, or a schedule seed other
than 0; or the request holds no row at all (nothing was played: there is nothing to look at).

**One look a round.** :func:`drive` resumes an interrupted play (``main.h2h`` skips the recorded batches) and then
reads ONCE: a work dir that holds ``read.json`` is finished, and a second ``drive`` on it exits 5. A work dir
belongs to one round's cross; a plan that no longer matches its ``shas.json`` (a file moved) exits 5 too.

    python -m main.x72.cross drive --round K --pin-dir <checkout at the play pin> --work-dir W --device cuda
    python -m main.x72.cross plan --round K --work-dir W            # the two files only, nothing played
    python -m main.x72.cross venue-check --work-dir W --device cuda
    python -m main.x72.cross read --work-dir W                      # re-derive the read from the ledger (no game)

``drive``'s exit codes are X70's driver's: 0 read, 2 usage, 3 the plan or the family registration refused, 4 play
failed (re-run to resume), 5 HOLD (the pin checkout does not verify; the work dir is finished or another cross's),
6 INCONCLUSIVE, 7 REFUSED rows (above). On ``cuda`` the caller holds the GPU lease (its token in the environment):
the play runs under ``scripts/ops/gpu_lock.sh`` and ``mem_cap.sh 32``; on ``cpu`` at nice 19 on the agent cores under
``mem_cap.sh 24`` with three threads, as X70's.

``--rehearsal-pairs N`` is a REHEARSAL of the tool path (a CPU check with a tiny count): family
``x72_rehearsal_h2h``, request ``x72_rehearsal_r<k>``, and ``h_pp`` is ALWAYS ``[]`` (the matrix goes to
``rehearsal_h_pp``), so no rehearsal can be handed to the rule as a round's cross.

**The two END crosses (PREREG §9: REPORTED, they decide nothing and carry no class).** Once each after the STOP,
when the READ round R is >= 1, at the same pin and protocol; ``--round`` is then R, which the CALLER takes from the
stop decision (this tool cannot see it):

    python -m main.x72.cross drive --end swap --round R --pin-dir <checkout at the play pin> --work-dir W1 --device cuda
    python -m main.x72.cross drive --end c0   --round R --pin-dir <checkout at the play pin> --work-dir W2 --device cuda

* ``swap``: the round-R finals x the three banked swap finals, 9 cells x 4,000 pairs; family ``x72_end_swap_h2h``,
  request ``x72_end_swap_r<R>``. **At R = 1 it IS round 1's cross** (the same six files): the plan says so
  (``reuses``), nothing is registered or played, and the read is of request ``x72_r1``; if that request holds no row
  the answer is exit 7, never a replay.
* ``c0``: the round-R finals x the eight banked C0 finals (:func:`main.x72.finals.load_c0`), 24 cells x 1,000
  pairs; family ``x72_end_c0_h2h``, request ``x72_end_c0_r<R>``. Its report carries ``beside``: CE-3d's banked
  deciding cross of the three swap finals against the same eight files (+3.97 pp; :func:`banked_c0`).

The statistic is **X70's ``rect_stat``, run from X70's frozen rule file** (:func:`x70_rule`; never restated here),
with the cell variance of the cross's pair count. The plan refuses (exit 3, before any game) unless that file
reproduces X70's own banked interval and, for ``c0``, CE-3d's banked one. The read writes ``report.json``
(``"decides": "nothing"``: the matrix, the interval, no class, no reading), ``cells.json`` and ``read.json``, and
NEVER an ``h.json``: nothing an end cross writes can be handed to ``decide-round``. The cell rule, the refusals,
the one look a work dir and the exit codes are the round cross's; a rehearsal's statistic goes under
``rehearsal_stat``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.training import eval_ledger as L
from main.x72 import finals as F
from utils.paths import repo_path

#: the pin the cross is played at [PREREG §5]: CE-3d's and X70's play pin
PLAY_PIN = "333a73ddda7829b9ccdf9e17ef04624eb40edeba"
PAIRS = 4000          # [x72_rule.PAIRS]
BATCH_PAIRS = 500
SCHEDULE_SEED = 0
PROTOCOL = "gen3_eval_protocol_v1_h2h"
FAMILY = "x72_round_h2h"
LABEL = "x72_swap_plateau"
FAMILY_RULE = (f"{F.PREREG} §5-§6: round-k finals x round-(k-1) finals, 9 cells x 4,000 mirrored pairs, one look a "
               f"round; x72_rule.py decide-round")
REHEARSAL_FAMILY = "x72_rehearsal_h2h"
REHEARSAL_RULE = "a rehearsal of main.x72.cross's tool path: no registered read, no class"
RULE_FILE = "designs/research_state/measurements/x72_swap_plateau_2026-11-25/x72_rule.py"
#: the mem_cap a venue's play declares (GB) and its extra play flags [X70's drive_cross.sh]
VENUES: Dict[str, Tuple[str, Tuple[str, ...]]] = {"cuda": ("32", ()), "cpu": ("24", ("--threads", "3",
                                                                                 "--torch-threads", "3"))}
EXIT_OK, EXIT_USAGE, EXIT_REFUSED, EXIT_PLAY, EXIT_HOLD, EXIT_INCONCLUSIVE, EXIT_ROWS = 0, 2, 3, 4, 5, 6, 7
#: §9's two REPORTED end crosses: kind -> (pairs a cell, family, what the columns are) [PREREG §9]
ENDS: Dict[str, Tuple[int, str, str]] = {"swap": (PAIRS, "x72_end_swap_h2h", "the three banked swap finals"),
                                         "c0": (1000, "x72_end_c0_h2h", "the eight banked C0 finals")}
END_RULE = (f"{F.PREREG} §9: REPORTED, decides nothing, no class: the read finals x {{what}}, {{pairs}} mirrored "
            f"pairs a cell, once after the STOP; X70's rect_stat interval")
_M = "designs/research_state/measurements"
#: X70's frozen rule file (the owner of ``rect_stat``) and the banked deciding cross it must reproduce
X70_RULE = f"{_M}/x70_ent0_control_2026-11-13/screen_rule.py"
X70_ANCHOR = f"{_M}/x70_ent0_control_2026-11-13/record/anchor"
#: CE-3d's deciding cross, the banked +3.97 pp §9 prints the C0 end cross beside
CE3D_DECIDING = f"{_M}/ce3d_magnet_screen_2026-11-11/record/deciding"


class CrossRefused(Exception):
    """A refusal with the driver's exit code."""

    def __init__(self, code: int, why: str):
        super().__init__(why)
        self.code = code


def request(k: int) -> str:
    return f"x72_r{F.check_round(k, lo=1)}"


# ------------------------------------------------------------------------------------------------------- the plan
def plan(k: int, *, root: Optional[Path] = None, models_dir: Optional[Path] = None,
         rehearsal_pairs: Optional[int] = None) -> Tuple[List[List[str]], Dict[str, Any]]:
    """``(cells, shas)`` of round ``k``'s cross: ``cells`` is ``main.h2h play-many --cells``' list (rows outer),
    ``shas`` names the cross for the reader. A final that does not verify is :class:`main.x72.finals.FinalsError`."""
    F.check_round(k, lo=1)
    rows = F.load_round(k, root=root, models_dir=models_dir)
    cols = F.load_round(k - 1, root=root, models_dir=models_dir)
    shas = [f.sha256 for f in rows + cols]
    if len(set(shas)) != len(shas):
        raise F.FinalsError(f"round {k}'s cross names the same file twice (sha256 {sorted(shas)}): six distinct "
                            f"finals are registered")
    return _planned(f"x72_round_{k}", k, rows, cols, PAIRS, FAMILY, f"r{k}", rehearsal_pairs)


def _planned(cross: str, k: int, rows: Sequence[F.Final], cols: Sequence[F.Final], full: int, family: str,
             tag: str, rehearsal_pairs: Optional[int]) -> Tuple[List[List[str]], Dict[str, Any]]:
    """The cells and the ``shas.json`` of ``rows`` x ``cols`` at ``full`` pairs a cell (or a rehearsal's count)."""
    reh = rehearsal_pairs is not None
    pairs = full if rehearsal_pairs is None else int(rehearsal_pairs)
    if reh and not 0 < pairs < full:
        raise F.FinalsError(f"--rehearsal-pairs {rehearsal_pairs!r} is not in 1..{full - 1}")
    doc = {"cross": cross, "round": k, "pairs": pairs, "batch_pairs": min(BATCH_PAIRS, pairs),
           "family": REHEARSAL_FAMILY if reh else family,
           "request": f"x72_rehearsal_{tag}" if reh else f"x72_{tag}", "rehearsal": reh, "play_pin": PLAY_PIN,
           "rows": [f.sha256 for f in rows], "cols": [f.sha256 for f in cols],
           "row_seeds": [f.seed for f in rows], "col_seeds": [f.seed for f in cols],
           "row_runs": [f.run for f in rows], "col_runs": [f.run for f in cols],
           "row_files": [f.path for f in rows], "col_files": [f.path for f in cols],
           "row_steps": [f.step for f in rows], "col_steps": [f.step for f in cols]}
    return [[r.path, c.path] for r in rows for c in cols], doc


# ------------------------------------------------------------------------------- §9's end crosses: plan, statistic
def reuses(end: Optional[str], k: int) -> bool:
    """The swap end cross at read round 1 IS round 1's cross [PREREG §9]: the same six files, nothing to play."""
    return end == "swap" and k == 1


def _close(got: Mapping[str, Any], banked: Mapping[str, Any]) -> bool:
    a = [got["delta_hat_pp"], *got["interval90_pp"]]
    b = [banked["delta_hat_pp"], *banked["interval90_pp"]]
    return max(abs(x - y) for x, y in zip(a, b)) <= 1e-9


def x70_rule() -> Dict[str, Any]:
    """X70's frozen rule file, RUN (§9 registers ITS ``rect_stat``; it is never restated), after it reproduces X70's
    own banked deciding interval from X70's banked matrix. Anything else is a :class:`main.x72.finals.FinalsError`."""
    rule: Dict[str, Any] = runpy.run_path(str(repo_path(X70_RULE)))
    d = repo_path(X70_ANCHOR)
    h, v = json.loads((d / "h.json").read_text()), json.loads((d / "verdict.json").read_text())
    if not _close(rule["rect_stat"](h["h_pp"], cv=rule["cell_var"](h["pairs"])), v):
        raise F.FinalsError(f"{X70_RULE} does not reproduce X70's banked interval ({X70_ANCHOR}/verdict.json)")
    return rule


def banked_c0(rule: Mapping[str, Any]) -> Dict[str, Any]:
    """CE-3d's deciding cross, the +3.97 pp the C0 end cross is printed beside: the three banked swap finals x the
    SAME eight C0 finals, 1,000 pairs a cell, at the same play pin (its files, pin and venue are pinned to
    :mod:`main.x72.finals`' tables by test). Refused unless ``rule``'s ``rect_stat`` reproduces its banked interval
    at this cross's cell variance."""
    d = repo_path(CE3D_DECIDING)
    h, v = json.loads((d / "h.json").read_text()), json.loads((d / "verdict.json").read_text())
    if not _close(rule["rect_stat"](h["h_pp"], cv=rule["cell_var"](ENDS["c0"][0])), v):
        raise F.FinalsError(f"{X70_RULE} does not reproduce CE-3d's banked interval ({CE3D_DECIDING}/verdict.json)")
    return {"what": "CE-3d's deciding cross: the three banked swap finals (this lineage's round 0) x the same eight "
                    "C0 finals, 1,000 pairs a cell, the same play pin",
            "source": f"{CE3D_DECIDING}/verdict.json", "delta_hat_pp": v["delta_hat_pp"],
            "interval90_pp": v["interval90_pp"], "shape": v["shape"],
            "played_on": "cpu / eager (its run.log); an end cross on the card is another venue"}


def plan_end(end: str, k: int, *, root: Optional[Path] = None, models_dir: Optional[Path] = None,
             rehearsal_pairs: Optional[int] = None) -> Tuple[List[List[str]], Dict[str, Any]]:
    """``(cells, shas)`` of §9's end cross ``end`` at READ round ``k``: the round-k finals against the banked swap
    finals (``swap``) or the eight banked C0 finals (``c0``). ``shas`` is :func:`plan`'s plus ``end``, ``reuses``
    and ``beside``. Refused before any game unless the statistic's file reproduces its banked intervals."""
    if end not in ENDS:
        raise F.FinalsError(f"end cross {end!r} is not one of {sorted(ENDS)}")
    F.check_round(k, lo=1)
    full, family, _what = ENDS[end]
    rule = x70_rule()
    if reuses(end, k):
        if rehearsal_pairs is not None:
            raise F.FinalsError("the swap end cross at read round 1 IS round 1's cross: it plays nothing, so there "
                                "is nothing to rehearse (--rehearsal-pairs)")
        cells, doc = plan(1, root=root, models_dir=models_dir)
        doc["reuses"] = doc["cross"]
    else:
        rows = F.load_round(k, root=root, models_dir=models_dir)
        cols = F.load_round(0, models_dir=models_dir) if end == "swap" else F.load_c0(models_dir=models_dir)
        shas = [f.sha256 for f in rows + cols]
        if len(set(shas)) != len(shas):
            raise F.FinalsError(f"the end cross '{end}' names the same file twice (sha256 {sorted(shas)})")
        cells, doc = _planned("", k, rows, cols, full, family, f"end_{end}_r{k}", rehearsal_pairs)
        doc["reuses"] = None
    doc.update(cross=f"x72_end_{end}_r{k}", end=end, beside=banked_c0(rule) if end == "c0" else None)
    return cells, doc


def family_rule(shas: Mapping[str, Any]) -> str:
    """The rule text the cross's family is registered with."""
    if shas["rehearsal"]:
        return REHEARSAL_RULE
    if shas.get("end") and not shas.get("reuses"):
        pairs, _family, what = ENDS[shas["end"]]
        return END_RULE.format(what=what, pairs=f"{pairs:,}")
    return FAMILY_RULE


def _registered(shas: Mapping[str, Any]) -> Tuple[int, str, str]:
    """``(pairs, family, request)`` a plan must carry to be a REGISTERED cross (a rehearsal is none)."""
    end, k = shas.get("end"), F.check_round(shas["round"], lo=1)
    if end is None or reuses(end, k):
        return PAIRS, FAMILY, request(k)
    return ENDS[end][0], ENDS[end][1], f"x72_end_{end}_r{k}"


def write_plan(work: Path, cells: List[List[str]], shas: Dict[str, Any]) -> None:
    """Write ``cells_plan.json`` and ``shas.json``; a work dir that holds another cross's ``shas.json`` is a HOLD."""
    work.mkdir(parents=True, exist_ok=True)
    old = work / "shas.json"
    if old.is_file() and json.loads(old.read_text()) != shas:
        raise CrossRefused(EXIT_HOLD, f"{old} is another cross's plan (or a planned file moved): a work dir belongs "
                                      f"to ONE round's cross and nothing in it was overwritten")
    (work / "cells_plan.json").write_text(json.dumps(cells, indent=1))
    old.write_text(json.dumps(shas, indent=1))


def _shas(work: Path) -> Dict[str, Any]:
    p = work / "shas.json"
    if not p.is_file():
        raise CrossRefused(EXIT_USAGE, f"{p}: no plan in this work dir (python -m main.x72.cross plan)")
    doc: Dict[str, Any] = json.loads(p.read_text())
    return doc


# ---------------------------------------------------------------------------------------------------- the ledger
def _ledger_root(root: Optional[Path]) -> Path:
    return Path(root) if root is not None else L.archive_ledger_root()


#: The rows of the cross's family, read to check the venue, the pin and the seed BEFORE any cell is pooled: an A/B
#: family at the h2h protocol, greedy on both sides, mirrored, across runs (the terms of
#: ``main.h2h.cross.FAMILY_READ_OFF``, which the cells are read under; declared here because the reader gate wants
#: each reader's own declaration, and pinned equal to it by test).
ROWS_READ = L.ReaderDecl(
    name="main.x72.cross.rows", purposes=frozenset({"ab"}),
    regime=L.RegimeFilter(protocol="gen3_eval_protocol_v1_h2h", play="greedy", opponent_play="greedy", mirrored=True),
    requests="family", selection="include", flags_ok=frozenset(), inference="across_runs",
    decision_kind="ab_verdict")


def request_rows(root: Path, family: str, request_id: str) -> List[Mapping[str, Any]]:
    rows = L.read(ROWS_READ, root=root, family=family).rows
    return [r for r in rows if (r.get("request") or {}).get("id") == request_id]


def venues(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], int]:
    """``{(device, backend): rows}`` (``cpu`` / ``eager`` on the agent cores, ``cuda`` / ``graph`` on the card)."""
    where: Dict[Tuple[str, str], int] = {}
    for r in rows:
        c = r.get("compute") or {}
        k = (str(c.get("device")), str(c.get("backend")))
        where[k] = where.get(k, 0) + 1
    return where


def _where(where: Mapping[Tuple[str, str], int]) -> str:
    return ", ".join(f"{d}/{b} x{n} rows" for (d, b), n in sorted(where.items()))


def row_problems(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Why the request's rows are NOT one registered cross (a refusal before any cell is pooled)."""
    probs: List[str] = []
    where = venues(rows)
    if len(where) > 1:
        probs.append(f"the cross was played on more than one device/backend: {_where(where)} (one venue a cross)")
    commits = sorted({str(r.get("commit")) for r in rows} - {PLAY_PIN})
    if commits:
        probs.append(f"rows played at {commits}, not the play pin {PLAY_PIN[:8]}")
    seeds = sorted({str((r.get("seed") or {}).get("schedule_seed")) for r in rows} - {str(SCHEDULE_SEED)})
    if seeds:
        probs.append(f"rows with schedule seed {seeds}, not {SCHEDULE_SEED}")
    return probs


def venue_check(shas: Mapping[str, Any], device: str, root: Path) -> None:
    """Before playing or resuming: the request's recorded rows, if any, are one registered cross played on
    ``device`` (a cross is finished on the device its first batch was played on)."""
    rows = request_rows(root, shas["family"], shas["request"])
    probs = row_problems(rows)
    other = {k: n for k, n in venues(rows).items() if k[0].split(":")[0] != device}
    if other:
        probs.append(f"request {shas['request']} already holds rows played on {_where(other)}; it is finished on "
                     f"that device, never on '{device}' (one venue a cross)")
    if probs:
        raise CrossRefused(EXIT_ROWS, "; ".join(probs))


# ------------------------------------------------------------------------------------------------------ the read
def read(work: Path, root: Path, *, now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    """Read the planned cross from the ledger and write ``cells.json``, ``h.json``, ``read.json`` (module
    docstring). Returns the read record; :class:`CrossRefused` (exit 7) when the rows are not one registered cross."""
    import main.h2h.cross as X

    shas = _shas(work)
    pairs, reh = int(shas["pairs"]), bool(shas["rehearsal"])
    if not reh and (pairs, shas["family"], shas["request"]) != _registered(shas):
        raise CrossRefused(EXIT_REFUSED, f"{work / 'shas.json'} is not the registered cross {shas['cross']} "
                                         f"(pairs, family, request: {_registered(shas)})")
    never = "; it is round 1's cross, which `drive --round 1` plays and this never replays"
    try:
        rows = request_rows(root, shas["family"], shas["request"])
    except L.ReaderDeclError as exc:                  # the round's family is not registered: round 1 never started
        if not shas.get("reuses"):
            raise
        raise CrossRefused(EXIT_ROWS, f"{exc}{never}") from exc
    if not rows:
        raise CrossRefused(EXIT_ROWS, f"request {shas['request']} holds no row under {root}: nothing was played, so "
                                      f"there is nothing to read (the look is not spent)"
                           + (never if shas.get("reuses") else ""))
    probs = row_problems(rows)
    if probs:
        raise CrossRefused(EXIT_ROWS, "; ".join(probs) + " (nothing was read)")
    cells = X.look_cells(root, shas["family"], shas["request"], PROTOCOL)
    m = X.matrix_from_cells(cells, shas["rows"], shas["cols"], min_pairs=pairs)
    reasons = list(m.inconclusive)
    reasons += [f"cell {c['player']} vs {c['opponent']}: {c['n_pairs']} completed pairs > the registered {pairs}"
                for c in m.cells if c["n_pairs"] > pairs]
    matrix = [list(r) for r in m.h] if m.h is not None and not reasons else []
    when = (now or _dt.datetime.now(_dt.timezone.utc)).isoformat(timespec="seconds")
    if shas.get("end"):
        return _read_end(work, shas, venues(rows), len(cells), list(m.cells), matrix, reasons, when)
    h = {"h_pp": [] if reh else matrix, "round": shas["round"], "pairs": pairs,
         "rows": shas["row_runs"], "cols": shas["col_runs"], "row_seeds": shas["row_seeds"],
         "col_seeds": shas["col_seeds"], "row_sha256": shas["rows"], "col_sha256": shas["cols"],
         "row_steps": shas["row_steps"], "col_steps": shas["col_steps"], "inconclusive": reasons,
         "family": shas["family"], "request": shas["request"], "play_pin": PLAY_PIN, "rehearsal": reh,
         "played_on": [{"device": d, "backend": b, "rows": n} for (d, b), n in sorted(venues(rows).items())]}
    if reh:
        h["rehearsal_h_pp"] = matrix
    outcome = "INCONCLUSIVE" if reasons else ("REHEARSAL" if reh else "READ")
    rec = dict(h, outcome=outcome, n_cells_read=len(cells), cells=list(m.cells), read_at=when,
               decide=f"python {RULE_FILE} decide-round {shas['round']} {work / 'h.json'}")
    (work / "cells.json").write_text(json.dumps(list(m.cells), indent=1))
    (work / "h.json").write_text(json.dumps(h, indent=1))
    (work / "read.json").write_text(json.dumps(rec, indent=1))
    return rec


def _read_end(work: Path, shas: Mapping[str, Any], where: Mapping[Tuple[str, str], int], n_cells: int,
              cells: List[Dict[str, Any]], matrix: List[List[float]], reasons: List[str], when: str) -> Dict[str, Any]:
    """An end cross's REPORT (§9): the matrix and X70's ``rect_stat`` on it, no class, no reading, no ``h.json``."""
    reh, pairs = bool(shas["rehearsal"]), int(shas["pairs"])
    rep: Dict[str, Any] = {
        "decides": "nothing", "cross": shas["cross"], "end": shas["end"], "read_round": shas["round"],
        "reuses": shas["reuses"], "read": bool(matrix) and not reh, "inconclusive": reasons, "pairs": pairs,
        "rows": shas["row_runs"], "cols": shas["col_runs"], "row_seeds": shas["row_seeds"],
        "col_seeds": shas["col_seeds"], "row_sha256": shas["rows"], "col_sha256": shas["cols"],
        "row_steps": shas["row_steps"], "col_steps": shas["col_steps"], "family": shas["family"],
        "request": shas["request"], "play_pin": PLAY_PIN, "rehearsal": reh,
        "played_on": [{"device": d, "backend": b, "rows": n} for (d, b), n in sorted(where.items())],
        "statistic": f"{X70_RULE} rect_stat, cell variance 2500 / (2 x {pairs}) pp^2", "beside": shas["beside"]}
    if matrix:
        rule = x70_rule()
        stat = rule["rect_stat"](matrix, cv=rule["cell_var"](pairs))
        rep.update({"rehearsal_matrix_pp": matrix, "rehearsal_stat": stat} if reh else {"matrix_pp": matrix, **stat})
    outcome = "INCONCLUSIVE" if reasons else ("REHEARSAL" if reh else "REPORTED")
    rec = dict(rep, outcome=outcome, n_cells_read=n_cells, cells=cells, read_at=when)
    (work / "cells.json").write_text(json.dumps(cells, indent=1))
    (work / "report.json").write_text(json.dumps(rep, indent=1))
    (work / "read.json").write_text(json.dumps(rec, indent=1))
    return rec


# ---------------------------------------------------------------------------------------------------- the driver
def play_argv(py: str, work: Path, shas: Mapping[str, Any], device: str, root: Path) -> List[str]:
    """``main.h2h play-many`` as registered (X70's protocol), without the venue's wrappers."""
    return [py, "-m", "main.h2h", "play-many", "--cells", str(work / "cells_plan.json"),
            "--pairs", str(shas["pairs"]), "--batch-pairs", str(shas["batch_pairs"]), "--seed", str(SCHEDULE_SEED),
            "--device", device, "--purpose", "ab", "--family", shas["family"], "--request", shas["request"],
            "--label", LABEL if not shas["rehearsal"] else "x72_rehearsal", "--oracle-reveal-mode", "off",
            "--out", str(root)]


def venue_argv(play: Sequence[str], device: str, cpus: str) -> List[str]:
    """The play under its venue's wrappers: the card through the GPU lock, the CPU at nice 19 on the agent cores."""
    cap, extra = VENUES[device]
    ops = repo_path("scripts", "ops")
    capped = [str(ops / "mem_cap.sh"), cap, *play, *extra]
    if device == "cuda":
        return [str(ops / "gpu_lock.sh"), *capped]
    return ["nice", "-n", "19", "taskset", "-c", cpus, *capped]


def _pin_problems(pin_dir: Path, run: Callable[..., Any]) -> List[str]:
    head = run(["git", "-C", str(pin_dir), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0 or head.stdout.strip() != PLAY_PIN:
        return [f"{pin_dir} is at {head.stdout.strip() or 'no commit'}, not the play pin {PLAY_PIN}"]
    dirty = run(["git", "-C", str(pin_dir), "status", "--porcelain", "--untracked-files=no", "--", "src", "data"],
                capture_output=True, text=True)
    if dirty.returncode != 0 or dirty.stdout.strip():
        return [f"{pin_dir} has tracked changes under src/ or data/: {dirty.stdout.strip()[:200]!r}"]
    return []


def drive(k: int, pin_dir: Path, work: Path, device: str, *, ledger_root: Optional[Path] = None,
          archive_root: Optional[Path] = None, models_dir: Optional[Path] = None,
          rehearsal_pairs: Optional[int] = None, run: Callable[..., Any] = subprocess.run,
          py: Optional[str] = None, cpus: Optional[str] = None, end: Optional[str] = None) -> int:
    """Plan, register the family, check the venue, play (or resume), read ONCE. Returns the exit code (module
    docstring). ``run`` is ``subprocess.run`` (the tests pass a recorder). With ``end``, §9's end cross of that
    kind at READ round ``k``; the one that IS round 1's cross is planned and read, and nothing else runs."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    log = work / "run.log"

    def say(msg: str) -> None:
        print(msg, flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    try:
        if device not in VENUES:
            raise CrossRefused(EXIT_USAGE, f"device {device!r} is not one of {sorted(VENUES)}")
        if (work / "read.json").is_file():
            raise CrossRefused(EXIT_HOLD, f"{work / 'read.json'} exists: this round's ONE look is spent (no top-up, "
                                          f"no replay); `read` re-derives it from the ledger without a game")
        probs = [] if reuses(end, k) else _pin_problems(Path(pin_dir), run)
        if probs:
            raise CrossRefused(EXIT_HOLD, "; ".join(probs))
        cells, shas = _plan(end, k, archive_root, models_dir, rehearsal_pairs)
        write_plan(work, cells, shas)
        root = _ledger_root(ledger_root)
        if shas.get("reuses"):
            say(f"[x72 cross] end cross 'swap' at read round 1 IS round 1's cross: nothing is registered or played; "
                f"reading request {shas['request']} of {shas['family']} from {root}")
            return _report(read(work, root), say)
        py = py or os.environ.get("GEN3AI_PYTHON") or sys.executable
        env = dict(os.environ, PYTHONPATH=str(Path(pin_dir) / "src"), GEN3AI_SHARED_COMPILE_CACHE="0")
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        say(f"[x72 cross] {_what(shas)}: {len(cells)} cells x {shas['pairs']} pairs on {device}; family "
            f"{shas['family']}, request {shas['request']}; ledger {root}; play pin {PLAY_PIN[:8]} at {pin_dir}")
        with open(log, "a", encoding="utf-8") as lf:
            reg = run([py, "-m", "main.eval_ledger", "family-register", "--root", str(root), "--id", shas["family"],
                       "--decision-kind", "ab_verdict", "--rule", family_rule(shas),
                       "--protocol", PROTOCOL, "--commit", PLAY_PIN],
                      cwd=str(pin_dir), env=env, stdout=lf, stderr=subprocess.STDOUT)
        if reg.returncode != 0:
            raise CrossRefused(EXIT_REFUSED, f"family-register exited {reg.returncode} (see {log})")
        venue_check(shas, device, root)
        if cpus is None and device == "cpu":
            from utils.core_split import agent_cpus

            cpus = agent_cpus()
        argv = venue_argv(play_argv(py, work, shas, device, root), device, cpus or "")
        say("[x72 cross] play: " + " ".join(argv))
        with open(log, "a", encoding="utf-8") as lf:
            played = run(argv, cwd=str(pin_dir), env=env, stdout=lf, stderr=subprocess.STDOUT)
        if played.returncode != 0:
            raise CrossRefused(EXIT_PLAY, f"play-many exited {played.returncode} (see {log}); re-run to resume")
        rec = read(work, root)
    except CrossRefused as exc:
        say(f"[x72 cross] {'HOLD' if exc.code == EXIT_HOLD else 'REFUSED'} (exit {exc.code}): {exc}")
        return exc.code
    return _report(rec, say)


def _plan(end: Optional[str], k: int, archive_root: Optional[Path], models_dir: Optional[Path],
          rehearsal_pairs: Optional[int]) -> Tuple[List[List[str]], Dict[str, Any]]:
    try:
        if end is None:
            return plan(k, root=archive_root, models_dir=models_dir, rehearsal_pairs=rehearsal_pairs)
        return plan_end(end, k, root=archive_root, models_dir=models_dir, rehearsal_pairs=rehearsal_pairs)
    except F.FinalsError as exc:
        raise CrossRefused(EXIT_REFUSED, str(exc)) from exc


def _what(doc: Mapping[str, Any]) -> str:
    return f"{doc['cross']} (REPORTED, decides nothing)" if doc.get("end") else f"round {doc['round']}"


def _report(rec: Mapping[str, Any], say: Callable[[str], None] = print) -> int:
    say(f"[x72 cross] {_what(rec)}: {rec['outcome']}; {rec['n_cells_read']} cell(s) read"
        + "".join(f"\n  - {r}" for r in rec["inconclusive"]))
    if not rec.get("end"):
        say(f"[x72 cross] next: {rec['decide']}")
    elif rec["read"]:
        lo, hi = rec["interval90_pp"]
        say(f"[x72 cross] D = {rec['delta_hat_pp']:+.2f} pp, 90 % interval [{lo:+.2f}, {hi:+.2f}] (X70's rect_stat, "
            f"{rec['shape'][0]} x {rec['shape'][1]} at {rec['pairs']} pairs a cell); no class")
        b = rec["beside"]
        if b:
            say(f"[x72 cross] beside: D = {b['delta_hat_pp']:+.2f} pp, [{b['interval90_pp'][0]:+.2f}, "
                f"{b['interval90_pp'][1]:+.2f}]: {b['what']}; played on {b['played_on']} ({b['source']})")
    return EXIT_INCONCLUSIVE if rec["inconclusive"] else EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x72.cross", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, planned: bool) -> None:
        p.add_argument("--work-dir", required=True, help="this ONE cross's directory (plan, log, read)")
        p.add_argument("--ledger-root", default=None, help="default: the run archive's _ledger")
        if planned:
            p.add_argument("--round", type=int, required=True, help="the round; with --end, the READ round")
            p.add_argument("--end", choices=sorted(ENDS), default=None,
                           help="§9's REPORTED end cross: the read finals against the banked swap / C0 finals")
            p.add_argument("--archive-root", default=None, help="default: <gen3ai archive>/x72_swap_plateau/finals")
            p.add_argument("--models-dir", default=None, help="default: the run archive (models/)")
            p.add_argument("--rehearsal-pairs", type=int, default=None, help="a tool-path rehearsal, never a look")

    common(sub.add_parser("plan", help="verify the finals; write cells_plan.json and shas.json"), True)
    v = sub.add_parser("venue-check", help="exit 7 if the request already holds rows of another venue / pin / seed")
    common(v, False)
    v.add_argument("--device", required=True, choices=sorted(VENUES))
    common(sub.add_parser("read", help="re-derive the read from the ledger (no game)"), False)
    d = sub.add_parser("drive", help="plan, register, check the venue, play or resume, read ONCE")
    common(d, True)
    d.add_argument("--pin-dir", required=True, help="a checkout at the play pin with clean src/ and data/")
    d.add_argument("--device", required=True, choices=sorted(VENUES))
    a = ap.parse_args(argv)
    work = Path(a.work_dir)

    def opt(name: str) -> Optional[Path]:
        return Path(getattr(a, name)) if getattr(a, name, None) else None

    try:
        if a.cmd == "drive":
            return drive(a.round, Path(a.pin_dir), work, a.device, ledger_root=opt("ledger_root"),
                         archive_root=opt("archive_root"), models_dir=opt("models_dir"),
                         rehearsal_pairs=a.rehearsal_pairs, end=a.end)
        if a.cmd == "plan":
            cells, shas = _plan(a.end, a.round, opt("archive_root"), opt("models_dir"), a.rehearsal_pairs)
            write_plan(work, cells, shas)
            print(f"{len(cells)} cells planned in {work}"
                  + (f": {shas['cross']} IS {shas['reuses']} (request {shas['request']}); nothing to play"
                     if shas.get("reuses") else f" x {shas['pairs']} pairs: {shas['cross']}" if a.end else ""))
            return EXIT_OK
        root = _ledger_root(opt("ledger_root"))
        if a.cmd == "venue-check":
            venue_check(_shas(work), a.device, root)
            print(f"venue OK: nothing recorded for this request off '{a.device}'")
            return EXIT_OK
        return _report(read(work, root))
    except CrossRefused as exc:
        print(f"REFUSED (exit {exc.code}): {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
