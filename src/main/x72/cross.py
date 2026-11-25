"""X72's ROUND CROSS (PREREG §5): plan, driver and reader. Its ``h.json`` is what ``x72_rule.py decide-round`` reads.

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
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
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
    reh = rehearsal_pairs is not None
    pairs = PAIRS if rehearsal_pairs is None else int(rehearsal_pairs)
    if reh and not 0 < pairs < PAIRS:
        raise F.FinalsError(f"--rehearsal-pairs {rehearsal_pairs!r} is not in 1..{PAIRS - 1}")
    doc = {"cross": f"x72_round_{k}", "round": k, "pairs": pairs, "batch_pairs": min(BATCH_PAIRS, pairs),
           "family": REHEARSAL_FAMILY if reh else FAMILY,
           "request": f"x72_rehearsal_r{k}" if reh else request(k), "rehearsal": reh, "play_pin": PLAY_PIN,
           "rows": [f.sha256 for f in rows], "cols": [f.sha256 for f in cols],
           "row_seeds": [f.seed for f in rows], "col_seeds": [f.seed for f in cols],
           "row_runs": [f.run for f in rows], "col_runs": [f.run for f in cols],
           "row_files": [f.path for f in rows], "col_files": [f.path for f in cols],
           "row_steps": [f.step for f in rows], "col_steps": [f.step for f in cols]}
    return [[r.path, c.path] for r in rows for c in cols], doc


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
    if not reh and (pairs != PAIRS or shas["family"] != FAMILY or shas["request"] != request(shas["round"])):
        raise CrossRefused(EXIT_REFUSED, f"{work / 'shas.json'} is not round {shas['round']}'s registered cross "
                                         f"({PAIRS} pairs, {FAMILY}, {request(shas['round'])})")
    rows = request_rows(root, shas["family"], shas["request"])
    if not rows:
        raise CrossRefused(EXIT_ROWS, f"request {shas['request']} holds no row under {root}: nothing was played, so "
                                      f"there is nothing to read (the look is not spent)")
    probs = row_problems(rows)
    if probs:
        raise CrossRefused(EXIT_ROWS, "; ".join(probs) + " (nothing was read)")
    cells = X.look_cells(root, shas["family"], shas["request"], PROTOCOL)
    m = X.matrix_from_cells(cells, shas["rows"], shas["cols"], min_pairs=pairs)
    reasons = list(m.inconclusive)
    reasons += [f"cell {c['player']} vs {c['opponent']}: {c['n_pairs']} completed pairs > the registered {pairs}"
                for c in m.cells if c["n_pairs"] > pairs]
    matrix = [list(r) for r in m.h] if m.h is not None and not reasons else []
    h = {"h_pp": [] if reh else matrix, "round": shas["round"], "pairs": pairs,
         "rows": shas["row_runs"], "cols": shas["col_runs"], "row_seeds": shas["row_seeds"],
         "col_seeds": shas["col_seeds"], "row_sha256": shas["rows"], "col_sha256": shas["cols"],
         "row_steps": shas["row_steps"], "col_steps": shas["col_steps"], "inconclusive": reasons,
         "family": shas["family"], "request": shas["request"], "play_pin": PLAY_PIN, "rehearsal": reh,
         "played_on": [{"device": d, "backend": b, "rows": n} for (d, b), n in sorted(venues(rows).items())]}
    if reh:
        h["rehearsal_h_pp"] = matrix
    outcome = "INCONCLUSIVE" if reasons else ("REHEARSAL" if reh else "READ")
    rec = dict(h, outcome=outcome, n_cells_read=len(cells), cells=list(m.cells),
               read_at=(now or _dt.datetime.now(_dt.timezone.utc)).isoformat(timespec="seconds"),
               decide=f"python {RULE_FILE} decide-round {shas['round']} {work / 'h.json'}")
    (work / "cells.json").write_text(json.dumps(list(m.cells), indent=1))
    (work / "h.json").write_text(json.dumps(h, indent=1))
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
          py: Optional[str] = None, cpus: Optional[str] = None) -> int:
    """Plan, register the family, check the venue, play (or resume), read ONCE. Returns the exit code (module
    docstring). ``run`` is ``subprocess.run`` (the tests pass a recorder)."""
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
        probs = _pin_problems(Path(pin_dir), run)
        if probs:
            raise CrossRefused(EXIT_HOLD, "; ".join(probs))
        try:
            cells, shas = plan(k, root=archive_root, models_dir=models_dir, rehearsal_pairs=rehearsal_pairs)
        except F.FinalsError as exc:
            raise CrossRefused(EXIT_REFUSED, str(exc)) from exc
        write_plan(work, cells, shas)
        root = _ledger_root(ledger_root)
        py = py or os.environ.get("GEN3AI_PYTHON") or sys.executable
        env = dict(os.environ, PYTHONPATH=str(Path(pin_dir) / "src"), GEN3AI_SHARED_COMPILE_CACHE="0")
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        say(f"[x72 cross] round {k}: {len(cells)} cells x {shas['pairs']} pairs on {device}; family "
            f"{shas['family']}, request {shas['request']}; ledger {root}; play pin {PLAY_PIN[:8]} at {pin_dir}")
        with open(log, "a", encoding="utf-8") as lf:
            reg = run([py, "-m", "main.eval_ledger", "family-register", "--root", str(root), "--id", shas["family"],
                       "--decision-kind", "ab_verdict", "--rule", REHEARSAL_RULE if shas["rehearsal"] else FAMILY_RULE,
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


def _report(rec: Mapping[str, Any], say: Callable[[str], None] = print) -> int:
    say(f"[x72 cross] round {rec['round']}: {rec['outcome']}; {rec['n_cells_read']} cell(s) read"
        + "".join(f"\n  - {r}" for r in rec["inconclusive"]))
    say(f"[x72 cross] next: {rec['decide']}")
    return EXIT_INCONCLUSIVE if rec["inconclusive"] else EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x72.cross", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, planned: bool) -> None:
        p.add_argument("--work-dir", required=True, help="this ONE cross's directory (plan, log, read)")
        p.add_argument("--ledger-root", default=None, help="default: the run archive's _ledger")
        if planned:
            p.add_argument("--round", type=int, required=True)
            p.add_argument("--archive-root", default=None, help="default: <gen3ai archive>/x72_swap_plateau/finals")
            p.add_argument("--models-dir", default=None, help="default: the run archive (models/)")
            p.add_argument("--rehearsal-pairs", type=int, default=None, help="a tool-path rehearsal, never a look")

    common(sub.add_parser("plan", help="verify the six finals; write cells_plan.json and shas.json"), True)
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
                         rehearsal_pairs=a.rehearsal_pairs)
        if a.cmd == "plan":
            try:
                cells, shas = plan(a.round, root=opt("archive_root"), models_dir=opt("models_dir"),
                                   rehearsal_pairs=a.rehearsal_pairs)
            except F.FinalsError as exc:
                raise CrossRefused(EXIT_REFUSED, str(exc)) from exc
            write_plan(work, cells, shas)
            print(f"{len(cells)} cells planned in {work}")
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
