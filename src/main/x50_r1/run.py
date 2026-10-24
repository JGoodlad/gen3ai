"""The X50 rung-1 extractor driver: logs → labelled rows, deterministic, resumable, chunked (design §6).

    python -m main.x50_r1.run humans --dates 2026-09-21,2026-09-22,2026-09-23 --out <archive>/humans
    python -m main.x50_r1.run traces --run <main>/models/rb_x26_s2601 --min-step 360000000 --out <archive>/traces

* ``humans`` reads the SPECTATOR MANIFEST's logs of the named date folders only (never the archive pulls; design
  §4), both seats each, in sorted order, ``--part-size`` logs per part file.
* ``traces`` reads the run's eval traces of every cycle at or after ``--min-step`` (sorted): the TRAINEE's seat,
  through the spectator view (``mode`` 0) and, when its stream holds no refused choice, its own stream (``mode`` 1,
  the training row; the control). Each battle's spectator decision points must equal the trainee's real request
  positions, and the two reads' labels must agree: a :class:`main.x50_r1.extract.GuardError` otherwise.

Every part is one ``<name>.npz`` (rows of LABELLED decisions only: kind MOVE / SWITCH) + ``<name>.json`` (battle ids,
counters, the commit and the reader's hash); a part whose JSON exists is skipped (resume). Each seat's guarded
decision (``extract.guard_index``) is re-read from its prefix alone. Output never goes under ``data/``.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from main.live.reader import LiveReader, ReaderRefusal
from main.live.replay_scan import battle_lines
from main.x50_r1 import extract as E
from main.x50_r1 import spectate as S
from main.x50_r1.points import PointError, decision_points

MANIFEST = "SPECTATOR_MANIFEST_2026-09-23.txt"
FIELDS = ("rows", "mask", "label", "hp", "forced", "placeholder", "turn", "k", "battle", "seat", "mode",
          "rating_labelled", "rating_viewer", "own_action", "n_legal")
#: the row modes: 0 spectator (R1's public-only own side), 1 full (the trainee's own stream: the training row),
#: 2 COMPLETED (C-GAME, rung 1b step 0), 3 OWN-TRUE (the trainee's TRUE team, synthesised decisions; traces only)
MODE_SPECTATOR, MODE_FULL, MODE_COMPLETED, MODE_OWN_TRUE = 0, 1, 2, 3
_READER: Optional[LiveReader] = None
#: emit the COMPLETED modes (2, 3) beside R1's (rung 1b step 0; ``--complete``)
COMPLETE = False
#: keep only the COMPLETED rows of a human seat (``--rows completed``; the spectator read still runs, as the label
#: guard): the HOM's training extraction, which reads mode 2 alone
COMPLETED_ONLY = False
#: the heldout72 manifest of the TEAM-MEMORY EXCLUSION (F-1b-2, ``memory.py``; ``--exclude-teams``), None = off
EXCLUDE_TEAMS: Optional[str] = None
_MATCHER = None


def _matcher():
    global _MATCHER
    if _MATCHER is None and EXCLUDE_TEAMS is not None:
        from main.x50_r1.memory import HeldoutMatcher
        _MATCHER = HeldoutMatcher(Path(EXCLUDE_TEAMS))
    return _MATCHER


def _reader() -> LiveReader:
    global _READER
    if _READER is None:
        _READER = LiveReader()
    return _READER


def _rating(info: Dict[str, Dict[str, Optional[str]]], seat: str) -> int:
    r = info.get(seat, {}).get("rating")
    return int(r) if r else -1


def _pack(read: E.SeatRead, *, battle: int, seat: str, mode: int, r_lab: int, r_view: int,
          counts: collections.Counter, tag: str, own: Optional[E.CompletedRead] = None) -> Dict[str, np.ndarray]:
    """A seat's rows: every decision with an opponent-action label (use (a)) or, on a completed read, a legal own
    action (use (b)'s label, ``own_action``: the 11-way index; -1 none)."""
    lab, hp = read.labels()
    n = len(read.rows)
    counts[f"{tag}decisions"] += n
    oa = np.asarray(own.own_action, np.int8) if own is not None else np.full(n, -1, np.int8)
    nl = np.asarray([int(np.asarray(m).sum()) for m in read.masks], np.int8)
    keep = (lab[:, 0] != E.KIND_UNKNOWN) | (oa >= 0)
    if own is not None:
        counts[f"{tag}own_action_labelled"] += int((oa >= 0).sum())
        counts[f"{tag}own_action_one_legal"] += int(((oa >= 0) & (nl == 1)).sum())
    for k in range(n):
        counts[f"{tag}label_kind_{int(lab[k, 0])}"] += 1
    counts[f"{tag}labels_hp"] += int(hp.sum())
    counts[f"{tag}placeholder"] += int(sum(read.placeholder))
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return {}
    return {
        "rows": np.stack([read.rows[i] for i in idx]).astype(np.float32),
        "mask": np.stack([read.masks[i] for i in idx]).astype(np.int8),
        "label": lab[idx], "hp": hp[idx],
        "forced": np.array([read.forced[i] for i in idx], bool),
        "placeholder": np.array([read.placeholder[i] for i in idx], bool),
        "turn": np.array([read.turn[i] for i in idx], np.int32),
        "k": idx.astype(np.int32),
        "battle": np.full(len(idx), battle, np.int32),
        "seat": np.full(len(idx), 0 if seat == "p1" else 1, np.int8),
        "mode": np.full(len(idx), mode, np.int8),
        "rating_labelled": np.full(len(idx), r_lab, np.int32),
        "rating_viewer": np.full(len(idx), r_view, np.int32),
        "own_action": oa[idx], "n_legal": nl[idx],
    }


def _guarded_spectator(reader: LiveReader, lines: List[str], seat: str, name: str, battle_id: str,
                       counts: collections.Counter) -> E.SeatRead:
    pts = decision_points(lines, seat)
    read = E.read_spectator(reader, lines, seat, name, points=pts)
    if read.rows:
        k = E.guard_index(battle_id, seat, len(read.rows))
        E.prefix_guard(reader, lines, seat, name, pts, k, read.rows[k])
        counts["guard_checks"] += 1
    return read


_COMPLETER = None


def _completer():
    global _COMPLETER
    if _COMPLETER is None:
        from main.x50_r1.complete import Completer
        _COMPLETER = Completer()
    return _COMPLETER


def completed_seat(reader: LiveReader, lines: List[str], seat: str, name: str, battle_id: str,
                   counts: collections.Counter, tag: str = "c_",
                   team: Optional[str] = None, genders: Optional[Dict[str, str]] = None) -> Optional[E.CompletedRead]:
    """One seat's COMPLETED read (design §2): C-GAME's team (``team`` None) redrawn until every own labelled action
    is legal under it (:data:`complete.MAX_ATTEMPTS`), or the given ``team`` (the trace's TRUE team: no redraw; an
    illegal own action is a refusal). Then the prefix guard on the seat's guarded decision. None = excluded."""
    from main.x50_r1 import complete as C

    pts = decision_points(lines, seat)
    locks = E.decision_locks(lines, seat, pts)
    if team is not None:
        cr = E.read_completed(reader, lines, seat, name, team, genders, points=pts, locks=locks)
        if any(a == -2 for a in cr.own_action):
            raise E.GuardError(f"{seat}: an own action is illegal under the TRUE team's synthesised requests")
        attempt = 0
    else:
        try:
            facts = C.seat_facts(lines, seat)
        except C.CompletionError:
            counts[f"{tag}excluded:facts"] += 1
            return None
        genders = facts.genders
        comp = _completer()
        cr, attempt = None, -1
        keep: Dict[str, str] = {}
        ban: Dict[str, List[str]] = {}
        for attempt in range(C.MAX_ATTEMPTS):
            try:
                team, sets = comp.team(facts, C.seat_rng(battle_id, seat, attempt), keep=keep, ban_items=ban)
            except C.TeamIllegal:
                counts[f"{tag}attempt_team_illegal"] += 1
                keep = {}
                continue
            except C.CompletionError as exc:
                counts[f"{tag}excluded:{'no_prior' if 'no Smogon prior' in str(exc) else 'revealed_illegal'}"] += 1
                return None
            got = E.read_completed(reader, lines, seat, name, team, genders, points=pts, locks=locks)
            bad = [k for k, a in enumerate(got.own_action) if a == -2]
            if not bad:
                cr = got
                break
            counts[f"{tag}attempt_action_illegal"] += 1
            refuted = {got.active[k] for k in bad if got.active[k] is not None}
            for k in bad:
                if got.choice_lock[k] is not None and got.active[k] is not None:
                    ban.setdefault(got.active[k], []).append("choiceband")    # the mon broke a Band lock
            keep = {n: e for n, e in sets.items() if n not in refuted}
        if cr is None:
            counts[f"{tag}excluded:no_consistent_team"] += 1
            return None
        counts[f"{tag}attempts_{min(attempt, 9)}"] += 1
    assert team is not None
    if cr.read.rows:
        k = E.guard_index(battle_id, seat, len(cr.read.rows))
        E.prefix_guard_completed(reader, lines, seat, name, team, genders, pts, locks, k, cr.read.rows[k])
        counts[f"{tag}guard_checks"] += 1
    counts[f"{tag}seats"] += 1
    return cr


# ------------------------------------------------------------------------------------------------ humans

def human_job(args: Tuple[int, str]) -> Dict[str, Any]:
    battle, path = args
    counts: collections.Counter = collections.Counter()
    parts: List[Dict[str, np.ndarray]] = []
    counts["logs"] += 1
    lines = battle_lines(Path(path).read_text(encoding="utf-8", errors="replace"))
    info = E.player_info(lines)
    mt = _matcher()
    if mt is not None:
        hits = mt.matches(lines)
        if hits:                                   # F-1b-2: both seats out, before anything is read
            counts["excluded:team_memory"] += 1
            for _, tid, _ in hits:
                counts[f"team_memory:{tid}"] += 1
            return {"counts": counts, "parts": [], "memory": [path, [list(h) for h in hits]]}
        counts["team_memory_checked"] += 1
    for seat, other in (("p1", "p2"), ("p2", "p1")):
        try:
            read = _guarded_spectator(_reader(), lines, seat, info.get(seat, {}).get("name") or seat, path, counts)
        except (ReaderRefusal, PointError, E.LabelError, RuntimeError) as exc:
            counts[f"refused:{type(exc).__name__}"] += 1
            continue
        counts["seats_read"] += 1
        name = info.get(seat, {}).get("name") or seat
        d = _pack(read, battle=battle, seat=seat, mode=MODE_SPECTATOR, r_lab=_rating(info, other),
                  r_view=_rating(info, seat), counts=counts, tag="")
        if d and not COMPLETED_ONLY:
            parts.append(d)
        if not COMPLETE:
            continue
        try:
            cr = completed_seat(_reader(), lines, seat, name, path, counts)
        except (ReaderRefusal, PointError, E.LabelError, RuntimeError) as exc:
            counts[f"c_refused:{type(exc).__name__}"] += 1
            counts["c_errors"] += 1
            continue
        if cr is None:
            continue
        if not np.array_equal(cr.read.labels()[0], read.labels()[0]):
            counts["c_refused:label_mismatch"] += 1
            continue
        d = _pack(cr.read, battle=battle, seat=seat, mode=MODE_COMPLETED, r_lab=_rating(info, other),
                  r_view=_rating(info, seat), counts=counts, tag="c_", own=cr)
        if d:
            parts.append(d)
    return {"counts": counts, "parts": parts}


def human_paths(corpus: Path, dates: List[str]) -> List[str]:
    """The manifest's logs under the named date folders; an entry ``FROM..TO`` names every folder in that inclusive
    ISO-date range (the training dates: ``2026-05-18..2026-09-20``)."""
    rel = [ln.strip() for ln in (corpus / MANIFEST).read_text().splitlines() if ln.strip()]
    want = {d for d in dates if ".." not in d}
    spans = [tuple(d.split("..", 1)) for d in dates if ".." in d]

    def keep(folder: str) -> bool:
        return folder in want or any(a <= folder <= b for a, b in spans)

    return sorted(str(corpus / r) for r in rel if keep(r.split("/", 1)[0]))


# ------------------------------------------------------------------------------------------------ traces

def trace_summaries(run: Path, min_step: int) -> List[str]:
    out: List[str] = []
    for d in sorted((run / "eval_traces").glob("step_*")):
        if int(d.name.split("_", 1)[1]) >= min_step:
            out.extend(str(p) for p in sorted(d.glob("*/*_summary.json")) if p.parent.name != "random")
    return out


def trace_job(args: Tuple[int, str]) -> Dict[str, Any]:
    battle, sp = args
    counts: collections.Counter = collections.Counter()
    parts: List[Dict[str, np.ndarray]] = []
    sp_p = Path(sp)
    meta = json.loads(sp_p.read_text())["meta"]
    ts = meta["trace_source"]
    d = sp_p.parent
    seat = ts["trainee_side"]
    _, s1 = S.read_stream(d / ts["records"]["p1"])
    _, s2 = S.read_stream(d / ts["records"]["p2"])
    s1, s2 = battle_lines("\n".join(s1)), battle_lines("\n".join(s2))
    own = s1 if seat == "p1" else s2
    counts["logs"] += 1
    try:
        view = S.spectator_view(s1, s2)
        pts = decision_points(view.lines, seat)
        real = [(view.public_pos(seat, q), f) for q, f in S.request_points(own)]
        refused_choice = any(ln.startswith("|error|") for ln in own)
        if not refused_choice and [(p, k == "forced") for p, k in pts] != real:
            raise E.GuardError(f"{sp}: spectator decision points differ from the trainee's real requests")
        name = E.player_info(view.lines).get(seat, {}).get("name") or seat
        reader = _reader()
        spec = _guarded_spectator(reader, view.lines, seat, name, sp, counts)
        rec = json.loads((d / ts["reconstruction"]).read_text())
        team = [json.loads(x.split(" ", 2)[2])["team"] for x in rec["input_log"] if x.startswith(f">player {seat}")][0]
        full = None
        if refused_choice:
            counts["full_skipped_refused_choice"] += 1
        else:
            full = E.read_full(reader, own, seat, name, team)
            if not np.array_equal(spec.labels()[0], full.labels()[0]):
                raise E.GuardError(f"{sp}: the spectator and full reads' labels differ")
    except (ReaderRefusal, PointError, S.SpectateError, E.LabelError, E.GuardError, RuntimeError) as exc:
        counts[f"refused:{type(exc).__name__}"] += 1
        return {"counts": counts, "parts": parts, "error": f"{sp}: {exc}"[:400]}
    counts["seats_read"] += 1
    for mode, read in ((MODE_SPECTATOR, spec), (MODE_FULL, full)):
        if read is None:
            continue
        dd = _pack(read, battle=battle, seat=seat, mode=mode, r_lab=-1, r_view=-1, counts=counts,
                   tag="" if mode == 0 else "full_")
        if dd:
            parts.append(dd)
    if COMPLETE:
        parts.extend(_trace_completed(battle, sp, view.lines, seat, name, own, team, spec, counts))
    return {"counts": counts, "parts": parts}


def _real_genders(own: List[str]) -> Dict[str, str]:
    """The genders the simulator drew for the trainee's team (its first request): part of the TRUE team."""
    for ln in own:
        if ln.startswith("|request|") and ln[len("|request|"):].strip():
            q = json.loads(ln[len("|request|"):])
            return {m["ident"].split(": ", 1)[1]: next((x for x in m["details"].split(", ")[1:] if x in ("M", "F")), "")
                    for m in q["side"]["pokemon"]}
    return {}


def _trace_completed(battle: int, sp: str, lines: List[str], seat: str, name: str, own: List[str], team: str,
                     spec: E.SeatRead, counts: collections.Counter) -> List[Dict[str, np.ndarray]]:
    """The trainee's seat COMPLETED (mode 2: C-GAME from the public log) and OWN-TRUE (mode 3: its true team,
    synthesised decisions). The true team must pass the validator (the design's stop rule: a defect, never an
    exclusion)."""
    from main.value_search.world_legality import LegalityRuleError

    out: List[Dict[str, np.ndarray]] = []
    try:
        _completer().L.require_true_team(team, sp)
    except LegalityRuleError:
        counts["true_team_illegal"] += 1
        raise
    for mode, tag, kw in ((MODE_COMPLETED, "c_", {}),
                          (MODE_OWN_TRUE, "t_", {"team": team, "genders": _real_genders(own)})):
        try:
            cr = completed_seat(_reader(), lines, seat, name, sp, counts, tag=tag, **kw)
        except (ReaderRefusal, PointError, E.LabelError, RuntimeError) as exc:
            counts[f"{tag}refused:{type(exc).__name__}"] += 1
            continue
        if cr is None:
            continue
        if not np.array_equal(cr.read.labels()[0], spec.labels()[0]):
            counts[f"{tag}refused:label_mismatch"] += 1
            continue
        dd = _pack(cr.read, battle=battle, seat=seat, mode=mode, r_lab=-1, r_view=-1, counts=counts, tag=tag, own=cr)
        if dd:
            out.append(dd)
    return out


# ------------------------------------------------------------------------------------------------ driver

def _concat(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not parts:
        return {}
    return {f: np.concatenate([p[f] for p in parts]) for f in FIELDS}


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_part(out: Path, name: str, ids: List[str], results: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    counts: collections.Counter = collections.Counter()
    parts: List[Dict[str, np.ndarray]] = []
    errors = []
    memory = []
    for r in results:
        counts.update(r["counts"])
        parts.extend(r["parts"])
        if r.get("error"):
            errors.append(r["error"])
        if r.get("memory"):
            memory.append(r["memory"])
    arrays = _concat(parts)
    np.savez_compressed(out / f"{name}.npz", **arrays)  # type: ignore[arg-type]
    body = {"part": name, "battles": ids, "counts": dict(sorted(counts.items())), "errors": errors,
            "team_memory_excluded": memory,
            "labelled_rows": int(len(arrays.get("k", []))), **meta}
    tmp = out / f"{name}.json.tmp"
    tmp.write_text(json.dumps(body, indent=1))
    os.replace(tmp, out / f"{name}.json")


def under_data(path: Path) -> bool:
    """Is ``path`` inside this checkout's or the main checkout's ``data/``?"""
    from utils.git import get_main_repo_root
    from utils.paths import repo_path

    roots = [repo_path("data").resolve(), (Path(get_main_repo_root()) / "data").resolve()]
    return any(path == r or r in path.parents for r in roots)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x50_r1.run", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("humans")
    h.add_argument("--dates", required=True)
    h.add_argument("--corpus", default=None, help="default: the main checkout's replays/showdown/gen3ou")
    h.add_argument("--rows", choices=("all", "completed"), default="all",
                   help="completed: keep only the completed rows (mode 2); needs --complete")
    h.add_argument("--exclude-teams", default=None,
                   help="heldout72_hpfix_manifest.json: the team-memory exclusion (F-1b-2), both seats")
    t = sub.add_parser("traces")
    t.add_argument("--run", required=True)
    t.add_argument("--min-step", type=int, required=True)
    for p in (h, t):
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=3)
        p.add_argument("--part-size", type=int, default=500)
        p.add_argument("--deadline", type=float, default=None, help="epoch seconds: start no part after it")
        p.add_argument("--complete", action="store_true",
                       help="also emit the completed own side (modes 2 / 3; rung 1b step 0)")
    a = ap.parse_args(argv)
    global COMPLETE, COMPLETED_ONLY, EXCLUDE_TEAMS
    COMPLETE = bool(a.complete)
    if a.cmd == "humans":
        COMPLETED_ONLY = a.rows == "completed"
        if COMPLETED_ONLY and not COMPLETE:
            print("[x50_r1] --rows completed needs --complete", file=sys.stderr)
            return 2
        EXCLUDE_TEAMS = a.exclude_teams
    out = Path(a.out).resolve()
    if under_data(out):
        print("[x50_r1] refusing an output under a checkout's data/ (design §6)", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    from utils.bridge.sim_bridge_bin import resolve_live_reader_bin
    from utils.git import get_git_hash
    meta = {"commit": get_git_hash(), "live_reader_sha256": _sha(resolve_live_reader_bin())}
    if a.cmd == "humans":
        from main.live.replay_scan import default_corpus
        corpus = Path(a.corpus or default_corpus() or "")
        items = human_paths(corpus, [x for x in a.dates.split(",") if x])
        job, prefix = human_job, "humans"
    else:
        items = trace_summaries(Path(a.run), a.min_step)
        job, prefix = trace_job, "traces"
    chunks = [items[i:i + a.part_size] for i in range(0, len(items), a.part_size)]
    print(f"[x50_r1] {a.cmd}: {len(items)} battles in {len(chunks)} parts -> {out}", flush=True)
    t0 = time.time()
    meta["complete"] = COMPLETE
    meta["rows"] = "completed" if COMPLETED_ONLY else "all"
    if _matcher() is not None:
        from main.x50_r1.memory import NEAR
        meta["exclude_teams"] = {"manifest": str(EXCLUDE_TEAMS), "sha256": _matcher().sha256, "near": NEAR}
    with Pool(a.workers) as pool:
        for ci, chunk in enumerate(chunks):
            name = f"{prefix}_{ci:04d}"
            if (out / f"{name}.json").exists():
                continue
            if a.deadline is not None and time.time() >= a.deadline:
                print(f"[x50_r1] deadline reached before {name}", flush=True)
                return 3
            base = ci * a.part_size
            results = pool.map(job, [(base + i, p) for i, p in enumerate(chunk)], chunksize=8)
            write_part(out, name, chunk, results, {**meta, "part_size": a.part_size})
            print(f"[x50_r1] {name} done ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
