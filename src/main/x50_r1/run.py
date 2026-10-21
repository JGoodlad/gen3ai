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
          "rating_labelled", "rating_viewer")
_READER: Optional[LiveReader] = None


def _reader() -> LiveReader:
    global _READER
    if _READER is None:
        _READER = LiveReader()
    return _READER


def _rating(info: Dict[str, Dict[str, Optional[str]]], seat: str) -> int:
    r = info.get(seat, {}).get("rating")
    return int(r) if r else -1


def _pack(read: E.SeatRead, *, battle: int, seat: str, mode: int, r_lab: int, r_view: int,
          counts: collections.Counter, tag: str) -> Dict[str, np.ndarray]:
    lab, hp = read.labels()
    n = len(read.rows)
    counts[f"{tag}decisions"] += n
    keep = lab[:, 0] != E.KIND_UNKNOWN
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


# ------------------------------------------------------------------------------------------------ humans

def human_job(args: Tuple[int, str]) -> Dict[str, Any]:
    battle, path = args
    counts: collections.Counter = collections.Counter()
    parts: List[Dict[str, np.ndarray]] = []
    counts["logs"] += 1
    lines = battle_lines(Path(path).read_text(encoding="utf-8", errors="replace"))
    info = E.player_info(lines)
    for seat, other in (("p1", "p2"), ("p2", "p1")):
        try:
            read = _guarded_spectator(_reader(), lines, seat, info.get(seat, {}).get("name") or seat, path, counts)
        except (ReaderRefusal, PointError, E.LabelError, RuntimeError) as exc:
            counts[f"refused:{type(exc).__name__}"] += 1
            continue
        counts["seats_read"] += 1
        d = _pack(read, battle=battle, seat=seat, mode=0, r_lab=_rating(info, other), r_view=_rating(info, seat),
                  counts=counts, tag="")
        if d:
            parts.append(d)
    return {"counts": counts, "parts": parts}


def human_paths(corpus: Path, dates: List[str]) -> List[str]:
    want = set(dates)
    rel = [ln.strip() for ln in (corpus / MANIFEST).read_text().splitlines() if ln.strip()]
    return sorted(str(corpus / r) for r in rel if r.split("/", 1)[0] in want)


# ------------------------------------------------------------------------------------------------ traces

def trace_summaries(run: Path, min_step: int) -> List[str]:
    out = []
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
    for mode, read in ((0, spec), (1, full)):
        if read is None:
            continue
        dd = _pack(read, battle=battle, seat=seat, mode=mode, r_lab=-1, r_view=-1, counts=counts,
                   tag="" if mode == 0 else "full_")
        if dd:
            parts.append(dd)
    return {"counts": counts, "parts": parts}


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
    for r in results:
        counts.update(r["counts"])
        parts.extend(r["parts"])
        if r.get("error"):
            errors.append(r["error"])
    arrays = _concat(parts)
    np.savez_compressed(out / f"{name}.npz", **arrays)
    body = {"part": name, "battles": ids, "counts": dict(sorted(counts.items())), "errors": errors,
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
    t = sub.add_parser("traces")
    t.add_argument("--run", required=True)
    t.add_argument("--min-step", type=int, required=True)
    for p in (h, t):
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=3)
        p.add_argument("--part-size", type=int, default=500)
        p.add_argument("--deadline", type=float, default=None, help="epoch seconds: start no part after it")
    a = ap.parse_args(argv)
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
