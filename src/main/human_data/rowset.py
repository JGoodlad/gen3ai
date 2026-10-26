"""The ROWSET builder: a registered snapshot → filtered, completed, labelled training rows (plan §3; X59-S).

    python -m main.human_data.rowset scan    --snapshot hr_spec_v1 --out <dir> [--workers 3]
    python -m main.human_data.rowset extract --out <dir> [--workers 3] [--deadline <epoch s>]   # resumable parts
    python -m main.human_data.rowset compact --out <dir> --id hr_spec_v1.r1

The rules run in the plan's order (``designs/endstate/design_human_replay_lever.md`` §3), every one on the log BEFORE
anything is completed, and every count lands in the output:

1. **scan** (text only, :mod:`.textpass`): the snapshot's MANIFEST is read and every log re-hashed against it
   (:mod:`.snapshots`; a mismatch REFUSES the whole build). Kept: TRAIN dates (<= :data:`TRAIN_LAST`), rated, a
   result, >= :data:`MIN_TURNS` turns (REVIEW D M7's information floor, which also drops every walk-over), the first
   of each duplicate body, no side NEAR a held-out panel team (>= 3 move-revealed fits, every held-out panel,
   ``forum216b`` included: plan §3.1). A SEAT is read when its account is not bot-flagged and its pre-battle rating is
   >= :data:`RATING_MIN`. The kept / excluded population profile (forfeits, short games, sides seen, pool-unfamiliar
   is a later read) is written to ``scan.json`` (REVIEW D M7).
2. **extract** (the Rust live reader): each kept seat COMPLETED by C-GAME (``main.x50_r1.run.completed_seat``,
   battle key = the log's absolute path in the main corpus, so the fill equals the HOM extraction's), one row per
   decision with >= 2 legal actions and either a KNOWN own action (the label) or a HIDDEN MOVE (plan §3.2: a ``cant``
   other than a recharge, a confusion self-hit, or the active fainting before it acted, with no foe Pursuit in the
   window, prove a move was chosen; such a row's label is the SET of legal moves). Per row it also records the
   CONSISTENT set (REVIEW D M2): the legal actions the log leaves possible for the human, i.e. no move the log never
   showed this mon use and no switch to a mon the log never showed. A row whose human label is outside that set
   (a Transform / Mimic slot) is dropped and counted. The quitter's last :data:`QUIT_DROP` decisions are dropped
   (plan §3.5).
3. **compact**: accounts split TRAIN / VALIDATION by a salted hash (plan §3.3: ``acct mod 50 == 0`` validates),
   shards of :data:`SHARD` rows, and ``rows_manifest.json`` (every shard's sha256, the counts, the filter spec, the
   code and encoder identity); ITS sha256 is the rowset's identity in the registry.

Output never goes under a checkout's ``data/`` or ``models/``. CPU only.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from main.human_data import snapshots as S
from main.human_data import textpass as T

TRAIN_LAST = "2026-09-20"
RATING_MIN = 1300
MIN_TURNS = 11          # a battle of <= 10 turns is dropped (REVIEW D M7; it drops every walk-over too)
QUIT_DROP = 2
VAL_MOD = 50
SHARD = 16384
PART_SIZE = 400
N_ACT = 11
SCHEMA = "gen3_human_rowset_v1"
FIELDS = ("rows", "mask", "act", "hidden", "cons", "moveset", "rid", "bkey", "acct", "seat", "k", "turn", "forced",
          "rating", "quit_dist", "end")
END_CODE = {"ko": 0, "forfeit": 1, "timer": 2, "tie": 3, "none": 4}


class RowsetError(RuntimeError):
    """A rowset input or output that does not match what it claims (a defect: stop)."""


def key64(s: str) -> int:
    """A stable int64 key of a string (rows, battles, accounts)."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)


def is_val_key(acct_key: int) -> bool:
    """The account-disjoint validation split (plan §3.3): the salted account id's :func:`key64` mod 50 == 0."""
    return int(acct_key) % VAL_MOD == 0


# ------------------------------------------------------------------------------------------------ the windows

def _subject(tok: str) -> Optional[Tuple[str, str]]:
    head, sep, name = tok.partition(": ")
    if not sep or len(head) < 2 or head[0] != "p" or head[1] not in "12":
        return None
    return head[:2], name.strip()


def hidden_reasons(lines: Sequence[str], seat: str, points: Sequence[Tuple[int, str]]) -> List[Optional[str]]:
    """Per decision, why the log proves a MOVE was chosen although none is shown: ``cant`` (a ``cant`` other than a
    recharge), ``confusion`` (a confusion self-hit), ``faint`` (the active fainted before acting); ``pursuit`` marks
    a faint after a foe's Pursuit in the same window (the human may have been switching: EXCLUDED, plan §3.2); None
    when the window shows the action or proves nothing. The window and its stopping events are
    ``main.x50_r1.own.viewer_actions``'s (an own switch / move / cant / faint ends it); a forced decision is never a
    hidden move."""
    from main.x50_r1.own import CALLERS, toid

    out: List[Optional[str]] = []
    for k, (pos, kind) in enumerate(points):
        end = points[k + 1][0] if k + 1 < len(points) else len(lines)
        why: Optional[str] = None
        if kind == "forced":
            out.append(None)
            continue
        foe_pursuit = False
        for ln in lines[pos:end]:
            p = ln.split("|")
            if len(p) < 3:
                continue
            sn = _subject(p[2])
            if sn is None:
                continue
            kw = p[1]
            if sn[0] != seat:
                if kw == "move" and len(p) > 3 and toid(p[3]) == "pursuit":
                    foe_pursuit = True
                continue
            if kw == "switch":
                break
            if kw == "move" and len(p) > 3:
                frm = next((t[len("[from]"):].strip() for t in p[3:] if t.startswith("[from]")), None)
                if frm is not None and toid(frm.replace("move:", "")) in CALLERS:
                    continue                      # a called move: its caller's line came first
                break
            if kw == "cant":
                why = None if (len(p) > 3 and toid(p[3]) == "recharge") else "cant"
                break
            if kw == "-damage" and any(t.strip() == "[from] confusion" for t in p[3:]):
                why = "confusion"
                break
            if kw == "faint":
                why = "pursuit" if foe_pursuit else "faint"
                break
        out.append(why)
    return out


def request_roster(req: str) -> Tuple[Optional[str], Dict[str, str]]:
    """(the active's species id, {nickname: species id}) of a synthesised ``|request|`` line."""
    q = json.loads(req[len("|request|"):])
    roster: Dict[str, str] = {}
    active = None
    for m in q["side"]["pokemon"]:
        nick = m["ident"].split(": ", 1)[1]
        sid = T.toid(m["details"].split(",")[0])
        roster[nick] = sid
        if m.get("active"):
            active = sid
    return active, roster


def consistent_mask(tokens: Dict[int, str], mask: Sequence[int], req: str,
                    reveals: Dict[str, Tuple[List[str], Optional[str], Optional[str]]]) -> np.ndarray:
    """The legal actions the LOG leaves possible for the human (REVIEW D M2): a move the log showed THIS mon use
    (a bare Hidden Power covers a typed one; Struggle always), a switch to a mon the log showed. A move C-GAME drew
    and a bench mon it drew are never consistent."""
    active, roster = request_roster(req)
    rev_moves = set(reveals.get(active or "", ([], None, None))[0]) if active else set()
    out = np.zeros(N_ACT, np.int8)
    for i, tok in tokens.items():
        if not int(mask[i]):
            continue
        if tok.startswith("move "):
            mid = tok[5:]
            ok = (mid == "struggle" or mid in rev_moves
                  or (mid.startswith("hiddenpower") and "hiddenpower" in rev_moves))
        elif tok.startswith("switch "):
            ok = roster.get(tok[7:]) in reveals
        else:
            ok = False
        out[i] = 1 if ok else 0
    return out


def moveset_mask(tokens: Dict[int, str], mask: Sequence[int]) -> np.ndarray:
    """The legal MOVE actions (a hidden move's set label, plan §3.2)."""
    out = np.zeros(N_ACT, np.int8)
    for i, tok in tokens.items():
        if int(mask[i]) and tok.startswith("move "):
            out[i] = 1
    return out


# ------------------------------------------------------------------------------------------------ scan

def _scan_job(args) -> Dict[str, Any]:
    snap, population, corpus, item = args
    b = S.read_verified(snap, population, Path(corpus), tuple(item))
    rel = item[0]
    rec = T.record(b.decode(errors="replace"))
    return {"rel": rel, "sha": item[1], "n": item[2], "rec": rec}


_MATCHER: Optional[T.PanelMatcher] = None


def _panel_job(args) -> Dict[str, Any]:
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = T.PanelMatcher()
    rel, reveals = args
    return {"rel": rel, "fits": {seat: _MATCHER.side_max_fits(mons) for seat, mons in reveals.items()}}


def scan(snap: str, out: Path, workers: int, store: Path, registry: Path, corpus: Path) -> Dict[str, Any]:
    """Pass 1 (module docstring). Writes ``scan.json`` and ``keep.jsonl.gz``."""
    row = S.registry_row(snap, registry)
    items = S.read_manifest(snap, store, registry)
    out.mkdir(parents=True, exist_ok=True)
    recs: List[Dict[str, Any]] = []
    with Pool(workers) as pool:
        for r in pool.imap(_scan_job, [(snap, row["population"], str(corpus), list(it)) for it in items],
                           chunksize=256):
            recs.append(r)
    all_recs = [r["rec"] for r in recs if r["rec"] is not None]
    flags = T.bot_flags(T.accounts(all_recs))
    salt = T.load_salt()
    c: collections.Counter = collections.Counter()
    seen_body: set = set()
    cand: List[Dict[str, Any]] = []
    for r in recs:
        rec = r["rec"]
        c["logs"] += 1
        if rec is None:
            c["drop:no_player"] += 1
            continue
        if r["rel"][:10] > TRAIN_LAST:
            c["drop:held_out_date"] += 1
            continue
        if not rec["rated"]:
            c["drop:unrated"] += 1
            continue
        if rec["end"] == "none" or (rec["win"] is None and rec["end"] != "tie"):
            c["drop:no_result"] += 1
            continue
        if rec["body"] in seen_body:
            c["drop:duplicate"] += 1
            continue
        seen_body.add(rec["body"])
        if rec["turns"] < MIN_TURNS:
            c["drop:short_le10_turns"] += 1
            c[f"drop:short_le10_turns:{rec['end']}"] += 1
            continue
        seats = [s for s in ("p1", "p2") if s in rec["names"]
                 and T.toid(rec["names"][s]) not in flags and (rec["rating"].get(s) or 0) >= RATING_MIN]
        if not seats:
            c["drop:no_eligible_seat"] += 1
            continue
        cand.append({**r, "seats": seats})
    with Pool(workers) as pool:
        fits = {f["rel"]: f["fits"] for f in pool.imap(
            _panel_job, [(r["rel"], r["rec"]["reveals"]) for r in cand], chunksize=64)}
    keep: List[Dict[str, Any]] = []
    prof: Dict[str, collections.Counter] = {"kept": collections.Counter(), "excluded": collections.Counter()}
    for r in cand:
        rec = r["rec"]
        f = fits[r["rel"]]
        near = sorted({p for s in f for p, n in f[s].items() if n >= T.NEAR})
        side = "excluded" if near else "kept"
        pc = prof[side]
        pc["battles"] += 1
        pc["forfeit_or_timer"] += int(rec["end"] in ("forfeit", "timer"))
        pc["turns_sum"] += rec["turns"]
        for s in ("p1", "p2"):
            nseen = len(rec["reveals"][s])
            pc["sides"] += 1
            pc["mons_seen_sum"] += nseen
            pc["full_six_sides"] += int(nseen == 6)
        if near:
            c["drop:panel_near"] += 1
            for p in near:
                c[f"panel_near:{p}"] += 1
            continue
        names = rec["names"]
        keep.append({"rel": r["rel"], "sha": r["sha"], "n": r["n"], "seats": r["seats"], "end": rec["end"],
                     "quit": rec["quit"], "turns": rec["turns"],
                     "rating": {s: rec["rating"].get(s) for s in r["seats"]},
                     "acct": {s: T.acct(names[s], salt) for s in r["seats"]},
                     "max_fits": {s: f[s] for s in r["seats"]}})
        c["kept_battles"] += 1
        c["kept_seats"] += len(r["seats"])
    with gzip.open(out / "keep.jsonl.gz", "wt") as fh:
        for k in keep:
            fh.write(json.dumps(k, separators=(",", ":")) + "\n")
    profile = {side: {"battles": pc["battles"],
                      "forfeit_or_timer": round(pc["forfeit_or_timer"] / max(pc["battles"], 1), 4),
                      "mean_turns": round(pc["turns_sum"] / max(pc["battles"], 1), 2),
                      "mons_seen_per_side": round(pc["mons_seen_sum"] / max(pc["sides"], 1), 3),
                      "full_six_sides": round(pc["full_six_sides"] / max(pc["sides"], 1), 4)}
               for side, pc in prof.items()}
    res = {"schema": SCHEMA + ":scan", "snapshot": snap, "manifest_sha256": row["manifest_sha256"],
           "panels": _panel_set(), "near": T.NEAR,
           "rating_min": RATING_MIN, "min_turns": MIN_TURNS, "train_last": TRAIN_LAST,
           "bot_flagged_accounts": len(flags), "counts": dict(sorted(c.items())),
           "panel_exclusion_profile": profile,
           "panel_exclusion_share": round(c["drop:panel_near"] / max(len(cand), 1), 4),
           "keep_sha256": S_sha(out / "keep.jsonl.gz"), "code": _code_identity()}
    (out / "scan.json").write_text(json.dumps(res, indent=1))
    return res


def _panel_set() -> List[str]:
    from main import heldout_panels as HP

    return sorted(HP.panel_names(held_out_only=True))


def S_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ extract

def seat_rows(lines: List[str], seat: str, name: str, battle_id: str, rel: str, k_meta: Dict[str, Any],
              reveals: Dict[str, Tuple[List[str], Optional[str], Optional[str]]],
              counts: collections.Counter) -> Optional[Dict[str, np.ndarray]]:
    """One kept seat's rows (module docstring, step 2); None when C-GAME excludes the seat."""
    from main.x50_r1 import run as R
    from main.x50_r1.points import decision_points

    cr = R.completed_seat(R._reader(), lines, seat, name, battle_id, counts, tag="c_")
    if cr is None:
        return None
    pts = decision_points(lines, seat)
    if len(pts) != len(cr.read.rows):
        raise RowsetError(f"{rel} {seat}: {len(pts)} points != {len(cr.read.rows)} completed rows")
    why = hidden_reasons(lines, seat, pts)
    n = len(pts)
    quitter = k_meta["quit"] == seat
    keep_idx, act, hid, cons, mset = [], [], [], [], []
    for k in range(n):
        m = np.asarray(cr.read.masks[k], np.int8)
        nl = int(m.sum())
        a = int(cr.own_action[k])
        if a == -2:
            raise RowsetError(f"{rel} {seat} decision {k}: an illegal own action survived C-GAME")
        if nl < 2:
            counts["row_drop:one_legal"] += int(a >= 0 or why[k] is not None)
            continue
        if quitter and k >= n - QUIT_DROP:
            counts["row_drop:quitter_tail"] += 1
            continue
        hidden = a < 0 and why[k] in ("cant", "confusion", "faint")
        if a < 0 and not hidden:
            counts[f"row_drop:unlabelled:{why[k] or 'none'}"] += 1
            continue
        c = consistent_mask(cr.read.tokens[k], m, cr.requests[k], reveals)
        if a >= 0 and not c[a]:
            counts["row_drop:label_outside_consistent"] += 1
            continue
        ms = moveset_mask(cr.read.tokens[k], m)
        if hidden and not ms.any():
            counts["row_drop:hidden_no_move"] += 1
            continue
        if hidden:
            counts[f"row_hidden:{why[k]}"] += 1
        keep_idx.append(k)
        act.append(a)
        hid.append(hidden)
        cons.append(c)
        mset.append(ms)
    if not keep_idx:
        return {}
    acct_id = k_meta["acct"][seat]
    nk = len(keep_idx)
    return {
        "rows": np.stack([cr.read.rows[i] for i in keep_idx]).astype(np.float32),
        "mask": np.stack([np.asarray(cr.read.masks[i], np.int8) for i in keep_idx]),
        "act": np.asarray(act, np.int8), "hidden": np.asarray(hid, bool),
        "cons": np.stack(cons), "moveset": np.stack(mset),
        "rid": np.asarray([key64(f"{rel}|{seat}|{i}") for i in keep_idx], np.int64),
        "bkey": np.full(nk, key64(rel), np.int64), "acct": np.full(nk, key64(acct_id), np.int64),
        "seat": np.full(nk, 0 if seat == "p1" else 1, np.int8),
        "k": np.asarray(keep_idx, np.int16), "turn": np.asarray([cr.read.turn[i] for i in keep_idx], np.int16),
        "forced": np.asarray([cr.read.forced[i] for i in keep_idx], bool),
        "rating": np.full(nk, int(k_meta["rating"][seat]), np.int16),
        "quit_dist": np.asarray([(n - 1 - i) if quitter else -1 for i in keep_idx], np.int16),
        "end": np.full(nk, END_CODE[k_meta["end"]], np.int8),
    }


def _extract_job(args) -> Dict[str, Any]:
    from main.live.reader import ReaderRefusal
    from main.live.replay_scan import battle_lines
    from main.x50_r1 import extract as E
    from main.x50_r1.points import PointError

    snap, population, corpus, k_meta = args
    counts: collections.Counter = collections.Counter()
    b = S.read_verified(snap, population, Path(corpus), (k_meta["rel"], k_meta["sha"], k_meta["n"]))
    text = b.decode("utf-8", errors="replace")
    lines = battle_lines(text)
    rec = T.record(text)
    info = E.player_info(lines)
    battle_id = str(Path(corpus) / k_meta["rel"])
    parts = []
    for seat in k_meta["seats"]:
        name = info.get(seat, {}).get("name") or seat
        try:
            d = seat_rows(lines, seat, name, battle_id, k_meta["rel"], k_meta, rec["reveals"][seat], counts)
        except RowsetError:
            raise                                       # a defect of this builder: stop
        except (ReaderRefusal, PointError, E.LabelError, E.GuardError, RuntimeError) as exc:
            # the rung-1b extraction's refusals (an OwnError the own-side tracker raises, a reader refusal): the seat
            # is out and counted, exactly as ``main.x50_r1.run.human_job`` counts them
            counts[f"seat_refused:{type(exc).__name__}"] += 1
            continue
        if d is None:
            counts["seat_excluded:c_game"] += 1
            continue
        counts["seats_read"] += 1
        if d:
            parts.append(d)
    return {"counts": counts, "parts": parts}


def _concat(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {f: np.concatenate([p[f] for p in parts]) for f in FIELDS} if parts else {}


def extract(out: Path, workers: int, deadline: Optional[float], store: Path, registry: Path, corpus: Path,
            part_size: int = PART_SIZE, limit: Optional[int] = None) -> int:
    """Pass 2 (module docstring): parts of ``part_size`` kept battles, resumable. 0 = done, 3 = deadline."""
    scan_res = json.loads((out / "scan.json").read_text())
    if S_sha(out / "keep.jsonl.gz") != scan_res["keep_sha256"]:
        raise RowsetError("keep.jsonl.gz differs from the scan's record")
    snap = scan_res["snapshot"]
    row = S.registry_row(snap, registry)
    if row["manifest_sha256"] != scan_res["manifest_sha256"]:
        raise RowsetError("the registry's manifest digest moved since the scan")
    if scan_res["code"]["source_sha256"] != source_sha256():
        raise RowsetError("the row code changed since the scan (source_sha256): re-run the scan")
    with gzip.open(out / "keep.jsonl.gz", "rt") as fh:
        keep = [json.loads(x) for x in fh]
    if limit is not None:
        keep = keep[:limit]
    pdir = out / "parts"
    pdir.mkdir(exist_ok=True)
    chunks = [keep[i:i + part_size] for i in range(0, len(keep), part_size)]
    meta = _code_identity()
    t0 = time.time()
    with Pool(workers) as pool:
        for ci, chunk in enumerate(chunks):
            name = f"part_{ci:04d}"
            if (pdir / f"{name}.json").exists():
                continue
            if deadline is not None and time.time() >= deadline:
                print(f"[rowset] deadline reached before {name}", flush=True)
                return 3
            res = pool.map(_extract_job, [(snap, row["population"], str(corpus), k) for k in chunk], chunksize=4)
            counts: collections.Counter = collections.Counter()
            parts = []
            for r in res:
                counts.update(r["counts"])
                parts.extend(r["parts"])
            arr = _concat(parts)
            tmp = pdir / f"{name}.tmp.npz"
            np.savez_compressed(tmp, **arr)  # type: ignore[arg-type]
            os.replace(tmp, pdir / f"{name}.npz")
            body = {"part": name, "battles": [k["rel"] for k in chunk], "counts": dict(sorted(counts.items())),
                    "rows": int(len(arr.get("act", []))), "npz_sha256": S_sha(pdir / f"{name}.npz"), **meta}
            (pdir / f"{name}.json.tmp").write_text(json.dumps(body, indent=1))
            os.replace(pdir / f"{name}.json.tmp", pdir / f"{name}.json")
            print(f"[rowset] {name}: {body['rows']} rows ({time.time() - t0:.0f} s)", flush=True)
    return 0


#: the source files whose bytes make a row (REVIEW D m2: a commit hash alone does not pin an uncommitted tree)
ROW_SOURCES = ("main/human_data/rowset.py", "main/human_data/textpass.py", "main/human_data/snapshots.py",
               "main/x50_r1/run.py", "main/x50_r1/extract.py", "main/x50_r1/own.py", "main/x50_r1/complete.py",
               "main/x50_r1/points.py", "main/x50_r1/memory.py", "main/live/reader.py", "main/live/replay_scan.py",
               "agents/info_levers/revealed.py")


def source_sha256() -> str:
    from utils.paths import src_path

    h = hashlib.sha256()
    for rel in ROW_SOURCES:
        h.update(rel.encode() + b"\0" + src_path(*rel.split("/")).read_bytes() + b"\0")
    return h.hexdigest()


def _code_identity() -> Dict[str, Any]:
    import subprocess

    from main.policy_spectrum.bank import encoder_identity
    from utils.bridge.sim_bridge_bin import resolve_live_reader_bin
    from utils.git import get_git_hash
    from utils.paths import src_path

    enc = encoder_identity()
    try:
        dirty = bool(subprocess.run(["git", "-C", str(src_path()), "status", "--porcelain", "--", *ROW_SOURCES],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    return {"commit": get_git_hash(), "source_sha256": source_sha256(), "sources_dirty_vs_commit": dirty,
            "live_reader_sha256": S_sha(Path(resolve_live_reader_bin())), "encoder_identity": enc,
            "layout_hash": hashlib.sha256(json.dumps(enc, sort_keys=True).encode()).hexdigest()}


# ------------------------------------------------------------------------------------------------ compact

def _write_shard(d: Path, name: str, arr: Dict[str, np.ndarray]) -> Dict[str, Any]:
    p = d / f"{name}.npz"
    tmp = d / f"{name}.tmp.npz"
    np.savez_compressed(tmp, **arr)  # type: ignore[arg-type]
    os.replace(tmp, p)
    return {"name": name, "rows": int(len(arr["act"])), "sha256": S_sha(p),
            "hidden": int(arr["hidden"].sum()), "battles": int(len(np.unique(arr["bkey"])))}


def compact(out: Path, rowset_id: str) -> Dict[str, Any]:
    """Pass 3 (module docstring): ``<out>/rows/`` shards + ``rows_manifest.json``."""
    scan_res = json.loads((out / "scan.json").read_text())
    pdir = out / "parts"
    with gzip.open(out / "keep.jsonl.gz", "rt") as fh:
        n_keep = sum(1 for _ in fh)
    parts = sorted(pdir.glob("part_*.json"))
    if len(parts) != -(-n_keep // PART_SIZE):
        raise RowsetError(f"{len(parts)} parts, but {n_keep} kept battles need {-(-n_keep // PART_SIZE)}")
    rdir = out / "rows"
    rdir.mkdir(exist_ok=True)
    counts: collections.Counter = collections.Counter()
    metas = []
    train_buf: List[Dict[str, np.ndarray]] = []
    val_buf: List[Dict[str, np.ndarray]] = []
    shards: List[Dict[str, Any]] = []
    n_buf = 0
    for js in parts:
        meta = json.loads(js.read_text())
        npz = js.with_suffix(".npz")
        if S_sha(npz) != meta["npz_sha256"]:
            raise RowsetError(f"{npz}: sha256 differs from its part record")
        metas.append({k: meta[k] for k in ("part", "rows", "npz_sha256", "commit", "source_sha256",
                                           "sources_dirty_vs_commit", "live_reader_sha256", "layout_hash")})
        counts.update(meta["counts"])
        if not meta["rows"]:
            continue
        z = np.load(npz)
        arr = {f: z[f] for f in FIELDS}
        val = np.array([is_val_key(a) for a in arr["acct"]], bool)
        val_buf.append({f: arr[f][val] for f in FIELDS})
        train_buf.append({f: arr[f][~val] for f in FIELDS})
        n_buf += int((~val).sum())
        while n_buf >= SHARD:
            cat = _concat(train_buf)
            shards.append(_write_shard(rdir, f"train_{len(shards):04d}", {f: cat[f][:SHARD] for f in FIELDS}))
            rest = {f: cat[f][SHARD:] for f in FIELDS}
            train_buf, n_buf = ([rest] if len(rest["act"]) else []), len(rest["act"])
    if n_buf:
        shards.append(_write_shard(rdir, f"train_{len(shards):04d}", _concat(train_buf)))
    vrec = _write_shard(rdir, "val", _concat(val_buf))
    profile = _row_profile(rdir, shards, vrec)
    ids = {(m["source_sha256"], m["live_reader_sha256"], m["layout_hash"]) for m in metas}
    if len(ids) != 1:
        raise RowsetError(f"the parts were extracted under {len(ids)} code identities: {sorted(ids)}")
    src_sha, reader_sha, layout = ids.pop()
    commit = sorted({m["commit"] for m in metas})
    man = {"schema": SCHEMA, "rowset": rowset_id, "snapshot": scan_res["snapshot"],
           "manifest_sha256": scan_res["manifest_sha256"], "scan_keep_sha256": scan_res["keep_sha256"],
           "filter_spec": {"train_last": TRAIN_LAST, "rated": True, "result": True, "min_turns": MIN_TURNS,
                           "duplicates": "first body kept", "bots": "the card's rule (textpass.bot_flags)",
                           "rating_min_actor": RATING_MIN, "panel_exclusion": f">= {T.NEAR} move-revealed fits, "
                           "either side, every held-out panel, before C-GAME", "legal_actions": ">= 2",
                           "labels": "own action, or a hidden move's set (cant / confusion / faint, no foe Pursuit)",
                           "consistent": "legal AND the log showed the move / the mon",
                           "quitter_tail_dropped": QUIT_DROP, "split": f"key64(salted account id) mod {VAL_MOD} "
                           "== 0 validates"},
           "panel_exclusion_set": scan_res["panels"], "scan_counts": scan_res["counts"],
           "panel_exclusion_profile": scan_res["panel_exclusion_profile"],
           "extract_counts": dict(sorted(counts.items())), "c_game_commit": commit, "extraction_commit": commit,
           "source_sha256": src_sha, "row_sources": list(ROW_SOURCES),
           "live_reader_sha256": reader_sha, "layout_hash": layout, "parts": metas, "train_shards": shards,
           "val": vrec, "train_rows": int(sum(s["rows"] for s in shards)), "val_rows": vrec["rows"],
           "train_hidden": int(sum(s["hidden"] for s in shards)), "row_profile": profile}
    (rdir / "rows_manifest.json").write_text(json.dumps(man, indent=1))
    return man


def _row_profile(rdir: Path, shards: List[Dict[str, Any]], vrec: Dict[str, Any]) -> Dict[str, Any]:
    """Who and what the rows are (REVIEW D M7: the kept population, stated): accounts, the heaviest accounts' share,
    rating strata, endings, hidden and switch labels."""
    cols = ("acct", "rating", "end", "hidden", "act", "quit_dist")
    parts = [{f: np.load(rdir / f"{r['name']}.npz")[f] for f in cols} for r in shards]
    t = {f: np.concatenate([p[f] for p in parts]) for f in cols} if parts else {f: np.zeros(0) for f in cols}
    v = {f: np.load(rdir / f"{vrec['name']}.npz")[f] for f in cols}
    n = max(len(t["acct"]), 1)
    _, per = np.unique(t["acct"], return_counts=True)
    top = np.sort(per)[::-1]
    lab = t["act"] >= 0
    return {"train_accounts": int(len(per)), "val_accounts": int(len(np.unique(v["acct"]))),
            "accounts_in_both": int(len(np.intersect1d(np.unique(t["acct"]), np.unique(v["acct"])))),
            "top10_accounts_row_share": round(float(top[:10].sum() / n), 4),
            "top100_accounts_row_share": round(float(top[:100].sum() / n), 4),
            "rating_1300_1499_share": round(float(((t["rating"] >= 1300) & (t["rating"] < 1500)).mean()), 4)
            if len(t["rating"]) else None,
            "rating_ge1500_rows": int((t["rating"] >= 1500).sum()),
            "ending_share": {k: round(float((t["end"] == c).mean()), 4) for k, c in END_CODE.items()}
            if len(t["end"]) else {},
            "hidden_share": round(float(t["hidden"].mean()), 4) if len(t["hidden"]) else None,
            "switch_label_share": round(float((lab & (t["act"] <= 5)).sum() / max(lab.sum(), 1)), 4),
            "quitter_rows_share": round(float((t["quit_dist"] >= 0).mean()), 4) if len(t["quit_dist"]) else None}


def load_manifest(rows: Path) -> Tuple[Dict[str, Any], str]:
    p = Path(rows) / "rows_manifest.json"
    return json.loads(p.read_text()), S_sha(p)


def load_shard(rows: Path, rec: Dict[str, Any], fields: Sequence[str] = FIELDS) -> Dict[str, np.ndarray]:
    p = Path(rows) / f"{rec['name']}.npz"
    if S_sha(p) != rec["sha256"]:
        raise RowsetError(f"{p}: sha256 differs from the rows manifest")
    z = np.load(p)
    return {f: z[f] for f in fields}


def _guard_out(out: Path) -> None:
    from main.x50_r1.run import under_data
    from utils.git import get_main_repo_root

    if under_data(out) or (Path(get_main_repo_root()) / "models").resolve() in out.resolve().parents:
        raise RowsetError("refusing an output under a checkout's data/ or models/")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.human_data.rowset", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--snapshot", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--deadline", type=float, default=None, help="epoch seconds: start no part after it")
    e.add_argument("--limit", type=int, default=None, help="only the first N kept battles (a smoke)")
    c = sub.add_parser("compact")
    c.add_argument("--id", required=True)
    for p in (s, e, c):
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=3)
        p.add_argument("--store", default=None)
        p.add_argument("--registry", default=None)
        p.add_argument("--corpus", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    store = Path(a.store) if a.store else S.default_store()
    registry = Path(a.registry) if a.registry else S.default_registry()
    corpus = Path(a.corpus) if a.corpus else S.default_corpus()
    try:
        _guard_out(out)
        if a.cmd == "scan":
            res = scan(a.snapshot, out, a.workers, store, registry, corpus)
            print(json.dumps({k: res[k] for k in ("counts", "panel_exclusion_share")}, indent=1), flush=True)
            return 0
        if a.cmd == "extract":
            return extract(out, a.workers, a.deadline, store, registry, corpus, limit=a.limit)
        man = compact(out, a.id)
        print(json.dumps({k: man[k] for k in ("rowset", "train_rows", "val_rows", "train_hidden")}), flush=True)
        return 0
    except (S.SnapshotError, RowsetError) as exc:
        print(f"[rowset] REFUSED: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
