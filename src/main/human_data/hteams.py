"""H-TEAMS: the human LADDER TEAM corpus ``hr_spec_v1.r3`` (X60-T; ``designs/endstate/design_human_replay_lever.md``
§6.4 and §7, REVIEW F's F-M3 / F-M7 / m4 / m5).

    python -m main.human_data.hteams seats      [--root R] [--workers 3]
    python -m main.human_data.hteams complete   [--root R] [--workers 3] [--deadline <epoch s>]   # resumable parts
    python -m main.human_data.hteams fill-check [--root R] --parts 0,1                            # r1's C-GAME counts
    python -m main.human_data.hteams filter     [--root R]                                        # main's half
    python -m main.human_data.hteams pin-check  [--root R] --pin-tree <detached checkout of SIDE_PIN>
    python -m main.human_data.hteams finalize   [--root R]
    python -m main.human_data.hteams soak       [--root R] --where pin|main [--pin-tree T] [--rounds 10]
    python -m main.human_data.hteams stamp      [--root R]                                        # soaks -> manifest
    python -m main.human_data.hteams r72        [--root R]                                        # AFTER stamp (m5)

🚨 **Human data, never in git.** Every output lands under ``--root`` (default ``~/gen3ai_archive/human_teams/
hr_spec_v1.r3``), outside every checkout; only counts and sha256s may be committed. The corpus is NOT registered in
``utils.team_corpora.CORPORA`` and is never reachable through ``--team-mix`` (REVIEW F F-M7): the ``--human-data
teams:`` switch loads its directory by sha256. Its layout is the corpora's (``gen3_team_corpus_v1``: ``manifest.json``
with ``"human_data": true``, ``teams.jsonl.gz``, ``soak_selfcheck.json``), so :func:`utils.team_corpora.load_dir` reads
it with every guard (data sha256, row sha256, the AM-D1-FIX label guard, the tier sha256).

**The chain** ``hr_spec_v1`` -> r1 -> r2 -> r3 is verified against the snapshot registry before anything runs
(:func:`verify_chain`) and recorded in the manifest. The rules, in order; a seat or team leaves at the FIRST rule that
drops it and every rule is counted:

1. **seats** (:func:`select_seats`, then the log): every seat of r1's KEPT battles (``keep.jsonl.gz``: TRAIN dates,
   rated, a result, >= 11 turns, the first of each duplicate body, no bot, the actor >= 1300, and plan §3.1's
   exclusion on REVEALED facts, all applied by ``rowset scan``) with the actor rated >= :data:`RATING_MIN`, whose
   account is not on the Kakuna-family audit's drop list (r2's ``drop_list.json``), whose SIX mons the log showed. §3.1
   is re-applied on the log against the CURRENT held-out panels (both seats, before any fill), so a panel registered
   after r1 is honoured.
2. **completion**: C-GAME's team for the seat, by ``main.x50_r1.run.completed_seat`` with the battle key r1 used (the
   log's absolute path in the main corpus), so the fill IS the rowset's (``fill-check`` re-runs r1's own parts and
   compares C-GAME's counters). Exported to a paste by Showdown's ``Teams.export`` with the nickname dropped (no
   battle effect; human-authored text).
3. **Hidden Power repair** (``gen3_hp_repair_v1``, :data:`HP_SEED`); 4. **validator** (the pinned
   ``TeamValidator('gen3ou')`` through ``ladder_corpus_filter.js``); 5. **level** (a mon not at level 100 is DROPPED;
   the label guard re-checks the survivors and throws);
6. **the exact-set guard** (``heldout_panels.contamination``: >= 4 shared mon sets with a HELD-OUT panel team) and
   7. **REVIEW F m4's net**: >= 3 shared mon sets with a team of EVERY registered panel except one drawn from this
   corpus (:func:`net_panels`; m5: R72, registered with ``drawn_from_corpus``, never excludes its own source);
8. **main's engine and packer**: an id main's ``scan_move_probe`` cannot run, or a pack main's oracle refuses;
9. **at the SIDE PIN** (F-M3; ``pin-check``): the pin's ``--team-mix`` packer (``Gen3Teambuilder._install_mix``)
   packs every team, and main's pack oracle (``gen3_team_pack_oracle_v1``) must read each pack as the authored team
   (the pin mis-packs happiness-0 Frustration and Hidden Power type-line IVs): a team that differs is DROPPED; then
   every species / move / item / ability must RUN on the pin's ``scan_move_probe``: a team with an id the pin cannot
   run is DROPPED (the pin would mis-simulate or refuse it, a defect ON would meet and C0 would not);
10. **duplicate** (the same six mon sets, ``gen3_team_overlap_v1``, as a team already kept, in id order);
11. **the account cap**: <= :data:`ACCOUNT_CAP` teams per account, a seeded draw per account (:data:`CAP_SEED`).

The kept teams are ordered by a seeded shuffle of their sorted ids (:data:`ORDER_SEED`); one tier ``ALL``. The corpus
mons touching a 10-26 parity class (``parity_classes_1026.json``) are counted in the manifest. The soak
(``gen3_team_corpus_soak_v2``, every core label family) runs at BOTH pins: the side pin's env build (the pin's own
``utils.rust_env``, packer and labels, driving main's ``soak_texts``) writes ``soak_selfcheck.json``; main's writes
``soak_main.json``; ``stamp`` records both in the manifest.

**R72** (§6.3's positive control; m5's order: build -> draw -> register): 72 teams of the FINAL corpus, <= 1 per
account, seeded, under ``<root>/r72/``. This module never registers it; ``r72/manifest.json`` carries the proposed
registry entry.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from main.human_data import rowset as RS
from main.human_data import snapshots as S
from main.human_data import textpass as T

ROWSET_ID = "hr_spec_v1.r3"
SNAPSHOT = "hr_spec_v1"
PARENT = "hr_spec_v1.r2"
GRANDPARENT = "hr_spec_v1.r1"
#: C0's production pin (X60-T's side pin; REVIEW F F-M3).
SIDE_PIN = "f4b25238d"
RATING_MIN = 1500
SIX = 6
ACCOUNT_CAP = 5
HP_SEED = 20261106
CAP_SEED = 6004
ORDER_SEED = 6005
R72_N = 72
R72_SEED = 6072
NET_M4_MIN_SHARED = 3
PART = 200
TIER = "ALL"
SOAK_SEED = 6060
SCHEMA_WORK = "gen3_hteams_v1"
PARITY_REL = ("main", "human_data", "parity_classes_1026.json")


class HTeamsError(RuntimeError):
    """An input that does not match what it claims, or a step that would silently lose a team: stop."""


def default_root() -> Path:
    return Path.home() / "gen3ai_archive" / "human_teams" / ROWSET_ID


def default_x59s() -> Path:
    return Path.home() / "gen3ai_archive" / "x59s"


def team_id(rel: str, seat: str) -> str:
    """A team's id: a digest of its battle and seat (no account name enters it)."""
    return "hr3_" + hashlib.sha256(f"{rel}|{seat}".encode()).hexdigest()[:16]


def _write_json(p: Path, obj: Any) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, p)


def _write_jsonl_gz(p: Path, rows: Iterable[Dict[str, Any]]) -> None:
    tmp = p.with_name(p.name + ".tmp")
    with gzip.open(tmp, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(tmp, p)


def _read_jsonl_gz(p: Path) -> List[Dict[str, Any]]:
    with gzip.open(p, "rt") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def _sha(p: Path) -> str:
    return RS.S_sha(Path(p))


def _source_sha256() -> str:
    from utils.paths import src_path

    return hashlib.sha256(src_path("main", "human_data", "hteams.py").read_bytes()).hexdigest()


def _commit() -> str:
    from utils.git import get_git_hash

    return get_git_hash()


# ------------------------------------------------------------------------------------------------ the chain

def verify_chain(x59s: Path, registry: Path, store: Path) -> Dict[str, Any]:
    """The rowset chain ``hr_spec_v1`` -> r1 -> r2, each link checked against the registry; raises on any mismatch."""
    reg = json.loads(Path(registry).read_text())
    rows = reg["rowsets"]
    if ROWSET_ID in rows and rows[ROWSET_ID].get("builder", "").find("hteams") < 0:
        raise HTeamsError(f"{ROWSET_ID} is registered by another builder: not the next free id")
    snap = reg["snapshots"][SNAPSHOT]
    r1d, r2d = Path(x59s) / GRANDPARENT, Path(x59s) / PARENT
    r1m, r2m = r1d / "rows" / "rows_manifest.json", r2d / "rows" / "rows_manifest.json"
    got = {GRANDPARENT: _sha(r1m), PARENT: _sha(r2m)}
    for rid, sha in got.items():
        if sha != rows[rid]["rows_manifest_sha256"]:
            raise HTeamsError(f"{rid}: rows manifest sha256 {sha} != registry {rows[rid]['rows_manifest_sha256']}")
    r2 = json.loads(r2m.read_text())
    if r2.get("parent_rows_manifest_sha256") != got[GRANDPARENT] or r2.get("parent_rowset") != GRANDPARENT:
        raise HTeamsError(f"{PARENT}'s parent is not {GRANDPARENT}'s rows manifest")
    if r2.get("manifest_sha256") != snap["manifest_sha256"]:
        raise HTeamsError(f"{PARENT} cites snapshot manifest {r2.get('manifest_sha256')}, registry {snap['manifest_sha256']}")
    S.read_manifest(SNAPSHOT, store, registry)            # the snapshot's own manifest vs the registry (raises)
    keep = r1d / "keep.jsonl.gz"
    if _sha(keep) != r2["scan_keep_sha256"]:
        raise HTeamsError(f"{keep}: sha256 differs from {PARENT}'s scan_keep_sha256")
    drop = Path(x59s) / "kakuna_audit_v1" / "drop_list.json"
    want_drop = rows[PARENT]["derived"]["drop_list_sha256"]
    if _sha(drop) != want_drop:
        raise HTeamsError(f"{drop}: sha256 differs from {PARENT}'s registered drop list {want_drop}")
    return {"snapshot": SNAPSHOT, "snapshot_manifest_sha256": snap["manifest_sha256"],
            GRANDPARENT: got[GRANDPARENT], PARENT: got[PARENT], "keep_sha256": r2["scan_keep_sha256"],
            "drop_list_sha256": want_drop, "chain": [SNAPSHOT, GRANDPARENT, PARENT, ROWSET_ID]}


# ------------------------------------------------------------------------------------------------ 1. seats

def select_seats(keep: Sequence[Dict[str, Any]], drop: Set[int], counts: collections.Counter) -> List[Dict[str, Any]]:
    """Rule 1's row part: the kept battles' seats at actor >= :data:`RATING_MIN`, not on the audit's drop list (keys
    are ``rowset.key64`` of the salted account id), grouped by battle (``seats`` lists the battle's candidate seats)."""
    out = []
    for k in keep:
        seats = []
        for s in k["seats"]:
            counts["seat:kept_in_r2_battle"] += 1
            if int(k["rating"].get(s) or 0) < RATING_MIN:
                counts["drop:rating_lt_1500"] += 1
                continue
            if RS.key64(k["acct"][s]) in drop:
                counts["drop:flagged_account"] += 1
                continue
            seats.append(s)
        if seats:
            out.append({"rel": k["rel"], "sha": k["sha"], "n": k["n"], "seats": seats,
                        "acct": {s: k["acct"][s] for s in seats}, "rating": {s: k["rating"][s] for s in seats}})
    return out


def judge_battle(rec: Dict[str, Any], seats: Sequence[str], fits: Dict[str, Dict[str, int]]) -> Dict[str, str]:
    """Per candidate seat, ``"ok"`` or the rule that drops it: ``panel_near_revealed`` (plan §3.1 on the log: either
    seat's revealed facts fit >= ``textpass.NEAR`` mons of one held-out panel team: the whole battle leaves) or
    ``not_six_seen`` (the log showed fewer than six of the seat's mons)."""
    near = any(n >= T.NEAR for f in fits.values() for n in f.values())
    return {s: ("panel_near_revealed" if near else ("ok" if len(rec["reveals"].get(s, {})) == SIX else "not_six_seen"))
            for s in seats}


_MATCHER: Optional[T.PanelMatcher] = None


def _seat_job(args) -> Dict[str, Any]:
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = T.PanelMatcher()
    snap, population, corpus, b = args
    raw = S.read_verified(snap, population, Path(corpus), (b["rel"], b["sha"], b["n"]))
    rec = T.record(raw.decode(errors="replace"))
    if rec is None:
        raise HTeamsError(f"{b['rel']}: a kept battle has no players")
    fits = {s: _MATCHER.side_max_fits(m) for s, m in rec["reveals"].items()}
    return {"b": b, "verdict": judge_battle(rec, b["seats"], fits)}


def run_seats(root: Path, x59s: Path, workers: int, store: Path, registry: Path, corpus: Path) -> Dict[str, Any]:
    chain = verify_chain(x59s, registry, store)
    keep = _read_jsonl_gz(Path(x59s) / GRANDPARENT / "keep.jsonl.gz")
    drop = {int(x) for x in json.loads((Path(x59s) / "kakuna_audit_v1" / "drop_list.json").read_text())["drop_accounts"]}
    counts: collections.Counter = collections.Counter({"battles:kept_in_r2": len(keep)})
    cand = select_seats(keep, drop, counts)
    population = S.registry_row(SNAPSHOT, registry)["population"]
    seats: List[Dict[str, Any]] = []
    with Pool(workers) as pool:
        for r in pool.imap(_seat_job, [(SNAPSHOT, population, str(corpus), b) for b in cand], chunksize=32):
            b = r["b"]
            for s, v in sorted(r["verdict"].items()):
                if v != "ok":
                    counts[f"drop:{v}"] += 1
                    continue
                seats.append({"id": team_id(b["rel"], s), "rel": b["rel"], "sha": b["sha"], "n": b["n"], "seat": s,
                              "acct": b["acct"][s], "rating": int(b["rating"][s])})
    counts["seats"] = len(seats)
    work = Path(root) / "work"
    work.mkdir(parents=True, exist_ok=True)
    _write_jsonl_gz(work / "seats.jsonl.gz", seats)
    res = {"schema": SCHEMA_WORK + ":seats", "chain": chain, "rating_min": RATING_MIN, "panels_revealed_rule":
           _held_out(), "near": T.NEAR,
           "counts": dict(sorted(counts.items())), "seats_sha256": _sha(work / "seats.jsonl.gz"),
           "commit": _commit(), "source_sha256": _source_sha256()}
    _write_json(work / "seats.json", res)
    return res


def _held_out() -> List[str]:
    from main import heldout_panels as HP

    return sorted(HP.panel_names(held_out_only=True))


# ------------------------------------------------------------------------------------------------ 2. completion

def _complete_seat_rows(lines, info, battle_id, k_seats, counts, sink_all) -> None:
    from main.live.reader import ReaderRefusal
    from main.x50_r1 import extract as E
    from main.x50_r1 import run as R
    from main.x50_r1.points import PointError

    for s in k_seats:
        seat = s["seat"]
        name = info.get(seat, {}).get("name") or seat
        sink: Dict[str, Any] = {}
        try:
            cr = R.completed_seat(R._reader(), lines, seat, name, battle_id, counts, tag="c_", sink=sink)
        except (ReaderRefusal, PointError, E.LabelError, E.GuardError, RuntimeError) as exc:
            # the rowset's own refusal classes (``rowset._extract_job``): the seat is out and counted
            counts[f"seat_refused:{type(exc).__name__}"] += 1
            continue
        if cr is None:
            counts["seat_excluded:c_game"] += 1
            continue
        counts["seats_completed"] += 1
        if sink_all is not None:
            sink_all.append({**s, "packed": sink["team"], "attempt": int(sink["attempt"])})


def _complete_job(args) -> Dict[str, Any]:
    from main.live.replay_scan import battle_lines
    from main.x50_r1 import extract as E

    snap, population, corpus, group, keep_teams = args
    counts: collections.Counter = collections.Counter()
    b = S.read_verified(snap, population, Path(corpus), (group["rel"], group["sha"], group["n"]))
    lines = battle_lines(b.decode("utf-8", errors="replace"))
    teams: Optional[List[Dict[str, Any]]] = [] if keep_teams else None
    # the battle key r1's extraction used (``rowset._extract_job``): the fill is seeded by it
    _complete_seat_rows(lines, E.player_info(lines), str(Path(corpus) / group["rel"]), group["seats"], counts, teams)
    return {"counts": counts, "teams": teams or []}


def _groups(seats: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Seats grouped by battle, in first-seen order (both seats of a battle are read in one job)."""
    by: Dict[str, Dict[str, Any]] = {}
    for s in seats:
        g = by.setdefault(s["rel"], {"rel": s["rel"], "sha": s["sha"], "n": s["n"], "seats": []})
        g["seats"].append(s)
    return list(by.values())


def run_complete(root: Path, workers: int, deadline: Optional[float], registry: Path, corpus: Path) -> int:
    """Rule 2 in resumable parts of :data:`PART` battles. 0 = done, 3 = deadline."""
    work = Path(root) / "work"
    meta = json.loads((work / "seats.json").read_text())
    if _sha(work / "seats.jsonl.gz") != meta["seats_sha256"]:
        raise HTeamsError("seats.jsonl.gz differs from its record")
    groups = _groups(_read_jsonl_gz(work / "seats.jsonl.gz"))
    population = S.registry_row(SNAPSHOT, registry)["population"]
    pdir = work / "parts"
    pdir.mkdir(exist_ok=True)
    chunks = [groups[i:i + PART] for i in range(0, len(groups), PART)]
    ident = {"commit": _commit(), "source_sha256": _source_sha256(), "row_source_sha256": RS.source_sha256(),
             "live_reader_sha256": _live_reader_sha()}
    t0 = time.time()
    with Pool(workers) as pool:
        for ci, chunk in enumerate(chunks):
            out = pdir / f"part_{ci:04d}.json.gz"
            if out.exists():
                continue
            if deadline is not None and time.time() >= deadline:
                print(f"[hteams] deadline reached before part {ci}", flush=True)
                return 3
            res = pool.map(_complete_job, [(SNAPSHOT, population, str(corpus), g, True) for g in chunk], chunksize=4)
            counts: collections.Counter = collections.Counter()
            teams: List[Dict[str, Any]] = []
            for r in res:
                counts.update(r["counts"])
                teams.extend(r["teams"])
            body = {"part": ci, "battles": [g["rel"] for g in chunk], "counts": dict(sorted(counts.items())),
                    "teams": teams, **ident}
            tmp = out.with_name(out.name + ".tmp")
            with gzip.open(tmp, "wt") as fh:
                json.dump(body, fh, sort_keys=True)
            os.replace(tmp, out)
            print(f"[hteams] part {ci}/{len(chunks)}: {len(teams)} teams ({time.time() - t0:.0f} s)", flush=True)
    return 0


def _live_reader_sha() -> str:
    from utils.bridge.sim_bridge_bin import resolve_live_reader_bin

    return _sha(Path(resolve_live_reader_bin()))


def load_parts(root: Path) -> Tuple[List[Dict[str, Any]], collections.Counter, List[Dict[str, Any]]]:
    """Every completed team, the summed counters and each part's identity; refuses a missing part or mixed code."""
    work = Path(root) / "work"
    n_groups = len(_groups(_read_jsonl_gz(work / "seats.jsonl.gz")))
    need = -(-n_groups // PART)
    parts = sorted((work / "parts").glob("part_*.json.gz"))
    if len(parts) != need:
        raise HTeamsError(f"{len(parts)} completion parts, {need} needed: run `complete` to the end")
    teams: List[Dict[str, Any]] = []
    counts: collections.Counter = collections.Counter()
    idents = []
    for p in parts:
        with gzip.open(p, "rt") as fh:
            body = json.load(fh)
        teams.extend(body["teams"])
        counts.update(body["counts"])
        idents.append({k: body[k] for k in ("commit", "source_sha256", "row_source_sha256", "live_reader_sha256")})
    if len({json.dumps(i, sort_keys=True) for i in idents}) != 1:
        raise HTeamsError(f"completion parts were made by different code: {sorted({json.dumps(i) for i in idents})}")
    return teams, counts, idents


#: C-GAME's counters ``fill-check`` compares with r1's parts (the fill itself and its refusals; row rules excluded).
FILL_KEYS = ("c_", "seat_refused:", "seat_excluded:c_game")


def fill_counts(counts: Dict[str, int]) -> Dict[str, int]:
    return {k: int(v) for k, v in sorted(counts.items()) if k.startswith(FILL_KEYS) and int(v)}


def run_fill_check(root: Path, x59s: Path, parts: Sequence[int], workers: int, registry: Path,
                   corpus: Path) -> Dict[str, Any]:
    """Re-run C-GAME on EVERY seat of r1's extraction parts ``parts`` and compare C-GAME's counters with the part's
    record: equal counters (attempt histogram, redraw causes, exclusions, refusals, guard checks) are the evidence that
    the fill this build exports is the one the rowset used (same code, same seeds, a rebuilt reader)."""
    keep = {k["rel"]: k for k in _read_jsonl_gz(Path(x59s) / GRANDPARENT / "keep.jsonl.gz")}
    population = S.registry_row(SNAPSHOT, registry)["population"]
    out: Dict[str, Any] = {"schema": SCHEMA_WORK + ":fill_check", "parts": {}}
    with Pool(workers) as pool:
        for pi in parts:
            rec = json.loads((Path(x59s) / GRANDPARENT / "parts" / f"part_{pi:04d}.json").read_text())
            groups = [{"rel": rel, "sha": keep[rel]["sha"], "n": keep[rel]["n"],
                       "seats": [{"seat": s} for s in keep[rel]["seats"]]} for rel in rec["battles"]]
            cnt: collections.Counter = collections.Counter()
            for r in pool.map(_complete_job, [(SNAPSHOT, population, str(corpus), g, False) for g in groups],
                              chunksize=4):
                cnt.update(r["counts"])
            want, got = fill_counts(rec["counts"]), fill_counts(cnt)
            diff = {k: [want.get(k, 0), got.get(k, 0)] for k in sorted(set(want) | set(got))
                    if want.get(k, 0) != got.get(k, 0)}
            out["parts"][str(pi)] = {"battles": len(groups), "r1_commit": rec.get("commit"),
                                     "r1_live_reader_sha256": rec.get("live_reader_sha256"), "match": not diff,
                                     "diff_r1_vs_now": diff, "counters_compared": len(set(want) | set(got))}
    out["match"] = all(p["match"] for p in out["parts"].values())
    out["live_reader_sha256_now"] = _live_reader_sha()
    _write_json(Path(root) / "work" / "fill_check.json", out)
    return out


# ------------------------------------------------------------------------------------------------ 3-8. main's half

#: Showdown's own unpack -> export, the nickname dropped (``set.name = ''`` exports the species alone).
_EXPORT_JS = ("const {Teams}=require(process.argv[1]);const ls=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "process.stdout.write(JSON.stringify(ls.map(p=>{const t=Teams.unpack(p);if(!t)return null;"
              "for(const s of t){s.name='';}return Teams.export(t);})));")


def export_pastes(packed: Sequence[str]) -> List[Optional[str]]:
    from utils.paths import repo_path

    if not packed:
        return []
    p = subprocess.run(["node", "-e", _EXPORT_JS, str(repo_path("deps", "pokemon-showdown"))],
                       input=json.dumps(list(packed)), capture_output=True, text=True, check=True)
    out = json.loads(p.stdout)
    if len(out) != len(packed):
        raise HTeamsError(f"export answered {len(out)} of {len(packed)} teams")
    return [None if t is None else str(t).strip() for t in out]


def net_panels() -> Tuple[List[str], List[str]]:
    """REVIEW F m4's panels: every registered panel EXCEPT one whose entry says ``drawn_from_corpus`` is this corpus
    (m5: R72 is drawn from it, so it must never exclude its own source); returns ``(panels, skipped)``."""
    from main import heldout_panels as HP

    reg = HP.load_registry()["panels"]
    skip = sorted(n for n, e in reg.items() if e.get("drawn_from_corpus") == ROWSET_ID)
    return [n for n in HP.panel_names() if n not in skip], skip


def apply_nets(items: Sequence[Tuple[str, str]]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Rules 6 and 7 over ``(id, text)``: ``{id: rule}`` for every dropped team, and drops per ``rule:panel``."""
    from main import heldout_panels as HP

    dropped: Dict[str, str] = {}
    per: collections.Counter = collections.Counter()
    for h in HP.contamination(items):
        per[f"exact_set_guard:{h.panel}"] += 1
        dropped.setdefault(h.label, "exact_set_guard")
    panels, _ = net_panels()
    for h in HP.hits([it for it in items if it[0] not in dropped], panels, min_shared=NET_M4_MIN_SHARED):
        per[f"m4_near3:{h.panel}"] += 1
        dropped.setdefault(h.label, "m4_near3")
    return dropped, dict(sorted(per.items()))


def main_packs(texts: Sequence[str]) -> List[str]:
    """Main's training packer, the ``--team-mix`` path (``Gen3Teambuilder``: parse, Hidden Power IV fix, join)."""
    from utils.gen3_utils import fix_gen3_hp_ivs
    from utils.teambuilder import Gen3Teambuilder

    tb = object.__new__(Gen3Teambuilder)
    return [tb.join_team(fix_gen3_hp_ivs(tb.parse_showdown_team(t))) for t in texts]


def run_filter(root: Path) -> Dict[str, Any]:
    from agents.gen3_data.hp_repair import REPAIR_VERSION, repair_team, team_seed
    from main import heldout_panels as HP
    from utils.bridge.team_validator import pack_identity_defects
    from utils.ladder_corpus.build import engine_verdicts
    from utils.team_corpora import check_label_encodable
    from utils.team_corpora.build_w_forum import level_defects, showdown_rows

    work = Path(root) / "work"
    teams, ccounts, idents = load_parts(root)
    seats_meta = json.loads((work / "seats.json").read_text())
    counts: collections.Counter = collections.Counter()
    rule: Dict[str, str] = {}
    pastes = export_pastes([t["packed"] for t in teams])
    cand: Dict[str, Dict[str, Any]] = {}
    hp_repaired_mons = 0
    for t, text in zip(teams, pastes):
        if text is None:
            rule[t["id"]] = "export"
            continue
        new, recs = repair_team(text, team_seed(HP_SEED, t["id"]))
        hp_repaired_mons += len(recs)
        cand[t["id"]] = {**{k: t[k] for k in ("id", "acct", "rating", "attempt")}, "text": new.strip()}
    sd = showdown_rows({i: c["text"] for i, c in cand.items()}) if cand else {}
    why_validator: collections.Counter = collections.Counter()
    for i in sorted(cand):
        r = sd[i]
        if r["verdict"] != "ok":
            rule[i] = "validator"
            why_validator[f"{r['verdict']}: {str(r.get('why', ''))[:120]}"] += 1
            continue
        if level_defects(i, cand[i]["text"], r["packed"]):
            rule[i] = "level"
            continue
        cand[i]["ids"] = {"species": r["species"], "moves": r["moves"], "items": r["items"],
                          "abilities": r["abilities"]}
    live = sorted(i for i in cand if i not in rule)
    check_label_encodable(live, [cand[i]["text"] for i in live], f"{ROWSET_ID} after rule level")
    net_drop, net_per = apply_nets([(i, cand[i]["text"]) for i in live])
    rule.update(net_drop)
    live = [i for i in live if i not in rule]
    verdicts = engine_verdicts([{"verdict": "ok", "species": cand[i]["ids"]["species"], "moves": cand[i]["ids"]["moves"],
                                 "items": cand[i]["ids"]["items"], "abilities": cand[i]["ids"]["abilities"]}
                                for i in live]) if live else {}
    main_unplayable = _unplayable(verdicts)
    for i in live:
        if _team_uses(cand[i]["ids"], main_unplayable):
            rule[i] = "engine_main"
    live = [i for i in live if i not in rule]
    texts = [cand[i]["text"] for i in live]
    defects = pack_identity_defects("gen3ou", texts, main_packs(texts)) if live else []
    for i, d in zip(live, defects):
        if d:
            rule[i] = "pack_main"
    live = [i for i in live if i not in rule]
    for r in rule.values():
        counts[f"drop:{r}"] += 1
    stages = _stages(seats_meta["counts"]["seats"], int(ccounts.get("seats_completed", 0)), rule,
                     ("export", "validator", "level"), ("exact_set_guard", "m4_near3"), ("engine_main", "pack_main"))
    _write_jsonl_gz(work / "candidates.jsonl.gz", [cand[i] for i in live])
    res = {"schema": SCHEMA_WORK + ":filter", "completion_counts": fill_counts(ccounts),
           "completion_ident": idents[0], "counts": dict(sorted(counts.items())), "stages": stages,
           "hp_repair": {"version": REPAIR_VERSION, "seed": HP_SEED, "mons_repaired": hp_repaired_mons},
           "validator_failures_by_reason": dict(why_validator.most_common()),
           "nets": {"exact_set_guard": {"panels": _held_out(), "min_shared": HP.NEAR_NEIGHBOUR_MIN_SHARED},
                    "m4": {"panels": net_panels()[0], "skipped_drawn_from_this_corpus": net_panels()[1],
                           "min_shared": NET_M4_MIN_SHARED}, "drops_by_rule_and_panel": net_per,
                    "registry_sha256": HP.registry_sha256(), "matcher": HP.MATCHER},
           "engine_main": {"unplayable": main_unplayable, "ids_asked": {k: len(v) for k, v in verdicts.items()}},
           "candidates": len(live), "candidates_sha256": _sha(work / "candidates.jsonl.gz"),
           "commit": _commit(), "source_sha256": _source_sha256()}
    _write_json(work / "filter.json", res)
    return res


def _stages(seats: int, completed: int, rule: Dict[str, str], *groups: Sequence[str]) -> List[List[Any]]:
    """The corpus count after each stage: ``[[stage, n], ...]``."""
    n = completed
    out: List[List[Any]] = [["seats", seats], ["completed", completed]]
    for g in groups:
        n -= sum(1 for r in rule.values() if r in g)
        out.append(["+".join(g), n])
    return out


#: ``scan_move_probe``'s kinds and the Showdown filter's id lists (``utils.ladder_corpus.build.KINDS``).
PROBE_KINDS = (("species", "species"), ("move", "moves"), ("item", "items"), ("ability", "abilities"))


def _unplayable(verdicts: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
    return {k: sorted(i for i, v in verdicts.get(k, {}).items() if v != "ran") for k, _ in PROBE_KINDS}


def _team_uses(ids: Dict[str, List[str]], unplayable: Dict[str, List[str]]) -> List[str]:
    return [f"{k}:{i}" for k, key in PROBE_KINDS for i in ids[key] if i in set(unplayable.get(k, ()))]


# ------------------------------------------------------------------------------------------------ 9. the side pin

#: Run with the PIN's ``src`` first on ``sys.path``: its ``--team-mix`` packer over stdin's pastes.
_PIN_PACK_PY = r"""
import json, sys
import utils.teambuilder as TB
texts = json.load(sys.stdin)
tb = object.__new__(TB.Gen3Teambuilder)
tb._install_mix(texts, 0.5)
dropped = {int(i): str(e)[:300] for i, e in tb.mix_dropped}
packs, k = [], 0
for i in range(len(texts)):
    if i in dropped:
        packs.append(None)
    else:
        packs.append(tb.mix_packed_teams[k]); k += 1
assert k == len(tb.mix_packed_teams)
print(json.dumps({"packs": packs, "dropped": dropped, "module": TB.__file__}))
"""

#: Run with the PIN's ``src``: the pin's own ``scan_move_probe`` binary path, built by the pin's own corpus builder
#: path (``utils.ladder_corpus.build.scan_probe_bin``: cargo into the pin's crate; the pin has no stamp resolver).
_PIN_PROBE_PY = r"""
from utils.ladder_corpus.build import scan_probe_bin
print(scan_probe_bin())
"""


def _pin_env(pin_tree: Path) -> Dict[str, str]:
    return dict(os.environ, PYTHONPATH=str(Path(pin_tree) / "src"), CUDA_VISIBLE_DEVICES="")


def pin_commit(pin_tree: Path) -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(pin_tree), capture_output=True, text=True,
                         check=True).stdout.strip()
    want = subprocess.run(["git", "rev-parse", SIDE_PIN], cwd=str(pin_tree), capture_output=True, text=True,
                          check=True).stdout.strip()
    if sha != want:
        raise HTeamsError(f"{pin_tree} is at {sha}, not the side pin {want}")
    return sha


def _pin_python(pin_tree: Path, code: str, stdin: str = "") -> str:
    p = subprocess.run([sys.executable, "-c", code], input=stdin, capture_output=True, text=True,
                       cwd=str(pin_tree), env=_pin_env(pin_tree))
    if p.returncode != 0:
        raise HTeamsError(f"the pin's code failed (exit {p.returncode}): {p.stderr.strip()[-1500:]}")
    return p.stdout


def pack_classes(defects: Sequence[str]) -> List[str]:
    """The pack-oracle fields a team's defects name (``mon k (Species) <field>: ...`` -> ``<field>``)."""
    import re

    out = set()
    for d in defects:
        m = re.match(r"mon \d+ \([^)]*\) (\w+):", d)
        out.add(m.group(1) if m else "other")
    return sorted(out)


def probe(binary: str, kind: str, ids: Sequence[str]) -> Dict[str, str]:
    """``scan_move_probe``'s verdict per id (``ran`` = playable); refuses a partial answer."""
    if not ids:
        return {}
    p = subprocess.run([binary], input="\n".join(ids) + "\n", capture_output=True, text=True,
                       env=dict(os.environ, PROBE_KIND=kind), check=True)
    got = {}
    for ln in p.stdout.splitlines():
        if ln.strip():
            d = json.loads(ln)
            got[d.get("id", d.get("move"))] = d["verdict"]
    if set(got) != set(ids):
        raise HTeamsError(f"{binary} answered {len(got)} of {len(ids)} {kind} ids")
    return got


def run_pin_check(root: Path, pin_tree: Path) -> Dict[str, Any]:
    """Rule 9 (F-M3): the pin's packer judged by main's pack oracle, then the pin's engine coverage."""
    from utils.bridge.team_validator import pack_identity_defects
    from utils.ladder_corpus.build import engine_verdicts

    work = Path(root) / "work"
    flt = json.loads((work / "filter.json").read_text())
    if _sha(work / "candidates.jsonl.gz") != flt["candidates_sha256"]:
        raise HTeamsError("candidates.jsonl.gz differs from the filter's record")
    cand = _read_jsonl_gz(work / "candidates.jsonl.gz")
    sha = pin_commit(pin_tree)
    texts = [c["text"] for c in cand]
    got = json.loads(_pin_python(pin_tree, _PIN_PACK_PY, json.dumps(texts)))
    if not str(Path(got["module"]).resolve()).startswith(str(Path(pin_tree).resolve())):
        raise HTeamsError(f"the pin's packer resolved to {got['module']}, outside {pin_tree}")
    rule: Dict[str, str] = {}
    cls_teams: collections.Counter = collections.Counter()
    idx = [k for k, p in enumerate(got["packs"]) if p is not None]
    for k in range(len(cand)):
        if got["packs"][k] is None:
            rule[cand[k]["id"]] = "pin_validator"
    defects = pack_identity_defects("gen3ou", [texts[k] for k in idx], [got["packs"][k] for k in idx]) if idx else []
    for k, d in zip(idx, defects):
        if d:
            rule[cand[k]["id"]] = "pin_pack_differs"
            for f in pack_classes(d):
                cls_teams[f] += 1
    n_pack = len(cand) - len(rule)
    live = [c for c in cand if c["id"] not in rule]
    pin_probe = _pin_python(pin_tree, _PIN_PROBE_PY).strip().splitlines()[-1]
    if not str(Path(pin_probe).resolve()).startswith(str(Path(pin_tree).resolve())):
        raise HTeamsError(f"the pin's scan_move_probe resolved to {pin_probe}, outside {pin_tree}")
    pin_v: Dict[str, Dict[str, str]] = {}
    for kind, key in PROBE_KINDS:
        pin_v[kind] = probe(pin_probe, kind, sorted({i for c in live for i in c["ids"][key]}))
    pin_unplayable = _unplayable(pin_v)
    main_v = engine_verdicts([{"verdict": "ok", **{key: c["ids"][key] for _, key in PROBE_KINDS}} for c in live]) \
        if live else {}
    uses: collections.Counter = collections.Counter()
    for c in live:
        bad = _team_uses(c["ids"], pin_unplayable)
        if bad:
            rule[c["id"]] = "engine_pin"
            uses.update(bad)
    res = {"schema": SCHEMA_WORK + ":pin_check", "pin": sha, "pin_tree": str(pin_tree),
           "candidates_in": len(cand),
           "pack_identity": {"oracle": "gen3_team_pack_oracle_v1 (main's pack_oracle.js)",
                             "packer": "the pin's Gen3Teambuilder._install_mix (the --team-mix path)",
                             "pin_validator_dropped": sum(1 for r in rule.values() if r == "pin_validator"),
                             "differs": sum(1 for r in rule.values() if r == "pin_pack_differs"),
                             "differs_share": round(sum(1 for r in rule.values() if r == "pin_pack_differs")
                                                    / max(len(cand), 1), 4),
                             "teams_by_field": dict(sorted(cls_teams.items())), "after": n_pack},
           "engine": {"pin_probe_sha256": _sha(Path(pin_probe)), "pin_unplayable_in_corpus": pin_unplayable,
                      "main_unplayable_in_corpus": _unplayable(main_v),
                      "teams_using_pin_unplayable": dict(uses.most_common()),
                      "dropped": sum(1 for r in rule.values() if r == "engine_pin"),
                      "decision": "DROP: an id the pin's engine cannot run would be refused or mis-simulated for ON "
                                  "and never met by C0 (F-M3's asymmetric input defect)"},
           "after": len(cand) - len(rule), "dropped": rule, "commit": _commit(), "source_sha256": _source_sha256()}
    _write_json(work / "pin_check.json", res)
    return res


# ------------------------------------------------------------------------------------------------ 10-11. finalize

def dedupe(items: Sequence[Tuple[str, str]]) -> Tuple[List[str], int]:
    """Rule 10 in id order: keep a team unless its six mon sets (``gen3_team_overlap_v1``) equal a kept team's."""
    from main.anchors.team_overlap import parse_team

    seen: Set[Any] = set()
    keep, dup = [], 0
    for i, text in sorted(items):
        k = parse_team(text, i).key
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        keep.append(i)
    return keep, dup


def cap_accounts(ids_by_acct: Dict[str, List[str]], cap: int = ACCOUNT_CAP, seed: int = CAP_SEED) -> List[str]:
    """Rule 11: per account, its ids sorted then shuffled by ``Random(f"{seed}|{account}")``; the first ``cap`` stay."""
    out = []
    for a in sorted(ids_by_acct):
        ids = sorted(ids_by_acct[a])
        random.Random(f"{seed}|{a}").shuffle(ids)
        out.extend(ids[:cap])
    return sorted(out)


def order(ids: Sequence[str], seed: int = ORDER_SEED) -> List[str]:
    out = sorted(ids)
    random.Random(seed).shuffle(out)
    return out


def parity_report(texts: Sequence[str]) -> Dict[str, Any]:
    """(c): corpus mons and teams touching each 10-26 parity class (``parity_classes_1026.json``)."""
    from utils.paths import src_path
    from utils.showdown_id import to_id_str

    spec = json.loads(src_path(*PARITY_REL).read_text())
    mons: collections.Counter = collections.Counter()
    teams: collections.Counter = collections.Counter()
    for t in texts:
        hit_t = set()
        for block in [b for b in t.replace("\r", "").split("\n\n") if b.strip()]:
            ids = _mon_ids(block, to_id_str)
            for name, c in spec["classes"].items():
                if mon_matches(ids, c):
                    mons[name] += 1
                    hit_t.add(name)
        teams.update(hit_t)
    return {"spec_sha256": hashlib.sha256(src_path(*PARITY_REL).read_bytes()).hexdigest(),
            "mons_by_class": {k: mons.get(k, 0) for k in sorted(spec["classes"])},
            "teams_by_class": {k: teams.get(k, 0) for k in sorted(spec["classes"])},
            "teams_touching_any": sum(1 for t in texts if any(
                mon_matches(_mon_ids(b, to_id_str), c) for b in t.split("\n\n") if b.strip()
                for c in spec["classes"].values())),
            "mons_total": sum(1 for t in texts for b in t.split("\n\n") if b.strip())}


def _mon_ids(block: str, to_id) -> Set[Tuple[str, str]]:
    """``{(kind, id)}`` of one paste block: species, item, ability, moves (Hidden Power as ``hiddenpower``)."""
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    head = lines[0]
    out: Set[Tuple[str, str]] = set()
    name, _, item = head.partition(" @ ")
    name = name.strip()
    for g in (" (M)", " (F)"):
        if name.endswith(g):
            name = name[: -len(g)].rstrip()
    sp = name[name.rindex("(") + 1:-1] if name.endswith(")") and "(" in name else name
    out.add(("species", to_id(sp)))
    if item:
        out.add(("item", to_id(item)))
    for ln in lines[1:]:
        if ln.startswith("Ability:"):
            out.add(("ability", to_id(ln[len("Ability:"):])))
        elif ln.startswith("- "):
            mv = to_id(ln[2:].split("[")[0])
            out.add(("move", "hiddenpower" if mv.startswith("hiddenpower") else mv))
    return out


def mon_matches(ids: Set[Tuple[str, str]], cls: Dict[str, Any]) -> bool:
    """A mon touches a class if it carries ANY ``any_of`` id or ALL ids of ANY ``all_of_groups`` group."""
    if any(tuple(x) in ids for x in cls.get("any_of", ())):
        return True
    return any(g and all(tuple(x) in ids for x in g) for g in cls.get("all_of_groups", ()))


def run_finalize(root: Path) -> Dict[str, Any]:
    from main import heldout_panels as HP
    from utils.team_corpora import (DATA_NAME, MANIFEST_NAME, SCHEMA, check_label_encodable, encode_rows, load_dir,
                                    tier_sha256)

    root = Path(root)
    work = root / "work"
    flt = json.loads((work / "filter.json").read_text())
    pin = json.loads((work / "pin_check.json").read_text())
    seats_meta = json.loads((work / "seats.json").read_text())
    cand = {c["id"]: c for c in _read_jsonl_gz(work / "candidates.jsonl.gz")}
    if _sha(work / "candidates.jsonl.gz") != flt["candidates_sha256"]:
        raise HTeamsError("candidates.jsonl.gz differs from the filter's record")
    live = sorted(i for i in cand if i not in pin["dropped"])
    keep, dup = dedupe([(i, cand[i]["text"]) for i in live])
    by_acct: Dict[str, List[str]] = collections.defaultdict(list)
    for i in keep:
        by_acct[cand[i]["acct"]].append(i)
    final = order(cap_accounts(by_acct))
    capped = len(keep) - len(final)
    texts = [cand[i]["text"] for i in final]
    check_label_encodable(final, texts, ROWSET_ID)
    rows = [{"i": k, "id": i, "tier": TIER, "curator": cand[i]["acct"], "sha256": hashlib.sha256(t.encode()).hexdigest(),
             "text": t} for k, (i, t) in enumerate(zip(final, texts))]
    blob = encode_rows(rows)
    (root / DATA_NAME).write_bytes(blob)
    per_acct = collections.Counter(cand[i]["acct"] for i in final)
    ratings = sorted(int(cand[i]["rating"]) for i in final)
    stages = [*flt["stages"], ["pin_validator+pin_pack_differs", pin["pack_identity"]["after"]],
              ["engine_pin", pin["after"]], ["duplicate", len(keep)], ["account_cap (final)", len(final)]]
    man = {
        "schema": SCHEMA, "name": ROWSET_ID, "human_data": True, "data_file": DATA_NAME,
        "data_sha256": hashlib.sha256(blob).hexdigest(), "default_tier": TIER,
        "tiers": {TIER: {"n": len(final), "sha256": tier_sha256(texts)}},
        "what": "X60-T's H-TEAMS corpus: C-GAME-completed human LADDER teams (hr_spec_v1's spectator stream, the kept "
                "battles of hr_spec_v1.r2), actor >= 1500, six mons seen; TRAINING DATA, never a prior; LOCAL ONLY "
                "(human data never enters git); reached only by `--human-data teams:`, never by `--team-mix`",
        "design": "designs/endstate/design_human_replay_lever.md §6.4, §7; REVIEW_F_x60.md F-M3, F-M7, m4, m5",
        "built_by": "python -m main.human_data.hteams", "rowset": ROWSET_ID, "parent": PARENT,
        "chain": seats_meta["chain"], "stages": stages,
        "rules": ["seats: r2's kept battles (rowset scan's rules + the >= 11-turn floor), actor >= 1500, not on the "
                  "Kakuna-family audit's drop list, six mons seen, plan §3.1 re-applied on revealed facts",
                  "completion: C-GAME (main.x50_r1.run.completed_seat), r1's battle key; nickname dropped",
                  "hp_repair gen3_hp_repair_v1", "validator TeamValidator('gen3ou') (ladder_corpus_filter.js)",
                  "level: non-100 dropped; check_label_encodable on the survivors",
                  "exact_set_guard: >= 4 shared mon sets with a held-out panel team",
                  "m4_near3: >= 3 shared mon sets with a team of every registered panel but one drawn from this corpus",
                  "engine_main / pack_main: main's scan_move_probe and pack oracle",
                  "pin_validator / pin_pack_differs: the side pin's --team-mix packer judged by main's pack oracle",
                  "engine_pin: an id the side pin's scan_move_probe cannot run",
                  "duplicate: the same six mon sets as a kept team (id order)",
                  f"account_cap: <= {ACCOUNT_CAP} teams per account, Random(f'{CAP_SEED}|<account>')"],
        "seats": seats_meta["counts"], "completion": flt["completion_counts"],
        "completion_ident": flt["completion_ident"], "hp_repair": flt["hp_repair"], "filter": flt["counts"],
        "validator_failures_by_reason": flt["validator_failures_by_reason"], "nets": flt["nets"],
        "engine_main": flt["engine_main"], "duplicates_dropped": dup, "account_cap_dropped": capped,
        "side_pin": {"commit": pin["pin"], "pack_identity": pin["pack_identity"], "engine": pin["engine"]},
        "parity_classes_1026": parity_report(texts),
        "describe": {"accounts": len(per_acct), "teams_per_account_hist": dict(sorted(collections.Counter(
            per_acct.values()).items())), "rating_median": ratings[len(ratings) // 2] if ratings else None,
            "rating_min": ratings[0] if ratings else None, "rating_max": ratings[-1] if ratings else None},
        "seeds": {"hp": HP_SEED, "cap": CAP_SEED, "order": ORDER_SEED},
        "panel_registry": {"path": HP.REGISTRY_REL, "sha256": HP.registry_sha256()},
        "soak": {"side_pin": None, "main": None},
        "commit": _commit(), "source_sha256": _source_sha256(),
    }
    _write_json(root / MANIFEST_NAME, man)
    load_dir(root, TIER, ROWSET_ID, human_data_ok=True)  # the readers' every guard, on what was written
    return man


# ------------------------------------------------------------------------------------------------ the soaks

#: Run with the PIN's ``src`` on ``sys.path``: MAIN's ``soak_texts`` (schema v2, every label family) loaded by path,
#: so its ``utils.rust_env`` / ``Gen3Teambuilder`` / label families are the PIN's own.
_PIN_SOAK_PY = r"""
import importlib.util, json, sys
args = json.load(sys.stdin)
spec = importlib.util.spec_from_file_location("main_soak", args["soak_py"])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import utils.rust_env.ffi as F
res = m.soak_texts(args["ids"], args["texts"], args["ident"], rounds=args["rounds"], first_round=args["first_round"],
                   seed=args["seed"])
res["env_module"] = F.__file__
print(json.dumps(res))
"""


def soak_ident(root: Path) -> Tuple[List[str], List[str], Dict[str, Any]]:
    from utils.team_corpora import SCHEMA, load_dir

    c = load_dir(Path(root), TIER, ROWSET_ID, human_data_ok=True)
    ident = {"corpus": ROWSET_ID, "tier": TIER, "n": len(c), "tier_sha256": c.tier_sha256,
             "data_sha256": c.data_sha256, "schema": SCHEMA, "human_data": True}
    return list(c.ids), list(c.texts), ident


def run_soak(root: Path, where: str, pin_tree: Optional[Path], rounds: int, chunk_rounds: int) -> Dict[str, Any]:
    """The label-path crash soak at one pin, in resumable chunks of ``chunk_rounds`` rounds, merged into one record."""
    from utils.paths import src_path
    from utils.team_corpora import soak as SK

    root = Path(root)
    ids, texts, ident = soak_ident(root)
    cdir = root / "work" / f"soak_{where}"
    cdir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for r0 in range(0, rounds, chunk_rounds):
        k = min(chunk_rounds, rounds - r0)
        p = cdir / f"r{r0:03d}.json"
        if not p.exists():
            seed = SOAK_SEED + r0
            if where == "pin":
                assert pin_tree is not None
                commit = pin_commit(pin_tree)
                body = {"soak_py": str(src_path("utils", "team_corpora", "soak.py")), "ids": ids, "texts": texts,
                        "ident": ident, "rounds": k, "first_round": r0, "seed": seed}
                res = json.loads(_pin_python(pin_tree, _PIN_SOAK_PY, json.dumps(body)).strip().splitlines()[-1])
                if not str(Path(res.pop("env_module")).resolve()).startswith(str(Path(pin_tree).resolve())):
                    raise HTeamsError("the pin soak loaded an env core outside the pin tree")
                res["env_pin"] = commit
            else:
                res = SK.soak_texts(ids, texts, ident, rounds=k, first_round=r0, seed=seed)
                res["env_pin"] = _commit()
            _write_json(p, res)
            print(f"[hteams] soak {where} rounds {r0}..{r0 + k - 1}: {res['verdict']} {res['battles_finished']} "
                  f"battles {len(res['core_errors'])} core errors ({res['wall_s']} s)", flush=True)
        chunks.append(json.loads(p.read_text()))
    pins = {c.pop("env_pin") for c in chunks}
    if len(pins) != 1:
        raise HTeamsError(f"soak chunks of {where} ran at different commits: {sorted(pins)}")
    rec = SK.merge(chunks)
    rec["env_pin"] = pins.pop()
    rec["where"] = where
    _write_json(root / ("soak_selfcheck.json" if where == "pin" else "soak_main.json"), rec)
    return rec


def soak_summary(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: rec.get(k) for k in ("schema", "env_pin", "verdict", "rounds", "battles_planned", "battles_finished",
                                    "decisions_p1", "forfeits", "wall_s")} | {"core_errors": len(rec["core_errors"]),
                                                                              "labels": len(rec.get("labels") or [])}


def run_stamp(root: Path) -> Dict[str, Any]:
    """Records both pins' soaks in the manifest (F-M7 fix 3); refuses a soak of other bytes or one that failed."""
    from utils.team_corpora import MANIFEST_NAME, load_dir

    root = Path(root)
    man = json.loads((root / MANIFEST_NAME).read_text())
    out = {}
    for key, name in (("side_pin", "soak_selfcheck.json"), ("main", "soak_main.json")):
        rec = json.loads((root / name).read_text())
        sc = rec["corpus"]
        if sc.get("data_sha256") != man["data_sha256"] or sc.get("tier_sha256") != man["tiers"][TIER]["sha256"]:
            raise HTeamsError(f"{name} soaked other bytes than the manifest's")
        if rec["verdict"] != "PASS" or rec["core_errors"] or rec["battles_finished"] != rec["battles_planned"]:
            raise HTeamsError(f"{name}: {rec['verdict']}, {len(rec['core_errors'])} core errors")
        out[key] = {**soak_summary(rec), "file": name, "sha256": _sha(root / name)}
    if out["side_pin"]["env_pin"] != man["side_pin"]["commit"]:
        raise HTeamsError("the side-pin soak ran at another commit than the pin check")
    man["soak"] = out
    _write_json(root / MANIFEST_NAME, man)
    load_dir(root, TIER, ROWSET_ID, human_data_ok=True)
    return man


# ------------------------------------------------------------------------------------------------ R72

def draw_r72(rows: Sequence[Dict[str, Any]], n: int = R72_N, seed: int = R72_SEED) -> List[Dict[str, Any]]:
    """``n`` teams, <= 1 per account: the accounts sorted then shuffled by ``Random(seed)``, the first ``n`` taken,
    each giving one of its teams by ``Random(f"{seed}|{account}")`` over its sorted ids."""
    by: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by[str(r["curator"])].append(r)
    accts = sorted(by)
    if len(accts) < n:
        raise HTeamsError(f"{len(accts)} accounts, R72 needs {n}")
    random.Random(seed).shuffle(accts)
    out = []
    for a in accts[:n]:
        cands = sorted(by[a], key=lambda r: str(r["id"]))
        out.append(random.Random(f"{seed}|{a}").choice(cands))
    return sorted(out, key=lambda r: str(r["id"]))


def run_r72(root: Path) -> Dict[str, Any]:
    from main.anchors.session import team_set_digest
    from utils.team_corpora import MANIFEST_NAME, read_dir

    root = Path(root)
    man, rows, data_sha, man_sha = read_dir(root)
    if not man.get("soak", {}).get("side_pin"):
        raise HTeamsError("R72 is drawn from the FINAL corpus (m5: build -> draw -> register): run `stamp` first")
    pick = draw_r72(rows)
    d = root / "r72"
    tdir = d / "teams"
    tdir.mkdir(parents=True, exist_ok=True)
    for p in tdir.glob("*.txt"):
        p.unlink()
    for r in pick:
        (tdir / f"{r['id']}.txt").write_text(str(r["text"]) + "\n")
    digest = team_set_digest([str(r["text"]) for r in pick])
    if team_set_digest(tdir) != digest:
        raise HTeamsError("R72's written files do not digest as drawn")
    body = {"schema": SCHEMA_WORK + ":r72", "name": "r72", "n": len(pick), "content_digest": digest,
            "identity": "gen3_team_set_digest_v1", "ids": [r["id"] for r in pick], "seed": R72_SEED,
            "rule": f"<= 1 per account: accounts shuffled by Random({R72_SEED}), the first {R72_N}, one team each by "
                    f"Random(f'{R72_SEED}|<account>')",
            "source": {"corpus": ROWSET_ID, "data_sha256": data_sha, "manifest_sha256": man_sha},
            "proposed_registration": {
                "role": f"PRACTISED positive control (X60-T R72, plan §6.3): 72 teams of {ROWSET_ID}, trained on by "
                        "ON by design; never a held-out panel",
                "source": f"{ROWSET_ID} (human ladder teams; LOCAL ONLY, never in git)",
                "team_dir": str(tdir), "n": len(pick), "content_digest": digest, "held_out_from": [],
                "drawn_from_corpus": ROWSET_ID, "seed": R72_SEED}}
    _write_json(d / MANIFEST_NAME, body)
    return body


# ------------------------------------------------------------------------------------------------ CLI

def _guard_root(root: Path) -> None:
    from utils.paths import repo_root

    r = Path(root).resolve()
    for base in {Path(repo_root()).resolve(), Path(__file__).resolve().parents[3]}:
        if r == base or base in r.parents:
            raise HTeamsError(f"--root {r} is inside a checkout ({base}): human data never goes under git")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("seats", "complete", "fill-check", "filter", "pin-check", "finalize", "soak",
                                    "stamp", "r72"))
    ap.add_argument("--root", type=Path, default=default_root())
    ap.add_argument("--x59s", type=Path, default=default_x59s())
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--parts", default="0,1")
    ap.add_argument("--pin-tree", type=Path, default=None)
    ap.add_argument("--where", choices=("pin", "main"), default="pin")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--chunk-rounds", type=int, default=5)
    ap.add_argument("--store", type=Path, default=S.default_store())
    ap.add_argument("--registry", type=Path, default=S.default_registry())
    ap.add_argument("--corpus", type=Path, default=S.default_corpus())
    a = ap.parse_args(argv)
    _guard_root(a.root)
    Path(a.root).mkdir(parents=True, exist_ok=True)
    if a.cmd in ("pin-check",) or (a.cmd == "soak" and a.where == "pin"):
        if a.pin_tree is None:
            ap.error(f"{a.cmd} at the pin needs --pin-tree")
    if a.cmd == "seats":
        res: Any = run_seats(a.root, a.x59s, a.workers, a.store, a.registry, a.corpus)
        print(json.dumps(res["counts"], indent=1))
    elif a.cmd == "complete":
        return run_complete(a.root, a.workers, a.deadline, a.registry, a.corpus)
    elif a.cmd == "fill-check":
        res = run_fill_check(a.root, a.x59s, [int(x) for x in a.parts.split(",") if x], a.workers, a.registry,
                             a.corpus)
        print(json.dumps({"match": res["match"], **{k: v["diff_r1_vs_now"] for k, v in res["parts"].items()}}))
        return 0 if res["match"] else 1
    elif a.cmd == "filter":
        res = run_filter(a.root)
        print(json.dumps(res["stages"]))
    elif a.cmd == "pin-check":
        res = run_pin_check(a.root, a.pin_tree)
        print(json.dumps({k: res[k] for k in ("pin", "candidates_in", "after")} | {
            "pack_differs": res["pack_identity"]["differs"], "engine_dropped": res["engine"]["dropped"]}))
    elif a.cmd == "finalize":
        res = run_finalize(a.root)
        print(json.dumps({"stages": res["stages"], "data_sha256": res["data_sha256"]}))
    elif a.cmd == "soak":
        res = run_soak(a.root, a.where, a.pin_tree, a.rounds, a.chunk_rounds)
        print(json.dumps(soak_summary(res)))
        return 0 if res["verdict"] == "PASS" else 1
    elif a.cmd == "stamp":
        res = run_stamp(a.root)
        print(json.dumps(res["soak"]))
    elif a.cmd == "r72":
        res = run_r72(a.root)
        print(json.dumps({k: res[k] for k in ("n", "content_digest")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
