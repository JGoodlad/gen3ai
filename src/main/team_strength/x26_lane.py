"""PILOT A's lane: ONE checkpoint pilots BOTH sides of every game, on the in-process Rust eval core (``main.h2h``).

One process holds one :class:`main.h2h.play.H2HEngine` (the player and the opponent are the same checkpoint, so the
engine declares ONE slot group) whose BOTH sides draw from a :class:`~main.team_strength.scheduled.ScheduledTeams`
over every rated and reference team. It works the plan's units in order (``main.team_strength.plan``), skipping a unit
whose file exists: a unit is played as ONE batch (``H2HEngine.play_batch``) and written ATOMICALLY (temp file, fsync,
rename), so an interrupted run resumes at the first missing unit and never re-plays or half-records one.

Every game is checked against the plan before its unit is written: game ``2k`` unswapped with the player on the
pair's RATED team and the opponent on its reference team, game ``2k+1`` swapped with the teams handed over
(``h2h.play.score_games`` asserts the mirror; the team ids are read back from the core's team table). Before every
unit: the T28 halt marker (``main.live.halt``) must be clear and the checkpoint's sha256 must be the registered one.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from main.team_strength import panel as PN
from main.team_strength import plan as PLN
from main.team_strength.scheduled import ScheduledTeams

UNIT_SCHEMA = "gen3_tsr_x26_unit_v1"
RESULTS = ("WIN", "LOSS", "DRAW")


class LaneError(RuntimeError):
    pass


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def check_halt() -> None:
    from main.live.halt import read_halt

    if read_halt() is not None:
        raise LaneError("a T28 live-parse HALT marker is set (python -m main.live.halt status): no play")


def x26_units(reg: Mapping[str, Any]) -> List[PLN.Unit]:
    a = reg["pilots"]["x26"]
    subjects = sorted({t for blk in reg["sets"].values() for t in blk["team_ids"].values()})
    refs = [m["team_id"] for m in reg["reference"]["members"]]
    return PLN.plan_units(subjects, refs, pairs_per_team=int(a["pairs_per_team"]), unit_pairs=int(a["unit_pairs"]),
                          seed=int(a["seed"]), key=PLN.plan_key(reg["digest"]))


def unit_rows(games: Sequence[Mapping[str, Any]], unit: PLN.Unit, team_packed: Sequence[str],
              id_by_packed: Mapping[str, str]) -> List[Dict[str, Any]]:
    """The unit's per-pair records, after checking every game against the plan (module docs)."""
    from main.h2h.play import score_games

    n = len(unit.pairs)
    score_games(games, team_packed, n)          # the mirror's invariants (raises H2HError)
    by = {int(g["game"]): g for g in games}
    rows = []
    for k, p in enumerate(unit.pairs):
        g0, g1 = by[2 * k], by[2 * k + 1]
        t0 = [id_by_packed.get(team_packed[int(i)]) for i in g0["teams"]]
        t1 = [id_by_packed.get(team_packed[int(i)]) for i in g1["teams"]]
        if t0 != [p.subject, p.ref] or t1 != [p.ref, p.subject]:
            raise LaneError(f"unit {unit.name} pair {k}: played {t0} / {t1}, planned {p.subject} vs {p.ref}")
        for g in (g0, g1):
            if g["result"] not in RESULTS:
                raise LaneError(f"unit {unit.name} game {g['game']}: result {g['result']!r}")
        rows.append({"k": k, "subject": p.subject, "ref": p.ref, "j": p.index,
                     "games": [{key: g.get(key) for key in ("game", "result", "swapped", "end_turn", "forfeit",
                                                             "near_ties", "near_ties_wide", "winner")}
                               for g in (g0, g1)]})
    return rows


def run(reg: Mapping[str, Any], state: Path, *, compute: Any, stop_at: Optional[float] = None,
        max_units: Optional[int] = None, emit: Callable[[str], None] = print) -> Dict[str, Any]:
    """Work the plan (module docs). ``stop_at`` (epoch seconds): start no unit after it."""
    from main.h2h.play import H2HEngine, file_sha256, resolve_player

    a = reg["pilots"]["x26"]
    sets = PN.load_registered(reg)
    texts = PN.text_of(sets)
    units = x26_units(reg)
    out_dir = Path(state) / "x26" / "units"
    todo = [u for u in units if not (out_dir / f"{u.name}.json").exists()]
    emit(f"[tsr x26] {len(units) - len(todo)} / {len(units)} units done; {len(todo)} to play")
    if not todo:
        return {"played": 0, "remaining": 0}
    check_halt()
    if file_sha256(a["model"]) != a["sha256"]:
        raise LaneError(f"{a['model']}: sha256 is not the registered {a['sha256'][:12]}")
    ref = resolve_player(a["model"])
    builder = ScheduledTeams(texts, team_source="gen3_tsr_x26: rated team vs reference team, mirrored, scheduled")
    eng = H2HEngine(ref, ref, compute, emit=emit, team_builder=builder)
    played = 0
    try:
        for u in todo:
            if stop_at is not None and time.time() > stop_at:
                emit(f"[tsr x26] stop-at reached; {len(todo) - played} units left")
                break
            if max_units is not None and played >= max_units:
                break
            check_halt()
            if file_sha256(a["model"]) != a["sha256"]:
                raise LaneError(f"{a['model']}: sha256 changed mid-run")
            t0 = time.time()
            builder.schedule(u.seed, [(p.subject, p.ref) for p in u.pairs])
            games, stats = eng.play_batch(len(u.pairs), u.seed)
            rows = unit_rows(games, u, eng.team_packed, builder.id_by_packed)
            rec = {"schema": UNIT_SCHEMA, "unit": u.name, "index": u.index, "seed": u.seed, "pairs": rows,
                   "n_pairs": len(rows), "model_sha256": a["sha256"], "registration_digest": reg["digest"],
                   "regime_id": eng.regime["regime_id"], "team_set_id": builder.team_set_id,
                   "compute": compute.block(), "core_stamp": eng.core_stamp, "torch": eng.torch_version,
                   "commit": _commit(), "near_ties": stats.get("near_ties"), "seconds": round(time.time() - t0, 1),
                   "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
            atomic_write_json(out_dir / f"{u.name}.json", rec)
            played += 1
            emit(f"[tsr x26] {u.name}: {len(rows)} pairs in {rec['seconds']} s "
                 f"({2 * len(rows) / max(rec['seconds'], 1e-9):.2f} games/s)")
    finally:
        eng.close()
    left = sum(1 for u in units if not (out_dir / f"{u.name}.json").exists())
    return {"played": played, "remaining": left}


def _commit() -> str:
    from utils.git import get_git_hash

    try:
        return get_git_hash() or ""
    except Exception:
        return ""


def load_games(reg: Mapping[str, Any], state: Path) -> List[Any]:
    """Every recorded pair as two :class:`~main.team_strength.fit.Game` rows (from the RATED team's side), cluster =
    the pair. Refuses a unit recorded under another registration or checkpoint."""
    from main.team_strength.fit import Game

    pts = {"WIN": 1.0, "DRAW": 0.5, "LOSS": 0.0}
    out = []
    for f in sorted((Path(state) / "x26" / "units").glob("x26_*.json")):
        rec = json.loads(f.read_text())
        if rec["registration_digest"] != reg["digest"] or rec["model_sha256"] != reg["pilots"]["x26"]["sha256"]:
            raise LaneError(f"{f}: recorded under another registration or checkpoint")
        for row in rec["pairs"]:
            cl = f"{rec['unit']}:{row['k']}"
            g0, g1 = row["games"]
            # game 0: the player (p1) pilots the rated team; game 1: the player pilots the reference team
            out.append(Game(a=row["subject"], b=row["ref"], s=pts[g0["result"]], owner=row["subject"], cluster=cl))
            out.append(Game(a=row["subject"], b=row["ref"], s=1.0 - pts[g1["result"]], owner=row["subject"],
                            cluster=cl))
    return out
