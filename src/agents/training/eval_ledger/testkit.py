"""Test helpers for the ledger's own tests and its producers' contract tests: a VALID v2 row built from a
pentanomial, a v1 row, a controllable clock. Never imported by production code."""
from __future__ import annotations

import copy
import datetime as _dt
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agents.training.eval_ledger import schema as S

SHA_A, SHA_B, SHA_C, SHA_D = "a" * 64, "b" * 64, "c" * 64, "d" * 64
T0 = _dt.datetime(2026, 10, 3, 12, 0, 0, tzinfo=_dt.timezone.utc)
PROTO = "gen3_eval_protocol_v1_h2h"


class Clock:
    """A clock the test moves by hand (no sleeps, no wall time)."""

    def __init__(self, t: _dt.datetime = T0):
        self.t = t

    def __call__(self) -> _dt.datetime:
        return self.t

    def advance(self, seconds: float) -> _dt.datetime:
        self.t = self.t + _dt.timedelta(seconds=seconds)
        return self.t

    def iso(self, offset_s: float = 0.0) -> str:
        return (self.t + _dt.timedelta(seconds=offset_s)).isoformat(timespec="seconds")


def regime(**over: Any) -> Dict[str, Any]:
    base = {"play": "greedy", "opponent_play": "greedy", "mirrored": True, "mirror_rule": "gen3_mirrored_pairs_v1",
            "eval_core": "rust", "turn_limit": 250, "seed_rule": "gen3_eval_game_seed_v1", "team_source": "pool",
            "protocol": PROTO, "seat_rule": "fixed_p1", "player_temp": None, "opponent_temp": None,
            "team_set": S.team_set_id({"t": "pool"})}
    base.update(over)
    return S.with_regime_id(base)


def player(sha: str, kind: str = "checkpoint") -> Dict[str, Any]:
    return {"id": f"run_{sha[0]}@100", "sha256": sha, "path": f"/x/{sha[0]}.zip", "run": f"run_{sha[0]}",
            "step": 100, "rung": "explicit_zip", "kind": kind}


def counts_of(pc: Sequence[int]) -> Dict[str, int]:
    """W / L / D of a pentanomial, reading 4 = WW, 3 = WD, 2 = WL, 1 = LD, 0 = LL."""
    return {"w": 2 * pc[4] + pc[3] + pc[2], "l": 2 * pc[0] + pc[1] + pc[2], "d": pc[3] + pc[1]}


def make_row(row_id: str = "w1:0", *, request: Optional[Mapping[str, Any]] = None, batch: int = 0,
             pc: Sequence[int] = (0, 0, 2, 0, 0), p: str = SHA_A, o: str = SHA_B, voided: int = 0,
             aborted: Optional[int] = None, purpose: str = "audit", reg: Optional[Dict[str, Any]] = None,
             cycle_seed: Optional[int] = None, ts: str = "2026-10-03T12:00:05+00:00",
             schedule_key: str = "h2h:k", near: Sequence[int] = ()) -> Dict[str, Any]:
    """A VALID v2 row. ``request`` is an ``open`` event (as ``LedgerWriter.open_request`` returns it) or ``None``
    (then the row is marked backfilled, the only legal request-less v2 row)."""
    c = counts_of(pc)
    games = c["w"] + c["l"] + c["d"]
    ab = voided if aborted is None else aborted
    t1, t2 = S.team_id("team one"), S.team_id("team two")
    teams = {t1: {"p": [games, c["w"]], "o": [0, 0]}, t2: {"p": [0, 0], "o": [games, c["l"]]}}
    outcomes = [(i, "W", 30) for i in range(games + 2 * voided)]
    row: Dict[str, Any] = {
        "schema": S.SCHEMA, "row_id": row_id, "supersedes": None, "ts": ts,
        "t_start": "2026-10-03T12:00:00+00:00", "t_end": "2026-10-03T12:00:04+00:00", "run": "study",
        "commit": "deadbeef", "player": player(p), "opponent": player(o), "regime": reg or regime(),
        "compute": {"device": "cpu", "outcome_digest": S.outcome_digest(outcomes, near),
                    "near_tie_games": sorted(near), "digest_margin": 2e-3},
        "purpose": purpose,
        "request": None if request is None else {"id": request["request_id"], "kind": request["kind"],
                                                 "family": request["family"], "opened": request["ts"],
                                                 "batch": int(batch)},
        "counts": {**c, "aborted": ab},
        "pairs": {"counts": list(pc), "n_pairs": sum(pc), "voided": voided},
        "teams": teams,
        "seed": {"rule": "gen3_eval_game_seed_v1", "schedule_seed": 0, "schedule_key": schedule_key,
                 "batch": int(batch), "cycle_seed": 7 + batch if cycle_seed is None else cycle_seed,
                 "item_key": "h2h", "game_lo": 0, "game_hi": games + 2 * voided - 1},
        "flags": [],
        "provenance": None if request is not None else {"source_file": "x.jsonl", "source_line": 1,
                                                        "source_sha256": "e" * 64, "backfill_id": "bf-test"},
    }
    return row


def make_v1_row(row_id: str = "20261003T120000Z-1:0", *, item_key: str = "h2h", play: str = "greedy",
                mirror_rule: str = "gen3_mirrored_pairs_v1", p_id: str = "run_a@100", o_id: str = "run_b@200",
                voided: int = 0) -> Dict[str, Any]:
    """A VALID v1 row (``gen3_eval_count_row_v1``), as ``main.h2h`` / the bot round robin wrote them."""
    reg = {"play": play, "opponent_play": play, "mirrored": True, "mirror_rule": mirror_rule, "eval_core": "rust",
           "turn_limit": 250, "seed_rule": "gen3_eval_game_seed_v1", "team_source": "pool"}
    reg = S.with_regime_id(reg)
    teams = {S.team_id("team one"): {"p": [2, 1], "o": [2, 1]}, S.team_id("team two"): {"p": [2, 1], "o": [2, 1]}}
    pl = {"id": p_id, "sha256": SHA_A, "path": "/x/a.zip", "run": "run_a", "step": 100, "rung": "explicit_zip"}
    op = {"id": o_id, "sha256": SHA_B, "path": "/x/b.zip", "run": "run_b", "step": 200, "rung": "explicit_zip"}
    return {"schema": S.SCHEMA_V1, "row_id": row_id, "supersedes": None, "ts": "2026-10-03T12:00:05+00:00",
            "t_start": "2026-10-03T12:00:00+00:00", "t_end": "2026-10-03T12:00:04+00:00", "run": "study",
            "commit": "deadbeef", "player": pl, "opponent": op, "regime": reg,
            "compute": {"device": "cpu", "near_tie_games": 3, "near_tie_games_wide": 5}, "purpose": "audit",
            "counts": {"w": 2, "l": 2, "d": 0}, "pairs": {"counts": [0, 0, 2, 0, 0], "n_pairs": 2, "voided": voided},
            "teams": copy.deepcopy(teams),
            "seed": {"rule": "gen3_eval_game_seed_v1", "schedule_seed": 0, "schedule_key": "h2h:k", "batch": 0,
                     "cycle_seed": 7, "item_key": item_key, "game_lo": 0, "game_hi": 3 + 2 * voided}}


def rows_of(objs: List[Dict[str, Any]]) -> List[str]:
    return [o["row_id"] for o in objs]
