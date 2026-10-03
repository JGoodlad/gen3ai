"""THE EVAL COUNT LEDGER — the row schema, the per-writer shard and the reader (``gen3_eval_count_row_v1``).

``designs/endstate/design_evaluation.md`` §0b (owner, 2026-10-02): ONE append-only ledger of COUNTS, not of
games — one row per (BATCH x MATCHUP), where a matchup is one player against one opponent under one regime
for one purpose. Counts are SUFFICIENT STATISTICS for every planned estimator (Bradley-Terry / Elo and
HodgeRank need the win counts per pair; Nash averaging the pair win-rate matrix; the GSPRT's verdict the
pentanomial counts; the power prior the counts by regime and purpose; per-team reads the team counters).

THIS MODULE IS THE FIRST §0b WRITER's schema (``main.h2h`` writes through it). What is built here, and what
is not:

* BUILT: the row schema, its validator (a malformed row is a typed :class:`LedgerSchemaError`, never
  written and never read past), one shard file per WRITER process (``ledger.<writer id>.jsonl`` — no
  interleaved appends), a reader that globs the shards and validates every row, and the REFUSAL of an
  output directory under ``models/``.
* NOT BUILT (design_evaluation.md §0b TODOs): the ARCHIVE-level ledger (main's ``models/_ledger/``), the
  compression of closed shards, the readers' declarations of which ``purpose`` / regime they consume, the
  backfill of ``eval_results.jsonl``. Until the archive ledger exists, a writer's output directory is the
  CALLER's (``refuse_under_models``): ``models/`` is read-only to a measurement, and a ledger that lived in
  a worktree's own tree would die with it.

A row, in words:

* ``schema`` · ``row_id`` (``<writer id>:<seq>``, the handle a correction points at) · ``supersedes``
  (``None``, or the ``row_id`` this row corrects — rows are NEVER edited);
* ``ts`` (when it was written) and the batch's time span ``t_start`` / ``t_end`` (UTC, ISO-8601);
* ``run`` (the study / run label that produced it) and ``commit`` (the code's git hash);
* ``player`` and ``opponent`` — each ``{id, sha256, path, run, step, rung}`` (a snapshot id and the
  checkpoint's content hash). The PLAYER is the MEASURED side: its ``counts`` and its ``score``;
* ``regime`` — how the games were played, and the thing a reader REFUSES to mix: ``play`` /
  ``opponent_play`` (``greedy`` or ``sampled``), ``mirrored`` (+ the rule), ``eval_core``, ``turn_limit``,
  the seed rule, the team source; ``regime_id`` is a digest of exactly these.
  ``compute`` (device, backend, ...) is recorded beside it and is NOT part of the identity;
* ``purpose`` — one of :data:`PURPOSES` (§0b's closed list);
* ``counts`` — ``w`` / ``l`` / ``d`` from the PLAYER's side (a draw is a tie or a turn-cap timeout, as in
  ``trace_result.classify_result``);
* ``pairs`` — for a MIRRORED row the **pentanomial** pair-outcome counts (half-points 0 .. 4 to the player;
  ``mirrored_pairs``), their ``n_pairs`` and the ``voided`` pairs left out of every count here;
  ``None`` for an unmirrored row;
* ``teams`` — a per-team COUNTER MAP ``{team id: {"p": [games, wins], "o": [games, wins]}}``: ``p`` counts
  the games the PLAYER piloted that team (and the player's wins with it), ``o`` those the OPPONENT did
  (and the opponent's wins with it). The team id is ``t:`` + the first 16 hex of the blake2b digest of the
  packed team string, so it names a team, not a table position;
* ``seed`` — what reproduces the games: the per-game seed rule, the schedule seed and key, the batch index,
  the cycle seed and the opponent key / game-index range it was played at.

Pure stdlib on purpose: every writer and reader (the eval core's host, the offline meters, a future
scheduler) imports it.
"""
from __future__ import annotations

import datetime as _dt
import glob
import gzip
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

SCHEMA = "gen3_eval_count_row_v1"

#: §0b's closed list of purposes. A new one is the OWNER's / the orchestrator's to add (design_evaluation.md).
PURPOSES = ("promotion", "plateau", "matrix", "audit", "anchor", "training", "cycle")

#: ``play`` / ``opponent_play`` values.
PLAYS = ("greedy", "sampled")

SHARD_PREFIX = "ledger."
SHARD_SUFFIX = ".jsonl"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TEAM_ID = re.compile(r"^t:[0-9a-f]{16}$")

_ROW_KEYS = ("schema", "row_id", "supersedes", "ts", "t_start", "t_end", "run", "commit", "player", "opponent",
             "regime", "compute", "purpose", "counts", "pairs", "teams", "seed")
_PLAYER_KEYS = ("id", "sha256", "path", "run", "step", "rung")
_REGIME_KEYS = ("play", "opponent_play", "mirrored", "mirror_rule", "eval_core", "turn_limit", "seed_rule",
                "team_source", "regime_id")
_SEED_KEYS = ("rule", "schedule_seed", "schedule_key", "batch", "cycle_seed", "item_key", "game_lo", "game_hi")


class LedgerSchemaError(ValueError):
    """A ledger row that does not satisfy the schema (the message lists every problem found)."""


class LedgerPathError(RuntimeError):
    """A ledger directory this writer may not use (under ``models/``)."""


def team_id(packed_team: str) -> str:
    """The stable id of a team: ``t:`` + 16 hex of the blake2b digest of its packed string."""
    return "t:" + hashlib.blake2b(packed_team.encode(), digest_size=8).hexdigest()


def regime_id(regime: Mapping[str, Any]) -> str:
    """The digest of a regime block's IDENTITY fields (everything but ``regime_id`` itself)."""
    ident = {k: regime[k] for k in sorted(regime) if k != "regime_id"}
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]


def with_regime_id(regime: Mapping[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in regime.items() if k != "regime_id"}
    out["regime_id"] = regime_id(out)
    return out


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _parse_ts(s: Any) -> Optional[_dt.datetime]:
    if not isinstance(s, str):
        return None
    try:
        return _dt.datetime.fromisoformat(s)
    except ValueError:
        return None


def _check_keys(where: str, got: Mapping[str, Any], want: tuple, problems: List[str]) -> bool:
    if not isinstance(got, Mapping):
        problems.append(f"{where}: not an object")
        return False
    missing, extra = [k for k in want if k not in got], [k for k in got if k not in want]
    if missing:
        problems.append(f"{where}: missing {missing}")
    if extra:
        problems.append(f"{where}: unknown keys {extra}")
    return not missing


def _check_player(where: str, p: Any, problems: List[str]) -> None:
    if not _check_keys(where, p, _PLAYER_KEYS, problems):
        return
    if not (isinstance(p["id"], str) and p["id"]):
        problems.append(f"{where}.id: not a non-empty string")
    if not (isinstance(p["sha256"], str) and _SHA256.match(p["sha256"])):
        problems.append(f"{where}.sha256: not 64 lowercase hex")
    for k in ("path", "run", "rung"):
        if not (isinstance(p[k], str) and p[k]):
            problems.append(f"{where}.{k}: not a non-empty string")
    if p["step"] is not None and not _is_int(p["step"]):
        problems.append(f"{where}.step: neither an int nor null")


def validate_row(row: Any) -> List[str]:
    """Every way ``row`` fails the §0b schema (EMPTY = valid). Checks the shapes, the closed vocabularies
    and the ARITHMETIC that must tie the blocks together (W + L + D = the games; a mirrored row's
    half-points = 2W + D; the team counters sum to the games and the wins)."""
    problems: List[str] = []
    if not _check_keys("row", row, _ROW_KEYS, problems):
        return problems
    if row["schema"] != SCHEMA:
        problems.append(f"schema: {row['schema']!r}, expected {SCHEMA!r}")
    if not (isinstance(row["row_id"], str) and ":" in row["row_id"]):
        problems.append("row_id: not '<writer id>:<seq>'")
    if row["supersedes"] is not None and not isinstance(row["supersedes"], str):
        problems.append("supersedes: neither a row_id nor null")
    ts = {k: _parse_ts(row[k]) for k in ("ts", "t_start", "t_end")}
    for k, v in ts.items():
        if v is None:
            problems.append(f"{k}: not an ISO-8601 timestamp")
    if all(ts.values()) and ts["t_start"] > ts["t_end"]:  # type: ignore[operator]
        problems.append("t_start is after t_end")
    for k in ("run", "commit"):
        if not (isinstance(row[k], str) and row[k]):
            problems.append(f"{k}: not a non-empty string")
    _check_player("player", row["player"], problems)
    _check_player("opponent", row["opponent"], problems)
    if row["purpose"] not in PURPOSES:
        problems.append(f"purpose: {row['purpose']!r} not in {PURPOSES}")

    reg = row["regime"]
    mirrored = False
    if _check_keys("regime", reg, _REGIME_KEYS, problems):
        for k in ("play", "opponent_play"):
            if reg[k] not in PLAYS:
                problems.append(f"regime.{k}: {reg[k]!r} not in {PLAYS}")
        if not isinstance(reg["mirrored"], bool):
            problems.append("regime.mirrored: not a bool")
        mirrored = reg["mirrored"] is True
        if not _is_int(reg["turn_limit"]) or reg["turn_limit"] < 1:
            problems.append("regime.turn_limit: not a positive int")
        for k in ("mirror_rule", "eval_core", "seed_rule", "team_source"):
            if not (isinstance(reg[k], str) and reg[k]):
                problems.append(f"regime.{k}: not a non-empty string")
        if reg["regime_id"] != regime_id(reg):
            problems.append("regime.regime_id: does not match the regime's identity fields")
    if not isinstance(row["compute"], Mapping):
        problems.append("compute: not an object")

    c = row["counts"]
    games = None
    if _check_keys("counts", c, ("w", "l", "d"), problems):
        if all(_is_int(c[k]) and c[k] >= 0 for k in ("w", "l", "d")):
            games = c["w"] + c["l"] + c["d"]
        else:
            problems.append("counts: w / l / d must be non-negative ints")

    pr = row["pairs"]
    if mirrored:
        if _check_keys("pairs", pr, ("counts", "n_pairs", "voided"), problems):
            pc = pr["counts"]
            ok = (isinstance(pc, list) and len(pc) == 5 and all(_is_int(x) and x >= 0 for x in pc)
                  and _is_int(pr["n_pairs"]) and _is_int(pr["voided"]) and pr["voided"] >= 0)
            if not ok:
                problems.append("pairs: counts must be 5 non-negative ints, n_pairs and voided ints")
            else:
                if pr["n_pairs"] != sum(pc):
                    problems.append(f"pairs.n_pairs {pr['n_pairs']} != sum of counts {sum(pc)}")
                if games is not None:
                    if games != 2 * sum(pc):
                        problems.append(f"counts: {games} games != 2 x {sum(pc)} pairs (a voided pair's games "
                                        "are counted nowhere on the row)")
                    half = sum(i * x for i, x in enumerate(pc))
                    if half != 2 * c["w"] + c["d"]:
                        problems.append(f"pairs: half-points {half} != 2W + D = {2 * c['w'] + c['d']}")
    elif pr is not None:
        problems.append("pairs: must be null on an unmirrored row")

    t = row["teams"]
    if not isinstance(t, Mapping):
        problems.append("teams: not an object")
    else:
        sums = {"pg": 0, "pw": 0, "og": 0, "ow": 0}
        for tid, v in t.items():
            if not _TEAM_ID.match(str(tid)):
                problems.append(f"teams: bad team id {tid!r}")
                continue
            if not (isinstance(v, Mapping) and set(v) == {"p", "o"}
                    and all(isinstance(v[s], list) and len(v[s]) == 2 and all(_is_int(x) and x >= 0 for x in v[s])
                            for s in ("p", "o"))):
                problems.append(f"teams[{tid}]: must be {{'p': [games, wins], 'o': [games, wins]}} of non-negative ints")
                continue
            if v["p"][1] > v["p"][0] or v["o"][1] > v["o"][0]:
                problems.append(f"teams[{tid}]: more wins than games")
            sums["pg"] += v["p"][0]
            sums["pw"] += v["p"][1]
            sums["og"] += v["o"][0]
            sums["ow"] += v["o"][1]
        if games is not None and not problems:
            if sums["pg"] != games or sums["og"] != games:
                problems.append(f"teams: the player's / opponent's team games {sums['pg']} / {sums['og']} != {games}")
            if sums["pw"] != c["w"] or sums["ow"] != c["l"]:
                problems.append(f"teams: wins {sums['pw']} / {sums['ow']} != the row's W {c['w']} / L {c['l']}")

    s = row["seed"]
    if _check_keys("seed", s, _SEED_KEYS, problems):
        if not (isinstance(s["rule"], str) and s["rule"]):
            problems.append("seed.rule: not a non-empty string")
        for k in ("schedule_seed", "batch", "cycle_seed", "game_lo", "game_hi"):
            if not _is_int(s[k]):
                problems.append(f"seed.{k}: not an int")
        for k in ("schedule_key", "item_key"):
            if not (isinstance(s[k], str) and s[k]):
                problems.append(f"seed.{k}: not a non-empty string")
        if _is_int(s["game_lo"]) and _is_int(s["game_hi"]):
            if s["game_hi"] < s["game_lo"]:
                problems.append("seed: game_hi < game_lo")
            elif games is not None and mirrored and pr and _is_int(pr.get("voided")) \
                    and s["game_hi"] - s["game_lo"] + 1 != games + 2 * pr["voided"]:
                problems.append("seed: the game-index range does not match the games played + voided")
    return problems


def check_row(row: Any, where: str = "row") -> None:
    problems = validate_row(row)
    if problems:
        raise LedgerSchemaError(f"{where}: {len(problems)} schema problem(s):\n  " + "\n  ".join(problems))


def refuse_under_models(path: "str | os.PathLike[str]") -> None:
    """REFUSE an output directory under ``models/`` (until the archive ledger exists, a measurement writes
    where its CALLER names, and ``models/`` is read-only to it): main's archive, ``$GEN3AI_MODELS_DIR``,
    and this checkout's own ``models/`` (a worktree's dies with it)."""
    from utils.paths import main_models_dir, repo_root

    rp = Path(path).resolve()
    roots = [repo_root() / "models"]
    md = main_models_dir()
    if md is not None:
        roots.append(md)
    env = os.environ.get("GEN3AI_MODELS_DIR")
    if env:
        roots.append(Path(env))
    for root in roots:
        r = root.resolve()
        if rp == r or r in rp.parents:
            raise LedgerPathError(f"REFUSED: {path} is under {root} — a count ledger is not written into the run "
                                  "archive until the archive ledger exists (design_evaluation.md §0b); name a "
                                  "directory outside models/")


class LedgerWriter:
    """ONE writer process's shard: ``<dir>/ledger.<writer id>.jsonl``, append-only, one row per line, flushed
    and fsynced per row (a batch is minutes of play: it must survive a kill). Every row is validated BEFORE
    it is written. ``writer id`` is unique per process (UTC time + pid), so two writers never interleave."""

    def __init__(self, out_dir: "str | os.PathLike[str]", writer_id: Optional[str] = None):
        refuse_under_models(out_dir)
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.writer_id = writer_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
        self.path = self.dir / f"{SHARD_PREFIX}{self.writer_id}{SHARD_SUFFIX}"
        self._seq = 0

    def next_row_id(self) -> str:
        return f"{self.writer_id}:{self._seq}"

    def append(self, row: Dict[str, Any]) -> str:
        """Validate and append ``row`` (its ``row_id`` must be :meth:`next_row_id`); returns the row id."""
        if row.get("row_id") != self.next_row_id():
            raise LedgerSchemaError(f"row_id {row.get('row_id')!r} is not this writer's next id {self.next_row_id()!r}")
        check_row(row, f"{self.path.name} row {self._seq}")
        line = json.dumps(row, sort_keys=True, separators=(",", ":"))
        with open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._seq += 1
        return str(row["row_id"])


def shard_paths(ledger_dir: "str | os.PathLike[str]") -> List[str]:
    d = str(ledger_dir)
    return sorted(glob.glob(os.path.join(d, f"{SHARD_PREFIX}*{SHARD_SUFFIX}"))
                  + glob.glob(os.path.join(d, f"{SHARD_PREFIX}*{SHARD_SUFFIX}.gz")))


def read_rows(ledger_dir: "str | os.PathLike[str]") -> List[Dict[str, Any]]:
    """Every row of every shard under ``ledger_dir`` (globbed), each validated — a malformed line is a typed
    :class:`LedgerSchemaError` naming the file and line, never skipped. Superseded rows are DROPPED (a
    correction row names the ``row_id`` it replaces); the order is the shards' names, then line order."""
    rows: List[Dict[str, Any]] = []
    for p in shard_paths(ledger_dir):
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt") as f:  # type: ignore[operator]
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as e:
                    raise LedgerSchemaError(f"{p}:{n}: not JSON ({e})") from None
                check_row(row, f"{os.path.basename(p)}:{n}")
                rows.append(row)
    dead = {r["supersedes"] for r in rows if r["supersedes"]}
    return [r for r in rows if r["row_id"] not in dead]
