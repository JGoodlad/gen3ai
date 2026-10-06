"""THE LEDGER'S SCHEMAS — the count row (``gen3_eval_count_row_v2``), the v1 row it upgrades on read, the three
companion streams' records, the closed lists, and the KEYS that make a batch unique.

``designs/endstate/design_evaluation.md`` §0b is the spec; this module is its code. In one paragraph: ONE
append-only ledger of COUNTS, one row per (BATCH x MATCHUP), where a matchup is one player against one opponent
under one regime for one purpose. Counts are SUFFICIENT STATISTICS for every planned estimator (BT / HodgeRank
need the win counts per pair; Nash averaging the pair win-rate matrix; the GSPRT the pentanomial; per-team reads
the team counters). What is given up is per-game data; what is KEPT for audit is a per-batch OUTCOME DIGEST plus
the indices of the games that rested on a near-tie decision, so a replay is checked game for game (§9.2).

A v2 row, in words (v1's fields keep their meaning; §0b.2's table is the field-by-field why):

* ``schema`` · ``row_id`` (``<writer id>:<seq>``) · ``supersedes`` (``None`` or the corrected ``row_id`` — rows are
  NEVER edited) · ``ts`` / ``t_start`` / ``t_end`` (UTC ISO-8601) · ``run`` (the study label) · ``commit``;
* ``player`` / ``opponent`` — ``{id, sha256, path, run, step, rung, kind}``; ``kind`` is ``checkpoint`` / ``bot`` /
  ``external``. For a non-checkpoint the ``sha256`` digests the agent's identity and code. The PLAYER is the
  measured side;
* ``request`` — ``{id, kind, family, opened, batch}``: the request the batch was played FOR (the peeking rule made
  exact, §0c rule 1), the batch index inside it, and the FAMILY a group-sequential decision reads together.
  ``None`` only on a row that was not written by a v2 producer: an upgraded v1 row or a backfilled one;
* ``regime`` — how the games were played, and what readers refuse to mix. v2 adds ``protocol`` (a NAMED eval
  protocol, :data:`PROTOCOLS`), ``seat_rule``, the two temperatures and ``team_set`` (a DIGEST of the team set and
  builder parameters; the free-text ``team_source`` stays as a label and is NOT identity). ``regime_id`` digests
  the identity fields (:data:`REGIME_IDENTITY_V2`); an upgraded row keeps its old id as ``v1_id``;
* ``compute`` — free-form, never identity, EXCEPT three audit keys: ``outcome_digest`` (sha256 over the ordered
  per-game ``(index, W/L/D/A, turns)`` vector of the games WITHOUT a near-tie decision), ``near_tie_games`` (the
  indices that HAD one) and ``digest_margin`` (the margin that defined "near tie"; ``None`` when the agents have no
  float margin, e.g. scripted bots);
* ``purpose`` (:data:`PURPOSES`) · ``counts`` ``{w, l, d, aborted}`` from the player's side (``aborted`` = games
  started and not finished; the INCONCLUSIVE rule's denominator) · ``pairs`` (the pentanomial, mirrored rows
  only) · ``teams`` (per-team counters) · ``seed`` (what reproduces the games);
* ``flags`` — a SORTED subset of :data:`FLAGS`: what a legacy row LACKS (a reader declares which it accepts);
* ``provenance`` — ``{source_file, source_line, source_sha256, backfill_id}`` on a backfilled row, else ``None``.

Pure stdlib on purpose: every writer and reader imports it.
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "gen3_eval_count_row_v2"
SCHEMA_V1 = "gen3_eval_count_row_v1"
EVENT_SCHEMA = "gen3_eval_request_v1"
DECISION_SCHEMA = "gen3_eval_decision_v1"
REFERENCE_SCHEMA = "gen3_eval_reference_v1"
DIGEST_TAG = "gen3_eval_outcome_digest_v1"

# ------------------------------------------------------------------------------------------- closed lists
# Every list below EQUALS a table of design_evaluation.md §0b.2 — `src/eval_ledger_reader_gate_test.py` parses the
# doc and fails on any difference, in either direction. A new value is the owner's / the orchestrator's to add.

#: v1's purposes (the v1 validator's list; an upgraded row keeps its purpose).
PURPOSES_V1 = ("promotion", "plateau", "matrix", "audit", "anchor", "training", "cycle")
#: §0b.2 (owner Q1, provisional): v1's + ``ab``, ``ladder``, ``untaught``, ``gap``, ``monitor``. ``ladder`` is for
#: BACKFILLED snapshot-ladder rows ONLY (the validator requires ``provenance``); new pool rows are ``matrix``.
PURPOSES = PURPOSES_V1 + ("ab", "ladder", "untaught", "gap", "monitor")
#: A purpose that only a backfilled row may carry.
BACKFILL_ONLY_PURPOSES = ("ladder",)

REQUEST_KINDS = ("cycle", "sprt", "plateau_t1", "monitor_row", "monitor_topup", "panel", "matrix_dense",
                 "matrix_target", "matrix_probe", "ab_cell", "anchor_read", "untaught_read", "gap_read",
                 "audit_replay", "audit_dense", "adhoc")
#: A request kind whose rows DELIBERATELY replay a banked seed block (the replay audit, §9.2), so the seed-block
#: uniqueness (:func:`seed_key`) does not apply to them. Their purpose is ``audit``; estimates never pool it.
REPLAY_REQUEST_KINDS = ("audit_replay",)

PLAYER_KINDS = ("checkpoint", "bot", "external")
SEAT_RULES = ("fixed_p1", "balanced")
FLAGS = ("draws_folded", "teams_unrecorded", "seed_unrecorded", "sha_unrecorded", "eval_core_unrecorded",
         "digest_unrecorded")
#: The named eval protocols. Every SEMANTIC change to what a game measures bumps one; readers never pool across a
#: protocol; a pinned family freezes one (§0c rule 6). ``v1_<writer>`` names the protocol a v1 writer played — and
#: the two migrated writers KEEP playing it (U1 is storage-only; the h2h outcome-digest proof is in
#: ``designs/research_state/measurements/eval_ledger_u1_2026-10-03/``).
#: ``v1_inloop`` (eval U2) is the trainer's in-loop eval cycle and its SPRT promotion on the Rust eval core
#: (``agents.training.cycle_ledger``).
#: ``v1_h2h_oracle_one_sided`` / ``v1_h2h_oracle_both_sided`` are ``main.h2h``'s games with the PER-SIDE oracle reveal
#: (X5 A/B §7.7(a)): one-sided clairvoyance (only the oracle checkpoint's side is told the other team, at its own
#: recorded level) and both-sided (the other side is told the oracle's team too, at the same level). Same seats, teams,
#: seeds and turn limit as ``v1_h2h``; what differs is what an observation carries, so they never pool with it.
PROTOCOLS = ("gen3_eval_protocol_v1_h2h", "gen3_eval_protocol_v1_bot_rr", "gen3_eval_protocol_v1_inloop",
             "gen3_eval_protocol_v1_h2h_oracle_one_sided", "gen3_eval_protocol_v1_h2h_oracle_both_sided")
#: The protocol a v1 row is upgraded to, by its writer (§0b.2's upgrade rule).
V1_WRITER_PROTOCOL = {"h2h": "gen3_eval_protocol_v1_h2h", "bot_rr": "gen3_eval_protocol_v1_bot_rr"}

DECISION_KINDS = ("promotion", "eviction", "plateau", "ab_verdict", "cycle_flag")
#: Decision kinds registered as GROUP-SEQUENTIAL: only these may read a request FAMILY (§0b.7).
GROUP_SEQUENTIAL_KINDS = ("ab_verdict", "plateau")
#: Decision kinds that SELECT a node; ``selection="exclude"`` drops the rows they consumed (§0c rule 3).
SELECTING_DECISION_KINDS = ("promotion",)

#: The requests stream's events (§0b.4's claim protocol). ``family`` registers a request family (its decision
#: kind, rule and pinned protocol); ``row`` records, under the lock, that a claimed batch's row is on disk.
EVENT_KINDS = ("open", "family", "claim", "void", "row", "done", "cancel")
VOID_REASONS = ("pid_dead", "expired")

PLAYS = ("greedy", "sampled")
#: The sampled-temperature value of a scripted agent that draws from its own seeded streams.
BOT_NATIVE_TEMP = "bot_native"

# ------------------------------------------------------------------------------------------- field lists
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TEAM_ID = re.compile(r"^t:[0-9a-f]{16}$")
_TEAM_SET = re.compile(r"^ts:[0-9a-f]{16}$")

_ROW_KEYS_V1 = ("schema", "row_id", "supersedes", "ts", "t_start", "t_end", "run", "commit", "player", "opponent",
                "regime", "compute", "purpose", "counts", "pairs", "teams", "seed")
ROW_KEYS = _ROW_KEYS_V1 + ("request", "flags", "provenance")
_PLAYER_KEYS_V1 = ("id", "sha256", "path", "run", "step", "rung")
PLAYER_KEYS = _PLAYER_KEYS_V1 + ("kind",)
_REGIME_KEYS_V1 = ("play", "opponent_play", "mirrored", "mirror_rule", "eval_core", "turn_limit", "seed_rule",
                   "team_source", "regime_id")
#: The fields ``regime_id`` digests on a v2 row. ``team_source`` (a label), ``v1_id`` and ``regime_id`` are not.
REGIME_IDENTITY_V2 = ("play", "opponent_play", "mirrored", "mirror_rule", "eval_core", "turn_limit", "seed_rule",
                      "protocol", "seat_rule", "player_temp", "opponent_temp", "team_set")
REGIME_KEYS = REGIME_IDENTITY_V2 + ("team_source", "regime_id")
_REGIME_OPTIONAL = ("v1_id",)
_SEED_KEYS = ("rule", "schedule_seed", "schedule_key", "batch", "cycle_seed", "item_key", "game_lo", "game_hi")
REQUEST_KEYS = ("id", "kind", "family", "opened", "batch")
PROVENANCE_KEYS = ("source_file", "source_line", "source_sha256", "backfill_id")
COMPUTE_AUDIT_KEYS = ("outcome_digest", "near_tie_games", "digest_margin")


class LedgerSchemaError(ValueError):
    """A ledger record that does not satisfy its schema (the message lists every problem found)."""


# ------------------------------------------------------------------------------------------- helpers
def team_id(packed_team: str) -> str:
    """The stable id of a team: ``t:`` + 16 hex of the blake2b digest of its packed string."""
    return "t:" + hashlib.blake2b(packed_team.encode(), digest_size=8).hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def team_set_id(parts: Mapping[str, Any]) -> str:
    """The identity of a TEAM SET: ``ts:`` + 16 hex of sha256 over the canonical JSON of what defines the draws
    (the ordered team lists and the builder parameters). It changes when the TEAMS change, not when a label does."""
    return "ts:" + hashlib.sha256(canonical(dict(parts)).encode()).hexdigest()[:16]


def regime_id(regime: Mapping[str, Any]) -> str:
    """The digest of a regime block's IDENTITY fields: :data:`REGIME_IDENTITY_V2` on a v2 block (one that carries
    ``protocol``), every field but ``regime_id`` on a v1 block (v1's own rule, kept so a v1 row still validates)."""
    if "protocol" in regime:
        ident = {k: regime.get(k) for k in REGIME_IDENTITY_V2}
    else:
        ident = {k: regime[k] for k in sorted(regime) if k != "regime_id"}
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]


def with_regime_id(regime: Mapping[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in regime.items() if k != "regime_id"}
    out["regime_id"] = regime_id(out)
    return out


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def parse_ts(s: Any) -> Optional[_dt.datetime]:
    """An aware ISO-8601 timestamp, or ``None`` (a naive one is not a ledger timestamp)."""
    if isinstance(s, _dt.datetime):
        return s if s.tzinfo is not None else None
    if not isinstance(s, str):
        return None
    try:
        t = _dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def outcome_digest(games: Iterable[Tuple[int, str, Optional[int]]], near_tie: Iterable[int] = ()) -> str:
    """sha256 over the ORDERED per-game outcome vector ``(game index, W/L/D/A, turns)`` of every game NOT in
    ``near_tie`` (§0b.2, review item 16). ``A`` = aborted (started, not finished). A replay of the same seed block
    must reproduce it exactly; a game with a decision inside the near-tie margin may flip with the device or the
    batch shape, so it is listed by index instead of hashed (standing rule 8)."""
    skip = {int(i) for i in near_tie}
    seen = set()
    lines = [DIGEST_TAG]
    for idx, res, turns in sorted(games, key=lambda g: int(g[0])):
        i = int(idx)
        if i in seen:
            raise ValueError(f"game {i} appears twice in the outcome vector")
        seen.add(i)
        if res not in ("W", "L", "D", "A"):
            raise ValueError(f"game {i}: outcome {res!r} is not W / L / D / A")
        if i not in skip:
            lines.append(f"{i}:{res}:{'' if turns is None else int(turns)}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _sha_or_id(p: Mapping[str, Any]) -> str:
    return str(p["sha256"]) if p.get("sha256") else f"id:{p['id']}"


def batch_key(row: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
    """``(request.id, request.batch, player, opponent, regime_id)`` — UNIQUE across the ledger (§0b.4, review M4),
    enforced by the writer, every reader and ``audit``. ``None`` for a row without a request."""
    req = row.get("request")
    if not req:
        return None
    return ("batch", req["id"], int(req["batch"]), _sha_or_id(row["player"]), _sha_or_id(row["opponent"]),
            row["regime"]["regime_id"])


def seed_key(row: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
    """The SEED BLOCK a row's games were drawn from: ``(player, opponent, regime_id, seed rule, schedule seed,
    schedule key, item key, cycle seed)``. Two rows on one seed block are the SAME GAMES recorded twice, whatever
    their requests say, so it is unique too — except under a replay request (:data:`REPLAY_REQUEST_KINDS`), whose
    whole point is to replay one. The game range is deliberately NOT in the key: one cycle seed under two batch
    lengths overlaps. ``None`` for a row without a seed block, or a replay row."""
    s = row.get("seed")
    req = row.get("request")
    if not s or (req and req.get("kind") in REPLAY_REQUEST_KINDS):
        return None
    return ("seed", _sha_or_id(row["player"]), _sha_or_id(row["opponent"]), row["regime"]["regime_id"], s["rule"],
            int(s["schedule_seed"]), s["schedule_key"], s["item_key"], int(s["cycle_seed"]))


def key_list(k: Optional[Tuple[Any, ...]]) -> Optional[List[Any]]:
    return list(k) if k is not None else None


def content_sha(row: Mapping[str, Any]) -> str:
    """sha256 of a record's canonical JSON AS STORED — the unit a decision's consumed-rows digest is built from,
    so the digest does not move when the upgrade-on-read code does."""
    return hashlib.sha256(canonical(row).encode()).hexdigest()


def rows_digest(pairs: Iterable[Tuple[str, str]]) -> str:
    """The digest of a SET of rows: sha256 over the sorted ``(row_id, content sha)`` pairs."""
    body = "\n".join(f"{rid}:{sha}" for rid, sha in sorted(pairs))
    return hashlib.sha256(body.encode()).hexdigest()


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _nonempty(x: Any) -> bool:
    return isinstance(x, str) and bool(x)


def _check_keys(where: str, got: Any, want: Sequence[str], problems: List[str],
                optional: Sequence[str] = ()) -> bool:
    if not isinstance(got, Mapping):
        problems.append(f"{where}: not an object")
        return False
    missing = [k for k in want if k not in got]
    extra = [k for k in got if k not in want and k not in optional]
    if missing:
        problems.append(f"{where}: missing {missing}")
    if extra:
        problems.append(f"{where}: unknown keys {extra}")
    return not missing


def _check_ts(where: str, x: Any, problems: List[str]) -> Optional[_dt.datetime]:
    t = parse_ts(x)
    if t is None:
        problems.append(f"{where}: not an aware ISO-8601 timestamp")
    return t


# ------------------------------------------------------------------------------------------- the v1 row
def _check_player_v1(where: str, p: Any, problems: List[str]) -> None:
    if not _check_keys(where, p, _PLAYER_KEYS_V1, problems):
        return
    if not _nonempty(p["id"]):
        problems.append(f"{where}.id: not a non-empty string")
    if not (isinstance(p["sha256"], str) and _SHA256.match(p["sha256"])):
        problems.append(f"{where}.sha256: not 64 lowercase hex")
    for k in ("path", "run", "rung"):
        if not _nonempty(p[k]):
            problems.append(f"{where}.{k}: not a non-empty string")
    if p["step"] is not None and not _is_int(p["step"]):
        problems.append(f"{where}.step: neither an int nor null")


def _check_counts_block(row: Mapping[str, Any], mirrored: bool, aborted_ok: bool, problems: List[str]
                        ) -> Optional[int]:
    """The W / L / D (+ aborted) and pentanomial arithmetic shared by v1 and v2. Returns the games, or None."""
    c = row["counts"]
    want = ("w", "l", "d", "aborted") if aborted_ok else ("w", "l", "d")
    games = None
    if _check_keys("counts", c, want, problems):
        if all(_is_int(c[k]) and c[k] >= 0 for k in want):
            games = c["w"] + c["l"] + c["d"]
        else:
            problems.append(f"counts: {' / '.join(want)} must be non-negative ints")
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
                    if aborted_ok and _is_int(c.get("aborted")) and not (pr["voided"] <= c["aborted"]
                                                                         <= 2 * pr["voided"]):
                        problems.append(f"counts.aborted {c['aborted']} is not within [voided, 2 x voided] = "
                                        f"[{pr['voided']}, {2 * pr['voided']}] (a voided pair holds 1 or 2 aborted "
                                        "games)")
    elif pr is not None:
        problems.append("pairs: must be null on an unmirrored row")
    return games


def _check_teams(row: Mapping[str, Any], games: Optional[int], problems: List[str]) -> None:
    t, c = row["teams"], row["counts"]
    if not isinstance(t, Mapping):
        problems.append("teams: not an object")
        return
    n0 = len(problems)
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
    if games is not None and len(problems) == n0:
        if sums["pg"] != games or sums["og"] != games:
            problems.append(f"teams: the player's / opponent's team games {sums['pg']} / {sums['og']} != {games}")
        if sums["pw"] != c["w"] or sums["ow"] != c["l"]:
            problems.append(f"teams: wins {sums['pw']} / {sums['ow']} != the row's W {c['w']} / L {c['l']}")


def _check_seed(row: Mapping[str, Any], games: Optional[int], mirrored: bool, problems: List[str]) -> None:
    s = row["seed"]
    if not _check_keys("seed", s, _SEED_KEYS, problems):
        return
    if not _nonempty(s["rule"]):
        problems.append("seed.rule: not a non-empty string")
    for k in ("schedule_seed", "batch", "cycle_seed", "game_lo", "game_hi"):
        if not _is_int(s[k]):
            problems.append(f"seed.{k}: not an int")
    for k in ("schedule_key", "item_key"):
        if not _nonempty(s[k]):
            problems.append(f"seed.{k}: not a non-empty string")
    if _is_int(s["game_lo"]) and _is_int(s["game_hi"]):
        if s["game_hi"] < s["game_lo"]:
            problems.append("seed: game_hi < game_lo")
        elif games is not None:
            pr, c = row["pairs"], row["counts"]
            span = s["game_hi"] - s["game_lo"] + 1
            if mirrored and isinstance(pr, Mapping) and _is_int(pr.get("voided")):
                if span != games + 2 * pr["voided"]:
                    problems.append("seed: the game-index range does not match the games played + voided")
            elif not mirrored and _is_int(c.get("aborted")) and span != games + c["aborted"]:
                problems.append("seed: the game-index range does not match the games played + aborted")


def validate_row_v1(row: Any) -> List[str]:
    """Every way ``row`` fails ``gen3_eval_count_row_v1`` (EMPTY = valid) — v1's rules exactly, kept so a v1 row is
    validated AS WRITTEN before it is upgraded."""
    problems: List[str] = []
    if not _check_keys("row", row, _ROW_KEYS_V1, problems):
        return problems
    if row["schema"] != SCHEMA_V1:
        problems.append(f"schema: {row['schema']!r}, expected {SCHEMA_V1!r}")
    _check_common(row, problems)
    _check_player_v1("player", row["player"], problems)
    _check_player_v1("opponent", row["opponent"], problems)
    if row["purpose"] not in PURPOSES_V1:
        problems.append(f"purpose: {row['purpose']!r} not in {PURPOSES_V1}")
    reg = row["regime"]
    mirrored = False
    if _check_keys("regime", reg, _REGIME_KEYS_V1, problems):
        for k in ("play", "opponent_play"):
            if reg[k] not in PLAYS:
                problems.append(f"regime.{k}: {reg[k]!r} not in {PLAYS}")
        if not isinstance(reg["mirrored"], bool):
            problems.append("regime.mirrored: not a bool")
        mirrored = reg["mirrored"] is True
        if not _is_int(reg["turn_limit"]) or reg["turn_limit"] < 1:
            problems.append("regime.turn_limit: not a positive int")
        for k in ("mirror_rule", "eval_core", "seed_rule", "team_source"):
            if not _nonempty(reg[k]):
                problems.append(f"regime.{k}: not a non-empty string")
        if reg["regime_id"] != regime_id(reg):
            problems.append("regime.regime_id: does not match the regime's identity fields")
    if not isinstance(row["compute"], Mapping):
        problems.append("compute: not an object")
    games = _check_counts_block(row, mirrored, False, problems)
    _check_teams(row, games, problems)
    _check_seed(row, games, mirrored, problems)
    return problems


def _check_common(row: Mapping[str, Any], problems: List[str]) -> None:
    if not (isinstance(row["row_id"], str) and ":" in row["row_id"]):
        problems.append("row_id: not '<writer id>:<seq>'")
    if row["supersedes"] is not None and not isinstance(row["supersedes"], str):
        problems.append("supersedes: neither a row_id nor null")
    ts = {k: _check_ts(k, row[k], problems) for k in ("ts", "t_start", "t_end")}
    if ts["t_start"] and ts["t_end"] and ts["t_start"] > ts["t_end"]:
        problems.append("t_start is after t_end")
    for k in ("run", "commit"):
        if not _nonempty(row[k]):
            problems.append(f"{k}: not a non-empty string")


# ------------------------------------------------------------------------------------------- the v2 row
def _check_player(where: str, p: Any, flags: Sequence[str], problems: List[str]) -> None:
    if not _check_keys(where, p, PLAYER_KEYS, problems):
        return
    if not _nonempty(p["id"]):
        problems.append(f"{where}.id: not a non-empty string")
    if p["sha256"] is None:
        if "sha_unrecorded" not in flags:
            problems.append(f"{where}.sha256: null without the sha_unrecorded flag")
    elif not (isinstance(p["sha256"], str) and _SHA256.match(p["sha256"])):
        problems.append(f"{where}.sha256: not 64 lowercase hex")
    for k in ("path", "run", "rung"):
        if not _nonempty(p[k]):
            problems.append(f"{where}.{k}: not a non-empty string")
    if p["step"] is not None and not _is_int(p["step"]):
        problems.append(f"{where}.step: neither an int nor null")
    if p["kind"] not in PLAYER_KINDS:
        problems.append(f"{where}.kind: {p['kind']!r} not in {PLAYER_KINDS}")


def _check_temp(where: str, play: Any, temp: Any, problems: List[str]) -> None:
    if play == "greedy":
        if temp is not None:
            problems.append(f"{where}: must be null on a greedy side (got {temp!r})")
    elif play == "sampled":
        if not (temp == BOT_NATIVE_TEMP or (_is_num(temp) and temp > 0)):
            problems.append(f"{where}: a sampled side needs a positive temperature or {BOT_NATIVE_TEMP!r}")


def _check_regime(reg: Any, flags: Sequence[str], problems: List[str]) -> bool:
    """Returns ``mirrored``."""
    if not _check_keys("regime", reg, REGIME_KEYS, problems, optional=_REGIME_OPTIONAL):
        return False
    for k in ("play", "opponent_play"):
        if reg[k] not in PLAYS:
            problems.append(f"regime.{k}: {reg[k]!r} not in {PLAYS}")
    if not isinstance(reg["mirrored"], bool):
        problems.append("regime.mirrored: not a bool")
    if not _is_int(reg["turn_limit"]) or reg["turn_limit"] < 1:
        problems.append("regime.turn_limit: not a positive int")
    for k in ("mirror_rule", "seed_rule", "team_source"):
        if not _nonempty(reg[k]):
            problems.append(f"regime.{k}: not a non-empty string")
    if reg["eval_core"] is None:
        if "eval_core_unrecorded" not in flags:
            problems.append("regime.eval_core: null without the eval_core_unrecorded flag")
    elif not _nonempty(reg["eval_core"]):
        problems.append("regime.eval_core: not a non-empty string")
    if reg["protocol"] not in PROTOCOLS:
        problems.append(f"regime.protocol: {reg['protocol']!r} is not a known protocol {PROTOCOLS}")
    if reg["seat_rule"] not in SEAT_RULES:
        problems.append(f"regime.seat_rule: {reg['seat_rule']!r} not in {SEAT_RULES}")
    _check_temp("regime.player_temp", reg["play"], reg["player_temp"], problems)
    _check_temp("regime.opponent_temp", reg["opponent_play"], reg["opponent_temp"], problems)
    if not (isinstance(reg["team_set"], str) and _TEAM_SET.match(reg["team_set"])):
        problems.append("regime.team_set: not 'ts:' + 16 hex")
    if "v1_id" in reg and not _nonempty(reg["v1_id"]):
        problems.append("regime.v1_id: not a non-empty string")
    if reg["regime_id"] != regime_id(reg):
        problems.append("regime.regime_id: does not match the regime's identity fields")
    return reg["mirrored"] is True


def _check_request(row: Mapping[str, Any], problems: List[str]) -> None:
    req = row["request"]
    if req is None:
        legacy = row["provenance"] is not None or (isinstance(row["regime"], Mapping) and "v1_id" in row["regime"])
        if not legacy:
            problems.append("request: null on a row that is neither upgraded (regime.v1_id) nor backfilled "
                            "(provenance) — a v2 producer writes every batch FOR a request")
        return
    if not _check_keys("request", req, REQUEST_KEYS, problems):
        return
    if not _nonempty(req["id"]):
        problems.append("request.id: not a non-empty string")
    if req["kind"] not in REQUEST_KINDS:
        problems.append(f"request.kind: {req['kind']!r} not in {REQUEST_KINDS}")
    if req["family"] is not None and not _nonempty(req["family"]):
        problems.append("request.family: neither a non-empty string nor null")
    opened = _check_ts("request.opened", req["opened"], problems)
    ts = parse_ts(row["ts"])
    if opened and ts and opened > ts:
        problems.append("request.opened is after the row's ts")
    if not (_is_int(req["batch"]) and req["batch"] >= 0):
        problems.append("request.batch: not a non-negative int")


def _check_compute(row: Mapping[str, Any], flags: Sequence[str], problems: List[str]) -> None:
    cp = row["compute"]
    if not isinstance(cp, Mapping):
        problems.append("compute: not an object")
        return
    missing = [k for k in COMPUTE_AUDIT_KEYS if k not in cp]
    if missing:
        problems.append(f"compute: missing the audit keys {missing}")
        return
    dg, nt, margin = cp["outcome_digest"], cp["near_tie_games"], cp["digest_margin"]
    if "digest_unrecorded" in flags:
        if dg is not None or nt is not None:
            problems.append("compute: the digest_unrecorded flag needs a null outcome_digest and near_tie_games")
        return
    if dg is None:
        problems.append("compute.outcome_digest: null without the digest_unrecorded flag")
    elif not (isinstance(dg, str) and _SHA256.match(dg)):
        problems.append("compute.outcome_digest: not 64 lowercase hex")
    if not (isinstance(nt, list) and all(_is_int(x) for x in nt) and nt == sorted(set(nt))):
        problems.append("compute.near_tie_games: not a sorted list of distinct ints")
    else:
        s = row["seed"]
        if isinstance(s, Mapping) and _is_int(s.get("game_lo")) and _is_int(s.get("game_hi")):
            out = [x for x in nt if not s["game_lo"] <= x <= s["game_hi"]]
            if out:
                problems.append(f"compute.near_tie_games: {out[:4]} outside the game range")
    if margin is not None and not (_is_num(margin) and margin >= 0):
        problems.append("compute.digest_margin: neither a non-negative number nor null")


def _check_provenance(pv: Any, problems: List[str]) -> None:
    if pv is None:
        return
    if not _check_keys("provenance", pv, PROVENANCE_KEYS, problems):
        return
    if not _nonempty(pv["source_file"]):
        problems.append("provenance.source_file: not a non-empty string")
    if not (_is_int(pv["source_line"]) and pv["source_line"] >= 1):
        problems.append("provenance.source_line: not a positive int")
    if not (isinstance(pv["source_sha256"], str) and _SHA256.match(pv["source_sha256"])):
        problems.append("provenance.source_sha256: not 64 lowercase hex")
    if not _nonempty(pv["backfill_id"]):
        problems.append("provenance.backfill_id: not a non-empty string")


def validate_row(row: Any) -> List[str]:
    """Every way ``row`` fails ``gen3_eval_count_row_v2`` (EMPTY = valid). The shapes, the closed vocabularies, the
    flag rules (a field is null EXACTLY when its ``*_unrecorded`` flag says so) and the ARITHMETIC that ties the
    blocks together (W + L + D = games; half-points = 2W + D; voided <= aborted <= 2 voided; the team counters sum
    to the games and the wins; the seed's game range = games + what was not counted)."""
    problems: List[str] = []
    if not _check_keys("row", row, ROW_KEYS, problems):
        return problems
    if row["schema"] != SCHEMA:
        problems.append(f"schema: {row['schema']!r}, expected {SCHEMA!r}")
    flags = row["flags"]
    if not (isinstance(flags, list) and all(f in FLAGS for f in flags) and flags == sorted(set(flags))):
        problems.append(f"flags: not a sorted list of distinct values from {FLAGS}")
        flags = [f for f in flags if f in FLAGS] if isinstance(flags, list) else []
    _check_common(row, problems)
    _check_player("player", row["player"], flags, problems)
    _check_player("opponent", row["opponent"], flags, problems)
    _check_provenance(row["provenance"], problems)
    if row["purpose"] not in PURPOSES:
        problems.append(f"purpose: {row['purpose']!r} not in {PURPOSES}")
    elif row["purpose"] in BACKFILL_ONLY_PURPOSES and row["provenance"] is None:
        problems.append(f"purpose: {row['purpose']!r} is for BACKFILLED rows only (no provenance here); new pool "
                        "rows are 'matrix'")
    mirrored = _check_regime(row["regime"], flags, problems)
    _check_request(row, problems)
    games = _check_counts_block(row, mirrored, True, problems)
    if "draws_folded" in flags and isinstance(row["counts"], Mapping) and row["counts"].get("d") not in (0, None):
        problems.append("counts.d: must be 0 under the draws_folded flag (draws are indistinguishable there)")
    if row["teams"] is None:
        if "teams_unrecorded" not in flags:
            problems.append("teams: null without the teams_unrecorded flag")
    elif "teams_unrecorded" in flags:
        problems.append("teams: must be null under the teams_unrecorded flag")
    else:
        _check_teams(row, games, problems)
    if row["seed"] is None:
        if "seed_unrecorded" not in flags:
            problems.append("seed: null without the seed_unrecorded flag")
    elif "seed_unrecorded" in flags:
        problems.append("seed: must be null under the seed_unrecorded flag")
    else:
        _check_seed(row, games, mirrored, problems)
    _check_compute(row, flags, problems)
    return problems


def check_row(row: Any, where: str = "row") -> None:
    problems = validate_row(row)
    if problems:
        raise LedgerSchemaError(f"{where}: {len(problems)} schema problem(s):\n  " + "\n  ".join(problems))


# ------------------------------------------------------------------------------------------- v1 -> v2 on read
def v1_writer(row: Mapping[str, Any]) -> str:
    """Which v1 writer produced ``row`` — read off its seed block (the two v1 writers' item keys are disjoint):
    ``h2h`` (item key ``h2h``) or ``bot_rr`` (``botrr:<a>:<b>``). Anything else is refused, never guessed."""
    key = str((row.get("seed") or {}).get("item_key", ""))
    if key == "h2h":
        return "h2h"
    if key.startswith("botrr:"):
        return "bot_rr"
    raise LedgerSchemaError(f"row {row.get('row_id')!r}: a v1 row from an unknown writer (seed.item_key {key!r}); "
                            "only main.h2h and the bot round robin wrote v1 rows")


def upgrade_v1(row: Mapping[str, Any], where: str = "row") -> Dict[str, Any]:
    """The DETERMINISTIC v1 -> v2 upgrade on read (§0b.2). The stored row is never rewritten. ``row`` is validated
    as v1 first, then converted, then validated as v2:

    * ``request = None``, ``flags = ["digest_unrecorded"]``, ``provenance = None``;
    * ``kind = bot`` when the id starts ``bot:``, else ``checkpoint``;
    * ``protocol = gen3_eval_protocol_v1_<writer>``; ``seat_rule = balanced`` when ``mirror_rule`` contains
      ``seat_alternating_by_pair``, else ``fixed_p1``; each temperature ``None`` on a greedy side, ``bot_native`` on
      a sampled one; ``team_set`` = the digest of ``team_source``; ``regime_id`` recomputed over the v2 identity,
      the v1 id kept as ``v1_id``;
    * ``counts.aborted = 2 x pairs.voided`` (0 on an unmirrored row);
    * ``compute`` gains ``outcome_digest = near_tie_games = digest_margin = None``; v1 h2h's integer near-tie
      COUNTS (``near_tie_games`` / ``near_tie_games_wide``) move to ``near_tie_game_count`` /
      ``near_tie_game_count_wide`` (v2's ``near_tie_games`` is the list of game INDICES)."""
    problems = validate_row_v1(row)
    if problems:
        raise LedgerSchemaError(f"{where}: {len(problems)} v1 schema problem(s):\n  " + "\n  ".join(problems))
    writer = v1_writer(row)
    out = copy.deepcopy(dict(row))
    out["schema"] = SCHEMA
    for side in ("player", "opponent"):
        out[side]["kind"] = "bot" if str(out[side]["id"]).startswith("bot:") else "checkpoint"
    reg = out["regime"]
    v1_id = reg.pop("regime_id")
    reg["protocol"] = V1_WRITER_PROTOCOL[writer]
    reg["seat_rule"] = "balanced" if "seat_alternating_by_pair" in str(reg["mirror_rule"]) else "fixed_p1"
    reg["player_temp"] = None if reg["play"] == "greedy" else BOT_NATIVE_TEMP
    reg["opponent_temp"] = None if reg["opponent_play"] == "greedy" else BOT_NATIVE_TEMP
    reg["team_set"] = team_set_id({"team_source": reg["team_source"]})
    reg["v1_id"] = v1_id
    out["regime"] = with_regime_id(reg)
    out["counts"]["aborted"] = 2 * int(out["pairs"]["voided"]) if out["pairs"] else 0
    cp = out["compute"]
    for old, new in (("near_tie_games", "near_tie_game_count"), ("near_tie_games_wide", "near_tie_game_count_wide")):
        if old in cp:
            cp[new] = cp.pop(old)
    cp["outcome_digest"] = None
    cp["near_tie_games"] = None
    cp["digest_margin"] = None
    out["request"] = None
    out["flags"] = ["digest_unrecorded"]
    out["provenance"] = None
    check_row(out, f"{where} (upgraded from v1)")
    return out


def as_v2(row: Mapping[str, Any], where: str = "row") -> Dict[str, Any]:
    """``row`` as a validated v2 row: a v2 row checked, a v1 row upgraded. Any other schema is refused."""
    schema = row.get("schema") if isinstance(row, Mapping) else None
    if schema == SCHEMA:
        check_row(row, where)
        return dict(row)
    if schema == SCHEMA_V1:
        return upgrade_v1(row, where)
    raise LedgerSchemaError(f"{where}: schema {schema!r} is neither {SCHEMA!r} nor {SCHEMA_V1!r}")


# ------------------------------------------------------------------------------------------- companion streams
_EVENT_COMMON = ("schema", "event", "seq", "ts", "writer_id")
_UNIT = ("request_id", "batch", "player", "opponent", "regime_id")
_EVENT_KEYS = {
    "open": _EVENT_COMMON + ("request_id", "kind", "purpose", "family", "regime_id", "protocol", "spec"),
    "family": _EVENT_COMMON + ("family_id", "decision_kind", "rule", "protocol", "commit"),
    "claim": _EVENT_COMMON + _UNIT + ("producer", "host", "pid", "expires_at"),
    "void": _EVENT_COMMON + _UNIT + ("claim_seq", "reason"),
    "row": _EVENT_COMMON + _UNIT + ("claim_seq", "row_id", "seed_key"),
    "done": _EVENT_COMMON + ("request_id",),
    "cancel": _EVENT_COMMON + ("request_id", "reason"),
}


def validate_event(e: Any) -> List[str]:
    """Every way ``e`` fails ``gen3_eval_request_v1`` (EMPTY = valid)."""
    problems: List[str] = []
    if not isinstance(e, Mapping) or e.get("event") not in EVENT_KINDS:
        return [f"event: {(e or {}).get('event') if isinstance(e, Mapping) else e!r} not in {EVENT_KINDS}"]
    kind = e["event"]
    if not _check_keys(f"{kind} event", e, _EVENT_KEYS[kind], problems):
        return problems
    if e["schema"] != EVENT_SCHEMA:
        problems.append(f"schema: {e['schema']!r}, expected {EVENT_SCHEMA!r}")
    if not (_is_int(e["seq"]) and e["seq"] >= 1):
        problems.append("seq: not a positive int")
    _check_ts("ts", e["ts"], problems)
    if not _nonempty(e["writer_id"]):
        problems.append("writer_id: not a non-empty string")
    if kind == "family":
        if not _nonempty(e["family_id"]):
            problems.append("family_id: not a non-empty string")
        if e["decision_kind"] not in GROUP_SEQUENTIAL_KINDS:
            problems.append(f"decision_kind: {e['decision_kind']!r} is not a group-sequential kind "
                            f"{GROUP_SEQUENTIAL_KINDS}")
        if not _nonempty(e["rule"]):
            problems.append("rule: a family must name its rule")
        if e["protocol"] not in PROTOCOLS:
            problems.append(f"protocol: {e['protocol']!r} not a known protocol")
        if e["commit"] is not None and not _nonempty(e["commit"]):
            problems.append("commit: neither a non-empty string nor null")
        return problems
    if not _nonempty(e["request_id"]):
        problems.append("request_id: not a non-empty string")
    if kind == "open":
        if e["kind"] not in REQUEST_KINDS:
            problems.append(f"kind: {e['kind']!r} not in {REQUEST_KINDS}")
        if e["purpose"] not in PURPOSES:
            problems.append(f"purpose: {e['purpose']!r} not in {PURPOSES}")
        for k in ("family", "regime_id"):
            if e[k] is not None and not _nonempty(e[k]):
                problems.append(f"{k}: neither a non-empty string nor null")
        if e["protocol"] is not None and e["protocol"] not in PROTOCOLS:
            problems.append(f"protocol: {e['protocol']!r} not a known protocol")
        if not isinstance(e["spec"], Mapping):
            problems.append("spec: not an object")
    elif kind in ("claim", "void", "row"):
        if not (_is_int(e["batch"]) and e["batch"] >= 0):
            problems.append("batch: not a non-negative int")
        for k in ("player", "opponent", "regime_id"):
            if not _nonempty(e[k]):
                problems.append(f"{k}: not a non-empty string")
        if kind == "claim":
            if not _nonempty(e["producer"]) or not _nonempty(e["host"]):
                problems.append("producer / host: not non-empty strings")
            if not (_is_int(e["pid"]) and e["pid"] > 0):
                problems.append("pid: not a positive int")
            _check_ts("expires_at", e["expires_at"], problems)
        elif not (_is_int(e["claim_seq"]) and e["claim_seq"] >= 1):
            problems.append("claim_seq: not a positive int")
        if kind == "void" and e["reason"] not in VOID_REASONS:
            problems.append(f"reason: {e['reason']!r} not in {VOID_REASONS}")
        if kind == "row":
            if not _nonempty(e["row_id"]):
                problems.append("row_id: not a non-empty string")
            if e["seed_key"] is not None and not isinstance(e["seed_key"], list):
                problems.append("seed_key: neither a list nor null")
    elif kind == "cancel" and not _nonempty(e["reason"]):
        problems.append("reason: a cancel says why")
    return problems


_DECISION_KEYS = ("schema", "decision_id", "kind", "subject", "request_id", "family", "consumed", "as_of", "rule",
                  "rule_version", "verdict", "ts", "writer_id")


def validate_decision(d: Any) -> List[str]:
    """Every way ``d`` fails ``gen3_eval_decision_v1`` (EMPTY = valid). A decision names what it decided (a request
    or a family), the rows it consumed (their ids, count and digest), its ``as_of``, its rule and version, and its
    verdict. Decision rows are never edited."""
    problems: List[str] = []
    if not _check_keys("decision", d, _DECISION_KEYS, problems):
        return problems
    if d["schema"] != DECISION_SCHEMA:
        problems.append(f"schema: {d['schema']!r}, expected {DECISION_SCHEMA!r}")
    for k in ("decision_id", "subject", "rule", "rule_version", "verdict", "writer_id"):
        if not _nonempty(d[k]):
            problems.append(f"{k}: not a non-empty string")
    if d["kind"] not in DECISION_KINDS:
        problems.append(f"kind: {d['kind']!r} not in {DECISION_KINDS}")
    if d["request_id"] is None and d["family"] is None:
        problems.append("a decision names the request_id or the family it decided")
    for k in ("request_id", "family"):
        if d[k] is not None and not _nonempty(d[k]):
            problems.append(f"{k}: neither a non-empty string nor null")
    _check_ts("as_of", d["as_of"], problems)
    _check_ts("ts", d["ts"], problems)
    c = d["consumed"]
    if _check_keys("consumed", c, ("digest", "count", "row_ids"), problems):
        if not (isinstance(c["digest"], str) and _SHA256.match(c["digest"])):
            problems.append("consumed.digest: not 64 lowercase hex")
        ids = c["row_ids"]
        if not (isinstance(ids, list) and all(_nonempty(x) for x in ids) and ids == sorted(set(ids))):
            problems.append("consumed.row_ids: not a sorted list of distinct row ids")
        elif c["count"] != len(ids):
            problems.append(f"consumed.count {c['count']} != {len(ids)} row ids")
    return problems


_REFERENCE_KEYS = ("schema", "reference_id", "members", "weights", "solver", "solver_version", "draws", "seed",
                   "rows_digest", "created_at", "writer_id")


def validate_reference(r: Any) -> List[str]:
    """Every way ``r`` fails ``gen3_eval_reference_v1`` (EMPTY = valid): an IMMUTABLE frozen reference mixture
    (§2.3) — members (by sha256), weights summing to 1 within 1e-9, the solver and its version, the posterior draw
    count and seed, and the digest of the rows it was solved from."""
    problems: List[str] = []
    if not _check_keys("reference", r, _REFERENCE_KEYS, problems):
        return problems
    if r["schema"] != REFERENCE_SCHEMA:
        problems.append(f"schema: {r['schema']!r}, expected {REFERENCE_SCHEMA!r}")
    for k in ("reference_id", "solver", "solver_version", "writer_id"):
        if not _nonempty(r[k]):
            problems.append(f"{k}: not a non-empty string")
    m, w = r["members"], r["weights"]
    if not (isinstance(m, list) and m and all(isinstance(x, Mapping) and set(x) == {"id", "sha256"}
                                              and _nonempty(x["id"]) and isinstance(x["sha256"], str)
                                              and _SHA256.match(x["sha256"]) for x in m)):
        problems.append("members: not a non-empty list of {id, sha256}")
    elif len({x["sha256"] for x in m}) != len(m):
        problems.append("members: a member appears twice")
    if not (isinstance(w, list) and all(_is_num(x) and x >= 0 for x in w)):
        problems.append("weights: not a list of non-negative numbers")
    elif isinstance(m, list) and len(w) != len(m):
        problems.append(f"weights: {len(w)} weights for {len(m)} members")
    elif abs(sum(w) - 1.0) > 1e-9:
        problems.append(f"weights: sum {sum(w)!r} is not 1 within 1e-9")
    for k in ("draws", "seed"):
        if not (_is_int(r[k]) and r[k] >= 0):
            problems.append(f"{k}: not a non-negative int")
    if not (isinstance(r["rows_digest"], str) and _SHA256.match(r["rows_digest"])):
        problems.append("rows_digest: not 64 lowercase hex")
    _check_ts("created_at", r["created_at"], problems)
    return problems


def check(problems: List[str], where: str) -> None:
    if problems:
        raise LedgerSchemaError(f"{where}: {len(problems)} schema problem(s):\n  " + "\n  ".join(problems))
