"""THE IN-LOOP EVAL PRODUCERS ON THE COUNT LEDGER — eval unit U2 (``designs/endstate/design_evaluation.md`` §0b.5).

The trainer's in-process eval cycle (``PerOpponentEvalCallback`` / ``SelfPlayCallback`` → ``rust_eval``) and the
T6 SPRT promotion (``sprt_promotion``) write ``gen3_eval_count_row_v2`` rows through the U1 writer
(``agents.training.eval_ledger``) to the ARCHIVE ledger, ``run_archive_dir()/_ledger/rows/inloop/``. STORAGE ONLY:
the games are the executor's, unchanged — a :class:`GameSink` only RECORDS each finished game (its outcome, end
turn, teams and whether any decision sat inside the near-tie margin); nothing it does feeds back into a game. The
proof is the outcome-digest check in ``designs/research_state/measurements/eval_ledger_u2_2026-10-04/``.

THE TRANSITION (DUAL WRITE). ``<run>/eval_results.jsonl`` (``record_elo``), ``metadata.json``'s eval block and
``<run>/sprt_promotion.jsonl`` are written EXACTLY as before and every existing reader keeps reading them; the
ledger rows are written BESIDE them. Moving those readers onto the ledger and retiring the file is eval unit U3c.

WHAT IS WRITTEN

* **The cycle** — purpose ``cycle``, request kind ``cycle`` (§0b.5). ONE request per (cycle step x REGIME): the
  ledger holds one regime per request, and a cycle's opponents are not one regime (a bot plays from its own seeded
  streams, a sentinel greedy or at ``--self-play-temp``, a fixed opponent on its pinned teams). Request id
  ``<run>:cycle:<step>:<regime_id>``; one row per opponent, ``request.batch`` 0; the request is ``done`` once its
  rows are down.
* **The SPRT** — purpose ``promotion``, request kind ``sprt``: ONE request per candidate (``<run>:sprt:<step>``),
  one row per (batch x sentinel) with ``request.batch`` = the test's batch index, and at the verdict ONE decision
  row (kind ``promotion``, subject = the candidate's sha256, consuming the request's rows; verdict ``accept`` /
  ``reject`` / ``abandoned``). Resuming a test across a restart is U4b — today's abandon rule is unchanged.
* **The regime**, stamped on every row: protocol :data:`PROTOCOL`, ``seat_rule = fixed_p1`` (the core seats the
  trainee at p1, §9.3), the trainee GREEDY (``player_temp`` null), the opponent's play + temperature, ``mirrored`` +
  its rule, and ``team_set`` = the digest of the trainee's and the opponent's eval team builders (their ordered
  team lists and bias parameters).
* **Counts**: exact W / L / D (a draw is COUNTED, never folded — F-ED-8), ``aborted`` (0: a committed cycle
  finished every game), the pentanomial on a mirrored row, per-team counters, and the per-batch OUTCOME DIGEST
  with the near-tie game indices at :data:`DIGEST_MARGIN` — plus ``compute.outcome_digest_all`` over EVERY game
  (no exclusion), for a SAME-device replay: on an early, near-uniform policy nearly every game holds a decision
  inside the GPU bar, so the margin-filtered digest covers almost nothing.

ABORTS (P10-A2's safe points). Rows are appended only AFTER the cycle (or the SPRT batch) has played to its end and
its published shard results agree with the game log. A stop honoured at a safe point exits the process inside the
cycle, so it writes NOTHING; its claims are left live and the deterministic void rule (§0b.4) voids them once the
pid is dead, so a re-eval at the same step re-claims and replays the same seed block. A cycle that FAILS on the eval
core (``EvalCoreError``) writes no row either: its requests are CANCELLED with the reason.

GIGO GUARD. Every row is cross-checked against the executor's published shard results (``merge_eval_results``): W,
finished, draws and the pentanomial must agree per opponent, or :class:`CycleLedgerError` is RAISED — a row that
disagrees with what the eval curve records is never written.

DECLARED LIFECYCLE. One :class:`CycleLedger` (one ``LedgerWriter``) per trainer process, built at startup
(``main.train.callbacks``); the writer opens its files per append and takes the ledger's file lock per claim /
append. Nothing here builds a module or an optimizer.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.training import eval_ledger as L
from agents.training import mirrored_pairs as MP

#: The game protocol the in-loop producers play (``eval_ledger.schema.PROTOCOLS``): the Rust eval core
#: (``rust_eval.executor``), the trainee at p1 and greedy, games from ``rust_eval.seeds`` (``pair_game`` when
#: mirrored), teams from ``rust_eval.build.eval_builders``, the game ending at the stall threshold. A change to
#: what a game measures bumps it; a storage-only change keeps it.
PROTOCOL = "gen3_eval_protocol_v1_inloop"
PRODUCER = "inloop"
#: A decision whose top-2 margin is below this is "within a rounding error of a tie" for the OUTCOME DIGEST: the
#: GPU bar (``main.h2h.play.DIGEST_MARGIN`` / ``rust_eval.parity.BAR_GPU`` x 2) — the in-loop cycle plays on the
#: training card, so a game that could flip between devices is listed by index, not hashed.
DIGEST_MARGIN = 2e-3
#: The executor's own narrow near-tie bar (``rust_eval.executor.NEAR_TIE``), counted per row as a diagnostic.
NEAR_TIE = 2e-5
UNMIRRORED = "none"
#: A scripted bot's identity: its name + the sources of the Rust bot implementation at this commit (a bot is code).
BOT_SOURCES = ("rust_env/src/bots", "rust_env/src/opponents.rs")
#: The first claim's expected wall, in games per second (the claim floor, 10 min, dominates at the eval sizes).
EXPECTED_GAMES_PER_S = 50.0
TEAM_SOURCE = ("rust_eval.build.eval_builders: the trainee builder (eval_teams.build_trainee_tb) vs the opponent's "
               "(RustEvalCore.opponent_builder)")


class CycleLedgerError(RuntimeError):
    """The game log and the executor's published results disagree (GIGO): the row is NOT written."""


# ---------------------------------------------------------------------------------------------- the game sink
class GameSink(list):
    """The executor's ``game_log``, trimmed AT THE SOURCE: a game's input log and per-decision arrays are dropped;
    what is kept is what a count row needs — item, game index, result, end turn, the mirror fields, the teams, the
    trainee's decision count and whether any decision (either side's) sat inside :data:`DIGEST_MARGIN`."""

    def append(self, row: Mapping[str, Any]) -> None:  # type: ignore[override]
        margins = [float(m) for m in row.get("margins", ())]
        opp = [float(o[3]) for o in row.get("opp", ())]
        super().append({
            "item": row["item"], "kind": row.get("kind"), "game": int(row["game"]), "result": row["result"],
            "end_turn": None if row.get("end_turn") is None else int(row["end_turn"]),
            "swapped": bool(row.get("swapped", False)), "teams": [int(x) for x in row["teams"]],
            "decisions": len(margins),
            "near_ties": sum(1 for m in margins if m < NEAR_TIE),
            "near_tie_wide": any(m < DIGEST_MARGIN for m in margins + opp)})


# ---------------------------------------------------------------------------------------------- identities
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bot_sha256(name: str) -> str:
    """sha256 of (``bot:<name>``, the Rust bot sources at this commit) — §0b.2: a non-checkpoint's digest of its
    agent name and code."""
    from utils.paths import src_path

    h = hashlib.sha256(f"bot:{name}".encode())
    files: List[str] = []
    for rel in BOT_SOURCES:
        p = src_path(*rel.split("/"))
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".rs"))
        elif os.path.exists(p):
            files.append(str(p))
    if not files:
        raise CycleLedgerError(f"no Rust bot source under {BOT_SOURCES}: a bot's identity cannot be digested")
    for p in files:
        with open(p, "rb") as f:
            h.update(os.path.basename(p).encode())
            h.update(f.read())
    return h.hexdigest()


def builder_ident(builder: Any) -> Dict[str, Any]:
    """What defines a ``Gen3Teambuilder``'s draws: its ordered team list, its bias teams and bias probability."""
    return {"teams": list(getattr(builder, "packed_teams", ()) or ()),
            "bias_teams": list(getattr(builder, "bias_packed_teams", ()) or ()),
            "bias_prob": float(getattr(builder, "bias_prob", 0.0) or 0.0)}


def opponent_play(kind: str, *, sentinel_greedy: bool, self_play_temp: float) -> Tuple[str, Any]:
    """``(opponent_play, opponent_temp)`` of an eval item kind on the executor (module docs of ``rust_eval``)."""
    from agents.training.eval_sharding import BOT, FIXED, SENTINEL

    if kind == BOT:
        return "sampled", L.BOT_NATIVE_TEMP
    if kind == SENTINEL:
        return ("greedy", None) if sentinel_greedy else ("sampled", float(self_play_temp))
    if kind == FIXED:
        return "greedy", None
    raise CycleLedgerError(f"unknown eval item kind {kind!r}")


# ---------------------------------------------------------------------------------------------- scoring
@dataclass
class ItemScore:
    w: int
    l: int
    d: int
    n_games: int
    pair_counts: Optional[List[int]]
    teams: Dict[str, Dict[str, List[int]]]
    outcome_digest: str
    near_tie_games: List[int]
    #: the digest over EVERY game (no near-tie exclusion): what a SAME-device replay checks game for game
    outcome_digest_all: str
    trainee_decisions: int
    near_tie_decisions: int


def score_item(games: Sequence[Mapping[str, Any]], *, n_games: int, team_packed: Sequence[str],
               mirrored: bool) -> ItemScore:
    """One opponent's games (the sink's rows for one item) into counts. Every check is an ASSERTION, never a branch:
    games ``0 .. n_games - 1`` exactly once each; on a mirrored plan game ``2k`` unswapped and ``2k+1`` swapped with
    the teams handed over."""
    from agents.training.trace_result import DRAW, LOSS, WIN

    by_g: Dict[int, Mapping[str, Any]] = {}
    for g in games:
        k = int(g["game"])
        if k in by_g:
            raise CycleLedgerError(f"game {k} appears twice")
        by_g[k] = g
    want = set(range(int(n_games)))
    if set(by_g) != want:
        raise CycleLedgerError(f"the game log holds games {sorted(set(by_g) ^ want)[:8]} off the plan "
                               f"0..{n_games - 1}")
    teams: Dict[str, Dict[str, List[int]]] = {}
    w = l = d = 0
    pts: List[Optional[int]] = []
    letter = {WIN: "W", LOSS: "L", DRAW: "D"}
    vec: List[Tuple[int, str, Optional[int]]] = []
    for k in range(int(n_games)):
        g = by_g[k]
        r = g["result"]
        if r not in letter:
            raise CycleLedgerError(f"game {k}: unknown result {r!r}")
        w, l, d = w + (r == WIN), l + (r == LOSS), d + (r == DRAW)
        tp, to = L.team_id(team_packed[int(g["teams"][0])]), L.team_id(team_packed[int(g["teams"][1])])
        for tid, side, won in ((tp, "p", r == WIN), (to, "o", r == LOSS)):
            t = teams.setdefault(tid, {"p": [0, 0], "o": [0, 0]})
            t[side][0] += 1
            t[side][1] += int(won)
        pts.append(MP.result_points(r))
        vec.append((k, letter[r], g.get("end_turn")))
    pc = None
    if mirrored:
        if n_games % 2:
            raise CycleLedgerError(f"{n_games} games is not a whole number of mirrored pairs")
        for k in range(0, int(n_games), 2):
            a, b = by_g[k], by_g[k + 1]
            if bool(a["swapped"]) or not bool(b["swapped"]):
                raise CycleLedgerError(f"pair {k // 2}: games are not (unswapped, swapped)")
            if list(b["teams"]) != [a["teams"][1], a["teams"][0]]:
                raise CycleLedgerError(f"pair {k // 2}: the second game's teams are not the first's handed over")
        pc = MP.pair_counts(pts)
    near = sorted(k for k, g in by_g.items() if g.get("near_tie_wide"))
    return ItemScore(w=w, l=l, d=d, n_games=int(n_games), pair_counts=pc, teams=teams,
                     outcome_digest=L.outcome_digest(vec, near), near_tie_games=near,
                     outcome_digest_all=L.outcome_digest(vec),
                     trainee_decisions=sum(int(g.get("decisions", 0)) for g in by_g.values()),
                     near_tie_decisions=sum(int(g.get("near_ties", 0)) for g in by_g.values()))


def cross_check(key: str, sc: ItemScore, merged: Mapping[str, Any]) -> None:
    """The GIGO guard (module docs): the row's counts against the executor's published, merged results."""
    cnt = (merged.get("counts") or {}).get(key)
    if cnt is None:
        raise CycleLedgerError(f"{key}: no published result to check the game log against")
    won, finished = int(cnt[0]), int(cnt[1])
    drawn = (merged.get("draws") or {}).get(key)
    bad = []
    if won != sc.w:
        bad.append(f"W {sc.w} vs published {won}")
    if finished != sc.w + sc.l + sc.d:
        bad.append(f"finished {sc.w + sc.l + sc.d} vs published {finished}")
    if drawn is not None and int(drawn) != sc.d:
        bad.append(f"D {sc.d} vs published {drawn}")
    if sc.pair_counts is not None and list((merged.get("pairs") or {}).get(key) or []) != sc.pair_counts:
        bad.append(f"pairs {sc.pair_counts} vs published {(merged.get('pairs') or {}).get(key)}")
    if bad:
        raise CycleLedgerError(f"{key}: the game log disagrees with the published shard results: {'; '.join(bad)}")


# ---------------------------------------------------------------------------------------------- one batch
@dataclass
class Cell:
    """One opponent of one batch: its blocks, its regime, its request (the ``open`` event) and its claim."""

    item: Any
    opponent: Dict[str, Any]
    regime: Dict[str, Any]
    request: Dict[str, Any]
    claim: Any


@dataclass
class LedgerBatch:
    """One batch's claimed cells, between the claims and the rows (:meth:`CycleLedger.commit`)."""

    kind: str
    purpose: str
    step: int
    batch: int
    schedule_key: str
    player: Dict[str, Any]
    mirrored: bool
    cells: Dict[str, Cell] = field(default_factory=dict)
    t_start: str = ""
    t0: float = 0.0
    done_requests: Tuple[str, ...] = ()


def build_row(*, row_id: str, run_label: str, commit: str, player: Mapping[str, Any], cell: Cell, purpose: str,
              batch: int, sc: ItemScore, seed: Mapping[str, Any], t_start: str, t_end: str,
              compute: Mapping[str, Any]) -> Dict[str, Any]:
    """The §0b.2 row of one cell (the producer contract test pins it valid)."""
    req = cell.request
    return {
        "schema": L.SCHEMA, "row_id": row_id, "supersedes": None, "ts": L.utc_now(), "t_start": t_start,
        "t_end": t_end, "run": run_label, "commit": commit, "player": dict(player), "opponent": dict(cell.opponent),
        "regime": dict(cell.regime),
        "compute": {**compute, "trainee_decisions": sc.trainee_decisions,
                    "near_tie_decisions": sc.near_tie_decisions, "near_tie_margin": NEAR_TIE,
                    "outcome_digest": sc.outcome_digest, "near_tie_games": list(sc.near_tie_games),
                    "digest_margin": DIGEST_MARGIN, "outcome_digest_all": sc.outcome_digest_all},
        "purpose": purpose,
        "request": {"id": req["request_id"], "kind": req["kind"], "family": req["family"], "opened": req["ts"],
                    "batch": int(batch)},
        "counts": {"w": sc.w, "l": sc.l, "d": sc.d, "aborted": 0},
        "pairs": ({"counts": list(sc.pair_counts), "n_pairs": sum(sc.pair_counts), "voided": 0}
                  if sc.pair_counts is not None else None),
        "teams": sc.teams,
        "seed": {**seed, "game_lo": 0, "game_hi": sc.n_games - 1},
        "flags": [], "provenance": None}


class CycleLedger:
    """The in-loop producers' handle on the ledger (module docs). One per trainer process, built at startup."""

    def __init__(self, root: "str | os.PathLike[str] | None", *, run_label: str, commit: str,
                 writer: Optional[L.LedgerWriter] = None, emit: Callable[[str], None] = print):
        self.writer = writer if writer is not None else L.LedgerWriter(root, producer=PRODUCER)
        self.root = self.writer.root
        self.run_label = run_label
        self.code_commit = commit or "unknown"
        self.emit = emit
        self._sha: Dict[Tuple[str, int, int], str] = {}
        self._bot: Dict[str, str] = {}
        self._ts: Dict[Tuple[int, int], str] = {}
        self.rows_written = 0

    @classmethod
    def for_run(cls, model_dir: str, emit: Callable[[str], None] = print) -> "CycleLedger":
        """STARTUP: the archive ledger (``run_archive_dir()/_ledger``), the run's label and this code's commit."""
        from utils.git import get_git_hash

        try:
            commit = get_git_hash() or "unknown"
        except Exception:                                    # noqa: BLE001 - a row still records "unknown"
            commit = "unknown"
        led = cls(None, run_label=os.path.basename(os.path.normpath(model_dir)), commit=commit, emit=emit)
        emit(f"📒 [EVAL LEDGER] the in-loop eval writes COUNT rows (protocol {PROTOCOL}) to {led.root} — beside "
             "eval_results.jsonl, which every existing reader still reads (dual write until eval U3c)")
        return led

    def close(self) -> None:
        self.writer.close()

    # ------------------------------------------------------------------ identities
    def sha_of(self, path: str) -> str:
        st = os.stat(path)
        k = (os.path.realpath(path), int(st.st_size), int(st.st_mtime_ns))
        if k not in self._sha:
            self._sha[k] = file_sha256(path)
        return self._sha[k]

    def bot_block(self, name: str) -> Dict[str, Any]:
        if name not in self._bot:
            self._bot[name] = bot_sha256(name)
        return {"id": f"bot:{name}", "sha256": self._bot[name], "path": "src/" + BOT_SOURCES[0],
                "run": "scripted_bots", "step": None, "rung": "scripted_rust", "kind": "bot"}

    def player_block(self, snapshot: str, step: int) -> Dict[str, Any]:
        return {"id": f"{self.run_label}@{int(step)}", "sha256": self.sha_of(snapshot), "path": str(snapshot),
                "run": self.run_label, "step": int(step), "rung": "eval_cycle_snapshot", "kind": "checkpoint"}

    def opponent_block(self, item: Any) -> Dict[str, Any]:
        from agents.training.eval_sharding import BOT, SENTINEL

        if item.kind == BOT:
            return self.bot_block(item.key)
        if item.kind == SENTINEL:
            return {"id": f"{self.run_label}:pool@{item.step}", "sha256": self.sha_of(item.path), "path": str(item.path),
                    "run": self.run_label, "step": None if item.step is None else int(item.step),
                    "rung": "pool_sentinel", "kind": "checkpoint"}
        run = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(item.path)))) or "external"
        return {"id": f"ext:{item.key}", "sha256": self.sha_of(item.path), "path": str(item.path), "run": run,
                "step": None if item.step is None else int(item.step), "rung": "fixed_opponent", "kind": "checkpoint"}

    def team_set(self, trainee_builder: Any, opp_builder: Any) -> str:
        k = (id(trainee_builder), id(opp_builder))
        if k not in self._ts:
            self._ts[k] = L.team_set_id({"builder": "Gen3Teambuilder", "trainee": builder_ident(trainee_builder),
                                         "opponent": builder_ident(opp_builder)})
        return self._ts[k]

    def regime_for(self, ev: Any, item: Any, *, mirrored: bool, sentinel_greedy: bool,
                   self_play_temp: float) -> Dict[str, Any]:
        from agents.training.rust_eval import seeds as SD

        oplay, otemp = opponent_play(item.kind, sentinel_greedy=sentinel_greedy, self_play_temp=self_play_temp)
        return L.with_regime_id({
            "play": "greedy", "opponent_play": oplay, "mirrored": bool(mirrored),
            "mirror_rule": MP.SCHEMA if mirrored else UNMIRRORED, "eval_core": "rust",
            "turn_limit": int(ev.turn_limit), "seed_rule": SD.SCHEMA, "team_source": TEAM_SOURCE,
            "protocol": PROTOCOL, "seat_rule": "fixed_p1", "player_temp": None, "opponent_temp": otemp,
            "team_set": self.team_set(ev.trainee_builder, ev.opponent_builder(item, sentinel_greedy))})

    # ------------------------------------------------------------------ claims
    def _claim(self, b: LedgerBatch, item: Any, request: Dict[str, Any], regime: Dict[str, Any],
               opponent: Dict[str, Any]) -> None:
        try:
            c = self.writer.claim(request["request_id"], batch=b.batch, player=b.player["sha256"],
                                  opponent=opponent["sha256"], regime_id=regime["regime_id"],
                                  expected_wall_s=item.n_games / EXPECTED_GAMES_PER_S)
        except (L.AlreadyRecordedError, L.ClaimHeldError) as e:
            self.emit(f"📒 [EVAL LEDGER] {b.kind} @{b.step:,} batch {b.batch} vs {item.key}: {e} — no row this time")
            return
        b.cells[item.key] = Cell(item=item, opponent=opponent, regime=regime, request=request, claim=c)

    def open_cycle(self, ev: Any, pool: Any, *, step: int, snapshot: str, mirrored: bool, sentinel_greedy: bool,
                   self_play_temp: float) -> LedgerBatch:
        """Open the cycle's requests (one per regime) and claim one cell per opponent — BEFORE the cycle plays."""
        b = LedgerBatch(kind="cycle", purpose="cycle", step=int(step), batch=0,
                        schedule_key=f"inloop:cycle:{int(step)}", player=self.player_block(snapshot, step),
                        mirrored=bool(mirrored))
        groups: Dict[str, List[Tuple[Any, Dict[str, Any]]]] = {}
        for it in pool.items:
            reg = self.regime_for(ev, it, mirrored=mirrored, sentinel_greedy=sentinel_greedy,
                                  self_play_temp=self_play_temp)
            groups.setdefault(reg["regime_id"], []).append((it, reg))
        opened: List[str] = []
        for reg_id, members in groups.items():
            rid = f"{self.run_label}:cycle:{int(step)}:{reg_id}"
            spec = {"producer": PRODUCER, "step": int(step), "mirrored": bool(mirrored),
                    "games": {it.key: int(it.n_games) for it, _r in members}}
            try:
                req = self.writer.open_request(rid, kind="cycle", purpose="cycle", regime_id=reg_id,
                                               protocol=PROTOCOL, spec=spec)
            except L.RequestSpecError as e:
                self.emit(f"📒 [EVAL LEDGER] cycle @{int(step):,}: {e} — this regime's opponents write no row")
                continue
            opened.append(rid)
            for it, reg in members:
                self._claim(b, it, req, reg, self.opponent_block(it))
        b.done_requests = tuple(opened)
        b.t_start, b.t0 = L.utc_now(), time.perf_counter()
        return b

    def cancel(self, b: LedgerBatch, reason: str) -> None:
        """A cycle that failed on the eval core: no row; its requests are CANCELLED with the reason."""
        for rid in b.done_requests:
            self.writer.finish_request(rid, cancel_reason=reason)
        self.emit(f"📒 [EVAL LEDGER] {b.kind} @{b.step:,}: no rows ({reason})")

    # ------------------------------------------------------------------ rows
    def commit(self, b: LedgerBatch, games: Sequence[Mapping[str, Any]], merged: Mapping[str, Any], *,
               team_packed: Sequence[str], run_seed: int, cycle_seed: int, compute: Mapping[str, Any]) -> List[str]:
        """Score, cross-check and append one row per claimed cell (module docs); ``done`` the cycle's requests."""
        t_end = L.utc_now()
        wall = time.perf_counter() - b.t0
        by_item: Dict[str, List[Mapping[str, Any]]] = {}
        for g in games:
            by_item.setdefault(str(g["item"]), []).append(g)
        scores = {}
        for key, cell in b.cells.items():                 # score + check EVERY cell before appending ANY row
            sc = score_item(by_item.get(key, []), n_games=int(cell.item.n_games), team_packed=team_packed,
                            mirrored=b.mirrored)
            cross_check(key, sc, merged)
            scores[key] = sc
        out = []
        for key, cell in b.cells.items():
            sc = scores[key]
            row = build_row(row_id=self.writer.next_row_id(), run_label=self.run_label, commit=self.code_commit,
                            player=b.player, cell=cell, purpose=b.purpose, batch=b.batch, sc=sc,
                            seed={"rule": cell.regime["seed_rule"], "schedule_seed": int(run_seed),
                                  "schedule_key": b.schedule_key, "batch": int(b.batch), "cycle_seed": int(cycle_seed),
                                  "item_key": key},
                            t_start=b.t_start, t_end=t_end,
                            compute={**compute, "batch_wall_s": round(wall, 3), "step": b.step})
            try:
                out.append(self.writer.append_row(row, cell.claim))
            except (L.ClaimVoidedError, L.DuplicateBatchError) as e:
                self.emit(f"📒 [EVAL LEDGER] {b.kind} @{b.step:,} batch {b.batch} vs {key}: {e} — row dropped")
        if b.kind == "cycle":
            for rid in b.done_requests:
                self.writer.finish_request(rid)
        self.rows_written += len(out)
        return out

    # ------------------------------------------------------------------ the SPRT
    def open_sprt(self, ev: Any, *, step: int, snapshot: str, sentinel_items: Sequence[Any], mirrored: bool,
                  sentinel_greedy: bool, self_play_temp: float, config: Mapping[str, Any]) -> "SprtLedger":
        """One request per candidate (``<run>:sprt:<step>``), opened at the test's start."""
        player = self.player_block(snapshot, step)
        regs = {self.regime_for(ev, it, mirrored=mirrored, sentinel_greedy=sentinel_greedy,
                                self_play_temp=self_play_temp)["regime_id"] for it in sentinel_items}
        if len(regs) != 1:
            raise CycleLedgerError(f"the SPRT's sentinels are {len(regs)} regimes; one request holds one")
        rid = f"{self.run_label}:sprt:{int(step)}"
        req = self.writer.open_request(rid, kind="sprt", purpose="promotion", regime_id=regs.pop(), protocol=PROTOCOL,
                                       spec={"producer": PRODUCER, "step": int(step), "config": dict(config),
                                             "candidate_sha256": player["sha256"],
                                             "pool": sorted(str(it.path) for it in sentinel_items)})
        return SprtLedger(self, ev, req, player, step=int(step), mirrored=mirrored, sentinel_greedy=sentinel_greedy,
                          self_play_temp=self_play_temp)


#: The SPRT decision's read of its OWN request's rows (§0c rule 1: a sequential decision counts only the rows
#: produced FOR it). ``selection="include"``: these rows ARE the decision's evidence.
SPRT_DECISION_READ = L.ReaderDecl(
    name="cycle_ledger.sprt_decision", purposes=frozenset({"promotion"}),
    regime=L.RegimeFilter(protocol=PROTOCOL, play="greedy", mirrored=True, seat_rule="fixed_p1"),
    requests="own", selection="include", flags_ok=frozenset(), inference="conditional")


class SprtLedger:
    """One candidate's SPRT on the ledger: a claimed batch per test batch, then ONE decision row."""

    def __init__(self, led: CycleLedger, ev: Any, request: Dict[str, Any], player: Dict[str, Any], *, step: int,
                 mirrored: bool, sentinel_greedy: bool, self_play_temp: float):
        self.led, self.ev, self.request, self.player = led, ev, request, player
        self.step, self.mirrored = step, mirrored
        self.sentinel_greedy, self.self_play_temp = sentinel_greedy, self_play_temp

    @property
    def request_id(self) -> str:
        return str(self.request["request_id"])

    def claim_batch(self, items: Sequence[Any], batch: int) -> LedgerBatch:
        b = LedgerBatch(kind="sprt", purpose="promotion", step=self.step, batch=int(batch),
                        schedule_key=f"inloop:sprt:{self.step}", player=self.player, mirrored=self.mirrored)
        for it in items:
            reg = self.led.regime_for(self.ev, it, mirrored=self.mirrored, sentinel_greedy=self.sentinel_greedy,
                                      self_play_temp=self.self_play_temp)
            self.led._claim(b, it, self.request, reg, self.led.opponent_block(it))
        b.t_start, b.t0 = L.utc_now(), time.perf_counter()
        return b

    def decide(self, *, verdict: str, rule: str, config: Mapping[str, Any]) -> Dict[str, Any]:
        """The test's ONE decision row (consuming every row of its request), then ``done``."""
        consumed = L.read(SPRT_DECISION_READ, root=self.led.root, request_id=self.request_id)
        d = self.led.writer.append_decision(kind="promotion", subject=self.player["sha256"], consumed=consumed,
                                            verdict=verdict, rule=rule,
                                            rule_version=json.dumps(dict(config), sort_keys=True),
                                            request_id=self.request_id)
        self.led.writer.finish_request(self.request_id)
        return d
