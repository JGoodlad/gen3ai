"""THE UNTAUGHT METER'S GAMES ON THE RUST EVAL CORE (poke-env retirement P2, ``gen3_untaught_rust_v1``).

``agents.training.untaught_meter.play_cells`` used to play every game through two poke-env ``RLPlayer`` s over the
in-process bridge, with the PYTHON encoder — while every trainee it scores trained on the Rust core's rows (survey
finding A-F2, ``designs/research_state/measurements/pokeenv_and_hotpath_survey_2026-10-06/``). It now plays here: the
Rust eval core (``rust_eval.executor.RustEvalCore.run_cycle``, the code the in-loop eval, the SPRT promotion and
``main.h2h`` play on), its observation rows (the training encoder) and the T2 inference service, on the CPU. The host
is ``main.h2h``'s engine pattern, REUSED rather than re-derived: the strict loader (``main.h2h.arch._load_host``), the
architecture declaration (``declare_engine``: at most two slot groups, one eval core per (player group, opponent
group)), the core-flag refusal and the team-pool cwd check.

THE REGIME, mapped from the poke-env path (what the meter measures is unchanged as far as the core allows):

* ONE CELL = one (ref, team) = ONE eval cycle. The PILOT plays p1 on its PINNED team (the cycle's ``trainee_builder``
  override: a one-team builder, so every game draws it); the fixed OPPONENT plays p2 from the FLAT pool builder
  (``rust_eval.build.eval_builders``' flat builder — the old ``PairedPool(TeamLoader().get_all_teams())``'s teams).
* ``stochastic=True`` (the default — the TRAINING regime): BOTH sides sample at T = 1.0 by the KEYED DRAW. The
  opponent is a SENTINEL item played under ``sentinel_greedy=False`` (``KD.STREAM_OPPONENT``, the opponent draws the
  flat pool); the pilot samples through the executor's opt-in ``trainee_temp`` (``KD.STREAM_TRAINEE``). Both streams
  key on the game's ``sample_seed``. ``stochastic=False`` (the EVAL regime): both sides GREEDY — the opponent is a
  FIXED item (greedy, flat pool when it pins no team: the regime a fixed cross-run opponent is evaluated in), the
  pilot the executor's default argmax.
* UNMIRRORED — ``mirrored=True`` stays REFUSED for a pinned-team meter (P13, ``play_cells``).
* CRN ACROSS REFS: the cycle seed is a function of (``--seed``, team index) ONLY (:func:`cycle_seed`), and the item
  key is constant, so game ``j`` of team ``ti`` has the same key — the same opponent team, battle seed and sampling
  stream starts — for every ref. Each game is a pure function of its key (``rust_eval.seeds``), so the draw is
  PREFIX-CONSISTENT (a ref at 12 games/team plays the first 12 of another's 200).
* THE TURN LIMIT: the core ends a game at the declared turn limit (``utils.rust_env.episode.stall_threshold``) as a
  DRAW, as training does. There is no TIMEOUT bucket any more: ``attempted == finished`` and those games are TIES
  (``Cell.turn_limit_draws`` counts them).
* REPRODUCIBLE: every game is a function of its key, never of the env count, the thread count or which shard
  process played the cell — EXCEPT that a decision within a rounding error of a tie (a near-tie log-prob margin, or
  a sampled draw near a CDF boundary) can flip with the forward's BATCH SHAPE (the env count). At a FIXED compute
  (``--n-envs`` / threads) a cell replays bit for bit, and sharding over teams cannot move a number.

TRANSPORT BOUNDARY. Every cell this module plays carries ``transport = "rust_eval"`` (:data:`TRANSPORT`); a banked
cell / artifact without it is ``"python_bridge"`` (pre-boundary). The readers refuse to put the two side by side
(``untaught_meter.cells_transport``).
"""
from __future__ import annotations

import hashlib
import math
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.training.untaught_meter import (TRANSPORT_RUST, Cell, MeterError, ResolvedRef, TeamSlice)

#: This module's schema tag (stamped on every artifact the meter writes from its games).
SCHEMA = "gen3_untaught_rust_v1"
TRANSPORT = TRANSPORT_RUST
#: The observation rows the games are played on: the Rust core's (the training encoder).
ENCODER = "rust"
#: The opponent's item key in each cycle's plan (constant: part of every game's seed key, so CRN across refs).
ITEM_KEY = "untaught"
CYCLE_SEED_TAG = "gen3_untaught_cycle_seed_v1"
#: The TRAINING regime's temperature, both sides.
SAMPLE_TEMP = 1.0


def cycle_seed(seed: int, team_index: int) -> int:
    """One CELL's per-game seed base: a hash of (``--seed``, team index) on this meter's own namespace — never of
    the ref, which is what makes every ref-vs-ref difference PAIRED on the same games."""
    d = hashlib.blake2b(f"{CYCLE_SEED_TAG}:{int(seed)}:{int(team_index)}".encode(), digest_size=8).digest()
    return int.from_bytes(d, "big") & ((1 << 62) - 1)


def regime_block(stochastic: bool, turn_limit: int) -> Dict[str, Any]:
    """The eval regime the cells were played in (stamped on every artifact)."""
    from agents.training.rust_eval import seeds as SD

    play = f"sample T={SAMPLE_TEMP} (keyed draw)" if stochastic else "greedy"
    return {"pilot_play": play, "opponent_play": play,
            "pilot_stream": "keyed STREAM_TRAINEE from the game's sample_seed" if stochastic else None,
            "opponent_stream": "keyed STREAM_OPPONENT from the game's sample_seed" if stochastic else None,
            "opponent_item_kind": "sentinel" if stochastic else "fixed",
            "pilot_team": "its PINNED team (one cell per (ref, team))",
            "opponent_team": "the flat pool builder (rust_eval.build.eval_builders)",
            "mirrored": False, "seat": "pilot p1, opponent p2",
            "turn_limit": int(turn_limit), "turn_limit_rule": "a game at the turn limit is a DRAW (a tie)",
            "seed_rule": SD.SCHEMA, "cycle_seed": f"{CYCLE_SEED_TAG}: blake2b(seed, team_index)",
            "item_key": ITEM_KEY}


def player_ref(r: ResolvedRef) -> Any:
    """A meter :class:`ResolvedRef` as ``main.h2h``'s ``PlayerRef`` (the content hash is computed here)."""
    import os

    from main.h2h.play import PlayerRef, file_sha256

    return PlayerRef(spec=r.ref, zip_path=r.zip_path, config_path=r.config_path,
                     run_dir=os.path.dirname(r.config_path) or r.run_base, run_base=r.run_base, rung=r.rung,
                     num_timesteps=r.num_timesteps, sha256=file_sha256(r.zip_path))


class _Sink(list):
    """The executor's ``game_log``, trimmed at the source (a row's input log is tens of KB)."""

    KEEP = ("game", "result", "draw_kind", "teams", "seed", "end_turn", "forfeit", "winner")

    def append(self, row: Mapping[str, Any]) -> None:  # type: ignore[override]
        slim = {k: row[k] for k in self.KEEP if k in row}
        slim["decisions"] = len(row.get("actions", ()))
        slim["sampled_off_argmax"] = (sum(int(a != g) for a, g in zip(row["actions"], row["argmax"]))
                                      if "argmax" in row else None)
        super().append(slim)


class UntaughtEngine:
    """ONE T2 service and one eval core per (player group, opponent group) for a set of refs against ONE opponent,
    built once (the declared lifecycle: every served resource acquired here; a cell only LOADS the ref's weights
    into its group's player slot). ``teams`` are the pinned pilot teams every cell may draw; they are in the core's
    team table from startup."""

    def __init__(self, refs: Sequence[Any], opponent: Any, teams: Sequence[TeamSlice], *, stochastic: bool,
                 compute: Any, host: Callable[[Any], Any],
                 emit: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)):
        import torch

        from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
        from agents.training.rust_eval.build import EvalDecl, build_eval_core, eval_builders
        from agents.training.rust_rollout.build import RustEnvDecl
        from main.h2h.arch import GROUP_NAMES, declare_engine
        from main.h2h.play import check_core_flags, check_team_pool
        from main.h2h.reveal import OFF_OFF
        from utils.rust_env import episode as EP
        from utils.rust_env.build import ensure_built
        from utils.teambuilder import Gen3Teambuilder

        if not str(compute.device).startswith("cpu"):
            raise MeterError(f"the untaught meter plays on the CPU only (device {compute.device!r} refused)")
        self.compute, self.emit, self.stochastic, self.opponent = compute, emit, bool(stochastic), opponent
        self._host = host
        self._off = OFF_OFF
        check_team_pool()
        for r in [*refs, opponent]:
            check_core_flags(r)
        ensure_built(compute.profile, emit=emit)
        self.turn_limit = EP.stall_threshold()
        self.regime = regime_block(self.stochastic, self.turn_limit)
        self.cores: Dict[Tuple[int, int, Tuple[str, str]], Any] = {}
        self._threads0 = torch.get_num_threads()
        try:
            if compute.torch_threads:
                torch.set_num_threads(int(compute.torch_threads))
            t0 = time.perf_counter()
            self.decl = declare_engine([(r, opponent) for r in refs], host)
            self.archs = self.decl.archs
            # the pinned pilot teams: one ONE-team builder each (its only yield is the team), and their union as
            # the core's declared trainee builder, so every pinned team is in the team table from startup
            texts: List[str] = []
            self.pinned: Dict[str, Any] = {}
            for t in teams:
                with open(t.path) as fh:
                    text = fh.read()
                b = Gen3Teambuilder([text])
                if len(b.packed_teams) != 1:
                    raise MeterError(f"team {t.key} ({t.path}): the teambuilder packed {len(b.packed_teams)} teams "
                                     "from it, not 1 (an invalid team is SKIPPED by the builder) — refusing")
                self.pinned[t.key] = b
                texts.append(text)
            union = Gen3Teambuilder(texts)
            _tb, flat, _fixed = eval_builders(None, [])
            self.flat_index: Dict[str, int] = {}
            for i, p in enumerate(flat.packed_teams):
                self.flat_index.setdefault(p, i)
            n = int(compute.n_envs)
            cdecl = RustEnvDecl(n_envs=n, threads=int(compute.threads), front=compute.front, profile=compute.profile,
                                device=compute.device, backend=compute.resolved_backend, turn_limit=self.turn_limit)
            buckets = cdecl.resolved_buckets
            groups = tuple(SlotGroupSpec(GROUP_NAMES[gi], len(self.decl.roles[gi]), host(src).policy)
                           for gi, src in enumerate(self.decl.sources))
            self.svc = InferenceService(ServiceSpec(
                groups=groups, device=compute.device, backend=compute.resolved_backend, buckets=buckets, lanes=1,
                max_rows_per_flush=max(1024, self.decl.n_slots * max(buckets), 4 * n))).startup()
            self.om = host(opponent)
            edecl = EvalDecl(n_envs=n, n_sentinels=1 if self.stochastic else 0,
                             fixed_labels=() if self.stochastic else (ITEM_KEY,))
            for gp, go, lv in self.decl.cores:
                if tuple(lv) != tuple(OFF_OFF):
                    raise MeterError(f"the untaught meter plays no oracle reveal (cell levels {lv})")
                self.cores[(gp, go, lv)] = build_eval_core(
                    edecl, collector_decl=cdecl, svc=self.svc,
                    extra_ids=[self.decl.slot_of(gp, "player"), self.decl.slot_of(go, "opponent")],
                    trainee_builder=union, opp_builder=flat, fixed_builders={}, turn_limit=self.turn_limit,
                    terminal=self.archs[gp].terminal,
                    fixed_policies=None if self.stochastic else {ITEM_KEY: self.om.policy}, emit=lambda _m: None)
            first = next(iter(self.cores.values()))
            # the BUILD STAMP (`utils.rust_env.stamp`): the proc front reads it at its handshake; the ffi front's
            # is the loaded library's own (the same build, checked against this tree at load)
            self.core_stamp = str(getattr(first.core, "stamp", "") or "") or first.lib.rust_env_stamp().decode()
            self.torch_version = torch.__version__
            self.startup_s = time.perf_counter() - t0
            emit(f"[untaught] Rust eval engine up in {self.startup_s:.1f}s: {len(refs)} ref(s) vs {opponent.id}; "
                 f"{n} envs, T2 {compute.resolved_backend} on {compute.device}; {len(groups)} slot group(s), "
                 f"{len(self.cores)} eval core(s); {'TRAINING regime (both sample T=1)' if self.stochastic else 'EVAL regime (both greedy)'}")
        except BaseException:
            self.close()
            raise

    def block(self) -> Dict[str, Any]:
        """The engine's provenance (stamped on the artifact beside the transport)."""
        return {"transport": TRANSPORT, "encoder": ENCODER, "schema": SCHEMA, "core_stamp": self.core_stamp,
                "torch": self.torch_version, "compute": self.compute.block(), "eval_regime": dict(self.regime),
                "engine": self.decl.block()}

    def play_cell(self, ref: Any, team: TeamSlice, games: int, seed: int,
                  sink: Optional[List[Dict[str, Any]]] = None) -> Cell:
        """Play ``games`` games of ``ref`` piloting ``team`` against the opponent — ONE eval cycle at
        :func:`cycle_seed` ``(seed, team.index)`` — and return the cell. ``sink`` receives the trimmed game rows."""
        from agents.training.eval_quota import ForensicQuota
        from agents.training.eval_sharding import FIXED, SENTINEL, EvalItem, ShardedEvalPool
        from agents.training.trace_result import DRAW, DRAW_TIMEOUT, LOSS, WIN

        if games < 1:
            raise MeterError("games_per_team must be positive")
        pm = self._host(ref)
        key = (self.decl.members[ref.sha256], self.decl.members[self.opponent.sha256], tuple(self._off))
        ev = self.cores[key]
        pinned = self.pinned[team.key]
        p1_index = ev.team_table.index(pinned.packed_teams[0], f"pinned team {team.key}")
        n = int(self.compute.n_envs)
        kind = SENTINEL if self.stochastic else FIXED
        item = EvalItem(ITEM_KEY, kind, int(games), path=self.opponent.zip_path,
                        step=0 if self.stochastic else None, config_path=self.opponent.config_path)
        pool = ShardedEvalPool([item], max(1, math.ceil(games / n)), step=0, mirrored=False)
        cs = cycle_seed(seed, team.index)
        rows = _Sink()
        run_dir = tempfile.mkdtemp(prefix="untaught_cycle_")
        try:
            pool.write_plan(run_dir)
            ev.run_cycle(pool, run_dir, step=0, trainee_policy=pm.policy,
                         sentinel_policies={ITEM_KEY: self.om.policy} if self.stochastic else {},
                         forensic_root=None, quota=ForensicQuota(0, 0, 0), gamma=float(getattr(pm, "gamma", 1.0)),
                         sentinel_greedy=False, self_play_temp=SAMPLE_TEMP, cycle_seed=cs, game_log=rows,
                         trainee_temp=SAMPLE_TEMP if self.stochastic else None, trainee_builder=pinned)
            merged, missing = pool.collect(run_dir)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)
        self.verify_slots(ev, pm, ref, f"after the cell {ref.id} x {team.key}")
        if missing:
            raise MeterError(f"the eval core published no result for {missing}")
        by_g = sorted(rows, key=lambda r: int(r["game"]))
        if [int(r["game"]) for r in by_g] != list(range(games)):
            raise MeterError(f"cell {ref.id} x {team.key}: the game log holds games "
                             f"{[r['game'] for r in by_g][:8]}..., not 0..{games - 1}")
        cell = Cell(transport=TRANSPORT, turn_limit_draws=0)
        for r in by_g:
            if int(r["teams"][0]) != p1_index:
                raise MeterError(f"cell {ref.id} x {team.key} game {r['game']}: the pilot played team table index "
                                 f"{r['teams'][0]}, not its pinned team's {p1_index}")
            res = r["result"]
            cell.attempted += 1
            cell.finished += 1
            cell.wins += int(res == WIN)
            cell.ties += int(res == DRAW)
            cell.losses += int(res == LOSS)
            if res == DRAW and r.get("draw_kind") == DRAW_TIMEOUT:
                cell.turn_limit_draws = int(cell.turn_limit_draws or 0) + 1
            packed = ev.team_table.teams[int(r["teams"][1])]
            idx = self.flat_index.get(packed)
            if idx is None:
                raise MeterError(f"cell {ref.id} x {team.key} game {r['game']}: the opponent's team is not in the flat pool")
            cell.opp_teams.append(int(idx))
        if tuple(merged["counts"][ITEM_KEY]) != (cell.wins, cell.finished):
            raise MeterError(f"cell {ref.id} x {team.key}: the executor's count {merged['counts'][ITEM_KEY]} != the "
                             f"game log's {(cell.wins, cell.finished)}")
        if sink is not None:
            sink.extend(by_g)
        return cell

    def verify_slots(self, ev: Any, pm: Any, ref: Any, where: str) -> None:
        """BIT-EXACT: the cell's player slot holds THIS ref and the opponent's slot the opponent
        (``SlotGroup.verify_copy``, ``gen3_slot_copy_verify_v1``) — the games just played were served by the cell's
        weights, not a previous cell's."""
        from agents.inference.service.slots import served_state_dict
        from main.h2h.arch import GROUP_NAMES

        tb = ev.table
        opp_slot = int(tb.sentinel_slots[0]) if self.stochastic else int(tb.fixed_slots[0][1])
        for slot, model, who in ((int(tb.trainee_slot), pm, ref.id), (opp_slot, self.om, self.opponent.id)):
            gi, i = self.decl.local(slot)
            self.svc.groups[gi].verify_copy(i, served_state_dict(model.policy),
                                            f"{where}: slot {slot} ({GROUP_NAMES[gi]}[{i}], {who})")

    def close(self) -> None:
        import torch

        try:
            for ev in self.cores.values():
                ev.close()
        finally:
            torch.set_num_threads(self._threads0)


def _engine_groups(refs: Sequence[Any], opponent: Any, host: Callable[[Any], Any]) -> List[List[Any]]:
    """The refs split into runs of consecutive refs ONE engine can serve (at most two architectures with the
    opponent's): the whole list when it fits — the common case — else greedily, in order."""
    from main.h2h.arch import CellArchMismatch, declare_engine

    try:
        declare_engine([(r, opponent) for r in refs], host)
        return [list(refs)]
    except CellArchMismatch:
        pass
    out: List[List[Any]] = []
    cur: List[Any] = []
    for r in refs:
        try:
            declare_engine([(x, opponent) for x in cur + [r]], host)
            cur.append(r)
        except CellArchMismatch:
            if not cur:
                raise
            out.append(cur)
            cur = [r]
    if cur:
        out.append(cur)
    return out


def play_cells_rust(refs: Sequence[ResolvedRef], teams: Sequence[TeamSlice], opponent: ResolvedRef, *,
                    games_per_team: int, seed: int, stochastic: bool, compute: Any, progress=None,
                    game_log: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]] = None,
                    info: Optional[Dict[str, Any]] = None,
                    emit: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)
                    ) -> Dict[str, Dict[str, Cell]]:
    """Every (ref x team) cell on the Rust eval core. ``game_log`` (optional) receives each cell's trimmed game rows
    under ``(ref label, team key)``; ``info`` (optional) receives the engines' provenance (:meth:`UntaughtEngine.block`)."""
    from main.h2h.arch import _load_host

    hosts: Dict[str, Any] = {}

    def host(ref: Any) -> Any:
        m = hosts.get(ref.sha256)
        if m is None:
            m = hosts[ref.sha256] = _load_host(ref)
        return m

    opp = player_ref(opponent)
    players = [(r, player_ref(r)) for r in refs]
    out: Dict[str, Dict[str, Cell]] = {}
    blocks: List[Dict[str, Any]] = []
    if not players:
        return out
    for group in _engine_groups([p for _r, p in players], opp, host):
        eng = UntaughtEngine(group, opp, teams, stochastic=stochastic, compute=compute, host=host, emit=emit)
        try:
            blocks.append(eng.block())
            shas = {p.sha256 for p in group}
            for r, p in players:
                if p.sha256 not in shas or r.label in out:
                    continue
                out[r.label] = {}
                for team in teams:
                    sink: Optional[List[Dict[str, Any]]] = [] if game_log is not None else None
                    cell = eng.play_cell(p, team, games_per_team, seed, sink=sink)
                    out[r.label][team.key] = cell
                    if game_log is not None and sink is not None:
                        game_log[(r.label, team.key)] = sink
                    if progress is not None:
                        progress(r.label, team.key, cell)
        finally:
            eng.close()
    if info is not None:
        info.update(blocks[0])
        info["engines"] = len(blocks)
        stamps = sorted({b["core_stamp"] for b in blocks})
        if len(stamps) > 1:
            raise MeterError(f"the engines ran different core builds {stamps}")
    return out
