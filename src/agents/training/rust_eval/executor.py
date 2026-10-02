"""THE EVAL CYCLE ON THE CORE (M5 Lane H) — one eval cycle's plan played on the Rust env core.

The eval callbacks write the cycle's PLAN exactly as today (``ShardedEvalPool.write_plan``: the bot
roster, the pool sentinels, the fixed / stable opponents, split into shard UNITS) and the manifest;
under ``--env-core rust`` they hand the plan to :meth:`RustEvalCore.run_cycle` instead of spawning
``main.eval_worker`` processes. It publishes one ``ShardResult`` per unit into the cycle's run dir —
the SAME raw additive record a Python worker publishes — so the collect, the aggregation, every
metric, the manifest's trace selection and the ELO inputs are the unchanged Python code.

HOW A CYCLE PLAYS (every resource DECLARED at startup — ``rust_eval.build``; nothing acquired here):

* The eval CORE (its own ``Core``, N envs, the eval ROUTE TABLE: every roster bot as an in-core bot
  route with per-episode streams, one POLICY route per declared sentinel slot and per fixed opponent,
  one filler route) and the eval SLOTS of the trainer's T2 service (the trainee's eval slot, the
  sentinel slots, the fixed opponents' slots — in the trainee's slot group where the architecture
  matches, so eval rides the rollout's compiled buckets). A cycle LOADS the trainee's current weights
  and each sentinel's snapshot into their slots (T2 ``load``: an in-place copy, parity-verified).
* UNITS → ENVS: a unit is played on ONE env, its games in plan order, one after another — exactly a
  Python worker's ``_play_unit`` at concurrency 1 — so the forensic QUOTA (per unit, ``per_shard``)
  is decided game by game in the same order: a game is CAPTURED iff the unit's quota is open when it
  starts, and its trace is KEPT iff its outcome's bucket is not yet full. An env with no unit left
  plays FILLER games (the filler route, p1 its highest legal action; nothing recorded) until the
  cycle ends; the next cycle's RESET drops them.
* Every game's teams, battle seed and bot streams come from the per-GAME seed rule
  (``rust_eval.seeds``), so a game's result is a function of (cycle seed, opponent, game index,
  weights) — not of the env, N, the thread count or the schedule.
* One host step: the trainee's rows (every real game with a p1 decision below the stall threshold —
  ``EvalRLPlayer`` forfeits BEFORE the forward at the threshold, so that decision is no row) and the
  policy opponents' rows go to T2 at ``Priority.EVAL``; ``drain()``; the trainee plays GREEDY
  (``argmax`` of the served log-probs — ``stochastic=False``), a sentinel greedy under
  ``eval_sentinel_greedy`` (the default regime) else a keyed sample at ``--self-play-temp``, a fixed
  opponent greedy; ``step()``; every ended game is read from ``Core::finished`` (winner, end turn,
  forfeit, the input log).
* A game's metrics are the Python worker's: WIN / LOSS / DRAW by ``classify_result`` (a tie and the
  250-turn timeout are DRAWs), the core's terminal reward, the end turn (``battle.turn``), and — for a
  CAPTURED game — the critic residuals ``δ = γ·V(s_t+1) − V(s_t)`` over its decisions (the recorder's
  formula; the per-decision reward is 0 before the terminal). A KEPT game is written as a CORE TRACE
  (``rust_eval.traces``) from its input log.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agents.training import keyed_draw as KD
from agents.training import mirrored_pairs as MP
from agents.training.eval_sharding import BOT, FIXED, SENTINEL, ShardResult
from agents.training.eval_sharding.units import game_range
from agents.training.rust_eval import seeds as SD
from agents.training.rust_eval.traces import write_core_trace
from agents.training.trace_result import DRAW, LOSS, WIN, classify_result, outcome_prefix


class EvalCoreError(RuntimeError):
    """A typed failure of a Rust eval cycle (a quarantined game, a replay that disagrees with the
    core, an undeclared opponent, a cycle that does not finish)."""


@dataclass(frozen=True)
class EvalTable:
    """The DECLARED eval opponent table (startup): the roster bots, the sentinel slots, the fixed
    opponents' slots (label → T2 slot), the trainee's eval slot and the filler bot."""

    bots: Tuple[str, ...]
    trainee_slot: int
    sentinel_slots: Tuple[int, ...] = ()
    fixed_slots: Tuple[Tuple[str, int], ...] = ()
    filler_bot: str = "random"

    def spec_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for i, b in enumerate(self.bots):
            rows.append({"kind": "bot", "bot": b, "seed": SD.bot_route_seed(b), "streams": "episode"})
        rows += [{"kind": "policy", "slot": int(s)} for s in self.sentinel_slots]
        rows += [{"kind": "policy", "slot": int(s)} for _l, s in self.fixed_slots]
        rows.append({"kind": "bot", "bot": self.filler_bot, "seed": SD.BOT_ROUTE_SEED + 999, "streams": "episode"})
        return rows

    @property
    def bot_route(self) -> Dict[str, int]:
        return {b: i for i, b in enumerate(self.bots)}

    def sentinel_route(self, i: int) -> int:
        return len(self.bots) + int(i)

    def fixed_route(self, label: str) -> int:
        labels = [lab for lab, _s in self.fixed_slots]
        return len(self.bots) + len(self.sentinel_slots) + labels.index(label)

    @property
    def filler_route(self) -> int:
        return len(self.bots) + len(self.sentinel_slots) + len(self.fixed_slots)


@dataclass
class _Unit:
    unit: Any
    games: List[int]
    quota: Any
    n_won: int = 0
    n_finished: int = 0
    sum_reward: float = 0.0
    n_episodes: int = 0
    sum_ep_len: float = 0.0
    td: List[float] = field(default_factory=list)
    #: the unit's games in plan order (``games`` is consumed as they are staged) and each ended game's
    #: half-points to the trainee — a MIRRORED plan scores its pairs from these at publish
    all_games: List[int] = field(default_factory=list)
    points: Dict[int, int] = field(default_factory=dict)
    kept: Dict[str, int] = field(default_factory=lambda: {WIN: 0, LOSS: 0, DRAW: 0})
    draws_seen: int = 0
    trace_idx: int = 0
    done: int = 0
    t0: Optional[float] = None
    t1: float = 0.0

    def quota_of(self, result: str) -> int:
        return {WIN: self.quota.win, LOSS: self.quota.loss, DRAW: self.quota.draw}[result]

    @property
    def quota_open(self) -> bool:
        return any(self.kept[r] < self.quota_of(r) for r in (WIN, LOSS, DRAW))


@dataclass
class _Game:
    u: Optional[_Unit]              # None = filler
    g: int = -1
    key: Tuple[int, str, int] = (0, "", 0)
    route: int = 0
    teams: Tuple[int, int] = (0, 0)
    #: MIRRORED pairs: the second game of a pair — the trainee pilots the team the opponent drew
    swapped: bool = False
    seed: Tuple[int, int, int, int] = (1, 2, 3, 4)
    capturing: bool = False
    rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    margins: List[float] = field(default_factory=list)
    logp: List[float] = field(default_factory=list)
    sample_seed: int = 0
    #: the policy opponent's decisions ``[dec_n, action, argmax, margin, turn]`` (only with a ``game_log``;
    #: the margin is the keyed draw's CDF margin for a sampled sentinel, else the top-2 log-prob margin)
    opp: List[List[Any]] = field(default_factory=list)
    #: the battle turn at each trainee decision (the gate's clock for ordering two streams' first flips)
    turns: List[int] = field(default_factory=list)


@dataclass
class CycleStats:
    step: int
    games: int = 0
    units: int = 0
    host_steps: int = 0
    trainee_decisions: int = 0
    p2_policy_decisions: int = 0
    traces: int = 0
    near_ties: int = 0
    seconds: Dict[str, float] = field(default_factory=lambda: {
        "load": 0.0, "stage": 0.0, "submit": 0.0, "drain": 0.0, "act": 0.0, "core": 0.0, "finish": 0.0,
        "trace": 0.0, "total": 0.0})
    lifecycle: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "games": self.games, "units": self.units, "host_steps": self.host_steps,
                "trainee_decisions": self.trainee_decisions, "p2_policy_decisions": self.p2_policy_decisions,
                "traces": self.traces, "near_ties": self.near_ties, "seconds": dict(self.seconds),
                "lifecycle": dict(self.lifecycle)}


#: A near-tie among the trainee's top-2 legal log-probs (counted, like the rollout's ``near_boundary``).
NEAR_TIE = 2e-5


class RustEvalCore:
    """See the module docs. Built once at startup by ``rust_eval.build.build_eval_core``."""

    def __init__(self, *, core: Any, svc: Any, lib: Any, table: EvalTable, team_table: Any, turn_limit: int,
                 trainee_builder: Any, opp_builder: Any, fixed_builders: Mapping[str, Any], names: Sequence[str],
                 commit: str = "", emit: Callable[[str], None] = print):
        self.core, self.svc, self.lib, self.table = core, svc, lib, table
        self.cols = core.cols
        self.n = int(core.n)
        self.team_table = team_table
        self.turn_limit = int(turn_limit)
        self.trainee_builder, self.opp_builder = trainee_builder, opp_builder
        self.fixed_builders = dict(fixed_builders)
        self.names = tuple(names)
        self.commit = commit
        self.emit = emit
        self.cycles = 0
        self.last_stats: Optional[CycleStats] = None

    # ------------------------------------------------------------------ the per-game inputs
    def _opp_builder_for(self, item: Any, sentinel_greedy: bool) -> Any:
        if item.kind == SENTINEL:
            return self.trainee_builder if sentinel_greedy else self.opp_builder
        if item.kind == FIXED:
            return self.fixed_builders.get(item.key, self.opp_builder)
        return self.opp_builder

    def _route_for(self, item: Any, sentinel_index: Mapping[str, int]) -> int:
        if item.kind == BOT:
            r = self.table.bot_route.get(item.key)
            if r is None:
                raise EvalCoreError(f"the eval core declares no bot route {item.key!r} (declared: {self.table.bots})")
            return r
        if item.kind == SENTINEL:
            return self.table.sentinel_route(sentinel_index[item.key])
        if item.kind == FIXED:
            if item.key not in [lab for lab, _s in self.table.fixed_slots]:
                raise EvalCoreError(f"the eval core declares no fixed opponent {item.key!r}")
            return self.table.fixed_route(item.key)
        raise EvalCoreError(f"unknown eval item kind {item.kind!r}")

    def _make_game(self, u: _Unit, g: int, cycle_seed: int, sentinel_greedy: bool,
                   sentinel_index: Mapping[str, int], mirrored: bool = False) -> _Game:
        item = u.unit.item
        # THE MIRRORED-PAIR RULE (`seeds.pair_game`): a pair's two games share the key of its first game
        # — the same two teams, battle seed, bot streams and sample seed — and the second is SWAPPED.
        key, swapped = SD.pair_game(cycle_seed, item.key, g, mirrored)
        t = self.team_table
        p1 = t.index(SD.draw_team(self.trainee_builder, key, SD.TRAINEE), f"eval {item.key} game {g} trainee")
        p2 = t.index(SD.draw_team(self._opp_builder_for(item, sentinel_greedy), key, SD.OPPONENT),
                     f"eval {item.key} game {g} opponent")
        return _Game(u=u, g=g, key=key, route=self._route_for(item, sentinel_index),
                     teams=(p2, p1) if swapped else (p1, p2), swapped=swapped,
                     seed=tuple(SD.battle_seed(key)), sample_seed=SD.sample_seed(key))  # type: ignore[arg-type]

    def _filler(self) -> _Game:
        return _Game(u=None, route=self.table.filler_route, teams=(0, min(1, len(self.team_table) - 1)))

    def _stage(self, e: int, gm: _Game) -> None:
        c = self.cols
        c["ep_team"][e] = gm.teams
        c["ep_seed"][e] = gm.seed
        c["ep_opp"][e] = gm.route

    # ------------------------------------------------------------------ one cycle
    def run_cycle(self, pool: Any, result_dir: str, *, step: int, trainee_policy: Any,
                  sentinel_policies: Mapping[str, Any], forensic_root: Optional[str], quota: Any, gamma: float,
                  sentinel_greedy: bool, self_play_temp: float, cycle_seed: int,
                  game_log: Optional[List[Dict[str, Any]]] = None) -> CycleStats:
        """Play every unit of ``pool`` (a ``ShardedEvalPool``) and publish one ``ShardResult`` per unit
        into ``result_dir``. ``sentinel_policies`` maps each SENTINEL item key to its loaded policy.
        ``forensic_root`` = ``<run>/eval_traces/step_<N>`` (None = capture nothing)."""
        from agents.inference.service.spec import Priority
        from agents.training.eval_callback import ForensicQuota, trace_filename_stem
        from utils.rust_env import ffi

        st = CycleStats(step=int(step))
        sec = st.seconds
        t_start = time.perf_counter()
        c, n, svc, tb = self.cols, self.n, self.svc, self.table

        # ---- the cycle's weights: LOADS into declared slots (never an acquisition)
        t0 = time.perf_counter()
        svc.load(int(tb.trainee_slot), trainee_policy, f"eval:trainee@{int(step)}")
        sentinel_items = [it for it in pool.items if it.kind == SENTINEL]
        if len(sentinel_items) > len(tb.sentinel_slots):
            raise EvalCoreError(f"{len(sentinel_items)} sentinels this cycle but {len(tb.sentinel_slots)} sentinel "
                                "slots were declared at startup (--n-sentinels)")
        sentinel_index = {it.key: i for i, it in enumerate(sentinel_items)}
        for it in sentinel_items:
            pol = sentinel_policies.get(it.key)
            if pol is None:
                raise EvalCoreError(f"no policy for sentinel {it.key!r}")
            svc.load(int(tb.sentinel_slots[sentinel_index[it.key]]), pol, f"eval:{it.key}@{it.path}")
        sec["load"] += time.perf_counter() - t0

        # ---- units → envs
        q = ForensicQuota.coerce(quota)
        mirrored = bool(getattr(pool, "mirrored", False))
        units: List[_Unit] = []
        for unit in pool.units:
            per = q.per_shard(pool.shard_count(unit.item.key))
            games = list(game_range(unit, pool.shard_games, mirrored))
            units.append(_Unit(unit=unit, games=list(games), all_games=games, quota=per))
        queue = list(units)
        env_unit: List[Optional[_Unit]] = [None] * n
        cur: List[_Game] = [self._filler() for _ in range(n)]
        staged: List[_Game] = [self._filler() for _ in range(n)]
        pending = len(units)
        slot_route_temp = float(self_play_temp)
        log_opp = game_log is not None

        def next_game(e: int) -> _Game:
            u = env_unit[e]
            if u is None or not u.games:
                u = queue.pop(0) if queue else None
                env_unit[e] = u
            if u is None:
                return self._filler()
            return self._make_game(u, u.games.pop(0), cycle_seed, sentinel_greedy, sentinel_index, mirrored)

        def promote(e: int) -> None:
            gm = staged[e]
            if gm.u is not None:
                gm.capturing = forensic_root is not None and gm.u.quota_open
                if gm.u.t0 is None:
                    gm.u.t0 = time.perf_counter()
            cur[e] = gm
            staged[e] = next_game(e)
            self._stage(e, staged[e])

        t0 = time.perf_counter()
        for e in range(n):
            staged[e] = next_game(e)
            self._stage(e, staged[e])
        self.core.reset()
        self._check_refused("RESET")
        last_ep = c["episode"].astype(np.int64).copy()
        for e in range(n):
            promote(e)
        sec["stage"] += time.perf_counter() - t0

        guard = 0
        limit = 4 * (sum(len(u.games) + 0 for u in units) + len(units)) * (2 * self.turn_limit + 16) + 1000
        while pending:
            guard += 1
            if guard > limit:
                raise EvalCoreError(f"the eval cycle at step {step} did not finish in {limit} host steps")
            # -- the trainee's rows and the policy opponents' rows, one drain
            t0 = time.perf_counter()
            need1 = np.flatnonzero(c["need"][:, 0] == 1)
            fwd = np.asarray([e for e in need1 if cur[e].u is not None and int(c["turn"][e]) < self.turn_limit],
                             dtype=np.int64)
            fwd_set = set(fwd.tolist())
            for e in need1:
                if int(e) not in fwd_set:                    # filler / the stall-forfeit decision: no forward
                    c["action"][e, 0] = int(np.flatnonzero(c["mask"][e, 0]).max())
            ticket = svc.submit(int(tb.trainee_slot), c["obs"][fwd, 0], c["mask"][fwd, 0], Priority.EVAL) \
                if fwd.size else None
            need2 = np.flatnonzero(c["need"][:, 1] == 1)
            p2t: List[Tuple[np.ndarray, Any]] = []
            if need2.size:
                slots = c["opp_slot"][need2]
                if bool((slots < 0).any()):
                    raise EvalCoreError("a p2 decision was exposed on a non-policy route")
                for s in np.unique(slots):
                    rows = need2[slots == s]
                    p2t.append((rows, svc.submit(int(s), c["obs"][rows, 1], c["mask"][rows, 1], Priority.EVAL)))
            t1 = time.perf_counter()
            svc.drain()
            t2 = time.perf_counter()
            if ticket is not None:
                lp, v, gr = ticket.host()
                lp, v, gr = np.array(lp, copy=True), np.array(v, copy=True), np.array(gr, copy=True)
                for j, e in enumerate(fwd.tolist()):
                    a = int(gr[j])
                    c["action"][e, 0] = a
                    gm = cur[e]
                    srt = np.sort(lp[j][np.isfinite(lp[j])])
                    margin = float(srt[-1] - srt[-2]) if srt.size > 1 else float("inf")
                    st.near_ties += int(margin < NEAR_TIE)
                    gm.actions.append(a)
                    gm.turns.append(int(c["turn"][e]))
                    gm.margins.append(margin)
                    gm.logp.append(float(lp[j, a]))
                    if gm.capturing:
                        gm.rows.append((c["obs"][e, 0].copy(), c["mask"][e, 0].astype(bool), lp[j].copy(),
                                        float(v[j]), a))
                st.trainee_decisions += int(fwd.size)
            for rows, t in p2t:
                lp2, _v2, gr2 = t.host()
                for j, e in enumerate(rows.tolist()):
                    gm = cur[e]
                    if gm.u is not None and gm.u.unit.item.kind == SENTINEL and not sentinel_greedy:
                        u = KD.keyed_uniforms(gm.sample_seed, KD.STREAM_OPPONENT, 0, 0, int(c["dec_n"][e, 1]))
                        a, m = KD.keyed_actions(np.asarray(lp2[j:j + 1]), u, slot_route_temp)
                        c["action"][e, 1] = int(a[0])
                        om = float(m[0])
                    else:
                        c["action"][e, 1] = int(gr2[j])
                        om = None
                    if log_opp and gm.u is not None:
                        if om is None:
                            srt = np.sort(np.asarray(lp2[j])[np.isfinite(lp2[j])])
                            om = float(srt[-1] - srt[-2]) if srt.size > 1 else float("inf")
                        gm.opp.append([int(c["dec_n"][e, 1]), int(c["action"][e, 1]), int(gr2[j]), om,
                                       int(c["turn"][e])])
                st.p2_policy_decisions += int(rows.size)
            t3 = time.perf_counter()
            self.core.step()
            t4 = time.perf_counter()
            self._check_refused("STEP")
            for f in self.core.finished():
                e = int(f["env"])
                gm = cur[e]
                if gm.u is None:
                    continue
                pending -= self._finish(gm, f, float(c["reward"][e]), step=step, gamma=gamma,
                                        forensic_root=forensic_root, result_dir=result_dir, pool=pool, st=st,
                                        game_log=game_log, trace_stem=trace_filename_stem, ffi=ffi,
                                        mirrored=mirrored)
            t5 = time.perf_counter()
            ep = c["episode"].astype(np.int64)
            for e in np.flatnonzero(ep != last_ep).tolist():
                promote(e)
            last_ep = ep.copy()
            t6 = time.perf_counter()
            st.host_steps += 1
            sec["submit"] += t1 - t0
            sec["drain"] += t2 - t1
            sec["act"] += t3 - t2
            sec["core"] += t4 - t3
            sec["finish"] += t5 - t4
            sec["stage"] += t6 - t5
        st.units = len(units)
        st.lifecycle = self.check_lifecycle()
        sec["total"] = time.perf_counter() - t_start
        self.cycles += 1
        self.last_stats = st
        return st

    def _check_refused(self, op: str) -> None:
        c = self.cols
        if bool((c["refused"] == 1).any()):
            bank = self.core.bank()
            raise EvalCoreError(f"eval core {op}: a battle was QUARANTINED (env(s) "
                                f"{np.flatnonzero(c['refused'] == 1).tolist()}; bank tail {bank[-1:] if bank else []})")

    def _finish(self, gm: _Game, f: Mapping[str, Any], reward: float, *, step: int, gamma: float,
                forensic_root: Optional[str], result_dir: str, pool: Any, st: CycleStats,
                game_log: Optional[List[Dict[str, Any]]], trace_stem: Any, ffi: Any, mirrored: bool = False) -> int:
        """Book one ended game into its unit; publish the unit when its last game ends (returns 1)."""
        u = gm.u
        assert u is not None
        item = u.unit.item
        turn = int(f["end_turn"])
        won, lost = int(f["winner"]) == 1, int(f["winner"]) == 2
        result, draw_kind = classify_result(won=won, lost=lost, finished=True, turn=turn,
                                            turn_cap=self.turn_limit)
        u.n_finished += 1
        u.n_won += int(won)
        u.sum_reward += reward
        u.n_episodes += 1
        u.sum_ep_len += turn
        if result == DRAW:
            u.draws_seen += 1
        u.points[gm.g] = MP.result_points(result)
        st.games += 1
        kept_path = None
        if gm.capturing and gm.rows:
            vals = [r[3] for r in gm.rows]
            u.td.extend(float(gamma) * vals[t + 1] - vals[t] for t in range(len(vals) - 1))
            if u.kept[result] < u.quota_of(result):
                t0 = time.perf_counter()
                bid = f"core-{int(step)}-{item.key}-g{gm.g}"
                tr = ffi.trace(self.lib, f["script"], commit=self.commit, label=bid)
                if int(tr["winner"]) != int(f["winner"]) or int(tr["turn"]) != turn:
                    raise EvalCoreError(f"{bid}: the trace replay ended (winner {tr['winner']}, turn {tr['turn']}) "
                                        f"where the core ended it (winner {f['winner']}, turn {turn})")
                u.trace_idx += 1
                prefix = os.path.join(forensic_root or "", item.key,
                                      trace_stem(outcome_prefix(result), f"s{u.unit.shard_index}_", u.trace_idx))
                write_core_trace(prefix, trace=tr, step=int(step), battle_id=bid, result=result, draw_kind=draw_kind,
                                 turns=turn, trainee_username=self.names[0],
                                 obs=np.stack([r[0] for r in gm.rows]), logp=np.stack([r[2] for r in gm.rows]),
                                 values=np.asarray(vals, dtype=np.float32),
                                 actions=np.asarray([r[4] for r in gm.rows]), masks=np.stack([r[1] for r in gm.rows]),
                                 cycle={"key": list(gm.key), "game": gm.g, "seed_rule": SD.SCHEMA,
                                        **({"mirrored_pair": MP.SCHEMA, "swapped": gm.swapped} if mirrored else {})},
                                 turn_limit=self.turn_limit)
                u.kept[result] += 1
                st.traces += 1
                kept_path = prefix
                st.seconds["trace"] += time.perf_counter() - t0
        if game_log is not None:
            game_log.append({"item": item.key, "kind": item.kind, "game": gm.g, "shard": u.unit.shard_index,
                             "winner": int(f["winner"]), "end_turn": turn, "forfeit": int(f["forfeit"]),
                             "reward": reward, "result": result, "draw_kind": draw_kind, "actions": list(gm.actions),
                             "margins": list(gm.margins), "logp": list(gm.logp), "teams": list(gm.teams), "seed": list(gm.seed),
                             **({"swapped": gm.swapped} if mirrored else {}),
                             "captured": gm.capturing, "trace": kept_path, "script": f["script"], "opp": list(gm.opp),
                             "turns": list(gm.turns)})
        u.done += 1
        u.t1 = time.perf_counter()
        if u.done < u.unit.n_games:
            return 0
        res = ShardResult(
            unit_id=u.unit.unit_id, item_key=item.key, worker_id=0, n_won=u.n_won, n_finished=u.n_finished,
            sum_reward=u.sum_reward, n_episodes=u.n_episodes, sum_ep_len=u.sum_ep_len,
            duration_sec=float(u.t1 - (u.t0 or u.t1)), td_residuals=list(u.td),
            traces_written=sum(u.kept.values()), traces_won=u.kept[WIN], n_drawn=u.draws_seen,
            traces_drawn=u.kept[DRAW],
            pair_counts=MP.pair_counts([u.points.get(g) for g in u.all_games]) if mirrored else None)
        pool.publish(result_dir, res)
        return 1

    # ------------------------------------------------------------------ the declared lifecycle
    def check_lifecycle(self) -> Dict[str, int]:
        after = dict(self.core.after_freeze())
        svc_c = {k: int(self.svc.counters.get(k, 0)) for k in
                 ("compiles_after_freeze", "captures_after_freeze", "cuda_segments_after_freeze")}
        bad = {k: v for k, v in after.items() if v and k != "PROC_SPAWNS_AFTER_FREEZE"}
        spawns = int(after.get("PROC_SPAWNS_AFTER_FREEZE", 0))
        if bad or spawns or any(svc_c.values()):
            from agents.training.rust_rollout.collector import LifecycleViolation

            raise LifecycleViolation(f"eval core: resources acquired after the freeze: core {bad}, "
                                     f"respawns {spawns}, T2 {svc_c}")
        return {**after, **svc_c}

    def close(self) -> None:
        try:
            self.core.close()
        except Exception:
            pass
