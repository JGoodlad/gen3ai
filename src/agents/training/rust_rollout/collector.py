"""THE HOST LOOP — the trainer's rollout on the Rust env core (M5 Lane G; package docstring).

One host STEP:

1. p2's POLICY rows are submitted to their T2 slots (Lane E's ``PolicyOpponentServer.submit``); an
   EXTERNAL route's p2 (a harness only) is answered by the caller's function;
2. the trainee's rows (every env with ``need[:, 0]``) are submitted to the trainee's T2 slot — the SAME
   flush serves both (T2 lanes replay the slots concurrently);
3. ONE ``flush``;
4. p2's actions (greedy, or the route's sample); the trainee's action is the KEYED DRAW
   (``keyed_draw``: key = (run seed, trainee stream, env, episode, dec_n)) from the served log-probs,
   its behaviour log-prob μ(a|s) and V(s) are kept, and the row is written into the arena WITH the
   policy version T2 served;
5. ``core.step()``;
6. every env whose game ended closes its game (reward, the raw flags, the learner's re-label, the
   outcome, a tie vs the stall forfeit — F-LD-4), and every env whose ``episode`` MOVED is re-staged:
   its next route (Lane E's ``after_op``, F-LE-4), teams and seed (``teams.TeamStager``).

``collect`` repeats host steps until the trigger fires, firing the SB3 callbacks once per ``n_envs``
trainee decisions (a "vec step", so every step-counted cadence — the checkpoint's ``save_freq`` — keeps
its meaning), then fills the learner's buffer. ``after_update`` LOADS the new weights into T2 (a
declared in-place load, parity-verified) and bumps the version; with PER-GAME VERSION PINNING
(declared, OFF by default) the new weights go to a FREE trainee slot and every game in progress
keeps the slot — the version — it started with.

THE DECLARED LIFECYCLE: ``startup`` (the caller) builds the core (``Core::new`` validates every team
of the table by use), T2 (compiles, captures, parity) and the arena; then nothing is acquired.
``check_lifecycle`` reads every ``*_after_freeze`` counter (core + T2) after each update and raises
`LifecycleViolation` on a non-zero one; a process-front-end RESPAWN is the one counted steady-state
event, allowed up to a DECLARED budget (F-LB-1) — its games are cut, the core is RESET.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agents.training import keyed_draw as KD
from agents.training.rust_rollout import store as S
from agents.training.rust_rollout.teams import TeamStager
from agents.training.rust_rollout.trigger import WindowTrigger


class LifecycleViolation(S.CollectorError):
    """A resource acquired after the freeze (a non-zero ``*_after_freeze`` counter), or a respawn
    beyond the declared budget."""


class SlotCapacityExceeded(S.CollectorError):
    """Per-game version pinning found no trainee slot free of an in-progress game."""


@dataclass
class CollectorConfig:
    n_envs: int
    trigger: Any                          # SampleTrigger | WindowTrigger
    gamma: float
    gae_lambda: float
    run_seed: int
    turn_limit: int
    capacity: int
    max_game_rows: int
    label_keys: Tuple[str, ...]
    version_pinning: bool = False
    respawn_budget: int = 0
    victory_value: float = 1.0


@dataclass
class CollectorStats:
    host_steps: int = 0
    trainee_decisions: int = 0
    p2_policy_decisions: int = 0
    p2_external_decisions: int = 0
    games_ended: int = 0
    games_cut: int = 0
    quarantines: int = 0
    respawns: int = 0
    updates: int = 0
    near_boundary: int = 0
    seconds: Dict[str, float] = field(default_factory=lambda: {
        "submit": 0.0, "flush": 0.0, "draw": 0.0, "write": 0.0, "core": 0.0, "post": 0.0, "fill": 0.0})


class RustCollector:
    """See the module docstring. ``core`` is a started ``FfiCore`` / ``ProcCore`` whose spec declared
    ``plan.spec_rows(...)`` as its opponents; ``svc`` a STARTED inference service; ``trainee_slots``
    the global T2 slots of the trainee (1, or K under version pinning); ``opponents`` Lane E's
    ``RustEnvOpponents``; ``server`` its ``PolicyOpponentServer`` (None when no policy route);
    ``external_p2(cols, envs) -> actions`` answers p2 on an EXTERNAL route (harnesses only)."""

    def __init__(self, cfg: CollectorConfig, *, core: Any, obs_space: Any, svc: Any, trainee_slots: Sequence[int],
                 opponents: Any, server: Any, stager: TeamStager,
                 external_p2: Optional[Callable[[Mapping[str, np.ndarray], np.ndarray], np.ndarray]] = None,
                 load_trainee: Optional[Callable[[int, Any, str], None]] = None):
        from agents.training.reward_term_stats import RewardTermAccumulator

        self.cfg = cfg
        self.n = int(cfg.n_envs)
        self.core, self.svc = core, svc
        self.cols = core.cols
        if int(core.n) != self.n:
            raise S.CollectorError(f"the core declares {core.n} envs, the collector {self.n}")
        self.trainee_slots = [int(s) for s in trainee_slots]
        if not self.trainee_slots:
            raise S.CollectorError("the collector needs at least one trainee slot")
        if cfg.version_pinning and len(self.trainee_slots) < 2:
            raise S.CollectorError("per-game version pinning needs >= 2 trainee slots (the current version "
                                   "and the one games in progress still play)")
        self.opponents, self.server, self.stager = opponents, server, stager
        self.external_p2 = external_p2
        self._load_trainee = load_trainee
        self.obs_space = obs_space
        self.sources = S.obs_key_sources(obs_space, cfg.label_keys)
        self.store = S.RowStore(obs_space, cfg.capacity)
        self.log = S.GameLog(self.store, self.n, mode=cfg.trigger.mode, gamma=cfg.gamma,
                             gae_lambda=cfg.gae_lambda, max_game_rows=cfg.max_game_rows)
        self.version = 0
        self.slot_version = {s: 0 for s in self.trainee_slots}
        self.current_slot = self.trainee_slots[0]
        self.env_slot = np.full(self.n, self.current_slot, dtype=np.int64)
        self.last_turn = np.zeros(self.n, dtype=np.int64)
        self.cur_class = np.zeros(self.n, dtype=np.int64)
        self.cur_episode = np.zeros(self.n, dtype=np.int64)
        self.ep_start_time = np.zeros(self.n, dtype=np.float64)
        self.p2_policy_by_env = np.zeros(self.n, dtype=np.int64)
        self.exploiter_games = np.zeros(self.n, dtype=np.int64)
        self.exploiter_wins = np.zeros(self.n, dtype=np.float64)
        self._klass = np.array([r.klass for r in opponents.routes], dtype=np.int64)
        self.stats = CollectorStats()
        self._pending_decisions = 0
        self._infos: List[dict] = []
        self.last_fill: Optional[S.FillReport] = None
        self.last_versions: Optional[np.ndarray] = None
        self.hooks: List[Any] = []            # harness recorders: .episode_started / .p2 / .trainee
        #: the `reward/` export's accumulator (`drain_reward_terms`): the terminal is the ONE term
        self.reward_terms = RewardTermAccumulator(("win_loss",))
        self._t_start = time.time()
        self._started = False

    # ------------------------------------------------------------------ startup
    def start(self) -> None:
        """Stage every env's first episode, RESET, stage the second (declared-lifecycle startup)."""
        c = self.cols
        self.opponents.stage_all(c)
        self.stager.stage(c, range(self.n), self.opponents.staged, np.zeros(self.n, dtype=np.int64))
        self.core.reset()
        self._restage(self.opponents.after_op(c))
        self._started = True

    def _restage(self, moved: np.ndarray) -> None:
        if moved.size == 0:
            return
        c = self.cols
        self.stager.started(moved)
        self.cur_class[moved] = self._klass[c["opp_route"][moved].astype(np.int64)]
        self.cur_episode[moved] = c["episode"][moved]
        self.ep_start_time[moved] = time.time()
        self.env_slot[moved] = self.current_slot
        for h in self.hooks:
            h.episode_started(self, moved)
        self.stager.stage(c, moved, self.opponents.staged[moved], c["episode"][moved].astype(np.int64) + 1)

    # ------------------------------------------------------------------ one host step
    def host_step(self) -> int:
        """One host step (module docs). Returns the trainee decisions it played."""
        c = self.cols
        st = self.stats
        sec = st.seconds
        t0 = time.perf_counter()
        pend = self.server.submit(c) if self.server is not None else None
        if self.external_p2 is not None:
            ext = np.flatnonzero((c["need"][:, 1] == 1) & (c["opp_slot"] < 0))
            if ext.size:
                c["action"][ext, 1] = np.asarray(self.external_p2(c, ext), dtype=np.int32)
                st.p2_external_decisions += int(ext.size)
        envs1 = np.flatnonzero(c["need"][:, 0] == 1)
        tickets: List[Tuple[np.ndarray, Any]] = []
        if envs1.size:
            from agents.inference.service.spec import Priority

            slots = self.env_slot[envs1]
            for s in np.unique(slots):
                rows = envs1[slots == s]
                tickets.append((rows, self.svc.submit(int(s), c["obs"][rows, 0], c["mask"][rows, 0],
                                                      Priority.ROLLOUT)))
        t1 = time.perf_counter()
        if pend is not None or tickets:
            self.svc.flush()
        t2 = time.perf_counter()
        if pend is not None:
            served = self.server.complete(c, pend)
            self.p2_policy_by_env[served] += 1
            st.p2_policy_decisions += int(served.size)
        k = int(envs1.size)
        if k:
            logp = np.empty((k, c["mask"].shape[2]), dtype=np.float32)
            value = np.empty(k, dtype=np.float32)
            version = np.empty(k, dtype=np.int64)
            pos = {int(e): j for j, e in enumerate(envs1.tolist())}
            for rows, t in tickets:
                lp, v, _g = t.host()
                at = np.fromiter((pos[int(e)] for e in rows), dtype=np.int64, count=rows.size)
                logp[at] = lp
                value[at] = v
                version[at] = self.slot_version[int(self.env_slot[rows[0]])]
            t3 = time.perf_counter()
            u = KD.keyed_uniforms(self.cfg.run_seed, KD.STREAM_TRAINEE, envs1, c["episode"][envs1],
                                  c["dec_n"][envs1, 0])
            act, margin = KD.keyed_actions(logp, u)
            blogp = logp[np.arange(k), act]
            st.near_boundary += int((margin < 1e-6).sum())
            t4 = time.perf_counter()
            slots_ = self.store.alloc(k)
            S.write_rows(self.store, slots_, c, envs1, label_keys=self.cfg.label_keys,
                         opp_class=self.cur_class[envs1], actions=act, logp=blogp, values=value,
                         version=version, u=u, margin=margin,
                         starts=(c["dec_n"][envs1, 0] == 0).astype(np.float32))
            self.log.add(envs1, slots_)
            self.last_turn[envs1] = c["turn"][envs1]
            c["action"][envs1, 0] = act
            for h in self.hooks:
                h.trainee(self, envs1, logp, act, blogp, value, u, slots_)
            t5 = time.perf_counter()
            sec["draw"] += t4 - t3
            sec["write"] += t5 - t4
        for h in self.hooks:
            h.p2(self)
        t6 = time.perf_counter()
        self.core.step()
        t7 = time.perf_counter()
        self._after_step()
        t8 = time.perf_counter()
        sec["submit"] += t1 - t0
        sec["flush"] += t2 - t1
        sec["core"] += t7 - t6
        sec["post"] += t8 - t7
        st.host_steps += 1
        st.trainee_decisions += k
        return k

    def _after_step(self) -> None:
        c = self.cols
        done = np.flatnonzero(c["done"] == 1)
        for e in done.tolist():
            term, trunc = bool(c["terminated"][e]), bool(c["truncated"][e])
            if not (term or trunc):                          # quarantined IN PROGRESS (refused = 1)
                self.stats.quarantines += 1
                self._observe_rows(self.log.cut(e), 0.0)
                self.stats.games_cut += 1
                continue
            self._end_game(e, term, trunc)
        self._restage(self.opponents.after_op(c))

    def _end_game(self, e: int, term: bool, trunc: bool) -> None:
        c = self.cols
        reward = float(c["reward"][e])
        won = 1.0 if reward > 0.0 else 0.0
        p1_forfeit = bool(trunc and self.last_turn[e] >= self.cfg.turn_limit)
        draw = bool(trunc and not won and not p1_forfeit)
        klass = int(self.cur_class[e])
        g = self.log.end(e, reward=reward, outcome=won, draw=draw, forfeit=p1_forfeit, terminated=term,
                         truncated=trunc, episode=int(self.cur_episode[e]), opp_class=klass)
        if g is None:
            return
        self.stats.games_ended += 1
        self._observe_rows(g.length, reward)
        self.stager.record_outcome(e, won, klass)
        from agents.training.rust_env_opponents import OPP_CLASS_EXPLOITER

        if klass == OPP_CLASS_EXPLOITER:
            self.exploiter_games[e] += 1
            self.exploiter_wins[e] += won
        self._infos.append({
            "episode": {"r": reward, "l": g.length, "t": round(time.time() - self._t_start, 6)},
            "win_outcome": won, "win_draw": float(draw), "opponent_class": klass,
            "rust_env": {"env": e, "episode": g.episode, "terminated": term, "truncated": trunc,
                         "forfeit": p1_forfeit, "versions": g.versions}})

    def _observe_rows(self, n_rows: int, terminal: float) -> None:
        """The `reward/` term export, per trainee DECISION (``process_turn_reward``'s cadence): a game's
        ``n_rows`` decisions carry 0 but the last, which carries the terminal."""
        if n_rows <= 0:
            return
        for _ in range(n_rows - 1):
            self.reward_terms.observe(_ZERO, 0.0)
        self.reward_terms.observe(_Terminal(terminal), terminal)

    # ------------------------------------------------------------------ the PPO hook
    def ready(self) -> bool:
        trig = self.cfg.trigger
        if isinstance(trig, WindowTrigger):
            return self.log.window_ready(trig.n_steps)
        return trig.ready(self.log.completed_rows)

    def collect(self, model: Any, callback: Any, rollout_buffer: Any) -> bool:
        """``collect_rollouts``' contract for the learner (module docs). Returns False iff a callback
        stopped training."""
        if not self._started:
            raise S.CollectorError("collect before start(): the declared startup has not run")
        snap = self._snapshot()
        callback.on_rollout_start()
        while not self.ready():
            k = self.host_step()
            model.num_timesteps += k
            self._pending_decisions += k
            while self._pending_decisions >= self.n:
                self._pending_decisions -= self.n
                infos, self._infos = self._infos, []
                dones = np.ones(len(infos), dtype=bool)
                callback.update_locals({"infos": infos, "dones": dones, "env": getattr(model, "env", None),
                                        "rollout_buffer": None, "n_steps": None})
                if not callback.on_step():
                    return False
                model._update_info_buffer(infos, dones)
        t0 = time.perf_counter()
        self._ensure_buffer(model)
        trig = self.cfg.trigger
        if isinstance(trig, WindowTrigger):
            rep, versions = S.fill_window(model.rollout_buffer, self.log, trig.n_steps,
                                          current_version=self.version)
        else:
            rep, versions = S.fill_complete(model.rollout_buffer, self.log, trig.take(),
                                            current_version=self.version)
        self.stats.seconds["fill"] += time.perf_counter() - t0
        self.last_fill, self.last_versions = rep, versions
        model._rust_fill = rep
        model._rust_row_versions = versions
        model._rust_version = self.version
        model._win_prob_terminal_outcome = None
        self._record(model, snap)
        callback.on_rollout_end()
        return True

    def _snapshot(self) -> Dict[str, float]:
        st = self.stats
        out = {k: float(getattr(st, k)) for k in ("host_steps", "trainee_decisions", "p2_policy_decisions",
                                                  "p2_external_decisions", "games_ended", "games_cut",
                                                  "quarantines", "near_boundary")}
        out.update({f"s_{k}": v for k, v in st.seconds.items()})
        out["t"] = time.perf_counter()
        return out

    def _record(self, model: Any, snap: Dict[str, float]) -> None:
        """``rust_env/*``: this rollout's collector read (per-phase ms per host step, decisions/s, games)."""
        logger = getattr(model, "_logger", None)
        if logger is None:
            return
        now = self._snapshot()
        d = {k: now[k] - snap[k] for k in snap}
        steps = max(1.0, d["host_steps"])
        logger.record("rust_env/host_steps", d["host_steps"])
        logger.record("rust_env/trainee_decisions_per_s", d["trainee_decisions"] / max(1e-9, d["t"]))
        logger.record("rust_env/trainee_rows_per_host_step", d["trainee_decisions"] / steps)
        for k in ("submit", "flush", "draw", "write", "core", "post"):
            logger.record(f"rust_env/{k}_ms_per_host_step", 1000.0 * d[f"s_{k}"] / steps)
        logger.record("rust_env/fill_ms", 1000.0 * d["s_fill"])
        for k in ("p2_policy_decisions", "games_ended", "games_cut", "quarantines", "near_boundary"):
            logger.record(f"rust_env/{k}", d[k])
        if self.server is not None:
            logger.record("rust_env/p2_near_boundary_total", float(getattr(self.server, "near_boundary", 0)))
        logger.record("rust_env/version", float(self.version))
        logger.record("rust_env/arena_live_rows", float(self.store.live_count))

    def _ensure_buffer(self, model: Any) -> None:
        """The learner's buffer is ``[n_steps, n_envs]`` for THIS update (complete-game: target / n_envs;
        the adaptive-batch hook may have moved the target). A new size is a new sb3 buffer object —
        sb3 itself re-allocates every array at every rollout's ``reset()``."""
        trig = self.cfg.trigger
        n_steps = trig.n_steps if isinstance(trig, WindowTrigger) else trig.take() // self.n
        buf = model.rollout_buffer
        if int(buf.buffer_size) == n_steps and int(buf.n_envs) == self.n:
            return
        model.n_steps = n_steps
        model.rollout_buffer = type(buf)(n_steps, buf.observation_space, buf.action_space, device=buf.device,
                                         gamma=buf.gamma, gae_lambda=buf.gae_lambda, n_envs=self.n)

    # ------------------------------------------------------------------ after an update
    def after_update(self, model: Any) -> None:
        """The learner stepped: LOAD its weights into T2 (the current slot, or — pinning — a free one)
        and bump the version. Then the lifecycle check."""
        self.version += 1
        self.stats.updates += 1
        target = self.current_slot
        if self.cfg.version_pinning:
            in_use = set(self.env_slot[self._live_envs()].tolist())
            free = [s for s in self.trainee_slots if s not in in_use]
            if not free:
                raise SlotCapacityExceeded(
                    f"per-game version pinning: every trainee slot {self.trainee_slots} holds a version a game "
                    "in progress still plays; declare more trainee slots")
            target = free[0]
        if self._load_trainee is not None:
            self._load_trainee(target, model.policy, f"trainee:v{self.version}")
        else:
            self.svc.load(target, model.policy, f"trainee:v{self.version}")
        self.slot_version[target] = self.version
        self.current_slot = target
        if self.cfg.version_pinning:
            # a game that has STARTED but not yet played a row pins to the new version
            fresh = np.asarray([len(r) == 0 for r in self.log.cur])
            self.env_slot[fresh] = target
        else:
            self.env_slot[:] = target
        self.check_lifecycle()

    def _live_envs(self) -> np.ndarray:
        return np.flatnonzero(np.asarray([len(r) > 0 for r in self.log.cur]))

    def check_lifecycle(self) -> Dict[str, int]:
        after = dict(self.core.after_freeze())
        spawns = int(after.pop("PROC_SPAWNS_AFTER_FREEZE", 0))
        bad = {k: v for k, v in after.items() if v}
        svc_bad = {k: int(self.svc.counters.get(k, 0)) for k in
                   ("compiles_after_freeze", "captures_after_freeze", "cuda_segments_after_freeze")
                   if int(self.svc.counters.get(k, 0))}
        if bad or svc_bad:
            raise LifecycleViolation(f"resources acquired after the freeze: core {bad}, T2 {svc_bad}")
        if spawns > self.cfg.respawn_budget:
            raise LifecycleViolation(f"{spawns} core respawns exceed the declared budget {self.cfg.respawn_budget}")
        return {**after, "PROC_SPAWNS_AFTER_FREEZE": spawns, **{k: int(self.svc.counters.get(k, 0)) for k in
                ("compiles_after_freeze", "captures_after_freeze", "cuda_segments_after_freeze")}}

    # ------------------------------------------------------------------ a dead core child
    def recover_respawn(self) -> int:
        """After ``CoreProcessDied`` with auto-respawn: every game in progress is CUT (no outcome, F-LB-1),
        the staged inputs survive in the mapping (F-LB-2), and the core is RESET. Returns rows cut."""
        self.stats.respawns += 1
        cut = 0
        for e in range(self.n):
            n0 = len(self.log.cur[e])
            if n0:
                self.log.cut(e)
                self.stats.games_cut += 1
                cut += n0
        self.core.reset()
        self._restage(self.opponents.after_op(self.cols))
        return cut

    def close(self) -> None:
        try:
            self.core.close()
        except Exception:
            pass


class _Terminal:
    """A ``RewardBreakdown``-shaped view of one terminal reward (``win_loss``), for the term export."""

    __slots__ = ("win_loss",)

    def __init__(self, r: float):
        self.win_loss = float(r)


_ZERO = _Terminal(0.0)
