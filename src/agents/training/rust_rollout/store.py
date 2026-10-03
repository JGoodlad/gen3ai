"""THE ROW ARENA and the two fills of the learner's buffer (M5 Lane G; package docstring).

Every trainee decision the collector plays becomes ONE row in a preallocated arena (declared at
startup: ``capacity`` rows of every learner obs key + the PPO fields + per-row metadata), and stays
there until an update consumes it. A row carries what PPO needs AND what staleness needs:

* the observation dict (``observation``, ``action_mask``, every core label column, ``opp_class``),
* ``action``, the BEHAVIOUR log-prob μ(a|s) the action was drawn under, ``value`` V(s) at play time,
* the policy ``version`` that played it (the update count of the weights T2 served),
* the ``reward`` (terminal alone), the episode-start flag, and — once its game ends — the game's
  outcome, and (complete-game mode) its GAE advantage and return.

TWO FILLS of the model's own buffer (``agents/training/rollout_buffer.RolloutBuffer``, ``[n_steps, n_envs]``,
the layout sb3-contrib's ``MaskableDictRolloutBuffer`` had, so ``train()`` and every buffer reader are untouched):

* ``fill_window`` — TODAY'S SCHEDULE: column ``i`` = env ``i``'s next ``n_steps`` rows in play order;
  GAE by the buffer's own ``compute_returns_and_advantage`` (sb3's arithmetic) with the bootstrap V of each env's
  NEXT row; the win labels by ``win_prob_callback.backfill_terminal_labels`` (a game unfinished at the edge
  gets ``win_mask`` 0). It exists so the collector is proven against today's path (the rollout-level
  slice N) before it changes the schedule. Rows beyond a column's ``n_steps`` carry to the next window.
* ``fill_complete`` — ORDER CONSTRAINT 6: completed games only, FIFO by completion; exactly ``D`` rows
  (the trigger's target), laid column-major into the same buffer; GAE computed per COMPLETE game at
  its end (``game_gae``: sb3's arithmetic over a contiguous episode, bit for bit); every row labelled
  with its game's real outcome (``win_mask`` 1). A game straddling the D-th row is split: its tail
  stays at the FIFO head for the next update — **no row is ever dropped or down-weighted for age**
  (owner, 2026-09-29).

The one thing that removes rows is a CUT game (a quarantine, a core respawn): it has no outcome and no
next state, so its in-progress rows are released and COUNTED (``rows_cut``), never fabricated into a
loss. Window mode refuses a cut row outright (it is the parity tool, where a cut is a gate failure).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agents.training.win_prob_callback import backfill_terminal_labels  # the ONE window back-fill

#: Learner obs keys the collector fills, by source. Anything else in the observation space is REFUSED
#: at startup by name (a key nothing fills would be a silent zero in the buffer).
KEY_OBSERVATION = "observation"
KEY_MASK = "action_mask"
KEY_OPP_CLASS = "opp_class"
KEY_WIN_TARGET = "win_target"
KEY_WIN_MASK = "win_mask"
FILL_TIME_KEYS = (KEY_WIN_TARGET, KEY_WIN_MASK)
#: The fork arm's per-row POLICY-TERM mask (`fork_arm.PG_MASK_KEY`; declared only with
#: `--fork-fraction > 0`): 1.0 on every collected row, 0.0 on a branch's fork step (`fork.py`).
KEY_FORK_PG = "fork_pg_m"


class CollectorError(RuntimeError):
    """Base of every typed Lane-G collector failure."""


class ArenaExhausted(CollectorError):
    """More rows are live than the declared arena holds (capacity is a startup declaration)."""


class UnfillableKey(CollectorError):
    """The learner's observation space holds a key no source of the collector fills."""


@dataclass
class EndedGame:
    """One game that reached its end (a terminal row with its outcome)."""

    env: int
    episode: int
    slots: np.ndarray            # its rows, play order (complete-game mode); the terminal row only (window)
    reward: float
    outcome: float               # 1.0 win, 0.0 otherwise (a tie and the stall forfeit included)
    draw: bool                   # a TIE (truncated without the stall forfeit)
    forfeit: bool                # p1's stall forfeit (a plain loss)
    terminated: bool             # RAW env flags (before the learner's re-label)
    truncated: bool
    length: int                  # trainee decisions
    opp_class: int
    versions: Tuple[int, int]    # (oldest, newest) policy version that played it


@dataclass
class FillReport:
    """What one fill handed the learner — the STALENESS record of the update."""

    rows: int
    mode: str
    current_version: int
    age_hist: Dict[int, int] = field(default_factory=dict)   # age (updates) -> rows
    games_whole: int = 0          # complete-game mode: games entirely inside this update
    games_split: int = 0          # games whose rows straddle this update and the next
    carry_rows: int = 0           # completed rows left at the FIFO head for the next update
    in_progress_rows: int = 0     # rows of games still being played
    labelled_rows: int = 0        # win_mask == 1
    cut_rows_total: int = 0       # rows released by a cut game (run total)
    #: Each buffer row's COLLECTION PROVENANCE, ``[n_steps, n_envs]``-aligned (``row_provenance``): the
    #: K9(b) violation dump reads it so a mismatched row names where it came from and is replayable.
    provenance: Dict[str, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    @property
    def mean_age(self) -> float:
        n = sum(self.age_hist.values())
        return (sum(a * k for a, k in self.age_hist.items()) / n) if n else 0.0

    @property
    def current_share(self) -> float:
        n = sum(self.age_hist.values())
        return (self.age_hist.get(0, 0) / n) if n else 0.0


def obs_key_sources(obs_space: Any, core_label_keys: Sequence[str]) -> Dict[str, str]:
    """``{key: source}`` for every key of the learner's Dict observation space, or `UnfillableKey`.
    Sources: ``core_obs`` / ``core_mask`` / ``core_label`` / ``host_episode`` / ``fill``."""
    out: Dict[str, str] = {}
    bad: List[str] = []
    for key in obs_space.spaces:
        if key == KEY_OBSERVATION:
            out[key] = "core_obs"
        elif key == KEY_MASK:
            out[key] = "core_mask"
        elif key in core_label_keys:
            out[key] = "core_label"
        elif key == KEY_OPP_CLASS:
            out[key] = "host_episode"
        elif key in FILL_TIME_KEYS:
            out[key] = "fill"
        elif key == KEY_FORK_PG:
            out[key] = "host_const"
        else:
            bad.append(key)
    if bad:
        raise UnfillableKey(
            f"the learner's observation space holds {bad}, which the Rust collector does not fill "
            "(the core writes the production label columns; the host fills opp_class / win_target / "
            "win_mask). Turn the flag that declares them off, or port the key first — a key nothing "
            "fills would be a silent zero in every buffer row.")
    return out


#: ``RowStore.slot`` for a row whose serving slot was not recorded / a fork-arm branch row.
ROW_SOURCE_UNKNOWN = -1
ROW_SOURCE_FORK = -2
#: The per-row columns `row_provenance` hands the learner (with ``logp_all`` and, when present, ``turn``).
PROVENANCE_COLUMNS = ("env", "episode", "dec_n", "version", "slot", "u", "margin")


def row_provenance(store: "RowStore", idx: np.ndarray) -> Dict[str, np.ndarray]:
    """Each buffer row's collection provenance (``idx`` is the fill's ``[n_steps, n_envs]`` arena index):
    the env, episode and decision index that make the row REPLAYABLE on the deterministic core, the policy
    version and T2 slot that served it, the keyed draw (``u``, ``margin``) and the full masked log-prob
    row it was drawn from. Copied BEFORE the fill releases the rows (~45 bytes x 11 per row)."""
    out = {k: getattr(store, k)[idx].copy() for k in PROVENANCE_COLUMNS}
    out["logp_all"] = store.logp_all[idx].copy()
    if store.turn is not None:
        out["turn"] = store.turn[idx].copy()
    return out


class RowStore:
    """The preallocated arena (``capacity`` rows). ``alloc`` / ``release`` manage a free-slot stack;
    nothing is allocated after construction."""

    def __init__(self, obs_space: Any, capacity: int, *, n_actions: int = 11, fork: bool = False):
        if capacity < 1:
            raise ValueError("RowStore: capacity must be >= 1")
        self.capacity = int(capacity)
        self.n_actions = int(n_actions)
        self.obs: Dict[str, np.ndarray] = {}
        for key, sp in obs_space.spaces.items():
            if key in FILL_TIME_KEYS:
                continue
            self.obs[key] = np.zeros((self.capacity, *sp.shape), dtype=sp.dtype)
        c = self.capacity
        self.action = np.zeros(c, dtype=np.int64)
        self.logp = np.zeros(c, dtype=np.float32)
        self.value = np.zeros(c, dtype=np.float32)
        self.reward = np.zeros(c, dtype=np.float32)
        self.start = np.zeros(c, dtype=np.float32)       # 1.0 = the first row of its game
        self.terminal = np.zeros(c, dtype=bool)          # the last row of a game that ENDED
        self.outcome = np.zeros(c, dtype=np.float32)
        self.adv = np.zeros(c, dtype=np.float32)
        self.ret = np.zeros(c, dtype=np.float32)
        self.version = np.zeros(c, dtype=np.int64)
        self.env = np.zeros(c, dtype=np.int32)
        self.episode = np.zeros(c, dtype=np.int64)
        self.dec_n = np.zeros(c, dtype=np.int32)
        self.u = np.zeros(c, dtype=np.float64)
        self.margin = np.zeros(c, dtype=np.float64)
        # K9(b) provenance (gen3_behaviour_provenance_v1): the FULL masked log-prob row the stored
        # behaviour log-prob was drawn from (illegal = -inf; NaN = unknown), and the T2 slot that served it
        # (`ROW_SOURCE_FORK` for a fork-arm branch row).
        self.logp_all = np.full((c, self.n_actions), np.nan, dtype=np.float32)
        self.slot = np.full(c, ROW_SOURCE_UNKNOWN, dtype=np.int32)
        self._free = np.arange(c - 1, -1, -1, dtype=np.int64)   # a stack: pop from the end
        self._top = c
        # The fork arm (`fork.py`, declared only when it is on): each row's battle TURN (the selector's
        # band) and which live rows are BRANCH rows (the fork row budget counts them).
        self.turn: Optional[np.ndarray] = np.zeros(c, dtype=np.int32) if fork else None
        self.is_fork: Optional[np.ndarray] = np.zeros(c, dtype=bool) if fork else None
        self.fork_live = 0

    @property
    def free_count(self) -> int:
        return int(self._top)

    @property
    def live_count(self) -> int:
        return self.capacity - int(self._top)

    def alloc(self, k: int) -> np.ndarray:
        k = int(k)
        if k > self._top:
            raise ArenaExhausted(
                f"the row arena holds {self.capacity} rows and {self.live_count} are live; {k} more were "
                "asked for. The capacity is a startup declaration (max update rows + 2 x envs x the "
                "longest game) — a longer game or a larger trigger target needs a larger declaration.")
        out = self._free[self._top - k:self._top][::-1].copy()
        self._top -= k
        self.terminal[out] = False
        self.outcome[out] = 0.0
        self.reward[out] = 0.0
        self.adv[out] = 0.0
        self.ret[out] = 0.0
        self.logp_all[out] = np.nan
        self.slot[out] = ROW_SOURCE_UNKNOWN
        return out

    def release(self, slots: np.ndarray) -> None:
        s = np.asarray(slots, dtype=np.int64).reshape(-1)
        if s.size == 0:
            return
        if self._top + s.size > self.capacity:
            raise CollectorError("RowStore.release: more slots released than allocated")
        if self.is_fork is not None:
            self.fork_live -= int(self.is_fork[s].sum())
            self.is_fork[s] = False
        self._free[self._top:self._top + s.size] = s[::-1]
        self._top += s.size


    def mark_fork(self, slots: np.ndarray) -> None:
        """Mark freshly allocated rows as BRANCH rows (the fork row budget's count)."""
        if self.is_fork is None:
            raise CollectorError("RowStore.mark_fork on a store built without the fork arm")
        s = np.asarray(slots, dtype=np.int64).reshape(-1)
        self.is_fork[s] = True
        self.fork_live += int(s.size)


def game_gae(rewards: np.ndarray, values: np.ndarray, gamma: float, gae_lambda: float
             ) -> Tuple[np.ndarray, np.ndarray]:
    """GAE over ONE COMPLETE game (its last row terminal): ``(advantages, returns)`` in float32 with
    sb3's ``compute_returns_and_advantage`` arithmetic, operation for operation, so a game that lies
    wholly inside an sb3 window gets the SAME bits there and here (``store_test`` pins it)."""
    r = np.asarray(rewards, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32)
    n = r.shape[0]
    adv = np.zeros(n, dtype=np.float32)
    last = np.float32(0.0)
    one = np.float32(1.0)
    for step in range(n - 1, -1, -1):
        if step == n - 1:
            nnt = one - np.float32(1.0)          # the game ended: dones = 1
            nv = np.float32(0.0)
        else:
            nnt = one - np.float32(0.0)          # the next row is the same game (episode_start 0)
            nv = v[step + 1]
        delta = r[step] + gamma * nv * nnt - v[step]
        last = delta + gamma * gae_lambda * nnt * last
        adv[step] = last
    return adv, adv + v


class GameLog:
    """Per-env game assembly over a ``RowStore`` (module docs). ``mode`` is ``window`` or
    ``complete_game``; ``gamma`` / ``gae_lambda`` are the POLICY's (complete-game GAE)."""

    def __init__(self, store: RowStore, n_envs: int, *, mode: str, gamma: float, gae_lambda: float,
                 max_game_rows: int):
        if mode not in ("window", "complete_game"):
            raise ValueError(f"GameLog: mode {mode!r}")
        self.store, self.n, self.mode = store, int(n_envs), mode
        self.gamma, self.gae_lambda = float(gamma), float(gae_lambda)
        self.max_game_rows = int(max_game_rows)
        self.cur: List[List[int]] = [[] for _ in range(self.n)]          # the in-progress game's rows
        self.order: List[Deque[int]] = [deque() for _ in range(self.n)]  # window: unconsumed rows
        self.completed: Deque[np.ndarray] = deque()                       # complete-game FIFO
        self.completed_rows = 0
        self.games_ended = 0
        self.rows_cut = 0
        self.games_cut = 0
        self.games_without_rows = 0
        self.cut_in_window = 0

    # ---- rows in
    def add(self, envs: np.ndarray, slots: np.ndarray) -> None:
        for e, s in zip(np.asarray(envs).tolist(), np.asarray(slots).tolist()):
            cur = self.cur[e]
            cur.append(s)
            if len(cur) > self.max_game_rows:
                raise CollectorError(
                    f"env {e}: a game reached {len(cur)} trainee decisions, above the declared "
                    f"max_game_rows {self.max_game_rows} (the arena's sizing assumption)")
            if self.mode == "window":
                self.order[e].append(s)

    def in_progress_rows(self) -> int:
        return sum(len(c) for c in self.cur)

    # ---- a game ends
    def end(self, env: int, *, reward: float, outcome: float, draw: bool, forfeit: bool,
            terminated: bool, truncated: bool, episode: int, opp_class: int) -> Optional[EndedGame]:
        rows = self.cur[env]
        self.cur[env] = []
        if not rows:
            self.games_without_rows += 1
            return None
        st = self.store
        last = rows[-1]
        st.reward[last] = np.float32(reward)
        st.terminal[last] = True
        self.games_ended += 1
        if self.mode == "window":
            st.outcome[last] = np.float32(outcome)
            slots = np.asarray([last], dtype=np.int64)
            vs = (int(st.version[last]), int(st.version[last]))
        else:
            slots = np.asarray(rows, dtype=np.int64)
            st.outcome[slots] = np.float32(outcome)
            adv, ret = game_gae(st.reward[slots], st.value[slots], self.gamma, self.gae_lambda)
            st.adv[slots] = adv
            st.ret[slots] = ret
            self.completed.append(slots)
            self.completed_rows += int(slots.size)
            v = st.version[slots]
            vs = (int(v.min()), int(v.max()))
        return EndedGame(env=int(env), episode=int(episode), slots=slots, reward=float(reward),
                         outcome=float(outcome), draw=bool(draw), forfeit=bool(forfeit),
                         terminated=bool(terminated), truncated=bool(truncated), length=len(rows),
                         opp_class=int(opp_class), versions=vs)

    def cut(self, env: int) -> int:
        """The in-progress game of ``env`` ends WITHOUT an outcome (quarantine / respawn): its rows are
        released (complete-game) or poisoned for the window (window mode refuses them). Returns rows."""
        rows = self.cur[env]
        self.cur[env] = []
        if not rows:
            return 0
        self.rows_cut += len(rows)
        self.games_cut += 1
        if self.mode == "window":
            self.cut_in_window += len(rows)
            raise CollectorError(
                f"env {env}: a game was CUT ({len(rows)} rows, no outcome) in WINDOW mode — the window "
                "is the parity schedule, where a quarantine is itself a failure")
        self.store.release(np.asarray(rows, dtype=np.int64))
        return len(rows)

    # ---- window mode
    def window_ready(self, n_steps: int) -> bool:
        return all(len(o) >= n_steps + 1 for o in self.order)

    def take_window(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(idx [n_steps, n_envs], next_slots [n_envs])``: each env's next ``n_steps`` rows and the
        row after them (the bootstrap row, which stays for the next window)."""
        if not self.window_ready(n_steps):
            raise CollectorError("take_window before every env holds n_steps + 1 rows")
        idx = np.empty((n_steps, self.n), dtype=np.int64)
        nxt = np.empty(self.n, dtype=np.int64)
        for e, o in enumerate(self.order):
            for t in range(n_steps):
                idx[t, e] = o.popleft()
            nxt[e] = o[0]
        return idx, nxt

    # ---- complete-game mode
    def insert_after(self, parent: np.ndarray, games: Sequence[np.ndarray]) -> None:
        """The fork arm (`fork.py`): complete BRANCH games join the FIFO right after their PARENT game
        (so a branch is trained in its parent's update whenever the parent is). A parent no longer in
        the FIFO (cannot happen: the pass runs before the fill over games no fill has taken) is a
        refusal, never a silent append."""
        if self.mode != "complete_game":
            raise CollectorError("insert_after: branch games need the complete-game FIFO")
        if not games:
            return
        p0 = int(np.asarray(parent).reshape(-1)[0])
        at = next((i for i, g in enumerate(self.completed) if g.size and int(g[0]) == p0), None)
        if at is None:
            raise CollectorError(f"insert_after: the parent game (first row {p0}) is not in the completed FIFO")
        for k, g in enumerate(games):
            self.completed.insert(at + 1 + k, np.asarray(g, dtype=np.int64))
            self.completed_rows += int(np.asarray(g).size)

    def take_complete(self, d: int) -> Tuple[np.ndarray, int, int]:
        """The first ``d`` completed rows, FIFO by game completion: ``(slots [d], whole games, split
        games)``. A game straddling row ``d`` leaves its tail at the FIFO head."""
        if d > self.completed_rows:
            raise CollectorError(f"take_complete({d}) with {self.completed_rows} completed rows")
        parts: List[np.ndarray] = []
        got = whole = split = 0
        while got < d:
            g = self.completed[0]
            need = d - got
            if g.size <= need:
                parts.append(self.completed.popleft())
                got += g.size
                whole += 1
            else:
                parts.append(g[:need])
                self.completed[0] = g[need:]
                got += need
                split += 1
        self.completed_rows -= d
        return np.concatenate(parts), whole, split


def _age_hist(versions: np.ndarray, current: int) -> Dict[int, int]:
    ages = int(current) - np.asarray(versions, dtype=np.int64).reshape(-1)
    if ages.size and int(ages.min()) < 0:
        raise CollectorError(f"a row's policy version is AHEAD of the learner's ({int(-ages.min())} updates)")
    u, k = np.unique(ages, return_counts=True)
    return {int(a): int(c) for a, c in zip(u, k)}


def _copy_rows(buf: Any, store: RowStore, idx: np.ndarray) -> None:
    """Gather the arena rows ``idx`` ``[n_steps, n_envs]`` into the buffer's arrays (in place)."""
    obs = buf.observations
    for key, arr in store.obs.items():
        obs[key][...] = arr[idx]
    buf.actions[...] = store.action[idx].reshape(buf.actions.shape).astype(buf.actions.dtype)
    buf.rewards[...] = store.reward[idx]
    buf.episode_starts[...] = store.start[idx]
    buf.values[...] = store.value[idx]
    buf.log_probs[...] = store.logp[idx]
    m = store.obs[KEY_MASK][idx] if KEY_MASK in store.obs else None
    if m is not None:
        buf.action_masks[...] = m.reshape(buf.action_masks.shape).astype(np.float32)


def _check_buffer(buf: Any, n_steps: int, n_envs: int) -> None:
    if int(buf.buffer_size) != n_steps or int(buf.n_envs) != n_envs:
        raise CollectorError(f"the learner's buffer is [{buf.buffer_size}, {buf.n_envs}], the fill is "
                             f"[{n_steps}, {n_envs}]")


def fill_window(buf: Any, log: GameLog, n_steps: int, *, current_version: int) -> Tuple[FillReport, np.ndarray]:
    """TODAY'S SCHEDULE (module docs): the bootstrap is V of each env's NEXT row (played by the policy
    of the time, which stays in the arena for the next window). Returns ``(report, versions)``; the
    window's rows are released after the copy."""
    import torch as th

    st = log.store
    _check_buffer(buf, n_steps, log.n)
    buf.reset()
    idx, nxt = log.take_window(n_steps)
    _copy_rows(buf, st, idx)
    dones = st.start[nxt] >= 0.5                      # the window's last row ended its game
    buf.compute_returns_and_advantage(
        last_values=th.as_tensor(np.asarray(st.value[nxt], dtype=np.float32)), dones=dones)
    if KEY_WIN_TARGET in buf.observations:
        scratch = np.where(st.terminal[idx], st.outcome[idx], np.nan).astype(np.float32)
        backfill_terminal_labels(scratch, buf.episode_starts, buf.observations[KEY_WIN_TARGET],
                                 buf.observations[KEY_WIN_MASK])
    buf.pos = n_steps
    buf.full = True
    rep = FillReport(rows=int(idx.size), mode="window", current_version=int(current_version),
                     age_hist=_age_hist(st.version[idx], current_version),
                     in_progress_rows=log.in_progress_rows(), cut_rows_total=log.rows_cut)
    if KEY_WIN_MASK in buf.observations:
        rep.labelled_rows = int((buf.observations[KEY_WIN_MASK] >= 0.5).sum())
    rep.provenance = row_provenance(st, idx)
    versions = st.version[idx].copy()
    st.release(idx.reshape(-1))
    return rep, versions


def fill_complete(buf: Any, log: GameLog, d: int, *, current_version: int) -> Tuple[FillReport, np.ndarray]:
    """ORDER CONSTRAINT 6 (module docs): exactly ``d`` completed rows, column-major into the buffer
    ``[d / n_envs, n_envs]``; the precomputed complete-game GAE; every row labelled. Returns
    ``(report, versions [n_steps, n_envs])``."""
    n = log.n
    if d % n:
        raise CollectorError(f"the update size {d} is not a multiple of n_envs {n}")
    n_steps = d // n
    st = log.store
    _check_buffer(buf, n_steps, n)
    buf.reset()
    flat, whole, split = log.take_complete(d)
    idx = flat.reshape(n, n_steps).T                  # column j = FIFO rows [j*n_steps, (j+1)*n_steps)
    _copy_rows(buf, st, idx)
    buf.advantages[...] = st.adv[idx]
    buf.returns[...] = st.ret[idx]
    if KEY_WIN_TARGET in buf.observations:
        buf.observations[KEY_WIN_TARGET][..., 0] = st.outcome[idx]
        buf.observations[KEY_WIN_MASK][..., 0] = 1.0
    buf.pos = n_steps
    buf.full = True
    rep = FillReport(rows=int(d), mode="complete_game", current_version=int(current_version),
                     age_hist=_age_hist(st.version[idx], current_version), games_whole=whole,
                     games_split=split, carry_rows=log.completed_rows,
                     in_progress_rows=log.in_progress_rows(), labelled_rows=int(d),
                     cut_rows_total=log.rows_cut)
    rep.provenance = row_provenance(st, idx)
    versions = st.version[idx].copy()
    st.release(flat)
    return rep, versions


def write_rows(store: RowStore, slots: np.ndarray, cols: Mapping[str, np.ndarray], envs: np.ndarray, *,
               label_keys: Sequence[str], opp_class: np.ndarray, actions: np.ndarray, logp: np.ndarray,
               values: np.ndarray, version: np.ndarray, u: np.ndarray, margin: np.ndarray,
               starts: np.ndarray, logp_all: Optional[np.ndarray] = None,
               served_by: Optional[np.ndarray] = None) -> None:
    """Write one host step's trainee rows (``envs`` with need[:, 0] = 1) from the core's columns.
    ``logp_all`` (the full masked log-prob rows the draw used) and ``served_by`` (the T2 slot per row)
    are the K9(b) provenance (`row_provenance`)."""
    o = store.obs
    if KEY_OBSERVATION in o:
        o[KEY_OBSERVATION][slots] = cols["obs"][envs, 0]
    if KEY_MASK in o:
        o[KEY_MASK][slots] = cols["mask"][envs, 0]
    for k in label_keys:
        if k in o:
            o[k][slots] = cols[k][envs, 0]
    if KEY_OPP_CLASS in o:
        o[KEY_OPP_CLASS][slots, 0] = opp_class
    store.action[slots] = actions
    store.logp[slots] = logp
    store.value[slots] = values
    store.version[slots] = version
    store.env[slots] = envs
    store.episode[slots] = cols["episode"][envs]
    store.dec_n[slots] = cols["dec_n"][envs, 0]
    if KEY_FORK_PG in o:
        o[KEY_FORK_PG][slots] = 1.0
    if store.turn is not None:
        store.turn[slots] = cols["turn"][envs]
    store.u[slots] = u
    store.margin[slots] = margin
    store.start[slots] = starts
    if logp_all is not None:
        store.logp_all[slots] = logp_all
    if served_by is not None:
        store.slot[slots] = served_by
