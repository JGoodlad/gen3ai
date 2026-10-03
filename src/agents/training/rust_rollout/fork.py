"""THE FORK ARM ON THE RUST CORE (`gen3_fork_rust_v1`; ``designs/training/forks.md`` §14).

A DECLARED-BUT-OFF capability (owner, 2026-10-01: in-loop additions are one ply and subsample-eligible;
playouts to a terminal are DEFERRED). It ported the Python arm's tested rules (`fork_arm`, `fork_buffer.FILL`)
onto the Rust collector; the Python arm itself was deleted in deletion pass L5. ``--fork-fraction 0`` (the default
everywhere) builds NOTHING: no pass, no handle, no ``fork_pg_m`` key, no arena row.

THE PASS (``RustForkPass.run``) is the collector's ``fork`` phase, between the trigger firing and
the fill (``collector.COLLECT_PHASES``):

1. the CANDIDATES are the games that ended since the last pass (each with its core input log,
   read from ``core.finished()`` in the op it ended);
2. the SELECTOR is §3's: eligible rows (turn band, a move round with >= 3 legal actions), a pool
   drawn uniformly over games (``--fork-max-per-battle`` per game), scored by the CURRENT trainee
   slot, the ``--fork-contested-gap`` quantile of the top-2 log-prob gap, tightest first;
3. each fork REPLAYS its game to the fork decision on a declared playout handle (Lane I) and plays
   ``--fork-branches`` continuations (top1, top2, a decision-keyed random legal action) to the end,
   all forks of a wave in lockstep, every pending decision of every branch in ONE T2 flush per step;
4. every branch's trainee rows go into the ARENA (row 0 = the fork step, ``fork_pg_m`` 0; the rest
   1), its GAE is ``store.game_gae`` with the core's indicator reward, and the branch joins the
   completed-game FIFO RIGHT AFTER ITS PARENT — so it competes for the same update rows ``D``.

COMMON RANDOM NUMBERS (§14.3): the dice are the battle's own PRNG from the branch point (playout seed
``null``); ``dice_and_draws`` keys every draw on the PARENT's (segment run seed, env, episode) and the
decision's frame index ``n`` (read from the playout, never counted), stream 0 for p1 and 1 for p2 —
so a branch taking the parent's action against the parent's opponent IS the parent (gate (a),
``fork_crn_integration_test``). ``dice`` keys each branch on its own streams (the control).
"""
from __future__ import annotations

import ctypes
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.training import fork_arm as FA
from agents.training import keyed_draw as KD
from agents.training.lever_supply import LEVERS, DryStreakGuard, LeverConfigError, loud
from agents.training.rust_rollout import store as S

FORK_RUST_ID = "gen3_fork_rust_v1"

#: Playout handles (forks played at once, in lockstep). Each holds ``branches`` branches; one wave step
#: serves at most ``2 x branches x FORK_CONCURRENCY`` decisions in ONE flush (checked at startup).
FORK_CONCURRENCY = 32

#: The update-row budget of fork rows live in the arena, as a multiple of the trigger's target — the
#: deleted Python arm's rule too: branch rows may at most match own rows.
ROW_BUDGET_MULTIPLE = 1.0

#: ``dice`` (the control): branch ``b``'s side ``s`` draws on stream ``FORK_STREAM_BASE + 2 b + s`` —
#: unpaired across branches, and never a training stream (0 / 1) or a respawn's (11).
FORK_STREAM_BASE = 16

#: ``MaskableAgentWrapper.OPP_CLASS_POOL`` — the label of a row played against a SUBSTITUTED
#: (self-like) opponent (``fork_buffer.FILL``'s rule).
OPP_CLASS_POOL = 1

#: The port panics at 1,000 committed turns; the playout's ceiling.
MAX_TURNS = 999


class ForkKeyMismatch(S.CollectorError):
    """A replayed fork decision is not the one its arena row names (the frame index ``n`` or the
    legal set differs) — the CRN identity would be broken, so the pass refuses rather than inject."""


@dataclass(frozen=True)
class ForkDecl:
    """The arm's startup declaration (the flags, resolved)."""

    fraction: float
    branches: int = FA.DEFAULT_BRANCHES
    contested_gap: float = FA.DEFAULT_CONTESTED_GAP
    contested_absv: float = FA.DEFAULT_CONTESTED_ABSV
    max_per_battle: int = FA.DEFAULT_MAX_PER_BATTLE
    crn: str = FA.DEFAULT_CRN
    seed: int = 0
    concurrency: int = FORK_CONCURRENCY
    row_budget_multiple: float = ROW_BUDGET_MULTIPLE
    max_forks: int = FA.MAX_FORKS_PER_ROLLOUT
    starve_cycles: int = LEVERS["fork"].default_cycles

    def __post_init__(self) -> None:
        if not float(self.fraction) > FA.FORK_OFF:
            raise ValueError("ForkDecl: fraction must be > 0 (0 = OFF builds no ForkDecl at all)")
        if int(self.branches) not in FA.BRANCH_CHOICES:
            raise ValueError(f"ForkDecl: branches {self.branches} not in {FA.BRANCH_CHOICES}")
        if self.crn not in FA.CRN_MODES:
            raise ValueError(f"ForkDecl: crn {self.crn!r} not in {FA.CRN_MODES}")
        if int(self.concurrency) < 1:
            raise ValueError("ForkDecl: concurrency must be >= 1")

    def row_budget(self, target_hi: int) -> int:
        return int(np.ceil(float(self.row_budget_multiple) * int(target_hi)))

    def rows_per_wave_step(self) -> int:
        return 2 * int(self.branches) * int(self.concurrency)


@dataclass
class GameRecord:
    """One ended game the next pass may fork (captured at its end)."""

    env: int
    episode: int
    slots: np.ndarray            # its arena rows, play order
    run_seed: int                # the segment seed its draws were keyed on
    route: int                   # its opponent route
    model_id: Optional[str]      # the model its route's slot served when the episode started
    opp_class: int
    script: Optional[str]        # the core input log (``InputLog::script``), None if not read


@dataclass
class Branch:
    name: str
    action: int
    slots: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    end: Optional[dict] = None
    decisions: int = 0
    cmds: Optional[List[str]] = None     # the branch's whole input log (only under ``keep_cmds``: a gate)


@dataclass
class Fork:
    rec: GameRecord
    t: int                       # the fork row's index in its game
    top1: int
    top2: int
    legal: List[int]
    branches: List[Branch]
    opp_slot: int                # the T2 slot p2 is served from
    opp_real: bool
    opp_temperature: float
    opp_greedy: bool
    dropped: str = ""            # why the fork was dropped whole ("" = kept)


def parse_script(script: str) -> dict:
    """``InputLog::script`` (``START <json>`` + one command per line) as the playout's log dict."""
    lines = [ln for ln in script.split("\n") if ln]
    if not lines or not lines[0].startswith("START "):
        raise S.CollectorError("a finished episode's script has no START line")
    st = json.loads(lines[0][len("START "):])
    return {"format_id": st["formatid"], "seed": st["seed"], "names": [st["p1"]["name"], st["p2"]["name"]],
            "teams": [st["p1"]["team"], st["p2"]["team"]], "cmds": lines[1:]}


def p1_command_index(cmds: Sequence[str], dec: int) -> int:
    """The index in ``cmds`` of p1's ``dec``-th answered decision's command (0-based): the fork's
    ``at`` (the core feeds p1 FIRST, so p2's simultaneous decision is still open there)."""
    k = -1
    for i, c in enumerate(cmds):
        if c.startswith("CHOOSE p1 "):
            k += 1
            if k == dec:
                return i
    raise S.CollectorError(f"the input log holds {k + 1} p1 commands; decision {dec} is not among them")


class PlayoutHandle:
    """One declared playout handle (``SearchCore`` + its scratch), driven step by step."""

    def __init__(self, core: Any, branches: int, obs_dim: int, n_actions: int):
        self.core = core
        cap = 2 * int(branches)
        self.cap = cap
        self.rows = np.zeros((cap, obs_dim), dtype=np.float32)
        self.masks = np.zeros((cap, n_actions), dtype=np.uint8)
        self.who = np.zeros(cap, dtype=np.uint32)
        self.ns = np.zeros(cap, dtype=np.uint32)
        self.acts = np.zeros(cap, dtype=np.int32)
        self.n_fed = 0

    def open(self, req: dict) -> dict:
        self.n_fed = 0
        return self.core._cstr(self.core.lib.rust_env_playout_open, json.dumps(req).encode())

    def step(self) -> int:
        from utils.rust_env.successors import _raise

        c = self.core
        with c._lock:
            k = c.lib.rust_env_playout_step(c._h, self.acts.ctypes.data, self.n_fed, self.rows.ctypes.data,
                                            self.masks.ctypes.data, self.who.ctypes.data, self.cap)
            if k == ctypes.c_size_t(-1).value:
                _raise(c.lib)
            if k:
                m = c.lib.rust_env_playout_pending_n(c._h, self.ns.ctypes.data, self.cap)
                if m != k:
                    _raise(c.lib)
        return int(k)

    def results(self) -> dict:
        return self.core._cstr(self.core.lib.rust_env_playout_results)


@dataclass
class PassReport:
    candidates: int = 0
    own_rows: int = 0
    records_missing: int = 0
    eligible: int = 0
    pool: int = 0
    threshold: float = float("nan")
    asked: int = 0               # what the fraction asked for (capped by the measured rows per fork)
    requested: int = 0           # forks SELECTED and played (the pool can hold fewer than the ask)
    forks: List[Fork] = field(default_factory=list)
    injected_rows: int = 0
    masked_rows: int = 0
    dropped_forks: int = 0
    seconds: float = 0.0


class RustForkPass:
    """The arm on one collector (module docs). Built at STARTUP (``build_collector``): the playout
    handles and their scratch are acquired here and nothing after."""

    def __init__(self, decl: ForkDecl, *, lib_path: Any, nan_poison: bool, turn_limit: int, victory_value: float,
                 target_hi: int, max_rows_per_flush: int, obs_dim: int, n_actions: int = 11, emit=loud):
        from utils.rust_env.successors import SearchCore, spec_json

        self.decl = decl
        need = decl.rows_per_wave_step()
        if need > int(max_rows_per_flush):
            raise S.CollectorError(
                f"the fork arm's wave step serves up to {need} decisions (2 x {decl.branches} branches x "
                f"{decl.concurrency} handles) in one flush; T2 declares max_rows_per_flush {max_rows_per_flush}")
        from utils.rust_env import ffi as F

        lib = F.load(lib_path, nan_poison=nan_poison)
        self.handles = [PlayoutHandle(SearchCore(spec_json(max_nodes=1, max_branches=int(decl.branches)), lib=lib),
                                      int(decl.branches), obs_dim, n_actions) for _ in range(int(decl.concurrency))]
        self.turn_limit = int(turn_limit)
        self.victory_value = float(victory_value)
        self.row_budget = decl.row_budget(target_hi)
        self.pending: List[GameRecord] = []
        self.rows_per_fork: Optional[float] = None
        self.passes = 0
        self.last: Optional[PassReport] = None
        #: the dice: the battle's own stream from the branch point (§14.3). A gate reseeds it to prove the
        #: comparison sees the dice; ``keep_cmds`` makes a branch report its whole input log.
        self.seeds: List[Optional[str]] = [None]
        self.keep_cmds = False
        self.guard = DryStreakGuard("fork", int(decl.starve_cycles), emit=emit)
        self._emit = emit
        emit(f"🍴 [FORK] {FORK_RUST_ID}: fraction {decl.fraction}, {decl.branches} branches, CRN {decl.crn}, "
             f"{decl.concurrency} playout handles, fork row budget {self.row_budget:,}")
        emit(self.guard.announce())

    # ------------------------------------------------------------------ capture (collector._end_game)
    def capture(self, rec: GameRecord) -> None:
        self.pending.append(rec)

    # ------------------------------------------------------------------ the pass
    def run(self, col: Any, model: Any = None) -> PassReport:
        """One pass over the games ended since the last (module docs). Returns the report it logs."""
        t0 = time.perf_counter()
        recs, self.pending = self.pending, []
        rep = PassReport(candidates=len(recs), own_rows=int(sum(r.slots.size for r in recs)))
        self.passes += 1
        try:
            forks = self._select(col, recs, rep)
            for w in range(0, len(forks), len(self.handles)):
                self._play_wave(col, forks[w:w + len(self.handles)], rep)
            rep.forks = forks
            self._inject(col, forks, rep)
        finally:
            rep.seconds = time.perf_counter() - t0
            self.last = rep
        kept = [f for f in rep.forks if not f.dropped]
        if kept:
            self.rows_per_fork = rep.injected_rows / len(kept)
        self.guard.observe(int(rep.injected_rows), why=self._why(rep))
        if model is not None:
            self.record(model, rep)
        return rep

    @staticmethod
    def _why(rep: PassReport) -> str:
        if rep.candidates == 0:
            return "no game ended since the last pass"
        if rep.eligible == 0:
            return "no eligible decision (turn band, a move round with >= 3 legal actions)"
        return f"{len(rep.forks)} forks selected, {rep.dropped_forks} dropped, 0 rows injected"

    # ---- 1. the selector (§3)
    def _select(self, col: Any, recs: List[GameRecord], rep: PassReport) -> List[Fork]:
        st = col.store
        recs_ok = []
        for r in recs:
            if r.script is None:
                rep.records_missing += 1
            else:
                recs_ok.append(r)
        if not recs_ok:
            return []
        L = max(r.slots.size for r in recs_ok)
        G = len(recs_ok)
        idx = np.full((L, G), -1, dtype=np.int64)
        for g, r in enumerate(recs_ok):
            idx[:r.slots.size, g] = r.slots
        valid = idx >= 0
        safe = np.where(valid, idx, 0)
        turns = np.where(valid, st.turn[safe], -1)
        masks = st.obs[S.KEY_MASK][safe] * valid[..., None]
        el = FA.eligible_mask(valid.astype(np.float32), turns, masks)
        rep.eligible = int(el.sum())
        starts = np.zeros((L, G), dtype=np.float32)
        starts[0, :] = 1.0
        n_ask = FA.n_forks_for(self.decl.fraction, rep.own_rows, 1, cap=int(self.decl.max_forks))
        if self.rows_per_fork:
            n_ask = min(n_ask, int(self.row_budget // max(1.0, self.rows_per_fork)))
        rep.asked = int(n_ask)
        if n_ask <= 0 or not el.any():
            return []
        rng = np.random.default_rng([int(self.decl.seed) & 0xFFFFFFFF, self.passes, int(col.cfg.run_seed) & 0xFFFFFFFF])
        pool = FA.candidate_pool(el, starts, FA.pool_size_for(n_ask, self.decl.contested_gap),
                                 int(self.decl.max_per_battle), rng)
        rep.pool = len(pool)
        if not pool:
            return []
        rows = np.asarray([idx[t, g] for t, g in pool], dtype=np.int64)
        logp, value = self._score(col, st.obs[S.KEY_OBSERVATION][rows], st.obs[S.KEY_MASK][rows])
        gaps, a1, a2 = FA.top2_gaps(logp, st.obs[S.KEY_MASK][rows])
        thr = FA.contested_threshold(gaps, self.decl.contested_gap)
        rep.threshold = thr
        absv = (value - 0.5) if self.decl.contested_absv > 0.0 else None
        chosen = FA.contested_select(pool, gaps, absv, thr, self.decl.contested_absv, n_ask)
        out: List[Fork] = []
        for i in chosen:
            t, g = pool[i]
            rec = recs_ok[g]
            slot = int(rows[i])
            legal = [int(a) for a in np.flatnonzero(st.obs[S.KEY_MASK][slot])]
            acts = FA.branch_actions(legal, int(a1[i]), int(a2[i]), int(self.decl.branches),
                                     salt=f"{rec.env}|{rec.episode}|{int(st.dec_n[slot])}", seed=int(self.decl.seed))
            out.append(self._fork(col, rec, int(t), int(a1[i]), int(a2[i]), legal, acts))
        rep.requested = len(out)
        return out

    def _fork(self, col: Any, rec: GameRecord, t: int, top1: int, top2: int, legal: List[int],
              acts: Dict[str, int]) -> Fork:
        """§14.4: the parent's REAL policy slot when it still serves the episode's model, else the
        current trainee (the self-like substitute, labelled POOL)."""
        trainee = int(col.current_slot)
        route = col.opponents.routes[rec.route] if 0 <= rec.route < len(col.opponents.routes) else None
        real = (route is not None and route.kind == "policy" and col.server is not None and rec.model_id is not None
                and col.svc.model_id(int(route.slot)) == rec.model_id)
        if real:
            assert route is not None
            greedy = bool(col.server.force_greedy or not route.stochastic)
            temp = float(col.server.temperature.get(route.player, 1.0))
            slot = int(route.slot)
        else:
            greedy, temp, slot = False, 1.0, trainee
        return Fork(rec=rec, t=t, top1=top1, top2=top2, legal=legal,
                    branches=[Branch(name=n, action=int(a)) for n, a in acts.items()],
                    opp_slot=slot, opp_real=bool(real), opp_temperature=temp, opp_greedy=greedy)

    def _score(self, col: Any, obs: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The CURRENT trainee slot's ``(logp, value)`` over ``obs``, chunked to T2's flush arena."""
        from agents.inference.service.spec import Priority

        R = int(col.svc.spec.max_rows_per_flush)
        lp_out = np.empty((obs.shape[0], masks.shape[1]), dtype=np.float32)
        v_out = np.empty(obs.shape[0], dtype=np.float32)
        for a in range(0, obs.shape[0], R):
            b = min(obs.shape[0], a + R)
            tk = col.svc.submit(int(col.current_slot), obs[a:b], masks[a:b], Priority.ROLLOUT)
            col.svc.flush()
            lp, v, _g = tk.host()
            lp_out[a:b] = lp
            v_out[a:b] = v
        return lp_out, v_out

    # ---- 2. playing a wave
    def _uniforms(self, f: Fork, b: int, side: int, n: int) -> float:
        rec = f.rec
        if self.decl.crn == "dice_and_draws":
            stream = KD.STREAM_TRAINEE if side == 0 else KD.STREAM_OPPONENT
        else:
            stream = FORK_STREAM_BASE + 2 * b + side
        return float(KD.keyed_uniforms(int(rec.run_seed), stream, int(rec.env), int(rec.episode), int(n)))

    def _alloc(self, col: Any, k: int) -> Optional[np.ndarray]:
        st = col.store
        if st.fork_live + k > self.row_budget:
            return None
        s = st.alloc(k)
        st.mark_fork(s)
        return s

    def _drop(self, col: Any, f: Fork, why: str) -> None:
        if f.dropped:
            return
        f.dropped = why
        sl = [s for b in f.branches for s in b.slots]
        if sl:
            col.store.release(np.asarray(sl, dtype=np.int64))
        for b in f.branches:
            b.slots, b.values = [], []

    def _play_wave(self, col: Any, wave: List[Fork], rep: PassReport) -> None:
        from agents.inference.service.spec import Priority

        st = col.store
        for h, f in zip(self.handles, wave):
            rec = f.rec
            parent = int(rec.slots[f.t])
            dec = int(st.dec_n[parent])
            log = parse_script(rec.script or "")
            at = p1_command_index(log["cmds"], dec)
            root = h.open({"log": log, "at": at, "side": "p1", "actions": [b.action for b in f.branches],
                           "seeds": list(self.seeds), "stall": {"turn_limit": self.turn_limit, "side": "p1"},
                           "max_turns": MAX_TURNS, "keep_cmds": bool(self.keep_cmds)})
            legal = sorted(int(k) for k in root["tokens"])
            if int(root["n"]) != dec or legal != f.legal:
                raise ForkKeyMismatch(
                    f"env {rec.env} episode {rec.episode}: the replayed fork decision has n={root['n']} and legal "
                    f"{legal}; its arena row names n={dec} and legal {f.legal}")
        # row 0 of every branch: the fork step, re-served by the CURRENT trainee slot
        prow = np.asarray([int(f.rec.slots[f.t]) for f in wave], dtype=np.int64)
        lp0, v0 = self._score(col, st.obs[S.KEY_OBSERVATION][prow], st.obs[S.KEY_MASK][prow])
        for j, f in enumerate(wave):
            for b in f.branches:
                s = self._alloc(col, 1)
                if s is None:
                    self._drop(col, f, "row budget")
                    break
                self._write(col, f, b, int(s[0]), obs_from=int(prow[j]), mask_from=None, action=b.action,
                            logp=float(lp0[j, b.action]), value=float(v0[j]), dec=int(st.dec_n[prow[j]]),
                            u=0.0, margin=1.0, start=1.0, pg=0.0, logp_row=lp0[j])
        active = [(h, f) for h, f in zip(self.handles, wave) if not f.dropped]
        while active:
            pend: List[Tuple[PlayoutHandle, Fork, int]] = []
            nxt = []
            for h, f in active:
                k = h.step()
                h.n_fed = k                     # the next step feeds exactly this pending list's answers
                if k == 0:
                    self._finish(f, h.results())
                    continue
                nxt.append((h, f))
                pend.extend((h, f, i) for i in range(k))
            active = nxt
            if not pend:
                break
            groups: Dict[int, List[Tuple[PlayoutHandle, Fork, int]]] = {}
            for e in pend:
                h, f, i = e
                side = int(h.who[i]) % 2
                slot = int(col.current_slot) if side == 0 else f.opp_slot
                groups.setdefault(slot, []).append(e)
            tickets = []
            for slot, es in groups.items():
                obs = np.stack([h.rows[i] for h, _f, i in es])
                msk = np.stack([h.masks[i] for h, _f, i in es]).astype(bool)
                tickets.append((es, col.svc.submit(slot, obs, msk, Priority.ROLLOUT)))
            col.svc.flush()
            for es, tk in tickets:
                lp, v, gr = tk.host()
                lp, v, gr = lp.copy(), v.copy(), gr.copy()
                for j, (h, f, i) in enumerate(es):
                    b = int(h.who[i]) // 2
                    side = int(h.who[i]) % 2
                    n = int(h.ns[i])
                    br = f.branches[b]
                    br.decisions += 1
                    if side == 1 and f.opp_greedy:
                        h.acts[i] = int(gr[j])
                        continue
                    u = self._uniforms(f, b, side, n)
                    temp = 1.0 if side == 0 else f.opp_temperature
                    a, margin = KD.keyed_actions(lp[j:j + 1], np.asarray([u]), temp)
                    h.acts[i] = int(a[0])
                    if side == 1 or f.dropped:
                        continue
                    s = self._alloc(col, 1)
                    if s is None:
                        self._drop(col, f, "row budget")
                        continue
                    self._write(col, f, br, int(s[0]), obs_from=None, mask_from=(h, i), action=int(a[0]),
                                logp=float(lp[j, int(a[0])]), value=float(v[j]), dec=n, u=u,
                                margin=float(margin[0]), start=0.0, pg=1.0, logp_row=lp[j])
            active = [(h, f) for h, f in active if not f.dropped]
        for f in wave:
            if not f.dropped and any(b.end is None for b in f.branches):
                self._drop(col, f, "unfinished")

    def _write(self, col: Any, f: Fork, b: Branch, slot: int, *, obs_from: Optional[int], mask_from: Any,
               action: int, logp: float, value: float, dec: int, u: float, margin: float, start: float,
               pg: float, logp_row: Optional[np.ndarray] = None) -> None:
        st = col.store
        o = st.obs
        if obs_from is not None:
            o[S.KEY_OBSERVATION][slot] = o[S.KEY_OBSERVATION][obs_from]
            o[S.KEY_MASK][slot] = o[S.KEY_MASK][obs_from]
        else:
            h, i = mask_from
            o[S.KEY_OBSERVATION][slot] = h.rows[i]
            o[S.KEY_MASK][slot] = h.masks[i]
        from agents.training.fork_buffer import FILL

        ctx = {"outcome": 0.0, "pg_mask": np.asarray([pg], dtype=np.float32)}
        for key, arr in o.items():
            if key in (S.KEY_OBSERVATION, S.KEY_MASK):
                continue
            if key == S.KEY_OPP_CLASS:
                arr[slot] = f.rec.opp_class if f.opp_real else OPP_CLASS_POOL
                continue
            builder, _why = FILL[key]
            arr[slot] = builder(1, ctx)[0]
        st.action[slot] = int(action)
        st.logp[slot] = np.float32(logp)
        st.value[slot] = np.float32(value)
        st.version[slot] = int(col.version)
        st.env[slot] = int(f.rec.env)
        st.episode[slot] = int(f.rec.episode)
        st.dec_n[slot] = int(dec)
        st.u[slot] = float(u)
        st.margin[slot] = float(margin)
        st.start[slot] = np.float32(start)
        st.slot[slot] = S.ROW_SOURCE_FORK                 # K9(b) provenance: a fork-arm branch row
        if logp_row is not None:
            st.logp_all[slot] = logp_row
        b.slots.append(int(slot))
        b.values.append(float(value))

    def _finish(self, f: Fork, res: dict) -> None:
        for b, r in zip(f.branches, res["branches"]):
            if int(r["action"]) != b.action:
                raise S.CollectorError(f"playout branch order: {r['action']} vs {b.action}")
            b.end = r["end"]
            b.decisions = int(sum(r["decisions"]))
            if self.keep_cmds:
                b.cmds = list(r["cmds"])

    def capped(self, b: Branch) -> bool:
        e = b.end or {}
        return bool(e.get("truncated") or e.get("forfeit") or int(e.get("turn", 0)) >= self.turn_limit)

    # ---- 3. into the FIFO
    def _inject(self, col: Any, forks: List[Fork], rep: PassReport) -> None:
        st = col.store
        log = col.log
        for f in forks:
            if f.dropped:
                continue
            for b in f.branches:
                if self.capped(b) or len(b.slots) > log.max_game_rows:
                    if b.slots:
                        st.release(np.asarray(b.slots, dtype=np.int64))
                    b.slots, b.values = [], []
            if sum(1 for b in f.branches if b.slots) < 2:
                self._drop(col, f, "fewer than two scorable branches")
        rep.dropped_forks = sum(1 for f in forks if f.dropped)
        for f in forks:
            if f.dropped:
                continue
            games = []
            for b in f.branches:
                if not b.slots:
                    continue
                s = np.asarray(b.slots, dtype=np.int64)
                won = 1.0 if (b.end or {}).get("winner") == 0 else 0.0
                last = int(s[-1])
                st.reward[s] = 0.0
                st.reward[last] = np.float32(self.victory_value * won)
                st.terminal[last] = True
                st.outcome[s] = np.float32(won)
                adv, ret = S.game_gae(st.reward[s], st.value[s], log.gamma, log.gae_lambda)
                st.adv[s] = adv
                st.ret[s] = ret
                games.append(s)
                rep.injected_rows += int(s.size)
                rep.masked_rows += 1
            log.insert_after(f.rec.slots, games)

    # ------------------------------------------------------------------ telemetry
    def metrics(self, rep: PassReport) -> Dict[str, float]:
        forks = []
        for f in rep.forks:
            if f.dropped:
                continue
            br = {}
            for b in f.branches:
                won = None if b.end is None else (1.0 if b.end.get("winner") == 0 else 0.0)
                br[b.name] = {"outcome": won, "capped": self.capped(b), "decisions": b.decisions,
                              "succ_value": (b.values[1] if len(b.values) > 1 else None)}
            forks.append({"branches": br})
        out = FA.fork_metrics(forks=forks, requested=rep.requested, eligible=rep.eligible, pool=rep.pool,
                              threshold=rep.threshold, injected_rows=rep.injected_rows, buffer_rows=rep.own_rows,
                              masked_rows=rep.masked_rows, fraction=self.decl.fraction, branches=self.decl.branches,
                              crn=self.decl.crn, seconds=rep.seconds, records_missing=rep.records_missing,
                              n_steps=max(1, rep.own_rows), n_envs=1,
                              opp_class=[f.rec.opp_class for f in rep.forks if not f.dropped])
        kept = [f for f in rep.forks if not f.dropped]
        nb = sum(len(f.branches) for f in rep.forks)
        out.update({
            "rate": (len(kept) / rep.candidates) if rep.candidates else float("nan"),
            "own_rows": float(rep.own_rows),
            "asked": float(rep.asked),
            "dropped_forks": float(rep.dropped_forks),
            "rows_per_fork": float(self.rows_per_fork or 0.0),
            "row_budget": float(self.row_budget),
            "opp_real": float(sum(1 for f in kept if f.opp_real)),
            "opp_substituted": (float(sum(1 for f in kept if not f.opp_real)) / len(kept)) if kept else float("nan"),
            "branches_capped_frac": (float(sum(1 for f in rep.forks for b in f.branches if b.end and self.capped(b)))
                                     / nb) if nb else float("nan"),
        })
        return out

    def record(self, model: Any, rep: PassReport) -> None:
        logger = getattr(model, "_logger", None)
        m = self.metrics(rep)
        model._fork_metrics = m
        if logger is None:
            return
        for k, v in m.items():
            logger.record(f"fork/{k}", float(v))
        logger.record("supply/fork_dry_streak", float(self.guard.streak))

    def close(self) -> None:
        for h in self.handles:
            try:
                h.core.close()
            except Exception:
                pass


def fork_decl_from_args(args: Any) -> Optional[ForkDecl]:
    """The arm's declaration from the trainer's args, or None (OFF: nothing is built)."""
    frac = float(getattr(args, "fork_fraction", 0.0) or 0.0)
    if frac <= FA.FORK_OFF:
        return None
    from agents.training.lever_supply import starve_cycles_for

    return ForkDecl(fraction=frac, branches=int(getattr(args, "fork_branches", FA.DEFAULT_BRANCHES)),
                    contested_gap=float(getattr(args, "fork_contested_gap", FA.DEFAULT_CONTESTED_GAP)),
                    contested_absv=float(getattr(args, "fork_contested_absv", FA.DEFAULT_CONTESTED_ABSV)),
                    max_per_battle=int(getattr(args, "fork_max_per_battle", FA.DEFAULT_MAX_PER_BATTLE)),
                    crn=str(getattr(args, "fork_crn", FA.DEFAULT_CRN)), seed=int(getattr(args, "seed", 0) or 0),
                    starve_cycles=starve_cycles_for(args, "fork"))


def check_terminal(terminal: Any) -> None:
    """§7: a branch's reward is rebuilt from its outcome bit — only the INDICATOR terminal (the win-prob critic)
    makes that exact. Any other terminal REFUSES at startup."""
    if not bool(terminal["terminal_indicator"]):
        raise LeverConfigError("\n[SUPPLY] FATAL: the fork arm rebuilds a branch's reward from its outcome bit, "
                               "which needs the indicator terminal (the win-prob critic's win-indicator terminal)")


def check_obs_keys(obs_keys: Sequence[str]) -> None:
    """The FILL-table gate at startup (``fork_buffer.unfillable_keys``): a key no rule covers REFUSES."""
    from agents.training.fork_buffer import refusal_text, unfillable_keys

    bad = [k for k in unfillable_keys(obs_keys) if k not in (S.KEY_OPP_CLASS,)]
    if bad:
        raise LeverConfigError("\n[SUPPLY] FATAL: " + refusal_text(bad))
