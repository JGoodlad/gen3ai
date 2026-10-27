"""The LEAF ENGINE: batched greedy playouts of (world, first action, dice seed) requests from one root,
optionally TRUNCATED at a turn depth and scored by the critic there.

Every request names a world (a full core input log, all with the same ``at``), a first action of the
searched side and a dice seed. The requests of one world are one ``rust_env_playout_open`` (its
actions × its seed, COMMON dice across the actions of that world), and all worlds are stepped in
LOCKSTEP so every pending decision of every world goes through ONE forward per step (identical
pending rows deduplicated). Both sides play the forward's greedy action.

A leaf's value, on the WIN scale [0, 1]: the terminal outcome (1 win, 0 loss incl. a stall forfeit, ½
tie / truncated at ``max_turns``), or, at ``depth`` d, the forward's V at the searched side's first
decision whose turn ≥ root turn + d (the turn decoded from the obs clock). A branch captured at depth
is abandoned (the next open resets the handle).

THE REFEREE (:func:`referee`): the IMP metric's disjoint judge (``search_ledger.md`` §6.2), every legal action x
the referee's seeds played to the END in the TRUE world, the forward greedy on the searched side. The other side
is the forward too (the SELF-referee, the default, which every banked IMP used) or, with ``opp_bot``, the
state's real scripted bot played INSIDE the core (the ``bot`` key of ``rust_env_playout_open``; never pending).
The bot is a referee option ONLY: :class:`LeafEngine` (the search's own leaves) never takes one (E1-0,
``designs/research_state/measurements/search_e1_0_2026-10-26/``).
"""
from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ACT = 11

#: ``forward(rows [k, OBS] f32, masks [k, 11] u8) -> [k, 12]``: the 11 policy logits, then V ∈ [0, 1].
Forward = Callable[[np.ndarray, np.ndarray], np.ndarray]


def clock_offset() -> int:
    """The obs offset of the clock's ``remaining_linear`` scalar (turn = MAX_TURNS · (1 − x))."""
    from agents.observation.constants import CLOCK_OFFSET_IN_GLOBAL, OFFSET_GLOBAL

    return int(OFFSET_GLOBAL) + int(CLOCK_OFFSET_IN_GLOBAL) + 1


def decode_turn(rows: np.ndarray, off: int) -> np.ndarray:
    from agents.observation.constants import MAX_TURNS

    return np.rint(MAX_TURNS * (1.0 - rows[:, off].astype(np.float64))).astype(np.int64)


def dedup_forward(forward: Forward) -> Forward:
    """``forward`` over the DISTINCT (row, mask) pairs only."""

    def f(rows: np.ndarray, masks: np.ndarray) -> np.ndarray:
        key = np.concatenate([rows.view(np.uint8).reshape(len(rows), -1), masks.astype(np.uint8)], axis=1)
        v = np.ascontiguousarray(key).view(np.dtype((np.void, key.shape[1]))).ravel()
        _, first, inv = np.unique(v, return_index=True, return_inverse=True)
        return forward(np.ascontiguousarray(rows[first]), np.ascontiguousarray(masks[first]))[inv.ravel()]

    return f


@dataclass
class Request:
    world: int
    action: int
    seed: str


def release_lib():
    """THIS checkout's RELEASE ``libpokesim_env.so``, stamp-checked (:func:`utils.rust_env.ffi.load`: a missing, foreign
    or stale build is refused, naming the rebuild). The one place the search's engine library is chosen, so a
    preflight (``main.anchors --our-search``) and the engines load exactly the same file.
    Built by ``scripts/bootstrap.sh --with-rust``."""
    from utils.rust_env import ffi as F

    return F.load(F.default_path("release"))


class LeafEngine:
    """One handle per world slot (``n_cores``); ``forward`` serves both sides' decisions."""

    def __init__(self, forward: Forward, n_cores: int = 16, lib=None):
        from utils.rust_env.successors import SearchCore

        self.lib = lib if lib is not None else release_lib()
        self.cores = [SearchCore(lib=self.lib) for _ in range(n_cores)]
        self.forward = dedup_forward(forward)
        self.off = clock_offset()
        self.rows_forwarded = 0
        self.playouts = 0

    def close(self) -> None:
        for c in self.cores:
            c.close()

    def root(self, log: dict, at: int, side: str) -> dict:
        """The playout root (``tokens``, ``other_open``, ``turn``) without playing anything."""
        req = {"log": log, "at": int(at), "side": side, "actions": None, "seeds": [None], "stall": None,
               "max_turns": 999, "keep_cmds": False}
        return self.cores[0]._cstr(self.cores[0].lib.rust_env_playout_open, json.dumps(req).encode())

    def leaves(self, logs: Sequence[dict], at: int, side: str, root_turn: int, requests: Sequence[Request],
               depth: Optional[int] = None) -> List[float]:
        """One value per request (win scale), in request order."""
        by_world: Dict[int, List[int]] = {}
        for i, r in enumerate(requests):
            by_world.setdefault(r.world, []).append(i)
        out = [float("nan")] * len(requests)
        worlds = sorted(by_world)
        for c0 in range(0, len(worlds), len(self.cores)):
            part = worlds[c0:c0 + len(self.cores)]
            self._lockstep([(self.cores[k], logs[w], [requests[i] for i in by_world[w]], by_world[w])
                            for k, w in enumerate(part)], at, side, root_turn, depth, out)
        return out

    def _lockstep(self, jobs, at, side, root_turn, depth, out) -> None:
        from utils.rust_env.successors import SuccessorsError, _raise, production_stall

        me = 0 if side == "p1" else 1
        st = []
        for core, log, reqs, idx in jobs:
            seeds = sorted({r.seed for r in reqs})
            if len(seeds) != 1:
                raise SuccessorsError("one dice seed per world per call")
            acts = sorted({r.action for r in reqs})
            req = {"log": log, "at": int(at), "side": side, "actions": acts, "seeds": seeds,
                   "stall": production_stall(side), "max_turns": 999, "keep_cmds": False}
            root = core._cstr(core.lib.rust_env_playout_open, json.dumps(req).encode())
            nb = int(root["branches"])
            cap = 2 * nb
            st.append({"core": core, "nb": nb, "cap": cap, "n": 0, "live": True, "idx": idx, "reqs": reqs,
                       "rows": np.empty((cap, core.obs_dim), dtype=np.float32),
                       "masks": np.empty((cap, ACT), dtype=np.uint8), "who": np.empty(cap, dtype=np.uint32),
                       "acts": np.zeros(cap, dtype=np.int32), "val": [None] * nb})
            self.playouts += nb
        while True:
            pend: List[Tuple[dict, int]] = []
            for s in st:
                if not s["live"]:
                    continue
                core = s["core"]
                with core._lock:
                    k = core.lib.rust_env_playout_step(core._h, s["acts"].ctypes.data, s["n"], s["rows"].ctypes.data,
                                                       s["masks"].ctypes.data, s["who"].ctypes.data, s["cap"])
                    if k == ctypes.c_size_t(-1).value:
                        _raise(core.lib)
                if k == 0:
                    s["live"] = False
                    continue
                pend.append((s, int(k)))
            if not pend:
                break
            res = self.forward(np.concatenate([s["rows"][:k] for s, k in pend]),
                               np.concatenate([s["masks"][:k] for s, k in pend]))
            self.rows_forwarded += len(res)
            o = 0
            for s, k in pend:
                r = res[o:o + k]
                o += k
                lg = np.where(s["masks"][:k].astype(bool), r[:, :ACT].astype(np.float64), -np.inf)
                s["acts"][:k] = lg.argmax(axis=1).astype(np.int32)
                if depth is not None:
                    mine = np.flatnonzero((s["who"][:k] % 2) == me)
                    if len(mine):
                        turns = decode_turn(s["rows"][mine], self.off)
                        for t, i in zip(turns, mine):
                            b = int(s["who"][i] // 2)
                            if s["val"][b] is None and t >= root_turn + depth:
                                s["val"][b] = float(r[i, ACT])
                    if all(v is not None for v in s["val"]):
                        s["live"] = False      # every branch captured: abandon the rest
                s["n"] = k
        for s in st:
            res = s["core"]._cstr(s["core"].lib.rust_env_playout_results)
            for b, br in enumerate(res["branches"]):
                v = s["val"][b]
                if v is None:
                    e = br["end"]
                    if e is None:
                        raise SuccessorsError("a branch neither captured nor ended")
                    v = 0.5 if (e["truncated"] or e["winner"] is None) else (1.0 if int(e["winner"]) == me else 0.0)
                s["val"][b] = v
                for i, rq in zip(s["idx"], s["reqs"]):
                    if rq.action == int(br["action"]):
                        out[i] = v


@dataclass(frozen=True)
class OppBot:
    """The REFEREE's opponent as an in-core scripted bot: ``name`` a ``crate::bots::Kind`` name (``random``,
    ``heuristic``, ``heuristic2``, ``staller``, ``staller_v2``, ``aggressive``, ``aggressive_v2``, ``setup_sweep``,
    ``setup_sweep_v2``) and ``seed`` the route seed of its dice streams."""

    name: str
    seed: int


@dataclass
class RefereeResult:
    """``out[a]``: one outcome character per seed, in seed order (``+`` win, ``-`` loss incl. a stall forfeit,
    ``0`` tie or truncated at ``max_turns``), the banked truth rows' encoding. ``fwd_rows[side]``: the rows the
    forward answered per side (0 = p1, 1 = p2); with ``opp_bot`` the bot's side is never forwarded."""

    out: Dict[int, str]
    tokens: Dict[int, str]
    other_open: bool
    fwd_rows: List[int]


def referee(forward: Forward, log: dict, at: int, side: str, seeds: Sequence[str], *,
            actions: Optional[Sequence[int]] = None, opp_bot: Optional[OppBot] = None, lib=None,
            max_turns: int = 999) -> RefereeResult:
    """Play ``actions`` (default: every legal action) x ``seeds`` to the end from command ``at`` of ``log``, the
    forward greedy on ``side`` and, unless ``opp_bot`` is given (default OFF), on the other side too.

    ONE playout open PER ACTION, all stepped in lockstep through one forward per step (identical rows
    deduplicated). A branch's outcome is a function of (log, at, action, seed, policy), so the self-referee
    reproduces the single-open play of every action (the banked truth rows). With ``opp_bot`` the per-action open
    makes branch ``b`` = the seed's index, so the bot's dice streams (``stream_seed(seed, b, k)``) are COMMON
    across actions, as the battle's dice are. Production's stall forfeit applies to ``side``."""
    from utils.rust_env.successors import SearchCore, SuccessorsError, _raise, production_stall

    lib = lib if lib is not None else release_lib()
    fwd = dedup_forward(forward)
    me = 0 if side == "p1" else 1
    bot = None if opp_bot is None else {"side": "p2" if me == 0 else "p1", "name": str(opp_bot.name),
                                        "seed": int(opp_bot.seed)}
    probe = SearchCore(lib=lib)
    try:
        req = {"log": log, "at": int(at), "side": side, "actions": None, "seeds": [None], "stall": None,
               "max_turns": int(max_turns), "keep_cmds": False}
        root = probe._cstr(probe.lib.rust_env_playout_open, json.dumps(req).encode())
    finally:
        probe.close()
    tokens = {int(k): v for k, v in root["tokens"].items()}
    acts = sorted(tokens) if actions is None else [int(a) for a in actions]
    seeds = list(seeds)
    st = []
    fwd_rows = [0, 0]
    try:
        for a in acts:
            core = SearchCore(lib=lib)
            st.append({"core": core, "a": a, "live": True, "n": 0})
            req = {"log": log, "at": int(at), "side": side, "actions": [a], "seeds": seeds,
                   "stall": production_stall(side), "max_turns": int(max_turns), "keep_cmds": False}
            if bot is not None:
                req["bot"] = bot
            r = core._cstr(core.lib.rust_env_playout_open, json.dumps(req).encode())
            cap = 2 * int(r["branches"])
            st[-1].update({"cap": cap, "rows": np.empty((cap, core.obs_dim), dtype=np.float32),
                           "masks": np.empty((cap, ACT), dtype=np.uint8), "who": np.empty(cap, dtype=np.uint32),
                           "acts": np.zeros(cap, dtype=np.int32)})
        while True:
            pend = []
            for s in st:
                if not s["live"]:
                    continue
                core = s["core"]
                with core._lock:
                    k = core.lib.rust_env_playout_step(core._h, s["acts"].ctypes.data, s["n"], s["rows"].ctypes.data,
                                                       s["masks"].ctypes.data, s["who"].ctypes.data, s["cap"])
                    if k == ctypes.c_size_t(-1).value:
                        _raise(core.lib)
                if k == 0:
                    s["live"] = False
                    continue
                pend.append((s, int(k)))
            if not pend:
                break
            res = fwd(np.concatenate([s["rows"][:k] for s, k in pend]),
                      np.concatenate([s["masks"][:k] for s, k in pend]))
            o = 0
            for s, k in pend:
                r = res[o:o + k]
                o += k
                for q in (0, 1):
                    fwd_rows[q] += int(np.count_nonzero((s["who"][:k] % 2) == q))
                lg = np.where(s["masks"][:k].astype(bool), r[:, :ACT].astype(np.float64), -np.inf)
                s["acts"][:k] = lg.argmax(axis=1).astype(np.int32)
                s["n"] = k
        out: Dict[int, str] = {}
        for s in st:
            res = s["core"]._cstr(s["core"].lib.rust_env_playout_results)
            ch = [None] * len(seeds)
            for br in res["branches"]:
                if int(br["action"]) != s["a"]:
                    raise SuccessorsError(f"a branch of action {br['action']} in action {s['a']}'s open")
                e = br["end"]
                if e is None:
                    raise SuccessorsError("a referee branch did not end")
                ch[int(br["seed"])] = ("0" if (e["truncated"] or e["winner"] is None)
                                       else "+" if int(e["winner"]) == me else "-")
            if any(c is None for c in ch):
                raise SuccessorsError(f"action {s['a']} is missing a seed")
            out[s["a"]] = "".join(ch)
    finally:
        for s in st:
            s["core"].close()
    return RefereeResult(out=out, tokens=tokens, other_open=bool(root.get("other_open")), fwd_rows=fwd_rows)
