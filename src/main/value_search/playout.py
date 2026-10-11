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


class LeafEngine:
    """One handle per world slot (``n_cores``); ``forward`` serves both sides' decisions."""

    def __init__(self, forward: Forward, n_cores: int = 16, lib=None):
        from utils.rust_env import ffi as F
        from utils.rust_env.successors import SearchCore

        self.lib = lib if lib is not None else F.load(F.default_path("release"))
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
