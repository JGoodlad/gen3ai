"""SEARCH ON ``successors()``, IN PROCESS — the Python half of M5 Lane I
(``designs/endstate/program_rust_core.md`` §2 M5; the Rust half is ``src/rust_env/src/search/``).

Two consumers, one FFI handle (the env core's ``cdylib``, loaded and stamp-checked by
:func:`utils.rust_env.ffi.load` — the rows ``rust_env_search_*`` / ``rust_env_playout_*`` of
``ffi.FUNCTIONS``):

* :class:`Successors` — a DROP-IN for :class:`utils.bridge.search_session.SearchSession` on the CORE
  road (``open_root(core="text", trackers=True)`` / ``expand_many(rows=True)``): the same
  :class:`RootView` / :class:`ExpandedNode`, the same node ids, but no child process and no JSON
  row: each leaf's ENCODED row lands in a NumPy buffer (``core_pN["row"]`` is a read-only float32
  ``(OBS_DIM,)`` array, not a base64 wire frame). Gate: ``successors_parity_integration_test.py``
  (the depth-3 successor slice, byte-equal to the ``search_driver`` binary's rows).
* :func:`play_out` — Lane S's ground truth: from a decision of a banked core INPUT LOG, branch every
  legal action of one side × a set of dice seeds (COMMON RANDOM NUMBERS across siblings), and play
  each branch to the end with a caller-supplied POLICY (``policy(rows, masks, who) -> actions``,
  batched over every live branch; :func:`greedy` wraps a scorer). Rows come from the TRAINING
  observation path (program §6c).

The GIL is released during every foreign call (``ctypes``). One handle is single-caller; a caller
that wants parallelism holds one :class:`SearchCore` per thread.

Errors: a refused request, an unknown node or a battle the port refuses raises
:class:`SuccessorsError` (a :class:`SearchError`, so :mod:`main.search_dividend.search`'s handlers
catch it as they catch the JSON road's) and does NOT poison the handle; a Rust panic raises
:class:`protocol.CorePanic` and poisons it (every later call: :class:`protocol.LifecycleViolation`).
"""
from __future__ import annotations

import ctypes
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from utils.bridge.search_session import ElidedSide, ExpandedNode, RootView, SearchError
from utils.rust_env import ffi as F
from utils.rust_env import protocol as P

ACT = 11

#: The declared capacities of a default handle (the spec's `max_nodes` / `max_branches`).
DEFAULT_MAX_NODES = 250_000
DEFAULT_MAX_BRANCHES = 4_096


class SuccessorsError(SearchError):
    """A search / playout request the in-process core refused (the handle stays usable)."""


def spec_json(*, decision_tense: Optional[bool] = None, switch_freeze: Optional[bool] = None,
              max_nodes: int = DEFAULT_MAX_NODES, max_branches: int = DEFAULT_MAX_BRANCHES) -> str:
    """A search handle's STARTUP declaration. The clock flags default to the PRODUCTION config's
    (``progress_decision_tense`` / ``progress_switch_freeze``; both OFF, which is also what
    ``search_driver`` always uses)."""
    if decision_tense is None or switch_freeze is None:
        from agents.training.baselines import production_config

        pc = production_config()
        decision_tense = bool(pc.get("progress_decision_tense", False)) if decision_tense is None else decision_tense
        switch_freeze = bool(pc.get("progress_switch_freeze", False)) if switch_freeze is None else switch_freeze
    return json.dumps({"clock": {"decision_tense": bool(decision_tense), "switch_freeze": bool(switch_freeze)},
                       "max_nodes": int(max_nodes), "max_branches": int(max_branches)})


def _raise(lib: ctypes.CDLL) -> None:
    text = lib.rust_env_last_error()
    err = P.error_from_json(text.decode()) if text else P.error_for(P.STATUSES[1].code, "no error recorded")
    if isinstance(err, P.CallerError) and '"kind":"search"' in (text or b"").decode():
        raise SuccessorsError(str(err)) from err
    raise err


class SearchCore:
    """One in-process search handle (``rust_env_search_new``)."""

    def __init__(self, spec: Optional[str] = None, *, lib: Optional[ctypes.CDLL] = None,
                 path: Optional[Path] = None, nan_poison: Optional[bool] = None):
        self.lib = lib if lib is not None else F.load(path or F.default_path(), nan_poison=nan_poison)
        self.obs_dim = int(self.lib.rust_env_obs_dim())
        self._lock = threading.Lock()
        self._h = self.lib.rust_env_search_new((spec or spec_json()).encode())
        if not self._h:
            _raise(self.lib)

    def _cstr(self, fn, *args) -> dict:
        with self._lock:
            if not self._h:
                raise P.LifecycleViolation("a closed SearchCore")
            out = fn(self._h, *args)
            if out is None:
                _raise(self.lib)
            return json.loads(out.decode())

    def stats(self) -> dict:
        return self._cstr(self.lib.rust_env_search_stats)

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_h", None):
                self.lib.rust_env_search_free(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ================================================================== the search tree


class Successors(SearchCore):
    """``SearchSession``'s core road, in process (see the module header). ``record`` is the default
    battle for :meth:`open_root`, exactly as ``SearchSession(record)``."""

    impl = "inproc"

    def __init__(self, record=None, *, spec: Optional[str] = None, lib: Optional[ctypes.CDLL] = None,
                 path: Optional[Path] = None, nan_poison: Optional[bool] = None, chunks: bool = True):
        super().__init__(spec, lib=lib, path=path, nan_poison=nan_poison)
        self._record = record
        #: Render ``pN_chunks`` on every arm (the JSON road does; a core-road reader never reads them).
        self.chunks = chunks

    def open_root(self, turn: int, *, record=None, core: Optional[str] = "text",
                  side: Optional[str] = None, trackers: bool = True) -> RootView:
        rec = record if record is not None else self._record
        if rec is None:
            raise SuccessorsError("open_root needs a record (pass record= or construct with one)")
        req: dict = {"cmd": "open_root", "record": rec.to_dict() if hasattr(rec, "to_dict") else rec,
                     "turn": int(turn), "trackers": bool(trackers)}
        if core is not None:
            req["core"] = core
        if side is not None:
            req["side"] = side
        out = self._cstr(self.lib.rust_env_search_open_root, json.dumps(req).encode())
        return RootView(node_id=out["node_id"], requests=out["requests"], recorded_choices=out["recorded_choices"],
                        pre_state=out["pre_state"], prefix_p1_chunks=out["prefix_p1_chunks"],
                        prefix_p2_chunks=out["prefix_p2_chunks"])

    def expand_many(self, arms: Sequence[dict], *, side: Optional[str] = None, rows: bool = True) -> List[ExpandedNode]:
        if not rows:
            raise SuccessorsError("the in-process road serves ROWS only (expand_many(rows=True))")
        req: dict = {"cmd": "expand_many", "arms": [dict(a) for a in arms], "rows": True, "chunks": self.chunks}
        if side is not None:
            if side not in ("p1", "p2"):
                raise SuccessorsError(f"expand_many: side must be 'p1' or 'p2', got {side!r}")
            req["side"] = side
        cap = len(arms) * (1 if side else 2)
        buf = np.empty((max(cap, 1), self.obs_dim), dtype=np.float32)
        out = self._cstr(self.lib.rust_env_search_expand, json.dumps(req).encode(),
                         buf.ctypes.data if cap else None, cap)
        buf.flags.writeable = False

        def core(a: dict, key: str):
            c = a.get(key)
            if c is None:
                return None
            if c.get("row") is not None:
                c = dict(c)
                c["row"] = buf[int(c["row"])]
            return c

        def chunks_of(a: dict, key: str, elided: bool):
            if key in a:
                return a[key] or []
            return ElidedSide(key) if elided else []

        return [
            ExpandedNode(label=a.get("label"), node_id=a.get("node_id"), ended=bool(a.get("ended")),
                         stuck=bool(a.get("stuck")), outcome=a.get("outcome") or {}, requests=a.get("requests"),
                         choices_used=a.get("choices_used") or {},
                         p1_chunks=chunks_of(a, "p1_chunks", side == "p2"),
                         p2_chunks=chunks_of(a, "p2_chunks", side == "p1"),
                         core_p1=core(a, "core_p1"), core_p2=core(a, "core_p2"))
            for a in out["arms"]
        ]


# ================================================================== playouts


def record_to_log(record) -> dict:
    """A reconstruction record (``ReconstructionRecord`` or its dict) as a core INPUT LOG — the
    shape :func:`play_out` takes. ``forcelose`` entries become ``FORCELOSE pN``."""
    d = record.to_dict() if hasattr(record, "to_dict") else record
    start = None
    players: dict = {}
    for line in d["input_log"]:
        if line.startswith(">start "):
            start = json.loads(line[len(">start "):])
        elif line.startswith(">player "):
            s, _, payload = line[len(">player "):].partition(" ")
            players[s] = json.loads(payload)
    if start is None or set(players) != {"p1", "p2"}:
        raise SuccessorsError("record_to_log: the record needs a >start and two >player lines")
    seed = start["seed"]
    if isinstance(seed, list):
        seed = ",".join(str(int(x)) for x in seed)
    cmds = [f"FORCELOSE {p}" if s == "forcelose" else f"CHOOSE {s} {p}" for s, p in d["commands"]]
    return {"format_id": start["formatid"], "seed": seed, "names": [players["p1"]["name"], players["p2"]["name"]],
            "teams": [players["p1"]["team"], players["p2"]["team"]], "cmds": cmds}


def log_to_record(log: dict, *, battle_tag: Optional[str] = None):
    """A core input log as a :class:`ReconstructionRecord` (what the search tree / ``search_driver``
    open a root from)."""
    from utils.bridge.reconstruction import ReconstructionRecord

    start = {"formatid": log["format_id"], "seed": log["seed"]}
    input_log = (f">start {json.dumps(start)}",
                 f">player p1 {json.dumps({'name': log['names'][0], 'team': log['teams'][0]})}",
                 f">player p2 {json.dumps({'name': log['names'][1], 'team': log['teams'][1]})}")
    cmds = []
    for c in log["cmds"]:
        if c.startswith("FORCELOSE "):
            cmds.append(("forcelose", c.split(" ", 1)[1]))
        else:
            _, side, tok = c.split(" ", 2)
            cmds.append((side, tok))
    return ReconstructionRecord(format_id=log["format_id"], prng_seed=log["seed"], input_log=input_log,
                                commands=tuple(cmds), battle_tag=battle_tag)


#: ``policy(rows (k, OBS_DIM) f32, masks (k, 11) u8, who (k,) u32 = 2*branch+side) -> k action indices``.
Policy = Callable[[np.ndarray, np.ndarray, np.ndarray], Sequence[int]]


def greedy(scorer: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Policy:
    """A GREEDY policy over a scorer returning ``(k, 11)`` logits: the legal argmax (ties to the
    lowest index). T2's ``score()`` is the intended scorer; any batched forward works."""

    def policy(rows: np.ndarray, masks: np.ndarray, who: np.ndarray) -> np.ndarray:
        logits = np.asarray(scorer(rows, masks), dtype=np.float64)
        logits = np.where(masks.astype(bool), logits, -np.inf)
        return logits.argmax(axis=1).astype(np.int32)

    return policy


def uniform_random(rng: np.random.Generator) -> Policy:
    """A seeded uniform-over-the-mask policy (a harness policy, not a continuation anyone scores)."""

    def policy(rows: np.ndarray, masks: np.ndarray, who: np.ndarray) -> np.ndarray:
        out = np.empty(len(masks), dtype=np.int32)
        for i, m in enumerate(masks):
            legal = np.flatnonzero(m)
            out[i] = legal[rng.integers(len(legal))]
        return out

    return policy


@dataclass
class PlayoutResult:
    side: str
    at: int
    #: One dict per branch: ``action``, ``seed`` (index), ``reseed`` (the seed string or None),
    #: ``end`` {winner (0/1/None), forfeit, truncated, turn}, ``decisions`` [p1, p2], ``cmds``.
    branches: List[dict]
    #: Policy decisions answered, forward batches, and the wall of the whole playout.
    answered: int = 0
    batches: int = 0
    wall_s: float = 0.0
    tokens: Dict[int, str] = field(default_factory=dict)

    def value(self, b: dict, *, tie: float = 0.0, truncated: float = 0.0) -> float:
        """One branch's value from ``side``'s view: +1 win, -1 loss (a stall forfeit included), ``tie``
        on a tie, ``truncated`` at ``max_turns``."""
        e = b["end"]
        if e["truncated"]:
            return truncated
        if e["winner"] is None:
            return tie
        return 1.0 if e["winner"] == (0 if self.side == "p1" else 1) else -1.0

    def values(self, **kw) -> Dict[int, float]:
        """The mean value per first action over its seeds (every action saw the SAME seeds — CRN)."""
        acc: Dict[int, List[float]] = {}
        for b in self.branches:
            acc.setdefault(int(b["action"]), []).append(self.value(b, **kw))
        return {a: float(np.mean(v)) for a, v in acc.items()}


def play_out(log: dict, at: int, side: str, *, policy: Policy, seeds: Sequence[Optional[str]],
             actions: Optional[Sequence[int]] = None, stall: Optional[dict] = "production",
             max_turns: int = 999, keep_cmds: bool = False, core: Optional[SearchCore] = None) -> PlayoutResult:
    """Branch ``side``'s decision at command ``at`` of ``log`` — every legal action (or ``actions``)
    × every seed — and play each branch to the end under ``policy``. ``seeds``: one per dice draw,
    SHARED by every action (common random numbers); ``None`` keeps the battle's own dice.
    ``stall="production"`` is training's stall forfeit for ``side`` (``StallConfig().threshold``);
    ``None`` disables it (then ``max_turns`` < 1000 bounds the battle)."""
    import time

    if stall == "production":
        from agents.training.stall import StallConfig

        stall = {"turn_limit": int(StallConfig().threshold), "side": side}
    own = core is None
    core = core or SearchCore()
    try:
        t0 = time.monotonic()
        req = {"log": log, "at": int(at), "side": side, "actions": None if actions is None else [int(a) for a in actions],
               "seeds": list(seeds), "stall": stall, "max_turns": int(max_turns), "keep_cmds": bool(keep_cmds)}
        root = core._cstr(core.lib.rust_env_playout_open, json.dumps(req).encode())
        cap = 2 * int(root["branches"])
        rows = np.empty((cap, core.obs_dim), dtype=np.float32)
        masks = np.empty((cap, ACT), dtype=np.uint8)
        who = np.empty(cap, dtype=np.uint32)
        acts = np.zeros(cap, dtype=np.int32)
        n, answered, batches = 0, 0, 0
        while True:
            with core._lock:
                k = core.lib.rust_env_playout_step(core._h, acts.ctypes.data, n, rows.ctypes.data, masks.ctypes.data,
                                                   who.ctypes.data, cap)
                if k == ctypes.c_size_t(-1).value:
                    _raise(core.lib)
            if k == 0:
                break
            a = np.asarray(policy(rows[:k], masks[:k], who[:k]), dtype=np.int32)
            if a.shape != (k,):
                raise SuccessorsError(f"the policy returned shape {a.shape} for {k} pending decisions")
            acts[:k] = a
            n = k
            answered += k
            batches += 1
        res = core._cstr(core.lib.rust_env_playout_results)
        return PlayoutResult(side=res["side"], at=res["at"], branches=res["branches"], answered=answered,
                             batches=batches, wall_s=time.monotonic() - t0,
                             tokens={int(k): v for k, v in root["tokens"].items()})
    finally:
        if own:
            core.close()
