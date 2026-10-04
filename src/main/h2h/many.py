"""MANY CELLS ON ONE ENGINE: the multi-cell offline head-to-head (X5 build unit U0's cross; eval build unit U6).

``main.h2h play-many`` — the same edges ``main.h2h play`` plays, many at a time. Why: a single-cell run builds a
fresh engine per edge, and on the GPU the T2 graphs compile and capture for every engine — 114–167 s per edge against
~17 s of play per 1,000-pair cell (F-P0-4, ``designs/research_state/measurements/x5_p0_h2h_2026-10-03/``). The X5 A/B
needs 9 / 25 / 64 cells per look, and the cycle monitor and the plateau check need many edges per check.

WHAT IS REUSED, EXACTLY. One :class:`main.h2h.play.H2HEngine` — one T2 service with its two declared slots, one eval
core — serves every cell; a cell only changes which checkpoints the next cycles LOAD into those slots
(``H2HEngine.set_cell``). Everything that decides a game is the single-cell tool's, unchanged: the plan per cell
(``play.plan_edge``: the request, the batches recorded, the refusal of another batch size), the per-batch play and
row (``play.play_planned``: the claim, ``cycle_seed(schedule seed, the cell's schedule key, batch)``, the scoring, the
§0b v2 row, protocol ``gen3_eval_protocol_v1_h2h``). STORAGE AND ENGINE REUSE ONLY: the games of a cell are
byte-identical to single-cell ``main.h2h play`` on the same seeds — proved by the outcome digests
(``play_many_integration_test.py`` and ``designs/research_state/measurements/h2h_multicell_2026-10-04/``).

HOT-SWAP SAFETY. Every cycle loads both slots through ``InferenceService.load`` (an in-place copy, the bit-exact copy
check, the parity gate at every bucket the slot serves, the copy check again; a failure POISONS the service), and
after every cycle the engine checks both slots bit-exact against THIS cell's checkpoints (``H2HEngine.verify_slots``)
— a slot never carries the previous cell's weights into a row.

ONE ARCHITECTURE PER ENGINE. Every side of every cell to play is checked against the first cell's player
(``play.EngineArch``: toggles, model version, served state-dict signature, forward fingerprint, and the player's
terminal) in a PRE-FLIGHT, before any engine is built or any game played: a foreign-architecture cell is a typed
:class:`main.h2h.play.CellArchMismatch` naming both. (So an X5 ``fixed_mass``-vs-``blob`` cell — two architectures —
is refused here exactly as single-cell ``play`` refuses it; FINDING F-U6-1.)

RESUMABLE. Each cell's rows resume as a single-cell edge's do (the request's uniqueness), so a re-run skips every
recorded batch of every cell and builds no engine at all when nothing is left to play.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agents.training import eval_ledger as L
from main.h2h import play as PL

Cell = Tuple[PL.PlayerRef, PL.PlayerRef]

#: The per-cell summary read: THIS cell's rows in its request, at the h2h protocol — ``play``'s own resume read.
MANY_CELL = L.ReaderDecl(
    name="main.h2h.many.cell", purposes=L.ALL_PURPOSES,
    regime=L.RegimeFilter(protocol=PL.PROTOCOL, play="greedy", opponent_play="greedy", mirrored=True),
    requests="own", selection="include", flags_ok=frozenset(), inference="conditional")


def cells_from_file(path: str) -> List[Tuple[str, str]]:
    """A JSON list of cells, each ``[player, opponent]`` or ``{"player": ..., "opponent": ...}`` (the spec forms of
    ``main.h2h play``)."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list) or not raw:
        raise PL.H2HError(f"{path}: a cell file is a non-empty JSON list")
    out = []
    for i, c in enumerate(raw):
        if isinstance(c, dict) and set(c) >= {"player", "opponent"}:
            out.append((str(c["player"]), str(c["opponent"])))
        elif isinstance(c, (list, tuple)) and len(c) == 2:
            out.append((str(c[0]), str(c[1])))
        else:
            raise PL.H2HError(f"{path}: cell {i} is {c!r}, not [player, opponent] / {{player, opponent}}")
    return out


def cross(players: Sequence[str], opponents: Sequence[str]) -> List[Tuple[str, str]]:
    """Every player against every opponent, players outer (the X5 cross: X5 seed i against blob seed j)."""
    return [(p, o) for p in players for o in opponents]


def resolve_cells(specs: Sequence[Tuple[str, str]]) -> List[Cell]:
    """Resolve every spec once (a checkpoint named twice is hashed once); REFUSES a cell listed twice."""
    memo: Dict[str, PL.PlayerRef] = {}

    def ref(s: str) -> PL.PlayerRef:
        if s not in memo:
            memo[s] = PL.resolve_player(s)
        return memo[s]

    cells = [(ref(p), ref(o)) for p, o in specs]
    seen: Dict[Tuple[str, str], int] = {}
    for i, (p, o) in enumerate(cells):
        k = (p.sha256, o.sha256)
        if k in seen:
            raise PL.H2HError(f"cell {i} ({p.id} vs {o.id}) repeats cell {seen[k]}: one cell, one plan")
        seen[k] = i
    return cells


def preflight(cells: Sequence[Cell], emit: Callable[[str], None] = lambda _m: None) -> PL.EngineArch:
    """Check EVERY side of every cell against the first cell's player BEFORE any engine exists: the deleted core
    variants and the team source (as ``play`` does) and the architecture (:class:`main.h2h.play.EngineArch`). Loads
    each distinct checkpoint once on the CPU and drops it."""
    if not cells:
        raise PL.H2HError("no cell to check")
    t0 = time.perf_counter()
    sides: Dict[str, Tuple[PL.PlayerRef, set]] = {}
    for p, o in cells:
        sides.setdefault(p.sha256, (p, set()))[1].add("player")
        sides.setdefault(o.sha256, (o, set()))[1].add("opponent")
    arch: Optional[PL.EngineArch] = None
    for ref, roles in sides.values():                 # dict order: the first cell's player first
        PL.check_core_flags(ref)
        PL.check_team_source(ref)
        model = PL._load_host(ref)
        if arch is None:
            arch = PL.EngineArch.of(ref, model)
        for side in sorted(roles, reverse=True):      # "player" (the terminal binds it) before "opponent"
            arch.check(ref, model, side)
        del model
    assert arch is not None
    emit(f"[h2h] pre-flight: {len(cells)} cell(s), {len(sides)} checkpoint(s) — one architecture "
         f"(declared from {arch.source}) in {time.perf_counter() - t0:.1f}s")
    return arch


@dataclass
class CellTiming:
    cell: str
    set_cell_s: float
    play_s: float
    load_s: float
    rows: int


@dataclass
class EngineReport:
    startup_s: Optional[float] = None
    preflight_s: Optional[float] = None
    cells: List[CellTiming] = field(default_factory=list)

    def block(self) -> Dict[str, Any]:
        return {"startup_s": self.startup_s, "preflight_s": self.preflight_s,
                "cells": [vars(c) for c in self.cells]}


def play_cells(out_dir: Optional[str], cells: Sequence[Cell], *, pairs: int,
               batch_pairs: int = PL.DEFAULT_BATCH_PAIRS, schedule_seed: int = 0, schedule_key: Optional[str] = None,
               purpose: str = PL.DEFAULT_PURPOSE, run_label: str, compute: PL.Compute,
               request_id: Optional[str] = None, family: Optional[str] = None, request_kind: Optional[str] = None,
               emit: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)) -> Dict[str, Any]:
    """Play ``pairs`` mirrored pairs of every cell on ONE engine, each cell exactly as :func:`main.h2h.play.play_edge`
    plays it (module docs). ``request_id`` = ONE request for every cell (an X5 LOOK: ``--purpose ab --family F
    --request <look>``); ``None`` = each cell its own default request (a re-run of single-cell ``play`` on the
    same arguments resumes the same rows). Returns ``{"cells": [edge summary per cell], "engine": timings}``."""
    from main.h2h import stats as ST
    from utils.rust_env import episode as EP

    if not cells:
        raise PL.H2HError("no cells")
    root, writer = PL.open_writer(out_dir, purpose)
    regime = PL.regime_for(EP.stall_threshold())
    plans = [PL.plan_edge(writer, root, regime, p, o, pairs=pairs, batch_pairs=batch_pairs,
                          schedule_seed=schedule_seed, schedule_key=schedule_key, purpose=purpose,
                          request_id=request_id, family=family, request_kind=request_kind) for p, o in cells]
    todo = [ep for ep in plans if ep.todo]
    emit(f"[h2h] play-many: {len(plans)} cell(s) x {pairs} pairs in batches of {batch_pairs}; {len(todo)} with "
         f"batches to play ({sum(len(ep.todo) for ep in plans)} batch(es)), "
         f"{sum(len(ep.done) for ep in plans)} already recorded; purpose {purpose}"
         f"{f', family {family}' if family else ''}{f', request {request_id}' if request_id else ''} under {root}")
    rep = EngineReport()
    if todo:
        t0 = time.perf_counter()
        preflight([(ep.player, ep.opponent) for ep in todo], emit)
        rep.preflight_s = round(time.perf_counter() - t0, 3)
        commit = PL.current_commit()
        with PL.engine_lock(compute):
            eng = PL.H2HEngine(todo[0].player, todo[0].opponent, compute, emit)
            rep.startup_s = round(eng.startup_s, 3)
            try:
                if eng.regime["regime_id"] != regime["regime_id"]:
                    raise PL.H2HError("the engine's regime differs from the one planned")
                for i, ep in enumerate(todo):
                    swap = eng.set_cell(ep.player, ep.opponent) if i else 0.0
                    emit(f"[h2h] cell {i + 1}/{len(todo)}: {ep.player.id} vs {ep.opponent.id} — {len(ep.todo)} "
                         f"batch(es) to play, {len(ep.done)} recorded; request {ep.rid}; swap {swap:.2f}s")
                    load0, t1 = float(eng.svc.load_seconds), time.perf_counter()
                    wrote = PL.play_planned(eng, writer, ep, schedule_seed=schedule_seed, purpose=purpose,
                                            run_label=run_label, commit=commit, emit=emit)
                    rep.cells.append(CellTiming(cell=f"{ep.player.id} vs {ep.opponent.id}", set_cell_s=round(swap, 3),
                                                play_s=round(time.perf_counter() - t1, 3),
                                                load_s=round(float(eng.svc.load_seconds) - load0, 3), rows=wrote))
            finally:
                eng.close()
    writer.close()
    out = []
    for ep in plans:
        got = L.read(MANY_CELL, root=root, request_id=ep.rid, regime_id=regime["regime_id"],
                     players=[ep.player.sha256], opponents=[ep.opponent.sha256])
        out.append(ST.edge_summary(list(got.rows)))
    return {"cells": out, "engine": rep.block()}
