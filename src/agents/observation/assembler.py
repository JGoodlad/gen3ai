"""``ObsAssembler`` — the incremental observation cache (`gen3_obs_assembler_v1`).

Stage B of `designs/ai_v9/design_incremental_obs_encoder.md`. The census there measured that
**~95% of the 2,501 obs dims are static-per-episode, reveal-monotone or event-sparse**, yet every
decision rebuilt all of them: 12 per-mon slot encodes, 36 pair cells, ≤32 event rows, every
sub-encoder's small-array allocations. This module keeps the emitted vector in a persistent
buffer and re-derives only what an event (or the request) says has moved.

**Byte-identity is the whole contract.** There is no flag and no second implementation: the same
per-block *writers* run either way, and this object only decides *which* of them run. The
`state_encoder.encode` full path is the cold path and the oracle; `GEN3AI_OBS_VERIFY=1` runs both
per decision and raises on the first differing index.

What makes the invalidation sound
---------------------------------
Three signals, and it takes all three — each is named here because each one is a hole the other
two do not cover:

1. **The event log** (`strict.events_since`). `MESSAGE_POLICY`'s `STATE_ONLY` bucket is empty in
   gen3ou — every state-mutating protocol line is also battle content and emits a `BattleEvent` —
   and an unclassified keyword RAISES. So no *protocol* mutation can bypass this.
2. **The request** (`strict.request_change_seq`). A `|request|` is NOT a protocol line in that
   sense: it emits no event, yet `Pokemon.update_from_request` writes active/ability/condition/
   item/details/moves/stats. It is the one door the event stream does not cover, and it is
   per-mon rather than global precisely so a bench mon can stay cached (a request arrives every
   decision, so a coarse signal would dirty our whole side every decision and give the cache
   back). See `Gen3Battle._diff_request_side` for why an unchanged record proves no mutation.
3. **The HP tracker's `revision`.** `HiddenPowerTracker` narrowing writes 17 dims of an opponent
   slot and is driven by our own code, not by a line. Its narrowing IS event-triggered today, so
   an event-only rule would *usually* be right — and "usually right" is the shape of a silent
   staleness bug, so the tracker states it instead.

Everything else is recomputed unconditionally, because it is either cheap or because caching it
is where correctness dies: the two 58-dim active contexts (a switch clears boosts/volatiles with
**no per-field event**, and a Baton Pass *keeps* them — so "write zeros on SWITCH" is wrong in
both directions), the global block, the board block, the 180-dim pair history (every cell's
recency ticks on every turn anyway), the per-mon recency triplets, and the trapped/active bits.

Two whole-log folds that used to run per encode are incremental here — the pending-Wish belief
and the sleep-source map. Both were linear scans of the battle log on every decision, i.e.
O(turns²) over a game, and both are pure functions of a small event family.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from agents import gen3_data
from agents.battle.battle_event import OPP, OURS, EventKind
from agents.battle.turn_view import faint_cause_id
from agents.observation.constants import (
    EVENT_COL,
    EVENT_T_MOVE,
    EVENT_TOKEN_DIM,
    EVENT_WINDOW_N,
    OFFSET_EVENT_WINDOW,
    TEAM_SIZE,
)
from agents.observation.gen3_effects import cant_reason_id
from agents.observation.sleep_belief import _reason_is_rest, _SLEEP_USABLE_MOVES
from agents.observation.wish_belief import _WISH

_gen3_movedex = gen3_data.moves

# `log1p(min(n, 10)) / log(11)` for n = 0..10 — the H-A / recency / progress-clock saturation
# curve, whose codomain has exactly 11 values. Precomputed in float64 from the identical
# expression, so a lookup is bit-for-bit what the arithmetic produced.
_SAT_CAP = 10
SAT_LUT: Tuple[float, ...] = tuple(
    math.log1p(n) / math.log(_SAT_CAP + 1.0) for n in range(_SAT_CAP + 1)
)
_SAT_LUT_F32 = np.array(SAT_LUT, dtype=np.float32)


def write_event_row(vec: np.ndarray, o: int, rec: Dict[str, Any], cur_turn: int) -> None:
    """Write ONE H-B event record into ``vec`` at flat offset ``o``.

    **The single writer for the event window**, shared by the full rebuild
    (`state_encoder.encode`) and the incremental ring below — so the two schedulers cannot
    drift in *content*, only in *when* they run, which is exactly what the byte-identity fuzz
    checks. Every column is addressed through `EVENT_COL`, never a bare integer
    (`gen3_event_col_names_v1`).
    """
    _c = EVENT_COL
    _actor = gen3_data.species.get(rec["actor"]) if rec["actor"] else None
    _target = gen3_data.species.get(rec["target"]) if rec["target"] else None
    _mv = _gen3_movedex.get(rec["move_id"]) if rec["move_id"] else None
    vec[o + _c.TYPE] = float(rec["t"])
    vec[o + _c.ACTOR_SPECIES] = float(_actor.num) if _actor is not None else 0.0
    vec[o + _c.ACTOR_SIDE] = (1.0 if rec["side"] == "ours"
                              else (-1.0 if rec["side"] == "opp" else 0.0))
    vec[o + _c.TARGET_SPECIES] = float(_target.num) if _target is not None else 0.0
    vec[o + _c.MOVE] = float(_mv.num) if _mv is not None else 0.0
    _mag = rec["hp_delta"]
    vec[o + _c.MAGNITUDE] = (max(-1.0, min(1.0, _mag)) if rec["t"] == EVENT_T_MOVE
                             else max(-1.0, min(1.0, _mag / 6.0)))
    if rec["t"] == EVENT_T_MOVE:
        vec[o + _c.OUT_HIT] = 0.0 if (rec["missed"] or rec["failed"]) else 1.0
        vec[o + _c.OUT_MISS] = 1.0 if rec["missed"] else 0.0
        vec[o + _c.OUT_FAIL] = 1.0 if rec["failed"] else 0.0
        vec[o + _c.CRIT] = 1.0 if rec["crit"] else 0.0
        # the eff one-hot is INDEXED, not written per column — `EVENT_EFF_GROUP`'s order is
        # the contract and its contiguity is asserted by event_window_test.
        vec[o + _c.EFF_NEUTRAL + int(rec["eff"])] = 1.0
    vec[o + _c.WE_FIRST] = 1.0 if rec["we_first"] else 0.0
    vec[o + _c.STATUS] = float(rec["status"])
    # `.get` because only the CANT / FAINT / ITEM branches set their key — every other record
    # type leaves it absent, which must read as a clean 0.
    vec[o + _c.CANT] = float(cant_reason_id(rec.get("cant")))
    vec[o + _c.FAINT_CAUSE] = float(faint_cause_id(rec.get("faint_cause")))
    vec[o + _c.ITEM_TRANSITION] = float(rec.get("item_tr", 0))
    vec[o + _c.TURNS_AGO] = SAT_LUT[min(max(0, cur_turn - int(rec["turn"])), _SAT_CAP)]
    vec[o + _c.FORCED_WINDOW] = float(rec["forced_window"])
    vec[o + _c.VALID] = 1.0


def _clear_event_row(vec: np.ndarray, o: int) -> None:
    vec[o:o + EVENT_TOKEN_DIM] = 0.0


# ``GEN3AI_OBS_VERIFY=1`` — shadow-encode every decision both ways and raise on the first
# difference. Read once at import (it is a launch-time switch, and a per-encode getenv on this
# path would itself be a measurable cost); tests set the module attribute directly.
OBS_VERIFY: bool = os.environ.get("GEN3AI_OBS_VERIFY") == "1"


def describe_offset(index: int) -> str:
    """Name the obs BLOCK a flat index falls in — for the verify-mode error message.

    Resolved from the declared offsets, never from literals: a diagnostic that quietly names
    the wrong block after a layout move is worse than one that names none (the positional-
    binding sweep's lesson, applied to an error path).
    """
    from agents.observation import constants as C

    if index < C.OFFSET_OPP_TEAM:
        i, off = divmod(index, C.POKEMON_FULL_DIM)
        return f"our_team[{i}] +{off}"
    if index < C.OFFSET_CONTEXT:
        i, off = divmod(index - C.OFFSET_OPP_TEAM, C.POKEMON_FULL_DIM)
        return f"opp_team[{i}] +{off}"
    if index < C.OFFSET_GLOBAL:
        i, off = divmod(index - C.OFFSET_CONTEXT, C.ACTIVE_CONTEXT_DIM)
        return f"active_context[{'ours' if i == 0 else 'opp'}] +{off}"
    if index < C.OFFSET_REACTIVE:
        return f"global +{index - C.OFFSET_GLOBAL}"
    if index < C.OFFSET_PAIR_HISTORY:
        return f"board/reactive +{index - C.OFFSET_REACTIVE}"
    if index < OFFSET_EVENT_WINDOW:
        cell, off = divmod(index - C.OFFSET_PAIR_HISTORY, C.PAIR_HISTORY_CELL_DIM)
        return f"pair_history[opp {cell // TEAM_SIZE}][our {cell % TEAM_SIZE}] +{off}"
    row, col = divmod(index - OFFSET_EVENT_WINDOW, EVENT_TOKEN_DIM)
    names = {int(c): c.name for c in C.EventCol}
    return f"event_window row {row} col {col} ({names.get(col, '?')})"


class ObsAssembler:
    """Per-episode incremental obs cache, owned by :class:`~agents.training.episode_tracker.EpisodeTracker`.

    **It lives inside the tracker on purpose** (`design_incremental_obs_encoder.md` §5.3): the
    offline materializer deep-copies the whole player graph per counterfactual arm, so a cache
    that rides the object it describes is automatically arm-consistent. A module-level or
    encoder-instance cache keyed by ``battle_tag`` would serve arm 1's forward-state bytes to a
    rewound arm 2 — that shape is unrepresentable here. The re-decide rollback
    (``EpisodeTracker.restore``) rolls back tracker state without rolling back the battle, so it
    calls :meth:`mark_all_dirty` and pays one full rebuild rather than reasoning about it.
    """

    def __init__(self, dimension: int) -> None:
        self._dim = int(dimension)
        self.buf: np.ndarray = np.zeros(self._dim, dtype=np.float32)
        self.reset()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Back to episode start: no valid buffer, no folds, no window."""
        self.buf.fill(0.0)
        self._ready = False
        self._sig: Optional[tuple] = None
        self._tag: Optional[str] = None
        self._fold_cursor: int = 0
        self._pending_cursor: int = 0
        self._all_dirty = True
        self._dirty: Set[Tuple[str, str]] = set()
        # Sides whose WHOLE roster an event changed (see the CURESTATUS branch in
        # `_fold_events`). Collected during the fold and expanded against the live roster in
        # `prepare`, which is the only place the roster is in hand.
        self._side_dirty: Set[str] = set()
        # slot identity: 12 entries, our 6 then opp 6, holding the species last encoded there.
        self._slot_species: List[Optional[str]] = [None] * (2 * TEAM_SIZE)
        self._prev_active: Dict[str, Optional[str]] = {OURS: None, OPP: None}
        self._req_seen: Dict[str, int] = {}
        self._hp_revision: int = -1
        self._reset_folds()
        self._reset_window()

    def mark_all_dirty(self) -> None:
        """Invalidate the cached vector. The next encode is a full rebuild.

        The event-driven folds (Wish / sleep sources) are NOT reset: they are pure functions of
        the battle's event log, which nothing here rolls back.
        """
        self._ready = False
        self._all_dirty = True

    # ------------------------------------------------------------------ #
    # Per-decision entry point                                            #
    # ------------------------------------------------------------------ #
    def prepare(self, strict: Any, live: Any, signature: tuple,
                hp_revision: Optional[int]) -> bool:
        """Fold everything that happened since the last encode; report whether the WARM path
        may be used.

        ``signature`` records which optional inputs the caller threaded this decision (hp
        tracker / legal / trackers). A caller that changes the set changes what the cached dims
        mean, so a change forces a rebuild rather than silently mixing two conventions.
        """
        tag = strict.battle_tag
        cursor = int(strict.event_cursor)
        if tag != self._tag or cursor < self._fold_cursor:
            # A different battle, or a log that went backwards (neither happens on the
            # production path). Refold from the beginning rather than guess.
            self._tag = tag
            self._fold_cursor = 0
            self._reset_folds()
            self._ready = False
        if signature != self._sig:
            self._sig = signature
            self._ready = False
        if hp_revision is not None and hp_revision != self._hp_revision:
            self._hp_revision = hp_revision
            for m in live.opp.mons:
                self._dirty.add((OPP, m.species))

        self._fold_events(strict.events_since(self._fold_cursor))
        self._pending_cursor = cursor
        if self._side_dirty:
            for side, sd in ((OURS, live.ours), (OPP, live.opp)):
                if side in self._side_dirty:
                    for m in sd.mons:
                        self._dirty.add((side, m.species))
            self._side_dirty.clear()

        # The two ACTIVES are re-encoded unconditionally. It costs ~2 slot encodes and buys the
        # request-order trapping bits, the H-A1 last-action tuple and every per-turn counter on
        # the mons that actually move — and it shrinks the event→dirty map to the families that
        # touch a BENCHED mon (reveal / consume / status / faint / forme).
        for side, sd in ((OURS, live.ours), (OPP, live.opp)):
            sp = sd.active.species if sd.active is not None else None
            prev = self._prev_active[side]
            if sp:
                self._dirty.add((side, sp))
            if prev and prev != sp:
                self._dirty.add((side, prev))
            self._prev_active[side] = sp

        # The REQUEST door (see the module docstring). Opponent mons never appear in a request.
        for m in live.ours.mons:
            seq = strict.request_change_seq(m.species)
            if self._req_seen.get(m.species) != seq:
                self._req_seen[m.species] = seq
                self._dirty.add((OURS, m.species))

        return self._ready and not self._all_dirty

    def begin_full(self) -> np.ndarray:
        """Zero the buffer for a full rebuild and return it."""
        self.buf.fill(0.0)
        self._reset_window()
        return self.buf

    def commit(self) -> None:
        """The buffer now holds a complete, correct encode.

        The fold cursor advances HERE, not in :meth:`prepare`, so an encode that raises between
        the two re-folds the same window next time. Every fold is idempotent under that replay
        (the dirty set is a set; the Wish double-cast guard re-derives the same answer; the
        sleep map is keyed by the latest slp seq), so the retry is correct rather than merely
        survivable.
        """
        self._ready = True
        self._all_dirty = False
        self._dirty.clear()
        self._side_dirty.clear()
        self._fold_cursor = self._pending_cursor

    # ------------------------------------------------------------------ #
    # Per-mon slot cache                                                  #
    # ------------------------------------------------------------------ #
    def slot_is_clean(self, side: str, index: int, species: Optional[str]) -> bool:
        """May slot ``index`` of ``side`` keep the 119 dims already in the buffer?

        Keyed by SPECIES, never by list position: the opponent's team list grows as mons are
        revealed and `get_team_list`'s "active opp not in team" fallback can hand a mon a
        temporary index, so a position-keyed cache would serve one mon's bytes under another's
        name. The recorded slot species is the join, and a mismatch is a miss.
        """
        if self._all_dirty or species is None:
            return False
        flat = index if side == OURS else TEAM_SIZE + index
        if self._slot_species[flat] != species:
            return False
        return (side, species) not in self._dirty

    def note_slot(self, side: str, index: int, species: Optional[str]) -> None:
        """Record which species the buffer's slot ``index`` currently holds."""
        self._slot_species[index if side == OURS else TEAM_SIZE + index] = species

    # ------------------------------------------------------------------ #
    # Incremental folds that replace per-encode whole-log scans           #
    # ------------------------------------------------------------------ #
    def _reset_folds(self) -> None:
        self._wish_ok: Dict[str, Set[int]] = {OURS: set(), OPP: set()}
        self._wish_last_ok: Dict[str, int] = {OURS: -10, OPP: -10}
        self._slp_seq: Dict[Tuple[str, str], int] = {}
        self._slp_is_rest: Dict[Tuple[str, str], bool] = {}
        self._slp_usable: Dict[Tuple[str, str], bool] = {}

    def wish_pending(self, cur_turn: int) -> Dict[str, bool]:
        """``{OURS: bool, OPP: bool}`` — is a Wish resolving at the END of this turn?

        The incremental twin of `wish_belief.build_wish_pending`, which folded the WHOLE event
        log on every encode. Same double-Wish rule (a cast at turn ``t`` fails if this side
        already had a successful cast at ``t-1``), same turn arithmetic.
        """
        target = int(cur_turn) - 1
        return {OURS: target in self._wish_ok[OURS], OPP: target in self._wish_ok[OPP]}

    def sleep_sources(self) -> Dict[Tuple[str, str], Tuple[bool, bool]]:
        """``{(side, species): (is_rest_sleep, sleep_usable_move_seen)}`` for each mon's CURRENT
        sleep episode — the incremental twin of `sleep_belief.build_sleep_sources` (two whole-log
        passes per encode, gated on "is anyone asleep")."""
        return {k: (self._slp_is_rest[k], self._slp_usable[k]) for k in self._slp_seq}

    # ------------------------------------------------------------------ #
    # The event → dirty map                                               #
    # ------------------------------------------------------------------ #
    def _fold_events(self, events: Sequence[Any]) -> None:
        dirty = self._dirty
        for e in events:
            k = e.kind
            side = getattr(e, "side", None)
            sp = getattr(e, "actor_species", None)
            if k is EventKind.CURESTATUS and side is not None:
                # 🚨 THE DOOR THE §2.2 EVENT→DIRTY MAP MISSED, found by the byte-identity fuzz
                # (11 mismatches in 9,272 decisions, all of them a stale `slp`/`brn` bit on a
                # BENCHED opponent). `EventKind.CURESTATUS` covers TWO protocol keywords, and
                # the second is TEAM-WIDE: `|-cureteam|` (Heal Bell / Aromatherapy) makes
                # poke-env loop `for mon in team.values(): mon.cure_status()` while the line
                # names only the ACTIVE mon — so the event's actor is one of the six mons it
                # changed. The narrower fix (discriminate on `e.raw[1]`) was deliberately NOT
                # taken: a cure is a handful of events per battle, six slot re-encodes is
                # noise, and being wrong here is a silently stale status bit rather than a
                # crash. Same reasoning as the always-dirty actives.
                self._side_dirty.add(side)
            if k in _WHOLE_SLOT_NUKE:
                # TRANSFORM / FORMECHANGE (incl. detailschange / replace) / SWAP rewrite
                # "static" fields — species num, base stats, types, and for Transform the whole
                # moveset — and a forme change moves the very key this cache is joined on.
                # Per-field surgery here is where correctness dies, so don't attempt it.
                self._all_dirty = True
                self._ready = False
                continue
            if k is EventKind.MOVE and e.move_id == _WISH and side in self._wish_ok:
                # The Wish fold's own gate is `e.side in (OURS, OPP)` — NOT actor_species —
                # so it sits outside the per-mon guard below, matching `build_wish_pending`
                # exactly rather than approximately.
                t = int(e.turn)
                if self._wish_last_ok[side] != t - 1:
                    self._wish_last_ok[side] = t
                    self._wish_ok[side].add(t)
            if side is not None and sp:
                dirty.add((side, sp))
                if k is EventKind.SWITCH or k is EventKind.DRAG:
                    # poke-env clears the LEAVING mon's boosts / volatiles and resets its
                    # protect + toxic counters internally, with no per-field event. The event
                    # only names the ARRIVING mon, so the outgoing one is dirtied from the
                    # active we recorded at the previous decision.
                    prev = self._prev_active.get(side)
                    if prev:
                        dirty.add((side, prev))
                    self._prev_active[side] = sp
                elif k is EventKind.MOVE:
                    if e.move_id in _SLEEP_USABLE_MOVES:
                        key = (side, sp)
                        seq0 = self._slp_seq.get(key)
                        if seq0 is not None and e.seq > seq0:
                            self._slp_usable[key] = True
                elif k is EventKind.STATUS and e.status == "slp":
                    key = (side, sp)
                    self._slp_seq[key] = e.seq
                    self._slp_is_rest[key] = _reason_is_rest(e.reason)
                    self._slp_usable[key] = False
                elif k is EventKind.FAINT:
                    if self._prev_active.get(side) == sp:
                        self._prev_active[side] = None

    # ------------------------------------------------------------------ #
    # The H-B event window ring                                           #
    # ------------------------------------------------------------------ #
    def _reset_window(self) -> None:
        self._ew_recs: List[Dict[str, Any]] = []
        self._ew_turns = np.zeros(EVENT_WINDOW_N, dtype=np.int32)
        self._ew_turn: Optional[int] = None

    def seed_window(self, rows: Sequence[Dict[str, Any]], cur_turn: int) -> None:
        """Adopt a window the FULL path has just written into the buffer verbatim."""
        rows = list(rows)[-EVENT_WINDOW_N:]
        self._ew_recs = rows
        self._ew_turns[:] = 0
        n = len(rows)
        for i, r in enumerate(rows):
            self._ew_turns[EVENT_WINDOW_N - n + i] = int(r["turn"])
        self._ew_turn = int(cur_turn)

    def update_window(self, rows: Sequence[Dict[str, Any]], cur_turn: int,
                      open_records: Sequence[Dict[str, Any]]) -> None:
        """Bring the buffer's 704-dim event block up to date **without rewriting every row**.

        The flat layout front-pads (most-recent LAST), so an append shifts every retained row
        left by exactly one slot — in both the full and the not-yet-full case. That shift is one
        numpy move of ≤704 floats; what it saves is re-deriving 22 values and three dex lookups
        for up to 32 rows that did not change. Three things can move:

        * **appends** — ``k`` new rows arrive at the end after the shift;
        * **in-place mutation** — a MOVE row accumulates damage / outcome / effectiveness after
          it was appended. Every such write goes through ``EventWindowTracker._open_move``, so
          re-writing exactly the open records covers the whole mutation surface by construction;
        * **the turn** — ``TURNS_AGO`` is a function of the CURRENT turn, so every valid row's
          one column is re-patched (vectorised) when the turn moves.
        """
        rows = list(rows)[-EVENT_WINDOW_N:]
        n = len(rows)
        prev = self._ew_recs
        base = OFFSET_EVENT_WINDOW
        buf = self.buf
        mat = buf[base:base + EVENT_WINDOW_N * EVENT_TOKEN_DIM].reshape(
            EVENT_WINDOW_N, EVENT_TOKEN_DIM)

        # How many rows were appended since we last materialised? Identify the previously-last
        # record by IDENTITY (records are mutable dicts owned by the tracker).
        k = n
        if prev:
            last = prev[-1]
            for idx in range(n - 1, -1, -1):
                if rows[idx] is last:
                    k = n - 1 - idx
                    break

        if k >= n or not prev:
            # Nothing survives (or we have no history): rewrite the whole block.
            mat[:] = 0.0
            self._ew_turns[:] = 0
            for i, r in enumerate(rows):
                o = base + (EVENT_WINDOW_N - n + i) * EVENT_TOKEN_DIM
                write_event_row(buf, o, r, cur_turn)
                self._ew_turns[EVENT_WINDOW_N - n + i] = int(r["turn"])
            self._ew_recs = rows
            self._ew_turn = int(cur_turn)
            return

        if k:
            mat[:-k] = mat[k:]
            mat[-k:] = 0.0
            self._ew_turns[:-k] = self._ew_turns[k:]
            self._ew_turns[-k:] = 0
            for j in range(n - k, n):
                o = base + (EVENT_WINDOW_N - n + j) * EVENT_TOKEN_DIM
                write_event_row(buf, o, rows[j], cur_turn)
                self._ew_turns[EVENT_WINDOW_N - n + j] = int(rows[j]["turn"])

        # Re-write the still-open MOVE rows — the entire in-place mutation surface.
        for rec in open_records:
            for j in range(n - 1, -1, -1):
                if rows[j] is rec:
                    o = base + (EVENT_WINDOW_N - n + j) * EVENT_TOKEN_DIM
                    _clear_event_row(buf, o)
                    write_event_row(buf, o, rec, cur_turn)
                    break

        if self._ew_turn != cur_turn and n:
            lo = EVENT_WINDOW_N - n
            ago = np.clip(int(cur_turn) - self._ew_turns[lo:], 0, _SAT_CAP)
            mat[lo:, EVENT_COL.TURNS_AGO] = _SAT_LUT_F32[ago]

        self._ew_recs = rows
        self._ew_turn = int(cur_turn)


_WHOLE_SLOT_NUKE = frozenset({
    EventKind.TRANSFORM, EventKind.FORMECHANGE, EventKind.SWAP, EventKind.UNKNOWN,
})
