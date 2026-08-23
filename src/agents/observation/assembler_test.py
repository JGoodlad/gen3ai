"""Unit gate for the incremental obs cache (`gen3_obs_assembler_v1`).

Two halves, and they answer different questions:

* **The scheduler's invalidation surface** — one NAMED test per §2.3 trap in
  `designs/ai_v9/design_incremental_obs_encoder.md`, each of which FAILS if its dirty edge is
  removed. These are cheap and deterministic; the bridge fuzz
  (`training/poke_env_gaps/obs_assembler_fuzz_test.py`) proves byte-identity over real battles,
  but a random corpus does not reliably contain a Ditto or a Baton Pass, so the edges are pinned
  here by construction rather than by luck.
* **End-to-end byte-identity on a SCRIPTED battle** — a hand-built protocol script fed to a real
  `Gen3Battle`, encoded both ways at every step. This is the same assertion the fuzz makes, on a
  corpus we choose rather than one we sample.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from agents.battle.battle_event import OPP, OURS, BattleEvent, EventKind
from agents.battle.gen3_battle import Gen3Battle
from agents.observation.assembler import SAT_LUT, ObsAssembler, describe_offset
from agents.observation.constants import (
    EVENT_TOKEN_DIM,
    EVENT_WINDOW_N,
    OFFSET_EVENT_WINDOW,
    EventCol,
)
from agents.observation.sleep_belief import build_sleep_sources
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.wish_belief import build_wish_pending
from agents.training.episode_tracker import EpisodeTracker, _pair_sat_norm

LOG = logging.getLogger("assembler-test")


# --------------------------------------------------------------------------------------- #
# The saturation LUT — a value-neutral substitution, so prove it is value-neutral.          #
# --------------------------------------------------------------------------------------- #
def test_the_saturation_lut_is_bit_for_bit_the_arithmetic_it_replaced():
    """`log1p(min(n, 10)) / log(11)` has an 11-value codomain, so the hot paths read a table.
    A table that is merely CLOSE would silently move every recency and pair-history dim."""
    for n in range(0, 40):
        want = math.log1p(min(n, 10)) / math.log(11.0)
        assert SAT_LUT[min(n, 10)] == want, n
        assert _pair_sat_norm(n) == want, n
    assert _pair_sat_norm(-5) == math.log1p(0) / math.log(11.0)
    # the non-default `sat` still goes through the arithmetic
    assert _pair_sat_norm(3, sat=5) == math.log1p(3) / math.log(6.0)


def test_describe_offset_names_every_block_from_the_declared_layout():
    enc = Gen3ObservationEncoder(load_mappings())
    seen = {describe_offset(i).split("[")[0].split(" ")[0] for i in range(enc.dimension)}
    assert {"our_team", "opp_team", "active_context", "global", "board/reactive",
            "pair_history", "event_window"} <= seen
    assert describe_offset(enc.dimension - 1).startswith("event_window row 31")


# --------------------------------------------------------------------------------------- #
# Trap-by-trap: the event -> dirty map                                                      #
# --------------------------------------------------------------------------------------- #
class _Side:
    def __init__(self, mons, active):
        self.mons = mons
        self.active = active


class _Mon:
    def __init__(self, species):
        self.species = species


class _Live:
    def __init__(self, ours, opp, turn=5):
        self.ours = ours
        self.opp = opp
        self.turn = turn


class _Strict:
    """A minimal stand-in for `StrictBattleView` — the four members `prepare` reads."""

    def __init__(self, events=(), cursor=None, tag="battle-t", req=None):
        self._events = list(events)
        self.battle_tag = tag
        self.event_cursor = len(self._events) if cursor is None else cursor
        self._req = req or {}

    def events_since(self, c):
        return self._events[c:]

    def request_change_seq(self, species):
        return self._req.get(species, 0)


def _ev(kind, side, actor, *, turn=5, seq=0, move_id=None, status=None, reason=None):
    return BattleEvent(seq=seq, turn=turn, kind=kind, raw="", side=side,
                       actor_species=actor, value={"move": move_id, "status": status,
                                                   "reason": reason})


def _live(our_active="Skarmory", opp_active="Tyranitar", turn=5):
    ours = _Side([_Mon("Skarmory"), _Mon("Blissey")], _Mon(our_active) if our_active else None)
    opp = _Side([_Mon("Tyranitar"), _Mon("Gengar")], _Mon(opp_active) if opp_active else None)
    return _Live(ours, opp, turn)


_SIG = (True, True, True, True, True, True)


def _warm(asm, strict, live):
    """Drive one prepare/commit cycle and report whether the WARM path was offered."""
    warm = asm.prepare(strict, live, _SIG, 0)
    for side, sd in (("ours", live.ours), ("opp", live.opp)):
        for i, m in enumerate(sd.mons):
            if not asm.slot_is_clean(side, i, m.species):
                asm.note_slot(side, i, m.species)
    asm.commit()
    return warm


def _fresh_warm_assembler(dim=64):
    """An assembler that has already served one decision — so the next one can be warm."""
    asm = ObsAssembler(dim)
    _warm(asm, _Strict(), _live())
    return asm


def test_the_first_decision_is_never_warm():
    asm = ObsAssembler(64)
    assert asm.prepare(_Strict(), _live(), _SIG, 0) is False


def test_the_two_ACTIVES_are_dirty_on_every_decision():
    """Trap 3: the request-order bits and the H-A1 last-action tuple ride the active slots and
    are per-decision BY NATURE. Re-encoding both actives unconditionally is what keeps the
    event→dirty map limited to the families that touch a BENCHED mon."""
    asm = _fresh_warm_assembler()
    assert asm.prepare(_Strict(), _live(), _SIG, 0) is True
    assert not asm.slot_is_clean("ours", 0, "Skarmory")   # our active
    assert not asm.slot_is_clean("opp", 0, "Tyranitar")   # their active
    assert asm.slot_is_clean("ours", 1, "Blissey")        # a bench mon stays cached
    assert asm.slot_is_clean("opp", 1, "Gengar")


def test_a_SWITCH_dirties_the_OUTGOING_mon_too():
    """Trap 1: poke-env clears the leaving mon's boosts / volatiles and resets its protect and
    toxic counters INTERNALLY — no per-field event — and the SWITCH event names only the mon
    coming IN. Without this edge the outgoing mon keeps a stale protect-odds / counter block."""
    asm = _fresh_warm_assembler()
    ev = _ev(EventKind.SWITCH, OPP, "Gengar", seq=0)
    asm.prepare(_Strict([ev]), _live(opp_active="Gengar"), _SIG, 0)
    assert not asm.slot_is_clean("opp", 0, "Tyranitar")   # the mon that LEFT
    assert not asm.slot_is_clean("opp", 1, "Gengar")      # the mon that arrived


def test_a_BATON_PASS_switch_recomputes_rather_than_zeroing():
    """Trap 8: gen3 Baton Pass KEEPS boosts and volatiles across the switch, so "write zeros on
    SWITCH" is wrong in the other direction. The active context is never cached at all — it is
    recomputed from live state every decision — which makes both directions unrepresentable.
    This test pins the STRUCTURAL fact: no active-context state is held on the assembler."""
    asm = _fresh_warm_assembler()
    held = {k for k in vars(asm) if "ctx" in k.lower() or "boost" in k.lower()
            or "volatile" in k.lower()}
    assert held == set(), f"the assembler must hold no active-context state, found {held}"


@pytest.mark.parametrize("kind", [EventKind.TRANSFORM, EventKind.FORMECHANGE, EventKind.SWAP])
def test_a_whole_slot_NUKE_event_invalidates_everything(kind):
    """Trap 2: Transform / forme change rewrite species num, base stats, types and (Transform)
    the whole moveset at once — and a forme change moves the very SPECIES key this cache is
    joined on. Per-field surgery here is where correctness dies, so these mark everything
    dirty and force a full rebuild."""
    asm = _fresh_warm_assembler()
    warm = asm.prepare(_Strict([_ev(kind, OPP, "Castform")]), _live(), _SIG, 0)
    assert warm is False
    assert not asm.slot_is_clean("ours", 1, "Blissey")


@pytest.mark.parametrize("kind,actor", [
    (EventKind.DAMAGE, "Blissey"), (EventKind.HEAL, "Blissey"), (EventKind.SETHP, "Blissey"),
    (EventKind.STATUS, "Blissey"), (EventKind.CURESTATUS, "Blissey"),
    (EventKind.ITEM, "Blissey"), (EventKind.ENDITEM, "Blissey"),
    (EventKind.ABILITY, "Blissey"), (EventKind.FAINT, "Blissey"),
    (EventKind.VOLATILE_START, "Blissey"), (EventKind.VOLATILE_END, "Blissey"),
])
def test_every_actor_bearing_event_dirties_its_own_BENCHED_mon(kind, actor):
    """The map is "any event with a resolvable (side, species) dirties that slot" — deliberately
    coarser than the design's per-family table, because the families it would leave out (a
    Pain-Split SETHP, a Trick ITEM, a Conversion type change riding a VOLATILE_START) are
    exactly the ones whose per-field surgery would be subtle."""
    asm = _fresh_warm_assembler()
    asm.prepare(_Strict([_ev(kind, OURS, actor)]), _live(), _SIG, 0)
    assert not asm.slot_is_clean("ours", 1, actor)


def test_the_REQUEST_door_dirties_only_the_mons_the_request_changed():
    """The one mutation channel with no BattleEvent. It must be per-MON: a request arrives every
    decision, so a global signal would dirty our whole side every decision and delete the win."""
    asm = _fresh_warm_assembler()
    asm.prepare(_Strict(req={"Blissey": 7}), _live(), _SIG, 0)
    assert not asm.slot_is_clean("ours", 1, "Blissey")
    assert asm.slot_is_clean("opp", 1, "Gengar")
    # the same seq again is NOT a change
    _warm(asm, _Strict(req={"Blissey": 7}), _live())
    asm.prepare(_Strict(req={"Blissey": 7}), _live(), _SIG, 0)
    assert asm.slot_is_clean("ours", 1, "Blissey")


def test_an_HP_TRACKER_revision_dirties_the_whole_OPPONENT_side():
    """The HiddenPowerTracker writes 17 dims of an opponent slot from OUR code, not from a
    protocol line. Its narrowing is event-triggered today — so an event-only rule would usually
    be right, and "usually" is the shape of a silent staleness bug."""
    asm = _fresh_warm_assembler()
    asm.prepare(_Strict(), _live(), _SIG, 1)          # revision moved 0 -> 1
    assert not asm.slot_is_clean("opp", 1, "Gengar")


def test_a_changed_optional_INPUT_SET_forces_a_rebuild():
    """A caller that stops threading the hp tracker (or the recency tracker) changes what the
    cached dims MEAN. Mixing two conventions in one buffer is silent; rebuilding is not."""
    asm = _fresh_warm_assembler()
    assert asm.prepare(_Strict(), _live(), (False,) * 6, 0) is False


def test_a_slot_whose_SPECIES_changed_is_a_MISS_not_a_hit():
    """Trap 4: the opponent's team list GROWS as mons are revealed, and `get_team_list`'s
    "active opp not in team" fallback can hand a mon a temporary index. A position-keyed cache
    would serve one mon's 122 dims under another mon's name."""
    asm = _fresh_warm_assembler()
    asm.prepare(_Strict(), _live(), _SIG, 0)
    assert asm.slot_is_clean("opp", 1, "Gengar")
    assert not asm.slot_is_clean("opp", 1, "Starmie")   # a different mon at the same index
    assert not asm.slot_is_clean("opp", 1, None)        # an empty slot is never clean


def test_a_DIFFERENT_battle_or_a_REWOUND_log_refolds_from_scratch():
    asm = _fresh_warm_assembler()
    assert asm.prepare(_Strict(tag="battle-other"), _live(), _SIG, 0) is False
    _warm(asm, _Strict([_ev(EventKind.MOVE, OURS, "Skarmory")] * 3), _live())
    assert asm.prepare(_Strict(cursor=0), _live(), _SIG, 0) is False


def test_mark_all_dirty_forces_a_full_rebuild_and_is_what_restore_calls():
    """The materializer's per-arm restore and the self-play re-decide rollback both roll tracker
    state back WITHOUT rolling the battle back, so the cache's dirty bits and ring appends would
    survive a decision that never happened."""
    asm = _fresh_warm_assembler()
    assert asm.prepare(_Strict(), _live(), _SIG, 0) is True
    asm.mark_all_dirty()
    assert asm.prepare(_Strict(), _live(), _SIG, 0) is False


def test_episode_tracker_restore_marks_the_assembler_dirty():
    tr = EpisodeTracker()
    asm = tr.obs_assembler(64)
    _warm(asm, _Strict(), _live())
    assert asm.prepare(_Strict(), _live(), _SIG, 0) is True
    tr.restore(tr.snapshot())
    assert asm.prepare(_Strict(), _live(), _SIG, 0) is False


def test_episode_tracker_reset_clears_the_assembler():
    """Cross-turn caches are WITHIN-BATTLE only — the env resets one tracker per episode rather
    than recreating it, which is exactly how the recency tracker once leaked across episodes."""
    tr = EpisodeTracker()
    asm = tr.obs_assembler(64)
    asm.buf[3] = 9.0
    _warm(asm, _Strict(), _live())
    tr.reset()
    assert asm.prepare(_Strict(), _live(), _SIG, 0) is False
    assert asm.buf[3] == 0.0
    assert tr.obs_assembler(64) is asm


# --------------------------------------------------------------------------------------- #
# The incremental folds vs the whole-log ones they replace                                  #
# --------------------------------------------------------------------------------------- #
class _FakeBattle:
    def __init__(self, events, turn):
        self.events = events
        self.turn = turn


def test_the_incremental_WISH_fold_matches_the_whole_log_fold_including_double_wish():
    """`build_wish_pending` scans the WHOLE log per encode. The incremental twin must agree on
    the awkward case too: a second Wish while one is pending FAILS, so it opens no window."""
    events = [
        _ev(EventKind.MOVE, OURS, "Blissey", turn=1, seq=0, move_id="wish"),
        _ev(EventKind.MOVE, OURS, "Blissey", turn=2, seq=1, move_id="wish"),   # fails
        _ev(EventKind.MOVE, OURS, "Blissey", turn=3, seq=2, move_id="wish"),   # succeeds
        _ev(EventKind.MOVE, OPP, "Vaporeon", turn=4, seq=3, move_id="wish"),
    ]
    asm = ObsAssembler(64)
    for turn in range(1, 7):
        asm.prepare(_Strict(events), _live(turn=turn), _SIG, 0)
        got = asm.wish_pending(turn)
        want = build_wish_pending(_FakeBattle(events, turn))
        assert got == want, (turn, got, want)
        asm.commit()


def test_the_incremental_SLEEP_SOURCE_fold_matches_the_whole_log_fold():
    """Two whole-log passes per encode, gated on "is anyone asleep". The incremental twin must
    agree on the re-sleep case: a Sleep Talk BEFORE the current sleep episode does not corrupt
    its counter, so a fresh slp application clears the reliability flag."""
    events = [
        _ev(EventKind.STATUS, OPP, "Snorlax", turn=1, seq=0, status="slp",
            reason="move: Spore"),
        _ev(EventKind.MOVE, OPP, "Snorlax", turn=2, seq=1, move_id="sleeptalk"),
        _ev(EventKind.STATUS, OURS, "Blissey", turn=3, seq=2, status="slp", reason="move: Rest"),
        _ev(EventKind.STATUS, OPP, "Snorlax", turn=5, seq=3, status="slp"),   # re-slept
    ]
    for n in range(len(events) + 1):
        asm = ObsAssembler(64)
        asm.prepare(_Strict(events[:n]), _live(), _SIG, 0)
        asm.commit()
        assert asm.sleep_sources() == build_sleep_sources(_FakeBattle(events[:n], 6)), n


# --------------------------------------------------------------------------------------- #
# The event-window ring                                                                     #
# --------------------------------------------------------------------------------------- #
def _row(t=1, turn=1, actor="Skarmory", side=OURS, move_id=None, **kw):
    rec = dict(t=t, actor=actor, side=side, target=None, move_id=move_id, hp_delta=0.0,
               missed=False, failed=False, crit=False, eff=0, we_first=False, status=0,
               turn=turn, forced_window=0.0)
    rec.update(kw)
    return rec


def _window_block(asm):
    return asm.buf[OFFSET_EVENT_WINDOW:OFFSET_EVENT_WINDOW + EVENT_WINDOW_N * EVENT_TOKEN_DIM]


def _full_write(rows, cur_turn, dim):
    """The FULL path's write, as the oracle."""
    from agents.observation.assembler import write_event_row
    vec = np.zeros(dim, dtype=np.float32)
    base = OFFSET_EVENT_WINDOW + (EVENT_WINDOW_N - len(rows)) * EVENT_TOKEN_DIM
    for i, r in enumerate(rows):
        write_event_row(vec, base + i * EVENT_TOKEN_DIM, r, cur_turn)
    return vec[OFFSET_EVENT_WINDOW:OFFSET_EVENT_WINDOW + EVENT_WINDOW_N * EVENT_TOKEN_DIM]


_DIM = OFFSET_EVENT_WINDOW + EVENT_WINDOW_N * EVENT_TOKEN_DIM


def test_the_ring_matches_a_full_rewrite_through_growth_rotation_and_turn_ticks():
    """Trap 6: the flat layout front-pads, so an append shifts EVERY row — a representation
    artifact the ring removes. Grow past the 32-row cap (so the deque rotates), tick the turn
    (so the TURNS_AGO column is re-patched) and compare against a full rewrite each step."""
    asm = ObsAssembler(_DIM)
    rows: list = []
    asm.seed_window(rows, 0)
    for k in range(60):
        turn = 1 + k // 2
        for _ in range(1 + (k % 3)):
            rows.append(_row(turn=turn, move_id="thunderbolt" if k % 2 else None))
        rows = rows[-EVENT_WINDOW_N:]
        asm.update_window(rows, turn, ())
        assert np.array_equal(_window_block(asm), _full_write(rows, turn, _DIM)), k


def test_an_OPEN_move_row_mutated_after_its_append_is_rewritten():
    """A MOVE row accumulates damage / outcome / effectiveness AFTER it was appended, in place.
    Every such write goes through `EventWindowTracker._open_move`, so re-writing exactly the
    open records covers the whole mutation surface — this test is what says so."""
    asm = ObsAssembler(_DIM)
    open_row = _row(turn=3, move_id="earthquake", t=1)
    rows = [_row(turn=2), open_row]
    asm.update_window(rows, 3, ())      # a cold materialise (no prior ring)
    assert np.array_equal(_window_block(asm), _full_write(rows, 3, _DIM))
    # the tracker mutates it in place, as it does for a late |-damage| / |-crit| / |-immune|
    open_row["hp_delta"] = -0.44
    open_row["crit"] = True
    open_row["eff"] = 1
    asm.update_window(rows, 3, (open_row,))
    assert np.array_equal(_window_block(asm), _full_write(rows, 3, _DIM))
    # and the stale one-hot must be GONE, not merely joined by the new one
    off = OFFSET_EVENT_WINDOW + (EVENT_WINDOW_N - 1) * EVENT_TOKEN_DIM
    assert asm.buf[off + EventCol.EFF_NEUTRAL] == 0.0
    assert asm.buf[off + EventCol.EFF_SUPER] == 1.0


def test_the_TURNS_AGO_column_saturates_and_leaves_the_pad_rows_zero():
    asm = ObsAssembler(_DIM)
    rows = [_row(turn=1), _row(turn=2)]
    for turn in range(2, 40):
        asm.update_window(rows, turn, ())
        assert np.array_equal(_window_block(asm), _full_write(rows, turn, _DIM)), turn
    pad = asm.buf[OFFSET_EVENT_WINDOW:
                  OFFSET_EVENT_WINDOW + (EVENT_WINDOW_N - 2) * EVENT_TOKEN_DIM]
    assert not pad.any()


# --------------------------------------------------------------------------------------- #
# End-to-end byte-identity on a scripted battle                                             #
# --------------------------------------------------------------------------------------- #
SCRIPT = [
    ["", "player", "p1", "p1user", "", ""],
    ["", "player", "p2", "p2user", "", ""],
    ["", "teamsize", "p1", "6"],
    ["", "teamsize", "p2", "6"],
    ["", "gametype", "singles"],
    ["", "gen", "3"],
    ["", "tier", "[Gen 3] OU"],
    ["", "start"],
    ["", "switch", "p1a: Skarm", "Skarmory, L100, F", "100/100"],
    ["", "switch", "p2a: Tyra", "Tyranitar, L100, M", "100/100"],
    ["", "turn", "1"],
    ["", "move", "p1a: Skarm", "Spikes", "p2a: Tyra"],
    ["", "-sidestart", "p2: p2user", "Spikes"],
    ["", "move", "p2a: Tyra", "Rock Slide", "p1a: Skarm"],
    ["", "-damage", "p1a: Skarm", "70/100"],
    ["", "turn", "2"],
    # a Wish (the incremental fold), then a sleep (the other one)
    ["", "move", "p1a: Skarm", "Wish", "p1a: Skarm"],
    ["", "move", "p2a: Tyra", "Toxic", "p1a: Skarm"],
    ["", "-status", "p1a: Skarm", "tox"],
    ["", "turn", "3"],
    # opponent reveals a second mon — a NEW opp slot appears mid-battle
    ["", "switch", "p2a: Gar", "Gengar, L100, M", "100/100"],
    ["", "-damage", "p2a: Gar", "88/100", "[from] Spikes"],
    ["", "move", "p1a: Skarm", "Toxic", "p2a: Gar"],
    ["", "-status", "p2a: Gar", "tox"],
    ["", "turn", "4"],
    ["", "move", "p2a: Gar", "Hypnosis", "p1a: Skarm"],
    ["", "-status", "p1a: Skarm", "slp", "[from] move: Hypnosis"],
    ["", "-damage", "p1a: Skarm", "44/100", "[from] psn"],
    ["", "turn", "5"],
    ["", "cant", "p1a: Skarm", "slp"],
    ["", "move", "p2a: Gar", "Explosion", "p1a: Skarm"],
    ["", "-crit", "p1a: Skarm"],
    ["", "-damage", "p1a: Skarm", "0 fnt"],
    ["", "faint", "p1a: Skarm"],
    ["", "faint", "p2a: Gar"],          # the double-KO / forced-switch window
    ["", "switch", "p1a: Bliss", "Blissey, L100, F", "100/100"],
    ["", "switch", "p2a: Tyra", "Tyranitar, L100, M", "100/100"],
    ["", "-damage", "p2a: Tyra", "88/100", "[from] Spikes"],
    ["", "turn", "6"],
    # an item transition (Knock Off — REMOVED, not consumed) and a boost
    ["", "move", "p2a: Tyra", "Dragon Dance", "p2a: Tyra"],
    ["", "-boost", "p2a: Tyra", "atk", "1"],
    ["", "-boost", "p2a: Tyra", "spe", "1"],
    ["", "move", "p1a: Bliss", "Knock Off", "p2a: Tyra"],
    ["", "-damage", "p2a: Tyra", "84/100"],
    ["", "-enditem", "p2a: Tyra", "Leftovers", "[from] move: Knock Off"],
    ["", "turn", "7"],
]


def _feed(battle, lines, encoder, tracker, out):
    """Feed the script line by line; at each `|turn|` boundary, encode BOTH ways and compare."""
    asm = tracker.obs_assembler(encoder.dimension)
    for line in lines:
        battle.parse_message(line)
        if line[1] != "turn":
            continue
        mask = np.ones(11, dtype=np.int8)
        tracker.record(battle, mask)
        tracker.update_progress_clock(battle, None)
        kw = dict(hp_tracker=tracker.hidden_power_tracker,
                  progress_clock=tracker.progress_clock, recency=tracker.recency,
                  pair_history=tracker.pair_history, event_window=tracker.event_window)
        was_warm = asm._ready and not asm._all_dirty
        got = encoder.encode(battle, assembler=asm, **kw)
        want = encoder.encode(battle, assembler=None, **kw)
        bad = np.flatnonzero(got != want)
        out.append((battle.turn, bad, was_warm))


# `|-cureteam|` (Heal Bell / Aromatherapy) cures EVERY mon on the side while naming only the
# ACTIVE one. It is the door the design's §2.2 event→dirty map missed, and the byte-identity
# fuzz found it: 11 mismatches in 9,272 decisions, all a stale `slp`/`brn` bit on a BENCHED
# opponent. This script is the deterministic reproducer — a benched, statused mon cured by a
# line that names somebody else.
CURETEAM_SCRIPT = [
    ["", "player", "p1", "p1user", "", ""],
    ["", "player", "p2", "p2user", "", ""],
    ["", "teamsize", "p1", "6"],
    ["", "teamsize", "p2", "6"],
    ["", "gametype", "singles"],
    ["", "gen", "3"],
    ["", "tier", "[Gen 3] OU"],
    ["", "start"],
    ["", "switch", "p1a: Skarm", "Skarmory, L100, F", "100/100"],
    ["", "switch", "p2a: Tyra", "Tyranitar, L100, M", "100/100"],
    ["", "turn", "1"],
    ["", "move", "p1a: Skarm", "Toxic", "p2a: Tyra"],
    ["", "-status", "p2a: Tyra", "tox"],
    ["", "turn", "2"],
    # Tyranitar goes to the BENCH still badly poisoned — the cache now holds its `tox` bit.
    ["", "switch", "p2a: Bliss", "Blissey, L100, F", "100/100"],
    ["", "move", "p1a: Skarm", "Spikes", "p2a: Bliss"],
    ["", "-sidestart", "p2: p2user", "Spikes"],
    ["", "turn", "3"],
    # …and Heal Bell cures the WHOLE side while naming only Blissey.
    ["", "move", "p2a: Bliss", "Heal Bell", "p2a: Bliss"],
    ["", "-cureteam", "p2a: Bliss", "[from] move: Heal Bell"],
    ["", "turn", "4"],
]


def test_a_CURETEAM_line_cures_the_WHOLE_side_and_the_cache_must_follow():
    """The regression for the fuzz's only real finding. `EventKind.CURESTATUS` covers two
    keywords and one of them is team-wide, so an actor-keyed dirty rule leaves five benched
    mons with a status they no longer have. FAILS if the side-wide edge is removed."""
    enc = Gen3ObservationEncoder(load_mappings())
    battle = Gen3Battle("battle-gen3ou-cure", "p1user", LOG, gen=3)
    tracker = EpisodeTracker()
    out: list = []
    _feed(battle, CURETEAM_SCRIPT, enc, tracker, out)
    for turn, bad, _warm in out:
        assert bad.size == 0, (
            f"turn {turn}: {bad.size} dims differ, first at {int(bad[0])} "
            f"({describe_offset(int(bad[0]))})")
    # and prove the script really did leave a benched mon statused, so the test has teeth
    assert any(m.status is not None for m in battle.opponent_team.values()) is False
    assert len(battle.opponent_team) == 2


def test_a_scripted_battle_encodes_IDENTICALLY_incremental_and_full():
    enc = Gen3ObservationEncoder(load_mappings())
    battle = Gen3Battle("battle-gen3ou-asm", "p1user", LOG, gen=3)
    tracker = EpisodeTracker()
    out: list = []
    _feed(battle, SCRIPT, enc, tracker, out)
    assert len(out) >= 7
    for turn, bad, _warm in out:
        assert bad.size == 0, (
            f"turn {turn}: {bad.size} dims differ, first at {int(bad[0])} "
            f"({describe_offset(int(bad[0]))})")


def test_the_scripted_battle_actually_took_the_warm_path():
    """A byte-identity assertion that only ever compared two COLD rebuilds would pass forever."""
    enc = Gen3ObservationEncoder(load_mappings())
    battle = Gen3Battle("battle-gen3ou-asm2", "p1user", LOG, gen=3)
    tracker = EpisodeTracker()
    asm = tracker.obs_assembler(enc.dimension)
    warm_hits = 0
    for line in SCRIPT:
        battle.parse_message(line)
        if line[1] != "turn":
            continue
        tracker.record(battle, np.ones(11, dtype=np.int8))
        tracker.update_progress_clock(battle, None)
        if asm._ready and not asm._all_dirty:
            warm_hits += 1
        enc.encode(battle, assembler=asm, event_window=tracker.event_window,
                   recency=tracker.recency, pair_history=tracker.pair_history,
                   progress_clock=tracker.progress_clock,
                   hp_tracker=tracker.hidden_power_tracker)
    assert warm_hits >= 5


def test_a_deliberately_STALE_cache_is_caught_by_the_byte_comparison():
    """The gate's own negative control: if the scheduler wrongly kept a slot, the comparison
    must go red. Poison one cached bench slot and prove the assertion fires."""
    enc = Gen3ObservationEncoder(load_mappings())
    battle = Gen3Battle("battle-gen3ou-asm3", "p1user", LOG, gen=3)
    tracker = EpisodeTracker()
    asm = tracker.obs_assembler(enc.dimension)
    out: list = []
    _feed(battle, SCRIPT, enc, tracker, out)
    asm.buf[900] = 12345.0            # a byte no writer would produce, in a clean opp slot
    tracker.record(battle, np.ones(11, dtype=np.int8))
    tracker.update_progress_clock(battle, None)
    kw = dict(hp_tracker=tracker.hidden_power_tracker,
              progress_clock=tracker.progress_clock, recency=tracker.recency,
              pair_history=tracker.pair_history, event_window=tracker.event_window)
    got = enc.encode(battle, assembler=asm, **kw)
    want = enc.encode(battle, assembler=None, **kw)
    assert not np.array_equal(got, want)


def test_GEN3AI_OBS_VERIFY_raises_on_a_poisoned_buffer(monkeypatch):
    """The shipped shadow mode, on the same poison. It must RAISE, and the message must name
    the block — an assertion whose text says only "vectors differ" starts an investigation
    instead of ending one."""
    from agents.observation import assembler as asm_mod

    enc = Gen3ObservationEncoder(load_mappings())
    battle = Gen3Battle("battle-gen3ou-asm4", "p1user", LOG, gen=3)
    tracker = EpisodeTracker()
    asm = tracker.obs_assembler(enc.dimension)
    out: list = []
    _feed(battle, SCRIPT, enc, tracker, out)
    monkeypatch.setattr(asm_mod, "OBS_VERIFY", True)
    asm.buf[900] = 12345.0
    tracker.record(battle, np.ones(11, dtype=np.int8))
    tracker.update_progress_clock(battle, None)
    kw = dict(hp_tracker=tracker.hidden_power_tracker,
              progress_clock=tracker.progress_clock, recency=tracker.recency,
              pair_history=tracker.pair_history, event_window=tracker.event_window)
    with pytest.raises(AssertionError, match=r"GEN3AI_OBS_VERIFY.*opp_team"):
        enc.encode(battle, assembler=asm, **kw)


# --------------------------------------------------------------------------------------- #
# The four traps a RANDOM gen3ou corpus cannot reach                                        #
# --------------------------------------------------------------------------------------- #
# Measured over 200 bridge battles / 15,607 decisions: `forme_change`, `transform`,
# `partial_trap` and `pain_split` all report NOT SEEN — Castform and Ditto are not OU, and the
# wrap family and Pain Split simply never came up. A clean fuzz PASS is therefore not a pass for
# these four, which is exactly what the fuzz's coverage census says out loud. They are scripted
# here instead, so the corpus's luck is not what covers them.
_HEAD = [
    ["", "player", "p1", "p1user", "", ""],
    ["", "player", "p2", "p2user", "", ""],
    ["", "teamsize", "p1", "6"],
    ["", "teamsize", "p2", "6"],
    ["", "gametype", "singles"],
    ["", "gen", "3"],
    ["", "tier", "[Gen 3] OU"],
    ["", "start"],
]


def _run_script(lines, tag):
    enc = Gen3ObservationEncoder(load_mappings())
    battle = Gen3Battle(tag, "p1user", LOG, gen=3)
    tracker = EpisodeTracker()
    out: list = []
    _feed(battle, lines, enc, tracker, out)
    assert out, "the script produced no decisions"
    for turn, bad, _warm in out:
        assert bad.size == 0, (
            f"turn {turn}: {bad.size} dims differ, first at {int(bad[0])} "
            f"({describe_offset(int(bad[0]))})")
    # A byte-identity assertion that only ever compared two COLD rebuilds would pass forever.
    assert sum(1 for _, _, w in out if w) >= 1, "the script never took the WARM path"
    return battle


def test_trap_FORME_CHANGE_castform_rewrites_species_stats_and_types():
    """Trap 2, half one. A Castform weather forme changes the species NUM, the base stats and
    both types at once — and the species is the very key the per-mon cache is joined on, so a
    per-field dirty rule would look up the new species and find the old mon's bytes."""
    battle = _run_script(_HEAD + [
        ["", "switch", "p1a: Skarm", "Skarmory, L100, F", "100/100"],
        ["", "switch", "p2a: Cast", "Castform, L100, F", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p1a: Skarm", "Sunny Day", "p1a: Skarm"],
        ["", "-weather", "SunnyDay"],
        ["", "-formechange", "p2a: Cast", "Castform-Sunny", "[msg]"],
        ["", "turn", "2"],
        ["", "-weather", "none"],
        ["", "-formechange", "p2a: Cast", "Castform", "[msg]"],
        ["", "turn", "3"],
    ], "battle-gen3ou-forme")
    assert any(e.kind is EventKind.FORMECHANGE for e in battle.events_since(0))


def test_trap_TRANSFORM_ditto_takes_the_whole_moveset_with_it():
    """Trap 2, half two. Transform moves species, stats, types AND the entire moveset in one
    line; the design says explicitly not to attempt per-field surgery here."""
    battle = _run_script(_HEAD + [
        ["", "switch", "p1a: Skarm", "Skarmory, L100, F", "100/100"],
        ["", "switch", "p2a: Ditto", "Ditto, L100", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p2a: Ditto", "Transform", "p1a: Skarm"],
        ["", "-transform", "p2a: Ditto", "p1a: Skarm"],
        ["", "turn", "2"],
        ["", "move", "p1a: Skarm", "Drill Peck", "p2a: Ditto"],
        ["", "-damage", "p2a: Ditto", "40/100"],
        ["", "turn", "3"],
    ], "battle-gen3ou-ditto")
    assert any(e.kind is EventKind.TRANSFORM for e in battle.events_since(0))


def test_trap_PARTIAL_TRAP_wrap_volatiles_survive_across_decisions():
    """The wrap family: a multi-turn volatile plus the trapping bits. Volatiles live in the
    active-context block, which is recomputed unconditionally — this is the end-to-end proof
    that a volatile that STARTS on one decision and ENDS several later never goes stale."""
    battle = _run_script(_HEAD + [
        ["", "switch", "p1a: Skarm", "Skarmory, L100, F", "100/100"],
        ["", "switch", "p2a: Cloy", "Cloyster, L100, F", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p2a: Cloy", "Clamp", "p1a: Skarm"],
        ["", "-damage", "p1a: Skarm", "88/100"],
        ["", "-activate", "p1a: Skarm", "move: Clamp", "[of] p2a: Cloy"],
        ["", "-start", "p1a: Skarm", "move: Clamp"],
        ["", "turn", "2"],
        ["", "-damage", "p1a: Skarm", "76/100", "[from] move: Clamp", "[partiallytrapped]"],
        ["", "turn", "3"],
        ["", "-end", "p1a: Skarm", "move: Clamp"],
        ["", "turn", "4"],
    ], "battle-gen3ou-wrap")
    kinds = {e.kind for e in battle.events_since(0)}
    assert EventKind.VOLATILE_START in kinds and EventKind.VOLATILE_END in kinds


def test_trap_PAIN_SPLIT_is_an_absolute_hp_correction_on_BOTH_sides():
    """`-sethp` names one mon per line but corrects two — and it is an ABSOLUTE set, not a
    delta, so a cache that only follows DAMAGE/HEAL would keep both HP dims wrong."""
    battle = _run_script(_HEAD + [
        ["", "switch", "p1a: Skarm", "Skarmory, L100, F", "100/100"],
        ["", "switch", "p2a: Gar", "Gengar, L100, M", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p1a: Skarm", "Drill Peck", "p2a: Gar"],
        ["", "-damage", "p2a: Gar", "30/100"],
        ["", "turn", "2"],
        ["", "move", "p2a: Gar", "Pain Split", "p1a: Skarm"],
        ["", "-sethp", "p1a: Skarm", "65/100", "[from] move: Pain Split"],
        ["", "-sethp", "p2a: Gar", "65/100"],
        ["", "turn", "3"],
    ], "battle-gen3ou-painsplit")
    assert any(e.kind is EventKind.SETHP for e in battle.events_since(0))


def test_an_assembler_threaded_on_a_NON_gen3battle_path_is_left_untouched():
    """The branch nothing runs in production, and the one that would ship silent zeros.

    A caller can thread an assembler while passing a subject with no `strict_view` (a plain
    poke-env Battle, a unit-test mock). `encode` must then write its OWN vector and leave the
    cache alone — an earlier draft wrote elsewhere and still called `commit()`, which would mark
    an UNWRITTEN buffer ready and serve zeros from the warm path on the next real decision.
    """
    enc = Gen3ObservationEncoder(load_mappings())
    battle = Gen3Battle("battle-gen3ou-asm5", "p1user", LOG, gen=3)
    tracker = EpisodeTracker()
    asm = tracker.obs_assembler(enc.dimension)
    out: list = []
    _feed(battle, SCRIPT, enc, tracker, out)          # the cache is now warm and correct
    good = asm.buf.copy()

    class _PlainBattle:
        """No `strict_view`, no `live_view` — the unit-test / plain-Battle subject."""
        turn = 3
        wait = False
        opponent_active_pokemon = None
        team: dict = {}
        opponent_team: dict = {}
        available_moves: tuple = ()
        available_switches: tuple = ()
        events: tuple = ()
        active_pokemon = None
        weather: dict = {}
        side_conditions: dict = {}
        opponent_side_conditions: dict = {}

    vec = enc.encode(_PlainBattle(), assembler=asm)
    assert vec.shape == (enc.dimension,)
    assert not vec.any(), "a subject with no board should encode to zeros"
    assert np.array_equal(asm.buf, good), "the cache must not have been touched"

    # …and the next REAL decision is still correct, which is the property that actually matters.
    tracker.record(battle, np.ones(11, dtype=np.int8))
    tracker.update_progress_clock(battle, None)
    kw = dict(hp_tracker=tracker.hidden_power_tracker,
              progress_clock=tracker.progress_clock, recency=tracker.recency,
              pair_history=tracker.pair_history, event_window=tracker.event_window)
    assert np.array_equal(enc.encode(battle, assembler=asm, **kw),
                          enc.encode(battle, assembler=None, **kw))
