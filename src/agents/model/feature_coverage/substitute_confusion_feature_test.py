"""feature_coverage — Substitute absorption, confusion self-hit, and the `|cant|` vocabulary.

The 2026-08-17 follow-up audit of the H-B event window (`gen3_frame_deletion_v1`). Three cells
were asked of the window; each verdict below was confirmed against the REAL protocol on real
bridge battles (`tmp/p0_event_window_audit.py`, node impl, `gen3ou` + the omniscient
`utils/bridge/damage_probe.js`), not inferred from reading intent.

Unlike its sibling files this one probes the FOLD (`EventWindowTracker`) rather than the
network hop: the questions here are "does the fact produce a row at all" and "does the row say
something TRUE", which the `assert_delta_reaches_network` half cannot answer — a row that
reaches the network carrying the wrong number is worse than one that never arrives. Where a
fact does have a home, the encoded/network half is already owned by
`move_resolution_feature_test.py` (cant reasons) and `static_blocks_feature_test.py`
(active-context volatiles).

MEASURED PROTOCOL (the ground truth these probes encode):

  Substitute absorbs a hit, sub SURVIVES:
      |move|p2a: Machamp|Seismic Toss|p1a: Blissey
      |-activate|p1a: Blissey|Substitute|[damage]
  Substitute absorbs a hit, sub BREAKS:
      |move|p2a: Machamp|Rock Smash|p1a: Blissey
      |-supereffective|p1a: Blissey
      |-end|p1a: Blissey|Substitute
  NEITHER form emits a `|-damage|` on the defender. The effectiveness / crit trio DOES still
  fire (Showdown emits it inside `getDamage`, which the substitute path calls).

  Confusion self-hit:
      |-activate|p2a: Snorlax|confusion
      |-damage|p2a: Snorlax|87/100|[from] confusion
  No `|move|` line and no `|cant|` line — gen3 never emits a cant for confusion, so its absence
  from `CANT_REASONS` is CORRECT, not a gap.

  Damp blocking Explosion / Self-Destruct (`abilities.ts` `onAnyTryMove`, gen3-reachable):
      |cant|p1a: Quagsire|ability: Damp|Self-Destruct|[of] p2a: Snorlax
"""
import numpy as np
import pytest

from agents.battle.battle_event import OPP, OURS, BattleEvent, EventKind
from agents.observation.constants import EVENT_T_CANT, EVENT_T_MOVE
from agents.observation.gen3_effects import (
    CANT_REASONS,
    VOLATILE_SLOTS,
    cant_reason_id,
    encode_volatiles,
)
from agents.training.episode_tracker import EventWindowTracker

OUR_SP = "blissey"
OPP_SP = "machamp"


# ---------------------------------------------------------------------------
# Fold helpers — drive the REAL EventWindowTracker on protocol-shaped events
# ---------------------------------------------------------------------------

def _ev(seq, kind, side, sp, turn=1, **value) -> BattleEvent:
    return BattleEvent(seq=seq, turn=turn, kind=kind, side=side,
                       actor_species=sp, value=dict(value))


def _fold(events, turn: int = 1) -> list:
    """Run the tracker with both actives established, then fold `events` for `turn`."""
    t = EventWindowTracker()
    t.update(0, [], OUR_SP, OPP_SP)          # establish actives (the decision-time resync)
    t.update(turn, events, OUR_SP, OPP_SP)
    return t.window()


def _move_rows(rows) -> list:
    return [r for r in rows if r["t"] == EVENT_T_MOVE]


# ===========================================================================
# 1. Substitute absorption — what the ATTACKER's move row reads
# ===========================================================================

class TestSubstituteAbsorption:
    """The attacker's row is the one fact this cell DOES represent, so pin it: an absorbed
    hit reads outcome=HIT with magnitude 0, and the effectiveness trio still attaches."""

    def test_absorbed_hit_reads_hit_with_zero_magnitude(self):
        """No `|-damage|` reaches the defender, so the move row's magnitude stays 0 while the
        outcome stays HIT — NOT miss, NOT fail. That pairing ("it connected and did nothing")
        is the only in-window trace of a Substitute eating an attack."""
        rows = _move_rows(_fold([
            _ev(0, EventKind.MOVE, OPP, OPP_SP, move_id="seismictoss"),
            # `-activate|...|Substitute|[damage]` -> EventKind.ACTIVATE (no DAMAGE line at all)
            _ev(1, EventKind.ACTIVATE, OURS, OUR_SP, effect="Substitute"),
        ]))
        assert len(rows) == 1, rows
        r = rows[0]
        assert r["hp_delta"] == 0.0, f"absorbed hit must carry no magnitude, got {r['hp_delta']}"
        assert not r["missed"] and not r["failed"], (
            "an absorbed hit is a HIT — the sim emits no |-miss|/|-fail|; "
            f"got missed={r['missed']} failed={r['failed']}")

    def test_effectiveness_still_attaches_through_a_substitute(self):
        """`|-supereffective|` IS emitted on the sub path (Showdown raises it inside
        `getDamage`, which `substitute.onTryPrimaryHit` calls), so the type read survives even
        though the damage does not. Confirmed live: Rock Smash vs a subbed Blissey printed
        `|-supereffective|` immediately before `|-end|...|Substitute`."""
        rows = _move_rows(_fold([
            _ev(0, EventKind.MOVE, OPP, OPP_SP, move_id="rocksmash"),
            _ev(1, EventKind.SUPEREFFECTIVE, OURS, OUR_SP),
            _ev(2, EventKind.VOLATILE_END, OURS, OUR_SP, effect="Substitute", op="end"),
        ]))
        assert rows[0]["eff"] == 1, f"super-effective must tag the move row, got {rows[0]['eff']}"

    def test_substitute_presence_rides_the_active_context_volatiles(self):
        """"A Substitute is up RIGHT NOW" is EXPLICIT — `substitute` is a binary volatile slot
        in the active-context block. What that cannot carry is the TRANSITION (it went up / it
        just broke), which is what the xfails below register."""
        assert "substitute" in VOLATILE_SLOTS
        vec = encode_volatiles(["substitute"])
        assert vec[VOLATILE_SLOTS.index("substitute")] == 1.0

    # ---- the gaps ---------------------------------------------------------

    @pytest.mark.xfail(strict=True, reason=(
        "2026-08-17 follow-up audit: the Substitute BREAKING has NO event-window home — see "
        "designs/ai_v9/design_frame_deletion_coverage_gaps.md §3.4. `|-end|...|Substitute` folds "
        "to a volatile-end event and `EventWindowTracker.update` has no branch for it, so the "
        "single most decision-relevant moment of a sub turn emits no row. Strict xfail so "
        "closing the gap turns this RED instead of silently passing."))
    def test_substitute_break_produces_an_event_row(self):
        rows = _fold([
            _ev(0, EventKind.MOVE, OPP, OPP_SP, move_id="rocksmash"),
            _ev(1, EventKind.VOLATILE_END, OURS, OUR_SP, effect="Substitute", op="end"),
        ])
        assert any(r["t"] not in (EVENT_T_MOVE,) for r in rows), (
            f"no row records the Substitute breaking; window = {rows}")

    @pytest.mark.xfail(strict=True, reason=(
        "2026-08-17 follow-up audit: a Substitute ABSORBING damage without breaking "
        "(`|-activate|...|Substitute|[damage]`) folds to EventKind.ACTIVATE, which the tracker "
        "has no branch for — see §3.4. Strict xfail so closing the gap turns this RED."))
    def test_substitute_absorbing_damage_produces_an_event_row(self):
        rows = _fold([
            _ev(0, EventKind.MOVE, OPP, OPP_SP, move_id="seismictoss"),
            _ev(1, EventKind.ACTIVATE, OURS, OUR_SP, effect="Substitute"),
        ])
        assert any(r["t"] != EVENT_T_MOVE for r in rows), (
            f"no row records the sub absorbing a hit; window = {rows}")

    @pytest.mark.xfail(strict=True, reason=(
        "2026-08-17 follow-up audit — a MISATTRIBUTION, not an absence (see §3.5). Substitute's "
        "own 25% HP cost is a `|-damage|` with NO `[from]` clause on the mon that is also the "
        "opponent's recorded move target, so the tracker's damage-attach rule credits it to the "
        "OPPONENT'S move. Confirmed live: `machamp seismictoss hp_delta=-0.2493` on a turn its "
        "Seismic Toss dealt ZERO (the sub ate it) — 0.2493 is exactly Blissey's 178/714 sub "
        "cost. Strict xfail so a fix turns this RED."))
    def test_substitute_self_cost_is_not_credited_to_the_opponents_move(self):
        rows = _move_rows(_fold([
            _ev(0, EventKind.MOVE, OPP, OPP_SP, move_id="seismictoss"),
            _ev(1, EventKind.VOLATILE_END, OURS, OUR_SP, effect="Substitute", op="end"),
            _ev(2, EventKind.MOVE, OURS, OUR_SP, move_id="substitute"),
            _ev(3, EventKind.VOLATILE_START, OURS, OUR_SP, effect="Substitute", op="start"),
            _ev(4, EventKind.DAMAGE, OURS, OUR_SP, amount=-0.2493),   # the sub's OWN cost
        ]))
        opp_row = next(r for r in rows if r["side"] == OPP)
        assert opp_row["hp_delta"] == 0.0, (
            "the defender's own Substitute cost was credited to the attacker's move: "
            f"hp_delta={opp_row['hp_delta']}")


# ===========================================================================
# 2. Confusion self-hit
# ===========================================================================

class TestConfusionSelfHit:
    def test_confusion_is_correctly_absent_from_the_cant_vocabulary(self):
        """gen3 emits `|-activate|...|confusion` + `|-damage|...|[from] confusion`, never a
        `|cant|`. So confusion's absence from CANT_REASONS is the RIGHT modelling of the
        protocol — the gap is that nothing else carries it either (see the xfails)."""
        assert "confusion" not in CANT_REASONS
        assert cant_reason_id(None) == 0

    def test_confused_state_rides_the_active_context_volatiles(self):
        """"I am confused RIGHT NOW" is EXPLICIT (a binary volatile slot). The 33% coin-flip
        risk is therefore visible; the RESOLVED outcome of a past flip is not."""
        assert "confusion" in VOLATILE_SLOTS
        vec = encode_volatiles(["confusion"])
        assert vec[VOLATILE_SLOTS.index("confusion")] == 1.0

    @pytest.mark.xfail(strict=True, reason=(
        "2026-08-17 follow-up audit: a confusion self-hit — a LOST TURN, strategically the twin "
        "of a full-paralysis CANT — produces NO event row (see §3.6). `|-activate|...|confusion` "
        "is EventKind.ACTIVATE (no tracker branch) and the self-damage carries a `[from]` clause. "
        "Strict xfail so closing the gap turns this RED."))
    def test_confusion_self_hit_produces_an_event_row(self):
        rows = _fold([
            _ev(0, EventKind.MOVE, OURS, OUR_SP, move_id="confuseray"),
            _ev(1, EventKind.ACTIVATE, OPP, OPP_SP, effect="confusion"),
            _ev(2, EventKind.DAMAGE, OPP, OPP_SP, amount=-0.12, reason="confusion"),
        ])
        assert any(r["t"] in (EVENT_T_CANT,) or r["actor"] == OPP_SP for r in rows), (
            f"nothing in the window says the confused mon lost its turn; window = {rows}")

    # gen3_event_semantics_v1 FIXED this — the xfail is removed, not flipped to skip.
    # Registered independently by the 2026-08-17 follow-up audit while the same defect
    # was being fixed from the other side; this test passing on a case the fix's own
    # author did not write is the stronger confirmation of the two.
    def test_confusion_self_damage_is_not_credited_to_the_opponents_move(self):
        rows = _move_rows(_fold([
            _ev(0, EventKind.MOVE, OURS, OUR_SP, move_id="confuseray"),
            _ev(1, EventKind.ACTIVATE, OPP, OPP_SP, effect="confusion"),
            _ev(2, EventKind.DAMAGE, OPP, OPP_SP, amount=-0.12, reason="confusion"),
        ]))
        our_row = next(r for r in rows if r["side"] == OURS)
        assert our_row["hp_delta"] == 0.0, (
            "the confused mon's SELF-damage was credited to our status move: "
            f"confuseray hp_delta={our_row['hp_delta']}")

    # gen3_event_semantics_v1 FIXED this — the xfail is removed, not flipped to skip.
    # Registered independently by the 2026-08-17 follow-up audit while the same defect
    # was being fixed from the other side; this test passing on a case the fix's own
    # author did not write is the stronger confirmation of the two.
    def test_end_of_turn_residuals_are_not_credited_to_the_attackers_move(self):
        rows = _move_rows(_fold([
            _ev(0, EventKind.MOVE, OPP, OPP_SP, move_id="rockslide"),
            _ev(1, EventKind.DAMAGE, OURS, OUR_SP, amount=-0.10),                    # the real hit
            _ev(2, EventKind.DAMAGE, OURS, OUR_SP, amount=-0.0625, reason="Sandstorm"),
            _ev(3, EventKind.DAMAGE, OURS, OUR_SP, amount=-0.0625, reason="brn"),
        ]))
        assert rows[0]["hp_delta"] == pytest.approx(-0.10), (
            "sandstorm + burn residuals were summed into the attacker's move magnitude: "
            f"hp_delta={rows[0]['hp_delta']} (want -0.10)")


# ===========================================================================
# 3. The `|cant|` vocabulary — freeze is COVERED; Damp is NOT
# ===========================================================================

class TestCantVocabulary:
    def test_freeze_is_a_first_class_cant_reason(self):
        """`gen3_event_window_v1`'s comment in constants.py abbreviates the reason list as
        "(full paralysis / sleep / flinch / recharge)". That is PROSE, not coverage: `frz` is in
        the vocabulary and always was. This probe exists so the abbreviation can never be read
        as the coverage again."""
        assert "frz" in CANT_REASONS
        assert cant_reason_id("frz") == CANT_REASONS.index("frz") + 1 != 0

    @pytest.mark.parametrize("raw", [
        "slp", "frz", "par", "flinch", "recharge",       # data/conditions.ts
        "Attract", "Disable", "Focus Punch",             # data/moves.ts (bare names)
        "move: Taunt", "move: Imprison",                 # data/moves.ts (move: prefix)
        "nopp",                                          # data/mods/gen3/moves.ts (Sleep Talk)
        "ability: Truant",                               # data/mods/gen3/abilities.ts
    ])
    def test_every_gen3_emittable_cant_reason_normalises(self, raw):
        """The full `add('cant', …)` census over the gen3-reachable Showdown sources. A reason
        the log can carry that the vocabulary lacks is the exact bug class `cant_id` was built
        to fix — and it CRASHES (crash-don't-drop), it does not degrade."""
        assert cant_reason_id(raw) > 0

    # gen3_damp_cant_v1 FIXED this (register §3.7). `damp` joins
    # CANT_REASONS_LIVE — NOT CANT_REASONS, which is frozen at 12 because it sizes
    # TURN_DELTA_DIM (159), the lag-frame width 79 archived runs recorded and the
    # prober still decodes. The row is also re-attributed to the BLOCKED mon via
    # `[of]`, so fixing the crash did not ship the lying row alongside it.
    def test_ability_damp_is_a_known_cant_reason(self):
        assert cant_reason_id("ability: Damp") > 0

    # gen3_damp_cant_v1 FIXED this. The event below now carries the `[of]` token the
    # real protocol line always sends — this test omitted it, and `[of]` is precisely
    # what the fix keys on. That is not incidental: `[of]` means "caused by that OTHER
    # mon", which is the only sound discriminator here. Keying on the `ability:` prefix
    # instead would be WRONG, because `ability: Truant` is equally ability-sourced and
    # entirely self-inflicted — it carries no `[of]`, and correctly keeps its own actor.
    def test_damp_cant_row_names_the_side_that_actually_lost_its_turn(self):
        rows = _fold([
            _ev(0, EventKind.MOVE, OPP, OPP_SP, move_id="selfdestruct"),
            # `|cant|p1a: Quagsire|ability: Damp|Self-Destruct|[of] p2a: Snorlax`
            _ev(1, EventKind.CANT, OURS, OUR_SP, reason="ability: Damp", move="selfdestruct",
                **{"of": f"p2a: {OPP_SP}", "of_side": OPP, "of_actor": OPP_SP}),
        ])
        cant = next(r for r in rows if r["t"] == EVENT_T_CANT)
        assert cant["side"] == OPP, (
            "the CANT row blames the Damp HOLDER, not the mon whose move was blocked: "
            f"side={cant['side']} actor={cant['actor']} move_id={cant['move_id']}")


def test_cant_reason_ids_are_collision_free():
    """Belt-and-braces on the embedding axis: 1-based, dense, no duplicates, 0 reserved."""
    ids = [cant_reason_id(r) for r in CANT_REASONS]
    assert ids == list(range(1, len(CANT_REASONS) + 1))
    assert np.all(np.asarray(ids) > 0)
