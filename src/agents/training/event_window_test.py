"""gen3_event_window_v1 (Tier H-B) — the event fold, the obs block, and the seat consumer.

Three layers, one file: EventWindowTracker's fold rules (attach/idempotence/actives/forced
windows), the encoder's 20-column row contract (offsets, padding-at-front, recency), and the
EventSeats module's build/mask contract (OFF builds nothing; PAD rows get zero attention weight
by key-mask; ON forward reads the block).
"""
import numpy as np
import pytest
import torch

from agents.battle.battle_event import BattleEvent, EventKind, OURS, OPP
from agents.observation.constants import (
    EVENT_T_BOOST, EVENT_T_CANT, EVENT_T_FAINT, EVENT_T_MOVE, EVENT_T_STATUS_APPLIED,
    EVENT_T_SWITCH_IN, EVENT_TOKEN_DIM, EVENT_WINDOW_DIM, EVENT_WINDOW_N, OFFSET_EVENT_WINDOW,
)
from agents.training.episode_tracker import EventWindowTracker


def _ev(seq, turn, kind, side=None, sp=None, **value):
    return BattleEvent(seq=seq, turn=turn, kind=kind, side=side, actor_species=sp, value=value)


def _basic_events():
    return [
        _ev(1, 1, EventKind.SWITCH, OURS, "tyranitar", prev_active=None),
        _ev(2, 1, EventKind.SWITCH, OPP, "skarmory", prev_active=None),
        _ev(3, 2, EventKind.MOVE, OURS, "tyranitar", move_id="rockslide"),
        _ev(4, 2, EventKind.DAMAGE, OPP, sp="skarmory", amount=-0.31),
        _ev(5, 2, EventKind.CRIT, OURS, "tyranitar", op="crit"),
        _ev(6, 2, EventKind.SUPEREFFECTIVE, OPP, "skarmory", multiplier=2.0),
        _ev(7, 2, EventKind.MOVE, OPP, "skarmory", move_id="spikes"),
        _ev(8, 3, EventKind.STATUS, OPP, "skarmory", status="par"),
        _ev(9, 3, EventKind.BOOST, OURS, "tyranitar", stat="atk", amount=2),
        _ev(10, 4, EventKind.FAINT, OPP, "skarmory"),
    ]


def test_fold_attaches_modifiers_and_is_seq_idempotent():
    t = EventWindowTracker(maxlen=16)
    t.update(4, _basic_events(), "tyranitar", None)
    t.update(4, _basic_events(), "tyranitar", None)    # replay: nothing may double
    w = t.window()
    kinds = [r["t"] for r in w]
    assert kinds == [EVENT_T_SWITCH_IN, EVENT_T_SWITCH_IN, EVENT_T_MOVE, EVENT_T_MOVE,
                     EVENT_T_STATUS_APPLIED, EVENT_T_BOOST, EVENT_T_FAINT]
    our_move = w[2]
    assert our_move["move_id"] == "rockslide" and our_move["target"] == "skarmory"
    assert our_move["hp_delta"] == pytest.approx(-0.31)
    assert our_move["crit"] and our_move["eff"] == 1                 # supereffective
    assert our_move["we_first"] is True
    assert w[3]["we_first"] is False                                  # skarmory moved second
    assert w[4]["status"] == 2                                        # par
    assert w[5]["hp_delta"] == pytest.approx(2 / 1.0)                 # boost magnitude raw (+2)
    assert t._forced["opp"] is True                                   # faint opened the window


def test_residual_and_recoil_damage_never_attach():
    """The attach rule's two guards: a `[from]`-claused DAMAGE (recoil / sand / status / item)
    and a clause-free DAMAGE on a NON-target mon must both leave the move's hp_delta alone."""
    t = EventWindowTracker(maxlen=16)
    evs = [
        _ev(1, 1, EventKind.SWITCH, OURS, "tyranitar", prev_active=None),
        _ev(2, 1, EventKind.SWITCH, OPP, "skarmory", prev_active=None),
        _ev(3, 2, EventKind.MOVE, OURS, "tyranitar", move_id="doubleedge"),
        _ev(4, 2, EventKind.DAMAGE, OPP, sp="skarmory", amount=-0.20),                  # the hit
        BattleEvent(seq=5, turn=2, kind=EventKind.DAMAGE, side=OURS,
                    actor_species="tyranitar", value={"amount": -0.07, "from": "Recoil"}),
        BattleEvent(seq=6, turn=2, kind=EventKind.DAMAGE, side=OPP,
                    actor_species="skarmory", value={"amount": -0.06, "from": "Sandstorm"}),
        _ev(7, 2, EventKind.DAMAGE, OPP, sp="blissey", amount=-0.10),                   # not the target
    ]
    t.update(2, evs, "tyranitar", "skarmory")
    mv = [r for r in t.window() if r["t"] == EVENT_T_MOVE][0]
    assert mv["hp_delta"] == pytest.approx(-0.20)


def test_forced_window_tags_and_clears():
    t = EventWindowTracker(maxlen=16)
    evs = _basic_events() + [
        _ev(11, 4, EventKind.SWITCH, OPP, "blissey", prev_active=None),   # replacement
    ]
    t.update(4, evs, "tyranitar", "blissey")
    w = t.window()
    assert w[-1]["t"] == EVENT_T_SWITCH_IN and w[-1]["actor"] == "blissey"
    assert w[-1]["forced_window"] == 1.0        # emitted while the opp slot was empty
    assert t._forced["opp"] is False            # the arrival closed it


def test_window_is_bounded():
    t = EventWindowTracker(maxlen=4)
    evs = [_ev(i, i, EventKind.MOVE, OURS, "tyranitar", move_id="rockslide")
           for i in range(1, 10)]
    t.update(9, evs, None, None)
    assert len(t.window()) == 4


def test_encoder_writes_rows_back_padded_and_typed():
    """The obs contract: rows most-recent-LAST, zero-padding at the FRONT, ids in the id
    columns, valid=1 on real rows, recency log-saturated."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    enc = Gen3ObservationEncoder(load_mappings())
    t = EventWindowTracker()
    t.update(4, _basic_events(), "tyranitar", None)

    class _B:                                       # the minimal battle stub encode() accepts
        team = {}
        opponent_team = {}
        active_pokemon = None
        opponent_active_pokemon = None
        available_moves = []
        available_switches = []
        weather = None
        side_conditions = {}
        opponent_side_conditions = {}
        turn = 4

    vec = enc.encode(_B(), event_window=t)
    block = vec[OFFSET_EVENT_WINDOW:OFFSET_EVENT_WINDOW + EVENT_WINDOW_DIM] \
        .reshape(EVENT_WINDOW_N, EVENT_TOKEN_DIM)
    n_valid = int(block[:, 18].sum())
    assert n_valid == 7
    assert float(block[: EVENT_WINDOW_N - n_valid].sum()) == 0.0     # front padding all-zero
    move_row = block[EVENT_WINDOW_N - n_valid + 2]                   # our rockslide
    assert move_row[0] == EVENT_T_MOVE
    assert move_row[2] == 1.0                                        # our side
    assert move_row[4] > 0                                           # move num present
    assert move_row[5] == pytest.approx(-0.31)                       # attributed damage
    assert move_row[9] == 1.0 and move_row[11] == 1.0                # crit + supereffective
    faint_row = block[EVENT_WINDOW_N - 1]
    assert faint_row[0] == EVENT_T_FAINT and faint_row[2] == -1.0
    assert 0.0 <= float(block[:, 16].max()) <= 1.0                   # recency in range


def test_event_seats_off_builds_nothing_on_reads_block():
    pytest.importorskip("sb3_contrib")
    from agents.model.identity_init_test import _build_real_policy

    off, _ = _build_real_policy()
    assert not any("history_events" in k for k in off.policy.state_dict())
    on, enc = _build_real_policy(history_events=True)
    fe = on.policy.features_extractor
    assert any("history_events.proj" in k for k in on.policy.state_dict())
    # PAD rows are key-masked; a real row is not
    ev = torch.zeros(2, EVENT_WINDOW_N, EVENT_TOKEN_DIM)
    ev[:, -1, 0] = EVENT_T_MOVE
    ev[:, -1, 18] = 1.0
    tokens, pad = fe.history_events(ev, fe.embeddings)
    assert tokens.shape == (2, EVENT_WINDOW_N, tokens.shape[-1])
    assert bool(pad[:, :-1].all()) and not bool(pad[:, -1].any())
    # and the full forward moves when the block content changes
    rng = np.random.default_rng(0)
    obs = torch.as_tensor(rng.random((2, enc.dimension), dtype=np.float32))
    o1 = {"observation": obs.clone(), "action_mask": torch.ones(2, 11)}
    obs2 = obs.clone()
    sl = slice(OFFSET_EVENT_WINDOW, OFFSET_EVENT_WINDOW + EVENT_WINDOW_DIM)
    obs2[:, sl] = 0.0
    obs2[:, OFFSET_EVENT_WINDOW + 18] = 1.0          # one valid pad-ish row, different content
    o2 = {"observation": obs2, "action_mask": torch.ones(2, 11)}
    with torch.no_grad():
        pi1, vf1 = fe(o1)
        pi2, vf2 = fe(o2)
    assert not torch.allclose(pi1, pi2) or not torch.allclose(vf1, vf2)


def test_pre_floor_config_is_refused():
    """gen3_frame_deletion_v1 raised MIGRATION_FLOOR to 90 (ARCH_SIGNATURE bumped), so a
    pre-floor config is REFUSED rather than migrated — the floor's stated purpose: "refuses
    pre-floor configs outright instead of walking dead branches". This asserts the behaviour
    that is now true rather than propping up a branch nothing can reach."""
    from agents.model.model_version import ModelVersionError, _migrate_config
    with pytest.raises(ModelVersionError, match="PRE-GENERATION|floor"):
        _migrate_config({"config_version": 80})



# ---------------------------------------------------------------------------
# gen3_frame_deletion_v1 — EVENT_T_CANT, the lag frames' one unsubstituted fact
# ---------------------------------------------------------------------------

def test_cant_event_folds_with_its_reason():
    """A `|cant|` must produce a CANT row carrying its REASON as `cant_id`.

    This exists because it shipped broken for one commit. The fold read `e.cause`, which
    `BattleEvent` does not define (`e.reason` is the accessor) — so EVERY live battle in which a
    mon was fully paralysed / asleep / flinched / recharging raised `AttributeError` mid-fold.
    The whole unit tier passed clean: nothing there drives a real event log, so the crash only
    appeared in the sim tier (16 failures across the bridge, better-line, falsifier and obs-parity
    suites — all one bug). This test is the cheap deterministic guard that would have caught it,
    and it fails if the attribute name is ever changed back."""
    from agents.observation.gen3_effects import cant_reason_id
    t = EventWindowTracker(maxlen=16)
    t.update(1, [_ev(1, 1, EventKind.SWITCH, OURS, "snorlax", prev_active=None)], "snorlax", "gengar")
    t.update(2, [_ev(2, 2, EventKind.CANT, OURS, "snorlax", reason="par")], "snorlax", "gengar")
    rows = [r for r in t.window() if r["t"] == EVENT_T_CANT]
    assert len(rows) == 1, f"expected exactly one CANT row, got {len(rows)}"
    r = rows[0]
    assert r["actor"] == "snorlax" and r["side"] == OURS
    assert r["cant"] == "par", f"the reason must survive the fold verbatim, got {r['cant']!r}"
    assert cant_reason_id(r["cant"]) > 0, "a real reason must map to a nonzero id"


def test_cant_reason_ids_are_distinct_and_zero_means_none():
    """Distinct reasons must get distinct ids, and 0 must be reserved for 'not a cant row'.

    A collapse here is silent: the row still exists, the column still reads nonzero, and the model
    simply cannot tell paralysis from sleep. (The sibling status mapping DID collapse this way
    during the frame-deletion work — a Status enum's `.value` is an int, so every status hashed to
    one id — which is why this is asserted rather than assumed.)"""
    from agents.observation.gen3_effects import CANT_REASONS, cant_reason_id
    assert cant_reason_id(None) == 0
    ids = {r: cant_reason_id(r) for r in CANT_REASONS}
    assert 0 not in ids.values(), "0 is reserved for 'no cant' and must not collide with a reason"
    assert len(set(ids.values())) == len(CANT_REASONS), f"reason ids collide: {ids}"


# ---------------------------------------------------------------------------
# The residual-attribution defect (shipped v81, found 2026-08-17 by coverage audit)
# ---------------------------------------------------------------------------

def test_residual_damage_is_not_folded_into_the_move_magnitude():
    """A move's magnitude is ITS OWN hit — residual chip on the same target must not join it.

    The defect this pins: `EventWindowTracker` tested `e.value.get("from")` to mean "this damage
    carries a [from] clause, so it is NOT the move's own hit". On a DAMAGE event the parser
    stores that clause under `value["reason"]` instead, so the raw key was ALWAYS absent and the
    guard NEVER fired. Every sandstorm / burn / poison / Leech Seed / recoil tick landing on the
    move's target that turn was added to the move's attributed hp_delta — measured -0.3625 for a
    -0.3000 hit under sandstorm. It shipped in v81 and trained through two generations.

    Asserted on the ARITHMETIC, not on the guard's spelling: a future refactor that reaches for
    the wrong key again fails here regardless of how it phrases the test."""
    t = EventWindowTracker(maxlen=16)
    t.update(1, [_ev(1, 1, EventKind.SWITCH, OURS, "snorlax", prev_active=None),
                 _ev(2, 1, EventKind.SWITCH, OPP, "tyranitar", prev_active=None)],
             "snorlax", "tyranitar")
    t.update(2, [
        _ev(3, 2, EventKind.MOVE, OPP, "tyranitar", move_id="rockslide"),
        _ev(4, 2, EventKind.DAMAGE, OURS, sp="snorlax", amount=-0.30),                 # the hit
        _ev(5, 2, EventKind.DAMAGE, OURS, sp="snorlax", amount=-0.0625, reason="Sandstorm"),
        _ev(6, 2, EventKind.DAMAGE, OURS, sp="snorlax", amount=-0.0625, reason="brn"),
    ], "snorlax", "tyranitar")
    moves = [r for r in t.window() if r["t"] == EVENT_T_MOVE]
    assert len(moves) == 1
    assert abs(moves[0]["hp_delta"] - (-0.30)) < 1e-9, (
        f"move magnitude {moves[0]['hp_delta']} != -0.30 — residual damage was folded in")


def test_from_clause_reads_both_storage_keys():
    """`[from]` lives under two different keys depending on event kind — one accessor must span it.

    DAMAGE/HEAL/SETHP/STATUS store it as `value["reason"]`; ITEM/ENDITEM/WEATHER/effect kinds
    merge the parsed cause dict so it lands under `value["from"]`. BOTH raw accessors therefore
    return None for half the event kinds, silently. `from_clause` is the one safe reader, and
    this pins that — the inconsistency is the actual defect generator, not the one call site."""
    dmg = _ev(1, 1, EventKind.DAMAGE, OURS, "snorlax", amount=-0.06, reason="Sandstorm")
    itm = _ev(2, 1, EventKind.ENDITEM, OURS, "snorlax", item="leftovers", **{"from": "move: Knock Off"})
    assert dmg.from_cause is None, "precondition: the raw key really is absent on DAMAGE"
    assert dmg.from_clause == "Sandstorm"
    assert itm.from_clause == "move: Knock Off"
    assert _ev(3, 1, EventKind.DAMAGE, OURS, "snorlax", amount=-0.30).from_clause is None


# ---------------------------------------------------------------------------
# gen3_damp_cant_v1 — the ability-sourced cant (register §3.7)
# ---------------------------------------------------------------------------

def test_damp_cant_does_not_crash_and_is_attributed_to_the_blocked_mon():
    """`ability: Damp` must fold cleanly AND name the mon that actually lost its turn.

    TWO defects rode this one row and fixing only the first would have shipped the second.

    (a) CRASH. `damp` was absent from the cant vocabulary, and `normalize_cant_reason` is
        crash-don't-drop — so the first blocked Explosion raised out of `state_encoder.encode`
        and killed the episode, and in training the run. Damp is gen3-legal (Quagsire, Golduck,
        Politoed, …) and Explosion is ubiquitous in gen3ou, so it is reachable in ordinary play;
        reproduced on battle #1 of a scripted Quagsire-vs-Snorlax bridge battle.

    (b) A LYING ROW. Showdown files an ability-sourced cant against the ability HOLDER with the
        BLOCKED move as its argument:
            |cant|p1a: Quagsire|ability: Damp|Self-Destruct|[of] p2a: Snorlax
        Taken at face value that says Quagsire could not use Self-Destruct — a move it never had
        — while the side that really lost its turn goes unmentioned. The `[of]` mon is resolved
        at EMISSION (the log gains the fact; the fold stays pure) and preferred here."""
    t = EventWindowTracker(maxlen=16)
    t.update(1, [_ev(1, 1, EventKind.SWITCH, OURS, "quagsire", prev_active=None),
                 _ev(2, 1, EventKind.SWITCH, OPP, "snorlax", prev_active=None)],
             "quagsire", "snorlax")
    t.update(2, [_ev(3, 2, EventKind.CANT, OURS, "quagsire",
                     reason="ability: Damp", move="selfdestruct",
                     **{"of": "p2a: Snorlax", "of_side": OPP, "of_actor": "snorlax"})],
             "quagsire", "snorlax")
    rows = [r for r in t.window() if r["t"] == EVENT_T_CANT]
    assert len(rows) == 1
    assert rows[0]["actor"] == "snorlax", "the row must name the mon that LOST its turn"
    assert rows[0]["side"] == OPP, "...and its side, not the Damp holder's"
    assert rows[0]["cant"] == "ability: Damp"


def test_an_ordinary_cant_still_uses_its_own_actor():
    """The `[of]` preference must not disturb self-inflicted cants, where holder == blocked mon."""
    t = EventWindowTracker(maxlen=16)
    t.update(1, [_ev(1, 1, EventKind.SWITCH, OURS, "snorlax", prev_active=None)], "snorlax", None)
    t.update(2, [_ev(2, 2, EventKind.CANT, OURS, "snorlax", reason="par")], "snorlax", None)
    r = [x for x in t.window() if x["t"] == EVENT_T_CANT][0]
    assert r["actor"] == "snorlax" and r["side"] == OURS


def test_truant_is_ability_sourced_but_keeps_its_OWN_actor():
    """The `[of]` preference must key on `[of]`, NOT on the `ability:` prefix.

    `ability: Truant` is every bit as ability-sourced as `ability: Damp` and is entirely
    SELF-inflicted — the loafing mon blocks itself. Showdown sends no `[of]` for it, because
    `[of]` means "caused by that OTHER mon". A rule that re-attributed on the prefix would
    hand every Truant turn to the opponent, turning one fixed lie into a new one."""
    t = EventWindowTracker(maxlen=16)
    t.update(1, [_ev(1, 1, EventKind.SWITCH, OURS, "slaking", prev_active=None)], "slaking", None)
    t.update(2, [_ev(2, 2, EventKind.CANT, OURS, "slaking", reason="ability: Truant")],
             "slaking", None)
    r = [x for x in t.window() if x["t"] == EVENT_T_CANT][0]
    assert r["actor"] == "slaking" and r["side"] == OURS, "Truant must keep its own actor"


def test_the_archive_cant_vocabulary_is_FROZEN():
    """`CANT_REASONS` sizes TURN_DELTA_DIM (159) — the lag-frame width 79 archived runs recorded.

    The frames are deleted from the live obs, so `TurnDeltaEncoder` survives only as the prober's
    decoder for that archive. Growing `CANT_REASONS` would shift every offset after the cant block
    and make it mis-slice historical data — silently, since it would still return a plausible
    dict. New reasons go in `CANT_REASONS_LIVE`. This test is what makes that split enforced
    rather than merely intended."""
    from agents.observation.gen3_effects import CANT_REASONS, CANT_DIM, CANT_REASONS_LIVE
    from agents.observation.turn_delta_encoder import TURN_DELTA_DIM
    assert CANT_DIM == 12 and TURN_DELTA_DIM == 159, "the ARCHIVE format moved — the prober will mis-slice"
    assert CANT_REASONS_LIVE[:len(CANT_REASONS)] == CANT_REASONS, "live must EXTEND the archive, never reorder it"
    assert "damp" in CANT_REASONS_LIVE and "damp" not in CANT_REASONS
