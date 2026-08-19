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
    EVENT_EFF_GROUP, EVENT_OUTCOME_GROUP, EventCol as C,
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
    n_valid = int(block[:, C.VALID].sum())
    assert n_valid == 7
    assert float(block[: EVENT_WINDOW_N - n_valid].sum()) == 0.0     # front padding all-zero
    move_row = block[EVENT_WINDOW_N - n_valid + 2]                   # our rockslide
    assert move_row[C.TYPE] == EVENT_T_MOVE
    assert move_row[C.ACTOR_SIDE] == 1.0                             # our side
    assert move_row[C.MOVE] > 0                                      # move num present
    assert move_row[C.MAGNITUDE] == pytest.approx(-0.31)             # attributed damage
    assert move_row[C.CRIT] == 1.0 and move_row[C.EFF_SUPER] == 1.0  # crit + supereffective
    faint_row = block[EVENT_WINDOW_N - 1]
    assert faint_row[C.TYPE] == EVENT_T_FAINT and faint_row[C.ACTOR_SIDE] == -1.0
    assert 0.0 <= float(block[:, C.TURNS_AGO].max()) <= 1.0          # recency in range


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
    ev[:, -1, C.TYPE] = EVENT_T_MOVE
    ev[:, -1, C.VALID] = 1.0
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
    obs2[:, OFFSET_EVENT_WINDOW + C.VALID] = 1.0     # one valid pad-ish row, different content
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
# The DECISION CYCLE — an event must reach the obs of the decision AFTER it.
#
# Every test above drives `EventWindowTracker.update` directly with a hand-built event list, so
# none of them says anything about WHICH events a real decision folds. That window comes from
# `EpisodeTracker`: `record()` captures `battle.event_cursor`, and the NEXT decision's
# `update_progress_clock()` folds `events_since(that cursor)`. The tests below drive that real
# chain (scripted `Gen3Battle` -> record -> update_progress_clock -> encode) for the two rows
# whose producers do NOT look like the others.
# ---------------------------------------------------------------------------

_CANON_OPENING = [
    ["", "player", "p1", "p1user", "", ""],
    ["", "player", "p2", "p2user", "", ""],
    ["", "teamsize", "p1", "6"],
    ["", "teamsize", "p2", "6"],
    ["", "gametype", "singles"],
    ["", "gen", "3"],
    ["", "tier", "[Gen 3] OU"],
    ["", "start"],
    ["", "switch", "p1a: Zappy", "Zapdos, L100", "100/100"],
    ["", "switch", "p2a: Tyra", "Tyranitar, L100, M", "100/100"],
    ["", "turn", "1"],
]


def _opened_battle():
    """A scripted `Gen3Battle` with both leads in and turn 1 marked."""
    import logging

    from agents.battle.gen3_battle import Gen3Battle

    b = Gen3Battle("battle-gen3ou-canon", "p1user", logging.getLogger("event-window-test"), gen=3)
    for line in _CANON_OPENING:
        b.parse_message(line)
    return b


def _decide(tracker, battle, action=None):
    """One decision, in `Gen3Env.embed_battle`'s exact order: record -> update_progress_clock
    (the ONLY caller of `EventWindowTracker.update`) -> the caller encodes. Returns the folded
    delta, as the env caches it."""
    tracker.record(battle, np.ones(11, dtype=np.int8))
    delta = tracker.update_progress_clock(battle, None)
    if action is not None:
        tracker.advance(action)
    return delta


def _event_rows(obs):
    """The obs event block as (type, row) pairs for its VALID rows, oldest-first."""
    return [
        (int(obs[OFFSET_EVENT_WINDOW + r * EVENT_TOKEN_DIM + C.TYPE]), r)
        for r in range(EVENT_WINDOW_N)
        if obs[OFFSET_EVENT_WINDOW + r * EVENT_TOKEN_DIM + C.VALID] >= 0.5
    ]


def _encoder():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    return Gen3ObservationEncoder(load_mappings())


def test_an_out_of_band_choice_rejection_reaches_the_NEXT_decisions_obs():
    """A refused switch must arrive in the obs of the decision the server re-prompts for.

    `CHOICE_REJECTED` is the ONE event kind recorded OUTSIDE the parse pass: poke-env intercepts
    `|error|[Unavailable choice]` in `_handle_battle_message` before `parse_message` ever sees it,
    and calls `Gen3Battle.record_choice_rejected` directly. So the fact this pins is that an
    out-of-band append still lands INSIDE the next decision's `[cursor, now)` window — the cursor
    is captured at `record()` time against the same log `_record` appends to, and nothing about
    the parse pass is load-bearing for that.

    It is the trap-reveal signal: after the frame deletion this row is the model's only route to
    "we tried to pivot and were refused", so a miss here is silent GIGO, not a lost nicety.
    """
    from agents.observation.constants import EVENT_T_SWITCH_REJECTED
    from agents.training.episode_tracker import EpisodeTracker

    b, tr, enc = _opened_battle(), EpisodeTracker(history_cap=1), _encoder()

    _decide(tr, b, action=2)                                  # decision 1: we press a switch
    obs1 = enc.encode(b, event_window=tr.event_window)
    assert not any(t == EVENT_T_SWITCH_REJECTED for t, _ in _event_rows(obs1)), \
        "no rejection has happened yet — a row here would be a fabrication"

    # The server refuses it. This is the whole out-of-band path, called exactly as poke-env does.
    b.record_choice_rejected(
        ["", "error", "[Unavailable choice] Can't switch: The active Pokemon is trapped"])

    delta = _decide(tr, b)                                    # decision 2: the re-prompt
    assert delta.attempted_switch_rejected, "the TurnDelta fold must see the rejection"
    obs2 = enc.encode(b, event_window=tr.event_window)
    rows = _event_rows(obs2)
    assert rows, "decision 2 folded an empty window — the cursor did not cover the rejection"
    typ, row = rows[-1]
    assert typ == EVENT_T_SWITCH_REJECTED, (
        f"the NEWEST row must be the rejection (it is the last event on the wire — the "
        f"`|error|` is followed only by the re-prompt `|request|`, which is not an event); "
        f"got type {typ}")
    off = OFFSET_EVENT_WINDOW + row * EVENT_TOKEN_DIM
    assert obs2[off + C.ACTOR_SIDE] == 1.0, "a rejection is always OURS"
    assert obs2[off + C.ACTOR_SPECIES] > 0.0, \
        "the actor is the trapped mon — an unattributed row cannot say WHO is stuck"


def test_a_cant_line_reaches_the_obs_through_the_same_decision_cycle():
    """The CANT row's end-to-end pin — the sibling of the rejection above.

    CANT is fed by an ORDINARY `|cant|` protocol line (the normal parse pass), so it does NOT
    share the rejection's out-of-band exposure. That is asserted here rather than assumed: the
    two rows arrived in this window for the same reason (the lag frames were their only route
    before `gen3_frame_deletion_v1`), the CANT fold reads the un-obvious `blocked_side` /
    `blocked_actor` attribution, and nothing else drives a `|cant|` line all the way to an obs
    index."""
    from agents.observation.gen3_effects import cant_reason_id
    from agents.training.episode_tracker import EpisodeTracker

    b, tr, enc = _opened_battle(), EpisodeTracker(history_cap=1), _encoder()

    _decide(tr, b, action=6)
    b.parse_message(["", "cant", "p1a: Zappy", "par"])
    _decide(tr, b)

    obs = enc.encode(b, event_window=tr.event_window)
    rows = _event_rows(obs)
    assert rows, "decision 2 folded an empty window — the |cant| never reached it"
    typ, row = rows[-1]
    assert typ == EVENT_T_CANT, f"the newest row must be the CANT, got type {typ}"
    off = OFFSET_EVENT_WINDOW + row * EVENT_TOKEN_DIM
    assert obs[off + C.CANT] == float(cant_reason_id("par")), \
        "the reason must reach the obs, not just the record — a bare 'could not move' row " \
        "cannot tell paralysis from sleep"
    assert obs[off + C.ACTOR_SIDE] == 1.0 and obs[off + C.ACTOR_SPECIES] > 0.0


def test_the_window_block_is_ZERO_without_update_progress_clock():
    """The trap that made the trapping-signals fuzz read FAIL on a working signal.

    `update_progress_clock` is the ONLY caller of `EventWindowTracker.update`, and `encode`'s
    `event_window=` is optional (None leaves the block zero). Miss either and the whole 32-row
    block reads structurally zero — which a presence check on any single row type reports as
    "the signal never reached the model", indistinguishable from a real miss. Pinned so the
    next harness author is told this by a test name instead of by a day of debugging."""
    from agents.training.episode_tracker import EpisodeTracker

    b, tr, enc = _opened_battle(), EpisodeTracker(history_cap=1), _encoder()
    tr.record(b, np.ones(11, dtype=np.int8))                 # record only — no clock update
    tr.advance(2)
    b.record_choice_rejected(["", "error", "[Unavailable choice] trapped"])
    tr.record(b, np.ones(11, dtype=np.int8))

    block = slice(OFFSET_EVENT_WINDOW, OFFSET_EVENT_WINDOW + EVENT_WINDOW_DIM)
    assert not np.any(enc.encode(b, event_window=tr.event_window)[block]), \
        "without update_progress_clock the tracker's window is never fed"
    assert not np.any(enc.encode(b)[block]), \
        "without event_window= the encoder writes nothing, however well fed the tracker is"


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


def test_the_fuzz_ORACLE_reads_the_from_clause_too():
    """The mirrored-oracle half of the same defect (positional-binding sweep).

    `test_residual_damage_is_not_folded_into_the_move_magnitude` pins the TRACKER. The
    independent fold in `event_window_fuzz_test` — the thing that is supposed to catch the
    tracker regressing — kept the ORIGINAL `e.value.get("from")` spelling, which is
    unconditionally falsy on DAMAGE. An oracle that repeats its subject's key drift cannot
    detect that drift coming back, and 30 battles of green fuzz would say nothing about it."""
    from agents.training.poke_env_gaps.event_window_fuzz_test import attributable_damage

    hit = _ev(1, 2, EventKind.DAMAGE, OURS, "snorlax", amount=-0.30)
    sand = _ev(2, 2, EventKind.DAMAGE, OURS, "snorlax", amount=-0.0625, reason="Sandstorm")
    assert attributable_damage(hit) is True
    assert attributable_damage(sand) is False, (
        "the oracle counted sandstorm chip as part of the move's hit — it is reading "
        "`value['from']`, which DAMAGE never carries, instead of `from_clause`.")


def test_the_fuzz_ORACLE_derives_the_three_id_columns_it_used_to_declare_unmodelled():
    """`CANT` / `FAINT_CAUSE` / `ITEM_TRANSITION` are modelled by the fuzz oracle now, and its
    two derivations are written independently of the producer's — so they need their own cheap
    pin, because a mistyped label reaches `FAINT_CAUSE_VOCAB.index()` only mid-battle.

    What this asserts is the SEMANTIC content, not agreement with the tracker (agreement is
    what the fuzz measures over real battles): every distinct residual cause gets its own id,
    a self-KO outranks whatever clause the last damage carried, and the three ways a gen3 item
    stops being held stay distinct from a mere reveal."""
    from agents.training.poke_env_gaps.event_window_fuzz_test import (
        oracle_faint_cause_id, oracle_item_transition,
    )
    from agents.battle.battle_event import EventKind as K
    from agents.observation.constants import ITEM_TR_REVEALED

    causes = {c: oracle_faint_cause_id(fc, False) for c, fc in (
        ("attack", None), ("hazard", "Spikes"), ("weather", "Sandstorm"),
        ("status", "psn"), ("recoil", "Recoil"), ("leechseed", "Leech Seed"),
        ("other", "item: Life Orb"))}
    assert len(set(causes.values())) == len(causes), f"faint causes collide: {causes}"
    assert 0 not in causes.values(), "0 is reserved for 'not a FAINT row'"
    selfko = oracle_faint_cause_id(None, True)
    assert oracle_faint_cause_id("Sandstorm", True) == selfko, \
        "a self-KO must outrank whatever clause the last chip carried"
    assert selfko not in causes.values(), "selfko must be its own id, not folded into another"

    trs = {oracle_item_transition(K.ITEM, None),
           oracle_item_transition(K.ENDITEM, None),                    # berry spent
           oracle_item_transition(K.ENDITEM, "move: Knock Off"),       # permanent in ADV
           oracle_item_transition(K.ENDITEM, "move: Trick")}           # the opp holds it now
    assert len(trs) == 4, f"the item transitions collapsed onto each other: {trs}"
    assert oracle_item_transition(K.ITEM, "move: Trick") == ITEM_TR_REVEALED, \
        "an |-item| line is a DISCLOSURE — it must not read as a transfer whatever it cites"


def test_an_unknown_status_name_CRASHES_rather_than_reading_as_none():
    """Crash-don't-drop at the H-B status vocabulary (the `normalize_cant_reason` contract).

    `_EVENT_STATUS_IDS.get(name, 0)` coded any unrecognised status as 0 — the id that MEANS
    "no status". A parser or vocabulary drift would therefore tell the model the opponent was
    clean on a turn it was badly poisoned, with no metric anywhere to show it."""
    from agents.training.episode_tracker import _event_status_id

    assert _event_status_id(None) == 0 and _event_status_id("") == 0
    assert _event_status_id("tox") == 6 and _event_status_id("TOX") == 6
    with pytest.raises(ValueError, match="unknown status"):
        _event_status_id("frostbite")
    # …but a `[...]` protocol modifier in the status slot is ABSENCE, not a bad name. The parser
    # reads the status positionally, and a real `|-curestatus|` from Heal Bell / Aromatherapy
    # lands `'[from] move: Aromatherapy'` there — caught by the routine gate on live battles when
    # this guard first shipped without the distinction.
    assert _event_status_id("[from] move: Aromatherapy") == 0
    assert _event_status_id("[silent]") == 0


def test_the_status_seat_table_covers_the_producer_vocabulary():
    """The clamp-sweep contract at the sixth site the model-wide sweep did not reach.

    `EventSeats.status_emb` was `Embedding(8)` with a literal `clamp(max=7)` — a WIDTH and a
    CLAMP both spelled as bare numbers, with the producer's vocabulary living in a different
    package. Gen-3's status set is closed, so the spare rows are a genuine no-op safety net;
    what was missing is the relationship that makes a grown vocabulary FAIL instead of clamping
    a new id onto `tox`."""
    from agents.model.team_transformer import EventSeats
    from agents.observation.constants import EVENT_STATUS_IDS, N_EVENT_STATUS

    assert N_EVENT_STATUS == max(EVENT_STATUS_IDS.values()) + 1
    assert N_EVENT_STATUS <= EventSeats._STATUS_ROWS
    seats = EventSeats({"event_window_n": 4, "species_embedding_dim": 8,
                        "move_embedding_dim": 8})
    assert seats.status_emb.num_embeddings == EventSeats._STATUS_ROWS
    # every live id addresses its OWN row — none of them is the clamp target
    assert max(EVENT_STATUS_IDS.values()) < seats.status_emb.num_embeddings - 1


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


# ---------------------------------------------------------------------------
# gen3_event_col_names_v1 — the 22-column contract as a NAMED declaration
# ---------------------------------------------------------------------------
# `EventCol` is imported by BOTH `state_encoder` (the producer) and
# `team_transformer.EventSeats` (the consumer), plus the feature-coverage probes and every
# oracle. Before it existed the contract was a comment plus ~30 bare integers spread over five
# files, which is the shape the positional-binding sweep convicted five times: two ends bound to
# the same subject by POSITION, with nothing relating them. These tests are what make the
# declaration load-bearing rather than decorative.


def test_event_column_map_tiles_the_token_exactly():
    """The members must TILE `range(EVENT_TOKEN_DIM)` — no gaps, no overlaps, no overflow.

    A gap is a column nobody writes and the consumer still slices (reading a constant 0 as a
    feature); an overlap is two facts sharing an address, which reads as whichever wrote last.
    Neither raises anywhere on its own — the token is a flat float row."""
    values = [int(c) for c in C]
    assert len(set(values)) == len(values), f"EventCol has DUPLICATE column indices: {values}"
    assert sorted(values) == list(range(EVENT_TOKEN_DIM)), (
        f"EventCol must cover 0..{EVENT_TOKEN_DIM - 1} exactly, got {sorted(values)} — "
        "widening the token means adding a member AND bumping EVENT_TOKEN_DIM")


def test_the_two_one_hot_groups_are_contiguous_and_in_order():
    """Both groups are written by INDEXING, so their order and contiguity are the contract.

    `state_encoder` writes the effectiveness one-hot as `EFF_NEUTRAL + eff` where `eff` is the
    TurnDelta effectiveness code (0 neutral / 1 super / 2 resist / 3 immune), and `EventSeats`
    takes the raw scalars as one `MAGNITUDE..WE_FIRST` slice that spans both groups. Reorder a
    member and the producer writes 'resisted' where the consumer reads 'super effective' — a
    pure relabelling with no shape change, so nothing else would notice."""
    for group, name in ((EVENT_OUTCOME_GROUP, "EVENT_OUTCOME_GROUP"),
                        (EVENT_EFF_GROUP, "EVENT_EFF_GROUP")):
        cols = [int(c) for c in group]
        assert cols == list(range(cols[0], cols[0] + len(cols))), (
            f"{name} must be CONTIGUOUS and ascending (it is indexed as base+offset): {cols}")
    assert [int(c) for c in EVENT_EFF_GROUP] == [C.EFF_NEUTRAL, C.EFF_SUPER,
                                                 C.EFF_RESIST, C.EFF_IMMUNE], (
        "the eff order is the TurnDelta effectiveness code order — reordering silently "
        "relabels every historical row")
    # The consumer's scalar run spans MAGNITUDE..WE_FIRST inclusive; both one-hots sit inside it,
    # which is why EventSeats can take three slices instead of thirteen columns.
    assert C.MAGNITUDE < C.OUT_HIT and C.EFF_IMMUNE < C.WE_FIRST


def test_producer_and_consumer_import_the_SAME_declaration():
    """Not "agree on the numbers" — are the SAME object.

    Two modules each holding their own copy of the map would pass every value assertion above
    right up until one of them was edited. The point of the declaration is that there is one.
    (Both read it through the `EVENT_COL` plain-int mirror, itself generated from `EventCol` —
    see `test_the_plain_int_mirror_agrees_with_the_enum_member_for_member`.)"""
    from agents.observation.constants import EVENT_COL
    from agents.observation import state_encoder as _producer
    from agents.model import team_transformer as _consumer
    assert _producer.EVENT_COL is EVENT_COL
    assert _consumer.EVENT_COL is EVENT_COL


def test_event_seats_scalar_count_matches_the_column_map():
    """`EventSeats._N_SCALARS` is a WEIGHT SHAPE (it sizes `proj`'s input), and the column map
    is what it is supposed to count. A column that changes routing — id ↔ raw scalar — moves
    that width, and a stale `_N_SCALARS` builds a Linear of the wrong size against a `torch.cat`
    of the right one, which raises far from the cause."""
    from agents.model.team_transformer import EventSeats
    ids = {C.TYPE, C.ACTOR_SPECIES, C.TARGET_SPECIES, C.MOVE, C.STATUS,
           C.CANT, C.FAINT_CAUSE, C.ITEM_TRANSITION}
    scalars = set(C) - ids - {C.VALID}
    assert EventSeats._N_SCALARS == len(scalars), (
        f"EventSeats._N_SCALARS={EventSeats._N_SCALARS} but the column map says {len(scalars)} "
        f"raw scalars ({sorted(int(c) for c in scalars)})")


def test_the_plain_int_mirror_agrees_with_the_enum_member_for_member():
    """`EVENT_COL` is a generated mirror, so this is a tautology TODAY — its value is that it
    stays one. Both live consumers read the MIRROR, not the enum, for reasons that have nothing
    to do with the contract (`state_encoder` because an enum member's `LOAD_ATTR` costs ~36% of
    the obs write loop; `team_transformer` because `torch.fx` renders an enum member as its repr
    `<EventCol.TYPE: 0>` into generated graph code, which is a SyntaxError at compile). A future
    edit that hand-writes an entry into the mirror instead of deriving it fails here."""
    from agents.observation.constants import EVENT_COL
    mirror = {k: v for k, v in vars(EVENT_COL).items() if not k.startswith("_")}
    assert mirror == {c.name: int(c) for c in C}
    assert all(type(v) is int for v in mirror.values()), (
        "the mirror must hold PLAIN ints — an IntEnum member here defeats both of its purposes")
