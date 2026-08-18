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
