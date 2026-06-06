import numpy as np
import pytest

from agents.training.battle_recorder import BattleRecorder
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents.battle.live_view import LiveView
from agents.battle.strict_view import StrictBattleView
from agents.battle.battle_event import BattleEvent, EventKind, OURS, OPP


def _ev(kind, side, actor, turn=2, **value):
    """Build a BattleEvent for a recorder test's injected event window."""
    return BattleEvent(seq=0, turn=turn, kind=kind, side=side,
                       actor_species=actor, value=value)


def _with_events(b, events):
    """Override a fake battle's empty event window with a crafted one (the recorder folds
    the TurnDelta from ``events_since`` — in production this is the real Gen3Battle log)."""
    b.events_since = lambda cursor: list(events)
    return b


class _FakeMon:
    def __init__(self, species, hp_fraction=1.0, fainted=False):
        self.species = species
        self.current_hp_fraction = hp_fraction
        self.fainted = fainted
        self.item = None
        self.last_move = None
        self.last_cant_reason = None
        self.moves = {}
        self.status = None
        self.effects = {}
        # Fields LiveView.from_battle / LivePokemon.from_pokemon read, so the
        # recorder's strict-view path (battle.strict_view().live) works against a
        # real LiveView built from these fakes.
        self.revealed = True
        self.ability = None
        self.types = ()
        self.boosts = {}
        self.base_stats = {}
        self.ivs = None
        self.evs = None
        self.nature = None
        self.consumed_item = None
        self.status_counter = 0


class _Ns:
    """Minimal namespace battle object — only the attributes BattleContext.from_battle
    and LiveView.from_battle need."""
    pass


def _battle(our_mons, opp_mons, our_active, opp_active,
            turn=1, force_switch=False, won=False, lost=False, move_ids=None):
    move_ids = move_ids or []
    b = _Ns()
    b.team = {m.species: m for m in our_mons}
    b.opponent_team = {m.species: m for m in opp_mons}
    b.active_pokemon = b.team[our_active]
    b.opponent_active_pokemon = b.opponent_team[opp_active]
    b.turn = turn
    b.force_switch = force_switch
    b.won = won
    b.lost = lost
    b.finished = bool(won or lost)
    b.battle_tag = "battle-gen3ou-test"
    b.last_request = {"active": [{"moves": [{"id": m, "disabled": False} for m in move_ids]}]}
    # LegalActions.from_battle reads these (the recorder snapshots legality per record()).
    b.available_switches = []
    b.available_moves = []
    b.trapped = False
    b.maybe_trapped = False
    b.wait = False
    b.our_last_effectiveness = None
    b.opp_last_effectiveness = None
    b.our_last_damaging_move = None
    b.opp_last_damaging_move = None
    b.we_moved_first = None
    b.our_move_crit = False
    b.opp_move_crit = False
    b.our_move_missed = False
    b.opp_move_missed = False
    b.our_move_failed = False
    b.opp_move_failed = False
    # Fields LiveView.from_battle reads, plus the strict-view entry points the
    # recorder now goes through (battle.strict_view().live / .legal / .won / …).
    b._player_role = "p1"
    b.side_conditions = {}
    b.opponent_side_conditions = {}
    # Event-log surface the reward tracker folds the TurnDelta from (no real log in this
    # unit fake → an empty window, which folds the current-board snapshot for HP).
    b.event_cursor = 0
    b.events_since = lambda cursor: []
    b.live_view = lambda: LiveView.from_battle(b)
    b.strict_view = lambda: StrictBattleView(b)
    return b


class _ZeroReward:
    def reset(self): pass
    def record_action(self, ctx, action): pass
    def process_turn_reward(self, battle, delta): return 0.0
    def report_episode(self, battle): pass


class _SeqReward:
    """Returns a fixed reward sequence — one per complete_pending/finalize — so a recorder
    test can assert the TD-residual δ = r + γV(s') − V(s) against known numbers."""
    def __init__(self, seq):
        self._seq = list(seq)
        self._i = 0
    def reset(self): pass
    def record_action(self, ctx, action): pass
    def process_turn_reward(self, battle, delta):
        r = self._seq[self._i]
        self._i += 1
        return r
    def report_episode(self, battle): pass


def _rec():
    return BattleRecorder("battle-gen3ou-test", reward_fn_factory=_ZeroReward)


def _probs():
    return np.ones(11, dtype=np.float32) / 11


def _mask(*valid):
    m = np.zeros(11, dtype=np.int8)
    for a in valid:
        m[a] = 1
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_hp_delta_on_move():
    """Active mon uses a move and takes damage — hp_delta reflects the active mon."""
    zapdos = _FakeMon("zapdos", 1.0)
    opp = _FakeMon("tyranitar", 1.0)
    b1 = _battle([zapdos], [opp], "zapdos", "tyranitar", turn=1, move_ids=["thunderbolt"])

    zapdos2 = _FakeMon("zapdos", 0.70)  # took 30% damage
    opp2 = _FakeMon("tyranitar", 0.80)
    b2 = _battle([zapdos2], [opp2], "zapdos", "tyranitar", turn=2, move_ids=["thunderbolt"])

    rec = _rec()
    rec.record(b1, 6, _probs(), _mask(6))  # thunderbolt = action 6
    rec.record(b2, 6, _probs(), _mask(6))  # triggers _complete_pending

    outcome = rec._invocations[0]["outcome"]
    assert outcome["our"]["hp_delta"] == "-30%"


def test_hp_delta_on_switch():
    """
    We switch from Zapdos (100%) to Tyranitar. The opponent's Thunderbolt hits
    Tyranitar on switch-in, dropping it to 64%.

    Before the fix, hp_delta reported "+0%" (Zapdos's delta, not Tyranitar's).
    After the fix it must report "-36%".
    """
    zapdos = _FakeMon("zapdos", 1.0)
    ttar = _FakeMon("tyranitar", 1.0)
    opp = _FakeMon("opp_zapdos", 1.0)
    b1 = _battle([zapdos, ttar], [opp], "zapdos", "opp_zapdos", turn=1, move_ids=["thunderbolt"])

    zapdos2 = _FakeMon("zapdos", 1.0)   # switched out safely, no damage
    ttar2 = _FakeMon("tyranitar", 0.64)  # took Thunderbolt on switch-in
    opp2 = _FakeMon("opp_zapdos", 1.0)
    b2 = _battle([zapdos2, ttar2], [opp2], "tyranitar", "opp_zapdos", turn=2, move_ids=["rockslide"])
    _with_events(b2, [
        _ev(EventKind.SWITCH, OURS, "tyranitar", prev_active="zapdos"),
        _ev(EventKind.MOVE, OPP, "opp_zapdos", move_id="thunderbolt", target_status=None),
        _ev(EventKind.DAMAGE, OURS, "tyranitar", amount=-0.36, hp_before=1.0, hp_after=0.64),
    ])

    rec = _rec()
    rec.record(b1, 1, _probs(), _mask(1))  # action 1 = switch to tyranitar (team slot 1)
    rec.record(b2, 6, _probs(), _mask(6))  # triggers _complete_pending

    outcome = rec._invocations[0]["outcome"]
    assert outcome["our"]["action"] == "switched_to:tyranitar"
    assert outcome["our"]["hp_delta"] == "-36%"


def test_finalize_hp_delta_on_switch():
    """
    Same switch scenario but as the last action of the battle — goes through
    finalize(), not _complete_pending(). hp_delta must still reflect Tyranitar's damage.
    """
    zapdos = _FakeMon("zapdos", 1.0)
    ttar = _FakeMon("tyranitar", 1.0)
    opp = _FakeMon("opp_zapdos", 1.0)
    b1 = _battle([zapdos, ttar], [opp], "zapdos", "opp_zapdos", turn=1, move_ids=["thunderbolt"])

    # Battle ends with Tyranitar active at 64%, opponent fainted
    zapdos_end = _FakeMon("zapdos", 1.0)
    ttar_end = _FakeMon("tyranitar", 0.64)
    opp_end = _FakeMon("opp_zapdos", 0.0, fainted=True)
    b_end = _battle([zapdos_end, ttar_end], [opp_end], "tyranitar", "opp_zapdos",
                    turn=2, won=True)
    _with_events(b_end, [
        _ev(EventKind.SWITCH, OURS, "tyranitar", prev_active="zapdos"),
        _ev(EventKind.MOVE, OPP, "opp_zapdos", move_id="thunderbolt", target_status=None),
        _ev(EventKind.DAMAGE, OURS, "tyranitar", amount=-0.36, hp_before=1.0, hp_after=0.64),
        _ev(EventKind.FAINT, OPP, "opp_zapdos"),
    ])

    rec = _rec()
    rec.record(b1, 1, _probs(), _mask(1))  # switch to tyranitar
    rec.finalize(b_end)

    outcome = rec._invocations[0]["outcome"]
    assert outcome["our"]["hp_delta"] == "-36%"
    assert "result:win" in outcome["events"]


def test_hp_delta_opp_switch():
    """
    Opponent switches from Blissey (100%) to Snorlax, and we hit Snorlax on switch-in
    for 30% damage. opp.hp_delta must reflect Snorlax's -30%, not Blissey's 0%.
    """
    our = _FakeMon("metagross", 1.0)
    blissey = _FakeMon("blissey", 1.0)
    snorlax = _FakeMon("snorlax", 1.0)
    b1 = _battle([our], [blissey, snorlax], "metagross", "blissey", turn=1, move_ids=["meteormash"])

    our2 = _FakeMon("metagross", 1.0)
    blissey2 = _FakeMon("blissey", 1.0)   # switched out, took no damage
    snorlax2 = _FakeMon("snorlax", 0.70)  # switched in, took Meteor Mash
    b2 = _battle([our2], [blissey2, snorlax2], "metagross", "snorlax", turn=2, move_ids=["meteormash"])
    _with_events(b2, [
        _ev(EventKind.SWITCH, OPP, "snorlax", prev_active="blissey"),
        _ev(EventKind.MOVE, OURS, "metagross", move_id="meteormash", target_status=None),
        _ev(EventKind.DAMAGE, OPP, "snorlax", amount=-0.30, hp_before=1.0, hp_after=0.70),
    ])

    rec = _rec()
    rec.record(b1, 6, _probs(), _mask(6))  # we use a move
    rec.record(b2, 6, _probs(), _mask(6))  # triggers _complete_pending

    outcome = rec._invocations[0]["outcome"]
    assert outcome["opp"]["hp_delta"] == "-30%"


def test_forced_switch_opp_action_is_none():
    """
    On a forced-switch turn the opponent doesn't act.
    outcome['opp']['action'] must be 'none', not the stale last_move from
    the turn our mon fainted.
    """
    class _FakeMove:
        def __init__(self, move_id): self.id = move_id

    # Turn 1: zapdos uses thunderbolt, tyranitar uses rockslide and kills zapdos.
    zapdos = _FakeMon("zapdos", 0.30)
    opp = _FakeMon("tyranitar", 0.80)
    b1 = _battle([zapdos], [opp], "zapdos", "tyranitar", turn=1, move_ids=["thunderbolt"])

    # Turn 2: zapdos fainted → forced switch.
    # Tyranitar's last_move is still rockslide (stale from turn 1).
    zapdos_fainted = _FakeMon("zapdos", 0.0, fainted=True)
    swampert = _FakeMon("swampert", 1.0)
    opp2 = _FakeMon("tyranitar", 0.80)
    opp2.last_move = _FakeMove("rockslide")
    b2 = _battle([zapdos_fainted, swampert], [opp2], "zapdos", "tyranitar",
                 turn=2, force_switch=True, move_ids=[])

    # Turn 3: swampert is now active; completing b2's invocation triggers the bug.
    swampert3 = _FakeMon("swampert", 1.0)
    opp3 = _FakeMon("tyranitar", 0.80)
    b3 = _battle([zapdos_fainted, swampert3], [opp3], "swampert", "tyranitar",
                 turn=3, move_ids=["surf"])

    rec = _rec()
    rec.record(b1, 6, _probs(), _mask(6))   # turn 1: thunderbolt
    rec.record(b2, 1, _probs(), _mask(1))   # forced switch: pick swampert (slot 1)
    rec.record(b3, 6, _probs(), _mask(6))   # turn 3: completes the forced-switch invocation

    forced_outcome = rec._invocations[1]["outcome"]
    assert forced_outcome["opp"]["action"] == "none"


# ── TD-residual δ accumulation (#4) ───────────────────────────────────────────

def _move_battle(turn):
    return _battle([_FakeMon("zapdos", 1.0)], [_FakeMon("tyranitar", 1.0)],
                   "zapdos", "tyranitar", turn=turn, move_ids=["thunderbolt"])


def test_td_residuals_match_prober_formula():
    """δ(t) = r(t) + γ·V(s_{t+1}) − V(s_t), closed at the NEXT record(); last decision yields no δ."""
    gamma = 0.9
    rewards = [2.0, -5.0]          # r0 (closed at 2nd record), r1 (closed at 3rd)
    values = [1.0, 0.5, -3.0]      # V(s0), V(s1), V(s2)
    rec = BattleRecorder("battle-gen3ou-test", reward_fn_factory=lambda: _SeqReward(rewards),
                         gamma=gamma)
    for t in range(3):
        rec.record(_move_battle(t + 1), 6, _probs(), _mask(6),
                   state={"value": values[t], "obs": np.zeros(4, np.float32),
                          "logits": np.zeros(11, np.float32)})
    expected = [
        rewards[0] + gamma * values[1] - values[0],
        rewards[1] + gamma * values[2] - values[1],
    ]
    assert rec.td_residuals() == pytest.approx(expected)


def test_td_residuals_empty_without_captured_values():
    """No state['value'] (the cheap fast path) → no residuals, never a spurious 0."""
    rec = BattleRecorder("battle-gen3ou-test", reward_fn_factory=_ZeroReward, gamma=0.9)
    for t in range(3):
        rec.record(_move_battle(t + 1), 6, _probs(), _mask(6))
    assert rec.td_residuals() == []


# ── write_battle_record (shared forensic writer) ──────────────────────────────

class _StubRecorder:
    """Stand-in exposing only the surface write_battle_record touches."""
    def __init__(self, summary, states):
        self._summary = summary
        self._states = states
        self.finalized = False

    def finalize(self, battle):
        self.finalized = True

    def to_summary(self, battle, step):
        return dict(self._summary, step=step)

    def states_arrays(self):
        return self._states


def test_write_battle_record_writes_json_and_npz(tmp_path):
    import json
    from agents.training.battle_recorder import write_battle_record

    summary = {"meta": {"result": "WIN"},
               "actions": {"thunderbolt": {"prob": "90.0%", "valid": True}}}
    states = {"obs": np.zeros((2, 4), dtype=np.float32),
              "logits": np.zeros((2, 11), dtype=np.float32)}
    prefix = str(tmp_path / "sub" / "win_001")  # nested dir must be auto-created

    write_battle_record(prefix, _StubRecorder(summary, states), object(), step=42)

    with open(prefix + "_summary.json") as f:
        text = f.read()
    data = json.loads(text)
    assert data["step"] == 42
    # The leaf {"prob":..,"valid":..} object is collapsed onto one line.
    assert '{"prob": "90.0%", "valid": true}' in text

    npz = np.load(prefix + "_states.npz")
    assert npz["obs"].shape == (2, 4)


def test_write_battle_record_skips_npz_when_no_states(tmp_path):
    import os
    from agents.training.battle_recorder import write_battle_record

    prefix = str(tmp_path / "loss_001")
    write_battle_record(prefix, _StubRecorder({"meta": {}}, {}), object(), step=1)

    assert os.path.exists(prefix + "_summary.json")
    assert not os.path.exists(prefix + "_states.npz")
