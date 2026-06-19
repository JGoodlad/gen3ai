from types import SimpleNamespace

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


def test_states_arrays_captures_win_prob():
    """states_arrays() carries P(win) parallel to `values`; NaN where a decision had no win_prob
    (no head / mixed capture) so the prober distinguishes 'unavailable' from a real P(win)=0.0."""
    rec = BattleRecorder("battle-gen3ou-test", reward_fn_factory=_ZeroReward, gamma=0.9)
    wps = [0.6, None, 0.4]
    for t in range(3):
        st = {"value": 1.0, "obs": np.zeros(4, np.float32), "logits": np.zeros(11, np.float32)}
        if wps[t] is not None:
            st["win_prob"] = wps[t]
        rec.record(_move_battle(t + 1), 6, _probs(), _mask(6), state=st)
    arr = rec.states_arrays()["win_probs"]
    assert arr[0] == pytest.approx(0.6) and arr[2] == pytest.approx(0.4) and np.isnan(arr[1])


def test_states_arrays_captures_value_dist():
    """states_arrays() carries the value-dist head's per-atom distribution [T, bins] parallel to
    `values`; a row with no distribution stays all-NaN, and the key is OMITTED entirely when no
    decision carried one (so the prober's KeyError 'unavailable' path fires on a headless run)."""
    bins = 8
    rec = BattleRecorder("battle-gen3ou-test", reward_fn_factory=_ZeroReward, gamma=0.9)
    dists = [list(np.full(bins, 1.0 / bins)), None, list(np.eye(bins)[3])]  # uniform, absent, one-hot@3
    for t in range(3):
        st = {"value": 1.0, "obs": np.zeros(4, np.float32), "logits": np.zeros(11, np.float32)}
        if dists[t] is not None:
            st["value_dist"] = dists[t]
        rec.record(_move_battle(t + 1), 6, _probs(), _mask(6), state=st)
    out = rec.states_arrays()
    vd = out["value_dist"]
    assert vd.shape == (3, bins)
    assert vd[0] == pytest.approx(1.0 / bins) and np.isnan(vd[1]).all() and vd[2][3] == pytest.approx(1.0)

    # Headless run: no state carries a distribution → the key is absent entirely.
    rec2 = BattleRecorder("battle-gen3ou-test", reward_fn_factory=_ZeroReward, gamma=0.9)
    for t in range(2):
        rec2.record(_move_battle(t + 1), 6, _probs(), _mask(6),
                    state={"value": 1.0, "obs": np.zeros(4, np.float32), "logits": np.zeros(11, np.float32)})
    assert "value_dist" not in rec2.states_arrays()


def test_states_arrays_captures_move_logits_and_spread():
    """states_arrays() carries the captured move-belief posterior [T, n_moves] + the opp-active believed
    spread [T, 5] (the prober's axis-B trajectory inputs), parallel to value_dist: NaN for an uncaptured
    row, key OMITTED entirely when the head was off the whole run."""
    rec = BattleRecorder("battle-gen3ou-test", reward_fn_factory=_ZeroReward, gamma=0.9)
    mls = [[0.9, 0.1, 0.5, 0.2], None, [0.3, 0.7, 0.4, 0.6]]
    sbs = [[299.0, 200.0, 150.0, 180.0, 250.0], None, [310.0, 205.0, 150.0, 180.0, 255.0]]
    for t in range(3):
        st = {"value": 1.0, "obs": np.zeros(4, np.float32), "logits": np.zeros(11, np.float32)}
        if mls[t] is not None:
            st["move_logits"] = mls[t]
        if sbs[t] is not None:
            st["spread_belief"] = sbs[t]
        rec.record(_move_battle(t + 1), 6, _probs(), _mask(6), state=st)
    out = rec.states_arrays()
    assert out["move_logits"].shape == (3, 4) and out["spread_belief"].shape == (3, 5)
    assert out["move_logits"][0] == pytest.approx([0.9, 0.1, 0.5, 0.2])
    assert np.isnan(out["move_logits"][1]).all() and np.isnan(out["spread_belief"][1]).all()
    assert out["spread_belief"][2] == pytest.approx([310.0, 205.0, 150.0, 180.0, 255.0])

    # Heads-off run: neither key is captured → both omitted entirely.
    rec2 = BattleRecorder("battle-gen3ou-test", reward_fn_factory=_ZeroReward, gamma=0.9)
    for t in range(2):
        rec2.record(_move_battle(t + 1), 6, _probs(), _mask(6),
                    state={"value": 1.0, "obs": np.zeros(4, np.float32), "logits": np.zeros(11, np.float32)})
    arr2 = rec2.states_arrays()
    assert "move_logits" not in arr2 and "spread_belief" not in arr2


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


class _ReplayBattle:
    """Battle stub exposing only `save_replay` — the raw poke-env seam
    write_battle_record calls to emit the browser-watchable HTML."""
    def __init__(self, raises: bool = False):
        self._raises = raises
        self.saved_to = None

    def save_replay(self, path):
        if self._raises:
            raise RuntimeError("no replay log")
        self.saved_to = str(path)
        with open(path, "w") as f:
            f.write("<!DOCTYPE html><title>replay</title>")


def test_write_battle_record_writes_json_npz_and_replay(tmp_path):
    import os
    import json
    from agents.training.battle_recorder import write_battle_record

    summary = {"meta": {"result": "WIN"},
               "actions": {"thunderbolt": {"prob": "90.0%", "valid": True}}}
    states = {"obs": np.zeros((2, 4), dtype=np.float32),
              "logits": np.zeros((2, 11), dtype=np.float32)}
    prefix = str(tmp_path / "sub" / "win_001")  # nested dir must be auto-created
    battle = _ReplayBattle()

    write_battle_record(prefix, _StubRecorder(summary, states), battle, step=42)

    with open(prefix + "_summary.json") as f:
        text = f.read()
    data = json.loads(text)
    assert data["step"] == 42
    # The leaf {"prob":..,"valid":..} object is collapsed onto one line.
    assert '{"prob": "90.0%", "valid": true}' in text

    npz = np.load(prefix + "_states.npz")
    assert npz["obs"].shape == (2, 4)

    # The human-watchable replay is co-located with the forensic dump.
    assert battle.saved_to == prefix + "_replay.html"
    assert os.path.exists(prefix + "_replay.html")


def test_write_battle_record_skips_npz_when_no_states(tmp_path):
    import os
    from agents.training.battle_recorder import write_battle_record

    prefix = str(tmp_path / "loss_001")
    write_battle_record(prefix, _StubRecorder({"meta": {}}, {}), _ReplayBattle(), step=1)

    assert os.path.exists(prefix + "_summary.json")
    assert not os.path.exists(prefix + "_states.npz")
    # Replay HTML is written even on the no-states (cheap) path.
    assert os.path.exists(prefix + "_replay.html")


def test_write_battle_record_replay_failure_keeps_forensic_trace(tmp_path):
    """A save_replay error must never cost us the forensic dump already written."""
    import os
    from agents.training.battle_recorder import write_battle_record

    prefix = str(tmp_path / "loss_002")
    states = {"obs": np.zeros((1, 4), dtype=np.float32),
              "logits": np.zeros((1, 11), dtype=np.float32)}

    # Must not raise even though save_replay does.
    write_battle_record(prefix, _StubRecorder({"meta": {}}, states),
                        _ReplayBattle(raises=True), step=7)

    assert os.path.exists(prefix + "_summary.json")
    assert os.path.exists(prefix + "_states.npz")
    assert not os.path.exists(prefix + "_replay.html")


# ---------------------------------------------------------------------------
# Hidden-opponent belief field (decoded by RLPlayer, passed through via `state`)
# ---------------------------------------------------------------------------

def test_record_includes_belief_when_state_carries_it():
    """A `state` carrying the decoded hidden-opp belief surfaces as the entry's `belief` field,
    placed right after `opp` ('the board, then what we think is still hidden')."""
    zapdos = _FakeMon("zapdos", 1.0)
    opp = _FakeMon("tyranitar", 1.0)
    b1 = _battle([zapdos], [opp], "zapdos", "tyranitar", turn=1, move_ids=["thunderbolt"])
    belief = [{"slot": 3, "top": [{"species": "skarmory", "prob": "40.0%"},
                                  {"species": "metagross", "prob": "20.0%"}]}]

    rec = _rec()
    rec.record(b1, 6, _probs(), _mask(6), state={"belief": belief})
    entry = rec._pending_entry

    assert entry["belief"] == belief
    keys = list(entry.keys())
    assert keys.index("belief") == keys.index("opp") + 1
    assert keys.index("belief") < keys.index("actions")


def test_action_labels_use_typed_own_hidden_power():
    """OUR Hidden Power is recorded with its TYPED id ("hiddenpowerice"), not the wire-bare
    "hiddenpower" — we always know our own HP type and these labels are human/prober-facing. The
    wire-truth ids the mask/mapper use stay bare; only the recorded LABEL is typed."""
    sceptile = _FakeMon("sceptile", 1.0)
    # live moveset keyed bare but each Move carries the typed id (poke-env's raw_id patch)
    sceptile.moves = {"hiddenpower": SimpleNamespace(id="hiddenpowerice", current_pp=24, max_pp=24)}
    opp = _FakeMon("gengar", 1.0)
    b = _battle([sceptile], [opp], "sceptile", "gengar", turn=1, move_ids=["hiddenpower"])

    rec = _rec()
    rec.record(b, 6, _probs(), _mask(6), state=None)   # action 6 == move slot 0 == the HP
    entry = rec._pending_entry

    assert entry["chosen"] == "hiddenpowerice"
    assert "hiddenpowerice" in entry["actions"]
    assert "hiddenpower" not in entry["actions"]


def test_record_omits_belief_when_absent_or_empty():
    """No `belief` key when the belief is off (no state / None) or no slot is hidden ([])."""
    zapdos = _FakeMon("zapdos", 1.0)
    opp = _FakeMon("tyranitar", 1.0)
    b1 = _battle([zapdos], [opp], "zapdos", "tyranitar", turn=1, move_ids=["thunderbolt"])

    for state in (None, {"belief": None}, {"belief": []}):
        rec = _rec()
        rec.record(b1, 6, _probs(), _mask(6), state=state)
        assert "belief" not in rec._pending_entry


def test_write_battle_record_collapses_belief_leaves(tmp_path):
    """Each species-guess leaf is collapsed onto one line, like the action/prob leaves."""
    import json
    from agents.training.battle_recorder import write_battle_record

    summary = {"meta": {"result": "LOSS"},
               "invocations": [{"belief": [
                   {"slot": 3, "top": [{"species": "skarmory", "prob": "40.0%"}]}]}]}
    prefix = str(tmp_path / "loss_003")
    write_battle_record(prefix, _StubRecorder(summary, {}), _ReplayBattle(), step=5)

    with open(prefix + "_summary.json") as f:
        text = f.read()
    data = json.loads(text)
    assert data["invocations"][0]["belief"][0]["top"][0]["species"] == "skarmory"
    assert '{"species": "skarmory", "prob": "40.0%"}' in text
