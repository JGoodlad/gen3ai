"""Pure unit tests for the probe engine — no torch, no checkpoint.

A FakeProbeModel stands in for the torch boundary so the whole analysis is
exercised deterministically. A separate regression test pins the resolved obs
offsets against the live encoder layout, so a silent obs-layout shift fails loud.
"""

import warnings

import numpy as np

from agents.action.constants import MOVE_START
from main.prober.engine import analyze_invocation
from main.prober.model import ObsOffsets

# Small synthetic layout so the obs vector stays tiny.
_OFF = ObsOffsets(
    mm_off=10, om_off=20, tm_off=164, active_block_dim=5,
    turn_history_offset=200, turn_history_dim=10,
)
_OBS_LEN = 256


class FakeProbeModel:
    """Deterministic stand-in. Move probs scale with the move's matchup slot, so
    the intervention sweep produces a monotonic, checkable response."""

    def __init__(self, offsets: ObsOffsets):
        self.offsets = offsets
        self.calls: list = []

    def action_dist(self, obs, mask):
        self.calls.append(np.asarray(obs).copy())
        w = np.ones(len(mask), dtype=np.float64)
        for s in range(4):  # the 4 move slots, weighted by their multiplier
            w[MOVE_START + s] = 1.0 + float(obs[self.offsets.mm_off + s]) * 10.0
        w = w * mask.astype(np.float64)
        probs = w / w.sum()
        logits = np.arange(len(mask), dtype=np.float64)
        return probs, logits

    def logit_grad(self, obs, mask, action_idx):
        return np.arange(len(obs), dtype=np.float64)  # |grad| = [0,1,2,...]

    def value(self, obs, mask):
        return 3.14  # deterministic critic re-run

    def value_grad(self, obs, mask):
        return np.ones(len(obs), dtype=np.float64)  # uniform critic |grad| → mean_abs == 1 per block

    def describe_global(self, obs):
        return {"weather": "RAIN", "our_spikes": 1, "opp_spikes": 0, "turn": 8.0}

    def describe_team(self, obs):
        # Simulate the per-mon obs decode (item + moves, both sides; display-name keys, lenient).
        return {"Zapdos": {"item": "leftovers", "moves": ("thunderbolt", "hiddenpower")},
                "Steelix": {"item": "choiceband", "moves": ("earthquake",)}}

    def describe_turn_outcome(self, obs):
        # Simulate the next-turn TurnDelta decode: opp crit us, our move was fine.
        return {"our_crit": False, "opp_crit": True, "our_cant": None, "opp_cant": None}


def _summary(chosen="thunderbolt", events=None):
    actions = {
        "switch:Dragonite": {"prob": "3.2%", "valid": True},
        "switch:Skarmory": {"prob": "1.5%", "valid": True},
        "switch:slot2": {"prob": "0.0%", "valid": False},
        "switch:slot3": {"prob": "0.0%", "valid": False},
        "switch:slot4": {"prob": "0.0%", "valid": False},
        "switch:slot5": {"prob": "0.0%", "valid": False},
        "thunderbolt": {"prob": "92.1%", "valid": True},   # idx 6 = MOVE_START
        "earthquake": {"prob": "2.8%", "valid": True},     # idx 7
        "move2": {"prob": "0.0%", "valid": False},          # idx 8
        "move3": {"prob": "0.0%", "valid": False},          # idx 9
        "struggle": {"prob": "0.0%", "valid": False},       # idx 10
    }
    return {
        "meta": {"step": 7000000, "battle_id": "b1", "result": "WIN",
                 "turns": 12, "invocations": 1},
        "invocations": [{
            "i": 1, "turn": 8, "phase": "move", "chosen": chosen,
            "our": {"species": "Zapdos"}, "opp": {"species": "Steelix"},
            "actions": actions,
            "outcome": {"our": {"action": chosen, "hp_delta": "-10%"},
                        "reward": {"total": -1.4, "base": "hp_ours=-1.2"},
                        "events": events or []},
        }],
    }


def _npz(has_state=1, value=1.5, n=1):
    obs = np.zeros((n, _OBS_LEN), dtype=np.float32)
    # stored /4-normalised → *4 gives [2.0, 1.0, 0.0, 0.5]
    obs[0, _OFF.mm_off:_OFF.mm_off + 4] = [0.5, 0.25, 0.0, 0.125]
    return {"obs": obs, "has_state": np.full(n, has_state, dtype=np.int8),
            "values": np.full(n, value, dtype=np.float32)}


def test_meta_and_faithfulness():
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(), _npz(), 0, summary_path="s.json", npz_path="s.npz")
    assert a.meta.step == 7000000 and a.meta.result == "WIN" and a.meta.n_invocations == 1
    assert a.turn == 8 and a.our_species == "Zapdos" and a.opp_species == "Steelix"
    assert len(a.actions) == 11
    chosen_row = next(r for r in a.actions if r.label == "thunderbolt")
    assert chosen_row.is_chosen and chosen_row.valid
    assert abs(chosen_row.recorded - 0.921) < 1e-9
    assert not next(r for r in a.actions if r.label == "switch:slot2").valid


def test_damage_op_view_attached_when_model_exposes_it():
    from dataclasses import asdict
    model = FakeProbeModel(_OFF)
    # the default fake has no damage_op_view → the field stays None (back-compat, op-off run).
    assert analyze_invocation(model, _summary(), _npz(), 0).damage_op is None
    # a model that exposes the op view → the engine attaches it (and it stays JSON-serializable).
    chan = {"low": 0.1, "high": 0.12, "crit": 0.24, "pko": 0.0, "acc": 1.0}
    view = {"incoming": [{"phys": chan, "spec": chan, "p_outspeed": 0.5, "provenance": 1.0}] * 6,
            "choice_band": {"phys_high_cb": [0.0] * 6, "phys_pko_cb": [0.0] * 6, "p_cb": 0.0},
            "outgoing": {"moves": [{"low": 0.0, "high": 0.0, "crit": 0.0, "pko": 0.0}] * 4,
                         "p_outspeed": 0.5},
            # the discrete per-move incoming MATRIX (K=2 here) — opp's likely moves with exact names +
            # per-OUR-mon [low, high, crit, pko, type_mult, status_lands] (the safe-switch read).
            "incoming_matrix": {
                "moves": [{"latent": [0.0] * 4, "belief": 0.6, "accuracy": 1.0, "is_phys": 0.0,
                           "move": "icebeam", "effect": [0.0] * 6, "secondary": [0.0] * 10},
                          {"latent": [0.0] * 4, "belief": 0.4, "accuracy": 1.0, "is_phys": 0.0,
                           "move": "thunderwave", "effect": [0.0] * 6, "secondary": [0.0] * 10}],
                "per_defender": [[{"low": 0.6, "high": 0.8, "crit": 1.4, "pko": 0.9, "type_mult": 2.0,
                                   "status_lands": 0.0},
                                  {"low": 0.0, "high": 0.0, "crit": 0.0, "pko": 0.0, "type_mult": 1.0,
                                   "status_lands": 1.0}]] * 6}}
    model.damage_op_view = lambda obs, mask: view
    a = analyze_invocation(model, _summary(), _npz(), 0)
    assert a.damage_op is not None and len(a.damage_op["incoming"]) == 6        # incoming = our 6 team slots
    assert set(a.damage_op["outgoing"]["moves"][0]) == {"low", "high", "crit", "pko"}
    # the discrete per-move matrix rides through (exact names + 6 × K per-defender pivot reads).
    itk = a.damage_op["incoming_matrix"]
    assert [m["move"] for m in itk["moves"]] == ["icebeam", "thunderwave"]
    assert len(itk["per_defender"]) == 6 and len(itk["per_defender"][0]) == 2
    assert set(itk["per_defender"][0][0]) == {"low", "high", "crit", "pko", "type_mult", "status_lands"}
    asdict(a)   # rides the `analyze` CLI JSON output


def _move_belief_raw(*, ib, tb, tox, surf, nmoves):
    """A synthetic ProbeModel.move_belief output: opp slot 0 = revealed blissey (icebeam shown), and
    the model believes thunderbolt 0.80 / toxic 0.30 unseen (surf 0.04 below the floor)."""
    import numpy as np
    probs = np.zeros((6, nmoves), dtype=np.float64)
    probs[0, ib] = 0.95     # already revealed → must be filtered out of the belief
    probs[0, tb] = 0.80
    probs[0, tox] = 0.30
    probs[0, surf] = 0.04   # below the prob_floor → excluded
    empty = {"species": "", "revealed_moves": (), "known": False, "active": False}
    return {"opp_probs": probs,
            "opp_slots": [{"species": "blissey", "revealed_moves": ("icebeam",), "known": True, "active": True}]
                         + [dict(empty) for _ in range(5)],
            "our_slots": [{"species": "magneton", "active": True}, {"species": "skarmory", "active": False}]
                         + [{"species": "", "active": False} for _ in range(4)]}


def test_move_belief_view_filters_revealed_and_ranks_unseen():
    from agents import gen3_data
    from main.prober.engine import move_belief_view
    g = gen3_data.moves
    nmoves = max(g.get(m).num for m in g.raw()) + 1
    raw = _move_belief_raw(ib=g.get("icebeam").num, tb=g.get("thunderbolt").num,
                           tox=g.get("toxic").num, surf=g.get("surf").num, nmoves=nmoves)
    mb = move_belief_view(raw, top_k=4, prob_floor=0.10)
    assert mb is not None and len(mb.opp) == 1           # only the REVEALED opp slot is decoded
    ob = mb.opp[0]
    assert ob.species == "blissey"
    names = [n for n, _ in ob.believed]
    assert "icebeam" not in names                        # revealed move filtered out of the belief
    assert names[0] == "thunderbolt"                      # highest believed-UNSEEN ranked first
    assert "toxic" in names and "surf" not in names       # surf below the 0.10 floor
    # revealed moves carry their (pinned) belief — the icebeam we PUT at 0.95 comes back with its prob
    assert abs(dict(ob.revealed).get("icebeam", 0.0) - 0.95) < 1e-6
    assert mb.our_labels[0] == (0, "magneton", True)      # TEAM-SLOT order + active flag (op-incoming labels)
    assert mb.our_labels[1] == (1, "skarmory", False)


def test_move_belief_view_none_when_absent_or_empty():
    import numpy as np
    from main.prober.engine import move_belief_view
    assert move_belief_view(None) is None
    # no known opp slot AND no our labels → nothing to show
    assert move_belief_view({"opp_probs": np.zeros((6, 4)), "opp_slots": [], "our_slots": []}) is None


def test_move_belief_view_hidden_power_normalizes():
    """A revealed hiddenpower(grass) filters the believed bare hiddenpower (the type-collapsed HP num)."""
    import numpy as np
    from agents import gen3_data
    from main.prober.engine import move_belief_view, _move_maps
    g = gen3_data.moves
    nmoves = max(g.get(m).num for m in g.raw()) + 1
    hp_num = next(int(g.get(m).num) for m in g.raw() if m.startswith("hiddenpower"))
    assert _move_maps()[hp_num] == "hiddenpower"          # type-collapsed to the bare canonical name
    probs = np.zeros((6, nmoves), dtype=np.float64)
    probs[0, hp_num] = 0.9
    raw = {"opp_probs": probs,
           "opp_slots": [{"species": "zapdos", "revealed_moves": ("hiddenpower(grass)",),
                          "known": True, "active": True}]
                        + [{"species": "", "revealed_moves": (), "known": False, "active": False}
                           for _ in range(5)],
           "our_slots": []}
    mb = move_belief_view(raw)
    assert mb is not None and "hiddenpower" not in [n for n, _ in mb.opp[0].believed]   # revealed → filtered


def test_move_belief_view_caps_unseen_at_open_move_slots():
    """A mon with k revealed moves has at most 4−k slots left, so the believed-UNSEEN list is capped
    there (the multi-label head over-shows otherwise — 2 known moves shouldn't list 4 guesses)."""
    import numpy as np
    from agents import gen3_data
    from main.prober.engine import move_belief_view
    g = gen3_data.moves
    nmoves = max(g.get(m).num for m in g.raw()) + 1
    cand = ["icebeam", "thunderbolt", "toxic", "surf", "psychic", "calmmind"]   # 6 high-prob candidates
    probs = np.zeros((6, nmoves), dtype=np.float64)
    for mv in cand:
        probs[0, g.get(mv).num] = 0.9
    slot = {"species": "blissey", "known": True, "active": True, "revealed_moves": ("icebeam", "thunderbolt", "toxic")}
    raw = {"opp_probs": probs, "our_slots": [],
           "opp_slots": [slot] + [{"species": "", "revealed_moves": (), "known": False, "active": False}
                                  for _ in range(5)]}
    mb = move_belief_view(raw)
    assert len(mb.opp[0].revealed) == 3 and len(mb.opp[0].believed) == 1   # 3 known → only 1 open slot
    slot["revealed_moves"] = ("icebeam", "thunderbolt", "toxic", "surf")    # fully known → no open slots
    assert len(move_belief_view(raw).opp[0].believed) == 0


def test_opp_voluntary_switch_flag():
    """An opponent VOLUNTARY pivot is detected from the recorded opp action → `opp-switch` flag (so the
    'our move resolved vs a switch-in, not the active we computed against' trap is markable/jumpable). A
    forced post-faint replacement (`_sent_in`) or a plain move is NOT a voluntary pivot."""
    from main.prober.engine import opp_voluntary_switch, summary_flags
    sw = {"chosen": "earthquake", "actions": {"earthquake": {"prob": "53.2%", "valid": True}},
          "outcome": {"opp": {"action": "switched_to:claydol"}, "our": {"action": "earthquake"}}}
    assert opp_voluntary_switch(sw) == "claydol" and "opp-switch" in summary_flags(sw)
    forced = {"chosen": "x", "actions": {}, "outcome": {"opp": {"action": "claydol_sent_in"}}}
    assert opp_voluntary_switch(forced) is None and "opp-switch" not in summary_flags(forced)
    move = {"chosen": "x", "actions": {}, "outcome": {"opp": {"action": "icebeam"}}}
    assert opp_voluntary_switch(move) is None
    assert opp_voluntary_switch({"outcome": {}}) is None      # no opp action → None (no crash)


def test_status_cure_taxonomy_is_data_driven():
    """Only Refresh (self) and Heal Bell / Aromatherapy (team) CLEAR status. Recover heals HP and
    Rest cures by INFLICTING sleep — neither is a cure, and conflating them is exactly the confusion
    the `cure-skipped` flag exists to prevent."""
    from main.prober.engine import has_curable_status, is_status_cure
    assert is_status_cure("refresh") and is_status_cure("healbell") and is_status_cure("aromatherapy")
    assert not is_status_cure("recover") and not is_status_cure("rest") and not is_status_cure("toxic")
    assert not is_status_cure("") and not is_status_cure(None)
    # Recorder bundles status+volatiles into one string; only the real statuses are curable.
    assert has_curable_status("TOX(2)") and has_curable_status("TOX(5)|SUB") and has_curable_status("PAR")
    assert not has_curable_status("TAUNT") and not has_curable_status("SUB|TAUNT")
    assert not has_curable_status("") and not has_curable_status(None)


def test_cure_skipped_flag():
    """Statused + a LEGAL cure available + we picked something else → `cure-skipped`. Taking the cure,
    an ILLEGAL cure (Taunted / no PP), and 'statused with no cure in the moveset' all clear it — and so
    does an un-statused mon whose Refresh would be a no-op."""
    from main.prober.engine import self_cure_options, summary_flags

    def inv(status, chosen, acts):
        return {"chosen": chosen, "our": {"species": "milotic", "status": status},
                "actions": {k: {"prob": "10.0%", "valid": v} for k, v in acts.items()}}

    skipped = inv("TOX(1)", "recover", {"recover": True, "refresh": True, "surf": True})
    assert self_cure_options(skipped) == ("refresh",) and "cure-skipped" in summary_flags(skipped)
    took = inv("TOX(1)", "refresh", {"recover": True, "refresh": True})
    assert "cure-skipped" not in summary_flags(took)
    taunted = inv("TOX(1)|TAUNT", "surf", {"refresh": False, "surf": True})
    assert self_cure_options(taunted) == () and "cure-skipped" not in summary_flags(taunted)
    healthy = inv("", "recover", {"recover": True, "refresh": True})
    assert self_cure_options(healthy) == () and "cure-skipped" not in summary_flags(healthy)
    no_cure = inv("TOX(1)", "surf", {"recover": True, "surf": True})
    assert self_cure_options(no_cure) == () and "cure-skipped" not in summary_flags(no_cure)


def test_recorded_actions_are_action_index_aligned():
    """REGRESSION (move-slot-misalignment class, FIXED): the recorded `actions` dict is ALREADY in
    action-index order (`BattleRecorder._all_action_labels` keys move slot m on `legal.move_ids[m]` at action
    6+m), so `labels[i]` ↔ `probs[i]` directly — the engine must NOT re-sort the move labels. A model that
    exposes a SCRAMBLING `our_active_move_slots` (the per-mon block's moveset order, which differs from the
    request order after a server reorder) must be IGNORED; the old `_reorder_move_labels` used it and
    transposed correct labels (hiddenpower↔thunderbolt), flipping the re-run argmax + faithfulness labels."""

    class _IdxModel(FakeProbeModel):
        """action_dist puts the mass on action INDEX 7 (the 2nd move slot); also exposes a scrambling
        move-slot decode that the engine must NOT use."""
        def action_dist(self, obs, mask):
            p = np.full(len(mask), 0.01, dtype=np.float64)
            p[7] = 0.9                                   # index 7 = the 2nd move slot
            p = p * mask.astype(np.float64)
            return p / p.sum(), np.arange(len(mask), dtype=np.float64)

        def our_active_move_slots(self, obs):           # a SCRAMBLED order (the old bug's input)
            return ("earthquake", "thunderbolt", "move3", "move2")

    # The recorded chosen is the move at index 7 (earthquake) — what the policy actually picked.
    a = analyze_invocation(_IdxModel(_OFF), _summary(chosen="earthquake"), _npz(), 0)
    # The re-run argmax must be the label at index 7 (earthquake), NOT index 6 (thunderbolt) — i.e. the
    # scrambling decode is ignored and the exact-reproducing model AGREES with the recorded choice.
    assert a.rerun_argmax == "earthquake" and a.agrees is True
    # Per-label alignment by index: index-7's label carries the re-run mass, index-6's does not.
    eq = next(r for r in a.actions if r.label == "earthquake")
    tb = next(r for r in a.actions if r.label == "thunderbolt")
    assert eq.rerun > 0.8 and tb.rerun < 0.1
    # The recorded probs stay attached to their own labels (thunderbolt 92.1%, earthquake 2.8%).
    assert abs(tb.recorded - 0.921) < 1e-9 and abs(eq.recorded - 0.028) < 1e-9


def test_matchups_read_correct_dims():
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(), _npz(), 0)
    assert a.matchups.multipliers == (2.0, 1.0, 0.0, 0.5)
    assert a.matchups.move_labels == ("thunderbolt", "earthquake", "move2", "move3")
    # applicable flags which slots' multipliers mean anything: real damaging moves yes,
    # unknown placeholder ids no (is_damaging tolerates them → False).
    assert a.matchups.applicable == (True, True, False, False)


def _summary_moves(move_labels, our_species="Zapdos"):
    """A summary whose 4 move-action labels (request order, idx 6–9) are `move_labels`."""
    actions = {f"switch:slot{i}": {"prob": "0.0%", "valid": False} for i in range(6)}
    for lbl in move_labels:
        actions[lbl] = {"prob": "10.0%", "valid": True}
    actions["struggle"] = {"prob": "0.0%", "valid": False}
    s = _summary()
    s["invocations"][0]["actions"] = actions
    s["invocations"][0]["chosen"] = move_labels[0]
    s["invocations"][0]["our"]["species"] = our_species
    return s


def test_matchups_typed_own_hidden_power_shown_with_type():
    """A NEW trace records OUR HP with its typed id ("hiddenpowerice"); the matchups label renders
    the readable typed form, and the multiplier (already typed in the obs) is meaningful."""
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary_moves(["hiddenpowerice", "earthquake", "move2", "move3"]),
                           _npz(), 0)
    assert a.matchups.move_labels[0] == "hiddenpower(ice)"
    assert a.matchups.applicable[0]   # Hidden Power is a damaging move → real multiplier


def test_matchups_bare_own_hidden_power_typed_from_reconstruction():
    """An OLDER trace recorded OUR HP bare ("hiddenpower"); the reconstruction `our_hp_types` map
    recovers the type for display (our side only — same source the Board/move-belief retype use)."""
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary_moves(["hiddenpower", "earthquake", "move2", "move3"],
                                                 our_species="Zapdos"),
                           _npz(), 0, our_hp_types={"zapdos": "hiddenpower(grass)"})
    assert a.matchups.move_labels[0] == "hiddenpower(grass)"
    assert a.matchups.applicable[0]


def test_matchups_bare_hidden_power_without_record_stays_bare():
    """No reconstruction record (websocket/legacy trace) → a bare own HP can't be typed, so it is
    left unchanged rather than guessed (and an opponent's HP would never be typed here anyway)."""
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary_moves(["hiddenpower", "earthquake", "move2", "move3"]),
                           _npz(), 0)
    assert a.matchups.move_labels[0] == "hiddenpower"
    assert a.matchups.applicable[0]


def test_multiplier_meaningful_predicate():
    """The display predicate: damaging moves (incl. Hidden Power + fixed/variable-power, which read
    base_power 0 in the dex) keep their multiplier; genuine status/self/field moves are phantom."""
    from main.prober.engine import _multiplier_meaningful
    # damaging → multiplier is real
    assert _multiplier_meaningful("thunderbolt")
    assert _multiplier_meaningful("hiddenpower")        # bare id, but a typed ~70-BP attack
    assert _multiplier_meaningful("hiddenpowerice")
    assert _multiplier_meaningful("seismictoss")        # fixed-damage: immunity still applies
    assert _multiplier_meaningful("return")             # variable-power
    # status / self / field → phantom multiplier → n/a
    for mv in ("spikes", "toxic", "recover", "calmmind", "protect", "roar", "", "move2"):
        assert not _multiplier_meaningful(mv)


def test_intervention_sweep_for_a_move():
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary("thunderbolt"), _npz(), 0)
    assert a.sweep.applicable and a.sweep.request_slot == 0
    assert tuple(r.multiplier for r in a.sweep.rows) == (0.0, 1.0, 2.0, 4.0)
    # P(chosen) increases monotonically as we raise the matchup multiplier.
    pc = [r.p_chosen for r in a.sweep.rows]
    assert pc == sorted(pc) and pc[3] > pc[0]
    # the engine wrote mult/4 into obs[mm_off+slot] for each sweep call
    sweep_calls = model.calls[-4:]
    assert [c[_OFF.mm_off] for c in sweep_calls] == [0.0, 0.25, 0.5, 1.0]


def test_intervention_absent_for_a_switch():
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary("switch:Dragonite"), _npz(), 0)
    assert not a.sweep.applicable and a.sweep.request_slot == -1 and a.sweep.rows == ()


def test_saliency_block_spans():
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(), _npz(), 0)
    names = [b.name for b in a.saliency.blocks]
    # gen3_cpu_damage_deleted_v1: the "active move_multipliers(4)" block is gone with the obs region.
    assert names == [
        "our_matchups(144)", "their_matchups(144)",
        "our active pokemon block(99)", "turn-history block",
    ]
    # |grad| = [0,1,2,...]; active-pokemon block spans 0:5 → mean 2.0, sum 10.
    active_block = a.saliency.blocks[2]
    assert active_block.mean_abs == 2.0 and active_block.total_abs == 10.0
    assert a.saliency.overall_mean_abs == np.arange(_OBS_LEN).mean()


def test_an_empty_obs_block_is_OMITTED_from_saliency_not_reported_as_zero():
    """The layout that actually ships: `turn_history_dim == 0` after `gen3_frame_deletion_v1`.

    `np.mean` of an empty slice is `nan` with a `RuntimeWarning`, so this block came back NaN on
    every single `analyze` of every post-deletion run until 2026-08-23 — the fixture above pins
    `turn_history_dim=10`, which is exactly why no test ever saw it.

    Omitted rather than zeroed, and that is the substantive half: "the actor's gradient does not
    reach this block" is a real saliency finding, so it must not be spelled the same way as
    "this block is not in the observation". A `0.0` here would be a confident wrong reading.
    """
    off = ObsOffsets(mm_off=10, om_off=20, tm_off=164, active_block_dim=5,
                     turn_history_offset=200, turn_history_dim=0)   # ← the shipped layout
    model = FakeProbeModel(off)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # the NaN's own warning fails the test
        a = analyze_invocation(model, _summary(), _npz(), 0)
    names = [b.name for b in a.saliency.blocks]
    assert "turn-history block" not in names, "an empty block was reported instead of omitted"
    assert names == ["our_matchups(144)", "their_matchups(144)",
                     "our active pokemon block(99)"]
    # And nothing that DID survive is NaN — the real symptom, stated as the invariant.
    assert all(np.isfinite(b.mean_abs) and np.isfinite(b.total_abs) for b in a.saliency.blocks)
    assert np.isfinite(a.saliency.overall_mean_abs)


def test_no_captured_state_skips_model():
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(), _npz(has_state=0), 0)
    assert not a.has_state and a.warnings and a.matchups is None
    assert a.sweep is None and a.saliency is None
    assert model.calls == []  # model never touched
    # outcome + cheap flags still surface without a model
    assert a.outcome["reward"]["total"] == -1.4
    assert a.value is None


def test_outcome_and_value_surface():
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(), _npz(value=2.5), 0)
    assert a.outcome["reward"]["total"] == -1.4
    assert a.outcome["events"] == []
    # single-invocation trace → recorded value, re-run value, no next/delta
    assert a.value.recorded == 2.5 and a.value.rerun == 3.14
    assert a.value.next_recorded is None and a.value.delta is None


def test_flags_switch_uncertain_faint():
    model = FakeProbeModel(_OFF)
    # a switch with a faint event, top recorded prob 3.2% (<0.5) → uncertain
    a = analyze_invocation(model, _summary("switch:Dragonite", events=["opp:Steelix:fainted"]),
                           _npz(), 0)
    assert set(a.flags) >= {"switch", "uncertain", "faint"}


def test_disagree_flag_when_rerun_argmax_differs(tmp_path):
    # Chosen says "earthquake" but the fake favors the highest-multiplier move slot,
    # so the re-run argmax is "thunderbolt" (slot 0, mult 0.5) → disagreement.
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary("earthquake"), _npz(), 0)
    assert a.rerun_argmax == "thunderbolt" and not a.agrees
    assert "disagree" in a.flags


def test_build_board_parses_sides():
    from main.prober.engine import build_board
    inv = {
        "chosen": "surf",
        "our": {"species": "milotic", "hp": "100%", "status": "PAR",
                "bench": "metagross(100%), celebi(50%,TOX(3)|SUB), snorlax(faint)"},
        "opp": {"species": "zapdos", "hp": "80%", "bench": "tyranitar(faint)"},
        "actions": {**{f"switch:m{i}": {"prob": "0%", "valid": True} for i in range(6)},
                    "hypnosis": {"prob": "0%", "valid": True},
                    "surf": {"prob": "0%", "valid": True},
                    "icebeam": {"prob": "0%", "valid": True},
                    "move3": {"prob": "0%", "valid": False},   # placeholder → dropped
                    "struggle": {"prob": "0%", "valid": False}},
    }
    b = build_board(inv)
    assert b.ours.active_species == "milotic" and b.ours.active_hp == "100%"
    assert b.ours.status == "PAR"
    assert b.ours.moves == ("hypnosis", "surf", "icebeam")        # move3 placeholder dropped
    assert [(m.species, m.hp, m.fainted) for m in b.ours.bench] == [
        ("metagross", "100%", False), ("celebi", "50%", False), ("snorlax", "faint", True)]
    # benched status+volatiles split out of the hp tail (not crammed into hp)
    assert [m.status for m in b.ours.bench] == ["", "TOX(3)|SUB", ""]
    assert b.opp.active_species == "zapdos" and b.opp.moves == ()
    assert b.opp.bench[0].fainted is True

    # item + moves annotate BOTH sides and match leniently (display-name key vs board id).
    team = {"Milotic": {"item": "choiceband", "moves": ("surf", "icebeam")},
            "celebi": {"item": "leftovers", "moves": ("psychic",)},
            "Zapdos": {"item": "salacberry", "moves": ()}}
    b2 = build_board(inv, team)
    assert b2.ours.item == "choiceband"                       # "Milotic" → "milotic"
    assert b2.ours.bench[1].item == "leftovers"               # celebi
    assert b2.ours.bench[1].moves == ("psychic",)             # moves threaded to the bench mon
    assert b2.ours.bench[0].item == ""                        # metagross: no entry
    assert b2.opp.item == "salacberry"                        # opp side annotated too


def test_analysis_carries_board():
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(), _npz(), 0)
    assert a.board is not None and a.board.ours.active_species == "Zapdos"


def test_obs_item_overlay_does_not_erase_a_known_item():
    """REGRESSION: an EMPTY obs item must not override a known summary item (our own bench mon
    showed no item because its obs slot decoded blank). The obs only OVERLAYS info, never erases."""
    from main.prober.engine import _merge_team
    base = {"tyranitar": {"item": "leftovers", "moves": ()}}
    obs_team = {"Tyranitar": {"item": "", "moves": ("crunch", "roar")}}   # blank item, has moves
    merged = _merge_team(base, obs_team)
    assert merged["tyranitar"]["item"] == "leftovers"          # NOT erased
    assert merged["tyranitar"]["moves"] == ("crunch", "roar")  # obs moves still overlaid
    # a non-empty obs item DOES win (per-turn truth, e.g. opp's revealed item)
    assert _merge_team(base, {"tyranitar": {"item": "choiceband"}})["tyranitar"]["item"] == "choiceband"


def test_obs_version_mismatch_is_flagged():
    """REGRESSION: when the trace's obs length differs from the current encoder (an obs change
    landed after the model was trained), flag it so the UI warns the obs panels are unreliable."""
    import dataclasses
    bigger = FakeProbeModel(dataclasses.replace(_OFF, total_dim=_OBS_LEN + 2))   # encoder grew by 2
    a = analyze_invocation(bigger, _summary(), _npz(), 0)                        # trace obs = _OBS_LEN
    assert a.obs_mismatch == (_OBS_LEN, _OBS_LEN + 2)
    matched = FakeProbeModel(dataclasses.replace(_OFF, total_dim=_OBS_LEN))      # same length → fine
    assert analyze_invocation(matched, _summary(), _npz(), 0).obs_mismatch is None


def test_next_board_is_the_resolved_after_state():
    """`next_board` = the board at inv+1 (the RESOLVED 'after' state); None on the last decision."""
    model = FakeProbeModel(_OFF)
    summ = _summary()
    nxt = dict(summ["invocations"][0])
    nxt["our"] = {"species": "Snorlax", "hp": "60%"}   # after a switch/hit: a different mon at 60%
    summ = {**summ, "invocations": [summ["invocations"][0], nxt]}
    a = analyze_invocation(model, summ, _npz(n=2), 0)
    assert a.next_board.ours.active_species == "Snorlax" and a.next_board.ours.active_hp == "60%"
    # the LAST decision has no following invocation → next_board is None
    assert analyze_invocation(model, summ, _npz(n=2), 1).next_board is None


def test_analysis_decodes_items_from_obs():
    """With captured state, the board picks up items + movesets from the model's obs decode —
    incl. the OPPONENT's revealed item/moves (the summary teams block only carries our side)."""
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(), _npz(), 0)
    assert a.board.opp.item == "choiceband"      # opp item surfaced from the obs decode
    assert a.board.ours.item == "leftovers"
    assert a.board.opp.moves == ("earthquake",)             # opp revealed moveset from the obs
    assert a.board.ours.moves == ("thunderbolt", "earthquake")  # trace move labels (our side)


def test_analysis_carries_crit_from_next_turn():
    """The realized crit/cant is read from the NEXT decision's TurnDelta and attached to outcome
    (inv 0 has a following captured state; the fake reports opp crit)."""
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(events=["our:zapdos:fainted"]), _npz(n=2), 0)
    assert a.outcome.get("opp_crit") is True and a.outcome.get("our_crit") is False


def test_analysis_carries_field_when_model_decodes():
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(), _npz(), 0)
    assert a.field == {"weather": "RAIN", "our_spikes": 1, "opp_spikes": 0, "turn": 8.0}


def test_win_prob_recorded_and_delta():
    """win_probs in the npz → WinProbView: recorded P(win) here + ΔP(win) to the next decision."""
    model = FakeProbeModel(_OFF)
    summ = _summary()
    summ["invocations"].append(dict(summ["invocations"][0], i=2, turn=9))   # a 2nd captured decision
    npz = {"obs": np.zeros((2, _OBS_LEN), dtype=np.float32),
           "has_state": np.array([1, 1], dtype=np.int8),
           "values": np.array([1.0, 1.0], dtype=np.float32),
           "win_probs": np.array([0.62, 0.71], dtype=np.float32)}
    a = analyze_invocation(model, summ, npz, 0)
    assert a.win_prob is not None
    assert abs(a.win_prob.recorded - 0.62) < 1e-6
    assert abs(a.win_prob.next_recorded - 0.71) < 1e-6
    assert abs(a.win_prob.delta - 0.09) < 1e-6


def test_win_prob_absent_is_none():
    """Old trace / no win-prob head → no win_probs array → win_prob is None (not a fake 0)."""
    model = FakeProbeModel(_OFF)
    assert analyze_invocation(model, _summary(), _npz(), 0).win_prob is None


def test_win_prob_nan_is_none():
    """A NaN recorded P(win) (head off for this decision) → None, distinguished from a real 0.0."""
    model = FakeProbeModel(_OFF)
    npz = {"obs": np.zeros((1, _OBS_LEN), dtype=np.float32),
           "has_state": np.array([1], dtype=np.int8),
           "values": np.array([1.0], dtype=np.float32),
           "win_probs": np.array([np.nan], dtype=np.float32)}
    assert analyze_invocation(model, _summary(), npz, 0).win_prob is None


def test_threats_decode_incoming():
    """`their_matchups` → ThreatView: present / revealed_frac / max / per-our-slot."""
    model = FakeProbeModel(_OFF)
    obs = np.zeros((1, 320), dtype=np.float32)        # long enough to hold tm_off(164)+144
    # their_matchups laid out [opp_mon, move_slot, our_mon] (6×4×6), stored /4.
    block = np.zeros((6, 4, 6), dtype=np.float32)
    block[0, 0, 2] = 2.0 / 4.0                          # opp mon0 move0 hits OUR slot2 for 2×
    block[1, 1, 2] = 4.0 / 4.0                          # opp mon1 move1 hits OUR slot2 for 4× (worst)
    obs[0, _OFF.tm_off:_OFF.tm_off + 144] = block.reshape(-1)
    npz = {"obs": obs, "has_state": np.array([1], dtype=np.int8),
           "values": np.array([1.0], dtype=np.float32)}
    a = analyze_invocation(model, _summary(), npz, 0)
    t = a.threats
    assert t is not None and t.present is True
    assert abs(t.max_incoming - 4.0) < 1e-6            # the 4× cell, ×4-denormalised
    assert t.per_our_slot_max[2] == 4.0 and t.per_our_slot_max[0] == 0.0   # threat concentrated on slot2
    assert 0.0 < t.revealed_frac < 0.05                # only 2 of 144 cells populated
    # their_matchups is now its own saliency block
    assert any(b.name == "their_matchups(144)" for b in a.saliency.blocks)


def test_threats_none_when_block_absent():
    """A too-short obs (no their_matchups room) yields threats=None, not a crash."""
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(), _npz(), 0)   # _OBS_LEN=256 < tm_off+144
    assert a.threats is None


def test_offsets_resolve_matches_layout():
    """Regression guard: pin the obs offsets the engine depends on."""
    off = ObsOffsets.resolve()
    # gen3_cpu_damage_deleted_v1: OFFSET_REACTIVE is 1454, but the reactive block now starts with only
    # 11 scalars — the 8 active-move scalars, the 44-dim move-effect block and the 51-dim incoming-damage
    # block were DELETED from the obs (all three had live GPU homes and were merely MASKED before).
    # So the matchups moved 1568 -> 1465 and the obs total 2992 -> 2889.
    assert off.mm_off == 0            # active-move multipliers: DELETED (0 = absent; consumers no-op)
    # gen3_entity_rehome_v1: the two 144-dim matchup matrices are DELETED from the obs —
    # the D/V edge families own pair effectiveness GPU-side. 0 = absent; ThreatView/saliency no-op.
    assert off.om_off == 0
    assert off.tm_off == 0
    assert off.active_block_dim == 99
    assert off.incoming_off == 0      # incoming-damage block: DELETED from the obs (decoder no-ops)
    assert off.incoming_dim == 0
    assert off.pokemon_full_dim == 122  # pair_history: 119 vector + 2 trapping + 1 active
    # gen3_wish_wired_v1 → rehome: the pending-Wish scalars ride the lean board block.
    # gen3_pair_history_v1: the per-mon slot widened 116 -> 122 (+6 last-action), so every
    # top-level offset after the team blocks shifted +72, and the 180-dim pair block sits
    # between reactive and base; the H-B event window follows (total 3529).
    assert off.wish_our_off == 1603   # OFFSET_REACTIVE(1600) + wish_floating_our offset(3)
    assert off.wish_opp_off == 1604   # OFFSET_REACTIVE(1600) + wish_floating_opp offset(4)
    # gen3_frame_deletion_v1 took this 3529 -> 2437 (the event window became the LAST block,
    # so total == base); gen3_event_semantics_v1 then took it -> 2501, adding `faint_cause_id`
    # and `item_transition` to every event row (32 rows x 20 -> x 22). A LITERAL on purpose:
    # this is the tripwire that makes an UNINTENDED obs-width change fail loudly, so it moves
    # only when someone decided it should.
    assert off.total_dim == 2501

    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    lay = Gen3ObservationEncoder(load_mappings()).get_layout()
    assert off.turn_history_offset == lay.get("turn_history_offset", 0)
    # gen3_frame_deletion_v1: absent keys ⇒ 0, which is how `Offsets` says "no lag-frame block".
    # Kept (rather than deleted) because the prober still reads ARCHIVED runs whose layouts DO
    # carry them, and 0-vs-present is exactly the branch its saliency views switch on.
    assert off.turn_history_dim == lay.get("n_history_turns", 0) * lay.get("turn_delta_dim", 0)


# A synthetic layout WITH the incoming-damage block, for the belief-decode tests. Active flag is
# the last dim of each pokemon_full_dim(8)-wide our-mon block; our slot 1 is active (idx 1*8+7=15).
_OFF_INC = ObsOffsets(
    mm_off=10, om_off=20, tm_off=200, active_block_dim=5,
    turn_history_offset=400, turn_history_dim=10,
    incoming_off=350, incoming_dim=33, incoming_per_mon=5, incoming_recovery=3,
    pokemon_full_dim=8,
)


def _obs_with_belief(active_slot=1, pko=0.9, exp=0.7, outspeed=0.25):
    """Build a synthetic obs carrying an incoming-damage block (active slot's special channel hot)
    and the per-mon active flag set for ``active_slot``."""
    obs = np.zeros((1, 512), dtype=np.float32)
    obs[0, active_slot * 8 + 7] = 1.0                       # active flag on our slot `active_slot`
    base = 350 + active_slot * 5
    obs[0, base + 1] = exp                                   # spec_exp
    obs[0, base + 3] = pko                                   # spec_pko
    obs[0, base + 4] = outspeed                              # p_outspeed
    obs[0, 350 + 30:350 + 33] = (0.35, 0.35, 1.0)           # recovery scalars
    return obs


def test_incoming_belief_decode_active_slot():
    """`incoming_damage` block → IncomingBeliefView: active slot resolved from the per-mon active
    flag, P(KO)=max(phys,spec), recovery scalars at the tail."""
    model = FakeProbeModel(_OFF_INC)
    obs = _obs_with_belief(active_slot=1, pko=0.9, exp=0.7, outspeed=0.25)
    npz = {"obs": obs, "has_state": np.array([1], dtype=np.int8),
           "values": np.array([1.0], dtype=np.float32)}
    a = analyze_invocation(model, _summary(), npz, 0)
    inc = a.incoming
    assert inc is not None and inc.present is True
    assert abs(inc.active_pko - 0.9) < 1e-6                  # active slot's spec_pko
    assert abs(inc.active_exp - 0.7) < 1e-6
    assert abs(inc.active_outspeed - 0.25) < 1e-6
    assert abs(inc.max_pko - 0.9) < 1e-6
    assert inc.per_slot_pko[1] == inc.active_pko and inc.per_slot_pko[0] == 0.0
    assert abs(inc.recovery_rate - 0.35) < 1e-6 and abs(inc.cures_status - 0.35) < 1e-6
    assert inc.recovery_known == 1.0
    # the block is now a named saliency region for BOTH heads
    assert any(b.name == "incoming_damage(33)" for b in a.saliency.blocks)
    assert a.value_saliency is not None
    assert any(b.name == "incoming_damage(33)" for b in a.value_saliency.blocks)


# The live crit-split layout: per_mon=8, dim=51, total_dim set so the length guard is exercised.
_OFF_INC8 = ObsOffsets(
    mm_off=10, om_off=20, tm_off=200, active_block_dim=5,
    turn_history_offset=400, turn_history_dim=10,
    incoming_off=300, incoming_dim=51, incoming_per_mon=8, incoming_recovery=3,
    pokemon_full_dim=8, total_dim=512,
)


def test_incoming_belief_decode_crit_split_8field():
    """The crit-split 8-field layout decodes: active_pko = RECONSTRUCTED crit-inclusive (nocrit+delta),
    active_pko_nocrit = the modal line (≤ active_pko), threat_revealed = the provenance scalar; and a
    wrong-length (old/foreign-arch) obs is REFUSED by the total_dim guard."""
    from main.prober.engine import decode_incoming_belief
    obs = np.zeros(512, dtype=np.float32)
    obs[1 * 8 + 7] = 1.0                          # active flag → our slot 1
    base = 300 + 1 * 8                            # incoming_off + slot 1 * per_mon
    obs[base + 0] = 0.7                           # phys_exp
    obs[base + 2] = 0.5                           # phys_pko_nocrit (modal line)
    obs[base + 4] = 0.06                          # phys_crit_delta (the crit tax)
    obs[base + 6] = 0.25                          # p_outspeed
    obs[base + 7] = 0.8                           # threat_revealed (a 0.8-prior guess)
    obs[300 + 6 * 8: 300 + 6 * 8 + 3] = (0.35, 0.35, 1.0)   # recovery scalars at the block tail
    bel = decode_incoming_belief(obs, _OFF_INC8)
    assert bel is not None and bel.present
    assert abs(bel.active_pko - 0.56) < 1e-6          # crit-inclusive = nocrit 0.5 + delta 0.06
    assert abs(bel.active_pko_nocrit - 0.5) < 1e-6    # the modal line
    assert bel.active_pko_nocrit <= bel.active_pko
    assert abs(bel.threat_revealed - 0.8) < 1e-6
    assert abs(bel.active_exp - 0.7) < 1e-6
    assert abs(bel.max_pko - 0.56) < 1e-6
    # the total_dim guard refuses a wrong-length (archived old-arch) obs rather than mis-slicing it
    assert decode_incoming_belief(np.zeros(400, dtype=np.float32), _OFF_INC8) is None


def test_incoming_belief_none_when_block_absent():
    """The default _OFF has incoming_dim=0 → no belief decoded, no incoming saliency block, no crash.
    value_saliency is still None there because the default FakeProbeModel path is unaffected."""
    model = FakeProbeModel(_OFF)
    npz = {"obs": np.zeros((1, _OBS_LEN), dtype=np.float32),
           "has_state": np.array([1], dtype=np.int8), "values": np.array([1.0], dtype=np.float32)}
    a = analyze_invocation(model, _summary(), npz, 0)
    assert a.incoming is None
    assert not any(b.name.startswith("incoming_damage") for b in a.saliency.blocks)


# --- loss attribution taxonomy (pure, model-free) --------------------------

def _cat(**feat):
    """attribute_turning_point over a feature dict with sensible neutral defaults — so a test
    sets ONLY the discriminating fields and trusts the rest not to trip an earlier rule."""
    from main.prober.engine import attribute_turning_point
    base = {"turns": 50, "is_switch": False, "is_setup": False, "our_hp": 0.8,
            "our_hp_delta": 0.0, "faint": False, "active_pko": 0.1, "active_outspeed": 1.0,
            "max_pko": 0.1, "n_healthy_bench": 3, "min_other_pko": 0.0, "delta_v": -10.0,
            "td": -10.0, "v_at": 5.0}
    base.update(feat)
    return attribute_turning_point(base)["category"]


def test_taxonomy_stall_timeout_wins_first():
    assert _cat(turns=240, faint=True, active_pko=1.0) == "stall_timeout"


def test_taxonomy_post_faint_replacement_before_combat():
    # our active already fainted (hp≈0) → a forced replacement, not the combat decision that lost it.
    assert _cat(our_hp=0.0, faint=True, is_switch=True, active_pko=0.0) == "post_faint_replacement"


def test_taxonomy_surprise_ohko_is_obs_underread():
    # healthy mon DIED, belief UNDER-read it → obs lever.
    assert _cat(faint=True, our_hp=0.9, active_pko=0.1) == "surprise_ohko"


def test_taxonomy_ignored_threat_death_belief_fired_with_pivot():
    # belief FIRED, healthy pivot available, mon died anyway → reward/policy lever.
    # Outspeed is irrelevant now (the old gate wrongly excluded outspeed deaths).
    assert _cat(faint=True, our_hp=0.4, active_pko=0.95, active_outspeed=1.0,
                n_healthy_bench=2) == "ignored_threat_death"


def test_taxonomy_doomed_already_no_pivot_left():
    assert _cat(faint=True, active_pko=0.95, n_healthy_bench=0) == "doomed_already"


def test_taxonomy_greedy_setup_before_attrition():
    # a setup move punished — but only when it's not a clearer death pattern (no pivot/belief fired).
    assert _cat(is_setup=True, faint=False, active_pko=0.2) == "greedy_setup"


def test_taxonomy_attrition_death_partial_belief():
    # a worn-down mon died with the belief only partly fired (mid pko) → attrition.
    assert _cat(faint=True, our_hp=0.45, active_pko=0.4, n_healthy_bench=2) == "attrition_death"


def test_taxonomy_critic_blindspot_vs_positional_grind_fallback_on_v():
    # No death, NO win-prob recorded → fall back to V vs v_even (default 0).
    assert _cat(faint=False, v_at=8.0, wp_at=None) == "critic_blindspot"   # V above even → was winning
    assert _cat(faint=False, v_at=-8.0, wp_at=None) == "positional_grind"  # V below even → already behind
    assert _cat(faint=False, v_at=None, wp_at=None) == "positional_grind"  # unknown → not a blindspot claim


def test_taxonomy_split_prefers_winprob_over_v_sign():
    # The KEY re-centering: V's zero is NOT "even" (a 50/50 self-mirror reads V<0), so a NEGATIVE V with
    # the calibrated P(win) ≥ 0.5 is a THROW (was winning), NOT a grind. wp_at must OVERRIDE the V sign.
    assert _cat(faint=False, v_at=-8.0, wp_at=0.62) == "critic_blindspot"   # behind on V, ahead on P(win)
    assert _cat(faint=False, v_at=5.0, wp_at=0.40) == "positional_grind"    # ahead on V, behind on P(win)
    # threshold + the configurable wp_even
    assert _cat(faint=False, wp_at=0.50) == "critic_blindspot"              # exactly at 0.5 counts as winning
    assert _cat(faint=False, wp_at=0.49) == "positional_grind"
    assert _cat(faint=False, wp_at=0.60, wp_even=0.65) == "positional_grind"  # raise the bar → behind


def test_taxonomy_v_even_recenters_no_winprob_run():
    # On a run without a win-prob head, pass v_even = the checkpoint's structural even-point (e.g. −6.5)
    # so a slightly-negative V no longer reads as "behind".
    assert _cat(faint=False, v_at=-5.0, wp_at=None, v_even=-6.5) == "critic_blindspot"  # above the even-point
    assert _cat(faint=False, v_at=-8.0, wp_at=None, v_even=-6.5) == "positional_grind"  # below it


def test_taxonomy_total_on_empty_dict():
    # Every predicate is None-tolerant: a feature dict missing everything still categorizes.
    from main.prober.engine import attribute_turning_point
    out = attribute_turning_point({})
    assert out["category"] in {"positional_grind", "other"} and out["lever"]


# --- representation probe (pure stats) -------------------------------------

def test_fit_probe_recovers_decodable_classification():
    from main.prober.engine import fit_probe
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 16))
    w = rng.standard_normal(16)
    y = (X @ w > np.median(X @ w)).astype(float)
    r = fit_probe(X, y, "classification", groups=np.where(X[:, 0] > 0, "easy", "hard"), seed=0)
    assert r["overall"]["accuracy"] > 0.85 and r["overall"]["auc"] > 0.9
    assert r["overall"]["lift"] > 0.3                      # well above the majority baseline
    assert set(r["by_group"]) == {"easy", "hard"} and all(r["by_group"].values())


def test_fit_probe_noise_label_collapses_to_baseline():
    from main.prober.engine import fit_probe
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 16))
    y = (rng.standard_normal(400) > 0).astype(float)     # label independent of X
    r = fit_probe(X, y, "classification", seed=0)
    assert abs(r["overall"]["lift"]) < 0.08               # no real signal → ~majority baseline
    assert 0.4 < r["overall"]["auc"] < 0.6                # AUC ~0.5


def test_fit_probe_regression_decodable_vs_noise():
    from main.prober.engine import fit_probe
    rng = np.random.default_rng(2)
    X = rng.standard_normal((400, 12))
    w = rng.standard_normal(12)
    good = fit_probe(X, X @ w + 0.1 * rng.standard_normal(400), "regression", seed=0)
    noise = fit_probe(X, rng.standard_normal(400), "regression", seed=0)
    assert good["overall"]["r2"] > 0.9
    assert noise["overall"]["r2"] < 0.1                   # mean-predictor R² ≈ 0 on noise


def test_fit_probe_too_few_samples_is_graceful():
    from main.prober.engine import fit_probe
    r = fit_probe(np.zeros((4, 3)), np.array([0.0, 1, 0, 1]), "classification", seed=0)
    assert r["overall"] is None and r["n"] == 4          # <5 usable → None, no crash


def test_history_slot_saliency_splits_block():
    from main.prober.engine import history_slot_saliency
    off = ObsOffsets(mm_off=0, om_off=0, tm_off=0, active_block_dim=5,
                     turn_history_offset=10, turn_history_dim=12, turn_delta_dim=3)  # 4 slots × 3
    g = np.zeros(40)
    g[10:13] = 1.0          # slot 0 high
    g[19:22] = 0.6          # slot 3 (10 + 3*3) medium
    s = history_slot_saliency(g, off)
    assert len(s) == 4
    assert s[0] == 1.0 and abs(s[3] - 0.6) < 1e-9 and s[1] == 0.0 and s[2] == 0.0
    # no turn_delta_dim → can't split → empty (graceful)
    off0 = ObsOffsets(mm_off=0, om_off=0, tm_off=0, active_block_dim=5,
                      turn_history_offset=10, turn_history_dim=12)
    assert history_slot_saliency(g, off0) == []


# ---------------------------------------------------------------------------
# Hidden-opponent species belief (build_belief + analyze wiring) — model-free
# ---------------------------------------------------------------------------

def test_build_belief_parses_hidden_slots():
    from main.prober.engine import build_belief, BeliefView, BeliefSlotView
    inv = {"belief": [
        {"slot": 2, "top": [{"species": "tyranitar", "prob": "41.2%"},
                            {"species": "skarmory", "prob": "18.7%"}]},
        {"slot": 3, "top": [{"species": "metagross", "prob": "33.0%"}]},
    ]}
    bv = build_belief(inv)
    assert isinstance(bv, BeliefView)
    assert [s.slot for s in bv.slots] == [2, 3]
    assert isinstance(bv.slots[0], BeliefSlotView)
    assert bv.slots[0].top[0][0] == "tyranitar" and abs(bv.slots[0].top[0][1] - 0.412) < 1e-6
    assert bv.slots[0].top[1][0] == "skarmory" and abs(bv.slots[0].top[1][1] - 0.187) < 1e-6
    assert bv.slots[1].top == (("metagross", bv.slots[1].top[0][1]),) and abs(bv.slots[1].top[0][1] - 0.33) < 1e-6


def test_build_belief_absent_or_empty_is_none():
    from main.prober.engine import build_belief
    assert build_belief({}) is None                                   # belief off (no block)
    assert build_belief({"belief": []}) is None                       # on, but nothing hidden this turn
    assert build_belief({"belief": [{"slot": 2, "top": []}]}) is None  # empty top → slot dropped → None


def test_analyze_includes_belief_when_present():
    model = FakeProbeModel(_OFF)
    summary = _summary()
    summary["invocations"][0]["belief"] = [
        {"slot": 4, "top": [{"species": "gengar", "prob": "55.0%"}]}]
    a = analyze_invocation(model, summary, _npz(), 0)
    assert a.belief is not None
    assert a.belief.slots[0].slot == 4
    assert a.belief.slots[0].top[0][0] == "gengar"


def test_analyze_belief_none_when_off():
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(), _npz(), 0)   # default summary has no belief block
    assert a.belief is None


def test_analyze_belief_present_without_captured_state():
    """Belief is model-free (from the summary), so it's available even on a no-state invocation."""
    model = FakeProbeModel(_OFF)
    summary = _summary()
    summary["invocations"][0]["belief"] = [
        {"slot": 5, "top": [{"species": "snorlax", "prob": "60.0%"}]}]
    a = analyze_invocation(model, summary, _npz(has_state=0), 0)
    assert a.has_state is False
    assert a.belief is not None and a.belief.slots[0].top[0][0] == "snorlax"


# ---------------------------------------------------------------------------
# Opponent intent — α/β (build_opp_intent + opp_intent_text + analyze wiring), model-free
# ---------------------------------------------------------------------------

_INTENT = {
    "alpha": [{"name": "earthquake", "p": 0.52}, {"name": "SWITCH", "p": 0.21},
              {"name": "icebeam", "p": 0.18}],
    "beta": [{"slot": 3, "p": 0.44, "species": "blissey"},
             {"slot": 4, "p": 0.31, "species": None}],
}


def test_build_opp_intent_parses_alpha_and_beta():
    from main.prober.engine import build_opp_intent, OppIntentView
    v = build_opp_intent({"opp_intent": _INTENT})
    assert isinstance(v, OppIntentView)
    # The recorder already ranked α; the order is passed through, NOT re-derived.
    assert [o.name for o in v.alpha] == ["earthquake", "SWITCH", "icebeam"]
    assert v.top.name == "earthquake" and abs(v.top.p - 0.52) < 1e-9
    # SWITCH is identified structurally, so no surface has to compare a magic string itself.
    assert [o.is_switch for o in v.alpha] == [False, True, False]
    assert abs(v.switch_p - 0.21) < 1e-9
    assert [c.slot for c in v.beta] == [3, 4]
    assert v.beta[0].species == "blissey"
    # A slot with no species head to name it stays an honest bare index, not a fabricated mon.
    assert v.beta[1].species is None


def test_build_opp_intent_absent_or_empty_is_none():
    """Every trace written before the v67 heads has no block at all — that must read as 'no data',
    which is a different thing from 'the model expected nothing'."""
    from main.prober.engine import build_opp_intent
    assert build_opp_intent({}) is None                             # heads off (no block)
    assert build_opp_intent({"opp_intent": {}}) is None             # block present but empty
    assert build_opp_intent({"opp_intent": {"alpha": []}}) is None  # no named option → nothing to say


def test_build_opp_intent_without_beta():
    """β is built only when the head is on; α alone is a complete, renderable view."""
    from main.prober.engine import build_opp_intent
    v = build_opp_intent({"opp_intent": {"alpha": [{"name": "fireblast", "p": 0.7}]}})
    assert v.beta == () and v.switch_p is None and v.top.name == "fireblast"


def test_opp_intent_text_is_the_shared_sentence():
    from main.prober.engine import build_opp_intent, opp_intent_text
    assert opp_intent_text(None) == ""
    text = opp_intent_text(build_opp_intent({"opp_intent": _INTENT}))
    assert text == "expects earthquake 52% · SWITCH 21% · icebeam 18%"
    # β is appended ONLY when α actually leads with the switch — otherwise it is noise on the line.
    assert "in:" not in text
    sw = {"alpha": [{"name": "SWITCH", "p": 0.61}, {"name": "toxic", "p": 0.39}],
          "beta": [{"slot": 2, "p": 0.58, "species": "blissey", "revealed": True}]}
    assert opp_intent_text(build_opp_intent({"opp_intent": sw})) == \
        "expects SWITCH 61% · toxic 39% → in: blissey 58%"


# ---------------------------------------------------------------------------
# β name PROVENANCE (`gen3_beta_revealed_naming_v1`) — the read-time half
# ---------------------------------------------------------------------------
# The recorder used to name EVERY β slot from the species posterior, which is un-supervised on a
# revealed slot: measured over a 843-battle sweep (2026-08-19) the rendered mon was not on the
# opponent's team at all in 73.3% of 6,876 pivots, and one such label was read as "β predicts
# porygon2" on a turn where β's slot held the revealed Salamence and β was CORRECT. Traces already
# on disk baked that name and do not carry the board, so read time attaches a CAVEAT — it never
# re-derives a name it cannot ground.

def test_beta_names_from_an_OLD_trace_are_all_caveated():
    """Every trace written before the recorder fix carries no `revealed` key at all. That must read
    as 'posterior decode', because that is exactly what those names are."""
    from main.prober.engine import build_opp_intent, BELIEF_NAME_CAVEAT
    v = build_opp_intent({"opp_intent": _INTENT})       # no `revealed` key anywhere
    assert [c.revealed for c in v.beta] == [False, False]
    assert v.beta[0].species == "blissey" and v.beta[0].caveat == BELIEF_NAME_CAVEAT
    # …and the name itself is passed through UNCHANGED. Repairing it is impossible (the board is
    # not in the trace) and guessing a replacement would be the same defect facing the other way.
    assert v.beta[0].species == "blissey"


def test_a_revealed_beta_name_carries_no_caveat():
    """A slot the recorder read off the board is a FACT — caveating it would teach the reader to
    discount the honest half too."""
    from main.prober.engine import build_opp_intent
    v = build_opp_intent({"opp_intent": {
        "alpha": [{"name": "SWITCH", "p": 1.0}],
        "beta": [{"slot": 0, "p": 0.7, "species": "salamence", "revealed": True},
                 {"slot": 3, "p": 0.3, "species": "snorlax", "revealed": False}]}})
    assert (v.beta[0].revealed, v.beta[0].caveat) == (True, None)
    assert v.beta[1].revealed is False and v.beta[1].caveat


def test_a_bare_index_is_not_caveated_because_there_is_no_name_to_qualify():
    """No species head ⇒ the row renders as `slot 4`, which already claims nothing. A caveat there
    would attach a qualifier to a name that does not exist."""
    from main.prober.engine import build_opp_intent
    v = build_opp_intent({"opp_intent": {"alpha": [{"name": "SWITCH", "p": 1.0}],
                                         "beta": [{"slot": 4, "p": 0.9, "species": None}]}})
    assert v.beta[0].species is None and v.beta[0].caveat is None


def test_the_shared_sentence_carries_the_caveat_for_a_posterior_beta_name():
    """The sentence is the shared vocabulary — a surface that prints only `text` would otherwise
    lose the qualifier silently, which is the exact failure the engine/renderer split prevents."""
    from main.prober.engine import build_opp_intent, opp_intent_text, BELIEF_NAME_CAVEAT
    old = {"alpha": [{"name": "SWITCH", "p": 0.61}, {"name": "toxic", "p": 0.39}],
           "beta": [{"slot": 2, "p": 0.58, "species": "blissey"}]}
    assert opp_intent_text(build_opp_intent({"opp_intent": old})) == \
        f"expects SWITCH 61% · toxic 39% → in: blissey 58% · {BELIEF_NAME_CAVEAT}"


def test_analyze_includes_opp_intent_with_and_without_state():
    """Model-free, so it survives an invocation with no captured state — same contract as belief."""
    model = FakeProbeModel(_OFF)
    summary = _summary()
    summary["invocations"][0]["opp_intent"] = _INTENT
    assert analyze_invocation(model, summary, _npz(), 0).opp_intent.top.name == "earthquake"
    a = analyze_invocation(model, summary, _npz(has_state=0), 0)
    assert a.has_state is False and a.opp_intent.top.name == "earthquake"
    assert analyze_invocation(model, _summary(), _npz(), 0).opp_intent is None


# ---------------------------------------------------------------------------
# Privileged belief-vs-truth (build_belief_truth + slot-matching + analyze wiring)
# ---------------------------------------------------------------------------

def _maps10():
    """A synthetic 10-species vocab ({num->id}, {id->num}) so the matching unit-tests need no data."""
    return ({i: f"sp{i}" for i in range(10)}, {f"sp{i}": i for i in range(10)})


def test_build_belief_truth_matches_and_marks_correct():
    from main.prober.engine import build_belief_truth
    logits = np.full((6, 10), -5.0)
    logits[4, 7] = 8.0       # believed slot 4 strongly predicts sp7
    logits[5, 3] = 8.0       # believed slot 5 strongly predicts sp3
    mask = np.array([False, False, False, False, True, True])
    true_team = ["sp1", "sp2", "sp7", "sp3"]      # sp1/sp2 revealed; sp7/sp3 hidden
    v = build_belief_truth(logits, mask, ["sp1", "sp2"], true_team, top_k=3, maps=_maps10())
    assert v is not None
    by = {m.species: m for m in v.mons}
    assert by["sp1"].revealed and by["sp2"].revealed
    assert not by["sp7"].revealed and not by["sp3"].revealed
    # Each hidden mon is matched to the slot that predicts it, and is the top-1 → ✓.
    assert by["sp7"].guessed_right and by["sp7"].guess[0][0] == "sp7"
    assert by["sp3"].guessed_right and by["sp3"].guess[0][0] == "sp3"
    assert v.n_hidden == 2 and v.n_correct == 2


def test_build_belief_truth_wrong_guess_marks_x_and_rank():
    from main.prober.engine import build_belief_truth
    logits = np.full((6, 10), -5.0)
    logits[5, 8] = 5.0       # model's top guess is sp8 (wrong)
    logits[5, 2] = 4.0       # the true sp2 is the model's 2nd choice
    mask = np.array([False, False, False, False, False, True])
    v = build_belief_truth(logits, mask, ["sp1"], ["sp1", "sp2"], top_k=3, maps=_maps10())
    by = {m.species: m for m in v.mons}
    assert not by["sp2"].guessed_right
    assert by["sp2"].true_rank == 2           # sp2 was the model's 2nd choice
    assert by["sp2"].guess[0][0] == "sp8"     # top guess was the wrong sp8
    assert v.n_correct == 0 and v.n_hidden == 1


def test_build_belief_truth_none_without_privileged_team():
    from main.prober.engine import build_belief_truth
    assert build_belief_truth(np.zeros((6, 10)), np.ones(6, bool), [], None, maps=_maps10()) is None
    assert build_belief_truth(np.zeros((6, 10)), np.ones(6, bool), [], [], maps=_maps10()) is None


def test_belief_view_from_logits_decodes_only_believed_slots():
    from main.prober.engine import belief_view_from_logits
    logits = np.full((6, 10), -5.0)
    logits[2, 7] = 6.0; logits[2, 3] = 3.0
    logits[0, 1] = 9.0                        # slot 0 NOT believed → must be skipped
    mask = np.array([False, False, True, False, False, False])
    v = belief_view_from_logits(logits, mask, top_k=2, num_to_id=_maps10()[0])
    assert [s.slot for s in v.slots] == [2]
    assert v.slots[0].top[0][0] == "sp7"
    assert v.slots[0].top[0][1] > v.slots[0].top[1][1]


def test_revealed_opp_species_from_board():
    from main.prober.engine import revealed_opp_species, BoardView, SideBoard, MonState
    opp = SideBoard(active_species="metagross", active_hp="100%", status="", boosts="", moves=(),
                    bench=(MonState(species="salamence", hp="faint", fainted=True),
                           MonState(species="jirachi", hp="80%", fainted=False)))
    ours = SideBoard(active_species="zapdos", active_hp="100%", status="", boosts="", moves=(), bench=())
    assert set(revealed_opp_species(BoardView(ours=ours, opp=opp))) == {"metagross", "salamence", "jirachi"}
    assert revealed_opp_species(None) == ()


class _BeliefFakeModel(FakeProbeModel):
    """FakeProbeModel that also exposes a (synthetic) hidden-opp belief, so the analyze→belief_truth
    wiring is exercised without loading a real checkpoint."""
    def __init__(self, offsets, species_logits, believed_mask):
        super().__init__(offsets)
        self._sp, self._bm = species_logits, believed_mask

    def belief(self, obs, mask):
        return (self._sp, self._bm)


def test_analyze_belief_truth_end_to_end_with_model_belief():
    # slot 5 strongly predicts tyranitar (national-dex num 248); uses the REAL species vocab.
    logits = np.full((6, 400), -8.0)
    logits[5, 248] = 9.0
    bmask = np.array([False, False, False, False, False, True])
    model = _BeliefFakeModel(_OFF, logits, bmask)
    a = analyze_invocation(model, _summary(), _npz(), 0, opp_team=("steelix", "tyranitar"))
    # Anonymous belief is re-computed from the model (the summary has no belief block here).
    assert a.belief is not None and a.belief.slots[0].top[0][0] == "tyranitar"
    # Privileged truth: steelix revealed (it's the opp active), tyranitar hidden + correctly guessed.
    assert a.belief_truth is not None
    by = {m.species: m for m in a.belief_truth.mons}
    assert by["steelix"].revealed and not by["tyranitar"].revealed
    assert by["tyranitar"].guessed_right and by["tyranitar"].guess[0][0] == "tyranitar"
    assert a.belief_truth.n_hidden == 1 and a.belief_truth.n_correct == 1


def test_analyze_belief_truth_none_without_opp_team():
    logits = np.full((6, 400), -8.0); logits[5, 248] = 9.0
    bmask = np.array([False, False, False, False, False, True])
    model = _BeliefFakeModel(_OFF, logits, bmask)
    a = analyze_invocation(model, _summary(), _npz(), 0)   # no privileged team
    assert a.belief is not None and a.belief_truth is None


# ── build_result_timeline (the RESULT panel data model) ──────────────────────────────────────────
from main.prober.engine import build_result_timeline   # noqa: E402


def _tl(our_action, our_delta, opp_action, opp_delta, *, ours="salamence", opps="tyranitar",
        events=None, phase="move_selection", our_before="100%", opp_before="100%",
        our_after=None, opp_after=None, **extra):
    out = {"our": {"action": our_action, "hp_delta": our_delta},
           "opp": {"action": opp_action, "hp_delta": opp_delta}, "events": events or [], **extra}
    return build_result_timeline(out, ours, opps, phase, our_hp_before=our_before,
                                 opp_hp_before=opp_before, our_hp_after=our_after, opp_hp_after=opp_after)


def test_timeline_attributes_loss_to_the_opponents_move():
    # The core bug: a mon's HP loss must show on the OPPONENT's move, not its own. salamence(we)
    # brickbreak hits tyranitar for 85%; tyranitar's rockslide hits salamence for 73%.
    tl = _tl("brickbreak", "-73%", "rockslide", "-85%", our_after="27%", opp_after="15%",
             move_order="we_first")
    assert [e["side"] for e in tl] == ["we", "opp"]                 # we_first → we on top
    we, opp = tl
    assert we["move"] == "brickbreak" and we["target"] == "tyranitar"
    assert (we["damage"], we["hp_before"], we["hp_after"]) == ("85%", "100%", "15%")  # before=after+dmg
    assert opp["move"] == "rockslide" and opp["target"] == "salamence"
    assert (opp["damage"], opp["hp_before"], opp["hp_after"]) == ("73%", "100%", "27%")


def test_timeline_order_follows_move_order():
    we_first = _tl("icebeam", "-72%", "hiddenpower", "-100%", move_order="we_first")
    opp_first = _tl("icebeam", "-72%", "hiddenpower", "-100%", move_order="opp_first",
                    events=["opp:salamence:fainted"])
    assert [e["side"] for e in we_first][:2] == ["we", "opp"]
    assert [e["side"] for e in opp_first][:2] == ["opp", "we"]      # opp_first → opp on top


def test_timeline_faint_and_forced_replacement():
    # our icebeam KO's salamence(opp); opp's hiddenpower hit us; opp sends in metagross.
    tl = _tl("icebeam", "-72%", "hiddenpower → metagross_sent_in", "-100%",
             ours="tyranitar", opps="salamence", events=["opp:salamence:fainted"],
             our_after="28%", opp_after="100%", move_order="opp_first", opp_crit=True)
    kinds = [(e["side"], e["kind"]) for e in tl]
    assert kinds == [("opp", "move"), ("we", "move"), ("opp", "send_in")]
    opp_move, we_move, send = tl
    assert opp_move["crit"] is True and opp_move["damage"] == "72%"   # crit on the attacker
    assert we_move["target"] == "salamence" and we_move["hp_after"] == "faint"
    assert send["sent_in"] == "metagross"


def test_timeline_our_attack_visible_on_bare_sent_in():
    # jolteon thunderbolts, KO's suicune before it acts (bare 'claydol_sent_in', no opp move).
    # Regression: our KO must be a line, not hidden behind the opponent's replacement.
    tl = _tl("thunderbolt", "+0%", "claydol_sent_in", "-31%", ours="jolteon", opps="suicune",
             events=["opp:suicune:fainted"], opp_before="31%", move_order="we_first")
    assert [(e["side"], e["kind"]) for e in tl] == [("we", "move"), ("opp", "send_in")]
    assert tl[0]["move"] == "thunderbolt" and tl[0]["target"] == "suicune"
    assert tl[0]["damage"] == "31%" and tl[0]["hp_after"] == "faint"
    assert tl[1]["sent_in"] == "claydol"


def test_timeline_switch_resolves_first_and_redirects_target():
    # We switch tyranitar→skarmory; the switch resolves first, so opp's meteormash hits skarmory.
    tl = _tl("switched_to:skarmory", "-25%", "meteormash", "+0%", ours="tyranitar",
             opps="metagross", our_after="75%", move_order="opp_first")
    assert [(e["side"], e["kind"]) for e in tl] == [("we", "switch"), ("opp", "move")]
    assert tl[0]["switch_to"] == "skarmory"
    assert tl[1]["target"] == "skarmory" and tl[1]["damage"] == "25%"   # redirected to the switch-in


def test_timeline_couldnt_move_deals_no_damage():
    tl = _tl("earthquake", "+0%", "focuspunch", "-55%", ours="swampert", opps="tyranitar",
             opp_after="45%", move_order="we_first", opp_cant="focuspunch")
    we, opp = tl
    assert we["move"] == "earthquake" and we["damage"] == "55%"
    assert opp["cant"] == "focuspunch" and opp["damage"] == "" and opp["target"] == ""


def test_timeline_status_move_not_credited_with_damage():
    # our toxic applies TOX (status) — never a damage number; opp earthquake KO's our magneton.
    tl = _tl("toxic", "-79%", "earthquake", "+0%", ours="magneton", opps="swampert",
             events=["our:magneton:fainted", "opp:swampert:TOX"], move_order="opp_first")
    by_move = {e["move"]: e for e in tl if e["kind"] == "move"}
    assert by_move["toxic"]["damage"] == "" and by_move["toxic"]["status"] == "TOX"
    assert by_move["earthquake"]["target"] == "magneton" and by_move["earthquake"]["hp_after"] == "faint"


def test_timeline_hidden_power_is_attributed_despite_bp0_quirk():
    # 'hiddenpower' has static base power 0 (is_damaging False) but really hits — must be attributed.
    # opp's Hidden Power hits our celebi for 58% (our side's loss); our surf did nothing back.
    tl = _tl("surf", "-58%", "hiddenpower", "+0%", ours="celebi", opps="metagross",
             our_after="42%", move_order="opp_first")
    opp = next(e for e in tl if e["move"] == "hiddenpower")
    assert opp["damage"] == "58%" and opp["target"] == "celebi"


def test_timeline_non_damaging_move_does_not_steal_residual_loss():
    # We Spikes while the opp loses 12% to sandstorm — Spikes must NOT be credited with that loss.
    tl = _tl("spikes", "+0%", "meteormash", "+0%", ours="skarmory", opps="tyranitar",
             opp_after="88%")
    we = next(e for e in tl if e["move"] == "spikes")
    assert we["damage"] == "" and we["target"] == ""


def test_timeline_forced_switch_phase_is_just_the_send_in():
    tl = _tl("switched_to:celebi", "+0%", "none", "+0%", ours="skarmory", opps="metagross",
             phase="forced_switch")
    assert tl == [{"side": "we", "kind": "send_in", "sent_in": "celebi"}]


def test_timeline_standalone_faint_when_attacker_unknown():
    tl = _tl("hiddenpower", "-10%", "unknown", "+0%", ours="donphan", opps="skarmory",
             events=["our:donphan:fainted", "result:loss"])
    faint = [e for e in tl if e["kind"] == "faint"]
    assert faint == [{"side": "we", "kind": "faint", "actor": "donphan"}]


def test_timeline_order_certainty():
    # Both sides moved, move_order recorded → certain.
    certain = _tl("surf", "-18%", "earthquake", "-12%", ours="milotic", opps="swampert",
                  our_after="82%", opp_after="88%", move_order="opp_first")
    assert all(e["order_certain"] for e in certain if e["kind"] == "move")
    # Both moved, NO move_order (no-state / model-free) → uncertain.
    unknown = _tl("surf", "-18%", "earthquake", "-12%", ours="milotic", opps="swampert",
                  our_after="82%", opp_after="88%")
    moves = [e for e in unknown if e["kind"] == "move"]
    assert len(moves) == 2 and not any(e["order_certain"] for e in moves)
    # A couldn't-move turn: only one side truly moved → order is certain even without move_order.
    canted = _tl("earthquake", "+0%", "focuspunch", "-55%", ours="swampert", opps="tyranitar",
                 opp_after="45%", opp_cant="focuspunch")
    assert all(e["order_certain"] for e in canted if e["kind"] == "move")
    # A switch fixes order (resolves first) → certain without move_order.
    switched = _tl("switched_to:skarmory", "-25%", "meteormash", "+0%", ours="tyranitar",
                   opps="metagross", our_after="75%")
    assert all(e["order_certain"] for e in switched if e["kind"] == "move")


def test_timeline_switch_in_hit_shows_resulting_hp():
    # opp voluntarily switches to celebi; our rockslide hits the switch-IN. The recorded delta is 0
    # across the switch, but the next board's HP is truth → show the resulting HP, not a blank line.
    tl = _tl("rockslide", "+0%", "switched_to:celebi", "+0%", ours="aerodactyl", opps="salamence",
             opp_after="11%", our_crit=True)
    we = next(e for e in tl if e["kind"] == "move")
    assert we["resulting"] is True and we["target"] == "celebi" and we["hp_after"] == "11%"
    assert we["damage"] == "" and we["no_effect"] == ""


def test_timeline_no_effect_immune():
    # Seismic Toss (Normal, fixed-damage) vs a Ghost → immune; our gengar took nothing.
    tl = _tl("hypnosis", "+0%", "seismictoss", "+3%", ours="gengar", opps="registeel",
             opp_effectiveness="immune", move_order="opp_first")
    opp = next(e for e in tl if e["move"] == "seismictoss")
    assert opp["no_effect"] == "immune"


def test_timeline_no_effect_missed_status_move():
    # Hypnosis (60% acc) applied no SLP (no status event) → flagged 'missed', not a blank line.
    tl = _tl("hypnosis", "+0%", "seismictoss", "+3%", ours="gengar", opps="registeel",
             move_order="opp_first")
    we = next(e for e in tl if e["move"] == "hypnosis")
    assert we["no_effect"] == "missed"


def test_timeline_utility_move_not_flagged_no_effect():
    # Spikes sets a hazard (invisible in the outcome) — must NOT read "no effect"/"missed".
    tl = _tl("spikes", "+0%", "earthquake", "-20%", ours="skarmory", opps="tyranitar",
             our_after="80%", move_order="opp_first")
    we = next(e for e in tl if e["move"] == "spikes")
    assert we["no_effect"] == "" and we["damage"] == ""


# ── our-side Hidden Power typing (from the reconstruction record) ─────────────────────────────────
from main.prober.engine import build_our_hp_types, build_board   # noqa: E402


def test_build_our_hp_types_extracts_typed_hp():
    td = [{"species": "Forretress", "moves": ["spikes", "hiddenpowerbug", "counter"]},
          {"species": "Moltres", "moves": ["flamethrower", "hiddenpowergrass"]},
          {"species": "Jirachi", "moves": ["wish", "bodyslam"]}]           # no HP → absent
    assert build_our_hp_types(td) == {"forretress": "hiddenpower(bug)", "moltres": "hiddenpower(grass)"}
    assert build_our_hp_types(None) == {}


def _board_inv(moves, bench=""):
    actions = {f"switch:{i}": {"valid": True} for i in range(6)}
    for mv in moves:
        actions[mv] = {"valid": True}
    actions["struggle"] = {"valid": False}
    return {"actions": actions, "our": {"species": "forretress", "hp": "100%", "bench": bench},
            "opp": {"species": "tyranitar", "hp": "100%", "bench": ""}}


def test_build_board_types_our_active_hidden_power():
    inv = _board_inv(["hiddenpower", "spikes", "rapidspin", "counter"])
    b = build_board(inv, our_hp_types={"forretress": "hiddenpower(bug)"})
    assert "hiddenpower(bug)" in b.ours.moves and "hiddenpower" not in b.ours.moves


def test_build_board_types_our_bench_hidden_power():
    inv = _board_inv(["icebeam", "surf", "spikes", "rapidspin"], bench="moltres(50%)")
    team = {"moltres": {"item": "leftovers", "moves": ("hiddenpower", "flamethrower")}}
    b = build_board(inv, team=team, our_hp_types={"moltres": "hiddenpower(grass)"})
    moltres = next(m for m in b.ours.bench if m.species == "moltres")
    assert "hiddenpower(grass)" in moltres.moves and "hiddenpower" not in moltres.moves


def test_build_board_no_hp_map_leaves_bare():
    inv = _board_inv(["hiddenpower", "spikes", "rapidspin", "counter"])
    assert "hiddenpower" in build_board(inv).ours.moves        # no reconstruction → unchanged


def test_timeline_no_effect_uses_recorded_move_outcome():
    # gen3_move_outcome_v1 records each move's fate — prefer it over the accuracy guess.
    tl = _tl("hypnosis", "+0%", "seismictoss", "+3%", ours="gengar", opps="registeel",
             our_move_outcome="miss", opp_move_outcome="hit", opp_effectiveness="immune",
             move_order="opp_first")
    assert next(e for e in tl if e["move"] == "hypnosis")["no_effect"] == "missed"   # recorded miss
    assert next(e for e in tl if e["move"] == "seismictoss")["no_effect"] == "immune"  # hit-but-immune
    # a recorded 'fail' (e.g. a status move that fizzled) reads 'no effect', not 'missed'
    failed = _tl("toxic", "+0%", "spikes", "+0%", ours="a", opps="b",
                 our_move_outcome="fail", move_order="we_first")
    assert next(e for e in failed if e["move"] == "toxic")["no_effect"] == "failed"


# ── timeline_entry_text (the PLAIN-TEXT battle log the CLI + web render) ─────────────────────────
# The TUI paints these entries with Rich styles; the JSON CLI and the browser need the same words as
# text. Both readings come from one function so a new fact (a crit, a miss, a boost) cannot show up
# on one surface and be silently missing from the other.
from main.prober.engine import timeline_entry_text   # noqa: E402


def _texts(tl):
    return [timeline_entry_text(e) for e in tl]


def test_timeline_text_reads_as_a_battle_log():
    tl = _tl("brickbreak", "-73%", "rockslide", "-85%", our_after="27%", opp_after="15%",
             move_order="we_first")
    assert _texts(tl) == ["we brickbreak did 85%  (tyranitar 100% → 15%)",
                          "opp rockslide did 73%  (salamence 100% → 27%)"]


def test_timeline_text_covers_faint_switch_and_send_in():
    tl = _tl("icebeam", "-72%", "hiddenpower → metagross_sent_in", "-100%",
             ours="tyranitar", opps="salamence", events=["opp:salamence:fainted"],
             our_after="28%", opp_after="100%", move_order="we_first", opp_crit=True)
    text = _texts(tl)
    assert text[0] == "we icebeam did 100%  (salamence 100% → faint)"
    assert text[1] == "opp hiddenpower did 72%  (tyranitar 100% → 28%)  ⚡CRIT"
    assert text[2] == "opp sends in metagross"

    switched = _tl("switched_to:skarmory", "+0%", "spikes", "+0%", ours="tyranitar", opps="skarmory")
    assert "we switch tyranitar → skarmory" in _texts(switched)


def test_timeline_text_explains_a_move_that_did_nothing():
    """A blank line reads as missing data. Every no-visible-effect case must SAY why."""
    tl = _tl("hypnosis", "+0%", "seismictoss", "+3%", ours="gengar", opps="registeel",
             our_move_outcome="miss", opp_move_outcome="hit", opp_effectiveness="immune",
             move_order="we_first")
    assert _texts(tl) == ["we hypnosis — missed", "opp seismictoss — no effect (immune)"]


def test_timeline_text_spells_out_a_cant_reason_rather_than_its_code():
    tl = _tl("earthquake", "+0%", "rest", "+0%", ours="donphan", opps="snorlax", our_cant="slp")
    assert "we earthquake — couldn't move (asleep)" in _texts(tl)


def test_timeline_does_not_infer_a_miss_against_a_switch_in():
    """FOUND ON A REAL TRACE: our Meteor Mash into a Jirachi the opponent pivoted in read
    `— missed`, while the same battle's protocol log showed -resisted / -damage 84/100 / -boost.

    A voluntary switch makes the recorded hp_delta price the mon that LEFT, so it says nothing
    about the hit — and `_no_effect_reason`'s last resort (infer a miss from the move's accuracy)
    then converts "no evidence" into a confident wrong claim. Here the switch-in also healed to
    exactly 90% on Leftovers, so the resulting-HP branch missed its <90 threshold. No explanation
    is honest; a wrong one is worse than none.
    """
    tl = _tl("meteormash", "+0%", "switched_to:jirachi", "+0%",
             ours="metagross", opps="tyranitar", opp_after="90%")
    we = next(e for e in tl if e.get("move") == "meteormash")
    assert we["no_effect"] == "", "an accuracy guess must not become a stated miss"
    assert "missed" not in timeline_entry_text(we)

    # A RECORDED miss is still a fact and must survive.
    recorded = _tl("meteormash", "+0%", "switched_to:jirachi", "+0%", ours="metagross",
                   opps="tyranitar", opp_after="90%", our_move_outcome="miss")
    assert next(e for e in recorded if e.get("move") == "meteormash")["no_effect"] == "missed"

    # And with no switch involved, the accuracy fallback is still the right call.
    plain = _tl("meteormash", "+0%", "protect", "+0%", ours="metagross", opps="tyranitar")
    assert next(e for e in plain if e.get("move") == "meteormash")["no_effect"] == "missed"


def test_the_rendered_line_draws_its_words_from_the_engines_vocabulary():
    """The regression this guards: a renderer owning PRIVATE copies of the `cant` / `no effect`
    wordings, so teaching one surface a new reason leaves another printing a raw code.

    It used to assert this against the TUI's copies; the TUI is gone and the web renders
    `timeline_entry_text` directly, so the guard is now that the rendered SENTENCE is built from the
    shared tables — a new `cant` code added to `CANT_PHRASE` must appear in the text with no
    renderer change, and an unknown code must pass through rather than vanish."""
    from main.prober.engine import CANT_PHRASE, NO_EFFECT_TEXT, cant_phrase, timeline_entry_text
    assert cant_phrase("frz") == CANT_PHRASE["frz"] == "frozen"
    assert cant_phrase("something_new") == "something_new"      # unknown code passes through
    frozen = timeline_entry_text({"side": "we", "kind": "move", "move": "icebeam", "cant": "frz"})
    assert CANT_PHRASE["frz"] in frozen, "the line must read the shared table, not a private copy"
    immune = timeline_entry_text({"side": "we", "kind": "move", "move": "seismictoss",
                                  "no_effect": "immune"})
    assert NO_EFFECT_TEXT["immune"] in immune


# ── raw Showdown protocol (replay.html → per-turn slice) ──────────────────────────────────────────
from main.prober.engine import parse_protocol_log, protocol_for_turn   # noqa: E402


def test_parse_protocol_log_extracts_block():
    html = ('<html><body><script type="text/plain" class="battle-log-data">\n'
            '|teamsize|p1|6\n|start\n|switch|p1a: A|A|100/100\n|turn|1\n'
            '|move|p1a: A|Tackle|p2a: B\n|-damage|p2a: B|80/100\n|turn|2\n'
            '|move|p2a: B|Surf|p1a: A\n|-miss|p2a: B\n|turn|3\n|win|x\n</script></body></html>')
    lines = parse_protocol_log(html)
    assert "<html>" not in "".join(lines)                    # only the |-protocol, no HTML chrome
    assert "|turn|1" in lines and "|-miss|p2a: B" in lines


def test_protocol_for_turn_slices_by_turn_marker():
    lines = ("|start", "|switch|p1a: A|A|100/100", "|turn|1", "|move|p1a: A|Tackle|p2a: B",
             "|-damage|p2a: B|80/100", "|turn|2", "|move|p2a: B|Surf|p1a: A", "|-miss|p2a: B",
             "|turn|3", "|win|x")
    assert protocol_for_turn(lines, 1) == ("|turn|1", "|move|p1a: A|Tackle|p2a: B", "|-damage|p2a: B|80/100")
    t2 = protocol_for_turn(lines, 2)
    assert "|-miss|p2a: B" in t2 and "|turn|3" not in t2     # stops at the next turn marker
    assert protocol_for_turn((), 1) == ()


# ---------------------------------------------------------------------------
# build_value_dist — the distributional value head's per-decision distribution
# (v29). Pure: a synthetic npz dict + a (vmin, vmax, bins) support, no torch.
# ---------------------------------------------------------------------------

from main.prober.engine import build_value_dist  # noqa: E402


def test_build_value_dist_peaked_recovers_mean_and_low_entropy():
    bins = 8
    probs = np.zeros((2, bins), dtype=np.float32)
    probs[:, 5] = 1.0                                   # all mass on atom 5
    vd = build_value_dist({"value_dist": probs}, 0, (-4.0, 4.0, bins))
    assert vd is not None
    z = np.linspace(-4.0, 4.0, bins)
    assert abs(vd.mean - float(z[5])) < 1e-5
    assert vd.entropy < 0.1 and vd.std < 0.1            # near-one-hot ⇒ confident
    assert len(vd.probs) == bins and len(vd.support) == bins


def test_build_value_dist_absent_or_nan_is_none():
    # No array (old trace / head off) → the KeyError "unavailable" path.
    assert build_value_dist({}, 0, (-4.0, 4.0, 8)) is None
    # A captured-but-headless row is all-NaN → None (so the prober shows it only when real).
    nan = np.full((1, 8), np.nan, dtype=np.float32)
    assert build_value_dist({"value_dist": nan}, 0, (-4.0, 4.0, 8)) is None


def test_build_value_dist_bin_mismatch_is_none():
    """Support/trace bin counts disagree (config drift) → bail safely, don't mis-render."""
    probs = np.ones((1, 8), dtype=np.float32) / 8.0
    assert build_value_dist({"value_dist": probs}, 0, (-4.0, 4.0, 16)) is None


def test_build_value_dist_popart_denormalizes_mean():
    probs = np.zeros((1, 8), dtype=np.float32)
    probs[:, 4] = 1.0
    vd = build_value_dist({"value_dist": probs}, 0, (-4.0, 4.0, 8), popart=(10.0, 2.0))
    z = np.linspace(-4.0, 4.0, 8)
    assert vd.mean_real is not None
    assert abs(vd.mean_real - (float(z[4]) * 2.0 + 10.0)) < 1e-4


def test_build_value_dist_bimodal_flagged():
    """Two separated humps ⇒ high bimodality (mass outside the dominant peak's neighborhood)."""
    bins = 16
    probs = np.zeros((1, bins), dtype=np.float32)
    probs[0, 1] = 0.5
    probs[0, 14] = 0.5
    vd = build_value_dist({"value_dist": probs}, 0, (-4.0, 4.0, bins))
    assert vd.bimodality > 0.35


# ---------------------------------------------------------------------------
# GPU-obs observability — spread belief, refine rounds, belief trajectory
# (gen3_unified_spread_belief_v1 / gen3_iterative_damage_v1 / belief trajectory axis B)
# ---------------------------------------------------------------------------

def test_build_spread_belief_derived_stat_match():
    """A revealed mon's believed spread is matched to its TRUE mon by species and the true DERIVED stats
    are computed from team_details base+IV+EV+nature (the gen3 L100 formula). Tauros adamant 252 Atk."""
    from main.prober.engine import build_spread_belief
    raw = {
        "spread": np.array([[300, 200, 150, 180, 250],   # slot 0 = tauros (believed)
                            [1, 1, 1, 1, 1], [0] * 5, [0] * 5, [0] * 5, [0] * 5], dtype=float),
        "believed_mask": np.array([False, True, True, True, True, True]),   # only slot 0 revealed
        "opp_species": ["tauros", "", "", "", "", ""],
        "opp_active": 0,
    }
    details = [{"species": "tauros",
                "evs": {"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252},
                "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31},
                "nature": "adamant"}]
    sv = build_spread_belief(raw, details)
    assert sv is not None and sv.n_slots == 1
    slot = sv.slots[0]
    assert slot.species == "tauros" and slot.matched and slot.nature == "adamant"
    atk = next(r for r in slot.rows if r.stat == "atk")
    # Tauros base Atk 100, adamant (×1.1), 252 EVs: (2*100+31+63+5)*11//10 = 299*1.1 = 328.
    assert atk.true == 328.0 and atk.believed == 300.0 and abs(atk.err + 28.0) < 1e-6
    # Spe base 110 neutral 252: 2*110+31+63+5 = 319.
    spe = next(r for r in slot.rows if r.stat == "spe")
    assert spe.true == 319.0
    assert atk.prior is not None   # Smogon prior column populated for a species with usage data


def test_build_spread_belief_skips_hidden_and_handles_no_truth():
    """Hidden slots (no species) are skipped; with no team_details the believed column still renders but
    true/err/prior-from-truth are None (the websocket / no-ground-truth path)."""
    from main.prober.engine import build_spread_belief
    raw = {
        "spread": np.array([[300, 200, 150, 180, 250]] + [[1, 1, 1, 1, 1]] * 5, dtype=float),
        "believed_mask": np.array([False, True, True, True, True, True]),
        "opp_species": ["snorlax", "", "", "", "", ""],
        "opp_active": 0,
    }
    sv = build_spread_belief(raw, None)
    assert sv is not None and sv.n_slots == 1          # the 5 hidden slots are skipped
    assert sv.slots[0].rows[0].true is None and not sv.slots[0].matched
    assert sv.mean_abs_err is None                     # no truth → no error stat


def test_build_spread_belief_none_when_off():
    from main.prober.engine import build_spread_belief
    assert build_spread_belief(None, [{"species": "tauros"}]) is None


def test_build_belief_trajectory_scores_correctness_with_truth():
    """Axis B: per-decision top-1 confidence + ✓ correctness vs the privileged team, model-free from the
    summary belief blocks. With no opp_team correctness stays 0 but confidence is still populated."""
    from main.prober.engine import build_belief_trajectory
    summary = {"invocations": [
        {"turn": 1, "belief": [{"slot": 2, "top": [{"species": "snorlax", "prob": "40.0%"},
                                                    {"species": "tauros", "prob": "20.0%"}]},
                               {"slot": 3, "top": [{"species": "zapdos", "prob": "10.0%"}]}]},
        {"turn": 3, "belief": [{"slot": 2, "top": [{"species": "snorlax", "prob": "80.0%"}]}]},
        {"turn": 5},   # no belief block this decision
    ]}
    tj = build_belief_trajectory(summary, opp_team=("snorlax", "suicune"))
    assert tj is not None and len(tj.points) == 2
    assert tj.points[0].n_hidden == 2 and tj.points[0].n_correct == 1   # snorlax top-1 right, zapdos wrong
    assert tj.points[1].mean_top1_conf == 0.8
    # no-truth path: correctness collapses to 0, confidence preserved.
    tj2 = build_belief_trajectory(summary, opp_team=None)
    assert [p.n_correct for p in tj2.points] == [0, 0]
    assert build_belief_trajectory({"invocations": [{"turn": 1}]}, None) is None


def test_build_belief_trajectory_consumes_hidden_multiset_no_double_count():
    """The n_correct scoring must NOT double-count (the set-membership bug build_belief_truth avoids):
    two believed slots both top-1'ing the SAME single true species score 1, not 2; a top-1 naming a
    species NOT on the team scores 0. One-time consumption per still-hidden true mon."""
    from main.prober.engine import build_belief_trajectory
    summary = {"invocations": [
        # both hidden slots guess 'snorlax' top-1, but the team has exactly ONE snorlax → score 1, not 2.
        {"turn": 1, "belief": [{"slot": 2, "top": [{"species": "snorlax", "prob": "55.0%"}]},
                               {"slot": 3, "top": [{"species": "snorlax", "prob": "30.0%"}]}]},
        # a guess for a species not on the team → 0.
        {"turn": 3, "belief": [{"slot": 2, "top": [{"species": "mewtwo", "prob": "60.0%"}]}]},
    ]}
    tj = build_belief_trajectory(summary, opp_team=("snorlax", "suicune"))
    assert tj.points[0].n_hidden == 2 and tj.points[0].n_correct == 1   # consumed once, not double-counted
    assert tj.points[1].n_correct == 0                                  # mewtwo not on the team


def test_build_belief_trajectory_reads_npz_move_and_spread():
    """When the trace npz carries the captured move_logits / spread_belief (active-row) arrays, each point
    gets the opp-active move-belief entropy + believed Atk/Spe; absent/NaN rows leave them None."""
    from main.prober.engine import build_belief_trajectory
    summary = {"invocations": [
        {"turn": 1, "belief": [{"slot": 2, "top": [{"species": "snorlax", "prob": "40.0%"}]}]},
        {"turn": 3, "belief": [{"slot": 2, "top": [{"species": "snorlax", "prob": "80.0%"}]}]},
    ]}
    npz = {
        "move_logits": np.array([[2.0, 2.0, -2.0, -2.0], [5.0, 5.0, -5.0, -5.0]], dtype=np.float32),
        "spread_belief": np.array([[300, 200, 150, 180, 250], [310, 205, 150, 180, 255]], dtype=np.float32),
    }
    tj = build_belief_trajectory(summary, opp_team=("snorlax",), npz=npz)
    assert tj.points[0].move_entropy is not None and tj.points[1].move_entropy is not None
    assert tj.points[1].move_entropy < tj.points[0].move_entropy        # sharper logits → lower entropy
    assert tj.points[0].believed_atk == 300.0 and tj.points[0].believed_spe == 250.0
    # absent arrays → the move/spread fields stay None (older trace).
    tj_bare = build_belief_trajectory(summary, opp_team=("snorlax",), npz=None)
    assert tj_bare.points[0].move_entropy is None and tj_bare.points[0].believed_atk is None


# ── opponent full-team view (privileged truth + revealed-or-not tags) ─────────────────────────────
from main.prober.engine import build_opp_full_team, BoardView, SideBoard   # noqa: E402


def test_build_opp_full_team_tags_revealed_item_and_moves():
    details = [
        {"species": "Registeel", "item": "leftovers",
         "moves": ["seismictoss", "thunderwave", "toxic", "explosion"]},
        {"species": "Swampert", "item": "salacberry",
         "moves": ["surf", "earthquake", "icebeam", "protect"]},
    ]
    board = BoardView(                                  # registeel active, has shown thunderwave; no bench
        ours=SideBoard("zapdos", "100%", "", "", (), (), ""),   # SideBoard: …, moves, bench, item
        opp=SideBoard("registeel", "100%", "", "", ("thunderwave",), (), "leftovers"))
    by = {m.species: m for m in build_opp_full_team(details, board).mons}
    reg = by["Registeel"]
    assert reg.revealed and reg.active and reg.item_revealed
    assert dict(reg.moves)["thunderwave"] is True and dict(reg.moves)["seismictoss"] is False
    sw = by["Swampert"]                                 # never seen → everything unrevealed
    assert not (sw.revealed or sw.active or sw.item_revealed)
    assert all(not seen for _, seen in sw.moves)
    assert build_opp_full_team(None, board) is None     # no privileged team → revealed-only fallback


def test_build_opp_full_team_matches_typed_hidden_power():
    # the truth carries 'hiddenpowerfire'; a revealed bare 'hiddenpower' (or typed display) still matches
    details = [{"species": "Magneton", "item": "magnet", "moves": ["thunderbolt", "hiddenpowerfire"]}]
    board = BoardView(ours=SideBoard("a", "100%", "", "", (), ()),
                      opp=SideBoard("magneton", "100%", "", "", ("hiddenpower(fire)",), ()))
    mv = dict(build_opp_full_team(details, board).mons[0].moves)
    assert mv["hiddenpowerfire"] is True                # the typed HP is recognised as revealed


from main.prober.engine import build_switch_in_outgoing, SwitchInOutgoingView   # noqa: E402


def test_build_switch_in_outgoing_forced_switch():
    """Each ALIVE bench candidate → its best BP move vs the opp active (type-eff + KO + outspeed).
    Fainted + status-only candidates are excluded; the super-effective multiplier is faithful."""
    from types import SimpleNamespace
    board = SimpleNamespace(
        ours=SimpleNamespace(bench=[
            SimpleNamespace(species="celebi", hp="100%"),      # Grass/Psychic → Giga Drain 4× on Swampert
            SimpleNamespace(species="magneton", hp="faint"),   # fainted → excluded
        ]),
        opp=SimpleNamespace(active_species="swampert", active_hp="40%"),
    )
    our = [{"species": "celebi", "moves": ["gigadrain", "recover"],
            "evs": {"spa": 252}, "ivs": {}, "nature": "modest"}]
    opp = [{"species": "swampert", "moves": ["earthquake"],
            "evs": {"hp": 252, "def": 252}, "ivs": {}, "nature": "relaxed"}]
    v = build_switch_in_outgoing(board, our, opp)
    assert isinstance(v, SwitchInOutgoingView) and v.opp_species == "swampert"
    species = [r.species for r in v.rows]
    assert "celebi" in species and "magneton" not in species          # fainted excluded
    c = next(r for r in v.rows if r.species == "celebi")
    assert c.move == "gigadrain"             # recover (BP 0) excluded → Giga Drain is the only BP move
    assert c.type_mult == 4.0                # Grass 4× vs Water/Ground
    assert c.high > 0 and 0.0 <= c.pko <= 1.0 and 0.0 <= c.outspeed <= 1.0


def test_build_switch_in_outgoing_none_without_details():
    """No reconstruction (no team_details) or no opp active → None, never a crash."""
    from types import SimpleNamespace
    board = SimpleNamespace(ours=SimpleNamespace(bench=[]),
                            opp=SimpleNamespace(active_species="swampert", active_hp="100%"))
    assert build_switch_in_outgoing(board, None, None) is None
    assert build_switch_in_outgoing(board, [{"species": "celebi"}], []) is None

# -- move order read off the raw protocol log --------------------------------------------------
#
# The TurnDelta's `move_order` is decoded from the OBS, so it only exists on the model-loading
# path; measured over five runs, ZERO decisions carry one in a model-free `battle_turns` read. The
# replay therefore had to declare the order unknown whenever both sides moved — 56.3% of decisions
# with a timeline (measured on ai_v9_17_tdaux_lam3, 389 decisions). The protocol log states the
# execution order outright, and this is the fixture it states it in: the exact turn a reader
# reported as permanently "unordered".

_TURN_7 = (
    "|turn|7",
    "|",
    "|move|p1a: Jirachi|Protect|p1a: Jirachi",
    "|-singleturn|p1a: Jirachi|Protect",
    "|move|p2a: Raikou|Thunderbolt|p1a: Jirachi",
    "|-activate|p1a: Jirachi|Protect",
    "|",
    "|-heal|p1a: Jirachi|165/404|[from] item: Leftovers",
    "|-heal|p2a: Raikou|88/100 par|[from] item: Leftovers",
    "|upkeep",
)


def test_move_order_is_read_off_the_protocol_for_either_side():
    """The log lists `|move|` lines in EXECUTION order, so the answer is a reading, not a guess —
    and which of p1/p2 is 'us' is resolved by matching the actor's nickname to the active species,
    never by assuming we are p1. Same turn, both perspectives."""
    from main.prober.engine import move_order_from_protocol

    assert move_order_from_protocol(_TURN_7, "jirachi", "raikou") == "we_first"
    assert move_order_from_protocol(_TURN_7, "raikou", "jirachi") == "opp_first"
    # Display forms and ids both normalize.
    assert move_order_from_protocol(_TURN_7, "Jirachi", "Raikou") == "we_first"


def test_move_order_refuses_rather_than_guessing():
    """Every `None` here is a case where the log cannot settle it, and a 50/50 dressed up as a
    reading would be worse than the honest 'not recorded' the renderer already handles."""
    from main.prober.engine import move_order_from_protocol

    # A MIRROR: the nickname names both sides equally.
    assert move_order_from_protocol(_TURN_7, "jirachi", "jirachi") is None
    # A nicknamed mon matches neither active.
    nicked = ("|turn|7", "|move|p1a: Sparky|Thunderbolt|p2a: Jirachi")
    assert move_order_from_protocol(nicked, "raikou", "jirachi") is None
    # No move lines at all (a switch-only turn), and no protocol at all.
    assert move_order_from_protocol(("|turn|7", "|switch|p1a: Jirachi|Jirachi|404/404"),
                                    "jirachi", "raikou") is None
    assert move_order_from_protocol(None, "jirachi", "raikou") is None
    assert move_order_from_protocol((), "jirachi", "raikou") is None
    # A missing species on either side cannot be matched against.
    assert move_order_from_protocol(_TURN_7, "", "raikou") is None


def test_the_timeline_orders_by_the_protocol_when_the_turndelta_did_not_record_it():
    """The end-to-end effect: both sides moved, no recorded `move_order` — previously flagged
    `order_certain=False` and rendered without a sequence. The protocol settles it, and the SLOWER
    side's line now comes second."""
    from main.prober.engine import build_result_timeline, move_order_from_protocol

    outcome = {"our": {"action": "protect", "hp_delta": "0%"},
               "opp": {"action": "thunderbolt", "hp_delta": "0%"}}
    hint = move_order_from_protocol(_TURN_7, "jirachi", "raikou")

    without = build_result_timeline(outcome, "jirachi", "raikou", "move_selection")
    assert not all(e.get("order_certain", True) for e in without if e["kind"] == "move"), (
        "fixture must be a genuinely ambiguous turn for this test to mean anything")

    with_hint = build_result_timeline(outcome, "jirachi", "raikou", "move_selection",
                                      move_order_hint=hint)
    assert all(e.get("order_certain", True) for e in with_hint if e["kind"] == "move")
    assert [e["side"] for e in with_hint if e["kind"] == "move"] == ["we", "opp"]

    # …and the other way round, so the assertion above is about the DATA and not the default.
    flipped = build_result_timeline(
        {"our": {"action": "thunderbolt", "hp_delta": "0%"},
         "opp": {"action": "protect", "hp_delta": "0%"}},
        "raikou", "jirachi", "move_selection",
        move_order_hint=move_order_from_protocol(_TURN_7, "raikou", "jirachi"))
    assert [e["side"] for e in flipped if e["kind"] == "move"] == ["opp", "we"]


def test_a_recorded_move_order_still_wins_over_the_protocol():
    """The TurnDelta value folds from the real event-log sequence the analysis was built on. The
    protocol read is a FALLBACK for when there is none — it must not override the record."""
    from main.prober.engine import build_result_timeline

    outcome = {"our": {"action": "protect", "hp_delta": "0%"},
               "opp": {"action": "thunderbolt", "hp_delta": "0%"},
               "move_order": "opp_first"}
    entries = build_result_timeline(outcome, "jirachi", "raikou", "move_selection",
                                    move_order_hint="we_first")
    assert [e["side"] for e in entries if e["kind"] == "move"] == ["opp", "we"]


def test_a_voluntary_switch_still_outranks_both():
    """A switch resolves before a move, which is mechanics rather than a recording — neither the
    record nor the protocol hint may reorder that."""
    from main.prober.engine import build_result_timeline

    outcome = {"our": {"action": "thunderbolt", "hp_delta": "0%"},
               "opp": {"action": "switched_to:blissey", "hp_delta": "0%"}}
    entries = build_result_timeline(outcome, "raikou", "jirachi", "move_selection",
                                    move_order_hint="we_first")
    assert entries[0]["side"] == "opp", "the opponent's voluntary switch must resolve first"


# ---------------------------------------------------------------------------
# The SPECIES-CLAUSE reading of the belief (`build_exclusive_belief`)
# ---------------------------------------------------------------------------

def test_exclusive_belief_is_none_when_no_slot_is_hidden():
    """Same "nothing to show" contract as `belief_view_from_logits`, so the raw and the adjusted
    blocks appear and disappear together instead of one surviving the other."""
    from main.prober.engine import build_exclusive_belief
    assert build_exclusive_belief(np.zeros((6, 10)), np.zeros(6, bool), [], maps=_maps10()) is None


def test_exclusive_belief_reports_a_COHERENT_belief_as_coherent_and_changes_nothing():
    """Two hidden slots confidently naming DIFFERENT mons: no clause is violated, so the adjusted
    top-1s equal the raw ones, every row moved zero, and `coherent` is True — which is what lets a
    surface print one line instead of an identical second table."""
    from main.prober.engine import build_exclusive_belief
    logits = np.full((6, 10), -8.0)
    logits[1, 4] = 6.0
    logits[2, 7] = 6.0
    mask = np.array([False, True, True, False, False, False])
    v = build_exclusive_belief(logits, mask, [], maps=_maps10())
    assert v.coherent
    assert v.duplicate_top1 == 0
    assert v.max_expected_count <= 1.0
    assert v.illegal_mass == 0.0
    assert [s.adj_top1 for s in v.slots] == ["sp4", "sp7"]
    assert not any(s.differs or s.hypothesis_differs for s in v.slots)
    assert all(s.total_variation == 0.0 for s in v.slots)
    assert v.iterations == 0


def test_exclusive_belief_flags_a_DUPLICATE_top1_even_when_the_expected_count_is_LEGAL():
    """The defect a reader actually sees, and the one this whole view is for. Two hidden slots both
    naming sp4 at ~0.38 give an expected count of ~0.76 — no clause forbids it — while the panel
    still reads "both hidden mons are sp4", which no gen3 team can be. `duplicate_top1` must fire
    and the point hypothesis must name two different mons."""
    from main.prober.engine import build_exclusive_belief
    logits = np.full((6, 10), -8.0)
    logits[1, 4] = 1.0; logits[1, 5] = 0.9; logits[1, 6] = 0.7
    logits[2, 4] = 1.0; logits[2, 6] = 0.9; logits[2, 7] = 0.7
    mask = np.array([False, True, True, False, False, False])
    v = build_exclusive_belief(logits, mask, [], maps=_maps10())
    assert v.max_expected_count <= 1.0          # the DISTRIBUTION is legal…
    assert v.duplicate_top1 == 1                # …but the DISPLAY is not
    assert not v.coherent
    assert len(set(v.team_hypothesis)) == 2
    assert sum(s.hypothesis_differs for s in v.slots) == 1


def test_exclusive_belief_caps_an_over_full_species_and_keeps_rows_normalized():
    from main.prober.engine import build_exclusive_belief
    logits = np.full((6, 10), -8.0)
    for slot in (1, 2, 3):
        logits[slot, 4] = 2.0; logits[slot, 5] = 1.0
    mask = np.array([False, True, True, True, False, False])
    v = build_exclusive_belief(logits, mask, [], maps=_maps10())
    assert v.max_expected_count > 1.0           # the RAW headline, pre-adjustment
    assert v.illegal_mass > 0.0
    assert v.converged
    assert sum(p for _, p in v.slots[0].top) <= 1.0 + 1e-9
    # Every slot's adjusted sp4 mass is below its raw mass — the constraint only ever removes.
    for s in v.slots:
        adj4 = dict(s.top).get("sp4", 0.0)
        assert adj4 < s.raw_top1_prob


def test_exclusive_belief_measures_and_removes_mass_on_a_REVEALED_species():
    """Stage-0's defect, kept distinct from joint incoherence: a hidden bench slot cannot be the mon
    standing on the field. `revealed_leak_max` REPORTS it on the raw belief; the adjusted rows carry
    none of it."""
    from main.prober.engine import build_exclusive_belief
    logits = np.full((6, 10), -8.0)
    logits[1, 3] = 3.0        # sp3 is REVEALED — this is the leak
    logits[1, 7] = 2.0
    mask = np.array([False, True, False, False, False, False])
    v = build_exclusive_belief(logits, mask, ["sp3"], maps=_maps10())
    assert v.revealed_leak_max > 0.5
    assert not v.coherent
    assert "sp3" not in dict(v.slots[0].top)
    assert v.slots[0].raw_top1 == "sp3"          # the model's OWN read is preserved verbatim…
    assert v.slots[0].adj_top1 == "sp7"          # …beside the clause-consistent one
    assert v.slots[0].differs
    assert v.revealed == ("sp3",)


def test_exclusive_belief_ignores_a_revealed_species_outside_the_vocab():
    """An unrecognised name is DROPPED, not guessed at — a wrong num would zero an innocent
    species' column, which is worse than one missing constraint."""
    from main.prober.engine import build_exclusive_belief
    logits = np.full((6, 10), -8.0)
    logits[1, 3] = 3.0
    mask = np.array([False, True, False, False, False, False])
    v = build_exclusive_belief(logits, mask, ["not-a-real-mon"], maps=_maps10())
    assert v.revealed_leak_max == 0.0
    assert v.slots[0].adj_top1 == "sp3"


def test_exclusive_belief_slot_indices_are_the_real_opp_slots():
    """Rows are hidden slots only, so a view labelling them 0..H−1 would name the wrong mon."""
    from main.prober.engine import build_exclusive_belief
    logits = np.full((6, 10), -8.0)
    logits[3, 4] = 5.0; logits[5, 7] = 5.0
    mask = np.array([False, False, False, True, False, True])
    v = build_exclusive_belief(logits, mask, [], maps=_maps10())
    assert [s.slot for s in v.slots] == [3, 5]

# -- opponent ordering: sentinels first, strongest first ---------------------------------------

def test_sentinel_zero_is_the_STRONGEST_and_sorts_first():
    """The index is a STRENGTH RANK, not a creation order — `sentinel_0` is the best self in the
    pool and the labels float, so a promotion re-seats every sentinel. Getting this backwards is
    the whole risk of the feature: it would put the WEAKEST opponent at the top of the list and
    read as if it were the most informative."""
    from main.prober.engine import opponent_rank, sort_opponents

    assert opponent_rank("sentinel_0") < opponent_rank("sentinel_1")
    assert opponent_rank("sentinel_1") < opponent_rank("sentinel_4")
    # …and every sentinel outranks every bot.
    assert opponent_rank("sentinel_9") < opponent_rank("aggressive")

    got = sort_opponents(["random", "sentinel_2", "heuristic", "sentinel_0", "sentinel_1"])
    assert got[:3] == ["sentinel_0", "sentinel_1", "sentinel_2"]


def test_sentinels_sort_NUMERICALLY_not_lexically():
    """`sentinel_10` after `sentinel_9`, which a plain string sort gets wrong — and a pool that
    grows past ten is the only place the bug would ever show."""
    from main.prober.engine import sort_opponents

    assert sort_opponents(["sentinel_10", "sentinel_2", "sentinel_9"]) == [
        "sentinel_2", "sentinel_9", "sentinel_10"]


def test_non_sentinels_keep_the_order_they_arrived_in():
    """The key answers "which of these matters most", and inventing a strength order for the fixed
    bots is a different claim that the ELO ladder owns. So they are moved as a block, not sorted."""
    from main.prober.engine import sort_opponents

    assert sort_opponents(["zeta", "alpha", "sentinel_0", "mid"]) == [
        "sentinel_0", "zeta", "alpha", "mid"]
    # A name that merely looks sentinel-ish is not one.
    assert sort_opponents(["sentinel_x", "sentinel_0"]) == ["sentinel_0", "sentinel_x"]
    assert sort_opponents([]) == []


# -- a move that never executed --------------------------------------------------------------
#
# The recorded outcome says what a side CHOSE. Nothing in a model-free trace says whether the
# choice ever ran, so the timeline assumed it did and explained the missing damage as an
# ineffective move — a claim about the MOVE on a turn where the move never happened. Both fixtures
# below are the real protocol of gen-16 `loss_s0_004`, which is where it was reported.

_TURN_6_KO_BEFORE_MOVING = (
    "|turn|6",
    "|move|p2a: Tyranitar|Rock Slide|p1a: Forretress",
    "|-damage|p1a: Forretress|0 fnt",
    "|faint|p1a: Forretress",
    "|switch|p1a: Swampert|Swampert, F|401/401",
    "|upkeep",
)

_TURN_5_FROZEN = (
    "|turn|5",
    "|move|p2a: Tyranitar|Dragon Dance|p2a: Tyranitar",
    "|-boost|p2a: Tyranitar|atk|1",
    "|cant|p1a: Forretress|frz",
    "|upkeep",
)


def _outcome_both_moved(our_move, opp_move, events=()):
    return {"our": {"action": our_move, "hp_delta": "-100%"},
            "opp": {"action": opp_move, "hp_delta": "0%"},
            "events": list(events)}


def test_a_mon_KOd_before_it_moved_says_so_instead_of_no_effect():
    """THE REPORTED BUG. Forretress chose Explosion, was outsped by a +2 Tyranitar and killed; the
    protocol carries no `|move|` line for our side at all, and the timeline read
    `we explosion — no effect` — which describes a move that ran and whiffed."""
    from main.prober.engine import (build_result_timeline, protocol_action_fate,
                                    timeline_entry_text)

    fate = protocol_action_fate(_TURN_6_KO_BEFORE_MOVING, "forretress", "tyranitar")
    assert fate == "absent", "the log has no move line for our side; that is the whole signal"

    entries = build_result_timeline(
        _outcome_both_moved("explosion", "rockslide", ["our:forretress:fainted"]),
        "forretress", "tyranitar", "move_selection", our_fate_hint=fate)
    ours = [e for e in entries if e["side"] == "we" and e["kind"] == "move"]
    assert ours and ours[0]["never_moved"] is True
    assert timeline_entry_text(ours[0]) == "we explosion — never moved (fainted first)"
    assert "no effect" not in timeline_entry_text(ours[0])


def test_a_blocked_move_names_the_reason_instead_of_no_effect():
    """The `|cant|` fall-through. A model-free trace has no decoded TurnDelta, so `our_cant` is
    empty and a FROZEN Forretress read `— no effect` too (an 843-battle sweep saw a full-para Surf
    do the same). The reason is right there in the log."""
    from main.prober.engine import (build_result_timeline, protocol_action_fate,
                                    timeline_entry_text)

    fate = protocol_action_fate(_TURN_5_FROZEN, "forretress", "tyranitar")
    assert fate == "cant:frz"

    entries = build_result_timeline(
        _outcome_both_moved("explosion", "dragondance"),
        "forretress", "tyranitar", "move_selection", our_fate_hint=fate)
    ours = [e for e in entries if e["side"] == "we" and e["kind"] == "move"]
    assert ours and ours[0]["cant"] == "frz"
    assert timeline_entry_text(ours[0]) == "we explosion — couldn't move (frozen)"

    # …and the same for the reason that motivated the sweep.
    para = ("|turn|9", "|move|p2a: Tyranitar|Rock Slide|p1a: Suicune",
            "|cant|p1a: Suicune|par")
    assert protocol_action_fate(para, "suicune", "tyranitar") == "cant:par"


def test_a_move_that_DID_execute_is_untouched():
    """The guard on the two above: the hint must not rewrite an ordinary turn."""
    from main.prober.engine import build_result_timeline, protocol_action_fate

    moved = ("|turn|4", "|move|p1a: Forretress|Spikes|p2a: Tyranitar",
             "|move|p2a: Tyranitar|Rock Slide|p1a: Forretress")
    assert protocol_action_fate(moved, "forretress", "tyranitar") == "moved"
    entries = build_result_timeline(
        {"our": {"action": "spikes", "hp_delta": "0%"},
         "opp": {"action": "rockslide", "hp_delta": "0%"}},
        "forretress", "tyranitar", "move_selection", our_fate_hint="moved")
    ours = [e for e in entries if e["side"] == "we" and e["kind"] == "move"]
    assert ours and not ours[0]["never_moved"] and not ours[0]["cant"]


def test_never_moved_is_claimed_ONLY_when_the_mon_actually_fainted():
    """"No move line for this side" has other explanations — a partial slice, a turn that could not
    be aligned. Without a faint the honest answer is to say nothing, not to invent a reason."""
    from main.prober.engine import build_result_timeline

    entries = build_result_timeline(
        _outcome_both_moved("explosion", "rockslide"),        # NO faint event
        "forretress", "tyranitar", "move_selection", our_fate_hint="absent")
    ours = [e for e in entries if e["side"] == "we" and e["kind"] == "move"]
    assert ours and ours[0]["never_moved"] is False


def test_protocol_action_fate_refuses_rather_than_guessing():
    """Every `None` is a case the log cannot settle, and a wrong explanation is worse than none."""
    from main.prober.engine import protocol_action_fate

    # A MIRROR: the nickname names both sides equally.
    assert protocol_action_fate(_TURN_6_KO_BEFORE_MOVING, "forretress", "forretress") is None
    # A nickname matching NEITHER active — `absent` needs the other side positively identified,
    # which is the guard against this pool's localized-species nicknames.
    nicked = ("|turn|6", "|move|p2a: Sparky|Thunderbolt|p1a: Rocky")
    assert protocol_action_fate(nicked, "forretress", "tyranitar") is None
    # No protocol at all, and a missing species.
    assert protocol_action_fate(None, "forretress", "tyranitar") is None
    assert protocol_action_fate((), "forretress", "tyranitar") is None
    assert protocol_action_fate(_TURN_6_KO_BEFORE_MOVING, "", "tyranitar") is None


def test_the_protocols_own_cant_spellings_read_as_english():
    """The reason now arrives straight off the log, so the map has to speak the log's dialect. The
    live set over a run's replays is par/slp/frz/flinch/`Focus Punch`/`move: Taunt`/recharge."""
    from main.prober.engine import cant_phrase

    assert cant_phrase("Focus Punch") == "lost its focus"
    assert cant_phrase("move: Taunt") == "taunted"
    assert cant_phrase("par") == "fully paralyzed"
    assert cant_phrase("recharge") == "recharging"
    assert cant_phrase("something new") == "something new"   # never dropped


_TURN_7_IMMUNE = (
    "|turn|7",
    "|switch|p2a: Gengar|Gengar, M|100/100",
    "|move|p1a: Swampert|Earthquake|p2a: Gengar",
    "|-immune|p2a: Gengar",
    "|-weather|Sandstorm|[upkeep]",
    "|upkeep",
)


def test_an_immunity_in_the_log_is_named_instead_of_a_bare_no_effect():
    """`_no_effect_reason` can already say "immune", but it keys on the TurnDelta's decoded
    effectiveness — which a MODEL-FREE trace does not have, so a Levitate Gengar eating an
    Earthquake read `— no effect`. Across 100 battles, 67 of 400 bare no-effect lines sat on a turn
    whose log carried an `|-immune|`."""
    from main.prober.engine import (build_result_timeline, protocol_move_result,
                                    timeline_entry_text)

    assert protocol_move_result(_TURN_7_IMMUNE, "swampert", "gengar") == "immune"
    entries = build_result_timeline(
        {"our": {"action": "earthquake", "hp_delta": "0%"},
         "opp": {"action": "switched_to:gengar", "hp_delta": "0%"}},
        "swampert", "tyranitar", "move_selection",
        our_result_hint=protocol_move_result(_TURN_7_IMMUNE, "swampert", "gengar"))
    ours = [e for e in entries if e["side"] == "we" and e["kind"] == "move"]
    assert ours and ours[0]["no_effect"] == "immune"
    assert timeline_entry_text(ours[0]).endswith("— no effect (immune)")


def test_a_miss_is_attributed_to_the_ATTACKER_that_missed():
    """`|-miss|` names the attacker FIRST, so the other side's miss on the same turn must not be
    borrowed — and the window stops at the next `|move|` so neither can an immunity."""
    from main.prober.engine import protocol_move_result

    both = ("|turn|3",
            "|move|p1a: Zapdos|Thunder|p2a: Swampert",
            "|-miss|p1a: Zapdos|p2a: Swampert",
            "|move|p2a: Swampert|Earthquake|p1a: Zapdos",
            "|-immune|p1a: Zapdos")
    assert protocol_move_result(both, "zapdos", "swampert") == "missed"
    assert protocol_move_result(both, "swampert", "zapdos") == "immune"


def test_move_result_refuses_rather_than_guessing():
    from main.prober.engine import protocol_move_result

    assert protocol_move_result(_TURN_7_IMMUNE, "gengar", "gengar") is None      # mirror
    assert protocol_move_result(None, "swampert", "gengar") is None
    assert protocol_move_result(_TURN_7_IMMUNE, "", "gengar") is None
    # An actor that never appears in the slice: nothing to attribute.
    assert protocol_move_result(_TURN_7_IMMUNE, "blissey", "gengar") is None
    # A clean hit says nothing — silence, not "failed".
    hit = ("|turn|2", "|move|p1a: Zapdos|Thunderbolt|p2a: Swampert",
           "|-damage|p2a: Swampert|180/401")
    assert protocol_move_result(hit, "zapdos", "swampert") is None


def test_a_RECORDED_outcome_still_beats_the_log():
    """The TurnDelta is what the analysis was built on; the protocol only fills what it did not
    decode. A recorded `miss` must not be overwritten by an immunity elsewhere in the window."""
    from main.prober.engine import build_result_timeline

    entries = build_result_timeline(
        {"our": {"action": "earthquake", "hp_delta": "0%"},
         "opp": {"action": "protect", "hp_delta": "0%"},
         "our_move_outcome": "miss"},
        "swampert", "gengar", "move_selection", our_result_hint="immune")
    ours = [e for e in entries if e["side"] == "we" and e["kind"] == "move"]
    assert ours and ours[0]["no_effect"] == "missed"
