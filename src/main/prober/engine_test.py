"""Pure unit tests for the probe engine — no torch, no checkpoint.

A FakeProbeModel stands in for the torch boundary so the whole analysis is
exercised deterministically. A separate regression test pins the resolved obs
offsets against the live encoder layout, so a silent obs-layout shift fails loud.
"""

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


def _npz(has_state=1, value=1.5):
    obs = np.zeros((1, _OBS_LEN), dtype=np.float32)
    # stored /4-normalised → *4 gives [2.0, 1.0, 0.0, 0.5]
    obs[0, _OFF.mm_off:_OFF.mm_off + 4] = [0.5, 0.25, 0.0, 0.125]
    return {"obs": obs, "has_state": np.array([has_state], dtype=np.int8),
            "values": np.array([value], dtype=np.float32)}


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


def test_matchups_read_correct_dims():
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(), _npz(), 0)
    assert a.matchups.multipliers == (2.0, 1.0, 0.0, 0.5)
    assert a.matchups.move_labels == ("thunderbolt", "earthquake", "move2", "move3")


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
    assert names == [
        "active move_multipliers(4)", "our_matchups(144)", "their_matchups(144)",
        "our active pokemon block(99)", "turn-history block",
    ]
    # |grad| = [0,1,2,...]; active-pokemon block spans 0:5 → mean 2.0, sum 10.
    active_block = a.saliency.blocks[3]
    assert active_block.mean_abs == 2.0 and active_block.total_abs == 10.0
    assert a.saliency.overall_mean_abs == np.arange(_OBS_LEN).mean()


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
                "bench": "metagross(100%), celebi(50%), snorlax(faint)"},
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
    assert b.opp.active_species == "zapdos" and b.opp.moves == ()
    assert b.opp.bench[0].fainted is True


def test_analysis_carries_board():
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(), _npz(), 0)
    assert a.board is not None and a.board.ours.active_species == "Zapdos"


def test_analysis_carries_field_when_model_decodes():
    model = FakeProbeModel(_OFF)
    a = analyze_invocation(model, _summary(), _npz(), 0)
    assert a.field == {"weather": "RAIN", "our_spikes": 1, "opp_spikes": 0, "turn": 8.0}


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
    # OFFSET_REACTIVE resolves to 1418 at runtime (root CLAUDE.md obs table;
    # the inline "# 1247" comments in observation/constants.py are stale).
    assert off.mm_off == 1422   # OFFSET_REACTIVE(1418) + move_multiplier(4) — unchanged (before vec[14])
    assert off.om_off == 1502   # OFFSET_REACTIVE(1418) + our_matchups(84, post gen3_markovian_progress_v1)
    assert off.tm_off == 1646   # OFFSET_REACTIVE(1418) + their_matchups(228 = our_matchups 84 + 144)
    assert off.active_block_dim == 99
    # incoming-damage / OHKO belief block: reactive offset 51 (post turns_since_progress vec[14]) → 1469.
    assert off.incoming_off == 1469   # OFFSET_REACTIVE(1418) + incoming_damage(51)
    assert off.incoming_dim == 33     # 6*5 per-mon + 3 recovery
    assert off.incoming_per_mon == 5 and off.incoming_recovery == 3
    assert off.pokemon_full_dim == 107

    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    lay = Gen3ObservationEncoder(load_mappings()).get_layout()
    assert off.turn_history_offset == lay["turn_history_offset"]
    assert off.turn_history_dim == lay["n_history_turns"] * lay["turn_delta_dim"]


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


def test_incoming_belief_none_when_block_absent():
    """The default _OFF has incoming_dim=0 → no belief decoded, no incoming saliency block, no crash.
    value_saliency is still None there because the default FakeProbeModel path is unaffected."""
    model = FakeProbeModel(_OFF)
    npz = {"obs": np.zeros((1, _OBS_LEN), dtype=np.float32),
           "has_state": np.array([1], dtype=np.int8), "values": np.array([1.0], dtype=np.float32)}
    a = analyze_invocation(model, _summary(), npz, 0)
    assert a.incoming is None
    assert not any(b.name.startswith("incoming_damage") for b in a.saliency.blocks)
