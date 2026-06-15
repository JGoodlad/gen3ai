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
    assert off.om_off == 1520   # OFFSET_REACTIVE(1418) + matchup_offset(102 = scalar 15 + move_eff 36 + incoming 51)
    assert off.tm_off == 1664   # om_off + our_matchups(144)
    assert off.active_block_dim == 99
    # incoming-damage / OHKO belief block: reactive offset 51 (post turns_since_progress vec[14]) → 1469.
    assert off.incoming_off == 1469   # OFFSET_REACTIVE(1418) + incoming_damage offset(51 = scalar 15 + move_eff 36)
    assert off.incoming_dim == 51     # gen3_incoming_crit_split: 6*8 per-mon + 3 recovery
    assert off.incoming_per_mon == 8 and off.incoming_recovery == 3
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


def test_taxonomy_critic_blindspot_vs_positional_grind_split_on_v():
    # No death; the split is purely the critic's pre-cliff value sign (scale-invariant).
    assert _cat(faint=False, v_at=8.0) == "critic_blindspot"      # thought it was WINNING
    assert _cat(faint=False, v_at=-8.0) == "positional_grind"     # already knew it was losing
    assert _cat(faint=False, v_at=None) == "positional_grind"     # unknown V → not a blindspot claim


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
