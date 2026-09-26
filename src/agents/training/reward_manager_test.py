"""`Gen3RewardManager` — the TERMINAL-only reward (`gen3_shaped_reward_deletion_v1`).

The shaped reward path was deleted on 2026-09-26 (program_rust_core §4 M3 row). What is pinned here:

* **THE TERMINAL** — the indicator (production) and the signed terminal, at both magnitudes, with
  the pre-cap tie and the 250-turn timeout that each have their own trap;
* **NOTHING BUT THE TERMINAL** — a non-terminal turn pays exactly 0.0 whatever the board does;
* **THE `win_margin` OBS KEY** — the material margin still published every turn (it is an
  observation feature, not a reward term), with its formula pinned;
* **THE `reward/` EXPORT and the episode counters** — the live export's GIGO guard still reads 0.

The byte-level before/after proof lives in `reward_golden_test.py` (recorded at the last
pre-deletion commit). Pure (no battle sim) — shared fakes in `reward_test_fakes.py`.
"""
import numpy as np
import pytest

from agents.training.material_margin import MAT_ALIVE_WEIGHT, MAT_HP_WEIGHT, material_margin
from agents.training.reward_manager import (
    Gen3RewardManager, RewardConfig, VICTORY_VALUE, _TIMEOUT_TURN_CAP, format_reward_composition,
    inert_reward_flags,
)
from agents.training.reward_terminal_test_support import OUTCOMES, terminal_reward
from agents.training.reward_test_fakes import _Battle, _delta, _full_team_live, _Live

_PRODUCTION = RewardConfig(terminal_indicator=True, victory_value=1.0, draw_penalty=0.0)


# ─── 1. THE TERMINAL ────────────────────────────────────────────────────────────────────────────

def test_the_production_terminal_is_the_WIN_INDICATOR():
    """+1 on a win, 0.0 on a loss, a pre-cap tie AND a 250-turn timeout alike."""
    got = {o: terminal_reward(_PRODUCTION, o) for o in OUTCOMES}
    assert got == {"win": 1.0, "loss": 0.0, "tie": 0.0, "timeout": 0.0}


@pytest.mark.parametrize("victory", [30.0, 1.0])
def test_the_SIGNED_terminal_scores_win_loss_and_the_pre_cap_tie_by_victory_value(victory):
    cfg = RewardConfig(victory_value=victory)
    assert terminal_reward(cfg, "win") == victory
    assert terminal_reward(cfg, "loss") == -victory
    # The pre-cap tie shares the decisive-loss branch (it is NOT the timeout).
    assert terminal_reward(cfg, "tie") == -victory


def test_a_TIMEOUT_takes_draw_penalty_under_the_signed_terminal():
    assert terminal_reward(RewardConfig(victory_value=1.0, draw_penalty=-1.2), "timeout") == -1.2
    assert terminal_reward(RewardConfig(), "timeout") == -35.0


def test_the_default_terminal_is_exactly_plus_or_minus_thirty():
    assert RewardConfig().victory_value == VICTORY_VALUE == 30.0
    assert terminal_reward(RewardConfig(), "win") == 30.0
    assert terminal_reward(RewardConfig(), "loss") == -30.0


def test_the_timeout_is_detected_by_TURN_COUNT_not_by_won_lost():
    """A forfeit-loss one turn before the cap is a decisive loss, at the cap a timeout."""
    cfg = RewardConfig(victory_value=1.0, draw_penalty=-2.0)
    for turn, want in ((_TIMEOUT_TURN_CAP - 1, -1.0), (_TIMEOUT_TURN_CAP, -2.0)):
        m = Gen3RewardManager(config=cfg)
        live = _full_team_live(our_alive=2, opp_alive=2, lost=True, finished=True)
        assert m.process_turn_reward(_Battle(live, turn=turn), _delta()) == want


# ─── 2. NOTHING BUT THE TERMINAL ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cfg", [RewardConfig(), _PRODUCTION], ids=["signed", "indicator"])
def test_a_non_terminal_turn_pays_EXACTLY_zero_whatever_the_board_does(cfg):
    """The shaped potentials and BIAS terms are gone: a KO, a switch, a spike layer, a boost — none
    of it moves the reward. (Under the deleted path each of these emitted a term.)"""
    m = Gen3RewardManager(config=cfg)
    ctx = type("Ctx", (), {"phase": "move"})()
    boards = [
        (_full_team_live(), _delta()),
        (_full_team_live(opp_alive=5), _delta(opp_fainted=True, our_move_id="earthquake")),
        (_full_team_live(our_alive=5, our_hp=0.3), _delta(we_fainted=True)),
        (_full_team_live(), _delta(our_switch_to="skarmory")),
        (_full_team_live(), _delta(our_move_id="spikes",
                                   our_boost_delta=np.ones(7, dtype=np.int8))),
    ]
    for i, (live, delta) in enumerate(boards):
        m.record_action(ctx, 6 if i != 3 else 1)
        assert m.process_turn_reward(_Battle(live, turn=i + 1), delta) == 0.0
        assert m._last_breakdown.to_dict() == {"total": 0.0}
    assert m.total_reward == 0.0


def test_the_composition_line_is_ONE_TERMINAL():
    line = format_reward_composition(_PRODUCTION)
    assert line == ("[Reward] composition: 1 TERMINAL + 0 PBRS + 0 BIAS "
                    "(none — fully policy-invariant)")
    assert inert_reward_flags(_PRODUCTION) == ["draw_penalty"]
    assert inert_reward_flags(RewardConfig()) == []


# ─── 3. THE `win_margin` OBS KEY ────────────────────────────────────────────────────────────────

def test_the_material_margin_formula():
    """Declared-team material, normalised: 2·ΔHP + 1.25·Δalive over a ±6·(2+1.25) bound."""
    bound = (MAT_HP_WEIGHT + MAT_ALIVE_WEIGHT) * 6
    assert material_margin(_full_team_live()) == 0.0
    assert material_margin(_full_team_live(opp_alive=4)) == pytest.approx(
        (MAT_HP_WEIGHT * 2 + MAT_ALIVE_WEIGHT * 2) / bound)
    assert material_margin(_full_team_live(our_hp=0.5)) == pytest.approx(-MAT_HP_WEIGHT * 3 / bound)
    # Unrevealed opp mons count as full-HP-alive: 2 revealed at full HP + 4 unrevealed == even.
    assert material_margin(_Live([1.0] * 6, [1.0, 1.0])) == 0.0
    assert material_margin(_Live([], [1.0])) == 0.0          # no known mons → neutral


def test_the_manager_publishes_the_margin_EVERY_turn_under_the_production_reward():
    """The margin is an OBSERVATION feature and must not depend on the reward composition — it once
    did (pinned at 0.0 for the win-prob arm's whole life), which is why it is pinned here too."""
    m = Gen3RewardManager(config=_PRODUCTION)
    m.process_turn_reward(_Battle(_full_team_live(opp_alive=2), turn=3), _delta())
    assert m._last_material_margin == pytest.approx(material_margin(_full_team_live(opp_alive=2)))
    assert m._last_material_margin > 0.5
    m.reset()
    assert m._last_material_margin == 0.0


# ─── 4. THE EXPORT AND THE COUNTERS ─────────────────────────────────────────────────────────────

def test_the_reward_export_tracks_the_terminal_and_its_residual_reads_zero():
    m = Gen3RewardManager(config=_PRODUCTION)
    m.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
    m.process_turn_reward(_Battle(_full_team_live(opp_alive=0, won=True, finished=True), turn=2),
                          _delta(opp_fainted=True))
    out = m.drain_reward_terms()
    assert out["n"] == 2 and out["total_sum"] == 1.0
    assert out["residual_abs_sum"] == 0.0          # the GIGO guard: census == fold
    assert set(out["sum"]) == {"win_loss"}
    assert m.drain_reward_terms()["n"] == 0        # drained


def test_the_episode_counters_count_what_happened():
    m = Gen3RewardManager()
    move = type("Ctx", (), {"phase": "move"})()
    forced = type("Ctx", (), {"phase": "forced_switch"})()
    m.record_action(move, 7)
    m.process_turn_reward(_Battle(_full_team_live()), _delta())
    m.record_action(move, 10)                      # struggle
    m.process_turn_reward(_Battle(_full_team_live()), _delta())
    m.record_action(move, 2)                       # voluntary switch that EXECUTED
    m.process_turn_reward(_Battle(_full_team_live()), _delta(our_switch_to="b"))
    m.record_action(move, 3)                       # pressed, but never executed (gap=0)
    m.process_turn_reward(_Battle(_full_team_live()), _delta())
    m.record_action(forced, 4)
    m.process_turn_reward(_Battle(_full_team_live()), _delta())
    assert (m.attack_count, m.struggle_turns, m.switch_count, m.forced_switch_count) == (2, 1, 1, 1)
    m.reset()
    assert (m.attack_count, m.struggle_turns, m.switch_count, m.forced_switch_count) == (0, 0, 0, 0)
