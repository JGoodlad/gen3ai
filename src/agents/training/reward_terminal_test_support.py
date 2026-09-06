"""ONE terminal fold, by outcome NAME — the shared helper for terminal-shape tests.

`gen3_winprob_critic_mode_v1`. The four terminal outcomes this reward distinguishes are not four
flags on a config; they are four BOARDS, and each one has a trap that a hand-built case gets
wrong in a way no assertion catches:

* a **timeout** is a FORFEIT-loss at the turn cap (`lost=True`, `turn >= _TIMEOUT_TURN_CAP`), not
  a tie — `gen3_env` issues a `ForfeitBattleOrder` there, so the terminal detects it by TURN COUNT
  and a `finished`-but-not-`lost` board at turn 250 is a different case entirely;
* a **tie** is `finished` with neither `won` nor `lost`, *before* the cap — it shares the decisive
  loss's branch, which is exactly the conflation `--victory-value` had to be threaded through;
* a **decisive loss** has to be well before the cap or it reads as the timeout;
* and every one of them needs a prior non-terminal turn first, or `_prev_phi_mat` is unset and the
  material PBRS folds its first-window special case into the number under test.

Writing those four boards inline is what makes a terminal test a test of the harness. This module
is the one place they live, so `reward_end_state_test` (the ±30 / −35 ordering) and
`critic_mode_test` (the win INDICATOR) fold the SAME boards and any disagreement between them is
about the config, never about the setup.

Pure test support — imported by tests only, no production consumer.
"""
from __future__ import annotations

from agents.training.reward_manager import Gen3RewardManager
from agents.training.reward_test_fakes import _Battle, _delta, _full_team_live
from agents.training.reward_weights import _TIMEOUT_TURN_CAP

#: The four terminal boards, as `(kwargs for `_full_team_live`, turn, delta kwargs)`.
_BOARDS = {
    "win":     (dict(our_alive=3, opp_alive=0, won=True, finished=True), 40, dict(opp_fainted=True)),
    "loss":    (dict(our_alive=0, opp_alive=3, lost=True, finished=True), 40, dict(we_fainted=True)),
    # A pre-cap TIE: finished, nobody won or lost. It shares the decisive loss's branch.
    "tie":     (dict(our_alive=2, opp_alive=2, finished=True), 40, {}),
    # The 250-turn stall: a forfeit-LOSS detected by the turn count, not by won/lost.
    "timeout": (dict(our_alive=2, opp_alive=2, lost=True, finished=True), _TIMEOUT_TURN_CAP, {}),
}

OUTCOMES = tuple(_BOARDS)


def terminal_reward(config, outcome: str) -> float:
    """The `win_loss` TERMINAL term a fresh manager folds for `outcome` under `config`.

    Returns the TERMINAL field alone, not the episode's total: the shaping classes are a separate
    question with their own tests, and a total would make a terminal assertion depend on whichever
    potentials the config happens to leave on.
    """
    if outcome not in _BOARDS:
        raise KeyError(f"unknown terminal outcome {outcome!r} (want one of {OUTCOMES})")
    live_kw, turn, delta_kw = _BOARDS[outcome]
    mgr = Gen3RewardManager(config=config)
    # Seed a non-terminal turn first so `_prev_phi_mat` is set — otherwise the material PBRS's
    # first-window branch lands inside the same fold as the terminal being measured.
    mgr.process_turn_reward(_Battle(_full_team_live(), turn=1), _delta())
    mgr.process_turn_reward(_Battle(_full_team_live(**live_kw), turn=turn), _delta(**delta_kw))
    return float(mgr._last_breakdown.win_loss)
