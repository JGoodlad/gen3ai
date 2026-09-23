"""`gen3_curated_sample_split_v1` (2026-09-23) — WHICH teams fill the default trainee builder's 10%
bias draw, on each side of the train / measure line.

* TRAINING (`build_matchup_and_opponents`) biases toward Smogon's 32 CURATED sample teams — a
  TRAINING-INPUT CHANGE: before the split it biased toward all 72 teams then in `data/teams/sample/`
  (the 32 + the 40 promoted exploiter trainees).
* MEASUREMENT (the eval worker, and through `measurement_bias_teams` the snapshot ladder and the bot
  calibration/matrix) stays on the FROZEN pre-split 72, so `eval/elo`, `win_rate_vs_pool`,
  `ladder.json` and the bot anchors have no team-regime boundary at the split, and the ladder may
  still reuse eval-measured pairs (both play the same builder).
"""
from types import SimpleNamespace

import pytest

from utils.team_loader import TeamLoader, pins
from utils.teambuilder import Gen3Teambuilder


def _args():
    return SimpleNamespace(
        trainee_team=None, trainee_teams=None, exploiter=None, self_play=False, bot_weights=None,
        exploiter_keep_bots=False, exploiter_bot_fraction=0.5, exploiter_temp_start=None,
        exploiter_temp_mode="fixed", stable_opponent_temp=1.0, _distill_pairs=None,
        allow_nonsample_trainee=False, allow_untaught_teacher=False, team_pfsp="off",
        team_pfsp_cap=3.0, team_pfsp_floor=0.05, team_block_episodes=1)


class _Built(Exception):
    def __init__(self, tb):
        self.tb = tb


def _packed(teams):
    return sorted(Gen3Teambuilder(list(teams)).packed_teams)


def test_training_biases_toward_the_32_curated_teams(monkeypatch):
    """The REAL training construction, stopped right after the trainee builder is built."""
    from main.train import matchup_setup

    def stop(args, all_teams, tb):
        raise _Built(tb)
    monkeypatch.setattr(matchup_setup, "apply_distill_team_bias", stop)
    with pytest.raises(_Built) as got:
        matchup_setup.build_matchup_and_opponents(_args())
    tb = got.value.tb
    loader = TeamLoader()
    assert len(loader.get_sample_teams()) == 32
    assert sorted(tb.bias_packed_teams) == _packed(loader.get_sample_teams())
    # exactly 32 — the pre-split builder biased toward 72 (curated + the 40 promoted). (Compared by
    # PACK, one promoted team — dbbfac7bd5, "Blue Offense by Noitulover" — is indistinguishable
    # from curated 01cb64e16c: the same team, only its paste text differs.)
    assert len(tb.bias_packed_teams) == 32
    assert tb.bias_prob == pytest.approx(0.1)
    assert len(tb.packed_teams) == 719


def test_eval_keeps_the_frozen_pre_split_bias_set(monkeypatch):
    """The eval worker's REAL `_run`, stopped at the trainee builder: its bias set is the frozen
    pre-split 72 — NOT `get_sample_teams()` (32 since the split) — so eval series stay one regime."""
    from main import eval_worker
    captured = {}

    def capture(cfg, all_teams, bias_teams):
        captured["bias"] = list(bias_teams)
        raise _Built(None)
    monkeypatch.setattr(eval_worker, "load_mappings", lambda: {})
    monkeypatch.setattr(eval_worker, "_build_trainee_tb", capture)
    with pytest.raises(_Built):
        eval_worker._run({})
    assert [pins.team_sha(t) for t in captured["bias"]] == list(pins.PRE_SPLIT_SAMPLE_72)
    assert len(TeamLoader().get_sample_teams()) == 32      # i.e. NOT the curated set


def test_the_frozen_measurement_set_differs_from_training_only_in_the_bias_draw():
    loader = TeamLoader()
    frozen = {pins.team_sha(t) for t in pins.measurement_bias_teams(loader)}
    curated = {pins.team_sha(t) for t in loader.get_training_bias_teams()}
    promoted = {pins.team_sha(t) for t in loader.get_promoted_teams()}
    superseded = {pins.team_sha(t) for t in loader.get_superseded_teams()}
    # frozen = curated − the new Curse RestLax paste + the old one + the 40 promoted
    assert frozen == (curated - {"1808014a9a"}) | superseded | promoted
