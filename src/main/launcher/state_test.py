"""Unit tests for LauncherState.update_metrics — eval-section replace semantics."""

from main.launcher.state import LauncherState


def _state():
    return LauncherState(interval_hours=3.0)


def test_eval_push_replaces_section_dropping_stale_rows():
    s = _state()
    # Cycle 1: 5 sentinels + an Aggressive bot.
    s.update_metrics({
        "eval/win_rate_mean": 0.70,
        "eval/win_rate_vs_aggressive": 0.80,
        "eval/win_rate_vs_sentinel_0": 0.60,
        "eval/win_rate_vs_sentinel_4": 0.70,
        "eval/sentinel_step_4": 1_000_000.0,
        "_step": 1000,
    })
    m1 = s.snapshot().metrics
    assert "eval/win_rate_vs_sentinel_4" in m1 and "eval/win_rate_vs_aggressive" in m1

    # Cycle 2: only 3 sentinels, and Aggressive's result went missing this cycle.
    s.update_metrics({
        "eval/win_rate_mean": 0.72,
        "eval/win_rate_vs_sentinel_0": 0.65,
        "eval/win_rate_vs_sentinel_2": 0.71,
        "_step": 2000,
    })
    m2 = s.snapshot().metrics
    assert "eval/win_rate_vs_sentinel_4" not in m2     # evicted slot dropped
    assert "eval/sentinel_step_4" not in m2            # its step label dropped too
    assert "eval/win_rate_vs_aggressive" not in m2     # missing-result opponent dropped
    assert m2["eval/win_rate_vs_sentinel_0"] == 0.65   # surviving row refreshed


def test_training_metrics_survive_an_eval_push():
    s = _state()
    s.update_metrics({"rollout/ep_rew_mean": 5.0, "time/fps": 700.0, "_step": 1000})
    s.update_metrics({"eval/win_rate_mean": 0.7, "_step": 1000})  # eval push
    m = s.snapshot().metrics
    assert m["rollout/ep_rew_mean"] == 5.0 and m["time/fps"] == 700.0   # not eval/* → kept
    assert m["eval/win_rate_mean"] == 0.7


def test_training_push_does_not_clear_eval_rows():
    s = _state()
    s.update_metrics({"eval/win_rate_mean": 0.7, "eval/win_rate_vs_random": 0.9, "_step": 1000})
    s.update_metrics({"time/fps": 720.0, "_step": 1100})  # training push, no eval/* keys
    m = s.snapshot().metrics
    assert m["eval/win_rate_vs_random"] == 0.9   # eval persists across training pushes
    assert m["time/fps"] == 720.0


def test_eval_metrics_ts_set_only_on_eval_push():
    s = _state()
    s.update_metrics({"time/fps": 700.0, "_step": 10})
    assert s.snapshot().eval_metrics_ts is None
    s.update_metrics({"eval/win_rate_mean": 0.7, "_step": 20})
    assert s.snapshot().eval_metrics_ts is not None


def test_selfplay_train_metrics_in_eval_push_are_kept():
    # The eval push also carries train/selfplay_* keys — those are NOT eval/* and must
    # survive the eval-section replace.
    s = _state()
    s.update_metrics({
        "eval/win_rate_mean": 0.7,
        "train/selfplay_fraction": 0.5,
        "_step": 1000,
    })
    m = s.snapshot().metrics
    assert m["train/selfplay_fraction"] == 0.5
