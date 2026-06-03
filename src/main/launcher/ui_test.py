"""Unit tests for the launcher TUI eval table (pure rendering — no Textual app)."""

from main.launcher.ui import LauncherUI


def _eval_table(metrics, eval_summary=None):
    # _make_eval_table uses no instance state, so a __new__'d shell is enough to call it.
    ui = LauncherUI.__new__(LauncherUI)
    table = ui._make_eval_table(list(metrics), metrics, eval_summary or {}, "")
    labels = [str(c) for c in table.columns[0]._cells]
    win_rates = [str(c) for c in table.columns[1]._cells]
    rewards = [str(c) for c in table.columns[2]._cells]
    return labels, win_rates, rewards


def test_sentinel_row_label_shows_checkpoint_step():
    labels, _wr, _rw = _eval_table({
        "eval/win_rate_vs_sentinel_0": 0.63,
        "eval/sentinel_step_0": 46_963_120.0,
    })
    # Capitalised to match the bot rows (Random/SetupSweep/…), with the checkpoint step.
    assert any("Sentinel_0 (47.0M)" in lab for lab in labels), labels


def test_sentinel_reward_column_populated():
    _labels, _wr, rewards = _eval_table({
        "eval/win_rate_vs_sentinel_0": 0.63,
        "eval/mean_reward_vs_sentinel_0": 7.7,
        "eval/sentinel_step_0": 46_963_120.0,
    })
    assert any("7.7" in r for r in rewards), rewards


def test_sentinel_step_key_does_not_create_a_phantom_row():
    # eval/sentinel_step_<i> is metadata for the label, not an opponent — it must not
    # render as its own "vs sentinel_step_0" row.
    labels, _wr, _rw = _eval_table({
        "eval/win_rate_vs_sentinel_0": 0.63,
        "eval/sentinel_step_0": 46_963_120.0,
    })
    assert not any("step" in lab.lower() for lab in labels), labels


def test_sentinel_label_falls_back_without_step():
    # No step metric yet (e.g. a resumed eval re-published from metadata) → plain label.
    labels, _wr, _rw = _eval_table({"eval/win_rate_vs_sentinel_0": 0.63})
    assert any(lab.strip() == "vs Sentinel_0" for lab in labels), labels
