"""Unit tests for SelfPlayCallback.

Covers the pure helpers (_monotonicity_score, _check_bot_regression) and the
NON-BLOCKING subprocess lifecycle (launch → poll → collect → promote / best /
drain), mirroring the bot-eval orchestrator tests in eval_callback_test.py with a
fake work-stealing worker that handles BOTH the bot roster and pool sentinels.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.training.snapshot_pool import SnapshotEntry
from agents.training.selfplay_callback import (
    _monotonicity_score,
    SelfPlayCallback,
    _REGRESSION_WARN_THRESHOLD,
)


# ── _monotonicity_score ──────────────────────────────────────────────────────

def test_monotonicity_single_entry():
    assert _monotonicity_score([0.7]) == pytest.approx(1.0)


def test_monotonicity_empty():
    assert _monotonicity_score([]) == pytest.approx(1.0)


def test_monotonicity_perfectly_monotone():
    # index 0 = most recent (hardest) → lowest win rate; rises to oldest (easiest)
    assert _monotonicity_score([0.4, 0.5, 0.6, 0.7, 0.8]) == pytest.approx(1.0)


def test_monotonicity_perfectly_inverted():
    assert _monotonicity_score([0.8, 0.7, 0.6, 0.5, 0.4]) == pytest.approx(-1.0)


def test_monotonicity_two_entries_ordered():
    assert _monotonicity_score([0.4, 0.8]) == pytest.approx(1.0)


def test_monotonicity_two_entries_inverted():
    assert _monotonicity_score([0.8, 0.4]) == pytest.approx(-1.0)


def test_monotonicity_ties_count_as_concordant():
    assert _monotonicity_score([0.5, 0.5, 0.5]) == pytest.approx(1.0)


def test_monotonicity_mixed():
    # [0.4, 0.8, 0.6]: 2 concordant / 3 total → τ = 2*(2/3) - 1 = 1/3
    assert _monotonicity_score([0.4, 0.8, 0.6]) == pytest.approx(1 / 3, abs=1e-6)


# ── builders ─────────────────────────────────────────────────────────────────

def _mock_pool(n_sentinels=3):
    """A MagicMock SnapshotPool with the surface SelfPlayCallback touches."""
    pool = MagicMock()
    pool.load_persisted_win_rate.return_value = 0.0
    pool.is_empty.return_value = False
    pool.__len__.return_value = n_sentinels
    entries = [
        SnapshotEntry(path=Path(f"/snap/snapshot_{s * 1_000_000:012d}.zip"), step=s * 1_000_000)
        for s in range(n_sentinels)
    ]
    # sentinel_entries returns newest-first in the real pool; order only matters for
    # monotonicity sign, which these tests don't assert on.
    pool.sentinel_entries.return_value = list(reversed(entries))
    pool.entry_weight.return_value = 1.0
    return pool


def _make_callback(tmp_path, *, pool=None, promote_threshold=0.65,
                   best_dir=None, n_sentinels=3, debug=False):
    cb = SelfPlayCallback(
        pool=pool or _mock_pool(n_sentinels),
        model_dir=str(tmp_path),
        server_config=MagicMock(),
        showdown_port=9999,
        best_model_save_path=best_dir,
        promote_threshold=promote_threshold,
        n_workers=3,
        debug=debug,
    )
    cb.model = MagicMock()
    cb.model.save = lambda base: open(base + ".zip", "w").close()
    cb._logger = MagicMock()
    # `training_env` is a read-only property reading model.get_env(); configure the env
    # there. opponent_default_stats telemetry: (decisions, defaults, redecides) per env.
    cb.model.get_env.return_value.env_method.return_value = [(100, 5, 2)]
    cb.num_timesteps = 2_000_000  # first eval boundary is now 2M
    return cb


def _fake_selfplay_popen(ec, bot_win=0.8, sentinel_win=0.7):
    """Fake Popen whose 'worker' work-steals BOTH bots and sentinels from the claim dir
    and writes per-item result files, exactly like the real eval_worker."""
    class _FakeProc:
        returncode = 0
        def poll(self):
            return 0
        def wait(self, timeout=None):
            return 0

    def fake_popen(argv, stdout=None, stderr=None, env=None):
        import json as _json
        cfg = _json.load(open(argv[-1]))
        sentinel_labels = {s["label"] for s in cfg.get("sentinels", [])}
        universe = list(cfg["opponent_pool"]) + list(sentinel_labels)
        while True:
            name = ec.claim_next_opponent(cfg["claim_dir"], universe)
            if name is None:
                break
            win = sentinel_win if name in sentinel_labels else bot_win
            out = os.path.join(cfg["result_dir"], f"result__{name}.json")
            with open(out, "w") as f:
                _json.dump({"win_rate": win, "reward_mean": 1.0,
                            "ep_len": 20.0, "duration_sec": 5.0}, f)
        return _FakeProc()
    return fake_popen


# ── _check_bot_regression (edge-triggered warn) ───────────────────────────────

def test_regression_no_warning_before_threshold(tmp_path):
    cb = _make_callback(tmp_path)
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"Heuristic": 0.40})
        mock_emit.assert_not_called()


def test_regression_peak_recorded(tmp_path):
    cb = _make_callback(tmp_path)
    cb._check_bot_regression({"Heuristic": 0.75})
    assert cb._bot_peak["Heuristic"] == pytest.approx(0.75)


def test_regression_warning_fires_when_drop_below_threshold(tmp_path):
    cb = _make_callback(tmp_path)
    cb._check_bot_regression({"Heuristic": 0.72})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"Heuristic": 0.55})
        assert mock_emit.call_count == 1
        assert "BOT_REGRESSION" in mock_emit.call_args[0][0]


def test_regression_no_warning_if_still_above_threshold(tmp_path):
    cb = _make_callback(tmp_path)
    cb._check_bot_regression({"Heuristic": 0.80})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"Heuristic": 0.65})
        mock_emit.assert_not_called()


def test_regression_only_fires_for_named_bots(tmp_path):
    cb = _make_callback(tmp_path)
    cb._check_bot_regression({"Random": 0.90})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"Random": 0.10})
        mock_emit.assert_not_called()


def test_regression_fires_only_once_while_regressed(tmp_path):
    cb = _make_callback(tmp_path)
    cb._check_bot_regression({"Heuristic": 0.75})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"Heuristic": 0.50})
        cb._check_bot_regression({"Heuristic": 0.45})
        cb._check_bot_regression({"Heuristic": 0.40})
        assert mock_emit.call_count == 1


def test_regression_re_arms_after_recovery(tmp_path):
    cb = _make_callback(tmp_path)
    cb._check_bot_regression({"Heuristic": 0.75})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"Heuristic": 0.50})  # fires
        cb._check_bot_regression({"Heuristic": 0.70})  # recovered → re-arm
        cb._check_bot_regression({"Heuristic": 0.45})  # fires again
        assert mock_emit.call_count == 2


# ── schedule / trigger ─────────────────────────────────────────────────────────

def test_schedule_uses_shared_function(tmp_path):
    cb = _make_callback(tmp_path)
    cb.num_timesteps = 5_000_000
    assert cb._schedule() == (2_000_000, 100)
    cb.num_timesteps = 25_000_000
    assert cb._schedule() == (3_500_000, 300)


def test_schedule_debug_fast_cadence(tmp_path):
    cb = _make_callback(tmp_path, debug=True)
    cb.num_timesteps = 8_000
    assert cb._schedule() == (4_000, 3)


def test_no_eval_at_step_zero(tmp_path):
    cb = _make_callback(tmp_path)
    cb._init_callback()
    cb.num_timesteps = 0
    with patch.object(cb, "_launch_eval") as mock_launch:
        cb._on_step()
        mock_launch.assert_not_called()


def test_triggers_at_freq_boundary(tmp_path):
    cb = _make_callback(tmp_path)
    cb._init_callback()
    cb.num_timesteps = 2_000_000  # first eval boundary is now 2M
    with patch.object(cb, "_launch_eval") as mock_launch:
        cb._on_step()
        mock_launch.assert_called_once()
    assert cb._last_eval_step == 2_000_000


def test_skips_launch_while_previous_eval_running(tmp_path):
    cb = _make_callback(tmp_path)
    cb._init_callback()
    cb.num_timesteps = 2_000_000
    cb._pending = {"step": 1_000_000,
                   "procs": [{"proc": MagicMock(**{"poll.return_value": None})}]}
    with patch.object(cb, "_launch_eval") as mock_launch:
        cb._on_step()
        mock_launch.assert_not_called()  # previous cycle still running → skip
    assert cb._last_eval_step == 2_000_000  # boundary consumed


# ── full lifecycle: launch → collect → promote / best ─────────────────────────

def test_lifecycle_collect_records_promotes_and_saves_best(tmp_path, monkeypatch):
    from agents.training import eval_callback as ec
    best_dir = tmp_path / "best"
    pool = _mock_pool(n_sentinels=3)
    cb = _make_callback(tmp_path, pool=pool, best_dir=str(best_dir),
                        promote_threshold=0.65)
    cb._init_callback()
    # sentinel win 0.70 > 0.65 threshold → promotion; bot win 0.80 → best model.
    monkeypatch.setattr(ec.subprocess, "Popen",
                        _fake_selfplay_popen(ec, bot_win=0.8, sentinel_win=0.7))

    cb._on_step()                              # boundary → spawn workers (non-blocking)
    assert cb._pending is not None and cb._pending["step"] == 2_000_000
    snapshot = cb._pending["snapshot"]

    cb.num_timesteps = 2_000_001
    cb._on_step()                              # all done → merge + record + promote + best

    assert cb._pending is None
    # Promotion: the FROZEN snapshot (not the live model) is added at the trigger step.
    pool.add_from_path.assert_called_once()
    assert pool.add_from_path.call_args[0][0] == snapshot
    assert pool.add_from_path.call_args[0][1] == 2_000_000
    # Best model is the COPIED frozen snapshot.
    assert (best_dir / "best_model.zip").exists()
    assert cb._best_aggregate_win_rate == pytest.approx(0.8)
    # win_rate_vs_bots persisted for the next run's heuristic_fraction.
    pool.persist_win_rate.assert_called_with(pytest.approx(0.8))

    recorded = {c.args[0] for c in cb.logger.record.call_args_list}
    assert "eval/win_rate_vs_pool" in recorded
    assert "eval/sentinel_monotonicity" in recorded
    assert "train/selfplay_fraction" in recorded
    assert "eval/win_rate_vs_bots" in recorded
    assert "eval/win_rate_vs_Heuristic" in recorded
    assert "eval/win_rate_vs_sentinel_0" in recorded
    assert "train/selfplay_promoted_steps" in recorded
    # opponent default-rate telemetry recorded from the (paused) training env.
    assert "train/selfplay_opp_redecide_rate" in recorded


def test_lifecycle_no_promotion_below_threshold(tmp_path, monkeypatch):
    from agents.training import eval_callback as ec
    pool = _mock_pool(n_sentinels=3)
    cb = _make_callback(tmp_path, pool=pool, promote_threshold=0.65)
    cb._init_callback()
    monkeypatch.setattr(ec.subprocess, "Popen",
                        _fake_selfplay_popen(ec, bot_win=0.8, sentinel_win=0.50))

    cb._on_step()
    cb.num_timesteps = 2_000_001
    cb._on_step()

    assert cb._pending is None
    pool.add_from_path.assert_not_called()      # 0.50 ≤ 0.65 → no promotion
    recorded = {c.args[0] for c in cb.logger.record.call_args_list}
    assert "train/selfplay_promoted_steps" not in recorded
    assert "eval/win_rate_vs_pool" in recorded   # still recorded


def test_lifecycle_no_sentinels_handled(tmp_path, monkeypatch):
    """Pool with only the (newly seeded) step-0 entry still evals bots; pool win=0 → no promote."""
    from agents.training import eval_callback as ec
    pool = _mock_pool(n_sentinels=1)
    cb = _make_callback(tmp_path, pool=pool, promote_threshold=0.65)
    cb._init_callback()
    monkeypatch.setattr(ec.subprocess, "Popen",
                        _fake_selfplay_popen(ec, bot_win=0.9, sentinel_win=0.4))

    cb._on_step()
    cb.num_timesteps = 2_000_001
    cb._on_step()

    assert cb._pending is None
    recorded = {c.args[0] for c in cb.logger.record.call_args_list}
    assert "eval/win_rate_vs_bots" in recorded
    pool.add_from_path.assert_not_called()


def test_lifecycle_worker_failure_logs_and_continues(tmp_path, monkeypatch):
    from agents.training import eval_callback as ec
    pool = _mock_pool(n_sentinels=3)
    cb = _make_callback(tmp_path, pool=pool, best_dir=str(tmp_path / "best"))
    cb._init_callback()

    class _FailProc:
        returncode = 1
        def poll(self):
            return 1
        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(ec.subprocess, "Popen", lambda *a, **k: _FailProc())  # writes nothing

    cb._on_step()
    cb.num_timesteps = 2_000_001
    cb._on_step()                               # all workers failed → no record, no crash

    assert cb._pending is None
    pool.add_from_path.assert_not_called()
    assert cb._best_aggregate_win_rate == -1.0
    assert not cb.logger.dump.called


def test_drain_waits_for_inflight_then_collects(tmp_path, monkeypatch):
    from agents.training import eval_callback as ec
    pool = _mock_pool(n_sentinels=3)
    cb = _make_callback(tmp_path, pool=pool)
    cb._init_callback()

    waited = {"n": 0}

    class _SlowProc:
        returncode = 0
        def poll(self):
            return None                          # looks alive until wait() is called
        def wait(self, timeout=None):
            waited["n"] += 1
            return 0

    def fake_popen(argv, stdout=None, stderr=None, env=None):
        import json as _json
        cfg = _json.load(open(argv[-1]))
        sentinel_labels = {s["label"] for s in cfg.get("sentinels", [])}
        for nm in list(cfg["opponent_pool"]) + list(sentinel_labels):
            with open(os.path.join(cfg["result_dir"], f"result__{nm}.json"), "w") as f:
                _json.dump({"win_rate": 0.5, "reward_mean": 0.0,
                            "ep_len": 10.0, "duration_sec": 1.0}, f)
        return _SlowProc()

    monkeypatch.setattr(ec.subprocess, "Popen", fake_popen)

    cb._on_step()                                # launch; poll()=None → stays pending
    assert cb._pending is not None
    cb.drain(timeout=5)                          # graceful shutdown blocks then collects
    assert cb._pending is None
    assert waited["n"] >= 1
    assert cb.logger.record.called


def test_init_seeds_empty_pool(tmp_path):
    pool = _mock_pool()
    pool.is_empty.return_value = True
    cb = _make_callback(tmp_path, pool=pool)
    cb._init_callback()
    pool.seed.assert_called_once_with(cb.model)
