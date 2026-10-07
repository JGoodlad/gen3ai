"""Unit tests for SelfPlayCallback.

Covers the pure helpers (_monotonicity_score, _check_bot_regression) and the
NON-BLOCKING subprocess lifecycle (launch → poll → collect → promote / best /
drain), mirroring the bot-eval orchestrator tests in eval_callback_test.py with a
fake work-stealing worker that handles BOTH the bot roster and pool sentinels.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.training import eval_callback as ec
from agents.training.snapshot_pool import SnapshotEntry
from agents.training.selfplay_callback import (
    _monotonicity_score,
    SelfPlayCallback,
    _MASTERY_CONFIRM_CYCLES,
)


@pytest.fixture(autouse=True)
def _isolate_force_eval_event():
    """The forced-eval request is a module-global Event; clear it around every test so a
    request set by one test can never leak into the next (they share one process)."""
    ec._force_eval_event.clear()
    yield
    ec._force_eval_event.clear()


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
        best_model_save_path=best_dir,
        promote_threshold=promote_threshold,
        n_sentinels=n_sentinels,
        debug=debug,
    )
    cb.model = MagicMock()
    cb.model.save = lambda base: open(base + ".zip", "w").close()
    cb._logger = MagicMock()
    # `training_env` is a read-only property reading model.get_env(); configure the env
    # there. opponent_default_stats telemetry: (decisions, defaults, redecides) per env.
    cb.model.get_env.return_value.env_method.return_value = [(100, 5, 2)]
    cb.model.gamma = 0.99
    cb.num_timesteps = 2_000_000  # first eval boundary is now 2M
    return cb


class _Stats:
    def __init__(self, n):
        self.n = n

    def as_dict(self):
        return {"games": self.n, "units": self.n, "host_steps": 1, "trainee_decisions": 10 * self.n,
                "p2_policy_decisions": 0, "traces": 0, "near_ties": 0,
                "seconds": {k: 0.0 for k in ("load", "stage", "submit", "drain", "act", "core", "finish", "trace",
                                             "total")}, "lifecycle": {}}


class _FakeEvaluator:
    """What ``RustEvalCore`` does for the callback: plays the cycle's plan IN PROCESS and publishes one raw
    ShardResult per unit (win rate keyed by kind). ``kinds`` restricts which kinds get a result — units of
    other kinds are left unreported, simulating a cycle that measured no sentinel. ``fail`` raises the
    eval core's own cycle error (a QUARANTINED battle), which the callback logs as missing results."""

    def __init__(self, *, bot_win=0.8, sentinel_win=0.7, fixed_win=0.5, reward=1.0, ep_len=20.0,
                 kinds=None, fail=False):
        from agents.training.eval_sharding import BOT, SENTINEL, FIXED
        self.wins = {BOT: bot_win, SENTINEL: sentinel_win, FIXED: fixed_win}
        self.reward, self.ep_len, self.kinds, self.fail = reward, ep_len, kinds, fail
        self.calls = []

    def run_cycle(self, pool, run_dir, **kw):
        from agents.training.eval_sharding import ShardResult
        self.calls.append(kw)
        if self.fail:
            from agents.training.rust_eval.executor import EvalCoreError
            raise EvalCoreError("a battle was QUARANTINED")
        for u in pool.units:
            if self.kinds is not None and u.kind not in self.kinds:
                continue
            n = u.n_games
            pool.publish(run_dir, ShardResult(
                unit_id=u.unit_id, item_key=u.item_key, worker_id=0,
                n_won=round(self.wins[u.kind] * n), n_finished=n, sum_reward=self.reward * n, n_episodes=n,
                sum_ep_len=self.ep_len * n, duration_sec=5.0, td_residuals=[]))
        return _Stats(len(pool.units))


def attach_fake_evaluator(cb, monkeypatch, **kw):
    """Give the callback's model a declared eval core that plays the cycle with the fake (win rates by
    kind) and stub the sentinel snapshot loads (the mock pool's snapshot paths are not files)."""
    from types import SimpleNamespace
    from agents.training.rust_eval import launch as rust_launch
    ev = _FakeEvaluator(**kw)
    cb.model._rust_collector = SimpleNamespace(evaluator=ev, cfg=SimpleNamespace(run_seed=5))
    monkeypatch.setattr(rust_launch, "load_sentinels", lambda *_a, **_k: {})
    return ev


# ── _check_bot_regression (edge-triggered warn) ───────────────────────────────

def test_regression_no_warning_before_threshold(tmp_path):
    cb = _make_callback(tmp_path)
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"heuristic": 0.40})
        mock_emit.assert_not_called()


def test_regression_peak_recorded(tmp_path):
    cb = _make_callback(tmp_path)
    cb._check_bot_regression({"heuristic": 0.75})
    assert cb._bot_peak["heuristic"] == pytest.approx(0.75)


def test_regression_warning_fires_when_drop_below_threshold(tmp_path):
    cb = _make_callback(tmp_path)
    cb._check_bot_regression({"heuristic": 0.72})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"heuristic": 0.55})
        assert mock_emit.call_count == 1
        assert "BOT_REGRESSION" in mock_emit.call_args[0][0]


def test_regression_no_warning_if_still_above_threshold(tmp_path):
    cb = _make_callback(tmp_path)
    cb._check_bot_regression({"heuristic": 0.80})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"heuristic": 0.65})
        mock_emit.assert_not_called()


def test_regression_only_fires_for_named_bots(tmp_path):
    cb = _make_callback(tmp_path)
    cb._check_bot_regression({"random": 0.90})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"random": 0.10})
        mock_emit.assert_not_called()


def test_regression_fires_only_once_while_regressed(tmp_path):
    cb = _make_callback(tmp_path)
    cb._check_bot_regression({"heuristic": 0.75})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"heuristic": 0.50})
        cb._check_bot_regression({"heuristic": 0.45})
        cb._check_bot_regression({"heuristic": 0.40})
        assert mock_emit.call_count == 1


def test_regression_re_arms_after_recovery(tmp_path):
    cb = _make_callback(tmp_path)
    cb._check_bot_regression({"heuristic": 0.75})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"heuristic": 0.50})  # fires
        cb._check_bot_regression({"heuristic": 0.70})  # recovered → re-arm
        cb._check_bot_regression({"heuristic": 0.45})  # fires again
        assert mock_emit.call_count == 2


# ── schedule / trigger ─────────────────────────────────────────────────────────

def test_schedule_is_flat(tmp_path):
    # Flat cadence + game count at every step — same constants as the bot-eval path.
    cb = _make_callback(tmp_path)
    cb.num_timesteps = 5_000_000
    assert cb._schedule() == (2_000_000, 100)
    cb.num_timesteps = 25_000_000
    assert cb._schedule() == (2_000_000, 100)


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


def test_n_sentinels_defaults_to_five():
    # The CLI/callback default must stay 5 (the historical behaviour) so an unset run is unchanged.
    cb = SelfPlayCallback(pool=_mock_pool(3), model_dir=None)
    assert cb._n_sentinels == 5


def test_n_sentinels_threads_into_pool_sentinel_selection(tmp_path, monkeypatch):
    # --n-sentinels N must drive how many evenly-spaced pool snapshots the eval requests (the set
    # PFSP gets fresh win-rates for). Drive the real _launch_eval on the fake eval core.
    cb = _make_callback(tmp_path, n_sentinels=9)
    assert cb._n_sentinels == 9
    cb._init_callback()
    attach_fake_evaluator(cb, monkeypatch)
    cb.num_timesteps = 2_000_000
    cb._launch_eval()
    cb._pool.sentinel_entries.assert_called_once_with(n=9)


def test_triggers_at_freq_boundary(tmp_path):
    cb = _make_callback(tmp_path)
    cb._init_callback()
    cb.num_timesteps = 2_000_000  # first eval boundary is now 2M
    with patch.object(cb, "_launch_eval") as mock_launch:
        cb._on_step()
        mock_launch.assert_called_once()
    assert cb._last_eval_step == 2_000_000


# ── force-eval (launcher button → SIGUSR2) shares the eval_callback mixin ───────

def test_force_eval_launches_off_cadence_when_idle(tmp_path):
    cb = _make_callback(tmp_path)
    cb._init_callback()                # populates _eval_root
    cb.num_timesteps = 1_234_567       # off a 2M cadence boundary → only the request can launch
    ec.request_forced_eval()           # event isolation: _isolate_force_eval_event (autouse)
    with patch.object(cb, "_launch_eval") as mock_launch, \
         patch("agents.training.eval_callback.send_event"):
        cb._on_step()
        mock_launch.assert_called_once()
    assert cb._last_eval_step == 1_234_567


# ── full lifecycle: launch → collect → promote / best ─────────────────────────

def test_lifecycle_collect_records_promotes_and_saves_best(tmp_path, monkeypatch):
    best_dir = tmp_path / "best"
    pool = _mock_pool(n_sentinels=3)
    cb = _make_callback(tmp_path, pool=pool, best_dir=str(best_dir),
                        promote_threshold=0.65)
    cb._init_callback()
    # sentinel win 0.70 > 0.65 threshold → promotion; bot win 0.80 → best model.
    ev = attach_fake_evaluator(cb, monkeypatch, bot_win=0.8, sentinel_win=0.7)
    snapshot = str(tmp_path / ".eval_runs" / "step_2000000" / "snapshot.zip")   # the cycle's frozen file

    cb._on_step()                              # boundary → play + merge + record + promote + best

    assert len(ev.calls) == 1 and ev.calls[0]["step"] == 2_000_000
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
    assert "eval/mean_reward_vs_pool" in recorded           # pool reward aggregate (was missing → blank TUI cell)
    assert "eval/mean_ep_len_vs_pool" in recorded           # pool ep-len, mirrors mean_ep_len_vs_bots
    assert "eval/sentinel_monotonicity" in recorded
    assert "train/selfplay_fraction" in recorded   # repointed: POOL-only share, not the curriculum coin
    assert "train/stable_fraction" in recorded
    assert "train/nonbot_fraction" in recorded
    assert "eval/win_rate_vs_bots" in recorded
    assert "eval/mean_ep_len_vs_bots" in recorded
    assert "eval/win_rate_vs_heuristic" in recorded
    assert "eval/win_rate_vs_sentinel_0" in recorded
    assert "eval/mean_reward_vs_sentinel_0" in recorded     # sentinel reward now recorded
    assert "eval/mean_ep_len_vs_sentinel_0" in recorded     # per-sentinel ep-len now recorded
    assert "train/selfplay_promoted_steps" in recorded
    # opponent default-rate telemetry recorded from the (paused) training env.
    assert "train/selfplay_opp_redecide_rate" in recorded


def test_collect_pushes_sentinel_reward_and_step_to_tui(tmp_path, monkeypatch):
    """The TUI builds the reward column from eval/mean_reward_vs_<opp> and labels sentinels
    from eval/sentinel_step_<i> — both must be pushed live (were missing → '—' / no step)."""
    from agents.training import selfplay_callback as sp
    pool = _mock_pool(n_sentinels=3)
    cb = _make_callback(tmp_path, pool=pool)
    cb._init_callback()
    captured: dict = {}
    monkeypatch.setattr(sp, "send_metrics", lambda d: captured.update(d))
    attach_fake_evaluator(cb, monkeypatch, bot_win=0.8, sentinel_win=0.7)

    cb._on_step()

    # Sentinel reward + step surfaced to the TUI...
    assert captured.get("eval/mean_reward_vs_sentinel_0") == pytest.approx(1.0)
    assert "eval/sentinel_step_0" in captured          # newest sentinel's checkpoint step
    # ...the Pool aggregate reward is pushed (drives the "vs Pool" reward cell)...
    assert "eval/mean_reward_vs_pool" in captured
    # ...and per-bot reward is pushed LIVE (not just win rate), so it isn't stale-from-resume.
    assert "eval/mean_reward_vs_heuristic" in captured


def test_lifecycle_no_promotion_below_threshold(tmp_path, monkeypatch):
    pool = _mock_pool(n_sentinels=3)
    cb = _make_callback(tmp_path, pool=pool, promote_threshold=0.65)
    cb._init_callback()
    attach_fake_evaluator(cb, monkeypatch, bot_win=0.8, sentinel_win=0.50)

    cb._on_step()                              # launches AND collects (blocking, in process)

    pool.add_from_path.assert_not_called()      # 0.50 ≤ 0.65 → no promotion
    recorded = {c.args[0] for c in cb.logger.record.call_args_list}
    assert "train/selfplay_promoted_steps" not in recorded
    assert "eval/win_rate_vs_pool" in recorded   # still recorded


def test_lifecycle_no_sentinels_handled(tmp_path, monkeypatch):
    """Pool with only the (newly seeded) step-0 entry still evals bots; pool win=0 → no promote."""
    pool = _mock_pool(n_sentinels=1)
    cb = _make_callback(tmp_path, pool=pool, promote_threshold=0.65)
    cb._init_callback()
    attach_fake_evaluator(cb, monkeypatch, bot_win=0.9, sentinel_win=0.4)

    cb._on_step()                              # launches AND collects (blocking, in process)

    recorded = {c.args[0] for c in cb.logger.record.call_args_list}
    assert "eval/win_rate_vs_bots" in recorded
    pool.add_from_path.assert_not_called()


def test_lifecycle_failed_cycle_logs_and_continues(tmp_path, monkeypatch):
    pool = _mock_pool(n_sentinels=3)
    cb = _make_callback(tmp_path, pool=pool, best_dir=str(tmp_path / "best"))
    cb._init_callback()
    attach_fake_evaluator(cb, monkeypatch, fail=True)       # the eval core fails the cycle: nothing published

    cb._on_step()                               # cycle failed → no record, no crash

    pool.add_from_path.assert_not_called()
    assert cb._best_aggregate_win_rate == -1.0
    assert not cb.logger.dump.called


def test_resume_restores_eval_step_no_immediate_re_eval(tmp_path):
    """A resume must NOT re-eval the same checkpoint right away — it restores the last eval
    step from metadata and waits for the next cadence boundary."""
    import json
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"latest_eval": {"step": 46_963_120, "win_rate_mean": 0.7, "opponents": {}}}))
    cb = _make_callback(tmp_path)
    # The RESUME's own step, on the MODEL (where _init_callback reads it — the callback's own
    # num_timesteps mirror is still 0 at that point). It must be present: the anchor is CLAMPED to
    # it, so a model left at 0 would model a fork-from-step-0 (see eval_fork_cadence_test.py).
    cb.model.num_timesteps = 46_963_136
    cb._init_callback()
    assert cb._last_eval_step == 46_963_120          # restored, not 0

    # Resume step in the SAME 3.5M bucket as the last eval → no immediate eval.
    cb.num_timesteps = 46_963_136
    with patch.object(cb, "_launch_eval") as ml:
        cb._on_step()
        ml.assert_not_called()
    # Crossing the next boundary (49M) → eval fires normally.
    cb.num_timesteps = 49_500_000
    with patch.object(cb, "_launch_eval") as ml:
        cb._on_step()
        ml.assert_called_once()


def test_fresh_run_with_no_metadata_evals_at_first_boundary(tmp_path):
    cb = _make_callback(tmp_path)          # tmp_path has no metadata.json
    cb._init_callback()
    assert cb._last_eval_step == 0
    cb.num_timesteps = 2_000_000
    with patch.object(cb, "_launch_eval") as ml:
        cb._on_step()
        ml.assert_called_once()


def test_init_does_not_seed_pool(tmp_path):
    # Seeding is GATED on competence (done by _maybe_seed_pool at startup / _collect_pending on
    # crossing) — _init_callback must NOT seed, or a weak model would pin a near-random opponent.
    pool = _mock_pool()
    pool.is_empty.return_value = True
    cb = _make_callback(tmp_path, pool=pool)
    cb._init_callback()
    pool.seed.assert_not_called()


def _env_method_calls(cb, name):
    return [c for c in cb.model.get_env.return_value.env_method.call_args_list
            if c.args and c.args[0] == name]


def test_collect_pushes_live_self_play_target(tmp_path, monkeypatch):
    pool = _mock_pool(n_sentinels=3)
    cb = _make_callback(tmp_path, pool=pool, promote_threshold=0.65)
    cb._init_callback()
    attach_fake_evaluator(cb, monkeypatch, bot_win=0.8, sentinel_win=0.7)
    cb._on_step()                              # launches AND collects (blocking, in process)
    # The live fraction (1 - heuristic_fraction(0.8) = 0.90) is pushed to the envs each eval.
    pushes = _env_method_calls(cb, "set_self_play_target")
    assert pushes, "expected a set_self_play_target env_method push"
    frac, gen = pushes[-1].args[1], pushes[-1].args[2]
    assert frac == pytest.approx(0.90, abs=1e-6)   # bot win 0.8 → 90% self-play
    assert isinstance(gen, int)


def test_collect_seeds_pool_when_crossing_threshold(tmp_path, monkeypatch):
    pool = _mock_pool(n_sentinels=0)          # empty pool → no sentinels yet
    pool.is_empty.return_value = True
    pool.sentinel_entries.return_value = []
    cb = _make_callback(tmp_path, pool=pool)
    cb._init_callback()
    attach_fake_evaluator(cb, monkeypatch, bot_win=0.8, sentinel_win=0.0)
    cb._on_step()                              # launches AND collects (blocking, in process)
    # Win rate 0.8 ≥ threshold → fraction > 0 with an empty pool → seed from the frozen snapshot.
    pool.add_from_path.assert_called_once()
    assert pool.add_from_path.call_args[0][1] == 2_000_000   # at the trigger step


def test_collect_does_not_seed_below_threshold(tmp_path, monkeypatch):
    pool = _mock_pool(n_sentinels=0)
    pool.is_empty.return_value = True
    pool.sentinel_entries.return_value = []
    cb = _make_callback(tmp_path, pool=pool)
    cb._init_callback()
    # Bot win 0.40 < SELF_PLAY_START (0.55) → fraction 0 → no seed, push fraction 0.
    attach_fake_evaluator(cb, monkeypatch, bot_win=0.40, sentinel_win=0.0)
    cb._on_step()                              # launches AND collects (blocking, in process)
    pool.add_from_path.assert_not_called()
    pushes = _env_method_calls(cb, "set_self_play_target")
    assert pushes and pushes[-1].args[1] == pytest.approx(0.0)


def test_collect_persists_summary(tmp_path, monkeypatch):
    pool = _mock_pool(n_sentinels=3)
    cb = _make_callback(tmp_path, pool=pool)
    cb._init_callback()
    attach_fake_evaluator(cb, monkeypatch, bot_win=0.8, sentinel_win=0.7)
    cb._on_step()                              # launches AND collects (blocking, in process)
    pool.persist_summary.assert_called()
    kw = pool.persist_summary.call_args.kwargs
    assert set(kw) >= {"win_rate_vs_bots", "self_play_fraction", "last_eval_step",
                       "seeded", "pool_generation"}
    assert kw["self_play_fraction"] == pytest.approx(0.90, abs=1e-6)


# ── gen3_tb_relevance_v1: NO SENTINEL MEASURED ⇒ NO POOL CURVE ───────────────────────────────
# The three `_vs_pool` scalars default to 0.0 on an empty sentinel list, and a published 0.0 is
# indistinguishable from "lost every game against the pool" — which is how it read for the first
# cycles of ai_v12_01_winprob_critic (win_rate_vs_pool 0.0 beside pool_snapshot_count 1).
# `sentinel_monotonicity` likewise published a perfect 1.0 on a pool too small to have an order.


def _cycle_recorded_tags(tmp_path, monkeypatch, *, n_sentinels, kinds=None):
    """Run ONE full eval cycle and return the set of tags it recorded."""
    pool = _mock_pool(n_sentinels=n_sentinels)
    cb = _make_callback(tmp_path, pool=pool)
    cb._init_callback()
    ev = attach_fake_evaluator(cb, monkeypatch, bot_win=0.6, sentinel_win=0.5, kinds=kinds)
    cb._on_step()                       # launches AND collects (blocking, in process)
    assert len(ev.calls) == 1, "the cycle did not run — the harness, not the gate, is wrong"
    return {c.args[0] for c in cb.logger.record.call_args_list}


def test_no_sentinel_result_publishes_no_pool_curve(tmp_path, monkeypatch):
    """Sentinels never report → the pool tags are ABSENT, not a confident 0.0."""
    from agents.training.eval_sharding import BOT
    recorded = _cycle_recorded_tags(tmp_path, monkeypatch, n_sentinels=3, kinds={BOT})
    assert "eval/win_rate_vs_bots" in recorded                 # the cycle DID run
    for tag in ("eval/win_rate_vs_pool", "eval/mean_reward_vs_pool",
                "eval/mean_ep_len_vs_pool", "eval/sentinel_monotonicity"):
        assert tag not in recorded, f"{tag} published with no sentinel measured"


def test_sentinel_results_publish_the_pool_curve(tmp_path, monkeypatch):
    """The positive control: with sentinels measured, every pool tag is back."""
    recorded = _cycle_recorded_tags(tmp_path, monkeypatch, n_sentinels=3)
    for tag in ("eval/win_rate_vs_pool", "eval/mean_reward_vs_pool",
                "eval/mean_ep_len_vs_pool", "eval/sentinel_monotonicity"):
        assert tag in recorded, f"{tag} missing on a cycle that DID measure sentinels"


def test_single_sentinel_publishes_pool_but_not_monotonicity(tmp_path, monkeypatch):
    """A one-entry pool has a win rate but no ORDER — monotonicity needs two."""
    recorded = _cycle_recorded_tags(tmp_path, monkeypatch, n_sentinels=1)
    assert "eval/win_rate_vs_pool" in recorded
    assert "eval/sentinel_monotonicity" not in recorded


# ── opponent-mix reporting fractions (pool / stable / nonbot) — pure, no battles ──
# These mirror rust_env_opponents.EpisodeOpponentSampler's draw for REPORTING only.

def _fo(label):
    """A stand-in FixedOpponentEntry — the fraction helper reads only `.label`."""
    from types import SimpleNamespace
    return SimpleNamespace(label=label)


def test_opponent_mix_no_stable_pool_only(tmp_path):
    """No stable opponents: train/selfplay_fraction = POOL share = sf·P (NOT the curriculum coin)."""
    cb = _make_callback(tmp_path)
    cb._fixed_opponents = []
    cb._floor_roster_count = 8
    # Pool seeded → pool takes the whole challenge: pool=sf, stable=0, nonbot=sf.
    assert cb._opponent_mix_fractions(0.9, pool_ready=True) == pytest.approx((0.9, 0.0, 0.9))
    # Pool NOT seeded → challenge returns None, all mass falls to the bot floor: pool=stable=nonbot=0.
    assert cb._opponent_mix_fractions(0.9, pool_ready=False) == pytest.approx((0.0, 0.0, 0.0))


def test_opponent_mix_unmastered_stable_caps_challenge(tmp_path):
    """An un-mastered stable opponent peels the capped share s off the pool's challenge bulk."""
    cb = _make_callback(tmp_path)
    cb._fixed_opponents = [_fo("ext_run")]
    cb._stable_mastered = set()
    cb._stable_challenge_share = 0.2
    cb._floor_roster_count = 8
    sp, st, nb = cb._opponent_mix_fractions(0.9, pool_ready=True)
    assert sp == pytest.approx(0.9 * 0.8)   # pool = sf·(1−s)
    assert st == pytest.approx(0.9 * 0.2)   # un-mastered stable = sf·s (challenge side)
    assert nb == pytest.approx(0.9)         # nonbot = sf (s cancels); bot = 0.1


def test_opponent_mix_unmastered_stable_no_pool_takes_whole_challenge(tmp_path):
    """Pool not seeded but un-mastered stable exists → stable IS the whole challenge (no s cap)."""
    cb = _make_callback(tmp_path)
    cb._fixed_opponents = [_fo("ext_run")]
    cb._stable_mastered = set()
    cb._floor_roster_count = 8
    sp, st, nb = cb._opponent_mix_fractions(0.6, pool_ready=False)
    assert (sp, st, nb) == pytest.approx((0.0, 0.6, 0.6))


def test_opponent_mix_mastered_stable_lives_in_weighted_floor(tmp_path):
    """A MASTERED stable opponent leaves the challenge for the WEIGHTED floor (it 'becomes a bot'),
    so it adds to stable_fraction via the floor — and nonbot != sf as a result."""
    cb = _make_callback(tmp_path)
    cb._fixed_opponents = [_fo("ext_run")]
    cb._stable_mastered = {"ext_run"}
    cb._bot_weight_vec = None                # uniform bots
    cb._floor_roster_count = 8
    cb._stable_challenge_share = 0.2
    sp, st, nb = cb._opponent_mix_fractions(0.9, pool_ready=True)
    assert sp == pytest.approx(0.9)              # challenge = 100% pool (no un-mastered stable)
    assert st == pytest.approx(0.1 * (1 / 9))    # floor 0.1 over 8 bots + 1 mastered → 1/9 slice
    assert nb == pytest.approx(0.9 + 0.1 / 9)    # > sf — the mastered-stable floor mass


def test_opponent_mix_weighted_floor(tmp_path):
    """--bot-weights changes W_h, so the mastered-stable floor slice is k_m/(W_h+k_m)."""
    cb = _make_callback(tmp_path)
    cb._fixed_opponents = [_fo("ext_run")]
    cb._stable_mastered = {"ext_run"}
    cb._bot_weight_vec = [3.0, 1.0]          # W_h = 4
    cb._floor_roster_count = 2
    sp, st, nb = cb._opponent_mix_fractions(0.5, pool_ready=True)
    assert sp == pytest.approx(0.5)
    assert st == pytest.approx(0.5 * (1 / 5))    # floor 0.5 over W_h=4 + k_m=1
    assert nb == pytest.approx(0.5 + 0.5 / 5)


# ── stable-opponent mastery: N-cycle confirm (eval-noise guard) ───────────────

def test_stable_mastery_requires_consecutive_cycles(tmp_path):
    """A single ≥-threshold cycle must NOT master (the flip is irreversible, so guard eval noise);
    only _MASTERY_CONFIRM_CYCLES consecutive cycles confirm it."""
    cb = _make_callback(tmp_path)
    cb._fixed_opponents = [object()]            # non-empty guard (content unused by the push)
    cb._stable_opponent_mastered_wr = 0.80
    lab = "ext_run"
    for _ in range(_MASTERY_CONFIRM_CYCLES - 1):
        cb._push_stable_mastered({lab: 0.85})
        assert lab not in cb._stable_mastered   # not yet — still warming the streak
    cb._push_stable_mastered({lab: 0.85})
    assert lab in cb._stable_mastered           # confirmed → mastered (one-way from here)


def test_stable_mastery_streak_resets_on_dip(tmp_path):
    """A below-threshold cycle resets the streak — a transient spike can't accumulate toward mastery."""
    cb = _make_callback(tmp_path)
    cb._fixed_opponents = [object()]
    cb._stable_opponent_mastered_wr = 0.80
    lab = "ext_run"
    cb._push_stable_mastered({lab: 0.85})       # streak 1
    cb._push_stable_mastered({lab: 0.50})       # dip → reset to 0
    for _ in range(_MASTERY_CONFIRM_CYCLES - 1):
        cb._push_stable_mastered({lab: 0.85})
        assert lab not in cb._stable_mastered   # streak restarted after the dip
    cb._push_stable_mastered({lab: 0.85})
    assert lab in cb._stable_mastered


# ── the eval row's OPPONENT-REGIME stamp (gen3_eval_sentinel_greedy_default_v1, 2026-09-07) ────
# `_sentinel_regime` is what tells `snapshot_ladder` whether a cycle's sentinel edges were measured
# under the ladder's own protocol. Getting it wrong in the permissive direction would put a
# systematically trainee-favouring edge (+8.9 pp [+7.0, +10.7]) into the dense matrix as the ONLY
# measurement of that pair, so the two conditions are asserted separately rather than as a pair.

def test_sentinel_regime_is_greedy_and_symmetric_by_default(tmp_path):
    cb = SelfPlayCallback(pool=_mock_pool(1), model_dir=str(tmp_path),
                          eval_sentinel_greedy=True)
    assert cb._sentinel_regime() == {"greedy": True, "symmetric_teams": True}


def test_sentinel_regime_is_neither_under_no_eval_sentinel_greedy(tmp_path):
    cb = SelfPlayCallback(pool=_mock_pool(1), model_dir=str(tmp_path),
                          eval_sentinel_greedy=False)
    assert cb._sentinel_regime() == {"greedy": False, "symmetric_teams": False}


def test_a_SPECIALIST_run_is_greedy_but_NOT_ladder_symmetric(tmp_path):
    """`--trainee-team` draws BOTH players from the taught team — symmetric between the players,
    but not the LADDER's draw (full pool, 0.1 sample bias). So the pair is not reusable, and the
    row must say so rather than let the ladder infer symmetry from the greedy half."""
    cb = SelfPlayCallback(pool=_mock_pool(1), model_dir=str(tmp_path),
                          eval_sentinel_greedy=True,
                          trainee_team_str="Tyranitar @ Leftovers\n")
    assert cb._sentinel_regime() == {"greedy": True, "symmetric_teams": False}


def test_the_detached_ladder_updater_never_sees_the_gpu(tmp_path, monkeypatch):
    """gen3_ladder_off_gpu_v1: the promotion's round-robin updater plays on the CPU, but it opened a
    CUDA context anyway (330 MiB of the training card each, several alive at once — measured
    2026-10-01). It is spawned with no visible CUDA device."""
    import types
    from agents.training import selfplay_callback as sc
    seen = {}

    def fake_popen(argv, **kw):
        seen.update(kw, argv=argv)
        return MagicMock()
    monkeypatch.setattr(sc.subprocess, "Popen", fake_popen)
    me = types.SimpleNamespace(_model_dir=str(tmp_path), _ladder_games=4)
    sc.SelfPlayCallback._spawn_snapshot_ladder_update(me, 1000)
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "" and "--promote" in seen["argv"]


def test_the_detached_ladder_updater_passes_no_poke_env_play_knob(tmp_path, monkeypatch):
    """P2 (2026-10-06): the updater plays on the Rust eval engine (`snapshot_ladder_play`), so the poke-env
    play path's `--impl` / `--concurrency` are gone from its argv (the CLI ignores them, loudly, for an older
    caller). The argv the updater is spawned with must still PARSE under the ladder CLI."""
    import types
    from agents.training import selfplay_callback as sc
    seen = {}
    monkeypatch.setattr(sc.subprocess, "Popen", lambda argv, **kw: seen.update(argv=argv) or MagicMock())
    sc.SelfPlayCallback._spawn_snapshot_ladder_update(
        types.SimpleNamespace(_model_dir=str(tmp_path), _ladder_games=4), 1000)
    argv = seen["argv"]
    assert argv[1:3] == ["-m", "agents.training.snapshot_ladder"]
    assert "--impl" not in argv and "--concurrency" not in argv
    assert argv[argv.index("--promote") + 1] == "1000" and argv[argv.index("--n-games") + 1] == "4"
