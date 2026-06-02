import os
from unittest.mock import MagicMock, patch

import pytest

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from agents.opponents import Gen3StallerPlayer
from agents.training.eval_callback import (
    PerOpponentEvalCallback, bot_mean, opponent_name, RANDOM_OPPONENT_NAME,
    _per_opponent_concurrency, _EVAL_TOTAL_CONCURRENCY, _EVAL_CONCURRENCY,
)


# ── bot_mean ─────────────────────────────────────────────────────────────────

def test_bot_mean_excludes_random():
    assert bot_mean({"Random": 0.9, "Heuristic": 0.4, "Staller": 0.6}) == pytest.approx(0.5)


def test_bot_mean_all_random_returns_zero():
    assert bot_mean({"Random": 0.9}) == pytest.approx(0.0)


def test_bot_mean_empty_returns_zero():
    assert bot_mean({}) == pytest.approx(0.0)


def test_bot_mean_no_random_averages_all():
    assert bot_mean({"Heuristic": 0.4, "Staller": 0.6}) == pytest.approx(0.5)


# ── opponent_name ─────────────────────────────────────────────────────────────

def test_opponent_name_random():
    assert opponent_name(RandomPlayer) == "Random"


def test_opponent_name_heuristic():
    assert opponent_name(SimpleHeuristicsPlayer) == "Heuristic"


def test_opponent_name_staller():
    assert opponent_name(Gen3StallerPlayer) == "Staller"


def test_opponent_name_unknown_falls_back_to_class_name():
    class MyCustomPlayer:
        pass
    assert opponent_name(MyCustomPlayer) == "MyCustomPlayer"


def test_random_opponent_name_constant_matches_function():
    assert RANDOM_OPPONENT_NAME == opponent_name(RandomPlayer)


# ── _per_opponent_concurrency ─────────────────────────────────────────────────

def test_per_opponent_concurrency_splits_budget():
    # Aggregate (n × per-player) stays at/below the total budget.
    assert _per_opponent_concurrency(5) == _EVAL_TOTAL_CONCURRENCY // 5
    assert _per_opponent_concurrency(9) * 9 <= _EVAL_TOTAL_CONCURRENCY + 9


def test_per_opponent_concurrency_floored_at_16():
    # Many opponents still get enough concurrency to saturate inference.
    assert _per_opponent_concurrency(100) == 16


def test_per_opponent_concurrency_zero_falls_back():
    assert _per_opponent_concurrency(0) == _EVAL_CONCURRENCY


# ── eval_one_matchup (shared single-matchup body) ─────────────────────────────

def test_eval_one_matchup_returns_metrics_and_arms_forensics(tmp_path):
    """Pins the contract the self-play worker relies on for sentinel matchups."""
    import asyncio
    from agents.training import eval_callback as ec

    class _FakeBattle:
        def __init__(self, finished, turn):
            self.finished = finished
            self.turn = turn

    class _FakeTrainee:
        def __init__(self):
            self.n_finished_battles = 2          # nonzero → exercises reset_battles()
            self.n_won_battles = 0
            self.mean_episode_reward = 0.0
            self._battles = {}
            self.reset_battles_called = False
            self.reset_reward_called = False
            self.forensic = None

        def reset_battles(self):
            self.reset_battles_called = True

        def reset_reward_tracking(self):
            self.reset_reward_called = True

        def begin_forensic_cycle(self, d, step):
            self.forensic = (d, step)

        async def battle_against(self, opp, n_battles):
            self.n_finished_battles = n_battles
            self.n_won_battles = 3
            self.mean_episode_reward = 1.5
            self._battles = {f"b{i}": _FakeBattle(True, 10) for i in range(n_battles)}

    class _FakeOpp:
        n_finished_battles = 0
        def reset_battles(self):
            pass

    tr = _FakeTrainee()
    m = asyncio.run(ec.eval_one_matchup(tr, _FakeOpp(), n_games=4,
                                        model_dir=str(tmp_path), step=100, name="sentinel_0"))
    assert m == {
        "name": "sentinel_0",
        "win_rate": pytest.approx(3 / 4),
        "reward_mean": pytest.approx(1.5),
        "ep_len": pytest.approx(10.0),
        "duration_sec": m["duration_sec"],  # wall-clock, just assert presence
    }
    assert tr.reset_battles_called and tr.reset_reward_called
    assert tr.forensic == (
        os.path.join(str(tmp_path), "eval_traces", "step_100", "sentinel_0"), 100,
    )


def _make_callback(best_model_save_path=None, model_dir=None):
    cb = PerOpponentEvalCallback(
        model_dir=model_dir,
        server_config=MagicMock(),
        use_v2_bots=False,
        best_model_save_path=best_model_save_path,
    )
    cb.model = MagicMock()
    cb.model.save = MagicMock()
    cb.model.logger = MagicMock()
    cb.num_timesteps = 0
    return cb


# --- Schedule ---

def test_schedule_early_phase():
    cb = _make_callback()
    cb.num_timesteps = 5_000_000
    freq, n_games = cb._schedule()
    assert freq == 2_000_000
    assert n_games == 100


def test_schedule_mid_phase():
    cb = _make_callback()
    cb.num_timesteps = 30_000_000
    freq, n_games = cb._schedule()
    assert freq == 3_500_000
    assert n_games == 300


def test_schedule_late_phase():
    cb = _make_callback()
    cb.num_timesteps = 60_000_000
    freq, n_games = cb._schedule()
    assert freq == 3_500_000
    assert n_games == 300


def test_schedule_very_late_phase():
    cb = _make_callback()
    cb.num_timesteps = 120_000_000
    freq, n_games = cb._schedule()
    assert freq == 5_000_000
    assert n_games == 300


def test_schedule_boundary_20m():
    cb = _make_callback()
    cb.num_timesteps = 20_000_000  # 20M is NOT < 20M → falls in the 20–100M tier
    freq, n_games = cb._schedule()
    assert freq == 3_500_000
    assert n_games == 300


def test_schedule_boundary_100m():
    cb = _make_callback()
    cb.num_timesteps = 100_000_000  # 100M is NOT < 100M → falls in the 100M+ tier
    freq, n_games = cb._schedule()
    assert freq == 5_000_000
    assert n_games == 300


# --- Trigger logic ---

def test_no_eval_at_step_zero():
    cb = _make_callback()
    cb.num_timesteps = 0
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_not_called()


def test_no_eval_before_first_freq_boundary():
    cb = _make_callback()
    cb.num_timesteps = 500_000  # below the first 1M-step boundary
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_not_called()


def test_triggers_at_early_freq():
    cb = _make_callback()
    cb.num_timesteps = 2_000_000  # first boundary is now 2M
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_called_once()


def test_triggers_at_mid_freq():
    cb = _make_callback()
    cb._last_eval_step = 20_000_000
    cb.num_timesteps = 22_000_000
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_called_once()


def test_triggers_at_late_freq():
    cb = _make_callback()
    cb._last_eval_step = 51_000_000
    cb.num_timesteps = 54_000_000
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_called_once()


def test_no_double_trigger_within_interval():
    cb = _make_callback()
    cb._last_eval_step = 1_000_000
    cb.num_timesteps = 1_500_000
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_not_called()


def test_skips_launch_while_previous_eval_running():
    cb = _make_callback()
    cb.num_timesteps = 2_000_000
    cb._pending = {"step": 1_000_000,
                   "procs": [{"proc": MagicMock(**{"poll.return_value": None})}]}
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_not_called()  # previous cycle still running → skip
    assert cb._last_eval_step == 2_000_000  # but the boundary is consumed


def test_updates_last_eval_step_on_trigger():
    cb = _make_callback()
    cb.num_timesteps = 2_000_000  # first boundary is now 2M
    with patch.object(cb, '_launch_eval'):
        cb._on_step()
    assert cb._last_eval_step == 2_000_000


# --- Best model saving ---

def test_saves_best_model_on_first_improvement(tmp_path):
    cb = _make_callback(best_model_save_path=str(tmp_path))
    cb._best_aggregate_win_rate = -1.0
    aggregate = 0.65
    if aggregate > cb._best_aggregate_win_rate:
        cb._best_aggregate_win_rate = aggregate
        cb.model.save(str(tmp_path / "best_model"))
    cb.model.save.assert_called_once()
    assert cb._best_aggregate_win_rate == pytest.approx(0.65)


def test_does_not_save_when_aggregate_does_not_improve(tmp_path):
    cb = _make_callback(best_model_save_path=str(tmp_path))
    cb._best_aggregate_win_rate = 0.80
    aggregate = 0.75
    if aggregate > cb._best_aggregate_win_rate:
        cb.model.save(str(tmp_path / "best_model"))
    cb.model.save.assert_not_called()


# ── roster ────────────────────────────────────────────────────────────────────

from agents.training.eval_callback import (
    eval_opponent_names, claim_next_opponent, read_latest_eval_block,
)


def test_eval_opponent_names_excludes_v2_by_default():
    names = eval_opponent_names(False)
    assert names == ["Random", "Heuristic", "Staller", "Aggressive", "SetupSweep"]


def test_eval_opponent_names_includes_v2_when_enabled():
    assert len(eval_opponent_names(True)) == 9


# ── work-stealing claim (atomic O_EXCL) ───────────────────────────────────────

def test_claim_next_opponent_claims_each_exactly_once(tmp_path):
    names = ["a", "b", "c"]
    claimed = []
    while (n := claim_next_opponent(str(tmp_path), names)) is not None:
        claimed.append(n)
    assert sorted(claimed) == names              # every opponent claimed
    assert len(claimed) == len(set(claimed))     # none twice
    assert claim_next_opponent(str(tmp_path), names) is None  # pool exhausted


def test_claim_next_opponent_no_double_claim_across_workers(tmp_path):
    cd = str(tmp_path)
    names = ["a", "b", "c", "d"]
    # Two "workers" interleave their claims; no opponent may be handed out twice.
    a1 = claim_next_opponent(cd, names)
    b1 = claim_next_opponent(cd, names)
    a2 = claim_next_opponent(cd, names)
    b2 = claim_next_opponent(cd, names)
    got = [x for x in (a1, b1, a2, b2) if x]
    assert sorted(got) == names
    assert claim_next_opponent(cd, names) is None


# ── read_latest_eval_block (TUI resume source) ────────────────────────────────

def test_read_latest_eval_block_reads_top_level(tmp_path):
    import json as _json
    meta = {"latest_eval": {"step": 200, "win_rate_mean": 0.5,
                            "opponents": {"Random": {"win_rate": 0.7}}}}
    p = tmp_path / "metadata.json"
    p.write_text(_json.dumps(meta))
    blk = read_latest_eval_block(str(p))
    assert blk["step"] == 200 and blk["win_rate_mean"] == 0.5


def test_read_latest_eval_block_legacy_per_checkpoint_fallback(tmp_path):
    # Older metadata.json nested evals under each checkpoint — still readable.
    import json as _json
    meta = {"snapshot_history": {
        "checkpoint_100_steps.zip": {"evals": {"step": 100, "win_rate_mean": 0.3, "opponents": {}}},
        "checkpoint_200_steps.zip": {"evals": {"step": 200, "win_rate_mean": 0.5,
                                               "opponents": {"Random": {"win_rate": 0.7}}}},
    }}
    p = tmp_path / "metadata.json"
    p.write_text(_json.dumps(meta))
    blk = read_latest_eval_block(str(p))
    assert blk["step"] == 200 and blk["win_rate_mean"] == 0.5


def test_read_latest_eval_block_missing_or_empty():
    assert read_latest_eval_block(None) is None
    assert read_latest_eval_block("/no/such/metadata.json") is None


# ── orchestrator: work-stealing launch → collect → best-model (stubbed Popen) ──

def _fake_worker_popen(ec, win=0.8):
    """A fake Popen whose 'worker' work-steals from the claim dir and writes
    per-opponent result files, exactly like the real eval_worker."""
    class _FakeProc:
        returncode = 0
        def poll(self): return 0
        def wait(self, timeout=None): return 0

    def fake_popen(argv, stdout=None, stderr=None, env=None):
        import json as _json
        cfg = _json.load(open(argv[-1]))
        while True:
            name = ec.claim_next_opponent(cfg["claim_dir"], cfg["opponent_pool"])
            if name is None:
                break
            out = os.path.join(cfg["result_dir"], f"result__{name}.json")
            with open(out, "w") as f:
                _json.dump({"win_rate": win, "reward_mean": 1.0,
                            "ep_len": 20.0, "duration_sec": 5.0}, f)
        return _FakeProc()
    return fake_popen


def test_orchestrator_workstealing_collect_and_promote_best(tmp_path, monkeypatch):
    from agents.training import eval_callback as ec
    best_dir = tmp_path / "best"
    cb = PerOpponentEvalCallback(
        model_dir=str(tmp_path), server_config=MagicMock(),
        use_v2_bots=False, best_model_save_path=str(best_dir),
        n_workers=3, showdown_port=9999,
    )
    cb.model = MagicMock()
    cb.model.save = lambda base: open(base + ".zip", "w").close()
    cb._logger = MagicMock()
    cb.num_timesteps = 2_000_000       # first eval boundary is now 2M
    cb._init_callback()
    monkeypatch.setattr(ec.subprocess, "Popen", _fake_worker_popen(ec, win=0.8))

    cb._on_step()                      # boundary → spawn work-stealing workers
    assert cb._pending is not None and cb._pending["step"] == 2_000_000

    cb.num_timesteps = 2_000_001
    cb._on_step()                      # all done → merge per-opponent results + promote

    assert cb._pending is None
    assert (best_dir / "best_model.zip").exists()
    assert cb._best_aggregate_win_rate == pytest.approx(0.8)
    # Every opponent's win-rate made it into the recorded metrics.
    recorded = {c.args[0] for c in cb.logger.record.call_args_list}
    for name in eval_opponent_names(False):
        assert f"eval/win_rate_vs_{name}" in recorded


def test_orchestrator_worker_failure_logs_and_continues(tmp_path, monkeypatch):
    from agents.training import eval_callback as ec
    cb = PerOpponentEvalCallback(
        model_dir=str(tmp_path), server_config=MagicMock(),
        use_v2_bots=False, best_model_save_path=str(tmp_path / "best"),
        n_workers=2, showdown_port=9999,
    )
    cb.model = MagicMock()
    cb.model.save = lambda base: open(base + ".zip", "w").close()
    cb._logger = MagicMock()
    cb.num_timesteps = 2_000_000       # first eval boundary is now 2M
    cb._init_callback()

    class _FailProc:
        returncode = 1
        def poll(self): return 1
        def wait(self, timeout=None): return 1

    monkeypatch.setattr(ec.subprocess, "Popen", lambda *a, **k: _FailProc())  # no results written

    cb._on_step()
    cb.num_timesteps = 2_000_001
    cb._on_step()                      # all workers failed → no record, no crash

    assert cb._pending is None
    assert cb._best_aggregate_win_rate == -1.0
    assert not cb.logger.dump.called


def test_drain_waits_for_inflight_eval_then_collects(tmp_path, monkeypatch):
    """Graceful shutdown: drain() must wait on the workers and still record."""
    from agents.training import eval_callback as ec
    cb = PerOpponentEvalCallback(
        model_dir=str(tmp_path), server_config=MagicMock(),
        use_v2_bots=False, best_model_save_path=str(tmp_path / "best"),
        n_workers=1, showdown_port=9999,
    )
    cb.model = MagicMock()
    cb.model.save = lambda base: open(base + ".zip", "w").close()
    cb._logger = MagicMock()
    cb.num_timesteps = 2_000_000       # first eval boundary is now 2M
    cb._init_callback()

    waited = {"n": 0}

    class _SlowProc:
        returncode = 0
        def poll(self): return None            # looks alive until wait() is called
        def wait(self, timeout=None): waited["n"] += 1; return 0

    def fake_popen(argv, stdout=None, stderr=None, env=None):
        import json as _json
        cfg = _json.load(open(argv[-1]))
        for nm in cfg["opponent_pool"]:        # results are ready on disk
            with open(os.path.join(cfg["result_dir"], f"result__{nm}.json"), "w") as f:
                _json.dump({"win_rate": 0.5, "reward_mean": 0.0,
                            "ep_len": 10.0, "duration_sec": 1.0}, f)
        return _SlowProc()

    monkeypatch.setattr(ec.subprocess, "Popen", fake_popen)

    cb._on_step()                              # launch; poll()=None so it stays pending
    assert cb._pending is not None
    cb.drain(timeout=5)                        # graceful shutdown blocks then collects
    assert cb._pending is None
    assert waited["n"] >= 1                     # actually waited on the worker
    assert cb.logger.record.called


def test_replay_last_eval_publishes_to_tui_on_init(tmp_path, monkeypatch):
    """Resume: _init_callback re-publishes the most recent eval to the TUI."""
    import json as _json
    from agents.training import eval_callback as ec
    meta = {"latest_eval": {
        "step": 200, "win_rate_mean": 0.5, "win_rate_vs_bots": 0.4,
        "mean_reward_vs_bots": -1.0, "mean_ep_len_vs_bots": 30.0,
        "opponents": {"Random": {"win_rate": 0.7, "mean_reward": 0.1, "mean_ep_len": 25.0}},
    }}
    (tmp_path / "metadata.json").write_text(_json.dumps(meta))

    sent = {}
    monkeypatch.setattr(ec, "send_metrics", lambda d: sent.update(d))
    cb = PerOpponentEvalCallback(model_dir=str(tmp_path), server_config=MagicMock())
    cb._init_callback()

    assert sent.get("eval/win_rate_vs_Random") == 0.7
    assert sent.get("eval/win_rate_mean") == 0.5
    assert sent.get("_step") == 200
