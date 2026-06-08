import os
import math
from unittest.mock import MagicMock, patch

import pytest

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from agents.opponents import Gen3StallerPlayer
from agents.training.eval_callback import (
    PerOpponentEvalCallback, bot_mean, opponent_name, RANDOM_OPPONENT_NAME,
    external_elo, record_external_elos,
    _per_opponent_concurrency, _EVAL_TOTAL_CONCURRENCY, _EVAL_CONCURRENCY,
)
from unittest.mock import MagicMock


# ── external_elo — display-only ballpark from the trainee's bot-anchored rating ──

def test_external_elo_even_winrate_equals_trainee():
    assert external_elo(1500.0, 0.5) == pytest.approx(1500.0)


def test_external_elo_stronger_opponent_when_trainee_loses():
    # Trainee wins only 25% → the opponent is rated ABOVE the trainee.
    assert external_elo(1500.0, 0.25) > 1500.0


def test_external_elo_weaker_opponent_when_trainee_wins():
    assert external_elo(1500.0, 0.75) < 1500.0


def test_external_elo_symmetric_about_even():
    # logit is odd around 0.5, so equal-distance win rates give equal-and-opposite gaps.
    hi = external_elo(1500.0, 0.80) - 1500.0
    lo = external_elo(1500.0, 0.20) - 1500.0
    assert hi == pytest.approx(-lo)


def test_external_elo_clamps_extremes_finite():
    # 0% / 100% would be ±inf un-clamped; clamped to a finite ≈±676 gap.
    assert external_elo(1500.0, 0.0) == pytest.approx(1500.0 + 676, abs=2)
    assert external_elo(1500.0, 1.0) == pytest.approx(1500.0 - 676, abs=2)


def test_record_external_elos_writes_per_opponent_keys():
    logger, tui = MagicMock(), {}
    record_external_elos(logger, tui, 1500.0, {"ext_run_a": 0.5, "ext_run_b": 0.25})
    assert tui["eval/elo_vs_ext_run_a"] == 1500           # even → trainee rating
    assert tui["eval/elo_vs_ext_run_b"] > 1500            # losing → stronger
    recorded = {c.args[0] for c in logger.record.call_args_list}
    assert "eval/elo_vs_ext_run_a" in recorded and "eval/elo_vs_ext_run_b" in recorded


def test_record_external_elos_prefers_recorded_over_ballpark():
    """A carried source ELO wins over the trainee-derived ballpark; it's shown even with no trainee
    rating (None), while a fallback-only opponent is skipped when there's no rating yet."""
    logger, tui = MagicMock(), {}
    record_external_elos(logger, tui, None, {"ext_carried": 0.9, "ext_fallback": 0.9},
                         source_elos={"ext_carried": 1888.0})
    assert tui["eval/elo_vs_ext_carried"] == 1888         # recorded ELO used verbatim
    assert "eval/elo_vs_ext_fallback" not in tui          # no carried ELO + no trainee rating → skipped


# ── write_best_model_sidecar (best_model.json, reusing the checkpoint-sidecar code) ──

def test_write_best_model_sidecar_includes_elo(tmp_path):
    from agents.training.eval_callback import write_best_model_sidecar
    from agents.model.snapshot import record_eval_results, read_checkpoint_metadata
    model_dir = str(tmp_path)
    best = os.path.join(model_dir, "best_model")
    os.makedirs(best)
    record_eval_results(model_dir, step=100, metrics={"elo": 1888.0, "win_rate_vs_bots": 0.8})
    best_zip = os.path.join(best, "best_model.zip")
    open(best_zip, "w").close()
    model = MagicMock(n_epochs=7)
    model.policy.optimizer.param_groups = [{"lr": 1e-4}]
    write_best_model_sidecar(model_dir, best_zip, model)
    sidecar = read_checkpoint_metadata(best_zip)          # reads <best_zip − .zip>.json = best_model.json
    assert os.path.exists(os.path.join(best, "best_model.json"))
    assert sidecar["latest_eval"]["elo"] == pytest.approx(1888.0)
    assert sidecar["lr"] == pytest.approx(1e-4) and sidecar["n_epochs"] == 7


# ── bot_mean ─────────────────────────────────────────────────────────────────

def test_bot_mean_excludes_random():
    assert bot_mean({"random": 0.9, "heuristic": 0.4, "staller": 0.6}) == pytest.approx(0.5)


def test_bot_mean_all_random_returns_zero():
    assert bot_mean({"random": 0.9}) == pytest.approx(0.0)


def test_bot_mean_empty_returns_zero():
    assert bot_mean({}) == pytest.approx(0.0)


def test_bot_mean_no_random_averages_all():
    assert bot_mean({"heuristic": 0.4, "staller": 0.6}) == pytest.approx(0.5)


# ── opponent_name ─────────────────────────────────────────────────────────────

def test_opponent_name_random():
    assert opponent_name(RandomPlayer) == "random"


def test_opponent_name_heuristic():
    assert opponent_name(SimpleHeuristicsPlayer) == "heuristic"


def test_opponent_name_staller():
    assert opponent_name(Gen3StallerPlayer) == "staller"


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

        def td_tail(self):
            return None   # no captured residuals in this stub → omitted from the result dict

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
        "td_resid_tail": None,              # stub captured no residuals → None
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
        best_model_save_path=best_model_save_path,
    )
    cb.model = MagicMock()
    cb.model.save = MagicMock()
    cb.model.logger = MagicMock()
    cb.num_timesteps = 0
    return cb


# --- Schedule ---

@pytest.mark.parametrize("step", [1_000_000, 30_000_000, 120_000_000])
def test_schedule_is_flat_at_every_step(step):
    # One cadence, one game count — no maturity tiers, no per-opponent caps.
    cb = _make_callback()
    cb.num_timesteps = step
    freq, n_games = cb._schedule()
    assert freq == 2_000_000
    assert n_games == 100


# --- Trigger logic ---

def test_no_eval_at_step_zero():
    cb = _make_callback()
    cb.num_timesteps = 0
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_not_called()


def test_no_eval_before_first_freq_boundary():
    cb = _make_callback()
    cb.num_timesteps = 500_000  # below the first 2M-step boundary
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_not_called()


def test_triggers_at_first_boundary():
    cb = _make_callback()
    cb.num_timesteps = 2_000_000  # first 2M boundary
    with patch.object(cb, '_launch_eval') as mock_run:
        cb._on_step()
        mock_run.assert_called_once()


def test_triggers_at_each_later_boundary():
    # Flat 2M cadence: crossing any 2M boundary fires, regardless of training maturity.
    cb = _make_callback()
    cb._last_eval_step = 20_000_000
    cb.num_timesteps = 22_000_000
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


def test_eval_opponent_names_is_full_roster():
    # All eight archetype bots (both v1 and v2 of each) + Random as the eval-only floor.
    # snake_case names match _EVAL_OPPONENT_SPECS keys + the metric-key convention (1e50634).
    assert eval_opponent_names() == [
        "random",
        "heuristic", "heuristic2",
        "staller", "staller_v2",
        "aggressive", "aggressive_v2",
        "setup_sweep", "setup_sweep_v2",
    ]


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
                            "opponents": {"random": {"win_rate": 0.7}}}}
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
                                               "opponents": {"random": {"win_rate": 0.7}}}},
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
        best_model_save_path=str(best_dir),
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
    for name in eval_opponent_names():
        assert f"eval/win_rate_vs_{name}" in recorded


def test_orchestrator_worker_failure_logs_and_continues(tmp_path, monkeypatch):
    from agents.training import eval_callback as ec
    cb = PerOpponentEvalCallback(
        model_dir=str(tmp_path), server_config=MagicMock(),
        best_model_save_path=str(tmp_path / "best"),
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
        best_model_save_path=str(tmp_path / "best"),
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


# ── process-unique eval account nonce / cycle_tag (the restart-collision fix) ──

def test_eval_run_nonce_is_short_and_varies():
    from agents.training.eval_callback import eval_run_nonce
    a, b = eval_run_nonce(), eval_run_nonce()
    assert len(a) == 3 and len(b) == 3
    assert a != b  # per-process counter guarantees successive calls differ


def test_eval_account_names_within_showdown_18_char_limit():
    # The bug was account-name COLLISION; this guards the other constraint — that the
    # process-unique tag still fits Showdown's 18-char username cap for every prefix.
    from agents.training.eval_callback import eval_run_nonce, _b36, _EVAL_OPPONENT_SPECS
    cycle_tag = eval_run_nonce() + _b36(35, 1)          # nonce(3) + cycle(1) = 4
    tag = f"{cycle_tag}913"                              # pessimistic wid=9, claim_seq=13
    for (_name, _cls, prefix) in _EVAL_OPPONENT_SPECS:
        assert len(prefix + tag) <= 18, (prefix, prefix + tag)
    assert len(f"RLEv{tag}9") <= 18   # trainee player
    assert len(f"SPtr{tag}") <= 18    # self-play sentinel trainee
    assert len(f"SPse{tag}") <= 18    # self-play sentinel opponent


def test_cycle_tag_is_process_unique_not_step_derived(tmp_path, monkeypatch):
    """cycle_tag must come from the per-process nonce + per-cycle counter, NOT the step —
    the resume re-eval fires at the same step every restart, so a step tag collided."""
    from agents.training import eval_callback as ec
    cb = PerOpponentEvalCallback(model_dir=str(tmp_path), server_config=MagicMock(),
                                 n_workers=1, showdown_port=9999)
    cb.model = MagicMock()
    cb.model.save = lambda b: open(b + ".zip", "w").close()
    cb._logger = MagicMock()
    cb._init_callback()

    captured = []
    monkeypatch.setattr(ec, "spawn_eval_workers",
                        lambda run_dir, base_cfg, n: captured.append(base_cfg["cycle_tag"]) or [])

    cb.num_timesteps = 2_000_000
    cb._launch_eval()
    cb._pending = None
    cb.num_timesteps = 4_000_000
    cb._launch_eval()

    assert captured[0].startswith(cb._eval_run_nonce)   # carries the process nonce
    assert captured[0] != captured[1]                   # per-cycle counter advances
    # Two launches at the SAME step (the resume-collision scenario) still differ:
    cb._pending = None
    cb.num_timesteps = 2_000_000
    cb._launch_eval()
    assert captured[2] != captured[0]
    # And it is NOT the old step-derived tag.
    assert captured[0] != f"{2_000_000 // 100 % 10000:04d}"


# ── eval-cycle watchdog (a hung worker must not wedge eval forever) ───────────

def _hung_proc_factory(killed):
    class _HungProc:
        returncode = None
        def poll(self):
            return None                       # never finishes on its own
        def kill(self):
            self.returncode = -9
            killed["n"] += 1
        def wait(self, timeout=None):
            return -9
    return _HungProc


def test_watchdog_aborts_hung_cycle_and_collects_partial(tmp_path, monkeypatch):
    from agents.training import eval_callback as ec
    cb = PerOpponentEvalCallback(model_dir=str(tmp_path), server_config=MagicMock(),
                                 best_model_save_path=str(tmp_path / "best"),
                                 n_workers=2, showdown_port=9999)
    cb.model = MagicMock()
    cb.model.save = lambda b: open(b + ".zip", "w").close()
    cb._logger = MagicMock()
    cb.num_timesteps = 2_000_000
    cb._init_callback()

    killed = {"n": 0}
    HungProc = _hung_proc_factory(killed)

    def fake_popen(argv, stdout=None, stderr=None, env=None):
        import json as _json
        cfg = _json.load(open(argv[-1]))
        # Partial: only the first two opponents wrote results before the (simulated) hang.
        for nm in cfg["opponent_pool"][:2]:
            with open(os.path.join(cfg["result_dir"], f"result__{nm}.json"), "w") as f:
                _json.dump({"win_rate": 0.5, "reward_mean": 0.0,
                            "ep_len": 10.0, "duration_sec": 1.0}, f)
        return HungProc()

    monkeypatch.setattr(ec.subprocess, "Popen", fake_popen)

    cb._on_step()                                 # launch; procs report poll()=None
    assert cb._pending is not None
    # Make the cycle look overdue, then step again → watchdog fires.
    cb._pending["launched_at"] = ec.time.monotonic() - (ec._EVAL_CYCLE_TIMEOUT_SEC + 1)
    cb.num_timesteps = 2_000_001                  # same freq bucket → no relaunch
    cb._on_step()

    assert killed["n"] >= 1                        # the hung workers were killed
    assert cb._pending is None                     # cleared → eval can resume
    assert cb.logger.record.called                 # partial results were still recorded


def test_watchdog_leaves_a_fresh_cycle_running(tmp_path, monkeypatch):
    from agents.training import eval_callback as ec
    cb = PerOpponentEvalCallback(model_dir=str(tmp_path), server_config=MagicMock(),
                                 n_workers=1, showdown_port=9999)
    cb.model = MagicMock()
    cb.model.save = lambda b: open(b + ".zip", "w").close()
    cb._logger = MagicMock()
    cb.num_timesteps = 2_000_000
    cb._init_callback()

    killed = {"n": 0}
    HungProc = _hung_proc_factory(killed)
    monkeypatch.setattr(ec.subprocess, "Popen", lambda *a, **k: HungProc())

    cb._on_step()                                 # launch; launched_at = now
    assert cb._pending is not None
    cb.num_timesteps = 2_000_001
    cb._on_step()                                 # not overdue → no abort
    assert killed["n"] == 0 and cb._pending is not None


def test_latest_recorded_eval_step():
    from agents.training.eval_callback import latest_recorded_eval_step
    assert latest_recorded_eval_step(None, None) == 0


def test_resume_restores_eval_step_no_immediate_re_eval(tmp_path):
    """Bot path: a resume restores _last_eval_step from metadata so it doesn't re-eval the
    same checkpoint immediately (it waits for the next cadence boundary)."""
    import json
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"latest_eval": {"step": 46_963_120, "win_rate_mean": 0.7, "opponents": {}}}))
    cb = _make_callback(model_dir=str(tmp_path))
    cb._init_callback()
    assert cb._last_eval_step == 46_963_120

    cb.num_timesteps = 46_963_136                 # same 3.5M bucket → no eval
    with patch.object(cb, "_launch_eval") as ml:
        cb._on_step()
        ml.assert_not_called()
    cb.num_timesteps = 49_500_000                 # next boundary → eval
    with patch.object(cb, "_launch_eval") as ml:
        cb._on_step()
        ml.assert_called_once()


def test_replay_last_eval_publishes_to_tui_on_init(tmp_path, monkeypatch):
    """Resume: _init_callback re-publishes the most recent eval to the TUI."""
    import json as _json
    from agents.training import eval_callback as ec
    meta = {"latest_eval": {
        "step": 200, "win_rate_mean": 0.5, "win_rate_vs_bots": 0.4,
        "mean_reward_vs_bots": -1.0, "mean_ep_len_vs_bots": 30.0,
        "opponents": {"random": {"win_rate": 0.7, "mean_reward": 0.1, "mean_ep_len": 25.0}},
    }}
    (tmp_path / "metadata.json").write_text(_json.dumps(meta))

    sent = {}
    monkeypatch.setattr(ec, "send_metrics", lambda d: sent.update(d))
    cb = PerOpponentEvalCallback(model_dir=str(tmp_path), server_config=MagicMock())
    cb._init_callback()

    assert sent.get("eval/win_rate_vs_random") == 0.7
    assert sent.get("eval/win_rate_mean") == 0.5
    assert sent.get("_step") == 200


def test_replay_last_eval_republishes_pool_block(tmp_path, monkeypatch):
    """Resume: the saved self-play pool block (aggregate + per-sentinel rows) is re-published,
    so the Pool/sentinel rows aren't blank until the next cycle — parity with the bot rows."""
    import json as _json
    from agents.training.eval_callback import replay_last_eval_to_tui
    from agents.training import eval_callback as ec
    meta = {"latest_eval": {
        "step": 300, "win_rate_mean": 0.6, "win_rate_vs_bots": 0.55,
        "mean_reward_vs_bots": 5.0, "mean_ep_len_vs_bots": 20.0,
        "elo": 1532.4, "elo_ci": 41.0,
        "opponents": {"random": {"win_rate": 0.9, "mean_reward": 30.0, "mean_ep_len": 15.0}},
        "pool": {
            "win_rate": 0.72, "mean_reward": 13.4, "mean_ep_len": 22.0,
            "monotonicity": 0.4, "snapshot_count": 13,
            "sentinels": [
                {"step": 63_000_000, "win_rate": 0.657, "mean_reward": 8.7, "mean_ep_len": 21.0},
                {"step": 0, "win_rate": 0.707, "mean_reward": 13.2, "mean_ep_len": 23.0},
            ],
        },
    }}
    (tmp_path / "metadata.json").write_text(_json.dumps(meta))

    sent = {}
    monkeypatch.setattr(ec, "send_metrics", lambda d: sent.update(d))
    replay_last_eval_to_tui(str(tmp_path))

    # Pool aggregate — including the reward that used to be missing.
    assert sent.get("eval/win_rate_vs_pool") == 0.72
    assert sent.get("eval/mean_reward_vs_pool") == 13.4
    assert sent.get("eval/sentinel_monotonicity") == 0.4
    assert sent.get("eval/pool_snapshot_count") == 13.0
    # Per-sentinel rows, positional, with their saved step tags (seed = step 0).
    assert sent.get("eval/win_rate_vs_sentinel_0") == 0.657
    assert sent.get("eval/mean_reward_vs_sentinel_0") == 8.7
    assert sent.get("eval/sentinel_step_0") == 63_000_000.0
    assert sent.get("eval/sentinel_step_1") == 0.0
    # Skill rating re-published on resume so the 🏅 badge isn't blank until the next cycle.
    assert sent.get("eval/elo") == 1532.4
    assert sent.get("eval/elo_ci") == 41.0


def test_replay_computes_elo_when_block_predates_field(tmp_path, monkeypatch):
    """Resuming a checkpoint saved BEFORE the elo field: the block has no `elo`, but the
    republish computes it from the block's win rates so the 🏅 badge isn't blank for a full
    cadence. (Robust to the anchor file's presence — value just shifts scale.)"""
    import json as _json
    from agents.training.eval_callback import replay_last_eval_to_tui
    from agents.training import eval_callback as ec
    meta = {"latest_eval": {  # NOTE: no "elo"/"elo_ci" — a pre-feature checkpoint
        "step": 128_000_010, "win_rate_mean": 0.6, "win_rate_vs_bots": 0.55,
        "mean_reward_vs_bots": 5.0, "mean_ep_len_vs_bots": 20.0,
        "opponents": {n: {"win_rate": w, "mean_reward": 0.0, "mean_ep_len": 15.0}
                      for n, w in [("random", 0.99), ("heuristic", 0.77), ("staller", 0.78)]},
        "pool": {"win_rate": 0.71, "mean_reward": 14.0, "mean_ep_len": 22.0,
                 "monotonicity": 0.8, "snapshot_count": 20,
                 "sentinels": [{"step": 126_000_000, "win_rate": 0.71, "mean_reward": 14.0,
                                "mean_ep_len": 21.0}]},
    }}
    (tmp_path / "metadata.json").write_text(_json.dumps(meta))

    sent = {}
    monkeypatch.setattr(ec, "send_metrics", lambda d: sent.update(d))
    replay_last_eval_to_tui(str(tmp_path))

    assert "eval/elo" in sent and math.isfinite(sent["eval/elo"])
    assert sent["eval/elo"] > 1000.0          # a strong model is well above the base
    assert sent.get("eval/elo_ci", -1) >= 0   # CI computed (Z95 * SE)
    # Per-opponent ELO for the eval panel: each bot + each (positional) sentinel.
    assert "eval/elo_vs_random" in sent and math.isfinite(sent["eval/elo_vs_random"])
    assert "eval/elo_vs_heuristic" in sent
    assert "eval/elo_vs_sentinel_0" in sent and math.isfinite(sent["eval/elo_vs_sentinel_0"])


def test_replay_skips_pool_block_when_unseeded(tmp_path, monkeypatch):
    """A pre-seed eval persists an empty sentinels list — don't re-publish a misleading 'vs Pool 0%'."""
    import json as _json
    from agents.training.eval_callback import replay_last_eval_to_tui
    from agents.training import eval_callback as ec
    meta = {"latest_eval": {
        "step": 100, "win_rate_mean": 0.3, "win_rate_vs_bots": 0.25,
        "mean_reward_vs_bots": -2.0, "mean_ep_len_vs_bots": 18.0,
        "opponents": {"random": {"win_rate": 0.5, "mean_reward": 0.0, "mean_ep_len": 12.0}},
        "pool": {"win_rate": 0.0, "mean_reward": 0.0, "mean_ep_len": 0.0,
                 "monotonicity": 1.0, "snapshot_count": 0, "sentinels": []},
    }}
    (tmp_path / "metadata.json").write_text(_json.dumps(meta))

    sent = {}
    monkeypatch.setattr(ec, "send_metrics", lambda d: sent.update(d))
    replay_last_eval_to_tui(str(tmp_path))

    assert "eval/win_rate_vs_pool" not in sent
    assert "eval/win_rate_vs_sentinel_0" not in sent
    assert sent.get("eval/win_rate_vs_random") == 0.5  # bot rows still re-published
