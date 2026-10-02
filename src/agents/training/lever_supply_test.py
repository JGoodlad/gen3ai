"""`gen3_supply_guard_v2` — every OTHER live lever's supply is a declared resource.

Pins the shared guard (`agents.training.lever_supply`), the exit-code mapping of both error classes,
the launcher's response to each code, and each lever's wiring: the self-play POOL and PFSP
(`SelfPlayCallback`) and team-PFSP (`TeamPFSPCallback`). The fork arm is pinned in its own test
files; the real-trainer exits are in `lever_supply_integration_test.py`. (`--bot-weights` and the
warm-start — FATAL_CONFIG, never CRASH — are `main/train/fatal_config_exits_test.py`.)

Every test here FAILS on a revert of the piece it names: a guard that never raises, a callback that
never observes, an error class mapped to CRASH, a streak that resets at a launcher restart.
"""
from __future__ import annotations

import json
import os
import types

import pytest

from agents.training.lever_supply import (LEVERS, DryStreakGuard, LeverConfigError,
                                          LeverStarvedError, parse_starve_overrides,
                                          starve_cycles_for)
from main.exit_codes import FatalConfigError, TrainExitCode, exit_code_for


# -- the guard ---------------------------------------------------------------------------------

def _quiet(_msg):
    pass


def test_a_live_lever_dry_for_its_floor_raises_the_typed_fatal_and_maps_to_FATAL_SUPPLY():
    g = DryStreakGuard("fork", 3, emit=_quiet)
    g.observe(0, why="a")
    g.observe(0, why="b")
    with pytest.raises(LeverStarvedError) as ei:
        g.observe(0, why="the bridge wedged")
    assert exit_code_for(ei.value) == int(TrainExitCode.FATAL_SUPPLY) == 5
    msg = str(ei.value)
    assert "--fork-fraction" in msg and "the bridge wedged" in msg and "fork=3" in msg


def test_a_delivery_resets_the_streak_and_cycles_that_are_not_live_never_count():
    g = DryStreakGuard("pfsp", 2, emit=_quiet)
    g.observe(0)
    g.observe(4)                      # delivered → reset
    g.observe(0)
    for _ in range(10):
        g.observe(0, live=False)      # not supposed to deliver: neither arms nor breaks
    assert g.streak == 1 and g.total == 4 and g.cycles == 13 and g.live_cycles == 3
    with pytest.raises(LeverStarvedError):
        g.observe(0)


def test_zero_disables_the_fatal_but_announces_it_and_the_summary_is_still_loud():
    g = DryStreakGuard("pfsp", 0, emit=_quiet)
    for _ in range(20):
        g.observe(0, why="no traces")
    assert "DISABLED" in g.announce() and "pfsp=0" in g.announce()
    (line,) = g.summary_lines()
    assert line.startswith("🚨🚨 [SUPPLY] ZERO") and "NOT evidence" in line


def test_a_record_only_observation_never_raises():
    g = DryStreakGuard("self_play_pool", 1, emit=_quiet)
    g.observe(0, may_raise=False)
    assert g.streak == 1


def test_the_counters_round_trip_so_a_streak_survives_a_restart():
    g = DryStreakGuard("self_play_pool", 3, emit=_quiet)
    g.observe(0)
    g.observe(0)
    g2 = DryStreakGuard("self_play_pool", 3, state=g.state(), emit=_quiet)
    with pytest.raises(LeverStarvedError):
        g2.observe(0)


def test_a_misconfiguration_maps_to_FATAL_CONFIG():
    assert exit_code_for(LeverConfigError("x")) == int(TrainExitCode.FATAL_CONFIG) == 3
    assert exit_code_for(FatalConfigError("x")) == 3
    # and through a wrap, the way a callback's raise reaches the trainer's handler
    try:
        try:
            raise LeverConfigError("inner")
        except LeverConfigError as e:
            raise RuntimeError("outer") from e
    except RuntimeError as outer:
        assert exit_code_for(outer) == 3
    assert exit_code_for(ValueError("x")) == int(TrainExitCode.CRASH)


def test_the_overrides_parse_and_a_typo_is_refused():
    assert parse_starve_overrides("self_play_pool=5, fork=0") == {"self_play_pool": 5, "fork": 0}
    assert parse_starve_overrides(None) == {}
    for bad in ("nope=3", "fork", "fork=-1", "fork=x"):
        with pytest.raises(ValueError):
            parse_starve_overrides(bad)
    ns = types.SimpleNamespace(supply_starve_cycles="team_pfsp=9")
    assert starve_cycles_for(ns, "team_pfsp") == 9
    assert starve_cycles_for(ns, "fork") == LEVERS["fork"].default_cycles


def test_the_launcher_does_not_restart_a_starved_lever(tmp_path):
    from main.launcher.nonfinite_exit_test import _drive
    code, spawned, _ev = _drive(tmp_path, int(TrainExitCode.FATAL_SUPPLY))
    assert (code, spawned) == (5, 1)


# -- the self-play POOL and PFSP (SelfPlayCallback) --------------------------------------------

class _SummaryPool:
    """A MagicMock pool whose summary.json merges for real — the persistence is under test."""

    def __init__(self, n_sentinels, store):
        from agents.training.selfplay_callback_test import _mock_pool
        self.mock = _mock_pool(n_sentinels=n_sentinels)
        self.mock.is_empty.return_value = n_sentinels == 0
        if n_sentinels == 0:
            self.mock.sentinel_entries.return_value = []
        self.mock.persist_summary.side_effect = lambda **kw: store.update(kw)
        self.mock.load_summary.side_effect = lambda: dict(store)


def _selfplay(tmp_path, monkeypatch, *, store, n_sentinels=0, bot_win=0.40, sentinel_win=0.0,
              pfsp_scale=0.0, popen=None):
    from agents.training import eval_callback as ec
    from agents.training.selfplay_callback import SelfPlayCallback
    from agents.training.selfplay_callback_test import _fake_selfplay_popen
    from unittest.mock import MagicMock
    pool = _SummaryPool(n_sentinels, store).mock
    cb = SelfPlayCallback(pool=pool, model_dir=str(tmp_path), server_config=MagicMock(),
                          showdown_port=9999, n_workers=1, n_sentinels=max(1, n_sentinels),
                          pfsp_scale=pfsp_scale)
    cb.model = MagicMock()
    cb.model.save = lambda base: open(base + ".zip", "w").close()
    cb._logger = MagicMock()
    cb.model.get_env.return_value.env_method.return_value = [(100, 5, 2)]
    cb.num_timesteps = 0
    cb._init_callback()
    monkeypatch.setattr(ec.subprocess, "Popen", popen or _fake_selfplay_popen(
        ec, bot_win=bot_win, sentinel_win=sentinel_win))
    return cb, pool


def _cycle(cb, k):
    """Launch + collect eval cycle k (the fake workers finish at once)."""
    cb.num_timesteps = 2_000_000 * k + 1
    cb._on_step()                 # launch
    cb._on_step()                 # collect


def test_an_EMPTY_pool_for_its_floor_of_eval_cycles_is_FATAL_SUPPLY(tmp_path, monkeypatch):
    """ai_v12_27_ladder_ctrl10M_shaped_dense: win_rate_vs_bots ended at 0.53 < the 0.55 seeding
    gate, so `--self-play` trained against the BOT fallback for all 10M steps."""
    store = {}
    cb, pool = _selfplay(tmp_path, monkeypatch, store=store, bot_win=0.40)
    _cycle(cb, 1)
    _cycle(cb, 2)
    with pytest.raises(LeverStarvedError) as ei:
        _cycle(cb, 3)
    assert exit_code_for(ei.value) == 5
    assert "--self-play" in str(ei.value) and "--self-play-start-wr" in str(ei.value)
    assert store["supply_guard"]["self_play_pool"]["streak"] == 3, "persisted BEFORE the raise"
    pool.add_from_path.assert_not_called()


def test_the_pool_streak_is_RUN_level_it_survives_a_launcher_restart(tmp_path, monkeypatch):
    store = {}
    cb, _ = _selfplay(tmp_path, monkeypatch, store=store)
    _cycle(cb, 1)
    _cycle(cb, 2)
    cb2, _ = _selfplay(tmp_path, monkeypatch, store=store)     # a new segment, same run dir
    with pytest.raises(LeverStarvedError):
        _cycle(cb2, 3)


def test_a_FORKS_inherited_parent_counters_are_not_its_own(tmp_path, monkeypatch):
    store = {"supply_guard": {"run": "/some/parent/run",
                              "self_play_pool": {"streak": 2, "cycles": 2}}}
    cb, _ = _selfplay(tmp_path, monkeypatch, store=store)
    _cycle(cb, 1)
    assert cb._pool_guard.streak == 1


def test_FAILED_eval_cycles_count_toward_the_floor(tmp_path, monkeypatch):
    class _FailProc:
        returncode = 1

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    store = {}
    cb, _ = _selfplay(tmp_path, monkeypatch, store=store, bot_win=0.9,
                      popen=lambda *a, **k: _FailProc())
    _cycle(cb, 1)
    _cycle(cb, 2)
    with pytest.raises(LeverStarvedError) as ei:
        _cycle(cb, 3)
    assert "FAILED" in str(ei.value) and "Failed eval cycles so far: 3" in str(ei.value)
    assert store["supply_guard"]["failed_eval_cycles"] == 3


def test_a_seeded_pool_never_trips_and_the_summary_says_so(tmp_path, monkeypatch):
    store = {}
    cb, pool = _selfplay(tmp_path, monkeypatch, store=store, n_sentinels=3, bot_win=0.8,
                         sentinel_win=0.5)
    for k in range(1, 6):
        _cycle(cb, k)
    assert cb._pool_guard.streak == 0
    assert all("ZERO" not in line for line in cb.supply_summary_lines())


def test_a_graceful_drain_records_but_never_turns_a_finished_run_into_a_FATAL(tmp_path,
                                                                              monkeypatch, capsys):
    store = {}
    cb, _ = _selfplay(tmp_path, monkeypatch, store=store)
    _cycle(cb, 1)
    _cycle(cb, 2)
    cb.num_timesteps = 6_000_001
    cb._on_step()                 # launch cycle 3 …
    cb._on_training_end()         # … and training ends: the drain collects it
    assert cb._pool_guard.streak == 3
    assert "🚨🚨 [SUPPLY] ZERO" in capsys.readouterr().out


def test_PFSP_with_sentinels_but_no_measured_win_rate_is_FATAL(tmp_path, monkeypatch):
    from agents.training.selfplay_callback_test import _publish_fake_selfplay_shards
    from agents.training.eval_sharding import BOT

    class _P:
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def popen(argv, **_kw):
        _publish_fake_selfplay_shards(json.load(open(argv[-1])), bot_win=0.8, kinds={BOT})
        return _P()

    store = {}
    cb, _ = _selfplay(tmp_path, monkeypatch, store=store, n_sentinels=3, pfsp_scale=1.0,
                      popen=popen)
    _cycle(cb, 1)
    _cycle(cb, 2)
    with pytest.raises(LeverStarvedError) as ei:
        _cycle(cb, 3)
    assert "--pfsp-scale" in str(ei.value)
    assert cb._pool_guard.streak == 0, "the pool is seeded — only PFSP starved"


# -- team-PFSP ---------------------------------------------------------------------------------

def _team_cb(drain, live, starve=2):
    from agents.training.team_pfsp_callback import TeamPFSPCallback
    from agents.training.team_pfsp_callback_test import _FakeLogger, _FakeVecEnv
    env = _FakeVecEnv(drain)
    cb = TeamPFSPCallback(cap=3.0, floor=0.05, update_every=1, mode="var",
                          live_probe=live, starve_cycles=starve, emit=_quiet)
    cb.model = types.SimpleNamespace(logger=_FakeLogger(), get_env=lambda: env)
    return cb


def test_team_PFSP_with_live_self_play_and_no_team_game_is_FATAL():
    cb = _team_cb([([0.0, 0.0], [0.0, 0.0], 2)], live=lambda: True)
    cb._on_rollout_end()
    with pytest.raises(LeverStarvedError) as ei:
        cb._on_rollout_end()
    assert "--team-pfsp" in str(ei.value)


def test_team_PFSP_is_not_judged_while_self_play_is_not_live():
    cb = _team_cb([([0.0, 0.0], [0.0, 0.0], 2)], live=lambda: False)
    for _ in range(10):
        cb._on_rollout_end()
    assert cb._guard.streak == 0 and cb._guard.cycles == 10


def test_team_PFSP_games_keep_it_alive():
    cb = _team_cb([([1.0, 0.0], [2.0, 1.0], 2)], live=lambda: True)
    for _ in range(10):
        cb._on_rollout_end()
    assert cb._guard.total == 30


def test_the_live_probe_reads_the_pool_and_the_curriculum_fraction():
    from unittest.mock import MagicMock
    from main.train.callbacks import team_pfsp_live_probe
    assert team_pfsp_live_probe(types.SimpleNamespace(exploiter="models/t"), None) is None
    pool = MagicMock()
    pool.is_empty.return_value = True
    probe = team_pfsp_live_probe(types.SimpleNamespace(exploiter=None), pool)
    assert probe() is False
    pool.is_empty.return_value = False
    pool.load_summary.return_value = {"self_play_fraction": 0.0}
    assert probe() is False
    pool.load_summary.return_value = {"self_play_fraction": 0.4}
    assert probe() is True


def test_every_lever_names_a_real_flag():
    from main.train.parser import build_parser
    opts = {o for a in build_parser()._actions for o in a.option_strings}
    for lv in LEVERS.values():
        assert lv.flag in opts, lv.flag
    assert "--supply-starve-cycles" in opts
    assert os.path.basename(__file__) == "lever_supply_test.py"
