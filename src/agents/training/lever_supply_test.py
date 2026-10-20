"""`gen3_supply_guard_v2` — every OTHER live lever's supply is a declared resource.

Pins the shared guard (`agents.training.lever_supply`), the exit-code mapping of both error classes,
the launcher's response to each code, and each lever's wiring: the self-play POOL and PFSP
(`SelfPlayCallback`). (Team-PFSP's guard was deleted with the lever, deletion pass L4.) The fork arm is pinned in its own test
files; the real-trainer exits are in `lever_supply_integration_test.py`. (`--bot-weights` and the
warm-start — FATAL_CONFIG, never CRASH — are `main/train/fatal_config_exits_test.py`.)

Every test here FAILS on a revert of the piece it names: a guard that never raises, a callback that
never observes, an error class mapped to CRASH, a streak that resets at a launcher restart.
"""
from __future__ import annotations

import os
import types

import pytest

from agents.training.lever_supply import (LEVERS, DryStreakGuard, LeverConfigError,
                                          LeverStarvedError, parse_starve_overrides,
                                          starve_cycles_for)
from main.exit_codes import FatalConfigError, TrainExitCode, exit_code_for
from utils.model_save_testkit import fake_save


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
    ns = types.SimpleNamespace(supply_starve_cycles="pfsp=9")
    assert starve_cycles_for(ns, "pfsp") == 9
    with pytest.raises(ValueError):
        parse_starve_overrides("team_pfsp=3")        # the deleted lever's key is a typo now
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
              pfsp_scale=0.0, fail=False, kinds=None):
    from agents.training.selfplay_callback import SelfPlayCallback
    from agents.training.selfplay_callback_test import attach_fake_evaluator
    from unittest.mock import MagicMock
    pool = _SummaryPool(n_sentinels, store).mock
    cb = SelfPlayCallback(pool=pool, model_dir=str(tmp_path), n_sentinels=max(1, n_sentinels),
                          pfsp_scale=pfsp_scale)
    cb.model = MagicMock()
    cb.model.save = fake_save
    cb.model.gamma = 0.99
    cb._logger = MagicMock()
    cb.model.get_env.return_value.env_method.return_value = [(100, 5, 2)]
    cb.num_timesteps = 0
    cb._init_callback()
    attach_fake_evaluator(cb, monkeypatch, bot_win=bot_win, sentinel_win=sentinel_win, fail=fail, kinds=kinds)
    return cb, pool


def _cycle(cb, k):
    """Launch + collect eval cycle k (the cycle plays blocking, in process, in one step)."""
    cb.num_timesteps = 2_000_000 * k + 1
    cb._on_step()


def test_a_pool_that_STAYS_empty_after_the_gate_opened_for_its_floor_of_cycles_is_FATAL_SUPPLY(tmp_path, monkeypatch):
    """The supply the guard exists for: the curriculum asked for a seed (win rate 0.9 over the 0.55 gate) and the
    pool is STILL empty after the cycle — here the mock pool swallows `add_from_path` — three cycles running.
    (A fresh run still BELOW the gate is not this: next test, `gen3_supply_guard_v3`.)"""
    store = {}
    cb, pool = _selfplay(tmp_path, monkeypatch, store=store, bot_win=0.9)
    _cycle(cb, 1)
    _cycle(cb, 2)
    with pytest.raises(LeverStarvedError) as ei:
        _cycle(cb, 3)
    assert exit_code_for(ei.value) == 5
    assert "--self-play" in str(ei.value) and "--self-play-start-wr" in str(ei.value)
    assert "cleared the seeding gate" in str(ei.value)
    assert store["supply_guard"]["self_play_pool"]["streak"] == 3, "persisted BEFORE the raise"
    assert pool.add_from_path.call_count == 3, "each cycle DID try to seed it"


def test_a_fresh_run_still_BELOW_the_seeding_gate_is_never_FATAL_but_says_so_every_cycle(tmp_path, monkeypatch, capsys):
    """backlog row "the supply guard counts a NEVER-SEEDED pool as starving": a from-scratch run whose win rate
    vs bots (0.40) has not reached `--self-play-start-wr` (0.55) legitimately trains on bots with an empty pool —
    `lever_supply`'s own contract (live=False). SIX cycles (twice the floor of 3) must not raise; the cycles are
    counted and reported, and nothing is silent: a loud line per cycle, the gated-cycle scalar, the ZERO summary."""
    store = {}
    cb, pool = _selfplay(tmp_path, monkeypatch, store=store, bot_win=0.40)
    for k in range(1, 7):
        _cycle(cb, k)                                    # would be FATAL at cycle 3 if the cycles counted
    g = cb._pool_guard
    assert (g.streak, g.cycles, g.live_cycles, g.dry_cycles) == (0, 6, 0, 0)
    assert store["supply_guard"]["self_play_pool"]["cycles"] == 6, "the gated cycles are persisted, run-level"
    pool.add_from_path.assert_not_called()               # the curriculum asked for no seed
    out = capsys.readouterr().out
    assert out.count("below the seeding gate --self-play-start-wr 0.55") == 6, out
    assert "6 gated cycle(s) so far" in out
    cb.model.logger.record.assert_any_call("supply/selfplay_pool_gated_cycles", 6.0)
    cb._on_training_end()
    assert "🚨🚨 [SUPPLY] ZERO" in capsys.readouterr().out, "a bots-only segment is still said to be no evidence about self-play"


def test_the_gated_cycles_before_the_gate_opened_do_not_pre_arm_the_floor(tmp_path, monkeypatch):
    """Two cycles below the gate, then the gate opens (0.9) and the seed fails to land: the floor of 3 counts from
    the FIRST gate-open cycle — FATAL at cycle 5, not at cycle 3."""
    from agents.training.selfplay_callback_test import attach_fake_evaluator

    store = {}
    cb, _ = _selfplay(tmp_path, monkeypatch, store=store, bot_win=0.40)
    _cycle(cb, 1)
    _cycle(cb, 2)
    assert cb._pool_guard.streak == 0
    attach_fake_evaluator(cb, monkeypatch, bot_win=0.9, sentinel_win=0.0)
    _cycle(cb, 3)
    _cycle(cb, 4)
    assert cb._pool_guard.streak == 2
    with pytest.raises(LeverStarvedError):
        _cycle(cb, 5)
    assert (cb._pool_guard.cycles, cb._pool_guard.live_cycles) == (5, 3)


def test_the_pool_streak_is_RUN_level_it_survives_a_launcher_restart(tmp_path, monkeypatch):
    store = {}
    cb, _ = _selfplay(tmp_path, monkeypatch, store=store, bot_win=0.9)
    _cycle(cb, 1)
    _cycle(cb, 2)
    cb2, _ = _selfplay(tmp_path, monkeypatch, store=store, bot_win=0.9)     # a new segment, same run dir
    with pytest.raises(LeverStarvedError):
        _cycle(cb2, 3)


def test_a_FORKS_inherited_parent_counters_are_not_its_own(tmp_path, monkeypatch):
    store = {"supply_guard": {"run": "/some/parent/run",
                              "self_play_pool": {"streak": 2, "cycles": 2}}}
    cb, _ = _selfplay(tmp_path, monkeypatch, store=store, bot_win=0.9)
    _cycle(cb, 1)
    assert cb._pool_guard.streak == 1


def test_FAILED_eval_cycles_count_toward_the_floor(tmp_path, monkeypatch):
    store = {}
    cb, _ = _selfplay(tmp_path, monkeypatch, store=store, bot_win=0.9, fail=True)
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


def test_the_end_of_segment_summary_is_loud_at_zero_and_raises_nothing(tmp_path, monkeypatch, capsys):
    """Training ending is not a judge: `_on_training_end` only PRINTS the supply summary (there is no
    in-flight cycle to collect — the cycle is blocking), so a finished run is never turned into a FATAL
    from inside its own shutdown path."""
    store = {}
    cb, _ = _selfplay(tmp_path, monkeypatch, store=store, bot_win=0.9)
    _cycle(cb, 1)
    _cycle(cb, 2)                 # the gate is open and the pool still empty after two cycles (floor 3): no FATAL yet
    cb._on_training_end()
    assert cb._pool_guard.streak == 2
    assert "🚨🚨 [SUPPLY] ZERO" in capsys.readouterr().out


def test_PFSP_with_sentinels_but_no_measured_win_rate_is_FATAL(tmp_path, monkeypatch):
    from agents.training.eval_sharding import BOT

    store = {}
    cb, _ = _selfplay(tmp_path, monkeypatch, store=store, n_sentinels=3, pfsp_scale=1.0,
                      bot_win=0.8, kinds={BOT})
    _cycle(cb, 1)
    _cycle(cb, 2)
    with pytest.raises(LeverStarvedError) as ei:
        _cycle(cb, 3)
    assert "--pfsp-scale" in str(ei.value)
    assert cb._pool_guard.streak == 0, "the pool is seeded — only PFSP starved"


def test_every_lever_names_a_real_flag():
    from main.train.parser import build_parser
    opts = {o for a in build_parser()._actions for o in a.option_strings}
    for lv in LEVERS.values():
        assert lv.flag in opts, lv.flag
    assert "--supply-starve-cycles" in opts
    assert os.path.basename(__file__) == "lever_supply_test.py"
