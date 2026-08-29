"""The driver CLI's two non-obvious knobs: the arm DEFAULT and the per-battle time bounds."""

from __future__ import annotations

import pytest

from main.search_dividend.__main__ import DEFAULT_ARMS, _raise_battle_backstop, build_parser
from main.search_dividend.defensive import DefensiveConfig
from main.search_dividend.search import ARMS, ROOT_STRATEGIES, SearchConfig


def test_playoff_is_selectable_but_not_a_default_arm():
    """A flagless run must play what it always played. ``playoff`` costs orders of magnitude more
    than a critic sweep (it plays whole battles inside a decision), so adding it to ``ARMS`` must
    not silently change every existing invocation into a different, much longer experiment."""
    assert "playoff" in ARMS
    assert "playoff" not in DEFAULT_ARMS
    assert set(DEFAULT_ARMS) < set(ARMS)
    assert build_parser().parse_args(["m", "--arm", "playoff"]).arm == ["playoff"]


def test_the_playoff_knobs_default_to_the_registered_values():
    a = build_parser().parse_args(["m", "--arm", "playoff"])
    assert (a.playoff_rollouts, a.playoff_se_k, a.playoff_min_pairs) == (12, 2.0, 4)
    assert a.battle_timeout_s is None and a.battle_idle_s is None


def test_raising_the_battle_bounds_patches_BOTH_and_only_when_asked():
    """The two bounds answer different questions and the playoff arm breaks both assumptions — a
    nested rollout silences the live stream (idle) and a 25-decision game outruns 180 s (total).
    Raising one without the other still loses games, and a lost game poisons the rest of the cell.
    """
    from utils.bridge import local_battle_runner as lbr

    total0, idle0 = lbr._PER_BATTLE_TIMEOUT, lbr._BATTLE_IDLE_BUDGET
    try:
        _raise_battle_backstop(None, None)
        assert (lbr._PER_BATTLE_TIMEOUT, lbr._BATTLE_IDLE_BUDGET) == (total0, idle0)
        _raise_battle_backstop(5400.0, 120.0)
        assert lbr._PER_BATTLE_TIMEOUT == pytest.approx(5400.0)
        assert lbr._BATTLE_IDLE_BUDGET == pytest.approx(120.0)
    finally:
        lbr._PER_BATTLE_TIMEOUT, lbr._BATTLE_IDLE_BUDGET = total0, idle0


def test_defensive_is_selectable_and_grid_is_still_what_a_flagless_run_gets():
    p = build_parser()
    assert "defensive" in ROOT_STRATEGIES
    assert p.parse_args(["m"]).root_strategy == "grid"
    assert p.parse_args(["m", "--root-strategy", "defensive"]).root_strategy == "defensive"


def test_the_defensive_knobs_default_to_the_probe_measured_operating_point():
    a = build_parser().parse_args(["m", "--root-strategy", "defensive"])
    assert a.defensive_leaf == "winprob"          # probe G: the value head does not clear zero
    assert a.defensive_wp_margin == 0.15          # probe H's chosen frontier point
    assert a.defensive_confirm == 0               # one new mechanism at a time in the first cell
    # UNSET, not a number: the flagless default must reproduce the first registered cell.
    assert a.defensive_contested_deadline_s is None


def test_every_defensive_flag_reaches_the_config_it_names():
    """The CLI-to-config hop is where a flag goes quietly inert (the `train` CLI's unlaunchable
    edge family), so the whole triple is asserted on the object the engine actually reads."""
    a = build_parser().parse_args(["m", "--root-strategy", "defensive",
                                   "--defensive-leaf", "value",
                                   "--defensive-wp-margin", "0.3",
                                   "--defensive-confirm", "6",
                                   "--defensive-contested-deadline-s", "3"])
    cfg = SearchConfig(root_strategy=a.root_strategy, budget_s=1.0,
                       defensive=DefensiveConfig(
                           wp_margin=a.defensive_wp_margin, leaf=a.defensive_leaf,
                           confirm_rollouts=a.defensive_confirm,
                           contested_deadline_s=a.defensive_contested_deadline_s))
    assert cfg.defensive_cfg() == DefensiveConfig(wp_margin=0.3, leaf="value",
                                                  confirm_rollouts=6,
                                                  contested_deadline_s=3.0)
    assert cfg.effective_score() == "value"
    assert cfg.contested_budget_s() == 3.0


def test_the_contested_deadline_flag_survives_the_hop_the_cell_actually_takes():
    """The mirror opponent is built by ``dataclasses.replace(cfg, arm='base', budget_s=0.0)``, so
    a knob that only worked on the searched side would still read correctly here while the
    UNSEARCHED side quietly acquired it. ``arm='base'`` short-circuits before the gate, so it must
    resolve to no search at all rather than a 3 s one."""
    from dataclasses import replace

    cfg = SearchConfig(arm="honest", root_strategy="defensive", budget_s=1.0,
                       defensive=DefensiveConfig(contested_deadline_s=3.0))
    mirror = replace(cfg, arm="base", budget_s=0.0)
    assert mirror.resolved_caps().k_worlds == 0        # the base arm never reaches the clock


def test_games_start_defaults_to_zero_and_shards_a_half_open_window():
    p = build_parser()
    assert p.parse_args(["m"]).games_start == 0
    a = p.parse_args(["m", "--games-start", "267", "--games", "266"])
    assert (a.games_start, a.games) == (267, 266)


def test_the_parser_refuses_a_leaf_it_does_not_know():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["m", "--defensive-leaf", "scalar_v"])


def test_every_help_string_renders():
    """One unescaped ``%`` turns a help string into a format conversion and `--help` raises. That
    is a shipped failure in this project's history (`checkargs_test`'s guard); the defensive flags
    quote several percentages, so the parser is rendered here rather than assumed."""
    assert "defensive" in build_parser().format_help()


def test_the_idle_bound_is_read_at_CALL_time_so_patching_it_takes_effect():
    """A module-level constant captured at import would make the flag a no-op that looks like it
    worked. ``_await_battle`` reads the global when it builds its ``ProgressDeadline``."""
    import inspect

    from utils.bridge import local_battle_runner as lbr

    src = inspect.getsource(lbr._await_battle)
    assert "ProgressDeadline(_BATTLE_IDLE_BUDGET" in src
    assert "total_budget_s=_PER_BATTLE_TIMEOUT" in src
