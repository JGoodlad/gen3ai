"""The CLI plumbing — what the tool REFUSES, and what a plan resolves to before anything starts.

Four refusals are the point of this file, because each corresponds to a way an anchor read has
already produced a number nobody could place:

1. **a reserved port** (8000 dev / 8001 the live training run) on either the `--port` path or the
   `--server-uri` path;
2. **an unmatched regime** — `--regime t1` against Foul Play, which is a search bot with no
   sampling knob, so only OUR side would move. The 2026-09-14 battery exists because a mixed-regime
   read was quoted as if it were a matched one;
3. **an unpinned Metamon agent** — a policy is an anchor only once its checkpoint is in
   `designs/ops/anchors.json`;
4. **a real read with no `--model`** — only `--dry-run` / `--show-config` work without one.

Everything else here pins that `--dry-run` prints a plan a human can execute and a machine can
diff: both peer commands verbatim, both team sources with their counts, and every deadline.
"""
from __future__ import annotations

import pytest

from main.anchors import runner as runner_mod
from main.anchors.cli import build_parser, build_plan, parse_opponent, render_plan
from main.anchors.config import load_config
from main.anchors.server import RESERVED_PORTS, ServerError


@pytest.fixture()
def cfg():
    return load_config()


def _args(*argv: str):
    return build_parser().parse_args(["--model", "", *argv])


# -------------------------------------------------------------------------------- opponent names
@pytest.mark.parametrize("spec, kind, agent", [
    ("metamon:SmallRL", "metamon", "SmallRL"),
    ("metamon:SyntheticRLV2", "metamon", "SyntheticRLV2"),
    ("foulplay", "foulplay", ""),
    ("FoulPlay", "foulplay", ""),
])
def test_opponent_names_parse(spec: str, kind: str, agent: str) -> None:
    assert parse_opponent(spec) == (kind, agent)


@pytest.mark.parametrize("spec", ["metamon", "kakuna:Kakuna", "", "showdown:bot"])
def test_an_unknown_or_agentless_opponent_is_refused(spec: str) -> None:
    with pytest.raises(SystemExit):
        parse_opponent(spec)


def test_an_unpinned_metamon_agent_is_refused_at_plan_time(cfg) -> None:
    """REFUSAL 3. The refusal must come from the CONFIG, not from a hardcoded list here."""
    from main.anchors.config import AnchorConfigError

    plan = build_plan(_args("--opponent", "metamon:Kakuna", "--dry-run"), cfg)
    with pytest.raises(AnchorConfigError) as excinfo:
        runner_mod.peer_plan(plan, cfg, "acceptor", 2, "ours_challenge")
    assert "Kakuna" in str(excinfo.value)


# ------------------------------------------------------------------------------- reserved ports
@pytest.mark.parametrize("port", sorted(RESERVED_PORTS))
def test_a_reserved_port_is_refused_on_the_port_flag(cfg, port: int) -> None:
    with pytest.raises(ServerError) as excinfo:
        build_plan(_args("--port", str(port)), cfg)
    assert RESERVED_PORTS[port] in str(excinfo.value)


@pytest.mark.parametrize("port", sorted(RESERVED_PORTS))
def test_a_reserved_port_is_refused_through_the_server_uri_too(cfg, port: int) -> None:
    """The `--server-uri` seam (which the websocket front end will use) is not a way around it."""
    with pytest.raises(ServerError):
        build_plan(_args("--server-uri", f"ws://localhost:{port}/showdown/websocket"), cfg)


def test_an_auto_picked_port_is_in_the_9xxx_range_and_the_tool_owns_it(cfg) -> None:
    plan = build_plan(_args(), cfg)
    assert 9500 <= plan.server_port <= 9599
    assert plan.started_server is True
    assert plan.server_uri == f"ws://localhost:{plan.server_port}/showdown/websocket"


def test_a_given_server_uri_starts_nothing(cfg) -> None:
    """The seam the in-repo websocket front end over the Rust bridge will plug into."""
    plan = build_plan(_args("--server-uri", "ws://127.0.0.1:9543/showdown/websocket"), cfg)
    assert plan.started_server is False
    assert plan.server_port == 9543


# -------------------------------------------------------------------------------- the regime
def test_greedy_sets_our_temperature_to_zero_and_t1_to_one(cfg) -> None:
    """Our half of the regime, read out of the argv `main.play`'s OWN parser receives."""
    greedy = runner_mod.our_argv(build_plan(_args("--regime", "greedy"), cfg), "challenge", 4)
    t1 = runner_mod.our_argv(build_plan(_args("--regime", "t1"), cfg), "challenge", 4)
    assert greedy[greedy.index("--temperature") + 1] == "0.0"
    assert t1[t1.index("--temperature") + 1] == "1.0"


def test_our_argv_parses_cleanly_through_plays_own_parser(cfg) -> None:
    """The whole point of reusing `play.py`: if this argv did not parse, the tool would be running
    a client of its own invention."""
    import main.play as play

    plan = build_plan(_args("--regime", "greedy"), cfg)
    args = play.build_parser().parse_args(runner_mod.our_argv(plan, "accept", 7))
    assert args.mode == "accept"
    assert args.n_battles == 7
    assert args.temperature == 0.0
    assert args.concurrency == 1
    # The forfeit limit is the TRAINER's number, inherited and not restated.
    assert args.forfeit_turn_limit == play.DEFAULT_FORFEIT_TURN_LIMIT == 250


def test_t1_against_foulplay_is_refused_because_only_one_side_would_move(cfg) -> None:
    """REFUSAL 2. Foul Play searches; it has no temperature and no sampling knob."""
    with pytest.raises(SystemExit) as excinfo:
        build_plan(_args("--opponent", "foulplay", "--regime", "t1"), cfg)
    message = str(excinfo.value)
    assert "NOT a matched regime" in message
    assert "--allow-unmatched-regime" in message


def test_the_escape_hatch_stamps_every_row_unmatched_rather_than_hiding_it(cfg) -> None:
    plan = build_plan(
        _args("--opponent", "foulplay", "--regime", "t1", "--allow-unmatched-regime"), cfg)
    assert plan.regime_matched is False


def test_a_matched_greedy_cell_says_so(cfg) -> None:
    assert build_plan(_args("--opponent", "foulplay"), cfg).regime_matched is True
    assert build_plan(_args("--opponent", "metamon:SmallRL", "--regime", "t1"),
                      cfg).regime_matched is True


# --------------------------------------------------------------------------- roles and budgets
@pytest.mark.parametrize("games, expect", [
    (100, {"ours_challenge": 50, "peer_challenge": 50}),
    (2, {"ours_challenge": 1, "peer_challenge": 1}),
    (7, {"ours_challenge": 4, "peer_challenge": 3}),
    (1, {"ours_challenge": 1, "peer_challenge": 0}),
])
def test_games_are_role_balanced_and_an_odd_game_is_not_dropped(cfg, games, expect) -> None:
    """Showdown makes the CHALLENGER p1, and p1/p2 is not a priori neutral — so role is balanced
    INSIDE every read rather than confounded with one."""
    assert build_plan(_args("--games", str(games)), cfg).half_sizes() == expect


def test_the_search_budget_is_carried_only_for_the_opponent_that_has_one(cfg) -> None:
    metamon = build_plan(_args("--opponent", "metamon:SmallRL"), cfg)
    foulplay = build_plan(_args("--opponent", "foulplay", "--search-time-ms", "2500"), cfg)
    assert metamon.search_time_ms is None, "Metamon has no search budget; None, never a stray 1000"
    assert foulplay.search_time_ms == 2500


def test_the_default_peer_username_is_inside_showdowns_18_character_ceiling(cfg) -> None:
    """A 19-character name gets a ';'-prefixed REFUSAL from action.php that Foul Play accepts as
    an assertion and then hangs on forever (foul-play de-risk H1)."""
    from main.anchors.peers import MAX_USERNAME_LEN, check_username

    for opponent in ("metamon:SmallRL", "metamon:SyntheticRLV2", "foulplay"):
        plan = build_plan(_args("--opponent", opponent), cfg)
        assert len(plan.peer_username) <= MAX_USERNAME_LEN
        check_username(plan.peer_username, "peer")
        check_username(plan.our_username, "our")


# ------------------------------------------------------------------------------------ dry run
def test_dry_run_prints_a_plan_carrying_the_regime_the_teams_and_both_commands(cfg) -> None:
    plan = build_plan(_args("--opponent", "metamon:SmallRL", "--teamset", "away",
                            "--games", "4"), cfg)
    text = render_plan(plan, cfg)
    assert "nothing has been started" in text
    assert "regime            greedy" in text
    assert "team set          away" in text
    assert "forfeit limit     250 turns" in text
    # Both halves, both peer commands, and the peer's team source with its COUNT.
    assert text.count("--- half ") == 2
    assert "metamon_side.py" in text
    assert "--regime greedy" in text
    assert "peer teams:" in text
    assert "deadlines" in text


def test_dry_run_on_foulplay_states_that_the_budget_is_wall_clock(cfg) -> None:
    text = render_plan(build_plan(_args("--opponent", "foulplay"), cfg), cfg)
    assert "WALL CLOCK" in text
    assert "--search-time-ms 1000" in text


def test_a_real_read_without_a_model_is_refused(cfg) -> None:
    """REFUSAL 4 — and `--dry-run` must still work, which is what makes the tool readable."""
    from main.anchors.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--opponent", "metamon:SmallRL", "--games", "2"])
    assert "--model is required" in str(excinfo.value)
    assert main(["--opponent", "metamon:SmallRL", "--games", "2", "--dry-run"]) == 0
    assert main(["--show-config"]) == 0
