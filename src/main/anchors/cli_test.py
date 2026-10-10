"""The CLI plumbing — what the tool REFUSES, and what a plan resolves to before anything starts.

Four refusals are the point of this file, because each corresponds to a way an anchor read has
already produced a number nobody could place:

1. **a reserved port** (8000 dev / 8001 the reserved training-tools server) on either the `--port` path or the
   `--server-uri` path;
2. **an unmatched regime** — `--regime t1` against Foul Play, which is a search bot with no
   sampling knob, so only OUR side would move. The 2026-09-14 battery exists because a mixed-regime
   read was quoted as if it were a matched one;
3. **a `--seed-base` / `--capture-dir` on a transport that cannot honour it** — both belong to
   the websocket FRONT END, and a seed the Node server silently ignores would make an
   unrepeatable series look seeded;
4. **an unpinned Metamon agent** — a policy is an anchor only once its checkpoint is in
   `designs/ops/anchors.json`;
5. **a real read with no `--model`** — only `--dry-run` / `--show-config` work without one.

Everything else here pins that `--dry-run` prints a plan a human can execute and a machine can
diff: both peer commands verbatim, both team sources with their counts, and every deadline.
"""
from __future__ import annotations

import pytest

from main.anchors import cli as cli_mod
from main.anchors import runner as runner_mod
from main.anchors import server as server_mod
from main.anchors.cli import build_parser, build_plan, parse_opponent, render_plan, showdown_pin
from main.anchors.config import load_config
from main.anchors.results import REQUIRED_ROW_FIELDS
from main.anchors.server import RESERVED_PORTS, ServerError
from utils.paths import repo_root


@pytest.fixture()
def cfg():
    return load_config()


def _args(*argv: str):
    return build_parser().parse_args(["--model", "", *argv])


# -------------------------------------------------------------------------------- opponent names
@pytest.mark.parametrize("spec, kind, agent", [
    ("metamon:SmallRL", "metamon", "SmallRL"),
    ("metamon:SyntheticRLV2", "metamon", "SyntheticRLV2"),
    ("metamon:Kakuna", "metamon", "Kakuna"),
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
    """REFUSAL 4. The refusal must come from the CONFIG, not from a hardcoded list here."""
    from main.anchors.config import AnchorConfigError

    plan = build_plan(_args("--opponent", "metamon:NotAMetamonAgent", "--dry-run"), cfg)
    with pytest.raises(AnchorConfigError) as excinfo:
        runner_mod.peer_plan(plan, cfg, "acceptor", 2, "ours_challenge")
    assert "NotAMetamonAgent" in str(excinfo.value)


def test_kakuna_plans_with_its_pinned_checkpoint(cfg) -> None:
    """X22(a): `metamon:Kakuna` is a pinned anchor — the peer command carries the checkpoint from
    the committed config (34, upstream's own default), never the hub's latest epoch."""
    plan = build_plan(_args("--opponent", "metamon:Kakuna", "--dry-run"), cfg)
    theirs = runner_mod.peer_plan(plan, cfg, "acceptor", 2, "ours_challenge")
    line = theirs.command_line()
    assert "--agent Kakuna" in line and "--checkpoint 34" in line
    assert theirs.version == "Kakuna@ckpt34"


def test_opponent_temperature_samples_only_the_metamon_peer_and_is_stamped_unmatched(cfg) -> None:
    """X22(b): `--opponent-temperature T` — the peer runs `t1` at action_temperature T, OUR side
    stays `--temperature 0.0`, and the cell says regime_matched=false with their_regime naming T."""
    plan = build_plan(_args("--opponent", "metamon:Kakuna", "--opponent-temperature", "0.5",
                            "--dry-run"), cfg)
    assert plan.regime == "greedy" and plan.regime_matched is False
    assert plan.opponent_temperature == 0.5
    theirs = runner_mod.peer_plan(plan, cfg, "acceptor", 2, "ours_challenge")
    line = theirs.command_line()
    assert "--regime t1" in line and "--temperature 0.5" in line
    assert theirs.their_regime == "sample:T=0.5"
    assert cli_mod.core_side.our_stochastic(plan) == (False, 1.0)       # OUR side stays greedy
    # without the flag the peer command carries no temperature at all (greedy ignores it, H7)
    plain = build_plan(_args("--opponent", "metamon:Kakuna", "--dry-run"), cfg)
    assert "--temperature" not in runner_mod.peer_plan(
        plain, cfg, "acceptor", 2, "ours_challenge").command_line()


@pytest.mark.parametrize("argv", [
    ("--opponent", "foulplay", "--opponent-temperature", "0.5"),
    ("--opponent", "metamon:Kakuna", "--regime", "t1", "--opponent-temperature", "0.5"),
    ("--opponent", "metamon:Kakuna", "--opponent-temperature", "0"),
    ("--opponent", "metamon:Kakuna", "--our-side", "bot:random", "--opponent-temperature", "0.5"),
])
def test_opponent_temperature_is_refused_outside_its_one_shape(cfg, argv) -> None:
    with pytest.raises(SystemExit):
        build_plan(_args(*argv, "--dry-run"), cfg)



def test_our_temperature_samples_OUR_checkpoint_and_is_stamped_on_the_cell(cfg) -> None:
    """X22(f): `--our-temperature T` — our side samples at T, the peer follows
    --regime greedy (or --opponent-temperature), our_regime names T, and the cell is MATCHED only
    when both nominal temperatures are equal."""
    vs_greedy = build_plan(_args("--opponent", "metamon:Kakuna", "--our-temperature", "1.0",
                                 "--dry-run"), cfg)
    assert vs_greedy.our_temperature == 1.0 and vs_greedy.regime_matched is False
    assert cli_mod.core_side.our_stochastic(vs_greedy) == (True, 1.0)
    assert "--temperature" not in runner_mod.peer_plan(
        vs_greedy, cfg, "acceptor", 2, "ours_challenge").command_line()
    spec = runner_mod.cell_spec(vs_greedy, {"their_regime": "greedy"}, 72)
    assert spec.our_regime == "sample:T=1" and spec.regime_matched is False

    both = build_plan(_args("--opponent", "metamon:Kakuna", "--our-temperature", "1.0",
                            "--opponent-temperature", "1.0", "--dry-run"), cfg)
    assert both.regime_matched is True
    assert "--temperature 1.0" in runner_mod.peer_plan(
        both, cfg, "acceptor", 2, "ours_challenge").command_line()
    uneven = build_plan(_args("--opponent", "metamon:Kakuna", "--our-temperature", "1.0",
                              "--opponent-temperature", "0.5", "--dry-run"), cfg)
    assert uneven.regime_matched is False
    # without the flag our side is exactly as before
    plain = build_plan(_args("--opponent", "metamon:Kakuna", "--dry-run"), cfg)
    assert plain.our_temperature is None
    assert runner_mod.cell_spec(plain, {}, 72).our_regime == "greedy"


@pytest.mark.parametrize("argv", [
    ("--opponent", "foulplay", "--our-temperature", "1.0"),
    ("--opponent", "metamon:Kakuna", "--regime", "t1", "--our-temperature", "1.0"),
    ("--opponent", "metamon:Kakuna", "--our-temperature", "0"),
    ("--opponent", "metamon:Kakuna", "--our-side", "bot:random", "--our-temperature", "1.0"),
])
def test_our_temperature_is_refused_outside_its_one_shape(cfg, argv) -> None:
    with pytest.raises(SystemExit):
        build_plan(_args(*argv, "--dry-run"), cfg)

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


def test_a_given_server_uri_starts_nothing_and_is_stamped_external(cfg) -> None:
    """An EXTERNAL server. The rows may not claim a transport this tool never started — that is
    how a number ends up attributed to a stack it never touched."""
    plan = build_plan(_args("--server-uri", "ws://127.0.0.1:9543/showdown/websocket"), cfg)
    assert plan.started_server is False
    assert plan.server_port == 9543
    assert plan.server_impl == "external"
    assert plan.server_version == ""


# ------------------------------------------------------------------------- the transport switch
def test_the_default_transport_is_the_RUST_front_end_and_no_node_server_is_named(cfg) -> None:
    """🚨 The owner's direction: evals come off Node. `--server rust` is the DEFAULT, and the
    plan must be able to say so before anything is started."""
    plan = build_plan(_args(), cfg)
    assert plan.server_impl == "rust"
    # P3: with a checkpoint our-side the default front end runs IN-PROCESS (our side is its slot)
    assert plan.server_version.startswith("ws_frontend[in-process]@")
    rendered = render_plan(plan, cfg)
    assert "[rust, IN-PROCESS]" in rendered and "NO NODE SERVER IS STARTED" in rendered
    # the LIVE client (a bot: our-side) keeps the SUBPROCESS front end, stopped by its PID
    live = build_plan(_args("--our-side", "bot:random"), cfg)
    assert live.our_transport == "live" and live.server_version.startswith("ws_frontend@")
    assert "[rust] (this tool starts and stops it by PID)" in render_plan(live, cfg)


# --------------------------------------------------------- P3: who builds OUR observation
def test_the_default_our_transport_is_the_CORE_slot_and_the_plan_says_so(cfg) -> None:
    """P3 of the poke-env retirement: a checkpoint on `--server rust` plays as an in-process slot
    of the front end, on the core's own row — and the plan names the reader and why it was chosen."""
    plan = build_plan(_args(), cfg)
    assert plan.our_transport == "core"
    assert plan.our_transport_note.startswith("auto")
    rendered = render_plan(plan, cfg)
    assert "our transport     core" in rendered and "NO poke-env here" in rendered
    assert "in-process core slot" in rendered and "python -m main.play" not in rendered


@pytest.mark.parametrize("argv, why", [
    (("--server", "node"), "--server rust started by this tool"),
    (("--server-uri", "ws://127.0.0.1:9543/showdown/websocket"), "--server rust started by this tool"),
    (("--our-side", "bot:random"), "a checkpoint our-side"),
])
def test_auto_falls_back_to_the_LIVE_client_SAYING_why(cfg, argv, why) -> None:
    """P6: every shape the core slot cannot serve plays as a websocket client on the Rust stack."""
    plan = build_plan(_args(*argv), cfg)
    assert plan.our_transport == "live"
    assert why in plan.our_transport_note
    rendered = render_plan(plan, cfg)
    assert why in rendered and "a websocket CLIENT on the Rust stack" in rendered
    assert "python -m main.play" not in rendered and "poke-env" not in rendered.split("our transport", 1)[1][:200]


@pytest.mark.parametrize("argv", [
    ("--server", "node"),
    ("--server-uri", "ws://127.0.0.1:9543/showdown/websocket"),
    ("--our-side", "bot:random"),
])
def test_an_explicit_core_transport_that_cannot_be_served_is_REFUSED(cfg, argv) -> None:
    with pytest.raises(SystemExit, match="--our-transport core needs"):
        build_plan(_args("--our-transport", "core", *argv), cfg)


@pytest.mark.parametrize("argv", [(), ("--server", "node"), ("--our-side", "bot:random")])
def test_the_DELETED_legacy_client_is_refused_with_its_reason(cfg, argv) -> None:
    """P6 deleted the poke-env `main.play` → `RLPlayer` client; a typed `--our-transport poke-env` names why."""
    with pytest.raises(SystemExit, match="DELETED in P6") as excinfo:
        build_plan(_args("--our-transport", "poke-env", *argv), cfg)
    assert "--our-transport live" in str(excinfo.value)


@pytest.mark.parametrize("argv", [(), ("--server", "node"), ("--our-side", "bot:random")])
def test_pipelined_challenges_are_refused_with_their_reason(cfg, argv) -> None:
    """`--challenge-mode pipelined` was poke-env's own loop (hazard H14); both of our transports are serial."""
    with pytest.raises(SystemExit, match="pipelined"):
        build_plan(_args("--challenge-mode", "pipelined", *argv), cfg)


def test_a_bot_our_side_declares_its_stream_seed(cfg) -> None:
    """`--bot-seed` (default `--team-seed`) is the bot's DECLARED stream seed, stamped on every row."""
    default = build_plan(_args("--our-side", "bot:staller", "--team-seed", "31", "--dry-run"), cfg)
    assert default.bot_seed == 31
    explicit = build_plan(_args("--our-side", "bot:staller", "--bot-seed", "7", "--dry-run"), cfg)
    assert explicit.bot_seed == 7
    assert runner_mod.cell_spec(explicit, {}, 1).stamp()["our_bot_seed"] == 7
    assert "stream_seed(7, 0, k)" in render_plan(explicit, cfg) and "stream_seed(7, 1, k)" in render_plan(explicit, cfg)
    assert build_plan(_args("--dry-run"), cfg).bot_seed is None
    assert runner_mod.cell_spec(build_plan(_args("--dry-run"), cfg), {}, 1).stamp()["our_bot_seed"] is None


def test_an_unknown_bot_is_refused_by_name(cfg) -> None:
    with pytest.raises(SystemExit, match="unknown eval bot"):
        build_plan(_args("--our-side", "bot:baitbot", "--dry-run"), cfg)


def test_a_peer_our_side_has_no_transport_of_ours(cfg) -> None:
    plan = build_plan(_args("--our-side", "metamon:SyntheticRLV2", "--dry-run"), cfg)
    assert plan.our_transport == "peer"


@pytest.mark.parametrize("argv, stamp", [
    ((), "rust_core_slot"),
    (("--server", "node"), "rust_live_reader"),
    (("--our-side", "bot:random"), "rust_live_bot"),
    (("--our-side", "metamon:SyntheticRLV2"), "peer"),
])
def test_every_row_stamps_WHICH_reader_built_our_observation(cfg, argv, stamp) -> None:
    plan = build_plan(_args(*argv, "--dry-run"), cfg)
    assert runner_mod.cell_spec(plan, {}, 1).stamp()["our_transport"] == stamp


def test_node_is_reachable_as_the_explicit_opt_out(cfg) -> None:
    """The differential's reference transport. It must stay one flag away."""
    plan = build_plan(_args("--server", "node"), cfg)
    assert plan.server_impl == "node"
    assert plan.server_version == f"showdown:{showdown_pin()}"


def test_the_transport_is_stamped_on_every_row_not_once_per_file(cfg) -> None:
    """The regime rule, applied to the transport: a row that did not say which stack served it
    cannot be compared with one that did."""
    plan = build_plan(_args(), cfg)
    cell = runner_mod.cell_spec(plan, {}, 719)
    assert cell.server_impl == "rust"
    assert cell.server_version == plan.server_version
    assert "server_impl" in REQUIRED_ROW_FIELDS and "server_version" in REQUIRED_ROW_FIELDS


@pytest.mark.parametrize("extra", [
    ["--seed-base", "914001"],
    ["--capture-dir", "/tmp/anchors-caps"],
])
def test_the_reproducibility_pair_is_refused_on_every_path_that_is_not_the_front_end(
        cfg, extra) -> None:
    """A seed the Node server would ignore must be a REFUSAL. A series that looks seeded and is
    not is the one failure a seed exists to prevent."""
    for argv in (["--server", "node", *extra],
                 ["--server-uri", "ws://127.0.0.1:9543/showdown/websocket", *extra]):
        with pytest.raises(SystemExit) as excinfo:
            build_plan(_args(*argv), cfg)
        assert "--server rust" in str(excinfo.value)


def test_the_front_end_carries_the_seed_and_capture_into_the_plan(cfg, tmp_path) -> None:
    plan = build_plan(_args("--seed-base", "914001", "--capture-dir", str(tmp_path)), cfg)
    assert plan.seed_base == 914001 and plan.capture_dir == tmp_path
    argv = server_mod.build_server(
        plan.server_impl, 9555, seed_base=plan.seed_base, capture_dir=plan.capture_dir).argv()
    assert argv[argv.index("--seed-base") + 1] == "914001"
    assert argv[argv.index("--capture-dir") + 1] == str(tmp_path)


# -------------------------------------------------------------------------------- the regime
def test_greedy_sets_our_temperature_to_zero_and_t1_to_one(cfg) -> None:
    """Our half of the regime: `(stochastic, temperature)` both of our transports decide with."""
    greedy = cli_mod.core_side.our_stochastic(build_plan(_args("--regime", "greedy"), cfg))
    t1 = cli_mod.core_side.our_stochastic(build_plan(_args("--regime", "t1"), cfg))
    assert greedy == (False, 1.0)
    assert t1 == (True, 1.0)
    # The forfeit limit is the TRAINER's number, inherited and not restated.
    import main.play as play

    assert build_plan(_args(), cfg).forfeit_turn_limit == play.DEFAULT_FORFEIT_TURN_LIMIT == 250


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


# ------------------------------------------------------------------- our side: a PINNED eval bot
# A bot as our side is what places an EXTERNAL anchor on the ABSOLUTE scale without routing through
# one of our own checkpoints: the nine roster bots carry fixed ratings from the bot-vs-bot round
# robin (`data/gen3_bot_elo_anchors.json`), so a Metamon-vs-bot edge is an edge to a PINNED node.
# The refusals below exist because each failure mode presents as a HANG at game 1, not an error.
def test_our_side_defaults_to_the_model(cfg) -> None:
    plan = build_plan(_args("--dry-run"), cfg)
    assert plan.our_side == "model"
    assert plan.our_side_is_bot is False


@pytest.mark.parametrize("name", [
    "random", "heuristic", "heuristic2", "staller", "staller_v2",
    "aggressive", "aggressive_v2", "setup_sweep", "setup_sweep_v2",
])
def test_every_roster_bot_is_a_legal_our_side(cfg, name: str) -> None:
    """All NINE, by name — the set that is pinned is the set that must be playable, and a roster
    rename that silently drops one would otherwise only show up as a missing anchor edge."""
    plan = build_plan(_args("--our-side", f"bot:{name}", "--dry-run"), cfg)
    assert plan.our_side == f"bot:{name}"
    assert plan.our_side_is_bot


def test_an_unknown_bot_name_is_refused_before_anything_starts(cfg) -> None:
    from main.anchors.cli import parse_our_side

    with pytest.raises(SystemExit) as excinfo:
        parse_our_side("bot:kakuna")
    # The message must NAME the roster: a bad bot name is otherwise indistinguishable from a
    # server that never answered.
    assert "heuristic2" in str(excinfo.value)


@pytest.mark.parametrize("spec", ["bot", "heuristic", "bot:", "foulplay", "metamon"])
def test_a_malformed_our_side_is_refused(spec: str) -> None:
    from main.anchors.cli import parse_our_side

    with pytest.raises(SystemExit):
        parse_our_side(spec)


def test_a_bot_cell_is_stamped_unmatched_because_a_bot_has_no_knob(cfg) -> None:
    """🚨 The honest label. A bot plays its own fixed policy — the SAME policy the round robin
    pinned — and there is no temperature to set to the peer's regime. Same shape as Foul Play."""
    plan = build_plan(_args("--our-side", "bot:staller", "--dry-run"), cfg)
    assert plan.regime_matched is False
    cell = runner_mod.cell_spec(plan, {}, 719)
    assert cell.our_regime == "bot:staller"
    assert cell.regime_matched is False
    assert cell.our_side == "bot:staller"


def test_t1_against_a_bot_our_side_is_refused(cfg) -> None:
    """Only the PEER would move, which is exactly the mixed-regime read the 2026-09-14 battery
    exists to correct."""
    with pytest.raises(SystemExit) as excinfo:
        build_plan(_args("--our-side", "bot:heuristic", "--regime", "t1", "--dry-run"), cfg)
    assert "no sampling knob" in str(excinfo.value)


def test_a_bot_our_side_needs_no_model_and_says_so_in_the_plan(cfg) -> None:
    plan = build_plan(build_parser().parse_args(
        ["--our-side", "bot:aggressive", "--dry-run"]), cfg)
    assert plan.model_zip == ""
    text = render_plan(plan, cfg)
    assert "our side          bot:aggressive" in text
    # P6: the bot is the RUST port on the live client, named in each half's line with its streams
    assert "the Rust bot:aggressive (streams random.Random(stream_seed(" in text
    assert "never forfeits" in text and "(none — our side is not a checkpoint)" in text


# ------------------------------------------------------------- which loader built our checkpoint
@pytest.mark.parametrize("mode", ["auto", "bare", "foreign"])
def test_the_model_loader_is_a_declared_choice_carried_onto_the_plan(cfg, mode: str) -> None:
    plan = build_plan(_args("--model-load", mode, "--dry-run"), cfg)
    assert plan.model_loader == mode


def test_the_loader_default_is_auto_and_the_plan_says_which_one(cfg) -> None:
    """A cross-run frozen snapshot FAILS a bare `MaskablePPO.load` with an unexpected
    `ExtractorBuild` kwarg, so `auto` exists — but a fallback nobody can see is a cell whose
    policy nobody can identify, which is why the plan prints the mode and the row carries it."""
    plan = build_plan(_args("--dry-run"), cfg)
    assert plan.model_loader == "auto"
    assert "loader=auto" in render_plan(plan, cfg)


# ───────────────────────────────────────────── our side is ANOTHER anchor (anchor vs anchor) ───
# The two external anchors are connected to our scale through 19 shared opponents, but the DIRECT
# edge between them is the transitivity check on the whole joint fit: if the fit orders them one
# way and 200 head-to-head games order them the other, that disagreement is the finding.
def test_a_metamon_our_side_is_a_peer_not_a_model(cfg) -> None:
    plan = build_plan(_args("--our-side", "metamon:SyntheticRLV2",
                            "--opponent", "metamon:SmallRL", "--dry-run"), cfg)
    assert plan.our_side_is_peer and not plan.our_side_is_bot
    assert plan.model_zip == ""
    # Both peers take the same --regime and both verify it per decision, so this cell IS matched —
    # unlike a bot cell, where one side has no knob at all.
    assert plan.regime_matched is True


def test_an_agentless_metamon_our_side_is_refused() -> None:
    from main.anchors.cli import parse_our_side

    with pytest.raises(SystemExit):
        parse_our_side("metamon:")


def test_a_policy_against_itself_is_refused(cfg) -> None:
    """Both sides would derive their username from the same agent, and the cell would measure
    nothing about the scale anyway."""
    with pytest.raises(SystemExit) as excinfo:
        build_plan(_args("--our-side", "metamon:SmallRL",
                         "--opponent", "metamon:SmallRL", "--dry-run"), cfg)
    assert "against ITSELF" in str(excinfo.value)


def test_the_two_peers_take_opposite_roles_and_name_each_other(cfg) -> None:
    """🚨 Showdown DROPS a challenge aimed at a user who is not online, and each side must address
    the other by the name it will actually log in under — the failure is a hang, not an error."""
    plan = build_plan(_args("--our-side", "metamon:SyntheticRLV2",
                            "--opponent", "metamon:SmallRL", "--dry-run"), cfg)
    for half, their_role in (("ours_challenge", "acceptor"), ("peer_challenge", "challenger")):
        ours = runner_mod.our_peer_plan(plan, cfg, their_role, 2, half)
        theirs = runner_mod.peer_plan(plan, cfg, their_role, 2, half)
        our_name, peer_name = runner_mod.half_usernames(plan, half)
        assert f"--role {'challenger' if their_role == 'acceptor' else 'acceptor'}" \
            in ours.command_line()
        assert f"--role {their_role}" in theirs.command_line()
        assert f"--opponent-username {peer_name}" in ours.command_line()
        assert f"--opponent-username {our_name}" in theirs.command_line()


def test_the_two_peers_write_to_DIFFERENT_report_and_results_paths(cfg) -> None:
    """Metamon APPENDS to its battle CSV. Two peers sharing one results dir would interleave two
    players' games into one file and the head-to-head edge would count both sides as wins."""
    plan = build_plan(_args("--our-side", "metamon:SyntheticRLV2",
                            "--opponent", "metamon:SmallRL", "--dry-run"), cfg)
    ours = runner_mod.our_peer_plan(plan, cfg, "acceptor", 2, "ours_challenge")
    theirs = runner_mod.peer_plan(plan, cfg, "acceptor", 2, "ours_challenge")
    assert ours.report_path != theirs.report_path
    assert str(ours.report_path.parent) != str(theirs.report_path.parent)


def test_our_peer_side_draws_from_its_own_team_set_and_a_different_seed(cfg) -> None:
    plan = build_plan(_args("--our-side", "metamon:SyntheticRLV2",
                            "--opponent", "metamon:SmallRL", "--dry-run"), cfg)
    ours = runner_mod.our_peer_plan(plan, cfg, "acceptor", 2, "ours_challenge")
    theirs = runner_mod.peer_plan(plan, cfg, "acceptor", 2, "ours_challenge")
    assert f"--team-seed {plan.team_seed}" in ours.command_line()
    assert f"--team-seed {plan.team_seed + 1}" in theirs.command_line()
    assert ours.team_dir == theirs.team_dir     # the SAME set; a different draw


def test_the_plan_prints_OUR_peers_command_not_a_play_py_line(cfg) -> None:
    """A plan a human cannot execute is worse than no plan: `main.play` never runs in this cell."""
    plan = build_plan(_args("--our-side", "metamon:SyntheticRLV2",
                            "--opponent", "metamon:SmallRL", "--dry-run"), cfg)
    text = render_plan(plan, cfg)
    assert "python -m main.play" not in text
    assert "--agent SyntheticRLV2" in text and "--agent SmallRL" in text


def test_our_peer_username_names_its_own_agent_and_fits_showdowns_ceiling(cfg) -> None:
    """Metamon keys its per-battle CSV by the player's username, so an anchor-vs-anchor row must
    name the agent that played it — and 19+ characters is a REFUSAL that presents as a hang."""
    from main.anchors.peers import MAX_USERNAME_LEN, check_username

    plan = build_plan(_args("--our-side", "metamon:SyntheticRLV2",
                            "--opponent", "metamon:SmallRL", "--dry-run"), cfg)
    for half in ("ours_challenge", "peer_challenge"):
        for name in runner_mod.half_usernames(plan, half):
            assert len(name) <= MAX_USERNAME_LEN
            check_username(name, "test")
    assert plan.our_username.startswith("MetaSynthetic")


# ------------------------------------------------------------------- the DEFAULT output directory
# 🚨 REFUSAL 6 in spirit, though it is a redirection rather than a refusal: the old default was
# `Path.cwd() / "anchors_out"`, and an anchor read is taken from the MAIN checkout because that is
# the only tree with `models/`. So the default filled the repo it was measuring — one `git clean`
# from a lost measurement, one `git status` from a landing that stops. It already happened on
# 2026-09-16.
def test_the_default_out_dir_is_never_the_calling_directory(cfg, tmp_path, monkeypatch) -> None:
    """The regression itself. Run from a cwd we own and assert nothing lands under it."""
    monkeypatch.delenv(cli_mod.OUT_ROOT_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    plan = build_plan(_args("--opponent", "metamon:SmallRL"), cfg)
    assert tmp_path not in plan.out_dir.parents and plan.out_dir != tmp_path, (
        f"the --out-less default resolved to {plan.out_dir}, inside the calling directory")
    assert not (tmp_path / "anchors_out").exists()
    # and it is not inside the repo under measurement either
    assert repo_root() not in plan.out_dir.parents


def test_the_default_out_dir_is_run_scoped_and_names_the_cell(cfg, monkeypatch, tmp_path) -> None:
    """Two different cells started in the same second must not share a directory, or a
    half-written summary.json could belong to either."""
    monkeypatch.setenv(cli_mod.OUT_ROOT_ENV_VAR, str(tmp_path))
    at = 1_758_000_000.0
    a = cli_mod.default_out_dir(_args("--opponent", "metamon:SmallRL", "--teamset", "away"), at)
    b = cli_mod.default_out_dir(_args("--opponent", "metamon:SmallRL", "--teamset", "home"), at)
    c = cli_mod.default_out_dir(_args("--opponent", "foulplay", "--teamset", "away"), at)
    assert a != b != c and a != c
    assert a.parent == tmp_path
    for part in ("metamon-smallrl", "greedy", "away"):
        assert part in a.name
    assert a.name[:4].isdigit()          # the timestamp prefix, so a listing sorts by time


def test_the_out_root_env_var_moves_the_default(cfg, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(cli_mod.OUT_ROOT_ENV_VAR, str(tmp_path / "scratch"))
    plan = build_plan(_args("--opponent", "metamon:SmallRL"), cfg)
    assert plan.out_dir.parent == tmp_path / "scratch"


def test_an_explicit_out_is_taken_verbatim(cfg, tmp_path) -> None:
    """An explicit --out is NOT redirected: a measurement directory under
    designs/research_state/measurements/ is a deliberate, committed destination."""
    plan = build_plan(_args("--opponent", "metamon:SmallRL", "--out", str(tmp_path / "cell")), cfg)
    assert plan.out_dir == tmp_path / "cell"


def test_the_defaulted_out_dir_is_PRINTED_before_anything_runs(cfg, monkeypatch, tmp_path,
                                                               capsys) -> None:
    """A directory a reader cannot find is a measurement they cannot bank."""
    monkeypatch.setenv(cli_mod.OUT_ROOT_ENV_VAR, str(tmp_path))
    assert cli_mod.main(["--model", "", "--opponent", "metamon:SmallRL", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "no --out given" in out
    assert str(tmp_path) in out


# ------------------------------------------- the REGISTERED failing Metamon cell (hazard H-H)
def test_an_unmatched_regime_metamon_cell_is_refused(cfg) -> None:
    """REFUSAL 7. `mixed` (us greedy, Metamon at T=1.0) is the one cell shape with three recorded
    crashes and no recorded success — and a mixed-regime number cannot be read beside any other
    strength number here in the first place (SOP rule 1)."""
    with pytest.raises(SystemExit) as exc:
        build_plan(_args("--opponent", "metamon:SmallRL", "--allow-unmatched-regime"), cfg)
    assert "REFUSED" in str(exc.value)
    assert "RecursionError" in str(exc.value)


def test_the_escape_hatch_still_works_for_foulplay(cfg) -> None:
    """The refusal is keyed to metamon, not to the flag: Foul Play has no sampling knob at all,
    so --allow-unmatched-regime is the documented way to take a t1 cell against it."""
    plan = build_plan(_args("--opponent", "foulplay", "--regime", "t1",
                            "--allow-unmatched-regime"), cfg)
    assert plan.regime_matched is False


@pytest.mark.parametrize("regime", ["greedy", "t1"])
def test_a_matched_metamon_cell_is_NOT_refused(cfg, regime: str) -> None:
    """🚨 The refusal must not reach the standing procedure. The fourth H-H occurrence was in a
    MATCHED greedy cell, so refusing every cell that can hit the upstream bug would refuse the
    recurring read itself — and the bug costs no games."""
    plan = build_plan(_args("--opponent", "metamon:SmallRL", "--regime", regime), cfg)
    assert plan.regime_matched is True


# ------------------------------------------------------------ opponent-vs-opponent (pair) cells
def _pair(*argv: str):
    return build_parser().parse_args(list(argv))


def test_a_pair_cell_is_reported_from_the_metamon_side_whatever_the_flag_order(cfg) -> None:
    """Foul Play writes no per-game record we parse; Metamon's battle CSV is the instrument, so the
    Metamon peer is the record side even when it is named second."""
    plan = build_plan(_pair("--opponent-a", "foulplay", "--opponent-b", "metamon:SyntheticRLV2",
                            "--search-time-ms", "1000", "--dry-run"), cfg)
    assert plan.our_side == "metamon:SyntheticRLV2" and plan.our_side_is_peer
    assert plan.opponent_kind == "foulplay" and plan.search_time_ms == 1000
    assert plan.model_zip == "" and plan.regime_matched is True
    same = build_plan(_pair("--opponent-a", "metamon:SyntheticRLV2", "--opponent-b", "foulplay",
                            "--dry-run"), cfg)
    assert (same.our_side, same.opponent) == (plan.our_side, plan.opponent)


def test_a_metamon_pair_keeps_the_A_side_as_the_record_side(cfg) -> None:
    plan = build_plan(_pair("--opponent-a", "metamon:SmallRL", "--opponent-b",
                            "metamon:SyntheticRLV2", "--dry-run"), cfg)
    assert plan.our_side == "metamon:SmallRL" and plan.opponent == "metamon:SyntheticRLV2"


@pytest.mark.parametrize("argv", [
    ("--opponent-a", "foulplay"),                                        # half a pair
    ("--opponent-b", "metamon:SmallRL"),
    ("--opponent-a", "foulplay", "--opponent-b", "foulplay"),            # no per-game record
    ("--opponent-a", "metamon:SmallRL", "--opponent-b", "metamon:SmallRL"),  # itself
    ("--model", "x.zip", "--opponent-a", "foulplay", "--opponent-b", "metamon:SmallRL"),
    ("--our-side", "bot:random", "--opponent-a", "foulplay", "--opponent-b", "metamon:SmallRL"),
])
def test_a_malformed_pair_is_refused_before_anything_starts(cfg, argv) -> None:
    with pytest.raises(SystemExit):
        build_plan(_pair(*argv, "--dry-run"), cfg)


def test_a_pair_cell_forfeits_nothing_and_the_plan_says_so(cfg) -> None:
    """The 250-turn forfeit lives in main.play, which a pair cell never runs — stamping 250 on its
    rows would claim a rule nobody enforced."""
    plan = build_plan(_pair("--opponent-a", "foulplay", "--opponent-b", "metamon:SmallRL",
                            "--dry-run"), cfg)
    assert plan.forfeit_turn_limit == 0
    assert "forfeit limit     NONE" in render_plan(plan, cfg)
    model = build_plan(_args("--opponent", "foulplay", "--dry-run"), cfg)
    assert model.forfeit_turn_limit > 0


@pytest.mark.parametrize("half, ours_err, theirs_err, expect", [
    # OUR Metamon challenges in ours_challenge: its post-game recursion there is H17
    ("ours_challenge", "RecursionError: maximum recursion depth", "", "peer_recursion_upstream"),
    # ... but not in the half where it ACCEPTS
    ("peer_challenge", "RecursionError: maximum recursion depth", "", None),
    # Foul Play is never given the excuse
    ("peer_challenge", "", "RecursionError: maximum recursion depth", None),
])
def test_a_pair_half_names_H17_only_for_the_metamon_side_that_challenged(
        cfg, half, ours_err, theirs_err, expect) -> None:
    plan = build_plan(_pair("--opponent-a", "foulplay", "--opponent-b", "metamon:SyntheticRLV2",
                            "--dry-run"), cfg)
    note = runner_mod.pair_exit_note(plan, half, {"error": theirs_err}, {"error": ours_err},
                                     n_records=5, expected=5)
    assert (note or {}).get("cause") == expect
    # an INCOMPLETE half is never excused
    assert runner_mod.pair_exit_note(plan, half, {"error": theirs_err}, {"error": ours_err},
                                     n_records=4, expected=5) is None


def test_each_side_of_a_pair_cell_is_read_by_its_OWN_adapter(cfg) -> None:
    """A Metamon report read through Foul Play's reader has no argmax rate, and the cell then reads
    UNVERIFIED on a side whose every decision matched (the n0_endofrun_2026-09-27 smoke)."""
    from main.anchors import peers as peers_mod

    plan = build_plan(_pair("--opponent-a", "foulplay", "--opponent-b", "metamon:SyntheticRLV2",
                            "--dry-run"), cfg)
    theirs, ours = runner_mod.pair_adapters(plan)
    assert theirs is peers_mod.PEERS["foulplay"] and ours is peers_mod.PEERS["metamon"]


# -------------------------------------------------------------------------------- --forfeit-turn-limit
def test_the_forfeit_turn_defaults_to_the_trainers_and_may_only_be_LOWERED(cfg) -> None:
    """A websocket game ends where a training episode ends (root CLAUDE.md, LADDER); a deliberately
    shorter series — the routine smoke — may lower it, and the lowered limit reaches our side's
    decision rule and the plan (every row stamps `forfeit_turn_limit` from it)."""
    from main.play import DEFAULT_FORFEIT_TURN_LIMIT

    default = build_plan(_args("--dry-run"), cfg)
    assert default.forfeit_turn_limit == DEFAULT_FORFEIT_TURN_LIMIT
    assert "the TRAINER's number" in render_plan(default, cfg)
    low = build_plan(_args("--forfeit-turn-limit", "10", "--dry-run"), cfg)
    assert low.forfeit_turn_limit == 10 and "LOWERED" in render_plan(low, cfg)
    assert "forfeit at turn 10" in render_plan(low, cfg)
    live = build_plan(_args("--forfeit-turn-limit", "10", "--server", "node", "--dry-run"), cfg)
    assert "forfeit at turn 10" in render_plan(live, cfg)
    for bad in (str(DEFAULT_FORFEIT_TURN_LIMIT + 1), "0", "-3"):
        with pytest.raises(SystemExit):
            build_plan(_args("--forfeit-turn-limit", bad, "--dry-run"), cfg)


def test_a_lowered_forfeit_turn_is_refused_for_a_peer_our_side(cfg) -> None:
    """A PEER our-side runs no client of ours, so nothing of ours forfeits: the flag would be stamped
    and never applied — refused instead."""
    with pytest.raises(SystemExit):
        build_plan(_args("--our-side", "metamon:SmallRL", "--opponent", "metamon:Kakuna",
                         "--forfeit-turn-limit", "10", "--dry-run"), cfg)
