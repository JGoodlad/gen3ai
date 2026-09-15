"""The peer adapters — the hazard workarounds, and what each peer reports about ITSELF.

Every assertion here corresponds to something a 2026-09-14 de-risk paid for in wall clock:

* the hardcoded ``ws://localhost:8000`` rebind (Metamon H4) — one port from the live training run;
* ``PYTHONUNBUFFERED`` (H5), without which the banner a driver waits on never lands;
* ``CUDA_VISIBLE_DEVICES=""`` — a training arm owns the GPU;
* ``PYTHONPATH=""`` — the Metamon env runs UPSTREAM poke-env and ours is a vendored fork; one
  process resolving ``import poke_env`` to the wrong one fails SILENTLY, which is the whole reason
  ``src/poke_env_fork_gate_test.py`` exists;
* the 19-character username ceiling (foul-play H1), whose symptom is a hang, not an error;
* Foul Play's REALIZED search width, because its only budget is wall clock and a win rate without
  the visit count is not reproducible on another box (UNDERSTANDING rule 23).

Nothing here starts a process. The commands are built and read back against recorded log text.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.anchors.config import load_config
from main.anchors.peers import (
    PEERS,
    FoulPlayPeer,
    MetamonPeer,
    PeerError,
    PeerPlan,
    check_username,
    team_source_count,
)


@pytest.fixture()
def cfg():
    return load_config()


def _metamon_plan(cfg, tmp_path: Path, role: str = "acceptor", regime: str = "greedy",
                  teamset: str = "away") -> PeerPlan:
    return MetamonPeer().plan(
        cfg=cfg.metamon, agent="SmallRL", regime=regime, teamset=teamset,
        battle_format="gen3ou", server_uri="ws://localhost:9501/showdown/websocket",
        username="MetaSmallRL", opponent_username="Gen3AIAnchor", role=role, n_games=4,
        team_seed=7, out_dir=tmp_path)


def _foulplay_plan(cfg, tmp_path: Path, role: str = "challenger") -> PeerPlan:
    return FoulPlayPeer().plan(
        cfg=cfg.foulplay, regime="greedy", teamset="home", battle_format="gen3ou",
        server_uri="ws://localhost:9501/showdown/websocket", username="FoulPlay",
        opponent_username="Gen3AIAnchor", role=role, n_games=4,
        search_time_ms=1000, search_parallelism=1, out_dir=tmp_path)


# ------------------------------------------------------------------------------ usernames (H1)
@pytest.mark.parametrize("name", ["Gen3AIAnchor", "MetaSyntheticRLV2"[:18], "A"])
def test_a_legal_username_passes(name: str) -> None:
    check_username(name, "our")


def test_a_19_character_username_is_refused_with_the_hang_named() -> None:
    """action.php answers 19+ with ';;Your username must be less than 19 characters long.' — a
    REFUSAL Foul Play's guest path accepts as an assertion, logs 'Successfully logged in' for, and
    then blocks on forever. A refusal must never present as a hang."""
    with pytest.raises(PeerError) as excinfo:
        check_username("A" * 19, "peer")
    message = str(excinfo.value)
    assert "19" in message and "HANG" in message


def test_a_non_alphanumeric_username_is_refused() -> None:
    with pytest.raises(PeerError):
        check_username("Gen3-AI", "our")


# -------------------------------------------------------------------------------- metamon plan
def test_the_metamon_command_names_the_regime_the_checkpoint_and_the_team_set(cfg, tmp_path) -> None:
    plan = _metamon_plan(cfg, tmp_path)
    argv = plan.argv
    assert argv[0] == str(cfg.metamon.python), "the METAMON interpreter, never gen3ai_stable"
    assert argv[1].endswith("peer_scripts/metamon_side.py")
    assert "--regime" in argv and argv[argv.index("--regime") + 1] == "greedy"
    # The checkpoint comes from the config, so a policy is only an anchor once it is pinned.
    assert argv[argv.index("--checkpoint") + 1] == "40"
    assert argv[argv.index("--team-set") + 1] == "competitive"
    assert plan.version == "SmallRL@ckpt40"
    assert plan.their_regime == "greedy"


def test_the_metamon_env_carries_every_hazard_workaround(cfg, tmp_path) -> None:
    env = _metamon_plan(cfg, tmp_path).env
    assert env["PYTHONUNBUFFERED"] == "1", "H5: unflushed prints hide the readiness banner"
    assert env["CUDA_VISIBLE_DEVICES"] == "", "a training arm owns the GPU"
    assert env["METAMON_CACHE_DIR"] == str(cfg.metamon.cache_dir)
    assert env["PYTHONPATH"] == "", (
        "our src/ must not reach the Metamon env: it runs UPSTREAM poke-env and we vendor a fork, "
        "and one process resolving `import poke_env` to the wrong one fails SILENTLY"
    )


def test_the_server_uri_is_passed_explicitly_because_metamon_hardcodes_8000(cfg, tmp_path) -> None:
    """H4. `PokeEnvWrapper.server_configuration` returns poke-env's module-level
    `LocalhostServerConfiguration` = ws://localhost:8000 — our shared dev server, one port from
    the live training server — and there is no flag, env var or constructor argument for it."""
    argv = _metamon_plan(cfg, tmp_path).argv
    assert argv[argv.index("--server-uri") + 1] == "ws://localhost:9501/showdown/websocket"
    assert "8000" not in " ".join(argv)


def test_the_metamon_driver_refuses_the_reserved_ports_in_its_own_code() -> None:
    """The driver runs in ANOTHER interpreter, so the guard has to be inside it too. Read as text
    because this process cannot import it (it needs `metamon`/`amago`)."""
    from utils.paths import src_path

    source = src_path("main", "anchors", "peer_scripts", "metamon_side.py").read_text()
    assert "RESERVED_PORTS = {8000:" in source
    assert "refusing port" in source


def test_a_metamon_report_is_read_back_with_its_regime_verification(cfg, tmp_path) -> None:
    plan = _metamon_plan(cfg, tmp_path)
    plan.report_path.write_text(json.dumps({
        "sample_kwarg_values": [False], "argmax_match_rate": 1.0, "regime_check_ok": True,
        "n_decisions": 152, "median_s": 0.0142}))
    report = MetamonPeer.read_report(plan)
    assert report["regime_verified"] is True
    assert report["sample_kwargs"] == [False]
    assert report["argmax_match_rate"] == 1.0


def test_a_missing_metamon_report_reads_UNVERIFIED_and_never_as_a_pass(cfg, tmp_path) -> None:
    """A peer killed by H-B writes nothing if the report is only written on the happy path — so
    'no report' must be a distinct, loud state, not an implicit success."""
    report = MetamonPeer.read_report(_metamon_plan(cfg, tmp_path))
    assert report["regime_verified"] is False
    assert report["argmax_match_rate"] is None
    assert "no peer report" in report["note"]


def test_a_sampling_cell_reads_materially_below_one_which_is_the_positive_control(
        cfg, tmp_path) -> None:
    """An instrument that read 1.000 in BOTH regimes would have no power. Measured 2026-09-14:
    0.647 for SmallRL at T=1.0 and 0.880 for SyntheticRLV2 — the finding that 'temperature 1.0'
    is a different amount of noise for every model."""
    plan = _metamon_plan(cfg, tmp_path, regime="t1")
    plan.report_path.write_text(json.dumps({
        "sample_kwarg_values": [True], "argmax_match_rate": 0.647, "regime_check_ok": True}))
    report = MetamonPeer.read_report(plan)
    assert report["sample_kwargs"] == [True]
    assert report["argmax_match_rate"] < 1.0


# -------------------------------------------------------------------------------- foulplay plan
def test_the_foulplay_command_carries_its_only_budget_and_the_right_bot_mode(cfg, tmp_path) -> None:
    challenger = _foulplay_plan(cfg, tmp_path, role="challenger").argv
    acceptor = _foulplay_plan(cfg, tmp_path, role="acceptor").argv
    assert challenger[0] == str(cfg.foulplay.python), "its OWN conda env, never gen3ai_stable"
    assert challenger[challenger.index("--bot-mode") + 1] == "challenge_user"
    assert challenger[challenger.index("--user-to-challenge") + 1] == "Gen3AIAnchor"
    assert acceptor[acceptor.index("--bot-mode") + 1] == "accept_challenge"
    assert "--user-to-challenge" not in acceptor
    assert challenger[challenger.index("--search-time-ms") + 1] == "1000"
    # The full URI, scheme included — Foul Play passes anything that is not "ps"/"local" straight
    # through to `websockets.connect`.
    assert challenger[challenger.index("--websocket-uri") + 1].startswith("ws://")


def test_the_foulplay_regime_is_named_as_a_search_budget_not_as_greedy(cfg, tmp_path) -> None:
    """Foul Play does not sample and does not have a temperature; calling its regime 'greedy'
    would invite a comparison against a Metamon greedy cell that is not the same operation."""
    plan = _foulplay_plan(cfg, tmp_path)
    assert plan.their_regime == "search:1000ms"
    assert "1000ms" in plan.version


def test_the_realized_search_width_is_scraped_out_of_foulplays_own_log(cfg, tmp_path) -> None:
    """🚨 UNDERSTANDING rule 23: a wall-clock budget on a contended box is a WIDTH METER. Measured
    2026-09-14, session means ran 1.21-1.53 M visits/decision at a CONSTANT nominal 1000 ms."""
    plan = _foulplay_plan(cfg, tmp_path)
    plan.log_path.write_text(
        "2026-09-14 Iterations 41233: 1400000\n"
        "2026-09-14 Iterations 39112: 1200000\n"
        "2026-09-14 something else entirely\n")
    report = FoulPlayPeer.read_report(plan)
    assert report["visits_n"] == 2
    assert report["visits_mean"] == pytest.approx(1_300_000.0)
    assert report["visits_min"] == 1_200_000.0
    assert report["regime_verified"] is True


def test_a_foulplay_log_with_no_visit_lines_is_UNVERIFIED_and_says_why(cfg, tmp_path) -> None:
    plan = _foulplay_plan(cfg, tmp_path)
    plan.log_path.write_text("Successfully logged in\nStarting battle\n")
    report = FoulPlayPeer.read_report(plan)
    assert report["regime_verified"] is False
    assert "REALIZED" in report["note"] and "rule 23" in report["note"]


def test_team_source_counts_exclude_bookkeeping_files(cfg, tmp_path) -> None:
    """Metamon's `competitive` directory ships an `index.csv` beside its 20 teams. Counting it
    would report a team-source asymmetry that does not exist."""
    plan = _foulplay_plan(cfg, tmp_path)
    plan.team_dir = tmp_path / "teams"
    plan.team_dir.mkdir()
    (plan.team_dir / "a.gen3ou_team").write_text("x")
    (plan.team_dir / "b.txt").write_text("x")
    (plan.team_dir / "index.csv").write_text("filename\n")
    (plan.team_dir / ".hidden").write_text("x")
    assert team_source_count(plan) == 2


def test_an_absent_team_directory_counts_zero_rather_than_guessing(cfg, tmp_path) -> None:
    plan = _foulplay_plan(cfg, tmp_path)
    plan.team_dir = tmp_path / "nope"
    assert team_source_count(plan) == 0


def test_both_anchors_are_registered_under_the_names_the_cli_accepts() -> None:
    assert set(PEERS) == {"metamon", "foulplay"}
    assert PEERS["metamon"].kind == "metamon"
    assert PEERS["foulplay"].kind == "foulplay"


def test_the_metamon_peer_env_pins_ONE_thread(tmp_path) -> None:
    """🚨 B=1 CPU INFERENCE WANTS ONE THREAD. Measured 2026-09-14 on a box at load 63: each peer
    burned 210% CPU — two cores of thread synchronisation per peer — and a three-lane campaign ran
    at 33 s/game against the SOP's measured 3.0. The parallelism belongs ACROSS cells.

    Asserted on the ENV the plan actually carries, not on the module that sets it, because the
    knob only bites if it is in the child's environment before torch imports."""
    from main.anchors.config import load_config
    from main.anchors.peers import MetamonPeer

    plan = MetamonPeer().plan(
        cfg=load_config().opponent("metamon"), agent="SmallRL", regime="greedy",
        teamset="home", battle_format="gen3ou",
        server_uri="ws://localhost:9500/showdown/websocket",
        username="A1", opponent_username="B1", role="acceptor", n_games=2,
        team_seed=1, out_dir=tmp_path)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        assert plan.env[key] == "1", f"{key} is not pinned to one thread"
    # and the two that were already load-bearing
    assert plan.env["CUDA_VISIBLE_DEVICES"] == ""      # a training arm owns the GPU
    assert plan.env["PYTHONPATH"] == ""                # two `poke_env` packages must not meet
