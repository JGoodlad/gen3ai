"""The peer adapters — the hazard workarounds, and what each peer reports about ITSELF.

Every assertion here corresponds to something a 2026-09-14 de-risk paid for in wall clock:

* the hardcoded ``ws://localhost:8000`` rebind (Metamon H4) — one port from the reserved training-tools server;
* ``PYTHONUNBUFFERED`` (H5), without which the banner a driver waits on never lands;
* ``CUDA_VISIBLE_DEVICES=""`` — a training arm owns the GPU;
* ``PYTHONPATH=""`` — the Metamon env runs UPSTREAM poke-env and Metamon's own packages; our ``src/``
  on its path would put our generic top-level names (``utils``, ``main``, ``agents``) ahead of theirs, and a
  wrong-module import fails SILENTLY (``src/packaging_gate_test.py`` holds the mechanism);
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
from main.anchors.peer_scripts import team_pack as tp
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


def pin_report_fields(plan: PeerPlan) -> dict:
    """What a CURRENT peer writes into its report: the weight pin it VERIFIED (FR-1 FINDING 3) and its verified Showdown
    team pack (gen3_peer_showdown_pack_v1, SOP H22; here a team set the legacy pack played identically)."""
    pin = plan.extra["weight_pin"]
    return {"hf_hub_offline": True, "hf_revision": pin.revision, "weights_verified": dict(pin.weights),
            "weights_loaded": sorted(pin.weights), **team_pack_fields()}


def team_pack_fields(changed: bool = False) -> dict:
    """A peer report's team-pack record (``metamon_side.install_showdown_pack``)."""
    files = [{"file": "sft431.gen3ou_team", "diffs": ["mon 2 (gyarados) happiness: legacy 255, now 0"]}] if changed else []
    return {"team_pack": {"schema": tp.PEER_PACK_SCHEMA, "n_files": 72, "files_sha256": "0" * 64,
                          "legacy_compared": True, "changed": files, "peer_pack_changed": changed},
            "peer_pack_changed": changed}


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


def test_an_alias_plays_its_POLICY_at_its_own_pinned_checkpoint(cfg, tmp_path) -> None:
    """X51 (2026-10-20): ``Kakuna28`` = Metamon's ``Kakuna`` policy at checkpoint 28. The peer must load the
    POLICY (``get_pretrained_model("Kakuna")`` — there is no ``Kakuna28`` model upstream) at the ALIAS's
    checkpoint, and the version stamp must name what was played, never the alias."""
    plan = MetamonPeer().plan(
        cfg=cfg.metamon, agent="Kakuna28", regime="greedy", teamset="away", battle_format="gen3ou",
        server_uri="ws://localhost:9501/showdown/websocket", username="MetaKakuna28",
        opponent_username="MetaKakuna", role="acceptor", n_games=4, team_seed=7, out_dir=tmp_path)
    argv = plan.argv
    assert argv[argv.index("--agent") + 1] == "Kakuna"
    assert argv[argv.index("--checkpoint") + 1] == "28"
    assert plan.version == "Kakuna@ckpt28"
    # the un-aliased name is unchanged: its policy is its own name
    base = MetamonPeer().plan(
        cfg=cfg.metamon, agent="Kakuna", regime="greedy", teamset="away", battle_format="gen3ou",
        server_uri="ws://localhost:9501/showdown/websocket", username="MetaKakuna",
        opponent_username="MetaKakuna28", role="challenger", n_games=4, team_seed=7, out_dir=tmp_path)
    assert base.argv[base.argv.index("--agent") + 1] == "Kakuna" and base.version == "Kakuna@ckpt34"


def test_the_metamon_env_carries_every_hazard_workaround(cfg, tmp_path) -> None:
    env = _metamon_plan(cfg, tmp_path).env
    assert env["PYTHONUNBUFFERED"] == "1", "H5: unflushed prints hide the readiness banner"
    assert env["CUDA_VISIBLE_DEVICES"] == "", "a training arm owns the GPU"
    assert env["METAMON_CACHE_DIR"] == str(cfg.metamon.cache_dir)
    assert env["PYTHONPATH"] == "", (
        "our src/ must not reach the Metamon env: it runs UPSTREAM poke-env and its own packages, and "
        "our generic top-level names on its path would shadow theirs SILENTLY"
    )


def test_the_server_uri_is_passed_explicitly_because_metamon_hardcodes_8000(cfg, tmp_path) -> None:
    """H4. `PokeEnvWrapper.server_configuration` returns poke-env's module-level
    `LocalhostServerConfiguration` = ws://localhost:8000 — our shared dev server, one port from
    the reserved training-tools server — and there is no flag, env var or constructor argument for it."""
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
        "n_decisions": 152, "median_s": 0.0142, **pin_report_fields(plan)}))
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
        "sample_kwarg_values": [True], "argmax_match_rate": 0.647, "regime_check_ok": True,
        **pin_report_fields(plan)}))
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


# ------------------------------------------------- an ENSEMBLE agent (PastaMittens, the Kakuna router)
def _ensemble_plan(cfg, tmp_path: Path) -> PeerPlan:
    return MetamonPeer().plan(
        cfg=cfg.metamon, agent="PastaMittens", regime="greedy", teamset="away",
        battle_format="gen3ou", server_uri="ws://localhost:9501/showdown/websocket",
        username="MetaPastaMittens1", opponent_username="Gen3AIAnchor", role="acceptor", n_games=4,
        team_seed=7, out_dir=tmp_path)


def test_an_ensemble_is_stamped_by_its_PRESET_and_the_peer_is_told_to_check_it(cfg, tmp_path) -> None:
    plan = _ensemble_plan(cfg, tmp_path)
    name, sha = cfg.metamon.preset("PastaMittens")
    assert plan.version == f"PastaMittens@preset:{name}"
    i = plan.argv.index("--expect-preset")
    assert plan.argv[i + 1:i + 4] == [name, "--expect-preset-sha256", sha]
    assert "--checkpoint" not in plan.argv


def test_a_single_policy_carries_no_preset_check(cfg, tmp_path) -> None:
    plan = _metamon_plan(cfg, tmp_path)
    assert "--expect-preset" not in plan.argv and plan.version.startswith("SmallRL@ckpt")


@pytest.mark.parametrize("entry", [{"preset": "x"}, {"preset_sha256": "0" * 64},
                                   {"preset": "x", "preset_sha256": "0" * 63},
                                   {"preset": "x", "preset_sha256": "0" * 64, "checkpoint": 3}])
def test_a_half_specified_ensemble_entry_is_refused(cfg, entry) -> None:
    import dataclasses

    from main.anchors.config import AnchorConfigError

    meta = dataclasses.replace(cfg.metamon, agents={"E": entry})
    with pytest.raises(AnchorConfigError):
        meta.preset("E")


def test_the_router_rate_reaches_the_report(cfg, tmp_path) -> None:
    plan = _ensemble_plan(cfg, tmp_path)
    plan.report_path.write_text(json.dumps({"regime_verified_decisions": True, "peer_error_free": True,
                                            "sample_kwarg_values": [False], "argmax_match_rate": None,
                                            "router_no_rng_rate": 1.0, **pin_report_fields(plan)}))
    rep = MetamonPeer.read_report(plan)
    assert rep["router_no_rng_rate"] == 1.0 and rep["argmax_match_rate"] is None
    assert rep["regime_verified_decisions"] is True


# ------------------------------------------------------- the team pack (gen3_peer_showdown_pack_v1, SOP H22)
def test_the_metamon_peer_is_told_where_to_pack_its_teams_through_showdown(cfg, tmp_path) -> None:
    argv = _metamon_plan(cfg, tmp_path).argv
    bridge = Path(argv[argv.index("--bridge-dir") + 1])
    assert (bridge / "showdown_pack.js").is_file() and (bridge / "pack_oracle.js").is_file()


def _pack_report(cfg, tmp_path, **override) -> dict:
    plan = _metamon_plan(cfg, tmp_path)
    blob = {"sample_kwarg_values": [False], "argmax_match_rate": 1.0, "n_decisions": 9,
            "regime_verified_decisions": True, "peer_error_free": True, **pin_report_fields(plan)}
    blob.update(override)
    plan.report_path.write_text(json.dumps({k: v for k, v in blob.items() if v is not ...}))
    return MetamonPeer.read_report(plan)


def test_a_peer_that_did_not_verify_its_team_pack_never_counts(cfg, tmp_path) -> None:
    """A report from a peer that packed with poke-env (no ``team_pack``) voids the regime verification: FAIL CLOSED."""
    assert _pack_report(cfg, tmp_path)["regime_verified_decisions"] is True
    for override in ({"team_pack": ..., "peer_pack_changed": ...}, {"peer_pack_changed": ...},
                     {"team_pack": {"schema": "something_else"}}):
        r = _pack_report(cfg, tmp_path, **override)
        assert r["regime_verified_decisions"] is False and "team pack" in r["team_pack_note"], override


def test_the_peer_pack_change_reaches_the_report(cfg, tmp_path) -> None:
    r = _pack_report(cfg, tmp_path, **team_pack_fields(changed=True))
    assert r["peer_pack_changed"] is True and r["team_pack_changed_files"] == ["sft431.gen3ou_team"]
    assert _pack_report(cfg, tmp_path)["peer_pack_changed"] is False


def test_the_cell_stamps_the_peer_pack_change_and_a_cell_changed_if_any_half_did(cfg, tmp_path) -> None:
    from main.anchors import runner as runner_mod
    from main.anchors.cli import build_parser, build_plan

    plan = build_plan(build_parser().parse_args(["--model", "", "--opponent", "metamon:SmallRL", "--dry-run"]), cfg)
    r = _pack_report(cfg, tmp_path, **team_pack_fields(changed=True))
    stamp = runner_mod.cell_spec(plan, {**r, "our_peer_pack_changed": False}, 20).stamp()
    assert stamp["peer_pack_changed"] is True and stamp["our_peer_pack_changed"] is False
    assert runner_mod.cell_spec(plan, {}, 20).stamp()["peer_pack_changed"] is None, "no report, no claim"
    assert runner_mod.any_half([False, True]) is True and runner_mod.any_half([False, None]) is False
    assert runner_mod.any_half([None, None]) is None
