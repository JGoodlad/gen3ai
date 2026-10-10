"""Guards on the live-play entry point (`main.play`).

The ones that matter are SAFETY RAILS, not conveniences:

* the reserved-port refusal — a client that connects to :8001 / :8000 has no business on the reserved training-tools
  server or the shared dev server (root CLAUDE.md § Showdown Server; training itself is in-process and uses neither),
  so the refusal lives in code rather than in a warning, and its remedy names the in-repo Rust front end;
* `--server official` never reachable through a local port typo;
* the flags DELETED with the legacy poke-env client (P6) are refused with their reason, never silently ignored.
"""

import pytest

from main.play import DELETED_FLAGS, RESERVED_PORTS, build_parser, refuse_deleted_flags, resolve_uri, run


@pytest.mark.parametrize("port", sorted(RESERVED_PORTS))
def test_reserved_local_ports_are_refused(port):
    with pytest.raises(SystemExit) as exc:
        resolve_uri("local", port)
    assert str(port) in str(exc.value)


def test_the_reserved_set_is_exactly_dev_and_training():
    assert set(RESERVED_PORTS) == {8000, 8001}


def test_the_port_refusal_points_at_the_in_repo_rust_front_end_not_at_node():
    """The refusal's remedy is a server of your own on a 9XXX port. Since the poke-env retirement that is the
    in-repo Rust websocket front end, so the message must name the command that exists - and the module it names
    must exist and take `--port` (a remedy that does not run is worse than none)."""
    import importlib.util

    with pytest.raises(SystemExit) as exc:
        resolve_uri("local", 8001)
    msg = str(exc.value)
    assert "python -m utils.bridge.ws_frontend --port 9" in msg, msg
    assert "npm run showdown" not in msg, "the Node server is no longer the way to get a server of your own"
    assert "in-process" in msg, "the 8001 reason must not claim a live training server is hanging off it"
    assert importlib.util.find_spec("utils.bridge.ws_frontend") is not None
    from utils.bridge.ws_frontend import build_parser as ws_parser
    assert ws_parser().parse_args(["--port", "9017"]).port == 9017


def test_the_8001_reason_no_longer_claims_the_trainer_connects_to_it():
    """Training and eval are in-process on the Rust core (root CLAUDE.md § Showdown Server): the reason shown for
    :8001 must not say a training run's websockets hang off it."""
    assert "live" not in RESERVED_PORTS[8001].lower()
    assert "in-process" in RESERVED_PORTS[8001]


def test_a_9xxx_port_is_allowed_and_points_at_localhost():
    assert resolve_uri("local", 9017) == "ws://127.0.0.1:9017/showdown/websocket"


def test_official_ignores_the_port_and_uses_wss():
    """`--server official` must never be reachable via a local port typo."""
    uri = resolve_uri("official", 8001)
    assert uri.startswith("wss://")
    assert "127.0.0.1" not in uri and "localhost" not in uri


@pytest.mark.parametrize("flag", sorted(DELETED_FLAGS))
def test_a_deleted_flag_is_refused_with_its_reason(flag):
    """P6: `--client poke-env` (and the poke-env client's own `--avatar` / `--concurrency`) is refused before any
    connection, naming why — in both spellings."""
    for argv in ([flag, "x"], [f"{flag}=x"]):
        with pytest.raises(SystemExit, match=f"{flag} was DELETED"):
            refuse_deleted_flags(argv)
        with pytest.raises(SystemExit, match=f"{flag} was DELETED"):
            run(["--mode", "selfplay", "--port", "9", *argv])


def test_the_deleted_flags_are_gone_from_the_parser():
    opts = {o for a in build_parser()._actions for o in a.option_strings}
    assert not opts & set(DELETED_FLAGS)


def test_default_port_is_not_reserved():
    default = build_parser().get_default("port")
    assert default not in RESERVED_PORTS


def test_every_help_string_renders():
    """An unescaped `%` in a help string makes `--help` raise at render time, and
    nothing else in the tree renders them (see main/checkargs_test.py for the same
    guard on the trainer's parser)."""
    build_parser().format_help()


def test_a_halt_marker_refuses_every_mode_before_any_connection(tmp_path, monkeypatch):
    """T28: while the parse-panic marker exists, `main.play` exits FATAL_LIVE_PARSE before it touches a
    socket — selfplay included (the port is one nothing listens on; reaching it would hang or raise)."""
    from main.exit_codes import TrainExitCode
    from main.live import halt as H
    from main.play import run

    monkeypatch.setenv(H.HALT_ENV, str(tmp_path / "halt.json"))
    H.record_halt(reason="planted", entry_point="test")
    for mode in ("selfplay", "challenge", "accept"):
        assert run(["--mode", mode, "--port", "9", "--opponent", "x"]) == int(TrainExitCode.FATAL_LIVE_PARSE)
