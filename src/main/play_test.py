"""Guards on the ladder entry point (`main.play`).

The two that matter are SAFETY RAILS, not conveniences:

* the reserved-port refusal — a laddering process that connects to :8001 drops every
  poke-env websocket on the live training run at once (root CLAUDE.md § Showdown
  Server), so the refusal lives in code rather than in a warning;
* the guest refusal on `--server official` — the public ladder does not rate a guest,
  so a run without credentials would play unrated games and report nothing.
"""

import pytest

from main.play import RESERVED_PORTS, build_account, build_parser, resolve_server


@pytest.mark.parametrize("port", sorted(RESERVED_PORTS))
def test_reserved_local_ports_are_refused(port):
    with pytest.raises(SystemExit) as exc:
        resolve_server("local", port)
    assert str(port) in str(exc.value)


def test_the_reserved_set_is_exactly_dev_and_training():
    assert set(RESERVED_PORTS) == {8000, 8001}


def test_a_9xxx_port_is_allowed_and_points_at_localhost():
    cfg = resolve_server("local", 9017)
    assert cfg.websocket_url == "ws://localhost:9017/showdown/websocket"


def test_official_ignores_the_port_and_uses_wss():
    """`--server official` must never be reachable via a local port typo."""
    cfg = resolve_server("official", 8001)
    assert cfg.websocket_url.startswith("wss://")
    assert "localhost" not in cfg.websocket_url


def test_official_without_a_username_is_refused():
    with pytest.raises(SystemExit, match="username"):
        build_account(None, None, "official")


def test_local_without_a_username_is_a_guest():
    assert build_account(None, None, "local") is None


def test_default_port_is_not_reserved():
    default = build_parser().get_default("port")
    assert default not in RESERVED_PORTS


def test_every_help_string_renders():
    """An unescaped `%` in a help string makes `--help` raise at render time, and
    nothing else in the tree renders them (see main/checkargs_test.py for the same
    guard on the trainer's parser)."""
    build_parser().format_help()
