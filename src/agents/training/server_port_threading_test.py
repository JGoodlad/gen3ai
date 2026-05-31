"""Regression guard: every training-path component that creates Showdown clients
must thread the configured server (i.e. respect ``--showdown-port``), NOT hardcode
the default :8000 ``LocalhostServerConfiguration``.

The bug this originally guarded against: the (now-retired) ReplayCallback's
player-creation hardcoded ``LocalhostServerConfiguration``, so replays connected to
:8000 even when training ran on a custom port — surfacing as ``ConnectionRefused
('127.0.0.1', 8000)``. Forensic traces now come from the eval players
(PerOpponentEvalCallback), which this guard still covers.

The single constructor is ``localhost_server_configuration(port)`` (built once in
train_rl_agent.main from ``--showdown-port``); these are the components it must
thread to.
"""
import inspect

import pytest

from poke_env.ps_client.server_configuration import localhost_server_configuration
from agents.training.eval_callback import (
    PerOpponentEvalCallback, build_eval_opponents, build_eval_players,
)
from agents.training.selfplay_callback import SelfPlayCallback


def test_single_constructor_overrides_port():
    """The single constructor builds a config for an arbitrary port."""
    cfg = localhost_server_configuration(9999)
    assert "ws://localhost:9999/showdown/websocket" == cfg.websocket_url


# Every training-path callback that builds players must (a) accept a server_config
# param and (b) build its players from the stored config, never from a bare
# LocalhostServerConfiguration. We assert this structurally on the player-creating
# method's source so the guard holds without standing up a full callback + server.
@pytest.mark.parametrize("cls", [PerOpponentEvalCallback, SelfPlayCallback])
def test_callback_accepts_server_config_param(cls):
    sig = inspect.signature(cls.__init__)
    assert "server_config" in sig.parameters, (
        f"{cls.__name__}.__init__ must accept server_config so the port can be threaded"
    )


def _player_creation_sources(cls):
    """Source of every method on cls that passes server_configuration= to a player."""
    out = []
    for name, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        if "server_configuration=" in src:
            out.append((name, src))
    return out


# SelfPlayCallback still builds players in-process, so its player-creating methods
# must thread the stored config. (PerOpponentEvalCallback no longer creates players —
# the eval subprocess does, via the builder functions guarded below.)
@pytest.mark.parametrize("cls, store_attr", [
    (SelfPlayCallback, "self._server_config"),
])
def test_player_creation_uses_stored_config_not_hardcoded_port(cls, store_attr):
    """Any method that creates a player must pass the STORED config, never a bare
    LocalhostServerConfiguration (the :8000 hardcode that caused the replay bug)."""
    creators = _player_creation_sources(cls)
    assert creators, f"{cls.__name__}: no player-creation method found (test stale?)"
    for name, src in creators:
        assert "server_configuration=LocalhostServerConfiguration" not in src, (
            f"{cls.__name__}.{name} hardcodes LocalhostServerConfiguration (:8000) "
            f"instead of threading {store_attr} — won't respect --showdown-port"
        )
        assert store_attr in src, (
            f"{cls.__name__}.{name} passes server_configuration= but not from "
            f"{store_attr}; it must thread the configured port"
        )


# The subprocess eval path builds players via these module functions, which take the
# server_config as a parameter and must thread it (the worker rebuilds it from the
# --showdown-port the trainer passes). Same anti-:8000-hardcode guard, function form.
@pytest.mark.parametrize("fn", [build_eval_opponents, build_eval_players])
def test_eval_builders_thread_server_config_param(fn):
    sig = inspect.signature(fn)
    assert "server_config" in sig.parameters, (
        f"{fn.__name__} must take server_config so the eval worker can thread the port"
    )
    src = inspect.getsource(fn)
    assert "server_configuration=server_config" in src, (
        f"{fn.__name__} must build players from its server_config arg"
    )
    assert "server_configuration=LocalhostServerConfiguration" not in src
