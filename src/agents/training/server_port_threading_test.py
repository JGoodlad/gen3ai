"""Regression guard: every component that creates Showdown clients
must thread the configured server (a `server_config` argument), NOT hardcode
the default :8000 ``LocalhostServerConfiguration``.

The bug this originally guarded against: the (now-retired) ReplayCallback's
player-creation hardcoded ``LocalhostServerConfiguration``, so replays connected to
:8000 even when training ran on a custom port — surfacing as ``ConnectionRefused
('127.0.0.1', 8000)``. Forensic traces now come from the eval players
(PerOpponentEvalCallback), which this guard still covers.

The single constructor is ``localhost_server_configuration(port)`` (built once in
train_rl_agent.main from the deleted ``--showdown-port``, deletion pass P11); these are the components it must
thread to.
"""
import inspect

import pytest

from poke_env.ps_client.server_configuration import localhost_server_configuration
from agents.training.eval_callback import (
    PerOpponentEvalCallback, build_eval_opponents, build_eval_players,
)
from agents.training.selfplay_callback import SelfPlayCallback
from main.eval_worker import _play_unit


def test_single_constructor_overrides_port():
    """The single constructor builds a config for an arbitrary port."""
    cfg = localhost_server_configuration(9999)
    assert "ws://localhost:9999/showdown/websocket" == cfg.websocket_url


def test_callback_has_no_in_process_player_creation():
    """Neither eval callback constructs a player: the cycle plays on the Rust eval core
    (`rust_eval.launch`) and the standalone `main.eval_worker` builds its own.

    This guards the contract: if a future change re-adds an in-process
    ``server_configuration=`` player on a callback, it would dodge the eval core (and the
    port threading below) — fail loudly so it's reconsidered. (The callbacks' own
    ``server_config`` parameter, which this file used to require, was deleted with the Python
    worker branch — deletion pass P10-F2.)"""
    for cls in (PerOpponentEvalCallback, SelfPlayCallback):
        n_read = 0
        for name, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
            try:
                src = inspect.getsource(fn)
            except (OSError, TypeError):
                continue
            n_read += 1
            assert "server_configuration=" not in src, (
                f"{cls.__name__}.{name} creates a player in-process; eval now runs in the "
                f"subprocess worker (eval_worker) — route player creation there instead"
            )
        # A NEGATIVE assertion over a loop that swallows its own setup failure passes just as
        # cheerfully over zero iterations. Both callbacks define dozens of methods; if `getsource`
        # starts failing wholesale (a C extension, a decorator that loses __wrapped__, a frozen
        # install) this test would report green while inspecting nothing at all.
        assert n_read >= 5, (
            f"only {n_read} source-readable methods on {cls.__name__} — the 'no in-process player'"
            f" scan inspected almost nothing, so its green verdict is meaningless"
        )


# The standalone eval worker (the Python oracle of the Rust eval) builds players via these functions:
# build_eval_* for the trainee + bot opponents, and _play_unit (eval_worker) for the per-shard matchup —
# which additionally builds the pool-sentinel / stable-opponent RLPlayer. All take server_config and must
# thread it. Same anti-:8000-hardcode guard.
@pytest.mark.parametrize("fn", [build_eval_opponents, build_eval_players, _play_unit])
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
