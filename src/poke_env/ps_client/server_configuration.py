"""This module contains objects related to server configuration."""

from typing import NamedTuple


class ServerConfiguration(NamedTuple):
    """Server configuration object. Represented with a tuple with two entries: server url
    and authentication endpoint url."""

    websocket_url: str
    authentication_url: str


_SMOGON_AUTH_URL = "https://play.pokemonshowdown.com/action.php?"


def localhost_server_configuration(port: int = 8000) -> ServerConfiguration:
    """Build a localhost :class:`ServerConfiguration` for ``port`` (default 8000)."""
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        _SMOGON_AUTH_URL,
    )


LocalhostServerConfiguration = localhost_server_configuration()
"""Server configuration with localhost and smogon's authentication endpoint (port 8000)."""

ShowdownServerConfiguration = ServerConfiguration(
    "wss://sim3.psim.us/showdown/websocket",
    _SMOGON_AUTH_URL,
)
"""Server configuration with smogon's server and authentication endpoint."""
