"""Thin shared Textual base for the Gen3AI UIs.

``Gen3App`` carries only the *presentation* concerns that the prober (now) and a
future launcher Textual port (later) genuinely share: the shared theme, a
Header/Footer chrome convention, and a quit binding. It deliberately does NOT
model the launcher's live-subprocess / IPC / threaded-state machinery — that is
launcher-specific and stays in ``src/main/launcher/``.

Subclasses implement ``compose_body()`` (the chrome is added around it) and may
append their own ``.tcss`` by setting::

    CSS_PATH = [str(THEME_PATH), "my_app.tcss"]

The first entry is absolute (resolves to this package's theme.tcss regardless of
where the subclass lives); the second resolves relative to the subclass module.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

# Absolute path so subclasses in other packages can reference the shared theme
# in their own CSS_PATH list without a relative-path collision.
THEME_PATH = Path(__file__).parent / "theme.tcss"


class Gen3App(App):
    """Base Textual app with shared chrome, theme, and a quit binding."""

    CSS_PATH = [str(THEME_PATH)]
    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield from self.compose_body()
        yield Footer()

    def compose_body(self) -> ComposeResult:
        """Yield the app's body widgets. Overridden by subclasses."""
        return iter(())
