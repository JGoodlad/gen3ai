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
from textual.reactive import reactive
from textual.widgets import Footer, Header

# Absolute path so subclasses in other packages can reference the shared theme
# in their own CSS_PATH list without a relative-path collision.
THEME_PATH = Path(__file__).parent / "theme.tcss"


class Gen3App(App):
    """Base Textual app with shared chrome, theme, and a quit binding."""

    CSS_PATH = [str(THEME_PATH)]
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        # macOS ⌘C — arrives as `super+c` via the kitty keyboard protocol (the terminal
        # must forward it; see src/main/tui/CLAUDE.md). Copies the Textual-native text
        # selection; SkipAction (nothing selected) falls through harmlessly. Inherited
        # by every Gen3App subclass — the launcher's ctrl+c→quit rebind doesn't shadow it.
        Binding("super+c", "screen.copy_text", "Copy", show=False),
        # `v` → copy mode: the portable copy path for terminals that support NEITHER ⌘C
        # forwarding NOR OSC 52 (e.g. macOS Terminal.app), where both `super+c` above and
        # Textual's own clipboard write are dead. It releases the mouse back to the
        # terminal (so native click-drag selection works again) and freezes live updates
        # (so a repaint can't wipe the selection mid-drag) — then you copy with the
        # terminal's OWN selection + copy, no app/clipboard-escape support required. Same
        # key resumes. See src/main/tui/CLAUDE.md → Copy mode.
        Binding("v", "toggle_copy_mode", "Copy mode"),
    ]

    # Flipped by the `v` binding. Reactive so a subclass can `watch_copy_mode` too; init
    # is False and we never touch the driver until a user keystroke (post-mount).
    copy_mode: reactive[bool] = reactive(False, init=False)

    def compose(self) -> ComposeResult:
        yield Header()
        yield from self.compose_body()
        yield Footer()

    def compose_body(self) -> ComposeResult:
        """Yield the app's body widgets. Overridden by subclasses."""
        return iter(())

    # ── copy mode (terminal-native selection escape hatch) ──────────────────────
    def action_toggle_copy_mode(self) -> None:
        self.copy_mode = not self.copy_mode

    def watch_copy_mode(self, active: bool) -> None:
        """Enter/leave copy mode: hand the mouse to the terminal + freeze redraws."""
        driver = getattr(self, "_driver", None)
        # Private Textual driver hooks (stable since 0.x): they just write the
        # xterm mouse-tracking enable/disable escapes. getattr-guarded so a headless
        # test driver that lacks them is a no-op rather than a crash.
        toggle = getattr(driver, "_disable_mouse_support" if active else "_enable_mouse_support", None)
        if active:
            self._pause_live_updates()
            if callable(toggle):
                toggle()
            self.notify(
                "Drag to select, ⌘C to copy, then press v to resume.",
                title="📋 Copy mode — live updates paused",
                timeout=10,
            )
        else:
            if callable(toggle):
                toggle()
            self._resume_live_updates()

    def _pause_live_updates(self) -> None:
        """Hook: subclasses with a live-refresh timer pause it here so the frozen
        screen holds still while the user selects. No-op by default (static apps)."""

    def _resume_live_updates(self) -> None:
        """Hook: counterpart to :meth:`_pause_live_updates` — resume + redraw."""
