"""Shared Textual base for the Gen3AI UIs (prober now; launcher port later).

Public API:
    Gen3App     — base App with shared chrome/theme/quit binding
    THEME_PATH  — absolute path to the shared theme.tcss (for subclass CSS_PATH)
    gradient_color, ACCENT, GOOD, WARN, BAD, DIM — shared color helpers
"""

from main.tui.app_base import THEME_PATH, Gen3App
from main.tui.colors import ACCENT, BAD, DIM, GOOD, WARN, gradient_color

__all__ = [
    "Gen3App",
    "THEME_PATH",
    "gradient_color",
    "ACCENT",
    "GOOD",
    "WARN",
    "BAD",
    "DIM",
]
