"""Shared color helpers for the Gen3AI Textual UIs.

Pulled out of ``launcher/ui.py``'s inline ``_gradient_color`` so both the
launcher (win-rate / reward cells) and the prober (faithfulness-drift cells)
draw from one definition. Returns a ``#rrggbb`` hex string usable in both Rich
markup (``[bold {col}]``) and Textual styles.
"""

from __future__ import annotations

# Gen3AI palette — kept here so the theme.tcss and Python-side styling agree.
ACCENT = "#4ea3ff"   # blue — titles, focus
GOOD = "#00d75f"     # green
WARN = "#ffd700"     # yellow
BAD = "#ff5f5f"      # red
DIM = "#6c7086"      # muted


def gradient_color(t: float) -> str:
    """Map ``t`` in [0, 1] to a red→yellow→green hex color.

    0.0 = full red, 0.5 = yellow, 1.0 = full green. Values outside [0, 1] are
    clamped. Identical math to the launcher's original ``_gradient_color`` so the
    two UIs render the same gradient.
    """
    t = max(0.0, min(1.0, t))
    r = int(255 * (1 - t) * 2) if t < 0.5 else 255 - int(255 * (t - 0.5) * 2)
    g = int(255 * t * 2) if t < 0.5 else 255
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    return f"#{r:02x}{g:02x}00"
