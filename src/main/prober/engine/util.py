"""Leaf helpers shared across the engine — percent parsing, npz access, species/move keys.

Nothing here knows about an analysis; everything else in the package may import from it, and it
imports from nothing in the package. Keeping the shared leaves in one module is what makes the
rest of the split a DAG.
"""

from __future__ import annotations

import re

import numpy as np

from agents import gen3_data
from agents.observation import incoming_damage as _inc


# ---------------------------------------------------------------------------
# Small parsing / npz helpers
# ---------------------------------------------------------------------------

def parse_pct(s: str) -> float:
    """'92.1%' -> 0.921. Tolerates a missing '%'."""
    return float(str(s).rstrip("%")) / 100.0


def _has_state(npz, i: int) -> bool:
    try:
        flags = npz["has_state"]
    except KeyError:
        return True
    return bool(flags[i])


def _npz_value(npz, i: int) -> "float | None":
    try:
        vals = npz["values"]
    except KeyError:
        return None
    return float(vals[i]) if 0 <= i < len(vals) else None


def _npz_array(npz, key: str):
    """A captured per-decision array (e.g. `move_logits` / `spread_belief`) or None when the key is absent
    (the head was off / an older trace). `npz` may be None (no captured state)."""
    if npz is None:
        return None
    try:
        return npz[key]
    except (KeyError, TypeError, IndexError):
        return None


def _npz_win_prob(npz, i: int) -> "float | None":
    """Recorded P(win) at decision i. None when the array is absent (old trace) or NaN (the run had
    no win-prob head / this decision wasn't captured) — so the prober shows P(win) only when real."""
    try:
        vals = npz["win_probs"]
    except KeyError:
        return None
    if not (0 <= i < len(vals)):
        return None
    v = float(vals[i])
    return None if np.isnan(v) else v


def _norm_species(s: str) -> str:
    """Lenient species key — lowercase, alnum only — so an item map keyed by an obs-decoded
    display name ('Tyranitar') matches a board id ('tyranitar') regardless of source form."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _pct(hp: "str | float | None") -> "float | None":
    """Parse a recorded HP string (``'28%'`` → 28.0, ``'faint'`` → 0.0) to a percent, else None."""
    if isinstance(hp, (int, float)):
        return float(hp)
    s = str(hp or "").strip().lower()
    if not s:
        return None
    if "faint" in s:
        return 0.0
    try:
        return float(s.rstrip("%"))
    except ValueError:
        return None


def _loss_pct(hp_delta: "str | None") -> "float | None":
    """Magnitude of a NEGATIVE recorded ``hp_delta`` (``'-72%'`` → 72.0); None for a gain/zero."""
    v = _pct(str(hp_delta or "").lstrip("+"))
    return -v if (v is not None and v < 0) else None


# Moves that read base_power 0 in the dex but whose type multiplier is still real (fixed/variable
# power — type IMMUNITY applies even when the amount is constant). Reuses the canonical sets the
# incoming-damage belief prices from, so this stays in sync with the rest of the codebase.
_BP0_DAMAGING = set(_inc.FIXED_DAMAGE) | {"return", "frustration"}


def _multiplier_meaningful(move_id: str) -> bool:
    """Does the active-move type multiplier carry signal for ``move_id``? True for any damaging move
    — positive-BP, fixed-/variable-power, or Hidden Power (revealed as the bare id but a typed
    attack). False for a genuine status/self/field move (Spikes/Recover/Protect), where the obs
    still computes a phantom multiplier the UI should render as n/a. See MatchupView.applicable."""
    if not move_id:
        return False
    if gen3_data.moves.is_damaging(move_id) or move_id.startswith("hiddenpower"):
        return True
    return move_id in _BP0_DAMAGING
