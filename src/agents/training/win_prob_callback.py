"""Win-probability label plumbing (`--win-prob-mode read_only|shaping`): the window back-fill.

The auxiliary win-probability head (`WinProbHead`, `features_extractor.py`) is supervised by the
Monte-Carlo episode OUTCOME (win=1 / loss=0) — but unlike the belief labels (privileged info known
each step), the outcome is a FUTURE quantity, only known when the battle ends. The Rust collector
declares the two TRAINING-ONLY obs keys `win_target` and `win_mask` ([1] float32 each) as host
columns, records each episode's terminal OUTCOME as it ends, and — before `train()` — propagates it
backward to every step of its episode (`backfill_terminal_labels`, below, for the WINDOW trigger;
the complete-game trigger writes every row's own outcome directly). `train()` then reads
``rollout_data.observations["win_target"]`` / ``["win_mask"]`` (shuffle-aligned for free) and folds
the BCE (at ``vf_coef`` under the win-prob critic).

Transitions in the trailing IN-PROGRESS episode (no terminal yet within the buffer) get ``win_mask=0``
and are excluded from the loss — never trained toward a fabricated label. The win label is **undiscounted**
(γ_win = 1): every step of an episode carries that episode's actual outcome, so P(win|s) is directly
"the probability this state leads to a win" — the interpretable quantity the head exists to give.

(The `WinProbLabelCallback` that did this scan per step on the Python env core was deleted with that
core — deletion pass U3 — together with the λ-return, R-rollout Monte-Carlo and anchor-weight treatments
that once rewrote the label here, deleted by L2; `designs/deleted_flags.md`.)
"""

from __future__ import annotations

import numpy as np


def backfill_terminal_labels(scratch: np.ndarray, episode_starts: np.ndarray, wt: np.ndarray,
                             wm: np.ndarray) -> None:
    """THE WINDOW BACK-FILL, as a function: the Rust collector's WINDOW fill (M5 Lane G,
    ``rust_rollout.store.fill_window``) runs this scan.

    Backward scan per timestep (vectorised over envs): carry each episode's terminal outcome back to
    its earlier steps, resetting at an episode boundary. ``scratch[t, e]`` is the outcome at a
    terminal row (NaN elsewhere); ``known`` flags steps whose episode finished within the buffer (so
    the label is real); the trailing in-progress episode stays 0. ``wt`` / ``wm`` are the buffer's
    ``[n_steps, n_envs, 1]`` ``win_target`` / ``win_mask``, written IN PLACE."""
    n_steps, n_envs = scratch.shape
    known = np.zeros(n_envs, dtype=bool)
    val = np.zeros(n_envs, dtype=np.float32)
    for t in range(n_steps - 1, -1, -1):
        s = scratch[t]                            # [n_envs]
        has_terminal = ~np.isnan(s)
        known = known | has_terminal
        val = np.where(has_terminal, s, val).astype(np.float32)
        wt[t, :, 0] = val
        wm[t, :, 0] = known.astype(np.float32)
        # An episode start at step t means the EARLIER (t-1) step belongs to a prior episode.
        starts = episode_starts[t] >= 0.5
        known = known & ~starts
        val = np.where(starts, 0.0, val).astype(np.float32)
