"""TD-consistency auxiliary — the Bellman-residual term that de-noises ΔV.

THE DEFECT. The critic's only training signal is a PER-STATE regression: `MSE(V(s_t), G_t)`. That
constrains each state's LEVEL and says nothing whatever about the DIFFERENCE between two adjacent
states — so if V carries independent per-state noise ε, the delta `V(s_{t+1}) − V(s_t)` inherits
`2·Var(ε)` exactly where the truth is nearly constant. That injected dispersion is what the C4
probe measured on self-KO transitions (ΔV RMSE 4.95 against a truth whose constant predictor sits
at 1.33), and it is GAE noise on every transition, not only the dramatic ones.

THE TERM. Add the Bellman identity the critic is already supposed to satisfy, as an explicit loss:

    L_td = λ · mean_t[ ( V(s_t) − r_t − γ·V(s_{t+1}) )² ]

Both ends live (no detach) — this is the residual-gradient form, per the pre-registration in
`designs/research_state/levers/td_consistency_aux.md`. Rung 1 (offline, frozen tokens) met its
pre-registered gate at λ=1.0 and λ=3.0; this module is the live-training half.

FOUR THINGS THAT ARE EASY TO GET WRONG, and how they are handled here:

1. **PPO's minibatches are SHUFFLED**, so `rollout_data` contains no adjacent pairs at all. The
   pairs have to come from the buffer's own `[n_steps, n_envs]` structure. After the first
   `RolloutBuffer.get()`, `observations` / `actions` / … are `swap_and_flatten`ed to an ENV-MAJOR
   flat layout — row `e·n_steps + t` — so temporal adjacency is still recoverable, and
   `sample_contiguous_pairs` returns rows in exactly that convention. `rewards` and
   `episode_starts` are NOT in the flatten list, so they stay `[n_steps, n_envs]` and are read in
   their native shape.

2. **Episode boundaries.** `episode_starts[t+1] == 1` means row t+1 begins a NEW episode, so
   (t, t+1) is not a transition at all. Such a pair is **DROPPED, never zeroed** — zeroing would
   train V toward a fabricated `V(s_t) = r_t` at every terminal. (This also disposes of SB3's
   time-limit bootstrap, which folds `γ·V(s_term)` into the stored reward at the done step: that
   row's successor always starts an episode, so the pair never forms.)

3. **Contiguous SEGMENTS, not random pairs.** A pair needs two forwards; a contiguous run of L
   states serves L−1 pairs off L forwards, which is the "K+1 contiguous forwards serve K pairs"
   economy the pre-registration calls for (~2× cheaper per pair than sampling pairs
   independently). Rung 1 also batched whole battles and found it beat a random-permutation
   control by 12%, so the correlation inside a segment is a feature here, not a compromise.

4. **Units.** The residual mixes a reward with two values, so both must be in the SAME space.
   `policy._critic_value` returns DE-normalized (real-unit) values under `--use-popart` and the
   buffer's rewards are real-unit, so the raw residual is real-unit. But the main value loss
   trains in NORMALIZED space under PopArt, so a real-unit residual squared would arrive ~σ²
   larger and λ would lose the meaning rung 1 calibrated. Dividing the residual by σ puts it in
   the value loss's space — and that is exactly the normalized-space residual, since
   `normalize(V) − normalize(r + γV′) = (V − r − γV′)/σ` (the μ cancels). `scale` is that σ; it
   is 1.0 with PopArt off, where the value loss is already real-unit.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch as th

# How many buffer rows the TD term forwards per minibatch. This is the term's whole cost: one extra
# critic forward of this many states per minibatch, on top of the `batch_size` the PPO objective
# already runs. Sized as a small fraction of a production minibatch (4096-16384) so the throughput
# cost is ~10% of the train step rather than a doubling; large enough that the per-minibatch
# residual mean is not dominated by sampling noise.
TD_AUX_STATES = 512

# Length of each contiguous run. L states serve L-1 pairs, so the forwards-per-pair overhead is
# L/(L-1) — 1.07x at 16. Longer runs are cheaper per pair but correlate the batch harder and lose
# more of it to episode boundaries (a run that lands on a battle end contributes fewer pairs).
TD_AUX_SEG_LEN = 16


def sample_contiguous_pairs(
    episode_starts: np.ndarray,
    n_states: int,
    seg_len: int,
    rng: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Draw contiguous per-env runs from a rollout buffer and pair up their adjacent rows.

    ``episode_starts`` is the buffer's ``[n_steps, n_envs]`` array (1.0 where the row BEGINS an
    episode). ``rng`` is anything with ``.integers(low, high, size)`` (``np.random.Generator``).

    Returns ``(state_rows, pair_a, pair_b, n_candidate)``:

    * ``state_rows``  [S] — buffer rows to forward, in the POST-``swap_and_flatten`` convention
      (``row = env·n_steps + t``), so they index the flattened ``observations`` directly.
    * ``pair_a`` / ``pair_b``  [P] — positions WITHIN ``state_rows`` (0..S-1) forming a valid
      (t, t+1) transition. Indices into the forwarded window, NOT into the buffer: that is what
      lets one forward serve every pair it touches.
    * ``n_candidate`` — how many adjacent positions were considered, so the caller can report the
      episode-boundary DROP rate as a first-class metric rather than silently losing pairs.

    A pair is emitted iff its successor row does not begin a new episode. Nothing is zero-filled.
    """
    ep = np.asarray(episode_starts)
    if ep.ndim != 2:
        raise ValueError(
            f"episode_starts must be the buffer's [n_steps, n_envs] array, got shape {ep.shape}. "
            "Read it BEFORE assuming any flattening — RolloutBuffer.get() does not flatten it.")
    n_steps, n_envs = ep.shape
    seg_len = int(min(max(2, seg_len), n_steps))
    if n_steps < 2:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, empty_i, 0
    n_seg = max(1, int(n_states) // seg_len)

    envs = np.asarray(rng.integers(0, n_envs, size=n_seg), dtype=np.int64)          # [G]
    t0 = np.asarray(rng.integers(0, n_steps - seg_len + 1, size=n_seg), dtype=np.int64)
    ts = t0[:, None] + np.arange(seg_len, dtype=np.int64)[None, :]                  # [G, L]
    state_rows = (envs[:, None] * n_steps + ts).reshape(-1)                         # [G*L] env-major

    # A successor that BEGINS an episode is not a transition — drop the pair.
    succ_is_start = ep[ts[:, 1:], envs[:, None]] > 0.5                              # [G, L-1]
    within = (np.arange(n_seg, dtype=np.int64)[:, None] * seg_len
              + np.arange(seg_len - 1, dtype=np.int64)[None, :])                    # [G, L-1]
    pair_a = within[~succ_is_start]
    return state_rows, pair_a, pair_a + 1, int(n_seg * (seg_len - 1))


def td_residual(
    values: th.Tensor,
    rewards: th.Tensor,
    pair_a: th.Tensor,
    pair_b: th.Tensor,
    gamma: float,
    scale: float = 1.0,
) -> th.Tensor:
    """δ = ( V(s_t) − r_t − γ·V(s_{t+1}) ) / scale, over the sampled pairs.

    ``values`` / ``rewards`` are aligned to the forwarded window (index space of ``pair_a``/``_b``,
    not of the buffer). Both residual ends carry gradient — the residual-gradient form the
    pre-registration specifies. ``scale`` is PopArt's σ (1.0 when PopArt is off); see the module
    docstring on units.
    """
    return (values[pair_a] - rewards[pair_a] - gamma * values[pair_b]) / scale


def td_aux_loss(
    values: th.Tensor,
    rewards: th.Tensor,
    pair_a: th.Tensor,
    pair_b: th.Tensor,
    gamma: float,
    scale: float = 1.0,
    n_candidate: int = 0,
) -> Optional[Tuple[th.Tensor, Dict[str, float]]]:
    """Mean squared Bellman residual + its diagnostics, or ``None`` when nothing is pairable.

    ``None`` (not a zero tensor) on an empty pair set, so a degenerate minibatch contributes no
    term and no metric rather than a fake 0.0 that would read as "perfectly consistent".
    """
    if pair_a.numel() == 0:
        return None
    resid = td_residual(values, rewards, pair_a, pair_b, gamma, scale)
    loss = (resid ** 2).mean()
    with th.no_grad():
        metrics = {
            "loss": float(loss.item()),
            # SIGNED mean: the term suppresses DISPERSION, so a drifting bias here is the thing to
            # watch (rung 1's decomposition — error std falls monotonically, bias should not).
            "resid_mean": float(resid.mean().item()),
            "resid_rms": float(resid.pow(2).mean().sqrt().item()),
            "n_pairs": float(pair_a.numel()),
            "scale": float(scale),
        }
        if n_candidate > 0:
            # Episode-boundary loss rate. High here means the segments keep landing on battle ends
            # (short episodes vs TD_AUX_SEG_LEN), which costs pairs but never corrupts them.
            metrics["pair_drop_frac"] = 1.0 - float(pair_a.numel()) / float(n_candidate)
    return loss, metrics
