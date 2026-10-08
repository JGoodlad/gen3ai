"""THE KEYED DRAW — a counter-based, per-decision action sample (`gen3_keyed_draw_v1`, M5 Lane G).

Every stochastic action the Rust collector takes — the trainee's, and every POLICY opponent's
(F-LE-8) — is a pure function of the decision's KEY and the served log-probabilities:

    key  = (run seed, stream, env, episode, decision)          five non-negative integers
    h    = splitmix64 chain over the key (below)                 a 64-bit word
    u    = (h >> 11) * 2**-53                                    a uniform in [0, 1)
    x    = logp / T  (T == 1: no division)                       float64, illegal entries -inf
    p    = exp(x - max_legal(x)),  0 where illegal               float64, UNnormalised
    c    = cumsum(p)                                             float64, sequential, index order
    a    = the first index i with c[i] > u * c[-1]               always a LEGAL index

There is no generator and no state: the draw for a decision can be recomputed anywhere from its key
and its log-probabilities — the collector draws it, and the parity gate's Python replay recomputes
it from ITS OWN log-probabilities, so the gate stays EXACT (an action either equals or it does not)
instead of dropping to a distribution test. It replaces Lane E's per-row ``torch.Generator`` loop
with one vectorised NumPy op. Its reason is REPLAYABILITY, not speed: F-LE-8's "5.1 ms of sampling"
was the host waiting for the T2 forward (corrected 2026-09-30); the generator loop itself costs
~0.13 / 0.25 / 0.38 ms at 8 / 40 / 48 rows, so the keyed draw saves ~0.2 ms a step at most.

The chain (every step wraps at 2**64):

    splitmix64(z):  z += 0x9E3779B97F4A7C15
                    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
                    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
                    return z ^ (z >> 31)
    h = splitmix64(seed)
    h = splitmix64(h ^ stream); h = splitmix64(h ^ env); h = splitmix64(h ^ episode)
    h = splitmix64(h ^ decision)

STREAMS (a decision's side): ``STREAM_TRAINEE`` (p1) and ``STREAM_OPPONENT`` (p2). A decision is
named by the core's own columns — ``episode`` (the env's episode ordinal) and ``dec_n`` (the side's
decision index in that battle, sim_bridge's ``__OBS__`` ``n``) — which the Python ``Gen3Env`` replay
counts identically (Lane 0 gate ①'s alignment key).

THE NEAR-BOUNDARY MARGIN. Two paths whose log-probabilities differ in the last bits (a padded GPU
batch vs a CPU B = 1 forward) draw the same action unless ``u * c[-1]`` lies within that difference of
a CDF boundary. ``keyed_actions`` returns each row's margin ``min_i |c[i] - u * c[-1]| / c[-1]`` over
the legal boundaries, so a gate counts a disagreement at a margin below its bar as a NEAR-TIE (the
sampled twin of the greedy near-tie), never as a silent pass or an unexplained failure.

Distribution: for a uniform ``u`` the action is distributed exactly as ``softmax(logp / T)`` over the
legal entries (inverse-CDF sampling). The key is hashed, not drawn from a stream, so two decisions
never share a uniform unless they share a key.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

KEYED_DRAW_ID = "gen3_keyed_draw_v1"

#: A decision's side: the trainee (p1) or the opponent (p2).
#: THE near-boundary threshold for a keyed draw's margin (a CDF fraction): the collector's and the
#: opponent server's ``near_boundary`` telemetry, both sides of the rollout gate's counts, and its tie
#: rule all use it — 2 (``rust_eval.offline.NEAR_TIE_FACTOR``) x the rollout gate's |Δ log-prob|
#: bar 1e-5 (the rollout gate's FLOAT_BAR, deleted with the Python core): two paths whose log-probs differ by less than the bar
#: can disagree only on a row whose margin is below it.
NEAR_MARGIN = 2e-5

STREAM_TRAINEE = 0
STREAM_OPPONENT = 1

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_TWO_M53 = float(2.0 ** -53)

ArrayLike = Union[int, np.ndarray]


def splitmix64(z: np.ndarray) -> np.ndarray:
    """SplitMix64's finaliser (Steele et al. 2014) over a uint64 array, wrapping at 2**64."""
    z = np.asarray(z, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def _u64(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype.kind == "i" and bool((a < 0).any()):
        raise ValueError("keyed draw: every key component must be >= 0")
    return a.astype(np.uint64)


def draw_keys(seed: int, stream: int, env: ArrayLike, episode: ArrayLike, decision: ArrayLike) -> np.ndarray:
    """The 64-bit hash of each decision's key (broadcast over ``env`` / ``episode`` / ``decision``)."""
    if int(seed) < 0 or int(stream) < 0:
        raise ValueError("keyed draw: seed and stream must be >= 0")
    h = splitmix64(np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF))
    h = splitmix64(h ^ np.uint64(int(stream)))
    e, k, d = np.broadcast_arrays(_u64(env), _u64(episode), _u64(decision))
    h = splitmix64(h ^ e)
    h = splitmix64(h ^ k)
    return splitmix64(h ^ d)


def uniforms(keys: np.ndarray) -> np.ndarray:
    """The uniform in [0, 1) each key names: its top 53 bits, exactly representable in float64."""
    return (np.asarray(keys, dtype=np.uint64) >> np.uint64(11)).astype(np.float64) * _TWO_M53


def keyed_uniforms(seed: int, stream: int, env: ArrayLike, episode: ArrayLike, decision: ArrayLike) -> np.ndarray:
    return uniforms(draw_keys(seed, stream, env, episode, decision))


def keyed_actions(logp: np.ndarray, u: np.ndarray, temperature: Union[float, np.ndarray] = 1.0
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """The inverse-CDF action for each row of ``logp`` ``[B, A]`` (illegal entries ``-inf`` — T2's
    output contract) at uniform ``u`` ``[B]``, and each row's near-boundary margin (module docs).

    Every operation is ROW-LOCAL and elementwise-then-sequential (``exp`` per entry, ``cumsum`` along
    the row), so a row's action does not depend on which batch it arrived in — the property the
    gate's B = 1 replay relies on (pinned by ``keyed_draw_test.py``)."""
    x = np.asarray(logp, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"keyed_actions: logp must be [B, A], got {x.shape}")
    b = x.shape[0]
    uu = np.asarray(u, dtype=np.float64).reshape(-1)
    if uu.shape[0] != b:
        raise ValueError(f"keyed_actions: {uu.shape[0]} uniforms for {b} rows")
    if bool(((uu < 0.0) | (uu >= 1.0)).any()):
        raise ValueError("keyed_actions: uniforms must lie in [0, 1)")
    legal = np.isfinite(x)
    if not bool(legal.any(axis=1).all()):
        raise ValueError("keyed_actions: a row with no legal action has no decision to draw")
    t = np.broadcast_to(np.asarray(temperature, dtype=np.float64).reshape(-1), (b,)) \
        if np.ndim(temperature) else np.full(b, float(temperature))
    if bool((t <= 0.0).any()):
        raise ValueError("keyed_actions: temperature must be > 0")
    scaled = np.where((t != 1.0)[:, None], x / t[:, None], x)
    m = np.max(np.where(legal, scaled, -np.inf), axis=1, keepdims=True)
    p = np.where(legal, np.exp(np.where(legal, scaled - m, 0.0)), 0.0)
    c = np.cumsum(p, axis=1)
    total = c[:, -1]
    target = uu * total
    a = (c <= target[:, None]).sum(axis=1)
    # u < 1 so target < total in exact arithmetic; a product that rounds up to `total` would index
    # past the row — the last LEGAL index is the draw there.
    last_legal = x.shape[1] - 1 - np.argmax(legal[:, ::-1], axis=1)
    a = np.minimum(a, last_legal)
    bound = np.where(legal, np.abs(c - target[:, None]), np.inf)
    margin = bound.min(axis=1) / total
    return a.astype(np.int32), margin


def keyed_sample(logp: np.ndarray, *, seed: int, stream: int, env: ArrayLike, episode: ArrayLike,
                 decision: ArrayLike, temperature: Union[float, np.ndarray] = 1.0
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(actions, uniforms, margins)`` for a batch of decisions named by their keys."""
    u = keyed_uniforms(seed, stream, env, episode, decision)
    a, margin = keyed_actions(logp, u, temperature)
    return a, u, margin
