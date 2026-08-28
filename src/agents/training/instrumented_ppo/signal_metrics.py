"""The `signal/` scalar group — how much ACTION-ATTRIBUTABLE learning signal is arriving.

Two families, and **the pair is the instrument**; either one alone misleads.

* **Advantage density** (`signal/adv_*`) — the spread and the SHAPE of the raw GAE advantages
  PPO is about to fit. Read off `rollout_buffer.advantages` once per `train()`, BEFORE the
  per-minibatch `normalize_advantage` rescale, because that rescale is exactly what destroys the
  quantity we want: it forces std→1 on every minibatch, so the post-normalization advantages carry
  no information about whether the rollout contained decisive moments or was flat noise.
* **Outcome entropy** (`signal/outcome_entropy*`) — `p(1−p)` over a rolling window of recent
  episode outcomes. The Bernoulli variance of the win indicator: the amount of outcome variation
  a win/loss-terminal reward can possibly carry.

**Why neither is readable alone — the MIRROR PARADOX.** Outcome entropy is MAXIMAL (0.25) against
a near-twin opponent, which is the regime where the sides are hardest to tell apart and a given
action's effect on the outcome is smallest. So a high outcome entropy is not "lots of signal"; it
is "lots of outcome VARIANCE", which is only signal to the extent the advantages localize it onto
actions. The joint reading is what means something:

| outcome entropy | advantage density | reading |
|---|---|---|
| high | high | decisive moments exist and the critic finds them — healthy |
| high | LOW | coin-flip games the critic cannot attribute — the mirror paradox / a stale critic |
| LOW | high | lopsided matchup, but the few live decisions are sharp — a curriculum problem |
| low | low | the opponent is a wall or a pushover — no gradient to be had |

**UNITS — read within a run, only cautiously across runs.** Advantages are built from returns that
ride the run's own PopArt normalizer (`--use-popart`, default on), whose σ moves over training. So
`adv_raw_std` is in *this run's current normalized-return units*, not a fixed scale. Within a run
the trend is meaningful; between two runs (different reward composition, different PopArt state,
different `gamma`/`gae_lambda`) only the SHAPE metric `adv_kurtosis` — which is scale-free by
construction — compares directly.

**This is not the attributable-share measurement.** For the real decomposition of how much of an
outcome was action-reducible, the offline counterfactual instruments remain the gold standard
(`prober falsify-scan`'s luck / unattributed / policy_reducible bracket, and `cf_audit`). The
`signal/` group is a live, free, always-on *tripwire* that tells you when to go run one.

Everything here is PURE (numpy in, floats out) and read-only — no torch, no RNG, no gradient path.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, Optional

import numpy as np

# Below this raw-advantage std the higher moments are numerically meaningless (a constant-advantage
# rollout has an undefined kurtosis: 0/0). Reported as NaN rather than a fabricated 0.0 or a crash —
# TensorBoard drops NaN points, so a degenerate rollout leaves a GAP in the curve, which is the
# honest rendering.
_ADV_DEGENERATE_STD = 1e-12

# Rolling window (in episodes) for the outcome-entropy meters. Sized to be long enough to denoise a
# Bernoulli rate (the sd of p̂ at n=200 is ~0.035) and short enough to track a curriculum change
# within a launcher restart window. Per-kind windows are independent, so a rare kind is not starved
# by a common one.
_OUTCOME_WINDOW = 200


def advantage_density_metrics(advantages) -> Dict[str, float]:
    """`signal/adv_*` from the RAW (pre-normalization) GAE advantages of one rollout.

    Returns ``{"adv_raw_std", "adv_raw_abs_mean", "adv_kurtosis"}`` — keys WITHOUT the ``signal/``
    prefix (the caller adds it, matching the `belief/`/`win_prob/` idiom in `ppo.py`).

    * ``adv_raw_std`` — population std. The headline density: how much the critic thinks the
      actions in this rollout mattered.
    * ``adv_raw_abs_mean`` — E|Â|. Std's robust companion: std is dominated by the tail, so std
      rising while abs-mean is flat means the density did not broaden, a few points ran away.
    * ``adv_kurtosis`` — EXCESS kurtosis (Fisher: normal = 0.0). Exploit signal is sparse — a few
      decisive turns per game inside a long stretch of forced/irrelevant ones — so a healthy
      rollout is HEAVY-TAILED (positive). Near 0 or negative means the advantage mass is spread
      evenly across decisions, i.e. the critic is not localizing anything.

    Degenerate inputs never raise: an empty buffer returns ``{}``, and a constant-advantage rollout
    reports a real std/abs-mean with ``adv_kurtosis`` NaN.
    """
    a = np.asarray(advantages, dtype=np.float64).ravel()
    if a.size == 0:
        return {}
    # Non-finite entries would poison every moment; drop them and keep measuring the rest rather
    # than emitting a NaN triple that hides which rollout went bad.
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"adv_raw_std": float("nan"),
                "adv_raw_abs_mean": float("nan"),
                "adv_kurtosis": float("nan")}

    mean = a.mean()
    centered = a - mean
    var = float(np.mean(centered ** 2))
    std = float(np.sqrt(var))
    out = {"adv_raw_std": std, "adv_raw_abs_mean": float(np.mean(np.abs(a)))}

    # Excess kurtosis m4/m2² − 3. Guarded on the SECOND moment, not on n: a large constant array is
    # exactly as undefined as a small one.
    if std <= _ADV_DEGENERATE_STD or a.size < 4:
        out["adv_kurtosis"] = float("nan")
    else:
        m4 = float(np.mean(centered ** 4))
        out["adv_kurtosis"] = m4 / (var * var) - 3.0
    return out


def outcome_entropy(p: Optional[float]) -> float:
    """`p(1−p)` — the Bernoulli variance of a win indicator at win rate ``p``.

    Maximal 0.25 at p=0.5, zero at 0 or 1. NaN for ``None`` (no episodes seen yet), so a
    not-yet-populated window leaves a gap rather than reporting a confident 0.0 (which would read
    as "the opponent is a wall").

    Named ENTROPY for what it measures — outcome uncertainty — though the functional form is the
    variance rather than the Shannon entropy. The variance is the right one here because it is the
    quantity that actually bounds a terminal win/loss reward's contribution to the advantage: a
    Bernoulli return's variance IS p(1−p), so this number and `adv_raw_std` are in a direct
    relationship, which −p log p − (1−p) log(1−p) would not be.
    """
    if p is None or not np.isfinite(p):
        return float("nan")
    p = float(p)
    return p * (1.0 - p)


class OutcomeEntropyTracker:
    """Rolling per-KIND win-rate windows, and the `signal/outcome_entropy*` scalars off them.

    ``observe(won, kind)`` pushes one finished episode; ``metrics()`` reads the meters. The POOLED
    window is fed by every episode regardless of kind, so it is never empty when any split is
    populated. State is process-local and NOT checkpointed — a launcher restart re-warms in a few
    hundred episodes, which is the same contract the noise-scale EMAs take.
    """

    def __init__(self, window: int = _OUTCOME_WINDOW) -> None:
        self._window = int(window)
        self._pooled: Deque[float] = deque(maxlen=self._window)
        self._by_kind: Dict[str, Deque[float]] = {}

    def observe(self, won: bool, kind: Optional[str] = None) -> None:
        """Record one finished episode. ``kind`` is the opponent class (`target` / `pool` / `bots`);
        ``None`` (unknown) still feeds the pooled window — an episode whose opponent we cannot name
        is still an episode."""
        w = 1.0 if won else 0.0
        self._pooled.append(w)
        if kind:
            self._by_kind.setdefault(kind, deque(maxlen=self._window)).append(w)

    def observe_many(self, outcomes: Iterable[bool], kind: Optional[str] = None) -> None:
        for won in outcomes:
            self.observe(won, kind)

    @staticmethod
    def _rate(window: "Deque[float]") -> Optional[float]:
        return float(np.mean(window)) if window else None

    def win_rate(self, kind: Optional[str] = None) -> Optional[float]:
        w = self._pooled if kind is None else self._by_kind.get(kind)
        return self._rate(w) if w is not None else None

    def metrics(self) -> Dict[str, float]:
        """`{"outcome_entropy", "outcome_entropy_<kind>", "outcome_n", ...}` — keys WITHOUT the
        `signal/` prefix. Empty dict until at least one episode has been seen, so nothing is
        published from a window that has not started."""
        if not self._pooled:
            return {}
        out = {"outcome_entropy": outcome_entropy(self._rate(self._pooled)),
               "outcome_win_rate": float(np.mean(self._pooled)),
               "outcome_n": float(len(self._pooled))}
        for kind, w in sorted(self._by_kind.items()):
            if w:
                out[f"outcome_entropy_{kind}"] = outcome_entropy(self._rate(w))
                out[f"outcome_n_{kind}"] = float(len(w))
        return out
