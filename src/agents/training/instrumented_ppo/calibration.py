"""Win-probability CALIBRATION — `win_prob/ece`, `win_prob/mce`, the reliability histogram, and
the EPISODE-START read (`win_prob/start_*`).

`gen3_winprob_calibration_export_v1`. Pure NumPy; no torch, no SB3, no I/O. `ppo.py` supplies the
predictions and labels it already has and records what comes back.

WHAT WAS MISSING AND WHY IT MATTERS
-----------------------------------
The head already reported `brier` (a PROPER SCORING RULE) and `acc`. Neither is a calibration
measurement: Brier decomposes as ``reliability - resolution + uncertainty``, so a head can trade
calibration for sharpness and hold its Brier flat. When the next generation makes this head the
critic's only signal, the quantity that matters is whether "0.7" MEANS 0.7 — which is the
reliability term, on its own.

* **`ece`** — the standard 10-bin, count-weighted Expected Calibration Error:
  ``sum_b (n_b/N) * |mean_pred_b - mean_label_b|``. Interpretable directly as "the head's stated
  probability is off by this much, on average, in probability units".
* **`mce`** — the WORST populated bin's gap. ECE is an average, so a head that is badly wrong on
  the confident tail and fine everywhere else can hold a small ECE; MCE is what shows it.
* **`rel_gap_b0..b9`** — the reliability histogram itself, one scalar per bin, so the SHAPE of the
  miscalibration is readable (over-confident at the top, floor-collapsed at the bottom, …) rather
  than summarized into one number. **An under-populated bin publishes NaN, never a gap of 0.0** —
  a bin holding three samples has an "error" that is sampling noise, and TensorBoard renders NaN
  as a hole in the curve, which is the honest picture.
* **`ece_contested`** — the same, restricted to material-EVEN decisions (the existing
  `_WIN_CONTESTED_TAU` band). A blowout's P(win) is trivially recoverable from material, so the
  pooled ECE is flattered by exactly the states nobody needs the head for.

THE EPISODE-START READ (`win_prob/start_*`) IS A PAIRED CALIBRATION, NOT TWO WINDOWS
------------------------------------------------------------------------------------
"What does the head predict at the start of a self-play game, and does that match the realized
self-play win rate?" is a question about the LEAST-informed state, where a miscalibration cannot be
excused by a lost position. It is answered from ONE set of rows: the episode-start rows of the
rollout, whose ``win_target`` (back-filled by `WinProbLabelCallback` from the episode's own
outcome) IS the realized outcome of the episode that starts there. So `start_pred_mean` and
`start_realized_mean` are computed over the same episodes, and `start_gap` is a paired difference —
not the difference of two independently-windowed averages, which would carry the two windows'
disagreement as well as the head's error.

**Stratification is OPPORTUNISTIC.** The per-opponent-class split needs the `opp_class` obs key,
which the env emits only when the opponent-intent labels are on (`--opp-intent-coef > 0`). Without
it the pooled read still ships, and `signal/outcome_win_rate_<kind>` carries the realized per-class
win rate unconditionally (from the info dicts) — so the self-play realized rate is never missing,
only its paired partner is.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

# Reliability-histogram resolution. Ten equal-width bins on [0,1] is the ECE convention (Guo et
# al. 2017); the bin count is fixed rather than flagged because an ECE is only comparable against
# another ECE at the same binning, and a per-run knob would make two runs' curves incomparable
# with nothing saying so.
N_BINS: int = 10

# Below this many samples a bin's gap is sampling noise, not a calibration reading, and it is
# published as NaN. Sized so a bin's binomial sd is under ~0.05 at p=0.5 (sd = 0.5/sqrt(n)).
_MIN_BIN_N: int = 100


class CalibrationAccumulator:
    """Bin counts for a reliability diagram, folded across the minibatches of ONE rollout.

    Accumulates rather than averaging per-minibatch ECEs, because an ECE is a nonlinear function
    of the bin populations: the mean of per-minibatch ECEs is not the ECE of the pooled sample,
    and at a 4096-row minibatch the tail bins are thin enough for the difference to be large.
    """

    __slots__ = ("n_bins", "_count", "_pred", "_label")

    def __init__(self, n_bins: int = N_BINS) -> None:
        self.n_bins = int(n_bins)
        self._count = np.zeros(self.n_bins, dtype=np.float64)
        self._pred = np.zeros(self.n_bins, dtype=np.float64)
        self._label = np.zeros(self.n_bins, dtype=np.float64)

    def observe(self, pred, label, mask=None) -> None:
        """Fold one minibatch. ``pred`` / ``label`` are [B] (or [B,1]) in [0,1]; ``mask`` [B] is 1
        where the label is KNOWN (the trailing in-progress episode is excluded, exactly as the BCE
        excludes it). Non-finite rows are dropped rather than poisoning a bin."""
        p = np.asarray(pred, dtype=np.float64).reshape(-1)
        y = np.asarray(label, dtype=np.float64).reshape(-1)
        if p.size == 0 or p.size != y.size:
            return
        keep = np.isfinite(p) & np.isfinite(y)
        if mask is not None:
            m = np.asarray(mask, dtype=np.float64).reshape(-1)
            if m.size == p.size:
                keep &= m > 0.5
        if not keep.any():
            return
        p = np.clip(p[keep], 0.0, 1.0)
        y = y[keep]
        # `minimum` keeps p == 1.0 in the last bin rather than creating an n_bins+1'th.
        idx = np.minimum((p * self.n_bins).astype(np.int64), self.n_bins - 1)
        self._count += np.bincount(idx, minlength=self.n_bins)
        self._pred += np.bincount(idx, weights=p, minlength=self.n_bins)
        self._label += np.bincount(idx, weights=y, minlength=self.n_bins)

    @property
    def n(self) -> float:
        return float(self._count.sum())

    def metrics(self, prefix: str = "") -> Dict[str, float]:
        """`{prefix}ece` / `{prefix}mce` / `{prefix}rel_gap_b<k>` / `{prefix}rel_n`. Empty dict
        when nothing was folded, so an unlabelled rollout leaves a gap rather than a zero."""
        total = self._count.sum()
        if total <= 0.0:
            return {}
        populated = self._count > 0
        gap = np.full(self.n_bins, np.nan, dtype=np.float64)
        gap[populated] = np.abs(self._pred[populated] - self._label[populated]) / self._count[populated]
        weights = self._count / total
        ece = float(np.nansum(np.where(populated, weights * gap, 0.0)))
        readable = self._count >= _MIN_BIN_N
        out: Dict[str, float] = {f"{prefix}ece": ece,
                                 f"{prefix}rel_n": float(total)}
        # MCE over READABLE bins only — the max of a set that includes a 3-sample bin is a
        # measurement of that bin's sampling noise, which is the opposite of what a worst-case
        # calibration number is for.
        out[f"{prefix}mce"] = float(np.max(gap[readable])) if readable.any() else float("nan")
        for k in range(self.n_bins):
            out[f"{prefix}rel_gap_b{k}"] = float(gap[k]) if readable[k] else float("nan")
        return out


def episode_start_rows(episode_starts, n_steps: int, n_envs: int) -> np.ndarray:
    """Buffer rows (POST-`swap_and_flatten`, ENV-MAJOR ``row = env*n_steps + t``) that BEGIN an
    episode, from the buffer's un-flattened ``[n_steps, n_envs]`` ``episode_starts``.

    Raises on a flattened input rather than indexing it: at ``n_envs > 1`` a flattened array would
    silently pair a start flag with the wrong row, which is the class of defect `td_aux`'s own
    fail-loud guard exists for."""
    ep = np.asarray(episode_starts)
    if ep.ndim != 2 or ep.shape != (n_steps, n_envs):
        raise ValueError(
            f"episode_starts must be the buffer's [n_steps, n_envs] array, got shape {ep.shape}. "
            "Reading it after `swap_and_flatten` would mis-pair rows with their start flags.")
    ts, envs = np.nonzero(ep > 0.5)                       # ts indexes n_steps, envs indexes n_envs
    return (envs.astype(np.int64) * n_steps + ts.astype(np.int64))


def start_metrics(pred, realized, mask=None,
                  opp_class=None,
                  class_names: Optional[Mapping[int, str]] = None) -> Dict[str, float]:
    """`win_prob/start_*` — the paired episode-start read (keys WITHOUT the `win_prob/` prefix).

    ``pred`` [S] is P(win) at each episode-start row, ``realized`` [S] that episode's own outcome
    (the back-filled `win_target`), ``mask`` [S] which of them is KNOWN. ``opp_class`` [S] and
    ``class_names`` add the per-class split when the obs carried the key; omit both for the pooled
    read alone. Returns ``{}`` when nothing is scorable.

    ``start_gap = mean(pred) - mean(realized)`` over the SAME episodes: positive = the head is
    OPTIMISTIC at the opening board against what those very games went on to do."""
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(realized, dtype=np.float64).reshape(-1)
    if p.size == 0 or p.size != y.size:
        return {}
    keep = np.isfinite(p) & np.isfinite(y)
    if mask is not None:
        m = np.asarray(mask, dtype=np.float64).reshape(-1)
        if m.size == p.size:
            keep &= m > 0.5
    if not keep.any():
        return {}
    out: Dict[str, float] = {}

    def _fold(sel: np.ndarray, suffix: str) -> None:
        if not sel.any():
            return
        pp, yy = p[sel], y[sel]
        out[f"start_pred_mean{suffix}"] = float(pp.mean())
        out[f"start_realized_mean{suffix}"] = float(yy.mean())
        out[f"start_gap{suffix}"] = float(pp.mean() - yy.mean())
        out[f"start_n{suffix}"] = float(sel.sum())

    _fold(keep, "")
    if opp_class is not None and class_names:
        c = np.asarray(opp_class).reshape(-1)
        if c.size == p.size:
            for code, name in sorted(class_names.items()):
                _fold(keep & (c == code), f"_{name}")
    return out


def contested_mask(margin, tau: float) -> Optional[np.ndarray]:
    """Rows whose normalized material margin is inside ``±tau`` — the material-EVEN decisions the
    contested Brier is already restricted to. ``None`` when no margin key was present."""
    if margin is None:
        return None
    m = np.asarray(margin, dtype=np.float64).reshape(-1)
    return np.abs(m) < float(tau)


def as_numpy(t) -> Optional[np.ndarray]:
    """Detached CPU numpy view of a torch tensor (or ``None`` for ``None``). Kept here so the
    caller's fold sites stay one line each and nothing in `ppo.py` grows a torch import it did not
    already have."""
    if t is None:
        return None
    return t.detach().to("cpu").numpy()


def sigmoid(x: Sequence[float]) -> np.ndarray:
    """Numerically stable logistic, so a saturated logit maps to 0/1 rather than overflowing."""
    a = np.asarray(x, dtype=np.float64)
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    e = np.exp(a[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# --------------------------------------------------------------------------------------------
# THE CRITIC'S OWN RELIABILITY, under `--critic winprob` (gen3_winprob_critic_mode_v1)
# --------------------------------------------------------------------------------------------

#: The Murphy/Brier keys lifted out of `scaffolding.reliability_table` for the live export. A
#: fixed list rather than the whole dict: the table also returns the per-bin `table` rows, which
#: are a curve rather than a scalar and are already served by `rel_gap_b*` above.
_CRITIC_KEYS = ("brier", "skill", "ece", "mce", "reliability", "resolution", "uncertainty",
                "decomp_residual", "base_rate", "n")


def critic_reliability(rollout_buffer) -> Dict[str, float]:
    """The Murphy split of the ROLLOUT's own critic values against the realized outcome.

    Under `--critic winprob` the buffer's `values` ARE `sigmoid(win-prob logit)`, and
    `win_target` / `win_mask` are the Monte-Carlo outcome `WinProbLabelCallback` back-fills for
    every step whose episode finished inside this buffer. So this needs no forward at all: both
    columns are already there, in probability units, for the exact states GAE bootstrapped from.

    **This is a DIFFERENT read from `CalibrationAccumulator` above, and the difference is the
    reason both exist.** That one reads the HEAD's logits per minibatch, inside `train()`; this
    one reads the DEPLOYED value — what the critic actually told GAE — once per rollout.

    🚨 **The meter is `resolution`, not `reliability`.** A base-rate forecaster scores a perfect 0
    reliability and a useless 0 resolution, and the committed 2026-09-06 baseline measured this
    head at reliability ~0.002 against a resolution of 0.062 out of an available 0.182 — already
    calibrated in the MEAN and starved of SEPARATION. A promotion that improves ECE and leaves
    resolution flat has moved the meter that was never the disease.

    ⚠️ It is computed on the TRAINING population, not on the eval recorder's loss-enriched quota,
    so it needs no selection reweighting — and for that same reason its LEVEL is not comparable
    with `main.scaffolding_gauge --reliability`'s, only its trend. The STATISTIC is shared
    (`scaffolding.reliability_table`, imported rather than re-implemented) so the two are at least
    the same question asked of two populations.

    Returns `{}` — never zeros — when the obs keys are absent or no row carries a known label. A
    calibration of nothing and a perfect calibration must not render the same.
    """
    from agents.training.scaffolding import reliability_table

    obs = getattr(rollout_buffer, "observations", None)
    if not isinstance(obs, dict) or "win_target" not in obs or "win_mask" not in obs:
        return {}
    y = np.asarray(obs["win_target"], dtype=np.float64).reshape(-1)
    m = np.asarray(obs["win_mask"], dtype=np.float64).reshape(-1) > 0.5
    p = np.asarray(rollout_buffer.values, dtype=np.float64).reshape(-1)
    if p.shape != y.shape or not bool(m.any()):
        return {}
    table = reliability_table(p[m], y[m])
    return {k: float(table[k]) for k in _CRITIC_KEYS if k in table}


# --------------------------------------------------------------------------------------------
# IS `--vf-coef` IN A SANE RANGE? — the FIRST-ROLLOUT scale readout (gen3_winprob_critic_mode_v1)
# --------------------------------------------------------------------------------------------

#: The BCE of a calibrated forecaster at a 0.5 base rate: `-ln(0.5) = ln 2`. It is the value a
#: freshly-initialised head sits at (a zero logit is P = 0.5), so it is the right anchor for
#: "is the first reading normal, or has the run started somewhere strange".
LN2 = 0.6931471805599453


def vf_coef_scale_line(vf_coef: float, bce: float, policy_loss: float) -> str:
    """One human line: what `--vf-coef` is actually multiplying on this run's first rollout.

    🚨 **`--vf-coef` MEANS SOMETHING DIFFERENT UNDER `--critic winprob`, AND NOTHING IN A METRIC
    NAME SAYS SO.** Under `shaped` it weights an MSE on a PopArt-normalised shaped return; under
    `winprob` it weights the win-prob head's **BCE against a Bernoulli outcome**, which is bounded
    near `ln 2 ~ 0.693` at initialisation and falls from there. The historical default 0.5 was
    tuned against the first quantity and carries no information about the second — so the first arm
    needs to READ the ratio rather than inherit a number, and this prints it where the operator is
    already looking (the `[CRITIC] winprob` banner family).

    Deliberately a PRINT and not a scalar: it is a once-per-run sanity reading whose whole job is
    to be seen at startup by the person choosing the coefficient. The per-rollout series it would
    duplicate already exist — `train/policy_gradient_loss` and `win_prob/loss` — and a fourth name
    for a ratio of two published numbers is a surface, not a measurement.

    Pure: takes the three numbers and returns the sentence, so the wording is testable without a
    PPO. `policy_loss` is the CLIPPED SURROGATE as folded, which is signed and can sit near zero on
    a well-fit rollout, so the ratio is taken on magnitudes and reported as UNAVAILABLE rather than
    as a division by ~0 — an infinite ratio would read as "the value term dominates", which is the
    opposite of what a near-zero policy loss means.
    """
    term = float(vf_coef) * float(bce)
    pol = abs(float(policy_loss))
    ratio = (f"{term / pol:.3g}x" if pol > 1e-6 else
             f"UNAVAILABLE (|policy loss| {pol:.3g} is ~0 this rollout)")
    return (f"🎯 [CRITIC] winprob — first rollout scale: value term = --vf-coef {float(vf_coef):g} "
            f"x BCE {float(bce):.4f} = {term:.4f}, against |policy loss| {pol:.4f} -> {ratio}. "
            f"⚠️ --vf-coef now multiplies a BCE, not the shaped-return MSE 0.5 was tuned for: a "
            f"BCE at a 0.5 base rate is ln 2 ~ {LN2:.3f} per sample at init and falls, where that "
            f"MSE on a +-30 return was O(100). Read the ratio, not the coefficient — see "
            f"designs/ai_v12/design_winprob_only_critic.md 5.4.")


def announce_vf_coef_scale(model, bce: Optional[Sequence[float]],
                           pg_losses: Sequence[float]) -> None:
    """Print :func:`vf_coef_scale_line` ONCE per process, on the first rollout that can read it.

    Stateful half, kept out of the pure function so the wording stays testable. Three properties:

    * **ONCE**, latched on the model (`_vf_scale_announced`), because it answers a question about
      the RUN's configuration, not about this rollout — repeating it every rollout would bury the
      startup banner it belongs beside.
    * **NEVER on a rollout it cannot read.** A `train()` whose minibatches carried no scorable
      win-prob label (an absent or EMPTY `bce`) or no policy loss produces no line and does NOT
      latch, so the next rollout tries again. A number invented from a missing one is worse than
      a late one. Both inputs are the per-minibatch LISTS `train()` accumulates and averages for
      its own `record` calls — averaged here the same way, so the printed pair is exactly the pair
      `win_prob/loss` and `train/policy_gradient_loss` publish for that rollout.
    * It goes to stdout via `print`, the same channel as `train_rl_agent`'s own `[CRITIC]` banner,
      so a launcher-managed run finds both in `launcher_child.log` and a bare run finds both on
      its terminal.
    """
    if getattr(model, "_vf_scale_announced", False):
        return
    if bce is None or not len(bce) or not len(pg_losses):
        return
    model._vf_scale_announced = True
    print(vf_coef_scale_line(model.vf_coef, float(np.mean(bce)), float(np.mean(pg_losses))),
          flush=True)
