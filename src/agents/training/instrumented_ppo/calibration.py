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
    contested Brier is already restricted to. ``None`` when the margin cannot stratify anything.

    ⚠️ A SPREAD-FREE margin is ABSENT, not "every row is contested" (gen3_tb_relevance_v1). A run
    whose composition never computed the material potential carries an identically-constant
    ``win_margin`` — 0.0 on every pre-``gen3_obs_margin_unconditional_v1`` win-prob-critic run —
    and ``|const| < tau`` is then all-ones, so the 13 ``win_prob/contested_*`` calibration tags
    emit as byte-identical copies of their pooled twins and read as a measurement of a split that
    did not happen. The predicate is ``max − min > 0``, MIRRORING ``value_terms._win_prob_loss``
    exactly: the two consumers stratify on the same column, so a run in which one publishes the
    split and the other does not would be worse than either answer alone.
    """
    if margin is None:
        return None
    m = np.asarray(margin, dtype=np.float64).reshape(-1)
    if m.size == 0 or not float(m.max() - m.min()) > 0.0:
        return None
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


def vf_coef_scale_line(vf_coef: float, bce: float,
                       policy_grad_norm: float, value_grad_norm: float) -> str:
    """One human line: what `--vf-coef` is actually doing to the SHARED TRUNK on this run.

    🚨 **`--vf-coef` MEANS SOMETHING DIFFERENT UNDER `--critic winprob`, AND NOTHING IN A METRIC
    NAME SAYS SO.** Under `shaped` it weights an MSE on a PopArt-normalised shaped return; under
    `winprob` it weights the win-prob head's **BCE against a Bernoulli outcome**, which is bounded
    near `ln 2 ~ 0.693` at initialisation and falls from there. The historical default 0.5 was
    tuned against the first quantity and carries no information about the second — so the first arm
    needs to READ the balance rather than inherit a number, and this prints it where the operator
    is already looking (the `[CRITIC] winprob` banner family).

    🚨 **IT REPORTS A RATIO OF GRADIENTS, NOT OF LOSSES, AND THAT IS THE WHOLE POINT.** The first
    shipped version of this line divided the value term by `|policy loss|`, and that denominator is
    **degenerate BY CONSTRUCTION**: on epoch 1 of a rollout PPO's clipped surrogate has
    ``ratio == 1`` exactly and sits at its stationary point, so `|policy loss| ~ 0` and any ratio
    against it is an artifact of the epoch, not a reading of the coefficient. Live evidence
    (`ai_v12_01_winprob_critic`, 2026-09-06): the loss form printed **165x** on rollout 1, where
    the gradient form is **unreadable** on that same rollout, reads **91x** on rollout 2, and has
    converged to **4.6x** by rollout 17. What competes on the shared trunk is the GRADIENT, and it
    is a quantity the run already computes — see :func:`announce_vf_coef_scale`.

    Both numbers are printed because they answer different questions and neither substitutes for
    the other:

    * **the raw BCE** is informative on its own — `ln 2 ~ 0.693` is chance, and a value well below
      it means the head is already calling lopsided games. It is a statement about the HEAD.
    * **the gradient-norm ratio** ``||g_value|| / ||g_policy||`` over
      :data:`agents.training.grad_balance.SHARED_TRUNK_PHASES` is a statement about the
      COEFFICIENT: the value norm scales linearly in `vf_coef`, so the ratio IS the factor to
      divide the coefficient by. It is exactly ``10 ** grad/value_policy_logratio``, the
      aux-independent balance gauge that keeps reading every rollout after this one.

    Deliberately a PRINT and not a scalar: it is a once-per-run sanity reading whose whole job is
    to be seen at startup by the person choosing the coefficient. The per-rollout series it would
    duplicate already exist — `grad/value_policy_logratio`, `grad/value_share` and `win_prob/loss`.

    Pure: takes the four numbers and returns the sentence, so the wording is testable without a
    PPO. A non-positive policy norm reports UNAVAILABLE rather than dividing by ~0 — an infinite
    ratio would read as "the value gradient dominates", which is a statement about a degenerate
    minibatch, not about `--vf-coef`. (:func:`announce_vf_coef_scale` refuses that case up front;
    the guard here only keeps the pure function total.)
    """
    n_pi = float(policy_grad_norm)
    n_vf = float(value_grad_norm)
    ratio = (f"{n_vf / n_pi:.3g}x" if n_pi > 0.0 else
             f"UNAVAILABLE (||g_policy|| {n_pi:.3g} is ~0 on this update)")
    return (f"\U0001f3af [CRITIC] winprob \u2014 --vf-coef {float(vf_coef):g} scale readout (first "
            f"non-degenerate update): BCE {float(bce):.4f} (ln 2 ~ {LN2:.3f} = chance; lower = the "
            f"head already calls lopsided games). SHARED-TRUNK GRADIENT: ||g_value|| {n_vf:.4g} / "
            f"||g_policy|| {n_pi:.4g} -> {ratio}. READING: >>1 the value gradient dominates the "
            f"trunk, cut --vf-coef by that factor; <<1 raise it. \u26a0\ufe0f this is a ratio of "
            f"GRADIENTS, not of losses \u2014 on epoch 1 the clipped surrogate sits at its "
            f"stationary point (ratio == 1), so |policy loss| ~ 0 and a loss ratio against it is an "
            f"artifact. And --vf-coef now multiplies a BCE, not the shaped-return MSE 0.5 was tuned "
            f"for, which on a +-30 return was O(100). Confirm with grad/value_policy_logratio (log10 "
            f"of this ratio) \u2014 see designs/ai_v12/design_winprob_only_critic.md 5.4.")


#: The floor a shared-trunk POLICY gradient norm must clear before the ratio above is a reading
#: rather than an artifact. **The threshold is on the NORM, not on the epoch, and the live arm is
#: why.** The grad-balance probe samples ONE minibatch per `train()` and by construction that
#: minibatch is in epoch 1 — so "wait until after the first epoch" cannot be the rule; there is no
#: later epoch to wait for within the same reading. Nor is epoch 1 degenerate in general: unlike
#: the LOSS, the policy GRADIENT at ``ratio == 1`` is ``A * grad log pi``, which is not zero, so an
#: epoch rule would also discard perfectly readable measurements. Reading the norm directly is both
#: necessary and sufficient. The value: `ai_v12_01_winprob_critic` measured `grad/policy_norm_shared`
#: at **exactly 0.0** on rollout 1 and **4.9e-3** on rollout 2 (~3.7 decades above this floor), and
#: a float32 norm accumulated over ~1e6 trunk parameters has a noise floor near 1e-6 — so below it
#: the number is accumulation noise, not a pull.
MIN_POLICY_GRAD_NORM: float = 1e-6


def announce_vf_coef_scale(model, bce: Optional[Sequence[float]],
                           grad_balance: Optional[Mapping[str, float]]) -> None:
    """Print :func:`vf_coef_scale_line` ONCE per process, on the first rollout that can read it.

    Stateful half, kept out of the pure function so the wording stays testable. Four properties:

    * **THE GRADIENT NORMS ARE READ, NEVER RECOMPUTED.** ``grad_balance`` is the dict
      `grad_balance_metrics` already produced for this `train()` — the read-only
      `autograd.grad(retain_graph=True)` probe that has run per-term on every rollout for
      generations. Under `--critic winprob` its ``grad/value_norm_shared`` is the norm of
      ``vf_coef * BCE`` (the term as folded) and ``grad/policy_norm_shared`` is
      ``policy_loss + ent_coef * entropy_loss``, both over the same shared-trunk parameter set. A
      second backward pass here would cost a rollout's worth of graph and could disagree with the
      series the operator is told to confirm against.
    * **ONCE**, latched on the model (`_vf_scale_announced`), because it answers a question about
      the RUN's configuration, not about this rollout.
    * **NEVER on an update it cannot read, and it does NOT latch there.** No scorable win-prob
      label (an absent or EMPTY `bce`), no grad probe at all (a non-Gen3 extractor yields `{}`), a
      non-finite norm, or a policy norm under :data:`MIN_POLICY_GRAD_NORM` all produce no line and
      leave the latch clear, so the next rollout tries again. The live arm needed exactly this: its
      rollout-1 probe read a policy norm of 0.0 against a value norm of 7.53, and the first honest
      reading is rollout 2's.
    * It goes to stdout via `print`, the same channel as `train_rl_agent`'s own `[CRITIC]` banner,
      so a launcher-managed run finds both in `launcher_child.log` and a bare run finds both on
      its terminal.
    """
    if getattr(model, "_vf_scale_announced", False):
        return
    if bce is None or not len(bce) or not grad_balance:
        return
    n_pi = float(grad_balance.get("grad/policy_norm_shared", 0.0))
    n_vf = float(grad_balance.get("grad/value_norm_shared", 0.0))
    if not (np.isfinite(n_pi) and np.isfinite(n_vf)):
        return
    if n_pi < MIN_POLICY_GRAD_NORM or n_vf <= 0.0:
        return
    model._vf_scale_announced = True
    print(vf_coef_scale_line(model.vf_coef, float(np.mean(bce)), n_pi, n_vf), flush=True)
