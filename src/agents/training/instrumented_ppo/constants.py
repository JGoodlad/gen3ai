"""The four module-level tuning constants of the PPO training fold.

Kept together, and in the leaf of the package's import graph, because `ppo.py`, `value_terms.py`
and `noise_scale.py` all read them and none of them may import each other.
"""

# Fraction of each minibatch that forms the "tail" for the tail-weighted value loss — the worst
# _VALUE_TAIL_FRAC by squared value error (the V-tail craters the critic under-prices). 0.1 = worst
# 10%; loosely tracks the eval/td_resid_tail CVaR@5% diagnostic the loss is meant to pull down.
_VALUE_TAIL_FRAC = 0.1

# Win-prob closeness threshold: a decision is "contested" (the band where the head's value lives — a
# blowout's P(win) is trivially recoverable from material) when |normalized material margin| < this.
# margin ∈ [−1,1] = Φ_mat/bound; bound ≈ 19.5, so 0.25 ≈ a material lead of up to ~1.5 mons.
_WIN_CONTESTED_TAU = 0.25

# MOVE-latent VICReg variance floor (the belief-latent leg that also used it is DELETED, v75): a
# hinge `relu(_LATENT_STD_TARGET - std)` per latent dim pushes the predicted latents to stay spread
# (≈unit std), the belt-and-braces collapse guard on top of the stop-grad + task-anchored target.
# Weighted by _LATENT_VICREG_WEIGHT inside the move-latent loss. The `movelatent_std` metric (mean
# per-dim std) is the NO-GO monitor: std→0 while cosine→1 is collapse.
# (moved to belief_bank with the latent loss; re-exported below for old imports)

# Gradient-noise-scale EMA decay (McCandlish et al. 2018, "An Empirical Model of Large-Batch
# Training"). The single-step estimates of |G|² (true-gradient squared norm) and tr(Σ) (per-example
# gradient-variance trace) are noisy; their RATIO B_simple = tr(Σ)/|G|² is unstable per step, so we
# EMA the numerator and denominator SEPARATELY (this constant) and divide the smoothed values. 0.99
# ≈ a few-hundred-train()-call window — long enough to denoise, short enough to track drift.
_NOISE_SCALE_EMA_DECAY = 0.99
