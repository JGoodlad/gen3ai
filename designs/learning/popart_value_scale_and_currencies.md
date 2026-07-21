# PopArt, value scale, and the two-currency boundary

**TL;DR:** PopArt splits the value system into two currencies — normalized z-space (what the
network predicts, what losses see; O(1) forever) and raw return space (what rewards/GAE mean) —
with one conversion boundary `denorm(v) = μ + σ·v`. It exists because raw-scale value gradients
swamp the shared trunk (measured: `value_share` 1.0 → 0.39, ELO 1400 → 1850 when shipped). The
distributional head lives in z-space ON PURPOSE (a fixed categorical support and a drifting raw
scale are natural enemies; normalized returns are scale-stationary so a fixed support works
forever). Corollaries: (1) any comparison/distill/GAE hookup across the boundary must name its
currency — the 2026-07-21 E[Z]-vs-V probe first read slope 0.066 = exactly 1/σ (a currency
error), and in matched units read pearson 0.988 / slope 1.03; (2) the ±12 support is 12σ in the
head's actual units — nothing legitimate lives there; narrow to ~±4 (the head occupies ±2.2σ);
(3) exact output-preservation under μ/σ updates exists only for LINEAR heads — the categorical
head's raw meaning drifts slightly and is re-fit continuously (negligible at converged σ; watch
`pit_mean` early in a fresh run).

## Intuitive level

RL returns drift in scale as the policy improves; a critic regressing raw returns emits
gradients proportional to that scale into the SHARED trunk — the policy's gradient drowns
(the pre-PopArt pathology, measured). PopArt: keep running μ/σ of returns, train the head on
(G−μ)/σ so the optimizer always sees O(1) numbers, and when μ/σ move, counter-adjust the
scalar head's last linear layer so the DENORMALIZED output is unchanged ("preserving outputs
precisely") — statistics drift without yanking the function.

The mental model: the network earns and thinks in z-dollars; the environment pays and settles
in raw-yen; there is exactly one exchange window (`_denorm` in `predict_values`). Every
consumer must know which side of the window it stands on. The scalar readout hands out yen;
the `ValueDistHead`'s E[Z] is quoted in z-dollars (its CE targets are normalized before the
loss — see `_value_dist_loss`'s docstring).

## Why the categorical head is z-native

Its support (atoms) is a frozen buffer. A fixed raw-space support sized at any one point of
training is wrong at every other point as return scale drifts. Normalized returns are
stationary BY CONSTRUCTION (μ≈0, σ≈1 for the whole run) → one fixed support serves forever,
and the CE gradients stay O(1) (the same anti-swamping protection the scalar gets). This is
also why support narrowing (±12 → ±4) is drift-safe: PopArt guarantees the z-distribution
never wanders.

## Where this bit us / where it will bite again

- The Phase-B continuity probe (`tmp/dist_vs_scalar_probe.py`): raw-vs-z comparison looked
  broken (slope 1/σ, bias from μ, "47% of V beyond support"); matched-units comparison showed
  near-identity (tails best: 0.993, zero sign flips). Rule: NAME THE CURRENCY in every
  cross-head comparison, distillation target (`_value_distill_mse` takes a popart arg for
  this reason), and the Phase-B GAE hookup (GAE reads denorm(E[Z]) — same window as the
  scalar).
- Fresh-run early phase: μ/σ move fast; the categorical head's raw meaning swims until they
  settle (no exact preservation identity for non-linear heads) — HL-Gauss + continuous refit
  absorb it; watch `pit_mean` ≈ 0.5.

## See also
- `src/agents/model/popart.py`, `policy.py` `_denorm` (the one exchange window)
- `designs/ai_v6/design_distributional_value_critic.md` (v29) + `designs/ai_v8/next_run_plan.md`
  item 2 (the measured Phase-B warm-start verdict)
- [[regularization_and_noise_in_ppo]] — the sibling "where may X live" rule for randomness
