# TD-consistency auxiliary (delta-denoising the critic)

**Status:** 🔬 open — rung 1 PASSED, rung 2 (fork A/B) licensed and pre-registered below · **Ledger id:** C5

One-line claim: *the critic's excess ΔV dispersion on trade transitions (C4) is injected noise that
an explicit Bellman-residual term can suppress — per-state MSE never constrains adjacent-state
differences, so the delta inherits ~double the state noise exactly where the truth is nearly
constant.*

## Known (established)

- **Rung 1 (2026-08-17, `measurements/gen13_td_aux_gate.json`): PASS.** Same frozen-token
  population as the C4 gate (87,064 decisions / 2,111 battles / 402 self-KO transitions), same
  pooled head, only the loss changes: `MSE(V, G99) + λ·(V(s_t) − r_t − γV(s_{t+1}))²`, both
  residual ends live, whole-battle segment batches. Pre-registered gate met at **λ=1.0**
  (self-KO ΔV RMSE 4.951 → 4.351, CI95 on the improvement [+0.18, +1.08]) and **λ=3.0** (→ 3.989,
  CI [+0.58, +1.38]); the +5% no-harm budget on overall value RMSE **never bound anywhere**
  (λ=3.0 is actually the best overall arm, −0.44%). Post-hoc extension λ=10/30 (excluded from the
  gate) continues monotonically: 3.23 / 2.82, dispersion ratio 3.37× → 1.83×.
- **Decomposition: it is dispersion suppression, not bias shift** — error std falls monotonically
  (4.94 → 2.79) while bias stays small and non-monotone; and it is **targeted**, not global
  shrinkage (delta std shrinks 1.1–1.4× faster than level std).
- **The ceiling, stated honestly:** ΔV-truth Pearson goes −0.221 → **−0.013 — toward zero, never
  positive**. The loss removes injected noise; it cannot create per-transition signal the tokens
  do not carry (C4's finding). The λ→∞ limit of this mechanism on self-KO ΔV *is* the constant
  predictor (1.33), and even λ=30 sits 2.1× above it.
- Non-monotone low end: **λ=0.1 is significantly WORSE than control** (CI [−0.52, −0.04]) —
  a small TD weight perturbs the optimum without constraining it. Avoid the λ≤0.1 regime.
- Incidental: whole-battle batching alone beat the C4 probe's random-permutation baseline by 12%
  (4.951 vs 5.622 control) — the rung-1 control is the *harder* baseline, and the
  segment-minibatch plumbing the training-loop version needs is the same trick (K+1 contiguous
  forwards serve K pairs, so the "second forward" cost mostly dissolves).

## Not-known

- Whether the mechanism survives **live training dynamics**: GAE/PopArt targets instead of
  return-to-go, a co-adapting trunk, and PPO's optimization interplay. Rung 1 tests the loss on a
  frozen representation only.
- Whether delta-denoising moves **behavior**: the expected payoff channel is advantage denoising
  (spurious ΔV variance is GAE noise on every transition, not just self-KO), which should show as
  cleaner credit assignment — but that is a hypothesis until rung 2.
- The right λ under live training (rung 1's band: 1.0–3.0; the sweep's shape suggests higher may
  be tolerable, but 10/30 were post-hoc).

## Pros

- Philosophy-clean: enforces the Bellman identity the critic is already supposed to satisfy — an
  ESTIMATOR fix, no reward bias, no opinion about Pokémon injected (unlike the layer-1
  `--self-ko-hp-penalty`, which is a BIAS-class tourniquet).
- Cheap at every stage: rung 1 cost one agent-hour; the training-loop change is a loss term + a
  segment-contiguous sampler over the existing `[n_steps, n_envs]` buffer; coef-0 byte-identical,
  training-coefficient class, no signature bump.
- Both C4 findings compose: tokens lack trade signal (so no readout fixes it) + the variance is
  injected noise (so a consistency term CAN remove it). The two probes together locate the fix.

## Cons

- The ceiling is real: this cannot make the critic *rank* trades; it can only stop it hallucinating
  variance. If H1's behavioral harm requires positive per-trade discrimination, this alone won't
  close it — that half belongs to the representation line (ai_v10 / elicitation).
- Baird residual-gradient double-sampling bias: minimizing the per-sample residual biases V toward
  reducing target conditional variance. Here that is close to the medicine, but under live
  stochastic transitions the coefficient is a bias-variance dial — gate on level calibration
  (explained variance, pool-PIT) not regressing.
- A loss term is not post-hoc ablatable (its effect lives in the weights) — attribution needs its
  own arm/generation, per the one-behavioral-change discipline.

## Next test — rung 2, the fork A/B (pre-registered here)

Fork gen-13's (or the then-current base) checkpoint, ~2–4M steps per arm, aux OFF vs ON at
**λ=1.0 and λ=3.0** (two ON arms; λ=3.0 is the favorite). Gates, fixed now:

1. **Mechanism**: recorded self-KO ΔV dispersion ratio vs truth falls from ~4.5× toward ≤2×
   (the C4 instrument, measured on the fork's own eval traces).
2. **Behavior**: explosion-when-available rate and `decision-table --cat selfko` dV_med move
   toward the reward's verdict.
3. **No-harm**: value explained-variance and the pool rollout-PIT level gap flat-or-better; quick
   bot eval non-inferior.

Any gate failing → re-tune or kill before a generation slot. **Pass → gen-15 headline arm**
(NOT gen-14 — the frame deletion rides alone), and the flywheel era inherits the denoised critic.
