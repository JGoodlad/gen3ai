# Opponent-PP observability (the stall-class blind spot)

**Status:** ❌ killed 2026-08-17 — probe ran AS REGISTERED, primary NULL · **Ledger id:** C3

One-line claim (dead in its strong form): *the critic is blind on stall losses because the
opponent's PP ledger — the quantity that decides a Gen-3 stall war — is structurally absent from
the observation.*

## Verdict (2026-08-17, `measurements/gen12_opp_pp_probe.json`)

Pre-registered in `gen13_endofrun_runbook.md` §8 BEFORE computing anything; run same-day on
gen-12 traces, zero deviations from the registration.

- **Primary NULL.** 608 battles (≥50 turns; 253 L / 355 W), 39,656 decisions at turn ≥30.
  Baseline (recorded win_prob + turn) AUC **0.8874**; + PP features **0.8848**; ΔAUC **−0.0026**,
  battle-bootstrap CI95 [−0.0178, +0.0102], permutation p = 0.12. Count-coverage 0.738 — above
  the ~0.7 floor the KILL gate required, so the null is meaningful, not an artifact of thin
  usage data.
- **Secondary below its bar.** PP-features-only AUC on the win_prob ≥ 0.7 slice (23,326
  decisions, 4,453 losses): **0.595**, CI95 [0.512, 0.668]. The CI excludes 0.5 — the PP ledger
  carries a real trace of signal when the critic is confident — but it sits far under the
  registered 0.65 bar and nowhere near enough to explain the blindness.
- The archetype confound biases TOWARD conviction, so the null is conservative.

## What survives the kill (the honest residue)

- The **facts** stand and stay useful: opponent `current_pp` IS encoded always-full
  (`moves.py:129-130`), no usage tracker exists, and opponent move usage IS public information —
  so if some future finding needs the PP ledger, the feature remains cheap and prior-clean. What
  died is the claim that its absence explains the stall blindness.
- The MC-supervision argument stands: the win-prob head trains on ground-truth outcomes
  (`win_prob_callback.py`), so its class-conditional miss still means off-distribution or
  missing input. This probe killed the best-motivated *missing-input* candidate — which moves
  the weight onto **off-distribution: the training DISTRIBUTION of stall games** (the registered
  next suspect; how much loss-side stall mass does a rollout actually train on, given self-play
  pool win rates are pinned near 50% and stall wars are a thin slice of episodes?).
- Scope note recorded at kill time (not a bar renegotiation): the population is ALL ≥50-turn
  games — the broad long-game class, not certified PP-wars. A maximally-narrow PP-war-only test
  was never registered; the secondary's weakness (0.595 on the confident slice) makes it
  unpromising, and anyone reviving it must pre-register it fresh.
- Methodological: the permutation null's mean is **−0.008** — under grouped CV, nine irrelevant
  features *cost* AUC, so a naive "did AUC go up" read without the null would have been biased
  toward killing; the real features merely lost less than noise. The bake-in-the-null rule cuts
  both ways.

## Next test

- None on this lever. The stall-blindness investigation moves to the training-distribution
  branch (measure loss-side stall mass in actual rollouts vs its share of eval losses), and the
  gen-13 §7 critic-calibration gate remains the outcome measurement either way.
