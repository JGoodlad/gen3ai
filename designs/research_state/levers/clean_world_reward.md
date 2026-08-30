# The CLEAN WORLD — terminal ±1 plus an outcome-grounded potential, and nothing else

**Status:** 🟡 **REGISTERED + flag-reachable, NOT launched** · **Ledger:** `579279d` (registered) ·
`4d22ae4` (the famine correction) · `cfbc9bf` (probe N amendments) · `e22bd08` (the three-arm
ladder) · `627ab58` (draw = loss, no launch bias) · `2d38a4a` (PopArt retirement) · `db9bb5c`
(turn-zero mechanics) · `132d198` (wave A landed, config v105)

One-line claim (owner's, adopted verbatim as the purpose line): *"if we can get to a conceptually
cleaner world view, we liberate the complexity hidden in the hand-tuned terms."*

## Known (cleared the honesty gates)

- **The scaffolding is already nearly gone, and nobody knew (`1d5a866`).** Of 29 registry BIAS
  members, `--all-shaping-pbrs` (default-ON since 2026-08-18) leaves exactly ONE live: production
  reward is **1 TERMINAL + 7 PBRS + 1 BIAS**. The remembered anti-stall family has not run in
  months. The clean world is therefore a *smaller* step than it sounds.
- **The arm structure (`e22bd08`), all at terminal {+1, −1} with draw = −1.** Three potential
  sources, and **every pairwise difference is a named quantity**: **SPARSE** (no potential — the
  famine test) · **SELF-φ** (the run's own live head, co-evolving) · **FROZEN-φ** (a mature
  prior-generation head). SELF − SPARSE = the value of self-shaping. FROZEN − SELF = the value of
  maturity plus exact-vs-approximate invariance. FROZEN − SPARSE = the total worth of
  outcome-grounded shaping. The incumbent comparison comes FREE via existing rev-1-class 25M
  checkpoints (h2h + anchored ELO), with era-config differences named as the imperfection.
- **FROZEN-φ solves two problems at once (`579279d`).** A fresh run's own head is noise at step 0 —
  PBRS from a noise potential is harmless (invariance) but helpless. A frozen mature head gives
  dense outcome-grounded shaping from step 0 **and restores the PBRS invariance theorem EXACTLY**,
  which is what makes the learned-drifting-φ caveat vanish for that arm.
- **🚨 draw = −1 is load-bearing (`cfbc9bf`, `627ab58`).** {+1, −1, 0} would make the 250-turn stall
  the **best non-winning outcome** in an arm with zero anti-stall terms — inverting the ordering the
  old draw_penalty (−35 < −30) exists to enforce. **Stall rate is promoted to a PRIMARY endpoint.**
  Owner's framing: "a tie is just as bad as a loss."
- **NO anti-stall bias at launch; escalation must be EARNED (`627ab58`).** With draw = loss,
  stalling is **weakly dominated** — any line with ε win probability beats it — so a bias is needed
  only if the model lands in a can't-win-won't-lose local optimum, which is an EMPIRICAL condition,
  not a theoretical one. Pre-registered: the bias enters only if the stall-rate endpoint fires, and
  enters as a registered change, never a mid-run patch. Probe O then measured the exposure and
  **the trigger is NOT fired — the no-bias launch STANDS** (`32c39df`).
- **🚨 "Shaping was necessary" is NARRATIVE (`4d22ae4`).** The famine claim was stated as fact and is
  unmeasured lore: **no sparse-reward arm has ever run in this project.** The original reward design
  predates PBRS knowledge and any understanding of what the features could represent. Evidence cuts
  both ways — AlphaZero-class sparse success is a different regime (scale + search targets), but our
  own win-prob head trains well on pure MC outcome labels at our scale (calibrated −0.011, 96.4%
  whiff knowledge — supervised prediction, not control credit, yet proof the representation learns
  from outcome-only signal. Hence the PURE-SPARSE arm and a **paired ~5M sparse-vs-shaped pre-test**
  that sizes the full arms in hours before three generation-scale runs are committed.
- **The coefficient SPELLING is `coef 2` on φ = p, never 2p − 1 (`cfbc9bf`).** The affine constant at
  γ < 1 pays a per-step bonus for LONGER episodes (wrong sign), and the terminal φ := 0 convention is
  correct for [0, 1] and wrong for [−1, +1].
- **Turn-zero mechanics (`db9bb5c`).** PBRS pays TRANSITIONS only — φ(s₀) is never paid in.
  Constants are free by construction (the critic baseline absorbs them); **offsets are charged**
  (γ<1 drift + terminal refund). *The centering intuition INVERTS under PBRS — supervised-learning
  instincts about output means do not transfer to differential-payment schemes.*
- **Correction of record (`db9bb5c`):** "V becomes directly readable as expected outcome" holds for
  the **SPARSE arm and the win-prob head only**. In shaped arms a good potential drives V_shaped
  toward a **CONSTANT** (the classic φ = V* result — all evaluative content migrates into the reward
  stream); the outcome-readable quantity is V_shaped + coef·φ, which is what the scaffolding gauge
  computes. **The near-constancy of V_shaped in the frozen-φ arm is itself a checkable prediction**,
  and it shipped as a one-line sanity row (`4867537`).
- **PopArt RETIREMENT registered for the sparse and clean arms (`2d38a4a`).** Its job — scale-30
  drifting shaped returns swamping the trunk — is deleted by the reward design. Buys one fewer
  moving part, cleaner weight-drift diagnostics (PopArt's pop-rescale confounded the plasticity
  audit), a simpler vf_coef/resume story. **±1's real gift is STATIONARITY, not range** —
  policy-side magnitude comparability was already scale-free via per-minibatch advantage
  normalization, which is also why ent-coef is not rescaled.
- **It is LAUNCHABLE (wave A, `132d198`, config v105).** `--no-hand-shaping --victory-value 1.0
  --draw-penalty -1.0 --win-prob-mode read_only --win-prob-pbrs-coef <2c> --win-prob-pbrs-source
  <ckpt>` verified end-to-end through the real parser to `[Reward] composition: 1 TERMINAL + 0 PBRS
  + 0 BIAS`; checkargs exit 0. Honest deviation: B1's individual gates alone cannot reach zero (six
  potentials die only via the anti-correlated `--all-shaping-pbrs` flag), so probe N's
  `hand_shaping` MASTER was required — and the gap is pinned by a test so nobody deletes the master
  later.

## Not-known

- **Everything the arms exist to measure.** No sparse arm has run; no PBRS term has ever run at any
  coefficient.
- The stall rate under draw = −1 with zero anti-stall terms — the endpoint the whole no-bias
  decision rides on.
- Whether the frozen ancestor's φ is *good enough* off its own distribution once the student's play
  diverges from it.
- A real CUDA compile with a frozen source attached is UNEXERCISED (documented, not assumed).
- Whether a PopArt-free near-sparse stream needs a value-loss-scale tripwire in practice (probe N's
  σ-collapse caveat cuts both ways).

## Pros

- Retires an entire class of hidden hypotheses at once — and the switch tax proves the class is
  real, not hypothetical (see `no_progress_switch_tax.md`).
- Route 1 is the only mechanism that converts a post-whiff probability drop into *literal reward the
  policy gradient must answer for*, and PBRS's telescoping invariance means a miscalibrated
  potential costs speed, not correctness.
- The frozen-φ arm gets exact invariance, so its risk is bounded by a theorem rather than by care.
- Creates the virtuous loop: better head → better shaping → better policy → better outcome data.
- The three-arm design cannot return "no result" — sparse-trains-fine, sparse-craters and
  sparse-is-slower are three *different* findings, and the third prices the shaping↔wall-clock
  exchange rate.

## Cons

- **Cost: 2–3 × 25M fresh runs**, i.e. two to three base runs, sequenced after the rev-3/rev-4
  obligations.
- It is a DEMONSTRATION, not an ablation — many things change at once, so the tax-only arm
  (`--no-progress-penalty 0.0`, zero code) remains the causal test for the under-switching endpoint.
- The design leans hardest on the win-prob head exactly where the head is historically weakest (the
  stall tail, 34.8% residual over-confidence) — mitigated by the registered head-fix-before-bias
  ordering, not eliminated.
- A learned, drifting φ weakens exact invariance to approximate in the SELF-φ arm — named in the
  arm design as required text, not a footnote.

## Next test

**The 5M paired sparse-vs-shaped PRE-TEST** (same seeds) — hours of GPU, and it decides
crater / crawl / keep-pace before three generation-scale runs are committed. Then register the three
arms with the endpoints already frozen: **stall rate (primary)**, switch rate, whiff/loop census,
scaffolding-gauge trajectory, plus the V_shaped-constancy sanity row on the frozen-φ arm.
