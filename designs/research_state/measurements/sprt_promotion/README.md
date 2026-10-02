# SPRT promotion — Monte-Carlo operating characteristics (T6, `gen3_sprt_promotion_v1`, 2026-10-01)

**What it measures.** The false-promotion and false-rejection rates of `agents.training.sprt` (the
promotion test behind `--promotion-sprt`), plus the expected / median / p95 mirrored pairs to a
verdict. The true per-game score of a candidate vs the pool is set to 0.45 / 0.50 / 0.525 / 0.55 / 0.60.
The simulator runs the shipped test exactly as it runs: the same pentanomial GSPRT (its vectorized LLR
and increment variance are asserted equal to the shipped scalar code before any cell runs), checks only
at 40-pair batch boundaries, the minimum, the cap.

**The game model (an assumption, stated).**
- Five sentinels, played round-robin, with logit offsets 0.4 × (−1, −0.5, 0, 0.5, 1).
- A per-pair TEAM IMBALANCE d ~ N(0, τ²): game 1 is won w.p. σ(m + o + d), the mirrored game 2 w.p.
  σ(m + o − d). τ ∈ {0, 1, 2} runs from "no team effect" to "strong".
- No draws.
- m is solved so the expected score is exactly the stated rate.

**How the schedule was chosen.** The rule was declared before the run; see `choice_rule` in
`results.json`.
- **Cap:** the untruncated Wald test's worst-cell p95 = **1,680 pairs** (rate 0.525, τ 0).
- **Minimum:** the smallest of {40, 80, 160} holding both error rates ≤ 5% by the Monte-Carlo margin
  (the upper 95% bound) = **40**, i.e. the first batch.
- **Bounds:** Wald, unless Siegmund-corrected bounds also hold by that margin AND spend ≤ 85% of the
  pairs. Here they did not: they saved ~21% of pairs, but held the rates at nominal only within
  simulation noise (false promotion 5.07% at min 40).

**Result (shipped rule `wald/min40`, cap 1,680, 20,000 runs per cell):**
- False promotion at a true 0.50: 2.7–2.8%.
- False rejection at a true 0.55: 2.6–2.8% (the batch-overshoot margin under the nominal 5%).
- At 0.525 (the indifference point): promoted about half the time; 0.9–4.7% of tests end at the cap.
- Expected pairs at 0.50: 225–346, depending on τ.
- p95 is at most 1,680 pairs.

Full tables: `results.md`.

**Rerun** (CPU, about 3 minutes, deterministic for a seed):

    export PYTHONPATH=$PWD/src
    python designs/research_state/measurements/sprt_promotion/simulate.py --runs 20000 --seed 0

`src/agents/training/sprt_test.py` pins the shipped `SprtConfig` to this directory's `results.json`.
