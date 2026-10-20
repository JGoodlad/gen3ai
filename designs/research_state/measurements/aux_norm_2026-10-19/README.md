# F-LG-12 — subset-normalised loss terms under gradient accumulation (2026-10-19)

TECH_DEBT P1 row "The aux / belief loss terms' accumulated means are UNAUDITED on both learner paths" (M5 Lane G
F-LG-12). The question: every loss term in the learner's region R1 (`instrumented_ppo/micro_step.py`) is a mean
over its micro-batch, and gradient accumulation averages K micro-batch losses into one optimizer step. A term whose
mean runs over a per-micro SUBSET (the rows or slots that carry a label) is then a MEAN OF PER-MICRO MEANS: a
labelled row in a sparsely-labelled micro-batch weighs more than one in a densely-labelled micro-batch. How much,
on real rows, at the shapes the learner actually trains at?

## Method (`measure.py`, CPU only, run under rule 5b)

1. **Rows** (`collect`): one real complete-game rollout on the Rust collector at the PRODUCTION surface, 32 envs x
   256 steps = 8,192 rows: the K9(a) golden's seeded learner plays p1 (the same `TK.fresh_model` seeds), p2 a seeded
   random external route (the golden buffer's recipe at a bigger shape).
2. **Atoms** (`atoms`): the rollout, after the learner's own intent-label alignment, is cut into 512 atoms of 16
   rows (one `rollout_buffer.get` permutation). The REAL R1 (`micro_step`, the production `MicroStatic`: every
   belief / intent / win-prob term on) runs on each atom with grad enabled (as in `train()`), and every
   subset-normalised term is split into its numerator S (the summed per-element loss) and its count n, read back
   from R1's own metrics (the hidden-team set rows from `set_bce`); `S / n` reproduces R1's per-term loss on every
   atom (the split's self-check: max relative error printed by the run).
3. **Shapes** (`simulate`): S and n are additive, so a micro-batch of M rows is M / 16 random atoms. For each
   shape, 4,000 accumulation groups are drawn (atoms with replacement: the live rollout is 12x this pool) and each
   term's two normalisations compared — per-micro, TODAY (`(1/k) Σ_m S_m / n_m`, an empty micro contributing 0
   as R1's `present` gate does), and whole-group (`Σ S_m / Σ n_m`).

Reported per term: labelled elements per row; the per-micro count's CV (= the spread of the per-row weight
n̄ / n_m that today's normaliser applies); the share of empty micro-batches; and the relative difference of the
two normalisations — its MEAN is the systematic shift of the term's effective coefficient, its SD the extra noise.

## Results

**PENDING** (2026-10-20): the CPU run is queued for a heavy slot (rule 5b). It will land in a later commit as
`run.log` + `simulate.json`, run per chunk (`collect`, `atoms`, `simulate`; each ≤ 45 min in its slot), with the
decision in [`designs/training/step_size_and_batch.md`](../../../training/step_size_and_batch.md)
"Subset-normalised terms under accumulation". A dry run on the K9(a) golden buffer (64 rows, 4 atoms) exercised
every step, and the split's self-check read 0.0.

## Caveats (each a FINDING in the report)

- The p1 policy is a FRESH (seeded, perturbed) production network, p2 random — not a trained agent in self-play.
  Per-row label rates follow the games; a trained agent's games are longer and reveal more. The conclusion rests
  on the per-micro counts being large (thousands at the live micro-batch), which holds for any plausible rate.
- The move-latent term's VICReg floor is a BATCH STATISTIC (the std of the micro-batch's predicted latents), not a
  count normaliser; it is not decomposable into atoms and is not measured here (only its cosine part).
- `opp_intent`'s rows are weighted (`--intent-label-bot-weight` 0.25 on bot rows) with the supervised-row COUNT as
  denominator (`belief_losses.md`); the split keeps that definition.
