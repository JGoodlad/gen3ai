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

Run 2026-10-20 (`run.log`, `simulate.json`; `collect` 2026-10-20 01:05, `atoms` and `simulate` each in its own heavy
slot, CPU, `nice -n 19`, 1.0 GB and 0.03 GB peak). 8,192 rows (32 envs x 256 steps) -> 512 atoms of 16 rows; the
split's self-check read **0.00e+00** (S / n reproduces R1's own per-term loss on every atom). Each cell is 4,000
simulated accumulation groups.

**The live shape: micro 2048, K 32, 98,304 rows (production / X26 / X35 / the toy)**, groups per epoch `[32, 16]`.
"CV n_m" is the spread of the per-micro count, i.e. of the per-row weight `n_bar / n_m` that today's normaliser
applies. "bias" is the MEAN of (per-micro mean of means / whole-update mean - 1) = the systematic shift of the
term's effective coefficient; "sd" is the extra group-to-group noise.

| term | labelled / row | CV n_m | empty micros | bias (mean) | sd |
|---|---|---|---|---|---|
| ev | 5.354 | 0.0048 | 0 | +3.4e-06 | 6.6e-06 |
| hp_type | 2.019 | 0.0105 | 0 | -5.8e-06 | 2.7e-05 |
| item | 5.354 | 0.0048 | 0 | +2.5e-06 | 2.6e-05 |
| move_belief | 5.648 | 0.0029 | 0 | +4.7e-06 | 3.1e-06 |
| move_latent (cosine) | 5.354 | 0.0048 | 0 | +7.2e-06 | 9.3e-06 |
| nature_ce | 5.354 | 0.0048 | 0 | +2.1e-06 | 1.1e-05 |
| opp_intent | 0.718 | 0.0142 | 0 | +3.7e-06 | 2.0e-05 |
| set_presence_bce | 0.284 | 0.0343 | 0 | -4.7e-06 | 1.2e-04 |
| set_beliefhead_bce | 0.284 | 0.0342 | 0 | -3.5e-07 | 1.1e-04 |
| set_hyp_moves_bce | 0.294 | 0.0507 | 0 | -1.8e-06 | 7.2e-06 |
| spread | 5.354 | 0.0048 | 0 | +2.1e-06 | 7.5e-06 |
| win_prob (the critic) | 1.000 | 0 | 0 | 0 | 0 |

Largest |bias| 7.2e-06 (the 4,000-group standard error of a mean is sd / 63, up to 1.9e-06, so several cells are
indistinguishable from 0); largest sd 1.2e-04. No micro-batch is ever empty. The worst per-row weight spread is 5 %
(`set_hyp_moves_bce`, the term with the fewest labels).

The other two shapes, for scale (full table in `simulate.json`): the K9(a) golden (micro 16, K 3) has |bias| up to
2.3e-02 and empty micros up to 2.3 % on the set terms, which is why the shape matters; the `--debug` smoke
(micro 256, K 9) has |bias| <= 5.3e-05.

**Decision (F-LG-12 part 1): CLOSED, nothing to fix.** The materiality bar is a coefficient shift of 1e-3 (0.1 %), a
bar this measurement sets itself (UNREGISTERED), chosen far below any coefficient change an A/B could resolve; the live shape sits two orders
under it (largest bias 7.2e-06) because a 2048-row micro-batch carries thousands of labels per term. The costed exact
fix (host-side group counts for the obs-determined terms, a no-grad pre-pass for the model-dependent ones) is NOT
built. Re-open only if the micro-batch drops toward the golden's 16-row scale (a `--batch-size` under ~256 on a
sparsely labelled term) or a new subset-normalised term is added (rerun `measure.py` and look at "labelled / row").

## Caveats (each a FINDING in the report)

- The p1 policy is a FRESH (seeded, perturbed) production network, p2 random — not a trained agent in self-play.
  Per-row label rates follow the games; a trained agent's games are longer and reveal more. The conclusion rests
  on the per-micro counts being large (thousands at the live micro-batch), which holds for any plausible rate.
- The move-latent term's VICReg floor is a BATCH STATISTIC (the std of the micro-batch's predicted latents), not a
  count normaliser; it is not decomposable into atoms and is not measured here (only its cosine part).
- `opp_intent`'s rows are weighted (`--intent-label-bot-weight` 0.25 on bot rows) with the supervised-row COUNT as
  denominator (`belief_losses.md`); the split keeps that definition.
- The atoms are drawn WITH replacement from a 512-atom pool (the live rollout is ~12x larger, and a live micro-batch
  is a without-replacement draw), so the simulated sd is, if anything, slightly over-stated; the pool is one
  rollout of one seeded network, so label rates carry that seed's games (a single-pool sampling error is not
  estimated here). Both leave the margin (two orders) intact.
- The result is the bias of the LOSS VALUE's normalisation; it is read as the shift of the term's effective
  coefficient because each term's gradient is its loss' gradient with the same denominator. It does not measure
  any second-order interaction with the optimizer.
