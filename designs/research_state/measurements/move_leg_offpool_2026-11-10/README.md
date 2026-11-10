# The move leg of the opponent belief, off the pool (2026-11-10): DESCRIPTIVE, no verdict

**The question.** The baseline of 2026-11-09
([`../offpool_belief_baseline_2026-11-09/README.md`](../offpool_belief_baseline_2026-11-09/README.md)) left the MOVE
leg of the opponent belief below its own Smogon prior off the training pool, unrepaired by team breadth and
unexplained ([`../../../endstate/design_offpool_belief.md`](../../../endstate/design_offpool_belief.md) §3). Is the
deficit memorisation of the pool's movesets, the conditioning on species, or something else? CPU only: no run was
trained and no game played. Nothing here is a strength claim. The ledger entry of 2026-11-10 (*MOVE-LEG DIAGNOSIS*)
is the record; what it led to is the training lever `--belief-grad-mode move_label_only`
([`../../../model/phase_pipeline.md`](../../../model/phase_pipeline.md)) and the design's §8.

Names. P\* = baseline `pstar_plateau` (`rb_x26_s2601` @280,000,170). C0 = P\* + 20M steps on the pool alone
(`rb_x35_c0_s3501`). N1-P = P\* + 20M with half the games on procedural teams (`rb_x35_n1p_s3501`). X68-FT = the
export of X68's belief-only fine-tune. Panels: `home72` (IN the pool), H = `heldout72_hpfix`, F = `forum72`,
P72 = `p72`. The head: `move logits = move_head(token) + Smogon prior[species]`, `move_head` a zero-initialised
`Linear(128, 400)`; "the residual" is `move_head`'s output.

## Method

- **Units** are the meter's (`python -m main.offpool_belief`): a (decision row, revealed opponent mon) with at least
  one true move the observation has not shown; its candidates are the legal, unshown moves outside the Hidden Power
  family. 28,057 / 25,984 / 26,443 / 26,966 units on home72 / H / F / P72, 72 teams a panel, the banks of
  2026-11-09. `scripts/dump.py` runs the meter's two forwards a row and keeps the raw logits; it reproduces the
  baseline's P\* row exactly, with 0 truth-illegal units and 0 reveal mismatches.
- **Three scores a unit**, each reported as belief − prior: *truths-only* (the meter's `moves_logp`: the mean log σ
  over the true unshown moves); the *full log-likelihood* (the Bernoulli log-probability of the truth summed over
  every candidate, a proper score: it also charges a raised move that is not there); *closed* (truths-only after
  the count closure the policy's reinjection reads under `--move-set-closure on`).
- **Statistics** as the meter: the mean over teams of team means; brackets are 95 % team-bootstrap intervals
  (10,000 draws, seed 20261109).
- **The pool as a yardstick, never a prior** (`scripts/pool_table.py`): per species, the share of the pool's sets
  that carry each move (the *pool marginal*) and the list of its sets (the *set-completer*: the marginal over the
  pool sets consistent with the shown moves). Both are what an ideal memoriser of the pool would predict.

## 1. The deficit, by checkpoint (belief − prior)

Prior levels a unit (home72 / H / F / P72): truths-only −0.874 / −0.855 / −0.998 / −1.015; full log-likelihood
−4.799 / −4.969 / −5.114 / −5.597.

| checkpoint | score | home72 (in pool) | H | F | P72 |
|---|---|---|---|---|---|
| P\* | truths-only | +0.14 [+0.06, +0.22] | −0.33 [−0.42, −0.25] | −0.05 [−0.11, +0.02] | −0.34 [−0.40, −0.28] |
| P\* | full log-likelihood | +0.39 [+0.01, +0.75] | −1.83 [−2.21, −1.46] | −0.61 [−1.07, −0.23] | −2.21 [−2.54, −1.90] |
| P\* | closed | +0.27 [+0.18, +0.37] | −0.38 [−0.49, −0.28] | −0.02 [−0.14, +0.08] | −0.47 [−0.55, −0.40] |
| C0 | truths-only | +0.15 [+0.07, +0.22] | −0.33 [−0.41, −0.25] | −0.05 [−0.11, +0.02] | −0.34 [−0.40, −0.28] |
| C0 | full log-likelihood | +0.42 [+0.01, +0.78] | −1.81 [−2.20, −1.45] | −0.60 [−1.06, −0.21] | −2.20 [−2.52, −1.89] |
| N1-P | truths-only | +0.09 [+0.02, +0.16] | −0.30 [−0.39, −0.22] | −0.14 [−0.20, −0.08] | −0.36 [−0.42, −0.30] |
| N1-P | full log-likelihood | +0.48 [+0.11, +0.81] | −1.40 [−1.74, −1.08] | −0.52 [−0.93, −0.20] | −1.86 [−2.18, −1.55] |
| X68-FT | truths-only | −0.00 [−0.08, +0.08] | −0.33 [−0.42, −0.25] | −0.14 [−0.21, −0.06] | −0.37 [−0.43, −0.31] |
| X68-FT | full log-likelihood | −9.22 [−9.86, −8.57] | −10.97 [−11.53, −10.42] | −10.34 [−11.12, −9.58] | −11.71 [−12.32, −11.10] |

- The deficit is 5 to 6 times larger on the proper score than on the meter's column, and F, which the meter reads
  as NOT DETECTED, is below the prior on it.
- It is not a mass effect: P\*'s belief puts LESS total probability on a mon's candidates than the prior (3.50 /
  3.66 / 3.61 / 3.72 against 3.85 / 3.83 / 3.83 / 3.88; the true count is 2.5), and the closed score agrees in sign.
- 20M more pool-only steps change nothing (C0's residual correlates 0.99 with P\*'s). Breadth moves the residual
  (N1-P: correlation 0.89 to 0.91, RMS difference ≈ 1.0 logit) and recovers a fifth of the deficit on H and P72.

## 2. What the deficit is made of (P\*)

| | H | F | P72 |
|---|---|---|---|
| true unshown moves whose (species, move) pair the pool attests: share, truths-only | 0.958, −0.29 | 0.945, −0.01 | 0.941, −0.29 |
| pair the pool never shows: share, truths-only | 0.019, −1.68 | 0.042, −1.30 | 0.045, −1.54 |
| species absent from the pool: share, truths-only | 0.023, −0.29 | 0.013, −0.24 | 0.015, −0.61 |
| units whose full moveset IS a pool set: share, truths-only | 0.39, −0.13 [−0.22, −0.04] | 0.71, +0.04 [−0.03, +0.11] | 0.31, −0.19 [−0.28, −0.10] |
| units whose moveset is not a pool set: truths-only | −0.43 [−0.54, −0.34] | −0.24 [−0.38, −0.10] | −0.42 [−0.50, −0.34] |

Moves the pool never pairs with the species are punished hard but are 2 to 5 % of the truths: 94 to 96 % of the
deficit's units are familiar moves in an unfamiliar COMBINATION, and the belief is below the prior even on mons
that carry an exact pool set (H, P72): it has learned which set goes with which TEAM.

## 3. The residual is a species × move table, and it is over-confident

| P\* | home72 | H | F | P72 |
|---|---|---|---|---|
| residual RMS, logits | 2.17 | 2.21 | 2.14 | 2.11 |
| share of its mean square that is a fixed (species, move) offset | 0.834 | 0.836 | 0.826 | 0.823 |
| mean residual on true / on non-true candidates | +1.17 / −1.48 | +0.68 / −1.38 | +0.97 / −1.36 | +0.39 / −1.30 |
| residual scale that maximises the full log-likelihood | 0.5 | 0.125 | 0.375 | 0.125 |

Truths-only / full log-likelihood, belief − prior, of predictors built from P\*'s residual and from the pool:

| predictor | home72 | H | F | P72 |
|---|---|---|---|---|
| P\* | +0.14 / +0.39 | −0.33 / −1.83 | −0.05 / −0.61 | −0.34 / −2.21 |
| only the static table (fitted on home72) | +0.00 / +0.34 | −0.29 / −1.09 | −0.20 / −0.40 | −0.28 / −1.13 |
| only what is left after the in-panel static table | +0.14 / −0.27 | −0.08 / −0.94 | +0.08 / −0.55 | −0.09 / −1.08 |
| IDEAL pool marginal | +0.11 / +1.22 [+0.98, +1.46] | −0.15 / +0.06 [−0.09, +0.21] | −0.18 / +0.19 [−0.04, +0.40] | −0.27 / −0.40 [−0.62, −0.20] |
| IDEAL pool set-completer | +0.22 / +2.17 [+1.91, +2.43] | −0.56 / −0.35 [−0.57, −0.13] | −0.17 / +0.83 [+0.59, +1.07] | −0.70 / −1.00 [−1.22, −0.78] |
| residual × 0.25 | +0.09 / +0.63 | −0.04 / +0.06 | +0.03 / +0.37 | −0.04 / −0.04 |
| residual × 0.5 | +0.14 / +0.86 | −0.11 / −0.27 | +0.03 / +0.34 | −0.12 / −0.47 |
| its raises only | +0.36 / −0.65 [−0.97, −0.38] | +0.18 / −2.04 | +0.28 / −1.26 | +0.20 / −2.30 |
| its lowerings only | −0.22 / +1.04 [+0.91, +1.17] | −0.51 / +0.22 | −0.32 / +0.64 | −0.54 / +0.09 |
| bounded: 1.0 · tanh(residual / 1.0) | +0.10 / +0.86 [+0.73, +0.99] | −0.11 / +0.03 [−0.07, +0.13] | +0.01 / +0.45 [+0.35, +0.54] | −0.12 / −0.08 [−0.17, +0.00] |
| floored on the prior at 0.25 | +0.17 / +0.72 | −0.13 / −0.47 [−0.63, −0.31] | +0.04 / +0.12 | −0.13 / −0.61 |

- **Conditioning on species is most of it.** 83 % of the residual does not depend on the battle at all, and that
  static table alone carries 60 % of the log-likelihood deficit on H, 66 % on F and 51 % on P72.
- **It is not what a memoriser of the pool would do.** The pool's own per-species marginal beats the prior on the
  proper score on H and F and is +1.22 in the pool, where P\* is +0.39. P\*'s residual correlates only 0.45 to 0.56
  with either ideal memoriser: it is a noisier and more extreme version of the pool's table. In the pool its
  best scale is 0.5: **the head is over-confident against its own labels where it was trained.**
- **The harm is in the raises.** Keeping only the lowerings beats the prior on every panel on the proper score;
  keeping only the raises is below it everywhere, in the pool too.
- What breadth cannot remove: even the ideal pool marginal is 0.15 to 0.27 below the prior truths-only off the pool.

## 4. Which loss does that: gradients into `move_head` at P\* (in-pool rows)

Routes (`designs/model/phase_pipeline.md`): A = the head's label loss; C = every loss that reads the head's
PUBLISHED posterior (the policy, the critic, and the species and item belief losses downstream of it).

`results/gradroute3.log`, two 128-row batches, gradient norm into `move_head.weight`:

| term | batch 0 | batch 1 | under the built `move_label_only` |
|---|---|---|---|
| the move label loss, 0.05 × BCE (route A) | 1.75e-04 | 1.74e-04 | unchanged, bit for bit |
| the value loss, 0.5 × MSE (route C) | 6.18e-03 (35× the label) | 5.19e-03 (30×) | NO PATH |
| the entropy bonus, 0.05 × H (route C) | 6.12e-03 | 1.03e-02 | NO PATH |
| the policy's log-probability, unweighted (route C) | 1.58e-01 | 2.48e-01 | NO PATH |

Cosine of the value gradient with the label gradient: +0.11, −0.07. Under `--move-set-closure on` the head's own
`publish_detach` cuts nothing (the closure hands the reinjection explicit weights); only the extractor-level
boundary the lever adds does.

`results/gradscale_home72.log`, 8 batches × 128 rows: d(term) / d(residual scale) at scale 1 (positive = the
term's descent SHRINKS the residual), and the gradient averaged over the batches:

| term at its production coefficient | d / d(scale): mean [min, max] | batches < 0 | norm of the mean gradient | route |
|---|---|---|---|---|
| move label BCE × 0.05 | +2.46e-04 [+2.26e-04, +2.57e-04] | 0 / 8 | 1.66e-04 | A |
| move LATENT loss × 0.05 | −4.56e-04 [−1.74e-03, +6.50e-06] | 7 / 8 | 6.27e-04 | A (label side) |
| species set loss × 0.05 | +1.31e-04 [−1.76e-03, +1.96e-03] | 2 / 8 | 5.61e-03 | C |
| item loss × 0.05 | +3.81e-05 [−7.58e-04, +5.63e-04] | 2 / 8 | 2.22e-03 | C |
| value 0.5 × MSE | +1.20e-03 [+1.67e-05, +2.66e-03] | 0 / 8 | 3.78e-03 | C |
| entropy −0.05 × H | −3.12e-04 [−3.02e-03, +1.64e-03] | 5 / 8 | 2.45e-03 | C |
| policy (a Monte-Carlo-advantage PROXY) | −8.79e-04 [−3.63e-02, +5.01e-02] | 5 / 8 | 5.79e-02 | C |

The mean route-C gradient is 352 times the label's and at cosine −0.15 with it.

**The latent loss, split** (`results/gradlatent_home72.log`, the same 8 batches;
`results/gradlatent_home72_n1024.log`, 3 batches × 1,024 rows). The move latent loss is
`(1 − cosine(predicted latent, the mean latent of the true moves)) + relu(1 − the standard deviation of the
predicted latent over the batch's slots)`, the predicted latent being `softmax(move logits) @ the move latent
table`. d / d(residual scale) of each part at its coefficient 0.05:

| part | 8 × 128 rows: mean [min, max] | < 0 | 3 × 1,024 rows: mean [min, max] | < 0 |
|---|---|---|---|---|
| the cosine part | +3.83e-06 [−6.78e-06, +1.88e-05] | 3 / 8 | +1.21e-05 [+9.55e-06, +1.37e-05] | 0 / 3 |
| the variance hinge | −4.60e-04 [−1.75e-03, 0] | 6 / 8 | −4.61e-04 [−1.01e-03, −9.05e-05] | 3 / 3 |
| the latent loss (their sum) | −4.56e-04 [−1.74e-03, +6.50e-06] | 7 / 8 | −4.49e-04 [−1.01e-03, −7.68e-05] | 3 / 3 |
| the move label BCE × 0.05, same slots | +2.25e-04 | 0 / 8 | +2.32e-04 [+2.28e-04, +2.34e-04] | 0 / 3 |

- **The push toward a larger residual is the variance hinge.** It rewards predicted latents that differ more from
  slot to slot, whatever the truth is, and a sharper move belief differs more. The predicted latent's standard
  deviation sits at the hinge (mean 1.31 a dimension, 8 % of the dimensions under the target of 1.0 at both batch
  sizes): the hinge is an active constraint at its boundary, not a dead term. It is as large at 1,024 rows as at
  128 and more consistent in sign. Production's micro-batch is 2,048 rows: not measured.
- **Along the scale direction the label side as a whole asks for a LARGER residual:** the latent loss's −4.5e-04
  outweighs the label BCE's +2.3e-04 at their production coefficients (0.05 each).
- **The reveal pin is not the mechanism.** A revealed move's logit is pinned at 10.0 and takes nearly all of the
  latent's softmax (0.87 of the mass with one revealed move, 0.95 with two), so the first hypothesis was that the
  cosine part rewards unshown logits near the pin. Its sign is right on slots with one revealed move (7 of 8
  batches below zero) and its size is 8 times smaller than the label BCE's on the same slots (−9.1e-06 against
  +7.3e-05); slots with no revealed move cancel it (+1.4e-05, 0 of 8 below zero).

## 5. What X68's fine-tune did to this leg

X68-FT's truths-only score is within 0.1 nats of P\*'s off the pool; its proper score is 9 to 12 nats below the
prior where P\*'s is 0.6 to 2.2 below (§1). It raised moves its training
labels never contained: Explosion-class cells such as Dugtrio (species 51) move 253 from a prior of 0.02 to 0.993
and move 222 to 0.978, Gengar (94) move 325 to 0.944, Zapdos (145) move 214 from 0.04 to 0.936, with 0 labels of
moves 253, 325 and 222 in its training and validation shards (`results/xcheck.log`). Non-true candidates above 0.9
a unit: X68-FT 1.89 / 2.04 / 1.96 / 2.23, P\* 0.10 / 0.32 / 0.18 / 0.43, the prior 0.03 / 0.10 / 0.06 / 0.16
(`results/peek_*.log`). Cause, measured on its own objective at P\* (`results/gradroute3.log`, last block): into
`move_head.weight` the species loss sends 1.131 and the item loss 0.1375 against the move label's 4.63e-03 (244×
and 30×), through the live publication; under the built mode both read NO PATH. So X68's move-leg result is a
read of route C, and says nothing about training this head on labels.

## 6. The reading

1. **Memorisation of pool movesets: partly, and not the main term.** The belief does know which set goes with
   which pool team (§2), but an ideal memoriser of the pool would be at or above the prior off the pool on the
   proper score (§3). The breadth part is real and small: ≈ 0.15 to 0.27 nats truths-only.
2. **The conditioning on species: yes, this is the carrier.** 83 % of the residual is a fixed species × move table.
3. **Something else: the table is over-confident, in the pool too, and the label loss does not ask for it.** The
   label BCE's descent shrinks the residual on 8 of 8 batches (§4). Two measured candidates hold it up:
   **route C**, whose gradient into the head is two orders larger than the label's, unrelated to it in direction,
   and proven to wreck the leg when it is the only strong signal (§5); and **the move latent loss's variance hinge**,
   the one term whose descent consistently GROWS the residual, at twice the label BCE's rate along the scale
   direction (§4). Which of the two holds production's over-confidence is NOT shown here: the policy term is a
   proxy whose sign is noise over 8 batches, and one direction (the scale) is not the whole gradient.

## FINDINGs

- The meter's verdict column `moves_logp` scores only the true moves, so it is blind to a belief that raises
  moves that are not there. X68-FT is the proof: within 0.1 nats of P\* on that column off the pool, 9 to 12 nats
  below the prior on the proper score. The tool gains the proper score as the column `moves_ll` beside this record
  (`designs/tools/offpool_belief.md`); the scripts here stay the frozen record of how these numbers were made.
- X68 chose its epoch on the summed validation loss, and its move head was trained by the species and item losses
  through the publication; its "move leg not repaired" line is not evidence about label training.
- `MoveBelief.publish_detach` alone cuts no gradient under `--move-set-closure on`.
- The move latent loss's variance hinge rewards a sharper move belief whatever the truth is: along the residual's
  scale it outweighs the label BCE two to one at production's coefficients, at 128 and at 1,024 rows a batch. It
  is not cut by `move_label_only` (`--move-belief-latent-coef 0` removes the loss), and is a named confound of any
  read of that lever. The first hypothesis, that the reveal pin (10.0) in the latent's softmax did this, is
  refuted in size (§4).
- Not measured: the true PPO policy gradient (the bank holds greedy rows and no advantage); the routes of the
  spread, Hidden Power type and latent-predictive losses; off-pool rows in §4 (the bank's off-pool panels were not
  run through the gradient scripts); `rb_x60t_s3501` was dumped and not analysed.
- The design's preference for a floor over a bounded residual (its §7) was written for the species leg; on the
  move leg the bounded residual is the better post-hoc transform on every panel (§3).

## Files

| file | what |
|---|---|
| `scripts/dump.py`, `pool_table.py` | the per-unit logits of one checkpoint over the banks; the pool's per-species tables |
| `scripts/analyze.py`, `analyze2.py`, `analyze3.py` | §1–§2; §3's predictors and structure; §3's transforms and calibration slope |
| `scripts/gradroute2.py`, `gradscale.py`, `gradlatent.py` | §4 |
| `scripts/peek.py`, `xcheck.py`, `wdiff.py` | §5 |
| `scripts/compile_check.py` | the lever's compiled CPU forward + backward against eager (`results/compile_check.log`: both modes OK, worst relative gradient error 1.18e-05) |
| `results/analyze*.json` | every number of §1–§3 with its interval, aggregates only |
| `results/*.log` | the scripts' printed output, scheduler lines removed |

The banks and the per-unit dumps state held-out teams in the clear and stay outside the checkout:
`~/gen3ai_archive/offpool_belief_2026-11-09/bank/`, `~/gen3ai_archive/move_leg_diag_2026-11-10/`. This folder is a
frozen record, not a tool.
