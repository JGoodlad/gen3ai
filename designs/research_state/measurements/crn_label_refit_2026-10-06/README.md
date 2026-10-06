# CRN label refit (2026-10-06): is sibling discrimination limited by the value head's TRAINING LABEL?

**Status: PRE-REGISTERED.** This file was committed before any outcome of this measurement was
read. Amendments go BELOW the line at the end, dated, each one saying whether any outcome had been
seen when it was written. The result is `result.md`.

## 0. The question

The win-prob value head ranks two moves from the same position (siblings, shared dice) at about
0.57-0.59 ([UNDERSTANDING §4.x](../../UNDERSTANDING.md): 0.587 [0.553, 0.623], and twelve heads
on 0.567-0.578). An 8-rollout leaf reads 0.652 on the same column (`rollout_leaf_kcurve_2026-09-19/`,
knee K = 4). The 2026-09-19 controls showed that a single-rollout label mis-orders the top1|top2 pair
39.7 % of the time, which caps the metric near 0.61 when single outcomes are the yardstick. The
critic is also blind to the near-best moves the policy starves (`x4_preread/READOUT.md`).

**Asked here:** if the value READOUT is refit, trunk frozen, on a less noisy label (the mean of K
shared-dice rollouts to the end), does its sibling accuracy climb toward the rollout ceiling? If it
does, the LABEL was limiting. If it stays flat, the representation (what the frozen trunk encodes)
or the coverage (which states it was trained on) is limiting.

Why it matters: search re-ranks siblings with this head (it is the leaf), and starvation is the
policy pushing near-equally-good moves below 1 % (the owner's "near-equally-good moves weighted
near the bottom").

## 1. What is fixed

| item | value |
|---|---|
| checkpoint | `models/rb_x5ab_blob_s1001/final_model.zip` (the X5 blob arm, 15.05M steps, pinned `e5e660dd`), READ-ONLY, loaded at this worktree's HEAD through `main.policy_spectrum.reader.load_checkpoint` (strict `load_foreign_opponent` path; `check_winprob` passes: critic `winprob`, no extra obs keys) |
| positions | the Lane S ground-truth subset `m5_laneS/truth_v2/gt_subset_v2.json` (1,600 turns of the committed bank `m5_laneS/bank_v1/`, 549 battles), or a battle-seeded subsample of it (§5) |
| dice | `main.policy_spectrum.truth.turn_seeds(id, S)`: nested, so the first k seeds of S are `turn_seeds(id, k)` |
| continuation | the blob checkpoint's GREEDY policy on BOTH sides to the end (`truth run`, CPU), the existing ground-truth construction; value +1 win / −1 loss (stall forfeit included) / 0 tie or truncated, from the banked side's view |
| split | by BATTLE: `sha256("crn_label_refit_2026-10-06:" + battle)`; first hex digit even = HELD-OUT, odd = TRAIN. Inside TRAIN, a second hash (`"…:val:" + battle`, first hex digit `0`-`2`, ~3/16) is the early-stopping VALIDATION part. Never by row |

### Seeds per role (disjoint where it matters)

| seeds | TRAIN turns | HELD-OUT turns |
|---|---|---|
| 0-7 | label seeds (K = 1, 4, 8 = the first K) and the one-ply successor feature seeds | the rollout-leaf scorers (K = 1, 4, 8 = the first K) and the successor feature seeds |
| 8-31 | not played | the TRUTH (24 seeds): the yardstick every scorer is scored against |

No scorer and no fit ever sees seeds 8-31 of a held-out turn.

## 2. The scorers (all one-ply successor readouts, except the rollout leaf)

For an action `a` of a turn, the one-ply successors `s′_j(a)`, j = 0..7, are captured with
`qhat.one_ply` under seed j (the opponent's open root decision, and any opponent-only decision
before our next one, answered by the checkpoint's greedy choice: truth's construction, so `s′_j(a)`
is the state truth playout j continued from). A branch that ends first scores its terminal value.
Every head scorer is

    Q_h(a) = mean_j [ terminal_j(a) if branch j ended, else 2·σ(h(φ(s′_j(a)))) − 1 ]

with `φ` = the frozen trunk's `value_pooled` (the 128-wide whole-board pool the win-prob head
reads; `stash.value_pooled`) and `h` a `WinProbHead` (LayerNorm → Linear → ReLU → Linear → 1).

| scorer | h | label it was fit on |
|---|---|---|
| **BASE** (the head as trained) | the checkpoint's own `win_head` | 15M steps of training-time episode outcomes |
| **REFIT-K1 / K4 / K8** | a copy of `win_head`, WARM-started, trunk frozen | mean of the first K label seeds' outcomes, mapped to [0, 1] |
| **LEAF-K1 / K4 / K8** | none: the mean outcome of the first K rollouts (seeds 0..K−1) | n/a (the ceiling) |

**The refit.** Fit on TRAIN turns' actions (all legal actions of every train turn, not only
contested ones). Loss = BCE(`(Q_h(a)+1)/2`, `(ȳ_K(a)+1)/2`), the action-level score against the
action-level label, so the fit is exactly the quantity that is scored. Adam, lr 1e-3, weight decay
1e-4, minibatch 256 actions, at most 300 epochs, early stopping on the VALIDATION battles' loss
against their OWN K-label (patience 20, best epoch restored). Fit seed 0 is the registered fit;
fit seeds 1 and 2 are reported as the fit-noise spread only. Identical features, optimiser and
stopping rule for every K: the K-label is the only thing that changes.

## 3. The pairs and the meters

**Contested turns** (the 2026-09-14 definition, applied to this checkpoint): a `free` turn (not a
forced switch), turn ∈ [2, 40], ≥ 3 legal actions, and the blob policy's top-2 masked-logit gap
below the 40th percentile of that gap over all eligible subset turns.

**Pairs.** On HELD-OUT contested turns, every unordered pair of legal actions (a, b). TRUTH
Δ = mean over seeds 8-31 of `o_a − o_b`. Δ is a multiple of 1/24, so an exactly-zero Δ is a TIE
and is EXCLUDED (rule 8: no pair sits within a rounding error of the sign boundary); every other
pair is at least 1/24 from it.

**Pairwise accuracy** of a scorer = the share of non-tied pairs where sign(score_a − score_b) =
sign(Δ). A scorer tie (|score_a − score_b| < 1e-6, a deterministic threshold so a float
reassociation cannot flip it) counts ½. Pooled over pairs.

**CIs**: battle-clustered bootstrap (resample held-out BATTLES with replacement, 2,000 draws,
`numpy.random.default_rng(20261006)`), percentile 95 %; every contrast is PAIRED (the same draws).

| meter | pairs |
|---|---|
| **M1, headline** | all non-tied sibling pairs on held-out contested turns |
| M2 | the leaf column: (policy top-1, policy top-2) on held-out contested turns |
| **M3, the starved slice** | on ALL held-out free turns: pairs with at least one STARVED near-best action. Lane S's definitions on the truth seeds: a* = the action with the best truth mean; gap(a) = paired mean of `o_{a*} − o_a`; NEAR-BEST = gap ≤ 0.1; STARVED = near-best and π_blob(a) < 1 % (π = the blob policy's masked softmax at the banked root row) |
| M3a / M3b | M3 split: the partner is a* itself / the partner is DOMINATED (gap − 1.96·SE > 0.1) |

Also reported, descriptive: per-scorer accuracy split by pair category; ECE against the truth mean;
the fit's stopping epoch.

## 4. Prediction and decision rule

**The decision rule (registered):** with Δ8 = REFIT-K8 − BASE on M1 (paired, 95 % CI) and
CLOSED = Δ8 / (LEAF-K8 − BASE) (the share of the rollout gap closed, point estimate):

| outcome | condition | reading |
|---|---|---|
| **LABEL-LIMITED** | Δ8's lower bound > +0.0005 AND CLOSED ≥ 0.5 | the label was the binding constraint; a better value label buys most of the rollout leaf |
| **PARTIAL** | Δ8's lower bound > +0.0005 AND CLOSED < 0.5 | the label binds somewhat; the representation or coverage binds the rest |
| **FLAT** | Δ8's 95 % CI lies inside ±0.02 | representation or coverage limits it, not the label |
| **NOT DETECTED, NOT SHOWN FLAT** | neither of the above (the CI straddles 0 and is wider than ±0.02, or a lower bound within ±0.0005 of 0) | the instrument cannot tell at this n; nothing is concluded |
| **NO ROOM (read void)** | LEAF-K8 − BASE on M1 has its lower bound ≤ +0.0005 | the rollout ceiling is not reproduced on this checkpoint, so there is nothing for a label to close |

A bound within ±0.0005 of its boundary is a BOUNDARY result and counts as not clear (rule 8).

**Secondary contrasts (registered, they do not override the verdict):**
- **REFIT-K8 − REFIT-K1** isolates label NOISE (the count): both fit on labels with the same
  expectation, so only the label variance differs.
- **REFIT-K1 − BASE** isolates the label's ESTIMAND plus the refit: the refit's labels are
  greedy-continuation outcomes (the truth's own estimand), whereas the trained head learned
  training-time outcomes (stochastic self-play against the pool), so this is a target shift, not a
  noise reduction.
- The same three contrasts on M2 and on M3.

**Prediction (written before measurement):**
- LEAF-K8 − BASE on M1: +0.04 to +0.08, detected (the 2026-09-19 K-curve reproduced).
- REFIT-K8 − BASE on M1: **FLAT or NOT DETECTED**, point estimate within ±0.02. Reason: the
  K-label has the SAME expectation as a single outcome, so it only lowers label variance; the trained
  head's 15M steps of single outcomes already averaged that variance out, and the 2026-09-18 offline
  fit found a head fitted from scratch in 1.1 s matching the 75M head. My account is that the bound
  is in what `value_pooled` encodes about the one-ply difference (bias, not variance:
  UNDERSTANDING §3.5).
- REFIT-K8 − REFIT-K1: small and positive (≤ +0.02), because 5-6k action labels is a small set
  for a 16.6k-parameter head.
- REFIT-K1 − BASE: the estimand shift could be worth a little (≤ +0.02); unknown sign.
- M3 (starved): BASE near chance on pairs against a* (the X4 pre-read); the refit does not
  change that.

## 5. Size, cost and engineering

- Two `truth run` jobs (`python -m main.policy_spectrum.truth run`, CPU, greedy continuation):
  TRAIN turns at S = 8, HELD-OUT turns at S = 32. Rows are append-only, one fsync'd JSONL row per
  turn, resumable (a re-run skips every turn already on disk at the same S and continuation).
- One one-ply capture job (`capture.py`): seeds 0-7 per action, the frozen `value_pooled` and
  BASE's V per captured successor, one durable `.npz` per chunk of turns and a JSONL index.
- Detached (`setsid nohup`), under `scripts/ops/mem_cap.sh`, at nice 15, with at least 4 cores
  left free. No GPU (no lease; `--device cpu`; `GEN3AI_TEST_ALLOW_GPU` never set).
- **Size rule:** a pilot (§ Amendment 1) measures the CPU cost per playout on this box. If the
  whole 1,600-turn subset costs more than ~10 CPU-hours, the job runs a SUBSAMPLE of battles,
  chosen by the hash above (the lowest `sha256("crn_label_refit_2026-10-06:size:" + battle)` first)
  until the budget is met. The pilot reads only throughput, never an outcome.

## 6. Files

| file | role |
|---|---|
| `README.md` | this pre-registration (+ amendments) |
| `make_subsets.py` | the split, the subsample and the two subset files |
| `capture.py` | the one-ply successor capture (features + BASE's V) |
| `refit.py` | the fits and the scorers' per-action values |
| `score.py` | the meters, the bootstrap, the verdict |
| `run_all.sh` | the detached driver |
| `rows/` | `truth_train_S8.jsonl`, `truth_held_S32.jsonl` (compact gz when committed), the capture index |
| `result.md`, `result.json` | the read |

---

## Amendments
