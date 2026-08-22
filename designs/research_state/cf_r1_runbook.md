# RUNBOOK — R1, the counterfactual value-grounding experiment

> **SIGNED OFF — owner, 2026-08-22.** Written after the experiment-readiness batch so the arm's
> rules exist before any of its numbers do; signed off the same day. The pre-registration below is
> now BINDING: edits after this line require new evidence, stated beside the edit. Launch now waits
> only on a training slot.
>
> **PREREQUISITE DISCHARGED — the label PRODUCER DRIVER exists** (`agents/training/cf_producer.py`,
> 2026-08-22). *This edit changes no rule: it records that the one build item the sign-off named as
> outstanding has landed. The pre-registration below is untouched.* Run it as a detached sidecar
> beside the arm:
>
> ```bash
> export PYTHONPATH=$PYTHONPATH:src && nohup nice -n 10 python -m agents.training.cf_producer \
>     models/<arm> --rollouts 8 --top-n 3 --max-labels-per-hour 2000 --impl rust \
>     > models/<arm>/cf_producer.log 2>&1 &
> ```
>
> It watches `<run>/cf_records/`, reloads the freshest `checkpoints/` snapshot each cycle (stamping
> its step on every label), ranks decisions by the declared, versioned
> `cf_producer_priority_v1` sampler (`1.00·critic_surprise + 0.35·policy_entropy`), rolls the top
> `--top-n` out `--rollouts` times, and writes `<run>/cf_labels/labels_cf_producer_<step>_<seq>.jsonl`.
> `tail -f` the log for its per-cycle heartbeat; `<run>/cf_producer_state.json` is the readable
> state. A failed anchor exits **3** and produces nothing further.
>
> ⚠️ **THE ECOLOGY NOTE, and it is a caveat on this arm's labels, not a detail.** A training record
> carries **no opponent identity** — the tap's `__RECON__` holds the seed, both teams and the
> committed choices, and nothing that names the policy on the other side. So v1 rolls out with the
> **CURRENT snapshot playing BOTH sides, stochastic at temp 1.0**: a documented approximation
> matching the ~90% self-play share of the training mixture, wrong in a known direction for the
> rest (a bot-opponent episode gets a stronger, self-like opponent, so that label is biased LOW).
> Every row says `opponent: "self_current"` — never a bot name it cannot verify. **R1's labels are
> therefore measured against the self-play ecology**, which is the population §2's primary meter
> must be read against too. Closing the approximation means threading the opponent's identity
> through the training-side tap.

Gate **G4** of [`designs/ai_v10/design_counterfactual_value_grounding.md`](../ai_v10/design_counterfactual_value_grounding.md).
R1 only, alone: tight Monte-Carlo P(win) labels on **visited** states, delivered to the **win-prob
head**. R2 (counterfactual successors) and R3 (k-step grounded targets) are not in this arm and do
not get a preliminary read from it.

## 0. What the arm is

A fork of the current base at a snapshot boundary, with the label loop live:

```
--cf-records                       # the training-decision tap (ring, default keep 512)
--cf-winprob-coef <from the ladder>
--cf-label-likelihood binomial     # the default; the labels carry n_rollouts, so use them
--cf-head-only                     # the DEFAULT and the first stage — see §0a
--cf-label-lag-steps 150000        # default; one production PPO iteration
```

plus the background producer (`python -m agents.training.cf_producer models/<arm>`, launch line in
the header) writing `<run>/cf_labels/labels_cf_producer_<step>_<seq>.jsonl`.

Control arm: the same fork, same seed policy, **without** `--cf-winprob-coef` (the tap may stay on
in both — it is byte-identical to the update and the records are useful either way).

### 0a. Head-only FIRST, and what would license opening the trunk

`--cf-head-only` (default) stop-grads the term's input, so it trains the win-prob head's own
parameters and provably cannot perturb the trunk (`train/cf_grad_share` reads **exactly 0.0** — the
verification, not a defect). That is the whole first stage: the head is MC-native, so R1 needs no
critic ROUTE change and owes no C4 gate.

Opening the trunk (`--no-cf-head-only`) is a **separate, later decision** requiring a head-only
result that moved the primary meter. It is not a tuning knob to reach for if the first arm reads
flat — a flat head-only arm is evidence about the labels or the meter, not an argument for more
trunk exposure.

## 1. The coefficient ladder

**Three points, log-spaced, selected by a MEASURED gradient share, not by loss magnitude.**

- Target band: **`train/cf_grad_share` 5–15%** of the grad-balance denominator, read on the
  trunk-open probe (head-only reads 0.0 by construction, so the share must be calibrated in a short
  `--no-cf-head-only` diagnostic run, or read from the term's own gradient norm — declare which
  before launching).
- Precedent for the band: the **win-prob 0.10 collateral** result — the on-policy win-prob BCE at
  coefficient 0.10 sits in this range and has never been convicted of collateral damage to the
  policy. Above it we are trading the main objective for an aux with no such record; below it the
  arm risks the "shrinking aux loss, nothing else moves" null for reasons of dosage rather than
  mechanism.
- Ladder: `c/3, c, 3c` around the coefficient that lands mid-band. Pick ONE for the full arm before
  seeing any meter; the other two are for a short calibration pass only.

## 2. PRIMARY meter — `sd_true_excess` on HELD-OUT states

**The G0 bias map, re-run — same strata, same sampler version, states the producer never labelled.**

```
python -m agents.training.cf_audit models/<arm> --rollouts 8 --states 400
```

- The headline is `population_weighted_sd_true_excess` and the per-decile `sd_true_excess` column,
  **not** the mean gap. G0's amendment is the whole reason: population-mean predicted−MC gaps are
  |0.05|–|0.07| with a sign that flips with the population you weight to, while the true
  within-decile spread is 0.11–0.36. **A re-centred head would score a success on the wrong meter.**
- **HELD OUT is load-bearing.** The producer's own labelled states are training data for this term;
  measuring the meter on them measures memorisation. The audit must draw from a trace step (or a
  battle set) the producer did not consume, and the runbook records which. *(As shipped the two are
  structurally disjoint — `cf_producer` reads `<run>/cf_records/` and `cf_audit` reads
  `<run>/eval_traces/` — so the held-out property holds by construction rather than by discipline.
  Record the audited `step_N` anyway; "by construction" is a claim about today's code.)*
- Comparison is arm vs control at matched step and matched sampler version, with the audit's
  battle-clustered CIs. Never quote a mid-run number as a result.

## 3. SECONDARY meters

| meter | source | reads |
|---|---|---|
| calibration strata | `cf_audit` `by_decile_outcome`, `conviction_class` | the +0.23 conviction gap and the loss−win difference (+0.307 [+0.227, +0.392] at G0) |
| critic-surprise rate on FRESH losses | `prober.query scan` / the sentinel sweep | the 0.827 class's live descendant — the behavioural readout |
| `width_vs_blur_spearman` | `cf_audit` `evidential` block | **only if the arm carries `--cf-evidential`** — does the confessed Beta width track the measured blur, per stratum, with a battle-clustered CI |

The evidential head is **optional in this arm** and orthogonal to R1's claim: it cannot remove the
blur (it reads the same `value_pooled`), only confess it. If it is on, its coefficient rides the
same declare-before-launch rule and its meter is the correlation, never the falling `nll`.

## 4. GUARDS

- **Belief-bank canaries** — the per-edge-family liveness metrics must not degrade. A cf term that
  quietly starves the belief heads is a cost the primary meter cannot see.
- **ELO non-inferiority vs the control arm**, on the dense `snapshot_ladder/ladder.json` tail,
  matched snapshot COUNT, at run END, SE from the paired refit. Standard bar: Δ ≥ −15.0 with
  CI95-low > −40.0. R1 is a *bias* lever; it is allowed to be ELO-neutral, it is not allowed to
  cost strength.

## 5. KILLS — pre-registered, and the standing rule applies

- **The learns≠helps rule.** `train/cf_loss` falling while the primary meter is flat is a **NULL**,
  not a partial result. This tree's aux-loss body count (BYOL decodable-but-null; SimSiam deleted at
  13% of the step; the top-K structural null) is why every gate here is meter-based.
- **Flat meters at 2× sample closes the rung.** If the primary meter has not moved at twice the
  labels, R1 is done and the finding is that MC re-measurement of visited states does not buy
  resolution — which is itself a real result about where the critic's blur comes from.
- **A wide-everywhere evidential width is the same null as a wide-nowhere one** (`spearman` returns
  `None`, not 0, exactly so these cannot be confused).

## 6. LAUNCH-WINDOW checks — read these in the first hour, before trusting anything

The label path is a two-process contract with no IPC, so its failure modes are all silent by
default. Each of these has a scalar, and each scalar has one reading that means "stop":

| scalar | healthy | what a bad reading means |
|---|---|---|
| `cf/labels_ingested_total` | rising | **FLAT = the producer is dead.** Distinct from lag. |
| `cf/buffer_fill` | at/near capacity | 0 with a rising ingest = expiry is eating everything |
| `cf/rows_sampled` | > 0 every `train()` | 0 with a nonzero fill = the fold is not running (check the head / the coefficient) |
| `cf/labels_expired_total` | ~flat after warmup | rising fast = the producer is lagging more than `--cf-label-lag-steps` |
| **`cf/labels_future_total`** | **0** | **non-zero = labels from a NEWER snapshot than this process — a crash-restart rollback.** The buffer expires them and warns once by name; clear `<run>/cf_labels/` or restart the producer at this step |
| **`cf/labels_replaced_total`** | small | large = the producer is re-labelling ground it already covered; the effective sample is smaller than the file count suggests |
| `cf/labels_skipped_total` | 0 | non-zero = GIGO (schema, digest, obs width) — read `skip_reasons` |
| `cf/label_age_steps_p50` | 0 < age < lag bound | **negative is impossible at steady state now**; if it appears, the symmetric-expiry guard has regressed |
| `train/cf_grad_share` | 0.0 under head-only | anything else under head-only = the stop-grad is gone |

## 7. What this arm does NOT get to conclude

Not that the critic's *delivery* is fixed (C6's line stays closed — this attacks targets). Not
anything about R2's bait/loop cure claim, which needs an R2 arm and names the loop rate as its own
readout. Not a V-head result: G0 mapped the **win-prob head**, and the main critic V comparison
carries a PBRS caveat that a separate read owes.
