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

> ## 🔶 AMENDED — TWIN HEADS + SHADOW CRITIC (2026-08-22)
>
> **AUTHORIZED OWNER DESIGN CHANGE to the signed pre-registration.** Ledger 2026-08-22 evening,
> *"Three owner sign-offs"*, item 3: *"The TWIN-HEADS amendment to the R1 runbook is AUTHORIZED
> (owner design change to the signed pre-registration): the primary comparison becomes WITHIN-RUN
> paired head differences — three win-prob heads (control BCE-only / same-states single-outcome /
> same-states tight-MC, isolating prioritization from variance reduction) — plus the passive SHADOW
> CRITIC (a value twin on mc_return labels, never computing an advantage) as the staged promotion
> path for critic surgery. Cross-run forks retained only for the later trunk/policy-transfer stage.
> Build dispatched."* Built and landed the same day (`gen3_cf_twin_heads_v1`, MODEL_CONFIG_VERSION
> 99). **§0 and §2 below are amended accordingly; every other rule stands unchanged.**
>
> This is a change to what the arm MEASURES, not to what it claims. The kills (§5), the guards
> (§4), the coefficient discipline (§1), the launch-window table (§6) and the boundary on what the
> arm may conclude (§7) are all untouched.

Gate **G4** of [`designs/ai_v10/design_counterfactual_value_grounding.md`](../ai_v10/design_counterfactual_value_grounding.md).
R1 only, alone: tight Monte-Carlo P(win) labels on **visited** states, delivered to the **win-prob
head**. R2 (counterfactual successors) and R3 (k-step grounded targets) are not in this arm and do
not get a preliminary read from it.

## 0. What the arm is  *(AMENDED — the control is now a HEAD, not a run)*

**ONE run, three win-prob heads.** The arm no longer needs a paired fork to answer its primary
question, because the control arm lives inside it:

```
--cf-records                       # the training-decision tap (ring, default keep 512)
--win-prob-mode read_only          # REQUIRED — head A must exist for the twins to mirror it
--cf-twin-heads                    # BUILD heads B and C (structural, v99, version-gated)
--cf-twin-coef <from the ladder>   # ONE coefficient for both twins
--cf-label-likelihood binomial     # the default; the labels carry n_rollouts, so use them
--cf-head-only                     # the DEFAULT and the first stage — see §0a
--cf-label-lag-steps 150000        # default; one production PPO iteration
--cf-shadow-critic --cf-shadow-coef <declared>   # the passive value twin; optional, orthogonal
```

plus the background producer (`python -m agents.training.cf_producer models/<arm>`, launch line in
the header) writing `<run>/cf_labels/labels_cf_producer_<step>_<seq>.jsonl`. The producer now
additionally ships `outcome_label` (free — it already computes the recorded outcome) and, with
`--mc-return` (**default ON**), `mc_return` + a reward digest.

| head | trained by | isolates |
|---|---|---|
| **A** = `win_head` (the EXISTING head, untouched) | the on-policy single-outcome BCE only | the control |
| **B** = `cf_twin_head_b` | A's loss **+** the cf states with **SINGLE-OUTCOME** labels (n≡1) | **B−A = coverage / prioritization** |
| **C** = `cf_twin_head_c` | A's loss **+** the same states with **TIGHT-MC** labels (n=R) | **C−B = pure variance reduction** |

`C−A` is the original R1 claim; the amendment's value is that it now DECOMPOSES. `--cf-winprob-coef`
is **not** part of the amended arm — head A must stay a clean control, and a live scalar cf
coefficient would fold the tight-MC labels into it. Leave it at 0.

**Why:** two runs differ in every random draw they ever make, and this meter carries a MEASURED
floor of ~39% of its own variance (§2's amendment). Three heads on one trunk hold the trunk, the
states, the seeds and the floor **identical by construction** rather than matched by design.

### 0b. THE SPILLOVER BOUNDARY — state this beside any result

Head-only is not a stage here, it is the twins' definition in v1: B and C read a **detached**
`value_pooled` in every term they take, including the on-policy mirror
(`train/cf_twin_grad_share` reads exactly 0.0). So:

- **What the paired difference measures: the LABEL effect on a FIXED representation.** Given this
  trunk, which label stream produces a better-calibrated win-prob head.
- **What it does NOT measure, and cannot:** (a) whether tight-MC labels would build a BETTER TRUNK
  if allowed to shape it — the twins are frozen with respect to it, and A's own `shaping` mode (if
  used) shapes it for all three equally; (b) whether any of it changes the POLICY — none of these
  heads feeds pi or vf, so the acting path is bit-identical across the three arms by construction.
- **Both remain CROSS-RUN questions**, and the fork machinery is retained for exactly them (the
  owner sign-off: *"Cross-run forks retained only for the later trunk/policy-transfer stage"*). A
  positive within-run result is what would license spending a fork on the trunk stage; it is not a
  substitute for it.

There is one coupling head-only does not remove, and it is honest to name: the **global gradient
clip** rescales every gradient by a factor computed over all parameters, so any live aux perturbs
the policy/value update in the last bits. It is pinned as a distinct mechanism (a test shows the
updates are bit-identical with the clip raised out of the way) and it is shared by every aux this
tree runs, but it is not zero.

### 0c. The SHADOW CRITIC — orthogonal, and not part of the primary

A passive value twin on `mc_return` labels (the mean realized **shaped return**, in V's own units).
It **never computes an advantage and never enters GAE**. Its meter is
`cf/shadow_shadow_vs_live_v` — the signed real-unit gap to the live critic on the same states — and
its purpose is the **staged promotion path** for critic surgery, which owes the C4 offline gate.
Nothing about R1's verdict depends on it, and it may not be cited as evidence for a route change on
its own; what it produces is the evidence a C4 submission would be built from.

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

## 2. PRIMARY meter — the PAIRED HEAD DIFFERENCE on HELD-OUT states  *(AMENDED)*

> **AMENDED 2026-08-22 by the authorized twin-heads change** (ledger, "Three owner sign-offs" item
> 3). The primary is now a **within-run paired proper-score difference across heads A/B/C on the
> same held-out rows**. `sd_true_excess` is retained below as the SECONDARY continuity link to G0
> and as the meter head A is still read on; the floor amendment that follows it still governs any
> absolute reading of it.

```
python -m agents.training.cf_audit models/<arm> --rollouts 8 --states 400
```

The audit forwards **every head the checkpoint carries** over the labelled states and emits the
`twin_paired` block. Per row it computes `brier_X = (pred_X − mc)²` and `abs_err_X = |pred_X − mc|`
for X ∈ {A, B, C}, then differences them **across heads on the same row**, with a battle-clustered
bootstrap CI on the difference.

| contrast | isolates | reads |
|---|---|---|
| **`B_minus_A`** | COVERAGE / prioritization — the same loss form on extra states | did labelling these states at all help? |
| **`C_minus_B`** | PRECISION — the same states, a tight-MC target instead of one draw | did the *tightness* help? **This is the amendment's whole point.** |
| `C_minus_A` | the TOTAL effect | the original R1 claim, now decomposed |

- 🔒 **SIGN: these are ERROR scores. A NEGATIVE difference means the first-named head is BETTER.**
  Read it backwards and the arm reports its own opposite.
- **The hidden-information floor cancels EXACTLY, not in expectation.** It is a property of the
  STATE, and every head scores the same state — so it is the same additive constant in both terms of
  the difference. The §2 amendment below argued the floor cancels in an arm-vs-control difference at
  matched STEP; twins strengthen that to matched **STATE**. Effect sizes therefore no longer need
  floor subtraction, and the "flat near the floor is ambiguous" caveat does not apply to this meter.
- **No stratification means no selection correction is owed.** The eval capture quota over-samples
  losses, which is why the bias map recombines cells at population shares; a paired difference over
  identical rows carries that bias identically in both terms.
- ⚠️ **READ `mean_abs_pred_diff` BEFORE reading a null.** A near-zero contrast with a near-zero
  prediction divergence means the label streams **did not separate the heads** — a coverage or
  dosage reading, not the §5 kill. The kill applies to a divergence that HAPPENED and bought
  nothing. The live counterpart is `cf/twin_b_vs_c_abs`.
- ⚠️ **READ `cf/twin_b_coverage` FIRST OF ALL.** A producer shipping no `outcome_label` trains head
  B on nothing; B then equals A and `C_minus_B` silently becomes `C_minus_A` while every other
  counter reads healthy. It is the one way this arm produces a confident wrong answer.
- **HELD OUT is still load-bearing, unchanged**: `cf_producer` reads `<run>/cf_records/` and
  `cf_audit` reads `<run>/eval_traces/`, so the two are structurally disjoint. Record the audited
  `step_N` anyway.
- **The budget-matched-control question is now PARTIALLY ANSWERED by head B.** B eats the same
  number of extra gradient folds on the same extra states as C, differing only in the label's
  precision — so `C_minus_B` is budget-matched by construction. What B does *not* control for is the
  cost of PRODUCING the tight-MC labels (R rollouts vs one recorded outcome); that remains a
  compute-budget question about the factory, not about the head.
- **A cross-run control fork is no longer required for the primary**, and is retained only for the
  trunk/policy-transfer stage (§0b). If one is run anyway, its comparison rules are the ones below.

### 2a. SECONDARY (retained) — `sd_true_excess` on HELD-OUT states

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
- **The audit's `twin_resolution` block is NOT this number.** It reports each head's own
  `sd_true_excess` binned by its own prediction, and its cells are **UNWEIGHTED** — the population
  re-weighting is unavailable for B and C, because the eval frame carries only head A's predictions
  and their decile membership over the whole frame is unknown. The block says so in its own
  `weighting` field. Use it for SHAPE (did the blur move where the arm predicted?), never for an
  absolute level compared against `population_weighted_sd_true_excess`.

> **§2 AMENDMENT (2026-08-22, evidence: the hidden-information floor probe,
> `tmp/hidden_info_floor_report.md`, ledger same date) — `sd_true_excess` has a MEASURED FLOOR and
> must be read against it.** The meter is computed from OMNISCIENT rollouts, so it sums LEARNABLE
> blur and the irreducible variance of the opponent's hidden half. Measured on gen-17 @24M over
> 123 decisions / 70 bot battles / 10,040 rollouts by alternative pool-consistent determinizations
> of the opponent's never-revealed slots: the floor is **sd 0.151 [0.119, 0.186] in predicted
> deciles 7–9**, i.e. **39% [24%, 87%] of the meter's variance**, and **~⅓** in the wp≥0.75
> conviction region. Three consequences are binding. (a) The meter has a **non-zero asymptote**:
> `sd_true_excess` cannot fall below ~0.15 in deciles 7–9 however good the head gets, so no target
> may be expressed as a fraction of its current value. (b) Effect sizes are quoted on the **excess
> over the floor** — a 20% reduction of the learnable variance reads as ~12% on the raw meter, and
> a raw-meter comparison silently understates the arm. (c) A flat meter is **ambiguous near the
> floor**, so the §5 "flat at 2× sample closes the rung" kill is evaluated against floor-subtracted
> values. The floor is a property of the POPULATION, not of the arm, so it cancels in an
> arm-vs-control *variance* difference at matched step and matched sampler version — **that
> difference is the primary comparison**. The floor is measured under uniform pool-consistent
> opponents (verified the exact posterior for this run's battles, 228/228 recorded opponent teams
> are pool members) and is an **upper bound** on what an optimal behaviour-conditioned head would
> face. Structure worth exploiting: the floor is CONCENTRATED (49% of states carry ~none; the top
> 10% carry half), flat in hidden-slot count and in game turn — a state-level property ("does the
> unknown decide this position"), not a fog.

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
  *(AMENDED: evaluated on the §2 paired `C_minus_B`, and ONLY once `mean_abs_pred_diff` shows the
  heads actually diverged. A flat contrast between two heads that never separated is a dosage or
  coverage reading and closes nothing. The floor-subtraction clause below no longer applies to the
  paired meter — the floor cancels exactly there — and still applies to the §2a secondary.)*
- **A DECOMPOSITION is a result even when the total is null.** `C_minus_A` ≈ 0 with a clearly
  negative `C_minus_B` and an offsetting positive `B_minus_A` is not a null: it says the
  prioritization is COSTING what the precision buys, which is an actionable finding about the
  sampler and not about the labels. The amendment exists to make that distinguishable.
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
| **`cf/twin_b_coverage`** | **~1.0** | **the FIRST reading. 0 = the producer ships no `outcome_label`, so head B trains on nothing, B==A, and `C_minus_B` silently becomes `C_minus_A` with no other tell** |
| `cf/outcome_label_coverage` | ~1.0 | the buffer-side half of the same fact (residency rather than throughput) |
| `cf/twin_b_vs_c_abs` | rising off 0 | pinned near 0 = the label streams have not separated the heads; a flat primary would then be a DOSAGE reading, not the §5 kill |
| `train/cf_twin_grad_share` | **0.0** | anything else = the twins' unconditional detach is gone, and the "identical trunk" claim with it |
| `cf/shadow_coverage` | ~1.0 *(shadow arm only)* | 0 = no `mc_return` arrived; the shadow head is then randomly initialised and its divergence is pure noise |
| `cf/labels_mc_return_rejected_total` | **0** | non-zero = the producer's reward digest disagrees with this run's — labels measured under a DIFFERENT reward composition. Fix the producer's run_dir/metadata, do not average them in |
| `train/cf_shadow_grad_share` | **0.0** | anything else = the shadow is no longer passive, which is the one thing it must never be |

## 7. What this arm does NOT get to conclude

*(AMENDED, additions only — every existing entry stands.)* Not that a tight-MC label stream would
build a better TRUNK: the twins are frozen with respect to it (§0b). Not anything about the POLICY:
none of these heads feeds pi or vf, so the acting path is bit-identical across A, B and C by
construction, and no behavioural claim can come from a within-run head difference. Not a critic
route change on the SHADOW critic's evidence: that owes the C4 offline gate, and the shadow is the
path to a submission, not a substitute for one.

Not that the critic's *delivery* is fixed (C6's line stays closed — this attacks targets). Not
anything about R2's bait/loop cure claim, which needs an R2 arm and names the loop rate as its own
readout. Not a V-head result: G0 mapped the **win-prob head**, and the main critic V comparison
carries a PBRS caveat that a separate read owes.
