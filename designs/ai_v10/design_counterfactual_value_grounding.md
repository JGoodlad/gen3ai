# design — COUNTERFACTUAL VALUE GROUNDING: three reroll-based attacks on critic bias

> **[STATE 2026-08-22] FORWARD DESIGN — nothing built, nothing scheduled.** Owner-initiated from
> the 2026-08-21 probe triad (expected-SARSA kill → three-axis ordering → label cost model; all
> three in the ledger and memory). Earliest sensible slot: post-flywheel-tick-1, as background
> instrumentation first (§6 G0–G2 need no training run). This is the pre-registered NEW MECHANISM
> that ledger **C6** requires before anyone touches the critic again — it attacks the critic's
> *targets*, not its delivery, which is the line C6 closed. Companion docs:
> [`design_outcome_latent.md`](design_outcome_latent.md) (the representation substrate — per-action
> STATE prediction; this doc is about VALUE truth), [`design_flywheel_tick_tock.md`](design_flywheel_tick_tock.md)
> (whose value-transfer rule §5.1 makes explicit).

## 0. The object

A **counterfactual label factory** — background workers that reroll recorded training decisions
through the reconstruction stack (warm rust `SearchSession` → materializer → forward) to
manufacture **ground-truth value labels the on-policy stream structurally cannot produce** — and
the three training-time consumers of those labels, ordered by the bias cause each attacks:

| rung | label | attacks (bias cause) | staleness class |
|---|---|---|---|
| **R1** | rollout-to-end MC win/loss on **visited** states, R samples | calibration bias (the 0.827 class) | policy-dependent, bounded-lag |
| **R2** | same label on **counterfactual successor** states (unplayed actions' s′) | extrapolation optimism — the optimizer's-curse fuel (cause 2) | policy-dependent, bounded-lag |
| **R3** | k-step **simulator-grounded** bootstrap targets | bootstrap self-reference (cause 4) | none (real transitions) |

All three are training-time / offline. Nothing here is inference-time search; the no-search
constraint is untouched.

## 1. The disease, with its measured record

The critic is the amplifier that turns ~1 bit/game into per-decision credit, so a biased critic
mis-teaches every decision at once. The convictions on file:

- **Sentinel sweep**: median win-prob **0.827** on the top-50 decisions that lost their games.
- **PIT calibration** (gen-10): pit_mean 0.396, coverage80 **0.44** vs nominal 0.80 — optimistic
  AND overconfident.
- **The bait verdict** (2026-08-21): the habit is **exploration starvation** — alternatives at
  p≈0.01–0.03 are never sampled, so their advantages are never realized, so the policy
  self-seals. That is a *value-side blindness with an exploration-side cause*: the critic's
  opinion of unplayed actions is pure extrapolation, corrected by nothing, forever.
- **The optimizer's curse** ties these together: the policy is an adversarial search over the
  critic's outputs — it migrates toward the states where V is most wrongly HIGH, then trains on
  data from exactly those states. Optimistic and pessimistic errors are not symmetric harms:
  undervaluation costs a missed line; overvaluation gets *farmed*.

The offline-RL literature's answer is pessimism (CQL/IQL: when uncertain, undershoot — because
you cannot know). **We own a deterministic, cloneable simulator: we can know.** Our version of the
pessimism fix is strictly stronger — not conservative, *correct*. That asymmetry (a perfect model,
so the entire learned-model-error caveat structure of the MVE/model-based literature evaporates)
is the licence for this whole document.

## 2. Why BIAS and not VARIANCE — the banked kills this doc stands on

Do not re-derive a variance plan; the kills are in the ledger (2026-08-21):

1. **Dice marginalization is dead**: dice are **5.4%** of one-step target variance (3.2% on the
   replication), 15.8% of decisions are dice-free, the worst decile carries 77–82% of it.
2. **No variance lever can pay at all right now**: `train/noise_scale_ratio` reads 0.05–0.10
   across the three newest runs — **~10–20× over-batched**. Gradient noise is not binding.
3. **Bias has a different geometry**: on-policy data has *zero* coverage of unsampled actions —
   a hole, not noise; no batch size averages it away. Bias levers are gated by **bias meters**
   (V-vs-MC divergence, PIT/coverage, exploiter lower bounds), not by `noise_scale` — and those
   meters currently read guilty.

The budget ordering comes from the three-axis decomposition (memory
`project_three_axis_value_variance.md`): **opponent branches first** (59.7% behavior-weighted /
36.5% uniform, the flattest tail — never zero, matters most turns), **our-action branches only
where π is undecided** (π-weighted OUR variance concentrates 86% in the worst decile — exactly
the states the policy is *about* to move on), **dice never**. The OUR×OPP interaction (16.4%
uniform) prices the *joint* branch for R2's enumerator, not for noise.

## 3. The three rungs

### R1 — re-measured MC labels on visited states (the calibration attack)

The realized return is ONE Monte-Carlo sample of V(s); the win-prob head eats exactly one per
state today, drenched in the measured variance. Rollout-to-end rerolls give **many** samples of
the *same* state — fresh dice, fresh opponent lines, same board — turning "one noisy unbiased
sample" into a tight unbiased estimate. Delivery: a supervised pull on the **win-prob head first**
(MC-native, no route change, no C4 exposure) and optionally on V where |V − MC| exceeds a
threshold. Priority policy: the critic-surprise population — exactly the 0.827 class — plus
high-|δ| decisions from the reward tracker. **Price (measured)**: 792 ms per R=8 win-prob label →
~109k labels/day/core; the priority population needs thousands per cycle → *minutes of one
background core*. R1 was never compute-gated; it just didn't know its price until 2026-08-21.

### R2 — MC labels on counterfactual successor states (the curse interruption)

The curse's sequence: V is optimistic about a not-yet-visited region → the policy improves toward
it → by the time on-policy data corrects V, the policy has moved. R2 breaks it at step one:
branch **unplayed** actions on recorded decisions, roll their successors to termination, and
supervise V on those off-distribution states **before the policy farms them**. The critic's
supervised support expands to a *neighborhood around* the on-policy distribution — precisely the
buffer zone where extrapolation lives. Enumerator = §2's ordering (opponent branches broadly;
our-action branches where policy entropy is high or the loops detector fires — the bait states
are R2's home turf, and this is the off-policy correction signal the exploration-starvation
verdict says on-policy learning cannot generate for itself). This rung is the direct value-side
cure claim for the bait pathology, and §6 G4 pre-registers the loop rate as its behavioral
readout.

### R3 — k-step grounded targets (the bootstrap attack)

Replace `r + γV(s′)` with k real simulator steps under the current policy, bootstrapping at depth
k: each unit of critic self-reference is replaced by ground-truth transitions, and the
backward-travelling bias of the bootstrap chain gets k fewer links per update. This is MVE
(model-based value expansion) with a model that cannot be wrong — the literature's compounding
caveat does not exist here. **Honest reach note**: production `gae_lambda=0.80` already mixes
horizons (the one-step target carries weight 0.20), so R3's marginal value over GAE must be
measured, not assumed — it is the *weakest* rung on priors and is ordered last. Cost: k
obs-builds per target, which is the dominant rollout CPU — so R3 lives in the same background
regime, never inline.

### What is deliberately NOT a rung

- **One-ply outcome-fact labels** (damage/KO/status of a branch — policy-independent, never
  stale): those are the *representation* substrate and belong to
  [`design_outcome_latent.md`](design_outcome_latent.md)'s supervision menu. This doc is value
  truth, not representation. The factory serves both; the docs stay separate because the gates
  differ (behavioral deltas there, bias meters here).
- **Teacher value distillation**: ruled out on record. Scalar V-distill **crystallized** the
  critic (dropped `value_cls` rank; the shipped compromise is the FitNets cosine hint on
  `value_pooled`, `a6ae04f`) — and it is *conceptually* wrong besides: V is a fact about a board
  **under an ecology**, and every teacher's V carries its ecology's bias. **The division of
  labour this doc makes explicit: policy knowledge transfers by distillation; value truth does
  not transfer — it is re-measured from the simulator under the student's own ecology.** That is
  the flywheel tick's value rule.

## 4. The factory — five components, three already built

| # | component | status | notes |
|---|---|---|---|
| 1 | **Tap** — reconstructable training decisions | MOSTLY EXISTS | the rust bridge emits `__RECON__` for every episode (seeded AND seedless since `bc00d4d`); build = a ring buffer of recent episode records per env worker + decision index. Training currently discards these; eval keeps them. |
| 2 | **Enumerator** — which decisions, which branches | BUILD (~small) | the priority sampler: critic-surprise, high policy entropy, loops-detector hits, high-\|δ\|; branch budget per §2's ordering. Where most value-per-flop lives. |
| 3 | **Workers** — reroll → materialize → forward | EXISTS, PROVEN | the probe path: warm rust `SearchSession`, `obs_materializer`, critic forward. Anchor: 140/140 + 50/50 reproduce recorded next-V (max 4.8e-05); node/rust label-V bit-identical (0.0 on 50/50). Architecture = the eval-worker pattern (frozen snapshot, nice, work-stealing). |
| 4 | **Label buffer + aux loss** | BUILD (known genre) | same delivery pattern as distill/win-prob losses; flag class `training_coef` (resume-mutable), coefficient-zero byte-identical. |
| 5 | **Consuming heads** | DECIDED | win-prob head first (R1); V main head second; **any new critic ROUTE owes the C4 offline gate** — standing rule, restated here so this doc cannot be cited around it. |

**The cost model** (measured 2026-08-21, `project_counterfactual_label_costs.md`; self-validating
to 0.987):

| configuration | ms/label (K=8, mean turn 24.7) |
|---|---|
| today (node transport, per-arm materialize, eager critic) | **162** |
| existing code only (warm rust session + compiled batched critic) | **28.4** |
| + batch-aware materializer (ESTIMATE) | **~7.7** |

- **The one real build item**: the materializer replays the whole prefix from turn 1 per arm
  (arm_ms = 4.78 + 0.853·turn, R²=0.996; hotspot `LiveView.from_pokemon`, 1084 calls/arm = 50%
  cum) — prefix-share across a decision's arms; the branched turn itself is ~0.5 ms.
- **The transport swap is free**: warm rust `SearchSession` is a paired **289×** over the node
  `reroll_many` path. The compiled critic is real but small (5.9× at B=1, 2.5% of the bill) —
  and **0.91× at B=64 under production BLAS pinning**; never benchmark it unpinned.
- **Coverage**: 4 nice-10 cores ≈ **12M one-ply labels/day = 1.74% of production decisions at
  K=4** (0.88% @K=8); 100% ≈ 230 cores. **The factory is a prioritized SAMPLER by construction.**
- **Known gaps**: rust `search_driver` cannot open **turn 1** (3.35% of move decisions; error is
  JSON on stdout with empty stderr — task #26); forced-switch rounds are uncovered by the
  decision picker (structural, both probes); **CRN shares the dice STREAM, not the roll→event
  mapping** — any within-factory variance estimate needs replicated dice + the independent-half
  cross-product estimator, never a one-seed sweep.

## 5. Design decisions of record

1. **Ground truth or nothing.** Labels come from the simulator, never from another network's
   value output (§3's teacher-distill ruling). V(s′) reads of the *current* critic are permitted
   only as R2 *coverage* supplements, never as bias-correction labels (teaching the critic its
   own opinions is circular).
2. **Staleness bound, not staleness denial.** R1/R2 rollout labels are MC under π_t, consumed at
   π_{t+Δ} — the same staleness class PPO already accepts within an iteration, *bounded and
   shrinking*, traded against bootstrap bias which is *unbounded and self-reinforcing*. Labels
   older than a lag bound (tunable, order one PPO iteration of steps) decay or drop.
3. **Sampler weights are declared, logged, and versioned** — a silent priority change is a
   distribution-shift confound for every downstream readout (the eval-quota lesson from
   `calibration`'s bias_on_wins/bias_on_losses).
4. **Rollouts under the CURRENT snapshot, refreshed on the eval-worker cadence** — the frozen
   snapshot pattern; a factory that lags many iterations is measuring an ancestor.
5. **Budget shape**: R1 gets the priority queue (small, surgical); R2 gets the bulk sampler;
   R3 waits for R1/R2 evidence. Dice branches never (banked kill).

## 6. Pre-registered gates (G0–G4) — each with its kill

- **G0 (free, offline, FIRST)**: the **bias map** — on existing eval traces, compute tight-MC
  (R=8) labels for ~2k states stratified by recorded V, and publish the V-vs-MC reliability
  curve *by state family*. This is the meter that all later gates read. KILL: if the divergence
  is small and unstructured (the 0.827 reading fails to reproduce as a *systematic* V−MC gap),
  the program stops at an instrument. (Priced: ~30 core-minutes.)
- **G1 (offline)**: label quality — R=8 MC labels on states with known outcomes reproduce
  empirical win rates within binomial CI; the anchor arm (recorded action + dice → recorded
  next-V) re-passes on every factory build. KILL: any silent divergence → the factory is GIGO
  and nothing downstream may run.
- **G2 (build, still no training)**: the factory as a **standalone audit instrument** over eval
  traces — "counterfactual audit of a checkpoint" (falsify generalized: per-state, was ANY
  unplayed branch measurably better, with tight MC on both). Useful forever even if the aux
  never ships; also the integration-risk sponge (workers, records, priorities all debugged
  outside the train loop).
- **G3 (training integration)**: tap + buffer + `--cf-*-coef` flags at coefficient zero —
  byte-identity gated, launcher-restart-survival tested (the revival lessons: fork inheritance,
  eval-cadence anchors, `.splitlines()` — assume the integration bugs exist and hunt them).
- **G4 (the A/B fork, pre-registered readouts)**: R1 first, alone. PRIMARY: the G0 bias map
  shrinks on held-out states (V-vs-MC divergence, PIT/coverage80). BEHAVIORAL: the
  critic-surprise rate on losses (the 0.827 class) falls; for R2, the loops/bait rate is the
  named readout (its cure claim is on record — an R2 arm that fixes calibration but leaves the
  loop rate untouched has *failed its motivating claim* and says the bait residual is
  policy-side delivery, not value blindness). GUARD: ELO non-inferiority vs the base arm.
  KILLS: the standing learns≠helps rule — a shrinking aux loss with flat meters is a null; flat
  meters at 2× sample closes the rung.

## 7. Risks, honestly

- **Aux losses that HELP are rare here** (BYOL decodable-but-null; SimSiam deleted at 13% of the
  step; top-K structural null). This doc's counterweight is that its losses are *ground-truth
  supervised on the exact meters that are failing* — but the body count is why every gate above
  is behavioral/meter-based, never loss-curve-based.
- **Train-loop integration is the real cost** (new IPC beside SubprocVecEnv, worker lifecycle
  across launcher restarts) — hence G2's standalone phase and G3's byte-identity discipline.
- **Compute contention**: factory cores compete with eval workers on a box that is always
  training; the factory inherits `nice 10` and the eval-sharding citizenship rules, and its
  throughput numbers were measured on a BUSY box (load 11.6–33.7) — treat them as conservative.
- **The G0 meter is selection-exposed**: eval traces over-capture losses; the bias map must be
  computed selection-aware (the `calibration` machinery's bias_on_wins/bias_on_losses split),
  or it will convict the critic of the sampler's sins.

## 8. What this doc is NOT

Not inference-time search (offline/training-time only; the no-search constraint stands). Not the
outcome latent (representation; separate doc, separate gates). Not a variance program (§2's
kills are banked — cite them, do not re-litigate). Not a critic *delivery* proposal (C6's closed
line stays closed; this attacks targets, and any route change it ever wants owes C4 first).
