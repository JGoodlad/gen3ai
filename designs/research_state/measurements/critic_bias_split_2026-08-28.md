# PROBE G — the critic's leaf error, split into SHARED vs DIFFERENTIAL

*Measured 2026-08-28/29 (labels rolled 21:54–00:38 PDT) · commit `f5b7da5 (tree state the probe ran on)` · CPU-only, 2 cores, `nice 15`.*

## The question

A search that ranks candidate actions by a learned critic only cares about the critic's
error to the extent that error **differs between the actions it is comparing**. Split the
per-decision error vector into the part every action shares and the part that does not:

```
e_d[a]     = C_d[a] - L_d[a]          critic minus Monte-Carlo label, win-prob units
offset_d   = mean_a e_d[a]            SHARED    -- paired evaluation cancels it EXACTLY
resid_d[a] = e_d[a] - offset_d        DIFFERENTIAL -- pairing does not touch it

mean_a e^2 == offset^2 + mean_a resid^2      (an identity, exact per decision)
```

If the error is mostly offset, a paired search is reading a critic far better than its MSE
suggests. If it is mostly differential, the critic mis-ranks and contrastive training is the
binding lever.

## What was measured

- **Subject**: `models/ai_v9_29_rev1_0823` / `step_24000000`, scored by the run's OWN eval snapshot
  (`step_24000000/snapshot.zip`) — the exact weights that wrote the traces.
- **317 recorded decisions** over **178 battles**, **2222 (decision × legal action) cells**.
  Every decision has ≥3 legal actions; half drawn from the top-15% |TD δ| tail (`pivotal`),
  half from the rest (`ordinary`); ≤3 decisions per battle.
- **Labels** `L[a]`: substitute action `a` at the recorded decision, then play the rest LIVE
  to a terminal win/loss — the trainee greedy vs the RELOADED real opponent (a bot rebuilt
  exactly; a pool sentinel reloaded from its pinned snapshot and played STOCHASTIC, the
  regime `eval_worker` recorded). **R = 64 rollouts per action.** No
  horizon adjudication and no |V| cut-off: every label is a rollout to an actual terminal.
- **Common random numbers**: the R post-divergence dice seeds are derived from
  `<battle_tag>:<inv>:cf` with **no action in the salt**, so every sibling action at one
  decision is rolled on the same seed list. The paired difference `L[a]-L[b]` is the
  quantity this probe needs resolved, and CRN is what resolves it.
- **Critic** `C[a]`: the one-ply read a search would use — re-roll the turn under `a` with
  the opponent playing its RECORDED move, materialize the successor obs through the real
  encoder, and read the **win-prob head at s′**. Same units as the label. An action whose
  turn ends the battle is scored 1.0 / 0.0 (a search sees the terminal exactly).

### The label noise floor is MEASURED, not assumed

Every label was rolled as **two independent CRN blocks** A and B of 32
dice each, with `L = (L_A + L_B)/2`. Within a block CRN is intact; across blocks the dice
are independent. With `D = L_A − L_B`:

| quantity | estimator |
|---|---|
| sampling variance of the full-R label | `E[D²]/4` |
| …of the DIFFERENTIAL component | `E[mean_a (D − mean_a D)²]/4` |
| …of the SHARED component | `E[(mean_a D)²]/4` |

The differential floor is the load-bearing one, and it is why a closed-form binomial floor
would not do: CRN has already removed an unknown share of the paired variance, and only a
measurement knows how much. `critic_bias_split_selftest.py` pins the estimators on synthetic data
(shipped beside this file) — the corrected residual returns ~0 for a critic that is a PURE
offset of the truth, and
recovers a known injected differential error to within 2%.

## The decomposition

All numbers are mean-squared error in win-prob units, averaged with **equal weight per
decision** (so a 9-action decision does not outvote a 3-action one). `_true` = raw minus the
measured noise floor.

| component | raw MSE | measured noise floor | **true MSE** | **RMS** |
|---|---|---|---|---|
| SHARED (offset) | 0.04025 | 0.00030 | **0.03995** | **0.200** |
| DIFFERENTIAL (residual) | 0.01636 | 0.00147 | **0.01490** | **0.122** |
| total | 0.05661 | — | **0.05484** | **0.234** |

**Offset share of true MSE: 0.728** (95% CI over battles [0.674, 0.780]) · differential share 0.272.

Mean SIGNED offset: -0.0118 [-0.0403, 0.0157] — the direction the critic is wrong on
average, which paired evaluation removes entirely.

### …and the offset itself splits again, which is what decides SEARCH DEPTH

A *constant* bias shifts every state equally and cancels in any comparison at any depth. A
*per-decision* offset cancels only between siblings — a deeper tree compares states at
different nodes, where it does not. So the same MSE means three different things:

| component | cancels when | true MSE | RMS | share |
|---|---|---|---|---|
| global calibration bias | always (any depth) | 0.00014 | 0.012 | 0.003 |
| per-decision offset | siblings only (1-ply paired) | 0.03981 | 0.200 | 0.726 |
| **differential** | **never** | **0.01490** | **0.122** | **0.272** |

For scale: a single label's own RMS sampling error is 0.042, and the
mean within-decision spread of the labels themselves (max L − min L) is 0.267.

## The decision-relevant loss

MSE is not what a search pays. What it pays is **regret**: the win-prob it gives up by
picking `argmax C` instead of the best action.

Three readings, because label noise biases the naive one:

- **naive** — `max(L) − L[argmax C]`. Biased **UP**: `max(L)` is a winner's-curse maximum
  over noisy estimates, so it exceeds the true best action's value.
- **cross-fitted** — select the comparison action on one half, score it on the OTHER
  (`L_B[argmax L_A] − L_B[argmax C]`, symmetrized). Immune to the curse; biased **DOWN**,
  because the selector is itself only an R/2 estimate. **The truth is bracketed.**
- **noise reference** — `max(L_B) − L_B[argmax L_A]`, symmetrized: the regret a scorer with
  **zero bias and only sampling noise** would post. This is the floor a flip has to clear
  to be a flip.

| statistic | value (95% CI, bootstrap over BATTLES) |
|---|---|
| flip rate — naive | 0.549 [0.491, 0.608] |
| flip rate — cross-fitted | 0.202 [0.174, 0.229] |
| flip rate — noise reference | 0.476 [0.420, 0.536] |
| **regret — naive (upper bracket)** | **0.0859 [0.0692, 0.1057]** |
| **regret — cross-fitted (lower bracket)** | **0.0572 [0.0393, 0.0780]** |
| regret — noise reference | 0.0404 [0.0344, 0.0468] |
| **regret_cf − noise reference** | **0.0168 [-0.0036, 0.0399]** |
| regret_cf median / p90 / p95 | 0.000 / 0.250 / 0.409 |
| regret_naive median / p90 | 0.016 / 0.263 |
| regret — RANDOM action (no-information control) | 0.0988 [0.0820, 0.1169] |
| Spearman ρ(C, L) per decision | 0.282 [0.228, 0.335] |
| Spearman ρ(L_A, L_B) — the LABEL's own reliability | 0.513 [0.459, 0.567] |

The two Spearman rows belong together: `ρ(L_A, L_B)` is the ceiling on `ρ(C, L)` — a critic
cannot correlate with the labels better than two independent measurements of those labels
correlate with each other. Read `ρ(C,L)` **against** it, never against 1.0.

**Capture fraction: 0.712** of the
achievable ranking gain — 1.0 would be ranking as well as a tight-MC oracle at this R, 0.0
no better than choosing at random. It is the regret scale expressed as a statement rather
than a magnitude, and it is the number to quote.

### Does ranking by the critic beat what the policy actually did?

The most program-relevant comparison, and it needs no cross-fitting: `argmax C` and the
recorded action are BOTH noise-free selections, so their CRN-paired label difference is an
unbiased estimate of the true difference. (The eval trainee played greedy — verified 995/995
recorded actions equal the masked argmax — so the recorded action IS the policy's choice.)

| statistic | value |
|---|---|
| **1-ply SEARCH DIVIDEND** `L[argmax C] − L[policy]` | **0.0219 [0.0089, 0.0364]** |
| policy already picks `argmax C` | 0.350 [0.301, 0.399] |
| regret of the POLICY's action (cross-fitted) | 0.0791 [0.0573, 0.1037] |
| regret of the CRITIC's argmax (cross-fitted) | 0.0572 [0.0393, 0.0780] |

## Splits

| cell | n dec | offset share | RMS diff | flip (cf) | regret (cf) | noise floor | capture | 1-ply dividend | ρ(C,L) |
|---|---|---|---|---|---|---|---|---|---|
| `ALL` | 317 | 0.728 | 0.122 | 0.202 | 0.057 [0.039, 0.078] | 0.040 | 0.71 | 0.022 [0.009, 0.036] | 0.282 |
| `stratum=pivotal` | 157 | 0.728 | 0.140 | 0.229 | 0.074 [0.047, 0.103] | 0.046 | 0.64 | 0.030 [0.006, 0.056] | 0.265 |
| `stratum=ordinary` | 160 | 0.729 | 0.101 | 0.175 | 0.040 [0.020, 0.065] | 0.035 | 0.85 | 0.014 [0.002, 0.029] | 0.302 |
| `opp_class=bot` | 159 | 0.728 | 0.140 | 0.201 | 0.081 [0.050, 0.114] | 0.024 | 0.37 | 0.025 [0.006, 0.045] | 0.248 |
| `opp_class=sentinel` | 158 | 0.728 | 0.101 | 0.203 | 0.033 [0.013, 0.054] | 0.057 | 1.89 | 0.019 [0.001, 0.039] | 0.313 |
| `n_legal=3-4` | 21 | 0.554 | 0.145 | 0.143 | 0.052 [-0.006, 0.153] | 0.019 | 0.69 | 0.052 [-0.017, 0.130] | 0.409 |
| `n_legal=5-6` | 106 | 0.690 | 0.140 | 0.212 | 0.077 [0.042, 0.117] | 0.043 | 0.51 | 0.013 [-0.013, 0.043] | 0.280 |
| `n_legal=7-9` | 190 | 0.773 | 0.108 | 0.203 | 0.047 [0.029, 0.065] | 0.041 | 0.88 | 0.023 [0.006, 0.042] | 0.271 |
| `turn_tercile=early` | 108 | 0.759 | 0.119 | 0.213 | 0.059 [0.032, 0.090] | 0.039 | 0.73 | 0.036 [0.012, 0.062] | 0.307 |
| `turn_tercile=mid` | 107 | 0.716 | 0.123 | 0.224 | 0.067 [0.035, 0.099] | 0.040 | 0.49 | 0.007 [-0.017, 0.033] | 0.213 |
| `turn_tercile=late` | 102 | 0.706 | 0.124 | 0.167 | 0.045 [0.017, 0.077] | 0.043 | 0.94 | 0.022 [-0.004, 0.051] | 0.325 |
| `outcome=win` | 132 | 0.823 | 0.076 | 0.140 | 0.004 [-0.005, 0.014] | 0.030 | MISSING | 0.009 [0.000, 0.020] | 0.356 |
| `outcome=loss` | 185 | 0.697 | 0.146 | 0.246 | 0.095 [0.067, 0.126] | 0.048 | 0.50 | 0.031 [0.010, 0.055] | 0.238 |

### Robustness: the SCALAR value head instead of the win-prob head

Rank statistics are invariant to any monotone re-scaling, so this arm asks a different
question: does the run's *other* critic readout order the actions the same way?

| statistic | win-prob head (headline) | scalar value head |
|---|---|---|
| flip rate (cf) | 0.202 [0.174, 0.229] | 0.221 [0.192, 0.248] |
| regret (cf) | 0.0572 [0.0393, 0.0780] | 0.0656 [0.0474, 0.0860] |
| regret (naive) | 0.0859 [0.0692, 0.1057] | 0.0942 [0.0768, 0.1139] |
| ρ(C,L) | 0.282 [0.228, 0.335] | 0.238 [0.181, 0.295] |
| capture fraction | 0.712 | 0.569 |
| **1-ply search dividend** | **0.0219 [0.0089, 0.0364]** | 0.0135 [-0.0007, 0.0280] |

The win-prob head wins on every row, and the dividend is the one that decides a build:
ranking by the win-prob head beats the policy significantly; ranking by the scalar value
head does not. **A search over this checkpoint should read the win-prob head.**

## Caveats

- **The opponent is held at its RECORDED move.** Every arm re-rolls the turn with the opponent replaying what it actually did, so this isolates the critic's error *across our own actions* with the opponent axis frozen. A real search must marginalize over opponent actions, and the three-axis variance read puts the OPPONENT axis at 36.5% (uniform) to 59.7% (behaviour-weighted) of V(s') variance — far above the dice. **So the differential component measured here is a LOWER BOUND on what a search actually faces.** This is the single most important limit on the verdict below.
- **The offset cancels between SIBLINGS, not between NODES.** `offset_d` is per-decision, and it is 72.6% of the error. A 1-ply paired comparison cancels it exactly. A depth-≥2 tree compares states reached at *different* decisions, where it does not cancel at all — and at RMS 0.200 it would then be the dominant term. The 'pairing suffices' half of the verdict is scoped to same-node comparisons and must not be carried to a deep tree.
- **The label is Q under the trainee's GREEDY continuation**, not under the stochastic policy the critic was regressed on during training. That mismatch is small here and was checked rather than assumed: the eval trainee played greedy (995/995 recorded actions equal the masked argmax), so the labels are the Q of the policy that generated these traces. It is still not the training-time regime.
- **Eval traces carry a win/loss quota**, so the `ALL` cell over-samples losses (185 loss / 132 win decisions) and is NOT the on-policy population. Every statistic is therefore also reported within outcome, opponent class, pivotality, legal-action count and turn tercile; read the split that matches the population you care about.
- **One run, one step.** `ai_v9_29_rev1_0823` @ 24M, scored by its own eval snapshot. Nothing here claims the split is stable across generations — though it is notably stable across every stratum *within* this run (offset share 0.728 in ALL, 0.728 pivotal, 0.729 ordinary, 0.728 bot, 0.728 sentinel), which is the kind of invariance that suggests a structural property rather than a sampling artifact.
- **A capture fraction above 1.0 is real, not an artifact** — it means the critic out-ranks the R/2 = 32-rollout Monte-Carlo reference on that cell (the reference is a noisy oracle, not a perfect one). The `outcome=win` cell reports MISSING instead: there the no-information baseline is only 0.007 above the noise floor, so there is essentially no ranking gain available to capture and the ratio is unstable.
- 15 of 2222 candidates ended the battle in the CRN line and were scored 1.0 / 0.0 (a search sees a terminal exactly). At 0.7% of cells this cannot move any statistic reported here.

## Accounting — what was cut, and why

- **Frame**: 6557 reconstructable `move_selection` decisions over 229 battles under `step_24000000`. Excluded by construction and counted, never silent: 229 turn-1 decisions (one per battle — the sampler's `MIN_LABELABLE_TURN` bound), 878 forced-switch rounds (the re-roll layer anchors at start-of-turn move rounds), and 93 decisions with fewer than 3 legal actions (a forced choice cannot exhibit a ranking error, so it is uninformative for this probe).
- **Sampled** 320 of the 6464 eligible: 160 from the top-15% |TD δ| tail (`pivotal`) and 160 from the rest (`ordinary`), at most 3 per battle, order shuffled.
- **Completed 317 of 320** (99.1%), 2222 action cells, **142,208 full-battle rollouts to a terminal**. The 3 losses are one structural gap, not a flake: all three are late decisions (turns 174–220) inside 250-turn stall battles, where the record ends at the forfeit deadline so the offline replay driver replays every command and still has no terminal to reach.
- **Nothing was cut for time.** Both shards finished their full allocation in 159 and 164 minutes against a 235-minute deadline. Measured cost: ~62 s per decision at R=64 over ~7 actions, on 1.86 cores total (2 processes at 93% each, `nice 15`), rust bridge + `torch.compile`d extractor.
- **Opponents were reloaded, never approximated**: 1134 bot cells rebuilt exactly, 1088 sentinel cells loaded from their pinned snapshots and played STOCHASTIC at temp 1.0 (the regime `eval_worker` recorded). Zero `self_model_approx` fallbacks.
- **Pipeline validation (the standing acid test), run before any labels**: reloading the run's own eval snapshot reproduces the recorded trace to `max|Δ|` = 3.1e-05 on V, 5.4e-07 on the win-prob head and 3.0e-06 on the action probabilities (210 decisions); and the lookahead CRN anchor — the chosen action's re-rolled successor must reproduce the real recorded next state — matches `recorded_next_value` to `max|Δ|` = 1e-04 across every sampled decision.
- **Estimator validation**: `critic_bias_split_selftest.py` checks the decomposition identity (exact to 5e-17) and the split-half noise floor on synthetic data where the truth is constructed — the corrected differential returns ~0 for a critic that is a pure offset of the truth, and recovers a known injected differential to within 2% at three magnitudes.
- **Reproduction**: the three scripts are shipped beside this file — `critic_bias_split_labels.py` (the sampler + label/critic rollout harness, sharded and resumable), `critic_bias_split_analyze.py` (the decomposition, run once per critic head), `critic_bias_split_selftest.py` (the estimator gate), `critic_bias_split_report.py` (this rendering). The `.json` beside this file carries a `decisions` array with the critic vector and BOTH label half-blocks for every decision, so every statistic here can be recomputed without re-running a single rollout.

## Verdict

**MIXED — and the mix is the finding.** The registered prediction (from the G0 bias-map verdict "the defect is RESOLUTION not offset") was that the differential component would be substantial enough that pairing alone is not enough and contrastive critic training is the lever, with the offset share large but decision-irrelevant. Scoring it honestly, in three parts:

**Confirmed — the offset share is large.** 72.6% of the critic's true leaf MSE is shared across the actions at a decision (95% CI [0.674, 0.780]), and a paired comparison cancels it exactly. Almost none of it is a global calibration bias (0.26% of the total; signed bias −0.0118, CI spanning zero) — it is a *per-decision* offset at RMS 0.200. So the critic is not systematically optimistic or pessimistic; it is differently wrong at each state, by a lot, in a way that hurts nothing when you only compare siblings.

**Confirmed — the differential is real and it costs decisions.** It survives the measured noise floor at 27.2% of true MSE, RMS 0.122 in win-prob units against a mean within-decision label spread of only 0.267 — the critic's ranking error is ~46% of the range it has to resolve. One decision in five is a genuine argmax flip after cross-fitting (0.202 [0.174, 0.229]), and the cost is bracketed at 0.057 [0.039, 0.078] (cross-fitted, biased low) to 0.086 [0.069, 0.106] (naive, biased high) win-prob per decision. It concentrates exactly where it should: losses (0.095) over wins (0.004), pivotal (0.074) over ordinary (0.040).

**REFUTED — that it is the binding lever.** Against a no-information baseline of 0.099, the critic already captures **71% of the achievable ranking gain**, and its excess regret over what a 32-rollout Monte-Carlo oracle itself incurs is **+0.017 [−0.004, +0.040] — it does not clear zero**. The prediction's ordering does not follow from its premise: a component can be 27% of MSE and still leave most of the decision quality already recovered. Contrastive training is *sized* here, not *convicted* — the whole prize is ≤5.7pp of per-decision regret and plausibly much less.

**The bankable result is a different one, and it is significant.** Ranking the legal actions by the critic's one-ply win-prob read beats the action the policy actually played by **+0.0219 [+0.0089, +0.0364] win probability per decision** — the policy and the critic pick the same action only 35% of the time, and the critic is the better of the two. Depth-1 search with paired evaluation is therefore already positive-value on this checkpoint before any critic repair. **And use the win-prob head, not the scalar value head**: the same measurement on the value readout gives capture 0.57 (vs 0.71) and a dividend of +0.0135 [−0.0007, +0.0280] that does not reach significance. That is a free choice with a measured cost to getting it wrong.

**Program ordering, in one sentence:** *pair first* — paired evaluation cancels 73% of the critic's error for free and its 1-ply ranking already beats the policy by a significant +2.2pp win-prob, so ledger item (2) moves ahead of item (1); contrastive critic training stays on the board as a sized second lever worth at most ~5.7pp of per-decision regret rather than the program's binding constraint — **with the sharp caveat that both halves of that sentence are 1-ply claims**: the per-decision offset that pairing cancels between siblings does *not* cancel between nodes at different decisions, so at RMS 0.200 it becomes the dominant error term the moment the tree goes to depth ≥2, and the opponent axis this probe froze is the larger variance source in the first place.

**What would change this verdict, and it is cheap to run:** re-run the identical decomposition with the opponent action *marginalized* instead of held at its recorded move (the α-head gives the weights, and the re-roll layer already accepts an explicit opponent choice). If the differential share rises materially once the opponent is free, the registered prediction is right after all and the ordering flips back. That arm is a change of one field in the `arms` list this probe already builds — it was not run here because it is a different question, not because it is expensive.

