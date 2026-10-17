# What the end state's compute buys: the cost / value ledger, the toy to its plateau, and plateau-breakers on both architectures

**Status: DESIGN 2026-10-16 (planning input; nothing built, nothing launched).** ALWAYS-CURRENT (owner, 2026-09-27):
a later decision or measurement that differs updates this doc in the same commit, and the Decision record (§7) says
what changed and why.

**Trigger.** The toy control T-ENT ("Toy: entity tokens + attention only") against production
([`../research_state/measurements/toy_crosses_2026-10-16/RESULTS.md`](../research_state/measurements/toy_crosses_2026-10-16/RESULTS.md);
ledger 2026-10-16 "The T-ENT toy at EQUAL COMPUTE BEATS production at 15M"):

| cross | toy score − 50 |
|---|---|
| matched steps, 15M | −14.1 pp [−17.2, −11.0] |
| matched steps, 5M | −25.5 pp [−28.0, −23.0] |
| equal GPU wall clock (toy ≈ 47M vs production 15M, ≈ 2.5 h) | **+6.5 pp [+4.5, +8.4]** |
| toy at its 100M cap vs production 15M | +10.2 pp [+7.5, +12.9] |

**The owner's framing (2026-10-16, binding on this doc).** The equal-time win is an interesting DATA POINT, not a
concern about production: production wins per step, and the ceiling and plateau-breaker questions are open. The plan
is (1) production to its plateau (X26, live); (2) the toy to ITS plateau too; (3) then PLATEAU-BREAKERS (PFSP / league,
search, recipe changes, …) on BOTH; (4) see what is meaningful, and whether the production architecture makes a
meaningful difference there. **This doc therefore makes NO "replace or slim production" recommendation.** It gives:
- §1–§3: the COST and VALUE ledgers, as planning input;
- §4: the toy deep run to its plateau (cost, stop rule, after X26);
- §5: how to run plateau-breakers on both architectures, with the toy as a cheap testbed where results transfer,
  and how we would know they transfer;
- §6: efficiency ideas, only as cost-reducers for that programme.

## 0. One screen

- **Parameters are not the cost.** The policy tower (`--policy-readout tower`) is 1,117,952 of production's 2,609,183
  parameters (43 %) and 2.7 % of its FLOPs. The toy keeps the same tower: 723,200 of its 1,423,748 (51 %).
- **The cost is TOKENS × ROUNDS.** The trunk is 69.5 % of the learner's counted FLOPs (3 rounds × 64 tokens). The 32
  EVENT seats are half of every round, so history costs ≈ **37 % of the learner's FLOPs**, about 18× its own encoder.
  That makes it the largest single cost. The third round is 23 % (+6.9 s per GPU update, MEASURED). The E4 / E5 /
  OTHER seats are ≈ 15 % (§1.3).
- **The damage op and the edges are light on FLOPs (≈ 7 %) and heavy on CPU time.** They take 35 % of the CPU micro-step
  and ≈ 64 % of CPU inference, together with the beliefs and the extractor glue. They are 51,000 small eager ops per
  forward. On the GPU, Inductor fuses them. Their GPU share is UNMEASURED; the kernel-class profile caps the whole
  non-GEMM class at ≈ 41 % (§1.4).
- **At the whole-model level the GPU update tracks counted FLOPs.** Production / toy update time is 42.1 / 9.23 s =
  4.56×, against a FLOP ratio of 4.67×. Collect does not track FLOPs: the Rust env is ≈ 7.8 s of every
  98,304-decision iteration, ESTIMATED. That is 13 % of production's iteration and ≈ 40 % of the toy's (§1.1).
- **Value.** Only two components have an ISOLATED strength read: static tokens (−2.3 pp alone) and X5 hypothesis
  tokens (equal per step, −5.9 pp per GPU-hour). The end state as a BUNDLE passed (+0.6 pp [−0.7, +1.9]). History,
  depth 3, edges, seats, the op, move resolution, the intent head and the policy tower are UNKNOWN in isolation. Almost
  every read is matched-step at 15M, i.e. PROVISIONAL (§2).
- **The plan, ranked by information per GPU-hour (§4–§5):**
  1. **X33 — the equal-time CURVE.** Toy checkpoints against X26 checkpoints at equal effective learn time, at 2–3
     budgets. CPU only, and the checkpoints already exist (X26 is past 45M).
  2. **R-T2.** Already registered; CPU.
  3. **X34 — the toy deep run.** One seed from its 100M cap, under X26's own Tier-1 rule, after X26: ≈ 12–29 GPU-h.
  4. **X35 — plateau-breakers on both architectures,** with a transfer calibration before the toy may gate anything.
  5. **X36 — the toy + 1 component ladder.** DEFERRED.
- **One owner question (§6.4):** run X34 on a rented cloud GPU in parallel with X26, or after X26 on ours.

## 1. The COST ledger

### 1.1 Where a training iteration goes (MEASURED, the runs' own per-iteration rows)

`iteration_split.py` (this doc's measurement folder,
[`../research_state/measurements/compute_efficiency_2026-10-16/`](../research_state/measurements/compute_efficiency_2026-10-16/))
reads the medians of each run's `collect_ms`, `update_wall_s` and `gpu_wait_ms_per_host_step` over its whole log.
One iteration = 98,304 decisions; bf16; RTX 3080 Ti.

| run | iteration (s) | update | collect | GPU wait / host step | fps |
|---|---|---|---|---|---|
| `rb_bf_eb_s2005` (production, 15M) | 59.3 | 42.1 s (71 %) | 17.2 s (29 %) | 18.2 ms | 1,697 |
| `rb_x26_s2601` (production + ride-along heads + ladder games, live) | 65.6 | 42.1 s (64 %) | 23.5 s (36 %) | 29.5 ms | 1,600 |
| `rb_toy_ent_s4002` (T-ENT, to 100M) | 19.4 | 9.23 s (48 %) | 10.2 s (52 %) | 4.64 ms | 5,075 |
| `rb_toy_ent_s4003` | 18.8 | 9.22 s (49 %) | 9.62 s (51 %) | 4.78 ms | 5,138 |

**The collect split is ESTIMATED, under two assumptions:** equal host steps per collect, and equal env time.
Solving production and toy together gives ≈ 516 host steps per collect, so:
- production's collect ≈ 9.4 s of inference wait + ≈ 7.8 s of Rust env and host work;
- the toy's ≈ 2.4 s + ≈ 7.8 s.

The env floor alone would cap any network at ≈ 12,600 fps.

**What follows from this:**
- **Production is UPDATE-bound.** The update is 71 % of an iteration, and inference another 16 %.
- **The toy is half ENV-bound.** It is the more exposed of the two to agent CPU load on the shared box (§5.4).
- X26's collect is +6.3 s over EB's, with 11 ms more GPU wait per host step. The ride-along heads are update-only, so
  they are not the cause. Its source is UNVERIFIED: longer games, agent CPU load and the in-learn ladder games are the
  candidates (FINDING F4).

### 1.2 Per component (CPU-MEASURED: FLOPs, params, CPU time; forward and backward)

**Method.** `component_cost.py`, at `38d76746`:
- CPU, eager, fp32 (production's GPU recipe is bf16), ONE torch thread, under rule 5b.
- 2,048 real labelled rows: the K9 learner golden's buffer via `r1_batch`.
- The learner exactly as `learner_golden.build_learner()` builds production, stepped through the REAL micro-step:
  PPO + every belief, intent and win-prob loss. The ride-along heads are the separate eager tail and are not
  included; their cost is in §1.4.

**Attribution:**
- Every module to depth 3, the `DamageOperator`'s public methods (the edge-cell kernels are methods, not modules) and
  the extractor's private helpers carry a `record_function` label.
- A forward op's self time and FLOPs go to its innermost label.
- A backward node goes, with its whole subtree, to the label of the forward op with the same autograd
  `sequence_nr`.
- FLOPs are the profiler's counted matmul-family FLOPs (mm / addmm / bmm / SDPA's decomposition); elementwise FLOPs
  are mostly NOT counted.
- Rolled up by `read_cost.py`; raw output in `result/cost_production.json` and `result/cost_toy.json`.

**Totals:**

| | production | toy (`rb_toy_ent_s4002` final) | ratio |
|---|---|---|---|
| learner params | 2,609,183 | 1,423,748 | 1.83× |
| counted MFLOP / row, forward | 94.8 | 17.5 | 5.4× |
| counted MFLOP / row, forward + backward | 244.6 (micro-step) / 240.7 (surrogate) | 52.4 (surrogate) | **4.6×** |
| CPU fwd + bwd, 2,048 rows (surrogate step, unprofiled median) | 24.2 s | 2.34 s | 10.4× |
| CPU inference (T2 forward, eval, no grad), 256 / 64 rows | 0.966 / 0.309 s | 0.073 / 0.025 s | 13× / 12× |
| GPU update, bf16 compiled (§1.1, MEASURED) | 42.1 s | 9.23 s | **4.56×** |

The "surrogate" step is the T2 forward with grad and loss = −mean log π(a) − mean V. It is the same for both nets,
because a toy checkpoint loaded at HEAD has no learner object to run the real micro-step. On production the two steps
agree within 2 % in FLOPs and 3 % in time.

**Production, by component.** The production micro-step at 2,048 rows. FLOP% is the share of the 244.6 MFLOP / row
(forward + backward). CPU T% is the share of PROFILED CPU time (forward self + backward subtree), and the profiler
inflates many-small-op code. "Inference CPU T%" is the T2 forward at 256 rows. The trunk's parameters (401,536,
≈ 132k per round) sit in its own row.

| component | params | fwd MFLOP/row | bwd MFLOP/row | FLOP% | CPU T% (update) | CPU T% (inference) |
|---|---|---|---|---|---|---|
| trunk rounds 1 + 2 + 3 (64 tokens; §1.3 splits them by token group) | 401,536 | 3 × 18.9 | 3 × 37.8 | **69.5** | 34.0 | 20.8 |
| edges: the cell kernels (the op's physics → 15 families' cells) | 0 | 16.4 | 0.0 | 6.7 | 11.4 | **31.1** |
| edges: the bias maps (cells → `[B, 4, 64, 64]`, 17 slice writes) | 696 | 0.05 | 0.06 | 0.0 | **24.0** (backward 37.9 % of all backward) | 1.3 |
| damage op: incoming rows, op content, the recovery / fact rules | 10,589 | 0.8 | 0.6 | 0.5 | 6.5 | 12.8 |
| static per-mon tokens + board + OBS-FACTS + embeddings | 170,464 | 4.9 | 9.8 | 6.0 | 2.4 | 3.1 |
| beliefs: move / HP-type / item / spread heads | 141,334 | 1.9 | 3.5 | 2.2 | 2.8 | 8.1 |
| beliefs: HiddenOppBeliefPool | 199,808 | 2.8 | 5.6 | 3.5 | 1.5 | 1.4 |
| beliefs: BeliefHead aux readout (training-only output) | 103,456 | 1.6 | 2.5 | 1.6 | 0.4 | 0.5 |
| beliefs: X5 hypothesis tokens + T0 prior | 87,152 | 0.9 | 0.3 | 0.5 | 1.0 | 4.0 |
| history: event-seat ENCODER only (its trunk cost is in §1.3) | 25,824 | 1.6 | 3.2 | 2.0 | 0.7 | 0.7 |
| entity seats E3 / E4 / E5 ENCODER only | 9,600 | 0.1 | 0.2 | 0.1 | 0.2 | 0.4 |
| opponent intent (flat pointer) | 24,832 | 0.7 | 0.7 | 0.6 | 0.2 | 0.3 |
| move resolution | 2,144 | 0.03 | 0.05 | 0.0 | 0.3 | 0.8 |
| heads: policy tower (`pre_proj_norm`, `projection`, `mlp_extractor`) | **1,117,952** | 2.2 | 4.5 | 2.7 | 0.7 | 0.7 |
| heads: CLS pools | 199,296 | 1.9 | 3.8 | 2.3 | 0.9 | 0.9 |
| heads: critic (win head + `value_entity_pool`) | 59,457 | 0.5 | 1.0 | 0.6 | 0.3 | 0.3 |
| heads: pointer action head | 55,043 | 0.3 | 0.6 | 0.3 | 0.1 | 0.2 |
| extractor glue + losses / masking | — | 1.3 | 0.3 | 0.7 | 11.8 | 12.2 |

**The toy, by component** (surrogate step):

| component | params | FLOP% |
|---|---|---|
| trunk, 2 rounds of ≈ 19 tokens (12 mons, 3 board tokens, 4 E3 move seats) | 269,056 | 57.2 |
| static tokens | 161,536 | 22.7 |
| CLS pools | 199,296 | 10.2 |
| policy tower | 723,200 | 8.3 |
| pointer + critic | 66,436 | 1.5 |

### 1.3 The trunk, split by token group (ANALYTIC; matches the measurement to 0.2 %)

**The formula.** One round at T tokens costs f(T) = 2 · (131,072 · T + 256 · T²) FLOP / row, forward:
- the 131,072 term is the per-token projections (QKV, out, FFN 128 → 256 → 128);
- the T² term is attention.

f(64) = 18.87 MFLOP; measured 18.91. Backward is ≈ 2× forward. Attention is only 11 % of a round at T = 64, so the
trunk's cost is nearly LINEAR in the token count.

| token group (production order) | tokens | share of a round | share of the learner's FLOPs (3 rounds, fwd + bwd) |
|---|---|---|---|
| 12 mons + 3 board + 4 E3 move seats | 19 | 27.4 % | 19.0 % |
| E4 (6) + E5 (6) + OTHER_species (1) | 13 | 19.9 % | 13.8 % (15.4 % as the marginal from production, attention included) |
| **event seats (history)** | **32** | **52.8 %** | **36.6 %** |

The groups and the rounds overlap: the third round's 23.2 % is spread over the same token groups. **Marginal costs
for the §5 ladder,** from the toy's 19 tokens:
- + the 32 event seats ≈ +9.5 MFLOP fwd per round (+185 % of the toy's trunk);
- + E4 / E5 / OTHER ≈ +3.7;
- + a third round at 19 tokens ≈ +5.2.

### 1.4 Where the GPU differs (MEASURED where cited; else ESTIMATED)

| quantity | value | tag / source |
|---|---|---|
| the third trunk round | **+6.9 s per update (+15.5 %)**, fp32, 44.5 → 51.4 s; T2 +8.6 % / +10.3 % per flush (64 / 256 rows) | MEASURED, `perf_phase_2026-10-10` window A / C |
| the rest of the end state over static + recovery + depth 3 (move resolution, speed physics, principled reductions, OBS-FACTS) | +1.8 s; exact KO +0.3 s, since folded into the closed form at 0 | MEASURED, same |
| X5 hypothesis tokens (legacy-encoding surface) | update +12.7 % (40.96 → 46.20 s), T2 +45.6 %, cycle +16.7 % | MEASURED, `x5_cost_ablation_2026-10-04/`, `x5_paired_speed_2026-10-05/` |
| ride-along heads (X26 only) | +0.65 s / +0.97 s with every variant (≤ 1.45 %), on a 67 s legacy update | MEASURED, `ridealong_baseline/overhead_variants_2026-09-30.json` |
| bf16 trunk | −22.5 % `train_ms` (54.26 → 42.07 s over 16 runs) | MEASURED, the bf16 A/B |
| end-state update kernel time, by class (fp32, one profiled epoch) | GEMM 35.8 %, Triton pointwise 26.4 %, attention 15.1 %, Triton reductions 14.8 %, other 7.8 % | MEASURED, `perf_phase` window A |
| whole-model update ratio, production / toy | 4.56× (GPU) vs 4.67× (counted FLOPs) | MEASURED both sides |
| the edges' bias-map writes on the GPU | — | **UNMEASURED.** On CPU they are 24 % of the update, all in the backward of 17 slice writes into a `[B, 4, 64, 64]` tensor. On the GPU they are memory-bound, and Inductor's functionalisation may or may not fuse them (T44) |
| the op's cell kernels and the belief heads on the GPU | — | **UNMEASURED** individually. At most the pointwise + reduction share (≈ 41 % of kernel time, including the trunk's own LayerNorm / softmax / residual work) |

**The reading.** The GPU spends LESS than its FLOP share on the trunk (one round: 23 % of FLOPs, 13–15 % of the
update) and more on everything else. The two are consistent with the whole-model ratio:
- the trunk is GEMM- and attention-bound and compute-efficient;
- the op, edges, beliefs and glue are memory- and launch-bound elementwise work, which bf16 does not speed up
  (pointwise −7 % under bf16 vs GEMM −28 %, attention −51 %).

**On CPU the picture inverts.** The op's ≈ 51,000 eager ops per forward dominate, which is why CPU inference is 13×
the toy's, and why every offline CPU meter (crosses, Tier-1 plateau checks) pays the op far above its FLOPs (§6, T41).

### 1.5 The top costs

| # | cost | how big | value (§2) |
|---|---|---|---|
| 1 | **History: the 32 event seats riding every trunk round** | ≈ 37 % of the learner's FLOPs (+2 % encoder); GPU share ESTIMATED ≈ 20 % of the update (3 rounds × ≈ 13 % per round, MEASURED for the third, × the events' 53 % of a round) | UNKNOWN in isolation |
| 2 | **Trunk depth 3 (the third round)** | 23 % of FLOPs; **+6.9 s per GPU update, +15.5 % (MEASURED)**, the largest measured GPU delta of the end state over legacy | UNKNOWN (never isolated) |
| 3 | **Edges: cell kernels + bias maps** | 6.7 % of FLOPs, but 35 % of the CPU update and 32 % of CPU inference; GPU UNMEASURED | UNKNOWN on any current surface |
| 4 | **E4 / E5 / OTHER seats in the trunk** | ≈ 15 % of FLOPs | UNKNOWN |
| 5 | **The belief stack** (X5 tokens + four heads + pool + aux readout) | 7.8 % of FLOPs; X5 alone +12.7 % update, +45.6 % T2 (MEASURED, legacy surface) | X5: equal per step, NEGATIVE per GPU-hour; told-the-truth knock-ins NULL at 15M |

**Not a top cost, despite appearances:**
- the policy tower: 43 % of parameters, 2.7 % of FLOPs;
- the damage op's own rows: 0.5 % of FLOPs, though CPU-heavy;
- move resolution: ≈ 0;
- the pointer head and the critic.

## 2. The VALUE ledger

What each component is KNOWN to buy, from measured evidence only. Read the table with these qualifiers:
- **Matched-step unless stated.** Every read below is matched-step except X5's matched-wall-time read and the toy's
  R-T5.
- **15M is PROVISIONAL.** A 15M read is provisional (owner rule, 2026-10-11: a path claim needs a plateaued model).
  The knock-in control gained +4.4 pp in 3M steps from a 15M parent, so 15M is far from plateau.
- **Legacy reads show DEPENDENCE, not value.** "Dependence" (|dV|, action flips under zeroing) says what a trained
  network LEANS on, never what the part buys (`ai_v9/design_frame_deletion_coverage_gaps.md`).

| component | measured value | where / when | verdict | citation |
|---|---|---|---|---|
| **the END-STATE bundle** vs legacy production | **+0.60 pp, 90 % [−0.72, +1.92]** | 8 × 8 at 15M, current surface | PASS (±2 pp rule; not superiority); the bisection never ran, so nothing is attributed | ledger 2026-10-12 "THE END-STATE CLOSING TEST: PASS"; `design_endstate_closing_test.md` §9 |
| static per-mon tokens (alone, vs the legacy role encoder) | −2.32 pp [−3.60, −1.04]; deficit flat 5/10/15M; −10.7 pp on stall | 8 × 8 at 15M, pre-end-state static port | NEGATIVE alone (registered outcome NOT DETECTED); adopted inside the bundle | ledger 2026-10-09 "STATIC-TOKEN SCREEN LOOK 3"; UNDERSTANDING §4.6 |
| damage operator (+ op_content, prefuse, worst-case rows) | no op-off arm ever trained; legacy dependence large (gen-5 zeroed: 65 % flips, \|dV\| 5.54); gen-5 concat deletion ~44 Elo SUGGESTIVE cost | legacy ai_v9, 9.6–40M | **UNKNOWN** | ledger 2026-08-11 "⚠️ CORRECTION … I quoted an SE as a CI"; 2026-08-08 "Gen-3 40M gate" |
| op reductions, exact KO, speed physics | correctness / calibration reads only (speed physics' Brier NOT better than the logistic) | CPU, real battles | UNKNOWN for strength | ledger 2026-10-08 "F6b PRINCIPLED OPERATOR REDUCTIONS"; 2026-10-07 "F7b's SPEED PHYSICS" |
| static-recovery / fact levers (hazard costs, EOT residual, status, effective stats, actor / target state, OBS-FACTS, closure) | only inside the bundle; chosen from decode gaps (Spikes R² 0.33 vs 0.46; move-seat HP 0.59 vs 0.80); the closure fixed a GIGO leak | static screen arms at 15M | UNKNOWN individually | ledger 2026-10-09 "THE REPRESENTATION PROBE BATTERY"; 2026-10-10 "THE FOUR-MOVE CLOSURE" |
| move resolution (vs the seven per-action blocks) | never screened alone | — | UNKNOWN (cheap: ≈ 0 % FLOPs) | readiness §3.1 (C-NOCELL moot) |
| **X5 hypothesis tokens** vs blob belief | **per step −0.99 pp [95 % −2.92, +0.94], non-inferior; per wall time −5.91 pp [−7.54, −4.28], INFERIOR**; intent log loss −0.151 nats (SIGNIFICANT) | 8 × 8 at 15M, legacy-encoding surface | EQUIVALENT per step, NEGATIVE per GPU-hour | ledger 2026-10-07 "X5 A/B LOOK 3"; UNDERSTANDING §4.5 |
| the belief ceiling (oracle reveal) | told species +2.44 pp [+0.31, +4.57]; full set +2.18 [+0.53, +3.84] | same A/B, 15M | WITHIN FLOOR: bounds any belief gain at this budget | same |
| belief knock-ins on the end state (told the truth, +3M) | moves −0.38 [−1.86, +1.10], items +0.13, species −0.59, spreads +0.34 [−2.51, +3.18] | current surface, 15 → 18M | NULL / NOT DETECTED (PROVISIONAL, one parent) | ledger 2026-10-15 "THE KNOCK-IN FORKS" |
| learned belief deltas zeroed at play | pool teams −5.2 pp [−7.4, −3.0]; ladder teams −0.4 [−2.5, +1.7] | legacy, 103M (near plateau) | SIGNIFICANT on pool only (memorisation) | ledger 2026-09-24 "BELIEF WIN-RATE A/B READ" |
| belief as a search input | honest worlds from the net's belief buy nothing measurable (Tier 1) | end-state / production 15M seeds | NOT DETECTED | ledger 2026-10-15 "HIDDEN-INFO STUDY TIER 1"; [`search_ledger.md`](search_ledger.md) |
| belief aux losses (trunk shaping), HiddenOppBeliefPool | never ablated (`belief_grad_mode detached` built, never screened) | — | UNKNOWN | `design_hand_computed_features.md` §5 |
| opponent intent head | no strength read; under-reacts (swings 58 % as far as the truth) | X5 finals, 15M | UNKNOWN | ledger 2026-10-07 "X20's bending detector" |
| entity seats E3 / E4 / E5 | legacy E4-off dependence (22.5 % flips); E4 K = 6 covers the true move 89.3 % | legacy gen-4 | UNKNOWN | ledger 2026-08-11 "G2a SEAT COVERAGE measured" |
| edge-bias families | no edges-off arm ever trained; legacy dependence grows 1.6–2.7× from 9.6M to 40M | legacy ai_v9 | UNKNOWN | ledger 2026-08-08; ARCHITECTURE §5.4 |
| history (32 event seats, pair history `h`) | gen-13 event seats \|dV\| 2.77 vs frames 1.30; the gen-14 bundle −38.3 Elo [−55.0, −21.6] unattributed; **the seat count 32 was never sized** (audit F19) | legacy, 25M | UNKNOWN | ledger 2026-08-17 "Method: a dV ablation cannot license an irreversible deletion on its own"; `design_arch_audit.md` F19 |
| trunk depth 3 vs 2 | never isolated (closing-test rung B2 not run); skipping round 2 changes ~40 % of actions (descriptive) | static screen finals, 15M | UNKNOWN | ledger 2026-10-09 (probe battery); `design_static_tokens.md` §13 |
| policy tower (`tower` vs `trunk` readout) | built and GPU-checked, never screened | — | UNKNOWN | ledger 2026-10-06 "Architecture audit F2" |
| pointer head, CLS pools, `value_entity_pool` | dependence only (`value_entity_pool` \|dV\| 6.28 = 97 % of the critic routes, gen-13) | legacy | UNKNOWN | ARCHITECTURE §3.4 |
| win-prob critic vs the shaped critic | −34 Elo vs a 38 floor at 26M | legacy | WITHIN FLOOR | UNDERSTANDING §4.2b |
| ride-along heads | detached, bit-identical-neutral by construction | — | no effect by design | ledger 2026-10-15 "X26's ride-along registration re-based" |
| **the machinery as a whole vs T-ENT** | per step −14.1 pp (15M), −25.5 pp (5M) for the toy; at equal GPU time the toy +6.5 pp; at the toy's 100M cap vs production 15M, the toy +10.2 pp | 3 × 8, 15M production | per step: production SIGNIFICANTLY ahead, and the gap SHRINKS with steps; the ceiling is UNREAD (R-T2) | ledger 2026-10-16 (toy crosses); readiness §3.5 |

**Equal-compute evidence anywhere:**
- X5's −5.9 pp per wall time;
- R-T5;
- bf16: +1.06 pp per step at −22.5 % time, so ahead per GPU-hour by an amount never measured;
- the learner battery's E5 (+15.7 % steps per GPU-hour at an equivalent untaught Δ).

**The end state's adoption has NO matched-compute read.** The closing test's speed term was not read with the verdict.

## 3. Value against cost: where the gaps are

| component | cost (§1) | value (§2) | gap class |
|---|---|---|---|
| history event seats | **HIGH** (≈ 37 % FLOPs) | UNKNOWN; 32 never sized | **high cost, unknown value** |
| trunk depth 3 | **HIGH** (23 % FLOPs; +15.5 % GPU update) | UNKNOWN; adopted inside the bundle, unscreened | **high cost, unknown value** |
| edges (cells + maps) | MEDIUM on GPU (unmeasured), HIGH on CPU | UNKNOWN on any current surface | unknown on both sides; measure the GPU cost first (T44) |
| E4 / E5 / OTHER seats | MEDIUM (≈ 15 %) | UNKNOWN | high-ish cost, unknown value |
| belief stack | MEDIUM (7.8 % FLOPs; +12.7 % GPU update for X5) | X5 NEGATIVE per GPU-hour; knock-ins NULL at 15M | measured value ≈ 0 at 15M; PROVISIONAL until plateau |
| damage op (rows, rules) | LOW FLOPs, HIGH CPU | UNKNOWN (large dependence) | cheap on the GPU; the CPU meters pay for it (T41) |
| static tokens | LOW | NEGATIVE alone, inside a passing bundle | the owner's architecture direction (`project_arch_direction`); kept |
| policy tower | LOW compute, 43 % params | UNKNOWN | not a compute question |
| move resolution, intent, pointer, critic | LOW | mostly UNKNOWN | not worth a screen on cost grounds |

**What this table is for.** It is not a slimming list (owner, 2026-10-16). It says where production's per-step lead
can live:
- ≈ 75 % of the FLOPs (history, depth 3, seats) have NO isolated value read;
- the two components that were measured in isolation show no per-GPU-hour gain at 15M.

**That makes the plateau the place to look** (§4–§5). It also makes the X36 ladder the place to attribute, if a
breaker result ever needs to know which component carries the difference.

## 4. The TOY DEEP RUN to its own plateau (X34)

### 4.1 Why, and what it answers

R-T2 (registered, readiness §3.5) plays the toy at its 100M CAP against X26's plateau. It is biased toward production
by the toy's unrealised gain: the toy is still climbing at 100M, at +18 to +27 Elo per 20M (readiness §3.8). Step (2)
of the owner's plan needs the toy AT ITS PLATEAU, for two reasons:
- **"CEILING" needs it.** CEILING can only be worded "at each one's plateau" with a toy plateau checkpoint.
- **§5 needs it.** Every plateau-breaker on the toy starts from it.

### 4.2 The recipe

- **What runs.** A CONTINUATION of `rb_toy_ent_s4002` from its 100M cap (`final_model.zip`).
  - s4002 is the clean seed. `s4001b` inherits ≈ 9.2M post-canary steps (F36). s4003 is the second candidate.
  - One seed: §4.4 explains why.
  - Same recipe as T-ENT (production's fresh recipe; bf16; no ride-along heads); `--checkpoint-every-steps 1000000`;
    `--snapshot-ladder-games 0`.
- **The canary (F25).** The T-ENT seeds are pinned at `3899277a`, before T36's canary-fixture fix. The continuation
  must carry T36's fixture, or the canary's boundary-sensitive false FATAL (I30's class) stays live for 100–400M more
  steps. How to do it is the Training session's call:
  - the toy's argv parses only at its pin (readiness §3.2);
  - **STAGED 2026-10-17 at HEAD** ([`x34_toy_deep_stage_2026-10-17/STAGED_LAUNCH.md`](../research_state/measurements/x34_toy_deep_stage_2026-10-17/STAGED_LAUNCH.md)):
    the 17 deleted flags are dropped (vacuous: no op, no opponent intent). The HEAD build matches the checkpoint's
    316 keys and shapes, the checkpoint loads strictly, and its fp32 forward on 191 real rows is BIT-IDENTICAL at
    HEAD (`6dcc769c`) and at the pin (`3899277a`). `toy_network_identity_test.py` pins this.
  - The continuation is a FORK into `rb_toy_ent_s4002_deep` with `--pin-commit <main>`. The argv types no arch flag:
    the arch is inherited and the recipe restored from the parent. `checkargs` and `--dry-run` both pass ("would
    launch", +399,918,572 steps). `launch_x34.sh` re-runs every pre-check at launch.
- **Data.** None of the 10-14 learnset fix reaches the toy (no priors). The observation did not move under it
  (991 / 991).

### 4.3 The stop rule: X26's OWN Tier-1 rule

- **The rule.** `python -m main.plateau tick <run> --delta-steps 10000000 --lag 5`, i.e. Δ 10M, W 50M, the registered
  X26 parameters. Run it from 100M, on the agent cores (rule 5b).
- **Why X26's rule and not the toy's.** The toy's own rule (Δ 5M, W 20M, readiness §3.4) asks for ≤ 0.51 against the
  node 20M back. A curve climbing +18 Elo per 20M reads GAIN forever: that rule cannot stop a slow climber, which is
  the F32 / F33 problem. With X26's rule, **"plateau" means the same test on both architectures, in steps:** the
  newest node scores ≤ ≈ 0.51 against the node 50M back, two consecutive FLAT checks.
- **STOP** at `TIER1_PLATEAU`. The toy's plateau checkpoint is the newest node of the second FLAT check.
- **CAP** at 500M (X26's cap). A cap stop is recorded as `CAP`, never as a plateau.
- **K9(b):** as for the toys (readiness §3.4).

The rule is REGISTERED here before the run launches. Any change before the first check is an amendment with its date.

### 4.4 Cost, and when

**Throughput.** The toy runs at ≈ 4,800–5,100 fps (s4001 60M → 70M: 2,088 s for 10M; s4002 0 → 100M in 20,345 s),
slowing as its games lengthen. That is **≈ 5.8 GPU-h per 100M**.

**Where it stops (ESTIMATE).** The plateau horizon is UNKNOWN:
- the 100M cap reads +18 to +27 Elo per 20M;
- a W 50M FLAT needs the gain per 50M under ≈ +7 Elo.

**Cost:**

| stops at | GPU-h from 100M | wall, exclusive GPU |
|---|---|---|
| 200M | ≈ 6 | ≈ 6 h |
| 300M | ≈ 12 | ≈ 12 h |
| 500M (cap) | ≈ 24 | ≈ 1 day |

Plus the Tier-1 checks on CPU: about 30 min each on the toy (s4002's 16 checks took 29.5 min wall in all, at Δ 5M). At
W 50M a check runs more pairs; budget ≈ 1 h per check.

**When.** AFTER X26 (owner, 2026-10-16). X26 holds the GPU: at ≈ 1,600 fps it is ≈ 52–86 GPU-h to 300–500M. The
toy goes next on the GPU queue, ahead of any plateau-breaker, because §5 needs both plateau checkpoints:
- if X26 plateaus early, the toy follows at once;
- interleaving inside X26 is ruled out, because the GPU is leased to one run.

**The one alternative is the cloud** (§6.4): the toy's per-step work is small, so a rented consumer GPU runs it in
parallel with X26, and the calendar saves ≈ 1 day.

**One seed, not three.** The plateau LEVEL across seeds is a second-order question for the owner's plan. The three
T-ENT seeds already agree within 3 pp at every cross (+5.9 / +6.3 / +7.2; +8.1 / +11.2 / +11.3). §5's breakers
replicate by FORK seeds from the one plateau checkpoint. A second seed (s4003) is added only if R-T2-at-plateau lands
UNDECIDED and seed variance is the binding term.

## 5. PLATEAU-BREAKERS on both architectures (X35), with the toy as a testbed

### 5.1 The design

**The starting points.** P\* = X26's plateau checkpoint; T\* = the toy's (§4).

**One breaker b** (a PFSP / league change, one-ply search, a recipe change, a curriculum — X28, X29, X30 are already
in the backlog) is TWO arms per architecture:
- **b-ON:** a fork of the plateau checkpoint with the breaker;
- **b-OFF CONTROL:** a fork of the same checkpoint, same length, breaker off. At a plateau the control should read
  flat. It is the matched-noise control (`feedback_matched_noise_control`), and it absorbs whatever a fork alone does.
- 3 fork seeds per arm.

**Fork lengths:**
- production L_P = 20M steps (≈ 3.4 GPU-h per seed);
- the toy at EQUAL GPU TIME (≈ 60M toy steps ≈ 3.5 GPU-h). The toy's 20M-step node is read as a secondary,
  matched-step view.

**The readout** for each breaker, on each architecture, is Δ_b = b-ON − b-OFF:
- the closing test's `cross_stat` on a square cross, or T40's rectangular statistic once built;
- the ±2.0 pp bar;
- classes GAIN (interval wholly above +2.0), NULL (inside [−2.0, +2.0]), LOSS (wholly below −2.0), else UNDECIDED.

**What answers the owner's step 4:**
- **(a) Does the architecture change what the breaker GIVES?** Δ_b(P) − Δ_b(T), with intervals combined. Its sign
  says which architecture the breaker helps more.
- **(b) Does the architecture still matter AFTER the breaker?** A cross of P\* + b against T\* + b (the b-ON forks'
  finals), at their own plateau-plus-breaker budgets. Read beside R-T2-at-plateau (P\* vs T\*):
  - if the P − T gap SHRINKS under b, the breaker substitutes for the machinery;
  - if it holds or grows, the machinery and the breaker are complements.

### 5.2 The toy as a cheap testbed, and how we know results transfer

A toy fork costs ≈ the same GPU time as a production fork at equal time, but reaches ≈ 3× the steps. Its per-step
cost is ≈ 1/3, so screening N breakers at matched steps costs ≈ 1/3. The toy may PRE-SCREEN breakers only once
transfer is shown:

1. **The transfer CALIBRATION (registered, before any breaker read).** The first TWO architecture-agnostic breakers
   run on BOTH architectures, whatever the toy reads. Where possible, one is expected to help (e.g. one-ply search) and
   one is a recipe change with a plausible null.
2. **Agreement.** A breaker TRANSFERS when its class (GAIN / NULL / LOSS) is the same on both architectures.
   UNDECIDED on either side is "not shown"; it is never counted as agreement.
3. **Gate status:**
   - **Before calibration (fewer than 2 / 2 agree):** the toy only RANKS breakers, ordering the production queue.
     It never drops one.
   - **After 2 / 2:** a breaker the toy reads NULL or LOSS, with an interval upper end below +2.0, may be DROPPED from
     the production queue. Every third dropped breaker is still run on production as a SPOT CHECK; one disagreement
     returns the toy to "ranks only".
   - **A disagreement during calibration** means the toy does not transfer for that breaker class. That class runs on
     production only, and the disagreement itself is a result for step 4: the architecture matters there.
4. **Architecture-bound breakers skip the toy.** A breaker that READS the machinery cannot be screened on a network
   without it: search over the network's belief worlds, an op-informed leaf, belief-conditioned exploiters. One-ply
   search with the network's own policy and value as the leaf runs on both, and it is the most informative single
   cross-architecture breaker. If search lifts the toy to production's level, the machinery is substitutable by
   search at that stage. If it does not, the machinery is a complement.
5. **Seeds and pins.** Both architectures' forks run at the same commit wherever both load. The toy's pin constraints
   are in §4.2. Forks carry a declared seed list. A replicate means the same pin and seed (`feedback_replicate`).

### 5.3 The ladder of intermediate arms (X36, DEFERRED)

"The toy plus the single most valuable component" was the brief's (b). The owner's plan does not need it, so it is
DEFERRED. It becomes useful in two cases:
- §5.2 calibration fails for a class, and we want to know WHICH component makes the architectures differ;
- R-T2-at-plateau reads CEILING, and the question becomes which component carries the ceiling.

Then it runs as equal-time screens at 15M production-equivalent wall clock (≈ 2.5 GPU-h per seed), in this order,
cheapest first:
1. toy + depth 3 (+5.2 MFLOP per round-row);
2. toy + E4 / E5 / OTHER seats (+3.7);
3. toy + the 32 event seats (+9.5 per round);
4. toy + the op's cells → edges (needs the op; no OFF value of the deleted levers exists at HEAD, so it runs PINNED at
   `pre-legacy-deletion`, D-LD4).

Each is a single-lever A/B (`feedback_reasonable_richness_vs_budget`), 3 seeds × 8 references, against the toy at the
same wall clock.

### 5.4 Equal compute, done rigorously

1. **The unit is EFFECTIVE LEARN GPU-TIME**: Σ (`collect_ms` + `update_wall_s`) over the surviving step path. It is
   restart-aware: re-done work is not double-counted (the method of `time_to_step.py`). It excludes process start,
   compile, checkpoint writes and in-learn eval.
   - 🚨 R-T5 used SB3's `time_elapsed`, which INCLUDES the in-learn eval callbacks. EB and the toys both ran
     `--snapshot-ladder-games 0`, so R-T5 is fair. **X26 runs 100 ladder games per cycle**, so any toy-vs-X26 equal-time
     read must use the Σ rows (FINDING F2).
2. **FLOPs are the hardware-free second axis.** The GPU UPDATE tracks counted FLOPs at the whole-model level (4.56× vs
   4.67×, §1.2). Collect does not: the env floor is ≈ 7.8 s per iteration, ESTIMATED. So FLOP-matching flatters the
   toy's collect and wall-matching flatters nothing on this box. Report wall-matched as PRIMARY and FLOP-matched as a
   descriptor, never the reverse. A different GPU class changes the wall ratio (`cloud_gpu_benchmark.md`).
3. **The shared box.** The GPU is leased, but collect runs on the CPU, and the toy's iteration is ≈ 40 % env against
   production's ≈ 13 % (§1.1). Agent CPU load therefore slows the TOY more. Every equal-time read records load1 and
   `gpu_wait_ms_per_host_step` per window, and carries R-T5's ±1 SD time band. A read whose conclusion flips inside
   the band is UNDECIDED.
4. **A curve, not a point.** Equal time is read at ≥ 3 budgets: X33 gives ≈ 2.5 h and ≈ 5–6 h now, and the deep runs
   add more. "The toy leads at equal time" is a claim about ONE budget until then.
5. **Matched everything else.** The same precision (bf16), compile mode, recipe and eval regime. X26's ride-along heads
   (≤ 1.45 % of the update) are declared, not corrected.

### 5.5 The ranked plan (information per GPU-hour)

| rank | id | what | GPU-h | CPU | what it answers |
|---|---|---|---|---|---|
| 1 | **X33** | **The equal-time CURVE from existing checkpoints**: each toy seed at the step whose effective learn time equals X26 at its checkpoints, for X26 at ≈ 15M (≈ 2.6 h), ≈ 30M (≈ 5.6 h, against the toy's 100M cap) and every later X26 node a toy node can match. 3 × 1 cells × 500 mirrored pairs per budget | **0** | ≈ 2–3 h per budget on the agent cores | does the toy's equal-time lead survive a larger budget? It is the first "curve, not a point" read, available NOW (X26 is past 45M) |
| 2 | R-T2 | the toy cap vs X26 plateau (registered) | 0 | ≈ 3 h | G_p(cap), biased toward production |
| 3 | **X34** | the toy deep run to plateau (§4) | ≈ 6–24 | Tier-1 checks ≈ 1 h each | the toy's ceiling; R-T2 re-read at plateau; the start point for §5 |
| 4 | **X35** | breaker transfer calibration: breakers 1–2 on both, 3 fork seeds × (ON + OFF) | per breaker ≈ 20 (production) + ≈ 21 (toy at equal time) | crosses | does the architecture matter at the plateau-breaker stage; may the toy pre-screen? |
| 5 | X35 (cont.) | further breakers, toy-first once calibrated | ≈ 21 per toy screen, ≈ 20 per production confirm | crosses | the breaker ranking |
| 6 | X36 | the toy + 1 component ladder (§5.3), only on its trigger | ≈ 7.5 per arm (3 seeds × 2.5) | crosses | which component carries a measured difference |

## 6. Efficiency ideas, as cost-reducers for the experiment programme only

Nothing here changes production's architecture. Each item is a PERF or TOOLING change that makes the programme
cheaper, and each would go through its own parity gate.

| id | idea | what it saves | evidence | risk |
|---|---|---|---|---|
| **T41** | **Compiled (Inductor) T2 inference for the CPU meters** (`main.h2h` / `main.plateau` default `eager` on CPU) | Every CPU cross and Tier-1 check. On CPU, ≈ 64 % of production inference is the op's cells, op rules, glue and beliefs (51,000 small eager ops per forward), which compile fuses. X33, R-T2 and every §5 cross read production on CPU | MEASURED: CPU inference 0.966 s per 256 rows, 13× the toy's | the CPU Inductor `simdlen` bug (`compile_regions.CPU_INDUCTOR_OPTIONS`); compile time ≈ minutes; a parity gate against eager |
| **T42** | **Edge-bias block assembly**: build each seat block once and write the dense `[B, 4, T, T]` bias in ONE assembly, not 17 read-modify-write slice writes, each with a full-size backward copy | CPU: up to 24 % of a CPU update, and 38 % of its backward. GPU: unknown until T44 | MEASURED on CPU | the per-pair sum order changes, so the result is equal up to fp32 reassociation, not bit-identical; the parity gate is the compile gate's tolerance |
| **T43** | **Skip training-only readouts in T2** (BeliefHead's species / moves heads; their output is stashed for the aux loss, never fed forward) | ≈ 1.7 % of inference FLOPs; GPU T2 and CPU meters | MEASURED (CPU FLOPs) | policy and value must be bit-identical with and without; tier contract |
| **T44** | **A GPU per-component profile in a lease window**: the same `record_function` labels on the compiled update (or eager-on-GPU as a bound) | Fills §1.4's two UNMEASURED rows: the edges' bias writes and the op / belief share on the GPU | — | needs a GPU lease; ≈ 30 min |
| — | **Toy throughput is env-bound** (≈ 50 % of its iteration is collect, ≈ 40 % Rust env) | For X34 and the toy side of X35, schedule toy GPU windows when agent CPU load is low (rule 5b's 3 cores still bind) | MEASURED (§1.1) | none; scheduling only |
| — | per-episode caching of the static S half of each mon token, or of the dex-table hypothesis encode | small: static tokens are 6 % of FLOPs, and the hypothesis encode is already once per forward | ESTIMATED | not worth a build now |

**Rejected here (owner framing):** a shorter event window, fewer seats, depth 2 or a lighter op AS PRODUCTION CHANGES.
They are recorded in §3 as cost / value gaps. If a breaker result ever needs attribution, X36 is where they get
measured.

### 6.4 Owner question (strategic only)

1. **The toy deep run's GPU: after X26 on ours (the default, ≈ 6–24 GPU-h at the queue's tail), or now on a rented
   cloud GPU in parallel with X26?**
   - The cloud saves ≈ 1 day of calendar time before §5 can start, at the cost of a rental. The toy is small, so a
     consumer card runs it at ≈ the local rate; the vast.ai kit and the cu126 / cu128 caveat are in
     [`../ops/cloud_gpu_benchmark.md`](../ops/cloud_gpu_benchmark.md).
   - It also mixes two GPU types into the toy's own time curve. X33 / R-T5-style equal-time reads would then need that
     segment excluded, or re-timed against a local calibration.

Everything else here is the orchestrator's tactical call (GPU order, seeds, the breaker queue).

## FINDINGS (standing rule 7)

- **F1. The 2.61M vs 1.42M parameter contrast is mostly the POLICY TOWER.** The tower is 1.118M of production and
  0.723M of the toy. Both keep `--policy-readout tower`, so the toy is not "half the network"; it is ≈ 0.70M of
  non-tower parameters against ≈ 1.49M. FLOPs (4.6×) are the right size ratio, not parameters.
- **F2. Equal-time reads against X26 must not use SB3's `time_elapsed`.** It includes the in-learn eval callbacks, and
  X26 runs 100 snapshot-ladder games per cycle where EB and the toys ran 0. Use Σ (`collect_ms` + `update_wall_s`), as
  in §5.4.
- **F3. ESTIMATED, not measured: the env and inference split of collect.** It assumes equal host steps per collect and
  equal env time (§1.1). A direct read needs the collect phase's own timers. `gpu_wait_ms_per_host_step` exists, but
  no env-time row does.
- **F4. X26's collect is 23.5 s against EB's 17.2 s** (GPU wait 29.5 vs 18.2 ms per host step) at the same update
  time. The ride-along heads are update-only. The cause is UNVERIFIED: game length, agent load or the ladder games.
- **F5. The 95.7 MFLOP / row production figure** (readiness §3.2) is the FORWARD count (measured here 94.8 at 2,048
  rows and 88.8–90.6 at T2's 64 / 256). Forward + backward is 244.6. The toy's 17.4 is likewise forward (measured
  17.5). The readiness doc gives no provenance for either. Both are now measured here.
- **F6. The 2026-10-10 per-module FLOP read (`perf_phase_2026-10-10/results/module_flops_endstate.json`) is
  mis-attributed.** `FlopCounterMode`'s module tracker double-counts across backward. Its rows sum far above 100 % (for
  example, each of 15 edge maps is listed at 7.6 %, while the maps' true share is ≈ 0 % of FLOPs). This doc's
  `sequence_nr` attribution supersedes it for per-component shares. The totals agree.
- **F7. Profiled CPU time over-weights many-small-op code.** The op's kernels issue ≈ 51,000 eager ops per forward.
  The CPU time shares in §1.2 are CPU facts, not GPU estimates.
- **F8. The third trunk round, the largest measured GPU delta of the end state (+6.9 s, 80 % of its update cost over
  legacy), has no isolated value read, and no ledger entry carries the +6.9 s.** It is only in
  `perf_phase_2026-10-10/README.md`. UNDERSTANDING §4.6's "the proposed next lever is a third trunk round … as a
  separate screen" was stale (the round was adopted inside the bundle); it is corrected in this commit.
- **F9. The closing test's bisection can no longer run at HEAD.** U0 / U1 / U3 deleted the OFF paths, and U3 moved the
  fresh-init stream. Any isolation of an adopted lever (X36's op arm) runs PINNED at `pre-legacy-deletion`.
- **F10. The T-ENT pin predates T36** (readiness F25). §4.2 makes the continuation's canary a launch requirement.
- **F11. The accidental simple control `ai_v12_01`** (24.4M steps; edges, seats, belief slots and history off; op ON)
  was never read (audit F23). Whether its final model still loads after the pre-Rustboro skeleton was not checked
  here.
- **F12. Measured on one box, one commit, one thread.** `component_cost.py` ran at `38d76746` beside a live X26
  (rule 5b: 3 cores, nice 19). Absolute CPU seconds carry that contention. Shares and FLOPs do not.

## 7. Decision record

| date | decision | chosen | rejected | evidence |
|---|---|---|---|---|
| 2026-10-16 | The doc's frame | **planning input** for the owner's plan (production to plateau → toy to plateau → plateau-breakers on both → does the architecture matter there) | a "replace or slim production" recommendation (owner, 2026-10-16: the toy's equal-time win is a data point, not a concern; production wins per step; the ceiling and breaker questions are open) | the owner's framing via the orchestrator, 2026-10-16 |
| 2026-10-16 | How to attribute cost | CPU-measured FLOPs and time per component by autograd `sequence_nr`, plus the GPU numbers already measured, each tagged MEASURED / ESTIMATED / UNMEASURED | `FlopCounterMode` module attribution (double-counts the backward, F6); parameters as the size measure (F1) | §1, `component_cost.py` |
| 2026-10-16 | The toy's stop rule from 100M | **X26's own Tier-1 rule** (Δ 10M, W 50M), cap 500M, one seed (s4002), after X26 | the toy's Δ 5M / W 20M rule (cannot stop a slow climber: readiness §3.8, F32 / F33); three seeds (the seeds agree within 3 pp; breakers replicate by fork seeds) | §4 |
| 2026-10-16 | Plateau-breakers | forks from BOTH plateau checkpoints with an OFF control fork each; a transfer CALIBRATION (2 breakers on both) before the toy may gate; architecture-bound breakers skip the toy | screening on the toy alone with no calibration; breakers judged without a control fork | §5 |
| 2026-10-16 | Equal compute | effective learn GPU-time (Σ collect + update) as PRIMARY, FLOPs as a descriptor, ≥ 3 budgets, load recorded | FLOP-matching as primary (collect is not FLOP-bound); `time_elapsed` against X26 (F2) | §5.4 |
| 2026-10-16 | The toy + 1 ladder | DEFERRED with two named triggers | running it now (the owner's plan does not need attribution yet) | §5.3 |
| 2026-10-17 | How X34 carries T36's canary fix | **a FORK at main** (`--pin-commit` main; the minimal argv inherits the arch and has the recipe restored), after network identity was PROVEN: the same keys and shapes, a strict load, and a bit-identical fp32 forward at HEAD vs the pin | resuming at the pin `3899277a` (the pre-T36 canary lottery for 400M steps); typing the toy's arch flags on the fork (resolves to the same config; more surface); a fork without `--pin-commit` (it pins the PARENT's commit, F1) | `x34_toy_deep_stage_2026-10-17/`; `toy_network_identity_test.py` |
