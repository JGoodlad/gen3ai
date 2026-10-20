# What the end state's compute buys: the cost / value ledger, the toy to its plateau, and plateau-breakers on both architectures

**Status: DESIGN 2026-10-16 (planning input; nothing built, nothing launched). §5.6, the X35 plan, is STAGED ON PAPER 2026-10-17: arms ranked, readout pre-registered, the first arm's launch block dry-run validated, nothing launched.** ALWAYS-CURRENT (owner, 2026-09-27):
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
     Launch-ready on paper since 2026-10-17 (§5.6): seven arms ranked, the archive opponent pool (A1) first, the
     readout scored against the POPULATION with the collapse battery and a Kakuna read as guards.
  5. **X36 — the toy + 1 component ladder.** DEFERRED.
- **Against the published bots (§1.6, 2026-10-18):** the ~75× steps/s gap to mikumiku37 is the card (×1.75), 10 PPO
  epochs against their ≈ 1–2, low GPU utilisation of a many-small-op network, and opponent inference fanned over ~21
  small replays per host step. It is NOT model size and NOT the env core (7.6 % of the cycle). The levers are ranked
  there: fewer epochs (X44), the fan-out cut (X45), overlap (X14).
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

### 1.6 Against the published bots: the throughput audit (2026-10-18)

The owner asked what separates us from the Gen 9 VGC bots of October 2026 (mikumiku37: ~100k env steps/s on one RTX
5090; X26: ~1,350 on our 3080 Ti). The full audit, its arithmetic and tomorrow's GPU cells are in
[`../research_state/measurements/throughput_audit_2026-10-18/README.md`](../research_state/measurements/throughput_audit_2026-10-18/README.md).
In short:

- **The card is ×1.75.** On the same card, the gap is two stacked gaps.
- **The UPDATE costs ≈ 28× theirs per sample.** Mostly that is **10 PPO epochs**, where their published budget fits only
  ≈ 1–2. The rest is our GPU utilisation: 6–8 % of peak on the counted arithmetic, because the op, edges and beliefs
  are many small memory-bound kernels (§1.4). It is not model size: they have 3.3× our parameters and a forward
  probably 1–3× ours.
- **COLLECTION + overheads cost ≈ 75× theirs per sample.** About half is inference fanned over ~21 small replays per
  host step (the trainee + up to 20 opponent snapshots): 88 µs of GPU wait per row against ≈ 14 µs for the same
  forward inside the learner.
- **The env core is only 7.6 % of X26's cycle.** Its per-decision time is ≈ 90 % protocol-text round trip, allocation
  and encoding, and ≈ 2–3 % simulation (self time, BUSY box).
- **Their figures are not one unit** (330M games / 48.3 h and 100k steps/s imply 52.7 steps per game). So the true
  per-decision gap is ≈ 15–75×.
- **Levers, ranked by expected gain × risk:**
  1. fewer epochs at a held optimizer-step count: ×1.39 (E5) to ×1.81 (E2); risk = sample efficiency (**X44**);
  2. cut the opponent fan-out: est. ×1.08–1.20, low risk (**X45**, sized by tomorrow's `c_pool*` cells);
  3. overlap collection with the update: est. ×1.15–1.3, more after 1 (**X14**). **BUILT 2026-10-19, default OFF,
     not yet measured on the GPU** (`--overlap-collect`; `designs/training/rust_collector.md` "Overlap (X14)"): the
     next rollout plays on a worker thread during the update; the update runs DECOUPLED PPO, because X26's own
     one-version-old rows start an update at KL 0.016 vs μ and 22.5 % outside the clip band, and plain PPO would let
     the KL→LR controller cut the rate for the lag. Added GPU memory ≈ 3 policy copies (134 MiB at X26's policy with
     its ride-along heads, 36 MiB for the toy), against peaks of 8,304 / 1,880 MiB. The throughput A/B command is
     written (≈ 1 GPU-h): [`../research_state/measurements/x14_overlap_2026-10-19/`](../research_state/measurements/x14_overlap_2026-10-19/README.md).
     The toy (X34) is the more exposed: its update (9.2 s) and collection (10.5 s) are near-equal, so overlap could
     hide up to ≈ 47 % of its cycle, against ≤ 41 % of production's (ESTIMATED; GPU contention takes some of it,
     and the decoupled update's π_prox pass — one T2 forward over the buffer, ESTIMATED 2–5 s on X26 — adds to
     the exposed half).

  Together on our card ≈ ×2.4–3. These are RECIPE and PERF levers. Nothing here slims production (the owner's
  2026-10-16 framing, §6); the residual ≈ 15× is the architecture's per-row GPU cost, which X36 would attribute if
  ever needed.
- **The fan-out's ceiling, MEASURED (the 2026-10-19 GPU gap, `~/gen3ai_archive/gpu_gap_2026-10-19/`):** collection at
  256 envs ran **5,453 env steps/s with a 5-snapshot pool against 3,507 with 20** (GPU wait 18.1 vs 41.4 ms per host
  step; `cells/phases/c_pool5.json`, `suite_3080ti/phases/collect.json`). A 1-snapshot pool was SLOWER than 5 (4,719;
  24.5 ms): all opponent rows on one slot replay as serial 64-row chunks on one lane. So the lever is "one forward for
  many slots", not "fewer slots".
- **X45 (b), the GROUPED FORWARD — BUILT 2026-10-19 behind `--t2-grouped-opponents` (default OFF), GPU effect
  UNMEASURED.** Every opponent slot of the trainee's slot group is served by ONE `torch.func.vmap` forward over the
  group's stacked weights per host step (`src/agents/inference/service/grouped.py`; the program doc's T2 DESIGN
  "GROUPED SLOTS"); the pool size and the opponent mix are unchanged. The 2026-09-29 blocker (the extractor's in-place
  writes into fresh tensors; 34 sites in 15 files today) is handled inside the vmapped body, the model untouched.
  **Parity:** fp32 reassociation, not bit-identity — CPU max|Δlogp| ≤ 9.6e-7, max|ΔV| ≤ 6.3e-7, greedy identical on
  every decisive row; every grouped slot is gated against its own eager reference at startup and on every load
  (`grouped_test.py` pins it at 1e-5 and fails on a weight mix-up or a revert). **CPU read** (eager, busy box: load
  ≈ 23 on 16 threads, so WARN-only; buckets 4 / 16, opponents only, 212 rows a step, `grouped_benchmark.py`):
  pool 5 — per-slot 6,118 ms vs grouped 5,625 ms per host step (block ratios 0.65–2.19, median 1.10);
  pool 20 — per-slot 5,241 ms vs grouped 3,179 ms (every block below 1: 0.55–0.84, median 0.61). Startup is slower
  grouped (237 s vs 56 s on CPU): every per-slot parity gate of a grouped slot runs the whole range (on the GPU, one
  replay each). The CPU eager path is dominated by Python dispatch (the factory mode intercepts every op), so it says
  nothing about the GPU, where the grouped forward is compiled and captured. **Costs to watch on the GPU:**
  an idle declared slot computes `b` pad rows per round (production: 24 declared × b, against the per-slot path's
  Σ bucket(rows) = ~20 × 64 at the default buckets 8 / 64 / 256 — so finer buckets 16 / 32 matter); three ops have no
  vmap batching rule and fall back to a per-slot loop (`scatter_`, `index_copy_`, CPU flash SDPA — whether CUDA's
  SDPA does is UNVERIFIED); the range's graph pool holds `S × b` rows of activations (24 × 64 = 1,536 rows, ≈ 6× the
  trainee's 256-row pool — memory UNVERIFIED on the 12 GB card); Inductor's lowering of the vmapped forward is
  UNVERIFIED (CPU `aot_eager` traces it `fullgraph`; CPU Inductor did not finish in 15 / 45 min at nice 19).
  **The GPU read (next window; needs the lease, root CLAUDE.md rule 5; ≈ 0.5 GPU-h):**

  ```bash
  # in a worktree at the X45 commit or later; OUT a new dir under ~/gen3ai_archive/
  export PYTHONPATH=$PWD/src OUT=~/gen3ai_archive/x45_grouped_$(date +%F) && mkdir -p "$OUT"
  scripts/ops/gpu_lease.sh acquire --owner <agent> --token-file "$OUT/.lease"
  GEN3AI_GPU_LEASE_TOKEN_FILE="$OUT/.lease" GEN3AI_TEST_ALLOW_GPU=1 scripts/ops/gpu_lock.sh \
      python3 -m pytest src/agents/inference/service/grouped_cuda_test.py -q 2>&1 | tee "$OUT/cuda_test.log"
  for P in 20 5; do for B in 8,64,256 8,16,32,64,256; do
    GEN3AI_GPU_LEASE_TOKEN_FILE="$OUT/.lease" scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 24 \
      python3 src/agents/inference/service/grouped_benchmark.py --opp-slots $P --buckets $B \
      --json "$OUT/pool${P}_b${B//,/-}.json" 2>&1 | tee "$OUT/pool${P}_b${B//,/-}.log"
  done; done
  GEN3AI_GPU_LEASE_TOKEN_FILE="$OUT/.lease" scripts/ops/gpu_lease.sh release
  ```

  ADOPT (the flag's default ON, a separate unit) if pool 20's ON / OFF step-median ratio is below 1 in every block at
  the chosen buckets, the CUDA test passes, and the trainer's startup fit check still passes at production's shape;
  then the 10-19 `collect` cell re-run with `--t2-grouped-opponents` in its trainer argv confirms the env steps/s.

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
- production L_P = 20M steps (≈ 4.4 GPU-h per seed at X26's MEASURED late-run rate, §5.6.1; the first draft's 3.4
  assumed ≈ 1,600 fps, F13);
- the toy at EQUAL GPU TIME (≈ 70M toy steps ≈ 4.2 GPU-h, §5.6.1). The toy's 20M-step node is read as a secondary,
  matched-step view.

**The readout** for each breaker, on each architecture, is Δ_b = b-ON − b-OFF. **AMENDED 2026-10-17 (§5.6.4):** Δ_b's
PRIMARY is scored against a frozen POPULATION of opponents, per `design_league_decisions.md` §0; the b-ON × b-OFF
head-to-head below becomes a guard (G-F) and the sensitive descriptor. The bar and the classes are unchanged:
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
| 4 | **X35** | breaker transfer calibration: breakers 1–2 on both, 3 fork seeds × (ON + OFF) | per breaker ≈ 14.6 (production: 3 seeds ≈ 13 + guards ≈ 1.4) + ≈ 14.6 (toy at equal time); the shared control ≈ 13 + 13 once per batch (§5.6) | crosses | does the architecture matter at the plateau-breaker stage; may the toy pre-screen? |
| 5 | X35 (cont.) | further breakers, toy-first once calibrated | ≈ 14.6 per toy screen, ≈ 14.6 per production confirm (§5.6) | crosses | the breaker ranking |
| 6 | X36 | the toy + 1 component ladder (§5.3), only on its trigger | ≈ 7.5 per arm (3 seeds × 2.5) | crosses | which component carries a measured difference |

**The published bots' architecture and recipe ideas (2026-10-19)** are costed as X35-frame arms in
[`design_bot_ideas_screen.md`](design_bot_ideas_screen.md):
- **X47**, toy depth 2 → 4 as a growth fork at T\*, joins the TOY batch after the calibration pair. It shares the toy
  C0, costs ≈ 17 GPU-h and needs the build B-GROW before the batch's pin. It is the first new candidate after X34
  plateaus.
- **X48** (per-kind feed-forward experts), **X47b** (production depth 3 → 4, conditional) and **X49** (intent
  shaping) are production arms ranked after A3 / A4.
- **X50** (human data) is a zero-GPU scoping read first.
- **Set mutation is X41.**
- X47 supersedes X36's rung 1 for the PLATEAU question. X36 keeps CEILING attribution.

### 5.6 X35 LAUNCH-READY ON PAPER: the arm list, the readout, the builds (2026-10-17)

**Status: STAGED ON PAPER, nothing launched; batch 1 + X47 has a LAUNCH SHEET (2026-10-19, [`BATCH1_LAUNCH_SHEET.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/BATCH1_LAUNCH_SHEET.md): the arms in GPU order, every argv dry-run validated, the T62 / FP32_ALT pin variants, C1–C4 ruled, and the review's amendments AM-1 to AM-7 ruled and built, 2026-10-19: the AMENDMENT below).** The day the plateau is called, the first arm can launch once the
orchestrator gives its GO; a design block is not a launch command. The staging record, with every dry-run transcript,
is [`../research_state/measurements/x35_breakers_stage_2026-10-17/STAGED_PLAN.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/STAGED_PLAN.md).
**Why now:** X26 is still CLIMBING at 180M (Tier 1 GAIN at every 10M check). The owner's order (2026-10-16) is X26 to
its plateau, then the toy (X34) to its plateau, then breakers on both. No PFSP or league lever goes inside X26 itself.

**AMENDMENT 2026-10-19 (the orchestrator's rulings on the adversarial review [`X35_BATCH1_REVIEW.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/X35_BATCH1_REVIEW.md), made BEFORE any X35 launch; the full table is the launch sheet's §0, each item BUILT in `src/main/x35/` and enforced by `launch_x35.sh`):**
- **AM-1 (blocking): ONE pin.** X35_PIN_1 is written ONCE (`x35_pin_batch1.txt`, by the first GO: sha, date, T62
  verdict); every launch's pin and every batch-1 run's recorded pin must equal it (pre-check 2b).
- **AM-2: the variance gate G1** after the first six forks: the pooled within-arm SD of S over A1 / C0 × s3501–s3503
  (4 df); PASS iff ≤ 1.2 pp (within 1e-9 counts above); a HOLD holds D1, A2 and the toy calibration for a re-plan
  and runs C0 s3504–s3505 next (`kit gate`). It reads variance only.
- **AM-3:** toy A1 / A2 launch only after production A1 / A2 has a FINAL class GAIN, NULL or LOSS; an UNDECIDED
  production read ends that calibration as NOT SHOWN (§5.2's rule cannot record agreement otherwise).
- **AM-4:** X47 registers look 2 (seeds 3604–3605 against toy C0 3604–3605, on a look-1 CONTINUE); its readout
  `main.x35.x47` (s_eq, the matched-step secondary, PER-STEP ONLY) is built.
- **AM-5:** D1's H interval is widened by √max(1, φ̂), the seed overdispersion (`design_diversity_teams_opponents.md`).
- **AM-6:** K9(b) is read at EACH arm's end over its own post-fork rows, deterministically (`kit k9b`); a crossing makes
  the arm VOID for every read.
- **AM-7: the order** is the first six forks → G1 → C0's look-2 seeds and A1's look 2 → D1, A2 → toy C0 + X47 → X47's
  look 2 → the conditional toy calibration (≈ 91 GPU-h on the expected path, ≈ 146–148 with every conditional stage).
  It REPLACES the batch-1 GPU order below.
- **POOL:** every arm of a side seeds the SAME snapshot set, recorded at the side's first launch and checked at every
  launch and after every spawn. **C0 is "P\* continued with X26's pool as of today: 14 snapshots, 164–268M", not
  "X26 unchanged"**: the pool is older and sparser than X26's at P\* (review F-RV-8), matched across arms.
- **G-E readers:** not launched in batch 1. An arm whose primary reads GAIN stays INCONCLUSIVE (never
  ADOPT-CANDIDATE) until its G-E readers (≈ 2.2 GPU-h per arm, argv still unregistered: F27) are registered and run
  at X35_PIN_1.
- **C1: (a).** Production launches at main HEAD with T62, pending the GPU battery's read.

#### 5.6.1 The shared frame (every arm)

- **The start.** P\* is X26's plateau checkpoint: the newest node of the second FLAT Tier-1 check (§4.3's convention),
  a 10M-multiple `checkpoints/checkpoint_<N>_steps.zip`, sha256 frozen at the call. T\* is the same for X34.
- **One pin per BATCH.** `X35_PIN_k` is one main commit that carries every build its arms need, T54 at least (§5.6.7).
  A control is valid only at its own pin (`feedback_replicate_means_pin_and_seed`). So an arm whose build lands after a
  batch's pin forms a later batch with its OWN control. Dispatching the builds now is what keeps batch 1 large.
- **The recipe is X26's own record, not `recipe.fork`.** The launcher restores X26's 12 recipe knobs typed (10 epochs,
  the KL → LR controller continuing from the inherited rate). `recipe.fork` (5 epochs, a frozen 5.6e-5) would be a
  second lever, the X27 / X28 question. Each arm records its dose (`main.dose`). A dose gap over 10 % between b-ON and
  b-OFF is reported as a confound, the threshold `main.best_response_gap` uses.
- **Typed on every production arm: `--behaviour-check warn`**, X26's Amendment 3 regime, with its fixed GIGO ceilings
  read by hand (`design_x26_readiness.md` Decision record, 2026-10-17). Untyped, a fork defaults to `fatal`
  (`main/train/rust_env_setup.py`). Amendment 3 predicts ≈ 110 rounding stops by 500M, so `fatal` would mean about 6
  stops per 20M arm (F14). The toy arms type `warn` too (the orchestrator's ruling C3, 2026-10-19; the kit had typed
  `fatal` since 2026-10-17): X34 itself has run `warn` with Amendment-3-style ceilings since its launch decision
  (STAGED_LAUNCH §7), and its own `behaviour/*` scalars never reached the 1e-4 bar in 2,160 updates to 212.4M (judged
  max 7.7e-5). The ceilings are read by hand; nothing enforces them (launch sheet F-LS-9).
- **Lengths.** Production L_P = 20M steps. The toy L_T = 70M, at equal effective learn time (§5.4): 4.2 GPU-h at
  ≈ 5.8 GPU-h per 100M is ≈ 72M. The realised Σ (`collect_ms` + `update_wall_s`) is recorded. A mismatch beyond ±15 %
  is reported, with the toy's +20M node read as the matched-step secondary.
- **Seeds.** Production 3501–3503 (look 1) and 3504–3505 (look 2); toy 3601–3603 (look 1) and 3604–3605 (look 2;
  X47's registered by AM-4). b-ON and b-OFF share seed i.
  Exploiters use 3511–3513, league exploiters 3521–3523 and best-response readers 3531+.
- **ONE shared control set, C0, per (start checkpoint, pin).** Every breaker in the batch is read against it.
- **The GPU order (batch 1): AM-7's (the amendment above; `kit.BATCH1`).** It opens with A1 s3501, C0 s3501, C0
  s3502, A1 s3502, A1 s3503, C0 s3503, an ABBA order, so drift in box load does not land on one side. The first
  A1-vs-C0 read lands after ≈ 26 GPU-h.
- **Throughput (MEASURED, X26's checkpoint mtimes):** 150,000,100 → 180,000,055 took 22,483 s, i.e. 7,494 s per 10M
  ≈ 1,334 steps/s of wall time, eval cycles included. So a 20M fork is ≈ 4.2 GPU-h, plus ≈ 0.2 h for start and compile:
  ≈ 4.4 GPU-h per seed and ≈ 13 GPU-h per arm of 3 seeds. The first draft's 3.4 GPU-h per seed is superseded (F13).
- **Eval is inherited:** every 2M steps, 9 bots and 5 greedy sentinels at 100 games each, plus
  `--snapshot-ladder-games 100` per promotion. An arm that adds evaluated opponents (A1: +8, A2: +5 sentinels) pays
  ≈ 2 % more wall time at equal steps. That is declared, not corrected.

#### 5.6.2 The ranked arm list

Rank is expected information per GPU-hour toward the plateau mechanism X26 actually shows. The collapse pilot
(`design_collapse_and_counterfactuals.md` §3–§5) finds no entropy collapse and flat starvation. It finds a slow
transitive climb with a small, real cycle: Hodge excess width 32 Elo, p = 0.005, with significant 3-cycles through
66M, 72M, 90M, 108M, 138M and 160M. Built arms rank above arms that need a build of equal promise. GPU-h is
production, look 1 (3 seeds), plus the guards (one best-response reader ≈ 1.1, the population read ≈ 0.3 in a lease
window). The toy side costs about the same again for an arm in the calibration pair.

| rank | arm | in plain words | the ONE lever | mechanism it targets | GPU-h (prod.) | built? | decision rule |
|---|---|---|---|---|---|---|---|
| 1 | **A1 ARCHIVE POOL** (fictitious self-play over the whole past) | half of the self-play games are played against 8 frozen selves spread over the whole run, not only the last 20 snapshots | `--stable-opponents <8 archive nodes> --stable-opponent-selfplay-share 0.5` | COVERAGE of forgotten strategies and CYCLING: the measured 3-cycles run through nodes the 20-deep window (80–176M today) no longer holds. It is league §B's "grow the pool" and the uniform (Borda) form of "train vs the archive's Nash mixture" | ≈ 13 + 1.4 | **YES** (flags; T54 recommended before the pin) | R1 (§5.6.4) |
| 2 | **A2 PFSP over the snapshot pool** | the recent selves the trainee loses to are drawn more often | `--pfsp-scale 2.5 --n-sentinels 10` (the extra sentinels feed PFSP fresh win rates; without PFSP they change no training) | pool-relative WEAK SPOTS, the treadmill | ≈ 13 + 1.4 | **YES** | R1. Prior: plausibly NULL. Pure PFSP over a homogeneous fresh self-pool was CONVICTED at −26 to −33 Elo in the pre-era gen-17 attribution (F15). That makes it the calibration pair's "plausible null" (§5.2) |
| 3 | **A3 MAIN EXPLOITERS, folded in** (the population loop) | train 3 exploiters to beat P\*, then continue P\* with them taking 40 % of its challenge games | the exploiter slice: `--stable-opponents mx1,mx2,mx3 --stable-opponent-selfplay-share 0.4 --stable-opponent-pfsp` (the registered round-1 B recipe) | EXPLOITABILITY, the holes a best responder finds (league §0(b)) | ≈ 3 × 1.1 + 13 + 1.4 ≈ 18 | **YES** (flags; a two-phase kit, T55) | R1, or the co-primary R2 (the gap falls) |
| 4 | **A4 TRAILING-MAGNET KL** (R-NaD / MMD style) | a pull toward a recent copy of itself, refreshed every eval cycle, beside the unchanged entropy bonus | `β · KL(π ‖ π_mag)`, β 0.05 (the entropy bonus's weight), the magnet refreshed every 2M steps | oscillation (the treadmill) and exploitable tie-breaks (collapse type a1): a trailing magnet converges toward equilibrium in two-player zero-sum games (`design_learner_recipe.md` §3.9; UNVERIFIED here) | ≈ 13 × 1.03 + 1.4 | **NO**: T56, ≈ 2 agent-days | R1, or R2 |
| 5 | **A5 NASH-MIXTURE ARCHIVE** | A1, with each archive node weighted by the archive's max-entropy Nash (the posterior mean) instead of uniformly | the weights | CYCLING (league §0(a)'s lever, literally) | ≈ 13 + 1.4, plus the archive matrix (≈ 45k games: ≈ 0.3 GPU-h or ≈ 11 CPU-h) | **NO**: T57, ≈ 2 agent-days | CONDITIONAL: runs only if A1 is not LOSS. R1 against C0, and Δ against A1 reported |
| 6 | **A6 LEAGUE EXPLOITERS** | 3 exploiters trained against the whole league (P\* plus the archive, PFSP-weighted), folded in as A3 is | the league-exploiter slice | holes the WHOLE league shares, which an exploiter of P\* alone may not find | ≈ 18 | **YES** (flags; a two-phase kit, T55) | CONDITIONAL on A3 reading ADOPT-CANDIDATE, or on gap(P\*) ≥ 0.15 at R0. R1, or R2 |
| 7 | **A7 ENTROPY / ANTI-STARVATION** | — | — | — | — | — | **NOT RUN.** The collapse doc finds no entropy collapse and flat starvation. No deployable counterfactual knows which starved move is good (its §0, §5, F-CC-2). TRIGGERS: B1 shows an entropy fall on C0, B2 shows a starvation rise on C0, or R0 shows exploitable tie-breaks. Then X23 is run, re-scoped |
| — | **C0 CONTROL**, plain continuation | P\* trained on, unchanged | none | the noise floor at the plateau (league G5) | ≈ 13 per batch | **YES** | — |

Plus **R0**, at the plateau call and before any arm: a fresh best responder against P\* itself (≈ 1.1 GPU-h). It gives
the plateau's exploitability baseline and A6's trigger.

**ADOPTED 2026-10-18 by the orchestrator (owner question Q5), CONDITIONAL on X37:** a diversity arm **D1** ranks 2nd,
after A1 and ahead of A2. The table above is unchanged until X37 reads. D1 widens the team pool on both sides with
HP-repaired `hl_05_26` teams, minus every frozen panel. It is read against C0 with its OWN primary: the pooled
UNFAMILIAR cells vs Kakuna (heldout72_hpfix, smogon_forum, P72). Its co-primary is no loss worse than −2.0 pp on
SAMPLE-TEAM ground, measured in-house. The design is [`design_diversity_teams_opponents.md`](design_diversity_teams_opponents.md)
§4.2 and §6. Every arm above draws POOL teams, and R1 and G-K read pool teams only (that doc's F-DV-1).
**D1 is LAUNCH-READY on paper (2026-10-19, NEEDS GO):** its corpus is the Smogon FORUM corpus `w_forum_v1` (2,424
teams; it replaced `hl_05_26`), its lever `--team-mix w_forum_v1@ABC:0.5 --team-mix-sides both` (`c1b0c1f0`), its
argv `main.x35.kit` arm `d1` (BATCH1 after the A1 / C0 ABBA), dry-run validated on the real P\* at pin `c1b0c1f0`
(STAGED_PLAN §7), and its readout `main.x35.d1` (self-check 10 / 10). Every batch pin must now carry `c1b0c1f0`
(`x35_required_commits.txt`) while D1 is in the batch. G-K is unchanged; the unfamiliar panels run on the same Kakuna
runner (`--teamsets unfamiliar`) for D1's primary H.

**The transfer calibration (§5.2) is A1 + A2.** Both are architecture-agnostic and both are built, so calibration
starts on the first day. A1 is the one expected to help and A2 the plausible null. One-ply search, §5.2's example,
belongs to the search programme (`search_ledger.md`). It takes a calibration slot when its harness is registered, and
is not ranked here. X28, X29 and X30 stay on their own backlog rows.

#### 5.6.3 Each arm's fork design (b-ON vs b-OFF from P\*)

Every arm is `--model P* --behaviour-check warn --steps P*+20M` on `X35_PIN_k`. b-OFF is C0, the same argv with no
lever.
- **A1.** The 8 archive nodes are N_k for k = 1..8: the 10M-multiple checkpoint nearest k·(P\* − 20M)/8, ties
  down. For P\* = 180M that is 20, 40, …, 160M. Each is passed as an explicit `.zip` with a label `:x26_<step>m`.
  - A `run@step` spelling would label every node `ext_rb_x26_s2601`. The explicit form resolves to 8 unique labels,
    arch-gated (verified offline, STAGED_PLAN §2).
  - Nodes the trainee masters (≥ 0.80 for 2 cycles) move to the bot FLOOR by design. That is part of the arm's
    mechanism, declared rather than prevented.
  - The toy side uses the same rule on T\*'s own archive (`rb_toy_ent_s4002` + `_deep`).
- **A2.** The pool is the auto-seeded parent pool (20 snapshots). PFSP starts cold: every factor is 1 until the first
  eval cycle measures a win rate.
- **A3.**
  - Phase 1: three exploiters `--model P* --exploiter P* --no-self-play`, 5M steps each, the full trainee pool
    (unpinned: a main exploiter plays the whole game). `--no-self-play` is required (F16).
  - Phase 2: the 3 seed continuations carry the exploiters' finals as stable opponents.
  - Our exploiters target a FROZEN P\*. AlphaStar's main exploiters track the moving main agent (UNVERIFIED, from
    memory). A co-trained exploiter would need two learners on one GPU, which is not designed (F24).
- **A4.** The magnet starts at P\* and is refreshed to the trainee's own frozen eval weights each cycle (2M).
  - A STATIC anchor (the magnet held at P\*) is the degenerate variant. It runs only if A4 reads ADOPT-CANDIDATE, as
    the attribution follow-up.
  - A Smogon-prior behaviour-cloned magnet (§3.9's second candidate) needs a BC policy that does not exist. It is
    deferred.
- **A5.** A1's 8 nodes, weighted by the posterior-mean max-entropy Nash (`design_evaluation.md` §2.3 candidate (i)).
  The weights are solved once, at fork time, on a matrix of the 8 nodes plus P\* at 500 mirrored pairs per edge, and
  frozen.
  - On a near-transitive pool the point Nash is near-pure (F-ED-3), and A5 can then degenerate into "self-play vs
    P\*". The posterior mean spreads the weight. A support of fewer than 3 nodes is reported, and A5 is not run.
- **A6.**
  - Phase 1: three league exploiters `--model P* --stable-opponents P*,<A1's 8 nodes> --stable-opponent-selfplay-share
    1.0 --stable-opponent-pfsp`, 5M steps each.
  - Phase 2: the same as A3, with these exploiters' finals.

#### 5.6.4 The readout and its PRE-REGISTERED decision rule

**The reference population R** is frozen at the plateau call, before any arm runs. It is written to the staging
directory with each file's sha256, by `python -m main.x35.refpop freeze` (T55; write-once, `verify` re-hashes every
file). Two choices the rule left open are declared there (F25): a midpoint that is not a 10M multiple rounds to the
nearest one, ties down, and an A-block node that lands on an archive node is a typed refusal.
- **A-block, our own archive:** P\*, P\* − 10M, and the midpoints between A1's adjacent pairs (N_2 | N_3, N_4 | N_5,
  N_6 | N_7). For P\* = 180M that is 180, 170, 130, 90 and 50M. It is disjoint by FILE from A1's training nodes but
  ADJACENT to them, so the A-block read leans toward A1 by construction (F20). The X-block, G-F and G-K are the
  unbiased checks.
- **X-block, another lineage that no arm trains against:** T\* (a cross-architecture cell, which `main.h2h play-many`
  supports) and the registry baseline `production`. Each is kept iff P\*'s score against it, at 500 mirrored pairs,
  lies in [0.20, 0.80]. That is Dota's 15–85 % band (`design_evaluation.md` §0a), with a rule-8 margin: a score within
  1e-9 of either edge is EXCLUDED, and a candidate without an OK 500-pair cell makes the freeze refuse.
- The toy side mirrors this: T\*'s own archive, with P\* in its X-block.

**The population score** S(final) is the equal-weight mean, over R's members, of the mirrored-pair score: 500 pairs
(1,000 games) per member on `main.h2h play-many`, greedy vs greedy with symmetric builders.

**R1, the PRIMARY.**
- **The statistic.** Δ_pop = mean over b-ON seeds of S − mean over b-OFF seeds of S, in pp. Seed finals are the units;
  the interval is Welch's two-sample t with Satterthwaite df, two-sided at 97.5 % per look (two looks, Bonferroni).
- **Classes**, at §5.1's ±2.0 pp bar:
  - GAIN: the interval wholly above +2.0;
  - NULL: wholly inside [−2.0, +2.0];
  - LOSS: wholly below −2.0;
  - else UNDECIDED.
  - A bound within 1e-9 of a boundary counts as UNDECIDED (rule 8).
- **Looks.** Look 1 is at 3 seeds per side. An UNDECIDED with Δ̂ > 0 goes to look 2, at 5 seeds per side; C0's two
  extra seeds serve every arm in the batch. Look 2 is final.

**R2, the CO-PRIMARY for A3, A4 and A6** (their mechanism is exploitability, league §0(b)):
- **The read.** For each arm's seed-3501 final (b-ON and b-OFF), one fresh best responder: the OFFENSE 5-team archetype
  (`exploiter_teamsets_2026-09-22.json`), 5M steps, at matched budget, dose and regime. `main.best_response_gap
  --rounds` reads it, as population-loop round 1 did.
- **"The gap FALLS"** iff the Newcombe 95 % upper bound of gap(ON) − gap(OFF) < 0. At n ≈ 400 per cell that detects a
  fall of about 7 pp; round 1 read −10.0.

**The guards.** Each one passes or fails deterministically.
- **G-F, the head-to-head frontier.** b-ON × b-OFF via `cross_stat` (square, 3 × 3 or 5 × 5). FAIL iff its two-sided
  95 % upper bound < −2.0 (a detected LOSS).
- **G-E, exploitability.** The R2 read on every arm. FAIL iff the Newcombe 95 % lower bound of gap(ON) − gap(OFF)
  > 0 (detected MORE exploitable).
- **G-C, the collapse battery** (§5.6.5).
- **G-K, the external anchor** (§5.6.6). FAIL iff ON − OFF is detected below 0 (Newcombe 95 %) on EITHER team set.

**Verdicts.**
- **ADOPT-CANDIDATE:** R1 is GAIN, or, for A3 / A4 / A6, R1 is NULL and R2 says the gap FALLS. In both cases every
  guard must pass.
- **TRADE:** R1 qualifies, but a guard fails. A GAIN on R with a head-to-head LOSS is the non-transitive case.
- **NULL / LOSS / UNDECIDED**, after look 2.
- An ADOPT-CANDIDATE goes to the owner as a recipe change for the deep continuation. Nothing is adopted by this rule
  alone.

**Always reported, descriptive:**
- each final's lag curve against the archive (§0(c)), A1's built in through its `ext_` eval rows;
- the max-entropy Nash score, once T57's solver exists;
- §5.1's Δ_b(P) − Δ_b(T), and P\* + b against T\* + b.

**The read's cost.**
- R1: R has ≤ 7 members × 1,000 games × 6 finals, ≈ 42k games per look; C0's games are reused across the batch.
- G-F: 9k games.
- On the GPU in a lease window: 118 games/s pooled on the production net of 2026-10-03 (`eval_and_rating.md`;
  UNVERIFIED for the end state), ≈ 10 min plus the engine starts.
- On rule 5b's CPU: ≈ 1.16 games/s (`design_x26_readiness.md` §2.6), ≈ 12 h. So the read is scheduled in the GPU windows between
  arms.

#### 5.6.5 The collapse battery as a guard (G-C)

Battery items from `design_collapse_and_counterfactuals.md` §4 are read on every final, b-ON and b-OFF, all seeds,
on CPU at rule 5b. Each takes minutes. The intervals are battle-clustered 95 %, ON against OFF, pooled over seeds.
- **B1, entropy** on X26's 7,028-turn bank. FAIL iff the interval of ON − OFF lies wholly below −0.05 nats. X26
  itself broadened by +0.062 from 30M to 160M; a fall of this size is the first sign of type-(b) collapse.
- **B2, the starved share of decisive turns** on X26's own 306-turn truth. FAIL iff the interval's lower bound > 0, a
  detected rise. Read A's resolution (±0.07–0.08, F-CC-3) excludes an ai_v14-sized rise (+0.14), not a small one.
- **B3**, the same on `truth_v2`: reported.
- **B6**, Tier 1 and the Hodge width on each arm's own ladder: reported.
- **Not in the guard:** B4 (3.7 CPU-h), B5 and B10 (not built), B8 and B9.
- **Truth refresh.** If P\* is ≥ 50M past the truth's 150M continuation, B4 re-branches the truth once under P\*'s
  continuation (≈ 3.7 CPU-h), before the first read. That is the collapse doc's own cadence.

#### 5.6.6 The external-anchor protocol at the breaker readout (`../ops/EXTERNAL_ANCHORS_SOP.md`)

Run on every arm (each b-ON and C0) on its seed finals, on P\* once per batch, and on the toy arms the same way:
- **The cell.** `metamon:Kakuna` (ckpt 34), GREEDY vs GREEDY (SOP rule 2). Our side is greedy on CPU, on the
  in-process Rust front end, on a 95xx port.
- **BOTH team sets:** home72 (the sanitised 72-team copy from `n0_endofrun_2026-09-27/scripts/make_home72.py`, as in
  the X26 read) and away (Metamon's 20 `competitive` teams).
  🚨 Both are TRAINING DATA for X26: home72 is 72 / 72 pool teams and away 19 / 20 (ledger 2026-10-18). Adding the
  held-out panel (`heldout72_hpfix` + a third source) to G-K is PROPOSED in
  [`design_diversity_teams_opponents.md`](design_diversity_teams_opponents.md) §5 (F-DV-1); it is not adopted.
- **n ≥ 420 per team set per arm:** 7 role-balanced 20-game units per seed final × 3 seeds, so away carries ≥ 400.
  The unit seeds 0–6 are SHARED across arms (`--team-seed = --seed-base = unit seed`).
- **A unit counts** under the X26 read's rule: `status == OK`, `n == 20`, `regime_verified_decisions`, no
  `team_source_asymmetry`, `peer_clean`.
- **Reported:** each arm against 0.50 (Wilson), ON − OFF (Newcombe; this is G-K), and P\* against X26 @161M's read
  (0.575 home72 / 0.497 away at n = 400, `k2_x26_kakuna_both_teamsets_2026-10-17`; 0.395 on the held-out heldout72). Each number carries its regime (SOP rule 1).
- **Where it runs:** from the checkout ROOT (the pool source's CWD hazard). Cost: ≈ 840 games × ≈ 9 s ≈ 2.1 h per arm
  on one lane (≈ 2 cores), at rule 5b. The home72 config must be committed or regenerated first (F21).

#### 5.6.7 The implementation audit, and the ordered build list

Every dry-run below ran at `9ebda190` on a stand-in P\* (`checkpoint_180000055_steps.zip`), `CUDA_VISIBLE_DEVICES=""`,
rule 5b. Nothing was created. The transcripts are in the staging directory.

| arm | what exists | proven how | what must be built |
|---|---|---|---|
| C0 | the fork path, the pool auto-seed, the recipe restore | `checkargs` exit 0; `launcher --dry-run` "would launch", role FORK, +20,000,000 steps | T54 (the pool seed's step filter, F18) |
| A1 | `--stable-opponents` (explicit zips + labels), `--stable-opponent-selfplay-share`, mastery → floor | `checkargs` 0; dry-run "would launch"; `resolve_stable_opponents` resolved 8 explicit-zip entries, 8 unique labels, arch-gated against X26's `ModelVersion` | nothing for the arm itself |
| A2 | `--pfsp-scale`, `--n-sentinels` (end-to-end skew tests, `self_play_and_pool.md`) | dry-run "would launch" | nothing |
| A3 | `--exploiter`, `--no-self-play`, the stable-opponent slice | phase 1 dry-run "would launch". Without `--no-self-play` it is REFUSED (F16). Phase 2 cannot be dry-run until the exploiters exist, because the resolver runs child-side | a two-phase launch kit (T55) |
| A4 | nothing: the L3 off-slice anchor (`distill_anchor.py`) was deleted 2026-10-02; it is recoverable at `LAST_COMMIT_L3 615a764f` as a reference | — | **T56** |
| A5 | uniform stable sampling and stable PFSP; `=<weight>` is refused at parse; no Nash solver in `src/` (scipy 1.17.1 is in the env) | — | **T57** |
| A6 | the same flags as A1 at share 1.0 + `--stable-opponent-pfsp` | phase 1 dry-run "would launch" (share 1.0 is inside the parser's [0, 1]) | a two-phase launch kit (T55) |
| toy C0 | the fork path from `rb_toy_ent_s4002` | dry-run "would launch" (X34 proved the toy at HEAD, §4.2) | — |
| readout | `main.h2h play-many` (two architectures per engine), `cross_stat` (square), `main.best_response_gap` (pinned archetypes); **T55 (2026-10-17): `src/main/x35/`** — the estimator and decision function (`readout`), the freeze (`refpop`), the argv kit + `launch_x35.sh` (`kit`), the Kakuna runner (`kakuna`) | `main.x35.readout self-check` 14 / 14 on constructed cases BEFORE any real read; every argv equal to its validated stand-in; A2 and both phase-1 exploiter argvs `checkargs` + dry-run at `acb99527` (STAGED_PLAN §6) | the arm-vs-arm / full-pool reader mode is optional (T58); R0 / reader argvs (F27) |

**No arm on this list needs a Rust or engine change.** A4 is learner-side and A5 lives in the Python opponent sampler
(`rust_env_opponents.EpisodeOpponentSampler._pick_stable`). Search breakers would need one; they are not on the list.

**The build list, in dispatch order** (all `TASK_BACKLOG.md` §2, rank X35-n):
1. **T54 — the pool seed's STEP FILTER** (opus-medium, ≈ 0.3 agent-day). A fork seeds only the parent's snapshots at
   or below the fork step, and prints and records the ones it dropped. Without it every arm, C0 included, trains
   against X26's post-P\* selves (F18). It must land before batch 1's pin, and it must fail on revert.
2. **T55 — the X35 stage kit** (opus-high, ≈ 1–1.5 agent-days). **BUILT 2026-10-17** (`src/main/x35/`, STAGED_PLAN §6).
   - `launch_x35.sh`, which re-checks P\*, the pin, the sha256s, `checkargs` and the dry-run before anything is
     created; the two-phase kits for A3 and A6.
   - The reference-freeze tool, with the band rule.
   - The population-score estimator and the deterministic decision function of §5.6.4, self-checked on constructed
     cases BEFORE any real read (T40's discipline).
   - The Kakuna unit runner (the X26 read's `xq.py` pattern) and the committed home72 config.
   - It must land before the first READ, not before the first launch.
3. **T56 — the trailing-magnet KL** for A4 (opus-high, ≈ 2 agent-days).
   - The magnet's logits are a no-grad pre-pass over the rollout buffer, once per rollout (≈ +3 % of the update,
     ESTIMATED). The loss adds `β · KL(π_θ ‖ π_mag)` over legal actions.
   - The magnet refreshes on a declared schedule (P\* static, or every K steps); telemetry `train/magnet_kl`.
   - β = 0 must be bit-identical to the current trainer. Not Rust. It joins batch 1 if it lands before the pin.
4. **T57 — Nash weights** for A5 (opus-high, ≈ 2 agent-days).
   - A deterministic max-entropy Nash solve (LP, then maximum entropy; fixed solver and tolerance) with the
     posterior-mean variant (§2.3 (i)).
   - Per-entry weights on `--stable-opponents`, honoured by `_pick_stable`, with the weights recorded in metadata.
   - The 9-node matrix is played on `main.h2h play-many`.
5. **T58 — `best_response_gap` reads arm against arm** (opus-high, ≈ 0.5–1 agent-day, OPTIONAL). It adds a full-pool
   pseudo-archetype, so unpinned readers stop being UNASSIGNED (F17). The interim is pinned offense readers with
   `--rounds`.
6. **T59 — `ridealong_read --truth-dir` takes generic rows** (F-CC-4; opus-medium, ≈ 0.25 agent-day, OPTIONAL). B5 is
   not in the guard.

#### 5.6.8 The first arm's launch block (A1, seed 3501), dry-run validated, NEEDS THE ORCHESTRATOR'S GO

The validated stand-in is in STAGED_PLAN §2. At the plateau call `launch_x35.sh` (T55, built) builds the argv from the
frozen R file (P\*, the 8 archive nodes) and the typed `--pin <X35_PIN_1>`, and runs STAGED_PLAN §3's pre-checks,
`checkargs` and the dry-run included; its default is a dry run and only `GO` launches (STAGED_PLAN §6).

```
# NEEDS GO. A design block is not a launch command. Stand-in values: P* = 180,000,055, pin 9ebda190.
C=models/rb_x26_s2601/checkpoints
python -m main.launcher --restart-interval-hours 24 --pin-commit <X35_PIN_1> \
  --model $C/checkpoint_<P*>_steps.zip --behaviour-check warn \
  --stable-opponents $C/checkpoint_<N8>_steps.zip:x26_<n8>m,…,$C/checkpoint_<N1>_steps.zip:x26_<n1>m \
  --stable-opponent-selfplay-share 0.5 \
  --steps <P*+20,000,000> --seed 3501 --checkpoint-every-steps 1000000 --device cuda --run-name rb_x35_a1arch_s3501
```

The control C0 s3501 is the same line without the two stable-opponent flags, with `--run-name rb_x35_c0_s3501`.
Pre-checks before the GO:
- X26 and X34 have stopped, and the GPU lease and window are the Training session's;
- the pin contains T54;
- none of the run dirs exists;
- every sha256 is unchanged;
- `checkargs --pin <X35_PIN_1>` exits 0, and the dry-run says "would launch", FORK of `rb_x26_s2601`, +20,000,000.

The startup banners to expect:
- `🌱 [SELFPLAY] [pool] seeded …`, with T54's dropped-step line;
- `🐴 [STABLE] 8 cross-run opponent(s) …`;
- `🧮 [TRUNK PRECISION] bf16`;
- the compile gate's `parity PASS … clip margin`.

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
- **F13. The X35 fork cost was understated.** §5.1 priced a 20M production fork at 3.4 GPU-h from ≈ 1,600 fps. X26's
  checkpoint mtimes give 7,494 s per 10M from 150M to 180M (≈ 1,334 steps/s of wall time, eval included): ≈ 4.2 GPU-h,
  or ≈ 4.4 with start and compile (§5.6.1).
- **F14 (HAZARD). A fork defaults to `--behaviour-check fatal`** (`main/train/rust_env_setup.py`), while X26 runs `warn`
  under Amendment 3. That amendment predicts ≈ 110 rounding stops by 500M, so an untyped 20M arm would stop about 6
  times. Every production X35 argv types `--behaviour-check warn`, and b-ON and b-OFF match.
- **F15. The PFSP prior is negative, and it is pre-era history.** Pure PFSP over a homogeneous fresh-lineage self-pool
  cost ≈ 26–33 Elo (ledger 2026-08-21, the gen-17 attribution). The one positive read, v8's +69, was over a diverse,
  ANCHORED pool under a distillation KL. The PFSP fork `ai_v12_04_pfsp_fork25M` was paused at 30.5M and never read.
  None of these is a comparator (ERAS). They only set A2's prior.
- **F16. An exploiter fork of a self-play parent is REFUSED unless it types `--no-self-play`.** The launcher restores
  `self_play=True` typed from the parent's record, and `--exploiter` is mutually exclusive with it (`resolve_config`).
- **F17. `main.best_response_gap` cannot pair an UNPINNED reader.** `archetype_of` returns UNASSIGNED with "no pinned
  teams", so a full-pool best responder never enters the paired delta. X35's readers are pinned offense-archetype
  readers until T58.
- **F18 (HAZARD, every arm). The fork pool seed copies the parent's WHOLE pool with no step filter**
  (`agents/training/pool_seed.seed_pool`). X26 keeps training while its Tier-1 checks run, so at the plateau call its
  pool holds snapshots from AFTER P\*. Every fork would train against future selves, unlike X26 at P\*. It is matched
  between b-ON and b-OFF, but it is not "plain continuation from P\*". T54 fixes it. Without T54 the leak must be
  declared in the read.
  **RESOLVED 2026-10-17 (T54 BUILT, `gen3_pool_seed_step_filter_v1`):** a fork seeds only the parent snapshots at
  or below the fork checkpoint's `num_timesteps` (equality kept), prunes `pfsp_win_rates` to match and records
  `dropped_steps` in `pool_seed.json`; a pin carrying it has no leak. Still the parent's LATEST, not P\*'s: the
  pool's scalar `win_rate_vs_bots` (the ramp's start) and the kept snapshots' PFSP EMA values.
- **F19. Power at 3 seeds per side is low.** With a seed-mean SD of ≈ 2 pp per side (an ASSUMPTION; the plateau noise
  floor G5 is unmeasured), a two-sample t at df ≈ 4 needs Δ̂ ≳ +6 pp for a GAIN wholly above +2.0. The second look (5
  seeds) is registered for that reason. C0's first seeds measure the SD.
- **F20. A1 trains next to the A-block.** Its archive nodes are 10M from R's own nodes, so the A-block score leans
  toward A1 by construction. The X-block, the head-to-head guard and the Kakuna guard are the unbiased checks. That is
  why A1 cannot be adopted on R alone when they fail.
- **F21. The home72 team config is not in the repo.** The X26 read used `anchors_home72_kakuna.json` from
  `/home/goodlad/dev/gen3ai-reads/…`, regenerated by `n0_endofrun_2026-09-27/scripts/make_home72.py`. T55 must commit
  it, or regenerate it under a recorded sha256, before any X35 Kakuna read. **RESOLVED by T55 (2026-10-17):** the
  config is committed byte-identical (`x35_breakers_stage_2026-10-17/anchors_home72_kakuna.json`, sha256 `e8a064db…`)
  with `home72_manifest.json` pinning all 72 team files on both sides; `main.x35.kakuna` re-checks it before every
  unit. No `data/` file was needed: the teams are the committed `n0_endofrun_2026-09-27/teams_home72/`.
- **F22. Disk.** The launcher requires ≈ 20.7 GiB of headroom per 20M production fork (13.35 GiB before margins).
  Batch 1 is 9 production forks + 9 toy forks + ≈ 7 exploiter and reader runs, against 318 GiB free today (X34 needs 48
  GiB). Finished arms must be thinned to their 10M nodes + final under the retention policy as they finish, or a later
  launch is refused.
- **F23. Literature is from memory or second-hand, UNVERIFIED here:** AlphaStar's league roles (main agents, main
  exploiters, league exploiters, PFSP; Vinyals et al. 2019), PSRO (Lanctot et al. 2017), and R-NaD / MMD (as cited in
  `design_learner_recipe.md` §3.9).
- **F24. Our "main exploiter" targets a FROZEN P\*.** AlphaStar's track the moving main agent (UNVERIFIED). A
  co-trained exploiter would need two learners sharing one GPU, which is neither designed nor costed.

The T55 build (2026-10-17) adds:
- **F25 (HAZARD, the toy side). The A-block rule is undefined for a start node at or below 150M.** There, two archive
  nodes sit 10M apart, their midpoint ties down onto one of them, and the A-block would not be disjoint from the
  archive. `main.x35.refpop` REFUSES such a freeze by name (P\* = 100–150M all refuse; 160M and up pass). X26's P\*
  is past 180M, so production is unaffected. T\* is X34's plateau node, which may be ≤ 150M: the toy freeze then
  needs an orchestrator rule first (no rule was invented here).
- **F26. The two Newcombe intervals differ by their z** (`best_response_gap`: 1.96; `main.anchors.results`:
  1.959963985), by ≈ 1e-5 at the bounds. At one z they agree to 1e-12 (the self-check). R2 / G-E use the R2 meter's
  interval, G-K the anchors' one, so each read quotes the interval its meter prints.
- **F27. R0 and the R2 readers have no registered argv.** The pinned offense-archetype best responder (§5.6.4) was
  never dry-run, so `launch_x35.sh` does not cover R0 (batch-1 queue #0) or the readers.
- **F28 (HAZARD, batch 1). Pre-check 8 HOLDs at today's disk:** 18 batch-1 forks × 20.68 GiB = 372 GiB against
  316 GiB free. A launch then needs `--thinning`, which declares that finished arms are being thinned (F22). Toy
  forks are counted at the production figure, so the check is conservative.
- **F29. The readout's declared choices** (`main.x35.readout` docstring): b-ON and b-OFF must have EQUAL seed sets of
  the look's size; zero variance on both sides is INCONCLUSIVE; G-K needs ≥ 400 counted games per team set per side,
  and a tie is a non-win; a guard whose bound sits within 1e-9 of its fail boundary does not FAIL (`main.h2h.cross`'s
  convention) but is recorded; a guard that was not read blocks only a verdict that needs it (R1 qualifying).
- **F30. Not exercised:** the `GO` path (the launch, the 180 s liveness HOLD, the watcher), because nothing may
  launch. While X26 is live pre-check 1 HOLDs, as it must, so checks 2–8 ran on a test copy of the script without
  check 1, on a stand-in R in `/tmp` (`kit_validation_acb99527.txt`). The A6 stand-in typed `--behaviour-check warn`
  twice; the kit types it once (argparse reads the same value).
- **F31 (A2, a KNOWN BIAS, not fixed; from T54 `cdab9cfe`).** The fork's pool scalars are the parent's LATEST, not their
  values at P\*: `win_rate_vs_bots`, the `pfsp_win_rates` of the KEPT snapshots and `pool_generation`. A2's early PFSP
  weights start from post-P\* rates; A2's read declares it. T54's seed runs in the child, so `launch_x35.sh` checks
  `pool_seed.json` AFTER the spawn (a null `fork_step` is a HOLD) and kills the fork by its PID on a failure.

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
| 2026-10-17 | **X35's arm list and order** (design worker, for the orchestrator) | **A1 archive pool (first) → A2 PFSP → A3 main exploiters → A4 trailing-magnet KL → A5 Nash-weighted archive (conditional) → A6 league exploiters (conditional) → A7 entropy NOT RUN (trigger-gated)**, with C0 as the shared control and R0 (exploitability of P\*) at the call. A1 + A2 are the transfer-calibration pair | PFSP first (a negative pre-era prior, F15; it reweights only the 20-deep recent window, which the measured cycles have left); an anti-starvation arm (the collapse doc finds nothing for it to fix, and no deployable counterfactual); one-ply search in this list (the search programme owns it, through the §5.2 slot) | §5.6.2; `design_collapse_and_counterfactuals.md` §3–§5; ledger 2026-08-21, 2026-09-24 |
| 2026-10-17 | **X35's shared frame** | forks from P\* at ONE pin per batch, with ONE shared control set per (start, pin); X26's own recipe (not `recipe.fork`); `--behaviour-check warn` typed; L_P 20M / L_T 70M; seeds 3501–3505 / 3601–3605; ABBA order, A1's seeds first | a control per breaker (doubles the cost for no information at a shared pin); `recipe.fork`'s 5 epochs and frozen LR (a second lever); the fork default `fatal` (≈ 6 rounding stops per arm, F14) | §5.6.1 |
| 2026-10-17 | **X35's readout (amends §5.1)** | the PRIMARY R1 = Δ_pop against a FROZEN reference population (own archive + another lineage, band-filtered), Welch over seed finals at 97.5 % per look, looks at 3 and 5 seeds, the ±2.0 bar; the co-primary R2 (the best-response gap falls) for A3 / A4 / A6; guards G-F (head-to-head), G-E (exploitability), G-C (collapse battery), G-K (Kakuna, both team sets) | §5.1's b-ON × b-OFF head-to-head as the primary (a single-opponent read, which league §0 rules out in a cyclic regime); the archive Nash mixture as the primary (not built, and unstable on our data, F-ED-3) | §5.6.4–§5.6.6; `design_league_decisions.md` §0 |
| 2026-10-17 | **The fork cost** | ≈ 4.4 GPU-h per 20M production seed (MEASURED from X26's late-run checkpoints) | 3.4 (assumed ≈ 1,600 fps) | F13 |
| 2026-10-17 | **T55, the stage kit** (build worker) | `src/main/x35/`: the readout as pure functions with a constructed-case self-check committed BEFORE any real read; R frozen write-once with every sha256 and re-verified before each launch; every argv built from the frozen R (never typed) and equal to its validated stand-in; a launch script whose default is a dry run, HOLDs (exit 7) on the first failed pre-check and launches only on a literal `GO`; the home72 config committed byte-identical with a 72-team manifest | hand-substituted argvs at the call (the drift F18 / F14 came from); a decision function reading the ledger itself (the readout takes declared cells or numbers); regenerating home72 (a second team set to prove identical) | §5.6.4, §5.6.7; F25–F31; STAGED_PLAN §6 |
| 2026-10-18 | **Cross-references to the diversity design (design worker; no change to X35)** | §5.6.2 and §5.6.6 point at `design_diversity_teams_opponents.md`. Its arm D1 is PROPOSED at rank 2, and adding a held-out panel to G-K is PROPOSED, both pending the orchestrator's GO. §5.6.6's "P\* against X26 @161M" reference values were updated from the n = 100 / 240 reads (0.600 / 0.570) to the n = 400 reads (0.575 home72 / 0.497 away; 0.395 heldout72) | editing the arm list or the readout here before adoption | `design_diversity_teams_opponents.md` §6, F-DV-1; ledger 2026-10-17 / 2026-10-18 |
| 2026-10-18 | **D1 ADOPTED at rank 2, conditional on X37** (orchestrator, Q5 of `design_diversity_teams_opponents.md`); its readout follows that doc's evaluation matrix (§5.6) | the §5.6.2 pointer updated from PROPOSED to ADOPTED-CONDITIONAL; the arm table itself unchanged until X37 reads | adding D1 to the table before X37 (its rank depends on X37's verdict) | `design_diversity_teams_opponents.md` §5.6, §6.1, §9 |
| 2026-10-18 | **The throughput audit against the published bots (§1.6; performance-analysis worker)** | rank the THROUGHPUT levers by expected gain × risk: (1) fewer PPO epochs at a held optimizer-step count, read at MATCHED GPU WALL (X44, ≈ 10 GPU-h pilot); (2) cut the opponent inference fan-out (X45, sized first by the GPU cells `c_pool1` / `c_pool5`); (3) overlap collection with the update (X14, as registered); GPU cells for the 10-19 gap written (`run_gpu_gap.sh`), not run | slimming the network (history seats, depth) as a throughput lever (the owner's 2026-10-16 framing; X36 stays deferred); env-core speed work (≤ ×1.08 by Amdahl: 7.6 % of the cycle); larger micro-batches (measured ≤ 7 %, slower on a 5090, 4,096 does not fit 12 GB); reading the 75× as a firm number (the source's units disagree by ≈ 5×, F1) | `../research_state/measurements/throughput_audit_2026-10-18/README.md`; X26 TensorBoard @312M; `gpu_bench_2026-10-18/suites/`; the 2026-10-18 literature survey |
| 2026-10-19 | **X35's R FROZEN (production side)** (measurement worker) | R at P\* = X26 @ 280,000,170: A-block 280 / 270 / 80 / 140 / 210M (mid(N_4, N_5) = 145M is a tie → 140M by F25), X-block `production` KEPT at 0.6935 [0.666, 0.721] (500 pairs, CPU); archive N = 30, 60, 100, 130, 160, 190, 230, 260M; frozen with every sha256, `verify` OK. T\* ABSENT (X34 not launched): ONE X member, the toy side not frozen; adding T\* is a NEW freeze, an orchestrator call (F-R1). R's paths stay in X26's run dir because the kit forks from R's P\* path (F-R3) | waiting for T\* before any production arm (X34 runs after X26); a stand-in T\* (`rb_toy_ent_s4002`'s 100M final is X34's start, not its plateau) | `../research_state/measurements/x35_refpop_freeze_2026-10-19/README.md`; `reference_population_production.json` |
| 2026-10-19 | **X45 (b): the GROUPED FORWARD built behind a flag (build worker)** | `--t2-grouped-opponents` (default OFF, `gen3_t2_grouped_forward_v1`): every opponent slot of a slot group in ONE `torch.func.vmap` forward over the stacked weights per host step, the model UNTOUCHED (a factory-batching `TorchFunctionMode` with an exact `-0.0` identity, the stashes reset in the body, a vmap rule for the SDPA-bias prim); adoption is a PARITY gate + the paired GPU read (§1.6), no learning A/B | the model-wide out-of-place rewrite the 2026-09-29 NO-GO named (34 sites in 15 files on the trainee's hot path, with X34 launching from main the same morning); `torch.func.functionalize` + vmap (a torch view-replay bug on an open slice); a per-call GATHER of the live slots' weights (a weight copy per flush); (a) the live-snapshot cap first (it changes the within-rollout mix; the GPU cells show the cost is the fan-out, not the slot count: pool 1 is slower than pool 5) | §1.6; `src/agents/inference/service/grouped.py`, `grouped_test.py`; `~/gen3ai_archive/gpu_gap_2026-10-19/` |
| 2026-10-19 | **The bot-ideas screen (design worker)** | five architecture / recipe ideas from the published bots costed as X35-frame GROWTH forks (function-preserving at the fork, sharing C0, equal effective learn time primary) in [`design_bot_ideas_screen.md`](design_bot_ideas_screen.md); X47 (toy depth 2 → 4) is the first new candidate after X34 plateaus; X48 / X47b / X49 slot after A3 / A4; X50 (human data) is a zero-GPU S0 read pending the owner's Q1; set mutation is X41; the §5.5 pointer added | fresh-from-scratch A/Bs; experts or an intent head on the toy; a BC warm start before ai_v11's rung 1 | that doc's §1–§3 and Decision record |
| 2026-10-19 | **X14 overlap BUILT behind `--overlap-collect` (default OFF)** (build worker) | the worker plays the NEXT rollout during each update (from a process's 2nd update; stops at the trigger, so ON runs are deterministic); DECOUPLED PPO as a buffer rewrite (old_log_prob := log π_prox, advantage × min(π_prox/μ, 2)); K9(b) on a 2-slot behaviour snapshot; the worker on its own CUDA stream | plain PPO on stale rows (X26's age-1 rows start at KL 0.016 vs the controller's 0.01 target and 22.5 % clipped: the controller would cut the rate for the lag); reloading T2 when the update ends mid-rollout (fresher rows, but timing-dependent, i.e. non-deterministic); V-trace (one version of lag) | `designs/training/rust_collector.md` "Overlap (X14)"; `research_state/measurements/x14_overlap_2026-10-19/` |
| 2026-10-20 | **X14's decoupled update NORMALISES the raw advantage, THEN weights it** (core review A2 + C-3, C-2; correctness worker) | w = min(π_prox/μ, 2) is stored BESIDE the raw advantage (`RolloutBuffer.is_weights`) and each micro-batch hands R1 w · norm(A) with R1's own normalisation off (Hilton et al.'s order; R1's code unchanged, the serial path byte-identical); the `signal/*` tags and the ride-along `adv` target read the raw advantage; `--overlap-collect` is a recorded `recipe.fresh` row (false), so a resume or fork of an ON arm keeps it | the build's fold (advantage × w, THEN the micro-step's normalisation): a BIAS, not a rescale — for an action-independent advantage it pushes along μ → π_prox, (−0.096, +0.096) a logit on the CPU repro, 0 to rounding after; passing w into R1 as its own input (it would change R1's signature and its compile; normalising outside R1 keeps it) | `designs/training/rust_collector.md` "Overlap (X14)"; `x14_overlap_2026-10-19/adv_norm_bias.py`; `overlap_test`, `recipe_surface_test` |
| 2026-10-19 | **X35 batch 1 + X47 LAUNCH SHEET (launch-staging worker)** | [`x35_breakers_stage_2026-10-17/BATCH1_LAUNCH_SHEET.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/BATCH1_LAUNCH_SHEET.md): the 24 forks in GPU order (production A1 / C0 ABBA, D1, A2 on P\*; then the toy block with X47 interleaved with the toy C0), ≈ 108 GPU-h at the MEASURED throughput (X26 1,333–1,358 steps/s, X34 4,719 steps/s); every argv `checkargs` exit 0 and "would launch" at `803a13e3` (toy on the 200M STAND-IN for T\*); `x35_required_commits.txt` now also requires `f9360870`, `4df10e65`, `24616919`, `aa5feaf5`, `bd3126e0`; nothing launched | deciding the open choices C1–C6 (the orchestrator's); adding X47 to the kit (out of scope, F-LS-1) | the sheet §4.3, `batch1_validation_803a13e3/`; the toy forward calls `fixed_size_tau` 0 times (counting patch, 20 in the positive control) |
| 2026-10-19 | **The orchestrator's rulings C2 / C3 / C4 on the batch-1 launch sheet, BUILT into the kit (launch-kit worker)** | **C2:** X47 interleaved with its toy C0 and toy A1, toy A2 last (the kit's `BATCH1` is the 24-fork order of the sheet). **C3:** the toy arms type `--behaviour-check warn` under X34's ceilings (judged max ≥ 1e-3, p99 ≥ 1e-4, any non-finite value ⇒ STOP, read by hand) instead of `fatal`; production unchanged. **C4:** batch 1 launches without R0. X47 is the kit arm `toy_x47` (F-LS-1 closed), `kit disk` counts 24 forks, and `validate_batch1.sh` gives every line a deterministic verdict (`main.x35.kit validate-run`) | X47 after toy A2 (C2a, un-interleaved with its C0); the toy on `fatal` (C3a: stricter, but a stop leaves an arm short, and X34's scalars never reached the bar: judged max 7.7e-5 over 2,160 updates to 212.4M, 0 at 1e-4); waiting for R0 (C4) | the sheet §5 and §7, `src/main/x35/kit_test.py`, `batch1_validation_9c3bf3c4/` |
| 2026-10-19 | **The orchestrator's rulings on the batch-1 review, BUILT before any X35 launch (X35 amendment worker)** | AM-1 ONE pin written once and enforced against every batch-1 run's recorded pin; AM-2 the variance gate G1 (σ̂ ≤ 1.2 pp, 4 df) after row 6; AM-3 the toy calibration only after production's FINAL class; AM-4 X47 look 2 + `main.x35.x47`; AM-5 D1's widened H; AM-6 K9(b) at each arm's end over its own rows (VOID on a crossing); AM-7 the staged order (`batch.STAGES`); POOL one recorded seeded set per side (C0 = X26's pool as of today, 14 snapshots 164–268M); G-E readers only for a GAIN arm; C1 (a) | per-arm pins ("HEAD at its launch"); 3-seed reads before σ is known; a toy calibration that cannot record agreement; X47 without look 2; hand reads of K9(b) between restarts; the G-E readers for every arm (≈ 4.4 GPU-h that decide nothing unless an arm reads GAIN) | `X35_BATCH1_REVIEW.md`; `BATCH1_LAUNCH_SHEET.md` §0; `src/main/x35/batch.py`, `batch_test.py` |
| 2026-10-19 | **X35's toy side STAGED ON THE REAL T\* (launch-staging worker)** | X34 PLATEAUED: T\* = the 290M node (`checkpoint_290000087_steps.zip`, sha256 `cfa7f7be…a22965e7`). The toy R is FROZEN on it: archive N_1…N_8 = 30 / 70 / 100 / 130 / 170 / 200 / 240 / 270M, A-block 290 / 280 / 80 / 150 / 220M (85 a tie → 80), X-block P\* KEPT at **0.4160** [0.3863, 0.4457] (T\* vs P\*, 500 mirrored pairs, a cross-architecture cell on one engine, 14.1 min of one CPU lane); `refpop verify` OK. All **32** batch-1 argvs re-validated at `03b983be` with the toy on the real T\* (`--steps 360000087`; X47 `grow=[DECLARED trunk_layers 2 -> 4]`); `x35_required_commits.txt` now also requires `03b983be` (the canary YARDSTICK fix, code in `ee49f1f9`); the toy pool digest recorded read-only (pool-check, no pin file written). The 200M / 210M stand-ins are retired | a 210M T\* (the pre-plateau guess); freezing without the P\* cell (an empty X-block); reading the toy cell for the production R (it has T\* as the PLAYER: C5 needs its own cell) | `../research_state/measurements/x35_toy_refpop_freeze_2026-10-19/README.md`; `reference_population_toy.json`; `BATCH1_LAUNCH_SHEET.md` §4.2–§4.3; `batch1_validation_03b983be/`; `src/main/x35/refpop_test.py` |
| 2026-10-19 | **X35's production R RE-FROZEN with T\* as a second X member (ruling C5; launch-staging worker)** | The P\*-player band cell was MEASURED, not assumed symmetric: P\* (X26 @ 280M) vs T\* (X34 @ 290M), 500 mirrored pairs, CPU, **0.5905 [0.5605, 0.6205]** (W/L/D 588 / 407 / 5), against the toy cell's 0.4160 (complement 0.5840; the difference is 0.3 SE). T\* KEPT. The old file moved aside as `reference_population_production_PRE_C5.json`; the new production R has **7 members** (A-block 280 / 270 / 80 / 140 / 210M, X `production` 0.6935, X `tstar` 0.5905), content sha256 `cef0960e…e4519`, `refpop verify` OK. `x35_required_commits.txt` also requires `2be4595f1` (K9(b) judged only on >= 1,024 rows) and `52ac48296` (the supply guard no longer counts a fresh, never-seeded pool as starving) | assuming exact symmetry and freezing from a declared `--band-json` score | `../research_state/measurements/x35_refpop_freeze_2026-10-19/README.md` §6; `reference_population_production.json`; `BATCH1_LAUNCH_SHEET.md` §5 C5 |
