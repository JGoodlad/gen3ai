# BUYING VALUE DISCRIMINATION: a measurement-first pilot, and the label-factory direction that enables search

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A decision or build that differs from this
doc updates it in the same commit, and the Decision record at the end says what changed and why.

**Status (2026-10-11): DESIGN, REVIEWED AND REVISED the same day** (an independent adversarial review at the owner's
request; §12 lists every change and what remains disputed). **The pilot is REGISTERED (§7.10 Amendment 1: both
closing-test seeds 2001, end2 and prod, oracle world) and its driver is built**
(`research_state/measurements/value_disc_pilot_2026-10-11/`); no outcome has been read. The only numbers measured under this doc
are the review's POST-HOC reads of BANKED rows (`research_state/measurements/value_disc_review_2026-10-11/`, §12.2):
CPU, no playout run, no outcome of the pilot seen. §7 is the pilot's DRAFT pre-registration: it becomes a registration
only when its own commit lands BEFORE the first playout and a ledger entry records it. §5's RNG findings are read from
the code at `c1fae513` and re-verified by the review. The pilot is an OFFLINE CPU measurement. Any IN-LOOP use of
playouts to the end (a label factory feeding training) stays DEFERRED under the owner's one-ply scope rule (2026-09-30,
§10 Q1); whether that rule also covers an OFFLINE evaluation battery with a playout leaf is an owner question (Q1).

## 0. Summary

| | |
|---|---|
| question (owner, 2026-10-11) | "We need at least an order of magnitude more rollouts to enable discrimination for search, which we can't get in the current regime. **Do we really?**" |
| short answer (revised) | **Probably not for DECISIONS; yes only for RESOLVING pairs, which is the wrong target.** On banked Rustboro-era rows (the X5 blob arm, the Rust core, off-distribution Lane S states, an ORACLE world), picking the action an 8-rollout leaf likes best among all legal actions beats the policy's own pick by **+0.062 win probability per decision [+0.045, +0.079]**; 1 rollout already buys **+0.025 [+0.012, +0.037]**; the critic buys **+0.008 [−0.005, +0.021]** (§2.3). The old playout arm bought nothing in games because its ACTING RULE demanded each pair be RESOLVED at 2 SE and so acted on ~5 % of pairs at 4 rollouts. More rollouts do still pay (8 vs 1: **+0.037 [+0.025, +0.050]**), so the curve is not flat, but the first few rollouts carry most of the value and cheaper multipliers exist (a truncated leaf, a luck-adjusting control variate, a regret-aware acting rule). **Pairing (CRN) is measured, not predicted, at VRF ≈ 1.06–1.13 in today's dice mode** (§5.2). The training side is separate: a frozen-trunk READOUT refit on de-noised labels does not help (one checkpoint); a TRUNK trained on sibling targets is untested |
| the caveats that bound every number above | ORACLE world: the playouts know the opponent's real team and sets, and the critic does not (§3 row 5); determinization is DEFERRED (owner, 2026-09-23), so none of this transfers to the ladder yet. Off-distribution states for that checkpoint. A greedy-continuation truth, not Nash. A per-decision improvement is not a per-game win rate. POST-HOC on banked rows: the pilot re-measures it on the adopted end-state checkpoint's own states |
| the RNG finding (re-verified) | ONE PRNG per battle (`BattleState.prng`, a port of Showdown's `prng.ts`); every die of both sides comes from it in execution order (47 direct call sites, 46 on the driver path, plus 7 `speed_sort` sites). A reseed at the branch shares the STREAM across siblings, not the roll-to-event MAPPING. Draw COUNTS depend on the action far more than the first draft said: speed-tie shuffles at every `eachEvent` when the two actives tie, two always-tied handler pairs, secondaries drawn into a KO, and in gen 3 ANY faint cancelling the rest of the turn (§5.1) |
| the proposal | two COUNTERFACTUAL-ONLY dice modes, never reachable from training: **S1 per-turn reseed** (≈ 0.5–1 agent-day; the hook must sit BEFORE the end-of-turn Quick Claw roll) and **S2 counter-keyed draws** (≈ 2–3 agent-days; keys refined in §5.3). Both are equal IN LAW to normal play by construction (§5.4). **Demoted:** each is built only if the pilot says a FULL-length rollout leaf is still needed after the cheaper levers (§7.5 R-PAIR) |
| the pilot (§7, revised) | the closing test's adopted checkpoint; **600 eligible states stratified by the policy's top-2 gap** (torn states are a stratum, not the sample: the banked read finds the leaf's value NOT concentrated there); **all legal actions**; 32 shared seeds (8 for the leaves, 24 for the truth); greedy continuation; a per-turn value trace. **Primary metric: DECISION IMPROVEMENT** over the policy's own pick (unbiased against held-out truth seeds). Scorers: the critic, the rollout leaf at K = 1/2/4/8, TRUNCATED leaves (d = 1/2/4 turns then the critic), an AlphaGo-style mix, policy-prior-restricted leaves. The independent-seed control is DROPPED (the derangement estimator reads 1.000 [0.996, 1.005] on 10,667 banked pairs) |
| cost | ≈ 134,000 CPU playouts ≈ 4–15 CPU-hours (0.10 CPU-s per playout measured in steady state on the blob arm; the end-state network's cost UNVERIFIED), capped at 12 by the size rule, at `nice -n 19` beside the live run. **No GPU** |
| fit | after the closing-test verdict (its adopted arm is the frozen checkpoint), alongside the bf16 A/B (GPU; this is CPU) |

---

## 1. The problem, and the owner's question

A one-ply search picks, at a decision, the action whose successor looks best. That needs a scorer that **orders
sibling actions at the same state**, and, more exactly, one whose ARGMAX is a better action than the policy's own pick.
Today's evidence on the win-prob critic, with its provenance (the critic has changed under these numbers):

- The ai_v12 / ai_v13 heads (September 2026, pre-Rustboro, the Python encoder on the Rust sim bridge) ordered
  contested siblings at **0.567–0.578** across twelve heads (UNDERSTANDING §4.x). The Rustboro-era X5 blob arm
  (`rb_x5ab_blob_s1001`, 15M, the Rust core) reads **0.5932 [0.5578, 0.6274]** on Lane S contested pairs
  (`crn_label_refit_2026-10-06/result.md`; a different state set and truth, so not a level comparison).
- An 8-rollout leaf reads **0.670** on those same blob pairs (+0.077 [+0.035, +0.115]).
- Every naive search built on a critic leaf lost or tied (`search_dividend_winprob_heads_2026-09-11`, ai_v12 heads:
  0.495–0.507 paired mirror win rate, none clear of 0.50). **The end-state critic's sibling ranking has never been
  measured** (the pilot's first read of it).

The owner's worry: buying the playout-level ordering by brute force needs ~10× more rollouts than the current regime
can produce. The orchestrator's answer was that brute force buys discrimination only at √n, and that cheaper levers
exist. This doc turns that answer into claims that can fail, prices each lever, and registers a pilot. **The review's
reframing:** search consumes DECISIONS, not resolved pairs. A pair the rollouts cannot resolve is, on average, a pair
where picking wrong costs little, so the right currency is the decision's improvement, and the question becomes "how
many rollouts per decision buy how much improvement", which the banked rows already answer in part (§2.3).

---

## 2. Definitions, and what is already measured

### 2.1 Definitions

| term | meaning | the meter here |
|---|---|---|
| **calibration** | predicted win probabilities match observed frequencies. A monotone remap fixes it without changing any ORDER | ECE against a truth mean |
| **discrimination (resolution)** | the scorer ORDERS states or actions the way their true values are ordered. A rank property, invariant to any monotone remap | AUC; pairwise accuracy |
| **GLOBAL AUC** | wins vs losses separated ACROSS states: "is this position winning?" Dominated by between-state variation (material, matchup) | the critic's own win-prob vs the outcome: **0.766 / 0.763** (legacy / static seed means; `probe_battery_2026-10-09/critic_auc.json`) |
| **SIBLING RANKING** | at ONE state, which of two ACTIONS leads to the better outcome: `sign(V(s′_a) − V(s′_b))` vs the sign of the true gap. A critic can have a high global AUC and a near-chance sibling ranking | pairwise accuracy on non-tied sibling pairs under shared dice (UNDERSTANDING rule 25). **Weights a near-tie the same as a decisive pair** |
| **DECISION IMPROVEMENT** (added by the review) | IMP(s) = E[ T(â_s) − T(top1) ]: how much better, in win probability, the scorer's argmax over the legal actions is than the policy's own top-1, scored against a truth T on seeds the scorer never saw. Unbiased (no max over noisy truth means is taken). Zero for a scorer that agrees with the policy; it counts a near-tie as nearly free and a decisive error as costly, which is what a search's results in games are made of | §7.4 D1 (the pilot's PRIMARY) |
| **true sibling gap** Δ_ab(s) | Q(s,a) − Q(s,b) under a FIXED continuation policy in the TRUE world (policy evaluation): the expected outcome of "this action, then that policy on both sides" | the mean over M shared seeds of `o_a − o_b` |
| **ORACLE vs HONEST world** (added) | every playout tool here plays from the TRUE battle (both real teams; `truth.py` `to_log`); each side's policy still sees only its own one-sided rows. So a rollout leaf is an ORACLE leaf: it estimates the true-world Q, while the critic estimates E[Q \| observation]. An HONEST leaf needs determinization (sampled opponent worlds), DEFERRED by the owner 2026-09-23 (`program_rust_core.md` §6a) | stamped `world=oracle` on every number |
| **TORN state** | the policy itself is undecided: its top-2 masked-logit gap is STRICTLY BELOW the 40th percentile of the eligible pool (`fork_arm.contested_threshold`, `gen3_fork_select_v1`; the strict `<` is what `crn_label_refit_2026-10-06/score.py` applied). Since π(top2)/π(top1) = exp(−gap) exactly under the masked softmax, a probability-ratio cut is the SAME ordering (§10 Q8). The blob arm's cut was 0.597 logits, i.e. π₂/π₁ > 0.55 | — |
| **separable pair** | a pair whose gap the playouts resolve at budget M: `|Δ̂_M| > 1.96 · SE_M` (paired). A property of the BUDGET as much as of the game: every pair with Δ ≠ 0 is separable as M → ∞ | §7.4 M2 |
| **the irreducible-luck ceiling** | two different limits: (i) a SINGLE game outcome is a Bernoulli draw, so a yardstick of single outcomes caps any scorer (the single-rollout label mis-ordered top1 vs top2 39.7 % of the time on arm W, ceiling ≈ 0.61; `leaf_ceiling_controls_2026-09-19/`); (ii) many sibling pairs are TRULY near-tied: no scorer orders them, and the decision currency charges almost nothing for them. (i) shrinks with M; (ii) does not | (i): split-half agreement; (ii): the method-of-moments gap SD (§7.4 M2) |

**The pooled-correlation trap, named in advance.** The Python fork arm saw **79.7 %** of branch pairs TIED (same
outcome; `designs/training/forks.md` §2). That is NOT evidence that pairing correlates sibling outcomes. With no
within-state correlation at all, heterogeneity in p across states inflates the tie rate (p ∈ {0.1, 0.9} gives
P(tie) = 0.82). Correlation is measured only WITHIN a state, across seeds, after centring each action's mean
(memory note `feedback_pooled_correlation_simpson_trap`). The review's §2.3 read does exactly that.

### 2.2 What the repo already knows (each claim with its tag AND its provenance)

**Provenance key.** *v13-W* = arm W, `ai_v13_02_flywheel_winprob` @ 75,005,952, config_version 119, `ARCH_SIGNATURE`
`gen3_critic_route_wave_v1`, `win_prob_mode shaping`: pre-Rustboro, pre-X5, Python encoder, playouts on the Rust sim
bridge. *rb-blob* = `rb_x5ab_blob_s1001` @ 15.05M (pin `e5e660dd`): Rustboro era, the Rust env core, the win-prob-only
critic, X5's blob arm (not the end state), read on the Lane S bank (ai_v14-lineage eval traces: OFF its own
distribution). Every playout in both is ORACLE-world, greedy on both sides.

| fact | tag | provenance | source |
|---|---|---|---|
| The critic ranks contested siblings at 0.567–0.578 across **twelve heads** (two 10M, two 75M+, refits, a 6× wider head, a ranking term, a pre-pool tap). A READOUT fitted on the frozen trunk's features in 1.1 s matches the 75M head | MEASURED | ai_v12 / v13-W heads; **transfer to the end-state critic: UNVERIFIED** | UNDERSTANDING §4.x; `offline_leaf_fit_2026-09-18/` |
| A **rollout leaf** beats it: 0.543 / 0.580 / 0.613 / 0.652 / 0.671 at K = 1/2/4/8/16 on the top1/top2 column vs the critic's 0.574. Knee at K = 4 (K = 8 on the leaf column) | MEASURED, DETECTED at K ≥ 8 on the column | v13-W | `rollout_leaf_kcurve_2026-09-19/` |
| **The acting rule threw the leaf away**: `|mean d| ≥ 2·SE`, ≥ 4 pairs, resolved 4.5 % of pairs at R = 4 and 12.5 % at R = 16; live, it changed 1 of 250 decisions. When it resolved, it was right 83–92 % of the time (a selection effect) | MEASURED | v13-W; **REPRODUCED by the review on rb-blob**: 6.7 / 15.3 / 17.7 / 26.9 % at R = 4/8/16/32 on contested pairs, 4.9 / 12.3 / 12.3 / 18.0 % on top1/top2 (§2.3) | same; `value_disc_review_2026-10-11/pairing_read.out.jsonl` |
| **A de-noised LABEL does not help a frozen-trunk READOUT**: refit on the mean of 8 shared-dice rollouts: REFIT-K8 − BASE **−0.014 [−0.032, +0.002]** (a gain is excluded); K8 vs K1 label no difference. The 8-rollout leaf beats the head by **+0.077 [+0.035, +0.115]**. The refit buys calibration (ECE 0.065 → 0.032), not resolution | MEASURED · NOT DETECTED, gain excluded · ONE checkpoint, 3,247 training actions, the trunk FROZEN, the readout sees `value_pooled` of the one-ply successor only | rb-blob | `crn_label_refit_2026-10-06/result.md` |
| On STARVED near-best actions vs DOMINATED ones the head is at 0.440 while 8 rollouts are at 0.885 | MEASURED · descriptive | rb-blob | same, M3b |
| λ-return targets for the win-prob BCE (λ 0.9, 0.95): the 0.9 resolution gain was NOT CONFIRMED by its replicate and 0.95 failed its registered claim. `--win-prob-lambda` is deleted (L2) | MEASURED · NOT CONFIRMED | ai_v12 ladder arms (2026-09-11) | `designs/deleted_flags.md`; ledger 2026-09-11 (L17453, L17578) |
| Contested-state FORKS into the PPO buffer (top-2 + random, CRN dice and draws) did not move the head: Δ −0.0003 [−0.0148, +0.0136] | MEASURED · NOT DETECTED | `ai_v13_03_fork` (pre-Rustboro) | `fork_arm_read_2026-09-16/`; `designs/training/forks.md` |
| The top-1 vs top-2 true gap: E\|gap\| ≈ 0.177 (sd 0.222) on the win scale, while the policy's top-1 edge over its top-2 is +0.008, not clear of zero. **Corrected reading:** this is NOISE-CORRECTED (method of moments: observed variance minus the label-noise variance), not "biased up by noise", and E\|gap\| = sd·√(2/π) ASSUMES a Normal gap; a spiky distribution (many near-ties, few large gaps) has a smaller E\|gap\| at the same sd. On rb-blob the same estimator gives a top-1/top-2 gap sd of **0.129** (contested) and the split-half agreement (0.577, n = 78) is below what a Normal gap of that sd predicts (≈ 0.67): the spiky shape | MEASURED (v13-W); REVIEW read (rb-blob) | v13-W; rb-blob | `leaf_ceiling_controls_2026-09-19/` §5.3; `value_disc_review_2026-10-11/` |
| Within a decision's one-step value-target variance, the DICE carry 13.9 % (uniform over legal) / 26.5 % (behaviour-weighted); the opponent's action the most | MEASURED · Node era, 140 decisions | Node era | memory `project_three_axis_value_variance` |
| 🚨 **"CRN in this sim shares the dice STREAM, not the roll-to-event MAPPING"**: a single-seed action sweep over-read the action effects by ~2× because a different action consumes the stream differently | MEASURED · Node era; **now measured on the Rust core too** (§2.3: ρ ≈ 0.06–0.12 at the terminal) | Node era; rb-blob | same; §2.3 |
| Playout cost: **0.21–0.23 CPU-s** per playout in the refit's PILOT on a loaded box; **0.10 CPU-s in steady state** over 160,256 playouts (~30 sim turns each, greedy both sides, CPU torch forward) | MEASURED | rb-blob | `crn_label_refit_2026-10-06/README.md` Amendment 1; `result.md` § Cost |

### 2.3 The review's reads of banked rows (POST-HOC, 2026-10-11)

The refit's committed held-out truth rows hold, for each of 563 Lane S turns, EVERY legal action's outcome on the SAME
32 `sodium,` seeds (rb-blob greedy on both sides, Rust core, oracle world). That is the pilot's M1 at the terminal,
already played. Three scripts (`research_state/measurements/value_disc_review_2026-10-11/`, outputs beside them) read
it; battle-clustered bootstrap, 1,000–2,000 draws, `default_rng(20261011)`. POST-HOC and descriptive: nothing here is a
registered verdict, and the pilot re-measures every one on the adopted checkpoint's own states.

| read | value [95 %] | what it settles |
|---|---|---|
| **VRF(terminal), S0, all 10,667 held-out pairs** | **1.057 [1.045, 1.072]** (ρ 0.059) | today's pairing is worth ~6 % fewer playouts |
| contested turns, all pairs (3,398) / top1\|top2 (122) | 1.086 [1.052, 1.125] / 1.125 [1.039, 1.221] | the first draft's prediction (1.0–1.3) holds |
| by class (contested): move-move / move-switch / switch-switch | **1.276 [1.194, 1.381]** (ρ 0.23) / 1.042 / 1.046 | some luck IS shared where the draw profiles match; almost none otherwise |
| derangement VRF (seed j vs j+1), all pairs | **1.000 [0.996, 1.005]** | the unpaired-variance estimator is unbiased from shared-seed data alone: the independent-seed control is redundant (Q4) |
| separable share at M = 32 (contested all pairs / top1\|top2) | 0.289 / 0.197 | the "ceiling" depends on the budget; 30 % of contested pairs resolve at 32 paired playouts |
| method-of-moments true-gap sd, win scale (contested all / top1\|top2) | 0.183 / 0.129 | real within-state spread exists; the top-2 pair is the hard column |
| **DECISION IMPROVEMENT over the policy's top-1, ALL legal actions, 468 eligible free turns** | **LEAF-K8 +0.062 [+0.045, +0.079]** · LEAF-K1 +0.025 [+0.012, +0.037] · CRITIC +0.008 [−0.005, +0.021] | the playouts carry decision value the critic lacks; even one rollout carries some |
| paired contrasts: LEAF-K8 − CRITIC / LEAF-K8 − LEAF-K1 | **[+0.038, +0.069]** / **[+0.025, +0.050]** | more rollouts DO still pay per decision (the owner's question is not moot) |
| by the policy's top-2 gap: the torn 40 % (187 turns) / the most decided 20 % (94) | LEAF-K8 +0.049 [+0.028, +0.073] / **+0.082 [+0.044, +0.125]** | the value is NOT concentrated on torn states: the policy's confident picks have larger stakes (mean truth spread 0.23 → 0.37 from the first quintile to the last). "Focus on torn states" is the wrong selector on this checkpoint |
| contested turns, 3 options (top1, top2, a hash-chosen random): pairwise accuracy CRITIC 0.609 vs LEAF-K8 0.604; improvement CRITIC +0.0001 vs LEAF-K8 +0.013 | descriptive (179 turns) | equal pairwise ACCURACY, unequal DECISIONS: the accuracy currency hides where the critic is wrong on what matters (Q3) |

**What these reads do NOT settle.** (1) The ORACLE world: an honest leaf (determinized) may lose much of the
+0.062; how much is the price of hidden information plus the determinizer's error (`program_rust_core.md` §6a's
oracle yardstick). (2) Off-distribution states: rb-blob's policy may be confidently wrong on ai_v14-lineage states,
inflating the improvement. (3) A per-decision improvement in the greedy continuation's Q is a one-step policy
improvement; how it compounds over a game against an opponent that adapts is a battery question. (4) The end-state
checkpoint. The pilot (§7) addresses (2) and (4) and stratifies by hidden information for (1).

**The reading that frames this doc (revised).** The critic READOUT is not label-limited, not loss-form-limited, not
width-limited and not (at the readout level) moving-target-limited (§6); whether the TRUNK is, under sibling-paired
targets, is untested. What separates siblings is in the PLAYOUTS, and part of what the playouts know is the
opponent's real team. So "buying discrimination" splits into two products with different economics:
- **A. a better VALUE FUNCTION** (training side): the bound is the representation / coverage of the states search
  queries. Rollout VOLUME into the same readout does not buy it (measured, one checkpoint). The levers are the
  representation (e), coverage, a trunk trained on sibling-level targets (j), and eventually a search-improved policy
  to evaluate (f).
- **B. a better SEARCH-TIME DECISION** (inference side): an oracle rollout leaf already improves decisions; its
  price is playouts per DECISION. Here the acting rule, truncation, control variates, candidate allocation and
  pairing are the levers, and the owner's "order of magnitude" is the wrong worry: the measured curve says the first
  rollouts buy the most.

---

## 3. The five error sources, and which lever addresses each

| source | what it is | levers that address it | evidence that it binds TODAY |
|---|---|---|---|
| **1. LABEL NOISE** | the training target is one game outcome: high variance | TD(λ) / bootstrapped targets; K-rollout labels; CRN-paired labels | **not binding for the readout** — K8 vs K1 refit no difference; λ arms not confirmed (§2.2). In the large-data limit a regressor's minimizer E[y \| x] is the same for a single outcome and a K-mean, so noise only costs samples; consistent with the null |
| **2. LUCK** (estimator variance) | a search-time or yardstick estimate of Δ from finite playouts carries dice variance (the opponent's team is fixed in the oracle world) | **an acting rule that spends noise wisely** (regret, not resolution); truncated rollouts; control variates (AIVAT-style); pairing (CRN, split streams); more rollouts (√n); candidate allocation (sequential halving) | **binding for the 2-SE acting rule** (5–7 % of pairs resolve at R = 4); **NOT binding for an argmax decision** at K = 8 (§2.3). Part of it is irreducible (§2.1 (ii)) |
| **3. REPRESENTATION / COVERAGE** (bias) | what `value_pooled` encodes about a one-ply difference; which states it was trained on (the successors of starved actions are rarely visited) | the end-state facts and depth (closing test; `probe_battery_2026-10-09` hunt list); a per-action value read (`design_q_head.md`); a trunk trained on sibling targets; training data on the states search queries | **the measured bound for the READOUT**: twelve ai_v12/v13 heads on 0.567–0.578 and the rb-blob refit null. UNVERIFIED for the trunk and the end state |
| **4. MOVING TARGET** | the critic chases π_t; its target V^{π_t} changes as π learns | freeze for measurement (policy evaluation); staleness meters (§6) | **not binding at the readout level** — readout fits on v13-W's frozen self-play match the live head (`offline_leaf_fit_2026-09-18/`). UNVERIFIED for a full fresh value NETWORK and for the end state |
| **5. HIDDEN INFORMATION** (added by the review) | the playouts (leaf AND yardstick) are in the TRUE world; the critic sees one side's view. In gen 3 the hidden part is IDENTITY and MOVESET, not material (`leaf_ceiling_controls_2026-09-19/` §4.4: 1.94 of 6 opponent mons unrevealed at the average successor) | determinization (DEFERRED, owner 2026-09-23); the X5 belief tokens as the sampler's proposal; for TRAINING, oracle labels are correct targets (a regressor on them converges to E[Q_true \| obs]) | **UNMEASURED.** Part of every leaf-minus-critic gap above is this, and no critic can close it. The pilot stratifies by the count of unrevealed opponent mons (§7.4 D2) as the free proxy |

---

## 4. The levers, with an expected cost per unit discrimination

**Assumptions** (back-of-envelope; MEASURED where marked):
- Outcomes on the win scale {0, 1}. The UNPAIRED difference of two actions' single outcomes has variance **≈ 0.28
  (SD ≈ 0.53) on contested Lane S pairs** (MEASURED, rb-blob: mean s²_a + s²_b = 1.11 on the ±1 scale); paired in S0,
  0.256 (SD 0.51).
- A pair with true gap Δ, estimated from K paired playouts with within-state correlation ρ, is ordered correctly with
  probability ≈ Φ(|Δ|·√K / (0.53·√(1−ρ))). The playouts needed for a fixed resolution scale as **K ∝ (1−ρ) / Δ²**;
  VRF = 1/(1−ρ) divides the bill.
- Worked numbers for a 2-SE RESOLUTION at 50 % power (|Δ̂| ≥ 2 SE when |Δ| = 2 SE): Δ = 0.18 needs K ≈ 32 paired;
  Δ = 0.05 needs K ≈ 420. 80 % power needs ≈ 2× more. **This is the cost of RESOLVING, which a decision does not
  need**: an argmax decision is wrong mostly where |Δ| is small, and there a wrong pick costs ≈ |Δ|.

| lever | attacks | price model | expected gain per unit cost | confidence |
|---|---|---|---|---|
| **(0) brute force**: more independent rollouts | luck | √n per pair. Per DECISION (rb-blob, all legal actions): K1 +0.025, K8 +0.062 improvement | the baseline every lever is priced against | MEASURED (K-curve; §2.3) |
| **(a) pairing / CRN** | luck in the DIFFERENCE | free in compute once the dice align; S1 ≈ 0.5–1, S2 ≈ 2–3 agent-days (§5.3) | S0 **MEASURED 1.06–1.13**; move-move 1.28. What S1 / S2 would reach is UNMEASURED; even a VRF of 2 halves K, and the cheaper levers below cut K or the rollout length more | S0 MEASURED (rb-blob); S1/S2 UNVERIFIED |
| **(b) bootstrapped targets (TD(λ))** | label noise (training side) | cheap (a target change) | **low**: the label is not the bound (§2.2). A bootstrapped target can also import the critic's own sibling blindness | MEASURED (indirectly) |
| **(c) focus the spend** (active learning) | where the budget goes | a selector | **a POLICY-gap selector is the wrong one on rb-blob**: the leaf's improvement is +0.049 on torn states and +0.082 on the most decided 20 % (§2.3). A STAKES selector (the critic's Q̂ spread, an ensemble's disagreement, a 1-rollout screen) is the candidate | REVIEW read (rb-blob); UNVERIFIED on the end state → **D2** |
| **(d) ensembles for uncertainty** | WHERE to spend rollouts (V's own doubt as a gate) | the X26 ride-along V ensemble (`--ridealong-ensemble K`) exists, OFF | a GATE for (c), not discrimination by itself. Whether disagreement predicts the leaf's improvement is untested | UNVERIFIED |
| **(e) representation** (the end state) | the critic's bias | already being paid (the closing test, ≈ 47 GPU-h) | the only lever with evidence against the measured readout bound. Its effect on SIBLING decisions has never been measured | UNVERIFIED → D1 on the adopted arm gives the first read |
| **(f) search-as-teacher** (the AlphaZero flywheel) | policy and critic together | needs a search that BEATS the policy first | the long-run product; not priced here | — |
| **(g) truncated / mixed leaf** (added) | rollout LENGTH and luck after the cut | d turns then the critic: cost ≈ d / 26 of a full rollout (a rollout runs ≈ 25.7 turns, `rollout_leaf_kcurve`); AlphaGo's mix (1−λ)·v + λ·z | if a d ≤ 4 truncated leaf keeps most of the improvement, it is a **6×–26× cheaper leaf** with less luck (luck after the cut is replaced by the critic's bias). Tesauro & Galperin 1996 found truncated rollouts "a favourable trade-off" in backgammon | UNVERIFIED → **R-TRUNC**, free in the pilot |
| **(h) luck-adjusting control variates** (added) | luck along each playout, no sibling alignment needed | branch-turn form: free (the per-turn value trace; X − β(V_{t+1} − its leave-one-out mean)); per-turn form: k extra one-turn dice samples per turn | AIVAT cut the SD of a poker match by 85 % (44× fewer games); GNU Backgammon's rollouts subtract each roll's luck as standard. Unknown here | UNVERIFIED → M1 reads the branch-turn form free; the per-turn form is a priced follow-up |
| **(i) the acting rule and candidate allocation** (added) | how the leaf's information becomes a decision | free (a rule change); sequential halving over the policy's top-k (Gumbel AlphaZero) spends the budget on contenders | the measured gap between "act on the argmax" (+0.062) and "act only on a resolved pair" (~0) is the largest single factor in this doc | REVIEW read (rb-blob) → **D1's PRIOR-k-K scorers** |
| **(j) sibling-level training signal for the TRUNK** (added) | representation | a per-action advantage head on CRN-paired DIFFERENCES (`design_q_head.md`); auxiliary one-ply consequence targets (KataGo-style); a trunk fine-tune on paired labels | the readout refits froze the trunk; the trunk's capacity to carry the within-state signal is the untested half of "the representation is the bound" | UNVERIFIED; GPU; its own registration |

**The answer to "do we really need 10×?" (revised).** For RESOLVING sibling pairs at 2 SE, yes: a near-tie needs
hundreds of playouts, and pairing in today's dice mode buys ~1.1×. **But resolution is the wrong target.** For the
DECISION a search makes, the banked read says 8 rollouts per action already buy ≈ +6 points of win probability per
decision over the policy (oracle world, off-distribution states, one checkpoint), 1 rollout buys ≈ +2.5, and the
critic ≈ 0. The binding constraints are the acting rule (it discarded the value), the information question (oracle
vs honest), and cost per decision (≈ 7 legal actions × K × ~26 turns; truncation attacks the 26). The pilot measures
the curve on the adopted checkpoint's own states with the cheaper leaf forms beside it.

---

## 5. The RNG-alignment investigation (from the code at `c1fae513`, re-verified by the review)

### 5.1 How the Rust engine draws randomness

| # | finding | where |
|---|---|---|
| R1 | **ONE PRNG per battle.** `BattleState.prng: Prng`, a port of Showdown's `sim/prng.ts`, pinned against **2,916** golden records from the real `prng.js` (NEXT, RANDN, RANDMN, CHANCE, SHUFFLE; `sample` and `random_float` are not covered directly). Two backends chosen by the seed prefix: `sodium,<hex>` (ChaCha20: one block per `next`, re-keyed from bytes 0..32, output bytes 32..36) and the gen-5 64-bit LCG (`gen5,<hex16>` or decimal `m,n,o,p`; output = the upper 32 bits). Every primitive takes ONE `next()`: `random(n)` = ⌊next·n / 2³²⌋, `random(m,n)` = ⌊next·(n−m) / 2³²⌋ + m, `random_chance(a,b)` = `random(b) < a`, `sample` = `random(len)` (even for len 1); `shuffle_range` takes k−1 draws for a group of k | `src/rust_sim/src/prng/{mod,gen5,sodium}.rs`; `state.rs` (`pub prng: Prng`); `src/rust_sim/tests/vectors/prng_golden.txt` |
| R2 | **Every die of BOTH sides comes from that one stream, in execution order.** 47 direct production call sites (46 on the driver path; the 47th is the legacy `run_turn`'s Quick Claw roll) plus 7 `speed_sort` sites (6 on the driver path). By purpose: ACCURACY (`turn/speed.rs`, every numeric accuracy, 100 % included, skipped only for never-miss, drawn even into an immune target); CRIT; DAMAGE ROLL `random(16)` (incl. Future Sight and the confusion self-hit); SECONDARY (incl. Tri Attack's chance and pick, King's Rock); MULTI-HIT count; DURATIONS (sleep, confusion, partial trap 3–6, Outrage/Thrash/Petal Dance, Uproar, Encore 3–6, Disable 2–5); ACTION CHECKS (thaw 1/5, confusion 1/2, attract 1/2, full paralysis 1/4, the Protect/Endure stall chance); PICKS (Sleep Talk, Conversion 2, stat berry, Effect Spore); the PHAZING pick; ITEM / ABILITY PROCS (contact procs, Cute Charm, **Focus Band on every damage event to its holder**, Shed Skin, Trace); QUICK CLAW; the `insertChoice` splice tie; the turn-0 gender draw; SPEED-TIE shuffles. No draw sites exist for Metronome, Assist, Mirror Move, Present, Magnitude, Spite or Bide (Psywave, Bide, Fly, Dig, Dive and Bounce panic as unsupported) | `src/rust_sim/src/turn/*.rs`, `event.rs` |
| R2b | 🚨 **QUICK CLAW is ONE battle-wide roll per turn, drawn whatever the items, at the END of turn t** (after residuals, the DisableMove / trap shuffles and `bump_active_turns`), and it sets turn t+1's order. So the BRANCH turn's roll was drawn from the RECORDED stream before any reseed, and is shared by every sibling | `turn/driver.rs` (the end-of-turn block before `bs.turn += 1`; turn 0's roll at the start) |
| R2c | 🚨 **SPEED-TIE shuffles reach far beyond the action queue**: `speed_sort` runs at every `eachEvent` (BeforeTurn, an Update after each action, switch-ins, residuals: 22 calls), the residual handler sort, and the end-of-turn DisableMove / trap shuffles. With the two actives at EQUAL speed, each of those draws a size-2 shuffle, whatever the actions, and their number depends on how many actions ran and whether a mon fainted. Two effect-handler pairs ALWAYS tie and draw with no speed tie at all: the SetStatus clause pair (one draw per status application) and the ModifyDamagePhase1 screens (one draw per hit with ≥ 2 screens up) | `turn/speed.rs` (incl. the always-tied pairs), `turn/residuals.rs`, `turn/driver.rs` |
| R3 | **PARITY-only DISCARDED draws** consume the stream with no effect: the `selfDrops` `random(100)` (Overheat / Superpower; defined in `residuals.rs`, called at move time), the non-Ghost Curse `selfDrops` `random(100)` (`status_moves.rs`), Rest's `slp.onStart` `random(2,6)` (`status_moves.rs`), Trace's `random(1)` (`event.rs`), the SetStatus clause and screen shuffles (their result is never used). And MOOT-value draws: the crit roll under Battle Armor / Shell Armor, Cute Charm on a same-gender attacker, a secondary behind a Substitute, `sample` with n = 1. Even a no-op mechanic shifts the stream | same |
| R4 | **Training AND eval battles use the LCG backend.** The env core stages `ep_seed` as four 16-bit words and joins them into the decimal `m,n,o,p` form; the Rust eval core's seeds are four 16-bit words too. ChaCha20 (sodium) appears only for a seedless bridge START and for the truth tool's seeds | `src/rust_env/src/core/pool.rs`; `src/agents/training/rust_eval/seeds.py` |
| R5 | **Counterfactual tooling REPLACES the whole stream at the branch point** — except the training fork arm. `BridgeSession::reseed(seed)` sets `st.prng = Prng::new(seed)`; the playout tool reseeds each branch before any feed (`Game::branch(Some(seed))`). A root at a divergence turn sits at the START of turn T, before any turn-T choice: the branch turn's own dice come from the new seed (its Quick Claw roll does not, R2b). The truth tool roots at a COMMAND INDEX (`branch_turn` → `cmd_index`); whether the other side's turn choice is the recorded one or answered by the greedy policy depends on the log's command order (`root.other_open`); either way it is identical across siblings. `search_driver`'s `reroll` / `reroll_many` rebuild each arm in a FRESH session from turn 1 and reseed at turn start, resolve ONE turn only, and pick forced switches inside it with `AuxRng`. The persistent search tree clones the node and reseeds unless the seed is `"original"`. **The training fork arm keeps the battle's own stream (seed `null`)**, still shared from the branch. The truth tool's seeds are `turn_seeds(id, S)` = `sodium,` + 32 hex of sha256, NESTED (seed k depends only on k) | `src/rust_sim/src/bridge.rs`; `src/rust_env/src/search/{game,playout}.rs`; `src/rust_sim/src/bin/search_driver.rs`; `src/main/policy_spectrum/truth.py`; `src/agents/training/rust_rollout/fork.py` |
| R6 | **Policy-SAMPLING draws are already counter-based**: a pure function of (run seed, stream = 0 for p1 / 1 for p2, env, episode, the per-SIDE decision ordinal `dec_n`) via a splitmix64 chain, inverse-CDF on 53 bits over the legal log-probs. Siblings that share the parent's keys consume the same uniform at the same `dec_n`; `dec_n` slips when one branch has an extra decision (a forced switch after a faint). `play_out` itself has no sampler (its policies are `greedy` and a numpy `uniform_random`) | `src/agents/training/keyed_draw.py`; `designs/training/forks.md` §14.3 |
| R7 | The search driver's follow-up `AuxRng` (FNV-1a of the seed string → mulberry32) is a SEPARATE stream from the battle's dice, used by its `random` follow-up policy and `reroll`'s forced switches. It does not touch a policy playout | `src/rust_sim/src/search.rs` |
| R8 | **Hidden information (added).** Every playout tool plays the TRUE world: `truth.py`'s `to_log` passes both real teams and the engine replays the real input log; each side's policy sees only its own one-sided rows. No determinization exists in the current tools (it lived in the deleted `main.search_dividend`, `designs/deleted_flags.md`) | `src/main/policy_spectrum/truth.py`; `src/rust_env/src/search/game.rs` |

### 5.2 The consequence: sibling alignment decays at the first differing draw (now MEASURED)

Two siblings that share a seed start the branch turn with the same stream. They stay aligned only while they consume
the SAME draws in the SAME order. That breaks almost at once:
- **a damaging move consumes several draws (accuracy, crit, roll, any secondary and procs); a switch usually none**
  (it can draw: Trace, a splice tie, a speed-tie shuffle in an Update, Pursuit), so move vs switch misaligns the
  opponent's dice in the branch turn itself;
- **two moves differ in draw count** when one has a secondary (Ice Beam's freeze roll vs Surf's none), a multi-hit
  count (each hit then draws crit, roll, secondary and procs, stopping at a faint), never-miss accuracy, or a
  fixed-damage effect (accuracy only); a secondary is drawn even into a KO or behind a Substitute (only Shield Dust
  skips it); Focus Band draws on every damage event to its holder;
- **ORDER changes**: a switch or a priority move goes first, so the opponent's draws come from different positions.
  Worse than independence in principle: a draw that was OUR luck in one sibling can be the OPPONENT's in the other
  (an anti-correlating coupling); measured on balance it is ≈ independent (below);
- **in gen 3 singles ANY faint ends the turn's actions**: Showdown cancels every remaining move AND switch of the
  non-fainted actives and skips to residuals (`battle.ts`, the `gen <= 3 && singles` branch; the port's
  `turn/driver.rs` `cancel_active_actions`), so a KO in one sibling removes every later draw of that turn;
- **speed ties and the always-tied handler pairs** (R2c) draw a number of shuffles that depends on how many actions
  ran, and the R3 parity draws make even a no-op difference shift the stream.

After the shift, sibling A's draw i is sibling B's draw i + δ, effectively independent for ChaCha20 or the LCG. **The
review measured it** on the refit's banked rows (§2.3, rb-blob, `sodium,` seeds, greedy): **VRF(terminal) = 1.057
[1.045, 1.072] over all pairs, 1.086 on contested pairs, 1.125 on top1|top2, and 1.276 [1.194, 1.381] on move-move
pairs** (ρ 0.23) against ≈ 1.04 on move-switch and switch-switch. So the pairing the tooling calls "COMMON RANDOM
NUMBERS" buys ~6–13 % in today's mode, and the only class with materially shared luck is the one whose draw profiles
match longest. Whether ALIGNED dice would share more luck (S1 / S2's premise) is UNMEASURED: the move-move ρ of 0.23
is the hint that some would be shared.

### 5.3 The proposal: two counterfactual-only dice modes (DEMOTED to conditional on R-PAIR)

**S0, the status quo, is unchanged and byte-identical.** Two new modes are selected ONLY by a new seed PREFIX that the
training path cannot produce:

| mode | seed prefix | what it does | aligns | build |
|---|---|---|---|---|
| **S0** (today) | `sodium,` / `gen5,` / `m,n,o,p` | one stream per battle | up to the first differing draw | — |
| **S1 per-turn reseed** | `crnturn,<hex>` | at each turn BOUNDARY the battle's stream is replaced by a fresh one seeded `H(base, t)`; within a turn it is one stream | every turn boundary: a misalignment in turn t cannot leak into turn t+1. Within a turn it still breaks at the first differing draw, so S1 pays only where the siblings' later turns make the same draws (same actions, same draw profile) | ≈ 0.5–1 agent-day: one hook, the prefix, the gates below |
| **S2 split, counter-keyed** | `crnsplit,<hex>` | every draw is `u = PRF(base, key)`, a 32-bit integer, with the key unique within the battle (below); each primitive maps u exactly as R1 maps `next()` | every draw of a given key, whatever the other draws did | ≈ 2–3 agent-days: a key at every call site (47 + 7), the counter table in `BattleState`, the PRF, the gates |

**Where S1's hook goes (corrected).** `bs.turn` increments at the end of turn t right after the Quick Claw roll
(turn 1 is the exception: it increments at commit). Every draw between turns (residuals, a forced replacement after
a move- or residual-faint, the DisableMove / trap shuffles, the Quick Claw roll) runs with `bs.turn == t` and
belongs to turn t, including a mid-turn forced switch. A hook at commit (`Driver::feed_await_move`, before
`sort_actions`) would leave the Quick Claw roll that governs turn t+1 in turn t's stream, the one place two
misaligned siblings could still differ. So: **reseed BEFORE the end-of-turn Quick Claw roll** (and handle turn 0's roll
at the start), or give the Quick Claw roll its own key. Replies and one-sided feeds draw nothing.

**S2's keys (refined; the brief's "is (turn, purpose, side) enough?").** In gen-3 SINGLES the side's active IS the
actor, a single-target move's target is the other side, and a per-key counter k orders multi-hit hits, so (turn,
purpose, side, k) is LAW-safe (keys never repeat). For COUPLING quality, three refinements:
1. **Purposes split by SOURCE**, not by mechanic alone: the move's own ACCURACY / CRIT / DAMAGE_ROLL / SECONDARY /
   MULTIHIT vs the confusion self-hit's roll vs each ability / item proc (Focus Band, King's Rock, Shed Skin, Effect
   Spore, Cute Charm, contact procs, Trace) vs the stall chance vs each pick. Otherwise a confused self-hit in one
   sibling shares its u with a move's damage roll in the other (harmless to the law, poor coupling).
2. **Durations keyed by (purpose, the mon that receives it, occurrence index)**, not by turn: "the sleep this mon
   gets lasts 3 turns" is then shared even when the siblings inflict it on different turns.
3. **Shuffles keyed by (turn, the event kind, k)**: the `eachEvent` Update shuffles, the residual sort, the end-of-turn
   shuffles, the always-tied handler pairs and the action-queue tie each get their own counter, so an extra Update in
   one sibling does not shift the residual-sort tie. The PARITY draws keep their own tags so they shift nothing.

"Actor" adds nothing beyond side in gen-3 singles, and "target" adds nothing except for a self-targeting effect
(covered by the source split). Mon-keying ACTION luck (accuracy, crit, roll) instead of side-keying is the open
empirical question (§10 Q5); it waits until S1 shows that aligned luck is shared at all.

Design rules for both modes:
- **Comonotone coupling.** With a shared u and the same inverse-CDF map, "both branches got a high roll" is the
  coupling that maximises the positive correlation of luck (the textbook CRN with inverse transforms). It cannot
  exceed the Fréchet bound for two Bernoullis with different p, so even perfect alignment cannot give ρ = 1 on a pair
  with a real gap.
- **Stream backends IGNORE the tag**: the default path calls `next()` in the same order as today, so S0 is
  byte-identical BY CONSTRUCTION, not by test alone. The single choke point is `Prng::next` (`prng/mod.rs`); S2
  replaces the typed entry points `random_below`, `random_range`, `random_chance`, `sample`, `shuffle_range`.
- **The PRF output is a 32-bit integer** mapped as R1 maps `next()` (⌊u·n / 2³²⌋), never the keyed policy draw's
  53-bit float; `shuffle_range` takes k−1 keyed draws.
- **The policy-sampling draws in a sampled continuation** move to the same keying: (turn, side, k) instead of the
  per-side `dec_n` (R6), so a forced switch in one branch does not shift the other's draws.
- **Never the training path.** `Prng::try_new` (the bridge's START and the env core's seed path) REFUSES the `crn*`
  prefixes; only a new `Prng::counterfactual(seed)` constructor, reached from `reseed` in the playout / search APIs,
  accepts them. The env core's `ep_seed` is four integer words, so it cannot spell one anyway. Showdown's `setSeed`
  throws on the prefix, so no log made under it can be mistaken for a Showdown battle.

### 5.4 Statistical equivalence, and the gates

**The argument.** In S0, a battle's dice are a sequence of 32-bit outputs treated as i.i.d. uniform. In S1 and S2,
every draw is a PRF output on a key no other draw of that battle uses (distinct t per stream in S1; a unique key in
S2), and which key comes next may depend on earlier outputs, which does not matter for a random function: outputs on
distinct, adaptively chosen keys are still i.i.d. uniform. Under the PRF assumption the draws a single battle
consumes are i.i.d. uniform, and the mechanics map them identically. **The law of one battle is therefore the same in
all three modes**, up to the backend's own quality (training's LCG is a weaker generator than ChaCha20; the truth tool
already uses ChaCha20, so the yardstick and training already differ in backend at that level). What changes is only
the JOINT law of SIBLINGS, which is the point. Two caveats: a key collision inside one battle would correlate two of
its draws (prevented by construction, and a gate asserts no repeat); and a batch of siblings is a correlated sample,
so sibling playouts are never used as if independent. The continuation POLICY is a separate choice from the dice
(greedy here); "identical to normal play" is a claim about the dice only.

**The gates** (each must FAIL on revert):
1. **S0 byte-identity**: the existing differential ladder, the obs golden (`golden_obs_core --check`), the K9 golden
   and the A/B fuzzers, unchanged, all on S0 seeds.
2. **Uniformity of the raw draws (added; tests the actual assumption)**: log every u a fixed set of N battles draws
   under S1 / S2 and test it directly (a χ² on 256 bins, lag-1 serial correlation, a no-repeated-key assertion), with
   bars set before the first run and wide enough (≥ 4 SE, Bonferroni over every statistic) that an equal law fails by
   chance with negligible probability. Fixed seeds make the verdict deterministic; a statistic within a rounding error
   of its bar counts as a FAIL (rule 8).
3. **Law equivalence downstream**: N greedy self-play battles per mode from the same starting set, with registered
   bars on the win rate and on per-purpose event rates (crit, miss, damage-roll histogram, sleep and confusion
   durations, speed-tie splits), under the same bar rules as gate 2. Gate 2 is the sharp test; this one checks that the
   mechanics consume the new draws as they consumed the old.
4. **Tag coverage** (S2): the untagged draw API is REMOVED, so an untagged site does not compile.
5. **Alignment**: two branches with different root actions; for every key both consume after the branch, the
   uniforms are equal (S2), and every draw of turn ≥ t+1 is equal when both reach it with the same draw sequence (S1).
6. **Determinism**: a `crn*` seed reproduces its battle byte for byte.
7. **Refusal**: the bridge START and the env core refuse a `crn*` seed with a typed error.

**Risk.** S2 touches ~54 sites in the parity-critical core (an `opus-high` / `opus-xhigh` unit per the agent rules).
The risk to production is contained by gate 1 and the tag-ignoring backends. The risk to the measurement is a law
break in the new mode, which gates 2–3 catch only at their N; that is the declared limit.

---

## 6. The MOVING TARGET: do we need a plateau?

**For MEASUREMENT: no.** Freeze a policy snapshot and do POLICY EVALUATION: the pilot's truth is "this action, then
the frozen policy's greedy play on both sides", a stationary target by construction. That is also what a search
consumes at play time: the leaf is the CURRENT policy's value, not an average over training.

**For TRAINING a better value function from playouts: measure how much target we chase before deciding.** Three
meters, all CPU, all offline on banked snapshots of one lineage:

| meter | what it reads | status |
|---|---|---|
| **MT1 policy drift** | per-snapshot KL, margin-bucketed flips and action mix vs the previous, the 10M-back and an anchor snapshot (`main.policy_drift`) | BUILT |
| **MT2 critic STALENESS** | the critic of checkpoint t scored on games played by π_{t+k} (k = 1, 2, 5, 10M steps): global AUC, Brier, sibling accuracy and DECISION IMPROVEMENT, against the critic of t+k on the same games. The staleness curve and its half-life k½. It sets how fresh a label must be to be worth anything in the loop | to build (≈ 0.5 agent-day, B6) |
| **MT3 frozen-policy readout vs the live critic** | a readout refit (trunk frozen) on π_t's own games vs π_t's live head | **measured at the READOUT level on v13-W only** (`offline_leaf_fit_2026-09-18/`: seven fits on 17,904 branched positions of arm W's frozen self-play, 0.567–0.576, matching the live head). The readout-level null REPLICATES on rb-blob in another form (the label refit). Pre-Rustboro for the frozen-policy form; **UNVERIFIED for the end-state critic and for a trunk** |

**Reading.** At the readout level the moving target is not what binds sibling ranking. The untested variant is a full
fresh VALUE NETWORK (its own trunk) trained to convergence on frozen-policy data (stationary, abundant): a GPU job, not
in this pilot (Q6 prices it).

---

## 7. THE PILOT (DRAFT pre-registration, house style; REVISED by the review)

Registered when this section's commit lands before the first playout, with a ledger REGISTRATION entry. Amendments go
in §7.10, dated, each saying whether any outcome had been seen. **Disclosure, required by the house style:** the
reviewer saw the §2.3 reads of BANKED rb-blob rows before revising this section; no row of the pilot's checkpoint,
states or seeds had been played or read.

### 7.1 The question

On the adopted end-state checkpoint's own states: (1) how much does each scorer's argmax improve on the policy's own
pick (the decision currency), and how does that grow with rollouts per action? (2) Does a TRUNCATED leaf (a few turns,
then the critic) keep most of a full leaf's improvement at a fraction of its cost? (3) How much luck does pairing
remove in today's dice mode as a function of how far past the branch it is read, and is there an alignment-loss
signature that S1 could recover? (4) Where is the improvement (by the policy's gap, by outcome stratum, by hidden
information)? (5) What does a decision cost?

### 7.2 What is fixed

| item | value |
|---|---|
| checkpoint **CP** | the closing test's ADOPTED arm, seed 2001's final (`design_endstate_closing_test.md` §3.3's verdict, §6): on PASS `rb_es_end2_s2001`, on FAIL `rb_es_prod_s2001`. READ-ONLY, loaded on CPU through the strict loader (`main.policy_spectrum.reader.load_checkpoint`; `qhat.check_winprob` passes), with `CUDA_VISIBLE_DEVICES=` empty (the CUDA-context lesson of `crn_label_refit` Amendment 2) |
| states | a bank built from CP's own Rust eval traces (each carries `<prefix>_reconstruction.json`, the input log) through `main.policy_spectrum.bank`, re-encode gate ① checked. **Fallback**, chosen and recorded BEFORE any outcome: the committed Lane S bank `m5_laneS/bank_v1/`, stamped OFF-DISTRIBUTION for CP. The eval quota is loss-enriched: the per-outcome stamp and the cycle's `eval_manifest.json` selection are carried; D1 is reported raw AND re-weighted to the natural win/loss rate (both registered; the raw one decides) |
| eligible | a free move round (not a forced switch), turn ∈ [2, 40], ≥ 3 legal actions, gate ① passed |
| **STRATA** | CP's top-2 masked-logit gap over the eligible pool, in QUINTILES (edges computed once over the whole eligible pool, recorded before any playout). A state within 1e-6 logits of an edge is EXCLUDED (rule 8). **TORN** = quintiles 1–2 (gap strictly below the 40th percentile), kept as the legacy stratum |
| sample | **600 states: 120 per quintile**, at most 2 per battle, in `sha256("value_disc_pilot_2026:" + decision_id)` order within each quintile |
| options | **ALL legal actions** (the truth tool's construction; ≈ 7.1 per eligible turn on Lane S). `top1` / `top2` by CP's masked logits at the root |
| continuation | CP GREEDY on both sides to the end (`utils.rust_env.successors.play_out`, `stall="production"`, `max_turns` 999), rooted as `truth.branch_turn` roots it (the decision's command index; the other side's root decision, where still open, answered by CP's greedy policy, identically across siblings; `root.other_open` recorded). Value: +1 win / −1 loss (a stall forfeit included) / 0 tie or truncated, from the searched side. Greedy makes the dice the only luck. **World: ORACLE**, stamped |
| **dice** | seeds j = 0..31, `truth.turn_seeds(id, 32)` (nested), every option on every seed. **No independent-seed arm** (the derangement estimator replaces it; §2.3) |
| **value trace** | for every (option, seed): CP's win-prob V at the searched side's FIRST decision of each turn after the branch (turn offset d = 1, 2, …), plus every forced-switch decision with its turn, and the terminal; after the end, X_d is the terminal value (absorbing). One forward gives logits and V (`qhat.logits_and_values`) |
| critic successors | `qhat.one_ply` on 8 SEPARATE seeds (indices 32–39 of `turn_seeds(id, 40)`): Q̂(a) = the mean over successors of 2·σ(head) − 1, a branch that ends scoring its terminal |
| truth | T(a) = the mean of `o_a` over seeds 8–31 (24 seeds). Every leaf scorer uses seeds 0..K−1 only (K ≤ 8) and never sees 8–31; the critic uses 32–39 |
| scorer ties | an argmax tie (\|Δscore\| < 1e-6) breaks toward the policy's own order (top1 first), deterministically |
| CIs | battle-clustered bootstrap, 2,000 draws, `numpy.random.default_rng(20261011)`, percentile 95 %; every contrast PAIRED on the same draws |

### 7.3 Why these choices (the revisions)

- **All legal actions, not three.** The decision a search makes is among all of them, and the banked read shows the
  difference matters: restricted to {top1, top2, rand} the leaf's improvement is +0.013, over all legal actions
  +0.062. It also makes the starved near-best moves (the X4 blind spot) part of the read.
- **Stratified by the gap, not restricted to torn states.** The banked read finds the leaf's improvement LARGER on the
  policy's most decided states (+0.082) than on its torn ones (+0.049). A torn-only sample would mis-price lever (c).
- **No independent-seed arm.** Its 20 % buys states instead; the derangement estimator is validated (§2.3).
- **Truncated leaves read from the value trace**: free, and the cheapest lever if it works.

### 7.4 The metrics

| meter | definition |
|---|---|
| **D1, DECISION IMPROVEMENT (PRIMARY)** | IMP(s) = mean over states of [T(â_s) − T(top1)] / 2 (win scale), â_s = the argmax of scorer s over all legal actions. Scorers: **CRITIC** (Q̂); **LEAF-K** = the mean outcome over seeds 0..K−1, K = 1, 2, 4, 8; **TRUNC-d-K** = the mean over seeds 0..K−1 of X_d (d = 1, 2, 4 turns), K = 1, 4, 8; **MIX-K** = ½·Q̂ + ½·LEAF-K (AlphaGo's λ = 0.5), K = 1, 4, 8; **PRIOR-k-K** = LEAF-K's argmax restricted to the policy's top-k, k = 2, 3, K = 4, 8; and the policy itself (IMP = 0 by definition). The ORACLE upper bound max_a T(a) − T(top1) is reported with its winner's-curse stamp |
| **D2, WHERE** (descriptive) | D1 for LEAF-K8, TRUNC-4-8 and CRITIC by gap quintile; by outcome stratum; by the number of opponent team slots never revealed at the root in the searched side's one-sided view (0–1 / 2–3 / 4–5: the free proxy for how much of the leaf's edge is hidden information); by game phase (turn 2–10 / 11–25 / 26–40) |
| **M1, PAIRING** (mechanism) | for each pair (a, b) of a state and turn offset d: s²_pair(d) = the sample variance over seeds of X_d(a,j) − X_d(b,j); s²_unp(d) = s²_a(d) + s²_b(d). **VRF(d) = Σ s²_unp / Σ s²_pair** (a ratio of sums); ρ(d) = Σ cov / Σ √(s²_a s²_b). Reported at d = 1, 2, 3, 5, 8 and the terminal, by pair class (move-move same draw profile at the branch turn where the trace can tell, move-move, move-switch, switch-switch). Note: X_d for d < terminal is the CRITIC's value, a smoother variable than the outcome, so ρ(d) is the comparable quantity across depths, not VRF(d). **Validity:** the DERANGEMENT VRF (b's seed j+1 mod 32 against a's seed j) must have a 95 % CI containing 1.0 on the terminal, or M1 is VOID. Also the **branch-turn control variate**: the variance of o(a,j) − β·(X_1(a,j) − X̄_1(a, −j)) (leave-one-out mean; β fitted on the other half of the states) over that of o, the free AIVAT-style read |
| **M2, the GAP DISTRIBUTION** (descriptive) | the method-of-moments true-gap SD (var(Δ̂_32) − mean s²_pair / 32), the separable share at M = 32 (paired), SPLIT-HALF agreement (seeds 0–15 vs 16–31, pairs non-tied in both halves) |
| **M3, PAIRWISE ACCURACY** (descriptive, continuity with UNDERSTANDING rule 25) | against the 24-seed truth on non-tied pairs (Δ is a multiple of 1/24; an exact zero is a TIE and is EXCLUDED, rule 8): CRITIC, LEAF-K1 / K4 / K8; all pairs and top1\|top2 |
| **M1b, the OLD ACTING RULE** (descriptive) | the share of top1-vs-alternative pairs resolved by `|mean d| ≥ 2·SE`, n ≥ 4, SE floored at 0.5/n on the win scale (the deleted `search_dividend/playoff.py` rule, re-implemented in `score.py`) at R = 4, 8, 16, 32, and the IMP of "act only when resolved" |
| **M4, COST** | CPU-s per playout and per simulated turn (the job's cgroup `cpu.stat`), playouts per state, and the derived CPU-s per DECISION for each scorer (LEAF-K: legal × K × the mean rollout length; TRUNC-d-K: legal × K × d) |

### 7.5 Decision rules (registered)

A bound within ±0.0005 of its boundary is a BOUNDARY result and counts as not clear (rule 8).

**R-LEAF (PRIMARY), on D = IMP(LEAF-K8) − IMP(CRITIC), paired:**

| outcome | condition | implies |
|---|---|---|
| **LEAF-AHEAD** | lower bound > +0.0005 | the playouts carry decision value the critic lacks on CP's own states (oracle world). Training side: the representation and a sibling-level trunk signal (lever (j)) are the route, not more labels of today's kind. Inference side: R-TRUNC picks the leaf's form; an offline battery needs the owner (Q1) and an honest-world read needs determinization (deferred) |
| **CRITIC-EQUIVALENT** | the CI lies inside ±0.01 | the end-state critic decides as well as an 8-rollout oracle leaf: rollout leaves are not worth building on this checkpoint; search can use the critic directly |
| **NOT DETECTED** | otherwise | nothing is concluded; report |

**R-TRUNC (secondary), on the cheapest leaf that keeps the value:** the smallest d ∈ {1, 2, 4} whose TRUNC-d-8
satisfies BOTH IMP(TRUNC-d-8) − IMP(LEAF-K8) lower bound > −0.01 (non-inferior within 1 point of win probability per
decision) AND IMP(TRUNC-d-8) − IMP(CRITIC) lower bound > +0.0005. If one exists, it is the recommended leaf (cost ≈
d / mean rollout length of a full leaf) and S1 / S2 are NOT built (pairing pays mainly on full-length rollouts). If none
does, the full leaf stays the reference and R-PAIR decides the RNG build.

**R-K (secondary, the owner's "10×" question), on IMP(LEAF-K8) − IMP(LEAF-K4), paired:**

| outcome | condition | implies |
|---|---|---|
| **SATURATING** | upper bound < +0.01 | past 4 rollouts per action, more buys < 1 point per decision: spend compute on breadth (more decisions searched), not depth |
| **STILL CLIMBING** | lower bound > +0.0005 | more rollouts still pay at 8; extrapolate on √K and price the next doubling against R-TRUNC's leaf |
| **UNRESOLVED** | otherwise | report the K-curve |

**R-PAIR (secondary, gates the RNG build only when R-TRUNC found no truncated leaf), in S0:**

| outcome | condition | implies |
|---|---|---|
| **ALIGNMENT-LOSS SIGNATURE** | ρ(1) over all pairs has lower bound ≥ 0.30 AND ρ(terminal) has upper bound < 0.20 | the luck IS shared while the dice align and lost after: **build S1**, re-run the leaf scorers on the SAME states and seeds (Phase 2). Build S2 only if S1's ρ(terminal) upper bound is still < 0.20 |
| **NO SIGNATURE** | otherwise | do not build split streams; pairing is not a lever on this engine at this depth |

### 7.6 Prediction (written before measurement; the banked rb-blob reads of §2.3 are its basis)

- IMP(LEAF-K8): **+0.02 to +0.06** (rb-blob +0.062 off-distribution; CP's own states and a stronger policy should
  leave less to find). IMP(CRITIC): **−0.01 to +0.02**. **R-LEAF: LEAF-AHEAD.**
- R-K: **UNRESOLVED or STILL CLIMBING** (rb-blob K8 − K1 +0.037 [+0.025, +0.050] across a factor 8).
- R-TRUNC: d = 4 non-inferior, d = 1 not (UNVERIFIED guess: one turn of dice is a small share of a 26-turn game's luck,
  but most of the decisive consequences of a move land within a few turns).
- D2: the improvement does NOT fall toward the decided quintiles (rb-blob: it rises), and it shrinks where more of
  the opponent is revealed (a guess about the information share).
- ρ(terminal): **0.05–0.12** (rb-blob contested 0.088); ρ(1): 0.2–0.5. **R-PAIR: NO SIGNATURE** (a guess).
- Cost: 0.10–0.40 CPU-s per playout (the end-state network is larger than the blob arm's).

### 7.7 Compute budget, and where it runs

- Playouts: 600 states × ≈ 7.1 options × 32 seeds ≈ **136,000**, plus ≈ 34,000 one-ply successors (cheap). At
  0.10–0.40 CPU-s ⇒ **≈ 4–15 CPU-hours**.
- **CPU only. No GPU, no lease.** CP's forward runs on CPU (the playout loop batches every pending decision of a root
  into one forward).
- Standing rule 5's heavy-job clause: under `scripts/ops/mem_cap.sh`, `nice -n 19`, ≤ 2 heavy jobs at once, detached
  (`setsid nohup`), durable rows (one fsync'd JSONL row per state), resumable by state id, kill-and-resume tested once
  on a pilot subset (the `crn_label_refit` pattern). The live run's trainer and Rust core keep priority.
- "Small GPU slices" belong to the LABEL FACTORY at scale (a T2-batched forward makes a playout's sim + encode the
  cost; an UNVERIFIED estimate is ~10 ms of CPU per playout vs ~100–250 ms with the CPU forward). Not needed here.

### 7.8 Size rule

A throughput pilot (≤ 20 states, throughput only, no outcome read) measures CPU-s per playout on CP. If the full job
exceeds **12 CPU-hours**, states are dropped in reverse hash order EQUALLY from every quintile until it fits (never
below 80 per quintile; below that the pilot runs at 400 states and says so), the count recorded as an amendment before
any outcome is read.

### 7.9 Files (to be created under `designs/research_state/measurements/value_disc_pilot_<date>/`)

`README.md` (this pre-registration, copied, plus amendments), `make_states.py`, `run_playouts.py` (the value-trace
driver), `score.py` (D1–D2, M1–M4, the re-implemented acting rule, the bootstrap, the verdicts), `run_all.sh`,
`rows/`, `result.md`, `result.json`.

### 7.10 Amendments

**Amendment 1 (2026-10-11, the REGISTRATION; written and committed before any playout outcome was played or read;
the only playouts run were a 4-state throughput pilot that printed cost alone).** The owner's brief (2026-10-11:
"you have my permission to make any decisions to gather data that would be useful for design or our roadmap")
widens the pilot. The full text, with every number of the state build, is
`research_state/measurements/value_disc_pilot_2026-10-11/README.md` §2–§5; the changes to §7.2–§7.8:

1. **Two checkpoints, not the adopted arm alone:** `rb_es_end2_s2001/final_model.zip` (pin `62085750`) and
   `rb_es_prod_s2001/final_model.zip` (pin `95d014fa`), each loaded, encoded and played at its OWN pin's code, each on
   its OWN states; every rule is evaluated per checkpoint, and an end2-vs-prod difference is descriptive (different
   state sets). The closing-test verdict is no longer needed to start.
2. **States:** the last three eval-trace cycles of each run (its 10M / 12M / 14M snapshots), EVERY trace (not the
   bank's 2/2/1 quota; the loss-enriched selection is stamped and re-weighted). Gate ① passed on every recorded
   decision (end2 24,036, prod 23,392); the obs clock decodes to the protocol turn on all of them. top1 / top2 /
   the gap come from the FINAL checkpoint's root forward. 600 states per checkpoint, 120 per quintile, ≤ 2 per
   battle (end2 417 battles, prod 416).
3. **Seeds:** `turn_seeds(id, 48)`: **0–15 leaf, 16–39 truth (24), 40–47 the critic's one-ply successors** (was
   0–7 / 8–31 / 32–39), so sequential halving at matched budgets up to n·4 and a K = 16 point fit; the truth stays
   24 seeds that no scorer reads.
4. **Scorers added:** LEAF-K16; TRUNC-d-K at K = 1, 4, 8, 16 (X_d = V at the searched side's first decision with
   turn ≥ T + d, the terminal if ended); **CV-K, the AIVAT-style branch-turn control variate**: LEAF-K(a) −
   (mean over the K leaf seeds of X₁′(a, j) − Q̂(a)), with X₁′ = V at the searched side's first decision after the
   root (exactly what `qhat.one_ply` captures) and Q̂ the critic's independent one-ply estimate on seeds 40–47
   (unbiased for LEAF-K's mean; the per-turn AIVAT form is DEFERRED: it needs extra dice samples at every turn).
5. **Acting rules, defined before any outcome:** ARGMAX; PRIOR-k (k = 2, 3); **GUMBEL** (the prior-aware rule:
   Gumbel MuZero's final selection without noise, `mctx` defaults: argmax of logit(a) + (50 + N_max)·0.1·q̃(a), q̃
   the scorer min-max rescaled over the legal actions, N_max the per-action sample count); PUCT (descriptive:
   s(a) + 1.25·π(a)·√(n·K)/(1 + K)); **SH-m, sequential halving** (Gumbel AlphaZero's allocation, no noise) over the
   policy's top-m (m = 4 and m = all) at budgets B = n·K, K = 1, 2, 4, 8, matched to LEAF-K's spend: ⌈log₂ m⌉
   rounds, max(1, ⌊B / (R·|S|)⌋) new playouts per survivor per round (its leaf seeds in order, ≤ 16), halving by the
   GUMBEL score, the spent budget reported; **GATE-2SE** (the deleted rule: switch from top1 to LEAF-K's argmax only if
   K ≥ 4 and the paired mean d ≥ 2·SE, SE = max(sd/√K, 0.5/K), win scale).
6. **Decision rules added (secondary):** R-PRIOR on IMP(GUMBEL∘LEAF-K4) − IMP(ARGMAX∘LEAF-K4) and R-SH on IMP(SH-4 at
   B = n·2) − IMP(ARGMAX∘LEAF-K2): HELPS (lower > +0.0005) / HURTS (upper < −0.0005) / NOT DETECTED. R-LEAF (primary),
   R-TRUNC, R-K and R-PAIR stand as written. D1 adds strata by phase and by the trace's outcome.
7. **Cost:** CPU-s per decision from the forward rows (both sides) each scorer's playouts need at the state's
   measured CPU-s per row, and the critic's measured one-ply CPU.
8. **Size rule:** the throughput pilot measured 0.060 CPU-s per playout (end2, CPU forward, identical pending rows
   deduplicated), ≈ 15 CPU-s per state, ≈ 2.5 CPU-hours per checkpoint: no state is dropped.
9. **Phase 2 (determinized worlds)** is NOT part of this registration; if built it is registered by its own
   amendment before its first playout.

---

## 8. Build list

| # | item | exists? | cost |
|---|---|---|---|
| — | `play_out` (CRN branches × seeds, an `actions` subset, a host-driven policy; returns the END outcome per branch only) | ✅ `src/utils/rust_env/successors.py` | — |
| — | `turn_seeds`, `branch_turn`, the truth convention | ✅ `src/main/policy_spectrum/truth.py` | — |
| — | `one_ply`, `logits_and_values`, `check_winprob` | ✅ `src/main/policy_spectrum/qhat.py` | — |
| — | the turn bank from eval traces (re-encode gate ①) | ✅ `src/main/policy_spectrum/bank.py` | — |
| — | the torn threshold and the random-branch draw | ✅ `src/agents/training/fork_arm.py` (`contested_threshold`) | — |
| — | the pair / bootstrap / tie-exclusion scoring pattern; the decision-improvement and pairing reads | ✅ `crn_label_refit_2026-10-06/score.py`; `value_disc_review_2026-10-11/*.py` | — |
| — | ~~the playoff (acting) rule~~ | ❌ **DELETED** with `main.search_dividend` (P6 slice 6d-1, `8b5c7e4d`); `rollout_leaf_kcurve_2026-09-19/playoff_rule_offline.py` imports it and no longer runs. Its constants: `SE_MULTIPLE` 2.0, `MIN_PAIRS` 4, the SE floor 0.5/n (win scale). Re-implemented in B3 | in B3 |
| B1 | the pilot DRIVER: a policy wrapper that records V per (branch, seed, side, decision, turn) from the `who` array | ❌ | ≈ 0.5 agent-day |
| B2 | a bank SOURCE for CP's Rust eval traces (verify `bank.py` reads a core trace's reconstruction record); the unrevealed-slot count per root from the one-sided view | ❌ (verify) | ≈ 0.25 |
| B3 | `score.py`: D1–D2 (incl. the truncated / mixed / prior scorers from the trace), M1 (ρ(d), derangement, the branch-turn CV), M2–M4, the acting rule | ❌ | ≈ 0.75 |
| B4 | **S1** per-turn reseed (`crnturn,`; the hook before the end-of-turn Quick Claw roll) + gates 1–3, 5–7 | ❌, CONDITIONAL on R-TRUNC (none) AND R-PAIR (signature) | ≈ 0.5–1 |
| B5 | **S2** counter-keyed draws (`crnsplit,`; §5.3's keys) + gates 1–7 + a per-battle key-tagged draw log | ❌, CONDITIONAL on S1 | ≈ 2–3 |
| B6 | MT2 critic staleness meter | ❌ | ≈ 0.5 |

Pilot (B1–B3): ≈ 1.5 agent-days of build plus ≈ 1 wall-day of CPU at low priority.

---

## 9. Roadmap fit

- **After the closing-test verdict** (`design_endstate_closing_test.md`): CP is the adopted arm's seed 2001 final, so
  D1 is also the first DECISION-level read of the end state's critic (lever (e)).
- **Alongside the bf16 A/B**: that is GPU work; the pilot is CPU at `nice -n 19`. They do not compete for the lease.
- **The plateau-first funnel** (memory `project_plateau_first_funnel`): the pilot is a measurement, not a screen or a
  run; it blocks nothing and informs the search half of the north stars (memory `project_north_stars_2026-09-27`).
- **Order after the pilot**: R-LEAF decides whether rollout leaves are worth anything on this checkpoint; R-TRUNC
  picks the leaf's form and gates the RNG build with R-PAIR; R-K answers the owner's "10×" question on the decision
  currency. An offline battery with a playout leaf, and any IN-LOOP use, wait for the owner (Q1). Determinization (an
  honest leaf) stays deferred (owner, 2026-09-23) and is the named prerequisite for any of this on the ladder.

---

## 10. Open questions, with the reviewer's recommendation on each

1. **The one-ply scope rule (OWNER DECISION).** The owner deferred playouts to the end IN THE TRAINING LOOP
   (2026-09-30: in-loop additions ≤ one ply). This pilot is offline. **Recommendation:** the pilot needs no lift. On
   LEAF-AHEAD, ask the owner one narrow question: may an OFFLINE evaluation battery play a K-rollout (or R-TRUNC's
   truncated) leaf acting on its argmax, oracle world, stamped `world=oracle`? That is evaluation, not training, and it
   is the test of whether +k points per decision becomes win rate. Keep the TRAINING-side deferral (the label factory)
   until a registered, priced experiment exists; and say plainly that nothing transfers to the ladder before
   determinization, which the owner also deferred.
2. **Greedy vs sampled continuation.** **Greedy for the pilot.** It is the eval regime (greedy vs greedy), what a
   greedy search leaf evaluates, and it keeps the dice the only luck; a sampled continuation adds policy-draw luck
   whose keyed draws slip on `dec_n` (R6). A sampled arm belongs to the LABEL FACTORY if its target is the training
   ecology (T = 1 self-play), after S1 / S2 if those are built.
3. **Is VRF the right currency?** **No: DECISION IMPROVEMENT is the decision currency, and the pilot's primary.** VRF
   is a mechanism diagnostic, and pairwise accuracy weights a near-tie like a decisive pair (on the banked 3-option
   set the critic and the leaf have equal accuracy, 0.609 vs 0.604, while only the leaf improves the decision). M1b's
   resolved share stays as the reason the old arm did nothing.
4. **The independent-seed control.** **Drop it.** The derangement estimator reads 1.000 [0.996, 1.005] on 10,667
   banked shared-seed pairs, and per-option means are unbiased under shared seeds by construction; the 20 % buys
   states.
5. **S2's keying by SIDE vs by MON.** **Hybrid, no factorial yet:** side for per-action luck (accuracy, crit, roll,
   secondary), the receiving mon plus an occurrence index for DURATIONS, the event kind for SHUFFLES (§5.3). A
   side-vs-mon factorial for action luck waits until S1 shows aligned luck is shared at all (R-PAIR).
6. **A full fresh value NETWORK on frozen-policy data** (MT3's untested variant). **Not now; price ≈ 10–20 GPU-hours,
   UNVERIFIED:** frozen-policy greedy self-play to ~50M frames on the Rust core (training runs at ~1,700–1,900 fps
   with learning; inference-only collection should be faster), a value-only end-state trunk trained to convergence, and
   the pilot's bank as its read. Gate it on R-LEAF = LEAF-AHEAD. The cheaper first step is lever (j): a per-action
   advantage head or a trunk fine-tune on CRN-paired sibling differences, which attacks the untested half of "the
   representation is the bound" directly.
7. **The ceiling's estimator.** **Drop the ceiling as a decision input.** The decision currency needs none (it is
   unbiased against held-out truth). Register the method-of-moments gap SD as the descriptive "how much is there to
   find", report split-half agreement beside it, and stamp the oracle max − top1 as an UPPER bound (winner's curse). A
   deconvolved accuracy needs a model of Δ's shape, which the split-half deficit on top1|top2 says is not Normal.
8. **The torn definition.** **Keep the quantile, strict `<`, rule-8 exclusion at the edge; stop restricting to it.**
   A probability-ratio threshold is the same ordering (π₂/π₁ = exp(−gap)); the quantile is portable across checkpoints
   and fixes the stratum's share. The banked read says policy-torn is not where the leaf's value is, so torn becomes a
   STRATUM and the sample spans every quintile.
9. **State source.** **CP's own eval traces** (the regime where a search leaf would act: greedy vs pool and bots),
   with the selection weights recorded and D1 reported raw and re-weighted. Greedy SELF-PLAY states are the label
   factory's distribution, a second stratum when that experiment is registered. Lane S is the fallback only.

---

## 11. Related work (added by the review)

| work | what it offers here |
|---|---|
| **Common random numbers and synchronization**: Glasserman & Yao 1992 ("Some guidelines and guarantees for common random numbers", Management Science); L'Ecuyer et al. 2002 (RngStreams: many streams and substreams) | CRN's guarantee needs monotone responses AND synchronized draws; the textbook fix for synchronization is a dedicated stream per source of randomness. S1 is a substream per turn; S2 is a stream per source. Our measured ρ ≈ 0.06 is what un-synchronized CRN gives |
| **PEGASUS** (Ng & Jordan 2000) | fixing the random scenarios turns a stochastic policy comparison into a deterministic one; the same idea as shared seeds, with the same synchronization requirement |
| **Variance reduction in MCTS** (Veness, Lanctot & Bowling, NIPS 2011) | CRN, antithetic variates and control variates applied inside MCTS on stochastic games (Pig, Can't Stop, Dominion); the direct precedent for choosing among (a), (h) and antithetic draws (a free S2 add-on: u ↔ 1 − u) |
| **AIVAT** (Burch, Schmid, Moravčík & Bowling, AAAI 2018); DIVAT / MIVAT before it | a provably unbiased evaluator that subtracts, at each chance node and each known-strategy decision, the value change a heuristic value function attributes to it: −85 % SD on a heads-up no-limit hold'em man-machine match (44× fewer games). The template for lever (h): our critic is the heuristic, each turn's dice the chance node |
| **GNU Backgammon rollouts** (the manual's Rollouts chapter) | the dice-game practice: "variance reduction" cancels each roll's luck as the evaluation difference before and after the roll, and the first 2 plies use quasi-random (stratified) dice. Backgammon is the closest analogue to a dice-driven game with a neural evaluator and rollouts |
| **On-line policy improvement with Monte-Carlo search** (Tesauro & Galperin, NIPS 1996); Bertsekas's rollout algorithms | rollouts of a base policy pick the action with the best measured return: one-step policy improvement, which is exactly D1's estimand; TRUNCATED rollouts that end on the evaluator were a favourable trade-off in backgammon (lever (g)) |
| **AlphaGo** (Silver et al. 2016) | the leaf as a mix (1 − λ)·v + λ·z of the value network and a fast rollout, λ = 0.5 best: D1's MIX scorer |
| **Sequential halving** (Karnin, Koren & Somekh, ICML 2013); **Gumbel AlphaZero / MuZero** (Danihelka, Guez, Schrittwieser & Silver, ICLR 2022) | fixed-budget best-arm identification; with the policy's Gumbel-top-k proposing candidates, a policy improvement guarantee at a handful of simulations. The principled replacement for the 2-SE resolve-or-defer rule (lever (i); D1's PRIOR scorers are its crude form) |
| **KataGo** (Wu 2019, "Accelerating Self-Play Learning in Go") | playout cap randomization (full searches on p = 0.25 of turns, only those recorded as training targets) resolves the policy-vs-value data tension; auxiliary ownership / score / opponent-move targets, "predicting subcomponents of desired targets", sped training. The analogue for lever (j): one-ply consequence targets (HP, faints, status) per action |
| **TD(λ), n-step returns, GAE** (Sutton 1988; Schulman et al. 2016); **MuZero** n-step targets and Reanalyse (Schrittwieser et al. 2020) | bootstrapped targets trade variance for the critic's own bias; our λ arms did not move resolution, and a bootstrapped target can import the critic's sibling blindness |
| **Distributional and ensemble critics** (C51, Bellemare et al. 2017; QR-DQN, Dabney et al. 2018; bootstrapped DQN, Osband et al. 2016) | for a binary outcome a distributional head adds nothing beyond p; an ensemble's value is as a GATE for where to spend rollouts (lever (d)), and it must predict the leaf's IMPROVEMENT, not just disagree |
| **Dueling / advantage heads** (Wang et al. 2016) | separating V(s) from A(s, a) is the architectural form of a per-action sibling signal; a CRN-paired DIFFERENCE target is A's natural label (`design_q_head.md`) |
| **Foul Play** (determinized MCTS for Showdown; read in `design_x5_belief_tokens.md`) | the honest-world baseline a deployed leaf must match: sampled opponent worlds, not the true one |

---

## 12. Reviewer notes (2026-10-11)

### 12.1 What changed, and why (most important first)

1. **Measured the pilot's headline before the pilot.** The refit's committed rows ARE shared-seed playouts of every
   legal action: VRF(terminal) in S0 = 1.057 [1.045, 1.072] (contested 1.086, move-move 1.276). The draft's prediction
   holds; the headline is no longer an unknown.
2. **Changed the currency to DECISION IMPROVEMENT.** The banked read shows the oracle 8-rollout leaf improves on the
   policy's pick by +0.062 per decision while the critic buys ≈ 0, and that equal pairwise accuracy can hide unequal
   decisions. The draft's R-GAP (on accuracy and a budget-dependent "separable share") is replaced by R-LEAF; the
   owner's "10×" question becomes R-K on the decision currency.
3. **Named the ORACLE-world asymmetry** (§2.1, §3 row 5, R8). Every rollout number here plays from the true teams;
   the critic sees one side. Part of every leaf-minus-critic gap is information no critic can have; an honest leaf
   needs determinization, deferred by the owner.
4. **Stopped restricting to torn states.** The leaf's value is not concentrated there on rb-blob (+0.049 torn vs
   +0.082 most decided); the sample is stratified by the gap and lever (c) re-priced.
5. **Added the cheaper levers the draft missed**: truncated / mixed leaves (free from the value trace), AIVAT-style
   control variates (the branch-turn form free), a regret-aware acting rule and candidate allocation, and a
   sibling-level TRUNK signal. S1 / S2 are demoted to conditional on R-TRUNC finding no truncated leaf.
6. **Corrected the RNG facts** (verified file by file): 47 + 7 sites, not ~49 + 6; Quick Claw is one battle-wide
   roll at the END of the previous turn (moves S1's hook, and the branch turn's roll is the recorded one); speed-tie
   shuffles at every `eachEvent` plus two always-tied handler pairs; the R3 list completed; secondaries drawn into a
   KO; ANY faint ends the turn's actions in gen-3 singles; the training fork arm does NOT reseed; `reroll` resolves one
   turn; `n` is the per-side `dec_n`; the eval core is LCG too; the golden vectors do not cover `sample`.
7. **Softened the training-side claims to their provenance.** "More rollouts of the same kind do not help critic
   training" is, as measured, "a frozen-trunk READOUT refit on 8-rollout labels does not help, on one rb-era X5-blob
   checkpoint, 3,247 training actions, off-distribution states". The twelve-head band and the frozen-policy readout
   fit are ai_v12 / v13-W (pre-Rustboro, pre-X5); none of it is measured on the end-state critic or on a trunk.
8. **Corrected the "E|gap| ≈ 0.177, biased up by noise" reading**: it is noise-CORRECTED, and the Normal assumption
   behind E|gap| = sd·√(2/π) fails on the top-2 column (split-half deficit).
9. **Fixed the build list**: the playoff acting rule was DELETED (`8b5c7e4d`); its constants are recorded and B3
   re-implements it. Dropped the independent-seed arm. [Orchestrator correction 2026-10-11: rule 5b DOES exist in root CLAUDE.md (added `0cecbda7` / `38136597`, CPU work at `nice -n 19` uncapped while a run is live); the reviewer's worktree predated it] Fixed "rule 5b" to standing rule 5's heavy-job
   clause. Defined torn with the strict `<` the refit applied.
10. **Tightened S1 / S2's design and gates**: S2's keys split by source, durations mon-keyed, shuffles event-keyed, a
    32-bit PRF output; a raw-draw uniformity gate added ahead of the downstream law gate, with bars wide enough that
    an equal law cannot fail by chance.

### 12.2 The review's reads (POST-HOC, banked, CPU, reproducible)

`designs/research_state/measurements/value_disc_review_2026-10-11/`: `pairing_read.py` (VRF, derangement, separable
share, acting rule, split-half, gap SD), `decision_read.py` (the 3-option decision currency and power projections at
500 states), `focus_read.py` (all-legal-action improvement by the policy's gap quintile); each `*.out.json*` is its
output. Inputs: the refit's committed `rows/truth_held_S32.compact.jsonl.gz` and `rows/scores.npz`, the committed Lane
S bank, and the blob root read `blob.probs.npz` (ARCHIVE only: `~/gen3ai_archive/crn_label_refit_2026-10-06/rows/reads/`;
not committed). No playout was run. **Power, projected from these rows:** at 500 states the 3-option LEAF-K8 − CRITIC
improvement contrast has a CI of about ±0.010; the all-legal-action contrast on 468 turns had ±0.016, so 600 stratified
states over ≥ 300 battles should give about ±0.012, enough to detect a 2-point difference.

### 12.3 What remains disputed or unverified

- **How much of the leaf's edge is hidden information.** UNMEASURED; D2's unrevealed-slot strata are a proxy, not an
  answer. Only an honest (determinized) leaf against the oracle one answers it, and that is deferred.
- **Whether "the representation is the bound" holds for a trained TRUNK** (only frozen-trunk readouts were refit), and
  for the end-state critic at all.
- **Whether per-decision improvement becomes win rate.** One-step improvement in the greedy continuation's Q; the
  opponent does not search; compounding over ~26 decisions is not additive. Only a battery answers it (Q1).
- **Whether aligned dice share luck** (S1 / S2's premise): the move-move ρ of 0.23 says some; nothing says how much.
- **Whether the rb-blob reads transfer to CP**: off-distribution states may inflate the improvement and the "torn is
  the wrong selector" finding; the pilot re-measures both on CP's own states.
- **The draft's cost model for S1 / S2** (agent-days) is the author's estimate, unchanged and UNVERIFIED.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-11 | **This doc CREATED at the owner's request** ("write the design doc, then have an agent review it and improve on it") | A measurement-first plan: the RNG read from the code (one stream per battle, alignment lost at the first differing draw); two counterfactual-only dice modes (S1 per-turn reseed, S2 counter-keyed split draws) proposed, each CONDITIONAL on the pilot; the pilot DRAFTED (frozen checkpoint, 500 torn states, K = 3, 32 shared seeds + an independent control, greedy continuation, M1–M4, R-PAIR and R-GAP); CPU only | building split streams first (unmeasured gain, a parity-critical crossing); brute-force rollout volume for the critic (excluded by the CRN label refit); a training-side label factory now (deferred by the one-ply scope rule) | §2.2; §5 (code at `c1fae513`); owner 2026-10-11 |
| 2026-10-11 | **REVIEWED AND REVISED (independent review, owner request)** | The pilot's PRIMARY becomes DECISION IMPROVEMENT over the policy's own pick (R-LEAF), with R-TRUNC (truncated leaves), R-K (the "10×" question on the decision currency) and R-PAIR (now gating S1 only when no truncated leaf works); 600 states stratified by the policy's gap over ALL legal actions; the independent-seed arm dropped; the ORACLE-world asymmetry named as a fifth error source; truncated / mixed leaves, AIVAT-style control variates, a regret-aware acting rule and a sibling-level trunk signal added as levers; the RNG table corrected file by file (47 + 7 sites; Quick Claw at the end of the previous turn moves S1's hook; speed-tie and always-tied shuffles; any faint ends the turn's actions; the fork arm keeps its stream); training-side claims scoped to their provenance (v13-W, rb-blob readout refits; UNVERIFIED for a trunk and the end state); related work added | keeping pairwise accuracy and the separable share as the decision currency (equal accuracy hid unequal decisions on banked rows); a torn-only sample (the banked leaf value is larger on decided states); the independent-seed control (the derangement estimator reads 1.000 [0.996, 1.005]); building S1 on an S0 VRF the banked rows already put at 1.06–1.13 without first trying truncation | §2.3 and §12 (POST-HOC reads of banked rows, `research_state/measurements/value_disc_review_2026-10-11/`); §5 re-verified at `f56621db`; owner 2026-10-11 |
| 2026-10-11 | **PILOT REGISTERED (§7.10 Amendment 1), owner brief "make any decisions to gather data useful for design or our roadmap"** | Both closing-test seeds 2001 (end2 at `62085750`, prod at `95d014fa`), each at its own pin and on its own eval-trace states (600 each, gap quintiles); seeds 0–15 leaf / 16–39 truth / 40–47 critic; added scorers LEAF-K16, TRUNC at K up to 16, the AIVAT-style branch-turn control variate CV-K; acting rules defined before any outcome (ARGMAX, PRIOR-k, GUMBEL with `mctx` defaults, PUCT, sequential halving at matched budgets, the old 2·SE gate); R-PRIOR and R-SH added as secondary rules | waiting for the closing-test verdict to pick one arm (both are read instead, so the end-state critic's decision value and production's are both measured); the Lane S bank (off-distribution); the per-turn AIVAT form (needs extra dice samples per turn: deferred) | `research_state/measurements/value_disc_pilot_2026-10-11/README.md`; ledger 2026-10-11 REGISTRATION |
