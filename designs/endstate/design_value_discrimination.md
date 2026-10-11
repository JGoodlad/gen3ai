# BUYING VALUE DISCRIMINATION: a measurement-first pilot, and the label-factory direction that enables search

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A decision or build that differs from this
doc updates it in the same commit, and the Decision record at the end says what changed and why.

**Status (2026-10-11): DESIGN, written at the owner's request; an independent review follows.** Nothing is built and
nothing has been measured under this doc. §7 is the pilot's DRAFT pre-registration: it becomes a registration only
when its own commit lands BEFORE the first playout and a ledger entry records it. §5's RNG findings are read from the
code at `c1fae513` (no run). The pilot is an OFFLINE CPU measurement. Any IN-LOOP use of playouts to the end (a label
factory feeding training, a playout leaf in live play) stays DEFERRED under the owner's one-ply scope rule
(2026-09-30, §10) until the owner lifts it.

## 0. Summary

| | |
|---|---|
| question (owner, 2026-10-11) | "We need at least an order of magnitude more rollouts to enable discrimination for search, which we can't get in the current regime. **Do we really?**" |
| short answer | **For TRAINING the critic: more rollouts of the same kind are already measured not to help** (a critic refit on de-noised 8-rollout labels ranks siblings no better, §2.2). **For a search-time ROLLOUT estimate: yes, a raw rollout leaf needs many playouts per decision.** The cheap multiplier is VARIANCE REDUCTION, and the code shows that our common-random-numbers pairing **probably buys almost none today**, because the battle draws all its dice from ONE stream (§5). The pilot MEASURES both before anything is built |
| the thesis under test | (a) pairing with COMMON RANDOM NUMBERS cuts the rollouts a sibling DIFFERENCE needs by a factor VRF = 1/(1−ρ); (b) bootstrapped targets; (c) focus on torn states; (d) ensembles to pick where to spend; (e) the representation (end state); (f) search-as-teacher. Search consumes SIBLING RANKING, not global AUC |
| the RNG finding | ONE PRNG per battle (`BattleState.prng`, a bit-exact port of Showdown's `prng.ts`); every die of both sides comes from it in execution order (~49 call sites plus 6 speed-tie shuffles). A reseed at the branch point shares the STREAM across siblings, not the roll-to-event MAPPING: the first draw one sibling consumes and the other does not shifts every later die. **Predicted: VRF ≈ 1 beyond the branch turn** |
| the proposal | two COUNTERFACTUAL-ONLY dice modes, never reachable from training: **S1 per-turn reseed** (≈ 0.5–1 agent-day; realigns the dice at every turn boundary) and **S2 per-(turn, purpose, side) counter-keyed draws** (≈ 2–3 agent-days; realigns every draw). Both are equal IN LAW to normal play by construction (§5.4); S0 (today) stays byte-identical |
| the pilot (§7) | a frozen checkpoint; ~500 TORN decision points (policy top-2 gap at or below the 40th percentile); K = 3 options (top-1, top-2, a random legal alternative); M = 32 SHARED seeds per option, plus an INDEPENDENT-seed control; greedy continuation on both sides; metrics: the pairing variance-reduction factor by depth, playout separability (the ceiling), the critic's agreement with the playouts (the gap), the cost per state; registered decision rules |
| cost | ≈ 60,000 CPU playouts ≈ 4–6 CPU-hours, at `nice -n 19` beside the live run (rule 5b). **No GPU** |
| fit | after the closing-test verdict (its adopted arm is the frozen checkpoint), alongside the bf16 A/B (which is GPU; this is CPU) |

---

## 1. The problem, and the owner's question

A one-ply search picks, at a decision, the action whose successor looks best. That needs a scorer that **orders
sibling actions at the same state**. Today's win-prob critic orders siblings at **0.587–0.593** pairwise accuracy
(UNDERSTANDING §4.x; `crn_label_refit_2026-10-06/result.md` BASE 0.5932 [0.5578, 0.6274]), while a playout leaf
of 8 rollouts reads **0.670** on the same pairs. Every naive search built on the critic lost or tied
(`search_dividend_winprob_heads_2026-09-11`: 0.495–0.507 paired mirror win rate, none clear of 0.50).

The owner's worry: buying the playout-level ordering by brute force needs ~10× more rollouts than the current regime
can produce. The orchestrator's answer was that brute force buys discrimination only at √n, and that cheaper levers
exist. This doc turns that answer into claims that can fail, prices each lever, and registers a pilot that measures
the two unknowns every lever's price depends on:
1. **how much luck PAIRING actually removes** in our engine (the variance-reduction factor), and
2. **how much of the critic's sibling-ranking deficit is luck, and how much is the critic** (the ceiling and the gap).

---

## 2. Definitions, and what is already measured

### 2.1 Definitions

| term | meaning | the meter here |
|---|---|---|
| **calibration** | predicted win probabilities match observed frequencies. A monotone remap fixes it without changing any ORDER | ECE against a truth mean |
| **discrimination (resolution)** | the scorer ORDERS states or actions the way their true values are ordered. A rank property, invariant to any monotone remap | AUC; pairwise accuracy |
| **GLOBAL AUC** | wins vs losses separated ACROSS states: "is this position winning?" Dominated by between-state variation (material, matchup) | the critic's own win-prob vs the outcome: **0.766 / 0.763** (legacy / static seed means; `probe_battery_2026-10-09/critic_auc.json`) |
| **SIBLING RANKING** | at ONE state, which of two ACTIONS leads to the better outcome: `sign(V(s′_a) − V(s′_b))` vs the sign of the true gap. What a one-ply search consumes. A critic can have a high global AUC and a near-chance sibling ranking, because the between-state signal is large and the within-state signal is small | pairwise accuracy on non-tied sibling pairs under shared dice (UNDERSTANDING rule 25) |
| **true sibling gap** Δ_ab(s) | Q(s,a) − Q(s,b) under a FIXED continuation policy (policy evaluation): the expected outcome of "this action, then that policy on both sides" | the mean over M shared seeds of `o_a − o_b` |
| **TORN state** | a decision where the policy itself is undecided: its top-2 masked-logit gap is at or below the 40th percentile of the eligible pool (`fork_arm.contested_threshold`, `gen3_fork_select_v1`) | — |
| **separable pair** | a pair whose gap the playouts can resolve at budget M: `|Δ̂_M| > 1.96 · SE_M` | §7.6 M2 |
| **the irreducible-luck ceiling** | two different limits that are easy to confuse: (i) a SINGLE game outcome is a Bernoulli draw (dice + hidden information), so even a perfect value function cannot be scored above a ceiling when single outcomes are the yardstick (the single-rollout label mis-orders top1 vs top2 39.7 % of the time, ceiling ≈ 0.61; `leaf_ceiling_controls_2026-09-19/`); (ii) many sibling pairs are TRULY near-tied (|Δ| ≈ 0): no scorer can order them and search gains nothing there. (i) is a property of the yardstick and shrinks with M; (ii) is a property of the game and does not | (i): split-half agreement of the truth; (ii): the distribution of Δ̂ and the separable share |

**The pooled-correlation trap, named in advance.** The Python fork arm saw **79.7 %** of branch pairs TIED (same
outcome; `designs/training/forks.md` §2). That is NOT evidence that pairing correlates sibling outcomes. With no
within-state correlation at all, heterogeneity in p across states inflates the tie rate (p ∈ {0.1, 0.9} gives
P(tie) = 0.82). Correlation is measured only WITHIN a state, across seeds, after centring each action's mean
(memory note `feedback_pooled_correlation_simpson_trap`).

### 2.2 What the repo already knows (each claim with its tag)

| fact | tag | source |
|---|---|---|
| The critic ranks siblings at ~0.57–0.59 across **twelve heads** (two 10M, two 75M+, refits, a 6× wider head, a ranking term, a pre-pool tap). A head fitted from scratch in 1.1 s matches the 75M head | MEASURED | UNDERSTANDING §4.x; `offline_leaf_fit_2026-09-18/` |
| A **rollout leaf** beats it: 0.543 / 0.580 / 0.613 / 0.652 / 0.671 at K = 1/2/4/8/16 on the top1/top2 column vs the critic's 0.574. Knee at K = 4 | MEASURED, DETECTED at K ≥ 8 on the column | `rollout_leaf_kcurve_2026-09-19/` |
| **The acting rule throws the leaf away**: `|mean d| ≥ 2·SE` over R paired rollouts resolves 4.5 % of pairs at R = 4 and 12.5 % at R = 16; live, it changed 1 of 250 decisions. When it does resolve, it is right 83–92 % of the time | MEASURED | same |
| **A de-noised LABEL does not help the critic**: frozen-trunk refit on the mean of 8 shared-dice rollouts: REFIT-K8 − BASE **−0.014 [−0.032, +0.002]** (a gain is excluded); K8 vs K1 label no difference. The 8-rollout leaf beats the head by **+0.077 [+0.035, +0.115]**. The refit buys calibration (ECE 0.065 → 0.032), not resolution | MEASURED · NOT DETECTED, gain excluded · one checkpoint, greedy continuation | `crn_label_refit_2026-10-06/result.md` |
| On STARVED near-best actions vs DOMINATED ones the head is at 0.440 (below chance on points) while 8 rollouts are at 0.885 | MEASURED · descriptive | same, M3b |
| λ-return targets for the win-prob BCE (λ 0.9, 0.95): the 0.9 resolution gain was NOT CONFIRMED by its replicate and 0.95 failed its registered claim. `--win-prob-lambda` is deleted (L2) | MEASURED · NOT CONFIRMED | `designs/deleted_flags.md`; ledger 2026-09-11 (L17453, L17578) |
| Contested-state FORKS into the PPO buffer (top-2 + random, CRN dice and draws) did not move the head: Δ −0.0003 [−0.0148, +0.0136] | MEASURED · NOT DETECTED | `fork_arm_read_2026-09-16/`; `designs/training/forks.md` |
| The true top-1 vs top-2 gap is LARGE on average: E\|p_a − p_b\| ≈ 0.177 (sd 0.222; an 8-rollout estimate, biased UP by noise), while the policy's top-1 edge over its top-2 is +0.008, not clear of zero. Much to re-rank | MEASURED | `leaf_ceiling_controls_2026-09-19/` |
| Within a decision's one-step value-target variance, the DICE carry 13.9 % (uniform over legal) / 26.5 % (behaviour-weighted); the opponent's action the most | MEASURED · Node era, 140 decisions | memory `project_three_axis_value_variance` |
| 🚨 **"CRN in this sim shares the dice STREAM, not the roll-to-event MAPPING"**: a single-seed action sweep over-read the action effects by ~2× because a different action consumes the stream differently | MEASURED · Node era; the Rust port is bit-exact to the same `prng.ts` semantics (§5.1), so it carries over | same |
| Playout cost: **0.21–0.23 CPU-s per playout** (greedy both sides, CPU torch forward, ~30 sim turns, a loaded box) | MEASURED | `crn_label_refit_2026-10-06/README.md` Amendment 1 |

**The reading that frames this doc.** The critic is not label-limited, not loss-form-limited, not width-limited and
not (at the head level) moving-target-limited (§6). What separates siblings is in the PLAYOUTS. So "buying
discrimination" splits into two products with different economics:
- **A. a better VALUE FUNCTION** (training side): the bound is the representation / coverage. Rollout VOLUME does not
  buy it (measured). The levers are (e) the representation, coverage of the states that matter, and (f) eventually a
  search-improved policy to evaluate.
- **B. a better SEARCH-TIME ESTIMATE** (inference side): a rollout leaf already beats the critic, and its price is
  playouts per resolved pair. Here pairing, focus and the acting rule are the levers, and the owner's "order of
  magnitude" is the right worry — unless pairing divides it.

---

## 3. The four error sources, and which lever addresses each

| source | what it is | levers that address it | evidence that it binds TODAY for sibling ranking |
|---|---|---|---|
| **1. LABEL NOISE** | the training target is one game outcome: high variance | TD(λ) / bootstrapped targets; K-rollout labels; CRN-paired labels | **not binding** — K8 vs K1 refit no difference; λ arms not confirmed (§2.2) |
| **2. LUCK** (estimator variance) | a search-time or yardstick estimate of Δ from finite playouts carries dice and hidden-information variance | **pairing** (CRN, split streams); more rollouts (√n); focus the spend on states where it resolves; a better acting rule | **binding for the rollout leaf**: the acting rule resolves 4.5 % of pairs at R = 4 (§2.2). Part of it is irreducible (§2.1 (ii)) |
| **3. REPRESENTATION / COVERAGE** (bias) | what `value_pooled` encodes about a one-ply difference; which states it was trained on | the end-state facts and depth (closing test; `probe_battery_2026-10-09` hunt list); a per-action value read (`design_q_head.md`); training data on the states search queries | **the measured bound** for the critic: twelve heads on 0.567–0.578 whatever the label, loss, width or data |
| **4. MOVING TARGET** | the critic chases π_t; its target V^{π_t} changes as π learns | freeze for measurement (policy evaluation); staleness meters (§6) | **not binding at the head level** — frozen-policy fits on arm W's frozen self-play match the live head (`offline_leaf_fit_2026-09-18/`). UNVERIFIED for a full fresh value NETWORK |

---

## 4. The levers, with an expected cost per unit discrimination

**Assumptions** (stated, for back-of-envelope only):
- Outcomes on the win scale {0, 1}; per-action outcome variance p(1−p) ≈ 0.21 at p ≈ 0.7 (production-ish win rates
  on torn states; UNVERIFIED).
- So the UNPAIRED difference of two actions' single outcomes has variance ≈ 0.42 (SD ≈ 0.65).
- A pair with true gap Δ, estimated from K paired playouts with within-state correlation ρ, is ordered correctly with
  probability ≈ Φ(|Δ|·√K / (0.65·√(1−ρ))). The playouts needed for a fixed resolution scale as
  **K ∝ (1−ρ) / Δ²**. VRF = 1/(1−ρ) divides the bill.
- Worked numbers at a 2-SE acting rule: Δ = 0.18 needs K ≈ 52 unpaired; Δ = 0.05 needs K ≈ 680. At VRF 2 these halve,
  at VRF 5 they fall 5×.

| lever | attacks | price model | expected gain per unit cost | confidence |
|---|---|---|---|---|
| **(0) brute force**: more independent rollouts | luck | √n: each 4× of playouts halves the SE. The K-curve's measured increments: +0.037, +0.033, +0.039, +0.019 per doubling (K 1 → 16) | the baseline every lever is priced against | MEASURED (K-curve) |
| **(a) pairing / CRN** | luck in the DIFFERENCE | free in compute once the dice align; S1 ≈ 0.5–1 agent-day, S2 ≈ 2–3 agent-days of build (§5.3) | a VRF of v cuts every pair's bill v× — **the largest multiplier IF ρ is high**. Today's stream mode: predicted v ≈ 1 beyond the branch turn (§5.2) | UNVERIFIED → **the pilot's M1** |
| **(b) bootstrapped targets (TD(λ))** | label noise (training side) | cheap (a target change) | **low**: the label is not the bound (§2.2). A bootstrapped target can also import the critic's own sibling blindness | MEASURED (indirectly) |
| **(c) focus on torn / separable states** (active learning) | where the budget goes | a selector (exists: `fork_arm.contested_select`) | raises the useful fraction of the spend by 1/(share of states where an action choice matters and resolves). At the 40th-percentile torn cut, ~2.5× over uniform before separability. The separable share is unknown | UNVERIFIED → **M2** |
| **(d) ensembles for uncertainty** | WHERE to spend rollouts (V's own doubt as a gate) | the X26 ride-along V ensemble (`--ridealong-ensemble K`) exists, OFF | a GATE for (c), not discrimination by itself. Its value: rollouts only where members disagree. Whether disagreement predicts separability is untested | UNVERIFIED |
| **(e) representation** (the end state) | the critic's bias | already being paid (the closing test, ≈ 47 GPU-h) | the only lever with evidence against the measured bound. Its effect on SIBLING ranking has never been measured | UNVERIFIED → M3 on the adopted arm gives the first read |
| **(f) search-as-teacher** (the AlphaZero flywheel) | policy and critic together | needs a search that BEATS the policy first, so it is downstream of (a)–(e) | the long-run product; not priced here | — |

**The answer to "do we really need 10×?"** depends on two numbers nobody has measured on this engine: ρ (the pairing
correlation) and the separable share of torn pairs. If ρ ≈ 0 and most torn pairs have |Δ| < 0.05, yes: a rollout leaf
needs hundreds of playouts per torn decision, and that is out of reach in the loop. If S2 lifts ρ to 0.8, the same
resolution costs 5× fewer playouts. If most torn pairs are truly near-tied, the deficit is mostly irreducible luck and
search should be aimed at the separable minority. The pilot reads both.

---

## 5. The RNG-alignment investigation (findings from the code, `c1fae513`)

### 5.1 How the Rust engine draws randomness

| # | finding | where |
|---|---|---|
| R1 | **ONE PRNG per battle.** `BattleState.prng: Prng`, a bit-for-bit port of Showdown's `sim/prng.ts`, pinned against ~2,900 vectors from the real `prng.js`. Two backends chosen by the seed prefix: `sodium,<hex>` (ChaCha20, Showdown's default) and the gen-5 64-bit LCG (`gen5,<hex16>` or decimal `m,n,o,p`) | `src/rust_sim/src/prng/mod.rs`; `state.rs` (`pub prng: Prng`) |
| R2 | **Every die of BOTH sides comes from that one stream, in execution order**: accuracy (`speed.rs`, `random(100) < acc`, drawn for every numeric-accuracy move, 100 % included), crit, the damage roll (`random(16)`), secondary chances, multi-hit counts, durations (sleep, confusion, bind, lock moves), full paralysis / confusion / attract checks, Metronome-style picks, phazing replacement picks, item / ability procs, the per-turn Quick Claw roll (`driver.rs`, unconditional), and SPEED-TIE shuffles (`event::speed_sort` → `shuffle_range`, only for tied groups). ~49 direct call sites (excluding `prng/` and tests) plus 6 `speed_sort` sites | `src/rust_sim/src/turn/*.rs`, `event.rs` |
| R3 | **Several parity-only DISCARDED draws** consume the stream with no effect: the `selfDrops` `random(100)` (Overheat / Superpower; `residuals.rs`), sleep's `onStart` `random(2,6)` (`status_moves.rs`), `random(1)` for the one-foe `sample` (`event.rs`). Even a no-op mechanic shifts the stream | same |
| R4 | **Training battles use the LCG backend.** The env core stages `ep_seed` as four 16-bit words and joins them into the decimal form, so every training battle is a gen-5 LCG stream | `src/rust_env/src/core/pool.rs` |
| R5 | **Counterfactual tooling REPLACES the whole stream at the branch point.** `BridgeSession::reseed(seed)` sets `st.prng = Prng::new(seed)`; the playout tool calls it per branch before any feed (`Game::branch(Some(seed))`), as do `search_driver`'s `reroll` / `reroll_many` (each arm in a FRESH session rebuilt from turn 1, then reseeded) and the persistent search tree. `None` keeps the battle's own stream (the recorded continuation). The truth tool's seeds are `turn_seeds(id, S)` = `sodium,<32 hex>` (ChaCha20) | `src/rust_sim/src/bridge.rs`; `src/rust_env/src/search/{game,playout}.rs`; `src/rust_sim/src/bin/search_driver.rs`; `src/main/policy_spectrum/truth.py` |
| R6 | **Policy-SAMPLING draws are already counter-based**: the keyed draw is a pure function of (run seed, stream, env, episode, decision ordinal `n`) via a splitmix64 chain, inverse-CDF over the legal log-probs. Siblings that share the parent's keys consume the same uniform at the same `n` (the Rust fork arm's `dice_and_draws`). `n` itself slips when one branch has an extra decision (a forced switch after a faint) | `src/agents/training/keyed_draw.py`; `designs/training/forks.md` §14.3 |
| R7 | The search driver's follow-up `AuxRng` (FNV-1a of the seed string → mulberry32) is a SEPARATE stream from the battle's dice, used only by its `random` follow-up policy. It does not touch a policy playout | `src/rust_sim/src/search.rs` |

### 5.2 The consequence: sibling alignment decays at the first differing draw

Two siblings that share a seed start the branch turn with the same stream. They stay aligned only while they consume
the SAME draws in the SAME order. That breaks almost at once:
- **a switch consumes no dice; a move consumes 2–4** (accuracy, crit, damage roll, a secondary): move vs switch
  misaligns the opponent's dice in the branch turn itself;
- **two moves differ in draw count** when one has a secondary (Ice Beam's freeze roll vs Surf's none) or a multi-hit
  count, or never-miss accuracy;
- **ORDER changes**: a switch or a priority move goes first, so the opponent's draws come from different positions;
- **a KO cancels the victim's move** and its draws; a status cancels a turn (full paralysis), and so on;
- the R3 parity draws make even a no-op difference shift the stream.

After the shift, sibling A's draw i is sibling B's draw i + δ, and for ChaCha20 or the LCG those are effectively
independent. **So beyond the first differing draw the dice are effectively UNPAIRED**: ρ from shared luck ≈ 0 and
VRF ≈ 1, and the pairing the tooling advertises ("COMMON RANDOM NUMBERS") pairs the dice only up to that draw. The
Node-era probe saw this empirically (§2.2). **Prediction for the pilot: VRF ≈ 1.0–1.3 at the terminal; modestly above 1
right after the branch turn for move-move siblings with the same draw profile; ≈ 1 for move-switch.** UNVERIFIED on the
Rust engine until M1.

### 5.3 The proposal: two counterfactual-only dice modes

**S0, the status quo, is unchanged and byte-identical.** Two new modes are selected ONLY by a new seed PREFIX that the
training path cannot produce:

| mode | seed prefix | what it does | aligns | build |
|---|---|---|---|---|
| **S0** (today) | `sodium,` / `gen5,` / `m,n,o,p` | one stream per battle | up to the first differing draw | — |
| **S1 per-turn reseed** | `crnturn,<hex>` | at the START of every turn t the battle's stream is replaced by a fresh one seeded `H(base, t)`; within a turn it is one stream | every turn boundary: a misalignment in turn t cannot leak into turn t+1 | ≈ 0.5–1 agent-day: one hook at the turn top, the prefix, the gates below |
| **S2 split, counter-keyed** | `crnsplit,<hex>` | every draw is `u = PRF(base, turn, purpose, side, k)`, k counting draws of that (turn, purpose, side); `random(n)` maps u exactly as `prng.ts` maps `next()` | every draw of a given purpose and side, whatever the other draws did | ≈ 2–3 agent-days: a `Draw { purpose, side }` tag at every call site, the counter table in `BattleState`, the PRF, the gates |

Design rules for both:
- **The purposes** (S2) come from the call sites: ACCURACY, CRIT, DAMAGE_ROLL, SECONDARY, MULTIHIT, DURATION,
  ACTION_CHECK (full paralysis, confusion self-hit, attract), SPEED_TIE (side = field), QUICK_CLAW (side = field),
  MOVE_PICK, PHAZE_PICK, ITEM_PROC, ABILITY_PROC, PARITY (the discarded draws, each with its own tag so they shift
  nothing). The SIDE is the side whose luck it is: the attacker for accuracy / crit / roll / secondary, the holder for a
  duration or an action check. A MIS-tag weakens coupling but cannot change the law (§5.4).
- **Comonotone coupling.** With a shared u and the same inverse-CDF map, "both branches got a high roll" is the
  coupling that maximises the positive correlation of luck (the textbook CRN with inverse transforms).
- **Stream backends IGNORE the tag**: the default path calls `next()` in the same order as today, so S0 is
  byte-identical BY CONSTRUCTION, not by test alone.
- **The policy-sampling draws in a sampled continuation** move to the same keying: (turn, side, k) instead of the
  decision ordinal `n` (R6), so a forced switch in one branch does not shift the other's draws.
- **Never the training path.** `Prng::try_new` (the bridge's START and the env core's seed path) REFUSES the `crn*`
  prefixes; only a new `Prng::counterfactual(seed)` constructor, reached from `reseed` in the playout / search APIs,
  accepts them. The env core's `ep_seed` is four integer words, so it cannot spell one anyway. A `crn*` battle is not a
  Showdown battle for any seed (Showdown's `setSeed` throws on the prefix), so no log made under it can be mistaken for
  one.

### 5.4 Statistical equivalence, and the gates

**The argument.** In S0, a battle's dice are a sequence of 32-bit outputs treated as i.i.d. uniform. In S1 and S2,
every draw is a PRF output on a key no other draw of that battle uses (distinct (t) per stream in S1; distinct
(turn, purpose, side, k) in S2), so under the PRF assumption the draws a single battle consumes are also i.i.d.
uniform, and the mechanics map them identically. **The law of one battle is therefore the same in all three modes.**
What changes is only the JOINT law of SIBLINGS, which is the point. Two caveats: a key collision inside one battle
would correlate two of its draws (prevented by construction: k is a per-key counter, and a gate asserts no repeat); and
a batch of siblings is a correlated sample, so sibling playouts are never used as if independent.

**The gates** (each must FAIL on revert):
1. **S0 byte-identity**: the existing differential ladder, the obs golden (`golden_obs_core --check`), the K9 golden
   and the A/B fuzzers, unchanged, all on S0 seeds.
2. **Law equivalence**: N greedy self-play battles per mode from the same starting set, with registered bars on the
   win rate and on per-purpose event rates (crit, miss, damage-roll histogram, sleep and confusion durations, speed-tie
   splits). Deterministic: fixed seeds, fixed N, the bars set before the first run; a statistic within a rounding error
   of its bar counts as a FAIL (rule 8).
3. **Tag coverage** (S2): the untagged draw API is REMOVED, so an untagged site does not compile.
4. **Alignment**: two branches with different root actions; for every (turn ≥ t+1, purpose, side, k) both consume,
   the uniforms are equal (S2), and every draw of turn ≥ t+1 is equal when both reach it (S1).
5. **Determinism**: a `crn*` seed reproduces its battle byte for byte.
6. **Refusal**: the bridge START and the env core refuse a `crn*` seed with a typed error.

**Risk.** S2 touches ~55 sites in the parity-critical core (an `opus-high` / `opus-xhigh` unit per the agent rules).
The risk to production is contained by gate 1 and the tag-ignoring backends. The risk to the measurement is a law
break in the new mode, which gate 2 catches only at its N; that is the declared limit.

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
| **MT2 critic STALENESS** | the critic of checkpoint t scored on games played by π_{t+k} (k = 1, 2, 5, 10M steps): global AUC, Brier and sibling accuracy, against the critic of t+k on the same games. The staleness curve and its half-life k½ (the k at which V_t loses half of V_{t+k}'s advantage). It sets how fresh a label must be to be worth anything in the loop | to build (≈ 0.5 agent-day, B6) |
| **MT3 frozen-policy head vs the live critic** | a head refit (trunk frozen) on π_t's own games vs π_t's live head | **already measured at the head level: no gain** (`offline_leaf_fit_2026-09-18/`: seven fits on 17,904 branched positions of arm W's frozen self-play, 0.567–0.576, matching the live head) |

**Reading.** At the head level the moving target is not what binds sibling ranking. The untested variant is a full
fresh VALUE NETWORK (its own trunk) trained to convergence on frozen-policy data (stationary, abundant): a GPU job, not
in this pilot (open question Q6).

---

## 7. THE PILOT (DRAFT pre-registration, house style)

Registered when this section's commit lands before the first playout, with a ledger REGISTRATION entry. Amendments go
in §7.10, dated, each saying whether any outcome had been seen.

### 7.1 The question

On a frozen checkpoint's torn decisions: (1) how much variance does pairing remove from a sibling DIFFERENCE, as a
function of how far past the branch it is read, in today's dice mode? (2) What share of torn sibling pairs can
playouts separate at all (the ceiling)? (3) How far is the critic from the playouts on those pairs (the gap)? (4) What
does a state cost?

### 7.2 What is fixed

| item | value |
|---|---|
| checkpoint **CP** | the closing test's ADOPTED arm, seed 2001's final (`design_endstate_closing_test.md` §3.3's verdict, §6): on PASS `rb_es_end2_s2001`, on FAIL `rb_es_prod_s2001`. READ-ONLY, loaded on CPU through the strict loader (`main.policy_spectrum.reader.load_checkpoint`; `qhat.check_winprob` passes), with `CUDA_VISIBLE_DEVICES=` empty (the CUDA-context lesson of `crn_label_refit` Amendment 2) |
| states | a bank built from CP's own Rust eval traces (each carries `<prefix>_reconstruction.json`, the input log) through `main.policy_spectrum.bank`, re-encode gate ① checked. **Fallback**, chosen and recorded BEFORE any outcome: the committed Lane S bank `m5_laneS/bank_v1/`, stamped OFF-DISTRIBUTION for CP. The eval quota is loss-enriched; the per-outcome stamp is carried and reported, never re-weighted silently |
| eligible | a free move round (not a forced switch), turn ∈ [2, 40], ≥ 3 legal actions, gate ① passed |
| **TORN** | CP's top-2 masked-logit gap at or below the 40th percentile of the eligible pool (`fork_arm.contested_threshold`) |
| sample | **500 torn states**, at most 2 per battle, in `sha256("value_disc_pilot_2026:" + decision_id)` order |
| **K = 3 options** | `top1`, `top2` by CP's masked logits at the root; `rand` uniform over legal \ {top1, top2} by the fork arm's decision-keyed sha256 draw (`fork_arm.py`). Pairs: (top1, top2) = the leaf column, (top1, rand), (top2, rand) |
| continuation | CP GREEDY on both sides to the end (`utils.rust_env.successors.play_out`, `stall="production"`, `max_turns` 999), rooted as `truth.branch_turn` roots it (the decision's command index; the other side's root decision, where still open, answered by CP's greedy policy, identically across siblings). Value: +1 win / −1 loss (a stall forfeit included) / 0 tie or truncated, from the searched side (the truth tool's convention). Greedy removes policy-sampling noise, so the DICE are the only luck: the cleanest read of R1–R5 |
| **dice, SHARED** | seeds j = 0..31, `truth.turn_seeds(id, 32)` (nested), every option on every seed: 96 playouts per state |
| **dice, INDEPENDENT control** | on the hash-chosen 25 % of states (`sha256("value_disc_pilot_2026:ind:" + id)`, first hex digit 0–3; ≈ 125 states), every option ALSO on its own seeds `turn_seeds(id + "|ind|" + option, 32)`: 96 more playouts per control state |
| **depth trace** | for every branch, CP's win-prob V at each of the SEARCHED side's decisions after the branch (depth d = 1, 2, …; a forced switch counts), with the turn number, and the terminal; after the end, X_d is the terminal value (absorbing). One forward gives logits and V (`qhat.logits_and_values`) |
| critic successors | `qhat.one_ply` on 8 SEPARATE seeds (indices 32–39 of `turn_seeds(id, 40)`): Q̂(a) = the mean over successors of 2·σ(head) − 1, a branch that ends scoring its terminal |
| truth for M3 | the mean of `o_a − o_b` over seeds 8–31 (24 seeds); the leaf scorers use seeds 0..K−1 (K = 1, 4, 8) and never see 8–31 |
| CIs | battle-clustered bootstrap, 2,000 draws, `numpy.random.default_rng(20261011)`, percentile 95 %; every contrast PAIRED on the same draws |

### 7.3 Why three options, and why these

top1 vs top2 is the column a re-ranking search consumes, and the critic's worst (0.545 on the 2026-09-19 forks).
The random alternative is where the new information is: in 4.5 % of forks it beat both policy candidates
(`forks.md` §2). Three options give three pairs per state at 1.5× the playouts of two.

### 7.4 The metrics

| meter | definition |
|---|---|
| **M1, PAIRING (the headline)** | for each pair (a, b) of a state and depth d: s²_pair(d) = the sample variance over the shared seeds j of X_d(a,j) − X_d(b,j); s²_unp(d) = s²_a(d) + s²_b(d) (the variance of a difference of INDEPENDENT draws, estimated from the same playouts). **VRF(d) = Σ s²_unp(d) / Σ s²_pair(d)** over pairs (a ratio of sums, so a zero-variance pair cannot blow it up); ρ(d) = Σ cov / Σ √(s²_a s²_b). Reported at d = 1, 2, 3, 5, 8 and the terminal, by pair class: move-move SAME draw profile, move-move different, move-switch, switch-switch. **Validity control (deterministic):** on the control states, the INDEPENDENT seeds' VRF(terminal) 95 % CI must contain 1.0 and their per-option means must agree with the shared seeds' (a paired difference CI containing 0); otherwise the read is VOID. A seed-DERANGEMENT estimate (pair a's seed j with b's seed j+1 mod 32) is the free cross-check on all 500 |
| **M1b, the ACTING RULE** | the share of pairs resolved by `|mean d| ≥ 2·SE` (the playoff rule) at R = 4, 8, 16, 32, with SE from the PAIRED differences vs from the unpaired variance: the operational form of VRF |
| **M2, SEPARABILITY (the ceiling)** | the share S of torn pairs with `|Δ̂_32| > 1.96 · SE_32` (paired); the distribution of |Δ̂_32|; SPLIT-HALF agreement (seeds 0–15 vs 16–31: the sign agreement on pairs non-tied in both halves) |
| **M3, the GAP** | pairwise accuracy against the 24-seed truth on non-tied pairs (Δ is a multiple of 1/24; an exact zero is a TIE and is EXCLUDED, rule 8): CRITIC (Q̂), LEAF-K1 / K4 / K8. **G = LEAF-K8 − CRITIC** (paired). Also: value-weighted accuracy (pairs weighted by |Δ|) and the top-1 REGRET among the three options, E[Q(a*) − Q(argmax scorer)], with the winner's-curse caveat stamped |
| **M4, COST** | CPU-s per playout (the job's cgroup `cpu.stat`), playouts per state, and the derived CPU-s per RESOLVED pair at the measured VRF |

### 7.5 Decision rules (registered)

A bound within ±0.0005 of its boundary is a BOUNDARY result and counts as not clear (rule 8).

**R-PAIR, on VRF(terminal) over all pairs, in S0:**

| outcome | condition | implies |
|---|---|---|
| **PAIRING STRONG** | lower bound ≥ 2.0 | stream CRN already halves the bill; no RNG build; the cost model uses the measured VRF |
| **PAIRING WEAK, ALIGNMENT-LOSS SIGNATURE** | upper bound < 1.5 AND VRF(1)'s lower bound > VRF(terminal)'s upper bound | the luck IS shared while the dice align and is lost after: **build S1**, re-run the shared-seed arm on the SAME states and seeds (Phase 2). Build S2 only if S1's VRF(terminal) upper bound is still < 1.5 |
| **PAIRING WEAK, NO SIGNATURE** | upper bound < 1.5 and no detected decay | even aligned dice do not correlate sibling outcomes (the same luck hits different states differently): **do not build split streams**; pairing is not a lever on this engine |
| **MODERATE** | otherwise | report; S1 is cheap and is built only if the owner or the orchestrator wants the sharper read |

**R-GAP, on G and the separable share S:**

| outcome | condition | implies |
|---|---|---|
| **CRITIC-BOUND** | G's lower bound > +0.02 AND S ≥ 0.25 | the value side is the constraint and the playouts have the information. Since the label is excluded (§2.2), "better targets" means DIFFERENT targets on the states search queries (per-action, sibling-paired, from playouts) and the representation, not more labels of today's kind. Route to the representation read on the adopted arm and to a registered label-factory experiment |
| **LUCK-BOUND** | S < 0.10 | most torn pairs are truly near-tied: no scorer orders them and search gains little there. Focus search on the separable minority: build a DECISIVENESS screen (ensemble disagreement, or a 4-playout screen) and price it |
| **MIXED** | otherwise | report both numbers; the next step is the owner's call with this doc's cost table |

### 7.6 Prediction (written before measurement)

- VRF(terminal) in S0: **1.0–1.3** (§5.2). VRF(1): 1.2–2.5 on move-move same-profile pairs, ≈ 1 on move-switch.
- G = LEAF-K8 − CRITIC: **+0.05 to +0.09**, detected (the 2026-09-19 and 2026-10-06 reads, on a new checkpoint).
- S: **0.15–0.30** (the playoff rule resolved 12.5 % at R = 16 under unaligned dice; 32 seeds resolve more).
- Cost: 0.25–0.40 CPU-s per playout (a larger network than the X5 blob arm's).

### 7.7 Compute budget, and where it runs

- Playouts: 500 × 96 (shared) + ≈ 125 × 96 (independent) ≈ **60,000**, plus ≈ 12,000 one-ply successors (cheap).
  At 0.25–0.40 CPU-s ⇒ **≈ 4–7 CPU-hours**. Phase 2 (S1) adds ≈ 48,000 playouts.
- **CPU only. No GPU, no lease.** CP's forward runs on CPU (the playout loop batches every pending decision of a root
  into one forward).
- Rule 5b: `nice -n 19`, under `scripts/ops/mem_cap.sh`, ≤ 2 heavy jobs at once, detached (`setsid nohup`),
  durable rows (one fsync'd JSONL row per state), resumable by state id, kill-and-resume tested once on a pilot
  subset (the `crn_label_refit` pattern). The live run's trainer and Rust core keep priority.
- "Small GPU slices" belong to the LABEL FACTORY at scale (a T2-batched forward makes a playout's sim + encode the
  cost; an UNVERIFIED estimate is ~10 ms of CPU per playout vs ~250 ms with the CPU forward). Not needed to answer
  §7.1.

### 7.8 Size rule

A throughput pilot (≤ 20 states, throughput only, no outcome read) measures CPU-s per playout. If the full job exceeds
**10 CPU-hours**, states are dropped in reverse hash order until it fits, the count recorded as an amendment before any
outcome is read.

### 7.9 Files (to be created under `designs/research_state/measurements/value_disc_pilot_<date>/`)

`README.md` (this pre-registration, copied, plus amendments), `make_states.py`, `run_playouts.py` (the depth-trace
driver), `score.py` (M1–M4, the bootstrap, the verdicts), `run_all.sh`, `rows/`, `result.md`, `result.json`.

### 7.10 Amendments

None.

---

## 8. Build list

| # | item | exists? | cost |
|---|---|---|---|
| — | `play_out` (CRN branches × seeds, an `actions` subset, a host-driven policy) | ✅ `src/utils/rust_env/successors.py` | — |
| — | `turn_seeds`, `branch_turn`, the truth convention | ✅ `src/main/policy_spectrum/truth.py` | — |
| — | `one_ply`, `logits_and_values`, `check_winprob` | ✅ `src/main/policy_spectrum/qhat.py` | — |
| — | the turn bank from eval traces (re-encode gate ①) | ✅ `src/main/policy_spectrum/bank.py` | — |
| — | the torn selector and the random-branch draw | ✅ `src/agents/training/fork_arm.py` | — |
| — | the pair / bootstrap / tie-exclusion scoring pattern | ✅ `crn_label_refit_2026-10-06/score.py` | — |
| — | the playoff (acting) rule | ✅ `rollout_leaf_kcurve_2026-09-19/playoff_rule_offline.py` | — |
| — | `main.policy_drift` (MT1) | ✅ | — |
| B1 | the pilot DRIVER: a policy wrapper that records V per (branch, side, decision, turn) from the `who` array, the shared + independent seed arms | ❌ | ≈ 0.5 agent-day |
| B2 | a bank SOURCE for CP's Rust eval traces (verify `bank.py` reads a core trace's reconstruction record) | ❌ (verify) | ≈ 0.25 |
| B3 | `score.py`: VRF(d), the derangement cross-check, separability, the gap, regret | ❌ | ≈ 0.5 |
| B4 | **S1** per-turn reseed (`crnturn,`) + gates 1, 2, 4–6 | ❌, CONDITIONAL on R-PAIR | ≈ 0.5–1 |
| B5 | **S2** counter-keyed split draws (`crnsplit,`) + gates 1–6 + a per-battle purpose-tagged draw log | ❌, CONDITIONAL on S1 | ≈ 2–3 |
| B6 | MT2 critic staleness meter | ❌ | ≈ 0.5 |

Pilot (B1–B3): ≈ 1.25 agent-days of build plus ≈ 1 wall-day of CPU at low priority.

---

## 9. Roadmap fit

- **After the closing-test verdict** (`design_endstate_closing_test.md`): CP is the adopted arm's seed 2001 final, so
  M3 is also the first SIBLING-ranking read of the end state's representation (lever (e)).
- **Alongside the bf16 A/B**: that is GPU work; the pilot is CPU at `nice -n 19`. They do not compete for the lease.
- **The plateau-first funnel** (memory `project_plateau_first_funnel`): the pilot is a measurement, not a screen or a
  run; it blocks nothing and informs the search half of the north stars (memory `project_north_stars_2026-09-27`).
- **Order after the pilot**: R-PAIR decides B4 / B5; R-GAP decides whether the next registered unit is a label-factory
  experiment (CRITIC-BOUND), a decisiveness screen (LUCK-BOUND), or the owner's call (MIXED). Any IN-LOOP use waits for
  the owner (§10).

---

## 10. Open questions (for the reviewer and the owner)

1. **The one-ply scope rule.** The owner deferred everything that needs playouts to the end (2026-09-30). This pilot is
   offline; its products (a playout leaf in play, playout labels in training) are not. Does the owner lift the deferral
   on a CRITIC-BOUND or PAIRING-STRONG read, or does the pilot only inform a later decision?
2. **Greedy vs sampled continuation.** Greedy makes the dice the only luck (the cleanest test of the RNG), and it is
   what a greedy search leaf evaluates. A sampled (T = 1) continuation is the training ecology and adds policy-draw luck
   that R6's keyed draw already pairs by decision ordinal. Is a sampled arm needed in the pilot, or only after S1 / S2?
3. **Is VRF the right currency?** A rank rule cares about P(sign correct), which depends on Δ too. The value-weighted
   accuracy and regret in M3 are the decision-relevant reads; should the R-PAIR rule be on the acting rule's resolved
   share (M1b) instead of VRF?
4. **The derangement estimator vs the independent control.** The unpaired variance is estimable from the shared-seed
   data alone (s²_a + s²_b). The independent arm costs 20 % of the job and serves as a validity control. Is it worth it,
   or should the 20 % buy more states?
5. **S2's keying by SIDE vs by MON.** Side-keying shares "our first attack's crit roll" across branches even when the
   attacker differs (stay vs switch). Mon-keying shares a mon's luck across branches. Which correlates sibling outcomes
   more is an empirical question; is it worth a factorial arm once S2 exists?
6. **A full fresh value NETWORK on frozen-policy data** (MT3's untested variant). If the critic's bound is the
   representation trained under a moving target, a stationary, abundant target might teach a different trunk. GPU cost
   and design not priced here.
7. **The ceiling's estimator.** The split-half sign agreement compresses toward 0.5 for small gaps; a deconvolved
   "perfect-ranker accuracy against the 24-seed truth" needs a model of Δ's distribution. Which estimator should be
   registered?
8. **The torn definition.** The 40th-percentile LOGIT-gap quantile is inherited from the fork arm. A probability-ratio
   threshold (π(top2)/π(top1) ≥ c) is closer to "the policy is torn". Is the quantile right for a frozen checkpoint, where
   the scale cannot drift?
9. **State source.** CP's own eval traces are loss-enriched and opponent-mixed (bots + pool). Should the states come
   from CP's greedy SELF-PLAY instead (the distribution a search inside self-play would see)?

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-11 | **This doc CREATED at the owner's request** ("write the design doc, then have an agent review it and improve on it") | A measurement-first plan: the RNG read from the code (one stream per battle, alignment lost at the first differing draw); two counterfactual-only dice modes (S1 per-turn reseed, S2 counter-keyed split draws) proposed, each CONDITIONAL on the pilot; the pilot DRAFTED (frozen checkpoint, 500 torn states, K = 3, 32 shared seeds + an independent control, greedy continuation, M1–M4, R-PAIR and R-GAP); CPU only | building split streams first (unmeasured gain, a parity-critical crossing); brute-force rollout volume for the critic (excluded by the CRN label refit); a training-side label factory now (deferred by the one-ply scope rule) | §2.2; §5 (code at `c1fae513`); owner 2026-10-11 |
