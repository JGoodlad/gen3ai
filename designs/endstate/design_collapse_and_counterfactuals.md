# What kind of COLLAPSE does our policy show, and how do we get COUNTERFACTUALS for it?

**Status: DESIGN + PILOT 2026-10-17 (CPU only; nothing built into training, nothing launched).** ALWAYS-CURRENT (owner,
2026-09-27): a later decision or measurement that differs updates this doc in the same commit, and the Decision record
(§7) says what changed and why.

**The owner's question (2026-10-17):** *"understand what type of collapse are we seeing? For example, equally good
plays. We need some way to give counterfactuals."* Earlier (2026-10-06): *"only concerned about action starvation where
near equally good moves are weighted near the bottom."*

**What this doc gives:**
- §1: a TAXONOMY of collapse, each type with an operational MEASUREMENT on checkpoints, the instrument that already
  exists for it, and what we know;
- §2: the COUNTERFACTUAL options (per-action values for the actions the policy does not take), what each costs and how
  far it can be trusted;
- §3: the PILOT on X26 (`rb_x26_s2601`) checkpoints: starvation with one-ply-branch rollout values, its trend, entropy
  and strategic cycling from existing reads, and the honest-vs-oracle detectability of starvation;
- §4: the RECOMMENDED measurement battery on X26's trajectory;
- §5: which LEVERS address each collapse type, tied to the plateau-breaker phase (X35) and the search programme;
- §6: FINDINGS; §7: the Decision record.

Words used throughout:
- **near-best**: an action whose paired value gap to the best action is ≤ ε = 0.1 on the ±1 outcome scale
  (≈ 5 pp of win probability);
- **dominated**: separably worse than the best by more than ε (gap − 1.96·SE > ε);
- **decisive turn**: a turn with at least one dominated action (on the other turns every action is about as good, so
  concentrating mass costs nothing);
- **starved**: a near-best action with < 1 % of the policy's probability (the Lane S definition,
  `main.policy_spectrum.truth`);
- **true world / oracle**: rollouts played in the real battle, with every hidden fact of the opponent's team as it
  was; **honest world**: rollouts in worlds sampled from what our side has seen (`main.value_search.worlds`, honest by
  default), i.e. what a player could compute at decision time;
- **IMP**: decision improvement, the win probability gained per decision by playing a scorer's pick instead of the
  policy's own top choice, scored on held-out seeds (`search_ledger.md` §6).

## 0. One screen

**What kind of collapse does X26 show? Not entropy collapse, and not GROWING starvation. A steady misranking whose
value lives in the hidden information, plus a slowing, mildly cyclic climb.** PROVISIONAL: the bullets after the first
read 10M–160M, before the plateau.
- **On the plateau (180M–380M, read 2026-10-20) X26 is a TREADMILL, not a collapse** (DESCRIPTIVE, one seed,
  [`../research_state/measurements/x26_collapse_battery_2026-10-20/`](../research_state/measurements/x26_collapse_battery_2026-10-20/README.md)):
  - steady ≈ 30 % greedy churn per 20M, with a stale self-play pool and a halved learning rate;
  - entropy −0.04 to −0.07 nats from 160M, then flat;
  - starvation up +0.03 to +0.06 on both referees while dominated mass and regret stop falling (B2 awaits B4's
    refreshed truth);
  - no coverage, rank, dormant-unit or opponent-model collapse.
  - the two plainest changes, pool staleness and the learning rate in force, are now battery items B11 and B12 (§4).
    Read with X34's records, the toy shows the same halving and staleness. There the halving came 70M BEFORE the plateau,
    so the rate covaries with the plateau without coinciding with it. The intervention is X35's batch-2 candidate LR-F,
    toy first (`../research_state/measurements/x35_breakers_stage_2026-10-17/STAGED_PLAN.md` §4.1).
- **(b) Entropy: none.** X26's entropy on its own states is flat at 1.05–1.16 nats from 10M to 160M (it broadened
  after 30M, +0.062 [+0.037, +0.087]). `ent_coef` 0.05 holds it (§3.2).
- **(a) Starvation is present and FLAT, not rising.**
  - Size: about 3 in 10 decisive turns put < 1 % on a near-best move (0.25–0.32 on X26's own truth, 32 seeds, X26@150M
    continuation; 0.31–0.38 on the pre-era truth).
  - Trend: 10M → 160M −0.011 [−0.090, +0.061] on its own truth and +0.013 to +0.023 (not detected, 3/3) on the
    pre-era one. Meanwhile dominated mass and greedy regret FALL (detected 3/3 pre-era; regret −0.046 [−0.091, −0.001]
    own truth). That is healthy sharpening, unlike the ai_v14 lineage (starvation +0.14 to +0.16). X26@160M starves
    0.13–0.14 less than K3 did.
- **The owner's "equally good plays" case (TIE-COLLAPSE) is the SMALLER part:** 0.05–0.09 of decisive turns. About 3/4
  of the starvation is MISRANK: the policy's top choice is worse than a move it puts under 1 % on. Greedy play loses
  nothing at a tie-collapse turn; its cost is exploitability, which a best responder reads, not this instrument.
- **The misranked moves are worth a lot, but only to the ORACLE:** +0.35 to +0.39 on the ±1 scale (≈ +18 pp win
  probability), held out, per starved decisive turn, in the TRUE world. An HONEST-world one-ply read cannot find them.
  Its flags are 1.35× chance against the oracle's 2.6×, and playing the move it flags LOSES −0.068 against the policy's
  top choice (§3.5). The value of the starved moves is hidden information.
- **(c) Strategic:** a slow transitive climb (Tier 1 CLIMBING at every check to 160M, but the GSPRT needs 5–9× more
  pairs than at 60–120M) with a small, real cycle: Hodge excess width 32 ELO, p = 0.005, 3.1 % of the flow (§3.4).
- **Counterfactuals (§2):**
  - Ground truth (one-ply rollouts to the end) costs ≈ 43 CPU-s per state on CPU; it is BUILT, and it is the right
    REFEREE.
  - Not one deployable counterfactual knows which starved move is good TODAY: not the honest-world rollouts, not V, and
    probably not the ride-along A head.
  - Completed-Q / Q-head targets are therefore only as good as the search programme's belief quality, and they are
    deferred by the owner's §5.0 anyway.
- **Recommendation (§4, §5):**
  - Run the battery every 20M on X26 and on every X35 breaker arm. Starvation is a GUARD there: a breaker that sharpens
    must not starve.
  - The levers by type:
    - (c): PFSP / league.
    - (a1): a trailing magnet where an exploiter shows the tie-breaks are exploitable.
    - (a2): honest search quality first, then counterfactual targets.
  - Do not spend a GPU arm on anti-starvation for X26 now.

## 1. A taxonomy of collapse

"Collapse" names several different failures that look alike on one scalar (the policy's entropy). They have different
causes, different instruments and different levers, so they are separated here. Each row: what it is, the operational
measurement on a checkpoint, the instrument that exists today, and what we know.

### (a) Action STARVATION of near-equal-value moves (the owner's concern)

**What.** The policy puts ~0 probability on an action whose counterfactual value is within ε of the best. It is a
statement about WHERE the mass goes, not how much entropy there is: a policy can be diffuse overall (mass on bad moves)
and still starve good ones. Two sub-types matter, and the pilot (§3) separates them:
- **(a1) TIE-COLLAPSE.** The policy's top choice is itself near-best and the other near-best moves are starved. Greedy
  play loses nothing at that turn against a fixed opponent. What is lost is (i) the alternatives' learning signal (PPO
  only learns about actions it samples; an action at π = 1e-3 is sampled once per thousand visits), and (ii) in an
  imperfect-information, simultaneous-move game, UNPREDICTABILITY: where the equilibrium mixes between equally good
  moves (a guess between a switch and an attack), a deterministic tie-break is exploitable by a best responder. The
  greedy-continuation truth below cannot see (ii); the best-response gap meter can.
- **(a2) MISRANK starvation.** A near-best move sits under 1 % while the policy's top choice is NOT near-best (worse by
  more than ε on the means). This loses value at greedy play, and it is the harmful case.

**Measurement.** Per checkpoint, on a fixed bank of turns with per-action ground-truth values (every legal action ×
S shared dice seeds played to the end, CRN): the starved share of decisive turns; its (a1) / (a2) split; the HELD-OUT
gain v(best starved) − v(argmax), with the starved set chosen on half the seeds and scored on the other half (no winner's
curse); the mass on dominated actions (healthy sharpening lowers it); regret. **Instrument:** `main.policy_spectrum`
(bank, reader, `truth`) + this pilot's `starvation_readout.py` (§3). **What we know:** the ai_v14 lineage's sharpening
was mostly starvation (decisive turns starved 0.33 → 0.50 while dominated mass fell only 0.018; UNDERSTANDING §4.4,
ledger 2026-09-29 `a41dcfca`); the pilot measures X26 (§3).

### (b) Entropy / mode collapse

**What.** The policy's total entropy falls toward a deterministic policy everywhere (the Mnih et al. 2016 A3C motivation
for the entropy bonus: premature convergence to a deterministic sub-optimal policy, **UNVERIFIED:** cited from memory).
In LLM RL the same failure is called entropy collapse; one analysis attributes the entropy change per update to the
covariance between an action's log-probability and its advantage, so confident actions with positive advantage drain
entropy fastest (Cui et al. 2025, **UNVERIFIED:** from memory), and "clip-higher" (a larger upper PPO clip, DAPO, Yu et
al. 2025, **UNVERIFIED:** from memory) is one published counter.

**Measurement.** Mean entropy, rank-1 mass and effective choices (exp H) on a FIXED bank, per stratum (forced switch vs
free, phase, legal count); the training-time `train/entropy_loss` is a second, on-distribution read (it moves with the
state distribution, so it is not a paired measure). **Instrument:** `main.policy_spectrum read` (forward passes, CPU,
~30 s per checkpoint on a 7k-turn bank). **What we know:** `ent_coef` 0.05 restored the v8 entropy regime (end-of-run
H ≈ 1.09, ledger 2026-09-12) and X26 runs it; on its own bank X26's entropy is FLAT from 10M to 160M (§3.2).

### (c) STRATEGIC collapse / cycling

**What.** The population (the policy and its self-play pool) converges onto a narrow set of strategies, or chases a
cycle (A beats B beats C beats A) without transitive progress. The canonical treatment is AlphaStar's league (main
agents, main exploiters, league exploiters, prioritised fictitious self-play, PFSP; Vinyals et al. 2019,
**UNVERIFIED:** from memory), with the game-theoretic framing of Balduzzi et al. 2019 (effective diversity, PSRO_rN)
and Czarnecki et al. 2020 ("spinning tops": real games are transitive at the extremes and cyclic in the middle)
(**UNVERIFIED:** both from memory; our note `designs/learning/population_game_theory.md`).

**Measurement.** (1) The Hodge decomposition of the dense snapshot ladder: transitive spine vs cyclic WIDTH against its
own noise floor (`python -m main.elo <run> --out <dir>`; `agents.training.hodge`); (2) the win rate of the newest node
against frozen sentinels 30–90M old (`eval_results.jsonl`, greedy, symmetric teams); (3) the plateau meter's Tier 1
(`python -m main.plateau status <run>`: the newest node vs the one 50M back, GSPRT); (4) exploitability by a fresh best
responder (`main.best_response_gap`). **What we know:** §3.4.

### (d) State-COVERAGE collapse

**What.** Self-play visits a narrowing set of states (same leads, same lines), so the value and policy are untrained off
it. **Measurement.** RND novelty on the observation (the X26 ride-along RND variants), visitation counts of a decoded
state key, off-distribution AUROC on held-out teams / exploiter games (`main.ridealong_read`, `rnd_states`).
**What we know:** RND tracks visitation (Spearman −0.34 against a coarse state key's count) and flags off-distribution
classes (AUROC ~0.75), but it does NOT separate the successors of starved near-best moves from other unplayed moves
(+0.001 [−0.037, +0.040]): novelty steers effort across STATES, not toward the right unplayed ACTION
(`measurements/ridealong_baseline/rnd_states_2026-09-30/`; UNDERSTANDING §4.4). So (d) and (a) need different
instruments, and an exploration bonus is not an anti-starvation lever.

### (e) Opponent-model collapse

**What.** The model of the opponent (α, the opponent pointer; B, the ride-along opponent-action contrast) concentrates on
few opponent actions, so the policy best-responds to a caricature. **Measurement.** α's intent NLL and OTHER rate against
the opponent's actual next action, its calibration by class (`main.belief_roles`); B's outcome information, R5 of the
ride-along registration (Brier(V) − Brier(V + A + B) at the played pair); B's per-action spread on rare opponent
actions. **What we know:** X5's hypothesis tokens predict the opponent's next action better than the blob did
(UNDERSTANDING §4.5); R5 on X26 is registered and not yet read.

### (f) Others the literature names, and which apply here

| type | what | our instrument / status |
|---|---|---|
| **Value-head collapse (under-dispersion)** | V's predictions shrink toward the mean; resolution falls while calibration may hold | `main.critic_gate`; X21 (native fix); the ride-along ensemble's R1 |
| **Plasticity / dormant-unit collapse** | the network loses the ability to fit new targets late in training (Nikishin et al. 2022 primacy bias; Sokar et al. 2023 dormant neurons; Lyle et al. 2023; **UNVERIFIED:** from memory) | `main.capacity` (capacity telemetry, cosine / velocity), dormant-unit share — no plasticity probe is registered on X26 (FINDING F-CC-6) |
| **Feature-rank collapse** | the representation's effective rank falls (Kumar et al. 2021 implicit under-parameterisation; **UNVERIFIED:** from memory) | the probe battery (`main.probe_battery`) reads representation use, not rank; a rank read is cheap to add |
| **Behavioural degenerate modes** | stalling to the 250-turn forfeit, one archetype for every team | the eval's draw / timeout bucket; per-team win rates (`team_win_rates` in `metadata.json`) |
| **Policy churn** | the greedy action flips often between nearby checkpoints (Schaul et al. 2022, **UNVERIFIED:** from memory) | `main.policy_drift`; the search ledger's maturity battery (15–22 % top-1 flips per 1M at 15M) |
| **Numerical near-ties** | K9(b)'s excluded share: forward-pass selections within 2e-4 of a cutoff | NOT a value-tie or starvation instrument: it measures numerics in the forward, not equal-value moves (`design_x26_readiness.md` §1.1) |

## 2. Counterfactuals: per-action values for the actions the policy does not take

A counterfactual here is an estimate of v(s, a) for every legal a, including the ones π starves. Four sources, from
cheapest-to-trust to cheapest-to-run.

### (i) One-ply rollout values from the Rust core (BUILT)

**What.** At a banked decision, branch every legal action, then play to the end under a declared continuation policy
(greedy, both sides) on S shared dice seeds (CRN: `utils.rust_env.successors.play_out`; `main.policy_spectrum.truth`;
the hidden-information study's harness for honest worlds). The value is "this action, then the continuation's play", a
DECLARED continuation, not a Nash value.

**Cost (measured).** This pilot (§3.1, read A), on CPU at rule 5b (one torch thread,
nice 19, the 3-core pin, beside a P0.1 runner and an anchor read): 306 turns × 6.9 legal actions × 32 seeds = 67,584
playouts to the end, ≈ 51 policy decisions each (both sides), in 3.5 h wall at ≈ 1.05 cores ≈ **3.7 CPU-h ≈ 43 CPU-s
per state** (≈ 0.2 CPU-s per playout; the CPU forward of the X26 network dominates, the sim does not). On the GPU the
Lane S truth took 49–61 min per 1,600 turns × 64 seeds (≈ 2 s per state, pre-era network). An honest-world playout
costs the same once its world is built; building and gating the worlds is the hidden-info harness's own cost
(`design_hidden_info_binding.md` §12), and the honest read needs MORE samples per action than the oracle one to say
anything (§3.5).

**Trust.**
- **True world** = an UPPER BOUND on what search could know: the opponent's whole team is the real one. It is the
  right referee for MEASURING starvation (were the starved moves good?), because it is the least noisy estimate of an
  action's value available.
- **Honest world** = the DEPLOYABLE read. The hidden-information study measured the gap: playing every action out in
  the true world improves the policy's pick by +0.081 (end state) / +0.071 (production) win probability per decision at
  8 rollouts; the FULLY honest 8-rollout leaf gains +0.006 [−0.004, +0.017]; G = oracle − honest = +0.070 [+0.060,
  +0.081], carried mostly by the never-seen Pokémon (`search_ledger.md` §1; PROVISIONAL at 15M). So an honest-world
  counterfactual today does not beat the policy's own choice. §3.5 measures what that means for DETECTING starvation.
- **Continuation dependence.** Values are conditional on the continuation. The Lane S read checked three continuations:
  of 84 verdicts none flipped sign. The pilot uses X26@150M greedy both sides.
- **Noise.** ±1 outcomes have SD ≈ 0.9, so a single action's mean at S = 32 carries SE ≈ 0.16; CRN pairing shrinks the
  GAP's SE (the quantity that classifies near-best), but at S ≤ 16 the near-best set is noisy, and the max of noisy
  means (V*) is biased up (the winner's curse). The pilot scores gains on held-out seeds.

### (ii) The ride-along A head (BUILT, detached, on X26)

**What it is.** A K = 5 bootstrapped ensemble over the pointer head's per-action tokens, centred under π, trained by MSE
on the GAE advantage of the action TAKEN (`design_q_head.md` §5.0a). Detached: it cannot move what X26 learns.

**What it can say.** A within-turn ranking of the actions the policy actually samples, and a per-action member spread
(epistemic uncertainty), at zero extra cost (one forward).

**What it cannot say.**
- About a starved action it has almost no labels: at π < 1 % the action is sampled < 1 % of the visits, so A(a) there
  is extrapolation through the shared scorer.
- Its label is GAE at the run's λ, V-bootstrapped, so A inherits V's within-turn blindness. The X4 pre-read measured that
  blindness: one-ply Q̂ from V put the STARVED near-best moves in its top-2 at chance (0.20–0.25 against ~0.27), and a
  cleaner V label did not fix it (the CRN label refit: the information that orders siblings is in the playouts, not in
  the successor's V).
- So A is a candidate UNCERTAINTY flag (R4: does its spread single out starved near-best moves?) more than a value
  estimate. R3 (A vs ground truth) and R4 are registered for X26's final read and read against `truth_v2` (pre-era
  continuations); this pilot's X26-continuation truth is a better on-distribution referee for them (§4).

### (iii) A learned Q head trained on rollout labels (DESIGNED, DEFERRED by the owner)

**What.** `design_q_head.md`'s Q(s, a, b) = V + A + B + I, with A and I trained by paired-difference regression onto
CRN playout outcomes (M = 1 per branch, breadth over repeats, §5.1). This is the route that would give EVERY action,
starved or not, a learned value from labels that carry the truth signal.

**Status.** The owner's 2026-09-30 label-scope rule (§5.0): only additions of at most one ply that can be subsampled
are in scope; anything that needs a playout to terminal to produce a value is DEFERRED, and a V-bootstrapped one-ply Q̂ is
never a distillation, anti-starvation or Q-head training target. Playout-outcome labels are recorded as the GROUNDED
option for when labels return.

**Cost.** `design_q_head.md` §8.1: a playout label ≈ 7 actions × ~12 remaining decisions ≈ 80–90 policy decisions per
side; the A layer at 1 / 2 / 5 % state coverage ≈ +0.8–0.9× / +1.6–1.8× / +4.0–4.5× the loop's decision rate (CPU
forwards infeasible beyond ~1 %, so the rows ride the GPU forward). This pilot's 0.2 CPU-s per playout (X26 network, one thread) is the measured
CPU anchor for that model: a label for one 7-action state at M = 1 ≈ 7 playouts ≈ 1.4 CPU-s.

**Trust.** As (i) for the label (true-world playouts during training would be an ORACLE label; honest-world labels are
the deployable ones and carry the honest gap). The learned head adds generalisation error; the CRN refit (ledger
2026-10-06) suggests the representation, not the label, binds sibling ordering, which is exactly what an
action-conditioned Q head changes.

### (iv) Search-improved policy targets: Gumbel completed-Q (DESIGN ONLY)

**What.** Gumbel MuZero (Danihelka et al. 2022, **UNVERIFIED:** from memory) builds the policy target
π' = softmax(logits + σ(completed Q)), where visited actions use their search Q and UNVISITED actions get the mixed value
v_mix, so the target is defined on every action, and the improvement is guaranteed in expectation even with few
simulations. Sampled MuZero (Hubert et al. 2021, **UNVERIFIED:** from memory) does the same over a sampled action
subset; MPO (Abdolmaleki et al. 2018, **UNVERIFIED:** from memory) uses softmax(Q/η) under a KL trust region.

**Why it is the bridge.** It turns counterfactual values (from (i) or (iii)) into a PPO-compatible policy target that
moves mass ONTO a starved near-best move in proportion to its value, which an entropy bonus cannot do (an entropy bonus
is a KL to the UNIFORM magnet: it feeds every legal action alike, good and bad; `design_learner_recipe.md` §3.9).

**Status and limits.**
- It needs a value per action that is BETTER than the policy's own ranking. One-ply from today's V is not (the X4
  pre-read: softmax(Q̂/τ) feeds starved moves no more than a uniform floor). True-world rollouts are, but they are an
  oracle; honest-world rollouts at 8 worlds do not beat the policy (search ledger).
- So completed-Q targets are as good as the counterfactual source under them; the binding constraint is the honest
  gap, i.e. the search programme's belief quality (`design_hidden_info_binding.md`; `search_ledger.md` §1 "how far").
- In scope only when the owner lifts §5.0's deferral (playout labels), or as an OFFLINE teacher.

### Summary

| source | built? | cost | knows WHICH starved move is good? | deployable? |
|---|---|---|---|---|
| (i) one-ply rollouts, TRUE world | yes | ≈ 43 CPU-s / state at 32 seeds (CPU); ≈ 2 s / state at 64 seeds (GPU) | **yes**: oracle precision 2.6× chance, a flagged move is worth +0.137 over the top choice (§3.5) | no (oracle): the right REFEREE for measuring starvation |
| (i) one-ply rollouts, HONEST worlds | yes (hidden-info harness) | the same per playout + world building | **no** at 8 worlds: precision 1.35× chance, a flagged move LOSES −0.068 (§3.5) | yes, but not yet useful |
| (ii) ride-along A head | yes (detached, X26) | one forward | unlikely: few labels on starved moves, V-bootstrapped; R4 decides | yes |
| (iii) Q head on rollout labels | designed | §8.1: +0.8–4.5× the loop's decision rate at 1–5 % coverage | as good as its labels (oracle: yes; honest: not yet) | DEFERRED (owner §5.0) |
| (iv) Gumbel completed-Q targets | design only | as (i) or (iii) underneath | as good as the values underneath | DEFERRED with (iii); usable offline |

## 3. The PILOT on X26 checkpoints

**Tag: MEASUREMENT, PILOT, DESCRIPTIVE.** Nothing here was registered before the data; every read is CPU, read-only on
`models/`, and its scripts and outputs are in
[`../research_state/measurements/collapse_pilot_2026-10-17/`](../research_state/measurements/collapse_pilot_2026-10-17/)
(`README.md` there has every command). X26 = `rb_x26_s2601`, pin `9c033489`, still training (the plateau meter's Tier 1
reads CLIMBING at 160M), so every trend below is PROVISIONAL in the sense of `feedback_strong_claims_need_plateau`: a
pre-plateau read.

### 3.1 What was run

| read | states | per-action values | policies read | cost |
|---|---|---|---|---|
| **A. X26's own truth** (the primary pilot read) | a NEW bank from X26's own eval traces at 30M / 90M / 150M (`build_bank.py`: 7,028 decisions in 210 battles, gate ① byte-equal on every recorded decision, 7,385 incl. the single-legal ones), a stratified subset of 306 turns (276 free + 30 forced switches, ≤ 2 per battle, every non-attack category legal on ≥ 30) | every legal action × 32 shared dice seeds played to the end in the TRUE world, both sides X26@150M greedy | X26 @ 10 / 30 / 50 / 90 / 130 / 150 / 160M | 3.5 h wall, ≈ 3.7 CPU-h (≈ 43 CPU-s per state) |
| **B. The pre-era truth** (a second, independent referee) | Lane S `bank_v1` / `truth_v2`: 1,600 turns from the ai_v14 lineage's eval traces | every legal action × 64 seeds, TRUE world, under THREE pre-era greedy continuations (K2 / N0 / C_fix finals) | the same X26 checkpoints, beside N0@75M and K3 (the lineage that starved) | forward passes only (~75 s per checkpoint) |
| **C. Honest vs oracle detectability** (post hoc) | the value-discrimination pilot's registered 600 states per checkpoint (`rb_es_end2_s2001`, `rb_es_prod_s2001`, both 15M) with a fully honest (F) row: 1,097 states, 777 battles | truth = 24 held-out true-world seeds; estimators = 8 / 16 oracle leaf seeds, the 8 honest F worlds, the critic one-ply on each | the two 15M checkpoints | zero playouts (`honest_vs_oracle_starvation.py`, ~1 min) |
| **D. Entropy** | X26's bank (7,028) and `bank_v1` (20,712) | — | all eleven X26 checkpoints | forward passes |
| **E. Strategic** | X26's dense snapshot ladder, its eval records, its plateau ledger | — | every snapshot | reads only |

### 3.2 (b) Entropy: NO entropy collapse on X26

On X26's own bank (all 7,028 turns), entropy in nats / rank-1 mass:

| X26 @ | 2M | 10M | 20M | 30M | 50M | 70M | 90M | 110M | 130M | 150M | 160M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 1.510 | 1.088 | 1.099 | 1.050 | 1.162 | 1.126 | 1.089 | 1.103 | 1.108 | 1.119 | 1.112 |
| rank 1 | 0.419 | 0.594 | 0.587 | 0.609 | 0.562 | 0.575 | 0.589 | 0.581 | 0.581 | 0.572 | 0.579 |

- The policy sharpens by 10M and then stays put: 10M → 160M rank 1 −0.016 [−0.026, −0.001], entropy +0.024 [−0.008,
  +0.049]; 30M → 160M entropy +0.062 [+0.037, +0.087] (it BROADENED after 30M). Forced switches alone keep sharpening
  (10M → 160M entropy −0.076 [−0.106, −0.045]). Battle-clustered 95 % intervals, `spectrum_x26bank.md`.
- On `bank_v1` the same checkpoints read 1.13–1.18 nats after 10M, the level of the ai_v14 lineage's N0@75M (1.161) and
  above K3's (0.993), the starving end of that lineage.
- This is consistent with the `ent_coef` 0.05 regime holding entropy, as it did for v8 (no ablation here). **Type (b)
  is not what X26 shows.**

### 3.3 (a) Starvation on the pre-era truth (read B): FLAT, at the healthy lineage's level

Decisive turns (948–950 of 1,600), starved share; ranges over the three continuations. The readout reproduces the
published Lane S levels exactly (N0@75M 0.31–0.35, K3 0.48–0.49):

| policy | starved | of which TIE-COLLAPSE (a1) | of which MISRANK (a2) | dominated mass | greedy regret |
|---|---|---|---|---|---|
| N0@75M (ai_v14) | 0.311–0.348 | 0.095–0.103 | 0.216–0.249 | 0.345–0.379 | 0.300–0.342 |
| K3@99M (ai_v14) | 0.476–0.486 | 0.137–0.150 | 0.333–0.347 | 0.328–0.361 | 0.300–0.333 |
| X26@10M | 0.315–0.335 | 0.089–0.103 | 0.225–0.241 | 0.354–0.383 | 0.309–0.339 |
| X26@30M | 0.360–0.384 | 0.105–0.110 | 0.250–0.274 | 0.340–0.362 | 0.295–0.316 |
| X26@90M | 0.326–0.359 | 0.099–0.111 | 0.217–0.248 | 0.324–0.358 | 0.286–0.311 |
| X26@150M | 0.310–0.354 | 0.101–0.116 | 0.209–0.238 | 0.325–0.356 | 0.281–0.306 |
| X26@160M | 0.337–0.359 | 0.101–0.109 | 0.233–0.250 | 0.327–0.353 | 0.288–0.310 |

Paired, same turns, battle-clustered 95 % (per continuation K2 / N0 / C_fix):
- **X26@10M → X26@160M: starvation NOT DETECTED** (+0.023 [−0.012, +0.055] / +0.013 [−0.018, +0.043] / +0.021 [−0.008,
  +0.053]); **dominated mass FALLS** (−0.029 [−0.043, −0.015] / −0.030 [−0.044, −0.015] / −0.027 [−0.042, −0.012]);
  **greedy regret FALLS** (−0.019 [−0.042, −0.001] / −0.029 [−0.051, −0.008] / −0.025 [−0.042, −0.005]). That is the
  signature of HEALTHY sharpening, 3 of 3 continuations.
- A small bump at 10M → 30M (starved +0.034 to +0.049, detected 3/3) gives itself back by 90M.
- **X26@160M starves less than K3** (−0.128 to −0.141, 3/3 detected), at the level of N0@75M (N0@75M → X26@160M
  −0.005 to +0.025, not detected).
- **Most starvation is MISRANK, not tie-collapse:** about two thirds of the starved decisive turns have a policy
  top choice that is itself NOT near-best (0.21–0.25 of decisive turns against 0.09–0.12).
- The held-out gain of the best starved move over the argmax on decisive turns is +0.07 to +0.10 (±1 scale), flat
  across X26.

**Caveats.** These states were visited by the ai_v14 lineage, not by X26, and the continuations are pre-era policies:
"near-best" means near-best when K2 / N0 / C_fix play the rest. Read A is the on-distribution read.

### 3.4 (c) Strategic: slow transitive climbing with a small, real cyclic component

- **Hodge decomposition of the dense ladder** (90 players, 1,673 edges, `python -m main.elo` at 162M): spine 941 ELO;
  cyclic width RMS 47 ELO raw against a 34 ELO bootstrap null, **excess 32 ELO, p = 0.005**; null-adjusted cyclic
  energy 3.1 % of the flow. Twenty significant 3-cycles, e.g. 66M > 138M > 72M > 66M (+295 ELO curl, z 4.7) and 90M >
  108M > 160M > 90M (z 3.8). The cycle is real and small next to the spine.
- **Plateau meter Tier 1** (newest vs the node 50M back, GSPRT): GAIN at every check 60M–160M, Tier 1 = CLIMBING. The
  pairs a check needs grew from 120–360 (60–120M) to 520–1,120 (130–160M): the edge per 50M is shrinking.
- **Sentinels** (greedy, symmetric teams, 100 games each): at 34M the newest node beat a 32M-older sentinel 0.91; from
  ~60M on, sentinels 30–90M older read 0.39–0.63 (162M: 0.49–0.56 against 28–90M-older ones). The dense ladder's
  headline is +48 ELO for 152M over 68M (± 17). So X26 is still climbing, slowly, and not chasing a cycle; its
  pool-relative edge is now a few points of win rate per 50M.
- Bots are saturated (0.92–0.95 mean win rate since ~20M), so they cannot see (c).

### 3.5 Can a DEPLOYABLE counterfactual see starvation? (read C: honest vs oracle)

Pooled over `rb_es_end2_s2001` / `rb_es_prod_s2001` at 15M; ±1 scale; brackets battle-clustered 95 %. The truth on these
states: starved share of decisive turns 0.402 [0.364, 0.439] (end2 0.301, prod 0.508), 2.99 near-best of 7.5 legal per
turn. "Chance" flags every π < 1 % action, so its precision is the base rate.

| estimator | precision (decisive) | recall (decisive) | within-turn Spearman vs truth | value of playing the flagged move vs the policy's top-1 | IMP, win prob |
|---|---|---|---|---|---|
| ORACLE, 8 seeds | 0.501 | 0.525 | 0.464 | **+0.137 [+0.104, +0.169]** | +0.072 |
| ORACLE, 16 seeds | 0.587 | 0.518 | 0.539 | +0.182 [+0.146, +0.217] | +0.086 |
| **HONEST F, 8 worlds** | **0.260** | **0.301** | **0.134** | **−0.068 [−0.105, −0.031]** | +0.002 [−0.007, +0.011] |
| critic one-ply, oracle worlds | 0.257 | 0.397 | 0.197 | +0.015 [−0.020, +0.051] | +0.015 |
| critic one-ply, honest worlds | 0.234 | 0.429 | 0.137 | −0.035 [−0.070, +0.001] | +0.001 |
| chance | 0.194 | 1 | — | −0.101 | 0 |

- **An honest-world one-ply read CANNOT tell which starved moves are good**: precision 1.35× chance (the oracle 2.6×),
  and playing the move it flags LOSES −0.068 against the policy's own top choice. The critic is no better.
- **The oracle can**, and what it finds is worth a lot: a starved move it flags beats the policy's top choice by +0.137
  (≈ +7 pp win probability) on the flagged turns. The value of the starved moves lives in the hidden information.
- The referee itself is noisy: per-action SE 0.124 at the truth's 24 seeds, and split-half near-best sets agree at
  Jaccard 0.53. At 8 samples "within ε of the best" means "tied with the best" (steps of 0.125 > ε).
- **Consequence for counterfactual TARGETS:** a completed-Q or Q-head target built from honest-world rollouts at the
  scale we can afford today would push mass onto the WRONG starved moves. Un-starving with counterfactuals waits on the
  search programme's belief quality (§5).

### 3.6 (a) Starvation on X26's own states and continuation (read A)

306 turns, 0 refused; 179 decisive (58 %); 2.86 near-best actions per turn of 6.9 legal (1.72 on decisive turns).
±1 scale (win probability = half); battle-clustered 95 %; S = 32.

| X26 @ | 10M | 30M | 50M | 90M | 130M | 150M | 160M |
|---|---|---|---|---|---|---|---|
| starved (decisive turns) | 0.296 [0.231, 0.368] | 0.307 | 0.251 | 0.279 | 0.296 | 0.324 | 0.285 [0.228, 0.341] |
| … TIE-COLLAPSE (a1) | 0.089 | 0.067 | 0.050 | 0.056 | 0.084 | 0.073 | 0.067 |
| … MISRANK (a2) | 0.207 | 0.240 | 0.201 | 0.224 | 0.212 | 0.251 | 0.218 |
| held-out gain of the best starved move over the argmax, per STARVED decisive turn | +0.39 [+0.25, +0.57] | +0.36 | +0.36 | +0.37 | +0.35 | +0.37 | +0.35 [+0.23, +0.47] |
| IMP, rollout argmax over the policy's (decisive / all turns) | +0.31 / +0.18 | +0.30 / +0.17 | +0.31 / +0.17 | +0.28 / +0.15 | +0.26 / +0.14 | +0.26 / +0.14 | +0.27 / +0.15 |
| greedy regret (decisive) | 0.400 | 0.390 | 0.395 | 0.373 | 0.360 | 0.357 | 0.354 |
| dominated mass (decisive) | 0.344 | 0.344 | 0.343 | 0.348 | 0.337 | 0.333 | 0.334 |

Paired (same 179 decisive turns):
- **10M → 160M: starvation NOT DETECTED, −0.011 [−0.090, +0.061]**; misrank +0.011 [−0.054, +0.061]; dominated mass
  −0.010 [−0.041, +0.017]; **greedy regret FALLS −0.046 [−0.091, −0.001]**. 10M → 150M: starved +0.028 [−0.040,
  +0.098], not detected.
- At the nested 16 seeds the 10M → 150M step reads +0.081 [+0.006, +0.169] (detected), and 10M → 160M +0.034 [−0.039,
  +0.103]. The 16-seed read is the noisier one (149 decisive turns, a coarser near-best set); the 32-seed read is primary.
  With ±0.08 intervals on 179 turns, the pilot can exclude an ai_v14-sized rise (+0.14) over X26's life, not a small one.

**What read A says.**
1. **X26 is not getting worse at starvation as it trains.** About 3 in 10 decisive turns starve a near-best move, at
   every checkpoint, while greedy regret falls. Read B (independent states and continuations) says the same with
   tighter intervals.
2. **Three quarters of X26's starvation is MISRANK, not tie-collapse.** On those turns the top choice is worse than a
   move the policy puts under 1 % on.
3. **When it misranks, it misranks big, but only with the oracle's knowledge.** On a starved decisive turn the best
   starved move beats the argmax by +0.35 to +0.39 (held out; ≈ +18 pp win probability) in the TRUE world under X26's
   own continuation. Per decisive turn that is +0.08 to +0.11. The honest-world read (§3.5) cannot find these moves, so
   most of that value is hidden information that a deployable search cannot see yet; how much of it a player could
   recover is the search programme's question, not a training-recipe one.
4. **The oracle IMP is +0.14 to +0.18 on all turns (≈ +7–9 pp win probability per decision)**, the same order as the
   hidden-info study's oracle leaf (+0.081). It reads +0.18 at 10M and +0.15 at 160M (DESCRIPTIVE: no paired interval
   was read on it).

**Caveats.** One continuation (X26@150M greedy both sides; not Nash); the true world (oracle); states from the 30 / 90 /
150M cycles, so 10M is read partly off its own distribution and 160M slightly so; 179 decisive turns; the bank draws
2 wins + 2 losses + 1 draw per (cycle, opponent) against a ~94 % bot win rate, so losses and draws are over-represented
and levels are not population rates; paired deltas are the robust part.

## 4. The recommended measurement battery on X26's trajectory

One battery, read every 20M of X26 (and on each X35 breaker arm's final), all CPU at rule 5b's priority, reusing what
exists. "New" marks the only pieces that do not exist today.

| # | type | read | instrument | cost per checkpoint | status |
|---|---|---|---|---|---|
| B1 | (b) entropy | rank-mass spectrum + entropy per stratum on X26's own bank, paired against the previous read | `main.policy_spectrum read` / `report` on `collapse_pilot_2026-10-17/bank_x26` | ~30 s | exists |
| B2 | (a) starvation, on-distribution | starved / tie-collapse / misrank shares, dominated mass, held-out gain, IMP, greedy regret on the 306-turn X26 truth | `starvation_readout.py` on read A's rows | forward pass only | exists (this pilot) |
| B3 | (a) starvation, cross-era referee | the same on `truth_v2` | `readout_oldtruth.sh` | ~75 s | exists (this pilot) |
| B4 | (a) truth refresh | re-branch read A's turns under the NEWEST checkpoint's greedy continuation once per ~50M, so "near-best" tracks the policy's own continuation (the 150M truth ages as X26 moves) | `main.policy_spectrum.truth run` | ≈ 3.7 CPU-h (306 turns, S = 32, CPU); minutes on the GPU | exists |
| B5 | (a) + (ii) | R3 (A vs ground truth) and R4 (A's spread on starved near-best moves) on read A's truth as well as `truth_v2` | `main.ridealong_read` | minutes | **new: `--truth-dir` reads only `truth_v2`'s three file names (`TRUTH_CONTINUATIONS`); it needs a generic rows argument** (F-CC-4) |
| B6 | (c) strategic | Hodge width vs its null; plateau Tier 1; sentinel win rates by age | `main.elo --out <dir outside models/>`, `main.plateau status`, `strategic_series.py` | seconds | exists |
| B7 | (c) exploitability | a fresh best responder's gap at P\* (and at one earlier node) | `main.best_response_gap` | GPU, per X35 | exists |
| B8 | (d) coverage | RND variants' coverage / saturation series | `main.ridealong_read` variants | minutes | exists |
| B9 | (e) opponent model | α's intent NLL / OTHER rate by class; R5 (B's outcome information) | `main.belief_roles`, `main.ridealong_read` | minutes | exists |
| B10 | (f) plasticity | dormant-unit share and a fixed-target refit probe on frozen checkpoints | **new** (F-CC-6) | minutes | not built |
| B11 | (c) pool STALENESS | promotions per 50M; the self-play pool's median member age and oldest member (the pool at t = the 20 latest promotions ≤ t, FIFO, checked against `snapshots/`); the pool-relative edge `eval/win_rate_vs_pool`. Promotion is a noise crossing when the edge sits near the 0.55 bar, and its rate depends on `--n-sentinels`, so compare runs only at equal sentinel counts | `main.ops.tb_read <run> --tag train/selfplay_promoted_steps --tag eval/win_rate_vs_pool` (a fork's own rows); the series: `../research_state/measurements/x35_breakers_stage_2026-10-17/batch2_lr_floor/lr_pool_series.py` | seconds (record reads) | exists |
| B12 | (f) the LEARNING RATE in force | the adaptive-KL controller's rate (`train/learning_rate`, rungs 3e-4 / 1.2^k), with approx KL, gradient norm and clip fraction beside it. Reported next to EVERY plateau or Tier-1 claim and every fork comparison: a plateau read at a falling rate is read at a falling weight-space dose, and one rung (20 %) trips the 10 % dose-gap rule | `python -m main.dose <run>` (`lr_now`, the controller block); `main.ops.tb_read <run> --tag train/learning_rate --tag train/approx_kl` | seconds | exists |

**Deployable vs oracle.** B2–B4 measure starvation with an ORACLE referee (true-world rollouts). That is right for
MEASURING it. It is not a deployable signal (§3.5), and no read in this battery may be turned into a training target
without the owner lifting `design_q_head.md` §5.0.

## 5. Levers by collapse type, tied to the plateau-breaker phase and the search programme

| type | what X26 shows (§3) | levers, cheapest first | where it runs |
|---|---|---|---|
| (a1) tie-collapse | 0.05–0.09 of decisive turns on its own truth (0.09–0.12 on the pre-era one), flat | an entropy floor already feeds tied moves (it is a uniform magnet). Its real cost is EXPLOITABILITY, which only a best responder can read (B7); the lever there is a MIXED target at guess turns: a trailing magnet (R-NaD / MMD, `design_learner_recipe.md` §3.9) or completed-Q over the tied set | X23 re-scoped (trailing magnet vs anneal vs control), inside X35 |
| (a2) misrank starvation | 0.20–0.25 of decisive turns, ~3/4 of starvation (~2/3 on the pre-era truth); flat; worth +0.35 per starved decisive turn in the TRUE world only | needs a counterfactual that knows WHICH starved move is good. Honest one-ply cannot (§3.5); the oracle can. So: (1) the search programme's belief quality first (`search_ledger.md` "how far"); (2) then honest-world playout labels into a Q head and Gumbel completed-Q targets, which need the owner to lift §5.0; (3) meanwhile one-ply search at inference is a plateau-breaker that does NOT train on counterfactuals | X35's one-ply-search breaker; X4 when labels return |
| (b) entropy collapse | not present | keep `ent_coef` 0.05 (or its trailing-magnet replacement); watch B1 | — |
| (c) strategic | slow transitive climb, small real cycle (32 ELO excess width) | PFSP / league (AlphaStar), exploiters, team curriculum (X30), a fresh generalist against the pool (X29) | X35, the plateau-breaker phase |
| (d) coverage | not read here; RND does not target starved actions | exploration bonuses (RND) steer STATE coverage; they do not un-starve actions | X35 if B8 shows narrowing |
| (e) opponent model | not read here | belief quality (X5's hypothesis tokens), honest determinization (the search programme) | the search programme |

**How this ties to the plateau-breaker phase.** X26's own data say its sharpening is healthy and its starvation flat,
so starvation is not the reason X26 is slowing; the evidence points at (c) (a shrinking pool-relative edge) and at
the information ceiling (§3.5: what the starved moves are worth is in the hidden information). The breakers that
address those are PFSP / league and search, in X35's order. Starvation reads (B2, B4) ride along every breaker arm as a
GUARD: a breaker that sharpens the policy (one-ply search distilled, a lower entropy) must not do what the ai_v14
lineage did (starvation +0.14 to +0.16 for −0.018 dominated mass). **X35's guard thresholds (2026-10-17, [`design_compute_efficiency.md`](design_compute_efficiency.md) §5.6.5):** an arm
FAILS G-C when B1's ON − OFF entropy interval lies wholly below −0.05 nats, or when B2's starved share shows a detected
rise (the interval's lower bound > 0). B3 and B6 are reported; B4 refreshes the truth once if P\* is ≥ 50M past
150M. No GPU arm targets anti-starvation (X35-A7 is trigger-gated).

## 6. FINDINGS

- **F-CC-1 (HAZARD, rule 6, caused by this pilot).** `python -m main.elo models/rb_x26_s2601` was run WITHOUT `--out`
  and rewrote `models/rb_x26_s2601/elo/elo_ratings.json` and `elo_curve.png` in the LIVE run's directory
  (2026-10-17 ~13:10 PDT). They are derived outputs of the same tool, re-fitted from the append-only
  `snapshot_ladder/games.jsonl`, and nothing in training reads them; the previous copies (written 2026-10-16 15:42) are
  not recoverable. Proposed fix (TECH_DEBT, not done): `main.elo` should refuse to write under `models/` unless asked.
- **F-CC-2.** No DEPLOYABLE counterfactual knows which starved move is good today (§3.5): an honest-world 8-world read
  flags at 1.35× chance and acting on it loses −0.068. Any anti-starvation TARGET built from honest one-ply values now
  would push mass onto the wrong moves. Read C is on the 15M end-state / production checkpoints, not on X26 (no honest
  worlds exist for X26's states); transfer to X26 is ASSUMED.
- **F-CC-3.** Read A's power is limited: 179 decisive turns, intervals ±0.07–0.08. It excludes an ai_v14-sized rise
  (+0.14), not a small one. The nested 16-seed read detects the 10M → 150M step (+0.081 [+0.006, +0.169]) where the
  32-seed read does not (+0.028 [−0.040, +0.098]); the 32-seed read is primary.
- **F-CC-4.** `main.ridealong_read --truth-dir` reads only `truth_v2`'s three file names (`TRUTH_CONTINUATIONS`), so R3 /
  R4 cannot be read against an on-distribution X26 truth without a small reader change (battery B5).
- **F-CC-5 (scope).** The brief's counterfactual options (iii) and (iv) conflict with the owner's 2026-09-30 label-scope
  rule (`design_q_head.md` §5.0: playout labels DEFERRED; V-bootstrapped one-ply Q̂ never a target). They are written up
  as DEFERRED designs here; lifting the deferral is the owner's call.
- **F-CC-6.** No plasticity / dormant-unit read is registered on X26 (battery B10, not built).
- **F-CC-7 (CPU).** The truth run took 3.5 h wall, not the forecast 2.8 h: it ran beside a P0.1 runner, an anchor read and
  (for ~30 min) this pilot's own checkpoint reads, so for that half hour this pilot ran TWO jobs on the agent cores
  against the brief's one. Total pilot CPU ≈ 4.2 CPU-h (truth 3.7, checkpoint reads ≈ 0.4, the rest minutes), under the 8 CPU-h
  budget.
- **F-CC-8.** Read B re-encodes `bank_v1`'s pre-era states with HEAD's encoder (obs as recorded 0 / 19,964, expected
  after the era's encoder changes); its truths are pre-era continuations. It is a cross-era referee, not X26's own.
- **F-CC-9.** K9(b) (named in the brief as "near-ties in logits") measures numerical near-ties in the forward pass, not
  equal-value moves. It is not a starvation instrument (§1 (f)).
- **F-CC-10.** The 100-game greedy sentinels (±0.05) are too coarse for (c) at this stage: they read ≈ 0.5 against
  nodes 30–90M older, while the dense ladder reads +48 ± 17 ELO for 152M over 68M. Use the dense ladder and Tier 1.
- **F-CC-11.** Literature citations in §1–§2 are from memory and marked UNVERIFIED.
- **F-CC-12 (from read C).** At 8 samples "within ε of the best" means "tied with the best" (steps of 0.125 > ε), and
  the referee itself is noisy (split-half near-best Jaccard 0.53 at 24 seeds), so every precision / recall is held down
  by the referee.

## 7. Decision record

| date | decision | chosen | rejected | evidence |
|---|---|---|---|---|
| 2026-10-17 | ESTABLISHED (owner request: "what type of collapse … we need some way to give counterfactuals") | The taxonomy (a1 tie-collapse / a2 misrank / b–f), each with its instrument; true-world one-ply rollouts as the REFEREE for starvation, honest-world ones as the deployable read; the battery B1–B10 on X26 every 20M and on every X35 arm, starvation as a GUARD on breakers | Treating entropy as the collapse meter (X26's entropy is flat while ~3 in 10 decisive turns starve); one undivided "starvation" number (it hides that 3/4 is misrank); an anti-starvation GPU arm on X26 now (its starvation is flat, and no deployable counterfactual can target the right moves, §3.5) | §3; `../research_state/measurements/collapse_pilot_2026-10-17/` |
| 2026-10-17 | Counterfactual targets (iii) Q head on rollout labels and (iv) Gumbel completed-Q | Written up as DEFERRED designs, in line with the owner's 2026-09-30 label-scope rule (`design_q_head.md` §5.0); their usefulness is gated on honest-world value quality, i.e. the search programme | Recommending them now: honest one-ply values flag the wrong starved moves (−0.068 vs the top choice) | §2, §3.5 |
| 2026-10-17 | The battery as X35's guard (design worker, for the orchestrator) | G-C = B1 (FAIL iff the ON − OFF entropy interval is wholly below −0.05 nats) + B2 (FAIL iff a detected rise in the starved share), on every b-ON and b-OFF final; B3 / B6 reported; an anti-starvation arm NOT run, with three named triggers | a guard on B4–B10 (cost; B5 and B10 not built); a fixed starvation bar of +0.05 (read A cannot resolve it, F-CC-3) | `design_compute_efficiency.md` §5.6.5 |
| 2026-10-20 | The battery READ on X26's plateau (180M–380M), every offline item | B1–B3, B5, B8, B9 (α), B10's dormant share (built as a measurement script) read; B4, B7, B10's refit probe, R5 SCOPED, not run; the pool's composition, churn and the update size added as descriptive reads | — (a read, not a decision; pool staleness as a battery item is PROPOSED, not adopted) | `../research_state/measurements/x26_collapse_battery_2026-10-20/` |
| 2026-10-20 | Pool STALENESS (B11) and the LEARNING RATE in force (B12) PROMOTED to named battery items (the battery read's F3 / F4) | Both are REPORTED on every 20M read and on every X35 arm final. Their readers already exist: TB scalars through `main.ops.tb_read`, `main.dose`, and the `snapshots/` directory. No code was built | Adding either to the G-C guard: no bar is registered, and B11 is confounded by A2's sentinel count (launch sheet §0 TM-1). A new reader module: the reads are record reads that existing tools serve | `../research_state/measurements/x26_collapse_battery_2026-10-20/` F3 / F4; the toy's series in `../research_state/measurements/x35_breakers_stage_2026-10-17/batch2_lr_floor/` |
