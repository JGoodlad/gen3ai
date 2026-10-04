# Evaluation end state — the unified system: ledger, scheduler, estimators, budget

**Status: ALWAYS-CURRENT.**
- The skeleton went in 2026-10-02: the owner said "put the skeleton in and the TODOs, and we fill it out with our implementation over time".
- **The DESIGN went in 2026-10-03,** written by the eval-system design agent. It is a PROPOSAL awaiting an independent review and the owner's answers to §11.2; nothing in it is built yet unless marked BUILT.

This doc is the SYSTEM: how eval evidence is produced, stored, scheduled and estimated. The population DECISIONS (promotion, eviction, plateau) live in [`design_league_decisions.md`](design_league_decisions.md). A build or decision that differs from this doc updates it and its Decision record in the same commit. Era context: [`era_plan_post_m5.md`](era_plan_post_m5.md). The evidence behind the numbers this design measured on banked data is in [`measurements/eval_design_2026-10-03/`](../research_state/measurements/eval_design_2026-10-03/README.md).

**Direction (owner):** robust, durable, auditable, and reusable later to answer new questions WITHOUT a new training run.
The owner's standard (2026-10-03):
- review the literature;
- prefer a grounded method to a homebrew one;
- make every decision deterministically (standing rule 8);
- list open gaps honestly.

**Map of this doc.**

| § | what |
|---|---|
| 0 | principles |
| 0a | literature review, applied to our setting |
| 0b | the LEDGER |
| 0c | reuse rules |
| 0d | the one scheduler |
| 1 | the questions eval answers |
| 2 | T20, the tiered pool matrix, Nash mixtures and the validation rule |
| 3 | statistics in use |
| 4 | T19, background eval and GPU ownership |
| 5 | the budget derived from precision |
| 6 | re-grounding the week-one assumptions |
| 7 | TODO |
| 8 | interfaces for the plateau test |
| 9 | determinism, audits and the seat RNG |
| 10 | build plan |
| 11 | open gaps, owner questions, findings |

## 0. Principles
1. **Every eval DECISION is deterministic, with a declared error rate.** That means a sequential test with stated α/β,
   or a margin that a measured noise floor justifies. It is never a threshold we hope the noise stays on the right
   side of. A decision whose statistic falls within a declared rounding band of its boundary is NOT a crossing (§9).
2. **Mirrored team pairs, with the PAIR as the unit,** for every SYMMETRIC comparison: pool eval, promotion, the
   pool matrix, the plateau test, anchors. Pinned-team meters (untaught, `best_response_gap --play`) stay unmirrored,
   because mirroring would mix piloting with response (`eval_and_rating.md`). The mirror's job is BALANCE: both
   sides pilot both teams, exactly. Its variance gain is small: ~12 % of games on a cross edge (§9.3).
3. **The regime is recorded on every row** (greedy or sampled, mirrored, the eval core, the protocol version, the seat
   rule). Readers refuse to mix regimes.
4. **A cheap method is trusted only after it reaches the dense method's decisions on our own data.**
5. **Re-ground week-one choices** (§6). Several eval parameters were set by what one CPU could afford in week
   one, not by the precision a question needs.
6. **One GPU owner at a time, never a queue of waiting agents** (owner, 2026-10-03).
   - While a training run is live, the TRAINER is the GPU owner. Every GPU eval game is played inside its process,
     in its startup-declared eval slots (§4).
   - Everything else runs on the CPU lane, which never initializes CUDA, or waits for an offline window when no run
     is live.
   - No eval component ever blocks on the GPU. Since `5876c2ea` the GPU is LEASED for an agent's lifetime
     (`scripts/ops/gpu_lease.sh`, granted by the orchestrator), and `gpu_lock` fails fast (`GpuLeased` / `GpuBusy`)
     instead of waiting. During a run the training agent holds the lease and the trainer runs under it.
7. **Estimates read COUNTS from the ledger, never traces, never a selected sample** (§0c).

## 0a. Literature review, applied to our setting

**Scope.** These are the references the owner listed in T20 and in the brief, each verified against a primary or
index source (arXiv, the publisher, Project Euclid, PMLR, AAAI, JMLR) on 2026-10-03. The items marked
**UNVERIFIED** could not be checked from a primary source in this pass. Every row ends with what the reference
implies FOR US.

### 0a.1 Rating and ranking under non-transitivity

| reference | key result | implication for us |
|---|---|---|
| Elo, *The Rating of Chessplayers, Past and Present* (Arco, 1978); Bradley & Terry, "Rank analysis of incomplete block designs I", *Biometrika* 39:324–345 (1952); Herbrich, Minka & Graepel, "TrueSkill™", NIPS 19 (2006/07) | One scalar strength per player, P(i beats j) = σ(r_i − r_j). TrueSkill adds a Gaussian uncertainty per player and an online message-passing update | **Keep BT as the SPINE** (the ladder, `main.elo`), because it is the right model where the data are transitive. On 70 banked within-run ladders, BT fits within noise (§11.3 F-ED-1). BT cannot represent cycles, so it is never the only read of a multi-lineage pool. TrueSkill's online update buys nothing over a batch BT fit on counts we keep forever |
| Balduzzi, Tuyls, Perolat & Graepel, "Re-evaluating Evaluation", NeurIPS 2018, arXiv:1806.02643 | For an antisymmetric payoff matrix there is a UNIQUE maximum-entropy Nash equilibrium (Prop. 4). The Nash average is invariant to redundant copies of an agent (Thm 1 P1). An ε-perturbed matrix gives an ε-Nash (P2). The equilibrium is uniform on the top-rated player(s) when the game is transitive (P3). It is found by an LP. The paper gives no sampling rule | **Two consequences, one good and one a trap.** (a) Copy invariance is exactly what our archive needs: consecutive snapshots are near-duplicates, and uniform averaging would over-weight eras with many snapshots. (b) In a near-transitive archive the Nash is PURE on the frontier. On N0's matrix the point estimate puts all weight on one node, and its posterior spreads over the top ~8 (F-ED-3). So "zero Nash weight" is the COMMON case and cannot alone decide eviction (§2.4). The reference mixture must be FROZEN per check, and its uncertainty handled explicitly (§2.3, §8) |
| Omidshafiei et al., "α-Rank: Multi-Agent Evaluation by Evolution", *Sci. Rep.* 9:9937 (2019), arXiv:1903.01373 | Ranks by the stationary distribution of an evolutionary chain over pure profiles. Handles general-sum and asymmetric games. As α → ∞ the ranking depends only on the response graph (the sign of each single-deviation comparison) | Our game is symmetric and zero-sum, where Nash averaging is unique and cheaper. **α-Rank stays in reserve**, for a general-sum question (e.g. team-vs-team asymmetric matchups) if one arises |
| Rowland et al., "Multiagent Evaluation under Incomplete Information", NeurIPS 2019, arXiv:1909.09849 | **ResponseGraphUCB:** keep the unresolved edges; sample from them (uniform, uniform-exhaustive or valence-weighted); drop an edge once its confidence intervals separate (Hoeffding, or the tighter Clopper–Pearson for Bernoulli outcomes); stop when none remain. With Hoeffding, O(Δ⁻² log(1/(δΔ))) samples suffice w.p. ≥ 1 − 2δ (Thm 4.2), where Δ is the smallest payoff gap. Payoff intervals also propagate into intervals on ranking weights | **The T2 targeted tier's sampling rule** (§2.2): an edge is unresolved iff its Clopper–Pearson interval contains ½ AND it bears on an undecided support/eviction decision. We adopt the stopping semantics, not the α-Rank target. The 1/Δ² cost says that resolving near-½ edges between neighbouring snapshots (Δ ≈ 1–3 pp) is unaffordable by design, so a decision must never NEED one |
| Jiang, Lim, Yao & Ye, "Statistical ranking and combinatorial Hodge theory", *Math. Programming* 127:203–244 (2011), arXiv:0811.1067 | Any edge flow on a comparison graph splits orthogonally into a GRADIENT part (the global ranking, by weighted least squares on the graph Laplacian), a CURL part (triangle-local cycles) and a HARMONIC part (cycles around longer loops, consistent on every triangle). Local consistency implies global only when the clique complex has no harmonic part (β₁ = 0) | **The cycling meter** (`hodge.py`, BUILT for trainee × bots). For pool × pool: report gradient / curl / harmonic shares against a noise floor computed from the same counts. **A sparse schedule must contain triangles:** a chain of "new vs previous" comparisons hides every cycle. That is why the T1 probe plays a spanning set WITH chords (§2.2). The density window in which random graphs have β₁ = 0 (Kahle, cited by Jiang et al.) is **UNVERIFIED** (the PDF text was garbled) |
| Czarnecki et al., "Real World Games Look Like Spinning Tops", NeurIPS 2020, arXiv:2004.09468 | Strategy space has a transitive axis and a non-transitive width. The width is largest at middling skill and shrinks toward the top. If the population contains a full Nash cluster, beating all its members guarantees transitive improvement (Thm 3). Fixed-memory fictitious play converges when the population is at least the size of the lowest occupied layer (Prop. 3) | **The pool cap should be set by the measured Nash-cluster size, not by habit.** Within one lineage the cluster is SMALL: no banked within-run pool shows a cyclic component above noise, and N0's posterior Nash support is ~1–8 nodes. So 20 is ample there. The open question is the MULTI-lineage pool (exploiters, forks) of era step 1, which is unmeasured. The T20 meter must measure it before the cap changes |
| McKelvey & Palfrey, "Quantal response equilibria for normal form games", *Games Econ. Behav.* 10:6–38 (1995) | A smoothed equilibrium (logit QRE) that is continuous in the payoffs | A grounded ALTERNATIVE if the posterior-mean Nash (§2.3) proves too unstable. Not adopted; listed so the choice is explicit |

### 0a.2 Active and sparse evaluation

| reference | key result | implication for us |
|---|---|---|
| Heckel, Shah, Ramchandran & Wainwright, "Active ranking from pairwise comparisons and when parametric assumptions do not help", *Ann. Statist.* 47:3099–3126 (2019), arXiv:1606.08842 | Rank by the Borda score τ_i = P(i beats a uniformly random opponent) with successive elimination on confidence intervals. The query count is ≤ c·log(n/δ)·Σ_i f(Δ_i) with f(x) ≈ log log(1/x)/x², instance-optimal up to log factors. Parametric models (BTL/Thurstone) buy at most logarithmic factors | **Two uses.** (a) The cost of an active ranking is dominated by the items near a decision boundary (Σ 1/Δ_i²). So spend games ONLY where a decision is undecided: the T2 rule. (b) Assuming BT "to save games" is not a real saving, so the cycling read never leans on BT to impute edges a decision needs. Borda against a uniform archive is the model-free alternative reference (§11.2 Q5); its weakness is the duplicate sensitivity Balduzzi fixes |
| Yue, Broder, Kleinberg & Joachims, "The K-armed dueling bandits problem", *JCSS* 78:1538–1556 (2012); Bengs, Busa-Fekete, El Mesaoudi-Paul & Hüllermeier, "Preference-based online learning with dueling bandits: a survey", *JMLR* 22(7) (2021) | Online best-arm selection from noisy duels. The survey sorts the problem by the winner concept (Condorcet, Copeland, Borda, von Neumann) and by the assumptions each needs | Our "best member" target under cycles is the **von Neumann winner, i.e. the Nash mixture**, which is the survey's map for choosing it. Regret minimization is the wrong objective: eval games do not cost wins, they cost GPU/CPU time, so we use a pure-exploration (best-arm-identification) rule, i.e. Rowland/Heckel, not a regret algorithm |
| Du, Yan, Chen, Wang & Zhang, "Estimating α-Rank from a few entries with low rank matrix completion", ICML 2021 (PMLR 139:2870–2879); Rashid, Zhang & Ciosek, "Estimating α-Rank by maximizing information gain", AAAI 2021 (35(6):5673–5681); Chen & Joachims, "Modeling intransitivity in matchup and comparison data", WSDM 2016 | Payoff matrices of similar-skill agents are near low rank, so O(n·r·log n) entries suffice. A Bayesian information-gain match picker beats ResponseGraphUCB on samples. The blade–chest model extends BT to cycles with low-dimensional vectors | **The ARCHIVE-scale tool**, for hundreds of snapshots across runs. It is not needed for an active pool of ≤ 40, where incremental dense is cheap (§2.1). If we ever estimate a cross-run archive matrix sparsely, a rank-r completion with the BT spine as its first factor is the grounded choice. It is validated against a dense audit like every cheap method (principle 4). Not built now |

### 0a.3 Reusing old evidence

| reference | key result | implication for us |
|---|---|---|
| Ibrahim & Chen, "Power prior distributions for regression models", *Stat. Sci.* 15:46–60 (2000) (author order **UNVERIFIED**: Project Euclid's page listed Chen & Ibrahim; the standard citation is Ibrahim & Chen) | π(θ \| D₀, a₀) ∝ L(θ \| D₀)^{a₀}·π₀(θ), 0 ≤ a₀ ≤ 1: each old observation counts as a₀ of a new one | **The T0 tier's form.** Old games of a matchup (cycle games around a snapshot's birth, SPRT games of an accepted candidate) enter an ESTIMATE as a₀-weighted counts, which is trivial on count rows. **a₀ = 0 until validated** (§2.2): a fixed a₀ chosen by taste is exactly the homebrew the owner rules out |
| Duan, Ye & Smith, "Evaluating water quality using power priors to incorporate historical information", *Environmetrics* 17:95–106 (2006); Hobbs, Carlin, Mandrekar & Sargent, "Hierarchical commensurate and power prior models…", *Biometrics* 67:1047–1056 (2011) | The NORMALIZED power prior lets a₀ carry its own prior and be learned. The commensurate prior θ_new ~ N(θ_old, 1/τ) learns how much to borrow and borrows LESS when old and new data conflict | **How a₀ gets set:** estimated from our own data, per SOURCE CLASS (cycle rows, SPRT rows), on the first dense audits, by the commensurate model (τ per class). Then it is FROZEN and registered. A class whose estimated τ says "conflict" (e.g. selection-inflated SPRT rows) gets a₀ = 0 |
| Meta-analysis (fixed- vs random-effects pooling) | Pooling across studies needs a between-study variance term when the studies differ | Already our practice for RUN-level reads (P0's σ_h, X5 §7.4): run strength is a random effect, and the meter noise is subtracted. The ledger keeps rows per (batch × matchup), so a random-effects pool over batches is always possible later |

### 0a.4 Sequential testing and repeated monitoring

| reference | key result | implication for us |
|---|---|---|
| Wald, "Sequential tests of statistical hypotheses", *Ann. Math. Statist.* 16:117–186 (1945); Van den Bergh, "Comments on normalized Elo" and the Fishtest GSPRT notes (fishtest mathematics, official-stockfish docs) | The SPRT; Fishtest's pentanomial GSPRT over colour-swapped game PAIRS, solved as a constrained multinomial MLE. Its worst-case length at α = β = 0.05 is ≈ 1,046,535/(e₁ − e₀)² games on the normalized-Elo scale | **BUILT (`sprt.py`)** and kept for promotion. The ledger stores per-batch pentanomial counts, so every SPRT verdict is re-derivable from rows (§9). One thing to correct: the SPRT is RESUMED across a restart rather than abandoned (§4.4) |
| Lan & DeMets, "Discrete sequential boundaries for clinical trials", *Biometrika* 70:659–663 (1983) (DOI and pages from memory, **UNVERIFIED**); O'Brien & Fleming, *Biometrics* 35:549–556 (1979); Jennison & Turnbull, *Group Sequential Methods…* (Chapman & Hall/CRC, 2000) | Alpha-spending: only the spending function α(t) is fixed in advance; the looks need not be pre-scheduled. O'Brien–Fleming boundaries are conservative early | Already in use (X5 §7.4). **For the plateau monitor (G1):** a run's plateau checks are an open-ended sequence. Alpha-spending needs a declared maximum information. The ledger supplies the information fraction at every look, from row counts per request (§8) |
| Howard, Ramdas, McAuliffe & Sekhon, "Time-uniform, nonparametric, nonasymptotic confidence sequences", *Ann. Statist.* 49:1055–1080 (2021), arXiv:1810.08240; Waudby-Smith & Ramdas, "Estimating means of bounded random variables by betting", *JRSS-B* 86:1–27 (2024); Johari, Koomen, Pekelis & Walsh, "Always valid inference", *Oper. Res.* 70:1806–1821 (2022) | Confidence sequences valid under CONTINUOUS peeking, with width shrinking at the iterated-logarithm rate (stitched boundary, Howard et al. eq. 11). Betting confidence sequences are the tightest known for [0, 1] outcomes such as win or pentanomial scores. Mixture-SPRT p-values are built for dashboards people peek at | **The natural fit for a monitor that never ends**, i.e. the plateau monitor and the live dashboard reads, where alpha-spending's "maximum information" is artificial. Recommended to the plateau unit as G1's first candidate (§8). It needs nothing new from the ledger beyond per-batch rows |
| Page, "Continuous inspection schemes", *Biometrika* 41:100–115 (1954) | CUSUM change detection with a declared average run length to false alarm | Optional refinement for plateau ONSET (G1). It needs a fixed check spacing, which the scheduler supplies (§8) |

### 0a.5 League practice

| reference | key result | implication for us |
|---|---|---|
| Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning", *Nature* 575:350–354 (2019) | A league per race (main agents, main exploiters, league exploiters); prioritized fictitious self-play, sampling opponent B with weight f(P[A beats B]). Past players are KEPT, not evicted. The 35 / 50 / 15 % opponent split and the exact f_hard form come from secondary sources (paywalled), **UNVERIFIED** | Two lessons: GROW and sample rather than evict (already the owner's 2026-10-02 decision), and the PFSP weight is the natural eviction tie-break. A member the trainee beats almost always carries almost no training weight anyway (§2.4) |
| Lanctot et al., "A unified game-theoretic approach to multiagent reinforcement learning" (PSRO), NeurIPS 2017, arXiv:1711.00832 | Build an empirical meta-game matrix by simulation, solve it for a meta-strategy, train a best response; the meta-game IS the evaluation | **Our pool matrix is PSRO's empirical meta-game**, and `main.best_response_gap` is its exploitability read. The ledger's job is to make that matrix durable and reusable across rounds |
| OpenAI et al., "Dota 2 with Large Scale Deep Reinforcement Learning", arXiv:1912.06680 (2019) | Training opponents: 80 % latest and 20 % past, with past opponents sampled by a learned quality score. Evaluation used 83 FROZEN reference agents with HELD-CONSTANT ratings; a test agent plays only references within 15–85 % win rate; ratings inflated for newer agents when the game version changed | **Frozen references with held ratings is exactly the G3 answer** (§8): a check's reference mixture is frozen and versioned, and scores against it are comparable. "Only references within 15–85 %" is a Fisher-information argument: a game against a far-weaker reference carries almost no information. Our scheduler weights reference opponents by the frozen mixture, not uniformly. Their version-change inflation is our regime boundary (principle 3) |

### 0a.6 RL evaluation methodology and variance reduction

| reference | key result | implication for us |
|---|---|---|
| Agarwal, Schwarzer, Castro, Courville & Bellemare, "Deep RL at the Edge of the Statistical Precipice", NeurIPS 2021, arXiv:2108.13264 (bibliography verified; method details from memory) | With few runs, point estimates mislead. Report interquartile means, stratified-bootstrap CIs, performance profiles and the probability of improvement | Already our rule for RUN-level claims (P0, X5 §7.4: across-seed inference, never a within-run CI). **For the ledger:** a read must say whether its CI is conditional on the run (meter noise only) or across runs (adds σ_h). The reader API carries that as a declared field (§0b.7) |
| Henderson, Islam, Bachman, Pineau, Precup & Meger, "Deep Reinforcement Learning That Matters", AAAI 2018, arXiv:1709.06560 | Seeds, implementation details and environment nondeterminism flip conclusions | The ledger records the commit, the protocol, the seed rule and the compute block on every row, so any row can be replayed and audited (§9.2) |
| Burch, Schmid, Moravčík, Morrill & Bowling, "AIVAT: A new variance reduction technique for agent evaluation in imperfect information games", AAAI 2018, arXiv:1612.06915 | Unbiased control variates built from a value function. They cut the SD of a heads-up no-limit hold'em match by 85 % (~44× fewer games). Duplicate play (same deals, seats swapped) is the classic special case | **Our mirrored pairs are duplicate play,** and they buy only ~12 % on a cross edge (F-ED-4), because the outcome variance is mostly NOT team luck. AIVAT-style control variates using our own value head are the grounded next step if eval games ever bind. **Not adopted now:** it needs a calibrated value, and our critic's calibration is itself under study (era step 3). Recorded as a gap (§11.1). A standalone primary citation for "duplicate poker" was not found (**UNVERIFIED**) |
| Domhan, Springenberg & Hutter, IJCAI 2015; Swersky, Snoek & Adams, "Freeze-thaw Bayesian optimization", arXiv:1406.3896 (2014) | Learning-curve extrapolation (a parametric ensemble by MCMC); a GP over exponential decays deciding which paused runs to resume | For the PLATEAU unit's G2 (a current slope rather than a window average). The ledger supplies the scores-vs-a-frozen-reference series it needs (§8) |

### 0a.7 What the review rules in and out

- **IN:**
  - the BT spine;
  - max-entropy Nash averaging against a FROZEN reference;
  - HodgeRank shares with a noise floor;
  - ResponseGraphUCB-style targeted sampling;
  - the power prior with a₀ estimated, not chosen;
  - the pentanomial GSPRT;
  - confidence sequences for open-ended monitoring;
  - frozen reference agents with held ratings (the Dota 2 practice).
- **OUT for now, in reserve:**
  - α-Rank (our game is zero-sum and symmetric);
  - matrix completion (only at archive scale);
  - regret-minimizing dueling bandits (wrong objective);
  - AIVAT (needs a calibrated value);
  - TrueSkill (no gain over batch BT on kept counts).
- **CORRECTED by the review:**
  - "evict at zero Nash weight" is near-vacuous in a transitive pool (§2.4);
  - "T0–T2 to save games" saves almost nothing for an active pool of ≤ 40, because incremental dense costs (K − 1)·n
    games per new member (§2.1).

## 0b. The eval LEDGER — append-only, aggregated (owner, 2026-10-02)

**One append-only JSONL ledger of COUNTS, not of games.** The owner put it as: "per-opponent would be great, running counters for teams; we don't need the richest data ever".
- **One row per (batch × matchup).** A matchup is one player against one opponent, under one regime, for one purpose.
- **Why counts are enough:** they are SUFFICIENT STATISTICS for every estimator we plan:
  - Bradley–Terry / Elo and HodgeRank need the win counts per pair;
  - Nash averaging needs the pair win-rate matrix;
  - the GSPRT's verdict needs the pentanomial counts;
  - the power prior needs the counts by regime and purpose;
  - per-team reads need the team counters.

  **What is given up:** per-game covariates (game length, the exact matchup sequence) and replaying a sequential
  test's PATH inside a batch. Neither has a planned reader. A batch is the test's check interval (§0d), so the path at
  the test's own resolution IS recoverable.

### 0b.1 What exists (BUILT, 2026-10-03, X5 P0, `54b78aed`)
`agents/training/eval_ledger.py` provides:
- the row schema `gen3_eval_count_row_v1` and its validator, which checks shapes, closed vocabularies and the arithmetic
  tying the blocks together (W + L + D = games; half-points = 2W + D; team counters sum to the games and wins);
- one shard per writer process, with an fsync per row;
- a globbing reader that validates every row and drops superseded ones;
- a refusal of any output directory under `models/`.

**Two writers exist.**
- `main.h2h`: 120 rows under `measurements/x5_p0_h2h_2026-10-03/rows/`, about 25 KB per row raw (a 530-team counter
  map) and 3.7 KB gzipped.
- The Rustboro-era bot round robin (`measurements/bot_base_ratings_2026-10-03/bot_rr.py`, `3ccef556`): 1,296 rows,
  purpose `anchor`, bots as players (`id: bot:<name>`; `sha256` = the digest of the name + the bot modules' sources),
  sampled vs sampled, and seat-balanced (`mirror_rule: gen3_mirrored_pairs_v1+seat_alternating_by_pair`).

It already shows two conventions v2 makes explicit: a bot identity, and a seat rule carried inside a free-text field.

### 0b.2 The schema, v2 (`gen3_eval_count_row_v2`)
v2 keeps every v1 field and meaning. **A v1 row is never rewritten:** the reader upgrades it on read, with the
defaults below. The additions are what the producers and the reuse rules need:

| field | v1 | v2 | why |
|---|---|---|---|
| `purpose` | closed list: promotion, plateau, matrix, audit, anchor, training, cycle | **+ `ab`, `ladder`, `untaught`, `gap`** (owner's closed list, §11.2 Q1) | F-P0-6 (a pre-registered A/B read had to be written as `audit`); the snapshot ladder, the untaught meter and the exploiter gap each need their own purpose so a reader can include or exclude them by name |
| `request` | — | `{id, kind, opened}`, or `null` only on a backfilled row | **The peeking rule made exact** (§0c rule 1): a sequential decision reads ONLY the rows of its own request. `kind` ∈ {cycle, sprt, plateau_check, matrix_dense, matrix_target, matrix_probe, ab_cell, anchor_read, untaught_read, gap_read, audit_replay, audit_dense, adhoc} |
| `player.kind`, `opponent.kind` | (checkpoints only) | `checkpoint` \| `bot` \| `external` | Bots and outside agents become first-class opponents. For a non-checkpoint, `sha256` is the digest of (agent name, version, code commit), and `path` is `bot:<name>` / `ext:<agent>@<version>` |
| `regime.protocol` | — | a named eval-protocol version, e.g. `gen3_eval_protocol_v2` | **Regime boundaries become explicit.** Every SEMANTIC change to what a game measures bumps it: the 2026-09-07 sentinel regime, F-LH-13's multi-team fixed opponent, a turn-limit change, a team-builder change. Readers never pool across a protocol (standing rule 15) |
| `regime.seat_rule` | (implicit: the player keeps p1) | `fixed_p1` \| `balanced` | §9.3: balance seats across pairs so a seat effect cancels by construction instead of by measurement |
| `regime.player_temp`, `regime.opponent_temp` | — | `null` when greedy, else the sampling temperature | The 2026-09-07 bug was a greedy-vs-T = 1.0 edge pooled with greedy-vs-greedy (+8.9 pp); `sampled` alone does not say at what temperature |
| `regime.team_set` | (`team_source`, free text) | a digest of the team set and builder parameters; `team_source` stays as the human label | The identity must change when the TEAMS change, not when a label does |
| `counts.aborted` | (only `pairs.voided` on mirrored rows) | games started and not finished, on every row | The standing INCONCLUSIVE rule ("timeouts > 25 % of attempted battles") needs the denominator |
| `flags` | — | a closed list: `draws_folded`, `teams_unrecorded`, `seed_unrecorded`, `sha_unrecorded`, `eval_core_unrecorded` | Backfilled rows say exactly what they lack. A reader must DECLARE which flags it accepts (§0b.7), so a draw-blind legacy row cannot slip into a score estimate silently |
| `provenance` | — | `{source_file, source_line, source_sha256, backfill_id}`, or `null` | Backfill auditability: every backfilled row points at the byte range it came from |

**The v1 → v2 upgrade on read is deterministic.** It sets:
- `request = null`;
- `kind = bot` when the id starts `bot:`, else `checkpoint`;
- `protocol = gen3_eval_protocol_v1_<writer>` (one value per v1 writer: `h2h`, `bot_rr`);
- `seat_rule = balanced` when `mirror_rule` contains `seat_alternating_by_pair`, else `fixed_p1`;
- both temperatures `null` for greedy rows; for the bot round robin's sampled rows, the bots' own sampling (recorded
  as `opponent_temp = player_temp = "bot_native"`);
- `team_set` = the digest of v1's `team_source` string;
- `aborted = 2 × pairs.voided`;
- `flags = []`.

`regime_id` is recomputed over the v2 identity, and the v1 id is kept as `regime.v1_id`.

**Two companion streams,** in the same append-only, one-writer-per-file discipline:
- **`decisions/`** (`gen3_eval_decision_v1`): one row per DECISION. Each records:
  - `decision_id`, `kind` (promotion / eviction / plateau / ab_verdict) and `subject`;
  - the `request_id` it decided, and the digest and count of the rows it consumed;
  - the rule and its version, the verdict, and `ts`.

  **Rows are never edited after a decision.** A reader that must exclude a node's SELECTION sample (§0c rule 3) joins
  the rows to the decision that consumed them.
- **`references/`** (`gen3_eval_reference_v1`): immutable FROZEN reference mixtures for strength reads (§2.3). Each
  records:
  - `ref_id`, the members (sha256), the weights and the solver with its version;
  - the posterior draw count and seed;
  - the digest of the ledger rows it was solved from, and `created_at`.

  **A reference is never modified.** A re-solve creates a new one.

**The scheduler's requests** live in `requests/` (`gen3_eval_request_v1`). They are open / claim / done / cancel events,
one file per writer, so the queue state is a deterministic fold of events (§0d).

### 0b.3 Where it lives
- **Archive-level:** `<archive>/_ledger/`, where `<archive>` = `utils.paths.run_archive_dir()` (main's `models/`, or
  `$GEN3AI_MODELS_DIR`). It is never per run, so cross-run questions need no new run. Pytest SEALS the archive (the
  `run_archive` fixture), so tests write to a sealed temp archive like every run writer.
- **Layout:**
  - `_ledger/rows/<producer>/ledger.<writer_id>.jsonl`, compressed to `.jsonl.gz` once closed;
  - `_ledger/decisions/`, `_ledger/requests/`, `_ledger/references/`;
  - `_ledger/backfill/manifest.json`;
  - `_ledger/README.md`, a pointer to this section.
- **One writer per file.** `writer_id` = UTC time + host + pid + producer. Rows are fsynced per append, so a batch
  survives a kill.
- **Closing a shard:** the writer gzips its own shard at a clean exit. A shard whose writer died is gzipped by `python -m
  main.eval_ledger close-stale`, which acts only when the pid is dead AND the file has been idle 24 h, so the
  one-writer rule holds.
- **`refuse_under_models` changes meaning:** a ledger writer may write ONLY under `<archive>/_ledger/`. The rest of
  `models/` stays refused.
- **Retention:** the ledger is NEVER deleted (`models_retention_policy.md` gets one line). Size, DERIVED from the
  per-row size above and the §5 budget: ~650 rows ≈ 16 MB raw / 2.4 MB compressed per 10M training steps, which is ~20
  MB compressed per 75M run.

### 0b.4 The schema gate
1. **Writer:** validate before append (BUILT).
2. **Reader:** validate every row and refuse on the first malformed one, naming the file and line (BUILT).
3. **`python -m main.eval_ledger audit`** validates the whole archive and the CROSS-row invariants:
   - `row_id` is unique;
   - every `supersedes` and every `request` resolves;
   - one regime per request;
   - every `protocol` is known;
   - the decision → rows digests match.

   The scheduler runs it at startup and refuses to start on a failure.
4. **A static gate** (`src/eval_ledger_reader_gate_test.py`, empty allowlist) has two parts:
   - (a) every module that reads the ledger passes a `ReaderDecl` (§0b.7). It is AST-checked, like
     `trace_summary_reader_gate_test.py`.
   - (b) the closed lists (`PURPOSES`, request kinds, flags, protocols) equal this section's tables, in the
     `recipe_doc_gate` pattern. A new purpose is then a doc change and a code change together, never one alone.
5. **A contract test per producer:** each producer's row builder emits a synthetic row that passes `validate_row`.

### 0b.5 Producers and their migration

| producer | what it writes today | v2 purpose / request kind | rows per | migration (unit, §10) |
|---|---|---|---|---|
| In-loop eval cycle (`eval_callback`, `selfplay_callback` → `rust_eval/`) | `<run>/eval_results.jsonl`: per cycle, rates and `[won, finished]` per opponent; **no draw count** (draws only in `eval_manifest.json`, 370 of 1,081 manifests) | `cycle` / `cycle` | per opponent per cycle | **U2:** DUAL-write a ledger row per opponent beside `eval_results.jsonl` (its TensorBoard/TUI/`main.elo` readers move later), with exact W/L/D and per-team counters from the eval core |
| SPRT promotion (`sprt_promotion.py`) | `<run>/sprt_promotion.jsonl` (state only; pentanomial pooled; scratch games deleted). **No run has one** | `promotion` / `sprt` | per batch (40 pairs) per pool member | **U2:** rows per batch + a decision row; **RESUME instead of abandon** at a restart (§4.4) |
| Snapshot ladder (`snapshot_ladder.py`) | `<run>/snapshot_ladder/games.jsonl`: `{a, b, wins_a, games, source}`; no draws, teams, seed or sha | `ladder` / `matrix_dense` | per pair | **U2:** dual-write; **U7:** the FIT reads the ledger |
| `main.h2h` | v1 rows to a caller-named directory, purpose `audit` | `audit` / `ab` / `plateau` / `matrix` as the caller declares | per batch | **U1:** archive default; `--purpose` adds `ab` |
| `main.anchors` | per-GAME `games.jsonl` + `summary.json` (temp dir by default) | `anchor` / `anchor_read` | per batch per opponent | **U3b:** aggregate to rows at the end of a read; the per-game file stays as the SOP's artifact |
| `main.untaught_meter` | per-team cells `{wins, ties, losses, games}` + `_meta` | `untaught` / `untaught_read` | per (player, opponent, team slice) | **U3b** |
| `main.best_response_gap` | READS `eval_results.jsonl` externals | reader of `cycle` rows (opponent = the frozen target); `--play` writes `gap` | — | **U3b** |
| `hodge.py` | reads cycle edges in process | reader | — | **U7** |
| Bot round robin (`data/gen3_bot_elo_*`) | `data/` (source of truth, poke-env-free) | not migrated: `data/` is the acquisition layer | — | none |
| Training games per team (`team_winrate_callback`) | `metadata.json` | `training` (sampled regime; a different population) | — | not now; optional later |

### 0b.6 The backfill (U3)
**One writer id (`backfill-<date>`), idempotent.** `_ledger/backfill/manifest.json` records each source file's sha256.
A re-run refuses unless a source changed, and then it writes SUPERSEDING rows, never edits.

| source | count on disk (2026-10-03) | maps to | recoverable | NOT recoverable → flag |
|---|---|---|---|---|
| `snapshot_ladder/games.jsonl` | 104 files, 5,766 rows, of which 824 are `source: eval_cycle` copies (DROPPED: counted in the cycle source) | `ladder`, greedy/greedy, unmirrored; protocol by `recipe_version` (absent = pre-2026-09-27 recipe v2) | wins/finished per pair; sha256 of each snapshot zip that still exists (hashed at backfill time) | draws (`draws_folded`); teams; seed; eval core (`*_unrecorded`); sha of a deleted snapshot |
| `eval_results.jsonl` | 265 run dirs, 2,071 rows (exact `counts` on 1,926; sentinel counts on 324; externals on 471) | `cycle`, one row per opponent per cycle | W/finished per opponent; draws from the cycle's `eval_manifest.json` `battles_drawn` where present | draws elsewhere (`draws_folded`); teams; sentinel sha (step only) |
| committed `measurements/anchors_*/` | per-game rows | `anchor` | W/L/D, per-team counts, regime; pairs where mirrored | our checkpoint's sha if the file is gone |
| P0 rows (`x5_p0_h2h_2026-10-03/rows/`) | 120 v1 rows | copied verbatim (v1, upgraded on read) | everything | — |
| bot round robin (`bot_base_ratings_2026-10-03/ledger/`) | 1,296 v1 rows | copied verbatim (v1, upgraded on read) | everything | — |

**The regime of a cycle row,** inferred in this order:
1. the row's `sentinel_regime` stamp (authoritative; 385 rows, all greedy);
2. else the run's `model_config.json` `eval_sentinel_greedy`;
3. else pre-boundary ASYMMETRIC (sentinel sampled at `--self-play-temp`, flat team builder).

Bot edges are regime-safe on both sides of the 2026-09-07 boundary; sentinel edges are not. **But the bots themselves
changed:** the setup-branch fix (`f885ad8f`) and Curse-as-setup (owner, 2026-09-29) put every pre-fix bot edge in
the old bot era, and `30ff42f3` installed the Rustboro-era bot anchors as an era boundary for bot-anchored headlines.
So a backfilled bot edge gets the bot identity of its era (`sha256` from the bot sources at the row's commit when it
can be resolved, else `sha_unrecorded`), and readers never pool bot eras. New rows carry the bot-source digest, so the
boundary is automatic from U2 on. F-LH-13 (a multi-team
fixed opponent evaluated on its first team only) gets its own protocol value on the 20 affected runs' `ext_` rows.
**No RUN's eval history holds pentanomial data** (no run ever enabled `--eval-mirrored-pairs`), so every backfilled
run row is unmirrored. The only mirrored rows are the two measurement writers' (P0 and the bot round robin).

### 0b.7 The reader API
Every estimator reads through ONE function and DECLARES what it consumes:

```
DECL = ReaderDecl(
    name="sprt_promotion",
    purposes={"promotion"},
    regime=RegimeFilter(protocol="gen3_eval_protocol_v2", play="greedy", opponent_play="greedy", mirrored=True),
    requests="own",            # "own" = only the caller's request id (a sequential decision); "any" = an estimate
    selection="exclude",       # drop the rows a decision consumed to select the node being estimated
    flags_ok=frozenset(),      # accept no legacy flag
    inference="conditional",   # "conditional" (meter noise only) or "across_runs" (adds the run term)
)
rows = eval_ledger.read(DECL, request_id=..., players=..., as_of=...)
```

- **One regime per call.** The call refuses (`MixedRegimeError`) if the rows span more than one `regime_id`. A report
  that wants several regimes asks for each separately; nothing pools them.
- **`as_of`** (a timestamp or a run step) restricts rows AND decisions to those that existed then. That is the
  time-travel a BACK-TEST needs so it cannot peek at the future (the N0 back-test, §8).
- **`selection="exclude"`** implements the ladder's recipe-v3 rule ("the games that selected a snapshot do not rate
  it") through the decision stream, for every estimator.
- **The static gate** (§0b.4) fails any read without a declaration.

## 0c. Reuse rules (one ledger, many readers)
1. **A sequential DECISION counts only rows produced FOR it, after it started.** That is SPRT's peeking rule, made exact
   by `request.id`. Older rows enter only as a declared, discounted prior (the power prior, §0a.3), and only after
   validation (§2.6). ESTIMATES (ratings, the matrix, Hodge) may pool every eligible row.
2. **Pool only rows of the same regime** (`regime_id`, which includes `protocol`).
3. **Never estimate from a SELECTED sample.**
   - The prober's trace quota prefers losses, so estimates read the ledger's counts, never traces.
   - The games that selected a node (the cycle that crossed a threshold, the SPRT that accepted it) do not rate that
     node (`selection="exclude"`).
   - Rule 1 already keeps them out of the node's own test.
4. **Legacy rows are opt-in.** A reader that accepts a `draws_folded` row says so in its declaration. Draws were 0.68 %
   of P0's games, so the bias is small, but it is never silent.
5. **A back-test reads `as_of` its decision time.**

## 0d. One scheduler

**What it is.** ONE request queue and ONE allocation rule, serving every eval consumer, playing on whichever lane is
eligible (§4), and appending counts to the ledger.

**A request** (`gen3_eval_request_v1`, an `open` event) declares:
- `id`, `kind` and priority class;
- the consumer;
- the players: a checkpoint sha, or a DESIGN such as "new member vs every pool member";
- the regime;
- a target: a fixed pair count, or a sequential rule id + cap;
- a latency class (`urgent`, `routine`, `idle`);
- the eligible lanes;
- the FROZEN inputs it depends on: the pool membership at an SPRT's start, the `ref_id` of a plateau check.

Requests are durable events. `claim` / `done` / `cancel` events follow, with a reason on every cancel.

**Priority classes** (owner order: promotion > plateau > matrix > audits):

| class | consumer | request kinds | latency | lanes |
|---|---|---|---|---|
| P0 | the monitoring CYCLE (fixed, not queued) | `cycle` | every 2M steps | GPU window |
| P1 | promotion SPRT | `sprt` | urgent: the pool waits on it | GPU window |
| P2 | plateau check | `plateau_check` (+ floor reads) | routine: once per 10M | CPU, else GPU window |
| P3 | pool matrix | `matrix_dense`, then `matrix_target`, `matrix_probe` | routine | CPU, else GPU window |
| P4 | audits | `audit_replay`, `audit_dense` | idle | CPU; offline GPU |
| P5 | one-offs | `ab_cell`, `anchor_read`, `gap_read`, `untaught_read`, `adhoc` | per request | offline GPU (foreign architecture) or CPU |

**The allocation rule (deterministic given the open requests, the rows and the budgets):**
1. The quantum is ONE BATCH, which is a sequential test's check interval: 40 mirrored pairs for SPRT and plateau, and 40
   pairs per edge for matrix kinds.
2. Within a lane's window, serve requests in the order (class, latency, `opened`, `id`).
3. **Anti-starvation:** when P3–P4 have open requests, they get at least 20 % of each window's games. A strict queue
   would never audit while promotions run.
4. **Within P3, value of information** (the review's choice, §0a.2):
   - FIRST the dense-incremental rows of new members: each new edge to n₀ = 100 pairs, §2.1.
   - THEN targeted top-ups: an edge is ELIGIBLE iff its Clopper–Pearson 95 % interval contains ½ AND at least one
     endpoint is UNDECIDED for the support/eviction read (posterior P(w > ε) in [0.05, 0.5], §2.4). Eligible edges are
     sampled uniform-exhaustively (Rowland et al.'s scheme), up to a per-edge cap of 1,000 pairs.
   - When nothing is eligible, the matrix is DONE for that pool.
5. **Lane routing:** among a request's eligible lanes, the CPU lane first for P2–P5, unless the CPU lane's measured
   rate would miss the request's latency class. P1 always goes to the GPU window.

**Determinism.** Allocation can depend on measured wall time (budgets, §4), so WHICH batch runs WHEN is not bit-
reproducible. That is harmless: every DECISION is a deterministic function of its own request's rows (§9.1). The
allocation is AUDITABLE, because every row names its request.

## 1. The questions eval answers (decisions: `design_league_decisions.md`)

| question | meter / test | status |
|---|---|---|
| Is plain training still paying? (**plateau**) | strength vs a FROZEN reference mixture of the archive (§2.3), its slope against < 2 Elo per training GPU-hour, AND the best-response gap not falling, with the plateau KIND diagnosed | DECIDED (league §C); the test is the NEXT unit. This doc supplies its interfaces (§8) |
| Does a candidate join the pool? (**promotion**) | GSPRT on mirrored pairs, H0 0.50 / H1 0.55, α = β = 0.05, cap 1,680 pairs (`sprt_promotion.py`) | BUILT, default OFF (`e9c5ab2d`); flips ON at X26. Ledger rows, decision rows and resume-on-restart in U2 |
| Who stays in the pool? (**eviction / cycling**) | max-entropy Nash on the pool matrix as a NECESSARY condition, the PFSP weight as the tie-break, the HodgeRank curl/harmonic share against its noise floor as the cycling meter (§2.4) | DESIGNED here (§2); the decision rule itself is league §B's (§11.2 Q6) |
| How strong is it? (**rating**) | `snapshot_ladder/ladder.json` (dense BT, recipe-stamped) at run end, matched snapshot count | BUILT (`main.elo`); U7 re-points its fit at the ledger |
| Is the loop working? (**exploiter gap**) | `main.best_response_gap`: must FALL round over round | BUILT; U3b moves its reads to the ledger |
| Against the outside world? (**anchors**) | `main.anchors` (SmallRL at milestones; Kakuna/Foul Play at run end), greedy vs greedy, mirrored | BUILT; cadence TODO (TASK_BACKLOG T18); U3b writes rows |
| Is one arm non-inferior to another? (**pre-registered A/B**) | the X5 cross design (`design_x5_belief_tokens.md` §7.4) | BUILT as `main.h2h`; purpose `ab` (§11.2 Q1); multi-cell engine U6 |
| Does the value tell states apart? (**discrimination**) | TODO: one registered within-game discrimination meter, by phase and opponent class | TODO (era step 3) |

## 2. T20 — the pool matrix, tiered (TASK_BACKLOG T20)

### 2.1 What the tiers are for: a correction to the skeleton
The skeleton's tiers (T0 reuse, T1 sparse probe, T2 targeted, T3 dense audit) were meant to save games against a dense
matrix of 190 pairs × 200 games ≈ 38k games. **That cost is only paid if the matrix is rebuilt.**

A frozen pair is a stationary quantity, so dense can be INCREMENTAL. Each new member plays the K − 1 others ONCE, which
is the snapshot ladder's pay-once tax (`snapshot_ladder.py`):
- K = 20 at 100 pairs per edge: 19 × 200 = **3,800 games per new member**, about 60 s in the GPU window or ~13 min on the
  CPU lane (derived from §5's rates);
- K = 40: 7,800 games.

**So, for the ACTIVE pool (≤ 40 members), incremental dense IS the default**, and it is its own ground truth. The sparse
tiers are needed only:
- (a) at ARCHIVE scale: a cross-run matrix over hundreds of snapshots, where (K − 1) · n per member stops being cheap;
- (b) for the targeted top-ups that resolve the edges a decision needs.

This cuts T20's build (§10 U7) and its risk. It is an owner choice (§11.2 Q4).

### 2.2 The tiers, as built under that correction

| tier | what | where it applies | status |
|---|---|---|---|
| **T3 dense, incremental** | each new member × every member, n₀ = 100 mirrored pairs per edge (`matrix_dense` requests at promotion) | the active pool, always | DESIGNED (U7). The ladder's 100-game edges become 100-PAIR mirrored edges: same order of games, balanced teams |
| **T2 targeted** | top-ups on ELIGIBLE edges only (§0d rule 4: interval contains ½ AND bears on an undecided member), uniform-exhaustive, cap 1,000 pairs per edge (Rowland et al. 2019) | the active pool | DESIGNED (U7) |
| **T1 sparse probe** | a new archive member vs a SPANNING set WITH CHORDS: the 3 nearest in step, log-spaced older members, and the current reference's support. HodgeRank needs triangles (§0a.1), so every probe set closes at least one triangle per new edge | archive scale only | DESIGNED; built only when the archive matrix is wanted |
| **T0 reuse (power prior)** | older rows of a matchup (cycle rows, SPRT rows) as a₀-weighted counts in ESTIMATES (never in decisions, §0c rule 1) | estimates | **a₀ = 0 until validated** (§2.6 part B); a₀ per source class from the commensurate model (Hobbs et al. 2011) on the first dense audits |
| **audit** | a periodic RE-PLAY of a random 2 % of banked dense batches (exact count equality, §9.2), plus a full dense re-fill of the archive matrix when the sparse path is used | both | DESIGNED (U4) |

### 2.3 The reference mixture: how it is computed, and its uncertainty
- **The game.** The symmetric zero-sum meta-game with payoff A_ij = P_ij − ½ on the frozen set (the archive at check
  time, or the pool). P_ij is the score (a draw counts ½), so A is antisymmetric by construction.
- **The solution.** The max-entropy Nash of A (Balduzzi et al. 2018): the LP for the Nash set {p ∈ Δ : (pᵀA)_j ≥ 0 ∀j},
  then maximum entropy over it (a convex program). The solve is deterministic (a fixed solver and tolerance, recorded in
  the reference).
- **Its uncertainty is LARGE on our data.** N0's 20-node archive at 100 games per edge:
  - point Nash: pure on one node (the transitive case, Balduzzi P3);
  - posterior Nash: P(w > 0.01) is 0.26–0.88 across the six strongest nodes;
  - the newest node's score vs the mixture has posterior SD **3.6 pp** from the mixture alone (MEASURED, F-ED-3).

  That is more than the 2 pp a plateau check must resolve.
- **The design consequence (G3 resolved at the system level):** a strength read is against a FROZEN reference, not
  against "the Nash".
  - The reference is the **posterior-mean max-entropy Nash**: M = 1,000 posterior draws of the matrix (Beta(w + ½,
    l + ½) per edge, half-points to each side for draws), with the seed = the digest of the consumed row ids. The
    weights are averaged, and members with mean weight < 0.005 are dropped and the rest renormalised.
  - It is written ONCE to `references/` (§0b.2) and never changed. Scores of different snapshots against the SAME
    reference are differences against one FIXED opponent distribution, so the mixture's estimation error does not
    enter them. It enters only the question "is this reference the true Nash?", which no decision asks.
  - Choosing the posterior mean over the point Nash spreads weight across the plausible frontier. That is what a robust
    yardstick wants, and it is copy-invariant in the limit.
  - Alternatives (owner choice, §11.2 Q5): the point max-entropy Nash (unstable, pure on the frontier); a uniform
    archive (Borda, Heckel et al.; stable but duplicate-sensitive); a logit QRE (McKelvey & Palfrey 1995).
- **Playing against a reference:** the pairs are allocated over its members in proportion to the weights (Neyman
  allocation reduces to this when per-member pair SDs are equal, ≈ 0.33 here). The score's SE is the pair-clustered
  SE of the weighted mean.
- **When a reference is re-solved:** when a new member's posterior P(w > 0.01) ≥ 0.5 against the current one (the
  frontier moved). The new reference is BRIDGED: the last 3 snapshots are scored against both, so a slope read can be
  chained across references (§8).

### 2.4 What the matrix can and cannot decide about eviction
- **In a near-transitive pool, almost every member has zero Nash weight.** That is the correct answer: the frontier
  dominates. So "evict only when confidently zero weight" is a NECESSARY condition that binds rarely. It does not choose
  WHICH member leaves when the cap binds. **The skeleton's design left that open.**
- **Proposed completion** (a DECISION, so it belongs to league §B; flagged §11.2 Q6). When the cap binds:
  - among the members with posterior P(w > 0.01) < 0.05 (confidently zero weight), evict the one the CURRENT trainee
    beats most confidently, i.e. the one with the lowest PFSP weight. It carries the least training signal (the
    AlphaStar logic, §0a.5);
  - members in the undecided band [0.05, 0.5] get T2 games first;
  - if no member is confidently zero-weight, GROW the pool (owner, 2026-10-02) up to a hard ceiling set by GPU-active-
    set memory;
  - a member inside the rounding band of 0.05 (±0.01, §9.1) is treated as NOT confidently zero (kept).
- **The cycling meter.** HodgeRank on the logit edge flows, weighted by inverse delta-method variance:
  - report the curl and harmonic shares of the weighted residual, and the overdispersion φ = Pearson χ²/df against
    its noise floor;
  - **the noise floor is EMPIRICAL, not binomial:** within-run ladders sit at φ ≈ 0.76 (median of 70), below binomial
    (F-ED-2). So the floor is the φ distribution of matched-size within-run pools (`measurements/eval_design_2026-10-03/out/cyc.txt`),
    and a pool is called CYCLIC iff its φ exceeds that distribution's 95th percentile at matched df;
  - **today's banked data: no within-run pool is cyclic** (F-ED-1). The first multi-lineage pool (era step 1) is the
    first real test.

### 2.5 Archive scale (later)
The cross-run archive (every frozen snapshot of every run at one protocol and architecture family) is where T1 + T0 +
matrix completion pay off. **Not built until a question needs it.** The ledger already makes it possible without a
new training run, which was the owner's durability requirement.

### 2.6 The pre-registered validation (principle 4): deterministic check rules

**Part A — targeted (T2) vs dense on the active pool.** It is run on each of the first THREE era-step-1 pools that reach
K ≥ 15. For each pool, a full dense matrix at 400 pairs per edge is the ground truth (≈ K(K − 1)/2 × 800 games:
84k games at K = 15, CPU lane overnight). Against it, the incremental-dense + T2 pipeline must satisfy, on every pool:
1. **No false eviction.** Every member the pipeline would evict has dense posterior P(w > 0.01) < 0.10. A member whose
   dense value lies in [0.09, 0.11] is EXCLUDED from the check (rounding band).
2. **No missed support.** Every member with dense P(w > 0.01) ≥ 0.5 is undecided-or-kept by the pipeline.
3. **Reference agreement.** The newest member's score vs the dense posterior-mean reference and vs the pipeline's
   reference agree within **1.5 pp**, judged on the difference's own paired 95 % interval (the bar must contain the
   whole interval; an interval that straddles the bar is NOT AGREED).
4. **Economy.** The pipeline used ≤ 60 % of the dense audit's games.

PASS = all four, on all three pools. FAIL on any one means the pipeline is NOT trusted: play dense at 400 pairs per
edge (affordable to ~40 members) and report.

**Part B — T0 reuse.** It uses the same three dense audits. a₀ per source class is estimated from them (commensurate
model), then frozen. T0 is switched ON for that class only if refitting the matrix with T0 added (and T2 games removed
in equal number) still passes 1–3. Otherwise a₀ = 0 permanently, recorded.

## 3. Statistics in use

| method | used for | reference (§0a) | status |
|---|---|---|---|
| GSPRT on the pentanomial, λ bisection, a minimum pair count, Wald bounds | promotion; a candidate for the plateau test | Wald 1945; Van den Bergh (Fishtest) | BUILT (`sprt.py`) |
| Group-sequential O'Brien–Fleming / alpha-spending | pre-registered A/Bs (X5 §7.4) | O'Brien & Fleming 1979; Lan & DeMets 1983; Jennison & Turnbull 2000 | in use (X5) |
| Confidence sequences (stitched / betting) | the open-ended plateau monitor (G1 candidate); live dashboard reads | Howard et al. 2021; Waudby-Smith & Ramdas 2024; Johari et al. 2022 | RECOMMENDED to the plateau unit |
| CUSUM | plateau onset (refinement) | Page 1954 | optional |
| Bradley–Terry | the rating spine | Bradley & Terry 1952; Elo 1978 | BUILT (`main.elo`, ladder) |
| HodgeRank (gradient / curl / harmonic) + overdispersion vs an empirical floor | the cycling meter | Jiang et al. 2011 | BUILT for trainee × bots; pool × pool U7 |
| Max-entropy Nash, posterior mean, frozen as a reference | strength reads; eviction's necessary condition | Balduzzi et al. 2018 | DESIGNED (U7) |
| ResponseGraphUCB-style targeted sampling (Clopper–Pearson) | T2 top-ups | Rowland et al. 2019 | DESIGNED (U7) |
| Active ranking (cost ∝ Σ 1/Δ²) | the principle behind T2 | Heckel et al. 2019 | principle |
| Power prior, a₀ by the commensurate model | T0 reuse | Ibrahim & Chen 2000; Duan et al. 2006; Hobbs et al. 2011 | DESIGNED, OFF until §2.6 B |
| Random-effects across runs (σ_h with meter variance subtracted) | run-level claims | (meta-analysis; P0 `main.h2h.runfloor`) | BUILT |
| α-Rank; low-rank completion; AIVAT | reserve | Omidshafiei 2019; Du 2021; Burch 2018 | not adopted (§0a.7) |

## 4. T19 — background eval, and how it coexists with a training run (the GPU owner)

### 4.1 The facts it starts from
- The blocking in-process Rust eval cycle costs **1.57 % of wall** at N = 256 E10:
  - the update cycles straddling an eval run +12–20 s against a 51.6 s median cycle, per 2M steps;
  - it is 2.9 % at E5;
  - source: `m5_sizing/PROGRESS.md` O9.
- **SETTLED 2026-10-02** (`program_rust_core.md` Decision record): the blocking design stands while its share is
  ≤ 10 % (`m5_sizing/REGISTRATION.md` §7). Above that, the background-FILLER build is queued.
- **The 2026-09-30 record refused a SECOND GPU service** (an eval subprocess with its own T2), which is consistent with
  the one-owner rule.
- T2 is single-caller ("not thread-safe: one caller loop drives it"). Its `ROLLOUT > EVAL > FILLER` priority classes
  are BUILT but UNUSED: eval calls `drain()` while blocking. The eval slots are declared at startup (1 trainee + 1 per
  sentinel + fixed opponents; 6 today) and hot-swap weights by `slots.copy_in`.
- At N = 256 a rollout is ~9.5 s of collection, with the 8-thread env core at 7.0–7.5 CPUs, followed by ~41 s of update.
  The two alternate strictly.
- **"The CPU is ~15/16 idle during the update" has NO measurement behind it** (F-ED-6).
- No CPU inference throughput exists at batch 64–256 on a quiet box. The one CPU figure is P0's 4.77 games/s (T2
  eager, 64 envs, 8 threads, load average 6–10).
- The engine start of a fresh T2 is 114–167 s, almost all compile + capture (F-P0-4). An AOT package loads in
  0.01–0.09 s but serves ~50 % slower.

### 4.2 The design: three lanes, one owner

| lane | runs where | GPU? | serves | budget |
|---|---|---|---|---|
| **GPU window** | INSIDE the trainer process, in today's blocking eval window at the 2M-step rollout boundary, on the startup-declared eval slots (`copy_in`; no new slot, bucket or capture after startup, so K6 holds) | yes: the trainer IS the owner | P0 cycle, P1 SPRT always; P2–P4 only when the CPU lane is short | a wall-share CAP c_gpu (proposed **5 %**, under the registered 10 %, §11.2 Q3), enforced by the scheduler from measured window seconds ÷ elapsed wall (read on the update cadence, never `eval/duration_sec`) |
| **CPU lane** | a separate, CUDA-FREE worker process (`CUDA_VISIBLE_DEVICES=""`, and it asserts CUDA is uninitialized), launched beside the trainer by the launcher, at `SCHED_IDLE` / nice 19, with its own declared lifecycle (a CPU T2 service on the eager backend, declared slots, frozen) | **never** | P2–P5 | its measured perturbation of training: must pass §4.3 |
| **Offline GPU** | a one-shot engine (the multi-cell h2h engine, U6) run by the agent that HOLDS the GPU lease (`scripts/ops/gpu_lease.sh`, granted by the orchestrator), i.e. only when no training run holds it | yes, as the sole owner for its lifetime | foreign-architecture work (an X5 A/B cell), large dense audits, back-fills | the grant |

- **Why not a GPU eval process beside the trainer:**
  - it would be a second GPU owner (owner rule; the 2026-09-30 record);
  - it would lower K6's device-free memory ceiling and could trip `[CudaMemTrend]` (F-ED-10);
  - it would pay a 2–3 min engine start.
- **Why not filler on the GPU now:**
  - the precision budget fits the blocking window plus the CPU lane at ~3–4 % of wall (§5);
  - filler needs the GPU-phase utilization that nobody has measured;
  - the 2026-10-02 rule already says when it is needed: share > 10 %.

  It stays the declared next step (§4.5).
- **Matrix edges in the GPU window:** an edge is two frozen members, so it occupies two eval slots (copied in from the
  CPU-resident snapshots). Six slots play up to three edges at once, or one SPRT candidate (copied into the trainee eval
  slot) against five members.
- **Regime equality across lanes:** CPU eager and GPU graph play the SAME games (F-P0-7: identical W/L/D, pentanomial
  and team map on a 100-pair probe). So the lane is `compute`, not regime, and rows from both lanes pool.

### 4.3 The throughput-perturbation plan (pre-registered; run by the training agent, the GPU owner)
1. **Phase-utilization descriptor (first):** sample `nvidia-smi dmon` (SM %, memory) at 1 s and `utils.cpu_meter`
   (busy CPUs, run queue) per phase across 10 updates of a production-shape run. This settles F-ED-6 and tells U5 how
   much idle CPU there is.
2. **CPU-lane A/B:** on the same run, alternate the CPU worker ON / OFF in blocks of 5 updates, 8 blocks each (80
   updates, ~70 min), with no eval window inside a measured block (C-1's reader rule).
   - The metric is the median update-cycle wall per block. Collection fps and `train_ms` are reported separately.
   - **ADOPT the CPU lane at its tested setting iff the one-sided 95 % upper bound of the ON/OFF ratio of block medians
     (t on the 7 df of the paired log-ratios) is ≤ 1.010** (a 1 % throughput budget).
   - A bound within 0.0005 of 1.010 is NOT ADOPTED at that setting (rule 8).
   - Report the lane's achieved games/s.
3. **If it fails,** re-test with the worker gated to the UPDATE phase only (the trainer writes a phase flag file; the
   worker pauses during collection, where the env core saturates the CPUs) and at half the threads. The same rule
   applies.
4. **GPU-window share:** reported per 10M steps from the cadence (O9's method). The scheduler's cap holds it at
   ≤ c_gpu by construction.

### 4.4 Restarts and aborts (A2's safe points)
- **GPU window:** A2 already abandons a partial eval cycle at a safe point (`deferred_abort.py`; never collected, never
  recorded). Under the scheduler, a partial BATCH is never written. Requests are durable events, so on resume the
  scheduler re-reads the open requests and their rows and continues.
- **SPRT: RESUME, not abandon.** Today `abandon_unfinished` drops an in-flight test at a restart (F-ED-9). With
  per-batch rows, the test's state is the SUM of its batch rows, and the stopping rule is checked only at batch
  boundaries. An interruption is outcome-independent (a restart's timing never depends on the games), so resuming
  preserves the test's error rates. The frozen pool lives in the request. The "no re-tests" rule is unchanged: one
  decision row per candidate.
- **CPU worker:** on SIGTERM it drops its partial batch and exits. The launcher restarts it with the trainer. It holds
  no state beyond the ledger and the requests. A request whose snapshot file is gone (evicted and deleted) is CANCELLED
  with a reason event.
- **Seeds:** every batch's seeds derive from (`request.id`, batch index) through `gen3_eval_game_seed_v1`. A resumed
  request never replays a batch it banked and never reuses a seed.

### 4.5 What comes after (not in this build)
- **GPU filler** (T2's built priority classes; eval rows ride on rollout flushes, at most one batch per flush): build it
  only if (a) the GPU-window share at the precision budget measures > 10 % (the registered rule), or (b) the CPU lane
  fails §4.3 AND the window share exceeds c_gpu.
- It is designed TOGETHER with X14 (rollout/update overlap), which competes for the same idle phases. Both wait for
  §4.3 step 1's phase measurement.

## 5. The eval budget, derived from PRECISION

**Inputs (with provenance):**
- **Pair SD** σ_pair = **0.33** in score units on a near-½ cross edge (P0: 0.328–0.336 over nine edges). Unmirrored
  per-game SD is 0.5.
- **Elo scale:** d(score)/d(Elo) at ½ = ln 10 / 400 × ¼ = **0.144 pp per Elo**.
- **Wall time:** 2M steps ≈ 20 updates × 51.6 s ≈ **17.2 min** of wall at N = 256 E10; 10M ≈ 86 min (O9).
- **GPU-window cost:** ≈ **16 ms of wall per game** (the O9 straddle, +12–20 s per ~1,000-game cycle).
- **CPU lane:** ≈ 4.8 games/s (P0, contended). The capacity DURING TRAINING is UNKNOWN; §4.3 measures it.

**Per consumer:**

| consumer | precision it needs | games | cadence | lane | per 10M steps |
|---|---|---|---|---|---|
| **P0 cycle** (monitor + supply guards) | a monitoring curve: ±5 pp per opponent is enough, because no DECISION reads it once SPRT is on | 100 per opponent × (9 bots + 5 sentinels) ≈ 1,400 | every 2M | GPU | 7,000 games ≈ 112 s ≈ **2.2 %** |
| **P1 SPRT** | α = β = 0.05 at H0 0.50 / H1 0.55 (decided) | E[pairs] ≈ 290–350 at the hypotheses, 530–610 in the indifference zone, cap 1,680 (`measurements/sprt_promotion/`, conservative τ = 0 rows, since P0's antithetic correlation is only −0.12) | per candidate (≤ 1 per 2M) | GPU | ≈ 4,000 mean ≈ 64 s ≈ **1.2 %**; worst 16,800 ≈ 5.2 % |
| **P2 plateau check** | 2 Elo / GPU-h × W = 7 GPU-h = 14 Elo = **2.0 pp**; at α = β = 0.05 (z sum 3.29) SE ≤ **0.61 pp** | fixed-n equivalent (0.33 / 0.0061)² ≈ **2,930 pairs = 5,860 games** for newest vs the frozen reference (the W-back snapshot's score vs the same reference is REUSED); the same again for the outside panel. A sequential design lowers this; the plateau unit sets it | once per 10M | CPU (else GPU) | ≤ 11,700 games |
| **P3 matrix** (incremental dense + T2) | eviction needs only to separate the clearly-weak from the frontier; at 100 GAMES per edge every N0 node at or below 42M but one (34M, 0.20) already reads P(w > 0.01) ≤ 0.04 (F-ED-3) | (K − 1) × 200 per promotion: 3,800 at K = 20; T2 ≤ +30 % | per promotion (1–5 per 10M) | CPU (else GPU) | 3,800–24,700 games |
| **P4 audits** | §9.2's 2 % replay; §2.6's dense validation (one-off, 84k per pool × 3 pools) | 2 % of rows | continuous / one-off | CPU, offline GPU | ~1,000 |
| **anchors** | per `EXTERNAL_ANCHORS_SOP.md` (tier B: SmallRL greedy, both team sets, 100 games each + one t1 cell) | ~300 | per milestone | its own websocket stack (CPU, offline) | — |
| **exploiter gap** | the run floor between exploiters (σ_h ~ 0.3–2 pp, P0) dominates once meter SE ≤ ~1 pp | 1,000 pairs per exploiter × 5–7 | per round | CPU / offline GPU | 10–14k per round |
| **pre-registered A/B (X5)** | §7.4: 1,000 pairs per cell | 9 / 25 / 64 cells × 2,000 games | per look | offline GPU (foreign architecture) | one-off, ≤ 128k |

**The fit (DERIVED; the CPU lane's real capacity is UNKNOWN until §4.3):**
- **GPU window:** P0 + P1 ≈ 11,000 games per 10M ≈ **3.4 %** of wall (worst case with capped SPRTs ≈ 7.4 %). That is
  under the 5 % cap on average and under the registered 10 % even at worst.
- **CPU lane:** P2 + P3 + P4 ≈ 16,500 (one promotion) to 37,400 (five promotions) games per 86 min = **3.2–7.3
  games/s**. That is at or above P0's contended 4.8 g/s, so the matrix fill is the elastic item.
- **If the CPU lane is short:**
  1. first halve n₀ to 50 pairs per edge (the ladder's 100 games);
  2. then move the plateau check into the GPU window (+~3.6 %). The total stays under 10 %.
- **If EVERYTHING ran in the blocking window,** the precision budget would cost ≈ 10 % (two promotions per 10M) to
  ≈ 15 % (five) of wall. That is the case for the CPU lane (F-ED-5).

## 6. Week-one assumptions, RE-GROUNDED (owner, 2026-10-02: "encourage the agent to re-evaluate key assumptions")

| parameter | today | verdict | evidence |
|---|---|---|---|
| games per opponent per eval cycle | 100 | **KEEP for the monitoring cycle; DECISIONS stop reading it** | Once SPRT is on, promotion plays its own pairs; plateau and matrix have their own requests (§5). ±5 pp is adequate for a monitor and the supply guards. It was a week-one CPU figure; it is now cheap (2.2 % of wall) |
| eval cadence | every 2M steps | **KEEP for the cycle; per question for the rest** | plateau once per 10M (league §C); matrix per promotion; audits continuous (§0d) |
| pool size | 20 (`DEFAULT_MAX_SNAPSHOTS`) | **KEEP for a single lineage; MEASURE before changing for era step 1** | 70 within-run ladders show no cyclic component above noise (F-ED-1); N0's posterior Nash support is ~1–8 nodes. Spinning tops: the pool must cover the Nash cluster, which is small here. The multi-lineage pool is unmeasured |
| eviction | oldest-first (or spread retention) | **CHANGE (T20), with a completion** | zero Nash weight is necessary but near-vacuous in a transitive pool; tie-break by PFSP weight; grow otherwise (§2.4, §11.2 Q6) |
| promotion | first threshold crossing (0.55 / 0.65) | **CHANGE → SPRT** (BUILT, flips at X26) | the crossing step is a coin flip inside the replicate floor (ledger 2026-09-10, `L17330`) |
| bots in the opponent mix (`heuristic_fraction`) | a few % | **OUT OF SCOPE here** (a training-recipe knob: `design_learner_recipe.md`) | eval's only input: the bot edges are saturated (random 0.99, heuristic ~0.8 on a 2026-10-02 row), so their information for a RATING is near zero (the Dota 2 15–85 % argument, §0a.5) |
| the eval team set | the trainee's builder (default pool, 10 % sample bias) | **KEEP; RECORD it as `regime.team_set`** | the ladder campaign (era step 5) will need team-set-specific reads; a pool-wide h2h dilutes team-local gains to invisibility by arithmetic (ledger 2026-08-28, `L05731`). A new team set is a new regime, never pooled |
| the trace quota preferring losses | — | **KEEP for the prober; ENFORCE that estimates never read traces** | §0c rule 3, enforced by the reader gate (§0b.4) |
| per-pair games on the snapshot ladder | 100 (200 vs promotion sentinels) | **KEEP the count; MIRROR it (100 pairs) under U7** | dense 19 edges × 100 games gives a node SE of ~8 Elo by formula (173.7 × √(1/(19 × 100 × ¼))); the ladder's per-edge noise is below binomial (F-ED-2), so that is conservative. N0's committed `ladder.json` reports se 16.9 per node (sparse: 141 of 190 pairs, plus bot anchoring), not ±10 (F-ED-13) |

## 7. TODO (fill in as implemented)
- [ ] U1 the ledger v2 core: schema, upgrade-on-read, archive location, decisions / requests / references, the reader
      API + gate, `audit` (§0b, §10).
- [ ] U2 in-loop migration: cycle rows, SPRT rows + decisions + resume, ladder dual-write, h2h archive default.
- [ ] U3 backfill; U3b anchors / untaught / gap / hodge onto the ledger.
- [ ] U4 the scheduler + GPU window; U5 the CPU lane + §4.3's measurement; U6 the multi-cell offline engine.
- [ ] U7 T20 estimators (BT / Hodge pool × pool, posterior-mean Nash references, dense-incremental + T2); U8 §2.6's
      validation.
- [ ] The plateau test (the next unit; §8 lists what it gets).
- [ ] The discrimination meter (era step 3).
- [ ] F-ED-2 (the ladder's sub-binomial noise): find the cause.
- [x] Literature review (this doc §0a, 2026-10-03).
- [x] §5 budget derivation (2026-10-03, DERIVED; the CPU-lane capacity is measured by §4.3).
- [x] §6 re-grounding (2026-10-03).
- [x] The blocking question (§4, settled 2026-10-02; T19 builds on it).

## 8. Interfaces the PLATEAU test will need (the next unit; league §C's gaps G1–G6)

| gap | what the ledger / scheduler supplies |
|---|---|
| **G1** repeated monitoring | per-batch rows tagged with the check's `request.id`, so the information fraction at every look is a count. A FIXED check schedule (every 10M steps, deterministic, written as requests at run start) is what CUSUM needs. `as_of` reads. Any of alpha-spending, a confidence sequence (§0a.4, recommended first) or CUSUM is implementable without a ledger change |
| **G2** a current slope, not a window average | `score(snapshot, ref_id)` with its SE, for EVERY promoted snapshot against the SAME frozen reference (a `plateau_check` request scores each new promotion against the active reference; ≈ 2,930 pairs at the fixed-n bound, fewer with a sequential rule), giving a series to fit a curve to (Domhan et al. 2015) |
| **G3** the archive's mixture moves | **resolved at the system level:** strength is read against an immutable reference (`references/`, §2.3); a re-solve is bridged by scoring the last 3 snapshots against both references |
| **G4** the Elo conversion | scores are model-free (pp). The linear conversion (0.144 pp per Elo at ½) is within ~4 % of the logistic slope for scores in [0.4, 0.6] (p(1 − p) = 0.24 vs 0.25); outside, the estimator converts by the logit and states the range |
| **G5** the noise floor AT the plateau | `plateau_check` requests of kind `floor`: the paired control continuations scored against the same reference; σ_h by `main.h2h.runfloor` (meter variance subtracted) |
| **G6** combined error rates | the inputs to simulate: σ_pair 0.33; the run floor σ_h 0.33–2.12 pp (P0, 1–3 df; UNVERIFIED at 15M); the check cadence; the reference's bridging error; and the gap meter's per-round SE |
| **N0 back-test** | N0's 20-node ladder (141 of 190 pairs × 100 games) backfilled as `ladder` rows, read `as_of` each check point. **A real back-test needs more:** the 49 missing pairs, scores of snapshots across 0–75M against one frozen reference, and per-edge n raised toward the fixed-n bound. That is ≈ 10k + 60k games on the CPU lane or offline GPU. **RISK:** N0 is architecture v121; loading its snapshots on HEAD's eval core needs `historical_load_kwargs` (`2409d899` covers archived TRAINEES); N0's snapshots specifically are **UNVERIFIED** |

**Also supplied:** the Hodge cycling read and its empirical floor (§2.4) for the TREADMILL diagnosis, and
`main.policy_drift` for drift (BUILT). The gap series comes from `best_response_gap` over `cycle` / `gap` rows.

## 9. Determinism, audits and the seat RNG

### 9.1 Every decision rule is deterministic
- A decision is a pure function of (its request's rows, declared constants, the rule's version). It writes a decision
  row naming all three.
- **A Monte Carlo inside a rule** (the posterior Nash, §2.3) uses M = 1,000 draws with the seed = the digest of the
  sorted consumed `row_id`s and the rule version. The same rows always give the same verdict.
- **Rounding bands** (standing rule 8):
  - an LLR or t within 1e-9 of a bound is not a crossing;
  - a posterior probability within ±0.01 of its threshold (Monte Carlo SE at M = 1,000 is ≤ 0.016, so this is about
    one SE) is UNDECIDED, which means kept or continue, never act;
  - §2.6 excludes members in [0.09, 0.11];
  - §4.3 excludes a bound within 0.0005 of 1.010.
- **INCONCLUSIVE, never a verdict, when:** aborted games exceed 25 % of the attempted games of a request (`counts.aborted`,
  the standing timeout rule), or a request's rows span two regimes (the reader refuses).

### 9.2 Audits
- **Replay audit (P4):** a random 2 % of banked batches, chosen by a seeded hash of `row_id`, are re-played from their
  seed block. The count must be EQUAL, excluding games that contain a near-tie decision (the `compute` near-tie census;
  0.18 % of P0's games). Any other difference is a FINDING and stops the scheduler's decisions until explained. F-P0-7
  shows exact replay holds across CPU and GPU.
- **`audit` CLI** at scheduler startup (§0b.4).
- **Decision audit:** every decision row is re-derivable from its rows by `python -m main.eval_ledger verify <decision_id>`.

### 9.3 The seat-dependent RNG (F-P0-5) and what it means for mirrored pairs
- **What was measured:**
  - a self-play pair is not exactly ½: 5.2 % of self-play pairs are WW or LL, and 779 of 14,963 pairs CLEAR of any near
    tie are off centre;
  - the mechanism (a speed tie broken by the RNG in a seat-dependent order) was diagnosed on perturbed fresh
    checkpoints only;
  - the seat effect is u = −0.006 ± 0.083 pp.
- **What it means:**
  1. **A mirrored pair is duplicate play, not an exact antithetic copy.** The pentanomial GSPRT treats the pair as one
     draw from a five-category distribution and estimates its variance from the data. It never assumes the two games
     are mirror images, so its validity is UNAFFECTED.
  2. **The variance gain is modest.** It is ~12 % of games on a near-½ cross edge (variance ratio 0.87–0.91, F-ED-4)
     and ~90 % on a self-play edge. Mirroring is kept for BALANCE (both sides pilot both teams exactly), not for
     variance. Its value is in bias control at small n and in team-attributed reads.
  3. **The seat effect is zero within ±0.17 pp (95 %) today,** but that is a measurement on one engine version.
     **Design it out:** `seat_rule = balanced` (half the pairs with the player at p1, half at p2) for every new row, so
     a seat effect cancels by construction. It costs nothing: the bot round robin already plays
     `seat_alternating_by_pair` on the Rust eval core (`3ccef556`), so it lands in U2.
  4. **An identical-players check** (a self-play edge per architecture per protocol) stays in the audit class. Its
     expected score is exactly ½ under `balanced`. A 95 % interval excluding ½ ± 0.25 pp is a FINDING.

## 10. Build plan

**Order:** the ledger + in-loop migration FIRST, landing before X26 continues from the X5 A/B winner (~Wed/Thu
2026-10-07/08); then the scheduler and T19; then T20. Sizes are agent-days.

| unit | what | size | tier | depends on | Sonnet-safe? |
|---|---|---|---|---|---|
| **U1** | ledger v2 core: schema v2 + validator, v1 upgrade-on-read, archive location under `run_archive_dir()`, the decisions / requests / references streams + schemas, shard close / `close-stale`, the `ReaderDecl` API + static gate + closed-list doc gate, `python -m main.eval_ledger audit / verify / show` | 2.0 | opus-high | — | no (the core schema; GIGO risk) |
| **U2** | in-loop migration: the cycle writes rows (exact W/L/D, team counters from the eval core) beside `eval_results.jsonl`; SPRT rows per batch + decision rows + RESUME; ladder dual-write; `main.h2h` to the archive with `--purpose ab`; `seat_rule = balanced` (already played by the bot round robin). Gate: the routine gate + the `--debug --debug-eval` smoke + the first two minutes of a real launch | 2.0 | opus-high | U1 | no (the training loop) |
| **U6** | the multi-cell offline engine: one T2 held across cells (`svc.load` per cell), the `startup_seconds` breakdown printed, a check of whether AOT packages reuse across processes (F-P0-4) | 1.0 | opus-medium | — (parallel to U2) | no (GPU engine) |
| **U3** | backfill per §0b.6's table | 1.0 | sonnet-xhigh | U1 | **yes**, with §0b.6 as the written mapping |
| **U3b** | `main.anchors`, `main.untaught_meter`, `best_response_gap` and `hodge` onto the ledger (writers and readers with declarations) | 1.0 | sonnet-high | U1, U2 | **yes** |
| **U4** | the scheduler: request store + events, priorities, deterministic allocation, the anti-starvation share, budgets and c_gpu, the GPU window generalizing today's cycle (matrix edges on eval slots), the replay audit | 2.5 | opus-high | U2 | no |
| **U5** | the CPU-lane worker (CUDA-free assert, SCHED_IDLE, CPU T2 eager, declared lifecycle) + launcher integration + §4.3's phase descriptor and A/B, run by the training agent | 2.0 + ~2 h GPU | opus-high | U4 | no |
| **U7** | T20 estimators: BT / Hodge pool × pool from the ledger with the empirical noise floor, posterior-mean max-entropy Nash + references, dense-incremental + T2 request generators, the ladder fit reading the ledger, the eviction-support interface for league §B | 2.5 | opus-high | U4 | no |
| **U8** | §2.6's validation on the first three era-step-1 pools + the N0 back-test's data supply (§8) | 0.5 + games | opus-medium | U7 | no (a pre-registered read) |

- **Total ≈ 14.5 agent-days.** U1 → U2 is the critical path to X26 (~4 agent-days; U6 and U3 run beside it).
- **T19** (U4 + U5) and **T20** (U7 + U8) must be ready before era step 1 (the population loop), which forks from
  the X26 baseline's plateau.
- **Each unit updates this doc,** its leaf `CLAUDE.md`, `eval_and_rating.md` and the Decision record in the same
  commit, and ships through `/gen3ai-ship` with green gates.

## 11. Open gaps, owner questions, findings

### 11.1 Open gaps (honest list)
1. **The CPU lane's capacity during training is UNKNOWN,** and so is whether it perturbs training by ≤ 1 %. §4.3
   measures both; §5's fit is DERIVED until then.
2. **GPU and CPU utilization per training phase is UNMEASURED** (F-ED-6). The filler-vs-X14 choice waits on it.
3. **The multi-lineage pool's cyclic width is unknown.** Every banked pool is one lineage. Era step 1 is the first test,
   and the pool cap stays at 20 until the meter reads it.
4. **The ladder's sub-binomial per-edge noise has no explanation** (F-ED-2). It makes binomial SEs conservative, which is
   safe, but an unexplained pattern in our core rating data is a GIGO smell.
5. **The posterior-mean Nash reference's stability** across re-solves is untested; the bridging rule (§2.3) is a design,
   not a measurement.
6. **AIVAT-style control variates** could cut eval games several-fold if our value were calibrated. That is deferred to
   era step 3's discrimination work.
7. **N0's snapshots on HEAD's eval core** (the back-test) are UNVERIFIED.
8. **Literature items not verified from a primary source:** AlphaStar's 35 / 50 / 15 split and f_hard; Lan & DeMets'
   DOI and pages; the Kahle β₁ window; Agarwal et al.'s method details (bibliography verified); a primary citation for
   duplicate poker; the Ibrahim & Chen author order on one index page.

### 11.2 Questions for the owner (real choices, with a recommendation)
- **Q1, purposes.** Add `ab`, `ladder`, `untaught` and `gap` to the closed list? **Recommend yes.** Without them a
  pre-registered A/B row is indistinguishable from an audit (F-P0-6).
- **Q2, T19's shape:**
  - (a) the blocking GPU window inside the trainer plus a CUDA-free CPU lane (**recommended**: it respects one GPU
    owner, fits the budget, and needs no new GPU machinery);
  - (b) GPU filler now;
  - (c) blocking only, with everything in the window (≈ 10–15 % of wall at the precision budget).
- **Q3, the GPU-window cap c_gpu:** **5 % (recommended)** vs the registered 10 % ceiling.
- **Q4, T20's scope:** **incremental dense for the active pool + T2 top-ups, with sparse tiers only at archive scale
  (recommended)**, vs building all four tiers for the pool now.
- **Q5, the strength reference:**
  - **the posterior-mean max-entropy Nash, frozen per check (recommended)**;
  - the point max-entropy Nash (pure on the frontier and unstable on our data);
  - a uniform archive (Borda: stable, but over-weights eras with many snapshots).
- **Q6, eviction when the cap binds** (league §B's decision, flagged here): among confidently zero-weight members, evict
  the one the trainee beats most (lowest PFSP weight); if none qualify, grow. **Recommend yes.**
- **Q7, SPRT across a restart:** **RESUME (recommended:** outcome-independent interruption, no error-rate cost) vs
  today's abandon.

### 11.3 Findings (standing rule 7; evidence in `measurements/eval_design_2026-10-03/` unless noted)
- **F-ED-1 (MEASURED): within-run snapshot pools are transitive within noise.** On 70 ladders, the BT residual Pearson
  χ²/df has median 0.76 (q10–q90 0.54–1.14). Simulated binomial data on the same graphs give 1.01–1.03.
- **F-ED-2 (MEASURED, cause UNKNOWN): the ladder's per-edge noise is below binomial.** χ²/df < 1 on most ladders; 133
  twice-played pairs give var(z) = 0.82. The ladder plays unseeded, unmirrored, with iid team draws.
- **F-ED-3 (MEASURED): max-entropy Nash on a near-transitive archive is pure on the frontier and unstable under noise.**
  N0's point Nash is one node. Its posterior spreads over the top ~8, and the newest node's score vs the mixture has
  posterior SD 3.6 pp at 100 games per edge. Hence frozen references (§2.3) and the eviction completion (§2.4).
- **F-ED-4 (MEASURED): mirroring saves ~12 % of games on a near-½ cross edge** (variance ratio 0.87–0.91; self-play
  0.09–0.11). The antithetic correlation is about −0.12, so outcome variance is mostly NOT team luck.
- **F-ED-5 (DERIVED): at the precision budget, an all-blocking eval would cost ≈ 10–15 % of wall.** The CPU lane keeps
  the GPU window at ≈ 3.4 %.
- **F-ED-6: "the CPU is ~15/16 idle during the update" has no measurement behind it** (`TASK_BACKLOG.md` T19,
  this doc's old §4).
- **F-ED-7: `sizing_C`'s `ladder.json` relative reference silently fell back to the first snapshot** ("baseline
  'untaught_meter_opponent_v14' unreadable (ImportError)"). A silent degradation of `ratings_relative`.
- **F-ED-8: in-loop eval rows carry no draw count;** draws survive only in 370 of 1,081 `eval_manifest.json` files.
  The backfill flags the rest `draws_folded`. Going forward, U2 records exact W/L/D.
- **F-ED-9: `sprt_promotion.abandon_unfinished` discards an in-flight test at every restart,** which wastes its games
  and leaves a candidate untested. It is resumable once rows are per batch (§4.4).
- **F-ED-10: a co-resident GPU eval process would lower K6's device-free memory ceiling** and could trip
  `[CudaMemTrend]`. This is a second reason for the one-owner design.
- **F-ED-11 (UNVERIFIED): how the learner-lifecycle static gate classifies `rust_eval.launch.load_sentinels`,** which
  builds a CPU `nn.Module` per cycle inside a declared STEP module.
- **F-ED-12: no pentanomial data exists in any RUN's eval history** (no run enabled `--eval-mirrored-pairs`; no run
  has a `sprt_promotion.jsonl`). The mirrored regime starts fresh at X26.
- **F-ED-13: root `CLAUDE.md` says `ladder.json` is "±10"; N0's committed file reports se 16.9 per node** (sparse 141
  of 190 pairs, plus bot anchoring). The claim is not what the file reports.

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-02 | Mirrored pairs **(owner)** | symmetric evals mirror; pinned-team meters refuse | mirroring everything | piloting vs response (`04aa4a45`) |
| 2026-10-02 | Promotion **(owner)** | GSPRT on mirrored pairs (Fishtest's method + a minimum pair count), Wald bounds | overshoot-corrected bounds (saved ~21 % of pairs, false promotion 5.07 %) | `measurements/sprt_promotion/` |
| 2026-10-02 | Performance checks **(owner)** | deterministic performance-SHAPE tests + an on-demand benchmark | a wall-clock perf guard | `48265bf8` |
| 2026-10-02 | Pool defence against cycling **(owner)** | tiered matrix + Nash averaging + Hodge meter, validated against dense; grow the pool rather than evict harder when cycles are wide | hope; recency-only eviction | TASK_BACKLOG T20 |
| 2026-10-02 | The eval LEDGER **(owner)** | append-only JSONL of COUNTS per (batch × matchup), with pentanomial pair counts and per-team counters, archive-level, one writer per file | a per-game ledger (richer than any planned reader needs) | §0b: counts are sufficient statistics for every planned estimator |
| 2026-10-03 | The ledger's FIRST WRITER (X5 P0, orchestrator) | `main.h2h` writes one row per (batch × matchup) to a caller-named directory through `eval_ledger` (`gen3_eval_count_row_v1`: per-side team counters `{p, o}`, pentanomial, regime + `regime_id`, a `compute` block incl. a near-tie census); the writer refuses `models/` until the archive ledger exists | waiting for the archive ledger before measuring anything | the P0 pre-study needed a durable, auditable row format now; the schema is the §0b one |
| 2026-10-02 | Doc split (orchestrator) | the SYSTEM here; the three population DECISIONS in `design_league_decisions.md` | one combined doc | separate use cases from implementation (owner's suggestion) |
| 2026-10-03 | **The eval-system DESIGN (eval design agent; PROPOSED, pending the independent review and the owner's Q1–Q7)** | ledger v2 with request ids, protocol versions, flags, decision / request / reference streams, and a declared-reader API (§0b); one scheduler with priority classes and a VOI rule (§0d); T19 as three lanes under ONE GPU owner, i.e. the trainer's blocking window + a CUDA-free CPU lane + offline GPU only when no run is live (§4); the budget from precision (§5); T20 as incremental dense + targeted top-ups with a frozen posterior-mean Nash reference and a pre-registered validation (§2); SPRT resumes across restarts | a second GPU eval process (two owners; K6's ceiling); GPU filler now (unmeasured phases; not needed by the budget); four tiers for an active pool ≤ 40 (incremental dense is cheaper); point-Nash references (pure and unstable on our data); a fixed power-prior a₀ (homebrew) | §0a; `measurements/eval_design_2026-10-03/`; P0 (`x5_p0_h2h_2026-10-03/`); O9 |
