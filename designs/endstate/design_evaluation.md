# Evaluation end state — the unified system: ledger, scheduler, estimators, budget

**Status: ALWAYS-CURRENT.**
- The skeleton went in 2026-10-02: the owner said "put the skeleton in and the TODOs, and we fill it out with our implementation over time".
- **The DESIGN went in 2026-10-03,** written by the eval-system design agent.
- **REVISED 2026-10-03** after an independent review (verdict SOUND WITH FIXES: six must-fix items, fourteen
  fix-during-build items) and three pieces of owner input:
  - the plateau test is TWO-TIER;
  - cycling detection is a STANDING instrument (the cycle monitor, §2.5);
  - background eval starts GPU-WINDOW-ONLY.

  Each must-fix item's resolution is in the Decision record. The design is still a PROPOSAL awaiting the owner's
  answers to §11.2. Nothing in it is built unless marked BUILT.

This doc is the SYSTEM: how eval evidence is produced, stored, scheduled and estimated. The population DECISIONS (promotion, eviction, plateau) live in [`design_league_decisions.md`](design_league_decisions.md). A build or decision that differs from this doc updates it and its Decision record in the same commit. Era context: [`era_plan_post_m5.md`](era_plan_post_m5.md). The evidence behind the numbers measured on banked data is in [`measurements/eval_design_2026-10-03/`](../research_state/measurements/eval_design_2026-10-03/README.md); the revision's simulations are in its `revision/` folder.

**Direction (owner):** robust, durable, auditable, and reusable later to answer new questions WITHOUT a new training run.
Since 2026-10-03 it should also be **simple and highly optimized**.
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
| 2 | T20 (the pool matrix), the reference mixture, eviction, **the CYCLE MONITOR (§2.5)**, the validation rule |
| 3 | statistics in use |
| 4 | T19: background eval in the GPU WINDOW, under the one GPU owner |
| 5 | the budget derived from precision |
| 6 | re-grounding the week-one assumptions |
| 7 | TODO |
| 8 | **the PLATEAU test: two tiers**, and what it reads |
| 9 | determinism, audits and the seat RNG |
| 10 | build plan |
| 11 | open gaps, owner questions, findings |

## 0. Principles
1. **Every eval DECISION is deterministic, with a declared error rate.** That means a sequential test with stated α/β,
   or a margin derived from a measured or simulated null. It is never a threshold we hope the noise stays on the right
   side of. A decision whose statistic falls within a declared rounding band of its boundary is NOT a crossing (§9).
2. **Mirrored team pairs, with the PAIR as the unit,** for every SYMMETRIC comparison: pool eval, promotion, the
   pool matrix, the cycle monitor, the plateau test, anchors. Pinned-team meters (untaught, `best_response_gap --play`)
   stay unmirrored, because mirroring would mix piloting with response (`eval_and_rating.md`). The mirror's job is
   BALANCE: both sides pilot both teams, exactly. Its variance gain is small: ~12 % of games on a cross edge (§9.3).
   **Estimators on mirrored rows are PAIR-level** (a Dirichlet over the pentanomial, pair-level intervals; §3).
3. **The regime is recorded on every row** (greedy or sampled, mirrored, the eval core, the protocol version, the seat
   rule). Readers refuse to mix regimes.
4. **A cheap method is trusted only after it reaches the dense method's decisions on our own data.**
5. **Re-ground week-one choices** (§6). Several eval parameters were set by what one CPU could afford in week
   one, not by the precision a question needs.
6. **One GPU owner at a time, never a queue of waiting agents** (owner, 2026-10-03).
   - While a training run is live, the TRAINER is the GPU owner. Every eval game of that run is played inside its
     process, in its startup-declared eval slots, in the blocking eval WINDOW (§4).
   - With no run live, the agent holding the GPU lease (`scripts/ops/gpu_lease.sh`, since `5876c2ea`) runs the
     offline engine. `gpu_lock` fails fast (`GpuLeased` / `GpuBusy`) instead of waiting.
   - A CUDA-free CPU worker is an OPTIONAL later unit, built only when the measured budget exceeds the window cap (§4.5).
7. **Estimates read COUNTS from the ledger, never traces, never a selected sample** (§0c).
8. **Simple first** (owner, 2026-10-03). One lane is built (the window). A second lane is built only after a
   measurement says the first cannot carry the budget.

## 0a. Literature review, applied to our setting

**Scope.** These are the references the owner listed in T20 and in the brief, each verified against a primary or
index source (arXiv, the publisher, Project Euclid, PMLR, AAAI, JMLR) on 2026-10-03. The review corrected six
citations, and they are corrected in place (Decision record, item 18). Items still marked **UNVERIFIED** could not be
checked from a primary source. Every row ends with what the reference implies FOR US.

### 0a.1 Rating and ranking under non-transitivity

| reference | key result | implication for us |
|---|---|---|
| Elo, *The Rating of Chessplayers, Past and Present* (Arco, 1978); Bradley & Terry, "Rank analysis of incomplete block designs I", *Biometrika* 39:324–345 (1952); Herbrich, Minka & Graepel, "TrueSkill™", NIPS 19 (2006/07) | One scalar strength per player, P(i beats j) = σ(r_i − r_j). TrueSkill adds a Gaussian uncertainty per player and an online message-passing update | **Keep BT as the SPINE** (the ladder, `main.elo`), because it is the right model where the data are transitive. On 70 banked within-run ladders no departure from BT was DETECTED (§11.3 F-ED-1, with its detection limit). BT cannot represent cycles, so it is never the only read of a multi-lineage pool. TrueSkill's online update buys nothing over a batch BT fit on counts we keep forever |
| Balduzzi, Tuyls, Perolat & Graepel, "Re-evaluating Evaluation", NeurIPS 2018, arXiv:1806.02643 | For an antisymmetric payoff matrix there is a UNIQUE maximum-entropy Nash equilibrium (Prop. 4). The Nash average is invariant to redundant copies of an agent (Thm 1 P1). An ε-perturbed matrix gives an ε-Nash (P2). **P3(ii):** when the meta-game is transitive, the max-entropy Nash is uniform on the top-rated player(s). It is found by an LP. The paper gives no sampling rule | **Two consequences, one good and one a trap.** (a) Copy invariance is exactly what an archive needs: consecutive snapshots are near-duplicates, and uniform averaging would over-weight eras with many snapshots. (b) In a near-transitive archive the Nash is PURE on the frontier (P3(ii)). On N0's matrix the point estimate puts all weight on one node, and its posterior spreads over the top ~8 (F-ED-3). So "zero Nash weight" is the COMMON case and cannot alone decide eviction (§2.4). **Nash SUPPORT size over time** is a cycling signal: transitive progress collapses the support onto the newest snapshot, cycling spreads it (§2.5 signal d) |
| Omidshafiei et al., "α-Rank: Multi-Agent Evaluation by Evolution", *Sci. Rep.* 9:9937 (2019), arXiv:1903.01373 | Ranks by the stationary distribution of an evolutionary chain over pure profiles. Handles general-sum and asymmetric games. As α → ∞ the ranking depends only on the response graph | Our game is symmetric and zero-sum, where Nash averaging is unique and cheaper. **α-Rank stays in reserve**, for a general-sum question (e.g. team-vs-team asymmetric matchups) if one arises |
| Rowland et al., "Multiagent Evaluation under Incomplete Information", NeurIPS 2019, arXiv:1909.09849 | **ResponseGraphUCB:** keep the unresolved edges; sample from them; drop an edge once its confidence intervals separate (Hoeffding or Clopper–Pearson); stop when none remain. With Hoeffding, O(Δ⁻² log(1/(δΔ))) samples suffice w.p. ≥ 1 − 2δ (Thm 4.2) | **The T2 targeted tier's sampling rule** (§2.2), with ONE stated deviation: our interval is a time-uniform BETTING confidence sequence on the pair score (Waudby-Smith & Ramdas 2024, §0a.4), not a fixed 95 % interval re-checked after every top-up, so the repeated looks do not inflate the error rate. The 1/Δ² cost says that resolving near-½ edges between neighbouring snapshots (Δ ≈ 1–3 pp) is unaffordable by design, so a decision must never NEED one |
| Jiang, Lim, Yao & Ye, "Statistical ranking and combinatorial Hodge theory", *Math. Programming* 127:203–244 (2011), arXiv:0811.1067 | Any edge flow on a comparison graph splits orthogonally into a GRADIENT part (the global ranking, by weighted least squares on the graph Laplacian), a CURL part (triangle-local cycles) and a HARMONIC part (cycles around longer loops). Local consistency implies global only when the clique complex has no harmonic part | **The cycling meter** (`hodge.py`, BUILT for trainee × bots), and signal (a) of the cycle monitor. Report gradient / curl / harmonic shares against a PARAMETRIC-BOOTSTRAP null computed from the same graph and counts (§2.5). **A sparse schedule must contain triangles:** a chain of "new vs previous" comparisons hides every cycle, which is why the cycle monitor's rows close triangles by construction (§2.5). The density window in which random graphs have β₁ = 0 (Kahle, cited by Jiang et al.) is **UNVERIFIED** |
| Czarnecki et al., "Real World Games Look Like Spinning Tops", NeurIPS 2020, arXiv:2004.09468 | Strategy space has a transitive axis and a non-transitive width, largest at middling skill. **Under the paper's own assumptions** (a finite game with the layered "game of skill" geometry it defines), a population covering a full Nash cluster gives transitive improvement when beaten (Thm 3), and fixed-memory fictitious play needs a population at least as large as a layer to converge (Prop. 3) | **Guidance, not a guarantee:** whether our game has that layered geometry is untested, so the theorems' conditions are not known to hold. The heuristic is that the pool cap should follow the measured Nash-cluster size, not habit. Within one lineage the cluster is SMALL on banked data (N0's posterior Nash support ~1–8 nodes; no within-run cycling DETECTED). The MULTI-lineage pool of era step 1 is unmeasured; the cycle monitor (§2.5) and the T20 matrix measure it before the cap changes |
| McKelvey & Palfrey, "Quantal response equilibria for normal form games", *Games Econ. Behav.* 10:6–38 (1995) | A smoothed equilibrium (logit QRE) that is continuous in the payoffs | One of the three candidate strength REFERENCES the N0 back-test compares (§2.3, Q5) |

### 0a.2 Active and sparse evaluation

| reference | key result | implication for us |
|---|---|---|
| Heckel, Shah, Ramchandran & Wainwright, "Active ranking from pairwise comparisons and when parametric assumptions do not help", *Ann. Statist.* 47:3099–3126 (2019), arXiv:1606.08842 | Rank by the Borda score with successive elimination on confidence intervals; query count ≤ c·log(n/δ)·Σ_i f(Δ_i), f(x) ≈ log log(1/x)/x². Parametric models buy at most logarithmic factors | (a) Spend games ONLY where a decision is undecided: the T2 rule. (b) Assuming BT "to save games" is not a real saving. Borda against a uniform recent archive is one of the candidate references (Q5) |
| Yue, Broder, Kleinberg & Joachims, *JCSS* 78:1538–1556 (2012); Bengs et al., *JMLR* 22(7) (2021) | Dueling bandits; the survey maps winner concepts (Condorcet, Copeland, Borda, von Neumann) to assumptions | Our "best member" target under cycles is the **von Neumann winner, i.e. the Nash mixture**. Eval games cost GPU time, not wins, so we use pure exploration (Rowland/Heckel), not regret minimization |
| Du et al., ICML 2021 (PMLR 139:2870–2879); Rashid, Zhang & Ciosek, AAAI 2021 (35(6):5673–5681); Chen & Joachims, WSDM 2016 | Near-low-rank payoff matrices need O(n·r·log n) entries; information-gain match picking; the blade–chest intransitivity model | **The ARCHIVE-scale tool** only. Not needed for an active pool of ≤ 40 or for the cycle monitor's thinned archive. Not built now |

### 0a.3 Reusing old evidence

| reference | key result | implication for us |
|---|---|---|
| **Chen & Ibrahim**, "Power prior distributions for regression models", *Stat. Sci.* 15(1):46–60 (2000), doi:10.1214/ss/1009212673 (author order VERIFIED at Project Euclid: Ming-Hui Chen, Joseph G. Ibrahim) | π(θ \| D₀, a₀) ∝ L(θ \| D₀)^{a₀}·π₀(θ), 0 ≤ a₀ ≤ 1 | **The T0 tier's form.** Old games of a matchup enter an ESTIMATE as a₀-weighted counts. **a₀ = 0 until validated** (§2.7) |
| Duan, Ye & Smith, *Environmetrics* 17:95–106 (2006); Hobbs, Carlin, Mandrekar & Sargent, *Biometrics* 67:1047–1056 (2011) | The normalized power prior and the commensurate prior learn how much to borrow, and borrow less on conflict | **How a₀ gets set:** estimated per SOURCE CLASS on the first dense audits, then FROZEN and registered; a class whose τ says "conflict" gets a₀ = 0 |
| Meta-analysis (fixed- vs random-effects pooling) | Pooling across studies needs a between-study variance term | Already our practice for RUN-level reads (P0's σ_h, X5 §7.4) |

### 0a.4 Sequential testing and repeated monitoring

| reference | key result | implication for us |
|---|---|---|
| Wald, *Ann. Math. Statist.* 16:117–186 (1945); Van den Bergh and the Fishtest GSPRT notes | The SPRT; Fishtest's pentanomial GSPRT over colour-swapped game PAIRS | **BUILT (`sprt.py`)**: promotion, and Tier 1 of the plateau test (§8). Every verdict is re-derivable from per-batch rows (§9) |
| **Lan & DeMets**, "Discrete sequential boundaries for clinical trials", *Biometrika* 70(3):659–663 (1983), doi:10.1093/biomet/70.3.659 (VERIFIED); O'Brien & Fleming, *Biometrics* 35:549–556 (1979); Jennison & Turnbull (2000) | Alpha-spending: only the spending function is fixed; the looks need not be pre-scheduled | In use (X5 §7.4, whose look cells the ledger serves as one request FAMILY, §0b.2) |
| Howard, Ramdas, McAuliffe & Sekhon, *Ann. Statist.* 49(2):1055–1080 (2021), arXiv:1810.08240 (journal ref VERIFIED); Waudby-Smith & Ramdas, *JRSS-B* 86:1–27 (2024); Johari et al., *Oper. Res.* 70:1806–1821 (2022) | Confidence sequences valid under continuous peeking; the stitched boundary is Howard et al.'s **eq. (10)** (corrected from "eq. 11" by the review); betting CSs are the tightest known for [0, 1] outcomes | **Used for every REPEATED interval check:** T2 eligibility (§2.2) and the panel's running score (§8). A fixed 95 % interval is used only where it is read once |
| Page, *Biometrika* 41:100–115 (1954) | CUSUM with a declared average run length to false alarm | Optional refinement for plateau onset; the fixed 10M cadence supplies the spacing |

### 0a.5 League practice

| reference | key result | implication for us |
|---|---|---|
| Vinyals et al., *Nature* 575:350–354 (2019) | A league per race; prioritized fictitious self-play. Past players are KEPT. The 35 / 50 / 15 % split and the exact f_hard come from secondary sources, **UNVERIFIED** | GROW and sample rather than evict (the owner's 2026-10-02 decision). **The PFSP weight is NOT our eviction tie-break** (review M5): `--pfsp-scale` defaults to 0, and PFSP's p is a stateful EMA, not a ledger read. Eviction is a declared ledger read (§2.4) |
| Lanctot et al. (PSRO), NeurIPS 2017, arXiv:1711.00832 | The empirical meta-game IS the evaluation | Our pool matrix and the cycle monitor's thinned matrix are PSRO's empirical meta-game; `main.best_response_gap` is its exploitability read |
| OpenAI et al., "Dota 2…", arXiv:1912.06680 (2019) | Eval against 83 FROZEN references with held ratings; only references within 15–85 % | **Frozen references:** the cycle monitor's fixed ANCHOR SET and the panel (§2.5, §8). The 15–85 % rule is a Fisher-information argument, so the panel keeps only in-band opponents |

### 0a.6 RL evaluation methodology and variance reduction

| reference | key result | implication for us |
|---|---|---|
| Agarwal et al., NeurIPS 2021, arXiv:2108.13264 (bibliography verified; method details from memory) | With few runs, report IQMs, stratified-bootstrap CIs, probability of improvement | A ledger read declares whether its CI is conditional on the run or across runs (`ReaderDecl.inference`, §0b.7) |
| Henderson et al., AAAI 2018, arXiv:1709.06560 | Seeds and nondeterminism flip conclusions | Every row records commit, protocol, seed rule, compute block and an OUTCOME DIGEST, so any batch can be replayed and audited (§9.2) |
| Burch, Schmid, Moravčík, Morrill & Bowling, "AIVAT", AAAI 2018, arXiv:1612.06915 | Unbiased control variates from a value function. **The paper's abstract claims "more than a factor of 10" fewer hands** in no-limit poker. The often-quoted **85 % SD reduction (~44× fewer games)** is DeepStack's report for its match against professionals (Moravčík et al., *Science* 356:508–513, 2017), corrected here from an earlier attribution to the AIVAT paper | Our mirrored pairs are duplicate play and buy only ~12 % on a cross edge (F-ED-4). AIVAT-style control variates are the grounded next step if eval games ever bind. **Not adopted now:** they need a calibrated value (era step 3). A standalone primary citation for "duplicate poker" was not found (**UNVERIFIED**) |
| Domhan et al., IJCAI 2015; Swersky, Snoek & Adams, arXiv:1406.3896 (2014) | Learning-curve extrapolation; freeze-thaw BO | For the plateau's G2 (a current slope); the reported strength secondary supplies the series (§8) |

### 0a.7 What the review rules in and out
- **IN:**
  - the BT spine;
  - max-entropy Nash on the cycle monitor's and the pool's matrices, read with its posterior;
  - HodgeRank shares against a parametric-bootstrap null;
  - ResponseGraphUCB-style targeted sampling, on betting confidence sequences;
  - the power prior with a₀ estimated, not chosen;
  - the pentanomial GSPRT (promotion and plateau Tier 1);
  - confidence sequences for every repeated interval check;
  - frozen reference agents (the anchor set and panel).
- **OUT for now, in reserve:** α-Rank; matrix completion; regret-minimizing dueling bandits; AIVAT; TrueSkill.
- **CORRECTED by the review and the revision:**
  - "evict at zero Nash weight" is near-vacuous in a transitive pool (§2.4);
  - incremental dense is cheap for a pool of ≤ 40 (§2.1);
  - a plateau read as the DIFFERENCE of two scores against a reference needs twice the pairs of a direct
    head-to-head (§8), which is one reason the head-to-head is the plateau's primary.

## 0b. The eval LEDGER — append-only, aggregated (owner, 2026-10-02)

**One append-only JSONL ledger of COUNTS, not of games.** The owner put it as: "per-opponent would be great, running counters for teams; we don't need the richest data ever".
- **One row per (batch × matchup).** A matchup is one player against one opponent, under one regime, for one purpose.
- **Why counts are enough:** they are SUFFICIENT STATISTICS for every planned estimator: win counts per pair (BT,
  HodgeRank), the pair win-rate matrix (Nash), pentanomial counts (the GSPRT, pair-level estimators), counts by regime
  and purpose (the power prior), team counters (per-team reads).

  **What is given up:** per-game covariates and replaying a sequential test's PATH inside a batch. A batch is the
  test's check interval (§0d), so the path at the test's own resolution IS recoverable. **What is kept for audit:** a
  per-batch OUTCOME DIGEST and the near-tie game indices (§0b.2), so a replay can be checked game for game without
  storing the games.

### 0b.1 What exists (BUILT, 2026-10-03, X5 P0, `54b78aed`)
`agents/training/eval_ledger.py` provides:
- the row schema `gen3_eval_count_row_v1` and its validator (shapes, closed vocabularies, and the arithmetic tying the
  blocks together: W + L + D = games; half-points = 2W + D; team counters sum to the games and wins);
- one shard per writer process, with an fsync per row;
- a globbing reader that validates every row and drops superseded ones;
- a refusal of any output directory under `models/`.

**Two writers exist.**
- `main.h2h`: 120 rows under `measurements/x5_p0_h2h_2026-10-03/rows/`, about 25 KB per row raw (a 530-team counter
  map) and 3.7 KB gzipped.
- The Rustboro-era bot round robin (`measurements/bot_base_ratings_2026-10-03/bot_rr.py`, `3ccef556`): 1,296 rows,
  purpose `anchor`, bots as players (`id: bot:<name>`), sampled vs sampled, seat-balanced
  (`mirror_rule: gen3_mirrored_pairs_v1+seat_alternating_by_pair`). Its seats can alternate because BOTH sides are
  Rust-native bots. A POLICY player cannot alternate today (§9.3).

### 0b.2 The schema, v2 (`gen3_eval_count_row_v2`)
v2 keeps every v1 field and meaning. **A v1 row is never rewritten:** the reader upgrades it on read, with the
defaults below.

| field | v1 | v2 | why |
|---|---|---|---|
| `purpose` | closed list: promotion, plateau, matrix, audit, anchor, training, cycle | **+ `ab`, `ladder`, `untaught`, `gap`, `monitor`** (§11.2 Q1). `ladder` is for BACKFILLED snapshot-ladder rows only; new pool rows are `matrix` | F-P0-6 (a pre-registered A/B read had to be written as `audit`); `monitor` names the cycle monitor's rows (§2.5) |
| `request` | — | `{id, kind, family, opened, batch}`, or `null` only on a backfilled row | **`id`**: the peeking rule made exact (§0c rule 1). **`batch`**: the batch index inside the request (review M4); `(request.id, batch, matchup)` is UNIQUE (§0b.4). **`family`**: the id of a group of requests one decision may read together (review M3(a)): every look of a group-sequential A/B, or the two tiers of one plateau check. `null` = the request is its own family. `kind` ∈ {cycle, sprt, plateau_t1, monitor_row, monitor_topup, panel, matrix_dense, matrix_target, matrix_probe, ab_cell, anchor_read, untaught_read, gap_read, audit_replay, audit_dense, adhoc} |
| `player.kind`, `opponent.kind` | (checkpoints only) | `checkpoint` \| `bot` \| `external` | Bots and outside agents become first-class. For a non-checkpoint, `sha256` is the digest of (agent name, version, code commit) |
| `regime.protocol` | — | a named eval-protocol version, e.g. `gen3_eval_protocol_v2` | Every SEMANTIC change to what a game measures bumps it. Readers never pool across a protocol. A request family can PIN one (review M3(c), §0c rule 6) |
| `regime.seat_rule` | (implicit: the player keeps p1) | `fixed_p1` \| `balanced` | **New policy rows are `fixed_p1`** until unit U10 builds a p1 policy route on the core (review M1, §9.3) |
| `regime.player_temp`, `regime.opponent_temp` | — | `null` when greedy, else the temperature | The 2026-09-07 bug pooled greedy-vs-T = 1.0 with greedy-vs-greedy (+8.9 pp) |
| `regime.team_set` | (`team_source`, free text) | a digest of the team set and builder parameters | The identity changes when the TEAMS change, not a label |
| `counts.aborted` | (only `pairs.voided`) | games started and not finished, on every row | The INCONCLUSIVE rule needs the denominator. It is applied per CELL, i.e. per (request, matchup) (§9.1) |
| `compute.outcome_digest`, `compute.near_tie_games` | (`compute` has a near-tie census) | sha256 over the ordered per-game outcome vector `(game index, W/L/D, turns)` of the games WITHOUT a near-tie decision; the list of game indices that HAD one | **The replay audit works on count rows** (item 16, §9.2) |
| `flags` | — | closed list: `draws_folded`, `teams_unrecorded`, `seed_unrecorded`, `sha_unrecorded`, `eval_core_unrecorded`, `digest_unrecorded` | Backfilled rows say exactly what they lack; a reader DECLARES which flags it accepts |
| `provenance` | — | `{source_file, source_line, source_sha256, backfill_id}`, or `null` | Backfill auditability |

**The v1 → v2 upgrade on read is deterministic.** It sets `request = null`; `kind = bot` when the id starts `bot:`,
else `checkpoint`; `protocol = gen3_eval_protocol_v1_<writer>` (`h2h`, `bot_rr`); `seat_rule = balanced` when
`mirror_rule` contains `seat_alternating_by_pair`, else `fixed_p1`; both temperatures `null` for greedy rows and
`"bot_native"` for the bot round robin's sampled rows; `team_set` = the digest of `team_source`; `aborted = 2 ×
pairs.voided`; `flags = ["digest_unrecorded"]`. `regime_id` is recomputed over the v2 identity; the v1 id is kept as
`regime.v1_id`.

**Companion streams,** in the same append-only, one-writer-per-file discipline:
- **`decisions/`** (`gen3_eval_decision_v1`): one row per DECISION: `decision_id`, `kind` (promotion / eviction /
  plateau / ab_verdict / cycle_flag), `subject`, the `request_id` or `family` it decided, the digest and count of the
  rows it consumed, its `as_of`, the rule and version, the verdict, `ts`. **Rows are never edited after a decision.**
- **`references/`** (`gen3_eval_reference_v1`): immutable FROZEN reference mixtures (§2.3): members (sha256), weights,
  solver and version, posterior draw count and seed, the digest of the rows it was solved from, `created_at`.
- **`requests/`** (`gen3_eval_request_v1`): `open` / `claim` / `void` / `done` / `cancel` events (§0b.4's claim
  protocol). The queue state is a deterministic fold of events.

### 0b.3 Where it lives
- **Archive-level:** `<archive>/_ledger/`, where `<archive>` = `utils.paths.run_archive_dir()`. It is never per run,
  so cross-run questions need no new run. Pytest SEALS the archive (the `run_archive` fixture).
- **Layout:** `_ledger/rows/<producer>/ledger.<writer_id>.jsonl` (gzipped once closed); `_ledger/decisions/`,
  `_ledger/requests/`, `_ledger/references/`; `_ledger/backfill/manifest.json`; `_ledger/README.md`.
- **One writer per file.** `writer_id` = UTC time + host + pid + producer. Rows are fsynced per append.
- **Closing a shard:** the writer gzips its own shard at a clean exit; `python -m main.eval_ledger close-stale`
  closes a dead writer's shard only when its pid is dead AND the file has been idle 24 h.
- **`refuse_under_models` changes meaning:** a ledger writer may write ONLY under `<archive>/_ledger/`.
- **Retention:** the ledger is NEVER deleted (`models_retention_policy.md` gets one line). Size from §5's budget:
  ~45k games per 10M steps in ~800 rows ≈ 20 MB raw / 3 MB compressed per 10M, ~22 MB compressed per 75M run.

### 0b.4 The schema gate, and the no-duplicate-batch invariant
**In one sentence (owner's framing):** two workers must never record the same seeded batch twice, because
double-counted games make a decision look twice as certain as it is. The window-only design has ONE writer per run,
so this is a cheap guard, not a hot path. It costs a claim id and an audit.

1. **Writer:** validate before append (BUILT).
2. **Reader:** validate every row and refuse on the first malformed one (BUILT). **Every reader also refuses a
   duplicate `(request.id, request.batch, player.sha256, opponent.sha256, regime_id)`** (`DuplicateBatchError`), so a
   duplicate can never reach an estimator, whether or not `audit` has run.
3. **Claims are atomic.** A batch is played only under a CLAIM:
   - The scheduler appends `claim {request, batch, writer_id, expires_at}` while holding an exclusive `flock` on
     `_ledger/requests/.lock` (a CPU file lock; nothing to do with the GPU).
   - `expires_at` = claim time + 4 × the request's measured batch wall, with a floor of 10 min.
   - The writer appends its row under the SAME lock, after re-reading that its claim is still live. A writer whose
     claim was voided drops its batch.
4. **A dead writer's claims are voided deterministically.** A claim is VOID iff no row exists for its (request,
   batch) AND (its writer's pid is dead on its host OR now > `expires_at`). The scheduler writes a `void` event. The
   batch is re-claimed and replayed on the SAME seed block (§4.4), so the replay is the same games, not new ones.
5. **`python -m main.eval_ledger audit`** validates the whole archive and the CROSS-row invariants. The scheduler runs
   it at startup and refuses to start on a failure. The invariants:
   - `row_id` is unique, and so is the batch key above;
   - every `supersedes`, `request` and `family` resolves;
   - every row has a live or completed claim;
   - one regime per request, and one protocol per pinned family;
   - every `protocol` is known;
   - the decision → rows digests match.
6. **A static gate** (`src/eval_ledger_reader_gate_test.py`, empty allowlist): (a) every module that reads the ledger
   passes a `ReaderDecl` (AST-checked, like `trace_summary_reader_gate_test.py`); (b) the closed lists (`PURPOSES`,
   request kinds, flags, protocols) equal this section's tables, in the `recipe_doc_gate` pattern.
7. **A contract test per producer:** each producer's row builder emits a synthetic row that passes `validate_row`.

### 0b.5 Producers and their migration

| producer | what it writes today | v2 purpose / request kind | migration (unit, §10) |
|---|---|---|---|
| In-loop eval cycle (`eval_callback`, `selfplay_callback` → `rust_eval/`) | `<run>/eval_results.jsonl`: rates and `[won, finished]` per opponent; **no draw count** | `cycle` / `cycle` | **U2:** DUAL-write a ledger row per opponent with exact W/L/D and team counters. **U3c:** move `eval_results.jsonl`'s readers (TensorBoard/TUI, `main.elo`, `best_response_gap`, the supply guards) to the ledger, then retire the file |
| SPRT promotion (`sprt_promotion.py`) | `<run>/sprt_promotion.jsonl` (state only). **No run has one** | `promotion` / `sprt` | **U2:** rows per batch + a decision row. **U4b:** RESUME instead of abandon (§4.4) |
| Snapshot ladder (`snapshot_ladder.py`, run by the DETACHED updater `_spawn_snapshot_ladder_update`) | `<run>/snapshot_ladder/games.jsonl`: `{a, b, wins_a, games, source}` | `ladder` (backfill) / `matrix` (new rows, `matrix_dense`) | **U2:** dual-write. **U7:** the window plays the pool rows (T3) and the FIT reads the ledger. **U7a:** retire the detached updater |
| `main.h2h` | v1 rows to a caller-named directory, purpose `audit` | as the caller declares (`ab` for X5) | **U1:** archive default and `--purpose ab`. **Storage-only:** the game protocol is unchanged (§0c rule 6) |
| `main.anchors` | per-GAME `games.jsonl` + `summary.json` | `anchor` / `anchor_read` | **U3b** |
| `main.untaught_meter` | per-team cells + `_meta` | `untaught` / `untaught_read` | **U3b** |
| `main.best_response_gap` | READS `eval_results.jsonl` externals | reader of `cycle` rows; `--play` writes `gap` | **U3b** |
| `hodge.py` | reads cycle edges in process | reader | **U5** (the cycle monitor) |
| Bot round robin (`data/gen3_bot_elo_*`) | `data/` (source of truth) | not migrated | none |
| Training games per team (`team_winrate_callback`) | `metadata.json` | `training` | not now |

### 0b.6 The backfill (U3)
**One writer id (`backfill-<date>`), idempotent.** `_ledger/backfill/manifest.json` records each source file's sha256.
A re-run refuses unless a source changed, and then it writes SUPERSEDING rows, never edits.

| source | count on disk (2026-10-03) | maps to | recoverable | NOT recoverable → flag |
|---|---|---|---|---|
| `snapshot_ladder/games.jsonl` | 104 files, 5,766 rows, of which 824 are `source: eval_cycle` copies (DROPPED) | `ladder`, greedy/greedy, unmirrored; protocol by `recipe_version` | wins/finished per pair; sha256 of each snapshot zip that still exists | draws (`draws_folded`); teams; seed; eval core; digest; sha of a deleted snapshot |
| `eval_results.jsonl` | 265 run dirs, 2,071 rows | `cycle`, one row per opponent per cycle | W/finished per opponent; draws from `eval_manifest.json` `battles_drawn` where present | draws elsewhere; teams; sentinel sha (step only); digest |
| committed `measurements/anchors_*/` | per-game rows | `anchor` | W/L/D, per-team counts, regime; pairs where mirrored | our checkpoint's sha if the file is gone |
| P0 rows | 120 v1 rows | copied verbatim | everything but the digest | `digest_unrecorded` |
| bot round robin | 1,296 v1 rows | copied verbatim | everything but the digest | `digest_unrecorded` |

**The regime of a cycle row,** inferred in this order: (1) the row's `sentinel_regime` stamp (authoritative; 385 rows,
all greedy); (2) else the run's `model_config.json` `eval_sentinel_greedy`; (3) else pre-boundary ASYMMETRIC.
**The bots themselves changed:** the setup-branch fix (`f885ad8f`) and Curse-as-setup put every pre-fix bot edge in
the old bot era, and `30ff42f3` installed the Rustboro-era anchors as the boundary. A backfilled bot edge gets the
bot identity of its era (else `sha_unrecorded`); readers never pool bot eras. F-LH-13's 20 affected runs' `ext_` rows
get their own protocol value. **No RUN's eval history holds pentanomial data**, so every backfilled run row is
unmirrored.

### 0b.7 The reader API
Every estimator reads through ONE function and DECLARES what it consumes:

```
DECL = ReaderDecl(
    name="sprt_promotion",
    purposes={"promotion"},
    regime=RegimeFilter(protocol="gen3_eval_protocol_v2", play="greedy", opponent_play="greedy", mirrored=True),
    requests="own",            # "own" = the caller's request; "family" = its family (group-sequential decisions only);
                               # "any" = an estimate
    selection="exclude",       # drop the rows a decision consumed to select the node being estimated
    flags_ok=frozenset(),      # accept no legacy flag
    inference="conditional",   # "conditional" (meter noise only) or "across_runs" (adds the run term)
)
rows = eval_ledger.read(DECL, request_id=..., players=..., as_of=...)
```

- **One regime per call** (`MixedRegimeError` otherwise).
- **No duplicates:** the read refuses a duplicate batch key (§0b.4).
- **`requests="family"`** is legal only for a decision kind registered as group-sequential (today `ab_verdict` and
  `plateau`), and the family must name its rule.
- **`as_of`** (a timestamp or a run step) restricts rows AND decisions to those that existed then: the time-travel a
  BACK-TEST or an eviction needs.
- **`selection="exclude"`** implements "the games that selected a node do not rate it" through the decision stream.
- **The static gate** (§0b.4) fails any read without a declaration.

## 0c. Reuse rules (one ledger, many readers)
1. **A sequential DECISION counts only rows produced FOR it, after it started:** its own request, or its own family
   when it is registered as group-sequential. Older rows enter only as a declared, discounted prior, and only after
   validation (§2.7). ESTIMATES (ratings, the matrices, Hodge) may pool every eligible row.
2. **Pool only rows of the same regime** (`regime_id`, which includes `protocol`).
3. **Never estimate from a SELECTED sample.** Estimates read counts, never traces (the prober's quota prefers losses).
   The games that selected a node do not rate it (`selection="exclude"`).
4. **Legacy rows are opt-in** (`flags_ok`).
5. **A back-test reads `as_of` its decision time.**
6. **A pinned family freezes its protocol** (review M3(c)). A group-sequential A/B's registration names the protocol
   version and the commit its cells are played at. The ledger refuses a row into the family at another protocol, and
   no change to that protocol's game semantics lands while the family has no decision row. **X5 §7.4 is the first
   such family.** U1's `main.h2h` change is storage-only (location + `--purpose`). A seeded batch's outcome digest is
   compared byte-for-byte before and after U1 to prove it.

## 0d. One scheduler

**What it is.** ONE request queue and ONE allocation rule, serving every eval consumer, playing in the GPU window of
the run that owns the GPU (§4), or in the offline engine when no run is live, and appending counts to the ledger.

**A request** (`gen3_eval_request_v1`, an `open` event) declares:
- `id`, `kind`, `family` and priority class;
- the consumer;
- the players;
- the regime;
- a target: a fixed pair count, or a sequential rule id + cap;
- a latency bound;
- the eligible lanes;
- the FROZEN inputs it depends on: the pool at an SPRT's start, the thinned node set of a monitor row, a `ref_id`.

**Priority classes** (owner order: promotion > plateau > matrix > audits):

| class | consumer | request kinds | latency | lanes |
|---|---|---|---|---|
| P0 | the monitoring CYCLE (fixed, not queued) | `cycle` | every 2M steps | GPU window |
| P1 | promotion SPRT | `sprt` | urgent | GPU window |
| P2 | the plateau check and the CYCLE MONITOR | `plateau_t1`, `monitor_row`, `monitor_topup`, `panel` | within the 10M interval that follows the check's node | GPU window |
| P3 | the pool matrix (T20) | `matrix_dense`, `matrix_target`, `matrix_probe` | before the next eviction | GPU window |
| P4 | audits | `audit_replay`, `audit_dense` | idle | GPU window (from P2–P4's share); offline GPU |
| P5 | one-offs | `ab_cell`, `anchor_read`, `gap_read`, `untaught_read`, `adhoc` | per request | **offline GPU only** (foreign architecture, or no run live); CPU when no run is live |

**The allocation rule (deterministic given the open requests, the rows and the budgets):**
1. **The quantum is ONE BATCH:** 40 mirrored pairs for an SPRT or a plateau Tier 1 batch, and 50 pairs per edge for a
   monitor or matrix batch.
2. **The window's budget per 2M interval** is the REGISTERED 10 % of the interval's measured wall
   (`m5_sizing/REGISTRATION.md` §7), split as follows:
   - **P0 + P1 are bounded by their own rules:** P0 is fixed, and P1 is bounded by the SPRT's cap (1,680 pairs)
     and one test in flight.
   - **P2–P4 share c_gpu,** a cap of the window wall per interval: provisionally **6 %**, fixed by U4's first
     measurement (§4.2, Q3).
   - **The total never exceeds 10 %.**
3. **Serve order inside a window:** P0, then the in-flight SPRT's batches, then P2, then P3, then P4. Ties go to
   (latency, `opened`, `id`).
4. **When demand exceeds the budget, the YIELD order is fixed:** P4 first, then P3, then P2's top-ups.
   - **A P2 Tier 1 batch or monitor row is DELAYED to the next window, never dropped.** If a check cannot finish
     within its latency bound, its plateau call simply waits.
   - **P1 yields to P0 only.** An in-flight SPRT that cannot finish in this window resumes in the next one from its
     banked batches (§4.4).
   - P0 never yields.
5. **One SPRT in flight; a backlog of at most one.** A candidate arriving while a test runs is queued. A newer
   candidate SUPERSEDES a queued one that has not started: it is cancelled with reason `superseded`, having played no
   game, so nothing is selected. An in-flight test is never cancelled for a newer candidate.
6. **Anti-starvation (per lane, defined):** in any window where P3 or P4 has open requests, P3 + P4 receive at least
   20 % of that window's c_gpu share. It is carved from c_gpu, never from P0/P1, which resolves the conflict with the
   priority order.
7. **P5 does not starve because it has its own lane:** the offline engine (U6), where it is the top class (served by
   `opened`). P5 never runs in a training window, because foreign architectures cannot load there.
8. **Within P3, value of information** (§0a.2):
   - FIRST the dense-incremental rows of new members, n₀ = 50 pairs per edge (§2.1).
   - THEN targeted top-ups. An edge is ELIGIBLE iff its betting confidence sequence contains ½ AND at least one
     endpoint is UNDECIDED for the eviction read, i.e. posterior P(w > 0.01) in [0.05, 0.5]. Eligible edges are
     sampled uniform-exhaustively up to 500 pairs per edge.
   - When nothing is eligible, the matrix is **DONE** for those members (§2.4 reads only at DONE).

**Determinism.** WHICH batch runs WHEN depends on measured wall time, so it is not bit-reproducible. Every DECISION is
a deterministic function of its own request's or family's rows (§9.1). The allocation is AUDITABLE: every row names
its request and batch.

## 1. The questions eval answers (decisions: `design_league_decisions.md`)

| question | meter / test | status |
|---|---|---|
| Is plain training still paying? (**plateau**) | **TWO TIERS** (owner, 2026-10-03; §8): Tier 1 is the owner-registered head-to-head GSPRT, newest vs the W-back snapshot. Tier 2 is a READ of the cycle monitor, plus the panel. Plateau only when Tier 1 is flat AND there is no cycling. Strength vs a frozen reference is the REPORTED secondary | PROPOSED (league §C; Q8). U9 builds it |
| Are we CYCLING? (**cycle monitor**, a standing instrument) | the thinned-archive matrix grown one row per 10M check from step 0; four signals with parametric-bootstrap nulls (§2.5) | PROPOSED (owner, 2026-10-03). U5 builds it |
| Does a candidate join the pool? (**promotion**) | GSPRT on mirrored pairs, H0 0.50 / H1 0.55, α = β = 0.05, cap 1,680 pairs | BUILT, default OFF (`e9c5ab2d`); flips ON at X26. Ledger rows in U2; resume in U4b |
| Who stays in the pool? (**eviction**) | a DECLARED LEDGER READ at matrix DONE (§2.4) | PROPOSED (league §B; Q6) |
| How strong is it? (**rating**) | `snapshot_ladder/ladder.json` (dense BT, recipe-stamped) at run end, matched snapshot count | BUILT (`main.elo`); U7 re-points its fit at the ledger |
| Is the loop working? (**exploiter gap**) | `main.best_response_gap`: must FALL round over round | BUILT; U3b moves its reads to the ledger |
| Against the outside world? (**anchors**) | `main.anchors` (SmallRL at milestones; Kakuna/Foul Play at run end), greedy vs greedy, mirrored | BUILT; cadence TODO (T18); U3b writes rows |
| Is one arm non-inferior to another? (**pre-registered A/B**) | the X5 cross design (`design_x5_belief_tokens.md` §7.4) as one request FAMILY | BUILT as `main.h2h`; purpose `ab`; multi-cell engine U6 |
| Does the value tell states apart? (**discrimination**) | TODO | TODO (era step 3) |

## 2. T20, the reference mixture, eviction, and the cycle monitor

### 2.1 What the tiers are for: a correction to the skeleton
The skeleton's tiers (T0 reuse, T1 sparse probe, T2 targeted, T3 dense audit) were meant to save games against a dense
matrix of 190 pairs × 200 games ≈ 38k games. **That cost is only paid if the matrix is rebuilt.** A frozen pair is a
stationary quantity, so dense can be INCREMENTAL. Each new member plays the K − 1 others ONCE:
- K = 20 at n₀ = 50 mirrored pairs per edge (the ladder's 100 games, now balanced): 19 × 100 = **1,900 games per new
  member**, ≈ 16 s in the GPU window at 118 games/s (§5);
- at 100 pairs per edge: 3,800 games.

**So, for the ACTIVE pool (≤ 40 members), incremental dense IS the default.** The sparse tiers are needed only at
ARCHIVE scale and for the targeted top-ups a decision needs (Q4).

### 2.2 The tiers, as built under that correction

| tier | what | where | status |
|---|---|---|---|
| **T3 dense, incremental** | each new member × every member, n₀ = 50 mirrored pairs per edge (`matrix_dense` at promotion). It REPLACES the detached ladder updater's job | the active pool | DESIGNED (U7; the updater retires in U7a) |
| **T2 targeted** | top-ups on ELIGIBLE edges only (§0d rule 8), on betting confidence sequences, ≤ 500 pairs per edge | the active pool | DESIGNED (U7) |
| **T1 sparse probe** | a spanning set WITH CHORDS for an archive member | archive scale only | DESIGNED; built only when an archive matrix is wanted |
| **T0 reuse (power prior)** | older rows as a₀-weighted counts in ESTIMATES only | estimates | **a₀ = 0 until validated** (§2.7) |
| **audit** | a RE-PLAY of a random 2 % of banked batches (outcome-digest equality, §9.2) | both | DESIGNED (U4) |

**The window plays a frozen-vs-frozen edge as ONE player against up to FIVE opponents per pass** (review M1). Today
the executor submits the player only to `tb.trainee_slot` at p1 (`rust_eval/executor.py`). A frozen member is copied
into the trainee eval slot as the player, and up to five members fill the other eval slots. U4 sizes this executor
change.

### 2.3 The reference mixture: a REPORTED secondary, frozen, chosen by back-test
- **The game.** The symmetric zero-sum meta-game with payoff A_ij = P_ij − ½ on a frozen set. P_ij is the score (a
  draw counts ½), estimated at pair level from the pentanomial (§3).
- **The solution.** The max-entropy Nash of A (Balduzzi et al. 2018): the LP for the Nash set, then maximum entropy
  over it. The solve is deterministic (a fixed solver and tolerance, recorded).
- **Its uncertainty is LARGE on our data** (F-ED-3, as corrected by the review). N0's 20-node archive at 100 games per
  edge:
  - point Nash: pure on one node;
  - the newest node's score vs the posterior mixture: posterior SD **3.6 pp**.
  - Of that, **2.5 pp** is the MIXTURE's uncertainty (the newest row fixed at its point estimates) and **1.9 pp** is
    the newest node's own edge noise (the mixture fixed at its posterior mean). The two do not add in quadrature,
    because they interact through the solve.
- **The design consequence:** a strength read is against a FROZEN reference, written once to `references/`. Scores of
  different snapshots against the same reference are differences against one fixed opponent distribution.
- **Its role is now SECONDARY** (§8; review M2). The plateau's primary is the direct head-to-head, which needs one
  estimate, not a difference of two, and does not depend on the mixture's instability. The reference read is REPORTED
  as the strength series (G2) and the plateau-kind diagnostic. It is read from the cycle monitor's rows: the
  reference's members are thinned nodes, so a newest row already covers most of them. Uncovered support members are
  topped up by a `monitor_topup`.
- **Which reference is NOT decided** (review item 10, Q5). Three grounded candidates go to a pre-registered N0
  BACK-TEST, run after the pair-level estimators (item 9) exist:
  - (i) the posterior-mean max-entropy Nash (M = 1,000 Dirichlet posterior draws; members with mean weight < 0.005
    dropped and the rest renormalised);
  - (ii) uniform over the last K thinned nodes (Borda-style; Heckel et al.);
  - (iii) a logit QRE (McKelvey & Palfrey) at a declared λ.

  The back-test reads N0's thinned matrix `as_of` each 10M point and scores each candidate on two things:
  - (a) **re-solve frequency**: how often the frontier moves enough to require a new reference;
  - (b) **bridged-slope error**: the error of the slope chained across re-solves, against the slope on one reference
    fixed after the fact.

  The rule is fixed BEFORE the back-test: the candidate with the smallest (b) wins, provided its (a) is at most one
  re-solve per 20M steps. Ties go to the simpler candidate, (ii) before (i) before (iii).
- **Re-solve and bridge:** a reference is re-solved when the chosen candidate's rule says the frontier moved. The new
  one is BRIDGED: the last 3 thinned nodes are scored against both.

### 2.4 Eviction: a declared ledger read (review M5)
- **In a near-transitive pool, almost every member has zero Nash weight.** So "confidently zero weight" is a
  NECESSARY condition that binds rarely. It does not choose WHICH member leaves.
- **The rule** (a DECISION, so league §B owns it; Q6). When the active pool's cap binds:
  1. **Read** the active pool's matrix rows through `ReaderDecl(name="eviction", purposes={"matrix"},
     requests="any", selection="exclude", flags_ok=∅)`, `as_of` the decision time. The read happens only when the
     matrix is DONE for every member involved (§0d rule 8; review item 11). The decision row records the `as_of` and
     the consumed rows' digest.
  2. **Candidates:** members with posterior P(w > 0.01) < 0.05 under the Dirichlet-pentanomial posterior of the
     matrix (M = 1,000 draws, seed = the digest of the consumed row ids). A member within ±0.01 of 0.05 is NOT a
     candidate (rounding band, §9.1).
  3. **Choose** the candidate with the HIGHEST posterior P(the newest pool member beats it). The newest member stands
     in for the trainee: its dense row exists by construction, and the live trainee has no frozen row. Ties within
     1e-9 go to the oldest by step, then the lowest sha256.
  4. **None qualify:** GROW the pool (owner, 2026-10-02) up to the hard ceiling set by the eval and opponent slots
     declared at startup.
- **Cycle rows are NOT an eviction input.** Their player is the moving trainee, they are unmirrored, they cover only
  the sampled sentinels, and they are the games that SELECT promotions (§0c rule 3).
- **Eviction from the ACTIVE POOL is not deletion from the ARCHIVE.** An evicted member stays on disk and in the
  archive; deletion follows `models_retention_policy.md`. **Never delete a snapshot that a `references/` entry, an open
  request, or the cycle monitor's node set names.** The retention tool reads all three before it acts.
- **Before U8 trusts any of this,** an operating-characteristics simulation (U7b) runs the rule on synthetic pools
  (transitive, treadmill, multi-lineage) and reports the false-eviction rate of a member that carries support.

### 2.5 The CYCLE MONITOR: a standing instrument (owner, 2026-10-03)
**Why it exists.** Knowing whether we are cycling matters outside the plateau test too. It decides which lever family
applies (league §C), whether the pool must grow, and whether a population round helped. So it runs in EVERY run, at
EVERY check, from step 0. **The plateau's Tier 2 is a READ of it, not a separate test.**

**What it is.** A matrix over THINNED snapshots, grown one row per check:
- **Nodes:** the snapshot saved at every multiple of the spacing Δ (a declared dial, default **10M steps**, aligned
  with the plateau check).
- **Each check adds ONE row:** the new node plays every earlier node at a lag in the lag set L, plus the PANEL (§8).
  Snapshots are frozen, so old edges never change. The accumulated rows ARE the periodic matrix, with no re-bake.
- **Lag set (recommended: HYBRID).** L = {1, 2, …, 8} ∪ {16, 32, 64} × Δ: dense over the last 80M, geometric beyond.
  - **Cost:** for a run of ≤ 90M it equals uniform (every older node played). Beyond that, a row costs at most 8 + 3
    edges up to 650M, growing like log(run length).
  - **Triangles close by construction:** node t − a played t − 2a at its own check, for every geometric pair.
  - **The fixed ANCHOR SET** (registry-named frozen snapshots at the current architecture) is part of the panel. It
    ties every row to the same external points.
- **Games per edge:** **500 mirrored pairs (1,000 games)**, from the detection target below.
- **Re-solve at every check:** BT, the Hodge decomposition, and the Nash posterior on the accumulated matrix.

**The four signals, each with a pre-registered deterministic rule** (null = a PARAMETRIC BOOTSTRAP per protocol: 2,000
replicate matrices drawn from the BT fit on the SAME graph and pair counts, pentanomial sampling; review item 17).
Each flags at α = 0.05:

| signal | statistic | flag iff | what it sees |
|---|---|---|---|
| **(a) Hodge cyclic share** | Pearson χ² of the logit-WLS BT residual on the accumulated matrix (= its curl + harmonic energy); the curl / harmonic shares are reported | above the null's 95th percentile | diffuse cycling anywhere in the archive |
| **(b) lag-curve non-monotonicity** | on the newest row, max over adjacent lags of z(score at the shorter lag − score at the longer lag) | max z above the one-sided Bonferroni critical value over the k − 1 adjacent pairs | "beats the recent snapshot, ties or loses to an older one": the intuitive dashboard curve |
| **(c) intransitive triangles** | for each triangle through the newest node, the cyclic logit flow / its SE | any triangle significant after Holm control at α = 0.05 | a specific hole: a rock-paper-scissors loop |
| **(d) Nash support** | posterior mean support size (members with w > 0.01) and entropy of the max-entropy Nash on the accumulated matrix; P(w_newest > 0.01) | posterior mean support size above the null's 95th percentile | transitive progress collapses the support onto the newest node, cycling spreads it. **The dashboard line** |

**The verdict per check:**
- **CYCLING SUSPECTED** iff any signal flags. The simulated per-check false-alarm rate of (a) + (b) + (c) together is
  6–10 % (`revision/monitor_power.log`). SUSPECTED blocks a plateau call, which is the safe direction for the owner's
  fear.
- **CONFIRMED** iff the flagged signal flags again on a FRESH top-up: 500 more pairs on each edge carrying the flagged
  residual, triangle or lag pair, tested on the top-up games ONLY (independent of the selection) at α = 0.05. That
  gives a false-confirmation rate ≤ ~0.005 per check, at ≤ 3,000 games. CONFIRMED is reported to the owner as
  CYCLING, the TREADMILL kind (league §C).
- **NONE** otherwise.

A flag is written as a `cycle_flag` decision row.

**Detection target and what it buys.** These numbers come from a simulation: a synthetic concave Elo curve,
mirrored-pair noise at σ_pair 0.333, and the three tests above (`revision/monitor_power.py`, 600 replicates per cell).
"Power" here is any-signal power at the check.

| scheme, at the 80M check | k (edges in the row) | games per row | diffuse excess cyclic SD 1.0 / 1.5 / 2.0 pp | a single treadmill HOLE of 4 / 6 / 8 pp vs one older node |
|---|---|---|---|---|
| uniform = hybrid, 500 pairs/edge | 7 | 7,000 | 0.36 / 0.71 / 0.92 | 0.36 / 0.70 / 0.96 |
| uniform = hybrid, 1,000 pairs/edge | 7 | 14,000 | 0.67 / 0.96 / 0.99 | 0.68 / 0.97 / 1.00 |
| geometric (lags 1, 2, 4), 500 pairs/edge | 3 | 3,000 | 0.27 / 0.56 / 0.78 | 0.39 / 0.73 / 0.91 *at a played lag only* |

- **Within one check, at 500 pairs per edge,** the limit at 80 % power is a diffuse excess cyclic SD of **≈ 1.7 pp**,
  or a single hole of **≈ 6.5 pp**.
- **On the ACCUMULATED matrix, persistent cycling gets easier to see:** at the 150M check a diffuse 1.0 pp reads
  0.70 with the hybrid scheme at 500 pairs.
- **A 1 pp target WITHIN one check** needs ~4× the games per edge (Q9).
- **Legacy comparison:** F-ED-1's ladders (100 unmirrored games per edge) could detect ≈ 3.0 pp at 80 % power
  (`revision/fed1_limit.log`). The new monitor is roughly twice as sensitive per check.
- **Geometric alone covers only k of the m − 1 older nodes.** A hole at an unplayed lag is invisible from the new row,
  and its table power is CONDITIONAL on the hole falling at a played lag. That is why the recommendation is hybrid,
  not geometric.

**The owner's rough guess** ("~1,000 games per edge, k ≈ 8, ~11k games per check, ~1.5–2 % of wall at one check per
10M") is **right for the row alone**: 7,000–10,000 games, 1.2–1.6 % at 118 games/s. With the panel (6,000) and
confirmation top-ups (≤ 3,000), a check costs **2.1–3.1 %** of wall before Tier 1, and 2.7–4.0 % with it (§5).

**For exploiter-era pools** (era step 1), the monitor extends without a new mechanism:
- each exploiter's final snapshot gets a row against the generalist's thinned nodes at the lags in L, plus the panel.
  Its node is tagged with its lineage;
- the active pool's T20 matrix (§2.2) covers the pool members among themselves;
- signals (a)–(d) are computed twice: on the generalist-only submatrix (the plateau's Tier 2) and on the full
  multi-lineage matrix (the population's cycling read, PSRO's meta-game);
- the exploiter gap (`best_response_gap`) stays the population loop's primary meter.

**For a pinned run that carries none of this code** (X26, F-ED-19), the same rows are played by the monitor's
OFFLINE driver (`python -m main.cycle_monitor <run>`, U5). It runs CUDA-free on CPU under `mem_cap.sh`, or on the
offline engine when no run is live. Snapshots are frozen, so a row played late is the same row.

### 2.6 Archive scale (later)
The cross-run archive is where T1 + T0 + matrix completion pay off. **Not built until a question needs it.** The
ledger already makes it possible without a new training run.

### 2.7 The pre-registered validation (principle 4): DERIVED check rules
**Part A, targeted (T2) vs dense on the active pool.** It is run on each of the first THREE era-step-1 pools that
reach K ≥ 15.
- **Ground truth:** a full dense matrix at 400 pairs per edge, K(K − 1)/2 × 800 games (84k at K = 15). It is played by
  the offline engine (≈ 12 min at 118 games/s) or overnight on CPU.
- **The pipeline:** incremental dense at n₀ = 50 pairs + T2.
- **Every threshold is DERIVED, not picked** (the review's point on the first draft's validation). For each pool, a parametric bootstrap draws R = 2,000
  replicate (pipeline-sized, dense-sized) data sets from the dense posterior-mean matrix, i.e. a world in which the
  pipeline is a CORRECT estimator. It runs both procedures on each, and sets each check's threshold at the
  (1 − 0.05/9) ≈ 99.44th percentile of that null (Bonferroni over 3 pools × 3 checks). So a correct pipeline fails the
  whole validation with probability ≤ 0.05.

| check | statistic | PASS iff |
|---|---|---|
| 1. No false eviction | max, over members the pipeline would evict, of the dense posterior P(w > 0.01) | ≤ its null threshold |
| 2. No missed support | the number of members with dense P(w > 0.01) ≥ 0.5 that the pipeline would evict | ≤ its null threshold |
| 3. Reference agreement | \|the newest member's score vs the pipeline's reference − vs the dense reference\| | ≤ its null threshold |

- **This replaces the old fixed bars.** The old "1.5 pp" bar was unpassable: the reference's mixture alone carries
  ~2.5 pp of SD at 100 games per edge.
- **Rounding band:** an observed statistic within one Monte Carlo SE of its threshold makes that pool INCONCLUSIVE.
  The next qualifying pool replaces it.
- **Economy is REPORTED, not gated:** T2's 500-pair cap bounds it.

**FAIL means the pipeline is NOT trusted, and the cost of the fallback is stated honestly.** Dense at 400 pairs per
edge costs 19 × 800 = 15,200 games per new member at K = 20. At 1–5 promotions per 10M that is 15k–76k games, or
**2.5–12.5 % of wall** in the window at 118 games/s. So in the window it is affordable only at ≤ ~2 promotions per 10M.
Above that, the fallback is dense at 200 pairs per edge (1.2–6.2 %), or the optional CPU worker (§4.5).

**Part B, T0 reuse.** It uses the same three dense audits. a₀ per source class is estimated (commensurate model), then
frozen. T0 is switched ON for a class only if the refit with T0 added (and T2 games removed in equal number) still
passes checks 1–3 against the same derived thresholds. Otherwise a₀ = 0 permanently, recorded.

## 3. Statistics in use

| method | used for | reference (§0a) | status |
|---|---|---|---|
| GSPRT on the pentanomial, Wald bounds, batch checks | promotion; **plateau Tier 1** | Wald 1945; Van den Bergh (Fishtest) | BUILT (`sprt.py`) |
| **Pair-level estimators on mirrored rows:** a Dirichlet(counts + ½) posterior over the five pentanomial categories, score = Σ p_k s_k; pair-level SEs from the pentanomial variance / n_pairs | every mirrored edge: matrices, monitor, panel, references (review item 9) | Fishtest's pentanomial model | DESIGNED (U1 reader helpers) |
| Group-sequential O'Brien–Fleming / alpha-spending | pre-registered A/Bs (X5 §7.4), as one request family | O'Brien & Fleming 1979; Lan & DeMets 1983; Jennison & Turnbull 2000 | in use (X5) |
| Betting confidence sequences | every REPEATED interval check: T2 eligibility, the panel's running score | Waudby-Smith & Ramdas 2024; Howard et al. 2021 | DESIGNED (U5, U7) |
| CUSUM | plateau onset (refinement) | Page 1954 | optional |
| Bradley–Terry | the rating spine | Bradley & Terry 1952; Elo 1978 | BUILT (`main.elo`, ladder) |
| HodgeRank + χ² vs a parametric-bootstrap null | the cycle monitor (a); the pool's cycling read | Jiang et al. 2011 | BUILT for trainee × bots; monitor U5 |
| Max-entropy Nash with its Dirichlet posterior | monitor signal (d); eviction's necessary condition; one candidate reference | Balduzzi et al. 2018 | DESIGNED (U5, U7) |
| Holm step-down | monitor signal (c) | Holm 1979 | DESIGNED (U5) |
| ResponseGraphUCB-style targeted sampling | T2 top-ups | Rowland et al. 2019 | DESIGNED (U7) |
| Power prior, a₀ by the commensurate model | T0 reuse | Chen & Ibrahim 2000; Duan et al. 2006; Hobbs et al. 2011 | DESIGNED, OFF until §2.7 B |
| Random-effects across runs | run-level claims | (meta-analysis; P0 `main.h2h.runfloor`) | BUILT |
| Hodges–Lehmann shift + Wilcoxon one-sided bound | the CPU worker's throughput test, if ever built (§4.5) | Hodges & Lehmann 1963 | DESIGNED (U11, optional) |

## 4. T19: background eval in the GPU window, under the one GPU owner

### 4.1 The facts it starts from
- **The blocking in-process Rust eval cycle costs 1.57 % of wall** at N = 256 E10: the update cycles straddling an eval
  run +12–20 s against a 51.6 s median cycle, per 2M steps; 2.9 % at E5 (`m5_sizing/PROGRESS.md` O9).
- **SETTLED 2026-10-02** (`program_rust_core.md` Decision record): the blocking design stands while its share is
  ≤ 10 % (`m5_sizing/REGISTRATION.md` §7).
- **The 2026-09-30 record refused a SECOND GPU service,** which is consistent with the one-owner rule.
- **T2 is single-caller.** Its priority classes are BUILT but unused. The eval slots are declared at startup
  (1 trainee + 1 per sentinel + fixed opponents; 6 today) and hot-swap weights by `slots.copy_in`.
- **Game rate:** standalone T2 graph, 64 envs: **45–171 games/s per 1,000-game batch, median 130, pooled 118** (P0,
  contended box; `eval_and_rating.md`).
  - The IN-TRAINER window rate is UNMEASURED. O9's straddle implies 50–83 games/s INCLUDING the cycle's fixed
    overheads (sentinel loads, plan setup), i.e. a pessimistic **≈ 62 games/s** (16 ms of wall per game).
  - §5 shows both.
- **`rust_eval.launch.load_sentinels` builds a CPU `nn.Module` per cycle** (`load_opponent_snapshot`) before
  `copy_in` (F-ED-11, now resolved by design: U4a).
- **A CPU eval lane ALREADY EXISTS:** the detached snapshot-ladder updater (`_spawn_snapshot_ladder_update`,
  CUDA-free since `gen3_ladder_off_gpu_v1`, plays 100 games per new pool edge on the bridge). It runs beside training
  today, with no perturbation measurement (F-ED-18).

### 4.2 The design: the GPU WINDOW is the one lane built (owner, 2026-10-03)

| lane | runs where | GPU? | serves | budget |
|---|---|---|---|---|
| **GPU window** (BUILT for the cycle; generalized by U4) | INSIDE the trainer, in the blocking eval window at the 2M-step boundary, on the startup-declared eval slots (`copy_in`; no new slot, bucket or capture after startup, so K6 holds) | yes: the trainer IS the owner | P0–P4 | the registered 10 % per interval: P0 + P1 by their own rules, P2–P4 within c_gpu (§0d rule 2) |
| **Offline engine** (U6) | a one-shot engine run by the agent holding the GPU lease, only when no training run holds it | yes, as the sole owner for its lifetime | P5 (foreign-architecture cells, the X5 A/B looks), large dense audits, back-fills, the N0 back-test | the lease grant |
| **CPU worker** (U11, OPTIONAL) | a CUDA-free process beside the trainer | never | P3–P4 overflow | **NOT BUILT** until the trigger in §4.5 fires |

- **How it works:**
  - The scheduler fills each 2M window from the open requests in §0d's order.
  - A 10M check's work does NOT all land in one window. Every input is frozen, so the check's rows (Tier 1, the monitor
    row, the panel) are spread over the following ≤ 5 windows. That smooths the stall, and the latency bound is one
    interval.
  - A frozen-vs-frozen edge uses the trainee eval slot for the frozen player and up to five eval slots for its
    opponents (§2.2).
- **The first measurement of U4** (review item 13, Q3): before c_gpu is fixed, U4 measures in a real run's window:
  - the wall of one SPRT batch INCLUDING `load_sentinels` and the slot copies;
  - the steady in-window games/s;
  - the fixed overhead per window.

  These numbers replace §5's assumed 118 games/s. c_gpu is then set to 10 % − (P0 + P1 at their worst), rounded down
  to 0.5 %. It is registered with the measurement.
- **Regime equality across compute:** CPU eager and GPU graph play the SAME games (F-P0-7: identical W/L/D, pentanomial
  and team map on a 100-pair probe). So the lane is `compute`, not regime.
- **X5's A/B is unaffected** (review M3(d)):
  - the window runs only at the 2M boundary, and the speed rule excludes every update whose window contains an eval
    cycle (C-1's reader rule);
  - no CPU eval lane is built;
  - the X5 look cells are P5, played by the offline engine between arms (the training agent holds the lease), or on
    CPU only when no arm is running.
  - **If U11 is ever built, it never runs while an X5 A/B arm trains.** The existing ladder updater is the exception
    today; it is flagged to the orchestrator (F-ED-18).

### 4.3 What the window costs, with the two-tier plateau and the monitor
§5 derives it. **Mean 5.3–7.9 % of wall at 118 games/s; 9.0 % worst case after the fixed yields.** Under the
registered 10 %. At the pessimistic 62 games/s it would be 10–15 %, which is exactly the case §4.5's trigger is for.
The two-tier plateau plus the monitor are 2.7–4.0 % of that.

### 4.4 Restarts and aborts (A2's safe points)
- **Window:** A2 already abandons a partial eval cycle at a safe point (`deferred_abort.py`). Under the scheduler, a
  partial BATCH is never written. Requests are durable, so on resume the scheduler re-reads the open requests, voids
  dead claims (§0b.4), and continues.
- **SPRT: RESUME, not abandon (U4b, after U4; Q7).** Today `abandon_unfinished` drops an in-flight test at a restart
  (F-ED-9). With per-batch rows, the test's state is the SUM of its batch rows, and the stopping rule is checked only at
  batch boundaries. A restart's timing is outcome-independent, so resuming preserves the error rates. One decision row
  per candidate.
- **Seeds (reworded, item 19):**
  - every batch's seeds derive from (`request.id`, batch index) through `gen3_eval_game_seed_v1`;
  - **a banked batch is never replayed into the ledger;**
  - a DROPPED partial batch (never written) or a VOIDED claim is replayed on the SAME seed block, so it is the same
    games;
  - no seed block is ever counted twice (§0b.4).
- **Snapshot files are kept while a request is open** (item 19). The candidate's zip, the frozen pool's members, the
  monitor's thinned nodes and the anchor set are pinned against retention and against the snapshot pruner while any
  request naming them is open. A request whose file is missing anyway is CANCELLED with a reason event, never played
  against a substitute.

### 4.5 The optional CPU worker: what it would buy and when it is worth building
- **What it buys:** eval capacity that does not stretch training wall. On P0's contended figure, 4.8 games/s is ~25k
  games per 10M interval, roughly the whole P3 + P4 load.
- **What it costs:**
  - a second lifecycle (a CPU T2 service, declared slots, `SCHED_IDLE`);
  - a launcher integration;
  - the throughput test below;
  - the rule that it never runs during an X5 A/B arm.
- **Build it only if** one of these fires. Each is checked once, by the unit that produces its measurement:
  - (a) U4's measured in-window rate puts the §5 mean above 10 % (roughly, a rate below ~90 games/s);
  - (b) era step 1's multi-lineage matrix pushes P3 above its c_gpu share for two consecutive 10M intervals;
  - (c) the §2.7 FAIL branch needs dense at 400 pairs with > 2 promotions per 10M.
- **Its throughput test, pre-registered now so it is ready (review M6).** The old block-median rule almost never
  adopted: block-median log-ratio SD 0.082 from level shifts, P(adopt a free lane) ≈ 0.09 (F-ED-14). The replacement:
  - **Per-phase metrics:** the update's `train/train_ms` and the collection's `rollout/collect_ms` per decision.
  - **Finer alternation:** the worker toggles ON/OFF per UPDATE in a seeded random order within adjacent pairs, so
    slow level shifts cancel inside a pair.
  - **Exclusions by FLAG, not by threshold:** updates whose window contains an eval cycle or a restart, and updates
    carrying a periodic side task (the trainer marks them). On banked runs every ~10th update takes ~48 s of
    `train_ms` against ~41 s.
  - **Estimator:** the Hodges–Lehmann shift of the paired log differences, with its one-sided 95 % Wilcoxon bound. The
    noise is heavy-tailed: SD 0.18 vs robust SD 0.068 on collection in `sizing_B`.
  - **ADOPT iff both phases' bounds are ≤ log 1.01.** A bound within 0.0005 of the bar is NOT ADOPTED.
  - **Size from a power calculation on measured noise** (`revision/phase_noise2.log`, Monte Carlo on the banked paired
    differences of `sizing_B` / `sizing_C`, power at a true zero effect):

    | pairs | `train_ms` power | collection power (`sizing_B` / `sizing_C` noise) |
    |---|---|---|
    | 50 | 0.94 / 1.00 | 0.37 / 0.84 |
    | 150 | 1.00 / 1.00 | 0.72 / 1.00 |
    | 300 | 1.00 / 1.00 | 0.95 / 1.00 |

  - **Registered size: 300 pairs** (600 updates, ≈ 8.6 h of a production run that is training anyway). A worker gated
    to the UPDATE phase (paused during collection, where the env core saturates the CPUs) is tested on `train_ms` at
    50 pairs, with collection REPORTED under a harm flag (HL point > 1 %).
  - **The game-length covariate** the orchestrator asked for is a REPORTED sensitivity, not the primary adjustment:
    residualising collection on `rollout/ep_len_mean` INCREASED its noise on both banked runs (F-ED-15).

### 4.6 What comes after (not in this build)
- **GPU filler** (T2's built priority classes) only if the window share at the precision budget exceeds 10 % AND the
  CPU worker is not enough.
- It is designed together with X14 (rollout/update overlap), and both wait for a phase-utilization measurement
  (F-ED-6).

## 5. The eval budget, derived from PRECISION

**Inputs (with provenance):**
- **Pair SD** σ_pair = **0.333** in score units on a near-½ cross edge (P0: 45,000 cross pairs pooled; 0.328–0.336
  per edge). Unmirrored per-game SD is 0.5.
- **Elo scale:** d(score)/d(Elo) at ½ = **0.144 pp per Elo**.
- **Wall per 10M steps:** 20 updates × 51.6 s per 2M ≈ **5,160 s** (86 min) at N = 256 E10 (O9).
- **Window rate:** **118 games/s** (P0 pooled), with the O9-implied **62 games/s** as the pessimistic bound. U4
  measures the real one.

**Per consumer, per 10M steps:**

| consumer | precision it needs | games per 10M | at 118 g/s | at 62 g/s |
|---|---|---|---|---|
| **P0 cycle** | a monitoring curve: ±5 pp per opponent (no decision reads it once SPRT is on) | 100 × (9 bots + 5 sentinels) × 5 cycles = 7,000 | 1.15 % | 2.2 % |
| **P1 SPRT** | α = β = 0.05 at H0 0.50 / H1 0.55 (decided) | **≈ 6,000 mean** (item 20): ≤ 5 candidates per 10M × ~600 pairs, since a candidate 2M newer than a recent pool sits in the indifference zone (530–610 pairs, `measurements/sprt_promotion/`); worst 5 × 1,680 pairs = 16,800 | 0.99 % (worst 2.76 %) | 1.9 % (5.2 %) |
| **P2 plateau Tier 1** | the owner-registered GSPRT, H0 0.50 / H1 0.52, α = β = 0.05, mirrored, 40-pair batches, cap 6,000 pairs (§8) | **E[games] 3,200 at either hypothesis, 5,100–5,500 at p = 0.51** (simulated with `sprt.py`'s own code); cap 12,000 | 0.53–0.90 % (worst 1.97 %) | 1.0–1.7 % |
| **P2 cycle monitor row** | §2.5's detection target: 500 pairs per edge | k ≤ 7 by 80M, ≤ 10 to 650M: 7,000–10,000 | 1.15–1.64 % | 2.2–3.1 % |
| **P2 monitor top-ups** | confirmation of a flag (§2.5) | ≤ 3,000 when a flag fires | ≤ 0.49 % | ≤ 0.93 % |
| **P2 panel** | anchor set + in-band bots, 6 opponents × 500 pairs per new node; topped up to 1,000 pairs (newest AND W-back) only at a candidate plateau (§8) | 6,000 (+12,000 once, at a candidate plateau) | 0.99 % | 1.9 % |
| **P3 pool matrix** | n₀ = 50 pairs per edge (the ladder's 100 games, balanced) at K = 20 | 1,900 per promotion × 1–5 = 1,900–9,500 | 0.31–1.56 % | 0.6–3.0 % |
| **P4 audits** | the 2 % replay (§9.2) | ~1,000 | 0.16 % | 0.3 % |
| anchors (SmallRL etc.) | `EXTERNAL_ANCHORS_SOP.md` | ~300 per milestone | its own websocket stack, offline | — |
| exploiter gap | the run floor dominates once meter SE ≤ ~1 pp | 1,000 pairs × 5–7 per round | offline engine | — |
| pre-registered A/B (X5) | §7.4: 1,000 pairs per cell | 9 / 25 / 64 cells × 2,000 games | offline engine (P5) | — |

**The fit (DERIVED; U4 measures the rate):**
- **Mean:** 32,100 (one promotion, no flags, a Tier 1 decided quickly) to 48,000 games (five promotions, a flag, a
  slow Tier 1) per 10M.
  - At 118 games/s that is 272–407 s of the 5,160 s, **5.3–7.9 % of wall**.
  - **The two-tier plateau plus the monitor** (Tier 1 + row + top-ups + panel) are 16,200–24,500 games,
    **2.7–4.0 %**.
- **Worst** (five capped SPRTs, Tier 1 at its cap, a flag): 65,300 games = 10.7 %.
  - The fixed yield order (§0d rule 4) defers P3 and P4 to the next interval, giving 54,800 games = **9.0 %**, under
    the registered 10 %.
- **c_gpu** (P2–P4) needs 3.1–5.8 % on the mean. **The provisional 6 % fits.**
- **At the pessimistic 62 games/s** the mean is 10.0–14.9 %, over the registered 10 %. The trims, in order:
  1. drop the saturated bots from the P0 cycle (random ~0.99, `heuristic` ~0.8 carry almost no information: Dota 2's
     15–85 % rule);
  2. halve the panel's routine pairs;
  3. halve n₀. If it is still over, §4.5's trigger (a) fires.
- **The reference-based plateau read, if it were the PRIMARY,** would need 5,860 pairs per score. The plateau is a
  difference of two scores, so SE(Δ) ≤ 0.61 pp needs each score's SE ≤ 0.43 pp. Each snapshot is scored once and
  reused later as a W-back, so that is 11,700 games per check (2.3 % at 118 games/s). As the REPORTED secondary it
  is read from the monitor's rows at no extra games beyond top-ups (§2.3).
- **F-ED-5 (revised):** window-only costs 5.3–7.9 % of wall at P0's rate, and the in-window rate decides whether a
  second lane is ever needed.

## 6. Week-one assumptions, RE-GROUNDED (owner, 2026-10-02: "encourage the agent to re-evaluate key assumptions")

| parameter | today | verdict | evidence |
|---|---|---|---|
| games per opponent per eval cycle | 100 | **KEEP for the monitoring cycle; DECISIONS stop reading it**; drop saturated bots first if the window is short | Promotion, the plateau and the monitor have their own requests (§5). ±5 pp is adequate for a monitor and the supply guards |
| eval cadence | every 2M steps | **KEEP for the cycle; per question for the rest** | plateau and monitor once per 10M; matrix per promotion; audits continuous (§0d) |
| pool size | 20 (`DEFAULT_MAX_SNAPSHOTS`) | **KEEP for a single lineage; MEASURE before changing for era step 1** | no within-run cycling DETECTED (F-ED-1, limit ≈ 3 pp); N0's posterior Nash support ~1–8 nodes; the multi-lineage pool is unmeasured, and the cycle monitor measures it |
| eviction | oldest-first (or spread retention) | **CHANGE (T20):** a declared ledger read (§2.4) | zero Nash weight is necessary but near-vacuous in a transitive pool; PFSP is not a ledger read |
| promotion | first threshold crossing (0.55 / 0.65) | **CHANGE → SPRT** (BUILT, flips at X26) | the crossing step is a coin flip inside the replicate floor (ledger 2026-09-10, `L17330`) |
| bots in the opponent mix | a few % | **OUT OF SCOPE here** (a training-recipe knob) | eval's only input: the bot edges are saturated, so their RATING information is near zero |
| the eval team set | the trainee's builder (default pool, 10 % sample bias) | **KEEP; RECORD it as `regime.team_set`** | a new team set is a new regime, never pooled |
| the trace quota preferring losses | — | **KEEP for the prober; ENFORCE that estimates never read traces** | §0c rule 3, enforced by the reader gate |
| per-pair games on the snapshot ladder | 100 (200 vs promotion sentinels) | **KEEP the count; MIRROR it (50 pairs) under U7** | a dense node SE of ~8 Elo by formula (173.7 × √(1/(19 × 100 × ¼))); N0's committed `ladder.json` reports se ≈ 8.7 Elo per node, i.e. a 95 % half-width of 16.9 (F-ED-13 WITHDRAWN) |
| the plateau measure | strength vs the archive's Nash mixture (league §C, 2026-10-02) | **CHANGE (proposed, Q8):** two tiers, with the owner-registered head-to-head as the primary | the reference read needs 2× the pairs (a difference of two estimates) and inherits the mixture's instability (F-ED-3) |

## 7. TODO (fill in as implemented)
- [ ] U1 the ledger v2 core (§0b, §10).
- [ ] U2 in-loop migration (cycle rows, SPRT rows + decisions, ladder dual-write, h2h archive default).
- [ ] U3 backfill; U3b anchors / untaught / gap onto the ledger; U3c the `eval_results.jsonl` readers.
- [ ] U4 the scheduler + generalized window (+ its first measurement); U4a slot-direct snapshot loads; U4b SPRT resume.
- [ ] U5 the cycle monitor (+ its offline driver); U9 the two-tier plateau check.
- [ ] U6 the multi-cell offline engine.
- [ ] U7 T20 estimators; U7a retire the ladder updater; U7b eviction operating characteristics; U8 §2.7's validation
      + the N0 back-test (references and plateau).
- [ ] U10 the Rust p1 policy route (balanced seats).
- [ ] F-ED-2 (the ladder's sub-binomial noise): resolved BEFORE U7 (unit U0).
- [ ] The discrimination meter (era step 3).
- [x] Literature review (this doc §0a, 2026-10-03; six citations corrected by the review).
- [x] §5 budget derivation (2026-10-03, revised for window-only; the in-window rate is measured by U4).
- [x] §6 re-grounding (2026-10-03).
- [x] The blocking question (§4, settled 2026-10-02; window-only proposed 2026-10-03).

## 8. The PLATEAU test: two tiers (owner, 2026-10-03), and what it reads

**The owner's fear:** calling a plateau when we are really cycling, and wasting weeks. So the test has two tiers, and a
plateau is declared only when BOTH say so.

### 8.1 Tier 1: progress against its older self (frequent, cheap, the PRIMARY)
- **The owner-registered measure** (`era_plan_post_m5.md`): once per 10M steps, the newest thinned node plays the node
  from **W ≈ 7 GPU-hours back** (≈ 50M steps at N = 256, i.e. lag 5 of the monitor) on mirrored pairs, as a pentanomial
  GSPRT.
  - **H0 p ≤ 0.50 vs H1 p ≥ 0.52.** 2 Elo per GPU-hour × 7 h ≈ 14 Elo ≈ 2 pp.
  - α = β = 0.05, Wald bounds, 40-pair batches.
  - **Cap 6,000 pairs.** It is its own request (§0c rule 1).
- **Verdicts:**
  - **GAIN** (H1 accepted): keep training.
  - **FLAT** (H0 accepted).
  - **UNDECIDED** (the cap is reached). Unlike promotion, the cap does NOT default to H0, so an undecided check never
    calls a plateau.
- **Operating characteristics** (simulated with `sprt.py`, P0's pooled cross pentanomial tilted to each mean, 2,000
  replicates; `revision/plateau_sprt.log`):

  | true p | P(FLAT) | P(GAIN) | P(UNDECIDED) | E[pairs] | 90th pct pairs |
  |---|---|---|---|---|---|
  | 0.49 | 0.999 | 0.001 | 0.000 | 890 | 1,520 |
  | 0.50 | 0.954 | 0.037 | 0.009 | 1,620 | 3,120 |
  | 0.505 | 0.79 | 0.17 | 0.04 | 2,180 | 4,440 |
  | 0.51 | 0.46 | 0.44 | 0.10 | 2,550 | 5,880 |
  | 0.52 | 0.036 | 0.957 | 0.007 | 1,600 | 3,040 |

  Mean cost: 3,200–5,500 games per check. A fixed-n test of the same α and β would need ≈ 2,950 pairs (5,900 games)
  EVERY time. **Why the stopping-time-biased Tier 1 rows are not reused by the monitor:** a sequentially stopped
  edge estimate is biased toward its boundary. So the monitor plays its own 500 pairs against lag 5, although sharing
  would save 1,000 games.

### 8.2 Tier 2: transitivity (a READ of the cycle monitor, plus the panel)
- **The cycle monitor's verdict at this check** (§2.5): NONE, SUSPECTED or CONFIRMED.
- **The panel:**
  - the fixed anchor set plus the bots inside the 15–85 % band, 6 opponents. SmallRL is not in the window; it stays on
    `main.anchors`' milestone cadence and is REPORTED;
  - the newest node's panel score is played at 500 pairs per opponent and banked. It is reused when that node is later
    the W-back;
  - **Δ_panel** = the newest node's panel score − the W-back node's, converted to Elo per opponent by the logit (G4);
  - **PANEL RISING** iff the lower one-sided 95 % bound of Δ_panel > 0;
  - **PANEL FLAT** iff the upper one-sided 95 % bound < 14 Elo (the plateau rate over W);
  - else UNDECIDED.
  - **At a CANDIDATE plateau** (Tier 1 FLAT, monitor NONE), both nodes' panel scores are topped up to 1,000 pairs per
    opponent (SE(Δ) ≈ 0.61 pp). That makes P(PANEL FLAT | zero true gain) ≈ 0.95, at 12,000 games once.
  - Repeated checks of a running score use a betting confidence sequence.

### 8.3 The signatures: deterministic, pre-registered

| kind | rule | lever family (league §C) |
|---|---|---|
| **TRANSITIVE IMPROVEMENT** | Tier 1 GAIN AND monitor NONE | keep training |
| **TREADMILL / CYCLING** | monitor CONFIRMED, whatever Tier 1 says. Its typical signature: beats the recent node but ties or loses to an older one, a non-monotone lag curve, the cyclic share and Nash support up, the panel flat | population: train vs the archive's Nash mixture, grow the pool, exploiters |
| **TRUE PLATEAU** | Tier 1 FLAT AND monitor NONE AND PANEL FLAT, **at two consecutive checks** | capacity / algorithm |
| **STAGNATION** | a TRUE PLATEAU with `main.policy_drift` below its floor | exploration: forks |
| **CONTINUE** | anything else (UNDECIDED, SUSPECTED, panel undecided): re-check at the next interval | — |

- **Why two consecutive checks** (G1, repeated monitoring). One check calls a false plateau on a run still gaining
  2 pp per W with probability ≤ 0.036 × P(monitor NONE) × P(panel FLAT). Requiring the pattern twice squares that.
  It costs one interval (~1.4 GPU-h) of latency. That is the owner's chosen direction: a delayed plateau call wastes
  ~1.4 GPU-h, a false one wastes weeks.
- **Simulated joint operating characteristics** of the whole rule (G6) are part of U9, before it is trusted.
- **The REPORTED secondary:** strength vs a frozen reference (§2.3), the series for G2's current-slope fit
  (Domhan et al. 2015), and the plateau-KIND diagnostic. It never gates.

### 8.4 What the plateau test gets from the system (league §C's G1–G6)

| gap | answer |
|---|---|
| **G1** repeated monitoring | each check is a fresh GSPRT on its own request; a plateau needs two consecutive checks (§8.3). A confidence sequence or CUSUM can replace the 2-check rule later without a ledger change |
| **G2** a current slope | the reference series (§2.3) fitted per Domhan et al.; REPORTED |
| **G3** the archive's mixture moves | moot for the primary (a head-to-head has no mixture). For the secondary: frozen references, bridged across re-solves (§2.3) |
| **G4** the Elo conversion | Tier 1 is model-free (pp at ½, ≈ 0.144 pp per Elo, within ~4 % on [0.4, 0.6]). The panel converts per opponent by the logit |
| **G5** the noise floor AT the plateau | the paired control continuations (era plan); σ_h by `main.h2h.runfloor` |
| **G6** combined error rates | U9 simulates the full rule: σ_pair 0.333; Tier 1's OC above; the monitor's false-alarm rate 6–10 % per check; the panel's SE; the 2-check rule |
| **N0 back-test** | N0's snapshots, read `as_of` each 10M point: does the rule call the plateau where the ladder flattens, and how often does it false-alarm on climbing segments? It needs the monitor's thinned rows for N0 (≈ 7 rows × ≤ 7k games ≈ 30k games) and Tier 1's checks (≈ 6 × 4k), played by the offline engine. **RISK:** N0 is architecture v121, and loading its snapshots on HEAD's core needs `historical_load_kwargs`; N0's snapshots specifically are **UNVERIFIED** |

## 9. Determinism, audits and the seat RNG

### 9.1 Every decision rule is deterministic
- A decision is a pure function of (its request's or family's rows, declared constants, the rule's version). It writes
  a decision row naming all three.
- **A Monte Carlo inside a rule** (a Nash posterior, a bootstrap null) uses a declared draw count, with the seed = the
  digest of the sorted consumed `row_id`s and the rule version.
- **Rounding bands** (standing rule 8):
  - an LLR or t within 1e-9 of a bound is not a crossing;
  - a posterior probability within ±0.01 of its threshold is UNDECIDED, which means kept or continue, never act;
  - a bootstrap-threshold statistic within one Monte Carlo SE of its threshold is INCONCLUSIVE (§2.5, §2.7);
  - the CPU worker's throughput bound within 0.0005 of the bar is NOT ADOPTED.
- **INCONCLUSIVE, never a verdict, when:**
  - aborted games exceed 25 % of a CELL's attempted games (per (request, matchup), as X5 §7.4 registers it;
    review M3(b));
  - a request's rows span two regimes;
  - a pinned family's rows span two protocols.

### 9.2 Audits
- **Replay audit (P4):** a random 2 % of banked batches, chosen by a seeded hash of `row_id`, are re-played from their
  seed block. **The outcome digest over the non-near-tie games must be EQUAL,** and so must the near-tie index list's
  length to within its census rate. Any other difference is a FINDING and stops the scheduler's decisions until it is
  explained. F-P0-7 shows exact replay holds across CPU and GPU. Backfilled rows (`digest_unrecorded`) fall back to
  count equality.
- **`audit` CLI** at scheduler startup (§0b.4).
- **Decision audit:** `python -m main.eval_ledger verify <decision_id>` re-derives a verdict from its rows.

### 9.3 Seats (review M1) and the seat-dependent RNG (F-P0-5)
- **What the core does:** a POLICY player is always at p1. `rust_eval/executor.py` submits the player to
  `tb.trainee_slot`; there is no p2 policy route. So `seat_rule = balanced` is NOT free. The design keeps
  **`seat_rule = fixed_p1`** for every new policy row.
- **What was measured** (P0):
  - the seat effect is **u = −0.006 ± 0.083 pp** (95 % within ±0.17 pp);
  - 5.2 % of self-play pairs are WW or LL, and 779 of 14,963 pairs clear of any near tie are off centre;
  - the mechanism (a speed tie broken by the RNG in a seat-dependent order) was diagnosed on perturbed fresh
    checkpoints only.
- **What it means:**
  1. **A mirrored pair is duplicate play, not an exact antithetic copy.** The pentanomial GSPRT estimates the pair
     distribution from the data, so its validity is UNAFFECTED.
  2. **The variance gain is modest:** ~12 % of games on a near-½ cross edge (F-ED-4), ~90 % on a self-play edge.
     Mirroring is kept for BALANCE.
  3. **The seat effect is zero within ±0.17 pp today,** on one engine version. A fixed seat is a bias of at most that
     size, applied identically to both sides of every comparison the plateau and the monitor make. It cancels in
     differences between snapshots of one protocol, but not in an absolute score.
  4. **The IDENTICAL-PLAYERS audit keeps watching it.** It is a self-play edge per architecture per protocol, in the
     audit class at each protocol bump. Under `fixed_p1` its expected score is ½ + u; a 95 % interval excluding
     ½ ± 0.25 pp is a FINDING that re-opens U10's priority.
  5. **Balanced seats become their OWN unit (U10):** a p1/p2 policy route on the core with parity gates. It is
     sequenced after the ledger and does not block it. When it lands, new rows switch to `balanced` under a new
     protocol version.

## 10. Build plan

**Order:**
- U1 → U2 FIRST, landing before X26 continues from the X5 A/B winner (~2026-10-07/08). SPRT resume and seat balancing
  are moved OUT of U2, so the critical path is ~4 agent-days.
- Then the scheduler and the window (U4), the cycle monitor and the plateau (U5, U9), then T20 (U7, U8).

Sizes are agent-days (the review expected 18–20 for the earlier three-lane plan; window-only drops the CPU worker but
adds the monitor and the plateau unit).

| # | unit | what | size | tier | depends on | Sonnet-safe? |
|---|---|---|---|---|---|---|
| 1 | **U1** | ledger v2 core: schema v2 (+ `request.batch` / `family`, outcome digest, `monitor` purpose) + validator, v1 upgrade-on-read, archive location, decisions / requests / references streams, the claim lock + void rule, duplicate refusal in every reader, `ReaderDecl` (incl. `requests="family"`) + static and closed-list gates, pair-level estimator helpers, `audit / verify / show`; `main.h2h` archive default + `--purpose ab` (storage-only, digest-proved) | 2.5 | opus-high | — | no (core schema; GIGO risk) |
| 2 | **U2** | in-loop migration: cycle rows dual-written beside `eval_results.jsonl` (exact W/L/D, team counters, digest); SPRT rows per batch + decision rows (resume NOT here); ladder dual-write. Gate: the routine gate + the `--debug --debug-eval` smoke + **the first two minutes of a real launch, which needs the GPU owner's cooperation** (the training agent runs it under its lease) | 1.5 | opus-high | U1 | no (the training loop) |
| 3 | **U0** | F-ED-2: resolve the ladder's sub-binomial noise: a seeded replicate experiment (one edge × 20 replays × 100 games, CPU), then the cause | 0.5 | opus-medium | — | no (GIGO hunt) |
| 4 | **U3** | backfill per §0b.6 | 1.0 | sonnet-xhigh | U1 | **yes**, with §0b.6 as the mapping |
| 5 | **U6** | the multi-cell offline engine (P5's lane; X5 look cells; dense audits; back-fills) | 1.0 | opus-medium | — | no (GPU engine) |
| 6 | **U4** | the scheduler + window: request store, claims, allocation and yields (§0d), budgets and c_gpu, the window spread over ≤ 5 windows, **the frozen-vs-frozen executor change** (a frozen player in the trainee eval slot, ≤ 5 opponents per pass), the replay audit. **First deliverable: the in-window measurement** (SPRT batch wall incl. `load_sentinels`, games/s, per-window overhead) that fixes c_gpu | 3.0 | opus-high | U2 | no |
| 7 | **U4a** | slot-direct snapshot loads (F-ED-11): state dicts go straight into the declared slots via `slots.copy_in`, with no per-cycle `nn.Module` build in `load_sentinels`; `rust_eval/launch.py` added to the learner-lifecycle gate's scope if it is not already | 0.5 | opus-high | U4 | no |
| 8 | **U4b** | SPRT resume across restarts (Q7) + snapshot files pinned while a request is open | 0.5 | opus-high | U4 | no |
| 9 | **U5** | the CYCLE MONITOR: thinned-node requests, rows + panel, the four signals with parametric-bootstrap nulls, confirmation top-ups, `cycle_flag` decisions, the dashboard (support size, entropy, lag curve), `hodge.py` pool × pool, and the offline driver `main.cycle_monitor` (CPU or offline engine, for pinned runs and N0) | 2.0 | opus-high | U4 (driver: U1 + U6) | no |
| 10 | **U9** | the two-tier PLATEAU check: Tier 1 GSPRT requests, the panel rule, the signature table, the 2-check rule, the G6 joint OC simulation, the decision rows | 1.0 | opus-high | U5 | no (a pre-registration) |
| 11 | **U3b** | `main.anchors`, `main.untaught_meter`, `best_response_gap` onto the ledger | 1.0 | sonnet-high | U1, U2 | **yes** |
| 12 | **U3c** | migrate the `eval_results.jsonl` readers (TensorBoard / TUI, `main.elo`, `best_response_gap`, the supply guards) to the ledger; retire the file's write | 1.0 | sonnet-xhigh | U2, U3 | **yes**, with the reader list |
| 13 | **U7** | T20 estimators: BT / Hodge pool × pool, the reference candidates (§2.3), T3 rows in the window, T2 on betting CSs, the eviction ledger read (§2.4), the ladder fit reading the ledger | 2.5 | opus-high | U4, U0 | no |
| 14 | **U7a** | retire the detached ladder updater (`_spawn_snapshot_ladder_update`) once U7's T3 rows and ledger fit are live | 0.5 | sonnet-high | U7 | **yes** |
| 15 | **U7b** | operating-characteristics simulation of eviction and the T20 pipeline (transitive / treadmill / multi-lineage pools) | 0.5 | opus-high | U7 | no |
| 16 | **U8** | §2.7's validation on the first three era-step-1 pools + the N0 back-tests (reference candidates, §2.3; the plateau rule, §8.4) | 1.0 + games | opus-high | U7b, U9 | no (pre-registered reads) |
| 17 | **U10** | the Rust p1 policy route (balanced seats) with parity gates; a protocol bump when it lands | 2.0 | opus-high | U1 (not blocking) | no (Rust core crossing) |
| 18 | U11 *(optional)* | the CPU worker + §4.5's throughput test, ONLY if §4.5's trigger fires | 2.0 + ~9 h of a production run | opus-high | U4 | no |

- **Total ≈ 22.5 agent-days without U11** (+2 if it is triggered).
- **The critical path to X26** is U1 → U2, ≈ 4 agent-days. U0, U3 and U6 run beside it.
- **T19** (U4–U4b) and the cycle monitor + plateau (U5, U9) must be ready before X26's first plateau-relevant check.
  - X26 is PINNED, so its own checks run through U5's offline driver (F-ED-19).
  - T20 (U7–U8) must be ready before era step 1.
- **Each unit updates this doc,** its leaf `CLAUDE.md`, `eval_and_rating.md` and the Decision record in the same
  commit, and ships through `/gen3ai-ship` with green gates.

## 11. Open gaps, owner questions, findings

### 11.1 Open gaps (honest list)
1. **The in-window game rate is UNMEASURED** (F-ED-20). §5's fit uses P0's standalone 118 games/s; O9 implies as low as
   62. U4 measures it first.
2. **GPU and CPU utilization per training phase is UNMEASURED** (F-ED-6). The filler-vs-X14 choice waits on it.
3. **The multi-lineage pool's cyclic width is unknown.** The cycle monitor and T20 measure it in era step 1.
4. **The ladder's sub-binomial per-edge noise has no explanation** (F-ED-2). It is resolved BEFORE U7 (U0), because the
   monitor's and T20's parametric nulls assume binomial-or-pentanomial noise.
5. **Which strength reference is best** is undecided; the N0 back-test decides (§2.3).
6. **The cycle monitor's detection limit** within one check is ≈ 1.7 pp diffuse or a ≈ 6.5 pp hole (§2.5). Smaller
   cycles are seen only on the accumulated matrix.
7. **The joint operating characteristics of the two-tier plateau rule** are simulated in U9, not yet.
8. **AIVAT-style control variates** are deferred to era step 3.
9. **N0's snapshots on HEAD's eval core** are UNVERIFIED.
10. **Literature items still not verified from a primary source:**
    - AlphaStar's 35 / 50 / 15 split and f_hard;
    - the Kahle β₁ window;
    - Agarwal et al.'s method details;
    - a primary citation for duplicate poker;
    - Balduzzi's P3(ii) and Czarnecki's theorem conditions, cited as the review read them;
    - Howard et al.'s eq. (10), likewise.

### 11.2 Questions for the owner (the orchestrator's provisional call in bold)
- **Q1, purposes.** Add `ab`, `ladder`, `untaught`, `gap` and `monitor`? **Yes, with `ladder` for backfilled rows
  only.**
- **Q2, background eval.** **GPU window only, inside the trainer that owns the GPU (recommended).** The CPU worker is an
  OPTIONAL later unit (§4.5). It would buy ~25k games per 10M without stretching wall, and it is worth building only if
  U4's measured rate pushes the budget over 10 %, or era step 1's matrix outgrows c_gpu. The existing ladder updater
  stays until U7a. One early CPU use: the monitor's offline driver for the pinned X26 (~1 h CPU per 10M check;
  F-ED-19).
- **Q3, c_gpu.** **It applies to P2–P4 only; provisionally 6 %, fixed by U4's first measurement.** P0 + P1 are bounded
  by the SPRT cap and the registered 10 %.
- **Q4, T20's scope.** **Incremental dense for K ≤ 20 (+ T2 top-ups); sparse tiers only at archive scale.**
- **Q5, the strength reference.** **Freeze a reference per read, and let the pre-registered N0 back-test (§2.3) choose
  among posterior-mean Nash, uniform over the last K, and a QRE.** No commitment before it.
- **Q6, eviction.** **A declared ledger read at matrix DONE:** among confidently zero-weight members, evict the one the
  newest member most surely beats; ties go to oldest, then id. Grow if none qualify. Archive deletion is separate
  (§2.4).
- **Q7, SPRT across a restart.** **RESUME, built after U4 (U4b).**
- **Q8 (NEW), the plateau measure.**
  - (A) The owner-registered head-to-head GSPRT, newest vs the W-back node, plus the panel, as the PRIMARY, with the
    cycle monitor as Tier 2. **Recommended:** it is direct, needs one estimate rather than a difference of two, and
    avoids the mixture's instability.
  - (B) The slope of strength vs a frozen Nash reference as the primary (league §C's 2026-10-02 wording). Under (A), B
    is REPORTED.
  - The proposal also asks two things of the owner: requiring PANEL FLAT (the registered "both accept H0"), and two
    consecutive checks (one interval of latency).
- **Q9 (NEW), the monitor's detection target.** **≈ 1.7 pp diffuse / ≈ 6.5 pp hole within one check at 500 pairs per
  edge (≈ 1.2–1.6 % of wall), with smaller cycles caught on the accumulated matrix.** The alternative is 1 pp within
  one check, at ~4× the row's games. **Thinning:** hybrid (dense over the last 80M, geometric beyond) is recommended
  over uniform (cost grows linearly after 90M) and over geometric alone (blind to holes at unplayed lags).

### 11.3 Findings (standing rule 7; evidence in `measurements/eval_design_2026-10-03/` unless noted)
- **F-ED-1 (MEASURED): no cycling DETECTED in within-run snapshot pools; NOT "transitive".** On 70 ladders, the BT
  residual Pearson χ²/df has median 0.76 (q10–q90 0.54–1.14); binomial data on the same graphs give 1.01–1.03.
  - **Detection limit** (N0's shape: 20 nodes, 141 edges × 100 unmirrored games): an excess per-edge SD of ≈ 3.0 pp at
    80 % power and ≈ 2.4 pp at 50 % (`revision/fed1_limit.log`). Cycling below that is not excluded.
  - The exploiter-era (multi-lineage) pools are UNTESTED.
  - The noise floor for every future read is a PARAMETRIC BOOTSTRAP per protocol, not these legacy rows (review
    item 17).
- **F-ED-2 (MEASURED, cause UNKNOWN): the ladder's per-edge noise is below binomial.** χ²/df < 1 on most ladders; 133
  twice-played pairs give var(z) = 0.82.
  - The ladder draws teams from the unseeded global `random` stream, so games are marginally iid Bernoulli. Per-game
    heterogeneity of win probability therefore cannot explain it alone.
  - Candidates to test (U0): outcome-dependent loss of unfinished games (a finished-only denominator), and dependence
    between concurrent games.
  - Resolved before U7.
- **F-ED-3 (MEASURED; CORRECTED by the review): the max-entropy Nash on a near-transitive archive is pure on the
  frontier and unstable under noise.** The newest node's score vs the posterior mixture has SD 3.6 pp at 100 games per
  edge: **2.5 pp from the mixture, 1.9 pp from the newest node's own edges** (`revision/nash_decomp.py`, the
  reviewer's decomposition).
  - **The spec actually used,** not the one the first draft stated: Beta(w + 1, l + 1) per played edge (uniform prior),
    200 posterior draws, the 49 unplayed pairs imputed from the BT fit with N(0, 0.15) logit noise, the archive = the
    19 nodes other than the newest, seed 1.
- **F-ED-4 (MEASURED): mirroring saves ~12 % of games on a near-½ cross edge** (variance ratio 0.87–0.91; self-play
  0.09–0.11).
- **F-ED-5 (DERIVED, revised): window-only costs 5.3–7.9 % of wall at P0's 118 games/s (9.0 % worst after yields), and
  10–15 % at O9's implied 62 games/s** (§5).
- **F-ED-6: "the CPU is ~15/16 idle during the update" has no measurement behind it.**
- **F-ED-7: `sizing_C`'s `ladder.json` relative reference silently fell back to the first snapshot** (ImportError on
  the registry baseline).
- **F-ED-8: in-loop eval rows carry no draw count;** draws survive only in 370 of 1,081 manifests.
- **F-ED-9: `sprt_promotion.abandon_unfinished` discards an in-flight test at every restart** (fixed by U4b).
- **F-ED-10: a co-resident GPU eval process would lower K6's device-free memory ceiling.**
- **F-ED-11 (RESOLVED by design): `load_sentinels` builds a CPU `nn.Module` per cycle** (`load_opponent_snapshot`)
  inside the eval path. How the learner-lifecycle static gate classifies it was never verified. U4a removes the build
  (state dicts straight into the declared slots via `copy_in`), so the classification no longer matters, and it puts
  the module under the gate.
- **F-ED-12: no pentanomial data exists in any RUN's eval history.**
- **F-ED-13 (WITHDRAWN by the review):** the first draft read N0's `ladder.json` "se 16.9" as an SE. It is the 95 %
  half-width; the se is ≈ 8.7 Elo, consistent with root `CLAUDE.md`'s "±10".
- **F-ED-14 (MEASURED, the reviewer's rerun + this revision): the block-median throughput rule almost never adopts.**
  - On `sizing_B`, 5-update block medians have a paired log-ratio SD of 0.082 (level shifts), so P(adopt a free lane)
    ≈ 0.09 (`revision/blk.py`).
  - Per-update paired differences fix it: robust SD 0.006 on `train_ms`, 0.068 / 0.022 on collection
    (`sizing_B` / `sizing_C`).
  - Every ~10th update carries a periodic side task (~48 s of `train_ms` vs ~41 s), which must be excluded by flag.
- **F-ED-15 (MEASURED): the game-length covariate makes collection noise WORSE.** Residualising `collect_ms` per
  decision on `ep_len_mean` raised its adjacent-update SD from 0.189 to 0.225 (`sizing_B`) and from 0.054 to 0.128
  (`sizing_C`). So §4.5 reports it as a sensitivity. This is a reported DEVIATION from the orchestrator's M6 wording.
- **F-ED-16 (SIMULATED): the owner-registered plateau GSPRT is affordable.** E[games] 3,200–5,500 per check, P(FLAT |
  p = 0.50) 0.954, P(FLAT | p = 0.52) 0.036, cap truncation ≤ 10 % (§8.1).
- **F-ED-17 (DERIVED): a plateau read as a DIFFERENCE of two reference scores needs 5,860 pairs per score** for SE(Δ) ≤
  0.61 pp. The first draft's 2,930 ignored the W-back score's own variance (review M2).
- **F-ED-18: the detached snapshot-ladder updater is a CPU eval lane running beside training TODAY** (1,900 games per
  promotion on the bridge, CUDA-free, unmeasured perturbation). It also runs in both X5 A/B arms. It is symmetric by
  design, but it adds CPU load the speed rule's cycle-wall medians were not registered with. **For the orchestrator:**
  decide whether X5's arms run it (`--ladder-games 0` would remove it), and record the choice in X5's registration.
- **F-ED-19: X26 is pinned and launches before U4–U5 can land,** so its cycle monitor and Tier 1 cannot run in its
  window. They need the offline driver beside it on CPU (≈ 17k games per 10M check ≈ 1 h at 4.8 games/s, unmeasured
  perturbation, the same class as F-ED-18), or a later offline replay of its frozen snapshots. A late replay gives a
  late plateau call.
- **F-ED-20: the in-window game rate is unmeasured** (P0 standalone 118 games/s vs O9's implied 50–83). The window-only
  budget fits 10 % only near the former. U4 measures it first.
- **F-ED-21 (SIMULATED): the cycle monitor's within-one-check limit** at 500 pairs per edge (hybrid, 80M check) is a
  diffuse excess cyclic SD of ≈ 1.7 pp or a single hole of ≈ 6.5 pp. Geometric thinning alone is blind to a hole at
  an unplayed lag (§2.5).

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-02 | Mirrored pairs **(owner)** | symmetric evals mirror; pinned-team meters refuse | mirroring everything | piloting vs response (`04aa4a45`) |
| 2026-10-02 | Promotion **(owner)** | GSPRT on mirrored pairs (Fishtest's method + a minimum pair count), Wald bounds | overshoot-corrected bounds (saved ~21 % of pairs, false promotion 5.07 %) | `measurements/sprt_promotion/` |
| 2026-10-02 | Performance checks **(owner)** | deterministic performance-SHAPE tests + an on-demand benchmark | a wall-clock perf guard | `48265bf8` |
| 2026-10-02 | Pool defence against cycling **(owner)** | tiered matrix + Nash averaging + Hodge meter, validated against dense; grow the pool rather than evict harder when cycles are wide | hope; recency-only eviction | TASK_BACKLOG T20 |
| 2026-10-02 | The eval LEDGER **(owner)** | append-only JSONL of COUNTS per (batch × matchup), with pentanomial pair counts and per-team counters, archive-level, one writer per file | a per-game ledger | §0b: counts are sufficient statistics for every planned estimator |
| 2026-10-03 | The ledger's FIRST WRITER (X5 P0, orchestrator) | `main.h2h` writes one row per (batch × matchup) through `eval_ledger` (`gen3_eval_count_row_v1`) to a caller-named directory; it refuses `models/` until the archive ledger exists | waiting for the archive ledger before measuring anything | the P0 pre-study needed a durable, auditable row format now |
| 2026-10-02 | Doc split (orchestrator) | the SYSTEM here; the three population DECISIONS in `design_league_decisions.md` | one combined doc | separate use cases from implementation (owner's suggestion) |
| 2026-10-03 | The eval-system DESIGN (eval design agent; first draft) | ledger v2, one scheduler, T19 as three lanes, budget from precision, T20 incremental dense + T2, frozen posterior-mean Nash reference, SPRT resume | a second GPU eval process; GPU filler now; four tiers for an active pool; point-Nash references; a fixed a₀ | §0a; `measurements/eval_design_2026-10-03/`; P0; O9 |
| 2026-10-03 | Independent review | **SOUND WITH FIXES**: six must-fix items (M1–M6), fourteen fix-during-build items (7–20) | — | the reviewer's reruns (`revision/nash_decomp.py`, `revision/blk.py`) |
| 2026-10-03 | **M1 seats (ORCHESTRATOR)** | keep `seat_rule = fixed_p1` with the measured u = −0.006 ± 0.083 pp and an identical-players audit; balanced seats become their own Rust unit (U10) after the ledger; window capacity is one player × ≤ 5 opponents per pass; the frozen-vs-frozen executor change is sized in U4 | "balanced costs nothing" (false: the core seats a policy at p1 only) | `rust_eval/executor.py`; §9.3 |
| 2026-10-03 | **M2 plateau measure (ORCHESTRATOR; owner Q8)** | the owner-registered head-to-head GSPRT as the PRIMARY (direct, one estimate, no mixture instability); strength vs a frozen reference REPORTED as the secondary and kind diagnostic; both re-sized (Tier 1 E 3.2–5.5k games; the reference read would need 5,860 pairs per score) | silently replacing the registered measure with the reference slope (the first draft) | §8.1; F-ED-16, F-ED-17 |
| 2026-10-03 | **Two-tier plateau (OWNER input, via the orchestrator)** | Tier 1 = head-to-head vs W-back; Tier 2 = transitivity read from the cycle monitor + the panel; plateau only when Tier 1 is flat AND no cycling; a treadmill is reported as its own kind | one tier | §8; the owner's fear of calling a plateau while cycling |
| 2026-10-03 | **The CYCLE MONITOR (OWNER input, via the orchestrator)** | a standing instrument at every check from step 0: thinned-archive matrix grown one row per check, hybrid thinning, 500 pairs per edge, four signals with parametric-bootstrap nulls, fresh-top-up confirmation | cycling read only inside the plateau test; uniform thinning (linear cost); geometric alone (blind to unplayed lags) | §2.5; `revision/monitor_power.log` |
| 2026-10-03 | **Background eval: GPU window only (OWNER input; orchestrator recommendation, owner Q2)** | the blocking window inside the trainer carries P0–P4; the offline engine carries P5; the CPU worker is an OPTIONAL unit with a stated trigger; the ladder updater stays until U7a | three lanes now (the first draft) | §4, §5: 5.3–7.9 % of wall at 118 games/s |
| 2026-10-03 | **M3 X5 coexistence (ORCHESTRATOR)** | (a) `request.family` + `requests="family"` for group-sequential decisions; (b) INCONCLUSIVE per CELL; (c) a pinned family freezes its protocol, and U1's h2h change is storage-only and digest-proved; (d) no CPU eval lane during an X5 arm, look cells between arms by the GPU owner or on CPU with no arm running | request-only scoping; per-request INCONCLUSIVE | X5 §7.4; §0b.2, §0c rule 6, §4.2 |
| 2026-10-03 | **M4 duplicate batches (ORCHESTRATOR)** | `request.batch`; (request, batch, matchup) unique in `audit` AND every reader; claims and appends under one file lock; a dead or expired writer's claim voided deterministically and replayed on the same seeds | trusting one writer per file alone | §0b.4 |
| 2026-10-03 | **M5 eviction (ORCHESTRATOR; owner Q6)** | a declared ledger read at matrix DONE, `as_of` + digest recorded: among confidently zero-weight members, evict the one with the highest posterior P(newest member beats it); ties oldest, then id; cycle rows are NOT an input; active-pool eviction ≠ archive deletion, and nothing a reference / request / monitor names is deleted | the PFSP-weight tie-break (degenerates to "oldest" at the default `--pfsp-scale 0`; a stateful EMA, not a ledger read) | §2.4 |
| 2026-10-03 | **M6 throughput test (ORCHESTRATOR)** | per-phase metrics, per-update alternation, Hodges–Lehmann bound, size 300 pairs from a power calculation on banked noise; pre-registered, but only needed if the optional CPU worker is built | the block-median rule (P(adopt a free lane) ≈ 0.09) | §4.5; F-ED-14, F-ED-15 |
| 2026-10-03 | **Fix-during-build items 7–20 (ORCHESTRATOR)** | F-ED-3 corrected and F-ED-13 withdrawn; pair-level estimators + betting CSs (deviation from ResponseGraphUCB disclosed); the reference chosen by an N0 back-test; eviction at matrix DONE; scheduler yields, one SPRT in flight, the anti-starvation share per lane, P5's own lane; c_gpu for P2–P4 only, fixed by U4's first measurement; the ladder updater absorbed then retired (U7a); slot-direct loads (U4a); outcome digests; parametric-bootstrap floors; six literature corrections; resume wording and snapshot pinning; SPRT ≈ 6,000 games per 10M; §2.7's thresholds derived and its FAIL cost stated; the build plan re-sized with the missing units | — | this revision |
